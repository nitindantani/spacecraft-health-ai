import asyncio
import os
import io
import base64
import copy
import tempfile
from datetime import datetime, timezone
from collections import deque

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt

from fastapi import FastAPI, WebSocket, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak, KeepTogether
)

# ================================================================
# CONFIGURATION
# ================================================================
DATA_FOLDER        = r"G:\Pro_NML\data\ch3_ilsa\ils\data\calibrated"
TELEMETRY_INTERVAL = 1.0
HISTORY_SIZE       = 400    # needs 128+ valid packets for real model inference
WARN_THRESHOLD     = 0.05
CRIT_THRESHOLD     = 0.20
SAMPLE_RATE        = 1.0    # Hz — 1 packet per second

# STA/LTA configuration (standard seismology algorithm)
STA_LEN  = 5    # Short-Term Average window (seconds)
LTA_LEN  = 30   # Long-Term Average window (seconds)
STALTA_TRIGGER  = 3.0   # trigger ON  threshold
STALTA_DETRIG   = 1.5   # trigger OFF threshold

# Lunar quake classification thresholds
MOONQUAKE_TYPES = {
    "deep_moonquake":    {"freq_band": (0.5, 1.0),  "min_duration": 60},
    "shallow_moonquake": {"freq_band": (1.0, 8.0),  "min_duration": 10},
    "thermal_cracking":  {"freq_band": (8.0, 15.0), "min_duration": 2},
    "meteorite_impact":  {"freq_band": (1.0, 20.0), "min_duration": 5},
}

# ================================================================
# TELEMETRY DATA LOADER
# ================================================================
csv_files = []
for root, dirs, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print(f"[DATA] Total CSV files found: {len(csv_files)}")

file_index = 0
data_index = 0
current_df = None


def load_next_file():
    global file_index, current_df, data_index
    if file_index >= len(csv_files):
        file_index = 0
    path = csv_files[file_index]
    print(f"[DATA] Loading: {path}")
    current_df = pd.read_csv(path)
    data_index = 0
    file_index += 1


def next_packet() -> dict:
    global data_index, current_df
    if current_df is None or data_index >= len(current_df):
        load_next_file()
    row = current_df.iloc[data_index]
    data_index += 1
    return row.to_dict()


load_next_file()

# ================================================================
# ROLLING BUFFERS
# ================================================================
history:        deque = deque(maxlen=HISTORY_SIZE)   # raw sensor dicts
score_history:  deque = deque(maxlen=1000)           # for reports + history API
event_log:      list  = []                           # all detected seismic events
session_snapshots: deque = deque(maxlen=500)         # for /history replay

# ── Image store: analyzed images saved here for PDF injection ──
analyzed_images: list = []   # list of {filename, image_b64, image_type, analysis, timestamp, context}
last_image_context: dict = {}  # most recently analyzed image — used for follow-up questions


def safe(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


# ================================================================
# YOUR REAL PYTORCH MODELS — via InferenceManager
# ================================================================
# Loads your trained weights:
#   outputs/weights/best_lstm.pt
#   outputs/weights/best_cnn.pt
#   outputs/weights/best_transformer.pt
#   outputs/weights/best_gnn.pt
# Uses your real LSTMAutoencoder, CNNAutoencoder,
# TransformerAutoencoder, SpacecraftGNN architectures.
# Falls back to statistical methods if weights not found.
# ================================================================

import sys as _sys
_sys.path.insert(0, r"G:\Pro_NML")

# ── Try loading your real InferenceManager ─────────────────────
_inference_manager = None
_use_real_models   = False

try:
    from models.inference_manager import InferenceManager as _IM
    _inference_manager = _IM(
        weights_dir=r"G:\Pro_NML\outputs\weights",
        scaler_path=r"G:\Pro_NML\outputs\scaler.save",
        window_size=128,
    )
    _use_real_models = True
    print("[ENGINE] ✅ Real PyTorch models loaded — LSTM, CNN, Transformer, GNN")
except Exception as _e:
    print(f"[ENGINE] ⚠ Could not load real models: {_e}")
    print("[ENGINE] Falling back to statistical methods")

# ── Sensor column order must match training ─────────────────────
# Your InferenceManager.sensor_names order:
# ["xTemp","zTemp","yTemp","X Coarse Acceleration","Y Coarse Acceleration",
#  "Z Coarse Acceleration","X Fine Acceleration","Y Fine Acceleration","Z Fine Acceleration"]
_SENSOR_COLS = [
    "xTemp", "zTemp", "yTemp",
    "X Coarse Acceleration", "Y Coarse Acceleration", "Z Coarse Acceleration",
    "X Fine Acceleration",   "Y Fine Acceleration",   "Z Fine Acceleration",
]

# Sensor valid ranges — values outside these are bad sensor readings
# Derived from your data stats: mean ± 5*std, with hard physical limits
_SENSOR_VALID_RANGES = {
    "xTemp":                   (-100.0,  100.0),   # °C, normal ~14
    "zTemp":                   (-100.0,  100.0),   # °C, normal ~14
    "yTemp":                   (-100.0,  100.0),   # °C, normal ~14
    "X Coarse Acceleration":   (-5.0,    5.0),
    "Y Coarse Acceleration":   (-5.0,    5.0),
    "Z Coarse Acceleration":   (-5.0,    5.0),
    "X Fine Acceleration":     (-1.0,    1.0),
    "Y Fine Acceleration":     (-1.0,    1.0),
    "Z Fine Acceleration":     (-1.0,    1.0),
}

def _is_valid_packet(packet: dict) -> bool:
    """Return False if packet contains sensor initialization garbage (e.g. xTemp=-4080)."""
    for col, (lo, hi) in _SENSOR_VALID_RANGES.items():
        v = safe(packet, col)
        if v < lo or v > hi:
            return False
    return True

def _build_window() -> np.ndarray:
    """
    Build a (N, 9) numpy array from rolling history for real model inference.
    - Skips invalid packets (sensor init values like -4080)
    - Returns None if fewer than 128 valid packets available
    """
    if len(history) < 2:
        return None
    rows = []
    for h in history:
        if _is_valid_packet(h):
            row = [safe(h, col) for col in _SENSOR_COLS]
            rows.append(row)
    if len(rows) < 128:
        return None   # not enough clean data yet
    return np.array(rows, dtype=np.float32)


# ================================================================
# STA/LTA SEISMIC EVENT DETECTOR (kept — real seismology algorithm)
# ================================================================

class STALTADetector:
    """
    Classic STA/LTA (Short-Term Average / Long-Term Average) ratio detector.
    Industry-standard algorithm used by InSight, Apollo ALSEP, Chandrayaan-3 ILSA.
    Reference: Allen (1982) — Bulletin of the Seismological Society of America.
    """
    def __init__(self, sta_len=STA_LEN, lta_len=LTA_LEN,
                 trigger=STALTA_TRIGGER, detrig=STALTA_DETRIG):
        self.sta_len = sta_len
        self.lta_len = lta_len
        self.trigger_on  = trigger
        self.detrig_off  = detrig
        self._triggered  = False
        self._event_start = None

    def compute_ratio(self, signal: np.ndarray) -> float:
        if len(signal) < self.lta_len:
            return 0.0
        sq  = signal ** 2
        sta = float(np.mean(sq[-self.sta_len:]))
        lta = float(np.mean(sq[-self.lta_len:]))
        return sta / lta if lta > 1e-10 else 0.0

    def update(self, signal: np.ndarray, t: int) -> dict:
        ratio          = self.compute_ratio(signal)
        event_detected = False
        event_ended    = False
        if not self._triggered and ratio >= self.trigger_on:
            self._triggered   = True
            self._event_start = t
            event_detected    = True
        elif self._triggered and ratio < self.detrig_off:
            self._triggered = False
            event_ended     = True
        return {
            "stalta_ratio":   round(ratio, 4),
            "triggered":      self._triggered,
            "event_detected": event_detected,
            "event_ended":    event_ended,
            "event_start":    self._event_start,
        }

stalta_detector = STALTADetector()


# ================================================================
# LUNAR QUAKE CLASSIFIER (bandpass filter — real signal processing)
# ================================================================

class CNNSeismicClassifier:
    """
    Classifies seismic events using real bandpass filters.
    Based on Nakamura (1977), Lognonné et al. (2020), Kumar et al. (2023).
    When real CNN weights are available, also runs your trained CNNAutoencoder
    and uses its reconstruction error to confirm the classification.
    """
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr

    def _bandpass_energy(self, signal: np.ndarray, low: float, high: float) -> float:
        nyq = self.sr / 2.0
        lo, hi = max(0.001, min(low/nyq, 0.999)), max(0.001, min(high/nyq, 0.999))
        if lo >= hi or len(signal) < 15:
            return float(np.mean(signal**2))
        try:
            b, a = butter(2, [lo, hi], btype='band')
            return float(np.mean(filtfilt(b, a, signal)**2))
        except Exception:
            return float(np.mean(signal**2))

    def run(self, data: dict) -> dict:
        signal_z = np.array([safe(h, "Z Fine Acceleration") for h in history])
        signal_x = np.array([safe(h, "X Fine Acceleration") for h in history])
        signal_y = np.array([safe(h, "Y Fine Acceleration") for h in history])
        fx = safe(data, "X Fine Acceleration")
        fy = safe(data, "Y Fine Acceleration")
        fz = safe(data, "Z Fine Acceleration")
        rms = float(np.sqrt(fx**2 + fy**2 + fz**2))
        combined = np.sqrt(signal_x**2 + signal_y**2 + signal_z**2) if len(signal_z) > 0 else np.array([rms])

        # Bandpass classification
        band_energies = {}
        for qtype, params in MOONQUAKE_TYPES.items():
            lo, hi = params["freq_band"]
            hi = min(hi, self.sr / 2.0 * 0.95)
            band_energies[qtype] = self._bandpass_energy(combined, lo, hi) if lo < hi else float(np.mean(combined**2))

        if len(combined) > 1:
            zc = np.where(np.diff(np.sign(combined)))[0]
            freq_dominant = len(zc) / (len(combined) / self.sr) / 2.0
        else:
            freq_dominant = 0.0

        best_type  = max(band_energies, key=band_energies.get) if band_energies else "unknown"
        total_e    = sum(band_energies.values()) or 1e-10
        confidence = round(band_energies[best_type] / total_e, 3) if band_energies else 0.0

        # ── Real CNN reconstruction error (if models loaded) ──────
        # CNN p95=0.000317, p99=0.00872 from your training data
        CNN_WARN_THRESHOLD = 0.000317
        CNN_CRIT_THRESHOLD = 0.00872
        cnn_recon_error = 0.0
        cnn_norm_score  = 0.0
        if _use_real_models and _inference_manager:
            try:
                win = _build_window()
                if win is not None and len(win) >= 128:
                    res = _inference_manager.predict("cnn", win)
                    cnn_recon_error = float(res["overall_score"])
                    if cnn_recon_error <= CNN_WARN_THRESHOLD:
                        cnn_norm_score = cnn_recon_error / CNN_WARN_THRESHOLD * WARN_THRESHOLD
                    else:
                        import math as _math
                        log_raw  = _math.log(cnn_recon_error + 1e-10)
                        log_warn = _math.log(CNN_WARN_THRESHOLD + 1e-10)
                        log_crit = _math.log(CNN_CRIT_THRESHOLD + 1e-10)
                        t = (log_raw - log_warn) / (log_crit - log_warn + 1e-10)
                        cnn_norm_score = float(min(WARN_THRESHOLD + t * (1.0 - WARN_THRESHOLD), 1.0))
            except Exception:
                pass

        # Severity — use normalised CNN score, not raw * 2.0
        combined_rms = max(rms, cnn_norm_score * 0.5)
        stalta_ratio = current_state.get("models", {}).get("lstm", {}).get("stalta_ratio", 0)
        if combined_rms > 0.3 or stalta_ratio > 5.0:
            severity, pattern = "critical", "STRONG_SEISMIC_EVENT"
        elif combined_rms > 0.1 or stalta_ratio > STALTA_TRIGGER:
            severity, pattern = "warning", f"SEISMIC_EVENT_{best_type.upper()}"
        else:
            severity, pattern = "nominal", "NORMAL"

        return {
            "model":            "CNN",
            "event_type":       best_type,
            "confidence":       confidence,
            "freq_dominant":    round(freq_dominant, 4),
            "band_energies":    {k: round(v, 8) for k, v in band_energies.items()},
            "pattern":          pattern,
            "severity":         severity,
            "rms_accel":        round(rms, 6),
            "cnn_recon_error":  round(cnn_recon_error, 6),
            "fine_y":           round(fy, 6),
            "fine_z":           round(fz, 6),
        }


# ================================================================
# LSTM — YOUR REAL LSTMAutoencoder + STA/LTA
# ================================================================

class LSTMModel:
    """
    Uses your real trained LSTMAutoencoder for reconstruction-based
    anomaly detection. Score = mean reconstruction error across all 9 sensors.
    Also runs STA/LTA on Z Fine Acceleration for seismic onset detection.
    Falls back to z-score if weights not available.
    """
    def run(self, data: dict) -> dict:
        t = current_state.get("packet_count", 0)

        # ── STA/LTA (always runs — real seismology) ──────────────
        signal_z = np.array([safe(h, "Z Fine Acceleration") for h in history])
        stalta   = stalta_detector.update(signal_z, t)
        ratio    = stalta["stalta_ratio"]
        # Only contribute to score when truly triggered (ratio > trigger threshold)
        # STA/LTA = 1.0 is NORMAL (equal short/long term energy) — not an anomaly
        # Only flag when ratio significantly EXCEEDS trigger (3.0)
        if ratio > STALTA_TRIGGER:
            seismic_score = float(min((ratio - STALTA_TRIGGER) / (STALTA_TRIGGER * 2.0), 1.0))
        else:
            seismic_score = 0.0

        # ── Real LSTM reconstruction error ────────────────────────
        # Thresholds derived from your actual error distribution:
        #   p95  = 0.000179  (normal operation ceiling)
        #   p99  = 0.434907  (anomaly boundary)
        #   warn = mean + 2*std scaled to 0-1
        # We normalise using: score = raw_error / p99_threshold
        # so score > 1.0 → genuine anomaly, clipped to 1.0
        LSTM_WARN_THRESHOLD = 0.00018   # p95 — below this is normal
        LSTM_CRIT_THRESHOLD = 0.43491   # p99 — above this is anomaly

        lstm_score = 0.0
        sensor_errors = {}
        if _use_real_models and _inference_manager:
            try:
                win = _build_window()
                if win is not None and len(win) >= 128:
                    res       = _inference_manager.predict("lstm", win)
                    raw_error = float(res["overall_score"])
                    # Normalise to 0-1 using p99 as the anomaly boundary
                    # Errors below p95 → near 0, errors at p99 → ~1.0
                    if raw_error <= LSTM_WARN_THRESHOLD:
                        lstm_score = raw_error / LSTM_WARN_THRESHOLD * WARN_THRESHOLD
                    else:
                        # Log-scale between warn and crit for smoother response
                        import math as _math
                        log_raw  = _math.log(raw_error + 1e-10)
                        log_warn = _math.log(LSTM_WARN_THRESHOLD + 1e-10)
                        log_crit = _math.log(LSTM_CRIT_THRESHOLD + 1e-10)
                        t = (log_raw - log_warn) / (log_crit - log_warn + 1e-10)
                        lstm_score = float(min(WARN_THRESHOLD + t * (1.0 - WARN_THRESHOLD), 1.0))
                    sensor_errors = {k: round(float(v), 6) for k, v in res["sensor_errors"].items()}
                    print(f"[LSTM] raw={raw_error:.6f} → score={lstm_score:.4f}")
            except Exception as e:
                print(f"[LSTM] inference error: {e}")

        # ── Fallback: z-score on xTemp ───────────────────────────
        x  = safe(data, "xTemp")
        mu = float(np.mean([safe(h, "xTemp") for h in history])) if history else x
        sigma = float(np.std([safe(h, "xTemp") for h in history])) or 1.0
        if lstm_score == 0.0:
            temps  = [safe(h, "xTemp") for h in history] if history else [x]
            lstm_score = float(min(abs(x - mu) / sigma / 3.0, 1.0))

        # ── Combined score ────────────────────────────────────────
        # Real model dominates (70%), STA/LTA only adds if genuinely triggered
        if _use_real_models and lstm_score > 0:
            score = round(min(0.7 * lstm_score + 0.3 * seismic_score, 1.0), 6)
        else:
            score = round(max(lstm_score, seismic_score), 6)
        status = "ANOMALY" if score > CRIT_THRESHOLD else "WARNING" if score > WARN_THRESHOLD else "NORMAL"
        cause  = "thermal" if lstm_score >= seismic_score else "seismic"

        return {
            "model":            "LSTM",
            "score":            score,
            "status":           status,
            "lstm_recon_score": round(lstm_score, 6),
            "z_score":          round(abs(x - mu) / sigma, 4),
            "mean_temp":        round(mu, 4),
            "xTemp":            round(x, 4),
            "seismic_score":    round(seismic_score, 6),
            "stalta_ratio":     ratio,
            "stalta_triggered": stalta["triggered"],
            "event_detected":   stalta["event_detected"],
            "dominant_cause":   cause,
            "sensor_errors":    sensor_errors,
            "real_model":       _use_real_models,
        }


# ================================================================
# TRANSFORMER — YOUR REAL TransformerAutoencoder
# ================================================================

class TransformerModel:
    """
    Uses your real trained TransformerAutoencoder for reconstruction.
    Reconstruction error per sensor reveals which channel is anomalous.
    Also computes linear trend on xTemp for temperature forecasting.
    """
    def run(self, data: dict) -> dict:
        x_now = safe(data, "xTemp")
        y_now = safe(data, "yTemp")
        z_now = safe(data, "zTemp")

        # ── Linear trend (fast, always available) ────────────────
        if len(history) >= 5:
            xs     = [safe(h, "xTemp") for h in history]
            t_arr  = np.arange(len(xs), dtype=float)
            coeffs = np.polyfit(t_arr, xs, 1)
            slope  = float(coeffs[0])
            pred   = float(np.polyval(coeffs, len(xs)))
        else:
            slope, pred = 0.0, x_now

        trend = "RISING" if slope > 0.01 else "FALLING" if slope < -0.01 else "STABLE"

        # ── Real Transformer reconstruction error ─────────────────
        # Transformer p95=0.000207, p99=0.001652 from your training data
        TRANS_WARN_THRESHOLD = 0.000207
        TRANS_CRIT_THRESHOLD = 0.001652
        trans_score   = 0.0
        trans_raw     = 0.0
        sensor_errors = {}
        if _use_real_models and _inference_manager:
            try:
                win = _build_window()
                if win is not None and len(win) >= 128:
                    res      = _inference_manager.predict("transformer", win)
                    trans_raw = float(res["overall_score"])
                    sensor_errors = {k: round(float(v), 6) for k, v in res["sensor_errors"].items()}
                    # Normalise using log-scale
                    if trans_raw <= TRANS_WARN_THRESHOLD:
                        trans_score = trans_raw / TRANS_WARN_THRESHOLD * WARN_THRESHOLD
                    else:
                        import math as _math
                        log_raw  = _math.log(trans_raw + 1e-10)
                        log_warn = _math.log(TRANS_WARN_THRESHOLD + 1e-10)
                        log_crit = _math.log(TRANS_CRIT_THRESHOLD + 1e-10)
                        t = (log_raw - log_warn) / (log_crit - log_warn + 1e-10)
                        trans_score = float(min(WARN_THRESHOLD + t * (1.0 - WARN_THRESHOLD), 1.0))
                    # Refine temperature prediction using transformer error
                    if sensor_errors:
                        xe   = sensor_errors.get("xTemp", 0)
                        pred = round(x_now + slope - xe * 0.05, 4)
            except Exception as e:
                print(f"[TRANS] inference error: {e}")

        return {
            "model":             "Transformer",
            "pred_xTemp":        round(pred, 4),
            "current_xTemp":     round(x_now, 4),
            "current_yTemp":     round(y_now, 4),
            "current_zTemp":     round(z_now, 4),
            "slope":             round(slope, 6),
            "trend":             trend,
            "recon_score":       round(trans_score, 6),
            "sensor_errors":     sensor_errors,
            "real_model":        _use_real_models,
        }


# ================================================================
# GNN — YOUR REAL SpacecraftGNN
# ================================================================

class GNNModel:
    """
    Uses your real trained SpacecraftGNN (GCNConv) to compute
    per-sensor risk scores across the sensor graph.
    Graph edges defined in core/sensor_graph.py:
    thermal ↔ thermal, coarse ↔ coarse, fine ↔ fine,
    plus cross-links: Temp X ↔ Accel coarse X etc.
    Falls back to rule-based if weights not available.
    """
    def _build_edge_index(self):
        """Rebuild edge index matching core/sensor_graph.py."""
        try:
            import torch as _torch
            _sys.path.insert(0, r"G:\Pro_NML")
            from core.sensor_graph import edge_index as _ei
            return _ei
        except Exception:
            return None

    def run(self, data: dict) -> dict:
        x   = safe(data, "xTemp")
        cx  = safe(data, "X Coarse Acceleration")
        cy  = safe(data, "Y Coarse Acceleration")
        cz  = safe(data, "Z Coarse Acceleration")
        fx  = safe(data, "X Fine Acceleration")
        fy  = safe(data, "Y Fine Acceleration")
        fz  = safe(data, "Z Fine Acceleration")
        c_mag = float(np.sqrt(cx**2 + cy**2 + cz**2))
        f_mag = float(np.sqrt(fx**2 + fy**2 + fz**2))

        issues = []
        if abs(x)  > 40:  issues.append("high_xTemp")       # normal is ~14, warn at 40
        if c_mag   > 1.5: issues.append("high_coarse_accel") # normal ~0.65 mag
        if f_mag   > 0.3: issues.append("high_fine_accel")   # normal ~0.124 mag

        stalta_active = current_state.get("models", {}).get("lstm", {}).get("stalta_triggered", False)
        # Coupling requires BOTH stalta triggered AND temperature genuinely elevated
        seismic_thermal_coupling = stalta_active and abs(x) > 30
        # Cross anomaly only if 2+ sensors genuinely out of range
        cross_anomaly = len(issues) >= 2 or seismic_thermal_coupling

        # ── Default subsystem scores ──────────────────────────────
        # Calibrated to your real sensor ranges:
        # xTemp normal ~14°C (max ~46), so /60 gives good headroom
        # Coarse accel normal ~0.65 magnitude, /3.0 keeps it high
        # Fine accel normal ~0.124 magnitude, /0.4 keeps it high
        subsystem_scores = {
            "thermal":    max(0.0, 1.0 - max(0, abs(x) - 10) / 50.0),
            "structural": max(0.0, 1.0 - c_mag / 3.0),
            "seismic":    max(0.0, 1.0 - f_mag / 0.4),
            "navigation": 1.0,
            "power":      1.0,
        }

        # ── Real GNN inference ────────────────────────────────────
        gnn_node_scores = {}
        graph_risk      = 0.0
        if _use_real_models and _inference_manager:
            try:
                import torch as _torch
                # Use LSTM sensor errors as node features for GNN
                lstm_out = current_state.get("models", {}).get("lstm", {})
                sensor_errs = lstm_out.get("sensor_errors", {})
                if sensor_errs:
                    sensor_vec = np.array([sensor_errs.get(c, 0.0) for c in _SENSOR_COLS], dtype=np.float32)
                    # Normalise
                    norm = np.linalg.norm(sensor_vec) + 1e-6
                    sensor_vec = sensor_vec / norm
                    sensor_vec = sensor_vec - sensor_vec.mean()

                    ei = self._build_edge_index()
                    if ei is not None:
                        from models.gnn.model import SpacecraftGNN as _GNN
                        gnn_model = _inference_manager.models.get("gnn")
                        if gnn_model is not None:
                            node_x = _torch.tensor(sensor_vec, dtype=_torch.float32).unsqueeze(-1).to(_inference_manager.device)
                            ei_dev = ei.to(_inference_manager.device)
                            with _torch.no_grad():
                                out = gnn_model(node_x, ei_dev)
                            scores_np = out.abs().cpu().numpy().flatten()
                            sensor_names = ["xTemp","zTemp","yTemp","CoarseX","CoarseY","CoarseZ","FineX","FineY","FineZ"]
                            gnn_node_scores = {sensor_names[i]: round(float(scores_np[i]), 4) for i in range(len(scores_np))}
                            graph_risk = float(np.tanh(scores_np.mean()))

                            # Update subsystem scores from GNN output
                            thermal_risk    = np.mean(scores_np[:3])
                            structural_risk = np.mean(scores_np[3:6])
                            seismic_risk    = np.mean(scores_np[6:9])
                            subsystem_scores["thermal"]    = round(max(0.0, 1.0 - float(np.tanh(thermal_risk))), 4)
                            subsystem_scores["structural"] = round(max(0.0, 1.0 - float(np.tanh(structural_risk))), 4)
                            subsystem_scores["seismic"]    = round(max(0.0, 1.0 - float(np.tanh(seismic_risk))), 4)
            except Exception as e:
                print(f"[GNN] inference error: {e}")

        return {
            "model":                    "GNN",
            "subsystem":                "structure",
            "issues":                   issues,
            "cross_anomaly":            cross_anomaly,
            "seismic_thermal_coupling": seismic_thermal_coupling,
            "subsystem_scores":         {k: round(v, 4) for k, v in subsystem_scores.items()},
            "gnn_node_scores":          gnn_node_scores,
            "graph_risk":               round(graph_risk, 4),
            "coarse_rms":               round(c_mag, 6),
            "fine_rms":                 round(f_mag, 6),
            "real_model":               _use_real_models and bool(gnn_node_scores),
        }


# ================================================================
# MODEL INSTANCES
# ================================================================
lstm        = LSTMModel()
cnn         = CNNSeismicClassifier()
transformer = TransformerModel()
gnn         = GNNModel()

# ================================================================
# MODEL ROUTER
# ================================================================
def select_model(question: str):
    q = question.lower()
    if any(k in q for k in ["pattern", "vibration", "seismic", "quake", "moonquake", "impact"]): return cnn
    if any(k in q for k in ["predict",  "forecast",  "trend"]):   return transformer
    if any(k in q for k in ["cause",    "graph",     "subsystem", "coupling"]): return gnn
    return lstm

def model_name(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["pattern", "vibration", "seismic", "quake"]): return "CNN"
    if any(k in q for k in ["predict",  "forecast",  "trend"]):            return "Transformer"
    if any(k in q for k in ["cause",    "graph",     "subsystem"]):        return "GNN"
    return "LSTM"



# ================================================================
# CHANDRAYAAN-3 ILSA KNOWLEDGE BASE
# ================================================================
# Maps anomaly patterns → physical causes → specific solutions
# Based on ILSA instrument design and lunar surface conditions
# ================================================================

ILSA_KNOWLEDGE_BASE = {

    # ── THERMAL ANOMALIES ──────────────────────────────────────────
    "thermal_high_xtemp": {
        "title":   "xTemp Sensor Anomaly — Above Normal Range",
        "causes": [
            "Lunar noon heating — surface temperature can reach +130°C",
            "ILSA instrument thermal cycling during day/night transition",
            "Internal heater circuit activation",
            "Solar radiation angle change causing direct illumination",
        ],
        "risk":    "MEDIUM — sustained high temp can affect sensor calibration",
        "actions": [
            "1. Check mission timeline — is spacecraft in lunar noon period?",
            "2. Compare xTemp with yTemp and zTemp — if all rising: external heating",
            "3. If only xTemp rising: check ILSA heater circuit status",
            "4. If xTemp > 40°C: activate thermal regulation mode",
            "5. If xTemp > 45°C: consider entering instrument safe mode",
            "6. Log timestamp and temperature value for ground station review",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("dominant_cause") == "thermal" and
            s.get("xTemp", 0) > 20
        ),
    },

    "thermal_all_sensors": {
        "title":   "Multi-Sensor Thermal Anomaly — All Temp Channels Affected",
        "causes": [
            "Spacecraft entering full lunar day — ambient temperature rise",
            "Thermal protection system degradation",
            "Heat dissipation from other spacecraft subsystems",
            "Lunar regolith thermal conductivity change",
        ],
        "risk":    "HIGH — affects all 3 temperature sensors simultaneously",
        "actions": [
            "1. Verify spacecraft attitude — is ILSA exposed to direct solar flux?",
            "2. Check power subsystem — excess current generates internal heat",
            "3. Activate thermal control sequence if available",
            "4. Reduce instrument duty cycle to lower self-heating",
            "5. Monitor trend — if all temps rising linearly: external cause",
            "6. Report to mission control with temperature gradient data",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("dominant_cause") == "thermal" and
            abs(s.get("xTemp", 0)) > 15 and
            abs(s.get("yTemp", 0)) > 15
        ),
    },

    "thermal_falling": {
        "title":   "Temperature Drop — Lunar Night Entry Detected",
        "causes": [
            "Spacecraft entering lunar night — temperature drops to -180°C",
            "Shadow from crater rim or surface feature",
            "Loss of solar power affecting instrument heaters",
        ],
        "risk":    "MEDIUM — cold temperatures can cause thermal stress on instrument",
        "actions": [
            "1. Verify lunar night timeline — expected or unexpected?",
            "2. Check instrument heater power status",
            "3. Monitor acceleration sensors for thermal cracking signatures",
            "4. If temperature dropping rapidly: prepare for hibernation mode",
            "5. Ensure data buffers are flushed before potential power loss",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("trend") == "FALLING" and
            m.get("dominant_cause") == "thermal"
        ),
    },

    # ── SEISMIC ANOMALIES ──────────────────────────────────────────
    "seismic_deep_moonquake": {
        "title":   "Deep Moonquake Detected",
        "causes": [
            "Tidal forces from Earth causing deep interior fracturing",
            "Deep moonquakes occur at 700-1200 km depth",
            "Characteristic low frequency (0.5-1.0 Hz), long duration (>60s)",
            "Repeating events — same source activated by tidal cycle",
        ],
        "risk":    "LOW for instrument — deep quakes are weak at surface",
        "actions": [
            "1. Log event timestamp, STA/LTA ratio, and duration",
            "2. Record full waveform for ground station analysis",
            "3. Check if event matches known deep moonquake nests (A-type sources)",
            "4. Correlate with Earth-Moon distance — tidal triggering likely",
            "5. No immediate instrument action required — continue monitoring",
            "6. Flag event in mission log as SCIENTIFIC DETECTION",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("event_type") == "deep_moonquake"
        ),
    },

    "seismic_shallow_moonquake": {
        "title":   "Shallow Moonquake Detected — High Priority",
        "causes": [
            "Tectonic stress release in lunar crust (0-200 km depth)",
            "Largest moonquakes — magnitude up to 5.5",
            "Can last 10+ minutes due to low lunar attenuation",
            "Rarer than deep moonquakes but significantly stronger",
        ],
        "risk":    "HIGH — shallow moonquakes can damage surface instruments",
        "actions": [
            "1. IMMEDIATE: Check instrument physical integrity",
            "2. Verify ILSA mounting and coupling to lunar surface",
            "3. Check all sensor channels for saturation or offset",
            "4. Record full event — this is rare and scientifically valuable",
            "5. Compare amplitude with Apollo seismic network data",
            "6. Alert mission control — potential instrument health check required",
            "7. After event: run instrument self-test and calibration check",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("event_type") == "shallow_moonquake"
        ),
    },

    "seismic_meteorite_impact": {
        "title":   "Meteorite Impact Detected",
        "causes": [
            "Meteoroid impacting lunar surface near ILSA deployment site",
            "Characteristic impulsive onset, broadband frequency content",
            "Impact energy converts to seismic and acoustic waves",
            "Moon receives ~100 ton of meteoritic material per day",
        ],
        "risk":    "LOW-MEDIUM — depends on proximity of impact",
        "actions": [
            "1. Note exact timestamp for crater identification via orbital imagery",
            "2. Estimate impact distance from waveform amplitude decay",
            "3. Record peak ground velocity for impact energy estimation",
            "4. Check for any ejecta-related instrument contamination",
            "5. Report to mission control — potential new crater formation",
            "6. Cross-reference with Lunar Reconnaissance Orbiter imagery",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("event_type") == "meteorite_impact"
        ),
    },

    "seismic_thermal_cracking": {
        "title":   "Thermal Cracking / Thermoelastic Event",
        "causes": [
            "Rapid temperature change causing rock fracture",
            "Common at lunar sunrise/sunset — 300°C/hour temperature gradient",
            "Regolith expansion/contraction generating seismic signals",
            "ILSA instrument mounting bracket thermal stress",
        ],
        "risk":    "LOW — natural lunar surface process, not structural threat",
        "actions": [
            "1. Check mission timeline — near lunar sunrise or sunset?",
            "2. Correlate with temperature sensor readings",
            "3. If both thermal anomaly AND seismic detected: thermoelastic confirmed",
            "4. No immediate action required — document event",
            "5. Monitor for increasing frequency of events",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("event_type") == "thermal_cracking"
        ),
    },

    # ── COUPLING ANOMALIES ─────────────────────────────────────────
    "seismic_thermal_coupling": {
        "title":   "Seismic-Thermal Coupling Detected — Complex Event",
        "causes": [
            "Thermoelastic moonquake: temperature change causing seismic signal",
            "Meteorite impact heating + seismic wave simultaneously",
            "Instrument thermal expansion causing spurious seismic reading",
            "Genuine moonquake coinciding with thermal transition period",
        ],
        "risk":    "MEDIUM-HIGH — complex event, root cause uncertain",
        "actions": [
            "1. Check if thermal change preceded seismic by <5 minutes (thermoelastic)",
            "2. If seismic preceded thermal: possible impact heating event",
            "3. Check instrument self-noise level — is seismic signal real or artifact?",
            "4. Run instrument bias check to separate thermal and seismic components",
            "5. Report as complex event requiring ground station analysis",
            "6. Do NOT dismiss as noise — document with full sensor data",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("seismic_thermal_coupling") == True
        ),
    },

    # ── ACCELERATION ANOMALIES ─────────────────────────────────────
    "high_coarse_acceleration": {
        "title":   "High Coarse Acceleration — Structural Concern",
        "causes": [
            "Physical disturbance to spacecraft or ILSA instrument",
            "Wind? No — lunar surface has no atmosphere",
            "Possible: lander settling into regolith",
            "Possible: thermal expansion of lander structure",
            "Possible: nearby impact ejecta hitting lander",
        ],
        "risk":    "HIGH — unexpected physical movement of instrument",
        "actions": [
            "1. Check lander attitude control system status",
            "2. Verify ILSA is still properly coupled to lunar surface",
            "3. Check for instrument tilt — level sensor if available",
            "4. Compare coarse vs fine acceleration — same direction?",
            "5. If lander moved: recalibrate ILSA orientation",
            "6. Alert mission control immediately",
        ],
        "threshold_trigger": lambda s, m: (
            "high_coarse_accel" in m.get("issues", [])
        ),
    },

    # ── CRITICAL / COMPOSITE ──────────────────────────────────────
    "critical_composite": {
        "title":   "CRITICAL — Multiple Systems Anomalous",
        "causes": [
            "Multiple simultaneous failures — rare but serious",
            "Power supply instability affecting all sensors",
            "Software/firmware issue causing sensor readout errors",
            "Major seismic event affecting entire instrument",
        ],
        "risk":    "CRITICAL — immediate attention required",
        "actions": [
            "1. IMMEDIATE: Alert mission control",
            "2. Run full instrument diagnostic sequence",
            "3. Check power supply voltage and current draw",
            "4. Attempt instrument soft reset if communication available",
            "5. Switch to backup data recording mode",
            "6. Preserve all data — do not overwrite buffers",
            "7. Prepare for potential safe mode entry",
        ],
        "threshold_trigger": lambda s, m: (
            m.get("score", 0) > 0.7 and
            len(m.get("issues", [])) >= 2
        ),
    },
}


def diagnose_anomaly(state: dict) -> dict:
    """
    Match current state against knowledge base to find best matching
    anomaly type, cause, and recommended actions.
    Returns the best matching knowledge base entry + confidence.
    """
    m      = state.get("models", {})
    lstm   = m.get("lstm",  {})
    cnn    = m.get("cnn",   {})
    trans  = m.get("transformer", {})
    gnn    = m.get("gnn",   {})
    s      = state.get("sensors", {})

    score  = lstm.get("score", 0)
    status = lstm.get("status", "NORMAL")

    # Build combined model context for trigger evaluation
    combined = {
        "score":                    score,
        "dominant_cause":           lstm.get("dominant_cause", ""),
        "event_type":               cnn.get("event_type", ""),
        "trend":                    trans.get("trend", "STABLE"),
        "seismic_thermal_coupling": gnn.get("seismic_thermal_coupling", False),
        "issues":                   gnn.get("issues", []),
        "stalta_triggered":         lstm.get("stalta_triggered", False),
    }

    if status == "NORMAL":
        return {
            "matched":  False,
            "title":    "All Systems Nominal",
            "causes":   ["No anomaly detected — LSTM reconstruction error within normal range"],
            "risk":     "NONE",
            "actions":  ["Continue standard monitoring", "No action required"],
        }

    # Check each knowledge base entry
    matches = []
    for key, entry in ILSA_KNOWLEDGE_BASE.items():
        try:
            if entry["threshold_trigger"](s, combined):
                matches.append((key, entry))
        except Exception:
            pass

    if not matches:
        # Default anomaly entry
        return {
            "matched":  True,
            "key":      "unknown",
            "title":    f"Anomaly Detected — Score {score:.4f}",
            "causes":   [
                f"LSTM reconstruction error elevated ({score:.4f} > threshold 0.20)",
                f"Dominant cause identified as: {combined['dominant_cause'].upper()}",
                "Specific physical cause requires further investigation",
            ],
            "risk":     "MEDIUM",
            "actions":  [
                "1. Monitor all sensor channels for trend continuation",
                "2. Check if anomaly correlates with known mission timeline events",
                "3. Compare with historical baseline from mission start",
                "4. Report to ground station if score remains elevated > 5 minutes",
            ],
        }

    # Return the highest-priority match
    # Priority: critical_composite > seismic > thermal
    priority_order = [
        "critical_composite",
        "seismic_shallow_moonquake",
        "meteorite_impact",
        "seismic_thermal_coupling",
        "seismic_deep_moonquake",
        "seismic_thermal_cracking",
        "high_coarse_acceleration",
        "thermal_all_sensors",
        "thermal_high_xtemp",
        "thermal_falling",
    ]

    best = None
    for pkey in priority_order:
        for mkey, mentry in matches:
            if mkey == pkey:
                best = mentry
                break
        if best:
            break

    if not best:
        best = matches[0][1]

    return {
        "matched":  True,
        "title":    best["title"],
        "causes":   best["causes"],
        "risk":     best["risk"],
        "actions":  best["actions"],
    }

# ================================================================
# LLM (Ollama llama3)
# ================================================================
def explain(model_result: dict, question: str, sensors: dict) -> str:
    m      = current_state.get("models", {})
    lstm_m = m.get("lstm", {})
    cnn_m  = m.get("cnn",  {})
    trans_m= m.get("transformer", {})
    gnn_m  = m.get("gnn",  {})

    # ── Run knowledge base diagnosis ──────────────────────────────
    diagnosis = diagnose_anomaly(current_state)

    # Find top anomalous sensors
    sensor_errs    = lstm_m.get("sensor_errors", {})
    top_sensors    = sorted(sensor_errs.items(), key=lambda x: x[1], reverse=True)[:3] if sensor_errs else []
    top_sensor_str = ", ".join(f"{k}={v:.5f}" for k,v in top_sensors) if top_sensors else "unavailable"

    score    = lstm_m.get("score", 0)
    cause    = lstm_m.get("dominant_cause", "unknown")
    trend    = trans_m.get("trend", "STABLE")
    slope    = trans_m.get("slope", 0)
    xtemp    = safe(sensors, "xTemp")
    gnn_sc   = gnn_m.get("subsystem_scores", {})

    # Format knowledge base diagnosis for the prompt
    kb_causes  = "\n".join(f"  - {c}" for c in diagnosis["causes"])
    kb_actions = "\n".join(f"  {a}" for a in diagnosis["actions"])

    prompt = f"""You are an expert spacecraft mission AI analyst for Chandrayaan-3 ILSA seismometer.
Answer in clear, structured paragraphs. Be specific — use exact values.

KNOWLEDGE BASE DIAGNOSIS:
- Anomaly type: {diagnosis["title"]}
- Risk level: {diagnosis["risk"]}
- Possible physical causes:
{kb_causes}
- Recommended actions:
{kb_actions}

LIVE SENSOR DATA:
- LSTM score: {score:.4f} (warn>0.05, CRITICAL>0.20) — Status: {lstm_m.get('status','?')}
- Dominant cause: {cause.upper()}
- Top anomalous sensors (reconstruction error): {top_sensor_str}
- xTemp={xtemp:.4f}°C  yTemp={safe(sensors,'yTemp'):.4f}°C  zTemp={safe(sensors,'zTemp'):.4f}°C
- Temperature trend: {trend} (slope={slope:.6f}, pred={trans_m.get('pred_xTemp',0):.4f}°C)
- CNN seismic event: {cnn_m.get('event_type','none')} (severity: {cnn_m.get('severity','?')})
- STA/LTA: {lstm_m.get('stalta_ratio',0):.4f} ({'TRIGGERED' if lstm_m.get('stalta_triggered') else 'idle'})
- GNN: thermal={gnn_sc.get('thermal',1):.3f} structural={gnn_sc.get('structural',1):.3f} seismic={gnn_sc.get('seismic',1):.3f}
- Coupling: {gnn_m.get('seismic_thermal_coupling', False)}

OPERATOR QUESTION: {question}

Using the knowledge base diagnosis AND live sensor data above, provide:
1. Direct answer to the question
2. The most likely physical cause based on the specific sensor values
3. What the operator should do RIGHT NOW (reference the recommended actions)
4. What to expect next based on current trend

Be technical and specific. Reference actual values. Do not be vague.
"""
    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=90)
        resp = r.json().get("response", "")
        if resp:
            return resp
        return "Ollama returned empty response. Try asking again."
    except requests.exceptions.ConnectionError:
        # Fallback: use knowledge base directly without Ollama
        score  = lstm_m.get("score", 0)
        trend  = trans_m.get("trend", "STABLE")
        slope  = trans_m.get("slope", 0)
        status = lstm_m.get("status", "NORMAL")
        q      = question.lower()

        # Always run diagnosis even without Ollama
        d = diagnose_anomaly(current_state)
        causes_str  = " | ".join(d["causes"][:2])
        actions_str = " → ".join(d["actions"][:3])

        if any(w in q for w in ["why","cause","reason","explain"]):
            return (
                f"[Ollama offline - Knowledge Base Answer]\n\n"
                f"ANOMALY TYPE: {d['title']}\n"
                f"RISK: {d['risk']}\n\n"
                f"MOST LIKELY CAUSE: {causes_str}\n\n"
                f"RECOMMENDED ACTIONS: {actions_str}\n\n"
                f"Score: {score:.4f} | Cause: {lstm_m.get('dominant_cause','?').upper()} | "
                f"xTemp: {safe(sensors,'xTemp'):.2f}°C ({trend})"
            )
        elif any(w in q for w in ["future","predict","forecast","will"]):
            direction = "rising — anomaly may worsen" if slope > 0.001 else "falling — returning to normal" if slope < -0.001 else "stable"
            return (
                f"[Ollama offline - Knowledge Base Answer]\n\n"
                f"Current score: {score:.4f} ({status}). Temperature trend: {trend} (slope={slope:.6f}).\n"
                f"Trajectory: {direction}.\n"
                f"{'Recommend immediate action.' if score > 0.5 else 'Continue monitoring.' if score > 0.05 else 'No action required.'}"
            )
        elif any(w in q for w in ["solution","fix","action","do","what should"]):
            return (
                f"[Ollama offline - Knowledge Base Answer]\n\n"
                f"For {d['title']} (Risk: {d['risk']}):\n"
                + "\n".join(d["actions"])
            )
        return (
            f"[Ollama offline - Knowledge Base Answer]\n"
            f"Status: {status} | Score: {score:.4f} | Cause: {lstm_m.get('dominant_cause','?')} | "
            f"Event: {cnn_m.get('event_type','none')} | Trend: {trend}\n"
            f"Diagnosis: {d['title']} — {d['risk']} risk"
        )
    except Exception as e:
        # Timeout or other error — use knowledge base for instant answer
        d      = diagnose_anomaly(current_state)
        score  = lstm_m.get("score", 0)
        status = lstm_m.get("status", "NORMAL")
        trend  = trans_m.get("trend", "STABLE")
        cause  = lstm_m.get("dominant_cause", "unknown")
        q      = question.lower()

        causes_txt  = "\n".join("* " + c for c in d["causes"])
        actions_txt = "\n".join(a for a in d["actions"])
        direction   = ("rising - monitor closely" if trans_m.get("slope",0) > 0.001
                       else "falling - returning to normal" if trans_m.get("slope",0) < -0.001
                       else "stable")
        cnn_event   = current_state.get("models",{}).get("cnn",{}).get("event_type","none")

        if any(w in q for w in ["why","cause","reason","explain","anomaly"]):
            lines = [
                "[Knowledge Base answer - Ollama timed out]",
                "",
                "ANOMALY TYPE: " + str(d["title"]),
                "RISK LEVEL: "   + str(d["risk"]),
                "",
                "MOST LIKELY CAUSES:",
                causes_txt,
                "",
                "RECOMMENDED ACTIONS:",
                actions_txt,
            ]
            return "\n".join(lines)

        elif any(w in q for w in ["solution","fix","action","do","should","what now"]):
            lines = [
                "[Knowledge Base answer - Ollama timed out]",
                "",
                "For: " + str(d["title"]) + " (Risk: " + str(d["risk"]) + ")",
                "",
                actions_txt,
            ]
            return "\n".join(lines)

        elif any(w in q for w in ["future","predict","forecast","next","will"]):
            xtemp_val = safe(sensors, "xTemp")
            lines = [
                "[Knowledge Base answer - Ollama timed out]",
                "",
                "Score: " + f"{score:.4f}" + " (" + status + ") - trend: " + direction,
                "Temperature: " + f"{xtemp_val:.2f}" + "C (" + trend + ")",
                "CNN event: " + cnn_event,
                ("WARNING: Score elevated - anomaly may persist." if score > 0.05
                 else "Score normal - no anomaly expected."),
            ]
            return "\n".join(lines)

        lines = [
            "[Knowledge Base answer - Ollama timed out]",
            "",
            "Status: " + status + " | Score: " + f"{score:.4f}" + " | Cause: " + cause,
            "Diagnosis: " + str(d["title"]) + " - Risk: " + str(d["risk"]),
            "",
            "Tip: Click the diagnose anomaly button for full instant analysis.",
        ]
        return "\n".join(lines)


def _ollama_models() -> list:
    """Return list of locally available Ollama model names (full names preserved)."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = r.json().get("models", [])
        # Return BOTH full name and base name so detection works regardless of tag
        names = []
        for m in models:
            full = m["name"]           # e.g. "llava:latest"
            base = full.split(":")[0]  # e.g. "llava"
            names.append(full)
            if base not in names:
                names.append(base)
        print(f"[OLLAMA] Available models: {names}")
        return names
    except Exception as e:
        print(f"[OLLAMA] Could not list models: {e}")
        return []


def _ollama_generate(model: str, prompt: str, images: list = None,
                     timeout: int = 90) -> str:
    """
    Call Ollama /api/generate with stream=False.
    Handles edge-cases:
      - response key missing (model loaded but returned empty)
      - streamed response accidentally returned line-by-line JSON
      - HTTP error codes
    Returns the text or raises an exception.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if images:
        payload["images"] = images

    r = requests.post("http://localhost:11434/api/generate",
                      json=payload, timeout=timeout)
    r.raise_for_status()

    # Ollama sometimes returns a stream of JSON lines even with stream=False
    # if the client or model version doesn't honour it.
    text = r.text.strip()
    if text.startswith("{"):
        try:
            data = r.json()
            # Normal single-object response
            response = data.get("response", "")
            if response:
                return response
            # Check for error field
            if data.get("error"):
                raise RuntimeError(f"Ollama model error: {data['error']}")
            # Empty response — model loaded but produced nothing
            if "done" in data:
                return "[Model returned empty response — try a different prompt or model]"
        except ValueError:
            pass

    # Streamed newline-delimited JSON — concatenate all response fragments
    fragments = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = __import__("json").loads(line)
            fragments.append(obj.get("response", ""))
            if obj.get("done"):
                break
        except Exception:
            pass
    if fragments:
        return "".join(fragments)

    return "[Could not parse Ollama response]"


def explain_image(image_b64: str, image_type: str, context: dict) -> str:
    """
    Send an image to the best available Ollama vision model.
    Priority: llava → llava-phi3 → moondream → bakllava
    Falls back to llama3 text analysis if no vision model is available.
    Always produces a useful response rather than "No response."
    """
    vision_prompt = f"""You are an expert spacecraft mission AI analyst for the Chandrayaan-3 ILSA seismometer on the lunar surface.

The operator has uploaded an image categorised as: {image_type}

Current live mission telemetry context:
- Anomaly score: {context.get('anomaly_score', 0):.4f}  (warn > 0.05, critical > 0.20)
- Mission status: {context.get('status', 'UNKNOWN')}
- LSTM status: {context.get('lstm_status', 'UNKNOWN')}
- STA/LTA seismic trigger: {'ACTIVE — seismic event in progress' if context.get('stalta_triggered') else 'idle'}
- CNN event type: {context.get('cnn_event_type', 'none')}
- CNN vibration pattern: {context.get('cnn_pattern', 'NORMAL')}
- Transformer temp trend: {context.get('trans_trend', 'STABLE')}
- GNN issues: {context.get('gnn_issues', [])}

Please provide a structured analysis:

1. DESCRIPTION: Describe exactly what this image shows. Be specific about visual features, patterns, scale, and content.

2. ANOMALIES: Identify any unusual features, patterns, artifacts, or concerns visible in the image. For seismograms, describe waveform characteristics. For lunar surface images, describe terrain features, shadows, or disturbances. For sensor charts, describe trends, spikes, or patterns.

3. CORRELATION: How does what you see in this image correlate with the current live telemetry data provided above? Connect visual evidence to sensor readings.

4. RECOMMENDATION: Based on the image and the telemetry context, what specific action should the mission operator take next?

Be technical, specific, and reference actual values from both the image and the telemetry context.
"""

    # ── Try available vision models in priority order ──────────────
    available = _ollama_models()
    print(f"[IMG] Available Ollama models: {available}")

    vision_candidates = ["llava", "llava-phi3", "moondream", "bakllava", "llava:13b", "llava:7b"]
    used_vision = None
    for candidate in vision_candidates:
        if any(candidate in m for m in available):
            used_vision = candidate
            break

    if used_vision:
        print(f"[IMG] Using vision model: {used_vision}")
        try:
            result = _ollama_generate(used_vision, vision_prompt,
                                      images=[image_b64], timeout=180)
            print(f"[IMG] llava result (first 100 chars): {result[:100]!r}")
            if result and not result.startswith("["):
                return f"[Vision model: {used_vision}]\n\n{result}"
            # Vision returned empty — still return it rather than NO_VISION
            if result:
                return f"[Vision model: {used_vision}]\n\n{result}"
            print(f"[IMG] {used_vision} returned empty — falling back")
        except Exception as e:
            print(f"[IMG] {used_vision} FAILED: {type(e).__name__}: {e}")
            # llava failed but IS installed — give a partial response
            return (f"[Vision model: {used_vision} — error: {type(e).__name__}]\n\n"
                    f"llava is installed but returned an error. Try uploading again.\n\n"
                    f"LIVE TELEMETRY:\n"
                    f"  Score: {context.get('anomaly_score',0):.4f} ({context.get('status','?')})\n"
                    f"  CNN event: {context.get('cnn_event_type','?')}\n"
                    f"  STA/LTA: {'TRIGGERED' if context.get('stalta_triggered') else 'idle'}")
    else:
        print(f"[IMG] No vision model found in: {available}")
        print(f"[IMG] Checked candidates: {vision_candidates}")

    # ── No vision model: be HONEST, never hallucinate image content ──
    # llama3 cannot see images. Do NOT ask it to describe or guess the image.
    score     = context.get("anomaly_score", 0)
    status    = context.get("status", "UNKNOWN")
    triggered = context.get("stalta_triggered", False)
    cnn_event = context.get("cnn_event_type", "none")
    trend     = context.get("trans_trend", "STABLE")
    gnn_issues= context.get("gnn_issues", [])
    print(f"[IMG] No vision model — honest NO_VISION response")
    return (
        f"NO_VISION_MODEL\n\n"
        f"I cannot see or analyze this image — no vision model is installed.\n"
        f"To enable real image analysis run:\n"
        f"  ollama pull llava\n\n"
        f"LIVE TELEMETRY (independent of image content):\n"
        f"  Anomaly score : {score:.4f}  ({status})\n"
        f"  STA/LTA       : {'TRIGGERED' if triggered else 'idle'}\n"
        f"  CNN event     : {cnn_event}\n"
        f"  Temp trend    : {trend}\n"
        f"  GNN issues    : {gnn_issues if gnn_issues else 'none'}"
    )


def generate_narrative(state: dict, hist_data: list) -> str:
    m     = state.get("models", {})
    lstm  = m.get("lstm",  {})
    cnn_m = m.get("cnn",   {})
    trans = m.get("transformer", {})
    gnn_m = m.get("gnn",  {})

    scores    = [d["score"] for d in hist_data] if hist_data else [0]
    max_sc    = max(scores); avg_sc = float(np.mean(scores))
    crit_n    = sum(1 for s in scores if s > CRIT_THRESHOLD)
    warn_n    = sum(1 for s in scores if WARN_THRESHOLD < s <= CRIT_THRESHOLD)
    n_events  = len([e for e in event_log if e.get("type") != "heartbeat"])

    prompt = f"""You are a senior spacecraft mission analyst writing an official mission health report for Chandrayaan-3 ILSA.

SESSION STATISTICS:
- Packets: {len(hist_data)} | Max score: {max_sc:.4f} | Avg score: {avg_sc:.4f}
- Critical events: {crit_n} | Warning events: {warn_n} | Seismic events detected: {n_events}

LSTM: score={lstm.get('score',0):.4f}, stalta_ratio={lstm.get('stalta_ratio',0):.3f}, triggered={lstm.get('stalta_triggered',False)}, dominant_cause={lstm.get('dominant_cause','—')}
CNN: event_type={cnn_m.get('event_type','—')}, pattern={cnn_m.get('pattern','—')}, freq_dominant={cnn_m.get('freq_dominant',0):.3f} Hz, confidence={cnn_m.get('confidence',0):.2f}
Transformer: trend={trans.get('trend','—')}, pred_xTemp={trans.get('pred_xTemp',0):.3f}, slope={trans.get('slope',0):.6f}
GNN: seismic_thermal_coupling={gnn_m.get('seismic_thermal_coupling',False)}, issues={gnn_m.get('issues',[])}, seismic_score={gnn_m.get('subsystem_scores',{}).get('seismic',1):.2f}

Write a professional 4-paragraph mission health narrative:
1. Executive summary (include seismic event count and STA/LTA status)
2. Seismic analysis — CNN event classification, frequency content, moonquake type assessment
3. Thermal + structural analysis (LSTM + Transformer + GNN coupling)
4. Recommendations

Use formal technical language. Reference the STA/LTA algorithm, Chandrayaan-3 ILSA instrument, and lunar seismology context.
"""
    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=60)
        return r.json().get("response", "")
    except Exception:
        status_w = "nominal" if avg_sc < WARN_THRESHOLD else "elevated" if avg_sc < CRIT_THRESHOLD else "critical"
        return f"""EXECUTIVE SUMMARY
The Chandrayaan-3 ILSA health monitoring system processed {len(hist_data)} telemetry packets. Mission status is {status_w.upper()} (avg anomaly score {avg_sc:.4f}). {n_events} seismic events were detected using the STA/LTA algorithm (trigger threshold {STALTA_TRIGGER}).

SEISMIC ANALYSIS
The CNN seismic classifier reports event type '{cnn_m.get('event_type','unknown')}' with confidence {cnn_m.get('confidence',0):.2f}. Dominant frequency: {cnn_m.get('freq_dominant',0):.3f} Hz. Current STA/LTA ratio: {lstm.get('stalta_ratio',0):.3f} (trigger: {STALTA_TRIGGER}, detrigger: {STALTA_DETRIG}). Pattern: {cnn_m.get('pattern','NORMAL')}.

THERMAL AND STRUCTURAL ANALYSIS
LSTM combined score {lstm.get('score',0):.4f} (thermal: {lstm.get('thermal_score',0):.4f}, seismic: {lstm.get('seismic_score',0):.4f}). Dominant cause: {lstm.get('dominant_cause','—')}. Transformer predicts xTemp={trans.get('pred_xTemp',0):.3f} ({trans.get('trend','STABLE')} trend, slope={trans.get('slope',0):.6f}). GNN seismic-thermal coupling: {'ACTIVE' if gnn_m.get('seismic_thermal_coupling') else 'NOT DETECTED'}.

RECOMMENDATIONS
{'IMMEDIATE: STA/LTA triggered — verify seismic event origin and instrument health.' if lstm.get('stalta_triggered') else 'No active seismic trigger. Continue standard ILSA monitoring protocol. Review event log for historical detections.'}
(Ollama unavailable — auto-generated narrative)"""


# ================================================================
# MISSION STATE
# ================================================================
current_state = {
    "sensors":       {},
    "anomaly_score": 0.0,
    "status":        "NOMINAL",
    "models":        {"lstm": {}, "cnn": {}, "transformer": {}, "gnn": {}},
    "packet_count":  0,
    "file_info":     "",
}


# ================================================================
# REAL-TIME ENGINE
# ================================================================
async def telemetry_loop():
    global current_state
    while True:
        packet = next_packet()
        history.append(packet)

        # Count valid packets for model readiness indicator
        valid_count = sum(1 for h in history if _is_valid_packet(h))
        current_state["valid_packets"] = valid_count
        current_state["model_ready"]   = valid_count >= 128

        lstm_out  = lstm.run(packet)
        cnn_out   = cnn.run(packet)
        trans_out = transformer.run(packet)
        gnn_out   = gnn.run(packet)

        score  = lstm_out["score"]
        status = lstm_out["status"]
        t      = current_state["packet_count"] + 1

        current_state.update({
            "sensors":       packet,
            "anomaly_score": score,
            "status":        status,
            "packet_count":  t,
            "file_info":     csv_files[file_index - 1] if file_index > 0 else "",
            "models":        {"lstm": lstm_out, "cnn": cnn_out, "transformer": trans_out, "gnn": gnn_out},
        })

        # ── Score history (for charts + history endpoint) ──
        entry = {
            "t":         t,
            "ts":        datetime.now(timezone.utc).isoformat(),
            "score":     score,
            "status":    status,
            "xTemp":     safe(packet, "xTemp"),
            "yTemp":     safe(packet, "yTemp"),
            "zTemp":     safe(packet, "zTemp"),
            "fine_x":    safe(packet, "X Fine Acceleration"),
            "fine_y":    safe(packet, "Y Fine Acceleration"),
            "fine_z":    safe(packet, "Z Fine Acceleration"),
            "coarse_x":  safe(packet, "X Coarse Acceleration"),
            "coarse_y":  safe(packet, "Y Coarse Acceleration"),
            "coarse_z":  safe(packet, "Z Coarse Acceleration"),
            "cnn_rms":   cnn_out.get("rms_accel", 0),
            "pred_temp": trans_out.get("pred_xTemp", 0),
            "stalta":    lstm_out.get("stalta_ratio", 0),
            "event_type": cnn_out.get("event_type", "none"),
        }
        score_history.append(entry)

        # ── Log seismic events ──
        if lstm_out.get("event_detected"):
            ev = {
                "type":       "seismic_onset",
                "t":          t,
                "ts":         entry["ts"],
                "stalta":     lstm_out.get("stalta_ratio", 0),
                "event_type": cnn_out.get("event_type", "unknown"),
                "confidence": cnn_out.get("confidence", 0),
                "freq_hz":    cnn_out.get("freq_dominant", 0),
                "score":      score,
                "sensors":    {k: round(float(v), 6) for k, v in packet.items()
                               if isinstance(v, (int, float))},
            }
            event_log.append(ev)
            print(f"[SEISMIC] Event detected at t={t}: {cnn_out.get('event_type')} "
                  f"stalta={lstm_out.get('stalta_ratio',0):.2f}")

        # ── Session snapshot for /history ──
        session_snapshots.append(copy.deepcopy(current_state))

        await asyncio.sleep(TELEMETRY_INTERVAL)


# ================================================================
# CHART BUILDERS
# ================================================================
DARK_BG = "#070f1a"; DARK_AX = "#0a1520"
C_CYAN="#00b4ff"; C_GREEN="#00ff88"; C_AMBER="#ffaa00"
C_RED="#ff3355"; C_PURPLE="#aa44ff"; C_LABEL="#4a7a99"; C_GRID="#0d2035"

def style_ax(ax, title=""):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=C_LABEL, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)
    ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.8)
    if title: ax.set_title(title, color=C_CYAN, fontsize=8, fontweight="bold", pad=5)


def make_chart_temperature(data):
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=DARK_BG)
    if data:
        t = [d["t"] for d in data]
        ax.plot(t, [d["xTemp"]    for d in data], color=C_CYAN,   lw=1.2, label="xTemp")
        ax.plot(t, [d["yTemp"]    for d in data], color=C_GREEN,  lw=1.2, label="yTemp")
        ax.plot(t, [d["zTemp"]    for d in data], color=C_AMBER,  lw=1.2, label="zTemp")
        ax.plot(t, [d["pred_temp"]for d in data], color=C_PURPLE, lw=0.8, ls="--", label="pred_xTemp")
    style_ax(ax, "TEMPERATURE SENSORS — xTemp / yTemp / zTemp + Transformer Prediction")
    ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white", edgecolor=C_GRID)
    plt.tight_layout(pad=0.4); buf=io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
    return buf.getvalue()


def make_chart_acceleration(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), facecolor=DARK_BG)
    if data:
        t = [d["t"] for d in data]
        ax1.plot(t, [d["fine_x"]  for d in data], color=C_CYAN,   lw=1,   label="Fine X")
        ax1.plot(t, [d["fine_y"]  for d in data], color=C_GREEN,  lw=1,   label="Fine Y")
        ax1.plot(t, [d["fine_z"]  for d in data], color=C_AMBER,  lw=1.3, label="Fine Z (key)")
        ax2.plot(t, [d["coarse_x"]for d in data], color=C_PURPLE, lw=1,   label="Coarse X")
        ax2.plot(t, [d["coarse_y"]for d in data], color="#ff6644", lw=1,   label="Coarse Y")
        ax2.plot(t, [d["coarse_z"]for d in data], color="#44ffcc", lw=1,   label="Coarse Z")
        ax2.plot(t, [d["cnn_rms"] for d in data], color=C_RED,    lw=1.5, ls="--", label="CNN RMS")
    style_ax(ax1, "FINE ACCELERATION (3-Component)"); style_ax(ax2, "COARSE ACCELERATION + CNN RMS")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white", edgecolor=C_GRID)
    plt.tight_layout(pad=0.4); buf=io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
    return buf.getvalue()


def make_chart_anomaly_score(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), facecolor=DARK_BG,
                                    gridspec_kw={"height_ratios": [2, 1]})
    if data:
        t  = [d["t"]      for d in data]
        sc = [d["score"]  for d in data]
        st = [d["stalta"] for d in data]
        ax1.fill_between(t, sc, alpha=0.2, color=C_RED)
        ax1.plot(t, sc, color=C_RED, lw=1.5, label="LSTM Anomaly Score")
        ax1.axhline(WARN_THRESHOLD, color=C_AMBER, lw=1, ls="--", label=f"Warn ({WARN_THRESHOLD})")
        ax1.axhline(CRIT_THRESHOLD, color=C_RED,   lw=1, ls=":",  label=f"Crit ({CRIT_THRESHOLD})")
        ax1.set_ylim(0, max(max(sc)*1.2, 0.3))
        # Mark seismic events
        for ev in event_log:
            if ev["t"] <= t[-1]:
                ax1.axvline(ev["t"], color=C_PURPLE, lw=0.8, alpha=0.6)
        ax2.fill_between(t, st, alpha=0.3, color=C_PURPLE)
        ax2.plot(t, st, color=C_PURPLE, lw=1.2, label="STA/LTA Ratio")
        ax2.axhline(STALTA_TRIGGER, color=C_AMBER, lw=1, ls="--", label=f"Trigger ({STALTA_TRIGGER})")
        ax2.set_ylim(0, max(max(st)*1.2, 5) if st else 5)
    style_ax(ax1, "LSTM ANOMALY SCORE + SEISMIC EVENT MARKERS (purple)")
    style_ax(ax2, "STA/LTA RATIO (seismic onset detector)")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white", edgecolor=C_GRID)
    plt.tight_layout(pad=0.4); buf=io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
    return buf.getvalue()


def make_chart_model_comparison(state):
    m = state.get("models", {})
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), facecolor=DARK_BG)
    panels = [("LSTM", m.get("lstm",{}), C_CYAN), ("CNN", m.get("cnn",{}), C_GREEN),
              ("TRANSFORMER", m.get("transformer",{}), C_AMBER), ("GNN", m.get("gnn",{}), C_PURPLE)]
    for ax, (name, out, color) in zip(axes, panels):
        ax.set_facecolor(DARK_AX); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.add_patch(plt.Rectangle((0,0.82),1,0.18,color=color,alpha=0.25))
        ax.text(0.5,0.91,name,ha="center",va="center",color=color,fontsize=9,fontweight="bold",fontfamily="monospace")
        if name == "LSTM":
            lines=[f"Score: {out.get('score',0):.4f}",f"Status: {out.get('status','—')}",
                   f"STA/LTA: {out.get('stalta_ratio',0):.3f}",f"Cause: {out.get('dominant_cause','—')}"]
        elif name == "CNN":
            lines=[f"Type: {out.get('event_type','—')}",f"Pattern: {out.get('pattern','NORMAL')[:18]}",
                   f"Freq: {out.get('freq_dominant',0):.3f} Hz",f"RMS: {out.get('rms_accel',0):.4f}"]
        elif name == "TRANSFORMER":
            lines=[f"Trend: {out.get('trend','—')}",f"pred_xTemp: {out.get('pred_xTemp',0):.3f}",
                   f"Slope: {out.get('slope',0):.5f}",f"Cur: {out.get('current_xTemp',0):.3f}"]
        else:
            sc2=out.get("subsystem_scores",{})
            lines=[f"Seismic: {sc2.get('seismic',1):.2f}",f"Thermal: {sc2.get('thermal',1):.2f}",
                   f"Struct: {sc2.get('structural',1):.2f}",f"Coupling: {out.get('seismic_thermal_coupling',False)}"]
        for i, line in enumerate(lines):
            ax.text(0.05, 0.72-i*0.17, line, ha="left", va="top", color="white", fontsize=7.5, fontfamily="monospace")
        for sp in ax.spines.values(): sp.set_edgecolor(color); sp.set_linewidth(1.5)
        ax.set_visible(True)
    plt.suptitle("ALL 4 MODEL OUTPUTS — SNAPSHOT", color=C_CYAN, fontsize=9, fontfamily="monospace", y=1.01)
    plt.tight_layout(pad=0.4); buf=io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
    return buf.getvalue()


def make_chart_seismic_events(events):
    """New: chart showing all detected seismic events with type and STA/LTA."""
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=DARK_BG)
    if events:
        type_colors = {
            "deep_moonquake":    C_PURPLE,
            "shallow_moonquake": C_RED,
            "thermal_cracking":  C_AMBER,
            "meteorite_impact":  C_GREEN,
            "insufficient_data": C_LABEL,
            "unknown":           C_LABEL,
        }
        for ev in events:
            color = type_colors.get(ev.get("event_type","unknown"), C_LABEL)
            ax.scatter(ev["t"], ev["stalta"], color=color, s=60, zorder=5, alpha=0.9)
            ax.annotate(ev.get("event_type","?")[:12],
                        (ev["t"], ev["stalta"]), textcoords="offset points",
                        xytext=(0, 6), fontsize=5, color=color, ha="center")
        # Legend
        for et, col in type_colors.items():
            ax.scatter([], [], color=col, s=30, label=et.replace("_"," "))
    ax.axhline(STALTA_TRIGGER, color=C_AMBER, lw=0.8, ls="--", label="trigger")
    style_ax(ax, f"DETECTED SEISMIC EVENTS ({len(events)} total) — Type / STA/LTA at onset")
    ax.set_xlabel("Packet #", fontsize=7); ax.set_ylabel("STA/LTA at onset", fontsize=7)
    ax.legend(fontsize=6, facecolor=DARK_BG, labelcolor="white", edgecolor=C_GRID,
              loc="upper right", ncol=2)
    plt.tight_layout(pad=0.4); buf=io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
    return buf.getvalue()


def make_analyzed_image_figure(img_record: dict):
    """Render an analyzed image + its AI analysis text as a matplotlib figure."""
    img_data = base64.b64decode(img_record["image_b64"])
    # Use PIL instead of plt.imread — handles JPEG, PNG, WebP, any format
    from PIL import Image as _PILImg
    import io as _io2
    pil_img  = _PILImg.open(_io2.BytesIO(img_data)).convert("RGB")
    img_arr  = np.array(pil_img)

    fig = plt.figure(figsize=(10, 4), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1], figure=fig)

    ax_img  = fig.add_subplot(gs[0])
    ax_txt  = fig.add_subplot(gs[1])

    ax_img.imshow(img_arr)
    ax_img.axis("off")
    ax_img.set_title(f"{img_record['image_type'].upper()} — {img_record['filename']}",
                     color=C_CYAN, fontsize=8, fontfamily="monospace", pad=4)

    ax_txt.set_facecolor(DARK_AX); ax_txt.axis("off")
    ax_txt.set_xlim(0,1); ax_txt.set_ylim(0,1)

    text = img_record.get("analysis", "No analysis")
    # Truncate for display
    lines = text.replace("\r","").split("\n")
    y = 0.97
    for line in lines[:20]:
        wrapped = line[:65]
        ax_txt.text(0.02, y, wrapped, va="top", ha="left",
                    color="white" if not line.startswith(("1.","2.","3.","4.")) else C_CYAN,
                    fontsize=6.5, fontfamily="monospace", transform=ax_txt.transAxes)
        y -= 0.048
        if y < 0.02: break

    ax_txt.text(0.02, 0.02, f"ts: {img_record.get('ts','—')[:19]}",
                va="bottom", ha="left", color=C_LABEL, fontsize=6, fontfamily="monospace")

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


# ================================================================
# REPORT BUILDERS
# ================================================================

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def build_html_report(state, hist_data, narrative, chart_temp, chart_accel,
                      chart_score, chart_models, chart_events, analyzed_img_charts):
    now    = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    m      = state.get("models", {})
    lstm   = m.get("lstm",  {})
    cnn_m  = m.get("cnn",   {})
    trans  = m.get("transformer", {})
    gnn_m  = m.get("gnn",  {})
    score  = state.get("anomaly_score", 0)
    status = state.get("status", "UNKNOWN")
    sc     = "#ff3355" if status=="ANOMALY" else "#ffaa00" if status=="WARNING" else "#00ff88"

    scores = [d["score"] for d in hist_data] if hist_data else [0]
    max_sc = max(scores); avg_sc = float(np.mean(scores))
    crit_n = sum(1 for s in scores if s > CRIT_THRESHOLD)
    warn_n = sum(1 for s in scores if WARN_THRESHOLD < s <= CRIT_THRESHOLD)
    n_ev   = len(event_log)

    paras  = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    para_html = "".join(f'<p style="margin:0 0 14px;line-height:1.7">{p}</p>' for p in paras)

    # Analyzed images section
    img_html = ""
    if analyzed_img_charts:
        img_html = '<div class="section"><div class="section-title">ANALYZED IMAGES + AI VISION CONTEXT</div>'
        for i, (chart_bytes, rec) in enumerate(zip(analyzed_img_charts, analyzed_images)):
            img_html += f'''
            <div style="background:var(--panel2);border:1px solid var(--b);border-radius:4px;padding:14px;margin-bottom:14px">
              <div style="font-family:'Share Tech Mono',monospace;font-size:8px;color:var(--cyan);margin-bottom:8px">
                IMAGE {i+1} — {rec["image_type"].upper()} — {rec["filename"]} — {rec.get("ts","")[:19]}
              </div>
              <img src="data:image/png;base64,{_b64(chart_bytes)}" style="width:100%;border-radius:3px;margin-bottom:8px"/>
              <div style="font-family:'Share Tech Mono',monospace;font-size:8px;line-height:1.6;color:#c8e8ff;background:rgba(0,0,0,.3);padding:10px;border-radius:3px;white-space:pre-wrap">{rec["analysis"][:800]}</div>
            </div>'''
        img_html += '</div>'

    # Event log table
    ev_rows = ""
    for ev in event_log[-20:]:
        ev_rows += f'<tr><td>{ev["t"]}</td><td>{ev["ts"][:19]}</td><td style="color:var(--purple)">{ev.get("event_type","—")}</td><td>{ev.get("stalta",0):.3f}</td><td>{ev.get("confidence",0):.2f}</td><td>{ev.get("freq_hz",0):.3f} Hz</td></tr>'

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Mission Health Report — {now}</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400&display=swap" rel="stylesheet">
<style>
:root{{--bg:#020508;--panel:#070f1a;--panel2:#0a1520;--cyan:#00b4ff;--green:#00ff88;--amber:#ffaa00;--red:#ff3355;--purple:#aa44ff;--tp:#c8e8ff;--ts:#4a7a99;--tl:#2a5a77;--b:rgba(0,180,255,0.15)}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--tp);font-family:'Exo 2',sans-serif;padding:40px;min-height:100vh}}
.rw{{max-width:1100px;margin:0 auto}}
.rh{{border-bottom:2px solid var(--cyan);padding-bottom:20px;margin-bottom:28px;display:flex;justify-content:space-between;align-items:flex-end}}
.rh h1{{font-family:'Orbitron',sans-serif;font-size:20px;font-weight:900;color:var(--cyan);letter-spacing:4px;text-shadow:0 0 20px rgba(0,180,255,.4)}}
.rh .sub{{font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--ts);letter-spacing:3px;margin-top:5px}}
.rh-r{{text-align:right;font-family:'Share Tech Mono',monospace;font-size:8.5px;color:var(--tl);line-height:1.9}}
.sbar{{background:var(--panel2);border:1px solid var(--b);border-left:4px solid {sc};border-radius:4px;padding:14px 20px;margin-bottom:22px;display:flex;align-items:center;gap:18px}}
.sdot{{width:12px;height:12px;border-radius:50%;background:{sc};box-shadow:0 0 10px {sc};flex-shrink:0}}
.stxt{{font-family:'Orbitron',sans-serif;font-size:12px;font-weight:700;color:{sc};letter-spacing:3px}}
.ssc{{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--ts);margin-left:auto}}
.section{{margin-bottom:26px}}
.section-title{{font-family:'Orbitron',sans-serif;font-size:9px;font-weight:700;color:var(--cyan);letter-spacing:4px;margin-bottom:12px;padding-bottom:5px;border-bottom:1px solid var(--b);display:flex;align-items:center;gap:7px}}
.section-title::before{{content:'';display:inline-block;width:4px;height:13px;background:var(--cyan);border-radius:2px}}
.sgrid{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-bottom:18px}}
.scard{{background:var(--panel2);border:1px solid var(--b);border-radius:4px;padding:13px;text-align:center}}
.slbl{{font-family:'Share Tech Mono',monospace;font-size:7.5px;color:var(--tl);letter-spacing:2px;margin-bottom:5px}}
.sval{{font-family:'Orbitron',sans-serif;font-size:16px;font-weight:700}}
.ssub{{font-family:'Share Tech Mono',monospace;font-size:7px;color:var(--tl);margin-top:3px}}
.cbox{{background:var(--panel2);border:1px solid var(--b);border-radius:4px;padding:2px;margin-bottom:12px}}
.cbox img{{width:100%;display:block;border-radius:3px}}
.mtbl{{width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:8.5px}}
.mtbl th{{background:var(--panel2);color:var(--tl);letter-spacing:2px;padding:7px 10px;border-bottom:1px solid var(--b);text-align:left;font-weight:normal}}
.mtbl td{{padding:7px 10px;border-bottom:1px solid rgba(0,180,255,.05);color:var(--tp)}}
.mtbl tr:hover td{{background:rgba(0,180,255,.03)}}
.bdg{{display:inline-block;padding:2px 7px;border-radius:2px;font-size:7.5px;letter-spacing:1px}}
.bok{{background:rgba(0,255,136,.15);color:#00ff88}}.bwn{{background:rgba(255,170,0,.15);color:#ffaa00}}.bcr{{background:rgba(255,51,85,.15);color:#ff3355}}
.narr{{background:var(--panel2);border:1px solid var(--b);border-radius:4px;padding:22px;font-size:12.5px;line-height:1.7}}
.foot{{border-top:1px solid var(--b);padding-top:14px;margin-top:28px;display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:7.5px;color:var(--tl)}}
</style></head><body><div class="rw">
<div class="rh"><div><h1>SPACECRAFT HEALTH AI</h1><div class="sub">CHANDRAYAAN-3 · ILSA SEISMOMETER · MISSION HEALTH REPORT</div></div>
<div class="rh-r"><div>Generated: {now}</div><div>Engine v2.0 · STA/LTA + CNN Lunar Quake Classifier</div><div>Packets: {len(hist_data)} · Seismic Events: {n_ev}</div></div></div>
<div class="sbar"><div class="sdot"></div><div class="stxt">MISSION STATUS: {status}</div>
<div class="ssc">Score: {score:.4f} | Max: {max_sc:.4f} | Avg: {avg_sc:.4f} | Criticals: {crit_n} | Warnings: {warn_n} | Events: {n_ev}</div></div>
<div class="section"><div class="section-title">SESSION STATISTICS</div>
<div class="sgrid">
<div class="scard"><div class="slbl">ANOMALY SCORE</div><div class="sval" style="color:{sc}">{score:.4f}</div><div class="ssub">{status}</div></div>
<div class="scard"><div class="slbl">STA/LTA RATIO</div><div class="sval" style="color:var(--purple)">{lstm.get('stalta_ratio',0):.3f}</div><div class="ssub">trigger: {STALTA_TRIGGER}</div></div>
<div class="scard"><div class="slbl">CNN EVENT TYPE</div><div class="sval" style="color:var(--green);font-size:10px">{cnn_m.get('event_type','—').upper()}</div><div class="ssub">conf: {cnn_m.get('confidence',0):.2f}</div></div>
<div class="scard"><div class="slbl">TEMP TREND</div><div class="sval" style="color:var(--amber);font-size:11px">{trans.get('trend','—')}</div><div class="ssub">pred: {trans.get('pred_xTemp',0):.3f}</div></div>
</div></div>
<div class="section"><div class="section-title">ANOMALY SCORE + STA/LTA HISTORY</div><div class="cbox"><img src="data:image/png;base64,{_b64(chart_score)}"/></div></div>
<div class="section"><div class="section-title">SEISMIC EVENTS DETECTED</div><div class="cbox"><img src="data:image/png;base64,{_b64(chart_events)}"/></div>
<table class="mtbl" style="margin-top:10px"><thead><tr><th>PKT</th><th>TIMESTAMP</th><th>EVENT TYPE</th><th>STA/LTA</th><th>CONFIDENCE</th><th>FREQ</th></tr></thead><tbody>{ev_rows if ev_rows else '<tr><td colspan="6" style="color:var(--tl);text-align:center">No seismic events detected in this session</td></tr>'}</tbody></table></div>
<div class="section"><div class="section-title">TEMPERATURE SENSORS</div><div class="cbox"><img src="data:image/png;base64,{_b64(chart_temp)}"/></div></div>
<div class="section"><div class="section-title">ACCELERATION SENSORS</div><div class="cbox"><img src="data:image/png;base64,{_b64(chart_accel)}"/></div></div>
<div class="section"><div class="section-title">ALL 4 MODEL OUTPUTS</div><div class="cbox"><img src="data:image/png;base64,{_b64(chart_models)}"/></div>
<table class="mtbl" style="margin-top:10px"><thead><tr><th>MODEL</th><th>KEY METRIC</th><th>VALUE</th><th>STATUS</th></tr></thead><tbody>
<tr><td style="color:var(--cyan)">LSTM</td><td>Score / STA-LTA / Cause</td><td>{lstm.get('score',0):.4f} / {lstm.get('stalta_ratio',0):.3f} / {lstm.get('dominant_cause','—')}</td><td><span class="bdg {'bcr' if lstm.get('status')=='ANOMALY' else 'bwn' if lstm.get('status')=='WARNING' else 'bok'}">{lstm.get('status','—')}</span></td></tr>
<tr><td style="color:var(--green)">CNN</td><td>Event Type / Freq / RMS</td><td>{cnn_m.get('event_type','—')} / {cnn_m.get('freq_dominant',0):.3f} Hz / {cnn_m.get('rms_accel',0):.4f}</td><td><span class="bdg {'bcr' if cnn_m.get('severity')=='critical' else 'bwn' if cnn_m.get('severity')=='warning' else 'bok'}">{cnn_m.get('severity','—').upper()}</span></td></tr>
<tr><td style="color:var(--amber)">TRANSFORMER</td><td>Trend / Prediction / Slope</td><td>{trans.get('trend','—')} / {trans.get('pred_xTemp',0):.3f} / {trans.get('slope',0):.6f}</td><td><span class="bdg bok">{trans.get('trend','—')}</span></td></tr>
<tr><td style="color:var(--purple)">GNN</td><td>Seismic-Thermal / Issues</td><td>coupling={'YES' if gnn_m.get('seismic_thermal_coupling') else 'NO'} / {', '.join(gnn_m.get('issues',[])) or 'none'}</td><td><span class="bdg {'bcr' if gnn_m.get('cross_anomaly') else 'bwn' if gnn_m.get('issues') else 'bok'}">{('CROSS-ANOM' if gnn_m.get('cross_anomaly') else 'ISSUES' if gnn_m.get('issues') else 'NOMINAL')}</span></td></tr>
</tbody></table></div>
{img_html}
<div class="section"><div class="section-title">AI MISSION HEALTH NARRATIVE</div><div class="narr">{para_html}</div></div>
<div class="foot"><div>Spacecraft Health AI v2.0 · Chandrayaan-3 ILSA · {now}</div><div>STA/LTA detector · CNN lunar quake classifier · llama3</div><div>AUTO-GENERATED — CONFIDENTIAL</div></div>
</div></body></html>"""
    return html


def build_pdf_report(state, hist_data, narrative, chart_temp, chart_accel,
                     chart_score, chart_models, chart_events, analyzed_img_charts):
    buf = io.BytesIO()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    m       = state.get("models", {})
    lstm    = m.get("lstm",  {})
    cnn_m   = m.get("cnn",   {})
    trans   = m.get("transformer", {})
    gnn_m   = m.get("gnn",  {})
    score   = state.get("anomaly_score", 0)
    status  = state.get("status", "UNKNOWN")
    scores  = [d["score"] for d in hist_data] if hist_data else [0]
    max_sc  = max(scores); avg_sc = float(np.mean(scores))
    crit_n  = sum(1 for s in scores if s > CRIT_THRESHOLD)
    warn_n  = sum(1 for s in scores if WARN_THRESHOLD < s <= CRIT_THRESHOLD)
    n_ev    = len(event_log)

    COL = {
        "bg":     colors.HexColor("#020508"),
        "panel":  colors.HexColor("#070f1a"),
        "cyan":   colors.HexColor("#00b4ff"),
        "green":  colors.HexColor("#00ff88"),
        "amber":  colors.HexColor("#ffaa00"),
        "red":    colors.HexColor("#ff3355"),
        "purple": colors.HexColor("#aa44ff"),
        "tp":     colors.HexColor("#c8e8ff"),
        "ts":     colors.HexColor("#4a7a99"),
        "tl":     colors.HexColor("#2a5a77"),
    }
    status_col = COL["red"] if status=="ANOMALY" else COL["amber"] if status=="WARNING" else COL["green"]

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm,
                            title="Spacecraft Health AI — Mission Report")
    S = {
        "title":   ParagraphStyle("t",  fontName="Helvetica-Bold", fontSize=17, textColor=COL["cyan"],   spaceAfter=3,  leading=21),
        "sub":     ParagraphStyle("s",  fontName="Helvetica",      fontSize=8,  textColor=COL["ts"],     spaceAfter=10, letterSpacing=2),
        "section": ParagraphStyle("sc", fontName="Helvetica-Bold", fontSize=9,  textColor=COL["cyan"],   spaceBefore=12, spaceAfter=7),
        "body":    ParagraphStyle("b",  fontName="Helvetica",      fontSize=9,  textColor=COL["tp"],     leading=14,    spaceAfter=5),
        "mono":    ParagraphStyle("m",  fontName="Courier",        fontSize=7.5,textColor=COL["tp"],     leading=11),
        "status":  ParagraphStyle("st", fontName="Helvetica-Bold", fontSize=12, textColor=status_col,    spaceAfter=5),
    }
    def sec(text): return [HRFlowable(width="100%",thickness=1,color=COL["cyan"],spaceAfter=4,spaceBefore=8), Paragraph(f"  {text}", S["section"])]
    def img(data, w=17*cm, h=None): return RLImage(io.BytesIO(data), width=w, height=h or w*0.3)

    story = []
    story += [Paragraph("SPACECRAFT HEALTH AI", S["title"]),
              Paragraph("CHANDRAYAAN-3  ILSA  SEISMOMETER  MISSION HEALTH REPORT  —  STA/LTA + CNN LUNAR QUAKE CLASSIFIER", S["sub"]),
              HRFlowable(width="100%",thickness=2,color=COL["cyan"],spaceAfter=8)]

    meta = [["Generated", now, "Engine", "Spacecraft Health AI v2.0"],
            ["Status", status, "Seismic Events", str(n_ev)],
            ["Packets", str(len(hist_data)), "Max Score", f"{max_sc:.4f}"]]
    mt = Table(meta, colWidths=[2.8*cm,5.2*cm,3.2*cm,5.8*cm])
    mt.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Courier"),("FONTSIZE",(0,0),(-1,-1),8),
        ("TEXTCOLOR",(0,0),(0,-1),COL["tl"]),("TEXTCOLOR",(2,0),(2,-1),COL["tl"]),
        ("TEXTCOLOR",(1,0),(1,-1),COL["tp"]),("TEXTCOLOR",(3,0),(3,-1),COL["tp"]),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[COL["panel"],colors.HexColor("#0a1520")]),
        ("GRID",(0,0),(-1,-1),0.3,COL["tl"]),("PADDING",(0,0),(-1,-1),5)]))
    story += [mt, Spacer(1,8)]
    story.append(Paragraph(f"STATUS: {status}  |  Score: {score:.4f}  |  Max: {max_sc:.4f}  |  Avg: {avg_sc:.4f}  |  Criticals: {crit_n}  |  Warnings: {warn_n}  |  Seismic Events: {n_ev}", S["status"]))
    story.append(HRFlowable(width="100%",thickness=1,color=status_col,spaceAfter=8))

    story += sec("ANOMALY SCORE + STA/LTA HISTORY")
    story.append(img(chart_score, h=6*cm))
    story.append(Spacer(1,6))

    story += sec("SEISMIC EVENTS DETECTED")
    story.append(img(chart_events, h=5*cm))
    story.append(Spacer(1,5))
    if event_log:
        ev_data = [["PKT", "TIMESTAMP", "EVENT TYPE", "STA/LTA", "CONF", "FREQ Hz"]]
        for ev in event_log[-15:]:
            ev_data.append([str(ev["t"]), ev["ts"][:19], ev.get("event_type","—"),
                            f"{ev.get('stalta',0):.3f}", f"{ev.get('confidence',0):.2f}", f"{ev.get('freq_hz',0):.3f}"])
        et = Table(ev_data, colWidths=[1.5*cm,4.5*cm,4.5*cm,2.2*cm,1.8*cm,2*cm])
        et.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Courier"),("FONTSIZE",(0,0),(-1,-1),7.5),
            ("TEXTCOLOR",(0,0),(-1,0),COL["tl"]),("TEXTCOLOR",(2,1),(2,-1),COL["purple"]),
            ("TEXTCOLOR",(0,1),(-1,-1),COL["tp"]),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[COL["panel"],colors.HexColor("#0a1520")]),
            ("GRID",(0,0),(-1,-1),0.3,COL["tl"]),("PADDING",(0,0),(-1,-1),5)]))
        story.append(et)
    story.append(Spacer(1,6))

    story += sec("TEMPERATURE SENSORS")
    story.append(img(chart_temp, h=5.5*cm))
    story += sec("ACCELERATION SENSORS")
    story.append(img(chart_accel, h=7.5*cm))
    story.append(PageBreak())

    story += sec("ALL 4 MODEL OUTPUTS")
    story.append(img(chart_models, h=5*cm))
    story.append(Spacer(1,7))
    tbl = [["MODEL","KEY METRIC","VALUE","STATUS"],
           ["LSTM",     "Score / STA-LTA / Cause",    f"{lstm.get('score',0):.4f} / {lstm.get('stalta_ratio',0):.3f} / {lstm.get('dominant_cause','—')}", lstm.get('status','—')],
           ["CNN",      "Event Type / Freq / RMS",     f"{cnn_m.get('event_type','—')} / {cnn_m.get('freq_dominant',0):.3f}Hz / {cnn_m.get('rms_accel',0):.4f}", cnn_m.get('severity','—').upper()],
           ["TRANSFORM","Trend / pred_xTemp / Slope",  f"{trans.get('trend','—')} / {trans.get('pred_xTemp',0):.3f} / {trans.get('slope',0):.6f}", trans.get('trend','—')],
           ["GNN",      "S-T Coupling / Issues",       f"{'YES' if gnn_m.get('seismic_thermal_coupling') else 'NO'} / {', '.join(gnn_m.get('issues',[])) or 'none'}", "CROSS" if gnn_m.get('cross_anomaly') else "NOMINAL"]]
    mt2 = Table(tbl, colWidths=[3*cm,5.5*cm,6*cm,3*cm])
    mt2.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"Courier"),("FONTSIZE",(0,0),(-1,-1),7.5),
        ("TEXTCOLOR",(0,0),(-1,0),COL["tl"]),("TEXTCOLOR",(0,1),(0,-1),COL["tp"]),
        ("TEXTCOLOR",(0,1),(0,1),COL["cyan"]),("TEXTCOLOR",(0,2),(0,2),COL["green"]),
        ("TEXTCOLOR",(0,3),(0,3),COL["amber"]),("TEXTCOLOR",(0,4),(0,4),COL["purple"]),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[COL["panel"],colors.HexColor("#0a1520")]),
        ("GRID",(0,0),(-1,-1),0.3,COL["tl"]),("PADDING",(0,0),(-1,-1),5)]))
    story += [mt2, Spacer(1,12)]

    # Analyzed images in PDF
    if analyzed_img_charts:
        story += sec(f"ANALYZED IMAGES + AI VISION CONTEXT ({len(analyzed_img_charts)} images)")
        for i, (chart_bytes, rec) in enumerate(zip(analyzed_img_charts, analyzed_images)):
            story.append(Paragraph(f"Image {i+1}: {rec['image_type'].upper()} — {rec['filename']} — {rec.get('ts','')[:19]}", S["mono"]))
            story.append(Spacer(1,4))
            story.append(img(chart_bytes, h=6*cm))
            story.append(Spacer(1,5))
            # AI analysis text (truncated for PDF)
            analysis_paras = [p.strip() for p in rec["analysis"].split("\n\n") if p.strip()]
            for p in analysis_paras[:3]:
                story.append(Paragraph(p[:400], S["body"]))
            story.append(Spacer(1,8))

    story += sec("AI MISSION HEALTH NARRATIVE")
    for para in [p.strip() for p in narrative.split("\n\n") if p.strip()]:
        for line in para.split("\n"):
            if line.strip(): story.append(Paragraph(line, S["body"]))
        story.append(Spacer(1,5))

    story += [Spacer(1,8), HRFlowable(width="100%",thickness=1,color=COL["tl"],spaceAfter=4),
              Paragraph(f"Spacecraft Health AI v2.0  Chandrayaan-3 ILSA  {now}  STA/LTA + CNN Lunar Quake Classifier  AUTO-GENERATED", S["mono"])]
    doc.build(story)
    return buf.getvalue()


# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI(title="Spacecraft Health AI", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def serve_dashboard():
    from fastapi.responses import FileResponse as FR
    resp = FR("dashboard/index.html")
    # Prevent browser caching so updates always load fresh
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


def explain_with_image(question: str, img_ctx: dict) -> str:
    """Answer a question about the last analyzed image using llava."""
    filename  = img_ctx.get("filename", "unknown")
    analysis  = img_ctx.get("analysis", "")
    image_b64 = img_ctx.get("image_b64", "")
    img_type  = img_ctx.get("image_type", "")

    prompt = f"""You are an expert image analyst. An operator uploaded an image called '{filename}' (type: {img_type}).

Previous AI analysis of this image:
{analysis[:600]}

Operator question: {question}

Answer the question directly based on what the image shows.
If the image is not related to spacecraft/mission data, answer based on its actual visual content.
Be specific and direct — if you can identify what's in the image, say so clearly.
"""
    try:
        # Try llava first (can see the image)
        if image_b64:
            result = _ollama_generate("llava", prompt, images=[image_b64], timeout=120)
            if result and not result.startswith("["):
                return result
    except Exception:
        pass

    # Fallback: use existing analysis text to answer
    try:
        text_prompt = f"""Based on this image analysis:
{analysis[:800]}

Answer this question: {question}

Be direct and specific. If the analysis mentions the image content clearly, use that to answer.
"""
        result = _ollama_generate("llama3", text_prompt, timeout=60)
        if result:
            return "[Based on image analysis]\n\n" + result
    except Exception:
        pass

    # Last resort: extract answer from stored analysis
    if analysis:
        return "[From stored image analysis]\n\n" + analysis[:600] + "\n\n(To get a direct answer, ensure Ollama is running.)"

    return "No image analysis available. Upload an image first, then ask about it."


@app.get("/diagnose")
def diagnose():
    """Returns full knowledge base diagnosis for current state."""
    d = diagnose_anomaly(current_state)
    m = current_state.get("models", {})
    return {
        "diagnosis":        d,
        "anomaly_score":    current_state.get("anomaly_score", 0),
        "status":           current_state.get("status", "NORMAL"),
        "dominant_cause":   m.get("lstm", {}).get("dominant_cause", "unknown"),
        "event_type":       m.get("cnn",  {}).get("event_type", "none"),
        "top_sensors":      sorted(
            m.get("lstm", {}).get("sensor_errors", {}).items(),
            key=lambda x: x[1], reverse=True
        )[:3],
    }


@app.get("/ask")
def ask(question: str):
    model  = select_model(question)
    result = model.run(current_state["sensors"])

    # Detect if question is about the last uploaded image
    q_lower = question.lower()
    img_keywords = ["image","photo","picture","dog","person","what is","who is",
                    "describe","see","show","look","name","identify","this image"]
    is_image_question = any(k in q_lower for k in img_keywords) and last_image_context

    if is_image_question and last_image_context:
        # Route to llava for image-aware question
        answer = explain_with_image(question, last_image_context)
        return {"response": answer, "model_used": "llava (image context)",
                "model_output": result, "anomaly_score": current_state["anomaly_score"],
                "status": current_state["status"]}

    answer = explain(result, question, current_state["sensors"])
    return {"response": answer, "model_used": result.get("model"),
            "model_output": result, "anomaly_score": current_state["anomaly_score"],
            "status": current_state["status"]}


@app.get("/state")
def get_state():
    return current_state


# ── /history endpoint ──────────────────────────────────────────
@app.get("/history")
def get_history(
    limit:  int = Query(default=200, ge=1,  le=1000),
    offset: int = Query(default=0,   ge=0),
    format: str = Query(default="timeseries")
):
    """
    Returns session history for dashboard replay and analysis.

    format=timeseries  → compact arrays for charting
    format=events      → seismic event log only
    format=snapshots   → full state snapshots (heavy)
    format=summary     → aggregated stats
    """
    if format == "events":
        return JSONResponse({"events": event_log, "total": len(event_log)})

    if format == "summary":
        scores = [d["score"] for d in score_history]
        ev_types = {}
        for ev in event_log:
            t = ev.get("event_type","unknown")
            ev_types[t] = ev_types.get(t, 0) + 1
        return JSONResponse({
            "total_packets":   len(score_history),
            "session_duration_s": len(score_history),
            "max_score":       max(scores) if scores else 0,
            "min_score":       min(scores) if scores else 0,
            "avg_score":       float(np.mean(scores)) if scores else 0,
            "critical_count":  sum(1 for s in scores if s > CRIT_THRESHOLD),
            "warning_count":   sum(1 for s in scores if WARN_THRESHOLD < s <= CRIT_THRESHOLD),
            "seismic_events":  len(event_log),
            "event_types":     ev_types,
            "stalta_trigger_count": sum(1 for e in event_log),
            "current_status":  current_state.get("status","UNKNOWN"),
        })

    if format == "snapshots":
        snaps = list(session_snapshots)[offset:offset+limit]
        return JSONResponse({"snapshots": snaps, "total": len(session_snapshots),
                             "offset": offset, "limit": limit})

    # Default: timeseries (compact, charting-friendly)
    data   = list(score_history)[offset:offset+limit]
    total  = len(score_history)
    return JSONResponse({
        "total":   total,
        "offset":  offset,
        "limit":   limit,
        "series": {
            "t":          [d["t"]          for d in data],
            "ts":         [d["ts"]         for d in data],
            "score":      [d["score"]      for d in data],
            "status":     [d["status"]     for d in data],
            "xTemp":      [d["xTemp"]      for d in data],
            "yTemp":      [d["yTemp"]      for d in data],
            "zTemp":      [d["zTemp"]      for d in data],
            "fine_z":     [d["fine_z"]     for d in data],
            "cnn_rms":    [d["cnn_rms"]    for d in data],
            "stalta":     [d["stalta"]     for d in data],
            "event_type": [d["event_type"] for d in data],
        },
        "events": event_log,
    })


# ── Report endpoints ───────────────────────────────────────────
def _build_all_charts():
    hist = list(score_history)
    return (hist,
            make_chart_anomaly_score(hist),
            make_chart_temperature(hist),
            make_chart_acceleration(hist),
            make_chart_model_comparison(current_state),
            make_chart_seismic_events(event_log),
            [make_analyzed_image_figure(r) for r in analyzed_images])


@app.get("/report/html")
def report_html():
    hist, cs, ct, ca, cm2, ce, aic = _build_all_charts()
    narrative = generate_narrative(current_state, hist)
    html = build_html_report(current_state, hist, narrative, ct, ca, cs, cm2, ce, aic)
    fn = f"mission_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    return HTMLResponse(content=html, headers={"Content-Disposition": f"attachment; filename={fn}"})


@app.get("/report/pdf")
def report_pdf():
    hist, cs, ct, ca, cm2, ce, aic = _build_all_charts()
    narrative = generate_narrative(current_state, hist)
    pdf = build_pdf_report(current_state, hist, narrative, ct, ca, cs, cm2, ce, aic)
    fn = f"mission_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename={fn}"})


# ── Image analysis endpoint ────────────────────────────────────
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...),
                        image_type: str = "sensor_chart"):
    contents  = await file.read()

    # ── Convert any image format (webp, png, jpg) to JPEG for llava ──
    # llava requires JPEG or PNG — webp is not supported
    try:
        from PIL import Image as PILImage
        import io as _io
        img_obj = PILImage.open(_io.BytesIO(contents))
        # Convert to RGB (handles RGBA, P mode etc.)
        if img_obj.mode not in ("RGB", "L"):
            img_obj = img_obj.convert("RGB")
        jpeg_buf = _io.BytesIO()
        img_obj.save(jpeg_buf, format="JPEG", quality=92)
        contents_for_llava = jpeg_buf.getvalue()
        print(f"[IMG] Converted {file.filename} ({len(contents)//1024}KB) → JPEG ({len(contents_for_llava)//1024}KB)")
    except Exception as conv_e:
        print(f"[IMG] Conversion failed ({conv_e}) — using original")
        contents_for_llava = contents

    image_b64 = base64.b64encode(contents_for_llava).decode()
    m = current_state.get("models", {})
    context = {
        "anomaly_score":     current_state.get("anomaly_score", 0),
        "status":            current_state.get("status", "UNKNOWN"),
        "lstm_status":       m.get("lstm",  {}).get("status", "UNKNOWN"),
        "stalta_triggered":  m.get("lstm",  {}).get("stalta_triggered", False),
        "cnn_event_type":    m.get("cnn",   {}).get("event_type", "unknown"),
        "cnn_pattern":       m.get("cnn",   {}).get("pattern", "NORMAL"),
        "trans_trend":       m.get("transformer", {}).get("trend", "STABLE"),
        "gnn_issues":        m.get("gnn",   {}).get("issues", []),
    }

    # Detect available models for diagnostics
    available_models = _ollama_models()
    vision_models    = [m2 for m2 in available_models
                        if any(v in m2.lower() for v in ["llava","moondream","bakllava"])]
    text_models      = [m2 for m2 in available_models if "llama" in m2.lower()]

    print(f"[IMG] File: {file.filename} ({len(contents)/1024:.1f} KB) type={image_type}")
    print(f"[IMG] All models: {available_models}")
    print(f"[IMG] Vision models found: {vision_models}")
    print(f"[IMG] Will use vision: {bool(vision_models)}")

    analysis = explain_image(image_b64, image_type, context)

    # ── Store for PDF injection ──────────────────────────────
    record = {
        "filename":   file.filename,
        "image_b64":  image_b64,
        "image_type": image_type,
        "analysis":   analysis,
        "ts":         datetime.now(timezone.utc).isoformat(),
        "context":    context,
    }
    analyzed_images.append(record)
    if len(analyzed_images) > 10:
        analyzed_images.pop(0)
    # Store as last image for follow-up questions in /ask
    global last_image_context
    last_image_context = {
        "filename":   file.filename,
        "image_type": image_type,
        "analysis":   analysis,
        "image_b64":  image_b64,
        "ts":         record["ts"],
    }

    return {"response": analysis, "analysis": analysis, "image_type": image_type,
            "diagnostics": {
                "available_models": available_models,
                "vision_models":    vision_models,
                "text_models":      text_models,
                "file_size_kb":     round(len(contents) / 1024, 1),
                "has_vision_model": len(vision_models) > 0,
                "install_hint":     "ollama pull llava" if not vision_models else None,
            },
            "filename": file.filename, "mission_context": context,
            "model_used": "llava (vision) / llama3 (fallback)",
            "stored_for_report": True, "stored_count": len(analyzed_images)}


@app.websocket("/telemetry")
async def telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")
    try:
        while True:
            await websocket.send_json(current_state)
            await asyncio.sleep(TELEMETRY_INTERVAL)
    except Exception:
        print("[WS] Client disconnected")


# ================================================================
# ENTRY POINT
# ================================================================
async def main():
    print("[ENGINE] Spacecraft Health AI v3.0 starting...")
    print("[ENGINE] Dashboard -> http://localhost:9000")
    print("[ENGINE] History   -> http://localhost:9000/history")
    print("[ENGINE] Events    -> http://localhost:9000/history?format=events")
    print("[ENGINE] Summary   -> http://localhost:9000/history?format=summary")
    print("[ENGINE] HTML Rpt  -> http://localhost:9000/report/html")
    print("[ENGINE] PDF  Rpt  -> http://localhost:9000/report/pdf")
    print("[ENGINE] Img AI    -> POST http://localhost:9000/analyze-image")
    print(f"[ENGINE] Dataset   -> {len(csv_files)} CSV files")
    loop = asyncio.get_event_loop()
    loop.create_task(telemetry_loop())
    config = uvicorn.Config(app, host="0.0.0.0", port=9000, log_level="info")
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ENGINE] Shutdown requested — goodbye.")
    except SystemExit:
        pass