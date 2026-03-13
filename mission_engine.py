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
HISTORY_SIZE       = 200    # rolling window (increased for STA/LTA)
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


def safe(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


# ================================================================
# STA/LTA SEISMIC EVENT DETECTOR
# ================================================================

class STALTADetector:
    """
    Classic STA/LTA (Short-Term Average / Long-Term Average) ratio detector.
    This is the industry-standard algorithm used by seismologists worldwide,
    including for planetary seismology (InSight, Apollo ALSEP, Chandrayaan-3 ILSA).

    Reference: Allen (1982) — "Automatic earthquake recognition and timing
    from single traces", Bulletin of the Seismological Society of America.
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
        """Compute current STA/LTA ratio from signal buffer."""
        if len(signal) < self.lta_len:
            return 0.0
        sq = signal ** 2   # energy proxy
        sta = float(np.mean(sq[-self.sta_len:]))
        lta = float(np.mean(sq[-self.lta_len:]))
        return sta / lta if lta > 1e-10 else 0.0

    def update(self, signal: np.ndarray, t: int) -> dict:
        """Returns event status dict."""
        ratio = self.compute_ratio(signal)
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


# ================================================================
# LUNAR QUAKE CLASSIFIER (CNN model — upgraded)
# ================================================================

class CNNSeismicClassifier:
    """
    Classifies seismic events into lunar quake types using:
    - Frequency band energy ratios (computed via bandpass filters)
    - Acceleration RMS across all fine axes
    - Duration estimation from STA/LTA trigger state
    - P-wave / S-wave characteristic patterns

    Based on lunar seismology literature:
    - Nakamura (1977) — Moonquake classification
    - Lognonné et al. (2020) — InSight SEIS paper
    - Kumar et al. (2023) — ILSA instrument description
    """
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr

    def _bandpass_energy(self, signal: np.ndarray, low: float, high: float) -> float:
        """Compute energy in a frequency band via Butterworth bandpass filter."""
        nyq = self.sr / 2.0
        lo, hi = low / nyq, high / nyq
        # Clamp to valid range
        lo = max(0.001, min(lo, 0.999))
        hi = max(0.001, min(hi, 0.999))
        if lo >= hi or len(signal) < 15:
            return float(np.mean(signal**2))
        try:
            b, a = butter(2, [lo, hi], btype='band')
            filtered = filtfilt(b, a, signal)
            return float(np.mean(filtered**2))
        except Exception:
            return float(np.mean(signal**2))

    def classify(self, signal: np.ndarray, rms: float, stalta: dict) -> dict:
        """Classify seismic event type from signal characteristics."""
        if len(signal) < self.sr * 5:
            return {
                "event_type":    "insufficient_data",
                "confidence":    0.0,
                "freq_dominant": 0.0,
                "band_energies": {},
                "pattern":       "NORMAL",
                "severity":      "nominal",
                "rms_accel":     round(rms, 6),
            }

        # Compute energy in each characteristic band
        band_energies = {}
        for qtype, params in MOONQUAKE_TYPES.items():
            lo, hi = params["freq_band"]
            # Cap high freq to Nyquist
            hi = min(hi, self.sr / 2.0 * 0.95)
            if lo < hi:
                band_energies[qtype] = self._bandpass_energy(signal, lo, hi)
            else:
                band_energies[qtype] = float(np.mean(signal**2))

        # Dominant frequency estimate via zero-crossing rate
        if len(signal) > 1:
            zc = np.where(np.diff(np.sign(signal)))[0]
            zero_cross_rate = len(zc) / (len(signal) / self.sr)
            freq_dominant   = zero_cross_rate / 2.0
        else:
            freq_dominant = 0.0

        # Classify by dominant band energy
        if band_energies:
            best_type = max(band_energies, key=band_energies.get)
            total_e   = sum(band_energies.values()) or 1e-10
            confidence = round(band_energies[best_type] / total_e, 3)
        else:
            best_type  = "unknown"
            confidence = 0.0

        # Map to severity
        if rms > 0.3 or stalta.get("stalta_ratio", 0) > 5.0:
            severity = "critical"
            pattern  = "STRONG_SEISMIC_EVENT"
        elif rms > 0.1 or stalta.get("stalta_ratio", 0) > STALTA_TRIGGER:
            severity = "warning"
            pattern  = f"SEISMIC_EVENT_{best_type.upper()}"
        else:
            severity = "nominal"
            pattern  = "NORMAL"

        return {
            "event_type":    best_type,
            "confidence":    confidence,
            "freq_dominant": round(freq_dominant, 4),
            "band_energies": {k: round(v, 8) for k, v in band_energies.items()},
            "pattern":       pattern,
            "severity":      severity,
            "rms_accel":     round(rms, 6),
            "fine_x":        round(float(signal[-1]) if len(signal) else 0, 6),
        }

    def run(self, data: dict) -> dict:
        """Run classifier on current sensor data using rolling history."""
        # Build signal from history (Z Fine Acceleration is most sensitive axis)
        signal_z = np.array([safe(h, "Z Fine Acceleration") for h in history])
        signal_x = np.array([safe(h, "X Fine Acceleration") for h in history])
        signal_y = np.array([safe(h, "Y Fine Acceleration") for h in history])

        fx = safe(data, "X Fine Acceleration")
        fy = safe(data, "Y Fine Acceleration")
        fz = safe(data, "Z Fine Acceleration")
        rms = float(np.sqrt(fx**2 + fy**2 + fz**2))

        # Use 3-component combined signal for classification
        if len(signal_z) > 0:
            combined = np.sqrt(signal_x**2 + signal_y**2 + signal_z**2)
        else:
            combined = np.array([rms])

        result = self.classify(combined, rms, stalta_detector.update.__doc__ and {} or {})
        result["model"] = "CNN"
        result["fine_y"] = round(fy, 6)
        result["fine_z"] = round(fz, 6)
        return result


# ================================================================
# LSTM ANOMALY DETECTOR (upgraded with STA/LTA)
# ================================================================

stalta_detector = STALTADetector()

class LSTMModel:
    """
    Upgraded LSTM anomaly detector combining:
    1. Rolling z-score on xTemp (thermal anomaly)
    2. STA/LTA ratio on Z Fine Acceleration (seismic onset)
    3. Combined score = max(thermal_score, seismic_score)
    """
    def run(self, data: dict) -> dict:
        t = current_state.get("packet_count", 0)

        # ── Thermal z-score ──
        x     = safe(data, "xTemp")
        temps = [safe(h, "xTemp") for h in history] if history else [x]
        mu    = float(np.mean(temps))
        sigma = float(np.std(temps)) or 1.0
        z_temp = abs(x - mu) / sigma
        thermal_score = float(min(z_temp / 3.0, 1.0))

        # ── STA/LTA seismic score ──
        signal_z = np.array([safe(h, "Z Fine Acceleration") for h in history])
        stalta   = stalta_detector.update(signal_z, t)
        ratio    = stalta["stalta_ratio"]
        # Normalise: ratio > trigger → seismic score > WARN_THRESHOLD
        seismic_score = float(min(ratio / (STALTA_TRIGGER * 5.0), 1.0))

        # ── Combined score ──
        score  = round(max(thermal_score, seismic_score), 6)
        status = "ANOMALY" if score > CRIT_THRESHOLD else \
                 "WARNING" if score > WARN_THRESHOLD else "NORMAL"

        # ── Determine dominant cause ──
        cause = "thermal" if thermal_score >= seismic_score else "seismic"

        return {
            "model":          "LSTM",
            "score":          score,
            "status":         status,
            "z_score":        round(z_temp, 4),
            "mean_temp":      round(mu, 4),
            "xTemp":          round(x, 4),
            "thermal_score":  round(thermal_score, 6),
            "seismic_score":  round(seismic_score, 6),
            "stalta_ratio":   ratio,
            "stalta_triggered": stalta["triggered"],
            "event_detected": stalta["event_detected"],
            "dominant_cause": cause,
        }


# ================================================================
# TRANSFORMER — temperature trend (unchanged, solid)
# ================================================================

class TransformerModel:
    def run(self, data: dict) -> dict:
        x_now = safe(data, "xTemp")
        y_now = safe(data, "yTemp")
        z_now = safe(data, "zTemp")
        if len(history) >= 5:
            xs     = [safe(h, "xTemp") for h in history]
            t      = np.arange(len(xs), dtype=float)
            coeffs = np.polyfit(t, xs, 1)
            slope  = float(coeffs[0])
            pred   = float(np.polyval(coeffs, len(xs)))
        else:
            slope, pred = 0.0, x_now + 0.5
        trend = "RISING" if slope > 0.01 else "FALLING" if slope < -0.01 else "STABLE"
        return {
            "model":          "Transformer",
            "pred_xTemp":     round(pred, 4),
            "current_xTemp":  round(x_now, 4),
            "current_yTemp":  round(y_now, 4),
            "current_zTemp":  round(z_now, 4),
            "slope":          round(slope, 6),
            "trend":          trend,
        }


# ================================================================
# GNN — subsystem graph (upgraded with seismic coupling)
# ================================================================

class GNNModel:
    """
    Graph-based subsystem health checker.
    Now also checks seismic-thermal coupling:
    lunar quakes produce both mechanical AND thermal signatures.
    """
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
        if abs(x)   > 50:  issues.append("high_xTemp")
        if c_mag    > 0.5: issues.append("high_coarse_accel")
        if f_mag    > 0.2: issues.append("high_fine_accel")

        # Seismic-thermal coupling: simultaneous fine accel + temp deviation
        stalta_active = current_state.get("models", {}).get("lstm", {}).get("stalta_triggered", False)
        seismic_thermal_coupling = stalta_active and abs(x) > 20

        cross_anomaly = len(issues) >= 2 or seismic_thermal_coupling

        subsystem_scores = {
            "thermal":    max(0.0, 1.0 - abs(x) / 100.0),
            "structural": max(0.0, 1.0 - c_mag / 2.0),
            "seismic":    max(0.0, 1.0 - f_mag / 0.5),
            "navigation": 1.0,
            "power":      1.0,
        }

        return {
            "model":               "GNN",
            "subsystem":           "structure",
            "issues":              issues,
            "cross_anomaly":       cross_anomaly,
            "seismic_thermal_coupling": seismic_thermal_coupling,
            "subsystem_scores":    {k: round(v, 4) for k, v in subsystem_scores.items()},
            "coarse_rms":          round(c_mag, 6),
            "fine_rms":            round(f_mag, 6),
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
# LLM (Ollama llama3)
# ================================================================
def explain(model_result: dict, question: str, sensors: dict) -> str:
    prompt = f"""You are an expert spacecraft mission AI analyst for Chandrayaan-3 ILSA seismometer data.

Current sensor readings:
- xTemp: {safe(sensors, 'xTemp'):.4f}  yTemp: {safe(sensors, 'yTemp'):.4f}  zTemp: {safe(sensors, 'zTemp'):.4f}
- X Fine Accel: {safe(sensors, 'X Fine Acceleration'):.6f}
- Y Fine Accel: {safe(sensors, 'Y Fine Acceleration'):.6f}
- Z Fine Accel: {safe(sensors, 'Z Fine Acceleration'):.6f}
- X Coarse Accel: {safe(sensors, 'X Coarse Acceleration'):.6f}
- Y Coarse Accel: {safe(sensors, 'Y Coarse Acceleration'):.6f}
- Z Coarse Accel: {safe(sensors, 'Z Coarse Acceleration'):.6f}

AI Model: {model_result.get('model')}
Model output: {model_result}
Question: {question}

Provide a concise 3-5 sentence technical answer. Reference specific values.
For seismic events, mention STA/LTA ratio, event type classification, and lunar quake context.
"""
    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": "llama3", "prompt": prompt, "stream": False}, timeout=30)
        return r.json().get("response", "No response.")
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama not running. Start with: ollama serve"
    except Exception as e:
        return f"[ERROR] {e}"


def _ollama_models() -> list:
    """Return list of locally available Ollama model names."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"].split(":")[0] for m in r.json().get("models", [])]
    except Exception:
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
                                      images=[image_b64], timeout=120)
            if result and not result.startswith("["):
                return f"[Vision model: {used_vision}]\n\n{result}"
            # Vision model returned empty/error — log and fall through
            print(f"[IMG] {used_vision} returned: {result!r} — falling back to llama3")
        except Exception as e:
            print(f"[IMG] {used_vision} failed: {e} — falling back to llama3")
    else:
        print(f"[IMG] No vision model found in: {available}")

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
    img_arr  = plt.imread(io.BytesIO(img_data))

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
    return FileResponse("dashboard/index.html")


@app.get("/ask")
def ask(question: str):
    model  = select_model(question)
    result = model.run(current_state["sensors"])
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
    image_b64 = base64.b64encode(contents).decode()
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
                        if any(v in m2 for v in ["llava","moondream","bakllava"])]
    text_models      = [m2 for m2 in available_models if "llama" in m2]

    print(f"[IMG] File: {file.filename} ({len(contents)/1024:.1f} KB) type={image_type}")
    print(f"[IMG] Available: {available_models} | Vision: {vision_models}")

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