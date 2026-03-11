import asyncio
import os
import io
import base64
import tempfile
from datetime import datetime
from collections import deque

import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ReportLab for PDF generation
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

DATA_FOLDER      = r"G:\Pro_NML\data\ch3_ilsa\ils\data\calibrated"
TELEMETRY_INTERVAL = 1.0
HISTORY_SIZE       = 60
WARN_THRESHOLD     = 0.05
CRIT_THRESHOLD     = 0.20

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
# ROLLING HISTORY
# ================================================================

history: deque = deque(maxlen=HISTORY_SIZE)
score_history: deque = deque(maxlen=500)   # for report charts
sensor_history: deque = deque(maxlen=500)  # for report charts


def safe(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default

# ================================================================
# AI MODELS
# ================================================================

class LSTMModel:
    def run(self, data: dict) -> dict:
        x    = safe(data, "xTemp")
        vals = [safe(h, "xTemp") for h in history] if history else [x]
        mu   = float(np.mean(vals))
        sigma = float(np.std(vals)) or 1.0
        z     = abs(x - mu) / sigma
        score = float(min(z / 3.0, 1.0))
        status = "ANOMALY" if score > CRIT_THRESHOLD else \
                 "WARNING" if score > WARN_THRESHOLD else "NORMAL"
        return {
            "model": "LSTM", "score": round(score, 6),
            "status": status, "z_score": round(z, 4),
            "mean_temp": round(mu, 4), "xTemp": round(x, 4),
        }


class CNNModel:
    def run(self, data: dict) -> dict:
        fx  = safe(data, "X Fine Acceleration")
        fy  = safe(data, "Y Fine Acceleration")
        fz  = safe(data, "Z Fine Acceleration")
        rms = float(np.sqrt(fx**2 + fy**2 + fz**2))
        if rms > 0.3:
            pattern, severity = "STRONG_VIBRATION", "critical"
        elif rms > 0.1:
            pattern, severity = "VIBRATION_PATTERN", "warning"
        else:
            pattern, severity = "NORMAL", "nominal"
        return {
            "model": "CNN", "pattern": pattern, "severity": severity,
            "rms_accel": round(rms, 6),
            "fine_x": round(fx, 6), "fine_y": round(fy, 6), "fine_z": round(fz, 6),
        }


class TransformerModel:
    def run(self, data: dict) -> dict:
        x_now = safe(data, "xTemp")
        y_now = safe(data, "yTemp")
        z_now = safe(data, "zTemp")
        if len(history) >= 5:
            xs    = [safe(h, "xTemp") for h in history]
            t     = np.arange(len(xs), dtype=float)
            coeffs = np.polyfit(t, xs, 1)
            slope  = float(coeffs[0])
            pred_next = float(np.polyval(coeffs, len(xs)))
        else:
            slope, pred_next = 0.0, x_now + 0.5
        trend = "RISING" if slope > 0.01 else "FALLING" if slope < -0.01 else "STABLE"
        return {
            "model": "Transformer", "pred_xTemp": round(pred_next, 4),
            "current_xTemp": round(x_now, 4), "current_yTemp": round(y_now, 4),
            "current_zTemp": round(z_now, 4),
            "slope": round(slope, 6), "trend": trend,
        }


class GNNModel:
    def run(self, data: dict) -> dict:
        x   = safe(data, "xTemp")
        cx  = safe(data, "X Coarse Acceleration")
        cy  = safe(data, "Y Coarse Acceleration")
        cz  = safe(data, "Z Coarse Acceleration")
        c_mag = float(np.sqrt(cx**2 + cy**2 + cz**2))
        issues = []
        if abs(x) > 50: issues.append("high_xTemp")
        if c_mag > 0.5:  issues.append("high_coarse_accel")
        cross_anomaly = len(issues) >= 2
        subsystem_scores = {
            "thermal":    max(0, 1 - abs(x) / 100),
            "structural": max(0, 1 - c_mag / 2),
            "navigation": 1.0,
            "power":      1.0,
        }
        return {
            "model": "GNN", "subsystem": "structure",
            "issues": issues, "cross_anomaly": cross_anomaly,
            "subsystem_scores": {k: round(v, 4) for k, v in subsystem_scores.items()},
            "coarse_rms": round(c_mag, 6),
        }


lstm        = LSTMModel()
cnn         = CNNModel()
transformer = TransformerModel()
gnn         = GNNModel()

# ================================================================
# MODEL ROUTER
# ================================================================

def select_model(question: str):
    q = question.lower()
    if "pattern"  in q or "vibration" in q: return cnn
    if "predict"  in q or "forecast"  in q: return transformer
    if "cause"    in q or "graph"     in q or "subsystem" in q: return gnn
    return lstm


def model_name(question: str) -> str:
    q = question.lower()
    if "pattern"  in q or "vibration" in q: return "CNN"
    if "predict"  in q or "forecast"  in q: return "Transformer"
    if "cause"    in q or "graph"     in q or "subsystem" in q: return "GNN"
    return "LSTM"

# ================================================================
# LLM REASONING (Ollama llama3)
# ================================================================

def explain(model_result: dict, question: str, sensors: dict) -> str:
    prompt = f"""You are an expert spacecraft mission AI analyst for Chandrayaan-3 ILSA seismometer data.

Current sensor readings:
- xTemp: {safe(sensors, 'xTemp'):.4f}
- yTemp: {safe(sensors, 'yTemp'):.4f}
- zTemp: {safe(sensors, 'zTemp'):.4f}
- X Fine Acceleration: {safe(sensors, 'X Fine Acceleration'):.6f}
- Y Fine Acceleration: {safe(sensors, 'Y Fine Acceleration'):.6f}
- Z Fine Acceleration: {safe(sensors, 'Z Fine Acceleration'):.6f}
- X Coarse Acceleration: {safe(sensors, 'X Coarse Acceleration'):.6f}
- Y Coarse Acceleration: {safe(sensors, 'Y Coarse Acceleration'):.6f}
- Z Coarse Acceleration: {safe(sensors, 'Z Coarse Acceleration'):.6f}

AI Model used: {model_result.get('model', 'Unknown')}
Model output: {model_result}
Operator question: {question}

Give a concise (3-5 sentence) technical answer.
State what the data shows, whether it is a concern, and what action the operator should take.
"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=30
        )
        return r.json().get("response", "No response from Ollama.")
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama not running. Start with: ollama serve"
    except Exception as e:
        return f"[ERROR] LLM call failed: {str(e)}"


def explain_image(image_b64: str, image_type: str, context: dict) -> str:
    """Send image to Ollama llava (vision model) for analysis."""
    # Build rich context prompt
    prompt = f"""You are an expert spacecraft mission AI analyst for Chandrayaan-3 ILSA seismometer data.

The operator has uploaded an image of type: {image_type}

Current mission context:
- Anomaly Score: {context.get('anomaly_score', 0):.4f}
- Mission Status: {context.get('status', 'UNKNOWN')}
- LSTM Status: {context.get('lstm_status', 'UNKNOWN')}
- CNN Pattern: {context.get('cnn_pattern', 'UNKNOWN')}
- Transformer Trend: {context.get('trans_trend', 'UNKNOWN')}
- GNN Issues: {context.get('gnn_issues', [])}

Please analyze this image and provide:
1. DESCRIPTION: What does this image show?
2. ANOMALIES: Any visible anomalies, patterns, or concerns?
3. CORRELATION: How does it correlate with current mission telemetry?
4. RECOMMENDATION: What action should the operator take?

Be specific and technical. Reference actual values from the image if visible.
"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [image_b64],
                "stream": False
            },
            timeout=60
        )
        return r.json().get("response", "No response from vision model.")
    except requests.exceptions.ConnectionError:
        # Fallback: use llama3 text-only with image metadata
        fallback_prompt = f"""You are a spacecraft mission AI analyst.
An operator uploaded a {image_type} image. The llava vision model is not available.

Based on current telemetry context:
- Anomaly Score: {context.get('anomaly_score', 0):.4f} (threshold: warn>0.05, crit>0.20)
- Status: {context.get('status', 'UNKNOWN')}
- CNN Pattern: {context.get('cnn_pattern', 'UNKNOWN')}
- GNN Issues: {context.get('gnn_issues', [])}

Provide analysis guidance for a {image_type}. What patterns should the operator look for?
What does the current telemetry suggest about spacecraft health?
"""
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": fallback_prompt, "stream": False},
                timeout=30
            )
            return "[llava not found — using llama3 text analysis]\n\n" + r.json().get("response", "")
        except Exception:
            return "[ERROR] No Ollama models available. Run: ollama pull llava"
    except Exception as e:
        return f"[ERROR] Image analysis failed: {str(e)}"

# ================================================================
# MISSION STATE
# ================================================================

current_state = {
    "sensors":       {},
    "anomaly_score": 0.0,
    "status":        "NOMINAL",
    "models": {
        "lstm":        {},
        "cnn":         {},
        "transformer": {},
        "gnn":         {},
    },
    "packet_count": 0,
    "file_info":    "",
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

        current_state.update({
            "sensors":       packet,
            "anomaly_score": score,
            "status":        status,
            "packet_count":  current_state["packet_count"] + 1,
            "file_info":     csv_files[file_index - 1] if file_index > 0 else "",
            "models": {
                "lstm":        lstm_out,
                "cnn":         cnn_out,
                "transformer": trans_out,
                "gnn":         gnn_out,
            },
        })

        # Store for report generation
        score_history.append({
            "t":     current_state["packet_count"],
            "score": score,
            "xTemp": safe(packet, "xTemp"),
            "yTemp": safe(packet, "yTemp"),
            "zTemp": safe(packet, "zTemp"),
            "fine_x": safe(packet, "X Fine Acceleration"),
            "fine_y": safe(packet, "Y Fine Acceleration"),
            "fine_z": safe(packet, "Z Fine Acceleration"),
            "coarse_x": safe(packet, "X Coarse Acceleration"),
            "coarse_y": safe(packet, "Y Coarse Acceleration"),
            "coarse_z": safe(packet, "Z Coarse Acceleration"),
            "cnn_rms": cnn_out.get("rms_accel", 0),
            "pred_temp": trans_out.get("pred_xTemp", 0),
        })

        await asyncio.sleep(TELEMETRY_INTERVAL)

# ================================================================
# CHART GENERATION  (matplotlib → PNG bytes)
# ================================================================

DARK_BG  = "#070f1a"
DARK_AX  = "#0a1520"
C_CYAN   = "#00b4ff"
C_GREEN  = "#00ff88"
C_AMBER  = "#ffaa00"
C_RED    = "#ff3355"
C_PURPLE = "#aa44ff"
C_LABEL  = "#4a7a99"
C_GRID   = "#0d2035"

def style_ax(ax, title=""):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=C_LABEL, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#0d2035")
    ax.yaxis.label.set_color(C_LABEL)
    ax.xaxis.label.set_color(C_LABEL)
    ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=C_CYAN, fontsize=8, fontweight="bold",
                     fontfamily="monospace", pad=6)


def make_chart_temperature(data: list) -> bytes:
    """Line chart: xTemp, yTemp, zTemp + Transformer prediction."""
    fig, ax = plt.subplots(figsize=(10, 3), facecolor=DARK_BG)
    if data:
        t  = [d["t"]      for d in data]
        xt = [d["xTemp"]  for d in data]
        yt = [d["yTemp"]  for d in data]
        zt = [d["zTemp"]  for d in data]
        pt = [d["pred_temp"] for d in data]
        ax.plot(t, xt, color=C_CYAN,   linewidth=1.2, label="xTemp (LSTM input)")
        ax.plot(t, yt, color=C_GREEN,  linewidth=1.2, label="yTemp")
        ax.plot(t, zt, color=C_AMBER,  linewidth=1.2, label="zTemp")
        ax.plot(t, pt, color=C_PURPLE, linewidth=0.8, linestyle="--", label="pred_xTemp (Transformer)")
    style_ax(ax, "TEMPERATURE SENSORS — xTemp / yTemp / zTemp + Transformer Prediction")
    ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white",
              edgecolor=C_GRID, loc="upper right")
    ax.set_xlabel("Packet #", fontsize=7)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def make_chart_acceleration(data: list) -> bytes:
    """Line chart: all 6 acceleration axes + CNN RMS."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), facecolor=DARK_BG)
    if data:
        t   = [d["t"] for d in data]
        ax1.plot(t, [d["fine_x"]   for d in data], color=C_CYAN,   linewidth=1,   label="Fine X")
        ax1.plot(t, [d["fine_y"]   for d in data], color=C_GREEN,  linewidth=1,   label="Fine Y")
        ax1.plot(t, [d["fine_z"]   for d in data], color=C_AMBER,  linewidth=1.2, label="Fine Z (CNN key)")
        ax2.plot(t, [d["coarse_x"] for d in data], color=C_PURPLE, linewidth=1,   label="Coarse X")
        ax2.plot(t, [d["coarse_y"] for d in data], color="#ff6644", linewidth=1,   label="Coarse Y")
        ax2.plot(t, [d["coarse_z"] for d in data], color="#44ffcc", linewidth=1,   label="Coarse Z")
        ax2.plot(t, [d["cnn_rms"]  for d in data], color=C_RED,    linewidth=1.5, linestyle="--", label="CNN RMS")
        ax2.axhline(0.1, color=C_AMBER, linewidth=0.7, linestyle=":", alpha=0.7, label="warn 0.1")
        ax2.axhline(0.3, color=C_RED,   linewidth=0.7, linestyle=":", alpha=0.7, label="crit 0.3")
    style_ax(ax1, "FINE ACCELERATION — X / Y / Z")
    style_ax(ax2, "COARSE ACCELERATION + CNN RMS")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white",
                  edgecolor=C_GRID, loc="upper right")
        ax.set_xlabel("Packet #", fontsize=7)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def make_chart_anomaly_score(data: list) -> bytes:
    """Anomaly score history with threshold bands."""
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor=DARK_BG)
    if data:
        t  = [d["t"]     for d in data]
        sc = [d["score"] for d in data]
        ax.fill_between(t, sc, alpha=0.2, color=C_RED)
        ax.plot(t, sc, color=C_RED, linewidth=1.5, label="LSTM Anomaly Score")
        ax.axhline(WARN_THRESHOLD, color=C_AMBER, linewidth=1,   linestyle="--", label=f"Warning ({WARN_THRESHOLD})")
        ax.axhline(CRIT_THRESHOLD, color=C_RED,   linewidth=1,   linestyle="--", label=f"Critical ({CRIT_THRESHOLD})")
        ax.set_ylim(0, max(max(sc) * 1.2, 0.3))
    style_ax(ax, "LSTM ANOMALY SCORE HISTORY")
    ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white",
              edgecolor=C_GRID, loc="upper right")
    ax.set_xlabel("Packet #", fontsize=7)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def make_chart_model_comparison(state: dict) -> bytes:
    """4-panel summary: one cell per model showing key metric."""
    m = state.get("models", {})
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), facecolor=DARK_BG)

    panels = [
        ("LSTM",        m.get("lstm", {}),        C_CYAN),
        ("CNN",         m.get("cnn", {}),          C_GREEN),
        ("TRANSFORMER", m.get("transformer", {}),  C_AMBER),
        ("GNN",         m.get("gnn", {}),          C_PURPLE),
    ]

    for ax, (name, out, color) in zip(axes, panels):
        ax.set_facecolor(DARK_AX)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        # Title bar
        ax.add_patch(plt.Rectangle((0, 0.82), 1, 0.18, color=color, alpha=0.25))
        ax.text(0.5, 0.91, name, ha="center", va="center", color=color,
                fontsize=9, fontweight="bold", fontfamily="monospace")
        # Key metric
        if name == "LSTM":
            lines = [
                f"Score: {out.get('score', 0):.4f}",
                f"Status: {out.get('status', '—')}",
                f"Z-score: {out.get('z_score', 0):.3f}",
                f"Mean T: {out.get('mean_temp', 0):.2f}",
            ]
        elif name == "CNN":
            lines = [
                f"Pattern: {out.get('pattern', '—')}",
                f"Severity: {out.get('severity', '—')}",
                f"RMS: {out.get('rms_accel', 0):.4f}",
                f"Fine Z: {out.get('fine_z', 0):.4f}",
            ]
        elif name == "TRANSFORMER":
            lines = [
                f"Trend: {out.get('trend', '—')}",
                f"pred_xTemp: {out.get('pred_xTemp', 0):.3f}",
                f"Slope: {out.get('slope', 0):.5f}",
                f"Cur xTemp: {out.get('current_xTemp', 0):.3f}",
            ]
        else:
            scores = out.get("subsystem_scores", {})
            issues = out.get("issues", [])
            lines = [
                f"Thermal: {scores.get('thermal', 1):.2f}",
                f"Structural: {scores.get('structural', 1):.2f}",
                f"Coarse RMS: {out.get('coarse_rms', 0):.4f}",
                f"Issues: {', '.join(issues) or 'none'}",
            ]
        for i, line in enumerate(lines):
            ax.text(0.05, 0.72 - i * 0.17, line, ha="left", va="top",
                    color="white", fontsize=7.5, fontfamily="monospace")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
        ax.set_visible(True)

    plt.suptitle("ALL 4 MODEL OUTPUTS — SNAPSHOT", color=C_CYAN,
                 fontsize=9, fontfamily="monospace", y=1.01)
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    return buf.getvalue()

# ================================================================
# NARRATIVE GENERATION
# ================================================================

def generate_narrative(state: dict, hist_data: list) -> str:
    """Ask llama3 to write a full mission health narrative for the report."""
    m      = state.get("models", {})
    lstm   = m.get("lstm",  {})
    cnn_m  = m.get("cnn",   {})
    trans  = m.get("transformer", {})
    gnn_m  = m.get("gnn",  {})

    # Compute session stats from history
    scores = [d["score"] for d in hist_data] if hist_data else [0]
    max_sc = max(scores)
    avg_sc = float(np.mean(scores))
    crit_events = sum(1 for s in scores if s > CRIT_THRESHOLD)
    warn_events = sum(1 for s in scores if WARN_THRESHOLD < s <= CRIT_THRESHOLD)

    prompt = f"""You are a senior spacecraft mission analyst writing an official mission health report for Chandrayaan-3 ILSA.

SESSION STATISTICS:
- Total packets processed: {len(hist_data)}
- Max anomaly score: {max_sc:.4f}
- Average anomaly score: {avg_sc:.4f}
- Critical events (score > {CRIT_THRESHOLD}): {crit_events}
- Warning events (score > {WARN_THRESHOLD}): {warn_events}

CURRENT MODEL OUTPUTS:
LSTM: score={lstm.get('score',0):.4f}, status={lstm.get('status','—')}, z_score={lstm.get('z_score',0):.3f}
CNN: pattern={cnn_m.get('pattern','—')}, severity={cnn_m.get('severity','—')}, rms={cnn_m.get('rms_accel',0):.4f}
Transformer: trend={trans.get('trend','—')}, pred_xTemp={trans.get('pred_xTemp',0):.3f}, slope={trans.get('slope',0):.6f}
GNN: issues={gnn_m.get('issues',[])}, cross_anomaly={gnn_m.get('cross_anomaly',False)}, thermal={gnn_m.get('subsystem_scores',{}).get('thermal',1):.2f}

Write a professional 4-paragraph mission health narrative:
1. Executive summary of mission status
2. Seismic / acceleration analysis (CNN + GNN findings)
3. Thermal analysis and temperature trend (LSTM + Transformer findings)
4. Recommendations for operators

Use formal technical language. Be specific with numbers. Reference the Chandrayaan-3 ILSA instrument.
"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=60
        )
        return r.json().get("response", "Narrative generation failed.")
    except Exception as e:
        # Fallback: generate a template narrative without LLM
        status_word = "nominal" if avg_sc < WARN_THRESHOLD else "elevated" if avg_sc < CRIT_THRESHOLD else "critical"
        return f"""EXECUTIVE SUMMARY
The Chandrayaan-3 ILSA spacecraft health monitoring system has processed {len(hist_data)} telemetry packets during this session. Overall mission status is {status_word.upper()} with an average anomaly score of {avg_sc:.4f}. A total of {crit_events} critical events and {warn_events} warning events were detected.

SEISMIC AND ACCELERATION ANALYSIS
The CNN vibration pattern classifier reports {cnn_m.get('pattern','NORMAL')} with an RMS acceleration of {cnn_m.get('rms_accel',0):.4f}. The GNN structural analysis detected the following subsystem issues: {', '.join(gnn_m.get('issues',[])) or 'none'}. Cross-anomaly correlation between thermal and mechanical subsystems is {'ACTIVE' if gnn_m.get('cross_anomaly') else 'NOT DETECTED'}.

THERMAL ANALYSIS
The LSTM anomaly detector reports a current z-score of {lstm.get('z_score',0):.3f} with status {lstm.get('status','NORMAL')}. The Transformer trend predictor indicates a {trans.get('trend','STABLE')} temperature trend with a slope of {trans.get('slope',0):.6f} per packet. Predicted next xTemp reading: {trans.get('pred_xTemp',0):.3f}.

RECOMMENDATIONS
{'IMMEDIATE ATTENTION REQUIRED: Anomaly score exceeds critical threshold.' if max_sc > CRIT_THRESHOLD else 'Monitor anomaly score trends. No immediate action required.' if max_sc > WARN_THRESHOLD else 'All systems nominal. Continue standard monitoring protocol.'}
(Note: Ollama unavailable — this is an auto-generated narrative. Start Ollama for AI-written reports.)"""

# ================================================================
# REPORT BUILDER  (PDF + HTML)
# ================================================================

def build_html_report(state: dict, hist_data: list, narrative: str,
                      chart_temp: bytes, chart_accel: bytes,
                      chart_score: bytes, chart_models: bytes) -> str:
    """Generate a complete self-contained HTML report."""
    now     = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    m       = state.get("models", {})
    lstm    = m.get("lstm",  {})
    cnn_m   = m.get("cnn",   {})
    trans   = m.get("transformer", {})
    gnn_m   = m.get("gnn",  {})
    score   = state.get("anomaly_score", 0)
    status  = state.get("status", "UNKNOWN")
    status_color = "#ff3355" if status == "ANOMALY" else "#ffaa00" if status == "WARNING" else "#00ff88"

    # Encode charts to base64
    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode()

    scores = [d["score"] for d in hist_data] if hist_data else [0]
    max_sc = max(scores); avg_sc = float(np.mean(scores))
    crit_n = sum(1 for s in scores if s > CRIT_THRESHOLD)
    warn_n = sum(1 for s in scores if WARN_THRESHOLD < s <= CRIT_THRESHOLD)

    # Format narrative paragraphs
    paras = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    para_html = "".join(f'<p style="margin:0 0 14px 0;line-height:1.7">{p}</p>' for p in paras)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mission Health Report — {now}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;500&display=swap');
  :root{{--bg:#020508;--panel:#070f1a;--panel2:#0a1520;--cyan:#00b4ff;--green:#00ff88;--amber:#ffaa00;--red:#ff3355;--purple:#aa44ff;--tp:#c8e8ff;--ts:#4a7a99;--tl:#2a5a77;--b:rgba(0,180,255,0.15);}}
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:var(--bg);color:var(--tp);font-family:'Exo 2',sans-serif;padding:40px;min-height:100vh}}
  .report-wrap{{max-width:1100px;margin:0 auto}}
  /* Header */
  .rh{{border-bottom:2px solid var(--cyan);padding-bottom:20px;margin-bottom:30px;display:flex;justify-content:space-between;align-items:flex-end}}
  .rh-left h1{{font-family:'Orbitron',sans-serif;font-size:22px;font-weight:900;color:var(--cyan);letter-spacing:4px;text-shadow:0 0 20px rgba(0,180,255,0.4);}}
  .rh-left .sub{{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--ts);letter-spacing:3px;margin-top:5px}}
  .rh-right{{text-align:right;font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--tl);line-height:1.8}}
  /* Status banner */
  .status-bar{{background:var(--panel2);border:1px solid var(--b);border-left:4px solid {status_color};border-radius:4px;padding:14px 20px;margin-bottom:24px;display:flex;align-items:center;gap:20px}}
  .status-dot{{width:12px;height:12px;border-radius:50%;background:{status_color};box-shadow:0 0 10px {status_color};flex-shrink:0}}
  .status-text{{font-family:'Orbitron',sans-serif;font-size:13px;font-weight:700;color:{status_color};letter-spacing:3px}}
  .status-score{{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--ts);margin-left:auto}}
  /* Section */
  .section{{margin-bottom:28px}}
  .section-title{{font-family:'Orbitron',sans-serif;font-size:10px;font-weight:700;color:var(--cyan);letter-spacing:4px;margin-bottom:14px;padding-bottom:6px;border-bottom:1px solid var(--b);display:flex;align-items:center;gap:8px}}
  .section-title::before{{content:'';display:inline-block;width:4px;height:14px;background:var(--cyan);border-radius:2px}}
  /* Stats grid */
  .stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}}
  .stat-card{{background:var(--panel2);border:1px solid var(--b);border-radius:4px;padding:14px;text-align:center}}
  .stat-lbl{{font-family:'Share Tech Mono',monospace;font-size:8px;color:var(--tl);letter-spacing:2px;margin-bottom:6px}}
  .stat-val{{font-family:'Orbitron',sans-serif;font-size:18px;font-weight:700}}
  .stat-sub{{font-family:'Share Tech Mono',monospace;font-size:7px;color:var(--tl);margin-top:4px}}
  /* Charts */
  .chart-box{{background:var(--panel2);border:1px solid var(--b);border-radius:4px;padding:2px;margin-bottom:14px}}
  .chart-box img{{width:100%;display:block;border-radius:3px}}
  /* Model table */
  .model-table{{width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:9px}}
  .model-table th{{background:var(--panel2);color:var(--tl);letter-spacing:2px;padding:8px 12px;border-bottom:1px solid var(--b);text-align:left;font-weight:normal}}
  .model-table td{{padding:8px 12px;border-bottom:1px solid rgba(0,180,255,0.06);color:var(--tp)}}
  .model-table tr:hover td{{background:rgba(0,180,255,0.03)}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:2px;font-size:8px;letter-spacing:1px}}
  .badge-ok{{background:rgba(0,255,136,0.15);color:#00ff88}}
  .badge-warn{{background:rgba(255,170,0,0.15);color:#ffaa00}}
  .badge-crit{{background:rgba(255,51,85,0.15);color:#ff3355}}
  /* Narrative */
  .narrative{{background:var(--panel2);border:1px solid var(--b);border-radius:4px;padding:22px;font-size:13px;color:var(--tp);line-height:1.7}}
  .narrative h2{{font-family:'Orbitron',sans-serif;font-size:10px;color:var(--cyan);letter-spacing:3px;margin-bottom:8px;margin-top:14px}}
  /* Footer */
  .rfooter{{border-top:1px solid var(--b);padding-top:16px;margin-top:30px;display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:8px;color:var(--tl)}}
  @media print{{body{{background:#fff;color:#000}}}}
</style>
</head>
<body>
<div class="report-wrap">

  <div class="rh">
    <div class="rh-left">
      <h1>SPACECRAFT HEALTH AI</h1>
      <div class="sub">CHANDRAYAAN-3 · ILSA SEISMOMETER · MISSION HEALTH REPORT</div>
    </div>
    <div class="rh-right">
      <div>Generated: {now}</div>
      <div>Engine: v2.0 · Models: LSTM · CNN · Transformer · GNN</div>
      <div>Dataset: CH3-ILSA Calibrated CSVs</div>
      <div>Packets Analysed: {len(hist_data)}</div>
    </div>
  </div>

  <div class="status-bar">
    <div class="status-dot"></div>
    <div class="status-text">MISSION STATUS: {status}</div>
    <div class="status-score">
      Anomaly Score: {score:.4f} &nbsp;|&nbsp;
      Max Score: {max_sc:.4f} &nbsp;|&nbsp;
      Avg Score: {avg_sc:.4f} &nbsp;|&nbsp;
      Critical Events: {crit_n} &nbsp;|&nbsp;
      Warning Events: {warn_n}
    </div>
  </div>

  <div class="section">
    <div class="section-title">SESSION STATISTICS</div>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-lbl">ANOMALY SCORE</div>
        <div class="stat-val" style="color:{status_color}">{score:.4f}</div>
        <div class="stat-sub">{status}</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">LSTM Z-SCORE</div>
        <div class="stat-val" style="color:var(--cyan)">{lstm.get('z_score',0):.3f}</div>
        <div class="stat-sub">std deviations</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">CNN PATTERN</div>
        <div class="stat-val" style="color:var(--green);font-size:12px">{cnn_m.get('pattern','—')}</div>
        <div class="stat-sub">RMS: {cnn_m.get('rms_accel',0):.4f}</div>
      </div>
      <div class="stat-card">
        <div class="stat-lbl">TEMPERATURE TREND</div>
        <div class="stat-val" style="color:var(--amber);font-size:12px">{trans.get('trend','—')}</div>
        <div class="stat-sub">pred: {trans.get('pred_xTemp',0):.3f}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">ANOMALY SCORE HISTORY</div>
    <div class="chart-box"><img src="data:image/png;base64,{b64(chart_score)}" alt="Anomaly Score Chart"/></div>
  </div>

  <div class="section">
    <div class="section-title">TEMPERATURE SENSORS</div>
    <div class="chart-box"><img src="data:image/png;base64,{b64(chart_temp)}" alt="Temperature Chart"/></div>
  </div>

  <div class="section">
    <div class="section-title">ACCELERATION SENSORS</div>
    <div class="chart-box"><img src="data:image/png;base64,{b64(chart_accel)}" alt="Acceleration Chart"/></div>
  </div>

  <div class="section">
    <div class="section-title">ALL 4 MODEL OUTPUTS</div>
    <div class="chart-box"><img src="data:image/png;base64,{b64(chart_models)}" alt="Model Comparison"/></div>
    <table class="model-table" style="margin-top:12px">
      <thead><tr><th>MODEL</th><th>KEY METRIC</th><th>VALUE</th><th>STATUS</th></tr></thead>
      <tbody>
        <tr><td style="color:var(--cyan)">LSTM</td><td>Anomaly Score / Z-Score</td><td>{lstm.get('score',0):.4f} / {lstm.get('z_score',0):.3f}</td><td><span class="badge {'badge-crit' if lstm.get('status')=='ANOMALY' else 'badge-warn' if lstm.get('status')=='WARNING' else 'badge-ok'}">{lstm.get('status','—')}</span></td></tr>
        <tr><td style="color:var(--green)">CNN</td><td>Vibration Pattern / RMS</td><td>{cnn_m.get('pattern','—')} / {cnn_m.get('rms_accel',0):.4f}</td><td><span class="badge {'badge-crit' if cnn_m.get('severity')=='critical' else 'badge-warn' if cnn_m.get('severity')=='warning' else 'badge-ok'}">{cnn_m.get('severity','—').upper()}</span></td></tr>
        <tr><td style="color:var(--amber)">TRANSFORMER</td><td>Temperature Trend / Prediction</td><td>{trans.get('trend','—')} / {trans.get('pred_xTemp',0):.3f}</td><td><span class="badge badge-ok">{trans.get('trend','—')}</span></td></tr>
        <tr><td style="color:var(--purple)">GNN</td><td>Issues / Cross-Anomaly</td><td>{', '.join(gnn_m.get('issues',[])) or 'none'} / {'YES' if gnn_m.get('cross_anomaly') else 'NO'}</td><td><span class="badge {'badge-crit' if gnn_m.get('cross_anomaly') else 'badge-warn' if gnn_m.get('issues') else 'badge-ok'}">{('CROSS-ANOMALY' if gnn_m.get('cross_anomaly') else 'ISSUES' if gnn_m.get('issues') else 'NOMINAL')}</span></td></tr>
      </tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">AI MISSION HEALTH NARRATIVE</div>
    <div class="narrative">{para_html}</div>
  </div>

  <div class="rfooter">
    <div>Spacecraft Health AI v2.0 · Chandrayaan-3 ILSA · {now}</div>
    <div>LSTM · CNN · Transformer · GNN · Ollama llama3</div>
    <div>AUTO-GENERATED REPORT — CONFIDENTIAL</div>
  </div>
</div>
</body>
</html>"""
    return html


def build_pdf_report(state: dict, hist_data: list, narrative: str,
                     chart_temp: bytes, chart_accel: bytes,
                     chart_score: bytes, chart_models: bytes) -> bytes:
    """Generate a professional PDF using ReportLab."""
    buf = io.BytesIO()
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    m       = state.get("models", {})
    lstm    = m.get("lstm",  {})
    cnn_m   = m.get("cnn",   {})
    trans   = m.get("transformer", {})
    gnn_m   = m.get("gnn",  {})
    score   = state.get("anomaly_score", 0)
    status  = state.get("status", "UNKNOWN")

    scores = [d["score"] for d in hist_data] if hist_data else [0]
    max_sc = max(scores); avg_sc = float(np.mean(scores))
    crit_n = sum(1 for s in scores if s > CRIT_THRESHOLD)
    warn_n = sum(1 for s in scores if WARN_THRESHOLD < s <= CRIT_THRESHOLD)

    # Colors
    COL_BG    = colors.HexColor("#020508")
    COL_PANEL = colors.HexColor("#070f1a")
    COL_CYAN  = colors.HexColor("#00b4ff")
    COL_GREEN = colors.HexColor("#00ff88")
    COL_AMBER = colors.HexColor("#ffaa00")
    COL_RED   = colors.HexColor("#ff3355")
    COL_PURP  = colors.HexColor("#aa44ff")
    COL_TP    = colors.HexColor("#c8e8ff")
    COL_TS    = colors.HexColor("#4a7a99")
    COL_TL    = colors.HexColor("#2a5a77")
    status_col = COL_RED if status == "ANOMALY" else COL_AMBER if status == "WARNING" else COL_GREEN

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        title="Spacecraft Health AI — Mission Report",
        author="Spacecraft Health AI v2.0",
    )

    styles = getSampleStyleSheet()
    S = {
        "title":   ParagraphStyle("title",   fontName="Helvetica-Bold",  fontSize=18, textColor=COL_CYAN,  spaceAfter=4,  leading=22, alignment=TA_LEFT),
        "sub":     ParagraphStyle("sub",      fontName="Helvetica",       fontSize=8,  textColor=COL_TS,    spaceAfter=12, letterSpacing=2),
        "section": ParagraphStyle("section",  fontName="Helvetica-Bold",  fontSize=9,  textColor=COL_CYAN,  spaceBefore=14, spaceAfter=8),
        "body":    ParagraphStyle("body",     fontName="Helvetica",       fontSize=9,  textColor=COL_TP,    leading=14, spaceAfter=6),
        "mono":    ParagraphStyle("mono",     fontName="Courier",         fontSize=8,  textColor=COL_TP,    leading=12),
        "status":  ParagraphStyle("status",   fontName="Helvetica-Bold",  fontSize=13, textColor=status_col, spaceAfter=6),
    }

    def section_title(text):
        return [
            HRFlowable(width="100%", thickness=1, color=COL_CYAN, spaceAfter=4, spaceBefore=10),
            Paragraph(f"▸  {text}", S["section"]),
        ]

    def chart_image(data: bytes, w=17*cm, h=None):
        img_buf = io.BytesIO(data)
        img = RLImage(img_buf, width=w, height=h or w * 0.3)
        return img

    story = []

    # ── HEADER ──
    story.append(Paragraph("SPACECRAFT HEALTH AI", S["title"]))
    story.append(Paragraph("CHANDRAYAAN-3  ·  ILSA SEISMOMETER  ·  MISSION HEALTH REPORT", S["sub"]))
    story.append(HRFlowable(width="100%", thickness=2, color=COL_CYAN, spaceAfter=10))

    # Meta table
    meta_data = [
        ["Generated", now,              "Engine",   "Spacecraft Health AI v2.0"],
        ["Status",    status,           "Models",   "LSTM · CNN · Transformer · GNN"],
        ["Packets",   str(len(hist_data)), "Dataset", "CH3-ILSA Calibrated CSVs"],
    ]
    meta_tbl = Table(meta_data, colWidths=[3*cm, 5*cm, 3*cm, 6*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Courier"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("TEXTCOLOR",   (0,0),(0,-1),  COL_TL),
        ("TEXTCOLOR",   (2,0),(2,-1),  COL_TL),
        ("TEXTCOLOR",   (1,0),(1,-1),  COL_TP),
        ("TEXTCOLOR",   (3,0),(3,-1),  COL_TP),
        ("ROWBACKGROUNDS", (0,0),(-1,-1), [COL_PANEL, colors.HexColor("#0a1520")]),
        ("GRID",        (0,0),(-1,-1),  0.3, COL_TL),
        ("PADDING",     (0,0),(-1,-1),  5),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 10))

    # Status banner
    story.append(Paragraph(f"MISSION STATUS: {status}  —  Score: {score:.4f}  |  Max: {max_sc:.4f}  |  Avg: {avg_sc:.4f}  |  Criticals: {crit_n}  |  Warnings: {warn_n}", S["status"]))
    story.append(HRFlowable(width="100%", thickness=1, color=status_col, spaceAfter=8))

    # ── ANOMALY SCORE ──
    story += section_title("ANOMALY SCORE HISTORY")
    story.append(chart_image(chart_score, h=5.5*cm))
    story.append(Spacer(1, 6))

    # ── TEMPERATURE ──
    story += section_title("TEMPERATURE SENSORS")
    story.append(chart_image(chart_temp, h=5.5*cm))
    story.append(Spacer(1, 6))

    # ── ACCELERATION ──
    story += section_title("ACCELERATION SENSORS")
    story.append(chart_image(chart_accel, h=8*cm))

    story.append(PageBreak())

    # ── MODEL COMPARISON ──
    story += section_title("ALL 4 MODEL OUTPUTS")
    story.append(chart_image(chart_models, h=5*cm))
    story.append(Spacer(1, 8))

    # Model table
    tbl_data = [
        ["MODEL", "KEY METRIC", "VALUE", "STATUS"],
        ["LSTM",        "Anomaly Score / Z-Score",        f"{lstm.get('score',0):.4f} / {lstm.get('z_score',0):.3f}",   lstm.get('status','—')],
        ["CNN",         "Vibration Pattern / RMS",        f"{cnn_m.get('pattern','—')} / {cnn_m.get('rms_accel',0):.4f}", cnn_m.get('severity','—').upper()],
        ["TRANSFORMER", "Temp Trend / Prediction",        f"{trans.get('trend','—')} / {trans.get('pred_xTemp',0):.3f}",  trans.get('trend','—')],
        ["GNN",         "Issues / Cross-Anomaly",         f"{', '.join(gnn_m.get('issues',[])) or 'none'} / {'YES' if gnn_m.get('cross_anomaly') else 'NO'}", "CROSS-ANOMALY" if gnn_m.get('cross_anomaly') else "NOMINAL"],
    ]
    model_tbl = Table(tbl_data, colWidths=[3.5*cm, 5.5*cm, 5.5*cm, 3.5*cm])
    model_colors = [COL_CYAN, COL_GREEN, COL_AMBER, COL_PURP]
    model_tbl.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTNAME",    (0,1),(-1,-1), "Courier"),
        ("FONTSIZE",    (0,0),(-1,-1), 8),
        ("TEXTCOLOR",   (0,0),(-1,0),  COL_TL),
        ("TEXTCOLOR",   (0,1),(-1,-1), COL_TP),
        *[("TEXTCOLOR", (0, i+1), (0, i+1), c) for i, c in enumerate(model_colors)],
        ("ROWBACKGROUNDS", (0,0),(-1,-1), [COL_PANEL, colors.HexColor("#0a1520")]),
        ("GRID",        (0,0),(-1,-1),  0.3, COL_TL),
        ("PADDING",     (0,0),(-1,-1),  6),
        ("ALIGN",       (0,0),(-1,-1),  "LEFT"),
    ]))
    story.append(model_tbl)
    story.append(Spacer(1, 14))

    # ── NARRATIVE ──
    story += section_title("AI MISSION HEALTH NARRATIVE")
    paras = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    for para in paras:
        lines = para.split("\n")
        for line in lines:
            if line.strip():
                story.append(Paragraph(line, S["body"]))
        story.append(Spacer(1, 6))

    # ── FOOTER LINE ──
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=COL_TL, spaceAfter=4))
    story.append(Paragraph(f"Spacecraft Health AI v2.0  ·  Chandrayaan-3 ILSA  ·  {now}  ·  AUTO-GENERATED REPORT", S["mono"]))

    doc.build(story)
    return buf.getvalue()

# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(title="Spacecraft Health AI", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def serve_dashboard():
    return FileResponse("dashboard/index.html")


@app.get("/ask")
def ask(question: str):
    model  = select_model(question)
    result = model.run(current_state["sensors"])
    answer = explain(result, question, current_state["sensors"])
    return {
        "response":     answer,
        "model_used":   result.get("model"),
        "model_output": result,
        "anomaly_score": current_state["anomaly_score"],
        "status":       current_state["status"],
    }


@app.get("/state")
def get_state():
    return current_state


@app.get("/report/html")
def report_html():
    """Generate and return a full HTML mission report."""
    hist = list(score_history)
    narrative   = generate_narrative(current_state, hist)
    chart_score  = make_chart_anomaly_score(hist)
    chart_temp   = make_chart_temperature(hist)
    chart_accel  = make_chart_acceleration(hist)
    chart_models = make_chart_model_comparison(current_state)
    html = build_html_report(
        current_state, hist, narrative,
        chart_temp, chart_accel, chart_score, chart_models
    )
    filename = f"mission_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/report/pdf")
def report_pdf():
    """Generate and return a professional PDF mission report."""
    hist = list(score_history)
    narrative   = generate_narrative(current_state, hist)
    chart_score  = make_chart_anomaly_score(hist)
    chart_temp   = make_chart_temperature(hist)
    chart_accel  = make_chart_acceleration(hist)
    chart_models = make_chart_model_comparison(current_state)
    pdf_bytes = build_pdf_report(
        current_state, hist, narrative,
        chart_temp, chart_accel, chart_score, chart_models
    )
    filename = f"mission_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    image_type: str = "sensor_chart"
):
    """
    Accept an uploaded image, send to Ollama llava for vision analysis.
    image_type: one of sensor_chart | satellite_image | seismogram | anomaly_plot
    """
    contents = await file.read()
    image_b64 = base64.b64encode(contents).decode()

    # Build context from current mission state
    m = current_state.get("models", {})
    context = {
        "anomaly_score": current_state.get("anomaly_score", 0),
        "status":        current_state.get("status", "UNKNOWN"),
        "lstm_status":   m.get("lstm", {}).get("status", "UNKNOWN"),
        "cnn_pattern":   m.get("cnn",  {}).get("pattern", "UNKNOWN"),
        "trans_trend":   m.get("transformer", {}).get("trend", "UNKNOWN"),
        "gnn_issues":    m.get("gnn",  {}).get("issues", []),
    }

    analysis = explain_image(image_b64, image_type, context)

    return {
        "analysis":      analysis,
        "image_type":    image_type,
        "filename":      file.filename,
        "mission_context": context,
        "model_used":    "llava (vision) / llama3 (fallback)",
    }


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
    print("[ENGINE] Spacecraft Health AI v2.0 starting...")
    print("[ENGINE] Dashboard -> http://localhost:9000")
    print("[ENGINE] WebSocket -> ws://localhost:9000/telemetry")
    print("[ENGINE] AI Query  -> http://localhost:9000/ask?question=...")
    print("[ENGINE] State API -> http://localhost:9000/state")
    print("[ENGINE] HTML Rpt  -> http://localhost:9000/report/html")
    print("[ENGINE] PDF  Rpt  -> http://localhost:9000/report/pdf")
    print("[ENGINE] Img  AI   -> POST http://localhost:9000/analyze-image")
    print(f"[ENGINE] Dataset   -> {len(csv_files)} CSV files loaded")

    loop = asyncio.get_event_loop()
    loop.create_task(telemetry_loop())

    config = uvicorn.Config(app, host="0.0.0.0", port=9000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


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