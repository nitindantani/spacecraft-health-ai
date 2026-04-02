# 🛰 Spacecraft Health AI
### Real-time anomaly detection for Chandrayaan-3 ILSA seismometer data

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Ollama-llama3%20%7C%20llava-black?style=flat-square"/>
  <img src="https://img.shields.io/badge/ISRO-Chandrayaan--3-orange?style=flat-square"/>
</p>

<p align="center">
  A production-grade AI system that monitors the Chandrayaan-3 ILSA (Instrument for Lunar Seismic Activity) seismometer in real time — detecting anomalies, classifying lunar seismic events, and providing natural language explanations via a conversational chat interface.
</p>
<p>
  Live : https://huggingface.co/spaces/nitindantani/spacecraft-health-ai
</p>
---

## What It Does

The system continuously streams 599 calibrated CSV files from the ILSA instrument, runs them through 4 real trained PyTorch models, and surfaces anomalies, seismic events, and health insights in a single-frame chat dashboard — no commands needed, just ask.

```
"Show temperature graph"           → renders live chart inline
"Why is there an anomaly?"         → AI explains root cause with sensor values
"What should I do now?"            → knowledge base gives specific actions
"Plot seismic events"              → scatter chart of all detected events
"Generate PDF report"              → full mission health report downloads
[upload any image]                 → llava vision AI analyzes it
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CHANDRAYAAN-3 ILSA DATA                   │
│              599 calibrated CSV files · 9 sensors            │
└──────────────────────────┬──────────────────────────────────┘
                           │ 1 packet/sec streaming
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     mission_engine.py                        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │    LSTM     │  │     CNN     │  │    Transformer     │  │
│  │ Autoencoder │  │ Autoencoder │  │    Autoencoder     │  │
│  │ 215KB .pt   │  │ 111KB .pt   │  │    2.2MB .pt       │  │
│  │ anomaly     │  │ seismic     │  │    temp forecast   │  │
│  │ score       │  │ event type  │  │    trend           │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────┬──────────┘  │
│         │                │                    │              │
│         └────────────────┼────────────────────┘              │
│                          │                                   │
│                   ┌──────▼──────┐                            │
│                   │     GNN     │                            │
│                   │ SpacecraftGNN│                           │
│                   │  4.4KB .pt  │                            │
│                   │ sensor graph│                            │
│                   └──────┬──────┘                            │
│                          │                                   │
│              ┌───────────▼───────────┐                       │
│              │   STA/LTA Detector    │                       │
│              │  Real seismology algo │                       │
│              │  Allen (1982)         │                       │
│              └───────────┬───────────┘                       │
│                          │                                   │
│              ┌───────────▼───────────┐                       │
│              │   Knowledge Base      │                       │
│              │  10 anomaly types     │                       │
│              │  Causes + Solutions   │                       │
│              └───────────┬───────────┘                       │
└──────────────────────────┼──────────────────────────────────┘
                           │ WebSocket + REST API
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    dashboard/index.html                      │
│                                                              │
│   Left Sidebar          │      Chat Interface               │
│   ─────────────         │      ────────────────             │
│   • Anomaly score ring  │      • Natural language queries   │
│   • Temperature chart   │      • Inline chart generation    │
│   • Acceleration chart  │      • Image analysis (llava)     │
│   • STA/LTA chart       │      • Auto-alerts                │
│   • 4 model outputs     │      • PDF/HTML reports           │
│   • Subsystem health    │      • Knowledge base diagnosis   │
│   • Session stats       │      • Multi-turn memory          │
└─────────────────────────────────────────────────────────────┘
```

---

## Models

All 4 models are **real trained PyTorch autoencoders** trained on Chandrayaan-3 ILSA calibrated data.

| Model | Architecture | Size | Purpose |
|---|---|---|---|
| **LSTM** | `LSTMAutoencoder` · encoder-decoder · hidden_dim=64 | 215 KB | Anomaly score from reconstruction error |
| **CNN** | `CNNAutoencoder` · 1D Conv · 3 encoder + 3 decoder layers | 111 KB | Seismic pattern classification |
| **Transformer** | `TransformerAutoencoder` · 2 layers · 4 heads · d_model=64 | 2.2 MB | Temperature trend forecasting |
| **GNN** | `SpacecraftGNN` · GCNConv · 9 nodes · 24 edges | 4.4 KB | Cross-sensor graph anomaly |

**Training approach:** Unsupervised reconstruction — anomaly score = mean reconstruction error across all 9 sensor channels. Thresholds calibrated from your actual error distribution (p95 = normal ceiling, p99 = anomaly boundary).

**Seismic detection:** STA/LTA (Short-Term Average / Long-Term Average) ratio detector — the industry-standard algorithm used by InSight, Apollo ALSEP, and Chandrayaan-3 ILSA. Trigger threshold: 3.0, detrigger: 1.5.

---

## Sensor Graph (GNN)

The GNN models relationships between all 9 sensors using a graph with 24 edges:

```
Temp X ──── Temp Y ──── Temp Z
  │                        │
  └── Coarse X          Coarse Z
      Coarse X ─ Coarse Y ─ Coarse Z
      Fine X   ─ Fine Y   ─ Fine Z
```

Cross-links: `Temp X ↔ Coarse X`, `Temp Y ↔ Coarse Y`, `Temp Z ↔ Coarse Z`

---

## Knowledge Base

10 anomaly types with physical causes and specific recommended actions:

| Type | Risk | Physical Cause |
|---|---|---|
| `thermal_high_xtemp` | MEDIUM | Lunar noon heating, instrument heater |
| `thermal_all_sensors` | HIGH | Lunar day entry, power dissipation |
| `thermal_falling` | MEDIUM | Lunar night entry |
| `seismic_deep_moonquake` | LOW | Earth tidal forces, 700–1200 km depth |
| `seismic_shallow_moonquake` | HIGH | Tectonic stress, magnitude up to 5.5 |
| `seismic_meteorite_impact` | LOW-MEDIUM | Meteoroid impact near deployment site |
| `seismic_thermal_cracking` | LOW | Sunrise/sunset 300°C/hr temperature gradient |
| `seismic_thermal_coupling` | MEDIUM-HIGH | Thermoelastic event |
| `high_coarse_acceleration` | HIGH | Physical disturbance to instrument |
| `critical_composite` | CRITICAL | Multiple simultaneous failures |

---

## Installation

### Prerequisites

```bash
# Python environment
conda create -n ilsa_gnn python=3.10
conda activate ilsa_gnn

# PyTorch (CPU)
pip install torch torchvision

# torch-geometric for GNN
pip install torch-geometric

# Core dependencies
pip install fastapi uvicorn requests numpy pandas scipy
pip install matplotlib pillow reportlab joblib scikit-learn
pip install torch-geometric

# Ollama (local LLM)
# Download from https://ollama.com
ollama pull llama3      # text reasoning (~4GB)
ollama pull llava       # vision analysis (~4GB)
```

### Data

Download Chandrayaan-3 ILSA calibrated data from ISSDC:
```
https://pradan.issdc.gov.in/ch3/
```

Place at:
```
data/ch3_ilsa/ils/data/calibrated/
  ├── 20230824/
  │   ├── ch3_ils_nop_calib_20230824t101208872_d_accln.csv
  │   └── ...
  └── 20230825/
      └── ...
```

### Train Models

```bash
# 1. Preprocess raw data
python pipeline/preprocess_all.py

# 2. Normalize (fit + save scaler)
python pipeline/normalize.py

# 3. Train all 4 models
python models/lstm/train.py
python models/cnn/train.py
python models/transformer/train.py

# 4. Evaluate to generate error files
python models/lstm/evaluate.py
python models/cnn/evaluate.py
python models/transformer/evaluate.py

# 5. Build graph features for GNN
python core/build_graph_features.py

# 6. Train GNN
python models/gnn/train.py
```

Trained weights are saved to `outputs/weights/`:
```
outputs/weights/
  ├── best_lstm.pt         (215 KB)
  ├── best_cnn.pt          (111 KB)
  ├── best_transformer.pt  (2.2 MB)
  └── best_gnn.pt          (4.4 KB)
```

### Run Dashboard

```bash
cd G:\Pro_NML
python mission_engine.py
```

Open `http://localhost:9000` in your browser.

---

## Usage

### Chat Interface

Just type naturally — the AI routes your question to the right system:

| Say this | What happens |
|---|---|
| `show temperature graph` | Live temp chart renders as chat bubble |
| `why is there an anomaly?` | AI explains root cause with sensor values |
| `what should I do now?` | Knowledge base gives specific actions |
| `plot seismic events` | Scatter chart of all detected moonquakes |
| `mission health summary` | Full health card with all 6 metrics |
| `generate PDF report` | Full report with charts + AI narrative |
| `🔬 diagnose anomaly` | Instant structured diagnosis card |
| Upload any image | llava vision AI analyzes it |
| Ask about uploaded image | `what is this?` / `who is this person?` |

### API Endpoints

```
GET  /                              Dashboard UI
WS   /telemetry                     Live telemetry stream (WebSocket)
GET  /ask?question=...              Natural language query → llama3
GET  /diagnose                      Knowledge base diagnosis for current state
GET  /state                         Full current state JSON
GET  /history                       Session history (timeseries)
GET  /history?format=events         Seismic event log
GET  /history?format=summary        Session statistics
GET  /history?format=snapshots      Full state snapshots
GET  /report/html                   Generate + download HTML report
GET  /report/pdf                    Generate + download PDF report
POST /analyze-image?image_type=...  Vision AI image analysis
```

---

## Project Structure

```
spacecraft-health-ai/
├── mission_engine.py          # Main backend — FastAPI + all 4 models
├── dashboard/
│   └── index.html             # Single-frame chat dashboard
│
├── models/
│   ├── lstm/
│   │   ├── model.py           # LSTMAutoencoder architecture
│   │   ├── train.py           # Training loop
│   │   └── evaluate.py        # Evaluation + error export
│   ├── cnn/
│   │   ├── model.py           # CNNAutoencoder architecture
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── transformer/
│   │   ├── model.py           # TransformerAutoencoder architecture
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── gnn/
│   │   ├── model.py           # SpacecraftGNN (GCNConv) architecture
│   │   └── train.py
│   └── inference_manager.py   # Loads all 4 models + runs inference
│
├── pipeline/
│   ├── preprocess_all.py      # Raw CSV cleaning
│   ├── normalize.py           # StandardScaler fit + transform
│   ├── make_windows.py        # Sliding window generation (128 samples)
│   ├── detect_anomalies.py    # Threshold-based anomaly detection
│   ├── final_detector.py      # Combined CNN + Transformer scoring
│   └── score_incidents.py     # GNN-based incident severity scoring
│
├── core/
│   ├── dataset.py             # SatelliteDataset (PyTorch Dataset)
│   ├── build_loader.py        # DataLoader factory
│   ├── sensor_graph.py        # GNN edge index (9 nodes, 24 edges)
│   ├── mission_core.py        # Risk fusion, subsystem aggregation
│   ├── mission_state_manager.py
│   └── subsystem_analyzer.py
│
└── outputs/
    ├── weights/               # Trained model .pt files
    ├── metrics/               # Evaluation outputs (.npy files)
    └── scaler.save            # Fitted StandardScaler
```

---

## Data Format

Each CSV file contains 9 sensor channels from the ILSA seismometer:

| Column | Description | Typical Range |
|---|---|---|
| `xTemp` | X-axis temperature | 10–46 °C |
| `yTemp` | Y-axis temperature | 10–46 °C |
| `zTemp` | Z-axis temperature | 10–46 °C |
| `X Coarse Acceleration` | Coarse IMU X | ±2 m/s² |
| `Y Coarse Acceleration` | Coarse IMU Y | ±2 m/s² |
| `Z Coarse Acceleration` | Coarse IMU Z | ±2 m/s² |
| `X Fine Acceleration` | Fine seismometer X | ±0.2 m/s² |
| `Y Fine Acceleration` | Fine seismometer Y | ±0.2 m/s² |
| `Z Fine Acceleration` | Fine seismometer Z | ±0.2 m/s² |

---

## References

- **ILSA Instrument**: Kumar, P. et al. (2023). "Instrument for Lunar Seismic Activity (ILSA) on Chandrayaan-3". *Current Science*.
- **STA/LTA Algorithm**: Allen, R.V. (1982). "Automatic earthquake recognition and timing from single traces". *BSSA*.
- **Lunar Seismology**: Nakamura, Y. (1977). "Seismic energy transmission in the lunar surface zone". *PEPI*.
- **InSight SEIS**: Lognonné, P. et al. (2020). "Constraints on the shallow elastic and anelastic structure of Mars from InSight seismic data". *Nature Geoscience*.
- **Dataset**: ISRO ISSDC — Chandrayaan-3 ILSA calibrated seismic data.

---

## License

**Copyright © 2024 Nitin Dantani. All Rights Reserved.**

This project is publicly visible for portfolio and educational purposes only.

**You may NOT:**
- Use this code in any project (personal, academic, or commercial)
- Copy, modify, or distribute any part of this codebase
- Deploy this system without explicit written permission from the author

**You may:**
- View and read the code
- Reference this project in academic citations (with attribution)

For permissions, collaborations, or licensing inquiries:
📧 Contact via [GitHub](https://github.com/nitindantani)

---

<p align="center">
  Built with real Chandrayaan-3 ILSA data · PyTorch · FastAPI · Ollama · llava
</p>
