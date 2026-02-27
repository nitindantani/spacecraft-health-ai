import numpy as np
import pandas as pd
import torch

from core.sensor_graph import edge_index
from models.gnn.model import SpacecraftGNN


print("Loading anomaly data...")

scores = np.load("outputs/metrics/final_scores.npy")
anoms  = np.load("outputs/metrics/final_anomalies.npy")

# load real per-sensor errors
sensor_errors = np.load("outputs/metrics/sensor_errors_lstm.npy")

# normalize anomaly scores
scores = (scores - scores.mean()) / (scores.std() + 1e-8)
scores = np.abs(scores)


print("Grouping anomalies into incidents...")

incidents = []
current = [anoms[0]]

for i in range(1, len(anoms)):
    if anoms[i] - anoms[i-1] < 20:
        current.append(anoms[i])
    else:
        incidents.append(current)
        current = [anoms[i]]

incidents.append(current)

print(f"Detected {len(incidents)} incidents")


print("Loading GNN model...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SpacecraftGNN().to(DEVICE)
model.load_state_dict(torch.load("outputs/weights/best_gnn.pt", map_location=DEVICE))
model.eval()

edge_index = edge_index.to(DEVICE)


print("\nScoring incidents...\n")

results = []

for idx, inc in enumerate(incidents):

    # ---------- anomaly strength ----------
    window_errors = scores[inc]
    anomaly_strength = float(window_errors.mean())

    dataset_energy = scores.mean() + 1e-6
    anomaly_strength = anomaly_strength / (3 * dataset_energy)
    anomaly_strength = float(np.clip(anomaly_strength, 0, 3))

    print("DEBUG anomaly_strength:", anomaly_strength)

    # ---------- duration ----------
    duration = len(inc)

    # ---------- REAL SENSOR VECTOR ----------
    sensor_vec = sensor_errors[inc].mean(axis=0)
    print("DEBUG sensor_vec:", sensor_vec)
    
    node_vec = torch.tensor(
        sensor_vec,
        dtype=torch.float32
    ).unsqueeze(-1).to(DEVICE)

    # ---------- GNN RISK ----------
    with torch.no_grad():
        gnn_out = model(node_vec, edge_index)
        graph_risk = float(gnn_out.abs().mean().item())

    graph_risk = graph_risk / (3 * dataset_energy)
    graph_risk = float(np.clip(graph_risk, 0, 3))

    print("DEBUG graph_risk:", graph_risk)

    # ---------- FINAL SEVERITY ----------
    severity = (
        0.55 * anomaly_strength +
        0.30 * graph_risk +
        0.15 * (duration / 50)
    )

    # ---------- LABEL ----------
    if severity < 0.15:
        label = "LOW"
    elif severity < 0.35:
        label = "MEDIUM"
    elif severity < 0.60:
        label = "HIGH"
    else:
        label = "CRITICAL"

    results.append({
        "incident": idx+1,
        "duration_windows": duration,
        "anomaly_strength": anomaly_strength,
        "graph_risk": graph_risk,
        "severity_score": severity,
        "label": label
    })


# ---------- SORT ----------
results = sorted(results, key=lambda x: x["severity_score"], reverse=True)


print("\nTOP INCIDENTS:\n")

for r in results[:5]:
    print(
        f"Incident {r['incident']} | "
        f"Severity={r['severity_score']:.3f} | "
        f"Level={r['label']} | "
        f"Duration={r['duration_windows']}"
    )


# ---------- SAVE ----------
df = pd.DataFrame(results)
df.to_csv("outputs/metrics/incidents_scored.csv", index=False)

print("\nSaved → outputs/metrics/incidents_scored.csv")