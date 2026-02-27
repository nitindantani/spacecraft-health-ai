import torch
import numpy as np
from core.build_loader import get_loader
from models.lstm.model import LSTMAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = LSTMAutoencoder().to(DEVICE)
model.load_state_dict(torch.load("best_lstm.pt", map_location=DEVICE))
model.eval()

print("Loading anomalies...")
anomalies = np.load("outputs/metrics/anomaly_indices.npy")

loader = get_loader(batch_size=1, shuffle=False)

print("Explaining anomalies...\n")

sensor_names = [
    "Temp X", "Temp Y", "Temp Z",
    "Accel coarse X", "Accel coarse Y", "Accel coarse Z",
    "Accel fine X", "Accel fine Y", "Accel fine Z"
]

count = 0

for i, (x, _) in enumerate(loader):

    if i not in anomalies:
        continue

    x = x.to(DEVICE)

    with torch.no_grad():
        recon = model(x)

    error = (recon - x).abs().mean(dim=1).cpu().numpy()[0]

    top = np.argsort(error)[-3:][::-1]

    print(f"Anomaly window: {i}")
    for t in top:
        print(f"   → {sensor_names[t]} abnormal (error={error[t]:.4f})")

    print()

    count += 1
    if count >= 10:
        break
