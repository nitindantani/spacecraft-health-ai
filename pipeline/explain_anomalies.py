import torch
import numpy as np
import pandas as pd
from models.lstm.model import LSTMAutoencoder
from core.build_loader import get_loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = LSTMAutoencoder().to(DEVICE)
model.load_state_dict(torch.load("best_lstm.pt", map_location=DEVICE))
model.eval()

print("Loading anomaly indices...")
anoms = np.load("anomaly_indices.npy")

loader = get_loader(batch_size=1, shuffle=False)

print("\nExplaining anomalies...\n")

for i, (x, y) in enumerate(loader):

    if i not in anoms:
        continue

    x = x.to(DEVICE)

    with torch.no_grad():
        out = model(x)

    # compute feature-wise error
    err = torch.abs(out - y.to(DEVICE))
    err = err.cpu().numpy()[0]

    # find top sensors
    top = np.argsort(err)[::-1][:3]

    print(f"Anomaly window: {i}")
    print("Top contributing features:", top)
    print("Errors:", err[top])
    print()
