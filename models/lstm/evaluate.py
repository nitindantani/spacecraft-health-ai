import torch
import numpy as np
from tqdm import tqdm

from core.build_loader import get_loader
from models.lstm.model import LSTMAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD DATA ----------
loader = get_loader(batch_size=64, shuffle=False)

# ---------- LOAD MODEL ----------
model = LSTMAutoencoder().to(DEVICE)
model.load_state_dict(torch.load("outputs/weights/best_lstm.pt", map_location=DEVICE))
model.eval()

print("Evaluating model...\n")

all_window_errors = []
all_sensor_errors = []

with torch.no_grad():
    for x, _ in tqdm(loader):

        x = x.to(DEVICE)

        pred = model(x)

        # 🔹 Window-level error (single value per window)
        window_error = torch.mean((pred - x) ** 2, dim=(1, 2))
        all_window_errors.append(window_error.cpu())

        # 🔹 Per-sensor error (this is what GNN needs)
        sensor_error = torch.mean((pred - x) ** 2, dim=1)
        all_sensor_errors.append(sensor_error.cpu())

# concatenate everything
all_window_errors = torch.cat(all_window_errors).numpy()
all_sensor_errors = torch.cat(all_sensor_errors).numpy()

print("\nEvaluation complete.")
print("Mean window error:", all_window_errors.mean())
print("Std window error:", all_window_errors.std())

# Save properly structured outputs
np.save("outputs/metrics/errors_lstm.npy", all_window_errors)
np.save("outputs/metrics/sensor_errors_lstm.npy", all_sensor_errors)

print("\nSaved:")
print("→ outputs/metrics/errors_lstm.npy")
print("→ outputs/metrics/sensor_errors_lstm.npy")