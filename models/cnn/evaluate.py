import torch
import numpy as np
from tqdm import tqdm

from core.build_loader import get_loader
from models.cnn.model import CNNAutoencoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading data...")
loader = get_loader(batch_size=64, shuffle=False)

print("Loading model...")
model = CNNAutoencoder().to(DEVICE)
model.load_state_dict(torch.load("outputs/weights/best_cnn.pt"))
model.eval()

errors = []

print("Evaluating CNN...")

with torch.no_grad():
    for x, _ in tqdm(loader):
        x = x.to(DEVICE)

        out = model(x)

        # reconstruction error per window
        err = torch.mean((x - out) ** 2, dim=(1,2))
        errors.extend(err.cpu().numpy())

errors = np.array(errors)

print("\nEvaluation complete.")
print("Mean error:", errors.mean())
print("Std error:", errors.std())
print("Max error:", errors.max())

np.save("outputs/metrics/errors_cnn.npy", errors)
print("Saved -> outputs/metrics/errors_cnn.npy")
