import numpy as np
import torch

print("Loading real sensor errors...")

sensor_errors = np.load("outputs/metrics/sensor_errors_lstm.npy")

# limit size
sensor_errors = sensor_errors[:1000]

# normalize per sensor
mean = sensor_errors.mean(axis=0)
std = sensor_errors.std(axis=0) + 1e-8

sensor_errors = (sensor_errors - mean) / std

# pattern normalization per time window
norms = np.linalg.norm(sensor_errors, axis=1, keepdims=True) + 1e-6
sensor_errors = sensor_errors / norms

sensor_errors = sensor_errors - sensor_errors.mean(axis=1, keepdims=True)

features = torch.tensor(sensor_errors, dtype=torch.float32)

print("Feature shape:", features.shape)

torch.save(features, "outputs/metrics/graph_features.pt")
print("Saved -> outputs/metrics/graph_features.pt")