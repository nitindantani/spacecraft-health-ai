import numpy as np

print("Loading reconstruction errors...")
errors = np.load("errors.npy")

mean = errors.mean()
std = errors.std()

threshold = mean + 3 * std

print(f"Mean: {mean:.6f}")
print(f"Std : {std:.6f}")
print(f"Threshold: {threshold:.6f}")

# detect anomalies
anomalies = np.where(errors > threshold)[0]

print(f"\nTotal windows: {len(errors)}")
print(f"Anomalies detected: {len(anomalies)}")

# save anomaly indices
np.save("anomaly_indices.npy", anomalies)

print("Saved anomaly indices → anomaly_indices.npy")
