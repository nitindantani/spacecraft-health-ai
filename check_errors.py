import numpy as np
import os

print("=== MODEL ERROR RANGES ===\n")

files = {
    "LSTM":        "outputs/metrics/errors_lstm.npy",
    "CNN":         "outputs/metrics/errors_cnn.npy",
    "Transformer": "outputs/metrics/errors_transformer.npy",
}

for name, path in files.items():
    if os.path.exists(path):
        e = np.load(path)
        print(f"{name}:")
        print(f"  min  = {e.min():.8f}")
        print(f"  max  = {e.max():.8f}")
        print(f"  mean = {e.mean():.8f}")
        print(f"  std  = {e.std():.8f}")
        print(f"  p95  = {np.percentile(e, 95):.8f}")
        print(f"  p99  = {np.percentile(e, 99):.8f}")
        print()
    else:
        print(f"{name}: FILE NOT FOUND at {path}\n")

# Also check final scores if they exist
final = "outputs/metrics/final_scores.npy"
if os.path.exists(final):
    e = np.load(final)
    print(f"Final combined scores:")
    print(f"  min  = {e.min():.8f}")
    print(f"  max  = {e.max():.8f}")
    print(f"  mean = {e.mean():.8f}")
    print(f"  p99  = {np.percentile(e, 99):.8f}")