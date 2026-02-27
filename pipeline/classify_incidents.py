import numpy as np

# load anomalies + explanations
anoms = np.load("outputs/metrics/anomaly_indices.npy")

print("\nClassifying incidents...\n")

def classify(features):
    # very simple rules (we upgrade later)
    if any("Temp" in f for f in features):
        return "Thermal anomaly"
    if any("Accel" in f for f in features):
        return "Vibration anomaly"
    return "Unknown anomaly"

# Example fake mapping for now
# (later we connect with your real feature explanations)
for i, idx in enumerate(anoms[:10]):

    # pretend features detected (we replace later)
    if i % 3 == 0:
        feats = ["Temp X", "Temp Y"]
    elif i % 3 == 1:
        feats = ["Accel coarse Y", "Accel fine Y"]
    else:
        feats = ["Temp Z", "Accel coarse Z"]

    label = classify(feats)

    print(f"Incident around window {idx}:")
    print("  Type:", label)
    print("  Sensors:", ", ".join(feats))
    print()
