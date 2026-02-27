import numpy as np
import matplotlib.pyplot as plt

print("Loading errors...")
errors = np.load("errors.npy")

print("Loading anomaly indices...")
anoms = np.load("anomaly_indices.npy")

plt.figure(figsize=(15,5))
plt.plot(errors, label="Reconstruction Error")

plt.scatter(
    anoms,
    errors[anoms],
    color="red",
    s=8,
    label="Anomalies"
)

plt.title("Anomaly Detection Over Time")
plt.xlabel("Window Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.tight_layout()

plt.savefig("anomaly_plot.png", dpi=200)
plt.show()
