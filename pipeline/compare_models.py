import numpy as np
import matplotlib.pyplot as plt

print("Loading errors...")

lstm = np.load("outputs/metrics/errors.npy")
cnn  = np.load("outputs/metrics/errors_cnn.npy")

# use same length
N = min(len(lstm), len(cnn))

lstm = lstm[:N]
cnn  = cnn[:N]

print("Plotting comparison...")

plt.figure(figsize=(12,5))

plt.plot(lstm, label="LSTM error", alpha=0.7)
plt.plot(cnn, label="CNN error", alpha=0.7)

plt.title("Model Reconstruction Error Comparison")
plt.xlabel("Window index")
plt.ylabel("Error")
plt.legend()

plt.savefig("outputs/plots/model_comparison.png")
plt.show()
