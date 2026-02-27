import numpy as np

print("Loading model errors...")

cnn = np.load("outputs/metrics/errors_cnn.npy")
trans = np.load("outputs/metrics/errors_transformer.npy")

N = min(len(cnn), len(trans))
cnn = cnn[:N]
trans = trans[:N]

# combine scores (Transformer stronger)
score = 0.4 * cnn + 0.6 * trans

mean = score.mean()
std = score.std()
threshold = mean + 3 * std

print("\nCombined stats:")
print("Mean:", mean)
print("Std:", std)
print("Threshold:", threshold)

anomalies = np.where(score > threshold)[0]

print("\nFinal anomalies:", len(anomalies))

np.save("outputs/metrics/final_scores.npy", score)
np.save("outputs/metrics/final_anomalies.npy", anomalies)

print("Saved final anomaly results.")
