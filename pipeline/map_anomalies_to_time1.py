import numpy as np
import pandas as pd

print("Loading anomalies...")
anoms = np.load("outputs/metrics/anomaly_indices.npy")

print("Loading first CSV to get timestamps...")
df = pd.read_csv("data/normalized/20230824.csv")

# assume timestamp column exists
# change name if needed
time_col = df.columns[0]

print("\nMapping anomalies to real time:\n")

for a in anoms[:20]:
    if a < len(df):
        t = df.iloc[a][time_col]
        print(f"Window {a} → time = {t}")
