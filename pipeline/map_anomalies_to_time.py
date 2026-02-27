import os
import numpy as np
import pandas as pd

WINDOW = 128
DATA_DIR = "normalized"

print("Loading anomaly indices...")
anoms = np.load("anomaly_indices.npy")

files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
])

print("Mapping anomalies to timestamps...\n")

current_index = 0

for file in files:
    path = os.path.join(DATA_DIR, file)

    df = pd.read_csv(path, usecols=[0])   # only timestamp column
    rows = len(df)

    usable = rows - WINDOW
    if usable <= 0:
        continue

    # check if anomaly index falls inside this file
    for a in anoms:
        if current_index <= a < current_index + usable:

            local = a - current_index
            timestamp = df.iloc[local + WINDOW][0]

            print(f"Anomaly in file: {file}")
            print(f"Timestamp: {timestamp}")
            print()

    current_index += usable
