import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "processed"
SAVE_PATH = "normalized"

os.makedirs(SAVE_PATH, exist_ok=True)

scaler = StandardScaler()

all_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".csv")])

print("Fitting scaler on all data...")

# --- first pass: fit scaler on all data
for file in all_files:
    df = pd.read_csv(os.path.join(DATA_PATH, file))
    df = df.drop(columns=["Time"])
    scaler.partial_fit(df.values)

# save scaler
joblib.dump(scaler, "scaler.save")
print("Scaler saved")

# --- second pass: transform and save
print("Applying normalization...")

for file in all_files:
    print("Processing", file)

    df = pd.read_csv(os.path.join(DATA_PATH, file))
    time_col = df["Time"]
    features = df.drop(columns=["Time"])

    scaled = scaler.transform(features.values)

    new_df = pd.DataFrame(scaled, columns=features.columns)
    new_df.insert(0, "Time", time_col)

    new_df.to_csv(os.path.join(SAVE_PATH, file), index=False)

print("ALL FILES NORMALIZED ✅")
