import joblib
import numpy as np
import pandas as pd
import os

print("=== SCALER CHECK ===\n")

scaler_path = "outputs/scaler.save"
if not os.path.exists(scaler_path):
    print(f"❌ scaler.save NOT FOUND at {scaler_path}")
    print("Run pipeline/normalize.py first!")
else:
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded: {type(scaler).__name__}")
    print(f"   n_features: {scaler.n_features_in_}")
    print(f"   mean_: {scaler.mean_}")
    print(f"   scale_: {scaler.scale_}")

print("\n=== RAW SENSOR VALUES (first CSV file) ===\n")

import glob
files = sorted(glob.glob(r"data\ch3_ilsa\ils\data\calibrated\**\*.csv", recursive=True))
if files:
    df = pd.read_csv(files[0])
    print(f"File: {files[0]}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst row values:")
    print(df.iloc[0])
    print(f"\nStats for numeric columns:")
    print(df.describe())
else:
    print("No CSV files found")

print("\n=== NORMALIZED DATA CHECK ===\n")
norm_files = sorted(glob.glob(r"data\normalized\*.csv"))
if norm_files:
    df2 = pd.read_csv(norm_files[0])
    print(f"Normalized file: {norm_files[0]}")
    print(df2.describe())
else:
    print("No normalized files found at data/normalized/")
    # Check other locations
    for loc in ["normalized", "outputs/normalized", "data/normalized"]:
        if os.path.exists(loc):
            print(f"Found folder: {loc}")
            print(os.listdir(loc)[:5])