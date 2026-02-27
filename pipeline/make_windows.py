import os
import glob
import torch
import pandas as pd
from tqdm import tqdm

# ===== SETTINGS =====
INPUT_DIR = "normalized"      # folder where normalized CSVs are
OUTPUT_DIR = "windows_pt"     # folder to save tensors
WINDOW = 128                  # your choice
STRIDE = 1                    # sliding step

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))

print(f"Found {len(files)} normalized files")

total_windows = 0

for file in files:
    print(f"\nProcessing {os.path.basename(file)}")

    df = pd.read_csv(file)

    # drop time column if exists
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    data = torch.tensor(df.values, dtype=torch.float32)

    N = data.shape[0]

    if N < WINDOW:
        print("Skipped (too small)")
        continue

    windows = []

    for i in range(0, N - WINDOW, STRIDE):
        win = data[i:i+WINDOW]
        windows.append(win)

    windows = torch.stack(windows)

    save_name = os.path.basename(file).replace(".csv", ".pt")
    save_path = os.path.join(OUTPUT_DIR, save_name)

    torch.save(windows, save_path)

    print(f"Saved {windows.shape} → {save_name}")

    total_windows += windows.shape[0]

print("\nTOTAL WINDOWS:", total_windows)
print("WINDOW GENERATION DONE ✅")
