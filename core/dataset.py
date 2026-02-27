import os
import pandas as pd
import torch
from torch.utils.data import Dataset


MODE = "debug"

LIMITS = {
    "debug": 500_000,
    "train": 5_000_000,
    "full": None
}


class SatelliteDataset(Dataset):

    def __init__(self, data_dir):
        print("Indexing CSV files...")

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv")
        ])

        self.window = 128
        self.index = []

        self.cache = {}   # 🔥 stores loaded files

        limit = LIMITS[MODE]
        total = 0

        for file in self.files:

            # Count rows quickly
            n_rows = sum(1 for _ in open(file)) - 1
            usable = n_rows - self.window

            if usable <= 0:
                continue

            for i in range(usable):
                self.index.append((file, i))
                total += 1

                if limit and total >= limit:
                    break

            if limit and total >= limit:
                break

        print(f"Total samples: {len(self.index):,}")

    # ---------------------------
    # Load file only once
    # ---------------------------
    def _load_file(self, file):

        if file not in self.cache:

            df = pd.read_csv(file, engine="python", on_bad_lines="skip")

            df = df.select_dtypes(include=['number'])
            df = df.astype("float32")

            tensor = torch.tensor(df.values, dtype=torch.float32)

            self.cache[file] = tensor

        return self.cache[file]

    # ---------------------------
    def __len__(self):
        return len(self.index)

    # ---------------------------
    def __getitem__(self, idx):

        file, start = self.index[idx]

        data = self._load_file(file)

        x = data[start:start+self.window]
        y = data[start+self.window]

        return x, y
