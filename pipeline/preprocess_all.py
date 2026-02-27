import os
import pandas as pd

DATA_PATH = r"ch3_ilsa/ils/data/calibrated"
SAVE_PATH = r"processed"

os.makedirs(SAVE_PATH, exist_ok=True)

def process_day(day_folder):

    day_path = os.path.join(DATA_PATH, day_folder)

    if not os.path.isdir(day_path):
        return

    files = [f for f in os.listdir(day_path) if f.endswith(".csv")]

    if len(files) == 0:
        return

    print(f"\nProcessing {day_folder} → {len(files)} files")

    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(day_path, file))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # cleaning
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time")
    df = df.drop_duplicates()
    df = df.ffill()

    print("Final shape:", df.shape)

    save_file = os.path.join(SAVE_PATH, f"{day_folder}.csv")
    df.to_csv(save_file, index=False)

    print("Saved →", save_file)


if __name__ == "__main__":

    folders = sorted(os.listdir(DATA_PATH))

    for folder in folders:
        process_day(folder)

    print("\nALL DAYS PROCESSED ✅")
