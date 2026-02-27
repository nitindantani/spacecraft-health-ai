import os
import pandas as pd

DATA_PATH = r"ch3_ilsa/ils/data/calibrated"
SAVE_PATH = r"src/proceed/processed_day1.csv"

def load_one_day(day_folder):
    day_path = os.path.join(DATA_PATH, day_folder)

    files = [f for f in os.listdir(day_path) if f.endswith(".csv")]
    print(f"Found {len(files)} files in {day_folder}")

    dfs = []
    for file in files[:5]: #limit first test
        file_path = os.path.join(day_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print("Merged shape:", merged.shape)

    return merged

def clean_data(df):
    # covert time 
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    # drop rows with bad timestamps
    df = df.dropna(subset=["Time"])

    # sort bu time 
    df = df.sort_values("Time")

    #remove duplicates
    df = df.drop_duplicates()

    #fill missing numeric values
    df = df.ffill()

    print("Cleaned shape:", df.shape)

    return df

if __name__ == "__main__":

    df = load_one_day("20230824")

    df = clean_data(df)

    df.to_csv(SAVE_PATH, index=False)
    print("Saved cleaned file")