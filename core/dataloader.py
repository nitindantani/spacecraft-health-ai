import os
import pandas as pd
import torch

DATA_PATH = "processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_day(day_file):

    path = os.path.join(DATA_PATH, day_file)

    df = pd.read_csv(path)

    # drop time column for model
    df = df.drop(columns=["Time"])

    # convert to tensor
    tensor = torch.tensor(df.values, dtype=torch.float32)

    return tensor


def day_generator():

    files = sorted(os.listdir(DATA_PATH))

    for file in files:
        if file.endswith(".csv"):
            print(f"Loading {file}")
            yield load_day(file)


if __name__ == "__main__":

    for day_tensor in day_generator():

        # send to GPU safely
        day_tensor = day_tensor.to(DEVICE)

        print("Tensor shape:", day_tensor.shape)
        print("Device:", day_tensor.device)

        # simulate model step
        mean = day_tensor.mean()

        print("Mean value:", mean.item())
