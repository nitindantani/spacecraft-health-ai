import os
import pandas as pd
import time
from collections import deque
from models.inference_manager import InferenceManager


class StreamingRealtime:

    def __init__(self, data_root):

        self.data_root = data_root
        self.window_size = 128
        self.buffer = deque(maxlen=self.window_size)

        self.inference = InferenceManager()

        self.feature_columns = [
            "xTemp",
            "zTemp",
            "yTemp",
            "X Coarse Acceleration",
            "Y Coarse Acceleration",
            "Z Coarse Acceleration",
            "X Fine Acceleration",
            "Y Fine Acceleration",
            "Z Fine Acceleration"
        ]

    # ------------------------------------------
    # Risk Interpretation
    # ------------------------------------------
    def interpret_score(self, score):

        if score < 1.0:
            return "SAFE"
        elif score < 3.0:
            return "WARNING"
        else:
            return "CRITICAL"

    # ------------------------------------------
    # Streaming Simulation
    # ------------------------------------------
    def start(self, delay=0.0):

        print("Starting real-time telemetry simulation...")

        date_folders = sorted(os.listdir(self.data_root))

        for date_folder in date_folders:

            folder_path = os.path.join(self.data_root, date_folder)

            if not os.path.isdir(folder_path):
                continue

            csv_files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith(".csv")
            ])

            for csv_file in csv_files:

                file_path = os.path.join(folder_path, csv_file)

                print(f"\nStreaming file: {csv_file}")

                for chunk in pd.read_csv(file_path, chunksize=1000):

                    for _, row in chunk.iterrows():

                        sample = row[self.feature_columns].values
                        self.buffer.append(sample)

                        if len(self.buffer) == self.window_size:

                            window_data = pd.DataFrame(self.buffer).values

                            score = self.inference.predict(
                                "lstm",
                                window_data
                            )

                            status = self.interpret_score(score)

                            print(
                                f"Score: {score:.4f} | Status: {status}"
                            )

                        if delay > 0:
                            time.sleep(delay)


# -------------------------------------------------
# Run Directly
# -------------------------------------------------
if __name__ == "__main__":

    data_root = "data/ch3_ilsa/ils/data/calibrated"

    engine = StreamingRealtime(data_root)

    engine.start(delay=0.0)