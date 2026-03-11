import os
import pandas as pd
import time


class TelemetryStreamer:

    def __init__(self, root_folder):

        self.root_folder = root_folder

        self.day_folders = sorted(
            [
                os.path.join(root_folder, d)
                for d in os.listdir(root_folder)
                if os.path.isdir(os.path.join(root_folder, d))
            ]
        )

    def stream(self):

        for day in self.day_folders:

            print(f"\n📅 Streaming day folder: {day}")

            csv_files = sorted(
                [
                    os.path.join(day, f)
                    for f in os.listdir(day)
                    if f.endswith(".csv")
                ]
            )

            for file in csv_files:

                print("Streaming file:", file)

                try:
                    df = pd.read_csv(file)
                except Exception as e:
                    print("Error reading:", file, e)
                    continue

                for _, row in df.iterrows():

                    telemetry = row.to_dict()

                    yield telemetry

                    time.sleep(0.01)  # simulate real-time telemetry