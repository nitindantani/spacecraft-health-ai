import os
import pandas as pd
from datetime import timedelta
from models.inference_manager import InferenceManager


class RealtimeController:

    def __init__(self, data_root="data/ch3_ilsa/ils/data/calibrated"):

        self.data_root = data_root
        self.inference = InferenceManager()

        # Feature columns used during training
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

    # -------------------------------------------------
    # Auto Model Selection Logic
    # -------------------------------------------------
    def select_model(self, hours):

        if hours <= 3:
            return "cnn"
        elif hours <= 12:
            return "lstm"
        else:
            return "transformer"

    # -------------------------------------------------
    # Load Relevant Files Dynamically
    # -------------------------------------------------
    def load_last_hours(self, hours):

        all_data = []

        # Scan date folders
        for date_folder in sorted(os.listdir(self.data_root)):

            folder_path = os.path.join(self.data_root, date_folder)

            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):

                if not file.endswith(".csv"):
                    continue

                file_path = os.path.join(folder_path, file)

                df = pd.read_csv(file_path)

                if "Time" not in df.columns:
                    continue

                df["Time"] = pd.to_datetime(df["Time"], utc=True)

                all_data.append(df)

        # Combine only relevant recent data
        full_df = pd.concat(all_data).sort_values("Time")

        latest_time = full_df["Time"].max()
        threshold = latest_time - timedelta(hours=hours)

        filtered = full_df[full_df["Time"] >= threshold]

        return filtered

    # -------------------------------------------------
    # Run Anomaly Detection
    # -------------------------------------------------
    def detect_anomaly(self, hours):

        model_name = self.select_model(hours)

        df = self.load_last_hours(hours)

        if len(df) < 128:
            raise ValueError("Not enough telemetry points for inference.")

        data_array = df[self.feature_columns].values

        anomaly_score = self.inference.predict(model_name, data_array)

        return {
            "model_used": model_name,
            "hours_analyzed": hours,
            "data_points": len(df),
            "anomaly_score": anomaly_score
        }