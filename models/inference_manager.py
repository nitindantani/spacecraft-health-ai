import torch
import numpy as np
import joblib
import os

from models.lstm.model import LSTMAutoencoder
from models.cnn.model import CNNAutoencoder
from models.transformer.model import TransformerAutoencoder
from models.gnn.model import SpacecraftGNN


class InferenceManager:

    def __init__(self,
                 weights_dir="outputs/weights",
                 scaler_path="outputs/scaler.save",
                 window_size=128):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.weights_dir = weights_dir
        self.window_size = window_size

        self.scaler = joblib.load(scaler_path)

        self.sensor_names = [
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

        self.models = {}
        self._load_models()

    def _load_models(self):

        lstm = LSTMAutoencoder()
        lstm.load_state_dict(
            torch.load(os.path.join(self.weights_dir, "best_lstm.pt"),
                       map_location=self.device)
        )
        lstm.to(self.device)
        lstm.eval()
        self.models["lstm"] = lstm

        cnn = CNNAutoencoder()
        cnn.load_state_dict(
            torch.load(os.path.join(self.weights_dir, "best_cnn.pt"),
                       map_location=self.device)
        )
        cnn.to(self.device)
        cnn.eval()
        self.models["cnn"] = cnn

        transformer = TransformerAutoencoder()
        transformer.load_state_dict(
            torch.load(os.path.join(self.weights_dir, "best_transformer.pt"),
                       map_location=self.device)
        )
        transformer.to(self.device)
        transformer.eval()
        self.models["transformer"] = transformer

        gnn = SpacecraftGNN()
        gnn.load_state_dict(
            torch.load(os.path.join(self.weights_dir, "best_gnn.pt"),
                       map_location=self.device)
        )
        gnn.to(self.device)
        gnn.eval()
        self.models["gnn"] = gnn

    def create_windows(self, data):

        windows = []

        for i in range(len(data) - self.window_size + 1):
            windows.append(data[i:i + self.window_size])

        return np.array(windows)

    def predict(self, model_name, data_array):

        model = self.models[model_name]

        data_scaled = self.scaler.transform(data_array)

        windows = self.create_windows(data_scaled)

        if len(windows) == 0:
            raise ValueError("Not enough data for inference window.")

        input_tensor = torch.tensor(
            windows,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():

            output = model(input_tensor)

            error = (output - input_tensor) ** 2

            # mean error per sensor
            sensor_error = torch.mean(error, dim=(0, 1))

            sensor_error = sensor_error.cpu().numpy()

            overall_score = float(sensor_error.mean())

        sensor_errors = dict(zip(self.sensor_names, sensor_error))

        return {
            "overall_score": overall_score,
            "sensor_errors": sensor_errors
        }