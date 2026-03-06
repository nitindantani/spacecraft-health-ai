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
        self.models = {}

        self._load_models()

    # ----------------------------------------------------
    # Load Models
    # ----------------------------------------------------
    def _load_models(self):

        cnn = CNNAutoencoder()
        cnn.load_state_dict(
            torch.load(os.path.join(self.weights_dir, "best_cnn.pt"),
                       map_location=self.device)
        )
        cnn.to(self.device)
        cnn.eval()
        self.models["cnn"] = cnn

        lstm = LSTMAutoencoder()
        lstm.load_state_dict(
            torch.load(os.path.join(self.weights_dir, "best_lstm.pt"),
                       map_location=self.device)
        )
        lstm.to(self.device)
        lstm.eval()
        self.models["lstm"] = lstm

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

    # ----------------------------------------------------
    # Create Sliding Windows
    # ----------------------------------------------------
    def create_windows(self, data):

        windows = []
        for i in range(len(data) - self.window_size + 1):
            windows.append(data[i:i + self.window_size])

        return np.array(windows)

    # ----------------------------------------------------
    # Predict (Memory Safe)
    # ----------------------------------------------------
    def predict(self, model_name, data_array):

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")

        model = self.models[model_name]

        data_scaled = self.scaler.transform(data_array)

        # ---------------- Sequence Models ----------------
        if model_name in ["lstm", "transformer", "cnn"]:

            windows = self.create_windows(data_scaled)

            if len(windows) == 0:
                raise ValueError(
                    f"Not enough data for window size {self.window_size}."
                )

            batch_size = 512
            total_error = 0.0
            total_count = 0

            for i in range(0, len(windows), batch_size):

                batch = windows[i:i + batch_size]

                input_tensor = torch.tensor(
                    batch, dtype=torch.float32
                ).to(self.device)

                with torch.no_grad():
                    output = model(input_tensor)

                    error = torch.mean(
                        (output - input_tensor) ** 2,
                        dim=(1, 2)
                    )

                total_error += torch.sum(error).item()
                total_count += len(error)

                del input_tensor
                del output
                torch.cuda.empty_cache()

            return total_error / total_count

        # ---------------- GNN ----------------
        elif model_name == "gnn":

            input_tensor = torch.tensor(
                data_scaled, dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                output = model(input_tensor)

            return torch.mean(output).item()