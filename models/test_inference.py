import numpy as np
from models.inference_manager import InferenceManager

manager = InferenceManager()

print("Scaler expects:", manager.scaler.n_features_in_, "features")

# Must be > 128 rows
dummy_data = np.random.randn(
    500,
    manager.scaler.n_features_in_
)

score = manager.predict("lstm", dummy_data)

print("Anomaly Score:", score)