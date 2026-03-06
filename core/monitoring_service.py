import os
import pandas as pd
from collections import deque
from models.inference_manager import InferenceManager
from core.subsystem_analyzer import SubsystemAnalyzer


class MonitoringService:

    def __init__(self, data_root, state_manager):

        self.subsystem_analyzer = SubsystemAnalyzer()

        self.data_root = data_root
        self.state_manager = state_manager
        self.buffer = deque(maxlen=128)

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

        self.counter = 0

    def interpret_score(self, score):

        if score < 1.0:
            return "SAFE"
        elif score < 3.0:
            return "WARNING"
        else:
            return "CRITICAL"

    def start(self):

        folders = sorted(os.listdir(self.data_root))

        for folder in folders:

            path = os.path.join(self.data_root, folder)

            if not os.path.isdir(path):
                continue

            files = sorted([
                f for f in os.listdir(path)
                if f.endswith(".csv")
            ])

            for file in files:

                full_path = os.path.join(path, file)

                for chunk in pd.read_csv(full_path, chunksize=1000):

                    for _, row in chunk.iterrows():

                        sample = row[self.feature_columns].values

                        self.buffer.append(sample)

                        if len(self.buffer) == 128:

                            self.counter += 1

                            if self.counter % 32 == 0:

                                window = pd.DataFrame(self.buffer).values

                                result = self.inference.predict("lstm", window)

                                score = result["overall_score"]

                                sensors = result["sensor_errors"]

                                status = self.interpret_score(score)

                                subsystems = self.subsystem_analyzer.analyze(sensors)

                                self.state_manager.update_state(
                                    score=score,
                                    status=status,
                                    sensors=sensors,
                                    subsystems=subsystems
                                )