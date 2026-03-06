class SubsystemAnalyzer:

    def __init__(self):

        self.subsystem_map = {
            "thermal": [
                "xTemp",
                "yTemp",
                "zTemp"
            ],

            "structure": [
                "X Coarse Acceleration",
                "Y Coarse Acceleration",
                "Z Coarse Acceleration"
            ],

            "navigation": [
                "X Fine Acceleration",
                "Y Fine Acceleration",
                "Z Fine Acceleration"
            ]
        }

    def analyze(self, sensor_errors):

        subsystem_status = {}

        for subsystem, sensors in self.subsystem_map.items():

            values = [sensor_errors[s] for s in sensors]

            score = sum(values) / len(values)

            if score < 1:
                status = "STABLE"
            elif score < 3:
                status = "WARNING"
            else:
                status = "UNSTABLE"

            subsystem_status[subsystem] = {
                "score": score,
                "status": status
            }

        return subsystem_status