from datetime import datetime


class MissionStateManager:

    def __init__(self):

        self.current_score = None
        self.current_status = "UNKNOWN"

        self.sensor_errors = {}
        self.subsystem_status = {}

        self.last_update = None

        # NEW: history storage
        self.history = []

    def update_state(self, score, status, sensors, subsystems):

        timestamp = datetime.utcnow()

        self.current_score = score
        self.current_status = status
        self.sensor_errors = sensors
        self.subsystem_status = subsystems
        self.last_update = timestamp

        # store historical snapshot
        self.history.append({
            "time": timestamp,
            "score": score,
            "status": status,
            "sensors": sensors,
            "subsystems": subsystems
        })

        # keep memory manageable
        if len(self.history) > 10000:
            self.history.pop(0)

    def get_current_state(self):

        return {
            "status": self.current_status,
            "score": self.current_score,
            "sensors": self.sensor_errors,
            "subsystems": self.subsystem_status,
            "last_update": self.last_update
        }

    def get_recent_history(self, n=10):

        return self.history[-n:]
    def get_subsystem_history(self, subsystem):

        events = []

        for entry in self.history:

            if subsystem in entry["subsystems"]:

                status = entry["subsystems"][subsystem]["status"]

                if status != "STABLE":

                    events.append({
                        "time": entry["time"],
                        "status": status
                    })

        return events
    
    def get_sensor_anomalies(self, threshold=1.0):

        anomalies = []

        for sensor, value in self.sensor_errors.items():

            if value > threshold:
                anomalies.append(sensor)

        return anomalies