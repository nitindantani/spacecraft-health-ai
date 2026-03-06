from datetime import datetime


class MissionStateManager:

    def __init__(self):

        self.current_score = None
        self.current_status = "UNKNOWN"
        self.last_update = None

        self.history = []

    def update_state(self, score, status):

        self.current_score = score
        self.current_status = status
        self.last_update = datetime.utcnow()

        self.history.append({
            "time": self.last_update,
            "score": score,
            "status": status
        })

    def get_current_state(self):

        return {
            "status": self.current_status,
            "score": self.current_score,
            "last_update": self.last_update
        }

    def get_history(self):

        return self.history[-50:]  # recent 50 entries