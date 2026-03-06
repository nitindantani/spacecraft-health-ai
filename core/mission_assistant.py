class MissionAssistant:

    def __init__(self, state_manager):

        self.state_manager = state_manager

    def answer(self, question):

        question = question.lower()

        state = self.state_manager.get_current_state()

        if state["score"] is None:
            return "Monitoring system is initializing."

        if "health" in question:

            return (
                f"Spacecraft health is {state['status']}. "
                f"Latest anomaly score is {state['score']:.4f}."
            )

        if "risk" in question:

            return f"Current mission risk level is {state['status']}."

        if "sensor" in question:

            sensors = self.state_manager.get_sensor_anomalies()

            if not sensors:
                return "No sensors are showing abnormal behaviour."

            return f"Anomalies detected in sensors: {', '.join(sensors)}."

        if "anomaly" in question:

            return f"Latest anomaly score is {state['score']:.4f}."

        return "Monitoring system is active."
    
        if "subsystem" in question:

            unstable = self.state_manager.get_unstable_subsystems()

            if not unstable:
                return "All spacecraft subsystems are stable."

            return f"Unstable subsystems detected: {', '.join(unstable)}."
        if "history" in question or "previous" in question:

            history = self.state_manager.get_recent_history()

            if not history:
                return "No historical telemetry events recorded yet."

            response = "Recent mission states:\n"

            for entry in history[-5:]:

                response += (
                    f"{entry['time']} → "
                    f"{entry['status']} "
                    f"(score {entry['score']:.2f})\n"
                )

            return response