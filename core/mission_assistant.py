class MissionAssistant:

    def __init__(self, state_manager):

        self.state_manager = state_manager

    def answer(self, question):

        state = self.state_manager.get_current_state()

        if "health" in question.lower():

            return (
                f"Current mission status is {state['status']}. "
                f"Latest anomaly score is {state['score']:.4f}. "
                f"Last updated at {state['last_update']}."
            )

        if "risk" in question.lower():

            return f"Current risk level is {state['status']}."

        return "Monitoring system is active."