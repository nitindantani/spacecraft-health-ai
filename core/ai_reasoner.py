from openai import OpenAI

class AIReasoner:

    def __init__(self, state_manager):

        self.state_manager = state_manager
        self.client = OpenAI()

    def answer(self, question):

        state = self.state_manager.get_current_state()

        prompt = f"""
You are an AI spacecraft monitoring assistant.

Mission state:
Status: {state['status']}
Anomaly score: {state['score']}
Sensor errors: {state['sensors']}
Subsystem status: {state['subsystems']}

User question:
{question}

Explain the spacecraft condition clearly.
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content