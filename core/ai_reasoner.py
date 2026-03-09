import requests
import numpy as np

def clean_dict(d):
    return {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in d.items()}

def format_sensor_values(sensor_dict):

    formatted = {}

    for k, v in sensor_dict.items():
        formatted[k] = round(float(v), 6)

    return formatted

class AIReasoner:

    def __init__(self, state_manager):

        self.state_manager = state_manager
        self.url = "http://localhost:11434/api/generate"

    def answer(self, question):

        state = self.state_manager.get_current_state()

        prompt = f"""
You are an AI spacecraft monitoring assistant.

Mission status: {state['status']}
Anomaly score: {state['score']}
sensors = clean_dict(state["sensors"])

Sensor errors:
{format_sensor_values(state["sensors"])}

Subsystem status:
{state['subsystems']}

Answer the operator question clearly.

Operator question:
{question}
"""

        response = requests.post(
            self.url,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]