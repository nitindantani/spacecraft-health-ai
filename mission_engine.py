import asyncio
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, WebSocket
import uvicorn


# ===============================
# LOAD TELEMETRY DATA
# ===============================

import os
import pandas as pd

DATA_FOLDER = r"G:\Pro_NML\data\ch3_ilsa\ils\data\calibrated"

csv_files = []

# find all CSV files in all day folders
for root, dirs, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(root, file))

print("Total CSV files found:", len(csv_files))

file_index = 0
data_index = 0
current_df = None


def load_next_file():
    global file_index, current_df, data_index

    if file_index >= len(csv_files):
        file_index = 0

    path = csv_files[file_index]

    print("Loading file:", path)

    current_df = pd.read_csv(path)

    data_index = 0

    file_index += 1


# load first file
load_next_file()


def next_packet():

    global data_index, current_df

    if data_index >= len(current_df):
        load_next_file()

    row = current_df.iloc[data_index]

    data_index += 1

    return row.to_dict()

# ===============================
# MODEL DEFINITIONS
# ===============================

class LSTMModel:

    def run(self, data):

        score = abs(data.get("xTemp", 0)) * 0.001

        if score > 0.05:
            status = "ANOMALY"
        else:
            status = "NORMAL"

        return {"model": "LSTM", "score": score, "status": status}


class CNNModel:

    def run(self, data):

        vib = abs(data.get("Z Fine Acceleration", 0))

        if vib > 0.1:
            pattern = "VIBRATION_PATTERN"
        else:
            pattern = "NORMAL"

        return {"model": "CNN", "pattern": pattern}


class TransformerModel:

    def run(self, data):

        pred_temp = data.get("xTemp", 0) + 0.5

        return {"model": "Transformer", "pred_temp": pred_temp}


class GNNModel:

    def run(self, data):

        return {"model": "GNN", "subsystem": "structure"}


# ===============================
# LOAD MODELS
# ===============================

lstm = LSTMModel()
cnn = CNNModel()
transformer = TransformerModel()
gnn = GNNModel()


# ===============================
# MODEL ROUTER
# ===============================

def select_model(question):

    q = question.lower()

    if "anomaly" in q:
        return lstm

    if "pattern" in q:
        return cnn

    if "predict" in q:
        return transformer

    if "cause" in q:
        return gnn

    return lstm


# ===============================
# LLM REASONING (OLLAMA)
# ===============================

def explain(result):

    prompt = f"""
You are spacecraft mission AI.

Model output:
{result}

Explain what happened and what operator should do.
"""

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return r.json()["response"]


# ===============================
# MISSION STATE
# ===============================

current_state = {
    "sensors": {},
    "anomaly_score": 0
}


# ===============================
# REAL TIME ENGINE
# ===============================

async def telemetry_loop():

    global current_state

    while True:

        packet = next_packet()

        current_state["sensors"] = packet

        anomaly = lstm.run(packet)

        current_state["anomaly_score"] = anomaly["score"]

        await asyncio.sleep(1)


# ===============================
# FASTAPI SERVER
# ===============================

app = FastAPI()
from fastapi.responses import FileResponse

@app.get("/")
def dashboard():
    return FileResponse("dashboard/index.html")

@app.get("/ask")
def ask(question: str):

    model = select_model(question)

    result = model.run(current_state["sensors"])

    answer = explain(result)

    return {"response": answer}

@app.websocket("/telemetry")
async def telemetry_stream(websocket: WebSocket):

    await websocket.accept()

    try:
        while True:
            await websocket.send_json(current_state)
            await asyncio.sleep(1)

    except:
        print("WebSocket disconnected")

# ===============================
# START SYSTEM
# ===============================

async def main():

    asyncio.create_task(telemetry_loop())

    config = uvicorn.Config(app, host="0.0.0.0", port=9000)
    server = uvicorn.Server(config)

    await server.serve()


if __name__ == "__main__":

    asyncio.run(main())
