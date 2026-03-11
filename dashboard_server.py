from fastapi import FastAPI, WebSocket
import pandas as pd
import json
import asyncio
import os
from fastapi.responses import HTMLResponse

app = FastAPI()

DATA_ROOT = r"G:\Pro_NML\data\ch3_ilsa\ils\data\calibrated"


def load_all_csv():

    files = []

    for day in sorted(os.listdir(DATA_ROOT)):

        day_path = os.path.join(DATA_ROOT, day)

        if os.path.isdir(day_path):

            for f in sorted(os.listdir(day_path)):

                if f.endswith(".csv"):
                    files.append(os.path.join(day_path, f))

    return files


@app.websocket("/telemetry")
async def telemetry_stream(websocket: WebSocket):

    await websocket.accept()

    files = load_all_csv()

    try:

        for file in files:

            df = pd.read_csv(file)

            for _, row in df.iterrows():

                await websocket.send_text(json.dumps(row.to_dict()))

                await asyncio.sleep(0.02)

    except Exception as e:

        print("WebSocket disconnected:", e)
        
@app.get("/")
def home():

    with open("dashboard/index.html") as f:

        return HTMLResponse(f.read())