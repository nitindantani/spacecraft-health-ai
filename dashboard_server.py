import asyncio
import pandas as pd
from fastapi import FastAPI, WebSocket

app = FastAPI()

csv_file = "data/ch3_ilsa/ils/data/calibrated/20230824/ch3_ils_nop_calib_20230824t101208872_d_accln.csv"

@app.websocket("/telemetry")

async def telemetry_stream(websocket: WebSocket):

    await websocket.accept()

    df = pd.read_csv(csv_file)

    for i in range(len(df)):

        data = {
            "xTemp": float(df["xTemp"].iloc[i]),
            "yTemp": float(df["yTemp"].iloc[i]),
            "zTemp": float(df["zTemp"].iloc[i]),
            "xAcc": float(df["X Fine Acceleration"].iloc[i]),
        }

        await websocket.send_json(data)

        await asyncio.sleep(0.1)