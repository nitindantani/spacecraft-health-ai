import pandas as pd
import numpy as np

print("Loading incident data...")

df = pd.read_csv("outputs/metrics/incidents_scored.csv")

# ensure sorted by incident id
df = df.sort_values("incident")

print("\nAnalyzing mission risk trend...\n")

# ---------- SEVERITY TREND ----------
sev = df["severity_score"].values

if len(sev) > 1:
    sev_trend = np.polyfit(range(len(sev)), sev, 1)[0]
else:
    sev_trend = 0

# ---------- FREQUENCY TREND ----------
if "start_time" in df.columns:
    df["start_time"] = pd.to_datetime(df["start_time"])
    gaps = df["start_time"].diff().dt.total_seconds().dropna()

    if len(gaps) > 1:
        freq_trend = np.polyfit(range(len(gaps)), gaps, 1)[0]
    else:
        freq_trend = 0
else:
    freq_trend = 0

# ---------- RISK INTERPRETATION ----------
if sev_trend > 0.05:
    severity_msg = "Severity increasing"
elif sev_trend < -0.05:
    severity_msg = "Severity decreasing"
else:
    severity_msg = "Severity stable"

if freq_trend < 0:
    freq_msg = "Incidents occurring closer together"
elif freq_trend > 0:
    freq_msg = "Incidents spacing increasing"
else:
    freq_msg = "Incident frequency stable"

# ---------- FINAL RISK LEVEL ----------
risk_score = sev.mean()

if risk_score > 2.0:
    level = "HIGH RISK"
elif risk_score > 1.0:
    level = "MODERATE RISK"
else:
    level = "LOW RISK"

print("MISSION RISK FORECAST\n")

print("Average severity:", round(risk_score,3))
print("Severity trend:", severity_msg)
print("Frequency trend:", freq_msg)
print("\nPredicted risk level:", level)

# optional save
with open("outputs/metrics/risk_forecast.txt","w") as f:
    f.write(f"Average severity: {risk_score}\n")
    f.write(f"Severity trend: {severity_msg}\n")
    f.write(f"Frequency trend: {freq_msg}\n")
    f.write(f"Predicted risk level: {level}\n")

print("\nForecast saved.")