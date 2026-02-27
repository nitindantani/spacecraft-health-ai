import pandas as pd

print("Loading scored incidents...")

df = pd.read_csv("outputs/metrics/incidents_scored.csv")

print("\nMISSION INCIDENT INTERPRETATION\n")

for i, row in df.iterrows():

    severity = row["severity_score"]
    subsystem = row.get("subsystem", "Unknown")
    duration = row["duration_windows"]

    print(f"Incident {row['incident']}")

    # --- Interpretation logic ---
    if severity < 0.2:
        meaning = "Minor fluctuation"
        risk = "Safe"

    elif severity < 0.5:
        meaning = "Localized disturbance"
        risk = "Monitor"

    elif severity < 1.5:
        meaning = "Subsystem stress detected"
        risk = "Warning"

    else:
        meaning = "Potential subsystem instability"
        risk = "Critical"

    # duration interpretation
    if duration <= 2:
        pattern = "Transient event"
    elif duration <= 10:
        pattern = "Short disturbance"
    else:
        pattern = "Persistent anomaly"

    print(f"Subsystem: {subsystem}")
    print(f"Type: {meaning}")
    print(f"Pattern: {pattern}")
    print(f"Risk Level: {risk}")
    print("-"*40)