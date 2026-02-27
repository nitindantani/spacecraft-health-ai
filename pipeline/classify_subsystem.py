import numpy as np
import pandas as pd

print("Loading incidents from CSV...")
df = pd.read_csv("outputs/metrics/incidents_scored.csv")

# each row already represents an incident
incidents = df.index.values

print("Loading sensor explanations...")

# -------- SENSOR GROUPS --------
THERMAL = ["Temp X", "Temp Y", "Temp Z"]
STRUCTURAL = [
    "Accel coarse X", "Accel coarse Y", "Accel coarse Z",
    "Accel fine X", "Accel fine Y", "Accel fine Z"
]

# adjust if your order differs
SENSOR_NAMES = [
    "Temp X","Temp Y","Temp Z",
    "Accel coarse X","Accel coarse Y","Accel coarse Z",
    "Accel fine X","Accel fine Y","Accel fine Z"
]

print("\nClassifying incidents...\n")

results = []

for i in incidents:

    # fake sensor vector from anomaly score
    # (until we wire full per-sensor errors)
    base = df.iloc[i, 1]

    sensor_strength = np.linspace(base*0.7, base*1.3, 9)

    top_ids = np.argsort(sensor_strength)[-3:][::-1]
    top_sensors = [SENSOR_NAMES[j] for j in top_ids]

    thermal_count = sum(s in THERMAL for s in top_sensors)
    structural_count = sum(s in STRUCTURAL for s in top_sensors)

    if structural_count > thermal_count:
        subsystem = "STRUCTURAL / IMU"
    elif thermal_count > structural_count:
        subsystem = "THERMAL"
    else:
        subsystem = "MIXED"

    print(f"Incident {i+1}")
    print("Subsystem:", subsystem)
    print("Top sensors:", ", ".join(top_sensors))
    print()

    results.append((i+1, subsystem, top_sensors))

    # ----------------------------
# MERGE SUBSYSTEM INTO CSV
# ----------------------------
import pandas as pd

print("\nMerging subsystem labels into incidents CSV...")

df = pd.read_csv("outputs/metrics/incidents_scored.csv")

# results = [(incident_id, subsystem, sensors)]
subsystems = [r[1] for r in results]

# attach column
df["subsystem"] = subsystems

# save back
df.to_csv("outputs/metrics/incidents_scored.csv", index=False)

print("✔ Subsystem column saved to CSV")