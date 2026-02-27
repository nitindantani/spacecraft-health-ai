import numpy as np
import pandas as pd

print("Loading anomalies...")
anoms = np.load("outputs/metrics/anomaly_indices.npy")

print("Loading timestamps...")
df = pd.read_csv("data/normalized/20230824.csv")
time_col = df.columns[0]

times = pd.to_datetime(df[time_col], format="mixed", utc=True)


# convert anomaly windows to timestamps
anomaly_times = [times.iloc[a] for a in anoms if a < len(times)]

# -----------------------------
# GROUP INTO INCIDENTS
# -----------------------------
INCIDENT_GAP = pd.Timedelta(seconds=2)

incidents = []
current = [anomaly_times[0]]

for t in anomaly_times[1:]:
    if t - current[-1] <= INCIDENT_GAP:
        current.append(t)
    else:
        incidents.append(current)
        current = [t]

incidents.append(current)

# -----------------------------
# PRINT INCIDENT SUMMARY
# -----------------------------
print("\nDetected incidents:\n")

for i, inc in enumerate(incidents, 1):
    start = inc[0]
    end = inc[-1]
    duration = (end - start).total_seconds()

    print(f"Incident {i}")
    print(f"Start: {start}")
    print(f"End:   {end}")
    print(f"Duration: {duration:.3f} sec")
    print(f"Windows: {len(inc)}\n")
