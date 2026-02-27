import pandas as pd

print("Loading incident severity data...")

df = pd.read_csv("outputs/metrics/incidents_scored.csv")

print("\nAnalyzing mission condition...")

# ---- compute stats ----
max_sev = df["severity_score"].max()
avg_sev = df["severity_score"].mean()
count = len(df)

print("\nMax Severity:", round(max_sev,3))
print("Average Severity:", round(avg_sev,3))
print("Incident Count:", count)

print("\nMISSION DECISION\n")

# ---- decision logic ----
if max_sev > 0.7:
    status = "CRITICAL"
elif max_sev > 0.4:
    status = "WARNING"
else:
    status = "NORMAL"

print("Status:", status)

# ---- save clean decision ----
decision = pd.DataFrame({
    "status": [status],
    "max_severity": [max_sev],
    "avg_severity": [avg_sev],
    "incident_count": [count]
})

decision.to_csv("outputs/metrics/mission_status.csv", index=False)

print("\nSaved → outputs/metrics/mission_status.csv")