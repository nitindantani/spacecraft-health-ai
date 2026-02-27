import pandas as pd

print("Loading incident + subsystem data...")

df = pd.read_csv("outputs/metrics/incidents_scored.csv")

print("\nAnalyzing subsystem health...\n")

# group by subsystem
subsystem_health = {}

for _, row in df.iterrows():
    subsystem = row.get("subsystem", "UNKNOWN")
    severity = row["severity_score"]

    if subsystem not in subsystem_health:
        subsystem_health[subsystem] = []

    subsystem_health[subsystem].append(severity)


print("SUBSYSTEM STATUS\n")

results = []

for subsystem, values in subsystem_health.items():

    avg = sum(values) / len(values)
    max_val = max(values)

    if max_val < 0.25:
        status = "NOMINAL"
    elif max_val < 0.45:
        status = "WATCH"
    elif max_val < 0.70:
        status = "WARNING"
    else:
        status = "CRITICAL"

    print(f"{subsystem} → {status}")

    results.append({
        "subsystem": subsystem,
        "avg_severity": avg,
        "max_severity": max_val,
        "status": status
    })

pd.DataFrame(results).to_csv(
    "outputs/metrics/subsystem_status.csv",
    index=False
)

print("\nSaved → outputs/metrics/subsystem_status.csv")