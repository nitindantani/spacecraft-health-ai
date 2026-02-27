import argparse
import pandas as pd
from datetime import timedelta

print("Loading mission data...")

# load incidents table
df = pd.read_csv("outputs/metrics/incidents_scored.csv")

# if timestamp column exists, convert it
if "start_time" in df.columns:
    df["start_time"] = pd.to_datetime(df["start_time"])

parser = argparse.ArgumentParser()

parser.add_argument("--health", action="store_true")
parser.add_argument("--last_hours", type=int)
parser.add_argument("--subsystem", type=str)

args = parser.parse_args()

print()

# -------------------------
# HEALTH SUMMARY
# -------------------------
if args.health:
    print("MISSION HEALTH SUMMARY\n")

    total = len(df)
    critical = (df["label"] == "CRITICAL").sum()

    print("Total incidents:", total)
    print("Critical incidents:", critical)

    if critical == 0:
        print("Status: HEALTHY")
    elif critical <= 2:
        print("Status: STABLE WITH MINOR ISSUES")
    else:
        print("Status: WARNING / INVESTIGATION NEEDED")

# -------------------------
# LAST HOURS REPORT
# -------------------------
elif args.last_hours:

    if "start_time" not in df.columns:
        print("No timestamps available in CSV.")
        exit()

    latest = df["start_time"].max()
    cutoff = latest - timedelta(hours=args.last_hours)

    recent = df[df["start_time"] >= cutoff]

    print(f"INCIDENTS IN LAST {args.last_hours} HOURS\n")

    if len(recent) == 0:
        print("No incidents detected.")
    else:
        print(recent[[
            "incident",
            "duration_windows",
            "anomaly_strength",
            "graph_risk",
            "severity_score",
            "label"
        ]])

# -------------------------
# SUBSYSTEM FILTER
# -------------------------
elif args.subsystem:

    if "subsystem" not in df.columns:
        print("Subsystem info not yet merged into CSV.")
        exit()

    sub = df[df["subsystem"].str.contains(args.subsystem, case=False)]

    print(f"{args.subsystem.upper()} INCIDENT REPORT\n")

    if len(sub) == 0:
        print("No incidents for this subsystem.")
    else:
        print(sub)

else:
    print("No option selected.")
    print("Try:")
    print("  --health")
    print("  --last_hours 7")
    print("  --subsystem IMU")