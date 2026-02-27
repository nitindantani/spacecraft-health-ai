import pandas as pd

print("Loading mission + subsystem status...")

mission = pd.read_csv("outputs/metrics/mission_status.csv")
subsys  = pd.read_csv("outputs/metrics/subsystem_status.csv")

mission_status = mission["status"][0]
subsystem = subsys["subsystem"][0]
sub_status = subsys["status"][0]

print("\nEvaluating mission response...\n")

# --- Decision Logic ---
if mission_status == "SAFE":
    action = "Nominal operation"

elif mission_status == "WARNING":
    if sub_status == "WARNING":
        action = f"Increase monitoring of {subsystem}"
    else:
        action = "Monitor system"

elif mission_status == "CRITICAL":
    if "STRUCTURAL" in subsystem:
        action = "Switch to SAFE MODE"
    elif "POWER" in subsystem:
        action = "Limit power usage"
    elif "THERMAL" in subsystem:
        action = "Activate thermal regulation"
    else:
        action = "Immediate investigation required"

else:
    action = "Unknown state"

print("MISSION ACTION RECOMMENDATION\n")
print("Action:", action)

pd.DataFrame([{
    "mission_status": mission_status,
    "subsystem": subsystem,
    "subsystem_status": sub_status,
    "recommended_action": action
}]).to_csv("outputs/metrics/mission_action.csv", index=False)

print("\nSaved → outputs/metrics/mission_action.csv")