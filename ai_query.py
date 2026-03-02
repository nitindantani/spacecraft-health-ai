import pandas as pd
import sys

print("\n==============================")
print("SPACECRAFT AI QUERY SYSTEM")
print("==============================\n")

if len(sys.argv) < 2:
    print("Ask something like:")
    print('python ai_query.py "mission health"')
    print('python ai_query.py "weak subsystem"')
    print('python ai_query.py "required action"')
    exit()

query = sys.argv[1].lower()

# ---------- LOAD DATA ----------
mission = pd.read_csv("outputs/metrics/mission_status.csv")
subsys  = pd.read_csv("outputs/metrics/subsystem_status.csv")
action  = pd.read_csv("outputs/metrics/mission_action.csv")

mission_status = mission["mission_status"][0]
subsystem = subsys["subsystem"][0]
subsystem_state = subsys["status"][0]
recommended_action = action["action"][0]

# ---------- QUERY ENGINE ----------

if "mission" in query or "health" in query:

    print("MISSION STATUS:", mission_status)

elif "subsystem" in query:

    print("WEAK SUBSYSTEM:", subsystem)
    print("STATUS:", subsystem_state)

elif "action" in query or "what should" in query:

    print("RECOMMENDED ACTION:")
    print(recommended_action)

elif "summary" in query:

    print("\nMISSION SUMMARY\n")
    print("Mission:", mission_status)
    print("Subsystem:", subsystem, "->", subsystem_state)
    print("Action:", recommended_action)

else:
    print("I don't understand the query.")
    print("Try: mission health / weak subsystem / action / summary")

print("\n==============================")