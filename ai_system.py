import os

print("\n==============================")
print("SPACECRAFT AI HEALTH SYSTEM")
print("==============================\n")


# -------------------------
# 1. FINAL ANOMALY DETECTOR
# -------------------------
print("Step 1: Running anomaly fusion...")
os.system("python -m pipeline.final_detector")

# -------------------------
# 2. BUILD INCIDENTS
# -------------------------
print("\nStep 2: Building incidents...")
os.system("python -m pipeline.build_incidents")

# -------------------------
# 3. SCORE INCIDENTS
# -------------------------
print("\nStep 3: Scoring incidents...")
os.system("python -m pipeline.score_incidents")

# -------------------------
# 4. CLASSIFY SUBSYSTEM
# -------------------------
print("\nStep 4: Classifying subsystem...")
os.system("python -m pipeline.classify_subsystem")

# -------------------------
# 5. PREDICT RISK
# -------------------------
print("\nStep 5: Predicting mission risk...")
os.system("python -m pipeline.predict_risk")

# -------------------------
# 6. FINAL REPORT
# -------------------------
print("\nStep 6: Generating mission report...\n")
os.system("python -m reports.mission_report --health")

print("\n==============================")
print("AI SYSTEM RUN COMPLETE")
print("==============================\n")