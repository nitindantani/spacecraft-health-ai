from core.mission_core import MissionCore

# Create core object
core = MissionCore()

# Dummy test data (simulate pipeline output)
anomaly_scores = {
    "imu": 0.82,
    "gyro": 0.76,
    "temp": 0.21,
    "pressure": 0.33
}

model_scores = [0.78, 0.81]  # LSTM, CNN

trend_score = 0.60

incident_list = [
    {"time": "12:03:21", "event": "IMU spike detected"}
]

# Generate mission report
report = core.generate_mission_report(
    anomaly_scores=anomaly_scores,
    model_scores=model_scores,
    trend_score=trend_score,
    incidents=incident_list
)

print("\nMISSION REPORT\n")
for key, value in report.items():
    print(f"{key}: {value}")