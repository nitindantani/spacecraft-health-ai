import numpy as np
from datetime import datetime


class MissionCore:

    def __init__(self):
        # Sensor → Subsystem mapping
        self.subsystem_map = {
            "imu": "navigation",
            "gyro": "navigation",
            "accelerometer": "navigation",

            "temp": "thermal",
            "heat": "thermal",

            "pressure": "propulsion",
            "flow": "propulsion"
        }

        # Decision thresholds
        self.anomaly_threshold = 0.7
        self.warning_threshold = 0.45
        self.critical_threshold = 0.75

    # ---------------------------------------------------------
    # Subsystem Aggregation
    # ---------------------------------------------------------
    def aggregate_subsystems(self, anomaly_scores):

        subsystem_scores = {}

        for sensor, score in anomaly_scores.items():
            subsystem = self.subsystem_map.get(sensor.lower())
            if subsystem:
                subsystem_scores.setdefault(subsystem, []).append(score)

        subsystem_state = {}
        subsystem_strength = {}

        for subsystem, scores in subsystem_scores.items():

            avg_score = np.mean(scores)
            subsystem_strength[subsystem] = round(float(avg_score), 3)

            if avg_score > self.anomaly_threshold:
                subsystem_state[subsystem] = "UNSTABLE"
            elif avg_score > self.warning_threshold:
                subsystem_state[subsystem] = "WARNING"
            else:
                subsystem_state[subsystem] = "STABLE"

        return subsystem_state, subsystem_strength

    # ---------------------------------------------------------
    # Confidence Engine
    # ---------------------------------------------------------
    def compute_confidence(self, model_scores, final_risk):

        agreement_count = 0
        total_models = len(model_scores)

        for score in model_scores:
            if score > self.anomaly_threshold:
                agreement_count += 1

        agreement_ratio = agreement_count / total_models if total_models > 0 else 0

        # Blend model agreement with risk magnitude
        confidence = (agreement_ratio * 0.6) + (final_risk * 0.4)

        return round(min(confidence, 1.0), 3)

    # ---------------------------------------------------------
    # Risk Fusion
    # ---------------------------------------------------------
    def compute_risk(self, anomaly_scores, trend_score, subsystem_strength):

        anomaly_intensity = np.mean(list(anomaly_scores.values())) if anomaly_scores else 0

        subsystem_intensity = np.mean(list(subsystem_strength.values())) if subsystem_strength else 0

        # Weighted fusion
        risk = (
            (0.4 * anomaly_intensity) +
            (0.3 * trend_score) +
            (0.3 * subsystem_intensity)
        )

        return round(min(risk, 1.0), 3)

    # ---------------------------------------------------------
    # Mission Status Decision
    # ---------------------------------------------------------
    def determine_status(self, risk_score):

        if risk_score >= self.critical_threshold:
            return "CRITICAL"
        elif risk_score >= self.warning_threshold:
            return "WARNING"
        else:
            return "SAFE"

    # ---------------------------------------------------------
    # Root Cause Analysis
    # ---------------------------------------------------------
    def root_cause_analysis(self, anomaly_scores):

        imu = anomaly_scores.get("imu", 0)
        temp = anomaly_scores.get("temp", 0)
        pressure = anomaly_scores.get("pressure", 0)

        if imu > 0.7 and temp < 0.4:
            return "Likely mechanical vibration affecting navigation subsystem"

        if temp > 0.7:
            return "Thermal stress condition detected"

        if pressure > 0.7:
            return "Propulsion system instability detected"

        return "No dominant failure pattern detected"

    # ---------------------------------------------------------
    # Action Recommendation
    # ---------------------------------------------------------
    def recommend_action(self, mission_status):

        if mission_status == "CRITICAL":
            return "Immediate subsystem isolation and switch to backup channels"

        if mission_status == "WARNING":
            return "Increase monitoring frequency and validate sensor integrity"

        return "Normal operation"

    # ---------------------------------------------------------
    # MASTER FUSION METHOD
    # ---------------------------------------------------------
    def generate_mission_report(
        self,
        anomaly_scores,
        model_scores,
        trend_score,
        incidents
    ):

        timestamp = datetime.utcnow().isoformat()

        # Subsystem Analysis
        subsystem_state, subsystem_strength = self.aggregate_subsystems(anomaly_scores)

        # Risk Computation
        risk_score = self.compute_risk(anomaly_scores, trend_score, subsystem_strength)

        # Mission Status
        mission_status = self.determine_status(risk_score)

        # Confidence
        confidence = self.compute_confidence(model_scores, risk_score)

        # Root Cause
        root_cause = self.root_cause_analysis(anomaly_scores)

        # Action
        action = self.recommend_action(mission_status)

        return {
            "timestamp": timestamp,
            "mission_status": mission_status,
            "risk_score": risk_score,
            "confidence": confidence,
            "subsystems": subsystem_state,
            "incidents": incidents,
            "root_cause": root_cause,
            "recommended_action": action
        }