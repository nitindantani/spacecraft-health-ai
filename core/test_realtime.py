from core.realtime_controller import RealtimeController

controller = RealtimeController()

result = controller.detect_anomaly(hours=7)

print(result)