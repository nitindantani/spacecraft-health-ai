from core.monitoring_service import MonitoringService
from core.mission_state_manager import MissionStateManager
from core.mission_assistant import MissionAssistant


if __name__ == "__main__":

    data_root = "data/ch3_ilsa/ils/data/calibrated"

    state_manager = MissionStateManager()

    monitoring_service = MonitoringService(
        data_root=data_root,
        state_manager=state_manager
    )

    assistant = MissionAssistant(state_manager)

    # Start monitoring in background
    print("Starting monitoring service...")
    monitoring_service.start()

    # Example query
    print(assistant.answer("What is spacecraft health?"))