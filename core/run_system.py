import threading
import time

from core.monitoring_service import MonitoringService
from core.mission_state_manager import MissionStateManager
from core.mission_assistant import MissionAssistant


def start_monitoring(service):
    service.start()


if __name__ == "__main__":

    data_root = "data/ch3_ilsa/ils/data/calibrated"

    # Create shared state
    state_manager = MissionStateManager()

    # Create monitoring service
    monitoring_service = MonitoringService(
        data_root=data_root,
        state_manager=state_manager
    )

    # Create assistant
    assistant = MissionAssistant(state_manager)

    # Start monitoring in background thread
    monitor_thread = threading.Thread(
        target=start_monitoring,
        args=(monitoring_service,),
        daemon=True
    )

    monitor_thread.start()

    print("🚀 Mission AI System Running...")
    print("Monitoring in background.")
    print("Type your question or 'exit' to stop.\n")

    # Interactive user loop
    while True:

        user_input = input("Ask Mission AI: ")

        if user_input.lower() == "exit":
            print("Stopping system...")
            break

        response = assistant.answer(user_input)
        print("AI:", response)