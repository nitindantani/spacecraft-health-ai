import threading

from core.monitoring_service import MonitoringService
from core.mission_state_manager import MissionStateManager
from core.mission_assistant import MissionAssistant
from core.ai_reasoner import AIReasoner

class MissionSystem:

    def __init__(self, data_root):

        self.reasoner = AIReasoner(self.state_manager)

        # Central state
        self.state_manager = MissionStateManager()

        # Monitoring engine
        self.monitoring = MonitoringService(
            data_root=data_root,
            state_manager=self.state_manager
        )

        # AI assistant
        self.assistant = MissionAssistant(self.state_manager)

    def start_monitoring(self):

        thread = threading.Thread(
            target=self.monitoring.start,
            daemon=True
        )

        thread.start()

    def ask(self, question):

        return self.assistant.answer(question)