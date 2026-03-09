import threading

from core.monitoring_service import MonitoringService
from core.mission_state_manager import MissionStateManager
from core.ai_reasoner import AIReasoner


class MissionSystem:

    def __init__(self, data_root):

        # 1️⃣ Create central mission state
        self.state_manager = MissionStateManager()

        # 2️⃣ Monitoring engine
        self.monitoring = MonitoringService(
            data_root=data_root,
            state_manager=self.state_manager
        )

        # 3️⃣ AI reasoning layer
        self.reasoner = AIReasoner(self.state_manager)

    def start_monitoring(self):

        thread = threading.Thread(
            target=self.monitoring.start,
            daemon=True
        )

        thread.start()

    def ask(self, question):

        # Send question to AI reasoning system
        return self.reasoner.answer(question)