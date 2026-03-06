from core.system_engine import MissionSystem


if __name__ == "__main__":

    data_root = "data/ch3_ilsa/ils/data/calibrated"

    system = MissionSystem(data_root)

    system.start_monitoring()

    print("Mission AI running...")
    print("Type questions. Type 'exit' to stop.\n")

    while True:

        q = input("Operator: ")

        if q == "exit":
            break

        answer = system.ask(q)

        print("AI:", answer)