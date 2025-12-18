import sys
from PySide6 import QtWidgets
from controller.simulation_controller import SimulationController
from view.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    ctrl = SimulationController()
    win = MainWindow(ctrl)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
