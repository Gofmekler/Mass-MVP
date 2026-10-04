"""MassLab — лабораторная работа «Времяпролётный масс-спектрометр».

Запуск:  python main.py
Проверка сборки (окно закроется само):  python main.py --smoke-test
"""
import random
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets

from masslab.presenters.app import AppPresenter
from masslab.views.qt.main_window import MainWindow


def main(argv):
    app = QtWidgets.QApplication(argv)
    app.setStyle("Fusion")
    window = MainWindow()
    presenter = AppPresenter(window, random.Random(), np.random.default_rng())
    presenter.start()
    if "--smoke-test" in argv:
        window.set_close_confirmation(None)
        QtCore.QTimer.singleShot(2000, app.quit)
        window.show()
    else:
        window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
