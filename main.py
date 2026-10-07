"""MassLab — лабораторная работа «Времяпролётный масс-спектрометр».

Запуск:  python main.py
Проверка (окно закроется само):  python main.py --smoke-test
"""
import random
import sys

import numpy as np

from masslab.views.qt.qt import QtCore, QtWidgets, exec_app, install_russian  # первым
from masslab.presenters.app import AppPresenter
from masslab.report_pdf import write_report_pdf
from masslab.views.qt.main_window import MainWindow


def main(argv):
    app = QtWidgets.QApplication(argv)
    app.setStyle("Fusion")
    install_russian(app)
    window = MainWindow()
    presenter = AppPresenter(window, random.Random(), np.random.default_rng(), write_report_pdf)
    presenter.start()
    if "--smoke-test" in argv:
        QtCore.QTimer.singleShot(2000, app.quit)
        window.show()
    else:
        window.showMaximized()
    return exec_app(app)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
