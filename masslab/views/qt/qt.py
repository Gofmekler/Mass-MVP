"""Совместимость Qt-привязок.

PySide6 (Qt 6) работает на Windows 10/11, PySide2 (Qt 5.15) — ещё и на
Windows 7 с Python 3.8. Код видов импортирует Qt только отсюда.
"""
import os

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    QT_API = "pyside6"
except ImportError:  # Windows 7: Python 3.8 + PySide2
    from PySide2 import QtCore, QtGui, QtWidgets
    QT_API = "pyside2"

# matplotlib выбирает Qt-привязку по этой переменной
os.environ.setdefault("QT_API", QT_API)

Signal = QtCore.Signal


def exec_app(app):
    run = getattr(app, "exec", None) or app.exec_
    return run()
