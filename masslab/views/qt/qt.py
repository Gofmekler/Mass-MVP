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


def exec_app(obj):
    """exec() в PySide6, exec_() в PySide2 — для приложения и диалогов."""
    run = getattr(obj, "exec", None) or obj.exec_
    return run()


def install_russian(app):
    """Русские надписи стандартных кнопок, диалогов и контекстных меню Qt."""
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.Russian, QtCore.QLocale.Russia))
    info = QtCore.QLibraryInfo
    if hasattr(info, "path"):
        folder = info.path(info.TranslationsPath)
    else:
        folder = info.location(info.TranslationsPath)
    for name in ("qtbase", "qt"):
        translator = QtCore.QTranslator(app)
        if translator.load(f"{name}_ru", folder):
            app.installTranslator(translator)
            break
