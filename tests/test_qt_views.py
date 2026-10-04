"""Qt-виды реализуют контракты интерфейсов, приложение собирается и рисует графики."""
import inspect
import os
import random

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from masslab.presenters.app import AppPresenter  # noqa: E402
from masslab.views import interfaces  # noqa: E402
from masslab.views.qt.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    yield win
    win.set_close_confirmation(None)
    win.close()
    app.processEvents()


def _members(protocol):
    methods = [n for n, v in vars(protocol).items()
               if inspect.isfunction(v) and not n.startswith("_")]
    attributes = [n for n in getattr(protocol, "__annotations__", {})]
    return methods + attributes


@pytest.mark.parametrize("attr, protocol", [
    (None, interfaces.IMainView),
    ("login", interfaces.ILoginView),
    ("quiz", interfaces.IQuizView),
    ("demo", interfaces.IDemoView),
    ("element", interfaces.IElementView),
    ("alloy", interfaces.IAlloyView),
    ("report", interfaces.IReportView),
])
def test_views_implement_interfaces(window, attr, protocol):
    view = window if attr is None else getattr(window, attr)
    missing = [m for m in _members(protocol) if not hasattr(view, m)]
    assert missing == []


def test_workspace_implements_interface(window):
    missing = [m for m in _members(interfaces.IWorkspaceView)
               if not hasattr(window.demo.workspace, m)]
    assert missing == []


def test_app_runs_through_all_pages(window):
    app = AppPresenter(window, random.Random(0), np.random.default_rng(0))
    window.show_message = lambda *a: None
    app.start()
    window.login._name.setText("Тестов Тест")
    window.login._group.setText("ФИЗ-101")
    window.login.start_requested.emit()
    assert window._stack.currentWidget() is window.quiz

    for stage in ("quiz", "task1", "task2", "task3"):
        presenter = app._stages[stage]
        if stage == "task1":
            window.demo.launch_requested.emit()
            window.demo.workspace.voltage_changed.emit(8000)
            window.demo.workspace.spectrum_hovered.emit(2.0)
            window.demo.workspace.log_scale_toggled.emit(True)
        presenter._on_completed(1)
    assert window._stack.currentWidget() is window.report
    assert window.report._verdict.text() == "ЗАЧТЕНО"
