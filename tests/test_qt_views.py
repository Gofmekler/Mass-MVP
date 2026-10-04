"""Qt-виды реализуют контракты интерфейсов, приложение собирается и проходится целиком."""
import inspect
import os
import random

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
qt = pytest.importorskip("masslab.views.qt.qt")
QtWidgets = qt.QtWidgets

from masslab.model.session import StageReport  # noqa: E402
from masslab.presenters.app import AppPresenter  # noqa: E402
from masslab.presenters.tour import TOUR_STEPS  # noqa: E402
from masslab.views import interfaces  # noqa: E402
from masslab.views.qt.main_window import MainWindow  # noqa: E402
from tests.fakes import TEST_PASSWORD  # noqa: E402


@pytest.fixture(scope="module")
def window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.show_message = lambda *a: None
    win.resize(1024, 768)
    win.show()
    yield win
    win.set_close_confirmation(None)
    win.close()
    app.processEvents()


def _members(protocol):
    names = []
    for cls in protocol.__mro__:
        if cls in (object,) or cls.__name__ in ("Protocol", "Generic"):
            continue
        names += [n for n, v in vars(cls).items()
                  if inspect.isfunction(v) and not n.startswith("_")]
        names += list(getattr(cls, "__annotations__", {}))
    return sorted(set(names))


@pytest.mark.parametrize("attr, protocol", [
    (None, interfaces.IMainView),
    ("login", interfaces.ILoginView),
    ("quiz", interfaces.IQuizView),
    ("demo", interfaces.IDemoView),
    ("element", interfaces.IElementView),
    ("alloy", interfaces.IAlloyView),
    ("report", interfaces.IReportView),
    ("sandbox", interfaces.ISandboxView),
])
def test_views_implement_interfaces(window, attr, protocol):
    view = window if attr is None else getattr(window, attr)
    assert [m for m in _members(protocol) if not hasattr(view, m)] == []


def test_workspace_implements_interface(window):
    ws = window.demo.workspace
    assert [m for m in _members(interfaces.IWorkspaceView) if not hasattr(ws, m)] == []


@pytest.mark.parametrize("page, view_attr", [("task1", "demo"), ("task2", "element"),
                                             ("task3", "alloy")])
def test_every_tour_target_exists(window, page, view_attr):
    targets = getattr(window, view_attr).tour_targets()
    missing = [s.target for s in TOUR_STEPS[page]
               if s.target not in targets and s.target not in ("theory", "help")]
    assert missing == []


def _process(ms=50):
    app = QtWidgets.QApplication.instance()
    timer = qt.QtCore.QElapsedTimer()
    timer.start()
    while timer.elapsed() < ms:
        app.processEvents()


def test_app_runs_through_all_pages(window):
    written = []
    app = AppPresenter(window, random.Random(0), np.random.default_rng(0),
                       lambda path, report: written.append(path))
    app.start()
    window.theory_requested.emit()
    assert window._theory.isVisible()
    window._theory.close()

    window.login._name.setText("Тестов Тест")
    window.login._group.setText("ФИЗ-301")
    window.login.start_requested.emit()
    assert window._stack.currentWidget() is window.quiz

    app._stages["quiz"]._on_completed(1, None)
    assert window._stack.currentWidget() is window.demo
    _process()
    assert window._overlay.isVisible()                 # тур при первом входе в задание
    for _ in TOUR_STEPS["task1"]:
        window.tour_next.emit()
        _process(10)
    assert not window._overlay.isVisible()

    window.demo.set_double_charge(True)
    window.demo.launch_requested.emit()
    window.demo.workspace.voltage_changed.emit(8000)
    window.demo.workspace.length_changed.emit(2.0)
    window.demo.workspace.spectrum_hovered.emit(2.0)
    window.demo.workspace.log_scale_toggled.emit(True)
    window.demo.workspace.bottom_tabs.setCurrentIndex(1)
    _process(100)
    for stage in ("task1", "task2", "task3"):
        window.tour_skip.emit()
        app._stages[stage]._on_completed(1, StageReport(stage))
    assert window._stack.currentWidget() is window.report
    assert window.report._verdict.text() == "ЗАЧТЕНО"
    assert window.report._qr.isVisibleTo(window.report)


def test_sandbox_page(window):
    app = AppPresenter(window, random.Random(1), np.random.default_rng(1), lambda p, r: None)
    app.start()
    window.login.secret_entered.emit(TEST_PASSWORD)
    assert window._stack.currentWidget() is window.sandbox
    window.sandbox.launch_requested.emit()
    assert window.sandbox._table.rowCount() == 3
    window.sandbox.exit_requested.emit()
    assert window._stack.currentWidget() is window.login
