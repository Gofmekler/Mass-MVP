import random

import numpy as np

from masslab.model.session import STAGES, LabSession
from masslab.presenters.alloy import AlloyPresenter
from masslab.presenters.demo import DemoPresenter
from masslab.presenters.element import ElementPresenter
from masslab.presenters.report import ReportPresenter
from masslab.report_pdf import write_report_pdf
from tests.fakes import LAUNCHER_EVENTS, FakeView, task_view


def _stage_reports():
    """Отчёты всех заданий, полученные настоящими презентерами."""
    reports = []
    rng, np_rng = random.Random(1), np.random.default_rng(1)

    view = task_view(*LAUNCHER_EVENTS, "answer_selected", "check_requested",
                     selected_elements=lambda: ["H", "N", "Ar"], double_charge_enabled=True)
    demo = DemoPresenter(view, rng, np_rng, lambda n, r: reports.append(r))
    demo.start()
    view.launch_requested.emit()
    for i, q in enumerate(demo._task.questions):
        view.answer_selected.emit(i, q.correct_index)
    view.check_requested.emit()

    state = {}
    view = task_view("check_requested", time_text=lambda: state["t"],
                     mass_text=lambda: state["m"], selected_option=lambda: state["o"])
    element = ElementPresenter(view, rng, np_rng, lambda n, r: reports.append(r))
    element.start()
    task, tof = element._task, element._ws.tof
    t = tof.flight_time(task.unknown.mass)
    state.update(t=str(t * 1e6), m=str(tof.mass_from_time(t)),
                 o=task.options.index(task.unknown.symbol))
    view.check_requested.emit()

    view = task_view("elements_check_requested", "check_requested",
                     selected_option=lambda: state["a"],
                     selected_elements=lambda: list(alloy._task.alloy.major_symbols))
    alloy = AlloyPresenter(view, rng, np_rng, lambda n, r: reports.append(r))
    alloy.start()
    view.elements_check_requested.emit()
    state["a"] = alloy._task.options.index(alloy._task.alloy)
    view.check_requested.emit()
    return reports


def test_pdf_report_contains_all_stages(tmp_path):
    session = LabSession("Иванов Иван", "ФИЗ-301")
    for stage, report in zip(STAGES[1:], _stage_reports()):
        session.start_stage(stage)
        session.finish_stage(stage, 1, report)
    session.start_stage("quiz")
    session.finish_stage("quiz", 2)
    path = tmp_path / "report.pdf"
    from datetime import datetime
    view = FakeView(["new_session_requested", "exit_requested", "export_requested"],
                    ask_save_path=str(path))
    presenter = ReportPresenter(view, lambda: None, lambda: None, write_report_pdf)
    presenter.start(session, datetime(2026, 10, 5, 10, 0))
    view.export_requested.emit()

    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= 4 or len(data) > 20000
    assert view.last("show_export_result")[1] is True
