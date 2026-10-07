import random
from datetime import datetime

import numpy as np
import pytest

from masslab.presenters.alloy import AlloyPresenter
from masslab.presenters.app import AppPresenter
from masslab.presenters.demo import DemoPresenter
from masslab.presenters.element import ElementPresenter
from masslab.presenters.formatting import attempts_text, duration, parse_number
from masslab.presenters.login import LoginPresenter
from masslab.presenters.quiz import QuizPresenter
from masslab.presenters.report import ReportPresenter
from masslab.presenters.sandbox import SandboxPresenter
from masslab.presenters.theory import TheoryPresenter
from masslab.presenters.tour import TOUR_STEPS, TourPresenter
from masslab.presenters.workspace import WorkspacePresenter
from tests.fakes import (LAUNCHER_EVENTS, LOGIN_EVENTS, TEST_PASSWORD, FakeClock, FakeMainView, FakeView, task_view,
                         workspace)


def np_rng():
    return np.random.default_rng(0)


# --- форматирование ---------------------------------------------------

def test_parse_number_accepts_comma():
    assert parse_number(" 22,9 ") == 22.9
    assert parse_number("22.9") == 22.9
    assert parse_number("abc") is None
    assert parse_number("-3") is None


def test_duration_and_attempts():
    assert duration(3725) == "01:02:05"
    assert [attempts_text(n) for n in (1, 2, 5, 11, 21)] == [
        "1 попытка", "2 попытки", "5 попыток", "11 попыток", "21 попытка"]


# --- вход -------------------------------------------------------------

@pytest.mark.parametrize("name, group", [("", "ФИЗ-101"), ("Иванов", "  "), ("x" * 61, "1")])
def test_login_rejects_invalid_input(name, group):
    logged = []
    view = FakeView(LOGIN_EVENTS, student_name=name, student_group=group)
    LoginPresenter(view, lambda *a: logged.append(a), lambda: None, lambda *a: None)
    view.start_requested.emit()
    assert logged == []
    assert view.called("show_error")


def test_login_normalizes_spaces():
    logged = []
    view = FakeView(LOGIN_EVENTS, student_name="  Иванов   Иван ", student_group="ФИЗ-101")
    LoginPresenter(view, lambda *a: logged.append(a), lambda: None, lambda *a: None)
    view.start_requested.emit()
    assert logged == [("Иванов Иван", "ФИЗ-101")]


def test_secret_field_opens_teacher_panel_only_with_password():
    opened, demo = [], []
    view = FakeView(LOGIN_EVENTS, student_name="", student_group="")
    LoginPresenter(view, lambda *a: None, lambda: opened.append(True), lambda *a: demo.append(a))
    view.secret_entered.emit("неверно")
    assert not view.called("show_teacher_panel") and view.called("clear_secret")
    view.secret_entered.emit(TEST_PASSWORD)
    assert view.last("show_teacher_panel") == (True,)
    view.sandbox_requested.emit()
    view.teacher_demo_requested.emit()
    assert opened == [True]
    assert demo == [("Преподаватель", "демонстрация")]


# --- тест -------------------------------------------------------------

def _quiz_view():
    return FakeView(["answer_selected", "next_requested", "prev_requested",
                     "finish_requested", "result_action_requested"])


def _answer_quiz(presenter, view, correct):
    quiz = presenter._quiz
    for i, q in enumerate(quiz.questions):
        option = q.correct_index if i < correct else (q.correct_index + 1) % len(q.options)
        view.answer_selected.emit(option)
        view.next_requested.emit()


def test_quiz_finish_enabled_only_when_all_answered():
    view = _quiz_view()
    presenter = QuizPresenter(view, random.Random(0), lambda n, r: None)
    presenter.start()
    assert view.last("set_navigation") == (False, True, False)
    _answer_quiz(presenter, view, 10)
    assert view.last("set_navigation")[2] is True


def test_quiz_failed_then_retry_with_new_questions_then_pass():
    done = []
    view = _quiz_view()
    presenter = QuizPresenter(view, random.Random(1), lambda n, r: done.append((n, r)))
    presenter.start()
    first = {q.text for q in presenter._quiz.questions}

    _answer_quiz(presenter, view, 8)
    view.finish_requested.emit()
    title, _, details, passed, _ = view.last("show_result")
    assert (title, passed) == ("Тест не сдан", False)
    assert len(details) == 1 + 2

    view.result_action_requested.emit()
    assert first.isdisjoint(q.text for q in presenter._quiz.questions)
    assert done == []

    _answer_quiz(presenter, view, 9)
    view.finish_requested.emit()
    assert view.last("show_result")[3] is True
    view.result_action_requested.emit()
    assert done[0][0] == 2
    assert dict(done[0][1].facts)["Результат"].startswith("9 из 10")


def test_quiz_keeps_selected_answer_when_navigating_back():
    view = _quiz_view()
    presenter = QuizPresenter(view, random.Random(2), lambda n, r: None)
    presenter.start()
    view.answer_selected.emit(3)
    view.next_requested.emit()
    view.prev_requested.emit()
    assert view.last("show_question")[4] == 3


# --- панель прибора ---------------------------------------------------

def test_workspace_voltage_change_rerenders():
    from masslab.model.spectrum import Peak
    view = workspace()
    ws = WorkspacePresenter(view, np_rng())
    ws.reset()
    ws.set_peaks([Peak(28.0, 1.0, "Si")])
    x_max_3kv = view.last("show_spectrum")[0].x_lim[1]
    view.voltage_changed.emit(12000)
    assert ws.voltage == 12000
    assert view.last("set_voltage") == (12000,)
    assert view.last("show_spectrum")[0].x_lim[1] == pytest.approx(x_max_3kv / 2)
    assert "12000" in view.last("set_parameters")[0]


def test_workspace_length_change_and_visibility():
    from masslab.model.spectrum import Peak
    view = workspace()
    ws = WorkspacePresenter(view, np_rng(), adjustable_length=True)
    assert view.last("set_length_visible") == (True,)
    ws.set_peaks([Peak(28.0, 1.0, "Si")])
    x_max = view.last("show_spectrum")[0].x_lim[1]
    view.length_changed.emit(2.4)
    assert ws.length == 2.4
    assert view.last("show_spectrum")[0].x_lim[1] == pytest.approx(2 * x_max)
    assert view.last("show_scheme")[0].length == 2.4


def test_workspace_hover_snaps_to_peak():
    from masslab.model.spectrum import Peak
    view = workspace()
    ws = WorkspacePresenter(view, np_rng(), show_mass=True)
    ws.set_peaks([Peak(28.0, 1.0, "Si")])
    t_us = ws.tof.flight_time(28.0) * 1e6
    view.spectrum_hovered.emit(t_us + 0.05)
    x, y, text = view.last("show_cursor")
    assert x == pytest.approx(t_us, abs=0.01)
    assert y is not None
    assert "m/z ≈ 28,0" in text and "R ≈" in text
    view.spectrum_hovered.emit(None)
    assert view.last("hide_cursor") == ()


def test_workspace_log_scale_keeps_values_positive():
    from masslab.model.spectrum import Peak
    view = workspace()
    ws = WorkspacePresenter(view, np_rng())
    ws.set_peaks([Peak(28.0, 1.0, "Si")])
    view.log_scale_toggled.emit(True)
    plot = view.last("show_spectrum")[0]
    assert plot.y_log and min(plot.curves[0].y) > 0


def test_unknown_peak_marked_red_question():
    from masslab.model.spectrum import Peak
    from masslab.presenters.workspace import UNKNOWN_COLOR
    view = workspace()
    ws = WorkspacePresenter(view, np_rng())
    ws.set_peaks([Peak(4.0, 1.0, "He"), Peak(23.0, 1.0, "?")])
    markers = view.last("show_spectrum")[0].markers
    labels = [m.label for m in markers]
    assert "He" in labels
    unknown = [m for m in markers if m.label == "?"]
    assert len(unknown) == 1 and unknown[0].color == UNKNOWN_COLOR
    assert {"H₂O⁺", "N₂⁺", "O₂⁺"} <= set(labels)       # остаточный газ подписан


def test_residual_gas_can_be_disabled():
    from masslab.model.spectrum import Peak
    view = workspace()
    ws = WorkspacePresenter(view, np_rng())
    ws.set_peaks([Peak(4.0, 1.0, "He")])
    ws.set_residual_gas(False)
    assert [m.label for m in view.last("show_spectrum")[0].markers] == ["He"]


# --- задание 1 --------------------------------------------------------

def _demo(selected=("H", "Ar"), double=False):
    done = []
    view = task_view(*LAUNCHER_EVENTS, "answer_selected", "check_requested",
                     selected_elements=lambda: list(selected), double_charge_enabled=double)
    presenter = DemoPresenter(view, random.Random(3), np_rng(), lambda n, r: done.append((n, r)))
    presenter.start()
    return presenter, view, done


def test_demo_launch_fills_flight_table():
    presenter, view, _ = _demo()
    view.launch_requested.emit()
    rows = view.last("show_flight_table")[0]
    assert [r[0] for r in rows] == ["H⁺", "Ar⁺"]
    view.workspace.voltage_changed.emit(12000)
    assert view.last("show_flight_table")[0][0][3] != rows[0][3]


def test_demo_double_charge_adds_ions():
    presenter, view, _ = _demo(selected=("Ar", "Ne"), double=True)
    view.launch_requested.emit()
    rows = view.last("show_flight_table")[0]
    assert [r[0] for r in rows] == ["Ne²⁺", "Ar²⁺", "Ne⁺", "Ar⁺"]
    t_ar2, t_ne = (float(rows[i][3].replace(",", ".")) for i in (1, 2))
    assert t_ar2 == pytest.approx(t_ne, rel=0.01)   # Ar²⁺ и Ne⁺ прилетают почти одновременно


def test_demo_launch_requires_selection():
    presenter, view, _ = _demo(selected=())
    view.launch_requested.emit()
    assert view.last("show_feedback")[1] is False


def test_demo_wrong_answers_replaced_correct_ones_locked():
    presenter, view, done = _demo()
    questions = list(presenter._task.questions)
    for i, q in enumerate(questions):
        right = i % 2 == 0
        view.answer_selected.emit(i, q.correct_index if right else (q.correct_index + 1) % len(q.options))
    view.check_requested.emit()
    assert done == []
    items = view.last("show_questions")[0]
    assert [it.locked for it in items] == [True, False, True, False, True]
    assert items[0].selected is not None and items[1].selected is None
    view.answer_selected.emit(0, (questions[0].correct_index + 1) % len(questions[0].options))
    for i, q in enumerate(presenter._task.questions):
        if i % 2:
            view.answer_selected.emit(i, q.correct_index)
    view.check_requested.emit()
    assert done[0][0] == 2
    report = done[0][1]
    assert report.plot is None or report.plot.curves
    assert len(report.notes) == 10


def test_demo_hint_after_two_failures():
    presenter, view, done = _demo()
    for _ in range(2):
        for i, q in enumerate(presenter._task.questions):
            view.answer_selected.emit(i, (q.correct_index + 1) % len(q.options))
        view.check_requested.emit()
    assert view.last("show_hint")[0].startswith("Подсказка 1")


def test_demo_requires_all_answers():
    presenter, view, done = _demo()
    view.answer_selected.emit(0, 0)
    view.check_requested.emit()
    assert done == []
    assert presenter._attempts == 0


# --- задание 2 --------------------------------------------------------

def _element(time_factor=1.0, mass_factor=1.0, right=True, time_text=None):
    done = []
    state = {}
    view = task_view("check_requested", time_text=lambda: state["time"],
                     mass_text=lambda: state["mass"], selected_option=lambda: state["option"])
    presenter = ElementPresenter(view, random.Random(4), np_rng(),
                                 lambda n, r: done.append((n, r)))
    presenter.start()
    task, tof = presenter._task, presenter._ws.tof
    t_us = tof.flight_time(task.unknown.mass) * 1e6 * time_factor
    state["time"] = time_text if time_text is not None else f"{t_us:.3f}".replace(".", ",")
    state["mass"] = f"{tof.mass_from_time(t_us * 1e-6) * mass_factor:.2f}".replace(".", ",")
    index = task.options.index(task.unknown.symbol)
    state["option"] = index if right else (index + 1) % len(task.options)
    return presenter, view, done


def test_element_correct_answer_completes_with_report():
    presenter, view, done = _element()
    view.check_requested.emit()
    assert done[0][0] == 1
    facts = dict(done[0][1].facts)
    assert "Вычисленная масса" in facts and done[0][1].plot is not None


@pytest.mark.parametrize("kwargs, phrase", [
    ({"time_factor": 1.06}, "время пролёта измерено"),
    ({"mass_factor": 1.3}, "масса вычислена с ошибкой"),
    ({"right": False}, "элемент выбран неверно"),
])
def test_element_diagnosis(kwargs, phrase):
    presenter, view, done = _element(**kwargs)
    old = presenter._task.unknown
    view.check_requested.emit()
    assert done == []
    text, ok = view.last("show_feedback")
    assert ok is False and phrase in text and old.symbol in text
    assert presenter._task.unknown is not old


def test_element_invalid_time_is_not_an_attempt():
    presenter, view, done = _element(time_text="abc")
    view.check_requested.emit()
    assert presenter._attempts == 0


# --- задание 3 --------------------------------------------------------

def test_alloy_flow():
    done = []
    state = {"option": None}
    view = task_view("check_requested", selected_option=lambda: state["option"])
    presenter = AlloyPresenter(view, random.Random(5), np_rng(), lambda n, r: done.append((n, r)))
    presenter.start()
    view.check_requested.emit()
    assert presenter._attempts == 0

    task = presenter._task
    state["option"] = (task.options.index(task.alloy) + 1) % 5
    view.check_requested.emit()
    assert done == [] and presenter._attempts == 1

    task = presenter._task
    state["option"] = task.options.index(task.alloy)
    view.check_requested.emit()
    assert done[0][0] == 2
    report = done[0][1]
    assert dict(report.facts)["Определённый сплав"] == task.alloy.name
    assert report.table_rows


# --- приложение целиком -----------------------------------------------

def test_full_lab_flow_produces_report():
    view = FakeMainView()
    clock = FakeClock()
    written = []
    app = AppPresenter(view, random.Random(6), np_rng(), lambda p, r: written.append((p, r)),
                       clock=clock, now=lambda: datetime(2026, 10, 4, 12, 30))
    app.start()
    assert view.last("show_page") == ("login",)
    assert view.last("set_close_confirmation") == (None,)

    view.login.start_requested.emit()
    assert view.last("show_page") == ("quiz",)
    assert view.last("set_close_confirmation")[0] is not None

    assert not view.called("show_tour_step")             # на тесте тура нет
    clock.advance(300)
    app._stages["quiz"]._on_completed(1, None)
    assert view.last("show_page") == ("task1",)
    assert view.last("show_tour_step")[0] == TOUR_STEPS["task1"][0].target
    view.tour_skip.emit()
    assert view.called("hide_tour")
    clock.advance(600)
    view.tick.emit()
    assert "00:10:00" in view.last("set_timer")[0]
    from masslab.model.session import StageReport
    app._stages["task1"]._on_completed(2, StageReport("Задание 1"))
    clock.advance(400)
    app._stages["task2"]._on_completed(1, StageReport("Задание 2"))
    clock.advance(500)
    app._stages["task3"]._on_completed(3, StageReport("Задание 3"))

    assert view.last("show_page") == ("report",)
    report = view.report.last("show_report")[0]
    assert report.student == "Иванов И. И." and report.group == "ФИЗ-101"
    assert [r.duration for r in report.rows] == ["00:05:00", "00:10:00", "00:06:40", "00:08:20"]
    assert report.rows[3].attempts == "3 попытки"
    assert report.total == "00:30:00"
    assert report.verdict == "ЗАЧТЕНО"
    assert report.finished_at == "04.10.2026 12:30"
    assert view.last("set_stages")[0][-1] == ("Итог", "current")
    assert [st.title for st in report.stages] == ["Задание 1", "Задание 2", "Задание 3"]

    view.report.returns["ask_save_path"] = "/tmp/отчёт"
    view.report.export_requested.emit()
    assert written[0][0] == "/tmp/отчёт.pdf"
    assert view.report.last("show_export_result")[1] is True

    view.report.new_session_requested.emit()
    assert app.session is None
    assert view.last("show_page") == ("login",)


def test_help_and_theory_buttons():
    view = FakeMainView()
    app = AppPresenter(view, random.Random(7), np_rng(), lambda p, r: None)
    app.start()
    view.theory_requested.emit()
    assert view.theory.called("open")
    assert view.theory.last("show_section")[0] == 0
    view.help_requested.emit()                 # на странице входа тура нет
    assert view.called("show_message") and not view.called("show_tour_step")


def test_sandbox_from_login_and_back():
    view = FakeMainView()
    app = AppPresenter(view, random.Random(8), np_rng(), lambda p, r: None)
    app.start()
    view.login.secret_entered.emit(TEST_PASSWORD)
    view.login.sandbox_requested.emit()
    assert view.last("show_page") == ("sandbox",)
    view.sandbox.launch_requested.emit()
    rows = view.sandbox.last("show_flight_table")[0]
    assert [r[0] for r in rows] == ["H²⁺", "H⁺"]
    view.sandbox.gas_toggled.emit(False)
    view.sandbox.exit_requested.emit()
    assert view.last("show_page") == ("login",)


def test_tour_steps_once_per_session_and_on_help():
    view = FakeView(["tour_next", "tour_skip"])
    tour = TourPresenter(view)
    tour.start_if_new("task2")
    n = len(TOUR_STEPS["task2"])
    for _ in range(n):
        view.tour_next.emit()
    assert view.called("hide_tour") and not tour.active
    shown = sum(c == "show_tour_step" for c, _ in view.calls)
    assert shown == n
    tour.start_if_new("task2")
    assert sum(c == "show_tour_step" for c, _ in view.calls) == n
    tour.start("task2")
    assert view.last("show_tour_step")[3:] == (1, n)


def test_report_export_failure_is_reported():
    from masslab.model.session import LabSession
    view = FakeView(["new_session_requested", "exit_requested", "export_requested"],
                    ask_save_path="/нет/доступа.pdf")

    def failing(path, report):
        raise OSError("нет доступа")
    presenter = ReportPresenter(view, lambda: None, lambda: None, failing)
    presenter.start(LabSession("Иванов", "ФИЗ-301"), datetime(2026, 10, 5))
    assert presenter.default_file_name() == "Отчёт_TOF_Иванов_ФИЗ-301.pdf"
    view.export_requested.emit()
    text, ok = view.last("show_export_result")
    assert ok is False and "нет доступа" in text


def test_teacher_demo_reveals_answers_everywhere():
    view = FakeMainView()
    app = AppPresenter(view, random.Random(9), np_rng(), lambda p, r: None)
    app.start()
    view.login.teacher_demo_requested.emit()
    assert app.session.teacher
    quiz = app._stages["quiz"]._quiz
    assert view.quiz.last("show_question")[5] == quiz.questions[0].correct_index
    app._stages["quiz"]._on_completed(1, None)
    view.tour_skip.emit()
    items = view.demo.last("show_questions")[0]
    task = app._stages["task1"]._task
    assert [it.correct for it in items] == [q.correct_index for q in task.questions]
    app._stages["task1"]._on_completed(1, None)
    element = app._stages["task2"]._task
    labels, correct = view.element.last("set_options")
    assert labels[correct].startswith(element.unknown.symbol)
    assert element.unknown.symbol in view.element.last("show_answer")[0]
    view.element.workspace.voltage_changed.emit(6000)
    assert "6000 В" in view.element.last("show_answer")[0]
    app._stages["task2"]._on_completed(1, None)
    alloy = app._stages["task3"]._task
    assert view.alloy.last("set_options")[1] == alloy.options.index(alloy.alloy)
    app._stages["task3"]._on_completed(1, None)
    assert view.report.last("show_report")[0].verdict == "ДЕМОНСТРАЦИЯ"

    app.start()                                   # обычная сессия — ответы скрыты
    view.login.start_requested.emit()
    assert view.quiz.last("show_question")[5] is None


def test_theory_navigation_and_visuals():
    from masslab.model.theory import THEORY_SECTIONS
    view = FakeView(["section_selected", "next_requested", "prev_requested",
                     "resolution_voltage_changed"])
    presenter = TheoryPresenter(view, np_rng())
    assert len(view.last("set_sections")[0]) == len(THEORY_SECTIONS)
    presenter.open()
    assert view.last("set_navigation") == (False, True)
    kinds = []
    for i in range(len(THEORY_SECTIONS)):
        view.section_selected.emit(i)
        index, total, title, points, formula, caption, visual = view.last("show_section")
        assert index == i and title == THEORY_SECTIONS[i].title and "<li>" in points
        kinds.append(visual.kind)
    assert view.last("set_navigation") == (True, False)
    assert kinds == [s.visual for s in THEORY_SECTIONS]
    view.prev_requested.emit()
    assert view.last("show_section")[0] == len(THEORY_SECTIONS) - 2


@pytest.mark.parametrize("voltage, resolved", [(1000, False), (3500, True), (20000, False)])
def test_theory_resolution_demo(voltage, resolved):
    from masslab.model.theory import THEORY_SECTIONS
    view = FakeView(["section_selected", "next_requested", "prev_requested",
                     "resolution_voltage_changed"])
    TheoryPresenter(view, np_rng())
    view.section_selected.emit([s.visual for s in THEORY_SECTIONS].index("resolution"))
    view.resolution_voltage_changed.emit(voltage)
    plot, text = view.last("show_resolution")
    assert f"{voltage} В" in text and ("разделены" in text) is resolved
    assert min(plot.curves[0].x) >= 202 and max(plot.curves[0].x) <= 210
