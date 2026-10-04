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
from masslab.presenters.workspace import WorkspacePresenter
from tests.fakes import FakeClock, FakeMainView, FakeView, task_view, workspace


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
    view = FakeView(["start_requested"], student_name=name, student_group=group)
    LoginPresenter(view, lambda *a: logged.append(a))
    view.start_requested.emit()
    assert logged == []
    assert view.called("show_error")


def test_login_normalizes_spaces():
    logged = []
    view = FakeView(["start_requested"], student_name="  Иванов   Иван ", student_group="ФИЗ-101")
    LoginPresenter(view, lambda *a: logged.append(a))
    view.start_requested.emit()
    assert logged == [("Иванов Иван", "ФИЗ-101")]


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
    presenter = QuizPresenter(view, random.Random(0), lambda n: None)
    presenter.start()
    assert view.last("set_navigation") == (False, True, False)
    _answer_quiz(presenter, view, 10)
    assert view.last("set_navigation")[2] is True


def test_quiz_failed_then_retry_with_new_questions_then_pass():
    done = []
    view = _quiz_view()
    presenter = QuizPresenter(view, random.Random(1), done.append)
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
    assert done == [2]


def test_quiz_keeps_selected_answer_when_navigating_back():
    view = _quiz_view()
    presenter = QuizPresenter(view, random.Random(2), lambda n: None)
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
    assert "m/z ≈ 28,0" in text
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


def test_unknown_peak_has_no_marker():
    from masslab.model.spectrum import Peak
    view = workspace()
    ws = WorkspacePresenter(view, np_rng())
    ws.set_peaks([Peak(4.0, 1.0, "He"), Peak(23.0, 1.0, "?")])
    assert [m.label for m in view.last("show_spectrum")[0].markers] == ["He"]


# --- задание 1 --------------------------------------------------------

def _demo(selected=("H", "Ar")):
    done = []
    view = task_view("launch_requested", "answer_selected", "check_requested",
                     selected_elements=lambda: list(selected))
    presenter = DemoPresenter(view, random.Random(3), np_rng(), done.append)
    presenter.start()
    return presenter, view, done


def test_demo_launch_fills_flight_table():
    presenter, view, _ = _demo()
    view.launch_requested.emit()
    rows = view.last("show_flight_table")[0]
    assert [r[0] for r in rows] == ["H", "Ar"]
    view.workspace.voltage_changed.emit(12000)
    assert view.last("show_flight_table")[0][0][3] != rows[0][3]


def test_demo_launch_requires_selection():
    presenter, view, _ = _demo(selected=())
    view.launch_requested.emit()
    assert view.last("show_feedback")[1] is False


def test_demo_wrong_answers_replace_questions():
    presenter, view, done = _demo()
    questions = presenter._task.questions
    for i, q in enumerate(questions):
        view.answer_selected.emit(i, (q.correct_index + 1) % len(q.options))
    view.check_requested.emit()
    assert done == []
    assert sum(call == "show_questions" for call, _ in view.calls) == 2
    for i, q in enumerate(presenter._task.questions):
        view.answer_selected.emit(i, q.correct_index)
    view.check_requested.emit()
    assert done == [2]


def test_demo_requires_all_answers():
    presenter, view, done = _demo()
    view.answer_selected.emit(0, 0)
    view.check_requested.emit()
    assert done == []
    assert presenter._attempts == 0


# --- задание 2 --------------------------------------------------------

def _element(mass_text, option):
    done = []
    state = {}
    view = task_view("check_requested", mass_text=lambda: state["mass"],
                     selected_option=lambda: state["option"])
    presenter = ElementPresenter(view, random.Random(4), np_rng(), done.append)
    presenter.start()
    task = presenter._task
    state["mass"] = mass_text(task)
    state["option"] = option(task)
    return presenter, view, done, state


def test_element_correct_answer_completes():
    presenter, view, done, _ = _element(
        lambda t: f"{t.unknown.mass:.1f}".replace(".", ","),
        lambda t: t.options.index(t.unknown.symbol))
    view.check_requested.emit()
    assert done == [1]


def test_element_wrong_answer_gives_new_sample_and_reveals_answer():
    presenter, view, done, _ = _element(
        lambda t: f"{t.unknown.mass * 1.5:.1f}",
        lambda t: t.options.index(t.unknown.symbol))
    old = presenter._task.unknown
    view.check_requested.emit()
    assert done == []
    text, ok = view.last("show_feedback")
    assert ok is False and old.symbol in text and "неточно" in text
    assert view.called("clear_inputs")


def test_element_invalid_mass_is_not_an_attempt():
    presenter, view, done, _ = _element(lambda t: "abc", lambda t: 0)
    view.check_requested.emit()
    assert presenter._attempts == 0


# --- задание 3 --------------------------------------------------------

def test_alloy_flow():
    done = []
    state = {"option": None}
    view = task_view("check_requested", selected_option=lambda: state["option"])
    presenter = AlloyPresenter(view, random.Random(5), np_rng(), done.append)
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
    assert done == [2]


# --- приложение целиком -----------------------------------------------

def test_full_lab_flow_produces_report():
    view = FakeMainView()
    clock = FakeClock()
    app = AppPresenter(view, random.Random(6), np_rng(), clock=clock,
                       now=lambda: datetime(2026, 10, 4, 12, 30))
    app.start()
    assert view.last("show_page") == ("login",)
    assert view.last("set_close_confirmation") == (None,)

    view.login.start_requested.emit()
    assert view.last("show_page") == ("quiz",)
    assert view.last("set_close_confirmation")[0] is not None

    clock.advance(300)
    app._stages["quiz"]._on_completed(1)
    assert view.last("show_page") == ("task1",)
    clock.advance(600)
    view.tick.emit()
    assert "00:10:00" in view.last("set_timer")[0]
    app._stages["task1"]._on_completed(2)
    clock.advance(400)
    app._stages["task2"]._on_completed(1)
    clock.advance(500)
    app._stages["task3"]._on_completed(3)

    assert view.last("show_page") == ("report",)
    report = view.report.last("show_report")[0]
    assert report.student == "Иванов И. И." and report.group == "ФИЗ-101"
    assert [r.duration for r in report.rows] == ["00:05:00", "00:10:00", "00:06:40", "00:08:20"]
    assert report.rows[3].attempts == "3 попытки"
    assert report.total == "00:30:00"
    assert report.verdict == "ЗАЧТЕНО"
    assert report.finished_at == "04.10.2026 12:30"
    assert view.last("set_stages")[0][-1] == ("Итог", "current")

    view.report.new_session_requested.emit()
    assert app.session is None
    assert view.last("show_page") == ("login",)
