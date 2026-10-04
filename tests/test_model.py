import random

import numpy as np
import pytest

from masslab.model.alloys import ALLOYS
from masslab.model.elements import ELEMENTS
from masslab.model.physics import TOFPhysics
from masslab.model.question_bank import QUESTION_BANK
from masslab.model.quiz import MAX_OPTIONS, MIN_OPTIONS, Quiz
from masslab.model.session import STAGES, LabSession
from masslab.model.spectrum import Peak, build_spectrum, find_peaks, time_window
from masslab.model.tasks import (CALIBRANTS, ELEMENT_POOL, AlloyTask, DemoTask,
                                 ElementTask, times_text)
from tests.fakes import FakeClock


# --- физика -----------------------------------------------------------

def test_hydrogen_flight_time():
    assert TOFPhysics(3000, 1.2).flight_time(1.008) == pytest.approx(1.583e-6, rel=1e-3)


def test_mass_from_time_inverts_flight_time():
    tof = TOFPhysics(5000)
    assert tof.mass_from_time(tof.flight_time(55.845)) == pytest.approx(55.845)


def test_flight_time_scales_with_sqrt_mass_and_voltage():
    tof = TOFPhysics(2000)
    assert tof.flight_time(64) / tof.flight_time(16) == pytest.approx(2)
    assert TOFPhysics(8000).flight_time(16) / tof.flight_time(16) == pytest.approx(0.5)


# --- спектр -----------------------------------------------------------

def test_spectrum_peak_at_flight_time():
    tof = TOFPhysics(3000)
    peaks = [Peak(28.085, 1.0)]
    t, s = build_spectrum(peaks, tof, time_window(tof, peaks), np.random.default_rng(0),
                          noise=0.0)
    assert t[np.argmax(s)] == pytest.approx(tof.flight_time(28.085), abs=5e-9)
    assert s.max() == pytest.approx(1.0, abs=1e-3)


def test_find_peaks_resolves_copper_isotopes():
    tof = TOFPhysics(3000)
    peaks = [Peak(m, a) for m, a in ELEMENTS["Cu"].isotope_peaks()]
    t, s = build_spectrum(peaks, tof, time_window(tof, peaks), np.random.default_rng(0),
                          noise=0.01)
    masses = [tof.mass_from_time(x) for x, _ in find_peaks(t, s)]
    assert [round(m) for m in masses] == [63, 65]


# --- элементы и сплавы ------------------------------------------------

@pytest.mark.parametrize("element", [e for e in ELEMENTS.values() if e.isotopes],
                         ids=lambda e: e.symbol)
def test_isotope_abundances_and_mean_mass(element):
    assert sum(i.abundance for i in element.isotopes) == pytest.approx(1)
    mean = sum(i.mass * i.abundance for i in element.isotopes)
    assert mean == pytest.approx(element.mass, rel=2e-3)


@pytest.mark.parametrize("alloy", ALLOYS, ids=lambda a: a.name)
def test_alloy_composition(alloy):
    assert sum(w for _, w in alloy.composition) == pytest.approx(100)
    assert all(ELEMENTS[s].isotopes for s in alloy.symbols)
    assert sum(alloy.atom_fractions().values()) == pytest.approx(1)


def test_alloy_names_unique():
    assert len({a.name for a in ALLOYS}) == len(ALLOYS)


# --- банк вопросов и тест ---------------------------------------------

def test_question_bank_size():
    assert len(QUESTION_BANK) >= 30


@pytest.mark.parametrize("question", QUESTION_BANK, ids=lambda q: q.text[:40])
def test_bank_question_format(question):
    options = [question.correct, *question.wrong]
    assert question.text.strip()
    assert MIN_OPTIONS <= len(options) <= MAX_OPTIONS
    assert len(set(options)) == len(options)
    assert all(o.strip() for o in options)


def test_bank_texts_unique():
    assert len({q.text for q in QUESTION_BANK}) == len(QUESTION_BANK)


def _answer_all(quiz, correct):
    for i, q in enumerate(quiz.questions):
        quiz.answer(i, q.correct_index if correct > i else (q.correct_index + 1) % len(q.options))


def test_quiz_attempt_has_ten_distinct_questions():
    quiz = Quiz(QUESTION_BANK, random.Random(1))
    quiz.new_attempt()
    assert len({q.text for q in quiz.questions}) == 10
    assert not quiz.all_answered()


def test_quiz_retry_uses_new_questions():
    quiz = Quiz(QUESTION_BANK, random.Random(2))
    quiz.new_attempt()
    first = {q.text for q in quiz.questions}
    quiz.new_attempt()
    assert first.isdisjoint(q.text for q in quiz.questions)


@pytest.mark.parametrize("correct, passed", [(10, True), (9, True), (8, False), (0, False)])
def test_quiz_pass_threshold(correct, passed):
    quiz = Quiz(QUESTION_BANK, random.Random(3))
    quiz.new_attempt()
    _answer_all(quiz, correct)
    assert quiz.score() == correct
    assert quiz.passed() is passed
    assert len(quiz.wrong_questions()) == 10 - correct


def test_options_are_shuffled():
    quiz = Quiz(QUESTION_BANK, random.Random(4))
    quiz.new_attempt()
    assert {q.correct_index for q in quiz.questions} != {0}


# --- задания ----------------------------------------------------------

def test_times_text():
    assert times_text(2) == "в 2 раза"
    assert times_text(9) == "в 9 раз"


@pytest.mark.parametrize("seed", range(20))
def test_demo_questions_are_well_formed(seed):
    task = DemoTask(random.Random(seed))
    assert len(task.questions) == 3
    for q in task.questions:
        assert MIN_OPTIONS <= len(q.options) <= MAX_OPTIONS
        assert len(set(q.options)) == len(q.options)
    assert all(task.check([q.correct_index for q in task.questions]))


def test_demo_arrival_question_answer_is_lightest_or_heaviest():
    task = DemoTask(random.Random(5))
    q = task.questions[0]
    symbols = [o.split()[0] for o in q.options]
    masses = [ELEMENTS[s].mass for s in symbols]
    answer = masses[q.correct_index]
    assert answer in (min(masses), max(masses))


def test_element_task_check():
    task = ElementTask(random.Random(6))
    assert task.unknown.symbol in task.options
    assert len(set(task.options)) == 5
    right = task.options.index(task.unknown.symbol)
    wrong = (right + 1) % 5
    m = task.unknown.mass
    assert task.check(m * 1.015, right).passed
    assert not task.check(m * 1.03, right).mass_ok
    result = task.check(m, wrong)
    assert result.mass_ok and not result.element_ok


def test_element_pool_separated_from_calibrants():
    for s in ELEMENT_POOL:
        for c in CALIBRANTS:
            assert abs(ELEMENTS[s].mass - ELEMENTS[c].mass) / ELEMENTS[c].mass > 0.05


def test_element_task_has_unlabelled_unknown_peak():
    task = ElementTask(random.Random(7))
    labels = [p.label for p in task.peaks()]
    assert labels.count("?") == 1
    assert set(labels) - {"?"} == set(CALIBRANTS)


@pytest.mark.parametrize("seed", range(10))
def test_alloy_task_options(seed):
    task = AlloyTask(random.Random(seed))
    assert task.alloy in task.options
    assert len({a.name for a in task.options}) == 5
    assert task.check(task.options.index(task.alloy))


# --- сессия -----------------------------------------------------------

def test_session_times_each_stage():
    clock = FakeClock()
    session = LabSession("Иванов", "ФИЗ-101", clock)
    for i, stage in enumerate(STAGES):
        session.start_stage(stage)
        clock.advance(60 * (i + 1))
        assert not session.completed
        session.finish_stage(stage, attempts=i + 1)
    assert session.completed
    assert [session.stage_duration(s) for s in STAGES] == [60, 120, 180, 240]
    assert session.total_duration() == 600
    clock.advance(1000)
    assert session.total_duration() == 600
