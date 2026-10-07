import random

import numpy as np
import pytest

from masslab.model.alloys import ALLOYS
from masslab.model.elements import ELEMENTS
from masslab.model.physics import TOFPhysics
from masslab.model.question_bank import QUESTION_BANK
from masslab.model.quiz import MAX_OPTIONS, MIN_OPTIONS, Quiz
from masslab.model.session import STAGES, LabSession
from masslab.model.spectrum import (Peak, build_spectrum, find_peaks, residual_gas_peaks,
                                    time_window)
from masslab.model.tasks import (CALIBRANTS, DOUBLE_CHARGE_PAIRS, ELEMENT_POOL, AlloyTask,
                                 DemoTask, ElementTask, ion_label, times_text)
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

def test_resolution_has_optimum_voltage():
    """Разброс энергии ионов даёт максимум R при промежуточном напряжении."""
    def best(mass):
        return max(range(500, 20001, 100), key=lambda u: TOFPhysics(u).resolution(mass))
    assert 2000 <= best(63.5) <= 4000
    assert best(1.008) < best(63.5) < best(207.2)


def test_longer_tube_improves_resolution():
    assert TOFPhysics(3000, 2.4).resolution(63.5) > TOFPhysics(3000, 1.2).resolution(63.5)


def test_double_charge_halves_mass_to_charge():
    tof = TOFPhysics(3000)
    assert tof.flight_time(40.0, charge=2) == pytest.approx(tof.flight_time(20.0))


def _spectrum(peaks, voltage=3000, noise=0.01):
    tof = TOFPhysics(voltage)
    t, s = build_spectrum(peaks, tof, time_window(tof, peaks), np.random.default_rng(0),
                          noise=noise)
    return tof, t, s


def test_spectrum_peak_at_flight_time():
    tof, t, s = _spectrum([Peak(28.085, 1.0)], noise=0.0)
    assert t[np.argmax(s)] == pytest.approx(tof.flight_time(28.085), abs=5e-9)
    assert s.max() == pytest.approx(1.0, abs=1e-3)


def test_find_peaks_resolves_copper_isotopes_and_measures_width():
    peaks = [Peak(m, a) for m, a in ELEMENTS["Cu"].isotope_peaks()]
    tof, t, s = _spectrum(peaks)
    found = find_peaks(t, s, tof.sigma_at_time)
    assert [round(tof.mass_from_time(p.time)) for p in found] == [63, 65]
    expected = 2.3548 * tof.sigma_at_time(found[0].time)
    assert found[0].fwhm == pytest.approx(expected, rel=0.15)


@pytest.mark.parametrize("voltage, resolved", [(1000, False), (4000, True), (20000, False)])
def test_lead_isotopes_resolved_only_near_optimum(voltage, resolved):
    peaks = [Peak(m, a) for m, a in ELEMENTS["Pb"].isotope_peaks()]
    tof, t, s = _spectrum(peaks, voltage, noise=0.003)
    masses = [round(tof.mass_from_time(p.time)) for p in find_peaks(t, s, tof.sigma_at_time, 0.01)]
    assert (206 in masses and 208 in masses) is resolved


def test_broad_noisy_peak_gives_single_peak():
    tof, t, s = _spectrum([Peak(207.2, 1.0)], voltage=500, noise=0.02)
    assert len(find_peaks(t, s, tof.sigma_at_time)) == 1


def test_residual_gas_peaks_are_background():
    gas = residual_gas_peaks(0.5)
    assert {p.label for p in gas} == {"H₂O⁺", "N₂⁺", "O₂⁺"}
    assert all(p.background and p.intensity <= 0.05 for p in gas)


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
    assert len(task.questions) == 5
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


def test_demo_replace_changes_only_given_questions():
    task = DemoTask(random.Random(8))
    before = list(task.questions)
    task.replace([1, 3])
    assert task.questions[0] is before[0] and task.questions[2] is before[2]
    assert task.questions[1].text != before[1].text or task.questions[1] is not before[1]


def test_double_charge_pairs_match():
    for heavy, light in DOUBLE_CHARGE_PAIRS:
        ratio = ELEMENTS[heavy].mass / 2 / ELEMENTS[light].mass
        assert ratio == pytest.approx(1, abs=0.03)


@pytest.mark.parametrize("seed", range(10))
def test_optimum_question_answer_has_best_resolution(seed):
    q = DemoTask(random.Random(seed)).questions[4]
    symbol = next(s for s in ("N", "Cu", "Ag", "Pb") if ion_label(s) in q.text)
    voltages = [int(o.split()[0]) for o in q.options]
    best = max(voltages, key=lambda u: TOFPhysics(u).resolution(ELEMENTS[symbol].mass))
    assert voltages[q.correct_index] == best


def _element_answer(task, tof, t_factor=1.0, mass=None, right=True):
    t = tof.flight_time(task.unknown.mass) * t_factor
    m = tof.mass_from_time(t) if mass is None else mass
    index = task.options.index(task.unknown.symbol)
    return task.check(tof, t, m, index if right else (index + 1) % len(task.options))


def test_element_task_check_diagnoses_each_step():
    task = ElementTask(random.Random(6))
    tof = TOFPhysics(3000)
    assert task.unknown.symbol in task.options and len(set(task.options)) == 5
    assert _element_answer(task, tof).passed
    r = _element_answer(task, tof, t_factor=1.05)          # время неверно, расчёт по нему верен
    assert not r.time_ok and r.calculation_ok
    r = _element_answer(task, tof, mass=task.unknown.mass * 1.3)   # ошибка в расчёте
    assert r.time_ok and not r.mass_ok and not r.calculation_ok
    r = _element_answer(task, tof, right=False)
    assert r.time_ok and r.mass_ok and not r.element_ok


def test_element_task_new_sample_differs():
    task = ElementTask(random.Random(9))
    for _ in range(10):
        old = task.unknown
        task.new_sample()
        assert task.unknown is not old


def test_element_pool_separated_from_calibrants():
    for s in ELEMENT_POOL:
        for c in CALIBRANTS:
            assert abs(ELEMENTS[s].mass - ELEMENTS[c].mass) / ELEMENTS[c].mass > 0.05


def test_element_pool_separated_from_residual_gas():
    """Пик неизвестного иона не должен прятаться под подписанным пиком газа."""
    for s in ELEMENT_POOL:
        for gas in residual_gas_peaks():
            assert abs(ELEMENTS[s].mass - gas.mass) / gas.mass > 0.02, (s, gas.label)


def test_element_task_has_unlabelled_unknown_peak():
    task = ElementTask(random.Random(7))
    labels = [p.label for p in task.peaks()]
    assert labels.count("?") == 1
    assert set(labels) - {"?"} == {ion_label(c) for c in CALIBRANTS}


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
