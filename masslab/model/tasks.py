"""Модели трёх заданий лабораторной работы.

Общее правило: при неверном ответе выдаётся новое задание, чтобы ответ
нельзя было подобрать перебором вариантов.
"""
from dataclasses import dataclass

from masslab.model.alloys import ALLOYS
from masslab.model.elements import ELEMENTS
from masslab.model.physics import TOFPhysics
from masslab.model.quiz import ChoiceQuestion
from masslab.model.spectrum import Peak


def times_text(n):
    """«в 2 раза», «в 9 раз»."""
    return f"в {n} раза" if n in (2, 3, 4) else f"в {n} раз"


def element_label(symbol):
    e = ELEMENTS[symbol]
    return f"{symbol} ({e.name})"


# ---------------------------------------------------------------- Задание 1

# Элементы с заметно различающимися массами (без пар Ar/K/Ca, Co/Ni)
DEMO_POOL = ("H", "He", "Li", "C", "N", "O", "Ne", "Na", "Mg", "Al", "Si",
             "Ar", "Ti", "Fe", "Cu", "Kr", "Ag", "Xe", "Au", "Pb")
DEMO_VOLTAGES = (1000, 1500, 2000, 4000, 5000, 8000, 10000)


class DemoTask:
    """Задание 1: контрольные вопросы, ответы на которые получают с помощью симулятора."""

    def __init__(self, rng):
        self._rng = rng
        self.questions = []
        self.new_questions()

    def new_questions(self):
        self.questions = [self._arrival_question(), self._voltage_question(),
                          self._measure_question()]

    def check(self, answers):
        return [q.is_correct(a) for q, a in zip(self.questions, answers)]

    def _arrival_question(self):
        symbols = self._rng.sample(DEMO_POOL, 4)
        last = self._rng.random() < 0.5
        pick = max if last else min
        answer = pick(symbols, key=lambda s: ELEMENTS[s].mass)
        names = ", ".join(symbols)
        text = (f"Ионы {names} (z = +1) ускорены одним и тем же напряжением. "
                f"Какой из них {'последним' if last else 'первым'} достигнет детектора?")
        wrong = [element_label(s) for s in symbols if s != answer]
        return ChoiceQuestion.create(text, element_label(answer), wrong, self._rng)

    def _voltage_question(self):
        k = self._rng.choice((4, 9))
        r = int(round(k ** 0.5))
        up = self._rng.random() < 0.5
        text = (f"Как изменится время пролёта иона, если ускоряющее напряжение "
                f"{'увеличить' if up else 'уменьшить'} {times_text(k)}?")
        same, opposite = ("Уменьшится", "Увеличится") if up else ("Увеличится", "Уменьшится")
        correct = f"{same} {times_text(r)}"
        wrong = [f"{opposite} {times_text(r)}", f"{same} {times_text(k)}",
                 f"{opposite} {times_text(k)}", "Не изменится"]
        return ChoiceQuestion.create(text, correct, wrong, self._rng)

    def _measure_question(self):
        symbol = self._rng.choice(DEMO_POOL)
        voltage = self._rng.choice(DEMO_VOLTAGES)
        t_us = TOFPhysics(voltage).flight_time(ELEMENTS[symbol].mass) * 1e6
        factors = self._rng.sample((0.5, 0.71, 1.41, 2.0), 3)
        text = (f"Установите U = {voltage} В, запустите ион {symbol} и определите "
                f"время его пролёта до детектора.")
        correct = _us(t_us)
        wrong = [_us(t_us * f) for f in factors]
        return ChoiceQuestion.create(text, correct, wrong, self._rng)


def _us(t_us):
    return f"{t_us:.2f} мкс".replace(".", ",")


# ---------------------------------------------------------------- Задание 2

# Неизвестный элемент: массы различаются между собой и с калибрантами > 5 %
ELEMENT_POOL = ("Li", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
                "S", "Cl", "Ti", "Cr", "Fe", "Cu", "Zn", "Kr", "Ag", "Sn")
CALIBRANTS = ("He", "Ar", "Xe")
MASS_TOLERANCE = 0.02   # допустимая относительная погрешность массы


@dataclass(frozen=True)
class ElementCheck:
    mass_ok: bool
    element_ok: bool

    @property
    def passed(self):
        return self.mass_ok and self.element_ok


class ElementTask:
    """Задание 2: определить массу и элемент по спектру с калибровочной смесью."""

    def __init__(self, rng, n_options=5):
        self._rng = rng
        self._n_options = n_options
        self.unknown = None
        self.options = []
        self._intensity = 1.0
        self.new_sample()

    def new_sample(self):
        symbol = self._rng.choice(ELEMENT_POOL)
        self.unknown = ELEMENTS[symbol]
        others = self._rng.sample([s for s in ELEMENT_POOL if s != symbol], self._n_options - 1)
        options = [symbol, *others]
        self._rng.shuffle(options)
        self.options = options
        self._intensity = self._rng.uniform(0.7, 1.0)

    def peaks(self):
        calibrants = [Peak(ELEMENTS[s].mass, 0.6, s) for s in CALIBRANTS]
        return calibrants + [Peak(self.unknown.mass, self._intensity, "?")]

    def check(self, mass, option_index):
        mass_ok = abs(mass - self.unknown.mass) <= MASS_TOLERANCE * self.unknown.mass
        element_ok = self.options[option_index] == self.unknown.symbol
        return ElementCheck(mass_ok, element_ok)


# ---------------------------------------------------------------- Задание 3

class AlloyTask:
    """Задание 3: определить сплав по изотопному масс-спектру."""

    def __init__(self, rng, n_options=5):
        self._rng = rng
        self._n_options = n_options
        self.alloy = None
        self.options = []
        self.new_sample()

    def new_sample(self):
        self.alloy = self._rng.choice(ALLOYS)
        others = [a for a in ALLOYS if a is not self.alloy]
        self._rng.shuffle(others)
        # Сначала сплавы с общими элементами — так варианты труднее различить
        shared = set(self.alloy.symbols)
        others.sort(key=lambda a: -len(shared & set(a.symbols)))
        options = [self.alloy, *others[:self._n_options - 1]]
        self._rng.shuffle(options)
        self.options = options

    def peaks(self):
        return self.alloy.peaks()

    def check(self, option_index):
        return self.options[option_index] is self.alloy
