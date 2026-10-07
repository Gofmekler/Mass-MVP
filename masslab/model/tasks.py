"""Модели трёх заданий лабораторной работы.

Общее правило: неверно решённое заменяется новым (вопрос или образец),
чтобы ответ нельзя было подобрать перебором вариантов.
"""
from dataclasses import dataclass

from masslab.model.alloys import ALLOYS
from masslab.model.elements import ELEMENTS
from masslab.model.physics import DRIFT_LENGTH, TOFPhysics
from masslab.model.quiz import ChoiceQuestion
from masslab.model.spectrum import Peak

SUPERSCRIPT_CHARGE = {1: "⁺", 2: "²⁺", 3: "³⁺"}


def ion_label(symbol, charge=1):
    return symbol + SUPERSCRIPT_CHARGE[charge]


def times_text(n):
    """«в 2 раза», «в 9 раз»."""
    return f"в {n} раза" if n in (2, 3, 4) else f"в {n} раз"


def element_label(symbol):
    return f"{symbol} ({ELEMENTS[symbol].name})"


def _num(value, digits):
    return f"{value:.{digits}f}".replace(".", ",")


# ---------------------------------------------------------------- Задание 1

# Элементы с заметно различающимися массами (без пар Ar/K/Ca, Co/Ni)
DEMO_POOL = ("H", "He", "Li", "C", "N", "O", "Ne", "Na", "Mg", "Al", "Si",
             "Ar", "Ti", "Fe", "Cu", "Kr", "Ag", "Xe", "Au", "Pb")
DEMO_VOLTAGES = (1000, 1500, 2000, 4000, 5000, 8000, 10000)
# Двухзарядный ион X²⁺ прилетает одновременно с однозарядным Y⁺ (m_X/2 ≈ m_Y)
DOUBLE_CHARGE_PAIRS = (("Ar", "Ne"), ("Fe", "Si"), ("Mg", "C"), ("Si", "N"))
OPTIMUM_ELEMENTS = ("N", "Cu", "Ag", "Pb")
OPTIMUM_VOLTAGES = (500, 1000, 2000, 4000, 8000, 16000)

DEMO_HINTS = (
    "Отвечайте по одному вопросу: настройте прибор так, как сказано в вопросе, "
    "нажмите «Запустить ионы» и посмотрите на таблицу «Результаты запуска».",
    "Время пролёта и разрешение R каждого иона указаны в таблице результатов. "
    "Точное напряжение удобно вводить в поле справа от ползунка. "
    "Не забудьте вернуть длину трубки L = 1,2 м.",
    "Двухзарядный ион X²⁺ ведёт себя как однозарядный ион с массой m/2: прибор "
    "измеряет отношение m/z. Включите «Двухзарядные ионы» и сравните время.",
)


class DemoTask:
    """Задание 1: контрольные вопросы, ответы на которые получают с помощью симулятора."""

    def __init__(self, rng):
        self._rng = rng
        self._makers = (self._arrival_question, self._voltage_question,
                        self._measure_question, self._double_charge_question,
                        self._optimum_question)
        self.questions = [make() for make in self._makers]

    def check(self, answers):
        return [q.is_correct(a) for q, a in zip(self.questions, answers)]

    def replace(self, indices):
        """Заменить вопросы с указанными номерами новыми того же типа."""
        for i in indices:
            old = self.questions[i]
            for _ in range(20):
                new = self._makers[i]()
                if new.text != old.text:
                    break
            self.questions[i] = new

    def _arrival_question(self):
        symbols = self._rng.sample(DEMO_POOL, 4)
        last = self._rng.random() < 0.5
        pick = max if last else min
        answer = pick(symbols, key=lambda s: ELEMENTS[s].mass)
        text = (f"Ионы {', '.join(symbols)} (z = +1) ускорены одним и тем же напряжением. "
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
        text = (f"Установите U = {voltage} В и L = 1,2 м, запустите ион {ion_label(symbol)} "
                f"и определите время его пролёта до детектора.")
        correct = _num(t_us, 2) + " мкс"
        wrong = [_num(t_us * f, 2) + " мкс" for f in factors]
        return ChoiceQuestion.create(text, correct, wrong, self._rng)

    def _double_charge_question(self):
        heavy, light = self._rng.choice(DOUBLE_CHARGE_PAIRS)
        others = [s for s in DEMO_POOL if s not in (heavy, light)
                  and abs(ELEMENTS[s].mass - ELEMENTS[light].mass) / ELEMENTS[light].mass > 0.1]
        wrong = [ion_label(heavy)] + [ion_label(s) for s in self._rng.sample(others, 3)]
        text = (f"Запустите ионы {heavy} с включёнными двухзарядными ионами. С пиком какого "
                f"однозарядного иона практически совпадает по времени пик {ion_label(heavy, 2)}?")
        return ChoiceQuestion.create(text, ion_label(light), wrong, self._rng)

    def _optimum_question(self):
        symbol = self._rng.choice(OPTIMUM_ELEMENTS)
        mass = ELEMENTS[symbol].mass
        while True:
            voltages = self._rng.sample(OPTIMUM_VOLTAGES, 4)
            ranked = sorted(voltages, key=lambda u: -TOFPhysics(u, DRIFT_LENGTH).resolution(mass))
            best, second = (TOFPhysics(u).resolution(mass) for u in ranked[:2])
            if best / second >= 1.1:
                break
        text = (f"При L = 1,2 м запустите ион {ion_label(symbol)} и подберите напряжение, при "
                f"котором разрешающая способность R для него наибольшая. Какое это напряжение?")
        options = [f"{u} В" for u in ranked]
        return ChoiceQuestion.create(text, options[0], options[1:], self._rng)


# ---------------------------------------------------------------- Задание 2

# Неизвестный элемент: массы отличаются от калибрантов > 5 % и от пиков остаточного
# газа > 2 % (Si совпал бы с N₂⁺, S — с O₂⁺, поэтому их нет)
ELEMENT_POOL = ("Li", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "P",
                "Cl", "Ti", "Cr", "Fe", "Cu", "Zn", "Kr", "Ag", "Sn")
CALIBRANTS = ("He", "Ar", "Xe")
TIME_TOLERANCE = 0.015  # допустимая относительная погрешность времени
MASS_TOLERANCE = 0.02   # допустимая относительная погрешность массы

ELEMENT_HINTS = (
    "Неизвестный ион — пик с красной меткой «?», его траектория — красная пунктирная. "
    "Наведите курсор на этот пик: внизу появится время пролёта.",
    "Масса по формуле: m = 2eU·t² / L², где e = 1,602·10⁻¹⁹ Кл, t — в секундах "
    "(1 мкс = 10⁻⁶ с), результат в килограммах разделите на 1,6605·10⁻²⁷ кг.",
    "Проще через калибрант: m = m_к·(t / t_к)². Например, для Ar: m_к = 39,948, "
    "t_к — время пика Ar при том же напряжении.",
)


@dataclass(frozen=True)
class ElementCheck:
    time_ok: bool
    mass_ok: bool
    calculation_ok: bool   # масса согласуется с введённым временем
    element_ok: bool

    @property
    def passed(self):
        return self.time_ok and self.mass_ok and self.element_ok


class ElementTask:
    """Задание 2: измерить время, вычислить массу и определить элемент."""

    def __init__(self, rng, n_options=5):
        self._rng = rng
        self._n_options = n_options
        self.unknown = None
        self.options = []
        self._intensity = 1.0
        self.new_sample()

    def new_sample(self):
        previous = self.unknown.symbol if self.unknown else None
        symbol = self._rng.choice([s for s in ELEMENT_POOL if s != previous])
        self.unknown = ELEMENTS[symbol]
        others = self._rng.sample([s for s in ELEMENT_POOL if s != symbol], self._n_options - 1)
        options = [symbol, *others]
        self._rng.shuffle(options)
        self.options = options
        self._intensity = self._rng.uniform(0.7, 1.0)

    def peaks(self):
        calibrants = [Peak(ELEMENTS[s].mass, 0.6, ion_label(s)) for s in CALIBRANTS]
        return calibrants + [Peak(self.unknown.mass, self._intensity, "?")]

    def check(self, tof, time_s, mass, option_index):
        true_time = tof.flight_time(self.unknown.mass)
        time_ok = abs(time_s - true_time) <= TIME_TOLERANCE * true_time
        mass_ok = abs(mass - self.unknown.mass) <= MASS_TOLERANCE * self.unknown.mass
        expected = tof.mass_from_time(time_s)
        calculation_ok = abs(mass - expected) <= MASS_TOLERANCE * expected
        element_ok = self.options[option_index] == self.unknown.symbol
        return ElementCheck(time_ok, mass_ok, calculation_ok, element_ok)


# ---------------------------------------------------------------- Задание 3

ALLOY_HINTS = (
    "Наведите курсор на каждый крупный пик и запишите m/z. Включите логарифмическую "
    "шкалу — так видны и слабые пики.",
    "Сравните найденные m/z с изотопами во вкладке «Справочник элементов»: например, "
    "у меди пики 63 и 65 (≈ 69 % и 31 %), у никеля — 58 и 60.",
    "Соотношение высот пиков разных элементов показывает их долю в сплаве. Если "
    "пики близких масс сливаются, подберите напряжение с лучшим разрешением.",
)


class AlloyTask:
    """Задание 3: определить сплав по изотопному масс-спектру."""

    def __init__(self, rng, n_options=5):
        self._rng = rng
        self._n_options = n_options
        self.alloy = None
        self.options = []
        self.new_sample()

    def new_sample(self):
        previous = self.alloy
        self.alloy = self._rng.choice([a for a in ALLOYS if a is not previous])
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
