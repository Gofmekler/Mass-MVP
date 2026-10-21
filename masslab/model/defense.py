"""Защита работы: вопросы по результатам, которые получил сам студент.

Вопросы строятся из данных заданий 2 и 3 (элемент, измеренное время, сплав),
поэтому у каждой пары они свои. Неверно решённый вопрос заменяется новым
того же типа, как в задании 1.
"""
from dataclasses import dataclass

from masslab.model.alloys import ALLOY_ELEMENTS
from masslab.model.elements import ELEMENTS
from masslab.model.physics import VOLTAGE_MAX, VOLTAGE_MIN
from masslab.model.quiz import ChoiceQuestion
from masslab.model.tasks import CALIBRANTS, _num, ion_label, times_text

VOLTAGE_FACTORS = (4, 9)


@dataclass(frozen=True)
class ElementResult:
    """Итог задания 2: определённый элемент и условия измерения."""
    symbol: str
    voltage: int
    length: float
    time_us: float           # время пролёта, измеренное студентом


def _us(value):
    return _num(value, 2) + " мкс"


class DefenseTask:
    def __init__(self, rng, element, alloy):
        self._rng = rng
        self._element = element
        self._alloy = alloy
        self._makers = (self._arrival_question, self._voltage_question,
                        self._double_charge_question, self._isotopes_question,
                        self._heaviest_question)
        self.questions = [make() for make in self._makers]

    def check(self, answers):
        return [q.is_correct(a) for q, a in zip(self.questions, answers)]

    def replace(self, indices):
        """Заменить вопросы с указанными номерами новыми того же типа."""
        for i in indices:
            old = self.questions[i]
            for _ in range(20):
                new = self._makers[i]()
                if new.text != old.text or new.options != old.options:
                    break
            self.questions[i] = new

    # --- задание 2 ------------------------------------------------------

    def _arrival_question(self):
        e = ELEMENTS[self._element.symbol]
        calibrant = ELEMENTS[self._rng.choice([s for s in CALIBRANTS if s != e.symbol])]
        later = e.mass > calibrant.mass
        text = (f"В задании 2 ваш неизвестный ион {ion_label(e.symbol)} прилетел к детектору "
                f"{'позже' if later else 'раньше'}, чем калибрант {ion_label(calibrant.symbol)}. "
                f"Почему?")
        correct = (f"Масса {e.symbol} {'больше' if later else 'меньше'}, чем у "
                   f"{calibrant.symbol}: при одинаковой энергии qU "
                   f"{'тяжёлый ион летит медленнее' if later else 'лёгкий ион летит быстрее'}")
        wrong = [f"Заряд иона {ion_label(e.symbol)} {'меньше' if later else 'больше'}, "
                 f"чем у {ion_label(calibrant.symbol)}",
                 f"Ион {e.symbol} ускорялся {'меньшим' if later else 'большим'} напряжением",
                 f"Ион {e.symbol} {'позже' if later else 'раньше'} вылетел из источника",
                 "Время пролёта не зависит от массы — это случайный разброс"]
        return ChoiceQuestion.create(text, correct, wrong, self._rng)

    def _voltage_question(self):
        r = self._element
        # новое напряжение — целое и в пределах прибора
        variants = [(k, up) for k in VOLTAGE_FACTORS for up in (True, False)
                    if (r.voltage * k <= VOLTAGE_MAX if up
                        else r.voltage % k == 0 and r.voltage // k >= VOLTAGE_MIN)]
        k, up = self._rng.choice(variants or [(VOLTAGE_FACTORS[0], True)])
        root = int(round(k ** 0.5))
        new_voltage = r.voltage * k if up else r.voltage // k
        t = r.time_us
        faster, slower = t / root, t * root
        text = (f"В задании 2 вы измерили t = {_us(t)} для иона {ion_label(r.symbol)} при "
                f"U = {r.voltage} В. Каким будет его время пролёта при U = {new_voltage} В "
                f"(напряжение изменено {times_text(k)}, длина трубки та же)?")
        correct = _us(faster if up else slower)
        wrong = [_us(slower if up else faster), _us(t / k if up else t * k),
                 _us(t * k if up else t / k), _us(t) + " — не изменится"]
        return ChoiceQuestion.create(text, correct, wrong, self._rng)

    def _double_charge_question(self):
        r = self._element
        t = r.time_us
        text = (f"Вы измерили t = {_us(t)} для иона {ion_label(r.symbol)}. При каком времени "
                f"появился бы пик того же элемента, будь ион двухзарядным "
                f"({ion_label(r.symbol, 2)})?")
        correct = f"{_us(t / 2 ** 0.5)} (в √2 раза раньше)"
        wrong = [f"{_us(t / 2)} (в 2 раза раньше)", f"{_us(t * 2 ** 0.5)} (в √2 раза позже)",
                 f"{_us(t * 2)} (в 2 раза позже)", f"{_us(t)} (заряд не влияет на время)"]
        return ChoiceQuestion.create(text, correct, wrong, self._rng)

    # --- задание 3 ------------------------------------------------------

    def _isotopes_question(self):
        alloy = self._alloy
        multi = [s for s in alloy.major_symbols if len(ELEMENTS[s].isotopes) > 1]
        symbol = self._rng.choice(multi or list(alloy.major_symbols))
        text = (f"В задании 3 элемент {symbol} ({ELEMENTS[symbol].name}) в спектре сплава "
                f"«{alloy.name}» дал несколько пиков рядом. Почему?")
        correct = "У элемента несколько стабильных изотопов с разными массами"
        wrong = ["Часть ионов была двухзарядной, а часть — однозарядной",
                 "Это пики остаточного газа в камере",
                 "Детектор регистрирует каждый ион несколько раз",
                 "Ионы ускорялись разными напряжениями"]
        return ChoiceQuestion.create(text, correct, wrong, self._rng)

    def _heaviest_question(self):
        alloy = self._alloy
        majors = list(alloy.major_symbols)
        heaviest = max(majors, key=lambda s: ELEMENTS[s].mass)
        others = [s for s in majors if s != heaviest]
        # отвлекающие: сначала более лёгкие элементы, затем тяжёлые, которых в сплаве нет
        lighter = [s for s in ALLOY_ELEMENTS if s not in majors
                   and ELEMENTS[s].mass < ELEMENTS[heaviest].mass]
        absent = [s for s in ALLOY_ELEMENTS if s not in majors and s not in lighter]
        for pool in (lighter, absent):
            need = max(0, 3 - len(others))
            others += self._rng.sample(pool, min(need, len(pool)))
        label = lambda s: f"{s} ({ELEMENTS[s].name})"
        text = (f"Пики какого элемента в спектре вашего сплава «{alloy.name}» были "
                f"дальше всех по времени пролёта?")
        return ChoiceQuestion.create(text, label(heaviest), [label(s) for s in others[:4]],
                                     self._rng)
