"""Сплавы для задания 3. Состав — в массовых процентах."""
from dataclasses import dataclass

from masslab.model.elements import ELEMENTS
from masslab.model.spectrum import Peak


@dataclass(frozen=True)
class Alloy:
    name: str
    composition: tuple   # ((символ, массовая доля %), ...)

    @property
    def symbols(self):
        return tuple(symbol for symbol, _ in self.composition)

    @property
    def major_symbols(self):
        """Основные элементы (их пики хорошо видны)."""
        return tuple(s for s, w in self.composition if w >= MINOR_PERCENT)

    @property
    def minor_symbols(self):
        """Малые добавки: отмечать в задании необязательно."""
        return tuple(s for s, w in self.composition if w < MINOR_PERCENT)

    def composition_text(self):
        return ", ".join(f"{symbol} {_percent(w)} %" for symbol, w in self.composition)

    def atom_fractions(self):
        """Мольные (атомные) доли: интенсивность пика пропорциональна числу атомов."""
        moles = {s: w / ELEMENTS[s].mass for s, w in self.composition}
        total = sum(moles.values())
        return {s: n / total for s, n in moles.items()}

    def peaks(self):
        """Изотопные пики сплава (без подписей — их нужно определить)."""
        result = []
        for symbol, fraction in self.atom_fractions().items():
            for mass, abundance in ELEMENTS[symbol].isotope_peaks():
                result.append(Peak(mass, fraction * abundance))
        return result


MINOR_PERCENT = 2.0    # добавки меньше 2 % по массе считаются малыми


def _percent(w):
    return f"{w:g}".replace(".", ",")


ALLOYS = (
    Alloy("Латунь Л63", (("Cu", 63), ("Zn", 37))),
    Alloy("Оловянная бронза БрО10", (("Cu", 90), ("Sn", 10))),
    Alloy("Мельхиор МН19", (("Cu", 81), ("Ni", 19))),
    Alloy("Константан МНМц40-1,5", (("Cu", 58.5), ("Ni", 40), ("Mn", 1.5))),
    Alloy("Нейзильбер МНЦ15-20", (("Cu", 65), ("Zn", 20), ("Ni", 15))),
    Alloy("Нержавеющая сталь 12Х18Н10Т", (("Fe", 71), ("Cr", 18), ("Ni", 10), ("Ti", 1))),
    Alloy("Инвар 36Н", (("Fe", 64), ("Ni", 36))),
    Alloy("Нихром Х20Н80", (("Ni", 80), ("Cr", 20))),
    Alloy("Дуралюмин Д16", (("Al", 93.5), ("Cu", 4.4), ("Mg", 1.5), ("Mn", 0.6))),
    Alloy("Силумин АК12", (("Al", 88), ("Si", 12))),
    Alloy("Припой ПОС-61", (("Sn", 61), ("Pb", 39))),
    Alloy("Титановый сплав ВТ6", (("Ti", 90), ("Al", 6), ("V", 4))),
)


# Элементы, из которых студент выбирает состав образца (по возрастанию массы)
ALLOY_ELEMENTS = tuple(sorted({s for a in ALLOYS for s in a.symbols},
                              key=lambda s: ELEMENTS[s].mass))
