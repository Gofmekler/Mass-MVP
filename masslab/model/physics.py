"""Физика времяпролётного (TOF) масс-спектрометра.

Ион с зарядом q = z·e, прошедший ускоряющую разность потенциалов U,
приобретает кинетическую энергию qU = mv²/2 и пролетает бесполевую
дрейфовую трубку длиной L за время t = L·√(m / 2qU).
"""
from dataclasses import dataclass

E_CHARGE = 1.602176634e-19  # Кл, элементарный заряд
AMU = 1.66053906660e-27     # кг, атомная единица массы

DRIFT_LENGTH = 1.2          # м
VOLTAGE_MIN = 500           # В
VOLTAGE_MAX = 20000         # В
VOLTAGE_DEFAULT = 3000      # В
VOLTAGE_STEP = 100          # В


@dataclass(frozen=True)
class TOFPhysics:
    voltage: float = VOLTAGE_DEFAULT
    length: float = DRIFT_LENGTH

    def velocity(self, mass_u, charge=1):
        """Скорость иона после ускорения, м/с."""
        return (2 * charge * E_CHARGE * self.voltage / (mass_u * AMU)) ** 0.5

    def flight_time(self, mass_u, charge=1):
        """Время пролёта дрейфовой трубки, с."""
        return self.length / self.velocity(mass_u, charge)

    def mass_from_time(self, t, charge=1):
        """Масса иона (а.е.м.) по времени пролёта t (с)."""
        return 2 * charge * E_CHARGE * self.voltage * t ** 2 / (self.length ** 2 * AMU)
