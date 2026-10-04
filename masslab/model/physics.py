"""Физика времяпролётного (TOF) масс-спектрометра.

Ион с зарядом q = z·e, прошедший ускоряющую разность потенциалов U,
приобретает кинетическую энергию qU = mv²/2 и пролетает бесполевую
дрейфовую трубку длиной L за время t = L·√(m / 2qU).

Ширина пика складывается из двух независимых вкладов:
  • временного разрешения детектора σ_д (не зависит от U);
  • разброса начальной энергии ионов ΔE: относительный разброс энергии
    ΔE / qU даёт разброс времени σ_E = t·ΔE / (2qU).
При большом U пики узкие, но стоят тесно (Δt ∝ 1/√U); при малом U пики
далеко друг от друга, но расплываются из-за ΔE. Поэтому существует
оптимальное напряжение, при котором разрешающая способность максимальна.
"""
from dataclasses import dataclass

E_CHARGE = 1.602176634e-19  # Кл, элементарный заряд
AMU = 1.66053906660e-27     # кг, атомная единица массы
FWHM_FACTOR = 2.3548        # полная ширина на полувысоте = 2,3548·σ

VOLTAGE_MIN = 500           # В
VOLTAGE_MAX = 20000         # В
VOLTAGE_DEFAULT = 3000      # В
VOLTAGE_STEP = 100          # В

DRIFT_LENGTH = 1.2          # м
LENGTH_MIN = 0.5            # м
LENGTH_MAX = 3.0            # м
LENGTH_STEP = 0.1           # м

ENERGY_SPREAD = 2.0         # эВ, разброс начальной энергии ионов
DETECTOR_SIGMA = 8e-9       # с, временное разрешение детектора


@dataclass(frozen=True)
class TOFPhysics:
    voltage: float = VOLTAGE_DEFAULT
    length: float = DRIFT_LENGTH
    energy_spread: float = ENERGY_SPREAD
    detector_sigma: float = DETECTOR_SIGMA

    def velocity(self, mass_u, charge=1):
        """Скорость иона после ускорения, м/с."""
        return (2 * charge * E_CHARGE * self.voltage / (mass_u * AMU)) ** 0.5

    def flight_time(self, mass_u, charge=1):
        """Время пролёта дрейфовой трубки, с."""
        return self.length / self.velocity(mass_u, charge)

    def mass_from_time(self, t, charge=1):
        """Масса иона (а.е.м.) по времени пролёта t (с). При charge=1 это m/z."""
        return 2 * charge * E_CHARGE * self.voltage * t ** 2 / (self.length ** 2 * AMU)

    def sigma_at_time(self, t, charge=1):
        """Стандартное отклонение пика, прилетающего в момент t, с."""
        spread = t * self.energy_spread / (2 * charge * self.voltage)
        return (self.detector_sigma ** 2 + spread ** 2) ** 0.5

    def peak_sigma(self, mass_u, charge=1):
        return self.sigma_at_time(self.flight_time(mass_u, charge), charge)

    def resolution(self, mass_u, charge=1):
        """Разрешающая способность R = m/Δm = t / (2·Δt), Δt — ширина пика на полувысоте."""
        t = self.flight_time(mass_u, charge)
        return t / (2 * FWHM_FACTOR * self.sigma_at_time(t, charge))
