import random
import numpy as np
from model.tof_physics import TOFPhysics
from model.spectrum import build_spectrum
from model.unknown import UnknownSample
from model.element_db import ELEMENTS, CALIBRANTS
from model.ion import Ion


class SimulationController:
    def __init__(self):
        self.U = 3000.0
        self.L = 1.2
        self.tof = TOFPhysics(self.U, self.L)
        self.sample = UnknownSample(mixture=False)

        # Параметры для спектра
        self.det_sigma = 3e-8
        self.noise = 0.02
        self.t_min = 0.0
        self.t_max = 2e-5
        self.n_points = 4000

        # Какие элементы показывать как калибровочные
        self.calibrants = CALIBRANTS

        # Элементы для сравнения
        self.comparison_elements = ["Na", "K", "Mg", "Al", "Cu", "C", "N", "O"]

        # Текущий неизвестный элемент
        self.current_unknown = None
        self._generate_unknown()

        # Для хранения данных спектра
        self.current_spectrum_data = None

        # Режим работы: одиночный элемент или смесь
        self.mode = "single"  # "single" или "mixture"

        # Предустановленные смеси газов
        self.gas_mixtures = {
            "Воздух": [
                {"formula": "N₂⁺", "mass": 28.0134, "intensity": 0.78, "color": "#4ECDC4"},
                {"formula": "O₂⁺", "mass": 31.9988, "intensity": 0.21, "color": "#FF6B6B"},
                {"formula": "Ar⁺", "mass": 39.948, "intensity": 0.01, "color": "#FFD166"},
                {"formula": "CO₂⁺", "mass": 44.0095, "intensity": 0.004, "color": "#06D6A0"}
            ],
            "Углекислый газ": [
                {"formula": "CO₂⁺", "mass": 44.0095, "intensity": 0.95, "color": "#06D6A0"},
                {"formula": "N₂⁺", "mass": 28.0134, "intensity": 0.03, "color": "#4ECDC4"},
                {"formula": "O₂⁺", "mass": 31.9988, "intensity": 0.02, "color": "#FF6B6B"}
            ],
            "Выхлопные газы": [
                {"formula": "CO⁺", "mass": 28.0101, "intensity": 0.45, "color": "#118AB2"},
                {"formula": "CO₂⁺", "mass": 44.0095, "intensity": 0.25, "color": "#06D6A0"},
                {"formula": "NO⁺", "mass": 30.0061, "intensity": 0.15, "color": "#EF476F"},
                {"formula": "NO₂⁺", "mass": 46.0055, "intensity": 0.10, "color": "#7209B7"},
                {"formula": "SO₂⁺", "mass": 63.9619, "intensity": 0.05, "color": "#F15BB5"}
            ],
            "Водород+Гелий": [
                {"formula": "H₂⁺", "mass": 2.016, "intensity": 0.70, "color": "#FF6B6B"},
                {"formula": "He⁺", "mass": 4.0026, "intensity": 0.30, "color": "#4ECDC4"}
            ],
            "Метан+Этан": [
                {"formula": "CH₄⁺", "mass": 16.0313, "intensity": 0.60, "color": "#00BBF9"},
                {"formula": "C₂H₆⁺", "mass": 30.0469, "intensity": 0.30, "color": "#FFD166"},
                {"formula": "C₂H₄⁺", "mass": 28.0313, "intensity": 0.10, "color": "#EF476F"}
            ]
        }

        # Текущая смесь
        self.current_mixture = None
        self.current_mixture_name = ""

    def _generate_unknown(self):
        """Генерирует новый неизвестный элемент"""
        available_elements = [e for e in ELEMENTS.keys()
                              if e not in self.comparison_elements[:4]]
        if not available_elements:
            available_elements = list(ELEMENTS.keys())

        self.current_unknown = random.choice(available_elements)
        self.sample = UnknownSample(mixture=False, specific_element=self.current_unknown)
        return self.current_unknown

    def set_mode(self, mode):
        """Установить режим работы: 'single' или 'mixture'"""
        self.mode = mode
        if mode == "mixture" and not self.current_mixture:
            self.set_mixture("Воздух")  # По умолчанию воздух

    def get_mode(self):
        """Получить текущий режим"""
        return self.mode

    def set_mixture(self, mixture_name):
        """Установить конкретную смесь"""
        if mixture_name in self.gas_mixtures:
            self.current_mixture = self.gas_mixtures[mixture_name]
            self.current_mixture_name = mixture_name
            self.mode = "mixture"

    def get_mixture_list(self):
        """Получить список доступных смесей"""
        return list(self.gas_mixtures.keys())

    def get_current_mixture_info(self):
        """Получить информацию о текущей смеси"""
        if self.mode == "mixture" and self.current_mixture:
            return {
                "name": self.current_mixture_name,
                "components": self.current_mixture,
                "total_peaks": len(self.current_mixture)
            }
        return None

    def set_voltage(self, U):
        """Установить напряжение"""
        self.U = float(U)
        self.tof.U = self.U

    def get_voltage(self):
        """Получить текущее напряжение"""
        return self.U

    def get_unknown_info(self):
        """Получить информацию о текущем неизвестном элементе"""
        if self.current_unknown and self.mode == "single":
            mass = ELEMENTS[self.current_unknown]
            time = self.tof.flight_time(mass)
            return {
                "name": self.current_unknown,
                "mass": mass,
                "time": time,
                "color": "red",
                "style": "--",
                "mode": "single"
            }
        elif self.mode == "mixture" and self.current_mixture:
            return {
                "name": self.current_mixture_name,
                "components": self.current_mixture,
                "mode": "mixture"
            }
        return None

    def check_guess(self, guess):
        """Проверить догадку пользователя"""
        if not self.current_unknown or self.mode != "single":
            return False
        return guess.strip().capitalize() == self.current_unknown

    def spectrum(self):
        """Получить TOF-спектр"""
        if self.mode == "single":
            # Спектр одиночного элемента
            t, s = build_spectrum(
                self.sample.ions(),
                self.tof,
                t_min=self.t_min,
                t_max=self.t_max,
                n=self.n_points,
                sigma=self.det_sigma,
                noise=self.noise
            )
        elif self.mode == "mixture" and self.current_mixture:
            # Спектр смеси
            ions = []
            for component in self.current_mixture:
                ions.append((Ion(component["mass"], charge=1), component["intensity"]))

            t, s = build_spectrum(
                ions,
                self.tof,
                t_min=self.t_min,
                t_max=self.t_max,
                n=self.n_points,
                sigma=self.det_sigma,
                noise=self.noise * 0.5  # Меньше шума для смесей
            )
        else:
            # По умолчанию одиночный элемент
            t, s = build_spectrum(
                self.sample.ions(),
                self.tof,
                t_min=self.t_min,
                t_max=self.t_max,
                n=self.n_points,
                sigma=self.det_sigma,
                noise=self.noise
            )

        # Сохраняем данные для подсказок
        self.current_spectrum_data = (t, s)
        return t, s

    def get_comparison_spectra(self):
        """Получить спектры элементов для сравнения"""
        spectra = []

        for element in self.comparison_elements[:4]:
            mass = ELEMENTS[element]
            ion = Ion(mass, charge=1)

            t, s = build_spectrum(
                [(ion, 1.0)],
                self.tof,
                t_min=self.t_min,
                t_max=self.t_max,
                n=self.n_points,
                sigma=self.det_sigma,
                noise=0.0
            )

            colors = {
                "Na": "#FF6B6B",
                "K": "#4ECDC4",
                "Mg": "#FFD166",
                "Al": "#06D6A0",
                "Cu": "#118AB2",
                "C": "#EF476F",
                "N": "#073B4C",
                "O": "#7209B7"
            }

            spectra.append({
                "name": element,
                "mass": mass,
                "time": self.tof.flight_time(mass),
                "data": (t, s),
                "color": colors.get(element, "gray"),
                "style": "-",
                "alpha": 0.6
            })

        return spectra

    def get_calibrants_data(self):
        """Получить данные калибровочных элементов для траекторий"""
        calibrants_data = []
        for element in self.calibrants:
            mass = ELEMENTS[element]
            v = self.tof.velocity(mass)
            t = self.tof.flight_time(mass)
            calibrants_data.append({
                "name": element,
                "mass": mass,
                "velocity": v,
                "time": t,
                "color": None,
                "style": "-"
            })
        return calibrants_data

    def get_unknown_trajectory(self):
        """Получить траекторию неизвестного элемента или смеси"""
        if self.mode == "single" and self.current_unknown:
            mass = ELEMENTS[self.current_unknown]
            v = self.tof.velocity(mass)
            t = self.tof.flight_time(mass)

            return [{
                "name": "Unknown",
                "formula": self.current_unknown,
                "mass": mass,
                "velocity": v,
                "time": t,
                "color": "#FF5252",
                "style": "--",
                "radius": 1.0  # Радиус для визуализации
            }]

        elif self.mode == "mixture" and self.current_mixture:
            trajectories = []
            for component in self.current_mixture:
                v = self.tof.velocity(component["mass"])
                t = self.tof.flight_time(component["mass"])
                trajectories.append({
                    "name": component["formula"],
                    "formula": component["formula"],
                    "mass": component["mass"],
                    "velocity": v,
                    "time": t,
                    "color": component["color"],
                    "style": "-",
                    "radius": component["intensity"] * 1.5,  # Радиус пропорционален интенсивности
                    "intensity": component["intensity"]
                })
            return trajectories

        return []

    def generate_new_unknown(self):
        """Сгенерировать новый неизвестный элемент"""
        if self.mode == "single":
            return self._generate_unknown()
        return None

    def get_mass_from_time(self, time, charge=1):
        """Определить массу по времени пролета"""
        return self.tof.mass_from_time(time, charge)

    def get_all_trajectories(self):
        """Получить все траектории (калибровочные + неизвестный/смесь)"""
        trajectories = self.get_calibrants_data()
        unknown = self.get_unknown_trajectory()
        if unknown:
            trajectories.extend(unknown)
        return trajectories

    def get_element_at_point(self, x, t, threshold=0.02):
        """Определить, какой элемент находится рядом с точкой (x, t)"""
        best_element = None
        best_distance = float('inf')

        # Проверяем калибровочные элементы
        for element in self.calibrants:
            mass = ELEMENTS[element]
            v = self.tof.velocity(mass)
            tf = self.tof.flight_time(mass)

            if t <= tf:
                x_calc = v * t
                distance = abs(x - x_calc)
                if distance < best_distance and distance < threshold:
                    best_distance = distance
                    best_element = element

        # Проверяем неизвестный элемент или смесь
        unknown_traj = self.get_unknown_trajectory()
        for traj in unknown_traj:
            if t <= traj["time"]:
                x_calc = traj["velocity"] * t
                distance = abs(x - x_calc)
                if distance < best_distance and distance < threshold:
                    best_distance = distance
                    best_element = traj["name"]

        return best_element

    def get_peak_at_time(self, time, threshold=1e-7):
        """Определить, какой элемент имеет пик в заданное время"""
        elements_to_check = []

        if self.mode == "single" and self.current_unknown:
            elements_to_check.append({
                "name": "Unknown",
                "formula": self.current_unknown,
                "mass": ELEMENTS[self.current_unknown],
                "time": self.tof.flight_time(ELEMENTS[self.current_unknown])
            })
        elif self.mode == "mixture" and self.current_mixture:
            for component in self.current_mixture:
                elements_to_check.append({
                    "name": component["formula"],
                    "formula": component["formula"],
                    "mass": component["mass"],
                    "time": self.tof.flight_time(component["mass"]),
                    "intensity": component["intensity"]
                })

        # Добавляем элементы для сравнения
        for element in self.comparison_elements[:4]:
            elements_to_check.append({
                "name": element,
                "formula": element,
                "mass": ELEMENTS[element],
                "time": self.tof.flight_time(ELEMENTS[element])
            })

        # Ищем ближайший элемент
        best_element = None
        best_difference = float('inf')

        for element in elements_to_check:
            difference = abs(element["time"] - time)
            if difference < best_difference and difference < threshold:
                best_difference = difference
                best_element = element

        return best_element

    def get_spectrum_peaks(self):
        """Получить информацию о пиках в текущем спектре"""
        if not self.current_spectrum_data:
            return []

        t, s = self.current_spectrum_data
        peaks = []

        # Находим локальные максимумы
        for i in range(1, len(s) - 1):
            if s[i] > s[i - 1] and s[i] > s[i + 1] and s[i] > 0.1 * np.max(s):
                peaks.append({
                    "time": t[i],
                    "intensity": s[i],
                    "index": i
                })

        return peaks