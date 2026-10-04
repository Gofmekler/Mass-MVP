"""Презентер панели прибора — общий для всех трёх заданий."""
import numpy as np

from masslab.model.physics import (VOLTAGE_DEFAULT, VOLTAGE_MAX, VOLTAGE_MIN,
                                   VOLTAGE_STEP, TOFPhysics)
from masslab.model.spectrum import build_spectrum, find_peaks, time_window
from masslab.presenters.formatting import num
from masslab.views.viewmodels import Curve, Marker, PlotData

PALETTE = ("#4FC3F7", "#81C784", "#FFB74D", "#BA68C8", "#4DB6AC",
           "#F06292", "#AED581", "#90A4AE", "#FFF176", "#7986CB")
UNKNOWN_COLOR = "#FF5252"
ISOTOPE_COLOR = "#B0BEC5"
DETECTOR_COLOR = "#757575"
LOG_FLOOR = 1e-4
HINT = "Наведите курсор на пик спектра, чтобы измерить время пролёта"
EMPTY = "Запустите ионы, чтобы получить спектр"


class WorkspacePresenter:
    def __init__(self, view, np_rng, noise=0.02, peak_threshold=0.03, show_mass=False):
        self._view = view
        self._rng = np_rng
        self._noise = noise
        self._threshold = peak_threshold
        self._show_mass = show_mass
        self._voltage = VOLTAGE_DEFAULT
        self._log = False
        self._peaks = []
        self._found = []            # [(t, высота)] найденные пики, с
        view.voltage_changed.connect(self._on_voltage_changed)
        view.log_scale_toggled.connect(self._on_log_toggled)
        view.spectrum_hovered.connect(self._on_hover)
        view.set_voltage_range(VOLTAGE_MIN, VOLTAGE_MAX, VOLTAGE_STEP)

    @property
    def tof(self):
        return TOFPhysics(self._voltage)

    @property
    def voltage(self):
        return self._voltage

    def reset(self):
        self._voltage = VOLTAGE_DEFAULT
        self._log = False
        self._peaks = []
        self._found = []
        self._view.set_voltage(self._voltage)
        self._view.set_log_scale(False)
        self._update_parameters()
        self._view.clear_plots(EMPTY)
        self._view.hide_cursor()
        self._view.set_readout(HINT)

    def set_peaks(self, peaks):
        self._peaks = list(peaks)
        self._render(animate=True)

    # --- события вида -------------------------------------------------

    def _on_voltage_changed(self, value):
        self._voltage = int(value)
        self._view.set_voltage(self._voltage)
        self._update_parameters()
        self._render(animate=True)

    def _on_log_toggled(self, enabled):
        self._log = bool(enabled)
        self._render(animate=False)

    def _on_hover(self, x_us):
        if x_us is None or not self._found:
            self._view.hide_cursor()
            self._view.set_readout(HINT)
            return
        snap = 0.015 * self._t_max * 1e6
        nearest = min(self._found, key=lambda p: abs(p[0] * 1e6 - x_us))
        if abs(nearest[0] * 1e6 - x_us) <= snap:
            t, height = nearest
            text = f"Пик: t = {num(t * 1e6, 3)} мкс"
            if self._show_mass:
                text += f"  →  m/z ≈ {num(self.tof.mass_from_time(t), 1)}"
            self._view.show_cursor(t * 1e6, self._y(height), text)
        else:
            text = f"t = {num(x_us, 3)} мкс"
            self._view.show_cursor(x_us, None, text)
        self._view.set_readout(text)

    # --- построение графиков -------------------------------------------

    def _update_parameters(self):
        tof = self.tof
        self._view.set_parameters(
            f"U = {self._voltage} В     L = {num(tof.length, 2)} м     z = +1")

    def _y(self, value):
        return max(value, LOG_FLOOR) if self._log else value

    def _render(self, animate):
        if not self._peaks:
            return
        tof = self.tof
        self._t_max = time_window(tof, self._peaks)
        t, s = build_spectrum(self._peaks, tof, self._t_max, self._rng, noise=self._noise)
        self._found = find_peaks(t, s, rel_threshold=self._threshold)
        self._view.show_spectrum(self._spectrum_plot(t, s, tof))
        self._view.show_trajectories(self._trajectory_plot(tof), animate)
        self._view.hide_cursor()
        self._view.set_readout(HINT)

    def _colors(self):
        colors, i = {}, 0
        for p in self._peaks:
            if p.label and p.label != "?" and p.label not in colors:
                colors[p.label] = PALETTE[i % len(PALETTE)]
                i += 1
        return colors

    def _spectrum_plot(self, t, s, tof):
        x_max = self._t_max * 1e6
        y = np.maximum(s, LOG_FLOOR) if self._log else s
        colors = self._colors()
        markers = tuple(Marker(tof.flight_time(p.mass) * 1e6, p.label, colors[p.label])
                        for p in self._peaks if p.label in colors)
        curve = Curve(tuple(t * 1e6), tuple(y), "#E0E0E0", width=1.0, animate=False)
        y_lim = (LOG_FLOOR, 2.0) if self._log else (0.0, 1.15)
        return PlotData("Масс-спектр (время пролёта)", "Время пролёта t, мкс",
                        "Интенсивность, отн. ед.", (curve,), markers,
                        (0.0, x_max), y_lim, self._log)

    def _trajectory_plot(self, tof):
        x_max = self._t_max * 1e6
        L = tof.length
        colors = self._colors()
        top = max(p.intensity for p in self._peaks)
        curves = [Curve((0.0, x_max), (L, L), DETECTOR_COLOR, "Детектор", "--", 1.0,
                        animate=False)]
        seen = set()
        for p in sorted(self._peaks, key=lambda p: p.mass):
            if p.intensity < 0.02 * top:
                continue
            t_us = tof.flight_time(p.mass) * 1e6
            if p.label == "?":
                curves.append(Curve((0.0, t_us), (0.0, L), UNKNOWN_COLOR, "Неизвестный",
                                    "--", 2.0))
            elif p.label in colors:
                label = p.label if p.label not in seen else ""
                seen.add(p.label)
                curves.append(Curve((0.0, t_us), (0.0, L), colors[p.label], label))
            else:
                alpha = 0.35 + 0.65 * p.intensity / top
                curves.append(Curve((0.0, t_us), (0.0, L), ISOTOPE_COLOR, "", "-", 1.2, alpha))
        return PlotData("Траектории ионов в дрейфовой трубке", "Время t, мкс",
                        "Координата x, м", tuple(curves), (), (0.0, x_max), (0.0, L * 1.1))
