"""Презентер панели прибора — общий для всех заданий и песочницы."""
import numpy as np

from masslab.model.physics import (DRIFT_LENGTH, LENGTH_MAX, LENGTH_MIN, LENGTH_STEP,
                                   VOLTAGE_DEFAULT, VOLTAGE_MAX, VOLTAGE_MIN, VOLTAGE_STEP,
                                   TOFPhysics)
from masslab.model.spectrum import (build_spectrum, find_peaks, residual_gas_peaks,
                                    time_window)
from masslab.presenters.formatting import num
from masslab.views.viewmodels import Curve, Marker, PlotData, SchemeData, SchemeIon

PALETTE = ("#4FC3F7", "#81C784", "#FFB74D", "#BA68C8", "#4DB6AC",
           "#F06292", "#AED581", "#90A4AE", "#FFF176", "#7986CB")
UNKNOWN_COLOR = "#FF5252"
ISOTOPE_COLOR = "#B0BEC5"
GAS_COLOR = "#8D8D8D"
DETECTOR_COLOR = "#757575"
LOG_FLOOR = 1e-4
HINT = "Наведите курсор на пик спектра, чтобы измерить время пролёта"
EMPTY = "Запустите ионы, чтобы получить спектр"


class WorkspacePresenter:
    def __init__(self, view, np_rng, noise=0.02, peak_threshold=0.03, show_mass=False,
                 adjustable_length=False):
        self._view = view
        self._rng = np_rng
        self._noise = noise
        self._threshold = peak_threshold
        self._show_mass = show_mass
        self._adjustable_length = adjustable_length
        self._voltage = VOLTAGE_DEFAULT
        self._length = DRIFT_LENGTH
        self._log = False
        self._gas = True
        self._peaks = []
        self._found = []
        self._t_max = 0.0
        self.last_spectrum = None       # PlotData последнего спектра (для отчёта)
        view.voltage_changed.connect(self._on_voltage_changed)
        view.length_changed.connect(self._on_length_changed)
        view.log_scale_toggled.connect(self._on_log_toggled)
        view.spectrum_hovered.connect(self._on_hover)
        view.set_voltage_range(VOLTAGE_MIN, VOLTAGE_MAX, VOLTAGE_STEP)
        view.set_length_range(LENGTH_MIN, LENGTH_MAX, LENGTH_STEP)
        view.set_length_visible(adjustable_length)

    @property
    def tof(self):
        return TOFPhysics(self._voltage, self._length)

    @property
    def voltage(self):
        return self._voltage

    @property
    def length(self):
        return self._length

    def reset(self):
        self._voltage = VOLTAGE_DEFAULT
        self._length = DRIFT_LENGTH
        self._log = False
        self._peaks = []
        self._found = []
        self.last_spectrum = None
        self._view.set_voltage(self._voltage)
        self._view.set_length(self._length)
        self._view.set_log_scale(False)
        self._update_parameters()
        self._view.clear_plots(EMPTY)
        self._view.hide_cursor()
        self._view.set_readout(HINT)

    def set_peaks(self, peaks):
        self._peaks = list(peaks)
        self._render(animate=True)

    def set_residual_gas(self, enabled):
        self._gas = bool(enabled)
        self._render(animate=False)

    def found_peaks(self):
        """[(m/z, высота над фоном)] найденных пиков, кроме остаточного газа."""
        tof = self.tof
        gas = [p.mass for p in residual_gas_peaks()] if self._gas else []
        result = []
        for p in self._found:
            mz = tof.mass_from_time(p.time)
            if all(abs(mz - g) > 0.5 for g in gas):
                result.append((mz, p.height))
        return result

    # --- события вида -------------------------------------------------

    def _on_voltage_changed(self, value):
        self._voltage = int(value)
        self._view.set_voltage(self._voltage)
        self._update_parameters()
        self._render(animate=True)

    def _on_length_changed(self, value):
        self._length = round(float(value), 2)
        self._view.set_length(self._length)
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
        nearest = min(self._found, key=lambda p: abs(p.time * 1e6 - x_us))
        if abs(nearest.time * 1e6 - x_us) <= snap:
            t = nearest.time
            text = f"Пик: t = {num(t * 1e6, 3)} мкс"
            if self._show_mass:
                text += f"   m/z ≈ {num(self.tof.mass_from_time(t), 1)}"
            text += f"   R ≈ {t / (2 * nearest.fwhm):.0f}"
            height = nearest.height + self._background_level()
            self._view.show_cursor(t * 1e6, self._y(height), text)
        else:
            text = f"t = {num(x_us, 3)} мкс"
            self._view.show_cursor(x_us, None, text)
        self._view.set_readout(text)

    # --- построение графиков -------------------------------------------

    def _update_parameters(self):
        self._view.set_parameters(
            f"U = {self._voltage} В     L = {num(self._length, 2)} м")

    def _background_level(self):
        return self._noise

    def _y(self, value):
        return max(value, LOG_FLOOR) if self._log else value

    def _all_peaks(self):
        peaks = list(self._peaks)
        if self._gas and peaks:
            scale = max(p.intensity for p in peaks)
            peaks += residual_gas_peaks(scale)
        return peaks

    def _render(self, animate):
        if not self._peaks:
            return
        tof = self.tof
        peaks = self._all_peaks()
        self._t_max = time_window(tof, peaks)
        t, s = build_spectrum(peaks, tof, self._t_max, self._rng, noise=self._noise)
        self._found = find_peaks(t, s, tof.sigma_at_time, rel_threshold=self._threshold)
        colors = self._colors()
        self.last_spectrum = self._spectrum_plot(t, s, tof, peaks, colors)
        self._view.show_spectrum(self.last_spectrum)
        self._view.show_trajectories(self._trajectory_plot(tof, colors), animate)
        self._view.show_scheme(self._scheme(tof, colors))
        self._view.hide_cursor()
        self._view.set_readout(HINT)

    def _colors(self):
        colors, i = {}, 0
        for p in self._peaks:
            if p.label and p.label != "?" and p.label not in colors:
                colors[p.label] = PALETTE[i % len(PALETTE)]
                i += 1
        return colors

    def _spectrum_plot(self, t, s, tof, peaks, colors):
        x_max = self._t_max * 1e6
        y = np.maximum(s, LOG_FLOOR) if self._log else s
        markers = []
        for p in peaks:
            x = tof.flight_time(p.mass, p.charge) * 1e6
            if p.background:
                markers.append(Marker(x, p.label, GAS_COLOR))
            elif p.label == "?":
                markers.append(Marker(x, "?", UNKNOWN_COLOR))   # время студент измеряет сам
            elif p.label in colors:
                markers.append(Marker(x, p.label, colors[p.label]))
        curve = Curve(tuple(t * 1e6), tuple(y), "#E0E0E0", width=1.0, animate=False)
        y_lim = (LOG_FLOOR, 2.0) if self._log else (0.0, 1.15)
        return PlotData("Масс-спектр (время пролёта)", "Время пролёта t, мкс",
                        "Интенсивность, отн. ед.", (curve,), tuple(markers),
                        (0.0, x_max), y_lim, self._log)

    def _visible_ions(self):
        top = max(p.intensity for p in self._peaks)
        return [p for p in sorted(self._peaks, key=lambda p: p.mz)
                if p.intensity >= 0.02 * top]

    def _ion_style(self, p, colors, top):
        if p.label == "?":
            return UNKNOWN_COLOR, "--", 2.0, 1.0
        if p.label in colors:
            return colors[p.label], "-", 1.5, 1.0
        return ISOTOPE_COLOR, "-", 1.2, 0.35 + 0.65 * p.intensity / top

    def _trajectory_plot(self, tof, colors):
        x_max = self._t_max * 1e6
        L = tof.length
        top = max(p.intensity for p in self._peaks)
        curves = [Curve((0.0, x_max), (L, L), DETECTOR_COLOR, "Детектор", "--", 1.0,
                        animate=False)]
        seen = set()
        for p in self._visible_ions():
            t_us = tof.flight_time(p.mass, p.charge) * 1e6
            color, style, width, alpha = self._ion_style(p, colors, top)
            label = {"?": "Неизвестный"}.get(p.label, p.label)
            if label in seen:
                label = ""
            seen.add(label)
            curves.append(Curve((0.0, t_us), (0.0, L), color, label, style, width, alpha))
        return PlotData("Траектории ионов в дрейфовой трубке", "Время t, мкс",
                        "Координата x, м", tuple(curves), (), (0.0, x_max), (0.0, L * 1.1))

    def _scheme(self, tof, colors):
        top = max(p.intensity for p in self._peaks)
        ions = []
        for p in self._visible_ions():
            color = self._ion_style(p, colors, top)[0]
            ions.append(SchemeIon(p.label, color, tof.flight_time(p.mass, p.charge) * 1e6))
        return SchemeData(tuple(ions), self._voltage, self._length)
