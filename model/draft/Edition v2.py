"""
masslab_unified.py
Универсальный учебный симулятор масс-спектрометра:
- Режимы: Magnetic sector (радиус) и TOF (time-of-flight)
- Вкладки: "Спектр" и "Траектории"
- Тёмная тема, переключение масштаба (реальный / нормированный)
- 3 калибрационных элемента: H, He, Ar
- Режим "Неизвестный элемент" с проверкой ответа
Автор: подсобный учебный скрипт для курсовой
Зависимости: numpy, scipy, matplotlib, PySide6 (или PyQt5)
"""

import sys, os, math, random, json
import sys
import os
import json
import csv
import math
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("QtAgg")

from PySide6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ---- Тёмная тема matplotlib ----
plt.style.use('dark_background')

# ---- Эталонная база (masses in atomic mass units - amu) ----
ELEMENTS = {
    "H": 1.00784,
    "He": 4.002602,
    "Ar": 39.948,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.999,
    "Ne": 20.1797,
    "Na": 22.9897,
    "Mg": 24.305,
    "Fe": 55.845,
    "Ag": 107.8682,
    "Au": 196.9666
}

# Default calibrants order (we will use H, He, Ar)
CALIBRANTS = ["H", "He", "Ar"]

# Constants
ELEM_AMU = 1.66053906660e-27  # kg
E_CHARGE = 1.602176634e-19  # C

# ---- Физические модели ----

def tof_time_for_mz(mz, a=0.0015, t0=0.0):
    """TOF model: t = a * sqrt(m/z) + t0  (t in seconds)"""
    return a * np.sqrt(np.array(mz)) + t0

def tof_mz_from_time(t, a, t0):
    return ((t - t0) / a) ** 2

def magnetic_radius(m, U, B, q=E_CHARGE):
    """Compute radius r (m) for mass m (kg), accelerating voltage U (V), magnetic field B (T)."""
    # v = sqrt(2 q U / m), r = m v / (q B) -> r = (1/B) * sqrt(2 m U / q)
    return (1.0 / B) * np.sqrt((2.0 * m * U) / q)

# ---- Сигнал / спектр генерация ----

def generate_tof_spectrum(mz_list, intens_list, a=0.0015, t0=0.0002,
                          t_min=0.0, t_max=0.02, n_points=3000,
                          detector_sigma=6e-5, noise_level=0.02, background=0.01, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(t_min, t_max, n_points)
    sig = np.zeros_like(t)
    for mz, inten in zip(mz_list, intens_list):
        mu = tof_time_for_mz(mz, a=a, t0=t0)
        sig += inten * np.exp(-0.5 * ((t - mu) / detector_sigma) ** 2)
    max_amp = np.max(sig) if np.max(sig) > 0 else 1.0
    sig += noise_level * max_amp * np.random.normal(0, 1, size=t.shape)
    sig += background * max_amp * np.sin(2 * np.pi * t / (t_max + 1e-9))
    sig[sig < 0] = 0.0
    return t, sig

def generate_magnetic_spectrum(mz_list, intens_list, U, B,
                               r_min=0.0, r_max=0.5, n_points=3000,
                               detector_sigma=1e-3, noise_level=0.02, background=0.01, seed=None):
    """
    Generate intensity vs radius (meters) for magnetic sector.
    mz_list are m/z values (amu), intens_list relative amplitudes.
    """
    if seed is not None:
        np.random.seed(seed)
    r = np.linspace(r_min, r_max, n_points)
    sig = np.zeros_like(r)
    for mz, inten in zip(mz_list, intens_list):
        m_kg = mz * ELEM_AMU
        mu = magnetic_radius(m_kg, U, B)
        sig += inten * np.exp(-0.5 * ((r - mu) / detector_sigma) ** 2)
    max_amp = np.max(sig) if np.max(sig) > 0 else 1.0
    sig += noise_level * max_amp * np.random.normal(0, 1, size=r.shape)
    sig += background * max_amp * np.sin(2 * np.pi * r / (r_max + 1e-9))
    sig[sig < 0] = 0.0
    return r, sig

# ---- Analysis helpers ----

def detect_peaks_simple(x, y, prominence_rel=0.05):
    prom = prominence_rel * np.max(y)
    peaks, props = find_peaks(y, prominence=prom)
    return peaks, props

def calibrate_tof_from_pairs(t_peaks, known_mz):
    # linear fit: t = a * sqrt(mz) + t0
    x = np.sqrt(np.array(known_mz))
    y = np.array(t_peaks)
    def lin(x, a, t0): return a * x + t0
    popt, pcov = curve_fit(lin, x, y)
    return float(popt[0]), float(popt[1]), pcov

def calibrate_mag_from_pairs(r_peaks, known_m_over_z, U, q=E_CHARGE):
    # From formula r = (1/B) * sqrt(2 m U / q) => r*B = sqrt(2 U / q) * sqrt(m)
    # If B and U known, we can compute effective m from r -> but calibration often fits r^2 ~ m
    # Fit: r^2 = k * (m/z)
    x = np.array(known_m_over_z)
    y = np.array(r_peaks) ** 2
    def lin(x, k): return k * x
    popt, pcov = curve_fit(lin, x, y)
    return float(popt[0]), pcov

# ---- GUI ----

class MplPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, width=6, height=4, dpi=110):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

class MassLabApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab — Универсальный симулятор масс-спектрометра")
        self.resize(1200, 760)
        # State
        self.mode = "Magnetic"  # or "TOF"
        self.scale_mode = "Real"  # "Real" or "Normalized"
        self.U = 3000.0  # V
        self.B = 0.3    # T
        self.tof_a = 0.0015
        self.tof_t0 = 0.0002
        self.det_sigma_mag = 0.005  # m
        self.det_sigma_tof = 6e-5
        self.noise = 0.02
        self.tmin, self.tmax = 0.0, 0.02
        self.npoints = 3000
        self.current_unknown = None  # tuple (name, mass_amu)
        self.current_unknown_radius = None
        self.current_unknown_time = None
        # UI
        self._init_ui()
        # initial plot
        self.sample_mz = [28.0, 32.0, 44.0]  # default sample list (amu)
        self.sample_int = [1.0, 0.6, 0.8]
        self.update_plots()

    def _init_ui(self):
        # central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # left panel: controls
        left = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left, stretch=0)

        # Mode selector
        mode_box = QtWidgets.QGroupBox("Режим анализа")
        mbx = QtWidgets.QVBoxLayout(mode_box)
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Magnetic", "TOF"])
        self.combo_mode.setCurrentText(self.mode)
        mbx.addWidget(QtWidgets.QLabel("Выбери режим:"))
        mbx.addWidget(self.combo_mode)
        left.addWidget(mode_box)

        # Scale selector
        scale_box = QtWidgets.QGroupBox("Масштаб траекторий")
        sbx = QtWidgets.QHBoxLayout(scale_box)
        self.combo_scale = QtWidgets.QComboBox()
        self.combo_scale.addItems(["Real", "Normalized"])
        self.combo_scale.setCurrentText(self.scale_mode)
        sbx.addWidget(QtWidgets.QLabel("Масштаб:"))
        sbx.addWidget(self.combo_scale)
        left.addWidget(scale_box)

        # Calibrants selection
        cal_box = QtWidgets.QGroupBox("Калибровочные элементы")
        cal_layout = QtWidgets.QVBoxLayout(cal_box)
        self.check_cal = {}
        for name in CALIBRANTS:
            cb = QtWidgets.QCheckBox(f"{name} (m={ELEMENTS.get(name):.4g} u)")
            cb.setChecked(True)
            cal_layout.addWidget(cb)
            self.check_cal[name] = cb
        left.addWidget(cal_box)

        # Sliders for U and B
        pb = QtWidgets.QGroupBox("Параметры прибора")
        pbl = QtWidgets.QFormLayout(pb)
        self.slider_U = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_U.setMinimum(100)  # 100 V
        self.slider_U.setMaximum(20000)  # 20 kV
        self.slider_U.setValue(int(self.U))
        self.lbl_U = QtWidgets.QLabel(f"U = {self.U:.0f} V")
        pbl.addRow(self.lbl_U, self.slider_U)

        self.slider_B = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_B.setMinimum(1)   # 0.01 T represented as 1
        self.slider_B.setMaximum(1000)  # 1.0 T as 1000
        self.slider_B.setValue(int(self.B * 1000))
        self.lbl_B = QtWidgets.QLabel(f"B = {self.B:.3f} T")
        pbl.addRow(self.lbl_B, self.slider_B)

        left.addWidget(pb)

        # Unknown element controls
        unk_box = QtWidgets.QGroupBox("Режим: Неизвестный элемент")
        unk_layout = QtWidgets.QVBoxLayout(unk_box)
        self.btn_new_unknown = QtWidgets.QPushButton("Сгенерировать неизвестный элемент")
        self.lbl_measured = QtWidgets.QLabel("—")
        unk_layout.addWidget(self.btn_new_unknown)
        unk_layout.addWidget(QtWidgets.QLabel("Измеренное:"))
        unk_layout.addWidget(self.lbl_measured)
        # choose answer
        self.combo_guess = QtWidgets.QComboBox()
        self.combo_guess.addItems(sorted(ELEMENTS.keys()))
        unk_layout.addWidget(self.combo_guess)
        self.btn_check = QtWidgets.QPushButton("Проверить ответ")
        unk_layout.addWidget(self.btn_check)
        self.lbl_feedback = QtWidgets.QLabel("")
        unk_layout.addWidget(self.lbl_feedback)
        left.addWidget(unk_box)

        # Buttons: generate / detect / export
        ops = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Обновить спектр")
        self.btn_detect = QtWidgets.QPushButton("Найти пики")
        ops.addWidget(self.btn_generate)
        ops.addWidget(self.btn_detect)
        left.addLayout(ops)

        # Spacer and credits
        left.addStretch()
        left.addWidget(QtWidgets.QLabel("MassLab — учебный симулятор"))

        # Right: tabs for Spectrum and Trajectories
        right = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right, stretch=1)

        self.tabs = QtWidgets.QTabWidget()
        right.addWidget(self.tabs)

        # Spectrum tab
        self.tab_spectrum = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_spectrum, "Спектр")
        sp_layout = QtWidgets.QVBoxLayout(self.tab_spectrum)
        self.panel_spec = MplPanel(self, width=7, height=4)
        sp_layout.addWidget(self.panel_spec)
        # spectrum info area
        self.spec_info = QtWidgets.QLabel("")
        sp_layout.addWidget(self.spec_info)

        # Trajectories tab
        self.tab_traj = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_traj, "Траектории")
        tr_layout = QtWidgets.QVBoxLayout(self.tab_traj)
        self.panel_traj = MplPanel(self, width=7, height=4)
        tr_layout.addWidget(self.panel_traj)
        # traj controls (checkboxes for elements and scale already in left)
        self.traj_info = QtWidgets.QLabel("")
        tr_layout.addWidget(self.traj_info)

        # Connections
        self.combo_mode.currentTextChanged.connect(self.on_mode_changed)
        self.combo_scale.currentTextChanged.connect(self.on_scale_changed)
        self.slider_U.valueChanged.connect(self.on_U_changed)
        self.slider_B.valueChanged.connect(self.on_B_changed)
        self.btn_generate.clicked.connect(self.update_plots)
        self.btn_detect.clicked.connect(self.on_detect_peaks)
        self.btn_new_unknown.clicked.connect(self.on_new_unknown)
        self.btn_check.clicked.connect(self.on_check_guess)

        # initial UI colors (dark theme adjustments)
        self.set_dark_palette()

    def set_dark_palette(self):
        # Set dark palette for Qt widgets
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#121212"))
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e6e6e6"))
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1e1e1e"))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#121212"))
        pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#e6e6e6"))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#e6e6e6"))
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#1f1f1f"))
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#e6e6e6"))
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#3d8bd9"))
        pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
        self.setPalette(pal)

    # ---- UI callbacks ----
    def on_mode_changed(self, txt):
        self.mode = txt
        # update labels and plots
        self.update_plots()

    def on_scale_changed(self, txt):
        self.scale_mode = txt
        self.update_plots()

    def on_U_changed(self, val):
        self.U = float(val)
        self.lbl_U.setText(f"U = {self.U:.0f} V")
        self.update_plots()

    def on_B_changed(self, val):
        self.B = float(val) / 1000.0
        self.lbl_B.setText(f"B = {self.B:.3f} T")
        self.update_plots()

    # ---- Unknown generation / checking ----
    def on_new_unknown(self):
        # pick random element among a subset (to avoid too large masses)
        choices = list(ELEMENTS.keys())
        name = random.choice(choices)
        mass = ELEMENTS[name]
        self.current_unknown = (name, mass)
        # simulate measurement depending on mode
        if self.mode == "Magnetic":
            m_kg = mass * ELEM_AMU
            r_true = magnetic_radius(m_kg, self.U, self.B)
            # add noise
            measured = r_true * (1.0 + np.random.normal(0, 0.02))
            # choose scale display
            if self.scale_mode == "Real":
                display = f"r = {measured*100:.3f} cm (± noise)"
            else:
                # normalized to Ar maybe
                display = f"r_norm = {measured / (magnetic_radius(ELEMENTS['Ar']*ELEM_AMU, self.U, self.B)):.3f} (norm to Ar)"
            self.current_unknown_radius = measured
            self.lbl_measured.setText(display)
        else:
            # TOF mode: simulate time
            mz = mass  # treat as m/z in amu (z=1)
            t_true = tof_time_for_mz(mz, a=self.tof_a, t0=self.tof_t0)
            measured = t_true * (1.0 + np.random.normal(0, 0.02))
            # display in ms
            self.current_unknown_time = measured
            self.lbl_measured.setText(f"t = {measured*1000:.4f} ms (± noise)")
        self.lbl_feedback.setText("")
        # switch to spectrum tab so student sees peak
        self.tabs.setCurrentIndex(0)
        # Update plots to reflect new unknown (we can add the unknown to sample list or show separately)
        self.update_plots()

    def on_check_guess(self):
        if self.current_unknown is None:
            self.lbl_feedback.setText("Сначала сгенерируй неизвестный элемент.")
            return
        guess = self.combo_guess.currentText()
        correct = self.current_unknown[0]
        if guess == correct:
            self.lbl_feedback.setText("✅ Верно!")
        else:
            # give hint based on relative difference
            if self.mode == "Magnetic":
                guessed_m = ELEMENTS[guess] * ELEM_AMU
                guessed_r = magnetic_radius(guessed_m, self.U, self.B)
                diff = abs(guessed_r - self.current_unknown_radius) / (self.current_unknown_radius + 1e-12)
                self.lbl_feedback.setText(f"❌ Неверно. Разница радиусов ≈ {diff*100:.1f}% (true={correct})")
            else:
                guessed_t = tof_time_for_mz(ELEMENTS[guess], a=self.tof_a, t0=self.tof_t0)
                diff = abs(guessed_t - self.current_unknown_time) / (self.current_unknown_time + 1e-12)
                self.lbl_feedback.setText(f"❌ Неверно. Разница времени ≈ {diff*100:.1f}% (true={correct})")

    # ---- Plotting ----
    def update_plots(self):
        # read which calibrants selected
        cal_names = [n for n, cb in self.check_cal.items() if cb.isChecked()]
        # create sample list: include calibrants + some other peaks
        # For demonstration, we build a sample with H, He, Ar and some other common species
        used = []
        intens = []
        # ensure calibrants present
        for n in cal_names:
            used.append(ELEMENTS[n])
            intens.append(1.0)
        # add a few others
        others = ["C", "O", "Ne", "Na"]
        for o in others:
            if o in ELEMENTS and ELEMENTS[o] not in used and len(used) < 7:
                used.append(ELEMENTS[o])
                intens.append(0.6)
        # if there's an unknown currently, add it to the sample so it shows up as a peak
        if self.current_unknown is not None:
            used.append(self.current_unknown[1])
            intens.append(1.0)

        if self.mode == "Magnetic":
            # generate intensity vs radius
            r, sig = generate_magnetic_spectrum(used, intens, self.U, self.B,
                                                r_min=0.0, r_max=0.5, n_points=self.npoints,
                                                detector_sigma=self.det_sigma_mag, noise_level=self.noise)
            ax = self.panel_spec.ax
            ax.clear()
            ax.plot(r*100, sig, label="Intensity vs radius (cm)")  # show cm
            ax.set_xlabel("Radius, cm")
            ax.set_ylabel("Intensity (arb.)")
            ax.set_title("Magnetic sector spectrum")
            # annotate expected peaks for calibrants
            legend_items = []
            for mz_val, inten, name in zip(used, intens, [*(cal_names), *others[:max(0, len(used)-len(cal_names))]]):
                # safe mapping: we used used[] order; name mapping approximated
                pass
            # find and mark peaks
            peaks, props = detect_peaks_simple(r, sig)
            ax.plot(r[peaks]*100, sig[peaks], 'x', label='Peaks')
            # annotate peaks with nearest element name (by radius)
            # compute expected radii for elements in used
            expected = {}
            for mz_val in used:
                mkg = mz_val * ELEM_AMU
                expected[mz_val] = magnetic_radius(mkg, self.U, self.B)
            # for each detected peak, find nearest expected mz
            for idx in peaks:
                rp = r[idx]
                # nearest expected mass
                nearest_mz = min(expected.keys(), key=lambda mzv: abs(expected[mzv] - rp))
                # try find element name for that mz
                name_guess = None
                for nm, mv in ELEMENTS.items():
                    if abs(mv - nearest_mz) < 1e-9:
                        name_guess = nm
                        break
                label = f"{name_guess or nearest_mz:.2f}"
                ax.annotate(label, (rp*100, sig[idx]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
            ax.legend()
            self.panel_spec.canvas.draw()
            self.spec_info.setText(f"Mode: Magnetic | U={self.U:.0f} V, B={self.B:.3f} T")
            # update trajectories
            self.plot_trajectories_mag()
        else:
            # TOF mode: intensity vs time
            mzs = used
            t, sig = generate_tof_spectrum(mzs, intens, a=self.tof_a, t0=self.tof_t0,
                                           t_min=self.tmin, t_max=self.tmax, n_points=self.npoints,
                                           detector_sigma=self.det_sigma_tof, noise_level=self.noise)
            ax = self.panel_spec.ax
            ax.clear()
            ax.plot(t*1000, sig, label="Intensity vs time (ms)")
            ax.set_xlabel("Time, ms")
            ax.set_ylabel("Intensity (arb.)")
            ax.set_title("TOF spectrum")
            peaks, props = detect_peaks_simple(t, sig)
            ax.plot(t[peaks]*1000, sig[peaks], 'x', label='Peaks')
            # annotate peaks with approximate m/z using current calibration a/t0
            for idx in peaks:
                tp = t[idx]
                try:
                    mz_est = tof_mz_from_time(tp, self.tof_a, self.tof_t0)
                    ax.annotate(f"{mz_est:.1f}", (tp*1000, sig[idx]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
                except Exception:
                    pass
            ax.legend()
            self.panel_spec.canvas.draw()
            self.spec_info.setText(f"Mode: TOF | a={self.tof_a:.6f}, t0={self.tof_t0:.6f}")
            # update trajectories (TOF visual)
            self.plot_trajectories_tof()

    def on_detect_peaks(self):
        # basic detection and quick info
        if self.mode == "Magnetic":
            # regenerate current spectrum to get arrays
            used = [ELEMENTS[n] for n, cb in self.check_cal.items() if cb.isChecked()]
            t = None
            r, sig = generate_magnetic_spectrum(used, [1.0]*len(used), self.U, self.B,
                                                r_min=0.0, r_max=0.5, n_points=self.npoints,
                                                detector_sigma=self.det_sigma_mag, noise_level=self.noise)
            peaks, props = detect_peaks_simple(r, sig)
            msg = f"Найдено пиков: {len(peaks)}. Радиусы (cm): " + ", ".join([f"{rr*100:.3f}" for rr in r[peaks]])
            QtWidgets.QMessageBox.information(self, "Поиск пиков", msg)
        else:
            used = [ELEMENTS[n] for n, cb in self.check_cal.items() if cb.isChecked()]
            t, sig = generate_tof_spectrum(used, [1.0]*len(used), a=self.tof_a, t0=self.tof_t0,
                                           t_min=self.tmin, t_max=self.tmax, n_points=self.npoints,
                                           detector_sigma=self.det_sigma_tof, noise_level=self.noise)
            peaks, props = detect_peaks_simple(t, sig)
            msg = f"Найдено пиков: {len(peaks)}. Времена (ms): " + ", ".join([f"{tt*1000:.3f}" for tt in t[peaks]])
            QtWidgets.QMessageBox.information(self, "Поиск пиков", msg)

    # ---- Trajectories plotting ----
    def plot_trajectories_mag(self):
        ax = self.panel_traj.ax
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        # background schematic: source at origin
        ax.set_title("Magnetic sector: trajectories")
        ax.set_facecolor("#121212")
        # draw source
        ax.scatter([0.0], [0.0], c='red', s=40)
        ax.text(0.02, -0.02, "Источник", color='white')
        # draw bend region center at (0,0) and semicircles of radii
        elems = [n for n, cb in self.check_cal.items() if cb.isChecked()]
        colors = ['cyan', 'orange', 'lime']
        maxr = 0.0
        radii = {}
        for i, name in enumerate(elems):
            mkg = ELEMENTS[name] * ELEM_AMU
            r = magnetic_radius(mkg, self.U, self.B)
            radii[name] = r
            maxr = max(maxr, r)
        # if normalized, scale so that largest r = 1 unit
        if self.scale_mode == "Normalized" and maxr > 0:
            scale = 0.3 / maxr  # fit in panel (~0.3 m -> visual)
        else:
            scale = 1.0
        for i, name in enumerate(elems):
            r = radii[name] * scale
            theta = np.linspace(0, math.pi/2, 300)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.plot(x, y, label=f"{name} r={radii[name]*100:.2f} cm", color=colors[i % len(colors)])
        # draw detector line (simple arc endpoint area)
        ax.plot([0.0, 0.6], [0.0, 0.0], color='gray', linestyle='--', linewidth=1)
        ax.text(0.5, 0.0, "Детектор", color='white')
        ax.legend(facecolor='#222', labelcolor='white')
        self.panel_traj.canvas.draw()
        self.traj_info.setText(f"Trajectories (Magnetic) | scale={self.scale_mode}")

    def plot_trajectories_tof(self):
        ax = self.panel_traj.ax
        ax.clear()
        ax.set_title("TOF: схема вылета и времени пролёта")
        ax.set_facecolor("#121212")
        # Draw source at left, drift tube to right, detector at end
        L = 1.0  # normalized tube length
        ax.plot([0, L], [0, 0], color='white', linewidth=2)  # tube center line
        ax.scatter([0], [0], c='red', s=40)
        ax.text(0, -0.05, "Источник", color='white')
        ax.scatter([L], [0], c='yellow', s=60)
        ax.text(L, -0.05, "Детектор", color='white')
        # show markers for arrival times mapped to positions along tube
        elems = [n for n, cb in self.check_cal.items() if cb.isChecked()]
        colors = ['cyan', 'orange', 'lime']
        # compute times and map to positions along tube proportional to time
        max_t = 0.0
        times = {}
        for name in elems:
            mz = ELEMENTS[name]
            t = tof_time_for_mz(mz, a=self.tof_a, t0=self.tof_t0)
            times[name] = t
            max_t = max(max_t, t)
        if self.scale_mode == "Normalized" and max_t > 0:
            # map t to 0..L
            for i, name in enumerate(elems):
                xpos = (times[name] / max_t) * L
                ax.scatter([xpos], [0], color=colors[i % len(colors)], s=60)
                ax.text(xpos, 0.05, f"{name}\n{times[name]*1e3:.2f} ms", color='white', ha='center')
        else:
            # real scale: map t to x using heuristic scale
            tscale = 0.6 / max_t if max_t > 0 else 1.0
            for i, name in enumerate(elems):
                xpos = min(L, times[name] * tscale)
                ax.scatter([xpos], [0], color=colors[i % len(colors)], s=60)
                ax.text(xpos, 0.05, f"{name}\n{times[name]*1e3:.2f} ms", color='white', ha='center')
        ax.set_ylim(-0.2, 0.4)
        ax.set_xlim(-0.1, 1.1)
        self.panel_traj.canvas.draw()
        self.traj_info.setText(f"Trajectories (TOF) | scale={self.scale_mode}")

# ---- Main ----

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MassLabApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
