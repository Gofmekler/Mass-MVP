"""
MassLab_v3.py
Универсальный учебный симулятор масс-спектрометра
- Magnetic sector (радиус) и TOF (time-of-flight)
- Вкладки: Спектр и Траектории
- Тёмная тема, масштаб Real / Normalized
- 3 калибровочных элемента: H, He, Ar
- Режим "Неизвестный элемент" с проверкой ответа
"""

import sys, random, math
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib
try:
    from PySide6 import QtWidgets, QtCore, QtGui
    matplotlib.use("QtAgg")
except:
    from PyQt5 import QtWidgets, QtCore, QtGui
    matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

plt.style.use('dark_background')

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

CALIBRANTS = ["H", "He", "Ar"]

ELEM_AMU = 1.66053906660e-27  # kg
E_CHARGE = 1.602176634e-19  # C

# ---- Физические модели ----
def tof_time_for_mz(mz, a=0.0015, t0=0.0):
    return a * np.sqrt(np.array(mz)) + t0

def tof_mz_from_time(t, a, t0):
    return ((t - t0) / a) ** 2

def magnetic_radius(m, U, B, q=E_CHARGE):
    return (1.0 / B) * np.sqrt((2.0 * m * U) / q)

# ---- Сигнал ----
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

def detect_peaks_simple(x, y, prominence_rel=0.05):
    prom = prominence_rel * np.max(y)
    peaks, props = find_peaks(y, prominence=prom)
    return peaks, props

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
        self.setWindowTitle("MassLab — учебный симулятор")
        self.resize(1200, 760)
        # State
        self.mode = "Magnetic"
        self.scale_mode = "Real"
        self.U = 3000.0
        self.B = 0.3
        self.tof_a = 0.0015
        self.tof_t0 = 0.0002
        self.det_sigma_mag = 0.005
        self.det_sigma_tof = 6e-5
        self.noise = 0.02
        self.tmin, self.tmax = 0.0, 0.02
        self.npoints = 3000
        self.current_unknown = None
        # UI
        self._init_ui()
        self.sample_mz = [28.0, 32.0, 44.0]
        self.sample_int = [1.0, 0.6, 0.8]
        self.update_plots()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

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

        # Sliders U and B
        pb = QtWidgets.QGroupBox("Параметры прибора")
        pbl = QtWidgets.QFormLayout(pb)
        self.slider_U = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_U.setMinimum(100)
        self.slider_U.setMaximum(20000)
        self.slider_U.setValue(int(self.U))
        self.lbl_U = QtWidgets.QLabel(f"U = {self.U:.0f} V")
        pbl.addRow(self.lbl_U, self.slider_U)

        self.slider_B = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_B.setMinimum(1)
        self.slider_B.setMaximum(1000)
        self.slider_B.setValue(int(self.B * 1000))
        self.lbl_B = QtWidgets.QLabel(f"B = {self.B:.3f} T")
        pbl.addRow(self.lbl_B, self.slider_B)

        left.addWidget(pb)

        # Unknown element
        unk_box = QtWidgets.QGroupBox("Режим: Неизвестный элемент")
        unk_layout = QtWidgets.QVBoxLayout(unk_box)
        self.btn_new_unknown = QtWidgets.QPushButton("Сгенерировать неизвестный элемент")
        self.lbl_measured = QtWidgets.QLabel("—")
        unk_layout.addWidget(self.btn_new_unknown)
        unk_layout.addWidget(QtWidgets.QLabel("Измеренное:"))
        unk_layout.addWidget(self.lbl_measured)
        self.combo_guess = QtWidgets.QComboBox()
        self.combo_guess.addItems(sorted(ELEMENTS.keys()))
        unk_layout.addWidget(self.combo_guess)
        self.btn_check = QtWidgets.QPushButton("Проверить ответ")
        unk_layout.addWidget(self.btn_check)
        self.lbl_feedback = QtWidgets.QLabel("")
        unk_layout.addWidget(self.lbl_feedback)
        left.addWidget(unk_box)

        # Buttons
        ops = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Обновить спектр")
        self.btn_detect = QtWidgets.QPushButton("Найти пики")
        ops.addWidget(self.btn_generate)
        ops.addWidget(self.btn_detect)
        left.addLayout(ops)
        left.addStretch()
        left.addWidget(QtWidgets.QLabel("MassLab — учебный симулятор"))

        # Tabs
        right = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right, stretch=1)

        self.tabs = QtWidgets.QTabWidget()
        right.addWidget(self.tabs)

        # Spectrum tab
        self.tab_spectrum = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_spectrum, "Спектр")
        self.panel_spec = MplPanel(self.tab_spectrum)
        tab_layout = QtWidgets.QVBoxLayout(self.tab_spectrum)
        tab_layout.addWidget(self.panel_spec)

        # Trajectory tab
        self.tab_traj = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_traj, "Траектория")
        self.panel_traj = MplPanel(self.tab_traj)
        tab_traj_layout = QtWidgets.QVBoxLayout(self.tab_traj)
        tab_traj_layout.addWidget(self.panel_traj)

        # Signals
        self.combo_mode.currentTextChanged.connect(self._change_mode)
        self.combo_scale.currentTextChanged.connect(self._change_scale)
        self.slider_U.valueChanged.connect(self._change_U)
        self.slider_B.valueChanged.connect(self._change_B)
        self.btn_generate.clicked.connect(self.update_plots)
        self.btn_detect.clicked.connect(self.detect_peaks)
        self.btn_new_unknown.clicked.connect(self.new_unknown_element)
        self.btn_check.clicked.connect(self.check_unknown)

    # ---- Handlers ----
    def _change_mode(self, v): self.mode = v; self.update_plots()
    def _change_scale(self, v): self.scale_mode = v; self.update_plots()
    def _change_U(self, v):
        self.U = float(v)
        self.lbl_U.setText(f"U = {self.U:.0f} V")
        self.update_plots()
    def _change_B(self, v):
        self.B = float(v)/1000
        self.lbl_B.setText(f"B = {self.B:.3f} T")
        self.update_plots()

    def new_unknown_element(self):
        self.current_unknown = random.choice(list(ELEMENTS.keys()))
        self.lbl_feedback.setText("Элемент сгенерирован. Измерьте его пиковое значение.")
        self.update_plots()

    def check_unknown(self):
        if not self.current_unknown:
            self.lbl_feedback.setText("Сначала сгенерируйте неизвестный элемент")
            return
        guess = self.combo_guess.currentText()
        correct = guess == self.current_unknown
        self.lbl_feedback.setText("✅ Верно!" if correct else f"❌ Неверно, правильный: {self.current_unknown}")

    # ---- Plots ----
    def update_plots(self):
        # Prepare spectrum
        active_cal = [name for name, cb in self.check_cal.items() if cb.isChecked()]
        mz_list = [ELEMENTS[n] for n in active_cal]
        intens_list = [1.0]*len(mz_list)
        # add unknown if mode active
        if self.current_unknown:
            mz_list.append(ELEMENTS[self.current_unknown])
            intens_list.append(1.0)
        if self.mode == "TOF":
            t, sig = generate_tof_spectrum(mz_list, intens_list,
                                           a=self.tof_a, t0=self.tof_t0,
                                           t_min=self.tmin, t_max=self.tmax,
                                           n_points=self.npoints,
                                           detector_sigma=self.det_sigma_tof,
                                           noise_level=self.noise)
            self.panel_spec.ax.clear()
            self.panel_spec.ax.plot(t*1000, sig, color="cyan")
            self.panel_spec.ax.set_xlabel("Время, ms")
            self.panel_spec.ax.set_ylabel("Интенсивность")
            # peaks
            peaks, _ = detect_peaks_simple(t, sig)
            for idx in peaks:
                nearest_mz = min(mz_list, key=lambda mzv: abs(tof_time_for_mz(mzv, self.tof_a, self.tof_t0)-t[idx]))
                name_guess = None
                for nm in ELEMENTS:
                    if abs(ELEMENTS[nm] - nearest_mz) < 1e-9: name_guess = nm; break
                label = name_guess if name_guess else f"{nearest_mz:.2f}"
                self.panel_spec.ax.annotate(label, (t[idx]*1000, sig[idx]), textcoords="offset points", xytext=(0,8),
                                            ha='center', fontsize=8)
            self.lbl_measured.setText(f"{mz_list[-1]:.4f} u" if self.current_unknown else "—")
        else:
            r, sig = generate_magnetic_spectrum(mz_list, intens_list,
                                                U=self.U, B=self.B,
                                                detector_sigma=self.det_sigma_mag,
                                                n_points=self.npoints,
                                                noise_level=self.noise)
            self.panel_spec.ax.clear()
            self.panel_spec.ax.plot(r*100, sig, color="orange")
            self.panel_spec.ax.set_xlabel("Радиус, см")
            self.panel_spec.ax.set_ylabel("Интенсивность")
            peaks, _ = detect_peaks_simple(r, sig)
            for idx in peaks:
                nearest_mz = min(mz_list, key=lambda mzv: abs(magnetic_radius(mzv*ELEM_AMU, self.U, self.B)-r[idx]))
                name_guess = None
                for nm in ELEMENTS:
                    if abs(ELEMENTS[nm] - nearest_mz) < 1e-9: name_guess = nm; break
                label = name_guess if name_guess else f"{nearest_mz:.2f}"
                self.panel_spec.ax.annotate(label, (r[idx]*100, sig[idx]), textcoords="offset points", xytext=(0,8),
                                            ha='center', fontsize=8)
            self.lbl_measured.setText(f"{mz_list[-1]:.4f} u" if self.current_unknown else "—")
        self.panel_spec.canvas.draw()

        # Trajectory tab: simple sketch
        self.panel_traj.ax.clear()
        self.panel_traj.ax.plot([0,0.2], [0,0.0], 'w')  # source
        self.panel_traj.ax.plot([0.2,0.5], [0.0,0.3], 'y')  # magnet
        self.panel_traj.ax.plot([0.5,0.7], [0.3,0.0], 'r')  # detector
        self.panel_traj.ax.text(0.0, -0.05, "Источник", color='white')
        self.panel_traj.ax.text(0.5, 0.35, "Анализатор", color='yellow')
        self.panel_traj.ax.text(0.7, -0.05, "Детектор", color='red')
        self.panel_traj.ax.set_xlim(-0.1, 0.8)
        self.panel_traj.ax.set_ylim(-0.1, 0.4)
        self.panel_traj.ax.axis('off')
        self.panel_traj.canvas.draw()

    def detect_peaks(self):
        self.update_plots()  # просто обновим спектр с аннотациями

# ---- Main ----
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MassLabApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
