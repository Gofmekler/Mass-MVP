"""
MassLab_v4.py
Улучшенный учебный симулятор масс-спектрометра
- Magnetic sector (радиус) и TOF (time-of-flight)
- Динамические панели управления для вкладок "Спектр" и "Траектория"
- Вкладки: Спектр и Траектории
- Тёмная тема, масштаб Real / Normalized
- 3 калибровочных элемента: H, He, Ar
- Режим "Неизвестный элемент" с проверкой ответа
- Визуализация радиусов кривизны с дугами
"""

import sys, random
import numpy as np
from scipy.signal import find_peaks
import matplotlib
from matplotlib.patches import Arc
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

def magnetic_radius(m, U, B, q=E_CHARGE):
    return (1.0 / B) * np.sqrt((2.0 * m * U) / q)

# ---- Сигналы ----
def generate_tof_spectrum(mz_list, intens_list, a=0.0015, t0=0.0002,
                          t_min=0.0, t_max=0.02, n_points=3000,
                          detector_sigma=6e-5, noise_level=0.02, seed=None):
    if seed is not None: np.random.seed(seed)
    t = np.linspace(t_min, t_max, n_points)
    sig = np.zeros_like(t)
    for mz, inten in zip(mz_list, intens_list):
        mu = tof_time_for_mz(mz, a=a, t0=t0)
        sig += inten * np.exp(-0.5 * ((t - mu) / detector_sigma) ** 2)
    max_amp = np.max(sig) if np.max(sig) > 0 else 1.0
    sig += noise_level * max_amp * np.random.normal(0,1,t.shape)
    sig[sig<0]=0
    return t, sig

def generate_magnetic_spectrum(mz_list, intens_list, U, B,
                               r_min=0.0, r_max=0.5, n_points=3000,
                               detector_sigma=1e-3, noise_level=0.02, seed=None):
    if seed is not None: np.random.seed(seed)
    r = np.linspace(r_min, r_max, n_points)
    sig = np.zeros_like(r)
    for mz, inten in zip(mz_list, intens_list):
        m_kg = mz * ELEM_AMU
        mu = magnetic_radius(m_kg, U, B)
        sig += inten * np.exp(-0.5 * ((r - mu) / detector_sigma) ** 2)
    max_amp = np.max(sig) if np.max(sig) > 0 else 1.0
    sig += noise_level * max_amp * np.random.normal(0,1,r.shape)
    sig[sig<0]=0
    return r, sig

def detect_peaks_simple(x, y, prominence_rel=0.05):
    prom = prominence_rel * np.max(y)
    peaks, _ = find_peaks(y, prominence=prom)
    return peaks

# ---- GUI ----
class MplPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, width=6, height=4, dpi=110):
        super().__init__(parent)
        self.fig = Figure(figsize=(width,height), dpi=dpi, tight_layout=True)
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
        self.sample_mz = [28.0,32.0,44.0]
        self.sample_int = [1.0,0.6,0.8]
        # UI
        self._init_ui()
        self.update_plots()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # === Left panel: dynamic controls ===
        self.panel_controls_spectrum = QtWidgets.QVBoxLayout()
        self.panel_controls_traj = QtWidgets.QVBoxLayout()

        left_widget = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QStackedLayout(left_widget)
        main_layout.addWidget(left_widget, stretch=0)

        # --- Spectrum controls ---
        spectrum_widget = QtWidgets.QWidget()
        spectrum_layout = QtWidgets.QVBoxLayout(spectrum_widget)

        # Mode
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Magnetic","TOF"])
        spectrum_layout.addWidget(QtWidgets.QLabel("Выбери режим:"))
        spectrum_layout.addWidget(self.combo_mode)

        # Scale
        self.combo_scale = QtWidgets.QComboBox()
        self.combo_scale.addItems(["Real","Normalized"])
        spectrum_layout.addWidget(QtWidgets.QLabel("Масштаб:"))
        spectrum_layout.addWidget(self.combo_scale)

        # Calibrants
        cal_box = QtWidgets.QGroupBox("Калибровочные элементы")
        cal_layout = QtWidgets.QVBoxLayout(cal_box)
        self.check_cal = {}
        for name in CALIBRANTS:
            cb = QtWidgets.QCheckBox(f"{name} (m={ELEMENTS[name]:.4g} u)")
            cb.setChecked(True)
            cal_layout.addWidget(cb)
            self.check_cal[name] = cb
        spectrum_layout.addWidget(cal_box)

        # U/B sliders
        self.slider_U = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_U.setMinimum(100)
        self.slider_U.setMaximum(20000)
        self.slider_U.setValue(int(self.U))
        self.lbl_U = QtWidgets.QLabel(f"U = {self.U:.0f} V")
        spectrum_layout.addWidget(self.lbl_U)
        spectrum_layout.addWidget(self.slider_U)

        self.slider_B = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_B.setMinimum(1)
        self.slider_B.setMaximum(1000)
        self.slider_B.setValue(int(self.B*1000))
        self.lbl_B = QtWidgets.QLabel(f"B = {self.B:.3f} T")
        spectrum_layout.addWidget(self.lbl_B)
        spectrum_layout.addWidget(self.slider_B)

        # Buttons
        self.btn_generate = QtWidgets.QPushButton("Обновить спектр")
        self.btn_detect = QtWidgets.QPushButton("Найти пики")
        spectrum_layout.addWidget(self.btn_generate)
        spectrum_layout.addWidget(self.btn_detect)

        spectrum_layout.addStretch()
        spectrum_widget.setLayout(spectrum_layout)
        self.left_layout.addWidget(spectrum_widget)

        # --- Trajectory controls ---
        traj_widget = QtWidgets.QWidget()
        traj_layout = QtWidgets.QVBoxLayout(traj_widget)

        traj_layout.addWidget(QtWidgets.QLabel("Слайдеры для изменения U и B:"))
        self.slider_U_traj = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_U_traj.setMinimum(100)
        self.slider_U_traj.setMaximum(20000)
        self.slider_U_traj.setValue(int(self.U))
        self.lbl_U_traj = QtWidgets.QLabel(f"U = {self.U:.0f} V")
        traj_layout.addWidget(self.lbl_U_traj)
        traj_layout.addWidget(self.slider_U_traj)

        self.slider_B_traj = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_B_traj.setMinimum(1)
        self.slider_B_traj.setMaximum(1000)
        self.slider_B_traj.setValue(int(self.B*1000))
        self.lbl_B_traj = QtWidgets.QLabel(f"B = {self.B:.3f} T")
        traj_layout.addWidget(self.lbl_B_traj)
        traj_layout.addWidget(self.slider_B_traj)

        traj_layout.addWidget(QtWidgets.QLabel("Выбор элементов для траекторий:"))
        self.check_traj_elements = {}
        for name in CALIBRANTS:
            cb = QtWidgets.QCheckBox(f"{name}")
            cb.setChecked(True)
            traj_layout.addWidget(cb)
            self.check_traj_elements[name] = cb
        traj_layout.addStretch()
        traj_widget.setLayout(traj_layout)
        self.left_layout.addWidget(traj_widget)

        # === Right panel: Tabs ===
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)
        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs)

        self.tab_spectrum = QtWidgets.QWidget()
        self.panel_spec = MplPanel(self.tab_spectrum)
        tab_layout = QtWidgets.QVBoxLayout(self.tab_spectrum)
        tab_layout.addWidget(self.panel_spec)
        self.tab_spectrum.setLayout(tab_layout)
        self.tabs.addTab(self.tab_spectrum,"Спектр")

        self.tab_traj = QtWidgets.QWidget()
        self.panel_traj = MplPanel(self.tab_traj)
        tab_traj_layout = QtWidgets.QVBoxLayout(self.tab_traj)
        tab_traj_layout.addWidget(self.panel_traj)
        self.tab_traj.setLayout(tab_traj_layout)
        self.tabs.addTab(self.tab_traj,"Траектория")

        # === Signals ===
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.combo_mode.currentTextChanged.connect(lambda v: setattr(self,'mode',v))
        self.combo_scale.currentTextChanged.connect(lambda v: setattr(self,'scale_mode',v))
        self.slider_U.valueChanged.connect(self.update_U_from_slider)
        self.slider_B.valueChanged.connect(self.update_B_from_slider)
        self.slider_U_traj.valueChanged.connect(self.update_U_from_slider_traj)
        self.slider_B_traj.valueChanged.connect(self.update_B_from_slider_traj)
        self.btn_generate.clicked.connect(self.update_plots)
        self.btn_detect.clicked.connect(self.update_plots)

    # --- Slider handlers ---
    def update_U_from_slider(self,v):
        self.U=float(v)
        self.lbl_U.setText(f"U={self.U:.0f} V")
        self.update_plots()
    def update_B_from_slider(self,v):
        self.B=float(v)/1000
        self.lbl_B.setText(f"B={self.B:.3f} T")
        self.update_plots()
    def update_U_from_slider_traj(self,v):
        self.U=float(v)
        self.lbl_U_traj.setText(f"U={self.U:.0f} V")
        self.update_plots()
    def update_B_from_slider_traj(self,v):
        self.B=float(v)/1000
        self.lbl_B_traj.setText(f"B={self.B:.3f} T")
        self.update_plots()

    def on_tab_changed(self,index):
        self.left_layout.setCurrentIndex(index)
        self.update_plots()

    # ---- Plots ----
    def update_plots(self):
        # --- Spectrum tab ---
        if self.tabs.currentWidget() == self.tab_spectrum:
            active_cal = [name for name, cb in self.check_cal.items() if cb.isChecked()]
            mz_list = [ELEMENTS[n] for n in active_cal]
            intens_list = [1.0]*len(mz_list)
            t, sig = generate_tof_spectrum(mz_list, intens_list)
            self.panel_spec.ax.clear()
            self.panel_spec.ax.plot(t*1000, sig, color="cyan")
            self.panel_spec.ax.set_xlabel("Время, ms")
            self.panel_spec.ax.set_ylabel("Интенсивность")
            self.panel_spec.canvas.draw()
        # --- Trajectory tab ---
        else:
            self.panel_traj.ax.clear()
            colors = ['yellow','magenta','cyan']
            for i,(name,cb) in enumerate(self.check_traj_elements.items()):
                if not cb.isChecked(): continue
                r_mag = magnetic_radius(ELEMENTS[name]*ELEM_AMU,self.U,self.B)
                arc = Arc((0,0), width=2*r_mag, height=2*r_mag, theta1=0, theta2=180, color=colors[i%3], lw=2)
                self.panel_traj.ax.add_patch(arc)
                self.panel_traj.ax.plot(0,0,'ro')  # источник
                self.panel_traj.ax.plot(r_mag,0,'bo')  # детектор
                self.panel_traj.ax.text(r_mag/2, r_mag/8, f"{name}",color=colors[i%3])
            self.panel_traj.ax.set_xlim(-0.05,0.6)
            self.panel_traj.ax.set_ylim(-0.05,0.35)
            self.panel_traj.ax.axis('off')
            self.panel_traj.canvas.draw()

# ---- Main ----
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MassLabApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
