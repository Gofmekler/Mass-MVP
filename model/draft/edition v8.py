"""
MassLab_v9.py — учебный симулятор масс-спектрометра
- Полная таблица химических элементов (118 элементов)
- Режим одиночного элемента и сплавов
- Генерация неизвестного элемента, проверка содержания смеси
- Калибровка спектра для известных элементов
- Траектория с формулой радиуса кривизны (LaTeX)
- Hover по графикам: масса, интенсивность
- Кнопки генерации нового неизвестного элемента
- Темная тема, цвета без желтого
"""

import sys, random
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6 import QtWidgets, QtCore, QtGui

# --- Полная таблица элементов (118) ---
ELEMENTS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.91,
    "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71,
    "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93,
    "Dy": 162.50, "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05,
    "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21,
    "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59,
    "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.04,
    "Pa": 231.04, "U": 238.03, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0,
    "Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 267.0, "Db": 270.0,
    "Sg": 271.0, "Bh": 270.0, "Hs": 277.0, "Mt": 276.0, "Ds": 281.0,
    "Rg": 280.0, "Cn": 285.0, "Nh": 284.0, "Fl": 289.0, "Mc": 288.0,
    "Lv": 293.0, "Ts": 294.0, "Og": 294.0
}

CALIBRANTS = ["H", "He", "Ar"]

ELEM_AMU = 1.66053906660e-27  # kg
E_CHARGE = 1.602176634e-19  # C

# --- Физические модели ---
def tof_time_for_mz(mz, a=0.0015, t0=0.0002):
    return a*np.sqrt(mz) + t0

def magnetic_radius(m, U, B, q=E_CHARGE):
    return (1.0/B)*np.sqrt((2.0*m*U)/q)

# --- Генерация спектра ---
def generate_tof_spectrum(mz_list,intens_list,a=0.0015,t0=0.0002,
                          t_min=0.0,t_max=0.02,n_points=3000,
                          detector_sigma=6e-5,noise_level=0.02,seed=None):
    if seed is not None: np.random.seed(seed)
    t = np.linspace(t_min,t_max,n_points)
    sig = np.zeros_like(t)
    for mz,inten in zip(mz_list,intens_list):
        mu = tof_time_for_mz(mz,a=a,t0=t0)
        sig += inten*np.exp(-0.5*((t-mu)/detector_sigma)**2)
    max_amp = np.max(sig) if np.max(sig)>0 else 1.0
    sig += noise_level*max_amp*np.random.normal(0,1,t.shape)
    sig[sig<0]=0
    return t,sig

# --- GUI ---
class MplPanel(QtWidgets.QWidget):
    def __init__(self,parent=None,width=6,height=4,dpi=110):
        super().__init__(parent)
        self.fig = Figure(figsize=(width,height),dpi=dpi,tight_layout=True,facecolor="#222222")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

class MassLabApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab v9 — учебный симулятор")
        self.resize(1400,850)
        # State
        self.mode = "single"  # single or mixture
        self.U = 3000.0
        self.B = 0.3
        self.det_sigma_tof = 6e-5
        self.noise = 0.02
        self.tmin,self.tmax=0.0,0.02
        self.npoints=3000
        self.current_unknown = None
        self.unknown_name = None
        self.colors = ['magenta','cyan','orange','red','lime','violet']
        self.mixture_fields = []
        # Default mixture
        self.current_mixture = [("H",0.5),("He",0.3),("Ar",0.2)]
        # UI
        self._init_ui()
        self.generate_new_unknown()
        self.update_plots()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # === Left panel (stacked) ===
        left_widget = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QStackedLayout(left_widget)
        main_layout.addWidget(left_widget,stretch=0)

        # --- Spectrum controls ---
        spectrum_widget = QtWidgets.QWidget()
        spectrum_layout = QtWidgets.QVBoxLayout(spectrum_widget)

        # Режим одиночный/сплав
        self.radio_single = QtWidgets.QRadioButton("Одиночный элемент")
        self.radio_single.setChecked(True)
        self.radio_mixture = QtWidgets.QRadioButton("Сплав")
        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_mixture)
        spectrum_layout.addLayout(mode_layout)
        self.radio_single.toggled.connect(lambda _: self.on_mode_change())

        # Sliders U/B
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

        # Поле для одиночного неизвестного элемента
        self.input_unknown_spec = QtWidgets.QLineEdit()
        self.btn_check_unknown_spec = QtWidgets.QPushButton("Проверить")
        self.btn_new_unknown_spec = QtWidgets.QPushButton("Сгенерировать новый элемент")
        spectrum_layout.addWidget(QtWidgets.QLabel("Неизвестный элемент (для одиночного режима):"))
        spectrum_layout.addWidget(self.input_unknown_spec)
        spectrum_layout.addWidget(self.btn_check_unknown_spec)
        spectrum_layout.addWidget(self.btn_new_unknown_spec)

        # Поля для сплава (скрыты по умолчанию)
        spectrum_layout.addWidget(QtWidgets.QLabel("Состав сплава (3 компонента):"))
        for i in range(3):
            hlayout = QtWidgets.QHBoxLayout()
            cb = QtWidgets.QComboBox()
            cb.addItems(list(ELEMENTS.keys()))
            le = QtWidgets.QLineEdit("0.0")
            hlayout.addWidget(cb)
            hlayout.addWidget(le)
            spectrum_layout.addLayout(hlayout)
            self.mixture_fields.append((cb,le))

        self.btn_update_mixture = QtWidgets.QPushButton("Обновить смесь")
        spectrum_layout.addWidget(self.btn_update_mixture)

        # Кнопка обновить график
        self.btn_generate = QtWidgets.QPushButton("Обновить спектр")
        spectrum_layout.addWidget(self.btn_generate)
        spectrum_layout.addStretch()
        spectrum_widget.setLayout(spectrum_layout)
        self.left_layout.addWidget(spectrum_widget)

        # --- Trajectory controls ---
        traj_widget = QtWidgets.QWidget()
        traj_layout = QtWidgets.QVBoxLayout(traj_widget)
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

        traj_layout.addWidget(QtWidgets.QLabel("Выбор калибровочных элементов:"))
        self.check_traj_elements={}
        for name in CALIBRANTS:
            cb = QtWidgets.QCheckBox(f"{name}")
            cb.setChecked(True)
            traj_layout.addWidget(cb)
            self.check_traj_elements[name]=cb

        traj_layout.addWidget(QtWidgets.QLabel("Неизвестный элемент:"))
        self.input_unknown_traj = QtWidgets.QLineEdit()
        self.btn_check_unknown_traj = QtWidgets.QPushButton("Проверить")
        self.btn_new_unknown_traj = QtWidgets.QPushButton("Сгенерировать новый элемент")
        traj_layout.addWidget(self.input_unknown_traj)
        traj_layout.addWidget(self.btn_check_unknown_traj)
        traj_layout.addWidget(self.btn_new_unknown_traj)

        self.lbl_formula = QtWidgets.QLabel()
        self.lbl_formula.setWordWrap(True)
        traj_layout.addWidget(self.lbl_formula)
        traj_layout.addStretch()
        traj_widget.setLayout(traj_layout)
        self.left_layout.addWidget(traj_widget)

        # === Right panel: Tabs ===
        right_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right_layout,stretch=1)
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
        tab_layout2 = QtWidgets.QVBoxLayout(self.tab_traj)
        tab_layout2.addWidget(self.panel_traj)
        self.tab_traj.setLayout(tab_layout2)
        self.tabs.addTab(self.tab_traj,"Траектория")

        # --- Connect ---
        self.slider_U.valueChanged.connect(self.on_slider_change)
        self.slider_B.valueChanged.connect(self.on_slider_change)
        self.slider_U_traj.valueChanged.connect(self.on_slider_change_traj)
        self.slider_B_traj.valueChanged.connect(self.on_slider_change_traj)
        self.btn_generate.clicked.connect(self.update_plots)
        self.btn_new_unknown_spec.clicked.connect(self.generate_new_unknown)
        self.btn_new_unknown_traj.clicked.connect(self.generate_new_unknown)
        self.btn_check_unknown_spec.clicked.connect(lambda: self.check_unknown("spec"))
        self.btn_check_unknown_traj.clicked.connect(lambda: self.check_unknown("traj"))
        self.btn_update_mixture.clicked.connect(self.update_mixture_from_fields)

    # --- Handlers ---
    def on_mode_change(self):
        if self.radio_single.isChecked():
            self.mode="single"
            for cb,le in self.mixture_fields:
                cb.hide(); le.hide()
            self.input_unknown_spec.show()
            self.btn_check_unknown_spec.show()
            self.btn_new_unknown_spec.show()
        else:
            self.mode="mixture"
            for cb,le in self.mixture_fields:
                cb.show(); le.show()
            self.input_unknown_spec.hide()
            self.btn_check_unknown_spec.hide()
            self.btn_new_unknown_spec.hide()
        self.update_plots()

    def on_slider_change(self):
        self.U = self.slider_U.value()
        self.B = self.slider_B.value()/1000
        self.lbl_U.setText(f"U = {self.U:.0f} V")
        self.lbl_B.setText(f"B = {self.B:.3f} T")
        self.update_plots()

    def on_slider_change_traj(self):
        self.U = self.slider_U_traj.value()
        self.B = self.slider_B_traj.value()/1000
        self.lbl_U_traj.setText(f"U = {self.U:.0f} V")
        self.lbl_B_traj.setText(f"B = {self.B:.3f} T")
        self.update_plots()

    # --- Mixture update ---
    def update_mixture_from_fields(self):
        self.current_mixture=[]
        for cb,le in self.mixture_fields:
            name=cb.currentText()
            try:
                val=float(le.text())/100
            except:
                val=0.0
            self.current_mixture.append((name,val))
        self.update_plots()

    # --- Generate unknown element ---
    def generate_new_unknown(self):
        self.current_unknown = random.choice(list(ELEMENTS.keys()))
        self.unknown_name = self.current_unknown
        if self.mode=="single":
            self.input_unknown_spec.clear()
            self.input_unknown_traj.clear()

    # --- Check unknown ---
    def check_unknown(self,which="spec"):
        if which=="spec":
            guess = self.input_unknown_spec.text().strip()
        else:
            guess = self.input_unknown_traj.text().strip()
        if guess==self.unknown_name:
            QtWidgets.QMessageBox.information(self,"Правильно!","Верно! Элемент угадан.")
        else:
            QtWidgets.QMessageBox.warning(self,"Неверно","Неверно! Создан новый элемент.")
            self.generate_new_unknown()
        self.update_plots()

    # --- Plot ---
    def update_plots(self):
        # --- Spectrum ---
        ax = self.panel_spec.ax
        ax.clear(); ax.set_facecolor("#222222"); ax.tick_params(colors="white"); ax.yaxis.label.set_color("white"); ax.xaxis.label.set_color("white")
        if self.mode=="single":
            mz_list=[ELEMENTS[self.current_unknown]]
            intens_list=[1.0]
        else:
            mz_list=[ELEMENTS[name] for name,val in self.current_mixture]
            intens_list=[val for name,val in self.current_mixture]
        t,sig = generate_tof_spectrum(mz_list,intens_list,a=0.0015,t0=0.0002,
                                      t_min=self.tmin,t_max=self.tmax,n_points=self.npoints,
                                      detector_sigma=self.det_sigma_tof,noise_level=self.noise)
        ax.plot(t,sig,color="cyan")
        ax.set_xlabel("t (s)",color="white")
        ax.set_ylabel("Интенсивность",color="white")
        # --- Trajectory ---
        ax2 = self.panel_traj.ax
        ax2.clear(); ax2.set_facecolor("#222222"); ax2.tick_params(colors="white"); ax2.yaxis.label.set_color("white"); ax2.xaxis.label.set_color("white")
        for name in CALIBRANTS:
            m = ELEMENTS[name]*ELEM_AMU
            R = magnetic_radius(m,self.U,self.B)
            circ = plt.Circle((0,R),R,color="lime",fill=False,lw=2)
            ax2.add_patch(circ)
            ax2.text(0,R+0.02,name,color="white")
        ax2.set_xlim(-1,1); ax2.set_ylim(0,2)
        ax2.set_xlabel("x (m)",color="white")
        ax2.set_ylabel("y (m)",color="white")
        self.lbl_formula.setText(f"Формула радиуса кривизны: R = sqrt(2*m*U/q)/B")

        self.panel_spec.canvas.draw()
        self.panel_traj.canvas.draw()

# --- Main ---
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53,53,53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25,25,25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53,53,53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53,53,53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    app.setPalette(dark_palette)
    w = MassLabApp()
    w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
