"""
MassLab_v7.py — обновлённая учебная версия
- Траектория: дуги радиусов, формула R красиво отображается
- Спектр: неизвестный элемент, анализ сплавов
- Hover-аннотации, интерактивные поля и кнопки проверки
- Цвета дуг: magenta, cyan, orange, red
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

# --- Qt ---
try:
    from PySide6 import QtWidgets, QtCore, QtGui
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui

# --- Константы ---
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
CALIBRANTS = ["H","He","Ar"]
ELEM_AMU = 1.66053906660e-27  # kg
E_CHARGE = 1.602176634e-19  # C

# --- Физические модели ---
def tof_time_for_mz(mz, a=0.0015, t0=0.0):
    return a*np.sqrt(mz) + t0

def magnetic_radius(m, U, B, q=E_CHARGE):
    return (1.0/B)*np.sqrt((2.0*m*U)/q)

# --- Генерация сигналов ---
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
        self.fig = Figure(figsize=(width,height),dpi=dpi,tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

class MassLabApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab v7 — учебный симулятор")
        self.resize(1300,800)
        # State
        self.mode = "TOF"
        self.U = 3000.0
        self.B = 0.3
        self.det_sigma_tof = 6e-5
        self.noise = 0.02
        self.tmin,self.tmax=0.0,0.02
        self.npoints=3000
        self.current_unknown = None
        self.unknown_name = None
        self.colors = ['magenta','cyan','orange','red']
        # Sample data
        self.sample_mz = [28.0,32.0,44.0]
        self.sample_int = [1.0,0.6,0.8]
        self.current_mixture = [("H",0.5),("He",0.3),("Ar",0.2)]
        # UI
        self._init_ui()
        self.generate_new_unknown()
        self.update_plots()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # === Left panel (stacked for spectrum/trajectory) ===
        left_widget = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QStackedLayout(left_widget)
        main_layout.addWidget(left_widget,stretch=0)

        # --- Spectrum controls ---
        spectrum_widget = QtWidgets.QWidget()
        spectrum_layout = QtWidgets.QVBoxLayout(spectrum_widget)

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["TOF","Magnetic"])
        spectrum_layout.addWidget(QtWidgets.QLabel("Режим:"))
        spectrum_layout.addWidget(self.combo_mode)

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

        # Поле для проверки неизвестного элемента
        spectrum_layout.addWidget(QtWidgets.QLabel("Введите неизвестный элемент:"))
        self.input_unknown_spec = QtWidgets.QLineEdit()
        self.btn_check_unknown_spec = QtWidgets.QPushButton("Проверить")
        spectrum_layout.addWidget(self.input_unknown_spec)
        spectrum_layout.addWidget(self.btn_check_unknown_spec)

        # Поля для сплава
        spectrum_layout.addWidget(QtWidgets.QLabel("Состав сплава:"))
        self.mixture_fields = []
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

        traj_layout.addWidget(QtWidgets.QLabel("Выбор элементов для траекторий:"))
        self.check_traj_elements={}
        for name in CALIBRANTS:
            cb = QtWidgets.QCheckBox(f"{name}")
            cb.setChecked(True)
            traj_layout.addWidget(cb)
            self.check_traj_elements[name]=cb

        traj_layout.addWidget(QtWidgets.QLabel("Неизвестный элемент:"))
        self.input_unknown_traj = QtWidgets.QLineEdit()
        self.btn_check_unknown_traj = QtWidgets.QPushButton("Проверить")
        traj_layout.addWidget(self.input_unknown_traj)
        traj_layout.addWidget(self.btn_check_unknown_traj)

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
        tab_traj_layout = QtWidgets.QVBoxLayout(self.tab_traj)
        tab_traj_layout.addWidget(self.panel_traj)
        self.tab_traj.setLayout(tab_traj_layout)
        self.tabs.addTab(self.tab_traj,"Траектория")

        # Signals
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.combo_mode.currentTextChanged.connect(lambda v:setattr(self,'mode',v))
        self.slider_U.valueChanged.connect(self.update_U_from_slider)
        self.slider_B.valueChanged.connect(self.update_B_from_slider)
        self.slider_U_traj.valueChanged.connect(self.update_U_from_slider_traj)
        self.slider_B_traj.valueChanged.connect(self.update_B_from_slider_traj)
        self.btn_generate.clicked.connect(self.update_plots)
        self.btn_check_unknown_spec.clicked.connect(self.check_unknown_spec)
        self.btn_check_unknown_traj.clicked.connect(self.check_unknown_traj)
        self.btn_update_mixture.clicked.connect(self.update_mixture)
        self.panel_spec.canvas.mpl_connect("motion_notify_event", self.on_hover_spec)

    # Slider handlers
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

    # Unknown element
    def generate_new_unknown(self):
        choices = list(set(ELEMENTS.keys()) - set(CALIBRANTS))
        self.unknown_name = random.choice(choices)
        self.current_unknown = ELEMENTS[self.unknown_name]
        # Скрываем студенту

    # Hover
    def on_hover_spec(self,event):
        if event.inaxes!=self.panel_spec.ax: return
        x,y = event.xdata, event.ydata
        self.statusBar().showMessage(f"x={x:.5f}, y={y:.3f}")

    # Проверка для спектра
    def check_unknown_spec(self):
        user_input = self.input_unknown_spec.text().strip()
        if user_input==self.unknown_name:
            self.statusBar().showMessage(f"✅ Верно! Это {self.unknown_name}")
        else:
            self.statusBar().showMessage(f"❌ Неверно! Был {self.unknown_name}, генерируем новый элемент")
            self.generate_new_unknown()
        self.input_unknown_spec.clear()
        self.update_plots()

    # Проверка для траектории
    def check_unknown_traj(self):
        user_input = self.input_unknown_traj.text().strip()
        if user_input==self.unknown_name:
            self.statusBar().showMessage(f"✅ Верно! Это {self.unknown_name}")
        else:
            self.statusBar().showMessage(f"❌ Неверно! Был {self.unknown_name}, генерируем новый элемент")
            self.generate_new_unknown()
        self.input_unknown_traj.clear()
        self.update_plots()

    # Обновление смеси
    def update_mixture(self):
        mixture=[]
        for cb,le in self.mixture_fields:
            name = cb.currentText()
            try:
                val = float(le.text())
            except:
                val=0.0
            mixture.append((name,val))
        self.current_mixture = mixture
        self.update_plots()

    # Update plots
    def update_plots(self):
        if self.tabs.currentWidget()==self.tab_spectrum:
            self.panel_spec.ax.clear()
            mz_list,intens_list=[],[]
            # Собираем смесь
            for name,val in self.current_mixture:
                if val>0:
                    mz_list.append(ELEMENTS[name])
                    intens_list.append(val)
            # Добавляем неизвестный элемент
            mz_list.append(self.current_unknown)
            intens_list.append(1.0)
            t,sig = generate_tof_spectrum(mz_list,intens_list)
            self.panel_spec.ax.plot(t*1000,sig,color="cyan")
            self.panel_spec.ax.set_xlabel("Время, ms")
            self.panel_spec.ax.set_ylabel("Интенсивность")
            self.panel_spec.canvas.draw()
        else:
            self.panel_traj.ax.clear()
            idx_color = 0
            for name,cb in self.check_traj_elements.items():
                if not cb.isChecked(): continue
                r_mag = magnetic_radius(ELEMENTS[name]*ELEM_AMU,self.U,self.B)
                arc = Arc((0,0),width=2*r_mag,height=2*r_mag,theta1=0,theta2=180,
                          color=self.colors[idx_color%len(self.colors)],lw=2)
                self.panel_traj.ax.add_patch(arc)
                self.panel_traj.ax.plot(0,0,'ro')  # источник
                self.panel_traj.ax.plot(r_mag,0,'bo')  # детектор
                self.panel_traj.ax.text(r_mag/2,r_mag/8,f"{name}\nR={r_mag*100:.1f} cm",
                                        color=self.colors[idx_color%len(self.colors)])
                idx_color +=1
            # Unknown element
            r_unknown = magnetic_radius(self.current_unknown*ELEM_AMU,self.U,self.B)
            arc = Arc((0,0),width=2*r_unknown,height=2*r_unknown,theta1=0,theta2=180,
                      color='red',lw=2,linestyle='--')
            self.panel_traj.ax.add_patch(arc)
            self.panel_traj.ax.text(r_unknown/2,r_unknown/5,f"??\nR={r_unknown*100:.1f} cm",color='red')
            formula = r"R = \frac{1}{B} \sqrt{\frac{2 m U}{q}}"
            self.lbl_formula.setText(f"<b>Формула радиуса кривизны:</b><br>"
                                     f"<span style='font-size:16pt'>${formula}$</span><br>"
                                     f"U={self.U:.0f} V, B={self.B:.3f} T")
            self.panel_traj.ax.set_xlim(-0.05,0.6)
            self.panel_traj.ax.set_ylim(-0.05,0.35)
            self.panel_traj.ax.axis('off')
            self.panel_traj.canvas.draw()

# Main
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MassLabApp()
    w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
