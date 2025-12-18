import sys, random
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from PySide6 import QtWidgets, QtCore, QtGui

ELEMENTS = {
    "H":1.008, "He":4.0026, "Li":6.94, "Be":9.0122, "B":10.81,
    "C":12.011, "N":14.007, "O":15.999, "F":18.998, "Ne":20.180,
    "Na":22.990, "Mg":24.305, "Al":26.982, "Si":28.085, "P":30.974,
    "S":32.06, "Cl":35.45, "Ar":39.948, "K":39.098, "Ca":40.078
}

CALIBRANTS = ["H","He","Ar"]
E_CHARGE = 1.602176634e-19
E_AMU = 1.66053906660e-27

def tof_time_for_mz(mz, a=0.0015, t0=0.0002):
    return a*np.sqrt(mz)+t0

def magnetic_radius(m, U, B, q=E_CHARGE):
    return (1.0/B)*np.sqrt((2.0*m*U)/q)

def generate_tof_spectrum(mz_list, intens_list, a=0.0015, t0=0.0002, t_min=0.0, t_max=0.02, n_points=3000, detector_sigma=6e-5, noise_level=0.02, seed=None):
    if seed is not None: np.random.seed(seed)
    t = np.linspace(t_min,t_max,n_points)
    sig = np.zeros_like(t)
    for mz,inten in zip(mz_list,intens_list):
        mu = tof_time_for_mz(mz,a=a,t0=t0)
        sig += inten*np.exp(-0.5*((t-mu)/detector_sigma)**2)
    max_amp = np.max(sig) if np.max(sig)>0 else 1.0
    sig += noise_level*max_amp*np.random.normal(0,1,t.shape)
    sig[sig<0]=0
    return t, sig

class MplPanel(QtWidgets.QWidget):
    def __init__(self, parent=None, width=6, height=4, dpi=110):
        super().__init__(parent)
        self.fig = Figure(figsize=(width,height),dpi=dpi,tight_layout=True,facecolor="#222222")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

class MassLabApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab v11.1 macOS — PySide6")
        self.resize(1700,950)
        self.mode="single"
        self.U = 3000.0
        self.B = 0.3
        self.current_unknown = None
        self.unknown_name = None
        self.current_mixture = [("H",0.5),("He",0.3),("Ar",0.2)]
        self.colors=['magenta','cyan','orange','red','lime','violet']
        self.det_sigma_tof = 6e-5
        self.noise = 0.02
        self.tmin,self.tmax = 0.0,0.02
        self.npoints = 3000
        self._init_ui()
        self.generate_new_unknown()
        self.update_plots()

    def _init_ui(self):
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        ctrl_panel = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_panel)

        self.radio_single = QtWidgets.QRadioButton("Одиночный элемент")
        self.radio_single.setChecked(True)
        self.radio_mixture = QtWidgets.QRadioButton("Сплав элементов")
        ctrl_layout.addWidget(self.radio_single)
        ctrl_layout.addWidget(self.radio_mixture)
        self.radio_single.toggled.connect(self.on_mode_change)

        self.input_unknown_spec = QtWidgets.QLineEdit()
        self.input_unknown_spec.setPlaceholderText("Введите неизвестный элемент")
        self.btn_check_unknown_spec = QtWidgets.QPushButton("Проверить")
        self.btn_new_unknown_spec = QtWidgets.QPushButton("Сгенерировать новый элемент")
        ctrl_layout.addWidget(self.input_unknown_spec)
        ctrl_layout.addWidget(self.btn_check_unknown_spec)
        ctrl_layout.addWidget(self.btn_new_unknown_spec)

        self.table_mixture = QtWidgets.QTableWidget(5,2)
        self.table_mixture.setHorizontalHeaderLabels(["Элемент","Процент"])
        ctrl_layout.addWidget(self.table_mixture)

        ctrl_layout.addStretch()

        self.tabs = QtWidgets.QTabWidget()
        self.panel_spec = MplPanel()
        self.panel_traj = MplPanel()
        self.tabs.addTab(self.panel_spec, "Спектр")
        self.tabs.addTab(self.panel_traj, "Траектория")

        main_layout.addWidget(ctrl_panel,1)
        main_layout.addWidget(self.tabs,3)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.btn_new_unknown_spec.clicked.connect(self.generate_new_unknown)
        self.btn_check_unknown_spec.clicked.connect(self.check_unknown)

        # Hover для графика спектра
        self.panel_spec.canvas.mpl_connect('motion_notify_event', self.on_hover_spec)

    def on_hover_spec(self, event):
        if event.inaxes == self.panel_spec.ax:
            x, y = event.xdata, event.ydata
            self.panel_spec.ax.set_title(f't = {x:.6f} s, Int = {y:.3f}')
            self.panel_spec.canvas.draw_idle()

    def on_mode_change(self):
        if self.radio_single.isChecked():
            self.mode = "single"
        else:
            self.mode = "mixture"
        self.update_plots()

    def generate_new_unknown(self):
        self.current_unknown = random.choice(list(ELEMENTS.keys()))
        self.unknown_name = self.current_unknown
        QtWidgets.QMessageBox.information(self,"Новый элемент",f"Сгенерирован новый неизвестный элемент: {self.current_unknown}")
        self.update_plots()

    def check_unknown(self):
        text = self.input_unknown_spec.text().strip()
        if text==self.current_unknown:
            QtWidgets.QMessageBox.information(self,"Верно","Вы угадали неизвестный элемент!")
        else:
            QtWidgets.QMessageBox.warning(self,"Неверно",f"Неправильно! Элемент изменён.")
            self.generate_new_unknown()

    def update_plots(self):
        self.panel_spec.ax.clear()
        mz_list = [ELEMENTS[e] for e,_ in self.current_mixture]
        intens_list = [p for _,p in self.current_mixture]
        t,sig = generate_tof_spectrum(mz_list,intens_list,n_points=self.npoints,noise_level=self.noise)
        self.panel_spec.ax.plot(t,sig,color='cyan')
        self.panel_spec.ax.set_xlabel('t (s)')
        self.panel_spec.ax.set_ylabel('Интенсивность')
        self.panel_spec.canvas.draw()

        self.panel_traj.ax.clear()
        radii = [magnetic_radius(ELEMENTS[e]*E_AMU,self.U,self.B) for e,_ in self.current_mixture]
        for i,r in enumerate(radii):
            circle = Circle((0,0),r,color=self.colors[i],fill=False,label=f'{self.current_mixture[i][0]}')
            self.panel_traj.ax.add_patch(circle)
        self.panel_traj.ax.set_xlim(-max(radii)*1.2,max(radii)*1.2)
        self.panel_traj.ax.set_ylim(-max(radii)*1.2,max(radii)*1.2)
        self.panel_traj.ax.set_xlabel('X (м)')
        self.panel_traj.ax.set_ylabel('Y (м)')
        self.panel_traj.ax.legend()
        self.panel_traj.canvas.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53,53,53))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25,25,25))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53,53,53))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53,53,53))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtCore.Qt.GlobalColor.white)
    app.setPalette(palette)
    w = MassLabApp()
    w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()