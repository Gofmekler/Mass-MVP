import sys, random
import numpy as np
from PySide6 import QtWidgets, QtCore
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model.tof_physics import TOFPhysics
from model.spectrum import generate_tof_spectrum


ELEMENTS = {
    "H": 1.008, "He": 4.0026, "C": 12.011, "N": 14.007,
    "O": 15.999, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "Fe": 55.845, "Cu": 63.546
}

CALIBRANTS = ["H", "He", "Si", "Fe"]


class MplPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 4), dpi=110, tight_layout=True, facecolor="#222222")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)


class MassLabApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab v10 — TOF масс-спектрометр")
        self.resize(1400, 850)

        self.U = 3000.0
        self.L = 1.2
        self.tof = TOFPhysics(self.U, self.L)

        self.det_sigma = 3e-8
        self.noise = 0.02
        self.tmin, self.tmax = 0.0, 2e-5
        self.npoints = 4000

        self.current_unknown = None
        self.traj_annotation = None

        self._init_ui()
        self.generate_new_unknown()
        self.update_plots()

    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        controls = QtWidgets.QVBoxLayout()
        layout.addLayout(controls, 0)

        self.slider_U = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_U.setRange(500, 20000)
        self.slider_U.setValue(int(self.U))
        self.lbl_U = QtWidgets.QLabel(f"U = {self.U:.0f} V")
        controls.addWidget(self.lbl_U)
        controls.addWidget(self.slider_U)

        self.input_guess = QtWidgets.QLineEdit()
        self.btn_check = QtWidgets.QPushButton("Проверить элемент")
        self.btn_new = QtWidgets.QPushButton("Новый неизвестный")

        controls.addWidget(QtWidgets.QLabel("Введите элемент:"))
        controls.addWidget(self.input_guess)
        controls.addWidget(self.btn_check)
        controls.addWidget(self.btn_new)
        controls.addStretch()

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.panel_spec = MplPanel()
        self.panel_traj = MplPanel()

        self.tabs.addTab(self.panel_spec, "TOF-спектр")
        self.tabs.addTab(self.panel_traj, "Траектория")

        self.slider_U.valueChanged.connect(self.on_slider)
        self.btn_new.clicked.connect(self.generate_new_unknown)
        self.btn_check.clicked.connect(self.check_unknown)

        self.panel_traj.canvas.mpl_connect(
            "motion_notify_event", self.on_traj_hover
        )

    def on_slider(self):
        self.U = self.slider_U.value()
        self.lbl_U.setText(f"U = {self.U:.0f} V")
        self.tof.U = self.U
        self.update_plots()

    def generate_new_unknown(self):
        self.current_unknown = random.choice(list(ELEMENTS.keys()))
        self.input_guess.clear()
        self.update_plots()

    def check_unknown(self):
        guess = self.input_guess.text().strip()
        if guess == self.current_unknown:
            QtWidgets.QMessageBox.information(self, "Верно", "Элемент определён правильно")
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Неверно, попробуйте снова")

    def update_plots(self):
        # ===== Spectrum =====
        ax = self.panel_spec.ax
        ax.clear()
        ax.set_facecolor("#222222")
        ax.tick_params(colors="white")

        m_u = ELEMENTS[self.current_unknown]
        t, sig = generate_tof_spectrum(
            [m_u], [1.0], self.tof,
            t_min=self.tmin, t_max=self.tmax,
            n_points=self.npoints,
            detector_sigma=self.det_sigma,
            noise_level=self.noise
        )

        ax.plot(t, sig, color="cyan")
        ax.set_xlabel("t (с)", color="white")
        ax.set_ylabel("Интенсивность", color="white")

        # ===== Trajectory =====
        ax2 = self.panel_traj.ax
        ax2.clear()
        ax2.set_facecolor("#222222")
        ax2.tick_params(colors="white")

        for name in CALIBRANTS:
            m = ELEMENTS[name]
            v = self.tof.velocity(m)
            tf = self.tof.flight_time(m)
            tt = np.linspace(0, tf, 300)
            x = v * tt
            ax2.plot(x, tt, label=name, linewidth=2)

        # Unknown
        m = ELEMENTS[self.current_unknown]
        v = self.tof.velocity(m)
        tf = self.tof.flight_time(m)
        tt = np.linspace(0, tf, 300)
        x = v * tt
        ax2.plot(x, tt, "--", color="red", linewidth=3, label="Unknown")

        ax2.set_xlabel("x (м)", color="white")
        ax2.set_ylabel("t (с)", color="white")
        ax2.legend(facecolor="#333333", labelcolor="white")

        self.panel_spec.canvas.draw()
        self.panel_traj.canvas.draw()

    def on_traj_hover(self, event):
        ax = self.panel_traj.ax
        if event.inaxes != ax:
            if self.traj_annotation:
                self.traj_annotation.set_visible(False)
                self.panel_traj.canvas.draw_idle()
            return

        best = None
        best_dist = float("inf")

        for line in ax.lines:
            x = line.get_xdata()
            y = line.get_ydata()
            d = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
            i = np.argmin(d)
            if d[i] < best_dist and d[i] < 0.02:
                best_dist = d[i]
                best = (line, x[i], y[i])

        if best is None:
            if self.traj_annotation:
                self.traj_annotation.set_visible(False)
                self.panel_traj.canvas.draw_idle()
            return

        line, x_pt, t_pt = best
        label = line.get_label()

        if label == "Unknown":
            name = "Unknown"
            mass = ELEMENTS[self.current_unknown]
        else:
            name = label
            mass = ELEMENTS[label]

        text = f"{name}\nm = {mass:.2f} u\nt = {t_pt:.2e} s"

        if self.traj_annotation is None:
            self.traj_annotation = ax.annotate(
                text, (x_pt, t_pt), xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="#333333"),
                color="white"
            )
        else:
            self.traj_annotation.xy = (x_pt, t_pt)
            self.traj_annotation.set_text(text)
            self.traj_annotation.set_visible(True)

        self.panel_traj.canvas.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MassLabApp()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
