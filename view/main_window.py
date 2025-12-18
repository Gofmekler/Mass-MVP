from PySide6 import QtWidgets, QtCore
from view.mpl_panel import MplPanel
import numpy as np


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.ctrl = controller
        self.setWindowTitle("MassLab — TOF (MVC)")
        self.resize(1200, 800)

        self.panel = MplPanel()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(500, 20000)
        self.slider.setValue(3000)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.slider)
        layout.addWidget(self.panel)
        self.setCentralWidget(central)

        self.slider.valueChanged.connect(self.update_voltage)
        self.update_plot()

    def update_voltage(self, v):
        self.ctrl.set_voltage(v)
        self.update_plot()

    def update_plot(self):
        t, s = self.ctrl.spectrum()
        ax = self.panel.ax
        ax.clear()
        ax.plot(t, s, color="cyan")
        ax.set_xlabel("t (с)")
        ax.set_ylabel("Интенсивность")
        self.panel.canvas.draw()
