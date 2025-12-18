from PySide6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.fig = Figure(facecolor="#222222")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
