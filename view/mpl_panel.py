from PySide6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Компактный график
        self.fig = Figure(figsize=(5, 3), dpi=80, facecolor="#1A1A1A")
        self.canvas = FigureCanvas(self.fig)

        self.ax = self.fig.add_subplot(111)

        # Простые настройки
        self.ax.set_facecolor("#1A1A1A")
        self.ax.tick_params(colors='#E0E0E0')
        self.ax.xaxis.label.set_color('#E0E0E0')
        self.ax.yaxis.label.set_color('#E0E0E0')

        # Минимальные отступы
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.ax.set_facecolor("#1A1A1A")
        self.ax.tick_params(colors='#E0E0E0')