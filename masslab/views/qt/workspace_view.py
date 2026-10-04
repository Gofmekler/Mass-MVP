from PySide6 import QtCore, QtWidgets

from masslab.events import Event
from masslab.views.qt.plot_widget import PlotWidget


class WorkspaceView(QtWidgets.QWidget):
    """Панель прибора (IWorkspaceView)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.voltage_changed = Event()
        self.log_scale_toggled = Event()
        self.spectrum_hovered = Event()
        self._last_voltage = None

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setTracking(False)
        self._spin = QtWidgets.QSpinBox(suffix=" В")
        self._spin.setKeyboardTracking(False)
        self._spin.setMinimumWidth(110)
        self._log = QtWidgets.QCheckBox("Логарифмическая шкала интенсивности")
        self._params = QtWidgets.QLabel(objectName="muted")
        self._readout = QtWidgets.QLabel()
        self._readout.setStyleSheet("color: #FFEB3B; font-size: 14px; padding: 2px 4px;")
        self._spectrum = PlotWidget()
        self._trajectories = PlotWidget()

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Ускоряющее напряжение U:"))
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._spin)
        controls.addSpacing(16)
        controls.addWidget(self._log)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self._spectrum, 1)
        top_layout.addWidget(self._readout)
        splitter.addWidget(top)
        splitter.addWidget(self._trajectories)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self._params)
        layout.addWidget(splitter, 1)

        self._slider.sliderMoved.connect(self._show_dragged_value)
        self._slider.valueChanged.connect(self._emit_voltage)
        self._spin.valueChanged.connect(self._emit_voltage)
        self._log.toggled.connect(self.log_scale_toggled.emit)
        self._spectrum.hovered.connect(self.spectrum_hovered.emit)

    def _show_dragged_value(self, value):
        self._spin.blockSignals(True)
        self._spin.setValue(value)
        self._spin.blockSignals(False)

    def _emit_voltage(self, value):
        if value != self._last_voltage:
            self._last_voltage = value
            self.voltage_changed.emit(value)

    # --- IWorkspaceView ------------------------------------------------

    def set_voltage_range(self, minimum, maximum, step):
        for w in (self._slider, self._spin):
            w.blockSignals(True)
            w.setRange(minimum, maximum)
            w.setSingleStep(step)
            w.blockSignals(False)
        self._slider.setPageStep(step * 10)

    def set_voltage(self, value):
        self._last_voltage = value
        for w in (self._slider, self._spin):
            w.blockSignals(True)
            w.setValue(value)
            w.blockSignals(False)

    def set_log_scale(self, enabled):
        self._log.blockSignals(True)
        self._log.setChecked(enabled)
        self._log.blockSignals(False)

    def set_parameters(self, text):
        self._params.setText(text)

    def show_spectrum(self, plot):
        self._spectrum.render(plot)

    def show_trajectories(self, plot, animate):
        self._trajectories.render(plot, animate)

    def clear_plots(self, message):
        self._spectrum.clear(message)
        self._trajectories.clear()

    def show_cursor(self, x, y, text):
        self._spectrum.show_cursor(x, y, text)

    def hide_cursor(self):
        self._spectrum.hide_cursor()

    def set_readout(self, text):
        self._readout.setText(text)
