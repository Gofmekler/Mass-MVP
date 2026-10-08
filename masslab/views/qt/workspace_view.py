from masslab.events import Event
from masslab.views.qt.plot_widget import PlotWidget
from masslab.views.qt.qt import QtCore, QtWidgets
from masslab.views.qt.style import color, themed
from masslab.views.qt.scheme_widget import SchemeWidget
from masslab.views.qt.widgets import label
from masslab.views.viewmodels import SchemeData

LENGTH_SCALE = 10   # ползунок длины — в десятых долях метра


class WorkspaceView(QtWidgets.QWidget):
    """Панель прибора (IWorkspaceView)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.voltage_changed = Event()
        self.length_changed = Event()
        self.log_scale_toggled = Event()
        self.spectrum_hovered = Event()
        self._last_voltage = None
        self._last_length = None

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setTracking(False)
        self._spin = QtWidgets.QSpinBox()
        self._spin.setSuffix(" В")
        self._spin.setKeyboardTracking(False)
        self._spin.setMinimumWidth(100)

        self._length_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._length_slider.setTracking(False)
        self._length_spin = QtWidgets.QDoubleSpinBox()
        self._length_spin.setSuffix(" м")
        self._length_spin.setDecimals(1)
        self._length_spin.setKeyboardTracking(False)
        self._length_spin.setMinimumWidth(100)

        self._log = QtWidgets.QCheckBox("Логарифмическая шкала")
        self._params = label("", "muted")
        self._readout = QtWidgets.QLabel()
        themed(self._readout,
               lambda: f"color: {color('highlight')}; font-size: 13px; padding: 2px 4px;")
        self._spectrum = PlotWidget()
        self._trajectories = PlotWidget()
        self._scheme = SchemeWidget()

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setVerticalSpacing(2)
        self._voltage_label = QtWidgets.QLabel("Напряжение U:")
        self._length_label = QtWidgets.QLabel("Длина трубки L:")
        controls.addWidget(self._voltage_label, 0, 0)
        controls.addWidget(self._slider, 0, 1)
        controls.addWidget(self._spin, 0, 2)
        controls.addWidget(self._length_label, 1, 0)
        controls.addWidget(self._length_slider, 1, 1)
        controls.addWidget(self._length_spin, 1, 2)
        controls.setColumnStretch(1, 1)
        info = QtWidgets.QHBoxLayout()
        info.addWidget(self._params, 1)
        info.addWidget(self._log)

        self.bottom_tabs = QtWidgets.QTabWidget()
        self.bottom_tabs.addTab(self._trajectories, "Траектории")
        self.bottom_tabs.addTab(self._scheme, "Схема прибора")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(self._spectrum, 1)
        top_layout.addWidget(self._readout)
        splitter.addWidget(top)
        splitter.addWidget(self.bottom_tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._controls = QtWidgets.QWidget()
        self._controls.setLayout(controls)
        layout.addWidget(self._controls)
        layout.addLayout(info)
        layout.addWidget(splitter, 1)

        self._slider.sliderMoved.connect(lambda v: self._quiet(self._spin, v))
        self._slider.valueChanged.connect(self._emit_voltage)
        self._spin.valueChanged.connect(self._emit_voltage)
        self._length_slider.sliderMoved.connect(
            lambda v: self._quiet(self._length_spin, v / LENGTH_SCALE))
        self._length_slider.valueChanged.connect(lambda v: self._emit_length(v / LENGTH_SCALE))
        self._length_spin.valueChanged.connect(self._emit_length)
        self._log.toggled.connect(self.log_scale_toggled.emit)
        self._spectrum.hovered.connect(self.spectrum_hovered.emit)

    def tour_targets(self):
        return {"voltage": _Row(self, 0),
                "length": _Row(self, 1),
                "parameters": self._params, "log": self._log,
                "spectrum": self._spectrum, "readout": self._readout,
                "trajectories": self.bottom_tabs}

    @staticmethod
    def _quiet(widget, value):
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

    def _emit_voltage(self, value):
        if value != self._last_voltage:
            self._last_voltage = value
            self.voltage_changed.emit(value)

    def _emit_length(self, value):
        value = round(value, 2)
        if value != self._last_length:
            self._last_length = value
            self.length_changed.emit(value)

    def row_widgets(self, row):
        if row == 0:
            return self._voltage_label, self._slider, self._spin
        return self._length_label, self._length_slider, self._length_spin

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
        self._quiet(self._slider, value)
        self._quiet(self._spin, value)

    def set_length_range(self, minimum, maximum, step):
        self._length_slider.blockSignals(True)
        self._length_slider.setRange(int(round(minimum * LENGTH_SCALE)),
                                     int(round(maximum * LENGTH_SCALE)))
        self._length_slider.blockSignals(False)
        self._length_spin.blockSignals(True)
        self._length_spin.setRange(minimum, maximum)
        self._length_spin.setSingleStep(step)
        self._length_spin.blockSignals(False)

    def set_length(self, value):
        self._last_length = round(value, 2)
        self._quiet(self._length_slider, int(round(value * LENGTH_SCALE)))
        self._quiet(self._length_spin, value)

    def set_length_visible(self, visible):
        for w in self.row_widgets(1):
            w.setVisible(visible)

    def set_log_scale(self, enabled):
        self._quiet_check(enabled)

    def _quiet_check(self, enabled):
        self._log.blockSignals(True)
        self._log.setChecked(enabled)
        self._log.blockSignals(False)

    def set_parameters(self, text):
        self._params.setText(text)

    def show_spectrum(self, plot):
        self._spectrum.render(plot)

    def show_trajectories(self, plot, animate):
        self._trajectories.render(plot, animate)

    def show_scheme(self, scheme):
        self._scheme.show_scheme(scheme)

    def clear_plots(self, message):
        self._spectrum.clear(message)
        self._trajectories.clear()
        self._scheme.show_scheme(SchemeData())

    def show_cursor(self, x, y, text):
        self._spectrum.show_cursor(x, y, text)

    def hide_cursor(self):
        self._spectrum.hide_cursor()

    def set_readout(self, text):
        self._readout.setText(text)


class _Row:
    """Цель тура, объединяющая подпись, ползунок и поле одной строки."""

    def __init__(self, workspace, row):
        self.widgets = workspace.row_widgets(row)
