"""Окно методички (ITheoryView): список разделов, иллюстрация, формула, пункты."""
from masslab.events import Event
from masslab.views.qt.plot_widget import PlotWidget
from masslab.views.qt.qt import QtCore, QtWidgets
from masslab.views.qt.scheme_widget import SchemeWidget
from masslab.views.qt.style import ACCENT, STYLESHEET
from masslab.views.qt.theory_widgets import AccelerationWidget, SpectrumBuildWidget, StepsWidget
from masslab.views.qt.widgets import label

VISUAL_HEIGHT = 250


class _ResolutionPanel(QtWidgets.QWidget):
    """Интерактивный спектр: ползунок напряжения + график + вывод разрешения."""

    def __init__(self, on_voltage):
        super().__init__()
        self._on_voltage = on_voltage
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTracking(False)
        self.slider.valueChanged.connect(on_voltage)
        self._value = QtWidgets.QLabel()
        self.slider.sliderMoved.connect(lambda v: self._value.setText(f"{v} В"))
        self.plot = PlotWidget()
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color: #FFEB3B; font-size: 14px;")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Напряжение U:"))
        row.addWidget(self.slider, 1)
        row.addWidget(self._value)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.status)

    def set_range(self, minimum, maximum, step, value):
        self.slider.blockSignals(True)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(step)
        self.slider.setPageStep(step * 10)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._value.setText(f"{value} В")

    def show_result(self, plot, text):
        self._value.setText(f"{self.slider.value()} В")
        self.plot.render(plot)
        self.status.setText(text)


class TheoryView(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Методичка — времяпролётный масс-спектрометр")
        self.setStyleSheet(STYLESHEET)
        self.section_selected = Event()
        self.next_requested = Event()
        self.prev_requested = Event()
        self.resolution_voltage_changed = Event()

        self._list = QtWidgets.QListWidget()
        self._list.setFixedWidth(230)
        self._list.setWordWrap(True)
        self._list.setTextElideMode(QtCore.Qt.ElideNone)
        self._list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._list.setStyleSheet("QListWidget { font-size: 13px; } "
                                 "QListWidget::item { padding: 6px 4px; } "
                                 f"QListWidget::item:selected {{ background: #2E7D32; }}")
        self._list.currentRowChanged.connect(self._on_row)

        self._title = label("", "title", wrap=True)
        self._title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {ACCENT};")

        self._scheme = SchemeWidget()
        self._accel = AccelerationWidget()
        self._build = SpectrumBuildWidget()
        self._plot = PlotWidget()
        self._steps = StepsWidget()
        self._resolution = _ResolutionPanel(self.resolution_voltage_changed.emit)
        self._visuals = QtWidgets.QStackedWidget()
        self._visuals.setFixedHeight(VISUAL_HEIGHT)
        self._kinds = {}
        for kinds, widget in ((("scheme", "race", "charge"), self._scheme),
                              (("acceleration",), self._accel), (("build",), self._build),
                              (("calibration", "isotopes"), self._plot),
                              (("steps",), self._steps), (("resolution",), self._resolution)):
            self._visuals.addWidget(widget)
            for kind in kinds:
                self._kinds[kind] = widget

        self._caption = label("", "muted", wrap=True)
        self._caption.setAlignment(QtCore.Qt.AlignCenter)
        self._formula = label("")
        self._formula.setAlignment(QtCore.Qt.AlignCenter)
        self._formula.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #FFE082; background: #2A2A2A; "
            "border: 1px solid #8D6E00; border-radius: 6px; padding: 8px;")
        self._points = QtWidgets.QTextBrowser()
        self._points.setStyleSheet("QTextBrowser { background: #1E1E1E; border: none; "
                                   "font-size: 15px; }")

        self._prev = QtWidgets.QPushButton("← Назад")
        self._next = QtWidgets.QPushButton("Далее →")
        self._next.setObjectName("primary")
        self._counter = label("", "muted")
        close = QtWidgets.QPushButton("Закрыть")
        self._prev.clicked.connect(self.prev_requested.emit)
        self._next.clicked.connect(self.next_requested.emit)
        close.clicked.connect(self.close)
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(self._prev)
        nav.addWidget(self._counter)
        nav.addWidget(self._next)
        nav.addStretch(1)
        nav.addWidget(close)

        page = QtWidgets.QVBoxLayout()
        page.addWidget(self._title)
        page.addWidget(self._visuals)
        page.addWidget(self._caption)
        page.addWidget(self._formula)
        page.addWidget(self._points, 1)
        page.addLayout(nav)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self._list)
        layout.addLayout(page, 1)

    def _on_row(self, row):
        if row >= 0:
            self.section_selected.emit(row)

    # --- ITheoryView ---------------------------------------------------

    def open(self):
        if not self.isVisible():
            parent = self.parentWidget()
            screen = parent.screen().availableGeometry() if parent is not None else None
            width, height = 1000, 720
            if screen is not None:
                width = min(width, screen.width() - 30)
                height = min(height, screen.height() - 50)
            self.resize(width, height)
        self.show()
        self.raise_()
        self.activateWindow()

    def set_sections(self, titles):
        self._list.blockSignals(True)
        self._list.clear()
        self._list.addItems(list(titles))
        self._list.blockSignals(False)

    def set_resolution_range(self, minimum, maximum, step, value):
        self._resolution.set_range(minimum, maximum, step, value)

    def show_section(self, index, total, title, points_html, formula, caption, visual):
        self._list.blockSignals(True)
        self._list.setCurrentRow(index)
        self._list.blockSignals(False)
        self._title.setText(title)
        self._counter.setText(f"{index + 1} из {total}")
        self._caption.setText(caption)
        self._formula.setText(formula)
        self._formula.setVisible(bool(formula))
        self._points.setHtml(points_html)
        widget = self._kinds[visual.kind]
        self._visuals.setCurrentWidget(widget)
        if widget is self._scheme:
            self._scheme.show_scheme(visual.data)
        elif widget is self._plot:
            self._plot.render(visual.data)
        elif widget is not self._resolution:
            widget.set_data(visual.data)

    def set_navigation(self, can_prev, can_next):
        self._prev.setEnabled(can_prev)
        self._next.setEnabled(can_next)

    def show_resolution(self, plot, text):
        self._resolution.show_result(plot, text)
