from masslab.events import Event
from masslab.views.qt.pages import LoginView, QuizView, ReportView
from masslab.views.qt.qt import QtCore, QtWidgets
from masslab.views.qt.scheme_widget import SchemeWidget
from masslab.views.qt.style import ACCENT, CHIP_STYLES, STYLESHEET
from masslab.views.qt.task_views import AlloyView, DemoView, ElementView, SandboxView
from masslab.views.qt.tour_overlay import TourOverlay

MIN_WIDTH, MIN_HEIGHT = 1000, 680


class TheoryDialog(QtWidgets.QDialog):
    """Методичка: схема прибора и краткая теория (немодальное окно)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Методичка — времяпролётный масс-спектрометр")
        self.setStyleSheet(STYLESHEET)
        self.scheme = SchemeWidget()
        self.scheme.setFixedHeight(170)
        self.text = QtWidgets.QTextBrowser()
        self.text.setStyleSheet("QTextBrowser { background-color: #242424; padding: 10px; "
                                "font-size: 13px; }")
        close = QtWidgets.QPushButton("Закрыть")
        close.clicked.connect(self.close)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.scheme)
        layout.addWidget(self.text, 1)
        layout.addWidget(close, 0, QtCore.Qt.AlignRight)


class MainWindow(QtWidgets.QMainWindow):
    """IMainView: шапка с этапами, таймером и кнопками, страницы этапов."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab — времяпролётный масс-спектрометр")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        self.resize(1366, 768)
        self.tick = Event()
        self.theory_requested = Event()
        self.help_requested = Event()
        self.tour_next = Event()
        self.tour_skip = Event()
        self._close_text = None
        self._theory = None

        self.login = LoginView()
        self.quiz = QuizView()
        self.demo = DemoView()
        self.element = ElementView()
        self.alloy = AlloyView()
        self.report = ReportView()
        self.sandbox = SandboxView()
        self._pages = {"login": self.login, "quiz": self.quiz, "task1": self.demo,
                       "task2": self.element, "task3": self.alloy, "report": self.report,
                       "sandbox": self.sandbox}
        self._stack = QtWidgets.QStackedWidget()
        for page in self._pages.values():
            self._stack.addWidget(page)

        header = QtWidgets.QWidget()
        header.setObjectName("header")
        header.setStyleSheet("QWidget#header { background-color: #252525; }")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        app_title = QtWidgets.QLabel("MassLab")
        app_title.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {ACCENT};")
        header_layout.addWidget(app_title)
        header_layout.addSpacing(10)
        self._chips_layout = QtWidgets.QHBoxLayout()
        self._chips_layout.setSpacing(4)
        self._chips = []
        header_layout.addLayout(self._chips_layout)
        header_layout.addStretch(1)
        self._timer = QtWidgets.QLabel()
        self._timer.setStyleSheet("font-size: 13px; font-family: monospace; color: #FFEB3B;")
        header_layout.addWidget(self._timer)
        header_layout.addSpacing(10)
        self._theory_button = QtWidgets.QPushButton("Методичка")
        self._theory_button.clicked.connect(self.theory_requested.emit)
        self._help_button = QtWidgets.QPushButton("?")
        self._help_button.setToolTip("Подсказки по элементам экрана")
        self._help_button.setFixedWidth(36)
        self._help_button.setStyleSheet("font-weight: bold;")
        self._help_button.clicked.connect(self.help_requested.emit)
        header_layout.addWidget(self._theory_button)
        header_layout.addWidget(self._help_button)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._overlay = TourOverlay(self)
        self._overlay.next_requested.connect(self.tour_next.emit)
        self._overlay.skip_requested.connect(self.tour_skip.emit)

        self._clock = QtCore.QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self.tick.emit)
        self._clock.start()

    def _tour_target(self, name):
        if name == "theory":
            return self._theory_button
        if name == "help":
            return self._help_button
        page = self._stack.currentWidget()
        targets = page.tour_targets() if hasattr(page, "tour_targets") else {}
        return targets.get(name)

    # --- IMainView -----------------------------------------------------

    def show_page(self, page):
        widget = self._pages[page]
        if hasattr(widget, "show_task_tab"):
            widget.show_task_tab()
        self._stack.setCurrentWidget(widget)

    def set_stages(self, stages):
        if len(self._chips) != len(stages):
            for chip in self._chips:
                chip.deleteLater()
            self._chips = []
            for _ in stages:
                chip = QtWidgets.QLabel()
                self._chips_layout.addWidget(chip)
                self._chips.append(chip)
        for chip, (title, state) in zip(self._chips, stages):
            chip.setText(("✓ " if state == "done" else "") + title)
            chip.setStyleSheet(CHIP_STYLES[state] + " padding: 3px 8px; border-radius: 9px;")

    def set_timer(self, text):
        self._timer.setText(text)

    def show_message(self, title, text):
        QtWidgets.QMessageBox.information(self, title, text)

    def confirm(self, title, text):
        answer = QtWidgets.QMessageBox.question(self, title, text)
        return answer == QtWidgets.QMessageBox.Yes

    def set_close_confirmation(self, text):
        self._close_text = text

    def close_app(self):
        self.close()

    def show_theory(self, html, scheme):
        if self._theory is None:
            self._theory = TheoryDialog(self)
            screen = self.screen().availableGeometry() if hasattr(self, "screen") else None
            width, height = 860, 640
            if screen is not None:
                width, height = min(width, screen.width() - 40), min(height, screen.height() - 60)
            self._theory.resize(width, height)
        self._theory.text.setHtml(html)
        self._theory.scheme.show_scheme(scheme)
        self._theory.show()
        self._theory.raise_()
        self._theory.activateWindow()

    def show_tour_step(self, target, title, text, index, total):
        self._overlay.show_step(self._tour_target(target), title, text, index, total)

    def hide_tour(self):
        self._overlay.hide()

    def closeEvent(self, event):
        if self._close_text and not self.confirm("Выход", self._close_text):
            event.ignore()
            return
        if self._theory is not None:
            self._theory.close()
        event.accept()
