from masslab.events import Event
from masslab.views.qt.pages import LoginView, QuizView, ReportView
from masslab.views.qt.qt import QtCore, QtWidgets, exec_app
from masslab.views.qt import style
from masslab.views.qt.style import color, themed
from masslab.views.qt.task_views import AlloyView, DemoView, ElementView, SandboxView
from masslab.views.qt.theory_view import TheoryView
from masslab.views.qt.tour_overlay import TourOverlay

MIN_WIDTH, MIN_HEIGHT = 1000, 680


class MainWindow(QtWidgets.QMainWindow):
    """IMainView: шапка с этапами, таймером и кнопками, страницы этапов."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab — времяпролётный масс-спектрометр")
        self.setStyleSheet(style.stylesheet())
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)
        self.resize(1366, 768)
        self.tick = Event()
        self.theory_requested = Event()
        self.help_requested = Event()
        self.tour_next = Event()
        self.tour_skip = Event()
        self.theme_toggled = Event()
        self._close_text = None

        self.login = LoginView()
        self.quiz = QuizView()
        self.demo = DemoView()
        self.element = ElementView()
        self.alloy = AlloyView()
        self.report = ReportView()
        self.sandbox = SandboxView()
        self.theory = TheoryView(self)
        self._pages = {"login": self.login, "quiz": self.quiz, "task1": self.demo,
                       "task2": self.element, "task3": self.alloy, "report": self.report,
                       "sandbox": self.sandbox}
        self._stack = QtWidgets.QStackedWidget()
        for page in self._pages.values():
            self._stack.addWidget(page)

        header = QtWidgets.QWidget()
        header.setObjectName("header")
        themed(header, lambda: f"QWidget#header {{ background-color: {color('header')}; }} "
                               "QWidget#header QLabel { background: transparent; }")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        app_title = QtWidgets.QLabel("MassLab")
        themed(app_title,
               lambda: f"font-size: 17px; font-weight: bold; color: {color('accent')};")
        header_layout.addWidget(app_title)
        header_layout.addSpacing(10)
        self._chips_layout = QtWidgets.QHBoxLayout()
        self._chips_layout.setSpacing(4)
        self._chips = []
        header_layout.addLayout(self._chips_layout)
        self._timer = QtWidgets.QLabel()
        themed(self._timer, lambda: "font-size: 13px; font-family: monospace; "
                                    f"color: {color('highlight')};")
        # при нехватке места сжимается таймер, а не названия этапов
        self._timer.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self._timer.setMinimumWidth(120)
        self._timer.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        header_layout.addWidget(self._timer, 1)
        header_layout.addSpacing(10)
        self._theory_button = QtWidgets.QPushButton("Методичка")
        self._theory_button.clicked.connect(self.theory_requested.emit)
        self._help_button = QtWidgets.QPushButton("?")
        self._help_button.setToolTip("Подсказки по элементам экрана")
        self._help_button.setFixedWidth(36)
        self._help_button.setStyleSheet("font-weight: bold;")
        self._help_button.clicked.connect(self.help_requested.emit)
        self._theme_button = QtWidgets.QPushButton("◐")
        self._theme_button.setFixedWidth(36)
        self._theme_button.clicked.connect(self.theme_toggled.emit)
        self._update_theme_button()
        header_layout.addWidget(self._theme_button)
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
            themed(chip, lambda state=state:
                   style.chip_style(state) + " padding: 3px 8px; border-radius: 9px;")

    def set_theme(self, name):
        style.apply_theme(self, name)
        self._update_theme_button()

    def _update_theme_button(self):
        light = style.theme() == "light"
        self._theme_button.setToolTip("Тёмная тема" if light else "Светлая тема")

    def set_timer(self, text):
        self._timer.setText(text)
        self._timer.setToolTip(text)

    def show_message(self, title, text):
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, title, text,
                                    QtWidgets.QMessageBox.NoButton, self)
        box.addButton("Продолжить", QtWidgets.QMessageBox.AcceptRole)
        exec_app(box)

    def confirm(self, title, text):
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, title, text,
                                    QtWidgets.QMessageBox.NoButton, self)
        yes = box.addButton("Да", QtWidgets.QMessageBox.YesRole)
        no = box.addButton("Нет", QtWidgets.QMessageBox.NoRole)
        box.setDefaultButton(no)
        box.setEscapeButton(no)
        exec_app(box)
        return box.clickedButton() is yes

    def set_close_confirmation(self, text):
        self._close_text = text

    def close_app(self):
        self.close()

    def show_tour_step(self, target, title, text, index, total):
        self._overlay.show_step(self._tour_target(target), title, text, index, total)

    def hide_tour(self):
        self._overlay.hide()

    def closeEvent(self, event):
        if self._close_text and not self.confirm("Выход", self._close_text):
            event.ignore()
            return
        self.theory.close()
        event.accept()
