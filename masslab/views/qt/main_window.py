from PySide6 import QtCore, QtWidgets

from masslab.events import Event
from masslab.views.qt.pages import LoginView, QuizView, ReportView
from masslab.views.qt.style import ACCENT, CHIP_STYLES, STYLESHEET
from masslab.views.qt.task_views import AlloyView, DemoView, ElementView


class MainWindow(QtWidgets.QMainWindow):
    """IMainView: шапка с этапами и таймером, страницы этапов."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MassLab — времяпролётный масс-спектрометр")
        self.setStyleSheet(STYLESHEET)
        self.resize(1366, 800)
        self.tick = Event()
        self._close_text = None

        self.login = LoginView()
        self.quiz = QuizView()
        self.demo = DemoView()
        self.element = ElementView()
        self.alloy = AlloyView()
        self.report = ReportView()
        self._pages = {"login": self.login, "quiz": self.quiz, "task1": self.demo,
                       "task2": self.element, "task3": self.alloy, "report": self.report}
        self._stack = QtWidgets.QStackedWidget()
        for page in self._pages.values():
            self._stack.addWidget(page)

        header = QtWidgets.QWidget()
        header.setStyleSheet("background-color: #252525;")
        header_layout = QtWidgets.QHBoxLayout(header)
        app_title = QtWidgets.QLabel("MassLab")
        app_title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {ACCENT};")
        header_layout.addWidget(app_title)
        header_layout.addSpacing(16)
        self._chips_layout = QtWidgets.QHBoxLayout()
        self._chips = []
        header_layout.addLayout(self._chips_layout)
        header_layout.addStretch(1)
        self._student = QtWidgets.QLabel()
        self._timer = QtWidgets.QLabel()
        self._timer.setStyleSheet("font-size: 14px; font-family: monospace; color: #FFEB3B;")
        header_layout.addWidget(self._student)
        header_layout.addSpacing(16)
        header_layout.addWidget(self._timer)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._clock = QtCore.QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self.tick.emit)
        self._clock.start()

    # --- IMainView -----------------------------------------------------

    def show_page(self, page):
        self._stack.setCurrentWidget(self._pages[page])

    def set_stages(self, stages):
        if len(self._chips) != len(stages):
            for chip in self._chips:
                chip.deleteLater()
            self._chips = []
            for _ in stages:
                chip = QtWidgets.QLabel(objectName="chip")
                self._chips_layout.addWidget(chip)
                self._chips.append(chip)
        for chip, (title, state) in zip(self._chips, stages):
            chip.setText(("✓ " if state == "done" else "") + title)
            chip.setStyleSheet(CHIP_STYLES[state] + " padding: 4px 10px; border-radius: 10px;")

    def set_student(self, text):
        self._student.setText(text)

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

    def closeEvent(self, event):
        if self._close_text and not self.confirm("Выход", self._close_text):
            event.ignore()
            return
        event.accept()
