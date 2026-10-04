"""Страницы входа, теста и итогового отчёта."""
from PySide6 import QtCore, QtWidgets

from masslab.events import Event
from masslab.views.qt.style import ACCENT, ERROR
from masslab.views.qt.widgets import QuestionWidget, ReferenceTable, scrollable


def _card(max_width=760):
    """Центрированная колонка фиксированной ширины."""
    outer = QtWidgets.QWidget()
    outer_layout = QtWidgets.QHBoxLayout(outer)
    card = QtWidgets.QWidget()
    card.setMaximumWidth(max_width)
    outer_layout.addStretch(1)
    outer_layout.addWidget(card, 10)
    outer_layout.addStretch(1)
    return outer, QtWidgets.QVBoxLayout(card)


class LoginView(QtWidgets.QWidget):
    """ILoginView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_requested = Event()
        outer, layout = _card(640)
        title = QtWidgets.QLabel("Лабораторная работа\nВремяпролётный масс-спектрометр",
                                 objectName="title", alignment=QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {ACCENT};")
        intro = QtWidgets.QLabel(wordWrap=True)
        intro.setTextFormat(QtCore.Qt.RichText)
        intro.setText(
            "<p>Порядок выполнения работы:</p><ol>"
            "<li><b>Входной тест</b> — 10 вопросов по принципу работы масс-спектрометра. "
            "Для допуска нужно ответить правильно не менее чем на 9. "
            "При пересдаче вопросы меняются.</li>"
            "<li><b>Задание 1</b> — запуск ионов разных элементов.</li>"
            "<li><b>Задание 2</b> — определение неизвестного элемента.</li>"
            "<li><b>Задание 3</b> — определение состава сплава.</li></ol>"
            "<p>Время выполнения каждого этапа фиксируется. "
            "В конце будет показан итоговый результат — покажите его преподавателю.</p>")
        self._name = QtWidgets.QLineEdit(placeholderText="Иванов Иван Иванович")
        self._group = QtWidgets.QLineEdit(placeholderText="например, ФИЗ-101")
        self._error = QtWidgets.QLabel(wordWrap=True)
        self._error.setStyleSheet(f"color: {ERROR}; font-weight: bold;")
        start = QtWidgets.QPushButton("Начать работу", objectName="primary")
        start.clicked.connect(self.start_requested.emit)
        for field in (self._name, self._group):
            field.returnPressed.connect(self.start_requested.emit)
        note = QtWidgets.QLabel(
            "Введённые данные и результаты нигде не сохраняются и удаляются "
            "при закрытии программы.", objectName="muted", wordWrap=True)

        form = QtWidgets.QFormLayout()
        form.addRow("ФИО:", self._name)
        form.addRow("Группа:", self._group)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addSpacing(12)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self._error)
        layout.addWidget(start)
        layout.addSpacing(8)
        layout.addWidget(note)
        layout.addStretch(1)
        QtWidgets.QVBoxLayout(self).addWidget(scrollable(outer))

    def student_name(self):
        return self._name.text()

    def student_group(self):
        return self._group.text()

    def show_error(self, text):
        self._error.setText(text)

    def reset(self):
        self._name.clear()
        self._group.clear()
        self._error.clear()
        self._name.setFocus()


class QuizView(QtWidgets.QWidget):
    """IQuizView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.answer_selected = Event()
        self.next_requested = Event()
        self.prev_requested = Event()
        self.finish_requested = Event()
        self.result_action_requested = Event()

        self._stack = QtWidgets.QStackedWidget()

        # Вопрос
        outer, layout = _card()
        heading = QtWidgets.QLabel("Входной тест", objectName="title")
        self._number = QtWidgets.QLabel(objectName="muted")
        self._progress = QtWidgets.QProgressBar()
        self._question = QuestionWidget(font_size=16)
        self._question.selected.connect(self.answer_selected.emit)
        self._prev = QtWidgets.QPushButton("← Назад")
        self._next = QtWidgets.QPushButton("Далее →")
        self._finish = QtWidgets.QPushButton("Завершить тест", objectName="primary")
        self._prev.clicked.connect(self.prev_requested.emit)
        self._next.clicked.connect(self.next_requested.emit)
        self._finish.clicked.connect(self.finish_requested.emit)
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(self._prev)
        nav.addWidget(self._next)
        nav.addStretch(1)
        nav.addWidget(self._finish)
        layout.addWidget(heading)
        layout.addWidget(self._number)
        layout.addWidget(self._progress)
        layout.addSpacing(16)
        layout.addWidget(self._question)
        layout.addSpacing(16)
        layout.addLayout(nav)
        layout.addStretch(1)
        self._stack.addWidget(scrollable(outer))

        # Результат
        outer, layout = _card()
        self._result_title = QtWidgets.QLabel(objectName="title")
        self._result_text = QtWidgets.QLabel(wordWrap=True)
        self._result_text.setStyleSheet("font-size: 15px;")
        self._details = QtWidgets.QLabel(wordWrap=True)
        self._details.setTextFormat(QtCore.Qt.RichText)
        self._action = QtWidgets.QPushButton(objectName="primary")
        self._action.clicked.connect(self.result_action_requested.emit)
        layout.addWidget(self._result_title)
        layout.addWidget(self._result_text)
        layout.addSpacing(8)
        layout.addWidget(self._details)
        layout.addSpacing(16)
        layout.addWidget(self._action, alignment=QtCore.Qt.AlignLeft)
        layout.addStretch(1)
        self._stack.addWidget(scrollable(outer))

        QtWidgets.QVBoxLayout(self).addWidget(self._stack)

    def show_question(self, number, total, text, options, selected):
        self._stack.setCurrentIndex(0)
        self._number.setText(f"Вопрос {number} из {total}")
        self._question.set_question(text, options, selected)

    def set_navigation(self, can_prev, can_next, can_finish):
        self._prev.setEnabled(can_prev)
        self._next.setEnabled(can_next)
        self._finish.setEnabled(can_finish)

    def set_progress(self, answered, total):
        self._progress.setRange(0, total)
        self._progress.setValue(answered)
        self._progress.setFormat(f"Отвечено: {answered} из {total}")

    def show_result(self, title, text, details, passed, action_text):
        self._stack.setCurrentIndex(1)
        self._result_title.setText(title)
        self._result_title.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {ACCENT if passed else ERROR};")
        self._result_text.setText(text)
        if details:
            head, *items = details
            self._details.setText(f"<p>{head}</p><ul>"
                                  + "".join(f"<li>{_escape(i)}</li>" for i in items) + "</ul>")
        else:
            self._details.setText("")
        self._action.setText(action_text)


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class ReportView(QtWidgets.QWidget):
    """IReportView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.new_session_requested = Event()
        self.exit_requested = Event()
        outer, layout = _card()
        title = QtWidgets.QLabel("Лабораторная работа выполнена", objectName="title")
        self._student = QtWidgets.QLabel()
        self._student.setStyleSheet("font-size: 16px;")
        self._date = QtWidgets.QLabel(objectName="muted")
        self._table = ReferenceTable(["Этап", "Время", "Попытки"])
        self._total = QtWidgets.QLabel()
        self._total.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._verdict = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self._verdict.setStyleSheet(
            f"font-size: 34px; font-weight: bold; color: {ACCENT}; "
            f"border: 3px solid {ACCENT}; border-radius: 10px; padding: 12px;")
        new = QtWidgets.QPushButton("Новая сессия")
        close = QtWidgets.QPushButton("Выход")
        new.clicked.connect(self.new_session_requested.emit)
        close.clicked.connect(self.exit_requested.emit)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(new)
        buttons.addWidget(close)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(self._student)
        layout.addWidget(self._date)
        layout.addSpacing(8)
        layout.addWidget(self._table)
        layout.addWidget(self._total)
        layout.addSpacing(12)
        layout.addWidget(self._verdict)
        layout.addSpacing(12)
        layout.addLayout(buttons)
        layout.addStretch(1)
        QtWidgets.QVBoxLayout(self).addWidget(scrollable(outer))

    def show_report(self, report):
        self._student.setText(f"{report.student}, группа {report.group}")
        self._date.setText(f"Завершено: {report.finished_at}")
        self._table.set_rows([[r.title, r.duration, r.attempts] for r in report.rows])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        table = self._table
        table.setFixedHeight(table.horizontalHeader().height() + 4
                             + sum(table.rowHeight(r) for r in range(table.rowCount())))
        self._total.setText(f"Общее время: {report.total}")
        self._verdict.setText(report.verdict)
