"""Страницы входа, теста и итогового результата."""
from masslab.events import Event
from masslab.views.qt.qt import QtCore, QtGui, QtWidgets, exec_app
from masslab.views.qt.style import ACCENT, ERROR, feedback_style
from masslab.views.qt.widgets import QuestionWidget, ReferenceTable, label, scrollable

try:
    import qrcode
except ImportError:  # QR-код необязателен: без библиотеки покажем только ссылку
    qrcode = None


def _card(max_width=760):
    """Центрированная колонка ограниченной ширины."""
    outer = QtWidgets.QWidget()
    outer_layout = QtWidgets.QHBoxLayout(outer)
    card = QtWidgets.QWidget()
    card.setMaximumWidth(max_width)
    outer_layout.addStretch(1)
    outer_layout.addWidget(card, 10)
    outer_layout.addStretch(1)
    return outer, QtWidgets.QVBoxLayout(card)


def _button(text, primary=False):
    button = QtWidgets.QPushButton(text)
    if primary:
        button.setObjectName("primary")
    return button


class LoginView(QtWidgets.QWidget):
    """ILoginView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_requested = Event()
        self.secret_entered = Event()
        self.teacher_demo_requested = Event()
        self.sandbox_requested = Event()
        outer, layout = _card(640)
        title = label("Лабораторная работа\nВремяпролётный масс-спектрометр")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {ACCENT};")
        intro = label("", wrap=True)
        intro.setTextFormat(QtCore.Qt.RichText)
        intro.setText(
            "<p>Порядок выполнения работы:</p><ol>"
            "<li><b>Входной тест</b> — 10 вопросов по принципу работы масс-спектрометра. "
            "Для допуска нужно ответить правильно не менее чем на 9. "
            "При пересдаче вопросы меняются.</li>"
            "<li><b>Задание 1</b> — запуск ионов разных элементов.</li>"
            "<li><b>Задание 2</b> — определение неизвестного элемента.</li>"
            "<li><b>Задание 3</b> — определение состава сплава.</li></ol>"
            "<p>Работа выполняется в паре, входит один студент. Теория и формулы — "
            "кнопка «Методичка» вверху. В конце сохраните отчёт в PDF.</p>")
        self._name = QtWidgets.QLineEdit()
        self._name.setPlaceholderText("Иванов Иван Иванович")
        self._group = QtWidgets.QLineEdit()
        self._group.setPlaceholderText("например, ФИЗ-301")
        self._error = label("", wrap=True)
        self._error.setStyleSheet(f"color: {ERROR}; font-weight: bold;")
        start = _button("Начать работу", primary=True)
        start.clicked.connect(self.start_requested.emit)
        for field in (self._name, self._group):
            field.returnPressed.connect(self.start_requested.emit)
        note = label("Введённые данные и результаты нигде не сохраняются и удаляются "
                     "при закрытии программы.", "muted", wrap=True)

        form = QtWidgets.QFormLayout()
        form.addRow("ФИО:", self._name)
        form.addRow("Группа:", self._group)

        layout.addStretch(1)
        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self._error)
        layout.addWidget(start)
        layout.addSpacing(6)
        layout.addWidget(note)

        self._teacher = QtWidgets.QGroupBox("Режим преподавателя")
        teacher_layout = QtWidgets.QVBoxLayout(self._teacher)
        teacher_layout.addWidget(label(
            "Работа с ответами: все этапы как у студента, но верные ответы подсвечены "
            "зелёным. Песочница: прибор со всеми настройками без теста и заданий.",
            "muted", wrap=True))
        demo = _button("Пройти работу с правильными ответами", primary=True)
        demo.clicked.connect(self.teacher_demo_requested.emit)
        sandbox = _button("Свободная работа с прибором (песочница)")
        sandbox.clicked.connect(self.sandbox_requested.emit)
        teacher_layout.addWidget(demo)
        teacher_layout.addWidget(sandbox)
        self._teacher.hide()
        layout.addWidget(self._teacher)
        layout.addStretch(1)

        # Скрытое поле без подписи: пароль открывает режим песочницы
        self._secret = QtWidgets.QLineEdit()
        self._secret.setEchoMode(QtWidgets.QLineEdit.Password)
        self._secret.setFixedWidth(90)
        self._secret.setFrame(False)
        self._secret.setStyleSheet("background: transparent; border: none; color: #2A2A2A;")
        self._secret.returnPressed.connect(
            lambda: self.secret_entered.emit(self._secret.text()))
        corner = QtWidgets.QHBoxLayout()
        corner.addStretch(1)
        corner.addWidget(self._secret)

        page = QtWidgets.QVBoxLayout(self)
        page.addWidget(scrollable(outer), 1)
        page.addLayout(corner)

    def student_name(self):
        return self._name.text()

    def student_group(self):
        return self._group.text()

    def show_error(self, text):
        self._error.setText(text)

    def clear_secret(self):
        self._secret.clear()

    def show_teacher_panel(self, visible):
        self._teacher.setVisible(visible)

    def reset(self):
        self._name.clear()
        self._group.clear()
        self._error.clear()
        self._secret.clear()
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

        outer, layout = _card()
        heading = label("Входной тест", "title")
        self._number = label("", "muted")
        self._progress = QtWidgets.QProgressBar()
        self._question = QuestionWidget(font_size=15)
        self._question.selected.connect(self.answer_selected.emit)
        self._prev = _button("← Назад")
        self._next = _button("Далее →")
        self._finish = _button("Завершить тест", primary=True)
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
        layout.addSpacing(12)
        layout.addWidget(self._question)
        layout.addSpacing(12)
        layout.addLayout(nav)
        layout.addStretch(1)
        self._stack.addWidget(scrollable(outer))

        outer, layout = _card()
        self._result_title = label("", "title")
        self._result_text = label("", wrap=True)
        self._result_text.setStyleSheet("font-size: 14px;")
        self._details = label("", wrap=True)
        self._details.setTextFormat(QtCore.Qt.RichText)
        self._action = _button("", primary=True)
        self._action.clicked.connect(self.result_action_requested.emit)
        layout.addWidget(self._result_title)
        layout.addWidget(self._result_text)
        layout.addSpacing(8)
        layout.addWidget(self._details)
        layout.addSpacing(12)
        layout.addWidget(self._action, 0, QtCore.Qt.AlignLeft)
        layout.addStretch(1)
        self._stack.addWidget(scrollable(outer))

        QtWidgets.QVBoxLayout(self).addWidget(self._stack)

    def show_question(self, number, total, text, options, selected, correct=None):
        self._stack.setCurrentIndex(0)
        self._number.setText(f"Вопрос {number} из {total}")
        self._question.set_question(text, options, selected, correct=correct)

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
            f"font-size: 20px; font-weight: bold; color: {ACCENT if passed else ERROR};")
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


class QrWidget(QtWidgets.QWidget):
    """QR-код ссылки (рисуется по матрице библиотеки qrcode)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._matrix = []
        self.setFixedSize(150, 150)

    def set_url(self, url):
        self._matrix = []
        if qrcode is not None:
            qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
            qr.add_data(url)
            qr.make(fit=True)
            self._matrix = qr.get_matrix()
        self.setVisible(bool(self._matrix))
        self.update()

    def paintEvent(self, event):
        if not self._matrix:
            return
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("white"))
        n = len(self._matrix)
        cell = min(self.width(), self.height()) / n
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("black"))
        for r, row in enumerate(self._matrix):
            for c, dark in enumerate(row):
                if dark:
                    p.drawRect(QtCore.QRectF(c * cell, r * cell, cell + 0.5, cell + 0.5))


class ReportView(QtWidgets.QWidget):
    """IReportView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.new_session_requested = Event()
        self.exit_requested = Event()
        self.export_requested = Event()
        outer, layout = _card(820)
        title = label("Лабораторная работа выполнена", "title")
        self._student = label("")
        self._student.setStyleSheet("font-size: 15px;")
        self._date = label("", "muted")
        self._table = ReferenceTable(["Этап", "Время", "Попытки"])
        self._total = label("")
        self._total.setStyleSheet("font-size: 15px; font-weight: bold;")
        self._verdict = label("")
        self._verdict.setAlignment(QtCore.Qt.AlignCenter)
        self._verdict.setStyleSheet(
            f"font-size: 30px; font-weight: bold; color: {ACCENT}; "
            f"border: 3px solid {ACCENT}; border-radius: 10px; padding: 8px;")
        export = _button("Сохранить отчёт в PDF", primary=True)
        export.clicked.connect(self.export_requested.emit)
        self._export_status = label("", wrap=True)

        self._qr = QrWidget()
        self._qr_text = label("", wrap=True)
        self._qr_link = label("", "muted", wrap=True)
        self._qr_link.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        feedback = QtWidgets.QHBoxLayout()
        feedback.addWidget(self._qr, 0, QtCore.Qt.AlignTop)
        texts = QtWidgets.QVBoxLayout()
        texts.addWidget(self._qr_text)
        texts.addWidget(self._qr_link)
        texts.addStretch(1)
        feedback.addLayout(texts, 1)

        new = _button("Новая сессия")
        close = _button("Выход")
        new.clicked.connect(self.new_session_requested.emit)
        close.clicked.connect(self.exit_requested.emit)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(export)
        buttons.addStretch(1)
        buttons.addWidget(new)
        buttons.addWidget(close)

        layout.addWidget(title)
        layout.addWidget(self._student)
        layout.addWidget(self._date)
        layout.addSpacing(6)
        layout.addWidget(self._table)
        layout.addWidget(self._total)
        layout.addSpacing(6)
        layout.addWidget(self._verdict)
        layout.addSpacing(8)
        layout.addLayout(buttons)
        layout.addWidget(self._export_status)
        layout.addSpacing(10)
        layout.addLayout(feedback)
        layout.addStretch(1)
        QtWidgets.QVBoxLayout(self).addWidget(scrollable(outer))

    def show_report(self, report):
        self._student.setText(f"{report.student}, группа {report.group}")
        self._date.setText(f"Завершено: {report.finished_at}")
        self._table.set_rows([[r.title, r.duration, r.attempts] for r in report.rows])
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table = self._table
        table.setFixedHeight(table.horizontalHeader().sizeHint().height() + 4
                             + sum(table.rowHeight(r) for r in range(table.rowCount())))
        self._total.setText(f"Общее время: {report.total}")
        self._verdict.setText(report.verdict)

    def show_feedback_qr(self, url, text):
        self._qr.set_url(url)
        self._qr_text.setText(text)
        self._qr_link.setText(url)

    def ask_save_path(self, default_name):
        folder = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.DesktopLocation)
        dialog = QtWidgets.QFileDialog(self, "Сохранить отчёт", f"{folder}/{default_name}",
                                       "Документ PDF (*.pdf)")
        # Встроенный диалог Qt — переведён на русский на любой системе
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("pdf")
        dialog.setLabelText(QtWidgets.QFileDialog.Accept, "Сохранить")
        dialog.setLabelText(QtWidgets.QFileDialog.Reject, "Отмена")
        dialog.setLabelText(QtWidgets.QFileDialog.FileName, "Имя файла:")
        dialog.setLabelText(QtWidgets.QFileDialog.FileType, "Тип файла:")
        dialog.setLabelText(QtWidgets.QFileDialog.LookIn, "Папка:")
        accepted = exec_app(dialog)
        files = dialog.selectedFiles()
        return files[0] if accepted and files else None

    def show_export_result(self, text, ok):
        self._export_status.setText(text)
        self._export_status.setStyleSheet(feedback_style(ok))
