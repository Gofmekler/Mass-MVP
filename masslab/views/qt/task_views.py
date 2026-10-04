"""Виды трёх заданий: панель прибора слева, задание и справочник справа."""
from PySide6 import QtCore, QtWidgets

from masslab.events import Event
from masslab.views.qt.widgets import (FeedbackLabel, QuestionWidget, ReferenceTable,
                                      scrollable)
from masslab.views.qt.workspace_view import WorkspaceView

SIDE_WIDTH = 430


class _TaskView(QtWidgets.QWidget):
    def __init__(self, title, instructions, reference_headers, parent=None):
        super().__init__(parent)
        self.workspace = WorkspaceView()

        self.task_panel = QtWidgets.QWidget()
        self.task_layout = QtWidgets.QVBoxLayout(self.task_panel)
        heading = QtWidgets.QLabel(title, objectName="title")
        heading.setWordWrap(True)
        text = QtWidgets.QLabel(instructions, wordWrap=True)
        text.setTextFormat(QtCore.Qt.RichText)
        self.task_layout.addWidget(heading)
        self.task_layout.addWidget(text)

        self._reference = ReferenceTable(reference_headers)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(scrollable(self.task_panel), "Задание")
        tabs.addTab(self._reference, "Справочник элементов")
        tabs.setFixedWidth(SIDE_WIDTH)

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.workspace, 1)
        layout.addWidget(tabs)

    def _finish_panel(self, *widgets):
        for w in widgets:
            self.task_layout.addWidget(w)
        self.task_layout.addStretch(1)

    def show_reference(self, rows):
        self._reference.set_rows(rows)


class DemoView(_TaskView):
    """Задание 1 (IDemoView)."""

    def __init__(self, parent=None):
        super().__init__(
            "Задание 1. Запуск ионов",
            "Выберите ионы, задайте ускоряющее напряжение и нажмите «Запустить ионы». "
            "Понаблюдайте, как время пролёта зависит от массы иона и напряжения.<br><br>"
            "Затем ответьте на контрольные вопросы, проводя измерения на приборе. "
            "При ошибке вопросы заменяются новыми.",
            ["Символ", "Элемент", "Масса, а.е.м."])
        self.launch_requested = Event()
        self.answer_selected = Event()
        self.check_requested = Event()

        ions = QtWidgets.QGroupBox("Ионы для запуска (z = +1)")
        self._list = QtWidgets.QListWidget()
        self._list.setMinimumHeight(170)
        launch = QtWidgets.QPushButton("Запустить ионы", objectName="primary")
        launch.clicked.connect(self.launch_requested.emit)
        ions_layout = QtWidgets.QVBoxLayout(ions)
        ions_layout.addWidget(self._list)
        ions_layout.addWidget(launch)

        results = QtWidgets.QGroupBox("Результаты запуска")
        self._table = ReferenceTable(["Ион", "m, а.е.м.", "v, км/с", "t, мкс"])
        self._table.setMinimumHeight(150)
        QtWidgets.QVBoxLayout(results).addWidget(self._table)

        questions = QtWidgets.QGroupBox("Контрольные вопросы")
        self._questions_layout = QtWidgets.QVBoxLayout(questions)
        self._question_widgets = []
        check = QtWidgets.QPushButton("Проверить ответы", objectName="primary")
        check.clicked.connect(self.check_requested.emit)
        self._feedback = FeedbackLabel()

        self._finish_panel(ions, results, questions, check, self._feedback)

    def set_element_choices(self, items):
        self._list.clear()
        for symbol, label in items:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, symbol)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self._list.addItem(item)

    def set_selected_elements(self, symbols):
        for i in range(self._list.count()):
            item = self._list.item(i)
            checked = item.data(QtCore.Qt.UserRole) in symbols
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)

    def selected_elements(self):
        return [self._list.item(i).data(QtCore.Qt.UserRole) for i in range(self._list.count())
                if self._list.item(i).checkState() == QtCore.Qt.Checked]

    def show_flight_table(self, rows):
        self._table.set_rows(rows)

    def show_questions(self, questions):
        for w in self._question_widgets:
            w.deleteLater()
        self._question_widgets = []
        for i, (text, options) in enumerate(questions):
            w = QuestionWidget()
            w.set_question(f"{i + 1}. {text}", options)
            w.selected.connect(lambda option, q=i: self.answer_selected.emit(q, option))
            self._questions_layout.addWidget(w)
            self._question_widgets.append(w)

    def show_feedback(self, text, ok):
        self._feedback.show_feedback(text, ok)


class _ChoiceTaskView(_TaskView):
    def __init__(self, title, instructions, headers, question, parent=None):
        super().__init__(title, instructions, headers, parent)
        self.check_requested = Event()
        self._question_text = question
        self._choice = QuestionWidget()
        self._check = QtWidgets.QPushButton("Проверить", objectName="primary")
        self._check.clicked.connect(self.check_requested.emit)
        self._feedback = FeedbackLabel()

    def set_options(self, labels):
        self._choice.set_question(self._question_text, labels)

    def selected_option(self):
        return self._choice.selected_index()

    def show_feedback(self, text, ok):
        self._feedback.show_feedback(text, ok)


class ElementView(_ChoiceTaskView):
    """Задание 2 (IElementView)."""

    def __init__(self, parent=None):
        super().__init__(
            "Задание 2. Неизвестный элемент",
            "В спектре — калибровочная смесь (He, Ar, Xe, подписаны) и неизвестный "
            "однозарядный ион (красная пунктирная траектория).<br><br>"
            "1. Измерьте время пролёта неизвестного иона (наведите курсор на пик).<br>"
            "2. Вычислите его массу: m = 2eU·t² / L² или по калибранту: "
            "m = m<sub>к</sub>·(t / t<sub>к</sub>)².<br>"
            "3. Определите элемент по справочнику.<br><br>"
            "Допуск по массе ±2 %. При ошибке выдаётся новый образец.",
            ["Символ", "Элемент", "Масса, а.е.м."],
            "Какой это элемент?")
        group = QtWidgets.QGroupBox("Ответ")
        form = QtWidgets.QVBoxLayout(group)
        form.addWidget(QtWidgets.QLabel("Масса неизвестного иона, а.е.м.:"))
        self._mass = QtWidgets.QLineEdit(placeholderText="например, 22,9")
        self._mass.returnPressed.connect(self.check_requested.emit)
        form.addWidget(self._mass)
        form.addSpacing(8)
        form.addWidget(self._choice)
        self._finish_panel(group, self._check, self._feedback)

    def mass_text(self):
        return self._mass.text()

    def clear_inputs(self):
        self._mass.clear()


class AlloyView(_ChoiceTaskView):
    """Задание 3 (IAlloyView)."""

    def __init__(self, parent=None):
        super().__init__(
            "Задание 3. Состав сплава",
            "Образец сплава испарён и ионизирован. В отличие от заданий 1–2, здесь "
            "учтён реальный изотопный состав элементов, поэтому каждый элемент даёт "
            "несколько пиков.<br><br>"
            "Прибор откалиброван: при наведении на пик показывается m/z. "
            "Определите, какие элементы присутствуют и в каком соотношении, "
            "сравнив пики с изотопами в справочнике.<br><br>"
            "Слабые пики лучше видны в логарифмической шкале. "
            "При ошибке выдаётся новый образец.",
            ["Символ", "Элемент", "Масса", "Изотопы (содержание)"],
            "Какой это сплав?")
        group = QtWidgets.QGroupBox("Ответ")
        QtWidgets.QVBoxLayout(group).addWidget(self._choice)
        self._finish_panel(group, self._check, self._feedback)

    def clear_inputs(self):
        self._choice.set_question(self._question_text, [])
