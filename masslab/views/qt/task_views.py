"""Виды заданий и песочницы: панель прибора слева, задание и справочник справа."""
from masslab.events import Event
from masslab.views.qt.qt import QtCore, QtWidgets
from masslab.views.qt.style import color, themed
from masslab.views.qt.widgets import (FeedbackLabel, HintBox, QuestionWidget, ReferenceTable,
                                      label, scrollable)
from masslab.views.qt.workspace_view import WorkspaceView

SIDE_MIN, SIDE_MAX = 335, 470


class TabTarget:
    """Цель тура — вкладка панели вкладок."""

    def __init__(self, tabs, index):
        self.tabs = tabs
        self.index = index


class _TaskView(QtWidgets.QWidget):
    def __init__(self, title, instructions, reference_headers, parent=None):
        super().__init__(parent)
        self.workspace = WorkspaceView()

        self.task_panel = QtWidgets.QWidget()
        self.task_layout = QtWidgets.QVBoxLayout(self.task_panel)
        self.task_layout.setContentsMargins(6, 6, 6, 6)
        heading = label(title, "title", wrap=True)
        text = label(instructions, wrap=True)
        text.setTextFormat(QtCore.Qt.RichText)
        self.task_layout.addWidget(heading)
        self.task_layout.addWidget(text)

        self._reference = ReferenceTable(reference_headers)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(scrollable(self.task_panel), "Задание")
        self.tabs.addTab(self._reference, "Справочник элементов")
        self.tabs.setMinimumWidth(SIDE_MIN)
        self.tabs.setMaximumWidth(SIDE_MAX)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.workspace)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)

    def _finish_panel(self, *widgets):
        for w in widgets:
            self.task_layout.addWidget(w)
        self.task_layout.addStretch(1)

    def tour_targets(self):
        targets = dict(self.workspace.tour_targets())
        targets["reference"] = TabTarget(self.tabs, 1)
        return targets

    def show_task_tab(self):
        self.tabs.setCurrentIndex(0)

    def show_reference(self, rows):
        self._reference.set_rows(rows)


class _LauncherMixin:
    """Панель выбора и запуска ионов + таблица результатов (IIonLauncherView)."""

    def _build_launcher(self):
        self.launch_requested = Event()
        self.double_charge_toggled = Event()
        self._ions_box = QtWidgets.QGroupBox("Ионы для запуска")
        self._list = QtWidgets.QListWidget()
        self._list.setMinimumHeight(130)
        self._double = QtWidgets.QCheckBox("Двухзарядные ионы (X²⁺)")
        self._double.toggled.connect(self.double_charge_toggled.emit)
        self._launch = QtWidgets.QPushButton("Запустить ионы")
        self._launch.setObjectName("primary")
        self._launch.clicked.connect(self.launch_requested.emit)
        ions_layout = QtWidgets.QVBoxLayout(self._ions_box)
        ions_layout.addWidget(self._list)
        ions_layout.addWidget(self._double)
        ions_layout.addWidget(self._launch)
        self._launch_feedback = FeedbackLabel()       # сразу под кнопкой — чтобы было видно
        ions_layout.addWidget(self._launch_feedback)

        self._results_box = QtWidgets.QGroupBox("Результаты запуска")
        self._table = ReferenceTable(["Ион", "m/z", "v, км/с", "t, мкс", "R"])
        self._table.setMinimumHeight(120)
        self._table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setStyleSheet("QTableWidget, QHeaderView::section { font-size: 12px; }")
        QtWidgets.QVBoxLayout(self._results_box).addWidget(self._table)
        self._feedback = FeedbackLabel()

    def _launcher_targets(self):
        return {"ions": self._list, "double": self._double, "launch": self._launch,
                "table": self._results_box}

    def set_element_choices(self, items):
        self._list.clear()
        for symbol, text in items:
            item = QtWidgets.QListWidgetItem(text)
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

    def double_charge_enabled(self):
        return self._double.isChecked()

    def set_double_charge(self, enabled):
        self._double.blockSignals(True)
        self._double.setChecked(enabled)
        self._double.blockSignals(False)

    def show_launch_feedback(self, text, ok):
        self._launch_feedback.show_feedback(text, ok)

    def show_flight_table(self, rows):
        self._table.set_rows(rows)

    def show_feedback(self, text, ok):
        self._feedback.show_feedback(text, ok)


class DemoView(_LauncherMixin, _TaskView):
    """Задание 1 (IDemoView)."""

    def __init__(self, parent=None):
        super().__init__(
            "Задание 1. Запуск ионов",
            "Выберите ионы, задайте напряжение U и длину трубки L, нажмите «Запустить "
            "ионы». Понаблюдайте, как время пролёта и разрешение зависят от массы, "
            "заряда, U и L.<br><br>Затем ответьте на контрольные вопросы, проводя "
            "измерения на приборе.",
            ["Символ", "Элемент", "Масса, а.е.м."])
        self._build_launcher()
        self.answer_selected = Event()
        self.check_requested = Event()

        self._questions_box = QtWidgets.QGroupBox("Контрольные вопросы")
        self._questions_layout = QtWidgets.QVBoxLayout(self._questions_box)
        self._question_widgets = []
        self._check = QtWidgets.QPushButton("Проверить ответы")
        self._check.setObjectName("primary")
        self._check.clicked.connect(self.check_requested.emit)
        self._hint = HintBox()

        self._finish_panel(self._ions_box, self._results_box, self._questions_box,
                           self._check, self._feedback, self._hint)

    def tour_targets(self):
        targets = super().tour_targets()
        targets.update(self._launcher_targets())
        targets["questions"] = self._questions_box
        return targets

    def show_questions(self, questions):
        if len(self._question_widgets) != len(questions):
            for w in self._question_widgets:
                w.deleteLater()
            self._question_widgets = []
            for i in range(len(questions)):
                w = QuestionWidget()
                w.selected.connect(lambda option, q=i: self.answer_selected.emit(q, option))
                self._questions_layout.addWidget(w)
                self._question_widgets.append(w)
        for i, (w, q) in enumerate(zip(self._question_widgets, questions)):
            w.set_question(f"{i + 1}. {q.text}", q.options, q.selected, q.locked, q.correct)

    def show_hint(self, text):
        self._hint.show_hint(text)


class SandboxView(_LauncherMixin, _TaskView):
    """Песочница преподавателя (ISandboxView)."""

    def __init__(self, parent=None):
        super().__init__(
            "Режим песочницы",
            "Свободная работа с прибором без теста и заданий — например, для "
            "демонстрации на лекции. Доступны все настройки.",
            ["Символ", "Элемент", "Масса", "Изотопы (содержание)"])
        self._build_launcher()
        self.gas_toggled = Event()
        self.exit_requested = Event()
        self._gas = QtWidgets.QCheckBox("Остаточный газ")
        self._gas.setToolTip("Пики H₂O⁺, N₂⁺, O₂⁺ от неидеального вакуума")
        self._gas.toggled.connect(self.gas_toggled.emit)
        exit_button = QtWidgets.QPushButton("Выйти из песочницы")
        exit_button.clicked.connect(self.exit_requested.emit)
        self._finish_panel(self._ions_box, self._gas, self._results_box, self._feedback,
                           exit_button)

    def tour_targets(self):
        targets = super().tour_targets()
        targets.update(self._launcher_targets())
        return targets

    def set_gas(self, enabled):
        self._gas.blockSignals(True)
        self._gas.setChecked(enabled)
        self._gas.blockSignals(False)


class _ChoiceTaskView(_TaskView):
    def __init__(self, title, instructions, headers, question, parent=None):
        super().__init__(title, instructions, headers, parent)
        self.check_requested = Event()
        self._question_text = question
        self._choice = QuestionWidget()
        self._check = QtWidgets.QPushButton("Проверить")
        self._check.setObjectName("primary")
        self._check.clicked.connect(self.check_requested.emit)
        self._feedback = FeedbackLabel()
        self._hint = HintBox()
        self._answer = HintBox(style=HintBox.GREEN)

    def tour_targets(self):
        targets = super().tour_targets()
        targets.update({"options": self._choice, "check": self._check})
        return targets

    def set_options(self, labels, correct=None):
        self._choice.set_question(self._question_text, labels, correct=correct)

    def show_answer(self, text):
        self._answer.show_hint(text)

    def selected_option(self):
        return self._choice.selected_index()

    def show_feedback(self, text, ok):
        self._feedback.show_feedback(text, ok)

    def show_hint(self, text):
        self._hint.show_hint(text)


class ElementView(_ChoiceTaskView):
    """Задание 2 (IElementView)."""

    def __init__(self, parent=None):
        super().__init__(
            "Задание 2. Неизвестный элемент",
            "В спектре — калибровочная смесь (He⁺, Ar⁺, Xe⁺, подписаны) и неизвестный "
            "однозарядный ион (красная метка «?» на спектре).<br><br>"
            "1. Измерьте время пролёта неизвестного иона.<br>"
            "2. Вычислите массу: m = 2eU·t² / L² или m = m<sub>к</sub>·(t / t<sub>к</sub>)².<br>"
            "3. Определите элемент по справочнику.<br><br>"
            "При ошибке программа подскажет, на каком шаге она допущена, и выдаст новый образец.",
            ["Символ", "Элемент", "Масса, а.е.м."],
            "3. Какой это элемент?")
        group = QtWidgets.QGroupBox("Ответ")
        form = QtWidgets.QVBoxLayout(group)
        form.addWidget(label("1. Время пролёта неизвестного иона, мкс:", wrap=True))
        self._time = QtWidgets.QLineEdit()
        self._time.setPlaceholderText("например, 5,90")
        form.addWidget(self._time)
        form.addWidget(label("2. Масса неизвестного иона, а.е.м.:", wrap=True))
        self._mass = QtWidgets.QLineEdit()
        self._mass.setPlaceholderText("например, 22,9")
        self._mass.returnPressed.connect(self.check_requested.emit)
        form.addWidget(self._mass)
        form.addSpacing(6)
        form.addWidget(self._choice)
        self._finish_panel(self._answer, group, self._check, self._feedback, self._hint)

    def tour_targets(self):
        targets = super().tour_targets()
        targets.update({"time_input": self._time, "mass_input": self._mass})
        return targets

    def time_text(self):
        return self._time.text()

    def mass_text(self):
        return self._mass.text()

    def clear_inputs(self):
        self._time.clear()
        self._mass.clear()


class AlloyView(_ChoiceTaskView):
    """Задание 3 (IAlloyView): шаг 1 — элементы образца, шаг 2 — сплав."""

    def __init__(self, parent=None):
        super().__init__(
            "Задание 3. Состав сплава",
            "Образец сплава испарён и ионизирован. Здесь учтён реальный изотопный состав, "
            "поэтому каждый элемент даёт несколько пиков. При наведении на пик "
            "показывается m/z.<br><br>"
            "<b>Шаг 1.</b> Сравните пики с изотопами в справочнике и отметьте элементы, "
            "которые есть в образце. Малые добавки (меньше 2 %) отмечать необязательно.<br>"
            "<b>Шаг 2.</b> Выберите сплав. При ошибке на шаге 2 выдаётся новый образец.",
            ["Символ", "Элемент", "Масса", "Изотопы (содержание)"],
            "Какой это сплав?")
        self.elements_check_requested = Event()

        self._step1 = QtWidgets.QGroupBox("Шаг 1. Элементы в образце")
        step1_layout = QtWidgets.QVBoxLayout(self._step1)
        self._elements_grid = QtWidgets.QGridLayout()
        self._elements_grid.setHorizontalSpacing(6)
        step1_layout.addLayout(self._elements_grid)
        self._element_boxes = {}
        self._elements_check = QtWidgets.QPushButton("Проверить элементы")
        self._elements_check.setObjectName("primary")
        self._elements_check.clicked.connect(self.elements_check_requested.emit)
        self._elements_feedback = FeedbackLabel()
        step1_layout.addWidget(self._elements_check)
        step1_layout.addWidget(self._elements_feedback)

        self._step2 = QtWidgets.QGroupBox("Шаг 2. Какой это сплав?")
        step2_layout = QtWidgets.QVBoxLayout(self._step2)
        self._choice.set_question("", [])
        step2_layout.addWidget(self._choice)
        step2_layout.addWidget(self._check)
        self._question_text = ""
        self._finish_panel(self._answer, self._step1, self._step2, self._feedback, self._hint)

    def tour_targets(self):
        targets = super().tour_targets()
        targets.update({"elements": self._step1, "options": self._step2})
        return targets

    def set_element_choices(self, items, correct=None):
        for box in self._element_boxes.values():
            self._elements_grid.removeWidget(box)
            box.deleteLater()
        self._element_boxes = {}
        correct = set(correct or ())
        for i, (symbol, text) in enumerate(items):
            box = QtWidgets.QCheckBox(text + (" ✓" if symbol in correct else ""))
            if symbol in correct:
                themed(box, lambda: f"color: {color('ok')}; font-weight: bold;")
            self._elements_grid.addWidget(box, i // 2, i % 2)
            self._element_boxes[symbol] = box

    def selected_elements(self):
        return [s for s, box in self._element_boxes.items() if box.isChecked()]

    def show_elements_feedback(self, text, ok):
        self._elements_feedback.show_feedback(text, ok)

    def set_step(self, step):
        first = step == 1
        for box in self._element_boxes.values():
            box.setEnabled(first)
        self._elements_check.setEnabled(first)
        self._step2.setEnabled(not first)
        self._step2.setTitle("Шаг 2. Какой это сплав?" if not first
                             else "Шаг 2 (откроется после шага 1)")
