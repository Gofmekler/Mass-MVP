"""Общие пассивные виджеты."""
from masslab.views.qt.qt import QtCore, QtWidgets, Signal
from masslab.views.qt.style import feedback_style


class _OptionLabel(QtWidgets.QLabel):
    """Переносимый текст варианта; щелчок по нему выбирает радиокнопку."""

    def __init__(self, text, button):
        super().__init__(text)
        self.setWordWrap(True)
        self._button = button
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if self._button.isEnabled():
            self._button.click()


class QuestionWidget(QtWidgets.QWidget):
    """Текст вопроса и варианты ответа (радиокнопки с переносом строк)."""
    selected = Signal(int)

    def __init__(self, parent=None, font_size=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._label = QtWidgets.QLabel()
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._font_size = font_size
        self._layout.addWidget(self._label)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.idClicked.connect(self.selected.emit)
        self._buttons = []
        self._rows = []
        self._label_style()

    def _label_style(self, color=None):
        style = "font-weight: bold;"
        if self._font_size:
            style += f" font-size: {self._font_size}px;"
        if color:
            style += f" color: {color};"
        self._label.setStyleSheet(style)

    def set_question(self, text, options, selected=None, locked=False):
        self._label.setText(("✓ " if locked else "") + text)
        self._label.setVisible(bool(text))
        self._label_style("#81C784" if locked else None)
        for button in self._buttons:
            self._group.removeButton(button)
        for row in self._rows:
            self._layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._buttons, self._rows = [], []
        for i, option in enumerate(options):
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            button = QtWidgets.QRadioButton()
            label = _OptionLabel(option, button)
            if self._font_size:
                label.setStyleSheet(f"font-size: {self._font_size - 2}px;")
            row_layout.addWidget(button, 0, QtCore.Qt.AlignTop)
            row_layout.addWidget(label, 1)
            self._group.addButton(button, i)
            self._layout.addWidget(row)
            self._buttons.append(button)
            self._rows.append(row)
            button.setEnabled(not locked)
            label.setEnabled(not locked)
        self._group.setExclusive(False)
        for i, button in enumerate(self._buttons):
            button.setChecked(i == selected)
        self._group.setExclusive(True)

    def selected_index(self):
        index = self._group.checkedId()
        return None if index < 0 else index


class ReferenceTable(QtWidgets.QTableWidget):
    def __init__(self, headers, parent=None):
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.setWordWrap(True)

    def set_rows(self, rows):
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.setItem(r, c, QtWidgets.QTableWidgetItem(value))
        if self.horizontalHeader().sectionResizeMode(0) == QtWidgets.QHeaderView.Interactive:
            self.resizeColumnsToContents()
        self.resizeRowsToContents()


class FeedbackLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)

    def show_feedback(self, text, ok):
        self.setText(text)
        self.setVisible(bool(text))
        self.setStyleSheet(feedback_style(ok))


class HintBox(QtWidgets.QLabel):
    """Жёлтая плашка с подсказкой после нескольких ошибок."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setStyleSheet("background-color: #3D3520; color: #FFE082; border: 1px solid "
                           "#8D6E00; border-radius: 5px; padding: 8px;")
        self.hide()

    def show_hint(self, text):
        self.setText(text or "")
        self.setVisible(bool(text))


def scrollable(widget):
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


def label(text, object_name=None, wrap=False):
    widget = QtWidgets.QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(wrap)
    return widget
