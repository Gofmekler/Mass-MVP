"""Общие пассивные виджеты."""
from PySide6 import QtCore, QtWidgets

from masslab.views.qt.style import feedback_style


class _OptionLabel(QtWidgets.QLabel):
    """Переносимый текст варианта; щелчок по нему выбирает радиокнопку."""

    def __init__(self, text, button):
        super().__init__(text, wordWrap=True)
        self._button = button
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        self._button.click()


class QuestionWidget(QtWidgets.QWidget):
    """Текст вопроса и варианты ответа (радиокнопки с переносом строк)."""
    selected = QtCore.Signal(int)

    def __init__(self, parent=None, font_size=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._label = QtWidgets.QLabel(wordWrap=True)
        self._label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        if font_size:
            self._label.setStyleSheet(f"font-size: {font_size}px; font-weight: bold;")
        self._font_size = font_size
        self._layout.addWidget(self._label)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.idClicked.connect(self.selected.emit)
        self._buttons = []
        self._rows = []

    def set_question(self, text, options, selected=None):
        self._label.setText(text)
        self._label.setVisible(bool(text))
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
            row_layout.setContentsMargins(0, 3, 0, 3)
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
        if rows and len(rows[0]) != self.columnCount():
            self.setColumnCount(len(rows[0]))
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.setItem(r, c, QtWidgets.QTableWidgetItem(value))
        self.resizeColumnsToContents()
        self.resizeRowsToContents()


class FeedbackLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent, wordWrap=True)

    def show_feedback(self, text, ok):
        self.setText(text)
        self.setStyleSheet(feedback_style(ok))


def scrollable(widget):
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area
