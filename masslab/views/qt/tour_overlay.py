"""Оверлей тур-подсказок: затемнение с «окном» вокруг элемента и облачко текста."""
from masslab.views.qt.qt import QtCore, QtGui, QtWidgets, Signal
from masslab.views.qt.style import color, themed

PADDING = 6
BUBBLE_WIDTH = 360


class TourOverlay(QtWidgets.QWidget):
    next_requested = Signal()
    skip_requested = Signal()

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self._target = None
        self._hole = QtCore.QRect()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        window.installEventFilter(self)

        self._bubble = QtWidgets.QFrame(self)
        self._bubble.setObjectName("tourBubble")
        themed(self._bubble, lambda: (
            f"QFrame#tourBubble {{ background-color: {color('bubble')}; "
            f"border: 2px solid {color('accent')}; border-radius: 8px; }} "
            "QFrame#tourBubble QLabel { background: transparent; }"))
        self._bubble.setFixedWidth(BUBBLE_WIDTH)
        self._title = QtWidgets.QLabel()
        themed(self._title,
               lambda: f"font-size: 15px; font-weight: bold; color: {color('accent')};")
        self._text = QtWidgets.QLabel()
        self._text.setWordWrap(True)
        self._click = QtWidgets.QLabel("Щёлкните в любом месте, чтобы продолжить")
        themed(self._click, lambda: f"color: {color('muted')}; font-size: 12px;")
        self._footer = QtWidgets.QLabel()
        themed(self._footer, lambda: f"color: {color('muted')}; font-size: 12px;")
        skip = QtWidgets.QPushButton("Пропустить")
        skip.setCursor(QtCore.Qt.PointingHandCursor)
        skip.clicked.connect(self.skip_requested.emit)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self._footer, 1)
        bottom.addWidget(skip)
        layout = QtWidgets.QVBoxLayout(self._bubble)
        layout.addWidget(self._title)
        layout.addWidget(self._text)
        layout.addWidget(self._click)
        layout.addLayout(bottom)
        self.hide()

    def show_step(self, target, title, text, index, total):
        self._target = target
        self._title.setText(title)
        self._text.setText(text)
        self._footer.setText(f"Подсказка {index} из {total}")
        self.setGeometry(self._window.rect())
        self.show()
        self.raise_()
        self.setFocus()
        QtCore.QTimer.singleShot(0, self._relayout)

    def eventFilter(self, obj, event):
        if obj is self._window and event.type() == QtCore.QEvent.Resize and self.isVisible():
            self.setGeometry(self._window.rect())
            self._relayout()
        return False

    def _relayout(self):
        self._hole = self._target_rect()
        layout = self._bubble.layout()
        layout.activate()
        self._bubble.setFixedHeight(layout.totalHeightForWidth(BUBBLE_WIDTH))
        self._bubble.move(self._bubble_position())
        self.update()

    def _widget_rect(self, widget):
        if widget is None or not widget.isVisible():
            return QtCore.QRect()
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QtWidgets.QScrollArea):
                parent.ensureWidgetVisible(widget, 20, 20)
                break
            parent = parent.parentWidget()
        top_left = widget.mapTo(self._window, QtCore.QPoint(0, 0))
        return QtCore.QRect(top_left, widget.size())

    def _target_rect(self):
        target = self._target
        if hasattr(target, "tabs"):
            bar = target.tabs.tabBar()
            rect = bar.tabRect(target.index)
            return QtCore.QRect(bar.mapTo(self._window, rect.topLeft()), rect.size())
        if hasattr(target, "widgets"):
            rect = QtCore.QRect()
            for w in target.widgets:
                rect = rect.united(self._widget_rect(w))
            return rect
        return self._widget_rect(target)

    def _bubble_position(self):
        b = self._bubble.size()
        area = self.rect()
        hole = self._hole.adjusted(-PADDING, -PADDING, PADDING, PADDING)
        if hole.isEmpty():
            return QtCore.QPoint((area.width() - b.width()) // 2,
                                 (area.height() - b.height()) // 2)
        gap = 12
        if hole.bottom() + gap + b.height() <= area.bottom():
            pos = QtCore.QPoint(hole.left(), hole.bottom() + gap)
        elif hole.top() - gap - b.height() >= 0:
            pos = QtCore.QPoint(hole.left(), hole.top() - gap - b.height())
        elif hole.right() + gap + b.width() <= area.right():
            pos = QtCore.QPoint(hole.right() + gap, hole.top())
        else:
            pos = QtCore.QPoint(hole.left() - gap - b.width(), hole.top())
        x = max(8, min(pos.x(), area.width() - b.width() - 8))
        y = max(8, min(pos.y(), area.height() - b.height() - 8))
        return QtCore.QPoint(x, y)

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRect(QtCore.QRectF(self.rect()))
        if not self._hole.isEmpty():
            hole = QtCore.QRectF(self._hole.adjusted(-PADDING, -PADDING, PADDING, PADDING))
            inner = QtGui.QPainterPath()
            inner.addRoundedRect(hole, 6, 6)
            path = path.subtracted(inner)
            p.fillPath(path, QtGui.QColor(0, 0, 0, 170))
            p.setPen(QtGui.QPen(QtGui.QColor(color('accent')), 2))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRoundedRect(hole, 6, 6)
        else:
            p.fillPath(path, QtGui.QColor(0, 0, 0, 170))

    def mousePressEvent(self, event):
        self.next_requested.emit()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.skip_requested.emit()
        else:
            self.next_requested.emit()
