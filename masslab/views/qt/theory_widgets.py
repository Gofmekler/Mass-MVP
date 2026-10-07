"""Анимации и рисунки методички (QPainter, без логики предметной области)."""
from masslab.views.qt.qt import QtCore, QtGui, QtWidgets

FRAME_MS = 30
BG = QtGui.QColor("#1A1A1A")
TEXT = QtGui.QColor("#BDBDBD")


class _Animated(QtWidgets.QWidget):
    """Базовый виджет с циклической анимацией: phase меняется от 0 до 1."""
    DURATION = 4.0      # с на цикл
    PAUSE = 1.0         # с паузы в конце цикла

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._t = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._step)

    @property
    def phase(self):
        return min(1.0, self._t / self.DURATION)

    def restart(self):
        self._t = 0.0
        if self.isVisible():
            self._timer.start()
        self.update()

    def _step(self):
        self._t += FRAME_MS / 1000
        if self._t > self.DURATION + self.PAUSE:
            self._t = 0.0
        self.update()

    def showEvent(self, event):
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _painter(self):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        font = p.font()
        font.setPointSizeF(9.5)
        p.setFont(font)
        return p


class AccelerationWidget(_Animated):
    """Два иона в ускоряющем промежутке: энергия одинакова, скорость — разная."""
    DURATION = 3.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ions = ()

    def set_data(self, ions):
        self._ions = ions
        self.restart()

    def paintEvent(self, event):
        p = self._painter()
        w, h = self.width(), self.height()
        gap_x0, gap_x1 = 40, 40 + w * 0.22
        end_x = w * 0.62
        bars_x = w * 0.68
        top, bottom = 40, h - 34

        # пластины ускоряющего промежутка и поле
        p.setPen(QtGui.QPen(QtGui.QColor("#4FC3F7"), 3))
        p.drawLine(QtCore.QPointF(gap_x0, top), QtCore.QPointF(gap_x0, bottom))
        p.drawLine(QtCore.QPointF(gap_x1, top), QtCore.QPointF(gap_x1, bottom))
        p.setPen(QtGui.QPen(QtGui.QColor("#37474F"), 1.5))
        for k in range(4):
            y = top + (bottom - top) * (k + 0.5) / 4
            p.drawLine(QtCore.QPointF(gap_x0 + 10, y), QtCore.QPointF(gap_x1 - 14, y))
            p.drawLine(QtCore.QPointF(gap_x1 - 14, y), QtCore.QPointF(gap_x1 - 22, y - 4))
            p.drawLine(QtCore.QPointF(gap_x1 - 14, y), QtCore.QPointF(gap_x1 - 22, y + 4))
        p.setPen(TEXT)
        p.drawText(QtCore.QRectF(gap_x0 - 30, 4, 60, 30), QtCore.Qt.AlignCenter, "+U")
        p.drawText(QtCore.QRectF(gap_x1 - 30, 4, 60, 30), QtCore.Qt.AlignCenter, "0 В")
        p.drawText(QtCore.QRectF(gap_x0, bottom + 4, w - gap_x0 - 10, 24),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop,
                   "ускоряющее поле  →  дальше поля нет, v = const")

        n = len(self._ions) or 1
        lane = (bottom - top) / n
        for i, ion in enumerate(self._ions):
            y = top + lane * (i + 0.5)
            x = self._ion_x(ion.speed, gap_x0, gap_x1, end_x)
            color = QtGui.QColor(ion.color)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(QtCore.QPointF(x, y), 7, 7)
            p.setPen(color)
            p.drawText(QtCore.QPointF(x - 12, y - 12), ion.label)
            self._bars(p, ion, bars_x, y, w - bars_x - 16, color)

    def _ion_x(self, speed, x0, x1, end):
        """Равноускоренно в промежутке, затем равномерно; скорость ∝ speed."""
        gap = x1 - x0
        v = speed * (end - x0) / (self.DURATION * 0.55)   # пикс/с у самого быстрого
        t_gap = 2 * gap / v
        t = self._t
        if t <= t_gap:
            return x0 + gap * (t / t_gap) ** 2
        return min(end, x1 + v * (t - t_gap))

    def _bars(self, p, ion, x, y, width, color):
        """Полоски «энергия» (одинаковая) и «скорость» (разная) после разгона."""
        grown = min(1.0, self._t / (self.DURATION * 0.45))
        for k, (name, value) in enumerate((("энергия qU", 1.0), ("скорость v", ion.speed))):
            yy = y - 14 + k * 18
            p.setPen(TEXT)
            p.drawText(QtCore.QRectF(x, yy - 7, 84, 14), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                       name)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QColor("#333333"))
            p.drawRect(QtCore.QRectF(x + 86, yy - 5, width - 86, 10))
            p.setBrush(color if k else QtGui.QColor("#FFB74D"))
            p.drawRect(QtCore.QRectF(x + 86, yy - 5, (width - 86) * value * grown, 10))


class SpectrumBuildWidget(_Animated):
    """Ионы прилетают на детектор, и из отсчётов «растут» пики спектра."""
    DURATION = 5.0
    PAUSE = 1.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._peaks = ()

    def set_data(self, peaks):
        self._peaks = peaks
        self.restart()

    def paintEvent(self, event):
        p = self._painter()
        if not self._peaks:
            return
        w, h = self.width(), self.height()
        left, right = 50, w - 20
        axis_y = h - 34
        top = 36
        t_max = max(pk.time_us for pk in self._peaks) * 1.15

        def x_of(t_us):
            return left + (right - left) * t_us / t_max

        # оси
        p.setPen(QtGui.QPen(QtGui.QColor("#616161"), 1.5))
        p.drawLine(QtCore.QPointF(left, axis_y), QtCore.QPointF(right, axis_y))
        p.drawLine(QtCore.QPointF(left, axis_y), QtCore.QPointF(left, top - 10))
        p.setPen(TEXT)
        p.drawText(QtCore.QRectF(left, axis_y + 6, right - left, 20), QtCore.Qt.AlignCenter,
                   "время пролёта t  →  (чем тяжелее ион, тем правее пик)")
        p.save()
        p.translate(18, (top + axis_y) / 2)
        p.rotate(-90)
        p.drawText(QtCore.QRectF(-70, -10, 140, 20), QtCore.Qt.AlignCenter, "число ионов")
        p.restore()

        # пики растут по мере прихода ионов; падающие «ионы» над ними
        cycle = self.phase
        for pk in self._peaks:
            color = QtGui.QColor(pk.color)
            x = x_of(pk.time_us)
            arrival = pk.time_us / t_max          # ионы лёгких видов приходят раньше
            grown = max(0.0, min(1.0, (cycle - arrival * 0.6) / 0.4))
            height = (axis_y - top - 20) * pk.share * grown
            path = QtGui.QPainterPath()
            sigma = 7
            path.moveTo(x - 4 * sigma, axis_y)
            for k in range(-20, 21):
                dx = k * sigma / 5
                path.lineTo(x + dx, axis_y - height * 2.718 ** (-(dx / sigma) ** 2 / 2))
            path.lineTo(x + 4 * sigma, axis_y)
            fill = QtGui.QColor(color)
            fill.setAlpha(110)
            p.setPen(QtGui.QPen(color, 2))
            p.setBrush(fill)
            p.drawPath(path)
            if 0 < grown < 1:
                for k in range(3):
                    fy = top + ((self._t * 180 + k * 40) % max(1.0, axis_y - height - top))
                    p.setPen(QtCore.Qt.NoPen)
                    p.setBrush(color)
                    p.drawEllipse(QtCore.QPointF(x + (k - 1) * 5, fy), 3.5, 3.5)
            if grown > 0.2:
                p.setPen(color)
                p.drawText(QtCore.QRectF(x - 30, axis_y - height - 22, 60, 18),
                           QtCore.Qt.AlignCenter, pk.label)


class StepsWidget(QtWidgets.QWidget):
    """Этапы лабораторной работы — карточки со стрелками."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._steps = ()

    def set_data(self, steps):
        self._steps = steps
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), BG)
        n = len(self._steps)
        if not n:
            return
        font = p.font()
        font.setPointSizeF(10.5)
        font.setBold(True)
        p.setFont(font)
        gap = 26
        box_w = (self.width() - 40 - gap * (n - 1)) / n
        box_h = 64
        y = (self.height() - box_h) / 2
        colors = ("#F9A825", "#4FC3F7", "#81C784", "#BA68C8", "#4DB6AC")
        for i, title in enumerate(self._steps):
            x = 20 + i * (box_w + gap)
            color = QtGui.QColor(colors[i % len(colors)])
            p.setPen(QtGui.QPen(color, 2))
            fill = QtGui.QColor(color)
            fill.setAlpha(40)
            p.setBrush(fill)
            p.drawRoundedRect(QtCore.QRectF(x, y, box_w, box_h), 10, 10)
            p.setPen(QtGui.QColor("#EEEEEE"))
            p.drawText(QtCore.QRectF(x + 4, y, box_w - 8, box_h),
                       QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, title)
            if i < n - 1:
                ax = x + box_w + 4
                p.setPen(QtGui.QPen(QtGui.QColor("#9E9E9E"), 2))
                p.drawLine(QtCore.QPointF(ax, y + box_h / 2), QtCore.QPointF(ax + gap - 8, y + box_h / 2))
                p.drawLine(QtCore.QPointF(ax + gap - 8, y + box_h / 2),
                           QtCore.QPointF(ax + gap - 14, y + box_h / 2 - 5))
                p.drawLine(QtCore.QPointF(ax + gap - 8, y + box_h / 2),
                           QtCore.QPointF(ax + gap - 14, y + box_h / 2 + 5))
