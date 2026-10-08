"""Анимированная схема TOF-масс-спектрометра.

Рисует источник ионов, ускоряющий промежуток, дрейфовую трубку и детектор,
по которым летят переданные ионы. Скорости пропорциональны реальным
(v = L / t), время растянуто так, что самый медленный ион летит ~4 с.
"""
from masslab.views.qt.qt import QtCore, QtGui, QtWidgets
from masslab.views.qt.style import color, data_color

ANIMATION_SECONDS = 4.0
PAUSE_SECONDS = 1.2
FRAME_MS = 30
GAP_SHARE = 0.10        # доля ширины схемы под ускоряющий промежуток


class SchemeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._ions = ()
        self._voltage = 0
        self._length = 0.0
        self._t = 0.0           # текущее модельное время, мкс
        self._t_end = 1.0
        self._pause = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._step)

    def show_scheme(self, scheme):
        self._ions = scheme.ions
        self._voltage = scheme.voltage
        self._length = scheme.length
        self._t_end = max((i.time_us for i in self._ions), default=1.0) * 1.08
        self._t = 0.0
        self._pause = 0.0
        if self._ions:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        if self._ions:
            self._timer.start()
        super().showEvent(event)

    def _step(self):
        if self._t >= self._t_end:
            self._pause += FRAME_MS / 1000
            if self._pause >= PAUSE_SECONDS:
                self._t, self._pause = 0.0, 0.0
        else:
            self._t = min(self._t_end,
                          self._t + self._t_end * FRAME_MS / 1000 / ANIMATION_SECONDS)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QtGui.QColor(color("plot_bg")))
        font = p.font()
        font.setPointSizeF(8.5)
        p.setFont(font)

        left, right = 16, w - 16
        top, bottom = 34, h - 34
        mid = (top + bottom) / 2
        source_w = 54
        gap_x0 = left + source_w
        drift_x0 = gap_x0 + (right - gap_x0 - 24) * GAP_SHARE
        detector_x = right - 14

        text = QtGui.QColor(color("draw_text"))
        # источник ионов
        p.setPen(QtGui.QPen(QtGui.QColor(color("source")), 2))
        p.setBrush(QtGui.QColor(color("source_fill")))
        p.drawRoundedRect(QtCore.QRectF(left, mid - 26, source_w - 8, 52), 6, 6)
        # сетки ускоряющего промежутка
        p.setPen(QtGui.QPen(QtGui.QColor(color("field")), 2, QtCore.Qt.DashLine))
        p.drawLine(QtCore.QPointF(gap_x0, top), QtCore.QPointF(gap_x0, bottom))
        p.drawLine(QtCore.QPointF(drift_x0, top), QtCore.QPointF(drift_x0, bottom))
        # дрейфовая трубка
        p.setPen(QtGui.QPen(QtGui.QColor(color("draw_line")), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRect(QtCore.QRectF(drift_x0, top + 6, detector_x - drift_x0, bottom - top - 12))
        # детектор
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(color("ok")))
        p.drawRect(QtCore.QRectF(detector_x, top, 8, bottom - top))

        p.setPen(text)
        p.drawText(QtCore.QRectF(left - 10, bottom + 4, source_w + 20, 26),
                   QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, "Источник")
        p.drawText(QtCore.QRectF(left, 2, drift_x0 - left + 40, 28),
                   QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   f"Ускорение\nU = {self._voltage} В")
        length = f"{self._length:.2f}".replace(".", ",")
        p.drawText(QtCore.QRectF(drift_x0 + 50, 2, detector_x - drift_x0 - 50, 28),
                   QtCore.Qt.AlignCenter, f"Дрейфовая трубка (без поля), L = {length} м")
        p.drawText(QtCore.QRectF(detector_x - 80, bottom + 4, 94, 26),
                   QtCore.Qt.AlignRight | QtCore.Qt.AlignTop, "Детектор")
        if not self._ions:
            p.drawText(QtCore.QRectF(drift_x0, mid - 12, detector_x - drift_x0, 24),
                       QtCore.Qt.AlignCenter, "Запустите ионы")
            return
        t_text = f"t = {self._t:.2f} мкс".replace(".", ",")
        p.drawText(QtCore.QRectF(drift_x0, bottom + 4, detector_x - drift_x0, 26),
                   QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, t_text)

        n = len(self._ions)
        lane = (bottom - top - 24) / max(n, 1)
        for i, ion in enumerate(self._ions):
            y = top + 12 + lane * (i + 0.5) if n > 1 else mid
            x = self._ion_x(ion.time_us, gap_x0, drift_x0, detector_x)
            ion_color = QtGui.QColor(data_color(ion.color))
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(ion_color)
            p.drawEllipse(QtCore.QPointF(x, y), 4.5, 4.5)
            if ion.label and lane >= 11:
                p.setPen(ion_color)
                p.drawText(QtCore.QPointF(x + 7, y + 4), ion.label)

    def _ion_x(self, drift_time, gap_x0, drift_x0, detector_x):
        """Ион равноускоренно проходит промежуток, затем летит равномерно."""
        drift_len = detector_x - drift_x0
        gap_len = drift_x0 - gap_x0
        v = drift_len / drift_time                 # пикселей в мкс
        t_gap = 2 * gap_len / v                    # время ускорения (средняя скорость v/2)
        t = self._t
        if t <= t_gap:
            return gap_x0 + gap_len * (t / t_gap) ** 2
        return min(detector_x - 5, drift_x0 + v * (t - t_gap))
