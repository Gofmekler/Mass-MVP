"""Пассивный график: рисует переданный PlotData и сообщает о положении курсора."""
import numpy as np

from masslab.views.qt.qt import QtCore, QtWidgets  # до matplotlib: задаёт QT_API
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from masslab.views.qt.style import PLOT_BG, TEXT

ANIMATION_FRAMES = 40
ANIMATION_INTERVAL_MS = 30


def _clip(x, y, cut):
    """Часть ломаной с x ≤ cut (x возрастает)."""
    if x.size == 0 or x[0] > cut:
        return x[:0], y[:0]
    n = int(np.searchsorted(x, cut, side="right"))
    xs, ys = x[:n], y[:n]
    if n < x.size and x[n] > x[n - 1]:
        f = (cut - x[n - 1]) / (x[n] - x[n - 1])
        xs = np.append(xs, cut)
        ys = np.append(ys, y[n - 1] + f * (y[n] - y[n - 1]))
    return xs, ys


class PlotWidget(QtWidgets.QWidget):
    hovered = QtCore.Signal(object)   # x в координатах данных или None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 2.5), dpi=90, facecolor=PLOT_BG,
                           constrained_layout=True)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(140)
        self._ax = self._fig.add_subplot(111)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._cursor = []
        self._animated = []           # [(line, dot, x, y)]
        self._anim_x_max = 0.0
        self._frame = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(ANIMATION_INTERVAL_MS)
        self._timer.timeout.connect(self._animate_step)

        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("axes_leave_event", lambda _: self.hovered.emit(None))
        self.clear()

    def _reset_axes(self):
        self._timer.stop()
        self._animated = []
        self._cursor = []
        ax = self._ax
        ax.clear()
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#555")
        ax.grid(True, color="#333", linewidth=0.6)

    def clear(self, message=""):
        self._reset_axes()
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        if message:
            self._ax.text(0.5, 0.5, message, color="#888", ha="center", va="center",
                          fontsize=12, transform=self._ax.transAxes)
        self._canvas.draw_idle()

    def render(self, plot, animate=False):
        self._reset_axes()
        ax = self._ax
        ax.set_yscale("log" if plot.y_log else "linear")
        for c in plot.curves:
            x, y = np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float)
            (line,) = ax.plot(x, y, color=c.color, linestyle=c.style, linewidth=c.width,
                              alpha=c.alpha, label=c.label or None)
            if animate and c.animate:
                (dot,) = ax.plot([], [], "o", color=c.color, markersize=5, alpha=c.alpha)
                self._animated.append((line, dot, x, y))
        for m in plot.markers:
            ax.axvline(m.x, color=m.color, linestyle=":", linewidth=1.0, alpha=0.9)
            ax.annotate(m.label, xy=(m.x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -14), textcoords="offset points", color=m.color,
                        fontsize=10, fontweight="bold")
        ax.set_title(plot.title, color=TEXT, fontsize=10)
        ax.set_xlabel(plot.x_label, color=TEXT, fontsize=9)
        ax.set_ylabel(plot.y_label, color=TEXT, fontsize=9)
        if plot.x_lim:
            ax.set_xlim(*plot.x_lim)
        if plot.y_lim:
            ax.set_ylim(*plot.y_lim)
        if any(c.label for c in plot.curves):
            legend = ax.legend(loc="lower right", fontsize=8, facecolor="#2A2A2A",
                               edgecolor="#555", labelcolor=TEXT)
            legend.set_draggable(True)
        if self._animated:
            self._anim_x_max = plot.x_lim[1] if plot.x_lim else max(
                float(x.max()) for _, _, x, _ in self._animated)
            self._frame = 0
            self._animate_step()
            self._timer.start()
        self._canvas.draw_idle()

    def _animate_step(self):
        self._frame += 1
        cut = self._anim_x_max * min(1.0, self._frame / ANIMATION_FRAMES)
        for line, dot, x, y in self._animated:
            xs, ys = _clip(x, y, cut)
            line.set_data(xs, ys)
            dot.set_data(xs[-1:], ys[-1:])
        if self._frame >= ANIMATION_FRAMES:
            self._timer.stop()
        self._canvas.draw_idle()

    def show_cursor(self, x, y, text):
        self.hide_cursor(redraw=False)
        ax = self._ax
        self._cursor.append(ax.axvline(x, color="#FFFFFF", linewidth=0.8, alpha=0.6))
        if y is not None:
            self._cursor.extend(ax.plot([x], [y], "o", color="#FFEB3B", markersize=6))
        lo, hi = ax.get_xlim()
        right = x > (lo + hi) / 2
        self._cursor.append(ax.annotate(
            text, xy=(x, 0.80), xycoords=("data", "axes fraction"),
            xytext=(-8 if right else 8, 0), textcoords="offset points",
            ha="right" if right else "left", color="#FFEB3B", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="#2A2A2A", ec="#FFEB3B", alpha=0.9)))
        self._canvas.draw_idle()

    def hide_cursor(self, redraw=True):
        for artist in self._cursor:
            artist.remove()
        self._cursor = []
        if redraw:
            self._canvas.draw_idle()

    def _on_motion(self, event):
        self.hovered.emit(event.xdata if event.inaxes is self._ax else None)
