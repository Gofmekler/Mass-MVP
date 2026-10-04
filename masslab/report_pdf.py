"""Сохранение отчёта студента в PDF (matplotlib, без Qt).

Файл создаётся только по явному действию студента («Сохранить отчёт в PDF»)
в выбранном им месте; сама программа ничего не сохраняет.
"""
import textwrap

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

PAGE_W, PAGE_H = 8.27, 11.69      # A4, дюймы
MARGIN = 0.75
WRAP = 100
MAX_POINTS = 3000


def write_report_pdf(path, report):
    with PdfPages(path) as pdf:
        doc = _Document(pdf, f"{report.student}, гр. {report.group}")
        doc.title(report.work_title)
        doc.fact("Студент", report.student)
        doc.fact("Группа", report.group)
        doc.fact("Дата выполнения", report.finished_at)
        doc.space(0.15)
        doc.heading("Сводка")
        doc.table(("Этап", "Время", "Попытки"),
                  [(r.title, r.duration, r.attempts) for r in report.rows], (0.6, 0.2, 0.2))
        doc.fact("Общее время", report.total)
        doc.fact("Результат", report.verdict)
        for stage in report.stages:
            doc.stage(stage)
        doc.finish()
        info = pdf.infodict()
        info["Title"] = report.work_title
        info["Author"] = report.student


class _Document:
    def __init__(self, pdf, footer):
        self._pdf = pdf
        self._footer = footer
        self._fig = None
        self._y = 0.0          # отступ от верха страницы, дюймы
        self._page = 0
        self._new_page()

    # --- страницы -----------------------------------------------------

    def _new_page(self):
        if self._fig is not None:
            self._save_page()
        self._fig = Figure(figsize=(PAGE_W, PAGE_H))
        self._page += 1
        self._y = MARGIN

    def _save_page(self):
        self._fig.text(MARGIN / PAGE_W, 0.4 / PAGE_H, self._footer, fontsize=8, color="#666")
        self._fig.text(1 - MARGIN / PAGE_W, 0.4 / PAGE_H, f"стр. {self._page}",
                       fontsize=8, color="#666", ha="right")
        self._pdf.savefig(self._fig)

    def finish(self):
        self._save_page()
        self._fig = None

    def _need(self, height):
        if self._y + height > PAGE_H - MARGIN:
            self._new_page()

    def _fy(self, y):
        return 1 - y / PAGE_H

    def space(self, height):
        self._y += height

    # --- элементы -----------------------------------------------------

    def title(self, text):
        self._need(0.6)
        self._fig.text(0.5, self._fy(self._y + 0.25), text, fontsize=15, weight="bold",
                       ha="center")
        self._y += 0.6

    def heading(self, text):
        self._need(0.45)
        self._fig.text(MARGIN / PAGE_W, self._fy(self._y + 0.25), text, fontsize=13,
                       weight="bold", color="#1B5E20")
        self._y += 0.4

    def fact(self, name, value):
        self._need(0.24)
        y = self._fy(self._y + 0.17)
        self._fig.text(MARGIN / PAGE_W, y, f"{name}:", fontsize=10, color="#444")
        self._fig.text(0.42, y, str(value), fontsize=10, weight="bold")
        self._y += 0.24

    def paragraph(self, text, size=9):
        for line in textwrap.wrap(text, WRAP, subsequent_indent="      ") or [""]:
            self._need(0.165)
            self._fig.text(MARGIN / PAGE_W, self._fy(self._y + 0.13), line, fontsize=size)
            self._y += 0.165

    def table(self, headers, rows, widths=None):
        row_h = 0.24
        rows = [list(r) for r in rows]
        while rows:
            fit = max(1, int((PAGE_H - MARGIN - self._y) / row_h) - 1)
            if fit < 2 and self._y > MARGIN:
                self._new_page()
                continue
            chunk, rows = rows[:fit], rows[fit:]
            height = row_h * (len(chunk) + 1)
            width = PAGE_W - 2 * MARGIN
            ax = self._fig.add_axes([MARGIN / PAGE_W, self._fy(self._y + height),
                                     width / PAGE_W, height / PAGE_H])
            ax.axis("off")
            table = ax.table(cellText=chunk, colLabels=list(headers), loc="upper left",
                             cellLoc="left", colLoc="left", colWidths=widths,
                             bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            for (r, _), cell in table.get_celld().items():
                cell.set_edgecolor("#BBBBBB")
                if r == 0:
                    cell.set_facecolor("#E8F5E9")
                    cell.set_text_props(weight="bold")
            self._y += height + 0.15
            if rows:
                self._new_page()

    def plot(self, data):
        height = 3.0
        self._need(height + 0.45)
        width = PAGE_W - 2 * MARGIN
        ax = self._fig.add_axes([(MARGIN + 0.45) / PAGE_W, self._fy(self._y + height - 0.1),
                                 (width - 0.55) / PAGE_W,
                                 (height - 0.6) / PAGE_H])
        for c in data.curves:
            x, y = _decimate(np.asarray(c.x, dtype=float), np.asarray(c.y, dtype=float))
            ax.plot(x, y, color="#202020", linewidth=0.7)
        for m in data.markers:
            color = _darken(m.color)
            ax.axvline(m.x, color=color, linestyle=":", linewidth=0.8)
            ax.annotate(m.label, xy=(m.x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -10), textcoords="offset points", fontsize=8,
                        color=color, weight="bold")
        if data.y_log:
            ax.set_yscale("log")
        if data.x_lim:
            ax.set_xlim(*data.x_lim)
        if data.y_lim:
            ax.set_ylim(*data.y_lim)
        ax.set_title(data.title, fontsize=10)
        ax.set_xlabel(data.x_label, fontsize=9)
        ax.set_ylabel(data.y_label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, color="#DDDDDD", linewidth=0.5)
        self._y += height + 0.45

    def stage(self, stage):
        if stage.plot is not None:
            self._new_page()
        else:
            self.space(0.2)
        self.heading(stage.title)
        for name, value in stage.facts:
            self.fact(name, value)
        if stage.plot is not None:
            self.space(0.1)
            self.plot(stage.plot)
        if stage.table_rows:
            self.table(stage.table_headers, stage.table_rows)
        if stage.notes:
            self._need(0.4)
            self.paragraph("Контрольные вопросы и ответы:", size=10)
            for note in stage.notes:
                self.paragraph(note.strip() if not note.startswith(" ") else "      " + note.strip())


def _decimate(x, y):
    """Сокращает число точек, сохраняя вершины пиков (максимум в каждом интервале)."""
    if len(x) <= MAX_POINTS:
        return x, y
    n = len(x) // (MAX_POINTS // 2)
    usable = len(x) // n * n
    xb = x[:usable].reshape(-1, n)
    yb = y[:usable].reshape(-1, n)
    idx = yb.argmax(axis=1)
    rows = np.arange(len(idx))
    return xb[rows, idx], yb[rows, idx]


def _darken(color, factor=0.65):
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)
