"""Оформление: тёмная и светлая палитры, таблица стилей, перекраска виджетов.

Тема хранится только в памяти (на время сеанса). Виджеты с собственными
цветами регистрируют построитель стиля через themed(); при смене темы
apply_theme() пересобирает все стили, перерисовывает графики и рисунки.
"""
import colorsys

from matplotlib.colors import to_hex, to_rgb

THEMES = {
    "dark": {
        "bg": "#1E1E1E", "panel": "#2A2A2A", "plot_bg": "#1A1A1A",
        "text": "#E0E0E0", "muted": "#9E9E9E", "accent": "#4CAF50", "error": "#FF6B6B",
        "border": "#444", "border_strong": "#555", "input": "#333",
        "button": "#3A3A3A", "button_hover": "#454545", "disabled": "#666",
        "primary": "#2E7D32", "primary_hover": "#388E3C", "primary_text": "#E0E0E0",
        "primary_disabled": "#2F3B30", "primary_disabled_text": "#777",
        "list": "#262626", "list_alt": "#2C2C2C", "header": "#252525",
        "highlight": "#FFEB3B", "grid": "#333", "ok": "#81C784",
        "hint_bg": "#3D3520", "hint_text": "#FFE082", "hint_border": "#8D6E00",
        "answer_bg": "#1B3320", "answer_text": "#A5D6A7", "answer_border": "#2E7D32",
        "bubble": "#263238", "chip_pending": "#333", "chip_pending_text": "#888",
        "chip_current": "#F9A825",
        "draw_text": "#BDBDBD", "draw_line": "#616161", "draw_dim": "#37474F",
        "draw_empty": "#333333", "draw_label": "#EEEEEE",
        "source": "#FFB74D", "source_fill": "#3E2F1C", "field": "#4FC3F7",
    },
    "light": {
        "bg": "#F3F4F6", "panel": "#FFFFFF", "plot_bg": "#FFFFFF",
        "text": "#212121", "muted": "#616161", "accent": "#2E7D32", "error": "#C62828",
        "border": "#C9CDD2", "border_strong": "#A3A8AE", "input": "#FFFFFF",
        "button": "#E6E8EB", "button_hover": "#D8DBDF", "disabled": "#9E9E9E",
        "primary": "#2E7D32", "primary_hover": "#388E3C", "primary_text": "#FFFFFF",
        "primary_disabled": "#C5D6C6", "primary_disabled_text": "#F5F5F5",
        "list": "#FFFFFF", "list_alt": "#F1F3F5", "header": "#E2E5E9",
        "highlight": "#A15C00", "grid": "#E3E3E3", "ok": "#2E7D32",
        "hint_bg": "#FFF8E1", "hint_text": "#5D4300", "hint_border": "#FFB300",
        "answer_bg": "#E8F5E9", "answer_text": "#1B5E20", "answer_border": "#66BB6A",
        "bubble": "#FFFFFF", "chip_pending": "#D5D8DC", "chip_pending_text": "#616161",
        "chip_current": "#F9A825",
        "draw_text": "#424242", "draw_line": "#9E9E9E", "draw_dim": "#B0BEC5",
        "draw_empty": "#E0E0E0", "draw_label": "#212121",
        "source": "#E65100", "source_fill": "#FFE0B2", "field": "#0277BD",
    },
}
DEFAULT_THEME = "dark"
_current = DEFAULT_THEME


def theme():
    return _current


def set_current(name):
    global _current
    if name not in THEMES:
        raise ValueError(f"Неизвестная тема: {name}")
    _current = name


def color(key):
    return THEMES[_current][key]


def data_color(value):
    """Цвет из презентера (подобран для тёмного фона), приведённый к текущей теме.

    На светлом фоне яркие цвета затемняются с сохранением оттенка,
    а серые инвертируются по яркости.
    """
    if _current == "dark":
        return value
    h, l, s = colorsys.rgb_to_hls(*to_rgb(value))
    if s < 0.15:
        l = min(1.0 - l, 0.55)
    else:
        l = min(l, 0.38)
    return to_hex(colorsys.hls_to_rgb(h, l, s))


def stylesheet():
    c = THEMES[_current]
    return f"""
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-size: 13px;
}}
QGroupBox {{
    font-weight: bold;
    border: 1px solid {c['border']};
    border-radius: 6px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    background-color: {c['panel']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {c['accent']};
}}
QGroupBox QWidget {{ background-color: {c['panel']}; }}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {c['input']};
    border: 1px solid {c['border_strong']};
    border-radius: 4px;
    padding: 5px;
}}
QPushButton {{
    background-color: {c['button']};
    border: 1px solid {c['border_strong']};
    border-radius: 4px;
    padding: 7px 14px;
}}
QPushButton:hover {{ background-color: {c['button_hover']}; }}
QPushButton:disabled {{ color: {c['disabled']}; border-color: {c['border']}; }}
QPushButton#primary {{
    background-color: {c['primary']};
    border-color: {c['primary_hover']};
    color: {c['primary_text']};
    font-weight: bold;
}}
QPushButton#primary:hover {{ background-color: {c['primary_hover']}; }}
QPushButton#primary:disabled {{
    background-color: {c['primary_disabled']};
    color: {c['primary_disabled_text']};
}}
QRadioButton, QCheckBox {{ spacing: 8px; padding: 3px 0; }}
QRadioButton:disabled, QCheckBox:disabled, QLabel:disabled {{ color: {c['disabled']}; }}
QRadioButton::indicator, QCheckBox::indicator, QListView::indicator {{
    width: 14px;
    height: 14px;
    border: 2px solid {c['muted']};
    background-color: {c['bg']};
}}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator, QListView::indicator {{ border-radius: 3px; }}
QRadioButton::indicator:hover, QCheckBox::indicator:hover, QListView::indicator:hover {{
    border-color: {c['accent']};
}}
QRadioButton::indicator:checked, QCheckBox::indicator:checked,
QListView::indicator:checked {{
    background-color: {c['accent']};
    border-color: {c['accent']};
}}
QListWidget, QTableWidget, QTextBrowser {{
    background-color: {c['list']};
    border: 1px solid {c['border']};
    alternate-background-color: {c['list_alt']};
}}
QHeaderView::section {{
    background-color: {c['input'] if _current == 'dark' else c['header']};
    border: none;
    padding: 4px;
    font-weight: bold;
}}
QTabWidget::pane {{ border: 1px solid {c['border']}; border-radius: 4px; }}
QTabBar::tab {{
    background: {c['panel'] if _current == 'dark' else c['button']};
    padding: 7px 16px;
    border: 1px solid {c['border']};
    border-bottom: none;
}}
QTabBar::tab:selected {{
    background: {c['button'] if _current == 'dark' else c['panel']};
    color: {c['accent']};
}}
QProgressBar {{
    border: 1px solid {c['border']};
    border-radius: 4px;
    text-align: center;
    background: {c['list']};
}}
QProgressBar::chunk {{ background-color: {c['primary']}; }}
QSlider::groove:horizontal {{ height: 6px; background: {c['border']}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {c['accent']};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QScrollArea {{ border: none; }}
QToolTip {{
    background-color: {c['panel']};
    color: {c['text']};
    border: 1px solid {c['border_strong']};
}}
QLabel#title {{ font-size: 20px; font-weight: bold; color: {c['accent']}; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#chip {{ padding: 4px 10px; border-radius: 10px; }}
"""


def chip_style(state):
    c = THEMES[_current]
    return {
        "done": f"background-color: {c['primary']}; color: white;",
        "current": f"background-color: {c['chip_current']}; color: black; font-weight: bold;",
        "pending": f"background-color: {c['chip_pending']}; color: {c['chip_pending_text']};",
    }[state]


def feedback_style(ok):
    if ok is None:
        return f"color: {color('text')};"
    return f"color: {color('accent' if ok else 'error')}; font-weight: bold;"


def hint_style(kind="hint"):
    c = THEMES[_current]
    if kind == "answer":
        return (f"background-color: {c['answer_bg']}; color: {c['answer_text']}; "
                f"border: 1px solid {c['answer_border']}; border-radius: 5px; padding: 8px; "
                "font-weight: bold;")
    return (f"background-color: {c['hint_bg']}; color: {c['hint_text']}; "
            f"border: 1px solid {c['hint_border']}; border-radius: 5px; padding: 8px;")


def themed(widget, build):
    """Ставит виджету стиль build() и пересобирает его при смене темы."""
    widget._theme_style = build
    widget.setStyleSheet(build())


def apply_theme(root, name):
    """Включает тему name для окна root и всех его дочерних виджетов."""
    set_current(name)
    root.setStyleSheet(stylesheet())
    from masslab.views.qt.qt import QtWidgets
    for widget in [root] + root.findChildren(QtWidgets.QWidget):
        build = getattr(widget, "_theme_style", None)
        if build is not None:
            widget.setStyleSheet(build())
        refresh = getattr(widget, "apply_theme", None)
        if refresh is not None and widget is not root:
            refresh()
        widget.update()
