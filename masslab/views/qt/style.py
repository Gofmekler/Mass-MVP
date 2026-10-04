BG = "#1E1E1E"
PANEL = "#2A2A2A"
PLOT_BG = "#1A1A1A"
TEXT = "#E0E0E0"
MUTED = "#9E9E9E"
ACCENT = "#4CAF50"
ERROR = "#FF6B6B"

STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-size: 13px;
}}
QGroupBox {{
    font-weight: bold;
    border: 1px solid #444;
    border-radius: 6px;
    margin-top: 10px;
    padding: 12px 8px 8px 8px;
    background-color: {PANEL};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {ACCENT};
}}
QGroupBox QWidget {{ background-color: {PANEL}; }}
QLineEdit, QSpinBox {{
    background-color: #333;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px;
}}
QPushButton {{
    background-color: #3A3A3A;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 7px 14px;
}}
QPushButton:hover {{ background-color: #454545; }}
QPushButton:disabled {{ color: #666; border-color: #444; }}
QPushButton#primary {{
    background-color: #2E7D32;
    border-color: #388E3C;
    font-weight: bold;
}}
QPushButton#primary:hover {{ background-color: #388E3C; }}
QPushButton#primary:disabled {{ background-color: #2F3B30; color: #777; }}
QRadioButton, QCheckBox {{ spacing: 8px; padding: 3px 0; }}
QRadioButton::indicator, QCheckBox::indicator, QListView::indicator {{
    width: 14px;
    height: 14px;
    border: 2px solid #9E9E9E;
    background-color: #1E1E1E;
}}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator, QListView::indicator {{ border-radius: 3px; }}
QRadioButton::indicator:hover, QCheckBox::indicator:hover, QListView::indicator:hover {{
    border-color: {ACCENT};
}}
QRadioButton::indicator:checked, QCheckBox::indicator:checked,
QListView::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QListWidget, QTableWidget {{
    background-color: #262626;
    border: 1px solid #444;
    alternate-background-color: #2C2C2C;
}}
QHeaderView::section {{
    background-color: #333;
    border: none;
    padding: 4px;
    font-weight: bold;
}}
QTabWidget::pane {{ border: 1px solid #444; border-radius: 4px; }}
QTabBar::tab {{
    background: #2A2A2A;
    padding: 7px 16px;
    border: 1px solid #444;
    border-bottom: none;
}}
QTabBar::tab:selected {{ background: #3A3A3A; color: {ACCENT}; }}
QProgressBar {{
    border: 1px solid #444;
    border-radius: 4px;
    text-align: center;
    background: #262626;
}}
QProgressBar::chunk {{ background-color: #2E7D32; }}
QSlider::groove:horizontal {{ height: 6px; background: #444; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QScrollArea {{ border: none; }}
QLabel#title {{ font-size: 20px; font-weight: bold; color: {ACCENT}; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#chip {{ padding: 4px 10px; border-radius: 10px; }}
"""

CHIP_STYLES = {
    "done": "background-color: #2E7D32; color: white;",
    "current": "background-color: #F9A825; color: black; font-weight: bold;",
    "pending": "background-color: #333; color: #888;",
}


def feedback_style(ok):
    if ok is None:
        return f"color: {TEXT};"
    return f"color: {ACCENT if ok else ERROR}; font-weight: bold;"
