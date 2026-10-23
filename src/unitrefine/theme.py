"""
Colours and the Qt style sheet shared by all UnitRefine windows: a light page with white cards,
rounded controls and one accent colour.
"""

PAGE = "#F4F6FA"
CARD = "#FFFFFF"
BORDER = "#E4E7EC"
CONTROL_BORDER = "#D0D5DD"
TEXT = "#1D2939"
MUTED = "#667085"
ACCENT = "#4F46E5"
ACCENT_HOVER = "#4338CA"
ACCENT_SOFT = "#EEF0FF"
DANGER = "#D92D20"
DANGER_SOFT = "#FEE4E2"
SUCCESS = "#067647"
SUCCESS_SOFT = "#DCFAE6"
WARNING_SOFT = "#FEF6E0"

# Each section has a light tint of the colour it had before (light blue, green, peach, pink): a very light
# background, a slightly darker border, and a strong accent for its number badge. Controls and lists stay
# white on top of the tint, so they keep their contrast.
SECTION_THEMES = {
    1: dict(accent="#2E78B7", tint="#F1F7FC", border="#CFE2F3"),
    2: dict(accent="#2F9E55", tint="#F1FAF3", border="#CDEBD5"),
    3: dict(accent="#E07B24", tint="#FFF6EE", border="#F8DCC3"),
    4: dict(accent="#D6457E", tint="#FDF2F6", border="#F5D0DE"),
    5: dict(accent="#7A5AF8", tint="#F7F5FF", border="#DCD5FD"),
}

# buttons which open the SpikeInterface-GUI (Curate, Inspect, Relabel) share a light violet style
SIGUI_BUTTON = dict(background="#F0EFFF", border="#CBC8FA", text="#4336C9", hover="#E3E0FF")

STYLE_SHEET = f"""
QWidget {{
    color: {TEXT};
}}
QWidget#page, QDialog, QMainWindow {{
    background-color: {PAGE};
}}
QFrame#card {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QFrame#row {{
    background-color: {CARD};
    border: none;
    border-top: 1px solid {BORDER};
}}
QLabel {{
    background: transparent;
}}
QLabel#muted {{
    color: {MUTED};
}}
QLabel#subheading {{
    font-weight: bold;
    font-size: 11pt;
    padding-top: 8px;
}}
QLabel#chip {{
    background-color: #F2F4F7;
    color: #475467;
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 2px 9px;
}}
QLabel#chip_done {{
    background-color: {SUCCESS_SOFT};
    color: {SUCCESS};
    border-radius: 9px;
    padding: 2px 9px;
}}
QPushButton {{
    background-color: white;
    border: 1px solid {CONTROL_BORDER};
    border-radius: 8px;
    padding: 6px 14px;
}}
QPushButton:hover {{
    background-color: #F2F4F7;
    border-color: #98A2B3;
}}
QPushButton:pressed {{
    background-color: #EAECF0;
}}
QPushButton:disabled {{
    color: #98A2B3;
    background-color: #F9FAFB;
    border-color: #EAECF0;
}}
QPushButton[primary="true"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
    font-weight: bold;
}}
QPushButton[primary="true"]:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[primary="true"]:disabled {{
    background-color: #C7C9F5;
    border-color: #C7C9F5;
}}
QPushButton[sigui="true"] {{
    background-color: {SIGUI_BUTTON["background"]};
    border: 1px solid {SIGUI_BUTTON["border"]};
    color: {SIGUI_BUTTON["text"]};
    font-weight: bold;
}}
QPushButton[sigui="true"]:hover {{
    background-color: {SIGUI_BUTTON["hover"]};
    border-color: {SIGUI_BUTTON["text"]};
}}
QPushButton[sigui="true"]:disabled {{
    background-color: #F9FAFB;
    border-color: #EAECF0;
    color: #98A2B3;
    font-weight: normal;
}}
QFrame#list {{
    background-color: white;
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QLabel#row_name {{
    font-weight: bold;
}}
QLabel#row_location {{
    color: {MUTED};
    font-size: 8pt;
}}
QPushButton[flat="true"] {{
    border: none;
    background: transparent;
    color: {ACCENT};
    padding: 4px 8px;
}}
QPushButton[flat="true"]:hover {{
    background-color: {ACCENT_SOFT};
}}
QPushButton[flat="true"]:disabled {{
    color: #98A2B3;
    background: transparent;
}}
QToolButton {{
    border: none;
    border-radius: 6px;
    padding: 4px;
    background: transparent;
}}
QToolButton:hover {{
    background-color: {DANGER_SOFT};
}}
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit, QListWidget, QTableWidget {{
    background-color: white;
    border: 1px solid {CONTROL_BORDER};
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid #6366F1;
}}
QLineEdit:read-only {{
    background-color: #F9FAFB;
    color: {MUTED};
}}
QComboBox QAbstractItemView {{
    background-color: white;
    border: 1px solid {CONTROL_BORDER};
}}
QListWidget::item {{
    padding: 3px 2px;
}}
QHeaderView::section {{
    background-color: #F9FAFB;
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px;
    font-weight: bold;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    width: 10px;
    background: transparent;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #D0D5DD;
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar:horizontal {{
    height: 10px;
    background: transparent;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #D0D5DD;
    border-radius: 3px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0px;
    height: 0px;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QProgressBar {{
    border: none;
    background-color: #EAECF0;
    border-radius: 4px;
}}
QProgressBar::chunk {{
    background-color: #6366F1;
    border-radius: 4px;
}}
QToolTip {{
    background-color: #101828;
    color: white;
    border: none;
    padding: 6px;
}}
QMessageBox {{
    background-color: white;
}}
"""


def make_primary(button):
    """Styles a button as the main action of its section."""
    button.setProperty("primary", True)
    return button


def make_sigui(button):
    """Styles a button which opens the SpikeInterface-GUI."""
    button.setProperty("sigui", True)
    return button


def make_flat(button):
    """Styles a button as a small link-like action."""
    button.setProperty("flat", True)
    return button
