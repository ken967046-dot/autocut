"""Dark theme: colour constants, font choice and the application style sheet."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

# ------------------------------------------------------------------ colours (hex strings)
BG = "#1b1d21"            # window background
PANEL = "#23262b"         # docks / panels
PANEL_ALT = "#2b2f36"     # inputs, headers
BORDER = "#3a3f47"
TEXT = "#e6e8eb"
TEXT_DIM = "#9aa1ab"
ACCENT = "#4c9be8"        # primary accent (selection, focus, buttons)
ACCENT_DARK = "#2f6db0"
OK = "#4cc38a"
WARN = "#f5a623"
ERROR = "#ef5350"

# timeline
TL_BG = "#17191c"
TL_LANE = "#1f2226"
TL_LANE_ALT = "#1c1f23"
TL_GRID = "#2c3036"
TL_RULER = "#262a30"
CLIP_VIDEO = "#3b6ea5"
CLIP_AUDIO = "#3f8f6b"
CLIP_IMAGE = "#8a6bb0"
CLIP_CAPTION = "#b08a3b"
CLIP_CARD = "#c2577f"          # title cards (字卡) in the caption lane
CLIP_DISABLED = "#555a62"
CLIP_SELECTED = "#ffd166"
PLAYHEAD = "#ff4d4d"
MARKER = "#f5a623"
SNAP_LINE = "#7fe0ff"
RANGE_FILL = "#4c9be855"


def qcolor(hex_or_color: str | QColor, alpha: int | None = None) -> QColor:
    c = QColor(hex_or_color)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def pick_font_family() -> str:
    """Best available UI font for Traditional Chinese (Windows first, then common Linux fallbacks)."""
    families = set(QFontDatabase.families())
    for name in ("Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Noto Sans TC",
                 "Noto Sans CJK JP", "WenQuanYi Micro Hei", "Segoe UI"):
        if name in families:
            return name
    return QApplication.font().family()


def qss() -> str:
    return f"""
    QWidget {{ background: {BG}; color: {TEXT}; }}
    QMainWindow, QDialog {{ background: {BG}; }}
    QToolTip {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
    QMenuBar {{ background: {PANEL}; }}
    QMenuBar::item:selected {{ background: {PANEL_ALT}; }}
    QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
    QMenu::item {{ padding: 5px 24px 5px 18px; }}
    QMenu::item:selected {{ background: {ACCENT_DARK}; }}
    QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
    QToolBar {{ background: {PANEL}; border: none; spacing: 4px; padding: 3px; }}
    QStatusBar {{ background: {PANEL}; color: {TEXT_DIM}; }}
    QDockWidget {{ titlebar-close-icon: none; }}
    QDockWidget::title {{ background: {PANEL_ALT}; padding: 5px 8px; }}
    QSplitter::handle {{ background: {BORDER}; }}
    QSplitter::handle:horizontal {{ width: 2px; }}
    QSplitter::handle:vertical {{ height: 2px; }}
    QPushButton, QToolButton {{
        background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 5px 12px; }}
    QToolButton {{ padding: 4px 8px; }}
    QPushButton:hover, QToolButton:hover {{ border-color: {ACCENT}; }}
    QPushButton:pressed, QPushButton:checked, QToolButton:pressed, QToolButton:checked {{ background: {ACCENT_DARK}; border-color: {ACCENT}; }}
    QPushButton:disabled, QToolButton:disabled {{ color: {TEXT_DIM}; background: {PANEL}; border-color: {PANEL_ALT}; }}
    QPushButton#primary {{ background: {ACCENT_DARK}; border-color: {ACCENT}; font-weight: bold; }}
    QPushButton#primary:hover {{ background: {ACCENT}; }}
    QPushButton#danger {{ border-color: {ERROR}; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 6px;
        selection-background-color: {ACCENT_DARK}; }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
        border-color: {ACCENT}; }}
    QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: {ACCENT_DARK}; border: 1px solid {BORDER}; }}
    QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; }}
    QTabBar::tab {{ background: {PANEL}; padding: 6px 14px; border: 1px solid {BORDER}; border-bottom: none; }}
    QTabBar::tab:selected {{ background: {PANEL_ALT}; color: {TEXT}; border-top: 2px solid {ACCENT}; }}
    QTabBar::tab:!selected {{ color: {TEXT_DIM}; }}
    QHeaderView::section {{ background: {PANEL_ALT}; border: none; border-right: 1px solid {BORDER}; padding: 4px 6px; }}
    QTableView, QTreeView, QListView, QListWidget {{
        background: {PANEL}; alternate-background-color: {PANEL_ALT}; border: 1px solid {BORDER};
        selection-background-color: {ACCENT_DARK}; gridline-color: {BORDER}; }}
    QScrollBar:vertical {{ background: {PANEL}; width: 12px; margin: 0; }}
    QScrollBar:horizontal {{ background: {PANEL}; height: 12px; margin: 0; }}
    QScrollBar::handle {{ background: {BORDER}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
    QScrollBar::handle:hover {{ background: {TEXT_DIM}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; margin: -5px 0; border-radius: 6px; }}
    QSlider::sub-page:horizontal {{ background: {ACCENT_DARK}; border-radius: 2px; }}
    QProgressBar {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 3px; text-align: center; }}
    QProgressBar::chunk {{ background: {ACCENT}; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 14px; height: 14px; border: 1px solid {TEXT_DIM}; background: {PANEL_ALT}; border-radius: 3px; }}
    QRadioButton::indicator {{ border-radius: 8px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {BORDER}; background: {PANEL}; }}
    QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; margin-top: 12px; padding-top: 8px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {TEXT_DIM}; }}
    QLabel#dim {{ color: {TEXT_DIM}; }}
    QLabel#banner {{ background: {ACCENT_DARK}; border-radius: 4px; padding: 8px 12px; }}
    """


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(PANEL))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL_ALT))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(PANEL_ALT))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT_DARK))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor(PANEL_ALT))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DIM))
    app.setPalette(pal)
    font = QFont(pick_font_family())
    font.setPointSizeF(max(9.0, app.font().pointSizeF() if app.font().pointSizeF() > 0 else 9.5))
    app.setFont(font)
    app.setStyleSheet(qss())
