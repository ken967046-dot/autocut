"""Application entry (GUI_SPEC §1): ``autocut-studio`` and ``python -m autocut studio [files / folders / .acproj]``.

``main(argv)`` takes the command-line arguments *without* the program name (``None`` means ``sys.argv[1:]``), builds the
``QApplication`` (names, dark theme, drawn icon, Windows taskbar id), opens the :class:`MainWindow` and runs the loop.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QLinearGradient, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QMessageBox

from .. import APP_NAME, __version__
from . import theme
from .workers import log_path

APP_ID = "AutoCutStudio"
APP_USER_MODEL_ID = "AutoCutStudio.AutoCutStudio.1"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
CRASH_DIALOG_INTERVAL = 30.0


# ============================================================================ icon
def _paint_icon(p: QPainter, size: int) -> None:
    """A blue rounded tile with a white play triangle that a bright diagonal 'cut' slices through."""
    s = float(size)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    grad = QLinearGradient(0, 0, s, s)
    grad.setColorAt(0.0, QColor(theme.ACCENT))
    grad.setColorAt(1.0, QColor(theme.ACCENT_DARK))
    p.setPen(Qt.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92), s * 0.22, s * 0.22)
    p.setBrush(QColor("#ffffff"))
    p.drawPolygon(QPolygonF([QPointF(s * 0.36, s * 0.26), QPointF(s * 0.74, s * 0.5), QPointF(s * 0.36, s * 0.74)]))
    p.setPen(QPen(QColor(theme.WARN), max(1.5, s * 0.06), Qt.SolidLine, Qt.RoundCap))
    p.drawLine(QPointF(s * 0.2, s * 0.84), QPointF(s * 0.84, s * 0.2))


def build_app_icon() -> QIcon:
    """The window icon, drawn with QPainter at every common size (no image files to ship)."""
    icon = QIcon()
    for size in ICON_SIZES:
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        _paint_icon(p, size)
        p.end()
        icon.addPixmap(pix)
    return icon


# ============================================================================ process setup
def set_windows_app_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Windows only: own taskbar identity (so the taskbar shows our icon, not python.exe's)."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except (AttributeError, OSError):
        return False


def parse_paths(args: Sequence[str]) -> list[str]:
    """Files / folders / ``.acproj`` from the command line: options (``-x``) dropped, quotes stripped, absolute, unique."""
    out: list[str] = []
    for raw in args:
        text = str(raw).strip().strip('"')
        if text and not text.startswith("-"):
            path = os.path.abspath(text)
            if path not in out:
                out.append(path)
    return out


def install_excepthook(parent=None) -> None:
    """Log every uncaught exception and tell the user once in a while (a windowed app has no console)."""
    last = {"shown": 0.0}

    def hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(log_path(), "a", encoding="utf-8") as f:
                f.write(f"--- uncaught\n{text}\n")
        except OSError:
            pass
        if sys.__stderr__ is not None:
            sys.__stderr__.write(text)
        if time.monotonic() - last["shown"] > CRASH_DIALOG_INTERVAL:
            last["shown"] = time.monotonic()
            QTimer.singleShot(0, lambda: QMessageBox.warning(
                parent, "發生未預期的錯誤",
                f"{exc_type.__name__}：{exc}\n\n詳細內容已寫進記錄檔：\n{log_path()}\n\n建議先儲存專案再繼續。"))

    sys.excepthook = hook


# ============================================================================ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the desktop app.  ``argv``: arguments without the program name (``None`` -> ``sys.argv[1:]``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    set_windows_app_id()
    app = QApplication.instance()
    if app is None:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication([APP_ID])
    app.setApplicationName(APP_ID)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_ID)
    app.setApplicationVersion(__version__)
    theme.apply_theme(app)
    app.setWindowIcon(build_app_icon())
    install_excepthook()
    from .main_window import MainWindow          # imported late: it pulls in every widget of the editor
    window = MainWindow(paths=parse_paths(args))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
