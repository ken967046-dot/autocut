"""Media bin (素材庫): thumbnails, name / duration / size, missing-file marking, drag source and drop target.

GUI_SPEC §1 / §4 / §9.  The bin only *reads* the project; every change goes through ``state.do`` /
``state.project_op``.  Internal drags carry ``application/x-autocut-asset`` = UTF-8 JSON ``{"assets": [ids]}``;
files and folders dropped from the OS go to ``importer.import_paths``.  Thumbnails come from
``media.cache.thumbnail`` in background tasks (placeholder first, at most ``MAX_THUMB_JOBS`` at a time).

Signals
    shortsRequested(str asset_id)       context menu 「自動剪成短影音…」 (the window opens the dialog)
    assembleRequested(list asset_ids)   context menu 「自動合併成長片…」
    longformRequested(str asset_id)     context menu 「長片精華…」
    importRequested()                   the 「匯入…」 button (the window owns the file dialog)
    importStarted(object runner)        an OS drop started an import (the window shows its progress)
    assetSelectionChanged()             the selected assets changed
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from functools import partial
from typing import Callable, Iterable, Optional

from PySide6.QtCore import QMimeData, QModelIndex, QPoint, QPointF, QRect, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QDrag, QFont, QFontMetrics, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListView,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QStyle,
                               QStyledItemDelegate, QStyleOptionViewItem, QToolButton, QVBoxLayout, QWidget)

from ..core import ops
from ..core.ops import OpError
from ..core.project import Asset
from ..media.cache import thumbnail
from ..media.ffmpeg import FFmpegError
from ..media.probe import probe_asset
from ..util import human_duration, human_size
from . import theme
from .importer import file_dialog_filter
from .state import AppState
from .workers import TaskRunner, run_task

ASSET_MIME = "application/x-autocut-asset"
THUMB_HEIGHT = 120
MAX_THUMB_JOBS = 2

ROLE_ID = Qt.UserRole                # asset id
ROLE_INFO = Qt.UserRole + 1          # second text line (duration · size)
ROLE_KIND = Qt.UserRole + 2          # video | audio | image
ROLE_MISSING = Qt.UserRole + 3       # file not found on disk
ROLE_THUMB = Qt.UserRole + 4         # QPixmap or None
ROLE_BADGE = Qt.UserRole + 5         # duration badge text (icon view)
ROLE_SEARCH = Qt.UserRole + 6        # lower-case haystack for the search box
ROLE_SIG = Qt.UserRole + 7           # signature of the asset the item was built from

KIND_COLOR = {"video": theme.CLIP_VIDEO, "audio": theme.CLIP_AUDIO, "image": theme.CLIP_IMAGE}
KIND_LABEL = {"video": "影片", "audio": "音訊", "image": "圖片"}
EMPTY_HINT = "把影片、音訊或圖片\n（或整個資料夾）拖到這裡"


# ============================================================================ dialogs (monkeypatched in tests)
def ask(parent: Optional[QWidget], title: str, text: str, yes: str = "確定", no: str = "取消") -> bool:
    """Modal question with Chinese buttons; True only when the user clicks ``yes``."""
    box = QMessageBox(QMessageBox.Question, title, text, QMessageBox.NoButton, parent)
    ok = box.addButton(yes, QMessageBox.AcceptRole)
    box.addButton(no, QMessageBox.RejectRole)
    box.setDefaultButton(ok)
    box.exec()
    return box.clickedButton() is ok


def inform(parent: Optional[QWidget], title: str, text: str, warning: bool = False) -> None:
    """Modal message with a single 「確定」 button."""
    box = QMessageBox(QMessageBox.Warning if warning else QMessageBox.Information, title, text,
                      QMessageBox.NoButton, parent)
    box.addButton("確定", QMessageBox.AcceptRole)
    box.exec()


def reveal_in_file_manager(path: str) -> bool:
    """Show ``path`` in the OS file manager (Explorer selects the file; otherwise its nearest existing folder opens)."""
    path = os.path.normpath(path)
    try:
        if os.path.exists(path):
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", f"/select,{path}"])
                return True
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
                return True
    except OSError:
        pass
    folder = os.path.dirname(path)
    while folder and not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return QDesktopServices.openUrl(QUrl.fromLocalFile(folder or os.getcwd()))


# ============================================================================ payloads
def encode_asset_payload(asset_ids: Iterable[str]) -> bytes:
    return json.dumps({"assets": list(asset_ids)}, ensure_ascii=False).encode("utf-8")


def decode_asset_payload(mime: QMimeData) -> list[str]:
    """Asset ids carried by an internal drag (empty when the payload is missing or malformed)."""
    if not mime.hasFormat(ASSET_MIME):
        return []
    try:
        data = json.loads(bytes(mime.data(ASSET_MIME)).decode("utf-8"))
        return [str(i) for i in data["assets"]]
    except (ValueError, KeyError, TypeError):
        return []


def local_paths(mime: QMimeData) -> list[str]:
    """Local files / folders of an OS drag (empty for anything else)."""
    if not mime.hasUrls():
        return []
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile() and u.toLocalFile()]


# ============================================================================ text
def describe_asset(a: Asset) -> tuple[str, str]:
    """(badge text, info line) for an asset: duration · resolution · fps."""
    dur = human_duration(a.duration) if a.kind != "image" and a.duration > 0 else ""
    parts: list[str] = []
    if a.kind == "audio" or (not a.has_video and a.kind != "image"):
        parts += [dur, "音訊"] if dur else ["音訊"]
    elif a.kind == "image":
        parts += ["圖片", f"{a.width}×{a.height}"] if a.width else ["圖片"]
    else:
        if dur:
            parts.append(dur)
        if a.width and a.height:
            parts.append(f"{a.width}×{a.height}")
        if a.fps > 0:
            parts.append(f"{a.fps:.4g} fps")
    return dur, " · ".join(p for p in parts if p)


def _signature(a: Asset) -> tuple:
    return (a.path, a.name, a.kind, round(a.duration, 3), a.width, a.height, round(a.fps, 3), a.has_video,
            a.has_audio, a.fingerprint, a.size_bytes)


def _thumb_key(a: Asset) -> tuple:
    return (a.path, a.fingerprint, a.kind)


def _thumb_time(a: Asset) -> float:
    return 0.0 if a.kind == "image" else min(2.0, max(0.0, a.duration * 0.1))


def _make_thumbnail(asset: Asset, _progress: Callable, _cancel) -> Optional[QImage]:
    """Worker thread: cached JPEG -> QImage (QImage, unlike QPixmap, may be created off the GUI thread).
    ``None`` when the file is gone (the item is red-marked instead of logging an error)."""
    if not os.path.exists(asset.path):
        return None
    img = QImage(str(thumbnail(asset, _thumb_time(asset), THUMB_HEIGHT)))
    if img.isNull():
        raise FFmpegError("縮圖讀取失敗")
    return img


# ============================================================================ painting
def draw_placeholder(p: QPainter, rect: QRectF, kind: str) -> None:
    """Thumbnail stand-in: kind colour + a play triangle / sound bars / landscape."""
    base = QColor(KIND_COLOR.get(kind, theme.PANEL_ALT))
    p.fillRect(rect, base.darker(280))
    p.setPen(Qt.NoPen)
    p.setBrush(base.lighter(150))
    c, s = rect.center(), min(rect.width(), rect.height()) * 0.34

    def poly(*pts: tuple[float, float]) -> QPolygonF:
        return QPolygonF([QPointF(c.x() + x * s, c.y() + y * s) for x, y in pts])

    if kind == "audio":
        for i, h in enumerate((0.45, 0.9, 0.65, 1.0, 0.55)):
            w = s * 0.3
            p.drawRoundedRect(QRectF(c.x() + (i - 2) * w * 1.6 - w / 2, c.y() - s * h, w, 2 * s * h), 1.5, 1.5)
    elif kind == "image":
        p.drawPolygon(poly((-1.0, 0.7), (-0.2, -0.6), (0.5, 0.7)))
        p.drawPolygon(poly((0.1, 0.7), (0.65, -0.1), (1.1, 0.7)))
        p.drawEllipse(QRectF(c.x() + s * 0.45, c.y() - s * 0.95, s * 0.4, s * 0.4))
    else:
        p.drawPolygon(poly((-0.5, -0.7), (0.75, 0.0), (-0.5, 0.7)))


def draw_thumb(p: QPainter, rect: QRectF, pix: Optional[QPixmap], kind: str, missing: bool) -> None:
    """Rounded thumbnail frame; a missing file gets a red tint, a red slash and a red border."""
    p.save()
    clip = QPainterPath()
    clip.addRoundedRect(rect, 4, 4)
    p.setClipPath(clip)
    p.fillRect(rect, QColor("#101215"))
    if pix is not None and not pix.isNull():
        fit = pix.size().scaled(rect.size().toSize(), Qt.KeepAspectRatio)
        target = QRectF(0, 0, fit.width(), fit.height())
        target.moveCenter(rect.center())
        p.drawPixmap(target, pix, QRectF(pix.rect()))
    else:
        draw_placeholder(p, rect, kind)
    if missing:
        p.fillRect(rect, QColor(120, 20, 20, 140))
        p.setPen(QPen(QColor(theme.ERROR), 2))
        p.drawLine(rect.bottomLeft(), rect.topRight())
    p.setClipping(False)
    p.setPen(QPen(QColor(theme.ERROR if missing else theme.BORDER), 1.6 if missing else 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(rect, 4, 4)
    p.restore()


class _AssetDelegate(QStyledItemDelegate):
    """List view: thumbnail + two text lines; icon view: thumbnail with a duration badge, name below."""

    ROW = QSize(220, 68)
    CELL = QSize(140, 112)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.icon_mode = False

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # noqa: N802 (Qt API)
        return self.CELL if self.icon_mode else self.ROW

    def paint(self, p: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        p.save()
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        rect = QRectF(option.rect).adjusted(2, 2, -2, -2)
        selected = bool(option.state & QStyle.State_Selected)
        if selected or option.state & QStyle.State_MouseOver:
            p.setPen(QPen(QColor(theme.ACCENT), 1) if selected else Qt.NoPen)
            p.setBrush(QColor(theme.ACCENT_DARK) if selected else QColor(theme.PANEL_ALT))
            p.drawRoundedRect(rect, 5, 5)
        (self._paint_cell if self.icon_mode else self._paint_row)(p, rect, index)
        p.restore()

    @staticmethod
    def _title_color(missing: bool) -> QColor:
        return QColor(theme.ERROR if missing else theme.TEXT)

    def _paint_row(self, p: QPainter, rect: QRectF, index: QModelIndex) -> None:
        missing = bool(index.data(ROLE_MISSING))
        side = rect.height() - 10
        thumb = QRectF(rect.left() + 5, rect.top() + 5, side * 16 / 9, side)
        draw_thumb(p, thumb, index.data(ROLE_THUMB), index.data(ROLE_KIND) or "video", missing)
        x = thumb.right() + 10
        width = max(10, int(rect.right() - x - 6))
        font = QFont(p.font())
        font.setBold(True)
        p.setFont(font)
        fm = QFontMetrics(font)
        p.setPen(self._title_color(missing))
        p.drawText(QPoint(int(x), int(rect.top() + rect.height() / 2 - 4)),
                   fm.elidedText(index.data(Qt.DisplayRole) or "", Qt.ElideMiddle, width))
        font.setBold(False)
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        p.setFont(font)
        fm = QFontMetrics(font)
        p.setPen(QColor(theme.ERROR if missing else theme.TEXT_DIM))
        info = "找不到檔案（右鍵 → 重新連結…）" if missing else (index.data(ROLE_INFO) or "")
        p.drawText(QPoint(int(x), int(rect.top() + rect.height() / 2 + 6 + fm.ascent())),
                   fm.elidedText(info, Qt.ElideRight, width))

    def _paint_cell(self, p: QPainter, rect: QRectF, index: QModelIndex) -> None:
        missing = bool(index.data(ROLE_MISSING))
        thumb = QRectF(rect.left() + 5, rect.top() + 5, rect.width() - 10, (rect.width() - 10) * 9 / 16)
        draw_thumb(p, thumb, index.data(ROLE_THUMB), index.data(ROLE_KIND) or "video", missing)
        font = QFont(p.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        p.setFont(font)
        fm = QFontMetrics(font)
        badge = "找不到檔案" if missing else (index.data(ROLE_BADGE) or "")
        if badge:
            box = QRectF(0, 0, fm.horizontalAdvance(badge) + 10, fm.height() + 2)
            box.moveBottomRight(thumb.bottomRight() - QPoint(4, 4))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.ERROR) if missing else QColor(0, 0, 0, 170))
            p.drawRoundedRect(box, 3, 3)
            p.setPen(QColor("#ffffff"))
            p.drawText(box, Qt.AlignCenter, badge)
        p.setPen(self._title_color(missing))
        p.drawText(QRect(int(rect.left()) + 4, int(thumb.bottom()) + 3, int(rect.width()) - 8, fm.height() + 2),
                   Qt.AlignHCenter | Qt.AlignTop, fm.elidedText(index.data(Qt.DisplayRole) or "", Qt.ElideMiddle,
                                                                 int(rect.width()) - 8))


# ============================================================================ the list
class _AssetList(QListWidget):
    """Drag source for assets, drop target for OS files (``filesDropped``)."""

    filesDropped = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hover = False
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.setUniformItemSizes(True)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)             # NOTE: this switches dragEnabled AND the viewport's acceptDrops off
        self.setFrameShape(QListWidget.NoFrame)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.enable_drag_and_drop()
        self.hint = QLabel(EMPTY_HINT, self.viewport())
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hint.setStyleSheet(f"background: transparent; color: {theme.TEXT_DIM};")

    def enable_drag_and_drop(self) -> None:
        """Drag source for assets + drop target for OS files.  Qt's ``setMovement`` / ``setViewMode`` silently switch
        ``dragEnabled`` and the viewport's ``acceptDrops`` OFF, so this must run after every one of those calls:
        otherwise nothing can be dragged out of the bin and the viewport never receives dragged files."""
        self.setDragDropMode(QAbstractItemView.DragOnly)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)

    # ---- drag source
    def mimeTypes(self) -> list[str]:                  # noqa: N802 (Qt API)
        return [ASSET_MIME]

    def mimeData(self, items) -> QMimeData:           # noqa: N802
        mime = QMimeData()
        rows = sorted(items, key=self.row)
        mime.setData(ASSET_MIME, encode_asset_payload(i.data(ROLE_ID) for i in rows))
        return mime

    def supportedDragActions(self) -> Qt.DropAction:  # noqa: N802
        return Qt.CopyAction

    def startDrag(self, _supported) -> None:          # noqa: N802
        items = self.selectedItems()
        if not items:
            return
        drag = QDrag(self)
        drag.setMimeData(self.mimeData(items))
        drag.setPixmap(self._drag_pixmap(items))
        drag.setHotSpot(QPoint(14, 14))
        self._exec_drag(drag)

    def _exec_drag(self, drag: QDrag) -> None:
        """Run the OS drag loop (tests replace this seam: it blocks until the drop in a real session)."""
        drag.exec(Qt.CopyAction, Qt.CopyAction)

    def _drag_pixmap(self, items: list[QListWidgetItem]) -> QPixmap:
        first = sorted(items, key=self.row)[0]
        pix = QPixmap(190, 44)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        p.setBrush(QColor(theme.ACCENT_DARK))
        p.setPen(QPen(QColor(theme.ACCENT), 1))
        p.setOpacity(0.92)
        p.drawRoundedRect(QRectF(0.5, 0.5, 189, 43), 6, 6)
        draw_thumb(p, QRectF(5, 5, 50, 34), first.data(ROLE_THUMB), first.data(ROLE_KIND) or "video",
                   bool(first.data(ROLE_MISSING)))
        p.setPen(QColor("#ffffff"))
        extra = f"  +{len(items) - 1}" if len(items) > 1 else ""
        fm = QFontMetrics(p.font())
        p.drawText(QRect(62, 0, 122, 44), Qt.AlignVCenter | Qt.AlignLeft,
                   fm.elidedText((first.data(Qt.DisplayRole) or "") + extra, Qt.ElideMiddle, 122))
        p.end()
        return pix

    # ---- OS files dropped on the list
    def dragEnterEvent(self, ev) -> None:             # noqa: N802
        if local_paths(ev.mimeData()):
            self._set_hover(True)
            ev.setDropAction(Qt.CopyAction)
            ev.accept()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev) -> None:              # noqa: N802
        if local_paths(ev.mimeData()):
            ev.setDropAction(Qt.CopyAction)
            ev.accept()
        else:
            ev.ignore()

    def dragLeaveEvent(self, ev) -> None:             # noqa: N802
        self._set_hover(False)
        ev.accept()

    def dropEvent(self, ev) -> None:                  # noqa: N802
        self._set_hover(False)
        paths = local_paths(ev.mimeData())
        if not paths:
            ev.ignore()
            return
        ev.setDropAction(Qt.CopyAction)
        ev.accept()
        self.filesDropped.emit(paths)

    def _set_hover(self, on: bool) -> None:
        self._hover = on
        self.viewport().update()

    # ---- painting / layout extras
    def paintEvent(self, ev) -> None:                 # noqa: N802
        super().paintEvent(ev)
        if not self._hover:
            return
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.viewport().rect()).adjusted(3, 3, -3, -3)
        p.setBrush(QColor(76, 155, 232, 45))
        pen = QPen(QColor(theme.ACCENT), 2, Qt.DashLine)
        p.setPen(pen)
        p.drawRoundedRect(r, 8, 8)
        font = QFont(p.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 2)
        p.setFont(font)
        label = "放開以匯入"
        box = QRectF(0, 0, QFontMetrics(font).horizontalAdvance(label) + 40, QFontMetrics(font).height() + 18)
        box.moveCenter(r.center())
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.ACCENT_DARK))                # a solid pill, so the text stays readable over the rows
        p.drawRoundedRect(box, 9, 9)
        p.setPen(QColor("#ffffff"))
        p.drawText(box, Qt.AlignCenter, label)
        p.end()

    def resizeEvent(self, ev) -> None:                # noqa: N802
        super().resizeEvent(ev)
        self.hint.setGeometry(self.viewport().rect().adjusted(16, 16, -16, -16))


# ============================================================================ the widget
class MediaBin(QWidget):
    shortsRequested = Signal(str)
    assembleRequested = Signal(list)
    longformRequested = Signal(str)
    importRequested = Signal()
    importStarted = Signal(object)
    assetSelectionChanged = Signal()

    def __init__(self, state: AppState, importer, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.importer = importer
        self.runners: list[TaskRunner] = []
        self._items: dict[str, QListWidgetItem] = {}
        self._thumb_keys: dict[str, tuple] = {}       # asset id -> thumb key the pixmap / failure belongs to
        self._thumb_queue: list[str] = []
        self._thumb_active: set[str] = set()
        self._thumb_failed: set[str] = set()
        self._scan: Optional[TaskRunner] = None
        self._scan_again = False
        self.setAcceptDrops(True)

        self.search = QLineEdit()
        self.search.setPlaceholderText("搜尋素材")
        self.search.setClearButtonEnabled(True)
        self.btn_view = QToolButton()
        self.btn_view.setCheckable(True)
        self.btn_view.setText("縮圖")
        self.btn_view.setToolTip("切換清單／縮圖顯示")
        self.btn_import = QPushButton("匯入…")
        self.btn_import.setToolTip("匯入影片、音訊、圖片或資料夾（Ctrl+I）")
        self.list = _AssetList()
        self.delegate = _AssetDelegate(self.list)
        self.list.setItemDelegate(self.delegate)
        self.footer = QLabel()
        self.footer.setObjectName("dim")
        self.footer.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.btn_view)
        top.addWidget(self.btn_import)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        lay.addLayout(top)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.footer)

        self.search.textChanged.connect(self._apply_filter)
        self.btn_view.toggled.connect(self.set_icon_mode)
        self.btn_import.clicked.connect(self.importRequested)
        self.list.itemSelectionChanged.connect(self.assetSelectionChanged)
        self.list.itemActivated.connect(lambda _item: self.place_selected())
        self.list.customContextMenuRequested.connect(self._show_menu)
        self.list.filesDropped.connect(self.import_dropped)
        state.edited.connect(lambda *_: self.refresh())
        state.project_replaced.connect(self.refresh)
        self.set_icon_mode(False)
        self.refresh()

    # ------------------------------------------------------------------ public API
    @property
    def busy(self) -> bool:
        """Thumbnails or the missing-file scan still running (tests / shutdown)."""
        return bool(self._thumb_queue or self._thumb_active) or self._scan is not None

    def selected_asset_ids(self) -> list[str]:
        """Selected assets in list order."""
        return [i.data(ROLE_ID) for i in sorted(self.list.selectedItems(), key=self.list.row)]

    def item_for(self, asset_id: str) -> Optional[QListWidgetItem]:
        return self._items.get(asset_id)

    def thumbnail_for(self, asset_id: str) -> Optional[QPixmap]:
        item = self._items.get(asset_id)
        return item.data(ROLE_THUMB) if item is not None else None

    def is_missing(self, asset_id: str) -> bool:
        item = self._items.get(asset_id)
        return bool(item.data(ROLE_MISSING)) if item is not None else False

    def visible_asset_ids(self) -> list[str]:
        return [self.list.item(r).data(ROLE_ID) for r in range(self.list.count()) if not self.list.item(r).isHidden()]

    def icon_mode(self) -> bool:
        return self.delegate.icon_mode

    def set_icon_mode(self, on: bool) -> None:
        """Icon (縮圖) or list (清單) view."""
        self.delegate.icon_mode = bool(on)
        if self.btn_view.isChecked() != bool(on):
            self.btn_view.setChecked(bool(on))
        self.btn_view.setText("清單" if on else "縮圖")
        self.list.setViewMode(QListView.IconMode if on else QListView.ListMode)
        self.list.enable_drag_and_drop()                     # setViewMode switched dragging / dropping off again
        self.list.setWrapping(bool(on))
        self.list.setFlow(QListView.LeftToRight if on else QListView.TopToBottom)
        self.list.setSpacing(2)
        self.list.setGridSize(self.delegate.CELL if on else QSize())
        self.list.doItemsLayout()
        self.list.viewport().update()

    def refresh(self) -> None:
        """Sync the items with ``project.assets`` (keeps the selection of unchanged assets)."""
        assets = self.state.project.assets
        for aid in [i for i in self._items if i not in assets]:
            self.list.takeItem(self.list.row(self._items.pop(aid)))
            self._thumb_keys.pop(aid, None)
            self._thumb_failed.discard(aid)
        rescan = False
        for pos, asset in enumerate(assets.values()):
            item = self._items.get(asset.id)
            if item is None:
                item = QListWidgetItem()
                item.setData(ROLE_ID, asset.id)
                self.list.insertItem(pos, item)
                self._items[asset.id] = item
            elif self.list.row(item) != pos:
                self.list.insertItem(pos, self.list.takeItem(self.list.row(item)))
            if item.data(ROLE_SIG) != _signature(asset):
                self._apply_asset(item, asset)
                rescan = True
        self._apply_filter()
        if rescan:
            self.recheck_missing()
        self._update_footer()

    def recheck_missing(self) -> None:
        """Look for vanished / reappeared files in the background and re-mark the items."""
        if self._scan is not None:
            self._scan_again = True
            return
        paths = {a.id: a.path for a in self.state.project.assets.values()}
        runner = TaskRunner(lambda _p, _c: {i: not os.path.exists(p) for i, p in paths.items()}, parent=self,
                            name="檢查素材檔案")
        runner.finished.connect(self._apply_missing)
        runner.done.connect(self._scan_done)
        self._scan = runner
        runner.start()

    def place_selected(self) -> list[str]:
        """Double-click / Enter: put the selected assets on the timeline (``importer.place_assets``)."""
        ids = self.selected_asset_ids()
        return self.importer.place_assets(ids) if ids else []

    def import_dropped(self, paths: list[str]) -> None:
        """OS files / folders dropped on the bin -> importer (the window shows the progress)."""
        runner = self.importer.import_paths(paths)
        self.importStarted.emit(runner)

    # ------------------------------------------------------------------ items
    def _apply_asset(self, item: QListWidgetItem, a: Asset) -> None:
        badge, info = describe_asset(a)
        item.setData(Qt.DisplayRole, a.name or os.path.basename(a.path))
        item.setData(ROLE_INFO, info)
        item.setData(ROLE_BADGE, badge)
        item.setData(ROLE_KIND, a.kind if a.kind in KIND_COLOR else "video")
        item.setData(ROLE_SEARCH, f"{a.name}\n{a.path}".lower())
        item.setData(ROLE_SIG, _signature(a))
        size = f"\n{human_size(a.size_bytes)}" if a.size_bytes else ""
        item.setToolTip(f"{a.path}\n{KIND_LABEL.get(a.kind, a.kind)} · {info}{size}")
        key = _thumb_key(a)
        if self._thumb_keys.get(a.id) != key:
            self._thumb_keys[a.id] = key
            self._thumb_failed.discard(a.id)
            item.setData(ROLE_THUMB, None)
            self._queue_thumbnail(a.id)

    def _apply_filter(self, *_args) -> None:
        needle = self.search.text().strip().lower()
        shown = 0
        for r in range(self.list.count()):
            it = self.list.item(r)
            hide = bool(needle) and needle not in (it.data(ROLE_SEARCH) or "")
            if hide and it.isSelected():
                it.setSelected(False)                      # an invisible selection would still be deleted / placed
            it.setHidden(hide)
            shown += 0 if hide else 1
        total = self.list.count()
        self.list.hint.setVisible(shown == 0)
        self.list.hint.setText(EMPTY_HINT if total == 0 else f"沒有符合「{self.search.text().strip()}」的素材")
        self._update_footer()

    def _update_footer(self) -> None:
        assets = list(self.state.project.assets.values())
        missing = sum(1 for a in assets if self.is_missing(a.id))
        total = sum(a.duration for a in assets if a.kind != "image")
        text = f"{len(assets)} 個素材" + (f" · 總長 {human_duration(total)}" if assets and total > 0 else "")
        if missing:
            text += f" · <span style='color:{theme.ERROR}'>{missing} 個找不到檔案</span>"
        self.footer.setTextFormat(Qt.RichText)
        self.footer.setText(text)

    def _apply_missing(self, result: dict) -> None:
        for aid, missing in result.items():
            item = self._items.get(aid)
            if item is not None and bool(item.data(ROLE_MISSING)) != missing:
                item.setData(ROLE_MISSING, missing)
                if not missing and item.data(ROLE_THUMB) is None:     # the file is back: try the thumbnail again
                    self._thumb_failed.discard(aid)
                    self._queue_thumbnail(aid)
        self.list.viewport().update()
        self._update_footer()

    def _scan_done(self) -> None:
        self._scan = None
        if self._scan_again:
            self._scan_again = False
            self.recheck_missing()

    # ------------------------------------------------------------------ thumbnails
    def _queue_thumbnail(self, asset_id: str) -> None:
        if asset_id not in self._thumb_queue and asset_id not in self._thumb_active:
            self._thumb_queue.append(asset_id)
        self._pump_thumbnails()

    def _pump_thumbnails(self) -> None:
        while self._thumb_queue and len(self._thumb_active) < MAX_THUMB_JOBS:
            aid = self._thumb_queue.pop(0)
            asset = self.state.project.assets.get(aid)
            if asset is None or aid in self._thumb_failed:
                continue
            if asset.kind == "audio" or not (asset.has_video or asset.kind == "image"):
                continue                                   # audio gets the drawn placeholder, no ffmpeg run
            snapshot, key = replace(asset), _thumb_key(asset)
            runner = TaskRunner(partial(_make_thumbnail, snapshot), parent=self, name="素材縮圖")
            runner.finished.connect(partial(self._thumb_ready, aid, key))
            runner.failed.connect(partial(self._thumb_error, aid, key))
            runner.done.connect(partial(self._thumb_done, aid, runner))
            self._thumb_active.add(aid)
            self.runners.append(runner)
            runner.start()

    def _thumb_ready(self, aid: str, key: tuple, image: Optional[QImage]) -> None:
        item = self._items.get(aid)
        if item is None or self._thumb_keys.get(aid) != key:
            return
        if image is None:
            self._thumb_failed.add(aid)
            return
        item.setData(ROLE_THUMB, QPixmap.fromImage(image))
        self.list.viewport().update()

    def _thumb_error(self, aid: str, key: tuple, _message: str) -> None:
        if self._thumb_keys.get(aid) == key:
            self._thumb_failed.add(aid)

    def _thumb_done(self, aid: str, runner: TaskRunner) -> None:
        self._thumb_active.discard(aid)
        if runner in self.runners:
            self.runners.remove(runner)
        if aid in self._items and self._thumb_keys.get(aid) != _thumb_key(self.state.project.assets[aid]):
            self._queue_thumbnail(aid)                     # the file changed (relink) while we were busy
        self._pump_thumbnails()

    # ------------------------------------------------------------------ context menu
    def build_context_menu(self, ids: list[str]) -> QMenu:
        """The right-click menu for the given selection (public so tests can trigger its actions)."""
        menu = QMenu(self)
        assets = self.state.project.assets
        videos = [i for i in ids if i in assets and assets[i].kind == "video" and assets[i].has_video]
        single = len(ids) == 1

        def add(text: str, slot: Callable[[], object], enabled: bool = True, tip: str = "") -> QAction:
            act = menu.addAction(text)
            act.setEnabled(enabled)
            if tip:
                act.setToolTip(tip)
            act.triggered.connect(lambda _checked=False: slot())
            return act

        if not ids:
            add("匯入素材…", self.importRequested.emit)
            return menu
        add("加入時間軸", lambda: self.importer.place_assets(ids))
        add("自動剪成短影音…", lambda: self.shortsRequested.emit(videos[0]), single and len(videos) == 1,
            "選取 1 支影片時可用")
        add("長片精華…", lambda: self.longformRequested.emit(videos[0]), single and len(videos) == 1,
            "選取 1 支影片時可用：留重點、剪水詞、說話的人自動放大")
        add("自動合併成長片…", lambda: self.assembleRequested.emit(list(videos)), len(videos) >= 2,
            "選取 2 支以上影片時可用")
        menu.addSeparator()
        add("在檔案總管顯示", lambda: reveal_in_file_manager(assets[ids[0]].path), single)
        add("重新連結…", lambda: self.relink(ids[0]), single)
        menu.addSeparator()
        add("移除素材", lambda: self.remove_assets(ids))
        return menu

    def _show_menu(self, pos: QPoint) -> None:
        item = self.list.itemAt(pos)
        if item is not None and not item.isSelected():
            self.list.clearSelection()
            item.setSelected(True)
        elif item is None:
            self.list.clearSelection()
        self._popup_menu(self.build_context_menu(self.selected_asset_ids()), self.list.viewport().mapToGlobal(pos))

    def _popup_menu(self, menu: QMenu, global_pos: QPoint) -> None:
        """Show the menu and wait for the choice (tests replace this seam: a real modal menu never returns
        under the offscreen platform)."""
        menu.exec(global_pos)
        menu.deleteLater()

    def keyPressEvent(self, ev) -> None:              # noqa: N802
        if ev.key() == Qt.Key_Delete and self.list.hasFocus() and self.selected_asset_ids():
            self.remove_assets(self.selected_asset_ids())
            ev.accept()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------ actions
    def clips_using(self, asset_ids: Iterable[str]) -> int:
        wanted = set(asset_ids)
        return sum(1 for s in self.state.project.sequences for _t, c in s.all_clips() if c.asset_id in wanted)

    def remove_assets(self, ids: list[str]) -> bool:
        """Remove assets from the project (one undo step); asks first when timeline clips use them."""
        ids = [i for i in ids if i in self.state.project.assets]
        if not ids:
            return False
        used = self.clips_using(ids)
        if used:
            these = "這個素材" if len(ids) == 1 else f"這 {len(ids)} 個素材"
            if not ask(self, "移除素材", f"{these}正被 {used} 個時間軸片段使用。\n移除後這些片段也會一併刪除"
                                         "（電腦裡的檔案不會被刪，可用 Ctrl+Z 復原）。\n\n確定要移除嗎？", "移除"):
                return False
        try:
            self.state.do("移除素材", lambda p: [ops.remove_asset(p, i, force=True) for i in ids])
        except OpError:
            return False
        self.state.status.emit(f"已移除 {len(ids)} 個素材" + (f"與 {used} 個片段" if used else ""))
        return True

    def relink(self, asset_id: str, new_path: Optional[str] = None) -> None:
        """「重新連結…」: pick the moved file (or take ``new_path``), probe it, then ``ops.relink_asset``."""
        asset = self.state.project.assets.get(asset_id)
        if asset is None:
            return
        if new_path is None:
            folder = os.path.dirname(asset.path)
            new_path, _ = QFileDialog.getOpenFileName(self, f"重新連結「{asset.name}」",
                                                      folder if os.path.isdir(folder) else "", file_dialog_filter())
        if not new_path:
            return
        run_task(lambda _p, _c: probe_asset(new_path), parent=self, name="讀取新檔案",
                 on_done=lambda info: self._finish_relink(asset_id, info),
                 on_error=lambda msg: inform(self, "無法重新連結", msg, warning=True))

    def _finish_relink(self, asset_id: str, info: Asset) -> None:
        old = self.state.project.assets.get(asset_id)
        if old is None:
            return
        problem = self._relink_problem(old, info)
        if problem and not ask(self, "重新連結", problem + "\n\n仍要重新連結嗎？", "仍要連結"):
            return
        try:
            self.state.project_op("重新連結素材", "relink_asset", asset_id, info)
        except OpError:
            return
        self.state.status.emit(f"已重新連結：{info.name}")
        self.recheck_missing()

    def _relink_problem(self, old: Asset, new: Asset) -> str:
        if new.kind != old.kind:
            return f"新檔案是{KIND_LABEL.get(new.kind, new.kind)}，原本的素材是{KIND_LABEL.get(old.kind, old.kind)}。"
        if new.kind == "image":
            return ""
        used_to = max((c.src_out for s in self.state.project.sequences for _t, c in s.all_clips()
                       if c.asset_id == old.id), default=0.0)
        if used_to > new.duration + 0.05:
            return (f"新檔案只有 {new.duration:.1f} 秒，但時間軸已經用到原素材的第 {used_to:.1f} 秒，"
                    "超出的片段會出錯，需要自己再修剪。")
        return ""

    # ------------------------------------------------------------------ OS drops on the container
    def dragEnterEvent(self, ev) -> None:             # noqa: N802
        if local_paths(ev.mimeData()):
            ev.setDropAction(Qt.CopyAction)
            ev.accept()
        else:
            ev.ignore()

    def dragMoveEvent(self, ev) -> None:              # noqa: N802
        self.dragEnterEvent(ev)

    def dropEvent(self, ev) -> None:                  # noqa: N802
        paths = local_paths(ev.mimeData())
        if not paths:
            ev.ignore()
            return
        ev.setDropAction(Qt.CopyAction)
        ev.accept()
        self.import_dropped(paths)
