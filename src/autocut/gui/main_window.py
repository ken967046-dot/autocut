"""Main window: menus, toolbar, docks, monitor + sequence tabs + timeline, window-wide drag & drop, suggestion banner,
autosave / recovery, unsaved-changes prompts and settings persistence (GUI_SPEC §1 / §4 / §9 / §11 / §11.1).

Shortcuts: only the global ones of §11.1 are real ``QAction`` shortcuts (new / open / save / save as / import / export /
render preview / undo / redo / play / quit / full screen).  Timeline commands in the menus carry a text hint
(``"在播放頭分割\\tS"``) and call the public methods of :class:`TimelineWidget`; the timeline binds those keys itself.
All project changes go through ``state.op`` / ``state.do`` / ``state.project_op``.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QRect, QRectF, QSettings, QSize, QStandardPaths, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QGuiApplication, QKeySequence, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QDockWidget, QFileDialog, QFrame,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabBar, QToolBar, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..core.ops import OpError
from ..core.project import Asset, Project
from ..core.subtitles import load_subtitle_file
from ..core.timecode import fmt_timecode
from ..util import PROJECT_EXT, app_data_dir, atomic_write_text, human_duration
from . import theme
from .app import build_app_icon
from .autoedit_dialog import AssembleDialog, LongformDialog, ShortsDialog
from .export_dialog import ExportDialog, safe_stem
from .importer import ImportController, file_dialog_filter, is_project_file
from .inspector import InspectorPanel
from .media_bin import MediaBin, ask, inform, local_paths
from .monitor import MonitorWidget
from .playback import PlaybackController
from .state import AppState
from .timeline import TimelineWidget
from .widgets import keep_keyboard_on_timeline, keeps_its_keys
from .workers import TaskRunner
from .workers import shutdown as shutdown_tasks

SETTINGS_ORG = SETTINGS_APP = "AutoCutStudio"
AUTOSAVE_MS = 60_000
MAX_RECENT = 10
MISSING_CHECK_GAP = 3.0
MIN_WINDOW = (1000, 640)
IDEAL_WINDOW = (1500, 920)
SPLIT_SIZES = [470, 450]            # monitor : timeline - about half and half, so that the audio lanes show at 1100x700 too
STATUS_MS = 8000
SEQUENCE_PRESETS = (("16:9 橫式（1920×1080）", 1920, 1080), ("9:16 直式（1080×1920）", 1080, 1920),
                    ("1:1 方形（1080×1080）", 1080, 1080), ("4:5 直式（1080×1350）", 1080, 1350))


def app_settings() -> QSettings:
    """The app's persistent settings (tests redirect them with ``QSettings.setPath`` / ``setDefaultFormat``)."""
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def default_window_size(avail: QRect) -> QSize:
    """First-run window size: 1500x920 on a big screen, else what fits the usable screen area (never below the minimum,
    so a 1366x768 laptop does not get a window taller than its screen)."""
    w = min(IDEAL_WINDOW[0], avail.width() - 40)
    h = min(IDEAL_WINDOW[1], avail.height() - 60)
    return QSize(max(MIN_WINDOW[0], w), max(MIN_WINDOW[1], h))


def _as_list(value) -> list[str]:
    """QSettings returns a bare string for one-element lists in some formats."""
    if value is None or value == "":
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


# ============================================================================ dialogs (monkeypatched in tests)
def ask_save_changes(parent: Optional[QWidget], name: str) -> str:
    """「儲存 / 不儲存 / 取消」 for unsaved changes; returns ``"save"``, ``"discard"`` or ``"cancel"``."""
    box = QMessageBox(QMessageBox.Warning, "尚未儲存", f"「{name}」有尚未儲存的變更。\n要先儲存嗎？",
                      QMessageBox.NoButton, parent)
    save = box.addButton("儲存", QMessageBox.AcceptRole)
    discard = box.addButton("不儲存", QMessageBox.DestructiveRole)
    cancel = box.addButton("取消", QMessageBox.RejectRole)
    box.setDefaultButton(save)
    box.setEscapeButton(cancel)
    box.exec()
    clicked = box.clickedButton()
    return "save" if clicked is save else "discard" if clicked is discard else "cancel"


def about_html() -> str:
    return (f"<h3>{APP_NAME} {__version__}</h3>"
            "<p>桌面自動剪輯軟體：把 16:9 長影片自動剪成 9:16 短影音、把很多支毛片自動剪成一支長片，"
            "剪完還能像一般剪輯軟體一樣在時間軸上逐段修正。</p>"
            "<p>授權：MIT License。</p>"
            "<p>自動剪輯的方法改編自 <b>video-autopilot-kit</b>（Hao0321 Studio，MIT 授權），感謝原作者。</p>"
            "<p>使用的開源元件：PySide6（Qt）、FFmpeg、OpenCV、NumPy。軟體執行時預設不連網，也不會上傳你的素材（安裝時才需要連網下載套件與 ffmpeg；語音模型要經你同意才會下載）。</p>"
            "<p>機器只做「技術檢查」；剪得好不好，永遠由你審片決定。</p>")


# ============================================================================ autosave / recovery
@dataclass
class Autosave:
    """An autosave file found on disk."""
    path: Path
    project_id: str
    name: str
    original: str           # the project file it belongs to ("" when the project was never saved)
    saved_at: float         # file mtime


def autosave_dir() -> Path:
    d = app_data_dir() / "autosave"
    d.mkdir(parents=True, exist_ok=True)
    return d


def autosave_path(project_id: str) -> Path:
    return autosave_dir() / f"{project_id}{PROJECT_EXT}"


def write_autosave(project: Project) -> Path:
    """Atomically write the project (plus ``autosave_of`` = its file path) to ``autosave/<project id>.acproj``.
    ``project.path`` is left alone (unlike ``Project.save``)."""
    data = project.to_dict()
    data["autosave_of"] = project.path or ""
    target = autosave_path(project.id)
    atomic_write_text(target, json.dumps(data, ensure_ascii=False))
    return target


def remove_autosave(project_id: str) -> None:
    try:
        autosave_path(project_id).unlink()
    except OSError:
        pass


def quarantine_autosave(info: "Autosave") -> Optional[Path]:
    """An autosave that cannot be read is moved aside (not deleted - it may still hold something a person could rescue by hand)
    so that it is not offered again at every start.  Returns where it went, ``None`` when it could not be moved."""
    target = info.path.with_name(info.path.name + ".unreadable")
    n = 1
    while target.exists():
        n += 1
        target = info.path.with_name(f"{info.path.name}.unreadable{n}")
    try:
        info.path.replace(target)
    except OSError:
        return None
    return target


# What a project file or an autosave that is damaged can raise while it is read (``Project.load`` turns the structural ones into
# ``ProjectFileError``; the rest is a second line of defence - the window must never crash on a bad file).
UNREADABLE = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError)


def list_autosaves() -> list[Autosave]:
    """All readable autosaves, newest first."""
    found: list[Autosave] = []
    for f in autosave_dir().glob(f"*{PROJECT_EXT}"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            found.append(Autosave(f, str(data.get("id") or f.stem), str(data.get("name") or f.stem),
                                  str(data.get("autosave_of") or ""), f.stat().st_mtime))
        except (OSError, ValueError, AttributeError):
            continue
    return sorted(found, key=lambda a: a.saved_at, reverse=True)


def is_newer_than_file(info: Autosave) -> bool:
    """True when the autosave holds work the project file does not (never saved, file gone, or file older)."""
    if not info.original or not os.path.isfile(info.original):
        return True
    return info.saved_at > os.path.getmtime(info.original) + 1.0


def find_recoverable() -> list[Autosave]:
    return [a for a in list_autosaves() if is_newer_than_file(a)]


def format_time(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


# ============================================================================ small widgets
class DropOverlay(QWidget):
    """Translucent 「放開以匯入」 cover while files are dragged over the window (never takes mouse / drag events)."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, _ev) -> None:                   # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(12, 28, 52, 190))
        frame = QRectF(self.rect()).adjusted(18, 18, -18, -18)
        p.setPen(QPen(QColor(theme.ACCENT), 3, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(frame, 14, 14)
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(28)
        p.setFont(font)
        p.setPen(QColor("#ffffff"))
        p.drawText(frame.adjusted(0, 0, 0, -30), Qt.AlignCenter, "放開以匯入")
        font.setBold(False)
        font.setPointSizeF(12)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(frame.adjusted(0, 70, 0, 0), Qt.AlignHCenter | Qt.AlignVCenter,
                   "影片、音訊、圖片或整個資料夾　·　.srt / .ass / .vtt 會當成字幕")
        p.end()


class SuggestionBanner(QFrame):
    """Dismissible import suggestion (GUI_SPEC §4).  Inline above the monitor, so it never blocks editing.

    ``actionChosen(action, kind, asset_ids)``: action is ``auto`` (open the auto-edit dialog), ``place`` (put the
    assets on the timeline) or ``skip``.
    """

    actionChosen = Signal(str, str, list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("suggestionBanner")
        self.setStyleSheet(f"#suggestionBanner {{ background: {theme.ACCENT_DARK}; border: 1px solid {theme.ACCENT};"
                           f" border-radius: 6px; }} #suggestionBanner QLabel {{ background: transparent; }}")
        self.kind = ""
        self.asset_ids: list[str] = []
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.btn_auto = QPushButton()
        self.btn_auto.setObjectName("primary")
        self.btn_place = QPushButton()
        self.btn_skip = QPushButton("略過")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 8, 8)
        lay.addWidget(self.label, 1)
        for b, action in ((self.btn_auto, "auto"), (self.btn_place, "place"), (self.btn_skip, "skip")):
            lay.addWidget(b)
            b.clicked.connect(lambda _c=False, a=action: self.actionChosen.emit(a, self.kind, list(self.asset_ids)))
        self.hide()

    def show_suggestion(self, kind: str, assets: list[Asset]) -> None:
        """``kind``: ``shorts`` (one long landscape video) or ``assemble`` (two or more videos)."""
        self.kind, self.asset_ids = kind, [a.id for a in assets]
        if kind == "shorts":
            a = assets[0]
            shape = " 16:9 " if a.height and abs(a.width / a.height - 16 / 9) < 0.05 else "橫式"
            self.label.setText(f"偵測到 1 支{shape}長影片「{a.name}」（{human_duration(a.duration)}），"
                               "要自動剪成 9:16 短影音嗎？")
            self.btn_auto.setText("自動剪成短影音")
            self.btn_place.setText("放到時間軸")
        else:
            self.label.setText(f"偵測到 {len(assets)} 支素材，要自動剪成一支長片嗎？")
            self.btn_auto.setText("自動合併成長片")
            self.btn_place.setText("依順序放到時間軸")
        self.show()

    def dismiss(self) -> None:
        self.kind, self.asset_ids = "", []
        self.hide()


class SequenceTabs(QWidget):
    """Sequence tab bar above the timeline: switch, 新增 / 複製 / 刪除 / 重新命名 (all undoable, via ops)."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._key: list = []
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.tabs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_add = QToolButton()
        self.btn_add.setText("＋ 新增序列")
        self.btn_add.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_add)
        for label, w, h in SEQUENCE_PRESETS:
            menu.addAction(label).triggered.connect(lambda _c=False, w=w, h=h: self.new_sequence(w, h))
        self.btn_add.setMenu(menu)
        self.btn_dup = QToolButton()
        self.btn_dup.setText("複製")
        self.btn_del = QToolButton()
        self.btn_del.setText("刪除")
        self.btn_rename = QToolButton()
        self.btn_rename.setText("重新命名")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 0)
        lay.addWidget(self.tabs, 1)
        for b in (self.btn_add, self.btn_dup, self.btn_rename, self.btn_del):
            lay.addWidget(b)
        self.btn_dup.clicked.connect(lambda: self.duplicate())
        self.btn_del.clicked.connect(lambda: self.delete())
        self.btn_rename.clicked.connect(lambda: self.rename())
        self.tabs.currentChanged.connect(self._on_current)
        self.tabs.tabBarDoubleClicked.connect(lambda i: self.rename(self._id_at(i)))
        self.tabs.customContextMenuRequested.connect(self._tab_menu)
        state.edited.connect(lambda *_: self.refresh())
        state.sequence_changed.connect(self.refresh)
        state.project_replaced.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ view
    def _ids(self) -> list[str]:
        return [s.id for s in self.state.project.sequences]

    def _id_at(self, index: int) -> Optional[str]:
        ids = self._ids()
        return ids[index] if 0 <= index < len(ids) else None

    def titles(self) -> list[str]:
        return [self.tabs.tabText(i) for i in range(self.tabs.count())]

    def refresh(self) -> None:
        seqs = self.state.project.sequences
        key = [(s.id, s.name, s.aspect_label) for s in seqs]
        self.tabs.blockSignals(True)
        if key != self._key:
            self._key = key
            while self.tabs.count():
                self.tabs.removeTab(0)
            for s in seqs:
                self.tabs.addTab(f"{s.name}  ·  {s.aspect_label}")
        for i, s in enumerate(seqs):
            self.tabs.setTabToolTip(i, f"{s.name}\n{s.width}×{s.height} · {s.fps:g} fps · "
                                       f"{fmt_timecode(s.duration, s.fps)}")
        ids = self._ids()
        if self.state.seq_id in ids:
            self.tabs.setCurrentIndex(ids.index(self.state.seq_id))
        self.tabs.blockSignals(False)
        self.btn_del.setEnabled(len(seqs) > 1)

    def _on_current(self, index: int) -> None:
        sid = self._id_at(index)
        if sid:
            self.state.set_sequence(sid)

    def _tab_menu(self, pos) -> None:
        sid = self._id_at(self.tabs.tabAt(pos))
        if sid is None:
            return
        menu = QMenu(self)
        menu.addAction("重新命名…").triggered.connect(lambda: self.rename(sid))
        menu.addAction("複製").triggered.connect(lambda: self.duplicate(sid))
        menu.addAction("刪除").triggered.connect(lambda: self.delete(sid))
        self._popup_menu(menu, self.tabs.mapToGlobal(pos))

    def _popup_menu(self, menu: QMenu, global_pos) -> None:
        """Seam for tests: ``QMenu.exec`` of a Shiboken instance cannot be monkeypatched on the class."""
        menu.exec(global_pos)

    # ------------------------------------------------------------------ commands
    def _next_name(self) -> str:
        names = {s.name for s in self.state.project.sequences}
        n = len(names) + 1
        while f"序列 {n}" in names:
            n += 1
        return f"序列 {n}"

    def new_sequence(self, width: int, height: int) -> Optional[str]:
        """New empty sequence of the given size (fps of the current one), made active.  Returns its id."""
        try:
            seq = self.state.project_op("新增序列", "create_sequence", self._next_name(), width, height,
                                        self.state.fps)
        except OpError:
            return None
        self.state.set_sequence(seq.id)
        return seq.id

    def duplicate(self, seq_id: Optional[str] = None) -> Optional[str]:
        try:
            new = self.state.editor.op("複製序列", "duplicate_sequence", seq_id or self.state.seq_id)
        except OpError as exc:
            self.state.status.emit(str(exc))
            return None
        self.state.set_sequence(new.id)
        return new.id

    def delete(self, seq_id: Optional[str] = None) -> bool:
        seq_id = seq_id or self.state.seq_id
        seq = next((s for s in self.state.project.sequences if s.id == seq_id), None)
        if seq is None or len(self.state.project.sequences) <= 1:
            self.state.status.emit("至少要保留一個序列")
            return False
        if not ask(self, "刪除序列", f"要刪除序列「{seq.name}」嗎？（可以用 Ctrl+Z 復原）", "刪除"):
            return False
        try:
            self.state.project_op("刪除序列", "delete_sequence", seq_id)
        except OpError:
            return False
        return True

    def rename(self, seq_id: Optional[str] = None) -> bool:
        seq_id = seq_id or self.state.seq_id
        seq = next((s for s in self.state.project.sequences if s.id == seq_id), None)
        if seq is None:
            return False
        name, ok = QInputDialog.getText(self, "重新命名序列", "序列名稱：", text=seq.name)
        if not ok or not name.strip() or name.strip() == seq.name:
            return False
        try:
            self.state.editor.op("重新命名序列", "rename_sequence", seq_id, name)
        except OpError as exc:
            self.state.status.emit(str(exc))
            return False
        return True


# ============================================================================ environment check
ENV_STATUS = {"ok": ("✔ 正常", theme.OK), "警告": ("▲ 警告", theme.WARN), "錯誤": ("✘ 錯誤", theme.ERROR),
              "選用": ("– 選用", theme.TEXT_DIM), "未執行": ("– 尚未實測", theme.TEXT_DIM)}
ENV_MODULES = (("numpy", "numpy", True), ("cv2", "OpenCV", True), ("PySide6", "PySide6（桌面介面）", True),
               ("faster_whisper", "faster-whisper（自動字幕）", False), ("opencc", "opencc（簡轉繁）", False))


def check_environment() -> list[tuple[str, str, str]]:
    """Rows ``(item, status, detail)`` like ``autocut doctor``; status is ok | 警告 | 錯誤 | 選用 | 未執行 (not tried yet:
    never shown as 「正常」)."""
    from ..media.ffmpeg import detect_hw_encoder, ffmpeg_info
    rows = [("資料夾", "ok", str(app_data_dir()))]
    try:
        info = ffmpeg_info()
        rows.append(("ffmpeg", "ok", f"{info.version_text.split(' Copyright')[0]}（{info.ffmpeg}）"))
        rows.append(("libass（燒入字幕）", "ok" if info.has_ass else "警告",
                     "可用" if info.has_ass else "缺少：字幕只能另存成 .srt"))
        rows.append(("libx264（CPU 編碼）", "ok" if info.has_encoder("libx264") else "錯誤", ""))
        cached = detect_hw_encoder.cache_info().currsize > 0
        rows.append(("硬體編碼器", "ok" if cached else "未執行",
                     detect_hw_encoder() if cached else "尚未實測（按「偵測硬體編碼器」）"))
    except Exception as exc:  # noqa: BLE001 - report, never crash the dialog
        rows.append(("ffmpeg", "錯誤", str(exc)))
    for module, label, required in ENV_MODULES:
        try:
            mod = importlib.import_module(module)
            rows.append((label, "ok", str(getattr(mod, "__version__", ""))))
        except Exception:  # noqa: BLE001
            hint = "未安裝" if required else "未安裝（選用：pip install -r requirements-speech.txt）"
            rows.append((label, "錯誤" if required else "選用", hint))
    return rows


class EnvironmentDialog(QDialog):
    """「檢查環境」: ffmpeg / PySide6 / OpenCV / Whisper status; the slow probes run in the background."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("檢查環境")
        self.setMinimumSize(640, 380)
        self.rows: list[tuple[str, str, str]] = []
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["狀態", "項目", "內容"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.Stretch)
        self.note = QLabel("檢查中…")
        self.note.setObjectName("dim")
        self.btn_hw = QPushButton("偵測硬體編碼器")
        self.btn_hw.setToolTip("實際用 ffmpeg 試跑顯示卡的 H.264 編碼器（可能要幾秒）")
        self.btn_hw.clicked.connect(self.detect_hardware)
        self.btn_copy = QPushButton("複製結果")
        self.btn_copy.clicked.connect(self.copy_text)
        btn_close = QPushButton("關閉")
        btn_close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(self.btn_hw)
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        row.addWidget(btn_close)
        lay = QVBoxLayout(self)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.note)
        lay.addLayout(row)
        self.runner = TaskRunner(lambda _p, _c: check_environment(), name="檢查環境")
        self.runner.finished.connect(self._fill)
        self.runner.start()

    def _paint_row(self, r: int, name: str, status: str, detail: str) -> None:
        text, color = ENV_STATUS[status]
        for c, value in enumerate((text, name, detail)):
            it = QTableWidgetItem(value)
            if c < 2:
                it.setForeground(QColor(color))
            self.table.setItem(r, c, it)

    def _fill(self, rows: list) -> None:
        self.rows = list(rows)
        self.table.setRowCount(len(rows))
        for r, (name, status, detail) in enumerate(rows):
            self._paint_row(r, name, status, detail)
        bad = [n for n, s, _ in rows if s == "錯誤"]
        self.note.setText("有項目需要處理：" + "、".join(bad) if bad else "必要的項目都正常。")

    def detect_hardware(self) -> None:
        from ..media.ffmpeg import detect_hw_encoder
        self.btn_hw.setEnabled(False)
        self.note.setText("正在偵測硬體編碼器…")
        runner = TaskRunner(lambda _p, _c: detect_hw_encoder(), name="偵測硬體編碼器")
        runner.finished.connect(self._hardware_found)
        runner.failed.connect(self._hardware_failed)
        runner.start()

    def _set_hardware_row(self, status: str, detail: str) -> None:
        for r, (name, _s, _d) in enumerate(self.rows):
            if name == "硬體編碼器":
                self.rows[r] = (name, status, detail)
                self._paint_row(r, name, status, detail)

    def _hardware_found(self, result: str) -> None:
        self.btn_hw.setEnabled(True)
        self._set_hardware_row("ok", str(result))
        self.note.setText(f"硬體編碼器偵測結果：{result}")

    def _hardware_failed(self, message: str) -> None:
        """The probe itself broke: that is not a result (and certainly not 「正常」)."""
        self.btn_hw.setEnabled(True)
        self._set_hardware_row("警告", f"偵測沒有成功：{message}")
        self.note.setText(f"硬體編碼器沒有偵測成功：{message}")

    def text(self) -> str:
        return "\n".join(f"{ENV_STATUS[s][0]}  {n}  {d}" for n, s, d in self.rows)

    def copy_text(self) -> None:
        QGuiApplication.clipboard().setText(self.text())


# ============================================================================ the window
class MainWindow(QMainWindow):
    """AutoCut Studio's main window.  ``MainWindow(state=None, paths=None)``: ``paths`` may hold media files, folders and
    one ``.acproj`` (opened first); they are handled right after the event loop starts."""

    def __init__(self, state: Optional[AppState] = None, paths: Optional[list[str]] = None) -> None:
        super().__init__()
        self.state = state if state is not None else AppState()
        self.settings = app_settings()
        self._start_paths = list(paths or [])
        self._started = False
        self._last_missing_check = 0.0
        self.setAcceptDrops(True)
        if QApplication.windowIcon().isNull():
            self.setWindowIcon(build_app_icon())

        self.importer = ImportController(self.state, self)
        self.playback = PlaybackController(self.state, self)
        self.monitor = MonitorWidget(self.state, self.playback)
        self.timeline = TimelineWidget(self.state)
        self.inspector = InspectorPanel(self.state)
        self.media_bin = MediaBin(self.state, self.importer)
        self.banner = SuggestionBanner()
        self.sequence_tabs = SequenceTabs(self.state)
        self.overlay = DropOverlay(self)

        self._build_central()
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_statusbar()
        self._connect()
        keep_keyboard_on_timeline(self)          # a click on a button / slider / combo does not take the keys from the timeline
        self._default_state = self.saveState()
        self.setMinimumSize(*MIN_WINDOW)
        screen = self.screen() or QGuiApplication.primaryScreen()
        self.resize(default_window_size(screen.availableGeometry()) if screen else QSize(*IDEAL_WINDOW))
        self._restore_settings()
        self._update_title()
        self._sync_actions()
        self._update_seq_info()

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()
        QTimer.singleShot(0, self.startup)

    # ------------------------------------------------------------------ layout
    def _build_central(self) -> None:
        top = QWidget()
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(4)
        set_banner = getattr(self.monitor, "set_banner_widget", None)
        if callable(set_banner):
            set_banner(self.banner)
        else:
            top_lay.addWidget(self.banner)
        top_lay.addWidget(self.monitor, 1)
        bottom = QWidget()
        bottom_lay = QVBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(0)
        bottom_lay.addWidget(self.sequence_tabs)
        bottom_lay.addWidget(self.timeline, 1)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setObjectName("mainSplitter")
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(top)
        self.splitter.addWidget(bottom)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes(SPLIT_SIZES)
        self.setCentralWidget(self.splitter)

    def _build_docks(self) -> None:
        self.dock_bin = QDockWidget("素材庫", self)
        self.dock_bin.setObjectName("dockMediaBin")
        self.dock_bin.setWidget(self.media_bin)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_bin)
        self.dock_inspector = QDockWidget("檢查器", self)
        self.dock_inspector.setObjectName("dockInspector")
        self.dock_inspector.setWidget(self.inspector)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_inspector)
        self.resizeDocks([self.dock_bin, self.dock_inspector], [300, 380], Qt.Horizontal)

    # ------------------------------------------------------------------ actions
    def _act(self, text: str, slot: Callable[[], object], shortcuts: tuple[str, ...] = (), tip: str = "") -> QAction:
        act = QAction(text, self)
        if shortcuts:
            act.setShortcuts([QKeySequence(s) for s in shortcuts])
        if tip:
            act.setToolTip(tip)
            act.setStatusTip(tip)
        act.triggered.connect(lambda _checked=False: slot())
        return act

    def _timeline_act(self, text: str, method: str, *args) -> QAction:
        """Menu entry for a timeline command: shortcut shown as text only (the timeline owns the key)."""
        return self._act(text, lambda: self.timeline_call(method, *args))

    def _build_actions(self) -> None:
        a = self._act
        self.act_new = a("新增專案", self.new_project, ("Ctrl+N",))
        self.act_open = a("開啟專案…", self.open_project_dialog, ("Ctrl+O",))
        self.act_save = a("儲存專案", self.save_project, ("Ctrl+S",))
        self.act_save_as = a("另存新檔…", self.save_project_as, ("Ctrl+Shift+S",))
        self.act_import = a("匯入素材", self.import_dialog, ("Ctrl+I",), "匯入影片、音訊、圖片（也可以直接把檔案拖進視窗）")
        self.act_import_folder = a("匯入資料夾…", self.import_folder_dialog)
        self.act_export = a("輸出影片", self.show_export_dialog, ("Ctrl+E",))
        self.act_quit = a("結束", self.close, ("Ctrl+Q",))
        self.act_undo = a("復原", self.undo, ("Ctrl+Z",))
        self.act_redo = a("重做", self.redo, ("Ctrl+Y", "Ctrl+Shift+Z"))
        self.act_play = a("播放／暫停", self.toggle_playback, ("Space",))
        self.act_preview = a("渲染預覽（含聲音）", self.render_preview, ("Ctrl+Return", "Ctrl+Enter"))
        self.act_fullscreen = a("全螢幕", self.toggle_fullscreen, ("F11",))
        self.act_fullscreen.setCheckable(True)
        self.act_reset_layout = a("重設版面", self.reset_layout)
        self.act_snap = QAction("吸附", self)
        self.act_snap.setCheckable(True)
        self.act_snap.setChecked(True)
        self.act_snap.toggled.connect(lambda on: self.timeline_call("set_snapping", on))
        self.act_shorts = a("自動剪成短影音…", self.open_shorts_dialog)
        self.act_assemble = a("自動合併成長片…", self.open_assemble_dialog)
        self.act_longform = a("長片精華…", self.open_longform_dialog)
        self.act_longform.setToolTip("一支長影片 → 精華長片：留重點、剪掉「嗯、呃、那個」、兩人對談時說話的人自動放大")
        self.act_add_vtrack = a("新增視訊軌", lambda: self.add_track("video"))
        self.act_add_atrack = a("新增音訊軌", lambda: self.add_track("audio"))
        self.act_about = a("關於 AutoCut Studio", self.show_about)
        self.act_env = a("檢查環境…", self.show_environment)
        self.act_data_dir = a("開啟資料與記錄資料夾", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(app_data_dir()))))
        t = self._timeline_act
        self.timeline_acts = {
            "split": t("在播放頭分割\tS", "split_at_playhead"), "delete": t("刪除\tDelete", "delete_selection", False),
            "ripple": t("波紋刪除\tShift+Delete", "delete_selection", True), "marker": t("加入標記\tM", "add_marker_at_playhead"),
            "in": t("設定入點\tI", "set_in_at_playhead"), "out": t("設定出點\tO", "set_out_at_playhead"),
            "clear_range": t("清除入出點", "clear_range"), "prev_cut": t("上一個剪輯點\t↑", "goto_prev_cut"),
            "next_cut": t("下一個剪輯點\t↓", "goto_next_cut"), "back": t("後退一格\t←", "step_frames", -1),
            "forward": t("前進一格\t→", "step_frames", 1), "zoom_in": t("放大\t+", "zoom_in"),
            "zoom_out": t("縮小\t-", "zoom_out"), "zoom_fit": t("符合視窗\t\\", "zoom_fit"),
            "select_all": t("全選\tCtrl+A", "select_all"),
        }

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("檔案(&F)")
        for act in (self.act_new, self.act_open):
            file_menu.addAction(act)
        self.recent_menu = file_menu.addMenu("開啟最近的專案")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        file_menu.addSeparator()
        for act in (self.act_import, self.act_import_folder):
            file_menu.addAction(act)
        file_menu.addSeparator()
        for act in (self.act_save, self.act_save_as):
            file_menu.addAction(act)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        edit = bar.addMenu("編輯(&E)")
        for act in (self.act_undo, self.act_redo):
            edit.addAction(act)
        edit.addSeparator()
        edit.addAction(self.timeline_acts["select_all"])

        tl = bar.addMenu("時間軸(&T)")
        order = (("split", "delete", "ripple"), ("marker", "in", "out", "clear_range"),
                 ("prev_cut", "next_cut", "back", "forward"), ("zoom_in", "zoom_out", "zoom_fit"))
        tl.addAction(self.act_play)
        for group in order:
            tl.addSeparator()
            for key in group:
                tl.addAction(self.timeline_acts[key])
        tl.addSeparator()
        tl.addAction(self.act_snap)
        tl.addSeparator()
        for act in (self.act_add_vtrack, self.act_add_atrack):
            tl.addAction(act)

        auto = bar.addMenu("自動剪輯(&A)")
        for act in (self.act_shorts, self.act_longform, self.act_assemble):
            auto.addAction(act)

        view = bar.addMenu("檢視(&V)")
        for dock, text in ((self.dock_bin, "素材庫"), (self.dock_inspector, "檢查器")):
            toggle = dock.toggleViewAction()
            toggle.setText(text)
            view.addAction(toggle)
        view.addSeparator()
        for act in (self.act_preview, self.act_fullscreen, self.act_reset_layout):
            view.addAction(act)

        help_menu = bar.addMenu("說明(&H)")
        for act in (self.act_env, self.act_data_dir):
            help_menu.addAction(act)
        help_menu.addSeparator()
        help_menu.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        tb = QToolBar("主要工具列", self)
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toolbar = tb
        self.addToolBar(Qt.TopToolBarArea, tb)
        for act in (self.act_import, self.act_save):
            tb.addAction(act)
        tb.addSeparator()
        for act in (self.act_undo, self.act_redo):
            tb.addAction(act)
        tb.addSeparator()
        split = self._act("分割", lambda: self.timeline_call("split_at_playhead"), tip="在播放頭分割（S）")
        delete = self._act("刪除", lambda: self.timeline_call("delete_selection", False), tip="刪除選取的片段（Delete）")
        for act in (split, delete, self.act_snap):
            tb.addAction(act)
        tb.addSeparator()
        for act in (self.act_shorts, self.act_longform, self.act_assemble):
            tb.addAction(act)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self.act_export)

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        self.seq_info = QLabel()
        self.seq_info.setObjectName("dim")
        self.import_box = QWidget()
        row = QHBoxLayout(self.import_box)
        row.setContentsMargins(0, 0, 0, 0)
        self.import_label = QLabel("匯入中")
        self.import_bar = QProgressBar()
        self.import_bar.setRange(0, 100)
        self.import_bar.setFixedWidth(160)
        self.import_cancel = QToolButton()
        self.import_cancel.setText("取消匯入")
        self.import_cancel.clicked.connect(self.cancel_imports)
        for w in (self.import_label, self.import_bar, self.import_cancel):
            row.addWidget(w)
        self.import_box.hide()
        bar.addPermanentWidget(self.import_box)
        bar.addPermanentWidget(self.seq_info)

    def _connect(self) -> None:
        s, imp = self.state, self.importer
        s.status.connect(lambda text: self.statusBar().showMessage(text, STATUS_MS))
        s.title_changed.connect(lambda *_: self._update_title())
        s.dirty_changed.connect(lambda *_: self._update_title())
        s.edited.connect(lambda *_: self._on_edited())
        s.sequence_changed.connect(self._update_seq_info)
        s.project_replaced.connect(self.banner.dismiss)
        imp.imported.connect(self._on_imported)
        imp.suggestion.connect(self._on_suggestion)
        imp.subtitleFilesDropped.connect(self.apply_subtitle_files)
        bin_ = self.media_bin
        bin_.shortsRequested.connect(self.open_shorts_dialog)
        bin_.assembleRequested.connect(self.open_assemble_dialog)
        bin_.longformRequested.connect(self.open_longform_dialog)
        bin_.importRequested.connect(self.import_dialog)
        bin_.importStarted.connect(self.track_import)
        self.banner.actionChosen.connect(self._on_banner_action)
        self.timeline.externalFilesDropped.connect(self._on_timeline_files)
        self.playback.playing_changed.connect(self._on_playing_changed)

    # ------------------------------------------------------------------ small state sync
    def _update_title(self) -> None:
        self.setWindowTitle(f"{self.state.title} — {APP_NAME}")

    def _on_edited(self) -> None:
        self._sync_actions()
        self._update_seq_info()

    def _sync_actions(self) -> None:
        h = self.state.history
        self.act_undo.setEnabled(h.can_undo)
        self.act_redo.setEnabled(h.can_redo)
        self.act_undo.setToolTip(f"復原：{h.undo_label}" if h.can_undo else "沒有可復原的動作")
        self.act_redo.setToolTip(f"重做：{h.redo_label}" if h.can_redo else "沒有可重做的動作")

    def _update_seq_info(self) -> None:
        try:
            seq = self.state.sequence()
        except KeyError:
            self.seq_info.setText("")
            return
        self.seq_info.setText(f"{seq.name} · {seq.width}×{seq.height} · {seq.fps:g} fps · "
                              f"{fmt_timecode(seq.duration, seq.fps)}")

    def _on_playing_changed(self, playing: bool) -> None:
        fn = getattr(self.timeline, "set_playing", None)
        if callable(fn):
            fn(playing)

    def timeline_call(self, method: str, *args) -> None:
        """Call a public :class:`TimelineWidget` command (§11.1); says so on the status line if it is missing."""
        fn = getattr(self.timeline, method, None)
        if callable(fn):
            fn(*args)
        else:
            self.state.status.emit("這個版本的時間軸還不支援這個指令")

    def focus_timeline(self) -> None:
        """Where the keyboard goes after a value was committed in a box or a control was clicked (``widgets.release_focus``)."""
        self.timeline.setFocus(Qt.OtherFocusReason)

    def keyPressEvent(self, ev) -> None:                          # noqa: N802 (Qt API)
        """A key the widget with the keyboard did not use (a button reached with Tab, the preview, the empty window ...) is a
        timeline key: S splits, Delete deletes, the arrows step ... as if the timeline had the keyboard.  Boxes the person
        types into, number boxes and lists / tables keep their keys (``keeps_its_keys``)."""
        focus = QApplication.focusWidget()
        if not keeps_its_keys(focus) and not (focus is not None and (focus is self.timeline
                                                                      or self.timeline.isAncestorOf(focus))):
            self.timeline.keyPressEvent(ev)
            if ev.isAccepted():
                return
        super().keyPressEvent(ev)

    def undo(self) -> None:
        self.state.undo()

    def redo(self) -> None:
        self.state.redo()

    def toggle_playback(self) -> None:
        self.playback.toggle()

    def render_preview(self) -> None:
        for owner in (self.monitor, self.playback):
            fn = getattr(owner, "render_preview", None)
            if callable(fn):
                fn()
                return
        self.state.status.emit("這個版本還沒有預覽渲染功能")

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self.act_fullscreen.setChecked(self.isFullScreen())

    def reset_layout(self) -> None:
        self.restoreState(self._default_state)
        self.resizeDocks([self.dock_bin, self.dock_inspector], [300, 380], Qt.Horizontal)
        self.splitter.setSizes(SPLIT_SIZES)

    def add_track(self, kind: str) -> None:
        try:
            self.state.op("新增視訊軌" if kind == "video" else "新增音訊軌", "add_track", kind)
        except OpError:
            pass

    # ------------------------------------------------------------------ files: new / open / save
    @property
    def project_title(self) -> str:
        return os.path.basename(self.state.project_path) or self.state.project.name

    def confirm_discard(self) -> bool:
        """True when it is fine to replace / close the project (saved, discarded or nothing to save)."""
        if not self.state.dirty:
            return True
        choice = ask_save_changes(self, self.project_title)
        if choice == "save":
            return self.save_project()
        return choice == "discard"

    def _replace_project(self, project: Project) -> None:
        old = self.state.project
        if old.id != project.id:
            remove_autosave(old.id)           # the user already decided what to do with the old project
        self.state.replace_project(project)

    def new_project(self) -> bool:
        if not self.confirm_discard():
            return False
        self._replace_project(Project.new("未命名專案"))
        self.state.status.emit("已建立新專案")
        return True

    def open_project_dialog(self) -> None:
        start = str(self.settings.value("dialog/project_dir", "") or "")
        path, _ = QFileDialog.getOpenFileName(self, "開啟專案", start, f"AutoCut 專案 (*{PROJECT_EXT});;所有檔案 (*)")
        if path:
            self.open_project(path)

    def open_project(self, path: str) -> bool:
        """Open a ``.acproj`` (asks about unsaved changes first, and offers a newer autosave)."""
        if not self.confirm_discard():
            return False
        try:
            project = Project.load(path)
        except UNREADABLE as exc:
            inform(self, "無法開啟專案", f"「{os.path.basename(path)}」不是有效的 AutoCut 專案，或檔案已損毀。\n\n{exc}",
                   warning=True)
            return False
        recovered = self._offer_newer_autosave(project, path)
        project = recovered or project
        self._replace_project(project)
        if recovered is not None:
            self._mark_dirty()
        self._remember_project(path)
        gone = sum(1 for a in project.assets.values() if a.missing)
        self.state.status.emit(f"已開啟 {os.path.basename(path)}" + (
            f"（有 {gone} 個素材找不到檔案，素材庫會用紅色標示，可按右鍵重新連結）" if gone else ""))
        return True

    def _offer_newer_autosave(self, project: Project, path: str) -> Optional[Project]:
        """A newer autosave of this project (it was not saved before the last exit)? Ask, then load it."""
        info = next((a for a in list_autosaves() if a.project_id == project.id), None)
        if info is None or not is_newer_than_file(info):
            return None
        if ask(self, "復原自動存檔", f"找到比「{os.path.basename(path)}」更新的自動存檔（{format_time(info.saved_at)}）。\n"
                                    "可能是上次沒有儲存就結束了。要復原自動存檔的內容嗎？", "復原", "使用原檔"):
            recovered = self._load_autosave(info)
            if recovered is not None:
                recovered.path = path
                return recovered
            return None
        remove_autosave(info.project_id)
        return None

    def _load_autosave(self, info: Autosave) -> Optional[Project]:
        try:
            return Project.from_dict(json.loads(info.path.read_text(encoding="utf-8")))
        except UNREADABLE as exc:
            moved = quarantine_autosave(info)
            where = f"\n\n已把它移到「{moved.name}」（不會再詢問；內容沒有刪除）。" if moved else ""
            inform(self, "自動存檔無法讀取", f"{info.path.name} 已損毀：{exc}{where}", warning=True)
            return None

    def _mark_dirty(self) -> None:
        """Recovered work counts as unsaved changes (closing asks first)."""
        self.state.project.touch()
        self.state.dirty_changed.emit(self.state.dirty)
        self.state.title_changed.emit(self.state.title)

    def offer_recovery(self) -> bool:
        """Startup: offer the newest autosave that holds unsaved work.  Returns True when a project was recovered."""
        found = find_recoverable()
        if not found:
            return False
        info = found[0]
        where = f"\n原檔：{info.original}" if info.original else "\n（這個專案還沒存過檔）"
        if not ask(self, "復原專案", f"發現上次沒有儲存就結束的專案「{info.name}」（自動存檔於 {format_time(info.saved_at)}）。"
                                    f"{where}\n\n要復原嗎？", "復原", "捨棄"):
            remove_autosave(info.project_id)
            return False
        project = self._load_autosave(info)
        if project is None:
            return False
        project.path = info.original or None
        self._replace_project(project)
        self._mark_dirty()
        self.state.status.emit("已復原自動存檔，請記得儲存")
        return True

    def save_project(self) -> bool:
        if not self.state.project_path:
            return self.save_project_as()
        return self._write_project(None)

    def save_project_as(self) -> bool:
        folder = str(self.settings.value("dialog/project_dir", "") or "")
        if not folder or not os.path.isdir(folder):
            folder = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or str(Path.home())
        default = self.state.project_path or os.path.join(folder, safe_stem(self.state.project.name) + PROJECT_EXT)
        path, _ = QFileDialog.getSaveFileName(self, "另存專案", default, f"AutoCut 專案 (*{PROJECT_EXT})")
        if not path:
            return False
        return self._write_project(path if path.lower().endswith(PROJECT_EXT) else path + PROJECT_EXT)

    def _write_project(self, path: Optional[str]) -> bool:
        try:
            target = self.state.save_project(path)
        except (OSError, ValueError) as exc:
            inform(self, "無法儲存專案", f"儲存失敗：{exc}", warning=True)
            return False
        remove_autosave(self.state.project.id)
        self._remember_project(target)
        return True

    # ------------------------------------------------------------------ recent files
    def recent_projects(self) -> list[str]:
        return [p for p in _as_list(self.settings.value("recent/projects", [])) if p]

    def _remember_project(self, path: str) -> None:
        path = os.path.abspath(path)
        recent = [path] + [p for p in self.recent_projects() if os.path.normcase(p) != os.path.normcase(path)]
        self.settings.setValue("recent/projects", recent[:MAX_RECENT])
        self.settings.setValue("dialog/project_dir", os.path.dirname(path))

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self.recent_projects() if os.path.isfile(p)]
        self.settings.setValue("recent/projects", recent)
        for path in recent:
            act = self.recent_menu.addAction(os.path.basename(path))
            act.setToolTip(path)
            act.triggered.connect(lambda _c=False, p=path: self.open_project(p))
        if not recent:
            self.recent_menu.addAction("（沒有最近的專案）").setEnabled(False)
        else:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction("清除記錄").triggered.connect(
                lambda: self.settings.setValue("recent/projects", []))

    # ------------------------------------------------------------------ autosave
    def autosave(self) -> Optional[Path]:
        """Write ``autosave/<project id>.acproj`` when there are unsaved changes (called every 60 s)."""
        if not self.state.dirty:
            return None
        try:
            return write_autosave(self.state.project)
        except OSError:
            return None

    # ------------------------------------------------------------------ startup
    def _is_pristine(self) -> bool:
        p = self.state.project
        return not self.state.dirty and not p.assets and not any(True for s in p.sequences for _ in s.all_clips())

    def startup(self) -> None:
        """Handle the command-line paths and offer autosave recovery (runs once, after the loop started)."""
        if self._started:
            return
        self._started = True
        projects = [p for p in self._start_paths if is_project_file(p)]
        media = [p for p in self._start_paths if not is_project_file(p)]
        if projects:
            self.open_project(projects[0])
        elif self._is_pristine():
            self.offer_recovery()
        if media:
            self.import_paths(media)

    # ------------------------------------------------------------------ import
    def import_paths(self, paths: list[str], **kwargs) -> TaskRunner:
        """Import files / folders (background) and show the progress in the status bar."""
        runner = self.importer.import_paths(paths, **kwargs)
        self.track_import(runner)
        return runner

    def track_import(self, runner: TaskRunner) -> None:
        """Status-bar progress of an import (the importer is also driven by the media bin and the timeline)."""
        self.import_bar.setValue(0)
        self.import_label.setText("匯入中")
        self.import_box.show()
        runner.progress.connect(self._on_import_progress)
        runner.done.connect(self._on_import_done)

    def _on_import_progress(self, fraction: float, message: str) -> None:
        self.import_bar.setValue(int(fraction * 100))
        self.statusBar().showMessage(message, STATUS_MS)

    def _on_import_done(self) -> None:
        if not self.importer.busy:
            self.import_box.hide()

    def cancel_imports(self) -> None:
        for runner in list(self.importer.runners):
            runner.cancel()

    def import_dialog(self) -> None:
        start = str(self.settings.value("dialog/import_dir", "") or "")
        paths, _ = QFileDialog.getOpenFileNames(self, "匯入素材", start, file_dialog_filter(subtitles=True))
        if paths:
            self.settings.setValue("dialog/import_dir", os.path.dirname(paths[0]))
            self.import_paths(paths)

    def import_folder_dialog(self) -> None:
        start = str(self.settings.value("dialog/import_dir", "") or "")
        folder = QFileDialog.getExistingDirectory(self, "匯入資料夾", start)
        if folder:
            self.settings.setValue("dialog/import_dir", folder)
            self.import_paths([folder])

    def handle_dropped_paths(self, paths: list[str]) -> None:
        """Files dropped on the window: a ``.acproj`` is opened, everything else is imported."""
        projects = [p for p in paths if is_project_file(p)]
        media = [p for p in paths if not is_project_file(p)]
        if projects and not self.open_project(projects[0]):
            return
        if media:
            self.import_paths(media)

    def _on_imported(self, _ids: list, failures: list) -> None:
        if failures:
            shown = "\n".join(f"．{m}" for m in failures[:10]) + ("\n…" if len(failures) > 10 else "")
            inform(self, "部分檔案無法匯入", f"其他檔案已經匯入。這些沒辦法匯入：\n{shown}", warning=True)

    def _on_timeline_files(self, paths: list, t: float, track_id: str) -> None:
        """OS files dropped on the timeline: import first, then put them at the drop position (one undo step)."""
        runner = self.importer.import_paths(
            paths, suggest=False,
            on_imported=lambda ids, _fails: self.importer.place_assets(ids, track_id or None, t, "overwrite"))
        self.track_import(runner)

    # ------------------------------------------------------------------ suggestion banner
    def _on_suggestion(self, kind: str, ids: list) -> None:
        assets = [self.state.project.assets[i] for i in ids if i in self.state.project.assets]
        if kind in ("shorts", "assemble") and assets:
            self.banner.show_suggestion(kind, assets)
            return
        self.banner.dismiss()
        self.statusBar().showMessage("已加入素材庫：雙擊素材或把它拖到時間軸就能使用", STATUS_MS)

    def _on_banner_action(self, action: str, kind: str, ids: list) -> None:
        self.banner.dismiss()
        if action == "place":
            self.importer.place_assets(ids)
        elif action == "auto" and kind == "shorts" and ids:
            self.open_shorts_dialog(ids[0])
        elif action == "auto" and kind == "assemble":
            self.open_assemble_dialog(ids)

    # ------------------------------------------------------------------ auto-edit / export dialogs
    def _video_assets(self, ids: Optional[list[str]] = None) -> list[Asset]:
        assets = self.state.project.assets
        pool = assets.values() if ids is None else [assets[i] for i in ids if i in assets]
        return [a for a in pool if a.kind == "video" and a.has_video]

    def _pick_shorts_source(self, title: str = "自動剪成短影音") -> Optional[str]:
        """Selected video in the bin, else the only video, else ask which one."""
        picked = self._video_assets(self.media_bin.selected_asset_ids())
        if len(picked) == 1:
            return picked[0].id
        videos = self._video_assets()
        if not videos:
            inform(self, title, "專案裡還沒有影片。\n請先把影片拖進視窗匯入。")
            return None
        if len(videos) == 1:
            return videos[0].id
        names = [f"{a.name}（{human_duration(a.duration)}）" for a in videos]
        name, ok = QInputDialog.getItem(self, title, "要用哪一支影片？", names, 0, False)
        return videos[names.index(name)].id if ok and name in names else None

    def open_shorts_dialog(self, asset_id: Optional[str] = None) -> None:
        """「自動剪成短影音」 for ``asset_id`` (or the selected / only / chosen video)."""
        asset_id = asset_id if isinstance(asset_id, str) and asset_id else self._pick_shorts_source()
        if asset_id:
            self._run_dialog(ShortsDialog(self.state, asset_id, self))

    def open_longform_dialog(self, asset_id: Optional[str] = None) -> None:
        """「長片精華」 for ``asset_id`` (or the selected / only / chosen video)."""
        asset_id = asset_id if isinstance(asset_id, str) and asset_id else self._pick_shorts_source("長片精華")
        if asset_id:
            self._run_dialog(LongformDialog(self.state, asset_id, self))

    def open_assemble_dialog(self, asset_ids: Optional[list[str]] = None) -> None:
        """「自動合併成長片」 for ``asset_ids`` (or the selected videos, else every video in the project)."""
        if not isinstance(asset_ids, list) or not asset_ids:
            chosen = self._video_assets(self.media_bin.selected_asset_ids())
            asset_ids = [a.id for a in (chosen if len(chosen) >= 2 else self._video_assets())]
        if len(asset_ids) < 2:
            inform(self, "自動合併成長片", "至少需要 2 支影片才能合併成一支長片。\n請先匯入更多影片。")
            return
        self._run_dialog(AssembleDialog(self.state, asset_ids, self))

    def show_export_dialog(self) -> None:
        self._run_dialog(ExportDialog(self.state, self))

    @staticmethod
    def _run_dialog(dialog) -> None:
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    # ------------------------------------------------------------------ subtitles
    def apply_subtitle_files(self, paths: list) -> bool:
        """.srt / .ass / .vtt dropped in: ask, then replace the active sequence's captions (one undo step)."""
        path = str(paths[0])
        try:
            cues = load_subtitle_file(path)
        except (OSError, ValueError) as exc:
            inform(self, "字幕讀取失敗", f"{os.path.basename(path)}：{exc}", warning=True)
            return False
        if not cues:
            inform(self, "字幕讀取失敗", f"{os.path.basename(path)} 裡沒有讀到任何字幕。", warning=True)
            return False
        seq = self.state.sequence()
        text = f"要把「{os.path.basename(path)}」的 {len(cues)} 句字幕放進序列「{seq.name}」嗎？"
        n_subs = sum(1 for q in seq.cues if not q.is_card)
        if n_subs:
            text += f"\n這會取代目前序列裡的 {n_subs} 句字幕（字卡會保留；可以用 Ctrl+Z 復原）。"
        if len(paths) > 1:
            text += f"\n（一次只能套用一個字幕檔，另外 {len(paths) - 1} 個已略過。）"
        if not ask(self, "匯入字幕", text, "套用字幕"):
            return False
        try:
            self.state.op("匯入字幕", "replace_cues", cues)
        except OpError:
            return False
        self.state.status.emit(f"已匯入 {len(cues)} 句字幕")
        return True

    # ------------------------------------------------------------------ help
    def show_about(self) -> None:
        inform(self, "關於 AutoCut Studio", about_html())

    def show_environment(self) -> None:
        self._run_dialog(EnvironmentDialog(self))

    # ------------------------------------------------------------------ drag & drop (whole window)
    def dragEnterEvent(self, ev) -> None:             # noqa: N802 (Qt API)
        if local_paths(ev.mimeData()):
            self.overlay.setGeometry(self.rect())
            self.overlay.raise_()
            self.overlay.show()
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
        self.overlay.hide()
        ev.accept()

    def dropEvent(self, ev) -> None:                  # noqa: N802
        self.overlay.hide()
        paths = local_paths(ev.mimeData())
        if not paths:
            ev.ignore()
            return
        ev.setDropAction(Qt.CopyAction)
        ev.accept()
        self.handle_dropped_paths(paths)

    def resizeEvent(self, ev) -> None:                # noqa: N802
        super().resizeEvent(ev)
        self.overlay.setGeometry(self.rect())

    def changeEvent(self, ev) -> None:                # noqa: N802
        super().changeEvent(ev)
        if ev.type() == QEvent.ActivationChange and self.isActiveWindow():
            if time.monotonic() - self._last_missing_check > MISSING_CHECK_GAP:
                self._last_missing_check = time.monotonic()
                self.media_bin.recheck_missing()          # files may have been moved while we were in the background

    # ------------------------------------------------------------------ settings / closing
    def _restore_settings(self) -> None:
        st = self.settings
        geometry, state = st.value("window/geometry"), st.value("window/state")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)
        splitter = st.value("window/splitter")
        if splitter:
            self.splitter.restoreState(splitter)
        self.media_bin.set_icon_mode(str(st.value("bin/icon_mode", "false")).lower() == "true")

    def _save_settings(self) -> None:
        st = self.settings
        st.setValue("window/geometry", self.saveGeometry())
        st.setValue("window/state", self.saveState())
        st.setValue("window/splitter", self.splitter.saveState())
        st.setValue("bin/icon_mode", self.media_bin.icon_mode())
        st.sync()

    def closeEvent(self, ev) -> None:                 # noqa: N802
        if not self.confirm_discard():
            ev.ignore()
            return
        self.autosave_timer.stop()
        self.cancel_imports()
        for owner, name in ((self.monitor, "shutdown"), (self.playback, "close")):     # stop previews / decoders
            fn = getattr(owner, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:                                     # closing must never fail on a preview helper
                    pass
        shutdown_tasks()                                              # analysis, renders, captions ...: stop and wait
        remove_autosave(self.state.project.id)      # closed on purpose (saved or discarded): nothing to recover
        self._save_settings()
        super().closeEvent(ev)
