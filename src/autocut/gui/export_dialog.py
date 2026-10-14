"""Export dialog (GUI_SPEC §7): settings -> background render -> technical QA -> human review.

Pages: 設定 (preset, output file, range, encoder, captions, loudness) -> 輸出中 (progress + cancel) -> 結果
(file facts, per-check technical-QA rows, 開啟資料夾 / 播放, and the :class:`HumanReviewBox`).

Honesty rules: the QA wording is always 「技術檢查」; it never says the edit was approved.  The only thing that can
flip the review state is the user clicking in the HumanReviewBox.  The render works on a *copy* of the project, so
editing elsewhere cannot corrupt it; the copy's sequence is the one that was active when the dialog opened.
"""
from __future__ import annotations

import errno
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer, QUrl, Signal
from shiboken6 import isValid
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton,
                               QRadioButton, QSizePolicy, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..core import plan as planmod
from ..core.project import Project, Sequence, validate_sequence
from ..core.timecode import fmt_timecode
from ..media.ffmpeg import detect_hw_encoder
from ..render.compiler import RenderError
from ..render.presets import PRESETS, get_preset, output_size
from ..render.qa import QAReport, technical_qa
from ..render.runner import RenderResult, RenderSettings, render_sequence
from ..util import human_duration, human_size
from . import theme
from .media_bin import ask, inform
from .state import AppState
from .widgets import HumanReviewBox
from .workers import TaskRunner

SETTINGS_ORG = SETTINGS_APP = "AutoCutStudio"
LAST_DIR_KEY = "export/last_dir"
DEFAULT_LUFS = -14.0
PAGE_SETTINGS, PAGE_PROGRESS, PAGE_RESULT = range(3)
HW_NAMES = {"h264_nvenc": "NVIDIA NVENC", "h264_qsv": "Intel Quick Sync（QSV）", "h264_amf": "AMD AMF"}
STATUS_ICON = {"pass": "✔", "warn": "▲", "fail": "✘", "not_run": "–"}
STATUS_COLOR = {"pass": theme.OK, "warn": theme.WARN, "fail": theme.ERROR, "not_run": theme.TEXT_DIM}
QA_HEADLINE = {
    "pass": ("技術檢查通過：檔案可以讀取，長度、尺寸、影格率符合預期。這不代表剪得好看，請你自己看過成品。", theme.OK),
    "warn": ("技術檢查有警告：請看下面標示 ▲ 的項目，再自己看過成品。", theme.WARN),
    "fail": ("技術檢查未通過：請看下面標示 ✘ 的項目，這個檔案可能有問題。", theme.ERROR),
    "not_run": ("技術檢查沒有執行。", theme.TEXT_DIM),
}
QA_PARTIAL = ("技術檢查只做了一部分：標示 – 的項目沒有執行，結果不完整。請你自己看過成品。", theme.TEXT_DIM)
REVIEW_PENDING_ROW = ("not_run", "人工審片", "尚未審片（機器不會代替你判斷好不好看）")
VERDICT_ROW = {                     # what the PERSON decided in the review box (these rows only appear after a click)
    "approved": ("pass", "人工審片", "你已標記為「通過」（這是你的判斷，不是機器的）"),
    "needs_changes": ("warn", "人工審片", "你已標記為「需修改」"),
    "rejected": ("fail", "人工審片", "你已標記為「退回」"),
    "stale": ("warn", "人工審片", "你之前的審片已過期：審片之後序列又改過了，請重新審片"),
}


# ============================================================================ pure helpers
_RESERVED_NAME = re.compile(r"(?i)(CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])")


def safe_stem(name: str) -> str:
    """A file-name stem without characters Windows rejects (illegal characters, reserved device names like ``NUL``)."""
    stem = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip(" ._") or "輸出"
    head, dot, rest = stem.partition(".")
    if _RESERVED_NAME.fullmatch(head.rstrip(" ")):          # "CON", "nul.final": Windows cannot create a file with that name
        stem = f"{head}_{dot}{rest}"
    return stem


def unique_path(folder: str | os.PathLike, stem: str, ext: str = ".mp4") -> str:
    """``folder/stem.ext``; when it exists, ``stem (2).ext``, ``stem (3).ext`` …  (never silently overwrite)."""
    base = Path(folder)
    candidate, n = base / f"{stem}{ext}", 2
    while candidate.exists():
        candidate = base / f"{stem} ({n}){ext}"
        n += 1
    return str(candidate)


def _settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def default_output_dir(project_path: str = "") -> str:
    """Next to the project file when it has been saved; else the last export folder; else the Videos folder."""
    if project_path and os.path.isdir(os.path.dirname(os.path.abspath(project_path))):
        return os.path.dirname(os.path.abspath(project_path))
    last = str(_settings().value(LAST_DIR_KEY, "") or "")
    if last and os.path.isdir(last):
        return last
    videos = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
    return videos if videos and os.path.isdir(videos) else str(Path.home())


def missing_media(project: Project, seq: Sequence) -> list[str]:
    """Names of the assets this sequence really reads whose files are gone.  Like the renderer, it skips disabled
    clips and hidden / muted tracks: a file nobody reads must not block the export."""
    used = {c.asset_id for tr in seq.tracks() if not tr.muted for c in tr.clips if c.enabled}
    names = []
    for aid in sorted(used):
        a = project.assets.get(aid)
        if a is not None and a.missing:
            names.append(a.name or os.path.basename(a.path))
    return names


def probe_writable(path: str) -> None:
    """Raise ``OSError`` when the output file ``path`` could not be written: the nearest existing folder must accept a new
    file, and an existing file must be openable for writing (a player holding it open fails here, not after minutes of
    rendering).  Nothing is created or modified."""
    folder = os.path.dirname(path)
    while folder and not os.path.exists(folder):               # the renderer creates missing folders: test the first real one
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    if not os.path.isdir(folder):
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), folder)
    fd, probe = tempfile.mkstemp(prefix=".autocut-write-test-", dir=folder)
    os.close(fd)
    try:
        os.unlink(probe)
    except OSError:
        pass
    if os.path.isfile(path):
        with open(path, "ab"):
            pass


def has_sound(project: Project, seq: Sequence) -> bool:
    """Would the exported file make any noise?  (Mirrors the renderer: disabled / muted clips, hidden or silent tracks and
    pictures contribute no audio.)  Loudness normalisation of pure silence makes ffmpeg's ``loudnorm`` fail, so the
    dialog does not offer it for such a sequence."""
    for tr in seq.tracks():
        if tr.muted or tr.volume <= 0:
            continue
        for c in tr.clips:
            a = project.assets.get(c.asset_id)
            if a is not None and c.enabled and not c.muted and c.volume > 0 and a.has_audio and a.kind != "image":
                return True
    return False


def open_folder(path: str) -> bool:
    """Open the folder holding ``path`` in the OS file manager."""
    return QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(path))))


def play_file(path: str) -> bool:
    """Open ``path`` with the default player."""
    return QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(path)))


_OS_REASONS = {
    errno.ENOSPC: "磁碟空間不足",
    errno.ENOENT: "找不到資料夾（路徑不存在）",
    errno.ENOTDIR: "路徑中有一段不是資料夾",
    errno.EROFS: "這個位置是唯讀的",
    errno.ENAMETOOLONG: "檔名或路徑太長",
    errno.EISDIR: "這個名稱已經是一個資料夾",
}


def explain_os_error(exc: OSError, path: str) -> str:
    """Plain 繁體中文 for the failures people actually run into; anything unusual keeps the system's own words."""
    name = os.path.basename(path)
    if isinstance(exc, PermissionError):
        return f"無法寫入「{name}」：檔案可能正被其他程式（例如播放器）開啟，或沒有寫入權限。"
    reason = _OS_REASONS.get(exc.errno)
    if reason:
        return f"無法寫入「{name}」：{reason}。"
    return f"無法寫入「{name}」：{exc.strerror or exc}（請確認磁碟空間與資料夾是否存在）。"


# ============================================================================ worker functions (testable)
class _RenderCancel(threading.Event):
    """The cancel event handed to the renderer.  The renderer sets the event it is given whenever a segment fails (to stop
    the other workers), and the task runner reads a set event as 「使用者取消」 -- so a real failure would be shown as
    「已取消輸出」 with no reason.  This one reads as set when the person cancelled (``user``) or the renderer set it, but
    setting it never touches the person's own flag."""

    def __init__(self, user: threading.Event) -> None:
        super().__init__()
        self._user = user

    def is_set(self) -> bool:
        return super().is_set() or self._user.is_set()

    isSet = is_set


def render_job(proj: Project, seq_id: str, out_path: str, settings: RenderSettings, progress, cancel) -> RenderResult:
    """Background render of a project copy; renderer / file-system errors become readable 繁體中文 messages."""
    try:
        return render_sequence(proj, proj.sequence(seq_id), out_path, settings, progress, _RenderCancel(cancel))
    except RenderError as exc:                       # includes MissingMediaError
        raise ValueError(str(exc)) from exc
    except OSError as exc:
        raise ValueError(explain_os_error(exc, out_path)) from exc


def qa_job(res: RenderResult, expect_size: tuple[int, int], expect_fps: float, progress, _cancel) -> QAReport:
    """Technical QA of the finished file: size and fps are expected from the sequence + preset, the length is the one
    the compiler planned for the chosen range (so a truncated or padded file is caught)."""
    progress(0.0, "技術檢查中…")
    return technical_qa(res.path, expect_duration=res.duration, expect_size=expect_size, expect_fps=expect_fps)


class _ReviewBox(HumanReviewBox):
    """HumanReviewBox that stays quiet when state signals arrive after the dialog's widgets were destroyed
    (the foundation widget connects a lambda to ``state.edited`` that cannot be disconnected from outside)."""

    on_refresh = None                      # the dialog keeps its QA table's last row in step with the person's verdict

    def refresh(self) -> None:
        if isValid(self):
            super().refresh()
            if self.on_refresh is not None:
                self.on_refresh()


# ============================================================================ the dialog
class ExportDialog(QDialog):
    """Export the active sequence.  ``exported(path)`` fires once the file exists (QA may still be running)."""

    exported = Signal(str)

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.seq_id = state.seq_id
        self.runner: Optional[TaskRunner] = None
        self.result: Optional[RenderResult] = None
        self.report: Optional[QAReport] = None
        self._busy = False
        self._detached = False
        self._closing = False
        self._started_at = 0.0
        self._range = state.get_range()
        self._preset_key = ""                  # preset the settings page was last adapted to
        self._exported_preset = ""             # preset of the render in flight (the QA expects its size)
        self._render_sha = ""                  # sequence_sha256 of the content being rendered (the check is bound to it)
        self.setWindowTitle("輸出影片")
        self.setModal(True)
        self.setMinimumWidth(640)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_settings_page())
        self.pages.addWidget(self._build_progress_page())
        self.pages.addWidget(self._build_result_page())
        lay = QVBoxLayout(self)
        lay.addWidget(self.pages)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)

        self._on_preset_changed()
        self._show_page(PAGE_SETTINGS)
        self._detect_encoder()

    @property
    def seq(self) -> Sequence:
        """The sequence being exported (looked up live: undo / redo replace the objects)."""
        return self.state.project.sequence(self.seq_id)

    # ------------------------------------------------------------------ pages
    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        seq = self.seq
        title = QLabel(f"輸出序列：{seq.name}")
        title.setStyleSheet("font-size: 15pt; font-weight: bold;")
        sub = QLabel(f"{seq.width}×{seq.height}（{seq.aspect_label}）· {seq.fps:g} fps · 長度 {fmt_timecode(seq.duration, seq.fps)}")
        sub.setObjectName("dim")

        self.preset = QComboBox()
        for key, p in PRESETS.items():
            if key != "preview_540p":                       # the in-app preview preset is not an export format
                self.preset.addItem(p.label, key)
        self.preset.setCurrentIndex(self.preset.findData("shorts_1080x1920" if seq.is_portrait else "youtube_1080p"))
        self.preset.currentIndexChanged.connect(self._on_preset_changed)
        self.preset_info = QLabel()
        self.preset_info.setObjectName("dim")
        self.preset_info.setWordWrap(True)

        self.path_edit = QLineEdit(unique_path(default_output_dir(self.state.project_path), safe_stem(seq.name)))
        self.path_edit.setPlaceholderText("輸出檔案（.mp4）")
        self.btn_browse = QPushButton("瀏覽…")
        self.btn_browse.clicked.connect(self.browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.btn_browse)

        self.range_all = QRadioButton(f"整個序列（{fmt_timecode(seq.duration, seq.fps)}）")
        self.range_all.setChecked(True)
        if self._range:
            t0, t1 = self._range
            self.range_marked = QRadioButton(
                f"入出點範圍（{fmt_timecode(t0, seq.fps)} – {fmt_timecode(t1, seq.fps)}，共 {fmt_timecode(t1 - t0, seq.fps)}）")
        else:
            self.range_marked = QRadioButton("入出點範圍（這個序列還沒設定入點／出點）")
            self.range_marked.setEnabled(False)

        self.encoder = QComboBox()
        self.encoder.addItem("CPU（libx264，相容性最好）", "libx264")
        self.encoder.addItem("自動（優先使用顯示卡硬體編碼，失敗會改用 CPU）", "auto")
        self.hw_info = QLabel("正在偵測硬體編碼器…")
        self.hw_info.setObjectName("dim")

        has_cues = any(q.text.strip() for q in seq.cues)
        has_subs = any(q.text.strip() for q in seq.cues if not q.is_card)
        has_cards = any(q.text.strip() for q in seq.cues if q.is_card)
        self.chk_burn = QCheckBox("把字幕與字卡燒進畫面" if has_cards else "把字幕燒進畫面")
        self.chk_srt = QCheckBox("另存字幕檔（.srt，放在影片旁邊）")
        self.chk_burn.setEnabled(has_cues)
        self.chk_srt.setEnabled(has_subs)
        self.chk_burn.setChecked(has_cues)
        if not has_cues:
            self.chk_burn.setToolTip("這個序列沒有字幕")
        if not has_subs:
            self.chk_srt.setToolTip("這個序列沒有字幕" + ("（字卡不會寫進 .srt）" if has_cards else ""))
        self.chk_loud = QCheckBox()
        self._has_sound = has_sound(self.state.project, seq)
        if not self._has_sound:
            self.chk_loud.setEnabled(False)
            self.chk_loud.setToolTip("這個序列沒有聲音，不需要調整音量")

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(8)
        rows = [("預設", self.preset), ("", self.preset_info), ("輸出檔案", path_row), ("範圍", self.range_all),
                ("", self.range_marked), ("編碼器", self.encoder), ("", self.hw_info), ("選項", self.chk_burn),
                ("", self.chk_srt), ("", self.chk_loud)]
        for r, (label, item) in enumerate(rows):
            if label:
                grid.addWidget(QLabel(label), r, 0, Qt.AlignTop | Qt.AlignRight)
            if isinstance(item, QHBoxLayout):
                grid.addLayout(item, r, 1)
            else:
                grid.addWidget(item, r, 1)

        self.btn_start = QPushButton("開始輸出")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_export)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_start)

        lay = QVBoxLayout(page)
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(8)
        lay.addLayout(grid)
        lay.addSpacing(8)
        lay.addLayout(buttons)
        return page

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        self.progress_title = QLabel("正在輸出…")
        self.progress_title.setStyleSheet("font-size: 15pt; font-weight: bold;")
        self.progress_target = QLabel()
        self.progress_target.setObjectName("dim")
        self.progress_target.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.stage = QLabel("準備中…")
        self.elapsed = QLabel("已用時間 0:00")
        self.elapsed.setObjectName("dim")
        self.btn_stop = QPushButton("取消輸出")
        self.btn_stop.setAutoDefault(False)         # focus lands here when the page appears: a second Enter must not cancel
        self.btn_stop.clicked.connect(self.cancel_export)
        row = QHBoxLayout()
        row.addWidget(self.stage, 1)
        row.addWidget(self.elapsed)
        lay = QVBoxLayout(page)
        lay.addWidget(self.progress_title)
        lay.addWidget(self.progress_target)
        lay.addSpacing(10)
        lay.addWidget(self.bar)
        lay.addLayout(row)
        lay.addSpacing(10)
        lay.addWidget(self.btn_stop, 0, Qt.AlignRight)
        return page

    def _build_result_page(self) -> QWidget:
        page = QWidget()
        self.result_title = QLabel()
        self.result_title.setStyleSheet("font-size: 15pt; font-weight: bold;")
        self.result_info = QLabel()
        self.result_info.setWordWrap(True)
        self.result_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_warnings = QLabel()
        self.result_warnings.setWordWrap(True)
        self.result_warnings.setStyleSheet(f"color: {theme.WARN};")

        self.qa_group = QGroupBox("技術檢查（機器檢查，不是審片）")
        self.qa_status = QLabel()
        self.qa_status.setWordWrap(True)
        self.btn_copy = QPushButton("複製報告")
        self.btn_copy.clicked.connect(self.copy_report)
        self.qa_table = QTableWidget(0, 3)
        self.qa_table.horizontalHeader().hide()
        self.qa_table.verticalHeader().hide()
        self.qa_table.verticalHeader().setDefaultSectionSize(24)
        self.qa_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.qa_table.setShowGrid(False)
        self.qa_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.qa_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.qa_table.setWordWrap(True)
        self.qa_table.setStyleSheet("QTableWidget::item { padding-right: 8px; }")     # air between the name and detail columns
        head = self.qa_table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.Fixed)
        head.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.Stretch)
        self.qa_table.setColumnWidth(0, 26)
        qa_top = QHBoxLayout()
        qa_top.addWidget(self.qa_status, 1)
        qa_top.addWidget(self.btn_copy, 0, Qt.AlignTop)
        qa_lay = QVBoxLayout(self.qa_group)
        qa_lay.addLayout(qa_top)
        qa_lay.addWidget(self.qa_table)

        self.review_group = QGroupBox("人工審片（由你決定）")
        self.review = _ReviewBox(self.state)
        self.review.on_refresh = self._update_review_row
        note = QLabel("技術檢查只確認檔案沒壞、長度與尺寸正確；剪得好不好，要你自己看過成品。")
        note.setObjectName("dim")
        note.setWordWrap(True)
        rv = QVBoxLayout(self.review_group)
        rv.addWidget(note)
        rv.addWidget(self.review)
        rv.addStretch(1)

        self.btn_folder = QPushButton("開啟資料夾")
        self.btn_folder.clicked.connect(lambda: self.result and open_folder(self.result.path))
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(lambda: self.result and play_file(self.result.path))
        self.btn_again = QPushButton("再輸出一次…")
        self.btn_again.clicked.connect(lambda: self._show_page(PAGE_SETTINGS))
        self.btn_close = QPushButton("關閉")
        self.btn_close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        for b in (self.btn_folder, self.btn_play):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_again)
        buttons.addWidget(self.btn_close)

        lay = QVBoxLayout(page)
        lay.addWidget(self.result_title)
        lay.addWidget(self.result_info)
        lay.addWidget(self.result_warnings)
        columns = QHBoxLayout()
        columns.addWidget(self.qa_group, 3)
        columns.addWidget(self.review_group, 2)
        lay.addLayout(columns)
        lay.addLayout(buttons)
        return page

    def _show_page(self, index: int) -> None:
        for i in range(self.pages.count()):
            w = self.pages.widget(i)
            policy = QSizePolicy.Preferred if i == index else QSizePolicy.Ignored
            w.setSizePolicy(policy, policy)
        self.pages.setCurrentIndex(index)
        # Enter in the file-name field starts the export.  (Set here, once the buttons are inside the dialog: set while the
        # pages were still being built it was ignored and Qt made 「瀏覽…」 the default, so Enter opened the file browser.)
        # The result page has no default button: Enter in the review note must neither close the dialog nor decide anything.
        self.btn_start.setDefault(index == PAGE_SETTINGS)
        self.setMinimumWidth(920 if index == PAGE_RESULT else 640)
        self.layout().activate()
        self._fit()

    def _fit(self) -> None:
        """Resize to the current page (``adjustSize`` would clamp to 2/3 of the screen, too small for the result).  The
        height is the one that page needs at the chosen *width*: the dialog's own size hint is computed at a narrow hint
        width, where the wrapped output path of the progress page runs to dozens of lines, and QStackedLayout reports the
        tallest page of all."""
        hint = self.sizeHint()
        avail = (self.screen() or QGuiApplication.primaryScreen()).availableGeometry()
        width = min(max(hint.width(), self.minimumWidth()), avail.width() - 40)
        margins = self.layout().contentsMargins()
        page = self.pages.currentWidget()
        inner = width - margins.left() - margins.right()
        content = page.layout().totalHeightForWidth(inner) if page.layout().hasHeightForWidth() else page.sizeHint().height()
        self.resize(width, min(content + margins.top() + margins.bottom(), avail.height() - 60))

    @property
    def page(self) -> int:
        return self.pages.currentIndex()

    @property
    def running(self) -> bool:
        """True from 「開始輸出」 until the result (or failure / cancel) is on screen."""
        return self._busy

    # ------------------------------------------------------------------ settings
    def _on_preset_changed(self, *_args) -> None:
        key = self.preset.currentData()
        preset = get_preset(key)
        w, h = output_size(self.seq.width, self.seq.height, preset)
        self.preset_info.setText(f"{preset.description}　→ 輸出 {w}×{h}")
        if preset.loudness is not None:
            self.chk_loud.setText(f"響度標準化（{preset.loudness:g} LUFS，各平台音量一致）")
        else:
            self.chk_loud.setText(f"響度標準化（{DEFAULT_LUFS:g} LUFS，這個預設原本不調整音量）")
        if key != self._preset_key:
            self.chk_loud.setChecked(preset.loudness is not None and self._has_sound)
            self._preset_key = key

    def _detect_encoder(self) -> None:
        runner = TaskRunner(lambda _p, _c: detect_hw_encoder(), name="偵測硬體編碼器")
        runner.finished.connect(self._on_encoder_detected)
        runner.failed.connect(self._on_encoder_failed)
        runner.start()

    def _on_encoder_detected(self, name: str) -> None:
        if name in HW_NAMES:
            self.hw_info.setText(f"偵測結果：{HW_NAMES[name]}（{name}）可用")
        else:
            self.hw_info.setText("偵測結果：這台電腦沒有可用的硬體編碼器，選「自動」也會使用 CPU")

    def _on_encoder_failed(self, _message: str) -> None:
        self.hw_info.setText("偵測結果：無法偵測硬體編碼器，選「自動」會使用 CPU")

    def browse(self) -> None:
        start = self.path_edit.text().strip() or default_output_dir(self.state.project_path)
        path, _ = QFileDialog.getSaveFileName(self, "輸出影片", start, "MP4 影片 (*.mp4)")
        if path:
            self.path_edit.setText(path if path.lower().endswith(".mp4") else path + ".mp4")

    def _typed_path(self) -> str:
        """The typed path made absolute ("" when empty).  A bare name goes to the usual output folder, not to whatever
        folder the app happened to be started from."""
        text = self.path_edit.text().strip().strip('"')
        if not text:
            return ""
        path = os.path.expanduser(text)
        if not os.path.isabs(path):
            path = os.path.join(default_output_dir(self.state.project_path), path)
        return os.path.abspath(path)

    def output_path(self) -> str:
        """The typed output path, absolute, with an ``.mp4`` extension."""
        path = self._typed_path()
        if not path:
            return ""
        return path if path.lower().endswith(".mp4") else path + ".mp4"

    def build_settings(self) -> RenderSettings:
        key = self.preset.currentData()
        preset = get_preset(key)
        settings = RenderSettings(preset=key, encoder=self.encoder.currentData(),
                                  burn_captions=self.chk_burn.isEnabled() and self.chk_burn.isChecked(),
                                  write_srt=self.chk_srt.isEnabled() and self.chk_srt.isChecked(),
                                  frame_range=self._range if self.range_marked.isChecked() else None)
        settings.normalize = self.chk_loud.isEnabled() and self.chk_loud.isChecked()
        if settings.normalize and preset.loudness is None:
            settings.loudness = DEFAULT_LUFS
        return settings

    # ------------------------------------------------------------------ start
    def preflight(self) -> Optional[str]:
        """Why the export cannot start (None when it can).  Shown politely, nothing is rendered."""
        proj, seq = self.state.project, self.seq
        if not any(True for _ in seq.all_clips()):
            return "這個序列還沒有任何片段，沒有東西可以輸出。\n請先把素材拖到時間軸，或用「自動剪輯」。"
        gone = missing_media(proj, seq)
        if gone:
            shown = "\n".join(f"．{n}" for n in gone[:8]) + ("\n…" if len(gone) > 8 else "")
            return f"有 {len(gone)} 個素材找不到檔案，沒辦法輸出：\n{shown}\n\n到素材庫對紅色標示的素材按右鍵 → 重新連結…。"
        out = self.output_path()
        if not out:
            return "請先選擇輸出檔案。"
        if os.path.isdir(out) or os.path.isdir(self._typed_path()):          # (".mp4" is appended to a typed folder path)
            return "輸出路徑是一個資料夾，請輸入檔名。"
        bad = sorted({ch for ch in os.path.basename(out) if ch in '<>:"|?*' or ord(ch) < 32})
        if bad:
            return ("檔名不能包含這些字元：\\ / : * ? \" < > |\n"
                    f"（目前的檔名裡有 {' '.join(bad)}），請改掉再輸出。")
        sources = {os.path.normcase(os.path.abspath(a.path)) for a in proj.assets.values()}
        if os.path.normcase(out) in sources:
            return "輸出檔案不能和素材檔相同，否則會蓋掉原始素材。請換一個檔名。"
        try:
            probe_writable(out)
        except OSError as exc:
            return explain_os_error(exc, out) + "\n請換一個資料夾或檔名。"
        return None

    def start_export(self) -> bool:
        """Validate and start rendering in the background.  Returns False when it was refused (nothing changed)."""
        if self._busy:
            return False
        problem = self.preflight()
        if problem:
            inform(self, "無法輸出", problem, warning=True)
            return False
        issues = [i.message for i in validate_sequence(self.state.project, self.seq) if i.level == "error"]
        if issues and not ask(self, "序列有問題", f"檢查到 {len(issues)} 個結構問題：\n"
                              + "\n".join(f"．{m}" for m in issues[:5]) + ("\n…" if len(issues) > 5 else "")
                              + "\n\n仍要輸出嗎？（輸出結果可能不正確）", "仍要輸出", "取消"):
            return False
        out = self.output_path()
        if os.path.exists(out) and not ask(self, "檔案已存在", f"「{os.path.basename(out)}」已經存在，要覆蓋嗎？",
                                           "覆蓋", "取消"):
            return False
        settings = self.build_settings()
        self._render_sha = planmod.sequence_sha256(self.seq)         # what this render (and its check) is about
        proj = Project.from_dict(self.state.project.to_dict())      # the render sees a frozen copy
        self._busy, self._detached = True, False
        self.result = self.report = None
        self._started_at = time.monotonic()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.stage.setText("準備中…")
        self.elapsed.setText("已用時間 0:00")
        self.btn_stop.setEnabled(True)
        self.btn_stop.setText("取消輸出")
        self.progress_title.setText("正在輸出…")
        self.progress_target.setText(f"{get_preset(settings.preset).label}\n{out}")
        self.runner = TaskRunner(lambda p, c: render_job(proj, self.seq_id, out, settings, p, c), name="輸出影片")
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_rendered)
        self.runner.failed.connect(self._on_render_failed)
        self.runner.cancelled.connect(self._on_render_cancelled)
        self._exported_preset = settings.preset
        self.timer.start()
        self._show_page(PAGE_PROGRESS)
        self.runner.start()
        return True

    # ------------------------------------------------------------------ progress / results
    def _tick(self) -> None:
        self.elapsed.setText(f"已用時間 {human_duration(time.monotonic() - self._started_at)}")

    def _on_progress(self, fraction: float, message: str) -> None:
        self.bar.setValue(int(fraction * 100))
        self.stage.setText(message or "輸出中…")

    def cancel_export(self) -> None:
        """「取消輸出」: kill the render; the dialog returns to the settings page when ffmpeg has stopped.  Once the file
        exists (``result`` is set) only the technical check is left; it cannot be cancelled, so this does nothing."""
        if self.runner is not None and self._busy and self.result is None:
            self.runner.cancel()
            self.btn_stop.setEnabled(False)
            self.stage.setText("正在取消…")

    def _on_render_cancelled(self) -> None:
        self._end_run()
        if self._closing:
            self.reject()
            return
        self._show_page(PAGE_SETTINGS)
        self.state.status.emit("已取消輸出")

    def _on_render_failed(self, message: str) -> None:
        self._end_run()
        if self._closing:
            self.reject()
            return
        self._show_failure(message)

    def _on_rendered(self, res: RenderResult) -> None:
        self.result = res
        self._remember_dir(res.path)
        self.exported.emit(res.path)
        self.state.status.emit(f"已輸出：{res.path}")
        if self._closing:
            self._end_run()
            self.reject()
            return
        self.progress_title.setText("正在做技術檢查…")
        self.bar.setRange(0, 0)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("技術檢查進行中…")
        self.stage.setText("技術檢查中…（檢查檔案能否讀取、黑畫面、靜音、響度）")
        seq = self.seq
        expect = output_size(seq.width, seq.height, get_preset(self._exported_preset))
        self.runner = TaskRunner(lambda p, c: qa_job(res, expect, seq.fps, p, c), name="技術檢查")
        self.runner.finished.connect(self._on_qa_done)
        self.runner.failed.connect(self._on_qa_failed)
        self.runner.start()

    def _on_qa_done(self, report: QAReport) -> None:
        self._end_run()
        if self._detached:
            return
        self.report = report
        run = self.state.run_for_sequence(self.seq_id)
        if run is not None:                          # machine evidence only; the review stays 「尚未審片」
            self.state.record_technical_qa(run["run_id"], report.status, report.summary(), seq_id=self.seq_id,
                                           sequence_sha=self._render_sha)
        self._show_result(report)

    def _on_qa_failed(self, message: str) -> None:
        self._end_run()
        if not self._detached:
            self._show_result(None, f"技術檢查沒辦法執行：{message}")

    def _end_run(self) -> None:
        self._busy = False
        self.timer.stop()

    def _remember_dir(self, path: str) -> None:
        _settings().setValue(LAST_DIR_KEY, os.path.dirname(os.path.abspath(path)))

    def _show_failure(self, message: str) -> None:
        self.result_title.setText("輸出失敗")
        self.result_title.setStyleSheet(f"font-size: 15pt; font-weight: bold; color: {theme.ERROR};")
        self.result_info.setText(message)
        self.result_warnings.setText("")
        self.qa_group.hide()
        self.review_group.hide()
        self.btn_folder.hide()
        self.btn_play.hide()
        self.btn_again.setText("回到設定")
        self._show_page(PAGE_RESULT)

    def _show_result(self, report: Optional[QAReport], qa_problem: str = "") -> None:
        res = self.result
        assert res is not None
        self.result_title.setText("輸出完成")
        self.result_title.setStyleSheet(f"font-size: 15pt; font-weight: bold; color: {theme.OK};")
        size = os.path.getsize(res.path) if os.path.isfile(res.path) else 0
        self.result_info.setText(
            f"{res.path}\n長度 {fmt_timecode(res.duration, res.fps)} · {res.width}×{res.height} · {res.fps:g} fps · "
            f"檔案 {human_size(size)} · 用時 {human_duration(res.elapsed)} · 編碼器 {res.encoder} · "
            f"快取命中 {res.cached}/{res.segments} 段")
        notes = [f"提醒：{w}" for w in res.warnings]
        if self._render_sha and self._render_sha != planmod.sequence_sha256(self.seq):
            notes.append("提醒：輸出之後這個序列又改過，這個檔案和現在的時間軸不一樣（上面的檢查說的是輸出當時的檔案），"
                         "要用現在的內容請重新輸出。")
        self.result_warnings.setText("\n".join(notes))
        self.btn_folder.show()
        self.btn_play.show()
        self.btn_again.setText("再輸出一次…")
        self.qa_group.show()
        self.review_group.show()
        self._fill_report(report, qa_problem)
        self.review.refresh()
        self._show_page(PAGE_RESULT)

    def _fill_report(self, report: Optional[QAReport], problem: str) -> None:
        if report is None:
            self.qa_status.setText(problem)
            self.qa_status.setStyleSheet(f"color: {theme.WARN};")
            rows = []
            self.btn_copy.setEnabled(False)
        else:
            text, color = QA_HEADLINE[report.status]
            if report.status == "not_run" and any(c.status != "not_run" for c in report.checks):
                text, color = QA_PARTIAL                 # some checks ran: 「沒有執行」 would be untrue
            self.qa_status.setText(text)
            self.qa_status.setStyleSheet(f"color: {color};")
            rows = [(c.status, c.name, c.detail) for c in report.checks]
            self.btn_copy.setEnabled(True)
        rows.append(self._review_row())
        self.qa_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._set_row(r, row)
        self.qa_table.setFixedHeight(min(len(rows) * 24 + 4, 320))

    def _set_row(self, r: int, row: tuple[str, str, str]) -> None:
        status, name, detail = row
        for c, text in enumerate((STATUS_ICON[status], name, detail)):
            it = QTableWidgetItem(text)
            it.setForeground(QColor(STATUS_COLOR[status] if c < 2 else theme.TEXT))
            it.setToolTip(detail)
            self.qa_table.setItem(r, c, it)

    def _review_row(self) -> tuple[str, str, str]:
        """The last table row: 「尚未審片」 until the PERSON clicks a verdict in the review box (machines never fill it in)."""
        run = self.state.run_for_sequence(self.seq_id)
        if run is None:
            return REVIEW_PENDING_ROW
        verdict, _notes = planmod.review_state(run, self.seq)
        return VERDICT_ROW.get(verdict, REVIEW_PENDING_ROW)

    def _update_review_row(self) -> None:
        n = self.qa_table.rowCount()
        if n:
            self._set_row(n - 1, self._review_row())

    def copy_report(self) -> None:
        if self.report is not None:
            QGuiApplication.clipboard().setText(self.report.summary())

    # ------------------------------------------------------------------ closing
    def reject(self) -> None:
        """Esc / 取消 / ✕.  A running render asks first; the QA pass (not cancellable) is simply abandoned."""
        if self._busy and self.page == PAGE_PROGRESS and not self._closing:
            if self.result is None:
                if not ask(self, "取消輸出", "正在輸出中，要取消並關閉嗎？", "取消輸出", "繼續輸出"):
                    return
                if self._busy and self.result is None:       # still rendering after the question: stop it, then close
                    self._closing = True
                    self.cancel_export()
                    return
                # (the render ended while the question was open: nothing is left to cancel, just close)
            if self._busy:                                    # the technical check is running: abandon it
                self._detached = True
        super().reject()
