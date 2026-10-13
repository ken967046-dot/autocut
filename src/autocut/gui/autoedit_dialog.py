"""Auto-edit dialogs (GUI_SPEC §6): ``ShortsDialog(state, asset_id)`` and ``AssembleDialog(state, asset_ids)``.

Flow (three pages in one dialog)::

    設定  -> 開始分析 -> [speech probe -> (ask before any model download)] -> propose_* + audit_plan (worker thread)
    審閱  -> one row per candidate, audit findings, 預覽此段 (throw-away project) -> 套用選取的片段
             = plan.subset(indexes) -> audit_plan again -> apply_plan (ONE undo step) -> state.set_sequence(first)

The real project is only touched by ``apply_plan`` through ``state.editor``; analysis and audit work on a snapshot.
``human_review`` is never called here: it stays "pending" until the person clicks it in the inspector.
Widgets of the settings live in ``autoedit_forms``; review-page widgets and the preview dialog in ``autoedit_review``.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Optional, Sequence

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QVBoxLayout, QWidget)

from ..analysis import speech as speech_mod
from ..autoedit import api
from ..autoedit.config import AssembleParams, AuditRules, ShortsParams
from ..core.plan import ERROR, WARN, AuditReceipt, EditPlan, Finding
from ..core.project import Asset, Project
from ..media.ffmpeg import Cancelled
from . import media_bin
from .autoedit_forms import (KEY_ASSEMBLE, KEY_SHORTS, PIP_COMMAND, AssembleForm, Note, ShortsForm, dim_label,
                             load_params, save_params)
from .autoedit_review import (Candidate, CandidateRow, FindingsView, PreviewDialog, ThumbLoader, UsageTable,
                              build_candidates, clock, compose_thumbnail, length_summary)
from .state import AppState
from .workers import TaskRunner, friendly_error, log_exception, log_path

PAGE_FORM, PAGE_PROGRESS, PAGE_REVIEW = range(3)
SPEECH_MODEL = "small"                      # the model the engine asks for (AnalyzeParams.model_size default)
_NO_SPEECH_MARK = "尚未安裝語音辨識"          # opening words of the engine's note and of ours (shown once)


# ============================================================================ engine entry points
# Thin module-level wrappers: tests (and other agents) can monkeypatch either ``autoedit.api.<name>`` or
# ``gui.autoedit_dialog.<name>``; the engine functions are looked up at call time.
def propose_shorts(*args, **kw):
    return api.propose_shorts(*args, **kw)


def propose_assemble(*args, **kw):
    return api.propose_assemble(*args, **kw)


def audit_plan(*args, **kw):
    return api.audit_plan(*args, **kw)


def apply_plan(*args, **kw):
    return api.apply_plan(*args, **kw)


def preview_project(*args, **kw):
    return api.preview_project(*args, **kw)


def ask(*args, **kw) -> bool:
    return media_bin.ask(*args, **kw)


def inform(*args, **kw) -> None:
    media_bin.inform(*args, **kw)


def _model_mb() -> int:
    return int(getattr(speech_mod, "APPROX_MB", {}).get(SPEECH_MODEL, 500))


def _snapshot(project: Project) -> Project:
    """A private copy for worker threads (``revision`` is kept, so a receipt for it is valid for the real project)."""
    return Project.from_dict(project.to_dict())


def _primary(btn: QPushButton) -> QPushButton:
    btn.setObjectName("primary")
    btn.setMinimumWidth(130)
    return btn


def _plain(btn: QPushButton) -> QPushButton:
    btn.setAutoDefault(False)
    btn.setMinimumWidth(84)
    return btn


# ============================================================================ the dialog
class _AutoEditDialog(QDialog):
    KIND = ""                                   # shorts | assemble
    TITLE = ""
    HEADER = ""
    FORM: type = ShortsForm
    PARAMS: type = ShortsParams
    KEY = KEY_SHORTS

    def __init__(self, state: AppState, asset_ids: Sequence[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.asset_ids: list[str] = list(asset_ids)
        self.plan: Optional[EditPlan] = None
        self.receipt: Optional[AuditReceipt] = None          # audit of the whole plan (what the review page shows)
        self.apply_receipt: Optional[AuditReceipt] = None    # audit of the subset the person tried to apply
        self.created_sequence_ids: list[str] = []
        self.candidates: list[Candidate] = []
        self.rows: list[CandidateRow] = []
        self.preview_dialogs: list[PreviewDialog] = []
        self.extra_notes: list[str] = []
        self.speech_status: Optional[tuple] = None           # (ok, reason, model cached)
        self.params_used: Any = None
        self._assets: dict[str, Asset] = {}
        self._runner: Optional[TaskRunner] = None
        self._probe_runner: Optional[TaskRunner] = None
        self._thumb_runner: Optional[TaskRunner] = None
        self._thumb_loader: Optional[ThumbLoader] = None
        self.autoplay_previews = True        # tests switch it off
        self._busy = False
        self._start_ok = True
        self._phase = ""
        self._token = 0
        self._closing = False
        self._started = 0.0
        self.setWindowTitle(self.TITLE)
        self.setModal(True)
        self.setMinimumSize(900, 640)
        self.resize(1000, 720)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._build_ui()
        self.form.set_params(load_params(self.PARAMS, self.KEY))
        self.form.restore_advanced()
        self._fill_source_info()
        self._show_page(PAGE_FORM)
        self._probe_speech()

    # ------------------------------------------------------------------ construction
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(8)
        self.title_label = QLabel(self.HEADER)
        self.title_label.setStyleSheet("QLabel { font-size: 15pt; font-weight: bold; }")
        self.subtitle = dim_label("")
        root.addWidget(self.title_label)
        root.addWidget(self.subtitle)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_form_page())
        self.pages.addWidget(self._build_progress_page())
        self.pages.addWidget(self._build_review_page())
        root.addWidget(self.pages, 1)

    def _build_form_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(8)
        self.source_note = Note()
        lay.addWidget(self.source_note)
        self.form = self.FORM()
        self.form.changed.connect(self._on_form_changed)
        self.form.advanced.toggled.connect(self._on_advanced_toggled)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self.form)
        self.form_scroll = scroll
        lay.addWidget(scroll, 1)
        self.form_note = Note()
        lay.addWidget(self.form_note)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_close = _plain(QPushButton("取消"))
        self.btn_close.clicked.connect(self.reject)
        self.btn_start = _primary(QPushButton("開始分析"))
        self.btn_start.setDefault(True)
        self.btn_start.clicked.connect(lambda *_: self.start_analysis())
        row.addWidget(self.btn_close)
        row.addWidget(self.btn_start)
        lay.addLayout(row)
        return page

    def _build_progress_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addStretch(2)
        col = QVBoxLayout()
        col.setSpacing(12)
        self.progress_title = QLabel("正在分析…")
        self.progress_title.setAlignment(Qt.AlignCenter)
        self.progress_title.setStyleSheet("QLabel { font-size: 13pt; font-weight: bold; }")
        self.progress_msg = dim_label("")
        self.progress_msg.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(560)
        self.progress_bar.setFixedHeight(18)
        self.progress_elapsed = dim_label("")
        self.progress_elapsed.setAlignment(Qt.AlignCenter)
        hint = dim_label("分析不會改動你的專案；可以隨時按「取消」，取消後設定還在。")
        hint.setAlignment(Qt.AlignCenter)
        self.btn_cancel_run = _plain(QPushButton("取消"))
        self.btn_cancel_run.clicked.connect(lambda *_: self.cancel_run())
        for w in (self.progress_title, self.progress_msg):
            col.addWidget(w)
        col.addWidget(self.progress_bar, 0, Qt.AlignHCenter)
        col.addWidget(self.progress_elapsed)
        col.addWidget(hint)
        col.addWidget(self.btn_cancel_run, 0, Qt.AlignHCenter)
        outer.addLayout(col)
        outer.addStretch(3)
        return page

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(8)
        # ---- top: (assemble) length note + usage table, then the candidate rows
        self.length_note = Note()
        lay.addWidget(self.length_note)
        self.usage = UsageTable()
        self.usage.setVisible(False)
        lay.addWidget(self.usage)
        self.rows_box = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_box)
        self.rows_layout.setContentsMargins(0, 0, 4, 0)
        self.rows_layout.setSpacing(8)
        self.rows_layout.addStretch(1)
        self.rows_scroll = QScrollArea()
        self.rows_scroll.setWidgetResizable(True)
        self.rows_scroll.setFrameShape(QFrame.NoFrame)
        self.rows_scroll.setWidget(self.rows_box)
        self.findings = FindingsView()
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.rows_scroll)
        self.splitter.addWidget(self.findings)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.rows_scroll.setMinimumHeight(150)
        self.findings.setMinimumHeight(90)
        lay.addWidget(self.splitter, 1)
        self.apply_note = Note()
        lay.addWidget(self.apply_note)
        row = QHBoxLayout()
        self.btn_back = _plain(QPushButton("← 重新設定"))
        self.btn_back.setToolTip("回到設定頁，改完參數可以重新分析。")
        self.btn_back.clicked.connect(lambda *_: self.back_to_form())
        self.selection_label = QLabel("")
        self.btn_close_review = _plain(QPushButton("取消"))
        self.btn_close_review.clicked.connect(self.reject)
        self.btn_apply = _primary(QPushButton("套用選取的片段"))
        self.btn_apply.clicked.connect(lambda *_: self.apply_selected())
        row.addWidget(self.btn_back)
        row.addSpacing(12)
        row.addWidget(self.selection_label, 1)
        row.addWidget(self.btn_close_review)
        row.addWidget(self.btn_apply)
        lay.addLayout(row)
        return page

    # ------------------------------------------------------------------ pages
    @property
    def page(self) -> int:
        return self.pages.currentIndex()

    @property
    def running(self) -> bool:
        """True while a background task (speech check, model download, analysis, audit) is in flight."""
        return self._busy

    def active_runners(self) -> list[TaskRunner]:
        return [r for r in (self._runner, self._probe_runner, self._thumb_runner) if r is not None]

    def _show_page(self, index: int) -> None:
        for i in range(self.pages.count()):
            w = self.pages.widget(i)
            policy = QSizePolicy.Expanding if i == index else QSizePolicy.Ignored
            w.setSizePolicy(policy, policy)
        self.pages.setCurrentIndex(index)
        self.subtitle.setText(self._subtitle_for(index))
        if index == PAGE_REVIEW:
            QTimer.singleShot(0, self._balance_splitter)
        if index == PAGE_FORM:
            self.btn_start.setDefault(True)
        elif index == PAGE_REVIEW:
            self.btn_apply.setDefault(True)

    def _balance_splitter(self) -> None:
        """Candidates get ~62% of the review page, the audit findings the rest (the handle still resizes)."""
        try:
            h = self.splitter.height()
            if h > 0 and not self._closing:
                want = self.rows_box.sizeHint().height() + 8          # few candidates: leave the rest to the findings
                top = max(150, min(int(h * 0.62), want))
                self.splitter.setSizes([top, max(90, h - top)])
        except RuntimeError:                                          # the dialog was deleted meanwhile
            pass

    def _subtitle_for(self, index: int) -> str:
        if index == PAGE_FORM:
            return "選好設定後按「開始分析」。分析不會改動你的專案，結果由你確認後才會套用。"
        if index == PAGE_PROGRESS:
            return "分析中…"
        n = len(self.candidates)
        if self.KIND == "shorts":
            return (f"找到 {n} 支。勾選要保留的，按「套用選取的片段」才會寫進專案（Ctrl+Z 可整批復原）。")
        return "看過總長與每支素材的使用情形後，按「套用選取的片段」才會建立新序列（Ctrl+Z 可復原）。"

    def _back_to_form(self, message: str = "", level: str = "info") -> None:
        self._end_task()
        self._show_page(PAGE_FORM)
        self.form_note.set_note(message, level)

    def back_to_form(self) -> None:
        """「← 重新設定」: keep the settings, drop nothing (the last result stays until the next analysis)."""
        if not self._busy:
            self._show_page(PAGE_FORM)

    # ------------------------------------------------------------------ task plumbing
    def _run(self, phase: str, fn: Callable, on_done: Callable, *, title: str = "", detail: str = "",
             indeterminate: bool = False, page: Optional[int] = PAGE_PROGRESS) -> TaskRunner:
        """Start ``fn(progress, cancel)`` on a worker; late signals of an older task are ignored (token)."""
        self._token += 1
        tok = self._token
        self._busy = True
        self._phase = phase
        self._started = time.monotonic()
        if page == PAGE_PROGRESS:
            self.form.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.progress_title.setText(title or "處理中…")
            self.progress_msg.setText(detail)
            self.progress_bar.setRange(0, 0 if indeterminate else 100)
            self.progress_bar.setValue(0)
            self.progress_elapsed.setText("")
            self.btn_cancel_run.setEnabled(True)
            self._show_page(PAGE_PROGRESS)
            self._timer.start()
        r = TaskRunner(fn, None, name=f"autoedit-{phase}")
        r.progress.connect(lambda f, m, t=tok: self._on_progress(t, f, m))
        r.finished.connect(lambda res, t=tok: self._on_finished(t, on_done, res))
        r.failed.connect(lambda msg, t=tok: self._on_failed(t, msg))
        r.cancelled.connect(lambda t=tok: self._on_cancelled(t))
        self._runner = r
        r.start()
        return r

    def _end_task(self) -> None:
        self._busy = False
        self._phase = ""
        self._timer.stop()
        self._runner = None
        self.form.setEnabled(True)
        self.btn_start.setEnabled(self._start_ok)
        self._set_review_busy(False)

    def _abandon(self) -> None:
        """Forget the running task (its late signals are ignored) and ask it to stop."""
        self._token += 1
        if self._runner is not None:
            self._runner.cancel()
        self._end_task()

    def _live(self, tok: int) -> bool:
        return tok == self._token and not self._closing

    def _on_progress(self, tok: int, fraction: float, message: str) -> None:
        if not self._live(tok):
            return
        if self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(int(round(fraction * 100)))
        if message:
            self.progress_msg.setText(message)

    def _on_finished(self, tok: int, on_done: Callable, result: Any) -> None:
        if not self._live(tok):
            return
        try:
            on_done(result)
        except Exception as exc:                                   # noqa: BLE001 - never leave a traceback on screen
            log_exception("autoedit dialog")
            self._fail(friendly_error(exc), unexpected=True)

    def _on_failed(self, tok: int, message: str) -> None:
        if self._live(tok):
            self._fail(message)

    def _fail(self, message: str, unexpected: bool = False) -> None:
        phase = self._phase
        if unexpected or message.startswith("發生未預期的錯誤"):
            message += f"\n詳細內容已記錄在：{log_path()}"
        if phase == "download":
            message = (f"無法下載語音模型：{message}\n可以檢查網路後再試一次，或把「字幕來源」改成「不加字幕」。")
        if phase == "audit":
            self._end_task()
            self.apply_note.set_note(message, "error")
            return
        self._back_to_form(message, "error")

    def _on_cancelled(self, tok: int) -> None:
        if not self._live(tok):
            return
        if self._phase == "audit":
            self._end_task()
            return
        self._back_to_form("已取消分析，專案沒有被改動。設定都還在，可以調整後再按「開始分析」。", "info")

    def cancel_run(self) -> None:
        """「取消」 on the progress page.  Analysis honours the cancel flag; a model download cannot be interrupted, so
        the dialog just stops waiting for it."""
        if not self._busy:
            return
        if self._phase in ("probe", "download"):
            note = "已取消。" + ("模型會在背景下載完成（之後再用就不必重下）。" if self._phase == "download" else "")
            self._abandon()
            self._show_page(PAGE_FORM)
            self.form_note.set_note(note, "info")
            return
        if self._runner is not None:
            self._runner.cancel()
        self.btn_cancel_run.setEnabled(False)
        self.progress_title.setText("正在取消…")

    def _tick(self) -> None:
        secs = int(time.monotonic() - self._started)
        self.progress_elapsed.setText(f"已進行 {secs // 60}:{secs % 60:02d}")

    # ------------------------------------------------------------------ speech (optional feature)
    @staticmethod
    def _probe_job(_progress, _cancel):
        ok, reason = speech_mod.speech_available()
        cached = bool(ok and speech_mod.model_cached(SPEECH_MODEL))
        return bool(ok), str(reason or ""), cached

    def _probe_speech(self) -> None:
        """Find out (off the GUI thread: importing faster-whisper is slow) whether captions can be generated."""
        r = TaskRunner(self._probe_job, None, name="autoedit-speech-probe")
        r.finished.connect(lambda res: self._apply_speech(res) if not self._closing else None)
        self._probe_runner = r
        r.start()

    def _apply_speech(self, res: tuple) -> None:
        ok, reason, cached = res
        self.speech_status = (ok, reason, cached)
        self.form.set_speech_status(ok, reason, cached, SPEECH_MODEL, _model_mb())

    def _after_probe(self, res: tuple) -> None:
        """Start of the analysis, step 2: ask before downloading a speech model (never silently)."""
        self._apply_speech(res)
        ok, _reason, cached = res
        params = self.params_used
        if ok and not cached:
            purpose = "自動字幕" if params.captions == "auto" else "用逐字稿挑選剪點"
            if ask(self, "下載語音模型",
                   f"{purpose}需要語音辨識模型「{SPEECH_MODEL}」（約 {_model_mb()} MB）。\n"
                   "第一次使用要連網下載一次，之後可離線使用；你的素材不會上傳，辨識都在這台電腦上進行。\n\n"
                   "要現在下載嗎？", "下載並繼續", "不下載（這次不用語音辨識）"):
                self._download_model(params)
                return
            params = self._without_speech(params)
        elif not ok and (params.captions == "auto"):
            self.extra_notes.append(f"{_NO_SPEECH_MARK}，所以這次沒有自動字幕。安裝：{PIP_COMMAND}，或改選「使用字幕檔」。")
        self._analyze(params)

    def _without_speech(self, p):
        """The person declined the download: captions off (and no transcript-guided cuts), with a note."""
        p = copy.deepcopy(p)
        if p.captions == "auto":
            p.captions = "off"
        if isinstance(p, ShortsParams) and p.captions != "file":
            p.use_speech = False
        self.extra_notes.append(
            "你沒有同意下載語音模型，所以這次沒有自動字幕" + ("，剪點也沒有用逐字稿" if isinstance(p, ShortsParams) else "")
            + "。想要字幕：再按一次「開始分析」並同意下載，或改選字幕檔。")
        return p

    def _download_model(self, params) -> None:
        mb = _model_mb()

        def job(progress, _cancel):
            progress(0.0, f"下載語音模型「{SPEECH_MODEL}」（約 {mb} MB）…")
            speech_mod._load_model(SPEECH_MODEL, True)
            return True

        def done(_ok) -> None:
            if self.speech_status:
                self.speech_status = (self.speech_status[0], self.speech_status[1], True)
            self.form.set_speech_status(True, "", True, SPEECH_MODEL, mb)
            self._analyze(params)

        self._run("download", job, done, title="正在下載語音模型…",
                  detail=f"第一次使用需要下載（約 {mb} MB），完成後可離線使用；素材不會上傳。", indeterminate=True)

    # ------------------------------------------------------------------ analysis
    def _precheck(self) -> str:                                    # overridden: 繁中 problem text or ""
        return ""

    def _fill_source_info(self) -> None:                           # overridden
        pass

    def _on_advanced_toggled(self, on: bool) -> None:
        if on:
            def reveal() -> None:
                y = self.form.advanced.mapTo(self.form, self.form.advanced.rect().topLeft()).y()
                self.form_scroll.verticalScrollBar().setValue(max(0, y - 4))
            QTimer.singleShot(0, reveal)

    def _on_form_changed(self) -> None:
        if self.form_note.text() and self.form_note.level == "error":
            self.form_note.set_note("")

    def start_analysis(self) -> bool:
        """「開始分析」: validate -> remember the settings -> (speech check) -> propose + audit in the background."""
        if self._busy:
            return False
        problem = self._precheck() or self.form.validate()
        if problem:
            self.form_note.set_note(problem, "error")
            return False
        self.form_note.set_note("")
        self.params_used = self.form.params()
        save_params(self.KEY, self.params_used)
        self.extra_notes = []
        if self.form.wants_speech():
            self._run("probe", self._probe_job, self._after_probe, title="檢查語音辨識…", indeterminate=True)
        else:
            self._analyze(self.params_used)
        return True

    def _propose(self, snapshot: Project, params, progress, cancel) -> EditPlan:      # overridden
        raise NotImplementedError

    def _analyze(self, params) -> None:
        snapshot = _snapshot(self.state.project)
        self._assets = dict(snapshot.assets)

        def job(progress, cancel):
            def sub(f: float, m: str = "") -> None:
                progress(0.95 * max(0.0, min(1.0, f)), m)

            plan = self._propose(snapshot, params, sub, cancel)
            if cancel.is_set():
                raise Cancelled()
            progress(0.96, "審核計畫…")
            receipt = audit_plan(plan, snapshot)
            progress(1.0, "完成")
            return plan, receipt

        self._run("analyze", job, self._on_analyzed, title=self.ANALYZING, detail="準備分析…")

    ANALYZING = "正在分析…"

    def _on_analyzed(self, result: tuple) -> None:
        plan, receipt = result
        self._end_task()
        if not plan.sequences:
            self._back_to_form("這次沒有找到可用的片段。可以放寬設定（例如縮短目標長度、減少支數）再試一次。", "warn")
            return
        self.plan, self.receipt, self.apply_receipt = plan, receipt, None
        self._build_review()
        self._show_page(PAGE_REVIEW)

    # ------------------------------------------------------------------ review page
    def _build_review(self) -> None:
        plan = self.plan
        assets = dict(self._assets)
        for pa in plan.assets:
            try:
                a = Asset.from_dict(pa)
                assets[a.id] = a
            except Exception:                                      # noqa: BLE001
                continue
        self._assets = assets
        if self._thumb_runner is not None:
            self._thumb_runner.cancel()
        for row in self.rows:
            row.setParent(None)
            row.deleteLater()
        self.rows = []
        self.candidates = build_candidates(plan, assets)
        selectable = self.KIND == "shorts"
        for n, cand in enumerate(self.candidates, 1):
            row = CandidateRow(cand, n, selectable=selectable)
            row.includeChanged.connect(lambda *_: self._refresh_review())
            row.previewRequested.connect(self.preview_candidate)
            self.rows_layout.insertWidget(self.rows_layout.count() - 1, row)
            self.rows.append(row)
        self.apply_note.set_note("")
        self._build_extras()
        self._refresh_review()
        loader = ThumbLoader(self)
        rows = list(self.rows)

        def ready(idx: int, path: str, rows=rows) -> None:
            if self._closing or idx >= len(rows):
                return
            row = rows[idx]
            try:
                row.set_pixmap(compose_thumbnail(path, crop=row.cand.crop) if path else QPixmap())
            except RuntimeError:                                   # the row was deleted (a newer analysis)
                pass

        loader.ready.connect(ready)
        jobs = [(c.index, c.thumb_asset, c.thumb_time) for c in self.candidates if c.thumb_asset is not None]
        for c in self.candidates:
            if c.thumb_asset is None:
                self.rows[c.index].set_pixmap(QPixmap())
        self._thumb_runner = loader.start(jobs) if jobs else None
        self._thumb_loader = loader

    def _build_extras(self) -> None:                               # overridden (assemble: usage table)
        self.length_note.set_note("")
        self.usage.setVisible(False)

    def selected_indexes(self) -> list[int]:
        return [r.index for r in self.rows if r.included()]

    def _seq_ids(self, indexes: Optional[list[int]] = None) -> set[str]:
        return {c.seq_id for c in self.candidates if indexes is None or c.index in indexes}

    def _describe(self, f: Finding) -> str:
        head = f.where.split("/")[0] if f.where else ""
        for n, c in enumerate(self.candidates, 1):
            if c.seq_id == head:
                return f"第 {n} 支" if self.KIND == "shorts" else "合併後的長片"
        asset = self._assets.get(head)
        return asset.name if asset is not None else ""

    def _refresh_review(self) -> None:
        """Selection changed (or a new receipt): badges, findings list, summary line, apply button."""
        if self.plan is None:
            return
        receipt = self.receipt
        chosen = set(self.selected_indexes())
        unchosen = {c.seq_id for c in self.candidates if c.index not in chosen}
        findings = list(receipt.findings) if receipt is not None else []
        for row in self.rows:
            mine = [f for f in findings if f.where.split("/")[0] == row.cand.seq_id]
            row.set_counts(sum(1 for f in mine if f.severity == ERROR), sum(1 for f in mine if f.severity == WARN))
        self.findings.set_content(findings, self._describe, lambda f: f.where.split("/")[0] in unchosen,
                                  self._all_notes())
        total = sum(c.duration for c in self.candidates if c.index in chosen)
        if self.KIND == "shorts":
            self.selection_label.setText(f"已選 {len(chosen)} / {len(self.candidates)} 支，合計約 {clock(total)}")
        else:
            self.selection_label.setText(f"成品長度 {clock(total)}")
        self.btn_apply.setEnabled(bool(chosen) and not self._busy)
        self.btn_apply.setToolTip("" if chosen else "請至少勾選一支。")
        if self.apply_note.text() and self.apply_note.level in ("error", "warn"):
            self.apply_note.set_note("")

    def _all_notes(self) -> list[str]:
        """Notes shown under the findings: the dialog's own (speech declined / missing) + the engine's."""
        engine = [str(n) for n in (self.plan.notes if self.plan else [])]
        mine = list(self.extra_notes)
        if any(_NO_SPEECH_MARK in n for n in engine):                  # the engine already says it: do not repeat
            mine = [n for n in mine if _NO_SPEECH_MARK not in n]
        return mine + engine

    def _set_review_busy(self, busy: bool) -> None:
        for row in self.rows:
            try:
                row.set_busy(busy)
            except RuntimeError:
                pass
        self.btn_back.setEnabled(not busy)
        self.btn_apply.setEnabled(not busy and bool(self.selected_indexes()))

    # ------------------------------------------------------------------ preview
    def preview_candidate(self, index: int) -> Optional[PreviewDialog]:
        """「預覽此段」: a small non-modal player on a THROW-AWAY project copy (the real project is not touched)."""
        if self.plan is None or not (0 <= index < len(self.candidates)):
            return None
        for dlg in self.preview_dialogs:
            if getattr(dlg, "plan_index", -1) == index and dlg.isVisible():
                dlg.raise_()
                dlg.activateWindow()
                return dlg
        try:
            scratch, seq_id = preview_project(self.state.project, self.plan, index)
            cand = self.candidates[index]
            title = (f"預覽：第 {index + 1} 支" if self.KIND == "shorts" else "預覽：合併後的長片") + \
                    f"（{clock(cand.duration)}）"
            dlg = PreviewDialog(scratch, seq_id, title, parent=self, autoplay=self.autoplay_previews)
        except Exception as exc:                                   # noqa: BLE001
            log_exception("autoedit preview")
            self.apply_note.set_note(f"無法建立預覽：{friendly_error(exc)}", "error")
            return None
        dlg.plan_index = index
        dlg.finished.connect(lambda *_: self._forget_preview(dlg))
        self.preview_dialogs.append(dlg)
        dlg.show()
        return dlg

    def _forget_preview(self, dlg: PreviewDialog) -> None:
        if dlg in self.preview_dialogs:
            self.preview_dialogs.remove(dlg)

    # ------------------------------------------------------------------ apply
    def apply_selected(self) -> bool:
        """「套用選取的片段」: subset -> audit again (background) -> apply (ONE undo step) or show the errors."""
        if self._busy or self.plan is None:
            return False
        indexes = self.selected_indexes()
        if not indexes:
            self.apply_note.set_note("請至少勾選一支再套用。", "warn")
            return False
        subset = self.plan.subset(indexes)
        snapshot = _snapshot(self.state.project)
        self.apply_note.set_note("")
        self._set_review_busy(True)
        self._run("audit", lambda _p, _c: audit_plan(subset, snapshot), lambda receipt: self._on_audited(subset, receipt),
                  page=None)
        return True

    def _on_audited(self, subset: EditPlan, receipt: AuditReceipt) -> None:
        self._end_task()
        if not receipt.accepted:
            self.apply_receipt = receipt
            errors = receipt.errors
            lines = "\n".join(f"・{f.message}" for f in errors[:4]) + ("\n…" if len(errors) > 4 else "")
            self.findings.set_content(list(receipt.findings), self._describe, lambda f: False, self._all_notes(),
                                      blocked=True)
            self.apply_note.set_note("審核未通過，沒有套用任何內容。請看上方列出的錯誤，或取消勾選有問題的片段再試一次。\n"
                                     + lines, "error")
            return
        try:
            new_ids = apply_plan(self.state.editor, subset, receipt)
        except Exception as exc:                                   # noqa: BLE001 - nothing was applied (atomic)
            log_exception("autoedit apply")
            msg = friendly_error(exc)
            if msg.startswith("發生未預期的錯誤"):
                msg += f"\n詳細內容已記錄在：{log_path()}"
            self.apply_note.set_note("沒有套用任何內容：" + msg, "error")
            return
        self.created_sequence_ids = list(new_ids)
        if new_ids:
            self.state.set_sequence(new_ids[0])
        self.state.status.emit(f"已建立 {len(new_ids)} 個序列（Ctrl+Z 可整批復原）")
        self.accept()

    # ------------------------------------------------------------------ closing
    def done(self, result: int) -> None:                          # noqa: N802 (Qt API)
        """Esc / ✕ / 取消 / accept: a running task is cancelled, open previews are closed."""
        if not self._closing:
            self._closing = True
            self._token += 1
            self._timer.stop()
            for r in self.active_runners():
                if r.running:
                    r.cancel()
            for dlg in list(self.preview_dialogs):
                dlg.close()
        super().done(result)


# ============================================================================ short videos
class ShortsDialog(_AutoEditDialog):
    KIND = "shorts"
    TITLE = "自動剪輯：長影片 → 短影音"
    HEADER = "把長影片剪成多支 9:16 短影音"
    FORM = ShortsForm
    PARAMS = ShortsParams
    KEY = KEY_SHORTS
    ANALYZING = "正在分析影片，挑選適合的段落…"

    def __init__(self, state: AppState, asset_id: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(state, [asset_id], parent)

    @property
    def asset_id(self) -> str:
        return self.asset_ids[0] if self.asset_ids else ""

    def _asset(self) -> Optional[Asset]:
        return self.state.project.assets.get(self.asset_id)

    def _precheck(self) -> str:
        a = self._asset()
        if a is None:
            return "專案裡找不到這個素材（它可能已被移除）。"
        if a.kind != "video" or not a.has_video:
            return "這個素材沒有影像，無法剪成直式短影音。"
        if a.missing:
            return f"找不到素材檔案：{a.path}\n請先在素材庫重新連結檔案。"
        return ""

    def _fill_source_info(self) -> None:
        a = self._asset()
        problem = self._precheck()
        if a is None:
            self.source_note.set_note(problem, "error")
        else:
            bits = [f"來源：{a.name}", clock(a.duration), f"{a.width}×{a.height}", f"{a.fps:g} fps",
                    "有聲音" if a.has_audio else "沒有聲音"]
            self.source_note.set_note("　·　".join(bits) + (f"\n{problem}" if problem else ""),
                                      "error" if problem else "info")
        self._start_ok = not problem
        self.btn_start.setEnabled(self._start_ok)
        self.btn_start.setToolTip(problem)

    def _propose(self, snapshot: Project, params, progress, cancel) -> EditPlan:
        return propose_shorts(snapshot, self.asset_id, params, progress, cancel)


# ============================================================================ many clips -> one long video
class AssembleDialog(_AutoEditDialog):
    KIND = "assemble"
    TITLE = "自動剪輯：多支素材 → 一支長片"
    HEADER = "把多支素材合併成一支長片"
    FORM = AssembleForm
    PARAMS = AssembleParams
    KEY = KEY_ASSEMBLE
    ANALYZING = "正在分析素材並組裝長片…"

    def __init__(self, state: AppState, asset_ids: Sequence[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(state, asset_ids, parent)

    def _assets_in_project(self) -> list[Asset]:
        proj = self.state.project
        return [proj.assets[i] for i in self.asset_ids if i in proj.assets]

    def _usable(self) -> list[Asset]:
        return [a for a in self._assets_in_project() if a.kind in ("video", "image") and not a.missing]

    def _precheck(self) -> str:
        if len(self._usable()) < 2:
            return "至少需要 2 個找得到檔案的影片或圖片素材才能合併。"
        return ""

    def _fill_source_info(self) -> None:
        found = self._assets_in_project()
        usable = self._usable()
        total = sum(a.duration for a in usable if a.kind == "video")
        n_vid = sum(1 for a in usable if a.kind == "video")
        n_img = sum(1 for a in usable if a.kind == "image")
        text = f"素材：{len(usable)} 個（影片 {n_vid}、圖片 {n_img}），影片總長 {clock(total)}"
        skipped = [a.name for a in found if a not in usable]
        gone = len(self.asset_ids) - len(found)
        extra = []
        if skipped:
            extra.append(f"{len(skipped)} 個素材（{'、'.join(skipped[:3])}{'…' if len(skipped) > 3 else ''}）"
                         "找不到檔案或不是影片／圖片，會被略過")
        if gone:
            extra.append(f"{gone} 個素材已不在專案裡")
        problem = self._precheck()
        level = "error" if problem else ("warn" if extra else "info")
        names = [f"{a.name}（{clock(a.duration) if a.kind == 'video' else '圖片'}）" for a in usable]
        listing = "包含：" + "、".join(names[:6]) + (f"…等 {len(names)} 個" if len(names) > 6 else "") if names else ""
        self.source_note.set_note("\n".join([text] + ([listing] if listing else []) + extra
                                            + ([problem] if problem else [])), level)
        self.source_note.setToolTip("\n".join(f"{i}. {n}" for i, n in enumerate(names, 1)))
        self._start_ok = not problem
        self.btn_start.setEnabled(self._start_ok)
        self.btn_start.setToolTip(problem)

    def _propose(self, snapshot: Project, params, progress, cancel) -> EditPlan:
        return propose_assemble(snapshot, self.asset_ids, params, progress, cancel)

    def _build_extras(self) -> None:
        text, level = length_summary(self.plan, AuditRules().target_tolerance)
        self.length_note.set_note(text, "ok" if level == "ok" else "warn")
        self.usage.set_plan(self.plan)
        self.usage.setVisible(True)
        h = self.usage.horizontalHeader().height() + 2 + sum(self.usage.rowHeight(r) for r in range(self.usage.rowCount()))
        self.usage.setFixedHeight(min(max(h, 60), 170))
