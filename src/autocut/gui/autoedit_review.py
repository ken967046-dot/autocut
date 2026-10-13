"""Review-page widgets of the auto-edit dialogs (GUI_SPEC §6).

* ``build_candidates(plan, assets)``  turns an ``EditPlan`` into one ``Candidate`` per sequence (time range, length,
  score, 「為什麼選這段」 from ``plan.decisions``, thumbnail request);
* ``CandidateRow``   one row per candidate: include checkbox, thumbnail (with the 9:16 crop window drawn on it),
  texts, audit badge and 「預覽此段」;
* ``FindingsView``   audit findings (errors red, warnings amber, info dim) + the plan's notes;
* ``UsageTable``     assemble only: per source seconds used / usable, the used stretches and dropped parts;
* ``PreviewDialog``  small non-modal player on a THROW-AWAY ``AppState`` (the real project is never touched).

No project mutation happens here; the dialog module owns the flow.
"""
from __future__ import annotations

import copy
import html
from dataclasses import dataclass, field
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..core import geometry as geo
from ..core.plan import ERROR, INFO, WARN, EditPlan, Finding
from ..core.project import Asset, Project, Sequence
from ..core.timecode import fmt_clock
from ..media import cache as media_cache
from ..media.ffmpeg import Cancelled
from . import theme
from .monitor import MonitorWidget
from .playback import PlaybackController
from .state import AppState
from .workers import TaskRunner

THUMB_W, THUMB_H = 160, 90
SEV_LABEL = {ERROR: "錯誤", WARN: "警告", INFO: "提示"}
SEV_COLOR = {ERROR: theme.ERROR, WARN: theme.WARN, INFO: theme.TEXT_DIM}
SEV_ORDER = {ERROR: 0, WARN: 1, INFO: 2}


def clock(t: float) -> str:
    """``m:ss`` (``h:mm:ss`` from one hour on)."""
    return fmt_clock(max(0.0, float(t or 0.0)), millis=False)


def sorted_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: SEV_ORDER.get(f.severity, 3))


# ================================================================================================ candidates
@dataclass
class Candidate:
    index: int                                   # position in ``plan.sequences``
    seq_id: str
    kind: str                                    # shorts | assemble | longform
    name: str
    duration: float
    score: Optional[float] = None
    src_range: Optional[tuple[float, float]] = None
    why: str = ""
    why_label: str = "為什麼選這段"
    details: list[str] = field(default_factory=list)
    clips: int = 0
    sources: int = 0
    width: int = 0
    height: int = 0
    thumb_asset: Optional[Asset] = None
    thumb_time: float = 0.0
    crop: Optional[tuple[float, float, float, float]] = None     # normalised crop window on the thumbnail

    def headline(self, number: int) -> str:
        """``<b>第 1 支</b>  <dim>原片 0:15–0:29 · 成品 14 秒 · 分數 0.76</dim>`` (rich text)."""
        dim = theme.TEXT_DIM
        if self.kind == "shorts":
            bits = []
            if self.src_range:
                bits.append(f"原片 {clock(self.src_range[0])}–{clock(self.src_range[1])}")
            bits.append(f"成品 {self.duration:.0f} 秒")
            if self.score is not None:
                bits.append(f"分數 {self.score:.2f}")
            return f"<b>第 {number} 支</b>　<span style='color:{dim}'>{html.escape(' · '.join(bits))}</span>"
        if self.kind == "longform":
            bits = [f"成品 {clock(self.duration)}", f"{self.clips} 個片段"]
            if self.src_range:
                bits.insert(0, f"原片 {clock(self.src_range[0])}–{clock(self.src_range[1])}")
            return f"<b>精華長片</b>　<span style='color:{dim}'>{html.escape(' · '.join(bits))}</span>"
        bits = [f"{clock(self.duration)}", f"{self.clips} 個鏡頭", f"用到 {self.sources} 支素材"]
        return f"<b>合併後的長片</b>　<span style='color:{dim}'>{html.escape(' · '.join(bits))}</span>"


DETAIL_KINDS = {"shorts": ("reframe", "tighten", "caption", "music"), "assemble": ("format", "audio", "music", "caption"),
                "longform": ("filler", "zoom", "caption")}


def _crop_of(asset: Optional[Asset], seq: Sequence, clip, t_src: float) -> Optional[tuple[float, float, float, float]]:
    """Normalised ``(x, y, w, h)`` of the window a ``cover`` clip crops out of the source at source time ``t_src``."""
    if asset is None or clip is None or clip.transform.fit != "cover":
        return None
    dw, dh = float(asset.width or 0), float(asset.height or 0)
    if dw <= 0 or dh <= 0 or seq.width <= 0 or seq.height <= 0:
        return None
    tf = clip.transform
    cw, ch = geo.crop_window(dw, dh, seq.width, seq.height, tf.zoom)
    if cw >= dw - 0.5 and ch >= dh - 0.5:
        return None
    t_local = max(0.0, (t_src - clip.src_in) / (clip.speed or 1.0))
    cx, cy = geo.interp_keys(tf.keys, t_local, tf.cx, tf.cy)
    x, y = geo.crop_topleft(dw, dh, cw, ch, cx, cy)
    return x / dw, y / dh, cw / dw, ch / dh


def build_candidates(plan: EditPlan, assets: dict[str, Asset]) -> list[Candidate]:
    by_seq: dict[str, list[dict]] = {}
    for d in plan.decisions:
        by_seq.setdefault(str(d.get("sequence") or ""), []).append(d)
    out: list[Candidate] = []
    for i, sd in enumerate(plan.sequences):
        seq = Sequence.from_dict(copy.deepcopy(sd))
        decs = by_seq.get(seq.id, [])
        main = next((d for d in decs if d.get("kind") == "short"), None)
        why_label = "為什麼選這段"
        if main is None and plan.kind == "assemble":
            main, why_label = next((d for d in decs if d.get("kind") == "order"), None), "排序依據"
        if main is None and plan.kind == "longform":
            main, why_label = next((d for d in decs if d.get("kind") == "longform"), None), "怎麼剪的"
        src = (main or {}).get("source") or {}
        rng = None
        try:
            if src.get("t1") is not None:
                rng = (float(src.get("t0", 0.0)), float(src["t1"]))
        except (TypeError, ValueError):
            rng = None
        score = (main or {}).get("score")
        why = str((main or {}).get("why") or (main or {}).get("summary") or "")
        if plan.kind in ("assemble", "longform") and main is not None and main.get("summary"):
            why = f"{main['summary']}。{why}".strip()
        details, seen = [], set()
        label = {"reframe": "取景", "tighten": "緊湊化", "caption": "字幕", "music": "配樂", "format": "格式",
                 "audio": "音量", "filler": "水詞", "zoom": "講者特寫"}
        for kind in DETAIL_KINDS.get(plan.kind, ()):
            for d in decs:
                s = str(d.get("summary") or "").strip()
                if d.get("kind") == kind and s and s not in seen:
                    seen.add(s)
                    details.append(f"{label.get(kind, kind)}：{s}")
                    break
        v1 = list(seq.video_tracks[0].clips) if seq.video_tracks else []
        first = min(v1, key=lambda c: c.start) if v1 else None
        asset = assets.get(first.asset_id) if first is not None else None
        t_src = 0.0
        if first is not None:
            t_src = first.src_in + min(0.6, 0.25 * max(first.duration, 0.0))
        if asset is not None and asset.kind == "image":
            t_src = 0.0
        out.append(Candidate(
            index=i, seq_id=seq.id, kind=plan.kind, name=seq.name, duration=float(seq.duration),
            score=float(score) if isinstance(score, (int, float)) else None, src_range=rng, why=why,
            why_label=why_label, details=details, clips=len(v1), sources=len({c.asset_id for c in v1}),
            width=seq.width, height=seq.height, thumb_asset=asset, thumb_time=t_src,
            crop=_crop_of(asset, seq, first, t_src)))
    return out


# ================================================================================================ thumbnails
def compose_thumbnail(path: str, size: tuple[int, int] = (THUMB_W, THUMB_H),
                      crop: Optional[tuple[float, float, float, float]] = None) -> QPixmap:
    """The frame fitted into ``size``; with ``crop`` everything outside the 9:16 window is darkened."""
    img = QImage(path)
    if img.isNull():
        return QPixmap()
    w, h = size
    scaled = img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    out = QPixmap(w, h)
    out.fill(QColor("#0e0f11"))
    p = QPainter(out)
    x0, y0 = (w - scaled.width()) // 2, (h - scaled.height()) // 2
    p.drawImage(x0, y0, scaled)
    if crop:
        fx, fy, fw, fh = crop
        frame = QRectF(x0, y0, scaled.width(), scaled.height())
        win = QRectF(x0 + fx * scaled.width(), y0 + fy * scaled.height(), fw * scaled.width(), fh * scaled.height())
        outside = QPainterPath()
        outside.addRect(frame)
        inside = QPainterPath()
        inside.addRect(win)
        p.fillPath(outside.subtracted(inside), QColor(0, 0, 0, 150))
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(win.adjusted(1, 1, -1, -1))
    p.end()
    return out


class ThumbLoader(QObject):
    """Makes the candidates' thumbnails on a worker thread; ``ready(index, jpg path or "")`` arrives in the GUI thread."""

    ready = Signal(int, str)

    def start(self, jobs: list[tuple[int, Asset, float]]) -> TaskRunner:
        def work(_progress, cancel):
            for idx, asset, t in jobs:
                if cancel.is_set():
                    raise Cancelled()
                try:
                    path = str(media_cache.thumbnail(asset, t, THUMB_H))
                except Exception:                        # noqa: BLE001 - a missing thumbnail must not break the review
                    path = ""
                try:
                    self.ready.emit(idx, path)
                except RuntimeError:                     # the dialog (and this loader) is already gone
                    return None
            return None

        return TaskRunner(work, None, name="autoedit-thumbs").start()


# ================================================================================================ candidate row
class CandidateRow(QFrame):
    """One candidate: ``[☑] [thumbnail] headline / 為什麼選這段 / details   [badge] [預覽此段]``."""

    includeChanged = Signal(int, bool)
    previewRequested = Signal(int)

    def __init__(self, cand: Candidate, number: int, selectable: bool = True,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.cand = cand
        self.index = cand.index
        self.selectable = bool(selectable)
        self.setObjectName("candidate")
        self.setProperty("included", True)
        self.setStyleSheet(
            f"QFrame#candidate {{ background: {theme.PANEL}; border: 1px solid {theme.BORDER}; border-radius: 6px; }}"
            f"QFrame#candidate[included=\"false\"] {{ background: {theme.BG}; }}"
            "QFrame#candidate QLabel, QFrame#candidate QCheckBox { background: transparent; }")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(12)

        self.check = QCheckBox()
        self.check.setChecked(True)
        self.check.setToolTip("勾選＝套用時納入這一段")
        self.check.setVisible(selectable)
        self.check.toggled.connect(self._on_toggled)
        lay.addWidget(self.check, 0, Qt.AlignTop)

        self.thumb = QLabel("載入縮圖…")
        self.thumb.setFixedSize(THUMB_W, THUMB_H)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet(f"QLabel {{ background: #0e0f11; color: {theme.TEXT_DIM}; border-radius: 3px; }}")
        lay.addWidget(self.thumb, 0, Qt.AlignTop)

        col = QVBoxLayout()
        col.setSpacing(4)
        self.title = QLabel(cand.headline(number))
        self.title.setTextFormat(Qt.RichText)
        col.addWidget(self.title)
        self.why = QLabel()
        self.why.setWordWrap(True)
        self.why.setTextFormat(Qt.RichText)
        if cand.why:
            self.why.setText(f"<b>{html.escape(cand.why_label)}：</b>{html.escape(cand.why)}")
            self.why.setToolTip(cand.why)
        else:
            self.why.setVisible(False)
        col.addWidget(self.why)
        self.details = QLabel("　｜　".join(cand.details))
        self.details.setObjectName("dim")
        self.details.setWordWrap(True)
        self.details.setVisible(bool(cand.details))
        self.details.setToolTip("\n".join(cand.details))
        col.addWidget(self.details)
        col.addStretch(1)
        lay.addLayout(col, 1)

        right = QVBoxLayout()
        right.setSpacing(8)
        self.badge = QLabel("")
        self.badge.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.badge.setTextFormat(Qt.RichText)
        right.addWidget(self.badge, 0, Qt.AlignRight)
        right.addStretch(1)
        self.btn_preview = QPushButton("預覽此段")
        self.btn_preview.setAutoDefault(False)
        self.btn_preview.setToolTip("開一個小視窗播放這一段；不會改動你的專案。")
        self.btn_preview.clicked.connect(lambda *_: self.previewRequested.emit(self.index))
        right.addWidget(self.btn_preview, 0, Qt.AlignRight)
        lay.addLayout(right)

    # ---- state
    def included(self) -> bool:
        return self.check.isChecked() if self.selectable else True

    def set_included(self, on: bool) -> None:
        self.check.setChecked(bool(on))

    def _on_toggled(self, on: bool) -> None:
        self.setProperty("included", bool(on))
        self.style().unpolish(self)
        self.style().polish(self)
        self.includeChanged.emit(self.index, bool(on))

    def set_pixmap(self, pm: QPixmap) -> None:
        if pm.isNull():
            self.thumb.setText("（無縮圖）")
        else:
            self.thumb.setText("")
            self.thumb.setPixmap(pm)

    def set_counts(self, errors: int, warnings: int) -> None:
        bits = []
        if errors:
            bits.append(f"<span style='color:{theme.ERROR}'><b>{errors} 個錯誤</b></span>")
        if warnings:
            bits.append(f"<span style='color:{theme.WARN}'><b>{warnings} 個警告</b></span>")
        self.badge.setText("　".join(bits))

    def set_busy(self, busy: bool) -> None:
        self.check.setEnabled(not busy)
        self.btn_preview.setEnabled(not busy)


# ================================================================================================ findings
class FindingsView(QScrollArea):
    """Headline + audit findings (errors red, warnings amber, info dim) + the plan's notes."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"QScrollArea {{ background: {theme.PANEL}; border: 1px solid {theme.BORDER}; "
                           f"border-radius: 4px; }} QWidget#findingsInner {{ background: {theme.PANEL}; }} "
                           "QLabel { background: transparent; }")
        inner = QWidget()
        inner.setObjectName("findingsInner")
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(12, 8, 12, 8)
        self._lay.setSpacing(5)
        self._lay.addStretch(1)
        self.setWidget(inner)
        self._rows: list[tuple[str, str]] = []
        self.headline_text = ""

    def _add(self, text: str, rich: bool = True, color: str = "") -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setTextFormat(Qt.RichText if rich else Qt.PlainText)
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if color:
            lab.setStyleSheet(f"QLabel {{ color: {color}; background: transparent; }}")
        self._lay.insertWidget(self._lay.count() - 1, lab)
        return lab

    def _clear(self) -> None:
        while self._lay.count() > 1:
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._rows = []

    def set_content(self, findings: list[Finding], describe: Callable[[Finding], str],
                    muted: Callable[[Finding], bool], notes: list[str], blocked: bool = False) -> None:
        self._clear()
        active = [f for f in findings if not muted(f)]
        n_err = sum(1 for f in active if f.severity == ERROR)
        n_warn = sum(1 for f in active if f.severity == WARN)
        n_info = sum(1 for f in active if f.severity == INFO)
        if n_err:
            head, color = f"審核結果：{n_err} 個錯誤、{n_warn} 個警告、{n_info} 則提示。有錯誤時不能套用。", theme.ERROR
        elif n_warn:
            head, color = f"審核結果：沒有錯誤，{n_warn} 個警告、{n_info} 則提示。可以套用，但請看一下警告。", theme.WARN
        else:
            head, color = f"審核結果：沒有錯誤也沒有警告（{n_info} 則提示）。", theme.OK
        if blocked:
            head = "審核未通過，沒有套用任何內容。" + head.replace("審核結果：", "", 1)
        self.headline_text = head
        self._add(f"<b>{html.escape(head)}</b>", True, color)
        for f in sorted_findings(findings):
            is_muted = muted(f)
            col = theme.TEXT_DIM if is_muted else SEV_COLOR.get(f.severity, theme.TEXT_DIM)
            where = describe(f)
            text_color = theme.TEXT_DIM if (is_muted or f.severity == INFO) else theme.TEXT
            tail = f"　<span style='color:{theme.TEXT_DIM}'>（{html.escape(where)}）</span>" if where else ""
            if is_muted:
                tail += f"<span style='color:{theme.TEXT_DIM}'>（這一支沒勾選，不影響套用）</span>"
            self._add(f"<span style='color:{col}'><b>{SEV_LABEL.get(f.severity, f.severity)}</b></span>"
                      f"　<span style='color:{text_color}'>{html.escape(f.message)}</span>{tail}")
            self._rows.append((f.severity, f.message))
        if notes:
            self._add("<b>分析備註</b>", True, theme.TEXT_DIM)
            for n in notes:
                self._add(f"・{html.escape(str(n))}", True, theme.TEXT_DIM)

    def rows(self) -> list[tuple[str, str]]:
        """``[(severity, message)]`` of the findings currently listed (errors first)."""
        return list(self._rows)


# ================================================================================================ assemble usage
def length_summary(plan: EditPlan, tolerance: float = 0.35) -> tuple[str, str]:
    """``(text, level)``: total length against the target for the assemble review page (level ok | warn)."""
    total = float(plan.total_duration)
    target = float(plan.brief.get("target_seconds", 0) or 0)
    usable = float(plan.brief.get("usable_seconds", 0) or 0)
    parts = [f"總長 {clock(total)}"]
    level = "ok"
    if target > 0:
        diff = total - target
        parts.append(f"目標 {clock(target)}（{'多' if diff >= 0 else '少'} {abs(diff):.0f} 秒）")
        if abs(diff) / target > tolerance:
            level = "warn"
    else:
        parts.append("目標：保留所有可用內容")
    if usable > 0:
        parts.append(f"可用內容共 {clock(usable)}")
    return "　｜　".join(parts), level


class UsageTable(QTableWidget):
    """Assemble review: one row per source — seconds used / usable, the stretches used, dropped parts."""

    HEADERS = ["素材", "使用 / 可用", "用到的片段", "丟棄"]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeaderItem(1).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)
        self.setShowGrid(False)
        hh = self.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)

    def set_plan(self, plan: EditPlan) -> None:
        shots: dict[str, list[tuple[float, float]]] = {}
        drops: dict[str, int] = {}
        for d in plan.decisions:
            src = d.get("source") or {}
            aid = str(src.get("asset_id") or "")
            if not aid:
                continue
            if d.get("kind") == "shot":
                shots.setdefault(aid, []).append((float(src.get("t0", 0.0)), float(src.get("t1", 0.0))))
            elif d.get("kind") == "drop":
                drops[aid] = drops.get(aid, 0) + 1
        sources = [s for s in plan.sources if s.get("asset_id")]
        self.setRowCount(len(sources))
        for r, s in enumerate(sources):
            aid = str(s["asset_id"])
            used = float(s.get("used_seconds", 0.0) or 0.0)
            usable = float(s.get("usable_seconds", s.get("duration", 0.0)) or 0.0)
            ranges = sorted(shots.get(aid, []))
            shown = "、".join(f"{clock(a)}–{clock(b)}" for a, b in ranges[:4])
            if len(ranges) > 4:
                shown += f"…等 {len(ranges)} 段"
            n_drop = int(s.get("dropped", drops.get(aid, 0)) or 0)
            pct = f"（{100 * used / usable:.0f}%）" if usable > 0 else ""
            cells = [str(s.get("name") or aid), f"{used:.1f} / {usable:.1f} 秒{pct}",
                     shown or "（沒有用到）", f"{n_drop} 段" if n_drop else "—"]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if c == 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if used <= 0 and c != 0:
                    it.setForeground(QColor(theme.WARN))
                if c == 2 and ranges:
                    it.setToolTip("\n".join(f"{clock(a)}–{clock(b)}" for a, b in ranges))
                self.setItem(r, c, it)
        self.resizeRowsToContents()

    def rows_text(self) -> list[list[str]]:
        return [[self.item(r, c).text() for c in range(self.columnCount())] for r in range(self.rowCount())]


# ================================================================================================ preview
class PreviewDialog(QDialog):
    """Non-modal player for ONE candidate on a throw-away project (``api.preview_project``).

    The monitor / playback controller are the editor's own; they run on a private ``AppState``, so the real project and
    its undo history are untouched.  Closing stops the decoder thread and closes the ``LiveCompositor``.
    """

    def __init__(self, scratch: Project, seq_id: str, title: str, parent: Optional[QWidget] = None,
                 autoplay: bool = True) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self._closed = False
        self.preview_state = AppState(scratch)
        self.preview_state.set_sequence(seq_id)
        self.playback = PlaybackController(self.preview_state, self)
        self.monitor = MonitorWidget(self.preview_state, self.playback)
        for name in ("btn_reframe", "btn_range"):          # nothing here may edit (even the throw-away copy)
            btn = getattr(self.monitor, name, None)
            if btn is not None:
                btn.setVisible(False)
        seq = scratch.sequence(seq_id)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        hint = QLabel("這是暫時的預覽：不會改動你的專案，關閉後就消失。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addWidget(self.monitor, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_close = QPushButton("關閉")
        self.btn_close.setAutoDefault(False)
        self.btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        portrait = seq.height >= seq.width
        avail = (QGuiApplication.primaryScreen().availableGeometry() if QGuiApplication.primaryScreen() else None)
        w, h = (540, 800) if portrait else (860, 640)
        if avail is not None:
            w, h = min(w, max(avail.width() - 40, 460)), min(h, max(avail.height() - 80, 520))
        self.setMinimumSize(440, 480)
        self.resize(w, h)
        self._autoplay = autoplay

    def showEvent(self, ev) -> None:                        # noqa: N802 (Qt API)
        super().showEvent(ev)
        if self._autoplay and not self._closed:
            self._autoplay = False
            try:
                self.playback.play()
            except Exception:                               # noqa: BLE001 - a preview that cannot play stays a still
                pass

    def shutdown(self) -> None:
        """Stop playback and the decoder thread, then release the compositor's open files (idempotent)."""
        if self._closed:
            return
        self._closed = True
        for step in (self.playback.close, self.monitor.shutdown, self.monitor.compositor.close):
            try:
                step()
            except Exception:                               # noqa: BLE001
                pass

    def done(self, result: int) -> None:                    # noqa: N802 (Qt API)
        self.shutdown()
        super().done(result)

    def closeEvent(self, ev) -> None:                       # noqa: N802 (Qt API)
        self.shutdown()
        super().closeEvent(ev)
