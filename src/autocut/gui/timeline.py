"""Self-drawn multi-track timeline (GUI_SPEC §2, §11.1).

``TimelineWidget(state)`` paints the caption lane, the video lanes (V(n)…V1) and the audio lanes (A1…An) with
QPainter (only the visible region is painted) and turns mouse / keyboard / drag-and-drop gestures into exactly ONE
undoable ``state.op`` / ``state.do`` per gesture.  While a gesture is running the widget only paints a ghost: it never
touches ``Clip`` / ``Track`` fields itself.

Signals
    externalFilesDropped(list paths, float t, str track_id)   OS files dropped on the timeline (nothing else happens)
    revealAssetRequested(str asset_id)                        context menu 「在素材庫顯示」
    propertiesRequested(str clip_id)                          context menu 「屬性」 / double click on a clip
    snappingChanged(bool)                                     the 「吸附」 toggle of the bottom bar was flipped
    zoomChanged(float)                                        pixels-per-second changed

Geometry helpers (also used by the tests): ``pps``, ``time_to_x``, ``x_to_time``, ``clip_rect``, ``cue_rect``,
``track_rect``, ``header_rect``, ``fade_handle_rect``, ``horizontal_scroll`` / ``vertical_scroll`` (+ setters).

Interaction summary (all committed on mouse release as one undo step)
    drag clip body            move (same-kind lanes; Ctrl at release = insert, default overwrite); snaps to clip
                              edges / playhead / markers / 0 within 8 px unless snapping is off or Alt is held
    drag clip edge            trim (Shift = ripple); the edge of a clip that overlaps its predecessor edits the crossfade
    Alt + drag clip body      slip;     drag the top-corner handles: fade in / fade out
    drag caption block        move / resize edges; double click: inline text edit
    drop from the media bin   place_clip (video -> video lane, audio-only -> audio lane, several files in order;
                              Ctrl = insert); OS files -> externalFilesDropped (the window imports, then places)
    right click               context menus (clip / caption block / track header / empty space / ruler / marker):
                              ``menu_for_hit(hit)`` builds the QMenu, every entry is also a public method
    Ctrl+C / Ctrl+V           copy the selected clips / paste them at the playhead

Editing commands that the menus and MainWindow call (each one undo step)
    split_at_playhead, delete_selection, add_marker_at_playhead, set_in/out_at_playhead, clear_range, step_frames,
    goto_prev_cut / goto_next_cut, zoom_in / zoom_out / zoom_fit, select_all, set_snapping, set_playing,
    toggle_mute_selected, toggle_enabled_selected, set_speed_selected, set_fades_selected, set_crossfade_selected,
    reframe_reset_selected / reframe_add_key / reframe_clear_keys, copy_selection / paste_at,
    add_track / remove_track / rename_track, close_gaps, add_marker_at / remove_marker,
    add_cue_at / edit_cue_text / merge_selected_cues
"""
from __future__ import annotations

import bisect
import copy
import json
import math
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QFontMetrics, QImage, QKeyEvent, QMouseEvent, QPainter,
                           QPen, QPixmap, QPolygonF, QWheelEvent)
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QScrollBar, QSlider, QToolButton,
                               QToolTip, QVBoxLayout, QWidget)

from ..core import ops
from ..core.geometry import interp_keys
from ..core.ops import IMAGE_DEFAULT, MAX_SPEED, MIN_CUE, MIN_SPEED, OpError
from ..core.project import AUDIO, VIDEO, Asset, CaptionCue, Clip, Sequence, Track
from ..core.timecode import EPS, MAX_TIMELINE, fmt_clock, fmt_timecode, snap_time
from ..media import cache as media_cache
from ..util import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS, clamp
from . import theme
from .state import AppState
from .workers import TaskRunner

__all__ = ["TimelineWidget", "VisualCache", "ASSET_MIME"]

ASSET_MIME = "application/x-autocut-asset"
CAPTION_LANE = "__captions__"

# ---- geometry (device-independent pixels)
PPS_MIN, PPS_MAX, PPS_DEFAULT = 2.0, 1500.0, 80.0
PX_LIMIT = 1 << 28                 # no widget coordinate needs to be further out than this; QRect and the scroll bars take 32-bit ints


def clamp_px(x: float) -> int:
    """A widget coordinate as the ``int`` a ``QRect`` takes.  The core keeps every time within a day, but a damaged project file
    can hold any number, and a coordinate beyond 32 bits raises OverflowError in the middle of a layout or a paint."""
    return int(max(-PX_LIMIT, min(PX_LIMIT, round(x)))) if x == x else 0
HEADER_W = 132
RULER_H = 34
MARKER_ROW_H = 13
BAR_H = 26
VBAR_W = 12
LANE_GAP = 2
H_CAPTION, H_VIDEO, H_AUDIO = 28, 62, 48
H_CAPTION_CARDS = 46            # the caption lane when the sequence has title cards (字卡): a card band above the subtitles
CLIP_INSET = 2
EDGE_PX = 8              # trim / resize zone
SNAP_PX = 8              # snapping threshold
DRAG_PX = 3              # movement before a press becomes a drag
LABEL_H = 15
WAVE_BAND_H = 14
HANDLE_R = 5
WAVE_RATE = 50           # waveform peaks per second (media.cache.waveform default)
STRIP_H = 54             # filmstrip tile height (media.cache.filmstrip default)
EMPTY_HINT = "把影片拖到這裡，或用『自動剪輯』"


# ============================================================================ pure helpers (no widgets)
def fmt_dur(d: float) -> str:
    """Short duration label for clip blocks: ``5.2s`` / ``1:05`` / ``1:02:03``."""
    d = max(0.0, float(d))
    if d < 60:
        return f"{d:.1f}s"
    m, s = divmod(int(round(d)), 60)
    return f"{m}:{s:02d}" if m < 60 else f"{m // 60}:{m % 60:02d}:{s:02d}"


_FRAME_STEPS = ((1, 0), (2, 2), (5, 5), (10, 5), (15, 3))                       # (frames, minor subdivisions)
_SEC_STEPS = ((1, 5), (2, 4), (5, 5), (10, 5), (15, 3), (30, 6), (60, 6), (120, 4), (300, 5), (600, 5),
              (900, 3), (1800, 6), (3600, 6), (7200, 4), (14400, 4), (28800, 4))


def ruler_step(pps: float, fps: float, min_px: float = 84.0) -> tuple[float, int]:
    """``(major step in seconds, minor subdivisions)``: the smallest nice step whose major ticks are ``min_px`` apart."""
    frame = 1.0 / max(1.0, fps)
    for k, sub in _FRAME_STEPS:
        if k * frame < 1.0 - 1e-9 and k * frame * pps >= min_px:
            return k * frame, sub
    for s, sub in _SEC_STEPS:
        if s * pps >= min_px:
            return float(s), sub
    return 28800.0, 4


def fmt_ruler(t: float, step: float, fps: float) -> str:
    """Tick label: ``MM:SS:FF`` below one second per tick, ``m:ss`` / ``h:mm:ss`` above."""
    if step < 1.0 - 1e-9:
        tc = fmt_timecode(t, fps)
        return tc[3:] if tc.startswith("00:") else tc
    total = int(round(t))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def column_peaks(padded: np.ndarray, src0: float, src1: float, cols: int, rate: int = WAVE_RATE) -> np.ndarray:
    """Max waveform peak of every pixel column covering source time ``src0..src1``.

    ``padded`` is the peak array with one extra trailing 0 (so ``reduceat`` may index ``len``)."""
    n = len(padded) - 1
    if cols <= 0:
        return np.zeros(0, np.float32)
    if n <= 0 or src1 <= src0:
        return np.zeros(cols, np.float32)
    edges = np.linspace(src0 * rate, src1 * rate, cols + 1)
    lo = np.floor(edges[:-1]).astype(np.int64)
    hi = np.ceil(edges[1:]).astype(np.int64)
    valid = (hi > 0) & (lo < n)
    lo = np.clip(lo, 0, n - 1)
    hi = np.minimum(np.maximum(hi, lo + 1), n)
    idx = np.empty(2 * cols, np.int64)
    idx[0::2], idx[1::2] = lo, hi
    out = np.maximum.reduceat(padded, idx)[0::2]
    return np.where(valid, out, 0.0).astype(np.float32)


def wave_image(amps: np.ndarray, h: int, color: QColor) -> tuple[QImage, np.ndarray]:
    """Mirrored waveform bars as an ARGB image (vectorised); the array must outlive the image."""
    n = len(amps)
    half = np.maximum(np.minimum(1.0, amps ** 0.6) * (h / 2.0), 0.6)
    ys = np.abs(np.arange(h, dtype=np.float32)[:, None] + 0.5 - h / 2.0)
    img = np.zeros((h, n, 4), np.uint8)
    a = color.alpha()
    img[ys <= half[None, :]] = (color.blue() * a // 255, color.green() * a // 255, color.red() * a // 255, a)
    return QImage(img.data, n, h, n * 4, QImage.Format_ARGB32_Premultiplied), img


def nearest_shift(edges: Iterable[float], points: list[float], thresh: float) -> Optional[tuple[float, float]]:
    """Smallest ``(shift, point)`` moving one of ``edges`` onto a sorted snap ``points`` entry within ``thresh``."""
    best: Optional[tuple[float, float]] = None
    for e in edges:
        i = bisect.bisect_left(points, e)
        for j in (i - 1, i):
            if 0 <= j < len(points):
                d = points[j] - e
                if abs(d) <= thresh and (best is None or abs(d) < abs(best[0])):
                    best = (d, points[j])
    return best


def _qc(hex_: str, alpha: Optional[int] = None) -> QColor:
    c = QColor(hex_)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


class _Palette:
    """Pre-built colours/pens (theme constants are hex strings)."""

    def __init__(self) -> None:
        self.bg, self.lane, self.lane_alt = _qc(theme.TL_BG), _qc(theme.TL_LANE), _qc(theme.TL_LANE_ALT)
        self.grid, self.ruler, self.header = _qc(theme.TL_GRID), _qc(theme.TL_RULER), _qc(theme.PANEL_ALT)
        self.text, self.dim, self.accent = _qc(theme.TEXT), _qc(theme.TEXT_DIM), _qc(theme.ACCENT)
        self.video, self.audio, self.image = _qc(theme.CLIP_VIDEO), _qc(theme.CLIP_AUDIO), _qc(theme.CLIP_IMAGE)
        self.caption, self.disabled = _qc(theme.CLIP_CAPTION), _qc(theme.CLIP_DISABLED)
        self.card = _qc(theme.CLIP_CARD)
        self.selected, self.playhead = _qc(theme.CLIP_SELECTED), _qc(theme.PLAYHEAD)
        self.marker, self.snap, self.error = _qc(theme.MARKER), _qc(theme.SNAP_LINE), _qc(theme.ERROR)
        self.range_fill = _qc(theme.ACCENT, 0x55)           # theme.RANGE_FILL is RRGGBBAA, Qt reads AARRGGBB
        self.warn, self.ok = _qc(theme.WARN), _qc(theme.OK)

    def clip_base(self, asset: Optional[Asset], custom: str) -> QColor:
        if custom:
            c = QColor(custom)
            if c.isValid():
                return c
        if asset is None:
            return self.disabled
        return {"audio": self.audio, "image": self.image}.get(asset.kind, self.video)


# ============================================================================ async filmstrips / waveforms
@dataclass
class Strip:
    """One horizontal filmstrip image: ``tiles`` equally wide frames spread evenly over the whole asset."""
    pixmap: QPixmap
    tiles: int

    @property
    def tile_w(self) -> float:
        return self.pixmap.width() / max(1, self.tiles)

    @property
    def aspect(self) -> float:
        return self.tile_w / max(1, self.pixmap.height())


def strip_count(asset: Asset) -> int:
    """Number of filmstrip tiles requested for an asset (deterministic, so the disk cache is reused)."""
    if asset.kind == "image" or asset.duration <= 0:
        return 1
    return int(clamp(round(asset.duration / 4.0), 12, 48))


class VisualCache(QObject):
    """Filmstrip pixmaps and waveform peaks per asset, produced on worker threads (never on the GUI thread).

    ``strip(asset)`` / ``peaks(asset)`` return the cached object or ``None`` and queue the load; ``ready(asset_id)``
    fires when a result arrived so the owner can repaint.  At most ``MAX_PARALLEL`` jobs run at once (the machine
    may be small); filmstrips go before waveforms.
    """

    ready = Signal(str)
    MAX_PARALLEL = 2
    MAX_STRIPS = 48

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._strips: "OrderedDict[tuple, Strip]" = OrderedDict()
        self._peaks: dict[tuple, np.ndarray] = {}
        self._queue: dict[str, deque] = {"strip": deque(), "wave": deque()}
        self._queued: set[tuple] = set()
        self._failed: set[tuple] = set()
        self._running: dict[tuple, TaskRunner] = {}

    @staticmethod
    def _key(kind: str, a: Asset) -> tuple:
        return (kind, a.id, a.path, a.fingerprint)

    # ---- lookups (cheap; queue the load on a miss)
    def strip(self, asset: Asset) -> Optional[Strip]:
        key = self._key("strip", asset)
        s = self._strips.get(key)
        if s is not None:
            self._strips.move_to_end(key)
            return s
        self._want("strip", key, asset)
        return None

    def peaks(self, asset: Asset) -> Optional[np.ndarray]:
        """Peak array (one extra trailing 0) or ``None`` while loading / unavailable."""
        key = self._key("wave", asset)
        p = self._peaks.get(key)
        if p is None:
            self._want("wave", key, asset)
        return p

    def runners(self) -> list[TaskRunner]:
        return list(self._running.values())

    def idle(self) -> bool:
        return not self._running and not self._queue["strip"] and not self._queue["wave"]

    # ---- scheduling
    def _want(self, kind: str, key: tuple, asset: Asset) -> None:
        if key in self._queued or key in self._failed:
            return
        self._queued.add(key)
        self._queue[kind].append((key, Asset.from_dict(asset.to_dict())))
        self._pump()

    def _pump(self) -> None:
        while len(self._running) < self.MAX_PARALLEL:
            q = self._queue["strip"] or self._queue["wave"]
            if not q:
                return
            key, asset = q.popleft()
            runner = TaskRunner(_visual_job(key, asset), None, "timeline-visual")
            runner.finished.connect(self._on_finished)
            self._running[key] = runner
            runner.start()

    def _on_finished(self, result: Any) -> None:                         # GUI thread
        key, payload = result
        self._running.pop(key, None)
        if payload is None:
            self._failed.add(key)
        elif key[0] == "strip":
            img, tiles = payload
            self._strips[key] = Strip(QPixmap.fromImage(img), tiles)
            while len(self._strips) > self.MAX_STRIPS:
                self._strips.popitem(last=False)
        else:
            self._peaks[key] = np.append(np.asarray(payload, np.float32), np.float32(0.0))
        if payload is not None:
            self.ready.emit(key[1])
        self._pump()


def _visual_job(key: tuple, asset: Asset) -> Callable:
    """Worker function: ``(key, payload-or-None)``; any failure simply means 'no picture'."""

    def job(progress, cancel):
        try:
            if key[0] == "strip":
                count = strip_count(asset)
                img = QImage(str(media_cache.filmstrip(asset, count=count, height=STRIP_H)))
                if img.isNull():
                    return key, None
                aspect = asset.aspect or 16 / 9
                tiles = max(1, min(count, int(round(img.width() / (img.height() * aspect)))))
                return key, (img, tiles)
            return key, media_cache.waveform(asset, per_second=WAVE_RATE)
        except Exception:                                    # noqa: BLE001 - ffmpeg trouble = no thumbnails
            return key, None
    return job


# ============================================================================ layout / hit / overlay records
@dataclass
class Lane:
    """One row of the timeline (``y`` / ``h`` are in content coordinates, i.e. before vertical scrolling)."""
    kind: str                      # caption | video | audio
    track: Optional[Track]
    y: int
    h: int
    idx: int = 0

    @property
    def id(self) -> str:
        return self.track.id if self.track is not None else CAPTION_LANE

    @property
    def name(self) -> str:
        return self.track.name if self.track is not None else "字幕"


@dataclass(frozen=True)
class Hit:
    """Result of ``TimelineWidget.hit_test``.

    kind: corner | ruler | marker | header | header_mute | header_lock | clip | clip_l | clip_r | xfade | fade_in |
          fade_out | cue | cue_l | cue_r | lane | caption_lane | none
    """
    kind: str = "none"
    lane: Optional[Lane] = None
    clip_id: str = ""
    cue_id: str = ""
    marker_id: str = ""
    t: float = 0.0


@dataclass
class Overlay:
    """Everything a running gesture wants painted on top of the project (the project itself is never touched)."""
    ghosts: list = field(default_factory=list)          # [(QRectF, style)]  style: move | trim | slip | drop | cue
    lane: Optional[QRectF] = None                       # highlighted target lane
    snap_t: Optional[float] = None                      # snap guide (seconds)
    tip: Optional[tuple[QPointF, str]] = None           # live tooltip (cursor position, text)
    band: Optional[QRectF] = None                       # rubber band
    preview_sel: set = field(default_factory=set)       # clip ids highlighted by the rubber band
    preview_cues: set = field(default_factory=set)
    dim: set = field(default_factory=set)               # clip / cue ids drawn faint (their ghost is elsewhere)
    fade: Optional[tuple[str, str, float]] = None       # (clip id, "in" | "out", seconds) while dragging a handle
    slip: Optional[tuple[str, float]] = None            # (clip id, new src_in)
    drop_hint: bool = False                             # something is being dragged over (hides the empty hint)

    def empty(self) -> bool:
        return not (self.ghosts or self.lane or self.snap_t is not None or self.tip or self.band or self.preview_sel
                    or self.preview_cues or self.dim or self.fade or self.slip or self.drop_hint)


# ============================================================================ painting
class _Painter:
    """All QPainter code of the timeline. Reads the widget's geometry/state; paints only the dirty ``rect``."""

    def __init__(self, tl: "TimelineWidget") -> None:
        self.tl = tl
        self.pal = _Palette()
        base = QFont(tl.font())

        def mk(px: int, bold: bool = False) -> QFont:
            f = QFont(base)
            f.setPixelSize(px)
            f.setBold(bold)
            return f

        self.f_label, self.f_small, self.f_head, self.f_hint = mk(11), mk(10), mk(12, True), mk(15)
        self.fm_label, self.fm_small = QFontMetrics(self.f_label), QFontMetrics(self.f_small)
        self._elide_cache: dict[tuple, str] = {}
        self._hatch = QBrush(_qc(theme.ERROR, 190), Qt.BDiagPattern)
        self._lock_hatch = QBrush(QColor(255, 255, 255, 22), Qt.BDiagPattern)

    # ---------------------------------------------------------------- helpers
    def elide(self, fm: QFontMetrics, text: str, width: float) -> str:
        w = int(width)
        if w <= 4:
            return ""
        key = (id(fm), text, w)
        s = self._elide_cache.get(key)
        if s is None:
            if len(self._elide_cache) > 4000:
                self._elide_cache.clear()
            s = self._elide_cache[key] = fm.elidedText(text, Qt.ElideRight, w)
        return s

    @staticmethod
    def _crisp(x: float) -> float:
        return math.floor(x) + 0.5

    # ---------------------------------------------------------------- entry point
    def paint(self, p: QPainter, rect: QRect) -> None:
        tl = self.tl
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(rect, self.pal.bg)
        area = rect & tl._tracks_view_rect()
        if not area.isEmpty():
            p.save()
            p.setClipRect(area)
            self._lanes(p, area)
            self._clips(p, area)
            self._cues(p, area)
            self._overlay(p, area)
            self._playhead_line(p, area)
            self._hint(p)
            p.restore()
        self._headers(p, rect)
        self._ruler(p, rect)
        self._corner(p, rect)
        self._tip(p, rect)

    # ---------------------------------------------------------------- lanes, grid, range, markers
    def _lanes(self, p: QPainter, area: QRect) -> None:
        tl = self.tl
        view_w = tl._view_w()
        for lane in tl._lanes_in(area):
            top = tl._lane_top(lane)
            r = QRectF(HEADER_W, top, view_w, lane.h)
            p.fillRect(r, self.pal.lane if lane.idx % 2 == 0 else self.pal.lane_alt)
            if lane.track is not None and lane.track.locked:
                p.fillRect(r, self._lock_hatch)
        step, _ = ruler_step(tl.pps, tl.fps)
        t_lo, t_hi = tl.x_to_time(area.left()), tl.x_to_time(area.right())
        top, bottom = float(area.top()), float(area.bottom() + 1)
        lines: list[QPointF] = []
        k = int(max(0.0, t_lo) // step)
        p.setPen(QPen(self.pal.grid, 1))
        while k * step <= t_hi:
            x = self._crisp(tl.time_to_x(k * step))
            lines.append(QPointF(x, top))
            lines.append(QPointF(x, bottom))
            k += 1
        if lines:
            p.drawLines(lines)
        rng = tl._range()
        if rng:
            x0, x1 = tl.time_to_x(rng[0]), tl.time_to_x(rng[1])
            p.fillRect(QRectF(x0, top, x1 - x0, bottom - top), QColor(76, 155, 232, 22))
            p.setPen(QPen(_qc(theme.ACCENT, 150), 1, Qt.DashLine))
            for x in (x0, x1):
                p.drawLine(QPointF(self._crisp(x), top), QPointF(self._crisp(x), bottom))
        seq = tl._seq()
        if seq:
            for m in seq.markers:
                x = tl.time_to_x(m.time)
                if area.left() - 2 <= x <= area.right() + 2:
                    p.setPen(QPen(_qc(m.color if QColor(m.color).isValid() else theme.MARKER, 110), 1, Qt.DashLine))
                    p.drawLine(QPointF(self._crisp(x), top), QPointF(self._crisp(x), bottom))

    def _playhead_line(self, p: QPainter, area: QRect) -> None:
        x = self.tl.time_to_x(self.tl.state.playhead)
        if area.left() - 2 <= x <= area.right() + 2:
            p.setPen(QPen(self.pal.playhead, 1.5))
            p.drawLine(QPointF(self._crisp(x), area.top()), QPointF(self._crisp(x), area.bottom() + 1))

    def _hint(self, p: QPainter) -> None:
        tl = self.tl
        seq = tl._seq()
        if seq is None or tl._ov.drop_hint or any(True for _ in seq.all_clips()) or seq.cues:
            return
        p.setFont(self.f_hint)
        p.setPen(self.pal.dim)
        p.drawText(QRectF(tl._tracks_view_rect()), Qt.AlignCenter, EMPTY_HINT)

    # ---------------------------------------------------------------- clips
    def _clips(self, p: QPainter, area: QRect) -> None:
        tl = self.tl
        t_lo, t_hi = tl.x_to_time(area.left()), tl.x_to_time(area.right())
        for lane in tl._lanes_in(area):
            if lane.track is None:
                continue
            clips = lane.track.clips
            for c in clips:
                if c.end < t_lo:
                    continue
                if c.start > t_hi:
                    break
                self._clip(p, lane, c, area)
            self._crossfades(p, lane, clips, t_lo, t_hi)

    def _clip(self, p: QPainter, lane: Lane, c: Clip, area: QRect) -> None:
        tl, pal, ov, st = self.tl, self.pal, self.tl._ov, self.tl.state
        a = st.project.assets.get(c.asset_id)
        x0 = tl.time_to_x(c.start)
        body = QRectF(x0, tl._lane_top(lane) + CLIP_INSET, max(1.5, c.duration * tl.pps), lane.h - 2 * CLIP_INSET)
        vis = body.intersected(QRectF(area))
        if vis.isEmpty():
            return
        fade_in, fade_out, src_in = tl._effective(c)
        selected = c.id in st.selection or c.id in ov.preview_sel
        missing = a is not None and tl._is_missing(a)
        opacity = 0.4 if c.id in ov.dim else 0.45 if not c.enabled else 0.6 if lane.track.muted else 1.0
        p.save()
        p.setOpacity(opacity)
        p.setPen(Qt.NoPen)
        p.setBrush(pal.clip_base(a, c.color) if c.enabled else pal.disabled)
        p.drawRoundedRect(body, 4, 4)
        if body.width() >= 10 and a is not None and not missing:
            self._content(p, lane, c, a, body, vis, src_in)
        if body.width() >= 6:
            self._label(p, lane, c, a, body, vis, missing)
            self._fades(p, body, c, fade_in, fade_out)
            if lane.kind == "video":
                self._keys(p, c, body, vis)
        if missing:
            p.setBrush(self._hatch)
            p.drawRoundedRect(body, 4, 4)
        p.setBrush(Qt.NoBrush)
        if selected:
            p.setPen(QPen(pal.selected, 2))
            p.drawRoundedRect(body.adjusted(1, 1, -1, -1), 4, 4)
        else:
            p.setPen(QPen(QColor(0, 0, 0, 120), 1))
            p.drawRoundedRect(body.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        p.restore()
        if body.width() >= 40 and (selected or c.id == tl._hover_clip):
            self._handles(p, lane, c, body, fade_in, fade_out)

    def _content(self, p: QPainter, lane: Lane, c: Clip, a: Asset, body: QRectF, vis: QRectF, src_in: float) -> None:
        tl = self.tl
        p.save()
        p.setClipRect(vis)
        if lane.kind == "video":
            strip = tl._visuals.strip(a) if (a.has_video or a.kind == "image") else None
            if strip is not None:
                self._filmstrip(p, strip, a, c, src_in, body, vis)
            if a.has_audio and a.kind != "image" and not c.muted and body.width() >= 16:
                peaks = tl._visuals.peaks(a)
                if peaks is not None:
                    band = QRectF(body.left(), body.bottom() - WAVE_BAND_H - 1, body.width(), WAVE_BAND_H)
                    p.fillRect(band.intersected(vis), QColor(0, 0, 0, 80))
                    self._wave(p, peaks, c, src_in, band, vis, QColor(255, 255, 255, 190), body.left())
        elif a.has_audio:
            peaks = tl._visuals.peaks(a)
            if peaks is not None:
                area = QRectF(body.left(), body.top() + LABEL_H, body.width(), body.height() - LABEL_H - 2)
                col = QColor(255, 255, 255, 110 if c.muted else 200)
                self._wave(p, peaks, c, src_in, area, vis, col, body.left())
        p.restore()

    def _filmstrip(self, p: QPainter, strip: Strip, a: Asset, c: Clip, src_in: float, body: QRectF,
                   vis: QRectF) -> None:
        pps = self.tl.pps
        th = body.height()
        tw = max(6.0, th * strip.aspect)
        left = body.left()
        i0, i1 = max(0, int((vis.left() - left) // tw)), int((vis.right() - left) // tw)
        dur = a.duration if a.duration > 0 else 1.0
        pix, sh, stw = strip.pixmap, strip.pixmap.height(), strip.tile_w
        for i in range(i0, i1 + 1):
            src_t = src_in + ((i + 0.5) * tw / pps) * c.speed
            k = 0 if strip.tiles == 1 else int(clamp(int(src_t / dur * strip.tiles + 0.5), 0, strip.tiles - 1))
            p.drawPixmap(QRectF(left + i * tw, body.top(), tw, th), pix, QRectF(k * stw, 0, stw, sh))

    def _wave(self, p: QPainter, peaks: np.ndarray, c: Clip, src_in: float, area: QRectF, vis: QRectF,
              color: QColor, body_left: float) -> None:
        h = int(area.height())
        ix0, ix1 = int(math.floor(vis.left())), int(math.ceil(vis.right()))
        if h < 4 or ix1 <= ix0:
            return
        pps = self.tl.pps
        s0 = src_in + ((ix0 - body_left) / pps) * c.speed
        s1 = src_in + ((ix1 - body_left) / pps) * c.speed
        img, keep = wave_image(column_peaks(peaks, s0, s1, ix1 - ix0), h, color)
        p.drawImage(QPointF(ix0, area.top()), img)
        del keep

    def _label(self, p: QPainter, lane: Lane, c: Clip, a: Optional[Asset], body: QRectF, vis: QRectF,
               missing: bool) -> None:
        if body.width() < 30:
            return
        bar = QRectF(body.left(), body.top(), body.width(), LABEL_H)
        p.fillRect(bar.adjusted(0.5, 0.5, -0.5, 0), QColor(0, 0, 0, 105))
        # clips wide enough to show fade handles keep their text clear of the 8 px handle squares
        pad_l, pad_r = (13.0, 12.0) if body.width() >= 40 else (5.0, 4.0)
        left = max(body.left() + pad_l, vis.left() + 5)
        right = min(body.right() - pad_r, vis.right() - 4)
        badges = self._badges(p, c, a, lane, QRectF(left, bar.top(), right - left, LABEL_H))
        name = c.label or (a.name if a is not None else "（素材已移除）")
        if missing:
            name = "素材遺失  " + name
        text = self.elide(self.fm_label, name, right - left - badges - (46 if body.width() > 110 else 0))
        p.setFont(self.f_label)
        p.setPen(self.pal.error.lighter(150) if missing else QColor(255, 255, 255, 235))
        p.drawText(QRectF(left, bar.top(), right - left, LABEL_H), Qt.AlignLeft | Qt.AlignVCenter, text)
        if body.width() > 110:
            p.setPen(QColor(255, 255, 255, 170))
            p.drawText(QRectF(left, bar.top(), right - left - badges, LABEL_H), Qt.AlignRight | Qt.AlignVCenter,
                       fmt_dur(c.duration))

    def _badges(self, p: QPainter, c: Clip, a: Optional[Asset], lane: Lane, r: QRectF) -> float:
        """Speed pill and mute icon at the right of the label bar; returns the width they take."""
        used = 0.0
        right = r.right()
        if c.muted and a is not None and a.has_audio and a.kind != "image" and r.width() > 70:
            used += self._mute_icon(p, right - 12, r.center().y())
        if abs(c.speed - 1.0) > 1e-3 and r.width() > 60:
            txt = f"{c.speed:g}×"
            w = self.fm_small.horizontalAdvance(txt) + 8
            pill = QRectF(right - used - w - (44 if r.width() > 110 else 0), r.top() + 2, w, LABEL_H - 4)
            p.setPen(Qt.NoPen)
            p.setBrush(self.pal.accent.darker(130))
            p.drawRoundedRect(pill, 3, 3)
            p.setFont(self.f_small)
            p.setPen(QColor(255, 255, 255))
            p.drawText(pill, Qt.AlignCenter, txt)
            used += w + 4
        return used

    @staticmethod
    def _mute_icon(p: QPainter, cx: float, cy: float) -> float:
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.2))
        p.setBrush(QColor(255, 255, 255, 220))
        p.drawPolygon(QPolygonF([QPointF(cx - 6, cy - 2), QPointF(cx - 3, cy - 2), QPointF(cx + 1, cy - 5),
                                 QPointF(cx + 1, cy + 5), QPointF(cx - 3, cy + 2), QPointF(cx - 6, cy + 2)]))
        p.setPen(QPen(QColor(239, 83, 80), 1.6))
        p.drawLine(QPointF(cx - 6, cy + 5), QPointF(cx + 5, cy - 5))
        p.setBrush(Qt.NoBrush)
        return 16.0

    def _fades(self, p: QPainter, body: QRectF, c: Clip, fade_in: float, fade_out: float) -> None:
        pps = self.tl.pps
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 95))
        lines = []
        if fade_in * pps >= 1.5:
            w = min(fade_in * pps, body.width())
            p.drawPolygon(QPolygonF([body.topLeft(), QPointF(body.left() + w, body.top()), body.bottomLeft()]))
            lines.append((QPointF(body.left(), body.bottom()), QPointF(body.left() + w, body.top())))
        if fade_out * pps >= 1.5:
            w = min(fade_out * pps, body.width())
            p.drawPolygon(QPolygonF([body.topRight(), QPointF(body.right() - w, body.top()), body.bottomRight()]))
            lines.append((QPointF(body.right() - w, body.top()), QPointF(body.right(), body.bottom())))
        p.setPen(QPen(QColor(255, 255, 255, 190), 1))
        for a, b in lines:
            p.drawLine(a, b)
        p.setBrush(Qt.NoBrush)

    def _keys(self, p: QPainter, c: Clip, body: QRectF, vis: QRectF) -> None:
        if not c.transform.keys:
            return
        pps = self.tl.pps
        y = body.bottom() - 8
        p.setPen(QPen(QColor(0, 0, 0, 200), 1))
        p.setBrush(QColor(255, 224, 102))
        for k in c.transform.keys:
            x = body.left() + float(k[0]) * pps
            if vis.left() - 5 <= x <= vis.right() + 5 and float(k[0]) <= c.duration + EPS:
                p.drawPolygon(QPolygonF([QPointF(x, y - 5), QPointF(x + 4, y), QPointF(x, y + 5), QPointF(x - 4, y)]))
        p.setBrush(Qt.NoBrush)

    def _handles(self, p: QPainter, lane: Lane, c: Clip, body: QRectF, fade_in: float, fade_out: float) -> None:
        p.setPen(QPen(QColor(0, 0, 0, 210), 1))
        p.setBrush(QColor(255, 255, 255, 235))
        for which in ("in", "out"):
            pt = self.tl._fade_handle_pos(lane, c, which, fade_in, fade_out)
            p.drawRoundedRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8), 2, 2)
        p.setBrush(Qt.NoBrush)

    def _crossfades(self, p: QPainter, lane: Lane, clips: list, t_lo: float, t_hi: float) -> None:
        tl = self.tl
        top, h = tl._lane_top(lane) + CLIP_INSET, lane.h - 2 * CLIP_INSET
        for i in range(1, len(clips)):
            b = clips[i]
            if b.start > t_hi:
                break
            for a in clips[max(0, i - 2):i]:
                end = min(a.end, b.end)
                if end <= b.start + EPS or end < t_lo:
                    continue
                x0, x1 = tl.time_to_x(b.start), tl.time_to_x(end)
                self._xfade(p, QRectF(x0, top, x1 - x0, h), end - b.start)

    def _xfade(self, p: QPainter, r: QRectF, dur: float) -> None:
        p.save()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 80))
        p.drawRect(r)
        p.setBrush(QColor(255, 255, 255, 80))
        p.drawPolygon(QPolygonF([r.topLeft(), r.bottomRight(), r.bottomLeft()]))
        p.setBrush(QColor(255, 255, 255, 45))
        p.drawPolygon(QPolygonF([r.bottomLeft(), r.topRight(), r.bottomRight()]))
        p.setPen(QPen(QColor(255, 255, 255, 215), 1.3))
        p.drawLine(r.topLeft(), r.bottomRight())
        p.drawLine(r.bottomLeft(), r.topRight())
        if r.width() >= 44:
            # duration pill at the bottom centre: the top of the overlap belongs to the clip label bar
            txt = f"{dur:.2f}s"
            p.setFont(self.f_small)
            tw = self.fm_small.horizontalAdvance(txt) + 8
            pill = QRectF(r.center().x() - tw / 2, r.bottom() - 17, tw, 14)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(pill, 3, 3)
            p.setPen(QColor(255, 255, 255, 240))
            p.drawText(pill, Qt.AlignCenter, txt)
        p.restore()

    # ---------------------------------------------------------------- caption cues
    def _cues(self, p: QPainter, area: QRect) -> None:
        tl, ov = self.tl, self.tl._ov
        seq, lane = tl._seq(), tl._lane_by_id.get(CAPTION_LANE)
        if seq is None or lane is None or not seq.cues:
            return
        top, h = tl._lane_top(lane) + CLIP_INSET, lane.h - 2 * CLIP_INSET
        if top + h < area.top() or top > area.bottom():
            return
        t_lo, t_hi = tl.x_to_time(area.left()), tl.x_to_time(area.right())
        sel = tl.state.cue_selection | ov.preview_cues
        p.setFont(self.f_label)
        for q in seq.cues:
            if q.end < t_lo:
                continue
            if q.start > t_hi:
                break
            x0 = tl.time_to_x(q.start)
            y0, bh = tl.cue_band(lane, q)
            r = QRectF(x0, y0, max(2.0, (q.end - q.start) * tl.pps), bh)
            p.setOpacity(0.4 if q.id in ov.dim else 1.0)
            p.setPen(Qt.NoPen)
            p.setBrush(self.pal.card if q.is_card else self.pal.caption)
            p.drawRoundedRect(r, 3, 3)
            if r.width() > 12:
                left = max(r.left(), area.left()) + 4
                label = ("字卡｜" if q.is_card else "") + q.text.replace("\n", " ")
                text = self.elide(self.fm_label, label, min(r.right(), area.right()) - left - 3)
                p.setPen(QColor(24, 26, 30))
                p.drawText(QRectF(left, r.top(), r.right() - left, r.height()), Qt.AlignLeft | Qt.AlignVCenter, text)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(self.pal.selected, 2) if q.id in sel else QPen(QColor(0, 0, 0, 110), 1))
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 3, 3)
        p.setOpacity(1.0)

    # ---------------------------------------------------------------- gesture overlay
    def _overlay(self, p: QPainter, area: QRect) -> None:
        ov = self.tl._ov
        if ov.lane is not None:
            p.fillRect(ov.lane, QColor(76, 155, 232, 36))
            p.setPen(QPen(_qc(theme.ACCENT, 140), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(ov.lane.adjusted(0.5, 0.5, -0.5, -0.5))
        for r, style in ov.ghosts:
            col = self.pal.ok if style == "drop" else self.pal.accent
            p.setPen(QPen(col, 1.5, Qt.DashLine))
            p.setBrush(QColor(col.red(), col.green(), col.blue(), 70))
            p.drawRoundedRect(r, 4, 4)
        if ov.band is not None:
            p.setPen(QPen(self.pal.accent, 1))
            p.setBrush(QColor(76, 155, 232, 40))
            p.drawRect(ov.band)
        if ov.snap_t is not None:
            x = self._crisp(self.tl.time_to_x(ov.snap_t))
            p.setPen(QPen(self.pal.snap, 1.2))
            p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom() + 1))
        p.setBrush(Qt.NoBrush)

    def _tip(self, p: QPainter, rect: QRect) -> None:
        tip = self.tl._ov.tip
        if tip is None:
            return
        pos, text = tip
        p.setFont(self.f_label)
        r = self.tl._tip_rect(pos, text, self.fm_label)
        if not r.intersects(QRectF(rect)):
            return
        p.setPen(QPen(self.pal.accent, 1))
        p.setBrush(QColor(37, 41, 48, 240))
        p.drawRoundedRect(r, 4, 4)
        p.setPen(self.pal.text)
        p.drawText(r, Qt.AlignCenter, text)
        p.setBrush(Qt.NoBrush)

    # ---------------------------------------------------------------- headers, corner, ruler
    def _headers(self, p: QPainter, rect: QRect) -> None:
        tl = self.tl
        area = QRect(0, RULER_H, HEADER_W, tl._view_h()) & rect
        if area.isEmpty():
            return
        p.save()
        p.setClipRect(area)
        p.fillRect(area, self.pal.bg)
        for lane in tl._lanes_in(area):
            self._header(p, lane)
        p.restore()

    def _header(self, p: QPainter, lane: Lane) -> None:
        tl, pal = self.tl, self.pal
        top = tl._lane_top(lane)
        p.fillRect(QRectF(0, top, HEADER_W - 1, lane.h), pal.header)
        chip = {"video": pal.video, "audio": pal.audio}.get(lane.kind, pal.caption)
        p.fillRect(QRectF(0, top, 4, lane.h), chip)
        tr = lane.track
        locked = tr is not None and tr.locked
        p.setFont(self.f_head)
        p.setPen(pal.dim if locked else pal.text)
        name_w = HEADER_W - 14 - (50 if tr is not None else 0)
        p.drawText(QRectF(10, top, name_w, lane.h), Qt.AlignLeft | Qt.AlignVCenter,
                   self.elide(QFontMetrics(self.f_head), lane.name, name_w))
        if tr is not None:
            for which, on, color, label in (("mute", tr.muted, pal.warn, "M"), ("lock", tr.locked, pal.error, "L")):
                r = QRectF(tl.header_button_rect(lane, which))
                p.setPen(QPen(pal.grid, 1))
                p.setBrush(color if on else pal.bg)
                p.drawRoundedRect(r, 3, 3)
                p.setFont(self.f_small)
                p.setPen(QColor(24, 26, 30) if on else pal.dim)
                p.drawText(r, Qt.AlignCenter, label)
            p.setBrush(Qt.NoBrush)
        p.setPen(pal.grid)
        p.drawLine(QPointF(0, top + lane.h + 0.5), QPointF(HEADER_W, top + lane.h + 0.5))

    def _corner(self, p: QPainter, rect: QRect) -> None:
        r = QRect(0, 0, HEADER_W, RULER_H)
        if not r.intersects(rect):
            return
        tl = self.tl
        p.save()
        p.setClipRect(r)
        p.fillRect(r, self.pal.header)
        seq = tl._seq()
        p.setFont(self.f_head)
        p.setPen(self.pal.text)
        p.drawText(QRectF(10, 2, HEADER_W - 14, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   fmt_timecode(tl.state.playhead, tl.fps))
        if seq is not None:
            p.setFont(self.f_small)
            p.setPen(self.pal.dim)
            p.drawText(QRectF(10, 18, HEADER_W - 14, 14), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{seq.width}×{seq.height}  {seq.fps:g} fps")
        p.setPen(self.pal.grid)
        p.drawLine(QPointF(0, RULER_H - 0.5), QPointF(HEADER_W, RULER_H - 0.5))
        p.restore()

    def _ruler(self, p: QPainter, rect: QRect) -> None:
        tl, pal = self.tl, self.pal
        r = QRect(HEADER_W, 0, tl._view_w() + VBAR_W, RULER_H) & rect
        if r.isEmpty():
            return
        p.save()
        p.setClipRect(r)
        p.fillRect(r, pal.ruler)
        rng = tl._range()
        if rng:
            x0, x1 = tl.time_to_x(rng[0]), tl.time_to_x(rng[1])
            p.fillRect(QRectF(x0, MARKER_ROW_H, x1 - x0, RULER_H - MARKER_ROW_H), pal.range_fill)
            p.setPen(QPen(pal.accent, 2))
            for x in (x0, x1):
                p.drawLine(QPointF(x, MARKER_ROW_H), QPointF(x, RULER_H))
        self._ticks(p, r)
        self._markers(p, r)
        p.setPen(pal.grid)
        p.drawLine(QPointF(r.left(), RULER_H - 0.5), QPointF(r.right() + 1, RULER_H - 0.5))
        x = tl.time_to_x(tl.state.playhead)
        if r.left() - 8 <= x <= r.right() + 8:
            p.setPen(QPen(QColor(0, 0, 0, 120), 1))
            p.setBrush(pal.playhead)
            p.drawPolygon(QPolygonF([QPointF(x - 6.5, RULER_H - 15), QPointF(x + 6.5, RULER_H - 15),
                                     QPointF(x + 6.5, RULER_H - 7), QPointF(x, RULER_H),
                                     QPointF(x - 6.5, RULER_H - 7)]))
        p.restore()

    def _ticks(self, p: QPainter, r: QRect) -> None:
        tl, pal = self.tl, self.pal
        step, sub = ruler_step(tl.pps, tl.fps)
        minor = step / sub if sub else step
        t_lo, t_hi = tl.x_to_time(r.left()), tl.x_to_time(r.right())
        p.setFont(self.f_small)
        major_pen, minor_pen = QPen(pal.dim, 1), QPen(pal.grid.lighter(150), 1)
        k = int(max(0.0, t_lo) // minor)
        while k * minor <= t_hi + minor:
            t = k * minor
            x = self._crisp(tl.time_to_x(t))
            if sub == 0 or k % sub == 0:
                p.setPen(major_pen)
                p.drawLine(QPointF(x, RULER_H - 12), QPointF(x, RULER_H))
                p.drawText(QPointF(x + 4, RULER_H - 14), fmt_ruler(t, step, tl.fps))
            elif minor * tl.pps >= 5:
                p.setPen(minor_pen)
                p.drawLine(QPointF(x, RULER_H - 6), QPointF(x, RULER_H))
            k += 1

    def _markers(self, p: QPainter, r: QRect) -> None:
        tl = self.tl
        seq = tl._seq()
        if seq is None:
            return
        p.setFont(self.f_small)
        for m in seq.markers:
            x = tl.time_to_x(m.time)
            if not (r.left() - 8 <= x <= r.right() + 8):
                continue
            col = QColor(m.color)
            col = col if col.isValid() else self.pal.marker
            p.setPen(QPen(QColor(0, 0, 0, 140), 1))
            p.setBrush(col)
            p.drawPolygon(QPolygonF([QPointF(x - 5, 1), QPointF(x + 5, 1), QPointF(x + 5, 8), QPointF(x, 12.5),
                                     QPointF(x - 5, 8)]))
            if m.name:
                p.setPen(col.lighter(130))
                p.drawText(QPointF(x + 8, 10), self.elide(self.fm_small, m.name, 110))
        p.setBrush(Qt.NoBrush)


# ============================================================================ gestures
class _Gesture:
    """A mouse gesture in progress. ``move`` / ``release`` receive widget coordinates and the current modifiers.

    Gestures only paint (via ``tl._set_overlay``); the project changes in ``release`` through ONE ``state.op`` /
    ``state.do``.  ``cancel`` (Esc, an external edit) throws the ghost away.
    """

    cursor: Optional[Qt.CursorShape] = None

    def __init__(self, tl: "TimelineWidget", pos: QPointF) -> None:
        self.tl = tl
        self.start = QPointF(pos)
        self.active = False                       # True once the pointer moved DRAG_PX

    def crossed(self, pos: QPointF) -> bool:
        if not self.active and (pos - self.start).manhattanLength() >= DRAG_PX:
            self.active = True
        return self.active

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        pass

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        pass

    def cancel(self) -> None:
        self.tl._set_overlay(Overlay())


class ScrubGesture(_Gesture):
    """Press / drag on the ruler: move the playhead (frame-snapped)."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF) -> None:
        super().__init__(tl, pos)
        self.active = True
        self._apply(pos)

    def _apply(self, pos: QPointF) -> None:
        self.tl.state.set_playhead(max(0.0, self.tl.x_to_time(pos.x())), snap=True)

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        self._apply(pos)

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        self._apply(pos)


class PanGesture(_Gesture):
    """Middle-button drag: hand-pan the view."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF) -> None:
        super().__init__(tl, pos)
        self.active = True
        self.sx, self.sy = tl.horizontal_scroll(), tl.vertical_scroll()
        tl.setCursor(Qt.ClosedHandCursor)

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        d = pos - self.start
        self.tl.set_horizontal_scroll(self.sx - d.x())
        self.tl.set_vertical_scroll(self.sy - d.y())

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        self.tl.unsetCursor()

    def cancel(self) -> None:
        self.tl.unsetCursor()


class RubberBandGesture(_Gesture):
    """Press on empty space: a click clears the selection, a drag selects every clip / cue the band touches."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        super().__init__(tl, pos)
        self.additive = bool(mods & (Qt.ControlModifier | Qt.ShiftModifier))
        self.base_clips, self.base_cues = set(tl.state.selection), set(tl.state.cue_selection)

    def _band(self, pos: QPointF) -> QRectF:
        return QRectF(self.start, pos).normalized().intersected(QRectF(self.tl._tracks_view_rect()))

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        if not self.crossed(pos):
            return
        band = self._band(pos)
        clips, cues = self.tl.items_in_rect(band)
        ov = Overlay(band=band, preview_sel=clips | self.base_clips if self.additive else clips,
                     preview_cues=cues | self.base_cues if self.additive else cues)
        self.tl._set_overlay(ov)

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        st = self.tl.state
        if self.active:
            clips, cues = self.tl.items_in_rect(self._band(pos))
            mode = "add" if self.additive else "replace"
            st.select(clips, mode)
            st.select_cues(cues, mode)
        elif not self.additive:
            st.select([])
            st.select_cues([])
        self.tl._set_overlay(Overlay())


class _EditGesture(_Gesture):
    """Base of the gestures that edit one clip: remembers the grabbed clip and the snap points of the gesture."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, clip_id: str) -> None:
        super().__init__(tl, pos)
        self.clip_id = clip_id
        self.lane, self.clip0 = tl._clip_index[clip_id]
        self.fps = tl.fps

    def q(self, t: float) -> float:
        """Quantise to the frame grid."""
        return round(t * self.fps) / self.fps

    def dt(self, pos: QPointF) -> float:
        return (pos.x() - self.start.x()) / self.tl.pps

    def lane_rect(self, lane: Lane) -> QRectF:
        return QRectF(HEADER_W, self.tl._lane_top(lane), self.tl._view_w(), lane.h)

    def ghost_rect(self, lane: Lane, start: float, dur: float) -> QRectF:
        tl = self.tl
        return QRectF(tl.time_to_x(start), tl._lane_top(lane) + CLIP_INSET, max(2.0, dur * tl.pps),
                      lane.h - 2 * CLIP_INSET)


class MoveGesture(_EditGesture):
    """Drag a clip (or the whole selection): ghost + target lane + snapping; release = move_clip / move_clips."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, clip_id: str, deferred: str) -> None:
        super().__init__(tl, pos, clip_id)
        self.deferred = deferred                       # "" | "collapse" | "toggle" (selection change on a plain click)
        self.ids = tl.selected_clip_ids()
        if clip_id not in self.ids:
            self.ids = [clip_id]
        self.orig = {i: (tl._clip_index[i][0], tl._clip_index[i][1].start, tl._clip_index[i][1].duration)
                     for i in self.ids}
        self.points = tl._snap_points(exclude_clips=self.ids)
        self.target = self.lane
        self.plan: Optional[tuple[float, Lane, Optional[float]]] = None
        self.warned = False

    def _plan(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> tuple[float, Lane, Optional[float]]:
        """``(time delta, target lane, snapped guide time)`` for the pointer at ``pos``."""
        tl = self.tl
        starts = [v[1] for v in self.orig.values()]
        lo = min(starts)
        delta = max(self.dt(pos), -lo)
        edges = [s + delta for s in starts] + [v[1] + v[2] + delta for v in self.orig.values()]
        shift, guide = tl._snap_edges(edges, self.points, mods)
        delta = max(delta + shift, -lo)
        delta = delta if guide is not None else self.q(delta)
        lane = tl.lane_at_y(pos.y())
        if (len(self.ids) == 1 and lane is not None and lane.track is not None and lane.kind == self.lane.kind
                and not lane.track.locked):
            self.target = lane
        return delta, self.target, guide

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        tl = self.tl
        if not self.crossed(pos):
            return
        if any(tl._clip_index[i][0].track.locked for i in self.ids):
            if not self.warned:
                self.warned = True
                tl.state.status.emit("軌道已鎖定，無法移動片段")
            return
        delta, lane, guide = self.plan = self._plan(pos, mods)
        ghosts = []
        for cid, (src_lane, start, dur) in self.orig.items():
            ghosts.append((self.ghost_rect(lane if len(self.ids) == 1 else src_lane, start + delta, dur), "move"))
        first = min(v[1] for v in self.orig.values()) + delta
        tip = f"{fmt_timecode(first, self.fps)}　{delta:+.2f}s"
        if lane is not self.lane:
            tip += f"　→ {lane.name}"
        if len(self.ids) == 1 and mods & Qt.ControlModifier:
            tip += "　插入"
        tl._set_overlay(Overlay(ghosts=ghosts, lane=self.lane_rect(lane), snap_t=guide, tip=(pos, tip),
                                dim=set(self.ids)))

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        tl = self.tl
        tl._set_overlay(Overlay())
        if not self.active:
            if self.deferred == "collapse":
                tl.state.select([self.clip_id])
            elif self.deferred == "toggle":
                tl.state.select([self.clip_id], "toggle")
            return
        if any(tl._clip_index[i][0].track.locked for i in self.ids):
            return
        delta, lane, _ = self._plan(pos, mods)
        if len(self.ids) == 1:
            cid = self.ids[0]
            if abs(delta) < EPS and lane.id == self.lane.id:
                return
            resolve = "insert" if mods & Qt.ControlModifier else "overwrite"
            tl._op("移動片段", "move_clip", cid, self.orig[cid][1] + delta, lane.id, resolve)
        elif abs(delta) >= EPS:
            tl._op(f"移動 {len(self.ids)} 個片段", "move_clips", self.ids, delta, "overwrite")


class SlipGesture(_EditGesture):
    """Alt + drag on a clip body: change the source in/out without moving the clip (slip_clip)."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, clip_id: str) -> None:
        super().__init__(tl, pos, clip_id)
        c = self.clip0
        a = tl.state.project.assets.get(c.asset_id)
        self.span = c.src_out - c.src_in
        self.limit = a.duration if (a is not None and a.kind != "image" and a.duration > 0) else None

    def _new_in(self, pos: QPointF) -> float:
        c = self.clip0
        new = c.src_in - self.q(self.dt(pos)) * c.speed       # dragging right shows earlier material
        hi = (self.limit - self.span) if self.limit is not None else new
        return clamp(new, 0.0, max(0.0, hi))

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        if not self.crossed(pos) or self.limit is None:
            return
        new_in = self._new_in(pos)
        tip = f"滑動素材　入點 {fmt_clock(new_in)}　出點 {fmt_clock(new_in + self.span)}"
        self.tl._set_overlay(Overlay(slip=(self.clip_id, new_in), tip=(pos, tip),
                                     ghosts=[(self.ghost_rect(self.lane, self.clip0.start, self.clip0.duration),
                                              "slip")]))

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        self.tl._set_overlay(Overlay())
        if self.active and self.limit is not None:
            delta = self._new_in(pos) - self.clip0.src_in
            if abs(delta) > EPS:
                self.tl._op("滑動素材", "slip_clip", self.clip_id, delta)


def trim_limits(proj, lane: Lane, c: Clip, edge: str, ripple: bool) -> tuple[float, float]:
    """The range the dragged edge may take: the very function ``ops.trim_clip_start / _end`` clamp to, so the ghost
    shows exactly what the edit will do (the op remains the authority)."""
    return ops.trim_limits(proj, lane.track, c, edge, ripple)


class TrimGesture(_EditGesture):
    """Drag a clip edge: trim (Shift = ripple); release = trim_clip_start / trim_clip_end."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, clip_id: str, edge: str) -> None:
        super().__init__(tl, pos, clip_id)
        self.edge = edge
        self.points = tl._snap_points(exclude_clips=[clip_id])
        self.edge_t0 = self.clip0.start if edge == "l" else self.clip0.end

    def _target(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> tuple[float, bool, Optional[float]]:
        tl = self.tl
        ripple = bool(mods & Qt.ShiftModifier)
        raw = self.edge_t0 + self.dt(pos)
        shift, guide = tl._snap_edges([raw], self.points, mods)
        lo, hi = trim_limits(tl.state.project, self.lane, self.clip0, self.edge, ripple)
        t = clamp(raw + shift, lo, hi)
        return (t if guide is not None and abs(t - (raw + shift)) < EPS else self.q(t)), ripple, guide

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        if not self.crossed(pos):
            return
        c = self.clip0
        t, ripple, guide = self._target(pos, mods)
        if self.edge == "l":
            d = t - c.start
            new_in, new_out, dur = c.src_in + d * c.speed, c.src_out, c.duration - d
            rect = self.ghost_rect(self.lane, c.start if ripple else t, dur)
        else:
            new_in, new_out, dur = c.src_in, c.src_in + (t - c.start) * c.speed, t - c.start
            rect = self.ghost_rect(self.lane, c.start, dur)
        tip = f"入點 {fmt_clock(new_in)}　出點 {fmt_clock(new_out)}　長度 {dur:.2f}s" + ("　波紋" if ripple else "")
        self.tl._set_overlay(Overlay(ghosts=[(rect, "trim")], snap_t=guide, tip=(pos, tip)))

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        tl = self.tl
        tl._set_overlay(Overlay())
        if not self.active:
            return
        t, ripple, _ = self._target(pos, mods)
        if abs(t - self.edge_t0) < EPS:
            return
        if self.edge == "l":
            tl._op("修剪片段開頭", "trim_clip_start", self.clip_id, t, ripple=ripple)
        else:
            tl._op("修剪片段結尾", "trim_clip_end", self.clip_id, t, ripple=ripple)


class FadeGesture(_EditGesture):
    """Drag a top-corner handle: fade in / fade out length (set_clip_props)."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, clip_id: str, which: str) -> None:
        super().__init__(tl, pos, clip_id)
        self.which = which
        c = self.clip0
        self.v0 = c.fade_in if which == "in" else c.fade_out
        self.other = c.fade_out if which == "in" else c.fade_in
        self.dur = c.duration

    def _value(self, pos: QPointF) -> float:
        d = self.dt(pos)
        v = self.v0 + d if self.which == "in" else self.v0 - d
        return self.q(clamp(v, 0.0, max(0.0, self.dur - self.other)))

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        if not self.crossed(pos):
            return
        v = self._value(pos)
        label = "淡入" if self.which == "in" else "淡出"
        self.tl._set_overlay(Overlay(fade=(self.clip_id, self.which, v), tip=(pos, f"{label} {v:.2f} 秒")))

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        self.tl._set_overlay(Overlay())
        if not self.active:
            return
        v = self._value(pos)
        if abs(v - self.v0) > EPS:
            key = "fade_in" if self.which == "in" else "fade_out"
            self.tl._op("調整淡入" if self.which == "in" else "調整淡出", "set_clip_props", self.clip_id, **{key: v})


class CrossfadeGesture(_EditGesture):
    """Drag the left edge of a clip that overlaps its predecessor: the overlap length (set_crossfade)."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, clip_id: str) -> None:
        super().__init__(tl, pos, clip_id)
        a = tl.previous_overlap(self.lane, self.clip0)
        assert a is not None
        self.a_end = a.end
        self.d0 = a.end - self.clip0.start
        self.limit = 0.5 * min(a.duration, self.clip0.duration)

    def _dur(self, pos: QPointF) -> float:
        return min(self.q(clamp(self.d0 - self.dt(pos), 0.0, self.limit)), self.limit)

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        if not self.crossed(pos):
            return
        d = self._dur(pos)
        rect = self.ghost_rect(self.lane, self.a_end - d, self.clip0.duration)
        self.tl._set_overlay(Overlay(ghosts=[(rect, "trim")], tip=(pos, f"交叉淡化 {d:.2f} 秒")))

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        self.tl._set_overlay(Overlay())
        if self.active:
            d = self._dur(pos)
            if abs(d - self.d0) > EPS:
                self.tl._op("調整交叉淡化", "set_crossfade", self.clip_id, d)


class CueGesture(_Gesture):
    """Base of the caption-block gestures (move / resize)."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, cue_id: str) -> None:
        super().__init__(tl, pos)
        self.cue_id = cue_id
        self.fps = tl.fps

    def q(self, t: float) -> float:
        return round(t * self.fps) / self.fps

    def dt(self, pos: QPointF) -> float:
        return (pos.x() - self.start.x()) / self.tl.pps

    def ghost(self, start: float, end: float, cue_id: Optional[str] = None) -> QRectF:
        tl = self.tl
        lane = tl._lane_by_id[CAPTION_LANE]
        y0, h = tl.cue_band(lane, tl._cue_index[cue_id or self.cue_id])
        return QRectF(tl.time_to_x(start), y0, max(2.0, (end - start) * tl.pps), h)


class CueMoveGesture(CueGesture):
    """Drag caption blocks (the whole cue selection): release = update_cue / shift_cues."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, cue_id: str, deferred: str) -> None:
        super().__init__(tl, pos, cue_id)
        self.deferred = deferred
        self.ids = tl.selected_cue_ids()
        if cue_id not in self.ids:
            self.ids = [cue_id]
        self.orig = {i: (tl._cue_index[i].start, tl._cue_index[i].end) for i in self.ids}
        self.points = tl._snap_points(exclude_cues=self.ids, with_cues=True)

    def _delta(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> tuple[float, Optional[float]]:
        lo = min(s for s, _ in self.orig.values())
        delta = max(self.dt(pos), -lo)
        edges = [s + delta for s, _ in self.orig.values()] + [e + delta for _, e in self.orig.values()]
        shift, guide = self.tl._snap_edges(edges, self.points, mods)
        delta = max(delta + shift, -lo)
        return (delta if guide is not None else self.q(delta)), guide

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        if not self.crossed(pos):
            return
        delta, guide = self._delta(pos, mods)
        ghosts = [(self.ghost(s + delta, e + delta, i), "cue") for i, (s, e) in self.orig.items()]
        first = min(s for s, _ in self.orig.values()) + delta
        self.tl._set_overlay(Overlay(ghosts=ghosts, snap_t=guide, dim=set(self.ids),
                                     tip=(pos, f"{fmt_timecode(first, self.fps)}　{delta:+.2f}s")))

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        tl = self.tl
        tl._set_overlay(Overlay())
        if not self.active:
            if self.deferred == "collapse":
                tl.state.select_cues([self.cue_id])
            elif self.deferred == "toggle":
                tl.state.select_cues([self.cue_id], "toggle")
            return
        delta, _ = self._delta(pos, mods)
        if abs(delta) < EPS:
            return
        if len(self.ids) == 1:
            s, e = self.orig[self.ids[0]]
            tl._op("移動字幕", "update_cue", self.cue_id, start=s + delta, end=e + delta)
        else:
            tl._op(f"移動 {len(self.ids)} 則字幕", "shift_cues", delta, self.ids)


class CueResizeGesture(CueGesture):
    """Drag a caption block edge: release = update_cue(start=...) / update_cue(end=...)."""

    def __init__(self, tl: "TimelineWidget", pos: QPointF, cue_id: str, edge: str) -> None:
        super().__init__(tl, pos, cue_id)
        q = tl._cue_index[cue_id]
        self.edge, self.s0, self.e0 = edge, q.start, q.end
        self.points = tl._snap_points(exclude_cues=[cue_id], with_cues=True)

    def _target(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> tuple[float, Optional[float]]:
        raw = (self.s0 if self.edge == "l" else self.e0) + self.dt(pos)
        shift, guide = self.tl._snap_edges([raw], self.points, mods)
        t = raw + shift
        t = clamp(t, 0.0, self.e0 - MIN_CUE) if self.edge == "l" else max(t, self.s0 + MIN_CUE)
        return (t if guide is not None else self.q(t)), guide

    def move(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        if not self.crossed(pos):
            return
        t, guide = self._target(pos, mods)
        s, e = (t, self.e0) if self.edge == "l" else (self.s0, t)
        self.tl._set_overlay(Overlay(ghosts=[(self.ghost(s, e), "cue")], snap_t=guide,
                                     tip=(pos, f"{fmt_timecode(t, self.fps)}　長度 {e - s:.2f}s")))

    def release(self, pos: QPointF, mods: Qt.KeyboardModifiers) -> None:
        self.tl._set_overlay(Overlay())
        if not self.active:
            return
        t, _ = self._target(pos, mods)
        if abs(t - (self.s0 if self.edge == "l" else self.e0)) < EPS:
            return
        key = "start" if self.edge == "l" else "end"
        self.tl._op("調整字幕", "update_cue", self.cue_id, **{key: t})


# ============================================================================ drag & drop plumbing
DROP_GHOST_S = 3.0                # ghost length for a dragged OS file (its real length is only known after probing)


@dataclass
class DropItem:
    """One dragged thing: ``asset_id`` is ``""`` for an OS file that is not imported yet."""
    asset_id: str
    dur: float
    kind: str                     # lane kind it belongs on: video | audio


@dataclass
class DropSession:
    internal: bool                # True: media-bin payload (asset ids); False: OS files
    items: list = field(default_factory=list)
    paths: list = field(default_factory=list)


@dataclass
class DropPlan:
    """Where a drag would land: ``entries`` = [(DropItem, Track, start)] in drop order."""
    start: float
    entries: list = field(default_factory=list)
    lane: Optional[Lane] = None             # lane of the first entry (highlighted)
    guide: Optional[float] = None           # snap guide time


def asset_lane_kind(a: Asset) -> str:
    """Pictures and anything with video go on a video track; audio-only files on an audio track."""
    return VIDEO if (a.has_video or a.kind == "image") else AUDIO


def asset_span(a: Asset) -> float:
    return IMAGE_DEFAULT if (a.kind == "image" or a.duration <= 0) else a.duration


def decode_assets(mime) -> list[str]:
    """Asset ids of an internal drag (UTF-8 JSON ``{"assets": [...]}``); empty when missing or malformed."""
    if not mime.hasFormat(ASSET_MIME):
        return []
    try:
        data = json.loads(bytes(mime.data(ASSET_MIME)).decode("utf-8"))
        return [str(i) for i in data["assets"]]
    except (ValueError, KeyError, TypeError):
        return []


def dropped_paths(mime) -> list[str]:
    if not mime.hasUrls():
        return []
    return [u.toLocalFile() for u in mime.urls() if u.isLocalFile() and u.toLocalFile()]


class CueEdit(QLineEdit):
    """The small in-place editor laid over a caption block: Enter / focus loss commits, Esc cancels."""
    committed = Signal(str)
    cancelled = Signal()

    def __init__(self, parent: QWidget, text: str) -> None:
        super().__init__(text, parent)
        self.setObjectName("cueEdit")
        self.setFrame(False)
        self._done = False
        self.setStyleSheet(f"QLineEdit#cueEdit {{ background: {theme.PANEL_ALT}; color: {theme.TEXT}; "
                           f"border: 2px solid {theme.ACCENT}; border-radius: 3px; padding: 0 4px; font-size: 12px; }}")
        self.returnPressed.connect(self.finish)

    def finish(self) -> None:
        if not self._done:
            self._done = True
            self.committed.emit(self.text())

    def abandon(self) -> None:
        if not self._done:
            self._done = True
            self.cancelled.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.abandon()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.finish()


# ============================================================================ the widget
class TimelineWidget(QWidget):
    """The multi-track timeline (see the module docstring)."""

    externalFilesDropped = Signal(list, float, str)
    revealAssetRequested = Signal(str)
    propertiesRequested = Signal(str)
    snappingChanged = Signal(bool)
    zoomChanged = Signal(float)
    _clipboard: list = []                                # copied clips, shared by every timeline (Ctrl+C / Ctrl+V)

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setObjectName("timeline")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setMinimumSize(380, 160)
        self._pps = PPS_DEFAULT
        self._sx = 0.0                                   # horizontal scroll in pixels (time * pps at the left edge)
        self._sy = 0.0                                   # vertical scroll in pixels
        self._snapping = True
        self._playing = False
        self._fit_pending = True
        self._lanes: list[Lane] = []
        self._lane_by_id: dict[str, Lane] = {}
        self._clip_index: dict[str, tuple[Lane, Clip]] = {}
        self._cue_index: dict[str, CaptionCue] = {}
        self._duration = 0.0
        self._content_h = 0
        self._hover_clip = ""
        self._anchor = ""                                # last clicked clip / cue (Shift-range anchor)
        self._gesture: Optional[_Gesture] = None
        self._ov = Overlay()
        self._ov_bounds = QRectF()
        self._drop: Optional[DropSession] = None         # an external / internal drag hovering over the widget
        self._cue_edit: Optional[CueEdit] = None
        self._cue_edit_id = ""
        self._ph_x = 0.0                                 # x where the playhead was last painted
        self._missing: dict[str, tuple[float, bool, str]] = {}
        self._visuals = VisualCache(self)
        self._visuals.ready.connect(self._on_visual_ready)
        self._painter = _Painter(self)
        self._build_bar()
        self._connect_state()
        self._relayout()

    # ================================================================== construction
    def _tool_button(self, text: str, tip: str, checkable: bool = False) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setFocusPolicy(Qt.NoFocus)
        b.setStyleSheet("QToolButton { padding: 1px 7px; font-size: 11px; }")
        return b

    def _build_bar(self) -> None:
        self._bar = QWidget(self)
        lay = QHBoxLayout(self._bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        left = QWidget()
        left.setFixedWidth(HEADER_W)
        ll = QHBoxLayout(left)
        ll.setContentsMargins(6, 2, 4, 2)
        ll.setSpacing(4)
        self._snap_btn = self._tool_button("吸附", "吸附到片段邊緣、播放頭與標記（按住 Alt 暫時關閉）", True)
        self._snap_btn.setChecked(True)
        self._snap_btn.toggled.connect(self.set_snapping)
        self._fit_btn = self._tool_button("符合視窗", "縮放到看見整個序列（\\）")
        self._fit_btn.clicked.connect(self.zoom_fit)
        ll.addWidget(self._snap_btn)
        ll.addWidget(self._fit_btn)
        self._hbar = QScrollBar(Qt.Horizontal)
        self._hbar.setFocusPolicy(Qt.NoFocus)
        self._hbar.valueChanged.connect(self._on_hbar)
        right = QWidget()
        right.setFixedWidth(190)
        rl = QHBoxLayout(right)
        rl.setContentsMargins(4, 2, 6, 2)
        rl.setSpacing(4)
        out_btn, in_btn = self._tool_button("−", "縮小（-）"), self._tool_button("＋", "放大（+）")
        out_btn.clicked.connect(self.zoom_out)
        in_btn.clicked.connect(self.zoom_in)
        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(0, 1000)
        self._zoom_slider.setFocusPolicy(Qt.NoFocus)
        self._zoom_slider.setToolTip("縮放")
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        rl.addWidget(out_btn)
        rl.addWidget(self._zoom_slider, 1)
        rl.addWidget(in_btn)
        lay.addWidget(left)
        lay.addWidget(self._hbar, 1)
        lay.addWidget(right)
        self._vbar = QScrollBar(Qt.Vertical, self)
        self._vbar.setFocusPolicy(Qt.NoFocus)
        self._vbar.valueChanged.connect(self._on_vbar)

    def _connect_state(self) -> None:
        st = self.state
        st.edited.connect(self._on_edited)
        st.selection_changed.connect(self._on_selection)
        st.cue_selection_changed.connect(self._on_selection)
        st.playhead_changed.connect(self._on_playhead)
        st.sequence_changed.connect(self._on_sequence_changed)
        st.project_replaced.connect(self._on_project_replaced)

    def sizeHint(self) -> QSize:
        return QSize(960, 320)

    def minimumSizeHint(self) -> QSize:
        return QSize(380, 160)

    # ================================================================== sequence access / layout
    def _seq(self) -> Optional[Sequence]:
        try:
            return self.state.sequence()
        except KeyError:
            return None

    def _range(self) -> Optional[tuple[float, float]]:
        try:
            return self.state.get_range()
        except KeyError:
            return None

    @property
    def fps(self) -> float:
        s = self._seq()
        return s.fps if s is not None else 30.0

    def _relayout(self) -> None:
        """Rebuild lanes and indexes from the active sequence (after any edit / sequence switch)."""
        seq = self._seq()
        lanes: list[Lane] = []
        y = 0
        if seq is not None:
            rows = [("caption", None, H_CAPTION_CARDS if any(q.is_card for q in seq.cues) else H_CAPTION)]
            rows += [("video", t, H_VIDEO) for t in reversed(seq.video_tracks)]
            rows += [("audio", t, H_AUDIO) for t in seq.audio_tracks]
            for i, (kind, tr, h) in enumerate(rows):
                lanes.append(Lane(kind, tr, y, h, i))
                y += h + LANE_GAP
        self._lanes = lanes
        self._content_h = y
        self._lane_by_id = {ln.id: ln for ln in lanes}
        self._clip_index = {c.id: (ln, c) for ln in lanes if ln.track is not None for c in ln.track.clips}
        self._cue_index = {q.id: q for q in seq.cues} if seq is not None else {}
        self._has_cards = any(q.is_card for q in self._cue_index.values())
        end = seq.duration if seq is not None else 0.0
        if seq is not None and seq.cues:
            end = max(end, max(q.end for q in seq.cues))
        self._duration = min(end, MAX_TIMELINE)             # (beyond a day only a damaged file has anything; keep the scroll range sane)
        self._update_scrollbars()

    def _lane_top(self, lane: Lane) -> float:
        return RULER_H + lane.y - self._sy

    def _lanes_in(self, area: QRect) -> Iterator[Lane]:
        for lane in self._lanes:
            top = self._lane_top(lane)
            if top + lane.h < area.top():
                continue
            if top > area.bottom():
                break
            yield lane

    def lane_at_y(self, y: float) -> Optional[Lane]:
        """The lane under widget-y ``y`` (the 2 px gap below a lane counts as that lane)."""
        if not (RULER_H <= y < RULER_H + self._view_h()):
            return None
        for lane in self._lanes:
            top = self._lane_top(lane)
            if top <= y < top + lane.h + LANE_GAP:
                return lane
        return None

    def lanes_of_kind(self, kind: str) -> list[Lane]:
        return [ln for ln in self._lanes if ln.kind == kind]

    # ================================================================== public geometry (also used by tests)
    @property
    def pps(self) -> float:
        return self._pps

    def time_to_x(self, t: float) -> float:
        return HEADER_W + t * self._pps - self._sx

    def x_to_time(self, x: float) -> float:
        return (x - HEADER_W + self._sx) / self._pps

    def _view_w(self) -> int:
        return max(0, self.width() - HEADER_W - VBAR_W)

    def _view_h(self) -> int:
        return max(0, self.height() - RULER_H - BAR_H)

    def _tracks_view_rect(self) -> QRect:
        return QRect(HEADER_W, RULER_H, self._view_w(), self._view_h())

    def visible_time_range(self) -> tuple[float, float]:
        return self.x_to_time(HEADER_W), self.x_to_time(HEADER_W + self._view_w())

    def track_rect(self, track_id: str) -> QRect:
        """Lane rectangle (tracks area, widget coordinates); ``CAPTION_LANE`` is the caption lane."""
        lane = self._lane_by_id[track_id]
        return QRect(HEADER_W, round(self._lane_top(lane)), self._view_w(), lane.h)

    def header_rect(self, track_id: str) -> QRect:
        lane = self._lane_by_id[track_id]
        return QRect(0, round(self._lane_top(lane)), HEADER_W, lane.h)

    def header_button_rect(self, lane: Lane, which: str) -> QRect:
        h = 18
        y = round(self._lane_top(lane)) + (lane.h - h) // 2
        return QRect(HEADER_W - (46 if which == "mute" else 24), y, 20, h)

    def clip_rect(self, clip_id: str) -> QRect:
        lane, c = self._clip_index[clip_id]
        x0, x1 = clamp_px(self.time_to_x(c.start)), clamp_px(self.time_to_x(c.end))
        return QRect(x0, round(self._lane_top(lane)) + CLIP_INSET, max(1, x1 - x0), lane.h - 2 * CLIP_INSET)

    def cue_band(self, lane: Lane, q: CaptionCue) -> tuple[float, float]:
        """``(top, height)`` of a caption block in the caption lane.  When the sequence has title cards (字卡), cards
        take the upper part of the lane and subtitles the lower part, so both stay visible and clickable where they
        overlap in time."""
        top, h = round(self._lane_top(lane)) + CLIP_INSET, lane.h - 2 * CLIP_INSET
        if not getattr(self, "_has_cards", False):
            return top, h
        hc = int(h * 0.42)
        return (top, hc - 1) if q.is_card else (top + hc, h - hc)

    def cue_rect(self, cue_id: str) -> QRect:
        q = self._cue_index[cue_id]
        lane = self._lane_by_id[CAPTION_LANE]
        x0, x1 = clamp_px(self.time_to_x(q.start)), clamp_px(self.time_to_x(q.end))
        y0, h = self.cue_band(lane, q)
        return QRect(x0, int(y0), max(1, x1 - x0), max(1, int(h)))

    def fade_handle_rect(self, clip_id: str, which: str) -> QRect:
        """Hit rectangle of a fade handle (``which``: ``"in"`` / ``"out"``) in widget coordinates."""
        lane, c = self._clip_index[clip_id]
        pt = self._fade_handle_pos(lane, c, which, c.fade_in, c.fade_out)
        return QRect(clamp_px(pt.x()) - HANDLE_R, round(pt.y()) - HANDLE_R, 2 * HANDLE_R, 2 * HANDLE_R)

    def _fade_handle_pos(self, lane: Lane, c: Clip, which: str, fade_in: float, fade_out: float) -> QPointF:
        y = self._lane_top(lane) + CLIP_INSET + 6
        if which == "in":
            return QPointF(self.time_to_x(c.start) + max(fade_in * self._pps, 6.0), y)
        return QPointF(self.time_to_x(c.end) - max(fade_out * self._pps, 6.0), y)

    # ================================================================== scrolling / zoom
    def horizontal_scroll(self) -> int:
        return int(round(self._sx))

    def vertical_scroll(self) -> int:
        return int(round(self._sy))

    def set_horizontal_scroll(self, px: float) -> None:
        self._sx = float(px)
        self._after_scroll()

    def set_vertical_scroll(self, px: float) -> None:
        self._sy = float(px)
        self._after_scroll()

    def _extent_px(self) -> float:
        return self._duration * self._pps + self._view_w() * 0.3

    def _max_scroll(self) -> tuple[float, float]:
        return (max(0.0, self._extent_px() - self._view_w()), max(0.0, self._content_h - self._view_h()))

    def _after_scroll(self) -> None:
        self._update_scrollbars()
        self._place_cue_edit()
        self.update()

    def _update_scrollbars(self) -> None:
        mx, my = self._max_scroll()
        self._sx, self._sy = clamp(self._sx, 0.0, mx), clamp(self._sy, 0.0, my)
        for bar, value, mx_, page in ((self._hbar, self._sx, mx, self._view_w()),
                                      (self._vbar, self._sy, my, self._view_h())):
            bar.blockSignals(True)
            bar.setRange(0, int(mx_))
            bar.setPageStep(max(1, page))
            bar.setSingleStep(max(16, page // 12))
            bar.setValue(int(round(value)))
            bar.blockSignals(False)
        self._sync_zoom_slider()

    def _on_hbar(self, v: int) -> None:
        self._sx = float(v)
        self.update()

    def _on_vbar(self, v: int) -> None:
        self._sy = float(v)
        self.update()

    def _sync_zoom_slider(self) -> None:
        v = int(round(1000 * math.log(self._pps / PPS_MIN) / math.log(PPS_MAX / PPS_MIN)))
        if self._zoom_slider.value() != v:
            self._zoom_slider.blockSignals(True)
            self._zoom_slider.setValue(v)
            self._zoom_slider.blockSignals(False)

    def _on_zoom_slider(self, v: int) -> None:
        self.set_pps(PPS_MIN * (PPS_MAX / PPS_MIN) ** (v / 1000.0))

    def set_pps(self, pps: float, anchor_x: Optional[float] = None) -> None:
        """Zoom to ``pps`` pixels per second keeping the time under ``anchor_x`` (default: the view centre) fixed."""
        pps = clamp(float(pps), PPS_MIN, PPS_MAX)
        self._fit_pending = False
        if abs(pps - self._pps) < 1e-9:
            return
        ax = HEADER_W + self._view_w() / 2 if anchor_x is None else clamp(anchor_x, HEADER_W, HEADER_W + self._view_w())
        t = self.x_to_time(ax)
        self._pps = pps
        self._sx = t * pps - (ax - HEADER_W)
        self._update_scrollbars()
        self._place_cue_edit()
        self.update()
        self.zoomChanged.emit(pps)

    def _zoom_anchor(self) -> Optional[float]:
        x = self.time_to_x(self.state.playhead)
        return x if HEADER_W <= x <= HEADER_W + self._view_w() else None

    def zoom_in(self) -> None:
        self.set_pps(self._pps * 1.25, self._zoom_anchor())

    def zoom_out(self) -> None:
        self.set_pps(self._pps / 1.25, self._zoom_anchor())

    def zoom_fit(self) -> None:
        """Fit the whole sequence (or 10 s when it is empty) into the view and scroll to the start."""
        dur = self._duration if self._duration > 0 else 10.0
        self._fit_pending = False
        self._pps = clamp(max(60, self._view_w() - 24) / dur, PPS_MIN, PPS_MAX)
        self._sx = 0.0
        self._update_scrollbars()
        self.update()
        self.zoomChanged.emit(self._pps)

    def _reveal(self, t: float, margin: int = 24) -> None:
        """Scroll the minimum amount that makes time ``t`` visible."""
        x, right = self.time_to_x(t), HEADER_W + self._view_w()
        if x < HEADER_W + 2:
            self.set_horizontal_scroll(self._sx + (x - HEADER_W - margin))
        elif x > right - 2:
            self.set_horizontal_scroll(self._sx + (x - right + margin))

    # ================================================================== hit testing
    def _marker_at(self, x: float):
        seq = self._seq()
        if seq is None:
            return None
        for m in seq.markers:
            if abs(self.time_to_x(m.time) - x) <= 6:
                return m
        return None

    def hit_test(self, pos) -> Hit:
        """What is under widget position ``pos`` (QPoint / QPointF)?  See :class:`Hit`."""
        x, y = float(pos.x()), float(pos.y())
        t = max(0.0, self.x_to_time(x)) if x >= HEADER_W else 0.0
        if y < RULER_H:
            if x < HEADER_W:
                return Hit("corner")
            m = self._marker_at(x) if y < MARKER_ROW_H else None
            return Hit("marker", marker_id=m.id, t=m.time) if m is not None else Hit("ruler", t=t)
        if y >= self.height() - BAR_H or x >= self.width() - VBAR_W:
            return Hit("none")
        lane = self.lane_at_y(y)
        if lane is None:
            return Hit("none", t=t)
        if x < HEADER_W:
            return self._header_hit(lane, x, y)
        return self._cue_hit(lane, x, y, t) if lane.kind == "caption" else self._clip_hit(lane, x, y, t)

    def _header_hit(self, lane: Lane, x: float, y: float) -> Hit:
        if lane.track is not None:
            for which in ("mute", "lock"):
                if self.header_button_rect(lane, which).contains(int(x), int(y)):
                    return Hit("header_" + which, lane)
        return Hit("header", lane)

    def _clip_hit(self, lane: Lane, x: float, y: float, t: float) -> Hit:
        assert lane.track is not None
        for c in reversed(lane.track.clips):                       # later clips are on top
            x0, x1 = self.time_to_x(c.start), self.time_to_x(c.end)
            if not (x0 <= x <= x1):
                continue
            w = x1 - x0
            top = self._lane_top(lane) + CLIP_INSET
            if w >= 40 and y - top <= 12:
                for which in ("in", "out"):
                    hp = self._fade_handle_pos(lane, c, which, c.fade_in, c.fade_out)
                    if abs(x - hp.x()) <= 6 and abs(y - hp.y()) <= 7:
                        return Hit("fade_" + which, lane, clip_id=c.id, t=t)
            zone = min(float(EDGE_PX), w / 3.0)
            if x - x0 <= zone:
                return Hit("xfade" if self.previous_overlap(lane, c) is not None else "clip_l", lane, c.id, t=t)
            if x1 - x <= zone:
                return Hit("clip_r", lane, c.id, t=t)
            return Hit("clip", lane, c.id, t=t)
        return Hit("lane", lane, t=t)

    def _cue_hit(self, lane: Lane, x: float, y: float, t: float) -> Hit:
        seq = self._seq()
        for q in reversed(seq.cues if seq is not None else []):
            x0, x1 = self.time_to_x(q.start), self.time_to_x(q.end)
            if not (x0 <= x <= x1):
                continue
            if self._has_cards:                                  # cards and subtitles sit in their own halves
                y0, bh = self.cue_band(lane, q)
                lo, hi = (y0 - CLIP_INSET, y0 + bh + 1) if q.is_card else (y0, y0 + bh + CLIP_INSET)
                if not lo <= y < hi:
                    continue
            zone = min(float(EDGE_PX), (x1 - x0) / 3.0)
            kind = "cue_l" if x - x0 <= zone else "cue_r" if x1 - x <= zone else "cue"
            return Hit(kind, lane, cue_id=q.id, t=t)
        return Hit("caption_lane", lane, t=t)

    @staticmethod
    def previous_overlap(lane: Lane, c: Clip) -> Optional[Clip]:
        """The clip that ``c`` crossfades from (it starts earlier and still runs when ``c`` starts), if any."""
        if lane.track is None:
            return None
        prev = [o for o in lane.track.clips if o is not c and o.start < c.start - EPS and o.end > c.start + EPS]
        return max(prev, key=lambda o: o.start) if prev else None

    def items_in_rect(self, band: QRectF) -> tuple[set[str], set[str]]:
        """Clip ids and cue ids whose block touches ``band`` (widget coordinates)."""
        clips: set[str] = set()
        cues: set[str] = set()
        for lane in self._lanes_in(band.toAlignedRect()):
            top = self._lane_top(lane)
            if not (top < band.bottom() and top + lane.h > band.top()):
                continue
            if lane.track is None:
                cues |= {q.id for q in self._cue_index.values()
                         if self.time_to_x(q.start) <= band.right() and self.time_to_x(q.end) >= band.left()}
                continue
            clips |= {c.id for c in lane.track.clips
                      if self.time_to_x(c.start) <= band.right() and self.time_to_x(c.end) >= band.left()}
        return clips, cues

    # ================================================================== overlay + paint plumbing
    def _tip_rect(self, pos: QPointF, text: str, fm: QFontMetrics) -> QRectF:
        w, h = fm.horizontalAdvance(text) + 18, fm.height() + 8
        x = clamp(pos.x() + 14, 2, max(2, self.width() - w - 2))
        y = pos.y() + 22
        if y + h > self.height() - BAR_H:
            y = pos.y() - h - 8
        return QRectF(x, max(RULER_H, y), w, h)

    def _overlay_bounds(self, ov: Overlay) -> QRectF:
        rects = [QRectF(r) for r, _ in ov.ghosts]
        if ov.lane is not None:
            rects.append(QRectF(ov.lane))
        if ov.band is not None:
            rects.append(QRectF(ov.band))
        if ov.snap_t is not None:
            rects.append(QRectF(self.time_to_x(ov.snap_t) - 3, RULER_H, 6, self._view_h()))
        if ov.tip is not None:
            rects.append(self._tip_rect(ov.tip[0], ov.tip[1], self._painter.fm_label))
        ids = set(ov.dim) | set(ov.preview_sel) | {x[0] for x in (ov.fade, ov.slip) if x}
        for cid in ids:
            if cid in self._clip_index:
                rects.append(QRectF(self.clip_rect(cid)))
        for qid in set(ov.preview_cues) | (ov.dim & set(self._cue_index)):
            if qid in self._cue_index:
                rects.append(QRectF(self.cue_rect(qid)))
        out = QRectF()
        for r in rects:
            out = out.united(r) if not out.isNull() else QRectF(r)
        return out

    def _set_overlay(self, ov: Overlay) -> None:
        """Swap the gesture overlay and repaint only the union of the old and new overlay areas."""
        old = self._ov_bounds
        self._ov = ov
        self._ov_bounds = self._overlay_bounds(ov)
        r = self._ov_bounds if old.isNull() else old if self._ov_bounds.isNull() else old.united(self._ov_bounds)
        if not r.isNull():
            self.update(r.toAlignedRect().adjusted(-3, -3, 3, 3))

    def _effective(self, c: Clip) -> tuple[float, float, float]:
        """``(fade_in, fade_out, src_in)`` of ``c`` including a live (not yet committed) handle / slip drag."""
        fi, fo, si = c.fade_in, c.fade_out, c.src_in
        ov = self._ov
        if ov.fade and ov.fade[0] == c.id:
            if ov.fade[1] == "in":
                fi = ov.fade[2]
            else:
                fo = ov.fade[2]
        if ov.slip and ov.slip[0] == c.id:
            si = ov.slip[1]
        return fi, fo, si

    def _is_missing(self, a: Asset) -> bool:
        """``asset.missing`` with a 3 s cache (it is a filesystem stat; painting must stay cheap)."""
        now = time.monotonic()
        ent = self._missing.get(a.id)
        if ent is None or now - ent[0] > 3.0 or ent[2] != a.path:
            ent = self._missing[a.id] = (now, a.missing, a.path)
        return ent[1]

    def visual_runners(self) -> list[TaskRunner]:
        """Running thumbnail / waveform jobs (tests wait on these with ``workers.wait_for_idle``)."""
        return self._visuals.runners()

    def visuals_idle(self) -> bool:
        return self._visuals.idle()

    def _on_visual_ready(self, asset_id: str) -> None:
        rect = QRect()
        for lane, c in self._clip_index.values():
            if c.asset_id == asset_id:
                rect = rect.united(self.clip_rect(c.id)) if not rect.isNull() else self.clip_rect(c.id)
        if not rect.isNull():
            self.update(rect.adjusted(-2, -2, 2, 2))

    # ================================================================== state signals
    def _on_edited(self, kind: str, label: str) -> None:
        self._abort_gesture()
        self._close_cue_edit(commit=False)
        self._relayout()
        self.update()

    def _on_selection(self) -> None:
        self.update(self._tracks_view_rect())

    def _on_sequence_changed(self) -> None:
        self._abort_gesture()
        self._close_cue_edit(commit=False)
        self._sx = self._sy = 0.0
        self._relayout()
        self._fit_pending = True
        self._apply_pending_fit()
        self.update()

    def _on_project_replaced(self) -> None:
        self._on_sequence_changed()

    def _apply_pending_fit(self) -> None:
        if self._fit_pending and self.isVisible() and self._view_w() > 120:
            if self._duration > 0:
                self.zoom_fit()
            else:
                self.set_pps(PPS_DEFAULT)
            self._fit_pending = False

    def _on_playhead(self, t: float) -> None:
        x = self.time_to_x(t)
        if self._playing and (x > HEADER_W + self._view_w() - 30 or x < HEADER_W):
            self.set_horizontal_scroll(t * self._pps - 0.12 * self._view_w())      # page-flip while playing
        else:
            self._reveal(t)
        new_x = self.time_to_x(t)
        bottom = self.height() - BAR_H
        for ph_x in (self._ph_x, new_x):
            self.update(QRect(clamp_px(ph_x) - 8, 0, 17, bottom))
        self.update(QRect(0, 0, HEADER_W, RULER_H))

    def set_playing(self, on: bool) -> None:
        """MainWindow tells the timeline whether playback runs (enables page-flip auto-scroll)."""
        self._playing = bool(on)

    def set_snapping(self, on: bool) -> None:
        on = bool(on)
        if on != self._snapping:
            self._snapping = on
            self.snappingChanged.emit(on)
        if self._snap_btn.isChecked() != on:
            self._snap_btn.blockSignals(True)
            self._snap_btn.setChecked(on)
            self._snap_btn.blockSignals(False)

    @property
    def snapping(self) -> bool:
        return self._snapping

    def _abort_gesture(self) -> None:
        g, self._gesture = self._gesture, None
        if g is not None:
            g.cancel()

    # ================================================================== Qt events: paint / layout
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        self._painter.paint(p, event.rect())
        p.end()
        self._ph_x = self.time_to_x(self.state.playhead)

    def resizeEvent(self, event) -> None:
        w, h = self.width(), self.height()
        self._bar.setGeometry(0, h - BAR_H, max(0, w - VBAR_W), BAR_H)
        self._vbar.setGeometry(w - VBAR_W, RULER_H, VBAR_W, max(0, h - RULER_H - BAR_H))
        self._update_scrollbars()
        self._apply_pending_fit()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_pending_fit()

    def leaveEvent(self, event) -> None:
        self._set_hover("")
        super().leaveEvent(event)

    def _set_hover(self, clip_id: str) -> None:
        if clip_id != self._hover_clip:
            old, self._hover_clip = self._hover_clip, clip_id
            for cid in (old, clip_id):
                if cid in self._clip_index:
                    self.update(self.clip_rect(cid).adjusted(-8, -8, 8, 8))

    # ================================================================== Qt events: wheel
    def wheelEvent(self, event: QWheelEvent) -> None:
        mods = event.modifiers()
        ad, pd = event.angleDelta(), event.pixelDelta()
        dx, dy = (ad.x(), ad.y()) if (ad.x() or ad.y()) else (pd.x(), pd.y())
        if mods & Qt.ControlModifier:
            if dy:
                self.set_pps(self._pps * 1.2 ** (dy / 120.0), max(float(HEADER_W), event.position().x()))
        elif (mods & Qt.ShiftModifier) or (dx and not dy):
            d = dy if (mods & Qt.ShiftModifier) and dy else dx
            self.set_horizontal_scroll(self._sx - d / 120.0 * self._view_w() * 0.15)
        elif dy:
            if self._max_scroll()[1] > 0:
                self.set_vertical_scroll(self._sy - dy / 120.0 * 42)
            else:                                   # nothing to scroll vertically: the wheel pans in time instead
                self.set_horizontal_scroll(self._sx - dy / 120.0 * self._view_w() * 0.15)
        event.accept()

    # ================================================================== Qt events: mouse
    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus(Qt.MouseFocusReason)
        pos, mods, btn = event.position(), event.modifiers(), event.button()
        self._abort_gesture()
        if btn == Qt.MiddleButton:
            self._gesture = PanGesture(self, pos)
        elif btn == Qt.LeftButton:
            self._gesture = self._make_gesture(self.hit_test(pos), pos, mods)
        event.accept()

    def _make_gesture(self, hit: Hit, pos: QPointF, mods: Qt.KeyboardModifiers) -> Optional[_Gesture]:
        """Press on ``hit`` -> the gesture it starts (selection side effects of the press happen here)."""
        k = hit.kind
        if k in ("ruler", "marker"):
            return ScrubGesture(self, pos)
        if k in ("header_mute", "header_lock"):
            self._toggle_track_flag(hit.lane, "muted" if k == "header_mute" else "locked")
            return None
        if hit.clip_id:
            return self._clip_gesture(hit, pos, mods)
        if hit.cue_id:
            return self._cue_gesture(hit, pos, mods)
        if k in ("lane", "caption_lane") or (k == "none" and self._tracks_view_rect().contains(pos.toPoint())):
            return RubberBandGesture(self, pos, mods)
        return None

    def _clip_gesture(self, hit: Hit, pos: QPointF, mods: Qt.KeyboardModifiers) -> Optional[_Gesture]:
        deferred = self._click_select_clip(hit.clip_id, mods)
        track = hit.lane.track
        if track.locked and hit.kind != "clip":
            self.state.status.emit(f"軌道「{track.name}」已鎖定，無法編輯")
            return None
        if hit.kind == "clip":
            return SlipGesture(self, pos, hit.clip_id) if mods & Qt.AltModifier else MoveGesture(
                self, pos, hit.clip_id, deferred)
        if track.locked:
            return None
        if hit.kind in ("clip_l", "clip_r"):
            return TrimGesture(self, pos, hit.clip_id, "l" if hit.kind == "clip_l" else "r")
        if hit.kind == "xfade":
            return CrossfadeGesture(self, pos, hit.clip_id)
        return FadeGesture(self, pos, hit.clip_id, "in" if hit.kind == "fade_in" else "out")

    def _cue_gesture(self, hit: Hit, pos: QPointF, mods: Qt.KeyboardModifiers) -> _Gesture:
        deferred = self._click_select_cue(hit.cue_id, mods)
        if hit.kind == "cue":
            return CueMoveGesture(self, pos, hit.cue_id, deferred)
        return CueResizeGesture(self, pos, hit.cue_id, "l" if hit.kind == "cue_l" else "r")

    # ---- selection side effects of a press
    def _click_select_clip(self, clip_id: str, mods: Qt.KeyboardModifiers) -> str:
        """Apply the selection rules of pressing a clip.  Returns the *deferred* change that only happens when the
        press turns out to be a plain click: ``"collapse"`` (plain press inside a multi-selection) or ``"toggle"``
        (Ctrl on an already selected clip)."""
        st = self.state
        if mods & Qt.ShiftModifier:
            st.select(self._range_ids(clip_id), "add" if mods & Qt.ControlModifier else "replace")
            st.select_cues([])
            return ""
        if mods & Qt.ControlModifier:
            if clip_id in st.selection:
                return "toggle"
            st.select([clip_id], "add")
            self._anchor = clip_id
            return ""
        self._anchor = clip_id
        st.select_cues([])
        if clip_id in st.selection:
            return "collapse" if len(st.selection) > 1 else ""
        st.select([clip_id])
        return ""

    def _range_ids(self, clip_id: str) -> list[str]:
        """Clips of the same track between the selection anchor and ``clip_id`` (inclusive)."""
        lane, _ = self._clip_index[clip_id]
        ids = [c.id for c in lane.track.clips]
        anchor = self._anchor if self._anchor in ids else clip_id
        i0, i1 = sorted((ids.index(anchor), ids.index(clip_id)))
        return ids[i0:i1 + 1]

    def _click_select_cue(self, cue_id: str, mods: Qt.KeyboardModifiers) -> str:
        st = self.state
        cues = list(self._cue_index.values())
        if mods & Qt.ShiftModifier:
            ids = [q.id for q in cues]
            anchor = self._anchor if self._anchor in ids else cue_id
            i0, i1 = sorted((ids.index(anchor), ids.index(cue_id)))
            st.select_cues(ids[i0:i1 + 1], "add" if mods & Qt.ControlModifier else "replace")
            st.select([])
            return ""
        if mods & Qt.ControlModifier:
            if cue_id in st.cue_selection:
                return "toggle"
            st.select_cues([cue_id], "add")
            self._anchor = cue_id
            return ""
        self._anchor = cue_id
        st.select([])
        if cue_id in st.cue_selection:
            return "collapse" if len(st.cue_selection) > 1 else ""
        st.select_cues([cue_id])
        return ""

    def _toggle_track_flag(self, lane: Lane, flag: str) -> None:
        tr = lane.track
        on = not getattr(tr, flag)
        if flag == "locked":
            label = "鎖定軌道" if on else "解除鎖定軌道"
        else:
            label = ("隱藏軌道" if on else "顯示軌道") if tr.kind == VIDEO else ("靜音軌道" if on else "取消靜音軌道")
        self._op(label, "set_track_props", tr.id, **{flag: on})

    # ---- snapping
    def _snap_points(self, exclude_clips: Iterable[str] = (), exclude_cues: Iterable[str] = (),
                     with_cues: bool = False) -> list[float]:
        """Sorted snap targets: clip edges (except ``exclude_clips``), playhead, markers, range ends and 0."""
        seq = self._seq()
        pts = {0.0, round(self.state.playhead, 6)}
        if seq is None:
            return sorted(pts)
        skip = set(exclude_clips)
        pts.update(round(m.time, 6) for m in seq.markers)
        rng = self._range()
        if rng:
            pts.update(rng)
        for tr in seq.tracks():
            for c in tr.clips:
                if c.id not in skip:
                    pts.add(round(c.start, 6))
                    pts.add(round(c.end, 6))
        if with_cues:
            skip_q = set(exclude_cues)
            for q in seq.cues:
                if q.id not in skip_q:
                    pts.add(round(q.start, 6))
                    pts.add(round(q.end, 6))
        return sorted(pts)

    def _snap_edges(self, edges: list[float], points: list[float],
                    mods: Qt.KeyboardModifiers) -> tuple[float, Optional[float]]:
        """``(shift to apply, snapped time)``; ``(0, None)`` when snapping is off or Alt is held."""
        if not self._snapping or (mods & Qt.AltModifier):
            return 0.0, None
        hit = nearest_shift(edges, points, SNAP_PX / self._pps)
        return (hit[0], hit[1]) if hit else (0.0, None)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._gesture is not None:
            self._gesture.move(pos, event.modifiers())
            return
        hit = self.hit_test(pos)
        self._set_hover(hit.clip_id)
        self.setCursor(self._cursor_for(hit))

    @staticmethod
    def _cursor_for(hit: Hit) -> Qt.CursorShape:
        if hit.kind in ("clip_l", "clip_r", "xfade", "cue_l", "cue_r", "fade_in", "fade_out"):
            return Qt.SizeHorCursor
        if hit.kind in ("header_mute", "header_lock", "marker"):
            return Qt.PointingHandCursor
        return Qt.ArrowCursor

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        g, self._gesture = self._gesture, None
        if g is not None and event.button() in (Qt.LeftButton, Qt.MiddleButton):
            g.release(event.position(), event.modifiers())
        elif g is not None:
            self._gesture = g

    # ================================================================== Qt events: keyboard
    def keyPressEvent(self, event: QKeyEvent) -> None:
        k, mods = event.key(), event.modifiers()
        ctrl, shift = bool(mods & Qt.ControlModifier), bool(mods & Qt.ShiftModifier)
        if k in (Qt.Key_Left, Qt.Key_Right):
            sign = -1 if k == Qt.Key_Left else 1
            self.step_frames(sign * (int(round(self.fps)) if shift else 1))
        elif k == Qt.Key_Home:
            self._goto(0.0)
        elif k == Qt.Key_End:
            self._goto(self._duration)
        elif k == Qt.Key_Up:
            self.goto_prev_cut()
        elif k == Qt.Key_Down:
            self.goto_next_cut()
        elif (k == Qt.Key_S and not ctrl) or (k == Qt.Key_K and ctrl):
            self.split_at_playhead()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selection(ripple=shift)
        elif k == Qt.Key_M and not ctrl:
            self.add_marker_at_playhead()
        elif k == Qt.Key_I and not ctrl:
            self.set_in_at_playhead()
        elif k == Qt.Key_O and not ctrl:
            self.set_out_at_playhead()
        elif k == Qt.Key_A and ctrl:
            self.select_all()
        elif k == Qt.Key_C and ctrl:
            self.copy_selection()
        elif k == Qt.Key_V and ctrl:
            self.paste_at()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_out()
        elif k == Qt.Key_Backslash:
            self.zoom_fit()
        elif k == Qt.Key_Escape:
            self._escape()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _escape(self) -> None:
        if self._gesture is not None:
            self._abort_gesture()
        else:
            self.state.clear_selection()

    # ================================================================== public editing / navigation API (§11.1)
    def _goto(self, t: float) -> None:
        self.state.set_playhead(t)
        self._reveal(self.state.playhead)

    def step_frames(self, n: int) -> None:
        self._goto(self.state.playhead + n / self.fps)

    def _cut_points(self) -> list[float]:
        seq = self._seq()
        if seq is None:
            return []
        pts = {0.0}
        for _, c in seq.all_clips():
            pts.add(round(c.start, 6))
            pts.add(round(c.end, 6))
        return sorted(pts)

    def goto_prev_cut(self) -> None:
        t = self.state.playhead
        prev = [p for p in self._cut_points() if p < t - 1e-6]
        if prev:
            self._goto(prev[-1])

    def goto_next_cut(self) -> None:
        t = self.state.playhead
        nxt = [p for p in self._cut_points() if p > t + 1e-6]
        if nxt:
            self._goto(nxt[0])

    def selected_clip_ids(self) -> list[str]:
        return [cid for cid in self.state.selection if cid in self._clip_index]

    def selected_cue_ids(self) -> list[str]:
        return [qid for qid in self.state.cue_selection if qid in self._cue_index]

    def select_all(self) -> None:
        self.state.select(self._clip_index.keys(), "replace")
        self.state.select_cues(self._cue_index.keys(), "replace")

    def add_marker_at_playhead(self) -> None:
        self._op("新增標記", "add_marker", self.state.playhead)

    def set_in_at_playhead(self) -> None:
        t, rng = self.state.playhead, self._range()
        end = rng[1] if rng and rng[1] > t else (self._duration if self._duration > t + 1.0 / self.fps else t + 1.0)
        self._op("設定入點", "set_sequence_range", t, end)

    def set_out_at_playhead(self) -> None:
        t, rng = self.state.playhead, self._range()
        start = rng[0] if rng and rng[0] < t else 0.0
        self._op("設定出點", "set_sequence_range", start, t)

    def clear_range(self) -> None:
        if self._range() is not None:
            self._op("清除入出點", "set_sequence_range", None, None)

    def _op(self, label: str, name: str, /, *args: Any, **kw: Any) -> Any:
        """``state.op`` that never raises: an ``OpError`` is already on the status line, we just repaint."""
        try:
            return self.state.op(label, name, *args, **kw)
        except OpError:
            self.update()
            return None

    def _do(self, label: str, fn: Callable) -> Any:
        try:
            return self.state.do(label, fn)
        except OpError:
            self.update()
            return None

    def delete_selection(self, ripple: bool = False) -> None:
        """Delete the selected clips and caption cues as one undo step (``ripple``: close the gaps on those tracks)."""
        clips, cues = self.selected_clip_ids(), self.selected_cue_ids()
        if not clips and not cues:
            return
        sid = self.state.seq_id

        def work(p):
            seq = p.sequence(sid)
            if clips:
                ops.remove_clips(p, seq, clips, ripple=ripple)
            if cues:
                ops.remove_cues(p, seq, cues)
        self._do("波紋刪除" if ripple else "刪除", work)

    def split_at_playhead(self) -> None:
        """Razor at the playhead: selected clips (and selected cues) under it, or every unlocked clip when nothing
        is selected.  One undo step."""
        t, sid = self.state.playhead, self.state.seq_id
        clips, cues = self.selected_clip_ids(), self.selected_cue_ids()

        def work(p):
            seq, n = p.sequence(sid), 0
            if clips or not cues:
                try:
                    n += len(ops.split_at_time(p, seq, t, clip_ids=clips or None))
                except OpError:
                    if not cues:
                        raise
            for qid in cues:
                q = seq.find_cue(qid)
                if q is not None and ops.cue_split_ok(seq, q, t):
                    ops.split_cue(p, seq, qid, t)
                    n += 1
            if not n:
                raise OpError("播放頭下沒有可分割的片段")
        self._do("分割", work)

    # ================================================================== track / marker / range commands
    def _confirm(self, title: str, text: str, yes: str = "確定") -> bool:
        from .media_bin import ask                        # late import: tests replace ``media_bin.ask``
        return bool(ask(self, title, text, yes, "取消"))

    def _ask_float(self, title: str, label: str, value: float, lo: float, hi: float, decimals: int = 2,
                   step: float = 0.1) -> Optional[float]:
        v, ok = QInputDialog.getDouble(self, title, label, value, lo, hi, decimals, Qt.WindowFlags(), step)
        return float(v) if ok else None

    def _ask_text(self, title: str, label: str, text: str = "") -> Optional[str]:
        v, ok = QInputDialog.getText(self, title, label, QLineEdit.Normal, text)
        return str(v) if ok else None

    def _ask_fades(self, fade_in: float, fade_out: float, limit: float) -> Optional[tuple[float, float]]:
        dlg = QDialog(self)
        dlg.setWindowTitle("淡入淡出")
        form = QFormLayout(dlg)
        boxes = []
        for text, v in (("淡入（秒）", fade_in), ("淡出（秒）", fade_out)):
            b = QDoubleSpinBox(dlg)
            b.setDecimals(2)
            b.setSingleStep(0.1)
            b.setRange(0.0, max(0.0, limit))
            b.setValue(min(v, limit))
            form.addRow(text, b)
            boxes.append(b)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        bb.button(QDialogButtonBox.Ok).setText("確定")
        bb.button(QDialogButtonBox.Cancel).setText("取消")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted:
            return None
        return boxes[0].value(), boxes[1].value()

    def add_track(self, kind: str) -> None:
        """New video track (on top of the others) or audio track (below them)."""
        self._op("新增視訊軌" if kind == VIDEO else "新增音訊軌", "add_track", kind)

    def remove_track(self, track_id: str, force: Optional[bool] = None) -> bool:
        """Delete a track.  A track that still holds clips is only removed after the user confirmed
        (``force=True`` skips the question).  Returns whether the track is gone."""
        seq = self._seq()
        tr = seq.find_track(track_id) if seq is not None else None
        if tr is None:
            return False
        if tr.clips and force is None:
            force = self._confirm("刪除軌道", f"軌道「{tr.name}」內還有 {len(tr.clips)} 個片段，要連同片段一起刪除嗎？", "刪除")
            if not force:
                return False
        self._op("刪除軌道", "remove_track", track_id, force=bool(force))
        seq = self._seq()
        return seq is not None and seq.find_track(track_id) is None

    def rename_track(self, track_id: str, name: Optional[str] = None) -> None:
        seq = self._seq()
        tr = seq.find_track(track_id) if seq is not None else None
        if tr is None:
            return
        if name is None:
            name = self._ask_text("重新命名軌道", "軌道名稱：", tr.name)
        name = (name or "").strip()
        if name and name != tr.name:
            self._op("重新命名軌道", "set_track_props", track_id, name=name)

    @staticmethod
    def _has_gap(track: Track) -> bool:
        end: Optional[float] = None
        for c in sorted(track.clips, key=lambda o: o.start):
            if end is not None and c.start > end + EPS:
                return True
            end = c.end if end is None else max(end, c.end)
        return False

    def close_gaps(self, track_id: str) -> float:
        """Remove every gap after the first clip of a track; returns the seconds closed."""
        seq = self._seq()
        tr = seq.find_track(track_id) if seq is not None else None
        if tr is None:
            return 0.0
        if not self._has_gap(tr):
            self.state.status.emit(f"軌道「{tr.name}」沒有間隙")
            return 0.0
        closed = self._op("移除間隙", "close_gaps", track_id)
        return float(closed or 0.0)

    def add_marker_at(self, t: float, name: str = "") -> None:
        self._op("新增標記", "add_marker", max(0.0, t), name)

    def remove_marker(self, marker_id: str) -> None:
        self._op("刪除標記", "remove_marker", marker_id)

    # ================================================================== caption commands
    def add_cue_at(self, t: float, dur: float = 2.0, text: str = "新字幕"):
        cue = self._op("新增字幕", "add_cue", max(0.0, t), max(0.0, t) + dur, text)
        if cue is not None:
            self.state.select([])
            self.state.select_cues([cue.id])
        return cue

    def merge_selected_cues(self) -> None:
        ids = self.selected_cue_ids()
        if len(ids) < 2:
            self.state.status.emit("請先選取兩則以上的字幕再合併")
            return
        self._op("合併字幕", "merge_cues", ids)

    def edit_cue_text(self, cue_id: str) -> Optional[CueEdit]:
        """Open the in-place editor over a caption block (Enter / focus loss = ``update_cue(text=…)``, Esc = cancel)."""
        q = self._cue_index.get(cue_id)
        if q is None:
            return None
        self._close_cue_edit(commit=True)
        self._reveal(q.start)
        ed = CueEdit(self, q.text)
        ed.committed.connect(lambda text, cid=cue_id: self._commit_cue_edit(cid, text))
        ed.cancelled.connect(self._drop_cue_edit)
        self._cue_edit = ed
        self._cue_edit_id = cue_id
        self._place_cue_edit()
        ed.show()
        ed.setFocus(Qt.OtherFocusReason)
        ed.selectAll()
        return ed

    def cue_editor(self) -> Optional[CueEdit]:
        return self._cue_edit

    def _place_cue_edit(self) -> None:
        ed = self._cue_edit
        if ed is None or self._cue_edit_id not in self._cue_index:
            return
        r, view = self.cue_rect(self._cue_edit_id), self._tracks_view_rect()
        x0 = clamp(r.left(), view.left(), max(view.left(), view.right() - 60))
        w = min(max(r.right() - x0, 220), view.right() - x0)
        ed.setGeometry(int(x0), r.top(), int(max(60, w)), r.height())

    def _drop_cue_edit(self) -> None:
        ed, self._cue_edit = self._cue_edit, None
        if ed is not None:
            ed.hide()
            ed.deleteLater()
            self.setFocus(Qt.OtherFocusReason)

    def _close_cue_edit(self, commit: bool) -> None:
        ed = self._cue_edit
        if ed is None:
            return
        ed.finish() if commit else ed.abandon()
        self._drop_cue_edit()

    def _commit_cue_edit(self, cue_id: str, text: str) -> None:
        self._drop_cue_edit()
        q = self._cue_index.get(cue_id)
        if q is not None and text != q.text:
            self._op("編輯字幕", "update_cue", cue_id, text=text)

    # ================================================================== clip commands (menus + MainWindow)
    def _sel_clips(self) -> list[Clip]:
        return [self._clip_index[i][1] for i in self.selected_clip_ids()]

    def set_props_selected(self, label: str, **props: Any) -> None:
        """``set_clip_props`` on every selected clip as ONE undo step."""
        ids, sid = self.selected_clip_ids(), self.state.seq_id
        if not ids:
            return

        def work(p):
            seq = p.sequence(sid)
            for cid in ids:
                ops.set_clip_props(p, seq, cid, **props)
        self._do(label, work)

    def toggle_mute_selected(self) -> None:
        clips = self._sel_clips()
        if clips:
            on = not all(c.muted for c in clips)
            self.set_props_selected("靜音片段" if on else "取消靜音片段", muted=on)

    def toggle_enabled_selected(self) -> None:
        clips = self._sel_clips()
        if clips:
            on = all(not c.enabled for c in clips)
            self.set_props_selected("啟用片段" if on else "停用片段", enabled=on)

    def set_speed_selected(self, speed: Optional[float] = None) -> None:
        ids, sid = self.selected_clip_ids(), self.state.seq_id
        if not ids:
            return
        if speed is None:
            speed = self._ask_float("速度", f"播放速度（{MIN_SPEED:g}–{MAX_SPEED:g} 倍）：",
                                    self._clip_index[ids[0]][1].speed, MIN_SPEED, MAX_SPEED, 2, 0.25)
            if speed is None:
                return

        def work(p):
            seq = p.sequence(sid)
            for cid in ids:
                ops.set_speed(p, seq, cid, speed, ripple=True)
        self._do("變更速度", work)

    def set_fades_selected(self, fade_in: Optional[float] = None, fade_out: Optional[float] = None) -> None:
        clips = self._sel_clips()
        if not clips:
            return
        if fade_in is None and fade_out is None:
            c = clips[0]
            got = self._ask_fades(c.fade_in, c.fade_out, c.duration)
            if got is None:
                return
            fade_in, fade_out = got
        props = {k: v for k, v in (("fade_in", fade_in), ("fade_out", fade_out)) if v is not None}
        self.set_props_selected("設定淡入淡出", **props)

    def set_crossfade_selected(self, dur: Optional[float] = None) -> None:
        ids, sid = self.selected_clip_ids(), self.state.seq_id
        if len(ids) != 1:
            self.state.status.emit("請只選取一個片段再設定交叉淡化")
            return
        lane, c = self._clip_index[ids[0]]
        prevs = [o for o in lane.track.clips if o is not c and o.start < c.start - EPS]
        if not prevs:
            self.state.status.emit("前面沒有片段可以做交叉淡化")
            return
        a = max(prevs, key=lambda o: o.start)
        limit = 0.5 * min(a.duration, c.duration)
        if dur is None:
            cur = max(0.0, a.end - c.start)
            dur = self._ask_float("交叉淡化", f"交叉淡化長度（秒，最長 {limit:.2f}）：", cur if cur > EPS else min(0.5, limit),
                                  0.0, max(limit, 0.0), 2, 0.1)
            if dur is None:
                return
        self._op("設定交叉淡化", "set_crossfade", ids[0], dur)

    def reframe_reset_selected(self) -> None:
        ids, sid = [i for i in self.selected_clip_ids() if self._clip_index[i][0].kind == "video"], self.state.seq_id
        if not ids:
            return

        def work(p):
            seq = p.sequence(sid)
            for cid in ids:
                ops.clear_transform_keys(p, seq, cid)
                ops.set_clip_props(p, seq, cid, zoom=1.0, cx=0.5, cy=0.5)
        self._do("重設取景", work)

    def reframe_add_key(self) -> None:
        """Add a crop keyframe at the playhead to the selected clip (it keeps the current framing there)."""
        t = self.state.playhead
        for c in self._sel_clips():
            if self._clip_index[c.id][0].kind == "video" and c.start - EPS <= t <= c.end + EPS:
                local = t - c.start
                cx, cy = interp_keys(c.transform.keys, local, c.transform.cx, c.transform.cy)
                self._op("加入取景關鍵格", "set_transform_key", c.id, local, cx, cy)
                return
        self.state.status.emit("播放頭不在選取的視訊片段內")

    def reframe_clear_keys(self) -> None:
        ids, sid = [i for i in self.selected_clip_ids() if self._clip_index[i][1].transform.keys], self.state.seq_id
        if not ids:
            return

        def work(p):
            seq = p.sequence(sid)
            for cid in ids:
                ops.clear_transform_keys(p, seq, cid)
        self._do("清除取景關鍵格", work)

    def reveal_in_bin(self, clip_id: str = "") -> None:
        cid = clip_id or next(iter(self.selected_clip_ids()), "")
        if cid in self._clip_index:
            self.revealAssetRequested.emit(self._clip_index[cid][1].asset_id)

    def show_properties(self, clip_id: str = "") -> None:
        cid = clip_id or next(iter(self.selected_clip_ids()), "")
        if cid in self._clip_index:
            self.propertiesRequested.emit(cid)

    # ---- copy / paste
    def copy_selection(self) -> int:
        """Remember the selected clips (with their settings) for ``paste_at``; returns how many."""
        seq = self._seq()
        ids = set(self.selected_clip_ids())
        if seq is None or not ids:
            return 0
        t0 = min(c.start for _, c in seq.all_clips() if c.id in ids)
        items = []
        for tr in seq.tracks():
            lst = seq.video_tracks if tr.kind == VIDEO else seq.audio_tracks
            for c in tr.clips:
                if c.id in ids:
                    items.append((tr.kind, lst.index(tr), c.start - t0, c.to_dict()))
        TimelineWidget._clipboard = items
        self.state.status.emit(f"已複製 {len(items)} 個片段")
        return len(items)

    def paste_at(self, t: Optional[float] = None, track_id: Optional[str] = None) -> list[str]:
        """Paste the copied clips with their first clip at ``t`` (default: the playhead); one undo step.
        ``track_id`` receives them when they came from a single track of the same kind."""
        items, sid = list(TimelineWidget._clipboard), self.state.seq_id
        if not items:
            self.state.status.emit("剪貼簿裡沒有片段（先選取片段按 Ctrl+C）")
            return []
        at = self.state.playhead if t is None else max(0.0, t)
        single = len({(k, i) for k, i, _, _ in items}) == 1

        def work(p):
            seq, out, skipped = p.sequence(sid), [], 0
            for kind, idx, rel, d in items:
                lst = seq.video_tracks if kind == VIDEO else seq.audio_tracks
                tr = seq.find_track(track_id) if (track_id and single) else None
                if tr is None or tr.kind != kind:
                    tr = lst[min(idx, len(lst) - 1)] if lst else None
                if tr is None or tr.locked:
                    skipped += 1
                    continue
                src = Clip.from_dict(d)
                c = ops.place_clip(p, seq, src.asset_id, tr.id, at + rel, src.src_in, src.src_out, "overwrite",
                                   speed=src.speed, volume=src.volume, muted=src.muted, enabled=src.enabled,
                                   fade_in=src.fade_in, fade_out=src.fade_out, label=src.label, color=src.color,
                                   note=src.note, transform=copy.deepcopy(src.transform), look=copy.deepcopy(src.look))
                out.append(c.id)
            if not out:
                raise OpError("沒有可以貼上的軌道（軌道已鎖定？）")
            return out
        ids = self._do("貼上片段", work)
        if ids:
            self.state.select(ids)
            self.state.select_cues([])
        return list(ids or [])

    # ================================================================== context menus
    def contextMenuEvent(self, event) -> None:
        pos = event.pos()
        hit = self.hit_test(pos)
        if hit.kind == "none" and not self._tracks_view_rect().contains(pos):
            return
        menu = self.menu_for_hit(hit)
        if menu is not None:
            self._popup_menu(menu, event.globalPos())
        event.accept()

    def _popup_menu(self, menu: QMenu, global_pos) -> None:
        """Run ``menu`` modally (tests replace this: a modal ``exec`` would block an offscreen run forever)."""
        menu.exec(global_pos)
        menu.deleteLater()

    def menu_for_hit(self, hit: Hit) -> Optional[QMenu]:
        """The context menu for what ``hit`` describes (an unselected clip / caption block is selected first, like
        every editor does).  Every entry has an object name; each one calls a public method of the widget."""
        if hit.clip_id in self._clip_index:
            if hit.clip_id not in self.state.selection:
                self.state.select([hit.clip_id])
                self.state.select_cues([])
            return self._clip_menu(hit)
        if hit.cue_id in self._cue_index:
            if hit.cue_id not in self.state.cue_selection:
                self.state.select_cues([hit.cue_id])
                self.state.select([])
            return self._cue_menu(hit)
        if hit.kind == "marker":
            return self._marker_menu(hit)
        if hit.kind == "ruler":
            return self._ruler_menu(hit)
        if hit.kind in ("header", "header_mute", "header_lock"):
            return self._header_menu(hit.lane)
        if hit.kind == "corner":
            return None
        return self._empty_menu(hit)

    @staticmethod
    def _entry(menu: QMenu, key: str, text: str, slot: Callable, enabled: bool = True) -> QAction:
        act = menu.addAction(text)
        act.setObjectName(key)
        act.setEnabled(bool(enabled))
        act.triggered.connect(lambda _checked=False, f=slot: f())
        return act

    def _clip_menu(self, hit: Hit) -> QMenu:
        ids = self.selected_clip_ids() or [hit.clip_id]
        pairs = [self._clip_index[i] for i in ids]
        clips = [c for _, c in pairs]
        unlocked = not any(ln.track.locked for ln, _ in pairs)
        t = self.state.playhead
        lane0, c0 = pairs[0]
        single = len(clips) == 1
        menu = QMenu(self)
        e = self._entry
        e(menu, "split", "在播放頭分割", self.split_at_playhead, unlocked and any(c.start + EPS < t < c.end - EPS for c in clips))
        e(menu, "delete", "刪除", self.delete_selection, unlocked)
        e(menu, "ripple_delete", "波紋刪除", lambda: self.delete_selection(True), unlocked)
        menu.addSeparator()
        e(menu, "mute", "取消靜音" if all(c.muted for c in clips) else "靜音", self.toggle_mute_selected, unlocked)
        e(menu, "disable", "啟用" if all(not c.enabled for c in clips) else "停用", self.toggle_enabled_selected, unlocked)
        menu.addSeparator()
        e(menu, "speed", "速度…", self.set_speed_selected, unlocked)
        e(menu, "fades", "淡入淡出…", self.set_fades_selected, unlocked)
        has_prev = single and any(o is not c0 and o.start < c0.start - EPS for o in lane0.track.clips)
        e(menu, "crossfade", "交叉淡化…", self.set_crossfade_selected, unlocked and has_prev)
        sub = menu.addMenu("取景")
        sub.menuAction().setObjectName("reframe_menu")
        sub.menuAction().setEnabled(unlocked and any(ln.kind == "video" for ln, _ in pairs))
        e(sub, "reframe_edit", "取景編輯…", lambda: self.state.request_reframe(hit.clip_id), single)
        e(sub, "reframe_reset", "重設取景", self.reframe_reset_selected)
        e(sub, "reframe_key", "在播放頭加入關鍵格", self.reframe_add_key)
        e(sub, "reframe_clear", "清除關鍵格", self.reframe_clear_keys, any(c.transform.keys for c in clips))
        menu.addSeparator()
        e(menu, "copy", "複製", self.copy_selection)
        e(menu, "reveal", "在素材庫顯示", lambda: self.reveal_in_bin(hit.clip_id))
        e(menu, "properties", "屬性", lambda: self.show_properties(hit.clip_id))
        return menu

    def _cue_menu(self, hit: Hit) -> QMenu:
        ids = self.selected_cue_ids() or [hit.cue_id]
        t = self.state.playhead
        menu = QMenu(self)
        e = self._entry
        e(menu, "edit_text", "編輯文字", lambda: self.edit_cue_text(hit.cue_id))
        e(menu, "split", "在播放頭分割", self.split_at_playhead,
          any(ops.cue_split_ok(self.state.sequence(), self._cue_index[i], t) for i in ids))
        e(menu, "merge", "合併字幕", self.merge_selected_cues, len(ids) >= 2)
        menu.addSeparator()
        e(menu, "delete", "刪除", self.delete_selection)
        return menu

    def _header_menu(self, lane: Lane) -> QMenu:
        tr, seq = lane.track, self._seq()
        menu = QMenu(self)
        e = self._entry
        e(menu, "add_video", "新增視訊軌", lambda: self.add_track(VIDEO))
        e(menu, "add_audio", "新增音訊軌", lambda: self.add_track(AUDIO))
        if tr is not None and seq is not None:
            menu.addSeparator()
            e(menu, "rename", "重新命名…", lambda: self.rename_track(tr.id))
            mute = ("顯示軌道" if tr.muted else "隱藏軌道") if tr.kind == VIDEO else ("取消靜音" if tr.muted else "靜音軌道")
            e(menu, "mute", mute, lambda: self._toggle_track_flag(lane, "muted"))
            e(menu, "lock", "解除鎖定" if tr.locked else "鎖定軌道", lambda: self._toggle_track_flag(lane, "locked"))
            e(menu, "close_gaps", "移除間隙", lambda: self.close_gaps(tr.id), not tr.locked and self._has_gap(tr))
            menu.addSeparator()
            removable = not tr.locked and not (tr.kind == VIDEO and len(seq.video_tracks) <= 1)
            e(menu, "remove_track", "刪除軌道", lambda: self.remove_track(tr.id), removable)
        return menu

    def _empty_menu(self, hit: Hit) -> QMenu:
        lane, t = hit.lane, snap_time(max(0.0, hit.t), self.fps)
        tr = lane.track if lane is not None else None
        menu = QMenu(self)
        e = self._entry
        e(menu, "paste", "貼上", lambda: self.paste_at(t, tr.id if tr is not None else None), bool(TimelineWidget._clipboard))
        e(menu, "marker", "在此新增標記", lambda: self.add_marker_at(t))
        if lane is not None and lane.kind == "caption":
            e(menu, "add_cue", "在此新增字幕", lambda: self._add_cue_and_edit(t))
        if tr is not None:
            e(menu, "close_gaps", "移除間隙", lambda: self.close_gaps(tr.id), not tr.locked and self._has_gap(tr))
        return menu

    def _add_cue_and_edit(self, t: float) -> None:
        cue = self.add_cue_at(t)
        if cue is not None:
            self.edit_cue_text(cue.id)

    def _marker_menu(self, hit: Hit) -> QMenu:
        menu = QMenu(self)
        self._entry(menu, "goto", "跳到標記", lambda: self._goto(hit.t))
        self._entry(menu, "remove_marker", "刪除標記", lambda: self.remove_marker(hit.marker_id))
        return menu

    def _ruler_menu(self, hit: Hit) -> QMenu:
        t = snap_time(max(0.0, hit.t), self.fps)
        menu = QMenu(self)
        self._entry(menu, "marker", "在此新增標記", lambda: self.add_marker_at(t))
        self._entry(menu, "clear_range", "清除入出點", self.clear_range, self._range() is not None)
        return menu

    # ================================================================== Qt events: double click
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        hit = self.hit_test(event.position())
        if hit.clip_id in self._clip_index:
            self.show_properties(hit.clip_id)
        elif hit.cue_id in self._cue_index:
            self.edit_cue_text(hit.cue_id)
        elif hit.kind == "header" and hit.lane is not None and hit.lane.track is not None:
            self.rename_track(hit.lane.track.id)
        event.accept()

    # ================================================================== Qt events: drag & drop
    def _drop_session(self, mime) -> Optional[DropSession]:
        if self._seq() is None:
            return None
        if mime.hasFormat(ASSET_MIME):
            assets = self.state.project.assets
            items = [DropItem(i, asset_span(assets[i]), asset_lane_kind(assets[i])) for i in decode_assets(mime)
                     if i in assets]
            return DropSession(True, items) if items else None
        paths = dropped_paths(mime)
        if not paths:
            return None
        items = []
        for pth in paths:
            ext = os.path.splitext(pth)[1].lower()
            if ext in AUDIO_EXTS:
                items.append(DropItem("", DROP_GHOST_S, AUDIO))
            elif ext in VIDEO_EXTS or ext in IMAGE_EXTS or os.path.isdir(pth):
                items.append(DropItem("", DROP_GHOST_S, VIDEO))
        return DropSession(False, items, paths)

    def _drop_track(self, kind: str, under: Optional[Lane]) -> Optional[Track]:
        """The lane under the pointer when it suits ``kind``, else the first unlocked track of that kind."""
        if under is not None and under.track is not None and under.kind == kind and not under.track.locked:
            return under.track
        seq = self._seq()
        tracks = (seq.video_tracks if kind == VIDEO else seq.audio_tracks) if seq is not None else []
        return next((t for t in tracks if not t.locked), None)

    def plan_drop(self, sess: DropSession, pos: QPointF, mods) -> Optional[DropPlan]:
        """Where dragged items would land for a pointer at ``pos`` (frame-aligned, snapped); ``None`` when no
        track can take them."""
        under = self.lane_at_y(pos.y())
        t0 = max(0.0, self.x_to_time(pos.x())) if pos.x() >= HEADER_W else 0.0
        picked = [(it, tr) for it in sess.items for tr in [self._drop_track(it.kind, under)] if tr is not None]
        if not picked:
            return None
        first = picked[0][1]
        chain = sum(it.dur for it, tr in picked if tr.id == first.id)
        shift, guide = self._snap_edges([t0, t0 + chain], self._snap_points(), mods)
        start = max(0.0, t0 + shift) if guide is not None else snap_time(t0, self.fps)
        ends: dict[str, float] = {}
        entries = []
        for it, tr in picked:
            at = ends.get(tr.id, start)
            entries.append((it, tr, at))
            ends[tr.id] = at + it.dur
        return DropPlan(start, entries, self._lane_by_id.get(first.id), guide)

    def _drop_overlay(self, sess: DropSession, plan: Optional[DropPlan], pos: QPointF, mods) -> Overlay:
        if plan is None:
            return Overlay(drop_hint=True)
        ghosts = []
        for it, tr, at in plan.entries:
            lane = self._lane_by_id[tr.id]
            ghosts.append((QRectF(self.time_to_x(at), self._lane_top(lane) + CLIP_INSET, max(2.0, it.dur * self._pps),
                                  lane.h - 2 * CLIP_INSET), "drop"))
        n = len(plan.entries)
        tip = fmt_timecode(plan.start, self.fps) + (f"　{n} 個素材" if n > 1 else "")
        if sess.internal and mods & Qt.ControlModifier:
            tip += "　插入"
        if not sess.internal:
            tip = "匯入並放到 " + tip
        lane = plan.lane
        rect = QRectF(HEADER_W, self._lane_top(lane), self._view_w(), lane.h) if lane is not None else None
        return Overlay(ghosts=ghosts, lane=rect, snap_t=plan.guide, tip=(pos, tip), drop_hint=True)

    def _drop_move(self, ev) -> None:
        if self._drop is not None:
            plan = self.plan_drop(self._drop, ev.position(), ev.modifiers())
            self._set_overlay(self._drop_overlay(self._drop, plan, ev.position(), ev.modifiers()))

    def dragEnterEvent(self, ev) -> None:
        sess = self._drop_session(ev.mimeData())
        if sess is None:
            ev.ignore()
            return
        self._abort_gesture()
        self._drop = sess
        ev.setDropAction(Qt.CopyAction)
        ev.accept()
        self._drop_move(ev)

    def dragMoveEvent(self, ev) -> None:
        if self._drop is None:
            ev.ignore()
            return
        ev.setDropAction(Qt.CopyAction)
        ev.accept()
        self._drop_move(ev)

    def dragLeaveEvent(self, ev) -> None:
        self._drop = None
        self._set_overlay(Overlay())
        ev.accept()

    def dropEvent(self, ev) -> None:
        sess = self._drop or self._drop_session(ev.mimeData())
        self._drop = None
        self._set_overlay(Overlay())
        if sess is None:
            ev.ignore()
            return
        pos, mods = ev.position(), ev.modifiers()
        plan = self.plan_drop(sess, pos, mods)
        if not sess.internal:                                   # OS files: the window imports them, then places them
            under = self.lane_at_y(pos.y())
            tid = under.track.id if under is not None and under.track is not None else ""
            t = plan.start if plan is not None else snap_time(max(0.0, self.x_to_time(pos.x())), self.fps)
            ev.setDropAction(Qt.CopyAction)
            ev.accept()
            self.externalFilesDropped.emit(list(sess.paths), float(t), tid)
            return
        if plan is None:
            self.state.status.emit("沒有可以放置的軌道（軌道已鎖定？）")
            ev.ignore()
            return
        ev.setDropAction(Qt.CopyAction)
        ev.accept()
        self.place_assets([(it.asset_id, tr.id) for it, tr, _ in plan.entries], plan.start,
                          "insert" if mods & Qt.ControlModifier else "overwrite")

    def place_assets(self, targets: list, start: float, mode: str = "overwrite") -> list[str]:
        """Put ``[(asset_id, track_id), ...]`` on the timeline back to back from ``start`` (per track), as ONE undo step;
        an empty sequence first takes the size / fps of the first picture (GUI_SPEC §4).  Returns the new clip ids."""
        sid = self.state.seq_id
        ids = [(a, t) for a, t in targets if a in self.state.project.assets]
        if not ids:
            return []

        def work(p):
            seq = p.sequence(sid)
            if not any(True for _ in seq.all_clips()):
                try:
                    from .importer import sequence_format_for
                    fmt = sequence_format_for(p.assets[a] for a, _ in ids)
                except ImportError:                              # pragma: no cover - importer is part of the app
                    fmt = None
                if fmt is not None:
                    ops.set_sequence_format(p, seq, *fmt)
            cursors: dict[str, float] = {}
            out = []
            for aid, tid in ids:
                at = cursors.get(tid, start)
                c = ops.place_clip(p, seq, aid, tid, at, mode=mode)
                cursors[tid] = c.end
                out.append(c.id)
            return out
        new = self._do("加入時間軸", work)
        if new:
            self.state.select(new)
            self.state.select_cues([])
            self.state.status.emit(f"已加入 {len(new)} 個片段到時間軸（Ctrl+Z 可復原）")
        return list(new or [])
