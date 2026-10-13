"""Program monitor (GUI_SPEC §3): picture, transport bar, preview-cache controls and the reframe (取景) editor.

Threading: every frame is composed by ``render.live.LiveCompositor`` on ONE decode thread (``_FrameWorker``).  The GUI
thread only *submits* requests; a newer request replaces an older one that was not started yet, so while playing or
scrubbing the decoder always works on the latest playhead and the GUI never queues work (frames are dropped, never
delayed).  Results come back through a queued signal; results computed for an older edit are discarded.

Reframe mode shows the *whole* source frame of the selected ``cover`` clip with the output-aspect crop window on top.
All window maths comes from :mod:`autocut.core.geometry` (the functions the compositor and the ffmpeg compiler use);
every change goes through ``state.op`` / ``state.do`` (one undo step per mouse release).
"""
from __future__ import annotations

import math
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QIcon, QImage, QKeySequence, QPainter,
                           QPainterPath, QPen, QPixmap, QPolygonF, QShortcut)
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget)

from ..core import geometry as geo
from ..core import ops
from ..core.project import CaptionStyle, Project, Sequence
from ..core.timecode import EPS, fmt_timecode, frame_of
from ..render.ass import wrap_text
from ..render.live import LiveCompositor, LiveError
from . import theme
from .playback import (PREVIEW_FRESH, PREVIEW_NONE, PREVIEW_RENDERING, PREVIEW_STALE, PlaybackController)
from .state import AppState
from .widgets import TimecodeEdit
from .workers import TaskRunner

MAX_SIDE = 960              # long side of the frames the monitor asks the compositor for
MAX_ZOOM = 8.0              # same limit as ``ops.set_clip_props``
PIC_BG = "#0e0f11"          # behind the picture
NUDGE_COMMIT_MS = 350       # keyboard / wheel edits are committed (ONE undo step) after this much silence
MIN_WIDTH = 420              # the monitor never needs to be wider than this (rows regroup below ~800)
ICON_ONLY_WIDTH = 470       # below this the loop / range buttons drop their text
SHORT_INFO_WIDTH = 640      # below this the sequence info line loses its duration and aspect label


# ============================================================================ small helpers
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def fit_rect(w: float, h: float, area: QRectF) -> QRectF:
    """Largest ``w:h`` rectangle centred in ``area`` (letterbox), on whole pixels."""
    if w <= 0 or h <= 0 or area.width() <= 0 or area.height() <= 0:
        return QRectF(area)
    s = min(area.width() / w, area.height() / h)
    rw, rh = round(w * s), round(h * s)
    return QRectF(round(area.x() + (area.width() - rw) / 2.0), round(area.y() + (area.height() - rh) / 2.0), rw, rh)


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Own copy of a BGR uint8 frame as a QImage (safe to create off the GUI thread)."""
    arr = np.ascontiguousarray(bgr)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format_BGR888).copy()


def clip_local_time(seq: Sequence, clip, t: float) -> float:
    """Clip-local seconds of timeline time ``t`` – the clock the crop keys use (frame accurate, like the compositor)."""
    fps = seq.fps or 30.0
    last = max(0, frame_of(clip.duration, fps) - 1)
    k = _clamp(frame_of(t, fps) - frame_of(clip.start, fps), 0, last)
    return k / fps


def clip_contains(seq: Sequence, clip, t: float) -> bool:
    fps = seq.fps or 30.0
    f, a, b = frame_of(t, fps), frame_of(clip.start, fps), frame_of(clip.end, fps)
    return a <= f < max(a + 1, b)


def empty_message(seq: Sequence) -> str:
    """Text for a sequence that has nothing to show ('' when there is picture content)."""
    if any(tr.clips for tr in seq.video_tracks):
        return ""
    if any(tr.clips for tr in seq.audio_tracks):
        return "這個序列只有聲音，沒有畫面"
    return "這個序列還沒有片段\n把素材拖進時間軸，或按 Ctrl+I 匯入檔案"


# ---------------------------------------------------------------------------- icons (drawn, so no font dependency)
def _glyph(kind: str, color: str, size: int = 20) -> QPixmap:
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    pm.setDevicePixelRatio(dpr)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 20.0, size / 20.0)
    c = QColor(color)
    p.setPen(Qt.NoPen)
    p.setBrush(c)

    def tri(*pts: float) -> None:
        p.drawPolygon(QPolygonF([QPointF(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]))

    def bar(x: float, w: float = 2.4) -> None:
        p.drawRoundedRect(QRectF(x, 3.5, w, 13), 0.8, 0.8)

    if kind == "play":
        tri(5.5, 3.5, 16.5, 10, 5.5, 16.5)
    elif kind == "pause":
        bar(5, 3.6)
        bar(11.4, 3.6)
    elif kind == "first":
        bar(2.6)
        tri(11.2, 3.5, 5.2, 10, 11.2, 16.5)
        tri(17.4, 3.5, 11.4, 10, 17.4, 16.5)
    elif kind == "back":
        tri(14.2, 3.5, 6.0, 10, 14.2, 16.5)
        bar(15.0)
    elif kind == "fwd":
        bar(2.6)
        tri(5.8, 3.5, 14.0, 10, 5.8, 16.5)
    elif kind == "last":
        tri(2.6, 3.5, 8.6, 10, 2.6, 16.5)
        tri(8.8, 3.5, 14.8, 10, 8.8, 16.5)
        bar(15.0)
    elif kind == "loop":
        pen = QPen(c, 1.9)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(4, 4, 12, 12), 40 * 16, 280 * 16)
        ang = math.radians(320)                                   # arrow head at the end of the arc
        ex, ey = 10 + 6 * math.cos(ang), 10 - 6 * math.sin(ang)
        dx, dy = -math.sin(ang), -math.cos(ang)
        nx, ny = -dy, dx
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        tri(ex + dx * 3.6, ey + dy * 3.6, ex + nx * 2.6, ey + ny * 2.6, ex - nx * 2.6, ey - ny * 2.6)
    elif kind == "range":
        pen = QPen(c, 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        left = QPainterPath()
        left.moveTo(5.0, 3.5)
        left.lineTo(2.8, 3.5)
        left.lineTo(2.8, 16.5)
        left.lineTo(5.0, 16.5)
        right = QPainterPath()
        right.moveTo(15.0, 3.5)
        right.lineTo(17.2, 3.5)
        right.lineTo(17.2, 16.5)
        right.lineTo(15.0, 16.5)
        p.drawPath(left)
        p.drawPath(right)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        tri(7.4, 6.0, 13.0, 10, 7.4, 14.0)
    p.end()
    return pm


_ICONS: dict[str, QIcon] = {}


def icon(kind: str) -> QIcon:
    ic = _ICONS.get(kind)
    if ic is None:
        ic = QIcon()
        ic.addPixmap(_glyph(kind, theme.TEXT), QIcon.Normal)
        dim = QColor(theme.TEXT_DIM)
        dim.setAlpha(150)
        ic.addPixmap(_glyph(kind, dim.name(QColor.HexArgb)), QIcon.Disabled)
        _ICONS[kind] = ic
    return ic


# ============================================================================ reframe geometry (pure, no Qt widgets)
@dataclass
class CropBox:
    """The output-aspect window of a ``cover`` clip, in *display pixels* of its source frame.

    ``cx, cy`` is the requested window centre (0..1 of the frame).  The window is always kept inside the frame (the
    renderer does the same), so :meth:`centre` is the *effective* centre and :meth:`normalise` stores it back.
    Every number comes from :mod:`autocut.core.geometry`.
    """

    dw: float
    dh: float
    out_w: float
    out_h: float
    cx: float = 0.5
    cy: float = 0.5
    zoom: float = 1.0

    def size(self) -> tuple[float, float]:
        return geo.crop_window(self.dw, self.dh, self.out_w, self.out_h, self.zoom)

    def rect(self) -> tuple[float, float, float, float]:
        cw, ch = self.size()
        x, y = geo.crop_topleft(self.dw, self.dh, cw, ch, self.cx, self.cy)
        return x, y, cw, ch

    def centre(self) -> tuple[float, float]:
        x, y, cw, ch = self.rect()
        return (x + cw / 2.0) / self.dw, (y + ch / 2.0) / self.dh

    def centre_px(self) -> tuple[float, float]:
        cx, cy = self.centre()
        return cx * self.dw, cy * self.dh

    def normalise(self) -> None:
        self.cx, self.cy = self.centre()

    def move_to(self, px: float, py: float) -> None:
        """Centre the window on display pixel (px, py) (clamped so it stays inside the frame)."""
        self.cx, self.cy = px / self.dw, py / self.dh
        self.normalise()

    def nudge(self, dx: float, dy: float) -> None:
        """Move by fractions of the display size."""
        self.normalise()
        self.cx, self.cy = self.cx + dx, self.cy + dy
        self.normalise()

    def set_zoom(self, zoom: float) -> None:
        """Zoom about the window centre (``zoom`` 1..8; bigger = smaller window)."""
        self.normalise()
        self.zoom = _clamp(zoom, 1.0, MAX_ZOOM)
        self.normalise()

    def anchor_for(self, corner: str) -> tuple[float, float]:
        """Display pixel of the window corner opposite to ``corner`` (tl | tr | bl | br)."""
        x, y, cw, ch = self.rect()
        ax = x if corner.endswith("r") else x + cw
        ay = y if corner.startswith("b") else y + ch
        return ax, ay

    def corner_points(self) -> dict[str, tuple[float, float]]:
        x, y, cw, ch = self.rect()
        return {"tl": (x, y), "tr": (x + cw, y), "bl": (x, y + ch), "br": (x + cw, y + ch)}

    def resize_corner(self, corner: str, px: float, py: float, anchor: tuple[float, float]) -> None:
        """Drag ``corner`` to display pixel (px, py) while the opposite corner stays put (aspect is fixed)."""
        ax, ay = anchor
        sx = 1.0 if corner.endswith("r") else -1.0
        sy = 1.0 if corner.startswith("b") else -1.0
        bw, bh = geo.base_window(self.dw, self.dh, self.out_w, self.out_h)
        ratio = bw / bh
        w = max((px - ax) * sx, (py - ay) * sy * ratio)
        room_x = (self.dw - ax) if sx > 0 else ax
        room_y = (self.dh - ay) if sy > 0 else ay
        w = max(min(w, bw, room_x, room_y * ratio), bw / MAX_ZOOM)
        self.zoom = _clamp(bw / w, 1.0, MAX_ZOOM)
        h = w / ratio
        self.cx, self.cy = (ax + sx * w / 2.0) / self.dw, (ay + sy * h / 2.0) / self.dh
        self.normalise()


# ============================================================================ caption overlay (approximation of the ASS render)
_FAMILIES: Optional[set] = None


def _caption_font(style: CaptionStyle, px: float) -> QFont:
    global _FAMILIES
    if _FAMILIES is None:
        _FAMILIES = set(QFontDatabase.families())
    f = QFont(style.font if style.font in _FAMILIES else theme.pick_font_family())
    f.setPixelSize(max(6, int(round(px))))
    f.setBold(bool(style.bold))
    return f


def active_cue_lines(seq: Sequence, t: float) -> list[list[str]]:
    """Wrapped text lines of every cue showing at time ``t`` (same wrapping as the ASS writer)."""
    out: list[list[str]] = []
    units = max(4, seq.caption_style.max_chars)
    for q in seq.cues:
        if q.start - EPS <= t < q.end - EPS and q.text.strip():
            out.append([ln for ln in wrap_text(q.text.replace("{", "(").replace("}", ")"), units).split("\\N") if ln])
    return out


def draw_captions(p: QPainter, rect: QRectF, style: CaptionStyle, blocks: list[list[str]]) -> None:
    """Draw caption blocks over the picture ``rect`` using the sequence caption style (size, outline and margin are
    fractions of the picture height, as in the renderer).  Approximate: the real render is the reference."""
    if not blocks:
        return
    h = rect.height()
    px = max(8.0, style.size * h)
    font = _caption_font(style, px)
    fm = QFontMetricsF(font)
    line_h = fm.lineSpacing()
    outline = max(0.0, style.outline * h)
    pad = max(outline, px * 0.18) if style.box else 0.0
    gap = px * 0.3
    heights = [len(b) * line_h + 2 * pad for b in blocks]
    total = sum(heights) + gap * (len(blocks) - 1)
    margin = style.margin * h
    if style.position == "top":
        y = rect.top() + margin
    elif style.position == "center":
        y = rect.center().y() - total / 2.0
    else:
        y = rect.bottom() - margin - total
    p.save()
    p.setClipRect(rect)
    p.setRenderHint(QPainter.Antialiasing)
    for lines, bh in zip(blocks, heights):
        if style.box:
            wmax = max(fm.horizontalAdvance(ln) for ln in lines)
            box = QRectF(rect.center().x() - wmax / 2.0 - pad, y, wmax + 2 * pad, bh)
            col = QColor(style.box_color)
            col.setAlphaF(_clamp(style.box_opacity, 0.0, 1.0))
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(box, pad * 0.5, pad * 0.5)
        for i, line in enumerate(lines):
            x = rect.center().x() - fm.horizontalAdvance(line) / 2.0
            path = QPainterPath()
            path.addText(QPointF(x, y + pad + i * line_h + fm.ascent()), font, line)
            if style.shadow > 0:
                off = style.shadow * h
                p.fillPath(path.translated(off, off), QColor(0, 0, 0, 150))
            if outline > 0 and not style.box:
                pen = QPen(QColor(style.outline_color), outline * 2.0)
                pen.setJoinStyle(Qt.RoundJoin)
                pen.setCapStyle(Qt.RoundCap)
                p.strokePath(path, pen)
            p.fillPath(path, QColor(style.color))
        y += bh + gap
    p.restore()


def draw_guides(p: QPainter, rect: QRectF) -> None:
    """Action-safe (90 %), title-safe (80 %) rectangles and centre lines."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing, False)
    pen = QPen(QColor(255, 255, 255, 170), 1, Qt.DashLine)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    small = QFont(p.font())
    small.setPixelSize(10)
    p.setFont(small)
    for frac, label in ((0.9, "90%"), (0.8, "80%")):
        r = QRectF(0, 0, rect.width() * frac, rect.height() * frac)
        r.moveCenter(rect.center())
        p.drawRect(r)
        p.drawText(QPointF(r.left() + 3, r.top() + 11), label)
    c = rect.center()
    p.drawLine(QPointF(c.x(), rect.top()), QPointF(c.x(), rect.bottom()))
    p.drawLine(QPointF(rect.left(), c.y()), QPointF(rect.right(), c.y()))
    p.restore()


# ============================================================================ decode thread
@dataclass
class FrameRequest:
    serial: int
    t: float
    seq_id: str
    project: Project
    epoch: int
    source_clip: str = ""             # non-empty: also produce the whole source frame (reframe mode)
    source_t_local: float = 0.0


@dataclass
class FrameResult:
    serial: int
    epoch: int
    t: float
    frame: Optional[QImage] = None
    source: Optional[QImage] = None
    error: str = ""                   # the composite failed (unexpected)
    source_error: str = ""            # the source frame failed (LiveError text is already 繁中)
    missing: list = field(default_factory=list)       # file names the compositor could not open
    ms: float = 0.0


class _Bridge(QObject):
    ready = Signal(object)


class _FrameWorker:
    """One daemon thread owning the compositor.  ``submit`` keeps only the newest request (coalescing)."""

    def __init__(self, compositor: LiveCompositor, bridge: _Bridge) -> None:
        self.compositor = compositor
        self._bridge = bridge
        self._cv = threading.Condition()
        self._pending: Optional[FrameRequest] = None
        self._stop = False
        self._epoch_seen = -1
        self._seq_seen: Optional[str] = None
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, name="monitor-frames", daemon=True)
        self._thread.start()

    def submit(self, req: FrameRequest) -> None:
        with self._cv:
            if self._pending is not None:
                self.dropped += 1
            self._pending = req
            self._cv.notify()

    def stop(self, join: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._pending = None
            self._cv.notify()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(join)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    break
                req, self._pending = self._pending, None
            res = self._compose(req)
            try:
                self._bridge.ready.emit(res)
            except RuntimeError:                                  # the monitor was deleted
                break
        try:
            self.compositor.close()
        except Exception:                                         # noqa: BLE001
            pass

    def _compose(self, req: FrameRequest) -> FrameResult:
        t0 = time.perf_counter()
        comp = self.compositor
        res = FrameResult(req.serial, req.epoch, req.t)
        try:
            if comp.project is not req.project:
                comp.set_project(req.project)
                self._seq_seen = None
            if self._seq_seen != req.seq_id:
                comp.set_sequence(req.seq_id)
                self._seq_seen = req.seq_id
            if req.epoch != self._epoch_seen:
                comp.invalidate()
                self._epoch_seen = req.epoch
            res.frame = bgr_to_qimage(comp.frame_at(req.t))
            res.missing = sorted({os.path.basename(m) for m in comp.missing})
        except Exception as exc:                                  # noqa: BLE001 - never kill the thread
            res.error = str(exc) or exc.__class__.__name__
        if req.source_clip:
            try:
                res.source = bgr_to_qimage(comp.source_frame_at(req.source_clip, req.source_t_local))
            except LiveError as exc:
                res.source_error = str(exc)
            except Exception as exc:                              # noqa: BLE001
                res.source_error = str(exc) or exc.__class__.__name__
        res.ms = (time.perf_counter() - t0) * 1000.0
        return res


def _stop_worker(worker: _FrameWorker) -> None:
    worker.stop(0.5)


# ============================================================================ picture view (painting + reframe interaction)
class _PictureView(QWidget):
    """The picture.  Normal mode: letterboxed composite + captions + guides.  Reframe mode: the whole source frame with
    a draggable crop window.  It only draws and reports gestures; the monitor owns the logic."""

    reframeChanged = Signal()         # the box moved / zoomed (live, nothing stored yet)
    reframeReleased = Signal()        # mouse released: store now
    reframeNudged = Signal()          # wheel / keyboard: store after a short pause
    escapePressed = Signal()

    HANDLE = 6.0                      # half size of a corner handle (px)
    MARGIN = 6.0

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.image: Optional[QImage] = None
        self.shown_t: Optional[float] = None
        self.out_size = (16, 9)
        self.background = "#000000"
        self.message = ""
        self.warn = False
        self.guides = False
        self.captions = True
        # reframe
        self.rf_active = False
        self.source: Optional[QImage] = None
        self.box: Optional[CropBox] = None
        self.keys: list[tuple[float, float]] = []
        self.key_here = -1
        self.editable = True
        self.rf_note = ""
        self._drag: Optional[dict] = None

    @property
    def dragging(self) -> bool:
        return self._drag is not None

    # ---- model setters (the monitor calls these)
    def set_reframe(self, box: Optional[CropBox], keys: list, key_here: int, editable: bool, note: str = "") -> None:
        self.box, self.keys, self.key_here, self.editable, self.rf_note = box, keys, key_here, editable, note
        self.update()

    # ---- geometry
    def _area(self) -> QRectF:
        return QRectF(self.rect()).adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)

    def picture_rect(self) -> QRectF:
        """Where the composite is drawn (widget coordinates)."""
        if self.image is not None:
            return fit_rect(self.image.width(), self.image.height(), self._area())
        return fit_rect(self.out_size[0], self.out_size[1], self._area())

    def source_rect(self) -> Optional[QRectF]:
        if self.source is None:
            return None
        return fit_rect(self.source.width(), self.source.height(), self._area())

    def box_rect(self) -> Optional[QRectF]:
        """The crop window in widget coordinates (None outside reframe mode)."""
        sr = self.source_rect()
        if sr is None or self.box is None:
            return None
        s = sr.width() / self.box.dw
        x, y, w, h = self.box.rect()
        return QRectF(sr.x() + x * s, sr.y() + y * s, w * s, h * s)

    def _to_display(self, pos: QPointF, sr: QRectF) -> tuple[float, float]:
        s = sr.width() / self.box.dw
        return (pos.x() - sr.x()) / s, (pos.y() - sr.y()) / s

    # ---- painting
    def paintEvent(self, _ev) -> None:                              # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(PIC_BG))
        if self.rf_active:
            self._paint_reframe(p)
        else:
            self._paint_picture(p)
        p.end()

    def _paint_picture(self, p: QPainter) -> None:
        rect = self.picture_rect()
        if self.image is not None:
            p.drawImage(rect, self.image)
        else:
            p.fillRect(rect, QColor(self.background))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5))
        if self.image is not None and self.captions:
            try:
                seq = self.state.sequence()
                t = self.shown_t if self.shown_t is not None else self.state.playhead
                draw_captions(p, rect, seq.caption_style, active_cue_lines(seq, t))
            except KeyError:
                pass
        if self.guides:
            draw_guides(p, rect)
        if self.message:
            self._draw_message(p, rect, self.message, self.warn)

    def _draw_message(self, p: QPainter, rect: QRectF, text: str, warn: bool) -> None:
        p.save()
        font = QFont(self.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() + 1)
        p.setFont(font)
        r = rect.adjusted(18, 18, -18, -18)
        flags = int(Qt.AlignCenter | Qt.TextWordWrap)
        bound = p.boundingRect(r, flags, text)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 175))
        p.drawRoundedRect(bound.adjusted(-14, -10, 14, 10), 6, 6)
        p.setPen(QColor(theme.WARN if warn else theme.TEXT))
        p.drawText(r, flags, text)
        p.restore()

    def _paint_reframe(self, p: QPainter) -> None:
        sr = self.source_rect()
        if sr is None:
            rect = fit_rect(16, 9, self._area())
            p.fillRect(rect, QColor("#15171a"))
            if self.message:
                self._draw_message(p, rect, self.message, self.warn)
            return
        p.drawImage(sr, self.source)
        box = self.box_rect()
        if box is None or not self.editable:
            p.fillRect(sr, QColor(0, 0, 0, 150))
            note = self.rf_note or self.message
            if note:
                self._draw_message(p, sr, note, bool(self.warn and not self.rf_note))
            return
        outer, inner = QPainterPath(), QPainterPath()
        outer.addRect(sr)
        inner.addRect(box)
        p.fillPath(outer.subtracted(inner), QColor(0, 0, 0, 150))
        p.setPen(QPen(QColor(255, 255, 255, 70), 1))                 # rule-of-thirds inside the window
        for i in (1, 2):
            p.drawLine(QPointF(box.left() + box.width() * i / 3, box.top()),
                       QPointF(box.left() + box.width() * i / 3, box.bottom()))
            p.drawLine(QPointF(box.left(), box.top() + box.height() * i / 3),
                       QPointF(box.right(), box.top() + box.height() * i / 3))
        if self.keys:                                                # the camera path made of the keys
            pts = [QPointF(sr.x() + kx * sr.width(), sr.y() + ky * sr.height()) for kx, ky in self.keys]
            p.setRenderHint(QPainter.Antialiasing)
            if len(pts) > 1:
                p.setPen(QPen(QColor(theme.WARN), 1.5, Qt.DotLine))
                p.drawPolyline(QPolygonF(pts))
            for i, pt in enumerate(pts):
                p.setPen(QPen(QColor(0, 0, 0, 210), 1))
                p.setBrush(QColor("#ffffff" if i == self.key_here else theme.WARN))
                p.drawEllipse(pt, 4.5, 4.5)
            p.setRenderHint(QPainter.Antialiasing, False)
        p.setPen(QPen(QColor(theme.ACCENT), 2))
        p.setBrush(Qt.NoBrush)
        p.drawRect(box)
        s = sr.width() / self.box.dw
        p.setPen(QPen(QColor("#0b1620"), 1))
        p.setBrush(QColor(theme.ACCENT))
        h = self.HANDLE
        for cx, cy in self.box.corner_points().values():
            p.drawRect(QRectF(sr.x() + cx * s - h, sr.y() + cy * s - h, 2 * h, 2 * h))

    # ---- mouse / wheel / keys (reframe mode only)
    def _hit(self, pos: QPointF) -> Optional[str]:
        sr, box = self.source_rect(), self.box_rect()
        if sr is None or box is None or not self.editable:
            return None
        s = sr.width() / self.box.dw
        for name, (cx, cy) in self.box.corner_points().items():
            if abs(pos.x() - (sr.x() + cx * s)) <= self.HANDLE + 3 and abs(pos.y() - (sr.y() + cy * s)) <= self.HANDLE + 3:
                return name
        if box.contains(pos):
            return "move"
        return "outside" if sr.contains(pos) else None

    def mousePressEvent(self, ev) -> None:                          # noqa: N802
        if not (self.rf_active and ev.button() == Qt.LeftButton):
            return super().mousePressEvent(ev)
        self.setFocus()
        hit = self._hit(ev.position())
        sr = self.source_rect()
        if hit is None or sr is None:
            return
        px, py = self._to_display(ev.position(), sr)
        if hit in ("move", "outside"):
            if hit == "outside":
                self.box.move_to(px, py)                              # click outside: jump the window here
            self._drag = {"kind": "move", "press": (px, py), "centre": self.box.centre_px()}
        else:
            self._drag = {"kind": "corner", "corner": hit, "anchor": self.box.anchor_for(hit)}
        self.reframeChanged.emit()
        self.update()

    def mouseMoveEvent(self, ev) -> None:                           # noqa: N802
        if not self.rf_active:
            return super().mouseMoveEvent(ev)
        sr = self.source_rect()
        if self._drag is None or sr is None or self.box is None:
            self._hover(ev.position())
            return
        px, py = self._to_display(ev.position(), sr)
        d = self._drag
        if d["kind"] == "move":
            self.box.move_to(d["centre"][0] + px - d["press"][0], d["centre"][1] + py - d["press"][1])
        else:
            self.box.resize_corner(d["corner"], px, py, d["anchor"])
        self.reframeChanged.emit()
        self.update()

    def mouseReleaseEvent(self, ev) -> None:                        # noqa: N802
        if self.rf_active and ev.button() == Qt.LeftButton and self._drag is not None:
            self._drag = None
            self.reframeReleased.emit()
            self.update()
            return
        super().mouseReleaseEvent(ev)

    def _hover(self, pos: QPointF) -> None:
        hit = self._hit(pos)
        cur = {"move": Qt.SizeAllCursor, "outside": Qt.CrossCursor, "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
               "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor}.get(hit or "", Qt.ArrowCursor)
        self.setCursor(cur)

    def wheelEvent(self, ev) -> None:                               # noqa: N802
        steps = ev.angleDelta().y() / 120.0
        if not (self.rf_active and self.editable and self.box is not None) or steps == 0:
            return super().wheelEvent(ev)
        self.box.set_zoom(self.box.zoom * (1.12 ** steps))             # wheel up = zoom in (smaller window)
        self.reframeChanged.emit()
        self.reframeNudged.emit()
        self.update()
        ev.accept()

    def keyPressEvent(self, ev) -> None:                            # noqa: N802
        if not self.rf_active:
            return super().keyPressEvent(ev)
        k = ev.key()
        if k == Qt.Key_Escape:
            self.escapePressed.emit()
            return
        if not (self.editable and self.box is not None):
            return super().keyPressEvent(ev)
        step = 0.05 if ev.modifiers() & Qt.ShiftModifier else 0.005
        moves = {Qt.Key_Left: (-step, 0.0), Qt.Key_Right: (step, 0.0), Qt.Key_Up: (0.0, -step), Qt.Key_Down: (0.0, step)}
        if k in moves:
            self.box.nudge(*moves[k])
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.box.set_zoom(self.box.zoom * 1.05)
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self.box.set_zoom(self.box.zoom / 1.05)
        else:
            return super().keyPressEvent(ev)
        self.reframeChanged.emit()
        self.reframeNudged.emit()
        self.update()
        ev.accept()


class _OutputInset(QWidget):
    """Small picture of what will really be rendered (the composite), shown beside the source frame in reframe mode.
    While the crop window is being dragged it shows the crop of the source frame instead, so it follows the mouse."""

    WIDTH = 160

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.frame: Optional[QImage] = None
        self.source: Optional[QImage] = None
        self.live: Optional[QRectF] = None              # crop window in source-image pixels (while editing)
        self.out_size = (9, 16)
        self.background = "#000000"
        self.info: list[str] = []

    def canvas_rect(self) -> QRectF:
        info_h = 17 * len(self.info) + 8 if self.info else 0
        area = QRectF(6, 24, self.width() - 12, max(10, self.height() - 24 - info_h - 4))
        rect = fit_rect(self.out_size[0], self.out_size[1], area)
        rect.moveTop(area.top())                                     # top-aligned under the title
        return rect

    def paintEvent(self, _ev) -> None:                              # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        p.setPen(QColor(theme.TEXT_DIM))
        p.drawText(QRectF(8, 4, self.width() - 16, 18), int(Qt.AlignLeft | Qt.AlignVCenter), "實際輸出")
        rect = self.canvas_rect()
        if self.live is not None and self.source is not None:
            p.drawImage(rect, self.source, self.live)
        elif self.frame is not None:
            p.drawImage(rect, self.frame)
        else:
            p.fillRect(rect, QColor(self.background))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect.adjusted(-0.5, -0.5, 0.5, 0.5))
        p.setPen(QColor(theme.TEXT))
        y = rect.bottom() + 8
        for line in self.info:
            p.drawText(QRectF(8, y, self.width() - 12, 17), int(Qt.AlignLeft | Qt.AlignVCenter), line)
            y += 17
        p.end()


# ============================================================================ the monitor
class _Ctx(NamedTuple):
    seq: Sequence
    track: object
    clip: object
    asset: object


class MonitorWidget(QWidget):
    """Picture + transport bar + preview-cache controls + reframe mode (see the module docstring).

    ``MonitorWidget(state, playback)``; ``set_banner_widget(w)`` puts a widget (the import suggestion) above the
    picture; ``render_preview()`` renders the preview file with sound (Ctrl+Enter).
    """

    reframeModeChanged = Signal(bool)

    def __init__(self, state: AppState, playback: Optional[PlaybackController] = None,
                 parent: Optional[QWidget] = None, max_side: int = MAX_SIDE) -> None:
        super().__init__(parent)
        self.state = state
        self.playback = playback if playback is not None else PlaybackController(state, self)
        self._epoch = 0                    # bumped on every project change; frames of older epochs are dropped
        self._serial = 0
        self._done_serial = 0
        self._stopped = False
        self._rf_clip = ""
        self._rf_dirty = False
        self._rf_base = (0.5, 0.5, 1.0)    # (cx, cy, zoom) as stored in the project at the playhead
        self._auto_pref = True
        self._syncing = False
        self._runner: Optional[TaskRunner] = None
        self._render_pct = 0.0
        self.frames_shown = 0
        self.last_decode_ms = 0.0
        self._bridge = _Bridge(self)
        self.compositor = LiveCompositor(state.project, max_side)
        self._worker = _FrameWorker(self.compositor, self._bridge)
        weakref.finalize(self, _stop_worker, self._worker)
        self._bridge.ready.connect(self._on_result)
        self._nudge_timer = QTimer(self)
        self._nudge_timer.setSingleShot(True)
        self._nudge_timer.setInterval(NUDGE_COMMIT_MS)
        self._nudge_timer.timeout.connect(self._commit_reframe)
        self._build_ui()
        self._connect()
        self._update_info()
        self._update_reframe_button()
        self._update_cache_label()
        self._on_playing(self.playback.is_playing)
        self.tc.set_time(state.playhead)
        self._request()

    # ====================================================================== construction
    def _tool(self, text: str, tip: str, checkable: bool = False) -> QToolButton:
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setFocusPolicy(Qt.NoFocus)
        return b

    def _icon_button(self, kind: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setIcon(icon(kind))
        b.setIconSize(QSize(20, 20))
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.NoFocus)
        b.setFixedSize(36, 30)
        return b

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._lay = lay

        top = QHBoxLayout()
        top.setContentsMargins(6, 0, 6, 0)
        top.setSpacing(6)
        self.btn_reframe = self._tool("取景", "重新取景", True)
        self.btn_guides = self._tool("安全區", "顯示安全區（90%／80%）與中心線", True)
        self.btn_caps = self._tool("字幕", "在畫面上疊加字幕（近似預覽，輸出以渲染為準）", True)
        self.btn_caps.setChecked(True)
        self.lbl_info = QLabel()
        self.lbl_info.setObjectName("dim")
        self.lbl_info.setMinimumWidth(0)
        self.lbl_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        for w in (self.btn_reframe, self.btn_guides, self.btn_caps):
            top.addWidget(w)
        top.addWidget(self.lbl_info, 1)
        lay.addLayout(top)

        self.reframe_bar = QWidget()
        rbv = QVBoxLayout(self.reframe_bar)
        rbv.setContentsMargins(6, 0, 6, 0)
        rbv.setSpacing(4)
        self._rb1, self._rb2 = QHBoxLayout(), QHBoxLayout()
        for row in (self._rb1, self._rb2):
            row.setSpacing(6)
            rbv.addLayout(row)
        self.chk_auto = QCheckBox("自動關鍵格")
        self.chk_auto.setChecked(True)
        self.chk_auto.setFocusPolicy(Qt.NoFocus)
        self.chk_auto.setStyleSheet(
            f"QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {theme.TEXT_DIM}; border-radius: 3px;"
            f" background: {theme.PANEL_ALT}; }} QCheckBox::indicator:checked {{ background: {theme.ACCENT};"
            f" border-color: {theme.ACCENT}; }} QCheckBox::indicator:disabled {{ border-color: {theme.BORDER}; }}")
        self.btn_prev_key = self._tool("上一個關鍵格", "跳到上一個關鍵格")
        self.btn_next_key = self._tool("下一個關鍵格", "跳到下一個關鍵格")
        self.btn_del_key = self._tool("刪除此關鍵格", "刪除播放頭所在的關鍵格")
        self.btn_clear_keys = self._tool("清除全部", "清除這個片段的所有關鍵格（取景框回到固定位置）")
        self.btn_done = self._tool("完成", "離開取景模式（Esc）")
        self.reframe_bar.hide()
        lay.addWidget(self.reframe_bar)

        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        row.setSpacing(6)
        self.view = _PictureView(self.state)
        self.inset = _OutputInset()
        self.inset.hide()
        row.addWidget(self.view, 1)
        row.addWidget(self.inset)
        lay.addLayout(row, 1)

        self.btn_first = self._icon_button("first", "跳到開頭（Home）")
        self.btn_back = self._icon_button("back", "上一格（←）")
        self.btn_play = self._icon_button("play", "播放／暫停（空白鍵）")
        self.btn_fwd = self._icon_button("fwd", "下一格（→）")
        self.btn_last = self._icon_button("last", "跳到結尾（End）")
        self.tc = TimecodeEdit(self.state.fps if self._has_sequence() else 30.0)
        self.tc.setFocusPolicy(Qt.ClickFocus)              # a fresh window must not open with the box in edit mode
        self.btn_loop = self._tool("循環", "循環播放：到結尾就回到開頭", True)
        self.btn_loop.setIcon(icon("loop"))
        self.btn_range = self._tool("入出點播放", "只播放入點到出點之間（在時間軸上按 I、O 設定入點、出點）")
        self.btn_range.setIcon(icon("range"))
        for b in (self.btn_loop, self.btn_range):
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setIconSize(QSize(18, 18))
        self.transport_a = QWidget()                        # buttons, timecode, loop, range
        ta = QHBoxLayout(self.transport_a)
        ta.setContentsMargins(0, 0, 0, 0)
        ta.setSpacing(4)
        for w in (self.btn_first, self.btn_back, self.btn_play, self.btn_fwd, self.btn_last):
            ta.addWidget(w)
        ta.addSpacing(6)
        ta.addWidget(self.tc)
        ta.addSpacing(6)
        ta.addWidget(self.btn_loop)
        ta.addWidget(self.btn_range)
        self.lbl_cache = QLabel()
        self.lbl_cache.setMinimumWidth(self.lbl_cache.fontMetrics().horizontalAdvance("預覽快取：渲染中 100%") + 4)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(96)
        self.progress.setFixedHeight(16)
        self.progress.setRange(0, 100)
        self.progress.hide()
        self.btn_render = QPushButton("渲染預覽（含聲音）")
        self.btn_render.setFocusPolicy(Qt.NoFocus)
        self.btn_render.setToolTip("把目前序列渲染成低解析度預覽檔，之後播放就有聲音（Ctrl+Enter）")
        self.cache_group = QWidget()                        # preview-cache status + render button
        cg = QHBoxLayout(self.cache_group)
        cg.setContentsMargins(0, 0, 0, 0)
        cg.setSpacing(6)
        cg.addWidget(self.lbl_cache)
        cg.addWidget(self.progress)
        cg.addWidget(self.btn_render)
        self._tr1, self._tr2 = QHBoxLayout(), QHBoxLayout()
        self._tr1.setContentsMargins(6, 0, 6, 0)
        self._tr2.setContentsMargins(6, 0, 6, 0)
        tr_box = QVBoxLayout()
        tr_box.setContentsMargins(0, 0, 0, 6)
        tr_box.setSpacing(4)
        tr_box.addLayout(self._tr1)
        tr_box.addLayout(self._tr2)
        lay.addLayout(tr_box)
        for w in (self.transport_a, self.cache_group, self.reframe_bar):
            w.setMinimumWidth(1)                            # rows regroup (see _relayout) instead of forcing a minimum
        self.setMinimumWidth(MIN_WIDTH)
        self._tier: Optional[tuple] = None
        self._info_text = ("", "")
        self._relayout()

        sc = QShortcut(QKeySequence(Qt.Key_Escape), self)
        sc.setContext(Qt.WidgetWithChildrenShortcut)
        sc.activated.connect(self._on_escape)

    def _connect(self) -> None:
        st, pb, v = self.state, self.playback, self.view
        st.playhead_changed.connect(self._on_playhead)
        st.edited.connect(self._on_edited)
        st.sequence_changed.connect(self._on_sequence)
        st.project_replaced.connect(self._on_project)
        st.selection_changed.connect(self._on_selection)
        st.reframe_requested.connect(self.enter_reframe)
        pb.playing_changed.connect(self._on_playing)
        pb.loop_changed.connect(self._on_loop_changed)
        pb.preview_status_changed.connect(lambda *_: self._update_cache_label())
        pb.audio_changed.connect(lambda *_: self._update_cache_label())
        self.btn_first.clicked.connect(lambda *_: pb.goto_start())
        self.btn_back.clicked.connect(lambda *_: pb.step(-1))
        self.btn_play.clicked.connect(lambda *_: pb.toggle())
        self.btn_fwd.clicked.connect(lambda *_: pb.step(1))
        self.btn_last.clicked.connect(lambda *_: pb.goto_end())
        self.tc.timeCommitted.connect(pb.seek)
        self.btn_loop.toggled.connect(pb.set_loop)
        self.btn_range.clicked.connect(lambda *_: self._on_range_clicked())
        self.btn_render.clicked.connect(lambda *_: self._on_render_clicked())
        self.btn_guides.toggled.connect(self._on_guides)
        self.btn_caps.toggled.connect(self._on_captions)
        self.btn_reframe.toggled.connect(self._on_reframe_toggled)
        v.reframeChanged.connect(self._on_rf_changed)
        v.reframeReleased.connect(self._on_rf_released)
        v.reframeNudged.connect(lambda: self._nudge_timer.start())
        v.escapePressed.connect(self._on_escape)
        self.chk_auto.toggled.connect(self._on_auto_toggled)
        self.btn_prev_key.clicked.connect(lambda *_: self.goto_key(-1))
        self.btn_next_key.clicked.connect(lambda *_: self.goto_key(1))
        self.btn_del_key.clicked.connect(lambda *_: self.delete_key_here())
        self.btn_clear_keys.clicked.connect(lambda *_: self.clear_keys())
        self.btn_done.clicked.connect(lambda *_: self.leave_reframe())

    # ====================================================================== public API
    def set_banner_widget(self, widget: QWidget) -> None:
        """Show ``widget`` (the import suggestion banner) above the picture."""
        self._lay.insertWidget(0, widget)

    @property
    def reframing(self) -> bool:
        return bool(self._rf_clip)

    @property
    def reframe_clip_id(self) -> str:
        return self._rf_clip

    def preview_status(self) -> str:
        return self.playback.preview_status()

    def is_idle(self) -> bool:
        """True when the newest frame request has been answered (tests / screenshots wait for this)."""
        return self._done_serial >= self._serial

    def wait_idle(self, timeout: float = 8.0) -> bool:
        """Process events until the newest frame request is answered (test helper; never call it from a slot)."""
        from PySide6.QtWidgets import QApplication
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            QApplication.processEvents()
            if self.is_idle():
                QApplication.processEvents()
                return True
            time.sleep(0.003)
        return False

    def flush_pending(self) -> None:
        """Store a reframe edit that is still waiting for its commit (wheel / keyboard)."""
        self._nudge_timer.stop()
        self._commit_reframe()

    def shutdown(self) -> None:
        """Stop the decode thread and any preview render (window closing)."""
        self._stopped = True
        self._nudge_timer.stop()
        if self._runner is not None and self._runner.running:
            self._runner.cancel()
        self._worker.stop()

    def closeEvent(self, ev) -> None:                               # noqa: N802
        self.shutdown()
        super().closeEvent(ev)

    def showEvent(self, ev) -> None:                                # noqa: N802
        super().showEvent(ev)
        self._request()

    def resizeEvent(self, ev) -> None:                              # noqa: N802
        super().resizeEvent(ev)
        self._relayout()

    def sizeHint(self) -> QSize:                                    # noqa: N802
        return QSize(720, 540)

    # ====================================================================== state helpers
    def _has_sequence(self) -> bool:
        try:
            self.state.sequence()
            return True
        except KeyError:
            return False

    def _clip_ctx(self, clip_id: str) -> Optional[_Ctx]:
        if not clip_id:
            return None
        try:
            seq = self.state.sequence()
        except KeyError:
            return None
        tr, clip = seq.find_clip(clip_id)
        if clip is None:
            return None
        return _Ctx(seq, tr, clip, self.state.project.assets.get(clip.asset_id))

    def _reframe_candidate(self) -> str:
        """Id of the one selected ``cover`` video clip ('' when the selection does not qualify)."""
        try:
            sel = self.state.selected_clips()
        except KeyError:
            return ""
        if len(sel) != 1:
            return ""
        ctx = self._clip_ctx(sel[0].id)
        if ctx is None or ctx.track is None or ctx.track.kind != "video" or ctx.clip.transform.fit != "cover":
            return ""
        a = ctx.asset
        return ctx.clip.id if a is not None and a.kind != "audio" and a.has_video else ""

    # ====================================================================== frame requests / results
    def _request(self) -> None:
        if self._stopped:
            return
        st = self.state
        try:
            seq = st.sequence()
        except KeyError:
            self.view.image = None
            self.view.message, self.view.warn = "專案沒有任何序列", True
            self.view.update()
            return
        self.view.out_size, self.view.background = (seq.width, seq.height), seq.background
        msg = empty_message(seq)
        if msg:
            self.view.image, self.view.source = None, None
            self.view.message, self.view.warn = msg, False
            self.inset.frame = None
            self.view.update()
            self.inset.update()
            return
        fps = seq.fps or 30.0
        t = st.playhead
        if seq.duration > 0 and t >= seq.duration - EPS:               # at / past the end: hold the last frame
            t = max(0.0, seq.duration - 1.0 / fps)
        self._serial += 1
        req = FrameRequest(self._serial, t, seq.id, st.project, self._epoch)
        if self._rf_clip:
            ctx = self._clip_ctx(self._rf_clip)
            if ctx is not None:
                req.source_clip = self._rf_clip
                req.source_t_local = clip_local_time(seq, ctx.clip, st.playhead)
        self._worker.submit(req)

    def _on_result(self, res: FrameResult) -> None:
        self._done_serial = max(self._done_serial, res.serial)
        if res.epoch != self._epoch or self._stopped:
            return                         # computed before the latest edit; a newer request is already queued
        v = self.view
        if res.frame is not None:
            v.image, v.shown_t = res.frame, res.t
        message, warn = "", False
        if res.error:
            message, warn = f"無法顯示畫面：{res.error}", True
        elif res.missing:
            names = "、".join(res.missing[:3]) + ("…" if len(res.missing) > 3 else "")
            message, warn = f"找不到或讀不到素材檔：{names}\n請到素材庫檢查檔案位置，或重新連結", True
        if self._rf_clip:
            v.source = res.source
            if res.source is None and res.source_error:
                message, warn = res.source_error, True
        v.message, v.warn = message, warn
        self.inset.frame = res.frame
        if not v.dragging and not self._rf_dirty:
            self.inset.live = None
        self.frames_shown += 1
        self.last_decode_ms = res.ms
        if self._rf_clip:
            self._refresh_reframe()
        v.update()
        self.inset.update()

    def _on_playhead(self, t: float) -> None:
        self.tc.set_time(t)
        if self._rf_clip:
            self._refresh_reframe()
        self._request()

    def _on_edited(self, _kind: str = "", _label: str = "") -> None:
        self._epoch += 1
        self._update_info()
        self._update_reframe_button()
        if self._rf_clip:
            self._refresh_reframe()
        self._request()

    def _on_sequence(self) -> None:
        self._epoch += 1
        self.leave_reframe()
        self.view.image, self.view.source = None, None
        self.view.message = ""
        self._update_info()
        self._update_reframe_button()
        self.tc.set_time(self.state.playhead)
        self._request()

    def _on_project(self) -> None:
        self._on_sequence()

    def _on_selection(self) -> None:
        self._update_reframe_button()
        if not self._rf_clip:
            return
        cand = self._reframe_candidate()
        if not cand:
            self.leave_reframe()
        elif cand != self._rf_clip:                                  # another cover clip got selected: follow it
            self.flush_pending()
            self._rf_clip = cand
            self._place_playhead_in_clip()
            self._refresh_reframe()
            self._request()

    def _update_info(self) -> None:
        try:
            seq = self.state.sequence()
        except KeyError:
            self.lbl_info.setText("沒有序列")
            return
        self.tc.set_fps(seq.fps)
        self.view.out_size, self.view.background = (seq.width, seq.height), seq.background
        self.inset.out_size, self.inset.background = (seq.width, seq.height), seq.background
        self._info_text = (f"{seq.width}×{seq.height}（{seq.aspect_label}）· {seq.fps:g} fps · "
                           f"總長 {fmt_timecode(seq.duration, seq.fps)}", f"{seq.width}×{seq.height} · {seq.fps:g} fps")
        self._relayout()

    # ====================================================================== transport
    def _on_playing(self, playing: bool) -> None:
        self.btn_play.setIcon(icon("pause" if playing else "play"))

    def _on_loop_changed(self, on: bool) -> None:
        self.btn_loop.blockSignals(True)
        self.btn_loop.setChecked(on)
        self.btn_loop.blockSignals(False)

    def _on_range_clicked(self) -> None:
        pb = self.playback
        if pb.in_out_mode:
            pb.pause()
        else:
            pb.pause()
            pb.play_in_out()

    def _on_guides(self, on: bool) -> None:
        self.view.guides = on
        self.view.update()

    def _on_captions(self, on: bool) -> None:
        self.view.captions = on
        self.view.update()

    # ====================================================================== preview cache
    _CACHE_WORDS = {PREVIEW_FRESH: ("最新", theme.OK), PREVIEW_STALE: ("過期", theme.WARN),
                    PREVIEW_RENDERING: ("渲染中", theme.ACCENT), PREVIEW_NONE: ("無", theme.TEXT_DIM)}
    _CACHE_TIPS = {
        PREVIEW_FRESH: "預覽檔是用目前的專案內容渲染的：播放時會有聲音。",
        PREVIEW_STALE: "專案在渲染預覽之後又改過，預覽檔已過期：播放沒有聲音。按「渲染預覽（含聲音）」更新。",
        PREVIEW_RENDERING: "正在渲染預覽檔…",
        PREVIEW_NONE: "還沒有預覽檔：即時播放沒有聲音。按「渲染預覽（含聲音）」建立。",
    }

    def _update_cache_label(self) -> None:
        s = self.playback.preview_status()
        word, color = self._CACHE_WORDS[s]
        if s == PREVIEW_RENDERING and self._render_pct > 0:
            word += f" {int(self._render_pct * 100)}%"
        self.lbl_cache.setText("預覽快取：" + word)
        self.lbl_cache.setStyleSheet(f"color: {color};")
        tip = self._CACHE_TIPS[s]
        if s == PREVIEW_FRESH and self.playback.is_playing and not self.playback.audio_active:
            tip += "（這台電腦目前無法播放聲音，已改為靜音。）"
        self.lbl_cache.setToolTip(tip)

    @staticmethod
    def _refill(layout, items) -> None:
        """Put ``items`` (widgets, or None for a stretch) into ``layout``, replacing what was there."""
        while layout.count():
            layout.takeAt(0)
        for it in items:
            layout.addStretch(1) if it is None else layout.addWidget(it)

    def _relayout(self) -> None:
        """Responsive rows: the preview-cache group and the second group of reframe buttons drop to their own line,
        and text labels shrink, when the monitor gets narrow (the dock layout can make it quite narrow)."""
        w = self.width()
        self.lbl_info.setText(self._info_text[0] if w >= SHORT_INFO_WIDTH else self._info_text[1])
        self.lbl_info.setVisible(w >= 360)
        need = self.transport_a.sizeHint().width() + self.cache_group.sizeHint().width() + 28
        wide = w >= need
        tier = (wide, w >= ICON_ONLY_WIDTH, w >= 600)
        if tier == self._tier:
            return
        self._tier = tier
        self._refill(self._tr1, [self.transport_a, None] + ([self.cache_group] if wide else []))
        self._refill(self._tr2, [] if wide else [None, self.cache_group])
        style = Qt.ToolButtonTextBesideIcon if w >= ICON_ONLY_WIDTH else Qt.ToolButtonIconOnly
        for b in (self.btn_loop, self.btn_range):
            b.setToolButtonStyle(style)
        rb_wide = tier[2]
        self._refill(self._rb1, [self.chk_auto, self.btn_prev_key, self.btn_next_key]
                     + ([self.btn_del_key, self.btn_clear_keys] if rb_wide else []) + [None, self.btn_done])
        self._refill(self._rb2, [] if rb_wide else [self.btn_del_key, self.btn_clear_keys, None])

    def _on_render_clicked(self) -> None:
        if self._runner is not None and self._runner.running:
            self._runner.cancel()
            self.state.status.emit("正在取消預覽渲染…")
        else:
            self.render_preview()

    def render_preview(self) -> bool:
        """Render the active sequence to the preview cache file (low resolution, WITH sound) in the background."""
        if self._runner is not None and self._runner.running:
            self.state.status.emit("預覽正在渲染中…（按「取消渲染」可以中止）")
            return False
        try:
            seq = self.state.sequence()
        except KeyError:
            self.state.status.emit("專案沒有任何序列，沒有東西可以渲染")
            return False
        if seq.duration <= 0:
            self.state.status.emit("這個序列是空的，沒有東西可以渲染預覽")
            return False
        proj = self.state.project
        snapshot = Project.from_dict(proj.to_dict())             # the worker must not see later edits
        revision, pid, sid = proj.revision, proj.id, seq.id

        def job(progress, cancel):
            from ..render import runner as render_runner
            from ..render.compiler import RenderError
            try:
                return render_runner.render_preview(snapshot, snapshot.sequence(sid), progress, cancel)
            except RenderError as exc:                             # already 繁中; keep the text as it is
                raise ValueError(str(exc)) from exc

        runner = TaskRunner(job, self, "render-preview")
        runner.progress.connect(self._on_render_progress)
        runner.finished.connect(lambda res, r=revision, p=pid, s=sid: self._on_render_done(res, r, p, s))
        runner.failed.connect(self._on_render_failed)
        runner.cancelled.connect(self._on_render_cancelled)
        runner.done.connect(self._on_render_end)
        self._runner = runner
        self._render_pct = 0.0
        self.playback.set_preview_rendering(True)
        self.btn_render.setText("取消渲染")
        self.progress.setValue(0)
        self.progress.show()
        self._update_cache_label()
        self.state.status.emit("開始渲染預覽…")
        runner.start()
        return True

    def _on_render_progress(self, frac: float, message: str) -> None:
        self._render_pct = frac
        self.progress.setValue(int(frac * 100))
        self.progress.setToolTip(message)
        self._update_cache_label()

    def _on_render_done(self, result, revision: int, pid: str, sid: str) -> None:
        self.playback.note_preview_rendered(result.path, revision, pid, sid, getattr(result, "duration", 0.0))
        stale = revision != self.state.project.revision
        msg = f"預覽已渲染完成（{result.duration:.1f} 秒內容，用了 {result.elapsed:.1f} 秒）"
        msg += "；但專案之後又改過，預覽已過期" if stale else "；之後播放會有聲音"
        if getattr(result, "warnings", None):
            msg += "。注意：" + "；".join(result.warnings)
        self.state.status.emit(msg)

    def _on_render_failed(self, message: str) -> None:
        self.playback.set_preview_rendering(False)
        self.state.status.emit(f"預覽渲染失敗：{message}")

    def _on_render_cancelled(self) -> None:
        self.playback.set_preview_rendering(False)
        self.state.status.emit("已取消預覽渲染")

    def _on_render_end(self) -> None:
        self.progress.hide()
        self._render_pct = 0.0
        self.btn_render.setText("渲染預覽（含聲音）")
        self._update_cache_label()

    # ====================================================================== reframe mode
    def _update_reframe_button(self) -> None:
        cand = self._reframe_candidate()
        self.btn_reframe.setEnabled(bool(self._rf_clip or cand))
        self.btn_reframe.setToolTip(
            "重新取景：顯示完整來源畫面，拖曳取景框調整構圖（Esc 離開）" if (self._rf_clip or cand)
            else "重新取景：先在時間軸選取「一個」填滿（cover）的視訊片段")

    def _set_reframe_button(self, on: bool) -> None:
        self.btn_reframe.blockSignals(True)
        self.btn_reframe.setChecked(on)
        self.btn_reframe.blockSignals(False)

    def _on_reframe_toggled(self, on: bool) -> None:
        if on:
            if not self.enter_reframe():
                self._set_reframe_button(False)
        else:
            self.leave_reframe()

    def _on_escape(self) -> None:
        if self._rf_clip:
            self.leave_reframe()

    def _place_playhead_in_clip(self) -> None:
        ctx = self._clip_ctx(self._rf_clip)
        if ctx is not None and not clip_contains(ctx.seq, ctx.clip, self.state.playhead):
            self.state.set_playhead(ctx.clip.start)

    def enter_reframe(self, clip_id: str = "") -> bool:
        """Open the reframe editor for ``clip_id`` (default: the selected ``cover`` clip)."""
        cid = clip_id or self._reframe_candidate()
        ctx = self._clip_ctx(cid)
        say = self.state.status.emit
        if ctx is None:
            say("取景：找不到這個片段")
            return False
        a = ctx.asset
        if ctx.track is None or ctx.track.kind != "video" or a is None or a.kind == "audio" or not a.has_video:
            say("取景：這個片段沒有畫面可以取景")
            return False
        if ctx.clip.transform.fit != "cover":
            say("取景只能用在「填滿（cover）」的片段，請先在檢查器把填滿方式改成 cover")
            return False
        if self._rf_clip and self._rf_clip != cid:
            self.flush_pending()
        self.playback.pause()
        self._rf_clip = cid
        self._rf_dirty = False
        if self.state.selection != {cid}:
            self.state.select([cid])
        self._place_playhead_in_clip()
        self.view.rf_active = True
        self.view.source, self.view.box, self.view.message = None, None, ""
        self.inset.source, self.inset.live = None, None
        self.reframe_bar.show()
        self.inset.show()
        self._set_reframe_button(True)
        self._refresh_reframe()
        self._request()
        self.view.setFocus()
        self.reframeModeChanged.emit(True)
        return True

    def leave_reframe(self, message: str = "") -> None:
        if not self._rf_clip:
            return
        self.flush_pending()
        self._rf_clip = ""
        self._rf_dirty = False
        v = self.view
        v.rf_active, v.source, v.box, v.keys, v.message = False, None, None, [], ""
        v.unsetCursor()
        self.inset.source, self.inset.live = None, None
        self.reframe_bar.hide()
        self.inset.hide()
        self._set_reframe_button(False)
        self._update_reframe_button()
        self._request()
        v.update()
        self.reframeModeChanged.emit(False)
        if message:
            self.state.status.emit(message)

    def _auto_key_on(self) -> bool:
        return self.chk_auto.isChecked()

    def _on_auto_toggled(self, on: bool) -> None:
        if not self._syncing:
            self._auto_pref = on

    def _key_index_here(self, seq: Sequence, clip, t_local: float) -> int:
        tol = 0.5 / (seq.fps or 30.0) + 1e-6
        for i, k in enumerate(clip.transform.keys):
            if abs(k[0] - t_local) <= tol:
                return i
        return -1

    def _refresh_reframe(self) -> None:
        """Rebuild the crop box from the project at the playhead (the project is the single source of truth)."""
        ctx = self._clip_ctx(self._rf_clip)
        if ctx is None or ctx.clip.transform.fit != "cover" or ctx.asset is None:
            self.leave_reframe("這個片段已經不能取景（被刪除，或填滿方式改了）")
            return
        v = self.view
        if v.dragging:
            return
        seq, clip, asset = ctx.seq, ctx.clip, ctx.asset
        tf = clip.transform
        fps = seq.fps or 30.0
        t = self.state.playhead
        inside = clip_contains(seq, clip, t)
        t_local = clip_local_time(seq, clip, t)
        has_keys = bool(tf.keys)
        # auto-key: forced on while the clip has keys (a static position could not show through them)
        self._syncing = True
        self.chk_auto.setEnabled(not has_keys)
        self.chk_auto.setChecked(True if has_keys else self._auto_pref)
        self._syncing = False
        self.chk_auto.setToolTip(
            "此片段已有關鍵格，每次放開滑鼠都會在目前時間加一個關鍵格。要改成固定取景，請先按「清除全部」。" if has_keys
            else "開啟：每次放開滑鼠，在目前時間加一個關鍵格（取景框會隨時間移動）。關閉：整個片段固定同一個取景框。")
        dw = float(asset.width or (v.source.width() if v.source is not None else 0))
        dh = float(asset.height or (v.source.height() if v.source is not None else 0))
        here = self._key_index_here(seq, clip, t_local)
        tol = 0.5 / fps + 1e-6
        last = max(0, frame_of(clip.duration, fps) - 1) / fps
        self.btn_prev_key.setEnabled(any(k[0] < t_local - tol for k in tf.keys))
        self.btn_next_key.setEnabled(any(min(k[0], last) > t_local + tol for k in tf.keys))
        self.btn_del_key.setEnabled(here >= 0 and inside)
        self.btn_clear_keys.setEnabled(has_keys)
        if dw <= 0 or dh <= 0:
            v.set_reframe(None, [], -1, False, "")
            return
        cx, cy = tf.center_at(t_local)
        box = CropBox(dw, dh, seq.width, seq.height, cx, cy, tf.zoom)
        box.normalise()
        self._rf_base = (*box.centre(), box.zoom)
        keys = [CropBox(dw, dh, seq.width, seq.height, k[1], k[2], tf.zoom).centre() for k in tf.keys]
        note = "" if inside else "播放頭不在這個片段上：請把播放頭移到片段內再取景"
        v.set_reframe(box, keys, here if inside else -1, inside, note)
        self._update_inset_info(box, len(tf.keys), here >= 0)

    def _update_inset_info(self, box: CropBox, n_keys: int, on_key: bool) -> None:
        cx, cy = box.centre()
        self.inset.info = [f"位置 x {cx * 100:.1f}%　y {cy * 100:.1f}%", f"縮放 {box.zoom:.2f}×",
                           f"關鍵格 {n_keys} 個" + ("（此處有）" if on_key else "")]
        self.inset.update()

    def _on_rf_changed(self) -> None:
        """The box is being edited (nothing stored yet): follow it in the inset."""
        v, box = self.view, self.view.box
        if box is None:
            return
        self._rf_dirty = True
        if v.source is not None:
            sx, sy = v.source.width() / box.dw, v.source.height() / box.dh
            x, y, w, h = box.rect()
            self.inset.source, self.inset.live = v.source, QRectF(x * sx, y * sy, w * sx, h * sy)
        keys = self.state.sequence().find_clip(self._rf_clip)[1].transform.keys if self._clip_ctx(self._rf_clip) else []
        self._update_inset_info(box, len(keys), v.key_here >= 0)

    def _on_rf_released(self) -> None:
        self._nudge_timer.stop()
        self._commit_reframe()

    def _commit_reframe(self) -> None:
        """Store the edited box.  Auto-key on: a key at the playhead (+ zoom) as ONE undo step; off: static values."""
        self._nudge_timer.stop()
        if not self._rf_clip or not self._rf_dirty:
            return
        self._rf_dirty = False
        ctx = self._clip_ctx(self._rf_clip)
        box = self.view.box
        if ctx is None or box is None or not clip_contains(ctx.seq, ctx.clip, self.state.playhead):
            self._refresh_reframe()
            return
        seq, clip = ctx.seq, ctx.clip
        t_local = clip_local_time(seq, clip, self.state.playhead)
        cx, cy = box.centre()
        zoom = box.zoom
        bx, by, bz = self._rf_base
        moved = abs(cx - bx) > 1e-4 or abs(cy - by) > 1e-4
        zoomed = abs(zoom - bz) > 1e-4
        cid, sid = clip.id, seq.id
        try:
            if not (moved or zoomed):
                pass
            elif self._auto_key_on():
                if moved and zoomed:
                    self.state.do("取景關鍵格", lambda p: (
                        ops.set_transform_key(p, p.sequence(sid), cid, t_local, cx, cy),
                        ops.set_clip_props(p, p.sequence(sid), cid, zoom=zoom)))
                elif moved:
                    self.state.op("取景關鍵格", "set_transform_key", cid, t_local, cx, cy)
                else:
                    self.state.op("取景縮放", "set_clip_props", cid, zoom=zoom)
            else:
                self.state.op("取景", "set_clip_props", cid, cx=cx, cy=cy, zoom=zoom)
        except ops.OpError:
            pass                                                      # state.op already put the reason on the status line
        self._refresh_reframe()

    def goto_key(self, direction: int) -> bool:
        """Move the playhead to the previous (-1) / next (+1) crop key of the reframed clip."""
        ctx = self._clip_ctx(self._rf_clip)
        if ctx is None:
            return False
        seq, clip = ctx.seq, ctx.clip
        fps = seq.fps or 30.0
        tol = 0.5 / fps + 1e-6
        last = max(0, frame_of(clip.duration, fps) - 1) / fps
        t_local = clip_local_time(seq, clip, self.state.playhead)
        times = [min(k[0], last) for k in clip.transform.keys]
        cands = [k for k in times if (k > t_local + tol if direction > 0 else k < t_local - tol)]
        if not cands:
            return False
        self.state.set_playhead(clip.start + (min(cands) if direction > 0 else max(cands)))
        return True

    def delete_key_here(self) -> bool:
        """Delete the crop key at the playhead (``remove_transform_key``)."""
        ctx = self._clip_ctx(self._rf_clip)
        if ctx is None:
            return False
        idx = self._key_index_here(ctx.seq, ctx.clip, clip_local_time(ctx.seq, ctx.clip, self.state.playhead))
        if idx < 0:
            return False
        try:
            self.state.op("刪除取景關鍵格", "remove_transform_key", ctx.clip.id, idx)
        except ops.OpError:
            return False
        return True

    def clear_keys(self) -> bool:
        """Remove every crop key of the reframed clip (``clear_transform_keys``)."""
        ctx = self._clip_ctx(self._rf_clip)
        if ctx is None or not ctx.clip.transform.keys:
            return False
        try:
            self.state.op("清除取景關鍵格", "clear_transform_keys", ctx.clip.id)
        except ops.OpError:
            return False
        return True
