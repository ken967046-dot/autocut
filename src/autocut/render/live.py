"""Real-time preview compositor (OpenCV, no Qt, no ffmpeg process).

``LiveCompositor.frame_at(t)`` draws one frame of a sequence the way ``render.compiler`` + ffmpeg would, at
preview resolution (long side <= ``max_side``): track order, cover / contain / blur fitting (cover geometry comes
from :mod:`autocut.core.geometry`, so the on-screen crop box and the renderer agree), animated crop keys, zoom,
``flip_h``, look (an affine approximation of ffmpeg ``eq``), own fades and same-track crossfades (the later clip
fades in over the earlier one), speed, hidden tracks, disabled clips and the background colour in gaps / past the end.

Thread-safety: one *consumer* thread should own an instance (the monitor's decode thread).  Every public method
takes an internal re-entrant lock, so concurrent calls are serialised rather than corrupting the decoder pool, but
they then wait for each other.  ``invalidate()`` never blocks: it only raises a flag that the next call honours.
Returned arrays are fresh and may be modified by the caller; arrays inside the decoder pool never are.
"""
from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from ..core.geometry import base_window, contain_rect, crop_topleft, crop_window, interp_keys
from ..core.project import Asset, Clip, Look, Project, Sequence, ordered_clips
from ..core.timecode import frame_of

MAX_READERS = 6         # open decoders (LRU)
MAX_PER_ASSET = 3       # decoders one asset may occupy (overlapping clips of the same file)
SKIP_GAP = 24           # forward gap (frames) that is decoded-and-dropped instead of seeking
_ROT_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class LiveError(RuntimeError):
    """A source frame could not be produced (missing / unreadable media). The message is Traditional Chinese."""


# ------------------------------------------------------------------------------ pixel helpers
def bgr_of(hex_rgb: str) -> tuple[int, int, int]:
    h = (hex_rgb or "#000000").lstrip("#")
    if len(h) != 6:
        return 0, 0, 0
    try:
        return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)
    except ValueError:
        return 0, 0, 0


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Always returns a new array; area filtering when shrinking."""
    sh, sw = img.shape[:2]
    interp = cv2.INTER_AREA if w * h < sw * sh else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(w)), max(1, int(h))), interpolation=interp)


@lru_cache(maxsize=32)
def _look_lut(b: float, c: float, s: float) -> np.ndarray:
    """(256, 1, 3) float table mapping full-range (Y, Cr, Cb) bytes to (Y', Cr' - 128, Cb' - 128) after ffmpeg ``eq``.
    ``eq`` edits limited-range code values and clips them to 0..255 *before* the colour conversion: contrast pivots
    on 128, brightness adds ``255*b`` (negative offsets come out ~3.5 codes lower), saturation scales chroma."""
    x = np.arange(256, dtype=np.float32)
    y_lim = np.clip(128.0 + c * (16.0 + x * (219.0 / 255.0) - 128.0) + 255.0 * b - (3.5 if b < 0 else 0.0), 0.0, 255.0)
    c_lim = np.clip(128.0 + s * (x - 128.0) * (224.0 / 255.0), 0.0, 255.0)
    lut = np.empty((256, 1, 3), np.float32)
    lut[:, 0, 0] = (y_lim - 16.0) * (255.0 / 219.0)
    lut[:, 0, 1] = lut[:, 0, 2] = (c_lim - 128.0) * (255.0 / 224.0)
    return lut


_YCC_TO_BGR = np.array([[1.0, 0.0, 1.773], [1.0, -0.714, -0.344], [1.0, 1.403, 0.0]], np.float32)   # (Y, Cr, Cb)


def apply_look(img: np.ndarray, look: Look) -> np.ndarray:
    """Brightness / contrast / saturation the way ffmpeg ``eq`` does it (BT.601, limited-range clipping included);
    about 1 ms for a 405x720 frame."""
    lut = _look_lut(round(look.brightness, 4), round(look.contrast, 4), round(look.saturation, 4))
    ycc = cv2.LUT(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), lut)
    return cv2.add(cv2.transform(ycc, _YCC_TO_BGR), (0, 0, 0, 0), dtype=cv2.CV_8U)


def _roi(frame: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    fh, fw = frame.shape[:2]
    x0 = min(max(int(round(x)), 0), fw - 1)
    y0 = min(max(int(round(y)), 0), fh - 1)
    x1 = min(max(int(round(x + w)), x0 + 1), fw)
    y1 = min(max(int(round(y + h)), y0 + 1), fh)
    return frame[y0:y1, x0:x1]


def _crop_rect(dw: int, dh: int, out_w: int, out_h: int, cx: float, cy: float, zoom: float
               ) -> tuple[int, int, int, int]:
    """Integer cover window ``(x, y, w, h)`` in display pixels, exactly as the ffmpeg ``crop`` filter built by the
    compiler places it: size truncated to even, position rounded to the nearest pixel and then made even.
    (``geometry.crop_rect_px`` truncates the position instead, which is up to 2 px off.)"""
    cw, ch = crop_window(dw, dh, out_w, out_h, zoom)
    cwi, chi = min(max(2, int(cw) // 2 * 2), dw // 2 * 2), min(max(2, int(ch) // 2 * 2), dh // 2 * 2)
    x, y = crop_topleft(dw, dh, cwi, chi, cx, cy)
    return round(x) // 2 * 2, round(y) // 2 * 2, cwi, chi


def _contain(frame: np.ndarray, dw: float, dh: float, ow: int, oh: int, canvas: Optional[np.ndarray] = None
             ) -> np.ndarray:
    """The whole picture scaled to fit inside ``ow x oh``, centred on ``canvas`` (black when omitted)."""
    out = np.zeros((oh, ow, 3), np.uint8) if canvas is None else canvas
    rx, ry, rw, rh = contain_rect(dw, dh, ow, oh)
    w, h = max(1, min(ow, int(round(rw)))), max(1, min(oh, int(round(rh))))
    x, y = min(max(int(round(rx)), 0), ow - w), min(max(int(round(ry)), 0), oh - h)
    out[y:y + h, x:x + w] = resize(frame, w, h)
    return out


def _boxblur(img: np.ndarray, radius: int, passes: int = 2) -> np.ndarray:
    """ffmpeg ``boxblur=radius:passes`` on yuv420p: luma uses ``radius``; chroma uses the same number on half-size
    planes, i.e. twice the radius in picture pixels."""
    def blur(plane: np.ndarray, r: int) -> np.ndarray:
        for _ in range(passes):
            plane = cv2.blur(plane, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)
        return plane
    y, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
    return cv2.cvtColor(cv2.merge([blur(y, radius), blur(cr, 2 * radius), blur(cb, 2 * radius)]), cv2.COLOR_YCrCb2BGR)


def _blur_fit(frame: np.ndarray, dw: float, dh: float, ow: int, oh: int) -> np.ndarray:
    """Sharp picture over a darkened, blurred, enlarged copy of itself (compiler ``fit == "blur"``)."""
    sx, sy = frame.shape[1] / dw, frame.shape[0] / dh
    cw, ch = base_window(dw, dh, ow, oh)                      # what "scale to cover, then centre-crop" keeps
    x, y = crop_topleft(dw, dh, cw, ch, 0.5, 0.5)
    bw, bh = max(2, (ow // 8) // 2 * 2), max(2, (oh // 8) // 2 * 2)
    small = resize(_roi(frame, x * sx, y * sy, cw * sx, ch * sy), bw, bh)
    bg = cv2.resize(_boxblur(small, max(2, min(bw, bh) // 6)), (ow, oh), interpolation=cv2.INTER_LINEAR)
    bg = apply_look(bg, Look(brightness=-0.08, contrast=1.0, saturation=0.9))
    return _contain(frame, dw, dh, ow, oh, canvas=bg)


# --------------------------------------------------------------------------------- decoding
def _stamp(path: str) -> tuple[int, int]:
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns
    except OSError:
        return -1, -1


def _short_path(path: str) -> str:
    """Windows 8.3 name, used when OpenCV cannot open a path with non-ASCII characters (not verifiable on Linux)."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))       # type: ignore[attr-defined]
        return buf.value if n else path
    except Exception:                                                              # noqa: BLE001
        return path


def _open_capture(path: str) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened() and os.name == "nt":
        cap = cv2.VideoCapture(_short_path(path), cv2.CAP_FFMPEG)
    return cap if cap.isOpened() else None


class _Reader:
    """One open ``VideoCapture`` with position bookkeeping: next frame -> read(), small forward gap -> grab(), else seek."""

    def __init__(self, asset: Asset, stats: dict, auto_orient: bool) -> None:
        self.asset, self.path, self.stats = asset, asset.path, stats
        self.stamp = _stamp(asset.path)
        self.cap = _open_capture(asset.path)
        self.pos = 0                      # index of the frame the next read() returns
        self.last_idx = -1
        self.last: Optional[np.ndarray] = None
        self.used = 0                     # compositor tick of the last use (LRU)
        self.n: Optional[int] = None      # frame count when known
        self.fps = 30.0
        self.rot = 0                      # clockwise degrees still to apply (manual orientation)
        self._checked = False
        self._auto = auto_orient
        stats["opens"] += 1
        if self.cap is None:
            return
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 480.0 else (asset.fps or 30.0)
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.n = n if n > 0 else None
        self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0 if auto_orient else 0.0)
        if not auto_orient:
            self.rot = asset.rotation % 360

    @property
    def ok(self) -> bool:
        return self.cap is not None

    def cost(self, idx: int) -> Optional[int]:
        """Frames to decode to deliver ``idx`` without seeking (None: needs a seek)."""
        if idx == self.last_idx and self.last is not None:
            return 0
        gap = idx - self.pos
        return gap + 1 if 0 <= gap <= SKIP_GAP else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.last = None

    def _orient(self, frame: np.ndarray) -> np.ndarray:
        a = self.asset
        if not self._checked:
            self._checked = True
            portrait_wanted, portrait_got = a.height > a.width > 0, frame.shape[0] > frame.shape[1]
            if self._auto and a.rotation in (90, 270) and portrait_wanted != portrait_got:
                self.rot = a.rotation                  # this OpenCV did not apply the rotation flag itself
        return cv2.rotate(frame, _ROT_CODES[self.rot]) if self.rot in _ROT_CODES else frame

    def _eof(self) -> None:
        self.n = max(1, self.pos)

    def get(self, idx: int) -> Optional[np.ndarray]:
        """Decoded frame ``idx`` (BGR, display orientation, shared with the pool: do not modify)."""
        if self.cap is None:
            return None
        idx = max(0, idx if self.n is None else min(idx, self.n - 1))
        if idx == self.last_idx and self.last is not None:
            return self.last
        gap = idx - self.pos
        if gap < 0 or gap > SKIP_GAP:
            self.stats["seeks"] += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            self.pos = idx
        else:
            for _ in range(gap):
                if not self.cap.grab():
                    self._eof()
                    return self.last
                self.pos += 1
                self.stats["skips"] += 1
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._eof()
            return self.last
        self.stats["reads"] += 1
        self.pos += 1
        self.last_idx, self.last = idx, self._orient(frame)
        return self.last


# -------------------------------------------------------------------------------- the compositor
@dataclass
class _Layer:
    clip: Clip
    asset: Asset
    k: int          # frame index inside the clip
    dc: int         # clip length in frames
    vfi: int        # alpha fade-in frames (own fade or crossfade)
    vfo: int        # alpha fade-out frames

    def alpha(self) -> float:
        a = 1.0
        if self.vfi > 0 and self.k < self.vfi:
            a *= self.k / self.vfi
        out_start = self.dc - self.vfo
        if self.vfo > 0 and self.k >= out_start:
            a *= (self.dc - self.k) / self.vfo
        return a


class LiveCompositor:
    """Preview renderer for one sequence of ``project``; see the module docstring."""

    def __init__(self, project: Project, max_side: int = 720, auto_orient: bool = True) -> None:
        self.project = project
        self.max_side = max(16, int(max_side))
        self.auto_orient = auto_orient
        self.stats = {"opens": 0, "seeks": 0, "reads": 0, "skips": 0}
        self.missing: set[str] = set()        # asset paths that could not be opened in the latest frames
        self._seq_id = ""
        self._lock = threading.RLock()
        self._readers: list[_Reader] = []
        self._images: dict[str, tuple[tuple[int, int], np.ndarray]] = {}
        self._failed: dict[str, tuple[int, int]] = {}
        self._tick = 0
        self._dirty = False

    # ---- configuration
    def set_sequence(self, seq_id: str) -> None:
        with self._lock:
            self._seq_id = seq_id

    def set_project(self, project: Project) -> None:
        with self._lock:
            self.project = project
            self._seq_id = ""
        self.invalidate()

    def invalidate(self) -> None:
        """The project changed: forget cached images / failures on the next call (non-blocking)."""
        self._dirty = True

    def close(self) -> None:
        with self._lock:
            for r in self._readers:
                r.release()
            self._readers.clear()
            self._images.clear()
            self._failed.clear()

    @property
    def open_readers(self) -> int:
        return len(self._readers)

    def output_size(self, seq_id: Optional[str] = None) -> tuple[int, int]:
        """Size of the frames ``frame_at`` returns (sequence aspect, long side <= ``max_side``)."""
        with self._lock:
            return self._out_size(self._sequence(seq_id))

    # ---- public drawing API
    def frame_at(self, t: float) -> np.ndarray:
        """BGR uint8 frame shown at timeline time ``t`` (the frame whose interval contains ``t``)."""
        with self._lock:
            self._refresh()
            seq = self._sequence()
            ow, oh = self._out_size(seq)
            canvas = np.empty((oh, ow, 3), np.uint8)
            canvas[:] = bgr_of(seq.background)
            f = max(0, int(math.floor(t * seq.fps + 1e-3)))
            self._tick += 1
            self.missing.clear()
            for ly in self._layers_at(seq, f):
                try:
                    img = self._draw_layer(seq, ly, ow, oh)
                except Exception:                                                  # noqa: BLE001  one bad clip must
                    img = None                                                      # not stop playback
                if img is None:
                    continue
                a = ly.alpha()
                canvas = img if a >= 1.0 else cv2.addWeighted(img, a, canvas, 1.0 - a, 0.0)
            return canvas

    def source_frame_at(self, clip_id: str, t_local: float, max_side: Optional[int] = None) -> np.ndarray:
        """The whole, un-cropped source picture of a clip at clip-local time ``t_local`` (display orientation and
        pixel aspect, long side <= ``max_side`` or the compositor's), for crop-window editing."""
        with self._lock:
            self._refresh()
            seq = self._sequence()
            _, clip = seq.find_clip(clip_id)
            asset = self.project.assets.get(clip.asset_id) if clip else None
            if clip is None or asset is None:
                raise LiveError("找不到這個片段或它的素材")
            if asset.kind == "audio" or not asset.has_video:
                raise LiveError(f"「{asset.name}」沒有影像")
            last = max(0, frame_of(clip.duration, seq.fps) - 1)
            k = min(last, max(0, int(math.floor(t_local * seq.fps + 1e-3))))
            self._tick += 1
            frame = self._source(asset, clip.src_in + k / seq.fps * clip.speed)
            if frame is None:
                raise LiveError(f"無法讀取素材「{asset.name}」（檔案遺失或損毀）")
            dw, dh = asset.width or frame.shape[1], asset.height or frame.shape[0]
            s = min(1.0, (max_side or self.max_side) / max(dw, dh))
            return resize(frame, max(2, round(dw * s)), max(2, round(dh * s)))

    # ---- planning (mirrors render.compiler)
    def _sequence(self, seq_id: Optional[str] = None) -> Sequence:
        sid = seq_id or self._seq_id
        try:
            return self.project.sequence(sid or None)
        except KeyError:
            return self.project.sequence()

    def _out_size(self, seq: Sequence) -> tuple[int, int]:
        s = min(1.0, self.max_side / max(seq.width, seq.height, 1))
        return max(2, round(seq.width * s)), max(2, round(seq.height * s))

    def _layers_at(self, seq: Sequence, f: int) -> list[_Layer]:
        """Active video layers at frame ``f``, bottom first (track order, then start)."""
        fps, out = seq.fps, []
        for tr in seq.video_tracks:
            if tr.muted:
                continue
            items: list[tuple[Clip, Asset, int, int]] = []
            for c in ordered_clips(tr.clips):
                a = self.project.assets.get(c.asset_id)
                if not c.enabled or a is None or (not a.has_video and a.kind != "image"):
                    continue
                cs = max(0, frame_of(c.start, fps))
                items.append((c, a, cs, max(cs + 1, frame_of(c.end, fps))))
            for i, (c, a, cs, ce) in enumerate(items):
                if not cs <= f < ce:
                    continue
                xin = 0
                for _, _, _, ace in reversed(items[:i]):          # closest earlier clip that still overlaps
                    if ace > cs:
                        xin = min(ace, ce) - cs
                        break
                out.append(_Layer(c, a, f - cs, ce - cs, max(int(round(c.fade_in * fps)), xin),
                                  int(round(c.fade_out * fps))))
        return out

    # ---- drawing
    def _draw_layer(self, seq: Sequence, ly: _Layer, ow: int, oh: int) -> Optional[np.ndarray]:
        a, c = ly.asset, ly.clip
        frame = self._source(a, c.src_in + ly.k / seq.fps * c.speed)
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        dw, dh = a.width or fw, a.height or fh
        tf = c.transform
        if tf.fit == "contain":
            img = _contain(frame, dw, dh, ow, oh)
        elif tf.fit == "blur":
            img = _blur_fit(frame, dw, dh, ow, oh)
        else:
            z = max(1.0, tf.zoom)
            if abs(dw / dh - seq.width / seq.height) < 0.002 and z <= 1.0 + 1e-9 and not tf.keys:
                x, y, cw, ch = 0, 0, dw, dh                         # compiler "noop": plain scale of the whole frame
            else:
                cx, cy = interp_keys(tf.keys, ly.k / seq.fps, tf.cx, tf.cy) if tf.keys else (tf.cx, tf.cy)
                x, y, cw, ch = _crop_rect(int(dw), int(dh), seq.width, seq.height, cx, cy, z)
            sx, sy = fw / dw, fh / dh
            img = resize(_roi(frame, x * sx, y * sy, cw * sx, ch * sy), ow, oh)
        if tf.flip_h:
            img = cv2.flip(img, 1)
        if not c.look.neutral:
            img = apply_look(img, c.look)
        return img

    # ---- sources
    def _refresh(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        self._images.clear()
        self._failed.clear()
        for r in [r for r in self._readers if _stamp(r.path) != r.stamp]:      # file replaced on disk
            r.release()
            self._readers.remove(r)

    def _source(self, asset: Asset, src_t: float) -> Optional[np.ndarray]:
        """Raw frame (display orientation, shared: do not modify) of ``asset`` at source time ``src_t``."""
        path = asset.path
        if self._failed.get(path) == _stamp(path):
            self.missing.add(path)
            return None
        if asset.kind == "image":
            img = self._image(asset)
        else:
            r = self._reader(asset, max(0, int(round(src_t * (self._fps_of(asset))))))
            img = r.get(max(0, int(round(src_t * r.fps)))) if r is not None else None
        if img is None:
            self._failed[path] = _stamp(path)
            self.missing.add(path)
        return img

    def _fps_of(self, asset: Asset) -> float:
        for r in self._readers:
            if r.path == asset.path and r.ok:
                return r.fps
        return asset.fps or 30.0

    def _image(self, asset: Asset) -> Optional[np.ndarray]:
        stamp = _stamp(asset.path)
        hit = self._images.get(asset.path)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        try:
            raw = cv2.imdecode(np.fromfile(asset.path, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        except (OSError, ValueError):
            raw = None
        if raw is None:
            return None
        if asset.rotation in (90, 270) and asset.height > asset.width > 0 and raw.shape[0] <= raw.shape[1]:
            raw = cv2.rotate(raw, _ROT_CODES[asset.rotation])
        lim = 2 * self.max_side                                    # keep head-room for zoomed crops, not 24 MP
        if max(raw.shape[:2]) > lim:
            s = lim / max(raw.shape[:2])
            raw = resize(raw, round(raw.shape[1] * s), round(raw.shape[0] * s))
        if len(self._images) >= 8:
            self._images.pop(next(iter(self._images)))
        self._images[asset.path] = (stamp, raw)
        return raw

    def _reader(self, asset: Asset, idx: int) -> Optional[_Reader]:
        """The pooled decoder that can deliver ``idx`` most cheaply (opening / repurposing one when none can)."""
        same = [r for r in self._readers if r.path == asset.path]
        best = min((r for r in same if r.cost(idx) is not None), key=lambda r: r.cost(idx), default=None)
        if best is None:
            idle = [r for r in same if r.used != self._tick]
            if len(same) >= MAX_PER_ASSET and idle:
                best = min(idle, key=lambda r: r.used)            # seek the least recently used one
            else:
                best = _Reader(asset, self.stats, self.auto_orient)
                if not best.ok:
                    return None
                self._readers.append(best)
                while len(self._readers) > MAX_READERS:
                    victim = min((r for r in self._readers if r is not best), key=lambda r: r.used)
                    victim.release()
                    self._readers.remove(victim)
        best.used = self._tick
        return best
