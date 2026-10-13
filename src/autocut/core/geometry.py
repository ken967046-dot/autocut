"""Reframing geometry shared by the ffmpeg compiler and the live preview.

Everything is expressed on the *display* frame of the source (rotation and
pixel-aspect already applied) so that the renderer and the on-screen crop box
always agree.
"""
from __future__ import annotations

from typing import Sequence

from .timecode import EPS

Key = Sequence[float]  # [t_local, cx, cy]


def interp_keys(keys: Sequence[Key], t: float, cx0: float = 0.5, cy0: float = 0.5) -> tuple[float, float]:
    """Piecewise-linear crop centre at clip-local time ``t`` (held outside the key range)."""
    if not keys:
        return cx0, cy0
    if t <= keys[0][0]:
        return float(keys[0][1]), float(keys[0][2])
    if t >= keys[-1][0]:
        return float(keys[-1][1]), float(keys[-1][2])
    lo, hi = 0, len(keys) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if keys[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, x0, y0 = keys[lo]
    t1, x1, y1 = keys[hi]
    a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return x0 + (x1 - x0) * a, y0 + (y1 - y0) * a


def split_keys(keys: Sequence[Key], at: float) -> tuple[list[list[float]], list[list[float]]]:
    """Split a key list at clip-local ``at``; both halves get an interpolated boundary key."""
    if not keys:
        return [], []
    cx, cy = interp_keys(keys, at)
    left = [list(k) for k in keys if k[0] < at - EPS] + [[at, cx, cy]]
    right = [[0.0, cx, cy]] + [[k[0] - at, k[1], k[2]] for k in keys if k[0] > at + EPS]
    return left, right


def shift_keys(keys: Sequence[Key], d: float) -> list[list[float]]:
    """Head of the clip moved by ``d`` seconds (d>0: head trimmed) -> keys move by ``-d``."""
    if not keys:
        return []
    if d <= 0:
        return [[k[0] - d, k[1], k[2]] for k in keys]
    cx, cy = interp_keys(keys, d)
    return [[0.0, cx, cy]] + [[k[0] - d, k[1], k[2]] for k in keys if k[0] > d + EPS]


def limit_keys(keys: Sequence[Key], duration: float) -> list[list[float]]:
    """Drop keys beyond ``duration`` (tail trimmed), keeping an interpolated end key."""
    if not keys:
        return []
    if keys[-1][0] <= duration + EPS:
        return [list(k) for k in keys]
    cx, cy = interp_keys(keys, duration)
    return [list(k) for k in keys if k[0] < duration - EPS] + [[duration, cx, cy]]


def scale_keys(keys: Sequence[Key], factor: float) -> list[list[float]]:
    return [[k[0] * factor, k[1], k[2]] for k in keys]


def _even(v: float) -> int:
    n = int(v)
    return n - (n % 2)


def base_window(dw: float, dh: float, out_w: float, out_h: float) -> tuple[float, float]:
    """Largest window with the output aspect that fits inside the display frame."""
    ar_out = out_w / out_h
    if dw / dh >= ar_out:
        return dh * ar_out, dh
    return dw, dw / ar_out


def crop_window(dw: float, dh: float, out_w: float, out_h: float, zoom: float = 1.0) -> tuple[float, float]:
    bw, bh = base_window(dw, dh, out_w, out_h)
    z = max(1.0, zoom)
    return bw / z, bh / z


def crop_topleft(dw: float, dh: float, cw: float, ch: float, cx: float, cy: float) -> tuple[float, float]:
    """Top-left of a ``cw x ch`` window centred on (cx, cy) (normalised), kept inside the frame."""
    x = min(max(cx * dw - cw / 2.0, 0.0), max(0.0, dw - cw))
    y = min(max(cy * dh - ch / 2.0, 0.0), max(0.0, dh - ch))
    return x, y


def crop_rect_px(dw: int, dh: int, out_w: int, out_h: int, cx: float, cy: float, zoom: float = 1.0
                 ) -> tuple[int, int, int, int]:
    """Integer (x, y, w, h), all even (yuv420-safe)."""
    cw, ch = crop_window(dw, dh, out_w, out_h, zoom)
    cwi, chi = max(2, _even(cw)), max(2, _even(ch))
    cwi, chi = min(cwi, _even(dw)), min(chi, _even(dh))
    x, y = crop_topleft(dw, dh, cwi, chi, cx, cy)
    return _even(x), _even(y), cwi, chi


def contain_rect(dw: float, dh: float, out_w: float, out_h: float) -> tuple[float, float, float, float]:
    """(x, y, w, h) of the whole source scaled to fit inside the output (letterbox)."""
    s = min(out_w / dw, out_h / dh)
    w, h = dw * s, dh * s
    return (out_w - w) / 2.0, (out_h - h) / 2.0, w, h
