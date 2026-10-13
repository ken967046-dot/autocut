"""Frame-grid arithmetic and timecode formatting. Times are float seconds."""
from __future__ import annotations

import math
from fractions import Fraction

EPS = 1e-6
# Two times closer than this (0.1 ms) are the same instant for anything that decides "do these clips touch / overlap":
# far below what a frame grid can tell apart, and far above the rounding of a saved, reopened or undone project (all
# floats are kept to 6 decimals there, so a clip end that was exactly equal to another one can differ by a few 1e-6).
TOUCH = 1e-4
# The longest timeline, and the furthest time anything (a clip's end, a caption, a marker) may sit at: 24 hours, the span of
# the HH:MM:SS:FF clock.  Without a limit a typed "1000000000" put a clip 31 years out and every pixel / scroll bar / frame
# count derived from it overflowed a 32-bit number somewhere.
MAX_TIMELINE = 24 * 3600.0

_NTSC = {23.976: (24000, 1001), 29.97: (30000, 1001), 59.94: (60000, 1001), 119.88: (120000, 1001)}


def frame_of(t: float, fps: float) -> int:
    """Nearest frame index for time ``t`` (round half up, stable for exact grid values)."""
    return int(math.floor(t * fps + 0.5 + 1e-9))


def snap_time(t: float, fps: float) -> float:
    return frame_of(t, fps) / fps


def fps_fraction(fps: float) -> Fraction:
    """Exact rational frame rate (29.97 -> 30000/1001)."""
    for k, (n, d) in _NTSC.items():
        if abs(fps - k) < 0.005:
            return Fraction(n, d)
    return Fraction(fps).limit_denominator(1001)


def fmt_clock(t: float, millis: bool = True) -> str:
    """``H:MM:SS.mmm`` (hours omitted when zero)."""
    t = max(0.0, t)
    total_ms = int(round(t * 1000))
    ms = total_ms % 1000
    s_total = total_ms // 1000
    h, rem = divmod(s_total, 3600)
    m, s = divmod(rem, 60)
    base = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return f"{base}.{ms:03d}" if millis else base


def fmt_timecode(t: float, fps: float) -> str:
    """Non-drop-frame ``HH:MM:SS:FF``."""
    nominal = max(1, int(round(fps)))
    frames = frame_of(max(0.0, t), fps)
    secs, ff = divmod(frames, nominal)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"


def parse_timecode(text: str, fps: float = 30.0) -> float:
    """Accepts ``90``, ``1:30``, ``1:30.5``, ``0:01:30``, ``00:01:30:15`` (frames), ``45f``."""
    s = text.strip().lower().replace("；", ":").replace("：", ":")
    if not s:
        raise ValueError("empty timecode")
    if s.endswith("f") and ":" not in s:
        return float(s[:-1]) / fps
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) == 4:
        h, m, sec, ff = (float(p) for p in parts)
        return h * 3600 + m * 60 + sec + ff / max(1, int(round(fps)))
    if len(parts) == 3:
        h, m, sec = (float(p) for p in parts)
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = (float(p) for p in parts)
        return m * 60 + sec
    raise ValueError(f"cannot parse timecode: {text!r}")
