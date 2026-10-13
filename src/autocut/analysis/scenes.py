"""Scene cuts from consecutive picture samples, and frame-exact refinement of one cut (spec §2.3).

A cut between samples ``p-1`` and ``p`` needs BOTH the grey 32-bin histogram L1 distance and the mean absolute grey
difference to exceed adaptive thresholds (``max(floor, median + 6 * MAD)`` of that series).  Detections within 1 s of a
stronger one are dropped, and a one-sample blip (camera flash, someone walking past the lens: the picture returns to what
it was) is not a cut.  The reported time is the midpoint of the two samples, so it is only good to +-1/(2*visual_fps);
:func:`refine_cut` decodes a small window at full frame rate to find the exact frame.

Known limits: dissolves and fades spread over several samples are not detected, and two shots with the same brightness
distribution *and* a small pixel difference (a cut between two similar talking-head angles) can be missed.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .audio import FFmpegStream, ffmpeg_input

HIST_BINS = 32
HIST_FLOOR = 0.35        # L1 distance of normalised histograms is 0..2
HIST_CAP = 1.2           # ... the adaptive threshold never exceeds this, so blatant cuts are always found
MAD_FLOOR = 0.07         # mean |grey diff| / 255
MAD_CAP = 0.25
SIGMA_K = 6.0
DEDUP_S = 1.0
TRANSIENT_RATIO = 0.4    # a "cut" whose skip-one distance is below this fraction of it is a blip, not a cut
REFINE_SIZE = 64         # refine_cut decodes 64x64 grey frames (aspect is irrelevant for change detection)
REFINE_MIN_CHANGE = 2.0  # (0..255) the biggest frame-to-frame change must be at least this ...
REFINE_PEAK = 3.0        # ... and this many times the window's median change, else there is no cut to refine


def grey_hist(grey: np.ndarray) -> np.ndarray:
    """Normalised 32-bin histogram (sums to 1) of an 8-bit grey frame."""
    g = np.asarray(grey, dtype=np.uint8).ravel()
    h = np.bincount(g >> 3, minlength=HIST_BINS).astype(np.float32)
    return h / max(float(h.sum()), 1.0)


def adaptive_threshold(values: Sequence[float], floor: float, cap: float, k: float = SIGMA_K) -> float:
    """``clamp(median + k * MAD, floor, cap)`` of a series (robust to the cuts themselves)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float(floor)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return min(float(cap), max(float(floor), med + k * mad))


def detect_cuts(l1: Sequence[float], mad: Sequence[float], fps: float, hists: Optional[np.ndarray] = None
                ) -> list[int]:
    """Indices ``p`` of the cuts (between sample ``p-1`` and ``p``), in time order.

    ``l1[p]`` / ``mad[p]`` describe the step ``p-1 -> p`` (index 0 is a placeholder); ``hists`` (one normalised histogram
    per sample) enables the one-sample-blip rejection.
    """
    n = len(l1)
    if n < 2:
        return []
    a = np.asarray(l1, dtype=np.float64)
    b = np.asarray(mad, dtype=np.float64)
    th_h = adaptive_threshold(a[1:], HIST_FLOOR, HIST_CAP)
    th_m = adaptive_threshold(b[1:], MAD_FLOOR, MAD_CAP)
    cand = [p for p in range(1, n) if a[p] > th_h and b[p] > th_m]
    if hists is not None and len(hists) == n:
        def blip(p: int) -> bool:
            before = p >= 2 and float(np.abs(hists[p] - hists[p - 2]).sum()) < TRANSIENT_RATIO * a[p]
            after = p + 1 < n and float(np.abs(hists[p + 1] - hists[p - 1]).sum()) < TRANSIENT_RATIO * a[p]
            return before or after
        cand = [p for p in cand if not blip(p)]
    score = {p: min(a[p] / th_h, b[p] / th_m) for p in cand}
    kept: list[int] = []
    for p in sorted(cand, key=lambda q: (-score[q], q)):          # strongest first; weaker ones within 1 s are dropped
        if all(abs(p - q) / fps >= DEDUP_S - 1e-9 for q in kept):
            kept.append(p)
    return sorted(kept)


def cut_time(p: int, fps: float) -> float:
    """Time of a cut between samples ``p-1`` and ``p`` = their midpoint."""
    return round((p - 0.5) / fps, 3)


class SceneDetector:
    """Feed grey samples one by one (``push``); ``finish`` returns the cut times in seconds."""

    def __init__(self, fps: float) -> None:
        self.fps = float(fps)
        self._hists: list[np.ndarray] = []
        self._prev: Optional[np.ndarray] = None
        self.l1: list[float] = [0.0]
        self.mad: list[float] = [0.0]

    def push(self, grey: np.ndarray, mad: Optional[float] = None) -> None:
        """``mad`` = mean |grey diff to the previous sample| / 255 when the caller already has it."""
        h = grey_hist(grey)
        if self._hists:
            self.l1.append(float(np.abs(h - self._hists[-1]).sum()))
            if mad is None:
                mad = float(np.mean(np.abs(grey.astype(np.int16) - self._prev.astype(np.int16)))) / 255.0
            self.mad.append(float(mad))
        self._hists.append(h)
        self._prev = grey

    def cut_indices(self) -> list[int]:
        n = len(self._hists)
        if n < 2:
            return []
        return detect_cuts(self.l1[:n], self.mad[:n], self.fps, np.asarray(self._hists))

    def finish(self) -> list[float]:
        return [cut_time(p, self.fps) for p in self.cut_indices()]


def refine_cut(path, t_est: float, fps: float, window: float = 0.6) -> float:
    """Exact time (seconds, on the frame grid) of the biggest picture change within ``t_est +- window``.

    Decodes only that window at the full frame rate as tiny grey frames and returns the time of the first frame of the new
    shot.  Falls back to ``t_est`` when anything fails or the window holds no clear cut.
    """
    t_est = float(t_est)
    try:
        fps = float(fps)
        if not (math.isfinite(fps) and fps > 0 and math.isfinite(t_est) and window > 0):
            return t_est
        first = max(0, int(math.floor((t_est - window) * fps + 1e-6)))     # first frame of the window, on the grid
        t0 = first / fps
        span = max(t_est + window - t0, 2.0 / fps)
        cwd, name = ffmpeg_input(path)
        args: list[str] = []
        if first > 0:                                                      # seek a quarter frame early: timestamp jitter
            args += ["-ss", f"{max(t0 - 0.25 / fps, 0.0):.6f}"]            # must not drop the first frame of the window
        s = REFINE_SIZE
        args += ["-i", name, "-t", f"{span:.6f}", "-an", "-sn", "-dn",
                 "-vf", f"fps={fps:.6f},scale={s}:{s}:flags=area", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
        limit = int(span * fps) + 4
        frames: list[np.ndarray] = []
        with FFmpegStream(args, cwd=cwd, stall_timeout=30.0) as st:
            while len(frames) < limit:
                buf = st.read(s * s)
                if len(buf) < s * s:
                    break
                frames.append(np.frombuffer(buf, dtype=np.uint8))
        if len(frames) < 2:
            return t_est
        d = np.abs(np.diff(np.stack(frames).astype(np.int16), axis=0)).mean(axis=1)      # d[k-1]: frame k-1 -> k
        if float(d.max()) < max(REFINE_MIN_CHANGE, REFINE_PEAK * float(np.median(d))):
            return t_est
        times = (first + np.arange(1, len(frames))) / fps
        best = np.flatnonzero(d >= float(d.max()) - 1e-9)
        k = int(best[np.argmin(np.abs(times[best] - t_est))])
        return round(float(times[k]), 6)
    except Exception:
        return t_est
