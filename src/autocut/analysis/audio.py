"""Streaming audio features: loudness, speech-band ratio, voice activity, silences (spec §2.2).

The maths (:class:`AudioFeatureStream`) consumes float32 mono blocks of any size, so it is unit-testable
without ffmpeg and never needs the whole file in memory.  :func:`stream_audio` produces those blocks from a
media file through one ffmpeg child process; :class:`FFmpegStream` is the cancel-safe pipe reader that the
picture pass (``visual.py``) and ``refine_cut`` (``scenes.py``) share.

Frame grid: frame ``i`` covers ``[i/audio_fps, (i+1)/audio_fps)``; its analysis window (~64 ms) is centred
on the frame centre, so a 50 ms hop sees 7 ms of each neighbour.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from ..media.ffmpeg import Cancelled, FFmpegError, _NOWINDOW, ffmpeg_info

DB_FLOOR = -90.0            # rms_db never goes below this (digital silence)
WINDOW_S = 0.064            # analysis window
BAND_LO, BAND_HI = 300.0, 3400.0
VOICE_BAND_MIN = 0.45       # speech-band energy share needed to *enter* voice
VOICE_HYST_DB = 4.0         # once in voice, stay while rms > voice_thr - this ...
VOICE_HYST_BAND = 0.35      # ... and band ratio > this
VOICE_MIN_S = 0.15          # shorter voice runs are dropped
VOICE_GAP_S = 0.2           # shorter gaps inside voice are filled
SILENCE_MIN_S = 0.3
SILENCE_CAP_DB = -25.0      # the adaptive silence threshold never goes above this
_BATCH = 256                # frames per vectorised batch (bounded memory)


# ----------------------------------------------------------------------------- ffmpeg pipe
def ffmpeg_input(path: str | os.PathLike) -> tuple[str, str]:
    """``(cwd, input-arg)`` so ffmpeg gets a short cwd-relative name (AGENTS rule 7; Chinese names OK)."""
    p = Path(path).resolve()
    return str(p.parent), "file:" + p.name


class FFmpegStream:
    """ffmpeg writing raw data to stdout, read in fixed-size chunks.

    A watchdog thread kills the child when ``cancel`` is set or nothing could be read for ``stall_timeout``
    seconds, so a read never blocks forever.  stderr goes to a temp file (no pipe deadlock).  Always use as a
    context manager (or call :meth:`close`): the child is killed and reaped on exit.
    """

    def __init__(self, args: Sequence[str], *, cancel=None, cwd: Optional[str] = None,
                 stall_timeout: float = 120.0, loglevel: str = "error") -> None:
        info = ffmpeg_info()
        cmd = [info.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", loglevel, *args]
        self._err = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=self._err, creationflags=_NOWINDOW)
        except OSError:
            self._err.close()
            raise
        self._cancel = cancel
        self._stall = stall_timeout
        self._last = time.monotonic()
        self._done = threading.Event()
        self.cancelled = False
        self.stalled = False
        self._watch = threading.Thread(target=self._watchdog, name="ffmpeg-watchdog", daemon=True)
        self._watch.start()

    def _watchdog(self) -> None:
        while not self._done.wait(0.1):
            if self._cancel is not None and self._cancel.is_set():
                self.cancelled = True
            elif time.monotonic() - self._last > self._stall:
                self.stalled = True
            else:
                continue
            self._kill()
            return

    def _kill(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass

    def read(self, n: int) -> bytes:
        """Up to ``n`` bytes; fewer only at end of stream (or after a kill)."""
        assert self.proc.stdout is not None
        data = self.proc.stdout.read(n)
        self._last = time.monotonic()
        return data or b""

    def stderr_tail(self, limit: int = 600) -> str:
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip()[-limit:]
        except (OSError, ValueError):
            return ""

    def finish(self) -> None:
        """Wait for a clean exit; raise :class:`Cancelled` / :class:`FFmpegError` otherwise."""
        rc = self.proc.wait()
        self._done.set()
        if self.cancelled or (self._cancel is not None and self._cancel.is_set()):
            raise Cancelled()
        if self.stalled:
            raise FFmpegError("ffmpeg 沒有回應（讀取逾時）", self.stderr_tail(), rc)
        if rc != 0:
            raise FFmpegError(f"ffmpeg 失敗（代碼 {rc}）：{self.stderr_tail()}", self.stderr_tail(), rc)

    def close(self) -> None:
        self._done.set()
        self._kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:          # pragma: no cover - kill() is not ignorable
            pass
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        self._watch.join(timeout=1)
        self._err.close()

    def __enter__(self) -> "FFmpegStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def stream_audio(path: str | os.PathLike, sr: int = 16000, cancel=None,
                 progress: Optional[Callable[[float], None]] = None, *, start: float = 0.0,
                 duration: Optional[float] = None, block_seconds: float = 0.5) -> Iterator[np.ndarray]:
    """Yield float32 mono blocks (``block_seconds`` long) of the audio of ``path`` resampled to ``sr``.

    ``progress(seconds_decoded)`` after every block.  Raises :class:`Cancelled` (child killed) or
    :class:`FFmpegError` (bad file / no audio stream).  Memory is one block, whatever the file length.
    """
    cwd, name = ffmpeg_input(path)
    args: list[str] = []
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    if duration is not None:
        args += ["-t", f"{max(duration, 0.0):.3f}"]
    args += ["-i", name, "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(int(sr)), "-f", "f32le", "-"]
    block = max(1, int(sr * block_seconds)) * 4
    with FFmpegStream(args, cancel=cancel, cwd=cwd) as st:
        decoded = 0
        while True:
            buf = st.read(block)
            usable = len(buf) - len(buf) % 4
            if usable and not st.cancelled:
                decoded += usable // 4
                yield np.frombuffer(buf[:usable], dtype="<f4")
                if progress is not None:
                    progress(decoded / sr)
            if len(buf) < block or (cancel is not None and cancel.is_set()):
                break
        st.finish()


# ----------------------------------------------------------------------------- features
@dataclass
class AudioFeatures:
    audio_fps: float
    duration: float                                      # seconds of audio analysed
    rms_db: list = field(default_factory=list)
    band_ratio: list = field(default_factory=list)       # speech-band energy share (not stored in Evidence)
    voice: list = field(default_factory=list)
    silences: list = field(default_factory=list)
    noise_floor_db: float = DB_FLOOR
    silence_thr_db: float = -38.0
    voice_thr_db: float = -38.0


class AudioFeatureStream:
    """Incremental per-frame RMS (dBFS) and speech-band ratio; :meth:`finish` adds thresholds and voice."""

    def __init__(self, sr: int = 16000, audio_fps: float = 20.0, window_s: float = WINDOW_S) -> None:
        if sr <= 0 or audio_fps <= 0:
            raise ValueError("sr 與 audio_fps 必須大於 0")
        self.sr, self.fps = int(sr), float(audio_fps)
        self.hop = self.sr / self.fps
        self.win = max(16, int(round(window_s * self.sr)))
        self._hann = np.hanning(self.win).astype(np.float32)
        freqs = np.fft.rfftfreq(self.win, 1.0 / self.sr)
        self._band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf0 = 0                    # global index of _buf[0]
        self._total = 0                   # samples received
        self._next = 0                    # next frame index to compute
        self._sumsq: list[np.ndarray] = []
        self._ratio: list[np.ndarray] = []

    def _start(self, i: np.ndarray | int) -> np.ndarray:
        return np.round((np.asarray(i, dtype=np.float64) + 0.5) * self.hop - self.win / 2).astype(np.int64)

    def push(self, block: np.ndarray) -> None:
        x = np.asarray(block, dtype=np.float32).ravel()
        if x.size == 0:
            return
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x.copy()
        self._total += x.size
        # frames whose whole window has arrived
        last = int((self._total + self.win / 2) / self.hop - 0.5)          # largest i with start+win <= total
        while last >= 0 and int(self._start(last)) + self.win > self._total:
            last -= 1
        self._compute(self._next, last + 1)

    def _compute(self, i0: int, i1: int) -> None:
        for a in range(i0, i1, _BATCH):
            b = min(i1, a + _BATCH)
            starts = self._start(np.arange(a, b))
            idx = starts[:, None] - self._buf0 + np.arange(self.win)[None, :]
            valid = (idx >= 0) & (idx < self._buf.size)
            frames = np.where(valid, self._buf[np.clip(idx, 0, max(self._buf.size - 1, 0))], 0.0) \
                if self._buf.size else np.zeros(idx.shape, np.float32)
            frames = frames.astype(np.float32, copy=False)
            n_valid = np.maximum(valid.sum(axis=1), 1)
            self._sumsq.append((np.einsum("ij,ij->i", frames, frames) / n_valid).astype(np.float64))
            power = np.abs(np.fft.rfft(frames * self._hann, axis=1)) ** 2
            total = power[:, 1:].sum(axis=1)
            self._ratio.append(np.where(total > 1e-20, power[:, self._band].sum(axis=1) / np.maximum(total, 1e-20),
                                        0.0))
        self._next = max(self._next, i1)
        keep_from = int(self._start(self._next)) - self._buf0               # drop samples no longer needed
        if keep_from > 0:
            self._buf = self._buf[keep_from:]
            self._buf0 += keep_from

    def finish(self, silence_db: float = -38.0) -> AudioFeatures:
        n = int(math.ceil(self._total / self.hop - 1e-6)) if self._total else 0
        if n > self._next:
            self._compute(self._next, n)
        msq = np.concatenate(self._sumsq)[:n] if self._sumsq else np.zeros(0)
        ratio = np.concatenate(self._ratio)[:n] if self._ratio else np.zeros(0)
        rms_db = np.maximum(10.0 * np.log10(np.maximum(msq, 1e-30)), DB_FLOOR)
        return features_from_frames(rms_db, ratio, self.fps, silence_db, duration=self._total / self.sr)


def adaptive_thresholds(rms_db: np.ndarray, silence_db: float = -38.0) -> tuple[float, float, float]:
    """``(noise_floor, silence_thr, voice_thr)`` as spec §2.2 (adapts to noisy rooms).

    ``floor`` = 10th percentile of the frame loudness; ``silence_thr = clamp(max(silence_db, floor + 8), <= -25)``;
    ``voice_thr = max(floor + 10, silence_db)``.  One addition for dense material (talking over music, no pauses, where the
    10th percentile is itself speech): ``floor + 10`` is capped at the midpoint between the floor and the 90th percentile
    (before the ``silence_db`` clamp), so a clean recording gets exactly the spec's numbers.
    """
    if len(rms_db) == 0:
        return DB_FLOOR, float(silence_db), float(silence_db)
    floor, loud = (float(v) for v in np.percentile(rms_db, (10, 90)))
    silence_thr = min(max(silence_db, floor + 8.0), SILENCE_CAP_DB)
    voice_thr = max(min(floor + 10.0, (floor + loud) / 2.0), silence_db)
    return floor, float(silence_thr), float(voice_thr)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[(start, end_exclusive), ...]`` of the True runs of a boolean array."""
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def hysteresis(enter: np.ndarray, stay: np.ndarray) -> np.ndarray:
    """Runs of ``stay`` that contain at least one ``enter`` frame."""
    out = np.zeros(len(stay), dtype=bool)
    for a, b in runs(np.asarray(stay, bool) | np.asarray(enter, bool)):
        if np.any(enter[a:b]):
            out[a:b] = True
    return out


def close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps of at most ``max_gap`` frames that sit between two True runs."""
    out = np.asarray(mask, dtype=bool).copy()
    for a, b in runs(~out):
        if a > 0 and b < len(out) and b - a <= max_gap:
            out[a:b] = True
    return out


def drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for a, b in runs(out):
        if b - a < min_len:
            out[a:b] = False
    return out


def features_from_frames(rms_db: np.ndarray, band_ratio: np.ndarray, audio_fps: float,
                         silence_db: float = -38.0, duration: Optional[float] = None) -> AudioFeatures:
    """Thresholds, voice (0..1) and silences from per-frame loudness and speech-band ratio."""
    rms_db = np.asarray(rms_db, dtype=np.float64)
    band_ratio = np.asarray(band_ratio, dtype=np.float64)
    n = len(rms_db)
    dur = float(duration if duration is not None else n / audio_fps)
    floor, s_thr, v_thr = adaptive_thresholds(rms_db, silence_db)
    # Entering voice needs >= VOICE_MIN_S of consecutive evidence *before* the hysteresis spreads it: white noise has a
    # speech-band share of 0.39 +- 0.03, so single frames cross 0.45 now and then and would otherwise turn a whole noise
    # burst into "voice".
    enter = drop_short((rms_db > v_thr) & (band_ratio > VOICE_BAND_MIN), int(math.ceil(VOICE_MIN_S * audio_fps - 1e-9)))
    stay = (rms_db > v_thr - VOICE_HYST_DB) & (band_ratio > VOICE_HYST_BAND)
    mask = close_gaps(hysteresis(enter, stay), int(round(VOICE_GAP_S * audio_fps)))
    # inside voice: 0.6 (just above the threshold, or a bridged gap) .. 1.0 (>= 12 dB above it)
    strength = 0.6 + 0.4 * np.clip((rms_db - v_thr) / 12.0, 0.0, 1.0)
    voice = np.where(mask, strength, 0.0)
    min_sil = int(math.ceil(SILENCE_MIN_S * audio_fps - 1e-9))
    silences = [[round(a / audio_fps, 3), round(min(b / audio_fps, dur), 3)]
                for a, b in runs(rms_db < s_thr) if b - a >= min_sil]
    return AudioFeatures(audio_fps=float(audio_fps), duration=dur,
                         rms_db=[round(float(v), 2) for v in rms_db],
                         band_ratio=[round(float(v), 3) for v in band_ratio],
                         voice=[round(float(v), 3) for v in voice], silences=silences,
                         noise_floor_db=round(floor, 2), silence_thr_db=round(s_thr, 2), voice_thr_db=round(v_thr, 2))


def analyze_audio_blocks(blocks: Iterable[np.ndarray], sr: int = 16000, audio_fps: float = 20.0,
                         silence_db: float = -38.0) -> AudioFeatures:
    """Features of a stream of float32 mono blocks (any block sizes)."""
    fs = AudioFeatureStream(sr, audio_fps)
    for b in blocks:
        fs.push(b)
    return fs.finish(silence_db)


def analyze_audio_file(path: str | os.PathLike, *, sr: int = 16000, audio_fps: float = 20.0,
                       silence_db: float = -38.0, cancel=None,
                       progress: Optional[Callable[[float], None]] = None) -> AudioFeatures:
    """Stream the audio of ``path`` through :class:`AudioFeatureStream` (``progress(seconds)``)."""
    gen = stream_audio(path, sr, cancel, progress)
    try:
        return analyze_audio_blocks(gen, sr, audio_fps, silence_db)
    finally:
        gen.close()
