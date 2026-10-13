"""Thumbnails, film strips and waveforms (disk-cached, safe to call from worker threads)."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from ..core.project import Asset
from ..util import cache_dir
from .ffmpeg import FFmpegError, _NOWINDOW, ffmpeg_info


def _key(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def thumbnail(asset: Asset, t: float = 0.0, height: int = 96) -> Path:
    """One JPEG frame at source time ``t`` (cached)."""
    out = cache_dir("thumbs") / f"{_key(asset.fingerprint, asset.path, round(t, 2), height)}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out
    info = ffmpeg_info()
    cmd = [info.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if asset.kind != "image" and t > 0:
        cmd += ["-ss", f"{t:.3f}"]
    cmd += ["-i", asset.path, "-frames:v", "1", "-vf", f"scale=-2:{height}", "-q:v", "4", str(out)]
    p = subprocess.run(cmd, capture_output=True, timeout=60, creationflags=_NOWINDOW)
    if p.returncode != 0 or not out.exists():
        raise FFmpegError("縮圖產生失敗：" + p.stderr.decode("utf-8", "replace")[-200:])
    return out


def filmstrip(asset: Asset, count: int = 12, height: int = 54) -> Path:
    """A ``count``-tile horizontal strip of evenly spaced frames (one ffmpeg run, cached)."""
    out = cache_dir("strips") / f"{_key(asset.fingerprint, asset.path, count, height)}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return out
    if asset.kind == "image" or asset.duration <= 0:
        return thumbnail(asset, 0.0, height)
    info = ffmpeg_info()
    rate = count / max(asset.duration, 0.1)
    vf = f"fps={rate:.6f},scale=-2:{height},tile={count}x1"
    cmd = [info.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", asset.path, "-an", "-vf", vf,
           "-frames:v", "1", "-q:v", "5", str(out)]
    p = subprocess.run(cmd, capture_output=True, timeout=180, creationflags=_NOWINDOW)
    if p.returncode != 0 or not out.exists():
        # very short clips can yield fewer than ``count`` frames: fall back to a single frame
        return thumbnail(asset, min(asset.duration / 2, 1.0), height)
    return out


def waveform(asset: Asset, per_second: int = 50) -> np.ndarray:
    """Peak amplitude (0..1) per 1/``per_second`` s, mono. Cached as .npy."""
    if not asset.has_audio:
        return np.zeros(0, dtype=np.float32)
    out = cache_dir("wave") / f"{_key(asset.fingerprint, asset.path, per_second)}.npy"
    if out.exists():
        try:
            return np.load(out)
        except (OSError, ValueError):
            pass
    info = ffmpeg_info()
    rate = 8000
    cmd = [info.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", asset.path, "-vn", "-ac", "1", "-ar",
           str(rate), "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=_NOWINDOW)
    if p.returncode != 0:
        raise FFmpegError("波形產生失敗")
    pcm = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    step = max(1, rate // per_second)
    n = len(pcm) // step
    if n == 0:
        peaks = np.zeros(0, dtype=np.float32)
    else:
        peaks = np.abs(pcm[: n * step]).reshape(n, step).max(axis=1)
    np.save(out, peaks)
    return peaks


def decode_audio_mono(path: str, rate: int = 16000, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Decode (a part of) the audio of ``path`` to a float32 mono array at ``rate`` Hz."""
    info = ffmpeg_info()
    cmd = [info.ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=3600, creationflags=_NOWINDOW)
    if p.returncode != 0:
        raise FFmpegError("音訊解碼失敗：" + p.stderr.decode("utf-8", "replace")[-200:])
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0
