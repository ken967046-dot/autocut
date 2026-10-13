"""Locate and drive ffmpeg / ffprobe."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..util import app_data_dir

_WIN = sys.platform.startswith("win")
_NOWINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if _WIN else 0


class FFmpegError(RuntimeError):
    def __init__(self, message: str, stderr: str = "", returncode: int | None = None) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode


class FFmpegMissing(FFmpegError):
    pass


class Cancelled(Exception):
    pass


def _exe(name: str) -> str:
    return name + ".exe" if _WIN else name


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("AUTOCUT_FFMPEG_DIR")
    if env:
        dirs += [Path(env), Path(env) / "bin"]
    root = Path(__file__).resolve().parents[3]            # <repo>/  (src/autocut/media/ffmpeg.py)
    dirs += [root / "tools" / "ffmpeg" / "bin", root / "tools" / "ffmpeg"]
    dirs += [app_data_dir() / "ffmpeg" / "bin", app_data_dir() / "ffmpeg"]
    if getattr(sys, "frozen", False):                       # PyInstaller bundle
        here = Path(sys.executable).resolve().parent         # the folder of AutoCutStudio.exe: where a person drops ffmpeg
        dirs += [here / "ffmpeg" / "bin", here / "ffmpeg", here]
        base = Path(getattr(sys, "_MEIPASS", here))          # ... and what a build made with AUTOCUT_BUNDLE_FFMPEG carries
        if base != here:
            dirs += [base / "ffmpeg", base]
    return dirs


def find_binary(name: str) -> Optional[str]:
    for d in _candidate_dirs():
        p = d / _exe(name)
        if p.is_file():
            return str(p)
    return shutil.which(name)


@dataclass
class FFmpegInfo:
    ffmpeg: str
    ffprobe: str
    version_text: str
    version: tuple[int, int]
    filters: frozenset = field(default_factory=frozenset)
    encoders: frozenset = field(default_factory=frozenset)

    @property
    def has_ass(self) -> bool:
        return "ass" in self.filters

    @property
    def amix_normalize(self) -> bool:
        return self.version >= (4, 4)

    def has_encoder(self, name: str) -> bool:
        return name in self.encoders


def _parse_version(text: str) -> tuple[int, int]:
    m = re.search(r"version\s+n?(\d+)\.(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 99, 0      # git/nightly builds: assume recent


def _run_text(args: Sequence[str], timeout: float = 30.0) -> str:
    p = subprocess.run(list(args), capture_output=True, text=True, timeout=timeout, creationflags=_NOWINDOW,
                       encoding="utf-8", errors="replace")
    return (p.stdout or "") + (p.stderr or "")


@lru_cache(maxsize=1)
def ffmpeg_info() -> FFmpegInfo:
    ff, fp = find_binary("ffmpeg"), find_binary("ffprobe")
    if not ff or not fp:
        raise FFmpegMissing("找不到 ffmpeg / ffprobe。請雙擊專案資料夾裡的 Setup-AutoCut.bat 安裝（或單獨執行 scripts\\windows\\Get-Ffmpeg.ps1），也可以把 ffmpeg 的 bin 資料夾加入 PATH。")
    vtext = _run_text([ff, "-hide_banner", "-version"]).splitlines()[0]
    filt = set()
    for ln in _run_text([ff, "-hide_banner", "-filters"]).splitlines():
        parts = ln.split()
        if len(parts) >= 3 and re.fullmatch(r"[T.][S.][C.]?|[A-Z.]{2,3}", parts[0] or ""):
            filt.add(parts[1])
    enc = set()
    for ln in _run_text([ff, "-hide_banner", "-encoders"]).splitlines():
        parts = ln.split()
        if len(parts) >= 2 and re.fullmatch(r"[VAS][A-Z.]{5}", parts[0]):
            enc.add(parts[1])
    return FFmpegInfo(ff, fp, vtext, _parse_version(vtext), frozenset(filt), frozenset(enc))


def reset_ffmpeg_cache() -> None:
    ffmpeg_info.cache_clear()


def ffmpeg_available() -> bool:
    try:
        ffmpeg_info()
        return True
    except FFmpegMissing:
        return False


def run_ffmpeg(args: Sequence[str], *, cwd: str | os.PathLike | None = None,
               on_frame: Optional[Callable[[int], None]] = None, cancel: Optional[threading.Event] = None,
               log_path: str | os.PathLike | None = None, timeout: float | None = None) -> str:
    """Run ffmpeg with ``-progress pipe:1``; returns the stderr tail. Raises FFmpegError / Cancelled."""
    info = ffmpeg_info()
    cmd = [info.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-nostats", "-progress", "pipe:1", *args]
    errf = open(log_path, "w+", encoding="utf-8", errors="replace") if log_path else None
    stderr_target = errf if errf is not None else subprocess.PIPE
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=stderr_target,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                            creationflags=_NOWINDOW)
    done = threading.Event()
    killed = {"cancel": False, "timeout": False}

    def watch() -> None:
        waited = 0.0
        while not done.is_set():
            if cancel is not None and cancel.is_set():
                killed["cancel"] = True
                proc.kill()
                return
            if timeout is not None and waited > timeout:
                killed["timeout"] = True
                proc.kill()
                return
            done.wait(0.2)
            waited += 0.2

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    err_chunks: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith("frame=") and on_frame:
                try:
                    on_frame(int(line.split("=", 1)[1]))
                except ValueError:
                    pass
        if errf is None and proc.stderr is not None:
            err_chunks.append(proc.stderr.read())
        rc = proc.wait()
    finally:
        done.set()
        t.join(timeout=1)
        if proc.poll() is None:
            proc.kill()
    if errf is not None:
        errf.seek(0)
        err_text = errf.read()
        errf.close()
    else:
        err_text = "".join(err_chunks)
    if killed["cancel"]:
        raise Cancelled()
    if killed["timeout"]:
        raise FFmpegError("ffmpeg 執行逾時", err_text, rc)
    if rc != 0:
        raise FFmpegError(f"ffmpeg 失敗（代碼 {rc}）：{err_text.strip()[-600:]}", err_text, rc)
    return err_text


def run_probe_json(path: str | os.PathLike) -> dict:
    import json
    info = ffmpeg_info()
    p = subprocess.run(
        [info.ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, timeout=60, creationflags=_NOWINDOW)
    if p.returncode != 0:
        raise FFmpegError("無法讀取媒體資訊：" + p.stderr.decode("utf-8", "replace").strip()[-300:],
                          p.stderr.decode("utf-8", "replace"), p.returncode)
    return json.loads(p.stdout.decode("utf-8", "replace") or "{}")


@lru_cache(maxsize=8)
def detect_hw_encoder() -> str:
    """Return the first working hardware H.264 encoder (nvenc / qsv / amf) or ``libx264``."""
    try:
        info = ffmpeg_info()
    except FFmpegMissing:
        return "libx264"
    for enc in ("h264_nvenc", "h264_qsv", "h264_amf"):
        if not info.has_encoder(enc):
            continue
        try:
            p = subprocess.run(
                [info.ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=black:s=256x256:r=30:d=0.2", "-c:v", enc, "-f", "null", "-"],
                capture_output=True, timeout=20, creationflags=_NOWINDOW)
            if p.returncode == 0:
                return enc
        except (subprocess.TimeoutExpired, OSError):
            continue
    return "libx264"
