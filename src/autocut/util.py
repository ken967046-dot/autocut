"""Small shared helpers (ids, hashing, paths, media discovery).

Nothing in here imports Qt, OpenCV or numpy so that the core stays importable
in the lightest possible environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".ts",
    ".mpg", ".mpeg", ".wmv", ".flv", ".3gp", ".mxf",
}
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aif", ".aiff",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUBTITLE_EXTS = {".srt", ".ass", ".vtt"}
PROJECT_EXT = ".acproj"
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS | IMAGE_EXTS


# --------------------------------------------------------------------------- ids
def new_id(prefix: str = "") -> str:
    return f"{prefix}{secrets.token_hex(4)}"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class IdGen:
    """Deterministic id generator (used so that edit plans hash identically)."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.n = 0

    def __call__(self, kind: str = "") -> str:
        self.n += 1
        return f"{self.prefix}{kind}{self.n:04d}"


# ------------------------------------------------------------------ canonical json
def canonicalize(obj: Any) -> Any:
    """Make ``obj`` JSON-stable: floats rounded to 6 places, keys stringified."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        r = round(obj, 6)
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        return {str(k): canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return canonicalize(obj.to_dict())
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, UTF-8 friendly: the input of every hash we make."""
    return json.dumps(canonicalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint_file(path: str | os.PathLike, chunk: int = 65536) -> str:
    """Cheap content fingerprint: size + head/middle/tail chunks (not mtime)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1()
    h.update(str(size).encode())
    with p.open("rb") as f:
        for off in (0, max(0, size // 2 - chunk // 2), max(0, size - chunk)):
            f.seek(off)
            h.update(f.read(chunk))
    return h.hexdigest()[:20]


# ----------------------------------------------------------------------- file io
def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | os.PathLike, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def read_text_guess(path: str | os.PathLike) -> str:
    """Read a text file that may be UTF-8 (with BOM), UTF-16 or Big5 (Taiwan legacy)."""
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ("utf-8-sig", "cp950", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# -------------------------------------------------------------------------- dirs
def app_data_dir() -> Path:
    override = os.environ.get("AUTOCUT_HOME")
    if override:
        base = Path(override)
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "AutoCutStudio"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "AutoCutStudio"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "autocut-studio"
    base.mkdir(parents=True, exist_ok=True)
    return base


def cache_dir(*parts: str) -> Path:
    p = app_data_dir().joinpath("cache", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ media listing
def natural_key(s: str) -> list:
    """Sort key that orders "clip 2" before "clip 10".  ``re.split`` with one group alternates text, number, text, number
    ...: the position says which is which (``str.isdigit`` is also true for "①" or "²", which ``int`` cannot read)."""
    return [int(t) if i % 2 else t.lower() for i, t in enumerate(re.split(r"(\d+)", s))]


def media_kind_by_ext(path: str | os.PathLike) -> str | None:
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in IMAGE_EXTS:
        return "image"
    return None


def collect_media_files(paths: Iterable[str | os.PathLike], recursive: bool = True) -> list[Path]:
    """Expand dropped files/folders into a flat, naturally-sorted list of media files.

    Order of the given paths is kept; folder contents are naturally sorted
    (``clip2`` before ``clip10``).  Hidden files and AutoCut's own output
    folders are skipped.
    """
    out: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        key = str(p.resolve()).lower() if sys.platform.startswith("win") else str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)

    for raw in paths:
        p = Path(str(raw).strip().strip('"'))
        if p.is_file():
            if media_kind_by_ext(p):
                add(p)
        elif p.is_dir():
            walker: Iterator[Path] = p.rglob("*") if recursive else p.glob("*")
            found = [
                f for f in walker
                if f.is_file() and media_kind_by_ext(f)
                and not any(part.startswith(".") or part == "AutoCut_Output" for part in f.relative_to(p).parts)
            ]
            for f in sorted(found, key=lambda x: natural_key(str(x.relative_to(p)))):
                add(f)
    return out


# ---------------------------------------------------------------------- formatting
def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def human_duration(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def clamp(v: float, lo: float, hi: float) -> float:
    """``v`` limited to ``[lo, hi]``.  Not-a-number gives ``lo``: it must never get through as a value."""
    return lo if (v < lo or v != v) else hi if v > hi else v
