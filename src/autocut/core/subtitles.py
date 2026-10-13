"""Read/write subtitle files (SRT, VTT, simple ASS) without third-party libraries."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..util import read_text_guess

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts_to_sec(m: re.Match) -> float:
    h = int(m.group(1) or 0)
    ms = int(m.group(4).ljust(3, "0")[:3])
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + ms / 1000.0


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    """Also accepts WebVTT bodies (``.`` as decimal separator, optional cue ids/headers)."""
    out: list[tuple[float, float, str]] = []
    blocks = re.split(r"\r?\n\r?\n+", text.replace("﻿", "").strip())
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip() != ""]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        a, _, rest = lines[idx].partition("-->")
        ma, mb = _TS.search(a), _TS.search(rest)
        if not ma or not mb:
            continue
        body = " ".join(ln.strip() for ln in lines[idx + 1:])
        body = re.sub(r"<[^>]+>", "", body)
        out.append((_ts_to_sec(ma), _ts_to_sec(mb), body.strip()))
    return out


def parse_ass(text: str) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    fmt: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln.lower().startswith("format:") and "Start" in ln:
            fmt = [p.strip().lower() for p in ln.split(":", 1)[1].split(",")]
        elif ln.lower().startswith("dialogue:") and fmt:
            parts = ln.split(":", 1)[1].split(",", len(fmt) - 1)
            row = dict(zip(fmt, parts))

            def t(s: str) -> float:
                h, m, sec = s.strip().split(":")
                return int(h) * 3600 + int(m) * 60 + float(sec)

            try:
                body = re.sub(r"\{[^}]*\}", "", row.get("text", "")).replace("\\N", " ").replace("\\n", " ")
                out.append((t(row["start"]), t(row["end"]), body.strip()))
            except (KeyError, ValueError):
                continue
    return out


def load_subtitle_file(path: str | Path) -> list[tuple[float, float, str]]:
    text = read_text_guess(path)
    ext = Path(path).suffix.lower()
    cues = parse_ass(text) if ext == ".ass" else parse_srt(text)
    return [c for c in cues if c[1] > c[0] and c[2]]


def fmt_srt_time(t: float) -> str:
    ms_total = int(round(max(0.0, t) * 1000))
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: Iterable) -> str:
    """``cues``: objects with ``start/end/text`` or ``(start, end, text)`` tuples."""
    lines: list[str] = []
    for i, q in enumerate(cues, 1):
        s, e, text = (q.start, q.end, q.text) if hasattr(q, "start") else q
        lines += [str(i), f"{fmt_srt_time(s)} --> {fmt_srt_time(e)}", text, ""]
    return "\n".join(lines)
