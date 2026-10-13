"""ASS (Advanced SubStation Alpha) generation for burned-in captions."""
from __future__ import annotations

from typing import Iterable

from ..core.project import CaptionStyle

_BREAK_AFTER = set(" 　,，、.。;；:：!！?？…")


def ass_color(rgb: str, alpha: float = 0.0) -> str:
    """``#RRGGBB`` -> ``&HAABBGGRR`` (alpha: 0 opaque .. 1 transparent)."""
    h = rgb.lstrip("#")
    if len(h) != 6:
        h = "FFFFFF"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = max(0, min(255, int(round(alpha * 255))))
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


def ass_time(t: float) -> str:
    cs_total = int(round(max(0.0, t) * 100))
    h, rem = divmod(cs_total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _unit(ch: str) -> float:
    return 1.0 if ord(ch) >= 0x2E80 else 0.55


def wrap_text(text: str, max_units: float) -> str:
    """Soft-wrap mixed CJK/ASCII text to ``max_units`` full-width characters per line (ASS ``\\N`` joins)."""
    lines: list[str] = []
    for para in text.replace("\r", "").strip().split("\n"):
        cur, width, last_break = "", 0.0, -1
        for ch in para:
            cur += ch
            width += _unit(ch)
            if ch in _BREAK_AFTER:
                last_break = len(cur)
            if width > max_units:
                if last_break > 0 and last_break >= len(cur) * 0.5:
                    lines.append(cur[:last_break].strip())
                    cur = cur[last_break:]
                else:
                    lines.append(cur[:-1].strip())
                    cur = cur[-1:]
                width = sum(_unit(c) for c in cur)
                last_break = -1
        if cur.strip():
            lines.append(cur.strip())
    return "\\N".join(ln for ln in lines if ln)


def _escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def build_ass(cues: Iterable[tuple[float, float, str]], style: CaptionStyle, width: int, height: int,
              t0: float = 0.0, t1: float | None = None) -> str:
    """ASS script for the cues overlapping ``[t0, t1)``; times in the script are relative to ``t0``."""
    size = max(8, int(round(style.size * height)))
    outline = max(0.0, style.outline * height)
    shadow = max(0.0, style.shadow * height)
    margin_v = int(round(style.margin * height))
    align = {"bottom": 2, "center": 5, "top": 8}.get(style.position, 2)
    border_style = 3 if style.box else 1
    primary = ass_color(style.color)
    if style.box:
        box = ass_color(style.box_color, 1.0 - max(0.0, min(1.0, style.box_opacity)))
        outline_col, back_col = box, box
        outline = max(outline, size * 0.18)
    else:
        outline_col, back_col = ass_color(style.outline_color), ass_color("#000000", 0.5)
    side = int(round(width * 0.05))
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font},{size},{primary},{primary},{outline_col},{back_col},"
        f"{-1 if style.bold else 0},0,0,0,100,100,0,0,{border_style},{outline:.2f},{shadow:.2f},"
        f"{align},{side},{side},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    max_units = max(4, style.max_chars)
    out = [head]
    for s, e, text in cues:
        if t1 is not None and (s >= t1 or e <= t0):
            continue
        if t1 is None and e <= t0:
            continue
        s2, e2 = max(s, t0), (min(e, t1) if t1 is not None else e)
        if e2 - s2 < 0.02 or not text.strip():
            continue
        out.append(f"Dialogue: 0,{ass_time(s2 - t0)},{ass_time(e2 - t0)},Default,,0,0,0,,"
                   f"{wrap_text(_escape(text), max_units)}\n")
    return "".join(out)
