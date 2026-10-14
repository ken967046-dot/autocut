"""ASS (Advanced SubStation Alpha) generation for burned-in captions."""
from __future__ import annotations

from typing import Iterable

from ..core.project import CaptionStyle, card_style

_CLOSERS = set("）」』》〉】〕)]")
_OPENERS = set("（「『《〈【〔([")
# a line may end after these; a closing mark that overflows hangs on its line instead of starting the next one
_BREAK_AFTER = set(" 　,，、.。;；:：!！?？…") | _CLOSERS


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


def _greedy(para: str, max_units: float) -> list[str]:
    lines: list[str] = []
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
                k = len(cur) - 1                          # the overflowing character starts the next line ...
                while k > 1 and cur[k - 1] in _OPENERS:   # ... and takes an opening mark along (never a line end)
                    k -= 1
                lines.append(cur[:k].strip())
                cur = cur[k:]
            width = sum(_unit(c) for c in cur)
            last_break = -1
    if cur.strip():
        lines.append(cur.strip())
    return lines


def _fits(line: str, max_units: float) -> bool:
    w = sum(_unit(c) for c in line)
    if line and line[-1] in _CLOSERS:                    # a closing mark may hang
        w -= _unit(line[-1])
    return w <= max_units + 1e-9


def _two_lines(para: str, max_units: float) -> list[str] | None:
    """The best split of ``para`` into two lines that both fit: as even as possible, never inside a Latin word, never
    starting a line with a closing mark or ending one with an opening mark; breaks at punctuation and outside brackets
    preferred.  ``None`` when no split fits."""
    best, best_cost = None, None
    depth = 0
    for i in range(1, len(para)):
        prev, nxt = para[i - 1], para[i]
        depth += (prev in _OPENERS) - (prev in _CLOSERS)
        a, b = para[:i].strip(), para[i:].strip()
        if not a or not b or nxt in _CLOSERS or prev in _OPENERS:
            continue
        if prev.isascii() and prev.isalnum() and nxt.isascii() and nxt.isalnum():
            continue                                      # inside a Latin word or number
        if not (_fits(a, max_units) and _fits(b, max_units)):
            continue
        cost = abs(sum(_unit(c) for c in a) - sum(_unit(c) for c in b))
        if prev in _BREAK_AFTER or nxt in _OPENERS or nxt == " ":
            cost -= 1.5                                   # a natural break
        if depth > 0 and nxt not in _OPENERS:
            cost += 5.0                                   # inside （…）: keep the bracket together if possible
        if best_cost is None or cost < best_cost - 1e-9:
            best, best_cost = [a, b], cost
    return best


def wrap_text(text: str, max_units: float) -> str:
    """Soft-wrap mixed CJK/ASCII text to ``max_units`` full-width characters per line (ASS ``\\N`` joins).  Text that
    needs two lines is split evenly at the most natural point (balanced subtitles, no one-character second line); a
    closing mark hangs on its line, an opening mark never ends one."""
    lines: list[str] = []
    for para in text.replace("\r", "").strip().split("\n"):
        if _fits(para.strip(), max_units):
            lines.append(para.strip())
            continue
        greedy = _greedy(para, max_units)
        two = _two_lines(para.strip(), max_units) if len(greedy) == 2 else None
        lines.extend(two or greedy)
    return "\\N".join(ln for ln in lines if ln)


def wrap_units(style: CaptionStyle, width: int, height: int) -> int:
    """Full-width characters per line: the style's ``max_chars``, but never more than fit across the frame between the
    5 % side margins (a 9:16 frame holds far fewer big characters than a 16:9 one -- without this a 15-character
    subtitle ran off both edges).  The renderer and the preview monitor both wrap with this."""
    size = max(8, int(round(style.size * height)))
    border = max(style.outline * height, size * 0.18 if style.box else 0.0)
    fit = int((width * 0.9 - 2.0 * border) / (size * 1.02))      # a little slack for bold glyphs
    return max(2, min(max(4, int(style.max_chars)), fit))


def _escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")")


def _style_line(name: str, style: CaptionStyle, width: int, height: int) -> str:
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
    return (f"Style: {name},{style.font},{size},{primary},{primary},{outline_col},{back_col},"
            f"{-1 if style.bold else 0},0,0,0,100,100,0,0,{border_style},{outline:.2f},{shadow:.2f},"
            f"{align},{side},{side},{margin_v},1\n")


def build_ass(cues: Iterable[tuple[float, float, str]], style: CaptionStyle, width: int, height: int,
              t0: float = 0.0, t1: float | None = None, cards: Iterable[tuple[float, float, str]] = ()) -> str:
    """ASS script for the cues overlapping ``[t0, t1)``; times in the script are relative to ``t0``.  ``cards`` are
    title cards (字卡), drawn with :func:`~autocut.core.project.card_style` on a layer above the subtitles."""
    cards = list(cards)
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {width}\nPlayResY: {height}\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + _style_line("Default", style, width, height)
        + (_style_line("Card", card_style(style), width, height) if cards else "")
        + "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    out = [head]
    for layer, name, st, items in ((0, "Default", style, cues), (1, "Card", card_style(style), cards)):
        max_units = wrap_units(st, width, height)
        for s, e, text in items:
            if t1 is not None and (s >= t1 or e <= t0):
                continue
            if t1 is None and e <= t0:
                continue
            s2, e2 = max(s, t0), (min(e, t1) if t1 is not None else e)
            if e2 - s2 < 0.02 or not text.strip():
                continue
            out.append(f"Dialogue: {layer},{ass_time(s2 - t0)},{ass_time(e2 - t0)},{name},,0,0,0,,"
                       f"{wrap_text(_escape(text), max_units)}\n")
    return "".join(out)
