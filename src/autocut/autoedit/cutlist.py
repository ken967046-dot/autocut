"""剪輯清單 (cut list): shorts chosen by a person (or an AI editor the person agreed with), cut exactly as listed.

The automatic window search finds *candidates*; once someone has decided which moments are the highlights, the editor
should cut exactly those -- including a cold open (a later line played first), skipped stumbles and title cards
(字卡).  A cut list is a small UTF-8 JSON file::

    {
      "schema": "autocut.cutlist/1",
      "source": "訪談.mp4",                      # optional, only a hint shown to the person
      "shorts": [
        {
          "title": "沒人幫我，只好自己拍",
          "ranges": [["5:33.33", "5:43.17"], ["5:44.63", "5:56.17"], ["6:10.97", "6:17.07"]],
          "cards": [{"text": "Ken桑一年半前的加盟糾紛", "at": 0, "duration": 3},
                    {"text": "（瞬間切回專業模式）", "source": "5:52.0"}],
          "note": "為什麼選這段（選填）"
        }
      ]
    }

* ``ranges`` are SOURCE times, played in the listed order (a range may come before an earlier one: a cold open; the
  same moment may appear twice).  Times: seconds (``333.3``), ``m:ss(.xx)``, ``h:mm:ss(.xx)`` or SRT style
  ``00:05:33,330``.
* ``cards`` are title cards (``duration`` seconds, default 3).  ``at`` places one on the SHORT's own timeline (seconds
  from its start); ``source`` pins it to a SOURCE time instead -- it appears when that moment plays, wherever trimmed
  pauses or a cold open moved it (the first time it plays, when a moment is listed twice).
* Everything is validated up front; problems raise :class:`~autocut.autoedit.api.AutoEditError` with a 繁中 message that
  names the short and the field.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..util import canonical_json, read_text_guess, sha256_hex
from .api import AutoEditError

SCHEMA = "autocut.cutlist/1"
MAX_SHORTS = 50
MAX_RANGES = 40
MIN_RANGE = 0.25                                   # = shorts.MIN_CLIP: a shorter piece is never cut out
CARD_DEFAULT_SECONDS = 3.0

_HMS = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:[.,]\d+)?)\s*$")


@dataclass
class Card:
    text: str
    at: float = 0.0
    duration: float = CARD_DEFAULT_SECONDS
    source: Optional[float] = None                 # pinned to this source time (then ``at`` is unused)


@dataclass
class CutShort:
    title: str
    ranges: list                                   # [(source_t0, source_t1)], in playing order
    cards: list = field(default_factory=list)      # [Card]
    note: str = ""


@dataclass
class CutList:
    shorts: list
    source: str = ""
    name: str = ""
    sha256: str = ""


def parse_time(value: Any) -> float:
    """``333.3`` / ``"5:33.3"`` / ``"0:05:33.3"`` / ``"00:05:33,330"`` -> seconds.  Raises ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, (int, float)):
        t = float(value)
    else:
        s = str(value).strip()
        m = _HMS.match(s)
        if m:
            h = int(m.group(1) or 0)
            mi = int(m.group(2))
            sec = float(m.group(3).replace(",", "."))
            if mi >= 60 and m.group(1) is not None or sec >= 60:
                raise ValueError(s)
            t = h * 3600 + mi * 60 + sec
        else:
            t = float(s.replace(",", "."))
    if t != t or t in (float("inf"), float("-inf")) or t < 0:
        raise ValueError(value)
    return t


def _fail(where: str, what: str) -> AutoEditError:
    return AutoEditError(f"剪輯清單有問題（{where}）：{what}")


def parse_cutlist(data: Any, name: str = "") -> CutList:
    """Validate a decoded cut list (dict) -> :class:`CutList`.  Source-range checks against the video length happen in
    :func:`check_against`."""
    if not isinstance(data, dict):
        raise _fail("整份檔案", "最外層必須是 { ... } 物件")
    schema = data.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise _fail("schema", f"不支援的格式「{schema}」（應為 {SCHEMA}）")
    shorts_in = data.get("shorts")
    if not isinstance(shorts_in, list) or not shorts_in:
        raise _fail("shorts", "至少要列一支短影音")
    if len(shorts_in) > MAX_SHORTS:
        raise _fail("shorts", f"一次最多 {MAX_SHORTS} 支")
    shorts: list[CutShort] = []
    for i, it in enumerate(shorts_in, 1):
        where = f"第 {i} 支"
        if not isinstance(it, dict):
            raise _fail(where, "每一支都必須是 { ... } 物件")
        title = str(it.get("title") or "").strip()
        ranges_in = it.get("ranges")
        if not isinstance(ranges_in, list) or not ranges_in:
            raise _fail(where, "ranges 至少要有一段 [開始, 結束]")
        if len(ranges_in) > MAX_RANGES:
            raise _fail(where, f"ranges 最多 {MAX_RANGES} 段")
        ranges = []
        for k, r in enumerate(ranges_in, 1):
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                raise _fail(f"{where}第 {k} 段", "每段必須是 [開始, 結束]")
            try:
                a, b = parse_time(r[0]), parse_time(r[1])
            except (TypeError, ValueError):
                raise _fail(f"{where}第 {k} 段", f"看不懂的時間 {r!r}（可用 333.3、5:33.3、0:05:33.3 或 00:05:33,330）") from None
            if b <= a:
                raise _fail(f"{where}第 {k} 段", f"結束時間必須晚於開始時間（{r[0]} → {r[1]}）")
            if b - a < MIN_RANGE - 1e-9:
                raise _fail(f"{where}第 {k} 段", f"只有 {b - a:.2f} 秒，太短（每段至少 {MIN_RANGE:g} 秒）")
            ranges.append((a, b))
        cards = []
        cards_in = it.get("cards") or []
        if not isinstance(cards_in, list):
            raise _fail(where, "cards 必須是清單")
        for k, c in enumerate(cards_in, 1):
            if isinstance(c, str):
                c = {"text": c}
            if not isinstance(c, dict) or not str(c.get("text") or "").strip():
                raise _fail(f"{where}字卡 {k}", "每張字卡都要有 text")
            if "at" in c and "source" in c:
                raise _fail(f"{where}字卡 {k}", "at（短影音內的秒數）和 source（來源時間）只能擇一")
            try:
                at = parse_time(c.get("at", 0))
                dur = parse_time(c.get("duration", CARD_DEFAULT_SECONDS))
                src = parse_time(c["source"]) if "source" in c else None
            except (TypeError, ValueError):
                raise _fail(f"{where}字卡 {k}", "at／source／duration 必須是秒數或時間") from None
            if dur <= 0:
                raise _fail(f"{where}字卡 {k}", "duration 必須大於 0")
            cards.append(Card(str(c["text"]).strip(), at, dur, src))
        shorts.append(CutShort(title, ranges, cards, str(it.get("note") or "").strip()))
    canon = {"schema": SCHEMA, "shorts": [{"title": s.title, "ranges": [[round(a, 4), round(b, 4)] for a, b in s.ranges],
                                           "cards": [[c.text, round(c.at, 4), round(c.duration, 4),
                                                      None if c.source is None else round(c.source, 4)] for c in s.cards],
                                           "note": s.note} for s in shorts]}
    return CutList(shorts, str(data.get("source") or ""), name, sha256_hex(canonical_json(canon)))


def load_cutlist(path: str | Path) -> CutList:
    p = Path(path)
    try:
        text = read_text_guess(p)
    except OSError as exc:
        raise AutoEditError(f"讀不到剪輯清單：{p.name}（{exc.__class__.__name__}）") from exc
    except (UnicodeError, ValueError) as exc:
        raise AutoEditError(f"剪輯清單的文字編碼無法辨識：{p.name}（請用記事本另存成 UTF-8）") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise AutoEditError(f"剪輯清單不是有效的 JSON：{p.name}（第 {getattr(exc, 'lineno', '?')} 行附近）") from exc
    return parse_cutlist(data, p.name)


def check_against(cl: CutList, duration: float) -> None:
    """Every range must lie inside the video (a small 0.25 s tolerance at the very end is clamped by the planner)."""
    for i, s in enumerate(cl.shorts, 1):
        for k, (a, b) in enumerate(s.ranges, 1):
            if a >= duration - 1e-6:
                raise _fail(f"第 {i} 支第 {k} 段", f"開始時間 {a:.2f} 秒已超過影片長度 {duration:.2f} 秒（剪輯清單是不是對應別支影片？）")
            if b > duration + 0.25:
                raise _fail(f"第 {i} 支第 {k} 段", f"結束時間 {b:.2f} 秒超過影片長度 {duration:.2f} 秒")


def dump_cutlist(shorts: list[dict], source: str = "") -> str:
    """Pretty JSON text of a cut list (for tools that write one)."""
    return json.dumps({"schema": SCHEMA, "source": source, "shorts": shorts}, ensure_ascii=False, indent=2)
