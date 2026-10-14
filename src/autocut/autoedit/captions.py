"""Captions for the automatic editor (spec 3.10 and 7): transcript / subtitle file -> timeline cues.

Pipeline
    source_segments()        transcript (speech recognition) or subtitle file, in SOURCE seconds
    cues_for_pieces()        map source time -> timeline time through the V1 pieces (cuts, jump cuts), drop what was cut
                             away, clip what crosses a cut, wrap/split CJK-aware, then fix the timing
    generate_cues_for_sequence()   the GUI's 「自動產生字幕」 button: transcribe an existing sequence

Everything here is pure data in -> data out and deterministic (ids come from ``util.IdGen``).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..core.project import CaptionCue, ordered_clips
from ..core.subtitles import load_subtitle_file
from ..core.timecode import snap_time
from ..media.ffmpeg import Cancelled
from ..util import IdGen
from .config import AuditRules

CAP_PAD = 0.15              # a cue stays this long after the speech ends ...
CAP_GAP = 0.12              # ... but always ends this long before the next cue starts
MIN_CUE_SECONDS = 0.4
MIN_VISIBLE = 0.1           # a cue that cannot be shown at least this long is dropped
HARD_PAUSE = 0.8            # a pause this long between two words always starts a new cue
SOFT_PAUSE = 0.25           # ... and this long is a good place to break a too-long cue
DEFAULT_MAX_CHARS = 16
LINE_COST = 10.0            # splitter: price of one more line ...
SHORT_LINE = 5.0            # ... and of a line narrower than this many full-width characters
PHRASE_MIN = 5.0            # a clause narrower than this is merged into its neighbour
CLAUSE_PAUSE = 0.35         # a pause this long between two words ends a clause

__all__ = ["Piece", "source_segments", "cues_for_pieces", "generate_cues_for_sequence", "display_width",
           "split_text", "CAP_PAD", "CAP_GAP", "MIN_CUE_SECONDS"]

_STRONG = set("。！？!?；;\n")
_WEAK = set("，、,：:…")
_CLOSERS = set("」』）)】》〉”’\"'")
_OPENERS = set("「『（(【《〈“‘")
_PARTICLES = set("的了嗎呢吧啊呀啦喔哦耶")
_TRIM = " 　，、；,;:：。."


# ----------------------------------------------------------------------------------------------- data
@dataclass
class Piece:
    """One V1 clip on the timeline (in timeline order)."""
    asset_id: str
    src_in: float
    src_out: float
    start: float
    speed: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, (self.src_out - self.src_in) / (self.speed or 1.0))

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class _Raw:
    start: float            # timeline seconds
    speech_end: float
    text: str
    order: int


def _report(progress: Optional[Callable], frac: float, msg: str) -> None:
    if progress is None:
        return
    try:
        progress(frac, msg)
    except TypeError:                                  # a one-argument callback
        progress(frac)


# ------------------------------------------------------------------------------------------ text helpers
def _cw(ch: str) -> float:
    """Display width of one character in full-width units (CJK / full-width = 1, everything else 0.5)."""
    if ch.isspace():
        return 0.5
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5


def display_width(text: str) -> float:
    """Width of ``text`` in full-width characters (the unit of ``max_chars``)."""
    return sum(_cw(c) for c in text)


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def _timing_weight(ch: str) -> float:
    return 0.0 if (ch.isspace() or _is_punct(ch)) else _cw(ch)


def _clean(text: str) -> str:
    """Whitespace collapsed and trailing/leading comma/full-stop style punctuation removed (``？！…`` are kept)."""
    return re.sub(r"\s+", " ", text).strip().strip(_TRIM)


def _reading_chars(text: str) -> int:
    """Characters that count for the reading speed (the audit: no spaces, no punctuation)."""
    return sum(1 for c in text if not c.isspace() and not _is_punct(c))


# ---------------------------------------------------------------------------- characters with times
Unit = tuple          # (char, t0, t1) in SOURCE seconds


def _proportional(text: str, s: float, e: float) -> list[Unit]:
    weights = [_timing_weight(c) for c in text]
    total = sum(weights)
    out: list[Unit] = []
    acc = 0.0
    for c, w in zip(text, weights):
        if total <= 0:
            out.append((c, s, s))
        else:
            out.append((c, s + (e - s) * acc / total, s + (e - s) * (acc + w) / total))
        acc += w
    return out


def _align_words(text: str, words: list, s: float, e: float) -> Optional[list[Unit]]:
    """Per-character times from word timings; ``None`` when the words do not line up with the text."""
    out: list[Unit] = []
    pos = 0
    last_end: Optional[float] = None
    for w in words:
        token = str(w.get("word", "")).strip()
        if not token:
            continue
        try:
            ws, we = float(w.get("start")), float(w.get("end"))
        except (TypeError, ValueError):
            return None
        k = text.find(token, pos)
        if k < 0:
            return None
        gap = text[pos:k]
        if any(not (c.isspace() or _is_punct(c)) for c in gap):
            return None                                   # a word was skipped: do not trust the alignment
        filler = ws if last_end is None else last_end
        out.extend((c, filler, filler) for c in gap)
        out.extend(_proportional(token, ws, max(we, ws)))
        pos = k + len(token)
        last_end = max(we, ws)
    if not out:
        return None
    filler = last_end if last_end is not None else s
    out.extend((c, filler, filler) for c in text[pos:])
    run = out[0][1]                                       # times never go backwards
    fixed: list[Unit] = []
    for c, a, b in out:
        a = max(a, run)
        b = max(b, a)
        fixed.append((c, a, b))
        run = b
    return fixed


def _timed_chars(seg: dict) -> list[Unit]:
    text = re.sub(r"\s+", " ", str(seg.get("text") or "")).strip()
    if not text:
        return []
    try:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
    except (TypeError, ValueError):
        return []
    e = max(e, s)
    words = seg.get("words") or []
    units = _align_words(text, list(words), s, e) if words else None
    return units if units is not None else _proportional(text, s, e)


# ----------------------------------------------------------------------------------- splitting
def _is_strong(units: list[Unit], i: int) -> bool:
    c = units[i][0]
    if c in _STRONG:
        return True
    if c == ".":                                          # "3.5" is not a sentence end
        nxt_space = i + 1 >= len(units) or units[i + 1][0].isspace()
        prev_digit = i > 0 and units[i - 1][0].isdigit()
        return nxt_space and not prev_digit
    return False


def _break_penalty(units: list[Unit], j: int) -> float:
    """Cost of a line break between ``units[j-1]`` and ``units[j]`` (smaller = nicer place to break)."""
    a, b = units[j - 1][0], units[j][0]
    if b in _CLOSERS or (_is_punct(b) and b not in _OPENERS):
        return 80.0
    if a in _OPENERS:
        return 80.0
    if a in _WEAK or a in _STRONG:
        return 0.0
    if units[j][1] - units[j - 1][2] >= SOFT_PAUSE:
        return 2.0
    extra = 10.0 if b in _PARTICLES else 0.0
    if a.isspace() or b.isspace():
        return 4.0 + extra
    fa, fb = _cw(a) >= 1.0, _cw(b) >= 1.0
    if fa and fb:
        return 15.0 + extra                               # between two CJK characters: no phrase boundary known
    if fa != fb:
        return 8.0 + extra
    if _is_punct(a):
        return 3.0
    return 60.0                                           # inside a Latin word / number


def _trim_units(units: list[Unit]) -> list[Unit]:
    i, j = 0, len(units)
    while i < j and units[i][0].isspace():
        i += 1
    while j > i and units[j - 1][0].isspace():
        j -= 1
    return units[i:j]


def _balanced_split(units: list[Unit], max_chars: float) -> list[list[Unit]]:
    """Lines <= max_chars wide, broken at phrase boundaries where possible; an extra line is cheaper than a break in
    the middle of a phrase, very short lines and unbalanced lines cost a little (dynamic programme)."""
    n = len(units)
    prefix = [0.0]
    for u in units:
        prefix.append(prefix[-1] + _cw(u[0]))
    total = prefix[-1]
    nmin = max(1, math.ceil(total / max_chars - 1e-9))
    ideal = total / nmin
    pen = [0.0] * (n + 1)
    for j in range(1, n):
        pen[j] = _break_penalty(units, j)
    inf = float("inf")
    best = [inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            w = prefix[j] - prefix[i]
            if w > max_chars + 1e-9 and i < j - 1:
                break
            if best[i] == inf:
                continue
            c = (best[i] + LINE_COST + 0.1 * (w - ideal) ** 2 + 3.0 * max(0.0, SHORT_LINE - w) ** 2
                 + (pen[j] if j < n else 0.0))
            if c < best[j] - 1e-12:
                best[j], back[j] = c, i
    cuts: list[int] = []
    j = n
    while j > 0:
        cuts.append(j)
        j = back[j]
    cuts.reverse()
    out, prev = [], 0
    for c in cuts:
        out.append(units[prev:c])
        prev = c
    return out


def _width(units: list[Unit]) -> float:
    return sum(_cw(u[0]) for u in _trim_units(units))


def _clauses(piece: list[Unit]) -> list[list[Unit]]:
    """A sentence cut at comma-like punctuation and at short pauses."""
    out: list[list[Unit]] = []
    cur: list[Unit] = []
    n = len(piece)
    for i, u in enumerate(piece):
        cur.append(u)
        if i + 1 < n:
            nxt = piece[i + 1]
            brk = ((u[0] in _WEAK or u[0] in _STRONG) and nxt[0] not in _CLOSERS and not _is_punct(nxt[0])) \
                or (nxt[1] - u[2] >= CLAUSE_PAUSE)
            if brk:
                out.append(cur)
                cur = []
    if cur:
        out.append(cur)
    return out


def _split_units(units: list[Unit], max_chars: float) -> list[list[Unit]]:
    """Sentence ends and long pauses always start a new cue; inside a sentence every clause is its own cue (a clause
    narrower than ``PHRASE_MIN`` joins its neighbour); a clause wider than ``max_chars`` is split further."""
    n = len(units)
    pieces: list[list[Unit]] = []
    cur: list[Unit] = []
    for i, u in enumerate(units):
        cur.append(u)
        if i + 1 < n:
            nxt = units[i + 1]
            hard = (_is_strong(units, i) and nxt[0] not in _CLOSERS and not _is_punct(nxt[0])) \
                or (nxt[1] - u[2] >= HARD_PAUSE)
            if hard:
                pieces.append(cur)
                cur = []
    if cur:
        pieces.append(cur)
    out: list[list[Unit]] = []
    for piece in pieces:
        groups: list[list[Unit]] = []
        for c in _clauses(piece):
            if not _trim_units(c):
                continue
            if groups and _width(groups[-1]) < PHRASE_MIN and _width(groups[-1]) + _width(c) <= max_chars + 1e-9:
                groups[-1] = groups[-1] + c
            else:
                groups.append(c)
        if len(groups) >= 2 and _width(groups[-1]) < PHRASE_MIN \
                and _width(groups[-2]) + _width(groups[-1]) <= max_chars + 1e-9:
            groups[-2:] = [groups[-2] + groups[-1]]
        for g in groups:
            g = _trim_units(g)
            if not g:
                continue
            if _width(g) <= max_chars + 1e-9:
                out.append(g)
            else:
                out.extend(c for c in (_trim_units(p2) for p2 in _balanced_split(g, max_chars)) if c)
    return out


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """CJK-aware wrapping of plain text into lines of at most ``max_chars`` full-width characters."""
    units = _proportional(text.strip(), 0.0, 0.0)
    return [c for c in (_clean("".join(u[0] for u in ch)) for ch in _split_units(units, max(2, max_chars))) if c]


# ----------------------------------------------------------------------------- source -> timeline
def _merge_contiguous(pieces: Iterable[Piece]) -> list[Piece]:
    """Pieces that continue each other on the timeline *and* in the source (e.g. a clip split for a punch-in) are one."""
    out: list[Piece] = []
    for p in sorted(pieces, key=lambda q: (q.start, q.src_in)):
        if p.duration <= 0:
            continue
        if out:
            q = out[-1]
            if (q.asset_id == p.asset_id and abs(q.speed - p.speed) < 1e-9 and abs(p.src_in - q.src_out) < 1e-3
                    and abs(p.start - q.end) < 1e-3):
                out[-1] = Piece(q.asset_id, q.src_in, p.src_out, q.start, q.speed)
                continue
        out.append(Piece(p.asset_id, p.src_in, p.src_out, p.start, p.speed or 1.0))
    return out


def _raw_cues(segments: Iterable[dict], pieces: list[Piece], max_chars: float) -> list[_Raw]:
    raw: list[_Raw] = []
    order = 0
    for seg in segments:
        units = _timed_chars(seg)
        if not units:
            continue
        sa, sb = units[0][1], units[-1][2]
        aid = seg.get("asset_id")
        for p in pieces:
            if aid is not None and p.asset_id != aid:
                continue
            if sb < p.src_in - 1e-9 or sa >= p.src_out:
                continue
            sel = [u for u in units if p.src_in <= (u[1] + u[2]) / 2.0 < p.src_out]
            if not sel:
                continue

            def tl(t: float, p: Piece = p) -> float:
                return p.start + (min(max(t, p.src_in), p.src_out) - p.src_in) / (p.speed or 1.0)

            for chunk in _split_units(sel, max_chars):
                text = _clean("".join(u[0] for u in chunk))
                if not text:
                    continue
                real = [u for u in chunk if _timing_weight(u[0]) > 0] or chunk
                raw.append(_Raw(tl(real[0][1]), tl(real[-1][2]), text, order))
                order += 1
    return raw


def _finalize(raw: list[_Raw], tl_end: float, rules: Optional[AuditRules], ids: Optional[Callable],
              fps: Optional[float]) -> list[CaptionCue]:
    rules = rules or AuditRules()
    min_cue = max(MIN_VISIBLE, float(getattr(rules, "min_cue_seconds", MIN_CUE_SECONDS) or MIN_CUE_SECONDS))
    cps = float(getattr(rules, "reading_warn_cps", 0.0) or 0.0)
    gen = ids if ids is not None else IdGen("p")
    raw = sorted(raw, key=lambda r: (r.start, r.speech_end, r.order))
    spans: list[list] = []                                # [start, end, text]
    # which cues can be shown at all, decided from the end: one with no room (overlapping speech / a cut right after)
    # is dropped -- and then no longer limits the one before it (a dropped one-word back-channel must not leave a hole)
    shown_next: list[float] = [math.inf] * len(raw)       # start of the next cue that IS shown
    keep = [False] * len(raw)
    nxt = math.inf
    for i in range(len(raw) - 1, -1, -1):
        shown_next[i] = nxt
        if min(nxt - CAP_GAP, tl_end) - raw[i].start >= MIN_VISIBLE:
            keep[i] = True
            nxt = raw[i].start
    for i, r in enumerate(raw):
        if not keep[i]:
            continue
        limit = min(shown_next[i] - CAP_GAP, tl_end)
        end = min(r.speech_end + CAP_PAD, limit)
        end = max(end, min(r.start + min_cue, limit))
        if cps > 0:
            end = max(end, min(r.start + _reading_chars(r.text) / cps, limit))
        spans.append([r.start, end, r.text])
    if fps:
        grid = 1.0 / fps
        for k, sp in enumerate(spans):
            s, e = snap_time(sp[0], fps), snap_time(sp[1], fps)
            if k + 1 < len(spans):
                e = min(e, snap_time(spans[k + 1][0], fps) - grid)
            if e <= s + 1e-9:
                e = s + grid
            sp[0], sp[1] = s, e
        spans = [sp for k, sp in enumerate(spans)
                 if k + 1 >= len(spans) or sp[1] <= snap_time(spans[k + 1][0], fps) + 1e-9]
    return [CaptionCue(id=gen("q"), start=round(s, 6), end=round(e, 6), text=t) for s, e, t in spans]


def cues_for_pieces(segments: Optional[list], pieces: list[Piece], *, max_chars: int = DEFAULT_MAX_CHARS, ids=None,
                    rules: Optional[AuditRules] = None, fps: Optional[float] = None) -> list[CaptionCue]:
    """Source-time ``segments`` -> timeline-time caption cues (spec 3.10).

    * source -> timeline goes through ``pieces``; text inside a removed range is dropped, a segment crossing a cut is
      clipped (word timings are used when present, otherwise the text is spread evenly over the segment);
    * every cue is at most ``max_chars`` full-width characters, broken at sentence ends / pauses / punctuation first;
    * ``end = min(speech end + CAP_PAD, next start - CAP_GAP)``, at least ``rules.min_cue_seconds`` and long enough for
      ``rules.reading_warn_cps`` as far as the next cue allows;
    * a segment dict may carry ``"asset_id"`` to bind it to the pieces of that asset.
    ``fps`` (optional) snaps all times to the frame grid.
    """
    if not segments or not pieces:
        return []
    merged = _merge_contiguous(pieces)
    if not merged:
        return []
    raw = _raw_cues(segments, merged, float(max(2, max_chars)))
    return _finalize(raw, max(p.end for p in merged), rules, ids, fps)


# ----------------------------------------------------------------------------------- source segments
def source_segments(asset, ev, mode: str, caption_file: str = "", *, language: str = "zh", traditional: bool = True,
                    notes: Optional[list] = None, progress=None, cancel=None, model_size: str = "small",
                    allow_download: bool = False) -> Optional[list[dict]]:
    """Caption source in SOURCE seconds: ``[{"start","end","text","words"?}]`` or ``None`` (no captions possible).

    ``auto``  the evidence's transcript, else a fresh transcription (``SpeechUnavailable`` -> note + ``None``);
    ``file``  a ``.srt/.ass/.vtt`` file (times are source times);  ``off`` -> ``None``.
    Reasons for ``None`` / degradations (繁體中文) are appended to ``notes``.
    """
    notes = notes if notes is not None else []
    mode = (mode or "off").lower()
    if mode == "off":
        return None
    if mode == "file":
        if not caption_file:
            notes.append("字幕來源設為「字幕檔」但沒有指定檔案，所以沒有加字幕。")
            return None
        try:
            cues = load_subtitle_file(caption_file)
        except (OSError, ValueError, UnicodeError) as exc:
            notes.append(f"讀取字幕檔失敗（{Path(str(caption_file)).name}）：{exc}")
            return None
        segs = [{"start": float(a), "end": float(b), "text": str(t)} for a, b, t in cues]
        if not segs:
            notes.append(f"字幕檔裡沒有可用的字幕：{Path(str(caption_file)).name}")
            return None
        return segs
    # auto
    if ev is not None and getattr(ev, "speech", None) is not None:
        return [dict(s) for s in ev.speech]
    has_sound = bool(getattr(asset, "has_audio", False)) or getattr(asset, "kind", "") == "audio"
    if not has_sound:
        notes.append("素材沒有聲音，無法用語音辨識產生字幕。")
        return None
    from ..analysis import speech as _speech
    cb = None
    if progress is not None:
        cb = lambda f: _report(progress, f, "語音辨識…")                     # noqa: E731
    try:
        segs = _speech.transcribe(asset.path, language=language, model_size=model_size, traditional=traditional,
                                  allow_download=allow_download, progress=cb, cancel=cancel)
    except Cancelled:
        raise
    except _speech.SpeechUnavailable as exc:
        notes.append(str(exc))
        return None
    except Exception as exc:                                                   # a broken install must not break planning
        notes.append(f"語音辨識失敗：{exc}")
        return None
    notes.extend(str(n) for n in getattr(segs, "notes", []))
    return [dict(s) for s in segs]


# ------------------------------------------------------------------------- captions for a sequence
def _audible_pieces(proj, seq) -> list[Piece]:
    """Timeline pieces whose sound is heard: enabled, unmuted V-track clips with audio, topmost layer first."""
    claimed: list[tuple[float, float]] = []
    out: list[Piece] = []
    for tr in reversed(seq.video_tracks):                                       # top layer first
        if tr.muted:
            continue
        for c in reversed(ordered_clips(tr.clips)):  # later start is on top in a crossfade
            a = proj.assets.get(c.asset_id)
            if (not c.enabled) or c.muted or a is None or not a.has_audio or a.kind == "image":
                continue
            parts = [(c.start, c.end)]
            for c0, c1 in claimed:
                nxt = []
                for p0, p1 in parts:
                    if c1 <= p0 or c0 >= p1:
                        nxt.append((p0, p1))
                        continue
                    if c0 > p0:
                        nxt.append((p0, c0))
                    if c1 < p1:
                        nxt.append((c1, p1))
                parts = nxt
            claimed.append((c.start, c.end))
            for p0, p1 in parts:
                if p1 - p0 < 0.05:
                    continue
                out.append(Piece(c.asset_id, c.src_in + (p0 - c.start) * c.speed, c.src_in + (p1 - c.start) * c.speed,
                                 p0, c.speed or 1.0))
    out.sort(key=lambda p: (p.start, p.asset_id))
    return out


def generate_cues_for_sequence(proj, seq, progress=None, cancel=None, *, language: str = "zh", traditional: bool = True,
                               model_size: str = "small", allow_download: bool = False) -> list[CaptionCue]:
    """Transcribe the audible V-track clips of ``seq`` and return caption cues on the sequence's timeline (spec 7).

    Raises ``SpeechUnavailable`` (message in 繁體中文) when speech recognition cannot run, ``Cancelled`` on cancel.
    The caller writes the result with ``ops.replace_cues``.
    """
    from ..analysis import speech as _speech
    ok, why = _speech.speech_available()
    if not ok:
        raise _speech.SpeechUnavailable(why)
    pieces = _audible_pieces(proj, seq)
    if not pieces:
        return []
    ranges: dict[str, list[list[float]]] = {}
    for p in pieces:
        ranges.setdefault(p.asset_id, []).append([p.src_in, p.src_out])
    jobs: list[tuple[str, float, float]] = []
    for aid, rs in ranges.items():                                              # insertion order = timeline order
        rs.sort()
        merged: list[list[float]] = [rs[0][:]]
        for lo, hi in rs[1:]:
            if lo - merged[-1][1] <= 1.5:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        dur = float(getattr(proj.assets[aid], "duration", 0.0) or 0.0)
        for lo, hi in merged:
            lo, hi = max(0.0, lo - 0.3), hi + 0.3
            if dur > 0:
                hi = min(hi, dur)
            jobs.append((aid, lo, hi))
    for aid, _lo, _hi in jobs:
        if proj.assets[aid].missing:
            raise RuntimeError(f"找不到素材檔案，無法產生字幕：{proj.assets[aid].path}")
    total = sum(max(0.0, hi - lo) for _a, lo, hi in jobs) or 1.0
    done = 0.0
    segments: list[dict] = []
    for aid, lo, hi in jobs:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        span = max(0.0, hi - lo)
        base = done / total

        def cb(f: float, base: float = base, span: float = span) -> None:
            _report(progress, min(0.99, base + (span / total) * min(max(f, 0.0), 1.0)), "語音辨識…")

        segs = _speech.transcribe(proj.assets[aid].path, language=language, model_size=model_size,
                                  traditional=traditional, allow_download=allow_download, start=lo,
                                  duration=max(span, 0.1), progress=cb, cancel=cancel)
        for s in segs:
            d = dict(s)
            d["asset_id"] = aid
            segments.append(d)
        done += span
    _report(progress, 1.0, "字幕分段…")
    max_chars = int(getattr(seq.caption_style, "max_chars", 0) or DEFAULT_MAX_CHARS)
    return cues_for_pieces(segments, pieces, max_chars=max_chars, rules=AuditRules(), fps=seq.fps)
