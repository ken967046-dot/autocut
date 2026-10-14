"""Scoring and window search for the shorts planner (spec 3.2 - 3.8).

Everything is a deterministic function of the :class:`~autocut.analysis.evidence.Evidence` (plus an optional transcript):

* :func:`plan_band`            length band / target / platform dead zone (3.2)
* :func:`find_boundaries`      good places to cut: sentence ends 1.0, silence middles 0.7, scene cuts 0.5, grid 0.1 (3.3)
* :func:`search_windows`       every (start, end) pair in the length band, scored with prefix sums + numpy (3.4 / 3.5)
* :func:`select_windows`       greedy, non-overlapping, ``min_gap_between`` apart (3.6)
* :func:`silence_removals`, :func:`tighten_ranges`   jump-cut the long pauses (3.7)
* :func:`first_cut_time`       where the 2-second rule splits the first clip (3.8)

Scores are heuristics, validated on synthetic material only (see docs/AUTOEDIT_METHOD.md 9).
"""
from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .captions import _CLOSERS, _is_punct, _is_strong, _timed_chars
from .config import PLATFORM_RULES, ShortsParams, resolve_band
from .highlights import ContentIndex, merge_fragments, score_sentences, sentences
from .reframe import _faces_at, _ran

# What was SAID dominates when a transcript / subtitle file exists ("content" + "content_hook" = 52 %).  Without text
# those two are unavailable and the remaining weights are renormalised -- they keep exactly the old relative sizes.
DEFAULT_WEIGHTS: dict[str, float] = {"voice": 0.11, "energy": 0.05, "motion": 0.05, "faces": 0.05, "hook": 0.09,
                                     "complete": 0.09, "content": 0.42, "content_hook": 0.10, "length": 0.025,
                                     "quality": 0.015}
WEIGHT_LABELS = {"voice": "人聲占比", "energy": "響度", "motion": "動作量", "faces": "人臉", "hook": "開頭鉤子",
                 "complete": "起訖完整", "content": "內容爆點", "content_hook": "開頭金句", "length": "長度貼近目標",
                 "quality": "畫質"}
KIND_LABEL = {"sentence": "句末", "silence": "靜音處", "scene": "鏡頭切換處", "grid": "固定間隔點", "edge": "素材頭尾"}
KIND_WEIGHT = {"sentence": 1.0, "silence": 0.7, "scene": 0.5, "grid": 0.1, "edge": 0.7}

LEAD_IN = 0.2               # a short starts this long before the first word
TAIL_OUT = 0.3              # ... and ends this long after the last one
LOOK = 1.5                  # how far from a boundary the speech onset / offset is looked for
PUNCH_ZOOM = 1.12
MIN_SOURCE_SECONDS = 5.0
GRID_GAP = 8.0              # stretches without any stronger boundary longer than this get a 1 s grid (last resort)
MAX_PAIRS = 1_500_000
VOICED = 0.5

_KW = re.compile(
    "重點|第一|第二|第三|第四|第五|一定|千萬|秘訣|祕訣|為什麼|為甚麼|怎麼|注意|小心|關鍵|必須|最重要|其實|方法|技巧|步驟|原因|結論|建議|不要|避免|錯誤"
    r"|[0-9０-９]+(?:[.．][0-9]+)?\s*(?:個|種|步|招|點|%|％|倍|元|塊|分鐘|秒|年|天|次)"
    r"|[一二兩三四五六七八九十百千萬]+(?:個|種|步|招|點|倍|元|塊|分鐘|年|天|次)"
    r"|(?i:\b(?:first|second|third|important|key|tip|never|always|mistake|why|how)\b)")


# ====================================================================================================== band
@dataclass
class Band:
    lo: float
    hi: float
    target: float
    deadzone: Optional[tuple[float, float]] = None
    label: str = ""
    notes: list = field(default_factory=list)

    def allowed(self, length, margin: float = 0.0):
        """Boolean mask / bool: ``length`` (after tightening) is inside the band and outside the dead zone."""
        length = np.asarray(length, dtype=float)
        ok = np.logical_and(length >= self.lo + margin, length <= self.hi - margin)
        if self.deadzone:
            in_dead = np.logical_and(length >= self.deadzone[0] - margin, length <= self.deadzone[1] + margin)
            ok = np.logical_and(ok, np.logical_not(in_dead))
        return ok if ok.ndim else bool(ok)


def plan_band(p: ShortsParams) -> Band:
    """Length band, target and platform dead zone (3.2).  Lengths are measured *after* tightening."""
    lo, hi = resolve_band(p)
    rule = PLATFORM_RULES.get(p.platform, PLATFORM_RULES["ig_reels"])
    label = str(rule.get("label", p.platform))
    notes: list[str] = []
    want = float(p.target_seconds)
    target = min(max(want, lo), hi)
    if abs(target - want) > 1e-9:
        notes.append(f"目標長度 {want:g} 秒超出 {label} 的長度帶（{lo:g}–{hi:g} 秒），已改為 {target:g} 秒。")
    dz = rule.get("deadzone")
    dead: Optional[tuple[float, float]] = None
    if dz:
        d0, d1 = float(dz[0]), float(dz[1])
        zones = [(lo, min(hi, d0 - 0.001)), (max(lo, d1 + 0.001), hi)]
        zones = [(a, b) for a, b in zones if a <= b + 1e-9]
        if not zones:
            notes.append(f"{label} 的範例死區（{d0:g}–{d1:g} 秒）涵蓋整個長度帶，這次不套用死區。")
        else:
            dead = (d0, d1)
            if d0 <= target <= d1:
                best = min(zones, key=lambda z: abs(min(max(target, z[0]), z[1]) - target))
                moved = round(min(max(target, best[0]), best[1]), 3)
                notes.append(f"目標長度 {target:g} 秒落在 {label} 的範例死區（{d0:g}–{d1:g} 秒），已移到最近的允許長度 {moved:g} 秒。")
                target = moved
    return Band(lo, hi, target, dead, label, notes)


# ============================================================================================== evidence view
def _prefix(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a, dtype=np.float64)])


def _rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank 0..1 among ``x`` (ties share the middle rank; a constant array is 0.5).

    Values are rounded (9 significant decimals) first: means of a constant signal computed from prefix sums differ in
    the last bits (0.02 vs 0.0199999...), and ranking would turn that float noise into a 0.2-vs-0.8 difference -- a
    static podcast camera has an almost constant motion level."""
    n = x.size
    if n <= 1:
        return np.full(n, 0.5)
    x = np.round(np.asarray(x, dtype=np.float64), 9)
    s = np.sort(x)
    lo = np.searchsorted(s, x, side="left")
    hi = np.searchsorted(s, x, side="right")
    return ((lo + hi - 1) / 2.0) / (n - 1)


def transcript_words(segments: Optional[Sequence[dict]]) -> list[tuple[float, float]]:
    """Sorted ``(start, end)`` of every word that has a timing (empty when the transcript has no word timings)."""
    out: list[tuple[float, float]] = []
    for s in segments or []:
        for w in s.get("words") or []:
            try:
                a, b = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if b > a:
                out.append((a, b))
    out.sort()
    return out


def inside_word(t: float, words: Sequence[tuple[float, float]], margin: float = 0.03) -> bool:
    """``t`` falls strictly inside a word (cutting there would cut the word in half)."""
    if not words:
        return False
    starts = [w[0] for w in words]
    k = bisect.bisect_right(starts, t) - 1
    for j in (k, k - 1):
        if 0 <= j < len(words) and words[j][0] + margin < t < words[j][1] - margin:
            return True
    return False


class Feat:
    """Numpy view of an Evidence: arrays, prefix sums and speech onset / offset lookups."""

    def __init__(self, ev, segments: Optional[Sequence[dict]] = None, content_overrides: Optional[dict] = None) -> None:
        self.ev = ev
        self.afps = float(getattr(ev, "audio_fps", 0.0) or 20.0)
        self.vfps = float(getattr(ev, "visual_fps", 0.0) or 5.0)
        rms = np.asarray(ev.rms_db if ev.rms_db is not None else [], dtype=np.float64)
        voice = np.asarray(ev.voice if ev.voice is not None else [], dtype=np.float64)
        if voice.size == 0 and rms.size:
            voice = (rms > float(getattr(ev, "voice_thr_db", -38.0))).astype(np.float64)
        n_a = max(voice.size, rms.size)
        if voice.size < n_a:
            voice = np.concatenate([voice, np.zeros(n_a - voice.size)])
        if rms.size and rms.size < n_a:
            rms = np.concatenate([rms, np.full(n_a - rms.size, -90.0)])
        self.n_a = n_a
        self.has_voice = n_a > 0
        self.has_rms = rms.size > 0
        self.voice = voice
        self.rms = rms
        self.voiced = voice > VOICED
        self.P_voice = _prefix(voice)
        self.P_voiced = _prefix(self.voiced.astype(np.float64))
        self.P_rms = _prefix(np.where(self.voiced, rms, 0.0)) if self.has_rms else None

        motion = np.asarray(ev.motion if ev.motion is not None else [], dtype=np.float64)
        sharp = np.asarray(ev.sharp if ev.sharp is not None else [], dtype=np.float64)
        bright = np.asarray(ev.bright if ev.bright is not None else [], dtype=np.float64)
        n_v = max(motion.size, len(ev.faces or []), sharp.size, bright.size)
        self.n_v = n_v
        self.has_motion = motion.size > 0
        self.motion = motion
        self.P_motion = _prefix(motion) if motion.size else None
        self.has_sharp = sharp.size > 0
        self.has_bright = bright.size > 0
        self.sharp_median = float(np.median(sharp)) if sharp.size else 0.0
        self.P_sharp = _prefix(sharp) if sharp.size else None
        self.P_bright = _prefix(bright) if bright.size else None
        ran = np.zeros(n_v)
        has = np.zeros(n_v)
        for i in range(n_v):
            if _ran(ev, i):
                ran[i] = 1.0
                if _faces_at(ev, i):
                    has[i] = 1.0
        self.has_faces = bool(ran.sum() > 0)
        self.P_ran = _prefix(ran)
        self.P_face = _prefix(has)
        self.duration = float(ev.duration or max(n_a / self.afps, n_v / self.vfps, 0.0))

        sil = sorted((float(a), float(b)) for a, b in (ev.silences or []) if b > a)
        merged: list[list[float]] = []
        for a, b in sil:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.sil = np.asarray(merged, dtype=np.float64).reshape(-1, 2)

        # subtitle fragments (no punctuation, ~1.4 s each) are joined into clauses first (highlights.merge_fragments)
        segs = merge_fragments([s for s in (segments or []) if str(s.get("text", "")).strip()])
        self.segments = segs
        self.has_text = bool(segs)
        self.words = transcript_words(segs)
        self.seg_starts = np.asarray(sorted(float(s["start"]) for s in segs))
        self.seg_ends = np.asarray(sorted(float(s["end"]) for s in segs))
        mids, hits = [], []
        for s in segs:
            h = min(3, len(_KW.findall(str(s.get("text", "")))))
            if h:
                mids.append((float(s["start"]) + float(s["end"])) / 2.0)
                hits.append(h)
        order = np.argsort(mids) if mids else np.zeros(0, dtype=int)
        self.kw_t = np.asarray(mids, dtype=np.float64)[order] if mids else np.zeros(0)
        self.kw_cum = _prefix(np.asarray(hits, dtype=np.float64)[order]) if mids else np.zeros(1)
        # content ("爆點") of every sentence: what was said + how it was said (autoedit.highlights)
        self.units = score_sentences(sentences(segs), self, content_overrides) if segs else []
        self.content = ContentIndex(self.units)

    # ---- index helpers (vectorised)
    def a_idx(self, t):
        return np.clip(np.floor(np.asarray(t) * self.afps + 0.5).astype(np.int64), 0, self.n_a)

    def v_idx(self, t):
        return np.clip(np.ceil(np.asarray(t) * self.vfps - 1e-9).astype(np.int64), 0, self.n_v)

    def a_sum(self, P: np.ndarray, ts, te):
        return P[self.a_idx(te)] - P[self.a_idx(ts)]

    def v_sum(self, P: np.ndarray, ts, te):
        return P[self.v_idx(te)] - P[self.v_idx(ts)]

    # ---- speech onset / offset around a time
    def onset_after(self, t: float, look: float = LOOK) -> Optional[float]:
        if self.has_voice:
            a = int(max(0, round(t * self.afps)))
            b = min(self.n_a, a + int(look * self.afps) + 1)
            idx = np.flatnonzero(self.voiced[a:b]) if b > a else np.zeros(0, dtype=int)
            return (a + int(idx[0])) / self.afps if idx.size else None
        if self.seg_starts.size:
            k = int(np.searchsorted(self.seg_starts, t - 0.05, side="left"))
            if k < self.seg_starts.size and self.seg_starts[k] <= t + look:
                return float(self.seg_starts[k])
        return None

    def offset_before(self, t: float, look: float = LOOK) -> Optional[float]:
        if self.has_voice:
            b = int(min(self.n_a, round(t * self.afps)))
            a = max(0, b - int(look * self.afps))
            idx = np.flatnonzero(self.voiced[a:b]) if b > a else np.zeros(0, dtype=int)
            return (a + int(idx[-1]) + 1) / self.afps if idx.size else None
        if self.seg_ends.size:
            k = int(np.searchsorted(self.seg_ends, t + 0.05, side="right")) - 1
            if k >= 0 and self.seg_ends[k] >= t - look:
                return float(self.seg_ends[k])
        return None

    def energy_at(self, t: float, span: float = 0.25) -> float:
        """Loudness (dB) around ``t``; used to pick a quiet spot for the 2-second-rule split."""
        if not self.has_rms:
            return 0.0
        a, b = int(self.a_idx(t - span / 2)), max(int(self.a_idx(t + span / 2)), int(self.a_idx(t - span / 2)) + 1)
        return float(np.mean(self.rms[a:min(b, self.n_a)])) if b > a else 0.0


# ==================================================================================== boundaries
@dataclass
class Boundary:
    t: float                  # raw time (sentence end, silence middle, scene cut, grid point, 0 / duration)
    kind: str                 # sentence | silence | scene | grid | edge
    weight: float
    t_in: float               # where a short that STARTS here really starts (speech onset - LEAD_IN)
    t_out: float              # where a short that ENDS here really ends (speech end + TAIL_OUT)
    mid_in: bool = False      # t_in falls inside a word
    mid_out: bool = False


def sentence_ends(segments: Sequence[dict]) -> list[float]:
    """End times of sentences: every segment end plus every sentence-final punctuation inside a segment."""
    out: list[float] = []
    for seg in segments:
        units = _timed_chars(seg)
        n = len(units)
        for i, u in enumerate(units):
            if _is_strong(units, i) and (i + 1 >= n or (units[i + 1][0] not in _CLOSERS and not _is_punct(units[i + 1][0]))):
                if i + 1 < n:                                       # the last one is the segment end anyway
                    out.append(float(u[2]))
        try:
            out.append(float(seg["end"]))
        except (KeyError, TypeError, ValueError):
            pass
    return sorted(set(round(t, 3) for t in out))


def find_boundaries(feat: Feat, *, with_grid: bool = True) -> list[Boundary]:
    """Candidate cut places (3.3), deduplicated by their effective start / end times."""
    dur = feat.duration
    raw: list[tuple[float, str]] = []
    if feat.has_text:
        raw += [(t, "sentence") for t in sentence_ends(feat.segments) if 0.0 < t < dur - 1e-6]
    raw += [((float(a) + float(b)) / 2.0, "silence") for a, b in feat.sil]
    raw += [(float(t), "scene") for t in (feat.ev.scenes or []) if 0.0 < float(t) < dur]
    raw += [(0.0, "edge"), (dur, "edge")]
    raw.sort(key=lambda r: (r[0], -KIND_WEIGHT[r[1]]))
    kept: list[tuple[float, str]] = []
    for t, k in raw:                                                # drop weaker boundaries right next to a stronger one
        if kept and abs(t - kept[-1][0]) < 0.25:
            if KIND_WEIGHT[k] > KIND_WEIGHT[kept[-1][1]]:
                kept[-1] = (t, k)
            continue
        kept.append((t, k))
    if with_grid and dur > 0:
        extra: list[tuple[float, str]] = []
        for (a, _ka), (b, _kb) in zip(kept, kept[1:]):
            if b - a > GRID_GAP:
                g = math.floor(a) + 1
                while g < b - 1.0:
                    if g > a + 1.0:
                        extra.append((float(g), "grid"))
                    g += 1
        kept = sorted(kept + extra, key=lambda r: r[0])
    out: list[Boundary] = []
    seen: dict[tuple[float, float], int] = {}
    for t, k in kept:
        t_in = t_out = t
        if k in ("sentence", "silence", "edge"):
            look = 3.0 if k == "edge" else LOOK
            on, off = feat.onset_after(t, look), feat.offset_before(t, look)
            is_head = k == "edge" and t <= 1e-9
            is_tail = k == "edge" and not is_head
            if not is_tail and on is not None:                      # as a START: just before the first word
                t_in = max(on - LEAD_IN, 0.0)
                if off is not None:
                    t_in = max(t_in, off + 0.05)
            if not is_head and off is not None:                     # as an END: a little after the last word
                t_out = off + TAIL_OUT
                if on is not None:
                    t_out = min(t_out, on - 0.05)
                t_out = max(min(t_out, dur), off)
        b = Boundary(t, k, KIND_WEIGHT[k], round(float(t_in), 4), round(float(t_out), 4),
                     inside_word(t_in, feat.words), inside_word(t_out, feat.words))
        key = (round(b.t_in, 2), round(b.t_out, 2))
        if key in seen:                                             # same effective cut as a stronger/earlier boundary
            j = seen[key]
            if b.weight > out[j].weight:
                out[j] = b
            continue
        seen[key] = len(out)
        out.append(b)
    out.sort(key=lambda b: (b.t, b.kind))
    return out


# ========================================================================================== tightening
def silence_removals(silences, t0: float, t1: float, pause_min: float, pause_keep: float,
                     words: Optional[Sequence[tuple[float, float]]] = None, word_margin: float = 0.03
                     ) -> list[tuple[float, float]]:
    """Ranges to cut out of ``t0..t1``: every pause (clipped to the range) of at least ``pause_min`` keeps ``pause_keep``
    seconds (half at each end); with word timings a cut never enters a word."""
    out: list[tuple[float, float]] = []
    for a, b in silences:
        a, b = max(float(a), t0), min(float(b), t1)
        if b - a < pause_min - 1e-9 or b - a <= pause_keep + 1e-9:
            continue
        ra, rb = a + pause_keep / 2.0, b - pause_keep / 2.0
        if words:
            ra, rb = _outside_words(ra, rb, words, word_margin)
        if rb - ra >= 0.12:
            out.append((ra, rb))
    return out


def _outside_words(ra: float, rb: float, words: Sequence[tuple[float, float]], margin: float) -> tuple[float, float]:
    """The longest part of ``ra..rb`` that stays ``margin`` away from every word."""
    parts = [(ra, rb)]
    for ws, we in words:
        if we + margin <= ra or ws - margin >= rb:
            continue
        nxt = []
        for p0, p1 in parts:
            if we + margin <= p0 or ws - margin >= p1:
                nxt.append((p0, p1))
                continue
            if ws - margin > p0:
                nxt.append((p0, ws - margin))
            if we + margin < p1:
                nxt.append((we + margin, p1))
        parts = nxt
        if not parts:
            break
    if not parts:
        return ra, ra
    return max(parts, key=lambda p: (p[1] - p[0], -p[0]))


def tighten_ranges(t0: float, t1: float, removals: Sequence[tuple[float, float]], min_clip: float
                   ) -> list[tuple[float, float]]:
    """Kept source ranges after cutting ``removals`` out of ``t0..t1``; a cut that would leave a piece shorter than
    ``min_clip`` is not made."""
    kept: list[tuple[float, float]] = []
    start = t0
    for ra, rb in sorted(removals):
        if ra - start >= min_clip - 1e-9 and rb < t1:
            kept.append((start, ra))
            start = rb
    if t1 - start < min_clip - 1e-9 and kept:                       # the last cut left a sliver: undo it
        start = kept.pop()[0]
    kept.append((start, t1))
    return [(a, b) for a, b in kept if b - a > 1e-9]


SECOND_HALF_MIN = 0.6                 # the punched-in rest of a split first clip, when the clip allows it


def first_cut_time(feat: Feat, a: float, b: float, first_cut_max: float, words: Sequence[tuple[float, float]] = ()
                   ) -> Optional[float]:
    """Source time (inside the first clip ``a..b``) where the 2-second rule splits it, at most 1.9 s after ``a``, at a
    quiet moment (and between two words when word timings exist); ``None`` when the clip is already short enough."""
    limit = min(1.9, max(first_cut_max - 0.15, 0.5))
    if b - a <= first_cut_max + 1e-9:
        return None
    hi = min(a + limit, b - 0.25)                                   # the second half must stay a real clip
    lo = min(a + min(1.0, limit * 0.55), hi - 0.1)
    if hi < a + 0.5 or hi <= lo:
        return None
    if min(hi, b - SECOND_HALF_MIN) > lo:                           # ...and preferably not a split-second zoom flash
        hi = min(hi, b - SECOND_HALF_MIN)
    cand = [t for t in np.arange(lo, hi + 1e-9, 0.05)]
    if words:
        gaps = [we for _ws, we in words if lo - 1e-9 <= we <= hi + 1e-9]       # between two words
        if gaps:
            cand = sorted(set(gaps))
        else:
            cand = [t for t in cand if not inside_word(t, words)] or cand
    best = min(cand, key=lambda t: (round(feat.energy_at(t), 1), -t))
    return float(best)


# ======================================================================================== search
@dataclass
class Pool:
    """All scored windows of one source plus what is needed to explain them."""
    ts: np.ndarray
    te: np.ndarray
    tight: np.ndarray
    saved: np.ndarray
    score: np.ndarray
    I: np.ndarray
    J: np.ndarray
    parts: dict
    stats: dict
    bounds: list
    weights: dict                    # normalised weights actually used
    available: dict                  # component -> bool
    band: Band = None

    def __len__(self) -> int:
        return int(self.ts.size)

    def info(self, k: int) -> dict:
        b0, b1 = self.bounds[int(self.I[k])], self.bounds[int(self.J[k])]
        return {"t0": float(self.ts[k]), "t1": float(self.te[k]), "tight": float(self.tight[k]),
                "saved": float(self.saved[k]), "score": float(self.score[k]),
                "parts": {n: round(float(v[k]), 4) for n, v in self.parts.items()},
                "stats": {n: round(float(v[k]), 4) for n, v in self.stats.items()},
                "start_kind": b0.kind, "end_kind": b1.kind,
                "mid_word": bool(b0.mid_in or b1.mid_out)}


def _soft_up(pct: np.ndarray) -> np.ndarray:
    """Above the median = 1, below it proportionally less."""
    return np.minimum(1.0, pct / 0.5)


def search_windows(feat: Feat, params: ShortsParams, band: Band, weights: Optional[dict] = None,
                   bounds: Optional[list] = None, should_cancel: Optional[Callable[[], None]] = None) -> Pool:
    """Score every (start boundary, end boundary) pair whose tightened length is allowed by ``band`` (3.4 / 3.5)."""
    w_nom = dict(DEFAULT_WEIGHTS)
    w_nom.update({k: float(v) for k, v in (weights or {}).items() if k in DEFAULT_WEIGHTS})
    bounds = bounds if bounds is not None else find_boundaries(feat)
    t_in = np.asarray([b.t_in for b in bounds])
    t_out = np.asarray([b.t_out for b in bounds])
    empty = Pool(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64),
                 np.zeros(0, dtype=np.int64), {}, {}, bounds, {}, {}, band)
    if len(bounds) < 2:
        return empty
    tighten = bool(params.tighten)
    keep, pmin = float(params.pause_keep), float(params.pause_min)
    sa, sb = feat.sil[:, 0], feat.sil[:, 1]
    full_cut = np.where((sb - sa) >= pmin - 1e-9, np.maximum(sb - sa - keep, 0.0), 0.0) if sa.size else np.zeros(0)
    P_cut = _prefix(full_cut)
    r_est = float(full_cut.sum() / feat.duration) if (tighten and feat.duration > 0 and sa.size) else 0.0
    hi_raw = band.hi / max(0.4, 1.0 - min(0.6, r_est * 1.3))

    # ---- enumerate (start, end) pairs
    S = len(bounds)
    t_out_m = np.maximum.accumulate(t_out)                          # (already monotone in practice)
    j_lo = np.searchsorted(t_out_m, t_in + band.lo, side="left")
    j_hi = np.searchsorted(t_out_m, t_in + hi_raw, side="right")
    cnt = np.maximum(j_hi - j_lo, 0)
    total = int(cnt.sum())
    if total == 0:
        return empty
    stride = 1
    if total > MAX_PAIRS:
        stride = int(math.ceil(total / MAX_PAIRS))
        keep_start = np.zeros(S, dtype=bool)
        keep_start[::stride] = True
        cnt = np.where(keep_start, cnt, 0)
        total = int(cnt.sum())
    I = np.repeat(np.arange(S), cnt)
    offs = np.repeat(np.cumsum(cnt) - cnt, cnt)
    J = np.arange(total) - offs + np.repeat(j_lo, cnt)
    if should_cancel:
        should_cancel()
    ts, te = t_in[I], t_out[J]
    raw_len = te - ts

    # ---- tightened length (exact saving, silences clipped to the window)
    saved = np.zeros(total)
    if tighten and sa.size:
        m = sa.size
        i0 = np.searchsorted(sa, ts, side="left")
        i1 = np.searchsorted(sb, te, side="right")
        full = np.where(i1 > i0, P_cut[np.minimum(i1, m)] - P_cut[np.minimum(i0, m)], 0.0)
        k0 = i0 - 1
        k0c = np.clip(k0, 0, m - 1)
        ok0 = (k0 >= 0) & (sb[k0c] > ts)
        l0 = np.where(ok0, np.minimum(sb[k0c], te) - ts, 0.0)
        k1c = np.clip(i1, 0, m - 1)
        ok1 = (i1 < m) & (sa[k1c] < te) & (sa[k1c] >= ts)
        l1 = np.where(ok1, te - sa[k1c], 0.0)
        saved = full + np.where(l0 >= pmin - 1e-9, l0 - keep, 0.0) + np.where(l1 >= pmin - 1e-9, l1 - keep, 0.0)
        saved = np.maximum(saved, 0.0)
    tight = raw_len - saved
    ok = band.allowed(tight, margin=0.05) & (raw_len > 0)
    if not ok.any():
        return empty
    I, J, ts, te, tight, saved, raw_len = I[ok], J[ok], ts[ok], te[ok], tight[ok], saved[ok], raw_len[ok]
    n = ts.size

    # ---- components -------------------------------------------------------------------------
    parts: dict[str, np.ndarray] = {}
    stats: dict[str, np.ndarray] = {}
    avail: dict[str, bool] = {}
    tl = np.maximum(tight, 1e-6)

    if feat.has_voice:
        vt = np.maximum(feat.a_sum(feat.P_voiced, ts, te), 0.0) / feat.afps          # voiced seconds
        voice_ratio = np.clip(vt / tl, 0.0, 1.0)
        parts["voice"] = voice_ratio
        stats["voice_ratio"] = voice_ratio
        avail["voice"] = True
    elif feat.has_motion:
        mo = feat.v_sum(feat.P_motion, ts, te) / np.maximum(feat.v_idx(te) - feat.v_idx(ts), 1)
        parts["voice"] = _rank(mo)
        stats["voice_ratio"] = np.zeros(n)
        avail["voice"] = True
    else:
        avail["voice"] = False

    if feat.has_voice and feat.has_rms:
        nv = np.maximum(feat.a_sum(feat.P_voiced, ts, te), 1.0)
        loud = feat.a_sum(feat.P_rms, ts, te) / nv
        loud = np.where(feat.a_sum(feat.P_voiced, ts, te) >= 1.0, loud, -90.0)
        parts["energy"] = _rank(loud)
        stats["loud_db"] = loud
        avail["energy"] = True
    else:
        avail["energy"] = False

    if feat.has_motion:
        mo = feat.v_sum(feat.P_motion, ts, te) / np.maximum(feat.v_idx(te) - feat.v_idx(ts), 1)
        pct = _rank(mo)
        parts["motion"] = np.where(pct <= 0.85, pct, 0.85 - (pct - 0.85) * 2.33)
        stats["motion_pct"] = pct
        avail["motion"] = True
    else:
        avail["motion"] = False

    if feat.has_faces:
        nr = feat.v_sum(feat.P_ran, ts, te)
        fr = np.where(nr > 0, feat.v_sum(feat.P_face, ts, te) / np.maximum(nr, 1.0), 0.0)
        parts["faces"] = fr
        stats["face_ratio"] = fr
        avail["faces"] = True
    else:
        avail["faces"] = False

    # hook: the first 3 s of the short (computed per start, then looked up)
    starts_all = t_in
    h_on = np.zeros(S)
    h_loud = np.full(S, -90.0)
    h_mot = np.zeros(S)
    h_face = np.zeros(S)
    if feat.has_voice:
        a1 = starts_all + 1.0
        n1 = np.maximum(feat.a_idx(a1) - feat.a_idx(starts_all), 1)
        h_on = np.minimum(1.0, 2.0 * np.clip(feat.a_sum(feat.P_voice, starts_all, a1) / n1, 0.0, 1.0))
    a3 = starts_all + 3.0
    if feat.has_voice and feat.has_rms:
        nv3 = feat.a_sum(feat.P_voiced, starts_all, a3)
        h_loud = np.where(nv3 >= 1.0, feat.a_sum(feat.P_rms, starts_all, a3) / np.maximum(nv3, 1.0), -90.0)
    if feat.has_motion:
        h_mot = feat.v_sum(feat.P_motion, starts_all, a3) / np.maximum(feat.v_idx(a3) - feat.v_idx(starts_all), 1)
    if feat.has_faces:
        nr3 = feat.v_sum(feat.P_ran, starts_all, a3)
        h_face = np.where(nr3 > 0, feat.v_sum(feat.P_face, starts_all, a3) / np.maximum(nr3, 1.0), 0.0)
    hw = {"on": 0.4 if feat.has_voice else 0.0, "loud": 0.2 if (feat.has_voice and feat.has_rms) else 0.0,
          "mot": 0.2 if feat.has_motion else 0.0, "face": 0.2 if feat.has_faces else 0.0}
    hsum = sum(hw.values())
    if hsum > 0:
        hook_s = (hw["on"] * h_on + hw["loud"] * _soft_up(_rank(h_loud)) + hw["mot"] * _soft_up(_rank(h_mot))
                  + hw["face"] * _soft_up(_rank(h_face))) / hsum
        parts["hook"] = hook_s[I]
        stats["hook_onset"] = h_on[I]
        stats["hook_loud_pct"] = _rank(h_loud)[I]
        stats["hook_motion_pct"] = _rank(h_mot)[I]
        stats["hook_face"] = h_face[I]
        avail["hook"] = True
    else:
        avail["hook"] = False

    # complete: both ends strong boundaries (sentence 1.0 / silence 0.7 / scene 0.5 / grid 0.1); a cut inside a word = 0
    wb = np.asarray([b.weight for b in bounds])
    bad_in = np.asarray([b.mid_in for b in bounds])
    bad_out = np.asarray([b.mid_out for b in bounds])
    w_s = np.where(bad_in, 0.0, wb)[I]
    w_e = np.where(bad_out, 0.0, wb)[J]
    parts["complete"] = (w_s + w_e) / 2.0
    stats["start_w"], stats["end_w"] = w_s, w_e
    avail["complete"] = True

    if feat.has_text and len(feat.content):
        cw = feat.content.window(ts, te)
        # richness of the whole window (relative to this video's rich sentences) + its single best moment (absolute):
        # a short needs at least one real highlight, and more is better
        parts["content"] = np.clip(0.5 * np.minimum(cw["mean"] / feat.content.rich, 1.0) + 0.5 * cw["peak"], 0.0, 1.0)
        parts["content_hook"] = np.clip(cw["hook"], 0.0, 1.0)
        stats["content_mean"], stats["content_peak"], stats["content_hook"] = cw["mean"], cw["peak"], cw["hook"]
        hits = feat.kw_cum[np.searchsorted(feat.kw_t, te, side="right")] - feat.kw_cum[np.searchsorted(feat.kw_t, ts, side="right")] \
            if feat.kw_t.size else np.zeros(n)
        stats["kw_hits"] = hits
        avail["content"] = avail["content_hook"] = True
    else:
        avail["content"] = avail["content_hook"] = False

    parts["length"] = np.clip(1.0 - np.abs(tight - band.target) / max(band.target - band.lo, band.hi - band.target, 1.0), 0.0, 1.0)
    avail["length"] = True

    if feat.has_sharp or feat.has_bright:
        q = np.ones(n)
        qw = 0.0
        acc = np.zeros(n)
        nvis = np.maximum(feat.v_idx(te) - feat.v_idx(ts), 1)
        if feat.has_sharp and feat.sharp_median > 0:
            rel = feat.v_sum(feat.P_sharp, ts, te) / nvis / feat.sharp_median
            acc += 0.6 * np.clip(rel / 0.9, 0.0, 1.0)
            qw += 0.6
            stats["sharp_rel"] = rel
        if feat.has_bright:
            br = feat.v_sum(feat.P_bright, ts, te) / nvis
            acc += 0.4 * np.minimum(1.0, br / 0.2) * np.minimum(1.0, (1.0 - br) / 0.07 + 1e-9)
            qw += 0.4
            stats["bright"] = br
        parts["quality"] = acc / qw if qw > 0 else q
        avail["quality"] = qw > 0
    else:
        avail["quality"] = False

    used = {k: w_nom[k] for k in DEFAULT_WEIGHTS if avail.get(k) and w_nom[k] > 0}
    z = sum(used.values())
    wn = {k: v / z for k, v in used.items()} if z > 0 else {}
    score = np.zeros(n)
    for k, w in wn.items():
        score += w * np.clip(parts[k], 0.0, 1.0)
    return Pool(ts, te, tight, saved, score, I, J, parts, stats, bounds, wn, avail, band)


# ===================================================================================== selection
def select_windows(pool: Pool, count: int, min_gap: float, accept: Optional[Callable[[int], bool]] = None) -> list[int]:
    """Greedy best-first picks that do not overlap and are at least ``min_gap`` seconds apart (3.6).

    ``accept(k)`` may veto a candidate (e.g. its exact length turned out wrong after building it).
    Returns indexes into the pool, best first.
    """
    if len(pool) == 0 or count <= 0:
        return []
    order = np.lexsort((pool.te, pool.ts, -np.round(pool.score, 9)))
    picked: list[int] = []
    for k in order:
        k = int(k)
        a, b = float(pool.ts[k]), float(pool.te[k])
        if any(not (a >= float(pool.te[p]) + min_gap - 1e-9 or b <= float(pool.ts[p]) - min_gap + 1e-9) for p in picked):
            continue
        if accept is not None and not accept(k):
            continue
        picked.append(k)
        if len(picked) >= count:
            break
    return picked
