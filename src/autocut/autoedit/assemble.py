"""Assemble: many raw clips -> one long video (spec §4).

    inputs --order--> analyse each (evidence) --split into shots (scene cuts, then ``max_shot``)--> drop bad shots
    --best sub-range per shot--> length budget (sqrt quotas, >= 1 shot per usable source) --> V1 layout (+ crossfades)
    --> sound (volume match, optional music) --> markers --> captions --> EditPlan(kind="assemble", one sequence)

The plan is *data*: deterministic ids (``util.IdGen("p")``), every time on the frame grid and rounded, nothing volatile,
nothing written to the project.  Same sources + same parameters -> same plan hash.

Honest limits (also written into ``plan.notes``): the thresholds below are heuristics checked on synthetic footage only; the
quality filter judges each shot against the median of *its own source*, so a source that is blurry/dark all the way through
is not flagged; shake detection is a rough rule of thumb.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from ..analysis import AnalyzeParams, Evidence, analyze_asset, refine_cut
from ..core.plan import EditPlan
from ..core.project import Asset, CaptionStyle, Clip, Marker, Project, Sequence, Transform
from ..core.timecode import snap_time
from ..media.ffmpeg import Cancelled
from ..util import IdGen, clamp, fingerprint_file, human_duration, natural_key
from .api import AutoEditError
from .config import ASPECTS, AssembleParams, AuditRules, pace_shot_range

ProgressCb = Callable[[float, str], None]

# ---- tunables (heuristic; every one of them is written into ``brief["thresholds"]``)
IMAGE_SECONDS = 3.0          # a still image is shown this long
SHOT_MARGIN = 0.3            # stay this far from a detected shot edge (flash frames of the neighbouring shot)
REFINE_NEAR = 0.2            # a window edge closer than this to a detected cut is corrected with refine_cut
BLUR_REL = 0.45              # shot median sharpness / source median sharpness below this = blurry
DARK_LEVEL, BRIGHT_LEVEL = 0.08, 0.95        # mean luma below / above = too dark / blown out
STATIC_MOTION = 0.002        # mean frame-to-frame change below this = nothing moves
SHAKY_CV, SHAKY_MIN_MOTION, SHAKY_MIN_SAMPLES = 1.0, 0.03, 8   # coefficient of variation of motion above this = shaky
SNAP_TOL = 0.5               # a window edge moves into a silence at most this far away
MAX_DROP_DECISIONS = 50      # individual ``drop`` decisions; the rest are summed up in one more
GAIN_MIN, GAIN_MAX = 0.35, 2.8               # clip.volume range when matching loudness between sources
STD_FPS = (24, 25, 30, 50, 60)
MATCH_TOL = 0.04             # |ln(source aspect / sequence aspect)| below this = the source fits the frame
W_SHARP, W_VOICE, W_FACE, W_MOTION = 0.35, 0.25, 0.20, 0.20     # window score weights (renormalised over what exists)
ORIENT_SIZE = {"landscape": (1920, 1080), "portrait": (1080, 1920), "square": (1080, 1080)}
ORIENT_LABEL = {"landscape": "橫式 16:9", "portrait": "直式 9:16", "square": "方形 1:1"}
REASON_LABEL = {"blur": "模糊", "dark": "太暗", "bright": "過曝", "static": "靜止無聲", "shaky": "晃動", "short": "太短"}
THRESHOLDS = {
    "image_seconds": IMAGE_SECONDS, "shot_margin": SHOT_MARGIN, "blur_rel": BLUR_REL, "dark_level": DARK_LEVEL,
    "bright_level": BRIGHT_LEVEL, "static_motion": STATIC_MOTION, "shaky_cv": SHAKY_CV, "shaky_min_motion": SHAKY_MIN_MOTION,
    "snap_tol": SNAP_TOL, "gain_range": [GAIN_MIN, GAIN_MAX],
}


def r6(x: float) -> float:
    return round(float(x), 6)


def _clock(t: float) -> str:
    m, sec = divmod(max(0.0, t), 60.0)
    return f"{int(m)}:{sec:04.1f}"


def _span(a: float, b: float) -> str:
    return f"{_clock(a)}–{_clock(b)}"


class _Progress:
    """Monotone progress reporter (fractions never go backwards)."""

    def __init__(self, cb: Optional[ProgressCb]) -> None:
        self.cb, self.last = cb, 0.0

    def __call__(self, frac: float, msg: str) -> None:
        self.last = max(self.last, min(1.0, float(frac)))
        if self.cb is not None:
            self.cb(self.last, msg)


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# ================================================================================== inputs, order, format
def _is_image(a: Asset) -> bool:
    return a.kind == "image"


def _created_epoch(a: Asset) -> Optional[float]:
    txt = (a.created or "").strip()
    if not txt:
        return None
    try:
        d = datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def _time_of(a: Asset) -> tuple[Optional[float], str]:
    t = _created_epoch(a)
    if t is not None:
        return t, "拍攝時間"
    try:
        return os.path.getmtime(a.path), "檔案修改時間"
    except OSError:
        return None, "無時間資訊"


def order_assets(assets: list[Asset], mode: str) -> tuple[list[Asset], dict]:
    """``(ordered, decision_evidence)``.  chronological = creation_time, else file mtime, then natural file name."""
    name_key = lambda a: natural_key((a.name or os.path.basename(a.path)).lower())      # noqa: E731
    info: list[dict] = []
    if mode == "as_given":
        out = list(assets)
    elif mode == "name":
        out = sorted(assets, key=name_key)
    else:
        times = {a.id: _time_of(a) for a in assets}
        out = sorted(assets, key=lambda a: (0 if times[a.id][0] is not None else 1, times[a.id][0] or 0.0, name_key(a)))
        for a in out:
            info.append({"asset_id": a.id, "name": a.name, "time_source": times[a.id][1],
                         "time": r6(times[a.id][0]) if times[a.id][0] is not None else None})
    if not info:
        info = [{"asset_id": a.id, "name": a.name} for a in out]
    return out, {"mode": mode, "order": info}


def _orientation(a: Asset) -> Optional[str]:
    if a.width <= 0 or a.height <= 0:
        return None
    r = a.width / a.height
    return "landscape" if r > 1.05 else "portrait" if r < 0.95 else "square"


def normalize_fps(fps: float) -> int:
    """Nearest of 24 / 25 / 30 / 50 / 60 (so 23.976 -> 24, 29.97 -> 30, 59.94 -> 60, 120 -> 60)."""
    return min(STD_FPS, key=lambda s: (abs(s - fps), s))


def resolve_format(assets: list[Asset], params: AssembleParams) -> tuple[int, int, float, dict]:
    """Sequence width, height, fps and the evidence behind them (spec §4.1)."""
    weights: dict[str, float] = {}
    for a in assets:
        o = _orientation(a)
        if o:
            weights[o] = weights.get(o, 0.0) + (IMAGE_SECONDS if _is_image(a) or a.duration <= 0 else a.duration)
    if params.aspect == "auto":
        pick = max(("landscape", "portrait", "square"), key=lambda o: (weights.get(o, 0.0), -("landscape", "portrait", "square").index(o)))
        w, h = ORIENT_SIZE[pick]
        reason = f"多數素材（依長度加權）是{ORIENT_LABEL[pick]}"
    elif params.aspect in ASPECTS:
        w, h = ASPECTS[params.aspect]
        reason = f"使用者指定 {params.aspect}"
    else:
        raise AutoEditError(f"不支援的畫面比例：{params.aspect}（可用：auto、{'、'.join(ASPECTS)}）")
    votes: dict[int, list[float]] = {}
    for a in assets:
        if a.fps > 0 and not _is_image(a):
            v = votes.setdefault(normalize_fps(a.fps), [0, 0.0])
            v[0] += 1
            v[1] += a.duration
    if params.fps and params.fps > 0:
        fps = float(clamp(params.fps, 1.0, 240.0))
        fps_reason = f"使用者指定 {fps:g} fps"
    elif votes:
        fps = float(max(votes, key=lambda k: (votes[k][0], votes[k][1], -k)))
        fps_reason = f"最多素材使用的影格率（正規化到 24/25/30/50/60）：{fps:g} fps"
    else:
        fps, fps_reason = 30.0, "素材沒有影格率資訊，使用 30 fps"
    ev = {"orientation_seconds": {k: r6(v) for k, v in sorted(weights.items())}, "fps_votes": {str(k): v[0] for k, v in sorted(votes.items())},
          "aspect_reason": reason, "fps_reason": fps_reason}
    return int(w), int(h), fps, ev


# ================================================================================== evidence features
def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float) if x is not None and len(x) else np.zeros(0)


def _prefix(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


class _Feat:
    """Numpy view of one :class:`Evidence`: per-sample features plus prefix sums, so any window scores in O(1)."""

    def __init__(self, ev: Evidence) -> None:
        self.ev = ev
        self.vf = float(ev.visual_fps or 5.0)
        self.af = float(ev.audio_fps or 20.0)
        sharp = _arr(ev.sharp)
        med = float(np.median(sharp)) if len(sharp) else 0.0
        self.rel = sharp / med if med > 1e-9 else np.ones(len(sharp))       # sharpness relative to this source's median
        self.bright, self.motion = _arr(ev.bright), _arr(ev.motion)
        nv = max(len(self.rel), len(self.bright), len(self.motion))
        found, ran = np.zeros(nv), np.zeros(nv)
        for i in range(nv):
            if i < len(ev.faces or []) and ev.faces[i]:
                found[i] = 1.0
            if i < len(ev.face_sample_mask or []) and ev.face_sample_mask[i]:
                ran[i] = 1.0
        found = np.maximum(found, 0.0)
        ran = np.maximum(ran, found)
        self.has_faces = bool(found.sum() > 0)
        self.c_found, self.c_ran = _prefix(found), _prefix(ran)
        self.c_sharp = _prefix(np.clip(self.rel, 0.0, 1.5) / 1.5)
        n = len(self.motion)
        if n > 1:                                                           # "moderate motion" = rank near the 60th percentile
            order = np.argsort(self.motion, kind="stable")
            ranks = np.empty(n)
            ranks[order] = np.arange(n) / (n - 1)
            mod = 1.0 - np.minimum(1.0, np.abs(ranks - 0.6) / 0.6)
        else:
            mod = np.full(n, 0.5)
        self.c_mod = _prefix(mod)
        self.voice, self.rms = _arr(ev.voice), _arr(ev.rms_db)
        self.c_voice = _prefix(self.voice)
        self.has_voice = bool(ev.has_audio and len(self.voice) and float(self.voice.max()) > 0.3)
        self.silences = [(float(s[0]), float(s[1])) for s in (ev.silences or []) if len(s) >= 2]

    # sample index ranges covering [a, b)
    def vrange(self, a: float, b: float, n: int) -> tuple[int, int]:
        i0 = max(0, min(n, int(math.ceil(a * self.vf - 1e-9))))
        i1 = max(i0, min(n, int(math.ceil(b * self.vf - 1e-9))))
        if i1 <= i0 and n > 0:                                              # shorter than one sample: the nearest one
            k = max(0, min(n - 1, int(round((a + b) / 2.0 * self.vf))))
            return k, k + 1
        return i0, i1

    def arange(self, a: float, b: float, n: int) -> tuple[int, int]:
        j0 = max(0, min(n, int(math.floor(a * self.af + 1e-9))))
        j1 = max(j0, min(n, int(math.ceil(b * self.af - 1e-9))))
        if j1 <= j0 and n > 0:
            k = max(0, min(n - 1, int(a * self.af)))
            return k, k + 1
        return j0, j1

    def score(self, a: float, b: float) -> tuple[float, dict]:
        """Window quality 0..1 = weighted mean of sharpness, voice, faces and moderate motion (whatever exists)."""
        parts: dict[str, float] = {}
        weights: dict[str, float] = {}
        n = len(self.c_sharp) - 1
        if n > 0:
            i0, i1 = self.vrange(a, b, n)
            if i1 > i0:
                parts["sharp"], weights["sharp"] = (self.c_sharp[i1] - self.c_sharp[i0]) / (i1 - i0), W_SHARP
                nm = len(self.c_mod) - 1
                if nm > 0:
                    m0, m1 = self.vrange(a, b, nm)
                    if m1 > m0:
                        parts["motion"], weights["motion"] = (self.c_mod[m1] - self.c_mod[m0]) / (m1 - m0), W_MOTION
                nf = len(self.c_ran) - 1
                if self.has_faces and nf > 0:
                    f0, f1 = self.vrange(a, b, nf)
                    ran = self.c_ran[f1] - self.c_ran[f0]
                    if ran > 0:
                        parts["face"], weights["face"] = (self.c_found[f1] - self.c_found[f0]) / ran, W_FACE
        if self.has_voice:
            j0, j1 = self.arange(a, b, len(self.voice))
            if j1 > j0:
                parts["voice"], weights["voice"] = (self.c_voice[j1] - self.c_voice[j0]) / (j1 - j0), W_VOICE
        tot = sum(weights.values())
        if tot <= 0:
            return 0.5, parts
        return float(sum(parts[k] * weights[k] for k in parts) / tot), {k: r6(v) for k, v in parts.items()}


def shot_quality(ev: Evidence, a: float, b: float, feat: Optional[_Feat] = None) -> tuple[str, dict]:
    """Quality filter for the stretch ``[a, b)`` of a source (spec §4.4).  Returns ``(reason, metrics)`` where reason is
    ``""`` (fine), ``dark``, ``bright``, ``blur``, ``static`` (nothing moves *and* no sound) or ``shaky``.

    Sharpness is judged against the median of the source itself; nothing can be flagged without picture evidence."""
    f = feat or _Feat(ev)
    m: dict[str, float] = {}
    if len(f.bright):
        i0, i1 = f.vrange(a, b, len(f.bright))
        m["bright"] = r6(float(np.mean(f.bright[i0:i1])))
    if len(f.rel):
        i0, i1 = f.vrange(a, b, len(f.rel))
        m["sharp_rel"] = r6(float(np.median(f.rel[i0:i1])))
    motion = np.zeros(0)
    if len(f.motion):
        i0, i1 = f.vrange(a, b, len(f.motion))
        motion = f.motion[i0:i1]
        m["motion"] = r6(float(np.mean(motion)))
        if len(motion) >= 2 and float(np.mean(motion)) > 1e-9:
            m["motion_cv"] = r6(float(np.std(motion) / np.mean(motion)))
    silent = True
    if ev.has_audio and len(f.voice):
        j0, j1 = f.arange(a, b, len(f.voice))
        voice = float(np.mean(f.voice[j0:j1])) if j1 > j0 else 0.0
        loud = float(np.percentile(f.rms[j0:j1], 90)) if len(f.rms) >= j1 > j0 else -90.0
        silent = voice < 0.05 and loud < ev.silence_thr_db + 3.0
        m["voice"] = r6(voice)
    if "bright" in m and m["bright"] < DARK_LEVEL:
        return "dark", m
    if "bright" in m and m["bright"] > BRIGHT_LEVEL:
        return "bright", m
    if "sharp_rel" in m and m["sharp_rel"] < BLUR_REL:
        return "blur", m
    if len(motion) and m["motion"] < STATIC_MOTION and float(np.max(motion)) < 5 * STATIC_MOTION and silent:
        return "static", m
    if len(motion) >= SHAKY_MIN_SAMPLES and m["motion"] >= SHAKY_MIN_MOTION and m.get("motion_cv", 0.0) > SHAKY_CV:
        return "shaky", m
    return "", m


def _reason_text(reason: str, m: dict, min_shot: float = 0.0, usable: float = 0.0) -> str:
    if reason == "blur":
        return f"畫面模糊：清晰度只有這支素材中位數的 {m.get('sharp_rel', 0) * 100:.0f}%（低於 {BLUR_REL * 100:.0f}%）"
    if reason == "dark":
        return f"畫面太暗：平均亮度 {m.get('bright', 0):.2f}（低於 {DARK_LEVEL}）"
    if reason == "bright":
        return f"畫面過曝：平均亮度 {m.get('bright', 0):.2f}（高於 {BRIGHT_LEVEL}）"
    if reason == "static":
        return "畫面幾乎沒有變化而且全程沒有聲音，沒有可用的內容"
    if reason == "shaky":
        return f"畫面晃動：動作量忽大忽小（變異係數 {m.get('motion_cv', 0):.1f}，高於 {SHAKY_CV}）"
    if reason == "short":
        return f"太短：去掉頭尾各約 {SHOT_MARGIN} 秒的緩衝後只剩 {usable:.1f} 秒，少於最短鏡頭 {min_shot:g} 秒"
    return reason


def _snap_edge(t: float, silences: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Move ``t`` (at most SNAP_TOL) to the nearest silence, staying inside ``[lo, hi]``; unchanged when none is near."""
    best, best_d = t, SNAP_TOL + 1e-9
    for s0, s1 in silences:
        inset = min(0.1, (s1 - s0) / 2.0)
        cand = clamp(t, s0 + inset, s1 - inset)
        cand = clamp(cand, lo, hi)
        d = abs(cand - t)
        if d < best_d - 1e-9:
            best, best_d = cand, d
    return best


# ================================================================================== shots and candidates
@dataclass
class _Shape:
    """Resolved shot-length policy."""
    min_shot: float        # no shot shorter than this
    lo_len: float          # typical shot length range (``pace``); windows are picked inside it
    hi_len: float
    max_chunk: float       # a continuous shot longer than this is split (0 = never)
    drop_bad: bool
    keep_all: bool         # target_seconds == 0: every usable moment, whole shots


def resolve_shape(params: AssembleParams) -> _Shape:
    pmin, pmax = pace_shot_range(params.pace)
    min_shot = max(0.1, float(params.min_shot))
    user_max = float(params.max_shot) if params.max_shot and params.max_shot > 0 else 0.0
    hi = max(min(float(pmax), user_max) if user_max else float(pmax), min_shot)
    lo = min(max(float(pmin), min_shot), hi)
    keep_all = not params.target_seconds or params.target_seconds <= 0
    # splitting long continuous shots: always when a length budget needs sub-ranges to pick from; when everything is kept,
    # only if the user asked for a maximum shot length
    chunk = max(user_max, min_shot) if user_max else (0.0 if keep_all else float(pmax))
    return _Shape(min_shot=min_shot, lo_len=lo, hi_len=hi, max_chunk=chunk, drop_bad=bool(params.drop_bad), keep_all=keep_all)


@dataclass
class _Chunk:
    a: float
    b: float
    mh: bool                       # keep SHOT_MARGIN away from the head (a real shot edge or the start of the file)
    mt: bool
    cut_head: Optional[float]      # detected scene-cut estimate at the head (None: file start / artificial split)
    cut_tail: Optional[float]


@dataclass
class _Cand:
    src: int
    lo: float                      # usable region inside the source ...
    hi: float
    s: float                       # ... and the window actually chosen inside it
    e: float
    score: float = 0.0
    parts: dict = field(default_factory=dict)
    fixed: bool = False            # still images: the length cannot change
    chunk: tuple = (0.0, 0.0)
    cut_head: Optional[float] = None
    cut_tail: Optional[float] = None

    @property
    def length(self) -> float:
        return self.e - self.s


@dataclass
class _Src:
    idx: int
    asset: Asset
    ev: Optional[Evidence]
    feat: Optional[_Feat]
    cands: list = field(default_factory=list)
    drops: list = field(default_factory=list)
    usable: float = 0.0
    volume: float = 1.0
    matched: bool = True
    fit: str = "cover"


def _split_long(a: float, b: float, max_len: float, feat: Optional[_Feat]) -> list[tuple[float, float]]:
    """Cut a long continuous shot into pieces of at most ``max_len`` (edges move into silences when someone talks)."""
    n = int(math.ceil((b - a) / max_len - 1e-9)) if max_len > 0 else 1
    if n <= 1:
        return [(a, b)]
    step = (b - a) / n
    edges = [a]
    for k in range(1, n):
        t = a + k * step
        if feat is not None and feat.has_voice:
            t = _snap_edge(t, feat.silences, t - 0.25 * step, t + 0.25 * step)
        edges.append(t)
    edges.append(b)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def chunks_for(ev: Evidence, duration: float, max_len: float, feat: Optional[_Feat] = None) -> list[_Chunk]:
    """Shots = stretches between scene cuts; shots longer than ``max_len`` are split further (spec §4.3)."""
    cuts = sorted({round(float(t), 3) for t in (ev.scenes or [])})
    cuts = [c for c in cuts if 0.2 < c < duration - 0.2]
    edges = [0.0] + cuts + [float(duration)]
    out: list[_Chunk] = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        parts = _split_long(a, b, max_len, feat)
        for j, (pa, pb) in enumerate(parts):
            first, last = j == 0, j == len(parts) - 1
            out.append(_Chunk(pa, pb, first, last, a if first and i > 0 else None,
                              b if last and i < len(edges) - 2 else None))
    return out


def _best_window(feat: _Feat, lo: float, hi: float, length: float) -> float:
    """Start of the best-scoring window of ``length`` inside ``[lo, hi]`` (earliest wins ties)."""
    step = 1.0 / feat.vf
    best_s, best_sc, k = lo, -1.0, 0
    while lo + k * step + length <= hi + 1e-9:
        s = lo + k * step
        sc, _ = feat.score(s, s + length)
        if sc > best_sc + 1e-9:
            best_s, best_sc = s, sc
        k += 1
    sc, _ = feat.score(hi - length, hi)                # the window flush with the end of the usable region
    if sc > best_sc + 1e-9:
        best_s = hi - length
    return best_s


def make_candidate(idx: int, ch: _Chunk, ev: Evidence, feat: _Feat, shape: _Shape) -> tuple[Optional[_Cand], Optional[dict]]:
    """Turn one shot into a candidate window, or into a ``drop`` record ``{a, b, reason, metrics, text}``."""
    room = (ch.b - ch.a) - shape.min_shot
    m = clamp(room / 2.0, 0.0, SHOT_MARGIN)            # shots that are barely long enough get a smaller margin
    lo, hi = ch.a + (m if ch.mh else 0.0), ch.b - (m if ch.mt else 0.0)
    usable = hi - lo
    if usable + 1e-9 < shape.min_shot:
        return None, {"a": ch.a, "b": ch.b, "reason": "short", "metrics": {"usable": r6(max(usable, 0.0))},
                      "text": _reason_text("short", {}, shape.min_shot, max(usable, 0.0))}
    if shape.drop_bad:
        reason, met = shot_quality(ev, lo, hi, feat)
        if reason:
            return None, {"a": ch.a, "b": ch.b, "reason": reason, "metrics": met, "text": _reason_text(reason, met)}
    if shape.keep_all or usable <= shape.hi_len + 1e-9:
        s, e = lo, hi
    else:
        length = max(min(usable, (shape.lo_len + shape.hi_len) / 2.0), shape.min_shot)
        s = _best_window(feat, lo, hi, length)
        e = s + length
        if feat.has_voice:                             # cut in a pause, not in the middle of a sentence
            s = _snap_edge(s, feat.silences, lo, e - shape.min_shot)
            e = _snap_edge(e, feat.silences, s + shape.min_shot, hi)
    s, e = round(max(lo, s), 3), round(min(hi, e), 3)
    sc, parts = feat.score(s, e)
    return _Cand(idx, round(lo, 3), round(hi, 3), s, e, sc, parts, False, (ch.a, ch.b), ch.cut_head, ch.cut_tail), None


def image_candidate(idx: int, ev: Optional[Evidence], shape: _Shape) -> tuple[Optional[_Cand], Optional[dict]]:
    if ev is not None and shape.drop_bad and len(ev.bright):
        reason, met = shot_quality(ev, 0.0, IMAGE_SECONDS)
        if reason in ("dark", "bright"):
            return None, {"a": 0.0, "b": IMAGE_SECONDS, "reason": reason, "metrics": met, "text": _reason_text(reason, met)}
    return _Cand(idx, 0.0, IMAGE_SECONDS, 0.0, IMAGE_SECONDS, 0.5, {}, True, (0.0, IMAGE_SECONDS)), None


# ================================================================================== budget
ADJ_TOL = 1e-3             # windows of one source whose edges meet within this are one continuous clip


def _abuts(a: _Cand, b: _Cand) -> bool:
    """``b`` starts where ``a`` ends in the same source: two halves of one continuous take."""
    return a.src == b.src and not (a.fixed or b.fixed) and abs(b.s - a.e) <= ADJ_TOL


def _runs(chosen: list[_Cand]) -> list[float]:
    """Lengths of the clips the timeline will really contain (windows that abut are merged, see ``merge_adjacent``)."""
    runs: list[float] = []
    prev: Optional[_Cand] = None
    for c in sorted(chosen, key=lambda c: (c.src, c.s)):
        if prev is not None and _abuts(prev, c):
            runs[-1] += c.length
        else:
            runs.append(c.length)
        prev = c
    return runs


def _eff(chosen: list[_Cand], xf: float) -> float:
    """Timeline length of ``chosen`` laid out in order, with crossfades eating up to ``xf`` at each junction."""
    runs = _runs(chosen)
    total = sum(runs)
    if xf > 0:
        total -= sum(min(xf, 0.5 * min(a, b)) for a, b in zip(runs, runs[1:]))
    return total


def merge_adjacent(chosen: list[_Cand]) -> list[_Cand]:
    """Windows that abut inside one source (two pieces of a long take that were both picked) become ONE clip: no crossfade
    in the middle of a take, and no frame lost to rounding each half separately.  Sorted by (source, start)."""
    out: list[_Cand] = []
    for c in sorted(chosen, key=lambda c: (c.src, c.s)):
        if out and _abuts(out[-1], c):
            p = out[-1]
            la, lb = p.length, c.length
            parts: dict = {}
            for k in sorted(set(p.parts) | set(c.parts)):
                have = [(d[k], n) for d, n in ((p.parts, la), (c.parts, lb)) if k in d]
                parts[k] = r6(sum(v * n for v, n in have) / sum(n for _, n in have))
            out[-1] = replace(p, hi=max(p.hi, c.hi), e=c.e, score=(p.score * la + c.score * lb) / (la + lb), parts=parts,
                              chunk=(p.chunk[0], c.chunk[1]), cut_tail=c.cut_tail)
        else:
            out.append(c)
    return out


def select_candidates(srcs: list[_Src], target: float, xf: float, min_len: float = 0.0) -> list[_Cand]:
    """Spec §4.6.  ``target <= 0``: everything.  Otherwise per-source quotas proportional to sqrt(usable length), at
    least one shot for every source that has usable content, then the remaining budget is filled (or an overshoot is
    trimmed) with whichever shots bring the total closest to the target, best score first."""
    pool = [c for s in srcs for c in s.cands]
    if target <= 0:
        return sorted(pool, key=lambda c: (c.src, c.s))
    live = [s for s in srcs if s.cands]
    if not live:
        return []
    weights = {s.idx: math.sqrt(max(s.usable, 1e-9)) for s in live}
    total_w = sum(weights.values())
    chosen: list[_Cand] = []
    for s in live:
        quota, acc, picked = target * weights[s.idx] / total_w, 0.0, []
        for c in sorted(s.cands, key=lambda c: (-c.score, c.s)):
            if not picked or acc + 0.5 * c.length <= quota + 1e-9:
                picked.append(c)
                acc += c.length
        chosen += picked
    eff = _eff(chosen, xf)
    picked_ids = {id(c) for c in chosen}
    for c in sorted((c for c in pool if id(c) not in picked_ids), key=lambda c: (-c.score, c.src, c.s)):
        new = _eff(chosen + [c], xf)
        if abs(new - target) < abs(eff - target) - 1e-9:
            chosen.append(c)
            eff = new
    gap = target - eff
    if gap > 1e-6 and min_len > 0:                     # exact fill: one more shot, trimmed so the total lands on the target
        taken = {id(c) for c in chosen}
        for c in sorted((c for c in pool if id(c) not in taken and not c.fixed), key=lambda c: (-c.score, c.src, c.s)):
            if c.length < min_len - 1e-9:
                continue
            total = lambda L, c=c: _eff(chosen + [replace(c, e=round(c.s + L, 6))], xf)      # noqa: E731
            if total(min_len) > target + 1e-9:
                continue                               # even the shortest extra shot would overshoot
            lo_l, hi_l = min_len, c.length
            if total(hi_l) <= target:
                lo_l = hi_l
            else:
                for _ in range(40):
                    mid = (lo_l + hi_l) / 2.0
                    lo_l, hi_l = (mid, hi_l) if total(mid) <= target else (lo_l, mid)
            chosen.append(replace(c, e=round(c.s + lo_l, 6)))
            eff = _eff(chosen, xf)
            break
    for c in sorted(list(chosen), key=lambda c: (c.score, -c.src, -c.s)):
        if eff <= target + 1e-9:
            break
        if sum(1 for x in chosen if x.src == c.src) <= 1:
            continue                                   # every usable source keeps at least one shot
        rest = [x for x in chosen if x is not c]
        new = _eff(rest, xf)
        if abs(new - target) < abs(eff - target) - 1e-9:
            chosen, eff = rest, new
    return sorted(chosen, key=lambda c: (c.src, c.s))


def fit_to_target(chosen: list[_Cand], target: float, xf: float, min_len: float, frame: float) -> None:
    """Stretch (best shots first) or shrink (weakest shots first) the chosen windows inside their usable region so the
    laid-out length lands on ``target``.  Still images keep their length."""
    if target <= 0 or not chosen:
        return
    r = target - _eff(chosen, xf)
    if abs(r) <= frame:
        return
    if r > 0:
        for c in sorted((c for c in chosen if not c.fixed), key=lambda c: (-c.score, c.src, c.s)):
            room_e, room_s = c.hi - c.e, c.s - c.lo
            take = min(r, room_e + room_s)
            if take <= 1e-9:
                continue
            de = min(take, room_e)
            c.e += de
            c.s -= min(take - de, room_s)
            r -= take
            if r <= frame:
                break
    else:
        need = -r
        for c in sorted((c for c in chosen if not c.fixed), key=lambda c: (c.score, -c.src, -c.s)):
            room = c.length - min_len
            if room <= 1e-9:
                continue
            cut = min(need, room)
            c.e -= cut
            need -= cut
            if need <= frame:
                break


def refine_edges(c: _Cand, path: str, fps: float) -> None:
    """A window edge that sits next to a detected scene cut is corrected to the exact frame with ``refine_cut`` so the
    clip never flashes a frame of the neighbouring shot (spec §3 / §4)."""
    f = fps if fps and fps > 0 else 30.0
    if c.cut_head is not None and c.s - c.cut_head < REFINE_NEAR:
        t = refine_cut(path, c.cut_head, f)
        if abs(t - c.cut_head) <= 0.6 + 1e-6:
            c.s = min(max(c.s, t), c.e - 1.0 / f)
    if c.cut_tail is not None and c.cut_tail - c.e < REFINE_NEAR:
        t = refine_cut(path, c.cut_tail, f)
        if abs(t - c.cut_tail) <= 0.6 + 1e-6:
            c.e = max(min(c.e, t), c.s + 1.0 / f)


def layout_frames(durs: list[int], xf_frames: int) -> list[int]:
    """Start frame of every clip.  Neighbours overlap by ``xf_frames`` (a crossfade), but never by more than just under
    half of the shorter clip, so at most two clips are ever visible at once; ``xf_frames == 0`` = hard cuts."""
    starts: list[int] = []
    prev_end = 0
    for i, n in enumerate(durs):
        if i == 0:
            st = 0
        else:
            cap = (min(durs[i - 1], n) - 1) // 2
            st = prev_end - max(0, min(xf_frames, cap))
        starts.append(st)
        prev_end = st + n
    return starts


# ================================================================================== the planner
_ORDERS, _TRANSITIONS, _FITS = ("chronological", "name", "as_given"), ("cut", "crossfade"), ("blur", "cover", "contain")
_CAPTION_MODES, _PACES = ("off", "auto", "file"), ("fast", "medium", "slow")


def _validate_params(p: AssembleParams) -> None:
    for value, allowed, label in ((p.order, _ORDERS, "排序方式"), (p.transition, _TRANSITIONS, "轉場"), (p.fit, _FITS, "畫面填滿方式"),
                                  (p.captions, _CAPTION_MODES, "字幕來源"), (p.pace, _PACES, "節奏")):
        if value not in allowed:
            raise AutoEditError(f"不支援的{label}：{value}（可用：{'、'.join(allowed)}）")
    try:
        p.target_seconds, p.crossfade_seconds, p.min_shot = float(p.target_seconds), float(p.crossfade_seconds), float(p.min_shot)
    except (TypeError, ValueError):
        raise AutoEditError("目標長度、交叉淡化與最短鏡頭必須是數字。") from None
    if p.target_seconds < 0 or p.crossfade_seconds < 0 or p.min_shot <= 0:
        raise AutoEditError("目標長度與交叉淡化秒數不能是負數，最短鏡頭必須大於 0。")


def _collect(proj: Project, asset_ids: list[str], notes: list[str]) -> list[Asset]:
    seen: set[str] = set()
    out: list[Asset] = []
    for aid in asset_ids:
        if aid in seen:
            continue
        seen.add(aid)
        a = proj.assets.get(aid)
        if a is None:
            raise AutoEditError(f"專案裡找不到素材（{aid}），請重新選擇要組裝的素材。")
        if a.kind == "audio" or (not a.has_video and not _is_image(a)):
            notes.append(f"已略過純音訊素材：{a.name}")
        elif a.missing:
            notes.append(f"找不到素材檔案，已略過：{a.name}")
        else:
            out.append(a)
    if not out:
        raise AutoEditError("沒有可以組裝的影片或圖片素材（純音訊檔案與找不到檔案的素材都會被略過）。")
    return out


def _matches(a: Asset, w: int, h: int) -> bool:
    if a.width <= 0 or a.height <= 0:
        return True
    return abs(math.log((a.width / a.height) / (w / h))) <= MATCH_TOL


def source_volumes(srcs: list[_Src]) -> dict[int, tuple[float, float]]:
    """``{source index: (p90 loudness dB, gain)}``: every source's p90 loudness is pulled to the median of all of them."""
    p90 = {s.idx: float(np.percentile(np.asarray(s.ev.rms_db, dtype=float), 90))
           for s in srcs if s.ev is not None and s.ev.has_audio and len(s.ev.rms_db)}
    if len(p90) < 2:
        return {i: (v, 1.0) for i, v in p90.items()}
    target = float(np.median(list(p90.values())))
    return {i: (v, round(float(clamp(10 ** ((target - v) / 20.0), GAIN_MIN, GAIN_MAX)), 3)) for i, v in p90.items()}


def _transform(src: _Src, w: int, h: int, fps: float, fit: str, s_in: float, dur: float, notes: list[str]) -> Transform:
    if src.matched:
        return Transform(fit="cover")
    if fit in ("blur", "contain"):
        return Transform(fit=fit)
    try:                                               # fit == "cover": a window of the sequence's aspect that follows the action
        from .reframe import reframe_keys
        keys, _info = reframe_keys(src.ev, src.asset, s_in, s_in + dur, w, h, mode="face", fps=fps)
    except Exception as exc:                           # no reframe module / broken evidence: centre crop
        msg = f"取景失敗，「{src.asset.name}」改用置中裁切（{type(exc).__name__}）"
        if msg not in notes:
            notes.append(msg)
        return Transform(fit="cover")
    if len(keys) <= 1:
        return Transform(fit="cover")
    return Transform(fit="cover", cx=r6(keys[0][1]), cy=r6(keys[0][2]), keys=[[r6(a), r6(b), r6(c)] for a, b, c in keys])


def _plan_music(proj: Project, params: AssembleParams, ids: IdGen, notes: list[str]) -> tuple[Optional[Asset], bool]:
    """``(music asset, is_new)``: ``is_new`` = it is not in the project yet and must travel in ``plan.assets``."""
    if not params.music_path:
        return None, False
    path = str(params.music_path)
    if not os.path.isfile(path):
        notes.append(f"找不到配樂檔案，沒有加配樂：{os.path.basename(path)}")
        return None, False
    existing = proj.find_asset_by_path(path)
    if existing is not None:
        if existing.has_audio and existing.duration > 0:
            return existing, False
        notes.append(f"配樂素材沒有可用的聲音，沒有加配樂：{existing.name}")
        return None, False
    from ..media.probe import ProbeError, probe_asset
    try:
        a = probe_asset(path)
    except (ProbeError, OSError) as exc:
        notes.append(f"讀不到配樂檔案，沒有加配樂：{exc}")
        return None, False
    if not a.has_audio or a.duration <= 0:
        notes.append(f"配樂檔案沒有可用的聲音，沒有加配樂：{a.name}")
        return None, False
    a.id = ids("a")
    return a, True


def _music_clips(music: Asset, total_frames: int, fps: float, volume: float, fade_out: float, ids: IdGen) -> list[Clip]:
    """Music repeated back to back to fill the sequence; static volume; the last clip fades out."""
    full = max(1, int(math.floor(music.duration * fps + 1e-6)))
    clips: list[Clip] = []
    pos = 0
    while pos < total_frames:
        n = min(full, total_frames - pos)
        if clips and n < max(1, int(0.25 * fps)):
            break                                       # never end with a sliver
        clips.append(Clip(id=ids("c"), asset_id=music.id, start=r6(pos / fps), src_in=0.0, src_out=r6(n / fps),
                          volume=r6(volume), label=music.name))
        pos += n
    if clips:
        clips[-1].fade_out = r6(min(max(0.0, fade_out), clips[-1].duration))
    return clips


def _caption_cues(params: AssembleParams, used: list[_Src], pieces: list, fps: float, ids: IdGen, notes: list[str],
                  cancel) -> tuple[list, str]:
    """``(cues, how)``: caption cues in timeline time for the pieces of the sources that are on the timeline."""
    from . import captions as cap
    local: list[str] = []
    segments: list[dict] = []
    videos = [s for s in used if not _is_image(s.asset)]
    file_owner: Optional[_Src] = None
    if params.captions == "file":
        stem = os.path.splitext(os.path.basename(params.caption_file or ""))[0].lower()
        by_stem = [s for s in videos if os.path.splitext(s.asset.name)[0].lower() == stem]
        file_owner = by_stem[0] if by_stem else (videos[0] if len(videos) == 1 else None)
        if file_owner is None:
            local.append("字幕檔案無法判斷屬於哪一支素材（檔名要和素材相同，或只組裝一支影片），所以沒有加字幕。")
    for s in videos:
        _check(cancel)
        if params.captions == "file" and s is not file_owner:
            continue
        segs = cap.source_segments(s.asset, s.ev, params.captions, params.caption_file, language=params.language,
                                   traditional=bool(params.traditional_chinese), notes=local, cancel=cancel)
        segments += [dict(x, asset_id=s.asset.id) for x in (segs or [])]
    for n in local:
        if n not in notes:
            notes.append(n)
    if not segments:
        return [], ""
    cues = cap.cues_for_pieces(segments, pieces, max_chars=16, ids=ids, rules=AuditRules(), fps=fps)
    how = "語音辨識" if params.captions == "auto" else f"字幕檔 {os.path.basename(params.caption_file)}"
    return cues, how


def _shot_why(src: _Src, c: _Cand, t0: float, t1: float, picked_from: float) -> str:
    a = src.asset
    if _is_image(a):
        return f"圖片素材，靜態顯示 {t1 - t0:.1f} 秒。"
    p, bits = c.parts, []
    if "sharp" in p:
        bits.append(f"畫面清晰（約為這支素材中位數清晰度的 {p['sharp'] * 1.5:.2f} 倍）")
    if "voice" in p and p["voice"] >= 0.3:
        bits.append(f"有人聲（{p['voice'] * 100:.0f}% 的時間）")
    if "face" in p and p["face"] > 0:
        bits.append(f"有人臉（{p['face'] * 100:.0f}% 的取樣）")
    if "motion" in p:
        bits.append("動作量適中" if p["motion"] >= 0.6 else "動作量偏靜或偏動")
    head = f"{a.name} {_span(t0, t1)}（{t1 - t0:.1f} 秒）"
    body = "、".join(bits) if bits else "沒有可比較的畫面或聲音證據，依時間順序取用"
    tail = f"；所在鏡頭共 {picked_from:.1f} 秒" if picked_from > t1 - t0 + 0.05 else ""
    return f"{head}：{body}；綜合分數 {c.score:.2f}{tail}。"


def plan_assemble(proj: Project, asset_ids: list[str], params: Optional[AssembleParams] = None,
                  progress: Optional[ProgressCb] = None, cancel=None) -> EditPlan:
    """Plan one long video from many raw clips (spec §4).  Reads the project, never modifies it.  Raises
    :class:`~autocut.autoedit.api.AutoEditError` (繁體中文) when nothing can be assembled and :class:`Cancelled` when
    ``cancel`` is set (polled between phases and between sources)."""
    prog = _Progress(progress)
    params = AssembleParams.from_dict((params or AssembleParams()).to_dict())      # never touch the caller's object
    _validate_params(params)
    notes: list[str] = []
    prog(0.0, "整理素材清單…")
    ordered, order_ev = order_assets(_collect(proj, list(asset_ids), notes), params.order)
    _check(cancel)

    # ---- 1. evidence
    aparams = AnalyzeParams(faces=True, speech=(params.captions == "auto"), language=params.language,
                            traditional=bool(params.traditional_chinese))
    srcs: list[_Src] = []
    n_in = len(ordered)
    for i, a in enumerate(ordered):
        _check(cancel)
        lo, hi, label = 0.02 + 0.78 * i / n_in, 0.02 + 0.78 * (i + 1) / n_in, f"（{i + 1}/{n_in}）{a.name}"
        sub = lambda f, m, lo=lo, hi=hi, label=label: prog(lo + (hi - lo) * f, f"{label}：{m}")     # noqa: E731
        try:
            ev: Optional[Evidence] = analyze_asset(a, aparams, progress=sub, cancel=cancel)
        except Cancelled:
            raise
        except RuntimeError as exc:                     # unreadable / empty file: say so and carry on without it
            if not _is_image(a):
                notes.append(f"無法分析素材「{a.name}」，已略過：{exc}")
                continue
            ev = None
            notes.append(f"圖片「{a.name}」無法分析，仍會照原樣放進去：{exc}")
        srcs.append(_Src(len(srcs), a, ev, _Feat(ev) if ev is not None else None))
    if not srcs:
        raise AutoEditError("所有素材都無法分析，沒有東西可以組裝。" + (f"（{notes[-1]}）" if notes else ""))
    _check(cancel)

    # ---- 2. sequence format
    prog(0.80, "決定輸出格式…")
    width, height, fps, fmt_ev = resolve_format([s.asset for s in srcs], params)
    for s in srcs:
        s.matched = _matches(s.asset, width, height)
        s.fit = "cover" if s.matched else params.fit

    # ---- 3. shots, quality filter, candidate windows
    prog(0.82, "切分鏡頭並檢查畫面品質…")
    shape = resolve_shape(params)
    all_drops: list[dict] = []
    for k, s in enumerate(srcs):
        _check(cancel)
        if _is_image(s.asset):
            cand, drop = image_candidate(s.idx, s.ev, shape)
            if cand is not None:
                s.cands.append(cand)
            if drop is not None:
                s.drops.append(drop)
        else:
            dur = float(s.asset.duration or (s.ev.duration if s.ev else 0.0))
            for ch in chunks_for(s.ev, dur, shape.max_chunk, s.feat):
                cand, drop = make_candidate(s.idx, ch, s.ev, s.feat, shape)
                if cand is not None:
                    s.cands.append(cand)
                if drop is not None:
                    s.drops.append(drop)
        s.usable = sum(c.hi - c.lo for c in s.cands)
        all_drops += [dict(d, src=s.idx) for d in s.drops]
        prog(0.82 + 0.06 * (k + 1) / len(srcs), "切分鏡頭並檢查畫面品質…")
    if not any(s.cands for s in srcs):
        raise AutoEditError("所有素材都被判定為不能用（模糊、太暗、靜止或太短）。可以關閉「丟棄壞段落」，或降低最短鏡頭秒數再試一次。")
    usable_total = sum(s.usable for s in srcs)

    # ---- 4. length budget
    prog(0.88, "分配長度…")
    xf_nominal = params.crossfade_seconds if params.transition == "crossfade" else 0.0
    target = 0.0 if shape.keep_all else params.target_seconds
    chosen = select_candidates(srcs, target, xf_nominal, shape.min_shot)
    fit_to_target(chosen, target, xf_nominal, shape.min_shot, 1.0 / fps)
    chosen = merge_adjacent(chosen)
    for c in chosen:
        if not c.fixed:
            refine_edges(c, srcs[c.src].asset.path, srcs[c.src].asset.fps)
    chosen.sort(key=lambda c: (c.src, c.s))
    _check(cancel)

    # ---- 5. layout on V1 (frame grid, crossfades by overlap)
    prog(0.92, "鋪排時間軸…")
    ids = IdGen("p")
    seq = Sequence.create("長片", width, height, fps, video_tracks=2, audio_tracks=2, seq_id=ids("s"))
    for t in seq.tracks():
        t.id = ids("t")
    durs: list[int] = []
    spans: list[tuple[float, float]] = []
    for c in chosen:
        a = srcs[c.src].asset
        n = max(1, int(math.floor((c.e - c.s) * fps + 1e-6)))
        s_in = c.s if c.fixed else (snap_time(c.s, a.fps) if a.fps > 0 else c.s)
        if not c.fixed:
            s_in = max(0.0, min(s_in, a.duration - n / fps)) if a.duration > n / fps else 0.0
            n = max(1, min(n, int(math.floor((a.duration - s_in) * fps + 1e-6))))
        durs.append(n)
        spans.append((r6(s_in), r6(s_in + n / fps)))
    xf_frames = int(math.floor(params.crossfade_seconds * fps + 1e-6)) if params.transition == "crossfade" else 0
    starts = layout_frames(durs, xf_frames)
    vols = source_volumes(srcs)
    v1 = seq.video_tracks[0]
    first_clip: dict[int, Clip] = {}
    shot_decisions: list[dict] = []
    for c, st, n, (s_in, s_out) in zip(chosen, starts, durs, spans):
        src = srcs[c.src]
        a = src.asset
        tf = _transform(src, width, height, fps, src.fit, s_in, n / fps, notes)
        clip = Clip(id=ids("c"), asset_id=a.id, start=r6(st / fps), src_in=s_in, src_out=s_out, speed=1.0,
                    volume=r6(vols.get(c.src, (0.0, 1.0))[1]) if params.keep_audio and not _is_image(a) else 1.0,
                    muted=(not params.keep_audio) or _is_image(a), transform=tf, label=a.name,
                    note=f"分數 {c.score:.2f}")
        v1.clips.append(clip)
        first_clip.setdefault(c.src, clip)
        shot_decisions.append({
            "id": ids("d"), "kind": "shot", "sequence": seq.id, "clip": clip.id, "score": r6(c.score),
            "source": {"asset_id": a.id, "t0": s_in, "t1": s_out},
            "summary": f"{a.name} {_span(s_in, s_out)}｜{n / fps:.1f} 秒｜分數 {c.score:.2f}",
            "why": _shot_why(src, c, s_in, s_out, c.chunk[1] - c.chunk[0]),
            "evidence": {"parts": c.parts, "chunk": [r6(c.chunk[0]), r6(c.chunk[1])], "usable": [r6(c.lo), r6(c.hi)]}})
    total_frames = max((st + n for st, n in zip(starts, durs)), default=0)
    seq.name = f"長片｜{len({c.src for c in chosen})} 支素材｜{human_duration(total_frames / fps)}"
    if height >= width:
        seq.caption_style = CaptionStyle(size=0.052, margin=0.20)

    # ---- 6. markers (first shot of each source), music, captions
    for idx, clip in sorted(first_clip.items()):
        seq.markers.append(Marker(id=ids("m"), time=clip.start, name=srcs[idx].asset.name))
    plan_assets: list[Asset] = []
    music, music_new = _plan_music(proj, params, ids, notes)
    music_clips: list[Clip] = []
    if music is not None:
        music_clips = _music_clips(music, total_frames, fps, 10 ** (params.music_gain_db / 20.0), params.fade_music_out, ids)
        seq.audio_tracks[0].clips = music_clips
        if music_new:
            plan_assets.append(music)
    prog(0.95, "產生字幕…")
    cues: list = []
    cue_how = ""
    if params.captions != "off":
        used = [srcs[i] for i in sorted({c.src for c in chosen})]
        pieces = []
        from . import captions as _cap
        for c, clip in zip(chosen, v1.clips):
            if not _is_image(srcs[c.src].asset):
                pieces.append(_cap.Piece(clip.asset_id, clip.src_in, clip.src_out, clip.start, 1.0))
        cues, cue_how = _caption_cues(params, used, pieces, fps, ids, notes, cancel)
        seq.cues = cues
        if not cues:
            notes.append("沒有產生任何字幕（沒有語音辨識結果或字幕檔）。")

    # ---- 7. internal sanity check: whatever we hand out must at least be a legal sequence
    from ..core.project import validate_sequence
    view = Project("assemble-check")
    view.assets = {s.asset.id: s.asset for s in srcs}
    view.assets.update({a.id: a for a in plan_assets})
    if music is not None and not music_new:
        view.assets[music.id] = music
    bad = [i for i in validate_sequence(view, seq) if i.level == "error"]
    if bad:
        raise AutoEditError("組裝結果沒有通過內部檢查，已取消：" + "；".join(f"{i.message}（{i.where}）" for i in bad[:3]))

    # ---- 8. decisions, sources, brief, notes
    prog(0.97, "整理計畫…")
    actual = seq.duration
    decisions: list[dict] = [{
        "id": ids("d"), "kind": "order", "sequence": seq.id,
        "summary": {"chronological": "依拍攝時間排序", "name": "依檔名排序", "as_given": "依你選取的順序"}[params.order]
                   + f"（{len(srcs)} 支素材）",
        "why": {"chronological": "先看影片內嵌的拍攝時間（creation_time），沒有就用檔案修改時間，再用自然排序的檔名決定先後。",
                "name": "依檔名的自然排序（clip2 排在 clip10 前面）。", "as_given": "照你傳入的順序，不重新排序。"}[params.order],
        "evidence": order_ev}, {
        "id": ids("d"), "kind": "format", "sequence": seq.id,
        "summary": f"輸出 {width}×{height} @ {fps:g} fps", "why": f"{fmt_ev['aspect_reason']}；{fmt_ev['fps_reason']}。", "evidence": fmt_ev}]
    for s in srcs:
        if not s.matched and s.cands:
            fit_name = {"blur": "模糊背景填滿", "cover": "裁切取景", "contain": "完整顯示（留黑邊）"}[s.fit]
            decisions.append({"id": ids("d"), "kind": "format", "sequence": seq.id,
                              "summary": f"{s.asset.name}：畫面比例與序列不同",
                              "why": f"素材是 {s.asset.width}×{s.asset.height}，序列是 {width}×{height}，使用「{fit_name}」。",
                              "evidence": {"asset_id": s.asset.id, "fit": s.fit}})
    decisions += shot_decisions
    for d in all_drops[:MAX_DROP_DECISIONS]:
        s = srcs[d["src"]]
        decisions.append({"id": ids("d"), "kind": "drop", "sequence": seq.id, "score": 0.0,
                          "source": {"asset_id": s.asset.id, "t0": r6(d["a"]), "t1": r6(d["b"])},
                          "summary": f"{s.asset.name} {_span(d['a'], d['b'])}｜丟棄（{REASON_LABEL.get(d['reason'], d['reason'])}）",
                          "why": d["text"] + "。", "evidence": {"reason": d["reason"], "metrics": d["metrics"]}})
    if len(all_drops) > MAX_DROP_DECISIONS:
        rest = all_drops[MAX_DROP_DECISIONS:]
        counts: dict[str, int] = {}
        for d in rest:
            counts[d["reason"]] = counts.get(d["reason"], 0) + 1
        decisions.append({"id": ids("d"), "kind": "drop", "sequence": seq.id, "score": 0.0,
                          "summary": f"另外還有 {len(rest)} 段被丟棄", "evidence": {"counts": counts},
                          "why": "統計：" + "、".join(f"{REASON_LABEL.get(k, k)} {v} 段" for k, v in sorted(counts.items())) + "。"})
    if len(vols) > 1:
        decisions.append({"id": ids("d"), "kind": "audio", "sequence": seq.id,
                          "summary": f"各素材音量對齊（{len(vols)} 支有聲音）",
                          "why": "把每支素材的 p90 響度拉到所有素材的中位數，音量倍率限制在 "
                                 f"{GAIN_MIN}–{GAIN_MAX}。",
                          "evidence": {srcs[i].asset.name: {"p90_db": r6(v[0]), "gain": v[1]} for i, v in sorted(vols.items())}})
    if music_clips:
        decisions.append({"id": ids("d"), "kind": "music", "sequence": seq.id, "summary": f"配樂：{music.name}",
                          "why": f"靜態音量 {params.music_gain_db:g} dB，不足的長度重複排滿，結尾淡出 {params.fade_music_out:g} 秒；"
                                 "沒有動態閃避（人聲出現時不會自動壓低配樂）。",
                          "evidence": {"asset_id": music.id, "clips": len(music_clips)}})
        notes.append("配樂是固定音量，沒有動態閃避：有人說話的段落配樂不會自動變小。")
    if cues:
        decisions.append({"id": ids("d"), "kind": "caption", "sequence": seq.id, "summary": f"字幕：{len(cues)} 條（{cue_how}）",
                          "why": "素材時間對應到剪輯後的時間軸；被剪掉的字幕丟棄、跨剪點的字幕裁切。", "evidence": {"cues": len(cues)}})

    used_seconds: dict[int, float] = {}
    for (c, _st, n, _sp) in zip(chosen, starts, durs, spans):
        used_seconds[c.src] = used_seconds.get(c.src, 0.0) + n / fps
    sources: list[dict] = []
    for s in srcs:
        fp = s.asset.fingerprint
        if not fp and os.path.isfile(s.asset.path):
            try:
                fp = fingerprint_file(s.asset.path)
            except OSError:
                fp = ""
        sources.append({"asset_id": s.asset.id, "name": s.asset.name, "kind": s.asset.kind, "fingerprint": fp,
                        "duration": r6(s.asset.duration), "evidence_sha256": s.ev.sha256() if s.ev is not None else "",
                        "usable_seconds": r6(s.usable), "used_seconds": r6(used_seconds.get(s.idx, 0.0)),
                        "dropped": len(s.drops)})

    if not shape.keep_all:
        if usable_total < params.target_seconds - 1e-6:
            notes.append(f"可用內容只有約 {usable_total:.0f} 秒，所以實際長度 {actual:.1f} 秒比目標 {params.target_seconds:.0f} 秒短。")
        elif abs(actual - params.target_seconds) > 0.1 * params.target_seconds:
            notes.append(f"目標 {params.target_seconds:.0f} 秒，實際 {actual:.1f} 秒（每個有內容的素材至少保留一個鏡頭，交叉淡化也會縮短總長）。")
    if all_drops:
        notes.append(f"共丟棄 {len(all_drops)} 段（" + "、".join(
            f"{REASON_LABEL.get(k, k)} {sum(1 for d in all_drops if d['reason'] == k)}"
            for k in sorted({d['reason'] for d in all_drops})) + "），原因都記在決策紀錄裡。")
    notes.append("挑選與丟棄的門檻是啟發式規則，只在合成素材上驗證過機制，沒有在大量真實影片上校準；請把它當成快速的第一版，再到時間軸上修。")
    brief = params.to_dict()
    brief.update({"analyze": aparams.to_dict(), "usable_seconds": r6(usable_total), "thresholds": THRESHOLDS,
                  "resolved": {"width": width, "height": height, "fps": fps, "order": params.order,
                               "shot_length": [r6(shape.lo_len), r6(shape.hi_len)], "min_shot": r6(shape.min_shot),
                               "max_chunk": r6(shape.max_chunk)},
                  "input_assets": [a.id for a in ordered]})
    prog(1.0, "組裝計畫完成")
    return EditPlan(kind="assemble", brief=brief, sources=sources, assets=[a.to_dict() for a in plan_assets],
                    sequences=[seq.to_dict()], decisions=decisions, notes=notes)
