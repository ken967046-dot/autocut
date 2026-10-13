"""Long-video planner: one long talk video -> one tighter long video (an :class:`EditPlan` of kind ``"longform"``).

    evidence (+ transcript) -> sentence units scored for content (autoedit.highlights)
        -> keep:  mode "highlights" = the best sentences up to ``keep_ratio`` / ``target_seconds`` of the talking, plus the
                  set-up sentence of a kept payoff and one-sentence holes;  mode "clean" = everything that is talk
        -> cut:   filler words (嗯/呃/那個…) and stutters (needs word timings), long pauses (keeps ``pause_keep``)
        -> zoom:  two or more people on screen and one of them talks >= ``zoom_min_turn`` s (analysis.speakers):
                  hard cut to a close-up of the speaker; long turns go back to the wide shot at a sentence boundary
                  and punch in again on the next strong sentence ("內容決定放大多久")
        -> one sequence: V1 pieces (+ V2 empty), A1/A2 empty, optional captions, decisions that explain everything

Pure data out, deterministic (``util.IdGen``, times on the frame grid), never touches the project (spec §0).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core.plan import EditPlan
from ..core.project import Asset, CaptionStyle, Clip, Project, Sequence, Transform
from ..core.timecode import frame_of, snap_time
from ..util import IdGen
from . import highlights as hl
from . import scoring as sc
from .api import AutoEditError, Cancelled
from .captions import Piece, cues_for_pieces, source_segments
from .config import AuditRules, LongformParams

MIN_CLIP = 0.25                 # never leave a piece shorter than this
FADE = 0.015                    # every real cut gets a tiny fade (no clicks)
LEAD, TAIL = 0.12, 0.22         # a kept sentence starts this much before its first word and ends this much after
JOIN_GAP = 1.2                  # kept sentences closer than this stay one stretch (the pause is tightened instead)
SETUP_MAX = 2.5                 # a short sentence right before a kept one is kept as its set-up
MIN_SOURCE = 10.0
MIN_CLOSEUP = 1.0               # a close-up shorter than this on the timeline is not worth a cut
ESTABLISH_LEFT = 2.0            # after the wide shot that opens a turn, at least this much of the turn must remain
HOT = 0.55                      # a sentence scoring this much is a highlight (close-up / tighter zoom)
FACE_TARGET = 0.42              # close-up: the face box height as a share of the frame
ZOOM_MIN, ZOOM_MAX = 1.35, 2.4
HOT_ZOOM = 1.12                 # extra punch-in for a highlight sentence
MAX_DECISIONS = 60
MODES = ("highlights", "clean")
STD_FPS = (24, 25, 30, 50, 60)


def _mmss(t: float) -> str:
    m, s = divmod(int(round(max(0.0, t))), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _f(x) -> float:
    return float(round(float(x), 4))


def _check(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _rep(cb, frac: float, msg: str) -> None:
    if cb is not None:
        cb(min(max(float(frac), 0.0), 1.0), msg)


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _out_fps(asset: Asset, p: LongformParams) -> float:
    if p.fps and p.fps > 0:
        return float(p.fps)
    f = float(asset.fps or 30.0)
    return float(min(STD_FPS, key=lambda s: abs(s - f)))


# ----------------------------------------------------------------------------------------------- units
def audio_units(feat: sc.Feat) -> list[hl.Sentence]:
    """Without any text: voiced stretches between silences are the units, scored by loudness only (rank)."""
    dur = feat.duration
    edges = [0.0]
    for a, b in feat.sil.tolist():
        edges += [a, b]
    edges.append(dur)
    out: list[hl.Sentence] = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < 0.3:
            continue
        if feat.has_voice:
            i0, i1 = int(feat.a_idx(a)), int(feat.a_idx(b))
            if i1 <= i0 or float(np.mean(feat.voiced[i0:i1])) < 0.3:
                continue
        out.append(hl.Sentence(a, b, ""))
    if out and feat.has_rms:
        loud = np.asarray([np.mean(feat.rms[int(feat.a_idx(u.start)):max(int(feat.a_idx(u.end)), int(feat.a_idx(u.start)) + 1)])
                           for u in out])
        r = hl._rank(np.round(loud, 0))
        for u, v in zip(out, r):
            u.score = round(float(v), 4)
            u.delivery = u.score
    return out


def select_units(units: list[hl.Sentence], target: float) -> tuple[set, dict]:
    """Indexes of the kept units for ``mode="highlights"``: best first up to ``target`` seconds, then set-ups of kept
    payoffs and one-unit holes between two kept units (both within 125 % of the target).  Deterministic."""
    n = len(units)
    why: dict[int, str] = {}
    if n == 0:
        return set(), why
    order = sorted(range(n), key=lambda i: (-units[i].score, units[i].start))
    chosen: set[int] = set()
    total = 0.0
    for i in order:
        if total >= target - 1e-9:
            break
        chosen.add(i)
        why[i] = "best"
        total += units[i].duration
    budget = target * 1.25
    for i in sorted(chosen):
        j = i - 1
        if j >= 0 and j not in chosen and total + units[j].duration <= budget \
                and units[i].start - units[j].end < 1.0 and units[j].score > 0 \
                and ("question" in units[j].tags or units[j].duration <= SETUP_MAX):
            chosen.add(j)
            why[j] = "setup"
            total += units[j].duration
    for i in range(1, n - 1):
        if i not in chosen and i - 1 in chosen and i + 1 in chosen and units[i].duration <= 3.0 \
                and total + units[i].duration <= budget:
            chosen.add(i)
            why[i] = "bridge"
            total += units[i].duration
    return chosen, why


def kept_ranges(units: list[hl.Sentence], chosen: set, dur: float) -> list[tuple[float, float, list[int]]]:
    """Source stretches ``(a, b, unit indexes)`` of the chosen units; neighbours closer than ``JOIN_GAP`` are joined;
    edges get ``LEAD`` / ``TAIL`` but never reach into a neighbouring unit that was not chosen."""
    out: list[tuple[float, float, list[int]]] = []
    for i in sorted(chosen):
        u = units[i]
        lo = units[i - 1].end + 0.05 if i > 0 and (i - 1) not in chosen else 0.0
        hi = units[i + 1].start - 0.05 if i + 1 < len(units) and (i + 1) not in chosen else dur
        a, b = max(u.start - LEAD, lo, 0.0), min(u.end + TAIL, hi, dur)
        a, b = max(min(a, u.start), 0.0), min(max(b, u.end), dur)          # never outside the source
        if b <= a:
            continue
        if out and a - out[-1][1] < JOIN_GAP:
            pa, _pb, idx = out[-1]
            out[-1] = (pa, max(b, out[-1][1]), idx + [i])
        else:
            out.append((a, b, [i]))
    return out


def cut_ranges(a: float, b: float, removals: list[tuple[float, float]], min_clip: float) -> list[tuple[float, float]]:
    """``a..b`` minus ``removals`` (merged, inside ``a..b``).  Unlike ``scoring.tighten_ranges`` a removal at the very
    head or tail of the stretch is honoured (a filler word that opens a kept sentence is cut too); a kept sliver shorter
    than ``min_clip`` is dropped rather than kept as a flash."""
    out: list[tuple[float, float]] = []
    pos = a
    for ra, rb in removals:
        ra, rb = max(ra, a), min(rb, b)
        if rb <= ra:
            continue
        if ra - pos >= min_clip - 1e-9:
            out.append((pos, ra))
        pos = max(pos, rb)
    if b - pos >= min_clip - 1e-9:
        out.append((pos, b))
    return out


def _merge(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# ----------------------------------------------------------------------------------------------- zoom
@dataclass
class CloseUp:
    a: float                      # source seconds
    b: float
    track: int
    zoom: float
    cx: float
    cy: float
    hot: bool = False
    why: str = ""
    turn: tuple = ()


def _frame(box: list[float], level: float, hot: bool) -> tuple[float, float, float]:
    x, y, w, h = box
    z = float(level) if level and level > 0 else FACE_TARGET / max(h, 1e-3)
    z = min(max(z, ZOOM_MIN), ZOOM_MAX) if not (level and level > 0) else min(max(z, 1.2), 3.0)
    if hot:
        z = min(z * HOT_ZOOM, 3.0)
    half = 0.5 / z
    cx = min(max(x + w / 2.0, half), 1.0 - half)
    cy = min(max(y + h / 2.0 + 0.1 / z, half), 1.0 - half)          # face in the upper-middle of the window
    return round(z, 3), round(cx, 4), round(cy, 4)


def plan_closeups(turns, units: list[hl.Sentence], sp, p: LongformParams, kept_len: Callable[[float, float], float],
                  people: Callable[[float, float], int]) -> tuple[list[CloseUp], list[dict]]:
    """Close-ups (source time) for the speaker turns, plus a decision-ish record of the turns that were skipped."""
    from ..analysis.speakers import track_box
    ups: list[CloseUp] = []
    skipped: list[dict] = []
    last_end = 0.0                                          # end of the previous close-up (0 = the opening wide shot)
    for t in turns:
        on_screen = people(t.start, t.end)
        kl = kept_len(t.start, t.end)
        if on_screen < 2:
            skipped.append({"t0": t.start, "t1": t.end, "why": "畫面上不到兩個人"})
            continue
        if kl < p.zoom_min_turn - 1e-9:
            if t.duration >= p.zoom_min_turn:
                skipped.append({"t0": t.start, "t1": t.end, "why": f"剪輯後只剩 {kl:.1f} 秒（不到 {p.zoom_min_turn:g} 秒）"})
            continue
        inside = [u for u in units if u.end > t.start + 0.05 and u.start < t.end - 0.05]
        bounds = sorted({round(min(max(u.end, t.start), t.end), 3) for u in inside if t.start < u.end < t.end})
        segs: list[tuple[float, float, bool]] = []         # (a, b, close?)
        if t.duration <= p.zoom_max + 1e-9 or not bounds:
            if t.duration <= p.zoom_max + 1e-9:
                segs = [(t.start, t.end, True)]
            else:                                           # no sentence boundaries: fixed rhythm
                a, close = t.start, True
                while a < t.end - 1e-6:
                    b = min(t.end, a + (p.zoom_max if close else p.zoom_back_min))
                    segs.append((a, b, close))
                    a, close = b, not close
        else:
            pts = [t.start] + bounds + [t.end]
            run_start, close = t.start, True
            for k in range(1, len(pts)):
                here = pts[k]
                if here >= t.end - 1e-6:
                    segs.append((run_start, t.end, close))
                    break
                nxt_unit = next((u for u in inside if u.start >= here - 0.3), None)
                nxt_hot = nxt_unit is not None and nxt_unit.score >= HOT
                run = here - run_start
                if close and run >= p.zoom_max * 0.6 and t.end - here >= p.zoom_back_min:
                    if run >= p.zoom_max or not nxt_hot:
                        segs.append((run_start, here, True))
                        run_start, close = here, False
                elif not close and run >= p.zoom_back_min and (nxt_hot or run >= p.zoom_max * 0.5) \
                        and t.end - here >= p.zoom_min_turn * 0.5:
                    segs.append((run_start, here, False))
                    run_start, close = here, True
        for a, b, close in segs:
            if not close:
                continue
            # the wide two-shot first: the video opens on it, and every change of speaker shows it for at least
            # zoom_back_min seconds before pushing in (the viewer sees who answers), preferably up to a sentence end
            earliest = last_end + p.zoom_back_min
            if a < earliest:
                later = [x for x in bounds if earliest - 1e-6 <= x < b - ESTABLISH_LEFT]
                a = later[0] if later else earliest
            if b - a < max(MIN_CLOSEUP, ESTABLISH_LEFT):
                continue
            box = track_box(sp, t.track, a, b)
            if box is None:
                continue
            hot_units = [u for u in inside if u.score >= HOT and u.start < b and u.end > a]
            hot = bool(hot_units)
            z, cx, cy = _frame(box, p.zoom_level, hot)
            why = f"講者 {chr(65 + t.track)} 連續說話 {t.duration:.1f} 秒（≥ {p.zoom_min_turn:g} 秒，畫面上有 {on_screen} 個人）"
            if t.duration > p.zoom_max:
                why += f"；超過 {p.zoom_max:g} 秒的長段在句子邊界切回全景再切回特寫"
            if hot:
                why += f"；含爆點句「{hl.quote(max(hot_units, key=lambda u: u.score).text)}」，放大多 {HOT_ZOOM:g} 倍"
            ups.append(CloseUp(a, b, t.track, z, cx, cy, hot, why, (round(t.start, 3), round(t.end, 3))))
            last_end = b
    return ups, skipped


# ----------------------------------------------------------------------------------------------- pieces
@dataclass
class _Piece:
    a: float
    b: float
    seam: bool = False                     # continues the previous piece in the source (a camera change, not a cut)
    up: Optional[CloseUp] = None
    start: float = 0.0


def split_for_closeups(pieces: list[tuple[float, float]], ups: list[CloseUp], fps: float) -> list[_Piece]:
    """Cut the kept pieces at close-up edges.  A sub-piece shorter than ``MIN_CLIP`` is not made (the edge moves to the
    piece boundary instead), a close-up shorter than ``MIN_CLOSEUP`` after cutting is dropped."""
    out: list[_Piece] = []
    for a, b in pieces:
        cuts: list[tuple[float, float, Optional[CloseUp]]] = []
        pos = a
        for u in ups:
            ua, ub = max(u.a, a), min(u.b, b)
            if ub - ua < MIN_CLOSEUP:
                continue
            ua, ub = snap_time(ua, fps), snap_time(ub, fps)
            if ua - pos < MIN_CLIP:
                ua = pos
            if b - ub < MIN_CLIP:
                ub = b
            if ua < pos - 1e-9 or ub - ua < MIN_CLOSEUP:
                continue
            if ua > pos + 1e-9:
                cuts.append((pos, ua, None))
            cuts.append((ua, ub, u))
            pos = ub
        if b - pos > 1e-9:
            cuts.append((pos, b, None))
        for k, (x, y, u) in enumerate(cuts):
            out.append(_Piece(x, y, seam=k > 0, up=u))
    return out


# ----------------------------------------------------------------------------------------------- the plan
def _validate(proj: Project, asset_id: str, p: LongformParams) -> Asset:
    asset = proj.assets.get(asset_id)
    if asset is None:
        raise AutoEditError("找不到指定的素材，請先把影片加入專案。")
    if asset.kind != "video" or not asset.has_video:
        raise AutoEditError("這個素材沒有影像，無法做長片剪輯。")
    if asset.missing:
        raise AutoEditError(f"找不到素材檔案：{asset.path}")
    if p.mode not in MODES:
        raise AutoEditError(f"不支援的長片模式：{p.mode}（可用：{'、'.join(MODES)}）")
    if p.captions not in ("auto", "file", "off"):
        raise AutoEditError(f"不支援的字幕模式：{p.captions}")
    if not (0.05 <= float(p.keep_ratio) <= 1.0):
        raise AutoEditError("保留比例需介於 5% 與 100% 之間。")
    if p.zoom_min_turn < 0.5 or p.zoom_max < p.zoom_min_turn:
        raise AutoEditError("講者放大的時間設定不合理（最長特寫不能短於觸發長度）。")
    w, h = int(p.width or asset.width or 1920), int(p.height or asset.height or 1080)
    if w < 16 or h < 16:
        raise AutoEditError(f"輸出尺寸太小：{w}x{h}")
    if asset.duration < MIN_SOURCE:
        raise AutoEditError(f"影片太短（{asset.duration:.0f} 秒），長片剪輯至少需要 {MIN_SOURCE:g} 秒。")
    return asset


def plan_longform(proj: Project, asset_id: str, params: Optional[LongformParams] = None, progress=None,
                  cancel=None) -> EditPlan:
    """Plan the long-video edit of ``asset_id`` (see the module docstring).  Raises ``AutoEditError`` / ``Cancelled``."""
    p = params or LongformParams()
    _rep(progress, 0.0, "準備分析…")
    _check(cancel)
    asset = _validate(proj, asset_id, p)
    notes: list[str] = []
    fps = _out_fps(asset, p)
    width = int(p.width or asset.width or 1920)
    height = int(p.height or asset.height or 1080)
    width, height = width - width % 2, height - height % 2
    rules = AuditRules()

    # ------------------------------------------------------------ 1. evidence + text
    from ..analysis import analyze_asset
    from ..analysis.evidence import AnalyzeParams
    want_speech = bool(p.use_speech) or p.captions == "auto"
    ap = AnalyzeParams(silence_db=float(p.silence_db), speech=want_speech, language=p.language,
                       traditional=bool(p.traditional_chinese), allow_download=False)
    zoom_on = bool(p.speaker_zoom)
    ev = analyze_asset(asset, ap, progress=(lambda f, m: _rep(progress, 0.55 * f, m)) if progress else None,
                       cancel=cancel)
    _check(cancel)
    notes.extend(str(n) for n in (ev.notes or []))
    tfile = p.transcript_file or (p.caption_file if p.captions == "file" else "")
    segs, tinfo = hl.choose_transcript(asset, ev, tfile, use_speech=bool(p.use_speech), notes=notes)
    overrides, judge_info = None, {"engine": "local"}
    if segs and p.highlight_engine != "local" and p.mode == "highlights":
        from .llm_judge import judge_sentences
        overrides, judge_info = judge_sentences(hl.sentences(segs), engine=p.highlight_engine, language=p.language,
                                                notes=notes, cancel=cancel)
    feat = sc.Feat(ev, segs, overrides)
    word_units = hl.sentences(hl.word_segments(ev, asset))
    if not feat.words:
        feat.words = sc.transcript_words(hl.word_segments(ev, asset))
    dur = float(ev.duration or asset.duration)
    units = feat.units if feat.units else audio_units(feat)
    # a subtitle file may run past the end of the video (wrong file, trailing credits): clip it to the source
    units = [u for u in units if u.start < dur - 0.05]
    for u in units:
        u.end = min(u.end, dur)
        u.words = [w for w in u.words if w[0] < dur]
    if not feat.units:
        notes.append("沒有文字可以判斷重點：以有人聲的段落為單位、依響度挑選（建議先產生字幕再剪）。")
    if not units:
        raise AutoEditError("影片裡找不到有人說話的段落，無法做長片剪輯。")
    talk = sum(u.duration for u in units)

    # ------------------------------------------------------------ 2. what to keep
    _rep(progress, 0.6, "挑選重點…")
    if p.mode == "clean":
        chosen = set(range(len(units)))
        why_kept = {i: "clean" for i in chosen}
        ranges = [(max(0.0, units[0].start - 0.3), min(dur, units[-1].end + 0.4), list(range(len(units))))]
        target = talk
    else:
        target = float(p.target_seconds) if p.target_seconds > 0 else float(p.keep_ratio) * talk
        target = min(max(target, 1.0), talk)
        chosen, why_kept = select_units(units, target)
        ranges = kept_ranges(units, chosen, dur)
    _check(cancel)

    # ------------------------------------------------------------ 3. what to cut inside
    removals: list[tuple[float, float]] = []
    fillers: list[tuple[float, float, str]] = []
    stutters: list[tuple[float, float, str]] = []
    if p.remove_fillers:
        if word_units:
            fillers = hl.filler_spans(word_units)
        else:
            notes.append("沒有逐字時間（需要語音辨識），無法精準剪掉「嗯、呃、那個」這類水詞；只剪停頓與整句都是水詞的字幕。")
            fillers = [(u.start - 0.02, u.end + 0.02, u.text) for u in units if u.filler_ratio >= 0.99]
    if p.remove_stutters and word_units:
        stutters = hl.repeated_word_spans(word_units)
    pieces_src: list[tuple[float, float]] = []
    cut_stats = {"pause": 0.0, "filler": 0.0, "stutter": 0.0}
    for a, b, _idx in ranges:
        rem: list[tuple[float, float]] = []
        if p.tighten:
            rem += sc.silence_removals(feat.sil.tolist(), a, b, float(p.pause_min), float(p.pause_keep), feat.words)
        rem += [(max(x, a), min(y, b)) for x, y, _w in fillers + stutters if y > a and x < b]
        rem = [(x, y) for x, y in _merge(rem) if y - x >= 0.06]
        kept = cut_ranges(a, b, rem, MIN_CLIP)
        for x, y in kept:
            x2, y2 = snap_time(x, fps), snap_time(y, fps)
            if y2 - x2 >= MIN_CLIP - 1e-9:
                pieces_src.append((x2, y2))
    pieces_src = [(a, b) for a, b in _merge(pieces_src) if b - a > 1e-9]
    if not pieces_src:
        raise AutoEditError("剪完之後什麼都沒有留下，請提高保留比例。")

    def kept_overlap(x: float, y: float, spans) -> float:
        return sum(max(0.0, min(y, b) - max(x, a)) for a, b in spans)

    for x, y, _w in fillers:
        cut_stats["filler"] += max(0.0, (y - x) - kept_overlap(x, y, pieces_src))
    for x, y, _w in stutters:
        cut_stats["stutter"] += max(0.0, (y - x) - kept_overlap(x, y, pieces_src))
    kept_total = sum(b - a for a, b in pieces_src)
    range_total = sum(b - a for a, b, _i in ranges)
    cut_stats["pause"] = max(0.0, range_total - kept_total - cut_stats["filler"] - cut_stats["stutter"])

    # ------------------------------------------------------------ 4. speaker close-ups
    ups: list[CloseUp] = []
    skipped: list[dict] = []
    sp_info: dict = {"enabled": zoom_on}
    if zoom_on:
        _rep(progress, 0.62, "分析誰在說話…")
        from ..analysis.speakers import analyze_speakers, speaker_turns, visible_tracks
        sp = analyze_speakers(asset, progress=(lambda f: _rep(progress, 0.62 + 0.3 * f, f"分析誰在說話 {int(f * 100)}%"))
                              if progress else None, cancel=cancel)
        _check(cancel)
        notes.extend(sp.notes)
        if len(sp.tracks) < 2 and not sp.notes:
            notes.append("畫面中偵測到的人不到兩個，講者放大需要兩個人以上，不會啟用。")

        def voiced_at(t: float) -> bool:
            return bool(feat.voiced[min(int(feat.a_idx(t)), feat.n_a - 1)]) if feat.has_voice else True

        turns = speaker_turns(sp, voiced_at) if len(sp.tracks) >= 2 else []
        ups, skipped = plan_closeups(turns, units, sp, p, lambda x, y: kept_overlap(x, y, pieces_src),
                                     lambda x, y: len(visible_tracks(sp, x, y)))
        sp_info.update({"tracks": len(sp.tracks), "turns": len(turns), "closeups": len(ups),
                        "people": [t["box"] for t in sp.tracks]})
        if len(sp.tracks) >= 2 and not ups:
            notes.append(f"偵測到 {len(sp.tracks)} 個人，但沒有人連續說話超過 {p.zoom_min_turn:g} 秒且能確定是誰在說，"
                         "所以沒有放大（判斷依據是嘴部動作，側臉、遮嘴、兩人同時說話都會判斷不出來）。")
        if ups and min(asset.width or 1920, asset.height or 1080) < 1000 and max(u.zoom for u in ups) > 1.8:
            notes.append("素材解析度低於 1080p，放大 1.8 倍以上的特寫畫質會明顯變軟。")
    pieces = split_for_closeups(pieces_src, ups, fps)

    # ------------------------------------------------------------ 5. the sequence
    _rep(progress, 0.94, "整理序列…")
    ids = IdGen("p")
    pct = 100.0 * kept_total / max(dur, 1e-6)
    name = f"長片精華｜{asset.name or 'video'}" if p.mode == "highlights" else f"長片去水詞｜{asset.name or 'video'}"
    seq = Sequence.create(name, width, height, fps, video_tracks=2, audio_tracks=2, seq_id=ids("s"))
    for tr in list(seq.video_tracks) + list(seq.audio_tracks):
        tr.id = ids("t")
    seq.caption_style = CaptionStyle(size=0.05, bold=True, outline=0.006, position="bottom", margin=0.07, max_chars=22)
    v1 = seq.video_tracks[0]
    frames = 0
    for i, pc in enumerate(pieces):
        pc.start = frames / fps
        frames += frame_of(pc.b, fps) - frame_of(pc.a, fps)
    for i, pc in enumerate(pieces):
        nxt_seam = i + 1 < len(pieces) and pieces[i + 1].seam
        if pc.up is not None:
            tf = Transform(fit="cover", zoom=float(pc.up.zoom), cx=float(pc.up.cx), cy=float(pc.up.cy), keys=[])
            note = "自動剪輯：講者特寫。" + pc.up.why + "。"
        else:
            tf = Transform(fit="cover", zoom=1.0, cx=0.5, cy=0.5, keys=[])
            note = "自動剪輯：全景（多人對談視角）。" if pc.seam else ("自動剪輯：保留的段落。" if i == 0 else "自動剪輯：剪掉停頓／水詞／非重點後接續。")
        v1.clips.append(Clip(id=ids("c"), asset_id=asset.id, start=pc.start, src_in=pc.a, src_out=pc.b, speed=1.0,
                             fade_in=0.0 if pc.seam else FADE, fade_out=0.0 if nxt_seam else FADE, transform=tf,
                             label="特寫" if pc.up is not None else "", note=note))
    total_len = frames / fps
    cap_segments: Optional[list] = None
    if p.captions == "auto":
        cap_segments = [dict(s) for s in ev.speech] if ev.speech is not None else (segs or None)
        if cap_segments is None:
            notes.append("沒有逐字稿也沒有字幕檔，所以沒有加字幕。")
    elif p.captions == "file":
        cap_segments = source_segments(asset, ev, "file", p.caption_file, language=p.language,
                                       traditional=bool(p.traditional_chinese), notes=notes, cancel=cancel)
    cues = []
    if cap_segments:
        cues = cues_for_pieces(cap_segments, [Piece(asset.id, c.src_in, c.src_out, c.start, 1.0) for c in v1.clips],
                               max_chars=seq.caption_style.max_chars, ids=ids, rules=rules, fps=fps)
    seq.cues = cues
    seq.meta = {"longform": {"asset_id": asset.id, "mode": p.mode, "kept_seconds": _f(total_len),
                             "source_seconds": _f(dur)}}
    sid = seq.id

    # ------------------------------------------------------------ 6. decisions
    decisions: list[dict] = []
    decisions.append({
        "id": ids("d"), "kind": "longform", "sequence": sid, "score": _f(kept_total / max(dur, 1e-6)),
        "source": {"asset_id": asset.id, "t0": _f(pieces_src[0][0]), "t1": _f(pieces_src[-1][1])},
        "summary": f"{_mmss(dur)} → {_mmss(total_len)}（保留 {pct:.0f}%）· {len(ranges)} 段 · {len(ups)} 個特寫",
        "why": _main_why(p, units, chosen, target, talk, feat, cut_stats, fillers, ups, tinfo),
        "evidence": {"mode": p.mode, "target_talk_seconds": _f(target), "talk_seconds": _f(talk),
                     "kept_units": len(chosen), "units": len(units), "cut": {k: _f(v) for k, v in cut_stats.items()},
                     "fillers": len(fillers), "stutters": len(stutters)}})
    if p.mode == "highlights":
        for a, b, idx in ranges[:MAX_DECISIONS]:
            best = max((units[i] for i in idx), key=lambda u: u.score)
            reasons = sorted({why_kept.get(i, "") for i in idx} - {""})
            label = {"best": "高分句", "setup": "鋪陳", "bridge": "銜接"}
            decisions.append({
                "id": ids("d"), "kind": "keep", "sequence": sid, "score": _f(best.score),
                "source": {"asset_id": asset.id, "t0": _f(a), "t1": _f(b)},
                "summary": f"{_mmss(a)}–{_mmss(b)} · {len(idx)} 句 · 最高 {best.score:.2f}",
                "why": (f"「{hl.quote(best.text, 24)}」（{best.reasons() or '語氣'}）" if best.text else "響度較高的人聲段落")
                       + "；保留原因：" + "、".join(label.get(r, r) for r in reasons) + "。"})
        if len(ranges) > MAX_DECISIONS:
            decisions.append({"id": ids("d"), "kind": "keep", "sequence": sid, "summary": f"另有 {len(ranges) - MAX_DECISIONS} 段",
                              "why": "決策太多，其餘段落不逐一列出。"})
    if fillers or stutters:
        ex = "、".join(_dedupe([w for _a, _b, w in fillers])[:6])
        decisions.append({
            "id": ids("d"), "kind": "filler", "sequence": sid,
            "summary": f"剪掉 {len(fillers)} 個水詞（{cut_stats['filler']:.1f} 秒）、{len(stutters)} 處重複字",
            "why": (f"水詞：{ex}。" if ex else "") + "「嗯、呃」一律剪；「那個、就是、然後、對」只有前後都停頓、單獨出現時才剪（「那個人」「對對方」不會被剪）；同一詞連續三次以上才算重複字。",
            "evidence": {"cuts": [[_f(a), _f(b), w] for a, b, w in (fillers + stutters)[:200]]}})
    for u in ups[:MAX_DECISIONS]:
        decisions.append({"id": ids("d"), "kind": "zoom", "sequence": sid, "score": _f(u.zoom),
                          "source": {"asset_id": asset.id, "t0": _f(u.a), "t1": _f(u.b)},
                          "summary": f"{_mmss(u.a)}–{_mmss(u.b)} · 講者 {chr(65 + u.track)} 特寫 {u.zoom:g} 倍",
                          "why": u.why + "。", "evidence": {"track": u.track, "zoom": _f(u.zoom), "cx": u.cx, "cy": u.cy,
                                                           "turn": list(u.turn)}})
    if zoom_on and not ups:
        decisions.append({"id": ids("d"), "kind": "zoom", "sequence": sid, "summary": "沒有講者特寫",
                          "why": "；".join(_dedupe([n for n in notes if "人" in n and ("放大" in n or "說話" in n)])) or
                                 "沒有符合條件的說話段落。", "evidence": {"skipped": skipped[:30]}})
    decisions.append({"id": ids("d"), "kind": "caption", "sequence": sid,
                      "summary": f"{len(cues)} 條字幕" if cues else "未加字幕",
                      "why": "字幕跟著剪輯後的片段對應到時間軸；剪掉的區間裡的字幕會丟棄。" if cues else "字幕設定為關閉或沒有字幕來源。",
                      "evidence": {"source": p.captions}})

    brief = {"kind": "longform", "mode": p.mode, "size": [width, height], "fps": _f(fps), "asset_id": asset.id,
             "keep_ratio": _f(p.keep_ratio), "target_seconds": _f(p.target_seconds), "transcript": tinfo,
             "highlight_engine": judge_info, "speakers": sp_info, "params": p.to_dict(), "analyze": ap.to_dict()}
    sources = [{"asset_id": asset.id, "name": asset.name, "fingerprint": asset.fingerprint,
                "duration": _f(asset.duration), "evidence_sha256": ev.sha256()}]
    _rep(progress, 1.0, "完成")
    return EditPlan(kind="longform", brief=brief, sources=sources, assets=[], sequences=[seq.to_dict()],
                    decisions=decisions, notes=_dedupe(notes))


def _main_why(p: LongformParams, units, chosen, target, talk, feat, cut_stats, fillers, ups, tinfo) -> str:
    src = {"file": "字幕檔", "sidecar": "影片旁的字幕檔", "speech": "語音辨識逐字稿", "none": "（沒有文字）"}.get(tinfo.get("source"), "")
    bits = []
    if p.mode == "highlights":
        bits.append(f"依{src}把內容切成 {len(units)} 句，依「爆點」分數（提問、反轉、情緒、乾貨、具體數字、語氣）挑出 {len(chosen)} 句，"
                    f"目標保留 {target:.0f} 秒說話內容（共 {talk:.0f} 秒）；被選中句子的前一句若是提問或短鋪陳一併保留，"
                    "兩個保留句中間只隔一句短句也一併保留，避免語意跳太快")
    else:
        bits.append("保留全部說話內容，只剪停頓與水詞")
    bits.append(f"停頓剪掉 {cut_stats['pause']:.1f} 秒（≥ {p.pause_min:g} 秒的停頓只留 {p.pause_keep:g} 秒）" if p.tighten else "不剪停頓")
    if p.remove_fillers:
        bits.append(f"水詞剪掉 {cut_stats['filler']:.1f} 秒（{len(fillers)} 處）")
    if p.speaker_zoom:
        bits.append(f"{len(ups)} 個講者特寫" if ups else "沒有講者特寫（原因見說明）")
    return "；".join(bits) + "。"
