"""Shorts planner (spec 3): one long 16:9 video -> up to ``params.count`` 9:16 shorts as an :class:`EditPlan`.

``plan_shorts`` reads the project and the source's evidence and returns *data only* (it never touches the project):
one sequence per short (V1 picture + sound, V2 empty, A1 music, A2 empty), the decisions that explain every choice and
notes about everything that had to be degraded.  The plan is deterministic: ids come from ``util.IdGen``, times are on
the frame grid, nothing volatile is inside the content.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.plan import EditPlan
from ..core.project import Asset, CaptionStyle, Clip, Project, Sequence, Transform
from ..core.timecode import frame_of, snap_time
from ..util import IdGen, sha256_hex
from . import highlights as hl
from . import scoring as sc
from .api import AutoEditError, Cancelled
from .captions import Piece, _reading_chars, cues_for_pieces, source_segments
from .config import AuditRules, ShortsParams
from .reframe import reframe_keys

MIN_CLIP = 0.25             # pieces shorter than this are never cut out (spec 3.7)
FADE = 0.015                # fade in/out of every piece: no clicks at jump cuts
CORRECT_NEAR = 0.25         # a piece edge within this of an estimated scene cut is corrected with refine_cut
MIN_LOOP_SECONDS = 3.0      # the last music loop is never shorter than this (it carries the 1.5 s fade-out)
NOVOICE_RATIO = 0.15        # below this share of voice the music gets +6 dB
NOVOICE_BOOST_DB = 6.0
REFRAMES = ("face", "motion", "center", "blur")
CAPTION_MODES = ("auto", "file", "off")


# ----------------------------------------------------------------------------------------- small helpers
def _mmss(t: float) -> str:
    m, s = divmod(int(round(max(0.0, t))), 60)
    return f"{m:02d}:{s:02d}"


def _f(x) -> float:
    return float(round(float(x), 4))


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


class _Progress:
    def __init__(self, cb: Optional[Callable[[float, str], None]]) -> None:
        self.cb = cb

    def __call__(self, frac: float, msg: str) -> None:
        if self.cb is not None:
            self.cb(min(max(float(frac), 0.0), 1.0), msg)

    def sub(self, lo: float, hi: float) -> Callable[[float, str], None]:
        return lambda f, m: self(lo + (hi - lo) * min(max(float(f), 0.0), 1.0), m)


def _check(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _snap_dir(t: float, fps: float, up: bool) -> float:
    n = math.ceil(t * fps - 1e-6) if up else math.floor(t * fps + 1e-6)
    return n / fps


# --------------------------------------------------------------------------- scene-cut correction
class _CutFixer:
    """Edges that sit on a scene cut are moved onto the exact frame (analysis.refine_cut), never into the wrong shot."""

    def __init__(self, ev, asset: Asset, fps_out: float, enabled: bool = True) -> None:
        self.scenes = sorted(float(t) for t in (ev.scenes or [])) if enabled else []
        self.path = asset.path
        self.src_fps = float(ev.fps or asset.fps or 30.0)
        self.fps = fps_out
        self.cache: dict[float, float] = {}
        self.fixed = 0

    def exact(self, est: float) -> float:
        if est not in self.cache:
            from ..analysis import refine_cut
            try:
                self.cache[est] = float(refine_cut(self.path, est, self.src_fps, window=0.6))
            except Exception:                                      # refine_cut falls back by itself; belt and braces
                self.cache[est] = est
        return self.cache[est]

    def near(self, t: float) -> Optional[float]:
        best = None
        for s in self.scenes:
            if abs(s - t) <= CORRECT_NEAR and (best is None or abs(s - t) < abs(best - t)):
                best = s
        return best

    def fix_in(self, t: float) -> float:
        """A piece starts at ``t``: if the true cut is later, start exactly on it (first frame of the new shot)."""
        s = self.near(t)
        if s is None:
            return t
        r = self.exact(s)
        if r > t + 1e-6:
            self.fixed += 1
            return _snap_dir(r, self.fps, True)
        return t

    def fix_out(self, t: float) -> float:
        """A piece ends at ``t``: if the true cut is earlier, end exactly on it (last frame of the old shot)."""
        s = self.near(t)
        if s is None:
            return t
        r = self.exact(s)
        if r < t - 1e-6:
            self.fixed += 1
            return _snap_dir(r, self.fps, False)
        return t

    def exact_scenes(self, t0: float, t1: float) -> list[float]:
        return [self.exact(s) for s in self.scenes if t0 < s < t1]


# --------------------------------------------------------------------------------------- one short
@dataclass
class _P:
    a: float
    b: float
    zoom: float = 1.0
    seam: bool = False                   # contiguous with the previous piece (the 2-second-rule split)
    keys: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    start: float = 0.0


@dataclass
class _Short:
    info: dict
    pieces: list
    removals: list
    length: float
    first_cut: Optional[float]
    first_cut_src: Optional[float]
    fixed_edges: int


def _snap_ranges(ranges, fps: float) -> list[tuple[float, float]]:
    out = []
    for a, b in ranges:
        a2, b2 = snap_time(a, fps), snap_time(b, fps)
        if b2 - a2 >= 1.0 / fps - 1e-9:
            out.append((a2, b2))
    return out


def _build_short(info: dict, *, feat: sc.Feat, ev, asset: Asset, params: ShortsParams, band: sc.Band,
                 fixer: _CutFixer, check_band: bool = True) -> Optional[_Short]:
    """Turn a scored window into V1 pieces (tighten, 2-second rule, zoom alternation); ``None`` when the exact length is
    not allowed by the band."""
    fps = float(params.fps)
    t0, t1 = float(info["t0"]), float(info["t1"])
    t0 = fixer.fix_in(t0)
    t1 = fixer.fix_out(t1)
    t0, t1 = snap_time(t0, fps), snap_time(t1, fps)
    if t1 - t0 < MIN_CLIP:
        return None
    removals: list[tuple[float, float]] = []
    if params.tighten:
        removals = sc.silence_removals(feat.sil.tolist(), t0, t1, float(params.pause_min), float(params.pause_keep),
                                       feat.words)
    kept = _snap_ranges(sc.tighten_ranges(t0, t1, removals, MIN_CLIP), fps)
    if not kept:
        return None
    # interior edges that sit on a scene cut
    fixed: list[tuple[float, float]] = []
    for i, (a, b) in enumerate(kept):
        if i > 0:
            a = fixer.fix_in(a)
        if i < len(kept) - 1:
            b = fixer.fix_out(b)
        if b - a >= 0.1:
            fixed.append((snap_time(a, fps), snap_time(b, fps)))
    kept = [(a, b) for a, b in fixed if b > a]
    if not kept:
        return None
    pieces = [_P(a, b) for a, b in kept]
    # the 2-second rule: a first clip longer than first_cut_max is split with a punch-in on its second half
    first_cut_src: Optional[float] = None
    if params.reframe != "blur":
        cut = sc.first_cut_time(feat, pieces[0].a, pieces[0].b, float(params.first_cut_max), feat.words)
        if cut is not None:
            cut = snap_time(cut, fps)
            if pieces[0].a + MIN_CLIP <= cut <= pieces[0].b - MIN_CLIP:
                first = pieces[0]
                pieces[0:1] = [_P(first.a, cut), _P(cut, first.b, seam=True)]
                first_cut_src = cut
        for i, p in enumerate(pieces):                              # alternate 1.0 / 1.12 so every cut is also a re-framing
            p.zoom = 1.0 if i % 2 == 0 else sc.PUNCH_ZOOM
    frames = 0                                                      # integer frames: starts stay exactly on the grid
    for p in pieces:
        p.start = frames / fps
        frames += frame_of(p.b, fps) - frame_of(p.a, fps)
    length = frames / fps
    if check_band and not band.allowed(length):
        return None
    first_cut = None
    if len(pieces) > 1:
        first_cut = pieces[0].b - pieces[0].a
    return _Short(info, pieces, removals, length, first_cut, first_cut_src, fixer.fixed)


# ------------------------------------------------------------------------------------------ the plan
def _validate(proj: Project, asset_id: str, params: ShortsParams) -> Asset:
    asset = proj.assets.get(asset_id)
    if asset is None:
        raise AutoEditError("找不到指定的素材，請先把影片加入專案。")
    if asset.kind != "video" or not asset.has_video:
        raise AutoEditError("這個素材沒有影像，無法剪成直式短影音。")
    if asset.missing:
        raise AutoEditError(f"找不到素材檔案：{asset.path}")
    if params.reframe not in REFRAMES:
        raise AutoEditError(f"不支援的取景模式：{params.reframe}（可用：{'、'.join(REFRAMES)}）")
    if params.captions not in CAPTION_MODES:
        raise AutoEditError(f"不支援的字幕模式：{params.captions}（可用：{'、'.join(CAPTION_MODES)}）")
    if params.width < 16 or params.height < 16 or params.width % 2 or params.height % 2:
        raise AutoEditError(f"輸出尺寸需為偶數且不小於 16：{params.width}x{params.height}")
    if not (1.0 <= float(params.fps) <= 240.0):
        raise AutoEditError(f"影格率不合理：{params.fps}")
    if int(params.count) < 1:
        raise AutoEditError("短影音數量至少要 1 支。")
    if asset.duration < sc.MIN_SOURCE_SECONDS:
        raise AutoEditError("影片太短，無法剪成短影音")
    return asset


def _music_asset(proj: Project, params: ShortsParams, notes: list[str]) -> tuple[Optional[Asset], bool]:
    """``(asset, is_new)`` for the background music, or ``(None, False)`` (with a note) when it cannot be used."""
    if not params.music_path:
        return None, False
    from ..media.probe import ProbeError, probe_asset
    try:
        ma = probe_asset(params.music_path)
    except ProbeError as exc:
        notes.append(f"配樂檔無法使用，已略過配樂：{exc}")
        return None, False
    if not ma.has_audio or ma.duration <= 0:
        notes.append(f"配樂檔沒有可用的聲音，已略過配樂：{ma.name}")
        return None, False
    existing = proj.find_asset_by_path(ma.path)
    if existing is not None:
        return existing, False
    ma.id = "pa_" + sha256_hex(ma.path)[:8]
    return ma, True


def _why(info: dict, sh: _Short, band: sc.Band, feat: sc.Feat, has_music: bool) -> str:
    st, parts = info.get("stats", {}), info.get("parts", {})
    bits: list[str] = []
    if len(feat.content):
        t0, t1 = float(info.get("t0", 0.0)), float(info.get("t1", 0.0))
        best = feat.content.best_in(t0, t1, 2)
        if best:
            lines = "；".join(f"「{hl.quote(u.text)}」（{u.reasons() or '語氣'}，{u.score:.2f}）" for u in best)
            bits.append(f"爆點句：{lines}")
        if "content_hook" in st:
            bits.append("開頭 3 秒就有金句" if st["content_hook"] >= 0.5 else "開頭 3 秒內容普通")
    sk, ek = info.get("start_kind", ""), info.get("end_kind", "")
    start_txt = {"sentence": "上一句結束之後（句首）", "silence": "停頓之後", "scene": "鏡頭切換處", "grid": "固定間隔點（附近沒有更好的邊界）",
                 "edge": "素材開頭"}.get(sk, sk)
    end_txt = {"sentence": "句末", "silence": "停頓之前", "scene": "鏡頭切換處", "grid": "固定間隔點（附近沒有更好的邊界）",
               "edge": "素材結尾"}.get(ek, ek)
    comp = parts.get("complete")
    if comp is not None:
        bits.append(f"起點在{start_txt}、終點在{end_txt}（起訖完整度 {comp:.0%}" + ("；有一端落在字中間" if info.get("mid_word") else "") + "）")
    if feat.has_voice and "hook_onset" in st:
        bits.append("開頭 1 秒內就有人聲" if st["hook_onset"] >= 0.9 else "開頭 1 秒內人聲偏少")
        hook = []
        if feat.has_rms and st.get("hook_loud_pct", 0.0) >= 0.5:
            hook.append("響度高於素材中位數")
        if feat.has_motion and st.get("hook_motion_pct", 0.0) >= 0.5:
            hook.append("動作量高於中位數")
        if feat.has_faces and st.get("hook_face", 0.0) >= 0.5:
            hook.append("畫面有人臉")
        if hook:
            bits.append("前 3 秒" + "、".join(hook))
    if "voice_ratio" in st and feat.has_voice:
        bits.append(f"有人聲的比例 {st['voice_ratio']:.0%}")
    if "motion_pct" in st:
        bits.append(f"動作量高於素材 {st['motion_pct']:.0%} 的片段")
    if "face_ratio" in st:
        bits.append(f"{st['face_ratio']:.0%} 的取樣有人臉")
    if st.get("kw_hits"):
        bits.append(f"含 {int(st['kw_hits'])} 個重點詞")
    elif not len(feat.content):
        bits.append("沒有字幕／逐字稿，無法判斷內容爆點")
    bits.append(f"長度 {sh.length:.1f} 秒（目標 {band.target:g} 秒，長度帶 {band.lo:g}–{band.hi:g} 秒）")
    return "；".join(bits) + "。"


def plan_shorts(proj: Project, asset_id: str, params: ShortsParams, progress=None, cancel=None) -> EditPlan:
    """Plan the shorts of ``asset_id`` (see the module docstring).  Raises ``AutoEditError`` / ``Cancelled``."""
    params = params or ShortsParams()
    rep = _Progress(progress)
    rep(0.0, "準備分析…")
    _check(cancel)
    asset = _validate(proj, asset_id, params)
    notes: list[str] = []
    fps = float(params.fps)
    rules = AuditRules()

    # ---------------------------------------------------------------- 1. evidence
    from ..analysis import analyze_asset
    from ..analysis.evidence import AnalyzeParams
    want_speech = bool(params.use_speech) or params.captions == "auto"
    ap = AnalyzeParams(silence_db=float(params.silence_db), speech=want_speech, language=params.language,
                       traditional=bool(params.traditional_chinese), allow_download=False)
    ev = analyze_asset(asset, ap, progress=rep.sub(0.0, 0.8), cancel=cancel)
    _check(cancel)
    notes.extend(str(n) for n in (ev.notes or []))
    rep(0.8, "整理逐字稿與字幕來源…")

    # the transcript that decides WHAT is a highlight (subtitle file > same-name file next to the video > speech)
    tfile = params.transcript_file or (params.caption_file if params.captions == "file" else "")
    plan_segments, tinfo = hl.choose_transcript(asset, ev, tfile, use_speech=bool(params.use_speech), notes=notes)
    # caption source (source seconds)
    cap_segments: Optional[list] = None
    if params.captions == "auto":
        if ev.speech is not None:
            cap_segments = [dict(s) for s in ev.speech]
        elif plan_segments:
            cap_segments = plan_segments
            notes.append("沒有語音辨識逐字稿，字幕改用字幕檔的文字。")
        else:
            notes.append("沒有逐字稿，所以沒有加字幕（語音辨識是選用功能；原因見上方說明）。")
    elif params.captions == "file":
        cap_segments = source_segments(asset, ev, "file", params.caption_file, language=params.language,
                                       traditional=bool(params.traditional_chinese), notes=notes, cancel=cancel)
    if not ev.has_audio and not feat_has_audio(ev):
        notes.append("素材沒有聲音：人聲、響度、開頭鉤子與停頓緊湊化都無法使用，評分改看動作量與畫面。")
    if params.reframe == "face" and not any(ev.faces or []):
        if not any("人臉" in n for n in notes):
            notes.append("未偵測到人臉，取景會改用動作重心")

    # ---------------------------------------------------------------- 2. band
    band = sc.plan_band(params)
    notes.extend(band.notes)
    overrides, judge_info = _judge(params, plan_segments, notes, cancel)
    feat = sc.Feat(ev, plan_segments, overrides)
    if not feat.words:                                      # subtitle-file text + speech-recognition word timings
        feat.words = sc.transcript_words(hl.word_segments(ev))
    dur = float(ev.duration or asset.duration)
    weights = dict(sc.DEFAULT_WEIGHTS)
    weights.update({k: float(v) for k, v in (getattr(params, "weights", None) or {}).items() if k in weights})
    _check(cancel)

    # ---------------------------------------------------------------- 3. windows
    rep(0.82, "搜尋最佳片段…")
    fixer = _CutFixer(ev, asset, fps, enabled=bool(ev.scenes))
    pool: Optional[sc.Pool] = None
    infos: list[dict] = []
    shorts: list[_Short] = []
    whole = dur < band.lo
    if whole:
        notes.append(f"素材只有 {dur:.0f} 秒，短於 {band.label} 長度帶下限 {band.lo:g} 秒，只產生 1 支（整段）短影音。")
        info = {"t0": 0.0, "t1": dur, "tight": dur, "saved": 0.0, "score": 0.0, "parts": {}, "stats": {},
                "start_kind": "edge", "end_kind": "edge", "mid_word": False}
        sh = _build_short(info, feat=feat, ev=ev, asset=asset, params=params, band=band, fixer=fixer, check_band=False)
        if sh is None:
            raise AutoEditError("影片太短，無法剪成短影音")
        shorts = [sh]
    else:
        bounds = sc.find_boundaries(feat)
        pool = sc.search_windows(feat, params, band, weights, bounds, should_cancel=lambda: _check(cancel))
        _check(cancel)
        built: dict[int, _Short] = {}

        def accept(k: int) -> bool:
            _check(cancel)
            sh = _build_short(pool.info(k), feat=feat, ev=ev, asset=asset, params=params, band=band, fixer=fixer)
            if sh is None:
                return False
            built[k] = sh
            rep(0.82 + 0.1 * min(len(built) / max(int(params.count), 1), 1.0), f"規劃第 {len(built)} 支短影音…")
            return True

        picked = sc.select_windows(pool, int(params.count), float(params.min_gap_between), accept)
        shorts = [built[k] for k in picked]
        if not shorts:
            # nothing fit the band exactly: fall back to the stretch from the start with the target length
            notes.append("找不到長度剛好落在長度帶內的片段，改用素材開頭的一段。")
            t1 = min(dur, band.target)
            info = {"t0": 0.0, "t1": t1, "tight": t1, "saved": 0.0, "score": 0.0, "parts": {}, "stats": {},
                    "start_kind": "edge", "end_kind": "grid", "mid_word": False}
            sh = _build_short(info, feat=feat, ev=ev, asset=asset, params=params, band=band, fixer=fixer, check_band=False)
            if sh is None:
                raise AutoEditError("找不到可用的片段，無法剪成短影音")
            shorts = [sh]
        elif len(shorts) < int(params.count):
            notes.append(f"只找到 {len(shorts)} 支符合條件的短影音（想要 {int(params.count)} 支）：素材長度、長度帶與"
                         f"間隔 {params.min_gap_between:g} 秒的限制讓可用片段不足。")
    if params.reframe == "blur":
        notes.append("取景模式為模糊背景：整張 16:9 畫面放在中間、背景模糊放大，不做 punch-in，所以第一個視覺剪點可能晚於 "
                     f"{params.first_cut_max:g} 秒。")

    # ---------------------------------------------------------------- 4. music (shared by all shorts)
    music, music_new = _music_asset(proj, params, notes)

    # ---------------------------------------------------------------- 5. sequences
    ids = IdGen("p")
    sequences: list[dict] = []
    decisions: list[dict] = []
    style = CaptionStyle(size=0.052, bold=True, outline=0.008, position="bottom", margin=0.20, max_chars=16)
    n_cues_total = 0
    for rank, sh in enumerate(shorts, start=1):
        _check(cancel)
        rep(0.92 + 0.07 * (rank - 1) / max(len(shorts), 1), f"整理第 {rank} 支短影音…")
        info = sh.info
        t_first, t_last = sh.pieces[0].a, sh.pieces[-1].b
        name = f"短影音 {rank}｜{_mmss(t_first)}–{_mmss(t_last)}"
        best = feat.content.best_in(t_first, t_last, 1) if len(feat.content) else []
        if best:
            name += f"｜{hl.quote(best[0].text, 12)}"
        seq = Sequence.create(name, int(params.width), int(params.height), fps, video_tracks=2, audio_tracks=2,
                              seq_id=ids("s"))
        for tr in list(seq.video_tracks) + list(seq.audio_tracks):          # Sequence.create randomises track ids
            tr.id = ids("t")
        seq.caption_style = style
        v1 = seq.video_tracks[0]
        scenes_here = fixer.exact_scenes(t_first, t_last) if (ev.scenes and params.reframe in ("face", "motion")) else []
        face_total = 0.0
        jumps = 0
        modes: dict[str, float] = {}
        for i, p in enumerate(sh.pieces):
            if params.reframe == "blur":
                tf = Transform(fit="blur", zoom=1.0, cx=0.5, cy=0.5, keys=[])
                p.info = {"mode_used": "blur", "face_ratio": 0.0, "n_keys": 0, "jumps": 0}
            else:
                keys, p.info = reframe_keys(ev, asset, p.a, p.b, int(params.width), int(params.height),
                                            mode=params.reframe, strength=float(params.reframe_strength), zoom=p.zoom,
                                            punch_in_at=0.0 if p.zoom > 1.0 else None, fps=fps,
                                            scenes=scenes_here if (ev.scenes and params.reframe in ("face", "motion")) else None)
                p.keys = keys
                tf = Transform(fit="cover", zoom=float(p.zoom), cx=float(keys[0][1]), cy=float(keys[0][2]),
                               keys=[[float(a), float(b), float(c)] for a, b, c in keys])
            jumps += int(p.info.get("jumps", 0))
            modes[p.info.get("mode_used", "center")] = modes.get(p.info.get("mode_used", "center"), 0.0) + (p.b - p.a)
            nxt_seam = i + 1 < len(sh.pieces) and sh.pieces[i + 1].seam
            note = ""
            if i == 0:
                note = "自動剪輯：" + _why(info, sh, band, feat, music is not None)
            elif p.seam:
                note = f"第一個視覺剪點（{p.start:.1f} 秒）：同一段連續素材，後半 punch-in {p.zoom:g} 倍。"
            else:
                note = "停頓後接續（緊湊化剪掉中間的靜音）" + (f"，取景 {p.zoom:g} 倍" if p.zoom > 1.0 else "") + "。"
            v1.clips.append(Clip(id=ids("c"), asset_id=asset.id, start=p.start, src_in=p.a, src_out=p.b,
                                 speed=1.0, fade_in=0.0 if p.seam else FADE, fade_out=0.0 if nxt_seam else FADE,
                                 transform=tf, label="", note=note))
        total_len = sh.length
        # ---- captions
        cues = []
        cap_note = ""
        if params.captions != "off" and cap_segments:
            cues = cues_for_pieces(cap_segments, [Piece(asset.id, p.a, p.b, p.start, 1.0) for p in sh.pieces],
                                   max_chars=style.max_chars, ids=ids, rules=rules, fps=fps)
        seq.cues = cues
        n_cues_total += len(cues)
        # ---- music
        gain_db = float(params.music_gain_db)
        voice_ratio = float(info.get("stats", {}).get("voice_ratio", 0.0)) if feat.has_voice else 0.0
        if feat.has_voice and "voice_ratio" not in info.get("stats", {}):
            vt = (feat.P_voiced[feat.a_idx(t_last)] - feat.P_voiced[feat.a_idx(t_first)]) / feat.afps
            voice_ratio = min(1.0, vt / max(total_len, 1e-6))
        boosted = False
        if music is not None:
            if voice_ratio < NOVOICE_RATIO:
                gain_db += NOVOICE_BOOST_DB
                boosted = True
            a1 = seq.audio_tracks[0]
            nf = int(round(total_len * fps))
            mf = max(1, int(math.floor(float(music.duration) * fps + 1e-6)))     # whole frames: loop seams stay on the grid
            min_f = min(int(math.ceil(MIN_LOOP_SECONDS * fps)), mf)             # the last loop carries the fade-out: never a sliver
            pos_f = 0
            loops = 0
            while pos_f < nf:
                d_f = min(mf, nf - pos_f)
                rest_f = nf - pos_f - d_f
                if 0 < rest_f < min_f and d_f - (min_f - rest_f) >= min_f:
                    d_f -= min_f - rest_f
                last = pos_f + d_f >= nf
                d = d_f / fps
                a1.clips.append(Clip(id=ids("c"), asset_id=music.id, start=pos_f / fps, src_in=0.0, src_out=d, speed=1.0,
                                     volume=float(10 ** (gain_db / 20.0)),
                                     fade_in=0.0 if loops == 0 else 0.02,
                                     fade_out=min(1.5, d) if last else 0.02, label="配樂",
                                     note="自動剪輯：固定音量配樂（沒有動態閃避）。"))
                pos_f += d_f
                loops += 1
        seq.meta = {"shorts": {"asset_id": asset.id, "t0": _f(t_first), "t1": _f(t_last), "rank": rank,
                               "score": _f(info.get("score", 0.0))}}
        seq_dict = seq.to_dict()
        sequences.append(seq_dict)
        sid = seq.id

        # ---- decisions
        score = float(info.get("score", 0.0))
        summary = f"{_mmss(t_first)}–{_mmss(t_last)} · {total_len:.0f} 秒 · 分數 {score:.2f}"
        decisions.append({
            "id": ids("d"), "kind": "short", "sequence": sid, "score": _f(score),
            "source": {"asset_id": asset.id, "t0": _f(t_first), "t1": _f(t_last)},
            "summary": summary, "why": _why(info, sh, band, feat, music is not None),
            "highlights": [{"t0": _f(u.start), "t1": _f(u.end), "text": u.text, "score": _f(u.score),
                            "tags": list(u.tags[:3])} for u in (feat.content.best_in(t_first, t_last, 3)
                                                                if len(feat.content) else [])],
            "evidence": {"parts": {k: _f(v) for k, v in info.get("parts", {}).items()},
                         "weights": {k: _f(v) for k, v in (pool.weights if pool is not None else {}).items()},
                         "start_kind": info.get("start_kind", ""), "end_kind": info.get("end_kind", ""),
                         "stats": {k: _f(v) for k, v in info.get("stats", {}).items()},
                         "length": _f(total_len), "pieces": len(sh.pieces),
                         "first_cut": None if sh.first_cut is None else _f(sh.first_cut)},
        })
        # tighten
        if params.tighten:
            n_cut = max(0, sum(1 for p in sh.pieces if not p.seam) - 1)
            cut_total = 0.0
            for i in range(1, len(sh.pieces)):
                if not sh.pieces[i].seam:
                    cut_total += sh.pieces[i].a - sh.pieces[i - 1].b
            tsum = (f"剪掉 {n_cut} 處停頓，共 {cut_total:.1f} 秒" if n_cut else "沒有需要剪掉的停頓")
            decisions.append({
                "id": ids("d"), "kind": "tighten", "sequence": sid, "summary": tsum,
                "why": (f"長度 ≥ {params.pause_min:g} 秒的停頓只保留 {params.pause_keep:g} 秒（兩端各留一半），"
                        f"小於 {MIN_CLIP:g} 秒的片段不切；有逐字時間時剪點不會落在字中間。"
                        if n_cut else f"視窗內沒有長度 ≥ {params.pause_min:g} 秒的停頓（或剪開後會產生小於 {MIN_CLIP:g} 秒的片段）。"),
                "evidence": {"cuts": [[_f(a), _f(b)] for a, b in sh.removals], "pause_min": _f(params.pause_min),
                             "pause_keep": _f(params.pause_keep), "saved": _f(cut_total)}})
        else:
            decisions.append({"id": ids("d"), "kind": "tighten", "sequence": sid, "summary": "未緊湊化",
                              "why": "設定為不剪停頓，整段原樣保留。", "evidence": {}})
        # reframe
        used = max(modes.items(), key=lambda kv: kv[1])[0] if modes else "center"
        face_ratio = 0.0
        if feat.has_faces:
            nr = float(feat.P_ran[feat.v_idx(t_last)] - feat.P_ran[feat.v_idx(t_first)])
            nf = float(feat.P_face[feat.v_idx(t_last)] - feat.P_face[feat.v_idx(t_first)])
            face_ratio = nf / nr if nr > 0 else 0.0
        decisions.append({
            "id": ids("d"), "kind": "reframe", "sequence": sid, "summary": _reframe_summary(params, used, face_ratio, sh),
            "why": _reframe_why(params, used, face_ratio, jumps, sh),
            "evidence": {"mode": params.reframe, "mode_used": used, "face_ratio": _f(face_ratio), "jumps": jumps,
                         "keys": [len(p.keys) for p in sh.pieces], "zoom": [_f(p.zoom) for p in sh.pieces],
                         "strength": _f(params.reframe_strength)}})
        # caption
        decisions.append(_caption_decision(ids, sid, params, cues, cap_segments, rules))
        # music
        if music is not None:
            mname = music.name or music.path
            decisions.append({
                "id": ids("d"), "kind": "music", "sequence": sid,
                "summary": f"配樂「{mname}」，音量 {gain_db:+.0f} dB（固定）",
                "why": (("這支短影音幾乎沒有人聲，配樂音量提高 6 dB；" if boosted else "有人聲，使用設定的配樂音量；")
                        + "沒有動態閃避（ducking）：人聲出現時配樂不會自動壓低，靜態音量是已知限制。結尾淡出 1.5 秒。"),
                "evidence": {"gain_db": _f(gain_db), "asset_id": music.id, "voice_ratio": _f(voice_ratio)}})
        else:
            decisions.append({"id": ids("d"), "kind": "music", "sequence": sid, "summary": "未加配樂",
                              "why": "沒有指定配樂檔，或配樂檔無法使用。", "evidence": {}})

    if music is not None:
        notes.append("配樂為固定音量，沒有動態閃避（人聲出現時不會自動壓低配樂）。")

    # ---------------------------------------------------------------- 6. plan
    brief = {
        "kind": "shorts", "platform": params.platform, "platform_label": band.label, "count": int(params.count),
        "target_seconds": _f(band.target), "band": [_f(band.lo), _f(band.hi)],
        "deadzone": None if band.deadzone is None else [_f(band.deadzone[0]), _f(band.deadzone[1])],
        "size": [int(params.width), int(params.height)], "fps": _f(fps), "reframe": params.reframe,
        "reframe_strength": _f(params.reframe_strength), "captions": params.captions, "tighten": bool(params.tighten),
        "pause_min": _f(params.pause_min), "pause_keep": _f(params.pause_keep), "min_clip_seconds": MIN_CLIP,
        "first_cut_max": _f(params.first_cut_max), "min_gap_between": _f(params.min_gap_between),
        "language": params.language, "use_speech": bool(params.use_speech), "asset_id": asset.id,
        "weights": {k: _f(v) for k, v in weights.items()},
        "transcript": tinfo, "highlight_engine": judge_info,
        "weights_used": {k: _f(v) for k, v in (pool.weights if pool is not None else {}).items()},
        "params": params.to_dict(),
    }
    sources = [{"asset_id": asset.id, "name": asset.name, "fingerprint": asset.fingerprint,
                "duration": _f(asset.duration), "evidence_sha256": ev.sha256()}]
    plan_assets = [music.to_dict()] if (music is not None and music_new) else []
    rep(1.0, "完成")
    return EditPlan(kind="shorts", brief=brief, sources=sources, assets=plan_assets, sequences=sequences,
                    decisions=decisions, notes=_dedupe(notes))


# ------------------------------------------------------------------------------------ highlight judge
def _judge(params, segments: Optional[list], notes: list[str], cancel) -> tuple[Optional[dict], dict]:
    """Optional language-model judge of the sentences (opt-in).  Returns ``(overrides for score_sentences, info)``."""
    engine = str(getattr(params, "highlight_engine", "local") or "local")
    if engine == "local" or not segments:
        return None, {"engine": "local"}
    from .llm_judge import judge_sentences
    units = hl.sentences(segments)
    return judge_sentences(units, engine=engine, language=params.language, notes=notes, cancel=cancel)


# ------------------------------------------------------------------------------------ decision texts
def feat_has_audio(ev) -> bool:
    return bool(ev.voice) or bool(ev.rms_db)


def _reframe_summary(params: ShortsParams, used: str, face_ratio: float, sh: _Short) -> str:
    if params.reframe == "blur":
        return "模糊背景（不裁切）"
    label = {"face": "人臉追蹤", "motion": "動作重心", "center": "固定置中"}.get(used, used)
    keys = sum(len(p.keys) for p in sh.pieces)
    return f"{label}取景 · 臉偵測比例 {face_ratio:.0%} · {keys} 個關鍵格"


def _reframe_why(params: ShortsParams, used: str, face_ratio: float, jumps: int, sh: _Short) -> str:
    if params.reframe == "blur":
        return "取景模式為模糊背景：整張 16:9 畫面放在中間、背景模糊放大，不裁切，也不做 punch-in。"
    if params.reframe == "center":
        return "取景模式為固定置中：裁切視窗固定在畫面中央（cx=0.5）。"
    parts = []
    if used == "face":
        parts.append(f"臉偵測比例 {face_ratio:.0%}，以人臉為主體取景（多人且放得進取景窗時取群體中心，否則跟最大、最連續的臉）")
    elif used == "motion":
        parts.append(f"臉偵測比例只有 {face_ratio:.0%}，改用畫面變化的重心取景（Haar 人臉偵測對側臉、低頭、逆光很弱）")
    else:
        parts.append("沒有可用的人臉或動作資訊，固定置中取景")
    parts.append(f"主體在視窗中心 ±15% 窗寬內不移動，平移速度有上限（強度 {params.reframe_strength:g}），目標先夾進合法範圍再平滑")
    if jumps:
        parts.append(f"{jumps} 個鏡頭切換處不平滑，用相隔 1 影格的兩個關鍵格直接跳變")
    if any(p.zoom > 1.0 for p in sh.pieces):
        parts.append(f"第 2 段起交替 {sc.PUNCH_ZOOM:g} 倍 punch-in，臉落在視窗上中段")
    return "；".join(parts) + "。"


def _caption_decision(ids, sid: str, params: ShortsParams, cues: list, segments: Optional[list], rules: AuditRules) -> dict:
    src = {"auto": "語音辨識逐字稿", "file": "字幕檔", "off": "關閉"}.get(params.captions, params.captions)
    if params.captions == "off":
        return {"id": ids("d"), "kind": "caption", "sequence": sid, "summary": "未加字幕", "why": "字幕設定為關閉。",
                "evidence": {}}
    if not segments:
        return {"id": ids("d"), "kind": "caption", "sequence": sid, "summary": "沒有字幕來源，未加字幕",
                "why": "沒有可用的逐字稿或字幕檔（語音辨識未安裝／模型未下載／字幕檔無法讀取），詳見計畫說明。",
                "evidence": {"source": params.captions}}
    if not cues:
        return {"id": ids("d"), "kind": "caption", "sequence": sid, "summary": "這支短影音範圍內沒有字幕",
                "why": f"來源（{src}）在這個範圍沒有對得上的文字。", "evidence": {"source": params.captions}}
    cps = max((_reading_chars(q.text) / max(q.end - q.start, 1e-6) for q in cues), default=0.0)
    longest = max(len(q.text) for q in cues)
    return {"id": ids("d"), "kind": "caption", "sequence": sid,
            "summary": f"{len(cues)} 條字幕（{src}）· 最長 {longest} 字 · 最快 {cps:.1f} 字/秒",
            "why": (f"素材時間經緊湊化後的片段對應到時間軸；落在剪掉區間的字幕丟棄、跨剪點的裁切；每條至多 16 個全形字，"
                    f"優先在標點與停頓處斷開；結束時間 = min(說話結束 + 0.15 秒, 下一條開始 − 0.12 秒)，"
                    f"至少 {rules.min_cue_seconds:g} 秒，閱讀速度超過 {rules.reading_warn_cps:g} 字/秒時在不撞到下一條的前提下延長。"),
            "evidence": {"source": params.captions, "cues": len(cues), "max_chars": longest, "max_cps": _f(cps)}}
