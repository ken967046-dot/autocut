"""Timeline -> independent render *segments* -> ffmpeg commands.

Why segments?  The timeline is cut at every clip boundary (and at most every
``MAX_SEGMENT_SECONDS``).  Inside one segment every active clip covers the
*whole* segment, so each segment is a tiny, self-contained ffmpeg job: a few
inputs, no timing offsets, no 100-input mega graph.  Segments are

* **cacheable**   - keyed by a hash of everything that influences their pixels/samples,
                    so re-exporting after a small edit only re-renders what changed;
* **parallel**    - independent jobs;
* **concat-safe** - each is encoded with identical settings and starts on an IDR frame,
                    so the final file is assembled by stream copy (no generation loss).

Crossfades are overlaps on one track: the later clip fades in (alpha) over the
earlier one while the audio uses equal-power ramps.  Fades that only partly fall
inside a segment decode a short "pre-roll" so the ramp stays exact.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from ..core.geometry import interp_keys
from ..core.project import AUDIO, VIDEO, Asset, CaptionStyle, Clip, Project, Sequence, card_style, ordered_clips
from ..core.timecode import fps_fraction, frame_of
from ..media.ffmpeg import FFmpegInfo
from ..util import canonical_json, sha256_hex
from .ass import build_ass

RENDER_VERSION = 9          # 9: no line starts with punctuation, two lines when two fit, a backslash is drawn as typed
                            # 8: captions wrap to what fits across the frame (9:16 lines no longer run off the edges)
                            # 7: sound that starts after the picture in the file keeps that offset (``first_pts=0``, no PTS rebase)
#                             6: the picture keeps its place relative to the sound after ``-ss`` (no PTS rebase of the first frame)
MAX_SEGMENT_SECONDS = 30.0
AUDIO_RATE = 48000


class RenderError(RuntimeError):
    pass


class MissingMediaError(RenderError):
    def __init__(self, paths: list[str]) -> None:
        super().__init__("找不到這些素材檔（可能被移動或刪除）：\n" + "\n".join(paths[:8]) + ("\n…" if len(paths) > 8 else ""))
        self.paths = paths


@dataclass
class EncodeSettings:
    encoder: str = "libx264"
    crf: int = 18
    x264_preset: str = "medium"
    scale_flags: str = "lanczos"
    burn_captions: bool = True
    fontsdir: Optional[str] = None


@dataclass
class VClip:
    clip: Clip
    asset: Asset
    cs: int
    ce: int
    vfi: int            # alpha fade-in frames (own fade or crossfade-in)
    vfo: int            # alpha fade-out frames
    z: int


@dataclass
class AClip:
    clip: Clip
    asset: Asset
    cs: int
    ce: int
    afi: int
    afo: int
    gain: float
    from_video: bool


@dataclass
class Segment:
    index: int
    f0: int
    f1: int
    v: list = field(default_factory=list)
    a: list = field(default_factory=list)
    cues: list = field(default_factory=list)
    cards: list = field(default_factory=list)       # title cards (字卡), same (start, end, text) form as ``cues``

    @property
    def n(self) -> int:
        return self.f1 - self.f0


@dataclass
class Compiled:
    seq_id: str
    width: int
    height: int
    fps: float
    fps_frac: Fraction
    f_start: int
    f_end: int
    segments: list
    cues: list
    style: CaptionStyle
    background: str
    stats: dict = field(default_factory=dict)
    cards: list = field(default_factory=list)       # title cards (字卡): burned in, never written to the .srt

    @property
    def total_frames(self) -> int:
        return self.f_end - self.f_start

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps


# ---------------------------------------------------------------------------- helpers
def _n(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def _frac_str(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def piecewise_expr(points: list[tuple[float, float]], var: str) -> str:
    """Linear interpolation through ``points`` (sorted by x) as an ffmpeg expression in ``var``,
    held constant outside the range. Uses a balanced if-tree to keep the nesting shallow."""
    if not points:
        return "0"
    if len(points) == 1:
        return _n(points[0][1])

    def build(lo: int, hi: int) -> str:
        if hi - lo == 1:
            (t0, v0), (t1, v1) = points[lo], points[hi]
            if t1 - t0 < 1e-9:
                return _n(v1)
            return f"({_n(v0)}+({_n(v1 - v0)})*({var}-({_n(t0)}))/{_n(t1 - t0)})"
        mid = (lo + hi) // 2
        return f"if(lt({var},{_n(points[mid][0])}),{build(lo, mid)},{build(mid, hi)})"

    t_first, v_first = points[0]
    t_last, v_last = points[-1]
    return f"if(lt({var},{_n(t_first)}),{_n(v_first)},if(gte({var},{_n(t_last)}),{_n(v_last)},{build(0, len(points) - 1)}))"


def window_keys(keys: list, t0: float, t1: float, cx: float, cy: float) -> list[list[float]]:
    """Keys needed to evaluate the path on ``[t0, t1]`` (boundary values included)."""
    if not keys:
        return []
    ax, ay = interp_keys(keys, t0, cx, cy)
    bx, by = interp_keys(keys, t1, cx, cy)
    inner = [list(k) for k in keys if t0 < k[0] < t1]
    return [[t0, ax, ay]] + inner + [[t1, bx, by]]


def atempo_chain(speed: float) -> list[str]:
    out: list[str] = []
    s = speed
    while s > 2.0 + 1e-9:
        out.append("atempo=2.0")
        s /= 2.0
    while s < 0.5 - 1e-9:
        out.append("atempo=0.5")
        s /= 0.5
    out.append(f"atempo={_n(s)}")
    return out


# ------------------------------------------------------------------------- compiling
def _track_items(proj: Project, tr, fps: float, missing: list[str]) -> list[tuple[Clip, Asset, int, int]]:
    items: list[tuple[Clip, Asset, int, int]] = []
    for c in ordered_clips(tr.clips):
        if not c.enabled:
            continue
        a = proj.assets.get(c.asset_id)
        if a is None:
            raise RenderError("時間軸上有片段指向不存在的素材")
        if not os.path.exists(a.path):
            if a.path not in missing:
                missing.append(a.path)
            continue
        cs = max(0, frame_of(c.start, fps))
        ce = max(cs + 1, frame_of(c.end, fps))
        items.append((c, a, cs, ce))
    return items


def _crossfades(items: list[tuple[Clip, Asset, int, int]]) -> tuple[list[int], list[int]]:
    xin = [0] * len(items)
    xout = [0] * len(items)
    for i in range(1, len(items)):
        _, _, bcs, bce = items[i]
        for j in range(i - 1, -1, -1):
            _, _, acs, ace = items[j]
            if ace > bcs:
                ov = min(ace, bce) - bcs
                if ov > 0:
                    xin[i] = max(xin[i], ov)
                    xout[j] = max(xout[j], ov)
                break
    return xin, xout


def compile_sequence(proj: Project, seq: Sequence, out_w: Optional[int] = None, out_h: Optional[int] = None,
                     frame_range: Optional[tuple[float, float]] = None,
                     max_segment_seconds: float = MAX_SEGMENT_SECONDS) -> Compiled:
    fps = seq.fps
    W, H = out_w or seq.width, out_h or seq.height
    missing: list[str] = []
    vclips: list[VClip] = []
    aclips: list[AClip] = []
    for z, tr in enumerate(seq.video_tracks):
        if tr.muted:
            continue
        items = _track_items(proj, tr, fps, missing)
        xin, xout = _crossfades(items)
        for i, (c, a, cs, ce) in enumerate(items):
            fi, fo = int(round(c.fade_in * fps)), int(round(c.fade_out * fps))
            vclips.append(VClip(c, a, cs, ce, vfi=max(fi, xin[i]), vfo=fo, z=z))
            if a.has_audio and not c.muted and a.kind != "image":
                aclips.append(AClip(c, a, cs, ce, afi=max(fi, xin[i]), afo=max(fo, xout[i]),
                                    gain=c.volume * tr.volume, from_video=True))
    for tr in seq.audio_tracks:
        if tr.muted:
            continue
        items = _track_items(proj, tr, fps, missing)
        xin, xout = _crossfades(items)
        for i, (c, a, cs, ce) in enumerate(items):
            if c.muted or not a.has_audio:
                continue
            fi, fo = int(round(c.fade_in * fps)), int(round(c.fade_out * fps))
            aclips.append(AClip(c, a, cs, ce, afi=max(fi, xin[i]), afo=max(fo, xout[i]),
                                gain=c.volume * tr.volume, from_video=False))
    if missing:
        raise MissingMediaError(missing)
    if not vclips and not aclips:
        raise RenderError("時間軸是空的，沒有可輸出的內容")
    end_all = max([v.ce for v in vclips] + [a.ce for a in aclips])
    fa, fb = 0, end_all
    if frame_range is not None:
        fa = max(0, frame_of(frame_range[0], fps))
        fb = min(end_all, frame_of(frame_range[1], fps))
        if fb <= fa:
            raise RenderError("輸出範圍是空的")
    bounds = {fa, fb}
    for obj in (*vclips, *aclips):
        for b in (obj.cs, obj.ce):
            if fa < b < fb:
                bounds.add(b)
    max_f = max(1, int(max_segment_seconds * fps))
    spans: list[tuple[int, int]] = []
    sb = sorted(bounds)
    for a, b in zip(sb, sb[1:]):
        n = b - a
        pieces = max(1, math.ceil(n / max_f))
        base, rem = divmod(n, pieces)
        s = a
        for i in range(pieces):
            ln = base + (1 if i < rem else 0)
            spans.append((s, s + ln))
            s += ln
    ordered = [q for q in sorted(seq.cues, key=lambda q: q.start) if q.text.strip()]
    cues = [(q.start, q.end, q.text) for q in ordered if not q.is_card]
    cards = [(q.start, q.end, q.text) for q in ordered if q.is_card]
    segments: list[Segment] = []
    for idx, (f0, f1) in enumerate(spans):
        seg = Segment(idx, f0, f1)
        seg.v = sorted((v for v in vclips if v.cs < f1 and v.ce > f0), key=lambda v: (v.z, v.cs))
        seg.a = [a for a in aclips if a.cs < f1 and a.ce > f0]
        t0, t1 = f0 / fps, f1 / fps
        seg.cues = [c for c in cues if c[1] > t0 and c[0] < t1]
        seg.cards = [c for c in cards if c[1] > t0 and c[0] < t1]
        segments.append(seg)
    return Compiled(seq.id, W, H, fps, fps_fraction(fps), fa, fb, segments, cues, seq.caption_style, seq.background,
                    {"video_clips": len(vclips), "audio_clips": len(aclips)}, cards)


# -------------------------------------------------------------------------- signature
_STAT_CACHE: dict[str, tuple[int, int]] = {}


def _stat(path: str) -> tuple[int, int]:
    try:
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns
    except OSError:
        return -1, -1


def segment_signature(comp: Compiled, seg: Segment, enc: EncodeSettings) -> str:
    """Hash of everything that influences the pixels/samples of this segment."""

    def vsig(v: VClip) -> dict:
        size, mt = _stat(v.asset.path)
        return {"p": v.asset.path, "sz": size, "mt": mt, "in": v.clip.src_in, "spd": v.clip.speed,
                "tf": v.clip.transform.to_dict(), "look": v.clip.look.to_dict(), "cs": v.cs, "ce": v.ce,
                "vfi": v.vfi, "vfo": v.vfo, "z": v.z, "sar": v.asset.sar, "img": v.asset.kind == "image"}

    def asig(a: AClip) -> dict:
        size, mt = _stat(a.asset.path)
        return {"p": a.asset.path, "sz": size, "mt": mt, "in": a.clip.src_in, "spd": a.clip.speed, "cs": a.cs,
                "ce": a.ce, "afi": a.afi, "afo": a.afo, "g": a.gain}

    burn = enc.burn_captions and bool(seg.cues or seg.cards)
    payload = {
        "rv": RENDER_VERSION, "w": comp.width, "h": comp.height, "fps": _frac_str(comp.fps_frac), "f0": seg.f0,
        "f1": seg.f1, "enc": [enc.encoder, enc.crf, enc.x264_preset, enc.scale_flags], "bg": comp.background,
        "v": [vsig(v) for v in seg.v], "a": [asig(a) for a in seg.a],
        "cues": [list(c) for c in seg.cues] if burn else [],
        "style": comp.style.to_dict() if burn else None, "fonts": enc.fontsdir if burn else None,
    }
    if burn and seg.cards:                    # only when present: segments without cards keep their cached renders
        payload["cards"] = [list(c) for c in seg.cards]
        payload["card_style"] = card_style(comp.style).to_dict()
    return sha256_hex(canonical_json(payload))


def segment_ass(comp: Compiled, seg: Segment, enc: EncodeSettings, info: FFmpegInfo) -> Optional[str]:
    if not (enc.burn_captions and (seg.cues or seg.cards) and info.has_ass):
        return None
    return build_ass(seg.cues, comp.style, comp.width, comp.height, t0=seg.f0 / comp.fps, t1=seg.f1 / comp.fps,
                     cards=seg.cards)


# ----------------------------------------------------------------- filtergraph building
def _stream_start(v: VClip, L0: int, L1: int) -> tuple[int, bool, bool]:
    """(pre-roll start S in clip-local frames, in-ramp?, out-ramp?) for a clip portion [L0, L1)."""
    dc = v.ce - v.cs
    in_ramp = v.vfi > 0 and L0 < v.vfi
    out_start = dc - v.vfo
    out_ramp = v.vfo > 0 and L1 > out_start
    if in_ramp:
        s = 0
    elif out_ramp:
        s = min(L0, out_start)
    else:
        s = L0
    return s, in_ramp, out_ramp


def _geometry(comp: Compiled, v: VClip, S: int, L1: int, enc: EncodeSettings, src: str, uid: str
              ) -> tuple[list[str], str]:
    """Statements that fit the clip into the output frame. Returns (statements, last_label)."""
    W, H = comp.width, comp.height
    a, tf = v.asset, v.clip.transform
    stmts: list[str] = []
    ar = W / H
    sf = enc.scale_flags
    if tf.fit == "contain":
        stmts.append(f"[{src}]scale={W}:{H}:force_original_aspect_ratio=decrease:flags={sf},"
                     f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[{uid}g]")
        return stmts, f"{uid}g"
    if tf.fit == "blur":
        bw, bh = max(2, (W // 8) // 2 * 2), max(2, (H // 8) // 2 * 2)
        radius = max(2, min(bw, bh) // 6)
        stmts.append(f"[{src}]split=2[{uid}b0][{uid}f0]")
        stmts.append(f"[{uid}b0]scale={W}:{H}:force_original_aspect_ratio=increase:flags=bilinear,crop={W}:{H},"
                     f"scale={bw}:{bh}:flags=bilinear,boxblur={radius}:2,scale={W}:{H}:flags=bilinear,"
                     f"eq=brightness=-0.08:saturation=0.9[{uid}b1]")
        stmts.append(f"[{uid}f0]scale={W}:{H}:force_original_aspect_ratio=decrease:flags={sf}[{uid}f1]")
        stmts.append(f"[{uid}b1][{uid}f1]overlay=(W-w)/2:(H-h)/2[{uid}g]")
        return stmts, f"{uid}g"
    # cover (crop a window of the output aspect, optionally animated)
    z = max(1.0, tf.zoom)
    src_ar = (a.width / a.height) if a.width and a.height else ar
    noop = abs(src_ar - ar) < 0.002 and z <= 1.0 + 1e-9 and not tf.keys
    pre = ""
    if not noop:
        w_expr = f"trunc(if(gte(iw/ih,{ar:.8f}),ih*{ar:.8f},iw)/{_n(z)}/2)*2"
        h_expr = f"trunc(if(gte(iw/ih,{ar:.8f}),ih,iw/{ar:.8f})/{_n(z)}/2)*2"
        if tf.keys:
            t_base = S / comp.fps          # keys are clip-local seconds; the stream clock starts at S
            var = f"(t+{_n(t_base)})"
            ks = window_keys(tf.keys, t_base, L1 / comp.fps, tf.cx, tf.cy)
            cx_expr = piecewise_expr([(k[0], k[1]) for k in ks], var)
            cy_expr = piecewise_expr([(k[0], k[2]) for k in ks], var)
        else:
            cx_expr, cy_expr = _n(tf.cx), _n(tf.cy)
        x_expr = f"clip(({cx_expr})*iw-ow/2,0,iw-ow)"
        y_expr = f"clip(({cy_expr})*ih-oh/2,0,ih-oh)"
        pre = f"crop=w='{w_expr}':h='{h_expr}':x='{x_expr}':y='{y_expr}',"
    stmts.append(f"[{src}]{pre}scale={W}:{H}:flags={sf},setsar=1[{uid}g]")
    return stmts, f"{uid}g"


def build_segment_args(comp: Compiled, seg: Segment, enc: EncodeSettings, info: FFmpegInfo, out_path: str,
                       ass_name: Optional[str]) -> tuple[list[str], str]:
    """Returns ``(ffmpeg args, filtergraph)`` for one segment."""
    fps = comp.fps
    ff = _frac_str(comp.fps_frac)
    W, H = comp.width, comp.height
    N = seg.n
    D = N / fps
    inputs: list[str] = []
    in_index: dict[str, int] = {}
    starts: dict[str, int] = {}
    stmts: list[str] = []

    def add_input(clip: Clip, asset: Asset, S: int, L1: int) -> int:
        if clip.id in in_index:
            return in_index[clip.id]
        span = (L1 - S) / fps * clip.speed + 0.5
        if asset.kind == "image":
            inputs.extend(["-loop", "1", "-framerate", ff, "-t", _n(span), "-i", asset.path])
        else:
            inputs.extend(["-ss", _n(clip.src_in + S / fps * clip.speed), "-t", _n(span), "-i", asset.path])
        in_index[clip.id] = len(in_index)
        starts[clip.id] = S
        return in_index[clip.id]

    # ---------------- video
    vlabels: list[str] = []
    any_alpha = False
    for k, v in enumerate(seg.v):
        L0, L1 = seg.f0 - v.cs, seg.f1 - v.cs
        S, in_ramp, out_ramp = _stream_start(v, L0, L1)
        idx = add_input(v.clip, v.asset, S, L1)
        uid = f"v{k}"
        # No ``setpts=PTS-STARTPTS`` here: after ``-ss`` the first frame is the first one AT OR AFTER the requested time, up to
        # one source frame later.  Putting that frame at 0 showed the picture up to a source frame (about 33 ms) AHEAD of the
        # sound, which is cut at the exact sample.  The stream's own timestamps are already relative to ``-ss``, and
        # ``fps=start_time=0`` repeats the first frame to fill the gap before it (measured in tests/test_render_oracle.py).
        pre: list[str] = []
        if abs(v.asset.sar - 1.0) > 1e-3:
            pre.append(f"scale=w='trunc(iw*{v.asset.sar:.6f}/2)*2':h=ih:flags=bicubic,setsar=1")
        if abs(v.clip.speed - 1.0) > 1e-9:
            pre.append(f"setpts=PTS/{_n(v.clip.speed)}")
        pre.append(f"fps=fps={ff}:round=near:start_time=0")
        pre.append("tpad=stop_mode=clone:stop_duration=2")
        stmts.append(f"[{idx}:v]{','.join(pre)}[{uid}p]")
        geo, last = _geometry(comp, v, S, L1, enc, f"{uid}p", uid)
        stmts.extend(geo)
        post: list[str] = []
        if v.clip.transform.flip_h:
            post.append("hflip")
        if not v.clip.look.neutral:
            lk = v.clip.look
            post.append(f"eq=brightness={_n(lk.brightness)}:contrast={_n(lk.contrast)}:saturation={_n(lk.saturation)}")
        if in_ramp or out_ramp:
            any_alpha = True
            post.append("format=yuva420p")
            dc = v.ce - v.cs
            if in_ramp:
                post.append(f"fade=t=in:st=0:d={_n(v.vfi / fps)}:alpha=1")
            if out_ramp:
                post.append(f"fade=t=out:st={_n((dc - v.vfo - S) / fps)}:d={_n(v.vfo / fps)}:alpha=1")
        post.append(f"trim=start_frame={L0 - S}:end_frame={L1 - S}")
        post.append("setpts=PTS-STARTPTS")
        stmts.append(f"[{last}]{','.join(post)}[{uid}o]")
        vlabels.append(f"{uid}o")

    # ---------------- video composite
    direct = len(vlabels) == 1 and not any_alpha
    if direct:
        cur = vlabels[0]
    else:
        stmts.append(f"color=c={_color(comp.background)}:s={W}x{H}:r={ff}:d={_n(D)},format=yuv420p[base]")
        cur = "base"
        for k, lab in enumerate(vlabels):
            nxt = f"ov{k}"
            stmts.append(f"[{cur}][{lab}]overlay=x=0:y=0:eof_action=pass:repeatlast=0[{nxt}]")
            cur = nxt
    if ass_name:
        opts = f"filename={ass_name}"
        if enc.fontsdir:
            opts += f":fontsdir='{_esc_filter_path(enc.fontsdir)}'"
        stmts.append(f"[{cur}]ass={opts}[vcap]")
        cur = "vcap"
    stmts.append(f"[{cur}]format=yuv420p[vout]")

    # ---------------- audio
    alabels: list[str] = []
    for k, a in enumerate(seg.a):
        L0, L1 = seg.f0 - a.cs, seg.f1 - a.cs
        dc = a.ce - a.cs
        if a.clip.id in starts:
            S = starts[a.clip.id]
            idx = in_index[a.clip.id]
        else:
            S = L0
            idx = add_input(a.clip, a.asset, S, L1)
        # No ``asetpts=PTS-STARTPTS`` here either: a file whose sound starts later than its picture (screen recordings, some
        # cameras) would lose that offset for a clip cut from the very start of the file - the sound came out as early as the
        # delay (measured: 77 ms of 78, 226 ms of 255).  ``first_pts=0`` pads the gap with silence instead, and does nothing
        # when the sound starts where the seek put it (the usual case).  Without ``async``: no compensation in the middle.
        fl = [f"aresample={AUDIO_RATE}:first_pts=0", "aformat=sample_fmts=fltp:channel_layouts=stereo"]
        if abs(a.clip.speed - 1.0) > 1e-9:
            fl += atempo_chain(a.clip.speed)
        tl, dcs = S / fps, dc / fps
        ramps: list[str] = []
        if a.afi > 0 and L0 < a.afi:
            ramps.append(f"sin(PI/2*clip((t+{_n(tl)})/{_n(a.afi / fps)},0,1))")
        if a.afo > 0 and L1 > dc - a.afo:
            ramps.append(f"sin(PI/2*clip(({_n(dcs)}-{_n(tl)}-t)/{_n(a.afo / fps)},0,1))")
        if ramps:
            fl.append(f"volume=volume='{_n(a.gain)}*{'*'.join(ramps)}':eval=frame")
        elif abs(a.gain - 1.0) > 1e-6:
            fl.append(f"volume={_n(a.gain)}")
        fl += [f"atrim=start={_n((L0 - S) / fps)}:end={_n((L1 - S) / fps)}", "asetpts=PTS-STARTPTS",
               f"apad=whole_dur={_n(D)}", f"atrim=end={_n(D)}"]
        lab = f"a{k}o"
        stmts.append(f"[{idx}:a]{','.join(fl)}[{lab}]")
        alabels.append(lab)
    if not alabels:
        stmts.append(f"anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=end={_n(D)},asetpts=PTS-STARTPTS[aout]")
    elif len(alabels) == 1:
        stmts.append(f"[{alabels[0]}]anull[aout]")
    else:
        ins = "".join(f"[{x}]" for x in alabels)
        if info.amix_normalize:
            stmts.append(f"{ins}amix=inputs={len(alabels)}:duration=longest:dropout_transition=0:normalize=0[aout]")
        else:
            stmts.append(f"{ins}amix=inputs={len(alabels)}:duration=longest:dropout_transition=0,"
                         f"volume={len(alabels)}[aout]")

    graph = ";".join(stmts)
    venc = encoder_args(enc)
    args = [*inputs, "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]", "-frames:v", str(N),
            *venc, "-c:a", "pcm_f32le", "-ar", str(AUDIO_RATE), "-ac", "2", "-t", _n(D), "-f", "mov", out_path]
    return args, graph


def encoder_args(enc: EncodeSettings) -> list[str]:
    if enc.encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(enc.crf), "-b:v", "0",
                "-pix_fmt", "yuv420p", "-profile:v", "high", "-g", "120"]
    if enc.encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", str(enc.crf), "-preset", "medium", "-pix_fmt", "nv12",
                "-g", "120"]
    if enc.encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", str(enc.crf),
                "-qp_p", str(enc.crf), "-pix_fmt", "yuv420p", "-g", "120"]
    return ["-c:v", "libx264", "-preset", enc.x264_preset, "-crf", str(enc.crf), "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-g", "120", "-bf", "2", "-sc_threshold", "0"]


def _color(hex_rgb: str) -> str:
    h = (hex_rgb or "#000000").lstrip("#")
    return "0x" + (h if len(h) == 6 else "000000")


def _esc_filter_path(p: str) -> str:
    return p.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
