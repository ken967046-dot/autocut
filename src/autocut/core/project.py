"""Project data model (schema ``autocut.project/1``).

A *project* owns media *assets* and one or more *sequences* (timelines).  A
sequence has video tracks (bottom -> top) and audio tracks.  A clip on a video
track plays its own picture *and* audio; audio tracks carry music/voice-over.

Overlapping clips on one track are a **crossfade** (the later clip fades in
over the earlier one); at most two clips may overlap at any instant.

All times are seconds on the sequence frame grid.  The model is plain data:
every edit goes through :mod:`autocut.core.ops` so it can be undone.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, Optional

from .. import __version__
from ..util import atomic_write_text, canonical_json, new_id, now_iso, sha256_hex
from .geometry import interp_keys
from .timecode import EPS, MAX_TIMELINE, TOUCH

SCHEMA = "autocut.project/1"
FIT_MODES = ("cover", "contain", "blur")
VIDEO, AUDIO = "video", "audio"


def _pick(cls: type, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in names}


class ProjectFileError(ValueError):
    """A project file (or autosave) that cannot be read: not JSON, not an AutoCut project, or damaged inside.  The message is
    written for the person who tried to open it."""


def _num(d: dict, key: str, default: float) -> float:
    v = d.get(key, default)
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    return x if math.isfinite(x) else default                  # "NaN" / "Infinity" are accepted by json but are not times


# ---------------------------------------------------------------------------- Asset
@dataclass
class Asset:
    id: str
    path: str
    name: str = ""
    kind: str = "video"            # video | audio | image
    duration: float = 0.0
    width: int = 0                 # display size (rotation + pixel aspect applied)
    height: int = 0
    fps: float = 0.0
    has_video: bool = True
    has_audio: bool = False
    audio_channels: int = 0
    sample_rate: int = 0
    vcodec: str = ""
    acodec: str = ""
    rotation: int = 0
    sar: float = 1.0
    size_bytes: int = 0
    fingerprint: str = ""
    created: str = ""              # container creation_time if present (ISO text)
    bitrate: int = 0

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Asset":
        d = _pick(cls, d)
        d.setdefault("id", new_id("a_"))
        d.setdefault("path", "")
        a = cls(**d)
        if not a.name:
            a.name = os.path.basename(a.path)
        return a

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width > 0

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def missing(self) -> bool:
        return not os.path.exists(self.path)


# ------------------------------------------------------------------------- Transform
@dataclass
class Transform:
    """Picture placement for a clip.

    ``cover``  : crop a window of the output aspect out of the source (9:16 reframing).
                 ``zoom`` >= 1 shrinks the window; ``(cx, cy)`` is its centre (0..1 of the
                 source). ``keys`` = ``[[t_local, cx, cy], ...]`` animates the centre.
    ``contain``: whole picture, black bars.
    ``blur``   : whole picture over a blurred, enlarged copy of itself.
    """

    fit: str = "cover"
    zoom: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    keys: list = field(default_factory=list)
    flip_h: bool = False

    def center_at(self, t_local: float) -> tuple[float, float]:
        return interp_keys(self.keys, t_local, self.cx, self.cy)

    def to_dict(self) -> dict:
        return {
            "fit": self.fit, "zoom": self.zoom, "cx": self.cx, "cy": self.cy,
            "keys": [[float(a), float(b), float(c)] for a, b, c in self.keys], "flip_h": self.flip_h,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "Transform":
        d = d or {}
        keys = []
        for k in d.get("keys") or []:
            try:
                keys.append([float(k[0]), float(k[1]), float(k[2])])
            except (TypeError, ValueError, IndexError):
                continue
        keys.sort(key=lambda k: k[0])
        fit = d.get("fit", "cover")
        return cls(
            fit=fit if fit in FIT_MODES else "cover",
            zoom=max(1.0, _num(d, "zoom", 1.0)),
            cx=_num(d, "cx", 0.5), cy=_num(d, "cy", 0.5),
            keys=keys, flip_h=bool(d.get("flip_h", False)),
        )


@dataclass
class Look:
    brightness: float = 0.0   # -1..1
    contrast: float = 1.0     # 0..3
    saturation: float = 1.0   # 0..3

    @property
    def neutral(self) -> bool:
        return abs(self.brightness) < 1e-6 and abs(self.contrast - 1) < 1e-6 and abs(self.saturation - 1) < 1e-6

    def to_dict(self) -> dict:
        return {"brightness": self.brightness, "contrast": self.contrast, "saturation": self.saturation}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Look":
        d = d or {}
        return cls(_num(d, "brightness", 0.0), _num(d, "contrast", 1.0), _num(d, "saturation", 1.0))


# ---------------------------------------------------------------------------- Clip
@dataclass
class Clip:
    id: str
    asset_id: str
    start: float = 0.0          # position on the timeline
    src_in: float = 0.0         # source in/out (seconds in the asset)
    src_out: float = 0.0
    speed: float = 1.0
    volume: float = 1.0         # linear gain of this clip's audio
    muted: bool = False
    enabled: bool = True
    fade_in: float = 0.0
    fade_out: float = 0.0
    transform: Transform = field(default_factory=Transform)
    look: Look = field(default_factory=Look)
    label: str = ""
    color: str = ""
    note: str = ""              # free text; the auto-editor writes its rationale here

    @property
    def duration(self) -> float:
        return max(0.0, (self.src_out - self.src_in) / (self.speed or 1.0))

    @property
    def end(self) -> float:
        return self.start + self.duration

    def src_at(self, t: float) -> float:
        """Source time shown at timeline time ``t``."""
        return self.src_in + (t - self.start) * self.speed

    def to_dict(self) -> dict:
        return {
            "id": self.id, "asset_id": self.asset_id, "start": self.start,
            "src_in": self.src_in, "src_out": self.src_out, "speed": self.speed,
            "volume": self.volume, "muted": self.muted, "enabled": self.enabled,
            "fade_in": self.fade_in, "fade_out": self.fade_out,
            "transform": self.transform.to_dict(), "look": self.look.to_dict(),
            "label": self.label, "color": self.color, "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        return cls(
            id=d.get("id") or new_id("c_"), asset_id=d.get("asset_id", ""),
            start=_num(d, "start", 0.0), src_in=_num(d, "src_in", 0.0), src_out=_num(d, "src_out", 0.0),
            speed=_num(d, "speed", 1.0) or 1.0, volume=_num(d, "volume", 1.0),
            muted=bool(d.get("muted", False)), enabled=bool(d.get("enabled", True)),
            fade_in=_num(d, "fade_in", 0.0), fade_out=_num(d, "fade_out", 0.0),
            transform=Transform.from_dict(d.get("transform")), look=Look.from_dict(d.get("look")),
            label=d.get("label", ""), color=d.get("color", ""), note=d.get("note", ""),
        )


MIN_FRAGMENT = 0.03             # a clip (or a piece left over by a cut) shorter than this is under one frame: not kept


def _runs(items, value) -> list[list]:
    """``items`` sorted by ``value`` and cut into runs whose neighbouring values differ by at most ``TOUCH``."""
    runs: list[list] = []
    run: list = []
    for x in sorted(items, key=value):
        if run and value(x) - value(run[-1]) > TOUCH:
            runs.append(run)
            run = []
        run.append(x)
    if run:
        runs.append(run)
    return runs


def ordered_clips(clips) -> list[Clip]:
    """The clips of one track in playing order: by start; clips that start together go in the order they end (the one
    that ends first is the layer that fades out, the longer one fades in over it); clips that start *and* end together
    go by id.  "Together" means within ``TOUCH`` - a saved / undone project is rounded to 6 decimals, so two times that
    were equal can differ by a few millionths of a second afterwards, and the order must not flip because of that.
    Every place that needs the order of a track's clips uses this one function, so the editor, the validator, the
    renderer and the previews always agree on who comes before whom."""
    out: list[Clip] = []
    for same_start in _runs(clips, lambda c: c.start):
        for same_end in _runs(same_start, lambda c: c.end):
            out += sorted(same_end, key=lambda c: c.id)
    return out


# --------------------------------------------------------------------------- Track
@dataclass
class Track:
    id: str
    name: str
    kind: str                    # video | audio
    clips: list = field(default_factory=list)
    muted: bool = False          # video track: hidden
    locked: bool = False
    volume: float = 1.0

    def sort(self) -> None:
        self.clips[:] = ordered_clips(self.clips)

    def end(self) -> float:
        return max((c.end for c in self.clips), default=0.0)

    def clip_at(self, t: float) -> Optional[Clip]:
        hit = None
        for c in self.clips:
            if c.start - EPS <= t < c.end - EPS:
                hit = c          # later start wins (it is on top of a crossfade)
        return hit

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "muted": self.muted,
            "locked": self.locked, "volume": self.volume, "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        t = cls(
            id=d.get("id") or new_id("t_"), name=d.get("name", ""), kind=d.get("kind", VIDEO),
            muted=bool(d.get("muted", False)), locked=bool(d.get("locked", False)),
            volume=_num(d, "volume", 1.0), clips=[Clip.from_dict(c) for c in d.get("clips", [])],
        )
        t.sort()
        return t


# --------------------------------------------------------------------- captions etc.
@dataclass
class CaptionStyle:
    font: str = "Microsoft JhengHei"
    size: float = 0.045            # fraction of frame height
    bold: bool = True
    color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: float = 0.006         # fraction of frame height
    shadow: float = 0.0
    position: str = "bottom"       # bottom | center | top
    margin: float = 0.12           # fraction of frame height kept free at the edge
    max_chars: int = 16            # soft wrap width (full-width characters)
    box: bool = False
    box_color: str = "#000000"
    box_opacity: float = 0.5

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "CaptionStyle":
        d = _pick(cls, d or {})
        return cls(**d)


CARD_COLOR = "#FFE14D"


def card_style(style: CaptionStyle) -> CaptionStyle:
    """The look of title cards (字卡), derived from the sequence's subtitle style so a font change applies to both: a
    larger bold line in a dark box near the top -- in the middle when the subtitles themselves sit at the top.  The
    renderer (ASS) and the preview monitor both draw cards with this one style."""
    return CaptionStyle(font=style.font, size=round(style.size * 1.15, 5), bold=True, color=CARD_COLOR,
                        outline_color="#000000", outline=0.0, shadow=0.0,
                        position="center" if style.position == "top" else "top", margin=0.075,
                        max_chars=max(6, min(int(style.max_chars), 14)), box=True, box_color="#000000",
                        box_opacity=1.0)       # opaque: libass draws one box per font run, a see-through
                                               # box would show darker seams where runs overlap (CJK + Latin)


@dataclass
class CaptionCue:
    id: str
    start: float
    end: float
    text: str = ""
    style: dict = field(default_factory=dict)   # per-cue overrides; {"kind": "card"} = a title card (字卡)

    @property
    def is_card(self) -> bool:
        """A title card (字卡): shown at the top in a box, on its own layer -- it may overlap the subtitles."""
        return isinstance(self.style, dict) and self.style.get("kind") == "card"

    def to_dict(self) -> dict:
        return {"id": self.id, "start": self.start, "end": self.end, "text": self.text, "style": dict(self.style)}

    @classmethod
    def from_dict(cls, d: dict) -> "CaptionCue":
        return cls(id=d.get("id") or new_id("q_"), start=_num(d, "start", 0.0), end=_num(d, "end", 0.0),
                   text=d.get("text", ""), style=dict(d.get("style") or {}))


@dataclass
class Marker:
    id: str
    time: float
    name: str = ""
    color: str = "#f5a623"

    def to_dict(self) -> dict:
        return {"id": self.id, "time": self.time, "name": self.name, "color": self.color}

    @classmethod
    def from_dict(cls, d: dict) -> "Marker":
        return cls(id=d.get("id") or new_id("m_"), time=_num(d, "time", 0.0), name=d.get("name", ""),
                   color=d.get("color", "#f5a623"))


# ------------------------------------------------------------------------ Sequence
@dataclass
class Sequence:
    id: str
    name: str = "序列 1"
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    sample_rate: int = 48000
    video_tracks: list = field(default_factory=list)   # bottom -> top
    audio_tracks: list = field(default_factory=list)
    cues: list = field(default_factory=list)
    markers: list = field(default_factory=list)
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    background: str = "#000000"
    meta: dict = field(default_factory=dict)

    @classmethod
    def create(cls, name: str = "序列 1", width: int = 1920, height: int = 1080, fps: float = 30.0,
               video_tracks: int = 2, audio_tracks: int = 2, seq_id: str | None = None) -> "Sequence":
        s = cls(id=seq_id or new_id("s_"), name=name, width=width, height=height, fps=fps)
        for i in range(video_tracks):
            s.video_tracks.append(Track(new_id("t_"), f"V{i + 1}", VIDEO))
        for i in range(audio_tracks):
            s.audio_tracks.append(Track(new_id("t_"), f"A{i + 1}", AUDIO))
        return s

    # ---- lookups
    def tracks(self) -> Iterator[Track]:
        yield from self.video_tracks
        yield from self.audio_tracks

    def find_track(self, track_id: str) -> Optional[Track]:
        for t in self.tracks():
            if t.id == track_id:
                return t
        return None

    def find_clip(self, clip_id: str) -> tuple[Optional[Track], Optional[Clip]]:
        for t in self.tracks():
            for c in t.clips:
                if c.id == clip_id:
                    return t, c
        return None, None

    def find_cue(self, cue_id: str) -> Optional[CaptionCue]:
        for q in self.cues:
            if q.id == cue_id:
                return q
        return None

    def all_clips(self) -> Iterator[tuple[Track, Clip]]:
        for t in self.tracks():
            for c in t.clips:
                yield t, c

    @property
    def duration(self) -> float:
        d = max((t.end() for t in self.tracks()), default=0.0)
        return d

    @property
    def aspect_label(self) -> str:
        from math import gcd
        g = gcd(self.width, self.height) or 1
        return f"{self.width // g}:{self.height // g}"

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    def used_asset_ids(self) -> set[str]:
        return {c.asset_id for _, c in self.all_clips()}

    def sort_cues(self) -> None:
        # by the values as they are SAVED (6 decimals): a project that is saved, reopened or undone must come back in the
        # very order it had, or two captions that start together would swap places and the file would "change"
        self.cues.sort(key=lambda q: (round(q.start, 6), round(q.end, 6), q.id))

    # ---- (de)serialisation
    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "width": self.width, "height": self.height, "fps": self.fps,
            "sample_rate": self.sample_rate,
            "video_tracks": [t.to_dict() for t in self.video_tracks],
            "audio_tracks": [t.to_dict() for t in self.audio_tracks],
            "cues": [q.to_dict() for q in self.cues],
            "markers": [m.to_dict() for m in self.markers],
            "caption_style": self.caption_style.to_dict(),
            "background": self.background, "meta": copy.deepcopy(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sequence":
        s = cls(
            id=d.get("id") or new_id("s_"), name=d.get("name", "序列"),
            width=int(_num(d, "width", 1920)), height=int(_num(d, "height", 1080)),
            fps=_num(d, "fps", 30.0) or 30.0, sample_rate=int(_num(d, "sample_rate", 48000)),
            video_tracks=[Track.from_dict(t) for t in d.get("video_tracks", [])],
            audio_tracks=[Track.from_dict(t) for t in d.get("audio_tracks", [])],
            cues=[CaptionCue.from_dict(q) for q in d.get("cues", [])],
            markers=[Marker.from_dict(m) for m in d.get("markers", [])],
            caption_style=CaptionStyle.from_dict(d.get("caption_style")),
            background=d.get("background", "#000000"), meta=dict(d.get("meta") or {}),
        )
        for t in s.video_tracks:
            t.kind = VIDEO
        for t in s.audio_tracks:
            t.kind = AUDIO
        if not s.video_tracks:
            s.video_tracks.append(Track(new_id("t_"), "V1", VIDEO))
        s.sort_cues()
        return s


# ------------------------------------------------------------------------ validation
@dataclass
class Issue:
    level: str        # error | warn
    code: str
    message: str
    where: str = ""

    def to_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "message": self.message, "where": self.where}


def clip_issues(c: Clip) -> list[tuple[str, str]]:
    """Problems of one clip on its own (``(code, message)``): a start before 0, a clip under one frame long, or one that
    ends beyond the longest timeline there is."""
    out: list[tuple[str, str]] = []
    if c.start < -TOUCH:
        out.append(("negative_start", "片段起點小於 0"))
    if c.duration < MIN_FRAGMENT:
        out.append(("too_short", f"片段太短（{c.duration:.3f}s）"))
    if c.end > MAX_TIMELINE + TOUCH:
        out.append(("too_long", "片段超出時間軸上限（24 小時）"))
    return out


def overlap_issues(tr: "Track") -> list[tuple[str, str, str]]:
    """Overlap problems of one track as ``(code, message, clip id)``: more than two layers at one instant, or a crossfade
    longer than one of its two clips (that includes a clip lying wholly inside another one)."""
    out: list[tuple[str, str, str]] = []
    active: list[Clip] = []                       # the clips before this one that are still running at its start
    for c in ordered_clips(tr.clips):
        active = [o for o in active if o.end > c.start + TOUCH]
        if len(active) > 1:
            out.append(("overlap_depth", "同一軌道同時重疊超過兩個片段", c.id))
        for o in active:
            if o.end - c.start > min(o.duration, c.duration) + TOUCH:
                out.append(("overlap_size", "交疊長度大於其中一個片段", c.id))
        active.append(c)
    return out


def geometry_issues(tr: "Track") -> list[tuple[str, str, str]]:
    """Everything about the *placement* of a track's clips that an edit must never break: ``(code, message, clip id)``."""
    out = [(code, message, c.id) for c in tr.clips for code, message in clip_issues(c)]
    return out + overlap_issues(tr)


def validate_sequence(proj: "Project", seq: Sequence) -> list[Issue]:
    """Structural checks used by tests, the auditor and the exporter."""
    out: list[Issue] = []
    if seq.width < 16 or seq.height < 16 or seq.width % 2 or seq.height % 2:
        out.append(Issue("error", "bad_size", f"輸出尺寸需為偶數且不小於 16：{seq.width}x{seq.height}", seq.id))
    if not (1 <= seq.fps <= 240):
        out.append(Issue("error", "bad_fps", f"影格率不合理：{seq.fps}", seq.id))
    for tr in seq.tracks():
        for c in tr.clips:
            where = f"{tr.name}/{c.id}"
            a = proj.assets.get(c.asset_id)
            if a is None:
                out.append(Issue("error", "missing_asset", "片段指向不存在的素材", where))
                continue
            for code, message in clip_issues(c):
                out.append(Issue("error", code, message, where))
            if not (0.25 - EPS <= c.speed <= 4.0 + EPS):
                out.append(Issue("error", "bad_speed", f"速度超出 0.25–4：{c.speed}", where))
            if a.kind != "image" and a.duration > 0:
                if c.src_in < -EPS or c.src_out > a.duration + 0.05:
                    out.append(Issue("error", "src_range", f"來源範圍 {c.src_in:.2f}–{c.src_out:.2f}s 超出素材長度 {a.duration:.2f}s", where))
            if tr.kind == VIDEO and not a.has_video and a.kind != "image":
                out.append(Issue("error", "kind_mismatch", "純音訊素材放在視訊軌", where))
            if tr.kind == AUDIO and not a.has_audio:
                out.append(Issue("error", "kind_mismatch", "沒有音訊的素材放在音訊軌", where))
        for code, message, clip_id in overlap_issues(tr):
            out.append(Issue("error", code, message, f"{tr.name}/{clip_id}"))
    for q in seq.cues:
        if q.end <= q.start + EPS:
            out.append(Issue("error", "bad_cue", "字幕結束時間早於開始時間", q.id))
        if q.end > MAX_TIMELINE + TOUCH:
            out.append(Issue("error", "too_long", "字幕超出時間軸上限（24 小時）", q.id))
    return out


# -------------------------------------------------------------------------- Project
class Project:
    def __init__(self, name: str = "未命名專案") -> None:
        self.id: str = new_id("p_")
        self.name = name
        self.created = now_iso()
        self.modified = self.created
        self.revision = 0
        self.assets: dict[str, Asset] = {}
        self.sequences: list[Sequence] = []
        self.active: str = ""
        self.meta: dict[str, Any] = {}
        self.path: Optional[str] = None   # not serialised

    @classmethod
    def new(cls, name: str = "未命名專案", with_sequence: bool = True) -> "Project":
        p = cls(name)
        if with_sequence:
            s = Sequence.create("序列 1")
            p.sequences.append(s)
            p.active = s.id
        return p

    # ---- lookups
    def sequence(self, seq_id: str | None = None) -> Sequence:
        sid = seq_id or self.active
        for s in self.sequences:
            if s.id == sid:
                return s
        if self.sequences and not seq_id:
            return self.sequences[0]
        raise KeyError(f"找不到序列 {sid}")

    def active_sequence(self) -> Sequence:
        return self.sequence(self.active)

    def asset(self, asset_id: str) -> Asset:
        try:
            return self.assets[asset_id]
        except KeyError:
            raise KeyError(f"找不到素材 {asset_id}") from None

    def find_asset_by_path(self, path: str) -> Optional[Asset]:
        norm = os.path.normcase(os.path.abspath(path))
        for a in self.assets.values():
            if os.path.normcase(os.path.abspath(a.path)) == norm:
                return a
        return None

    # ---- content snapshot (history / hashing)
    def content_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "created": self.created,
            "assets": [a.to_dict() for a in self.assets.values()],
            "sequences": [s.to_dict() for s in self.sequences],
            "active": self.active, "meta": copy.deepcopy(self.meta),
        }

    def content_json(self) -> str:
        return canonical_json(self.content_dict())

    def content_sha256(self) -> str:
        return sha256_hex(self.content_json())

    def replace_content(self, d: dict) -> None:
        self.id = d.get("id", self.id)
        self.name = d.get("name", self.name)
        self.created = d.get("created", self.created)
        self.assets = {}
        for ad in d.get("assets", []):
            a = Asset.from_dict(ad)
            self.assets[a.id] = a
        self.sequences = [Sequence.from_dict(s) for s in d.get("sequences", [])]
        self.active = d.get("active") or (self.sequences[0].id if self.sequences else "")
        if self.sequences and self.active not in {s.id for s in self.sequences}:
            self.active = self.sequences[0].id
        self.meta = copy.deepcopy(d.get("meta") or {})

    def touch(self) -> int:
        self.revision += 1
        self.modified = now_iso()
        return self.revision

    # ---- persistence
    def to_dict(self) -> dict:
        d = {"schema": SCHEMA, "app": f"autocut-studio {__version__}", "revision": self.revision,
             "modified": self.modified}
        d.update(self.content_dict())
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "Project":
        """The project in ``d`` (the parsed JSON of a file).  Anything that is not a project - a list, a number, a ``sequences`` that
        is a string, a clip that is a number ... - raises :class:`ProjectFileError`, never a bare ``AttributeError`` / ``TypeError``
        (the window shows that message instead of crashing)."""
        if not isinstance(d, dict):
            raise ProjectFileError("這不是 AutoCut 專案檔（內容不是專案資料）")
        schema = str(d.get("schema", ""))
        if not schema.startswith("autocut.project/"):
            raise ProjectFileError("這不是 AutoCut 專案檔")
        try:
            p = cls(str(d.get("name", "未命名專案")))
            p.replace_content(d)
            p.revision = int(d.get("revision", 0))
            p.modified = d.get("modified", p.modified)
        except (AttributeError, TypeError, KeyError, IndexError, ValueError, OverflowError, RecursionError) as exc:
            raise ProjectFileError(f"專案檔的內容不完整或已損毀（{type(exc).__name__}：{exc}）") from exc
        return p

    def save(self, path: str | os.PathLike | None = None) -> str:
        target = str(path or self.path or "")
        if not target:
            raise ValueError("尚未指定專案檔路徑")
        if os.path.exists(target):
            try:
                with open(target, "rb") as src:
                    prev = src.read()
                with open(target + ".bak", "wb") as bak:
                    bak.write(prev)
            except OSError:
                pass
        atomic_write_text(target, json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Project":
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (ValueError, RecursionError) as exc:              # not JSON (also: not UTF-8, or nested without end)
            raise ProjectFileError(f"檔案內容不是有效的專案檔（{type(exc).__name__}：{exc}）") from exc
        p = cls.from_dict(d)
        p.path = str(path)
        return p
