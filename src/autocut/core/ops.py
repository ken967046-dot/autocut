"""Editing operations.

Every function mutates the given project/sequence in place and raises
:class:`OpError` (message in 繁體中文, meant for the user) when the request is
not possible.  The GUI and CLI never call these directly: they go through
:class:`autocut.core.editor.Editor`, which wraps each call in an undoable
:class:`~autocut.core.history.History` transaction.

Conventions
-----------
* ``seq`` is a :class:`Sequence`; times are timeline seconds, snapped to the
  sequence frame grid.
* Clips on one track may overlap by at most two at a time: that overlap is a
  crossfade.  Every operation that can create an overlap resolves it
  (``overwrite`` / ``insert`` / ``reject``) unless told ``free``.
"""
from __future__ import annotations

import copy
import functools
import inspect
import math
import os
import re
from typing import Callable, Iterable, Optional

from ..util import clamp, new_id
from .geometry import limit_keys, scale_keys, shift_keys, split_keys
from .project import (AUDIO, FIT_MODES, MIN_FRAGMENT, VIDEO, Asset, CaptionCue, Clip, Marker, Project, Sequence, Track,
                      geometry_issues, ordered_clips)
from .timecode import EPS, MAX_TIMELINE, TOUCH, frame_of, snap_time

MIN_CLIP = 0.1
MIN_CUE = 0.2
MIN_SPEED, MAX_SPEED = 0.25, 4.0
IMAGE_DEFAULT = 5.0
INF = float("inf")


class OpError(ValueError):
    """The edit is not possible. ``str(e)`` is a user-facing 繁體中文 message."""


class OpInvalid(OpError):
    """The guard below caught an edit that would have broken the timeline.  The operations are written so that this does
    not happen (the randomised test in ``tests/test_core_fuzz.py`` fails when it does); it is the safety net, not a feature.
    ``issues`` lists what the edit would have broken: ``(track id, code, clip id)``."""

    issues: list = []


def _placement_problems(seq: Sequence) -> set[tuple[str, str, str]]:
    return {(tr.id, code, clip_id) for tr in seq.tracks() for code, _msg, clip_id in geometry_issues(tr)}


def _guarded(fn: Callable) -> Callable:
    """Fail closed: an operation that moves clips may not leave a track with a placement problem (a third layer, a
    crossfade longer than its clips, a clip under one frame, a negative start) that it did not already have.  The
    Editor rolls back any operation that raises, so nothing of the edit survives."""

    @functools.wraps(fn)
    def run(proj, seq, *args, **kw):
        before = _placement_problems(seq)
        result = fn(proj, seq, *args, **kw)
        fresh = _placement_problems(seq) - before
        if fresh:
            if all(code == "too_long" for _tid, code, _cid in fresh):      # an ordinary refusal, not a bug of the ops
                raise OpError(TOO_LONG)
            err = OpInvalid("這個動作會讓時間軸出現無法輸出的重疊或太短的片段，已取消，什麼都沒有改")
            err.issues = sorted(fresh)
            raise err
        return result

    return run


# ----------------------------------------------------------------------- helpers
TOO_LONG = "時間軸最長 24 小時，這個動作會超過，已取消，什麼都沒有改"


def _sn(seq: Sequence, t: float) -> float:
    """A time asked for by the caller, on the frame grid and never below 0.  Beyond the longest timeline it is refused: a
    number like 1e9 (or ``inf``) must not become a position (``snap_time`` cannot even snap ``inf``)."""
    t = max(0.0, float(t))
    if t > MAX_TIMELINE + 1.0:
        raise OpError(TOO_LONG)
    s = snap_time(t, seq.fps)
    if s > MAX_TIMELINE + EPS:
        raise OpError(TOO_LONG)
    return s


def _track(seq: Sequence, track_id: str, writable: bool = True) -> Track:
    t = seq.find_track(track_id)
    if t is None:
        raise OpError("找不到這條軌道")
    if writable and t.locked:
        raise OpError(f"軌道「{t.name}」已鎖定")
    return t


def _find(seq: Sequence, clip_id: str, writable: bool = True) -> tuple[Track, Clip]:
    tr, c = seq.find_clip(clip_id)
    if c is None or tr is None:
        raise OpError("找不到這個片段")
    if writable and tr.locked:
        raise OpError(f"軌道「{tr.name}」已鎖定")
    return tr, c


def _src_limit(proj: Project, c: Clip) -> float:
    a = proj.assets.get(c.asset_id)
    if a is None or a.kind == "image" or a.duration <= 0:
        return INF
    return a.duration


def _clamp_fades(c: Clip) -> None:
    d = c.duration
    c.fade_in = clamp(c.fade_in, 0.0, d)
    c.fade_out = clamp(c.fade_out, 0.0, max(0.0, d - c.fade_in))


def _split_obj(c: Clip, at: float) -> Clip:
    """Cut ``c`` at timeline time ``at`` (strictly inside it). ``c`` becomes the left half;
    the right half is returned and still has to be added to a track."""
    d = at - c.start
    right = copy.deepcopy(c)
    right.id = new_id("c_")
    right.start = at
    right.src_in = c.src_in + d * c.speed
    lk, rk = split_keys(c.transform.keys, d)
    c.transform.keys = lk
    right.transform.keys = rk
    c.src_out = right.src_in
    right.fade_in = 0.0
    c.fade_out = 0.0
    _clamp_fades(c)
    _clamp_fades(right)
    return right


def _trim_head_obj(c: Clip, t: float) -> None:
    d = t - c.start
    if abs(d) < EPS:
        return
    c.src_in += d * c.speed
    c.start = t
    c.transform.keys = shift_keys(c.transform.keys, d)
    if d > 0:
        c.fade_in = 0.0
    _clamp_fades(c)


def _trim_tail_obj(c: Clip, t: float) -> None:
    old_end = c.end
    c.src_out = c.src_in + (t - c.start) * c.speed
    c.transform.keys = limit_keys(c.transform.keys, c.duration)
    if t < old_end - EPS:
        c.fade_out = 0.0
    _clamp_fades(c)


def _cuts_a_fade(c: Clip, at: float) -> str:
    """A message when a razor cut at ``at`` goes through the clip's own fade-in / fade-out, else ``""``.  A fade belongs to
    the whole clip: the right half could not carry on from where the left one had faded to, so the picture and the sound
    would jump at the cut (the same reason a cut inside a crossfade is refused)."""
    if c.fade_in > EPS and at < c.start + c.fade_in - EPS:
        return "分割點落在這個片段的淡入裡，畫面和聲音會在切口跳一下；請把播放頭移到淡入之後，或先把淡入設成 0"
    if c.fade_out > EPS and at > c.end - c.fade_out + EPS:
        return "分割點落在這個片段的淡出裡，畫面和聲音會在切口跳一下；請把播放頭移到淡出之前，或先把淡出設成 0"
    return ""


def _twin_after_cut(tr: Track, c: Clip, at: float) -> str:
    """A message when a razor cut of ``c`` at ``at`` would leave a half with exactly the span of another clip of ``tr``
    (a cut at the very start of a crossfade partner that ends together with ``c``): two clips lying exactly on top of each
    other cannot be told apart or edited one by one.  ``""`` when the cut is fine."""
    for o in tr.clips:
        if o is c:
            continue
        for a, b in ((c.start, at), (at, c.end)):
            if abs(a - o.start) <= TOUCH and abs(b - o.end) <= TOUCH:
                return "分割後會有一段和另一個片段的位置、長度完全一樣（兩層剛好疊在一起），請把播放頭往旁邊移一格"
    return ""


def _tidy_cut(track: Track, pieces: Iterable[Clip]) -> None:
    """Clean up after a cut that left the ``pieces`` behind.

    * a piece under one frame long is dropped: it would only flicker (and the timeline check refuses such clips);
    * a piece that starts together with a longer clip is dropped: it is the left-over end of a transition whose other
      half was cut away, lies wholly inside that clip, and would only be a stack of two clips that is hard to edit.
    Neither leaves a hole (the second one is covered by the longer clip, the first one is under a frame).  A piece of one
    frame or more that stands alone is kept - an editor decides what to do with it."""
    pieces = list(pieces)
    gone = {id(c) for c in pieces if c.duration < MIN_FRAGMENT}
    for p in pieces:
        if id(p) in gone:
            continue
        if any(q is not p and id(q) not in gone and abs(q.start - p.start) <= EPS and q.end >= p.end - EPS
               for q in track.clips):
            gone.add(id(p))
    if gone:
        track.clips = [c for c in track.clips if id(c) not in gone]


def _cut_range(track: Track, t0: float, t1: float, exclude: Iterable[str] = ()) -> None:
    """Free ``[t0, t1)`` on a track (overwrite semantics)."""
    ex = set(exclude)
    pieces: list[Clip] = []
    for c in list(track.clips):
        if c.id in ex or c.end <= t0 + EPS or c.start >= t1 - EPS:
            continue
        if c.start >= t0 - EPS and c.end <= t1 + EPS:
            track.clips.remove(c)
        elif c.start < t0 - EPS and c.end > t1 + EPS:
            right = _split_obj(c, t0)
            _trim_head_obj(right, t1)
            track.clips.append(right)
            pieces += [c, right]
        elif c.start < t0 - EPS:
            _trim_tail_obj(c, t0)
            pieces.append(c)
        else:
            _trim_head_obj(c, t1)
            pieces.append(c)
    _tidy_cut(track, pieces)
    track.sort()


def _make_room(track: Track, at: float, dur: float) -> None:
    """Open a gap of ``dur`` at ``at`` by pushing later material right (insert edit)."""
    pieces: list[Clip] = []
    for c in list(track.clips):
        if c.start < at - EPS and c.end > at + EPS:
            right = _split_obj(c, at)
            track.clips.append(right)
            pieces += [c, right]
    for c in track.clips:
        if c.start >= at - EPS:
            c.start += dur
    _tidy_cut(track, pieces)
    track.sort()


# -------------------------------------------------------- neighbours (a crossfade is an overlap on the track)
def _around(tr: Track, c: Clip) -> tuple[list[Clip], list[Clip]]:
    """``(before, after)``: the clips of ``tr`` that come before / after ``c`` in playing order."""
    order = ordered_clips(tr.clips)
    i = next(k for k, o in enumerate(order) if o is c)
    return order[:i], order[i + 1:]


def _neighbours(tr: Track, c: Clip) -> tuple[Optional[Clip], float, Optional[Clip], float]:
    """``(prev, a, nxt, b)``: the clip ``c`` is crossfaded with before it and after it, and the two overlaps in seconds -
    ``a`` is how far ``prev`` reaches into the start of ``c``, ``b`` how far ``c`` reaches into ``nxt``.  0.0 means a hard
    cut (or a gap) on that side.  Edits that shorten ``c`` must keep it longer than ``a + b``."""
    before, after = _around(tr, c)
    prev = nxt = None
    a = b = 0.0
    for o in before:
        if o.end - c.start > max(a, TOUCH):
            prev, a = o, o.end - c.start
    for o in after:
        if c.end - o.start > max(b, TOUCH):
            nxt, b = o, c.end - o.start
    return prev, a, nxt, b


def _later(tr: Track, c: Clip) -> list[Clip]:
    """The clips that come after ``c`` on its track.  A ripple edit moves all of them - the one that fades in over the
    tail of ``c`` as well as the ones that follow it, so a crossfade keeps its length and no hole opens."""
    return _around(tr, c)[1]


def _tail_ceiling(tr: Track, c: Clip) -> float:
    """How far the end of ``c`` may reach when nothing else moves: up to the start of the next clip - or, when it already
    fades into the next clip, until just before that clip ends.  It never forces ``c`` to be shorter than it is."""
    hi = INF
    for o in _later(tr, c):
        hi = min(hi, o.start if o.start >= c.end - TOUCH else o.end - MIN_CLIP)
    return max(hi, c.end)


def _head_floor(tr: Track, c: Clip) -> float:
    """How early the start of ``c`` may reach when nothing else moves: back to the end of the previous clip - or, when a
    clip already fades into it, until just after that clip starts.  It never forces ``c`` to start later than it does."""
    lo = 0.0
    for o in _around(tr, c)[0]:
        lo = max(lo, o.end if o.end <= c.start + TOUCH else o.start + MIN_CLIP)
    return min(lo, c.start)


def _fps_of(proj: Project, tr: Track) -> float:
    """The frame rate of the sequence that holds ``tr`` (30 when it is in none)."""
    for s in proj.sequences:
        if any(t is tr for t in s.tracks()):
            return s.fps
    return 30.0


def _shortest(overlaps: float, fps: float) -> float:
    """The least a clip may last when its crossfades take ``overlaps`` seconds of it: ``MIN_CLIP``, and always a frame more
    than the crossfades - a clip made of nothing but its crossfades could end up exactly where its neighbour is, and then
    only a random id would say which of the two is on top."""
    return max(MIN_CLIP, overlaps + 1.0 / fps)


def trim_limits(proj: Project, tr: Track, c: Clip, edge: str, ripple: bool) -> tuple[float, float]:
    """The range the timeline time of a dragged clip edge may take (``edge``: ``"l"`` in-point, ``"r"`` out-point).

    What limits it: the source (a clip cannot show more than the asset has); a minimum length that keeps the clip longer
    than the crossfades on its two ends; and, without ripple, the neighbours - a hard stop at the next clip, while a clip
    it already crossfades with can be run into but not all the way through.  ``trim_clip_start`` / ``trim_clip_end``
    clamp to this range and so does the timeline's drag preview, so the ghost shows exactly what the edit will do."""
    _prev, a, _nxt, b = _neighbours(tr, c)
    fps = _fps_of(proj, tr)
    if edge == "l":
        lo = c.start - c.src_in / c.speed
        if not ripple:
            lo = max(lo, _head_floor(tr, c))
        hi = c.end - _shortest(((a + b) if ripple else b), fps)
        return min(lo, c.start), max(hi, c.start)
    hi = c.start + (_src_limit(proj, c) - c.src_in) / c.speed
    if ripple:
        lo = c.start + _shortest(a + b, fps)
    else:
        hi = min(hi, _tail_ceiling(tr, c))
        lo = c.start + _shortest(a, fps)
    return min(lo, c.end), max(hi, c.end)


def _overlaps(track: Track, t0: float, t1: float, exclude: Iterable[str] = ()) -> list[Clip]:
    ex = set(exclude)
    return [c for c in track.clips if c.id not in ex and c.end > t0 + EPS and c.start < t1 - EPS]


RESOLVES = ("overwrite", "insert", "reject", "free")


def _check_resolve(resolve: str, allowed: tuple = RESOLVES) -> None:
    """Refuse an unknown (or, for a group, unsupported) way of resolving a collision BEFORE anything is touched."""
    if resolve == "insert" and "insert" not in allowed:
        raise OpError("同時移動多個片段不能用「插入」，請改用覆蓋，或一個一個移")
    if resolve not in allowed:
        raise OpError(f"未知的放置方式：{resolve}")


def _crossfade_partners(tr: Track, c: Clip, skip: Iterable[str] = ()) -> list[Clip]:
    """The clips of ``tr`` that ``c`` overlaps right now - the ones it crossfades with - leaving out ``skip``."""
    ex = set(skip) | {c.id}
    return [o for o in tr.clips if o.id not in ex and min(o.end, c.end) - max(o.start, c.start) > TOUCH]


def _stays_a_partner(p: Clip, start: float, end: float) -> bool:
    """After a clip moves to ``[start, end)``, ``p`` still crossfades with it: they overlap and neither covers the other."""
    if min(p.end, end) - max(p.start, start) <= TOUCH:
        return False
    return not ((start <= p.start + TOUCH and end >= p.end - TOUCH) or (p.start <= start + TOUCH and p.end >= end - TOUCH))


def _keeps_it_valid(tr: Track, edit: Callable[[Track], None]) -> bool:
    """Would ``edit`` leave ``tr`` without any placement problem it did not already have?  ``edit`` is tried on a copy.

    A partner that is kept whole while the clip lands on it is only right when everything else the clip lands on can be
    cut away without trouble: the remainder of a cut clip may end up wholly inside the partner, or make a third layer.
    Then the edit has to be a plain overwrite instead, which cuts the partner too."""
    before = {(code, cid) for code, _msg, cid in geometry_issues(tr)}
    trial = copy.deepcopy(tr)
    edit(trial)
    return all((code, cid) in before for code, _msg, cid in geometry_issues(trial))


def _place(track: Track, clip: Clip, resolve: str, keep: Iterable[str] = ()) -> None:
    """Put ``clip`` on ``track``.  ``keep``: clips that stay as they are even where ``clip`` lies over them (its crossfade
    partners, when it was only nudged)."""
    _check_resolve(resolve)
    keep = tuple(keep)
    if resolve == "overwrite":
        _cut_range(track, clip.start, clip.end, exclude=(clip.id, *keep))
    elif resolve == "insert":
        _make_room(track, clip.start, clip.duration)
    elif resolve == "reject":
        if _overlaps(track, clip.start, clip.end, exclude=(clip.id, *keep)):
            raise OpError("這個位置已經有片段了")
    track.clips.append(clip)
    track.sort()


# ----------------------------------------------- captions and markers follow the story (the main video track)
# Captions and markers are timed against the story: the main video track, V1, where the speech is.  A ripple edit on V1
# changes WHEN the material after it plays, so the captions after it must move with it, or they drift off the words.
# Edits on any other track (a music bed, an overlay) leave them alone.  The helpers below are the only place that knows
# how; every op that moves later V1 material calls the one that matches what it did.
def _story(seq: Sequence, tr: Track) -> bool:
    return bool(seq.video_tracks) and tr is seq.video_tracks[0]


def _retime_captions(seq: Sequence, start_of: Callable[[float], float], end_of: Callable[[float], float],
                     keep_marker_at: Optional[float] = None) -> None:
    """Move every caption and marker through a time map (``start_of`` for starts and markers, ``end_of`` for ends).
    A caption that comes out shorter than the minimum is kept at the minimum: a retime loses no text.  A marker right at
    ``keep_marker_at`` stays (the label of a clip whose head was extended)."""
    for q in seq.cues:
        s, e = _sn(seq, start_of(q.start)), _sn(seq, end_of(q.end))
        if (s, e) != (q.start, q.end):
            q.start, q.end = s, max(e, s + MIN_CUE)
    for m in seq.markers:
        if keep_marker_at is None or abs(m.time - keep_marker_at) > EPS:
            m.time = _sn(seq, start_of(m.time))
    seq.sort_cues()
    seq.markers.sort(key=lambda m: m.time)


def _shift_captions(seq: Sequence, at: float, delta: float, keep_marker_at_start: bool = False) -> None:
    """Everything from ``at`` on plays ``delta`` seconds later (negative: earlier) and nothing is deleted: captions that
    start at or after ``at`` and markers there move, a caption that spans ``at`` is stretched (or shortened).  With
    ``keep_marker_at_start`` a marker right at ``at`` stays."""
    if abs(delta) < EPS:
        return
    _retime_captions(seq, lambda t: t + delta if t >= at - EPS else t, lambda t: t + delta if t > at + EPS else t,
                     keep_marker_at=at if keep_marker_at_start else None)


def _scale_captions(seq: Sequence, t0: float, t1: float, k: float, move_later: bool, split: Optional[float] = None) -> None:
    """The stretch ``[t0, t1]`` (a clip whose speed changed) now takes ``k`` times as long: captions inside it stretch
    with it; the ones after it follow when ``move_later`` (ripple).  ``split``: where the next clip starts when it fades
    in over the tail of this one - from there on the words belong to that clip, which does not stretch."""
    if abs(k - 1.0) < 1e-9:
        return
    after = (t1 - t0) * (k - 1.0) if move_later else 0.0
    p = t1 if split is None else min(max(split, t0), t1)

    def start_of(t: float) -> float:
        if t <= t0 + EPS:
            return t
        return t0 + (t - t0) * k if t < p - EPS else t + after

    def end_of(t: float) -> float:
        if t <= t0 + EPS:
            return t
        return t0 + (t - t0) * k if t <= p + EPS else t + after

    _retime_captions(seq, start_of, end_of)


def _cut_captions(seq: Sequence, t0: float, t1: float, keep_marker_at_start: bool = False) -> None:
    """The stretch ``[t0, t1)`` of time is gone: captions inside it vanish, those crossing an edge are trimmed (and go
    when too little is left), later captions and markers move up, markers inside it go - except, with
    ``keep_marker_at_start``, one sitting right at ``t0`` (the label of a clip whose head was trimmed off)."""
    d = t1 - t0
    if d < EPS:
        return
    kept: list[CaptionCue] = []
    for q in seq.cues:
        if q.end <= t0 + EPS:
            kept.append(q)
            continue
        if q.start >= t1 - EPS:
            s, e = q.start - d, q.end - d
        elif q.start < t0 - EPS and q.end > t1 + EPS:
            s, e = q.start, q.end - d
        elif q.start < t0 - EPS:
            s, e = q.start, t0
        elif q.end > t1 + EPS:
            s, e = t0, q.end - d
        else:
            continue                                         # it sat entirely inside the cut stretch
        if e - s >= MIN_CUE - EPS:                           # a trimmed caption with too little left goes
            q.start = _sn(seq, s)                            # (snapped and clamped: float noise must not give -4e-16)
            q.end = max(_sn(seq, e), q.start + MIN_CUE)
            kept.append(q)
    seq.cues = kept
    seq.sort_cues()
    seq.markers = [m for m in seq.markers if not (t0 - EPS <= m.time < t1 - EPS)
                   or (keep_marker_at_start and abs(m.time - t0) <= EPS)]
    for m in seq.markers:
        if m.time >= t1 - EPS:
            m.time = _sn(seq, m.time - d)


# ----------------------------------------------------------------------- tracks
def add_track(proj: Project, seq: Sequence, kind: str, name: Optional[str] = None,
              index: Optional[int] = None) -> Track:
    if kind not in (VIDEO, AUDIO):
        raise OpError("軌道類型只能是 video 或 audio")
    lst = seq.video_tracks if kind == VIDEO else seq.audio_tracks
    name = name or (f"V{len(lst) + 1}" if kind == VIDEO else f"A{len(lst) + 1}")
    t = Track(new_id("t_"), name, kind)
    lst.insert(len(lst) if index is None else max(0, min(index, len(lst))), t)
    return t


def remove_track(proj: Project, seq: Sequence, track_id: str, force: bool = False) -> None:
    t = _track(seq, track_id)
    lst = seq.video_tracks if t.kind == VIDEO else seq.audio_tracks
    if t.kind == VIDEO and len(lst) <= 1:
        raise OpError("至少要保留一條視訊軌")
    if t.clips and not force:
        raise OpError("軌道內還有片段，請先清除")
    lst.remove(t)


def set_track_props(proj: Project, seq: Sequence, track_id: str, **props) -> Track:
    t = _track(seq, track_id, writable=False)
    for k, v in props.items():
        if k == "name":
            t.name = str(v)
        elif k in ("muted", "locked"):
            setattr(t, k, bool(v))
        elif k == "volume":
            t.volume = clamp(float(v), 0.0, 4.0)
        else:
            raise OpError(f"未知的軌道屬性：{k}")
    return t


# ----------------------------------------------------------------------- clips
def place_clip(proj: Project, seq: Sequence, asset_id: str, track_id: str, start: Optional[float] = None,
               src_in: float = 0.0, src_out: Optional[float] = None, mode: str = "overwrite",
               **props) -> Clip:
    """Put (part of) an asset on a track.

    ``mode``: ``overwrite`` (default, like dropping on a timeline), ``insert`` (ripple later
    clips), ``append`` (after the last clip), ``reject`` (fail when occupied), ``free`` (exactly there, overlaps and
    all - nothing is checked, ``validate_sequence`` reports what is wrong; for building test timelines and imports).
    """
    run = _place_clip if mode == "free" else _guarded(_place_clip)
    return run(proj, seq, asset_id, track_id, start, src_in, src_out, mode, **props)


def _place_clip(proj: Project, seq: Sequence, asset_id: str, track_id: str, start: Optional[float],
                src_in: float, src_out: Optional[float], mode: str, **props) -> Clip:
    a = proj.assets.get(asset_id)
    if a is None:
        raise OpError("找不到素材")
    tr = _track(seq, track_id)
    if tr.kind == VIDEO and not (a.has_video or a.kind == "image"):
        raise OpError("純音訊素材不能放在視訊軌，請拖到音訊軌")
    if tr.kind == AUDIO and not a.has_audio:
        raise OpError("這個素材沒有聲音，不能放在音訊軌")
    speed = clamp(float(props.pop("speed", 1.0)), MIN_SPEED, MAX_SPEED)
    if a.kind == "image":
        s_in = 0.0
        s_out = float(src_out) if src_out else IMAGE_DEFAULT
        total = INF
    else:
        total = a.duration if a.duration > 0 else IMAGE_DEFAULT
        need = MIN_CLIP * speed                                           # source seconds that last MIN_CLIP on the timeline
        s_in = clamp(float(src_in), 0.0, max(0.0, total - need))
        s_out = clamp(float(src_out) if src_out is not None else total, s_in + need, max(total, s_in + need))
    frames = max(1, frame_of((s_out - s_in) / speed, seq.fps))
    s_out = min(total, s_in + frames / seq.fps * speed)
    if (s_out - s_in) / speed < MIN_FRAGMENT:
        raise OpError("素材太短，用這個速度放不上時間軸")
    append = mode == "append" or start is None
    st = tr.end() if append else _sn(seq, start)
    c = Clip(id=new_id("c_"), asset_id=asset_id, start=st, src_in=s_in, src_out=s_out, speed=speed)
    if a.kind == "image":
        c.muted = True
    for k, v in props.items():
        if hasattr(c, k) and k not in ("id", "asset_id"):
            setattr(c, k, v)
    _place(tr, c, "free" if append else mode)
    if mode == "insert" and not append and _story(seq, tr):
        _shift_captions(seq, c.start, c.duration)                           # an insert edit pushes the captions too
    return c


@_guarded
def split_clip(proj: Project, seq: Sequence, clip_id: str, at: float) -> tuple[Clip, Clip]:
    """Cut one clip in two at ``at``.  Not inside a crossfade (the cut would leave a piece shorter than its overlap and
    the picture would jump at the cut): move the playhead out of the transition, or set the crossfade to 0 first."""
    tr, c = _find(seq, clip_id)
    at = _sn(seq, at)
    _prev, a, _nxt, b = _neighbours(tr, c)
    if at < c.start + max(MIN_CLIP, a) - EPS or at > c.end - max(MIN_CLIP, b) + EPS:
        if (a > 0 and at < c.start + a - EPS) or (b > 0 and at > c.end - b + EPS):
            raise OpError("分割點落在交叉淡化裡，請把播放頭移到轉場之外")
        raise OpError("分割點太靠近片段邊緣")
    problem = _cuts_a_fade(c, at) or _twin_after_cut(tr, c, at)
    if problem:
        raise OpError(problem)
    right = _split_obj(c, at)
    tr.clips.append(right)
    tr.sort()
    return c, right


@_guarded
def split_at_time(proj: Project, seq: Sequence, at: float, clip_ids: Optional[Iterable[str]] = None,
                  track_ids: Optional[Iterable[str]] = None) -> list[Clip]:
    """Razor: cut every (selected / unlocked) clip under ``at``. Returns the new right halves.

    Not inside a crossfade: a crossfade is the overlap of two clips, so a cut through it would leave two short
    crossfades one after the other and the picture would jump back from the second clip to the first at the cut
    (measured on a red -> green transition).  The whole razor is refused then, nothing is cut on any track."""
    at = _sn(seq, at)
    want = set(clip_ids) if clip_ids else None
    tracks_ok = set(track_ids) if track_ids else None
    plan: list[tuple[Track, Clip]] = []
    for tr in seq.tracks():
        if tr.locked or (tracks_ok is not None and tr.id not in tracks_ok):
            continue
        for c in tr.clips:
            if (want is not None and c.id not in want) or not c.start + MIN_CLIP - EPS < at < c.end - MIN_CLIP + EPS:
                continue
            _prev, a, _nxt, b = _neighbours(tr, c)
            if (a > 0 and at < c.start + a - EPS) or (b > 0 and at > c.end - b + EPS):
                raise OpError("播放頭落在交叉淡化裡，在這裡分割會讓轉場的畫面跳動；請把播放頭移到轉場之外")
            problem = _cuts_a_fade(c, at) or _twin_after_cut(tr, c, at)
            if problem:
                raise OpError(problem)
            plan.append((tr, c))
    if not plan:
        raise OpError("播放頭下沒有可分割的片段")
    made: list[Clip] = []
    for tr, c in plan:
        right = _split_obj(c, at)
        tr.clips.append(right)
        made.append(right)
    for tr in {id(tr): tr for tr, _c in plan}.values():
        tr.sort()
    return made


@_guarded
def trim_clip_start(proj: Project, seq: Sequence, clip_id: str, new_start: float, ripple: bool = False) -> Clip:
    """Move a clip's in-point on the timeline. ``ripple``: the clip stays put and later clips follow.
    The range the edge may take is :func:`trim_limits`; a value outside it is clamped."""
    tr, c = _find(seq, clip_id)
    new_start = snap_time(float(new_start), seq.fps)      # not clamped at 0: a ripple trim may reveal footage "before" 0
    old_start = c.start
    lo, hi = trim_limits(proj, tr, c, "l", ripple)
    target = clamp(new_start, lo, hi)
    d = target - old_start
    if abs(d) < EPS:
        return c
    followers = _later(tr, c) if ripple else []
    c.src_in += d * c.speed
    c.transform.keys = shift_keys(c.transform.keys, d)
    if d > 0:
        c.fade_in = 0.0
    if ripple:
        for o in followers:
            o.start -= d
        if _story(seq, tr):                                # the head that went (d > 0) / came back (d < 0) is time too
            if d > 0:
                _cut_captions(seq, old_start, old_start + d, keep_marker_at_start=True)
            else:
                _shift_captions(seq, old_start, -d, keep_marker_at_start=True)
    else:
        c.start = old_start + d
    _clamp_fades(c)
    tr.sort()
    return c


@_guarded
def trim_clip_end(proj: Project, seq: Sequence, clip_id: str, new_end: float, ripple: bool = False) -> Clip:
    """Move a clip's out-point on the timeline. ``ripple``: later clips follow (and keep their crossfade).
    The range the edge may take is :func:`trim_limits`; a value outside it is clamped."""
    tr, c = _find(seq, clip_id)
    new_end = _sn(seq, new_end)
    old_end = c.end
    lo, hi = trim_limits(proj, tr, c, "r", ripple)
    target = clamp(new_end, lo, hi)
    d = target - old_end
    if abs(d) < EPS:
        return c
    followers = _later(tr, c) if ripple else []
    after = min((o.start for o in followers if o.start < old_end - EPS), default=old_end)   # a crossfade starts earlier
    c.src_out = c.src_in + (target - c.start) * c.speed
    c.transform.keys = limit_keys(c.transform.keys, c.duration)
    if d < 0:
        c.fade_out = 0.0
    for o in followers:
        o.start += d
    if ripple and _story(seq, tr):                         # what comes after moves by d: the captions with it
        if d < 0:
            _cut_captions(seq, after + d, after)
        else:
            _shift_captions(seq, after, d)
    _clamp_fades(c)
    tr.sort()
    return c


def slip_clip(proj: Project, seq: Sequence, clip_id: str, delta_src: float) -> Clip:
    """Slip edit: show a different part of the source without moving the clip on the timeline."""
    tr, c = _find(seq, clip_id)
    span = c.src_out - c.src_in
    lim = _src_limit(proj, c)
    new_in = clamp(c.src_in + float(delta_src), 0.0, max(0.0, lim - span))
    c.src_out = new_in + span
    c.src_in = new_in
    return c


def _move_within(t: Track, clip_id: str, start: float, keep: Iterable[str]) -> None:
    """The overwrite move of :func:`move_clip` on one track - what :func:`_keeps_it_valid` tries on a copy first."""
    c = next(o for o in t.clips if o.id == clip_id)
    t.clips.remove(c)
    c.start = start
    _place(t, c, "overwrite", keep=keep)


def _shift_within(t: Track, ids: set, d: float, keep: dict) -> None:
    """The overwrite move of :func:`move_clips` on one track (``keep``: moved clip id -> clips that stay whole)."""
    here = [c for c in t.clips if c.id in ids]
    for c in here:
        c.start += d
    for c in here:
        _cut_range(t, c.start, c.end, exclude=ids | keep[c.id])
    t.sort()


@_guarded
def move_clip(proj: Project, seq: Sequence, clip_id: str, new_start: float, track_id: Optional[str] = None,
              resolve: str = "overwrite") -> Clip:
    """Move a clip (to another track with ``track_id``).  What it lands on is resolved by ``resolve``: ``overwrite`` cuts
    it away, ``insert`` pushes it right, ``reject`` refuses, ``free`` leaves the overlap.  A clip that is already
    crossfaded with this one is not "landed on": nudging a clip only changes how long its crossfades are (unless it would
    now cover the neighbour completely).  Moving a clip to where it already is changes nothing at all."""
    _check_resolve(resolve)
    tr, c = _find(seq, clip_id)
    dst = _track(seq, track_id) if track_id else tr
    if dst.kind != tr.kind:
        raise OpError("視訊片段不能移到音訊軌（反之亦然）")
    st = _sn(seq, new_start)
    if dst is tr and abs(st - c.start) < EPS:
        return c
    keep = {p.id for p in _crossfade_partners(tr, c) if _stays_a_partner(p, st, st + c.duration)} if dst is tr else set()
    if keep and resolve == "overwrite" and not _keeps_it_valid(tr, lambda t: _move_within(t, c.id, st, keep)):
        keep = set()                           # cutting what else it lands on would break the partner: overwrite all of it
    if resolve == "reject" and _overlaps(dst, st, st + c.duration, exclude=(c.id, *keep)):
        raise OpError("這個位置已經有片段了")
    tr.clips.remove(c)
    c.start = st
    _place(dst, c, resolve, keep=keep)
    if resolve == "insert" and _story(seq, dst):
        _shift_captions(seq, c.start, c.duration)
    return c


@_guarded
def move_clips(proj: Project, seq: Sequence, clip_ids: Iterable[str], delta: float,
               resolve: str = "overwrite") -> list[Clip]:
    """Move several clips by the same time offset (keeps their relative layout).  ``resolve``: ``overwrite`` (default),
    ``reject`` or ``free``; a group cannot be inserted.  Like :func:`move_clip`, a clip that was crossfaded with a moved
    one stays whole."""
    _check_resolve(resolve, ("overwrite", "reject", "free"))
    ids = list(dict.fromkeys(clip_ids))
    moved: list[tuple[Track, Clip]] = [_find(seq, i) for i in ids]
    if not moved:
        return []
    d = snap_time(float(delta), seq.fps) if delta >= 0 else -snap_time(-float(delta), seq.fps)
    d = max(d, -min(c.start for _, c in moved))
    if abs(d) < EPS:
        return [c for _, c in moved]
    idset = set(ids)
    keep = {c.id: {p.id for p in _crossfade_partners(tr, c, idset) if _stays_a_partner(p, c.start + d, c.end + d)}
            for tr, c in moved}
    if resolve == "reject":
        for tr, c in moved:
            if _overlaps(tr, c.start + d, c.end + d, exclude=idset | keep[c.id]):
                raise OpError("這個位置已經有片段了")
    if resolve == "overwrite":                         # a partner stays whole only where the rest can be cut cleanly
        for tr in {id(t): t for t, _ in moved}.values():
            mine = {c.id for t, c in moved if t is tr}
            if any(keep[i] for i in mine) and not _keeps_it_valid(tr, lambda t, mine=mine: _shift_within(t, mine, d, keep)):
                for i in mine:
                    keep[i] = set()
    for _, c in moved:
        c.start += d
    if resolve == "overwrite":
        for tr in {id(t): t for t, _ in moved}.values():
            for c in [c for t, c in moved if t is tr]:
                _cut_range(tr, c.start, c.end, exclude=idset | keep[c.id])
    for tr in {id(t): t for t, _ in moved}.values():
        tr.sort()
    return [c for _, c in moved]


def _inside(spans: list[list[float]], a: float, b: float) -> float:
    """Total length of the sorted, disjoint ``spans`` that lies inside ``[a, b]``."""
    return sum(max(0.0, min(y, b) - max(x, a)) for x, y in spans)


def _holes_left_by(tr: Track, removed: list[Clip]) -> list[tuple[float, float]]:
    """The parts of the removed material that no remaining clip of ``tr`` covers: ``(start, end)`` in the old positions,
    ascending.  These are exactly the stretches a ripple delete closes."""
    spans: list[list[float]] = []
    for x, y in sorted((r.start, r.end) for r in removed):
        if spans and x <= spans[-1][1] + EPS:
            spans[-1][1] = max(spans[-1][1], y)
        else:
            spans.append([x, y])
    cover = sorted((c.start, c.end) for c in tr.clips)
    holes: list[tuple[float, float]] = []
    for x, y in spans:
        cur = x
        for cs, ce in cover:
            if ce <= cur + EPS:
                continue
            if cs >= y - EPS:
                break
            if cs > cur + EPS:
                holes.append((cur, cs))
            cur = max(cur, ce)
            if cur >= y - EPS:
                break
        if cur < y - EPS:
            holes.append((cur, y))
    return holes


def _close_removed_holes(tr: Track, removed: list[Clip]) -> list[tuple[float, float]]:
    """Pull the clips of ``tr`` back over the holes that taking ``removed`` out of it left (ripple delete); returns the
    holes (old positions) so the captions can close them too.

    A hole is the part of the removed material that no remaining clip covers.  A remaining clip that overlapped a removed
    one (a crossfade) covers its share, so it moves up only as far as the end of the clip before it: the picture never
    gets a black gap, and a gap the person left on purpose between two other clips stays as it was."""
    holes = _holes_left_by(tr, removed)
    spans: list[list[float]] = []
    for x, y in sorted((r.start, r.end) for r in removed):
        if spans and x <= spans[-1][1] + EPS:
            spans[-1][1] = max(spans[-1][1], y)
        else:
            spans.append([x, y])
    shift = covered = 0.0                              # `covered`: how far the remaining clips reach (old positions)
    for c in sorted(tr.clips, key=lambda c: (c.start, c.end)):
        start, end = c.start, c.end
        if start > covered + EPS:
            shift += _inside(spans, covered, start)
        covered = max(covered, end)
        c.start = max(0.0, start - shift)
    return holes


@_guarded
def remove_clips(proj: Project, seq: Sequence, clip_ids: Iterable[str], ripple: bool = False) -> int:
    """Delete clips.  ``ripple``: close the holes on those tracks (later clips move up); on the main video track the
    captions and markers close them too.  Everything is checked before anything is deleted."""
    ids = set(clip_ids)
    plan: list[tuple[Track, list[Clip]]] = []
    for tr in seq.tracks():
        rem = [c for c in tr.clips if c.id in ids]
        if not rem:
            continue
        if tr.locked:
            raise OpError(f"軌道「{tr.name}」已鎖定")
        plan.append((tr, rem))
    found = sum(len(rem) for _tr, rem in plan)
    if found != len(ids):
        raise OpError("有些片段已經不存在")
    story_holes: list[tuple[float, float]] = []
    for tr, rem in plan:
        tr.clips = [c for c in tr.clips if c.id not in ids]
        if ripple:
            holes = _close_removed_holes(tr, rem)
            if _story(seq, tr):
                story_holes = holes
        tr.sort()
    for x, y in reversed(story_holes):                 # last first: the earlier positions stay valid
        _cut_captions(seq, x, y)
    return found


@_guarded
def ripple_delete_range(proj: Project, seq: Sequence, t0: float, t1: float,
                        track_ids: Optional[Iterable[str]] = None, captions: bool = True) -> float:
    """Delete ``[t0, t1)`` on the chosen tracks (default: all) and close the gap. Returns seconds removed."""
    t0, t1 = _sn(seq, t0), _sn(seq, t1)
    if t1 - t0 < EPS:
        return 0.0
    d = t1 - t0
    tracks = [_track(seq, i) for i in track_ids] if track_ids else [t for t in seq.tracks() if not t.locked]
    for tr in tracks:
        pieces: list[Clip] = []
        for cut in (t1, t0):
            for c in list(tr.clips):
                if c.start + EPS < cut < c.end - EPS:
                    right = _split_obj(c, cut)
                    tr.clips.append(right)
                    pieces += [c, right]
        tr.sort()
        tr.clips = [c for c in tr.clips if not (c.start >= t0 - EPS and c.end <= t1 + EPS)]
        for c in tr.clips:
            if c.start >= t1 - EPS:
                c.start -= d
        _tidy_cut(tr, pieces)
        tr.sort()
    if captions:
        _cut_captions(seq, t0, t1)
    return d


@_guarded
def close_gaps(proj: Project, seq: Sequence, track_id: str) -> float:
    """Remove every gap after the first clip of a track (existing crossfades are kept). Returns seconds closed.  On the
    main video track the captions and markers close the gaps too."""
    tr = _track(seq, track_id)
    tr.sort()
    closed = shift = 0.0
    prev_end: Optional[float] = None
    gaps: list[tuple[float, float]] = []                                    # (old start, old end) of each gap closed
    for c in tr.clips:
        c.start -= shift
        if prev_end is not None and c.start > prev_end + EPS:
            gap = c.start - prev_end
            gaps.append((prev_end + shift, prev_end + shift + gap))
            c.start = prev_end
            shift += gap
            closed += gap
        prev_end = max(prev_end if prev_end is not None else 0.0, c.end)
    if _story(seq, tr):
        for x, y in reversed(gaps):
            _cut_captions(seq, x, y)
    return closed


@_guarded
def set_speed(proj: Project, seq: Sequence, clip_id: str, speed: float, ripple: bool = True) -> Clip:
    """Play a clip faster / slower.  ``ripple``: the clips after it move with its end (and keep their crossfade); without
    ripple it may not grow into the next clip.  A speed that would make the clip shorter than 0.1 s - or than the
    crossfades on its ends - is refused."""
    tr, c = _find(seq, clip_id)
    speed = clamp(float(speed), MIN_SPEED, MAX_SPEED)
    if abs(speed - c.speed) < 1e-9:
        return c
    _prev, a, _nxt, b = _neighbours(tr, c)
    old_dur = c.duration
    new_dur = (c.src_out - c.src_in) / speed
    if new_dur < _shortest((a + b) if ripple else a, seq.fps) - EPS:
        raise OpError("這個速度會讓片段短於 0.1 秒，或比它兩端的交叉淡化還短")
    if not ripple and new_dur > old_dur and c.start + new_dur > _tail_ceiling(tr, c) + EPS:
        raise OpError("後面的片段擋住了，放不下變慢後的長度（可以改用波紋，或先把後面的片段移開）")
    followers = _later(tr, c) if ripple else []
    t0, t1 = c.start, c.end
    split = min((o.start for o in _later(tr, c) if o.start < t1 - EPS), default=t1)     # a clip fading in over the tail
    c.speed = speed
    c.transform.keys = scale_keys(c.transform.keys, new_dur / old_dur if old_dur else 1.0)
    for o in followers:
        o.start += new_dur - old_dur
    if _story(seq, tr) and old_dur > 0:                    # the words inside play faster / slower, what follows moves
        _scale_captions(seq, t0, t1, new_dur / old_dur, move_later=ripple, split=split)
    _clamp_fades(c)
    tr.sort()
    return c


@_guarded
def set_crossfade(proj: Project, seq: Sequence, clip_id: str, dur: float) -> float:
    """Make the clip overlap the previous clip on its track by ``dur`` seconds (a crossfade).

    Later clips move with it, so total length changes by the difference.  ``dur = 0`` restores a plain cut.
    Returns the crossfade length actually applied: at most half of the shorter clip, and never so long that the clip
    would reach back into a clip before the one it fades over; rounded down to whole frames."""
    tr, b = _find(seq, clip_id)
    if any(o is not b and abs(o.start - b.start) <= TOUCH for o in tr.clips):
        raise OpError("有另一個片段和它同時開始，沒辦法設定交叉淡化；請先刪除或移開其中一個")
    prevs = [o for o in tr.clips if o.start < b.start - TOUCH]               # everything that starts earlier stays put
    if not prevs:
        raise OpError("前面沒有片段可以做交叉淡化")
    a = ordered_clips(prevs)[-1]
    before_a = max((o.end for o in prevs if o is not a), default=0.0)       # the clip may not reach back past these
    _p, _old, _n, tail = _neighbours(tr, b)                                 # b's own crossfade into the next clip stays
    limit = min(0.5 * min(a.duration, b.duration), a.end - before_a, b.duration - tail - 1.0 / seq.fps)
    dur = clamp(float(dur), 0.0, max(0.0, limit))
    dur = math.floor(dur * seq.fps + 1e-6) / seq.fps
    if dur < EPS and a.end - b.start <= TOUCH:
        return 0.0                  # "no crossfade" on a clip that has none is already true - a gap left on purpose stays
    new_start = a.end - dur
    delta = new_start - b.start
    if abs(delta) > EPS:
        at = b.start
        for c in tr.clips:                                                  # b and all that starts with it or later
            if c.start >= b.start - TOUCH:
                c.start += delta
        if _story(seq, tr):
            _shift_captions(seq, at, delta)                                 # their words moved with them
    tr.sort()
    return dur


_CLIP_SIMPLE = {"volume", "muted", "enabled", "label", "color", "note"}
_TRANSFORM = {"fit", "zoom", "cx", "cy", "flip_h"}
_LOOK = {"brightness", "contrast", "saturation"}


def set_clip_props(proj: Project, seq: Sequence, clip_id: str, **props) -> Clip:
    """Change simple clip properties: volume, muted, enabled, fade_in, fade_out, label, color, note,
    fit, zoom, cx, cy, flip_h, brightness, contrast, saturation."""
    tr, c = _find(seq, clip_id)
    for k, v in props.items():
        if k == "volume":
            c.volume = clamp(float(v), 0.0, 4.0)
        elif k in ("muted", "enabled"):
            setattr(c, k, bool(v))
        elif k in ("label", "color", "note"):
            setattr(c, k, str(v))
        elif k == "fade_in":
            c.fade_in = max(0.0, float(v))
        elif k == "fade_out":
            c.fade_out = max(0.0, float(v))
        elif k == "fit":
            if v not in FIT_MODES:
                raise OpError("填滿方式只能是 cover / contain / blur")
            c.transform.fit = v
        elif k == "zoom":
            c.transform.zoom = clamp(float(v), 1.0, 8.0)
        elif k in ("cx", "cy"):
            setattr(c.transform, k, clamp(float(v), 0.0, 1.0))
        elif k == "flip_h":
            c.transform.flip_h = bool(v)
        elif k in _LOOK:
            lim = {"brightness": (-1.0, 1.0), "contrast": (0.0, 3.0), "saturation": (0.0, 3.0)}[k]
            setattr(c.look, k, clamp(float(v), *lim))
        elif k == "speed":
            raise OpError("請用 set_speed 變更速度")
        else:
            raise OpError(f"未知的片段屬性：{k}")
    _clamp_fades(c)
    return c


def set_transform_key(proj: Project, seq: Sequence, clip_id: str, t_local: float, cx: float, cy: float) -> None:
    """Add/replace a crop-centre keyframe at clip-local time ``t_local`` (9:16 reframing)."""
    tr, c = _find(seq, clip_id)
    t = clamp(snap_time(max(0.0, t_local), seq.fps), 0.0, c.duration)
    cx, cy = clamp(float(cx), 0.0, 1.0), clamp(float(cy), 0.0, 1.0)
    tol = 0.5 / seq.fps
    keys = [k for k in c.transform.keys if abs(k[0] - t) > tol]
    keys.append([t, cx, cy])
    keys.sort(key=lambda k: k[0])
    c.transform.keys = keys


def remove_transform_key(proj: Project, seq: Sequence, clip_id: str, index: int) -> None:
    tr, c = _find(seq, clip_id)
    if not 0 <= index < len(c.transform.keys):
        raise OpError("沒有這個關鍵格")
    del c.transform.keys[index]


def clear_transform_keys(proj: Project, seq: Sequence, clip_id: str) -> None:
    tr, c = _find(seq, clip_id)
    c.transform.keys = []


# --------------------------------------------------------------------- captions
def add_cue(proj: Project, seq: Sequence, start: float, end: float, text: str = "", card: bool = False) -> CaptionCue:
    """A new caption; ``card=True``: a title card (字卡, its own layer at the top)."""
    s = _sn(seq, start)
    e = _sn(seq, max(end, s + MIN_CUE))
    q = CaptionCue(new_id("q_"), s, e, text, {"kind": "card"} if card else {})
    seq.cues.append(q)
    seq.sort_cues()
    return q


def update_cue(proj: Project, seq: Sequence, cue_id: str, start: Optional[float] = None,
               end: Optional[float] = None, text: Optional[str] = None, card: Optional[bool] = None) -> CaptionCue:
    """Change a caption's times and/or text; ``card`` turns it into a title card (字卡) or back into a subtitle."""
    q = seq.find_cue(cue_id)
    if q is None:
        raise OpError("找不到這則字幕")
    s = _sn(seq, start) if start is not None else q.start
    e = _sn(seq, end) if end is not None else q.end
    if e < s + MIN_CUE - EPS:
        if end is None:
            e = s + (q.end - q.start)
        else:
            e = s + MIN_CUE
    q.start, q.end = s, e
    if text is not None:
        q.text = text
    if card is not None and bool(card) != q.is_card:
        style = {k: v for k, v in (q.style if isinstance(q.style, dict) else {}).items() if k != "kind"}
        if card:
            style["kind"] = "card"
        q.style = style
    seq.sort_cues()
    return q


def remove_cues(proj: Project, seq: Sequence, ids: Iterable[str]) -> int:
    ids = set(ids)
    n = len(seq.cues)
    seq.cues = [q for q in seq.cues if q.id not in ids]
    return n - len(seq.cues)


def cue_split_ok(seq: Sequence, q: CaptionCue, at: float) -> bool:
    """Whether a caption can be cut at ``at``: both halves must last at least ``MIN_CUE`` (no caption is shorter than that).
    The one answer the timeline and :func:`split_cue` both go by."""
    at = _sn(seq, at)
    return q.start + MIN_CUE - EPS <= at <= q.end - MIN_CUE + EPS


def split_cue(proj: Project, seq: Sequence, cue_id: str, at: float, text_left: Optional[str] = None,
              text_right: Optional[str] = None) -> tuple[CaptionCue, CaptionCue]:
    q = seq.find_cue(cue_id)
    if q is None:
        raise OpError("找不到這則字幕")
    at = _sn(seq, at)
    if not cue_split_ok(seq, q, at):
        raise OpError("分割點太靠近字幕邊緣")
    if text_left is None or text_right is None:
        k = int(round(len(q.text) * (at - q.start) / (q.end - q.start)))
        text_left, text_right = q.text[:k].strip(), q.text[k:].strip()
    right = CaptionCue(new_id("q_"), at, q.end, text_right, dict(q.style))
    q.end, q.text = at, text_left
    seq.cues.append(right)
    seq.sort_cues()
    return q, right


def merge_cues(proj: Project, seq: Sequence, ids: Iterable[str]) -> CaptionCue:
    sel = sorted((q for q in seq.cues if q.id in set(ids)), key=lambda q: q.start)
    if len(sel) < 2:
        raise OpError("請選擇至少兩則字幕")
    if len({q.is_card for q in sel}) > 1:
        raise OpError("字卡和字幕不能合併")

    def cjk(ch: str) -> bool:
        return bool(ch) and ord(ch) >= 0x2E80

    text = sel[0].text
    for q in sel[1:]:
        if not text or not q.text or text.endswith(" ") or q.text.startswith(" "):
            text += q.text
        elif cjk(text[-1]) or cjk(q.text[0]):
            text += q.text
        else:
            text += " " + q.text
    keep = sel[0]
    keep.end = max(q.end for q in sel)
    keep.text = text
    drop = {q.id for q in sel[1:]}
    seq.cues = [q for q in seq.cues if q.id not in drop]
    seq.sort_cues()
    return keep


def shift_cues(proj: Project, seq: Sequence, delta: float, ids: Optional[Iterable[str]] = None) -> None:
    want = set(ids) if ids is not None else None
    for q in seq.cues:
        if want is None or q.id in want:
            s = _sn(seq, q.start + delta)                                  # on the frame grid, never below zero
            q.end = max(s + MIN_CUE, _sn(seq, q.end + (s - q.start)))
            q.start = s
    seq.sort_cues()


def replace_cues(proj: Project, seq: Sequence, cues: Iterable, keep_cards: bool = True) -> list[CaptionCue]:
    """Replace all subtitles. Items may be CaptionCue, dicts or ``(start, end, text)`` tuples.  Title cards (字卡) are
    their own layer and stay unless ``keep_cards`` is false (importing or recognising subtitles never wipes them).
    Returns the new cues."""
    out: list[CaptionCue] = []
    for it in cues:
        if isinstance(it, CaptionCue):
            q = copy.deepcopy(it)
        elif isinstance(it, dict):
            q = CaptionCue.from_dict(it)
        else:
            s, e, text = it
            q = CaptionCue(new_id("q_"), float(s), float(e), str(text))
        q.start, q.end = _sn(seq, q.start), _sn(seq, q.end)
        if q.end < q.start + MIN_CUE - EPS:
            q.end = q.start + MIN_CUE
        out.append(q)
    seq.cues = ([q for q in seq.cues if q.is_card] if keep_cards else []) + out
    seq.sort_cues()
    return out


def set_caption_style(proj: Project, seq: Sequence, **kw) -> None:
    for k, v in kw.items():
        if not hasattr(seq.caption_style, k):
            raise OpError(f"未知的字幕樣式：{k}")
        setattr(seq.caption_style, k, v)


def set_sequence_range(proj: Project, seq: Sequence, t_in: Optional[float], t_out: Optional[float]) -> None:
    """Set (or clear with ``None``) the export in/out range stored in ``seq.meta['range']``."""
    if t_in is None or t_out is None:
        seq.meta.pop("range", None)
        return
    a, b = _sn(seq, t_in), _sn(seq, t_out)
    if b - a < 1.0 / seq.fps - EPS:
        raise OpError("入點必須早於出點")
    seq.meta["range"] = [a, b]


# ---------------------------------------------------------------------- markers
def add_marker(proj: Project, seq: Sequence, time: float, name: str = "", color: str = "#f5a623") -> Marker:
    m = Marker(new_id("m_"), _sn(seq, time), name, color)
    seq.markers.append(m)
    seq.markers.sort(key=lambda x: x.time)
    return m


def remove_marker(proj: Project, seq: Sequence, marker_id: str) -> None:
    seq.markers = [m for m in seq.markers if m.id != marker_id]


# ---------------------------------------------------------------------- sequences
def create_sequence(proj: Project, name: str = "新序列", width: int = 1920, height: int = 1080, fps: float = 30.0,
                    activate: bool = True, video_tracks: int = 2, audio_tracks: int = 2) -> Sequence:
    if width < 16 or height < 16:
        raise OpError("尺寸太小")
    s = Sequence.create(name, width - width % 2, height - height % 2, fps, video_tracks, audio_tracks)
    proj.sequences.append(s)
    if activate or not proj.active:
        proj.active = s.id
    return s


_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


def set_sequence_format(proj: Project, seq: Sequence, width: Optional[int] = None, height: Optional[int] = None,
                        fps: Optional[float] = None, background: Optional[str] = None) -> Sequence:
    """Size, frame rate and background colour (``#RRGGBB``) of a sequence.  A bad colour is refused before anything changes."""
    if background is not None and not (isinstance(background, str) and _HEX_COLOR.fullmatch(background)):
        raise OpError("背景色的格式要像 #3A6EA5（# 加六位十六進位數字）")
    if background is not None:
        seq.background = background
    if width:
        seq.width = int(width) - int(width) % 2
    if height:
        seq.height = int(height) - int(height) % 2
    if fps:
        seq.fps = float(fps)
        for _, c in seq.all_clips():
            c.start = snap_time(c.start, seq.fps)
        for q in seq.cues:
            q.start, q.end = snap_time(q.start, seq.fps), snap_time(q.end, seq.fps)
    return seq


def rename_sequence(proj: Project, seq: Sequence, name: str) -> None:
    seq.name = name.strip() or seq.name


def delete_sequence(proj: Project, seq_id: str) -> None:
    if len(proj.sequences) <= 1:
        raise OpError("至少要保留一個序列")
    proj.sequences = [s for s in proj.sequences if s.id != seq_id]
    if proj.active == seq_id:
        proj.active = proj.sequences[0].id


def duplicate_sequence(proj: Project, seq: Sequence) -> Sequence:
    """A copy with fresh ids.  The copy is a NEW piece of work: it does not inherit the link to the auto-edit run
    (``meta["ai"]``), so a technical check or a person's review of the original never counts for it."""
    d = copy.deepcopy(seq.to_dict())
    d["id"] = new_id("s_")
    d["name"] = seq.name + " 複本"
    for t in d["video_tracks"] + d["audio_tracks"]:
        t["id"] = new_id("t_")
        for c in t["clips"]:
            c["id"] = new_id("c_")
    for q in d["cues"]:
        q["id"] = new_id("q_")
    for m in d["markers"]:
        m["id"] = new_id("m_")
    meta = dict(d.get("meta") or {})
    meta.pop("ai", None)
    d["meta"] = meta
    s = Sequence.from_dict(d)
    proj.sequences.append(s)
    proj.active = s.id
    return s


# ------------------------------------------------------------------------ assets
def add_asset(proj: Project, asset: Asset) -> Asset:
    """Register an asset (deduplicated by path). Returns the registered asset."""
    existing = proj.find_asset_by_path(asset.path)
    if existing is not None:
        return existing
    proj.assets[asset.id] = asset
    return asset


def remove_asset(proj: Project, asset_id: str, force: bool = False) -> int:
    """Remove an asset; with ``force`` also delete every clip that uses it. Returns clips removed."""
    if asset_id not in proj.assets:
        raise OpError("找不到素材")
    users = [(s, t, c) for s in proj.sequences for t, c in s.all_clips() if c.asset_id == asset_id]
    if users and not force:
        raise OpError(f"這個素材正被 {len(users)} 個片段使用")
    for s, t, c in users:
        t.clips.remove(c)
    del proj.assets[asset_id]
    return len(users)


def relink_asset(proj: Project, asset_id: str, info: Asset) -> Asset:
    """Point an asset at a moved/renamed file (``info`` = freshly probed metadata of the new file)."""
    a = proj.assets.get(asset_id)
    if a is None:
        raise OpError("找不到素材")
    keep = a.id
    for k, v in info.to_dict().items():
        setattr(a, k, v)
    a.id = keep
    return a


def normalize_sequence(proj: Project, seq: Sequence) -> None:
    """Snap everything to the frame grid and clamp fades (used after imports/auto-edits)."""
    for tr, c in seq.all_clips():
        c.start = snap_time(max(0.0, c.start), seq.fps)
        _clamp_fades(c)
        tr.sort()
    for q in seq.cues:
        q.start, q.end = snap_time(q.start, seq.fps), snap_time(q.end, seq.fps)
    seq.sort_cues()


# ------------------------------------------------------------------ numbers that are not numbers
def _bad_number(v) -> bool:
    """NaN or infinity anywhere in an argument (numbers, lists, dicts of numbers)."""
    if isinstance(v, float):
        return not math.isfinite(v)
    if isinstance(v, (list, tuple, set, frozenset)):
        return any(_bad_number(x) for x in v)
    if isinstance(v, dict):
        return any(_bad_number(x) for x in v.values())
    return False


def _finite(fn: Callable) -> Callable:
    """An edit given NaN or infinity is refused before it touches anything: such a number would be written into the project
    file (as ``NaN``, which is not even valid JSON) and nothing could place a clip there."""

    @functools.wraps(fn)
    def run(proj, *args, **kw):
        if any(_bad_number(v) for v in (*args, *kw.values())):
            raise OpError("數值不正確（不是有效的數字），已取消，什麼都沒有改")
        return fn(proj, *args, **kw)

    return run


# Every public edit of this module is wrapped, so a new one is safe without anyone remembering to do it.
_QUERIES = {"trim_limits", "cue_split_ok"}                      # they only answer a question: nothing to refuse
for _name, _fn in list(globals().items()):
    if (not _name.startswith("_") and inspect.isfunction(_fn) and _fn.__module__ == __name__ and _name not in _QUERIES):
        globals()[_name] = _finite(_fn)
del _name, _fn
