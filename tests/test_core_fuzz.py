"""Randomised editing: whatever a person does with the editing operations, the timeline stays valid.

Every seed builds a small project (a crossfaded chain on V1, a few overlay clips on V2, sound on A1) and applies
random user operations in a row - through the :class:`Editor`, exactly like the window does, so a refused edit is rolled
back.  After EACH operation:

* a *refused* edit (``OpError``) changed nothing at all - not one byte of the project;
* an accepted edit leaves ``validate_sequence`` without errors (at most two layers, overlaps no longer than the clips,
  source ranges inside the asset, no clip under one frame, speeds in range), clips sorted, no negative start;
* a *ripple* edit (ripple delete, ripple trim, speed change, range delete) on a track that had no gap must not leave a
  gap (a hole is a black picture in the exported video - this is the bug the end-to-end run found);
* the safety net ``ops.OpInvalid`` never fires: the operations themselves must be right, the net is only a last resort.

Any exception other than ``OpError`` is a bug.  Seeds are fixed, so a failure is reproducible: the message carries the
seed and the last operations.

The second test (``test_wider_random_edits_...``) adds what the first one never does - moving clips between tracks,
fast clips cut from the very end of a source, locked tracks, muted / disabled clips, captions and markers, one-frame
nudges, numbers that are not numbers - and checks more after every step: captions and markers stay sorted, finite and
long enough, no two clips of a track end up congruent, an edit that changed nothing leaves no undo step, and ONE undo
gives back exactly what was there before (redo exactly what the edit made).
"""
from __future__ import annotations

import math
import random

import pytest

from autocut.core import ops
from autocut.core.editor import Editor
from autocut.core.ops import MIN_CUE, OpError, OpInvalid
from autocut.core.project import Project, Sequence, Track, validate_sequence
from autocut.core.timecode import EPS, TOUCH
from conftest import fake_asset

SEEDS = range(250)
STEPS = 26
EDGE = 0.031                                           # "no gap": a frame is 0.033 s, a piece under 0.03 s is dropped


def build(seed: int):
    rng = random.Random(seed)
    proj = Project.new(f"fuzz{seed}")
    for i, dur in enumerate((14.0, 9.0, 25.0)):
        proj.assets[f"a{i}"] = fake_asset(f"a{i}", dur)
    seq = proj.active_sequence()
    v1 = seq.video_tracks[0]
    for _ in range(rng.randint(3, 6)):
        aid = rng.choice(list(proj.assets))
        total = proj.assets[aid].duration
        s_in = round(rng.uniform(0, total - 4), 2)
        ops.place_clip(proj, seq, aid, v1.id, src_in=s_in, src_out=min(total, s_in + rng.uniform(1.5, 5.0)), mode="append")
    for clip in list(v1.clips)[1:]:
        if rng.random() < 0.55:
            try:
                ops.set_crossfade(proj, seq, clip.id, rng.choice((0.2, 0.4, 0.5, 0.7)))
            except OpError:
                pass
    v2 = ops.add_track(proj, seq, "video")
    a1 = seq.audio_tracks[0]
    for _ in range(rng.randint(0, 2)):
        ops.place_clip(proj, seq, rng.choice(list(proj.assets)), v2.id, start=rng.uniform(0, 8), src_out=rng.uniform(1.5, 4))
    if rng.random() < 0.6:
        ops.place_clip(proj, seq, "a2", a1.id, start=0, src_out=rng.uniform(5, 12))
    return rng, proj, seq


def gaps(track: Track) -> list[tuple[float, float]]:
    """Holes of a track: a start later than everything before it reached (the very first clip counts from 0)."""
    out, reach = [], 0.0
    for c in sorted(track.clips, key=lambda c: (c.start, c.end)):
        if c.start > reach + EDGE:
            out.append((round(reach, 3), round(c.start, 3)))
        reach = max(reach, c.end)
    return out


def layout(seq: Sequence) -> str:
    return " | ".join(f"{t.name}: " + " ".join(f"[{c.start:.2f}-{c.end:.2f}]" for c in sorted(t.clips, key=lambda c: c.start))
                      for t in seq.tracks())


def pick_clip(rng, seq, kind=None):
    pool = [(t, c) for t in seq.tracks() if (kind is None or t.kind == kind) for c in t.clips]
    return rng.choice(pool) if pool else (None, None)


def time_near(rng, clip, seq) -> float:
    """A time at / around the clip: its edges, its middle, slightly outside, or anywhere in the sequence."""
    choice = rng.random()
    if choice < 0.2:
        return clip.start + rng.choice((0.0, 0.03, 0.1, -0.1))
    if choice < 0.4:
        return clip.end + rng.choice((0.0, -0.03, -0.1, 0.1, 1.0))
    if choice < 0.8:
        return rng.uniform(clip.start, clip.end)
    return rng.uniform(0, max(1.0, seq.duration + 2))


def one_step(rng, proj, seq, do):
    """Apply one random operation through ``do(name, *args, **kw)``; returns ``(description, is_ripple)``.  After a ripple
    edit no track that was gap free before may have a gap (the other tracks are untouched, so checking all is exact)."""
    kind = rng.choice(["split", "trim_start", "trim_end", "trim_start", "trim_end", "move", "remove", "remove", "speed",
                       "crossfade", "place", "slip", "range", "close_gaps", "fades", "move_many", "razor"])
    tr, c = pick_clip(rng, seq)
    if c is None and kind not in ("place",):
        kind = "place"
    ripple = rng.random() < 0.6
    if kind == "split":
        at = time_near(rng, c, seq)
        do("split_clip", c.id, at)
        return f"split {c.id} at {at:.3f}", False
    if kind == "razor":
        at = time_near(rng, c, seq)
        selected = [c.id] if rng.random() < 0.4 else None
        do("split_at_time", at, clip_ids=selected)
        return f"razor at {at:.3f} selected={selected}", False
    if kind == "trim_start":
        t = time_near(rng, c, seq)
        do("trim_clip_start", c.id, t, ripple=ripple)
        return f"trim_start {c.id} -> {t:.3f} ripple={ripple}", ripple
    if kind == "trim_end":
        t = time_near(rng, c, seq)
        do("trim_clip_end", c.id, t, ripple=ripple)
        return f"trim_end {c.id} -> {t:.3f} ripple={ripple}", ripple
    if kind == "move":
        t = time_near(rng, c, seq)
        resolve = rng.choice(("overwrite", "insert", "reject"))
        do("move_clip", c.id, t, resolve=resolve)
        return f"move {c.id} -> {t:.3f} {resolve}", False
    if kind == "move_many":
        chosen = [c.id for t2 in seq.tracks() for c in t2.clips if rng.random() < 0.4][:4] or [c.id]
        d = rng.uniform(-2, 3)
        do("move_clips", chosen, d)
        return f"move_clips {chosen} by {d:.3f}", False
    if kind == "remove":
        everything = [x.id for t2 in seq.tracks() for x in t2.clips]
        chosen = rng.sample(everything, k=min(len(everything), rng.choice((1, 1, 2))))
        do("remove_clips", chosen, ripple=ripple)
        return f"remove {chosen} ripple={ripple}", ripple
    if kind == "speed":
        sp = rng.choice((0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0))
        do("set_speed", c.id, sp, ripple=ripple)
        return f"speed {c.id} x{sp} ripple={ripple}", ripple
    if kind == "crossfade":
        d = rng.choice((0.0, 0.2, 0.4, 0.8, 1.5))
        do("set_crossfade", c.id, d)
        return f"crossfade {c.id} {d}", False
    if kind == "place":
        track = rng.choice(seq.video_tracks + seq.audio_tracks)
        aid = rng.choice(list(proj.assets))
        total = proj.assets[aid].duration
        s_in = rng.uniform(0, total - 3)
        mode = rng.choice(("overwrite", "insert", "append", "reject"))
        start = rng.uniform(0, max(1.0, seq.duration))
        do("place_clip", aid, track.id, start=start, src_in=s_in, src_out=s_in + rng.uniform(0.5, 4), mode=mode)
        return f"place {aid} on {track.name} at {start:.3f} {mode}", False
    if kind == "slip":
        d = rng.uniform(-3, 3)
        do("slip_clip", c.id, d)
        return f"slip {c.id} {d:.3f}", False
    if kind == "range":
        t0 = rng.uniform(0, max(0.5, seq.duration))
        t1 = t0 + rng.uniform(0.2, 3.0)
        do("ripple_delete_range", t0, t1)
        return f"ripple_delete_range {t0:.3f}-{t1:.3f}", True
    if kind == "close_gaps":
        do("close_gaps", tr.id)
        return f"close_gaps {tr.name}", False
    do("set_clip_props", c.id, fade_in=rng.uniform(0, 2), fade_out=rng.uniform(0, 2))
    return f"fades {c.id}", False


@pytest.mark.parametrize("seed", SEEDS)
def test_random_edits_keep_the_timeline_valid_and_ripples_never_leave_holes(seed):
    rng, proj, seq = build(seed)
    editor = Editor(proj)
    seq_id = seq.id
    log: list[str] = [f"start: {layout(seq)}"]

    def fail(msg: str) -> str:
        return f"seed {seed}: {msg}\n" + "\n".join(log[-14:])

    calls: list[str] = []

    def do(name: str, *args, **kw):
        calls.append(f"{name} {args} {kw}")
        return editor.op(name, name, seq_id, *args, **kw)

    issues = [i for i in validate_sequence(proj, seq) if i.level == "error"]
    assert not issues, fail(f"the starting project is invalid: {[i.message for i in issues]}")
    for step in range(STEPS):
        seq = proj.sequence(seq_id)
        before = {t.id: not gaps(t) for t in seq.tracks()}
        snapshot = proj.content_json()
        try:
            what, is_ripple = one_step(rng, proj, seq, do)
        except OpInvalid as exc:                                   # the net caught it: an operation is wrong
            log.append(f"#{step} NET {exc.issues} <- {calls[-1]}")
            raise AssertionError(fail(f"an operation relied on the safety net: {exc.issues}")) from exc
        except OpError as exc:                                     # "not possible": the editor said no
            log.append(f"#{step} refused: {exc}")
            assert proj.content_json() == snapshot, fail("a refused edit changed the project")
            continue
        except Exception as exc:                                   # noqa: BLE001 - any other exception is a bug
            log.append(f"#{step} CRASH {type(exc).__name__}: {exc}")
            raise AssertionError(fail(f"operation crashed with {type(exc).__name__}: {exc}")) from exc
        seq = proj.sequence(seq_id)
        log.append(f"#{step} {what}\n      -> {layout(seq)}")
        errors = [i for i in validate_sequence(proj, seq) if i.level == "error"]
        assert not errors, fail(f"{what} produced an invalid timeline: {[(i.code, i.where) for i in errors]}")
        for t in seq.tracks():
            starts = [c.start for c in t.clips]
            assert all(b >= a - EPS for a, b in zip(starts, starts[1:])), fail(f"{what}: {t.name} is not sorted")
            assert all(x > -EPS for x in starts), fail(f"{what}: negative start on {t.name}")
        if is_ripple:
            for t in seq.tracks():
                if before.get(t.id) and gaps(t):
                    raise AssertionError(fail(f"{what}: left a hole on {t.name}: {gaps(t)}"))


# ====================================================================================================== the wider fuzz
WIDE_SEEDS = range(200)
FRAME = 1 / 30
NOT_NUMBERS = (float("nan"), float("inf"), float("-inf"))


def one_more(rng, proj, seq, do):
    """One operation the first fuzzer never does; returns ``(description, must_be_refused)``."""
    kind = rng.choice(["move_across", "place_fast", "lock", "unlock", "cue_add", "cue_edit", "cue_split", "cue_merge",
                       "cue_shift", "cue_remove", "nudge", "nudge", "nudge_many", "nan", "mute_disable", "crossfade_zero",
                       "marker", "marker_remove"])
    tr, c = pick_clip(rng, seq)
    if c is None and kind in ("move_across", "nudge", "nudge_many", "nan", "mute_disable", "crossfade_zero"):
        kind = "cue_add"
    if kind == "move_across":
        others = [t for t in (seq.video_tracks if tr.kind == "video" else seq.audio_tracks) if t is not tr]
        dst = rng.choice(others) if others else tr
        t = time_near(rng, c, seq)
        resolve = rng.choice(("overwrite", "insert", "reject"))
        do("move_clip", c.id, t, track_id=dst.id, resolve=resolve)
        return f"move {c.id} -> {dst.name} {t:.3f} {resolve}", False
    if kind == "place_fast":
        track = rng.choice(seq.video_tracks + seq.audio_tracks)
        aid = rng.choice(list(proj.assets))
        total = proj.assets[aid].duration
        s_in = rng.uniform(max(0.0, total - 1.0), total)                 # right at the end of the source
        speed = rng.choice((0.25, 0.5, 2.0, 3.0, 4.0))
        mode = rng.choice(("overwrite", "insert", "append", "reject"))
        start = rng.uniform(0, max(1.0, seq.duration))
        do("place_clip", aid, track.id, start=start, src_in=s_in, src_out=rng.choice((None, total)), mode=mode, speed=speed)
        return f"place_fast {aid} {s_in:.3f}.. x{speed} on {track.name} at {start:.3f} {mode}", False
    if kind == "lock":
        t = rng.choice(list(seq.tracks()))
        do("set_track_props", t.id, locked=True)
        return f"lock {t.name}", False
    if kind == "unlock":
        t = rng.choice(list(seq.tracks()))
        do("set_track_props", t.id, locked=False)
        return f"unlock {t.name}", False
    if kind == "cue_add":
        s = rng.uniform(0, max(1.0, seq.duration))
        do("add_cue", s, s + rng.uniform(0.0, 3.0), "字幕")
        return f"add_cue {s:.3f}", False
    cue = rng.choice(seq.cues) if seq.cues else None
    if kind in ("cue_edit", "cue_split", "cue_remove") and cue is None:
        kind = "cue_add"
        do("add_cue", 1.0, 2.0, "字幕")
        return "add_cue 1.0", False
    if kind == "cue_edit":
        do("update_cue", cue.id, start=rng.uniform(0, 10), end=rng.choice((None, rng.uniform(0, 12))))
        return f"update_cue {cue.id}", False
    if kind == "cue_split":
        do("split_cue", cue.id, rng.uniform(cue.start, cue.end))
        return f"split_cue {cue.id}", False
    if kind == "cue_merge":
        ids = [q.id for q in seq.cues if rng.random() < 0.5]
        do("merge_cues", ids)
        return f"merge_cues {len(ids)}", False
    if kind == "cue_shift":
        ids = [q.id for q in seq.cues if rng.random() < 0.5] if rng.random() < 0.5 else None
        d = rng.uniform(-3, 3)
        do("shift_cues", d, ids)
        return f"shift_cues {d:.3f}", False
    if kind == "cue_remove":
        do("remove_cues", [cue.id])
        return f"remove_cues {cue.id}", False
    if kind == "nudge":
        d = rng.choice((-2, -1, 1, 2, 0, 0)) * FRAME
        do("move_clip", c.id, c.start + d, resolve=rng.choice(("overwrite", "reject", "overwrite")))
        return f"nudge {c.id} by {d:+.3f}", False
    if kind == "nudge_many":
        chosen = [x.id for t2 in seq.tracks() for x in t2.clips if rng.random() < 0.4][:3] or [c.id]
        d = rng.choice((-2, -1, 1, 2, 0)) * FRAME
        do("move_clips", chosen, d, resolve=rng.choice(("overwrite", "reject")))
        return f"nudge_many {len(chosen)} by {d:+.3f}", False
    if kind == "nan":
        bad = rng.choice(NOT_NUMBERS)
        pick = rng.choice([("move_clip", (c.id, bad), {}), ("trim_clip_end", (c.id, bad), {}), ("trim_clip_start", (c.id, bad), {}),
                           ("split_clip", (c.id, bad), {}), ("set_speed", (c.id, bad), {}), ("slip_clip", (c.id, bad), {}),
                           ("set_crossfade", (c.id, bad), {}), ("set_clip_props", (c.id,), {"volume": bad}),
                           ("ripple_delete_range", (bad, 3.0), {}), ("add_cue", (bad, 2.0, "x"), {}),
                           ("place_clip", ("a0", tr.id), {"start": bad})])
        do(pick[0], *pick[1], **pick[2])
        return f"NOT A NUMBER {pick[0]}({bad})", True
    if kind == "mute_disable":
        do("set_clip_props", c.id, muted=rng.random() < 0.5, enabled=rng.random() < 0.7)
        return f"mute/disable {c.id}", False
    if kind == "crossfade_zero":
        do("set_crossfade", c.id, 0.0)
        return f"crossfade_zero {c.id}", False
    if kind == "marker":
        do("add_marker", rng.uniform(0, max(1.0, seq.duration)), "m")
        return "add_marker", False
    if seq.markers:
        do("remove_marker", rng.choice(seq.markers).id)
        return "remove_marker", False
    do("add_marker", 1.0, "m")
    return "add_marker 1.0", False


def congruent(track: Track) -> list:
    cs = track.clips
    return [(a.id, b.id) for i, a in enumerate(cs) for b in cs[i + 1:]
            if abs(a.start - b.start) <= TOUCH and abs(a.end - b.end) <= TOUCH]


@pytest.mark.parametrize("seed", WIDE_SEEDS)
def test_wider_random_edits_keep_everything_valid_and_one_undo_gives_back_exactly_one_edit(seed):
    rng, proj, seq = build(10_000 + seed)
    editor = Editor(proj)
    editor.history.clear()
    seq_id = seq.id
    for q in range(rng.randint(0, 4)):                                     # some captions and markers to start with
        ops.add_cue(proj, seq, rng.uniform(0, 8), rng.uniform(0, 10), f"q{q}")
    ops.add_marker(proj, seq, rng.uniform(0, 8), "m0")
    log: list[str] = [f"start: {layout(seq)}"]
    calls: list[str] = []

    def fail(msg: str) -> str:
        return f"seed {seed}: {msg}\n" + "\n".join(log[-14:])

    def do(name: str, *args, **kw):
        calls.append(f"{name} {args} {kw}")
        return editor.op(name, name, seq_id, *args, **kw)

    for step in range(STEPS):
        seq = proj.sequence(seq_id)
        before_json = proj.content_json()
        depth = len(editor.history._undo)
        refused_needed = False
        try:
            if rng.random() < 0.5:
                what, _ripple = one_step(rng, proj, seq, do)
            else:
                what, refused_needed = one_more(rng, proj, seq, do)
        except OpInvalid as exc:
            log.append(f"#{step} NET {exc.issues} <- {calls[-1]}")
            raise AssertionError(fail(f"an operation relied on the safety net: {exc.issues}")) from exc
        except OpError as exc:
            log.append(f"#{step} refused: {exc}")
            assert proj.content_json() == before_json, fail("a refused edit changed the project")
            assert len(editor.history._undo) == depth, fail("a refused edit left an undo step")
            continue
        except Exception as exc:                                           # noqa: BLE001
            log.append(f"#{step} CRASH {type(exc).__name__}: {exc}")
            raise AssertionError(fail(f"operation crashed with {type(exc).__name__}: {exc}")) from exc
        assert not refused_needed, fail(f"{what} was accepted")
        seq = proj.sequence(seq_id)
        after_json = proj.content_json()
        log.append(f"#{step} {what}\n      -> {layout(seq)}")
        # -- the timeline
        errors = [i for i in validate_sequence(proj, seq) if i.level == "error"]
        assert not errors, fail(f"{what} produced an invalid timeline: {[(i.code, i.where) for i in errors]}")
        for t in seq.tracks():
            assert all(x > -EPS for x in (c.start for c in t.clips)), fail(f"{what}: negative start on {t.name}")
            assert congruent(t) == [], fail(f"{what}: two clips of {t.name} are exactly alike: {congruent(t)}")
        # -- captions and markers
        for q in seq.cues:
            assert math.isfinite(q.start) and math.isfinite(q.end), fail(f"{what}: a caption time is not a number")
            assert q.start >= 0.0 and math.copysign(1.0, q.start) > 0, fail(f"{what}: caption starts at {q.start!r}")
            assert q.end - q.start >= MIN_CUE - 2e-6, fail(f"{what}: a caption of {q.end - q.start:.6f} s")
        starts6 = [round(q.start, 6) for q in seq.cues]                     # the order is that of the values as they are saved
        assert starts6 == sorted(starts6), fail(f"{what}: captions not sorted")
        assert all(math.isfinite(m.time) and m.time >= 0.0 for m in seq.markers), fail(f"{what}: bad marker")
        # -- the history: no change, no step; a change is exactly one step, and undo / redo give back exactly that
        changed = after_json != before_json
        assert len(editor.history._undo) == depth + (1 if changed else 0), fail(f"{what}: wrong number of undo steps")
        if changed:
            editor.undo()
            assert proj.content_json() == before_json, fail(f"{what}: one undo did not give back the old project")
            editor.redo()
            assert proj.content_json() == after_json, fail(f"{what}: redo did not give back the edit")
