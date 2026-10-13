"""Edge cases of the editing core found by the independent code review (reviewer A, findings 4-10).

Each test is a case that really went wrong before it was fixed: a nudge that destroyed a transition, a number that was not
a number reaching the project file, a razor cut that made the picture jump, a clip placed so short that the safety net
fired.  The safety net (``OpInvalid``) must never be what stops a normal edit.
"""
from __future__ import annotations

import itertools
import json

import pytest

from autocut.core import ops
from autocut.core.editor import Editor
from autocut.core.ops import MIN_CLIP, OpError, OpInvalid
from autocut.core.project import validate_sequence
from autocut.core.timecode import MAX_TIMELINE, TOUCH
from autocut.util import clamp
from conftest import fake_asset

ONE_FRAME = 1 / 30


def vt(seq):
    return seq.video_tracks[0]


def starts(track):
    return [(round(c.start, 4), round(c.end, 4)) for c in track.clips]


def chain(proj):
    """V1: a 0-6 | b 5-11 | c 10-16 - one source, each clip crossfaded 1 s into the next."""
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    b = ops.place_clip(proj, seq, "a1", v.id, start=6, src_in=6, src_out=12)
    c = ops.place_clip(proj, seq, "a1", v.id, start=12, src_in=12, src_out=18)
    ops.set_crossfade(proj, seq, b.id, 1.0)
    ops.set_crossfade(proj, seq, c.id, 1.0)
    assert starts(v) == [(0.0, 6.0), (5.0, 11.0), (10.0, 16.0)]
    return seq, v, a, b, c


# ------------------------------------------------------------------------------------------------ A4: moving a clip
@pytest.mark.parametrize("resolve", ["overwrite", "insert", "reject", "free"])
def test_moving_a_clip_to_where_it_already_is_changes_nothing(proj, resolve):
    """It used to cut the clips it crossfades with: a move that goes nowhere destroyed two transitions."""
    seq, v, a, b, c = chain(proj)
    before = proj.content_json()
    ops.move_clip(proj, seq, b.id, 5.0, resolve=resolve)
    assert proj.content_json() == before
    ops.move_clips(proj, seq, [b.id], 0.0, resolve="overwrite" if resolve == "insert" else resolve)
    assert proj.content_json() == before


@pytest.mark.parametrize("resolve", ["overwrite", "reject"])
def test_nudging_a_crossfaded_clip_only_changes_the_length_of_its_crossfades(proj, resolve):
    """One frame to the right: the crossfade with ``a`` gets a frame shorter, the one with ``c`` a frame longer.  Both
    neighbours keep every frame of their pictures (it used to cut ``a`` and eat a second of ``c``)."""
    seq, v, a, b, c = chain(proj)
    a0, c0 = (a.src_in, a.src_out), (c.src_in, c.src_out)
    ops.move_clip(proj, seq, b.id, 5.0 + ONE_FRAME, resolve=resolve)
    assert starts(v) == [(0.0, 6.0), (round(5.0 + ONE_FRAME, 4), round(11.0 + ONE_FRAME, 4)), (10.0, 16.0)]
    assert (a.src_in, a.src_out) == a0 and (c.src_in, c.src_out) == c0 and validate_sequence(proj, seq) == []


def test_nudging_a_group_keeps_the_crossfades_of_the_clips_around_it(proj):
    seq, v, a, b, c = chain(proj)
    ops.move_clips(proj, seq, [b.id], -ONE_FRAME)
    assert starts(v) == [(0.0, 6.0), (round(5.0 - ONE_FRAME, 4), round(11.0 - ONE_FRAME, 4)), (10.0, 16.0)]
    assert validate_sequence(proj, seq) == []


def test_a_clip_dropped_on_one_it_was_not_crossfaded_with_still_overwrites_it(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    b = ops.place_clip(proj, seq, "a1", v.id, start=10, src_in=6, src_out=12)
    ops.move_clip(proj, seq, b.id, 4.0)                                       # lands on a's tail: a is cut, as before
    assert a.end == pytest.approx(4.0) and b.start == pytest.approx(4.0)
    with pytest.raises(OpError):
        ops.move_clip(proj, seq, b.id, 3.0, resolve="reject")                # a clip it did not overlap before: refused


def test_a_move_that_covers_the_neighbour_completely_still_overwrites_it(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    small = ops.place_clip(proj, seq, "a2", v.id, start=5, src_out=2, mode="free")   # 5-7: crossfaded into a's tail
    assert validate_sequence(proj, seq) == [] and small.start == pytest.approx(5.0) and a.end == pytest.approx(6.0)
    ops.move_clip(proj, seq, a.id, 3.0)                                       # a now spans 3-9 and covers all of `small`
    assert [x.id for x in v.clips] == [a.id] and validate_sequence(proj, seq) == []


def test_a_bad_way_of_resolving_is_refused_before_anything_is_touched(proj):
    """Called directly (no Editor to roll back), the clip must still be where it was."""
    seq, v, a, b, c = chain(proj)
    before = proj.content_json()
    with pytest.raises(OpError, match="未知"):
        ops.move_clip(proj, seq, b.id, 3.0, resolve="bogus")
    assert proj.content_json() == before and b in v.clips
    for resolve, text in (("insert", "插入"), ("bogus", "未知")):
        with pytest.raises(OpError, match=text):
            ops.move_clips(proj, seq, [b.id], 1.0, resolve=resolve)
        assert proj.content_json() == before
    with pytest.raises(OpError, match="未知"):
        ops.place_clip(proj, seq, "a2", v.id, start=2.0, src_out=1.0, mode="bogus")
    assert proj.content_json() == before


def test_a_move_is_one_undo_step_and_undo_brings_every_crossfade_back(proj):
    seq, v, a, b, c = chain(proj)
    ed = Editor(proj)
    ed.history.clear()
    before = proj.content_json()
    ed.op("移動", "move_clip", seq.id, b.id, 7.0)
    assert proj.content_json() != before
    ed.undo()
    assert proj.content_json() == before and not ed.history.can_undo


# ------------------------------------------------------------------------------------------------ A5: razor in a fade
def test_a_cut_inside_a_fade_is_refused_and_one_after_it_is_fine(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    c = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=10)
    ops.set_clip_props(proj, seq, c.id, fade_in=2.0, fade_out=2.0)
    before = proj.content_json()
    with pytest.raises(OpError, match="淡入"):
        ops.split_clip(proj, seq, c.id, 1.0)
    with pytest.raises(OpError, match="淡出"):
        ops.split_clip(proj, seq, c.id, 9.0)
    with pytest.raises(OpError, match="淡入"):
        ops.split_at_time(proj, seq, 1.0)
    assert proj.content_json() == before
    left, right = ops.split_clip(proj, seq, c.id, 5.0)                       # between the fades: nothing jumps
    assert left.fade_in == pytest.approx(2.0) and right.fade_out == pytest.approx(2.0)
    ops.set_clip_props(proj, seq, left.id, fade_in=0.0)
    ops.split_clip(proj, seq, left.id, 1.0)                                   # no fade, no problem


def test_a_razor_through_a_fade_on_any_track_cuts_nothing_on_the_others(proj):
    seq = proj.active_sequence()
    v, a1 = vt(seq), seq.audio_tracks[0]
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=10)
    music = ops.place_clip(proj, seq, "m1", a1.id, start=0, src_out=10)
    ops.set_clip_props(proj, seq, music.id, fade_in=3.0)
    before = proj.content_json()
    with pytest.raises(OpError, match="淡入"):
        ops.split_at_time(proj, seq, 2.0)
    assert proj.content_json() == before


# ------------------------------------------------------------------------------------------------ A6: NaN / infinity
BAD = [float("nan"), float("inf"), float("-inf")]


def _cases(proj):
    seq, v, a, b, c = chain(proj)
    cue = ops.add_cue(proj, seq, 1.0, 2.0, "x")
    ops.set_transform_key(proj, seq, b.id, 1.0, 0.5, 0.5)
    t = v.id

    def op(name, *args, **kw):
        return name, args, kw

    return seq, [
        op("place_clip", "a2", t, "bad", 0.0, 1.0), op("place_clip", "a2", t, 3.0, "bad", 1.0),
        op("place_clip", "a2", t, 3.0, 0.0, "bad"),
        op("split_clip", b.id, "bad"), op("split_at_time", "bad"),
        op("trim_clip_start", b.id, "bad"), op("trim_clip_end", b.id, "bad"), op("trim_clip_end", b.id, "bad", ripple=True),
        op("slip_clip", b.id, "bad"), op("move_clip", b.id, "bad"), op("move_clips", [b.id], "bad"),
        op("ripple_delete_range", "bad", 5.0), op("ripple_delete_range", 3.0, "bad"),
        op("set_speed", b.id, "bad"), op("set_crossfade", b.id, "bad"),
        op("set_clip_props", b.id, volume="bad"), op("set_clip_props", b.id, zoom="bad"), op("set_clip_props", b.id, cx="bad"),
        op("set_clip_props", b.id, fade_in="bad"), op("set_clip_props", b.id, brightness="bad"),
        op("set_transform_key", b.id, "bad", 0.5, 0.5), op("set_transform_key", b.id, 1.0, "bad", 0.5),
        op("set_transform_key", b.id, 1.0, 0.5, "bad"),
        op("add_cue", "bad", 3.0, "x"), op("add_cue", 1.0, "bad", "x"),
        op("update_cue", cue.id, start="bad"), op("update_cue", cue.id, end="bad"), op("shift_cues", "bad"),
        op("split_cue", cue.id, "bad"), op("add_marker", "bad", "m"), op("set_sequence_range", "bad", 5.0),
        op("set_track_props", v.id, volume="bad"),
    ]


@pytest.mark.parametrize("bad", BAD, ids=["nan", "inf", "-inf"])
def test_a_number_that_is_not_a_number_is_refused_and_nothing_changes(proj, bad):
    seq, cases = _cases(proj)
    ed = Editor(proj)
    ed.history.clear()
    before = proj.content_json()
    for name, args, kw in cases:
        args = tuple(bad if a == "bad" else a for a in args)
        kw = {k: (bad if v == "bad" else v) for k, v in kw.items()}
        with pytest.raises(OpError, match="數值不正確"):
            ed.op(name, name, seq.id, *args, **kw)
        assert proj.content_json() == before, name
        assert not ed.history.can_undo, name                                  # a refused edit leaves no undo step either
    assert "NaN" not in json.dumps(proj.to_dict()) and "Infinity" not in json.dumps(proj.to_dict())


def test_every_public_edit_is_wrapped_so_a_new_one_cannot_forget_the_check():
    import inspect
    names = [n for n, f in vars(ops).items()
             if not n.startswith("_") and inspect.isfunction(f) and f.__module__ == ops.__name__ and n not in ops._QUERIES]
    assert len(names) > 30 and ops._QUERIES == {"trim_limits", "cue_split_ok"}      # the questions, which edit nothing
    unwrapped = [n for n in names if getattr(getattr(ops, n), "__wrapped__", None) is None]
    assert unwrapped == []


def test_clamp_never_lets_not_a_number_through():
    assert clamp(float("nan"), 0.0, 1.0) == 0.0
    assert clamp(float("inf"), 0.0, 1.0) == 1.0 and clamp(-1e9, 0.0, 1.0) == 0.0 and clamp(0.5, 0.0, 1.0) == 0.5


# ------------------------------------------------------------------------------------------------ A7: place_clip, fast
def test_the_least_a_clip_needs_depends_on_its_speed(proj):
    """19.9-20.0 s of source at speed 4 lasts 0.025 s - under a frame - and tripped the safety net."""
    seq = proj.active_sequence()
    v = vt(seq)
    c = ops.place_clip(proj, seq, "a1", v.id, start=0, src_in=19.9, src_out=20.0, speed=4.0)
    assert c.duration >= MIN_CLIP - 1e-6 and c.src_out <= 20.0 + 1e-9 and validate_sequence(proj, seq) == []


def test_place_clip_never_fires_the_safety_net_for_any_range_and_speed(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    grid = [0.0, 0.05, 0.1, 1.0, 5.0, 19.0, 19.5, 19.8, 19.9, 19.95, 20.0]
    n = 0
    for s_in, s_out, speed, mode in itertools.product(grid, grid + [None], (0.25, 0.5, 1.0, 2.0, 3.0, 4.0),
                                                      ("overwrite", "insert", "append")):
        try:
            c = ops.place_clip(proj, seq, "a1", v.id, start=0.0, src_in=s_in, src_out=s_out, mode=mode, speed=speed)
        except OpInvalid as e:                                                # pragma: no cover - this is the failure
            pytest.fail(f"safety net fired for {(s_in, s_out, speed, mode)}: {e.issues}")
        except OpError:
            continue
        n += 1
        assert c.duration >= ops.MIN_FRAGMENT and c.src_out <= 20.0 + 1e-9
        v.clips.clear()
    assert n > 1000


def test_an_asset_too_short_for_the_speed_is_a_clear_error_not_the_safety_net(proj):
    proj.assets["tiny"] = fake_asset("tiny", 0.1)
    seq = proj.active_sequence()
    with pytest.raises(OpError, match="太短") as e:
        ops.place_clip(proj, seq, "tiny", vt(seq).id, start=0, speed=4.0)
    assert not isinstance(e.value, OpInvalid)
    ops.place_clip(proj, seq, "tiny", vt(seq).id, start=0, speed=1.0)         # at normal speed it is a (short) clip


# ------------------------------------------------------------------------------------------------ A8: congruent clips
def test_a_trim_cannot_make_a_clip_land_exactly_on_its_neighbour(proj):
    """Reviewer's case: ``c`` 8-11 with a 1 s clip inside its tail (10-11).  Trimming ``c``'s head to its crossfade (1 s)
    made both [10-11]: which one was on top then depended on a random id."""
    seq = proj.active_sequence()
    v = vt(seq)
    c = ops.place_clip(proj, seq, "a1", v.id, start=8, src_in=0, src_out=3)
    o = ops.place_clip(proj, seq, "a2", v.id, start=10, src_in=0, src_out=1, mode="free")
    assert validate_sequence(proj, seq) == []
    ops.trim_clip_start(proj, seq, c.id, 10.97)
    assert abs(c.start - o.start) > TOUCH and c.duration >= 1.0 + ONE_FRAME - 1e-6
    assert validate_sequence(proj, seq) == []


def test_a_clip_keeps_a_frame_of_its_own_besides_its_crossfades(proj):
    seq, v, a, b, c = chain(proj)
    lo, hi = ops.trim_limits(proj, v, b, "r", True)
    assert lo == pytest.approx(b.start + 2.0 + ONE_FRAME)
    lo, hi = ops.trim_limits(proj, v, b, "l", True)
    assert hi == pytest.approx(b.end - (2.0 + ONE_FRAME))
    with pytest.raises(OpError, match="交叉淡化"):
        ops.set_speed(proj, seq, b.id, 3.0, ripple=True)                      # 6 s -> 2 s: nothing of its own left
    ops.set_speed(proj, seq, b.id, 2.0, ripple=True)                          # 6 s -> 3 s: still longer than 2 s + a frame
    assert b.duration == pytest.approx(3.0) and validate_sequence(proj, seq) == []


# ------------------------------------------------------------------------------------------------ A9: crossfade of 0
def test_removing_a_crossfade_that_is_not_there_keeps_the_gap_the_person_left(proj):
    """``set_crossfade(c, 0)`` on a clip after a 3 s gap used to close the gap and pull every later clip up."""
    seq = proj.active_sequence()
    v = vt(seq)
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=5)
    b = ops.place_clip(proj, seq, "a1", v.id, start=8, src_in=5, src_out=10)
    d = ops.place_clip(proj, seq, "a1", v.id, start=14, src_in=10, src_out=15)
    before = proj.content_json()
    assert ops.set_crossfade(proj, seq, b.id, 0.0) == 0.0
    assert proj.content_json() == before and (b.start, d.start) == (8.0, 14.0)
    ed = Editor(proj)
    ed.history.clear()
    ed.op("交叉淡化", "set_crossfade", seq.id, b.id, 0.0)
    assert not ed.history.can_undo                                            # nothing changed: no undo step


def test_removing_a_crossfade_that_is_there_restores_a_plain_cut(proj):
    seq, v, a, b, c = chain(proj)
    assert ops.set_crossfade(proj, seq, b.id, 0.0) == 0.0
    assert starts(v) == [(0.0, 6.0), (6.0, 12.0), (11.0, 17.0)]               # c keeps its own crossfade into b
    assert validate_sequence(proj, seq) == []


# ------------------------------------------------------- found by the 10,000-seed random campaign after the review fixes
def stack(proj):
    """V1: m 0-2 | k 1-9 | x 4-9.  ``m`` crossfades 1 s into the head of ``k``; ``x`` is a 5 s clip lying on the end of
    ``k`` (it ends together with it: a transition into nothing)."""
    seq = proj.active_sequence()
    v = vt(seq)
    k = ops.place_clip(proj, seq, "a1", v.id, start=1, src_in=2, src_out=10, mode="free")
    x = ops.place_clip(proj, seq, "a2", v.id, start=4, src_in=0, src_out=5, mode="free")
    m = ops.place_clip(proj, seq, "a2", v.id, start=0, src_in=5, src_out=7, mode="free")
    assert starts(v) == [(0.0, 2.0), (1.0, 9.0), (4.0, 9.0)] and validate_sequence(proj, seq) == []
    return seq, v, m, k, x


@pytest.mark.parametrize("how", ["move_clip", "move_clips"])
def test_a_nudged_clip_does_not_keep_a_partner_the_rest_of_its_landing_would_bury(proj, how):
    """``m`` is moved to 8-10: it still crossfades with the end of ``k``, so ``k`` used to be spared, but ``x`` (which
    lies on ``k``'s end too) has to be cut at 8 - and what is left of it, 4-8, would sit wholly inside ``k``.  The track
    check refused it (the safety net fired).  Now this is a plain overwrite: ``k`` and ``x`` both end where ``m`` begins."""
    seq, v, m, k, x = stack(proj)
    if how == "move_clip":
        ops.move_clip(proj, seq, m.id, 8.0)
    else:
        ops.move_clips(proj, seq, [m.id], 8.0)
    assert starts(v) == [(1.0, 8.0), (4.0, 8.0), (8.0, 10.0)]
    assert validate_sequence(proj, seq) == []


@pytest.mark.parametrize("how", ["move_clip", "move_clips"])
def test_the_net_would_have_fired_without_that_check(proj, monkeypatch, how):
    seq, v, m, k, x = stack(proj)
    monkeypatch.setattr(ops, "_keeps_it_valid", lambda tr, edit: True)        # the old behaviour
    with pytest.raises(OpInvalid):
        if how == "move_clip":
            ops.move_clip(proj, seq, m.id, 8.0)
        else:
            ops.move_clips(proj, seq, [m.id], 8.0)


@pytest.mark.parametrize("how", ["move_clip", "move_clips"])
def test_a_partner_is_still_kept_when_nothing_else_is_in_the_way(proj, how):
    """The check must not turn every nudge back into an overwrite: without ``x`` the move keeps ``k`` whole."""
    seq, v, m, k, x = stack(proj)
    ops.remove_clips(proj, seq, [x.id])
    k_before = (k.start, k.end, k.src_in, k.src_out)
    if how == "move_clip":
        ops.move_clip(proj, seq, m.id, 8.0)
    else:
        ops.move_clips(proj, seq, [m.id], 8.0)
    assert (k.start, k.end, k.src_in, k.src_out) == k_before
    assert starts(v) == [(1.0, 9.0), (8.0, 10.0)] and validate_sequence(proj, seq) == []


def twins(track):
    cs = track.clips
    return [(a.id, b.id) for i, a in enumerate(cs) for b in cs[i + 1:]
            if abs(a.start - b.start) <= TOUCH and abs(a.end - b.end) <= TOUCH]


def tail_clip_pair(proj):
    """V1: a 0-6 and b 3-6, which lies on the end of ``a`` and ends together with it."""
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    b = ops.place_clip(proj, seq, "a2", v.id, start=3, src_out=3, mode="free")
    assert starts(v) == [(0.0, 6.0), (3.0, 6.0)] and validate_sequence(proj, seq) == []
    return seq, v, a, b


def test_a_cut_at_the_start_of_a_clip_lying_on_the_end_is_refused(proj):
    """Cutting ``a`` at 3.0 would leave its right half exactly on top of ``b`` (same start, same end): two clips that
    cannot be told apart.  It used to be accepted."""
    seq, v, a, b = tail_clip_pair(proj)
    before = proj.content_json()
    with pytest.raises(OpError, match="完全一樣"):
        ops.split_clip(proj, seq, a.id, 3.0)
    with pytest.raises(OpError, match="完全一樣"):
        ops.split_at_time(proj, seq, 3.0)
    assert proj.content_json() == before
    ops.split_clip(proj, seq, a.id, 3.0 - ONE_FRAME)                          # one frame earlier it is a normal cut
    assert twins(v) == [] and validate_sequence(proj, seq) == []


def test_a_cut_that_makes_the_left_half_a_twin_is_refused_too(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    b = ops.place_clip(proj, seq, "a2", v.id, start=0, src_out=3, mode="free")          # starts with ``a``, ends at 3
    assert starts(v) == [(0.0, 3.0), (0.0, 6.0)] and validate_sequence(proj, seq) == []
    with pytest.raises(OpError, match="完全一樣"):
        ops.split_clip(proj, seq, a.id, 3.0)


def test_the_razor_refuses_the_whole_cut_when_one_track_would_get_a_twin(proj):
    """Nothing is cut on any track (the other track's clip is not split either)."""
    seq, v, a, b = tail_clip_pair(proj)
    audio = seq.audio_tracks[0]
    ops.place_clip(proj, seq, "a1", audio.id, start=0, src_out=6)
    before = proj.content_json()
    with pytest.raises(OpError, match="完全一樣"):
        ops.split_at_time(proj, seq, 3.0)
    assert proj.content_json() == before and len(audio.clips) == 1


def test_that_check_has_teeth(proj, monkeypatch):
    seq, v, a, b = tail_clip_pair(proj)
    monkeypatch.setattr(ops, "_twin_after_cut", lambda tr, c, at: "")           # the old behaviour
    ops.split_clip(proj, seq, a.id, 3.0)
    assert len(twins(v)) == 1


# ------------------------------------------------------------------------------------------------ sequence background
def test_the_background_colour_is_set_through_the_format_op_and_checked(proj):
    seq = proj.active_sequence()
    assert seq.background == "#000000"
    ops.set_sequence_format(proj, seq, background="#3A6EA5")
    assert seq.background == "#3A6EA5"
    ops.set_sequence_format(proj, seq, width=1280)                            # not given: stays as it is
    assert seq.background == "#3A6EA5" and seq.width == 1280
    before = proj.content_json()
    for bad in ("red", "#12345", "#GGGGGG", "3A6EA5", "#3A6EA5 ", "", 123, ("#000000",)):
        with pytest.raises(OpError, match="背景色"):
            ops.set_sequence_format(proj, seq, width=640, background=bad)
        assert proj.content_json() == before                                  # refused before anything changed (the width too)


# ------------------------------------------------------------------------------------------ damaged project files
# The window used to crash (an AttributeError nobody caught, so no message at all) on a project file whose JSON was valid but
# whose shape was wrong.  Every such file must give one clear ProjectFileError.
BAD_PROJECT_FILES = {
    "a list": "[]",
    "null": "null",
    "a string": '"hello"',
    "a number": "42",
    "sequences is a string": '{"schema":"autocut.project/1","name":"x","sequences":"oops"}',
    "sequences are numbers": '{"schema":"autocut.project/1","name":"x","sequences":[1,2]}',
    "assets is a dict of strings": '{"schema":"autocut.project/1","name":"x","assets":{"a":"x"},"sequences":[]}',
    "assets is a list of numbers": '{"schema":"autocut.project/1","name":"x","assets":[1],"sequences":[]}',
    "a clip that is a string": '{"schema":"autocut.project/1","name":"x","sequences":[{"video_tracks":[{"clips":["z"]}]}]}',
    "tracks are a number": '{"schema":"autocut.project/1","name":"x","sequences":[{"video_tracks":5}]}',
    "revision is text": '{"schema":"autocut.project/1","name":"x","revision":"many","sequences":[]}',
    "empty file": "",
    "cut off": '{"schema":"autocut.project/1","name":"x","sequen',
    "another schema": '{"schema":"something.else/9","name":"x"}',
    "nested without end": "[" * 200000 + "]" * 200000,
}


@pytest.mark.parametrize("case", list(BAD_PROJECT_FILES))
def test_a_damaged_project_file_raises_one_clear_error(case, tmp_path):
    from autocut.core.project import Project, ProjectFileError
    f = tmp_path / "bad.acproj"
    f.write_text(BAD_PROJECT_FILES[case], encoding="utf-8")
    with pytest.raises(ProjectFileError) as ei:
        Project.load(f)
    assert isinstance(ei.value, ValueError) and str(ei.value).strip()           # still a ValueError for older callers


def test_a_project_file_that_is_not_text_raises_the_same_error(tmp_path):
    from autocut.core.project import Project, ProjectFileError
    f = tmp_path / "binary.acproj"
    f.write_bytes(b"\xff\xfe\x00\x01 not utf-8 \x80\x81")
    with pytest.raises(ProjectFileError):
        Project.load(f)
    with pytest.raises(FileNotFoundError):                                      # a missing file is the caller's OSError
        Project.load(tmp_path / "nowhere.acproj")


def test_a_good_project_still_loads_and_unknown_fields_are_ignored(proj, tmp_path):
    from autocut.core.project import Project
    seq = proj.active_sequence()
    ops.place_clip(proj, seq, "a1", vt(seq).id, start=0, src_out=3)
    f = tmp_path / "ok.acproj"
    proj.save(f)
    data = json.loads(f.read_text(encoding="utf-8"))
    data["from_the_future"] = {"x": [1, 2, 3]}
    f.write_text(json.dumps(data), encoding="utf-8")
    assert Project.load(f).content_json() == proj.content_json()


def test_nan_and_infinity_in_a_project_file_are_not_times(tmp_path):
    """JSON from other tools may contain NaN / Infinity (Python's parser accepts them): they must not become clip times."""
    from autocut.core.project import Project
    f = tmp_path / "nan.acproj"
    f.write_text('{"schema":"autocut.project/1","name":"x","sequences":[{"fps":NaN,"video_tracks":[{"clips":'
                 '[{"id":"c1","asset_id":"a","start":NaN,"src_in":0,"src_out":Infinity,"speed":-Infinity,"volume":NaN}]}]}]}',
                 encoding="utf-8")
    p = Project.load(f)
    clip = p.sequences[0].video_tracks[0].clips[0]
    assert (clip.start, clip.src_in, clip.src_out, clip.speed, clip.volume) == (0.0, 0.0, 0.0, 1.0, 1.0)
    assert p.sequences[0].fps == 30.0


# ------------------------------------------------------------------------------------------ a timeline has an end
# Typing 1000000000 into a clip's Start box used to put the clip 31 years out; the timeline's scroll bar takes 32-bit numbers,
# so the slot that lays the timeline out raised OverflowError.  Nothing may sit beyond 24 hours (the length of the
# HH:MM:SS:FF clock) any more: an edit that would put it there is refused like any other impossible edit.
def still_picture():
    from autocut.core.project import Asset
    return Asset(id="i1", path="/nonexistent/i1.png", name="i1.png", kind="image", duration=0.0, width=800, height=600,
                 has_video=False, has_audio=False)


@pytest.fixture
def story(proj):
    """V1: a 0-6 | picture 6-11 (a still has no end of its own), one caption 1-2, one marker at 3."""
    proj.assets["i1"] = still_picture()
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    pic = ops.place_clip(proj, seq, "i1", v.id, start=6, src_out=5)
    cue = ops.add_cue(proj, seq, 1, 2, "x")
    ops.add_marker(proj, seq, 3.0)
    return Editor(proj), seq, v, a, pic, cue


def far_edits(seq, v, a, pic, cue, t):
    """Every way of asking for a time ``t`` out in the distance: (label, op name, arguments).  (The head of a clip cannot go
    past its own end - ``trim_clip_start`` clamps to that - so it is not a way of putting anything out there.)"""
    return [("move", "move_clip", (a.id, t)), ("move, insert", "move_clip", (a.id, t), {"resolve": "insert"}),
            ("move group", "move_clips", ([a.id], t)), ("lengthen the picture", "trim_clip_end", (pic.id, t)),
            ("lengthen, ripple", "trim_clip_end", (pic.id, t), {"ripple": True}),
            ("place", "place_clip", ("i1", v.id, t)), ("place, insert", "place_clip", ("i1", v.id, t), {"mode": "insert"}),
            ("caption", "add_cue", (t, t + 1.0, "far")), ("caption start", "update_cue", (cue.id,), {"start": t}),
            ("caption end", "update_cue", (cue.id,), {"end": t}), ("shift captions", "shift_cues", (t,)),
            ("marker", "add_marker", (t,)), ("range", "set_sequence_range", (0.0, t)),
            ("ripple delete", "ripple_delete_range", (0.0, t))]


@pytest.mark.parametrize("t", [1e9, MAX_TIMELINE + 1.0, 1e300, float("inf")])
def test_nothing_can_be_put_out_beyond_the_end_of_the_timeline(story, t):
    ed, seq, v, a, pic, cue = story
    before, depth = ed.project.content_json(), len(ed.history._undo)
    for label, name, args, *kw in far_edits(seq, v, a, pic, cue, t):
        with pytest.raises(OpError) as ei:
            ed.op(label, name, seq.id, *args, **(kw[0] if kw else {}))
        assert not isinstance(ei.value, OpInvalid), f"{label}: the safety net must not be what stops it"
        if t != float("inf"):                          # (infinity is stopped earlier, as "not a valid number")
            assert "24" in str(ei.value), f"{label}: the message should say what the limit is"
        assert ed.project.content_json() == before and len(ed.history._undo) == depth, f"{label}: nothing may change"


def test_the_limit_itself_is_allowed_and_an_edit_may_end_exactly_there(story):
    ed, seq, v, a, pic, cue = story
    ed.op("move", "move_clip", seq.id, a.id, MAX_TIMELINE - a.duration, resolve="reject")
    moved = seq.find_clip(a.id)[1]
    assert moved.end == pytest.approx(MAX_TIMELINE, abs=1.0 / seq.fps) and moved.end <= MAX_TIMELINE + 1e-6
    assert validate_sequence(ed.project, seq) == []
    ed.undo()
    assert ed.project.active_sequence().find_clip(a.id)[1].start == 0.0


def test_pushing_later_clips_past_the_end_is_refused_as_a_whole(story):
    """Nothing is asked for out there, yet the ripple of a slow-motion change or a lengthened picture would push the clip
    behind it over the end: the whole edit goes, not just the part that still fits."""
    ed, seq, v, a, pic, cue = story
    ed.op("move", "move_clip", seq.id, pic.id, MAX_TIMELINE - pic.duration - 4.0, resolve="reject")
    before, depth = ed.project.content_json(), len(ed.history._undo)
    for label, name, args, kw in (("slow motion", "set_speed", (a.id, 0.25), {"ripple": True}),
                                  ("longer picture before it", "trim_clip_end", (a.id, a.end + 5.0), {"ripple": True}),
                                  ("insert before it", "place_clip", ("a1", v.id, 0.0), {"mode": "insert", "src_out": 10.0})):
        with pytest.raises(OpError) as ei:
            ed.op(label, name, seq.id, *args, **kw)
        assert not isinstance(ei.value, OpInvalid) and "24" in str(ei.value), label
        assert ed.project.content_json() == before and len(ed.history._undo) == depth, label


def test_a_clip_beyond_the_end_is_reported_by_the_validator(story):
    """A project file can hold any number; the validator (used by the audit and the exporter) says what is wrong."""
    ed, seq, v, a, pic, cue = story
    a.start = 1e9                                       # what a damaged or hand-edited project file can hold
    codes = {i.code for i in validate_sequence(ed.project, seq)}
    assert "too_long" in codes
