"""Core model / operations / history / plan tests (pure Python, no ffmpeg)."""
from __future__ import annotations

import json

import pytest

from autocut.core import ops
from autocut.core.editor import Editor
from autocut.core.geometry import crop_rect_px, interp_keys, split_keys
from autocut.core.history import History
from autocut.core.ops import OpError, OpInvalid
from autocut.core.plan import (AuditReceipt, EditPlan, Finding, PlanError, check_receipt, record_run,
                               set_human_review, set_technical_qa)
from autocut.core.project import CaptionCue, Clip, Project, Sequence, ordered_clips, validate_sequence
from autocut.core.subtitles import parse_srt, to_srt
from autocut.core.timecode import fmt_clock, fmt_timecode, frame_of, parse_timecode, snap_time
from autocut.util import canonical_json, sha256_hex


def vt(seq):  # first video track
    return seq.video_tracks[0]


def starts(track):
    return [(round(c.start, 3), round(c.end, 3)) for c in track.clips]


# ------------------------------------------------------------------ timecode / geometry
def test_timecode_roundtrip():
    assert frame_of(1.0, 30) == 30
    assert snap_time(1.016, 30) == pytest.approx(1.0333333, abs=1e-6) or snap_time(1.016, 30) == pytest.approx(1.0, abs=1e-6)
    assert fmt_timecode(61.5, 30) == "00:01:01:15"
    assert fmt_clock(3725.25) == "1:02:05.250"
    assert parse_timecode("1:30") == 90
    assert parse_timecode("00:01:30:15", 30) == pytest.approx(90.5)
    assert parse_timecode("45f", 30) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        parse_timecode("")


def test_crop_rect_916_from_169():
    x, y, w, h = crop_rect_px(1920, 1080, 1080, 1920, 0.5, 0.5, 1.0)
    assert (w, h) == (606, 1080)
    assert x == 656 and y == 0
    # clamped at edges
    x, _, w, _ = crop_rect_px(1920, 1080, 1080, 1920, 0.0, 0.5, 1.0)
    assert x == 0
    x, _, w, _ = crop_rect_px(1920, 1080, 1080, 1920, 1.0, 0.5, 1.0)
    assert x + w <= 1920
    # zoom shrinks the window
    _, _, w2, h2 = crop_rect_px(1920, 1080, 1080, 1920, 0.5, 0.5, 2.0)
    assert w2 == pytest.approx(w / 2, abs=2) and h2 == pytest.approx(540, abs=2)


def test_interp_and_split_keys():
    keys = [[0, 0.2, 0.5], [2, 0.8, 0.5]]
    assert interp_keys(keys, 1.0) == pytest.approx((0.5, 0.5))
    assert interp_keys(keys, -1) == (0.2, 0.5) and interp_keys(keys, 9) == (0.8, 0.5)
    left, right = split_keys(keys, 1.0)
    assert left[-1][0] == 1.0 and right[0][0] == 0.0
    assert left[-1][1] == pytest.approx(0.5) and right[0][1] == pytest.approx(0.5)
    assert right[-1][0] == pytest.approx(1.0)


def crossfaded_chain(proj):
    """V1: a 0..6 | b 5..11 | c 10..16 - three clips of one source, each crossfaded 1 s into the next."""
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    b = ops.place_clip(proj, seq, "a1", v.id, start=6, src_in=6, src_out=12)
    c = ops.place_clip(proj, seq, "a1", v.id, start=12, src_in=12, src_out=18)
    ops.set_crossfade(proj, seq, b.id, 1.0)
    ops.set_crossfade(proj, seq, c.id, 1.0)
    assert starts(v) == [(0.0, 6.0), (5.0, 11.0), (10.0, 16.0)]
    return seq, v, a, b, c


# ------------------------------------------------------ editing next to crossfades (found by the randomised test)
def test_ripple_trim_end_moves_the_clip_that_fades_in_over_the_tail(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    ops.trim_clip_end(proj, seq, b.id, 9.0, ripple=True)
    assert starts(v) == [(0.0, 6.0), (5.0, 9.0), (8.0, 14.0)]            # c moved with b's end, the 1 s crossfade is kept
    assert validate_sequence(proj, seq) == []
    ops.trim_clip_end(proj, seq, b.id, 12.0, ripple=True)                # and back out again
    assert starts(v) == [(0.0, 6.0), (5.0, 12.0), (11.0, 17.0)]


def test_ripple_trim_start_moves_the_clip_that_fades_in_over_the_tail(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    ops.trim_clip_start(proj, seq, b.id, 8.0, ripple=True)               # b keeps its place, loses 3 s from the head
    assert starts(v) == [(0.0, 6.0), (5.0, 8.0), (7.0, 13.0)]
    assert b.src_in == pytest.approx(9.0) and validate_sequence(proj, seq) == []


def test_a_trim_cannot_make_a_clip_shorter_than_its_crossfades(proj):
    seq, v, a, b, c = crossfaded_chain(proj)                             # b: 1 s fade-in from a, 1 s fade-out into c
    ops.trim_clip_end(proj, seq, b.id, 5.2, ripple=True)
    one_frame = 1 / 30
    assert b.duration == pytest.approx(2.0 + one_frame)                  # stops at the two crossfades and a frame of its own
    assert starts(v) == [(0.0, 6.0), (5.0, round(7.0 + one_frame, 3)), (round(6.0 + one_frame, 3), round(12.0 + one_frame, 3))]
    assert validate_sequence(proj, seq) == []


def test_a_trim_without_ripple_cannot_go_shorter_than_the_fade_in(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    ops.trim_clip_end(proj, seq, b.id, 5.5)                              # as long as the 1 s fade-in from a and a frame, no less
    assert b.end == pytest.approx(6.0 + 1 / 30) and validate_sequence(proj, seq) == []
    ops.trim_clip_end(proj, seq, b.id, 100.0)                            # c is no longer under its tail: a hard stop at c
    assert b.end == pytest.approx(c.start)


def test_a_trim_without_ripple_runs_into_a_crossfade_partner_but_not_through_it(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    ops.trim_clip_end(proj, seq, b.id, 100.0)                            # longest: just short of the end of c
    assert b.end == pytest.approx(c.end - ops.MIN_CLIP) and validate_sequence(proj, seq) == []
    ops.trim_clip_start(proj, seq, b.id, 0.0)                            # earliest: just after a starts
    assert b.start == pytest.approx(a.start + ops.MIN_CLIP) and validate_sequence(proj, seq) == []


@pytest.mark.parametrize("ripple", [False, True])
@pytest.mark.parametrize("edge", ["l", "r"])
@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("far", [-1000.0, 1000.0])
def test_the_edit_ends_exactly_at_the_limits_the_drag_preview_shows(proj, ripple, edge, index, far):
    """The timeline's ghost asks ``trim_limits``; a trim past a limit must end exactly there."""
    seq, v, *chain = crossfaded_chain(proj)
    clip = chain[index]
    lo, hi = ops.trim_limits(proj, v, clip, edge, ripple)
    assert lo <= hi
    want = lo if far < 0 else hi
    start0, duration0 = clip.start, clip.duration
    if edge == "l":
        ops.trim_clip_start(proj, seq, clip.id, far, ripple=ripple)
        if ripple:                                   # the clip stays where it is; its head moved to `want`
            assert clip.start == pytest.approx(start0) and clip.duration == pytest.approx(duration0 - (want - start0))
        else:
            assert clip.start == pytest.approx(want)
    else:
        ops.trim_clip_end(proj, seq, clip.id, far, ripple=ripple)
        assert clip.end == pytest.approx(want)
    assert validate_sequence(proj, seq) == []


def test_split_and_razor_inside_a_crossfade_are_refused_and_change_nothing(proj):
    """A cut through the overlap would leave two short crossfades in a row: the picture jumps back to the first clip."""
    seq, v, a, b, c = crossfaded_chain(proj)
    a1 = seq.audio_tracks[0]
    music = ops.place_clip(proj, seq, "m1", a1.id, start=0, src_out=16)  # a clip under the whole transition, on its own track
    before = proj.content_json()
    with pytest.raises(OpError, match="交叉淡化"):
        ops.split_clip(proj, seq, b.id, 5.5)                             # inside the 5..6 fade-in
    with pytest.raises(OpError, match="交叉淡化"):
        ops.split_clip(proj, seq, b.id, 10.5)                            # inside the 10..11 fade-out
    with pytest.raises(OpError, match="交叉淡化"):
        ops.split_at_time(proj, seq, 5.5, clip_ids=[b.id])               # only one of the two clips of the transition
    with pytest.raises(OpError, match="交叉淡化"):
        ops.split_at_time(proj, seq, 5.5, clip_ids=[a.id, b.id])         # both of them
    with pytest.raises(OpError, match="交叉淡化"):
        ops.split_at_time(proj, seq, 5.5)                                # nothing selected: the music is not cut either
    assert proj.content_json() == before
    ops.split_at_time(proj, seq, 5.5, clip_ids=[music.id])               # the music alone is not in a transition
    assert starts(a1) == [(0.0, 5.5), (5.5, 16.0)]
    ops.split_clip(proj, seq, b.id, 8.0)                                 # outside the transitions: fine
    assert starts(v) == [(0.0, 6.0), (5.0, 8.0), (8.0, 11.0), (10.0, 16.0)]
    ops.split_at_time(proj, seq, 8.0)                                    # at the same spot again: nothing to cut on V1
    assert validate_sequence(proj, seq) == []


def test_speed_change_ripples_the_crossfaded_successor_and_respects_the_crossfades(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    ops.set_speed(proj, seq, b.id, 2.0)                                  # b 6 s -> 3 s
    assert starts(v) == [(0.0, 6.0), (5.0, 8.0), (7.0, 13.0)] and validate_sequence(proj, seq) == []
    with pytest.raises(OpError, match="短於"):
        ops.set_speed(proj, seq, b.id, 4.0)                              # 1.5 s is less than the two 1 s crossfades
    assert b.speed == 2.0


def test_speed_without_ripple_cannot_run_into_the_next_clip(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    with pytest.raises(OpError, match="擋住"):
        ops.set_speed(proj, seq, b.id, 0.5, ripple=False)                # 6 s -> 12 s would end after c
    ops.set_speed(proj, seq, b.id, 1.5, ripple=False)                    # faster is fine: the clips after it stay put
    assert b.end == pytest.approx(9.0) and c.start == pytest.approx(10.0) and validate_sequence(proj, seq) == []


def test_crossfade_length_respects_the_clips_own_fade_out(proj):
    """After a cut the second half can be shorter than the crossfade it already has into the next clip."""
    seq, v, a, b, c = crossfaded_chain(proj)
    ops.split_clip(proj, seq, b.id, 9.8)                                 # b-right: 9.8..11, fades into c over 1 s
    right = next(x for x in v.clips if x.start == pytest.approx(9.8))
    got = ops.set_crossfade(proj, seq, right.id, 1.0)
    assert got == pytest.approx(0.2 - 1 / 30)                            # what is left of b-right after its fade-out, less the frame it keeps
    assert validate_sequence(proj, seq) == []


def test_crossfade_length_is_rounded_down_to_whole_frames(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=1.0)
    b = ops.place_clip(proj, seq, "a1", v.id, start=1.0, src_in=1, src_out=2.0)
    got = ops.set_crossfade(proj, seq, b.id, 0.49)                       # the limit is half of 1 s = 0.5; 0.49 is not a frame
    assert got == pytest.approx(14 / 30) and got <= 0.5


def test_overwrite_drops_a_left_over_under_one_frame_and_keeps_longer_ones(proj):
    seq = proj.active_sequence()
    v, v2 = seq.video_tracks[0], seq.video_tracks[1]
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=14.3)                    # 0..14.3
    m = ops.place_clip(proj, seq, "a2", v2.id, start=0, src_out=10)
    ops.set_speed(proj, seq, m.id, 0.7)                                             # 14.2857 s: not on the frame grid
    ops.move_clip(proj, seq, m.id, 0.0, track_id=v.id, resolve="overwrite")
    assert [x.id for x in v.clips] == [m.id]                                        # 14.2857..14.3 is under a frame: gone
    assert validate_sequence(proj, seq) == []
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=17)                      # 0..17 over it: m is now a piece of it
    assert [round(x.end, 3) for x in v.clips] == [17.0]
    n = ops.place_clip(proj, seq, "a2", v.id, start=2, src_out=5)                   # 2..7 inside 0..17: both sides stay
    assert starts(v) == [(0.0, 2.0), (2.0, 7.0), (7.0, 17.0)] and n in v.clips


def test_overwrite_inside_a_crossfade_does_not_leave_a_stack_of_two_clips(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    n = ops.place_clip(proj, seq, "a2", v.id, start=3, src_out=7.5)                 # 3..10.5 over a, b and the start of c
    assert starts(v) == [(0.0, 3.0), (3.0, 10.5), (10.5, 16.0)]
    assert [x.id for x in v.clips][1] == n.id and validate_sequence(proj, seq) == []


def test_ripple_delete_range_inside_a_crossfade_stays_valid_and_has_no_hole(proj):
    seq, v, a, b, c = crossfaded_chain(proj)
    ops.ripple_delete_range(proj, seq, 5.4, 5.8)
    assert validate_sequence(proj, seq) == []
    assert starts(v) == [(0.0, 5.4), (5.0, 5.4), (5.4, 10.6), (9.6, 15.6)]


def test_clips_that_start_together_are_ordered_by_their_end_even_with_rounding_noise(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    long_clip = Clip(id="c_long", asset_id="a1", start=5.0, src_in=0.0, src_out=4.0)
    short_clip = Clip(id="c_short", asset_id="a1", start=5.0000003, src_in=0.0, src_out=0.5)   # a saved project's rounding
    v.clips = [long_clip, short_clip]
    assert [x.id for x in ordered_clips(v.clips)] == ["c_short", "c_long"]
    assert [i for i in validate_sequence(proj, seq) if i.level == "error"] == []
    keep = v.clips
    v.sort()
    assert v.clips is keep and [x.id for x in v.clips] == ["c_short", "c_long"]       # sorted in place, same list


def test_a_clip_under_one_frame_is_invalid_but_one_frame_is_fine(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=2)
    tiny = ops.place_clip(proj, seq, "a1", v.id, start=2, src_out=1, mode="free")
    tiny.src_out = tiny.src_in + 1 / 30
    assert validate_sequence(proj, seq) == []
    tiny.src_out = tiny.src_in + 0.02
    assert [i.code for i in validate_sequence(proj, seq)] == ["too_short"]


def test_the_guard_refuses_an_edit_that_would_break_the_timeline_but_not_old_damage(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    ops.place_clip(proj, seq, "a1", v.id, start=3, src_out=6, mode="free")
    ops.place_clip(proj, seq, "a1", v.id, start=4, src_out=6, mode="free")          # three layers: already broken
    assert "overlap_depth" in {i.code for i in validate_sequence(proj, seq)}
    ops.place_clip(proj, seq, "a1", seq.video_tracks[1].id, start=0, src_out=2)     # an edit elsewhere is not blocked
    broken = ops._guarded(lambda p, s: setattr(s.video_tracks[1].clips[0], "start", -3.0))
    with pytest.raises(OpInvalid) as err:
        broken(proj, seq)
    assert err.value.issues and err.value.issues[0][1] == "negative_start" and isinstance(err.value, OpError)


# ----------------------------------------------------------------------- place / split
def test_place_append_and_clamp(proj):
    seq = proj.active_sequence()
    v1 = vt(seq)
    c1 = ops.place_clip(proj, seq, "a1", v1.id)                       # 0..20
    c2 = ops.place_clip(proj, seq, "a2", v1.id, src_in=2, src_out=6)  # append after c1
    assert c1.start == 0 and c1.end == pytest.approx(20)
    assert c2.start == pytest.approx(20) and c2.duration == pytest.approx(4)
    with pytest.raises(OpError):
        ops.place_clip(proj, seq, "m1", v1.id)                        # audio-only asset on a video track
    a1 = seq.audio_tracks[0]
    m = ops.place_clip(proj, seq, "m1", a1.id, start=0)
    assert m.duration == pytest.approx(60)
    assert validate_sequence(proj, seq) == []


def test_split_keeps_continuity(proj):
    seq = proj.active_sequence()
    c = ops.place_clip(proj, seq, "a1", vt(seq).id, start=0, src_in=0, src_out=10)
    c.transform.keys = [[0, 0.2, 0.5], [10, 0.8, 0.5]]
    c.fade_in, c.fade_out = 1.0, 1.0
    left, right = ops.split_clip(proj, seq, c.id, 4.0)
    assert left.end == pytest.approx(4.0) and right.start == pytest.approx(4.0)
    assert right.src_in == pytest.approx(4.0) and right.src_out == pytest.approx(10.0)
    assert left.fade_in == 1.0 and left.fade_out == 0.0 and right.fade_in == 0.0 and right.fade_out == 1.0
    # camera path continues without a jump
    lx, _ = left.transform.center_at(left.duration)
    rx, _ = right.transform.center_at(0.0)
    assert lx == pytest.approx(rx, abs=1e-6) == pytest.approx(0.2 + 0.6 * 0.4, abs=1e-6)
    with pytest.raises(OpError):
        ops.split_clip(proj, seq, left.id, 0.01)


def test_split_at_time_razor(proj):
    seq = proj.active_sequence()
    ops.place_clip(proj, seq, "a1", seq.video_tracks[0].id, start=0, src_out=10)
    ops.place_clip(proj, seq, "a2", seq.video_tracks[1].id, start=2, src_out=8)
    made = ops.split_at_time(proj, seq, 5.0)
    assert len(made) == 2
    with pytest.raises(OpError):
        ops.split_at_time(proj, seq, 50.0)


# ------------------------------------------------------------------------------- trims
def test_trim_head_tail_and_limits(proj):
    seq = proj.active_sequence()
    c = ops.place_clip(proj, seq, "a1", vt(seq).id, start=5, src_in=4, src_out=10)    # 5..11
    ops.trim_clip_start(proj, seq, c.id, 6.0)
    assert (c.start, c.src_in) == (pytest.approx(6.0), pytest.approx(5.0))
    ops.trim_clip_start(proj, seq, c.id, 0.0)       # can extend back only until the previous clip/zero and source start
    assert c.src_in >= 0
    ops.trim_clip_end(proj, seq, c.id, 100.0)       # limited by the 20 s source
    assert c.src_out <= 20.0 + 1e-6
    ops.trim_clip_end(proj, seq, c.id, c.start + 0.0)
    assert c.duration == pytest.approx(ops.MIN_CLIP, abs=0.04)


def test_trim_does_not_overlap_neighbours_unless_ripple(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=5)
    b = ops.place_clip(proj, seq, "a1", v.id, start=5, src_in=5, src_out=10)
    ops.trim_clip_end(proj, seq, a.id, 8.0)
    assert a.end == pytest.approx(5.0)            # blocked by b
    ops.trim_clip_end(proj, seq, a.id, 3.0, ripple=True)
    assert a.end == pytest.approx(3.0) and b.start == pytest.approx(3.0)
    ops.trim_clip_start(proj, seq, b.id, 4.0, ripple=True)    # b head trimmed by 1 s, b stays at 3.0
    assert b.start == pytest.approx(3.0) and b.src_in == pytest.approx(6.0) and b.duration == pytest.approx(4.0)


def test_slip(proj):
    seq = proj.active_sequence()
    c = ops.place_clip(proj, seq, "a1", vt(seq).id, start=0, src_in=2, src_out=6)
    ops.slip_clip(proj, seq, c.id, 3.0)
    assert (c.src_in, c.src_out) == (pytest.approx(5.0), pytest.approx(9.0)) and c.start == 0
    ops.slip_clip(proj, seq, c.id, 100)
    assert c.src_out == pytest.approx(20.0)


# ------------------------------------------------------------------------------- move
def test_move_overwrite_insert_reject(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=10)
    b = ops.place_clip(proj, seq, "a2", v.id, start=12, src_out=4)         # 12..16
    ops.move_clip(proj, seq, b.id, 8.0)                                     # lands on a's tail -> a is trimmed
    assert a.end == pytest.approx(8.0) and b.start == pytest.approx(8.0)
    c = ops.place_clip(proj, seq, "a2", v.id, start=20, src_out=2)
    with pytest.raises(OpError):
        ops.move_clip(proj, seq, c.id, 9.0, resolve="reject")
    ops.move_clip(proj, seq, c.id, 9.0, resolve="insert")                   # splits b, pushes right
    assert validate_sequence(proj, seq) == []
    assert c.start == pytest.approx(9.0)
    total_src = sum(x.src_out - x.src_in for x in v.clips)
    assert total_src == pytest.approx(8.0 + 4.0 + 2.0)                      # nothing lost by insert


def test_overwrite_splits_middle(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=10)
    ops.place_clip(proj, seq, "a2", v.id, start=4, src_out=2)               # overwrite 4..6 inside a
    assert starts(v) == [(0.0, 4.0), (4.0, 6.0), (6.0, 10.0)]
    assert v.clips[2].src_in == pytest.approx(6.0)


def test_move_clips_group(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=1, src_out=3)
    b = ops.place_clip(proj, seq, "a1", v.id, start=5, src_out=3)
    ops.move_clips(proj, seq, [a.id, b.id], -5.0)       # clamped so nothing goes below zero
    assert a.start == pytest.approx(0.0) and b.start == pytest.approx(4.0)


# -------------------------------------------------------------------------- remove / ripple
def test_remove_ripple_and_close_gaps(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=4)
    b = ops.place_clip(proj, seq, "a1", v.id, start=4, src_in=4, src_out=8)
    c = ops.place_clip(proj, seq, "a1", v.id, start=8, src_in=8, src_out=12)
    ops.remove_clips(proj, seq, [b.id], ripple=True)
    assert c.start == pytest.approx(4.0)
    d = ops.place_clip(proj, seq, "a2", v.id, start=20, src_out=2)
    closed = ops.close_gaps(proj, seq, v.id)
    assert closed == pytest.approx(12.0) and d.start == pytest.approx(8.0)
    with pytest.raises(OpError):
        ops.remove_clips(proj, seq, ["nope"])


def test_ripple_remove_next_to_a_crossfade_leaves_no_hole(proj):
    """Found by the end-to-end run: the clip that crossfaded into the removed one must move up with the rest, or the
    picture goes black for the length of the removed clip's exclusive part."""
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=4)
    x = ops.place_clip(proj, seq, "a1", v.id, start=4, src_in=4, src_out=9)
    c = ops.place_clip(proj, seq, "a1", v.id, start=9, src_in=9, src_out=13)
    d = ops.place_clip(proj, seq, "a1", v.id, start=13, src_in=13, src_out=17)
    ops.set_crossfade(proj, seq, c.id, 0.5)                   # c 8.5..12.5 overlaps the tail of x (4..9)
    ops.set_crossfade(proj, seq, d.id, 0.5)                   # d 12.0..16.0 overlaps the tail of c
    assert starts(v) == [(0.0, 4.0), (4.0, 9.0), (8.5, 12.5), (12.0, 16.0)]
    ops.remove_clips(proj, seq, [x.id], ripple=True)
    assert starts(v) == [(0.0, 4.0), (4.0, 8.0), (7.5, 11.5)]              # a | c  hard cut, c..d crossfade kept
    assert validate_sequence(proj, seq) == [] and seq.duration == pytest.approx(11.5)
    assert (a.start, c.start, d.start) == (0.0, pytest.approx(4.0), pytest.approx(7.5))


def test_ripple_remove_head_overlap_and_both_neighbours_crossfaded(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=4)
    x = ops.place_clip(proj, seq, "a1", v.id, start=4, src_in=4, src_out=9)
    c = ops.place_clip(proj, seq, "a1", v.id, start=9, src_in=9, src_out=13)
    ops.set_crossfade(proj, seq, x.id, 0.5)                   # x 3.5..8.5 overlaps the tail of a
    ops.set_crossfade(proj, seq, c.id, 0.5)                   # c 8.0..12.0 overlaps the tail of x
    assert starts(v) == [(0.0, 4.0), (3.5, 8.5), (8.0, 12.0)]
    ops.remove_clips(proj, seq, [x.id], ripple=True)
    assert starts(v) == [(0.0, 4.0), (4.0, 8.0)]              # c starts where a ends: no hole, no 1 s double overlap
    assert validate_sequence(proj, seq) == []


def test_ripple_remove_keeps_gaps_the_person_left_on_purpose(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=4)
    x = ops.place_clip(proj, seq, "a1", v.id, start=6, src_in=6, src_out=9)
    ops.place_clip(proj, seq, "a1", v.id, start=9, src_in=9, src_out=12)
    ops.place_clip(proj, seq, "a1", v.id, start=20, src_in=0, src_out=2)
    ops.remove_clips(proj, seq, [x.id], ripple=True)
    assert starts(v) == [(0.0, 4.0), (6.0, 9.0), (17.0, 19.0)]              # the 4..6 and 9..17 gaps stay as they were


def test_ripple_remove_two_clips_in_a_row_from_the_start(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    x1 = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=3)
    x2 = ops.place_clip(proj, seq, "a1", v.id, start=3, src_in=3, src_out=5)
    ops.place_clip(proj, seq, "a1", v.id, start=5, src_in=5, src_out=9)
    ops.remove_clips(proj, seq, [x1.id, x2.id], ripple=True)
    assert starts(v) == [(0.0, 4.0)]                          # both removed back to back: the rest starts at 0


def test_ripple_remove_of_a_clip_under_a_longer_one_frees_no_time(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    long_clip = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=10)
    under = ops.place_clip(proj, seq, "a1", v.id, start=3, src_out=2, mode="free")
    after = ops.place_clip(proj, seq, "a1", v.id, start=10, src_in=10, src_out=14)
    assert (long_clip.start, under.start, after.start) == (0.0, 3.0, 10.0)
    ops.remove_clips(proj, seq, [under.id], ripple=True)
    assert (long_clip.start, after.start) == (0.0, pytest.approx(10.0))


def test_ripple_delete_range_across_tracks_and_captions(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=10)
    ops.place_clip(proj, seq, "m1", seq.audio_tracks[0].id, start=0, src_out=10)
    ops.add_cue(proj, seq, 1, 2, "before")
    ops.add_cue(proj, seq, 4, 6, "inside")
    ops.add_cue(proj, seq, 3, 5, "straddle")
    ops.add_cue(proj, seq, 7, 8, "after")
    removed = ops.ripple_delete_range(proj, seq, 4.0, 6.0)
    assert removed == pytest.approx(2.0)
    assert seq.duration == pytest.approx(8.0)
    texts = {q.text: (round(q.start, 2), round(q.end, 2)) for q in seq.cues}
    assert "inside" not in texts
    assert texts["before"] == (1.0, 2.0)
    assert texts["straddle"] == (3.0, 4.0)
    assert texts["after"] == (5.0, 6.0)
    # no source material lost outside the cut
    assert sum(c.src_out - c.src_in for c in v.clips) == pytest.approx(8.0)


# ----------------------------------------------------------------------- speed / crossfade
def test_speed_changes_duration_and_ripples(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=8)
    b = ops.place_clip(proj, seq, "a1", v.id, start=8, src_in=8, src_out=12)
    ops.set_speed(proj, seq, a.id, 2.0)
    assert a.duration == pytest.approx(4.0) and b.start == pytest.approx(4.0)
    ops.set_speed(proj, seq, a.id, 100)      # clamped
    assert a.speed == ops.MAX_SPEED


def test_crossfade_overlap_and_restore(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    a = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6)
    b = ops.place_clip(proj, seq, "a1", v.id, start=6, src_in=6, src_out=12)
    c = ops.place_clip(proj, seq, "a1", v.id, start=12, src_in=12, src_out=18)
    got = ops.set_crossfade(proj, seq, b.id, 1.0)
    assert got == pytest.approx(1.0) and b.start == pytest.approx(5.0) and c.start == pytest.approx(11.0)
    assert validate_sequence(proj, seq) == []
    ops.set_crossfade(proj, seq, b.id, 0.0)
    assert b.start == pytest.approx(6.0) and c.start == pytest.approx(12.0)
    assert ops.set_crossfade(proj, seq, b.id, 99.0) == pytest.approx(3.0)     # half of the 6 s clips
    with pytest.raises(OpError):
        ops.set_crossfade(proj, seq, a.id, 1.0)


def test_validate_flags_triple_overlap(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    for i in range(3):
        c = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=6, mode="free")
        c.start = i * 1.0
    codes = {i.code for i in validate_sequence(proj, seq)}
    assert "overlap_depth" in codes


# ----------------------------------------------------------------------- props / keys
def test_set_clip_props_and_keys(proj):
    seq = proj.active_sequence()
    c = ops.place_clip(proj, seq, "a1", vt(seq).id, start=0, src_out=10)
    ops.set_clip_props(proj, seq, c.id, volume=0.5, fit="blur", zoom=1.5, cx=0.3, brightness=0.1, fade_in=2)
    assert (c.volume, c.transform.fit, c.transform.zoom, c.transform.cx, c.look.brightness, c.fade_in) == \
        (0.5, "blur", 1.5, 0.3, 0.1, 2.0)
    with pytest.raises(OpError):
        ops.set_clip_props(proj, seq, c.id, fit="stretch")
    with pytest.raises(OpError):
        ops.set_clip_props(proj, seq, c.id, nonsense=1)
    ops.set_transform_key(proj, seq, c.id, 2.0, 0.4, 0.5)
    ops.set_transform_key(proj, seq, c.id, 6.0, 0.7, 0.5)
    ops.set_transform_key(proj, seq, c.id, 2.0, 0.45, 0.5)      # replaces
    assert len(c.transform.keys) == 2 and c.transform.keys[0][1] == pytest.approx(0.45)
    ops.remove_transform_key(proj, seq, c.id, 0)
    assert len(c.transform.keys) == 1
    ops.clear_transform_keys(proj, seq, c.id)
    assert c.transform.keys == []


def test_locked_track_rejects_edits(proj):
    seq = proj.active_sequence()
    v = vt(seq)
    c = ops.place_clip(proj, seq, "a1", v.id, start=0, src_out=5)
    ops.set_track_props(proj, seq, v.id, locked=True)
    with pytest.raises(OpError):
        ops.split_clip(proj, seq, c.id, 2.0)
    with pytest.raises(OpError):
        ops.place_clip(proj, seq, "a2", v.id, start=10)


# ------------------------------------------------------------------------------ captions
def test_cue_ops(proj):
    seq = proj.active_sequence()
    q = ops.add_cue(proj, seq, 1.0, 3.0, "今天來聊聊剪輯")
    a, b = ops.split_cue(proj, seq, q.id, 2.0)
    assert a.text + b.text == "今天來聊聊剪輯" and a.end == b.start == pytest.approx(2.0)
    m = ops.merge_cues(proj, seq, [a.id, b.id])
    assert m.text == "今天來聊聊剪輯" and m.start == pytest.approx(1.0) and m.end == pytest.approx(3.0)
    ops.update_cue(proj, seq, m.id, start=5.0)
    assert m.start == pytest.approx(5.0) and m.end > m.start
    ops.shift_cues(proj, seq, -10)           # never below zero
    assert m.start == 0.0
    n = ops.replace_cues(proj, seq, [(0, 1, "a"), (1, 2, "b")])
    assert len(n) == 2 and len(seq.cues) == 2
    assert ops.remove_cues(proj, seq, [n[0].id]) == 1



def test_title_card_cue_ops(proj):
    """字卡 are cues with ``style.kind == "card"``: their own layer, kept by subtitle imports, never merged with a subtitle."""
    ed = Editor(proj)
    seq = proj.active_sequence()
    sub = ops.add_cue(proj, seq, 0.0, 2.0, "字幕")
    card = ops.add_cue(proj, seq, 0.5, 3.0, "字卡", card=True)
    assert card.is_card and not sub.is_card and card.to_dict()["style"] == {"kind": "card"}
    assert CaptionCue.from_dict(card.to_dict()).is_card
    with pytest.raises(OpError, match="字卡和字幕不能合併"):
        ops.merge_cues(proj, seq, [sub.id, card.id])
    left, right = ops.split_cue(proj, seq, card.id, 1.5)
    assert left.is_card and right.is_card                                     # both halves stay cards
    ops.merge_cues(proj, seq, [left.id, right.id])
    new = ops.replace_cues(proj, seq, [(0.0, 1.0, "新字幕")])                 # importing subtitles keeps the card
    assert [q.text for q in new] == ["新字幕"] and sorted(q.text for q in seq.cues) == ["字卡", "新字幕"]
    ops.replace_cues(proj, seq, [(0.0, 1.0, "全部重來")], keep_cards=False)
    assert [q.text for q in seq.cues] == ["全部重來"]
    # turning a subtitle into a card (and back) is one undo step and keeps any other per-cue style
    seq.cues[0].style = {"note": "x"}
    qid, before = seq.cues[0].id, proj.content_json()
    ed.op("設為字卡", "update_cue", seq.id, qid, card=True)
    q = proj.active_sequence().find_cue(qid)
    assert q.is_card and q.style == {"note": "x", "kind": "card"}
    ed.op("改回字幕", "update_cue", seq.id, qid, card=False)
    assert proj.active_sequence().find_cue(qid).style == {"note": "x"}
    ed.undo()
    ed.undo()
    assert proj.content_json() == before
    # ripple edits move cards like subtitles (they are on the sequence timeline)
    ed.op("放入素材", "place_clip", seq.id, "a1", proj.active_sequence().video_tracks[0].id, start=0, src_out=10)
    seq = proj.active_sequence()
    c = ops.add_cue(proj, seq, 6.0, 8.0, "後面的字卡", card=True)
    ops.ripple_delete_range(proj, seq, 1.0, 3.0)
    assert (c.start, c.end) == (pytest.approx(4.0), pytest.approx(6.0)) and c.is_card


def test_srt_roundtrip():
    text = to_srt([(0.0, 1.5, "你好"), (2.0, 3.25, "world")])
    cues = parse_srt(text)
    assert cues == [(0.0, 1.5, "你好"), (2.0, 3.25, "world")]
    vtt = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.500\n<b>hi</b>\n"
    assert parse_srt(vtt) == [(1.0, 2.5, "hi")]


# ------------------------------------------------------------------------------ history
def test_history_undo_redo_and_rollback(proj):
    ed = Editor(proj)
    seq = proj.active_sequence()
    ed.op("放入素材", "place_clip", seq.id, "a1", seq.video_tracks[0].id, start=0, src_out=10)
    rev1 = proj.revision
    cid = seq.video_tracks[0].clips[0].id
    ed.op("分割", "split_clip", seq.id, cid, 5.0)
    assert len(proj.active_sequence().video_tracks[0].clips) == 2
    assert ed.history.undo_label == "分割"
    ed.undo()
    assert len(proj.active_sequence().video_tracks[0].clips) == 1
    ed.redo()
    assert len(proj.active_sequence().video_tracks[0].clips) == 2
    assert proj.revision > rev1                                   # undo/redo also advance the revision
    before = proj.content_sha256()
    with pytest.raises(OpError):
        ed.op("壞操作", "split_clip", seq.id, cid, 9999)
    assert proj.content_sha256() == before                        # rolled back, no history entry
    assert ed.history.undo_label == "分割"
    # no-op edits are not recorded
    n = len(ed.history._undo)
    ed.op("什麼都沒改", "set_clip_props", seq.id, cid)
    assert len(ed.history._undo) == n


def test_history_transaction_groups_steps(proj):
    ed = Editor(proj)
    seq = proj.active_sequence()
    with ed.history.transaction("一次做兩件事") as p:
        ops.place_clip(p, p.sequence(seq.id), "a1", seq.video_tracks[0].id, start=0, src_out=4)
        ops.add_cue(p, p.sequence(seq.id), 0, 2, "hi")
    assert len(ed.history._undo) == 1
    ed.undo()
    s = proj.active_sequence()
    assert not s.video_tracks[0].clips and not s.cues


# ------------------------------------------------------------------------------ project io
def test_project_save_load_roundtrip(tmp_path, proj):
    seq = proj.active_sequence()
    c = ops.place_clip(proj, seq, "a1", vt(seq).id, start=0, src_out=7)
    c.transform.keys = [[0, 0.3, 0.5], [3, 0.6, 0.5]]
    ops.add_cue(proj, seq, 0.5, 2.0, "字幕")
    path = tmp_path / "x.acproj"
    proj.save(path)
    proj.save(path)                                    # second save keeps a .bak
    assert (tmp_path / "x.acproj.bak").exists()
    p2 = Project.load(path)
    assert p2.content_sha256() == proj.content_sha256()
    assert p2.path == str(path)
    with pytest.raises(ValueError):
        Project.from_dict({"schema": "other/1"})


# ------------------------------------------------------------------------------- plan
def make_plan(proj) -> EditPlan:
    seq = Sequence.create("S", 1080, 1920, 30, seq_id="s_plan")
    return EditPlan(kind="shorts", brief={"count": 1}, sources=[{"asset_id": "a1"}],
                    sequences=[seq.to_dict()], decisions=[{"id": "d1", "kind": "cut", "summary": "x"}])


def test_plan_hash_is_stable_and_tamper_evident(tmp_path, proj):
    plan = make_plan(proj)
    h = plan.sha256()
    plan.created = "different timestamp"
    assert plan.sha256() == h                       # volatile field excluded
    plan.save(tmp_path / "p.json")
    assert EditPlan.load(tmp_path / "p.json").sha256() == h
    raw = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    raw["brief"]["count"] = 99
    with pytest.raises(PlanError):
        EditPlan.from_dict(raw)
    assert sha256_hex(canonical_json({"b": 1, "a": [1.0000001, 2]})) == sha256_hex('{"a":[1.0,2],"b":1}')


def test_receipt_binding(proj):
    plan = make_plan(proj)
    ok = AuditReceipt(plan.sha256(), proj.revision, proj.content_sha256(), True, [])
    check_receipt(plan, ok, proj)
    bad_state = AuditReceipt(plan.sha256(), proj.revision, proj.content_sha256(), False, [Finding("r", "error", "m")])
    with pytest.raises(PlanError):
        check_receipt(plan, bad_state, proj)
    other = AuditReceipt("0" * 64, proj.revision, "", True, [])
    with pytest.raises(PlanError):
        check_receipt(plan, other, proj)
    proj.touch()
    with pytest.raises(PlanError):
        check_receipt(plan, ok, proj)               # project edited after the audit -> stale


def test_run_provenance_machine_cannot_approve(proj):
    plan = make_plan(proj)
    rc = AuditReceipt(plan.sha256(), proj.revision, "", True, [Finding("w", "warn", "careful")])
    run = record_run(proj, plan, rc, ["s_plan"])
    assert run["human_review"] == "pending" and run["technical_qa"] == "not_run"
    set_technical_qa(proj, run["run_id"], "pass", "ok")
    assert run["human_review"] == "pending"          # QA never certifies taste
    set_human_review(proj, run["run_id"], "approved", "looks good")
    assert run["human_review"] == "approved" and run["outcomes"]
    with pytest.raises(PlanError):
        set_human_review(proj, run["run_id"], "certified")


def test_sequence_ops(proj):
    s2 = ops.create_sequence(proj, "直式", 1080, 1920, 30)
    assert proj.active == s2.id and s2.is_portrait and s2.aspect_label == "9:16"
    d = ops.duplicate_sequence(proj, s2)
    assert d.id != s2.id and len(proj.sequences) == 3
    ops.delete_sequence(proj, d.id)
    assert len(proj.sequences) == 2
    seq = proj.sequence(s2.id)
    ops.place_clip(proj, seq, "a1", seq.video_tracks[0].id, start=0, src_out=5)
    with pytest.raises(OpError):
        ops.remove_asset(proj, "a1")
    assert ops.remove_asset(proj, "a1", force=True) == 1


def test_sequence_range(proj):
    seq = proj.active_sequence()
    ops.set_sequence_range(proj, seq, 1.0, 4.0)
    assert seq.meta["range"] == [1.0, 4.0]
    with pytest.raises(OpError):
        ops.set_sequence_range(proj, seq, 4.0, 4.0)
    ops.set_sequence_range(proj, seq, None, None)
    assert "range" not in seq.meta
