"""Captions and markers follow the main video track (V1) through ripple edits; other tracks leave them alone.

A caption is timed against the words on V1.  When a ripple edit changes WHEN the material after it plays, the captions
after it have to move with it, or they drift off the speech (found by the independent code review).  The randomised test
at the end states the rule in one sentence: *a caption stays glued to the footage moment it was on*.
"""
from __future__ import annotations

import math
import random

import pytest

from autocut.core import ops
from autocut.core.editor import Editor
from autocut.core.ops import MIN_CUE, OpError
from autocut.core.project import Project, ordered_clips, validate_sequence
from conftest import fake_asset

FPS = 30.0
ORIGINAL_CUES = [("a", 1.0, 2.0), ("b", 6.0, 7.0), ("c", 13.0, 14.0)]
ORIGINAL_MARKERS = [("m-b", 5.0), ("m-c", 12.5)]


def vt(seq):
    return seq.video_tracks[0]


def span(c):
    return (round(c.start, 3), round(c.end, 3))


def clips_of(seq):
    return [span(c) for c in vt(seq).clips]


def cues_of(seq):
    return [(q.text, round(q.start, 3), round(q.end, 3)) for q in seq.cues]


def markers_of(seq):
    return [(m.name, round(m.time, 3)) for m in seq.markers]


def story(proj):
    """V1: A 0-5 | B 5-12 | C 12-17 (one source, back to back).  Captions inside each, markers at B's start and inside C;
    a clip on V2 and one on A1 (a music bed) that must never move."""
    seq = proj.active_sequence()
    v1, v2, a1 = vt(seq), seq.video_tracks[1], seq.audio_tracks[0]
    A = ops.place_clip(proj, seq, "a1", v1.id, start=0, src_in=0, src_out=5)
    B = ops.place_clip(proj, seq, "a1", v1.id, start=5, src_in=5, src_out=12)
    C = ops.place_clip(proj, seq, "a1", v1.id, start=12, src_in=12, src_out=17)
    over = ops.place_clip(proj, seq, "a2", v2.id, start=6, src_in=0, src_out=3)
    music = ops.place_clip(proj, seq, "m1", a1.id, start=0, src_in=0, src_out=17)
    ops.add_cue(proj, seq, 1.0, 2.0, "a")
    ops.add_cue(proj, seq, 6.0, 7.0, "b")
    ops.add_cue(proj, seq, 13.0, 14.0, "c")
    ops.add_marker(proj, seq, 5.0, "m-b")
    ops.add_marker(proj, seq, 12.5, "m-c")
    assert validate_sequence(proj, seq) == [] and cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS
    return seq, A, B, C, over, music


def others_stay(over, music):
    assert span(over) == (6.0, 9.0) and span(music) == (0.0, 17.0)


# ---------------------------------------------------------------------------------------------------- delete
def test_ripple_delete_of_the_first_clip_pulls_the_captions_up_and_drops_its_own(proj):
    seq, A, B, C, over, music = story(proj)
    ops.remove_clips(proj, seq, [A.id], ripple=True)
    assert clips_of(seq) == [(0.0, 7.0), (7.0, 12.0)]
    assert cues_of(seq) == [("b", 1.0, 2.0), ("c", 8.0, 9.0)]                    # "a" went with its clip
    assert markers_of(seq) == [("m-b", 0.0), ("m-c", 7.5)]
    others_stay(over, music)                                                     # V2 and the music bed do not move


def test_ripple_delete_of_a_middle_clip_closes_the_captions_over_the_hole(proj):
    seq, A, B, C, over, music = story(proj)
    ops.remove_clips(proj, seq, [B.id], ripple=True)
    assert clips_of(seq) == [(0.0, 5.0), (5.0, 10.0)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("c", 6.0, 7.0)]
    assert markers_of(seq) == [("m-c", 5.5)]                                     # the marker at B's head was in the hole
    others_stay(over, music)


def test_a_delete_without_ripple_moves_nothing(proj):
    seq, A, B, C, over, music = story(proj)
    ops.remove_clips(proj, seq, [B.id])
    assert clips_of(seq) == [(0.0, 5.0), (12.0, 17.0)]
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


def test_a_gap_left_on_purpose_is_kept_and_its_captions_move_with_the_rest(proj):
    seq, A, B, C, over, music = story(proj)
    ops.remove_clips(proj, seq, [B.id])                                          # a gap 5 - 12 on purpose
    ops.remove_clips(proj, seq, [A.id], ripple=True)                             # only A's own 5 s are closed
    assert clips_of(seq) == [(7.0, 12.0)]                                        # the 7 s gap is still there
    assert cues_of(seq) == [("b", 1.0, 2.0), ("c", 8.0, 9.0)]
    assert markers_of(seq) == [("m-b", 0.0), ("m-c", 7.5)]


def test_a_ripple_delete_on_another_track_leaves_the_captions_alone(proj):
    seq, A, B, C, over, music = story(proj)
    ops.remove_clips(proj, seq, [over.id, music.id], ripple=True)
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS
    assert clips_of(seq) == [(0.0, 5.0), (5.0, 12.0), (12.0, 17.0)]


def test_removing_what_cannot_be_removed_changes_nothing_at_all(proj):
    """Everything is checked before anything is deleted: a clip on a locked track, or one that is gone, must not leave
    the V1 clip half removed (calling the op directly, without the Editor's rollback)."""
    seq, A, B, C, over, music = story(proj)
    seq.video_tracks[1].locked = True
    before = proj.content_json()
    with pytest.raises(OpError):
        ops.remove_clips(proj, seq, [A.id, over.id], ripple=True)
    assert proj.content_json() == before
    with pytest.raises(OpError):
        ops.remove_clips(proj, seq, [A.id, "c_nope"], ripple=True)
    assert proj.content_json() == before


# ---------------------------------------------------------------------------------------------------- trims
def test_ripple_trim_of_the_end_moves_what_follows_with_it(proj):
    seq, A, B, C, over, music = story(proj)
    ops.trim_clip_end(proj, seq, A.id, 3.0, ripple=True)                         # 2 s of A's tail go
    assert clips_of(seq) == [(0.0, 3.0), (3.0, 10.0), (10.0, 15.0)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("b", 4.0, 5.0), ("c", 11.0, 12.0)]
    assert markers_of(seq) == [("m-b", 3.0), ("m-c", 10.5)]
    ops.trim_clip_end(proj, seq, A.id, 4.0, ripple=True)                         # 1 s of A's source comes back
    assert cues_of(seq) == [("a", 1.0, 2.0), ("b", 5.0, 6.0), ("c", 12.0, 13.0)]
    assert markers_of(seq) == [("m-b", 4.0), ("m-c", 11.5)]
    others_stay(over, music)


def test_a_caption_in_the_trimmed_off_tail_goes_and_one_across_the_edge_is_cut(proj):
    seq, A, B, C, over, music = story(proj)
    ops.update_cue(proj, seq, seq.cues[0].id, 2.5, 4.5)                          # "a" now runs 2.5 - 4.5, inside A (0 - 5)
    ops.trim_clip_end(proj, seq, A.id, 3.0, ripple=True)
    assert cues_of(seq)[0] == ("a", 2.5, 3.0)                                    # cut at the new end ...
    ops.trim_clip_end(proj, seq, A.id, 2.0, ripple=True)
    assert [t for t, _s, _e in cues_of(seq)] == ["b", "c"]                       # ... and gone when too little is left


def test_ripple_trim_of_the_start_closes_the_head_for_the_captions_too(proj):
    seq, A, B, C, over, music = story(proj)
    ops.trim_clip_start(proj, seq, B.id, 7.0, ripple=True)                       # B loses its first 2 s and stays at 5
    assert clips_of(seq) == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("c", 11.0, 12.0)]                  # "b" was in the head that went
    assert markers_of(seq) == [("m-b", 5.0), ("m-c", 10.5)]                      # B's own label stays on B
    others_stay(over, music)
    ops.trim_clip_start(proj, seq, B.id, 4.0, ripple=True)                       # 1 s of earlier footage comes back
    assert cues_of(seq) == [("a", 1.0, 2.0), ("c", 12.0, 13.0)]
    assert markers_of(seq) == [("m-b", 5.0), ("m-c", 11.5)]


def test_a_marker_inside_the_head_that_went_goes_with_it(proj):
    seq, A, B, C, over, music = story(proj)
    ops.add_marker(proj, seq, 6.0, "inside")
    ops.trim_clip_start(proj, seq, B.id, 7.0, ripple=True)
    assert [m for m, _t in markers_of(seq)] == ["m-b", "m-c"]


def test_a_trim_without_ripple_moves_no_captions(proj):
    seq, A, B, C, over, music = story(proj)
    ops.trim_clip_end(proj, seq, A.id, 3.0)
    ops.trim_clip_start(proj, seq, C.id, 14.0)
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


def test_a_ripple_trim_on_another_track_leaves_the_captions_alone(proj):
    seq, A, B, C, over, music = story(proj)
    ops.trim_clip_end(proj, seq, over.id, 7.0, ripple=True)
    ops.trim_clip_start(proj, seq, music.id, 3.0, ripple=True)
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


# ---------------------------------------------------------------------------------------------------- speed
def test_a_speed_change_stretches_the_captions_inside_and_moves_the_ones_after(proj):
    seq, A, B, C, over, music = story(proj)
    ops.set_speed(proj, seq, B.id, 2.0, ripple=True)                             # B: 5 - 12  ->  5 - 8.5
    assert clips_of(seq) == [(0.0, 5.0), (5.0, 8.5), (8.5, 13.5)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("b", 5.5, 6.0), ("c", 9.5, 10.5)]
    assert markers_of(seq) == [("m-b", 5.0), ("m-c", 9.0)]
    ops.set_speed(proj, seq, B.id, 1.0, ripple=True)                             # and back again
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


def test_a_speed_change_without_ripple_leaves_what_follows_where_it_is(proj):
    seq, A, B, C, over, music = story(proj)
    ops.set_speed(proj, seq, B.id, 2.0, ripple=False)                            # B now ends at 8.5 and leaves a gap
    assert cues_of(seq) == [("a", 1.0, 2.0), ("b", 5.5, 6.0), ("c", 13.0, 14.0)]
    assert markers_of(seq) == ORIGINAL_MARKERS


def test_a_speed_change_on_another_track_leaves_the_captions_alone(proj):
    seq, A, B, C, over, music = story(proj)
    ops.set_speed(proj, seq, over.id, 2.0, ripple=True)
    ops.set_speed(proj, seq, music.id, 0.5, ripple=True)
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


def test_a_caption_squeezed_by_a_fast_clip_keeps_its_minimum_length(proj):
    seq, A, B, C, over, music = story(proj)
    ops.set_speed(proj, seq, B.id, 4.0, ripple=True)                             # "b" (1 s) would become 0.25 s - and 0.2 is the least
    (b,) = [q for q in seq.cues if q.text == "b"]
    assert b.end - b.start >= MIN_CUE - 1e-6 and b.text == "b"                   # shortened, never lost or negative


# ---------------------------------------------------------------------------------------------------- crossfade
def test_a_crossfade_pulls_the_words_of_what_follows_with_it(proj):
    seq, A, B, C, over, music = story(proj)
    ops.set_crossfade(proj, seq, B.id, 1.0)                                      # B (and C) start 1 s earlier
    assert clips_of(seq) == [(0.0, 5.0), (4.0, 11.0), (11.0, 16.0)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("b", 5.0, 6.0), ("c", 12.0, 13.0)]
    assert markers_of(seq) == [("m-b", 4.0), ("m-c", 11.5)]
    ops.set_crossfade(proj, seq, B.id, 0.0)                                      # a plain cut again
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


# ---------------------------------------------------------------------------------------------------- gaps and inserts
def test_closing_a_gap_closes_it_for_the_captions_and_a_caption_in_the_gap_goes(proj):
    seq, A, B, C, over, music = story(proj)
    ops.remove_clips(proj, seq, [B.id])                                          # a hole 5 - 12 (cue "b" stays in it)
    assert ops.close_gaps(proj, seq, vt(seq).id) == pytest.approx(7.0)
    assert clips_of(seq) == [(0.0, 5.0), (5.0, 10.0)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("c", 6.0, 7.0)]
    assert markers_of(seq) == [("m-c", 5.5)]


def test_closing_gaps_on_another_track_leaves_the_captions_alone(proj):
    seq, A, B, C, over, music = story(proj)
    ops.place_clip(proj, seq, "a2", seq.video_tracks[1].id, start=20, src_in=0, src_out=2)
    assert ops.close_gaps(proj, seq, seq.video_tracks[1].id) > 0
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


def test_an_insert_edit_pushes_the_captions_after_it(proj):
    seq, A, B, C, over, music = story(proj)
    ops.place_clip(proj, seq, "a2", vt(seq).id, start=5, src_in=0, src_out=2, mode="insert")
    assert clips_of(seq)[1:] == [(5.0, 7.0), (7.0, 14.0), (14.0, 19.0)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("b", 8.0, 9.0), ("c", 15.0, 16.0)]
    assert markers_of(seq) == [("m-b", 7.0), ("m-c", 14.5)]
    others_stay(over, music)


def test_other_ways_to_put_a_clip_down_do_not_push_the_captions(proj):
    seq, A, B, C, over, music = story(proj)
    ops.place_clip(proj, seq, "a2", vt(seq).id, start=30, src_in=0, src_out=2)                  # overwrite on free space
    ops.place_clip(proj, seq, "a2", vt(seq).id, src_in=0, src_out=2, mode="append")
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS


def test_moving_a_clip_in_insert_mode_pushes_the_captions_but_a_plain_move_does_not(proj):
    seq, A, B, C, over, music = story(proj)
    ops.move_clip(proj, seq, C.id, 30.0)
    assert cues_of(seq) == ORIGINAL_CUES and markers_of(seq) == ORIGINAL_MARKERS
    ops.move_clip(proj, seq, C.id, 5.0, resolve="insert")                        # C is 5 s long: 5 s of room open at 5
    assert clips_of(seq) == [(0.0, 5.0), (5.0, 10.0), (10.0, 17.0)]
    assert cues_of(seq) == [("a", 1.0, 2.0), ("b", 11.0, 12.0), ("c", 18.0, 19.0)]


# ---------------------------------------------------------------------------------------------------- range delete
def test_ripple_delete_range_closes_the_captions_as_it_always_did(proj):
    seq, A, B, C, over, music = story(proj)
    ops.ripple_delete_range(proj, seq, 4.0, 8.0)
    assert cues_of(seq) == [("a", 1.0, 2.0), ("c", 9.0, 10.0)]                   # "b" was inside the 4 s that went
    assert markers_of(seq) == [("m-c", 8.5)]


# ---------------------------------------------------------------------------------------------------- undo
def test_one_undo_gives_back_the_clips_and_the_captions_together(proj):
    ed = Editor(proj)
    seq, A, B, C, over, music = story(proj)
    ed.history.clear()
    before = proj.content_json()
    ed.op("波紋刪除", "remove_clips", seq.id, [A.id], ripple=True)
    assert proj.content_json() != before and cues_of(proj.active_sequence()) == [("b", 1.0, 2.0), ("c", 8.0, 9.0)]
    ed.undo()
    assert proj.content_json() == before
    ed.redo()
    assert cues_of(proj.active_sequence()) == [("b", 1.0, 2.0), ("c", 8.0, 9.0)]


# ---------------------------------------------------------------------------------------------------- the rule, randomised
def glue_world(proj, rng: random.Random):
    """V1 of 4-7 clips (some gaps, some crossfades) with captions at random places; every caption is glued to the footage
    moment it is on: ``(clip id, source time)``."""
    seq = proj.active_sequence()
    v1 = vt(seq)
    t = 0.0
    for _ in range(rng.randint(4, 7)):
        t += rng.choice([0.0, 0.0, 0.0, 0.5, 1.0])
        s_in = round(rng.uniform(0, 10) * FPS) / FPS
        length = round(rng.uniform(1.5, 6.0) * FPS) / FPS
        c = ops.place_clip(proj, seq, "a1", v1.id, start=t, src_in=s_in, src_out=s_in + length)
        t = c.end
    for c in list(v1.clips):
        if rng.random() < 0.3:
            try:
                ops.set_crossfade(proj, seq, c.id, rng.choice([0.5, 1.0]))
            except OpError:
                pass
    glued: dict[str, tuple[str, float]] = {}
    for _ in range(rng.randint(4, 9)):
        c = rng.choice(v1.clips)
        off = round(rng.uniform(0.0, max(0.0, c.duration - 0.8)) * FPS) / FPS
        q = ops.add_cue(proj, seq, c.start + off, c.start + off + MIN_CUE, f"q{len(glued)}")
        top = ordered_clips([o for o in v1.clips if o.start - 1e-6 <= q.start < o.end - 1e-6])[-1]
        glued[q.id] = (top.id, top.src_in + (q.start - top.start) * top.speed)
    return seq, glued


def check_glue(seq, glued, slack_frames: float, edge_frames: float = 2.0) -> int:
    """Every caption that still exists, whose footage moment is still on its clip and is not inside a crossfade (where two
    clips play at once and nobody can say which one the words belong to) sits where that moment now plays.  A caption
    that is inside a crossfade is glued anew, to the clip that is on top there.  So is one that sits within
    ``edge_frames`` of a cut point (a clip's start or end): rounding to the frame grid decides which side of the cut it
    is on, and either side is right.  Returns how many captions were checked."""
    v1 = vt(seq)
    checked = 0
    edge = edge_frames / FPS + 1e-9
    cut_points = [b for o in v1.clips for b in (o.start, o.end)]
    for q in seq.cues:
        cid, src_t = glued[q.id]
        c = next((o for o in v1.clips if o.id == cid), None)
        under = ordered_clips([o for o in v1.clips if o.start - 1e-6 <= q.start < o.end - 1e-6])
        if len(under) > 1:                                        # a crossfade: the owner is the clip on top
            top = under[-1]
            glued[q.id] = (top.id, top.src_in + (q.start - top.start) * top.speed)
            continue
        if c is None or not (c.src_in + 1e-6 <= src_t <= c.src_out - 1e-6):
            if len(under) == 1:                                   # its footage is gone: it hangs on the clip it is on now
                top = under[0]
                glued[q.id] = (top.id, top.src_in + (q.start - top.start) * top.speed)
            continue
        want = c.start + (src_t - c.src_in) / c.speed
        covering = ordered_clips([o for o in v1.clips if o.start - 1e-6 <= want < o.end - 1e-6])
        if len(covering) != 1 or covering[0] is not c:
            continue
        assert abs(q.start - want) <= slack_frames / FPS + 1e-6, (q.text, q.start, want, c.id)
        checked += 1
        if len(under) == 1 and any(abs(q.start - b) <= edge for b in cut_points):
            glued[q.id] = (under[0].id, under[0].src_in + (q.start - under[0].start) * under[0].speed)
    return checked


def random_ripple_step(ed: Editor, rng: random.Random, seq, c) -> float:
    """One random ripple-style edit of V1 through the Editor; returns by how much it stretched the clip it changed (1.0 when
    it was not a speed change: 4.0 means the clip now takes four times as long)."""
    v1 = vt(seq)
    kind = rng.choice(["remove", "trim_end", "trim_start", "speed", "xfade", "gaps", "insert"])
    if kind == "remove":
        ed.op("波紋刪除", "remove_clips", seq.id, [c.id], ripple=True)
    elif kind == "trim_end":
        ed.op("修剪結尾", "trim_clip_end", seq.id, c.id, c.end + rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0]), ripple=True)
    elif kind == "trim_start":
        ed.op("修剪開頭", "trim_clip_start", seq.id, c.id, c.start + rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0]), ripple=True)
    elif kind == "speed":
        old, new = c.speed, rng.choice([0.5, 0.75, 1.25, 1.5, 2.0])
        ed.op("速度", "set_speed", seq.id, c.id, new, ripple=True)
        return old / new
    elif kind == "xfade":
        ed.op("交叉淡化", "set_crossfade", seq.id, c.id, rng.choice([0.0, 0.5, 1.0]))
    elif kind == "gaps":
        ed.op("移除間隙", "close_gaps", seq.id, v1.id)
    else:
        ed.op("插入", "place_clip", seq.id, "a2", v1.id, rng.choice([o.start for o in v1.clips]), 0.0, 1.0, "insert")
    return 1.0


def run_glue_seed(seed: int, steps: int = 10, exact: bool = False) -> int:
    """``exact``: the caller has switched frame snapping off, so every caption must land exactly where its footage plays.
    Otherwise each edit may be off by the rounding of a frame and a later slow-down multiplies what is already off."""
    proj = Project.new("glue")
    proj.assets["a1"] = fake_asset("a1", 20.0)
    proj.assets["a2"] = fake_asset("a2", 10.0)
    rng = random.Random(5000 + seed)
    seq, glued = glue_world(proj, rng)
    ed = Editor(proj)
    ed.history.clear()
    checked, slack = 0, 0.0                                                       # slack in frames
    for _step in range(steps):
        seq = proj.active_sequence()
        if len(vt(seq).clips) < 2:
            break
        try:
            stretch = random_ripple_step(ed, rng, seq, rng.choice(vt(seq).clips))
        except OpError:
            continue
        slack = slack * max(1.0, stretch) + 1.0
        seq = proj.active_sequence()
        assert validate_sequence(proj, seq) == []
        bad = [(q.text, q.start, q.end) for q in seq.cues if not (q.end - q.start >= MIN_CUE - 2e-6 and q.start >= 0.0)]  # times are kept to 6 decimals
        assert not bad, bad
        assert [round(q.start, 6) for q in seq.cues] == sorted(round(q.start, 6) for q in seq.cues)
        if exact:
            checked += check_glue(seq, glued, slack_frames=1e-3, edge_frames=0.0)
        else:
            checked += check_glue(seq, glued, slack_frames=slack)
    return checked


@pytest.mark.parametrize("seed", range(150))
def test_captions_stay_glued_to_their_footage_through_random_ripple_edits(seed):
    run_glue_seed(seed)


@pytest.mark.parametrize("seed", range(150))
def test_with_frame_snapping_switched_off_the_captions_land_exactly_on_their_footage(seed, monkeypatch):
    """The strict version: no rounding at all, so any drift - however small - is a mistake in the mapping itself."""
    monkeypatch.setattr(ops, "_sn", lambda seq, t: max(0.0, float(t)))
    run_glue_seed(seed, steps=14, exact=True)


def test_a_cut_never_leaves_a_caption_a_hair_before_zero(proj):
    """Float noise (found by the 6000-seed campaign): a caption right after a cut moved up by ``q.start - d`` came out as
    -4.4e-16, below zero."""
    seq = proj.active_sequence()
    ops.add_cue(proj, seq, 2.0, 2.5, "x")
    ops.add_marker(proj, seq, 2.0, "m")
    ops._cut_captions(seq, 0.0, 2.0 + 4e-15)
    assert [q.start for q in seq.cues] == [0.0] and math.copysign(1.0, seq.cues[0].start) > 0
    assert seq.cues[0].end - seq.cues[0].start >= MIN_CUE - 1e-6
    assert [m.time for m in seq.markers] == [0.0]


def test_the_glue_test_really_checks_captions_and_catches_a_caption_left_behind(monkeypatch):
    checked = sum(run_glue_seed(seed) for seed in range(60))
    assert checked >= 200, checked                                               # not vacuous: captions were really compared
    monkeypatch.setattr(ops, "_sn", lambda seq, t: max(0.0, float(t)))           # from here on without frame snapping
    exact = sum(run_glue_seed(seed, steps=14, exact=True) for seed in range(60))
    assert exact >= 200, exact
    proj = Project.new("g")
    proj.assets["a1"] = fake_asset("a1", 20.0)
    seq = proj.active_sequence()
    a = ops.place_clip(proj, seq, "a1", vt(seq).id, start=0, src_in=0, src_out=4)
    ops.place_clip(proj, seq, "a1", vt(seq).id, start=4, src_in=4, src_out=8)
    q = ops.add_cue(proj, seq, 5.0, 6.0, "b")
    top = ordered_clips([o for o in vt(seq).clips if o.start - 1e-6 <= q.start < o.end - 1e-6])[-1]
    glued = {q.id: (top.id, top.src_in + (q.start - top.start) * top.speed)}
    ops.remove_clips(proj, seq, [a.id], ripple=True)
    assert check_glue(seq, glued, 1.0) == 1                                      # it followed its clip: fine
    seq.cues[0].start += 1.0                                                     # ... drifted by a second: caught
    seq.cues[0].end += 1.0
    with pytest.raises(AssertionError):
        check_glue(seq, glued, 1.0)
    seq.cues[0].start -= 1.0 - 0.02                                              # ... and 0.02 s (two thirds of a frame) is caught too
    seq.cues[0].end -= 1.0 - 0.02
    with pytest.raises(AssertionError):                                          # by the strict check, not the loose one
        check_glue(seq, glued, 1e-3, edge_frames=0.0)
