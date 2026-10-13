"""Assemble (many clips -> one long video): pure-logic unit tests with fake Evidence (no ffmpeg) plus integration tests on
synthetic ffmpeg/lavfi footage: landscape + portrait + a source with a blurry stretch and an all-black stretch + a still image.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from autocut.analysis.evidence import Evidence
from autocut.autoedit import assemble as A
from autocut.autoedit.api import AutoEditError, Cancelled, apply_plan, audit_plan, preview_project, propose_assemble
from autocut.autoedit.config import AssembleParams
from autocut.core.editor import Editor
from autocut.core.plan import EditPlan
from autocut.core.project import Asset, Project, validate_sequence
from conftest import HAVE_FFMPEG

needs_ffmpeg = pytest.mark.ffmpeg


# ======================================================================================= fake-evidence helpers
def fake_ev(sec: float = 20.0, vf: float = 5.0, af: float = 20.0, *, sharp=None, bright=None, motion=None, voice=None,
            rms=None, silences=None, scenes=None, has_audio: bool = True, faces=None) -> Evidence:
    nv, na = int(sec * vf), int(sec * af)
    ev = Evidence.empty("x", sec, visual_fps=vf, audio_fps=af, has_audio=has_audio)
    ev.sharp = list(sharp) if sharp is not None else [100.0] * nv
    ev.bright = list(bright) if bright is not None else [0.5] * nv
    ev.motion = list(motion) if motion is not None else [0.05] * nv
    if has_audio:
        ev.voice = list(voice) if voice is not None else [0.0] * na
        ev.rms_db = list(rms) if rms is not None else [-60.0] * na
        ev.silence_thr_db = -40.0
    ev.silences = [list(s) for s in (silences or [])]
    ev.scenes = list(scenes or [])
    ev.faces = faces if faces is not None else [[] for _ in range(nv)]
    ev.face_sample_mask = [1] * nv if faces is not None else [0] * nv
    return ev


def fake_asset(aid: str, w: int = 1920, h: int = 1080, dur: float = 10.0, fps: float = 30.0, kind: str = "video",
               created: str = "", name: str = "") -> Asset:
    return Asset(id=aid, path=f"/nonexistent/{name or aid}.mp4", name=name or f"{aid}.mp4", kind=kind, duration=dur, width=w,
                 height=h, fps=fps, has_video=True, has_audio=True, audio_channels=2, sample_rate=48000, created=created,
                 fingerprint=aid)


def cand(src: int, lo: float, hi: float, s: float | None = None, e: float | None = None, score: float = 0.5, **kw) -> A._Cand:
    return A._Cand(src, lo, hi, lo if s is None else s, hi if e is None else e, score, {}, **kw)


# ======================================================================================= quality filter (spec 4.4)
def test_quality_filter_clean_shot():
    ev = fake_ev()
    reason, m = A.shot_quality(ev, 2.0, 6.0)
    assert reason == "" and m["bright"] == 0.5 and m["sharp_rel"] == 1.0


def test_quality_filter_blurry_relative_to_the_source_median():
    sharp = [100.0] * 20 + [8.0] * 20 + [100.0] * 60                     # 4-8 s are blurry
    ev = fake_ev(sharp=sharp)
    assert A.shot_quality(ev, 4.0, 8.0)[0] == "blur"
    assert A.shot_quality(ev, 0.0, 4.0)[0] == ""
    assert A.shot_quality(ev, 8.0, 12.0)[0] == ""
    # a source that is blurry all the way through has no sharper reference: it is not flagged (documented limit)
    assert A.shot_quality(fake_ev(sharp=[8.0] * 100), 0.0, 5.0)[0] == ""


def test_quality_filter_dark_and_overexposed():
    bright = [0.5] * 20 + [0.02] * 20 + [0.98] * 20 + [0.5] * 40
    ev = fake_ev(bright=bright)
    assert A.shot_quality(ev, 4.0, 8.0)[0] == "dark"
    assert A.shot_quality(ev, 8.0, 12.0)[0] == "bright"
    assert A.shot_quality(ev, 12.0, 16.0)[0] == ""


def test_quality_filter_static_and_silent_only_when_both():
    still = [0.0] * 100
    quiet = fake_ev(motion=still)                                          # nothing moves, nothing is heard
    assert A.shot_quality(quiet, 2.0, 8.0)[0] == "static"
    talking = fake_ev(motion=still, voice=[0.9] * 400, rms=[-20.0] * 400)  # a static talking head is fine
    assert A.shot_quality(talking, 2.0, 8.0)[0] == ""
    no_audio = fake_ev(motion=still, has_audio=False)
    assert A.shot_quality(no_audio, 2.0, 8.0)[0] == "static"
    assert A.shot_quality(fake_ev(), 2.0, 8.0)[0] == ""                    # moving + quiet is fine


def test_quality_filter_shaky():
    jerky = ([0.0, 0.0, 0.6, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6] * 10)[:100]
    ev = fake_ev(motion=jerky)
    reason, m = A.shot_quality(ev, 0.0, 10.0)
    assert reason == "shaky" and m["motion_cv"] > A.SHAKY_CV
    assert A.shot_quality(fake_ev(motion=[0.2] * 100), 0.0, 10.0)[0] == ""            # steady strong motion is fine
    assert A.shot_quality(fake_ev(motion=jerky), 0.0, 1.0)[0] == ""                    # too few samples to judge


def test_quality_filter_priority_and_missing_evidence():
    ev = fake_ev(bright=[0.01] * 100, sharp=[100.0] * 50 + [1.0] * 50)
    assert A.shot_quality(ev, 12.0, 18.0)[0] == "dark"                    # dark wins over blurry
    empty = Evidence.empty("e", 10.0)                                      # no picture analysis at all
    assert A.shot_quality(empty, 0.0, 5.0)[0] == ""


# ======================================================================================= windows and shots
def test_window_prefers_the_sharp_part():
    ev = fake_ev(sec=10.0, sharp=[20.0] * 30 + [100.0] * 20)
    f = A._Feat(ev)
    assert A._best_window(f, 0.0, 10.0, 3.0) >= 6.0 - 1e-9


def test_window_prefers_the_voiced_part():
    ev = fake_ev(sec=10.0, voice=[1.0] * 60 + [0.0] * 140)                 # someone talks during the first 3 s
    f = A._Feat(ev)
    assert f.has_voice and A._best_window(f, 0.0, 10.0, 3.0) == 0.0
    s, parts = f.score(0.0, 3.0)
    assert parts["voice"] == 1.0 and s > f.score(6.0, 9.0)[0]


def test_window_edges_snap_into_silences():
    sil = [(2.0, 2.6), (7.0, 7.5)]
    assert A._snap_edge(2.3, sil, 0.0, 10.0) == 2.3                         # already inside a silence
    assert 2.0 < A._snap_edge(1.8, sil, 0.0, 10.0) < 2.6                    # nudged into it
    assert A._snap_edge(4.5, sil, 0.0, 10.0) == 4.5                         # nothing within 0.5 s
    assert A._snap_edge(1.8, sil, 1.9, 10.0) >= 1.9                         # never leaves the allowed range


def test_chunks_follow_scene_cuts_and_split_long_shots():
    ev = fake_ev(sec=18.0, scenes=[5.9, 9.9, 12.9])
    ch = A.chunks_for(ev, 18.0, 0.0)
    assert [(c.a, c.b) for c in ch] == [(0.0, 5.9), (5.9, 9.9), (9.9, 12.9), (12.9, 18.0)]
    assert ch[0].cut_head is None and ch[0].cut_tail == 5.9 and ch[1].cut_head == 5.9 and ch[3].cut_tail is None
    split = A.chunks_for(ev, 18.0, 3.0)
    assert all(c.b - c.a <= 3.0 + 1e-9 for c in split)
    assert sum(c.b - c.a for c in split) == pytest.approx(18.0)
    inner = [c for c in split if c.a == 0.0 or c.b == 5.9]
    assert inner[0].mh and not inner[0].mt                                  # artificial split edges need no flash margin
    assert A.chunks_for(ev, 18.0, 0.0)[1].mh and A.chunks_for(ev, 18.0, 0.0)[1].mt
    near_end = A.chunks_for(fake_ev(sec=18.0, scenes=[0.1, 17.9]), 18.0, 0.0)
    assert len(near_end) == 1                                               # cuts within 0.2 s of the ends are ignored


def test_long_shot_splits_at_silences_when_someone_talks():
    ev = fake_ev(sec=20.0, voice=[1.0] * 400, silences=[(9.6, 10.4)])
    f = A._Feat(ev)
    parts = A._split_long(0.0, 20.0, 12.0, f)                               # two halves: the cut moves into the pause
    assert len(parts) == 2 and 9.6 < parts[0][1] < 10.4 and parts[0][1] == parts[1][0]


def test_make_candidate_margins_and_too_short():
    ev = fake_ev(sec=20.0, scenes=[10.0])
    f = A._Feat(ev)
    shape = A.resolve_shape(AssembleParams(target_seconds=0, min_shot=1.2))
    ch = A.chunks_for(ev, 20.0, 0.0, f)[0]
    c, drop = A.make_candidate(0, ch, ev, f, shape)
    assert drop is None and (c.lo, c.hi) == (0.3, 9.7) and c.cut_tail == 10.0
    tiny = A._Chunk(3.0, 3.8, True, True, 3.0, 3.8)
    c2, d2 = A.make_candidate(0, tiny, ev, f, shape)
    assert c2 is None and d2["reason"] == "short" and "太短" in d2["text"]
    snug = A._Chunk(3.0, 4.5, True, True, 3.0, 4.5)                         # 1.5 s shot, min 1.2: margin shrinks to 0.15 s
    c3, _ = A.make_candidate(0, snug, ev, f, shape)
    assert c3.hi - c3.lo == pytest.approx(1.2)


def test_make_candidate_drops_bad_shots_and_respects_drop_bad_off():
    ev = fake_ev(sec=20.0, bright=[0.5] * 50 + [0.01] * 50)                 # dark from 10 s on
    f = A._Feat(ev)
    ch = A._Chunk(10.0, 16.0, True, True, 10.0, None)
    on = A.resolve_shape(AssembleParams(target_seconds=0, drop_bad=True))
    off = A.resolve_shape(AssembleParams(target_seconds=0, drop_bad=False))
    c, d = A.make_candidate(0, ch, ev, f, on)
    assert c is None and d["reason"] == "dark" and "太暗" in d["text"]
    c, d = A.make_candidate(0, ch, ev, f, off)
    assert c is not None and d is None


def test_budget_mode_picks_a_window_of_pace_length():
    ev = fake_ev(sec=30.0)
    f = A._Feat(ev)
    shape = A.resolve_shape(AssembleParams(target_seconds=60, pace="medium"))
    assert (shape.lo_len, shape.hi_len, shape.max_chunk) == (2.0, 5.0, 5.0)
    c, _ = A.make_candidate(0, A._Chunk(0.0, 30.0, True, True, None, None), ev, f, shape)
    assert c.length == pytest.approx(3.5) and c.lo <= c.s and c.e <= c.hi
    assert A.resolve_shape(AssembleParams(target_seconds=60, pace="fast")).hi_len == 3.0
    assert A.resolve_shape(AssembleParams(target_seconds=60, max_shot=2.5)).max_chunk == 2.5


# ======================================================================================= budget / layout
def test_select_everything_when_no_target():
    srcs = [A._Src(0, None, None, None, [cand(0, 0, 4), cand(0, 5, 9)], usable=8),
            A._Src(1, None, None, None, [cand(1, 0, 3)], usable=3)]
    assert len(A.select_candidates(srcs, 0.0, 0.4)) == 3


def test_select_quota_is_sqrt_and_every_source_keeps_a_shot():
    def source(i: int, usable: int) -> A._Src:
        cs = [cand(i, k * 2.0, k * 2.0 + 2.0, score=1.0 - k * 0.01) for k in range(usable // 2)]
        return A._Src(i, None, None, None, cs, usable=float(usable))
    srcs = [source(0, 4), source(1, 16), source(2, 36)]
    chosen = A.select_candidates(srcs, 24.0, 0.0, 1.2)
    per = {i: sum(c.length for c in chosen if c.src == i) for i in range(3)}
    assert all(per[i] >= 2.0 for i in per)
    assert per[0] < per[1] < per[2]
    assert per[2] / per[0] < 36 / 4                                         # sqrt, not proportional
    assert sum(per.values()) == pytest.approx(24.0, abs=0.01)
    best_first = [c for c in chosen if c.src == 2]
    assert min(c.s for c in best_first) == 0.0                              # the highest-scoring shot is always taken


def test_select_tiny_target_still_gives_every_source_a_shot():
    srcs = [A._Src(i, None, None, None, [cand(i, 0, 3)], usable=3.0) for i in range(4)]
    chosen = A.select_candidates(srcs, 2.0, 0.4, 1.2)
    assert sorted(c.src for c in chosen) == [0, 1, 2, 3]


def test_select_exact_fill_with_a_trimmed_shot():
    srcs = [A._Src(0, None, None, None, [cand(0, 0, 4, score=0.9), cand(0, 5, 9, score=0.8), cand(0, 10, 14, score=0.7)], usable=12)]
    chosen = A.select_candidates(srcs, 9.5, 0.0, 1.2)
    assert A._eff(chosen, 0.0) == pytest.approx(9.5, abs=1e-3)
    assert any(c.length < 4.0 for c in chosen)                              # one shot was trimmed to close the gap
    short = A.select_candidates(srcs, 9.0, 0.0, 1.2)                        # a 1.0 s gap is below the minimum shot length:
    assert all(c.length >= 1.2 for c in short)                              # no sliver shot is invented, fit_to_target stretches


def test_abutting_windows_are_one_continuous_clip():
    a, b = cand(0, 0, 4, score=0.8), cand(0, 4, 8, score=0.4)
    a.parts, b.parts = {"sharp": 0.8, "voice": 0.2}, {"sharp": 0.4}
    assert A._eff([a, b], 0.5) == pytest.approx(8.0)                        # no crossfade inside one take
    apart = cand(0, 4.5, 8.5, score=0.4)
    assert A._eff([a, apart], 0.5) == pytest.approx(7.5)                    # a real jump does crossfade
    assert A._eff([a, cand(1, 4, 8)], 0.5) == pytest.approx(7.5)            # another source never merges
    merged = A.merge_adjacent([b, a, cand(0, 9, 12), cand(1, 0, 3)])
    assert [(m.src, m.s, m.e) for m in merged] == [(0, 0, 8), (0, 9, 12), (1, 0, 3)]
    assert merged[0].score == pytest.approx(0.6) and merged[0].parts == {"sharp": 0.6, "voice": 0.2}
    assert merged[0].chunk[1] == b.chunk[1] and merged[0].hi == 8
    assert len(A.merge_adjacent([cand(0, 0, 3, fixed=True), cand(0, 3, 6, fixed=True)])) == 2     # stills never merge


def test_fit_to_target_stretches_and_shrinks_windows():
    stretch = [cand(0, 0, 6, 1, 4, 0.9), cand(1, 0, 6, 1, 4, 0.5)]
    A.fit_to_target(stretch, 8.0, 0.0, 1.2, 1 / 30)
    assert A._eff(stretch, 0.0) == pytest.approx(8.0, abs=1 / 30) and stretch[0].length > stretch[1].length
    shrink = [cand(0, 0, 6, 0, 6, 0.9), cand(1, 0, 6, 0, 6, 0.1)]
    A.fit_to_target(shrink, 8.0, 0.0, 1.2, 1 / 30)
    assert A._eff(shrink, 0.0) == pytest.approx(8.0, abs=1 / 30) and shrink[1].length < shrink[0].length
    img = [cand(0, 0, 3, fixed=True), cand(1, 0, 6, 1, 4)]
    A.fit_to_target(img, 4.0, 0.0, 1.2, 1 / 30)
    assert img[0].length == 3.0                                             # still images keep their length


def test_layout_frames_crossfades_are_always_legal():
    assert A.layout_frames([30, 40, 12, 50], 12) == [0, 18, 53, 60]
    assert A.layout_frames([30, 40], 0) == [0, 30]                          # hard cuts
    assert A.layout_frames([30, 2], 12) == [0, 30]                          # 2-frame clip: cap 0 -> a cut
    rnd = random.Random(7)
    for _ in range(300):
        durs = [rnd.randint(1, 90) for _ in range(rnd.randint(2, 12))]
        xf = rnd.randint(0, 40)
        st = A.layout_frames(durs, xf)
        for i in range(1, len(durs)):
            ov = st[i - 1] + durs[i - 1] - st[i]
            assert 0 <= ov <= xf and ov * 2 < min(durs[i - 1], durs[i]) + 1      # strictly below half of the shorter clip
            if i + 1 < len(durs):
                assert st[i - 1] + durs[i - 1] <= st[i + 1]                      # never three clips at once


def test_refine_edges_corrects_edges_next_to_a_cut(monkeypatch):
    calls = []

    def fake_refine(path, t_est, fps, window=0.6):
        calls.append((path, t_est))
        return t_est + 0.05
    monkeypatch.setattr(A, "refine_cut", fake_refine)
    head = cand(0, 5.9, 9.0, 5.92, 8.0, cut_head=5.9)
    A.refine_edges(head, "x.mp4", 30.0)
    assert head.s == pytest.approx(5.95) and calls[0] == ("x.mp4", 5.9)
    tail = cand(0, 1.0, 9.9, 3.0, 9.98, cut_tail=10.0)
    monkeypatch.setattr(A, "refine_cut", lambda p, t, f, window=0.6: t - 0.05)
    A.refine_edges(tail, "x.mp4", 30.0)
    assert tail.e == pytest.approx(9.95)
    far = cand(0, 5.9, 9.0, 6.4, 8.0, cut_head=5.9)                          # 0.5 s away: no refinement needed
    n = len(calls)
    A.refine_edges(far, "x.mp4", 30.0)
    assert far.s == 6.4 and len(calls) == n
    monkeypatch.setattr(A, "refine_cut", lambda p, t, f, window=0.6: t + 5.0)    # nonsense answer is ignored
    bad = cand(0, 5.9, 9.0, 5.92, 8.0, cut_head=5.9)
    A.refine_edges(bad, "x.mp4", 30.0)
    assert bad.s == 5.92


# ======================================================================================= format / order / volume
def test_format_follows_the_length_weighted_majority():
    land = [fake_asset("a", dur=10), fake_asset("b", dur=10)]
    port = fake_asset("c", 1080, 1920, dur=30)
    w, h, fps, ev = A.resolve_format(land + [port], AssembleParams())
    assert (w, h) == (1080, 1920) and "直式" in ev["aspect_reason"]          # 30 s portrait outweighs 2 x 10 s landscape
    assert A.resolve_format(land + [fake_asset("d", 1080, 1080, dur=3)], AssembleParams())[:2] == (1920, 1080)
    assert A.resolve_format([fake_asset("s", 1000, 1000)], AssembleParams())[:2] == (1080, 1080)
    assert A.resolve_format(land, AssembleParams(aspect="9:16"))[:2] == (1080, 1920)
    assert A.resolve_format(land, AssembleParams(aspect="4:5"))[:2] == (1080, 1350)
    with pytest.raises(AutoEditError, match="畫面比例"):
        A.resolve_format(land, AssembleParams(aspect="7:3"))
    img = fake_asset("i", 1080, 1920, dur=0.0, kind="image")
    assert A.resolve_format([fake_asset("v", dur=2.0), img], AssembleParams())[:2] == (1080, 1920)   # image counts 3 s


def test_fps_is_the_most_common_normalised_rate():
    a = [fake_asset("a", fps=29.97), fake_asset("b", fps=30.0), fake_asset("c", fps=25.0)]
    assert A.resolve_format(a, AssembleParams())[2] == 30.0
    assert A.resolve_format(a, AssembleParams(fps=24))[2] == 24.0
    assert A.resolve_format([fake_asset("a", fps=120.0)], AssembleParams())[2] == 60.0
    assert A.resolve_format([fake_asset("a", fps=0.0)], AssembleParams())[2] == 30.0
    assert [A.normalize_fps(x) for x in (23.976, 29.97, 59.94, 48, 15, 24)] == [24, 30, 60, 50, 24, 24]


def test_order_modes():
    a = [fake_asset("x1", name="clip10", created="2026-01-01T00:00:00Z"),
         fake_asset("x2", name="clip2", created="2026-01-03T00:00:00Z"),
         fake_asset("x3", name="clip1", created="2026-01-02T00:00:00Z"),
         fake_asset("x4", name="Clip3", created="")]
    names = lambda lst: [x.name for x in lst]                                  # noqa: E731
    assert names(A.order_assets(a, "name")[0]) == ["clip1", "clip2", "Clip3", "clip10"]        # natural, case-insensitive
    assert names(A.order_assets(a, "as_given")[0]) == ["clip10", "clip2", "clip1", "Clip3"]
    ordered, ev = A.order_assets(a, "chronological")
    assert names(ordered) == ["clip10", "clip1", "clip2", "Clip3"]             # no time information sorts last
    assert [e["time_source"] for e in ev["order"]] == ["拍攝時間"] * 3 + ["無時間資訊"]


def test_order_falls_back_to_file_mtime_then_name(tmp_path):
    files = []
    for i, name in enumerate(["b.mp4", "a.mp4", "c.mp4", "d.mp4"]):
        p = tmp_path / name
        p.write_bytes(b"x")
        os.utime(p, (1_000_000 + [30, 20, 10, 10][i], 1_000_000 + [30, 20, 10, 10][i]))
        a = fake_asset(f"id{i}", name=name[:-4])
        a.path = str(p)
        files.append(a)
    ordered, ev = A.order_assets(files, "chronological")
    assert [x.name for x in ordered] == ["c", "d", "a", "b"]                    # c/d tie on mtime -> by name
    assert {e["time_source"] for e in ev["order"]} == {"檔案修改時間"}


def test_source_volumes_match_loudness_within_limits():
    def src(i: int, db: float) -> A._Src:
        ev = fake_ev(sec=10.0, rms=[db] * 200)
        return A._Src(i, None, ev, None)
    vols = A.source_volumes([src(0, -20.0), src(1, -26.0), src(2, -14.0)])
    assert vols[0][1] == 1.0                                                  # the median source stays as it is
    assert vols[1][1] == pytest.approx(2.0, abs=0.01) and vols[2][1] == pytest.approx(0.5, abs=0.01)
    extreme = A.source_volumes([src(0, -20.0), src(1, -60.0), src(2, 0.0)])
    assert extreme[1][1] == A.GAIN_MAX and extreme[2][1] == A.GAIN_MIN
    assert A.source_volumes([src(0, -20.0)]) == {0: (-20.0, 1.0)}             # nothing to match against


def test_params_are_validated_in_traditional_chinese(proj):
    for bad in (AssembleParams(order="random"), AssembleParams(transition="wipe"), AssembleParams(fit="stretch"),
                AssembleParams(captions="maybe"), AssembleParams(pace="warp"), AssembleParams(target_seconds=-5),
                AssembleParams(min_shot=0)):
        with pytest.raises(AutoEditError) as e:
            A.plan_assemble(proj, ["a1"], bad, None, None)
        assert any("一" <= ch <= "鿿" for ch in str(e.value))


def test_unknown_asset_and_audio_only_inputs(proj):
    with pytest.raises(AutoEditError, match="找不到素材"):
        A.plan_assemble(proj, ["nope"], AssembleParams(), None, None)
    with pytest.raises(AutoEditError, match="沒有可以組裝"):
        A.plan_assemble(proj, ["m1"], AssembleParams(), None, None)             # audio only
    with pytest.raises(AutoEditError, match="沒有可以組裝"):
        A.plan_assemble(proj, ["a1"], AssembleParams(), None, None)             # a1 is a fake path: file missing


# ======================================================================================= synthetic footage
BURST = "sine=frequency=440:sample_rate=48000:duration={d},volume='if(lt(mod(t,2),1),1,0)':eval=frame"
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-g", "15", "-c:a", "aac"]


@pytest.fixture(scope="module")
def footage(tmp_path_factory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("assemble_media")

    def ff(name: str, *args: str) -> None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, str(d / name)], check=True)

    ff("land_a.mp4", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8", "-f", "lavfi", "-i", BURST.format(d=8), *ENC, "-shortest")
    ff("land_b.mp4", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6,hue=h=90", "-f", "lavfi", "-i", BURST.format(d=6), *ENC, "-shortest")
    ff("port.mp4", "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30:duration=5", "-f", "lavfi", "-i", BURST.format(d=5), *ENC, "-shortest")
    # sharp 6 s | blurry 4 s | all black 3 s | sharp 5 s   (the bad parts are 6-10 s and 10-13 s)
    ff("mixed.mp4",
       "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6",
       "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=4,boxblur=12:3,negate",
       "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=30:duration=3",
       "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=5,hue=h=180",
       "-f", "lavfi", "-i", BURST.format(d=18),
       "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,setsar=1,format=yuv420p[v]",
       "-map", "[v]", "-map", "4:a", *ENC, "-shortest")
    ff("black.mp4", "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=30:duration=4", *ENC[:6])
    ff("still.png", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=1:duration=1", "-frames:v", "1")
    ff("music.wav", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=5")
    for n in ("clip10.mp4", "clip2.mp4", "clip1.mp4"):
        shutil.copy(d / "land_b.mp4", d / n)
    return d


def make_project(footage: Path, names: list[str]) -> tuple[Project, list[str]]:
    from autocut.media.probe import probe_asset
    p = Project.new("assemble")
    ids = []
    for n in names:
        a = probe_asset(footage / n)
        p.assets[a.id] = a
        ids.append(a.id)
    return p, ids


def v1_clips(plan: EditPlan) -> list:
    return plan.sequence_objects()[0].video_tracks[0].clips


def asset_names(plan: EditPlan, proj: Project) -> list[str]:
    out: list[str] = []
    for c in v1_clips(plan):
        n = proj.assets[c.asset_id].name
        if not out or out[-1] != n:
            out.append(n)
    return out


def no_errors(proj: Project, plan: EditPlan) -> None:
    view = Project.from_dict(proj.to_dict())
    for a in plan.assets:
        view.assets[a["id"]] = Asset.from_dict(a)
    for seq in plan.sequence_objects():
        errs = [i.to_dict() for i in validate_sequence(view, seq) if i.level == "error"]
        assert errs == []


# ======================================================================================= integration
@needs_ffmpeg
def test_bad_segments_are_dropped_with_decisions(footage):
    proj, ids = make_project(footage, ["mixed.mp4", "land_a.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given"))
    drops = [d for d in plan.decisions if d["kind"] == "drop"]
    reasons = {d["evidence"]["reason"] for d in drops}
    assert {"blur", "dark"} <= reasons
    assert all(d["why"] and d["source"]["asset_id"] == ids[0] for d in drops)
    mixed = [c for c in v1_clips(plan) if c.asset_id == ids[0]]
    assert mixed and all(not (c.src_in < 13.0 and c.src_out > 6.0) for c in mixed)       # nothing from the 6-13 s bad stretch
    assert any(c.asset_id == ids[1] for c in v1_clips(plan))
    assert any("丟棄" in n for n in plan.notes)
    kept = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given", drop_bad=False))
    assert not [d for d in kept.decisions if d["kind"] == "drop"]
    assert any(c.src_in < 13.0 and c.src_out > 6.0 for c in v1_clips(kept) if c.asset_id == ids[0])
    no_errors(proj, plan)


@needs_ffmpeg
def test_all_black_source_is_dropped_entirely_and_all_bad_raises(footage):
    proj, ids = make_project(footage, ["black.mp4", "land_a.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0))
    assert {proj.assets[c.asset_id].name for c in v1_clips(plan)} == {"land_a.mp4"}
    src = {s["name"]: s for s in plan.sources}
    assert src["black.mp4"]["usable_seconds"] == 0 and src["black.mp4"]["used_seconds"] == 0
    only_bad, ids2 = make_project(footage, ["black.mp4"])
    with pytest.raises(AutoEditError, match="不能用"):
        propose_assemble(only_bad, ids2, AssembleParams())


@needs_ffmpeg
def test_portrait_source_gets_blur_and_format_follows_the_majority(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4", "port.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given"))
    seq = plan.sequence_objects()[0]
    assert (seq.width, seq.height, seq.fps) == (1920, 1080, 30.0)
    by = {proj.assets[c.asset_id].name: c.transform.fit for c in seq.video_tracks[0].clips}
    assert by["port.mp4"] == "blur" and by["land_a.mp4"] == "cover" and by["land_b.mp4"] == "cover"
    assert any(d["kind"] == "format" and "port.mp4" in d["summary"] for d in plan.decisions)
    contain = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given", fit="contain"))
    assert {proj.assets[c.asset_id].name: c.transform.fit for c in v1_clips(contain)}["port.mp4"] == "contain"
    cover = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given", fit="cover"))
    pc = [c for c in v1_clips(cover) if proj.assets[c.asset_id].name == "port.mp4"][0]
    assert pc.transform.fit == "cover" and 0.0 <= pc.transform.cx <= 1.0
    no_errors(proj, cover)
    vertical = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given", aspect="9:16"))
    vs = vertical.sequence_objects()[0]
    assert (vs.width, vs.height) == (1080, 1920)
    vfit = {proj.assets[c.asset_id].name: c.transform.fit for c in vs.video_tracks[0].clips}
    assert vfit["port.mp4"] == "cover" and vfit["land_a.mp4"] == "blur"
    assert vs.caption_style.margin == 0.20                                  # portrait: captions clear of the platform UI


@needs_ffmpeg
def test_crossfades_are_legal_and_at_most_two_layers(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4", "port.mp4", "mixed.mp4", "still.png"])
    for xf in (0.4, 2.5):
        plan = propose_assemble(proj, ids, AssembleParams(target_seconds=22, transition="crossfade", crossfade_seconds=xf))
        seq = plan.sequence_objects()[0]
        clips = seq.video_tracks[0].clips
        assert len(clips) >= 5
        no_errors(proj, plan)
        for prev, nxt in zip(clips, clips[1:]):
            ov = prev.end - nxt.start
            assert -1e-6 <= ov <= xf + 1e-6 and ov <= 0.5 * min(prev.duration, nxt.duration) + 1e-6
        assert any(prev.end - nxt.start > 0.1 for prev, nxt in zip(clips, clips[1:]))          # crossfades really happen
        for k in range(int(seq.duration * 30)):
            t = k / 30 + 1e-4
            assert sum(1 for c in clips if c.start - 1e-9 <= t < c.end - 1e-9) <= 2
    cut = propose_assemble(proj, ids, AssembleParams(target_seconds=22, transition="cut"))
    cc = v1_clips(cut)
    assert all(abs(p.end - n.start) < 1e-5 for p, n in zip(cc, cc[1:]))                     # hard cuts: no overlap, no gap (6-digit rounding)
    no_errors(proj, cut)


@needs_ffmpeg
def test_target_length_is_honoured(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4", "port.mp4", "mixed.mp4"])
    for target in (8, 12, 20):
        plan = propose_assemble(proj, ids, AssembleParams(target_seconds=target))
        dur = plan.sequence_objects()[0].duration
        assert abs(dur - target) / target <= 0.1, (target, dur)
        used = {c.asset_id for c in v1_clips(plan)}
        assert used == set(ids)                                              # every source keeps at least one shot
        assert not [f for f in audit_plan(plan, proj).findings if f.rule == "assemble.target"]
    everything = propose_assemble(proj, ids, AssembleParams(target_seconds=0))
    short_of = propose_assemble(proj, ids, AssembleParams(target_seconds=300))
    assert short_of.sequence_objects()[0].duration == pytest.approx(everything.sequence_objects()[0].duration, abs=0.2)
    assert short_of.sequence_objects()[0].duration < 300 and any("可用內容只有約" in n for n in short_of.notes)
    clips = v1_clips(short_of)
    assert len(clips) == len(v1_clips(everything))                           # pieces of one long take are merged back together,
    assert all(abs(n.src_in - p.src_out) > 0.1 for p, n in zip(clips, clips[1:]) if p.asset_id == n.asset_id)   # not crossfaded
    f = [f for f in audit_plan(short_of, proj).findings if f.rule == "assemble.target"]
    assert len(f) == 1 and f[0].severity == "info"                           # "sources too short", not a warning
    assert short_of.brief["usable_seconds"] > 0


@needs_ffmpeg
def test_order_by_name_as_given_and_chronological(footage):
    proj, ids = make_project(footage, ["clip10.mp4", "clip2.mp4", "clip1.mp4"])
    base = dict(target_seconds=0, transition="cut")
    assert asset_names(propose_assemble(proj, ids, AssembleParams(order="name", **base)), proj) == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
    assert asset_names(propose_assemble(proj, ids, AssembleParams(order="as_given", **base)), proj) == ["clip10.mp4", "clip2.mp4", "clip1.mp4"]
    for a, stamp in zip(proj.assets.values(), ("2026-03-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01-01T00:00:00Z")):
        a.created = stamp                                                    # container creation_time wins over file mtime
    chrono = propose_assemble(proj, ids, AssembleParams(order="chronological", **base))
    assert asset_names(chrono, proj) == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
    od = [d for d in chrono.decisions if d["kind"] == "order"]
    assert len(od) == 1 and [e["name"] for e in od[0]["evidence"]["order"]] == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
    for a, mtime in zip(proj.assets.values(), (1_000_300, 1_000_200, 1_000_100)):
        a.created = ""
        os.utime(a.path, (mtime, mtime))                                     # no creation_time -> file modification time
    by_mtime = propose_assemble(proj, ids, AssembleParams(order="chronological", **base))
    assert asset_names(by_mtime, proj) == ["clip1.mp4", "clip2.mp4", "clip10.mp4"]
    assert od[0]["evidence"]["order"][0]["time_source"] == "拍攝時間"


@needs_ffmpeg
def test_one_marker_per_source_at_its_first_clip(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4", "black.mp4", "still.png"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given"))
    seq = plan.sequence_objects()[0]
    used = []
    for c in seq.video_tracks[0].clips:
        if c.asset_id not in [u[0] for u in used]:
            used.append((c.asset_id, c.start))
    assert [m.name for m in seq.markers] == [proj.assets[a].name for a, _ in used] == ["land_a.mp4", "land_b.mp4", "still.png"]
    assert [round(m.time, 3) for m in seq.markers] == [round(t, 3) for _, t in used]     # black.mp4 has no clip -> no marker
    assert len({m.id for m in seq.markers}) == 3


@needs_ffmpeg
def test_still_image_is_a_three_second_silent_clip(footage):
    proj, ids = make_project(footage, ["still.png", "land_a.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="as_given"))
    img = v1_clips(plan)[0]
    assert proj.assets[img.asset_id].kind == "image" and img.muted and img.src_in == 0.0
    assert img.duration == pytest.approx(3.0, abs=1 / 30)
    only = propose_assemble(proj, ids[:1], AssembleParams(target_seconds=0))
    assert len(v1_clips(only)) == 1 and only.sequence_objects()[0].duration == pytest.approx(3.0, abs=1 / 30)
    no_errors(proj, plan)


@needs_ffmpeg
def test_audio_files_are_ignored_and_noted(footage):
    proj, ids = make_project(footage, ["music.wav", "land_a.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0))
    assert {proj.assets[c.asset_id].name for c in v1_clips(plan)} == {"land_a.mp4"}
    assert any("純音訊" in n and "music.wav" in n for n in plan.notes)
    with pytest.raises(AutoEditError, match="沒有可以組裝"):
        propose_assemble(proj, ids[:1], AssembleParams())


@needs_ffmpeg
def test_plan_hash_is_reproducible_and_plain_data(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "mixed.mp4", "port.mp4", "still.png"])
    params = AssembleParams(target_seconds=18)
    before = proj.content_json()
    a = propose_assemble(proj, ids, params)
    b = propose_assemble(proj, ids, params)                                  # second run reads the evidence cache
    assert a.sha256() == b.sha256()
    assert proj.content_json() == before                                      # planning never touches the project
    assert propose_assemble(proj, ids, AssembleParams(target_seconds=12)).sha256() != a.sha256()
    again = EditPlan.from_dict(json.loads(json.dumps(a.to_dict())))
    assert again.sha256() == a.sha256()
    ids_in_plan = [c.id for c in v1_clips(a)] + [t["id"] for t in a.sequences[0]["video_tracks"] + a.sequences[0]["audio_tracks"]]
    assert all(i.startswith("p") for i in ids_in_plan) and len(set(ids_in_plan)) == len(ids_in_plan)
    assert a.sequences[0]["id"] == "ps0001"
    assert a.kind == "assemble" and len(a.sequences) == 1 and a.brief["target_seconds"] == 18
    assert {s["asset_id"] for s in a.sources} == set(ids) and all(s["evidence_sha256"] for s in a.sources)
    for d in a.decisions:
        assert d["id"] and d["kind"] and d["summary"] and d["why"]
    assert {d["kind"] for d in a.decisions} >= {"order", "format", "shot", "drop"}
    shots = [d for d in a.decisions if d["kind"] == "shot"]
    assert len(shots) == len(v1_clips(a)) and all(d["source"]["asset_id"] in ids for d in shots)
    clip_ids = {c.id for c in v1_clips(a)}
    assert {d["clip"] for d in shots} == clip_ids


@needs_ffmpeg
def test_all_times_are_on_the_frame_grid(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4", "mixed.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=15))
    fps = plan.sequence_objects()[0].fps
    for c in v1_clips(plan):
        assert abs(c.start * fps - round(c.start * fps)) < 1e-3
        assert abs(c.duration * fps - round(c.duration * fps)) < 1e-3


@needs_ffmpeg
def test_cancel_before_and_during_planning(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4", "mixed.mp4"])
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        propose_assemble(proj, ids, AssembleParams(), cancel=ev)
    mid = threading.Event()
    seen = []

    def progress(f: float, msg: str) -> None:
        seen.append(f)
        if len(seen) == 3:
            mid.set()
    with pytest.raises(Cancelled):
        propose_assemble(proj, ids, AssembleParams(target_seconds=20), progress=progress, cancel=mid)
    assert seen == sorted(seen) and seen[-1] < 1.0                            # monotone progress, never reached "done"
    assert not _children()                                                    # no ffmpeg child left behind


def _children() -> list[int]:
    kids: list[int] = []
    try:
        for tid in os.listdir("/proc/self/task"):
            with open(f"/proc/self/task/{tid}/children") as fh:
                kids += [int(x) for x in fh.read().split()]
    except OSError:
        return []
    return kids


@needs_ffmpeg
def test_progress_is_monotone_and_reaches_one(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "port.mp4"])
    seen: list[tuple[float, str]] = []
    propose_assemble(proj, ids, AssembleParams(target_seconds=10), progress=lambda f, m: seen.append((f, m)))
    fr = [f for f, _ in seen]
    assert fr == sorted(fr) and fr[0] == 0.0 and fr[-1] == 1.0
    assert all(isinstance(m, str) and m for _, m in seen)


@needs_ffmpeg
def test_music_fills_the_sequence_and_fades_out(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4"])
    music = str(footage / "music.wav")
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=12, music_path=music))
    seq = plan.sequence_objects()[0]
    assert len(plan.assets) == 1 and plan.assets[0]["id"].startswith("p") and plan.assets[0]["kind"] == "audio"
    mus = seq.audio_tracks[0].clips
    assert len(mus) >= 3                                                      # 5 s of music repeated to fill ~12 s
    assert mus[0].start == 0.0 and all(abs(a.end - b.start) < 1e-5 for a, b in zip(mus, mus[1:]))
    assert abs(mus[-1].end - seq.duration) < 0.3 and all(c.src_out <= 5.0 + 1e-6 for c in mus)
    assert mus[-1].fade_out == pytest.approx(min(2.0, mus[-1].duration), abs=1e-5) and mus[0].fade_out == 0.0
    assert all(c.volume == pytest.approx(10 ** (-18 / 20), abs=1e-5) for c in mus)
    assert any("動態閃避" in n for n in plan.notes) and any(d["kind"] == "music" for d in plan.decisions)
    no_errors(proj, plan)
    assert plan.sha256() == propose_assemble(proj, ids, AssembleParams(target_seconds=12, music_path=music)).sha256()
    from autocut.media.probe import probe_asset
    known = probe_asset(footage / "music.wav")
    proj.assets[known.id] = known                                             # the project already has this file
    reuse = propose_assemble(proj, ids, AssembleParams(target_seconds=12, music_path=music))
    assert reuse.assets == [] and reuse.sequence_objects()[0].audio_tracks[0].clips[0].asset_id == known.id
    missing = propose_assemble(proj, ids, AssembleParams(target_seconds=12, music_path=str(footage / "nope.wav")))
    assert not missing.sequence_objects()[0].audio_tracks[0].clips and any("找不到配樂" in n for n in missing.notes)
    muted = propose_assemble(proj, ids, AssembleParams(target_seconds=12, keep_audio=False))
    assert all(c.muted for c in v1_clips(muted))


@needs_ffmpeg
def test_caption_file_is_mapped_through_the_cuts(footage, tmp_path):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4"])
    srt = tmp_path / "land_a.srt"                                             # the stem names the source it belongs to
    srt.write_text("1\n00:00:00,500 --> 00:00:02,500\n第一句字幕\n\n2\n00:00:04,000 --> 00:00:06,000\n第二句字幕在中間\n\n"
                   "3\n00:00:06,500 --> 00:00:07,600\n最後一句\n", encoding="utf-8")
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0, order="name", transition="cut", captions="file",
                                                      caption_file=str(srt)))
    seq = plan.sequence_objects()[0]
    assert [q.text for q in seq.cues] == ["第一句字幕", "第二句字幕在中間", "最後一句"]
    first = [c for c in seq.video_tracks[0].clips if c.asset_id == ids[0]][0]
    assert abs(seq.cues[0].start - (first.start + 0.5 - first.src_in)) < 0.05          # source time -> timeline time
    assert all(q.end <= seq.duration + 1e-6 and q.end > q.start for q in seq.cues)
    assert all(a.end <= b.start + 1e-9 for a, b in zip(seq.cues, seq.cues[1:]))
    assert any(d["kind"] == "caption" for d in plan.decisions)
    other = propose_assemble(proj, ids, AssembleParams(target_seconds=0, captions="file", caption_file=str(tmp_path / "x.srt")))
    assert other.sequence_objects()[0].cues == [] and any("字幕" in n for n in other.notes)       # unreadable file: honest note
    plain = propose_assemble(proj, ids, AssembleParams(target_seconds=0, captions="off"))
    assert plain.sequence_objects()[0].cues == []


@needs_ffmpeg
def test_auto_captions_without_speech_recognition_degrade_honestly(footage):
    from autocut.analysis.speech import speech_available
    if speech_available()[0]:
        pytest.skip("speech recognition is installed here")
    proj, ids = make_project(footage, ["land_a.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0, captions="auto"))
    assert plan.sequence_objects()[0].cues == []
    assert any("語音辨識" in n for n in plan.notes)


@needs_ffmpeg
def test_audit_apply_undo_end_to_end(footage):
    proj, ids = make_project(footage, ["land_a.mp4", "land_b.mp4", "port.mp4", "mixed.mp4", "still.png"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=20, music_path=str(footage / "music.wav")))
    receipt = audit_plan(plan, proj)
    assert receipt.accepted and not receipt.errors, [f.to_dict() for f in receipt.findings]
    assert not [f for f in receipt.findings if f.rule in ("source.missing", "source.file", "source.changed", "seq.invalid")]
    scratch, sid = preview_project(proj, plan, 0)                             # preview never touches the real project
    assert sid in [s.id for s in scratch.sequences] and not [i for i in validate_sequence(scratch, scratch.sequence(sid)) if i.level == "error"]
    before = proj.content_json()
    ed = Editor(proj)
    new_ids = apply_plan(ed, plan, receipt)
    seq = proj.sequence(new_ids[0])
    assert (seq.width, seq.height) == (1920, 1080) and proj.active == new_ids[0]
    assert any(a.kind == "audio" and a.path.endswith("music.wav") for a in proj.assets.values())
    assert not [i for i in validate_sequence(proj, seq) if i.level == "error"]
    run = proj.meta["ai_runs"][-1]
    assert run["kind"] == "assemble" and run["human_review"] == "pending" and seq.meta["ai"]["run_id"] == run["run_id"]
    assert seq.meta["ai"]["source"]["asset_ids"]                                # the footage that went into it
    ed.undo()
    assert proj.content_json() == before                                      # ONE undo removes everything
    ed.redo()
    assert proj.sequence(new_ids[0]).meta["ai"]["run_id"] == run["run_id"]


@needs_ffmpeg
def test_swapping_a_source_file_after_planning_blocks_the_audit(footage, tmp_path):
    work = tmp_path / "swap"
    work.mkdir()
    shutil.copy(footage / "land_a.mp4", work / "a.mp4")
    proj, ids = make_project(work, ["a.mp4"])
    plan = propose_assemble(proj, ids, AssembleParams(target_seconds=0))
    assert audit_plan(plan, proj).accepted
    shutil.copy(footage / "land_b.mp4", work / "a.mp4")                      # different content, same name
    r = audit_plan(plan, proj)
    assert not r.accepted and [f.rule for f in r.errors] == ["source.changed"]
