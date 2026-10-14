"""Shorts planner: scoring / window search, virtual camera, captions, and the plan itself (fake evidence + ffmpeg media)."""
from __future__ import annotations

import math
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest

from autocut.analysis.evidence import Evidence
from autocut.autoedit import captions as cap
from autocut.autoedit import reframe as rf
from autocut.autoedit import scoring as sc
from autocut.autoedit.api import AutoEditError, Cancelled, propose_shorts
from autocut.autoedit.config import AuditRules, ShortsParams
from autocut.autoedit.shorts import plan_shorts
from autocut.core.geometry import base_window, interp_keys
from autocut.core.project import Asset, Project, Sequence, validate_sequence
from autocut.core.timecode import frame_of
from autocut.util import IdGen

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
AFPS, VFPS = 20.0, 5.0


# ================================================================================================ helpers
def fake_ev(dur: float = 120.0, burst: float = 4.0, gap: float = 1.0, loud=(40.0, 80.0), faces=(40.0, 80.0),
            scenes=(), speech=False, audio: bool = True, words: bool = False) -> Evidence:
    """Tone bursts of ``burst`` s with ``gap`` s of silence; louder + faces inside ``loud`` / ``faces``."""
    ev = Evidence.empty("a1", dur, width=1920, height=1080, has_audio=audio, audio_fps=AFPS, visual_fps=VFPS, fps=30.0)
    na, nv = int(dur * AFPS), int(dur * VFPS)
    segs = []
    if audio:
        voice, rms, sil = np.zeros(na), np.full(na, -60.0), []
        t = 0.0
        while t + burst <= dur + 1e-9:
            voice[int(round(t * AFPS)):int(round((t + burst) * AFPS))] = 1.0
            rms[int(round(t * AFPS)):int(round((t + burst) * AFPS))] = -25.0 if loud[0] <= t < loud[1] else -32.0
            if t + burst + gap <= dur + 1e-9:
                sil.append([t + burst, t + burst + gap])
            seg = {"start": t, "end": t + burst, "text": "大家好，今天我們來聊聊重點，第一個是食材。"}
            if words:
                seg["words"] = [{"start": t + 0.5 * i, "end": t + 0.5 * i + 0.45, "word": w}
                                for i, w in enumerate(["大家好", "今天", "我們", "來聊聊", "重點", "第一個", "是", "食材"])]
            segs.append(seg)
            t += burst + gap
        ev.voice, ev.rms_db, ev.silences = voice.tolist(), rms.tolist(), sil
        if speech:
            ev.speech = segs
    ev.motion = [0.05] * nv
    ev.sharp = [100.0] * nv
    ev.bright = [0.5] * nv
    ev.faces = [[[0.3 + 0.2 * i / nv, 0.3, 0.1, 0.2]] if faces[0] <= i / VFPS < faces[1] else [] for i in range(nv)]
    ev.face_sample_mask = [1] * nv
    ev.motion_cx, ev.motion_cy = [-1.0] * nv, [-1.0] * nv
    ev.scenes = list(scenes)
    return ev


def make_pool(ev, **kw):
    p = ShortsParams(**{"platform": "ig_reels", "target_seconds": 30, "count": 3, **kw})
    band = sc.plan_band(p)
    feat = sc.Feat(ev, ev.speech)
    return p, band, feat, sc.search_windows(feat, p, band)


@pytest.fixture
def source_file(tmp_path):
    f = tmp_path / "long.mp4"
    f.write_bytes(b"x")
    return str(f)


def fake_proj(path: str, dur: float = 120.0) -> Project:
    proj = Project.new("t")
    proj.assets["a1"] = Asset(id="a1", path=path, name="long.mp4", kind="video", duration=dur, width=1920, height=1080,
                              fps=30.0, has_video=True, has_audio=True, audio_channels=2, sample_rate=48000,
                              fingerprint="fp1")
    return proj


@pytest.fixture
def patched_analysis(monkeypatch):
    """``analyze_asset`` returns the evidence set by ``patched_analysis.ev`` (no media files needed)."""
    class Holder:
        ev = None
    h = Holder()
    monkeypatch.setattr("autocut.analysis.pipeline.analyze_asset", lambda asset, params=None, **kw: h.ev)
    return h


# =============================================================================================== band / dead zone
def test_band_clamp_and_deadzone():
    b = sc.plan_band(ShortsParams(platform="ig_reels", target_seconds=35))
    assert (b.lo, b.hi, b.target, b.deadzone) == (13.0, 60.0, 35.0, None) and not b.notes
    b = sc.plan_band(ShortsParams(platform="ig_reels", target_seconds=100))
    assert b.target == 60.0 and "超出" in b.notes[0]
    y = sc.plan_band(ShortsParams(platform="yt_shorts", target_seconds=35))
    assert y.deadzone == (25.001, 44.999)
    assert y.target in (25.0, 45.0) and any("死區" in n for n in y.notes)
    assert not y.allowed(np.array([30.0]))[0] and y.allowed(np.array([24.0, 46.0])).all()
    y2 = sc.plan_band(ShortsParams(platform="yt_shorts", target_seconds=20, min_seconds=26, max_seconds=40))
    assert y2.deadzone is None and any("涵蓋整個長度帶" in n for n in y2.notes)       # nothing allowed: dead zone ignored


# ================================================================================================ boundaries
def test_boundaries_kinds_weights_and_trimmed_edges():
    ev = fake_ev(60.0, speech=True, scenes=[22.0])
    feat = sc.Feat(ev, ev.speech)
    bounds = sc.find_boundaries(feat)
    kinds = {b.kind for b in bounds}
    assert {"sentence", "silence", "scene", "edge"} <= kinds
    assert {b.kind: b.weight for b in bounds}["sentence"] == 1.0 and sc.KIND_WEIGHT["silence"] == 0.7
    assert sc.KIND_WEIGHT["scene"] == 0.5 and sc.KIND_WEIGHT["grid"] == 0.1
    sent = next(b for b in bounds if b.kind == "sentence" and abs(b.t - 4.0) < 0.1)
    # a short that ends at the 4 s sentence stops 0.3 s after the speech, one that starts after it begins 0.2 s before the next burst
    assert sent.t_out == pytest.approx(4.3, abs=0.06) and sent.t_in == pytest.approx(4.8, abs=0.06)
    # without a transcript the middle of the silence is the (weaker) boundary, trimmed the same way
    no_text = sc.find_boundaries(sc.Feat(fake_ev(60.0), None))
    sil = next(b for b in no_text if b.kind == "silence" and abs(b.t - 4.5) < 0.1)
    assert sil.weight == 0.7 and sil.t_out == pytest.approx(4.3, abs=0.06) and sil.t_in == pytest.approx(4.8, abs=0.06)
    assert not any(b.kind == "grid" for b in bounds)                       # the 1 s grid is only a last resort
    # with nothing at all the 1 s grid appears
    empty = Evidence.empty("a", 30.0, visual_fps=VFPS)
    grid = sc.find_boundaries(sc.Feat(empty, None))
    assert sum(1 for b in grid if b.kind == "grid") >= 20 and grid[0].kind == "edge"


# ======================================================================================== window search
def test_windows_do_not_overlap_respect_gap_and_band():
    ev = fake_ev(120.0)
    p, band, feat, pool = make_pool(ev, count=3)
    assert len(pool) > 100
    sel = sc.select_windows(pool, 3, p.min_gap_between)
    assert len(sel) == 3
    spans = sorted((float(pool.ts[k]), float(pool.te[k])) for k in sel)
    for (a0, b0), (a1, b1) in zip(spans, spans[1:]):
        assert a1 >= b0 + p.min_gap_between - 1e-6               # no overlap, at least min_gap apart
    for k in sel:
        assert band.lo <= pool.tight[k] <= band.hi                # tightened length inside the band
    assert float(pool.score[sel[0]]) >= float(pool.score[sel[1]]) >= float(pool.score[sel[2]])
    # the best one sits in the loud, face-rich stretch
    assert 40.0 <= pool.ts[sel[0]] and pool.te[sel[0]] <= 85.0


def test_tightened_length_matches_exact_cut_list():
    ev = fake_ev(120.0)
    p, band, feat, pool = make_pool(ev)
    for k in sc.select_windows(pool, 3, 8.0):
        rem = sc.silence_removals(ev.silences, float(pool.ts[k]), float(pool.te[k]), p.pause_min, p.pause_keep)
        kept = sc.tighten_ranges(float(pool.ts[k]), float(pool.te[k]), rem, 0.25)
        exact = sum(b - a for a, b in kept)
        assert exact == pytest.approx(float(pool.tight[k]), abs=0.02)
        assert float(pool.saved[k]) > 2.0                            # 1 s pauses -> 0.65 s saved each
    p2, _b, _f, pool2 = make_pool(ev, tighten=False)
    assert float(pool2.saved.max()) == 0.0


def test_tighten_ranges_never_makes_pieces_shorter_than_min_clip():
    kept = sc.tighten_ranges(0.0, 10.0, [(0.1, 0.5), (3.0, 3.5), (9.9, 9.95)], 0.25)
    assert kept == [(0.0, 3.0), (3.5, 10.0)]
    assert all(b - a >= 0.25 for a, b in kept)
    rem = sc.silence_removals([[2.0, 3.0], [5.0, 5.5], [7.0, 7.8]], 0.0, 10.0, 0.7, 0.35)
    assert rem == [(2.175, 2.825), (7.175, 7.625)]                    # 0.5 s pause < pause_min is left alone
    words = [(2.4, 2.6)]                                              # a word inside the pause: the cut avoids it
    rem2 = sc.silence_removals([[2.0, 3.0]], 0.0, 10.0, 0.7, 0.35, words)
    assert len(rem2) == 1 and not (rem2[0][0] < 2.6 and rem2[0][1] > 2.4)


def test_yt_shorts_deadzone_excluded_from_windows():
    ev = fake_ev(200.0, loud=(0, 0), faces=(0, 0))
    p = ShortsParams(platform="yt_shorts", target_seconds=35, count=4, min_gap_between=2.0)
    band = sc.plan_band(p)
    pool = sc.search_windows(sc.Feat(ev, None), p, band)
    assert len(pool) > 50
    assert not ((pool.tight > 25.0) & (pool.tight < 45.0)).any()
    plan_lengths = [float(pool.tight[k]) for k in sc.select_windows(pool, 4, 2.0)]
    assert plan_lengths and all(L <= 25.0 or L >= 45.0 for L in plan_lengths)


def test_sentence_end_is_preferred_over_grid_and_mid_word_cuts():
    # continuous speech without a single pause: only the transcript knows where sentences end (every 7 s)
    dur = 70.0
    ev = Evidence.empty("a1", dur, width=1920, height=1080, has_audio=True, audio_fps=AFPS, visual_fps=VFPS, fps=30.0)
    ev.voice, ev.rms_db = [1.0] * int(dur * AFPS), [-30.0] * int(dur * AFPS)
    ev.speech = [{"start": 7.0 * i, "end": 7.0 * (i + 1), "text": f"第{i}句話說完了。",
                  "words": [{"start": 7.0 * i + 0.7 * j, "end": 7.0 * i + 0.7 * j + 0.65, "word": w}
                            for j, w in enumerate(["第", f"{i}", "句話", "說完了"])]} for i in range(10)]
    p, band, feat, pool = make_pool(ev, count=1, target_seconds=21)
    best = int(np.argmax(pool.score))
    info = pool.info(best)
    assert info["end_kind"] == "sentence" and info["start_kind"] in ("sentence", "edge")
    assert info["parts"]["complete"] >= 0.85 and not info["mid_word"]
    # grid / scene cuts that fall inside a word score 0 for "complete"
    ev.scenes = [8.0, 12.0]                                             # 8.0 falls inside the word "1" (7.7-8.35), 12.0 sits between words
    bounds = sc.find_boundaries(sc.Feat(ev, ev.speech))
    scenes = {round(b.t, 1): b for b in bounds if b.kind == "scene"}
    assert scenes[8.0].mid_in and scenes[8.0].mid_out
    assert not scenes[12.0].mid_in and not scenes[12.0].mid_out
    assert sc.inside_word(1.5, [(1.0, 2.0)]) and not sc.inside_word(1.01, [(1.0, 2.0)]) and not sc.inside_word(2.5, [(1.0, 2.0)])


def test_hook_and_missing_components_are_reweighted():
    ev = fake_ev(120.0)
    _p, _b, _f, pool = make_pool(ev)
    hook, ts = pool.parts["hook"], pool.ts
    quiet, loud = (ts >= 1.0) & (ts < 36.0), (ts >= 41.0) & (ts < 76.0)
    assert quiet.any() and loud.any() and hook[loud].min() > hook[quiet].max()   # loud + face at the start beats the quiet part
    assert pool.available["content"] is False and "content" not in pool.weights and "content_hook" not in pool.weights
    assert sum(pool.weights.values()) == pytest.approx(1.0)
    # without text the remaining weights keep their old relative sizes (voice : hook : complete = 0.22 : 0.18 : 0.18)
    assert pool.weights["voice"] / pool.weights["hook"] == pytest.approx(0.22 / 0.18, rel=0.01)
    # no audio track: voice falls back to motion, energy / hook-onset are not used
    ev2 = fake_ev(120.0, audio=False)
    _p2, _b2, _f2, pool2 = make_pool(ev2)
    assert pool2.available["voice"] and not pool2.available["energy"] and len(pool2) > 0
    # content counts (and dominates) when a transcript exists
    ev3 = fake_ev(120.0, speech=True)
    _p3, _b3, _f3, pool3 = make_pool(ev3)
    assert pool3.available["content"] and float(pool3.stats["kw_hits"].max()) >= 3
    assert pool3.weights["content"] + pool3.weights["content_hook"] > 0.5


def test_select_windows_accept_can_veto():
    ev = fake_ev(120.0)
    p, _band, _feat, pool = make_pool(ev)
    first = sc.select_windows(pool, 1, 8.0)[0]
    again = sc.select_windows(pool, 1, 8.0, accept=lambda k: k != first)[0]
    assert again != first and float(pool.score[again]) <= float(pool.score[first])


# ================================================================================================ reframe
ASSET = Asset(id="a", path="x.mp4", duration=60.0, width=1920, height=1080, fps=30.0)
FW = base_window(1920, 1080, 1080, 1920)[0] / 1920            # window width as a fraction of the picture


def face_ev(dur, xfn, yfn=lambda t: 0.4, scenes=(), vfps=VFPS, size=0.1) -> Evidence:
    n = int(dur * vfps) + 1
    ev = Evidence.empty("a", dur, width=1920, height=1080, visual_fps=vfps)
    ev.motion = [0.1] * n
    ev.motion_cx, ev.motion_cy = [-1.0] * n, [-1.0] * n
    ev.faces = [[[xfn(i / vfps) - size / 2, yfn(i / vfps) - size, size, size * 1.6]] for i in range(n)]
    ev.face_sample_mask = [1] * n
    ev.scenes = list(scenes)
    return ev


def cam(keys, t):
    return interp_keys(keys, t)


def test_reframe_follows_a_face_moving_left_to_right():
    ev = face_ev(10.0, lambda t: 0.2 + 0.06 * t)
    keys, info = rf.reframe_keys(ev, ASSET, 0.0, 10.0, 1080, 1920)
    assert info["mode_used"] == "face" and info["face_ratio"] == 1.0 and info["n_keys"] == len(keys)
    assert keys[0][0] == 0.0 and keys[-1][0] == pytest.approx(10.0, abs=0.04)
    xs = [cam(keys, t / 5.0)[0] for t in range(0, 51)]
    assert all(b >= a - 1e-9 for a, b in zip(xs, xs[1:]))                  # only moves right
    assert xs[-1] - xs[0] > 0.3 and xs[-1] > 0.65                          # really follows
    for t in range(0, 51):                                                # the subject stays inside the crop window
        assert abs(cam(keys, t / 5.0)[0] - (0.2 + 0.06 * t / 5.0)) < FW / 2
    assert all(FW / 2 - 1e-6 <= k[1] <= 1 - FW / 2 + 1e-6 for k in keys)


def test_reframe_dead_zone_keeps_the_camera_still_for_small_movements():
    ev = face_ev(10.0, lambda t: 0.5 + 0.02 * math.sin(3.0 * t))          # +-0.02 < 15% of the window (0.047)
    keys, _info = rf.reframe_keys(ev, ASSET, 0.0, 10.0, 1080, 1920)
    assert len({round(k[1], 3) for k in keys}) == 1 and len(keys) == 2
    ev2 = face_ev(10.0, lambda t: 0.5 + (0.0 if t < 4 else 0.25))         # a real move leaves the dead zone
    keys2, _ = rf.reframe_keys(ev2, ASSET, 0.0, 10.0, 1080, 1920)
    assert cam(keys2, 9.9)[0] > 0.6 and abs(cam(keys2, 1.0)[0] - 0.5) < 0.05


def test_reframe_clamps_into_the_legal_window():
    left = face_ev(6.0, lambda t: 0.02)
    right = face_ev(6.0, lambda t: 0.98)
    kl, _ = rf.reframe_keys(left, ASSET, 0.0, 6.0, 1080, 1920)
    kr, _ = rf.reframe_keys(right, ASSET, 0.0, 6.0, 1080, 1920)
    assert all(k[1] == pytest.approx(FW / 2, abs=1e-3) for k in kl)
    assert all(k[1] == pytest.approx(1 - FW / 2, abs=1e-3) for k in kr)
    assert all(k[2] == 0.5 for k in kl + kr)                               # zoom 1: the window is as tall as the picture


def test_reframe_scene_cut_is_a_jump_not_a_pan():
    ev = face_ev(10.0, lambda t: 0.2 if t < 5.0 else 0.8, scenes=[5.0])
    keys, info = rf.reframe_keys(ev, ASSET, 0.0, 10.0, 1080, 1920, fps=30.0)
    assert info["jumps"] == 1
    pair = [(a, b) for a, b in zip(keys, keys[1:]) if b[0] - a[0] < 0.05]
    assert len(pair) == 1
    (t_a, x_a, _), (t_b, x_b, _) = pair[0]
    assert t_b == pytest.approx(5.0, abs=0.01) and t_b - t_a == pytest.approx(1 / 30, abs=0.004)
    assert x_b - x_a > 0.5                                               # new shot, new framing in one frame
    assert not [k for k in keys if 0.05 < abs(k[0] - 5.0) < 1.5 and x_a + 0.05 < k[1] < x_b - 0.05]   # no pan in between
    keys2, info2 = rf.reframe_keys(ev, ASSET, 0.0, 10.0, 1080, 1920, scenes=[])       # cut list overridden
    assert info2["jumps"] == 0


def test_reframe_key_count_is_capped():
    rng = np.random.default_rng(7)
    xs = rng.uniform(0.05, 0.95, 601)
    ev = face_ev(120.0, lambda t: float(xs[min(int(t * VFPS), 600)]))
    keys, info = rf.reframe_keys(ev, ASSET, 0.0, 120.0, 1080, 1920, strength=1.0)
    assert len(keys) <= rf.MAX_KEYS and keys[0][0] == 0.0 and keys[-1][0] == pytest.approx(120.0, abs=0.04)
    assert [k[0] for k in keys] == sorted(k[0] for k in keys)
    assert len(rf.reframe_keys(ev, ASSET, 0.0, 120.0, 1080, 1920, strength=1.0, max_keys=12)[0]) <= 12


def test_reframe_modes_and_fallbacks():
    ev = face_ev(8.0, lambda t: 0.3 + 0.05 * t)
    k, i = rf.reframe_keys(ev, ASSET, 0.0, 8.0, 1080, 1920, mode="center")
    assert i["mode_used"] == "center" and {x[1] for x in k} == {0.5}
    k, i = rf.reframe_keys(ev, ASSET, 0.0, 8.0, 1080, 1920, mode="blur")                # blur background: nothing to pan
    assert k == [] and i["mode_used"] == "blur" and i["n_keys"] == 0
    k, i = rf.reframe_keys(None, ASSET, 0.0, 8.0, 1080, 1920)
    assert i["mode_used"] == "center" and k[0] == [0.0, 0.5, 0.5]
    # no faces, but the picture changes on the right: motion centroid
    m = Evidence.empty("a", 8.0, width=1920, height=1080, visual_fps=VFPS)
    n = 41
    m.motion, m.motion_cx, m.motion_cy = [0.3] * n, [0.85] * n, [0.5] * n
    m.faces, m.face_sample_mask = [[] for _ in range(n)], [1] * n
    k, i = rf.reframe_keys(m, ASSET, 0.0, 8.0, 1080, 1920, mode="face")
    assert i["mode_used"] == "motion" and i["face_ratio"] == 0.0 and k[-1][1] == pytest.approx(1 - FW / 2, abs=1e-3)
    k, i = rf.reframe_keys(m, ASSET, 0.0, 8.0, 1080, 1920, mode="motion")
    assert i["mode_used"] == "motion"
    # neither faces nor motion information
    z = Evidence.empty("a", 8.0, width=1920, height=1080, visual_fps=VFPS)
    z.motion, z.motion_cx, z.motion_cy = [0.0] * n, [-1.0] * n, [-1.0] * n
    z.faces, z.face_sample_mask = [[] for _ in range(n)], [1] * n
    assert rf.reframe_keys(z, ASSET, 0.0, 8.0, 1080, 1920)[1]["mode_used"] == "center"
    locked, _ = rf.reframe_keys(ev, ASSET, 0.0, 8.0, 1080, 1920, strength=0.0)
    assert len({k[1] for k in locked}) == 1                                # strength 0 = locked camera


def test_reframe_punch_in_puts_the_face_in_the_upper_middle():
    ev = face_ev(4.0, lambda t: 0.5, yfn=lambda t: 0.5)                     # face box centre y = 0.5 - 0.1 + 0.08 = 0.48
    fh = 1 / 1.12
    plain, _ = rf.reframe_keys(ev, ASSET, 0.0, 4.0, 1080, 1920, zoom=1.12)
    punch, _ = rf.reframe_keys(ev, ASSET, 0.0, 4.0, 1080, 1920, zoom=1.12, punch_in_at=0.0)
    assert plain[0][2] == pytest.approx(0.48, abs=1e-3)
    assert punch[0][2] == pytest.approx(1 - fh / 2, abs=1e-3)               # wants 0.48 + 0.107; the legal window stops it
    # with a tighter zoom there is room: the window centre sits 0.12 window-heights below the face
    ev2 = face_ev(4.0, lambda t: 0.5, yfn=lambda t: 0.42)                    # face centre y = 0.40
    q2, _ = rf.reframe_keys(ev2, ASSET, 0.0, 4.0, 1080, 1920, zoom=2.0)
    p2, _ = rf.reframe_keys(ev2, ASSET, 0.0, 4.0, 1080, 1920, zoom=2.0, punch_in_at=0.0)
    assert q2[0][2] == pytest.approx(0.40, abs=1e-3) and p2[0][2] == pytest.approx(0.40 + 0.12 * 0.5, abs=1e-3)
    assert all(0.25 - 1e-9 <= k[2] <= 0.75 + 1e-9 for k in p2)              # always a legal window
    late, _ = rf.reframe_keys(ev2, ASSET, 0.0, 4.0, 1080, 1920, zoom=2.0, punch_in_at=2.0)
    assert late[0][2] == pytest.approx(q2[0][2], abs=1e-9)                   # before the punch-in nothing changes


# ================================================================================================ captions
def width_of(text: str) -> float:
    return cap.display_width(text)


def test_caption_cjk_wrapping_splits_long_text_without_losing_it():
    text = "今天我們來聊聊餐飲業最重要的三個重點第一個是食材的選擇第二個是人員培訓的方法"
    cues = cap.cues_for_pieces([{"start": 0.0, "end": 16.0, "text": text}], [cap.Piece("a", 0.0, 16.0, 0.0)],
                               ids=IdGen("p"))
    assert len(cues) >= 3 and all(width_of(c.text) <= 16 for c in cues)
    assert "".join(c.text for c in cues) == text
    assert all(cues[i].end <= cues[i + 1].start for i in range(len(cues) - 1))
    # punctuation is preferred, and trailing commas / full stops are not displayed
    cues = cap.cues_for_pieces([{"start": 0.0, "end": 9.0, "text": "第一個是食材的選擇，第二個是人員培訓的方法，第三個是成本控制。"}],
                               [cap.Piece("a", 0.0, 9.0, 0.0)])
    assert [c.text for c in cues] == ["第一個是食材的選擇", "第二個是人員培訓的方法", "第三個是成本控制"]
    # Latin words break at spaces, half-width characters count half
    lines = cap.split_text("iPhone 15 Pro Max is really good for filming 影片 and photos 還有很多功能", 16)
    assert all(width_of(x) <= 16 for x in lines) and all(not x.startswith(" ") for x in lines) and len(lines) >= 3
    assert " ".join(lines) == "iPhone 15 Pro Max is really good for filming 影片 and photos 還有很多功能"   # every break is a space: no Latin word is cut


def test_caption_mapping_through_pieces_drops_removed_and_clips_crossing():
    segs = [{"start": 1.0, "end": 3.0, "text": "第一句話"}, {"start": 5.0, "end": 7.0, "text": "被剪掉的句子"},
            {"start": 9.0, "end": 11.0, "text": "第三句話"}]
    pieces = [cap.Piece("a", 0.0, 4.0, 0.0), cap.Piece("a", 8.0, 12.0, 4.0)]
    cues = cap.cues_for_pieces(segs, pieces)
    assert [c.text for c in cues] == ["第一句話", "第三句話"]
    assert cues[0].start == pytest.approx(1.0) and cues[1].start == pytest.approx(5.0)       # 9.0 - 8.0 + 4.0
    # a segment that crosses a cut is clipped: only the characters inside a piece survive
    cross = cap.cues_for_pieces([{"start": 3.0, "end": 6.0, "text": "一二三四五六"}],
                                [cap.Piece("a", 0.0, 4.5, 0.0), cap.Piece("a", 5.5, 9.0, 4.5)])
    assert [c.text for c in cross] == ["一二三", "六"]
    assert cross[0].start == pytest.approx(3.0) and cross[1].start == pytest.approx(4.5)
    assert cross[0].end <= cross[1].start - cap.CAP_GAP + 1e-6
    # speed != 1: the timeline is shorter than the source
    fast = cap.cues_for_pieces([{"start": 2.0, "end": 4.0, "text": "加速的字幕"}], [cap.Piece("a", 0.0, 8.0, 0.0, speed=2.0)])
    assert fast[0].start == pytest.approx(1.0)
    # pieces that continue each other (a punch-in split) are not a cut: same cues as one piece
    one = cap.cues_for_pieces([{"start": 1.0, "end": 5.0, "text": "這是一句橫跨接縫的話"}], [cap.Piece("a", 0.0, 6.0, 0.0)])
    two = cap.cues_for_pieces([{"start": 1.0, "end": 5.0, "text": "這是一句橫跨接縫的話"}],
                              [cap.Piece("a", 0.0, 3.0, 0.0), cap.Piece("a", 3.0, 6.0, 3.0)])
    assert [(c.text, c.start, c.end) for c in one] == [(c.text, c.start, c.end) for c in two]
    assert cap.cues_for_pieces([], pieces) == [] and cap.cues_for_pieces(segs, []) == []


def test_caption_timing_rules_pad_gap_min_length_and_reading_speed():
    segs = [{"start": 0.0, "end": 1.0, "text": "一二三四五六七八九十"},          # 10 chars in 1 s: too fast for 5 cps
            {"start": 3.0, "end": 4.0, "text": "後面"},
            {"start": 10.0, "end": 10.1, "text": "好"},                      # right before ...
            {"start": 10.3, "end": 12.0, "text": "我們繼續"},
            {"start": 14.0, "end": 14.1, "text": "嗯"}]                      # alone: padded up to the minimum length
    cues = cap.cues_for_pieces(segs, [cap.Piece("a", 0.0, 20.0, 0.0)], rules=AuditRules())
    by = {c.text: c for c in cues}
    assert by["一二三四五六七八九十"].end == pytest.approx(2.0, abs=1e-3)       # stretched to 10 / 5 cps, still before "後面"
    assert by["後面"].end == pytest.approx(4.15, abs=1e-3)                     # speech end + CAP_PAD
    assert by["好"].end == pytest.approx(10.3 - cap.CAP_GAP, abs=1e-3)        # next start - CAP_GAP wins over the pad
    assert by["嗯"].end - by["嗯"].start == pytest.approx(0.4, abs=1e-3)       # min_cue_seconds
    for a, b in zip(cues, cues[1:]):
        assert b.start - a.end >= cap.CAP_GAP - 1e-6                         # never overlap
    assert all(0.0 <= c.start < c.end <= 20.0 for c in cues)
    for q in cues:
        n = cap._reading_chars(q.text)
        assert n / (q.end - q.start) <= 5.0 + 1e-6 or q.text == "好"          # reading speed kept unless the next cue blocks
    # the last cue never runs past the end of the timeline; frame snapping keeps the order
    tail = cap.cues_for_pieces([{"start": 3.0, "end": 4.0, "text": "結尾"}], [cap.Piece("a", 0.0, 4.0, 0.0)], fps=30.0)
    assert tail[0].end <= 4.0 + 1e-9 and frame_of(tail[0].start, 30.0) / 30.0 == pytest.approx(tail[0].start)


def test_caption_word_timings_and_pauses_decide_the_breaks():
    seg = {"start": 1.0, "end": 4.6, "text": "今天天氣很好",
           "words": [{"start": 1.0, "end": 1.5, "word": "今天"}, {"start": 1.5, "end": 2.0, "word": "天氣"},
                     {"start": 4.0, "end": 4.6, "word": "很好"}]}
    cues = cap.cues_for_pieces([seg], [cap.Piece("a", 0.0, 6.0, 0.0)])
    assert [c.text for c in cues] == ["今天天氣", "很好"]                       # a 2 s pause starts a new cue
    assert cues[0].start == pytest.approx(1.0) and cues[1].start == pytest.approx(4.0)


def test_source_segments_modes(tmp_path, monkeypatch):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,500\n你好世界\n\n2\n00:00:03,000 --> 00:00:04,000\n第二句\n", encoding="utf-8")
    asset = Asset(id="a", path=str(tmp_path / "v.mp4"), kind="video", duration=10.0, has_audio=True)
    notes: list[str] = []
    segs = cap.source_segments(asset, None, "file", str(srt), notes=notes)
    assert [(s["start"], s["end"], s["text"]) for s in segs] == [(1.0, 2.5, "你好世界"), (3.0, 4.0, "第二句")]
    assert cap.source_segments(asset, None, "off") is None
    assert cap.source_segments(asset, None, "file", "", notes=notes) is None and "沒有指定檔案" in notes[-1]
    assert cap.source_segments(asset, None, "file", str(tmp_path / "missing.srt"), notes=notes) is None
    ev = Evidence.empty("a", 10.0)
    ev.speech = [{"start": 0.0, "end": 1.0, "text": "辨識結果"}]
    assert cap.source_segments(asset, ev, "auto")[0]["text"] == "辨識結果"
    # no transcript in the evidence and no recogniser: the reason (繁中) goes to the notes, the result is None
    from autocut.analysis import speech as sp

    def unavailable(*a, **k):
        raise sp.SpeechUnavailable("尚未安裝語音辨識：測試")
    monkeypatch.setattr(sp, "transcribe", unavailable)
    notes2: list[str] = []
    assert cap.source_segments(asset, Evidence.empty("a", 10.0), "auto", notes=notes2) is None
    assert notes2 == ["尚未安裝語音辨識：測試"]
    silent = Asset(id="s", path="x", kind="video", duration=5.0, has_audio=False)
    notes3: list[str] = []
    assert cap.source_segments(silent, None, "auto", notes=notes3) is None and "沒有聲音" in notes3[0]


def _seq_with(proj: Project, clips):
    seq = Sequence.create("s", 1080, 1920, 30.0, seq_id="s1")
    for track, (aid, start, a, b, kw) in clips:
        t = seq.video_tracks[track]
        from autocut.core.project import Clip
        t.clips.append(Clip(id=f"c{len(t.clips)}{track}", asset_id=aid, start=start, src_in=a, src_out=b, **kw))
    return seq


def test_generate_cues_for_sequence_uses_audible_top_layer_clips(source_file, monkeypatch):
    from autocut.analysis import speech as sp
    proj = fake_proj(source_file)
    other = Path(source_file).with_name("other.mp4")
    other.write_bytes(b"x")
    proj.assets["a2"] = Asset(id="a2", path=str(other), name="other.mp4", kind="video", duration=60.0, width=1280, height=720,
                              fps=30.0, has_video=True, has_audio=True)
    # V1: a1[10-20] at 0 (audible), a1[30-40] at 10 (muted); V2: a2[0-4] at 2 covers V1 from 2 to 6
    seq = _seq_with(proj, [(0, ("a1", 0.0, 10.0, 20.0, {})), (0, ("a1", 10.0, 30.0, 40.0, {"muted": True})),
                           (1, ("a2", 2.0, 0.0, 4.0, {}))])
    calls = []
    transcripts = {"a1": [{"start": 11.0, "end": 12.0, "text": "底層說話"}, {"start": 15.0, "end": 16.0, "text": "被蓋住的話"},
                          {"start": 33.0, "end": 34.0, "text": "靜音片段"}],
                   "a2": [{"start": 1.0, "end": 2.0, "text": "上層說話"}]}

    def fake_transcribe(path, *, language, model_size, traditional, allow_download, start, duration, progress, cancel):
        aid = "a1" if Path(path).name == "long.mp4" else "a2"
        calls.append((aid, start, duration))
        return [dict(s) for s in transcripts[aid] if s["end"] > start and s["start"] < start + duration]
    monkeypatch.setattr(sp, "speech_available", lambda: (True, ""))
    monkeypatch.setattr(sp, "transcribe", fake_transcribe)
    cues = cap.generate_cues_for_sequence(proj, seq)
    texts = [c.text for c in cues]
    assert "底層說話" in texts and "上層說話" in texts
    assert "靜音片段" not in texts                                              # muted clip: no sound, no captions
    assert "被蓋住的話" not in texts                                            # V2 is on top at 5-6 s (src 15-16 of a1)
    low = next(c for c in cues if c.text == "底層說話")
    assert low.start == pytest.approx(1.0, abs=0.04)                          # 11.0 - 10.0 + 0
    up = next(c for c in cues if c.text == "上層說話")
    assert up.start == pytest.approx(3.0, abs=0.04)                           # 1.0 - 0.0 + 2
    assert all(0 <= c.start < c.end <= seq.duration + 1e-6 for c in cues)
    # a cancelled run and a missing recogniser
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        cap.generate_cues_for_sequence(proj, seq, cancel=ev)
    monkeypatch.setattr(sp, "speech_available", lambda: (False, "尚未安裝語音辨識：測試"))
    with pytest.raises(sp.SpeechUnavailable):
        cap.generate_cues_for_sequence(proj, seq)


# ===================================================================================== the plan (fake evidence)
def _check_plan(plan, proj, params, asset_dur=120.0):
    assert plan.kind == "shorts" and 1 <= len(plan.sequences) <= params.count
    seen_ids: set[str] = set()
    for sd in plan.sequences:
        seq = Sequence.from_dict(sd)
        assert (seq.width, seq.height, seq.fps) == (params.width, params.height, params.fps)
        assert [t.name for t in seq.video_tracks] == ["V1", "V2"] and [t.name for t in seq.audio_tracks] == ["A1", "A2"]
        v1 = seq.video_tracks[0].clips
        assert v1 and not seq.video_tracks[1].clips and not seq.audio_tracks[1].clips
        pos = 0.0
        for c in v1:
            assert c.start == pytest.approx(pos, abs=1e-6)                    # contiguous, no gaps / crossfades
            assert 0 <= c.src_in < c.src_out <= asset_dur and c.duration >= 0.25 - 1e-9
            assert frame_of(c.start, seq.fps) / seq.fps == pytest.approx(c.start, abs=1e-9)
            pos += c.duration
        assert seq.duration == pytest.approx(pos, abs=1e-6)
        for q in seq.cues:
            assert 0 <= q.start < q.end <= seq.duration + 1e-6 and cap.display_width(q.text) <= 16
        for a, b in zip(seq.cues, seq.cues[1:]):
            assert b.start >= a.end - 1e-9
        for oid in [seq.id] + [t.id for t in seq.tracks()] + [c.id for _t, c in seq.all_clips()] + [q.id for q in seq.cues]:
            assert oid not in seen_ids
            seen_ids.add(oid)
        assert not [i for i in validate_sequence(proj, seq) if i.level == "error"]
        kinds = [d["kind"] for d in plan.decisions if d.get("sequence") == seq.id]
        assert sorted(kinds) == ["caption", "music", "reframe", "short", "tighten"]
    assert plan.decisions and all(d["id"] for d in plan.decisions) and plan.sources[0]["asset_id"] == "a1"


def test_plan_shorts_with_fake_evidence_is_valid_deterministic_and_explained(source_file, patched_analysis):
    patched_analysis.ev = fake_ev(120.0, speech=True, words=True, scenes=[30.0, 90.0])
    proj = fake_proj(source_file)
    params = ShortsParams(count=3, target_seconds=30)
    plan = plan_shorts(proj, "a1", params)
    _check_plan(plan, proj, params)
    again = plan_shorts(proj, "a1", params)
    assert plan.sha256() == again.sha256()                                    # reproducible
    short = [d for d in plan.decisions if d["kind"] == "short"]
    assert len(short) == 3 and [d["score"] for d in short] == sorted((d["score"] for d in short), reverse=True)
    assert all("why" in d and "秒" in d["why"] and d["source"]["asset_id"] == "a1" and "分數" in d["summary"] for d in short)
    names = [Sequence.from_dict(s).name for s in plan.sequences]
    assert names[0].startswith("短影音 1｜") and "–" in names[0]
    # the source ranges of different shorts do not overlap and are >= min_gap apart
    ranges = sorted((d["source"]["t0"], d["source"]["t1"]) for d in short)
    assert all(b0 >= a1 + params.min_gap_between - 0.1 for (_a0, a1), (b0, _b1) in zip(ranges, ranges[1:]))
    assert plan.brief["platform"] == "ig_reels" and plan.brief["weights"]["voice"] == 0.11
    assert plan.brief["transcript"]["source"] == "speech" and plan.brief["weights"]["content"] == 0.42
    # untouched project
    assert proj.revision == 0 and len(proj.sequences) == 1


def test_first_cut_rule_splits_the_first_clip_with_a_punch_in(source_file, patched_analysis):
    patched_analysis.ev = fake_ev(120.0, speech=True, words=True)
    proj = fake_proj(source_file)
    plan = plan_shorts(proj, "a1", ShortsParams(count=1, target_seconds=30))
    seq = Sequence.from_dict(plan.sequences[0])
    c0, c1 = seq.video_tracks[0].clips[:2]
    assert c0.duration <= 1.9 + 1e-6 and c0.duration <= 2.05 and c0.transform.zoom == 1.0
    assert c1.transform.zoom == pytest.approx(1.12) and c1.src_in == pytest.approx(c0.src_out)      # same footage, new framing
    assert c0.fade_out == 0.0 and c1.fade_in == 0.0 and c0.fade_in == 0.015                          # no dip inside a seamless split
    nxt = seq.video_tracks[0].clips[2]
    assert nxt.src_in > c1.src_out and nxt.transform.zoom == 1.0                                    # jump cuts alternate back
    assert all(len(c.transform.keys) >= 2 and c.transform.fit == "cover" for c in seq.video_tracks[0].clips)
    # words are known: the split falls between two words
    words = [(w["start"], w["end"]) for s in patched_analysis.ev.speech for w in s["words"]]
    assert not sc.inside_word(c0.src_out, words, margin=0.03)
    blur = plan_shorts(proj, "a1", ShortsParams(count=1, target_seconds=30, reframe="blur"))
    bseq = Sequence.from_dict(blur.sequences[0])
    assert all(c.transform.fit == "blur" and c.transform.zoom == 1.0 for c in bseq.video_tracks[0].clips)
    assert any("不做 punch-in" in n for n in blur.notes)


def test_plan_without_tighten_keeps_one_run_and_still_obeys_the_two_second_rule(source_file, patched_analysis):
    patched_analysis.ev = fake_ev(120.0, speech=True)
    plan = plan_shorts(fake_proj(source_file), "a1", ShortsParams(count=1, target_seconds=25, tighten=False))
    seq = Sequence.from_dict(plan.sequences[0])
    clips = seq.video_tracks[0].clips
    assert len(clips) == 2 and clips[0].duration <= 1.9 + 1e-6 and clips[1].src_in == pytest.approx(clips[0].src_out)
    assert next(d for d in plan.decisions if d["kind"] == "tighten")["summary"] == "未緊湊化"


def test_plan_yt_shorts_never_lands_in_the_dead_zone(source_file, patched_analysis):
    patched_analysis.ev = fake_ev(300.0, loud=(0, 0), faces=(0, 0))
    proj = fake_proj(source_file, 300.0)
    plan = plan_shorts(proj, "a1", ShortsParams(platform="yt_shorts", target_seconds=35, count=4, min_gap_between=4.0))
    assert any("死區" in n for n in plan.notes)
    lengths = [Sequence.from_dict(s).duration for s in plan.sequences]
    assert lengths and all(L <= 25.0 + 1e-6 or L >= 45.0 - 1e-6 for L in lengths)


def test_plan_short_sources_and_errors(source_file, patched_analysis):
    patched_analysis.ev = fake_ev(10.0, loud=(0, 0), faces=(0, 0))
    plan = plan_shorts(fake_proj(source_file, 10.0), "a1", ShortsParams())
    assert len(plan.sequences) == 1 and any("只產生 1 支" in n for n in plan.notes)          # shorter than the band
    patched_analysis.ev = fake_ev(4.0)
    with pytest.raises(AutoEditError, match="影片太短"):
        plan_shorts(fake_proj(source_file, 4.0), "a1", ShortsParams())
    with pytest.raises(AutoEditError):
        plan_shorts(fake_proj(source_file), "nope", ShortsParams())
    with pytest.raises(AutoEditError, match="找不到素材檔案"):
        plan_shorts(fake_proj(source_file + ".gone"), "a1", ShortsParams())
    with pytest.raises(AutoEditError):
        plan_shorts(fake_proj(source_file), "a1", ShortsParams(reframe="spin"))
    audio = fake_proj(source_file)
    audio.assets["a1"].kind, audio.assets["a1"].has_video = "audio", False
    with pytest.raises(AutoEditError, match="沒有影像"):
        plan_shorts(audio, "a1", ShortsParams())


def test_plan_degrades_honestly_without_speech_audio_or_faces(source_file, patched_analysis):
    ev = fake_ev(120.0, audio=False, faces=(0, 0))
    ev.notes = ["尚未安裝語音辨識：請執行 pip install -r requirements-speech.txt"]
    patched_analysis.ev = ev
    proj = fake_proj(source_file)
    proj.assets["a1"].has_audio = False
    plan = plan_shorts(proj, "a1", ShortsParams(count=2, target_seconds=20))
    assert len(plan.sequences) >= 1
    assert any("尚未安裝語音辨識" in n for n in plan.notes) and any("沒有逐字稿" in n for n in plan.notes)
    assert any("沒有聲音" in n for n in plan.notes) and any("人臉" in n for n in plan.notes)
    cap_dec = next(d for d in plan.decisions if d["kind"] == "caption")
    assert "沒有字幕來源" in cap_dec["summary"]
    assert all(not s["cues"] for s in plan.sequences)


def test_plan_music_is_a_deterministic_plan_asset_with_static_gain(source_file, patched_analysis, tmp_path):
    patched_analysis.ev = fake_ev(120.0, speech=True)
    music = tmp_path / "bgm.wav"
    subprocess_ok = HAVE_FFMPEG
    if not subprocess_ok:
        pytest.skip("ffmpeg needed to make a music file")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=10",
                    str(music)], check=True)
    proj = fake_proj(source_file)
    params = ShortsParams(count=1, target_seconds=30, music_path=str(music), music_gain_db=-20.0)
    plan = plan_shorts(proj, "a1", params)
    assert len(plan.assets) == 1 and plan.assets[0]["id"].startswith("pa_") and plan.assets[0]["has_audio"]
    assert plan_shorts(proj, "a1", params).assets[0]["id"] == plan.assets[0]["id"] and plan.sha256() == plan_shorts(proj, "a1", params).sha256()
    seq = Sequence.from_dict(plan.sequences[0])
    a1 = seq.audio_tracks[0].clips
    assert len(a1) >= 3 and a1[0].start == 0.0 and a1[-1].fade_out == pytest.approx(1.5)         # 10 s file looped over ~30 s
    assert a1[-1].end == pytest.approx(seq.duration, abs=1e-6)
    assert a1[0].volume == pytest.approx(0.1, rel=1e-6) and a1[0].asset_id == plan.assets[0]["id"]
    assert any("閃避" in n for n in plan.notes)
    # a bad music path is skipped with a note
    bad = plan_shorts(proj, "a1", ShortsParams(count=1, target_seconds=30, music_path=str(tmp_path / "nope.wav")))
    assert not bad.assets and any("配樂" in n for n in bad.notes)
    # an existing project asset is reused instead of being added to the plan
    from autocut.media.probe import probe_asset
    pa = probe_asset(music)
    proj.assets[pa.id] = pa
    reused = plan_shorts(proj, "a1", params)
    assert not reused.assets and Sequence.from_dict(reused.sequences[0]).audio_tracks[0].clips[0].asset_id == pa.id


def test_plan_cancel_raises_before_any_work(source_file, patched_analysis):
    patched_analysis.ev = fake_ev(120.0)
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        plan_shorts(fake_proj(source_file), "a1", ShortsParams(), cancel=ev)
    ev2 = threading.Event()
    calls = []

    def progress(f, m):
        calls.append(f)
        if f >= 0.82:
            ev2.set()                                                          # cancel while planning
    with pytest.raises(Cancelled):
        plan_shorts(fake_proj(source_file), "a1", ShortsParams(), progress=progress, cancel=ev2)
    assert calls == sorted(calls) and calls[0] == 0.0


# ================================================================================ real media (ffmpeg synthetic)
@pytest.fixture(scope="module")
def long_media(tmp_path_factory):
    """40 s: four colour-block scenes (10 s each), 440 Hz bursts (4 s on / 1 s off)."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("shortsmedia")
    out = d / "bursts.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for col in ("red", "green", "blue", "yellow"):
        cmd += ["-f", "lavfi", "-i", f"color=c={col}:s=640x360:r=30:d=10"]
    cmd += ["-f", "lavfi", "-i", "aevalsrc=sin(2*PI*440*t)*0.5*lt(mod(t\\,5)\\,4):s=44100:d=40",
            "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]", "-map", "[v]", "-map", "4:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    return out


@pytest.mark.ffmpeg
def test_propose_shorts_on_synthetic_media(long_media):
    from autocut.media.probe import probe_asset
    proj = Project.new("real")
    a = probe_asset(long_media)
    proj.assets[a.id] = a
    params = ShortsParams(count=2, target_seconds=15, min_gap_between=2.0, captions="off", use_speech=False)
    plan = propose_shorts(proj, a.id, params)
    assert 1 <= len(plan.sequences) <= 2
    for sd in plan.sequences:
        seq = Sequence.from_dict(sd)
        assert (seq.width, seq.height, seq.fps) == (1080, 1920, 30.0)
        assert not [i for i in validate_sequence(proj, seq) if i.level == "error"]
        clips = seq.video_tracks[0].clips
        assert all(0 <= c.src_in < c.src_out <= a.duration + 0.05 for c in clips)
        assert 13.0 <= seq.duration <= 60.0 and not seq.cues
        assert clips[0].duration <= 1.9 + 1e-6                                   # 2-second rule
    ranges = sorted((float(d["source"]["t0"]), float(d["source"]["t1"])) for d in plan.decisions if d["kind"] == "short")
    assert all(b0 >= a1 + 1.9 for (_x, a1), (b0, _y) in zip(ranges, ranges[1:]))
    assert plan.sources[0]["fingerprint"] == a.fingerprint and plan.sources[0]["evidence_sha256"]
    again = propose_shorts(proj, a.id, params)                                  # second run: evidence from the cache
    assert again.sha256() == plan.sha256()
    notes = " ".join(plan.notes)
    assert "人臉" in notes                                                       # nobody in colour blocks: honest note


@pytest.mark.ffmpeg
def test_propose_shorts_whole_clip_music_and_cancel(real_proj, media_dir):
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    music = media_dir / "music.wav"
    plan = propose_shorts(real_proj, land.id, ShortsParams(captions="off", music_path=str(music)))
    assert len(plan.sequences) == 1 and any("只產生 1 支" in n for n in plan.notes)
    seq = Sequence.from_dict(plan.sequences[0])
    assert 7.0 <= seq.duration <= land.duration + 1e-6
    # the music file is already in this project: reused, not added to the plan
    assert not plan.assets and seq.audio_tracks[0].clips and seq.audio_tracks[0].clips[-1].fade_out > 0
    assert not [i for i in validate_sequence(real_proj, seq) if i.level == "error"]
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        propose_shorts(real_proj, land.id, ShortsParams(captions="off"), cancel=ev)
    # a music file the project does not know yet becomes a plan asset with a deterministic id
    lonely = Project.new("only-land")
    lonely.assets[land.id] = land
    plan2 = propose_shorts(lonely, land.id, ShortsParams(captions="off", music_path=str(music)))
    assert len(plan2.assets) == 1 and plan2.assets[0]["id"].startswith("pa_") and plan2.assets[0]["has_audio"]
    assert Sequence.from_dict(plan2.sequences[0]).audio_tracks[0].clips[0].asset_id == plan2.assets[0]["id"]
    assert propose_shorts(lonely, land.id, ShortsParams(captions="off", music_path=str(music))).sha256() == plan2.sha256()


def test_a_dropped_back_channel_does_not_cut_the_caption_before_it_short():
    """A one-word 「是」 squeezed between two lines has no room and is dropped -- the line before it must then run until
    the gap before the NEXT SHOWN caption, not stop where the dropped word would have started (real episode: 7.5 chars/s
    captions with a 0.27 s hole after them)."""
    segs = [{"start": 0.0, "end": 1.0, "text": "一二三四五六"}, {"start": 1.0, "end": 1.1, "text": "是"},
            {"start": 1.15, "end": 3.0, "text": "後面的一句話"}]
    cues = cap.cues_for_pieces(segs, [cap.Piece("a", 0.0, 3.0, 0.0)])
    assert [q.text for q in cues] == ["一二三四五六", "後面的一句話"]
    assert cues[0].end == pytest.approx(1.15 - cap.CAP_GAP)


def test_first_cut_split_leaves_a_real_second_half_when_the_clip_allows():
    """A first clip just over the 2-second limit (2.17 s, a cold open) must not end in a 0.27 s zoomed flash."""
    ev = fake_ev(60.0, burst=60.0, gap=0.0)                         # one long burst: equal energy everywhere
    feat = sc.Feat(ev, None)
    t = sc.first_cut_time(feat, 10.0, 12.17, 2.05)
    assert t is not None and 10.0 + 1.0 - 1e-9 <= t <= 12.17 - sc.SECOND_HALF_MIN + 1e-9
    # a clip so short that 0.6 s cannot be kept still gets a valid split (the old 0.25 s floor)
    t2 = sc.first_cut_time(feat, 10.0, 11.3, 1.0)
    assert t2 is not None and t2 <= 11.3 - 0.25 + 1e-9
