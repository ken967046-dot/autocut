"""LiveCompositor: pixel comparison against the ffmpeg renderer, plus unit tests (SAR / rotation, gaps, LRU, ...)."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from autocut.core import ops
from autocut.core.project import Look, Project
from autocut.media.probe import probe_asset
from autocut.render.live import MAX_READERS, LiveCompositor, LiveError, apply_look, bgr_of
from autocut.render.runner import RenderSettings, render_sequence

pytestmark = pytest.mark.ffmpeg
REPORT: dict[str, list[tuple[float, float, float]]] = {}      # case -> [(t, mean abs diff, mean abs diff after blur)]


# ------------------------------------------------------------------------------------- helpers
def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def _by_name(p: Project) -> dict:
    return {a.name: a for a in p.assets.values()}


def _seq(p: Project, name: str, w: int, h: int, fps: float = 30.0):
    return ops.create_sequence(p, name, w, h, fps)


def _frames_of(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    cap, out, i, want = cv2.VideoCapture(str(path)), {}, 0, set(indices)
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = fr
            want.discard(i)
        i += 1
    cap.release()
    return out


def _diff(live: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    if live.shape != ref.shape:
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = live.astype(np.int16), ref.astype(np.int16)
    blur = np.abs(cv2.GaussianBlur(live, (0, 0), 2).astype(np.int16) - cv2.GaussianBlur(ref, (0, 0), 2).astype(np.int16))
    return float(np.abs(a - b).mean()), float(blur.mean())


def _compare(p: Project, seq, times: list[float], tmp: Path, case: str, max_raw: float = 8.0) -> list:
    """Render ``seq`` with ffmpeg (draft_fast) and compare the live frames at ``times`` with the rendered ones."""
    out = tmp / f"{case}.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    idx = [int(round(t * seq.fps)) for t in times]
    ref = _frames_of(out, idx)
    live = LiveCompositor(p, max_side=max(res.width, res.height))
    live.set_sequence(seq.id)
    rows = []
    for t, i in zip(times, idx):
        got = live.frame_at(t)
        assert got.shape[:2] == ref[i].shape[:2], (got.shape, ref[i].shape)
        raw, blur = _diff(got, ref[i])
        rows.append((t, raw, blur))
    live.close()
    REPORT[case] = rows
    worst = max(r[1] for r in rows)
    assert np.mean([r[1] for r in rows]) <= max_raw and worst <= max_raw * 1.6, f"{case}: {rows}"
    assert np.mean([r[2] for r in rows]) <= 4.5, f"{case} (blurred): {rows}"
    return rows


@pytest.fixture(scope="module", autouse=True)
def _write_report():
    yield
    if REPORT:
        d = Path(__file__).resolve().parents[1] / "screenshots"
        d.mkdir(exist_ok=True)
        lines = []
        for case, rows in REPORT.items():
            lines.append(f"{case}: mean {np.mean([r[1] for r in rows]):.2f} max {max(r[1] for r in rows):.2f} | "
                         + ", ".join(f"t={t:.2f}:{raw:.1f}/{blur:.1f}" for t, raw, blur in rows))
        (d / "live_compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lines))


@pytest.fixture
def lp(real_proj):
    return real_proj, _by_name(real_proj)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    """Tiny clips with known content: frame-id luma ramp, a rotated copy, an anamorphic copy, an image."""
    d = tmp_path_factory.mktemp("livemedia")
    _ff("-f", "lavfi", "-i", "color=c=black:s=64x36:r=30:d=3", "-vf", "format=gray,geq=lum='16+2.4*N',format=yuv420p",
        "-c:v", "libx264", "-crf", "10", "-g", "10", "-bf", "2", "-pix_fmt", "yuv420p", str(d / "fid.mp4"))
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(d / "plain.mp4"))
    _ff("-display_rotation", "90", "-i", str(d / "plain.mp4"), "-c", "copy", str(d / "rot.mp4"))
    _ff("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=1", "-vf", "setsar=4/3", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(d / "sar.mp4"))
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = (40, 120, 200)
    cv2.rectangle(img, (40, 40), (140, 120), (0, 0, 255), -1)
    cv2.circle(img, (300, 200), 50, (255, 255, 0), -1)
    cv2.imwrite(str(d / "photo.png"), img)
    return d


def _fid_project(tiny: Path, name: str = "fid.mp4", w: int = 64, h: int = 36, **clip):
    p = Project.new("fid")
    a = probe_asset(tiny / name)
    p.assets[a.id] = a
    seq = _seq(p, "fid", w, h)
    c = ops.place_clip(p, seq, a.id, seq.video_tracks[0].id, start=0, src_in=clip.pop("src_in", 0.0),
                       src_out=clip.pop("src_out", 3.0), **clip)
    return p, seq, a, c


def _means(path: Path) -> list[float]:
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, fr = cap.read()
        if not ok:
            return out
        out.append(float(fr.mean()))


# ------------------------------------------------------------------ comparison with the ffmpeg renderer
def test_cmp_cover_keys_zoom(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "ck", 360, 640)
    c = ops.place_clip(p, seq, a["land.mp4"].id, seq.video_tracks[0].id, start=0, src_in=0, src_out=3)
    ops.set_clip_props(p, seq, c.id, zoom=1.5)
    c.transform.keys = [[0.0, 0.3, 0.4], [3.0, 0.75, 0.6]]
    _compare(p, seq, [0.0, 1.0, 2.0, 2.9], tmp_path, "cover_keys_zoom")


def test_cmp_cover_static_and_flip(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "cs", 360, 640)
    c = ops.place_clip(p, seq, a["land.mp4"].id, seq.video_tracks[0].id, start=0, src_in=1, src_out=2.5)
    ops.set_clip_props(p, seq, c.id, cx=0.2, cy=0.5, flip_h=True)
    _compare(p, seq, [0.0, 0.7, 1.4], tmp_path, "cover_static_flip")


def test_cmp_contain_and_blur(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "ct", 360, 640)
    v = seq.video_tracks[0]
    c1 = ops.place_clip(p, seq, a["land.mp4"].id, v.id, start=0, src_in=1, src_out=2)
    c2 = ops.place_clip(p, seq, a["land.mp4"].id, v.id, start=1, src_in=2, src_out=3)
    ops.set_clip_props(p, seq, c1.id, fit="contain")
    ops.set_clip_props(p, seq, c2.id, fit="blur")
    _compare(p, seq, [0.3, 0.9], tmp_path, "contain")
    rows_blur = _compare(p, seq, [1.3, 1.9], tmp_path, "blur")
    assert max(r[1] for r in rows_blur) < 8


def test_cmp_crossfade_and_fades(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "xf", 480, 270)
    v = seq.video_tracks[0]
    c1 = ops.place_clip(p, seq, a["land.mp4"].id, v.id, start=0, src_in=0, src_out=3)
    c2 = ops.place_clip(p, seq, a["silent.mp4"].id, v.id, start=3, src_in=0, src_out=3)
    ops.set_crossfade(p, seq, c2.id, 1.0)
    ops.set_clip_props(p, seq, c1.id, fade_in=0.5)
    ops.set_clip_props(p, seq, c2.id, fade_out=0.5)
    _compare(p, seq, [0.0, 0.2, 1.0, 2.2, 2.5, 2.8, 3.5, 4.8], tmp_path, "crossfade_fades")


def test_cmp_speed_flip_look(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "sf", 480, 270)
    c = ops.place_clip(p, seq, a["land.mp4"].id, seq.video_tracks[0].id, start=0, src_in=0, src_out=4)
    ops.set_speed(p, seq, c.id, 2.0)
    ops.set_clip_props(p, seq, c.id, flip_h=True, brightness=0.1, contrast=1.2, saturation=0.6)
    _compare(p, seq, [0.0, 0.5, 1.0, 1.9], tmp_path, "speed2_flip_look")


def test_cmp_look_negative_brightness(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "lk", 480, 270)
    c = ops.place_clip(p, seq, a["land.mp4"].id, seq.video_tracks[0].id, start=0, src_in=0, src_out=1)
    ops.set_clip_props(p, seq, c.id, brightness=-0.2, contrast=0.8, saturation=1.5)
    _compare(p, seq, [0.1, 0.6], tmp_path, "look_neg")


def test_cmp_two_track_overlay_and_gap(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "ov", 480, 270)
    seq.background = "#204060"
    ops.place_clip(p, seq, a["land.mp4"].id, seq.video_tracks[0].id, start=0.5, src_in=0, src_out=2.5)
    c2 = ops.place_clip(p, seq, a["silent.mp4"].id, seq.video_tracks[1].id, start=1.5, src_in=0, src_out=1.5)
    ops.set_clip_props(p, seq, c2.id, fade_in=1.0)
    _compare(p, seq, [0.2, 0.6, 1.6, 2.0, 2.4, 2.8, 2.95], tmp_path, "overlay_gap_bg")


def test_cmp_image_cover(lp, tiny, tmp_path):
    p, a = lp
    img = probe_asset(tiny / "photo.png")
    p.assets[img.id] = img
    seq = _seq(p, "im", 360, 640)
    c = ops.place_clip(p, seq, img.id, seq.video_tracks[0].id, start=0, src_out=2.0)
    ops.set_clip_props(p, seq, c.id, zoom=1.3, cx=0.4)
    _compare(p, seq, [0.0, 1.0], tmp_path, "image_cover")


# ------------------------------------------------------------------------------ decoding behaviour
def test_frame_accuracy_forward_backward_random(tiny):
    p, seq, a, _ = _fid_project(tiny)
    ref = _means(tiny / "fid.mp4")
    assert len(ref) == 90 and all(b - a >= 1.0 for a, b in zip(ref, ref[1:])), ref[:12]   # every frame is distinguishable
    lc = LiveCompositor(p)
    for k in [0, 1, 2, 3, 30, 31, 10, 11, 12, 70, 5, 88, 89, 45, 44, 60]:
        got = float(lc.frame_at(k / 30).mean())
        assert abs(got - ref[k]) < 1.0, (k, got, ref[k])
    lc.close()


def test_sequential_read_then_skip_then_seek(tiny):
    p, seq, a, _ = _fid_project(tiny)
    lc = LiveCompositor(p)
    for k in range(30):
        lc.frame_at(k / 30)
    assert lc.stats["reads"] == 30 and lc.stats["seeks"] == 0 and lc.stats["skips"] == 0
    lc.frame_at(40 / 30)                                     # small forward gap: grab() the frames in between
    assert lc.stats["seeks"] == 0 and lc.stats["skips"] == 10
    lc.frame_at(85 / 30)                                     # large gap: a 2nd decoder seeks, the 1st keeps its position
    assert lc.stats["seeks"] == 1 and lc.open_readers == 2
    lc.frame_at(50 / 30)                                     # "backwards" for the 2nd decoder, 9 frames ahead of the 1st
    assert lc.stats["seeks"] == 1 and lc.stats["skips"] == 19
    lc.frame_at(5 / 30)                                      # near the start: a 3rd decoder opens at 0 and skips 5 frames
    assert lc.stats["seeks"] == 1 and lc.open_readers == 3
    lc.frame_at(35 / 30)                                     # unreachable going forward and the pool is full: one seeks
    assert lc.stats["seeks"] == 2 and lc.open_readers == 3
    reads = lc.stats["reads"]
    lc.frame_at(35 / 30)                                     # same frame again: served from the pool
    assert lc.stats["reads"] == reads
    lc.close()


def test_speed_uses_source_time(tiny):
    p, seq, a, c = _fid_project(tiny, src_in=0.0, src_out=3.0)
    ops.set_speed(p, seq, c.id, 2.0)                         # 1.5 s on the timeline
    ref = _means(tiny / "fid.mp4")
    lc = LiveCompositor(p)
    for t in (0.0, 0.5, 1.0):
        n = int(round(c.src_at(t) * 30))
        assert abs(float(lc.frame_at(t).mean()) - ref[n]) < 1.0, t
    lc.close()


def test_lru_pool_never_exceeds_limit(tiny, tmp_path):
    p = Project.new("lru")
    seq = _seq(p, "lru", 64, 36)
    for i in range(MAX_READERS + 3):
        copy = tmp_path / f"c{i}.mp4"
        shutil.copy(tiny / "fid.mp4", copy)
        a = probe_asset(copy)
        p.assets[a.id] = a
        ops.place_clip(p, seq, a.id, seq.video_tracks[0].id, start=i * 0.5, src_in=0, src_out=0.5)
    lc = LiveCompositor(p)
    for i in range(MAX_READERS + 3):
        assert lc.frame_at(i * 0.5 + 0.1).mean() > 10
        assert lc.open_readers <= MAX_READERS
    assert lc.stats["opens"] == MAX_READERS + 3
    lc.frame_at(0.1)                                         # the oldest was evicted: re-opened on demand
    assert lc.stats["opens"] == MAX_READERS + 4 and lc.open_readers <= MAX_READERS
    lc.close()
    assert lc.open_readers == 0


def test_unicode_file_name(tiny, tmp_path):
    p = Project.new("zh")
    seq = _seq(p, "zh", 64, 36)
    copy = tmp_path / "測試影片 一.mp4"
    shutil.copy(tiny / "fid.mp4", copy)
    a = probe_asset(copy)
    p.assets[a.id] = a
    ops.place_clip(p, seq, a.id, seq.video_tracks[0].id, start=0, src_in=0, src_out=1)
    lc = LiveCompositor(p)
    assert lc.frame_at(0.5).mean() > 10 and not lc.missing
    lc.close()


# ---------------------------------------------------------------- orientation and pixel aspect
def test_rotation_matches_ffmpeg_and_manual_fallback(tiny):
    a = probe_asset(tiny / "rot.mp4")
    assert (a.width, a.height) == (180, 320) and a.rotation in (90, 270)
    expect = tiny / "rot_expect.png"
    _ff("-i", str(tiny / "rot.mp4"), "-frames:v", "1", str(expect))           # ffmpeg auto-rotates
    ref = cv2.imread(str(expect))
    assert ref.shape[:2] == (320, 180)
    wrong = cv2.rotate(ref, cv2.ROTATE_180)
    for auto in (True, False):
        p = Project.new("rot")
        p.assets[a.id] = a
        seq = _seq(p, "rot", 180, 320)
        c = ops.place_clip(p, seq, a.id, seq.video_tracks[0].id, start=0, src_in=0, src_out=1)
        lc = LiveCompositor(p, max_side=400, auto_orient=auto)
        src = lc.source_frame_at(c.id, 0.0)
        assert src.shape[:2] == (320, 180)
        good, bad = _diff(src, ref)[0], _diff(src, wrong)[0]
        assert good < 5 and bad > 3 * good, (auto, good, bad)
        full = lc.frame_at(0.0)                                  # 9:16 sequence, 9:16 display: plain scale
        assert _diff(full, ref)[0] < 6
        lc.close()


def test_sar_is_applied_to_geometry(tiny):
    a = probe_asset(tiny / "sar.mp4")
    assert a.sar == pytest.approx(4 / 3) and (a.width, a.height) == (427, 180)
    expect = tiny / "sar_expect.png"
    _ff("-i", str(tiny / "sar.mp4"), "-vf", "scale=iw*sar:ih,setsar=1", "-frames:v", "1", str(expect))
    ref = cv2.imread(str(expect))
    p = Project.new("sar")
    p.assets[a.id] = a
    seq = _seq(p, "sar", 360, 640)
    c = ops.place_clip(p, seq, a.id, seq.video_tracks[0].id, start=0, src_in=0, src_out=1)
    lc = LiveCompositor(p, max_side=720)
    src = lc.source_frame_at(c.id, 0.0)
    assert src.shape[:2] == (180, 427)
    assert _diff(src, ref)[0] < 6
    # cover into 9:16 crops a window of the *display* frame: 180 high -> 101 wide, centred
    got = lc.frame_at(0.0)
    window = src[:, (427 - 100) // 2 // 2 * 2:][:, :100]
    assert _diff(got, cv2.resize(window, (360, 640), interpolation=cv2.INTER_LINEAR))[0] < 8
    lc.close()


def test_source_frame_at_contract(lp):
    p, a = lp
    seq = _seq(p, "sf", 360, 640)
    c = ops.place_clip(p, seq, a["land.mp4"].id, seq.video_tracks[0].id, start=2, src_in=1, src_out=3)
    audio = ops.place_clip(p, seq, a["music.wav"].id, seq.audio_tracks[0].id, start=0, src_out=2)
    lc = LiveCompositor(p, max_side=640)
    s0 = lc.source_frame_at(c.id, 0.0)
    assert s0.shape[:2] == (360, 640)                                         # whole 16:9 picture, long side 640
    assert lc.source_frame_at(c.id, 0.0, max_side=320).shape[:2] == (180, 320)
    assert lc.source_frame_at(c.id, 99.0).shape == s0.shape                   # clamped to the clip
    assert not np.array_equal(s0, lc.source_frame_at(c.id, 1.0))              # different time -> different picture
    with pytest.raises(LiveError):
        lc.source_frame_at("nope", 0.0)
    with pytest.raises(LiveError):
        lc.source_frame_at(audio.id, 0.0)
    lc.close()


# -------------------------------------------------------------------------- compositing semantics
def _px(img: np.ndarray) -> np.ndarray:
    return img[img.shape[0] // 2, img.shape[1] // 2].astype(int)       # BGR of the centre


def test_gap_background_and_beyond_end(lp):
    p, a = lp
    seq = _seq(p, "gap", 160, 90)
    seq.background = "#336699"
    ops.place_clip(p, seq, a["silent.mp4"].id, seq.video_tracks[0].id, start=1, src_in=0, src_out=1)
    lc = LiveCompositor(p)
    for t in (0.0, 0.5, 2.5, 99.0, -3.0):                        # before / after / far beyond the end / negative
        assert (lc.frame_at(t) == np.array(bgr_of("#336699"), np.uint8)).all(), t
    assert abs(_px(lc.frame_at(1.5))[1] - 128) < 6               # the clip itself: green
    assert lc.frame_at(0.0).shape == (90, 160, 3) and lc.output_size() == (160, 90)
    lc.close()


def test_output_size_long_side_and_aspect(lp):
    p, _ = lp
    seq = _seq(p, "big", 1080, 1920)
    lc = LiveCompositor(p, max_side=720)
    assert lc.output_size() == (405, 720) and lc.frame_at(0.0).shape == (720, 405, 3)
    small = _seq(p, "small", 160, 90)
    lc.set_sequence(small.id)
    assert lc.output_size() == (160, 90)                         # never upscaled
    lc.set_sequence(seq.id)
    assert lc.output_size() == (405, 720)
    lc.close()


def test_tracks_hidden_disabled_and_stacking(lp):
    p, a = lp
    seq = _seq(p, "stack", 160, 90)
    ops.place_clip(p, seq, a["silent.mp4"].id, seq.video_tracks[0].id, start=0, src_in=0, src_out=2)   # green below
    top = ops.place_clip(p, seq, a["port.mp4"].id, seq.video_tracks[1].id, start=0, src_in=0, src_out=2)  # blue on top
    lc = LiveCompositor(p)
    blue = _px(lc.frame_at(0.5))
    assert blue[0] > 200 and blue[1] < 40                        # BGR: the upper track covers the lower one
    seq.video_tracks[1].muted = True                             # hidden track
    assert _px(lc.frame_at(0.5))[1] > 100
    seq.video_tracks[1].muted = False
    top.enabled = False                                          # disabled clip
    assert _px(lc.frame_at(0.5))[1] > 100
    top.enabled = True
    assert _px(lc.frame_at(0.5))[0] > 200
    lc.close()


def test_fade_in_out_and_crossfade_alpha(lp):
    p, a = lp
    seq = _seq(p, "fade", 160, 90)
    v = seq.video_tracks[0]
    g = ops.place_clip(p, seq, a["silent.mp4"].id, v.id, start=0, src_in=0, src_out=2)      # green 0-2
    ops.set_clip_props(p, seq, g.id, fade_in=1.0, fade_out=0.5)
    lc = LiveCompositor(p)
    green = [int(lc.frame_at(k / 30)[45, 80, 1]) for k in (0, 15, 29, 30, 44, 52, 59)]
    assert green[0] < 4 and 58 < green[1] < 70 and 118 < green[2] < 128 and 124 < green[3] < 132
    assert green[4] > 120 and 60 < green[5] < 76 and green[6] < 12, green      # fade-out 1.5 s -> 2.0 s
    ops.remove_clips(p, seq, [g.id])
    first = ops.place_clip(p, seq, a["silent.mp4"].id, v.id, start=0, src_in=0, src_out=2)    # green
    second = ops.place_clip(p, seq, a["port.mp4"].id, v.id, start=2, src_in=0, src_out=2)     # blue, 1 s crossfade
    ops.set_crossfade(p, seq, second.id, 1.0)
    mid = lc.frame_at(1.5)[45, 80]                               # 50 % blue over 100 % green
    assert 118 < mid[0] < 136 and 58 < mid[1] < 70, mid
    assert lc.frame_at(0.5)[45, 80][0] < 6 and lc.frame_at(2.5)[45, 80][1] < 6
    lc.close()


def test_missing_media_and_invalidate(lp, tmp_path):
    p, a = lp
    seq = _seq(p, "miss", 160, 90)
    ghost = tmp_path / "later.mp4"
    asset = probe_asset(Path(a["silent.mp4"].path))
    asset.path = str(ghost)
    p.assets[asset.id] = asset
    ops.place_clip(p, seq, asset.id, seq.video_tracks[0].id, start=0, src_in=0, src_out=1)
    lc = LiveCompositor(p)
    f = lc.frame_at(0.5)
    assert f.max() == 0 and lc.missing == {str(ghost)}           # background only, no exception
    shutil.copy(a["silent.mp4"].path, ghost)
    lc.invalidate()
    assert lc.frame_at(0.5)[45, 80, 1] > 100 and not lc.missing
    lc.close()


def test_image_cache_follows_file_changes(tiny, tmp_path):
    img = tmp_path / "pic.png"
    cv2.imwrite(str(img), np.full((90, 160, 3), (0, 0, 255), np.uint8))            # red
    p = Project.new("img")
    a = probe_asset(img)
    p.assets[a.id] = a
    seq = _seq(p, "img", 160, 90)
    ops.place_clip(p, seq, a.id, seq.video_tracks[0].id, start=0, src_out=1)
    lc = LiveCompositor(p)
    assert _px(lc.frame_at(0.1))[2] > 200
    time.sleep(0.02)
    cv2.imwrite(str(img), np.full((90, 160, 3), (255, 0, 0), np.uint8))            # blue
    os.utime(img, ns=(time.time_ns(), time.time_ns() + 5_000_000))
    lc.invalidate()
    assert _px(lc.frame_at(0.1))[0] > 200
    lc.close()


def test_edits_are_picked_up_immediately(lp):
    p, a = lp
    seq = _seq(p, "edit", 160, 90)
    c = ops.place_clip(p, seq, a["land.mp4"].id, seq.video_tracks[0].id, start=0, src_in=0, src_out=2)
    lc = LiveCompositor(p)
    before = lc.frame_at(0.5)
    ops.set_clip_props(p, seq, c.id, flip_h=True)
    assert not np.array_equal(before, lc.frame_at(0.5))
    assert np.array_equal(cv2.flip(before, 1), lc.frame_at(0.5))
    lc.close()


def test_apply_look_identity_and_direction():
    img = np.random.default_rng(1).integers(0, 255, (24, 32, 3), dtype=np.uint8)
    assert np.abs(apply_look(img, Look()).astype(int) - img.astype(int)).max() <= 1
    assert apply_look(img, Look(brightness=0.2)).mean() > img.mean() + 20
    assert apply_look(img, Look(saturation=0.0)).std(axis=2).mean() < 1.5         # grey: channels equal
    assert apply_look(img, Look(contrast=2.0)).std() > img.std()
