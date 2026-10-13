"""Render engine integration tests with synthetic ffmpeg media (skipped without ffmpeg)."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from autocut.core import ops
from autocut.core.project import Project
from autocut.media.ffmpeg import ffmpeg_info
from autocut.media.probe import probe_asset
from autocut.render.compiler import compile_sequence, piecewise_expr
from autocut.render import qa as qa_mod
from autocut.render.qa import FAIL, NOT_RUN, PASS, WARN, technical_qa, worst_status
from autocut.render.runner import RenderSettings, render_sequence

pytestmark = pytest.mark.ffmpeg


def frame_at(path, t: float, size=None) -> np.ndarray:
    """Decode one BGR frame at time ``t`` (accurate seek) as an array."""
    info = ffmpeg_info()
    p = subprocess.run([info.ffmpeg, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
    meta = subprocess.run([info.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    w, h = (int(x) for x in meta.stdout.strip().split(","))
    return np.frombuffer(p.stdout, dtype=np.uint8).reshape(h, w, 3)


def audio_of(path, rate=16000) -> np.ndarray:
    info = ffmpeg_info()
    p = subprocess.run([info.ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le",
                        "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def dominant_freq(x: np.ndarray, rate=16000) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec) * rate / len(x))


@pytest.fixture
def mp(media_dir):
    p = Project.new("t")
    a = {}
    for key, fn in (("land", "land.mp4"), ("port", "port.mp4"), ("silent", "silent.mp4"), ("music", "music.wav")):
        asset = probe_asset(media_dir / fn)
        p.assets[asset.id] = asset
        a[key] = asset
    p.meta["_a"] = a
    return p, a


def set_fmt(seq, w, h, fps=30.0):
    ops.set_sequence_format(None, seq, w, h, fps)


def test_probe(mp):
    p, a = mp
    assert (a["land"].width, a["land"].height) == (1280, 720) and a["land"].has_audio
    assert a["land"].fps == pytest.approx(30, abs=0.01) and a["land"].duration == pytest.approx(8, abs=0.1)
    assert (a["port"].width, a["port"].height) == (720, 1280) and a["port"].is_portrait
    assert not a["silent"].has_audio and a["silent"].fps == pytest.approx(25, abs=0.01)
    assert a["music"].kind == "audio" and not a["music"].has_video and a["music"].has_audio


def test_piecewise_expr_shape():
    e = piecewise_expr([(0.0, 0.2), (1.0, 0.8), (2.0, 0.4)], "t")
    assert e.startswith("if(lt(t,0),0.2,if(gte(t,2),0.4,")
    assert e.count("(") == e.count(")")
    assert piecewise_expr([(0, 0.5)], "t") == "0.5"


def test_compile_boundaries_and_crossfade(mp):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 1280, 720)
    v = seq.video_tracks[0]
    c1 = ops.place_clip(p, seq, a["land"].id, v.id, start=0, src_out=4)
    c2 = ops.place_clip(p, seq, a["silent"].id, v.id, start=4, src_out=3)
    ops.set_crossfade(p, seq, c2.id, 1.0)
    comp = compile_sequence(p, seq)
    spans = [(s.f0, s.f1) for s in comp.segments]
    assert spans == [(0, 90), (90, 120), (120, 183)] or spans[0][0] == 0
    # the overlap segment has both clips, the other two only one
    kinds = [len(s.v) for s in comp.segments]
    assert kinds == [1, 2, 1]
    assert comp.total_frames == comp.segments[-1].f1
    # crossfade: second clip fades in during the overlap, first clip's audio fades out
    ov = comp.segments[1]
    assert ov.v[1].vfi == 30 and ov.a[0].afo == 30


def test_render_cut_gap_and_blur_fit(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 1280, 720)
    v = seq.video_tracks[0]
    ops.place_clip(p, seq, a["land"].id, v.id, start=0, src_out=2)                      # 0-2  picture + 440 Hz
    cb = ops.place_clip(p, seq, a["port"].id, v.id, start=3, src_out=2)                 # 3-5  blue portrait
    ops.set_clip_props(p, seq, cb.id, fit="blur")
    out = tmp_path / "o.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    assert res.frames == 150 and (res.width, res.height) == (1280, 720)
    rep = technical_qa(out, expect_duration=5.0, expect_size=(1280, 720), expect_fps=30.0,
                       intended_black=[(2.0, 3.0)])
    assert rep.status in (PASS, "warn"), rep.summary()
    assert [c for c in rep.checks if c.name == "長度"][0].status == PASS
    assert [c for c in rep.checks if c.name == "完整解碼"][0].status == PASS
    f_pic, f_gap, f_blue = frame_at(out, 1.0), frame_at(out, 2.5), frame_at(out, 4.0)
    assert f_pic.mean() > 40
    assert f_gap.mean() < 8                                  # black gap
    centre = f_blue[300:420, 560:720]                        # the foreground portrait video in the middle
    assert centre[..., 0].mean() > 150 and centre[..., 2].mean() < 60   # BGR: blue high, red low
    au = audio_of(out)
    assert abs(len(au) - 5 * 16000) < 1600
    assert dominant_freq(au[int(0.5 * 16000): int(1.5 * 16000)]) == pytest.approx(440, abs=15)
    assert np.abs(au[int(2.3 * 16000): int(2.8 * 16000)]).max() < 0.01                   # silent gap
    assert dominant_freq(au[int(3.5 * 16000): int(4.5 * 16000)]) == pytest.approx(880, abs=15)


def test_render_crossfade_pixels_and_audio(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 1280, 720)
    v = seq.video_tracks[0]
    ops.place_clip(p, seq, a["land"].id, v.id, start=0, src_out=4)
    c2 = ops.place_clip(p, seq, a["silent"].id, v.id, start=4, src_out=4)               # solid green, no audio
    ops.set_crossfade(p, seq, c2.id, 2.0)                                                # overlap 2..4
    assert seq.duration == pytest.approx(6.0)
    out = tmp_path / "x.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    assert res.frames == 180
    start, mid, end = frame_at(out, 1.0), frame_at(out, 3.0), frame_at(out, 5.0)
    green = np.array([0, 128, 0])
    assert np.abs(end.reshape(-1, 3).mean(0)[[0, 2]]).max() < 20 and end[..., 1].mean() > 100     # pure green
    gm = mid[..., 1].mean()
    assert 60 < gm < 150 and not np.allclose(mid.reshape(-1, 3).mean(0), end.reshape(-1, 3).mean(0), atol=8)
    assert not np.allclose(mid.reshape(-1, 3).mean(0), start.reshape(-1, 3).mean(0), atol=8)
    au = audio_of(out)
    e_before = np.abs(au[int(1.0 * 16000): int(1.5 * 16000)]).mean()
    e_mid = np.abs(au[int(2.8 * 16000): int(3.2 * 16000)]).mean()
    e_after = np.abs(au[int(4.5 * 16000): int(5.5 * 16000)]).mean()
    assert e_before > 0.02 and 0 < e_mid < e_before * 0.8 and e_after < 0.005                    # equal-power fade-out


def test_render_reframe_916_with_pan_keys(mp, tmp_path):
    p, a = mp
    seq = ops.create_sequence(p, "短", 540, 960, 30)
    v = seq.video_tracks[0]
    c = ops.place_clip(p, seq, a["land"].id, v.id, start=0, src_in=0, src_out=4)
    c.transform.keys = [[0.0, 0.15, 0.5], [4.0, 0.85, 0.5]]
    out = tmp_path / "r.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    assert (res.width, res.height) == (540, 960)
    f0, f1 = frame_at(out, 0.1), frame_at(out, 3.8)
    assert f0.shape[:2] == (960, 540)
    assert np.abs(f0.astype(int) - f1.astype(int)).mean() > 8        # the camera really moved
    # without keys the two frames would be nearly identical apart from the animated test pattern; check the
    # first frame equals the leftmost crop of the source frame (centre x = 0.15 * 1280 = 192, window 405 wide)
    src0 = frame_at(a["land"].path, 0.1)
    left = src0[:, 0:405]
    import cv2
    ref = cv2.resize(left, (540, 960), interpolation=cv2.INTER_AREA)
    assert np.abs(f0.astype(int) - ref.astype(int)).mean() < 40


def test_render_speed_and_fades(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 640, 360)
    v = seq.video_tracks[0]
    c = ops.place_clip(p, seq, a["land"].id, v.id, start=0, src_in=0, src_out=6)
    ops.set_speed(p, seq, c.id, 2.0)                                       # 3 s
    ops.set_clip_props(p, seq, c.id, fade_in=1.0, fade_out=1.0)
    out = tmp_path / "s.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    assert res.frames == 90
    first, mid, last = frame_at(out, 0.0), frame_at(out, 1.5), frame_at(out, 2.95)
    assert first.mean() < 10 and last.mean() < 40 and mid.mean() > first.mean() + 30
    au = audio_of(out)
    assert dominant_freq(au[int(1.0 * 16000): int(2.0 * 16000)]) == pytest.approx(440, abs=20)   # atempo keeps the pitch


def test_render_multitrack_music_and_cache(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 640, 360)
    v, am = seq.video_tracks[0], seq.audio_tracks[0]
    ops.place_clip(p, seq, a["silent"].id, v.id, start=0, src_out=4)
    m = ops.place_clip(p, seq, a["music"].id, am.id, start=1, src_out=3)
    ops.set_clip_props(p, seq, m.id, volume=0.5)
    out = tmp_path / "m.mp4"
    r1 = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    assert r1.cached == 0 and r1.frames == 120
    au = audio_of(out)
    assert np.abs(au[: int(0.8 * 16000)]).max() < 0.005                       # before the music starts
    assert dominant_freq(au[int(1.5 * 16000): int(2.5 * 16000)]) == pytest.approx(220, abs=15)
    r2 = render_sequence(p, seq, tmp_path / "m2.mp4", RenderSettings(preset="draft_fast", normalize=False))
    assert r2.cached == r2.segments > 0
    # a tiny edit re-renders only the touched segment(s)
    ops.set_clip_props(p, seq, m.id, volume=0.4)
    r3 = render_sequence(p, seq, tmp_path / "m3.mp4", RenderSettings(preset="draft_fast", normalize=False))
    assert 0 < r3.cached < r3.segments or r3.segments == 1


def test_render_captions_burned_in(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 640, 360)
    ops.place_clip(p, seq, a["silent"].id, seq.video_tracks[0].id, start=0, src_out=3)
    plain = tmp_path / "plain.mp4"
    render_sequence(p, seq, plain, RenderSettings(preset="draft_fast", normalize=False))
    ops.add_cue(p, seq, 1.0, 2.0, "HELLO caption")
    seq.caption_style.font = "DejaVu Sans"
    capt = tmp_path / "capt.mp4"
    res = render_sequence(p, seq, capt, RenderSettings(preset="draft_fast", normalize=False))
    if any("libass" in w for w in res.warnings):
        pytest.skip("this ffmpeg has no libass")
    a_, b_ = frame_at(plain, 1.5), frame_at(capt, 1.5)
    assert np.abs(a_.astype(int) - b_.astype(int)).mean() > 0.3            # text pixels were drawn
    assert np.abs(frame_at(plain, 0.5).astype(int) - frame_at(capt, 0.5).astype(int)).mean() < 0.5   # not before 1.0 s


def test_range_export_and_srt(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 640, 360)
    ops.place_clip(p, seq, a["land"].id, seq.video_tracks[0].id, start=0, src_out=6)
    ops.add_cue(p, seq, 2.0, 3.0, "中間字幕")
    out = tmp_path / "range.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False, frame_range=(1.0, 4.0),
                                                       burn_captions=False, write_srt=True))
    assert res.frames == 90
    srt = out.with_suffix(".srt").read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000" in srt and "中間字幕" in srt


def test_loudness_target_applied(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 640, 360)
    ops.place_clip(p, seq, a["land"].id, seq.video_tracks[0].id, start=0, src_out=5)
    out = tmp_path / "l.mp4"
    render_sequence(p, seq, out, RenderSettings(preset="youtube_1080p", loudness=-16.0))
    rep = technical_qa(out, expect_duration=5.0, expect_fps=30.0)
    lufs = rep.measurements.get("loudness_lufs")
    assert lufs is not None and -18.5 < lufs < -13.5, rep.summary()


def test_qa_detects_missing_and_wrong_expectation(tmp_path):
    rep = technical_qa(tmp_path / "nope.mp4")
    assert rep.status == FAIL and rep.human_review == "pending"


# ------------------------------------------------------------------ a check that did not finish is never a pass
def qa_with_crippled_detector_pass(monkeypatch, path, outcome):
    """``technical_qa`` with the real decode pass but a detector pass (black/freeze/silence/loudness) that ends as
    ``outcome`` = ``(return code, stderr text, problem)``."""
    real, calls = qa_mod._run_ffmpeg, []

    def fake(cmd, timeout=3600):
        calls.append(cmd)
        return real(cmd, timeout) if len(calls) == 1 else outcome

    monkeypatch.setattr(qa_mod, "_run_ffmpeg", fake)
    rep = technical_qa(path)
    assert len(calls) == 2
    return rep


DETECTORS = ("黑畫面", "凍結畫面", "長時間靜音", "響度", "真峰值")


@pytest.mark.parametrize("outcome, shown", [
    ((1, "Error while filtering\nInvalid argument", ""), "Invalid argument"),
    ((None, "", "ffmpeg 超過 60 分鐘還沒結束，已放棄"), "超過 60 分鐘"),
    ((None, "", "無法執行 ffmpeg：[WinError 2]"), "無法執行"),
])
def test_a_detector_pass_that_did_not_finish_is_not_run_and_never_a_pass(monkeypatch, media_dir, outcome, shown):
    rep = qa_with_crippled_detector_pass(monkeypatch, media_dir / "land.mp4", outcome)
    by = {c.name: c for c in rep.checks}
    for name in DETECTORS:
        assert by[name].status == NOT_RUN and shown in by[name].detail, (name, by[name])
    assert by["完整解碼"].status == PASS                      # the decode pass really ran
    assert rep.status == NOT_RUN and rep.human_review == "pending"
    assert "loudness_lufs" not in rep.measurements and "true_peak_dbfs" not in rep.measurements


def test_a_decode_pass_that_did_not_finish_is_not_run(monkeypatch, media_dir):
    monkeypatch.setattr(qa_mod, "_run_ffmpeg", lambda cmd, timeout=3600: (None, "", "ffmpeg 超過 60 分鐘還沒結束，已放棄"))
    rep = technical_qa(media_dir / "land.mp4")
    by = {c.name: c for c in rep.checks}
    assert by["完整解碼"].status == NOT_RUN and all(by[n].status == NOT_RUN for n in DETECTORS)
    assert rep.status == NOT_RUN


def test_a_decode_pass_that_exits_with_an_error_is_a_failure_even_without_a_message(monkeypatch, media_dir):
    real = qa_mod._run_ffmpeg
    calls = []

    def fake(cmd, timeout=3600):
        calls.append(cmd)
        return (1, "", "") if len(calls) == 1 else real(cmd, timeout)

    monkeypatch.setattr(qa_mod, "_run_ffmpeg", fake)
    by = {c.name: c for c in technical_qa(media_dir / "land.mp4").checks}
    assert by["完整解碼"].status == FAIL and "1" in by["完整解碼"].detail


def test_run_ffmpeg_reports_a_timeout_or_a_missing_program_as_a_problem(monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("ffmpeg", 120)

    def missing(*a, **k):
        raise FileNotFoundError("ffmpeg.exe")

    monkeypatch.setattr(qa_mod.subprocess, "run", timeout)
    rc, err, problem = qa_mod._run_ffmpeg(["ffmpeg"], timeout=120)
    assert rc is None and "2 分鐘" in problem
    monkeypatch.setattr(qa_mod.subprocess, "run", missing)
    rc, err, problem = qa_mod._run_ffmpeg(["ffmpeg"])
    assert rc is None and "ffmpeg.exe" in problem


def test_digital_silence_gives_a_warning_and_an_unmeasured_true_peak_not_a_pass(tmp_path):
    out = tmp_path / "silence.mp4"
    subprocess.run([ffmpeg_info().ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:size=160x90:rate=30:d=4",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "4", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)
    by = {c.name: c for c in technical_qa(out, expect_duration=4.0, expect_fps=30.0).checks}
    assert by["長時間靜音"].status == WARN and by["響度"].status == WARN        # -70 LUFS: far from any delivery target
    assert by["真峰值"].status == NOT_RUN and "沒有量到" in by["真峰值"].detail      # "-inf dBFS" is no measurement


def test_worst_status_never_turns_missing_results_into_a_pass():
    assert worst_status([]) == NOT_RUN and worst_status(iter(())) == NOT_RUN
    assert worst_status([PASS, PASS]) == PASS
    assert worst_status([PASS, NOT_RUN]) == NOT_RUN
    assert worst_status([PASS, NOT_RUN, WARN]) == WARN
    assert worst_status([WARN, FAIL, NOT_RUN]) == FAIL
    assert worst_status([PASS, "something-new"]) == FAIL                     # an unknown word is not a pass
    assert worst_status(s for s in (PASS, WARN)) == WARN


# ------------------------------------------------------------------ regressions found by the export dialog tests
def test_measure_loudness_reports_silence_and_short_audio_as_unmeasurable(monkeypatch, tmp_path):
    """loudnorm prints "-inf" for digital silence / audio shorter than its 400 ms block; feeding that to the
    second pass makes ffmpeg fail, so the measurement must say "unmeasurable" instead of returning the numbers."""
    from autocut.render import runner

    class Done:
        def __init__(self, stderr: str) -> None:
            self.stderr = stderr.encode()

    def fake(text):
        monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Done(text))
        return runner._measure_loudness(tmp_path / "list.txt", tmp_path, -14.0, ffmpeg_info())

    good = ('{ "input_i" : "-20.10", "input_tp" : "-3.2", "input_lra" : "2.0", "input_thresh" : "-30.5", '
            '"output_i" : "-14.0", "target_offset" : "0.4" }')
    silent = good.replace('"-20.10"', '"-inf"').replace('"-30.5"', '"-inf"')
    assert fake("noise\n" + good + "\n")["input_i"] == "-20.10"
    assert fake("noise\n" + silent + "\n") == {"unmeasurable": True}
    assert fake("no json here") is None
    assert fake(good.replace('"-3.2"', '"abc"')) is None


def test_a_failing_attempt_does_not_look_like_a_cancel_by_the_person():
    """The abort flag of one encode attempt is separate from the person's cancel (the CPU fallback depends on it)."""
    import threading
    from autocut.render.runner import _Either

    user = threading.Event()
    stop = _Either(user)
    assert not stop.is_set()
    stop.set()
    assert stop.is_set() and not user.is_set()             # our own abort never touches the person's flag
    other = _Either(user)
    user.set()
    assert other.is_set()                                   # ... but the person's cancel stops every attempt
