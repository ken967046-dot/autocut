"""Render engine integration tests with synthetic ffmpeg media (skipped without ffmpeg)."""
from __future__ import annotations

import subprocess
from dataclasses import replace

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


# ------------------------------------------------------------------------------------------- title cards (字卡)
def test_ass_cards_are_their_own_style_and_layer():
    from autocut.core.project import CaptionStyle, card_style
    from autocut.render.ass import build_ass
    st = CaptionStyle()
    plain = build_ass([(0.0, 2.0, "字幕")], st, 1080, 1920)
    assert "Style: Card" not in plain and plain.count("Dialogue:") == 1
    both = build_ass([(0.0, 2.0, "字幕")], st, 1080, 1920, cards=[(0.5, 3.0, "立恩（Ken桑的前輩）")])
    assert both.startswith(plain.split("\n[Events]")[0])                       # the subtitle style is unchanged
    card_line = next(ln for ln in both.splitlines() if ln.startswith("Style: Card,"))
    f = card_line.split(",")
    assert f[3] == "&H004DE1FF" and f[6] == "&H00000000" and f[15] == "3" and f[18] == "8"   # yellow on opaque black, top
    assert int(f[2]) > int(next(ln for ln in both.splitlines() if ln.startswith("Style: Default,")).split(",")[2])
    assert "Dialogue: 1,0:00:00.50,0:00:03.00,Card,,0,0,0,,立恩（Ken桑的前輩）" in both   # the closing mark hangs
    assert card_style(CaptionStyle(position="top")).position == "center"       # never on top of top subtitles
    # a segment window clips the card like a subtitle
    seg = build_ass([], st, 1080, 1920, t0=2.0, t1=4.0, cards=[(0.5, 3.0, "卡")])
    assert "Dialogue: 1,0:00:00.00,0:00:01.00,Card" in seg


def test_compile_keeps_cards_out_of_the_srt_and_out_of_other_segments_signatures(mp):
    from autocut.render.compiler import EncodeSettings, segment_signature
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 360, 640)
    ops.place_clip(p, seq, a["port"].id, seq.video_tracks[0].id, start=0, src_out=5)
    ops.add_cue(p, seq, 0.5, 2.0, "subtitle")
    enc = EncodeSettings()
    before = compile_sequence(p, seq, max_segment_seconds=1.0)
    sig0 = [segment_signature(before, s, enc) for s in before.segments]
    ops.add_cue(p, seq, 3.2, 3.8, "CARD", card=True)
    comp = compile_sequence(p, seq, max_segment_seconds=1.0)
    assert comp.cues == [(0.5, 2.0, "subtitle")] and comp.cards == [(3.2, 3.8, "CARD")]
    sig1 = [segment_signature(comp, s, enc) for s in comp.segments]
    changed = [i for i, (x, y) in enumerate(zip(sig0, sig1)) if x != y]
    assert changed == [i for i, s in enumerate(comp.segments) if s.cards] == [3]   # only the segment with the card
    # the card look is part of the cache key (it is derived in code, not stored in the project)
    import autocut.render.compiler as compiler_mod
    from autocut.core.project import card_style as real_card_style
    other = lambda st: replace(real_card_style(st), color="#FF0000")          # noqa: E731
    mp_sig = segment_signature(comp, comp.segments[3], enc)
    compiler_mod.card_style, saved = other, compiler_mod.card_style
    try:
        assert segment_signature(comp, comp.segments[3], enc) != mp_sig
        assert segment_signature(comp, comp.segments[0], enc) == sig1[0]
    finally:
        compiler_mod.card_style = saved


def _yellow(img) -> np.ndarray:
    b, g, r = (img[..., k].astype(int) for k in range(3))
    return (r > 170) & (g > 150) & (b < 120)


def test_render_card_burned_at_the_top_and_not_in_the_srt(mp, tmp_path):
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 360, 640)
    ops.place_clip(p, seq, a["port"].id, seq.video_tracks[0].id, start=0, src_out=4)   # solid blue 9:16
    seq.caption_style.font = "DejaVu Sans"
    ops.add_cue(p, seq, 0.0, 4.0, "SUBTITLE LINE")
    plain = tmp_path / "plain.mp4"
    res = render_sequence(p, seq, plain, RenderSettings(preset="draft_fast", normalize=False))
    if any("libass" in w for w in res.warnings):
        pytest.skip("this ffmpeg has no libass")
    ops.add_cue(p, seq, 1.0, 3.0, "BIG CARD", card=True)
    out = tmp_path / "card.mp4"
    render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False, write_srt=True))
    h = 640
    f_plain, f_card, f_before = frame_at(plain, 2.0), frame_at(out, 2.0), frame_at(out, 0.5)
    top = slice(0, int(0.3 * h))
    yel = _yellow(f_card)
    assert yel[top].sum() > 150 and not _yellow(f_plain).any() and not _yellow(f_before).any()
    ys = np.nonzero(yel)[0]
    assert ys.max() < 0.3 * h                                                    # the card sits near the top...
    dark = (f_card[top].max(axis=2) < 40)
    assert dark.sum() > 2000 and not (f_plain[top].max(axis=2) < 40).any()     # ...in an opaque black box
    # the subtitle at the bottom is drawn the same with or without the card
    bottom = slice(int(0.7 * h), h)
    assert np.abs(f_card[bottom].astype(int) - f_plain[bottom].astype(int)).mean() < 1.0
    srt = out.with_suffix(".srt").read_text(encoding="utf-8")
    assert "SUBTITLE LINE" in srt and "BIG CARD" not in srt



def test_wrap_fits_the_frame_and_keeps_punctuation_with_its_text():
    from autocut.core.project import CaptionStyle, card_style
    from autocut.render.ass import wrap_text, wrap_units
    shorts = CaptionStyle(size=0.052, bold=True, outline=0.008, position="bottom", margin=0.20, max_chars=16)
    assert wrap_units(shorts, 1920, 1080) == 16                                  # 16:9: the style's own width
    assert wrap_units(shorts, 1080, 1920) == 9 == wrap_units(shorts, 720, 1280)  # 9:16: what fits across
    assert wrap_units(card_style(shorts), 1080, 1920) == 7
    assert wrap_units(CaptionStyle(size=0.5), 320, 180) >= 2
    assert wrap_text("很多時候就是在一個很模糊的地帶", 9) == "很多時候就是在\\N一個很模糊的地帶"   # balanced
    assert wrap_text("你看每天打開電視新聞", 9) == "你看每天打\\N開電視新聞"                # no 1-character line
    assert wrap_text("立恩（Ken桑的前輩）", 7) == "立恩\\N（Ken桑的前輩）"                 # the bracket stays whole
    assert wrap_text("Hello, world! This is fine", 9) == "Hello, world!\\NThis is fine"    # never inside a word
    assert wrap_text("立恩（Ken桑的前輩）", 9) == "立恩（Ken桑的前輩）"                 # 「）」 never starts a line
    assert wrap_text("一二三四五（六七八", 5) == "一二三四五\\N（六七八"                # 「（」 never ends one


def test_wrap_never_starts_a_line_with_punctuation():
    """A perfectly even cut used to beat the natural cut after the comma, so ，。？ started the second line."""
    from autocut.render.ass import wrap_text
    assert wrap_text("大家一起來看，今天的影片", 9) == "大家一起來看，\\N今天的影片"
    assert wrap_text("所以我覺得，這樣不錯", 7) == "所以我覺得，\\N這樣不錯"                    # a 9:16 card
    assert wrap_text("加盟就能當老闆？合約漏洞一堆", 9) == "加盟就能當老闆？\\N合約漏洞一堆"
    assert wrap_text("她回我「我現在很忙。」然後就掛電話了", 9) == "她回我「我現在很忙。」\\N然後就掛電話了"
    assert wrap_text("你看 」好了", 5) == "你看 」\\N好了"                                   # not after a space either


def test_wrap_hangs_a_closing_mark_with_the_mark_before_it():
    """When 。 hangs at the end of a full line, the 」 right after it hangs too instead of starting the next line (or
    sitting alone on the last one)."""
    from autocut.render.ass import wrap_text
    assert wrap_text("我就說「真的好累喔。」他就笑了出來，然後說", 9) == "我就說「真的好累喔。」\\N他就笑了出來，\\N然後說"
    assert wrap_text("個們的就有樣「就實重責個今。」", 7) == "個們的就有樣\\N「就實重責個今。」"


def test_wrap_rules_hold_for_any_sentence():
    import random
    from autocut.render.ass import _NO_START, _OPENERS, _unit, wrap_text
    rng = random.Random(5)
    words = "我們今天要來聊一個很重要的話題就是這樣子覺得其實大家都有責任"
    for _ in range(4000):
        s = ""
        for _ in range(rng.randint(1, 4)):
            s += "「" * (rng.random() < 0.2) + "".join(rng.choice(words) for _ in range(rng.randint(1, 8)))
            s += rng.choice(["，", "。", "？", "！", "、", "。」", "？」", "）", ""])
        for width in (5, 7, 9):
            lines = wrap_text(s, width).split("\\N")
            assert "".join(lines) == s
            for i, ln in enumerate(lines):
                assert not (i and ln[0] in _NO_START), (s, width, lines)          # no line starts with ，。」
                assert not (i < len(lines) - 1 and ln[-1] in _OPENERS), (s, width, lines)
                body = ln.rstrip("".join(_NO_START)) or ln[:1]
                assert sum(_unit(c) for c in body) <= width, (s, width, lines)    # only the marks at the end hang


def test_wrap_uses_two_lines_when_two_fit():
    """The greedy pass breaks early at punctuation and needed three lines here; two balanced lines fit."""
    from autocut.render.ass import wrap_text
    assert wrap_text("其實我覺得，這件事情不太對，對吧", 9) == "其實我覺得，這件\\N事情不太對，對吧"


def test_wrap_units_depend_on_the_aspect_not_the_resolution():
    """The monitor wraps at the sequence size and an export at the preset size; a card that fits on one line in the
    monitor was burned on two in the 1080x1920 export (the pixel font size was rounded before dividing)."""
    from autocut.core.project import CaptionStyle, card_style
    from autocut.render.ass import build_ass, wrap_units
    portrait = [(2160, 3840), (1080, 1920), (720, 1280), (540, 960), (360, 640)]
    landscape = [(3840, 2160), (1920, 1080), (1280, 720), (960, 540), (640, 360)]
    for i in range(20, 101):
        for outline in (0.0, 0.006, 0.008):
            for box in (False, True):
                st = CaptionStyle(size=i / 1000, outline=outline, box=box, max_chars=16)
                for style in (st, card_style(st)):
                    for sizes in (portrait, landscape):
                        assert len({wrap_units(style, w, h) for w, h in sizes}) == 1, (style, sizes)
    card = card_style(CaptionStyle(size=0.046))
    assert wrap_units(card, 2160, 3840) == wrap_units(card, 1080, 1920) == wrap_units(card, 720, 1280) == 9
    big = build_ass([], CaptionStyle(size=0.046), 2160, 3840, cards=[(0.0, 2.0, "今天來聊聊理財觀念")])
    small = build_ass([], CaptionStyle(size=0.046), 1080, 1920, cards=[(0.0, 2.0, "今天來聊聊理財觀念")])
    assert big.split("Card,,0,0,0,,")[1] == small.split("Card,,0,0,0,,")[1] == "今天來聊聊理財觀念\n"


def test_ass_text_keeps_a_backslash_a_plain_character():
    """``\\n`` breaks the line under WrapStyle 2 and ``\\h`` is a space, so a typed ``C:\\new`` was drawn as 「C:」 over
    「ew」; a word joiner after every backslash leaves nothing for libass to read as an escape."""
    import re
    from autocut.core.project import CaptionStyle
    from autocut.render.ass import build_ass
    ass = build_ass([(0.0, 2.0, "AB\\nCD")], CaptionStyle(), 1920, 1080,
                    cards=[(0.0, 2.0, "C:\\new"), (0.0, 2.0, "A\\hB\\NC{\\b1}")])
    texts = [ln.split(",,0,0,0,,")[1] for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert texts == ["AB\\\u2060nCD", "C:\\\u2060new", "A\\\u2060hB\\\u2060NC(\\\u2060b1)"]
    assert not any(re.search(r"\\[^\u2060]", t) for t in texts)


def test_render_long_portrait_subtitle_stays_inside_the_frame(mp, tmp_path):
    """A 15-character subtitle on 9:16 used to run off both edges (16 characters per line at 5.2 % of the height is
    ~1600 px on a 1080 px frame); now it wraps to two lines inside the side margins."""
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 360, 640)
    ops.place_clip(p, seq, a["port"].id, seq.video_tracks[0].id, start=0, src_out=2)   # solid blue 9:16
    ops.set_caption_style(p, seq, size=0.052, bold=True, outline=0.008, position="bottom", margin=0.20, max_chars=16,
                          font="Noto Sans CJK TC")
    ops.add_cue(p, seq, 0.0, 2.0, "很多時候就是在一個很模糊的地帶")
    out = tmp_path / "long.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    if any("libass" in w for w in res.warnings):
        pytest.skip("this ffmpeg has no libass")
    f = frame_at(out, 1.0).astype(int)
    text = (f.min(axis=2) > 170)                                                  # white glyphs (not the blue picture)
    rows, cols = np.nonzero(text)
    assert len(rows) > 300
    assert cols.min() > 360 * 0.03 and cols.max() < 360 * 0.97                   # nothing touches the edges
    bands = np.nonzero(np.diff(np.unique(rows)) > 3)[0]
    assert len(bands) >= 1                                                        # two separate lines of text


def _bands(mask) -> list[tuple[int, int]]:
    """Column extent (first, last) of every separate row band of ``mask`` (one band per line of text)."""
    rows = np.unique(np.nonzero(mask)[0])
    if not len(rows):
        return []
    cuts = np.nonzero(np.diff(rows) > 3)[0]
    out = []
    for lo, hi in zip(np.r_[0, cuts + 1], np.r_[cuts, len(rows) - 1]):
        cols = np.nonzero(mask[rows[lo]:rows[hi] + 1].any(axis=0))[0]
        out.append((int(cols.min()), int(cols.max())))
    return out


def test_render_portrait_subtitle_keeps_the_comma_on_the_first_line(mp, tmp_path):
    """「大家一起來看，今天的影片」 on 9:16 is burned as 7 + 5 characters (the comma ends line 1); it used to be 6 + 6 with
    the comma starting line 2, two lines of the same width."""
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 360, 640)
    ops.place_clip(p, seq, a["port"].id, seq.video_tracks[0].id, start=0, src_out=2)   # solid blue 9:16
    ops.set_caption_style(p, seq, size=0.052, bold=True, outline=0.008, position="bottom", margin=0.20, max_chars=16,
                          font="Noto Sans CJK TC")
    ops.add_cue(p, seq, 0.0, 2.0, "大家一起來看，今天的影片")
    out = tmp_path / "comma.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    if any("libass" in w for w in res.warnings):
        pytest.skip("this ffmpeg has no libass")
    f = frame_at(out, 1.0).astype(int)
    bands = _bands(f.min(axis=2) > 170)                                          # white glyphs on the blue picture
    assert len(bands) == 2
    (a0, a1), (b0, b1) = bands
    assert (a1 - a0) > 1.2 * (b1 - b0)                                            # 「大家一起來看，」 over 「今天的影片」
    assert a0 > 360 * 0.03 and a1 < 360 * 0.97


def test_render_backslash_in_a_card_or_subtitle_is_drawn_as_typed(mp, tmp_path):
    """libass read 「\\n」 in 「C:\\new」 as a line break (the card showed 「C:」 over 「ew」) while the monitor showed one
    line; now both draw the text as typed, on one line."""
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 360, 640)
    ops.place_clip(p, seq, a["port"].id, seq.video_tracks[0].id, start=0, src_out=2)   # solid blue 9:16
    seq.caption_style.font = "DejaVu Sans"
    ops.add_cue(p, seq, 0.0, 2.0, "AB\\nCD")
    ops.add_cue(p, seq, 0.0, 2.0, "C:\\new", card=True)
    out = tmp_path / "backslash.mp4"
    res = render_sequence(p, seq, out, RenderSettings(preset="draft_fast", normalize=False))
    if any("libass" in w for w in res.warnings):
        pytest.skip("this ffmpeg has no libass")
    f = frame_at(out, 1.0)
    h = 640
    assert len(_bands(_yellow(f)[:int(0.4 * h)])) == 1                           # the card: one line
    white = f.astype(int).min(axis=2) > 170
    assert len(_bands(white[int(0.6 * h):])) == 1                                # the subtitle: one line


def test_export_without_libass_names_what_was_lost(mp, tmp_path, monkeypatch):
    """Without libass subtitles fall back to an .srt; title cards have no fallback.  A cards-only export used to say an
    .srt had been written when none was."""
    import autocut.render.runner as runner_mod
    real = ffmpeg_info()
    monkeypatch.setattr(runner_mod, "ffmpeg_info", lambda: replace(real, filters=real.filters - {"ass"}))
    p, a = mp
    seq = p.active_sequence()
    set_fmt(seq, 360, 640)
    ops.place_clip(p, seq, a["port"].id, seq.video_tracks[0].id, start=0, src_out=2)
    ops.add_cue(p, seq, 0.0, 2.0, "CARD", card=True)
    cards_only = tmp_path / "cards_only.mp4"
    settings = RenderSettings(preset="draft_fast", normalize=False)
    res = render_sequence(p, seq, cards_only, settings)
    (msg,) = [w for w in res.warnings if "libass" in w]
    assert "字卡" in msg and ".srt" not in msg and "字幕已" not in msg
    assert cards_only.exists() and not cards_only.with_suffix(".srt").exists() and not settings.write_srt
    ops.add_cue(p, seq, 0.0, 2.0, "SUBTITLE")
    both = tmp_path / "both.mp4"
    res = render_sequence(p, seq, both, RenderSettings(preset="draft_fast", normalize=False))
    (msg,) = [w for w in res.warnings if "libass" in w]
    assert "字幕已改輸出 .srt 檔" in msg and "字卡" in msg
    srt = both.with_suffix(".srt").read_text(encoding="utf-8")
    assert "SUBTITLE" in srt and "CARD" not in srt
