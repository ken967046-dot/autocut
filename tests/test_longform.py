"""Long-video editor: highlights + filler words + speaker close-ups (fake evidence, plus one real ffmpeg pass where the
"faces" are given by a stand-in detector -- there are no real faces in synthetic media)."""
from __future__ import annotations

import shutil
import subprocess
import threading

import numpy as np
import pytest

from autocut.analysis import speakers as spk
from autocut.analysis.evidence import Evidence
from autocut.autoedit import highlights as hl
from autocut.autoedit import longform as lf
from autocut.autoedit.api import AutoEditError, Cancelled, apply_plan, audit_plan, propose_longform
from autocut.autoedit.config import LongformParams
from autocut.core.editor import Editor
from autocut.core.project import Asset, Project, Sequence, validate_sequence
from autocut.core.subtitles import to_srt
from autocut.util import fingerprint_file

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
AFPS, VFPS = 20.0, 5.0
DULL = "今天天氣不錯我們聊聊日常"
HOT = "但是沒想到我竟然賠了300萬"
LEFT, RIGHT = [0.15, 0.3, 0.12, 0.22], [0.7, 0.3, 0.12, 0.22]


# ---------------------------------------------------------------------------------------------- fakes
def podcast_ev(dur=120.0, hot=(60.0, 80.0), words=True):
    """Continuous talk in 4 s sentences with 1 s pauses; each sentence starts with 「嗯」 (a filler word)."""
    ev = Evidence.empty("a1", dur, width=1920, height=1080, has_audio=True, audio_fps=AFPS, visual_fps=VFPS, fps=30.0)
    na = int(dur * AFPS)
    voice, rms = np.zeros(na), np.full(na, -60.0)
    sil, segs = [], []
    t = 0.0
    while t + 5.0 <= dur + 1e-9:
        voice[int(t * AFPS):int((t + 4.0) * AFPS)] = 1.0
        rms[int(t * AFPS):int((t + 4.0) * AFPS)] = -28.0
        sil.append([t + 4.0, t + 5.0])
        text = HOT if hot[0] <= t < hot[1] else DULL
        seg = {"start": t, "end": t + 4.0, "text": "嗯，" + text + "。"}
        if words:
            seg["words"] = [{"start": t, "end": t + 0.4, "word": "嗯"}] + [
                {"start": t + 0.5 + 0.35 * k, "end": t + 0.5 + 0.35 * k + 0.3, "word": ch} for k, ch in enumerate(text[:10])]
        segs.append(seg)
        t += 5.0
    ev.voice, ev.rms_db, ev.silences, ev.speech = voice.tolist(), rms.tolist(), sil, segs
    nv = int(dur * VFPS)
    ev.motion, ev.sharp, ev.bright = [0.02] * nv, [100.0] * nv, [0.5] * nv
    ev.faces, ev.face_sample_mask = [[LEFT, RIGHT]] * nv, [1] * nv
    ev.motion_cx = ev.motion_cy = [-1.0] * nv
    return ev


def two_people(dur=120.0, who=lambda t: 0 if t < 30 else 1, fps=5.0, present=2):
    n = int(dur * fps)
    acts = [[], []]
    for i in range(n):
        w = who(i / fps)
        for j in range(2):
            if j >= present:
                acts[j].append(-1.0)
            else:
                acts[j].append(8.0 if w == j else 1.0)
    sp = spk.SpeakerEvidence("a1", "fp", dur, fps)
    sp.tracks = [{"id": 0, "box": LEFT, "share": 1.0}, {"id": 1, "box": RIGHT, "share": 1.0}][:max(present, 1)]
    sp.boxes = [[LEFT] * n, [RIGHT if present > 1 else None] * n][:max(present, 1)]
    sp.activity = acts[:max(present, 1)]
    return sp


def proj_with(path: str, dur=120.0) -> Project:
    proj = Project.new("t")
    proj.assets["a1"] = Asset(id="a1", path=path, name="podcast.mp4", kind="video", duration=dur, width=1920,
                              height=1080, fps=30.0, has_video=True, has_audio=True, fingerprint=fingerprint_file(path))
    return proj


@pytest.fixture
def video(tmp_path):
    f = tmp_path / "podcast.mp4"
    f.write_bytes(b"x")
    return str(f)


@pytest.fixture
def fake(monkeypatch):
    class H:
        ev = None
        sp = None
    h = H()
    monkeypatch.setattr("autocut.analysis.pipeline.analyze_asset", lambda asset, params=None, **kw: h.ev)
    monkeypatch.setattr("autocut.analysis.speakers.analyze_speakers", lambda asset, params=None, **kw: h.sp)
    return h


def clips_of(plan):
    return Sequence.from_dict(plan.sequences[0]).video_tracks[0].clips


# ---------------------------------------------------------------------------------------------- speakers
def test_mouth_activity_follows_who_is_talking_on_synthetic_frames():
    rng = np.random.default_rng(1)
    params = spk.SpeakerParams(fps=5.0, detect_every=1)
    W, H = 320, 180
    boxes = [[0.1, 0.2, 0.2, 0.4], [0.6, 0.2, 0.2, 0.4]]
    ana = spk.SpeakerAnalyzer(params, detect=lambda g: [list(b) for b in boxes])
    for i in range(60):                                          # 12 s: A talks 0-6 s, B 6-12 s
        f = np.full((H, W), 120, np.uint8)
        talker = 0 if i < 30 else 1
        x, y, w, h = boxes[talker]
        y0, y1 = int((y + 0.65 * h) * H), int((y + 1.0 * h) * H)
        x0, x1 = int((x + 0.25 * w) * W), int((x + 0.75 * w) * W)
        f[y0:y1, x0:x1] = rng.integers(0, 255, (y1 - y0, x1 - x0))
        f = np.clip(f.astype(int) + (i % 3), 0, 255).astype(np.uint8)   # a global flicker everyone shares
        ana.push(f)
    ev = ana.finish("a", "fp", 12.0)
    assert len(ev.tracks) == 2 and ev.tracks[0]["box"][0] < ev.tracks[1]["box"][0]
    turns = spk.speaker_turns(ev, lambda t: True)
    assert [t.track for t in turns] == [0, 1]
    assert turns[0].start <= 0.6 and 5.0 <= turns[0].end <= 7.0 and turns[1].end >= 11.5
    assert spk.visible_tracks(ev, 0, 12) == [0, 1]
    assert spk.track_box(ev, 1, 6, 12) == pytest.approx(boxes[1])


def test_one_person_or_nobody_means_no_turns_and_an_honest_note():
    ana = spk.SpeakerAnalyzer(spk.SpeakerParams(), detect=lambda g: [[0.4, 0.2, 0.2, 0.4]])
    for _ in range(20):
        ana.push(np.zeros((90, 160), np.uint8))
    ev = ana.finish()
    assert len(ev.tracks) == 1 and "只偵測到 1 個人" in ev.notes[0]
    assert spk.speaker_turns(ev, lambda t: True) == []
    ana2 = spk.SpeakerAnalyzer(spk.SpeakerParams(), detect=lambda g: [])
    ana2.push(np.zeros((90, 160), np.uint8))
    assert "沒有偵測到" in ana2.finish().notes[0]


def test_silence_and_ambiguous_mouths_are_nobody():
    sp = two_people(20.0, who=lambda t: 0)
    sp.activity[1] = [7.5] * len(sp.activity[1])                  # both mouths move the same: unknown
    assert spk.speaker_turns(sp, lambda t: True) == []
    sp2 = two_people(20.0, who=lambda t: 0)
    assert spk.speaker_turns(sp2, lambda t: False) == []          # nobody talks: nobody is the speaker


# ---------------------------------------------------------------------------------------------- selection
def test_select_units_takes_the_best_plus_setups_and_bridges():
    spans = [(0, 4), (4.5, 5.5), (6, 10), (10.5, 11.3), (11.8, 15.8), (16.5, 20.5), (21, 25)]
    u = [hl.Sentence(a, b, "x", score=s) for (a, b), s in zip(spans, [0.1, 0.2, 0.9, 0.1, 0.8, 0.1, 0.1])]
    u[1].tags = ["question"]
    chosen, why = lf.select_units(u, 8.0)                 # 4 + 4 = 8 s, then 1 s set-up + 0.8 s bridge (<= 125 %)
    assert {2, 4} <= chosen and why[2] == "best" and why[4] == "best"
    assert why.get(1) == "setup" and why.get(3) in ("setup", "bridge")
    ranges = lf.kept_ranges(u, chosen, 35.0)
    assert len(ranges) == 1 and ranges[0][0] < 4.5 and 15.8 < ranges[0][1] < 16.5
    assert lf.select_units(u, 3.0)[0] == {2}                   # no budget left for context


# ---------------------------------------------------------------------------------------------- the plan
def test_highlights_mode_keeps_the_story_cuts_fillers_and_zooms_into_the_speaker(video, fake):
    fake.ev = podcast_ev()
    fake.sp = two_people(who=lambda t: 0 if t < 62 else 1)
    proj = proj_with(video)
    p = LongformParams(keep_ratio=0.3)
    plan = propose_longform(proj, "a1", p)
    assert plan.kind == "longform" and len(plan.sequences) == 1
    seq = Sequence.from_dict(plan.sequences[0])
    assert (seq.width, seq.height, seq.fps) == (1920, 1080, 30.0)
    assert not [i for i in validate_sequence(proj, seq) if i.level == "error"]
    clips = seq.video_tracks[0].clips
    kept = sum(c.src_out - c.src_in for c in clips)
    assert kept < 0.4 * 120
    # the hot stretch (60-80 s) is in; most of the small talk is not
    hot_kept = sum(max(0, min(c.src_out, 80) - max(c.src_in, 60)) for c in clips)
    assert hot_kept > 10.0
    # every 「嗯」 inside the kept stretches is cut out (word timings exist)
    for c in clips:
        for s in fake.ev.speech:
            w = s["words"][0]
            assert not (c.src_in < (w["start"] + w["end"]) / 2 < c.src_out), (c.src_in, c.src_out, w)
    assert any(d["kind"] == "filler" for d in plan.decisions)
    # speaker close-ups: zoomed clips sit on a speaker, wide clips are the two-shot
    ups = [c for c in clips if c.transform.zoom > 1.0]
    assert ups and all(c.duration >= 1.0 for c in ups)
    for c in ups:
        assert 1.2 <= c.transform.zoom <= 3.0
        assert abs(c.transform.cx - (RIGHT[0] + RIGHT[2] / 2)) < 0.1 or abs(c.transform.cx - (LEFT[0] + LEFT[2] / 2)) < 0.15
    zoom_d = [d for d in plan.decisions if d["kind"] == "zoom"]
    assert zoom_d and "連續說話" in zoom_d[0]["why"]
    # deterministic and pure
    assert propose_longform(proj, "a1", p).sha256() == plan.sha256() and proj.revision == 0
    # audit -> apply (one undo step) -> human review pending
    rec = audit_plan(plan, proj)
    assert rec.accepted, [f.message for f in rec.findings if f.severity == "error"]
    ed = Editor(proj)
    n_before = len(proj.sequences)
    apply_plan(ed, plan, rec)
    assert len(proj.sequences) == n_before + 1 and proj.meta["ai_runs"][-1]["kind"] == "longform"
    ed.undo()
    assert len(proj.sequences) == n_before


def test_long_turn_goes_back_to_wide_and_punches_in_on_the_hot_sentence(video, fake):
    fake.ev = podcast_ev(hot=(60.0, 65.0))
    fake.sp = two_people(who=lambda t: 1)                             # B talks for two minutes
    plan = propose_longform(proj_with(video), "a1", LongformParams(mode="clean", zoom_max=12.0, zoom_back_min=2.0))
    clips = clips_of(plan)
    zooms = [c.transform.zoom > 1.0 for c in clips]
    assert True in zooms and False in zooms                           # alternates, never one 2-minute close-up
    longest = max((c.duration for c in clips if c.transform.zoom > 1.0), default=0)
    assert longest <= 12.0 + 5.0                                      # a sentence may run past the cap
    hot = [c for c in clips if c.src_in < 64.0 and c.src_out > 60.5 and c.transform.zoom > 1.0]
    assert hot and max(c.transform.zoom for c in hot) >= max(c.transform.zoom for c in clips if c.src_out < 50) - 1e-6


def test_clean_mode_keeps_all_talk_and_one_person_gets_no_zoom(video, fake):
    fake.ev = podcast_ev()
    fake.sp = two_people(present=1)
    plan = propose_longform(proj_with(video), "a1", LongformParams(mode="clean"))
    clips = clips_of(plan)
    kept = sum(c.src_out - c.src_in for c in clips)
    assert 0.6 * 120 < kept < 120 - 10                                # pauses and fillers out, every sentence in
    assert all(c.transform.zoom == 1.0 for c in clips)
    assert any("只偵測到 1 個人" in n or "兩個人" in n for n in plan.notes)


def test_short_turns_are_not_zoomed(video, fake):
    fake.ev = podcast_ev()
    fake.sp = two_people(who=lambda t: int(t // 2.5) % 2)             # 2.5 s each: under the 3 s rule
    plan = propose_longform(proj_with(video), "a1", LongformParams(mode="clean"))
    assert all(c.transform.zoom == 1.0 for c in clips_of(plan))


def test_without_word_timings_or_text(video, fake):
    ev = podcast_ev(words=False)
    fake.ev, fake.sp = ev, two_people()
    plan = propose_longform(proj_with(video), "a1", LongformParams(speaker_zoom=False))
    assert any("沒有逐字時間" in n for n in plan.notes)
    ev2 = podcast_ev()
    ev2.speech = None
    fake.ev = ev2
    plan2 = propose_longform(proj_with(video), "a1", LongformParams(speaker_zoom=False, keep_ratio=0.4))
    assert any("沒有文字可以判斷重點" in n for n in plan2.notes) and clips_of(plan2)


def test_errors_and_cancel(video, fake):
    fake.ev, fake.sp = podcast_ev(), two_people()
    proj = proj_with(video)
    with pytest.raises(AutoEditError):
        propose_longform(proj, "nope")
    with pytest.raises(AutoEditError):
        propose_longform(proj, "a1", LongformParams(mode="weird"))
    with pytest.raises(AutoEditError):
        propose_longform(proj, "a1", LongformParams(keep_ratio=0.0))
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        propose_longform(proj, "a1", cancel=ev)
    silent = podcast_ev()
    silent.voice, silent.rms_db, silent.speech, silent.silences = [0.0] * 2400, [-80.0] * 2400, None, [[0.0, 120.0]]
    fake.ev = silent
    with pytest.raises(AutoEditError):
        propose_longform(proj, "a1", LongformParams(speaker_zoom=False))


# ---------------------------------------------------------------------------------------------- real ffmpeg
@pytest.mark.ffmpeg
def test_longform_on_real_media_with_stand_in_faces(tmp_path, monkeypatch):
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    W, H, fps, dur = 640, 360, 10, 24
    out = tmp_path / "talk.mp4"
    rng = np.random.default_rng(0)
    boxes = [[0.1, 0.2, 0.2, 0.4], [0.6, 0.2, 0.2, 0.4]]
    frames = []
    for i in range(dur * fps):
        f = np.full((H, W), 110, np.uint8)
        x, y, w, h = boxes[0 if i < 12 * fps else 1]
        f[int((y + 0.65 * h) * H):int((y + h) * H), int((x + 0.25 * w) * W):int((x + 0.75 * w) * W)] = \
            rng.integers(0, 255, (int((y + h) * H) - int((y + 0.65 * h) * H), int((x + 0.75 * w) * W) - int((x + 0.25 * w) * W)))
        frames.append(f)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(fps),
         "-i", "-", "-f", "lavfi", "-i", f"aevalsrc=sin(2*PI*440*t)*0.5*lt(mod(t\\,4)\\,3.2):s=44100:d={dur}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        input=b"".join(fr.tobytes() for fr in frames), check=True)
    assert proc.returncode == 0
    srt = [(4.0 * k, 4.0 * k + 3.4, (HOT if 12 <= 4 * k < 20 else DULL) + "。") for k in range(dur // 4)]
    (tmp_path / "talk.srt").write_text(to_srt(srt), encoding="utf-8")         # 「一鍵字幕」 leaves exactly this
    monkeypatch.setattr("autocut.analysis.faces.detect_faces", lambda g, min_frac=0.06: [list(b) for b in boxes])
    monkeypatch.setattr("autocut.analysis.faces.faces_available", lambda: True)
    monkeypatch.setattr("autocut.analysis.speakers.cache_dir", lambda name: tmp_path / "cache")
    from autocut.media.probe import probe_asset
    proj = Project.new("real")
    a = probe_asset(out)
    proj.assets[a.id] = a
    sp = spk.analyze_speakers(a, use_cache=False)
    assert len(sp.tracks) == 2 and sp.n >= dur * 5 - 2
    turns = spk.speaker_turns(sp, lambda t: True)
    assert [t.track for t in turns][:2] == [0, 1] and abs(turns[0].end - 12.0) < 1.0
    plan = propose_longform(proj, a.id, LongformParams(mode="clean", use_speech=False, captions="auto"))
    seq = Sequence.from_dict(plan.sequences[0])
    assert plan.brief["transcript"]["source"] == "sidecar" and seq.cues
    assert any(c.transform.zoom > 1.0 for c in seq.video_tracks[0].clips)
    rec = audit_plan(plan, proj)
    assert rec.accepted, [f.message for f in rec.findings if f.severity == "error"]
    # render the plan and MEASURE it: length = timeline, 640x360, and a close-up really is a close-up (the talking
    # mouth -- the noise patch -- fills far more of the frame than in the wide shot)
    from autocut.render.runner import RenderSettings, render_sequence
    ed = Editor(proj)
    sid = apply_plan(ed, plan, rec)[0]
    seq = next(s for s in proj.sequences if s.id == sid)
    mp4 = tmp_path / "out.mp4"
    render_sequence(proj, seq, mp4, RenderSettings(preset="draft_fast", normalize=False))
    meta = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height:format=duration", "-of", "default=nw=1", str(mp4)],
                          capture_output=True, text=True, check=True).stdout
    kv = dict(line.split("=") for line in meta.split())
    assert (int(kv["width"]), int(kv["height"])) == (640, 360) and abs(float(kv["duration"]) - seq.duration) < 0.15

    def frame_at(t: float) -> np.ndarray:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(mp4), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(360, 640)

    clips = seq.video_tracks[0].clips
    up = next(c for c in clips if c.transform.zoom > 1.0 and c.duration > 1.5)
    wide = next(c for c in clips if c.transform.zoom == 1.0 and c.duration > 1.5)
    # the talking mouth is the only non-grey area (subtitles live in the bottom 20 %): its area grows with zoom²
    area = lambda f: float(np.mean(np.abs(f[: int(0.8 * f.shape[0])].astype(int) - 110) > 12))
    a_up, a_wide = area(frame_at(up.start + up.duration / 2)), area(frame_at(wide.start + wide.duration / 2))
    assert a_wide > 0.005 and a_up / a_wide == pytest.approx(up.transform.zoom ** 2, rel=0.25)


# ---------------------------------------------------------------------------------- REVIEW-3 (Codex) regressions
def test_kept_ranges_never_leave_the_source():
    u = [hl.Sentence(9.0, 12.0, "字幕超出影片", score=1.0)]
    for a, b, _i in lf.kept_ranges(u, {0}, 10.0):
        assert 0.0 <= a < b <= 10.0


def test_subtitles_past_the_end_of_the_video_do_not_make_out_of_range_clips(video, fake):
    ev = podcast_ev(dur=60.0)
    ev.speech = ev.speech + [{"start": 58.0, "end": 70.0, "text": "但是沒想到我竟然賠了300萬！"}]
    fake.ev, fake.sp = ev, two_people(60.0)
    plan = propose_longform(proj_with(video, 60.0), "a1", LongformParams(keep_ratio=0.4, speaker_zoom=False))
    assert all(0.0 <= c.src_in < c.src_out <= 60.0 + 1e-6 for c in clips_of(plan))


def test_smoothing_is_safe_when_there_are_fewer_samples_than_the_window():
    sp = spk.SpeakerEvidence("a", "", 0.4, 5.0)
    sp.tracks, sp.boxes = [{}, {}], [[None, None], [None, None]]
    sp.activity = [[1.0, 2.0], [1.0, 1.0]]
    lab, conf = spk.speaker_labels(sp, lambda t: True)
    assert lab.shape == conf.shape == (2,)
    assert spk.speaker_turns(sp, lambda t: True) is not None
