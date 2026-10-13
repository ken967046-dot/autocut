"""Evidence analysis tests (spec docs/AUTOEDIT_METHOD.md §2 and §8): sound, picture, scene cuts, faces, speech, cache.

Ground truth comes from synthetic media with KNOWN content: 440 Hz bursts between silences (written with numpy so every
edge is exact), colour segments whose lengths are whole frames (so every cut time is exact), a one-frame flash, ...
Face detection can only be checked for graceful degradation and the documented box format (there are no real faces here);
speech recognition is checked with faster-whisper absent and with a stand-in module.  Set ``AUTOCUT_LONG_TESTS=1`` to also
run the 10-minute video measurement (about a minute)."""
from __future__ import annotations

import importlib.machinery
import json
import math
import os
import subprocess
import sys
import threading
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from autocut.analysis import AnalyzeParams, Evidence, analyze_asset, refine_cut
from autocut.analysis import audio as A
from autocut.analysis import faces as F
from autocut.analysis import pipeline as P
from autocut.analysis import speech as S
from autocut.analysis import visual as V
from autocut.analysis.evidence import EVIDENCE_VERSION, cache_path
from autocut.core.project import Asset
from autocut.media.ffmpeg import Cancelled, ffmpeg_info
from autocut.media.probe import probe_asset
from autocut.util import cache_dir

ffmpeg = pytest.mark.ffmpeg                         # skipped by conftest when ffmpeg is not installed
linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and resource.getrusage")
SRC = Path(__file__).resolve().parents[1] / "src"
CJK = lambda s: any("一" <= c <= "鿿" for c in s)                      # noqa: E731

BURSTS = [(1.0, 3.0), (5.0, 6.5), (9.0, 12.0), (14.0, 15.0), (17.0, 19.5)]       # where the 440 Hz tone plays (seconds)
GAPS = [(0.0, 1.0), (3.0, 5.0), (6.5, 9.0), (12.0, 14.0), (15.0, 17.0), (19.5, 22.0)]
WAV_SECONDS = 22.0
CUT_FRAMES = [75, 60, 90, 75, 100, 75]                                           # black | white | testsrc2 | grey | bars | blue
CUTS = [3.0, 5.4, 9.0, 12.0, 16.0]                                               # exact cut times at 25 fps
FPS = 25.0


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own AUTOCUT_HOME, so evidence / speech caches never leak between tests."""
    home = tmp_path / "autocut home"
    monkeypatch.setenv("AUTOCUT_HOME", str(home))
    return home


# ------------------------------------------------------------------------------------------------ synthetic media
def write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> Path:
    x = np.atleast_2d(x) if x.ndim > 1 else x[None, :]                           # (channels, samples)
    pcm = (np.clip(x.T, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def tone_bursts(sr: int = 16000, seconds: float = WAV_SECONDS, bursts=BURSTS, freq: float = 440.0, amp: float = 0.5,
                noise_db: float | None = None, seed: int = 7) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    on = np.zeros_like(t)
    for a, b in bursts:
        on[(t >= a) & (t < b)] = 1.0
    x = amp * np.sin(2 * np.pi * freq * t) * on
    if noise_db is not None:
        x = x + np.random.default_rng(seed).standard_normal(t.size) * 10 ** (noise_db / 20.0)
    return x.astype(np.float32)


def ff(*args: str) -> None:
    subprocess.run([ffmpeg_info().ffmpeg, "-y", "-loglevel", "error", *args], check=True)


def asset_of(path: Path) -> Asset:
    return probe_asset(path)


def voice_runs(ev: Evidence) -> list[tuple[float, float]]:
    return [(a / ev.audio_fps, b / ev.audio_fps) for a, b in A.runs(np.asarray(ev.voice) > 0)]


def assert_edges(found, truth, tol: float, what: str) -> None:
    assert len(found) == len(truth), (what, found, truth)
    for (a, b), (ta, tb) in zip(found, truth):
        assert abs(a - ta) <= tol and abs(b - tb) <= tol, (what, (a, b), (ta, tb))


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    try:
        ffmpeg_info()
    except Exception:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("分析 media")
    m: dict[str, Path] = {}
    # picture with known cuts (whole frames per segment, 25 fps)
    srcs = ["color=c=black", "color=c=white", "testsrc2", "color=c=gray", "smptebars", "color=c=0x101080"]
    args: list[str] = []
    for s, n in zip(srcs, CUT_FRAMES):
        sep = ":" if "=" in s else "="                      # `color=c=black:s=..` but `testsrc2=s=..`
        args += ["-f", "lavfi", "-i", f"{s}{sep}s=640x360:r=25:d={n / FPS}"]
    ff(*args, "-filter_complex", f"{''.join(f'[{i}:v]' for i in range(6))}concat=n=6:v=1:a=0,format=yuv420p[v]", "-map", "[v]",
       "-c:v", "libx264", "-preset", "veryfast", "-g", "50", str(d / "cuts.mp4"))
    m["cuts"] = d / "cuts.mp4"
    # grey with a single white frame at 5.0 s and a 3-frame white flash at 8.0 s
    ff("-f", "lavfi", "-i", "color=c=0x606060:s=320x180:r=25:d=10,format=yuv420p", "-vf",
       "drawbox=x=0:y=0:w=320:h=180:color=white:t=fill:enable='eq(n,125)+between(n,200,202)'", "-c:v", "libx264",
       "-preset", "veryfast", "-g", "25", str(d / "flash.mp4"))
    m["flash"] = d / "flash.mp4"
    # 6 s picture + sound (progress / cancel / cache tests) and a 6 s picture without sound
    ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=6", "-i", str(write_wav(d / "tone6.wav", tone_bursts(seconds=6, bursts=[(1, 3)]))),
       "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "av6.mp4"))
    m["av6"] = d / "av6.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(d / "silent6.mp4"))
    m["silent"] = d / "silent6.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=s=320x240:d=1", "-frames:v", "1", str(d / "still.png"))
    m["still"] = d / "still.png"
    ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=0.5", "-i", str(write_wav(d / "tone05.wav", tone_bursts(seconds=0.5, bursts=[(0, 0.5)]))),
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "tiny.mp4"))
    m["tiny"] = d / "tiny.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=0.04", "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       str(d / "oneframe.mp4"))
    m["oneframe"] = d / "oneframe.mp4"
    # a download cut off at 55 % (moov atom first, so the file still opens)
    ff("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=20", "-i", str(write_wav(d / "tone20.wav", tone_bursts(seconds=20, bursts=[(1, 19)]))),
       "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
       str(d / "full20.mp4"))
    raw = (d / "full20.mp4").read_bytes()
    (d / "truncated.mp4").write_bytes(raw[: int(len(raw) * 0.55)])
    m["truncated"] = d / "truncated.mp4"
    m["full20"] = d / "full20.mp4"
    (d / "zero.mp4").write_bytes(b"")
    (d / "corrupt.mp4").write_bytes(os.urandom(3000))
    (d / "no moov.mp4").write_bytes(raw[: len(raw) // 2][: -1] if False else b"\x00\x00\x00\x18ftypmp42" + os.urandom(2000))
    m.update(zero=d / "zero.mp4", corrupt=d / "corrupt.mp4", nomoov=d / "no moov.mp4")
    # sound with known bursts: clean, quiet room tone, noisy kitchen; stereo 44.1 kHz
    m["wav_clean"] = write_wav(d / "bursts clean.wav", tone_bursts())
    m["wav_quiet"] = write_wav(d / "bursts quiet.wav", tone_bursts(noise_db=-58))
    m["wav_kitchen"] = write_wav(d / "bursts kitchen.wav", tone_bursts(noise_db=-36))
    st = np.stack([tone_bursts(44100), tone_bursts(44100, amp=0.3)])
    m["wav_stereo"] = write_wav(d / "bursts stereo.wav", st, 44100)
    return m


@pytest.fixture(scope="module")
def video90(tmp_path_factory) -> Path:
    """90 s of 1280x720 25 fps picture (hard to decode on purpose) with bursts."""
    try:
        ffmpeg_info()
    except Exception:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("長片 90s")
    wav = write_wav(d / "a.wav", tone_bursts(seconds=90, bursts=[(i * 3.0, i * 3.0 + 2.0) for i in range(30)]))
    ff("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=90", "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast",
       "-crf", "30", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "v90.mp4"))
    return d / "v90.mp4"


def ffmpeg_children() -> set[int]:
    out = set()
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if d.isdigit():
            try:
                st = Path(f"/proc/{d}/stat").read_text()
                rest = st[st.rindex(")") + 2:].split()
                if int(rest[1]) == os.getpid() and "ffmpeg" in st[st.index("(") + 1:st.rindex(")")]:
                    out.add(int(d))
            except (OSError, ValueError, IndexError):
                continue
    return out


# =========================================================================== sound: the maths (no ffmpeg needed)
def test_audio_frames_do_not_depend_on_the_block_size():
    x = tone_bursts(seconds=6, bursts=[(1, 2.5), (4, 5)], noise_db=-50)
    reference = A.analyze_audio_blocks([x])
    for size in (13, 977, 8000, 16000):
        got = A.analyze_audio_blocks(np.array_split(x, range(size, len(x), size)))
        assert got.rms_db == reference.rms_db and got.voice == reference.voice and got.silences == reference.silences, size
    assert len(reference.rms_db) == int(math.ceil(6 * 20))                           # 20 frames per second
    assert reference.duration == pytest.approx(6.0)


def test_voice_rules_minimum_run_gap_filling_and_silence_length():
    fps = 20
    rms = np.full(200, -80.0)                                                       # 10 s of digital-silence-ish
    band = np.zeros(200)
    for a, b in ((20, 60), (80, 83), (100, 104), (108, 140)):                        # 2 s | 0.15 s | 0.2 s + gap 0.2 s + 1.6 s
        rms[a:b], band[a:b] = -20.0, 0.8
    f = A.features_from_frames(rms, band, fps)
    voiced = [(a / fps, b / fps) for a, b in A.runs(np.asarray(f.voice) > 0)]
    # 3 frames (0.15 s) is the minimum run, kept; 4 frames then a 4-frame gap (0.2 s): bridged into one run with the next
    assert voiced == [(1.0, 3.0), (4.0, 4.15), (5.0, 7.0)], voiced
    assert all(0.6 <= v <= 1.0 for v in f.voice if v > 0)
    sil = f.silences
    assert [1.0, 3.0] not in sil and all(b - a >= 0.3 - 1e-9 for a, b in sil)       # silences are >= 0.3 s
    assert sil[0] == [0.0, 1.0] and sil[-1] == [7.0, 10.0]
    short = A.features_from_frames(np.where(np.arange(100) % 2 == 0, -20.0, -80.0), np.full(100, 0.8), fps)
    assert not any(v > 0 for v in short.voice)                                      # 1-frame blips are not voice


def test_adaptive_thresholds_follow_the_spec():
    rng = np.random.default_rng(1)
    clean = np.concatenate([np.full(300, -80.0), np.full(100, -15.0)]) + rng.normal(0, 0.1, 400)
    floor, s_thr, v_thr = A.adaptive_thresholds(clean)                              # clean room: floor < -46 -> fixed -38 dB
    assert floor < -75 and s_thr == -38.0 and v_thr == -38.0
    noisy = np.concatenate([np.full(300, -40.0), np.full(100, -10.0)]) + rng.normal(0, 0.1, 400)
    floor, s_thr, v_thr = A.adaptive_thresholds(noisy)                              # noisy room: floor + 8 / floor + 10
    assert floor == pytest.approx(-40.0, abs=0.3) and s_thr == pytest.approx(-32.0, abs=0.3)
    assert v_thr == pytest.approx(-30.0, abs=0.3)
    loud = np.full(400, -12.0)
    assert A.adaptive_thresholds(loud)[1] == A.SILENCE_CAP_DB                       # never above -25 dB
    assert A.adaptive_thresholds(np.zeros(0)) == (A.DB_FLOOR, -38.0, -38.0)


# =========================================================================== sound: from files through ffmpeg
@ffmpeg
@pytest.mark.parametrize("key", ["wav_clean", "wav_quiet", "wav_kitchen", "wav_stereo"])
def test_voice_and_silence_edges_match_the_known_bursts(media, key):
    ev = analyze_asset(asset_of(media[key]), use_cache=False)
    assert ev.has_audio and not ev.has_video and ev.motion == [] and ev.scenes == []
    assert ev.duration == pytest.approx(WAV_SECONDS, abs=0.1)
    assert_edges(voice_runs(ev), BURSTS, 0.15, f"voice {key}")
    assert_edges([tuple(s) for s in ev.silences], GAPS, 0.15, f"silences {key}")
    loud = [v for v, f in zip(ev.rms_db, range(len(ev.rms_db))) if 2.0 < f / ev.audio_fps < 2.8]
    quiet = [v for f, v in enumerate(ev.rms_db) if 3.5 < f / ev.audio_fps < 4.5]
    assert min(loud) > max(quiet) + 20                                              # a tone at 0.5 amplitude is ~ -9 dBFS
    assert ev.voice_thr_db >= ev.silence_thr_db - 1e-6 or ev.silence_thr_db > ev.voice_thr_db   # thresholds are reported


@ffmpeg
def test_noise_floor_adapts_to_the_room(media):
    floors = {k: analyze_asset(asset_of(media[k]), use_cache=False) for k in ("wav_clean", "wav_quiet", "wav_kitchen")}
    clean, quiet, kitchen = (floors[k] for k in ("wav_clean", "wav_quiet", "wav_kitchen"))
    assert clean.noise_floor_db < quiet.noise_floor_db < kitchen.noise_floor_db
    assert kitchen.noise_floor_db == pytest.approx(-36, abs=4)
    assert kitchen.silence_thr_db > clean.silence_thr_db and kitchen.silence_thr_db == pytest.approx(kitchen.noise_floor_db + 8, abs=0.1)
    assert kitchen.voice_thr_db > kitchen.noise_floor_db + 5                          # a fixed -38 dB would call the room 'voice'


@ffmpeg
def test_loud_sounds_outside_the_speech_band_are_not_voice(tmp_path):
    sr = 16000
    t = np.arange(int(12 * sr)) / sr
    rng = np.random.default_rng(3)
    x = rng.standard_normal(t.size) * 10 ** (-60 / 20)                              # quiet room tone
    hiss = (t >= 2) & (t < 4)
    x[hiss] += rng.standard_normal(hiss.sum()) * 10 ** (-20 / 20)                    # 2 s of loud white noise
    whistle = (t >= 6) & (t < 8)
    x[whistle] += 0.4 * np.sin(2 * np.pi * 6000 * t[whistle])                        # 2 s of loud 6 kHz tone
    speechy = (t >= 9) & (t < 11)
    x[speechy] += 0.4 * np.sin(2 * np.pi * 800 * t[speechy])                         # 2 s inside the speech band
    ev = analyze_asset(asset_of(write_wav(tmp_path / "mixed.wav", x.astype(np.float32))), use_cache=False)
    assert_edges(voice_runs(ev), [(9.0, 11.0)], 0.15, "only the speech-band tone is voice")
    loud_frames = [i for i, v in enumerate(ev.rms_db) if v > -30]
    loud_nonspeech = [i for i in loud_frames if 1.9 < i / 20 < 4.1 or 5.9 < i / 20 < 8.1]
    assert len(loud_nonspeech) > 4 * 20 * 0.9                                       # the hiss and the whistle (2 s + 2 s) ARE loud ...
    assert not any(ev.voice[i] > 0 for i in range(len(ev.voice)) if 1.9 < i / 20 < 4.1 or 5.9 < i / 20 < 8.1)   # ... but not voice


# =========================================================================== picture: scenes, refine_cut, features
@ffmpeg
def test_scene_cuts_are_found_within_a_quarter_second(media):
    ev = analyze_asset(asset_of(media["cuts"]), use_cache=False)
    assert ev.has_video and not ev.has_audio and ev.silences == [] and ev.rms_db == []
    assert len(ev.scenes) == len(CUTS), (ev.scenes, CUTS)
    assert ev.scenes == sorted(ev.scenes)
    for found, truth in zip(ev.scenes, CUTS):
        assert abs(found - truth) <= 0.25, (found, truth)
    assert ev.duration == pytest.approx(19.0, abs=0.1)
    assert len(ev.motion) == pytest.approx(19.0 * 5, abs=2)                          # 5 samples per second


@ffmpeg
@pytest.mark.parametrize("offset", [-0.25, 0.0, 0.1, 0.3])
def test_refine_cut_lands_on_the_exact_frame(media, offset):
    for truth in CUTS:
        got = refine_cut(media["cuts"], truth + offset, FPS)
        assert abs(got - truth) <= 1 / FPS + 1e-6, (truth, offset, got)              # the spec asks for <= 1 frame
        assert abs(got * FPS - round(got * FPS)) < 1e-3                              # and it is on the frame grid


@ffmpeg
def test_refine_cut_falls_back_to_the_estimate(media, tmp_path):
    assert refine_cut(media["cuts"], 1.5, FPS) == 1.5                                # window holds no cut
    assert refine_cut(media["cuts"], 6.5, FPS, window=0.4) == 6.5
    assert refine_cut(tmp_path / "missing.mp4", 3.0, FPS) == 3.0                     # unreadable: never raises
    assert refine_cut(media["corrupt"], 3.0, FPS) == 3.0
    for bad_fps in (0.0, -5.0, float("nan")):
        assert refine_cut(media["cuts"], 3.0, bad_fps) == 3.0
    assert refine_cut(media["cuts"], float("nan"), FPS) != refine_cut(media["cuts"], float("nan"), FPS)   # nan in, nan out
    before = ffmpeg_children()
    refine_cut(media["cuts"], 9.0, FPS)
    assert ffmpeg_children() <= before                                              # no ffmpeg left behind


@ffmpeg
def test_flashes_are_not_scene_cuts(media):
    ev = analyze_asset(asset_of(media["flash"]), use_cache=False)
    assert ev.scenes == [], ev.scenes                                                # 1- and 3-frame flashes return to the shot


@ffmpeg
def test_picture_features_describe_the_picture_on_the_documented_time_grid(media):
    ev = analyze_asset(asset_of(media["cuts"]), AnalyzeParams(faces=False), use_cache=False)
    assert ev.visual_fps == 5.0 and len(ev.bright) == len(ev.sharp) == len(ev.motion) == len(ev.faces) == len(ev.motion_cx)
    b = ev.bright
    assert b[10] < 0.08 and b[14] < 0.08                                              # black until 3.0 s
    assert b[15] > 0.9 and b[20] > 0.9 and b[26] > 0.9                                # sample i is the frame shown at t = i/5
    assert 0.35 < b[60] < 0.65 and 0.08 < b[85] < 0.2                                 # grey (12-16 s: bars), dark blue at the end
    assert max(ev.sharp[:15]) < 1.0 and max(ev.sharp[15:27]) < 1.0                    # flat colours have no edges
    assert ev.sharp[35] > 50 * max(ev.sharp[0], 0.1)                                  # testsrc2 is full of them
    assert ev.motion[10] == 0.0 or ev.motion[10] < 0.02                               # nothing moves inside a flat shot
    for cut in CUTS:                                                                  # a cut is not subject motion
        i = int(round(cut * 5))
        assert ev.motion[i] < 0.1, (cut, ev.motion[i])
        assert ev.motion_cx[i] == -1.0 and ev.motion_cy[i] == -1.0
    assert all(0.0 <= m <= 1.0 for m in ev.motion) and all(0.0 <= v <= 1.0 for v in ev.bright)


@ffmpeg
def test_asset_without_a_known_size_is_still_analysed(media):
    a = asset_of(media["cuts"])
    a.width = a.height = 0
    ev = analyze_asset(a, use_cache=False)
    assert len(ev.scenes) == len(CUTS) and len(ev.motion) == pytest.approx(95, abs=2)


@ffmpeg
def test_rotated_phone_video_keeps_the_frame_size_contract(tmp_path):
    src = tmp_path / "landscape.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src))
    rotated = tmp_path / "手機 rotated.mp4"
    try:
        ff("-display_rotation", "90", "-i", str(src), "-c", "copy", str(rotated))
    except subprocess.CalledProcessError:
        pytest.skip("this ffmpeg cannot write a display matrix")
    a = asset_of(rotated)
    assert (a.width, a.height) == (360, 640) and a.rotation in (90, 270)             # display size, rotation applied
    ev = analyze_asset(a, use_cache=False)
    assert ev.has_video and len(ev.motion) == pytest.approx(20, abs=1) and not ev.notes[1:]
    assert all(0.0 <= v <= 1.0 for v in ev.bright)


# =========================================================================== the cache
@ffmpeg
def test_second_run_reads_the_cache(media, monkeypatch):
    a = asset_of(media["wav_clean"])
    first = analyze_asset(a)
    assert cache_path(a.fingerprint, AnalyzeParams()).is_file()
    monkeypatch.setattr(P, "_compute_base", lambda *args, **kw: pytest.fail("the cache was not used"))
    seen: list[tuple[float, str]] = []
    second = analyze_asset(a, progress=lambda f, m: seen.append((f, m)))
    assert second.sha256() == first.sha256()
    assert seen[-1] == (1.0, "分析完成（使用快取）")


@ffmpeg
def test_different_parameters_use_different_cache_files(media):
    a = asset_of(media["wav_clean"])
    p1, p2 = AnalyzeParams(), AnalyzeParams(audio_fps=10.0)
    e1, e2 = analyze_asset(a, p1), analyze_asset(a, p2)
    f1, f2 = cache_path(a.fingerprint, p1), cache_path(a.fingerprint, p2)
    assert f1 != f2 and f1.is_file() and f2.is_file()
    assert len(e1.rms_db) == 2 * len(e2.rms_db) or abs(len(e1.rms_db) - 2 * len(e2.rms_db)) <= 2
    assert e2.params["audio_fps"] == 10.0 and e1.params["audio_fps"] == 20.0
    assert {p.name for p in cache_dir("evidence").iterdir()} == {f1.name, f2.name}
    # speech options are keyed by the speech cache, not by the numbers cache
    assert cache_path(a.fingerprint, AnalyzeParams(speech=True, language="en", model_size="tiny")) == f1
    assert f1.name == f"{a.fingerprint}_v{EVIDENCE_VERSION}_{p1.cache_key()}.json"


@ffmpeg
@pytest.mark.parametrize("damage", ["garbage", "truncated", "empty", "json list", "wrong types", "short list", "other version"])
def test_damaged_cache_files_are_recomputed(media, monkeypatch, damage):
    a = asset_of(media["wav_clean"])
    good = analyze_asset(a)
    f = cache_path(a.fingerprint, AnalyzeParams())
    d = json.loads(f.read_text(encoding="utf-8"))
    if damage == "garbage":
        f.write_text("{not json at all", encoding="utf-8")
    elif damage == "truncated":
        f.write_text(f.read_text(encoding="utf-8")[:300], encoding="utf-8")
    elif damage == "empty":
        f.write_bytes(b"")
    elif damage == "json list":
        f.write_text("[1, 2, 3]", encoding="utf-8")
    elif damage == "wrong types":
        d["rms_db"] = "oops"
        f.write_text(json.dumps(d), encoding="utf-8")
    elif damage == "short list":
        d["voice"] = d["voice"][:-5]
        f.write_text(json.dumps(d), encoding="utf-8")
    else:
        d["version"] = EVIDENCE_VERSION + 1
        f.write_text(json.dumps(d), encoding="utf-8")
    calls = []
    real = P._compute_base
    monkeypatch.setattr(P, "_compute_base", lambda *args, **kw: calls.append(1) or real(*args, **kw))
    again = analyze_asset(a)
    assert calls == [1], "a damaged cache must be recomputed"
    assert again.sha256() == good.sha256()
    assert Evidence.load(f).sha256() == good.sha256()                                 # and the file was repaired
    analyze_asset(a)
    assert calls == [1]                                                               # ... so the next run is a hit again


@ffmpeg
def test_use_cache_false_neither_reads_nor_writes(media):
    a = asset_of(media["wav_clean"])
    analyze_asset(a, use_cache=False)
    assert list(cache_dir("evidence").iterdir()) == []
    analyze_asset(a)
    f = cache_path(a.fingerprint, AnalyzeParams())
    d = json.loads(f.read_text(encoding="utf-8"))
    d["noise_floor_db"] = 123.0                                                       # a valid-looking cache that differs
    f.write_text(json.dumps(d), encoding="utf-8")
    assert analyze_asset(a).noise_floor_db == 123.0                                   # hit
    assert analyze_asset(a, use_cache=False).noise_floor_db != 123.0                  # ignored


@ffmpeg
def test_cache_key_follows_the_file_not_only_the_name(media, tmp_path):
    a = asset_of(media["wav_clean"])
    other = write_wav(tmp_path / "bursts clean.wav", tone_bursts(bursts=[(2.0, 4.0)]))     # same name, different sound
    b = asset_of(other)
    assert a.fingerprint != b.fingerprint
    ev_a, ev_b = analyze_asset(a), analyze_asset(b)
    assert voice_runs(ev_a)[0][0] == pytest.approx(1.0, abs=0.15) and voice_runs(ev_b)[0][0] == pytest.approx(2.0, abs=0.15)
    a.fingerprint = ""                                                                # no fingerprint: path + size + mtime
    ev_c = analyze_asset(a)
    assert voice_runs(ev_c)[0][0] == pytest.approx(1.0, abs=0.15) and len(list(cache_dir("evidence").iterdir())) == 3


def test_evidence_round_trip_and_hash_are_stable(tmp_path):
    ev = Evidence.empty("x", 3.0, rms_db=[-20.0, -30.5], voice=[1.0, 0.0], faces=[[[0.1, 0.2, 0.3, 0.4]], []], notes=["說明"])
    ev.save(tmp_path / "e.json")
    back = Evidence.load(tmp_path / "e.json")
    assert back == ev and back.sha256() == ev.sha256()
    assert json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))["schema"] == "autocut.evidence/1"
    assert Evidence.from_dict({**ev.to_dict(), "future_field": 1}) == ev              # unknown fields are ignored
    assert AnalyzeParams.from_dict({"audio_fps": 10, "nonsense": 1}).audio_fps == 10


# =========================================================================== graceful degradation
@ffmpeg
def test_audio_only_still_image_silent_and_tiny_files(media):
    ev = analyze_asset(asset_of(media["wav_clean"]), use_cache=False)               # audio only
    assert (ev.has_audio, ev.has_video, ev.motion, ev.faces, ev.scenes) == (True, False, [], [], [])
    ev = analyze_asset(asset_of(media["still"]), use_cache=False)                   # a photo: one sample, no sound
    assert (ev.has_audio, ev.has_video, ev.rms_db, ev.silences) == (False, True, [], [])
    assert len(ev.motion) == len(ev.bright) == len(ev.sharp) == len(ev.faces) == 1 and ev.scenes == []
    ev = analyze_asset(asset_of(media["silent"]), use_cache=False)                  # video without a sound track
    assert (ev.has_audio, ev.has_video, ev.rms_db, ev.voice, ev.silences) == (False, True, [], [], [])
    assert len(ev.motion) == pytest.approx(30, abs=1)
    ev = analyze_asset(asset_of(media["tiny"]), use_cache=False)                    # half a second
    assert ev.has_audio and ev.has_video and 9 <= len(ev.rms_db) <= 12 and 2 <= len(ev.motion) <= 4
    assert ev.duration == pytest.approx(0.5, abs=0.1)
    ev = analyze_asset(asset_of(media["oneframe"]), use_cache=False)                # a single frame of video
    assert ev.has_video and len(ev.motion) == 1 and ev.motion == [0.0]


@ffmpeg
def test_unreadable_files_raise_a_chinese_runtimeerror_quickly(media, tmp_path):
    def asset_for(path: Path, **kw) -> Asset:
        return Asset(id="a", path=str(path), name=path.name, kind="video", duration=5.0, width=640, height=360, fps=25.0,
                     has_video=True, has_audio=True, **kw)

    cases = {"zero bytes": (asset_for(media["zero"]), "空的"), "corrupt": (asset_for(media["corrupt"]), "無法讀取"),
             "no moov atom": (asset_for(media["nomoov"]), "無法讀取"), "missing": (asset_for(tmp_path / "gone.mp4"), "找不到"),
             "a folder": (asset_for(tmp_path), "找不到"),
             "no streams": (Asset(id="n", path=str(media["cuts"]), name="n", kind="video", duration=3.0, has_video=False,
                                  has_audio=False), "沒有可分析")}
    for name, (asset, needle) in cases.items():
        t0 = time.time()
        with pytest.raises(RuntimeError) as exc:
            analyze_asset(asset)
        assert needle in str(exc.value) and CJK(str(exc.value)), (name, str(exc.value))
        assert time.time() - t0 < 15, name                                           # nothing hangs
    assert not list(cache_dir("evidence").iterdir())                                  # and nothing was cached for them


@ffmpeg
def test_a_cut_off_download_is_flagged_and_evidence_ends_where_the_data_ends(media):
    a = asset_of(media["truncated"])
    assert a.duration == pytest.approx(20.0, abs=0.5)                                 # the header still promises 20 s
    ev = analyze_asset(a, use_cache=False)
    assert 8.0 < ev.duration < 14.0, ev.duration                                     # about 55 % was readable
    assert abs(len(ev.motion) / 5.0 - ev.duration) < 1.0 and abs(len(ev.rms_db) / 20.0 - ev.duration) < 1.5
    assert any("不完整" in n and "20" in n for n in ev.notes), ev.notes
    full = analyze_asset(asset_of(media["full20"]), use_cache=False)
    assert full.duration == pytest.approx(20.0, abs=0.3) and not any("不完整" in n for n in full.notes)


def test_bad_parameters_are_rejected_in_chinese(tmp_path):
    a = Asset(id="a", path=str(tmp_path / "x.mp4"), name="x.mp4", kind="video", duration=1.0)
    for params in (AnalyzeParams(audio_fps=0), AnalyzeParams(visual_fps=-1), AnalyzeParams(frame_width=4),
                   AnalyzeParams(audio_sr=100), AnalyzeParams(face_fps=0)):
        with pytest.raises(ValueError, match="分析參數不正確"):
            analyze_asset(a, params)


@ffmpeg
@pytest.mark.parametrize("params", [AnalyzeParams(audio_fps=50), AnalyzeParams(audio_fps=7.3, audio_sr=8000),
                                    AnalyzeParams(visual_fps=10, face_fps=10), AnalyzeParams(visual_fps=2.5, frame_width=16),
                                    AnalyzeParams(frame_width=700, faces=False)])
def test_unusual_parameters_give_consistent_lengths(media, params):
    a = asset_of(media["av6"])
    ev = analyze_asset(a, params, use_cache=False)
    assert abs(len(ev.rms_db) - a.duration * params.audio_fps) <= 2 and len(ev.voice) == len(ev.rms_db)
    assert abs(len(ev.motion) - a.duration * params.visual_fps) <= 2
    assert len(ev.faces) == len(ev.face_sample_mask) == len(ev.bright) == len(ev.motion_cx)


# =========================================================================== progress and cancel
def _recording():
    calls: list[tuple[float, str]] = []
    return calls, lambda f, m: calls.append((f, m))


def assert_progress_ok(calls) -> None:
    assert calls, "no progress was reported"
    fractions = [f for f, _ in calls]
    assert all(0.0 <= f <= 1.0 and f == f for f in fractions), fractions
    assert all(a <= b + 1e-12 for a, b in zip(fractions, fractions[1:])), fractions   # never goes backwards
    assert fractions[0] == 0.0 and calls[-1][0] == 1.0, (calls[0], calls[-1])
    assert all(isinstance(m, str) and m and CJK(m) for _, m in calls), calls          # messages are 繁體中文


@ffmpeg
@pytest.mark.parametrize("key", ["av6", "wav_clean", "silent", "still", "tiny", "cuts"])
def test_progress_is_monotonic_and_ends_at_one(media, key):
    calls, cb = _recording()
    analyze_asset(asset_of(media[key]), progress=cb)                                  # a miss ...
    assert_progress_ok(calls)
    calls2, cb2 = _recording()
    analyze_asset(asset_of(media[key]), progress=cb2)                                 # ... and a hit
    assert_progress_ok(calls2)
    assert calls2[-1][1] == "分析完成（使用快取）" and len(calls2) <= 4


@ffmpeg
def test_cancel_before_the_start_does_nothing(media):
    ev = threading.Event()
    ev.set()
    before = ffmpeg_children()
    t0 = time.time()
    with pytest.raises(Cancelled):
        analyze_asset(asset_of(media["av6"]), cancel=ev)
    assert time.time() - t0 < 2 and ffmpeg_children() <= before
    assert not list(cache_dir("evidence").iterdir())


@ffmpeg
@linux_only
@pytest.mark.parametrize("phase", ["sound", "picture"])
def test_cancel_midway_raises_cancelled_and_leaves_no_ffmpeg_behind(media, video90, tmp_path, phase):
    if phase == "sound":                                                              # 10 minutes of sound only
        path = write_wav(tmp_path / "long 10min.wav", tone_bursts(seconds=600, bursts=[(i * 3.0, i * 3.0 + 2.0) for i in range(200)]))
        trigger = 0.1
    else:
        path, trigger = video90, 0.5                                                  # inside the picture pass of a 90 s video
    a = asset_of(path)
    cancel = threading.Event()
    seen: list[float] = []
    stamp: list[float] = []

    def progress(f, m):
        seen.append(f)
        if f >= trigger and not cancel.is_set():
            stamp.append(time.time())
            cancel.set()

    before = ffmpeg_children()
    with pytest.raises(Cancelled):
        analyze_asset(a, progress=progress, cancel=cancel)
    assert stamp and time.time() - stamp[0] < 3.0, "the analysis kept running after cancel"
    assert max(seen) < 1.0
    deadline = time.time() + 2.0
    while ffmpeg_children() - before and time.time() < deadline:
        time.sleep(0.05)
    assert ffmpeg_children() - before == set(), "an ffmpeg child outlived the cancelled analysis"
    assert not list(cache_dir("evidence").iterdir()), "a cancelled run must not leave a (partial) cache file"
    assert analyze_asset(asset_of(media["wav_clean"])).has_audio                       # and the next run works normally


# =========================================================================== speed and memory
@ffmpeg
def test_ninety_seconds_of_720p_are_analysed_in_under_thirty_seconds(video90):
    a = asset_of(video90)
    assert (a.width, a.height, round(a.duration)) == (1280, 720, 90)
    calls, cb = _recording()
    t0 = time.perf_counter()
    ev = analyze_asset(a, progress=cb, use_cache=False)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"{elapsed:.1f}s for 90 s of video ({a.duration / elapsed:.1f}x real time)"
    assert_progress_ok(calls)
    assert len({round(f, 2) for f, _ in calls}) > 20, "progress should advance smoothly on a long file"
    assert len(ev.motion) == pytest.approx(450, abs=2) and len(ev.rms_db) == pytest.approx(1800, abs=2)
    assert len(ev.silences) == pytest.approx(29, abs=1)                               # 2 s of tone, 1 s of silence, 30 times
    print(f"\n[measured] 90 s of 1280x720: {elapsed:.1f}s ({a.duration / elapsed:.1f}x real time)")


_RSS_SCRIPT = r"""
import json, resource, sys, time
sys.path.insert(0, sys.argv[1])
from autocut.media.probe import probe_asset
from autocut.analysis import analyze_asset
a = probe_asset(sys.argv[2])
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t = time.perf_counter()
ev = analyze_asset(a, use_cache=False)
dt = time.perf_counter() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"delta_mb": (peak - base) / 1024.0, "peak_mb": peak / 1024.0, "seconds": dt, "duration": a.duration,
                  "n_audio": len(ev.rms_db), "n_visual": len(ev.motion)}))
"""


def peak_rss(path: Path, home: Path) -> dict:
    env = dict(os.environ, AUTOCUT_HOME=str(home), PYTHONPATH=str(SRC))
    p = subprocess.run([sys.executable, "-c", _RSS_SCRIPT, str(SRC), str(path)], env=env, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


@ffmpeg
@linux_only
def test_memory_does_not_grow_with_the_length_of_the_file(tmp_path, video90):
    short = write_wav(tmp_path / "sound 1min.wav", tone_bursts(seconds=60, bursts=[(i * 3.0, i * 3.0 + 2.0) for i in range(20)]))
    long = write_wav(tmp_path / "sound 10min.wav", tone_bursts(seconds=600, bursts=[(i * 3.0, i * 3.0 + 2.0) for i in range(200)]))
    r_short, r_long = peak_rss(short, tmp_path / "h1"), peak_rss(long, tmp_path / "h2")
    assert r_long["n_audio"] == pytest.approx(12000, abs=2)
    assert r_long["delta_mb"] < 40 and r_long["delta_mb"] - r_short["delta_mb"] < 20, (r_short, r_long)
    r_video = peak_rss(video90, tmp_path / "h3")
    assert r_video["delta_mb"] < 80, r_video                                           # one frame at a time, not 90 s of them
    print(f"\n[measured] peak RSS growth: 1 min sound {r_short['delta_mb']:.0f} MB, 10 min sound {r_long['delta_mb']:.0f} MB, "
          f"90 s 720p video {r_video['delta_mb']:.0f} MB (process peak {r_video['peak_mb']:.0f} MB)")


@ffmpeg
@linux_only
@pytest.mark.skipif(not os.environ.get("AUTOCUT_LONG_TESTS"), reason="set AUTOCUT_LONG_TESTS=1 for the 10-minute video run")
def test_ten_minute_video_memory_and_speed(tmp_path):
    wav = write_wav(tmp_path / "a.wav", tone_bursts(seconds=600, bursts=[(i * 3.0, i * 3.0 + 2.0) for i in range(200)]))
    ff("-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=600", "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast",
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(tmp_path / "ten.mp4"))
    r = peak_rss(tmp_path / "ten.mp4", tmp_path / "home")
    assert r["delta_mb"] < 80 and r["seconds"] < 120, r
    print(f"\n[measured] 10 min video: {r['seconds']:.1f}s ({r['duration'] / r['seconds']:.1f}x real time), "
          f"peak RSS {r['peak_mb']:.0f} MB (growth {r['delta_mb']:.0f} MB)")


# =========================================================================== faces
class _FakeCascade:
    def __init__(self, boxes):
        self.boxes, self.calls = np.asarray(boxes, dtype=np.int32).reshape(-1, 4), []

    def detectMultiScale(self, grey, **kw):                                          # noqa: N802 (OpenCV's name)
        self.calls.append((grey.shape, kw))
        return self.boxes


def test_detect_faces_degrades_to_nothing_on_odd_input():
    rng = np.random.default_rng(0)
    odd = [None, np.zeros((0, 0), np.uint8), np.zeros((10, 10, 3), np.uint8), np.zeros((144, 256, 3), np.uint8),
           rng.integers(0, 255, (144, 256, 3), dtype=np.uint8), rng.random((144, 256)), np.zeros((144, 256, 4), np.uint8),
           np.zeros((144, 256, 2), np.uint8), "not an image", np.zeros((144, 256, 1), np.uint8)]
    for frame in odd:
        assert F.detect_faces(frame) == [], type(frame)                              # blank / noise frames: no faces, no error


def test_detect_faces_box_format_with_a_stand_in_cascade(monkeypatch):
    fake = _FakeCascade([[10, 20, 30, 30], [100, 50, 60, 60], [250, 120, 20, 40]])      # on a 256 x 144 frame
    monkeypatch.setattr(F, "_load", lambda: fake)
    boxes = F.detect_faces(np.zeros((144, 256, 3), np.uint8))
    assert len(boxes) == 3 and all(len(b) == 4 and all(isinstance(v, float) for v in b) for b in boxes)
    assert boxes[0] == [round(100 / 256, 4), round(50 / 144, 4), round(60 / 256, 4), round(60 / 144, 4)]   # biggest first
    for x, y, w, h in boxes:                                                         # fractions of the frame, inside it
        assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9
    assert fake.calls[0][1]["minSize"] == (round(0.06 * 256),) * 2                   # smallest face = 6 % of the width
    assert F.detect_faces(np.zeros((144, 256, 3), np.uint8), min_frac=0.2)           # the argument is honoured
    assert fake.calls[-1][1]["minSize"] == (round(0.2 * 256),) * 2
    monkeypatch.setattr(F, "_load", lambda: None)
    assert F.detect_faces(np.zeros((144, 256, 3), np.uint8)) == [] and F.faces_available() is False


def test_sample_analyzer_runs_the_detector_on_the_face_grid_only():
    seen = []

    def detect(grey):
        seen.append(grey.shape)
        return [[0.25, 0.25, 0.5, 0.5]]

    ana = V.SampleAnalyzer(visual_fps=5.0, face_fps=2.0, faces=True, detect=detect)
    rng = np.random.default_rng(2)
    for _ in range(50):                                                              # 10 s at 5 samples per second
        ana.push(rng.integers(0, 255, (36, 64, 3), dtype=np.uint8))
    pic = ana.finish()
    assert pic.n == 50 and len(pic.faces) == len(pic.face_sample_mask) == 50
    assert pic.faces_run == len(seen) == sum(pic.face_sample_mask) and abs(pic.faces_run - 20) <= 1   # ~2 per second
    assert all((pic.faces[i] == [[0.25, 0.25, 0.5, 0.5]]) == bool(pic.face_sample_mask[i]) for i in range(50))
    assert pic.face_sample_mask[0] == 1 and pic.detector_ok and pic.faces_found == pic.faces_run
    boom = V.SampleAnalyzer(5.0, 5.0, True, detect=lambda g: 1 / 0)                  # a crashing detector must not crash
    boom.push(np.zeros((36, 64, 3), np.uint8))
    assert boom.finish().faces == [[]]


@ffmpeg
def test_pipeline_notes_when_there_are_no_faces_or_no_detector(media, monkeypatch):
    a = asset_of(media["av6"])
    ev = analyze_asset(a, use_cache=False)                                           # testsrc2 has no frontal faces
    assert len(ev.faces) == len(ev.face_sample_mask) == len(ev.motion)
    assert sum(ev.face_sample_mask) == pytest.approx(12, abs=2)                       # face_fps = 2 over 6 s
    assert all(f == [] for f, ran in zip(ev.faces, ev.face_sample_mask) if not ran)
    for boxes in ev.faces:
        for b in boxes:
            assert len(b) == 4 and all(0.0 <= v <= 1.0 for v in b)
    assert any("未偵測人臉" in n for n in ev.notes), ev.notes
    monkeypatch.setattr(F, "_load", lambda: None)                                    # no cascade file at all
    ev = analyze_asset(a, use_cache=False)
    assert ev.faces and all(f == [] for f in ev.faces) and sum(ev.face_sample_mask) == 0
    assert any("人臉偵測器無法使用" in n for n in ev.notes), ev.notes
    assert len(ev.motion) > 20 and ev.has_audio                                      # everything else still works
    ev = analyze_asset(a, AnalyzeParams(faces=False), use_cache=False)
    assert any("已停用人臉偵測" in n for n in ev.notes) and sum(ev.face_sample_mask) == 0


# =========================================================================== speech (optional dependency)
def _without_whisper(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)                         # `import faster_whisper` -> ImportError


def _with_fake_whisper(monkeypatch, *, cached: bool, segments=None):
    """A stand-in ``faster_whisper`` (and ``opencc``) package; returns the call log."""
    calls = {"load": [], "transcribe": 0, "download_checks": 0}

    def module(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        m.__path__ = []
        return m

    pkg, utils = module("faster_whisper"), module("faster_whisper.utils")

    class WhisperModel:
        def __init__(self, size, device="auto", compute_type="int8", local_files_only=False):
            if local_files_only and not cached:
                raise OSError("model is not on disk")
            calls["load"].append((size, device, compute_type, local_files_only))

        def transcribe(self, audio, language=None, vad_filter=False, word_timestamps=False):
            calls["transcribe"] += 1
            assert isinstance(audio, np.ndarray) and audio.dtype == np.float32 and vad_filter and word_timestamps
            return iter(segments if segments is not None else _SEGMENTS), types.SimpleNamespace(language=language)

    def download_model(size, local_files_only=False, **kw):
        calls["download_checks"] += 1
        if not cached:
            raise OSError("not cached")
        return "/fake/model"

    pkg.WhisperModel, pkg.utils, utils.download_model = WhisperModel, utils, download_model
    monkeypatch.setitem(sys.modules, "faster_whisper", pkg)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    monkeypatch.setattr(S, "_MODELS", {})
    return calls


def _seg(start, end, text, words=()):
    return types.SimpleNamespace(start=start, end=end, text=text,
                                 words=[types.SimpleNamespace(start=a, end=b, word=w) for a, b, w in words])


_SEGMENTS = [_seg(0.5, 1.9, " 你好，后天见 ", [(0.5, 0.9, "你好"), (1.0, 1.9, "后天见")]), _seg(2.0, 2.1, "   "),
             _seg(3.0, 5.0, "第二句", [(3.0, 5.0, "第二句")])]


def test_speech_is_reported_unavailable_without_faster_whisper(monkeypatch):
    _without_whisper(monkeypatch)
    ok, why = S.speech_available()
    assert ok is False and why == S.INSTALL_HINT and "pip install -r requirements-speech.txt" in why and CJK(why)
    assert S.model_cached("small") is False
    assert issubclass(S.SpeechUnavailable, RuntimeError)
    with pytest.raises(S.SpeechUnavailable, match="尚未安裝語音辨識"):
        S.transcribe("whatever.wav")
    assert S.APPROX_MB["small"] == 484 and "約 484 MB" in S._missing_model_message("small")


@ffmpeg
def test_analysis_with_speech_requested_still_returns_evidence_and_a_note(media, monkeypatch):
    _without_whisper(monkeypatch)
    a = asset_of(media["av6"])
    calls, cb = _recording()
    ev = analyze_asset(a, AnalyzeParams(speech=True), progress=cb)
    assert ev.speech is None and ev.has_audio and ev.has_video and ev.rms_db and ev.motion        # evidence is complete ...
    assert any("尚未安裝語音辨識" in n for n in ev.notes), ev.notes                                   # ... and says why not
    assert not ev.has_speech
    assert_progress_ok(calls)
    again = analyze_asset(a, AnalyzeParams(speech=True))                                           # from the cache: same note
    assert again.speech is None and any("尚未安裝語音辨識" in n for n in again.notes)
    assert not any("尚未安裝" in n for n in analyze_asset(a).notes)                                # speech off: no such note
    silent = analyze_asset(asset_of(media["silent"]), AnalyzeParams(speech=True))
    assert silent.speech is None and any("沒有聲音" in n for n in silent.notes)


def test_segments_become_asset_time_word_timestamped_dicts():
    seen = []
    out = S.segments_to_dicts(_SEGMENTS, offset=10.0, convert=lambda s: s.replace("后", "後").replace("见", "見"), total=6.0,
                              progress=seen.append)
    assert [s["text"] for s in out] == ["你好，後天見", "第二句"]                       # the blank segment is dropped
    assert out[0] == {"start": 10.5, "end": 11.9, "text": "你好，後天見",
                      "words": [{"start": 10.5, "end": 10.9, "word": "你好"}, {"start": 11.0, "end": 11.9, "word": "後天見"}]}
    assert seen == sorted(seen) and 0 < seen[0] <= 1 and seen[-1] == pytest.approx(5.0 / 6.0)
    ev = threading.Event()
    ev.set()
    with pytest.raises(Cancelled):
        S.segments_to_dicts(_SEGMENTS, cancel=ev)


@ffmpeg
def test_model_not_downloaded_is_not_downloaded_without_consent(media, monkeypatch):
    calls = _with_fake_whisper(monkeypatch, cached=False)
    assert S.speech_available() == (True, "") and S.model_cached("small") is False
    with pytest.raises(S.SpeechUnavailable, match="尚未下載語音模型「small」（約 484 MB）"):
        S.transcribe(media["wav_clean"])
    assert calls["load"] == [] and calls["transcribe"] == 0
    ev = analyze_asset(asset_of(media["wav_clean"]), AnalyzeParams(speech=True))
    assert ev.speech is None and any("尚未下載語音模型" in n for n in ev.notes) and ev.rms_db
    assert calls["load"] == []                                                       # still nothing downloaded
    ev = analyze_asset(asset_of(media["wav_clean"]), AnalyzeParams(speech=True, allow_download=True))
    assert calls["load"] and calls["load"][-1][3] is False and ev.speech is not None  # explicit consent -> it may download


@ffmpeg
def test_speech_success_uses_asset_times_converts_to_traditional_and_is_cached(media, monkeypatch):
    calls = _with_fake_whisper(monkeypatch, cached=True)
    opencc = types.ModuleType("opencc")
    opencc.OpenCC = lambda cfg: types.SimpleNamespace(convert=lambda s: s.replace("后", "後").replace("见", "見"))
    monkeypatch.setitem(sys.modules, "opencc", opencc)
    a = asset_of(media["wav_clean"])
    seen, cb = _recording()
    ev = analyze_asset(a, AnalyzeParams(speech=True), progress=cb)
    assert [s["text"] for s in ev.speech] == ["你好，後天見", "第二句"] and ev.has_speech
    assert ev.speech[1]["words"] == [{"start": 3.0, "end": 5.0, "word": "第二句"}]
    assert calls["transcribe"] == 1 and calls["load"][0][3] is True                  # local_files_only: no download
    assert_progress_ok(seen)
    assert len(list(cache_dir("speech").iterdir())) == 1
    seen2, cb2 = _recording()
    again = analyze_asset(a, AnalyzeParams(speech=True), progress=cb2)               # second run: both caches, no model
    assert calls["transcribe"] == 1 and again.speech == ev.speech and again.sha256() == ev.sha256()
    assert_progress_ok(seen2) and seen2[-1][1] == "分析完成（使用快取）"
    other_lang = analyze_asset(a, AnalyzeParams(speech=True, language="en"))          # another language: a new transcription
    assert calls["transcribe"] == 2 and other_lang.rms_db == ev.rms_db
    plain = analyze_asset(a)                                                          # without speech: never carries any
    assert plain.speech is None and not any("語音" in n for n in plain.notes)


@ffmpeg
def test_speech_without_opencc_says_the_text_may_be_simplified(media, monkeypatch):
    _with_fake_whisper(monkeypatch, cached=True)
    monkeypatch.setitem(sys.modules, "opencc", None)
    ev = analyze_asset(asset_of(media["wav_clean"]), AnalyzeParams(speech=True))
    assert ev.speech and ev.speech[0]["text"] == "你好，后天见"                          # not converted ...
    assert any("opencc" in n and "簡體" in n for n in ev.notes), ev.notes              # ... and the user is told


@ffmpeg
def test_a_broken_speech_install_never_breaks_the_analysis(media, monkeypatch):
    calls = _with_fake_whisper(monkeypatch, cached=True)

    def boom(*args, **kw):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(S, "transcribe", boom)
    ev = analyze_asset(asset_of(media["wav_clean"]), AnalyzeParams(speech=True))
    assert ev.speech is None and ev.rms_db and any("語音辨識失敗" in n and "out of memory" in n for n in ev.notes)
    assert calls["transcribe"] == 0


@ffmpeg
def test_cancel_during_speech_recognition_propagates(media, monkeypatch):
    _with_fake_whisper(monkeypatch, cached=True)
    ev = threading.Event()

    def segments():
        yield _seg(0.5, 1.0, "第一句")
        ev.set()
        yield _seg(1.0, 2.0, "第二句")

    _with_fake_whisper(monkeypatch, cached=True, segments=segments())
    with pytest.raises(Cancelled):
        analyze_asset(asset_of(media["wav_clean"]), AnalyzeParams(speech=True), cancel=ev)
    assert not list(cache_dir("speech").iterdir())
