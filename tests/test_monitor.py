"""Monitor + PlaybackController (offscreen Qt): frames, playback clock, audio fallback, preview cache, reframe editor.

Screenshots go to ``screenshots/monitor_<name>.png`` (git-ignored).  Tests that need real pixels use the ``real_proj``
fixture (ffmpeg); the playback / audio / cache-status tests run on the fake-asset ``proj`` fixture.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt                      # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent                          # noqa: E402
from PySide6.QtTest import QTest                                            # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel                          # noqa: E402

from autocut.core import geometry as geo                                    # noqa: E402
from autocut.core.project import Project                                   # noqa: E402
from autocut.gui import monitor as monitor_mod                              # noqa: E402
from autocut.gui.monitor import CropBox, MonitorWidget, active_cue_lines, clip_local_time, fit_rect  # noqa: E402
from autocut.gui.playback import AudioUnavailable, DriftGuard, PlaybackController           # noqa: E402
from autocut.gui.state import AppState                                      # noqa: E402
from autocut.media.ffmpeg import Cancelled                                  # noqa: E402
from autocut.render.compiler import RenderError                            # noqa: E402
from autocut.render.runner import RenderResult                              # noqa: E402

pytestmark = pytest.mark.gui
SHOTS = Path(__file__).resolve().parents[1] / "screenshots"


# ------------------------------------------------------------------------------ helpers
def pump(qapp, cond, timeout: float = 8.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.003)
    qapp.processEvents()
    return bool(cond())


def run_for(qapp, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.003)


def shot(widget, name: str) -> Path:
    SHOTS.mkdir(exist_ok=True)
    path = SHOTS / f"monitor_{name}.png"
    assert widget.grab().save(str(path))
    return path


def get_clip(st: AppState, clip_id: str):
    return st.sequence().find_clip(clip_id)[1]


def build(proj: Project, seconds: float = 3.0, size=(1080, 1920), asset: str = "land.mp4", fit: str = "cover"):
    """AppState with one clip of ``asset`` on V1 of a ``size`` sequence -> (state, clip id)."""
    ids = {a.name: a.id for a in proj.assets.values()}
    st = AppState(proj)
    if tuple(size) != (1920, 1080):
        st.op("格式", "set_sequence_format", width=size[0], height=size[1])
    clip = st.op("放置", "place_clip", ids[asset], st.sequence().video_tracks[0].id, 0.0, 0.0, seconds)
    if fit != "cover":
        st.op("填滿方式", "set_clip_props", clip.id, fit=fit)
    return st, clip.id


def short_state(proj: Project, seconds: float = 0.6) -> AppState:
    """Fake-asset project with one clip (nothing is decoded): enough for the playback clock."""
    st = AppState(proj)
    st.op("放置", "place_clip", "a1", st.sequence().video_tracks[0].id, 0.0, 0.0, seconds)
    return st


def non_black(view) -> float:
    """Fraction of sampled pixels inside the picture that are not (near) black."""
    img = view.grab().toImage()
    r = view.picture_rect().toRect()
    n = hits = 0
    for y in range(r.top() + 3, r.bottom() - 3, 7):
        for x in range(r.left() + 3, r.right() - 3, 7):
            c = img.pixelColor(x, y)
            n += 1
            hits += (c.red() + c.green() + c.blue()) > 60
    return hits / max(1, n)


def mouse(view, kind, pos: QPointF, button=Qt.LeftButton, buttons=Qt.LeftButton, mods=Qt.NoModifier) -> None:
    QApplication.sendEvent(view, QMouseEvent(kind, pos, view.mapToGlobal(pos), button, buttons, mods))


def drag(view, start: QPointF, end: QPointF, steps: int = 4) -> None:
    mouse(view, QEvent.MouseButtonPress, start)
    for i in range(1, steps + 1):
        mouse(view, QEvent.MouseMove, start + (end - start) * (i / steps), Qt.NoButton, Qt.LeftButton)
    mouse(view, QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton)


def wheel(view, pos: QPointF, delta: int) -> None:
    QApplication.sendEvent(view, QWheelEvent(pos, view.mapToGlobal(pos), QPoint(0, 0), QPoint(0, delta),
                                             Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))


@pytest.fixture
def made(qapp):
    """Factory: (state, width, height) -> (monitor, playback); everything is shut down afterwards."""
    created = []

    def make(state: AppState, width: int = 900, height: int = 640, **kw):
        pb = PlaybackController(state, **kw)
        mon = MonitorWidget(state, pb)
        mon.resize(width, height)
        mon.show()
        qapp.processEvents()
        created.append((mon, pb))
        return mon, pb

    yield make
    for mon, pb in created:
        pb.close()
        mon.shutdown()
        mon.close()


class FakeAudio:
    """Stands in for QMediaPlayer: records what the controller asks for."""

    def __init__(self, _parent=None):
        self.on_error = None
        self.loaded: list[str] = []
        self.pos = 0.0
        self.playing = False
        self.sets: list[float] = []
        self.released = 0

    def load(self, path):
        self.loaded.append(path)

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def position(self):
        return self.pos

    def set_position(self, t):
        self.sets.append(t)
        self.pos = t

    def release(self):
        self.released += 1

    def close(self):
        pass


def preview_file(tmp_path: Path) -> str:
    f = tmp_path / "preview.mp4"
    f.write_bytes(b"stub")
    return str(f)


# ============================================================================ playback clock
def test_playback_advances_in_real_time_and_stops_at_end(qapp, proj):
    st = short_state(proj, 0.6)
    pb = PlaybackController(st)
    heads, flips, ends = [], [], []
    st.playhead_changed.connect(heads.append)
    pb.playing_changed.connect(flips.append)
    pb.finished.connect(lambda: ends.append(1))
    t0 = time.perf_counter()
    assert pb.play() and pb.is_playing
    assert pump(qapp, lambda: not pb.is_playing, 5)
    elapsed = time.perf_counter() - t0
    assert flips == [True, False] and ends == [1]
    assert st.playhead == pytest.approx(0.6)                 # stops exactly at the sequence end
    assert 0.5 <= elapsed < 1.5                              # a wall clock, not a frame counter
    assert heads == sorted(heads) and len(heads) >= 10      # monotonic and (almost) every frame
    assert pb.play() and st.playhead == 0.0                   # playing from the end starts over
    pb.pause()


def test_slow_gui_drops_frames_but_never_lags(qapp, proj):
    st = short_state(proj, 0.8)
    pb = PlaybackController(st)
    heads = []
    st.playhead_changed.connect(heads.append)
    t0 = time.perf_counter()
    pb.play()
    run_for(qapp, 0.1)
    time.sleep(0.35)                                          # the GUI thread is blocked (slow decode / paint)
    assert pump(qapp, lambda: not pb.is_playing, 5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.2                                      # finished on time instead of 0.35 s late
    assert max(b - a for a, b in zip(heads, heads[1:])) >= 0.25   # one big jump = dropped frames
    assert st.playhead == pytest.approx(0.8)


def test_loop_wraps_around(qapp, proj):
    st = short_state(proj, 0.4)
    pb = PlaybackController(st)
    loops = []
    pb.loop_changed.connect(loops.append)
    pb.set_loop(True)
    assert pb.loop and loops == [True]
    heads = []
    st.playhead_changed.connect(heads.append)
    pb.play()
    assert pump(qapp, lambda: any(b < a - 0.2 for a, b in zip(heads, heads[1:])), 4)
    assert pb.is_playing                                      # kept going after the end
    pb.pause()
    assert not pb.is_playing


def test_play_in_out_range(qapp, proj):
    st = short_state(proj, 1.0)
    st.set_range(0.3, 0.6)
    pb = PlaybackController(st)
    st.set_playhead(0.9)
    assert pb.play_in_out()
    assert st.playhead == pytest.approx(0.3) and pb.in_out_mode      # starts at the in point
    assert pump(qapp, lambda: not pb.is_playing, 4)
    assert st.playhead == pytest.approx(0.6) and not pb.in_out_mode  # stops at the out point


def test_in_out_without_range_falls_back_to_whole_sequence(qapp, proj):
    st = short_state(proj, 0.4)
    msgs = []
    st.status.connect(msgs.append)
    pb = PlaybackController(st)
    assert pb.play(in_out=True)
    assert any("入點" in m for m in msgs) and not pb.in_out_mode
    assert pump(qapp, lambda: not pb.is_playing, 4)
    assert st.playhead == pytest.approx(0.4)


def test_seek_during_playback_continues_from_there(qapp, proj):
    st = short_state(proj, 2.0)
    pb = PlaybackController(st)
    pb.play()
    run_for(qapp, 0.1)
    pb.seek(1.5)
    run_for(qapp, 0.15)
    assert pb.is_playing and 1.5 <= st.playhead < 1.9
    pb.seek(99.0)                                             # clamped to the end -> playback ends
    qapp.processEvents()
    assert not pb.is_playing and st.playhead == pytest.approx(2.0)


def two_clips(proj: Project, each: float = 0.6):
    """AppState with two clips of ``each`` seconds one after the other on V1 -> (state, id of the second clip)."""
    st = short_state(proj, each)
    second = st.op("放置", "place_clip", "a1", st.sequence().video_tracks[0].id, each, 0.0, each)
    assert st.sequence().duration == pytest.approx(2 * each)
    return st, second.id


def test_playing_runs_to_the_new_end_when_an_edit_shortens_the_sequence(qapp, proj):
    """The end a running transport plays to is the sequence's CURRENT end, not the one it had when play was pressed
    (the bounds used to be fixed at that moment: playback ran on past the end of the picture)."""
    st, second = two_clips(proj, 0.6)
    pb = PlaybackController(st)
    ends = []
    pb.finished.connect(lambda: ends.append(1))
    pb.play()
    run_for(qapp, 0.1)
    st.op("刪除", "remove_clips", [second], ripple=True)       # 1.2 s -> 0.6 s while playing
    assert st.sequence().duration == pytest.approx(0.6)
    assert pump(qapp, lambda: not pb.is_playing, 3)
    assert st.playhead == pytest.approx(0.6) and ends == [1]


def test_playing_runs_to_the_new_end_when_an_edit_lengthens_the_sequence(qapp, proj):
    st = short_state(proj, 0.5)
    pb = PlaybackController(st)
    pb.play()
    run_for(qapp, 0.1)
    st.op("放置", "place_clip", "a1", st.sequence().video_tracks[0].id, 0.5, 0.0, 0.5)    # 0.5 s -> 1.0 s
    assert pump(qapp, lambda: not pb.is_playing, 3)
    assert st.playhead == pytest.approx(1.0)


def test_loop_wraps_at_the_new_end_after_an_edit(qapp, proj):
    st, second = two_clips(proj, 0.5)
    pb = PlaybackController(st)
    pb.set_loop(True)
    pb.play()
    run_for(qapp, 0.1)
    st.op("刪除", "remove_clips", [second], ripple=True)       # 1.0 s -> 0.5 s while looping
    highest = 0.0
    end = time.monotonic() + 1.5
    while time.monotonic() < end:
        qapp.processEvents()
        highest = max(highest, st.playhead)
        time.sleep(0.003)
    assert pb.is_playing and highest <= 0.5 + 1e-6, f"the loop must wrap at 0.5 s, the playhead reached {highest:.2f}"
    pb.pause()


def test_playback_stops_when_an_edit_empties_the_sequence(qapp, proj):
    st = short_state(proj, 1.0)
    clip = st.sequence().video_tracks[0].clips[0]
    msgs = []
    st.status.connect(msgs.append)
    pb = PlaybackController(st)
    pb.set_loop(True)                                         # even a loop has nothing to play
    pb.play()
    run_for(qapp, 0.1)
    st.op("刪除", "remove_clips", [clip.id], ripple=True)
    qapp.processEvents()
    assert not pb.is_playing and any("空" in m for m in msgs)
    st.undo()                                                 # the clip is back: play works again
    assert pb.play() and pb.is_playing
    pb.pause()


def test_in_out_playback_follows_an_edit_of_its_range(qapp, proj):
    st = short_state(proj, 2.0)
    st.set_range(0.2, 1.2)
    pb = PlaybackController(st)
    assert pb.play_in_out() and pb.in_out_mode
    run_for(qapp, 0.05)
    st.set_range(0.2, 0.5)                                    # the out point is moved while it plays
    assert pump(qapp, lambda: not pb.is_playing, 3)
    assert st.playhead == pytest.approx(0.5)


def test_empty_sequence_cannot_play_and_says_so(qapp):
    st = AppState(Project.new("空"))
    msgs = []
    st.status.connect(msgs.append)
    pb = PlaybackController(st)
    assert pb.play() is False and not pb.is_playing
    assert any("序列是空的" in m for m in msgs)


def test_switching_project_stops_playback(qapp, proj):
    st = short_state(proj, 2.0)
    pb = PlaybackController(st)
    pb.play()
    assert pb.is_playing
    st.replace_project(Project.new("另一個"))
    assert not pb.is_playing


def test_toggle_step_and_goto(qapp, proj):
    st = short_state(proj, 1.0)
    pb = PlaybackController(st)
    assert pb.toggle() is True and pb.is_playing
    assert pb.toggle() is False and not pb.is_playing
    pb.step(1)
    assert st.playhead == pytest.approx(1 / 30)
    pb.step(-5)
    assert st.playhead == 0.0                                  # never negative
    pb.goto_end()
    assert st.playhead == pytest.approx(1.0)
    pb.goto_start()
    assert st.playhead == 0.0


# ============================================================================ audio: fallback + sync
def test_audio_backend_missing_falls_back_silently_with_one_message(qapp, proj, tmp_path):
    def broken(_parent):
        raise AudioUnavailable("找不到音訊輸出裝置")

    st = short_state(proj, 0.4)
    pb = PlaybackController(st, audio_factory=broken)
    msgs = []
    st.status.connect(msgs.append)
    pb.note_preview_rendered(preview_file(tmp_path), st.project.revision)
    assert pb.preview_status() == "fresh"
    assert pb.play() and not pb.audio_active                  # plays, silently
    assert pump(qapp, lambda: not pb.is_playing, 4)
    assert st.playhead == pytest.approx(0.4)
    st.set_playhead(0.0)
    pb.play()
    pb.pause()
    sound = [m for m in msgs if "靜音" in m]
    assert len(sound) == 1 and "找不到音訊輸出裝置" in sound[0]   # ONE message, however often it is played again


def test_default_backend_is_unavailable_offscreen_and_falls_back(qapp, proj, tmp_path):
    st = short_state(proj, 0.3)
    pb = PlaybackController(st)                               # real QtMultimedia factory, offscreen platform
    msgs = []
    st.status.connect(msgs.append)
    pb.note_preview_rendered(preview_file(tmp_path), st.project.revision)
    assert pb.play() and not pb.audio_active
    assert pump(qapp, lambda: not pb.is_playing, 4)
    assert len([m for m in msgs if "靜音" in m]) == 1


def test_no_qtmultimedia_module_is_handled(qapp, proj, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "PySide6.QtMultimedia", None)      # import fails like on a machine without it
    st = short_state(proj, 0.3)
    pb = PlaybackController(st)
    msgs = []
    st.status.connect(msgs.append)
    pb.note_preview_rendered(preview_file(tmp_path), st.project.revision)
    assert pb.play() and not pb.audio_active
    pb.pause()
    assert len([m for m in msgs if "QtMultimedia" in m and "靜音" in m]) == 1


def test_modules_import_without_loading_qtmultimedia():
    code = ("import sys; import autocut.gui.playback, autocut.gui.monitor; "
            "assert 'PySide6.QtMultimedia' not in sys.modules, 'QtMultimedia imported at module level'")
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=60,
                       env={**__import__("os").environ, "PYTHONPATH": str(root / "src"), "QT_QPA_PLATFORM": "offscreen"})
    assert r.returncode == 0, r.stderr[-400:]


def test_audio_plays_only_a_fresh_preview_and_drops_when_stale(qapp, proj, tmp_path):
    st = short_state(proj, 2.0)
    fake = FakeAudio()
    pb = PlaybackController(st, audio_factory=lambda _p: fake)
    path = preview_file(tmp_path)
    pb.play()                                                  # no preview yet -> silent
    assert not pb.audio_active and fake.loaded == []
    pb.pause()
    pb.note_preview_rendered(path, st.project.revision)
    st.set_playhead(0.5)
    pb.play()
    assert pb.audio_active and fake.playing and fake.loaded == [path]
    assert fake.sets[-1] == pytest.approx(0.5)                # audio starts where the playhead is
    seen = []
    pb.audio_changed.connect(seen.append)
    st.op("音量", "set_clip_props", st.sequence().video_tracks[0].clips[0].id, volume=0.5)   # edit under the audio
    assert pb.preview_status() == "stale"
    assert not pb.audio_active and not fake.playing and seen == [False]
    assert pb.is_playing                                       # picture keeps playing, silently
    pb.pause()


def test_audio_resync_when_drift_exceeds_80ms(qapp, proj, tmp_path):
    st = short_state(proj, 3.0)
    fake = FakeAudio()
    pb = PlaybackController(st, audio_factory=lambda _p: fake)
    pb.note_preview_rendered(preview_file(tmp_path), st.project.revision)
    pb.play()
    base = time.perf_counter()
    fake.pos = 1.00
    n = len(fake.sets)
    pb._sync_audio(base + 1.0, 1.05)                          # 50 ms apart: leave it alone
    assert len(fake.sets) == n
    pb._sync_audio(base + 2.0, 1.30)                          # 300 ms apart: re-seek to the clock
    assert fake.sets[-1] == pytest.approx(1.30)
    pb.pause()


def test_drift_guard_rules():
    g = DriftGuard()
    g.reset(0.0)
    assert g.check(0.1, 9.0, 0.1) is None                     # grace period right after a start
    assert g.check(0.5, 0.50, 0.55) is None                   # within 80 ms
    assert g.check(0.8, 0.50, 0.80) == pytest.approx(0.80)    # 300 ms off -> seek
    assert g.check(1.0, 0.50, 1.00) is None                   # just seeked: grace again
    assert g.check(2.0, 0.50, 2.00) == pytest.approx(2.0)


def test_audio_errors_report_once_and_stay_silent(qapp, proj, tmp_path):
    class Flaky(FakeAudio):
        def play(self):
            super().play()
            if self.on_error:
                self.on_error("無法解碼這個檔案")

    st = short_state(proj, 0.5)
    pb = PlaybackController(st, audio_factory=lambda _p: Flaky())
    msgs = []
    st.status.connect(msgs.append)
    pb.note_preview_rendered(preview_file(tmp_path), st.project.revision)
    pb.play()
    assert not pb.audio_active and pb.is_playing
    pb.pause()
    st.set_playhead(0.0)
    pb.play()                                                  # the same file is not tried again
    pb.pause()
    assert len([m for m in msgs if "無法解碼" in m]) == 1


def test_audio_released_while_rendering_and_after_stop(qapp, proj, tmp_path):
    st = short_state(proj, 1.0)
    fake = FakeAudio()
    pb = PlaybackController(st, audio_factory=lambda _p: fake)
    path = preview_file(tmp_path)
    pb.note_preview_rendered(path, st.project.revision)
    pb.play()
    assert fake.playing
    pb.set_preview_rendering(True)                            # the render rewrites the file: it must be closed
    assert fake.released == 1 and not fake.playing and pb.preview_status() == "rendering"
    pb.pause()
    pb.note_preview_rendered(path, st.project.revision)
    assert pb.preview_status() == "fresh"
    pb.play()
    assert fake.loaded == [path, path]                         # reloaded after the render
    pb.stop()
    assert not pb.is_playing and fake.released == 2


# ============================================================================ preview-cache status
def test_preview_status_fresh_stale_none(qapp, proj, tmp_path):
    st = short_state(proj, 1.0)
    pb = PlaybackController(st)
    seen = []
    pb.preview_status_changed.connect(seen.append)
    assert pb.preview_status() == "none" and pb.preview_path() is None
    path = preview_file(tmp_path)
    pb.note_preview_rendered(path, st.project.revision)
    assert pb.preview_status() == "fresh" and pb.preview_path() == path
    cid = st.sequence().video_tracks[0].clips[0].id
    st.op("音量", "set_clip_props", cid, volume=0.5)
    assert pb.preview_status() == "stale" and pb.preview_path() is None
    st.undo()                                                  # revision moved on: fail closed, still stale
    assert pb.preview_status() == "stale"
    pb.note_preview_rendered(path, st.project.revision)
    assert pb.preview_status() == "fresh"
    assert seen == ["fresh", "stale", "fresh"]
    Path(path).unlink()                                        # file vanished
    assert pb.preview_status() == "none"


def test_preview_cache_is_per_sequence(qapp, proj, tmp_path):
    st = short_state(proj, 1.0)
    pb = PlaybackController(st)
    pb.note_preview_rendered(preview_file(tmp_path), st.project.revision)
    first = st.seq_id
    st.project_op("新序列", "create_sequence", "B", 1080, 1920, 30.0)
    other = [s.id for s in st.project.sequences if s.id != first][0]
    st.set_sequence(other)
    assert pb.preview_status() == "none"
    st.set_sequence(first)
    assert pb.preview_status() in ("stale", "fresh") and pb.preview_status() != "none"


def test_monitor_cache_label_follows_status(qapp, made, proj, tmp_path):
    st = short_state(proj, 1.0)
    mon, pb = made(st)
    assert mon.lbl_cache.text() == "預覽快取：無"
    pb.note_preview_rendered(preview_file(tmp_path), st.project.revision)
    assert mon.lbl_cache.text() == "預覽快取：最新"
    st.op("音量", "set_clip_props", st.sequence().video_tracks[0].clips[0].id, volume=0.4)
    assert mon.lbl_cache.text() == "預覽快取：過期"
    pb.set_preview_rendering(True)
    assert mon.lbl_cache.text().startswith("預覽快取：渲染中")


# ============================================================================ preview render (background)
def _result(path: Path, seq) -> RenderResult:
    return RenderResult(str(path), seq.duration, 15, 960, 540, 30.0, 1, 0, 0.1, "libx264", "ffmpeg")


def test_render_preview_background_progress_then_fresh_then_stale(qapp, made, proj, tmp_path, monkeypatch):
    st = short_state(proj, 0.5)
    mon, pb = made(st)
    out = Path(preview_file(tmp_path))
    info = {}

    def fake(project, seq, progress=None, cancel=None, frame_range=None):
        info["thread"] = threading.current_thread()
        info["project"] = project
        for i in range(5):
            progress(i / 5, f"渲染 {i}")
            time.sleep(0.01)
        return _result(out, seq)

    monkeypatch.setattr("autocut.render.runner.render_preview", fake)
    msgs, ticks = [], []
    st.status.connect(msgs.append)
    pb.preview_status_changed.connect(ticks.append)
    assert mon.render_preview() is True
    assert mon.preview_status() == "rendering" and mon.btn_render.text() == "取消渲染"
    assert mon.render_preview() is False                       # already running
    assert pump(qapp, lambda: mon.preview_status() != "rendering", 5)
    assert mon.preview_status() == "fresh" and mon.lbl_cache.text() == "預覽快取：最新"
    assert mon.btn_render.text() == "渲染預覽（含聲音）" and mon.progress.isHidden()
    assert info["thread"] is not threading.main_thread()      # never on the GUI thread
    assert info["project"] is not st.project                  # the worker renders a snapshot, not the live project
    assert ticks == ["rendering", "fresh"]
    assert any("預覽已渲染完成" in m for m in msgs)
    st.op("音量", "set_clip_props", st.sequence().video_tracks[0].clips[0].id, volume=0.5)
    assert mon.preview_status() == "stale" and mon.lbl_cache.text() == "預覽快取：過期"


def test_render_preview_can_be_cancelled(qapp, made, proj, tmp_path, monkeypatch):
    st = short_state(proj, 0.5)
    mon, pb = made(st)
    started = threading.Event()

    def slow(project, seq, progress=None, cancel=None, frame_range=None):
        started.set()
        for i in range(500):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            time.sleep(0.01)
        return _result(Path(preview_file(tmp_path)), seq)

    monkeypatch.setattr("autocut.render.runner.render_preview", slow)
    msgs = []
    st.status.connect(msgs.append)
    assert mon.render_preview()
    assert pump(qapp, started.is_set, 3)
    mon.btn_render.click()                                     # the same button now cancels
    assert pump(qapp, lambda: mon.preview_status() != "rendering", 5)
    assert mon.preview_status() == "none" and mon.btn_render.text() == "渲染預覽（含聲音）"
    assert any("已取消預覽渲染" in m for m in msgs)


def test_render_preview_failure_is_reported(qapp, made, proj, monkeypatch):
    st = short_state(proj, 0.5)
    mon, pb = made(st)

    def boom(project, seq, progress=None, cancel=None, frame_range=None):
        raise RenderError("找不到這些素材檔（可能被移動或刪除）：\n/x/a1.mp4")

    monkeypatch.setattr("autocut.render.runner.render_preview", boom)
    msgs = []
    st.status.connect(msgs.append)
    assert mon.render_preview()
    assert pump(qapp, lambda: mon.preview_status() != "rendering", 5)
    assert mon.preview_status() == "none"
    assert any("預覽渲染失敗" in m and "找不到這些素材檔" in m for m in msgs)     # the 繁中 text arrives untouched


def test_render_started_before_an_edit_is_stale_when_it_finishes(qapp, made, proj, tmp_path, monkeypatch):
    st = short_state(proj, 0.5)
    mon, pb = made(st)
    gate = threading.Event()

    def gated(project, seq, progress=None, cancel=None, frame_range=None):
        gate.wait(5)
        return _result(Path(preview_file(tmp_path)), seq)

    monkeypatch.setattr("autocut.render.runner.render_preview", gated)
    msgs = []
    st.status.connect(msgs.append)
    assert mon.render_preview()
    st.op("音量", "set_clip_props", st.sequence().video_tracks[0].clips[0].id, volume=0.3)   # edit while rendering
    gate.set()
    assert pump(qapp, lambda: mon.preview_status() != "rendering", 5)
    assert mon.preview_status() == "stale"
    assert any("已過期" in m for m in msgs)


def test_render_preview_refuses_empty_sequence(qapp, made):
    st = AppState(Project.new("空"))
    mon, _pb = made(st)
    msgs = []
    st.status.connect(msgs.append)
    assert mon.render_preview() is False
    assert any("空的" in m for m in msgs)


@pytest.mark.ffmpeg
def test_render_preview_really_renders_with_sound(qapp, made, real_proj):
    from autocut.media.probe import probe_asset
    st, cid = build(real_proj, seconds=0.6, size=(1080, 1920))
    mon, pb = made(st)
    assert mon.render_preview()
    assert pump(qapp, lambda: mon.preview_status() != "rendering", 90)
    assert mon.preview_status() == "fresh"
    path = pb.preview_path()
    assert path and Path(path).stat().st_size > 1000
    info = probe_asset(path)
    assert info.has_audio and info.duration == pytest.approx(0.6, abs=0.15)


# ============================================================================ picture: frames, messages, captions
@pytest.mark.ffmpeg
def test_frames_appear_and_follow_the_playhead(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=2.0, size=(1920, 1080))
    mon, _pb = made(st)
    assert mon.wait_idle() and mon.view.image is not None and mon.view.message == ""
    assert non_black(mon.view) > 0.5                          # colour bars, not an empty canvas
    pic = mon.view.picture_rect()
    assert pic.width() / pic.height() == pytest.approx(16 / 9, abs=0.02)    # letterboxed, aspect kept
    img0 = mon.view.grab().toImage()
    st.set_playhead(1.5)
    assert mon.wait_idle() and mon.view.shown_t == pytest.approx(1.5)
    assert mon.view.grab().toImage() != img0                  # a different frame is on screen
    assert mon.tc.text() == "00:00:01:15"


@pytest.mark.ffmpeg
def test_portrait_sequence_is_letterboxed_as_portrait(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=2.0, size=(1080, 1920))
    mon, _pb = made(st)
    assert mon.wait_idle()
    pic = mon.view.picture_rect()
    assert pic.width() / pic.height() == pytest.approx(9 / 16, abs=0.02)
    assert non_black(mon.view) > 0.5


@pytest.mark.ffmpeg
def test_playing_shows_frames_and_ends_on_the_last_one(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=1.0, size=(1080, 1920))
    mon, pb = made(st)
    assert mon.wait_idle()
    before = mon.frames_shown
    pb.play()
    assert pump(qapp, lambda: not pb.is_playing, 6)
    assert mon.wait_idle()
    assert st.playhead == pytest.approx(1.0) and mon.frames_shown - before >= 3
    assert mon.view.shown_t == pytest.approx(1.0 - 1 / 30)    # the end holds the last frame instead of black
    assert non_black(mon.view) > 0.5


@pytest.mark.ffmpeg
def test_requests_are_coalesced_while_scrubbing(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=3.0, size=(1080, 1920))
    mon, _pb = made(st)
    assert mon.wait_idle()
    for i in range(1, 80):                                    # a burst of playhead moves, no event processing
        st.set_playhead(i / 30)
    assert mon.wait_idle()
    assert mon._worker.dropped > 0                            # newer requests replaced the waiting ones
    assert mon.view.shown_t == pytest.approx(79 / 30)         # and the last one is what is on screen


def test_empty_sequence_shows_a_message_not_an_error(qapp, made):
    st = AppState(Project.new("空"))
    mon, _pb = made(st)
    assert "還沒有片段" in mon.view.message
    assert not mon.grab().isNull()
    shot(mon, "empty")
    st.set_playhead(2.0)                                       # moving around an empty sequence is harmless
    assert "還沒有片段" in mon.view.message


def test_audio_only_sequence_message(qapp, made, proj):
    st = AppState(proj)
    st.op("放置", "place_clip", "m1", st.sequence().audio_tracks[0].id, 0.0, 0.0, 2.0)
    mon, _pb = made(st)
    assert "只有聲音" in mon.view.message


def test_missing_media_shows_a_clear_message(qapp, made, proj):
    st = short_state(proj, 1.0)                                # a1 points at /nonexistent/a1.mp4
    mon, pb = made(st)
    assert mon.wait_idle()
    assert "找不到" in mon.view.message and "a1.mp4" in mon.view.message and mon.view.warn
    assert mon.view.image is not None                          # the background frame is still drawn
    assert not mon.grab().isNull()
    pb.step(3)                                                 # moving on does not raise either
    assert mon.wait_idle() and "a1.mp4" in mon.view.message
    shot(mon, "missing")


@pytest.mark.ffmpeg
def test_project_replaced_resets_the_view(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=1.0)
    mon, _pb = made(st)
    assert mon.wait_idle() and mon.view.image is not None
    st.select([cid])
    mon.btn_reframe.click()
    assert mon.reframing
    st.replace_project(Project.new("新的"))
    assert not mon.reframing and "還沒有片段" in mon.view.message and mon.view.image is None


def test_active_cue_lines_wrap_like_the_renderer():
    p = Project.new("c")
    seq = p.sequences[0]
    seq.caption_style.max_chars = 6
    from autocut.core.project import CaptionCue
    seq.cues = [CaptionCue("q1", 0.0, 1.0, "一二三四五六七八九十"), CaptionCue("q2", 2.0, 3.0, "後面的字幕")]
    assert active_cue_lines(seq, 0.5) == [["一二三四五六", "七八九十"]]
    assert active_cue_lines(seq, 1.5) == []
    assert active_cue_lines(seq, 2.0) == [["後面的字幕"]]
    assert active_cue_lines(seq, 1.0) == []                    # the end time is exclusive


@pytest.mark.ffmpeg
def test_caption_overlay_is_drawn_and_can_be_hidden(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=3.0, size=(1080, 1920))
    st.op("字幕樣式", "set_caption_style", size=0.04, max_chars=10, position="bottom", color="#FFFF00")
    st.op("字幕", "add_cue", 0.0, 1.5, "這是測試字幕")
    mon, _pb = made(st)
    assert mon.wait_idle()
    on = mon.view.grab().toImage()
    mon.btn_caps.click()                                       # toggle off
    off = mon.view.grab().toImage()
    assert on != off
    pic = mon.view.picture_rect().toRect()
    top = (pic.left() + 4, pic.top() + 4, pic.width() - 8, pic.height() // 3)
    assert on.copy(*top) == off.copy(*top)                      # the caption sits low, the top third is untouched
    mon.btn_caps.click()
    st.set_playhead(2.2)                                       # outside the cue: nothing drawn
    assert mon.wait_idle()
    after = mon.view.grab().toImage()
    mon.btn_caps.click()
    assert mon.view.grab().toImage() == after


@pytest.mark.ffmpeg
def test_safe_area_guides_toggle(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=2.0, size=(1080, 1920))
    mon, _pb = made(st)
    assert mon.wait_idle()
    plain = mon.view.grab().toImage()
    mon.btn_guides.click()
    guided = mon.view.grab().toImage()
    assert plain != guided and mon.view.guides
    mon.btn_guides.click()
    assert mon.view.grab().toImage() == plain


def test_banner_goes_above_the_picture(qapp, made, proj):
    st = short_state(proj)
    mon, _pb = made(st)
    banner = QLabel("匯入建議")
    mon.set_banner_widget(banner)
    assert mon.layout().itemAt(0).widget() is banner           # first thing in the column, above the picture
    assert banner.parentWidget() is mon


def test_transport_buttons_and_timecode(qapp, made, proj):
    st = short_state(proj, 1.0)
    mon, pb = made(st)
    mon.btn_fwd.click()
    assert st.playhead == pytest.approx(1 / 30)
    mon.btn_last.click()
    assert st.playhead == pytest.approx(1.0)
    mon.btn_first.click()
    assert st.playhead == 0.0
    mon.tc.setText("00:00:00:15")
    mon.tc.editingFinished.emit()
    assert st.playhead == pytest.approx(0.5)
    mon.btn_play.click()
    assert pb.is_playing
    mon.btn_play.click()
    assert not pb.is_playing
    mon.btn_loop.click()
    assert pb.loop
    pb.set_loop(False)
    assert not mon.btn_loop.isChecked()                        # the button follows the controller
    st.set_range(0.2, 0.4)
    mon.btn_range.click()
    assert pb.in_out_mode and st.playhead == pytest.approx(0.2)
    mon.btn_range.click()
    assert not pb.is_playing


def test_layout_regroups_rows_when_narrow(qapp, made, proj):
    st = short_state(proj)
    mon, _pb = made(st, 900, 640)
    assert mon._tr1.indexOf(mon.cache_group) >= 0 and mon._tr2.count() == 0         # wide: one transport row
    assert mon.btn_range.toolButtonStyle() == Qt.ToolButtonTextBesideIcon
    mon.resize(520, 640)
    qapp.processEvents()
    assert mon.width() == 520 and mon.layout().minimumSize().width() <= 520         # it can really get that narrow
    assert mon._tr2.indexOf(mon.cache_group) >= 0                                    # narrow: the cache group drops
    for w in (mon.transport_a, mon.cache_group, mon.btn_render, mon.lbl_cache, mon.tc):
        assert mon.rect().contains(w.mapTo(mon, w.rect().topLeft())) and \
            mon.rect().contains(w.mapTo(mon, w.rect().bottomRight()))                 # nothing is cut off
    mon.resize(430, 640)
    qapp.processEvents()
    assert mon.btn_range.toolButtonStyle() == Qt.ToolButtonIconOnly and mon.layout().minimumSize().width() <= 430
    mon.resize(900, 640)
    qapp.processEvents()
    assert mon._tr1.indexOf(mon.cache_group) >= 0 and mon.btn_render.text() == "渲染預覽（含聲音）"
    assert mon.lbl_cache.text() == "預覽快取：無" and "總長" in mon.lbl_info.text()


def test_shutdown_stops_the_decode_thread(qapp, made, proj):
    st = short_state(proj)
    mon, _pb = made(st)
    thread = mon._worker._thread
    assert thread.is_alive()
    mon.shutdown()
    assert not thread.is_alive()


# ============================================================================ reframe: geometry
def test_cropbox_matches_core_geometry():
    for zoom in (1.0, 1.7, 3.0):
        box = CropBox(1280, 720, 1080, 1920, 0.2, 0.5, zoom)
        x, y, w, h = box.rect()
        gx, gy, gw, gh = geo.crop_rect_px(1280, 720, 1080, 1920, 0.2, 0.5, zoom)
        assert (abs(x - gx) <= 2 and abs(y - gy) <= 2 and abs(w - gw) <= 2 and abs(h - gh) <= 2)
        assert w / h == pytest.approx(1080 / 1920, rel=0.01)


def test_cropbox_stays_inside_frame_and_normalises_centre():
    box = CropBox(1280, 720, 1080, 1920, 0.0, 0.9, 1.0)
    cx, cy = box.centre()
    assert cx == pytest.approx(202.5 / 1280) and cy == pytest.approx(0.5)      # clamped like the renderer does
    box.normalise()
    assert (box.cx, box.cy) == pytest.approx((cx, cy))
    box.move_to(5000, -40)
    x, y, w, h = box.rect()
    assert 0 <= x and x + w <= 1280 + 1e-6 and y == 0
    box.set_zoom(99)
    assert box.zoom == monitor_mod.MAX_ZOOM
    box.set_zoom(0.2)
    assert box.zoom == 1.0


def test_cropbox_corner_drag_keeps_the_opposite_corner():
    box = CropBox(1280, 720, 1080, 1920, 0.5, 0.5, 1.0)
    anchor = box.anchor_for("tl")
    x0, y0, w0, h0 = box.rect()
    assert (x0, y0, w0, h0) == pytest.approx((437.5, 0.0, 405.0, 720.0))
    assert anchor == pytest.approx((842.5, 720.0))              # the opposite corner = bottom-right
    box.resize_corner("tl", 600.0, 300.0, anchor)               # drag the top-left corner inward
    x, y, w, h = box.rect()
    assert box.zoom > 1.0 and (x + w, y + h) == pytest.approx((x0 + w0, y0 + h0), abs=1e-6)
    assert w / h == pytest.approx(1080 / 1920)
    box.resize_corner("tl", 840.0, 718.0, anchor)               # almost onto the anchor -> capped at the maximum zoom
    assert box.zoom == pytest.approx(monitor_mod.MAX_ZOOM)


def test_fit_rect_letterboxes_on_whole_pixels():
    from PySide6.QtCore import QRectF
    r = fit_rect(16, 9, QRectF(0, 0, 400, 400))
    assert (r.width(), r.height()) == (400, 225) and r.y() == 88
    r = fit_rect(9, 16, QRectF(0, 0, 400, 400))
    assert (r.width(), r.height()) == (225, 400) and r.x() == 88


# ============================================================================ reframe: editing
@pytest.fixture
def reframe(qapp, made, real_proj):
    """Monitor in reframe mode on a 3 s ``cover`` clip of the 1280x720 test clip, 1080x1920 sequence."""
    st, cid = build(real_proj, seconds=3.0, size=(1080, 1920))
    mon, pb = made(st, 900, 640)
    assert mon.wait_idle()
    st.select([cid])
    mon.btn_reframe.click()
    assert mon.reframing and mon.wait_idle() and mon.view.source is not None
    return st, cid, mon, pb


def display_scale(view) -> float:
    return view.source_rect().width() / 1280.0


@pytest.mark.ffmpeg
def test_reframe_button_enabled_only_for_one_cover_clip(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=2.0)
    mon, _pb = made(st)
    assert not mon.btn_reframe.isEnabled()                     # nothing selected
    st.select([cid])
    assert mon.btn_reframe.isEnabled()
    second = st.op("放置", "place_clip", [a.id for a in real_proj.assets.values() if a.name == "silent.mp4"][0],
                   st.sequence().video_tracks[1].id, 0.0, 0.0, 1.0)
    st.select([cid, second.id])
    assert not mon.btn_reframe.isEnabled()                     # two clips
    st.select([second.id])
    assert mon.btn_reframe.isEnabled()
    st.op("填滿方式", "set_clip_props", second.id, fit="contain")
    assert not mon.btn_reframe.isEnabled()                     # not a cover clip
    audio = st.op("放置", "place_clip", [a.id for a in real_proj.assets.values() if a.name == "music.wav"][0],
                  st.sequence().audio_tracks[0].id, 0.0, 0.0, 1.0)
    st.select([audio.id])
    assert not mon.btn_reframe.isEnabled()                     # audio clip
    assert mon.wait_idle()                                     # let the last edit's frame arrive before the shot
    shot(mon, "reframe_disabled")


@pytest.mark.ffmpeg
def test_reframe_shows_whole_source_with_output_aspect_box_and_inset(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    assert v.rf_active and mon.reframe_bar.isVisible() and mon.inset.isVisible() and mon.btn_reframe.isChecked()
    sr, br = v.source_rect(), v.box_rect()
    assert sr.width() / sr.height() == pytest.approx(16 / 9, abs=0.02)         # the WHOLE source frame
    assert br.width() / br.height() == pytest.approx(9 / 16, abs=0.01)         # window has the output aspect
    assert br.center().x() == pytest.approx(sr.center().x(), abs=1) and br.height() == pytest.approx(sr.height(), abs=1)
    f = mon.inset.frame
    assert f is not None and f.width() / f.height() == pytest.approx(9 / 16, abs=0.01)   # real output composition
    assert mon.inset.info[1] == "縮放 1.00×"


@pytest.mark.ffmpeg
def test_reframe_requested_signal_enters_and_escape_leaves(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=2.0)
    mon, _pb = made(st)
    assert mon.wait_idle()
    st.request_reframe(cid)
    assert mon.reframing and st.selection == {cid}              # the clip got selected too
    QTest.keyClick(mon.view, Qt.Key_Escape)
    assert not mon.reframing and not mon.btn_reframe.isChecked() and not mon.reframe_bar.isVisible()
    st.request_reframe(cid)
    mon.btn_done.click()
    assert not mon.reframing
    st.op("填滿方式", "set_clip_props", cid, fit="contain")
    msgs = []
    st.status.connect(msgs.append)
    st.request_reframe(cid)                                     # not a cover clip: refused with a reason
    assert not mon.reframing and any("cover" in m for m in msgs)


@pytest.mark.ffmpeg
def test_reframe_drag_adds_a_key_at_the_playhead(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    s = display_scale(v)
    c = v.box_rect().center()
    drag(v, c, c + QPointF(100, 0))
    clip = get_clip(st, cid)
    assert len(clip.transform.keys) == 1
    t, kx, ky = clip.transform.keys[0]
    assert t == 0.0
    assert kx == pytest.approx((640 + 100 / s) / 1280, abs=2e-3) and ky == pytest.approx(0.5, abs=1e-3)
    assert clip.transform.zoom == 1.0
    assert st.history.undo_label == "取景關鍵格"
    assert mon.wait_idle()
    bx = v.box_rect().center().x()
    assert bx == pytest.approx(c.x() + 100, abs=1.5)            # the box now shows what was stored
    st.undo()                                                   # ONE undo step
    assert get_clip(st, cid).transform.keys == []
    assert v.box_rect().center().x() == pytest.approx(c.x(), abs=1.5)
    st.redo()
    assert len(get_clip(st, cid).transform.keys) == 1


@pytest.mark.ffmpeg
def test_reframe_key_follows_the_playhead_time(reframe):
    st, cid, mon, _pb = reframe
    st.set_playhead(1.0)
    v = mon.view
    c = v.box_rect().center()
    drag(v, c, c + QPointF(-60, 0))
    keys = get_clip(st, cid).transform.keys
    assert len(keys) == 1 and keys[0][0] == pytest.approx(1.0)       # clip-local seconds = playhead - clip start
    st.set_playhead(2.0)
    c2 = v.box_rect().center()
    drag(v, c2, c2 + QPointF(80, 0))
    keys = get_clip(st, cid).transform.keys
    assert [round(k[0], 3) for k in keys] == [1.0, 2.0] and keys[0][1] < 0.5 < keys[1][1]
    st.set_playhead(1.5)                                        # between the keys the box shows the interpolation
    mid = (keys[0][1] + keys[1][1]) / 2
    assert v.box.centre()[0] == pytest.approx(mid, abs=1e-3)


@pytest.mark.ffmpeg
def test_reframe_drag_is_clamped_inside_the_frame(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    c = v.box_rect().center()
    drag(v, c, c + QPointF(5000, 0))
    t, kx, ky = get_clip(st, cid).transform.keys[0]
    assert kx == pytest.approx(1 - 202.5 / 1280, abs=1e-3)       # the window stops at the right edge


@pytest.mark.ffmpeg
def test_reframe_click_without_moving_changes_nothing(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    before = st.history.undo_label
    c = v.box_rect().center()
    drag(v, c, c, steps=1)
    assert get_clip(st, cid).transform.keys == [] and st.history.undo_label == before


@pytest.mark.ffmpeg
def test_reframe_click_outside_jumps_the_window(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    sr = v.source_rect()
    target = QPointF(sr.left() + sr.width() * 0.8, sr.center().y())
    drag(v, target, target, steps=1)
    t, kx, ky = get_clip(st, cid).transform.keys[0]
    assert kx == pytest.approx(0.8, abs=2e-3)


@pytest.mark.ffmpeg
def test_reframe_auto_key_off_sets_static_values(reframe):
    st, cid, mon, _pb = reframe
    mon.chk_auto.setChecked(False)
    v = mon.view
    s = display_scale(v)
    c = v.box_rect().center()
    drag(v, c, c + QPointF(-90, 0))
    clip = get_clip(st, cid)
    assert clip.transform.keys == []                            # no key: a static crop centre
    assert clip.transform.cx == pytest.approx((640 - 90 / s) / 1280, abs=2e-3)
    assert clip.transform.cy == pytest.approx(0.5, abs=1e-3)
    assert st.history.undo_label == "取景"
    st.undo()
    assert get_clip(st, cid).transform.cx == 0.5


@pytest.mark.ffmpeg
def test_reframe_wheel_zoom_is_one_undo_step_and_waits_for_the_pause(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    c = v.box_rect().center()
    wheel(v, c, 120)
    wheel(v, c, 120)
    assert get_clip(st, cid).transform.zoom == 1.0              # only a ghost until the commit
    assert v.box.zoom == pytest.approx(1.12 ** 2)
    assert v.box_rect().width() < 0.5625 * v.source_rect().height()
    mon.flush_pending()
    clip = get_clip(st, cid)
    assert clip.transform.zoom == pytest.approx(1.12 ** 2) and clip.transform.keys == []    # zoom only: no key
    assert st.history.undo_label == "取景縮放"
    st.undo()
    assert get_clip(st, cid).transform.zoom == 1.0
    wheel(v, c, -120)                                           # zooming out below 1x is impossible: nothing to store
    mon.flush_pending()
    assert get_clip(st, cid).transform.zoom == 1.0 and st.history.undo_label != "取景縮放"


@pytest.mark.ffmpeg
def test_reframe_wheel_commits_by_itself_after_a_pause(qapp, reframe):
    st, cid, mon, _pb = reframe
    c = mon.view.box_rect().center()
    wheel(mon.view, c, 120)
    assert pump(qapp, lambda: get_clip(st, cid).transform.zoom > 1.0, 3)


@pytest.mark.ffmpeg
def test_reframe_corner_drag_zoom_and_move_is_one_undo_step(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    tl = v.box_rect().topLeft()
    drag(v, tl, tl + QPointF(25, 60))                           # pull the top-left handle inward
    clip = get_clip(st, cid)
    assert clip.transform.zoom > 1.0 and len(clip.transform.keys) == 1
    t, kx, ky = clip.transform.keys[0]
    x, y, w, h = CropBox(1280, 720, 1080, 1920, kx, ky, clip.transform.zoom).rect()
    assert (x + w, y + h) == pytest.approx((842.5, 720.0), abs=1.5)       # the opposite corner did not move
    assert st.history.undo_label == "取景關鍵格"
    st.undo()                                                   # key AND zoom go away together
    clip = get_clip(st, cid)
    assert clip.transform.keys == [] and clip.transform.zoom == 1.0


@pytest.mark.ffmpeg
def test_reframe_arrow_keys_nudge_and_commit_once(reframe):
    st, cid, mon, _pb = reframe
    v = mon.view
    for _ in range(3):
        QTest.keyClick(v, Qt.Key_Right)
    assert get_clip(st, cid).transform.keys == []
    mon.flush_pending()
    keys = get_clip(st, cid).transform.keys
    assert len(keys) == 1 and keys[0][1] == pytest.approx(0.5 + 3 * 0.005, abs=1e-4)
    QTest.keyClick(v, Qt.Key_Down, Qt.ShiftModifier)            # Shift = bigger step
    mon.flush_pending()
    keys = get_clip(st, cid).transform.keys
    assert len(keys) == 1 and keys[0][2] == pytest.approx(0.5, abs=1e-4)     # 16:9 source: the window cannot move down
    QTest.keyClick(v, Qt.Key_Plus)
    mon.flush_pending()
    assert get_clip(st, cid).transform.zoom == pytest.approx(1.05)


@pytest.mark.ffmpeg
def test_reframe_keys_navigate_delete_clear_and_undo(reframe):
    st, cid, mon, _pb = reframe
    for t, cx in ((0.0, 0.30), (1.0, 0.50), (2.0, 0.70)):
        st.op("關鍵格", "set_transform_key", cid, t, cx, 0.5)
    assert mon.wait_idle()
    assert not mon.chk_auto.isEnabled() and mon.chk_auto.isChecked()        # keys exist: auto-key is forced on
    assert not mon.btn_prev_key.isEnabled() and mon.btn_next_key.isEnabled()
    assert mon.btn_del_key.isEnabled()                                       # the playhead sits on the key at 0
    mon.btn_next_key.click()
    assert st.playhead == pytest.approx(1.0) and mon.btn_prev_key.isEnabled()
    mon.btn_next_key.click()
    assert st.playhead == pytest.approx(2.0) and not mon.btn_next_key.isEnabled()
    mon.btn_prev_key.click()
    assert st.playhead == pytest.approx(1.0)
    assert mon.btn_del_key.isEnabled()
    mon.btn_del_key.click()                                     # remove_transform_key at the playhead
    assert [round(k[0], 3) for k in get_clip(st, cid).transform.keys] == [0.0, 2.0]
    assert not mon.btn_del_key.isEnabled()                      # no key under the playhead any more
    st.set_playhead(0.5)
    assert not mon.btn_del_key.isEnabled()
    assert mon.view.box.centre()[0] == pytest.approx(0.30 + (0.70 - 0.30) * 0.25, abs=1e-3)
    mon.btn_clear_keys.click()                                  # clear_transform_keys
    assert get_clip(st, cid).transform.keys == [] and mon.chk_auto.isEnabled()
    st.undo()
    assert len(get_clip(st, cid).transform.keys) == 2
    st.undo()
    assert len(get_clip(st, cid).transform.keys) == 3
    assert len(mon.view.keys) == 3                                          # the camera path is drawn


@pytest.mark.ffmpeg
def test_reframe_keeps_user_choice_of_auto_key_across_clips_without_keys(reframe):
    st, cid, mon, _pb = reframe
    mon.chk_auto.setChecked(False)
    st.op("關鍵格", "set_transform_key", cid, 0.5, 0.4, 0.5)
    assert mon.chk_auto.isChecked() and not mon.chk_auto.isEnabled()
    st.op("清除", "clear_transform_keys", cid)
    assert mon.chk_auto.isEnabled() and not mon.chk_auto.isChecked()        # the user's choice came back


@pytest.mark.ffmpeg
def test_reframe_leaves_when_selection_or_clip_changes(reframe):
    st, cid, mon, _pb = reframe
    msgs = []
    st.status.connect(msgs.append)
    st.clear_selection()
    assert not mon.reframing
    st.request_reframe(cid)
    assert mon.reframing
    st.op("刪除", "remove_clips", [cid])
    assert not mon.reframing


@pytest.mark.ffmpeg
def test_reframe_follows_selection_to_another_cover_clip(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=2.0)
    port = [a.id for a in real_proj.assets.values() if a.name == "port.mp4"][0]
    other = st.op("放置", "place_clip", port, st.sequence().video_tracks[0].id, 2.0, 0.0, 2.0)
    mon, _pb = made(st)
    st.select([cid])
    mon.btn_reframe.click()
    assert mon.reframe_clip_id == cid
    st.select([other.id])
    assert mon.reframing and mon.reframe_clip_id == other.id
    assert st.playhead == pytest.approx(2.0)                    # moved into the new clip
    assert mon.wait_idle() and mon.view.source is not None
    assert mon.view.source.width() < mon.view.source.height()   # the portrait clip's whole frame


@pytest.mark.ffmpeg
def test_reframe_playhead_outside_clip_blocks_editing(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=1.0)
    silent = [a.id for a in real_proj.assets.values() if a.name == "silent.mp4"][0]
    st.op("放置", "place_clip", silent, st.sequence().video_tracks[0].id, 1.0, 0.0, 2.0)
    mon, _pb = made(st)
    st.select([cid])
    mon.btn_reframe.click()
    assert mon.wait_idle()
    assert mon.view.editable
    st.set_playhead(1.5)
    assert mon.wait_idle()
    assert not mon.view.editable and "播放頭不在" in mon.view.rf_note
    c = mon.view.rect().center()
    drag(mon.view, QPointF(c), QPointF(c) + QPointF(40, 0))
    assert get_clip(st, cid).transform.keys == []              # ignored
    st.set_playhead(0.5)
    assert mon.view.editable


@pytest.mark.ffmpeg
def test_clip_local_time_uses_clip_start_and_clamps(qapp, real_proj):
    st, cid = build(real_proj, seconds=2.0)
    st.op("移動", "move_clip", cid, 1.0)
    seq, clip = st.sequence(), get_clip(st, cid)
    assert clip_local_time(seq, clip, 1.0) == 0.0 and clip_local_time(seq, clip, 2.0) == pytest.approx(1.0)
    assert clip_local_time(seq, clip, 0.2) == 0.0
    assert clip_local_time(seq, clip, 9.0) == pytest.approx(2.0 - 1 / 30)


# ============================================================================ screenshots (looked at by a person)
@pytest.mark.ffmpeg
def test_screenshots(qapp, made, real_proj):
    st, cid = build(real_proj, seconds=3.0, size=(1080, 1920))
    st.op("字幕樣式", "set_caption_style", size=0.04, max_chars=9, position="bottom", box=True, box_opacity=0.55)
    st.op("字幕", "add_cue", 0.0, 2.5, "這是一行測試字幕，會自動換行")
    mon, pb = made(st, 1000, 700)
    assert mon.wait_idle()
    shot(mon, "normal")
    mon.btn_guides.click()
    shot(mon, "guides")
    mon.btn_guides.click()
    mon.resize(520, 640)
    qapp.processEvents()
    assert mon.wait_idle()
    shot(mon, "narrow")
    mon.resize(1000, 700)
    qapp.processEvents()
    st.select([cid])
    mon.btn_reframe.click()
    assert mon.wait_idle()
    for t, cx in ((0.0, 0.25), (1.5, 0.55), (2.5, 0.75)):
        st.op("關鍵格", "set_transform_key", cid, t, cx, 0.5)
    st.set_playhead(1.5)
    assert mon.wait_idle()
    shot(mon, "reframe")
    v = mon.view
    c = v.box_rect().center()
    mouse(v, QEvent.MouseButtonPress, c)                        # a drag in progress: the inset follows the mouse
    mouse(v, QEvent.MouseMove, c + QPointF(70, 0), Qt.NoButton, Qt.LeftButton)
    shot(mon, "reframe_dragging")
    mouse(v, QEvent.MouseButtonRelease, c + QPointF(70, 0), Qt.LeftButton, Qt.NoButton)
    assert mon.wait_idle()
