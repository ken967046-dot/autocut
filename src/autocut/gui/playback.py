"""Playback clock + optional preview audio for the monitor (GUI_SPEC §3).

``PlaybackController`` advances ``state.playhead`` from a ``time.perf_counter`` clock.  The playhead is always where
the wall clock says it should be: when the monitor is slow to decode a frame it simply shows the next one it can
(frames are dropped, playback never lags behind).  The controller never decodes anything itself – the monitor
reacts to ``state.playhead_changed``.

Audio: live frames are silent.  When an up-to-date preview render exists (``render.runner.render_preview`` wrote it
for the *same* ``project.revision``) its audio is played through ``QMediaPlayer`` + ``QAudioOutput`` and re-synced
whenever it drifts more than 80 ms from the clock.  ANY problem (no QtMultimedia, offscreen platform, no audio
device, missing codec, playback error) falls back to silent playback and says so once on ``state.status``.

The module imports and works without QtMultimedia: it is imported lazily inside the audio backend only.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QGuiApplication

from ..core.timecode import EPS
from .state import AppState

DRIFT_SECONDS = 0.080          # re-sync the audio when it is further than this from the clock
TICK_MS_MIN = 4

PREVIEW_FRESH = "fresh"
PREVIEW_STALE = "stale"
PREVIEW_RENDERING = "rendering"
PREVIEW_NONE = "none"


# ------------------------------------------------------------------------------ audio backends
class AudioUnavailable(RuntimeError):
    """The preview audio cannot be played on this machine (the message is Traditional Chinese)."""


class QtAudioOutput:
    """``QMediaPlayer`` + ``QAudioOutput`` playing the audio of the preview mp4.

    No video sink is attached, so the picture of the file is never shown (the monitor draws its own frames).
    Raises :class:`AudioUnavailable` from the constructor when there is nothing to play on.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        self.on_error: Optional[Callable[[str], None]] = None
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
        except Exception as exc:                                         # ImportError / OSError (missing libs)
            raise AudioUnavailable(f"這個環境沒有 QtMultimedia（{exc.__class__.__name__}）") from exc
        app = QGuiApplication.instance()
        if app is None or app.platformName() in ("offscreen", "minimal"):
            raise AudioUnavailable("目前的顯示平台沒有音訊輸出")
        try:
            if not QMediaDevices.audioOutputs():
                raise AudioUnavailable("找不到音訊輸出裝置")
            self._media_status_invalid = QMediaPlayer.MediaStatus.InvalidMedia
            self._out = QAudioOutput(parent)
            self._out.setMuted(False)
            self._player = QMediaPlayer(parent)
            self._player.setAudioOutput(self._out)
            self._player.errorOccurred.connect(self._error)
            self._player.mediaStatusChanged.connect(self._status)
        except AudioUnavailable:
            raise
        except Exception as exc:                                         # noqa: BLE001
            raise AudioUnavailable(f"無法建立音訊播放器（{exc}）") from exc

    def _error(self, _code: Any, message: str = "") -> None:
        if self.on_error:
            self.on_error(message or "未知的音訊錯誤")

    def _status(self, status: Any) -> None:
        if status == self._media_status_invalid and self.on_error:
            self.on_error("預覽檔的聲音格式無法解碼")

    def load(self, path: str) -> None:
        self._player.setSource(QUrl.fromLocalFile(path))

    def play(self) -> None:
        self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def position(self) -> float:
        return self._player.position() / 1000.0

    def set_position(self, t: float) -> None:
        self._player.setPosition(int(round(max(0.0, t) * 1000)))

    def release(self) -> None:
        """Let go of the file (Windows cannot overwrite a file that is still open)."""
        self._player.stop()
        self._player.setSource(QUrl())

    def close(self) -> None:
        try:
            self.release()
        except Exception:                                                # noqa: BLE001
            pass


def create_qt_audio(parent: Optional[QObject] = None) -> QtAudioOutput:
    return QtAudioOutput(parent)


class DriftGuard:
    """Decides when the audio has to be re-seeked (pure logic, no Qt).

    ``check(now, audio_pos, clock_t)`` returns the position to seek to, or ``None``.  It looks at most every
    ``interval`` seconds, ignores the first ``grace`` seconds after a start / seek (the player needs time to
    begin) and never re-seeks twice within ``min_gap`` seconds, so a laggy backend is not seeked into stutter.
    """

    def __init__(self, threshold: float = DRIFT_SECONDS, interval: float = 0.25, grace: float = 0.4,
                 min_gap: float = 0.5) -> None:
        self.threshold, self.interval, self.grace, self.min_gap = threshold, interval, grace, min_gap
        self._started = 0.0
        self._last_check = 0.0
        self._last_seek = -1e9

    def reset(self, now: float) -> None:
        self._started = self._last_check = now
        self._last_seek = -1e9

    def check(self, now: float, audio_pos: float, clock_t: float) -> Optional[float]:
        if now - self._started < self.grace or now - self._last_check < self.interval:
            return None
        self._last_check = now
        if abs(audio_pos - clock_t) <= self.threshold or now - self._last_seek < self.min_gap:
            return None
        self._last_seek = now
        self._started = now
        return clock_t


@dataclass(frozen=True)
class PreviewRecord:
    """One finished ``render_preview``: the file and the exact project state it was rendered from."""
    path: str
    project_id: str
    seq_id: str
    revision: int
    duration: float = 0.0


# ------------------------------------------------------------------------------ controller
class PlaybackController(QObject):
    """Play / pause / seek the playhead of ``state`` (see the module docstring).

    Signals: ``playing_changed(bool)``, ``loop_changed(bool)``, ``finished()`` (stopped at the end),
    ``preview_status_changed(str)`` (``fresh | stale | rendering | none``), ``audio_changed(bool)``.
    ``audio_factory(parent) -> backend`` is injectable for tests; a backend offers ``load(path) play() pause()
    position() set_position(t) release() close()`` and an ``on_error`` callback attribute.
    """

    playing_changed = Signal(bool)
    loop_changed = Signal(bool)
    finished = Signal()
    preview_status_changed = Signal(str)
    audio_changed = Signal(bool)

    def __init__(self, state: AppState, parent: Optional[QObject] = None,
                 audio_factory: Optional[Callable[[QObject], Any]] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._audio_factory = audio_factory or create_qt_audio
        self._playing = False
        self._loop = False
        self._in_out = False
        self._bounds = (0.0, 0.0)
        self._anchor_t = 0.0
        self._anchor_wall = 0.0
        self._in_tick = False
        self._duration = 0.0
        self._fps = 30.0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        # preview cache (one record per project + sequence)
        self._previews: dict[tuple[str, str], PreviewRecord] = {}
        self._rendering = False
        self._last_status = PREVIEW_NONE
        # audio
        self._audio: Any = None
        self._audio_path = ""
        self._audio_on = False
        self._audio_disabled = False          # backend could not be created on this machine
        self._broken_path = ""                # preview file the backend failed on
        self._warned: set[str] = set()
        self._drift = DriftGuard()
        state.playhead_changed.connect(self._on_playhead)
        state.edited.connect(self._on_edited)
        state.sequence_changed.connect(self._on_sequence)
        state.project_replaced.connect(self._on_sequence)
        self._refresh_info()
        self._last_status = self.preview_status()

    # ------------------------------------------------------------------ public: state
    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def loop(self) -> bool:
        return self._loop

    @property
    def in_out_mode(self) -> bool:
        """True while the current run plays only the in -> out range."""
        return self._playing and self._in_out

    @property
    def audio_active(self) -> bool:
        """True while the audio of the preview file is playing along (False = silent live playback)."""
        return self._audio_on

    def set_loop(self, on: bool) -> None:
        on = bool(on)
        if on != self._loop:
            self._loop = on
            self.loop_changed.emit(on)

    # ------------------------------------------------------------------ public: transport
    def play(self, in_out: bool = False) -> bool:
        """Start playing from the playhead (from the start when it sits at the end).  ``in_out`` plays the in -> out
        range from its in point instead.  Returns False (and says why on ``state.status``) when nothing can play."""
        if self._playing:
            return True
        self._refresh_info()
        fps, dur = self._fps, self._duration
        if dur < 1.0 / fps - EPS:
            self.state.status.emit("序列是空的，沒有可以播放的內容")
            return False
        start, end, t = 0.0, dur, self.state.playhead
        use_range = False
        if in_out:
            rng = self.state.get_range()
            if rng is None:
                self.state.status.emit("尚未設定入點／出點（時間軸上按 I、O），改為播放整段")
            else:
                lo, hi = max(0.0, rng[0]), min(dur, rng[1])
                if hi - lo >= 1.0 / fps - EPS:
                    start, end, t, use_range = lo, hi, lo, True
                else:
                    self.state.status.emit("入出點在序列內容之外，改為播放整段")
        if t >= end - EPS or t < start - EPS:
            t = start
        self._bounds = (start, end)
        self._in_out = use_range
        self._playing = True
        self._set_playhead(t)
        self._anchor_t, self._anchor_wall = t, time.perf_counter()
        self._timer.start(max(TICK_MS_MIN, int(500.0 / fps)))
        self._start_audio(t)
        self.playing_changed.emit(True)
        return True

    def play_in_out(self) -> bool:
        return self.play(in_out=True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._in_out = False
        self._timer.stop()
        self._pause_audio()
        self.playing_changed.emit(False)

    def toggle(self) -> bool:
        """Play when paused, pause when playing.  Returns the new ``is_playing``."""
        if self._playing:
            self.pause()
        else:
            self.play()
        return self._playing

    def stop(self) -> None:
        """Pause (the playhead stays where it is) and let go of the audio file."""
        self.pause()
        self._release_audio_source()

    def seek(self, t: float) -> float:
        """Move the playhead (clamped to the sequence); playback continues from there when it is running."""
        self._refresh_info()
        t = max(0.0, min(float(t), self._duration))
        return self.state.set_playhead(t, snap=True)

    def step(self, frames: int) -> float:
        """Pause and move the playhead by whole frames."""
        self.pause()
        self._refresh_info()
        return self.seek(self.state.playhead + frames / self._fps)

    def goto_start(self) -> float:
        self.pause()
        return self.seek(0.0)

    def goto_end(self) -> float:
        self.pause()
        self._refresh_info()
        return self.seek(self._duration)

    def close(self) -> None:
        """Stop the clock and release the audio player (call when the window closes)."""
        self._timer.stop()
        self._playing = False
        self._release_audio_source()
        if self._audio is not None:
            try:
                self._audio.close()
            except Exception:                                            # noqa: BLE001
                pass
            self._audio = None

    # ------------------------------------------------------------------ public: preview cache bookkeeping
    def preview_status(self) -> str:
        """``fresh`` (rendered from the current revision), ``stale`` (the project changed since), ``rendering`` or
        ``none``."""
        if self._rendering:
            return PREVIEW_RENDERING
        rec = self._record()
        if rec is None:
            return PREVIEW_NONE
        return PREVIEW_FRESH if rec.revision == self.state.project.revision else PREVIEW_STALE

    def preview_path(self) -> Optional[str]:
        """The preview file whose audio matches the current project revision (None when there is none)."""
        rec = self._record()
        if rec is not None and rec.revision == self.state.project.revision:
            return rec.path
        return None

    def note_preview_rendered(self, path: str, revision: int, project_id: Optional[str] = None,
                              seq_id: Optional[str] = None, duration: float = 0.0) -> None:
        """Remember a finished preview render (``revision`` = ``project.revision`` it was rendered from)."""
        key = (project_id or self.state.project.id, seq_id or self.state.seq_id)
        self._previews[key] = PreviewRecord(str(path), key[0], key[1], int(revision), float(duration))
        self._rendering = False
        self._audio_path = ""                 # the file was rewritten: reload it
        self._broken_path = ""
        self._update_preview_status()
        if self._playing:                     # a render finished while playing: pick the audio up right away
            self._start_audio(self.state.playhead)

    def set_preview_rendering(self, on: bool) -> None:
        on = bool(on)
        if on == self._rendering:
            return
        self._rendering = on
        if on:
            self._pause_audio()
            self._release_audio_source()      # the render rewrites the file; Windows needs it closed
        self._update_preview_status()

    def clear_preview(self) -> None:
        self._previews.clear()
        self._release_audio_source()
        self._update_preview_status()

    # ------------------------------------------------------------------ clock
    def _refresh_info(self) -> None:
        try:
            seq = self.state.sequence()
        except KeyError:
            self._duration, self._fps = 0.0, 30.0
            return
        self._duration, self._fps = seq.duration, (seq.fps or 30.0)

    def _set_playhead(self, t: float) -> None:
        self._in_tick = True
        try:
            self.state.set_playhead(t, snap=True)
        finally:
            self._in_tick = False

    def _tick(self) -> None:
        if not self._playing:
            return
        now = time.perf_counter()
        t = self._anchor_t + (now - self._anchor_wall)
        start, end = self._bounds
        if t >= end - EPS:
            if self._loop and end - start > EPS:
                t = start + ((t - start) % (end - start))
                self._anchor_t, self._anchor_wall = t, now
                self._seek_audio(t)
            else:
                self._set_playhead(end)
                self.pause()
                self.finished.emit()
                return
        self._set_playhead(math.floor(t * self._fps + 1e-3) / self._fps)     # the frame whose interval holds t
        self._sync_audio(now, t)

    def _on_playhead(self, t: float) -> None:
        """Someone else moved the playhead (timeline click, timecode box…): continue from there."""
        if self._in_tick or not self._playing:
            return
        start, end = self._bounds
        if t >= self._duration - EPS:
            self.pause()
            return
        if t < start - EPS or t >= end - EPS:                 # left the in -> out range: carry on with the whole sequence
            self._bounds = (0.0, self._duration)
            self._in_out = False
        self._anchor_t, self._anchor_wall = t, time.perf_counter()
        self._seek_audio(t)

    def _on_edited(self, _kind: str = "", _label: str = "") -> None:
        self._refresh_info()
        if self._playing:
            self._follow_the_edit()
        if self._audio_on and self.preview_path() is None:    # the timeline changed under the preview audio
            self._pause_audio()
        self._update_preview_status()

    def _follow_the_edit(self) -> None:
        """The sequence (or the in -> out range) changed under a running transport: it plays to the NEW end, a loop wraps
        there, and a sequence that became empty stops it.  (The bounds used to be fixed when play was pressed, so playback ran
        on past the end of the picture, or stopped short of material that had just been added.)"""
        fps = self._fps
        if self._duration < 1.0 / fps - EPS:
            self.pause()
            self.state.status.emit("序列變成空的，播放已停止")
            return
        if self._in_out:
            rng = self.state.get_range()
            if rng is not None:
                lo, hi = max(0.0, rng[0]), min(self._duration, rng[1])
                if hi - lo >= 1.0 / fps - EPS:
                    self._bounds = (lo, hi)
                    return
            self._in_out = False                              # the range is gone or outside the sequence: the whole sequence
        self._bounds = (0.0, self._duration)

    def _on_sequence(self) -> None:
        self.pause()
        self._refresh_info()
        self._update_preview_status()

    def _update_preview_status(self) -> None:
        s = self.preview_status()
        if s != self._last_status:
            self._last_status = s
            self.preview_status_changed.emit(s)

    def _record(self) -> Optional[PreviewRecord]:
        try:
            rec = self._previews.get((self.state.project.id, self.state.seq_id))
        except Exception:                                      # noqa: BLE001
            return None
        return rec if rec is not None and os.path.isfile(rec.path) else None

    # ------------------------------------------------------------------ audio
    def _warn(self, key: str, message: str) -> None:
        """ONE status message per reason (playing again must not repeat it)."""
        if key not in self._warned:
            self._warned.add(key)
            self.state.status.emit(message)

    def _set_audio_on(self, on: bool) -> None:
        if on != self._audio_on:
            self._audio_on = on
            self.audio_changed.emit(on)

    def _ensure_audio(self) -> Any:
        if self._audio is not None:
            return self._audio
        if self._audio_disabled:
            return None
        try:
            audio = self._audio_factory(self)
        except Exception as exc:                               # noqa: BLE001 - ANY failure means: stay silent
            self._audio_disabled = True
            self._warn("unavailable", f"無法播放預覽聲音（{exc}），改為靜音播放")
            return None
        audio.on_error = self._on_audio_error
        self._audio = audio
        return audio

    def _start_audio(self, t: float) -> None:
        path = self.preview_path()
        if not path or path == self._broken_path:
            self._set_audio_on(False)
            return
        audio = self._ensure_audio()
        if audio is None:
            return
        try:
            if path != self._audio_path:
                audio.load(path)
                self._audio_path = path
            audio.set_position(t)
            audio.play()
        except Exception as exc:                               # noqa: BLE001
            self._on_audio_error(str(exc) or exc.__class__.__name__, path)
            return
        if self._broken_path == path:                          # the backend reported an error while starting
            return
        self._drift.reset(time.perf_counter())
        self._set_audio_on(True)

    def _pause_audio(self, force: bool = False) -> None:
        if self._audio is not None and (self._audio_on or force):
            try:
                self._audio.pause()
            except Exception:                                  # noqa: BLE001
                pass
        self._set_audio_on(False)

    def _release_audio_source(self) -> None:
        self._set_audio_on(False)
        if self._audio is not None and self._audio_path:
            try:
                self._audio.release()
            except Exception:                                  # noqa: BLE001
                pass
        self._audio_path = ""

    def _seek_audio(self, t: float) -> None:
        if self._audio is None or not self._audio_on:
            return
        try:
            self._audio.set_position(t)
        except Exception as exc:                               # noqa: BLE001
            self._on_audio_error(str(exc))
            return
        self._drift.reset(time.perf_counter())

    def _sync_audio(self, now: float, t: float) -> None:
        if self._audio is None or not self._audio_on:
            return
        try:
            target = self._drift.check(now, self._audio.position(), t)
            if target is not None:
                self._audio.set_position(target)
        except Exception as exc:                               # noqa: BLE001
            self._on_audio_error(str(exc))

    def _on_audio_error(self, message: str, path: str = "") -> None:
        """The backend failed on this file: stay silent for it, tell the user once."""
        self._broken_path = path or self._audio_path or self._broken_path
        self._warn(f"error:{self._broken_path}", f"預覽聲音播放失敗（{message}），改為靜音播放")
        self._pause_audio(force=True)
