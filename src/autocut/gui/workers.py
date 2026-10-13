"""Background work for the GUI.

Anything slower than ~50 ms (probing, thumbnails, analysis, rendering) runs through :class:`TaskRunner`.
The worker function receives ``(progress, cancel)``; ``progress(fraction, message)`` is thread-safe and
``cancel`` is a ``threading.Event`` that long loops / ffmpeg runs must honour.  All signals are delivered in the
GUI thread, so slots may touch widgets and the :class:`~autocut.gui.state.AppState` directly.

    runner = TaskRunner(lambda progress, cancel: do_work(progress, cancel), parent=self)
    runner.progress.connect(bar_slot)       # (float 0..1, str)
    runner.finished.connect(done_slot)      # (object result)
    runner.failed.connect(error_slot)       # (str message in 繁體中文)
    runner.cancelled.connect(cancel_slot)
    runner.start()
    ...
    runner.cancel()
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from ..media.ffmpeg import Cancelled
from ..util import app_data_dir

log = logging.getLogger("autocut.gui")
_ACTIVE: set["TaskRunner"] = set()          # keep runners alive until they finish


def log_path():
    d = app_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / "gui.log"


def log_exception(where: str) -> str:
    """Append the current traceback to the GUI log; returns the formatted text."""
    text = traceback.format_exc()
    try:
        with open(log_path(), "a", encoding="utf-8") as f:
            f.write(f"--- {where}\n{text}\n")
    except OSError:
        pass
    log.error("%s\n%s", where, text)
    return text


def friendly_error(exc: BaseException) -> str:
    """User-facing message: our own errors already carry 繁體中文 text, anything else gets a generic prefix."""
    from ..core.ops import OpError
    from ..core.plan import PlanError
    from ..media.ffmpeg import FFmpegError
    from ..media.probe import ProbeError
    own: tuple = (OpError, PlanError, FFmpegError, ProbeError, ValueError, FileNotFoundError, KeyError)
    try:
        from ..autoedit.api import AutoEditError
        own = own + (AutoEditError,)
    except Exception:                                    # pragma: no cover - engine may be absent in tests
        pass
    try:
        from ..analysis.speech import SpeechUnavailable
        own = own + (SpeechUnavailable,)
    except Exception:
        pass
    msg = str(exc).strip("'\"") or exc.__class__.__name__
    return msg if isinstance(exc, own) else f"發生未預期的錯誤（{exc.__class__.__name__}）：{msg}"


class _Job(QRunnable):
    def __init__(self, runner: "TaskRunner") -> None:
        super().__init__()
        self.runner = runner
        self.setAutoDelete(True)

    def run(self) -> None:                                # worker thread
        r = self.runner
        try:
            self._run(r)
        except RuntimeError:                              # the runner's QObject went away with the window: nobody listens
            pass
        finally:
            r._done.set()

    @staticmethod
    def _run(r: "TaskRunner") -> None:
        if r._cancel.is_set():                            # cancelled while it still waited for a free thread: never start
            r.cancelled.emit()
            return
        try:
            result = r._fn(r._emit_progress, r._cancel)
            if r._cancel.is_set():
                r.cancelled.emit()
            else:
                r.finished.emit(result)
        except Cancelled:
            r.cancelled.emit()
        except BaseException as exc:                      # noqa: BLE001 - report everything to the GUI
            log_exception(r.name or "task")
            if r._cancel.is_set():
                r.cancelled.emit()
            else:
                r.failed.emit(friendly_error(exc))


class TaskRunner(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    done = Signal()                                       # emitted after finished/failed/cancelled

    def __init__(self, fn: Callable[[Callable[[float, str], None], threading.Event], Any],
                 parent: Optional[QObject] = None, name: str = "") -> None:
        super().__init__(parent)
        self._fn = fn
        self.name = name
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._started = False
        self.finished.connect(self._finish)
        self.failed.connect(self._finish)
        self.cancelled.connect(self._finish)

    # ---- control
    def start(self) -> "TaskRunner":
        if self._started:
            return self
        self._started = True
        _ACTIVE.add(self)
        QThreadPool.globalInstance().start(_Job(self))
        return self

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel

    @property
    def running(self) -> bool:
        return self._started and not self._done.is_set()

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until the worker function has returned (tests / shutdown). Signals still need an event loop."""
        return self._done.wait(timeout)

    # ---- internals
    def _emit_progress(self, fraction: float, message: str = "") -> None:
        self.progress.emit(max(0.0, min(1.0, float(fraction))), message)

    def _finish(self, *_a) -> None:                       # GUI thread (queued)
        _ACTIVE.discard(self)
        self.done.emit()


def run_task(fn: Callable, *, parent: Optional[QObject] = None, on_progress: Optional[Callable] = None,
             on_done: Optional[Callable] = None, on_error: Optional[Callable] = None,
             on_cancel: Optional[Callable] = None, name: str = "") -> TaskRunner:
    """Convenience wrapper: build a runner, connect optional callbacks and start it."""
    r = TaskRunner(fn, parent, name)
    if on_progress:
        r.progress.connect(on_progress)
    if on_done:
        r.finished.connect(on_done)
    if on_error:
        r.failed.connect(on_error)
    if on_cancel:
        r.cancelled.connect(on_cancel)
    return r.start()


def active_runners() -> list["TaskRunner"]:
    """The background tasks that have been started and have not finished (a snapshot)."""
    return [r for r in list(_ACTIVE) if r.running]


def shutdown(timeout: float = 10.0) -> bool:
    """The window is closing: ask every running background task to stop and wait (``timeout`` seconds in all) for the worker
    functions to return - a render or an analysis that kept going would keep its ffmpeg child and the process alive after the
    window is gone.  ``True`` when everything has stopped.  Signals still in flight are not delivered (the loop is ending)."""
    runners = active_runners()
    for r in runners:
        r.cancel()
    end = time.monotonic() + max(0.0, timeout)
    stopped = True
    for r in runners:
        if not r.wait(max(0.0, end - time.monotonic())):
            stopped = False
    return stopped


def wait_for_idle(app, runners, timeout: float = 30.0) -> None:
    """Test helper: process events until every runner has finished and its signals were delivered."""
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if all(not r.running for r in runners) and not any(r in _ACTIVE for r in runners):
            app.processEvents()
            return
        time.sleep(0.005)
    raise TimeoutError("背景工作逾時")
