"""End-to-end acceptance: the user's whole job through the REAL main window, the REAL engine and the REAL ffmpeg.

Flow A   one 60 s 16:9 video is dropped on the window -> suggestion banner -> 「自動剪成短影音」 (real analysis, real
         audit) -> review page -> 套用 -> 9:16 sequences -> 輸出 (default 短影音 preset, Chinese folder name) -> technical
         QA -> the exported file is probed AND its pictures are compared with the SOURCE (the clips are solid colours,
         so "the right part of the source is on screen" is measurable) -> the person's review click -> save / reopen ->
         undo / redo.
Flow B   four raw clips dropped -> 「自動合併成長片」 -> one long sequence -> corrected by hand with the keyboard
         (S = split, Shift+Delete = ripple delete) -> 輸出 -> the file is as long as the CORRECTED timeline; a second
         export of the unchanged sequence comes from the segment cache.
Flow C   the manual road (no auto-edit): banner 「依順序放到時間軸」 -> 輸出 -> the file is the two clips back to back.

What this does NOT prove: that an edit is *good* (taste), the Windows look-and-feel, hardware encoders, speech
recognition.  The machine only records ``technical_qa``; ``human_review`` stays "pending" until the PERSON clicks.
Screenshots: ``screenshots/e2e_<name>.png`` (git-ignored).
"""
from __future__ import annotations

import faulthandler
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl                                   # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent                                             # noqa: E402
from PySide6.QtTest import QTest                                                                  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QMainWindow,    # noqa: E402
                               QMessageBox)

from autocut.analysis import speech as speech_mod                                                 # noqa: E402
from autocut.core.project import validate_sequence                                                # noqa: E402
from autocut.gui import main_window as mw                                                         # noqa: E402
from autocut.gui.autoedit_dialog import PAGE_REVIEW, AssembleDialog, ShortsDialog                # noqa: E402
from autocut.gui.export_dialog import PAGE_RESULT, ExportDialog                                   # noqa: E402
from autocut.media.probe import probe_asset                                                       # noqa: E402
from autocut.util import PROJECT_EXT                                                              # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
pytestmark = [pytest.mark.gui, pytest.mark.ffmpeg, pytest.mark.e2e]

SHOTS = Path(__file__).resolve().parents[1] / "screenshots"
BURST = "sine=frequency=440:sample_rate=48000:duration={d},volume='if(lt(mod(t,2),1),1,0)':eval=frame"
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-g", "15", "-c:a", "aac"]
BLOCKS = ["red", "green", "blue", "yellow", "purple", "orange"]                  # the 10 s blocks of long.mp4
NAMED = {"red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
         "purple": (128, 0, 128), "orange": (255, 165, 0)}
FRAME_TOL = 0.2                                      # seconds: output length vs timeline length (frame rounding, muxing)


# ================================================================================================ helpers
def shot(widget, name: str) -> Path:
    SHOTS.mkdir(exist_ok=True)
    path = SHOTS / f"e2e_{name}.png"
    assert widget.grab().save(str(path)), path
    return path


def pump(app, cond, timeout: float = 120.0, what: str = "condition") -> None:
    """Process events until ``cond()``: the background tasks deliver their results through the event loop."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.004)
    raise TimeoutError(f"timed out waiting for {what}")


def click(btn) -> None:
    assert btn.isVisibleTo(btn.window()), f"{btn.text()!r} is not visible"
    assert btn.isEnabled(), f"{btn.text()!r} is disabled"
    QTest.mouseClick(btn, Qt.LeftButton)


def pick(combo, value) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"{value!r} not in combo")


def v1(state) -> list:
    return sorted(state.sequence().video_tracks[0].clips, key=lambda c: c.start)


def all_clips(state) -> list:
    return [c for t in state.sequence().tracks() for c in sorted(t.clips, key=lambda c: c.start)]


def layout(state) -> str:
    """One line per track: ``V1: a.mp4[0.00-5.00] b.mp4[4.60-9.60] ...`` (for assertion messages)."""
    seq = state.sequence()
    names = {a.id: a.name for a in state.project.assets.values()}
    lines = []
    for tr in seq.tracks():
        parts = [f"{names.get(c.asset_id, c.asset_id)}[{c.start:.2f}-{c.end:.2f}]" for c in sorted(tr.clips, key=lambda c: c.start)]
        lines.append(f"{tr.name}: " + " ".join(parts))
    return "\n".join(lines)


def sequence_ids(state) -> list[str]:
    return [s.id for s in state.project.sequences]


def pixel_at(path: Path, t: float) -> tuple[int, int, int]:
    """The average colour of the frame at ``t`` seconds of a video file (one pixel, via ffmpeg)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                          "-vf", "scale=1:1:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    assert len(out) == 3, f"no frame at {t:.2f}s of {path.name}"
    return out[0], out[1], out[2]


def nearest_name(rgb: tuple[int, int, int]) -> str:
    return min(NAMED, key=lambda n: sum((a - b) ** 2 for a, b in zip(NAMED[n], rgb)))


def monitor_image(win, timeout: float = 30.0):
    """The picture the program monitor shows now (waits until its newest frame request was answered)."""
    assert win.monitor.wait_idle(timeout), "the monitor never finished its frame"
    img = win.monitor.view.image
    assert img is not None and not img.isNull(), f"the monitor shows no picture: {win.monitor.view.message!r}"
    return img


def centre_rgb(img) -> tuple[int, int, int]:
    c = img.pixelColor(img.width() // 2, img.height() // 2)
    return c.red(), c.green(), c.blue()


def brightness(img) -> float:
    """Mean of (r+g+b)/3 over a 24x24 grid of pixels (0 = black, 255 = white)."""
    total, n = 0.0, 0
    for iy in range(24):
        for ix in range(24):
            c = img.pixelColor((ix * 2 + 1) * img.width() // 48, (iy * 2 + 1) * img.height() // 48)
            total += (c.red() + c.green() + c.blue()) / 3
            n += 1
    return total / n


def block_at(src_t: float) -> str | None:
    """Which 10 s colour block of long.mp4 is on screen at source time ``src_t`` (None: too close to a boundary)."""
    block = min(int(src_t // 10), len(BLOCKS) - 1)
    if (block > 0 and src_t - block * 10 < 0.2) or (block + 1) * 10 - src_t < 0.2:
        return None
    return BLOCKS[block]


def mime_for(paths) -> QMimeData:
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


class Dnd:
    """A synthetic OS drag of files over the window: enter, then drop (events are sent straight to the window)."""

    def __init__(self, win, paths) -> None:
        self.win, self.mime, self.pos = win, mime_for(paths), QPoint(300, 300)

    def enter(self) -> None:
        QApplication.sendEvent(self.win, QDragEnterEvent(self.pos, Qt.CopyAction, self.mime, Qt.LeftButton, Qt.NoModifier))

    def drop(self) -> None:
        QApplication.sendEvent(self.win, QDropEvent(QPointF(self.pos), Qt.CopyAction, self.mime, Qt.LeftButton,
                                                    Qt.NoModifier))


def drop_and_import(qapp, win, paths) -> None:
    dnd = Dnd(win, paths)
    dnd.enter()
    dnd.drop()
    pump(qapp, lambda: not win.importer.busy and not win.importer.runners, timeout=90, what="the import")
    qapp.processEvents()


# ================================================================================================ fixtures
@pytest.fixture(autouse=True)
def watchdog():
    """A hung offscreen test must die with a traceback instead of blocking the run forever."""
    faulthandler.dump_traceback_later(420, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


class Modals:
    """Every question / message the app would have put in front of the person.  The happy paths must ask NOTHING."""

    def __init__(self) -> None:
        self.asks: list[tuple[str, str]] = []
        self.informs: list[tuple[str, str]] = []
        self.save_file = ""

    def ask(self, _parent, title, text, yes="確定", no="取消"):
        self.asks.append((title, text))
        return True

    def inform(self, _parent, title, text, warning=False):
        self.informs.append((title, text))

    def ask_save(self, _parent, _name):
        self.asks.append(("未儲存的變更", ""))
        return "discard"


@pytest.fixture
def modals(qapp, tmp_path, monkeypatch):
    """Isolated app data + settings; no modal dialog can open (it would hang the offscreen run)."""
    home = tmp_path / "apphome"
    home.mkdir()
    monkeypatch.setenv("AUTOCUT_HOME", str(home))
    settings = mw.app_settings()
    settings.clear()
    settings.sync()
    log = Modals()
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("autocut.gui"):
            continue
        for attr, fn in (("ask", log.ask), ("inform", log.inform), ("ask_save_changes", log.ask_save)):
            if callable(getattr(module, attr, None)):
                monkeypatch.setattr(module, attr, fn)

    def refuse(what):
        def _f(*_a, **_k):
            raise AssertionError(f"unexpected {what}")
        return staticmethod(_f)

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: (log.save_file, "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", refuse("QFileDialog.getOpenFileName"))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", refuse("QFileDialog.getOpenFileNames"))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", refuse("QFileDialog.getExistingDirectory"))
    monkeypatch.setattr(QInputDialog, "getText", refuse("QInputDialog.getText"))
    monkeypatch.setattr(QInputDialog, "getItem", refuse("QInputDialog.getItem"))
    monkeypatch.setattr(mw.QDesktopServices, "openUrl", staticmethod(lambda _url: True))
    yield log
    settings.clear()
    settings.sync()


@pytest.fixture(autouse=True)
def no_speech(monkeypatch):
    """Speech recognition is optional and needs a model download: these flows are about everything else."""
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (False, speech_mod.INSTALL_HINT))
    monkeypatch.setattr(speech_mod, "model_cached", lambda model_size="small": False)


class Dialogs:
    """Replaces ``MainWindow._run_dialog``: the dialog is shown (not ``exec``-ed, which would block) and handed over."""

    def __init__(self) -> None:
        self.shown: list = []

    def last(self, kind):
        found = [d for d in self.shown if isinstance(d, kind)]
        assert found, f"no {kind.__name__} was opened (opened: {[type(d).__name__ for d in self.shown]})"
        return found[-1]


@pytest.fixture
def dialogs(qapp, monkeypatch):
    log = Dialogs()

    def run(dialog) -> None:
        log.shown.append(dialog)
        if hasattr(dialog, "autoplay_previews"):
            dialog.autoplay_previews = False
        dialog.show()
        QTest.qWait(30)

    monkeypatch.setattr(mw.MainWindow, "_run_dialog", staticmethod(run))
    yield log
    for d in log.shown:
        try:
            d.hide()
            d.deleteLater()
        except RuntimeError:            # already deleted by Qt
            pass
    qapp.processEvents()


@pytest.fixture
def make_window(qapp, modals, dialogs):
    made: list[mw.MainWindow] = []

    def make(show: bool = True) -> mw.MainWindow:
        win = mw.MainWindow(None, None)
        made.append(win)
        win.resize(1500, 900)
        if show:
            win.show()
            QTest.qWaitForWindowActive(win, 2000)
        QTest.qWait(30)
        return win

    yield make
    for win in made:
        win._started = True
        win.state.editor.history.mark_saved()
        QMainWindow.close(win)
        win.deleteLater()
    qapp.processEvents()


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("e2e_media")

    def ff(name: str, *args: str) -> None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, str(d / name)], check=True)

    cmd: list[str] = []
    for col in BLOCKS:
        cmd += ["-f", "lavfi", "-i", f"color=c={col}:s=320x180:r=30:d=10"]
    ff("long.mp4", *cmd, "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*440*t)*0.5*lt(mod(t\\,5)\\,4):s=44100:d=60",
       "-filter_complex", "[0:v][1:v][2:v][3:v][4:v][5:v]concat=n=6:v=1:a=0[v]", "-map", "[v]", "-map", "6:a",
       *ENC, "-shortest")
    ff("a.mp4", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8", "-f", "lavfi", "-i", BURST.format(d=8),
       *ENC, "-shortest")
    ff("b.mp4", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=6,hue=h=90", "-f", "lavfi",
       "-i", BURST.format(d=6), *ENC, "-shortest")
    ff("c.mp4", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=7,hue=h=200", "-f", "lavfi",
       "-i", BURST.format(d=7), *ENC, "-shortest")
    ff("d.mp4", "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30:duration=5", "-f", "lavfi", "-i", BURST.format(d=5),
       *ENC, "-shortest")
    return d


def run_auto_edit(qapp, win, dialogs, kind, banner_kind: str, setup) -> object:
    """Banner -> auto-edit dialog -> ``setup(dialog)`` fills the form -> analyse -> review page -> apply.  Returns the dialog."""
    banner = win.banner
    assert banner.isVisible() and banner.kind == banner_kind, banner.kind
    click(banner.btn_auto)
    assert not banner.isVisible()
    dlg = dialogs.last(kind)
    setup(dlg)
    base = list(sequence_ids(win.state))
    revision = win.state.project.revision
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_REVIEW, timeout=180, what="the review page")
    pump(qapp, lambda: all(not r.thumb.pixmap().isNull() for r in dlg.rows), what="thumbnails")
    assert dlg.receipt is not None and dlg.receipt.accepted, [f.message for f in (dlg.receipt.findings if dlg.receipt else [])]
    assert win.state.project.revision == revision and sequence_ids(win.state) == base      # analysis never edits
    return dlg


def apply_auto_edit(qapp, dlg) -> list[str]:
    click(dlg.btn_apply)
    pump(qapp, lambda: dlg.result() == QDialog.Accepted, timeout=60, what="the apply")
    return list(dlg.created_sequence_ids)


def export_with_dialog(qapp, win, dialogs, out: Path, preset: str | None = None) -> ExportDialog:
    """Menu action 「輸出影片」 -> pick preset + file -> 開始輸出 -> wait for render AND technical QA."""
    win.act_export.trigger()
    dlg = dialogs.last(ExportDialog)
    assert dlg.page == 0 and dlg.btn_start.isEnabled()
    if preset:
        pick(dlg.preset, preset)
    out.parent.mkdir(parents=True, exist_ok=True)
    dlg.path_edit.setText(str(out))
    click(dlg.btn_start)
    assert dlg.running
    pump(qapp, lambda: dlg.page == PAGE_RESULT, timeout=360, what="the export and its technical check")
    assert dlg.result_title.text() == "輸出完成", dlg.result_info.text()
    return dlg


def assert_clean_run(modals: Modals) -> None:
    assert modals.asks == [], f"the app nagged the person: {modals.asks}"
    assert modals.informs == [], f"the app popped up messages: {modals.informs}"


# ================================================================================================ flow A
def test_flow_a_long_video_to_vertical_shorts_to_a_checked_file(qapp, media, tmp_path, modals, dialogs, make_window):
    win = make_window()
    state = win.state
    assert not state.project.assets and len(state.project.sequences) == 1
    base_ids = sequence_ids(state)

    # --- 1. the person drops one 16:9 video on the window
    drop_and_import(qapp, win, [media / "long.mp4"])
    assert len(state.project.assets) == 1
    aid = next(iter(state.project.assets))
    assert win.banner.isVisible() and win.banner.kind == "shorts" and win.banner.asset_ids == [aid]
    assert not all_clips(state), "the banner suggests, it never edits by itself"
    shot(win, "a1_banner")

    # --- 2. 「自動剪成短影音」: real analysis on the real file, then the review page
    def fill(dlg: ShortsDialog) -> None:
        w = dlg.form.w
        pick(w["captions"], "off")
        w["use_speech"].setChecked(False)
        w["count"].setValue(3)
        w["target_seconds"].setValue(15)
        dlg.form.advanced.set_expanded(True)
        w["min_gap_between"].setValue(2.0)

    dlg = run_auto_edit(qapp, win, dialogs, ShortsDialog, "shorts", fill)
    assert len(dlg.rows) >= 2 and dlg.selected_indexes() == [r.index for r in dlg.rows]       # everything ticked
    assert any("尚未經人工審片" in m for _s, m in dlg.findings.rows()), "the review page must say nobody reviewed it yet"
    shot(dlg, "a2_review")

    # --- 3. 套用: one undo step, the new 9:16 sequence is active and remembers which AI run made it
    new_ids = apply_auto_edit(qapp, dlg)
    apply_label = state.history.undo_label
    assert len(new_ids) == len(dlg.rows) and sequence_ids(state) == base_ids + new_ids
    assert state.seq_id == new_ids[0]
    assert win.sequence_tabs.tabs.count() == len(base_ids) + len(new_ids)
    assert win.sequence_tabs.tabs.currentIndex() == len(base_ids)
    seq = state.sequence()
    assert (seq.width, seq.height) == (1080, 1920) and seq.is_portrait
    assert 13.0 <= seq.duration <= 60.0, seq.duration            # the target (15 s) is a preference inside the platform band
    run_id = seq.meta["ai"]["run_id"]
    run = state.run_for_sequence()
    assert run["run_id"] == run_id and run["technical_qa"] == "not_run" and run["human_review"] == "pending"
    clips = v1(state)
    assert clips and all(c.asset_id == aid for c in all_clips(state) if c.asset_id)
    assert [i for i in validate_sequence(state.project, seq) if i.level == "error"] == []
    img = monitor_image(win)                                       # the program monitor shows the new short ...
    assert abs(img.width() / img.height() - 9 / 16) < 0.02, (img.width(), img.height())
    want = block_at(clips[0].src_at(0.0))
    assert want is None or nearest_name(centre_rgb(img)) == want, (centre_rgb(img), want)    # ... from the right place
    shot(win, "a3_after_apply")

    # --- 4. 輸出: default 短影音 preset, a folder name with Chinese and a space, real ffmpeg, then the technical QA
    out = tmp_path / "成品 資料夾" / "短影音 1.mp4"
    ex = export_with_dialog(qapp, win, dialogs, out)
    assert ex.preset.currentData() == "shorts_1080x1920"
    assert out.is_file() and ex.result.path == str(out)
    info = probe_asset(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.has_audio and info.has_video
    assert abs(info.duration - seq.duration) < FRAME_TOL, (info.duration, seq.duration)
    #     the technical QA is awake: this source is solid colour blocks, so it must warn about frozen pictures - and
    #     about nothing else.  (It says 「技術檢查有警告」, never anything like "approved".)
    assert ex.report is not None, ex.qa_status.text()
    checks = {c.name: c.status for c in ex.report.checks}
    assert ex.report.status == "warn" and checks["凍結畫面"] == "warn", ex.report.summary()
    assert all(v == "pass" for k, v in checks.items() if k != "凍結畫面"), ex.report.summary()
    assert "技術檢查有警告" in ex.qa_status.text() and "再自己看過成品" in ex.qa_status.text()
    shot(ex, "a4_export_result")

    #     the pictures in the file come from the right places in the SOURCE (solid colour blocks: red, green, ...)
    checked = 0
    for c in clips:
        t_mid = (c.start + c.end) / 2
        if any(o is not c and o.start <= t_mid < o.end for o in clips):
            continue                                                       # inside a crossfade: two sources mix
        src_t = c.src_at(t_mid)
        want = block_at(src_t)
        if want is None:
            continue                                                       # too close to a colour boundary
        got = nearest_name(pixel_at(out, t_mid))
        assert got == want, f"at {t_mid:.2f}s the file shows {got}, the source has {want} at {src_t:.2f}s"
        checked += 1
    assert checked >= 1, "no clip could be compared with the source"

    #     the machine recorded ITS check only; the review is still the person's
    run = state.run_for_sequence()
    assert run["technical_qa"] == "warn" and run["human_review"] == "pending"
    assert all(r["human_review"] == "pending" for r in state.project.meta["ai_runs"])
    assert ex.qa_table.item(ex.qa_table.rowCount() - 1, 1).text() == "人工審片"
    assert "尚未審片" in ex.qa_table.item(ex.qa_table.rowCount() - 1, 2).text()
    assert "尚未審片" in ex.review.verdict_label.text()
    assert state.history.undo_label == "記錄技術檢查結果"

    # --- 5. the PERSON watches the file and clicks 「通過」: only now does the review change
    click(ex.review.btn_ok)
    assert state.run_for_sequence()["human_review"] == "approved"
    assert "你已標記為「通過」" in ex.qa_table.item(ex.qa_table.rowCount() - 1, 2).text()
    ex.reject()
    assert not ex.isVisible()

    # --- 6. save, reopen in a second window: the sequences, the run and its verdicts survive
    project_file = tmp_path / "我的專案" / f"長片轉短影音{PROJECT_EXT}"
    project_file.parent.mkdir()
    modals.save_file = str(project_file)
    assert win.save_project_as() and project_file.is_file() and not state.dirty
    win2 = make_window()
    assert win2.open_project(str(project_file))
    assert sequence_ids(win2.state) == base_ids + new_ids
    run2 = win2.state.run_for_sequence(new_ids[0])
    assert run2["technical_qa"] == "warn" and run2["human_review"] == "approved"
    assert not any(a.missing for a in win2.state.project.assets.values())

    # --- 7. undo / redo walk back through exactly what happened (review click, QA record, the auto-edit)
    assert state.history.undo_label == "人工審片"
    win.undo()
    assert state.run_for_sequence()["human_review"] == "pending"
    win.undo()
    assert state.run_for_sequence()["technical_qa"] == "not_run"
    assert state.history.undo_label == apply_label
    win.undo()
    assert sequence_ids(state) == base_ids and not state.project.meta.get("ai_runs") and aid in state.project.assets
    assert win.sequence_tabs.tabs.count() == len(base_ids)
    for _ in range(3):
        win.redo()
    assert sequence_ids(state) == base_ids + new_ids
    assert state.run_for_sequence(new_ids[0])["human_review"] == "approved"

    assert_clean_run(modals)


# ================================================================================================ flow B
def test_flow_b_raw_clips_to_one_long_video_with_manual_fixes(qapp, media, tmp_path, modals, dialogs, make_window):
    win = make_window()
    state = win.state
    base_ids = sequence_ids(state)

    # --- 1. four raw clips (three landscape, one portrait) dropped together
    drop_and_import(qapp, win, [media / n for n in ("a.mp4", "b.mp4", "c.mp4", "d.mp4")])
    ids = {a.name: a.id for a in state.project.assets.values()}
    assert list(ids) == ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    assert win.banner.isVisible() and win.banner.kind == "assemble" and win.banner.asset_ids == list(ids.values())
    shot(win, "b1_banner")

    # --- 2. 「自動合併成長片」
    def fill(dlg: AssembleDialog) -> None:
        w = dlg.form.w
        w["target_seconds"].setValue(20)
        pick(w["order"], "as_given")
        pick(w["captions"], "off")
        pick(w["transition"], "crossfade")

    dlg = run_auto_edit(qapp, win, dialogs, AssembleDialog, "assemble", fill)
    assert len(dlg.rows) == 1 and dlg.usage.rowCount() == 4
    assert {r[0] for r in dlg.usage.rows_text()} == set(ids)
    shot(dlg, "b2_review")
    new_ids = apply_auto_edit(qapp, dlg)
    assert len(new_ids) == 1 and sequence_ids(state) == base_ids + new_ids and state.seq_id == new_ids[0]
    seq = state.sequence()
    assert seq.width > seq.height and seq.meta["ai"]["run_id"]
    assert state.run_for_sequence()["human_review"] == "pending"
    used = {c.asset_id for c in all_clips(state)}
    assert used == set(ids.values()), "every raw clip should be on the long timeline"
    long_before = seq.duration
    assert 8.0 < long_before <= 40.0, long_before
    assert brightness(monitor_image(win)) > 25
    shot(win, "b3_after_apply")

    # --- 3. fix it like in any editor: razor through the second clip with S, ripple-delete its second half
    second, third = v1(state)[1], v1(state)[2]
    cut = state.set_playhead(second.start + second.duration * 0.4)
    assert second.start + 0.5 < cut < second.end - 0.5
    overlap = max(0.0, second.end - third.start)                 # the crossfade into the third clip lives in the part we delete
    removed = second.end - cut - overlap                         # ... so only the exclusive part of it leaves the timeline
    win.timeline.setFocus()
    QTest.keyClick(win.timeline, Qt.Key_S)
    assert state.history.undo_label == "分割"
    right_pieces = [c.id for c in all_clips(state) if abs(c.start - cut) < 1e-3 and c.src_in > second.src_in]
    assert right_pieces, "S did not cut at the playhead"
    state.select(right_pieces)
    QTest.keyClick(win.timeline, Qt.Key_Delete, Qt.ShiftModifier)
    assert state.history.undo_label == "波紋刪除"
    fixed = state.sequence()
    assert fixed.duration == pytest.approx(long_before - removed, abs=0.1), (fixed.duration, long_before, removed, layout(state))
    assert [i for i in validate_sequence(state.project, fixed) if i.level == "error"] == []
    clips_v1 = v1(state)
    holes = [(a.end, b.start) for a, b in zip(clips_v1, clips_v1[1:]) if b.start > a.end + 1e-6]
    assert holes == [], f"the ripple delete left a hole in the picture: {holes}\n{layout(state)}"
    assert brightness(monitor_image(win)) > 25, "the picture at the playhead is black (a hole in the timeline?)"
    shot(win, "b4_after_manual_fix")

    # --- 4. 輸出 with the quick draft preset: the file is as long as the CORRECTED timeline
    out = tmp_path / "長片 輸出" / "合併長片.mp4"
    ex = export_with_dialog(qapp, win, dialogs, out, preset="draft_fast")
    info = probe_asset(out)
    assert (info.width, info.height) == (1280, 720) and info.has_audio
    assert abs(info.duration - fixed.duration) < FRAME_TOL, (info.duration, fixed.duration)
    assert info.duration < long_before - removed / 2, "the ripple delete did not reach the exported file"
    assert ex.report.status == "pass", "no black hole may be left by the ripple delete"
    assert ex.report is not None and ex.report.status == "pass", ex.report.summary() if ex.report else ex.qa_status.text()
    assert "技術檢查通過" in ex.qa_status.text() and "不代表剪得好看" in ex.qa_status.text()
    run = state.run_for_sequence()
    assert run["technical_qa"] == "pass" and run["human_review"] == "pending"
    first_cached = ex.result.cached
    ex.reject()

    # --- 5. the same sequence exported again is rendered from the segment cache
    ex2 = export_with_dialog(qapp, win, dialogs, tmp_path / "長片 輸出" / "合併長片 再來一次.mp4", preset="draft_fast")
    assert ex2.result.cached > first_cached and ex2.result.cached == ex2.result.segments, (first_cached, ex2.result.cached)
    assert abs(probe_asset(Path(ex2.result.path)).duration - info.duration) < 0.05
    ex2.reject()

    assert_clean_run(modals)


# ================================================================================================ flow C
def test_flow_c_manual_road_without_auto_edit(qapp, media, tmp_path, modals, dialogs, make_window):
    win = make_window()
    state = win.state

    drop_and_import(qapp, win, [media / "a.mp4", media / "b.mp4"])
    ids = [a.id for a in state.project.assets.values()]
    assert win.banner.isVisible() and win.banner.kind == "assemble"
    click(win.banner.btn_place)                                              # 「依順序放到時間軸」
    first, second = v1(state)
    assert (first.asset_id, second.asset_id) == (ids[0], ids[1])
    assert first.start == 0.0 and second.start == pytest.approx(first.end)
    seq = state.sequence()
    assert state.run_for_sequence() is None and "ai" not in seq.meta

    out = tmp_path / "手動剪輯" / "兩段.mp4"
    ex = export_with_dialog(qapp, win, dialogs, out, preset="draft_fast")
    info = probe_asset(out)
    assert (seq.width, seq.height) == (1920, 1080)
    assert (info.width, info.height) == (1280, 720) and info.has_audio        # draft preset: longest side 1280
    assert abs(info.duration - seq.duration) < FRAME_TOL, (info.duration, seq.duration)
    assert ex.report is not None and ex.report.status == "pass", ex.report.summary() if ex.report else ex.qa_status.text()
    assert "沒有審片紀錄" in ex.review.qa_label.text()                       # not made by the auto-editor: no verdict to give
    assert not state.project.meta.get("ai_runs")
    ex.reject()
    assert_clean_run(modals)
