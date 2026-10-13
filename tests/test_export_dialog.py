"""Export dialog (GUI_SPEC §7), offscreen: the REAL ``ExportDialog`` on tiny ffmpeg-synthesised media.

Groups
    pure helpers        safe_stem / unique_path / missing_media / default_output_dir / explain_os_error / has_sound
    settings page       presets -> output size + loudness, ``build_settings()`` == what the widgets show, range, encoder
    preflight           empty sequence, missing media, unusable output path -> nothing starts, the reason is shown
    real exports        ffmpeg renders; length / size / fps / audio are measured with ffprobe, SRT + burned captions
    cancel / closing    mid-render cancel (no half-written file, dialog usable again), asking before closing
    technical QA        recorded into ``ai_runs`` only for auto-edit sequences; wording is always 「技術檢查」
    human review        the verdict changes ONLY when the person clicks one of the three buttons (keyboard included)
    failure paths       ffmpeg missing / output locked / renderer raises -> plain message, buttons restored, no zombie task
    buttons / layout    result buttons, clock, default buttons (Enter), page sizes, screenshots + clipped-text check

Modal dialogs would hang an offscreen run, so every one of them goes through a module-level seam
(``ask`` / ``inform`` / ``open_folder`` / ``play_file`` / ``QFileDialog``) that is replaced by ``modals``.
Screenshots: ``screenshots/export_<name>.png`` (git-ignored).
"""
from __future__ import annotations

import copy
import errno
import faulthandler
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

import shiboken6                                                                          # noqa: E402
from PySide6.QtCore import QSettings, Qt, QThreadPool, QUrl                               # noqa: E402
from PySide6.QtGui import QColor, QGuiApplication                                         # noqa: E402
from PySide6.QtTest import QTest                                                          # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGroupBox, QLabel, QMessageBox,    # noqa: E402
                               QPushButton, QRadioButton, QTableWidget)

from autocut.core import ops                                                              # noqa: E402
from autocut.core import plan as planmod                                                  # noqa: E402
from autocut.core.plan import AuditReceipt, EditPlan                                      # noqa: E402
from autocut.core.project import Asset, Project                                           # noqa: E402
from autocut.gui import export_dialog as ed                                               # noqa: E402
from autocut.gui.export_dialog import PAGE_PROGRESS, PAGE_RESULT, PAGE_SETTINGS, ExportDialog   # noqa: E402
from autocut.gui.state import AppState                                                    # noqa: E402
from autocut.gui.widgets import QA_TEXT, VERDICT_TEXT                                      # noqa: E402
from autocut.gui.workers import wait_for_idle                                             # noqa: E402
from autocut.media.ffmpeg import FFmpegError, FFmpegMissing                               # noqa: E402
from autocut.media.probe import probe_asset                                               # noqa: E402
from autocut.render import runner as runner_mod                                           # noqa: E402
from autocut.render.compiler import MissingMediaError, RenderError                        # noqa: E402
from autocut.render.presets import PRESETS, output_size                                   # noqa: E402
from autocut.render.qa import Check, QAReport                                             # noqa: E402
from autocut.render.runner import RenderResult                                            # noqa: E402
from autocut.util import cache_dir                                                        # noqa: E402

pytestmark = pytest.mark.gui

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.ffmpeg                       # conftest skips these when ffmpeg / ffprobe are missing
SHOTS = Path(__file__).resolve().parents[1] / "screenshots"
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-g", "15", "-c:a", "aac"]
AI_REVIEW_PENDING = "尚未審片（機器不會代替你判斷好不好看）"
FORBIDDEN_CLAIMS = ("審片通過", "審核通過", "審片完成", "已審片", "人工審片通過", "已通過審")   # a machine must never say these
ILLEGAL_WINDOWS = set('\\/:*?"<>|')
EXPORT_FOLDERS = ("輸出 資料夾", "素材 資料夾")


# ================================================================================================ helpers
def pump(app, cond, timeout: float = 60.0, what: str = "condition") -> None:
    """Process events until ``cond()`` (background tasks deliver their signals through the event loop)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.004)
    raise TimeoutError(f"timed out waiting for {what}")


def click(btn) -> None:
    assert btn.isEnabled(), f"{btn.text()!r} is disabled"
    QTest.mouseClick(btn, Qt.LeftButton)


def pick(combo, value) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"{value!r} not in combo")


def mk_asset(aid: str, path, name: str = "", duration: float = 5.0) -> Asset:
    """A video asset that points at ``path`` (the file may or may not exist)."""
    return Asset(id=aid, path=str(path), name=name, kind="video", duration=duration, width=320, height=180, fps=30.0,
                 has_video=True, has_audio=True, audio_channels=2, sample_rate=48000, fingerprint=aid)


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def widget_texts(root) -> list[str]:
    """Every user-visible string below ``root``: labels, buttons, group titles, table cells, tooltips."""
    out: list[str] = []
    for w in [root, *root.findChildren(QLabel), *root.findChildren(QPushButton), *root.findChildren(QGroupBox)]:
        for attr in ("text", "title"):
            if hasattr(w, attr):
                out.append(getattr(w, attr)())
        if isinstance(w, QLabel) or isinstance(w, QPushButton):
            out.append(w.toolTip())
    for table in root.findChildren(QTableWidget):
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                it = table.item(r, c)
                if it is not None:
                    out += [it.text(), it.toolTip()]
    return [t for t in out if t]


def result_text(dlg) -> str:
    """Everything the result page says (so the wording rules can be checked in one place)."""
    page = dlg.pages.widget(PAGE_RESULT)
    return "\n".join(widget_texts(page))


def qa_rows(dlg) -> list[tuple[str, str, str]]:
    t = dlg.qa_table
    return [tuple(t.item(r, c).text() for c in range(3)) for r in range(t.rowCount())]


# --------------------------------------------------------------------------------------------- ffprobe
def probe_json(path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                       capture_output=True, check=True)
    return json.loads(p.stdout.decode("utf-8"))


def facts(path) -> dict:
    """What ffprobe says about a rendered file: length, size, fps, audio, frame count."""
    data = probe_json(path)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    num, den = v["r_frame_rate"].split("/")
    return {"duration": float(data["format"]["duration"]), "size": (int(v["width"]), int(v["height"])),
            "fps": float(num) / float(den), "audio": bool(audio), "frames": int(v.get("nb_frames") or 0),
            "vcodec": v["codec_name"], "acodec": audio[0]["codec_name"] if audio else ""}


def frame_at(path, t: float) -> np.ndarray:
    """One BGR frame at time ``t`` (accurate seek)."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-f", "rawvideo",
                        "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
    w, h = facts(path)["size"]
    return np.frombuffer(p.stdout, dtype=np.uint8).reshape(h, w, 3)


def mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ================================================================================================ fixtures
@pytest.fixture(autouse=True)
def watchdog():
    """A hung offscreen test must die with a traceback instead of blocking the run forever."""
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def clean_settings(qapp):
    def wipe():
        s = QSettings("AutoCutStudio", "AutoCutStudio")
        s.remove("export")
        s.sync()
    wipe()
    yield
    wipe()


class Modals:
    """Everything the dialog would have popped up or opened.  ``ask`` answers come from ``answers`` (default: no)."""

    def __init__(self) -> None:
        self.asks: list[tuple] = []          # (title, text, yes, no)
        self.informs: list[tuple] = []       # (title, text, warning)
        self.folders: list[str] = []
        self.played: list[str] = []
        self.saves: list[tuple] = []         # (start, filter)
        self.answers: dict[str, bool] = {}
        self.save_result = ""
        self.real_open_folder = ed.open_folder
        self.real_play_file = ed.play_file

    def yes(self, *titles: str) -> None:
        for t in titles:
            self.answers[t] = True

    def titles(self) -> list[str]:
        return [a[0] for a in self.asks]

    # the replacements
    def ask(self, parent, title, text, yes="確定", no="取消"):
        self.asks.append((title, text, yes, no))
        return self.answers.get(title, False)

    def inform(self, parent, title, text, warning=False):
        self.informs.append((title, text, warning))

    def save_dialog(self, parent=None, caption="", directory="", filter="", *a, **k):
        self.saves.append((caption, directory, filter))
        return self.save_result, ""


@pytest.fixture(autouse=True)
def modals(monkeypatch) -> Modals:
    m = Modals()
    monkeypatch.setattr(ed, "ask", m.ask)
    monkeypatch.setattr(ed, "inform", m.inform)
    monkeypatch.setattr(ed, "open_folder", lambda path: m.folders.append(path) or True)
    monkeypatch.setattr(ed, "play_file", lambda path: m.played.append(path) or True)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(m.save_dialog))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: m.informs.append(("QMessageBox.exec", self.text(), True)) or 0)
    return m


class ReviewSpy(list):
    """``core.plan.set_human_review`` calls.  A test that clicks a verdict button bumps ``expected``; at teardown the
    number of calls must equal it, i.e. nothing but a person's click ever changed a review."""
    expected = 0


@pytest.fixture(autouse=True)
def review_spy(monkeypatch):
    calls = ReviewSpy()
    real = planmod.set_human_review

    def spy(project, run_id, verdict, notes="", **kw):
        calls.append((run_id, verdict, notes))
        return real(project, run_id, verdict, notes, **kw)

    monkeypatch.setattr(planmod, "set_human_review", spy)
    yield calls
    assert len(calls) == calls.expected, f"unexpected human-review writes: {list(calls)}"


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    """Tiny synthetic clips (320x180 @ 30 fps, 3 s) made with ffmpeg lavfi; plus one bigger clip for slow renders."""
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("export_media")

    def ff(rel: str, *args: str) -> Path:
        target = d / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, str(target)], check=True)
        return target

    def tone(freq: int, dur: int) -> list[str]:
        return ["-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={dur}"]

    files = {
        "a": ff("a.mp4", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3", *tone(440, 3), *ENC, "-shortest"),
        "b": ff("b.mp4", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=3,hue=h=120", *tone(660, 3), *ENC,
                "-shortest"),
        "silent": ff("silent.mp4", "-f", "lavfi", "-i", "color=c=green:size=320x180:rate=30:duration=3", *ENC),
        "portrait": ff("portrait.mp4", "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=30:duration=3", *tone(550, 3), *ENC,
                       "-shortest"),
        "big": ff("big.mp4", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=12", *tone(330, 12), *ENC,
                  "-shortest"),
    }
    zh = d / EXPORT_FOLDERS[1] / "測試 影片.mp4"            # a source whose folder and file names have Chinese + spaces
    zh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(files["b"], zh)
    files["zh"] = zh
    return files


@pytest.fixture(scope="module")
def assets(media) -> dict[str, Asset]:
    return {k: probe_asset(p) for k, p in media.items()}


def add_ai_run(proj: Project, seq) -> dict:
    """Make ``seq`` look like the output of an auto-edit run (``ai_runs`` record: QA not_run, review pending)."""
    plan = EditPlan(kind="shorts")
    receipt = AuditReceipt(plan_sha256=plan.sha256(), project_revision=proj.revision,
                           project_sha256=proj.content_sha256(), accepted=True)
    run = planmod.record_run(proj, plan, receipt, [seq.id])
    seq.meta["ai"] = {"run_id": run["run_id"], "kind": "shorts"}
    return run


def build_state(assets: dict[str, Asset], *, names=("a", "b"), seconds: float = 1.5, size=(320, 180), fps: float = 30.0,
                seq_name: str = "測試序列", cues=(), style: dict | None = None, rng=None, ai: bool = False,
                project_path=None, look: dict | None = None) -> AppState:
    """A project with ``names`` laid end to end on V1 (``seconds`` each), optional cues / in-out range / auto-edit run."""
    proj = Project.new("export-gui")
    for n in names:
        a = Asset.from_dict(assets[n].to_dict())
        proj.assets[a.id] = a
    seq = proj.active_sequence()
    seq.name = seq_name
    ops.set_sequence_format(proj, seq, size[0], size[1], fps)
    v1 = seq.video_tracks[0]
    t = 0.0
    for n in names:
        clip = ops.place_clip(proj, seq, assets[n].id, v1.id, start=t, src_in=0.0, src_out=seconds)
        if look:
            for k, v in look.items():
                setattr(clip.look, k, v)
        t = clip.end
    for start, end, text in cues:
        ops.add_cue(proj, seq, start, end, text)
    if style:
        ops.set_caption_style(proj, seq, **style)
    if rng:
        ops.set_sequence_range(proj, seq, *rng)
    if ai:
        add_ai_run(proj, seq)
    if project_path:
        proj.path = str(project_path)
    return AppState(proj)


def fake_state(tmp_path: Path, *, with_clip: bool = True, exists: bool = True, size=(1920, 1080), cues=(),
               rng=None, ai: bool = False) -> tuple[AppState, Asset]:
    """A project that needs no ffmpeg: one fake clip (optionally pointing at a file that does not exist)."""
    proj = Project.new("export-fake")
    proj.path = str(tmp_path / "proj.acproj")                         # the default output folder is then tmp_path
    src = tmp_path / "src" / "clip.mp4"
    if exists:
        touch(src)
    a = mk_asset("a1", src, "clip.mp4")
    proj.assets[a.id] = a
    seq = proj.active_sequence()
    seq.name = "假序列"
    ops.set_sequence_format(proj, seq, size[0], size[1], 30.0)
    if with_clip:
        ops.place_clip(proj, seq, a.id, seq.video_tracks[0].id, start=0.0, src_in=0.0, src_out=4.0)
    for start, end, text in cues:
        ops.add_cue(proj, seq, start, end, text)
    if rng:
        ops.set_sequence_range(proj, seq, *rng)
    if ai:
        add_ai_run(proj, seq)
    return AppState(proj), a


def settle(app, dlg, timeout: float = 20.0) -> None:
    """Let a dialog's background work finish (cancelling it when it still runs) so tests cannot bleed into each other."""
    r = dlg.runner
    if r is not None and r.running:
        r.cancel()
        try:
            wait_for_idle(app, [r], timeout)
        except TimeoutError:
            pass
    dlg.timer.stop()
    app.processEvents()


@pytest.fixture
def make_dialog(qapp):
    made: list[ExportDialog] = []

    def make(state: AppState, show: bool = True) -> ExportDialog:
        dlg = ExportDialog(state)
        made.append(dlg)
        if show:
            dlg.show()
            qapp.processEvents()
        return dlg

    yield make
    for dlg in made:
        settle(qapp, dlg)
        dlg.hide()
        dlg.deleteLater()
    QThreadPool.globalInstance().waitForDone(20000)          # e.g. the encoder probe every dialog starts when it opens
    qapp.processEvents()


def pick_output(dlg: ExportDialog, path) -> str:
    """Type an output path into the dialog (what the person would do) and return the path as given."""
    dlg.path_edit.setText(str(path))
    return str(path)


def finish(app, dlg: ExportDialog, timeout: float = 90.0) -> None:
    """Wait until the dialog has left the progress page (result, failure or back at the settings)."""
    pump(app, lambda: not dlg.running and dlg.page != PAGE_PROGRESS, timeout, "the export to end")


def export_and_wait(app, dlg: ExportDialog, timeout: float = 90.0) -> None:
    assert dlg.start_export(), "start_export() was refused"
    finish(app, dlg, timeout)


def has_ffmpeg_children() -> bool:
    """Linux only: is an ffmpeg process still a child of this test process?"""
    if not Path("/proc").is_dir():
        return False
    out = subprocess.run(["pgrep", "-P", str(os.getpid()), "-x", "ffmpeg"], capture_output=True, text=True)
    return bool(out.stdout.strip())


# ================================================================================================ pure helpers
def test_safe_stem_removes_windows_illegal_characters():
    s = ed.safe_stem('a<b>c:d"e/f\\g|h?i*j')
    assert s == "a_b_c_d_e_f_g_h_i_j" and not (ILLEGAL_WINDOWS & set(s))
    assert ed.safe_stem("a\tb\nc\x00d") == "a_b_c_d"                 # control characters too
    assert ed.safe_stem("ab???cd") == "ab_cd"                        # a run of bad characters is one underscore


def test_safe_stem_trims_what_windows_silently_drops():
    assert ed.safe_stem("  name  ") == "name"
    assert ed.safe_stem("name. ") == "name"                          # a trailing dot / space vanishes on Windows
    assert ed.safe_stem("my.final.cut") == "my.final.cut"            # inner dots are fine


@pytest.mark.parametrize("raw", ["", "   ", "...", "???", " / \\ ", "._ ."])
def test_safe_stem_never_returns_an_empty_name(raw):
    assert ed.safe_stem(raw) == "輸出"


def test_safe_stem_keeps_chinese_and_inner_spaces():
    assert ed.safe_stem("我的 影片:第1集") == "我的 影片_第1集"          # ASCII colon is illegal ...
    assert ed.safe_stem("我的 影片：第1集") == "我的 影片：第1集"          # ... the full-width one is a legal character
    assert ed.safe_stem("日本語のタイトル／テスト") == "日本語のタイトル／テスト"   # full-width slash is legal as well


def windows_reserved(filename: str) -> bool:
    """CON, PRN, AUX, NUL, COM0-9, LPT0-9 (also with an extension: ``NUL.mp4``) cannot be created on Windows."""
    head = filename.split(".")[0].rstrip(" ")
    return bool(re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])", head))


@pytest.mark.parametrize("name", ["CON", "con", "PRN", "Aux", "NUL", "nul", "COM1", "com9", "LPT1", "lpt9", "CON.final",
                                  "nul.test", "Com3 ", "aux.", "COM¹"])
def test_safe_stem_avoids_windows_reserved_device_names(name):
    stem = ed.safe_stem(name)
    assert not windows_reserved(stem + ".mp4"), f"{name!r} -> {stem!r} still names a device on Windows"
    assert stem[:3].upper() == name[:3].upper()                      # still recognisable as what the user typed


@pytest.mark.parametrize("name", ["CONSOLE", "COM10", "LPT", "AUXILIARY", "NULL", "COMET", "my CON", "CON-trol", "PRNT"])
def test_safe_stem_leaves_lookalike_names_alone(name):
    assert ed.safe_stem(name) == name


def test_unique_path_never_overwrites_and_counts_up(tmp_path):
    assert ed.unique_path(tmp_path, "影片") == str(tmp_path / "影片.mp4")
    touch(tmp_path / "影片.mp4")
    assert ed.unique_path(tmp_path, "影片") == str(tmp_path / "影片 (2).mp4")
    touch(tmp_path / "影片 (2).mp4")
    assert ed.unique_path(tmp_path, "影片") == str(tmp_path / "影片 (3).mp4")
    touch(tmp_path / "影片 (4).mp4")                                  # a gap is reused, a later number is not skipped
    assert ed.unique_path(tmp_path, "影片") == str(tmp_path / "影片 (3).mp4")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["影片 (2).mp4", "影片 (4).mp4", "影片.mp4"]   # it only looks


def test_unique_path_works_with_other_extensions_and_spaces_and_missing_folders(tmp_path):
    folder = tmp_path / "輸出 資料夾" / "子 資料夾"                      # does not exist yet: still a plain string answer
    assert ed.unique_path(folder, "a b", ".srt") == str(folder / "a b.srt")
    assert not folder.exists()
    touch(tmp_path / "x.srt")
    assert ed.unique_path(str(tmp_path), "x", ".srt") == str(tmp_path / "x (2).srt")
    assert isinstance(ed.unique_path(tmp_path, "x"), str)


def project_with_clips(tmp_path: Path, spec: list[tuple]) -> tuple[Project, "ed.Sequence"]:
    """``spec``: (asset id, file exists?, name, clip kwargs) -> a V1 clip each; every asset belongs to the project."""
    proj = Project.new("miss")
    seq = proj.active_sequence()
    for i, (aid, exists, name, kw) in enumerate(spec):
        path = tmp_path / f"{aid}.mp4"
        if exists:
            touch(path)
        proj.assets[aid] = mk_asset(aid, path, name)
        if kw is not None:
            clip = ops.place_clip(proj, seq, aid, seq.video_tracks[0].id, start=float(i * 5), src_in=0.0, src_out=4.0)
            for k, v in kw.items():
                setattr(clip, k, v)
    return proj, seq


def test_missing_media_lists_only_used_assets_whose_file_is_gone(tmp_path):
    proj, seq = project_with_clips(tmp_path, [("a1", True, "ok.mp4", {}), ("a2", False, "gone.mp4", {}),
                                              ("a3", False, "unused-and-gone.mp4", None)])
    assert ed.missing_media(proj, seq) == ["gone.mp4"]


def test_missing_media_is_sorted_by_asset_id_and_falls_back_to_the_file_name(tmp_path):
    proj, seq = project_with_clips(tmp_path, [("a9", False, "z.mp4", {}), ("a1", False, "", {}), ("a5", False, "m.mp4", {})])
    assert ed.missing_media(proj, seq) == ["a1.mp4", "m.mp4", "z.mp4"]      # a1 has no name -> its file name
    proj2, seq2 = project_with_clips(tmp_path / "none", [("a1", True, "x.mp4", {})])
    assert ed.missing_media(proj2, seq2) == []


def test_missing_media_counts_an_asset_once_however_often_it_is_used(tmp_path):
    proj, seq = project_with_clips(tmp_path, [("a1", False, "gone.mp4", {})])
    ops.place_clip(proj, seq, "a1", seq.video_tracks[0].id, start=10.0, src_in=0.0, src_out=2.0)
    ops.place_clip(proj, seq, "a1", seq.video_tracks[1].id, start=0.0, src_in=0.0, src_out=2.0)
    assert ed.missing_media(proj, seq) == ["gone.mp4"]


def test_missing_media_ignores_what_the_renderer_ignores(tmp_path):
    """A disabled clip, or a clip on a hidden / muted track, is never read by the renderer: it must not block an export."""
    proj, seq = project_with_clips(tmp_path, [("a1", False, "disabled.mp4", {"enabled": False}),
                                              ("a2", False, "enabled.mp4", {})])
    assert ed.missing_media(proj, seq) == ["enabled.mp4"]
    seq.video_tracks[0].muted = True                                  # hide V1: nothing on it is rendered any more
    assert ed.missing_media(proj, seq) == []


def test_missing_media_skips_clips_that_point_at_unknown_assets(tmp_path):
    proj, seq = project_with_clips(tmp_path, [("a1", True, "ok.mp4", {})])
    clip = ops.place_clip(proj, seq, "a1", seq.video_tracks[0].id, start=8.0, src_in=0.0, src_out=2.0)
    clip.asset_id = "ghost"                                           # the renderer reports this one itself
    assert ed.missing_media(proj, seq) == []


def test_default_output_dir_prefers_the_folder_of_the_saved_project(qapp, tmp_path):
    folder = tmp_path / "專案 資料夾"
    folder.mkdir()
    last = tmp_path / "last"
    last.mkdir()
    QSettings("AutoCutStudio", "AutoCutStudio").setValue(ed.LAST_DIR_KEY, str(last))
    assert ed.default_output_dir(str(folder / "我的 專案.acproj")) == str(folder)


def test_default_output_dir_then_last_export_folder(qapp, tmp_path):
    last = tmp_path / "last"
    last.mkdir()
    QSettings("AutoCutStudio", "AutoCutStudio").setValue(ed.LAST_DIR_KEY, str(last))
    assert ed.default_output_dir("") == str(last)                                         # an unsaved project
    assert ed.default_output_dir(str(tmp_path / "gone" / "p.acproj")) == str(last)       # its folder no longer exists


def test_default_output_dir_then_videos_then_home(qapp, tmp_path, monkeypatch):
    videos = tmp_path / "Videos"
    videos.mkdir()
    where = {"path": str(videos)}

    class Paths:
        MoviesLocation = 0

        @staticmethod
        def writableLocation(_loc):
            return where["path"]

    monkeypatch.setattr(ed, "QStandardPaths", Paths)
    assert ed.default_output_dir("") == str(videos)
    QSettings("AutoCutStudio", "AutoCutStudio").setValue(ed.LAST_DIR_KEY, str(tmp_path / "deleted"))   # stale: ignored
    assert ed.default_output_dir("") == str(videos)
    where["path"] = str(tmp_path / "no such folder")
    assert ed.default_output_dir("") == str(Path.home())
    where["path"] = ""
    assert ed.default_output_dir("") == str(Path.home())


def test_explain_os_error_permission_names_the_file_and_the_likely_cause():
    msg = ed.explain_os_error(PermissionError(errno.EACCES, "Permission denied"), "/some/folder/我的 影片.mp4")
    assert "我的 影片.mp4" in msg and "/some/folder" not in msg           # the file name, not the whole path
    assert "無法寫入" in msg and "播放器" in msg and "權限" in msg
    assert "Permission denied" not in msg                                # no raw English in the user interface


@pytest.mark.parametrize("code,english,expect", [(errno.ENOSPC, "No space left on device", "磁碟空間"),
                                                 (errno.ENOENT, "No such file or directory", "資料夾"),
                                                 (errno.ENOTDIR, "Not a directory", "資料夾"),
                                                 (errno.EROFS, "Read-only file system", "唯讀"),
                                                 (errno.ENAMETOOLONG, "File name too long", "太長")])
def test_explain_os_error_common_failures_are_plain_chinese(code, english, expect):
    msg = ed.explain_os_error(OSError(code, english), r"C:\輸出\影片.mp4")
    assert "影片.mp4" in msg and expect in msg
    assert english not in msg


def test_explain_os_error_never_loses_an_unknown_reason():
    msg = ed.explain_os_error(OSError(errno.EIO, "Input/output error"), "/x/y.mp4")
    assert "y.mp4" in msg and "Input/output error" in msg                 # unknown errors keep the system's own words
    assert "boom" in ed.explain_os_error(OSError("boom"), "/x/y.mp4")



# ================================================================================================ fakes for the render / QA
class FakeRender:
    """Stands in for ``render_sequence``: reports progress, can block on ``gate`` (cancel-aware), fail, or write a file."""

    def __init__(self, monkeypatch, *, steps: int = 4, gate: threading.Event | None = None,
                 error: BaseException | None = None, write: bool = True, duration: float = 3.0, fps: float = 30.0,
                 size=(320, 180), warnings=(), payload: bytes = b"fake-mp4", sets_cancel: bool = False):
        self.calls: list[dict] = []
        self.steps, self.gate, self.error, self.write = steps, gate, error, write
        self.sets_cancel = sets_cancel                  # like render_sequence: a failing segment sets the cancel event
        self.duration, self.fps, self.size, self.warnings, self.payload = duration, fps, size, list(warnings), payload
        self.reached = threading.Event()               # set once the first progress report went out
        monkeypatch.setattr(ed, "render_sequence", self)

    def __call__(self, proj, seq, out_path, settings=None, progress=None, cancel=None):
        from autocut.media.ffmpeg import Cancelled
        self.calls.append({"out": str(out_path), "settings": settings, "seq_id": seq.id, "proj": proj})
        for i in range(self.steps):
            progress(i / self.steps, f"渲染片段 {i + 1}/{self.steps}")
            self.reached.set()
            if self.gate is not None and i == 1:
                end = time.monotonic() + 60
                while not self.gate.is_set() and time.monotonic() < end:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    time.sleep(0.005)
            if cancel is not None and cancel.is_set():
                raise Cancelled()
        if self.error is not None:
            if self.sets_cancel and cancel is not None:
                cancel.set()
            raise self.error
        if self.write:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(self.payload)
        progress(1.0, "完成")
        return RenderResult(str(out_path), self.duration, int(self.duration * self.fps), self.size[0], self.size[1],
                            self.fps, 2, 1, 0.5, "libx264", "ffmpeg (fake)", list(self.warnings))


def qa_report(path: str = "x.mp4", *checks: tuple[str, str, str]) -> QAReport:
    return QAReport(path, [Check(n, s, d) for n, s, d in checks] or
                    [Check("可讀取", "pass", "320x180、3.00 秒、h264"), Check("完整解碼", "pass", "沒有錯誤")])


def fake_qa(monkeypatch, report: QAReport | BaseException | None = None, gate: threading.Event | None = None) -> list:
    """Replace the QA pass; ``gate`` keeps it 'running' until set.  Returns the list of (path, expect_size, expect_fps)."""
    calls: list = []

    def job(res, expect_size, expect_fps, progress, _cancel):
        calls.append((res.path, tuple(expect_size), expect_fps, res.duration))
        progress(0.0, "技術檢查中…")
        if gate is not None:
            end = time.monotonic() + 60
            while not gate.is_set() and time.monotonic() < end:
                time.sleep(0.005)
        if isinstance(report, BaseException):
            raise report
        return report or qa_report(res.path)

    monkeypatch.setattr(ed, "qa_job", job)
    return calls


# ================================================================================================ settings page
def test_dialog_opens_on_the_settings_page_with_sensible_defaults(qapp, make_dialog, assets, tmp_path):
    state = build_state(assets, seq_name="我的 短片", project_path=tmp_path / "proj.acproj")
    dlg = make_dialog(state)
    assert dlg.windowTitle() == "輸出影片" and dlg.isModal()
    assert dlg.page == PAGE_SETTINGS and not dlg.running and dlg.runner is None
    assert dlg.btn_start.text() == "開始輸出" and dlg.btn_start.isEnabled() and dlg.btn_cancel.isEnabled()
    assert dlg.path_edit.text() == str(tmp_path / "我的 短片.mp4")          # next to the project, named after the sequence
    texts = widget_texts(dlg.pages.widget(PAGE_SETTINGS))
    assert "輸出序列：我的 短片" in texts
    assert any("320×180（16:9）· 30 fps · 長度 00:00:03:00" in t for t in texts)
    assert dlg.preset.currentData() == "youtube_1080p"                      # landscape -> YouTube
    assert dlg.range_all.isChecked() and not dlg.range_marked.isEnabled()
    assert "還沒設定入點／出點" in dlg.range_marked.text()
    assert dlg.encoder.currentData() == "libx264"
    assert dlg.chk_loud.isChecked()


def test_portrait_sequences_default_to_the_shorts_preset(qapp, make_dialog, tmp_path):
    dlg = make_dialog(fake_state(tmp_path, size=(1080, 1920))[0])
    assert dlg.preset.currentData() == "shorts_1080x1920"
    assert "1080×1920" in dlg.preset_info.text()


def test_the_preview_preset_is_not_an_export_format(qapp, make_dialog, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    keys = [dlg.preset.itemData(i) for i in range(dlg.preset.count())]
    assert keys == [k for k in PRESETS if k != "preview_540p"]
    assert [dlg.preset.itemText(i) for i in range(dlg.preset.count())] == [PRESETS[k].label for k in keys]


@pytest.mark.parametrize("seq_size,preset,expect", [
    ((3840, 2160), "youtube_1080p", (1920, 1080)), ((3840, 2160), "shorts_1080x1920", (1920, 1080)),
    ((3840, 2160), "source_quality", (3840, 2160)), ((3840, 2160), "draft_fast", (1280, 720)),
    ((1920, 1080), "draft_fast", (1280, 720)), ((1080, 1920), "youtube_1080p", (1080, 1920)),
    ((1080, 1920), "draft_fast", (720, 1280)), ((2160, 3840), "shorts_1080x1920", (1080, 1920)),
    ((640, 360), "youtube_1080p", (640, 360)), ((640, 360), "source_quality", (640, 360))])
def test_switching_the_preset_shows_the_size_that_will_be_written(qapp, make_dialog, tmp_path, seq_size, preset, expect):
    dlg = make_dialog(fake_state(tmp_path, size=seq_size)[0])
    pick(dlg.preset, preset)
    assert f"→ 輸出 {expect[0]}×{expect[1]}" in dlg.preset_info.text()
    assert PRESETS[preset].description in dlg.preset_info.text()
    assert output_size(*seq_size, PRESETS[preset]) == expect                # the label agrees with the render engine
    assert dlg.build_settings().preset == preset


def test_every_preset_label_matches_the_render_engine(qapp, make_dialog, tmp_path):
    dlg = make_dialog(fake_state(tmp_path, size=(2560, 1440))[0])
    for i in range(dlg.preset.count()):
        dlg.preset.setCurrentIndex(i)
        key = dlg.preset.currentData()
        w, h = output_size(2560, 1440, PRESETS[key])
        assert f"輸出 {w}×{h}" in dlg.preset_info.text()
        assert dlg.chk_loud.isChecked() == (PRESETS[key].loudness is not None)


def test_loudness_checkbox_follows_the_preset_and_respects_the_person(qapp, make_dialog, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    assert dlg.chk_loud.isChecked() and "-14 LUFS" in dlg.chk_loud.text() and "各平台音量一致" in dlg.chk_loud.text()
    s = dlg.build_settings()
    assert s.normalize is True and s.loudness is None                       # the preset's own target
    pick(dlg.preset, "draft_fast")
    assert not dlg.chk_loud.isChecked() and "原本不調整音量" in dlg.chk_loud.text()
    s = dlg.build_settings()
    assert s.normalize is False
    dlg.chk_loud.setChecked(True)                                            # the person wants it anyway
    s = dlg.build_settings()
    assert s.normalize is True and s.loudness == ed.DEFAULT_LUFS
    pick(dlg.preset, "youtube_1080p")                                        # a preset change re-applies its default
    assert dlg.chk_loud.isChecked() and dlg.build_settings().loudness is None
    dlg.chk_loud.setChecked(False)
    dlg.preset.setCurrentIndex(dlg.preset.currentIndex())                    # same preset again: the choice is kept
    dlg._on_preset_changed()
    assert not dlg.chk_loud.isChecked() and dlg.build_settings().normalize is False


def test_build_settings_mirrors_every_widget(qapp, make_dialog, assets):
    state = build_state(assets, cues=[(0.2, 1.0, "你好"), (1.2, 2.5, "Second")], rng=(0.5, 2.0))
    dlg = make_dialog(state)
    base = ed.RenderSettings(preset="youtube_1080p", encoder="libx264", burn_captions=True, write_srt=False,
                             frame_range=None, normalize=True)
    assert dlg.build_settings() == base
    pick(dlg.preset, "draft_fast")
    pick(dlg.encoder, "auto")
    dlg.chk_burn.setChecked(False)
    dlg.chk_srt.setChecked(True)
    dlg.chk_loud.setChecked(True)
    click(dlg.range_marked)
    assert dlg.range_marked.isChecked() and not dlg.range_all.isChecked()    # the two radios exclude each other
    assert dlg.build_settings() == ed.RenderSettings(preset="draft_fast", encoder="auto", burn_captions=False, write_srt=True,
                                                     frame_range=(0.5, 2.0), normalize=True, loudness=ed.DEFAULT_LUFS)
    click(dlg.range_all)
    assert dlg.build_settings().frame_range is None


def test_caption_options_need_real_text_cues(qapp, make_dialog, assets):
    dlg = make_dialog(build_state(assets))                                   # no cues at all
    assert not dlg.chk_burn.isEnabled() and not dlg.chk_srt.isEnabled() and not dlg.chk_burn.isChecked()
    assert dlg.chk_burn.toolTip() == "這個序列沒有字幕"
    dlg.chk_burn.setChecked(True)                                            # even forced on, nothing is requested
    dlg.chk_srt.setChecked(True)
    s = dlg.build_settings()
    assert s.burn_captions is False and s.write_srt is False
    blank = make_dialog(build_state(assets, cues=[(0.0, 1.0, "   "), (1.0, 2.0, "")]))   # blank cues are no captions
    assert not blank.chk_burn.isEnabled()
    real = make_dialog(build_state(assets, cues=[(0.0, 1.0, "字幕")]))
    assert real.chk_burn.isEnabled() and real.chk_srt.isEnabled() and real.chk_burn.isChecked()
    assert not real.chk_srt.isChecked()                                      # the .srt copy is opt-in


def test_range_option_shows_the_marked_times(qapp, make_dialog, tmp_path):
    state, _ = fake_state(tmp_path, rng=(0.5, 2.0))
    dlg = make_dialog(state)
    assert dlg.range_marked.isEnabled() and not dlg.range_marked.isChecked() and dlg.range_all.isChecked()
    assert "00:00:00:15 – 00:00:02:00" in dlg.range_marked.text() and "共 00:00:01:15" in dlg.range_marked.text()
    assert dlg.build_settings().frame_range is None                          # not exported unless chosen
    state.set_range(None, None)                                               # the dialog keeps what was marked when it opened
    assert dlg._range == (0.5, 2.0)


@pytest.mark.parametrize("hw,expect", [("libx264", "沒有可用的硬體編碼器"), ("h264_nvenc", "NVIDIA NVENC（h264_nvenc）可用"),
                                       ("h264_qsv", "Intel Quick Sync（QSV）（h264_qsv）可用"),
                                       ("h264_amf", "AMD AMF（h264_amf）可用")])
def test_hardware_encoder_detection_is_explained(qapp, make_dialog, tmp_path, monkeypatch, hw, expect):
    monkeypatch.setattr(ed, "detect_hw_encoder", lambda: hw)
    dlg = make_dialog(fake_state(tmp_path)[0], show=False)                  # (no event loop yet: the result is still queued)
    assert dlg.hw_info.text() == "正在偵測硬體編碼器…"                       # detection runs in the background
    pump(qapp, lambda: dlg.hw_info.text() != "正在偵測硬體編碼器…", 20, "hardware detection")
    assert expect in dlg.hw_info.text()
    if hw == "libx264":
        assert "選「自動」也會使用 CPU" in dlg.hw_info.text()
    assert dlg.encoder.itemData(0) == "libx264" and dlg.encoder.currentData() == "libx264"   # CPU is the safe default
    assert "相容性最好" in dlg.encoder.itemText(0) and "失敗會改用 CPU" in dlg.encoder.itemText(1)
    pick(dlg.encoder, "auto")
    assert dlg.build_settings().encoder == "auto"


def test_a_failing_hardware_detection_only_changes_the_hint(qapp, make_dialog, tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("driver exploded")
    monkeypatch.setattr(ed, "detect_hw_encoder", boom)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pump(qapp, lambda: dlg.hw_info.text() != "正在偵測硬體編碼器…", 20, "hardware detection")
    assert "無法偵測硬體編碼器" in dlg.hw_info.text() and "CPU" in dlg.hw_info.text()
    assert dlg.btn_start.isEnabled() and dlg.encoder.currentData() == "libx264"


def test_browse_fills_in_the_chosen_path(qapp, make_dialog, modals, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    before = dlg.path_edit.text()
    modals.save_result = str(tmp_path / "選的 檔案")
    click(dlg.btn_browse)
    assert dlg.path_edit.text() == str(tmp_path / "選的 檔案.mp4")           # the extension is added
    caption, start, flt = modals.saves[-1]
    assert caption == "輸出影片" and start == before and "*.mp4" in flt
    modals.save_result = str(tmp_path / "Upper.MP4")
    click(dlg.btn_browse)
    assert dlg.path_edit.text() == str(tmp_path / "Upper.MP4")               # an existing .mp4 (any case) is kept
    modals.save_result = ""                                                   # the person cancelled the file dialog
    click(dlg.btn_browse)
    assert dlg.path_edit.text() == str(tmp_path / "Upper.MP4")
    dlg.path_edit.setText("")
    click(dlg.btn_browse)
    assert modals.saves[-1][1] == ed.default_output_dir(dlg.state.project_path)   # empty box: it starts in the usual folder


def test_output_path_is_absolute_mp4_and_forgives_quotes(qapp, make_dialog, tmp_path, monkeypatch):
    state, _ = fake_state(tmp_path)
    state.project.path = str(tmp_path / "proj.acproj")
    dlg = make_dialog(state)
    dlg.path_edit.setText(f'  "{tmp_path / "x y"}"  ')                       # pasted from Explorer's "Copy as path"
    assert dlg.output_path() == str(tmp_path / "x y.mp4")
    dlg.path_edit.setText(str(tmp_path / "Clip.MP4"))
    assert dlg.output_path() == str(tmp_path / "Clip.MP4")
    dlg.path_edit.setText(str(tmp_path / "a" / ".." / "norm"))
    assert dlg.output_path() == str(tmp_path / "norm.mp4")
    for blank in ("", "   ", '""'):
        dlg.path_edit.setText(blank)
        assert dlg.output_path() == ""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    dlg.path_edit.setText("~/home clip")
    assert dlg.output_path() == str(tmp_path / "home clip.mp4")


def test_a_bare_file_name_goes_to_the_usual_folder_not_the_working_directory(qapp, make_dialog, tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                                              # e.g. the folder the app was started from
    state, _ = fake_state(tmp_path)
    state.project.path = str(tmp_path / "proj.acproj")
    dlg = make_dialog(state)
    dlg.path_edit.setText("短片")
    assert dlg.output_path() == str(tmp_path / "短片.mp4")
    dlg.path_edit.setText(os.path.join("sub", "短片.mp4"))
    assert dlg.output_path() == str(tmp_path / "sub" / "短片.mp4")


# ================================================================================================ preflight
def refused(dlg, modals) -> str:
    """start_export() must refuse politely: nothing starts, one warning explains why.  Returns the explanation."""
    n = len(modals.informs)
    assert dlg.start_export() is False
    assert len(modals.informs) == n + 1, modals.informs
    title, text, warning = modals.informs[-1]
    assert title == "無法輸出" and warning is True
    assert dlg.runner is None and not dlg.running and dlg.page == PAGE_SETTINGS and dlg.btn_start.isEnabled()
    assert text == dlg.preflight()
    return text


def test_preflight_refuses_an_empty_sequence(qapp, make_dialog, modals, tmp_path):
    state, _ = fake_state(tmp_path, with_clip=False)
    dlg = make_dialog(state)
    text = refused(dlg, modals)
    assert "還沒有任何片段" in text and "自動剪輯" in text


def test_preflight_names_the_missing_media_in_plain_words(qapp, make_dialog, modals, tmp_path):
    state, _ = fake_state(tmp_path, exists=False)
    dlg = make_dialog(state)
    text = refused(dlg, modals)
    assert "有 1 個素材找不到檔案" in text and "．clip.mp4" in text and "重新連結" in text
    assert "沒辦法輸出" in text


def test_preflight_lists_at_most_eight_missing_files(qapp, make_dialog, modals, tmp_path):
    proj, _ = project_with_clips(tmp_path, [(f"a{i:02d}", False, f"gone{i:02d}.mp4", {}) for i in range(10)])
    dlg = make_dialog(AppState(proj))
    text = refused(dlg, modals)
    assert "有 10 個素材找不到檔案" in text and "gone07.mp4" in text and "gone08.mp4" not in text and "…" in text


def test_preflight_only_cares_about_media_the_sequence_reads(qapp, make_dialog, tmp_path):
    state, a = fake_state(tmp_path)
    state.project.assets["z9"] = mk_asset("z9", tmp_path / "unused-and-gone.mp4")      # in the bin, not on the timeline
    dlg = make_dialog(state)
    assert dlg.preflight() is None
    seq = state.sequence()
    state.project.assets["a2"] = mk_asset("a2", tmp_path / "gone.mp4", "gone.mp4")
    clip = ops.place_clip(state.project, seq, "a2", seq.video_tracks[0].id, start=10.0, src_in=0.0, src_out=2.0)
    assert "gone.mp4" in dlg.preflight()                                              # an enabled clip needs its file
    clip.enabled = False
    assert dlg.preflight() is None                                                    # a disabled clip's file is not needed
    clip.enabled = True
    seq.video_tracks[0].muted = True                                                  # ... nor is a hidden track's
    assert dlg.preflight() is None


def test_preflight_wants_an_output_file(qapp, make_dialog, modals, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    dlg.path_edit.setText("   ")
    assert refused(dlg, modals) == "請先選擇輸出檔案。"


def test_preflight_refuses_a_folder_as_output(qapp, make_dialog, modals, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path)                                                        # an existing folder, no file name
    assert "資料夾" in refused(dlg, modals)
    assert not (tmp_path.parent / (tmp_path.name + ".mp4")).exists()                  # nothing was written beside it
    folder = tmp_path / "looks-like.mp4"
    folder.mkdir()
    pick_output(dlg, folder)                                                          # a folder that is named like a video
    assert "資料夾" in refused(dlg, modals)


@pytest.mark.parametrize("spell", ["plain", "quoted", "dots", "no-extension"] + (["upper"] if os.name == "nt" else []))
def test_preflight_never_lets_the_output_overwrite_a_source_file(qapp, make_dialog, modals, tmp_path, spell):
    state, a = fake_state(tmp_path)
    src = Path(a.path)
    dlg = make_dialog(state)
    typed = {"plain": str(src), "quoted": f'"{src}"', "dots": str(src.parent / "." / "x" / ".." / src.name),
             "no-extension": str(src)[:-4], "upper": str(src)[:-4] + ".MP4"}[spell]
    pick_output(dlg, typed)
    assert "不能和素材檔相同" in refused(dlg, modals)
    assert src.read_bytes() == b"x"                                                   # the source is untouched


def test_preflight_protects_unused_bin_assets_too(qapp, make_dialog, modals, tmp_path):
    state, _ = fake_state(tmp_path)
    other = touch(tmp_path / "bin" / "other.mp4", b"precious")
    state.project.assets["z9"] = mk_asset("z9", other)
    dlg = make_dialog(state)
    pick_output(dlg, other)
    assert "不能和素材檔相同" in refused(dlg, modals)
    assert other.read_bytes() == b"precious"


@pytest.mark.parametrize("name", ["a:b.mp4", "who?.mp4", 'say "hi".mp4', "a|b.mp4", "star*.mp4", "<tag>.mp4"])
def test_preflight_refuses_file_names_windows_cannot_create(qapp, make_dialog, modals, tmp_path, name):
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / name)
    text = refused(dlg, modals)
    assert "檔名" in text and "不能" in text


def test_preflight_refuses_a_location_under_a_plain_file(qapp, make_dialog, modals, tmp_path):
    blocker = touch(tmp_path / "blocker")
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, blocker / "sub" / "out.mp4")                                     # a "folder" that is really a file
    text = refused(dlg, modals)
    assert "無法寫入" in text and "out.mp4" in text and "不是資料夾" in text


def test_preflight_refuses_a_folder_that_cannot_be_written(qapp, make_dialog, modals, tmp_path, monkeypatch):
    seen = []

    def deny(path):
        seen.append(path)
        raise PermissionError(errno.EACCES, "Permission denied", path)

    monkeypatch.setattr(ed, "probe_writable", deny)
    dlg = make_dialog(fake_state(tmp_path)[0])
    target = pick_output(dlg, tmp_path / "locked" / "out.mp4")
    text = refused(dlg, modals)
    assert "無法寫入" in text and "沒有寫入權限" in text and "out.mp4" in text and "Permission denied" not in text
    assert seen[-1] == target
    assert not (tmp_path / "locked").exists()                                         # nothing was created while checking


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores file permissions")
def test_preflight_refuses_a_read_only_folder_for_real(qapp, make_dialog, modals, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        dlg = make_dialog(fake_state(tmp_path)[0])
        pick_output(dlg, ro / "out.mp4")
        assert "沒有寫入權限" in refused(dlg, modals)
    finally:
        ro.chmod(0o755)


def test_a_good_output_path_passes_without_touching_the_disk(qapp, make_dialog, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / "輸出 資料夾" / "新 影片.mp4")                         # the folder does not exist yet
    before = sorted(p.name for p in tmp_path.rglob("*"))
    assert dlg.preflight() is None and dlg.preflight() is None
    assert sorted(p.name for p in tmp_path.rglob("*")) == before                      # no folder, no probe file left over
    existing = touch(tmp_path / "已有.mp4", b"keep me")
    stamp = existing.stat().st_mtime_ns
    pick_output(dlg, existing)
    assert dlg.preflight() is None                                                    # overwriting is asked about later
    assert existing.read_bytes() == b"keep me" and existing.stat().st_mtime_ns == stamp


def test_structural_problems_are_confirmed_before_exporting(qapp, make_dialog, modals, tmp_path):
    state, _ = fake_state(tmp_path)
    next(c for _, c in state.sequence().all_clips()).src_out = 99.0                   # beyond the 5 s the file has
    dlg = make_dialog(state)
    assert dlg.start_export() is False                                                # (modals answer "no")
    title, text, yes, no = modals.asks[-1]
    assert title == "序列有問題" and "檢查到 1 個結構問題" in text and "超出素材長度" in text and "仍要輸出嗎" in text
    assert (yes, no) == ("仍要輸出", "取消")
    assert dlg.runner is None and not dlg.running and dlg.page == PAGE_SETTINGS and not modals.informs


def test_an_existing_output_file_is_only_replaced_after_confirmation(qapp, make_dialog, modals, tmp_path, monkeypatch):
    fake = FakeRender(monkeypatch)
    fake_qa(monkeypatch)
    out = touch(tmp_path / "out" / "影片.mp4", b"OLD CONTENT")
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, out)
    assert dlg.start_export() is False                                                # the person says 取消
    title, text, yes, no = modals.asks[-1]
    assert title == "檔案已存在" and "影片.mp4" in text and (yes, no) == ("覆蓋", "取消")
    assert out.read_bytes() == b"OLD CONTENT" and dlg.runner is None and not fake.calls
    modals.yes("檔案已存在")
    export_and_wait(qapp, dlg)
    assert out.read_bytes() == b"fake-mp4" and len(fake.calls) == 1                   # replaced only now


def test_a_second_start_while_busy_is_ignored(qapp, make_dialog, modals, tmp_path, monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / "o.mp4")
    assert dlg.start_export() is True
    runner = dlg.runner
    assert dlg.running and dlg.page == PAGE_PROGRESS
    assert dlg.start_export() is False and dlg.runner is runner                       # no second render
    gate.set()
    finish(qapp, dlg)
    assert len(fake.calls) == 1 and dlg.page == PAGE_RESULT


# ================================================================================================ helpers for real exports
def watch_bar(dlg) -> list[int]:
    """Every value the progress bar takes (recorded in the GUI thread from the moment this is called)."""
    values: list[int] = []
    dlg.bar.valueChanged.connect(values.append)
    return values


def watch_progress(dlg) -> list[tuple[float, str]]:
    """Every (fraction, message) the dialog is handed by the renderer; call BEFORE ``start_export()``."""
    seen: list[tuple[float, str]] = []
    original = dlg._on_progress

    def spy(fraction, message):
        seen.append((fraction, message))
        original(fraction, message)

    dlg._on_progress = spy
    return seen


def spy_qa(monkeypatch) -> list[dict]:
    """The keyword arguments ``technical_qa`` is called with (what the QA pass expects of the file)."""
    calls: list[dict] = []
    real = ed.technical_qa
    monkeypatch.setattr(ed, "technical_qa", lambda path, **kw: calls.append(kw) or real(path, **kw))
    return calls


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        m = re.fullmatch(r"(\d+):(\d\d):(\d\d),(\d{3}) --> (\d+):(\d\d):(\d\d),(\d{3})", lines[1])
        assert m, block
        g = [int(x) for x in m.groups()]
        out.append((round(g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, 3),
                    round(g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, 3), "\n".join(lines[2:])))
    return out


CUES = [(0.2, 1.0, "第一句"), (1.2, 2.5, "Second line"), (2.6, 2.9, "最後一句")]


# ================================================================================================ real exports (ffmpeg)
@needs_ffmpeg
def test_a_full_export_matches_the_sequence(qapp, make_dialog, assets, tmp_path, monkeypatch):
    state = build_state(assets, project_path=tmp_path / "p.acproj")            # a (1.5 s) + b (1.5 s) = 3 s, 320x180 @ 30 fps
    qa_kwargs = spy_qa(monkeypatch)
    dlg = make_dialog(state)
    out = Path(pick_output(dlg, tmp_path / "輸出 資料夾" / "我的 影片.mp4"))
    bar, exported, seen = watch_bar(dlg), [], watch_progress(dlg)
    dlg.exported.connect(exported.append)

    assert dlg.start_export() is True                                           # --- the progress page is up at once
    assert dlg.running and dlg.page == PAGE_PROGRESS and dlg.runner.running
    assert dlg.progress_title.text() == "正在輸出…" and dlg.stage.text() == "準備中…"
    assert PRESETS["youtube_1080p"].label in dlg.progress_target.text() and str(out) in dlg.progress_target.text()
    assert dlg.btn_stop.isEnabled() and dlg.btn_stop.text() == "取消輸出" and (dlg.bar.minimum(), dlg.bar.maximum()) == (0, 100)
    assert dlg.timer.isActive()
    finish(qapp, dlg)

    assert dlg.page == PAGE_RESULT and not dlg.running and not dlg.runner.running and not dlg.timer.isActive()
    moves = [v for v in bar if v >= 0]                                          # --- the bar moved, forwards only, to 100 %
    assert len(moves) >= 3 and moves == sorted(moves) and moves[-1] == 100 and any(0 < v < 100 for v in moves), bar
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and seen[-1] == (1.0, "完成") and len(set(fractions)) >= 4
    messages = [m for _, m in seen]
    assert any(m.startswith("渲染片段") for m in messages) and "量測響度…" in messages and "封裝成品…" in messages

    res = dlg.result                                                            # --- the file is what the sequence says
    assert res is not None and res.path == str(out) and exported == [str(out)] and dlg.result_title.text() == "輸出完成"
    assert out.is_file() and sorted(p.name for p in out.parent.iterdir()) == [out.name]      # no .partial / .srt left over
    f = facts(out)
    assert f["size"] == (320, 180) == (res.width, res.height)
    assert f["fps"] == pytest.approx(30.0) and f["audio"] and (f["vcodec"], f["acodec"]) == ("h264", "aac")
    assert f["duration"] == pytest.approx(3.0, abs=0.1) and res.duration == pytest.approx(3.0)
    assert abs(f["frames"] - 90) <= 1 and res.frames == 90
    assert len(qa_kwargs) == 1                                                  # --- the QA pass expected the same things
    assert qa_kwargs[0]["expect_size"] == (320, 180) and qa_kwargs[0]["expect_fps"] == 30.0
    assert qa_kwargs[0]["expect_duration"] == pytest.approx(3.0)
    assert dlg.report is not None and dlg.report.status == "pass", dlg.report.summary()
    info = dlg.result_info.text()
    for part in (str(out), "長度 00:00:03:00", "320×180", "30 fps", "編碼器 libx264", "快取命中"):
        assert part in info
    assert dlg.qa_status.text() == ed.QA_HEADLINE["pass"][0]
    assert QSettings("AutoCutStudio", "AutoCutStudio").value(ed.LAST_DIR_KEY) == str(out.parent)   # remembered for next time
    assert not state.dirty and not state.history.can_undo and "ai_runs" not in state.project.meta   # a hand-made sequence: untouched


@needs_ffmpeg
def test_exporting_the_marked_range_gives_exactly_that_range(qapp, make_dialog, assets, tmp_path, monkeypatch):
    state = build_state(assets, rng=(0.5, 2.0), project_path=tmp_path / "p.acproj")      # 3 s sequence, range 0.5 s .. 2.0 s
    qa_kwargs = spy_qa(monkeypatch)
    dlg = make_dialog(state)
    full = Path(pick_output(dlg, tmp_path / "full.mp4"))
    export_and_wait(qapp, dlg)
    assert facts(full)["duration"] == pytest.approx(3.0, abs=0.1)
    click(dlg.btn_again)                                                         # 「再輸出一次…」: the dialog is reusable
    assert dlg.page == PAGE_SETTINGS
    click(dlg.range_marked)
    part = Path(pick_output(dlg, tmp_path / "part.mp4"))
    export_and_wait(qapp, dlg)

    f = facts(part)
    assert f["duration"] == pytest.approx(1.5, abs=0.1) and abs(f["frames"] - 45) <= 1 and f["size"] == (320, 180)
    assert dlg.result.duration == pytest.approx(1.5) and dlg.result.frames == 45
    assert "長度 00:00:01:15" in dlg.result_info.text()
    assert qa_kwargs[-1]["expect_duration"] == pytest.approx(1.5)                # QA expects the range, not the whole sequence
    assert dlg.report.status == "pass", dlg.report.summary()
    # the picture is the right part of the sequence: first frame = full export at 0.5 s, last frame = full export at 1.9 s
    first, last = frame_at(part, 0.0), frame_at(part, 1.4)
    assert mean_diff(first, frame_at(full, 0.5)) < 6 and mean_diff(first, frame_at(full, 0.0)) > 10
    assert mean_diff(last, frame_at(full, 1.9)) < 6 and mean_diff(last, frame_at(full, 1.4)) > 10


@needs_ffmpeg
def test_the_srt_copy_is_written_next_to_the_video(qapp, make_dialog, assets, tmp_path):
    state = build_state(assets, cues=CUES, project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    dlg.chk_burn.setChecked(False)
    dlg.chk_srt.setChecked(True)
    out = Path(pick_output(dlg, tmp_path / "輸出 資料夾" / "帶 字幕.mp4"))
    export_and_wait(qapp, dlg)
    srt = out.with_suffix(".srt")
    assert sorted(p.name for p in out.parent.iterdir()) == ["帶 字幕.mp4", "帶 字幕.srt"]
    raw = srt.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and raw.decode("utf-8")           # UTF-8 text (Chinese survives)
    assert parse_srt(raw.decode("utf-8")) == [(0.2, 1.0, "第一句"), (1.2, 2.5, "Second line"), (2.6, 2.9, "最後一句")]
    assert dlg.result.warnings == [] and dlg.report.status == "pass"
    click(dlg.btn_again)                                                         # unticked again: no .srt is written
    dlg.chk_srt.setChecked(False)
    other = Path(pick_output(dlg, tmp_path / "輸出 資料夾" / "沒字幕.mp4"))
    export_and_wait(qapp, dlg)
    assert other.is_file() and not other.with_suffix(".srt").exists()


@needs_ffmpeg
def test_the_srt_copy_follows_the_exported_range(qapp, make_dialog, assets, tmp_path):
    state = build_state(assets, cues=CUES, rng=(0.5, 2.0), project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    click(dlg.range_marked)
    dlg.chk_burn.setChecked(False)
    dlg.chk_srt.setChecked(True)
    out = Path(pick_output(dlg, tmp_path / "range.mp4"))
    export_and_wait(qapp, dlg)
    cues = parse_srt(out.with_suffix(".srt").read_text(encoding="utf-8"))
    assert cues[0] == (0.0, 0.5, "第一句")                                        # starts before the range: cut at 0, ends 0.5 s in
    assert cues[1][0] == pytest.approx(0.7) and cues[1][2] == "Second line"       # 1.2 s on the timeline = 0.7 s in the file


@needs_ffmpeg
def test_the_srt_of_a_range_has_only_cues_that_fit_the_exported_length(qapp, make_dialog, assets, tmp_path):
    state = build_state(assets, cues=CUES, rng=(0.5, 2.0), project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    click(dlg.range_marked)
    dlg.chk_burn.setChecked(False)
    dlg.chk_srt.setChecked(True)
    out = Path(pick_output(dlg, tmp_path / "range.mp4"))
    export_and_wait(qapp, dlg)
    length = facts(out)["duration"]
    cues = parse_srt(out.with_suffix(".srt").read_text(encoding="utf-8"))
    assert all(start < length for start, _, _ in cues), cues                    # "最後一句" (2.6 s) is not in a 1.5 s video
    assert all(end <= length + 1 / 30 for _, end, _ in cues), cues


@needs_ffmpeg
def test_burned_in_captions_really_appear_in_the_picture(qapp, make_dialog, assets, tmp_path):
    style = {"size": 0.2, "bold": True, "box": True, "box_opacity": 0.9}       # big text on a dark box: unmistakable at 320x180
    state = build_state(assets, cues=[(0.4, 1.4, "字幕 Hello")], style=style, project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    assert dlg.chk_burn.isChecked()
    dlg.chk_burn.setChecked(False)
    plain = Path(pick_output(dlg, tmp_path / "plain.mp4"))
    export_and_wait(qapp, dlg)
    click(dlg.btn_again)
    dlg.chk_burn.setChecked(True)
    burned = Path(pick_output(dlg, tmp_path / "burned.mp4"))
    export_and_wait(qapp, dlg)
    assert dlg.report.status == "pass" and dlg.result.warnings == []

    a, b = frame_at(plain, 0.9), frame_at(burned, 0.9)                           # during the cue
    h = a.shape[0]
    assert mean_diff(a[int(0.55 * h):], b[int(0.55 * h):]) > 10                  # the caption sits in the lower part ...
    assert mean_diff(a[:int(0.4 * h)], b[:int(0.4 * h)]) < 1                     # ... and the rest of the picture is untouched
    assert mean_diff(frame_at(plain, 2.5), frame_at(burned, 2.5)) < 1            # after the cue: identical frames


@needs_ffmpeg
def test_chinese_and_space_paths_work_end_to_end(qapp, make_dialog, assets, tmp_path):
    folder = tmp_path / "專案 資料夾"
    folder.mkdir()
    state = build_state(assets, names=("zh", "a"), seq_name="我的 短片:第1集", cues=[(0.2, 1.0, "字幕 Hello")],
                        style={"size": 0.2, "box": True}, project_path=folder / "我的 專案.acproj")
    assert "素材 資料夾" in assets["zh"].path and "測試 影片" in assets["zh"].path     # the SOURCE has Chinese + spaces too
    dlg = make_dialog(state)
    assert dlg.path_edit.text() == str(folder / "我的 短片_第1集.mp4")             # default: project folder, ":" made safe
    dlg.chk_srt.setChecked(True)                                                  # burn is on by default
    export_and_wait(qapp, dlg)
    default = folder / "我的 短片_第1集.mp4"
    assert sorted(p.name for p in folder.iterdir()) == ["我的 短片_第1集.mp4", "我的 短片_第1集.srt"]
    assert facts(default)["duration"] == pytest.approx(3.0, abs=0.1) and dlg.report.status == "pass"
    assert parse_srt(default.with_suffix(".srt").read_text(encoding="utf-8")) == [(0.2, 1.0, "字幕 Hello")]

    click(dlg.btn_again)
    nested = Path(pick_output(dlg, tmp_path / "輸出 資料夾" / "二〇二六 年" / "最終版 (v2).mp4"))   # folders that do not exist yet
    export_and_wait(qapp, dlg)
    assert nested.is_file() and facts(nested)["audio"] and sorted(p.name for p in nested.parent.iterdir()) == [
        "最終版 (v2).mp4", "最終版 (v2).srt"]
    assert dlg.result.path == str(nested) and str(nested) in dlg.result_info.text()


@needs_ffmpeg
def test_a_portrait_sequence_exports_portrait(qapp, make_dialog, assets, tmp_path):
    state = build_state(assets, names=("portrait",), seconds=2.0, size=(180, 320), project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    assert dlg.preset.currentData() == "shorts_1080x1920"
    out = Path(pick_output(dlg, tmp_path / "直式.mp4"))
    export_and_wait(qapp, dlg)
    f = facts(out)
    assert f["size"] == (180, 320) and f["duration"] == pytest.approx(2.0, abs=0.1) and f["audio"]
    assert dlg.report.status == "pass" and "180×320" in dlg.result_info.text()


@needs_ffmpeg
def test_a_sequence_without_sound_exports_with_a_silent_track(qapp, make_dialog, assets, tmp_path):
    """Loudness normalisation of pure silence makes ffmpeg's loudnorm fail ("Value -inf for parameter 'measured_I'"), so the
    dialog does not offer it when nothing in the sequence makes noise."""
    state = build_state(assets, names=("silent",), seconds=2.0, project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    assert dlg.preset.currentData() == "youtube_1080p" and not dlg.chk_loud.isEnabled() and not dlg.chk_loud.isChecked()
    assert dlg.chk_loud.toolTip() == "這個序列沒有聲音，不需要調整音量"
    assert dlg.build_settings().normalize is False
    pick(dlg.preset, "shorts_1080x1920")                                          # switching presets does not turn it on
    assert not dlg.chk_loud.isChecked()
    out = Path(pick_output(dlg, tmp_path / "靜音.mp4"))
    export_and_wait(qapp, dlg)
    assert dlg.result_title.text() == "輸出完成", dlg.result_info.text()
    f = facts(out)
    assert f["audio"] and f["duration"] == pytest.approx(2.0, abs=0.1) and f["size"] == (320, 180)
    assert dlg.report.status in ("pass", "warn")                                 # a silent file may only get a warning


@needs_ffmpeg
def test_exporting_over_an_existing_file_replaces_it_without_leftovers(qapp, make_dialog, modals, assets, tmp_path):
    state = build_state(assets, project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    out = Path(pick_output(dlg, tmp_path / "again.mp4"))
    pick(dlg.preset, "draft_fast")
    export_and_wait(qapp, dlg)
    first = out.read_bytes()
    click(dlg.btn_again)
    pick(dlg.preset, "youtube_1080p")                                             # different settings -> different bytes
    modals.yes("檔案已存在")
    export_and_wait(qapp, dlg)
    assert modals.titles() == ["檔案已存在"]
    assert out.read_bytes() != first and facts(out)["duration"] == pytest.approx(3.0, abs=0.1)
    assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == ["again.mp4"]


@needs_ffmpeg
def test_auto_encoder_without_hardware_uses_the_cpu_quietly(qapp, make_dialog, assets, tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "detect_hw_encoder", lambda: "libx264")
    dlg = make_dialog(build_state(assets, project_path=tmp_path / "p.acproj"))
    pick(dlg.encoder, "auto")
    export_and_wait(qapp, dlg)
    assert dlg.result.encoder == "libx264" and dlg.result.warnings == [] and dlg.result_warnings.text() == ""
    assert "編碼器 libx264" in dlg.result_info.text()


@needs_ffmpeg
def test_auto_encoder_says_when_ffmpeg_lacks_the_hardware_encoder(qapp, make_dialog, assets, tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "detect_hw_encoder", lambda: "h264_nope")     # an encoder this ffmpeg does not have
    dlg = make_dialog(build_state(assets, project_path=tmp_path / "p.acproj"))
    pick(dlg.encoder, "auto")
    export_and_wait(qapp, dlg)
    assert dlg.result.encoder == "libx264"
    assert "提醒：這台電腦的 ffmpeg 沒有 h264_nope，改用 libx264" in dlg.result_warnings.text()
    assert dlg.result_title.text() == "輸出完成" and dlg.report.status == "pass"


@needs_ffmpeg
def test_auto_encoder_falls_back_to_the_cpu_when_the_hardware_encoder_fails(qapp, make_dialog, assets, tmp_path, monkeypatch):
    if "h264_nvenc" not in runner_mod.ffmpeg_info().encoders:
        pytest.skip("this ffmpeg has no h264_nvenc to fail with")
    monkeypatch.setattr(runner_mod, "detect_hw_encoder", lambda: "h264_nvenc")    # listed by ffmpeg, but no GPU here
    dlg = make_dialog(build_state(assets, project_path=tmp_path / "p.acproj"))
    pick(dlg.encoder, "auto")
    export_and_wait(qapp, dlg)
    assert dlg.result is not None and dlg.result.encoder == "libx264", dlg.result_info.text()
    assert "h264_nvenc 編碼失敗，已改用 libx264" in dlg.result_warnings.text()


@needs_ffmpeg
def test_a_media_file_ffmpeg_cannot_read_is_a_failure_not_a_cancel(qapp, make_dialog, modals, tmp_path):
    """The renderer sets the cancel event itself when a segment fails; the task runner would then report 「已取消」."""
    state, a = fake_state(tmp_path)                                               # the "video" is 1 byte of garbage
    dlg = make_dialog(state)
    statuses: list[str] = []
    state.status.connect(statuses.append)
    pick_output(dlg, tmp_path / "corrupt.mp4")
    export_and_wait(qapp, dlg)
    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出失敗"
    assert "ffmpeg 失敗" in dlg.result_info.text()
    assert "已取消輸出" not in statuses and not dlg.running and not dlg.runner.running
    assert not (tmp_path / "corrupt.mp4").exists() and not list(tmp_path.glob("*.partial*"))
    assert dlg.btn_start.isEnabled()


# ================================================================================================ cancel / closing
@needs_ffmpeg
def test_cancelling_mid_render_stops_cleanly_and_the_dialog_is_usable_again(qapp, make_dialog, modals, assets, tmp_path):
    state = build_state(assets, names=("big",), seconds=12.0, size=(1280, 720), rng=(0.0, 1.0),
                        project_path=tmp_path / "p.acproj")
    statuses: list[str] = []
    state.status.connect(statuses.append)
    dlg = make_dialog(state)
    pick(dlg.preset, "source_quality")                                           # slow encode: plenty of time to cancel
    out = Path(pick_output(dlg, tmp_path / "輸出 資料夾" / "取消 測試.mp4"))
    seen = watch_progress(dlg)
    assert dlg.start_export()
    pump(qapp, lambda: any(f > 0 for f, _ in seen), 60, "ffmpeg to report progress")   # really in the middle of rendering
    runner = dlg.runner
    assert dlg.running and runner.running and dlg.page == PAGE_PROGRESS
    assert has_ffmpeg_children()

    click(dlg.btn_stop)                                                          # the person presses 「取消輸出」
    assert not dlg.btn_stop.isEnabled() and dlg.stage.text() == "正在取消…"      # feedback at once, before ffmpeg has stopped
    pump(qapp, lambda: not dlg.running, 30, "the cancel to finish")
    qapp.processEvents()

    assert dlg.page == PAGE_SETTINGS and dlg.btn_start.isEnabled() and not dlg.timer.isActive()
    assert statuses[-1] == "已取消輸出" and modals.informs == [] and dlg.result is None and dlg.report is None
    assert not out.exists() and not [p for p in tmp_path.rglob("*") if "partial" in p.name.lower()]   # no half-written file
    assert not list(cache_dir("render", "segments").glob("*.partial*"))
    assert runner.wait(5) and not runner.running                                  # --- no zombies
    assert not has_ffmpeg_children()
    assert not [t for t in threading.enumerate() if t.name.startswith("ThreadPoolExecutor")]
    assert not state.dirty and not state.history.can_undo

    click(dlg.range_marked)                                                      # --- and it can export again (1 s this time)
    pick(dlg.preset, "draft_fast")
    export_and_wait(qapp, dlg, 120)
    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出完成" and out.is_file()
    assert facts(out)["duration"] == pytest.approx(1.0, abs=0.1) and not list(out.parent.glob("*.partial*"))


def test_cancel_export_without_a_running_render_does_nothing(qapp, make_dialog, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    dlg.cancel_export()
    assert dlg.page == PAGE_SETTINGS and dlg.runner is None and dlg.stage.text() == "準備中…" and dlg.btn_stop.isEnabled()


def test_a_cancelled_export_keeps_the_file_that_was_already_there(qapp, make_dialog, modals, tmp_path, monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    old = touch(tmp_path / "out" / "舊檔.mp4", b"OLD CONTENT")
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, old)
    modals.yes("檔案已存在")
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 20, "the render to start")
    click(dlg.btn_stop)
    pump(qapp, lambda: not dlg.running, 20, "the cancel")
    assert dlg.page == PAGE_SETTINGS and old.read_bytes() == b"OLD CONTENT" and dlg.result is None


def test_a_failing_render_is_a_failure_even_when_the_renderer_sets_the_cancel_event(qapp, make_dialog, modals, tmp_path,
                                                                                    monkeypatch):
    """render_sequence sets the event it is given when a segment fails; that is NOT the person pressing 取消."""
    FakeRender(monkeypatch, error=FFmpegError("ffmpeg 失敗（代碼 1）：Invalid data"), sets_cancel=True)
    fake_qa(monkeypatch)
    state, _ = fake_state(tmp_path)
    statuses: list[str] = []
    state.status.connect(statuses.append)
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path / "out.mp4")
    export_and_wait(qapp, dlg)
    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出失敗" and "ffmpeg 失敗（代碼 1）" in dlg.result_info.text()
    assert "已取消輸出" not in statuses


def test_the_technical_check_cannot_be_cancelled(qapp, make_dialog, tmp_path, monkeypatch):
    FakeRender(monkeypatch)
    gate = threading.Event()
    fake_qa(monkeypatch, gate=gate)
    dlg = make_dialog(fake_state(tmp_path, ai=True)[0])
    pick_output(dlg, tmp_path / "out.mp4")
    assert dlg.start_export()
    pump(qapp, lambda: dlg.result is not None, 20, "the render to finish")
    assert dlg.page == PAGE_PROGRESS and dlg.running                               # the technical check is running
    assert dlg.progress_title.text() == "正在做技術檢查…" and (dlg.bar.minimum(), dlg.bar.maximum()) == (0, 0)
    assert not dlg.btn_stop.isEnabled() and dlg.btn_stop.text() == "技術檢查進行中…"
    dlg.cancel_export()                                                            # nothing to cancel any more
    assert dlg.stage.text() != "正在取消…" and dlg.running
    gate.set()
    finish(qapp, dlg)
    assert dlg.page == PAGE_RESULT and dlg.report is not None and dlg.result_title.text() == "輸出完成"


def test_closing_while_rendering_asks_and_can_carry_on(qapp, make_dialog, modals, tmp_path, monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / "out.mp4")
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 20, "the render to start")
    dlg.reject()                                                                   # the person answers 「繼續輸出」
    assert modals.asks[-1] == ("取消輸出", "正在輸出中，要取消並關閉嗎？", "取消輸出", "繼續輸出")
    assert dlg.isVisible() and dlg.running and not dlg._closing and dlg.btn_stop.isEnabled()
    gate.set()
    finish(qapp, dlg)
    assert dlg.page == PAGE_RESULT and dlg.isVisible() and (tmp_path / "out.mp4").exists()


def test_closing_while_rendering_can_cancel_and_close(qapp, make_dialog, modals, tmp_path, monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    rejected = []
    dlg.rejected.connect(lambda: rejected.append(True))
    pick_output(dlg, tmp_path / "out.mp4")
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 20, "the render to start")
    modals.yes("取消輸出")
    dlg.reject()                                                                   # 「取消輸出」: cancel, then close
    assert dlg._closing and dlg.stage.text() == "正在取消…"
    pump(qapp, lambda: not dlg.isVisible(), 20, "the dialog to close")
    assert rejected == [True] and not dlg.running and not (tmp_path / "out.mp4").exists()
    assert dlg.runner.wait(5) and not dlg.runner.running


@pytest.mark.parametrize("how", ["escape", "window-close"])
def test_escape_and_the_close_button_ask_as_well(qapp, make_dialog, modals, tmp_path, monkeypatch, how):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / "out.mp4")
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 20, "the render to start")
    if how == "escape":
        QTest.keyClick(dlg, Qt.Key_Escape)
    else:
        dlg.close()
    assert modals.titles() == ["取消輸出"] and dlg.isVisible() and dlg.running      # asked, answered 「繼續輸出」
    gate.set()
    finish(qapp, dlg)


def test_closing_during_the_technical_check_does_not_wait_for_it(qapp, make_dialog, modals, tmp_path, monkeypatch):
    FakeRender(monkeypatch)
    gate = threading.Event()
    fake_qa(monkeypatch, gate=gate)
    state, _ = fake_state(tmp_path, ai=True)
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path / "out.mp4")
    assert dlg.start_export()
    pump(qapp, lambda: dlg.result is not None, 20, "the render to finish")
    dlg.reject()                                                                   # the file exists: nothing to confirm
    assert modals.asks == [] and not dlg.isVisible() and dlg._detached
    qa_runner = dlg.runner
    gate.set()
    wait_for_idle(qapp, [qa_runner], 20)                                           # the abandoned check ends quietly
    assert not dlg.running and (tmp_path / "out.mp4").exists()
    assert state.run_for_sequence()["technical_qa"] == "not_run"                   # an abandoned check records nothing


def test_closing_an_idle_dialog_just_closes_it(qapp, make_dialog, modals, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    click(dlg.btn_cancel)                                                          # 「取消」 on the settings page
    assert not dlg.isVisible() and modals.asks == [] and dlg.runner is None


# ================================================================================================ technical QA
def quick_export(app, make_dialog, tmp_path, monkeypatch, *, ai: bool = False, report=None, name: str = "o.mp4",
                 **render_kw) -> tuple[ExportDialog, AppState]:
    """A complete export that needs no ffmpeg: fake render -> fake QA -> result page."""
    FakeRender(monkeypatch, **render_kw)
    fake_qa(monkeypatch, report)
    state, _ = fake_state(tmp_path, ai=ai)
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path / name)
    export_and_wait(app, dlg)
    return dlg, state


def verdict_buttons(dlg) -> tuple:
    return dlg.review.btn_ok, dlg.review.btn_fix, dlg.review.btn_no


def visible_defaults(dlg) -> list[str]:
    return [b.text() for b in dlg.findChildren(QPushButton) if b.isDefault() and b.isVisible()]


QA_CASES = [
    ("pass", [("可讀取", "pass", "320x180、3.00 秒、h264"), ("長度", "pass", "3.000 秒（預期 3.000 ± 0.10）")], "技術檢查通過"),
    ("warn", [("可讀取", "pass", "ok"), ("黑畫面", "warn", "出現黑畫面：0.0–0.5s")], "技術檢查有警告"),
    ("fail", [("可讀取", "pass", "ok"), ("尺寸", "fail", "640x360（預期 320x180）")], "技術檢查未通過"),
    ("not_run", [("可讀取", "not_run", "略過"), ("完整解碼", "not_run", "略過")], "技術檢查沒有執行"),
    ("not_run", [("可讀取", "pass", "ok"), ("響度", "not_run", "沒有音訊")], "只做了一部分"),
]
QA_IDS = ["pass", "warn", "fail", "nothing-ran", "partly-ran"]


@pytest.mark.parametrize("status, checks, phrase", QA_CASES, ids=QA_IDS)
def test_the_check_of_an_auto_edit_run_is_recorded_as_machine_evidence_and_the_review_stays_open(
        qapp, make_dialog, tmp_path, monkeypatch, review_spy, status, checks, phrase):
    report = qa_report("x.mp4", *checks)
    assert report.status == status
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True, report=report)
    run = state.run_for_sequence()

    assert run["technical_qa"] == status and run["technical_qa_summary"] == report.summary()      # machine evidence ...
    assert run["human_review"] == "pending" and "human_review_at" not in run and run["outcomes"] == []   # ... and nothing more
    assert list(review_spy) == []
    assert dlg.page == PAGE_RESULT and dlg.report.status == status and phrase in dlg.qa_status.text()
    rows = qa_rows(dlg)                                                         # one row per check + the open review
    assert rows[:-1] == [(ed.STATUS_ICON[c.status], c.name, c.detail) for c in report.checks]
    assert rows[-1] == ("–", "人工審片", AI_REVIEW_PENDING)
    assert dlg.review.verdict_label.text() == AI_REVIEW_PENDING and dlg.review.qa_label.text() == QA_TEXT[status]
    assert all(b.isEnabled() for b in verdict_buttons(dlg))
    text = result_text(dlg)
    assert "技術檢查" in text and not [c for c in FORBIDDEN_CLAIMS if c in text], text


@pytest.mark.parametrize("preset, expect", [("youtube_1080p", (1920, 1080)), ("shorts_1080x1920", None)])
def test_the_check_expects_the_size_of_the_chosen_preset_not_the_size_of_the_sequence(qapp, make_dialog, tmp_path, monkeypatch,
                                                                                      preset, expect):
    FakeRender(monkeypatch)
    qa_calls = fake_qa(monkeypatch)
    state, _ = fake_state(tmp_path, size=(3840, 2160))                          # a 4K sequence
    dlg = make_dialog(state)
    pick(dlg.preset, preset)
    pick_output(dlg, tmp_path / "o.mp4")
    export_and_wait(qapp, dlg)
    wanted = expect or output_size(3840, 2160, PRESETS[preset])
    assert wanted != (3840, 2160)                                               # (the preset really does change the size)
    assert len(qa_calls) == 1 and qa_calls[0][1] == wanted and qa_calls[0][2] == 30.0


def test_a_clean_technical_check_still_says_it_is_not_a_verdict_on_taste(qapp, make_dialog, tmp_path, monkeypatch):
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    assert dlg.qa_group.title() == "技術檢查（機器檢查，不是審片）" and dlg.review_group.title() == "人工審片（由你決定）"
    assert dlg.qa_status.text().startswith("技術檢查通過") and "不代表剪得好看" in dlg.qa_status.text()
    assert "請你自己看過成品" in dlg.qa_status.text()
    assert dlg.qa_status.text() == ed.QA_HEADLINE["pass"][0]


def test_a_hand_made_sequence_shows_the_report_but_nothing_is_written_to_ai_runs(qapp, make_dialog, tmp_path, monkeypatch,
                                                                                   review_spy):
    report = qa_report("x.mp4", ("可讀取", "pass", "ok"), ("黑畫面", "warn", "出現黑畫面"))
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=False, report=report)
    assert "ai_runs" not in state.project.meta and state.run_for_sequence() is None
    assert not state.dirty and not state.history.can_undo                       # the project did not change at all
    assert dlg.report.status == "warn" and "技術檢查有警告" in dlg.qa_status.text()
    rows = qa_rows(dlg)
    assert rows[:-1] == [("✔", "可讀取", "ok"), ("▲", "黑畫面", "出現黑畫面")] and rows[-1] == ("–", "人工審片", AI_REVIEW_PENDING)
    assert dlg.review.qa_label.text() == "這個序列不是自動剪輯產生的，沒有審片紀錄。" and dlg.review.verdict_label.text() == ""
    assert not any(b.isEnabled() for b in (*verdict_buttons(dlg), dlg.review.notes))
    for b in verdict_buttons(dlg):                                              # a click on a disabled button changes nothing
        QTest.mouseClick(b, Qt.LeftButton)
    dlg.review._click("approved")                                               # nor does a stray call: there is no record
    assert "ai_runs" not in state.project.meta and list(review_spy) == [] and not state.dirty


@pytest.mark.parametrize("button, verdict, row", [("btn_ok", "approved", "pass"), ("btn_fix", "needs_changes", "warn"),
                                                  ("btn_no", "rejected", "fail")])
def test_only_a_click_on_one_of_the_three_buttons_records_a_verdict(qapp, make_dialog, tmp_path, monkeypatch, review_spy,
                                                                    button, verdict, row):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    run = state.run_for_sequence()
    headline = dlg.qa_status.text()
    assert run["human_review"] == "pending" and list(review_spy) == []         # the export and the check decided nothing
    dlg.review.notes.setText("  結尾太快  ")
    assert run["human_review"] == "pending" and list(review_spy) == []         # typing a note decides nothing either
    review_spy.expected += 1
    QTest.mouseClick(getattr(dlg.review, button), Qt.LeftButton)

    run = state.run_for_sequence()
    assert list(review_spy) == [(run["run_id"], verdict, "結尾太快")]
    assert run["human_review"] == verdict and run["human_review_notes"] == "結尾太快" and run["human_review_at"]
    assert [o["verdict"] for o in run["outcomes"]] == [verdict]
    assert dlg.review.verdict_label.text() == VERDICT_TEXT[verdict]
    assert qa_rows(dlg)[-1] == (ed.STATUS_ICON[row], *ed.VERDICT_ROW[verdict][1:])        # the table follows the person
    assert "你已標記為" in qa_rows(dlg)[-1][2] and qa_rows(dlg)[-1][1] == "人工審片"
    assert dlg.qa_status.text() == headline and run["technical_qa"] == "pass"   # the machine's own part did not move
    assert dlg.page == PAGE_RESULT and dlg.isVisible()                          # and the dialog stays open


def test_a_person_can_change_their_verdict_and_every_decision_is_kept(qapp, make_dialog, tmp_path, monkeypatch, review_spy):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    review_spy.expected += 2
    click(dlg.review.btn_fix)
    click(dlg.review.btn_ok)
    run = state.run_for_sequence()
    assert run["human_review"] == "approved" and [o["verdict"] for o in run["outcomes"]] == ["needs_changes", "approved"]
    assert dlg.review.verdict_label.text() == VERDICT_TEXT["approved"]
    assert qa_rows(dlg)[-1][2] == ed.VERDICT_ROW["approved"][2]


def test_exporting_again_never_moves_a_verdict_the_person_gave(qapp, make_dialog, modals, tmp_path, monkeypatch, review_spy):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    review_spy.expected += 1
    click(dlg.review.btn_fix)
    before = copy.deepcopy(state.run_for_sequence())
    click(dlg.btn_again)
    pick_output(dlg, tmp_path / "second.mp4")
    export_and_wait(qapp, dlg)
    run = state.run_for_sequence()
    assert (run["human_review"], run["human_review_at"], run["outcomes"]) == \
           ("needs_changes", before["human_review_at"], before["outcomes"])     # only the person's earlier click is there
    assert len(review_spy) == 1 and run["technical_qa"] == "pass"
    assert qa_rows(dlg)[-1][2] == ed.VERDICT_ROW["needs_changes"][2]


def test_a_technical_check_that_cannot_run_is_said_so_and_records_nothing(qapp, make_dialog, tmp_path, monkeypatch, review_spy):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True, report=RuntimeError("ffprobe 掛了"))
    run = state.run_for_sequence()
    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出完成" and dlg.report is None   # the file exists
    assert not dlg.running and dlg.runner.wait(5)
    status = dlg.qa_status.text()
    assert status.startswith("技術檢查沒辦法執行：") and "ffprobe 掛了" in status and "RuntimeError" in status
    assert not dlg.btn_copy.isEnabled()
    assert qa_rows(dlg) == [("–", "人工審片", AI_REVIEW_PENDING)]               # no invented check results
    assert run["technical_qa"] == "not_run" and "technical_qa_summary" not in run and run["human_review"] == "pending"
    assert list(review_spy) == []
    assert not [c for c in FORBIDDEN_CLAIMS if c in result_text(dlg)]


def test_an_edit_made_while_the_file_is_being_made_turns_its_check_stale(qapp, make_dialog, tmp_path, monkeypatch, review_spy):
    """The check describes the file that was rendered.  Someone who changes the timeline while the export is still
    running must not end up with a 「通過」 recorded for a picture they have already changed."""
    gate = threading.Event()
    FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    state, _ = fake_state(tmp_path, ai=True)
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path / "o.mp4")
    assert dlg.start_export()
    pump(qapp, lambda: dlg.stage.text().startswith("渲染片段"), 10, "the render to report progress")
    state.op("加標記", "add_marker", 1.0, "m")                                   # the timeline moves on mid-export
    gate.set()
    finish(qapp, dlg)
    run = state.run_for_sequence()
    assert dlg.page == PAGE_RESULT and run["technical_qa"] == "pass"             # the file itself checked out ...
    assert planmod.qa_state(run, state.sequence())[0] == planmod.STALE            # ... but not the sequence as it is now
    assert "又改過" in dlg.result_warnings.text()
    assert dlg.review.qa_label.text() == QA_TEXT["stale"]
    assert list(review_spy) == []


def test_an_unchanged_sequence_keeps_its_check_and_says_nothing_extra(qapp, make_dialog, tmp_path, monkeypatch):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    assert planmod.qa_state(state.run_for_sequence(), state.sequence())[0] == "pass"
    assert "又改過" not in dlg.result_warnings.text() and dlg.review.qa_label.text() == QA_TEXT["pass"]


def test_a_review_given_before_an_edit_is_shown_as_expired_and_needs_a_new_click(qapp, make_dialog, tmp_path, monkeypatch,
                                                                                  review_spy):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    review_spy.expected += 1
    click(dlg.review.btn_ok)
    run = state.run_for_sequence()
    assert planmod.review_state(run, state.sequence())[0] == "approved"
    assert qa_rows(dlg)[-1][2] == ed.VERDICT_ROW["approved"][2]
    state.op("加標記", "add_marker", 2.0, "後來加的")                            # an edit after the person approved
    dlg.review.refresh()
    dlg._update_review_row()
    assert planmod.review_state(run, state.sequence())[0] == planmod.STALE
    assert dlg.review.verdict_label.text() == VERDICT_TEXT["stale"]
    assert qa_rows(dlg)[-1][2] == ed.VERDICT_ROW["stale"][2] and qa_rows(dlg)[-1][0] == "▲"
    assert len(review_spy) == 1                                                  # nothing re-approved it behind their back
    review_spy.expected += 1
    click(dlg.review.btn_fix)                                                    # only a new click gives a new verdict
    assert planmod.review_state(run, state.sequence())[0] == "needs_changes" and run["human_review"] == "needs_changes"


def test_the_copy_button_copies_the_machine_report_with_its_unreviewed_line(qapp, make_dialog, tmp_path, monkeypatch):
    report = qa_report("x.mp4", ("可讀取", "pass", "ok"), ("響度", "warn", "-30.0 LUFS"))
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True, report=report)
    QGuiApplication.clipboard().setText("before")
    assert dlg.btn_copy.isEnabled()
    click(dlg.btn_copy)
    copied = QGuiApplication.clipboard().text()
    assert copied == report.summary() and "▲ 響度：-30.0 LUFS" in copied and "尚未審" in copied
    assert not [c for c in FORBIDDEN_CLAIMS if c in copied]


def test_the_review_area_cannot_be_reached_before_the_file_exists(qapp, make_dialog, tmp_path, monkeypatch):
    gate = threading.Event()
    FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path, ai=True)[0])
    for page in (PAGE_SETTINGS, PAGE_PROGRESS):
        if page == PAGE_PROGRESS:
            pick_output(dlg, tmp_path / "o.mp4")
            assert dlg.start_export()
            pump(qapp, lambda: dlg.stage.text().startswith("渲染片段"), 10, "the render to report progress")
        assert dlg.page == page and not any(b.isVisible() for b in verdict_buttons(dlg))
        assert not dlg.review_group.isVisible() and not dlg.qa_group.isVisible()
    gate.set()
    finish(qapp, dlg)
    assert all(b.isVisible() for b in verdict_buttons(dlg))


def test_enter_in_the_file_name_field_starts_the_export_instead_of_opening_the_browser(qapp, make_dialog, modals, tmp_path,
                                                                                         monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    assert visible_defaults(dlg) == ["開始輸出"]                                 # (it used to be 「瀏覽…」)
    pick_output(dlg, tmp_path / "o.mp4")
    dlg.path_edit.setFocus()
    QTest.keyClick(dlg.path_edit, Qt.Key_Return)
    assert modals.saves == [] and dlg.running and dlg.page == PAGE_PROGRESS    # the file browser stayed closed
    assert visible_defaults(dlg) == []                                         # a stray Enter cannot cancel or close anything
    QTest.keyClick(dlg, Qt.Key_Return)                                         # (e.g. a double press of Enter)
    QTest.keyClick(qapp.focusWidget() or dlg, Qt.Key_Return)
    assert dlg.running and dlg.isVisible() and modals.asks == [] and not dlg.runner.cancel_event.is_set()
    assert dlg.btn_stop.isEnabled() and dlg.btn_stop.text() == "取消輸出"
    gate.set()
    finish(qapp, dlg)
    assert len(fake.calls) == 1
    click(dlg.btn_again)                                                        # back on the settings page: Enter works again
    assert visible_defaults(dlg) == ["開始輸出"]


def test_enter_on_the_result_page_neither_closes_the_dialog_nor_decides_a_review(qapp, make_dialog, tmp_path, monkeypatch,
                                                                                  review_spy):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    closed: list[str] = []
    dlg.accepted.connect(lambda: closed.append("accepted"))
    dlg.rejected.connect(lambda: closed.append("rejected"))
    assert visible_defaults(dlg) == []
    dlg.review.notes.setFocus()
    dlg.review.notes.setText("先寫備註")
    QTest.keyClick(dlg.review.notes, Qt.Key_Return)                             # typing a note and pressing Enter ...
    QTest.keyClick(dlg, Qt.Key_Return)
    assert closed == [] and dlg.isVisible() and dlg.page == PAGE_RESULT
    assert state.run_for_sequence()["human_review"] == "pending" and list(review_spy) == []   # ... decides nothing
    assert dlg.review.notes.text() == "先寫備註"


def test_a_verdict_button_can_be_used_from_the_keyboard(qapp, make_dialog, tmp_path, monkeypatch, review_spy):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    dlg.review.btn_no.setFocus()
    qapp.processEvents()
    review_spy.expected += 1
    QTest.keyClick(dlg.review.btn_no, Qt.Key_Space)                             # the person chose this button themselves
    assert state.run_for_sequence()["human_review"] == "rejected" and len(review_spy) == 1


def test_the_review_box_stays_quiet_when_the_dialog_is_already_gone(qapp, modals, tmp_path, monkeypatch):
    errors: list = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    FakeRender(monkeypatch)
    fake_qa(monkeypatch)
    state, _ = fake_state(tmp_path, ai=True)
    dlg = ExportDialog(state)
    dlg.show()
    pick_output(dlg, tmp_path / "o.mp4")
    export_and_wait(qapp, dlg)
    dlg.runner.wait(5)
    shiboken6.delete(dlg)                                                       # the dialog and its review box are destroyed ...
    run_id = state.run_for_sequence()["run_id"]
    state.record_technical_qa(run_id, "warn", "later")                          # ... but the project keeps emitting edits
    state.do("改名", lambda p: ops.rename_sequence(p, p.sequence(state.seq_id), "新名字"))
    state.undo()
    qapp.processEvents()
    assert errors == [] and state.run_for_sequence()["technical_qa"] == "warn"
    QThreadPool.globalInstance().waitForDone(20000)                             # (no make_dialog here: wait for the encoder probe)


def test_the_slot_error_trap_really_catches_errors(qapp, monkeypatch):
    """Guards the previous test: without this a silent hook would make it pass for the wrong reason."""
    errors: list = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    from PySide6.QtCore import QObject, Signal

    class Emitter(QObject):
        ping = Signal()

    e = Emitter()
    e.ping.connect(lambda: 1 / 0)
    e.ping.emit()
    assert len(errors) == 1 and errors[0][0] is ZeroDivisionError


@needs_ffmpeg
def test_a_real_export_of_an_auto_edit_sequence_records_the_real_check(qapp, make_dialog, assets, tmp_path, review_spy):
    state = build_state(assets, ai=True, project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    out = Path(pick_output(dlg, tmp_path / "auto.mp4"))
    export_and_wait(qapp, dlg)
    run = state.run_for_sequence()
    assert out.is_file() and dlg.report.status == "pass", dlg.report.summary()
    assert run["technical_qa"] == "pass" and run["technical_qa_summary"] == dlg.report.summary()
    assert run["human_review"] == "pending" and list(review_spy) == []
    names = [r[1] for r in qa_rows(dlg)]
    assert names[:5] == ["可讀取", "尺寸", "影格率", "長度", "聲音串流"] and names[-1] == "人工審片"
    assert "完整解碼" in names and "響度" in names                                # the deep checks really ran
    assert [r[0] for r in qa_rows(dlg)][:-1] == ["✔"] * (len(names) - 1)
    assert qa_rows(dlg)[-1][2] == AI_REVIEW_PENDING


@needs_ffmpeg
def test_the_real_check_flags_a_file_that_is_not_a_video(qapp, make_dialog, tmp_path, monkeypatch, review_spy):
    FakeRender(monkeypatch, payload=b"this is not an mp4")                       # real QA, junk file
    state, _ = fake_state(tmp_path, ai=True)
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path / "junk.mp4")
    export_and_wait(qapp, dlg)
    run = state.run_for_sequence()
    assert dlg.report.status == "fail" and qa_rows(dlg)[0][:2] == ("✘", "可讀取")
    assert dlg.qa_status.text() == ed.QA_HEADLINE["fail"][0] and "技術檢查未通過" in dlg.qa_status.text()
    assert run["technical_qa"] == "fail" and run["human_review"] == "pending" and list(review_spy) == []


@needs_ffmpeg
def test_the_real_check_compares_the_file_with_the_sequence_not_with_itself(qapp, make_dialog, media, tmp_path, monkeypatch,
                                                                            review_spy):
    kwargs = spy_qa(monkeypatch)
    FakeRender(monkeypatch, payload=media["b"].read_bytes(), duration=3.0)       # a valid 320x180 file ...
    state, _ = fake_state(tmp_path, ai=True)                                     # ... for a 1920x1080 sequence
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path / "small.mp4")
    export_and_wait(qapp, dlg)
    sizes = {r[1]: r for r in qa_rows(dlg)}
    assert kwargs[0]["expect_size"] == (1920, 1080) and kwargs[0]["expect_fps"] == 30.0
    assert sizes["尺寸"][0] == "✘" and "320x180" in sizes["尺寸"][2] and "預期 1920x1080" in sizes["尺寸"][2]
    assert sizes["可讀取"][0] == "✔" and dlg.report.status == "fail"
    assert state.run_for_sequence()["technical_qa"] == "fail" and state.run_for_sequence()["human_review"] == "pending"


# ================================================================================================ failure paths
def encoder_probe_done(app, dlg: ExportDialog) -> None:
    """The dialog probes for a GPU encoder in the background (that is an ffmpeg child too): let it end first."""
    pump(app, lambda: not dlg.hw_info.text().startswith("正在偵測"), 30, "the encoder probe to end")


def assert_failed_cleanly(app, dlg: ExportDialog, out: Path, fragment: str, qa_calls=()) -> None:
    """The failure page is up, says why in plain words, and nothing is left running or half written."""
    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出失敗" and ed.theme.ERROR in dlg.result_title.styleSheet()
    shown = dlg.result_info.text()
    assert fragment in shown, shown
    assert "Traceback" not in shown and "File \"" not in shown                   # a message, never a stack trace
    assert not dlg.running and dlg.result is None and dlg.report is None
    assert dlg.runner.wait(5) and not dlg.runner.running and not dlg.timer.isActive()   # no zombie task, the clock stopped
    assert list(qa_calls) == []                                                 # nothing to check: no file was made
    assert not out.exists() and not list(out.parent.glob("*.partial*"))         # no half-written file either
    assert all(w.isHidden() for w in (dlg.qa_group, dlg.review_group, dlg.btn_folder, dlg.btn_play))
    assert dlg.btn_again.text() == "回到設定" and dlg.btn_again.isEnabled() and dlg.btn_close.isEnabled()
    encoder_probe_done(app, dlg)
    assert not has_ffmpeg_children()                                            # no render process left behind


def assert_usable_again(app, dlg: ExportDialog, out: Path, fake: FakeRender) -> None:
    """Back at the settings, the very same dialog exports successfully."""
    typed = dlg.path_edit.text()
    click(dlg.btn_again)
    assert dlg.page == PAGE_SETTINGS and dlg.btn_start.isEnabled() and dlg.path_edit.text() == typed
    fake.error = None
    export_and_wait(app, dlg)
    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出完成" and ed.theme.OK in dlg.result_title.styleSheet()
    assert out.is_file() and dlg.result is not None and dlg.report is not None
    assert not any(w.isHidden() for w in (dlg.qa_group, dlg.review_group, dlg.btn_folder, dlg.btn_play))
    assert dlg.btn_again.text() == "再輸出一次…"


FAILURES = [
    pytest.param(MissingMediaError(["/素材/clip.mp4", "/素材/b roll.mp4"]),
                 "找不到這些素材檔（可能被移動或刪除）：\n/素材/clip.mp4\n/素材/b roll.mp4", id="missing-media"),
    pytest.param(RenderError("輸出範圍是空的"), "輸出範圍是空的", id="render-error"),
    pytest.param(FFmpegError("ffmpeg 失敗（代碼 1）：Invalid data found when processing input"),
                 "ffmpeg 失敗（代碼 1）：Invalid data found", id="ffmpeg-error"),
    pytest.param(FFmpegMissing("找不到 ffmpeg / ffprobe。請雙擊專案資料夾裡的 Setup-AutoCut.bat 安裝（或單獨執行 scripts\\windows\\Get-Ffmpeg.ps1），也可以把 ffmpeg 的 bin 資料夾加入 PATH。"),
                 "找不到 ffmpeg / ffprobe", id="ffmpeg-missing"),
    pytest.param(PermissionError(errno.EACCES, "Permission denied", "/x/o.mp4"),
                 "無法寫入「o.mp4」：檔案可能正被其他程式（例如播放器）開啟", id="locked-file"),
    pytest.param(OSError(errno.ENOSPC, "No space left on device", "/x/o.mp4"), "無法寫入「o.mp4」：磁碟空間不足。", id="disk-full"),
    pytest.param(OSError(errno.EROFS, "Read-only file system", "/x/o.mp4"), "這個位置是唯讀的", id="read-only"),
    pytest.param(RuntimeError("boom"), "發生未預期的錯誤（RuntimeError）：boom", id="unexpected"),
]


@pytest.mark.parametrize("error, fragment", FAILURES)
def test_a_failing_export_is_explained_in_plain_words_and_the_dialog_is_usable_again(
        qapp, make_dialog, modals, tmp_path, monkeypatch, review_spy, error, fragment):
    fake = FakeRender(monkeypatch, error=error)
    qa_calls = fake_qa(monkeypatch)
    state, _ = fake_state(tmp_path, ai=True)
    seen_status: list[str] = []
    state.status.connect(seen_status.append)
    dlg = make_dialog(state)
    exported: list[str] = []
    dlg.exported.connect(exported.append)
    out = Path(pick_output(dlg, tmp_path / "o.mp4"))
    export_and_wait(qapp, dlg)

    assert_failed_cleanly(qapp, dlg, out, fragment, qa_calls)
    assert exported == [] and not [m for m in seen_status if m.startswith("已輸出")]     # nothing was announced as done
    run = state.run_for_sequence()
    assert run["technical_qa"] == "not_run" and run["human_review"] == "pending" and list(review_spy) == []
    assert modals.informs == [] and modals.asks == []                          # the page says it; no extra pop-up
    assert QSettings("AutoCutStudio", "AutoCutStudio").value(ed.LAST_DIR_KEY) in (None, "")    # a failed export is not remembered
    assert_usable_again(qapp, dlg, out, fake)
    assert exported == [str(out)]


def test_closing_after_a_failure_does_not_ask_anything(qapp, make_dialog, modals, tmp_path, monkeypatch):
    FakeRender(monkeypatch, error=RuntimeError("boom"))
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / "o.mp4")
    export_and_wait(qapp, dlg)
    closed: list[str] = []
    dlg.rejected.connect(lambda: closed.append("rejected"))
    click(dlg.btn_close)                                                        # 「關閉」 accepts
    dlg2 = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg2, tmp_path / "o2.mp4")
    export_and_wait(qapp, dlg2)
    dlg2.rejected.connect(lambda: closed.append("rejected"))
    dlg2.reject()                                                               # Esc / ✕
    assert closed == ["rejected"] and modals.asks == [] and not dlg.isVisible() and not dlg2.isVisible()


def test_missing_ffmpeg_is_reported_in_plain_words_by_the_real_renderer(qapp, make_dialog, modals, tmp_path, monkeypatch):
    def gone():
        raise FFmpegMissing("找不到 ffmpeg / ffprobe。請雙擊專案資料夾裡的 Setup-AutoCut.bat 安裝（或單獨執行 scripts\\windows\\Get-Ffmpeg.ps1），也可以把 ffmpeg 的 bin 資料夾加入 PATH。")

    monkeypatch.setattr(runner_mod, "ffmpeg_info", gone)
    qa_calls = fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    out = Path(pick_output(dlg, tmp_path / "o.mp4"))
    export_and_wait(qapp, dlg)
    assert_failed_cleanly(qapp, dlg, out, "找不到 ffmpeg / ffprobe", qa_calls)
    assert "Setup-AutoCut.bat" in dlg.result_info.text()                        # and what to do about it


@needs_ffmpeg
def test_a_locked_output_file_fails_after_the_render_with_a_plain_message_and_no_leftovers(qapp, make_dialog, modals, assets,
                                                                                            tmp_path, monkeypatch):
    """Windows: a player holds the old file open, so the final move onto it is refused."""
    out = touch(tmp_path / "播放中 的檔案.mp4", b"old video")
    modals.yes("檔案已存在")
    real_replace = os.replace

    def locked(src, dst, *a, **k):
        if os.path.abspath(str(dst)) == str(out):
            raise PermissionError(errno.EACCES, "Permission denied", str(dst))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", locked)
    qa_calls = fake_qa(monkeypatch)
    dlg = make_dialog(build_state(assets, project_path=tmp_path / "p.acproj"))
    pick_output(dlg, out)
    export_and_wait(qapp, dlg)

    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出失敗"
    assert "無法寫入「播放中 的檔案.mp4」" in dlg.result_info.text() and "正被其他程式" in dlg.result_info.text()
    assert out.read_bytes() == b"old video"                                     # the file that was there is untouched
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix != ".acproj") == [out.name]    # no .partial.mp4 left behind
    assert qa_calls == [] and not dlg.running and dlg.runner.wait(5) and not has_ffmpeg_children()


@needs_ffmpeg
def test_a_source_that_vanishes_after_the_check_is_reported_as_missing_media(qapp, make_dialog, modals, tmp_path, monkeypatch):
    state, _ = fake_state(tmp_path, exists=False)
    dlg = make_dialog(state)
    assert ed.missing_media(state.project, state.project.active_sequence()) == ["clip.mp4"]     # (the check would refuse ...)
    monkeypatch.setattr(ed, "missing_media", lambda p, s: [])                  # ... as if the file went away right after it
    qa_calls = fake_qa(monkeypatch)
    out = Path(pick_output(dlg, tmp_path / "gone.mp4"))
    export_and_wait(qapp, dlg)
    assert_failed_cleanly(qapp, dlg, out, "找不到這些素材檔（可能被移動或刪除）：", qa_calls)
    assert "clip.mp4" in dlg.result_info.text()


@needs_ffmpeg
def test_a_sequence_with_every_track_hidden_fails_politely_without_starting_ffmpeg(qapp, make_dialog, modals, tmp_path, monkeypatch):
    state, _ = fake_state(tmp_path)
    seq = state.project.active_sequence()
    for tr in list(seq.tracks()):
        state.do("隱藏軌道", lambda p, t=tr.id: ops.set_track_props(p, p.sequence(seq.id), t, muted=True))
    qa_calls = fake_qa(monkeypatch)
    dlg = make_dialog(state)
    out = Path(pick_output(dlg, tmp_path / "hidden.mp4"))
    export_and_wait(qapp, dlg)
    assert_failed_cleanly(qapp, dlg, out, "時間軸是空的", qa_calls)
    assert modals.asks == [] and modals.informs == []


@needs_ffmpeg
def test_a_range_outside_the_sequence_fails_politely(qapp, make_dialog, tmp_path, monkeypatch):
    state, _ = fake_state(tmp_path, rng=(9.0, 12.0))                            # the clip is 4 s long
    qa_calls = fake_qa(monkeypatch)
    dlg = make_dialog(state)
    click(dlg.range_marked)
    out = Path(pick_output(dlg, tmp_path / "late.mp4"))
    export_and_wait(qapp, dlg)
    assert_failed_cleanly(qapp, dlg, out, "輸出範圍是空的", qa_calls)


@needs_ffmpeg
def test_a_clip_without_its_asset_is_confirmed_and_then_fails_politely(qapp, make_dialog, modals, tmp_path, monkeypatch):
    state, asset = fake_state(tmp_path)
    _, clip = next(state.project.active_sequence().all_clips())
    clip.asset_id = "ghost"                                                     # (hand-edited / damaged project)
    modals.yes("序列有問題")
    qa_calls = fake_qa(monkeypatch)
    dlg = make_dialog(state)
    out = Path(pick_output(dlg, tmp_path / "ghost.mp4"))
    export_and_wait(qapp, dlg)
    assert modals.titles() == ["序列有問題"] and "片段指向不存在的素材" in modals.asks[0][1]
    assert_failed_cleanly(qapp, dlg, out, "片段指向不存在的素材", qa_calls)


@needs_ffmpeg
def test_a_very_short_range_exports_even_with_loudness_normalisation(qapp, make_dialog, assets, tmp_path):
    state = build_state(assets, rng=(1.4, 1.5))                                  # 0.1 s of a 3 s sequence
    dlg = make_dialog(state)
    click(dlg.range_marked)
    assert dlg.chk_loud.isChecked()
    out = Path(pick_output(dlg, tmp_path / "blink.mp4"))
    export_and_wait(qapp, dlg)
    assert dlg.page == PAGE_RESULT and dlg.result_title.text() == "輸出完成", dlg.result_info.text()
    assert out.is_file()


def release_during_question(qapp, dlg, modals, monkeypatch, release, until) -> None:
    """Answer 「取消並關閉」 only after ``release()`` made the render end *while the question is open*."""
    def ask(parent, title, text, yes="確定", no="取消"):
        modals.asks.append((title, text, yes, no))
        release()
        pump(qapp, until, 20, "the render to end while the question is open")
        return True

    monkeypatch.setattr(ed, "ask", ask)


def test_answering_the_close_question_after_the_render_already_failed_still_closes(qapp, make_dialog, modals, tmp_path,
                                                                                      monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate, error=RuntimeError("boom"))
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / "o.mp4")
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 10, "the render to start")
    closed: list[str] = []
    dlg.rejected.connect(lambda: closed.append("rejected"))
    release_during_question(qapp, dlg, modals, monkeypatch, gate.set, lambda: dlg.page == PAGE_RESULT)
    dlg.reject()
    assert modals.titles() == ["取消輸出"] and closed == ["rejected"] and not dlg.isVisible()      # the person said close: closed
    assert not dlg.running and dlg.runner.wait(5)


def test_answering_the_close_question_after_the_render_already_finished_still_closes(qapp, make_dialog, modals, tmp_path,
                                                                                        monkeypatch):
    gate, qa_gate = threading.Event(), threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    qa_calls = fake_qa(monkeypatch, gate=qa_gate)
    dlg = make_dialog(fake_state(tmp_path, ai=True)[0])
    out = Path(pick_output(dlg, tmp_path / "o.mp4"))
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 10, "the render to start")
    closed: list[str] = []
    dlg.rejected.connect(lambda: closed.append("rejected"))
    release_during_question(qapp, dlg, modals, monkeypatch, gate.set, lambda: dlg.result is not None)
    dlg.reject()
    assert closed == ["rejected"] and not dlg.isVisible() and out.is_file()    # closed; the finished file is kept
    qa_gate.set()                                                               # the abandoned technical check ends later ...
    pump(qapp, lambda: not dlg.running, 10, "the abandoned check to end")
    assert dlg.page == PAGE_PROGRESS and dlg.report is None and closed == ["rejected"]    # ... and shows nothing
    assert len(qa_calls) == 1


# ================================================================================================ buttons, clock, small helpers
@pytest.fixture
def desktop(monkeypatch):
    """Stands in for ``QDesktopServices`` so the real ``open_folder`` / ``play_file`` can be run without starting a program."""
    class Fake:
        opened: list[QUrl] = []
        ok = True

        @staticmethod
        def openUrl(url):
            Fake.opened.append(url)
            return Fake.ok

    monkeypatch.setattr(ed, "QDesktopServices", Fake)
    return Fake


def test_open_folder_opens_the_folder_that_holds_the_file(modals, desktop, tmp_path, monkeypatch):
    target = tmp_path / "輸出 資料夾" / "我的 影片.mp4"
    assert modals.real_open_folder(str(target)) is True
    assert [u.toLocalFile() for u in desktop.opened] == [str(target.parent)]     # the folder, not the (maybe gone) file
    monkeypatch.chdir(tmp_path)
    modals.real_open_folder(os.path.join("rel", "x.mp4"))                       # a relative path is made absolute first
    assert desktop.opened[-1].toLocalFile() == str(tmp_path / "rel")
    desktop.ok = False
    assert modals.real_open_folder(str(target)) is False                        # the OS refusing is passed on


def test_play_file_opens_the_file_itself(modals, desktop, tmp_path):
    target = tmp_path / "輸出 資料夾" / "我的 影片.mp4"
    assert modals.real_play_file(str(target)) is True
    assert [u.toLocalFile() for u in desktop.opened] == [str(target)]
    desktop.ok = False
    assert modals.real_play_file(str(target)) is False


def test_the_result_buttons_do_what_they_say(qapp, make_dialog, modals, tmp_path, monkeypatch):
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, name="我的 影片.mp4")
    out = str(tmp_path / "我的 影片.mp4")
    click(dlg.btn_folder)
    click(dlg.btn_play)
    assert modals.folders == [out] and modals.played == [out]
    dlg.result = None                                                           # nothing exported: nothing to open
    click(dlg.btn_folder)
    click(dlg.btn_play)
    assert modals.folders == [out] and modals.played == [out]
    done: list[str] = []
    dlg.accepted.connect(lambda: done.append("accepted"))
    click(dlg.btn_close)
    assert done == ["accepted"] and not dlg.isVisible()


def test_the_elapsed_clock_counts_up_while_rendering(qapp, make_dialog, tmp_path, monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path / "o.mp4")
    assert dlg.start_export()
    assert dlg.elapsed.text() == "已用時間 0:00" and dlg.timer.interval() == 1000 and dlg.timer.isActive()
    pump(qapp, fake.reached.is_set, 10, "the render to start")
    dlg._started_at = time.monotonic() - 75
    dlg._tick()
    assert dlg.elapsed.text() == "已用時間 1:15"
    dlg._started_at = time.monotonic() - 3725
    dlg._tick()
    assert dlg.elapsed.text() == "已用時間 1:02:05"
    gate.set()
    finish(qapp, dlg)
    assert not dlg.timer.isActive()


def test_the_render_works_on_a_frozen_copy_of_the_project(qapp, make_dialog, tmp_path, monkeypatch):
    gate = threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch)
    state, _ = fake_state(tmp_path)
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path / "o.mp4")
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 10, "the render to start")
    state.do("改名", lambda p: ops.rename_sequence(p, p.sequence(dlg.seq_id), "改過的名字"))      # edited while rendering
    rendered = fake.calls[0]["proj"]
    assert rendered is not state.project and rendered.sequence(dlg.seq_id).name == "假序列"
    assert state.project.sequence(dlg.seq_id).name == "改過的名字" and dlg.seq.name == "改過的名字"
    gate.set()
    finish(qapp, dlg)
    assert dlg.page == PAGE_RESULT and fake.calls[0]["seq_id"] == dlg.seq_id


def test_the_dialog_is_modal_and_titled(qapp, make_dialog, tmp_path):
    dlg = make_dialog(fake_state(tmp_path)[0])
    assert dlg.isModal() and dlg.windowTitle() == "輸出影片" and dlg.page == PAGE_SETTINGS
    assert dlg.minimumWidth() == 640 and not dlg.running and dlg.runner is None and dlg.result is None and dlg.report is None


@pytest.mark.parametrize("what, change, expect", [
    ("nothing changed", lambda proj, seq, clip: None, True),
    ("clip muted", lambda proj, seq, clip: setattr(clip, "muted", True), False),
    ("clip volume 0", lambda proj, seq, clip: setattr(clip, "volume", 0.0), False),
    ("clip disabled", lambda proj, seq, clip: setattr(clip, "enabled", False), False),
    ("track hidden", lambda proj, seq, clip: setattr(seq.video_tracks[0], "muted", True), False),
    ("track volume 0", lambda proj, seq, clip: setattr(seq.video_tracks[0], "volume", 0.0), False),
    ("asset without audio", lambda proj, seq, clip: setattr(proj.assets[clip.asset_id], "has_audio", False), False),
    ("asset is a picture", lambda proj, seq, clip: setattr(proj.assets[clip.asset_id], "kind", "image"), False),
    ("asset unknown", lambda proj, seq, clip: setattr(clip, "asset_id", "ghost"), False),
])
def test_has_sound_mirrors_what_the_renderer_would_mix(tmp_path, what, change, expect):
    proj, seq = project_with_clips(tmp_path, [("a1", True, "x.mp4", {})])
    _, clip = next(seq.all_clips())
    assert ed.has_sound(proj, seq) is True
    change(proj, seq, clip)
    assert ed.has_sound(proj, seq) is expect, what


def test_has_sound_needs_only_one_audible_clip(tmp_path):
    proj, seq = project_with_clips(tmp_path, [("a1", True, "x.mp4", {"muted": True}), ("a2", True, "y.mp4", {})])
    assert ed.has_sound(proj, seq) is True
    assert ed.has_sound(Project.new("empty"), Project.new("empty").active_sequence()) is False


def test_the_cancel_event_given_to_the_renderer_never_touches_the_persons_own_flag():
    user = threading.Event()
    mine = ed._RenderCancel(user)
    assert not mine.is_set() and not mine.isSet()
    mine.set()                                                                  # what render_sequence does when a segment fails
    assert mine.is_set() and mine.isSet() and not user.is_set()
    mine.clear()                                                                # ... and before it retries with the CPU encoder
    assert not mine.is_set() and not user.is_set()
    user.set()                                                                  # the person pressed cancel
    assert mine.is_set() and mine.isSet()
    mine.clear()
    assert mine.is_set()                                                        # the renderer cannot undo a cancel


def test_every_user_visible_string_of_the_dialog_is_traditional_chinese_and_never_claims_a_review(qapp, make_dialog, tmp_path,
                                                                                                     monkeypatch):
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    texts = widget_texts(dlg)
    assert texts and not [c for c in FORBIDDEN_CLAIMS for t in texts if c in t]
    simplified = "这个还没东输视频开设检审结错误启动态宽长时间进档级调标准乐击页"          # characters that only exist in Simplified Chinese
    assert not [t for t in texts if any(ch in simplified for ch in t)], [t for t in texts if any(ch in simplified for ch in t)]
    for page in (PAGE_SETTINGS, PAGE_PROGRESS):
        assert widget_texts(dlg.pages.widget(page))


# ================================================================================================ screenshots
def clipped_widgets(root) -> list[str]:
    """Visible labels / buttons / check boxes whose text does not fit the space they were given (cut off or elided)."""
    bad: list[str] = []
    for w in root.findChildren(QLabel):
        if not w.isVisibleTo(root) or not w.text() or w.textFormat() == Qt.RichText:
            continue
        if w.wordWrap():
            if w.heightForWidth(w.width()) > w.height() + 1:
                bad.append(f"label {w.text()[:30]!r}: needs {w.heightForWidth(w.width())}px, has {w.height()}px")
        elif w.fontMetrics().horizontalAdvance(w.text()) > w.width():
            bad.append(f"label {w.text()[:30]!r}: text {w.fontMetrics().horizontalAdvance(w.text())}px wide, label {w.width()}px")
    for cls in (QPushButton, QCheckBox, QRadioButton):
        for w in root.findChildren(cls):
            if w.isVisibleTo(root) and w.sizeHint().width() > w.width():
                bad.append(f"{cls.__name__} {w.text()[:30]!r}: wants {w.sizeHint().width()}px, has {w.width()}px")
    for table in root.findChildren(QTableWidget):
        if not table.isVisibleTo(root):
            continue
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                it = table.item(r, c)
                if it is None:
                    continue
                need = table.fontMetrics().boundingRect(0, 0, max(1, table.columnWidth(c) - 10), 10_000, Qt.TextWordWrap,
                                                        it.text()).height()
                if need > table.verticalHeader().defaultSectionSize():
                    bad.append(f"table row {r} col {c}: {it.text()[:30]!r} needs {need}px, rows are "
                               f"{table.verticalHeader().defaultSectionSize()}px")
    return bad


def shot(widget, name: str, bright_limit: float = 0.25) -> Path:
    """``screenshots/export_<name>.png`` (git-ignored); the picture is sanity-checked to be a dark-theme dialog."""
    try:
        SHOTS.mkdir(exist_ok=True)
    except OSError:
        pytest.skip("screenshots/ is not writable")
    path = SHOTS / f"export_{name}.png"
    pix = widget.grab()
    assert pix.save(str(path)), path
    img = pix.toImage()
    assert img.width() >= 600 and img.height() >= 150, (img.width(), img.height())
    bright = total = 0
    for y in range(0, img.height(), 3):
        for x in range(0, img.width(), 3):
            total += 1
            bright += QColor(img.pixel(x, y)).lightness() > 150
    assert bright / total < bright_limit, f"{name}: {bright / total:.0%} bright pixels -- part of the dialog is not dark-themed"
    return path


REAL_CHECKS = [("可讀取", "pass", "1920x1080、18.40 秒、h264"), ("尺寸", "pass", "1920x1080（預期 1920x1080）"),
               ("影格率", "pass", "30.000（預期 30.000）"), ("長度", "pass", "18.400 秒（預期 18.400 ± 0.10）"),
               ("聲音串流", "pass", "有"), ("完整解碼", "pass", "沒有錯誤"), ("黑畫面", "pass", "沒有意外的黑畫面"),
               ("凍結畫面", "pass", "沒有"), ("長時間靜音", "pass", "沒有"), ("響度", "pass", "-14.2 LUFS"),
               ("真峰值", "pass", "-1.6 dBFS")]
LONG_PATH = ("輸出 資料夾", "第二季 旅行 vlog 精華版 最終 v3（給客戶審閱）.mp4")


def test_screenshot_of_the_settings_page(qapp, make_dialog, tmp_path):
    state, _ = fake_state(tmp_path, cues=CUES, rng=(0.5, 3.0), ai=True)
    dlg = make_dialog(state)
    encoder_probe_done(qapp, dlg)
    pick_output(dlg, tmp_path.joinpath(*LONG_PATH))
    shot(dlg, "settings")
    assert clipped_widgets(dlg) == []


def test_screenshot_of_the_settings_page_when_there_is_nothing_to_choose(qapp, make_dialog, tmp_path):
    state, _ = fake_state(tmp_path)                                             # no captions, no range
    dlg = make_dialog(state)
    encoder_probe_done(qapp, dlg)
    pick_option = dlg.preset
    pick(pick_option, "shorts_1080x1920")
    shot(dlg, "settings_plain")
    assert clipped_widgets(dlg) == []


def test_screenshot_of_the_progress_page(qapp, make_dialog, tmp_path, monkeypatch):
    gate, qa_gate = threading.Event(), threading.Event()
    fake = FakeRender(monkeypatch, gate=gate, steps=5)
    fake_qa(monkeypatch, gate=qa_gate)
    dlg = make_dialog(fake_state(tmp_path)[0])
    encoder_probe_done(qapp, dlg)
    pick_output(dlg, tmp_path.joinpath(*LONG_PATH))
    assert dlg.start_export()
    pump(qapp, lambda: dlg.bar.value() >= 20, 10, "the bar to move")
    dlg._started_at = time.monotonic() - 42
    dlg._tick()
    shot(dlg, "progress")
    assert clipped_widgets(dlg) == []
    gate.set()                                                                   # --- the technical check (indeterminate bar)
    pump(qapp, lambda: dlg.result is not None, 10, "the render to end")
    qapp.processEvents()
    assert dlg.page == PAGE_PROGRESS and dlg.bar.maximum() == 0
    shot(dlg, "progress_check")
    assert clipped_widgets(dlg) == []
    qa_gate.set()
    finish(qapp, dlg)


@needs_ffmpeg
def test_screenshot_of_a_real_result(qapp, make_dialog, assets, tmp_path):
    state = build_state(assets, ai=True, cues=CUES, project_path=tmp_path / "p.acproj")
    dlg = make_dialog(state)
    pick_output(dlg, tmp_path.joinpath(*LONG_PATH))
    export_and_wait(qapp, dlg)
    assert dlg.page == PAGE_RESULT and dlg.report.status == "pass"
    shot(dlg, "result")
    assert clipped_widgets(dlg) == []


def test_screenshot_of_the_result_with_every_kind_of_finding(qapp, make_dialog, tmp_path, monkeypatch):
    checks = [*REAL_CHECKS[:6], ("黑畫面", "warn", "出現黑畫面：0.0–0.5s、17.9–18.4s"), ("凍結畫面", "pass", "沒有"),
              ("長時間靜音", "not_run", "略過"), ("響度", "warn", "-31.0 LUFS"), ("真峰值", "fail", "1.2 dBFS")]
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True, report=qa_report("x.mp4", *checks),
                          name=LONG_PATH[1], warnings=["這份 ffmpeg 不支援燒入字幕（缺少 libass），已改輸出 .srt 檔",
                                                       "h264_nvenc 編碼失敗，已改用 libx264 重新輸出（Cannot load nvcuda.dll）"])
    shot(dlg, "result_findings")
    assert clipped_widgets(dlg) == []


def test_screenshot_of_the_result_after_the_person_decided(qapp, make_dialog, tmp_path, monkeypatch, review_spy):
    dlg, state = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True, report=qa_report("x.mp4", *REAL_CHECKS))
    dlg.review.notes.setText("第 12 秒的跳接太明顯，結尾再留半秒")
    review_spy.expected += 1
    click(dlg.review.btn_fix)
    shot(dlg, "result_reviewed")
    assert clipped_widgets(dlg) == []


def test_screenshot_of_the_result_for_a_hand_made_sequence(qapp, make_dialog, tmp_path, monkeypatch):
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=False, report=qa_report("x.mp4", *REAL_CHECKS))
    shot(dlg, "result_manual")
    assert clipped_widgets(dlg) == []


def test_screenshot_of_the_result_when_the_check_could_not_run(qapp, make_dialog, tmp_path, monkeypatch):
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True,
                          report=FFmpegError("ffprobe 沒有回應（逾時 3600 秒）：請確認檔案沒有被其他程式占用"))
    shot(dlg, "result_check_failed")
    assert clipped_widgets(dlg) == []


def test_screenshot_of_the_failure_page(qapp, make_dialog, tmp_path, monkeypatch):
    message = ("ffmpeg 失敗（代碼 222）：[Parsed_loudnorm_0 @ 0x560211bfbb40] Value -inf for parameter 'measured_I' out of range "
               "[-99 - 0]\nError applying option 'measured_I' to filter 'loudnorm': Numerical result out of range\n"
               "[aost#0:1/aac @ 0x560211bb52c0] Error initializing a simple filtergraph")
    FakeRender(monkeypatch, error=FFmpegError(message))
    fake_qa(monkeypatch)
    dlg = make_dialog(fake_state(tmp_path)[0])
    pick_output(dlg, tmp_path.joinpath(*LONG_PATH))
    export_and_wait(qapp, dlg)
    assert dlg.result_title.text() == "輸出失敗"
    shot(dlg, "failure")
    assert clipped_widgets(dlg) == []


# ================================================================================================ layout guards
SLACK = 24                                                                     # px: a settled layout differs a little from the one measured at the switch


def blank_below(dlg: ExportDialog) -> int:
    """Pixels the dialog is taller than the current page needs at the dialog's own width (0 = fitted)."""
    m = dlg.layout().contentsMargins()
    page = dlg.pages.currentWidget().layout()
    return dlg.height() - (page.totalHeightForWidth(dlg.width() - m.left() - m.right()) + m.top() + m.bottom())


def test_every_page_is_only_as_tall_as_its_content_even_with_a_long_output_path(qapp, make_dialog, tmp_path, monkeypatch):
    gate, qa_gate = threading.Event(), threading.Event()
    fake = FakeRender(monkeypatch, gate=gate)
    fake_qa(monkeypatch, gate=qa_gate)
    dlg = make_dialog(fake_state(tmp_path, ai=True)[0])
    pick_output(dlg, tmp_path.joinpath("很長 的資料夾名稱 " * 3, *LONG_PATH))
    heights = {"settings": dlg.height()}
    assert 0 <= blank_below(dlg) <= SLACK
    assert dlg.start_export()
    pump(qapp, fake.reached.is_set, 10, "the render to start")
    heights["progress"] = dlg.height()
    assert 0 <= blank_below(dlg) <= SLACK, blank_below(dlg)                       # (it used to be 600+ px tall: a hint taken at 190 px width)
    gate.set()
    pump(qapp, lambda: dlg.result is not None, 10, "the render to end")
    qa_gate.set()
    finish(qapp, dlg)
    heights["result"] = dlg.height()
    assert 0 <= blank_below(dlg) <= SLACK
    click(dlg.btn_again)
    assert 0 <= blank_below(dlg) <= SLACK and abs(dlg.height() - heights["settings"]) <= SLACK     # back to the same size
    assert heights["progress"] < heights["settings"] <= heights["result"] + 150


def test_a_long_report_scrolls_instead_of_growing_the_dialog(qapp, make_dialog, tmp_path, monkeypatch):
    checks = [(f"檢查 {i}", "pass", "ok") for i in range(30)]
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True, report=qa_report("x.mp4", *checks))
    avail = dlg.screen().availableGeometry()
    assert dlg.page == PAGE_RESULT and len(qa_rows(dlg)) == 31 and dlg.height() <= avail.height()
    assert dlg.qa_table.height() == 320 and dlg.qa_table.verticalScrollBar().maximum() > 0
    dlg.qa_table.verticalScrollBar().setValue(dlg.qa_table.verticalScrollBar().maximum())
    assert qa_rows(dlg)[-1][1] == "人工審片"                                      # the open review row is always reachable


def test_the_clipping_check_used_by_the_screenshots_really_catches_cut_off_text(qapp):
    from PySide6.QtWidgets import QVBoxLayout, QWidget
    box = QWidget()
    lay = QVBoxLayout(box)
    narrow = QLabel("這是一行比視窗還長很多很多很多很多的說明文字")
    narrow.setFixedWidth(60)
    lay.addWidget(narrow)
    box.show()
    qapp.processEvents()
    try:
        assert any("label" in problem for problem in clipped_widgets(box))
    finally:
        box.close()
        box.deleteLater()


def test_the_review_note_field_keeps_what_was_typed_across_clicks_on_other_buttons(qapp, make_dialog, tmp_path, monkeypatch):
    dlg, _ = quick_export(qapp, make_dialog, tmp_path, monkeypatch, ai=True)
    dlg.review.notes.setText("先寫著")
    click(dlg.btn_copy)
    click(dlg.btn_folder)
    assert dlg.review.notes.text() == "先寫著"


# (next section)
