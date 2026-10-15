"""Auto-edit dialogs (GUI_SPEC §6), offscreen: ``ShortsDialog`` / ``AssembleDialog`` on the REAL engine and tiny ffmpeg media.

What is driven here: the settings form (defaults, remembered settings, validation, speech hint), the consent flow for
the speech model, analysis with progress + cancel, the review page (rows, thumbnails, findings), 「預覽此段」 on a
throw-away project, and 「套用選取的片段」 (subset -> audit again -> one undo step, ``human_review`` stays "pending").
Tests that do not need the engine's numbers reuse ONE real plan (computed once) through a fake ``propose_*``.

Modal dialogs would hang an offscreen run, so every one of them is patched (``modals`` records what was asked).
Screenshots: ``screenshots/autoedit_<name>.png`` (git-ignored).
"""
from __future__ import annotations

import copy
import faulthandler
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt                                   # noqa: E402
from PySide6.QtTest import QTest                                           # noqa: E402
from PySide6.QtWidgets import QDialog, QFileDialog, QLabel, QMessageBox    # noqa: E402

from autocut.analysis import speech as speech_mod                          # noqa: E402
from autocut.autoedit import api                                           # noqa: E402
from autocut.autoedit.api import AutoEditError, Cancelled                  # noqa: E402
from autocut.autoedit.config import AssembleParams, ShortsParams           # noqa: E402
from autocut.core.plan import AuditReceipt, EditPlan, Finding             # noqa: E402
from autocut.core.project import Asset, Project                            # noqa: E402
from autocut.gui import autoedit_dialog as dlg_mod                         # noqa: E402
from autocut.gui import autoedit_forms as forms                            # noqa: E402
from autocut.gui import media_bin                                          # noqa: E402
from autocut.gui.autoedit_dialog import (PAGE_FORM, PAGE_PROGRESS, PAGE_REVIEW, AssembleDialog,   # noqa: E402
                                         ShortsDialog)
from autocut.gui.autoedit_review import build_candidates                    # noqa: E402
from autocut.gui.state import AppState                                     # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
SHOTS = Path(__file__).resolve().parents[1] / "screenshots"
BURST = "sine=frequency=440:sample_rate=48000:duration={d},volume='if(lt(mod(t,2),1),1,0)':eval=frame"
ENC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-g", "15", "-c:a", "aac"]


# ================================================================================================ helpers
def pump(app, cond, timeout: float = 90.0, what: str = "condition") -> None:
    """Process events until ``cond()`` (the background tasks deliver their signals through the event loop)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.004)
    raise TimeoutError(f"timed out waiting for {what}")


def shot(widget, name: str) -> Path:
    SHOTS.mkdir(exist_ok=True)
    path = SHOTS / f"autoedit_{name}.png"
    assert widget.grab().save(str(path)), path
    return path


def click(btn) -> None:
    assert btn.isEnabled(), f"{btn.text()!r} is disabled"
    QTest.mouseClick(btn, Qt.LeftButton)


def pick(combo, value) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    raise AssertionError(f"{value!r} not in combo")


def probe_project(paths) -> Project:
    from autocut.media.probe import probe_asset
    p = Project.new("autoedit-gui")
    for path in paths:
        a = probe_asset(path)
        p.assets[a.id] = a
    return p


def spin_value(w):
    return w.value()


def shown(dlg, w: int = 900, h: int = 640):
    dlg.resize(w, h)
    dlg.show()
    QTest.qWait(30)
    return dlg


def clone_plan(plan_dict: dict) -> EditPlan:
    return EditPlan.from_dict(copy.deepcopy(plan_dict))


def fake_receipt(plan: EditPlan, proj: Project, findings: list[Finding]) -> AuditReceipt:
    ok = not any(f.severity == "error" for f in findings)
    return AuditReceipt(plan_sha256=plan.sha256(), project_revision=proj.revision, project_sha256=proj.content_sha256(),
                        accepted=ok, findings=findings)


def sequences_of(state: AppState) -> list[str]:
    return [s.id for s in state.project.sequences]


# ================================================================================================ fixtures
@pytest.fixture(autouse=True)
def watchdog():
    """A hung offscreen test must die with a traceback instead of blocking the run forever."""
    faulthandler.dump_traceback_later(170, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def close_leftover_dialogs(qapp):
    """Every dialog a test showed must be gone afterwards: a visible modal dialog swallows the keys of later tests."""
    yield
    for w in list(qapp.topLevelWidgets()):
        if isinstance(w, QDialog):
            w.hide()
            w.deleteLater()
    qapp.processEvents()


@pytest.fixture(autouse=True)
def clean_settings(qapp):
    def wipe():
        s = QSettings("AutoCutStudio", "AutoCutStudio")
        s.remove("autoedit")
        s.sync()
    wipe()
    yield
    wipe()


@pytest.fixture(autouse=True)
def modals(monkeypatch):
    """No modal dialog may open (it would hang): record the questions, answer 'no' unless a test says otherwise."""
    log: list = []

    def fake_ask(parent, title, text, yes="確定", no="取消"):
        log.append(("ask", title, text, yes, no))
        return False

    monkeypatch.setattr(media_bin, "ask", fake_ask)
    monkeypatch.setattr(media_bin, "inform", lambda parent, title, text, warning=False: log.append(("inform", title, text)))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: log.append(("QMessageBox.exec", self.text())) or 0)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (log.append(("QFileDialog",)), ("", ""))[1]))
    return log


@pytest.fixture(autouse=True)
def no_speech(monkeypatch):
    """Default: speech recognition is 'not installed' (the CI machine may or may not have it); tests override."""
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (False, speech_mod.INSTALL_HINT))
    monkeypatch.setattr(speech_mod, "model_cached", lambda model_size="small": False)


@pytest.fixture(autouse=True)
def human_review_guard(monkeypatch):
    """The dialogs must never set the human review (only the person's click in the inspector may)."""
    calls: list = []
    monkeypatch.setattr(AppState, "human_review", lambda self, *a, **k: calls.append(a))
    return calls


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> Path:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    d = tmp_path_factory.mktemp("autoedit_gui_media")

    def ff(name: str, *args: str) -> None:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, str(d / name)], check=True)

    cmd = []
    for col in ("red", "green", "blue", "yellow", "purple", "orange"):
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
    ff("music.wav", "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration=5")
    (d / "caps.srt").write_text("1\n00:00:01,000 --> 00:00:03,000\n大家好\n", encoding="utf-8")
    return d


class Env:
    """One probed project (as a dict, so every test gets the SAME asset ids) + the plan the real engine made for it."""

    def __init__(self, proj: Project, kind: str, ids: list[str], plan: EditPlan) -> None:
        self.proj_dict, self.kind, self.ids = proj.to_dict(), kind, ids
        self.plan_dict = plan.to_dict()

    def state(self) -> AppState:
        return AppState(Project.from_dict(copy.deepcopy(self.proj_dict)))

    def plan(self) -> EditPlan:
        return clone_plan(self.plan_dict)


@pytest.fixture(scope="module")
def shorts_env(media) -> Env:
    proj = probe_project([media / "long.mp4"])
    aid = next(iter(proj.assets))
    params = ShortsParams(count=3, target_seconds=15, min_gap_between=2.0, captions="off", use_speech=False)
    plan = api.propose_shorts(proj, aid, params)
    assert len(plan.sequences) >= 2, plan.notes
    return Env(proj, "shorts", [aid], plan)


@pytest.fixture(scope="module")
def assemble_env(media) -> Env:
    proj = probe_project([media / n for n in ("a.mp4", "b.mp4", "c.mp4", "d.mp4")])
    ids = list(proj.assets)
    plan = api.propose_assemble(proj, ids, AssembleParams(target_seconds=20, order="as_given", captions="off"))
    return Env(proj, "assemble", ids, plan)


def serve_plan(monkeypatch, env: Env, delay: float = 0.0, calls: list | None = None) -> None:
    """Make the dialog's ``propose_*`` return the stored real plan (optionally slowly, honouring ``cancel``)."""
    def fake(proj, ids, params=None, progress=None, cancel=None):
        if calls is not None:
            calls.append((ids, params))
        end = time.monotonic() + delay
        while time.monotonic() < end:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if progress:
                progress(1.0 - (end - time.monotonic()) / max(delay, 1e-6), "分析中…")
            time.sleep(0.01)
        return env.plan()

    monkeypatch.setattr(dlg_mod, "propose_shorts" if env.kind == "shorts" else "propose_assemble", fake)


def open_shorts(env: Env, qapp, monkeypatch=None, *, serve: bool = True, **form) -> tuple[ShortsDialog, AppState]:
    state = env.state()
    dlg = ShortsDialog(state, env.ids[0])
    dlg.autoplay_previews = False
    for k, v in form.items():
        w = dlg.form.w[k]
        pick(w, v) if hasattr(w, "currentData") else (w.setChecked(v) if hasattr(w, "isChecked") else w.setValue(v))
    return dlg, state


def analyse(dlg, qapp) -> None:
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_REVIEW, what="the review page")
    pump(qapp, lambda: all(not r.thumb.pixmap().isNull() for r in dlg.rows), what="thumbnails")


# ================================================================================================ form + settings
def test_forms_start_at_dataclass_defaults(qapp, shorts_env):
    dlg, _ = open_shorts(shorts_env, qapp)
    assert dlg.minimumSize().width() == 900 and dlg.minimumSize().height() == 640
    assert dlg.page == PAGE_FORM
    assert dlg.form.params() == ShortsParams()
    assert dlg.form.w["platform"].count() == 4 and "Reels" in dlg.form.w["platform"].itemText(1)
    assert not dlg.form.advanced.expanded()
    assert dlg.form.w["reframe_strength"].isEnabled() and spin_value(dlg.form.w["count"]) == 3
    pick(dlg.form.w["reframe"], "center")
    assert not dlg.form.w["reframe_strength"].isEnabled()                    # irrelevant for a fixed crop
    pick(dlg.form.w["captions"], "file")
    assert dlg.form.w["caption_file"].isEnabled()
    for name, w in dlg.form.w.items():                                       # short explanatory tooltips everywhere
        assert w.toolTip(), name
    assert AssembleDialog(shorts_env.state(), shorts_env.ids).form.params() == AssembleParams()


def test_last_parameters_are_remembered(qapp, shorts_env, monkeypatch):
    def refuse(*a, **k):
        raise AutoEditError("測試：不真的分析")
    monkeypatch.setattr(dlg_mod, "propose_shorts", refuse)
    dlg, _ = open_shorts(shorts_env, qapp)
    w = dlg.form.w
    w["count"].setValue(4)
    w["target_seconds"].setValue(22)
    pick(w["platform"], "tiktok")
    pick(w["reframe"], "motion")
    w["reframe_strength"].setValue(20)
    pick(w["captions"], "off")
    w["tighten"].setChecked(False)
    w["music_gain_db"].setValue(-27)
    dlg.form.advanced.set_expanded(True)
    w["pause_min"].setValue(1.1)
    w["use_speech"].setChecked(False)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_FORM and not dlg.running, what="the failed analysis")
    assert "測試：不真的分析" in dlg.form_note.text() and dlg.form_note.level == "error"
    again = ShortsDialog(shorts_env.state(), shorts_env.ids[0])
    p = again.form.params()
    assert (p.count, p.target_seconds, p.platform, p.reframe, p.captions, p.tighten) == (4, 22, "tiktok", "motion", "off", False)
    assert abs(p.reframe_strength - 0.2) < 1e-9 and p.music_gain_db == -27 and abs(p.pause_min - 1.1) < 1e-9
    assert not p.use_speech and again.form.advanced.expanded()
    # assemble keeps its own settings
    asm = AssembleDialog(shorts_env.state(), shorts_env.ids)
    asm.form.w["target_seconds"].setValue(95)
    pick(asm.form.w["transition"], "cut")
    forms.save_params(forms.KEY_ASSEMBLE, asm.form.params())
    again2 = AssembleDialog(shorts_env.state(), shorts_env.ids)
    assert again2.form.params().target_seconds == 95 and again2.form.params().transition == "cut"
    assert not again2.form.w["crossfade_seconds"].isEnabled()
    forms.settings().setValue(forms.KEY_SHORTS, "{not json")                  # garbage never blocks the dialog
    assert ShortsDialog(shorts_env.state(), shorts_env.ids[0]).form.params() == ShortsParams()


def test_form_validation_keeps_the_dialog_on_the_form(qapp, shorts_env, media, monkeypatch):
    calls: list = []
    serve_plan(monkeypatch, shorts_env, calls=calls)
    dlg, _ = open_shorts(shorts_env, qapp)
    pick(dlg.form.w["captions"], "file")
    assert dlg.start_analysis() is False and "字幕檔" in dlg.form_note.text()
    dlg.form.w["caption_file"].set_path(str(media / "nope.srt"))
    assert dlg.start_analysis() is False and "找不到字幕檔" in dlg.form_note.text()
    pick(dlg.form.w["captions"], "off")
    dlg.form.w["music_path"].set_path(str(media / "nope.wav"))
    assert dlg.start_analysis() is False and "找不到配樂檔" in dlg.form_note.text()
    dlg.form.advanced.set_expanded(True)
    dlg.form.w["min_seconds"].setValue(50)
    dlg.form.w["max_seconds"].setValue(20)
    dlg.form.w["music_path"].set_path("")
    assert dlg.start_analysis() is False and "最短長度" in dlg.form_note.text()
    assert calls == [] and dlg.page == PAGE_FORM and not dlg.running


def test_missing_source_disables_start(qapp):
    p = Project.new("x")
    p.assets["a1"] = Asset(id="a1", path="/nonexistent/gone.mp4", name="gone.mp4", kind="video", duration=30.0,
                           width=1920, height=1080, fps=30.0, has_video=True, has_audio=True)
    dlg = ShortsDialog(AppState(p), "a1")
    assert not dlg.btn_start.isEnabled() and "找不到素材檔案" in dlg.source_note.text()
    assert not ShortsDialog(AppState(p), "nope").btn_start.isEnabled()
    asm = AssembleDialog(AppState(p), ["a1"])
    assert not asm.btn_start.isEnabled() and "至少需要 2 個" in asm.source_note.text()


# ================================================================================================ speech hint + consent
def test_speech_hint_when_not_installed(qapp, shorts_env):
    dlg, _ = open_shorts(shorts_env, qapp)
    shown(dlg)
    pump(qapp, lambda: dlg.speech_status is not None, what="the speech probe")
    note = dlg.form.speech_note
    assert note.isVisible() and note.level == "warn"
    assert "pip install -r requirements-speech.txt" in note.text() and "不會自動產生字幕" in note.text()
    assert note.text().count("pip install") == 1
    pick(dlg.form.w["captions"], "file")                                       # a caption file needs no recognition
    dlg.form.w["use_speech"].setChecked(False)
    assert not note.isVisible()
    pick(dlg.form.w["captions"], "auto")
    assert note.isVisible()
    shot(dlg, "shorts_form_speech_hint")


def test_speech_hint_model_missing_and_ready(qapp, shorts_env, monkeypatch):
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    dlg, _ = open_shorts(shorts_env, qapp)
    pump(qapp, lambda: dlg.speech_status is not None)
    assert dlg.form.speech_note.level == "info" and "不會偷偷" not in dlg.form.speech_note.text()
    assert "484 MB" in dlg.form.speech_note.text() and "先問你" in dlg.form.speech_note.text()
    monkeypatch.setattr(speech_mod, "model_cached", lambda model_size="small": True)
    ready, _ = open_shorts(shorts_env, qapp)
    pump(qapp, lambda: ready.speech_status is not None)
    assert not ready.form.speech_note.text()


def test_consent_asked_before_download_and_refusal_turns_captions_off(qapp, shorts_env, modals, monkeypatch):
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    downloads: list = []
    monkeypatch.setattr(speech_mod, "_load_model", lambda size, allow: downloads.append((size, allow)))
    calls: list = []
    serve_plan(monkeypatch, shorts_env, calls=calls)
    dlg, state = open_shorts(shorts_env, qapp)
    shown(dlg)
    assert dlg.form.params().captions == "auto"
    click(dlg.btn_start)                                                        # refuse (the autouse answer is False)
    pump(qapp, lambda: dlg.page == PAGE_REVIEW, what="analysis without captions")
    asks = [m for m in modals if m[0] == "ask"]
    assert len(asks) == 1 and "484 MB" in asks[0][2] and "不會上傳" in asks[0][2]
    assert downloads == []                                                      # never downloaded silently
    ids, params = calls[0]
    assert params.captions == "off" and params.use_speech is False
    assert any("沒有同意下載語音模型" in n for n in dlg.extra_notes)
    assert dlg.form.params().captions == "auto"                                 # the person's setting is untouched


def test_consent_granted_downloads_in_the_background_then_analyses(qapp, shorts_env, modals, monkeypatch):
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    order: list = []
    threads: list = []

    def load(size, allow):
        threads.append(threading.current_thread() is threading.main_thread())
        order.append(("download", size, allow))
        time.sleep(0.15)

    monkeypatch.setattr(speech_mod, "_load_model", load)
    monkeypatch.setattr(media_bin, "ask", lambda *a, **k: (modals.append(("ask", a[1])), True)[1])
    calls: list = []

    def fake(proj, ids, params=None, progress=None, cancel=None):
        order.append(("propose", params.captions))
        return shorts_env.plan()

    monkeypatch.setattr(dlg_mod, "propose_shorts", fake)
    dlg, _ = open_shorts(shorts_env, qapp)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_PROGRESS)
    pump(qapp, lambda: dlg.page == PAGE_REVIEW, what="download + analysis")
    assert order == [("download", "small", True), ("propose", "auto")]
    assert threads == [False]                                                   # the download never blocks the GUI thread
    assert dlg.extra_notes == []


def test_failed_download_returns_to_the_form(qapp, shorts_env, monkeypatch):
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    monkeypatch.setattr(media_bin, "ask", lambda *a, **k: True)

    def boom(size, allow):
        raise speech_mod.SpeechUnavailable("無法下載或載入語音模型「small」：連線逾時（請確認網路連線與磁碟空間）")

    monkeypatch.setattr(speech_mod, "_load_model", boom)
    calls: list = []
    serve_plan(monkeypatch, shorts_env, calls=calls)
    dlg, _ = open_shorts(shorts_env, qapp)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_FORM and not dlg.running, what="the failed download")
    assert "無法下載語音模型" in dlg.form_note.text() and "連線逾時" in dlg.form_note.text()
    assert "Traceback" not in dlg.form_note.text() and calls == []


def test_cancel_during_download_returns_at_once(qapp, shorts_env, monkeypatch):
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    monkeypatch.setattr(media_bin, "ask", lambda *a, **k: True)
    release = threading.Event()
    monkeypatch.setattr(speech_mod, "_load_model", lambda size, allow: release.wait(20))
    calls: list = []
    serve_plan(monkeypatch, shorts_env, calls=calls)
    dlg, _ = open_shorts(shorts_env, qapp)
    shown(dlg)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_PROGRESS and dlg.progress_title.text().startswith("正在下載"))
    click(dlg.btn_cancel_run)
    assert dlg.page == PAGE_FORM and not dlg.running and "已取消" in dlg.form_note.text()
    release.set()                                                                # the download finishes later: ignored
    QTest.qWait(150)
    assert dlg.page == PAGE_FORM and calls == []


# ================================================================================================ analysis: progress / cancel / errors
def test_cancel_mid_analysis_leaves_the_dialog_usable(qapp, shorts_env, monkeypatch):
    seen = {"cancel": False, "n": 0}

    def slow(proj, ids, params=None, progress=None, cancel=None):
        seen["n"] += 1
        for i in range(500):
            if cancel is not None and cancel.is_set():
                seen["cancel"] = True
                raise Cancelled()
            progress(i / 500, f"分析聲音… {i}")
            time.sleep(0.02)
        return shorts_env.plan()

    monkeypatch.setattr(dlg_mod, "propose_shorts", slow)
    dlg, state = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    rev = state.project.revision
    shown(dlg)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_PROGRESS and dlg.progress_bar.value() >= 5, what="progress")
    assert dlg.running and not dlg.form.isEnabled() and dlg.progress_msg.text().startswith("分析聲音")
    assert not dlg.btn_start.isEnabled()                                         # no second run while one is going
    shot(dlg, "progress")
    click(dlg.btn_cancel_run)
    pump(qapp, lambda: dlg.page == PAGE_FORM and not dlg.running, what="the cancellation")
    assert seen["cancel"] and "已取消" in dlg.form_note.text()
    assert dlg.form.isEnabled() and dlg.btn_start.isEnabled()                    # usable again
    assert state.project.revision == rev and not state.history.can_undo
    # ... and a second run works
    serve_plan(monkeypatch, shorts_env)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_REVIEW, what="the second analysis")


def test_closing_the_dialog_cancels_a_running_analysis(qapp, shorts_env, monkeypatch):
    seen = {"cancel": threading.Event(), "started": threading.Event()}

    def slow(proj, ids, params=None, progress=None, cancel=None):
        seen["started"].set()
        while not cancel.is_set():
            time.sleep(0.01)
        seen["cancel"].set()
        raise Cancelled()

    monkeypatch.setattr(dlg_mod, "propose_shorts", slow)
    dlg, _ = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    shown(dlg)
    click(dlg.btn_start)
    pump(qapp, lambda: seen["started"].is_set())
    runner = dlg._runner
    dlg.reject()
    pump(qapp, lambda: seen["cancel"].is_set(), what="the worker to see the cancel flag")
    assert dlg.result() == QDialog.Rejected and not dlg.isVisible()
    runner.wait(10)
    qapp.processEvents()
    assert not runner.running


def test_failures_become_readable_messages(qapp, shorts_env, monkeypatch):
    dlg, _ = open_shorts(shorts_env, qapp, captions="off", use_speech=False)

    def fail_with(exc):
        def f(*a, **k):
            raise exc
        monkeypatch.setattr(dlg_mod, "propose_shorts", f)
        click(dlg.btn_start)
        pump(qapp, lambda: dlg.page == PAGE_FORM and not dlg.running)
        return dlg.form_note.text()

    assert fail_with(AutoEditError("影片太短，無法剪成短影音")) == "影片太短，無法剪成短影音"
    text = fail_with(ZeroDivisionError("division by zero"))
    assert text.startswith("發生未預期的錯誤（ZeroDivisionError）") and "Traceback" not in text and "記錄" in text
    assert dlg.form_note.level == "error" and dlg.form.isEnabled() and dlg.btn_start.isEnabled()

    def empty(*a, **k):
        return EditPlan(kind="shorts")

    monkeypatch.setattr(dlg_mod, "propose_shorts", empty)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_FORM and not dlg.running)
    assert "沒有找到可用的片段" in dlg.form_note.text() and dlg.form_note.level == "warn"


# ================================================================================================ review page
def test_candidates_come_from_the_plan(shorts_env, assemble_env):
    plan = shorts_env.plan()
    assets = {a["id"]: Asset.from_dict(a) for a in shorts_env.proj_dict["assets"]}
    cands = build_candidates(plan, assets)
    assert len(cands) == len(plan.sequences) >= 2
    for c in cands:
        assert c.kind == "shorts" and c.duration > 5 and c.why and c.score is not None and c.src_range
        assert c.thumb_asset is not None and c.src_range[0] <= c.thumb_time <= c.src_range[1] + 0.5
        assert c.crop is not None and 0 <= c.crop[0] <= 1 and 0 < c.crop[2] < 1       # the 9:16 window on a 16:9 frame
    assert "為什麼" not in cands[0].headline(1) and "第 1 支" in cands[0].headline(1)
    plan2 = assemble_env.plan()
    assets2 = {a["id"]: Asset.from_dict(a) for a in assemble_env.proj_dict["assets"]}
    one = build_candidates(plan2, assets2)
    assert len(one) == 1 and one[0].kind == "assemble" and one[0].sources >= 3 and one[0].why_label == "排序依據"


def test_review_page_rows_findings_and_screenshot(qapp, shorts_env, monkeypatch):
    serve_plan(monkeypatch, shorts_env)
    real_audit = api.audit_plan

    def noisy(plan, proj, rules=None):
        r = real_audit(plan, proj, rules)
        sid = plan.sequences[0]["id"]
        r.findings += [Finding("caption.reading", "warn", "字幕「大家好今天我們」閱讀速度 6.1 字/秒，偏快", f"{sid}/q1"),
                       Finding("shorts.first_cut", "warn", "第一個剪點在 2.4 秒，超過 2.05 秒", f"{sid}/c1"),
                       Finding("seq.invalid", "error", "測試錯誤：片段超出素材長度", f"{sid}/c1")]
        r.accepted = False
        return r

    monkeypatch.setattr(dlg_mod, "audit_plan", noisy)
    dlg, state = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    shown(dlg)
    analyse(dlg, qapp)
    assert len(dlg.rows) == len(dlg.plan.sequences) >= 2
    row = dlg.rows[0]
    assert row.check.isChecked() and row.btn_preview.text() == "預覽此段" and "為什麼選這段：" in row.why.text()
    assert "原片" in row.title.text() and "分數" in row.title.text() and "秒" in row.title.text()
    assert "1 個錯誤" in row.badge.text() and "2 個警告" in row.badge.text() and dlg.rows[1].badge.text() == ""
    sev = [s for s, _m in dlg.findings.rows()]
    assert sev[0] == "error" and sev.index("warn") > 0 and sev[-1] == "info" and sev.count("warn") == 2   # errors first
    assert "有錯誤時不能套用" in dlg.findings.headline_text
    assert dlg.selection_label.text().startswith(f"已選 {len(dlg.rows)} / {len(dlg.rows)} 支")
    QTest.qWait(50)
    shot(dlg, "review_findings")
    # unticking the row that owns the errors: they no longer count (the subset is audited again on apply)
    row.set_included(False)
    assert "不能套用" not in dlg.findings.headline_text and dlg.selection_label.text().startswith(f"已選 {len(dlg.rows) - 1} /")
    for r in dlg.rows:
        r.set_included(False)
    assert not dlg.btn_apply.isEnabled() and dlg.apply_selected() is False and "至少勾選" in dlg.apply_note.text()
    assert sequences_of(state) == sequences_of(shorts_env.state())


@needs_ffmpeg
def test_shorts_end_to_end_with_the_real_engine(qapp, shorts_env, media, human_review_guard):
    state = shorts_env.state()
    base_ids = sequences_of(state)
    assert not state.project.meta.get("ai_runs")
    dlg = ShortsDialog(state, shorts_env.ids[0])
    dlg.autoplay_previews = False
    shown(dlg)
    w = dlg.form.w
    pick(w["captions"], "off")
    w["use_speech"].setChecked(False)
    w["count"].setValue(3)
    w["target_seconds"].setValue(15)
    dlg.form.advanced.set_expanded(True)
    w["min_gap_between"].setValue(2.0)
    rev = state.project.revision
    audits: list = []
    real_audit = api.audit_plan
    dlg_mod_audit = dlg_mod.audit_plan
    try:
        dlg_mod.audit_plan = lambda plan, proj, rules=None: (audits.append(len(plan.sequences)), real_audit(plan, proj, rules))[1]
        analyse(dlg, qapp)
        assert state.project.revision == rev and sequences_of(state) == base_ids        # analysis never edits the project
        n = len(dlg.rows)
        assert n >= 2 and audits == [n]                                                  # audited once for the review
        assert dlg.receipt.accepted and dlg.findings.rows()[-1][0] == "info"
        assert any("尚未經人工審片" in m for _s, m in dlg.findings.rows())            # honest: machine audit only
        keep = [r.index for r in dlg.rows][1:]                                           # untick the first candidate
        dlg.rows[0].set_included(False)
        assert dlg.selected_indexes() == keep
        QTest.qWait(50)
        shot(dlg, "review")
        status: list = []
        state.status.connect(status.append)
        click(dlg.btn_apply)
        pump(qapp, lambda: dlg.result() == QDialog.Accepted, what="the apply")
    finally:
        dlg_mod.audit_plan = dlg_mod_audit
    assert audits == [n, len(keep)]                                                      # subset audited again
    new_ids = dlg.created_sequence_ids
    assert len(new_ids) == len(keep) and sequences_of(state) == base_ids + new_ids
    chosen = {dlg.plan.sequences[i]["meta"]["shorts"]["t0"] for i in keep}
    got = {state.project.sequence(i).meta["shorts"]["t0"] for i in new_ids}
    assert got == chosen                                                                 # exactly the ticked candidates
    assert state.seq_id == new_ids[0] and status and status[-1] == f"已建立 {len(new_ids)} 個序列（Ctrl+Z 可整批復原）"
    run = state.run_for_sequence(new_ids[0])
    assert run["human_review"] == "pending" and run["technical_qa"] == "not_run" and run["sequence_ids"] == new_ids
    assert human_review_guard == []                                                      # the dialog never reviews
    state.undo()                                                                         # ONE undo removes them all
    assert sequences_of(state) == base_ids and not state.project.meta.get("ai_runs")


@needs_ffmpeg
def test_real_engine_with_auto_captions_but_no_speech_says_so(qapp, shorts_env):
    """The default captions=auto while speech recognition is missing (autouse ``no_speech``): the analysis still runs on
    the real engine, nothing is downloaded or faked, and the review page says WHY there are no captions."""
    state = shorts_env.state()
    dlg = ShortsDialog(state, shorts_env.ids[0])
    dlg.autoplay_previews = False
    shown(dlg)
    assert dlg.form.params().captions == "auto"
    dlg.form.w["count"].setValue(2)
    dlg.form.w["target_seconds"].setValue(15)
    rev = state.project.revision
    analyse(dlg, qapp)
    assert dlg.receipt is not None and dlg.receipt.accepted and dlg.rows
    assert any("尚未安裝語音辨識" in n and "pip install -r requirements-speech.txt" in n for n in dlg.extra_notes)
    texts = [lab.text() for lab in dlg.findings.findChildren(QLabel)]
    assert any("分析備註" in t for t in texts)
    assert sum("尚未安裝語音辨識" in t for t in texts) == 1                              # said once, not twice
    assert state.project.revision == rev and not state.history.can_undo               # analysis never edits the project
    QTest.qWait(50)
    shot(dlg, "review_no_speech")


@needs_ffmpeg
def test_assemble_end_to_end_with_the_real_engine(qapp, assemble_env, media):
    state = assemble_env.state()
    base_ids = sequences_of(state)
    dlg = AssembleDialog(state, assemble_env.ids)
    dlg.autoplay_previews = False
    shown(dlg)
    assert "素材：4 個" in dlg.source_note.text() and "包含：a.mp4（0:08）、b.mp4" in dlg.source_note.text()
    assert dlg.source_note.toolTip().count("\n") == 3
    shot(dlg, "assemble_form")
    w = dlg.form.w
    w["target_seconds"].setValue(20)
    pick(w["order"], "as_given")
    pick(w["captions"], "off")
    pick(w["transition"], "crossfade")
    analyse(dlg, qapp)
    assert len(dlg.rows) == 1 and not dlg.rows[0].check.isVisibleTo(dlg) and dlg.rows[0].included()
    assert dlg.usage.isVisibleTo(dlg) and dlg.usage.rowCount() == 4
    rows = dlg.usage.rows_text()
    assert {r[0] for r in rows} == {"a.mp4", "b.mp4", "c.mp4", "d.mp4"} and all("秒" in r[1] for r in rows)
    assert "總長" in dlg.length_note.text() and "目標 0:20" in dlg.length_note.text()
    assert dlg.length_note.level in ("ok", "warn")
    assert dlg.selection_label.text().startswith("成品長度")
    QTest.qWait(50)
    shot(dlg, "assemble_review")
    click(dlg.btn_apply)
    pump(qapp, lambda: dlg.result() == QDialog.Accepted, what="the apply")
    assert len(dlg.created_sequence_ids) == 1 and sequences_of(state) == base_ids + dlg.created_sequence_ids
    seq = state.sequence()
    assert state.seq_id == dlg.created_sequence_ids[0] and seq.duration > 5 and seq.meta["ai"]["run_id"]
    assert state.run_for_sequence()["human_review"] == "pending"
    state.undo()
    assert sequences_of(state) == base_ids


def test_audit_errors_block_apply(qapp, shorts_env, monkeypatch):
    serve_plan(monkeypatch, shorts_env)
    dlg, state = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    shown(dlg)
    analyse(dlg, qapp)
    applied: list = []
    monkeypatch.setattr(dlg_mod, "apply_plan", lambda *a, **k: applied.append(a))
    audits: list = []

    def refuse(plan, proj, rules=None):
        audits.append(len(plan.sequences))
        return fake_receipt(plan, proj, [Finding("seq.invalid", "error", "片段「c9」超出素材長度（測試）", f"{plan.sequences[0]['id']}/c9"),
                                         Finding("human.pending", "info", "尚未審片", "plan")])

    monkeypatch.setattr(dlg_mod, "audit_plan", refuse)
    rev, base_ids = state.project.revision, sequences_of(state)
    click(dlg.btn_apply)
    pump(qapp, lambda: dlg.apply_note.text() != "" and not dlg.running, what="the blocked apply")
    assert audits == [len(dlg.rows)] and applied == []                                    # audited, NOT applied
    assert "審核未通過，沒有套用任何內容" in dlg.apply_note.text() and "超出素材長度" in dlg.apply_note.text()
    assert dlg.apply_note.level == "error" and dlg.page == PAGE_REVIEW and dlg.result() != QDialog.Accepted
    assert dlg.findings.rows()[0] == ("error", "片段「c9」超出素材長度（測試）")
    assert "審核未通過" in dlg.findings.headline_text
    assert state.project.revision == rev and sequences_of(state) == base_ids and dlg.created_sequence_ids == []
    assert dlg.btn_apply.isEnabled() and dlg.btn_back.isEnabled()                         # usable: fix the selection, retry
    QTest.qWait(50)
    shot(dlg, "review_blocked")
    dlg.rows[0].set_included(False)                                                       # changing the selection clears it
    assert dlg.apply_note.text() == ""


def test_apply_failure_stays_on_the_review_page(qapp, shorts_env, monkeypatch):
    serve_plan(monkeypatch, shorts_env)
    dlg, state = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    shown(dlg)
    analyse(dlg, qapp)
    from autocut.core.plan import PlanError

    def stale(*a, **k):
        raise PlanError("專案在審核之後又被編輯過，請重新審核")

    monkeypatch.setattr(dlg_mod, "apply_plan", stale)
    click(dlg.btn_apply)
    pump(qapp, lambda: dlg.apply_note.text() != "" and not dlg.running)
    assert "專案在審核之後又被編輯過" in dlg.apply_note.text() and dlg.page == PAGE_REVIEW
    assert dlg.result() != QDialog.Accepted and dlg.btn_apply.isEnabled()


def test_back_to_settings_and_reanalyse(qapp, shorts_env, monkeypatch):
    serve_plan(monkeypatch, shorts_env)
    dlg, _ = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    shown(dlg)
    analyse(dlg, qapp)
    n = len(dlg.rows)
    click(dlg.btn_back)
    assert dlg.page == PAGE_FORM and dlg.form.params().captions == "off"
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_REVIEW and len(dlg.rows) == n)
    assert dlg.parent() is None and len(dlg.rows) == n                                   # rows rebuilt, not duplicated


# ================================================================================================ preview
@needs_ffmpeg
def test_preview_opens_on_a_throwaway_project_and_shows_a_frame(qapp, shorts_env, monkeypatch):
    serve_plan(monkeypatch, shorts_env)
    dlg, state = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    shown(dlg)
    analyse(dlg, qapp)
    rev, base_ids = state.project.revision, sequences_of(state)
    click(dlg.rows[1].btn_preview)
    assert len(dlg.preview_dialogs) == 1
    pv = dlg.preview_dialogs[0]
    assert pv.isVisible() and not pv.isModal() and pv.windowTitle().startswith("預覽：第 2 支")
    mon = pv.monitor
    assert mon.wait_idle(20)
    pump(qapp, lambda: mon.frames_shown >= 1, what="the first preview frame")
    # a private AppState on a copy that holds the candidate; the real project and its undo history are untouched
    assert pv.preview_state is not state and pv.preview_state.project is not state.project
    assert pv.preview_state.seq_id in sequences_of(pv.preview_state) and pv.preview_state.seq_id not in base_ids
    assert state.project.revision == rev and sequences_of(state) == base_ids and not state.history.can_undo
    assert mon.playback.state is pv.preview_state and not mon.btn_reframe.isVisibleTo(pv)
    click(dlg.rows[1].btn_preview)                                                        # the same one is not opened twice
    assert len(dlg.preview_dialogs) == 1
    pv.resize(540, 800)
    mon.wait_idle(10)
    shot(pv, "preview")
    comp = mon.compositor
    pv.close()
    assert pv.isHidden() and dlg.preview_dialogs == []
    assert comp.open_readers == 0 and mon._stopped                                       # compositor closed, decoder stopped
    assert state.project.revision == rev and sequences_of(state) == base_ids


@needs_ffmpeg
def test_closing_the_dialog_closes_open_previews(qapp, shorts_env, monkeypatch):
    serve_plan(monkeypatch, shorts_env)
    dlg, _ = open_shorts(shorts_env, qapp, captions="off", use_speech=False)
    shown(dlg)
    analyse(dlg, qapp)
    pv = dlg.preview_candidate(0)
    assert pv is not None and pv.isVisible()
    pv.monitor.wait_idle(20)
    dlg.reject()
    assert pv.isHidden() and pv.monitor._stopped and dlg.preview_dialogs == []


@needs_ffmpeg
def test_assemble_preview_and_untouched_history(qapp, assemble_env, monkeypatch):
    serve_plan(monkeypatch, assemble_env)
    state = assemble_env.state()
    dlg = AssembleDialog(state, assemble_env.ids)
    dlg.autoplay_previews = False
    shown(dlg)
    analyse(dlg, qapp)
    pv = dlg.preview_candidate(0)
    assert pv is not None and pv.windowTitle().startswith("預覽：合併後的長片")
    assert pv.monitor.wait_idle(20)
    assert not state.history.can_undo
    pv.close()
    dlg.reject()


# ================================================================================================ screenshots of the forms
def test_form_screenshots_at_the_minimum_size(qapp, shorts_env):
    dlg, _ = open_shorts(shorts_env, qapp)
    shown(dlg, 900, 640)
    pump(qapp, lambda: dlg.speech_status is not None)
    shot(dlg, "shorts_form")
    dlg.form.advanced.set_expanded(True)
    QTest.qWait(30)
    shot(dlg, "shorts_form_advanced")
    asm = AssembleDialog(shorts_env.state(), shorts_env.ids)
    shown(asm, 900, 640)
    shot(asm, "assemble_form_min")
    for d in (dlg, asm):
        assert d.size().width() >= 900 and d.size().height() >= 640


def test_cutlist_row_validates_disables_count_and_is_not_remembered(qapp, shorts_env, media, monkeypatch, tmp_path):
    calls: list = []
    serve_plan(monkeypatch, shorts_env, calls=calls)
    dlg, _ = open_shorts(shorts_env, qapp)
    row, count = dlg.form.w["cutlist_file"], dlg.form.w["count"]
    assert count.isEnabled() and row.path() == ""
    row.set_path(str(tmp_path / "沒有.json"))
    assert not count.isEnabled() and not dlg.form.w["target_seconds"].isEnabled()   # the list decides count + length
    assert not dlg.form.w["highlight_engine"].isEnabled()           # nothing is searched: the AI judge is not called
    assert dlg.start_analysis() is False and "找不到剪輯清單" in dlg.form_note.text()
    bad = tmp_path / "壞的 清單.json"
    bad.write_text('{"shorts": [{"ranges": [[5, 3]]}]}', encoding="utf-8")
    row.set_path(str(bad))
    assert dlg.start_analysis() is False and "結束時間必須晚於開始時間" in dlg.form_note.text()
    assert calls == [] and dlg.page == PAGE_FORM
    good = tmp_path / "第一集.json"
    good.write_text('{"shorts": [{"title": "一", "ranges": [[1, 3]]}]}', encoding="utf-8")
    row.set_path(str(good))
    assert dlg.form.validate() == "" and dlg.form.params().cutlist_file == str(good)
    forms.save_params(forms.KEY_SHORTS, dlg.form.params())
    assert forms.load_params(ShortsParams, forms.KEY_SHORTS).cutlist_file == ""   # it belongs to one video
    row.set_path("")
    assert count.isEnabled() and dlg.form.w["target_seconds"].isEnabled() and dlg.form.w["highlight_engine"].isEnabled()
