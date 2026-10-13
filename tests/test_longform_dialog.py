"""GUI for the new features, offscreen: the 「長片精華」 dialog on the REAL engine, 「一鍵產生字幕」, the 找重點 status
line and the AI-key button (no network: the speech engine and the key dialog are stand-ins)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings                                       # noqa: E402

from autocut.analysis import speech as speech_mod                          # noqa: E402
from autocut.core.project import Sequence                                  # noqa: E402
from autocut.gui import autoedit_dialog as dlg_mod                         # noqa: E402
from autocut.gui import autoedit_forms as forms                            # noqa: E402
from autocut.gui import media_bin                                          # noqa: E402
from autocut.gui.autoedit_dialog import PAGE_FORM, PAGE_REVIEW, LongformDialog, ShortsDialog   # noqa: E402
from autocut.gui.state import AppState                                     # noqa: E402

from test_autoedit_dialog import (HAVE_FFMPEG, click, close_leftover_dialogs, clean_settings, human_review_guard,  # noqa: E402,F401
                                  media, modals, no_speech, pick, probe_project, pump, shot, watchdog)

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture(autouse=True)
def clean_ai(qapp, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = QSettings("AutoCutStudio", "AutoCutStudio")
    s.remove("ai")
    s.sync()
    yield
    s.remove("ai")
    s.sync()


@pytest.fixture
def talk(media, tmp_path):
    """A private copy of the long test video (so sidecar subtitle files never leak between tests)."""
    import shutil
    d = tmp_path / "影片 資料夾"
    d.mkdir()
    dst = d / "訪談.mp4"
    shutil.copy(media / "long.mp4", dst)
    proj = probe_project([dst])
    return proj, next(iter(proj.assets)), dst


@needs_ffmpeg
def test_longform_dialog_runs_the_real_engine_and_applies_one_undo_step(qapp, talk):
    proj, aid, path = talk
    (path.with_suffix(".srt")).write_text("1\n00:00:01,000 --> 00:00:03,500\n大家好\n\n2\n00:00:20,000 --> 00:00:23,500\n"
                                          "但是沒想到我竟然賠了300萬！\n", encoding="utf-8")
    state = AppState(proj)
    dlg = LongformDialog(state, aid)
    dlg.autoplay_previews = False
    dlg.resize(1100, 760)
    dlg.show()
    qapp.processEvents()
    assert dlg.page == PAGE_FORM and dlg.form.params().mode == "highlights"
    assert "訪談.srt" in dlg.form.transcript_note.text()                 # the sidecar file is found and named
    shot(dlg, "longform_form")
    pick(dlg.form.w["mode"], "clean")
    assert not dlg.form.w["keep_ratio"].isEnabled()
    dlg.form.w["speaker_zoom"].setChecked(False)
    assert not dlg.form.w["zoom_min_turn"].isEnabled()
    dlg.form.w["use_speech"].setChecked(False)
    click(dlg.btn_start)
    pump(qapp, lambda: dlg.page == PAGE_REVIEW, what="the review page")
    shot(dlg, "longform_review")
    assert dlg.plan.kind == "longform" and len(dlg.rows) == 1
    assert "精華長片" in dlg.rows[0].cand.headline(1) and dlg.rows[0].cand.why
    before = len(state.project.sequences)
    click(dlg.btn_apply)
    pump(qapp, lambda: len(state.project.sequences) == before + 1, what="the new sequence")
    seq = state.project.sequences[-1]
    assert seq.name.startswith("長片去水詞") and state.project.meta["ai_runs"][-1]["human_review"] == "pending"
    state.undo()
    assert len(state.project.sequences) == before


@needs_ffmpeg
def test_one_click_subtitles_writes_the_srt_and_uses_it(qapp, talk, monkeypatch, modals):
    proj, aid, path = talk
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    monkeypatch.setattr(speech_mod, "model_cached", lambda model_size="small": True)
    calls = []

    def fake_make(video, overwrite=False, allow_download=False, progress=None, cancel=None, **kw):
        calls.append((video, overwrite, allow_download, kw.get("language")))
        progress(0.5, "語音辨識…")
        out = Path(video).with_suffix(".srt")
        out.write_text("1\n00:00:01,000 --> 00:00:02,000\n測試\n", encoding="utf-8")
        return out, 1

    monkeypatch.setattr(dlg_mod, "make_subtitle_file", fake_make)
    dlg = ShortsDialog(AppState(proj), aid)
    dlg.show()
    qapp.processEvents()
    assert "還沒有字幕檔" in dlg.form.transcript_note.text()
    click(dlg.form.btn_subtitles)
    pump(qapp, lambda: dlg.page == PAGE_FORM and "已產生 1 條字幕" in dlg.form_note.text(), what="the subtitles")
    assert calls == [(str(path), False, False, "zh")]
    assert dlg.form.params().transcript_file == str(path.with_suffix(".srt"))
    assert "訪談.srt" in dlg.form.transcript_note.text()
    # a second click asks before overwriting; 'no' keeps the file and just selects it
    click(dlg.form.btn_subtitles)
    assert modals[-1][0] == "ask" and "已經存在" in modals[-1][2] and len(calls) == 1
    # the subtitle file is per video: never remembered for the next one
    forms.save_params(forms.KEY_SHORTS, dlg.form.params())
    assert forms.load_params(dlg.PARAMS, forms.KEY_SHORTS).transcript_file == ""


@needs_ffmpeg
def test_one_click_subtitles_explains_missing_speech_and_asks_before_download(qapp, talk, monkeypatch, modals):
    proj, aid, _path = talk
    dlg = ShortsDialog(AppState(proj), aid)
    assert dlg.make_subtitles() is False and "pip install" in dlg.form_note.text()
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    assert dlg.make_subtitles() is False                                  # model missing, 'no' to the download
    assert modals[-1][1] == "下載語音模型"


@needs_ffmpeg
def test_ai_key_button_and_status(qapp, talk, monkeypatch):
    proj, aid, _path = talk
    dlg = ShortsDialog(AppState(proj), aid)
    dlg.show()
    pick(dlg.form.w["highlight_engine"], "claude")
    qapp.processEvents()
    assert dlg.form.btn_ai_key.isVisible()
    monkeypatch.setattr(dlg_mod.importlib.util, "find_spec", lambda name: object())
    dlg._fill_transcript_status()
    assert "還沒有金鑰" in dlg.form.transcript_note.text()
    monkeypatch.setattr(dlg_mod, "ask_text", lambda *a, **k: "sk-test-123")
    assert dlg.set_ai_key()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-123" and forms.saved_ai_key() == "sk-test-123"
    assert "只會傳送字幕文字" in dlg.form.transcript_note.text()
    shot(dlg, "shorts_ai_highlights")
    monkeypatch.setattr(dlg_mod, "ask_text", lambda *a, **k: "")              # empty = forget the key
    dlg.set_ai_key()
    assert forms.saved_ai_key() == "" and "ANTHROPIC_API_KEY" not in os.environ
    pick(dlg.form.w["highlight_engine"], "local")
    qapp.processEvents()
    assert not dlg.form.btn_ai_key.isVisible()


@needs_ffmpeg
def test_longform_form_validation(qapp, talk):
    proj, aid, _path = talk
    dlg = LongformDialog(AppState(proj), aid)
    dlg.form.w["zoom_min_turn"].setValue(10.0)
    dlg.form.w["zoom_max"].setValue(5.0)
    assert dlg.start_analysis() is False and "最長特寫" in dlg.form_note.text()
