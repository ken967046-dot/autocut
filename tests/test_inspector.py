"""Inspector (片段 / 字幕 / 序列 / 檢查): the REAL ``InspectorPanel`` driven by typed text, slider presses / drags / wheel,
combo activations, button clicks and table-cell editing, judged by the PROJECT content (clips, cues, sequence
attributes, ``ai_runs``) and by the number of undo steps.

* a committed edit is exactly ONE undo step (``one_undo``): one undo restores the project byte for byte AND the controls
  show the original value again, one redo re-applies;
* anything still in progress (a slider being dragged, text being typed, a refresh) or refused (bad input, locked track,
  occupied slot) leaves the project untouched and adds no undo step (``untouched``);
* the panel never commits on its own: selecting, moving the playhead, undo / redo and ``refresh()`` only *show* values.

Offscreen Qt blocks forever on a modal dialog, so every dialog seam of ``inspector`` (``ask / inform / open_file /
save_file / pick_color / ask_number``) raises unless the test replaces it, and an exception inside a Qt slot (which Qt
would swallow) fails the test.  A watchdog kills a stuck run.
"""
from __future__ import annotations

import faulthandler
import importlib.util
import math
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt                                            # noqa: E402
from PySide6.QtGui import QColor, QInputMethodEvent, QWheelEvent                          # noqa: E402
from PySide6.QtTest import QTest                                                          # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox,   # noqa: E402
                               QLineEdit, QSlider, QStyle, QStyleOptionSlider, QWidget)

from autocut.analysis import speech as speech_mod                                         # noqa: E402
from autocut.autoedit import captions as captions_mod                                     # noqa: E402
from autocut.core import ops                                                              # noqa: E402
from autocut.core import plan as planmod                                                  # noqa: E402
from autocut.core.project import Asset, CaptionCue, Project, validate_sequence            # noqa: E402
from autocut.core.timecode import fmt_clock, fmt_timecode, snap_time                      # noqa: E402
from autocut.gui import inspector as insp                                                 # noqa: E402
from autocut.gui.inspector import (CaptionsTab, ChecksTab, ClipTab, InspectorPanel, SequenceTab,   # noqa: E402
                                   db_to_gain, gain_to_db)
from autocut.gui.state import AppState                                                    # noqa: E402
from autocut.gui.widgets import QA_TEXT, VERDICT_TEXT, HumanReviewBox                  # noqa: E402
from autocut.gui.workers import wait_for_idle                                             # noqa: E402
from autocut.media.ffmpeg import Cancelled                                                # noqa: E402
from autocut.util import IdGen, new_id                                                    # noqa: E402

pytestmark = pytest.mark.gui

DIALOGS = ("ask", "inform", "open_file", "save_file", "pick_color", "ask_number")
SHOTS = Path(__file__).resolve().parents[1] / "screenshots"


# ======================================================================================================== fixtures
@pytest.fixture(autouse=True)
def _watchdog():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _guards(monkeypatch):
    """Modal dialogs would hang offscreen Qt: each seam refuses unless the test replaces it.  Exceptions raised inside
    Qt slots are swallowed by Qt (only printed), so route them to the test instead."""
    problems: list[str] = []

    def refuse(name):
        def seam(*a, **k):
            problems.append(f"unexpected dialog: {name}{tuple(str(x)[:40] for x in a[1:3])}")
            raise AssertionError(problems[-1])
        return seam

    for name in DIALOGS:
        monkeypatch.setattr(insp, name, refuse(name))
    monkeypatch.setattr(sys, "excepthook", lambda t, v, tb: problems.append(f"{t.__name__}: {v}"))
    yield problems
    assert not problems, problems


def fake_video(aid: str, dur: float, w: int = 1920, h: int = 1080, audio: bool = True) -> Asset:
    return Asset(id=aid, path=f"/nonexistent/{aid}.mp4", name=f"{aid}.mp4", kind="video", duration=dur, width=w,
                 height=h, fps=30.0, has_video=True, has_audio=audio, audio_channels=2 if audio else 0,
                 sample_rate=48000 if audio else 0, fingerprint=aid)


def make_project() -> Project:
    p = Project.new("insp")
    p.assets["a1"] = fake_video("a1", 20.0)
    p.assets["a2"] = fake_video("a2", 10.0)
    p.assets["a3"] = fake_video("a3", 8.0, audio=False)
    p.assets["m1"] = Asset(id="m1", path="/nonexistent/m1.mp3", name="m1.mp3", kind="audio", duration=60.0,
                           has_video=False, has_audio=True, audio_channels=2, sample_rate=44100)
    p.assets["i1"] = Asset(id="i1", path="/nonexistent/i1.png", name="i1.png", kind="image", duration=0.0,
                           width=800, height=600, has_video=False, has_audio=False)
    return p


class Rig:
    """An ``InspectorPanel`` (shown, active, 420 px wide like the dock) on a fresh project plus typing / slider helpers.

    NB: undo / redo rebuild the project's objects, so never keep a ``Clip`` / ``Sequence`` across them: ask ``clip()`` /
    ``seq`` again.
    """

    def __init__(self, app, populate: bool = True, w: int = 420, h: int = 1000) -> None:
        self.app = app
        self.st = AppState(make_project())
        self.status: list[str] = []
        self.st.status.connect(self.status.append)
        seq = self.st.sequence()
        self.v1, self.v2 = (t.id for t in seq.video_tracks[:2])
        self.a1, self.a2 = (t.id for t in seq.audio_tracks[:2])
        self.ids: dict[str, str] = {}
        if populate:
            self.populate()
        self.panel = InspectorPanel(self.st)
        self.panel.resize(w, h)
        self.panel.show()
        self.panel.activateWindow()
        self.settle()

    # ---- project content
    def populate(self) -> None:
        """V1: A 0-4 (src 0-4) | B 5-8 (src 0-3) | C 10-14 (a1 src 5-9) -- V2: D 2-5 | S 6-10 (silent) | I 12-17 (image)
        -- A1: M 0-12 (music)  -- cues 1-2 / 3-4.5 / 6-7."""
        st = self.st

        def put(key, aid, tid, start, s_in=0.0, s_out=None):
            self.ids[key] = st.op("seed", "place_clip", aid, tid, start, s_in, s_out, "overwrite").id

        put("A", "a1", self.v1, 0.0, 0.0, 4.0)
        put("B", "a2", self.v1, 5.0, 0.0, 3.0)
        put("C", "a1", self.v1, 10.0, 5.0, 9.0)
        put("D", "a2", self.v2, 2.0, 0.0, 3.0)
        put("S", "a3", self.v2, 6.0, 0.0, 4.0)
        put("I", "i1", self.v2, 12.0, 0.0, 5.0)
        put("M", "m1", self.a1, 0.0, 0.0, 12.0)
        for s, e, txt in ((1.0, 2.0, "第一句"), (3.0, 4.5, "第二句"), (6.0, 7.0, "第三句")):
            st.op("seed", "add_cue", s, e, txt)
        st.history.clear()

    def seed(self, label: str, name: str, /, *args, **kw):
        """Setup edit made straight through the editor (not the thing under test); history starts clean afterwards."""
        r = self.st.op(label, name, *args, **kw)
        self.st.history.clear()
        self.settle()
        return r

    @property
    def seq(self):
        return self.st.sequence()

    def clip(self, key: str):
        return self.seq.find_clip(self.ids.get(key, key))[1]

    def cues(self) -> list[tuple[float, float, str]]:
        return [(round(q.start, 3), round(q.end, 3), q.text) for q in self.seq.cues]

    def layout(self, tid: str) -> list[tuple[float, float, float, float]]:
        tr = self.seq.find_track(tid)
        return [(round(c.start, 3), round(c.end, 3), round(c.src_in, 3), round(c.src_out, 3)) for c in tr.clips]

    def content(self) -> str:
        return self.st.project.content_json()

    @property
    def undos(self) -> int:
        return len(self.st.history._undo)

    # ---- panel
    @property
    def clip_tab(self) -> ClipTab:
        return self.panel.clip_tab

    @property
    def cap(self) -> CaptionsTab:
        return self.show_tab(self.panel.captions_tab)

    @property
    def seq_tab(self) -> SequenceTab:
        return self.show_tab(self.panel.sequence_tab)

    @property
    def checks(self) -> ChecksTab:
        return self.show_tab(self.panel.checks_tab)

    def show_tab(self, tab):
        """Bring a tab to the front (hidden widgets get no focus and no shortcuts, so a user-like test needs this)."""
        if self.panel.currentWidget() is not tab:
            self.panel.setCurrentWidget(tab)
            self.settle()
        return tab

    def pick(self, *keys: str) -> None:
        self.st.select([self.ids.get(k, k) for k in keys])
        self.settle()

    def settle(self, ms: int = 20) -> None:
        """Run the event loop (deferred ``QTimer.singleShot(0)`` commits need more than one ``processEvents``)."""
        QTest.qWait(ms)

    def wait_until(self, cond, timeout: int = 4000) -> bool:
        end = timeout
        while end > 0:
            if cond():
                return True
            QTest.qWait(20)
            end -= 20
        return bool(cond())

    # ---- typing
    def focus(self, w) -> None:
        w.setFocus()
        self.settle(5)
        assert w.hasFocus(), "the widget could not get keyboard focus"

    def type_in(self, w, text: str, finish: str = "enter") -> None:
        """Click-into-the-field behaviour: focus, select all, type ``text`` and finish with Enter / focus loss."""
        self.focus(w)
        w.selectAll()
        send_text(w, text)
        if finish == "enter":
            QTest.keyClick(w, Qt.Key_Return)
        elif finish == "blur":
            w.clearFocus()
        self.settle()


def send_text(w, text: str) -> None:
    """ASCII goes through key events; CJK is committed like an input method (QTest.keyClicks only knows ASCII)."""
    if text == "":
        QTest.keyClick(w, Qt.Key_Delete)                  # the selection is deleted
    elif text.isascii():
        QTest.keyClicks(w, text)
    else:
        ev = QInputMethodEvent()
        ev.setCommitString(text)
        QApplication.sendEvent(w if isinstance(w, QLineEdit) else w.lineEdit(), ev)


@pytest.fixture
def rig(qapp):
    r = Rig(qapp)
    yield r
    r.panel.close()
    r.panel.deleteLater()
    qapp.processEvents()


@pytest.fixture
def empty_rig(qapp):
    r = Rig(qapp, populate=False)
    yield r
    r.panel.close()
    r.panel.deleteLater()
    qapp.processEvents()


@contextmanager
def one_undo(r: Rig, label: str | None = None, ui=None):
    """The block must change the project by exactly ONE undo step.  One undo restores the project byte for byte (and
    ``ui()`` -- the displayed values of the controls -- shows the original again); one redo re-applies both."""
    before, n0 = r.content(), r.undos
    ui_before = ui() if ui else None
    yield
    r.settle()
    after = r.content()
    assert after != before, "the edit changed nothing"
    assert r.undos == n0 + 1, f"the edit must be exactly one undo step, got {r.undos - n0}"
    if label:
        assert r.st.history.undo_label == label
    ui_after = ui() if ui else None
    r.st.undo()
    r.settle()
    assert r.content() == before, "one undo must restore the project"
    if ui:
        assert ui() == ui_before, "after undo the controls must show the original value"
    r.st.redo()
    r.settle()
    assert r.content() == after, "redo must re-apply the edit"
    if ui:
        assert ui() == ui_after, "after redo the controls must show the edited value again"


@contextmanager
def untouched(r: Rig):
    """The block must not change the project and must not add an undo step."""
    before, n0 = r.content(), r.undos
    yield
    r.settle()
    assert r.content() == before, "the project must not change"
    assert r.undos == n0, "no undo step may be added"


# ---- slider helpers (real mouse events on the QSlider inside a SliderField)
def handle_pt(sl) -> QPoint:
    opt = QStyleOptionSlider()
    sl.initStyleOption(opt)
    return sl.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, sl).center()


def wheel_over(r: Rig, w, dy: int = 120):
    """One wheel notch with the pointer over ``w``, delivered the way Qt delivers a real one: to the widget under the
    pointer first, then up the parent chain until somebody accepts it (a synthetic ``sendEvent`` does not propagate
    by itself).  Returns the widget that accepted the notch, ``None`` when nobody did."""
    node, pos = w, QPointF(8, 5)
    while node is not None:
        ev = QWheelEvent(pos, QPointF(node.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, dy), Qt.NoButton,
                         Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(node, ev)
        if ev.isAccepted():
            break
        pos += QPointF(node.pos())
        node = node.parentWidget()
    r.settle()
    return node


def drag_slider(r: Rig, sl, dx: int, release: bool = True) -> None:
    """Press on the handle, move in three stages (never committing on the way), optionally release."""
    p = handle_pt(sl)
    QTest.mousePress(sl, Qt.LeftButton, Qt.NoModifier, p)
    for f in (0.3, 0.7, 1.0):
        QTest.mouseMove(sl, QPoint(p.x() + int(dx * f), p.y()))
        QApplication.processEvents()
    if release:
        QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, QPoint(p.x() + dx, p.y()))
        r.settle()


def step_and_enter(r: Rig, f, steps: int) -> None:
    """Arrow-key style stepping of a number box (it stops at the box limits), then Enter."""
    r.focus(f)
    f.stepBy(steps)
    QTest.keyClick(f, Qt.Key_Return)
    r.settle()


def choose(r: Rig, combo, data) -> None:
    """What picking an entry from the popup does: set the index, then ``activated`` (user-only signal)."""
    i = combo.findData(data)
    assert i >= 0, f"no entry {data!r}"
    combo.setCurrentIndex(i)
    combo.activated.emit(i)
    r.settle()


def tc(t: float, fps: float = 30.0) -> str:
    return fmt_timecode(t, fps)


# ============================================================================================ panel structure
def test_panel_has_four_tabs_in_spec_order(rig):
    p = rig.panel
    assert [p.tabText(i) for i in range(p.count())] == ["片段", "字幕", "序列", "檢查"]
    assert [type(p.widget(i)) for i in range(p.count())] == [ClipTab, CaptionsTab, SequenceTab, ChecksTab]
    assert p.minimumWidth() >= 340


def test_building_showing_and_refreshing_never_commits(rig):
    """Programmatic refreshes (selection, playhead, undo / redo, explicit refresh) only show values."""
    with untouched(rig):
        for keys in (("A",), ("B", "C"), ("D",), ("M",), ("S",), ("I",), ()):
            rig.pick(*keys)
            for t in (0.0, 1.0, 3.3, 20.0):
                rig.st.set_playhead(t)
            for tab in (rig.clip_tab, rig.cap, rig.seq_tab, rig.checks):
                tab.refresh()
        for i in range(rig.panel.count()):
            rig.show_tab(rig.panel.widget(i))
    assert rig.status == []


# ================================================================================================= 片段: states
def test_clip_tab_empty_state_without_selection(rig):
    t = rig.clip_tab
    assert not t.empty.isHidden() and t.empty.text()
    assert all(g.isHidden() for g in t.groups)
    assert t.title.text() == ""
    rig.pick("A")
    assert t.empty.isHidden() and not t.g_time.isHidden()
    rig.st.clear_selection()
    rig.settle()
    assert not t.empty.isHidden() and all(g.isHidden() for g in t.groups) and t.title.text() == ""


def test_empty_project_clip_tab_shows_empty_state(empty_rig):
    t = empty_rig.clip_tab
    assert not t.empty.isHidden() and all(g.isHidden() for g in t.groups)
    empty_rig.st.select(["nope"])                       # a stale id selects nothing
    empty_rig.settle()
    assert not t.empty.isHidden()


def test_single_video_clip_shows_its_values(rig):
    rig.pick("C")
    t = rig.clip_tab
    assert "a1.mp4" in t.title.text() and "V1" in t.title.text()
    assert [f.text() for f in (t.f_start, t.f_dur, t.f_in, t.f_out)] == [tc(10.0), tc(4.0), tc(5.0), tc(9.0)]
    assert (t.f_speed.value(), t.f_vol.value(), t.f_fade_in.value(), t.f_fade_out.value()) == (1.0, 0.0, 0.0, 0.0)
    assert t.f_fit.currentData() == "cover" and t.f_zoom.box.value() == 1.0
    assert (t.f_cx.box.value(), t.f_cy.box.value()) == (0.5, 0.5)
    assert t.f_flip.checkState() == Qt.Unchecked and not t.f_flip.isTristate()
    assert (t.f_bright.box.value(), t.f_contrast.box.value(), t.f_sat.box.value()) == (0.0, 1.0, 1.0)
    for g in t.groups:
        assert not g.isHidden()
    assert not t.f_xfade.isHidden()


def test_controls_follow_selection_and_external_changes(rig):
    t = rig.clip_tab
    rig.seed("seed", "set_clip_props", rig.ids["B"], volume=0.5, brightness=0.25, flip_h=True, fit="blur", zoom=2.0)
    rig.pick("A")
    assert t.f_vol.value() == 0.0 and t.f_bright.box.value() == 0.0 and t.f_fit.currentData() == "cover"
    rig.pick("B")
    assert t.f_vol.value() == pytest.approx(gain_to_db(0.5), abs=0.06)
    assert t.f_bright.box.value() == 0.25 and t.f_bright.slider.value() == 25
    assert t.f_fit.currentData() == "blur" and t.f_zoom.box.value() == 2.0 and t.f_flip.isChecked()
    assert t.f_start.text() == tc(5.0)
    # another widget (here: the editor) changes the clip: the open inspector follows, undo included
    rig.st.op("外部修改", "set_clip_props", rig.ids["B"], brightness=-0.5, volume=2.0)
    rig.settle()
    assert t.f_bright.box.value() == -0.5 and t.f_vol.value() == pytest.approx(6.02, abs=0.06)
    rig.st.undo()
    rig.settle()
    assert t.f_bright.box.value() == 0.25 and t.f_vol.value() == pytest.approx(gain_to_db(0.5), abs=0.06)
    rig.st.set_sequence(rig.st.sequence().id)           # same id: no-op
    rig.st.clear_selection()
    rig.settle()
    assert not t.empty.isHidden()


def test_group_visibility_by_clip_kind(rig):
    t = rig.clip_tab
    rig.pick("M")                                       # music clip on an audio track: no picture, no look
    assert not t.g_audio.isHidden() and t.g_picture.isHidden() and t.g_look.isHidden() and not t.g_time.isHidden()
    assert "A1" in t.title.text()
    rig.pick("S")                                       # silent video: picture yes, audio group no
    assert t.g_audio.isHidden() and not t.g_picture.isHidden() and not t.g_look.isHidden()
    rig.pick("I")                                       # still image: same
    assert t.g_audio.isHidden() and not t.g_picture.isHidden()
    rig.pick("A")
    assert not t.g_audio.isHidden() and not t.g_picture.isHidden()
    rig.pick("A", "S")                                  # a mix is only as capable as its weakest clip
    assert t.g_audio.isHidden()
    rig.pick("A", "M")
    assert t.g_picture.isHidden() and t.g_look.isHidden()


def test_missing_asset_clip_does_not_crash_the_inspector(rig):
    del rig.st.project.assets["a2"]
    rig.st.history.clear()
    rig.pick("B")
    assert "找不到素材" in rig.clip_tab.title.text()
    assert rig.clip_tab.g_audio.isHidden()


def test_clip_title_escapes_markup_in_labels(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], label="<b>&x")
    rig.pick("A")
    assert "&lt;b&gt;&amp;x" in rig.clip_tab.title.text()


# ============================================================================================ 片段: time fields
def test_time_fields_are_timecodes_of_the_sequence_fps(rig):
    rig.seed("seed", "set_sequence_format", fps=25.0)
    rig.pick("C")
    t = rig.clip_tab
    assert t.f_start.text() == fmt_timecode(rig.clip("C").start, 25.0) == "00:00:10:00"
    rig.st.op("改影格率", "set_sequence_format", fps=24.0)
    rig.settle()
    assert t.f_start.text() == fmt_timecode(rig.clip("C").start, 24.0)
    assert t.f_dur.text() == fmt_timecode(rig.clip("C").duration, 24.0)


def test_start_field_moves_the_clip_snapped_to_a_frame_as_one_undo(rig):
    rig.pick("C")
    f = rig.clip_tab.f_start
    with one_undo(rig, "移動片段", ui=lambda: f.text()):
        rig.type_in(f, "00:00:20:05")                    # 20 s + 5 frames
    c = rig.clip("C")
    assert c.start == pytest.approx(20 + 5 / 30) and c.end == pytest.approx(c.start + 4.0)
    assert (c.src_in, c.src_out) == (5.0, 9.0)
    assert f.text() == tc(20 + 5 / 30)


def test_start_field_accepts_plain_seconds_and_snaps_to_the_frame_grid(rig):
    rig.pick("C")
    f = rig.clip_tab.f_start
    with one_undo(rig):
        rig.type_in(f, "20.017")                         # 600.51 frames -> frame 601
    start = rig.clip("C").start
    assert start == pytest.approx(601 / 30, abs=1e-5)
    assert abs(start * 30 - round(start * 30)) < 1e-3           # on the frame grid (history keeps 6 decimals)


def test_committing_by_leaving_the_field_works_like_enter(rig):
    rig.pick("C")
    with one_undo(rig):
        rig.type_in(rig.clip_tab.f_start, "30", finish="blur")
    assert rig.clip("C").start == 30.0


def test_start_into_an_occupied_slot_is_refused_without_side_effects(rig):
    rig.pick("B")                                        # B 5-8 -> 0:00 would land on A (0-4)
    f = rig.clip_tab.f_start
    with untouched(rig):
        rig.type_in(f, "1")
    assert rig.status and "已經有片段" in rig.status[-1]
    assert f.text() == tc(5.0) and rig.clip("B").start == 5.0       # the field shows the real value again
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (5.0, 8.0, 0.0, 3.0), (10.0, 14.0, 5.0, 9.0)]


def test_start_field_never_overwrites_or_pushes_other_clips_on_its_track(rig):
    rig.pick("A")                                        # A is 4 s long: 8-12 would run into C (10-14)
    f = rig.clip_tab.f_start
    with untouched(rig):
        rig.type_in(f, "8")
    assert rig.status and "已經有片段" in rig.status[-1] and f.text() == tc(0.0)
    with one_undo(rig):
        rig.type_in(f, "20")                             # free space: only A moves
    assert rig.layout(rig.v1) == [(5.0, 8.0, 0.0, 3.0), (10.0, 14.0, 5.0, 9.0), (20.0, 24.0, 0.0, 4.0)]


@pytest.mark.parametrize("junk", ["abc", "", "1:2:3:4:5", "垃圾", "--3", "1.2.3"])
def test_garbage_in_a_time_field_is_rejected_and_reverted(rig, junk):
    rig.pick("C")
    for field in (rig.clip_tab.f_start, rig.clip_tab.f_dur, rig.clip_tab.f_in, rig.clip_tab.f_out):
        shown = field.text()
        with untouched(rig):
            rig.type_in(field, junk)
        assert field.text() == shown
    assert rig.panel.isEnabled()


@pytest.mark.parametrize("junk", ["inf", "-inf", "nan", "1e999", "-1e999"])
@pytest.mark.parametrize("name", ["f_start", "f_dur", "f_in", "f_out"])
def test_non_finite_numbers_in_a_time_field_are_rejected(rig, junk, name):
    """``nan`` must not be read as 0 (that would silently trim / move the clip) and ``1e999`` must not blow up."""
    rig.pick("C")
    f = getattr(rig.clip_tab, name)
    shown = f.text()
    with untouched(rig):
        rig.type_in(f, junk)
    assert f.text() == shown, "the field must show the real value again"


def test_duration_field_trims_the_tail_and_keeps_the_head(rig):
    rig.pick("C")
    f = rig.clip_tab.f_dur
    with one_undo(rig, "修改片段長度", ui=lambda: (f.text(), rig.clip_tab.f_out.text())):
        rig.type_in(f, "00:00:02:15")                    # 2.5 s
    c = rig.clip("C")
    assert (c.start, c.src_in, c.src_out) == (10.0, 5.0, 7.5)
    assert rig.clip_tab.f_out.text() == tc(7.5)


def test_duration_is_clamped_by_the_next_clip_the_source_and_the_minimum(rig):
    rig.pick("A")                                        # A 0-4 with B starting at 5
    with one_undo(rig):
        rig.type_in(rig.clip_tab.f_dur, "100")
    a = rig.clip("A")
    assert (a.start, a.src_out) == (0.0, 5.0)            # stopped at B, never overwrote it
    assert rig.layout(rig.v1)[1] == (5.0, 8.0, 0.0, 3.0)
    rig.pick("C")                                        # nothing after C: limited by the 20 s source
    with one_undo(rig):
        rig.type_in(rig.clip_tab.f_dur, "100")
    assert rig.clip("C").src_out == 20.0
    with one_undo(rig):
        rig.type_in(rig.clip_tab.f_dur, "0")
    assert rig.clip("C").duration == pytest.approx(ops.MIN_CLIP)
    assert not [i for i in validate_sequence(rig.st.project, rig.seq) if i.level == "error"]


def test_in_point_field_keeps_the_tail_in_place(rig):
    rig.pick("C")                                        # C: start 10, src 5-9, tail at 14
    f = rig.clip_tab.f_in
    with one_undo(rig, "修改入點", ui=lambda: (f.text(), rig.clip_tab.f_start.text())):
        rig.type_in(f, "00:00:07:00")
    c = rig.clip("C")
    assert (c.src_in, c.src_out) == (7.0, 9.0)
    assert (c.start, c.end) == (12.0, 14.0)


def test_in_point_is_limited_by_the_previous_clip_and_the_source_start(rig):
    rig.pick("C")
    with one_undo(rig):
        rig.type_in(rig.clip_tab.f_in, "0")              # would need start 5, but B ends at 8
    c = rig.clip("C")
    assert c.start == 8.0 and c.src_in == 3.0 and c.end == 14.0
    assert rig.clip_tab.f_in.text() == tc(3.0)
    with one_undo(rig):
        rig.type_in(rig.clip_tab.f_in, "100")            # past the out point: keeps the minimum length
    c = rig.clip("C")
    assert c.end == 14.0 and c.duration == pytest.approx(ops.MIN_CLIP)


def test_out_point_field_moves_the_tail(rig):
    rig.pick("C")
    f = rig.clip_tab.f_out
    with one_undo(rig, "修改出點", ui=lambda: (f.text(), rig.clip_tab.f_dur.text())):
        rig.type_in(f, "00:00:08:00")
    c = rig.clip("C")
    assert (c.start, c.src_in, c.src_out) == (10.0, 5.0, 8.0)
    with one_undo(rig):
        rig.type_in(f, "999")                            # clamped to the 20 s asset
    assert rig.clip("C").src_out == 20.0


def test_time_fields_respect_speed(rig):
    rig.seed("seed", "set_speed", rig.ids["C"], 2.0)     # src 5-9 at 2x -> 2 s on the timeline (10-12)
    rig.pick("C")
    t = rig.clip_tab
    assert t.f_dur.text() == tc(2.0) and t.f_in.text() == tc(5.0) and t.f_out.text() == tc(9.0)
    with one_undo(rig):
        rig.type_in(t.f_out, "00:00:07:00")              # source 7 -> 1 s on the timeline
    assert (rig.clip("C").src_out, rig.clip("C").duration) == (7.0, 1.0)
    with one_undo(rig):
        rig.type_in(t.f_in, "00:00:06:00")               # source 6 -> head moves 0.5 s
    c = rig.clip("C")
    assert (c.src_in, c.start, c.end) == (6.0, 10.5, 11.0)


def test_locked_track_refuses_time_edits(rig):
    rig.seed("lock", "set_track_props", rig.v1, locked=True)
    rig.pick("C")
    with untouched(rig):
        rig.type_in(rig.clip_tab.f_start, "20")
        rig.type_in(rig.clip_tab.f_dur, "1")
    assert any("已鎖定" in m for m in rig.status)
    assert rig.clip_tab.f_start.text() == tc(10.0)


def test_typing_in_a_time_field_is_not_clobbered_by_refreshes_and_undo_never_recommits(rig):
    rig.pick("C")
    f = rig.clip_tab.f_start
    rig.focus(f)
    f.selectAll()
    send_text(f, "00:00:21")
    # something else changes the clip while the user is typing
    rig.st.op("外部修改", "set_clip_props", rig.ids["C"], volume=0.8)
    rig.st.set_playhead(2.0)
    rig.settle()
    assert f.text() == "00:00:21", "text being typed must survive refreshes"
    assert rig.clip("C").start == 10.0
    QTest.keyClick(f, Qt.Key_Return)
    rig.settle()
    assert rig.clip("C").start == 21.0
    # an undo while the (unedited) field still has focus must update it and leaving it must not commit anything
    rig.focus(f)
    rig.st.undo()
    rig.st.undo()
    rig.settle()
    assert rig.clip("C").start == 10.0 and f.text() == tc(10.0)
    n0, content = rig.undos, rig.content()
    f.clearFocus()
    rig.settle()
    assert rig.undos == n0 and rig.content() == content


# =============================================================================================== 片段: playback
def test_speed_field_changes_speed_and_ripples_later_clips(rig):
    rig.pick("A")
    f = rig.clip_tab.f_speed
    with one_undo(rig, "修改速度", ui=lambda: (f.value(), rig.clip_tab.f_dur.text())):
        rig.type_in(f, "2")
    a = rig.clip("A")
    assert a.speed == 2.0 and a.duration == 2.0 and (a.src_in, a.src_out) == (0.0, 4.0)
    assert rig.clip("B").start == 3.0 and rig.clip("C").start == 8.0     # later clips followed
    assert rig.clip("D").start == 2.0                                      # other tracks stay
    assert rig.clip_tab.f_dur.text() == tc(2.0)


def test_speed_is_limited_by_the_box_range(rig):
    rig.pick("A")
    f = rig.clip_tab.f_speed
    with one_undo(rig):
        step_and_enter(rig, f, 10000)                     # arrow-key style stepping stops at the box limit
    assert rig.clip("A").speed == ops.MAX_SPEED == f.value()
    with one_undo(rig):
        step_and_enter(rig, f, -10000)
    assert rig.clip("A").speed == ops.MIN_SPEED == f.value()
    assert not [i for i in validate_sequence(rig.st.project, rig.seq) if i.level == "error"]
    # typing a number the box cannot take never produces an out-of-range speed
    for text in ("10", "0.01", "-3"):
        rig.type_in(f, text)
        assert ops.MIN_SPEED <= rig.clip("A").speed <= ops.MAX_SPEED


def test_arrow_keys_step_a_number_box_and_commit_once_on_enter(rig):
    rig.pick("A")
    f = rig.clip_tab.f_speed
    rig.focus(f)
    n0 = rig.undos
    for _ in range(3):
        QTest.keyClick(f, Qt.Key_Up)
    assert rig.undos == n0, "stepping alone is not a commit"
    QTest.keyClick(f, Qt.Key_Return)
    rig.settle()
    assert rig.undos == n0 + 1 and rig.clip("A").speed == pytest.approx(1.15)


def test_enter_on_an_unchanged_number_field_adds_no_undo_step(rig):
    rig.pick("A")
    t = rig.clip_tab
    with untouched(rig):
        for f in (t.f_speed, t.f_vol, t.f_fade_in, t.f_fade_out, t.f_zoom.box, t.f_cx.box, t.f_bright.box):
            rig.focus(f)
            f.selectAll()
            QTest.keyClick(f, Qt.Key_Return)
        rig.type_in(t.f_speed, "1")                       # typing the same value is no change either
        rig.type_in(t.f_fade_in, "0", finish="blur")
    assert rig.status == []


def test_fades_commit_clamp_to_the_clip_and_undo(rig):
    rig.pick("A")
    t = rig.clip_tab
    with one_undo(rig, "修改淡入", ui=lambda: t.f_fade_in.value()):
        rig.type_in(t.f_fade_in, "1.5")
    assert rig.clip("A").fade_in == 1.5
    with one_undo(rig, "修改淡出", ui=lambda: t.f_fade_out.value()):
        rig.type_in(t.f_fade_out, "1")
    assert (rig.clip("A").fade_in, rig.clip("A").fade_out) == (1.5, 1.0)
    with one_undo(rig):
        rig.type_in(t.f_fade_out, "30")                   # larger than what is left of the 4 s clip
    a = rig.clip("A")
    assert a.fade_out == pytest.approx(a.duration - a.fade_in) and a.fade_in == 1.5
    assert t.f_fade_out.value() == pytest.approx(2.5)     # the box shows what was really applied


def test_crossfade_field_overlaps_with_the_previous_clip(rig):
    t = rig.clip_tab
    rig.pick("A")                                         # first clip of its track: nothing to fade from
    assert not t.f_xfade.isEnabled() and "沒有片段" in t.f_xfade.toolTip()
    rig.pick("B")                                         # A 0-4, B 5-8: a 1 s gap
    assert t.f_xfade.isEnabled() and t.f_xfade.value() == 0.0
    with one_undo(rig, "修改交叉淡化", ui=lambda: (t.f_xfade.value(), t.f_start.text())):
        rig.type_in(t.f_xfade, "1")
    b = rig.clip("B")
    assert b.start == 3.0 and rig.clip("A").end - b.start == pytest.approx(1.0)
    assert rig.clip("C").start == 8.0                     # the rest of the track moved along
    assert t.f_xfade.value() == 1.0 and t.f_start.text() == tc(3.0)
    with one_undo(rig):
        rig.type_in(t.f_xfade, "99")                      # limited to half of the shorter clip (3 s) = 1.5 s
    assert rig.clip("A").end - rig.clip("B").start == pytest.approx(1.5)
    assert t.f_xfade.value() == pytest.approx(1.5)
    with one_undo(rig):
        rig.type_in(t.f_xfade, "0")                       # back to a plain cut
    assert rig.clip("B").start == rig.clip("A").end
    assert not [i for i in validate_sequence(rig.st.project, rig.seq) if i.level == "error"]


# =================================================================================================== 片段: audio
def test_volume_db_and_gain_conversion():
    assert gain_to_db(1.0) == 0.0 and db_to_gain(0.0) == 1.0
    assert gain_to_db(2.0) == pytest.approx(6.02, abs=0.01) and gain_to_db(0.5) == pytest.approx(-6.02, abs=0.01)
    assert db_to_gain(-6.0) == pytest.approx(0.5012, abs=1e-4) and db_to_gain(6.0) == pytest.approx(1.9953, abs=1e-4)
    assert gain_to_db(0.0) == insp.DB_FLOOR == -60.0 and db_to_gain(-60.0) == 0.0
    assert gain_to_db(0.0005) == -60.0                   # below the floor is "mute"
    for db in (-40.0, -12.5, -3.0, 0.0, 3.0, 9.5, 12.0):
        assert gain_to_db(db_to_gain(db)) == pytest.approx(db, abs=0.006)
    assert gain_to_db(4.0) == pytest.approx(12.04, abs=0.01)         # the op's maximum gain


def test_volume_field_is_in_db_and_stores_linear_gain(rig):
    rig.pick("A")
    f = rig.clip_tab.f_vol
    with one_undo(rig, "修改音量", ui=lambda: f.value()):
        rig.type_in(f, "-6")
    assert rig.clip("A").volume == pytest.approx(db_to_gain(-6.0), abs=1e-5)
    assert f.value() == -6.0
    with one_undo(rig):
        rig.type_in(f, "-60")                             # the floor means mute (gain 0)
    assert rig.clip("A").volume == 0.0
    assert f.value() == -60.0
    with one_undo(rig):
        step_and_enter(rig, f, 10000)                     # the box stops at +12 dB
    assert f.value() == 12.0
    assert rig.clip("A").volume == pytest.approx(db_to_gain(12.0), abs=1e-5) and rig.clip("A").volume <= 4.0


def test_volume_display_does_not_drift_when_nothing_is_typed(rig):
    """A gain that is not a round dB value (here +12.04 dB, the op's maximum) must survive focus-in / focus-out."""
    rig.seed("seed", "set_clip_props", rig.ids["A"], volume=4.0)
    rig.pick("A")
    f = rig.clip_tab.f_vol
    with untouched(rig):
        rig.focus(f)
        f.clearFocus()
        rig.type_in(f, "12.0")                            # what the box displays is no edit either
    assert rig.clip("A").volume == 4.0
    rig.seed("seed", "set_clip_props", rig.ids["B"], volume=1.2)
    rig.pick("B")
    with untouched(rig):
        rig.focus(f)
        QTest.keyClick(f, Qt.Key_Return)


def test_audio_group_and_edits_work_on_audio_track_clips(rig):
    rig.pick("M")
    t = rig.clip_tab
    with one_undo(rig):
        rig.type_in(t.f_vol, "-12")
    assert rig.clip("M").volume == pytest.approx(db_to_gain(-12.0), abs=1e-5)
    with one_undo(rig):
        rig.type_in(t.f_fade_out, "2")
    assert rig.clip("M").fade_out == 2.0
    with one_undo(rig):
        rig.type_in(t.f_dur, "6")                         # trim the music
    assert rig.clip("M").end == 6.0
    assert t.g_picture.isHidden()


# ================================================================================================ 片段: picture
def test_fit_combo_cover_contain_blur(rig):
    rig.pick("A")
    t = rig.clip_tab
    assert [t.f_fit.itemData(i) for i in range(t.f_fit.count())] == ["cover", "contain", "blur"]
    for fit in ("contain", "blur", "cover"):
        with one_undo(rig, "修改填滿方式", ui=lambda: t.f_fit.currentData()):
            choose(rig, t.f_fit, fit)
        assert rig.clip("A").transform.fit == fit
    with untouched(rig):
        choose(rig, t.f_fit, "cover")                     # choosing what is already set: nothing to undo
    assert t.f_fit.currentData() == "cover"


def test_programmatic_fit_changes_do_not_commit(rig):
    """Only a user ``activated`` commits: ``setCurrentIndex`` alone (what a refresh does) must not."""
    rig.pick("A")
    with untouched(rig):
        rig.clip_tab.f_fit.setCurrentIndex(2)
        rig.settle()
    assert rig.clip("A").transform.fit == "cover"
    rig.clip_tab.refresh()
    assert rig.clip_tab.f_fit.currentData() == "cover"


def test_zoom_box_commit_and_limits(rig):
    rig.pick("A")
    t = rig.clip_tab
    with one_undo(rig, "修改縮放", ui=lambda: (t.f_zoom.box.value(), t.f_zoom.slider.value())):
        rig.type_in(t.f_zoom.box, "1.5")
    assert rig.clip("A").transform.zoom == 1.5 and t.f_zoom.slider.value() == 150
    with one_undo(rig):
        rig.type_in(t.f_zoom.box, "6")                    # the box goes up to 8 while the slider stops at 4
    assert rig.clip("A").transform.zoom == 6.0
    assert t.f_zoom.box.value() == 6.0 and t.f_zoom.slider.value() == 400
    with one_undo(rig):
        step_and_enter(rig, t.f_zoom.box, 10000)
    assert rig.clip("A").transform.zoom == 8.0 and t.f_zoom.box.value() == 8.0


def test_slider_drag_commits_once_on_release_not_per_tick(rig):
    rig.pick("A")
    t = rig.clip_tab
    sl = t.f_zoom.slider
    with one_undo(rig, "修改縮放", ui=lambda: (t.f_zoom.box.value(), sl.value())):
        n0, before = rig.undos, rig.content()
        drag_slider(rig, sl, 60, release=False)
        assert sl.isSliderDown()
        assert rig.undos == n0 and rig.content() == before, "dragging must not touch the project"
        assert t.f_zoom.box.value() > 1.0, "the number box follows the handle while dragging"
        assert rig.clip("A").transform.zoom == 1.0
        QTest.qWait(450)                                  # longer than the keyboard-pause timer: still nothing
        assert rig.undos == n0 and rig.clip("A").transform.zoom == 1.0
        QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, handle_pt(sl))
        rig.settle()
    zoom = rig.clip("A").transform.zoom
    assert zoom > 1.0 and zoom == pytest.approx(t.f_zoom.box.value()) and sl.value() == round(zoom * 100)


def test_slider_release_stores_the_shown_value(rig):
    rig.pick("A")
    t = rig.clip_tab
    drag_slider(rig, t.f_bright.slider, 50)
    c = rig.clip("A")
    assert c.look.brightness == pytest.approx(t.f_bright.slider.value() / 100) and c.look.brightness > 0
    assert t.f_bright.box.value() == pytest.approx(c.look.brightness)
    assert rig.undos == 1


def test_refresh_during_a_drag_does_not_fight_the_handle(rig):
    rig.pick("A")
    t = rig.clip_tab
    sl = t.f_contrast.slider
    drag_slider(rig, sl, 40, release=False)
    held = sl.value()
    rig.st.op("外部修改", "set_clip_props", rig.ids["A"], label="x")   # triggers refresh() mid-drag
    rig.settle()
    assert sl.isSliderDown() and sl.value() == held, "the handle must stay where the user holds it"
    n = rig.undos
    QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, handle_pt(sl))
    rig.settle()
    assert rig.undos == n + 1 and rig.clip("A").look.contrast == pytest.approx(held / 100)


def test_keyboard_steps_on_a_slider_commit_once_after_a_pause(rig):
    rig.pick("A")
    t = rig.clip_tab
    sl = t.f_sat.slider
    rig.focus(sl)
    n0 = rig.undos
    for _ in range(3):
        QTest.keyClick(sl, Qt.Key_Right)
    assert rig.undos == n0, "steps within the pause must not commit"
    assert rig.wait_until(lambda: rig.undos > n0)
    rig.settle(450)
    assert rig.undos == n0 + 1, "three key steps are ONE undo step"
    assert rig.clip("A").look.saturation == pytest.approx(1.03)
    rig.st.undo()
    rig.settle()
    assert rig.clip("A").look.saturation == 1.0 and t.f_sat.slider.value() == 100


def test_clicking_a_slider_handle_without_moving_changes_nothing(rig):
    """A press + release on the handle is not an edit -- even when the stored value is not on the slider's grid."""
    rig.seed("seed", "set_clip_props", rig.ids["A"], cx=0.3333, zoom=1.234, brightness=0.123)
    rig.pick("A")
    t = rig.clip_tab
    with untouched(rig):
        for field in (t.f_cx, t.f_zoom, t.f_bright):
            p = handle_pt(field.slider)
            QTest.mousePress(field.slider, Qt.LeftButton, Qt.NoModifier, p)
            QTest.mouseRelease(field.slider, Qt.LeftButton, Qt.NoModifier, p)
    c = rig.clip("A")
    assert (c.transform.cx, c.transform.zoom, c.look.brightness) == (0.3333, 1.234, 0.123)


def test_a_zoom_beyond_the_slider_range_survives_a_click_on_the_handle(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], zoom=6.0)       # the box allows 8, the slider stops at 4
    rig.pick("A")
    sl = rig.clip_tab.f_zoom.slider
    with untouched(rig):
        p = handle_pt(sl)
        QTest.mousePress(sl, Qt.LeftButton, Qt.NoModifier, p)
        QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, p)
    assert rig.clip("A").transform.zoom == 6.0


def test_mouse_wheel_over_controls_scrolls_the_panel_instead_of_editing(rig):
    """The 片段 form is long: a wheel notch with the pointer over a slider / combo / number box must scroll the panel."""
    rig.panel.resize(420, 420)                           # short enough for the form to need a scroll bar
    rig.pick("A")
    t = rig.clip_tab
    bar = t.verticalScrollBar()
    assert bar.maximum() > 0
    controls = [t.f_bright.slider, t.f_contrast.slider, t.f_sat.slider, t.f_zoom.slider, t.f_cx.slider, t.f_cy.slider,
                t.f_fit, t.f_speed, t.f_vol, t.f_fade_in, t.f_zoom.box, t.f_cx.box]
    with untouched(rig):
        for w in controls:
            assert not w.hasFocus()
            bar.setValue(0)
            taker = wheel_over(rig, w, dy=-120)           # one notch down
            assert taker is not w, f"{type(w).__name__} swallowed the wheel notch"
            assert bar.value() > 0, f"{type(w).__name__}: the wheel never scrolled the panel"
        QTest.qWait(450)                                  # a swallowed notch on a slider would commit after 0.35 s
    c = rig.clip("A")
    assert (c.look.brightness, c.transform.zoom, c.transform.fit, c.speed) == (0.0, 1.0, "cover", 1.0)


def test_mouse_wheel_never_edits_any_unfocused_control_of_any_tab(rig):
    """Combos commit on ``activated`` and a wheel notch activates them: 尺寸 / 影格率 / 位置 / 字型 must be immune."""
    rig.pick("A")
    rig.st.select_cues([rig.seq.cues[0].id])
    guarded = [w for kind in (QSlider, QComboBox, QAbstractSpinBox) for w in rig.panel.findChildren(kind)]
    assert len(guarded) > 25
    with untouched(rig):
        for w in guarded:
            if w.hasFocus():
                continue
            for dy in (120, -120):
                wheel_over(rig, w, dy=dy)
        QTest.qWait(450)
    assert rig.status == []


def test_wheel_acts_on_a_focused_slider(rig):
    rig.pick("A")
    sl = rig.clip_tab.f_bright.slider
    rig.focus(sl)
    n0 = rig.undos
    wheel_over(rig, sl)
    assert rig.wait_until(lambda: rig.undos > n0)
    assert rig.clip("A").look.brightness > 0


def test_flip_checkbox(rig):
    rig.pick("A")
    t = rig.clip_tab
    with one_undo(rig, "水平翻轉", ui=lambda: t.f_flip.checkState()):
        t.f_flip.click()
        rig.settle()
    assert rig.clip("A").transform.flip_h is True and t.f_flip.isChecked()
    with one_undo(rig):
        t.f_flip.click()
    assert rig.clip("A").transform.flip_h is False


def test_look_sliders_and_boxes(rig):
    rig.pick("A")
    t = rig.clip_tab
    for field, attr, typed, expect in ((t.f_bright, "brightness", "0.4", 0.4), (t.f_contrast, "contrast", "1.8", 1.8),
                                       (t.f_sat, "saturation", "0", 0.0)):
        with one_undo(rig, ui=lambda f=field: (f.box.value(), f.slider.value())):
            rig.type_in(field.box, typed)
        assert getattr(rig.clip("A").look, attr) == expect
        assert field.slider.value() == round(expect * 100)
    with one_undo(rig):
        step_and_enter(rig, t.f_bright.box, -10000)       # the boxes stop at the op's limits
    assert rig.clip("A").look.brightness == -1.0 and t.f_bright.slider.value() == -100
    with one_undo(rig):
        step_and_enter(rig, t.f_contrast.box, 10000)
    assert rig.clip("A").look.contrast == 3.0


# ============================================================================================ 片段: reframing
def test_center_sliders_without_keyframes_set_a_static_center(rig):
    rig.pick("A")
    t = rig.clip_tab
    with one_undo(rig, "修改取景中心", ui=lambda: (t.f_cx.box.value(), t.f_cx.slider.value())):
        rig.type_in(t.f_cx.box, "0.25")
    assert rig.clip("A").transform.cx == 0.25 and rig.clip("A").transform.keys == []
    drag_slider(rig, t.f_cy.slider, -40)
    assert rig.clip("A").transform.cy < 0.5 and rig.clip("A").transform.keys == []
    assert rig.undos == 2
    assert "取景固定" in t.keys_label.text()


def test_add_key_button_stores_the_current_center_at_the_playhead(rig):
    rig.seed("seed", "set_clip_props", rig.ids["C"], cx=0.3, cy=0.6)
    rig.pick("C")                                          # C: 10-14
    t = rig.clip_tab
    rig.st.set_playhead(11.0)
    rig.settle()
    assert t.b_key_add.isEnabled() and not t.b_key_clear.isEnabled()
    with one_undo(rig, "加入取景關鍵格", ui=lambda: (t.b_key_clear.isEnabled(), t.keys_label.text())):
        t.b_key_add.click()
    assert rig.clip("C").transform.keys == [[1.0, 0.3, 0.6]]
    assert "1 個關鍵格" in t.keys_label.text() and t.b_key_clear.isEnabled()
    rig.st.set_playhead(13.0)
    rig.settle()
    with one_undo(rig):
        t.b_key_add.click()
    assert [k[0] for k in rig.clip("C").transform.keys] == [1.0, 3.0]
    assert "2 個關鍵格" in t.keys_label.text()
    with one_undo(rig):                                    # same time again replaces instead of duplicating
        rig.type_in(t.f_cx.box, "0.8")
    keys = rig.clip("C").transform.keys
    assert [k[0] for k in keys] == [1.0, 3.0] and keys[1][1] == 0.8 and keys[0][1] == 0.3
    assert rig.clip("C").transform.cx == 0.3, "with keyframes the slider edits the key, not the static center"


def test_center_sliders_with_keyframes_edit_the_key_under_the_playhead(rig):
    rig.pick("C")
    t = rig.clip_tab
    rig.st.set_playhead(10.0)
    rig.settle()
    t.b_key_add.click()
    rig.st.set_playhead(12.0)
    rig.settle()
    t.b_key_add.click()
    rig.st.history.clear()
    rig.st.set_playhead(11.0)                              # between the two keys: the interpolated center
    rig.settle()
    assert t.f_cx.box.value() == 0.5
    with one_undo(rig, "修改取景關鍵格"):
        rig.type_in(t.f_cy.box, "0.9")
    keys = rig.clip("C").transform.keys
    assert [k[0] for k in keys] == [0.0, 1.0, 2.0] and keys[1][2] == 0.9 and keys[0][2] == keys[2][2] == 0.5
    rig.st.set_playhead(12.0)
    rig.settle()
    assert t.f_cy.box.value() == 0.5, "the sliders follow the playhead along the keyframes"


def test_playhead_outside_the_clip_blocks_keyframe_edits(rig):
    rig.pick("C")
    t = rig.clip_tab
    rig.st.set_playhead(2.0)                               # nowhere near C (10-14)
    rig.settle()
    assert not t.b_key_add.isEnabled()
    with untouched(rig):
        t.b_key_add.click()
        t._add_key()                                       # even if called directly
    assert any("不在這個片段上" in m for m in rig.status)
    rig.st.set_playhead(11.0)
    rig.settle()
    t.b_key_add.click()
    rig.settle()
    rig.st.set_playhead(2.0)
    rig.settle()
    assert not t.f_cx.isEnabled() and not t.f_cy.isEnabled() and "播放頭不在這個片段上" in t.keys_label.text()
    with untouched(rig):
        t._set_center("cx", 0.9)                           # a late slider event is refused, too


def test_clear_keys_button(rig):
    rig.pick("C")
    t = rig.clip_tab
    for sec in (10.5, 12.0):
        rig.st.set_playhead(sec)
        rig.settle()
        t.b_key_add.click()
    rig.settle()
    rig.st.history.clear()
    assert len(rig.clip("C").transform.keys) == 2
    with one_undo(rig, "清除取景關鍵格", ui=lambda: (t.b_key_clear.isEnabled(), t.keys_label.text())):
        t.b_key_clear.click()
    assert rig.clip("C").transform.keys == [] and not t.b_key_clear.isEnabled()


def test_reframing_group_exists_for_every_fit_but_keys_follow_the_clip_not_the_selection(rig):
    rig.pick("A", "C")                                     # multi: no per-clip key buttons
    t = rig.clip_tab
    assert t.keys_row.isHidden() and t.keys_label.isHidden()
    rig.pick("A")
    assert not t.keys_row.isHidden() and not t.keys_label.isHidden()


# ============================================================================================ 片段: multi-select
def test_multi_selection_shows_only_shared_fields_and_marks_differences(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], volume=0.5, brightness=0.2, fit="blur", fade_in=0.5)
    rig.pick("A", "B")
    t = rig.clip_tab
    assert "已選取" in t.title.text() and "2" in t.title.text() and "—" in t.title.text()
    assert t.g_time.isHidden(), "start / length / in / out belong to ONE clip"
    assert t.f_xfade.isHidden() and t.keys_row.isHidden() and t.keys_label.isHidden()
    assert not t.f_speed.is_mixed() and t.f_speed.value() == 1.0                    # same everywhere: shown
    for box in (t.f_vol, t.f_fade_in, t.f_bright.box):                              # different: 「—」
        assert box.is_mixed() and "—" in box.lineEdit().text()
    assert t.f_fit.currentIndex() == -1                                             # mixed fit: the placeholder
    assert t.f_contrast.box.value() == 1.0 and not t.f_contrast.box.is_mixed()
    rig.pick("B", "C")                                                              # the fields follow the selection
    assert not t.f_vol.is_mixed() and not t.f_fade_in.is_mixed() and t.f_fit.currentData() == "cover"


def test_multi_edit_changes_only_the_edited_field_on_every_clip_as_one_undo(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], volume=0.5, brightness=0.2)
    rig.seed("seed", "set_clip_props", rig.ids["B"], volume=1.5, brightness=-0.3)
    rig.pick("A", "B", "C")
    t = rig.clip_tab
    before = {k: (rig.clip(k).volume, rig.clip(k).look.brightness) for k in "ABC"}
    with one_undo(rig, "修改淡入", ui=lambda: (t.f_fade_in.value(), t.f_fade_in.is_mixed())):
        rig.type_in(t.f_fade_in, "0.5")
    assert [rig.clip(k).fade_in for k in "ABC"] == [0.5, 0.5, 0.5]
    assert {k: (rig.clip(k).volume, rig.clip(k).look.brightness) for k in "ABC"} == before, \
        "fields the user did not touch must keep their differing values"
    assert t.f_vol.is_mixed() and t.f_bright.box.is_mixed()


def test_typing_into_a_mixed_field_overwrites_all_but_enter_alone_does_not(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], volume=0.5)
    rig.pick("A", "B")
    t = rig.clip_tab
    assert t.f_vol.is_mixed()
    with untouched(rig):
        rig.focus(t.f_vol)
        QTest.keyClick(t.f_vol, Qt.Key_Return)            # looking at it / tabbing through it is not an edit
        t.f_vol.clearFocus()
        rig.settle()
    assert t.f_vol.is_mixed() and "—" in t.f_vol.lineEdit().text()
    with one_undo(rig, "修改音量"):
        rig.type_in(t.f_vol, "-6")
    assert [rig.clip(k).volume for k in "AB"] == [pytest.approx(db_to_gain(-6.0), abs=1e-5)] * 2
    assert not t.f_vol.is_mixed() and t.f_vol.value() == -6.0


def test_arrow_key_on_a_mixed_field_starts_from_the_default(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], volume=0.5)
    rig.pick("A", "B")
    t = rig.clip_tab
    rig.focus(t.f_vol)
    QTest.keyClick(t.f_vol, Qt.Key_Up)
    QTest.keyClick(t.f_vol, Qt.Key_Return)
    rig.settle()
    assert [rig.clip(k).volume for k in "AB"] == [pytest.approx(db_to_gain(0.5), abs=1e-5)] * 2
    assert rig.undos == 1


def test_multi_fit_and_flip_with_mixed_values(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], fit="contain", flip_h=True)
    rig.pick("A", "B")
    t = rig.clip_tab
    assert t.f_fit.currentIndex() == -1
    assert t.f_flip.isTristate() and t.f_flip.checkState() == Qt.PartiallyChecked
    with one_undo(rig, "修改填滿方式", ui=lambda: t.f_fit.currentData()):
        choose(rig, t.f_fit, "blur")
    assert [rig.clip(k).transform.fit for k in "AB"] == ["blur", "blur"]
    with one_undo(rig, "水平翻轉", ui=lambda: t.f_flip.checkState()):
        t.f_flip.click()                                  # a click on 「mixed」 means: flip them all
        rig.settle()
    assert [rig.clip(k).transform.flip_h for k in "AB"] == [True, True]
    assert t.f_flip.checkState() == Qt.Checked and not t.f_flip.isTristate()


def test_a_mixed_flip_checkbox_is_visibly_different_from_checked_and_unchecked(rig):
    """The tri-state box must look mixed (the theme has no indicator for it): three states, three different pictures."""
    rig.seed("翻轉", "set_clip_props", rig.ids["C"], flip_h=True)
    t = rig.clip_tab
    looks = {}
    for state, keys in (("off", ("A",)), ("on", ("C",)), ("mixed", ("A", "C"))):
        rig.pick(*keys)
        looks[state] = t.f_flip.grab().toImage()
    assert t.f_flip.checkState() == Qt.PartiallyChecked
    assert looks["off"] != looks["on"] and looks["off"] != looks["mixed"] and looks["on"] != looks["mixed"]


def test_multi_slider_click_keeps_mixed_values_and_a_drag_unifies_them(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], brightness=0.4)
    rig.pick("A", "B")
    t = rig.clip_tab
    sl = t.f_bright.slider
    assert t.f_bright.box.is_mixed()
    with untouched(rig):                                  # press + release without moving: no overwrite
        p = handle_pt(sl)
        QTest.mousePress(sl, Qt.LeftButton, Qt.NoModifier, p)
        QTest.mouseRelease(sl, Qt.LeftButton, Qt.NoModifier, p)
    assert t.f_bright.box.is_mixed()
    with one_undo(rig, "修改亮度"):
        drag_slider(rig, sl, 40)
    values = {rig.clip(k).look.brightness for k in "AB"}
    assert len(values) == 1 and values.pop() > 0


def test_multi_center_sliders_are_hidden_when_any_clip_has_keyframes(rig):
    rig.seed("seed", "set_clip_props", rig.ids["A"], cx=0.2)
    rig.pick("A", "C")
    t = rig.clip_tab
    assert not t.f_cx.isHidden() and t.f_cx.box.is_mixed() and t.f_cy.box.value() == 0.5
    with one_undo(rig, "修改取景中心"):
        rig.type_in(t.f_cx.box, "0.7")
    assert [rig.clip(k).transform.cx for k in "AC"] == [0.7, 0.7]
    rig.pick("C")
    rig.st.set_playhead(11.0)
    rig.settle()
    t.b_key_add.click()
    rig.st.history.clear()
    rig.pick("A", "C")                                    # one clip is keyed: a static center would fight the keys
    assert t.f_cx.isHidden() and t.f_cy.isHidden()
    rig.pick("A", "B")
    assert not t.f_cx.isHidden() and not t.f_cy.isHidden()


def test_multi_speed_ripples_clip_by_clip_inside_one_undo(rig):
    rig.pick("A", "B")
    t = rig.clip_tab
    with one_undo(rig, "修改速度"):
        rig.type_in(t.f_speed, "2")
    assert rig.layout(rig.v1) == [(0.0, 2.0, 0.0, 4.0), (3.0, 4.5, 0.0, 3.0), (6.5, 10.5, 5.0, 9.0)]
    assert not [i for i in validate_sequence(rig.st.project, rig.seq) if i.level == "error"]


def test_multi_edit_across_tracks_applies_to_both(rig):
    rig.pick("A", "D", "M")                               # V1, V2 and an audio track
    t = rig.clip_tab
    assert t.g_picture.isHidden(), "the audio clip in the selection has no picture"
    with one_undo(rig):
        rig.type_in(t.f_fade_out, "0.5")
    assert [rig.clip(k).fade_out for k in "ADM"] == [0.5, 0.5, 0.5]


def test_multi_edit_is_all_or_nothing_when_one_clip_cannot_change(rig):
    """A locked track in the selection refuses its clip: nothing at all may be half-applied to the others."""
    rig.seed("lock", "set_track_props", rig.v2, locked=True)
    rig.pick("A", "D")
    t = rig.clip_tab
    with untouched(rig):
        rig.type_in(t.f_fade_in, "0.5")
    assert any("已鎖定" in m for m in rig.status)
    assert rig.clip("A").fade_in == 0.0 and rig.clip("D").fade_in == 0.0
    assert t.f_fade_in.value() == 0.0, "the box shows the real value again"
    rig.pick("A")                                        # without the locked clip it works
    with one_undo(rig):
        rig.type_in(t.f_fade_in, "0.5")


def test_selection_of_vanished_clips_is_dropped_after_undo(rig):
    rig.pick("B")
    rig.st.op("刪除", "remove_clips", [rig.ids["B"]])
    rig.settle()
    assert rig.st.selection == set() and not rig.clip_tab.empty.isHidden()
    rig.st.undo()
    rig.settle()
    rig.pick("B")
    assert rig.clip_tab.empty.isHidden() and rig.clip_tab.f_start.text() == tc(5.0)


# ================================================================================================ 字幕: helpers
class Calls(list):
    """Positional arguments (without the parent widget) of every call; ``kw`` has the keyword arguments."""

    def __init__(self) -> None:
        super().__init__()
        self.kw: list[dict] = []


def stub(monkeypatch, name: str, result=None) -> Calls:
    """Replace one dialog seam of the inspector; returns what it was called with."""
    calls = Calls()

    def seam(*a, **k):
        calls.append(a[1:])
        calls.kw.append(k)
        return result(*a, **k) if callable(result) else result

    monkeypatch.setattr(insp, name, seam)
    return calls


def cell_editor(r: Rig, row: int, col: int) -> QLineEdit:
    """Open the table's real in-place editor on a cell (what a double click does)."""
    table = r.cap.table
    table.setCurrentCell(row, col)
    table.editItem(table.item(row, col))
    r.settle(5)
    ed = table.findChild(QLineEdit)
    assert ed is not None, "the cell editor did not open"
    return ed


def edit_cell(r: Rig, row: int, col: int, text: str) -> None:
    ed = cell_editor(r, row, col)
    ed.selectAll()
    send_text(ed, text)
    QTest.keyClick(ed, Qt.Key_Return)
    r.settle()


def click_row(r: Rig, row: int, ctrl: bool = False) -> None:
    """Click a row of the cue table like the user (Ctrl adds to the selection)."""
    table = r.cap.table
    pos = table.visualRect(table.model().index(row, 1)).center()
    QTest.mouseClick(table.viewport(), Qt.LeftButton, Qt.ControlModifier if ctrl else Qt.NoModifier, pos)
    if ctrl:                                               # QTest leaves Ctrl "held": release it like the user would
        QTest.keyRelease(table, Qt.Key_Control)
    assert QApplication.keyboardModifiers() == Qt.NoModifier
    r.settle()


def table_rows(r: Rig) -> list[tuple[str, str, str]]:
    t = r.cap.table
    return [tuple(t.item(i, c).text() for c in range(3)) for i in range(t.rowCount())]


# ============================================================================================= 字幕: the table
def test_cue_table_lists_the_cues_in_time_order(rig):
    cap = rig.cap
    assert [cap.table.horizontalHeaderItem(i).text() for i in range(3)] == ["開始", "結束", "文字"]
    assert table_rows(rig) == [("0:01.000", "0:02.000", "第一句"), ("0:03.000", "0:04.500", "第二句"),
                               ("0:06.000", "0:07.000", "第三句")]
    ids = [cap.table.item(i, 0).data(Qt.UserRole) for i in range(3)]
    assert ids == [q.id for q in rig.seq.cues]
    # an out-of-order insert shows up where it belongs; cues added elsewhere appear without reopening
    rig.st.op("外部新增", "add_cue", 0.0, 0.5, "最前面")
    rig.settle()
    assert table_rows(rig)[0] == ("0:00.000", "0:00.500", "最前面") and cap.table.rowCount() == 4


def test_empty_sequence_has_an_empty_table_and_sane_buttons(empty_rig):
    cap = empty_rig.cap
    assert cap.table.rowCount() == 0
    assert cap.b_add.isEnabled() and cap.b_import.isEnabled() and cap.b_auto.isEnabled()
    for b in (cap.b_split, cap.b_merge, cap.b_delete, cap.b_shift, cap.b_export):
        assert not b.isEnabled()


def test_edit_cue_text_through_the_cell_editor_is_one_undo(rig):
    with one_undo(rig, "編輯字幕", ui=lambda: table_rows(rig)[1][2]):
        edit_cell(rig, 1, 2, "改過的文字，含標點！")
    assert rig.cues()[1] == (3.0, 4.5, "改過的文字，含標點！")
    assert table_rows(rig)[1] == ("0:03.000", "0:04.500", "改過的文字，含標點！")
    with untouched(rig):                                   # same text again: no step
        edit_cell(rig, 1, 2, "改過的文字，含標點！")


def test_edit_cue_times_accepts_clock_and_plain_seconds(rig):
    with one_undo(rig, "修改字幕時間", ui=lambda: table_rows(rig)[1][0]):
        edit_cell(rig, 1, 0, "0:03.500")
    assert rig.cues()[1][:2] == (3.5, 4.5)
    with one_undo(rig):
        edit_cell(rig, 1, 1, "5.25")                       # plain seconds, snapped to the 30 fps grid
    assert rig.cues()[1][:2] == (3.5, 5.267)
    with one_undo(rig):
        edit_cell(rig, 2, 1, "0:01:10.000")                # hours:minutes style
    assert rig.cues()[2][:2] == (6.0, 70.0)


def test_cue_time_edits_are_clamped_by_the_op_not_half_applied(rig):
    with one_undo(rig):
        edit_cell(rig, 0, 1, "0:00.100")                   # end before start: the minimum cue length applies
    assert rig.cues()[0] == (1.0, 1.2, "第一句")
    rig.st.undo()
    rig.settle()
    with one_undo(rig):
        edit_cell(rig, 0, 0, "0:09.000")                   # start after the end: the cue moves, keeping its length
    assert rig.cues()[-1] == (9.0, 10.0, "第一句")
    assert [q[2] for q in rig.cues()] == ["第二句", "第三句", "第一句"]       # and the table re-sorted


@pytest.mark.parametrize("junk", ["亂打", "", "1:2:3:4:5", "abc", "nan", "1e999"])
def test_bad_cue_time_text_is_refused_with_a_message_and_the_table_reverts(rig, junk):
    with untouched(rig):
        edit_cell(rig, 0, 0, junk)
        edit_cell(rig, 0, 1, junk)
    assert any("時間格式不正確" in m for m in rig.status)
    assert table_rows(rig)[0][:2] == ("0:01.000", "0:02.000")


def open_editors(r: Rig) -> list[QLineEdit]:
    return r.cap.table.viewport().findChildren(QLineEdit)


def test_tab_moves_the_editor_to_the_end_time_of_the_same_cue(rig):
    ed = cell_editor(rig, 0, 0)
    ed.selectAll()
    send_text(ed, "0:01.500")
    QTest.keyClick(ed, Qt.Key_Tab)
    rig.settle(50)
    assert rig.cues()[0] == (1.5, 2.0, "第一句") and rig.undos == 1
    editors = open_editors(rig)
    assert len(editors) == 1 and editors[0].text() == "0:02.000"
    editors[0].selectAll()
    send_text(editors[0], "0:02.500")
    QTest.keyClick(editors[0], Qt.Key_Return)
    rig.settle()
    assert rig.cues()[0] == (1.5, 2.5, "第一句") and rig.undos == 2


def test_tab_after_a_retiming_that_reorders_the_rows_keeps_editing_the_same_cue(rig):
    """The table is kept in time order.  Moving 「第一句」 behind the others and pressing Tab must open the END of
    「第一句」 (now the last row) -- not the end of whichever cue took over its old row."""
    ed = cell_editor(rig, 0, 0)
    ed.selectAll()
    send_text(ed, "0:09.000")
    QTest.keyClick(ed, Qt.Key_Tab)
    rig.settle(50)
    assert [r[2] for r in table_rows(rig)] == ["第二句", "第三句", "第一句"]
    editors = open_editors(rig)
    assert len(editors) == 1 and editors[0].text() == "0:10.000", "the editor must show the end of the moved cue"
    assert rig.cap.table.currentRow() == 2 and rig.st.cue_selection == {rig.seq.cues[2].id}
    editors[0].selectAll()
    send_text(editors[0], "0:12.000")
    QTest.keyClick(editors[0], Qt.Key_Return)
    rig.settle()
    assert rig.cues() == [(3.0, 4.5, "第二句"), (6.0, 7.0, "第三句"), (9.0, 12.0, "第一句")]
    assert rig.undos == 2 and open_editors(rig) == []


def test_cell_edit_does_not_rebuild_the_table_inside_the_views_own_commit(rig):
    """Rows keep their identity and order after an edit (a rebuilt table would have dropped the selection)."""
    edit_cell(rig, 0, 2, "新文字")                          # opening the editor selects its row, like a click
    assert rig.st.cue_selection == {rig.seq.cues[0].id}
    assert [i.row() for i in rig.cap.table.selectionModel().selectedRows()] == [0]
    assert rig.cues()[0][2] == "新文字" and rig.cap.table.findChild(QLineEdit) is None


def test_table_selection_and_project_selection_stay_in_sync_both_ways(rig):
    cap, st = rig.cap, rig.st
    click_row(rig, 1)
    assert st.cue_selection == {rig.seq.cues[1].id}
    click_row(rig, 2, ctrl=True)
    assert st.cue_selection == {rig.seq.cues[1].id, rig.seq.cues[2].id}
    assert cap.b_merge.isEnabled() and cap.b_delete.isEnabled()
    st.select_cues([rig.seq.cues[0].id])                  # the timeline selects a cue
    rig.settle()
    assert [i.row() for i in cap.table.selectionModel().selectedRows()] == [0]
    assert not cap.b_merge.isEnabled() and cap.b_delete.isEnabled()
    st.select_cues([])
    rig.settle()
    assert cap.table.selectionModel().selectedRows() == [] and not cap.b_delete.isEnabled()
    assert rig.undos == 0


# ============================================================================================== 字幕: the tools
def test_add_cue_at_playhead_selects_it_and_opens_the_editor(rig):
    rig.st.set_playhead(8.0)
    with one_undo(rig, "新增字幕"):
        rig.cap.b_add.click()
        rig.settle()
        assert rig.cues()[-1] == (8.0, 10.0, "新字幕")
        assert rig.st.cue_selection == {rig.seq.cues[-1].id}
        ed = rig.cap.table.findChild(QLineEdit)
        assert ed is not None and ed.text() == "新字幕", "the text cell is ready for typing"
        QTest.keyClick(ed, Qt.Key_Escape)                  # leave the editor without typing anything
        rig.settle()


def test_text_typed_into_a_just_added_cue_is_a_second_separate_step(rig):
    rig.st.set_playhead(8.0)
    rig.cap.b_add.click()
    rig.settle()
    ed = rig.cap.table.findChild(QLineEdit)
    ed.selectAll()
    send_text(ed, "剛新增的字幕")
    QTest.keyClick(ed, Qt.Key_Return)
    rig.settle()
    assert rig.cues()[-1] == (8.0, 10.0, "剛新增的字幕") and rig.undos == 2
    rig.st.undo()
    rig.settle()
    assert rig.cues()[-1] == (8.0, 10.0, "新字幕")


def test_add_cue_stops_at_the_next_cue_when_there_is_room(rig):
    rig.st.set_playhead(2.4)                               # next cue at 3.0 (0.6 s away)
    rig.cap.b_add.click()
    rig.settle()
    assert (2.4, 3.0) == tuple(rig.cues()[1][:2])
    rig.st.set_playhead(5.0)                               # 1.0 s before the cue at 6: a full 2 s would overlap it
    rig.cap.b_add.click()
    rig.settle()
    assert [c[:2] for c in rig.cues() if c[2] == "新字幕"][-1] == (5.0, 6.0)


def test_split_cue_at_the_playhead(rig):
    rig.st.set_playhead(3.75)                              # the middle of 「第二句」 (3.0 - 4.5)
    with one_undo(rig, "分割字幕"):
        rig.cap.b_split.click()
    # the playhead snaps to frame 113 (3.7667 s); the text splits in proportion to the time (2 of 3 characters)
    assert rig.cues() == [(1.0, 2.0, "第一句"), (3.0, 3.767, "第二"), (3.767, 4.5, "句"), (6.0, 7.0, "第三句")]


def test_split_prefers_the_selected_cue_among_overlapping_ones(rig):
    rig.seed("重疊字幕", "add_cue", 3.0, 4.5, "疊在一起的字幕")
    rig.st.set_playhead(3.75)
    twin = next(q for q in rig.seq.cues if q.text == "疊在一起的字幕")
    rig.st.select_cues([twin.id])
    rig.settle()
    with one_undo(rig, "分割字幕"):
        rig.cap.b_split.click()
    texts = [c[2] for c in rig.cues()]
    assert "第二句" in texts and "疊在一起的字幕" not in texts and len(texts) == 5
    assert {"疊在一起", "的字幕"} <= set(texts)


def test_split_is_refused_off_a_cue_or_too_close_to_its_edge(rig):
    with untouched(rig):
        rig.st.set_playhead(5.0)                           # between cues
        rig.cap.b_split.click()
        assert any("播放頭沒有落在任何字幕上" in m for m in rig.status)
        rig.st.set_playhead(3.033)                         # on a cue but 1 frame from its start
        rig.cap.b_split.click()
        assert any("太靠近" in m for m in rig.status)


def test_merge_selected_cues(rig):
    cap = rig.cap
    click_row(rig, 0)
    assert not cap.b_merge.isEnabled()
    click_row(rig, 1, ctrl=True)
    assert cap.b_merge.isEnabled()
    with one_undo(rig, "合併字幕"):
        cap.b_merge.click()
    assert rig.cues() == [(1.0, 4.5, "第一句第二句"), (6.0, 7.0, "第三句")]       # CJK text is joined without a space
    assert rig.st.cue_selection <= {q.id for q in rig.seq.cues}


def test_the_delete_key_inside_a_cell_editor_edits_the_text_and_never_removes_the_cue(rig):
    original = rig.cues()
    ed = cell_editor(rig, 1, 2)
    rig.focus(ed)
    ed.selectAll()
    QTest.keyClick(ed, Qt.Key_Delete)                      # the shortcut of the table must not see this key
    rig.settle()
    assert ed.text() == "" and rig.cap.table.rowCount() == 3 and rig.cues() == original and rig.undos == 0
    QTest.keyClick(ed, Qt.Key_Escape)                      # leaving with Escape drops the typed change
    rig.settle()
    assert rig.cues() == original and rig.undos == 0 and table_rows(rig)[1][2] == "第二句"


def test_delete_selected_cues_by_button_and_by_the_delete_key(rig):
    click_row(rig, 0)
    click_row(rig, 2, ctrl=True)
    with one_undo(rig, "刪除字幕"):
        rig.cap.b_delete.click()
    assert rig.cues() == [(3.0, 4.5, "第二句")]
    assert any("已刪除 2 則字幕" in m for m in rig.status)
    assert rig.st.cue_selection == set()
    click_row(rig, 0)
    rig.cap.table.setFocus()
    rig.settle(5)
    with one_undo(rig, "刪除字幕"):
        QTest.keyClick(rig.cap.table, Qt.Key_Delete)
    assert rig.cues() == []
    assert not rig.cap.b_delete.isEnabled() and not rig.cap.b_export.isEnabled()


def test_shift_all_cues(rig, monkeypatch):
    asked = stub(monkeypatch, "ask_number", 1.5)
    with one_undo(rig, "位移字幕"):
        rig.cap.b_shift.click()
    assert asked and asked[0][0] == "全部位移" and "秒" in asked[0][1]
    assert rig.cues() == [(2.5, 3.5, "第一句"), (4.5, 6.0, "第二句"), (7.5, 8.5, "第三句")]
    asked = stub(monkeypatch, "ask_number", -2.0)          # earlier; nothing may drop below zero
    with one_undo(rig):
        rig.cap.b_shift.click()
    assert rig.cues()[0][0] == 0.5 and all(c[0] >= 0 and c[1] > c[0] for c in rig.cues())


@pytest.mark.parametrize("answer", [None, 0.0])
def test_shift_cancelled_or_zero_changes_nothing(rig, monkeypatch, answer):
    stub(monkeypatch, "ask_number", answer)
    with untouched(rig):
        rig.cap.b_shift.click()


# ============================================================================================= 字幕: SRT in / out
SRT_TEXT = ("1\n00:00:00,500 --> 00:00:02,000\n大家好，歡迎來到我的頻道\n\n"
            "2\n00:00:02,500 --> 00:00:04,250\n今天要分享三個小技巧\n\n"
            "3\n00:00:05,000 --> 00:00:06,500\nHello, world!\n")


def write_srt(path: Path, eol: str = "\r\n", bom: bool = True) -> Path:
    """The same three cues as a file: optionally with a UTF-8 BOM and Windows line ends."""
    path.write_bytes((("\ufeff" if bom else "") + SRT_TEXT.replace("\n", eol)).encode("utf-8"))
    return path


def test_import_srt_with_bom_crlf_and_chinese_replaces_after_asking(rig, monkeypatch, tmp_path):
    path = write_srt(tmp_path / "字幕檔 測試.srt")                       # UTF-8 BOM + CRLF + CJK file name
    opened = stub(monkeypatch, "open_file", str(path))
    asked = stub(monkeypatch, "ask", True)
    with one_undo(rig, "匯入字幕"):
        rig.cap.b_import.click()
    assert opened and "字幕檔" in opened[0][1]
    assert asked and "3 則" in asked[0][1] and asked[0][2:] == ("取代", "取消")
    assert rig.cues() == [(0.5, 2.0, "大家好，歡迎來到我的頻道"), (2.5, 4.267, "今天要分享三個小技巧"),
                          (5.0, 6.5, "Hello, world!")]
    assert table_rows(rig)[0] == ("0:00.500", "0:02.000", "大家好，歡迎來到我的頻道")
    assert any("已匯入 3 則字幕" in m and "字幕檔 測試.srt" in m for m in rig.status)


def test_import_declined_keeps_the_existing_cues(rig, monkeypatch, tmp_path):
    stub(monkeypatch, "open_file", str(write_srt(tmp_path / "a.srt")))
    asked = stub(monkeypatch, "ask", False)
    with untouched(rig):
        rig.cap.b_import.click()
    assert asked and rig.cues()[0][2] == "第一句"


def test_import_into_an_empty_sequence_does_not_ask(empty_rig, monkeypatch, tmp_path):
    stub(monkeypatch, "open_file", str(write_srt(tmp_path / "a.srt", eol="\n", bom=False)))
    with one_undo(empty_rig, "匯入字幕"):
        empty_rig.cap.b_import.click()
    assert len(empty_rig.seq.cues) == 3 and empty_rig.cap.table.rowCount() == 3


def test_import_vtt_and_ass(rig, monkeypatch, tmp_path):
    vtt = tmp_path / "v.vtt"
    vtt.write_text("WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.500\r\n你好，世界\r\n\r\n00:00:03.000 --> 00:00:04.000\r\n第二行\r\n",
                   encoding="utf-8")
    ass = tmp_path / "a.ass"
    ass.write_text("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                   "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\b1}粗體字幕\\N第二行\n", encoding="utf-8")
    stub(monkeypatch, "ask", True)
    stub(monkeypatch, "open_file", str(vtt))
    rig.cap.b_import.click()
    rig.settle()
    assert rig.cues() == [(1.0, 2.5, "你好，世界"), (3.0, 4.0, "第二行")]
    stub(monkeypatch, "open_file", str(ass))
    rig.cap.b_import.click()
    rig.settle()
    assert rig.cues() == [(1.0, 2.0, "粗體字幕 第二行")]


def test_import_failures_are_reported_not_raised(rig, monkeypatch, tmp_path):
    told = stub(monkeypatch, "inform")
    stub(monkeypatch, "ask", True)
    empty = tmp_path / "empty.srt"
    empty.write_text("這不是字幕檔\n", encoding="utf-8")
    with untouched(rig):
        stub(monkeypatch, "open_file", "")                 # dialog cancelled
        rig.cap.b_import.click()
        assert told == []
        stub(monkeypatch, "open_file", str(empty))
        rig.cap.b_import.click()
        assert told and "沒有可用的字幕" in told[-1][1]
        stub(monkeypatch, "open_file", str(tmp_path / "missing.srt"))
        rig.cap.b_import.click()
        assert "無法讀取字幕檔" in told[-1][1] and told.kw[-1] == {"warning": True}


def test_export_srt_writes_utf8_and_reads_back_identically(rig, monkeypatch, tmp_path):
    target = tmp_path / "輸出資料夾" / "我的字幕"                                        # no extension, CJK folder
    saved = stub(monkeypatch, "save_file", str(target))
    rig.cap.b_export.click()
    rig.settle()
    out = tmp_path / "輸出資料夾" / "我的字幕.srt"
    assert out.is_file() and saved and saved[0][1] == "序列 1.srt"
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    assert raw.decode("utf-8") == ("1\n00:00:01,000 --> 00:00:02,000\n第一句\n\n2\n00:00:03,000 --> 00:00:04,500\n第二句\n\n"
                                   "3\n00:00:06,000 --> 00:00:07,000\n第三句\n")
    back = insp.load_subtitle_file(out)
    assert back == [(q.start, q.end, q.text) for q in rig.seq.cues]
    assert any("已匯出 3 則字幕" in m and "我的字幕.srt" in m for m in rig.status)
    assert rig.undos == 0, "exporting is not an edit"


def test_export_keeps_an_srt_extension_and_overwrites(rig, monkeypatch, tmp_path):
    out = tmp_path / "x.SRT"
    out.write_text("old", encoding="utf-8")
    stub(monkeypatch, "save_file", str(out))
    rig.cap.b_export.click()
    rig.settle()
    assert out.read_text(encoding="utf-8").startswith("1\n00:00:01,000") and not (tmp_path / "x.SRT.srt").exists()


def test_export_without_cues_or_without_a_path_does_nothing(rig, empty_rig, monkeypatch, tmp_path):
    saved = stub(monkeypatch, "save_file", "")
    assert not empty_rig.cap.b_export.isEnabled()
    empty_rig.cap._export()                                # even when called directly
    rig.cap.b_export.click()                               # dialog cancelled
    rig.settle()
    assert saved == [("匯出 SRT", "序列 1.srt", "SRT 字幕 (*.srt)")]
    assert list(tmp_path.iterdir()) == []


def test_export_write_failure_is_reported_not_raised(rig, monkeypatch, tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    stub(monkeypatch, "save_file", str(blocker / "sub" / "out.srt"))      # the parent "folder" is a file
    told = stub(monkeypatch, "inform")
    rig.cap.b_export.click()
    rig.settle()
    assert told and told[0][0] == "匯出 SRT" and "無法寫入檔案" in told[0][1]


# ============================================================================================== 字幕: the style
def style_dict(r: Rig) -> dict:
    return r.seq.caption_style.to_dict()


def only_changed(before: dict, after: dict, *fields: str) -> None:
    assert {k for k in before if before[k] != after[k]} == set(fields)


def test_style_controls_show_the_sequence_style(rig):
    cap = rig.cap
    assert cap.g_style.isEnabled()
    assert (cap.s_size.value(), cap.s_outline.value(), cap.s_margin.value(), cap.s_chars.value(),
            cap.s_box_opacity.value()) == (4.5, 0.6, 12.0, 16, 50.0)
    assert cap.s_bold.isChecked() and not cap.s_box.isChecked() and cap.s_pos.currentData() == "bottom"
    assert (cap.s_color.color(), cap.s_outline_color.color(), cap.s_box_color.color()) == ("#FFFFFF", "#000000", "#000000")
    assert cap.s_font.currentFont().family() == rig.seq.caption_style.font


# widget, typed text, CaptionStyle field, value stored in the project, value the box shows
STYLE_BOXES = [("s_size", "6", "size", 0.06, 6.0), ("s_outline", "0.3", "outline", 0.003, 0.3),
               ("s_margin", "20", "margin", 0.2, 20.0), ("s_chars", "12", "max_chars", 12, 12.0),
               ("s_box_opacity", "80", "box_opacity", 0.8, 80.0)]


@pytest.mark.parametrize("name,text,field,stored,shown", STYLE_BOXES)
def test_style_number_boxes_commit_a_typed_value_as_one_step(rig, name, text, field, stored, shown):
    box, before, cues = getattr(rig.cap, name), style_dict(rig), rig.cues()
    with one_undo(rig, "字幕樣式", ui=lambda: box.value()):
        rig.type_in(box, text)
    after = style_dict(rig)
    only_changed(before, after, field)
    assert after[field] == stored and type(after[field]) is type(before[field]) and box.value() == shown
    assert rig.cues() == cues


@pytest.mark.parametrize("name,field,lo,hi", [("s_size", "size", 0.01, 0.15), ("s_outline", "outline", 0.0, 0.02),
                                              ("s_margin", "margin", 0.0, 0.45), ("s_chars", "max_chars", 4, 60),
                                              ("s_box_opacity", "box_opacity", 0.0, 1.0)])
def test_style_number_boxes_stop_at_their_limits(rig, name, field, lo, hi):
    box = getattr(rig.cap, name)
    with one_undo(rig):
        step_and_enter(rig, box, 100000)
    assert style_dict(rig)[field] == hi
    with one_undo(rig):
        step_and_enter(rig, box, -100000)
    assert style_dict(rig)[field] == lo


def test_typing_in_a_style_box_and_leaving_without_a_change_adds_no_step(rig):
    cap = rig.cap
    with untouched(rig):
        for box in (cap.s_size, cap.s_outline, cap.s_margin, cap.s_chars, cap.s_box_opacity):
            rig.focus(box)
            box.selectAll()
            QTest.keyClick(box, Qt.Key_Return)
        rig.type_in(cap.s_size, "4.5")                    # the same value typed again
        rig.type_in(cap.s_chars, "16", finish="blur")


def test_bold_and_box_checkboxes_are_one_step_each(rig):
    cap = rig.cap
    assert cap.s_bold.isChecked() and not cap.s_box.isChecked()
    with one_undo(rig, "字幕樣式", ui=lambda: cap.s_bold.isChecked()):
        QTest.mouseClick(cap.s_bold, Qt.LeftButton)
    assert rig.seq.caption_style.bold is False and rig.seq.caption_style.box is False
    with one_undo(rig, "字幕樣式", ui=lambda: cap.s_box.isChecked()):
        QTest.mouseClick(cap.s_box, Qt.LeftButton)
    assert rig.seq.caption_style.box is True and rig.seq.caption_style.bold is False
    with one_undo(rig):
        QTest.mouseClick(cap.s_bold, Qt.LeftButton)
    assert rig.seq.caption_style.bold is True


COLOURS = [("s_color", "color", "#FFFFFF", "文字顏色"), ("s_outline_color", "outline_color", "#000000", "描邊顏色"),
           ("s_box_color", "box_color", "#000000", "底框顏色")]


@pytest.mark.parametrize("name,field,start,title", COLOURS)
def test_colour_buttons_commit_the_picked_colour(rig, monkeypatch, name, field, start, title):
    btn, before = getattr(rig.cap, name), style_dict(rig)
    asked = stub(monkeypatch, "pick_color", "#ff6600")    # lower case on purpose: the project stores #RRGGBB upper case
    with one_undo(rig, "字幕樣式", ui=lambda: btn.color()):
        QTest.mouseClick(btn, Qt.LeftButton)
    assert asked == [(start, title)]
    after = style_dict(rig)
    only_changed(before, after, field)
    assert after[field] == "#FF6600" and btn.color() == "#FF6600"


@pytest.mark.parametrize("answer", [None, "#FFFFFF", "#ffffff"])
def test_cancelled_or_unchanged_colour_picks_change_nothing(rig, monkeypatch, answer):
    asked = stub(monkeypatch, "pick_color", answer)
    with untouched(rig):
        QTest.mouseClick(rig.cap.s_color, Qt.LeftButton)
    assert len(asked) == 1 and rig.cap.s_color.color() == "#FFFFFF"


def test_position_combo_commits_each_choice_and_ignores_the_same_one(rig):
    cap = rig.cap
    for data in ("top", "center", "bottom"):
        with one_undo(rig, "字幕樣式", ui=lambda: cap.s_pos.currentData()):
            choose(rig, cap.s_pos, data)
        assert style_dict(rig)["position"] == data
    with untouched(rig):
        choose(rig, cap.s_pos, "bottom")                  # picking what is already there


def test_font_combo_stores_the_chosen_family(rig):
    cap = rig.cap
    families = [cap.s_font.itemText(i) for i in range(cap.s_font.count())]
    assert len(families) >= 3
    pick = next(f for f in families if f != cap.s_font.currentText() and f != style_dict(rig)["font"])
    i = cap.s_font.findText(pick)
    before = style_dict(rig)
    with one_undo(rig, "字幕樣式", ui=lambda: cap.s_font.currentFont().family()):
        cap.s_font.setCurrentIndex(i)
        cap.s_font.activated.emit(i)
        rig.settle()
    after = style_dict(rig)
    only_changed(before, after, "font")
    assert after["font"] == pick


def test_style_controls_follow_external_changes_and_undo_without_committing(rig):
    cap = rig.cap
    rig.st.op("外部樣式", "set_caption_style", size=0.08, bold=False, color="#FFCC00", position="top", margin=0.3,
              max_chars=10, box=True, box_color="#112233", box_opacity=0.25, outline=0.012, outline_color="#445566")
    rig.settle()
    assert (cap.s_size.value(), cap.s_margin.value(), cap.s_chars.value(), cap.s_box_opacity.value(),
            cap.s_outline.value()) == (8.0, 30.0, 10, 25.0, 1.2)
    assert not cap.s_bold.isChecked() and cap.s_box.isChecked() and cap.s_pos.currentData() == "top"
    assert (cap.s_color.color(), cap.s_box_color.color(), cap.s_outline_color.color()) == ("#FFCC00", "#112233", "#445566")
    assert rig.undos == 1
    rig.st.undo()
    rig.settle()
    assert cap.s_size.value() == 4.5 and cap.s_bold.isChecked() and cap.s_pos.currentData() == "bottom"
    assert cap.s_color.color() == "#FFFFFF" and not cap.s_box.isChecked()
    assert rig.undos == 0 and rig.st.history.can_redo


def test_the_style_belongs_to_one_sequence(rig):
    first = rig.seq.id
    rig.st.project_op("新增序列", "create_sequence", "第二個序列", 1080, 1920)
    second = rig.st.project.sequences[-1].id
    rig.st.history.clear()
    rig.st.set_sequence(second)
    rig.settle()
    cap = rig.cap
    assert cap.table.rowCount() == 0 and cap.s_pos.currentData() == "bottom"
    with one_undo(rig, "字幕樣式"):
        choose(rig, cap.s_pos, "top")
    with one_undo(rig, "字幕樣式"):
        rig.type_in(cap.s_size, "7")
    assert rig.st.project.sequence(second).caption_style.position == "top"
    assert rig.st.project.sequence(second).caption_style.size == 0.07
    assert rig.st.project.sequence(first).caption_style.position == "bottom"
    assert rig.st.project.sequence(first).caption_style.size == 0.045
    rig.st.set_sequence(first)
    rig.settle()
    assert cap.s_pos.currentData() == "bottom" and cap.s_size.value() == 4.5 and cap.table.rowCount() == 3
    rig.st.set_sequence(second)
    rig.settle()
    assert cap.s_pos.currentData() == "top"


# ================================================================================================ 字幕: 自動產生字幕
class FakeSpeech:
    """Stands in for ``generate_cues_for_sequence``: records what it was given, reports progress and (with ``hold``)
    waits in its worker thread until the test releases it or the user cancels."""

    def __init__(self) -> None:
        self.cached = True
        self.calls: list[dict] = []
        self.hold = False
        self.release = threading.Event()
        self.started = threading.Event()
        self.cancel_seen = False
        self.error: BaseException | None = None
        self.cues = [(0.5, 2.0, "辨識出來的第一句"), (2.5, 4.0, "辨識出來的第二句")]
        self.real_ids = False             # True: ids like the real recogniser's (IdGen("p"): pq0001, pq0002 ...)

    def __call__(self, proj, seq, progress=None, cancel=None, **kw):
        self.calls.append({"proj": proj, "seq": seq, "cues_then": [(q.start, q.end, q.text) for q in seq.cues], **kw})
        progress(0.4, "正在辨識第 1 段")
        self.started.set()
        if self.hold:
            while not self.release.wait(0.005):
                if cancel.is_set():
                    self.cancel_seen = True
                    raise Cancelled()
        if self.error is not None:
            raise self.error
        gen = IdGen("p") if self.real_ids else (lambda _kind: new_id("q_"))
        return [CaptionCue(gen("q"), s, e, t) for s, e, t in self.cues]


@pytest.fixture
def speech(monkeypatch):
    """Speech recognition installed, the model on disk and a controllable fake transcriber."""
    fake = FakeSpeech()
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (True, ""))
    monkeypatch.setattr(speech_mod, "model_cached", lambda size="small": fake.cached)
    monkeypatch.setattr(captions_mod, "generate_cues_for_sequence", fake)
    yield fake
    fake.release.set()                                    # never leave a worker thread waiting


def answer_asks(monkeypatch, **by_yes_label) -> Calls:
    """``ask`` answered by the label of its yes button, e.g. ``繼續=True``; an unexpected question fails the test."""
    def answer(*a, **k):
        yes = a[3] if len(a) > 3 else "確定"
        assert yes in by_yes_label, f"unexpected question: {a[1:3]}"
        return by_yes_label[yes]
    return stub(monkeypatch, "ask", answer)


def finish(r: Rig) -> None:
    wait_for_idle(r.app, [r.cap.runner], timeout=10)
    r.settle()


def start_held(r: Rig, speech: FakeSpeech) -> None:
    """Click 「自動產生字幕」 and wait until the (held) transcription is really running."""
    speech.hold = True
    r.cap.b_auto.click()
    assert speech.started.wait(5), "the transcription never started"
    assert r.wait_until(lambda: r.cap.progress_label.text() == "正在辨識第 1 段")


@pytest.mark.parametrize("reason,shown", [
    ("", "語音辨識目前無法使用。\n" + insp.INSTALL_HINT),
    ("缺少語音辨識套件", "缺少語音辨識套件\n" + insp.INSTALL_HINT),
    ("  前後有空白 \n", "前後有空白\n" + insp.INSTALL_HINT),
    ("已經有說明 pip install -r requirements-speech.txt", "已經有說明 pip install -r requirements-speech.txt")])
def test_auto_captions_without_speech_support_explains_how_to_install(rig, monkeypatch, reason, shown):
    monkeypatch.setattr(speech_mod, "speech_available", lambda: (False, reason))
    told = stub(monkeypatch, "inform")
    with untouched(rig):
        rig.cap.b_auto.click()
    assert told == [("自動產生字幕", shown)] and told.kw == [{}]
    assert "requirements-speech.txt" in told[0][1] and told[0][1].count("requirements-speech.txt") == 1
    assert rig.cap.runner is None and rig.cap.b_auto.isEnabled() and rig.cap.progress_row.isHidden()


@pytest.mark.skipif(importlib.util.find_spec("faster_whisper") is not None, reason="faster-whisper is installed here")
def test_auto_captions_in_an_environment_without_faster_whisper(rig, monkeypatch):
    """The unpatched check: nothing raises, the person gets the install instruction."""
    told = stub(monkeypatch, "inform")
    with untouched(rig):
        rig.cap.b_auto.click()
    assert len(told) == 1 and "requirements-speech.txt" in told[0][1] and rig.cap.runner is None


def test_auto_captions_when_the_speech_module_cannot_be_imported(rig, monkeypatch):
    monkeypatch.setitem(sys.modules, "autocut.analysis.speech", None)        # makes `import` raise ImportError
    told = stub(monkeypatch, "inform")
    with untouched(rig):
        rig.cap.b_auto.click()
    assert told == [("自動產生字幕", "語音辨識模組無法載入。\n" + insp.INSTALL_HINT)] and rig.cap.runner is None


def test_auto_captions_without_the_caption_module_is_reported(rig, monkeypatch, speech):
    monkeypatch.setitem(sys.modules, "autocut.autoedit.captions", None)
    told = stub(monkeypatch, "inform")
    answer_asks(monkeypatch, 繼續=True)
    with untouched(rig):
        rig.cap.b_auto.click()
    assert told and "尚未安裝完整" in told[0][1] and told.kw == [{"warning": True}] and rig.cap.runner is None


def test_a_missing_model_is_only_downloaded_after_the_person_agrees(rig, monkeypatch, speech):
    speech.cached = False
    asked = answer_asks(monkeypatch, 下載並繼續=False)
    with untouched(rig):
        rig.cap.b_auto.click()
    assert len(asked) == 1 and asked[0][0] == "下載語音模型" and "small" in asked[0][1] and "不會上傳" in asked[0][1]
    assert asked[0][2:] == ("下載並繼續", "取消")
    assert speech.calls == [] and rig.cap.runner is None and "已取消自動產生字幕" in rig.status
    asked = answer_asks(monkeypatch, 下載並繼續=True, 繼續=True)               # now: agree to the download
    rig.cap.b_auto.click()
    finish(rig)
    assert [a[0] for a in asked] == ["下載語音模型", "自動產生字幕"] and "目前已有 3 則字幕" in asked[1][1]
    assert speech.calls[0]["allow_download"] is True


def test_auto_captions_replace_the_existing_cues_in_one_undo_step(rig, monkeypatch, speech):
    asked = answer_asks(monkeypatch, 繼續=True)
    original = rig.cues()
    before_edit, n0 = rig.content(), rig.undos
    rig.cap.b_auto.click()
    finish(rig)
    assert len(asked) == 1 and asked[0][2:] == ("繼續", "取消")
    assert rig.cues() == [(0.5, 2.0, "辨識出來的第一句"), (2.5, 4.0, "辨識出來的第二句")]
    assert rig.undos == n0 + 1 and rig.st.history.undo_label == "自動字幕"
    assert table_rows(rig)[0] == ("0:00.500", "0:02.000", "辨識出來的第一句")
    assert any("已產生 2 則字幕" in m for m in rig.status)
    call = speech.calls[0]
    assert {k: call[k] for k in ("language", "traditional", "model_size", "allow_download")} == {
        "language": "zh", "traditional": True, "model_size": "small", "allow_download": False}
    assert call["proj"] is not rig.st.project and call["seq"] is not rig.seq and call["seq"].id == rig.seq.id, \
        "the worker must get a snapshot, never the live project"
    assert call["cues_then"] == [(q[0], q[1], q[2]) for q in original]
    assert rig.cap.progress_row.isHidden() and rig.cap.b_auto.isEnabled() and rig.cap.b_import.isEnabled()
    rig.st.undo()
    rig.settle()
    assert rig.content() == before_edit and rig.cues() == original


def test_declining_the_replacement_starts_nothing(rig, monkeypatch, speech):
    asked = answer_asks(monkeypatch, 繼續=False)
    with untouched(rig):
        rig.cap.b_auto.click()
    assert len(asked) == 1 and speech.calls == [] and rig.cap.runner is None


def test_auto_captions_for_an_empty_sequence_do_not_ask(empty_rig, speech):
    empty_rig.cap.b_auto.click()
    finish(empty_rig)
    assert empty_rig.cues() == [(0.5, 2.0, "辨識出來的第一句"), (2.5, 4.0, "辨識出來的第二句")]
    assert empty_rig.undos == 1 and empty_rig.st.history.undo_label == "自動字幕"


def test_progress_and_the_buttons_while_transcribing(empty_rig, speech):
    r, cap = empty_rig, empty_rig.cap
    start_held(r, speech)
    assert not cap.progress_row.isHidden() and cap.progress.value() == 400
    assert cap.b_cancel.isEnabled() and cap.b_cancel.text() == "取消"
    assert not cap.b_auto.isEnabled() and not cap.b_import.isEnabled()
    calls = len(speech.calls)
    cap._auto()                                           # a second start (no dialog, no second worker)
    r.settle()
    assert len(speech.calls) == calls and r.undos == 0
    speech.release.set()
    finish(r)
    assert cap.progress_row.isHidden() and cap.b_auto.isEnabled() and cap.b_import.isEnabled()
    assert len(r.seq.cues) == 2


def test_cancelling_the_transcription_changes_nothing(empty_rig, speech):
    r, cap = empty_rig, empty_rig.cap
    start_held(r, speech)
    with untouched(r):
        cap.b_cancel.click()
        assert not cap.b_cancel.isEnabled() and cap.b_cancel.text() == "正在取消…"
        finish(r)
    assert speech.cancel_seen and "已取消自動產生字幕" in r.status
    assert cap.progress_row.isHidden() and cap.b_auto.isEnabled() and r.seq.cues == []


def test_a_failing_transcription_is_reported_not_raised(empty_rig, monkeypatch, speech):
    told = stub(monkeypatch, "inform")
    speech.error = RuntimeError("boom")
    with untouched(empty_rig):
        empty_rig.cap.b_auto.click()
        finish(empty_rig)
    assert told == [("自動產生字幕", "發生未預期的錯誤（RuntimeError）：boom")] and told.kw == [{"warning": True}]
    assert empty_rig.cap.b_auto.isEnabled() and empty_rig.cap.progress_row.isHidden()


def test_a_speech_failure_gets_the_install_hint_once(empty_rig, monkeypatch, speech):
    told = stub(monkeypatch, "inform")
    speech.error = speech_mod.SpeechUnavailable("語音辨識模型載入失敗")
    empty_rig.cap.b_auto.click()
    finish(empty_rig)
    speech.error = speech_mod.SpeechUnavailable(speech_mod.INSTALL_HINT)
    empty_rig.cap.b_auto.click()
    finish(empty_rig)
    assert told[0][1] == "語音辨識模型載入失敗\n" + insp.INSTALL_HINT
    assert told[1][1] == speech_mod.INSTALL_HINT and told[1][1].count("requirements-speech.txt") == 1
    assert empty_rig.undos == 0


def test_silence_leaves_the_captions_alone(rig, monkeypatch, speech):
    told = stub(monkeypatch, "inform")
    answer_asks(monkeypatch, 繼續=True)
    speech.cues = []
    with untouched(rig):
        rig.cap.b_auto.click()
        finish(rig)
    assert told == [("自動產生字幕", "沒有辨識到任何語音，字幕沒有變更。")] and told.kw == [{}]


@pytest.mark.parametrize("apply", [True, False])
def test_an_edit_during_transcription_asks_before_applying(rig, monkeypatch, speech, apply):
    answer_asks(monkeypatch, 繼續=True)
    start_held(rig, speech)
    rig.st.op("外部修改", "add_cue", 9.0, 9.5, "中途新增")
    asked = answer_asks(monkeypatch, 仍要套用=apply)
    speech.release.set()
    finish(rig)
    assert len(asked) == 1 and "辨識期間專案有被修改過" in asked[0][1] and asked[0][2:] == ("仍要套用", "放棄")
    if apply:
        assert rig.cues() == [(0.5, 2.0, "辨識出來的第一句"), (2.5, 4.0, "辨識出來的第二句")]
        assert rig.undos == 2 and rig.st.history.undo_label == "自動字幕"
    else:
        assert "中途新增" in [c[2] for c in rig.cues()] and rig.undos == 1


def test_replacing_the_project_during_transcription_discards_the_result(rig, monkeypatch, speech):
    told = stub(monkeypatch, "inform")
    answer_asks(monkeypatch, 繼續=True)
    start_held(rig, speech)
    fresh = make_project()
    rig.st.replace_project(fresh)
    speech.release.set()
    finish(rig)
    assert told == [("自動產生字幕", "辨識期間專案或序列已經換掉，這次的結果沒有套用。")]
    assert rig.st.project is fresh and fresh.sequence().cues == [] and rig.undos == 0


def test_deleting_the_sequence_during_transcription_discards_the_result(empty_rig, monkeypatch, speech):
    r = empty_rig
    told = stub(monkeypatch, "inform")
    r.st.project_op("新增序列", "create_sequence", "另一個", activate=False)
    r.st.history.clear()
    first, other = r.seq.id, r.st.project.sequences[-1].id
    start_held(r, speech)
    r.st.set_sequence(other)
    r.st.project_op("刪除序列", "delete_sequence", first)
    r.st.history.clear()
    speech.release.set()
    finish(r)
    assert told == [("自動產生字幕", "辨識期間專案或序列已經換掉，這次的結果沒有套用。")]
    assert r.st.project.sequence(other).cues == []


def test_the_result_goes_to_the_sequence_it_was_started_on(empty_rig, speech):
    r = empty_rig
    r.st.project_op("新增序列", "create_sequence", "另一個", activate=False)
    r.st.history.clear()
    first, other = r.seq.id, r.st.project.sequences[-1].id
    start_held(r, speech)
    r.st.set_sequence(other)                              # the person looks at another sequence meanwhile
    r.settle()
    speech.release.set()
    finish(r)
    assert [q.text for q in r.st.project.sequence(first).cues] == ["辨識出來的第一句", "辨識出來的第二句"]
    assert r.st.project.sequence(other).cues == [] and r.cap.table.rowCount() == 0
    assert r.undos == 1


# ===================================================================================================== 序列
def pick_index(r: Rig, combo, index: int) -> None:
    """Choosing an entry of a combo like the user: set the index, then ``activated`` (user-only signal)."""
    combo.setCurrentIndex(index)
    combo.activated.emit(index)
    r.settle()


def size_entry(combo, size) -> int:
    return next(i for i in range(combo.count()) if combo.itemData(i) == size)       # (w, h) tuples: no findData


def fps_entry(combo, fps: float) -> int:
    return next(i for i in range(combo.count()) if combo.itemData(i) == fps)


def no_problems(r: Rig) -> list:
    return validate_sequence(r.st.project, r.seq)


def test_sequence_tab_shows_the_active_sequence(rig):
    t = rig.seq_tab
    assert t.isEnabled() and t.f_name.text() == "序列 1"
    assert t.f_size.currentText() == "16:9 橫式（1920×1080）" and t.f_size.currentData() == (1920, 1080)
    assert (t.f_w.value(), t.f_h.value()) == (1920, 1080) and t.f_fps.currentText() == "30 fps"
    assert t.f_bg.color() == "#000000"
    assert t.info.text() == "長度 0:17　·　7 個片段　·　3 則字幕\n2 條視訊軌 / 2 條音訊軌　·　比例 16:9"
    assert [t.f_size.itemText(i) for i in range(t.f_size.count())] == [
        "16:9 橫式（1920×1080）", "9:16 直式（1080×1920）", "1:1 方形（1080×1080）", "4:5 直式（1080×1350）", "自訂"]
    assert [t.f_fps.itemData(i) for i in range(t.f_fps.count())] == [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]


def test_renaming_the_sequence_is_one_step_and_trims_the_name(rig):
    t = rig.seq_tab
    with one_undo(rig, "重新命名序列", ui=lambda: t.f_name.text()):
        rig.type_in(t.f_name, "  我的短影音 ")
    assert rig.seq.name == "我的短影音" and t.f_name.text() == "我的短影音"
    with one_undo(rig, "重新命名序列"):
        rig.type_in(t.f_name, "Reel 2", finish="blur")          # leaving the field commits as well
    assert rig.seq.name == "Reel 2"


@pytest.mark.parametrize("typed", ["", "   ", "Reel 2"])
def test_rename_to_nothing_or_to_the_same_name_changes_nothing(rig, typed):
    t = rig.seq_tab
    rig.seed("改名", "rename_sequence", "Reel 2")
    with untouched(rig):
        rig.type_in(t.f_name, typed)
    assert rig.seq.name == "Reel 2" and t.f_name.text() == "Reel 2"


def test_sequence_names_are_limited_to_80_characters(rig):
    t = rig.seq_tab
    rig.type_in(t.f_name, "字" * 100)
    assert rig.seq.name == "字" * 80


def test_a_name_being_typed_survives_refreshes(rig):
    t = rig.seq_tab
    rig.focus(t.f_name)
    t.f_name.selectAll()
    send_text(t.f_name, "半途的名字")
    rig.st.op("外部", "add_cue", 9.0, 9.5, "別的修改")                       # an unrelated edit refreshes every tab
    rig.settle()
    assert t.f_name.text() == "半途的名字" and rig.seq.name == "序列 1"
    QTest.keyClick(t.f_name, Qt.Key_Return)
    rig.settle()
    assert rig.seq.name == "半途的名字"


@pytest.mark.parametrize("w,h,aspect", [(1080, 1920, "9:16"), (1080, 1080, "1:1"), (1080, 1350, "4:5")])
def test_size_presets_change_only_the_frame_size(rig, w, h, aspect):
    t = rig.seq_tab
    layout = {tid: rig.layout(tid) for tid in (rig.v1, rig.v2, rig.a1)}
    cues = rig.cues()
    with one_undo(rig, "變更序列尺寸", ui=lambda: (t.f_size.currentIndex(), t.f_w.value(), t.f_h.value())):
        pick_index(rig, t.f_size, size_entry(t.f_size, (w, h)))
    assert (rig.seq.width, rig.seq.height) == (w, h) and (t.f_w.value(), t.f_h.value()) == (w, h)
    assert t.f_size.currentData() == (w, h) and f"比例 {aspect}" in t.info.text()
    assert no_problems(rig) == []
    assert {tid: rig.layout(tid) for tid in (rig.v1, rig.v2, rig.a1)} == layout and rig.cues() == cues
    assert rig.seq.fps == 30.0 and rig.seq.background == "#000000"


def test_picking_the_current_size_or_fps_adds_no_step(rig):
    t = rig.seq_tab
    with untouched(rig):
        pick_index(rig, t.f_size, size_entry(t.f_size, (1920, 1080)))
        pick_index(rig, t.f_fps, fps_entry(t.f_fps, 30.0))


def test_custom_size_focuses_the_width_box_and_changes_nothing_yet(rig):
    t = rig.seq_tab
    with untouched(rig):
        pick_index(rig, t.f_size, t.f_size.count() - 1)
        assert t.f_w.hasFocus()
    assert t.f_size.currentData() is None and t.f_size.currentText() == "自訂"
    with one_undo(rig, "變更序列尺寸", ui=lambda: (t.f_w.value(), t.f_h.value())):
        QTest.keyClick(t.f_w, Qt.Key_Return)                 # nothing typed: no commit ...
        send_text(t.f_w, "1280")                              # ... but typing the width is one
        QTest.keyClick(t.f_w, Qt.Key_Return)
    assert (rig.seq.width, rig.seq.height) == (1280, 1080) and t.f_size.currentText() == "自訂"
    with one_undo(rig, "變更序列尺寸"):
        rig.type_in(t.f_h, "720")
    assert (rig.seq.width, rig.seq.height) == (1280, 720) and "比例 16:9" in t.info.text() and no_problems(rig) == []
    rig.type_in(t.f_w, "1080")
    rig.type_in(t.f_h, "1920")
    assert (rig.seq.width, rig.seq.height) == (1080, 1920)
    assert t.f_size.currentData() == (1080, 1920), "typing a preset's size selects that preset again"


def test_odd_sizes_are_rounded_down_to_even_and_never_invalid(rig):
    t = rig.seq_tab
    with one_undo(rig, "變更序列尺寸", ui=lambda: (t.f_w.value(), t.f_h.value())):
        rig.type_in(t.f_w, "1281")
    assert rig.seq.width == 1280 and t.f_w.value() == 1280 and no_problems(rig) == []
    with untouched(rig):
        rig.type_in(t.f_w, "1281")                           # rounds to what it already is
    assert rig.seq.width == 1280 and t.f_w.value() == 1280
    with one_undo(rig):
        rig.type_in(t.f_h, "721")
    assert rig.seq.height == 720 and no_problems(rig) == []


def test_size_boxes_stop_at_their_limits(rig):
    t = rig.seq_tab
    for box, attr in ((t.f_w, "width"), (t.f_h, "height")):
        with one_undo(rig):
            step_and_enter(rig, box, 100000)
        assert getattr(rig.seq, attr) == 8192 == box.value()
        with one_undo(rig):
            step_and_enter(rig, box, -100000)
        assert getattr(rig.seq, attr) == 16 == box.value()
    assert no_problems(rig) == []


def test_changing_the_size_keeps_every_clip_valid_on_a_busy_timeline(rig):
    rig.seed("交叉淡化", "set_crossfade", rig.ids["B"], 1.0)          # B overlaps A by a second
    assert rig.clip("A").end - rig.clip("B").start == pytest.approx(1.0)
    t = rig.seq_tab
    assert no_problems(rig) == []
    for size in ((1080, 1920), (1080, 1080), (1080, 1350), (1920, 1080)):
        pick_index(rig, t.f_size, size_entry(t.f_size, size))
        assert (rig.seq.width, rig.seq.height) == size and no_problems(rig) == []
    rig.type_in(t.f_w, "2560")
    rig.type_in(t.f_h, "1440")
    assert (rig.seq.width, rig.seq.height) == (2560, 1440) and no_problems(rig) == []


def test_frame_rate_change_snaps_clips_and_cues_in_one_undo_step(rig):
    rig.seed("格線外的片段", "place_clip", "a2", rig.v2, 18.0333, 0.0, 1.0, "overwrite")
    rig.seed("格線外的字幕", "add_cue", 8.0333, 8.5, "格線外")
    t = rig.seq_tab
    off_grid = next(c for c in rig.seq.find_track(rig.v2).clips if c.start > 18.0)
    cid = off_grid.id
    with one_undo(rig, "變更影格率", ui=lambda: (t.f_fps.currentData(), t.f_fps.currentText())):
        pick_index(rig, t.f_fps, fps_entry(t.f_fps, 24.0))
    assert rig.seq.fps == 24.0 and t.f_fps.currentText() == "24 fps"
    for _tr, c in rig.seq.all_clips():
        assert c.start * 24 == pytest.approx(round(c.start * 24), abs=1e-4), "clip starts sit on the new frame grid"
    for q in rig.seq.cues:
        assert q.start * 24 == pytest.approx(round(q.start * 24), abs=1e-4)
        assert q.end * 24 == pytest.approx(round(q.end * 24), abs=1e-4)
    moved = rig.seq.find_clip(cid)[1]
    assert moved.start == pytest.approx(18.0417, abs=1e-3)
    assert no_problems(rig) == []
    rig.pick(cid)                                             # the clip tab counts frames at the new rate
    assert rig.clip_tab.f_start.text() == fmt_timecode(moved.start, 24.0) and rig.clip_tab.f_start.text().endswith(":01")


def test_an_unusual_frame_rate_is_listed_and_selected(rig):
    rig.seed("12 fps", "set_sequence_format", fps=12.0)
    t = rig.seq_tab
    assert t.f_fps.currentText() == "12 fps" and t.f_fps.currentData() == 12.0 and t.f_fps.count() == 9
    with one_undo(rig, "變更影格率"):
        pick_index(rig, t.f_fps, fps_entry(t.f_fps, 25.0))
    assert rig.seq.fps == 25.0 and t.f_fps.count() == 8 and t.f_fps.currentText() == "25 fps"
    rig.st.undo()
    rig.settle()
    assert t.f_fps.currentText() == "12 fps" and t.f_fps.count() == 9


def test_background_colour_is_one_undo_step(rig, monkeypatch):
    t = rig.seq_tab
    asked = stub(monkeypatch, "pick_color", "#3a6ea5")
    with one_undo(rig, "變更背景色", ui=lambda: t.f_bg.color()):
        QTest.mouseClick(t.f_bg, Qt.LeftButton)
    assert asked == [("#000000", "背景色")]
    assert rig.seq.background == "#3A6EA5" and t.f_bg.color() == "#3A6EA5"
    assert no_problems(rig) == []
    assert Project.from_dict(rig.st.project.to_dict()).sequence().background == "#3A6EA5", "it is saved with the project"


@pytest.mark.parametrize("answer", [None, "#000000", "#000"])
def test_cancelled_or_unchanged_background_picks_change_nothing(rig, monkeypatch, answer):
    stub(monkeypatch, "pick_color", answer)
    with untouched(rig):
        QTest.mouseClick(rig.seq_tab.f_bg, Qt.LeftButton)


def test_sequence_tab_follows_external_changes_undo_and_sequence_switches(rig):
    t = rig.seq_tab
    rig.st.op("外部", "rename_sequence", "外部改的名")
    rig.st.op("外部", "set_sequence_format", width=1080, height=1920, fps=25.0)
    rig.settle()
    assert t.f_name.text() == "外部改的名" and t.f_size.currentData() == (1080, 1920) and t.f_fps.currentText() == "25 fps"
    rig.st.undo()
    rig.st.undo()
    rig.settle()
    assert t.f_name.text() == "序列 1" and t.f_size.currentData() == (1920, 1080) and t.f_fps.currentText() == "30 fps"
    rig.st.history.clear()
    rig.st.project_op("新增序列", "create_sequence", "直式短片", 1080, 1920, 24.0)
    other = rig.st.project.sequences[-1].id
    rig.st.set_sequence(other)
    rig.settle()
    assert t.f_name.text() == "直式短片" and t.f_size.currentData() == (1080, 1920) and t.f_fps.currentText() == "24 fps"
    assert "0 個片段" in t.info.text() and "比例 9:16" in t.info.text()
    assert rig.undos == 1                                    # only the explicit "create" step, no echo from the tab


def test_sequence_tab_is_disabled_without_a_sequence(rig):
    rig.st.replace_project(Project.new("空", with_sequence=False))
    rig.settle()
    assert not rig.seq_tab.isEnabled() and not rig.cap.g_style.isEnabled()
    for b in (rig.cap.b_add, rig.cap.b_split, rig.cap.b_merge, rig.cap.b_delete, rig.cap.b_shift, rig.cap.b_import,
              rig.cap.b_export, rig.cap.b_auto):
        assert not b.isEnabled()
    assert rig.cap.table.rowCount() == 0 and rig.checks.issues.count() == 0
    assert not rig.clip_tab.empty.isHidden()
    rig.pick("nothing")
    rig.settle()
    rig.st.replace_project(make_project())                    # and everything comes back with the next project
    rig.settle()
    assert rig.seq_tab.isEnabled() and rig.cap.g_style.isEnabled() and rig.cap.b_add.isEnabled()
    assert rig.seq_tab.f_name.text() == "序列 1" and rig.cap.table.rowCount() == 0


# ===================================================================================================== 檢查
def add_run(r: Rig, *, kind: str = "shorts", n: int = 0, warnings=(), when: str = "2026-10-01T10:30:15+08:00",
            link: bool = True, qa: str | None = None) -> tuple[planmod.EditPlan, str]:
    """Record an auto-edit run the way ``autoedit.apply`` does; ``link`` ties the active sequence to it."""
    plan = planmod.EditPlan(kind=kind, brief={"n": n}, notes=[f"計畫 {n}"])
    receipt = planmod.AuditReceipt(plan.sha256(), r.st.project.revision, "", True,
                                   [planmod.Finding("規則", planmod.WARN, w) for w in warnings])
    sid, made = r.st.seq_id, {}

    def fn(p: Project) -> None:
        run = planmod.record_run(p, plan, receipt, [sid])
        run["applied_at"] = when
        if link:
            p.sequence(sid).meta["ai"] = {"run_id": run["run_id"]}
        made["id"] = run["run_id"]

    r.st.do("seed", fn)
    if qa:
        r.st.record_technical_qa(made["id"], qa)
    r.st.history.clear()
    r.settle()
    return plan, made["id"]


def run_of(r: Rig, run_id: str) -> dict:
    return planmod.find_run(r.st.project, run_id)


def walk(item, depth: int = 0):
    yield "  " * depth + item.text(0)
    for i in range(item.childCount()):
        yield from walk(item.child(i), depth + 1)


def tree_lines(tab: ChecksTab) -> list[str]:
    return [line for i in range(tab.runs.topLevelItemCount()) for line in walk(tab.runs.topLevelItem(i))]


def issue_texts(tab: ChecksTab) -> list[str]:
    return [tab.issues.item(i).text() for i in range(tab.issues.count())]


def corrupt(r: Rig, fn, keep_undo: bool = False) -> None:
    """Put the project into a state the editing ops never produce (a hand-edited file, an older version)."""
    r.st.do("壞掉的專案", fn)
    if not keep_undo:
        r.st.history.clear()
    r.settle()


def click_item(r: Rig, lst, row: int, double: bool = False) -> None:
    """A click, or a double click the way the window system delivers it: press, release, then the double-click event
    (a lone double-click event is taken for a plain press by the item view)."""
    pos = lst.visualItemRect(lst.item(row)).center()
    QTest.mouseClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    if double:
        QTest.mouseDClick(lst.viewport(), Qt.LeftButton, Qt.NoModifier, pos)
    r.settle()


def test_checks_tab_empty_states(rig):
    t = rig.checks
    assert tree_lines(t) == ["這個專案還沒有自動剪輯紀錄。"]
    assert issue_texts(t) == ["沒有發現結構問題。"] and t.issues.item(0).data(Qt.UserRole) is None
    box = t.review
    assert box.qa_label.text() == "這個序列不是自動剪輯產生的，沒有審片紀錄。" and box.verdict_label.text() == ""
    assert not any(w.isEnabled() for w in (box.btn_ok, box.btn_fix, box.btn_no, box.notes))
    before = (rig.st.selection, rig.st.playhead)
    click_item(rig, t.issues, 0, double=True)                # the 「沒有問題」 line is not a link
    click_item(rig, t.issues, 0)
    assert (rig.st.selection, rig.st.playhead) == before and rig.undos == 0


def test_run_history_shows_hash_warnings_and_review_state(rig):
    plan, run_id = add_run(rig, warnings=["第 3 段太短", "音樂太大聲"], qa="pass")
    t = rig.checks
    assert tree_lines(t) == [
        "短影音 · 2026-10-01 10:30 · 1 個序列（目前序列）",
        f"  計畫雜湊：{plan.sha256()[:12]}",
        "  技術檢查：通過（只代表檔案技術面，不代表好看）",
        "  尚未審片（機器不會代替你判斷好不好看）",
        "  審核警告（2）", "    第 3 段太短", "    音樂太大聲"]
    top = t.runs.topLevelItem(0)
    assert top.isExpanded() and top.font(0).bold() and len(plan.sha256()[:12]) == 12
    assert t.review.qa_label.text() == "技術檢查：通過（只代表檔案技術面，不代表好看）"
    assert t.review.verdict_label.text() == "尚未審片（機器不會代替你判斷好不好看）"
    assert all(w.isEnabled() for w in (t.review.btn_ok, t.review.btn_fix, t.review.btn_no, t.review.notes))
    assert run_of(rig, run_id)["human_review"] == "pending"


def test_a_run_of_two_sequences_shows_the_own_check_and_review_of_each(rig):
    """A run can hold several sequences: a check or a review of one is not the other's, and an edit expires only its own."""
    _plan, run_id = add_run(rig)
    first = rig.st.seq_id
    second = rig.st.project_op("新增序列", "create_sequence", "第二支", 1080, 1920, 30.0, activate=False).id

    def link(p: Project) -> None:
        planmod.find_run(p, run_id)["sequence_ids"].append(second)
        p.sequence(second).meta["ai"] = {"run_id": run_id}

    rig.st.do("seed", link)
    rig.st.set_sequence(first)
    t = rig.checks
    rig.settle()
    lines = tree_lines(t)
    assert lines[0].endswith("2 個序列") or "2 個序列" in lines[0]
    assert lines[2] == "  技術檢查：尚未執行（2 個序列）"
    assert lines[3] == "  尚未審片（機器不會代替你判斷好不好看）（2 個序列）"
    rig.st.record_technical_qa(run_id, "pass", "ok", seq_id=first)
    rig.st.human_review(run_id, "approved", "", seq_id=first)
    rig.settle()
    lines = tree_lines(t)
    assert lines[2] == "  技術檢查：尚未檢查 1、通過 1" and lines[3] == "  審片：尚未審片 1、通過 1"
    assert t.review.qa_label.text() == QA_TEXT["pass"] and t.review.verdict_label.text() == VERDICT_TEXT["approved"]
    rig.st.set_sequence(second)                                   # the other sequence: nothing was done for it
    rig.settle()
    assert t.review.qa_label.text() == QA_TEXT["not_run"]
    assert t.review.verdict_label.text() == "尚未審片（機器不會代替你判斷好不好看）"
    rig.st.set_sequence(first)
    rig.st.op("加標記", "add_marker", 1.0, "後來加的")                 # edit the first sequence only
    rig.settle()
    lines = tree_lines(t)
    assert lines[2] == "  技術檢查：尚未檢查 1、已過期 1" and lines[3] == "  審片：尚未審片 1、已過期 1"
    assert t.review.qa_label.text() == QA_TEXT["stale"] and t.review.verdict_label.text() == VERDICT_TEXT["stale"]
    assert run_of(rig, run_id)["human_review"] == "approved"      # the run-level mirror still says what happened last ...
    assert planmod.review_state(run_of(rig, run_id), rig.st.sequence())[0] == planmod.STALE    # ... but it counts for nothing now


def test_run_history_without_warnings_and_each_qa_state(rig):
    _plan, run_id = add_run(rig, kind="assemble")
    t = rig.checks
    lines = tree_lines(t)
    assert lines[0] == "長片 · 2026-10-01 10:30 · 1 個序列（目前序列）"
    assert lines[2] == "  技術檢查：尚未執行" and lines[4] == "  審核警告：沒有" and len(lines) == 5
    for status, text in (("warn", "技術檢查：有警告"), ("fail", "技術檢查：未通過"),
                         ("pass", "技術檢查：通過（只代表檔案技術面，不代表好看）"), ("not_run", "技術檢查：尚未執行")):
        rig.st.record_technical_qa(run_id, status)
        rig.settle()
        assert tree_lines(t)[2] == "  " + text and t.review.qa_label.text() == text
        assert t.review.verdict_label.text() == "尚未審片（機器不會代替你判斷好不好看）", "a machine check never is a review"
        assert run_of(rig, run_id)["human_review"] == "pending"


def test_runs_are_listed_newest_first_and_capped(rig):
    ids = []
    for n in range(7):
        ids.append(add_run(rig, n=n, when=f"2026-10-0{n + 1}T09:00:00+08:00", link=(n == 6))[1])
    t = rig.checks
    tops = [t.runs.topLevelItem(i).text(0) for i in range(t.runs.topLevelItemCount())]
    assert tops == [f"短影音 · 2026-10-0{n} 09:00 · 1 個序列" + ("（目前序列）" if n == 7 else "") for n in (7, 6, 5, 4, 3)]
    assert t.runs.topLevelItem(0).isExpanded() and not t.runs.topLevelItem(1).isExpanded()
    assert planmod.find_run(rig.st.project, ids[0]) is not None, "older runs are only hidden, not deleted"


def test_the_current_sequence_run_is_marked_and_expanded_even_when_old(rig):
    _p, old_id = add_run(rig, n=1, when="2026-09-01T09:00:00+08:00")
    rig.st.project_op("新增序列", "create_sequence", "另一個", activate=False)
    mine = rig.st.seq_id
    rig.st.set_sequence(rig.st.project.sequences[-1].id)
    add_run(rig, n=2, when="2026-09-02T09:00:00+08:00")
    rig.st.set_sequence(mine)
    rig.settle()
    t = rig.checks
    texts = [t.runs.topLevelItem(i).text(0) for i in range(t.runs.topLevelItemCount())]
    assert texts == ["短影音 · 2026-09-02 09:00 · 1 個序列", "短影音 · 2026-09-01 09:00 · 1 個序列（目前序列）"]
    assert t.runs.topLevelItem(1).isExpanded() and t.review.btn_ok.isEnabled()
    rig.st.set_sequence(rig.st.project.sequences[-1].id)
    rig.settle()
    assert [t.runs.topLevelItem(i).text(0).endswith("（目前序列）") for i in range(2)] == [True, False]
    rig.st.set_sequence(mine)
    assert run_of(rig, old_id)["human_review"] == "pending"


def test_a_sequence_without_a_run_has_no_review_box_even_when_other_runs_exist(rig):
    add_run(rig, link=False)
    t = rig.checks
    assert t.runs.topLevelItem(0).text(0) == "短影音 · 2026-10-01 10:30 · 1 個序列"
    assert t.review.qa_label.text() == "這個序列不是自動剪輯產生的，沒有審片紀錄。"
    assert not t.review.btn_ok.isEnabled()
    with untouched(rig):
        t.review._click("approved")                           # even a stray call changes nothing


@pytest.mark.parametrize("button,verdict,box_text,tree_text", [
    ("btn_ok", "approved", "已通過（你審片後標記）", "你已審片：通過"),
    ("btn_fix", "needs_changes", "需要修改（你審片後標記）", "你已審片：需要修改"),
    ("btn_no", "rejected", "已退回（你審片後標記）", "你已審片：退回")])
def test_human_review_is_set_by_a_button_click_as_one_undo_step(rig, monkeypatch, button, verdict, box_text, tree_text):
    _p, run_id = add_run(rig, qa="pass")
    t = rig.checks
    seen = []
    real = planmod.set_human_review
    monkeypatch.setattr(planmod, "set_human_review", lambda *a, **k: (seen.append(a[1:]), real(*a, **k))[1])
    rig.focus(t.review.notes)
    send_text(t.review.notes, "  節奏很順，結尾太急  ")
    with one_undo(rig, "人工審片", ui=lambda: (t.review.verdict_label.text(), t.runs.topLevelItem(0).child(2).text(0))):
        QTest.mouseClick(getattr(t.review, button), Qt.LeftButton)
    run = run_of(rig, run_id)
    assert seen == [(run_id, verdict, "節奏很順，結尾太急")]
    assert run["human_review"] == verdict and run["human_review_notes"] == "節奏很順，結尾太急"
    assert [o["verdict"] for o in run["outcomes"]] == [verdict] and run["human_review_at"]
    assert t.review.verdict_label.text() == box_text and t.runs.topLevelItem(0).child(2).text(0) == tree_text
    assert t.review.qa_label.text() == "技術檢查：通過（只代表檔案技術面，不代表好看）"


def test_a_review_click_does_not_leave_the_keyboard_on_the_verdict_button(rig):
    """After a click on 通過 / 需修改 / 退回 the keyboard is not on that button: Space or Enter cannot press it again."""
    _p, run_id = add_run(rig, qa="pass")
    t = rig.checks
    rig.focus(t.review.notes)
    for name in ("btn_ok", "btn_fix", "btn_no"):
        QTest.mouseClick(getattr(t.review, name), Qt.LeftButton)
        rig.settle()
        assert not any(getattr(t.review, b).hasFocus() for b in ("btn_ok", "btn_fix", "btn_no")), name
        n = rig.undos
        for k in (Qt.Key_Space, Qt.Key_Return):
            w = QApplication.focusWidget()
            if w is not None:
                QTest.keyClick(w, k)
        rig.settle()
        assert rig.undos == n, f"{name}: a key press after the click must not review again"


def test_inspector_controls_take_the_keyboard_by_tab_only(rig):
    """Buttons, check boxes, sliders and non-editable combo boxes of every tab: a mouse click does not give them the
    keyboard (so the arrow keys, S and Delete keep editing the timeline); Tab still reaches them."""
    tab_stops = 0
    for i in range(rig.panel.count()):
        rig.panel.setCurrentIndex(i)
        rig.pick("A")
        for w in rig.panel.widget(i).findChildren(QWidget):
            if isinstance(w, (QAbstractButton, QAbstractSlider, QComboBox)) and not (isinstance(w, QComboBox) and w.isEditable()):
                assert not (w.focusPolicy() & Qt.ClickFocus), f"{rig.panel.tabText(i)}: {type(w).__name__} {w.objectName()}"
                tab_stops += bool(w.focusPolicy() & Qt.TabFocus)
    assert tab_stops >= 10, tab_stops


def test_the_widgets_that_are_used_on_their_own_take_the_keyboard_by_tab_only(qapp):
    """The review box and the focus-guarded slider / combo set this up themselves (they are also built outside a panel)."""
    box = HumanReviewBox(AppState(make_project()))
    for w in (box.btn_ok, box.btn_fix, box.btn_no, insp.FocusSlider(Qt.Horizontal), insp.FocusCombo()):
        assert not (w.focusPolicy() & Qt.ClickFocus) and (w.focusPolicy() & Qt.TabFocus), type(w).__name__


def test_the_machine_never_sets_the_review(rig, monkeypatch, speech):
    """Everything the program does by itself -- technical QA, refreshes, undo / redo, selection, switching tabs and
    sequences, auto captions -- leaves the verdict on 「尚未審片」; only the person's click changes it."""
    calls = []
    real_state, real_plan = AppState.human_review, planmod.set_human_review
    monkeypatch.setattr(AppState, "human_review", lambda self, *a, **k: (calls.append(("state", a)), real_state(self, *a, **k))[1])
    monkeypatch.setattr(planmod, "set_human_review", lambda *a, **k: (calls.append(("plan", a[1:])), real_plan(*a, **k))[1])
    _p, run_id = add_run(rig)
    t = rig.checks
    pending = "尚未審片（機器不會代替你判斷好不好看）"
    for status in ("pass", "warn", "fail", "pass"):
        rig.st.record_technical_qa(run_id, status)
        rig.settle()
    rig.st.undo()
    rig.st.redo()
    rig.pick("A", "B")
    rig.st.set_playhead(3.0)
    for i in range(rig.panel.count()):
        rig.show_tab(rig.panel.widget(i))
    for tab in (rig.clip_tab, rig.cap, rig.seq_tab, t, t.review):
        tab.refresh()
    rig.st.project_op("新增序列", "create_sequence", "旁邊的序列", activate=False)
    rig.st.set_sequence(rig.st.project.sequences[-1].id)
    rig.st.set_sequence(next(s.id for s in rig.st.project.sequences if s.id != rig.st.seq_id))
    answer_asks(monkeypatch, 繼續=True)
    rig.cap.b_auto.click()                                    # a machine-made edit of the very sequence under review
    finish(rig)
    assert len(rig.seq.cues) == 2
    assert calls == [] and run_of(rig, run_id)["human_review"] == "pending" and run_of(rig, run_id)["outcomes"] == []
    assert t.review.verdict_label.text() == pending
    assert run_of(rig, run_id)["technical_qa"] == "pass"
    QTest.mouseClick(t.review.btn_ok, Qt.LeftButton)          # only now
    rig.settle()
    assert [c[0] for c in calls] == ["state", "plan"] and run_of(rig, run_id)["human_review"] == "approved"


def test_review_can_be_changed_again_and_undone_step_by_step(rig):
    _p, run_id = add_run(rig)
    t = rig.checks
    QTest.mouseClick(t.review.btn_ok, Qt.LeftButton)
    rig.settle()
    send_text(t.review.notes, "再看一次")
    QTest.mouseClick(t.review.btn_fix, Qt.LeftButton)
    rig.settle()
    run = run_of(rig, run_id)
    assert run["human_review"] == "needs_changes" and [o["verdict"] for o in run["outcomes"]] == ["approved", "needs_changes"]
    assert rig.undos == 2
    rig.st.undo()
    rig.settle()
    assert t.review.verdict_label.text() == "已通過（你審片後標記）" and run_of(rig, run_id)["human_review"] == "approved"
    rig.st.undo()
    rig.settle()
    assert t.review.verdict_label.text() == "尚未審片（機器不會代替你判斷好不好看）"
    assert run_of(rig, run_id)["human_review"] == "pending"


def test_the_verdict_colour_is_green_only_after_the_person_approves(rig):
    from autocut.gui import theme
    add_run(rig, qa="pass")
    t = rig.checks
    assert theme.WARN in t.review.verdict_label.styleSheet() and theme.OK not in t.review.verdict_label.styleSheet()
    QTest.mouseClick(t.review.btn_fix, Qt.LeftButton)
    rig.settle()
    assert theme.OK not in t.review.verdict_label.styleSheet()
    QTest.mouseClick(t.review.btn_ok, Qt.LeftButton)
    rig.settle()
    assert theme.OK in t.review.verdict_label.styleSheet()
    assert t.runs.topLevelItem(0).child(2).foreground(0).color().name() == QColor(theme.OK).name()


def test_review_notes_typed_for_one_run_are_not_carried_over_to_another(rig):
    """The notes box belongs to the run on screen: switching sequences must not attach A's notes to B's verdict."""
    _p, run_a = add_run(rig, n=1)
    first = rig.st.seq_id
    rig.st.project_op("新增序列", "create_sequence", "第二個", activate=False)
    rig.st.set_sequence(rig.st.project.sequences[-1].id)
    _p, run_b = add_run(rig, n=2)
    rig.st.set_sequence(first)
    rig.settle()
    t = rig.checks
    rig.focus(t.review.notes)
    send_text(t.review.notes, "只屬於第一個序列的備註")
    rig.st.set_sequence(rig.st.project.sequences[-1].id)
    rig.settle()
    assert t.review.notes.text() == "", "the box starts empty for the other run"
    QTest.mouseClick(t.review.btn_no, Qt.LeftButton)
    rig.settle()
    assert run_of(rig, run_b)["human_review_notes"] == "" and run_of(rig, run_a)["human_review"] == "pending"


def test_typed_review_notes_survive_unrelated_edits_of_the_same_run(rig):
    add_run(rig)
    t = rig.checks
    send_text(t.review.notes, "還沒按按鈕")
    rig.st.op("別的修改", "add_cue", 9.0, 9.5, "x")
    rig.pick("A")
    rig.settle()
    assert t.review.notes.text() == "還沒按按鈕"


# ---- the structure check list
def test_issue_list_describes_each_problem_and_jumps_to_it(rig):
    t = rig.checks
    cue = rig.seq.cues[1]
    corrupt(rig, lambda p: (setattr(p.sequence().find_clip(rig.ids["M"])[1], "src_out", 99.0),
                            p.assets.pop("a3"),
                            setattr(p.sequence().cues[1], "end", p.sequence().cues[1].start - 1.0)))
    texts = issue_texts(t)
    assert len(texts) == 3 and all(x.startswith("錯誤：") for x in texts)
    assert any("來源範圍 0.00–99.00s 超出素材長度 60.00s" in x for x in texts)
    assert "錯誤：片段指向不存在的素材" in texts and "錯誤：字幕結束時間早於開始時間" in texts
    wheres = {t.issues.item(i).text(): t.issues.item(i).data(Qt.UserRole) for i in range(t.issues.count())}
    assert wheres["錯誤：片段指向不存在的素材"] == f"V2/{rig.ids['S']}"
    assert wheres["錯誤：字幕結束時間早於開始時間"] == cue.id
    # double click on the issue about the silent clip S (V2, 6.0 s): selects it and moves the playhead there
    row = texts.index("錯誤：片段指向不存在的素材")
    rig.pick("A", "B")
    click_item(rig, t.issues, row, double=True)
    assert rig.st.selection == {rig.ids["S"]} and rig.st.playhead == 6.0
    assert rig.clip_tab.f_start.text() == tc(6.0), "the clip tab shows the clip we jumped to"
    # ... and the caption issue selects the caption
    click_item(rig, t.issues, texts.index("錯誤：字幕結束時間早於開始時間"), double=True)
    assert rig.st.cue_selection == {cue.id} and rig.st.playhead == snap_time(cue.start, 30.0)
    assert rig.undos == 0, "jumping is navigation, not an edit"


def test_issue_list_jumps_with_the_keyboard_too(rig):
    t = rig.checks
    corrupt(rig, lambda p: setattr(p.sequence().find_clip(rig.ids["C"])[1], "src_out", 99.0))
    assert len(issue_texts(t)) == 1
    t.issues.setCurrentRow(0)
    rig.focus(t.issues)
    QTest.keyClick(t.issues, Qt.Key_Return)
    rig.settle()
    assert rig.st.selection == {rig.ids["C"]} and rig.st.playhead == 10.0


def test_sequence_level_issues_and_the_placeholder_do_not_jump(rig):
    t = rig.checks
    corrupt(rig, lambda p: setattr(p.sequence(), "width", 1921))
    assert issue_texts(t) == ["錯誤：輸出尺寸需為偶數且不小於 16：1921x1080"]
    assert t.issues.item(0).data(Qt.UserRole) == rig.seq.id
    rig.pick("D")
    rig.st.set_playhead(2.0)
    click_item(rig, t.issues, 0, double=True)
    assert rig.st.selection == {rig.ids["D"]} and rig.st.playhead == 2.0 and rig.st.cue_selection == set()


def test_warnings_use_the_warning_colour(rig, monkeypatch):
    from autocut.core.project import Issue
    from autocut.gui import theme
    monkeypatch.setattr(insp, "validate_sequence", lambda proj, seq: [
        Issue("warn", "x", "有點怪", f"V1/{rig.ids['A']}"), Issue("error", "y", "壞掉了", rig.seq.id)])
    t = rig.checks
    t.refresh()
    assert issue_texts(t) == ["警告：有點怪", "錯誤：壞掉了"]
    assert t.issues.item(0).foreground().color().name() == QColor(theme.WARN).name()
    assert t.issues.item(1).foreground().color().name() == QColor(theme.ERROR).name()
    click_item(rig, t.issues, 0, double=True)
    assert rig.st.selection == {rig.ids["A"]}


def test_issues_follow_edits_undo_and_redo(rig):
    t = rig.checks
    assert issue_texts(t) == ["沒有發現結構問題。"]
    corrupt(rig, lambda p: p.assets.pop("a3"), keep_undo=True)
    assert issue_texts(t) == ["錯誤：片段指向不存在的素材"]
    rig.st.undo()
    rig.settle()
    assert issue_texts(t) == ["沒有發現結構問題。"]
    rig.st.redo()
    rig.settle()
    assert issue_texts(t) == ["錯誤：片段指向不存在的素材"]
    rig.st.replace_project(make_project())
    rig.settle()
    assert issue_texts(t) == ["沒有發現結構問題。"] and tree_lines(t) == ["這個專案還沒有自動剪輯紀錄。"]


def test_long_lines_carry_their_whole_text_in_a_tooltip(rig):
    """A narrow dock elides long lines with 「…」: the full text must stay reachable (the old tooltip of an issue was
    its internal position, not its message)."""
    long_warning = "第 4 段（素材 a1.mp4 12.0–19.5 秒）畫面幾乎靜止，建議換一段或縮短，否則成品看起來會像卡住了"
    add_run(rig, warnings=[long_warning, "短的"], qa="warn")
    corrupt(rig, lambda p: setattr(p.sequence().find_clip(rig.ids["M"])[1], "src_out", 99.0))
    t = rig.checks

    def every(item):
        yield item
        for i in range(item.childCount()):
            yield from every(item.child(i))

    items = [x for i in range(t.runs.topLevelItemCount()) for x in every(t.runs.topLevelItem(i))]
    assert len(items) == 7 and all(x.toolTip(0) == x.text(0) for x in items)
    assert long_warning in [x.toolTip(0) for x in items]
    assert t.issues.count() == 1
    item = t.issues.item(0)
    assert item.toolTip() == item.text() == "錯誤：來源範圍 0.00–99.00s 超出素材長度 60.00s"
    assert item.data(Qt.UserRole) == f"A1/{rig.ids['M']}"


def test_opened_and_closed_run_nodes_keep_their_state_across_edits(rig):
    """Every edit rebuilds the list; reading the warnings while fixing the timeline must not fold them away again."""
    _p, run_id = add_run(rig, warnings=["第 3 段太短", "音樂太大聲"])
    t = rig.checks
    top, warns = t.runs.topLevelItem(0), t.runs.topLevelItem(0).child(3)
    assert top.isExpanded() and not warns.isExpanded() and warns.text(0) == "審核警告（2）"
    warns.setExpanded(True)                                  # the person opens the warnings ...
    rig.st.op("改別的", "add_cue", 9.0, 9.5, "x")
    rig.settle()
    assert t.runs.topLevelItem(0).isExpanded() and t.runs.topLevelItem(0).child(3).isExpanded()
    t.runs.topLevelItem(0).setExpanded(False)                # ... and folds the whole run away
    rig.st.undo()
    rig.settle()
    assert not t.runs.topLevelItem(0).isExpanded()
    t.runs.topLevelItem(0).setExpanded(True)
    assert t.runs.topLevelItem(0).child(3).isExpanded(), "the warnings node remembered its own state"
    rig.st.replace_project(make_project())                   # another project: nothing carried over
    rig.settle()
    assert tree_lines(t) == ["這個專案還沒有自動剪輯紀錄。"]


def test_the_checks_of_a_normal_editing_session_stay_clean(rig):
    """Ordinary edits through the inspector never produce a structural problem."""
    t = rig.checks
    rig.pick("A")
    rig.type_in(rig.clip_tab.f_speed, "2")
    rig.type_in(rig.clip_tab.f_fade_in, "0.5")
    rig.pick("B")
    rig.type_in(rig.clip_tab.f_xfade, "1")
    pick_index(rig, rig.seq_tab.f_size, size_entry(rig.seq_tab.f_size, (1080, 1920)))
    pick_index(rig, rig.seq_tab.f_fps, fps_entry(rig.seq_tab.f_fps, 25.0))
    assert issue_texts(t) == ["沒有發現結構問題。"]


# ================================================================================================= whole panel
def test_replacing_the_project_resets_every_tab(rig):
    rig.pick("A", "B")
    rig.st.select_cues([rig.seq.cues[0].id])
    add_run(rig, warnings=["舊專案的警告"])
    fresh = make_project()
    fresh.sequence().name = "全新序列"
    ops.add_cue(fresh, fresh.sequence(), 1.0, 2.0, "新專案的字幕")
    rig.st.replace_project(fresh)
    rig.settle()
    assert not rig.clip_tab.empty.isHidden() and rig.clip_tab.title.text() == ""
    assert table_rows(rig) == [("0:01.000", "0:02.000", "新專案的字幕")]
    assert rig.cap.table.selectionModel().selectedRows() == [] and not rig.cap.b_delete.isEnabled()
    assert rig.seq_tab.f_name.text() == "全新序列" and "0 個片段　·　1 則字幕" in rig.seq_tab.info.text()
    assert tree_lines(rig.checks) == ["這個專案還沒有自動剪輯紀錄。"] and issue_texts(rig.checks) == ["沒有發現結構問題。"]
    assert not rig.checks.review.btn_ok.isEnabled() and rig.checks.review.notes.text() == ""
    assert rig.undos == 0 and rig.st.project is fresh


# ================================================================================================ screenshots
def shot(widget, name: str) -> Path:
    """Save ``widget.grab()`` as ``screenshots/inspector_<name>.png`` (git-ignored) and sanity-check the picture."""
    try:
        SHOTS.mkdir(parents=True, exist_ok=True)
    except OSError:
        pytest.skip("screenshots/ is not writable")
    path = SHOTS / f"inspector_{name}.png"
    pix = widget.grab()
    assert pix.save(str(path)), path
    img = pix.toImage()
    assert img.width() >= 340 and img.height() >= 120
    bright = total = 0
    for y in range(0, img.height(), 3):
        for x in range(0, img.width(), 3):
            c = QColor(img.pixel(x, y))
            total += 1
            bright += c.lightness() > 150
    assert bright / total < 0.2, f"{name}: too many bright pixels -- is part of the panel not using the dark theme?"
    return path


def test_screenshots_of_every_tab(rig):
    r, p = rig, rig.panel
    # ---- 片段: one clip with a keyframe and non-default values, the multi-selection and the empty state
    shot(p, "clip_empty")
    r.seed("props", "set_clip_props", r.ids["C"], volume=0.5, brightness=0.2, contrast=1.3, saturation=0.8,
           fade_in=0.5, flip_h=True, fit="blur", zoom=1.6)
    r.seed("key", "set_transform_key", r.ids["C"], t_local=1.0, cx=0.3, cy=0.5)
    r.seed("xfade", "set_crossfade", r.ids["C"], 0.5)
    r.st.set_playhead(11.0)
    r.pick("C")
    p.setCurrentWidget(r.clip_tab)
    r.settle(50)
    shot(p, "clip")
    shot(r.clip_tab.widget(), "clip_full")
    r.seed("speed", "set_speed", r.ids["A"], 2.0)
    r.pick("A", "B", "C")
    shot(p, "clip_multi")
    r.pick("M")
    shot(p, "clip_audio")
    r.pick()
    # ---- 字幕
    r.st.select_cues([r.seq.cues[1].id])
    p.setCurrentWidget(r.panel.captions_tab)
    r.settle(50)
    shot(p, "captions")
    # ---- 序列
    p.setCurrentWidget(r.panel.sequence_tab)
    r.settle(50)
    shot(p, "sequence")
    # ---- 檢查: two runs (one with warnings), a structure error, review still pending
    add_run(r, n=1, kind="assemble", when="2026-09-30T20:15:00+08:00", link=False, warnings=["第 2 段超過 60 秒"])
    _plan, run_id = add_run(r, n=2, warnings=["音樂比人聲大聲 3 dB", "第 4 段畫面幾乎靜止", "結尾沒有淡出"], qa="warn")
    corrupt(r, lambda proj: proj.assets.pop("a3"))
    p.setCurrentWidget(r.panel.checks_tab)
    r.settle(50)
    shot(p, "checks")
    assert run_of(r, run_id)["human_review"] == "pending"


@pytest.fixture
def narrow_rig(qapp):
    r = Rig(qapp, w=340, h=900)                            # the dock's minimum width
    yield r
    r.panel.close()
    r.panel.deleteLater()
    qapp.processEvents()


def clipped_texts(root) -> list[tuple[str, int, int]]:
    """Visible buttons / one-line labels whose text is wider than the widget (the text would be cut off)."""
    from PySide6.QtWidgets import QAbstractButton, QLabel
    out = []
    for w in root.findChildren(QAbstractButton):
        if w.isVisibleTo(root) and w.text() and w.fontMetrics().horizontalAdvance(w.text()) > w.width() > 0:
            out.append((w.text(), w.fontMetrics().horizontalAdvance(w.text()), w.width()))
    for w in root.findChildren(QLabel):
        if w.isVisibleTo(root) and w.text() and not w.wordWrap() and "<" not in w.text():
            need = max(w.fontMetrics().horizontalAdvance(line) for line in w.text().split("\n"))
            if need > w.width() > 0:
                out.append((w.text(), need, w.width()))
    return out


def test_every_tab_fits_the_minimum_dock_width_without_clipped_text(narrow_rig):
    r = narrow_rig
    for tab in (r.clip_tab, r.panel.captions_tab, r.panel.sequence_tab, r.panel.checks_tab):
        assert tab.minimumSizeHint().width() <= r.panel.minimumWidth(), type(tab).__name__
    assert r.panel.width() == 340
    add_run(r, warnings=["第 3 段太短"], qa="warn")
    for keys in (("C",), ("A", "B"), ("M",), ()):
        r.pick(*keys)
        bar = r.clip_tab.horizontalScrollBar()
        assert bar.maximum() == 0, f"the clip tab scrolls sideways with selection {keys}"
        assert clipped_texts(r.clip_tab) == [], keys
    for tab in (r.cap, r.seq_tab, r.checks):
        r.show_tab(tab)
        assert clipped_texts(tab) == [], type(tab).__name__
    r.show_tab(r.panel.clip_tab)
    r.pick("C")
    shot(r.panel, "clip_narrow")
    r.pick("A", "B")
    shot(r.panel, "clip_multi_narrow")
    for name, tab in (("captions", r.panel.captions_tab), ("sequence", r.panel.sequence_tab),
                      ("checks", r.panel.checks_tab)):
        r.show_tab(tab)
        shot(r.panel, f"{name}_narrow")


# ============================================================================================= 字幕: title cards (字卡)
def test_card_button_turns_the_selection_into_title_cards_and_back(rig):
    cap = rig.cap
    assert not cap.b_card.isEnabled()                                           # nothing selected
    ids = [q.id for q in rig.seq.cues[:2]]
    rig.st.select_cues(ids)
    rig.settle()
    assert cap.b_card.isEnabled() and not cap.b_card.isChecked()
    with one_undo(rig, "設為字卡"):
        cap.b_card.click()
        rig.settle()
    assert [q.is_card for q in rig.seq.cues] == [True, True, False] and cap.b_card.isChecked()
    assert table_rows(rig)[0][2] == "第一句"                                    # the text is untouched
    assert cap.table.item(0, 2).background().color().alpha() > 0 and "字卡" in cap.table.item(0, 2).toolTip()
    assert not cap.table.item(2, 2).toolTip()
    assert any("已把 2 則改成字卡" in m for m in rig.status)
    shot(rig.panel, "captions_cards")
    rig.st.select_cues(ids)
    rig.settle()
    with one_undo(rig, "改回字幕"):
        cap.b_card.click()
        rig.settle()
    assert not any(q.is_card for q in rig.seq.cues) and not cap.b_card.isChecked()


def test_srt_export_and_import_leave_title_cards_alone(rig, monkeypatch, tmp_path):
    rig.st.op("字卡", "update_cue", rig.seq.cues[0].id, card=True)
    rig.settle()
    out = tmp_path / "sub.srt"
    stub(monkeypatch, "save_file", str(out))
    rig.cap.b_export.click()
    rig.settle()
    text = out.read_text(encoding="utf-8")
    assert "第一句" not in text and "第二句" in text and text.startswith("1\n")  # cards are not subtitles
    assert any("已匯出 2 則字幕" in m for m in rig.status)
    stub(monkeypatch, "open_file", str(write_srt(tmp_path / "a.srt")))
    asked = stub(monkeypatch, "ask", True)
    with one_undo(rig, "匯入字幕"):
        rig.cap.b_import.click()
    assert "2 則" in asked[0][1] and "字卡會保留" in asked[0][1]
    assert [(q.text, q.is_card) for q in rig.seq.cues if q.is_card] == [("第一句", True)]
    assert len([q for q in rig.seq.cues if not q.is_card]) == 3


def test_only_cards_means_nothing_to_export_as_srt(rig, monkeypatch):
    for q in list(rig.seq.cues):
        rig.st.op("字卡", "update_cue", q.id, card=True)
    rig.settle()
    saved = stub(monkeypatch, "save_file", "")
    rig.cap.b_export.click()
    assert not saved and any("只有字卡" in m for m in rig.status)
    rig.seq_tab
    assert "3 張字卡" in rig.panel.sequence_tab.info.text() and "0 則字幕" in rig.panel.sequence_tab.info.text()


def test_auto_captions_never_give_a_new_subtitle_the_id_of_a_card(rig, monkeypatch, speech):
    """The recogniser numbers its cues pq0001, pq0002 ... on every run.  A 字卡 made from an earlier run's pq0002 stays
    when the subtitles are recognised again, so the new pq0002 must get another id: editing or deleting that subtitle
    must never touch the card."""
    answer_asks(monkeypatch, 繼續=True)
    speech.real_ids = True
    speech.cues = [(0.5, 2.0, "辨識一"), (2.5, 4.0, "辨識二"), (6.0, 7.0, "辨識三")]
    rig.cap.b_auto.click()
    finish(rig)
    assert [q.id for q in rig.seq.cues] == ["pq0001", "pq0002", "pq0003"]
    rig.st.select_cues(["pq0002"])
    rig.settle()
    rig.cap.b_card.click()
    rig.settle()
    with one_undo(rig, "自動字幕"):
        rig.cap.b_auto.click()
        finish(rig)
    ids = [q.id for q in rig.seq.cues]
    assert len(set(ids)) == len(ids) == 4 and rig.cap.table.rowCount() == 4
    assert [(q.id, q.text) for q in rig.seq.cues if q.is_card] == [("pq0002", "辨識二")]
    twin = next(q for q in rig.seq.cues if not q.is_card and q.text == "辨識二")
    assert twin.id != "pq0002"
    ed = cell_editor(rig, rig.cap._row_of(twin.id), 2)                   # retype the subtitle's text in the table
    ed.selectAll()
    send_text(ed, "改過的字幕")
    QTest.keyClick(ed, Qt.Key_Return)
    rig.settle()
    assert [(q.text, q.is_card) for q in rig.seq.cues if q.text in ("辨識二", "改過的字幕")] == [
        ("辨識二", True), ("改過的字幕", False)]
    rig.st.select_cues([twin.id])
    rig.settle()
    with one_undo(rig, "刪除字幕"):
        rig.cap.b_delete.click()
    assert [(q.id, q.text) for q in rig.seq.cues if q.is_card] == [("pq0002", "辨識二")]
    assert "改過的字幕" not in [q.text for q in rig.seq.cues]


def test_shift_all_moves_the_subtitles_and_leaves_title_cards_in_place(rig, monkeypatch):
    """全部位移 is how an imported SRT is synced; a 字卡 is placed on purpose (often pinned to a source moment)."""
    rig.seed("字卡", "add_cue", 3.0, 5.0, "嘉賓名字", card=True)
    assert "字卡不動" in rig.cap.b_shift.toolTip()
    asked = stub(monkeypatch, "ask_number", -0.5)
    with one_undo(rig, "位移字幕"):
        rig.cap.b_shift.click()
    assert asked and "字卡不動" in asked[0][1]
    assert rig.cues() == [(0.5, 1.5, "第一句"), (2.5, 4.0, "第二句"), (3.0, 5.0, "嘉賓名字"), (5.5, 6.5, "第三句")]
    for q in [q for q in rig.seq.cues if not q.is_card]:                 # only cards left: nothing to shift
        rig.seed("字卡", "update_cue", q.id, card=True)
    assert not rig.cap.b_shift.isEnabled()
    asked = stub(monkeypatch, "ask_number", 1.0)
    with untouched(rig):
        rig.cap._shift()
    assert asked == [] and any("沒有可位移的字幕" in m for m in rig.status)


def test_split_without_a_selection_cuts_the_subtitle_not_the_card(rig):
    """分割 says 在播放頭把字幕切成兩則: with nothing chosen it cuts the subtitle under the playhead, even where a 字卡 that
    starts earlier overlaps it; a card is cut only when it is selected or is the only thing there."""
    card = rig.seed("字卡", "add_cue", 2.5, 6.0, "字卡標題", card=True)

    def cards():
        return [(round(q.start, 3), round(q.end, 3)) for q in rig.seq.cues if q.is_card]

    rig.st.set_playhead(4.0)
    with one_undo(rig, "分割字幕"):
        rig.cap.b_split.click()
    assert [q.text for q in rig.seq.cues if q.is_card] == ["字卡標題"] and cards() == [(2.5, 6.0)]
    assert [c for c in rig.cues() if c[2] in ("第二", "句")] == [(3.0, 4.0, "第二"), (4.0, 4.5, "句")]
    rig.st.set_playhead(3.5)                                             # the card, once selected, is what gets cut
    rig.st.select_cues([card.id])
    rig.settle()
    with one_undo(rig, "分割字幕"):
        rig.cap.b_split.click()
    assert cards() == [(2.5, 3.5), (3.5, 6.0)]
    assert [c for c in rig.cues() if c[2] in ("第二", "句")] == [(3.0, 4.0, "第二"), (4.0, 4.5, "句")]
    rig.st.select_cues([])
    rig.st.set_playhead(5.0)                                             # nothing but a card under the playhead
    rig.settle()
    with one_undo(rig, "分割字幕"):
        rig.cap.b_split.click()
    assert cards() == [(2.5, 3.5), (3.5, 5.0), (5.0, 6.0)]


def test_add_cue_is_not_cut_short_by_a_title_card(rig):
    """A 字卡 is its own layer and may overlap subtitles: only the next SUBTITLE limits a new one's default length."""
    rig.seed("字卡", "add_cue", 8.6, 11.0, "字卡", card=True)
    rig.st.set_playhead(8.0)
    with one_undo(rig, "新增字幕"):
        rig.cap.b_add.click()
        rig.settle()
        ed = rig.cap.table.findChild(QLineEdit)
        assert ed is not None
        QTest.keyClick(ed, Qt.Key_Escape)
        rig.settle()
    assert [c for c in rig.cues() if c[2] in ("新字幕", "字卡")] == [(8.0, 10.0, "新字幕"), (8.6, 11.0, "字卡")]
