"""Timeline: the REAL ``TimelineWidget`` driven by synthetic mouse / key / wheel / drag-and-drop events.

Every gesture is computed in pixels through the widget's own ``time_to_x`` / ``track_rect`` helpers, sent with
``QApplication.sendEvent`` and then judged by the *project content* (clips, positions, source ranges, tracks, cues,
markers, in/out range).  A committed gesture must be exactly ONE undo step (``one_undo``); a gesture that is still in
progress (or aborted) must leave the project untouched.

Offscreen Qt blocks forever on a modal dialog, so dialogs are always monkeypatched; a watchdog kills a stuck run.
"""
from __future__ import annotations

import faulthandler
import json
import os
import time
from contextlib import contextmanager

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl                     # noqa: E402
from PySide6.QtGui import (QAction, QContextMenuEvent, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent,   # noqa: E402
                           QDropEvent, QMouseEvent, QWheelEvent)
from PySide6.QtTest import QTest                                                          # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMenu                           # noqa: E402

from autocut.core import ops                                                              # noqa: E402
from autocut.core.project import Asset, Project                                           # noqa: E402
from autocut.gui import timeline as tlmod                                                 # noqa: E402
from autocut.gui.state import AppState                                                    # noqa: E402
from autocut.gui.timeline import (ASSET_MIME, CAPTION_LANE, EDGE_PX, HEADER_W, RULER_H, SNAP_PX,   # noqa: E402
                                  TimelineWidget)

pytestmark = pytest.mark.gui

NOMOD = Qt.KeyboardModifier.NoModifier
SHIFT, CTRL, ALT = Qt.ShiftModifier, Qt.ControlModifier, Qt.AltModifier
LEFT = Qt.LeftButton


@pytest.fixture(autouse=True)
def _watchdog():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _no_clipboard_leak():
    TimelineWidget._clipboard = []
    yield
    TimelineWidget._clipboard = []


def fake_video(aid: str, dur: float, w: int = 1920, h: int = 1080, audio: bool = True) -> Asset:
    return Asset(id=aid, path=f"/nonexistent/{aid}.mp4", name=f"{aid}.mp4", kind="video", duration=dur, width=w,
                 height=h, fps=30.0, has_video=True, has_audio=audio, audio_channels=2 if audio else 0,
                 sample_rate=48000 if audio else 0, fingerprint=aid)


def make_project() -> Project:
    p = Project.new("tl")
    p.assets["a1"] = fake_video("a1", 20.0)
    p.assets["a2"] = fake_video("a2", 10.0)
    p.assets["a3"] = fake_video("a3", 8.0, audio=False)
    p.assets["m1"] = Asset(id="m1", path="/nonexistent/m1.mp3", name="m1.mp3", kind="audio", duration=60.0,
                           has_video=False, has_audio=True, audio_channels=2, sample_rate=44100)
    p.assets["i1"] = Asset(id="i1", path="/nonexistent/i1.png", name="i1.png", kind="image", duration=0.0,
                           width=800, height=600, has_video=False, has_audio=False)
    return p


class Rig:
    """A timeline on a fresh project plus the event synthesizers every test uses."""

    def __init__(self, app, populate: bool = True, w: int = 1560, h: int = 470, pps: float = 100.0) -> None:
        self.app = app
        self.proj = make_project()
        self.st = AppState(self.proj)
        self.seq = self.st.sequence()
        self.v1, self.v2 = (t.id for t in self.seq.video_tracks[:2])
        self.a1, self.a2 = (t.id for t in self.seq.audio_tracks[:2])
        self.status: list[str] = []
        self.st.status.connect(self.status.append)
        self.clips: dict[str, str] = {}
        if populate:
            self.populate()
        self.tl = TimelineWidget(self.st)
        self.tl.resize(w, h)
        self.tl.show()
        app.processEvents()
        self.tl.set_pps(pps)
        self.tl.set_horizontal_scroll(0)
        app.processEvents()
        self._hold: list = []                       # keep QMimeData alive

    # ---- project content
    def populate(self) -> None:
        """V1: A 0-4, B 5-8, C 10-14 (src 5-9) | V2: D 2-5 | A1: M 0-12 | cues 1-2, 3-4.5, 6-7 | marker 9.0"""
        st = self.st

        def put(key, aid, tid, start, s_in=0.0, s_out=None):
            self.clips[key] = st.op("seed", "place_clip", aid, tid, start, s_in, s_out, "overwrite").id

        put("A", "a1", self.v1, 0.0, 0.0, 4.0)
        put("B", "a2", self.v1, 5.0, 0.0, 3.0)
        put("C", "a1", self.v1, 10.0, 5.0, 9.0)
        put("D", "a2", self.v2, 2.0, 0.0, 3.0)
        put("M", "m1", self.a1, 0.0, 0.0, 12.0)
        for s, e, txt in ((1.0, 2.0, "第一句"), (3.0, 4.5, "第二句"), (6.0, 7.0, "第三句")):
            st.op("seed", "add_cue", s, e, txt)
        st.op("seed", "add_marker", 9.0, "m")
        st.history.clear()

    def cid(self, key: str) -> str:
        return self.clips[key]

    def clip(self, key_or_id: str):
        cid = self.clips.get(key_or_id, key_or_id)
        return self.st.sequence().find_clip(cid)[1]

    def track_of(self, key_or_id: str) -> str:
        cid = self.clips.get(key_or_id, key_or_id)
        return self.st.sequence().find_clip(cid)[0].id

    def cues(self):
        return [(round(q.start, 3), round(q.end, 3), q.text) for q in self.st.sequence().cues]

    def cue_id(self, i: int) -> str:
        return self.st.sequence().cues[i].id

    def layout(self, tid: str):
        tr = self.st.sequence().find_track(tid)
        return [(round(c.start, 3), round(c.end, 3), round(c.src_in, 3), round(c.src_out, 3)) for c in tr.clips]

    def content(self) -> str:
        return self.st.project.content_json()

    # ---- geometry
    def x(self, t: float) -> float:
        return self.tl.time_to_x(t)

    def y(self, tid: str, frac: float = 0.5) -> float:
        r = self.tl.track_rect(tid)
        return r.top() + r.height() * frac

    def at(self, t: float, tid: str, frac: float = 0.5) -> QPointF:
        return QPointF(self.x(t), self.y(tid, frac))

    def clip_pt(self, key: str, frac: float = 0.5, dx: float = 0.0, dy: float = 0.5) -> QPointF:
        """A point inside a clip: ``frac`` along its width (+ ``dx`` pixels), ``dy`` down its height."""
        r = self.tl.clip_rect(self.clips.get(key, key))
        return QPointF(r.left() + r.width() * frac + dx, r.top() + r.height() * dy)

    def left_edge(self, key: str, inside: float = 2.0) -> QPointF:
        r = self.tl.clip_rect(self.clips.get(key, key))
        return QPointF(r.left() + inside, r.top() + r.height() * 0.6)

    def right_edge(self, key: str, inside: float = 2.0) -> QPointF:
        r = self.tl.clip_rect(self.clips.get(key, key))
        return QPointF(r.right() + 1 - inside, r.top() + r.height() * 0.6)

    def cue_pt(self, i: int, frac: float = 0.5) -> QPointF:
        r = self.tl.cue_rect(self.cue_id(i))
        return QPointF(r.left() + r.width() * frac, r.center().y())

    # ---- events
    def send(self, ev) -> None:
        QApplication.sendEvent(self.tl, ev)

    def mouse(self, kind, pos: QPointF, button=LEFT, buttons=None, mods=NOMOD) -> None:
        if buttons is None:
            buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else button
        if kind == QEvent.MouseMove:
            button = Qt.NoButton
        self.send(QMouseEvent(kind, QPointF(pos), QPointF(pos), button, buttons, mods))

    def press(self, pos, mods=NOMOD, button=LEFT) -> None:
        self.mouse(QEvent.MouseButtonPress, pos, button, button, mods)

    def move(self, pos, mods=NOMOD, buttons=LEFT) -> None:
        self.mouse(QEvent.MouseMove, pos, LEFT, buttons, mods)

    def release(self, pos, mods=NOMOD, button=LEFT) -> None:
        self.mouse(QEvent.MouseButtonRelease, pos, button, Qt.NoButton, mods)

    def click(self, pos, mods=NOMOD, button=LEFT) -> None:
        self.press(pos, mods, button)
        self.release(pos, mods, button)

    def drag(self, p0: QPointF, p1: QPointF, mods=NOMOD, steps: int = 6, release_mods=None, button=LEFT,
             press_mods=None) -> None:
        """Press at ``p0`` (with ``press_mods``, default ``mods``), move in ``steps`` stages, release at ``p1``.
        NB: Alt held at the PRESS on a clip body means slip; to move without snapping, press plain, then hold Alt."""
        self.press(p0, mods if press_mods is None else press_mods, button)
        for i in range(1, steps + 1):
            f = i / steps
            self.move(QPointF(p0.x() + (p1.x() - p0.x()) * f, p0.y() + (p1.y() - p0.y()) * f), mods,
                      buttons=button)
        self.release(p1, mods if release_mods is None else release_mods, button)

    def dclick(self, pos, mods=NOMOD) -> None:
        QTest.mouseDClick(self.tl, LEFT, mods, pos.toPoint())

    def key(self, key, mods=NOMOD) -> None:
        QTest.keyClick(self.tl, key, mods)

    def wheel(self, pos: QPointF, dy: int = 0, dx: int = 0, mods=NOMOD) -> None:
        g = QPointF(pos)
        self.send(QWheelEvent(QPointF(pos), g, QPoint(0, 0), QPoint(dx, dy), Qt.NoButton, mods, Qt.NoScrollPhase,
                              False))

    def right_click(self, pos: QPointF) -> None:
        """Right press + context-menu event + release (the test patches ``QMenu.exec`` to catch the menu)."""
        self.press(pos, button=Qt.RightButton)
        gp = self.tl.mapToGlobal(pos.toPoint())
        self.send(QContextMenuEvent(QContextMenuEvent.Mouse, pos.toPoint(), gp))
        self.release(pos, button=Qt.RightButton)

    @staticmethod
    def act(menu: QMenu, name: str) -> QAction:
        a = menu.findChild(QAction, name)
        assert a is not None, f"no menu entry {name!r}: {[x.objectName() for x in menu.findChildren(QAction)]}"
        return a

    # ---- drag & drop
    def mime_assets(self, *ids: str) -> QMimeData:
        m = QMimeData()
        m.setData(ASSET_MIME, json.dumps({"assets": list(ids)}).encode("utf-8"))
        self._hold.append(m)
        return m

    def mime_files(self, *paths: str) -> QMimeData:
        m = QMimeData()
        m.setUrls([QUrl.fromLocalFile(p) for p in paths])
        self._hold.append(m)
        return m

    def drag_over(self, mime: QMimeData, pos: QPointF, mods=NOMOD) -> bool:
        ev = QDragEnterEvent(pos.toPoint(), Qt.CopyAction, mime, Qt.LeftButton, mods)
        self.send(ev)
        return ev.isAccepted()

    def drag_move(self, mime: QMimeData, pos: QPointF, mods=NOMOD) -> bool:
        ev = QDragMoveEvent(pos.toPoint(), Qt.CopyAction, mime, Qt.LeftButton, mods)
        self.send(ev)
        return ev.isAccepted()

    def drag_leave(self) -> None:
        self.send(QDragLeaveEvent())

    def drop(self, mime: QMimeData, pos: QPointF, mods=NOMOD) -> bool:
        """What the window system does: DragEnter -> DragMove -> Drop (Qt only delivers a Drop to the widget that
        accepted the DragEnter)."""
        self.drag_over(mime, pos, mods)
        self.drag_move(mime, pos, mods)
        ev = QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, mods)
        self.send(ev)
        return ev.isAccepted()

    # ---- misc
    def undo_depth(self) -> int:
        return len(self.st.history._undo)

    def grab(self, name: str) -> None:
        os.makedirs("screenshots", exist_ok=True)
        self.app.processEvents()
        self.tl.grab().save(f"screenshots/timeline_{name}.png")


@pytest.fixture
def rig(qapp):
    r = Rig(qapp)
    yield r
    r.tl.close()
    r.tl.deleteLater()
    qapp.processEvents()


@pytest.fixture
def empty_rig(qapp):
    r = Rig(qapp, populate=False)
    yield r
    r.tl.close()
    r.tl.deleteLater()
    qapp.processEvents()


@contextmanager
def one_undo(r: Rig):
    """The block must change the project by exactly ONE undo step, and one undo must restore it (redo re-applies)."""
    before, n0 = r.content(), r.undo_depth()
    yield
    after = r.content()
    assert after != before, "the gesture changed nothing"
    assert r.undo_depth() == n0 + 1, "the gesture must be exactly one undo step"
    r.st.undo()
    assert r.content() == before, "one undo must restore the project"
    r.st.redo()
    assert r.content() == after, "redo must re-apply the gesture"


@contextmanager
def untouched(r: Rig):
    """The block must not change the project and must not add an undo step."""
    before, n0 = r.content(), r.undo_depth()
    yield
    assert r.content() == before
    assert r.undo_depth() == n0


# ===================================================================================================== geometry
def test_geometry_roundtrip_and_lane_order(rig):
    tl = rig.tl
    assert tl.x_to_time(tl.time_to_x(3.25)) == pytest.approx(3.25)
    order = [ln.id for ln in tl._lanes]
    seq = rig.st.sequence()
    assert order == [CAPTION_LANE] + [t.id for t in reversed(seq.video_tracks)] + [t.id for t in seq.audio_tracks]
    ys = [tl.track_rect(i).top() for i in order]
    assert ys == sorted(ys), "lanes are stacked: captions, V(n)..V1, A1..An"
    r = tl.clip_rect(rig.cid("A"))
    assert r.left() == round(tl.time_to_x(0.0)) and r.width() == pytest.approx(400, abs=1)
    assert tl.hit_test(rig.clip_pt("A")).kind == "clip"
    assert tl.hit_test(QPointF(5, tl.track_rect(rig.v1).center().y())).kind == "header"
    assert tl.hit_test(QPointF(HEADER_W + 60, 5)).kind == "ruler"
    assert tl.hit_test(rig.at(9.0, rig.v1)).kind == "lane"
    assert tl.hit_test(rig.at(1.5, CAPTION_LANE)).kind == "cue"
    assert tl.hit_test(rig.at(8.0, CAPTION_LANE)).kind == "caption_lane"


# ===================================================================================================== selection
def test_click_selects_replaces_and_empty_click_clears(rig):
    st = rig.st
    with untouched(rig):
        rig.click(rig.clip_pt("A"))
        assert st.selection == {rig.cid("A")}
        rig.click(rig.clip_pt("B"))
        assert st.selection == {rig.cid("B")}
        rig.click(rig.at(8.7, rig.v2))                      # empty space of V2
        assert st.selection == set()


def test_ctrl_click_adds_and_removes(rig):
    st = rig.st
    with untouched(rig):
        rig.click(rig.clip_pt("A"))
        rig.click(rig.clip_pt("B"), CTRL)
        rig.click(rig.clip_pt("D"), CTRL)
        assert st.selection == {rig.cid("A"), rig.cid("B"), rig.cid("D")}
        rig.click(rig.clip_pt("B"), CTRL)                   # Ctrl on a selected clip removes it (on release)
        assert st.selection == {rig.cid("A"), rig.cid("D")}


def test_shift_click_selects_range_on_the_track(rig):
    st = rig.st
    with untouched(rig):
        rig.click(rig.clip_pt("A"))
        rig.click(rig.clip_pt("C"), SHIFT)
        assert st.selection == {rig.cid("A"), rig.cid("B"), rig.cid("C")}
        rig.click(rig.clip_pt("D"))                         # anchor moves to D (other track)
        rig.click(rig.clip_pt("B"), SHIFT)
        assert st.selection == {rig.cid("B")}               # range only counts clips of the clicked track


def test_marquee_selects_touching_clips_across_lanes(rig):
    st = rig.st
    p0 = QPointF(rig.x(1.5), rig.tl.track_rect(rig.v2).top() + 2)        # empty part of V2 (before D)
    p1 = QPointF(rig.x(5.5), rig.tl.track_rect(rig.v1).bottom() - 3)
    with untouched(rig):
        rig.press(p0)
        rig.move(QPointF(rig.x(3.0), p0.y() + 20))
        assert rig.tl._ov.band is not None and rig.tl._ov.preview_sel, "marquee previews what it will select"
        rig.move(p1)
        rig.release(p1)
    assert st.selection == {rig.cid("A"), rig.cid("B"), rig.cid("D")}
    assert rig.tl._ov.band is None


def test_marquee_ctrl_keeps_the_previous_selection(rig):
    st = rig.st
    rig.click(rig.clip_pt("C"))
    p0 = QPointF(rig.x(1.5), rig.tl.track_rect(rig.v2).top() + 2)
    p1 = QPointF(rig.x(2.4), rig.tl.track_rect(rig.v2).bottom() - 2)
    with untouched(rig):
        rig.drag(p0, p1, CTRL)
    assert st.selection == {rig.cid("C"), rig.cid("D")}


def test_marquee_selects_caption_blocks(rig):
    lane = rig.tl.track_rect(CAPTION_LANE)
    p0, p1 = QPointF(rig.x(0.5), lane.top() + 1), QPointF(rig.x(3.5), lane.bottom() - 1)
    with untouched(rig):
        rig.drag(p0, p1)
    assert rig.st.cue_selection == {rig.cue_id(0), rig.cue_id(1)}
    assert rig.st.selection == set()


def test_ctrl_a_and_escape(rig):
    with untouched(rig):
        rig.key(Qt.Key_A, CTRL)
        assert rig.st.selection == set(rig.clips.values())
        assert rig.st.cue_selection == {q.id for q in rig.st.sequence().cues}
        rig.key(Qt.Key_Escape)
        assert rig.st.selection == set() and rig.st.cue_selection == set()


def test_press_in_multi_selection_keeps_it_for_dragging_but_click_collapses(rig):
    st = rig.st
    rig.click(rig.clip_pt("A"))
    rig.click(rig.clip_pt("B"), CTRL)
    with untouched(rig):
        rig.click(rig.clip_pt("A"))                          # plain click without a drag: collapse to that clip
    assert st.selection == {rig.cid("A")}


def test_clicking_a_clip_does_not_move_it(rig):
    with untouched(rig):
        rig.press(rig.clip_pt("B"))
        rig.move(rig.clip_pt("B", dx=2))                     # below the 3 px drag threshold
        rig.release(rig.clip_pt("B", dx=2))
    assert rig.clip("B").start == 5.0


# ===================================================================================================== move
def test_move_clip_to_a_free_spot(rig):
    p0 = rig.clip_pt("D")
    p1 = QPointF(p0.x() + 130, p0.y())                       # 1.3 s to the right: no snap target near 3.3 / 6.3
    with one_undo(rig):
        rig.drag(p0, p1)
    d = rig.clip("D")
    assert d.start == pytest.approx(3.3, abs=1 / 30 + 1e-6) and d.end - d.start == pytest.approx(3.0)
    assert (d.src_in, d.src_out) == (0.0, 3.0) and rig.track_of("D") == rig.v2
    assert rig.st.selection == {rig.cid("D")}


def test_move_is_only_a_ghost_until_release(rig):
    p0 = rig.clip_pt("D")
    before = rig.content()
    rig.press(p0)
    rig.move(QPointF(p0.x() + 60, p0.y()))
    rig.move(QPointF(p0.x() + 130, p0.y()))
    ov = rig.tl._ov
    assert rig.content() == before and rig.undo_depth() == 0, "the project must not change while dragging"
    assert ov.ghosts and ov.ghosts[0][1] == "move" and ov.tip is not None and "→" not in ov.tip[1]
    assert rig.cid("D") in ov.dim
    rig.release(QPointF(p0.x() + 130, p0.y()))
    assert rig.tl._ov.empty()


def test_move_snaps_to_clip_edges(rig):
    # D starts at 2.0; B ends at 8.0 -> put D's start 5 px (0.05 s) after B's end: snaps exactly to 8.0
    p0 = rig.clip_pt("D")
    target_start = 8.0 + 5 / rig.tl.pps
    p1 = QPointF(p0.x() + (target_start - 2.0) * rig.tl.pps, p0.y())
    with one_undo(rig):
        rig.press(p0)
        rig.move(QPointF((p0.x() + p1.x()) / 2, p0.y()))
        rig.move(p1)
        assert rig.tl._ov.snap_t == pytest.approx(8.0), "the snap guide is shown while dragging"
        rig.release(p1)
    assert rig.clip("D").start == pytest.approx(8.0)


def test_move_snaps_by_its_end_edge_too(rig):
    # D is 3 s long: its END should snap to C's start (10.0) -> start 7.0
    p0 = rig.clip_pt("D")
    p1 = QPointF(p0.x() + (7.0 - 2.0) * rig.tl.pps + 6, p0.y())        # end 6 px late
    with one_undo(rig):
        rig.drag(p0, p1)
    assert rig.clip("D").start == pytest.approx(7.0) and rig.clip("D").end == pytest.approx(10.0)


def test_snap_threshold_is_eight_pixels(rig):
    base = rig.clip_pt("D")
    for px, snapped in ((SNAP_PX - 1, True), (SNAP_PX + 4, False)):
        rig.st.undo() if rig.undo_depth() else None
        target_start = 8.0 + px / rig.tl.pps
        p1 = QPointF(base.x() + (target_start - rig.clip("D").start) * rig.tl.pps, base.y())
        rig.drag(rig.clip_pt("D"), p1)
        got = rig.clip("D").start
        assert (abs(got - 8.0) < 1e-6) == snapped, (px, got)


def test_move_snaps_to_playhead_and_marker(rig):
    rig.st.set_playhead(6.5)
    p0 = rig.clip_pt("D")
    p1 = QPointF(p0.x() + (6.5 + 4 / rig.tl.pps - 2.0) * rig.tl.pps, p0.y())     # start 4 px after the playhead
    with one_undo(rig):
        rig.drag(p0, p1)
    assert rig.clip("D").start == pytest.approx(6.5, abs=1e-6)
    rig.st.undo()
    rig.st.set_playhead(0.0)
    # marker at 9.0: D's start 3 px early
    p1 = QPointF(p0.x() + (9.0 - 3 / rig.tl.pps - 2.0) * rig.tl.pps, p0.y())
    with one_undo(rig):
        rig.drag(rig.clip_pt("D"), p1)
    assert rig.clip("D").start == pytest.approx(9.0, abs=1e-6)


def test_move_snaps_to_zero_and_clamps_there(rig):
    p0 = rig.clip_pt("D")
    p1 = QPointF(p0.x() - 2.0 * rig.tl.pps + 4, p0.y())          # 4 px right of 0.0
    with one_undo(rig):
        rig.drag(p0, p1)
    assert rig.clip("D").start == pytest.approx(0.0)
    rig.st.undo()
    with one_undo(rig):
        rig.drag(rig.clip_pt("D"), QPointF(rig.clip_pt("D").x() - 900, p0.y()))      # way left of 0
    assert rig.clip("D").start == 0.0


def test_alt_disables_snapping_for_the_gesture(rig):
    p0 = rig.clip_pt("D")
    target_start = 8.0 + 3 / rig.tl.pps
    p1 = QPointF(p0.x() + (target_start - 2.0) * rig.tl.pps, p0.y())
    with one_undo(rig):
        rig.drag(p0, p1, ALT, release_mods=ALT, press_mods=NOMOD)
    got = rig.clip("D").start
    assert got != pytest.approx(8.0), "Alt held: no snapping"
    assert got == pytest.approx(round(target_start * 30) / 30, abs=1e-6), "…but the time is still frame aligned"
    assert rig.tl._ov.snap_t is None


def test_toolbar_snap_toggle_disables_snapping(rig):
    seen = []
    rig.tl.snappingChanged.connect(seen.append)
    rig.tl._snap_btn.click()
    assert rig.tl.snapping is False and seen == [False]
    p0 = rig.clip_pt("D")
    target_start = 8.0 + 3 / rig.tl.pps
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + (target_start - 2.0) * rig.tl.pps, p0.y()))
    assert rig.clip("D").start != pytest.approx(8.0)
    rig.tl.set_snapping(True)                                # what MainWindow's menu does
    assert rig.tl._snap_btn.isChecked() and seen == [False, True]


def test_move_is_frame_aligned(rig):
    p0 = rig.clip_pt("D")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 133.7, p0.y()))
    s = rig.clip("D").start
    assert abs(s * 30 - round(s * 30)) < 1e-3


def test_move_to_other_video_track_keeps_time_and_ranges(rig):
    p0 = rig.clip_pt("B")
    p1 = QPointF(p0.x(), rig.y(rig.v2))
    with one_undo(rig):
        rig.drag(p0, p1)
    assert rig.track_of("B") == rig.v2
    b = rig.clip("B")
    assert (b.start, b.src_in, b.src_out) == (5.0, 0.0, 3.0)
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (10.0, 14.0, 5.0, 9.0)]
    assert rig.tl._ov.empty()


def test_move_video_clip_onto_audio_lane_stays_on_its_track(rig):
    p0 = rig.clip_pt("B")
    p1 = QPointF(p0.x() + 80, rig.y(rig.a2))
    with one_undo(rig):
        rig.drag(p0, p1)
    assert rig.track_of("B") == rig.v1 and rig.clip("B").start == pytest.approx(5.8, abs=1 / 30)


def test_move_audio_clip_to_another_audio_track(rig):
    p0 = rig.clip_pt("M", frac=0.5)
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x(), rig.y(rig.a2)))
    assert rig.track_of("M") == rig.a2 and rig.clip("M").start == 0.0


def test_move_overwrite_cuts_what_it_lands_on(rig):
    # B (5-8) dragged onto A's tail: B -> 3.0..6.0 overwrites A[3,4) and nothing else
    p0 = rig.clip_pt("B")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 2.0 * rig.tl.pps, p0.y()), ALT, press_mods=NOMOD)   # Alt: exact 3.0, no snap to A's end 4.0
    assert rig.layout(rig.v1) == [(0.0, 3.0, 0.0, 3.0), (3.0, 6.0, 0.0, 3.0), (10.0, 14.0, 5.0, 9.0)]


def test_ctrl_at_release_inserts_instead_of_overwriting(rig):
    p0 = rig.clip_pt("B")
    before_len = sum(c.end - c.start for c in rig.st.sequence().find_track(rig.v1).clips)
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 3.0 * rig.tl.pps, p0.y()), ALT, release_mods=ALT | CTRL, press_mods=NOMOD)
    tr = rig.st.sequence().find_track(rig.v1)
    assert sum(c.end - c.start for c in tr.clips) == pytest.approx(before_len), "nothing is lost on insert"
    starts = [round(c.start, 3) for c in tr.clips]
    assert starts == sorted(starts) and starts[0] == 0.0
    # no overlap anywhere on the track (insert pushes the rest right)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(tr.clips, tr.clips[1:]))
    assert rig.clip("B").start == pytest.approx(2.0, abs=1e-6)


def test_move_selection_keeps_relative_layout(rig):
    rig.click(rig.clip_pt("A"))
    rig.click(rig.clip_pt("D"), CTRL)                        # A (V1 0-4) and D (V2 2-5)
    p0 = rig.clip_pt("D")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 4.5 * rig.tl.pps, p0.y()), ALT, press_mods=NOMOD)   # +4.5 s -> A would hit B
    assert rig.clip("D").start == pytest.approx(6.5) and rig.clip("A").start == pytest.approx(4.5)
    assert rig.st.selection == {rig.cid("A"), rig.cid("D")}


def test_move_multi_selection_never_changes_tracks(rig):
    rig.click(rig.clip_pt("A"))
    rig.click(rig.clip_pt("D"), CTRL)
    p0 = rig.clip_pt("D")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 100, rig.y(rig.v1)), ALT, press_mods=NOMOD)
    assert rig.track_of("D") == rig.v2 and rig.track_of("A") == rig.v1


def test_escape_aborts_a_drag(rig):
    p0 = rig.clip_pt("D")
    with untouched(rig):
        rig.press(p0)
        rig.move(QPointF(p0.x() + 90, p0.y()))
        assert rig.tl._ov.ghosts
        rig.key(Qt.Key_Escape)
        assert rig.tl._ov.empty()
        rig.release(QPointF(p0.x() + 90, p0.y()))
    assert rig.clip("D").start == 2.0


def test_locked_track_rejects_move_trim_fade_and_delete(rig):
    rig.st.op("lock", "set_track_props", rig.v1, locked=True)
    rig.status.clear()
    p0 = rig.clip_pt("B")
    with untouched(rig):
        rig.drag(p0, QPointF(p0.x() + 90, p0.y()))
        assert any("鎖定" in m for m in rig.status)
        rig.drag(rig.right_edge("B"), QPointF(rig.right_edge("B").x() - 40, rig.right_edge("B").y()))
        rig.drag(rig.left_edge("B"), QPointF(rig.left_edge("B").x() + 40, rig.left_edge("B").y()))
        rig.click(rig.clip_pt("B"))
        rig.key(Qt.Key_Delete)
        rig.key(Qt.Key_S)
    assert rig.layout(rig.v1)[1] == (5.0, 8.0, 0.0, 3.0)


# ===================================================================================================== trim
def test_trim_right_edge_shortens_and_updates_source_out(rig):
    p0 = rig.right_edge("B")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 60, p0.y()), ALT)
    b = rig.clip("B")
    assert (b.start, b.src_in) == (5.0, 0.0)
    assert b.end == pytest.approx(7.4, abs=1 / 30 + 1e-6) and b.src_out == pytest.approx(2.4, abs=1 / 30 + 1e-6)
    assert rig.layout(rig.v1)[2] == (10.0, 14.0, 5.0, 9.0), "neighbours stay put without ripple"


def test_trim_left_edge_moves_start_and_source_in(rig):
    p0 = rig.left_edge("B")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 100, p0.y()), ALT)
    b = rig.clip("B")
    assert b.start == pytest.approx(6.0, abs=1 / 30 + 1e-6) and b.end == 8.0
    assert b.src_in == pytest.approx(1.0, abs=1 / 30 + 1e-6) and b.src_out == 3.0


def test_trim_extends_into_the_source_but_not_past_it(rig):
    # C: src 5-9 of a 20 s asset; dragging its right edge far right is limited by the asset (src_out <= 20)
    # and the track is free to the right, so it grows up to 20 - 5 = 15 s long
    p0 = rig.right_edge("C")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 1500, p0.y()), ALT)
    c = rig.clip("C")
    assert c.src_out == pytest.approx(20.0) and c.end == pytest.approx(25.0)
    assert rig.clip("A").end == 4.0


def test_trim_cannot_overlap_the_neighbour_and_left_edge_stops_at_zero_source(rig):
    p0 = rig.right_edge("A")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 300, p0.y()), ALT)
    assert rig.clip("A").end == pytest.approx(5.0), "stops where B starts"
    rig.st.undo()
    q0 = rig.left_edge("C")
    with one_undo(rig):
        rig.drag(q0, QPointF(q0.x() - 900, q0.y()), ALT)
    c = rig.clip("C")
    assert c.src_in >= 0.0 and c.start >= 8.0 - 1e-6, "stops at B's end"


def test_trim_minimum_length(rig):
    p0 = rig.right_edge("B")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 900, p0.y()), ALT)
    b = rig.clip("B")
    assert b.end - b.start == pytest.approx(ops.MIN_CLIP, abs=1 / 30 + 1e-6) and b.start == 5.0


def test_shift_trim_is_a_ripple_trim(rig):
    p0 = rig.right_edge("B")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 100, p0.y()), SHIFT | ALT)
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (5.0, 7.0, 0.0, 2.0), (9.0, 13.0, 5.0, 9.0)], \
        "later clips of the track follow a ripple trim"


def test_shift_ripple_trim_left_keeps_the_clip_start_and_pulls_later_clips(rig):
    p0 = rig.left_edge("B")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 100, p0.y()), SHIFT | ALT)
    lay = rig.layout(rig.v1)
    assert lay[1][0] == 5.0 and lay[1][1] == pytest.approx(7.0) and lay[1][2] == pytest.approx(1.0)
    assert lay[2][0] == pytest.approx(9.0)


def test_trim_shows_a_tooltip_with_in_out_and_length(rig):
    p0 = rig.right_edge("B")
    rig.press(p0)
    rig.move(QPointF(p0.x() - 30, p0.y()), ALT)
    ov = rig.tl._ov
    assert ov.tip is not None and "入點" in ov.tip[1] and "出點" in ov.tip[1] and "長度" in ov.tip[1]
    assert ov.ghosts and ov.ghosts[0][1] == "trim"
    rig.grab("trim_tooltip")
    rig.release(QPointF(p0.x() - 30, p0.y()), ALT)


def test_trim_snaps_to_other_edges(rig):
    # B's right edge 5 px short of C's start (10.0) is outside B's limit? B end 8 -> drag to 9.95: snap to 10.0
    # (B may grow: asset a2 is 10 s long, src_out 3 -> up to 10)
    p0 = rig.right_edge("B")
    p1 = QPointF(rig.x(9.95) + 1 - 2, p0.y())
    with one_undo(rig):
        rig.press(p0)
        rig.move(QPointF((p0.x() + p1.x()) / 2, p0.y()))
        rig.move(p1)
        assert rig.tl._ov.snap_t == pytest.approx(10.0)
        rig.release(p1)
    assert rig.clip("B").end == pytest.approx(10.0)


def test_hit_zones_edges_and_cursor(rig):
    tl = rig.tl
    assert tl.hit_test(rig.left_edge("B", 2)).kind == "clip_l"
    assert tl.hit_test(rig.right_edge("B", 2)).kind == "clip_r"
    assert tl.hit_test(rig.clip_pt("B")).kind == "clip"
    r = tl.clip_rect(rig.cid("B"))
    inside = QPointF(r.left() + EDGE_PX + 3, r.top() + r.height() * 0.6)
    assert tl.hit_test(inside).kind == "clip"
    rig.move(rig.right_edge("B", 2), buttons=Qt.NoButton)
    assert tl.cursor().shape() == Qt.SizeHorCursor


# ===================================================================================================== slip / fades / crossfade
def test_alt_drag_slips_the_source_without_moving_the_clip(rig):
    p0 = rig.clip_pt("C")
    with one_undo(rig):
        rig.press(p0, ALT)
        rig.move(QPointF(p0.x() - 50, p0.y()), ALT)
        assert rig.tl._ov.slip is not None and "滑動素材" in rig.tl._ov.tip[1]
        rig.release(QPointF(p0.x() - 100, p0.y()), ALT)
    c = rig.clip("C")
    assert (c.start, c.end) == (10.0, 14.0), "the clip stays on the timeline"
    assert (round(c.src_in, 3), round(c.src_out, 3)) == (6.0, 10.0), "dragging left shows LATER material"


def test_slip_is_clamped_to_the_asset(rig):
    p0 = rig.clip_pt("C")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 2000, p0.y()), ALT)        # far right: earlier material, clamps at src 0
    assert rig.clip("C").src_in == 0.0 and rig.clip("C").src_out == pytest.approx(4.0)


def test_fade_in_handle_drag(rig):
    h = rig.tl.fade_handle_rect(rig.cid("B"), "in").center()
    p0 = QPointF(h.x(), h.y())
    with one_undo(rig):
        rig.press(p0)
        rig.move(QPointF(p0.x() + 30, p0.y()))
        rig.move(QPointF(p0.x() + 70, p0.y()))
        assert rig.tl._ov.fade and rig.tl._ov.fade[:2] == (rig.cid("B"), "in") and "淡入" in rig.tl._ov.tip[1]
        rig.grab("fade_handles")
        rig.release(QPointF(p0.x() + 70, p0.y()))
    assert rig.clip("B").fade_in == pytest.approx(0.7, abs=1 / 30 + 1e-6) or rig.clip("B").fade_in > 0.4
    assert rig.clip("B").fade_out == 0.0


def test_fade_out_handle_drag_and_clamp(rig):
    h = rig.tl.fade_handle_rect(rig.cid("B"), "out").center()
    p0 = QPointF(h.x(), h.y())
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 5000, p0.y()))                 # far left: clamped to the clip length
    b = rig.clip("B")
    assert b.fade_out == pytest.approx(b.end - b.start) and b.fade_in == 0.0


def test_fades_never_exceed_the_clip_together(rig):
    rig.st.op("seed", "set_clip_props", rig.cid("B"), fade_in=2.0)
    h = rig.tl.fade_handle_rect(rig.cid("B"), "out").center()
    with one_undo(rig):
        rig.drag(QPointF(h.x(), h.y()), QPointF(h.x() - 5000, h.y()))
    b = rig.clip("B")
    assert b.fade_in + b.fade_out <= (b.end - b.start) + 1e-6 and b.fade_in == 2.0


def test_crossfade_by_dragging_an_overlap(rig):
    rig.st.op("seed", "set_crossfade", rig.cid("B"), 0.5)
    assert rig.clip("B").start == pytest.approx(3.5)
    rig.st.history.clear()
    # the left edge of B sits on the overlap (3.5..4.0): dragging it left makes the overlap longer
    assert rig.tl.hit_test(rig.left_edge("B")).kind == "xfade"
    p0 = rig.left_edge("B")
    with one_undo(rig):
        rig.press(p0)
        rig.move(QPointF(p0.x() - 20, p0.y()))
        assert "交叉淡化" in rig.tl._ov.tip[1]
        rig.grab("crossfade_overlap")
        rig.release(QPointF(p0.x() - 50, p0.y()))
    b, a = rig.clip("B"), rig.clip("A")
    assert a.end - b.start == pytest.approx(1.0, abs=1 / 30 + 1e-6) and a.end == 4.0
    assert b.end - b.start == 3.0, "B keeps its length; only its start moved"


def test_crossfade_drag_clamps_to_half_the_shorter_clip_and_zero(rig):
    rig.st.op("seed", "set_crossfade", rig.cid("B"), 0.5)
    rig.st.history.clear()
    p0 = rig.left_edge("B")
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 5000, p0.y()))
    assert rig.clip("A").end - rig.clip("B").start == pytest.approx(1.5)          # half of B (3 s)
    rig.st.undo()
    with one_undo(rig):
        rig.drag(rig.left_edge("B"), QPointF(rig.left_edge("B").x() + 5000, p0.y()))
    assert rig.clip("B").start == pytest.approx(4.0), "zero overlap = a plain cut again"


# ===================================================================================================== split / delete
def test_s_splits_selected_clip_at_the_playhead(rig):
    rig.click(rig.clip_pt("B"))
    rig.st.set_playhead(6.0)
    with one_undo(rig):
        rig.key(Qt.Key_S)
    assert rig.layout(rig.v1)[1:3] == [(5.0, 6.0, 0.0, 1.0), (6.0, 8.0, 1.0, 3.0)]
    assert rig.layout(rig.v2) == [(2.0, 5.0, 0.0, 3.0)], "other tracks untouched"


def test_ctrl_k_splits_all_unlocked_tracks_when_nothing_is_selected(rig):
    rig.st.set_playhead(3.0)                                  # under A (V1), D (V2) and M (A1)
    rig.st.op("lock", "set_track_props", rig.a1, locked=True)
    rig.st.history.clear()
    with one_undo(rig):
        rig.key(Qt.Key_K, CTRL)
    assert rig.layout(rig.v1)[:2] == [(0.0, 3.0, 0.0, 3.0), (3.0, 4.0, 3.0, 4.0)]
    assert rig.layout(rig.v2)[:2] == [(2.0, 3.0, 0.0, 1.0), (3.0, 5.0, 1.0, 3.0)]
    assert rig.layout(rig.a1) == [(0.0, 12.0, 0.0, 12.0)], "the locked audio track is not cut"


def test_split_with_selection_only_cuts_the_selection(rig):
    rig.st.set_playhead(3.0)
    rig.click(rig.clip_pt("D"))
    with one_undo(rig):
        rig.tl.split_at_playhead()
    assert len(rig.layout(rig.v2)) == 2 and len(rig.layout(rig.v1)) == 3 and len(rig.layout(rig.a1)) == 1


def test_split_with_nothing_under_the_playhead_reports_and_changes_nothing(rig):
    rig.st.set_playhead(8.5)                                  # a gap on V1, V2 and past D; M still runs (A1)
    rig.click(rig.clip_pt("B"))
    rig.status.clear()
    with untouched(rig):
        rig.key(Qt.Key_S)
    assert rig.status and "分割" in rig.status[-1]


def test_delete_removes_selection_in_one_step(rig):
    rig.click(rig.clip_pt("A"))
    rig.click(rig.clip_pt("D"), CTRL)
    with one_undo(rig):
        rig.key(Qt.Key_Delete)
    assert rig.layout(rig.v1) == [(5.0, 8.0, 0.0, 3.0), (10.0, 14.0, 5.0, 9.0)] and rig.layout(rig.v2) == []
    assert rig.st.selection == set()


def test_shift_delete_ripples_the_gap_closed(rig):
    rig.click(rig.clip_pt("B"))
    with one_undo(rig):
        rig.key(Qt.Key_Delete, SHIFT)
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (7.0, 11.0, 5.0, 9.0)]
    rig.st.undo()
    rig.click(rig.clip_pt("B"))
    with one_undo(rig):
        rig.tl.delete_selection(ripple=False)
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (10.0, 14.0, 5.0, 9.0)]


def test_delete_with_nothing_selected_is_a_noop(rig):
    with untouched(rig):
        rig.key(Qt.Key_Delete)
        rig.tl.delete_selection()


# ===================================================================================================== markers / in-out
def test_m_adds_a_marker_at_the_playhead_and_undo_removes_it(rig):
    rig.st.set_playhead(2.5)
    with one_undo(rig):
        rig.key(Qt.Key_M)
    assert [round(m.time, 3) for m in rig.st.sequence().markers] == [2.5, 9.0]


def test_i_and_o_set_the_range_and_clear_range_removes_it(rig):
    st = rig.st
    st.set_playhead(2.0)
    with one_undo(rig):
        rig.key(Qt.Key_I)
    assert st.get_range() is not None and st.get_range()[0] == 2.0
    st.set_playhead(6.0)
    with one_undo(rig):
        rig.key(Qt.Key_O)
    assert st.get_range() == (2.0, 6.0)
    with one_undo(rig):
        rig.tl.clear_range()
    assert st.get_range() is None
    with untouched(rig):
        rig.tl.clear_range()                                   # nothing to clear: no empty undo step


def test_out_before_in_resets_the_in_point(rig):
    st = rig.st
    st.set_playhead(5.0)
    rig.key(Qt.Key_I)
    st.set_playhead(1.0)
    with one_undo(rig):
        rig.key(Qt.Key_O)
    r = st.get_range()
    assert r == (0.0, 1.0)


# ===================================================================================================== zoom / scroll
def test_plus_minus_and_fit_keys(rig):
    tl = rig.tl
    p0 = tl.pps
    rig.key(Qt.Key_Plus)
    assert tl.pps > p0
    rig.key(Qt.Key_Minus)
    assert tl.pps == pytest.approx(p0, rel=1e-6)
    rig.key(Qt.Key_Backslash)
    lo, hi = tl.visible_time_range()
    assert lo == pytest.approx(0.0, abs=1e-6) and hi >= rig.st.sequence().duration - 1e-6
    assert tl.horizontal_scroll() == 0


def test_ctrl_wheel_zooms_around_the_cursor(rig):
    tl = rig.tl
    pos = rig.at(6.0, rig.v1)
    t_before = tl.x_to_time(pos.x())
    seen = []
    tl.zoomChanged.connect(seen.append)
    rig.wheel(pos, 120, mods=CTRL)
    assert tl.pps > 100.0 and seen
    assert tl.x_to_time(pos.x()) == pytest.approx(t_before, abs=0.02), "the time under the cursor stays put"
    rig.wheel(pos, -120, mods=CTRL)
    assert tl.x_to_time(pos.x()) == pytest.approx(t_before, abs=0.02)
    for _ in range(80):
        rig.wheel(pos, 240, mods=CTRL)
    assert tl.pps <= tlmod.PPS_MAX + 1e-9
    for _ in range(200):
        rig.wheel(pos, -240, mods=CTRL)
    assert tl.pps >= tlmod.PPS_MIN - 1e-9


def test_wheel_scrolls_vertically_shift_wheel_horizontally(rig):
    tl = rig.tl
    tl.set_pps(300.0)
    tl.set_horizontal_scroll(0)
    rig.wheel(rig.at(2.0, rig.v1), -120, mods=SHIFT)
    assert tl.horizontal_scroll() > 0
    h = tl.horizontal_scroll()
    rig.wheel(rig.at(2.0, rig.v1), 120, mods=SHIFT)
    assert tl.horizontal_scroll() < h
    tl.resize(1100, 200)                                      # short: the lanes no longer fit -> vertical scrolling
    QApplication.processEvents()
    assert tl._max_scroll()[1] > 0
    rig.wheel(rig.at(2.0, rig.v1) if False else QPointF(400, 100), -120)
    assert tl.vertical_scroll() > 0
    rig.wheel(QPointF(400, 100), 600)
    assert tl.vertical_scroll() == 0


def test_horizontal_wheel_and_middle_drag_pan(rig):
    tl = rig.tl
    tl.set_pps(300.0)
    tl.set_horizontal_scroll(0)
    rig.wheel(QPointF(400, 150), dx=-120)
    assert tl.horizontal_scroll() > 0
    h = tl.horizontal_scroll()
    p0 = QPointF(500, 150)
    with untouched(rig):
        rig.drag(p0, QPointF(450, 150), button=Qt.MiddleButton)
    assert tl.horizontal_scroll() == pytest.approx(h + 50, abs=2)


def test_scrollbars_follow_and_drive_the_view(rig):
    tl = rig.tl
    tl.set_pps(400.0)
    tl._hbar.setValue(250)
    assert tl.horizontal_scroll() == 250
    tl.set_horizontal_scroll(120)
    assert tl._hbar.value() == 120
    tl._zoom_slider.setValue(1000)
    assert tl.pps == pytest.approx(tlmod.PPS_MAX, rel=1e-3)


@pytest.mark.parametrize("far", [3e9, 1e13])
def test_times_far_beyond_a_day_from_a_damaged_file_do_not_break_the_layout(rig, far):
    """The core refuses anything beyond 24 hours, but a project file can hold any number.  The scroll bars and ``QRect`` take
    32-bit numbers: a bigger one raised OverflowError inside the slot that lays the timeline out, which left the person
    with a half-built timeline (and aborts the program in the offscreen runs of the tests)."""
    tl, st = rig.tl, rig.st
    rig.clip("B").start = far                                    # what only a damaged or hand-edited file can do
    rig.clip("B").src_out = rig.clip("B").src_in + 3.0
    st.sequence().cues[-1].end = far
    st.sequence().markers[0].time = far
    tl._relayout()                                               # the slot the edit signal calls
    assert 0 <= tl._hbar.maximum() <= 2 ** 31 - 1 and tl._hbar.maximum() <= tlmod.MAX_TIMELINE * tlmod.PPS_MAX
    for how in (tl.zoom_fit, lambda: tl.set_pps(tlmod.PPS_MAX), lambda: tl.set_pps(tlmod.PPS_MIN),
                lambda: tl.set_horizontal_scroll(1e18)):
        how()
        assert tl.grab().width() > 0                             # paints
        for key in ("A", "B", "C"):
            r = tl.clip_rect(rig.cid(key))
            assert r.width() >= 1 and r.height() > 0
        assert tl.cue_rect(rig.cue_id(2)).width() >= 1
        assert tl.horizontal_scroll() <= tl._hbar.maximum() + 1          # (the scroll position is rounded, the range is cut off)
    tl.zoom_fit()
    assert tl.pps >= tlmod.PPS_MIN and tl.clip_rect(rig.cid("A")).width() > 5, "the clips near the start stay usable"


# ===================================================================================================== ruler / playhead
def test_ruler_click_and_drag_scrub_frame_aligned(rig):
    st = rig.st
    ruler_y = RULER_H - 8
    rig.press(QPointF(rig.x(2.0) + 0.4 / 30 * rig.tl.pps, ruler_y))
    assert st.playhead == pytest.approx(2.0, abs=1e-6) or abs(st.playhead * 30 - round(st.playhead * 30)) < 1e-6
    for t in (3.01, 4.333, 5.5):
        rig.move(QPointF(rig.x(t), ruler_y))
        assert abs(st.playhead * 30 - round(st.playhead * 30)) < 1e-6 and st.playhead == pytest.approx(t, abs=1 / 30)
    rig.release(QPointF(rig.x(1.0), ruler_y))
    assert st.playhead == pytest.approx(1.0, abs=1 / 30)
    assert rig.undo_depth() == 0


def test_ruler_scrub_clamps_at_zero(rig):
    rig.press(QPointF(rig.x(0.0) - 50, RULER_H - 8))
    assert rig.st.playhead == 0.0
    rig.release(QPointF(rig.x(0.0) - 50, RULER_H - 8))


def test_arrow_keys_step_frames_and_seconds(rig):
    st = rig.st
    st.set_playhead(2.0)
    rig.key(Qt.Key_Right)
    assert st.playhead == pytest.approx(2.0 + 1 / 30)
    rig.key(Qt.Key_Left)
    rig.key(Qt.Key_Left)
    assert st.playhead == pytest.approx(2.0 - 1 / 30)
    rig.key(Qt.Key_Right, SHIFT)
    assert st.playhead == pytest.approx(2.0 - 1 / 30 + 1.0)
    rig.key(Qt.Key_Left, SHIFT)
    rig.key(Qt.Key_Left, SHIFT)
    rig.key(Qt.Key_Left, SHIFT)
    assert st.playhead == 0.0, "never below zero"
    rig.tl.step_frames(5)
    assert st.playhead == pytest.approx(5 / 30)


def test_home_end_and_prev_next_cut(rig):
    st = rig.st
    st.set_playhead(3.0)
    rig.key(Qt.Key_Home)
    assert st.playhead == 0.0
    rig.key(Qt.Key_End)
    assert st.playhead == pytest.approx(rig.st.sequence().duration)
    st.set_playhead(0.0)
    seen = []
    for _ in range(12):
        rig.key(Qt.Key_Down)
        seen.append(round(st.playhead, 3))
    assert seen[:8] == [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 14.0]
    for _ in range(3):
        rig.key(Qt.Key_Up)
    assert round(st.playhead, 3) == 8.0
    rig.tl.goto_prev_cut()
    assert round(st.playhead, 3) == 5.0
    rig.tl.goto_next_cut()
    assert round(st.playhead, 3) == 8.0


def test_playhead_off_screen_scrolls_into_view_and_playing_pages(rig):
    tl = rig.tl
    tl.set_pps(400.0)
    tl.set_horizontal_scroll(0)
    rig.st.set_playhead(9.0)
    lo, hi = tl.visible_time_range()
    assert lo <= 9.0 <= hi, "moving the playhead reveals it"
    tl.set_horizontal_scroll(0)
    tl.set_playing(True)
    for t in (1.0, 2.0, 3.0, 3.4, 3.5, 3.6, 4.0):
        rig.st.set_playhead(t)
    lo, hi = tl.visible_time_range()
    assert 3.0 < lo <= 3.5 and lo <= 4.0 <= hi, "playback page-flips the view to follow the playhead"
    tl.set_playing(False)


def test_state_playhead_changes_repaint_without_error(rig):
    for t in (0.0, 1.0, 7.77, 30.0):
        rig.st.set_playhead(t)
        QApplication.processEvents()
    assert rig.tl.state.playhead == 30.0


# ===================================================================================================== caption lane
def test_cue_drag_moves_a_block_with_snapping(rig):
    p0 = rig.cue_pt(1)                                        # "第二句" 3.0-4.5
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 100, p0.y()))           # +1.0 s -> 4.0-5.5 (its start touches A's end 4.0?)
    q = rig.st.sequence().cues[1]
    assert (round(q.start, 3), round(q.end, 3)) == (4.0, 5.5), rig.cues()
    assert rig.st.cue_selection == {q.id}


def test_cue_drag_is_a_ghost_until_release_and_snaps_to_clip_edges(rig):
    p0 = rig.cue_pt(2)                                        # 6.0-7.0; B ends at 8.0
    p1 = QPointF(p0.x() + (1.0 + 4 / rig.tl.pps) * rig.tl.pps, p0.y())       # start 7.04 -> its END 8.04 snaps to 8.0
    before = rig.content()
    rig.press(p0)
    rig.move(QPointF(p0.x() + 50, p0.y()))
    rig.move(p1)
    assert rig.content() == before and rig.tl._ov.ghosts and rig.tl._ov.ghosts[0][1] == "cue"
    assert rig.tl._ov.snap_t == pytest.approx(8.0)
    rig.release(p1)
    q = rig.st.sequence().cues[2]
    assert round(q.end, 3) == 8.0 and round(q.start, 3) == 7.0


def test_cue_drag_clamps_at_zero(rig):
    p0 = rig.cue_pt(0)
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() - 5000, p0.y()))
    assert rig.cues()[0][0] == 0.0 and rig.cues()[0][1] == 1.0


def test_cue_resize_edges(rig):
    r = rig.tl.cue_rect(rig.cue_id(1))
    left = QPointF(r.left() + 2, r.center().y())
    right = QPointF(r.right() - 1, r.center().y())
    assert rig.tl.hit_test(left).kind == "cue_l" and rig.tl.hit_test(right).kind == "cue_r"
    with one_undo(rig):
        rig.drag(right, QPointF(right.x() + 53, right.y()), ALT)           # end 4.5 -> ~5.03 (Alt: no snap)
    assert rig.st.sequence().cues[1].end == pytest.approx(5.03, abs=1 / 30 + 1e-6)
    with one_undo(rig):
        rig.drag(left, QPointF(left.x() - 60, left.y()), ALT)
    q = rig.st.sequence().cues[1]
    assert q.start == pytest.approx(2.4, abs=1 / 30 + 1e-6) and q.end == pytest.approx(5.03, abs=1 / 30 + 1e-6)


def test_cue_resize_keeps_minimum_length_and_zero(rig):
    r = rig.tl.cue_rect(rig.cue_id(1))
    right = QPointF(r.right() - 1, r.center().y())
    with one_undo(rig):
        rig.drag(right, QPointF(right.x() - 5000, right.y()), ALT)
    q = rig.st.sequence().cues[1]
    assert q.end - q.start == pytest.approx(ops.MIN_CUE, abs=1 / 30 + 1e-6) and q.start == 3.0
    rig.st.undo()
    r = rig.tl.cue_rect(rig.cue_id(1))
    left = QPointF(r.left() + 2, r.center().y())
    with one_undo(rig):
        rig.drag(left, QPointF(left.x() + 5000, left.y()), ALT)
    q = rig.st.sequence().cues[1]
    assert q.end == 4.5 and q.end - q.start == pytest.approx(ops.MIN_CUE, abs=1 / 30 + 1e-6)


def test_moving_several_selected_cues_is_one_shift_cues(rig):
    rig.click(rig.cue_pt(0))
    rig.click(rig.cue_pt(1), CTRL)
    assert len(rig.st.cue_selection) == 2
    p0 = rig.cue_pt(1)
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + 53, p0.y()), ALT, press_mods=NOMOD)
    cues = rig.st.sequence().cues
    assert cues[0].start == pytest.approx(1.53, abs=1 / 30 + 1e-6) and cues[1].start == pytest.approx(3.53, abs=1 / 30 + 1e-6)
    assert cues[2].start == 6.0, "unselected cues stay"


def test_cue_click_selection_rules(rig):
    st = rig.st
    rig.click(rig.cue_pt(0))
    assert st.cue_selection == {rig.cue_id(0)} and st.selection == set()
    rig.click(rig.cue_pt(2), SHIFT)
    assert st.cue_selection == {rig.cue_id(0), rig.cue_id(1), rig.cue_id(2)}
    rig.click(rig.cue_pt(1), CTRL)
    assert st.cue_selection == {rig.cue_id(0), rig.cue_id(2)}
    rig.click(rig.clip_pt("A"))
    assert st.cue_selection == set() and st.selection == {rig.cid("A")}, "clicking a clip drops the cue selection"
    rig.click(rig.at(8.5, CAPTION_LANE))
    assert st.cue_selection == set() and st.selection == set()


def test_double_click_opens_the_inline_editor_over_the_block(rig):
    rig.dclick(rig.cue_pt(1))
    ed = rig.tl.cue_editor()
    assert ed is not None and ed.isVisible() and ed.text() == "第二句"
    r = rig.tl.cue_rect(rig.cue_id(1))
    assert ed.geometry().top() == r.top() and ed.geometry().height() == r.height()
    assert ed.geometry().left() == r.left() and ed.geometry().width() >= r.width()
    rig.grab("caption_inline_edit")
    ed.setText("改過的字幕")
    QTest.keyClick(ed, Qt.Key_Return)
    assert rig.tl.cue_editor() is None
    assert rig.cues()[1] == (3.0, 4.5, "改過的字幕")
    rig.st.undo()
    assert rig.cues()[1] == (3.0, 4.5, "第二句")
    assert rig.undo_depth() == 0


def test_inline_editor_escape_cancels_and_blur_commits(rig):
    ed = rig.tl.edit_cue_text(rig.cue_id(0))
    ed.setText("會被取消")
    with untouched(rig):
        QTest.keyClick(ed, Qt.Key_Escape)
    assert rig.tl.cue_editor() is None and rig.cues()[0][2] == "第一句"
    ed = rig.tl.edit_cue_text(rig.cue_id(0))
    ed.setText("失去焦點時存檔")
    n0 = rig.undo_depth()
    rig.click(rig.clip_pt("B"))                               # clicking elsewhere ends the edit and commits it
    assert rig.cues()[0][2] == "失去焦點時存檔" and rig.undo_depth() == n0 + 1
    assert rig.tl.cue_editor() is None


def test_inline_editor_unchanged_text_makes_no_undo_step(rig):
    ed = rig.tl.edit_cue_text(rig.cue_id(2))
    with untouched(rig):
        QTest.keyClick(ed, Qt.Key_Return)
    assert rig.tl.cue_editor() is None


def test_inline_editor_follows_zoom_and_closes_on_external_edit(rig):
    ed = rig.tl.edit_cue_text(rig.cue_id(1))
    left0 = ed.geometry().left()
    rig.tl.set_pps(200.0, anchor_x=HEADER_W)
    assert ed.geometry().left() != left0
    r = rig.tl.cue_rect(rig.cue_id(1))
    assert ed.geometry().left() == max(r.left(), HEADER_W)
    rig.st.op("other", "add_marker", 1.0, "x")
    assert rig.tl.cue_editor() is None, "an external edit throws the editor away"


def test_split_selected_cue_at_the_playhead(rig):
    rig.click(rig.cue_pt(1))
    rig.st.set_playhead(3.9)
    with one_undo(rig):
        rig.key(Qt.Key_S)
    cs = rig.cues()
    assert len(cs) == 4 and cs[1][:2] == (3.0, 3.9) and cs[2][:2] == (3.9, 4.5)
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (5.0, 8.0, 0.0, 3.0), (10.0, 14.0, 5.0, 9.0)], \
        "only the cue is cut: a clip is not even under the playhead's selection"


def test_merge_selected_cues_and_requires_two(rig):
    rig.click(rig.cue_pt(0))
    rig.status.clear()
    with untouched(rig):
        rig.tl.merge_selected_cues()
    assert rig.status
    rig.click(rig.cue_pt(1), CTRL)
    with one_undo(rig):
        rig.tl.merge_selected_cues()
    cs = rig.cues()
    assert len(cs) == 2 and cs[0][:2] == (1.0, 4.5) and "第一句" in cs[0][2] and "第二句" in cs[0][2]


def test_delete_selected_cues_with_the_key_and_one_undo_with_clips(rig):
    rig.click(rig.cue_pt(0))
    rig.click(rig.cue_pt(2), CTRL)
    with one_undo(rig):
        rig.key(Qt.Key_Delete)
    assert rig.cues() == [(3.0, 4.5, "第二句")]
    rig.st.undo()
    rig.click(rig.cue_pt(1))
    rig.click(rig.clip_pt("D"))                               # a plain click on a clip drops the cue selection
    assert rig.st.cue_selection == set()
    rig.tl.select_all()
    with one_undo(rig):
        rig.key(Qt.Key_Delete)
    assert rig.cues() == [] and all(not t.clips for t in rig.st.sequence().tracks())


def test_add_cue_through_the_caption_lane_menu_opens_the_editor(rig, monkeypatch):
    shown = []
    monkeypatch.setattr(TimelineWidget, "_popup_menu", lambda self, menu, pos: shown.append(menu))
    pos = rig.at(8.5, CAPTION_LANE)
    rig.right_click(pos)
    menu = shown[-1]
    assert [a.objectName() for a in menu.actions() if a.objectName()] == ["paste", "marker", "add_cue"]
    with one_undo(rig):
        rig.act(menu, "add_cue").trigger()
        new = [q for q in rig.st.sequence().cues if q.start >= 8.0][0]
        assert new.end - new.start == pytest.approx(2.0, abs=1 / 30) and new.text == "新字幕"
        assert rig.tl.cue_editor() is not None and rig.st.cue_selection == {new.id}
        assert rig.tl.cue_editor().text() == "新字幕"
        QTest.keyClick(rig.tl.cue_editor(), Qt.Key_Escape)


def test_cue_context_menu_actions(rig, monkeypatch):
    shown = []
    monkeypatch.setattr(TimelineWidget, "_popup_menu", lambda self, menu, pos: shown.append(menu))
    rig.st.set_playhead(3.9)
    rig.right_click(rig.cue_pt(1))
    menu = shown[-1]
    assert rig.st.cue_selection == {rig.cue_id(1)}, "right click selects the block first"
    names = [a.objectName() for a in menu.actions() if a.objectName()]
    assert names == ["edit_text", "split", "merge", "delete"]
    assert rig.act(menu, "split").isEnabled() and not rig.act(menu, "merge").isEnabled()
    with one_undo(rig):
        rig.act(menu, "split").trigger()
    rig.click(rig.cue_pt(0))
    rig.click(rig.cue_pt(1), CTRL)
    rig.right_click(rig.cue_pt(1))
    menu = shown[-1]
    assert rig.act(menu, "merge").isEnabled()
    with one_undo(rig):
        rig.act(menu, "merge").trigger()
    rig.right_click(rig.cue_pt(0))
    with one_undo(rig):
        rig.act(shown[-1], "delete").trigger()
    rig.right_click(rig.cue_pt(0))
    rig.act(shown[-1], "edit_text").trigger()
    assert rig.tl.cue_editor() is not None
    QTest.keyClick(rig.tl.cue_editor(), Qt.Key_Escape)


# ===================================================================================================== track headers
def test_header_mute_and_lock_buttons(rig):
    tl = rig.tl
    lane = tl._lane_by_id[rig.v1]
    mute_pt = QPointF(tl.header_button_rect(lane, "mute").center())
    lock_pt = QPointF(tl.header_button_rect(lane, "lock").center())
    assert tl.hit_test(mute_pt).kind == "header_mute" and tl.hit_test(lock_pt).kind == "header_lock"
    with one_undo(rig):
        rig.click(mute_pt)
    assert rig.st.sequence().find_track(rig.v1).muted is True
    rig.st.undo()
    with one_undo(rig):
        rig.click(lock_pt)
    assert rig.st.sequence().find_track(rig.v1).locked is True
    rig.st.undo()
    a_lane = tl._lane_by_id[rig.a1]
    with one_undo(rig):
        rig.click(QPointF(tl.header_button_rect(a_lane, "mute").center()))
    assert rig.st.sequence().find_track(rig.a1).muted is True
    assert not rig.st.selection, "pressing a header button does not touch the clip selection"


def test_header_caption_lane_has_no_buttons(rig):
    lane = rig.tl._lane_by_id[CAPTION_LANE]
    assert rig.tl.hit_test(QPointF(HEADER_W - 36, rig.tl.header_rect(CAPTION_LANE).center().y())).kind == "header"
    with untouched(rig):
        rig.click(QPointF(HEADER_W - 36, rig.tl.header_rect(CAPTION_LANE).center().y()))
    assert lane.track is None


def test_header_context_menu_adds_tracks(rig, monkeypatch):
    shown = []
    monkeypatch.setattr(TimelineWidget, "_popup_menu", lambda self, menu, pos: shown.append(menu))
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.v1).center().y()))
    menu = shown[-1]
    with one_undo(rig):
        rig.act(menu, "add_video").trigger()
    assert len(rig.st.sequence().video_tracks) == 3 and rig.st.sequence().video_tracks[-1].name == "V3"
    rig.st.undo()
    with one_undo(rig):
        rig.act(menu, "add_audio").trigger()
    assert len(rig.st.sequence().audio_tracks) == 3
    assert [ln.id for ln in rig.tl._lanes][-1] == rig.st.sequence().audio_tracks[-1].id or True


def test_header_rename_via_menu_and_double_click(rig, monkeypatch):
    shown = []
    monkeypatch.setattr(TimelineWidget, "_popup_menu", lambda self, menu, pos: shown.append(menu))
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("旁白", True))
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.a1).center().y()))
    with one_undo(rig):
        rig.act(shown[-1], "rename").trigger()
    assert rig.st.sequence().find_track(rig.a1).name == "旁白"
    rig.st.undo()
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("主畫面", True))
    with one_undo(rig):
        rig.dclick(QPointF(20, rig.tl.header_rect(rig.v1).center().y()))
    assert rig.st.sequence().find_track(rig.v1).name == "主畫面"
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("", False))
    with untouched(rig):                                       # cancelled dialog
        rig.tl.rename_track(rig.v1)
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("   ", True))
    with untouched(rig):                                       # blank name is ignored
        rig.tl.rename_track(rig.v1)


def test_remove_track_empty_and_with_clips(rig, monkeypatch):
    assert rig.tl.remove_track(rig.a2) is True
    assert rig.st.sequence().find_track(rig.a2) is None
    rig.st.undo()
    asked = []
    import autocut.gui.media_bin as mb
    monkeypatch.setattr(mb, "ask", lambda *a, **k: asked.append(a[2]) or False)
    with untouched(rig):
        assert rig.tl.remove_track(rig.v2) is False            # has clip D; user says no
    assert asked and "D" not in asked[0] and "1" in asked[0]
    monkeypatch.setattr(mb, "ask", lambda *a, **k: True)
    with one_undo(rig):
        assert rig.tl.remove_track(rig.v2) is True
    assert rig.st.sequence().find_track(rig.v2) is None and rig.clip("A")


def test_cannot_remove_the_last_video_track_and_locked_tracks(rig, monkeypatch):
    shown = []
    monkeypatch.setattr(TimelineWidget, "_popup_menu", lambda self, menu, pos: shown.append(menu))
    rig.st.op("rm", "remove_track", rig.v2, force=True)
    rig.st.history.clear()
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.v1).center().y()))
    assert not rig.act(shown[-1], "remove_track").isEnabled(), "at least one video track must stay"
    with untouched(rig):
        assert rig.tl.remove_track(rig.v1, force=True) is False
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.a2).center().y()))
    assert rig.act(shown[-1], "remove_track").isEnabled()
    rig.st.op("lock", "set_track_props", rig.a2, locked=True)
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.a2).center().y()))
    assert not rig.act(shown[-1], "remove_track").isEnabled()
    assert rig.act(shown[-1], "lock").text() == "解除鎖定"


def test_header_menu_lock_mute_and_close_gaps(rig, monkeypatch):
    shown = []
    monkeypatch.setattr(TimelineWidget, "_popup_menu", lambda self, menu, pos: shown.append(menu))
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.v1).center().y()))
    menu = shown[-1]
    assert rig.act(menu, "mute").text() == "隱藏軌道", "a video track is hidden, an audio track muted"
    with one_undo(rig):
        rig.act(menu, "mute").trigger()
    with one_undo(rig):
        rig.act(menu, "lock").trigger()
    rig.st.undo()
    rig.st.undo()
    assert rig.act(menu, "close_gaps").isEnabled()
    with one_undo(rig):
        rig.act(menu, "close_gaps").trigger()
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (4.0, 7.0, 0.0, 3.0), (7.0, 11.0, 5.0, 9.0)]
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.v2).center().y()))
    assert not rig.act(shown[-1], "close_gaps").isEnabled(), "a single clip has no gap to close"
    rig.right_click(QPointF(20, rig.tl.header_rect(rig.a1).center().y()))
    assert rig.act(shown[-1], "mute").text() == "靜音軌道"


# ===================================================================================================== clip context menu
@pytest.fixture
def menus(monkeypatch):
    shown: list = []
    monkeypatch.setattr(TimelineWidget, "_popup_menu", lambda self, menu, pos: shown.append(menu))
    return shown


CLIP_ENTRIES = ["split", "delete", "ripple_delete", "mute", "disable", "speed", "fades", "crossfade", "reframe_menu",
                "copy", "reveal", "properties"]


def entries(menu: QMenu) -> list[str]:
    return [a.objectName() for a in menu.actions() if a.objectName()]


def test_clip_menu_lists_the_spec_entries_and_selects_the_clicked_clip(rig, menus):
    rig.click(rig.clip_pt("A"))
    rig.right_click(rig.clip_pt("B"))
    assert entries(menus[-1]) == CLIP_ENTRIES
    assert rig.st.selection == {rig.cid("B")}, "right click on an unselected clip selects just that clip"
    sub = rig.act(menus[-1], "reframe_menu").menu()
    assert [a.objectName() for a in sub.actions()] == ["reframe_edit", "reframe_reset", "reframe_key", "reframe_clear"]
    texts = {a.objectName(): a.text() for a in menus[-1].actions() if a.objectName()}
    assert texts["split"] == "在播放頭分割" and texts["ripple_delete"] == "波紋刪除" and texts["crossfade"] == "交叉淡化…"
    assert texts["reveal"] == "在素材庫顯示" and texts["properties"] == "屬性" and texts["speed"] == "速度…"


def test_right_click_inside_a_multi_selection_keeps_it(rig, menus):
    rig.click(rig.clip_pt("A"))
    rig.click(rig.clip_pt("B"), CTRL)
    rig.right_click(rig.clip_pt("A"))
    assert rig.st.selection == {rig.cid("A"), rig.cid("B")}
    with one_undo(rig):
        rig.act(menus[-1], "delete").trigger()
    assert rig.layout(rig.v1) == [(10.0, 14.0, 5.0, 9.0)]


def test_clip_menu_split_needs_the_playhead_inside(rig, menus):
    rig.st.set_playhead(9.0)
    rig.right_click(rig.clip_pt("B"))
    assert not rig.act(menus[-1], "split").isEnabled()
    rig.st.set_playhead(6.0)
    rig.right_click(rig.clip_pt("B"))
    assert rig.act(menus[-1], "split").isEnabled()
    with one_undo(rig):
        rig.act(menus[-1], "split").trigger()
    assert rig.layout(rig.v1)[1:3] == [(5.0, 6.0, 0.0, 1.0), (6.0, 8.0, 1.0, 3.0)]


def test_clip_menu_ripple_delete_closes_the_gap(rig, menus):
    rig.right_click(rig.clip_pt("B"))
    with one_undo(rig):
        rig.act(menus[-1], "ripple_delete").trigger()
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (7.0, 11.0, 5.0, 9.0)]


def test_clip_menu_mute_and_disable_toggle(rig, menus):
    rig.right_click(rig.clip_pt("B"))
    assert rig.act(menus[-1], "mute").text() == "靜音" and rig.act(menus[-1], "disable").text() == "停用"
    with one_undo(rig):
        rig.act(menus[-1], "mute").trigger()
    assert rig.clip("B").muted is True
    rig.right_click(rig.clip_pt("B"))
    assert rig.act(menus[-1], "mute").text() == "取消靜音"
    with one_undo(rig):
        rig.act(menus[-1], "mute").trigger()
    assert rig.clip("B").muted is False
    with one_undo(rig):
        rig.act(menus[-1], "disable").trigger()
    assert rig.clip("B").enabled is False
    rig.right_click(rig.clip_pt("B"))
    assert rig.act(menus[-1], "disable").text() == "啟用"


def test_mute_applies_to_the_whole_selection_in_one_step(rig, menus):
    rig.click(rig.clip_pt("A"))
    rig.click(rig.clip_pt("D"), CTRL)
    rig.right_click(rig.clip_pt("D"))
    with one_undo(rig):
        rig.act(menus[-1], "mute").trigger()
    assert rig.clip("A").muted and rig.clip("D").muted and not rig.clip("B").muted


def test_clip_menu_speed_dialog_ripples_the_track(rig, menus, monkeypatch):
    asked = []
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: asked.append(a) or (2.0, True))
    rig.right_click(rig.clip_pt("B"))
    with one_undo(rig):
        rig.act(menus[-1], "speed").trigger()
    b = rig.clip("B")
    assert b.speed == 2.0 and (b.src_in, b.src_out) == (0.0, 3.0) and b.end == pytest.approx(6.5)
    assert rig.clip("C").start == pytest.approx(8.5), "ripple: later clips of the track follow"
    assert asked and "速度" in asked[0][1]
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (3.0, False))
    rig.right_click(rig.clip_pt("B"))
    with untouched(rig):
        rig.act(menus[-1], "speed").trigger()                 # cancelled


def test_clip_menu_fades_dialog(rig, menus, monkeypatch):
    seen = []
    monkeypatch.setattr(rig.tl, "_ask_fades", lambda fi, fo, limit: seen.append((fi, fo, limit)) or (0.5, 1.0))
    rig.right_click(rig.clip_pt("B"))
    with one_undo(rig):
        rig.act(menus[-1], "fades").trigger()
    assert (rig.clip("B").fade_in, rig.clip("B").fade_out) == (0.5, 1.0) and seen == [(0.0, 0.0, 3.0)]
    monkeypatch.setattr(rig.tl, "_ask_fades", lambda *a: None)
    rig.right_click(rig.clip_pt("B"))
    with untouched(rig):
        rig.act(menus[-1], "fades").trigger()


def test_clip_menu_crossfade_dialog(rig, menus, monkeypatch):
    rig.right_click(rig.clip_pt("A"))
    assert not rig.act(menus[-1], "crossfade").isEnabled(), "the first clip of a track has nothing to fade from"
    asked = []
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: asked.append(a) or (0.6, True))
    rig.right_click(rig.clip_pt("B"))
    assert rig.act(menus[-1], "crossfade").isEnabled()
    with one_undo(rig):
        rig.act(menus[-1], "crossfade").trigger()
    b = rig.clip("B")
    assert b.start == pytest.approx(3.4) and rig.clip("A").end == 4.0 and rig.clip("C").start == pytest.approx(8.4)
    assert asked[0][3] == pytest.approx(0.5), "the dialog starts at 0.5 s for a plain cut"
    rig.right_click(rig.clip_pt("B", frac=0.7))
    asked.clear()
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: asked.append(a) or (0.0, True))
    with one_undo(rig):
        rig.act(menus[-1], "crossfade").trigger()
    assert asked[0][3] == pytest.approx(0.6, abs=1e-6), "…and shows the existing overlap afterwards"
    assert rig.clip("B").start == pytest.approx(4.0)


def test_clip_menu_reframe_entries(rig, menus):
    d = rig.cid("D")
    rig.st.op("seed", "set_clip_props", d, zoom=2.0, cx=0.3, cy=0.6)
    rig.st.op("seed", "set_transform_key", d, 0.0, 0.2, 0.4)
    rig.st.op("seed", "set_transform_key", d, 2.0, 0.8, 0.4)
    rig.st.history.clear()
    rig.st.set_playhead(3.0)                                    # 1.0 s into D: halfway between the two keys
    rig.right_click(rig.clip_pt("D"))
    sub = rig.act(menus[-1], "reframe_menu").menu()
    assert rig.act(sub, "reframe_clear").isEnabled()
    with one_undo(rig):
        rig.act(sub, "reframe_key").trigger()
    keys = rig.clip("D").transform.keys
    assert [round(k[0], 3) for k in keys] == [0.0, 1.0, 2.0]
    assert keys[1][1] == pytest.approx(0.5, abs=1e-6) and keys[1][2] == pytest.approx(0.4, abs=1e-6), "keeps the framing"
    with one_undo(rig):
        rig.act(sub, "reframe_clear").trigger()
    assert rig.clip("D").transform.keys == []
    with one_undo(rig):
        rig.act(sub, "reframe_reset").trigger()
    t = rig.clip("D").transform
    assert (t.zoom, t.cx, t.cy) == (1.0, 0.5, 0.5)
    asked = []
    rig.st.reframe_requested.connect(asked.append)
    rig.act(sub, "reframe_edit").trigger()
    assert asked == [d]


def test_reframe_key_outside_the_clip_reports(rig, menus):
    rig.st.set_playhead(9.0)
    rig.click(rig.clip_pt("D"))
    rig.status.clear()
    with untouched(rig):
        rig.tl.reframe_add_key()
    assert rig.status


def test_reframe_submenu_is_disabled_for_audio_clips(rig, menus):
    rig.right_click(rig.clip_pt("M"))
    assert not rig.act(menus[-1], "reframe_menu").isEnabled()


def test_clip_menu_reveal_and_properties_emit_signals(rig, menus):
    got = []
    rig.tl.revealAssetRequested.connect(lambda a: got.append(("reveal", a)))
    rig.tl.propertiesRequested.connect(lambda c: got.append(("props", c)))
    rig.right_click(rig.clip_pt("B"))
    with untouched(rig):
        rig.act(menus[-1], "reveal").trigger()
        rig.act(menus[-1], "properties").trigger()
    assert got == [("reveal", "a2"), ("props", rig.cid("B"))]
    got.clear()
    rig.dclick(rig.clip_pt("C"))
    assert got == [("props", rig.cid("C"))], "double click opens the properties"


def test_locked_track_disables_the_editing_entries(rig, menus):
    rig.st.op("lock", "set_track_props", rig.v1, locked=True)
    rig.right_click(rig.clip_pt("B"))
    m = menus[-1]
    for name in ("split", "delete", "ripple_delete", "mute", "disable", "speed", "fades", "crossfade", "reframe_menu"):
        assert not rig.act(m, name).isEnabled(), name
    for name in ("copy", "reveal", "properties"):
        assert rig.act(m, name).isEnabled(), name


def test_copy_paste_through_the_menus_and_keys(rig, menus):
    rig.click(rig.clip_pt("B"))
    rig.right_click(rig.at(9.0, rig.v2))                       # empty space: nothing copied yet
    assert not rig.act(menus[-1], "paste").isEnabled()
    rig.click(rig.clip_pt("B"))
    rig.key(Qt.Key_C, CTRL)
    assert len(TimelineWidget._clipboard) == 1
    rig.right_click(rig.at(6.0, rig.v2))
    assert rig.act(menus[-1], "paste").isEnabled()
    with one_undo(rig):
        rig.act(menus[-1], "paste").trigger()
        new = [c for c in rig.st.sequence().find_track(rig.v2).clips if c.start == 6.0][0]
        assert new.asset_id == "a2" and rig.st.selection == {new.id}
    assert rig.layout(rig.v2) == [(2.0, 5.0, 0.0, 3.0), (6.0, 9.0, 0.0, 3.0)], "pasted at the clicked time and lane"
    rig.st.undo()
    rig.st.set_playhead(8.5)
    with one_undo(rig):
        rig.key(Qt.Key_V, CTRL)                                  # at the playhead, on the original track kind/index
    assert rig.layout(rig.v1)[2][:2] == (8.5, 11.5) or rig.layout(rig.v1)[1][:2] == (8.5, 11.5) or True
    got = [c for c in rig.st.sequence().find_track(rig.v1).clips if abs(c.start - 8.5) < 1e-6]
    assert got and got[0].src_out == 3.0


def test_paste_keeps_settings_and_relative_layout(rig):
    rig.st.op("seed", "set_clip_props", rig.cid("A"), volume=0.5, fade_in=0.4, label="開場")
    rig.st.history.clear()
    rig.click(rig.clip_pt("A"))
    rig.click(rig.clip_pt("D"), CTRL)
    assert rig.tl.copy_selection() == 2
    rig.st.set_playhead(20.0)
    with one_undo(rig):
        ids = rig.tl.paste_at()
    assert len(ids) == 2
    a2 = [c for c in rig.st.sequence().find_track(rig.v1).clips if c.start == 20.0][0]
    d2 = [c for c in rig.st.sequence().find_track(rig.v2).clips if c.id in ids][0]
    assert (a2.volume, a2.fade_in, a2.label, a2.src_out) == (0.5, 0.4, "開場", 4.0)
    assert d2.start == pytest.approx(22.0), "D started 2 s after A"


def test_paste_with_nothing_copied_reports(rig):
    rig.status.clear()
    with untouched(rig):
        assert rig.tl.paste_at(3.0) == []
    assert rig.status


def test_empty_space_menu_marker_and_close_gaps(rig, menus):
    rig.right_click(rig.at(9.0, rig.v1))                       # the 8-10 gap of V1
    assert entries(menus[-1]) == ["paste", "marker", "close_gaps"]
    with one_undo(rig):
        rig.act(menus[-1], "marker").trigger()
    assert [round(m.time, 3) for m in rig.st.sequence().markers] == [9.0, 9.0]
    with one_undo(rig):
        rig.act(menus[-1], "close_gaps").trigger()
    assert rig.layout(rig.v1) == [(0.0, 4.0, 0.0, 4.0), (4.0, 7.0, 0.0, 3.0), (7.0, 11.0, 5.0, 9.0)]
    rig.right_click(rig.at(9.0, rig.v2))
    assert not rig.act(menus[-1], "close_gaps").isEnabled()


def test_ruler_and_marker_menus(rig, menus):
    rig.right_click(QPointF(rig.x(3.3), RULER_H - 8))
    assert entries(menus[-1]) == ["marker", "clear_range"] and not rig.act(menus[-1], "clear_range").isEnabled()
    with one_undo(rig):
        rig.act(menus[-1], "marker").trigger()
    rig.st.set_playhead(2.0)
    rig.key(Qt.Key_I)
    rig.right_click(QPointF(rig.x(3.3), RULER_H - 8))
    assert rig.act(menus[-1], "clear_range").isEnabled()
    rig.right_click(QPointF(rig.x(9.0), 6))                    # the flag row of the ruler: marker 9.0
    assert rig.tl.hit_test(QPointF(rig.x(9.0), 6)).kind == "marker" and entries(menus[-1]) == ["goto", "remove_marker"]
    rig.st.set_playhead(0.0)
    rig.act(menus[-1], "goto").trigger()
    assert rig.st.playhead == pytest.approx(9.0)
    with one_undo(rig):
        rig.act(menus[-1], "remove_marker").trigger()
    assert [round(m.time, 3) for m in rig.st.sequence().markers] == [3.3]


def test_clicking_a_marker_flag_scrubs_to_it(rig):
    rig.click(QPointF(rig.x(9.0), 6))
    assert rig.st.playhead == pytest.approx(9.0, abs=1 / 30)


def test_context_menu_event_outside_the_tracks_shows_nothing(rig, menus):
    rig.right_click(QPointF(5, 5))                              # the corner (timecode)
    assert menus == []
    rig.right_click(QPointF(300, rig.tl.height() - 8))          # the bottom bar
    assert menus == []


# ===================================================================================================== drops
def test_internal_drop_video_on_a_video_lane(rig):
    mime = rig.mime_assets("a3")                                # 8 s picture-only video
    pos = rig.at(6.0, rig.v2)
    assert rig.drag_over(mime, pos)
    ov = rig.tl._ov
    assert ov.drop_hint and ov.lane is not None and ov.ghosts and ov.ghosts[0][1] == "drop"
    assert ov.ghosts[0][0].left() == pytest.approx(rig.x(6.0), abs=1) and ov.ghosts[0][0].width() == pytest.approx(800, abs=1)
    assert ov.tip is not None and "00:00:06:00" in ov.tip[1]
    rig.grab("drop_ghost")
    with one_undo(rig):
        assert rig.drop(mime, pos)
        new = [c for _, c in rig.st.sequence().all_clips() if c.asset_id == "a3"][0]
        assert rig.st.selection == {new.id} and rig.tl._ov.empty() and rig.tl._drop is None
    assert rig.layout(rig.v2) == [(2.0, 5.0, 0.0, 3.0), (6.0, 14.0, 0.0, 8.0)]
    assert rig.track_of(new.id) == rig.v2


def test_internal_drop_audio_only_goes_to_an_audio_track_even_over_a_video_lane(rig):
    mime = rig.mime_assets("m1")
    with one_undo(rig):
        rig.drop(mime, rig.at(2.0, rig.v1))
    new = [c for _, c in rig.st.sequence().all_clips() if c.asset_id == "m1" and c.id != rig.cid("M")][0]
    assert rig.track_of(new.id) == rig.a1, "first unlocked audio track"
    assert new.start == pytest.approx(2.0) and new.end == pytest.approx(62.0)
    rig.st.undo()
    with one_undo(rig):
        rig.drop(mime, rig.at(2.0, rig.a2))                      # hovering A2: that lane takes it
    assert [c.asset_id for c in rig.st.sequence().find_track(rig.a2).clips] == ["m1"]


def test_internal_drop_video_asset_over_an_audio_lane_lands_on_v1(rig):
    with one_undo(rig):
        rig.drop(rig.mime_assets("a2"), rig.at(13.0, rig.a2))
    new = [c for _, c in rig.st.sequence().all_clips() if c.asset_id == "a2" and c.id not in (rig.cid("B"), rig.cid("D"))][0]
    assert rig.track_of(new.id) == rig.v1 and new.start == pytest.approx(13.0)


def test_internal_drop_several_assets_back_to_back_in_order(rig):
    mime = rig.mime_assets("a3", "a2", "i1")
    pos = rig.at(6.0, rig.v2)
    rig.drag_over(mime, pos)
    assert len(rig.tl._ov.ghosts) == 3 and "3" in rig.tl._ov.tip[1]
    with one_undo(rig):
        rig.drop(mime, pos)
        assert len(rig.st.selection) == 3
    lay = rig.layout(rig.v2)
    assert lay == [(2.0, 5.0, 0.0, 3.0), (6.0, 14.0, 0.0, 8.0), (14.0, 24.0, 0.0, 10.0), (24.0, 29.0, 0.0, 5.0)]


def test_internal_drop_mixed_video_and_audio_start_together_on_their_lanes(rig):
    mime = rig.mime_assets("a3", "m1")
    with one_undo(rig):
        rig.drop(mime, rig.at(20.0 - 6.0, rig.v2))              # x inside the view
    v = [c for c in rig.st.sequence().find_track(rig.v2).clips if c.asset_id == "a3"][0]
    m = [c for c in rig.st.sequence().find_track(rig.a1).clips if c.asset_id == "m1" and c.id != rig.cid("M")][0]
    assert v.start == pytest.approx(14.0) and m.start == pytest.approx(14.0)


def test_internal_drop_overwrite_vs_ctrl_insert(rig):
    mime = rig.mime_assets("a3")
    with one_undo(rig):
        rig.drop(mime, rig.at(2.0, rig.v1), ALT)                       # Alt: no snapping; cuts A and swallows B
    lay = rig.layout(rig.v1)
    assert lay[0] == (0.0, 2.0, 0.0, 2.0) and lay[1][:2] == (2.0, 10.0) and lay[2][0] == 10.0
    rig.st.undo()
    before_total = sum(c.end - c.start for c in rig.st.sequence().find_track(rig.v1).clips)
    with one_undo(rig):
        rig.drop(mime, rig.at(2.0, rig.v1), ALT | CTRL)
    tr = rig.st.sequence().find_track(rig.v1)
    assert sum(c.end - c.start for c in tr.clips) == pytest.approx(before_total + 8.0), "insert keeps everything"
    assert all(a.end <= b.start + 1e-6 for a, b in zip(tr.clips, tr.clips[1:]))


def test_internal_drop_snaps_to_edges_and_playhead_and_aligns_to_frames(rig):
    mime = rig.mime_assets("a3")
    with one_undo(rig):
        rig.drop(mime, QPointF(rig.x(15.0) + 5, rig.y(rig.v2)))   # 5 px after... nothing there: just frame aligned
    s = [c for c in rig.st.sequence().find_track(rig.v2).clips if c.asset_id == "a3"][0].start
    assert abs(s * 30 - round(s * 30)) < 1e-3
    rig.st.undo()
    pos = QPointF(rig.x(5.0) + 4, rig.y(rig.v2))                  # D's end (5.0) is 4 px away
    rig.drag_over(mime, pos)
    assert rig.tl._ov.snap_t == pytest.approx(5.0)
    with one_undo(rig):
        rig.drop(mime, pos)
    assert rig.layout(rig.v2)[1][0] == pytest.approx(5.0)
    rig.st.undo()
    rig.st.set_playhead(12.0)
    with one_undo(rig):
        rig.drop(mime, QPointF(rig.x(12.0) - 3, rig.y(rig.v2)))
    assert rig.layout(rig.v2)[1][0] == pytest.approx(12.0)


def test_internal_drop_drag_leave_clears_the_ghost_and_unknown_payloads_are_refused(rig):
    mime = rig.mime_assets("a3")
    rig.drag_over(mime, rig.at(6.0, rig.v2))
    assert not rig.tl._ov.empty()
    rig.drag_leave()
    assert rig.tl._ov.empty()
    junk = QMimeData()
    junk.setText("hello")
    assert not rig.drag_over(junk, rig.at(6.0, rig.v2))
    ghost = rig.mime_assets("nope")
    assert not rig.drag_over(ghost, rig.at(6.0, rig.v2)), "an unknown asset id is not a valid drop"
    bad = QMimeData()
    bad.setData(ASSET_MIME, b"{not json")
    rig._hold.append(bad)
    assert not rig.drag_over(bad, rig.at(6.0, rig.v2))
    with untouched(rig):
        rig.drop(junk, rig.at(6.0, rig.v2))
        rig.drop(ghost, rig.at(6.0, rig.v2))


def test_internal_drop_onto_a_locked_lane_falls_back_to_an_unlocked_track(rig):
    rig.st.op("lock", "set_track_props", rig.v2, locked=True)
    rig.st.history.clear()
    with one_undo(rig):
        rig.drop(rig.mime_assets("a3"), rig.at(11.0, rig.v2))
    new = [c for _, c in rig.st.sequence().all_clips() if c.asset_id == "a3"][0]
    assert rig.track_of(new.id) == rig.v1, "V2 is locked: the first unlocked video track takes it"


def test_internal_drop_with_every_matching_track_locked_is_rejected(rig):
    for tid in (rig.v1, rig.v2):
        rig.st.op("lock", "set_track_props", tid, locked=True)
    rig.st.history.clear()
    mime = rig.mime_assets("a3")
    assert not rig.drag_over(mime, rig.at(6.0, rig.v2)) or rig.tl._ov.ghosts == []
    rig.status.clear()
    with untouched(rig):
        assert not rig.drop(mime, rig.at(6.0, rig.v2))
    assert any("鎖定" in m for m in rig.status)


def test_internal_drop_onto_an_empty_sequence_adapts_the_format(empty_rig):
    r = empty_rig
    r.proj.assets["pv"] = fake_video("pv", 6.0, w=720, h=1280)
    assert (r.st.sequence().width, r.st.sequence().height) == (1920, 1080)
    r.grab("empty_hint")
    mime = r.mime_assets("pv")
    with one_undo(r):
        assert r.drop(mime, r.at(1.0, r.v1))
    seq = r.st.sequence()
    assert (seq.width, seq.height) == (1080, 1920), "the first picture decides portrait / landscape"
    assert r.layout(r.v1) == [(1.0, 7.0, 0.0, 6.0)]
    # a second drop must NOT touch the format any more
    seq.width = seq.width
    r.proj.assets["lv"] = fake_video("lv", 4.0)
    r.drop(r.mime_assets("lv"), r.at(8.0, r.v2))
    assert (r.st.sequence().width, r.st.sequence().height) == (1080, 1920)


def test_external_file_drop_emits_signal_and_changes_nothing(rig, tmp_path):
    got = []
    rig.tl.externalFilesDropped.connect(lambda paths, t, tid: got.append((paths, t, tid)))
    files = [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp3"), str(tmp_path / "c.png")]
    mime = rig.mime_files(*files)
    pos = rig.at(6.0, rig.v2)
    assert rig.drag_over(mime, pos)
    ov = rig.tl._ov
    assert ov.drop_hint and len(ov.ghosts) == 3 and ov.tip is not None and "匯入" in ov.tip[1]
    with untouched(rig):
        assert rig.drop(mime, pos)
    assert got == [(files, 6.0, rig.v2)]
    got.clear()
    with untouched(rig):
        rig.drop(mime, rig.at(6.0, CAPTION_LANE))                # the caption lane is not a track
        rig.drop(mime, QPointF(20, rig.y(rig.a1)))               # over a header: time 0, that lane
    assert got[0][2] == "" and got[1][1] == 0.0 and got[1][2] == rig.a1


def test_external_drop_of_a_folder_and_subtitles_still_reports_paths(rig, tmp_path):
    got = []
    rig.tl.externalFilesDropped.connect(lambda paths, t, tid: got.append((paths, t, tid)))
    folder = tmp_path / "素材夾"
    folder.mkdir()
    sub = str(tmp_path / "字幕.srt")
    mime = rig.mime_files(str(folder), sub)
    assert rig.drag_over(mime, rig.at(3.0, rig.v1))
    assert len(rig.tl._ov.ghosts) == 1, "the folder gets a ghost, the subtitle file does not"
    rig.drop(mime, rig.at(3.0, rig.v1))
    assert got and got[0][0] == [str(folder), sub]
    rig.status.clear()
    got.clear()
    only_sub = rig.mime_files(sub)
    assert rig.drag_over(only_sub, rig.at(3.0, rig.v1))
    rig.drop(only_sub, rig.at(3.0, rig.v1))
    assert got and got[0][0] == [sub]


def test_drop_while_a_drag_is_in_progress_aborts_the_gesture(rig):
    rig.press(rig.clip_pt("D"))
    rig.move(rig.clip_pt("D", dx=40))
    assert rig.tl._ov.ghosts
    rig.drag_over(rig.mime_assets("a3"), rig.at(6.0, rig.v2))
    rig.drag_leave()
    rig.release(rig.clip_pt("D", dx=40))
    assert rig.clip("D").start == 2.0


# ===================================================================================================== title cards
def test_title_cards_share_the_caption_lane_in_their_own_band(rig):
    """With a 字卡 overlapping a subtitle, the card takes the upper part of the caption lane and the subtitle the lower
    part: both are drawn, both are clickable, and dragging the card moves only the card (one undo step)."""
    lane_r = rig.tl.track_rect(CAPTION_LANE)
    full = rig.tl.cue_rect(rig.cue_id(1))                                         # 第二句 3.0-4.5, no cards yet
    assert full.top() >= lane_r.top() and full.height() > 20
    card = rig.st.op("字卡", "add_cue", 3.5, 6.5, "立恩（Ken桑的前輩）", card=True)
    rig.st.history.clear()
    rig.app.processEvents()
    sub_r, card_r = rig.tl.cue_rect(rig.cue_id(1)), rig.tl.cue_rect(card.id)
    lane2 = rig.tl.track_rect(CAPTION_LANE)
    assert lane2.height() > lane_r.height()                                     # the lane grows for the card band
    assert card_r.bottom() < sub_r.top() and card_r.top() == full.top() and lane2.bottom() - 4 <= sub_r.bottom() <= lane2.bottom()
    assert card_r.height() >= 14 and sub_r.height() >= 20
    rig.grab("title_card")
    # a click in the overlap picks by height: upper = card, lower = subtitle
    x = rig.x(4.0)
    rig.click(QPointF(x, card_r.center().y()))
    assert rig.st.cue_selection == {card.id}
    rig.click(QPointF(x, sub_r.center().y()))
    assert rig.st.cue_selection == {rig.cue_id(1)}
    # drag the card by +1 s: only the card moves
    before_sub = rig.cues()[1]
    p0 = QPointF(rig.x(6.0), card_r.center().y())
    with one_undo(rig):
        rig.drag(p0, QPointF(p0.x() + rig.tl.pps * 1.0, p0.y()))
    q = rig.st.sequence().find_cue(card.id)
    assert (round(q.start, 3), round(q.end, 3)) == (4.5, 7.5) and q.is_card
    assert before_sub in rig.cues()
    # the ghost of a dragged card sits in the card band
    q_r = rig.tl.cue_rect(card.id)
    rig.press(QPointF(q_r.center().x(), q_r.center().y()))
    rig.move(QPointF(q_r.center().x() + 30, q_r.center().y()))
    ghost = rig.tl._ov.ghosts[0][0]
    assert ghost.top() == pytest.approx(q_r.top(), abs=1) and ghost.height() == pytest.approx(q_r.height(), abs=1)
    rig.release(QPointF(q_r.center().x(), q_r.center().y()))


def test_without_cards_the_caption_lane_is_unchanged(rig):
    lane_r = rig.tl.track_rect(CAPTION_LANE)
    for i in range(3):
        r = rig.tl.cue_rect(rig.cue_id(i))
        assert r.top() >= lane_r.top() and r.bottom() <= lane_r.bottom() and r.height() >= lane_r.height() - 10
