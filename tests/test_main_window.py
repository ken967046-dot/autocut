"""MainWindow driven like a user (offscreen): window-wide drag & drop, import suggestions, project files, autosave,
shortcuts, menus / toolbar, sequence tabs, closing.  Dialogs never block: every modal helper is monkeypatched and
recorded; the screenshots go to ``screenshots/main_<name>.png``.
"""
from __future__ import annotations

import faulthandler
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl                    # noqa: E402
from PySide6.QtGui import (QAction, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent,   # noqa: E402
                           QShortcut)
from PySide6.QtTest import QTest                                                          # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,   # noqa: E402
                               QInputDialog, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QSlider, QTableWidget, QToolButton, QWidget)

from autocut.core import ops                                                              # noqa: E402
from autocut.core.project import Project                                                  # noqa: E402
from autocut.gui import app as app_mod                                                    # noqa: E402
from autocut.gui import main_window as mw                                                 # noqa: E402
from autocut.gui.state import AppState                                                    # noqa: E402
from autocut.gui.widgets import keeps_its_keys                                            # noqa: E402
from autocut.gui.workers import TaskRunner, wait_for_idle                                # noqa: E402
from autocut.media.probe import probe_asset                                               # noqa: E402
from autocut.util import PROJECT_EXT                                                      # noqa: E402

pytestmark = [pytest.mark.gui, pytest.mark.ffmpeg]

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "screenshots"


# ============================================================================ helpers
def shot(widget: QWidget, name: str) -> Path:
    SHOTS.mkdir(exist_ok=True)
    path = SHOTS / f"main_{name}.png"
    assert widget.grab().save(str(path))
    return path


def settle(qapp, ms: int = 20) -> None:
    QTest.qWait(ms)
    qapp.processEvents()


def make_media(path: Path, spec: list[str]) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *spec, str(path)], check=True)
    return path


def clips_of(state: AppState) -> list:
    """All clips of the active sequence ordered by track then start."""
    seq = state.sequence()
    return [c for t in seq.tracks() for c in sorted(t.clips, key=lambda c: c.start)]


def v1_clips(state: AppState) -> list:
    return sorted(state.sequence().video_tracks[0].clips, key=lambda c: c.start)


class Stubs:
    """Everything modal, recorded.  ``ask`` answers come from ``ask_answers`` (a queue) then ``ask_default``."""

    def __init__(self) -> None:
        self.asks: list[tuple[str, str, str, str]] = []
        self.informs: list[tuple[str, str, bool]] = []
        self.ask_answers: list[bool] = []
        self.ask_default = True
        self.save_prompts: list[str] = []
        self.save_choices: list[str] = []
        self.save_default = "cancel"
        self.open_file = ""
        self.open_files: list[str] = []
        self.save_file = ""
        self.folder = ""
        self.text: str | None = None            # QInputDialog.getText result (None = cancelled)
        self.item: str | None = None
        self.urls: list[QUrl] = []
        self.file_dialogs: list[str] = []

    def last_inform(self) -> str:
        return self.informs[-1][1] if self.informs else ""

    def last_ask(self) -> str:
        return self.asks[-1][1] if self.asks else ""


@pytest.fixture(autouse=True)
def _watchdog():
    faulthandler.dump_traceback_later(120, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture
def stubs(qapp, tmp_path, monkeypatch) -> Stubs:
    """Isolated app data + settings, and no modal dialog can open."""
    home = tmp_path / "apphome"
    home.mkdir()
    monkeypatch.setenv("AUTOCUT_HOME", str(home))
    s = mw.app_settings()
    s.clear()
    s.sync()
    st = Stubs()

    def ask(_parent, title, text, yes="確定", no="取消"):
        st.asks.append((title, text, yes, no))
        return st.ask_answers.pop(0) if st.ask_answers else st.ask_default

    def inform(_parent, title, text, warning=False):
        st.informs.append((title, text, warning))

    def ask_save(_parent, name):
        st.save_prompts.append(name)
        return st.save_choices.pop(0) if st.save_choices else st.save_default

    monkeypatch.setattr(mw, "ask", ask)
    monkeypatch.setattr(mw, "inform", inform)
    monkeypatch.setattr(mw, "ask_save_changes", ask_save)

    def get_open_file(*_a, **_k):
        st.file_dialogs.append("open")
        return st.open_file, ""

    def get_open_files(*_a, **_k):
        st.file_dialogs.append("open_many")
        return list(st.open_files), ""

    def get_save_file(*_a, **_k):
        st.file_dialogs.append("save")
        return st.save_file, ""

    def get_dir(*_a, **_k):
        st.file_dialogs.append("dir")
        return st.folder

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(get_open_file))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(get_open_files))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(get_save_file))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(get_dir))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(
        lambda *_a, **_k: (st.text or "", st.text is not None)))
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(
        lambda *_a, **_k: (st.item or "", st.item is not None)))
    monkeypatch.setattr(mw.QDesktopServices, "openUrl", staticmethod(lambda url: st.urls.append(url) or True))

    def boom(self, *_a, **_k):
        raise AssertionError(f"unexpected modal dialog: {type(self).__name__} {self.windowTitle()!r}")

    monkeypatch.setattr(QDialog, "exec", boom)
    monkeypatch.setattr(QMessageBox, "exec", boom)
    return st


@pytest.fixture
def make_window(qapp, stubs, monkeypatch):
    """Factory ``make(project=None, paths=None)`` -> shown MainWindow whose startup already ran."""
    made: list[mw.MainWindow] = []

    def make(project: Project | None = None, paths: list[str] | None = None, show: bool = True,
             size: tuple[int, int] | None = None) -> mw.MainWindow:
        state = AppState(project) if project is not None else None
        win = mw.MainWindow(state, paths)
        made.append(win)
        if size:
            win.resize(*size)
        if show:
            win.show()
            QTest.qWaitForWindowActive(win, 2000)
        settle(qapp)
        return win

    yield make
    for win in made:
        win._started = True
        win.state.editor.history.mark_saved()
        QMainWindow.close(win)
        win.deleteLater()
    qapp.processEvents()


@pytest.fixture(scope="module")
def extra(media_dir, tmp_path_factory) -> dict[str, Path]:
    """46 s landscape clip, odd frame rates, a picture and the three subtitle formats."""
    d = tmp_path_factory.mktemp("window_extra")
    fast = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    paths = {
        "long": make_media(d / "long_land.mp4", ["-f", "lavfi", "-i", "color=c=gray:size=192x108:rate=10:duration=46",
                                                 *fast]),
        "long2": make_media(d / "long_two.mp4", ["-f", "lavfi", "-i", "color=c=teal:size=192x108:rate=10:duration=47",
                                                 *fast]),
        "fps15": make_media(d / "fps15.mp4", ["-f", "lavfi", "-i", "color=c=red:size=128x72:rate=15:duration=2",
                                              *fast]),
        "fps120": make_media(d / "fps120.mp4", ["-f", "lavfi", "-i", "color=c=red:size=128x72:rate=120:duration=1",
                                                *fast]),
        "ntsc": make_media(d / "ntsc.mp4", ["-f", "lavfi", "-i", "color=c=red:size=128x72:rate=24000/1001:duration=2",
                                            *fast]),
        "short_land": make_media(d / "short_land.mp4", ["-f", "lavfi", "-i",
                                                        "color=c=blue:size=128x72:rate=10:duration=3", *fast]),
        "short_land2": make_media(d / "short_land2.mp4", ["-f", "lavfi", "-i",
                                                          "color=c=green:size=128x72:rate=10:duration=2", *fast]),
        "image": make_media(d / "photo.png", ["-f", "lavfi", "-i", "color=c=orange:size=160x90", "-frames:v", "1"]),
    }
    srt = d / "字幕.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\n你好\n\n2\n00:00:02,500 --> 00:00:04,000\n世界\n\n"
                   "3\n00:00:04,500 --> 00:00:06,000\n再見\n", encoding="utf-8")
    vtt = d / "words.vtt"
    vtt.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nhello vtt\n\n00:00:03.000 --> 00:00:04.000\nsecond\n",
                   encoding="utf-8")
    ass = d / "sub.ass"
    ass.write_text("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour\n"
                   "Style: Default,Arial,20,&H00FFFFFF\n\n[Events]\n"
                   "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                   "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,ass line one\n"
                   "Dialogue: 0,0:00:03.50,0:00:05.00,Default,,0,0,0,,ass line two\n", encoding="utf-8")
    empty_srt = d / "empty.srt"
    empty_srt.write_text("", encoding="utf-8")
    paths.update({"srt": srt, "vtt": vtt, "ass": ass, "empty_srt": empty_srt})
    return paths


def mime_for(paths) -> QMimeData:
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


class Dnd:
    """Synthetic OS drag of files over a window: enter / move / leave / drop events sent straight to the window."""

    def __init__(self, win: QWidget, paths) -> None:
        self.win = win
        self.mime = mime_for(paths)            # must outlive the events
        self.pos = QPoint(300, 300)

    def enter(self) -> QDragEnterEvent:
        ev = QDragEnterEvent(self.pos, Qt.CopyAction, self.mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(self.win, ev)
        return ev

    def move(self) -> QDragMoveEvent:
        ev = QDragMoveEvent(self.pos, Qt.CopyAction, self.mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(self.win, ev)
        return ev

    def leave(self) -> None:
        QApplication.sendEvent(self.win, QDragLeaveEvent())

    def drop(self) -> QDropEvent:
        ev = QDropEvent(QPointF(self.pos), Qt.CopyAction, self.mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(self.win, ev)
        return ev


def finish_imports(qapp, win: mw.MainWindow, timeout: float = 60.0) -> None:
    """Wait until every import runner of the window has finished (and its signals were delivered)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if not win.importer.busy and not win.importer.runners:
            qapp.processEvents()
            return
        time.sleep(0.01)
    raise TimeoutError("匯入逾時")


def drop_and_import(qapp, win: mw.MainWindow, paths) -> Dnd:
    dnd = Dnd(win, paths)
    dnd.enter()
    dnd.drop()
    finish_imports(qapp, win)
    return dnd


class Recorder:
    """Records the ``edited(do|undo|redo, label)`` signals of a state."""

    def __init__(self, state: AppState) -> None:
        self.events: list[tuple[str, str]] = []
        state.edited.connect(lambda kind, label: self.events.append((kind, label)))

    @property
    def labels(self) -> list[str]:
        return [label for kind, label in self.events if kind == "do"]


# ============================================================================ construction / layout
def test_window_builds_with_all_parts_and_default_title(qapp, make_window):
    win = make_window()
    assert win.windowTitle() == f"未命名專案 — AutoCut Studio"
    assert win.centralWidget() is win.splitter
    assert [win.inspector.tabText(i) for i in range(win.inspector.count())] == ["片段", "字幕", "序列", "檢查"]
    assert win.dock_bin.widget() is win.media_bin and win.dock_inspector.widget() is win.inspector
    assert win.dock_bin.windowTitle() == "素材庫" and win.dock_inspector.windowTitle() == "檢查器"
    assert not win.banner.isVisible() and not win.overlay.isVisible()
    assert win.acceptDrops()
    assert win.autosave_timer.isActive() and win.autosave_timer.interval() == 60_000
    assert not win.windowIcon().isNull()
    assert not win.act_undo.isEnabled() and not win.act_redo.isEnabled()
    assert "沒有可復原" in win.act_undo.toolTip()


def test_title_follows_dirty_and_file_name(qapp, make_window, real_proj, stubs, tmp_path):
    win = make_window(real_proj)
    assert win.windowTitle() == "real — AutoCut Studio"
    aid = next(iter(real_proj.assets))
    win.state.op("放", "place_clip", aid, win.state.sequence().video_tracks[0].id, 0.0)
    assert win.windowTitle() == "real * — AutoCut Studio"
    assert win.act_undo.isEnabled() and "放" in win.act_undo.toolTip()
    stubs.save_file = str(tmp_path / "我的專案")
    assert win.save_project()
    assert win.windowTitle() == f"我的專案{PROJECT_EXT} — AutoCut Studio"


def test_undo_redo_actions_and_status_line(qapp, make_window, real_proj):
    win = make_window(real_proj)
    rec = Recorder(win.state)
    aid = next(iter(real_proj.assets))
    win.state.op("放進去", "place_clip", aid, win.state.sequence().video_tracks[0].id, 0.0)
    assert win.act_undo.isEnabled() and not win.act_redo.isEnabled()
    win.act_undo.trigger()
    assert not clips_of(win.state)
    assert win.act_redo.isEnabled() and "放進去" in win.act_redo.toolTip()
    assert win.statusBar().currentMessage() == "已復原：放進去"
    win.act_redo.trigger()
    assert len(clips_of(win.state)) == 1
    assert win.statusBar().currentMessage() == "已重做：放進去"
    assert [k for k, _ in rec.events] == ["do", "undo", "redo"]


def test_window_fits_common_laptop_screens_without_clipping(qapp, make_window, real_proj):
    """1280x720 and 1366x768: every toolbar button and every control inside the window is fully visible and no text
    is squeezed below the size its widget asks for."""
    win = make_window(real_proj, size=(1280, 720))
    assert win.minimumSizeHint().width() <= 1280 and win.minimumSizeHint().height() <= 720
    for size in ((1280, 720), (1366, 768)):
        win.resize(*size)
        settle(qapp, 60)
        assert win.size().width() == size[0] and win.size().height() <= size[1]
        ext = win.toolbar.findChild(QToolButton, "qt_toolbar_ext_button")
        assert ext is None or not ext.isVisible(), "toolbar buttons are hidden behind the >> overflow button"
        for act in win.toolbar.actions():
            if act.isSeparator() or not act.isVisible():
                continue
            w = win.toolbar.widgetForAction(act)
            assert w is not None and w.isVisible()
            assert win.toolbar.rect().contains(w.geometry()), f"toolbar button {act.text()!r} is clipped at {size}"
        problems = squeezed_widgets(win)
        assert not problems, f"{size}: " + "; ".join(problems)
        bar = win.statusBar()
        assert bar.isVisible() and win.seq_info.isVisible()
        assert win.seq_info.width() >= win.seq_info.sizeHint().width() - 1


def squeezed_widgets(root: QWidget) -> list[str]:
    """Visible buttons / labels / combos with text that are laid out narrower than their minimum hint (text cut off)."""
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel
    bad = []
    for w in root.findChildren(QWidget):
        if not w.isVisible() or not isinstance(w, (QAbstractButton, QLabel, QComboBox)):
            continue
        text = w.currentText() if isinstance(w, QComboBox) else w.text()
        if not text or (isinstance(w, QLabel) and (w.wordWrap() or w.pixmap() is not None)):
            continue
        need = w.minimumSizeHint().width()
        if need > 0 and w.width() + 1 < need:
            bad.append(f"{type(w).__name__}({text!r}) {w.width()}<{need}")
    return bad


# ============================================================================ menus / toolbar / shortcuts
def walk_menu(menu, prefix: str = ""):
    for act in menu.actions():
        if act.isSeparator():
            continue
        text = act.text().replace("&", "")
        if act.menu() is not None:
            yield from walk_menu(act.menu(), f"{prefix}{text}/")
        else:
            yield f"{prefix}{text}", act


def menu_actions(win: mw.MainWindow) -> dict[str, QAction]:
    out: dict[str, QAction] = {}
    for top in win.menuBar().actions():
        label = top.text().replace("&", "")
        label = label.split("(")[0]
        for path, act in walk_menu(top.menu(), f"{label}/"):
            out[path] = act
    return out


SLOTS = ("new_project", "open_project_dialog", "save_project", "save_project_as", "import_dialog",
         "import_folder_dialog", "show_export_dialog", "close", "undo", "redo", "toggle_playback", "render_preview",
         "toggle_fullscreen", "reset_layout", "open_shorts_dialog", "open_assemble_dialog", "show_about",
         "show_environment")


def test_every_menu_and_toolbar_action_triggers_the_right_method(qapp, make_window, monkeypatch, stubs, real_proj):
    calls: list[tuple] = []
    for name in SLOTS:
        monkeypatch.setattr(mw.MainWindow, name, lambda self, *a, _n=name: calls.append((_n,)))
    win = make_window(real_proj)
    tl_calls: list[tuple] = []
    for name in ("split_at_playhead", "delete_selection", "add_marker_at_playhead", "set_in_at_playhead",
                 "set_out_at_playhead", "clear_range", "goto_prev_cut", "goto_next_cut", "step_frames", "zoom_in",
                 "zoom_out", "zoom_fit", "select_all", "set_snapping"):
        monkeypatch.setattr(win.timeline, name, lambda *a, _n=name: tl_calls.append((_n, *a)))

    expected_methods = {
        "檔案/新增專案": ("new_project",), "檔案/開啟專案…": ("open_project_dialog",),
        "檔案/匯入素材": ("import_dialog",), "檔案/匯入資料夾…": ("import_folder_dialog",),
        "檔案/儲存專案": ("save_project",), "檔案/另存新檔…": ("save_project_as",),
        "檔案/輸出影片": ("show_export_dialog",), "檔案/結束": ("close",),
        "編輯/復原": ("undo",), "編輯/重做": ("redo",),
        "時間軸/播放／暫停": ("toggle_playback",),
        "自動剪輯/自動剪成短影音…": ("open_shorts_dialog",), "自動剪輯/自動合併成長片…": ("open_assemble_dialog",),
        "檢視/渲染預覽（含聲音）": ("render_preview",), "檢視/全螢幕": ("toggle_fullscreen",),
        "檢視/重設版面": ("reset_layout",),
        "說明/檢查環境…": ("show_environment",), "說明/關於 AutoCut Studio": ("show_about",),
    }
    expected_timeline = {
        "編輯/全選\tCtrl+A": [("select_all",)], "時間軸/在播放頭分割\tS": [("split_at_playhead",)],
        "時間軸/刪除\tDelete": [("delete_selection", False)], "時間軸/波紋刪除\tShift+Delete": [("delete_selection", True)],
        "時間軸/加入標記\tM": [("add_marker_at_playhead",)], "時間軸/設定入點\tI": [("set_in_at_playhead",)],
        "時間軸/設定出點\tO": [("set_out_at_playhead",)], "時間軸/清除入出點": [("clear_range",)],
        "時間軸/上一個剪輯點\t↑": [("goto_prev_cut",)], "時間軸/下一個剪輯點\t↓": [("goto_next_cut",)],
        "時間軸/後退一格\t←": [("step_frames", -1)], "時間軸/前進一格\t→": [("step_frames", 1)],
        "時間軸/放大\t+": [("zoom_in",)], "時間軸/縮小\t-": [("zoom_out",)], "時間軸/符合視窗\t\\": [("zoom_fit",)],
    }
    others = {"時間軸/吸附", "時間軸/新增視訊軌", "時間軸/新增音訊軌", "檢視/素材庫", "檢視/檢查器",
              "說明/開啟資料與記錄資料夾"}
    win.state.op("標記一", "add_marker", 1.0)           # undo / redo are only enabled when there is something to do
    win.state.op("標記二", "add_marker", 2.0)
    win.state.undo()
    acts = menu_actions(win)
    assert set(acts) == set(expected_methods) | set(expected_timeline) | others | {
        p for p in acts if p.startswith("檔案/開啟最近的專案/")}, sorted(set(acts) ^ (
            set(expected_methods) | set(expected_timeline) | others))
    for path, method in expected_methods.items():
        calls.clear()
        acts[path].trigger()
        assert calls == [method], path
    for path, expect in expected_timeline.items():
        tl_calls.clear()
        acts[path].trigger()
        assert tl_calls == expect, path
    # snapping is a toggle that reaches the timeline
    tl_calls.clear()
    acts["時間軸/吸附"].toggle()
    assert tl_calls == [("set_snapping", False)]
    acts["時間軸/吸附"].toggle()
    assert tl_calls[-1] == ("set_snapping", True)
    # tracks: real ops, one undo step each
    rec = Recorder(win.state)
    nv, na = len(win.state.sequence().video_tracks), len(win.state.sequence().audio_tracks)
    acts["時間軸/新增視訊軌"].trigger()
    acts["時間軸/新增音訊軌"].trigger()
    assert (len(win.state.sequence().video_tracks), len(win.state.sequence().audio_tracks)) == (nv + 1, na + 1)
    assert rec.labels == ["新增視訊軌", "新增音訊軌"]
    # data folder
    acts["說明/開啟資料與記錄資料夾"].trigger()
    assert stubs.urls and stubs.urls[-1].toLocalFile() == str(mw.app_data_dir())
    # dock toggles hide / show the docks
    acts["檢視/素材庫"].trigger()
    assert win.dock_bin.isHidden()
    acts["檢視/素材庫"].trigger()
    assert not win.dock_bin.isHidden()
    acts["檢視/檢查器"].trigger()
    assert win.dock_inspector.isHidden()
    acts["檢視/檢查器"].trigger()
    assert not win.dock_inspector.isHidden()


def test_toolbar_buttons_edit_the_project(qapp, make_window, real_proj):
    """The real toolbar buttons: split at the playhead, delete the selection, undo, redo (one undo step each)."""
    win = make_window(real_proj)
    st = win.state
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    st.op("放", "place_clip", land.id, st.sequence().video_tracks[0].id, 0.0)
    rec = Recorder(st)
    buttons = {a.text(): win.toolbar.widgetForAction(a) for a in win.toolbar.actions() if not a.isSeparator()}
    assert list(buttons) == ["匯入素材", "儲存專案", "復原", "重做", "分割", "刪除", "吸附", "自動剪成短影音…",
                             "自動合併成長片…", "", "輸出影片"]          # "" is the stretch spacer
    assert buttons["復原"].isEnabled() and not buttons["重做"].isEnabled()
    st.set_playhead(3.0)
    buttons["分割"].click()
    assert [round(c.start, 3) for c in v1_clips(st)] == [0.0, 3.0]
    assert len(rec.labels) == 1
    buttons["復原"].click()
    assert len(v1_clips(st)) == 1 and buttons["重做"].isEnabled()
    buttons["重做"].click()
    assert len(v1_clips(st)) == 2 and not buttons["重做"].isEnabled()
    st.select([v1_clips(st)[1].id])
    buttons["刪除"].click()
    assert len(v1_clips(st)) == 1 and v1_clips(st)[0].end == pytest.approx(3.0)
    assert rec.labels == ["分割", "刪除"]           # undo / redo are not new steps
    buttons["復原"].click()
    assert len(v1_clips(st)) == 2 and buttons["重做"].isEnabled()


def test_toolbar_import_and_save_buttons_open_the_dialogs(qapp, make_window, stubs, real_proj, tmp_path):
    win = make_window(real_proj)
    buttons = {a.text(): win.toolbar.widgetForAction(a) for a in win.toolbar.actions() if not a.isSeparator()}
    buttons["匯入素材"].click()                  # no file chosen -> nothing imported
    assert stubs.file_dialogs == ["open_many"]
    stubs.save_file = str(tmp_path / "tb")
    buttons["儲存專案"].click()
    assert stubs.file_dialogs[-1] == "save" and (tmp_path / f"tb{PROJECT_EXT}").is_file()


def global_shortcuts(win: QWidget) -> dict[str, list[str]]:
    """key sequence -> owners, for every window / application-wide shortcut of the window tree."""
    found: dict[str, list[str]] = {}
    for act in win.findChildren(QAction):
        if act.shortcutContext() in (Qt.WidgetShortcut, Qt.WidgetWithChildrenShortcut):
            continue
        for ks in act.shortcuts():
            found.setdefault(ks.toString(), []).append(f"QAction:{act.text()}")
    for sc in win.findChildren(QShortcut):
        if sc.context() in (Qt.WidgetShortcut, Qt.WidgetWithChildrenShortcut):
            continue
        found.setdefault(sc.key().toString(), []).append("QShortcut")
    return found


def test_global_shortcuts_match_spec_and_never_collide_with_timeline_keys(qapp, make_window):
    """GUI_SPEC 11.1: the window owns only the global keys; the timeline keeps its own (no duplicate bindings)."""
    win = make_window()
    found = global_shortcuts(win)
    spec = {"Ctrl+N", "Ctrl+O", "Ctrl+S", "Ctrl+Shift+S", "Ctrl+I", "Ctrl+E", "Ctrl+Return", "Ctrl+Enter", "Ctrl+Z",
            "Ctrl+Y", "Ctrl+Shift+Z", "Space", "Ctrl+Q", "F11"}
    assert set(found) == spec, sorted(set(found) ^ spec)
    dupes = {k: v for k, v in found.items() if len(v) > 1}
    assert not dupes, dupes
    timeline_keys = {"Left", "Right", "Shift+Left", "Shift+Right", "Home", "End", "Up", "Down", "S", "Ctrl+K",
                     "Delete", "Shift+Delete", "M", "I", "O", "Ctrl+A", "+", "-", "\\", "Esc", "Escape", "=", "Backspace"}
    assert not (set(found) & timeline_keys)
    # the menu entries that belong to the timeline carry their key as text only
    for path, act in menu_actions(win).items():
        if "\t" in path:
            assert not act.shortcuts(), path
    # the actions bind exactly the documented keys
    assert [s.toString() for s in win.act_undo.shortcuts()] == ["Ctrl+Z"]
    assert {s.toString() for s in win.act_redo.shortcuts()} == {"Ctrl+Y", "Ctrl+Shift+Z"}
    assert [s.toString() for s in win.act_play.shortcuts()] == ["Space"]
    assert {s.toString() for s in win.act_preview.shortcuts()} == {"Ctrl+Return", "Ctrl+Enter"}
    assert [s.toString() for s in win.act_fullscreen.shortcuts()] == ["F11"]


def test_global_keys_work_and_timeline_keys_stay_with_the_timeline(qapp, make_window, real_proj, stubs, tmp_path,
                                                                   monkeypatch):
    win = make_window(real_proj)
    st = win.state
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    st.op("放", "place_clip", land.id, st.sequence().video_tracks[0].id, 0.0)
    assert win.isActiveWindow()
    # Space plays / pauses (global QAction), also when the key goes to the timeline
    QTest.keyClick(win.timeline, Qt.Key_Space)
    assert win.playback.is_playing
    QTest.keyClick(win.timeline, Qt.Key_Space)
    assert not win.playback.is_playing
    # S / Delete are the timeline's own keys
    st.set_playhead(3.0)
    QTest.keyClick(win.timeline, Qt.Key_S)
    assert len(v1_clips(st)) == 2
    QTest.keyClick(win.timeline, Qt.Key_Z, Qt.ControlModifier)
    assert len(v1_clips(st)) == 1
    QTest.keyClick(win.timeline, Qt.Key_Y, Qt.ControlModifier)
    assert len(v1_clips(st)) == 2
    QTest.keyClick(win.timeline, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)    # Ctrl+Shift+Z is redo too
    assert len(v1_clips(st)) == 2
    QTest.keyClick(win.timeline, Qt.Key_Z, Qt.ControlModifier)
    QTest.keyClick(win.timeline, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
    assert len(v1_clips(st)) == 2
    # Ctrl+S with no file name asks where to save
    stubs.save_file = str(tmp_path / "ks")
    QTest.keyClick(win.timeline, Qt.Key_S, Qt.ControlModifier)
    assert (tmp_path / f"ks{PROJECT_EXT}").is_file() and len(v1_clips(st)) == 2   # it did not split again
    # Ctrl+I imports, Ctrl+N asks about nothing when saved, F11 toggles full screen
    QTest.keyClick(win.timeline, Qt.Key_I, Qt.ControlModifier)
    assert stubs.file_dialogs[-1] == "open_many"
    full = []
    monkeypatch.setattr(win, "showFullScreen", lambda: full.append("full"))
    QTest.keyClick(win.timeline, Qt.Key_F11)
    assert full == ["full"]


def test_typing_in_text_fields_is_never_stolen_by_global_shortcuts(qapp, make_window, real_proj):
    """Space typed into a line edit stays text (it must not start playback); letters never trigger timeline edits."""
    win = make_window(real_proj)
    st = win.state
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    st.op("放", "place_clip", land.id, st.sequence().video_tracks[0].id, 0.0)
    st.set_playhead(2.0)
    field = win.inspector.sequence_tab.f_name
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "a b s m i o")
    assert field.text() == "a b s m i o"
    assert not win.playback.is_playing
    assert len(v1_clips(st)) == 1 and not st.get_range()
    search = win.media_bin.findChild(QLineEdit)
    if search is not None:
        search.setFocus()
        QTest.keyClick(search, Qt.Key_Space)
        assert not win.playback.is_playing and search.text() == " "


# ============================================================================ keyboard focus / undo
# The bugs these tests pin down (found by a fresh-context GUI review): Ctrl+Z did nothing while a number or time box had the
# keyboard (a line edit claims Undo for its own text history, empty or not), and after a click on a slider / button / check
# box the keys of the timeline (arrows, S, Delete) went to that control and edited it again.
def key(k, mods=Qt.NoModifier) -> QWidget:
    """A key press as the keyboard delivers it: to the widget that has the focus (and up its parents).  Returns that widget."""
    w = QApplication.focusWidget()
    assert w is not None, "no widget has the keyboard"
    QTest.keyClick(w, k, mods)
    QApplication.processEvents()
    return w


def typing_part(box) -> QLineEdit:
    """The line edit that really takes the keys of a number box (a spin box hands its focus to it)."""
    return box.lineEdit() if hasattr(box, "lineEdit") else box


def click_into(qapp, box) -> QLineEdit:
    """What a click into a number / time box does: the box gets the keyboard (a number box is the focus widget itself, its
    line edit only takes the keys over from it).  Returns the line edit, whose own text history can be asked."""
    box.setFocus(Qt.MouseFocusReason)
    settle(qapp, 5)
    assert QApplication.focusWidget() is box, "the box did not get the keyboard"
    return typing_part(box)


def type_and_enter(qapp, box, text: str) -> None:
    click_into(qapp, box)
    box.selectAll()
    QTest.keyClicks(QApplication.focusWidget(), text)
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_Return)
    settle(qapp, 30)                                         # the hand-back to the timeline is deferred one event-loop turn


@pytest.fixture
def clip_window(qapp, make_window, real_proj):
    """A window with one selected 8 s clip on V1 (the clip tab of the inspector shows it)."""
    win = make_window(real_proj)
    st = win.state
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    clip = st.op("放", "place_clip", land.id, st.sequence().video_tracks[0].id, 0.0)
    st.select([clip.id])
    win.timeline.setFocus()
    settle(qapp)
    assert QApplication.focusWidget() is win.timeline and win.isActiveWindow()
    return win


def test_ctrl_z_and_y_reach_the_project_from_every_number_and_time_box(qapp, clip_window):
    """Ctrl+Z / Ctrl+Y while a number or time box has the keyboard (nothing typed in it) undo / redo the PROJECT."""
    win, st = clip_window, clip_window.state
    tab = win.inspector.clip_tab
    for name, box, text in (("速度", tab.f_speed, "1.5"), ("淡入", tab.f_fade_in, "0.5"), ("音量", tab.f_vol, "-6"),
                            ("開始", tab.f_start, "1")):
        before = st.project.content_json()
        type_and_enter(qapp, box, text)
        after = st.project.content_json()
        assert after != before, name
        assert win.timeline.hasFocus(), f"{name}: Enter must give the keyboard back to the timeline"
        edit = click_into(qapp, box)                       # the person clicks into the box again: no text of its own to undo
        assert not edit.isUndoAvailable() and not edit.isRedoAvailable(), name
        key(Qt.Key_Z, Qt.ControlModifier)
        assert st.project.content_json() == before, f"{name}: Ctrl+Z must undo the edit"
        key(Qt.Key_Y, Qt.ControlModifier)
        assert st.project.content_json() == after, f"{name}: Ctrl+Y must redo it"
        key(Qt.Key_Z, Qt.ControlModifier)                  # and leave the project as it was for the next box
        assert st.project.content_json() == before, name
        win.timeline.setFocus()
        settle(qapp)


def test_a_box_with_typing_in_it_keeps_its_own_undo(qapp, clip_window):
    """Text typed into a box but not committed is undone by Ctrl+Z inside the box - the project is not touched."""
    win, st = clip_window, clip_window.state
    for box in (win.inspector.clip_tab.f_speed, win.inspector.clip_tab.f_start):
        edit = click_into(qapp, box)
        shown = edit.text()
        box.selectAll()
        QTest.keyClicks(box, "2")
        assert edit.text() != shown and edit.isUndoAvailable()
        before, undos = st.project.content_json(), st.history.can_undo
        key(Qt.Key_Z, Qt.ControlModifier)
        assert edit.text() == shown, "the typing is what Ctrl+Z undoes while there is typing to undo"
        assert st.project.content_json() == before and st.history.can_undo == undos
        box.clearFocus()
        settle(qapp)


def test_enter_in_a_box_gives_the_keyboard_back_to_the_timeline(qapp, clip_window):
    win, st = clip_window, clip_window.state
    tab = win.inspector.clip_tab
    type_and_enter(qapp, tab.f_speed, "2")
    assert win.timeline.hasFocus() and abs(v1_clips(st)[0].speed - 2.0) < 1e-9
    st.set_playhead(1.0)
    before = st.playhead
    key(Qt.Key_Right)                                     # the timeline's key: one frame on, the box is not touched
    assert st.playhead > before and abs(v1_clips(st)[0].speed - 2.0) < 1e-9 and abs(tab.f_speed.value() - 2.0) < 1e-9
    key(Qt.Key_S)                                         # ... and S splits instead of being typed into the box
    assert len(v1_clips(st)) == 2


def test_a_click_on_a_slider_button_check_box_or_combo_leaves_the_keyboard_with_the_timeline(qapp, clip_window):
    """The reviewer's case: click the zoom slider (or the mirror check box ...), then press an arrow key / S: the playhead
    steps / the clip splits - the control that was clicked is not edited again."""
    win, st = clip_window, clip_window.state
    tab = win.inspector.clip_tab
    slider = tab.f_zoom.findChild(QSlider)
    assert slider is not None
    for name, w in (("slider", slider), ("check box", tab.f_flip), ("combo", tab.f_fit),
                    ("tool button", win.toolbar.widgetForAction(win.act_undo))):
        st.set_playhead(2.0)
        win.timeline.setFocus()
        settle(qapp)
        QTest.mouseClick(w, Qt.LeftButton, Qt.NoModifier, w.rect().center() if name != "slider" else QPoint(3, 3))
        settle(qapp, 30)
        if name == "combo":
            tab.f_fit.hidePopup()                              # the popup took the keyboard while it was open; closing gives it back
            settle(qapp, 30)
        assert QApplication.focusWidget() is win.timeline, f"{name}: a click must not take the keyboard from the timeline"
    value, flip, n = slider.value(), tab.f_flip.isChecked(), len(v1_clips(st))
    key(Qt.Key_Right)
    assert st.playhead > 2.0 and slider.value() == value and tab.f_flip.isChecked() == flip
    key(Qt.Key_Space)                                     # Space is play / pause, not 'press the last button again'
    assert win.playback.is_playing
    key(Qt.Key_Space)
    st.set_playhead(3.0)
    key(Qt.Key_S)
    assert len(v1_clips(st)) == n + 1


def controls_that_took_clicks(root: QWidget) -> list[str]:
    """Buttons, check boxes, sliders and (non-editable) combo boxes that a mouse click gives the keyboard to."""
    from PySide6.QtWidgets import QAbstractButton, QAbstractSlider
    bad = []
    for w in root.findChildren(QWidget):
        if isinstance(w, (QAbstractButton, QAbstractSlider, QComboBox)) and not (isinstance(w, QComboBox) and w.isEditable()):
            if w.focusPolicy() & Qt.ClickFocus:
                bad.append(f"{type(w).__name__} {w.objectName() or w.text() if hasattr(w, 'text') else ''}".strip())
    return bad


def test_controls_take_the_keyboard_by_tab_only(qapp, clip_window):
    """Every tab of the inspector and the rest of the window: no button / slider / check box / combo gives itself the
    keyboard on a click (Tab still reaches them)."""
    win = clip_window
    assert controls_that_took_clicks(win) == []
    panel = win.inspector
    panel.setCurrentWidget(panel.clip_tab)
    settle(qapp)
    tab_stops = [w for w in panel.clip_tab.findChildren(QWidget) if w.focusPolicy() & Qt.TabFocus and w.isVisible()]
    assert any(isinstance(w, QSlider) for w in tab_stops) and any(isinstance(w, QCheckBox) for w in tab_stops), \
        "Tab must still reach the sliders and check boxes"
    for i in range(panel.count()):
        panel.setCurrentIndex(i)
        settle(qapp)
        assert controls_that_took_clicks(panel) == [], panel.tabText(i)


def test_keys_nobody_wanted_drive_the_timeline_but_typing_boxes_and_lists_keep_theirs(qapp, clip_window):
    win, st = clip_window, clip_window.state
    tab = win.inspector.clip_tab
    # a check box reached with Tab does not use S / Delete / Right: they are the timeline's
    tab.f_flip.setFocus(Qt.TabFocusReason)
    settle(qapp)
    assert QApplication.focusWidget() is tab.f_flip
    st.set_playhead(3.0)
    key(Qt.Key_S)
    assert len(v1_clips(st)) == 2
    st.select([v1_clips(st)[1].id])
    tab.f_flip.setFocus(Qt.TabFocusReason)
    key(Qt.Key_Delete)
    assert len(v1_clips(st)) == 1
    # the preview (it takes clicks) passes the keys on too; a check box would use Left / Right itself to move on
    st.set_playhead(1.0)
    win.monitor.view.setFocus(Qt.MouseFocusReason)
    settle(qapp)
    assert not keeps_its_keys(QApplication.focusWidget()) and not win.timeline.hasFocus()
    before = st.playhead
    key(Qt.Key_Right)
    assert st.playhead > before
    # a line edit, a number box and a time box keep every letter
    clips, play = len(v1_clips(st)), st.playhead
    for box in (win.inspector.sequence_tab.f_name, tab.f_speed, tab.f_start):
        win.inspector.setCurrentWidget(win.inspector.sequence_tab if box is win.inspector.sequence_tab.f_name
                                       else tab)
        settle(qapp)
        click_into(qapp, box)
        box.selectAll()
        QTest.keyClicks(box, "s")
        key(Qt.Key_Delete)
        key(Qt.Key_Right)
        key(Qt.Key_Up)                                    # a line edit ignores Up / Down: still not the timeline's 'previous cut'
        key(Qt.Key_Down)
        assert len(v1_clips(st)) == clips and st.playhead == play, type(box).__name__
        box.clearFocus()
        settle(qapp)


def test_which_widgets_keep_their_keys():
    for kept in (QLineEdit(), QDoubleSpinBox(), QPlainTextEdit(), QListWidget(), QTableWidget()):
        assert keeps_its_keys(kept), type(kept).__name__
    combo = QComboBox()
    combo.setEditable(True)
    assert keeps_its_keys(combo.lineEdit())
    for passed_on in (QPushButton(), QCheckBox(), QComboBox(), QSlider(), None):
        assert not keeps_its_keys(passed_on), type(passed_on).__name__


@pytest.mark.parametrize("typed", ["1000000000", "99:59:59:29", "1e308", "86400"])
def test_a_start_time_beyond_a_day_is_refused_with_a_message_and_the_timeline_stays_usable(qapp, clip_window, typed):
    """Typing 1000000000 into the clip's Start box put the clip 31 years out; the timeline then raised OverflowError in the
    slot that lays it out.  Now the edit is refused like any other impossible one: a message, the box shows the old
    value again, nothing changes and nothing is added to the undo history."""
    win, st = clip_window, clip_window.state
    tab = win.inspector.clip_tab
    before, depth = st.project.content_json(), len(st.history._undo)
    shown = tab.f_start.text()
    messages: list[str] = []
    st.status.connect(messages.append)
    type_and_enter(qapp, tab.f_start, typed)
    assert st.project.content_json() == before and len(st.history._undo) == depth
    assert any("24" in m for m in messages), messages
    assert tab.f_start.text() == shown, "the box goes back to the real start"
    win.timeline.zoom_fit()
    settle(qapp)
    assert win.timeline.clip_rect(v1_clips(st)[0].id).width() > 5 and win.timeline.grab().width() > 0


def test_a_start_time_just_inside_the_day_works_and_undo_brings_the_clip_back(qapp, clip_window):
    win, st = clip_window, clip_window.state
    tab = win.inspector.clip_tab
    clip = v1_clips(st)[0]
    last = int(24 * 3600 - clip.duration) - 1                                  # whole seconds, so the clip ends inside the day
    type_and_enter(qapp, tab.f_start, str(last))
    moved = v1_clips(st)[0]
    assert moved.start == pytest.approx(last, abs=1.0 / st.fps) and moved.end <= 24 * 3600
    win.timeline.zoom_fit()
    settle(qapp)
    assert win.timeline.grab().width() > 0 and win.timeline.pps >= 2.0
    st.undo()
    settle(qapp)
    win.timeline.zoom_fit()
    settle(qapp)
    assert v1_clips(st)[0].start == 0.0 and win.timeline.clip_rect(moved.id).width() > 5


# ============================================================================ window-wide drag & drop / import
class FakeDialog:
    """Stands in for the auto-edit dialogs (their file is being rewritten elsewhere): records what it was given."""

    log: list[tuple] = []

    def __init__(self, *args) -> None:
        self.args = args
        FakeDialog.log.append(args)
        self.executed = False

    def exec(self) -> int:
        self.executed = True
        return 0

    def deleteLater(self) -> None:
        pass


@pytest.fixture
def fake_dialogs(monkeypatch):
    FakeDialog.log = []

    class Shorts(FakeDialog):
        kind = "shorts"

    class Assemble(FakeDialog):
        kind = "assemble"

    monkeypatch.setattr(mw, "ShortsDialog", Shorts)
    monkeypatch.setattr(mw, "AssembleDialog", Assemble)
    return FakeDialog


def overlay_has_text_pixels(win: mw.MainWindow) -> bool:
    img = win.overlay.grab().toImage()
    bright = 0
    cx, cy = img.width() // 2, img.height() // 2
    for y in range(cy - 60, cy + 20, 2):
        for x in range(cx - 200, cx + 200, 2):
            c = img.pixelColor(x, y)
            if c.red() > 235 and c.green() > 235 and c.blue() > 235 and c.alpha() > 200:
                bright += 1
    return bright > 30


def test_drag_over_window_shows_overlay_and_leaving_hides_it(qapp, make_window, extra):
    win = make_window()
    dnd = Dnd(win, [extra["short_land"]])
    ev = dnd.enter()
    assert ev.isAccepted() and ev.dropAction() == Qt.CopyAction
    assert win.overlay.isVisible() and win.overlay.geometry() == win.rect()
    assert win.overlay.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert overlay_has_text_pixels(win), "the 「放開以匯入」 headline is not painted"
    assert dnd.move().isAccepted() and win.overlay.isVisible()
    shot(win, "drop_overlay")
    dnd.leave()
    assert not win.overlay.isVisible()
    assert not win.state.project.assets and not win.importer.runners
    win.resize(1100, 700)                                  # the overlay follows the window while visible
    dnd.enter()
    settle(qapp)
    assert win.overlay.geometry() == win.rect()
    dnd.leave()


def test_drop_hides_overlay_and_imports_files_and_folders(qapp, make_window, extra, media_dir, tmp_path):
    win = make_window()
    folder = tmp_path / "素材 資料夾"
    folder.mkdir()
    (folder / "b.mp4").write_bytes((extra["short_land"]).read_bytes())
    (folder / "a.mp4").write_bytes((extra["short_land2"]).read_bytes())
    (folder / "readme.txt").write_text("not media", encoding="utf-8")
    dnd = Dnd(win, [media_dir / "music.wav", folder])
    dnd.enter()
    assert win.overlay.isVisible()
    ev = dnd.drop()
    assert ev.isAccepted() and not win.overlay.isVisible()
    assert win.importer.runners and win.import_box.isVisible()
    finish_imports(qapp, win)
    assert not win.import_box.isVisible()
    names = [a.name for a in win.state.project.assets.values()]
    assert names == ["music.wav", "a.mp4", "b.mp4"]                    # drop order, folder content in natural order
    assert win.state.history.undo_label == "匯入素材"
    win.undo()                                                         # the whole drop is ONE undo step
    assert not win.state.project.assets


def test_drag_of_things_that_are_not_local_files_is_refused(qapp, make_window):
    win = make_window()
    text = QMimeData()
    text.setText("just text")
    web = QMimeData()
    web.setUrls([QUrl("https://example.com/video.mp4")])
    for mime in (text, web):
        ev = QDragEnterEvent(QPoint(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(win, ev)
        assert not ev.isAccepted() and not win.overlay.isVisible()
        drop = QDropEvent(QPointF(50, 50), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(win, drop)
        assert not drop.isAccepted() and not win.importer.runners


def test_one_long_landscape_video_offers_the_shorts_banner_and_its_buttons_work(qapp, make_window, extra, fake_dialogs):
    win = make_window()
    drop_and_import(qapp, win, [extra["long"]])
    aid = next(iter(win.state.project.assets))
    b = win.banner
    assert b.isVisible() and b.kind == "shorts" and b.asset_ids == [aid]
    assert "偵測到 1 支 16:9 長影片「long_land.mp4」" in b.label.text()
    assert "要自動剪成 9:16 短影音嗎？" in b.label.text()
    assert (b.btn_auto.text(), b.btn_place.text(), b.btn_skip.text()) == ("自動剪成短影音", "放到時間軸", "略過")
    shot(win, "banner_shorts")
    assert not clips_of(win.state)                                     # the banner never edits by itself

    QTest.mouseClick(b.btn_auto, Qt.LeftButton)                       # 1) auto-edit dialog for this video
    assert not b.isVisible()
    assert len(fake_dialogs.log) == 1
    state_arg, asset_arg, parent_arg = fake_dialogs.log[-1]
    assert state_arg is win.state and asset_arg == aid and parent_arg is win

    drop_and_import(qapp, win, [extra["long"]])                        # same file again: known asset, banner returns
    assert b.isVisible() and len(win.state.project.assets) == 1
    QTest.mouseClick(b.btn_skip, Qt.LeftButton)                       # 2) dismiss: nothing changes
    assert not b.isVisible() and not clips_of(win.state) and len(fake_dialogs.log) == 1

    drop_and_import(qapp, win, [extra["long"]])
    rec = Recorder(win.state)
    QTest.mouseClick(b.btn_place, Qt.LeftButton)                      # 3) put it on the timeline
    assert not b.isVisible()
    clip = v1_clips(win.state)[0]
    assert (clip.asset_id, clip.start) == (aid, 0.0) and clip.duration == pytest.approx(46.0, abs=0.2)
    assert rec.labels == ["放到時間軸"]
    seq = win.state.sequence()
    assert (seq.width, seq.height) == (1920, 1080)
    win.undo()
    assert not clips_of(win.state)


def test_two_videos_offer_the_assemble_banner_and_its_buttons_work(qapp, make_window, extra, media_dir, fake_dialogs):
    win = make_window()
    drop_and_import(qapp, win, [extra["short_land"], media_dir / "music.wav", extra["short_land2"], extra["image"]])
    ids = {a.name: a.id for a in win.state.project.assets.values()}
    b = win.banner
    assert b.isVisible() and b.kind == "assemble"
    assert b.asset_ids == [ids["short_land.mp4"], ids["short_land2.mp4"]]          # only the videos, in drop order
    assert b.label.text() == "偵測到 2 支素材，要自動剪成一支長片嗎？"
    assert (b.btn_auto.text(), b.btn_place.text(), b.btn_skip.text()) == ("自動合併成長片", "依順序放到時間軸", "略過")
    shot(win, "banner_assemble")
    QTest.mouseClick(b.btn_auto, Qt.LeftButton)
    assert not b.isVisible()
    state_arg, ids_arg, parent_arg = fake_dialogs.log[-1]
    assert state_arg is win.state and ids_arg == [ids["short_land.mp4"], ids["short_land2.mp4"]] and parent_arg is win

    drop_and_import(qapp, win, [extra["short_land"], extra["short_land2"]])
    rec = Recorder(win.state)
    QTest.mouseClick(b.btn_place, Qt.LeftButton)
    assert not b.isVisible() and rec.labels == ["放到時間軸"]
    first, second = v1_clips(win.state)
    assert (first.asset_id, second.asset_id) == (ids["short_land.mp4"], ids["short_land2.mp4"])
    assert first.start == 0.0 and second.start == pytest.approx(first.end)         # back to back, in order
    drop_and_import(qapp, win, [extra["short_land"], extra["short_land2"]])
    QTest.mouseClick(b.btn_skip, Qt.LeftButton)
    assert not b.isVisible() and len(v1_clips(win.state)) == 2


def test_other_media_goes_to_the_bin_only(qapp, make_window, extra, media_dir):
    win = make_window()
    for path in (extra["short_land"], media_dir / "music.wav", extra["image"], media_dir / "port.mp4"):
        drop_and_import(qapp, win, [extra["long"]])                    # a banner is showing ...
        assert win.banner.isVisible()
        drop_and_import(qapp, win, [path])                             # ... and a plain drop dismisses it
        assert not win.banner.isVisible(), path
        assert "雙擊素材或把它拖到時間軸" in win.statusBar().currentMessage()
    assert len(win.state.project.assets) == 5 and not clips_of(win.state)
    assert win.state.history.undo_label == "匯入素材"
    # one long landscape video plus a song is still ONE video: the shorts suggestion
    drop_and_import(qapp, win, [extra["long"], media_dir / "music.wav"])
    assert win.banner.isVisible() and win.banner.kind == "shorts"


def test_bad_files_are_reported_but_do_not_stop_the_rest(qapp, make_window, stubs, extra, media_dir, tmp_path):
    win = make_window()
    broken = tmp_path / "壞掉.mp4"
    broken.write_bytes(b"not a video" * 40)
    notes = tmp_path / "notes.txt"
    notes.write_text("x", encoding="utf-8")
    drop_and_import(qapp, win, [broken, media_dir / "port.mp4", notes, tmp_path / "gone.mp4"])
    assert [a.name for a in win.state.project.assets.values()] == ["port.mp4"]
    assert len(stubs.informs) == 1 and stubs.informs[0][2] is True
    title, text, _w = stubs.informs[0]
    assert title == "部分檔案無法匯入" and "壞掉.mp4" in text and "notes.txt" in text and "gone.mp4" in text
    assert "port.mp4" not in text


def test_place_on_empty_sequence_adapts_format_to_first_video_in_one_undo_step(qapp, make_window, extra, media_dir):
    """Portrait first -> 1080x1920, landscape first -> 1920x1080; fps follows the footage within 24-60; ONE step."""
    win = make_window()
    st = win.state
    base = (st.sequence().width, st.sequence().height, st.sequence().fps)
    assert base == (1920, 1080, 30.0)
    # portrait first
    drop_and_import(qapp, win, [media_dir / "port.mp4", media_dir / "land.mp4"])
    rec = Recorder(st)
    QTest.mouseClick(win.banner.btn_place, Qt.LeftButton)
    seq = st.sequence()
    assert (seq.width, seq.height, seq.fps) == (1080, 1920, 30.0)
    assert rec.labels == ["放到時間軸"] and len(v1_clips(st)) == 2
    win.undo()                                                          # a single Ctrl+Z restores size AND clips
    seq = st.sequence()
    assert (seq.width, seq.height, seq.fps) == base and not clips_of(st)
    # landscape first (re-drop: the assets are known, the suggestion comes back)
    drop_and_import(qapp, win, [media_dir / "land.mp4", media_dir / "port.mp4"])
    QTest.mouseClick(win.banner.btn_place, Qt.LeftButton)
    seq = st.sequence()
    assert (seq.width, seq.height) == (1920, 1080) and len(v1_clips(st)) == 2
    # a sequence that already has clips keeps its format
    st.op("格式", "set_sequence_format", 1080, 1080, 30.0)
    drop_and_import(qapp, win, [media_dir / "port.mp4", media_dir / "land.mp4"])
    QTest.mouseClick(win.banner.btn_place, Qt.LeftButton)
    assert (st.sequence().width, st.sequence().height) == (1080, 1080) and len(v1_clips(st)) == 4


@pytest.mark.parametrize("key, fps", [("fps15", 24.0), ("fps120", 60.0), ("ntsc", 23.976)])
def test_place_clamps_frame_rate_to_24_60(qapp, make_window, extra, key, fps):
    win = make_window()
    drop_and_import(qapp, win, [extra[key]])
    assert not win.banner.isVisible()
    win.media_bin.list.selectAll()
    rec = Recorder(win.state)
    assert win.media_bin.place_selected()
    assert win.state.sequence().fps == pytest.approx(fps, abs=0.001)
    assert rec.labels == ["放到時間軸"] and len(clips_of(win.state)) == 1


def test_files_dropped_on_the_timeline_are_imported_then_placed_at_the_drop_position(qapp, make_window, media_dir):
    win = make_window()
    st = win.state
    track = st.sequence().video_tracks[1]
    rec = Recorder(st)
    win.timeline.externalFilesDropped.emit([str(media_dir / "land.mp4")], 2.0, track.id)
    finish_imports(qapp, win)
    assert rec.labels == ["匯入素材", "放到時間軸"]
    assert not win.banner.isVisible()                                   # dropping straight on the timeline: no banner
    (clip,) = [c for c in track.clips]
    assert clip.start == pytest.approx(2.0) and clip.asset_id == next(iter(st.project.assets))
    assert not v1_clips(st)


def test_drop_of_a_project_file_opens_it(qapp, make_window, stubs, real_proj, tmp_path):
    win = make_window()
    saved = Project.new("來自檔案")
    path = saved.save(tmp_path / f"drop{PROJECT_EXT}")
    drop_and_import(qapp, win, [path])
    assert win.state.project.name == "來自檔案" and win.state.project_path == str(path)
    assert not win.state.dirty and not stubs.informs
    # an unreadable project file is explained, the open project stays
    bad = tmp_path / f"bad{PROJECT_EXT}"
    bad.write_text("{ not json", encoding="utf-8")
    drop_and_import(qapp, win, [bad])
    assert stubs.informs and stubs.informs[-1][0] == "無法開啟專案" and win.state.project.name == "來自檔案"


def test_import_progress_and_cancel_in_the_status_bar(qapp, make_window, extra, tmp_path):
    win = make_window()
    folder = tmp_path / "many"
    folder.mkdir()
    for i in range(6):
        (folder / f"c{i}.mp4").write_bytes(extra["short_land"].read_bytes())
    runner = win.import_paths([str(folder)])
    assert win.import_box.isVisible() and win.import_label.text() == "匯入中" and win.import_cancel.isVisible()
    win.import_cancel.click()                                           # 「取消匯入」: nothing of it is kept
    wait_for_idle(qapp, [runner], timeout=60)
    assert not win.import_box.isVisible()
    assert not win.state.project.assets and not win.state.history.can_undo
    assert win.statusBar().currentMessage() == "已取消匯入"


# ============================================================================ subtitle files
def cue_texts(state: AppState) -> list[tuple[float, float, str]]:
    return [(round(q.start, 3), round(q.end, 3), q.text) for q in state.sequence().cues]


def test_subtitle_drop_asks_then_replaces_cues_in_one_undo_step(qapp, make_window, stubs, extra):
    win = make_window()
    st = win.state
    rec = Recorder(st)
    stubs.ask_answers = [True]
    dnd = Dnd(win, [extra["srt"]])
    dnd.enter()
    dnd.drop()
    finish_imports(qapp, win)
    assert not st.project.assets                                   # a subtitle file is never an asset
    title, text, yes, no = stubs.asks[-1]
    assert title == "匯入字幕" and "字幕.srt" in text and "3 句字幕" in text and "序列 1" in text
    assert "取代" not in text and yes == "套用字幕"
    assert cue_texts(st) == [(0.5, 2.0, "你好"), (2.5, 4.0, "世界"), (4.5, 6.0, "再見")]
    assert rec.labels == ["匯入字幕"] and win.statusBar().currentMessage() == "已匯入 3 句字幕"
    win.undo()
    assert not st.sequence().cues


def test_subtitle_drop_over_existing_cues_warns_and_can_be_declined(qapp, make_window, stubs, extra):
    win = make_window()
    st = win.state
    stubs.ask_default = True
    drop_and_import(qapp, win, [extra["srt"]])
    before_ids = [q.id for q in st.sequence().cues]
    rec = Recorder(st)
    stubs.ask_answers = [False]                                    # decline: nothing changes, no undo step
    drop_and_import(qapp, win, [extra["vtt"]])
    assert "取代目前序列裡的 3 句字幕" in stubs.last_ask() and "Ctrl+Z" in stubs.last_ask()
    assert [q.id for q in st.sequence().cues] == before_ids and not rec.events
    stubs.ask_answers = [True]
    drop_and_import(qapp, win, [extra["vtt"]])                     # VTT: "." decimals and a WEBVTT header
    assert cue_texts(st) == [(1.0, 2.5, "hello vtt"), (3.0, 4.0, "second")]
    stubs.ask_answers = [True]
    drop_and_import(qapp, win, [extra["ass"]])
    assert cue_texts(st) == [(1.0, 3.0, "ass line one"), (3.5, 5.0, "ass line two")]
    assert rec.labels == ["匯入字幕", "匯入字幕"]


def test_subtitle_drop_with_media_and_several_subtitle_files(qapp, make_window, stubs, extra, media_dir):
    win = make_window()
    stubs.ask_answers = [True]
    drop_and_import(qapp, win, [media_dir / "port.mp4", extra["srt"], extra["vtt"]])
    assert [a.name for a in win.state.project.assets.values()] == ["port.mp4"]
    assert "另外 1 個已略過" in stubs.last_ask()
    assert [t for _s, _e, t in cue_texts(win.state)] == ["你好", "世界", "再見"]      # the first file only


def test_unreadable_or_empty_subtitle_files_are_explained(qapp, make_window, stubs, extra, tmp_path):
    win = make_window()
    rec = Recorder(win.state)
    drop_and_import(qapp, win, [extra["empty_srt"]])
    assert stubs.informs[-1][:1] == ("字幕讀取失敗",) and "empty.srt" in stubs.informs[-1][1]
    assert not stubs.asks and not rec.events
    assert win.apply_subtitle_files([str(tmp_path / "missing.srt")]) is False
    assert len(stubs.informs) == 2 and "missing.srt" in stubs.informs[-1][1]
    garbage = tmp_path / "garbage.vtt"
    garbage.write_bytes(bytes(range(256)) * 4)
    assert win.apply_subtitle_files([str(garbage)]) is False
    assert not win.state.sequence().cues and not rec.events


# ============================================================================ project files
def saved_project(tmp_path: Path, name: str, clip_asset: Path | None = None) -> Path:
    p = Project.new(name)
    if clip_asset is not None:
        a = probe_asset(clip_asset)
        p.assets[a.id] = a
        ops.place_clip(p, p.sequences[0], a.id, p.sequences[0].video_tracks[0].id, 0.0)
    return Path(p.save(tmp_path / f"{name}{PROJECT_EXT}"))


def test_save_as_writes_the_file_and_remembers_it(qapp, make_window, stubs, real_proj, tmp_path):
    win = make_window(real_proj)
    st = win.state
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    st.op("放", "place_clip", land.id, st.sequence().video_tracks[0].id, 0.0)
    assert st.dirty and not st.project_path
    mw.write_autosave(st.project)
    assert mw.autosave_path(st.project.id).is_file()
    stubs.save_file = str(tmp_path / "作品")                              # no extension typed
    assert win.act_save.isEnabled() and win.save_project()
    target = tmp_path / f"作品{PROJECT_EXT}"
    assert target.is_file() and st.project_path == str(target)
    assert not st.dirty and win.windowTitle() == f"作品{PROJECT_EXT} — AutoCut Studio"
    assert Project.load(target).content_json() == st.project.content_json()
    assert win.recent_projects() == [str(target)]
    assert mw.app_settings().value("dialog/project_dir") == str(tmp_path)
    assert not mw.autosave_path(st.project.id).exists()                 # nothing left to recover
    assert st.project.name == "作品"                                      # 「另存」 names the project after the file
    # saving again goes straight to the same file (no dialog) and keeps the previous version as .bak
    st.op("標記", "add_marker", 1.0)
    dialogs = len(stubs.file_dialogs)
    assert win.save_project() and len(stubs.file_dialogs) == dialogs
    assert (tmp_path / f"作品{PROJECT_EXT}.bak").is_file() and not st.dirty
    assert len(Project.load(target).sequences[0].markers) == 1


def test_save_cancelled_or_failing_keeps_the_project_dirty(qapp, make_window, stubs, real_proj, tmp_path):
    win = make_window(real_proj)
    win.state.op("標記", "add_marker", 1.0)
    stubs.save_file = ""                                                   # the dialog was cancelled
    assert win.save_project() is False and win.state.dirty and not stubs.informs
    stubs.save_file = str(tmp_path / "no" / "such" / "dir" / "x")
    (tmp_path / "no").write_text("a file where a folder should be", encoding="utf-8")
    assert win.save_project_as() is False and win.state.dirty
    assert stubs.informs[-1][0] == "無法儲存專案" and stubs.informs[-1][2] is True


def test_new_project_asks_about_unsaved_changes(qapp, make_window, stubs, real_proj, tmp_path):
    win = make_window(real_proj)
    st = win.state
    old_id = st.project.id
    st.op("標記", "add_marker", 1.0)
    stubs.save_choices = ["cancel"]
    assert win.act_new.isEnabled() and win.new_project() is False
    assert st.project.id == old_id and stubs.save_prompts == ["real"]      # asked with the project's name
    stubs.save_choices = ["discard"]
    mw.write_autosave(st.project)
    assert win.new_project() is True
    assert st.project.id != old_id and st.project.name == "未命名專案" and not st.dirty
    assert not mw.autosave_path(old_id).exists() and win.statusBar().currentMessage() == "已建立新專案"
    assert not clips_of(st) and not st.history.can_undo
    st.op("標記", "add_marker", 1.0)
    stubs.save_file = str(tmp_path / "先存")
    stubs.save_choices = ["save"]                                          # 「儲存」 -> Save As dialog -> then new
    saved_id = st.project.id
    assert win.new_project() is True
    assert (tmp_path / f"先存{PROJECT_EXT}").is_file() and st.project.id != saved_id
    assert Project.load(tmp_path / f"先存{PROJECT_EXT}").id == saved_id
    stubs.save_choices = ["save"]                                          # 「儲存」 but the dialog is cancelled
    st.op("標記", "add_marker", 1.0)
    stubs.save_file = ""
    kept = st.project.id
    assert win.new_project() is False and st.project.id == kept and st.dirty


def test_open_project_via_dialog_with_unsaved_changes_prompt(qapp, make_window, stubs, real_proj, media_dir, tmp_path):
    other = saved_project(tmp_path, "別的專案", media_dir / "land.mp4")
    win = make_window(real_proj)
    st = win.state
    st.op("標記", "add_marker", 1.0)
    stubs.open_file = str(other)
    stubs.save_choices = ["cancel"]
    win.act_open.trigger()
    assert st.project.name == "real" and stubs.file_dialogs == ["open"]     # cancelled: nothing opened
    stubs.save_choices = ["discard"]
    win.act_open.trigger()
    assert st.project.name == "別的專案" and st.project_path == str(other) and not st.dirty
    assert len(clips_of(st)) == 1 and win.windowTitle() == f"別的專案{PROJECT_EXT} — AutoCut Studio"
    assert win.recent_projects() == [str(other)]
    assert win.statusBar().currentMessage() == f"已開啟 別的專案{PROJECT_EXT}"
    assert not st.history.can_undo                                         # a fresh project has no undo history
    # save is the answer: the current (unsaved) project is written first, then the other one opens
    st.op("標記", "add_marker", 1.0)
    stubs.save_choices = ["save"]
    win.open_project(str(saved_project(tmp_path, "第三個")))
    assert st.project.name == "第三個"
    assert len(Project.load(other).sequences[0].markers) == 1


def test_open_broken_or_missing_project_keeps_the_current_one(qapp, make_window, stubs, real_proj, tmp_path):
    win = make_window(real_proj)
    bad = tmp_path / f"bad{PROJECT_EXT}"
    bad.write_text('{"schema": "something.else/1"}', encoding="utf-8")
    assert win.open_project(str(bad)) is False
    assert win.open_project(str(tmp_path / f"gone{PROJECT_EXT}")) is False
    assert [i[0] for i in stubs.informs] == ["無法開啟專案", "無法開啟專案"] and all(i[2] for i in stubs.informs)
    assert "bad.acproj" in stubs.informs[0][1] and "gone.acproj" in stubs.informs[1][1]
    assert win.state.project.name == "real" and not win.recent_projects()


def test_open_project_with_missing_media_says_so(qapp, make_window, stubs, media_dir, tmp_path):
    clip = tmp_path / "moved.mp4"
    clip.write_bytes((media_dir / "land.mp4").read_bytes())
    path = saved_project(tmp_path, "搬家", clip)
    clip.unlink()
    win = make_window()
    assert win.open_project(str(path))
    assert "有 1 個素材找不到檔案" in win.statusBar().currentMessage()
    assert win.state.project.assets and next(iter(win.state.project.assets.values())).missing


def test_recent_projects_menu(qapp, make_window, stubs, tmp_path):
    win = make_window()
    paths = [saved_project(tmp_path, f"p{i}") for i in range(12)]
    win.recent_menu.aboutToShow.emit()
    assert [a.text() for a in win.recent_menu.actions()] == ["（沒有最近的專案）"]
    assert not win.recent_menu.actions()[0].isEnabled()
    for p in paths:
        win.open_project(str(p))
    assert win.recent_projects() == [str(p) for p in reversed(paths)][:mw.MAX_RECENT]       # newest first, capped
    win.open_project(str(paths[5]))                                         # reopening moves it to the top, no dupes
    recent = win.recent_projects()
    assert recent[0] == str(paths[5]) and len(recent) == len(set(recent)) == mw.MAX_RECENT
    paths[11].unlink()                                                      # vanished files are dropped from the menu
    win.recent_menu.aboutToShow.emit()
    texts = [a.text() for a in win.recent_menu.actions() if a.text()]
    assert texts[0] == "p5.acproj" and "p11.acproj" not in texts and texts[-1] == "清除記錄"
    assert str(paths[11]) not in win.recent_projects()
    entry = next(a for a in win.recent_menu.actions() if a.text() == "p9.acproj")
    entry.trigger()
    assert win.state.project.name == "p9" and win.recent_projects()[0] == str(paths[9])
    next(a for a in win.recent_menu.actions() if a.text() == "清除記錄").trigger()
    assert win.recent_projects() == []


# ============================================================================ autosave / recovery
def autosave_files() -> list[Path]:
    return sorted(mw.autosave_dir().glob(f"*{PROJECT_EXT}"))


def project_with_clip(name: str, media: Path) -> Project:
    p = Project.new(name)
    a = probe_asset(media)
    p.assets[a.id] = a
    ops.place_clip(p, p.sequences[0], a.id, p.sequences[0].video_tracks[0].id, 0.0)
    p.touch()
    return p


def test_autosave_writes_only_unsaved_work_and_never_changes_the_project_path(qapp, make_window, stubs, real_proj,
                                                                              tmp_path):
    win = make_window(real_proj)
    st = win.state
    assert win.autosave() is None and not autosave_files()                # nothing to protect yet
    st.op("標記", "add_marker", 1.0)
    win.autosave_timer.timeout.emit()                                      # what the 60 s timer does
    (target,) = autosave_files()
    assert target == mw.autosave_path(st.project.id)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["autosave_of"] == "" and data["id"] == st.project.id
    assert Project.from_dict(data).content_json() == st.project.content_json()
    assert st.project.path is None and st.dirty
    stubs.save_file = str(tmp_path / "ok")
    assert win.save_project() and not autosave_files()                    # saving removes the autosave
    st.op("標記二", "add_marker", 2.0)
    win.autosave()
    data = json.loads(mw.autosave_path(st.project.id).read_text(encoding="utf-8"))
    assert data["autosave_of"] == str(tmp_path / f"ok{PROJECT_EXT}")
    assert st.project.path == str(tmp_path / f"ok{PROJECT_EXT}")
    st.editor.history.mark_saved()
    assert not st.dirty and win.autosave() is None                         # clean again: nothing is written


def test_startup_offers_to_recover_an_unsaved_project(qapp, make_window, stubs, media_dir):
    lost = project_with_clip("沒存過的工作", media_dir / "land.mp4")
    mw.write_autosave(lost)
    stubs.ask_answers = [True]
    win = make_window()
    title, text, yes, no = stubs.asks[0]
    assert title == "復原專案" and (yes, no) == ("復原", "捨棄")
    assert "沒存過的工作" in text and "這個專案還沒存過檔" in text
    st = win.state
    assert st.project.id == lost.id and len(clips_of(st)) == 1 and st.project.path is None
    assert st.dirty and win.windowTitle() == "沒存過的工作 * — AutoCut Studio"        # closing will ask first
    assert win.statusBar().currentMessage() == "已復原自動存檔，請記得儲存"
    assert win.act_save.isEnabled()


def test_startup_recovery_can_be_declined_and_asks_only_once(qapp, make_window, stubs, media_dir):
    mw.write_autosave(project_with_clip("丟掉", media_dir / "land.mp4"))
    stubs.ask_answers = [False]
    win = make_window()
    assert len(stubs.asks) == 1 and not autosave_files()                    # 「捨棄」 deletes the autosave
    assert win.state.project.name == "未命名專案" and not win.state.dirty and not clips_of(win.state)
    win.startup()                                                           # startup work runs once per window
    assert len(stubs.asks) == 1


def test_startup_recovery_points_at_the_original_file_and_ignores_older_autosaves(qapp, make_window, stubs, media_dir,
                                                                                   tmp_path):
    original = tmp_path / f"原檔{PROJECT_EXT}"
    proj = project_with_clip("原檔", media_dir / "land.mp4")
    proj.save(original)
    proj.touch()
    mw.write_autosave(proj)                                                 # autosave_of = original
    auto = mw.autosave_path(proj.id)
    later = time.time() + 30
    os.utime(auto, (later, later))                                          # newer than the saved file -> offer
    stubs.ask_answers = [True]
    win = make_window()
    assert "原檔：" in stubs.asks[0][1] and str(original) in stubs.asks[0][1]
    assert win.state.project.path == str(original) and win.state.dirty
    assert win.state.project_path == str(original)
    # an autosave OLDER than the saved file holds nothing new: no prompt, no recovery
    win2_asks = len(stubs.asks)
    stale = time.time() - 600
    os.utime(auto, (stale, stale))
    win2 = make_window()
    assert len(stubs.asks) == win2_asks and not win2.state.dirty and win2.state.project.id != proj.id


def test_corrupt_autosaves_never_crash_the_startup(qapp, make_window, stubs, media_dir):
    (mw.autosave_dir() / f"junk{PROJECT_EXT}").write_text("{ this is not json", encoding="utf-8")
    (mw.autosave_dir() / f"empty{PROJECT_EXT}").write_bytes(b"")
    win = make_window()
    assert not stubs.asks and not win.state.dirty
    # valid JSON that is not a project: the user is told, the blank project stays
    (mw.autosave_dir() / f"other{PROJECT_EXT}").write_text('{"id": "x", "name": "其他"}', encoding="utf-8")
    stubs.ask_answers = [True]
    win2 = make_window()
    assert stubs.asks and stubs.informs and stubs.informs[-1][0] == "自動存檔無法讀取" and stubs.informs[-1][2]
    assert not win2.state.dirty and win2.state.project.name == "未命名專案"
    # the unreadable file is moved aside (kept, not deleted) and never offered again
    assert not (mw.autosave_dir() / f"other{PROJECT_EXT}").exists()
    assert (mw.autosave_dir() / f"other{PROJECT_EXT}.unreadable").is_file()
    assert "other.acproj.unreadable" in stubs.informs[-1][1]
    asks, informs = len(stubs.asks), len(stubs.informs)
    win3 = make_window()
    assert len(stubs.asks) == asks and len(stubs.informs) == informs and not win3.state.dirty


def test_a_damaged_autosave_in_the_shape_of_a_project_is_moved_aside_once(qapp, make_window, stubs):
    """The reviewer's file: valid JSON with an id and a name, but ``sequences`` is text.  It used to raise out of the startup
    (Qt then aborted the process) and, being left in place, again at every start."""
    for name, text in (("p_bad", '{"id": "p_bad", "name": "壞掉的專案", "autosave_of": ""}'),
                       ("p_bad2", '{"id": "p_bad2", "name": "半截", "schema": "autocut.project/1", "sequences": "oops"}')):
        (mw.autosave_dir() / f"{name}{PROJECT_EXT}").write_text(text, encoding="utf-8")
    win = make_window()
    for _ in range(2):
        stubs.ask_answers = [True]
        win.offer_recovery()
    assert not autosave_files()
    assert sorted(f.name for f in mw.autosave_dir().glob("*.unreadable*")) == ["p_bad.acproj.unreadable",
                                                                                  "p_bad2.acproj.unreadable"]
    assert [i[0] for i in stubs.informs].count("自動存檔無法讀取") == 2             # told once per file, not at every start
    assert not win.state.dirty and win.state.project.name == "未命名專案"


def test_quarantine_never_overwrites_an_earlier_one(qapp, stubs):
    first = mw.autosave_dir() / f"same{PROJECT_EXT}"
    for n in range(3):
        first.write_text(f"{{\"n\": {n}}}", encoding="utf-8")
        moved = mw.quarantine_autosave(mw.Autosave(first, "same", "same", "", 0.0))
        assert moved is not None and not first.exists()
    kept = sorted(f.name for f in mw.autosave_dir().glob("*.unreadable*"))
    assert kept == ["same.acproj.unreadable", "same.acproj.unreadable2", "same.acproj.unreadable3"]
    assert mw.quarantine_autosave(mw.Autosave(mw.autosave_dir() / "gone.acproj", "gone", "gone", "", 0.0)) is None


DAMAGED_FILES = {                       # the same cases as tests/test_core_hardening.py (a few of them)
    "a list": "[]", "null": "null", "a number": "42",
    "sequences is a string": '{"schema":"autocut.project/1","name":"x","sequences":"oops"}',
    "assets is a dict of strings": '{"schema":"autocut.project/1","name":"x","assets":{"a":"x"},"sequences":[]}',
    "a clip that is a string": '{"schema":"autocut.project/1","name":"x","sequences":[{"video_tracks":[{"clips":["z"]}]}]}',
    "empty": "", "cut off": '{"schema":"autocut.project/1","name":"x","sequen',
    "nested without end": "[" * 100000 + "]" * 100000,
}


@pytest.mark.parametrize("case", list(DAMAGED_FILES))
def test_a_damaged_project_file_gives_a_message_and_leaves_the_open_project_alone(qapp, make_window, stubs, real_proj,
                                                                               tmp_path, case):
    win = make_window(real_proj)
    before = win.state.project.id
    path = tmp_path / f"壞檔{PROJECT_EXT}"
    path.write_text(DAMAGED_FILES[case], encoding="utf-8")
    assert win.open_project(str(path)) is False
    title, text, warning = stubs.informs[-1]
    assert title == "無法開啟專案" and warning and "壞檔.acproj" in text
    assert win.state.project.id == before and str(path) not in win.recent_projects()


def test_open_project_offers_its_newer_autosave_and_use_original_wins(qapp, make_window, stubs, media_dir, tmp_path):
    path = tmp_path / f"有自存{PROJECT_EXT}"
    proj = project_with_clip("有自存", media_dir / "land.mp4")
    proj.save(path)
    edited = Project.load(path)
    ops.add_marker(edited, edited.sequences[0], 1.5)
    edited.touch()
    win = make_window()
    # 「復原」: the autosaved edit comes back, the project counts as unsaved
    mw.write_autosave(edited)
    later = time.time() + 30
    os.utime(mw.autosave_path(proj.id), (later, later))
    stubs.ask_answers = [True]
    assert win.open_project(str(path))
    title, text, yes, no = stubs.asks[-1]
    assert title == "復原自動存檔" and (yes, no) == ("復原", "使用原檔") and "有自存.acproj" in text
    assert len(win.state.sequence().markers) == 1 and win.state.dirty and win.state.project.path == str(path)
    assert autosave_files()                                                 # still there until saved or closed
    # 「使用原檔」: the file on disk is opened and the autosave is thrown away
    win.state.editor.history.mark_saved()
    stubs.ask_answers = [False]
    assert win.open_project(str(path))
    assert len(win.state.sequence().markers) == 0 and not win.state.dirty and not autosave_files()


def test_command_line_project_and_media_are_opened_without_a_recovery_prompt(qapp, make_window, stubs, media_dir,
                                                                             tmp_path):
    mw.write_autosave(project_with_clip("別人的自存", media_dir / "land.mp4"))
    path = saved_project(tmp_path, "指令列")
    win = make_window(paths=[str(path), str(media_dir / "port.mp4")])
    finish_imports(qapp, win)
    assert not stubs.asks
    assert win.state.project.name == "指令列" and win.state.project_path == str(path)
    assert [a.name for a in win.state.project.assets.values()] == ["port.mp4"]
    # only media on the command line: the recovery offer comes first (declined here), then the files are imported
    stubs.ask_answers = [False]
    win2 = make_window(paths=[str(media_dir / "silent.mp4")])
    finish_imports(qapp, win2)
    assert len(stubs.asks) == 1 and [a.name for a in win2.state.project.assets.values()] == ["silent.mp4"]
    assert win2.state.project.name == "未命名專案" and not autosave_files()


# ============================================================================ closing
def test_close_releases_monitor_and_playback_and_saves_the_layout(qapp, make_window, stubs, real_proj, monkeypatch):
    win = make_window(real_proj)
    st = win.state
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    st.op("放", "place_clip", land.id, st.sequence().video_tracks[0].id, 0.0)
    st.editor.history.mark_saved()
    order: list[str] = []
    for owner, name in ((win.monitor, "shutdown"), (win.playback, "close")):
        real = getattr(owner, name)
        monkeypatch.setattr(owner, name, lambda real=real, tag=f"{type(owner).__name__}.{name}":
                            (order.append(tag), real())[1])
    assert win.playback.play() and win.playback.is_playing
    win.splitter.setSizes([300, 500])
    win.media_bin.set_icon_mode(True)
    assert QMainWindow.close(win)
    assert order == ["MonitorWidget.shutdown", "PlaybackController.close"]
    assert not win.playback.is_playing and not win.playback._timer.isActive()
    assert win.monitor._stopped and not win.monitor._worker._thread.is_alive()
    assert not win.autosave_timer.isActive() and not win.isVisible()
    s = mw.app_settings()
    assert s.value("window/geometry") and s.value("window/state") and s.value("window/splitter")
    assert str(s.value("bin/icon_mode")).lower() == "true"
    # the next window comes back with the saved layout
    win2 = make_window()
    assert win2.media_bin.icon_mode()
    sizes = win2.splitter.sizes()
    assert sizes[1] > sizes[0] * 0.9, sizes


def test_close_survives_a_failing_preview_helper(qapp, make_window, stubs, monkeypatch):
    win = make_window()
    monkeypatch.setattr(win.monitor, "shutdown", lambda: (_ for _ in ()).throw(RuntimeError("decoder stuck")))
    assert QMainWindow.close(win)
    assert mw.app_settings().value("window/geometry")


def test_close_asks_about_unsaved_changes(qapp, make_window, stubs, real_proj, tmp_path):
    win = make_window(real_proj)
    st = win.state
    st.op("標記", "add_marker", 1.0)
    mw.write_autosave(st.project)
    stubs.save_choices = ["cancel"]
    assert QMainWindow.close(win) is False and win.isVisible()               # cancelled: window stays
    assert stubs.save_prompts == ["real"] and autosave_files()
    assert win.autosave_timer.isActive()
    stubs.save_choices = ["save"]
    stubs.save_file = ""                                                     # 「儲存」 -> dialog cancelled -> stay
    assert QMainWindow.close(win) is False and win.isVisible()
    stubs.save_file = str(tmp_path / "關閉前")
    stubs.save_choices = ["save"]
    assert QMainWindow.close(win) and not win.isVisible()
    assert (tmp_path / f"關閉前{PROJECT_EXT}").is_file() and not autosave_files()
    win2 = make_window(real_proj)
    win2.state.op("標記", "add_marker", 2.0)
    mw.write_autosave(win2.state.project)
    stubs.save_choices = ["discard"]
    assert QMainWindow.close(win2) and not autosave_files()                  # closed on purpose: nothing to recover


def test_close_stops_every_running_background_task_and_waits_for_it(qapp, make_window):
    """An analysis, a render or a caption run that keeps going keeps its ffmpeg child (and the process) alive after the window
    is gone: closing cancels all of them and returns only when the worker functions have returned."""
    win = make_window()
    seen: dict[str, bool] = {}

    def slow(tag):
        def fn(_progress, cancel):
            seen[tag + ":started"] = True
            for _ in range(2000):                                           # at most 20 s, normally a few ms
                if cancel.is_set():
                    seen[tag + ":cancelled"] = True
                    return None
                time.sleep(0.01)
            return None
        return fn

    runners = [TaskRunner(slow("a"), name="a").start(), TaskRunner(slow("b"), name="b").start()]
    deadline = time.monotonic() + 5
    while not (seen.get("a:started") and seen.get("b:started")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen.get("a:started") and seen.get("b:started")
    assert QMainWindow.close(win)
    assert not any(r.running for r in runners), "close returned while a background task was still running"
    assert seen.get("a:cancelled") and seen.get("b:cancelled")
    wait_for_idle(qapp, runners)


def test_close_cancels_running_imports(qapp, make_window, extra, tmp_path):
    win = make_window()
    folder = tmp_path / "slow"
    folder.mkdir()
    for i in range(4):
        (folder / f"s{i}.mp4").write_bytes(extra["short_land"].read_bytes())
    runner = win.import_paths([str(folder)])
    assert QMainWindow.close(win)
    assert runner.cancel_event.is_set()
    wait_for_idle(qapp, [runner], timeout=60)
    assert not win.state.project.assets


# ============================================================================ sequence tabs
def seq_names(state: AppState) -> list[str]:
    return [s.name for s in state.project.sequences]


def test_sequence_tabs_add_switch_duplicate_delete_rename(qapp, make_window, stubs, real_proj):
    win = make_window(real_proj)
    st, tabs = win.state, win.sequence_tabs
    rec = Recorder(st)
    first = st.sequence()
    assert tabs.titles() == ["序列 1  ·  16:9"] and not tabs.btn_del.isEnabled()
    presets = tabs.btn_add.menu().actions()
    assert [a.text() for a in presets] == ["16:9 橫式（1920×1080）", "9:16 直式（1080×1920）", "1:1 方形（1080×1080）",
                                           "4:5 直式（1080×1350）"]
    presets[1].trigger()                                                      # new 9:16 sequence
    second = st.sequence()
    assert seq_names(st) == ["序列 1", "序列 2"] and st.seq_id == second.id and second.id != first.id
    assert (second.width, second.height, second.fps) == (1080, 1920, first.fps) and not list(second.all_clips())
    assert tabs.titles() == ["序列 1  ·  16:9", "序列 2  ·  9:16"] and tabs.tabs.currentIndex() == 1
    assert rec.labels == ["新增序列"] and tabs.btn_del.isEnabled()
    win.undo()                                                                # ONE step: the sequence is gone again
    assert seq_names(st) == ["序列 1"] and st.seq_id == first.id and tabs.titles() == ["序列 1  ·  16:9"]
    win.redo()
    assert seq_names(st) == ["序列 1", "序列 2"] and tabs.titles()[1] == "序列 2  ·  9:16"
    presets[2].trigger()
    presets[3].trigger()
    assert seq_names(st) == ["序列 1", "序列 2", "序列 3", "序列 4"]
    assert [(s.width, s.height) for s in st.project.sequences] == [(1920, 1080), (1080, 1920), (1080, 1080), (1080, 1350)]
    assert tabs.titles()[2:] == ["序列 3  ·  1:1", "序列 4  ·  4:5"]
    # switching tabs changes the active sequence, resets playhead and selection
    land = next(a for a in real_proj.assets.values() if a.name == "land.mp4")
    tabs.tabs.setCurrentIndex(0)
    assert st.seq_id == first.id and win.seq_info.text().startswith("序列 1 · 1920×1080")
    st.op("放", "place_clip", land.id, st.sequence().video_tracks[0].id, 0.0)
    st.select([v1_clips(st)[0].id])
    st.set_playhead(2.0)
    tabs.tabs.setCurrentIndex(2)
    assert st.seq_id == st.project.sequences[2].id and not st.selection and st.playhead == 0.0
    assert win.seq_info.text().startswith("序列 3 · 1080×1080")
    # duplicate the first sequence: same content, new ids, becomes active
    tabs.tabs.setCurrentIndex(0)
    rec.events.clear()
    tabs.btn_dup.click()
    dup = st.sequence()
    orig = st.project.sequence(first.id)                       # undo / redo rebuild the objects: look it up again
    assert rec.labels == ["複製序列"] and dup.id not in (first.id, second.id) and dup.name != orig.name
    assert len(seq_names(st)) == 5 and len(list(orig.all_clips())) == 1
    assert [round(c.src_out, 3) for _t, c in dup.all_clips()] == [round(c.src_out, 3) for _t, c in orig.all_clips()]
    assert {c.id for _t, c in dup.all_clips()}.isdisjoint({c.id for _t, c in orig.all_clips()})
    assert tabs.tabs.currentIndex() == 4
    # delete asks first and is undoable
    n = len(st.project.sequences)
    stubs.ask_answers = [False]
    tabs.btn_del.click()
    assert len(st.project.sequences) == n and f"「{dup.name}」" in stubs.last_ask()
    stubs.ask_answers = [True]
    rec.events.clear()
    tabs.btn_del.click()
    assert len(st.project.sequences) == n - 1 and rec.labels == ["刪除序列"] and st.seq_id != dup.id
    win.undo()
    assert len(st.project.sequences) == n and any(s.id == dup.id for s in st.project.sequences)
    assert [round(c.src_out, 3) for _t, c in st.project.sequence(dup.id).all_clips()]    # its clips came back too


def test_sequence_tabs_rename_and_last_sequence_is_protected(qapp, make_window, stubs, real_proj):
    win = make_window(real_proj)
    st, tabs = win.state, win.sequence_tabs
    rec = Recorder(st)
    stubs.text = "新名字"
    tabs.btn_rename.click()
    assert st.sequence().name == "新名字" and tabs.titles() == ["新名字  ·  16:9"] and rec.labels == ["重新命名序列"]
    for ignored in (None, "", "   ", "新名字"):                                 # cancelled / blank / unchanged
        stubs.text = ignored
        tabs.btn_rename.click()
    assert rec.labels == ["重新命名序列"] and st.sequence().name == "新名字"
    tabs.btn_add.menu().actions()[0].trigger()
    assert len(st.project.sequences) == 2 and st.seq_id == st.project.sequences[1].id
    stubs.text = "雙擊改名"
    tabs.tabs.tabBarDoubleClicked.emit(0)                                       # double click renames THAT tab
    assert seq_names(st) == ["雙擊改名", "序列 2"] and st.seq_id == st.project.sequences[1].id
    # only one sequence left: it cannot be deleted
    stubs.ask_default = True
    tabs.delete(st.project.sequences[0].id)
    assert len(st.project.sequences) == 1 and not tabs.btn_del.isEnabled()
    assert tabs.delete() is False and win.statusBar().currentMessage() == "至少要保留一個序列"
    assert len(st.project.sequences) == 1


def test_sequence_tab_context_menu_and_live_titles(qapp, make_window, stubs, real_proj, monkeypatch):
    win = make_window(real_proj)
    st, tabs = win.state, win.sequence_tabs
    tabs.btn_add.menu().actions()[0].trigger()
    picked: list[list[str]] = []
    choice = {"index": 0}

    def fake_exec(menu, *_a, **_k):
        picked.append([a.text() for a in menu.actions()])
        menu.actions()[choice["index"]].trigger()

    monkeypatch.setattr(mw.SequenceTabs, "_popup_menu", lambda self, menu, pos: fake_exec(menu))
    tabs.tabs.customContextMenuRequested.emit(tabs.tabs.tabRect(0).center())
    assert picked[-1] == ["重新命名…", "複製", "刪除"]
    choice["index"] = 1                                                          # 「複製」 of the first tab
    tabs.tabs.customContextMenuRequested.emit(tabs.tabs.tabRect(0).center())
    assert len(st.project.sequences) == 3
    choice["index"] = 2                                                          # 「刪除」 of the second tab
    stubs.ask_answers = [True]
    tabs.tabs.customContextMenuRequested.emit(tabs.tabs.tabRect(1).center())
    assert len(st.project.sequences) == 2
    stubs.ask_answers = []
    # a right click on empty space opens nothing
    before = len(picked)
    tabs.tabs.customContextMenuRequested.emit(QPoint(5000, 5))
    assert len(picked) == before
    # titles and tooltips follow edits made elsewhere (the inspector changes the format)
    st.set_sequence(st.project.sequences[0].id)
    st.op("格式", "set_sequence_format", 1080, 1920, 25.0)
    assert tabs.titles()[0].endswith("9:16")
    assert "1080×1920" in tabs.tabs.tabToolTip(0) and "25 fps" in tabs.tabs.tabToolTip(0)


# ============================================================================ the rest of the menus
def test_fullscreen_and_reset_layout(qapp, make_window):
    win = make_window()
    win.act_fullscreen.trigger()
    assert win.isFullScreen() and win.act_fullscreen.isChecked()
    win.act_fullscreen.trigger()
    assert not win.isFullScreen() and not win.act_fullscreen.isChecked()
    win.dock_inspector.hide()
    win.dock_bin.hide()
    win.splitter.setSizes([100, 560])
    settle(qapp)
    win.act_reset_layout.trigger()
    settle(qapp)
    assert win.dock_inspector.isVisible() and win.dock_bin.isVisible()
    top, bottom = win.splitter.sizes()
    # the default proportions again (monitor : timeline is about half and half, so that the audio lanes show at 1100x700 too),
    # not the squashed [100, 560] it was left in
    assert abs(top / bottom - mw.SPLIT_SIZES[0] / mw.SPLIT_SIZES[1]) < 0.15, (top, bottom)


def test_about_and_environment_dialogs(qapp, make_window, stubs, monkeypatch):
    win = make_window()
    win.act_about.trigger()
    title, text, warning = stubs.informs[-1]
    assert title == "關於 AutoCut Studio" and warning is False
    assert mw.__version__ in text and "MIT" in text and "video-autopilot-kit" in text and "不會上傳你的素材" in text
    shown = []
    monkeypatch.setattr(mw.MainWindow, "_run_dialog", staticmethod(shown.append))
    win.act_env.trigger()
    (dlg,) = shown
    assert isinstance(dlg, mw.EnvironmentDialog)
    wait_for_idle(qapp, [dlg.runner], timeout=60)
    items = {name: status for name, status, _d in dlg.rows}
    assert items["ffmpeg"] == "ok" and items["PySide6（桌面介面）"] == "ok" and items["資料夾"] == "ok"
    assert dlg.table.rowCount() == len(dlg.rows) >= 8
    assert "必要的項目都正常" in dlg.note.text() or "有項目需要處理" in dlg.note.text()
    dlg.copy_text()
    assert "ffmpeg" in QApplication.clipboard().text()
    dlg.btn_hw.click()                                                           # real probe of the hardware encoders
    assert not dlg.btn_hw.isEnabled()
    end = time.monotonic() + 60
    while not dlg.btn_hw.isEnabled() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert dlg.btn_hw.isEnabled() and dlg.note.text().startswith("硬體編碼器偵測結果：")
    shot(dlg, "environment")
    dlg.deleteLater()


def table_row(dlg, name: str) -> tuple[str, str, str]:
    for r in range(dlg.table.rowCount()):
        if dlg.table.item(r, 1).text() == name:
            return tuple(dlg.table.item(r, c).text() for c in range(3))
    raise AssertionError(f"no row {name!r}")


def test_a_hardware_encoder_nobody_tried_yet_is_not_shown_as_normal(qapp, monkeypatch):
    from autocut.media import ffmpeg as ff
    ff.detect_hw_encoder.cache_clear()
    status = {n: (s, d) for n, s, d in mw.check_environment()}["硬體編碼器"]
    assert status[0] == "未執行" and "尚未實測" in status[1]
    dlg = mw.EnvironmentDialog()
    wait_for_idle(qapp, [dlg.runner], timeout=60)
    icon, _name, detail = table_row(dlg, "硬體編碼器")
    assert "正常" not in icon and "尚未實測" in icon + detail
    assert "必要的項目都正常" in dlg.note.text() or "有項目需要處理" in dlg.note.text()
    ff.detect_hw_encoder()                                                          # once it was really tried, it is a result
    assert {n: s for n, s, _d in mw.check_environment()}["硬體編碼器"] == "ok"
    dlg.deleteLater()


def test_a_hardware_probe_that_breaks_is_a_warning_and_not_a_result(qapp, monkeypatch):
    from autocut.media import ffmpeg as ff
    dlg = mw.EnvironmentDialog()
    wait_for_idle(qapp, [dlg.runner], timeout=60)

    def boom():
        raise RuntimeError("nvenc 掛了")

    monkeypatch.setattr(ff, "detect_hw_encoder", boom)
    dlg.btn_hw.click()
    end = time.monotonic() + 30
    while not dlg.btn_hw.isEnabled() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert dlg.btn_hw.isEnabled()
    icon, _name, detail = table_row(dlg, "硬體編碼器")
    assert icon.startswith("▲") and "nvenc 掛了" in detail and "偵測沒有成功" in detail
    assert {n: s for n, s, _d in dlg.rows}["硬體編碼器"] == "警告"
    assert dlg.note.text().startswith("硬體編碼器沒有偵測成功") and "nvenc 掛了" in dlg.note.text()
    assert "偵測結果" not in dlg.note.text()
    dlg.deleteLater()


def test_import_dialogs_import_and_remember_the_folder(qapp, make_window, stubs, media_dir, tmp_path):
    win = make_window()
    stubs.open_files = [str(media_dir / "port.mp4"), str(media_dir / "music.wav")]
    win.act_import.trigger()
    finish_imports(qapp, win)
    assert [a.name for a in win.state.project.assets.values()] == ["port.mp4", "music.wav"]
    assert mw.app_settings().value("dialog/import_dir") == str(media_dir)
    folder = tmp_path / "資料夾"
    folder.mkdir()
    (folder / "x.mp4").write_bytes((media_dir / "silent.mp4").read_bytes())
    stubs.folder = str(folder)
    win.act_import_folder.trigger()
    finish_imports(qapp, win)
    assert [a.name for a in win.state.project.assets.values()][-1] == "x.mp4"
    assert mw.app_settings().value("dialog/import_dir") == str(folder)
    stubs.open_files = []                                                         # cancelled dialog: nothing happens
    win.act_import.trigger()
    assert not win.importer.runners


def test_auto_edit_menu_picks_its_source_videos(qapp, make_window, stubs, real_proj, fake_dialogs):
    win = make_window()
    win.act_shorts.trigger()                                                      # nothing imported yet
    assert stubs.informs[-1][0] == "自動剪成短影音" and "還沒有影片" in stubs.informs[-1][1] and not fake_dialogs.log
    win.act_assemble.trigger()
    assert "至少需要 2 支影片" in stubs.informs[-1][1] and not fake_dialogs.log
    win2 = make_window(real_proj)
    videos = {a.name: a.id for a in real_proj.assets.values() if a.kind == "video"}
    assert set(videos) == {"land.mp4", "port.mp4", "silent.mp4"}
    stubs.item = "port.mp4（0:05）"                                                 # several videos, none selected: ask
    win2.act_shorts.trigger()
    assert fake_dialogs.log[-1][1] == videos["port.mp4"] and fake_dialogs.log[-1][0] is win2.state
    win2.media_bin.list.clearSelection()
    win2.media_bin.item_for(videos["silent.mp4"]).setSelected(True)               # exactly one selected video: use it
    win2.act_shorts.trigger()
    assert fake_dialogs.log[-1][1] == videos["silent.mp4"]
    stubs.item = None                                                             # picker cancelled: no dialog
    win2.media_bin.list.clearSelection()
    count = len(fake_dialogs.log)
    win2.act_shorts.trigger()
    assert len(fake_dialogs.log) == count
    win2.act_assemble.trigger()                                                   # no selection: every video
    assert fake_dialogs.log[-1][1] == list(videos.values())
    win2.media_bin.list.clearSelection()
    for name in ("land.mp4", "silent.mp4"):
        win2.media_bin.item_for(videos[name]).setSelected(True)
    win2.act_assemble.trigger()                                                   # a selection of videos: just those
    assert set(fake_dialogs.log[-1][1]) == {videos["land.mp4"], videos["silent.mp4"]}


def test_export_dialog_opens_for_the_active_sequence(qapp, make_window, monkeypatch):
    shown = []
    monkeypatch.setattr(mw, "ExportDialog", lambda state, parent: type("D", (), {
        "exec": lambda self: shown.append((state, parent)), "deleteLater": lambda self: None})())
    win = make_window()
    win.act_export.trigger()
    assert shown == [(win.state, win)]


def test_render_preview_and_missing_timeline_commands_are_explained(qapp, make_window, monkeypatch):
    win = make_window()
    calls = []
    monkeypatch.setattr(win.monitor, "render_preview", lambda: calls.append("monitor") or True)
    win.act_preview.trigger()
    assert calls == ["monitor"]
    win.timeline_call("no_such_command_yet")
    assert win.statusBar().currentMessage() == "這個版本的時間軸還不支援這個指令"


def test_missing_files_are_rechecked_when_the_window_is_activated_again(qapp, make_window, monkeypatch):
    win = make_window()
    calls = []
    monkeypatch.setattr(win.media_bin, "recheck_missing", lambda: calls.append(1))
    win._last_missing_check = 0.0
    QApplication.sendEvent(win, QEvent(QEvent.ActivationChange))
    assert len(calls) == 1
    QApplication.sendEvent(win, QEvent(QEvent.ActivationChange))                  # not again within a few seconds
    assert len(calls) == 1
    win._last_missing_check = 0.0
    QApplication.sendEvent(win, QEvent(QEvent.ActivationChange))
    assert len(calls) == 2


# ============================================================================ app entry (gui/app.py)
def test_first_run_window_size_fits_the_screen():
    from PySide6.QtCore import QRect
    assert mw.default_window_size(QRect(0, 0, 2560, 1400)).toTuple() == (1500, 920)
    assert mw.default_window_size(QRect(0, 0, 1366, 728)).toTuple() == (1326, 668)
    assert mw.default_window_size(QRect(0, 0, 1280, 680)).toTuple() == (1240, 640)
    assert mw.default_window_size(QRect(0, 0, 800, 560)).toTuple() == mw.MIN_WINDOW          # never below the minimum


def test_parse_paths_strips_quotes_options_and_duplicates(tmp_path):
    a = tmp_path / "a b.mp4"
    got = app_mod.parse_paths([f'"{a}"', "-psn_0_123", f"  {a}  ", "", "rel.mp4", str(tmp_path / "x.acproj")])
    assert got == [str(a), os.path.abspath("rel.mp4"), str(tmp_path / "x.acproj")]
    assert app_mod.parse_paths([]) == []


def test_app_icon_is_drawn_at_every_size(qapp):
    icon = app_mod.build_app_icon()
    assert not icon.isNull()
    sizes = {s.width() for s in icon.availableSizes()}
    assert set(app_mod.ICON_SIZES) <= sizes
    img = icon.pixmap(64, 64).toImage()
    assert img.pixelColor(32, 32).alpha() == 255 and img.pixelColor(0, 0).alpha() == 0     # rounded tile, transparent corner
    assert app_mod.set_windows_app_id() is False                                    # only does something on Windows


def test_excepthook_logs_and_warns_at_most_once_in_a_while(qapp, stubs, monkeypatch):
    import sys
    from autocut.gui.workers import log_path
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(sys, "__stderr__", None)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    app_mod.install_excepthook()
    sys.excepthook(ValueError, ValueError("測試用的錯誤"), None)
    qapp.processEvents()
    assert "測試用的錯誤" in log_path().read_text(encoding="utf-8") and "--- uncaught" in log_path().read_text(encoding="utf-8")
    assert len(warned) == 1 and warned[0][1] == "發生未預期的錯誤" and "ValueError：測試用的錯誤" in warned[0][2]
    sys.excepthook(RuntimeError, RuntimeError("second"), None)
    qapp.processEvents()
    assert len(warned) == 1                                                           # no dialog storm


def test_main_builds_the_application_and_window(qapp, stubs, monkeypatch, tmp_path):
    import sys
    seen = {}

    class FakeWindow:
        def __init__(self, paths=None) -> None:
            seen["paths"] = paths

        def show(self) -> None:
            seen["shown"] = True

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(mw, "MainWindow", FakeWindow)
    monkeypatch.setattr(QApplication, "exec", staticmethod(lambda: 7))
    assert app_mod.main(['"' + str(tmp_path / "x y.mp4") + '"', "--flag"]) == 7
    assert seen == {"paths": [str(tmp_path / "x y.mp4")], "shown": True}
    assert qapp.applicationName() == "AutoCutStudio" and qapp.applicationDisplayName() == "AutoCut Studio"
    assert not qapp.windowIcon().isNull()
