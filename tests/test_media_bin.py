"""Media bin (素材庫): the REAL ``MediaBin`` on real (tiny, ffmpeg-made) media, driven with synthetic key / mouse /
drag-and-drop events and judged by what the *project* and the widget contain afterwards.

What is driven: items / texts / footer, background thumbnails (placeholder first, at most ``MAX_THUMB_JOBS`` at once,
failures remembered), missing-file marking and re-checking, the search box, list <-> icon view, double-click / Enter,
the context menu (what is enabled for which selection), removing assets (ONE undo step, asks when clips use them),
「重新連結…」, the drag source (payload + pixmap) and the drop target for OS files (hover overlay, importer hand-off,
real import end to end).

Modal dialogs and menus would hang an offscreen run, so every one of them is patched (``modals`` records them).
Screenshots: ``screenshots/media_bin_<name>.png`` (git-ignored).
"""
from __future__ import annotations

import faulthandler
import json
import os
import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QItemSelectionModel, QMimeData, QPoint, QPointF, Qt, QUrl                      # noqa: E402
from PySide6.QtGui import (QColor, QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent, QImage,   # noqa: E402
                           QMouseEvent, QPixmap)
from PySide6.QtTest import QTest                                                              # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QListView, QMessageBox                # noqa: E402

from autocut.core import ops                                                                  # noqa: E402
from autocut.core.project import Asset, Project                                               # noqa: E402
from autocut.gui import media_bin as mb                                                       # noqa: E402
from autocut.gui import theme                                                                 # noqa: E402
from autocut.gui.importer import ImportController                                             # noqa: E402
from autocut.gui.media_bin import (ASSET_MIME, MAX_THUMB_JOBS, ROLE_BADGE, ROLE_ID, ROLE_INFO, ROLE_KIND,   # noqa: E402
                                   ROLE_MISSING, ROLE_SEARCH, MediaBin, decode_asset_payload, describe_asset,
                                   encode_asset_payload, local_paths)
from autocut.gui.state import AppState                                                        # noqa: E402
from autocut.media.probe import probe_asset                                                   # noqa: E402
from autocut.util import human_duration                                                       # noqa: E402

pytestmark = pytest.mark.gui

SHOTS = Path(__file__).resolve().parents[1] / "screenshots"
LEFT = Qt.LeftButton


# ================================================================================================ helpers
@pytest.fixture(autouse=True)
def _watchdog():
    """A hung offscreen test must die with a traceback instead of blocking the run forever."""
    faulthandler.dump_traceback_later(150, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def pump(app, cond, timeout: float = 40.0, what: str = "condition") -> None:
    """Process events until ``cond()`` (background tasks deliver their results through the event loop)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return
        time.sleep(0.004)
    raise TimeoutError(f"timed out waiting for {what}")


def shot(widget, name: str) -> None:
    try:
        SHOTS.mkdir(exist_ok=True)
        widget.grab().save(str(SHOTS / f"media_bin_{name}.png"))
    except OSError:                                                   # pragma: no cover - read-only checkout
        pass


class FakeImporter:
    """What the bin needs of the ImportController: it only hands paths / ids over."""

    def __init__(self) -> None:
        self.placed: list[list[str]] = []
        self.imported: list[list[str]] = []
        self.runner = object()

    def place_assets(self, ids, *a, **k):
        self.placed.append(list(ids))
        return [f"clip-{i}" for i in ids]

    def import_paths(self, paths, **k):
        self.imported.append([str(p) for p in paths])
        return self.runner


class Modals:
    """Records every question / message / file dialog; ``answer`` is what 「確定」 returns."""

    def __init__(self) -> None:
        self.asks: list[tuple[str, str, str]] = []
        self.informs: list[tuple[str, str, bool]] = []
        self.file_dialogs: list[tuple[str, str]] = []
        self.answer = False
        self.picked: list[str] = []                                   # what the file dialog returns, in order

    def titles(self) -> list[str]:
        return [a[0] for a in self.asks]


@pytest.fixture(autouse=True)
def modals(monkeypatch) -> Modals:
    m = Modals()

    def ask(parent, title, text, yes="確定", no="取消"):
        m.asks.append((title, text, yes))
        return m.answer

    def inform(parent, title, text, warning=False):
        m.informs.append((title, text, warning))

    def get_open(*args, **kwargs):
        m.file_dialogs.append((args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else ""))
        return (m.picked.pop(0) if m.picked else "", "")

    monkeypatch.setattr(mb, "ask", ask)
    monkeypatch.setattr(mb, "inform", inform)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(get_open))
    return m


@pytest.fixture(autouse=True)
def menus(monkeypatch) -> list:
    """Menus the bin would have shown (a real modal menu never returns under the offscreen platform)."""
    shown: list = []
    monkeypatch.setattr(MediaBin, "_popup_menu", lambda self, menu, pos: shown.append((menu, pos)))
    return shown


@pytest.fixture
def files(media_dir, tmp_path) -> dict[str, Path]:
    """Private copies of the synthetic media (tests move / delete them) + a picture and a text file."""
    d = tmp_path / "素材 folder"
    d.mkdir()
    out: dict[str, Path] = {}
    for key, name in (("land", "land.mp4"), ("port", "port.mp4"), ("silent", "silent.mp4"), ("music", "music.wav")):
        out[key] = d / name
        shutil.copy(media_dir / name, out[key])
    img = QImage(64, 36, QImage.Format_RGB32)
    img.fill(QColor(255, 0, 255))
    out["pic"] = d / "magenta.png"
    assert img.save(str(out["pic"]))
    out["text"] = d / "notes.txt"
    out["text"].write_text("not a video", encoding="utf-8")
    return out


def project_of(*paths: Path) -> Project:
    p = Project.new("bin")
    for path in paths:
        a = probe_asset(path)
        p.assets[a.id] = a
    return p


class Rig:
    """A shown bin on a project plus the synthesizers every test uses."""

    def __init__(self, app, project: Project, menus: list, importer=None, size=(380, 560)) -> None:
        self.app = app
        self.menus = menus
        self.state = AppState(project)
        self.importer = importer if importer is not None else FakeImporter()
        self.status: list[str] = []
        self.state.status.connect(self.status.append)
        self.bin = MediaBin(self.state, self.importer)
        self.bin.resize(*size)
        self.bin.show()
        app.processEvents()
        self.picked: list[str] = []

    # ---- project facts
    @property
    def assets(self) -> dict[str, Asset]:
        return self.state.project.assets

    def id_of(self, name: str) -> str:
        return next(a.id for a in self.assets.values() if a.name == name)

    def names(self) -> list[str]:
        return [self.assets[i].name for i in self.bin.visible_asset_ids()]

    # ---- waiting
    def idle(self, timeout: float = 40.0) -> None:
        pump(self.app, lambda: not self.bin.busy, timeout, "thumbnails and the missing-file scan")
        self.app.processEvents()

    # ---- gestures
    def row_center(self, asset_id: str) -> QPoint:
        return self.bin.list.visualItemRect(self.bin.item_for(asset_id)).center()

    def select(self, *names: str) -> list[str]:
        self.bin.list.clearSelection()
        ids = [self.id_of(n) for n in names]
        for i in ids:
            self.bin.item_for(i).setSelected(True)
        self.app.processEvents()
        return ids

    def menu_for(self, *names: str):
        """Right-click the first named item (after selecting all of them) -> the menu the bin built."""
        ids = self.select(*names)
        before = len(self.menus)
        self.bin.list.customContextMenuRequested.emit(self.row_center(ids[0]))
        return self.menus[before][0]


@pytest.fixture
def make_rig(qapp, menus):
    made: list[Rig] = []

    def make(project: Project, importer=None, size=(380, 560)) -> Rig:
        r = Rig(qapp, project, menus, importer, size)
        made.append(r)
        return r

    yield make
    for r in made:
        try:
            pump(qapp, lambda: not r.bin.busy, 40.0, "background work to end")
        except TimeoutError:
            pass
        r.bin.close()
        r.bin.deleteLater()
    qapp.processEvents()


@pytest.fixture
def rig(make_rig, files) -> Rig:
    """land (video+audio), port (blue portrait), silent (green, no audio), music (audio only), magenta picture."""
    r = make_rig(project_of(files["land"], files["port"], files["silent"], files["music"], files["pic"]))
    r.idle()
    return r


def action(menu, text: str):
    return next(a for a in menu.actions() if a.text() == text)


def texts(menu) -> list[str]:
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def reddish(img: QImage, rect) -> int:
    """Pixels in ``rect`` that are clearly red (the missing-file tint, slash and text)."""
    n = 0
    for y in range(max(0, rect.top()), min(img.height(), rect.bottom())):
        for x in range(max(0, rect.left()), min(img.width(), rect.right())):
            c = QColor(img.pixel(x, y))
            if c.red() > 150 and c.red() > c.green() + 60 and c.red() > c.blue() + 60:
                n += 1
    return n


def count_clips(state: AppState, asset_id: str) -> int:
    return sum(1 for s in state.project.sequences for _t, c in s.all_clips() if c.asset_id == asset_id)


def undo_depth(state: AppState) -> int:
    return len(state.history._undo)


# ---- drag and drop synthesizers (what the window system does: DragEnter -> DragMove -> Drop)
def files_mime(*paths) -> QMimeData:
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return m


def drag_enter(widget, mime: QMimeData, pos: QPoint = QPoint(30, 30)) -> bool:
    ev = QDragEnterEvent(pos, Qt.CopyAction, mime, LEFT, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev.isAccepted()


def drag_move(widget, mime: QMimeData, pos: QPoint = QPoint(30, 30)) -> bool:
    ev = QDragMoveEvent(pos, Qt.CopyAction, mime, LEFT, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev.isAccepted()


def drag_leave(widget) -> None:
    QApplication.sendEvent(widget, QDragLeaveEvent())


def drop(widget, mime: QMimeData, pos: QPoint = QPoint(30, 30)) -> bool:
    drag_enter(widget, mime, pos)
    drag_move(widget, mime, pos)
    ev = QDropEvent(QPointF(pos), Qt.CopyAction, mime, LEFT, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev.isAccepted()


# ================================================================================================ pure helpers
def test_describe_asset_lines_for_every_kind():
    video = Asset(id="v", path="/x/v.mp4", name="v.mp4", kind="video", duration=75.4, width=1920, height=1080,
                  fps=29.97, has_video=True, has_audio=True)
    assert describe_asset(video) == ("1:15", "1:15 · 1920×1080 · 29.97 fps")
    audio = Asset(id="a", path="/x/a.wav", name="a.wav", kind="audio", duration=125.0, has_video=False,
                  has_audio=True)
    assert describe_asset(audio) == ("2:05", "2:05 · 音訊")
    picture = Asset(id="i", path="/x/i.png", name="i.png", kind="image", width=800, height=600, has_video=False)
    assert describe_asset(picture) == ("", "圖片 · 800×600")
    assert describe_asset(Asset(id="i2", path="/x/i.png", kind="image", has_video=False)) == ("", "圖片")
    bare = Asset(id="b", path="/x/b.mp4", kind="video", duration=3.0, width=0, height=0, fps=0.0)
    assert describe_asset(bare) == ("0:03", "0:03")
    long = Asset(id="l", path="/x/l.mp4", kind="video", duration=3725.0, width=640, height=360, fps=25.0)
    assert describe_asset(long)[0] == "1:02:05"
    # a "video" file without a picture stream is shown as sound
    sound = Asset(id="s", path="/x/s.mp4", kind="video", duration=10.0, has_video=False, has_audio=True)
    assert describe_asset(sound) == ("0:10", "0:10 · 音訊")


def test_payload_round_trip_and_malformed_payloads():
    ids = ["a_1", "b 中文", "c"]
    mime = QMimeData()
    mime.setData(ASSET_MIME, encode_asset_payload(ids))
    assert decode_asset_payload(mime) == ids
    assert json.loads(bytes(mime.data(ASSET_MIME)).decode("utf-8")) == {"assets": ids}
    for bad in (b"", b"{", b"[]", b'{"nope": 1}', b'{"assets": 5}', b"\xff\xfe"):
        m = QMimeData()
        m.setData(ASSET_MIME, bad)
        assert decode_asset_payload(m) == [], bad
    assert decode_asset_payload(QMimeData()) == []


def test_local_paths_keeps_only_local_files_and_folders(tmp_path):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(tmp_path / "a.mp4")), QUrl("https://example.com/x.mp4"),
               QUrl.fromLocalFile(str(tmp_path))])
    assert local_paths(m) == [str(tmp_path / "a.mp4"), str(tmp_path)]
    assert local_paths(QMimeData()) == []
    text = QMimeData()
    text.setText("hello")
    assert local_paths(text) == []


def test_reveal_in_file_manager_picks_the_nearest_existing_folder(monkeypatch, tmp_path):
    opened: list[str] = []
    monkeypatch.setattr(mb.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toLocalFile()) or True))
    monkeypatch.setattr(mb.sys, "platform", "linux")
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    assert mb.reveal_in_file_manager(str(f)) and opened[-1] == str(tmp_path)           # existing file: its folder
    assert mb.reveal_in_file_manager(str(tmp_path / "gone.mp4")) and opened[-1] == str(tmp_path)
    assert mb.reveal_in_file_manager(str(tmp_path / "x" / "y" / "z.mp4")) and opened[-1] == str(tmp_path)
    assert mb.reveal_in_file_manager("") and opened[-1]                                  # never raises, opens somewhere


def test_reveal_in_file_manager_uses_explorer_and_finder(monkeypatch, tmp_path):
    opened: list[str] = []
    cmds: list[list[str]] = []
    monkeypatch.setattr(mb.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toLocalFile()) or True))
    monkeypatch.setattr(mb.subprocess, "Popen", lambda cmd, *a, **k: cmds.append(list(cmd)))
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    monkeypatch.setattr(mb.sys, "platform", "win32")
    assert mb.reveal_in_file_manager(str(f))
    assert cmds[-1] == ["explorer", f"/select,{os.path.normpath(str(f))}"] and opened == []
    monkeypatch.setattr(mb.sys, "platform", "darwin")
    assert mb.reveal_in_file_manager(str(f))
    assert cmds[-1] == ["open", "-R", os.path.normpath(str(f))]
    # a file that is gone: only its folder can be shown
    cmds.clear()
    assert mb.reveal_in_file_manager(str(tmp_path / "gone.mp4")) and cmds == [] and opened[-1] == str(tmp_path)

    def boom(*a, **k):
        raise OSError("no explorer")

    monkeypatch.setattr(mb.sys, "platform", "win32")
    monkeypatch.setattr(mb.subprocess, "Popen", boom)
    assert mb.reveal_in_file_manager(str(f)) and opened[-1] == str(tmp_path)           # falls back to the folder


# ================================================================================================ items and texts
def test_removing_everything_brings_the_drop_hint_back(rig):
    b = rig.bin
    assert not b.list.hint.isVisible()
    assert b.remove_assets(list(rig.assets)) is True
    assert b.list.count() == 0 and b.list.hint.isVisible() and b.list.hint.text().startswith("把影片")
    assert b.footer.text() == "0 個素材"
    rig.state.undo()
    assert b.list.count() == 5 and not b.list.hint.isVisible()


def test_empty_bin_shows_the_drop_hint(qapp, make_rig):
    r = make_rig(Project.new("empty"))
    b = r.bin
    assert b.list.count() == 0 and b.list.hint.isVisible()
    assert "拖到這裡" in b.list.hint.text()
    assert b.footer.text() == "0 個素材" and not b.busy
    assert b.selected_asset_ids() == [] and b.place_selected() == [] and r.importer.placed == []


def test_items_follow_the_project_with_name_info_and_tooltip(rig, files):
    b = rig.bin
    ids = list(rig.assets)
    assert [b.list.item(r).data(ROLE_ID) for r in range(b.list.count())] == ids
    land = b.item_for(rig.id_of("land.mp4"))
    assert land.data(Qt.DisplayRole) == "land.mp4" and land.data(ROLE_KIND) == "video"
    info = land.data(ROLE_INFO)
    assert info.startswith("0:08") and "1280×720" in info and "30 fps" in info
    assert land.data(ROLE_BADGE) == "0:08"
    tip = land.toolTip()
    assert str(files["land"].resolve()) in tip and "影片" in tip and ("MB" in tip or "KB" in tip)
    music = b.item_for(rig.id_of("music.wav"))
    assert music.data(ROLE_KIND) == "audio" and music.data(ROLE_INFO) == "0:20 · 音訊"
    pic = b.item_for(rig.id_of("magenta.png"))
    assert pic.data(ROLE_KIND) == "image" and pic.data(ROLE_INFO) == "圖片 · 64×36" and pic.data(ROLE_BADGE) == ""
    assert "land.mp4" in land.data(ROLE_SEARCH) and "素材 folder" in land.data(ROLE_SEARCH)
    total = sum(a.duration for a in rig.assets.values() if a.kind != "image")
    assert b.footer.text() == f"5 個素材 · 總長 {human_duration(total)}"
    assert not b.list.hint.isVisible()


def test_bin_follows_added_removed_and_replaced_projects(rig, files, tmp_path):
    b, st = rig.bin, rig.state
    clone = tmp_path / "copy of land.mp4"
    shutil.copy(files["land"], clone)
    new = probe_asset(clone)
    st.do("加入", lambda p: ops.add_asset(p, new))
    assert b.list.count() == 6 and b.list.row(b.item_for(new.id)) == 5
    rig.idle()
    st.undo()
    assert b.list.count() == 5 and b.item_for(new.id) is None
    st.redo()
    assert b.list.count() == 6
    kept = rig.select("port.mp4")
    assert b.selected_asset_ids() == kept
    st.undo()
    assert b.list.count() == 5
    st.replace_project(project_of(files["silent"]))                      # open another project
    assert [rig.assets[i].name for i in b.visible_asset_ids()] == ["silent.mp4"]
    assert b.selected_asset_ids() == [] and b.footer.text().startswith("1 個素材")
    rig.idle()


def test_assets_added_in_a_new_order_keep_the_list_in_project_order(rig):
    b = rig.bin
    ids = list(rig.assets)
    rig.state.do("重排", lambda p: p.assets.update({i: p.assets.pop(i) for i in reversed(ids)}))
    assert [b.list.item(r).data(ROLE_ID) for r in range(b.list.count())] == list(reversed(ids))


# ================================================================================================ thumbnails
def test_thumbnails_arrive_in_the_background_for_video_and_pictures_only(qapp, make_rig, files, monkeypatch):
    calls: list[str] = []
    real = mb.thumbnail
    monkeypatch.setattr(mb, "thumbnail", lambda asset, t=0.0, height=96: calls.append(asset.name) or real(asset, t, height))
    r = make_rig(project_of(files["land"], files["port"], files["silent"], files["music"], files["pic"]))
    b = r.bin
    assert b.busy and all(b.thumbnail_for(i) is None for i in r.assets)       # placeholders first
    peak = 0
    while b.busy:
        qapp.processEvents()
        peak = max(peak, len(b._thumb_active))
        time.sleep(0.002)
    assert 1 <= peak <= MAX_THUMB_JOBS
    for name in ("land.mp4", "port.mp4", "silent.mp4", "magenta.png"):
        pix = b.thumbnail_for(r.id_of(name))
        assert isinstance(pix, QPixmap) and 0 < pix.height() <= mb.THUMB_HEIGHT, name
    assert b.thumbnail_for(r.id_of("music.wav")) is None                      # sound keeps the drawn placeholder
    assert sorted(calls) == ["land.mp4", "magenta.png", "port.mp4", "silent.mp4"]   # and ffmpeg never ran for it
    # what the thumbnails show
    port = b.thumbnail_for(r.id_of("port.mp4")).toImage()
    c = QColor(port.pixel(port.width() // 2, port.height() // 2))
    assert port.height() > port.width() and c.blue() > 150 and c.red() < 90, c.name()      # the blue portrait clip
    silent = b.thumbnail_for(r.id_of("silent.mp4")).toImage()
    c = QColor(silent.pixel(silent.width() // 2, silent.height() // 2))
    assert silent.width() > silent.height() and c.green() > 90 and c.green() > c.red() + 40, c.name()   # the green one
    pic = b.thumbnail_for(r.id_of("magenta.png")).toImage()
    c = QColor(pic.pixel(pic.width() // 2, pic.height() // 2))
    assert c.red() > 200 and c.blue() > 200 and c.green() < 80, c.name()


def test_many_assets_never_run_more_than_the_limit_of_thumbnail_jobs(qapp, make_rig, files):
    paths = []
    for i in range(7):
        p = files["land"].with_name(f"clip {i}.mp4")
        shutil.copy(files["land"], p)
        paths.append(p)
    r = make_rig(project_of(*paths))
    b = r.bin
    peak = 0
    end = time.monotonic() + 60
    while b.busy and time.monotonic() < end:
        qapp.processEvents()
        peak = max(peak, len(b._thumb_active), len(b.runners))
        assert len(b._thumb_active) <= MAX_THUMB_JOBS
        time.sleep(0.002)
    assert not b.busy and peak >= 1
    assert all(b.thumbnail_for(i) is not None for i in r.assets) and b.runners == []


def test_a_file_ffmpeg_cannot_read_stays_a_placeholder_and_is_tried_only_once(qapp, make_rig, files, monkeypatch):
    junk = files["text"].with_name("broken.mp4")
    junk.write_bytes(b"this is not a movie")
    p = Project.new("junk")
    a = Asset(id="junk", path=str(junk), name="broken.mp4", kind="video", duration=5.0, width=640, height=360,
              fps=25.0, has_video=True, fingerprint="junk-1", size_bytes=19)
    p.assets[a.id] = a
    calls: list[str] = []
    real = mb.thumbnail
    monkeypatch.setattr(mb, "thumbnail", lambda asset, t=0.0, height=96: calls.append(asset.id) or real(asset, t, height))
    r = make_rig(p)
    r.idle()
    assert calls == ["junk"] and r.bin.thumbnail_for("junk") is None and "junk" in r.bin._thumb_failed
    assert not r.bin.is_missing("junk")                                      # it exists, it is just unreadable
    r.bin.refresh()
    r.bin.recheck_missing()
    r.idle()
    assert calls == ["junk"]                                                 # not retried on every refresh
    assert r.bin.item_for("junk") is not None


# ================================================================================================ missing files
def test_missing_files_are_marked_and_unmarked_when_they_come_back(qapp, make_rig, files):
    r = make_rig(project_of(files["port"], files["silent"]))
    b = r.bin
    r.idle()
    port, silent = r.id_of("port.mp4"), r.id_of("silent.mp4")
    assert not b.is_missing(port) and "找不到" not in b.footer.text()
    saved = files["port"].read_bytes()
    files["port"].unlink()
    b.recheck_missing()
    pump(qapp, lambda: b.is_missing(port), 10, "the missing mark")
    assert not b.is_missing(silent) and "1 個找不到檔案" in b.footer.text()
    r.idle()
    # the row is painted red (tint, slash, text); the healthy row is not
    img = b.list.viewport().grab().toImage()
    bad_rect = b.list.visualItemRect(b.item_for(port))
    good_rect = b.list.visualItemRect(b.item_for(silent))
    assert reddish(img, bad_rect) > 150 and reddish(img, good_rect) == 0
    shot(b, "missing_list")
    files["port"].write_bytes(saved)                                          # the file is back
    b.recheck_missing()
    pump(qapp, lambda: not b.is_missing(port), 10, "the mark to go away")
    assert "找不到" not in b.footer.text()
    assert reddish(b.list.viewport().grab().toImage(), b.list.visualItemRect(b.item_for(port))) == 0


def test_a_project_with_vanished_files_marks_them_right_after_opening(qapp, make_rig, files):
    p = project_of(files["land"], files["silent"])
    files["land"].unlink()
    r = make_rig(p)
    r.idle()
    assert r.bin.is_missing(r.id_of("land.mp4")) and not r.bin.is_missing(r.id_of("silent.mp4"))
    assert r.bin.thumbnail_for(r.id_of("land.mp4")) is None                  # no ffmpeg error for a vanished file
    assert r.bin.item_for(r.id_of("land.mp4")).data(ROLE_MISSING) is True
    # repeated re-checks in a row collapse into at most one follow-up scan
    for _ in range(5):
        r.bin.recheck_missing()
    r.idle()
    assert r.bin._scan is None and not r.bin._scan_again


# ================================================================================================ search / views
def test_search_filters_by_name_and_path_and_drops_hidden_selection(rig):
    b = rig.bin
    assert len(b.visible_asset_ids()) == 5
    b.search.setText("PORT")                                                  # case-insensitive
    assert rig.names() == ["port.mp4"] and not b.list.hint.isVisible()
    b.search.setText("素材 folder")                                           # the folder name is searched too
    assert len(rig.names()) == 5
    rig.select("land.mp4", "silent.mp4", "music.wav")
    b.search.setText("mp4")
    assert rig.names() == ["land.mp4", "port.mp4", "silent.mp4"]
    assert b.selected_asset_ids() == [rig.id_of("land.mp4"), rig.id_of("silent.mp4")]   # the hidden one is deselected
    b.search.setText("zzz-no-such")
    assert rig.names() == [] and b.list.hint.isVisible() and b.list.hint.text() == "沒有符合「zzz-no-such」的素材"
    assert b.selected_asset_ids() == [] and b.place_selected() == []
    assert b.footer.text().startswith("5 個素材")                             # the footer still counts everything
    shot(b, "search_none")
    b.search.setText("  Music ")                                              # surrounding spaces are ignored
    assert rig.names() == ["music.wav"]
    b.search.clear()
    assert len(rig.names()) == 5 and not b.list.hint.isVisible()


def test_view_toggle_switches_between_list_and_icon_view(rig):
    b = rig.bin
    assert not b.icon_mode() and b.btn_view.text() == "縮圖" and not b.btn_view.isChecked()
    row = b.delegate.sizeHint(None, None)
    assert row == b.delegate.ROW and b.list.viewMode() == QListView.ListMode
    shot(b, "list")
    b.btn_view.click()
    assert b.icon_mode() and b.btn_view.text() == "清單" and b.list.viewMode() == QListView.IconMode
    assert b.delegate.sizeHint(None, None) == b.delegate.CELL and b.list.isWrapping()
    rig.app.processEvents()
    shot(b, "icons")
    # in icon view the cells flow left to right: two on the first line at this width
    first, second = (b.list.visualItemRect(b.list.item(r)) for r in (0, 1))
    assert second.left() > first.left() and abs(second.top() - first.top()) < 4
    b.set_icon_mode(False)                                                   # programmatic switch syncs the button
    assert not b.btn_view.isChecked() and b.btn_view.text() == "縮圖" and b.list.viewMode() == QListView.ListMode
    b.set_icon_mode(True)
    assert b.btn_view.isChecked()


# ================================================================================================ selection / activation
def test_selection_signal_and_selected_ids_are_in_list_order(rig):
    b = rig.bin
    seen: list[int] = []
    b.assetSelectionChanged.connect(lambda: seen.append(len(b.selected_asset_ids())))
    ids = rig.select("music.wav", "land.mp4")                                # selected in this order ...
    assert b.selected_asset_ids() == [rig.id_of("land.mp4"), rig.id_of("music.wav")]   # ... reported in list order
    assert seen and seen[-1] == 2
    b.list.clearSelection()
    assert seen[-1] == 0 and b.selected_asset_ids() == [] and ids


def test_double_click_and_enter_put_the_selection_on_the_timeline(rig):
    b, imp = rig.bin, rig.importer
    land = rig.id_of("land.mp4")
    QTest.mouseClick(b.list.viewport(), LEFT, Qt.NoModifier, rig.row_center(land))
    assert b.selected_asset_ids() == [land] and imp.placed == []
    QTest.mouseDClick(b.list.viewport(), LEFT, Qt.NoModifier, rig.row_center(land))
    assert imp.placed == [[land]]
    ids = rig.select("port.mp4", "music.wav")
    b.list.setCurrentItem(b.item_for(ids[0]), QItemSelectionModel.NoUpdate)      # current row without touching the selection
    b.list.setFocus()
    QTest.keyClick(b.list, Qt.Key_Return)
    assert imp.placed[-1] == ids and len(imp.placed) == 2
    assert b.place_selected() == [f"clip-{i}" for i in ids]


# ================================================================================================ context menu
def test_menu_without_selection_only_offers_the_import(rig):
    b = rig.bin
    seen: list[str] = []
    b.importRequested.connect(lambda: seen.append("import"))
    rig.select("land.mp4")
    pos = QPoint(b.list.viewport().width() - 4, b.list.viewport().height() - 4)     # empty space below the rows
    assert b.list.itemAt(pos) is None
    before = len(rig.menus)
    b.list.customContextMenuRequested.emit(pos)
    menu = rig.menus[before][0]
    assert b.selected_asset_ids() == [] and texts(menu) == ["匯入素材…"]
    action(menu, "匯入素材…").trigger()
    assert seen == ["import"]


def test_menu_for_one_video_offers_shorts_but_not_assemble(rig):
    b = rig.bin
    shorts: list[str] = []
    assembles: list[list[str]] = []
    b.shortsRequested.connect(shorts.append)
    b.assembleRequested.connect(assembles.append)
    menu = rig.menu_for("land.mp4")
    assert texts(menu) == ["加入時間軸", "自動剪成短影音…", "長片精華…", "自動合併成長片…", "在檔案總管顯示", "重新連結…", "移除素材"]
    assert action(menu, "長片精華…").isEnabled()
    assert action(menu, "自動剪成短影音…").isEnabled() and not action(menu, "自動合併成長片…").isEnabled()
    assert action(menu, "在檔案總管顯示").isEnabled() and action(menu, "重新連結…").isEnabled()
    action(menu, "自動剪成短影音…").trigger()
    assert shorts == [rig.id_of("land.mp4")] and assembles == []
    action(menu, "加入時間軸").trigger()
    assert rig.importer.placed[-1] == [rig.id_of("land.mp4")]


def test_menu_for_several_assets_offers_assemble_with_the_videos_only(rig):
    b = rig.bin
    assembles: list[list[str]] = []
    shorts: list[str] = []
    b.assembleRequested.connect(assembles.append)
    b.shortsRequested.connect(shorts.append)
    menu = rig.menu_for("land.mp4", "port.mp4", "music.wav", "magenta.png")
    assert action(menu, "自動合併成長片…").isEnabled() and not action(menu, "自動剪成短影音…").isEnabled()
    assert not action(menu, "在檔案總管顯示").isEnabled() and not action(menu, "重新連結…").isEnabled()
    action(menu, "自動合併成長片…").trigger()
    assert assembles == [[rig.id_of("land.mp4"), rig.id_of("port.mp4")]]       # sound and pictures are left out
    action(menu, "加入時間軸").trigger()
    assert rig.importer.placed[-1] == [rig.id_of(n) for n in ("land.mp4", "port.mp4", "music.wav", "magenta.png")]
    # one video + one sound: shorts needs exactly one VIDEO among the selection, but the selection must be one asset
    menu = rig.menu_for("land.mp4", "music.wav")
    assert not action(menu, "自動剪成短影音…").isEnabled() and not action(menu, "自動合併成長片…").isEnabled()
    assert shorts == []


def test_menu_for_sound_or_pictures_never_offers_the_auto_edits(rig):
    for name in ("music.wav", "magenta.png"):
        menu = rig.menu_for(name)
        assert not action(menu, "自動剪成短影音…").isEnabled(), name
        assert not action(menu, "自動合併成長片…").isEnabled(), name
        assert action(menu, "加入時間軸").isEnabled() and action(menu, "移除素材").isEnabled()


def test_right_click_selects_the_item_under_the_pointer_unless_it_is_already_selected(rig):
    b = rig.bin
    land, port, silent = (rig.id_of(n) for n in ("land.mp4", "port.mp4", "silent.mp4"))
    rig.select("land.mp4", "port.mp4")
    before = len(rig.menus)
    b.list.customContextMenuRequested.emit(rig.row_center(silent))            # outside the selection: replaces it
    assert b.selected_asset_ids() == [silent]
    rig.select("land.mp4", "port.mp4")
    b.list.customContextMenuRequested.emit(rig.row_center(port))              # inside the selection: keeps all of it
    assert b.selected_asset_ids() == [land, port]
    menu = rig.menus[-1][0]
    assert len(rig.menus) == before + 2 and action(menu, "自動合併成長片…").isEnabled()


def test_reveal_and_relink_actions_use_the_asset(rig, monkeypatch, modals):
    shown: list[str] = []
    monkeypatch.setattr(mb, "reveal_in_file_manager", lambda path: shown.append(path) or True)
    menu = rig.menu_for("port.mp4")
    action(menu, "在檔案總管顯示").trigger()
    assert shown == [rig.assets[rig.id_of("port.mp4")].path]
    action(menu, "重新連結…").trigger()                                         # the file dialog is patched: cancelled
    assert len(modals.file_dialogs) == 1 and "port.mp4" in modals.file_dialogs[0][0] and modals.informs == []


def test_delete_key_removes_the_selected_assets_after_the_question(rig, modals):
    b, st = rig.bin, rig.state
    rig.select("music.wav", "magenta.png")
    b.list.setFocus()
    QTest.qWaitForWindowActive(b, 1000)
    assert b.list.hasFocus()
    QTest.keyClick(b.list, Qt.Key_Delete)
    assert [rig.assets[i].name for i in st.project.assets] == ["land.mp4", "port.mp4", "silent.mp4"]
    assert modals.asks == [] and b.list.count() == 3 and rig.status[-1] == "已移除 2 個素材"
    st.undo()
    assert b.list.count() == 5                                                # ONE undo step brings both back
    # nothing selected / no focus: the key does nothing
    b.list.clearSelection()
    QTest.keyClick(b.list, Qt.Key_Delete)
    assert b.list.count() == 5


# ================================================================================================ remove
def put_clip(rig: Rig, name: str, start: float = 0.0, s_in: float = 0.0, s_out: float | None = None) -> str:
    seq = rig.state.sequence()
    track = seq.video_tracks[0].id if rig.assets[rig.id_of(name)].has_video else seq.audio_tracks[0].id
    return rig.state.op("放", "place_clip", rig.id_of(name), track, start, s_in, s_out, "overwrite").id


def test_removing_unused_assets_is_one_undo_step_without_a_question(rig, modals):
    b, st = rig.bin, rig.state
    depth = undo_depth(st)
    ids = [rig.id_of("port.mp4"), rig.id_of("silent.mp4")]
    assert b.remove_assets(ids) is True
    assert modals.asks == [] and b.list.count() == 3 and all(i not in st.project.assets for i in ids)
    assert rig.status[-1] == "已移除 2 個素材" and undo_depth(st) == depth + 1
    assert "3 個素材" in b.footer.text()
    st.undo()
    assert list(st.project.assets).count(ids[0]) == 1 and b.list.count() == 5 and b.item_for(ids[1]) is not None
    assert b.remove_assets(["no-such-id"]) is False and b.remove_assets([]) is False
    assert undo_depth(st) == depth


def test_removing_assets_that_clips_use_asks_first(rig, modals):
    b, st = rig.bin, rig.state
    land = rig.id_of("land.mp4")
    put_clip(rig, "land.mp4", 0.0, 0.0, 3.0)
    put_clip(rig, "land.mp4", 4.0, 3.0, 6.0)
    assert count_clips(st, land) == 2
    depth, before = undo_depth(st), st.project.content_json()
    modals.answer = False                                                    # 「取消」
    assert b.remove_assets([land]) is False
    assert modals.titles() == ["移除素材"] and "2 個時間軸片段" in modals.asks[0][1] and modals.asks[0][2] == "移除"
    assert land in st.project.assets and count_clips(st, land) == 2 and undo_depth(st) == depth
    modals.answer = True
    assert b.remove_assets([land]) is True
    assert land not in st.project.assets and count_clips(st, land) == 0 and b.item_for(land) is None
    assert rig.status[-1] == "已移除 1 個素材與 2 個片段" and undo_depth(st) == depth + 1
    st.undo()                                                                # the asset AND its clips come back
    assert land in st.project.assets and count_clips(st, land) == 2 and st.project.content_json() == before
    assert b.item_for(land) is not None
    # several assets: the question counts them
    put_clip(rig, "port.mp4", 10.0, 0.0, 2.0)
    modals.asks.clear()
    assert b.remove_assets([land, rig.id_of("port.mp4")]) is True
    assert "這 2 個素材" in modals.asks[0][1] and "3 個時間軸片段" in modals.asks[0][1]


# ================================================================================================ relink
def moved_copy(src: Path, name: str) -> Path:
    dst = src.parent / "moved" / name
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(src, dst)
    return dst


def test_relink_points_the_same_asset_at_the_moved_file(qapp, make_rig, files, modals):
    r = make_rig(project_of(files["land"], files["silent"]))
    b, st = r.bin, r.state
    r.idle()
    land = r.id_of("land.mp4")
    put_clip(r, "land.mp4", 0.0, 1.0, 5.0)
    new = moved_copy(files["land"], "land moved.mp4")
    files["land"].unlink()
    b.recheck_missing()
    pump(qapp, lambda: b.is_missing(land), 10, "the missing mark")
    depth = undo_depth(st)
    b.relink(land, str(new))
    pump(qapp, lambda: st.project.assets[land].path == str(new.resolve()), 20, "the relink")
    r.idle()
    a = st.project.assets[land]
    assert a.id == land and a.name == "land moved.mp4" and count_clips(st, land) == 1       # clips keep pointing at it
    assert modals.asks == [] and modals.informs == [] and undo_depth(st) == depth + 1
    assert r.status[-1] == "已重新連結：land moved.mp4"
    assert not b.is_missing(land) and b.item_for(land).data(Qt.DisplayRole) == "land moved.mp4"
    assert b.thumbnail_for(land) is not None
    st.undo()
    assert st.project.assets[land].path == str(files["land"].resolve())
    pump(qapp, lambda: b.is_missing(land), 10, "the missing mark after undo")


def test_relink_through_the_file_dialog_starts_in_the_old_folder(qapp, make_rig, files, modals):
    r = make_rig(project_of(files["port"]))
    r.idle()
    port = r.id_of("port.mp4")
    new = moved_copy(files["port"], "elsewhere.mp4")
    modals.picked = [str(new)]
    r.bin.relink(port)
    pump(qapp, lambda: r.assets[port].path == str(new.resolve()), 20, "the relink")
    caption, folder = modals.file_dialogs[-1]
    assert "port.mp4" in caption and folder == str(files["port"].parent)
    modals.picked = [""]                                                      # cancelled dialog: nothing happens
    depth = undo_depth(r.state)
    r.bin.relink(port)
    qapp.processEvents()
    assert undo_depth(r.state) == depth and r.assets[port].path == str(new.resolve())
    r.bin.relink("no-such-asset")                                            # an unknown asset is ignored
    assert len(modals.file_dialogs) == 2
    r.idle()


def test_relink_to_a_different_kind_or_a_shorter_file_asks_first(qapp, make_rig, files, modals):
    r = make_rig(project_of(files["land"]))
    b, st = r.bin, r.state
    r.idle()
    land = r.id_of("land.mp4")
    put_clip(r, "land.mp4", 0.0, 2.0, 7.0)                                    # uses the source up to 7 s
    old_path = st.project.assets[land].path
    # (1) a sound file instead of a video: the question names both kinds
    modals.answer = False
    b.relink(land, str(files["music"]))
    pump(qapp, lambda: modals.asks, 20, "the question")
    assert modals.asks[-1][0] == "重新連結" and "新檔案是音訊，原本的素材是影片" in modals.asks[-1][1]
    assert modals.asks[-1][2] == "仍要連結"
    qapp.processEvents()
    assert st.project.assets[land].path == old_path                            # declined: unchanged
    # (2) a 4 s video although the timeline uses 7 s of the old one
    b.relink(land, str(files["silent"]))
    pump(qapp, lambda: len(modals.asks) == 2, 20, "the second question")
    assert "新檔案只有 4.0 秒" in modals.asks[-1][1] and "第 7.0 秒" in modals.asks[-1][1]
    qapp.processEvents()
    assert st.project.assets[land].path == old_path
    # (3) the person insists
    modals.answer = True
    b.relink(land, str(files["silent"]))
    pump(qapp, lambda: st.project.assets[land].path == str(files["silent"].resolve()), 20, "the relink")
    assert len(modals.asks) == 3
    r.idle()


def test_relink_between_two_pictures_has_nothing_to_warn_about(qapp, make_rig, files, modals):
    other = files["pic"].with_name("second.png")
    img = QImage(32, 32, QImage.Format_RGB32)
    img.fill(QColor(0, 255, 0))
    assert img.save(str(other))
    r = make_rig(project_of(files["pic"]))
    r.idle()
    pid = r.id_of("magenta.png")
    r.bin.relink(pid, str(other))
    pump(qapp, lambda: r.assets[pid].name == "second.png", 20, "the relink")
    assert modals.asks == [] and r.assets[pid].width == 32
    r.idle()


def test_relink_to_something_unreadable_reports_it_and_changes_nothing(qapp, make_rig, files, modals):
    r = make_rig(project_of(files["land"]))
    r.idle()
    land = r.id_of("land.mp4")
    before = r.state.project.content_json()
    depth = undo_depth(r.state)
    r.bin.relink(land, str(files["text"]))                                    # a text file
    pump(qapp, lambda: modals.informs, 20, "the error message")
    title, text, warning = modals.informs[-1]
    assert title == "無法重新連結" and warning is True and text
    r.bin.relink(land, str(files["land"].parent / "does-not-exist.mp4"))
    pump(qapp, lambda: len(modals.informs) == 2, 20, "the second error message")
    assert r.state.project.content_json() == before and undo_depth(r.state) == depth


def test_relink_to_a_different_picture_replaces_the_thumbnail(qapp, make_rig, files):
    r = make_rig(project_of(files["port"]))                                   # a blue portrait clip
    r.idle()
    pid = r.id_of("port.mp4")
    blue = r.bin.thumbnail_for(pid).toImage()
    r.bin.relink(pid, str(files["silent"]))                                   # the green landscape one
    pump(qapp, lambda: r.assets[pid].name == "silent.mp4", 20, "the relink")
    r.idle()
    green = r.bin.thumbnail_for(pid).toImage()
    c = QColor(green.pixel(green.width() // 2, green.height() // 2))
    assert green.width() > green.height() and c.green() > 90 and c.red() < 90 and blue.height() > blue.width()


# ================================================================================================ drag source
def test_the_list_is_a_drag_source_and_its_viewport_accepts_dropped_files(rig):
    """QListView.setMovement(Static) silently switches dragEnabled and the viewport's acceptDrops OFF: without the
    re-enabling in the constructor nothing could be dragged to the timeline and dropped files skipped the list."""
    lst = rig.bin.list
    assert lst.dragEnabled() and lst.acceptDrops() and lst.viewport().acceptDrops()
    assert rig.bin.acceptDrops()
    rig.bin.set_icon_mode(True)                                               # switching the view keeps all of it
    assert lst.dragEnabled() and lst.acceptDrops() and lst.viewport().acceptDrops()
    rig.bin.set_icon_mode(False)
    assert lst.dragEnabled() and lst.viewport().acceptDrops()


def test_the_drag_payload_lists_the_selection_in_list_order(rig):
    b = rig.bin
    ids = rig.select("music.wav", "land.mp4", "silent.mp4")
    items = [b.item_for(i) for i in ids]
    mime = b.list.mimeData(list(reversed(items)))                              # Qt may hand them over in any order
    assert decode_asset_payload(mime) == [rig.id_of(n) for n in ("land.mp4", "silent.mp4", "music.wav")]
    assert b.list.mimeTypes() == [ASSET_MIME] and b.list.supportedDragActions() == Qt.CopyAction


def test_start_drag_runs_the_os_drag_with_payload_and_pixmap(rig, monkeypatch):
    b = rig.bin
    seen: list = []
    monkeypatch.setattr(type(b.list), "_exec_drag", lambda self, drag: seen.append(drag))
    b.list.clearSelection()
    b.list.startDrag(Qt.CopyAction)
    assert seen == []                                                         # nothing selected: no drag
    rig.select("land.mp4", "port.mp4", "music.wav")
    b.list.startDrag(Qt.CopyAction)
    drag = seen[0]
    assert decode_asset_payload(drag.mimeData()) == [rig.id_of(n) for n in ("land.mp4", "port.mp4", "music.wav")]
    pix = drag.pixmap()
    assert (pix.width(), pix.height()) == (190, 44) and drag.hotSpot() == QPoint(14, 14)
    img = pix.toImage()
    assert any(QColor(img.pixel(x, 22)).alpha() > 0 for x in range(img.width()))      # something is drawn


def test_dragging_with_the_mouse_starts_a_drag_of_the_clicked_item(rig, monkeypatch):
    b = rig.bin
    seen: list = []
    monkeypatch.setattr(type(b.list), "_exec_drag", lambda self, drag: seen.append(drag))
    land = rig.id_of("land.mp4")
    pos = rig.row_center(land)
    vp = b.list.viewport()
    QApplication.sendEvent(vp, QMouseEvent(QEvent.MouseButtonPress, QPointF(pos), QPointF(vp.mapToGlobal(pos)), LEFT,
                                           LEFT, Qt.NoModifier))
    for dx in (4, 12, 30):
        p2 = pos + QPoint(dx, 0)
        QApplication.sendEvent(vp, QMouseEvent(QEvent.MouseMove, QPointF(p2), QPointF(vp.mapToGlobal(p2)),
                                               Qt.NoButton, LEFT, Qt.NoModifier))
    assert len(seen) == 1 and decode_asset_payload(seen[0].mimeData()) == [land]
    QApplication.sendEvent(vp, QMouseEvent(QEvent.MouseButtonRelease, QPointF(pos), QPointF(vp.mapToGlobal(pos)), LEFT,
                                           Qt.NoButton, Qt.NoModifier))


# ================================================================================================ drop target
def test_os_files_dropped_on_the_list_go_to_the_importer(rig):
    b, imp = rig.bin, rig.importer
    started: list = []
    b.importStarted.connect(started.append)
    paths = [Path("/tmp/a.mp4"), Path("/tmp/folder")]
    vp = b.list.viewport()
    assert drag_enter(vp, files_mime(*paths)) and b.list._hover
    assert drag_move(vp, files_mime(*paths))
    drag_leave(vp)
    assert not b.list._hover
    assert drop(vp, files_mime(*paths)) is True
    assert imp.imported == [[str(p) for p in paths]] and started == [imp.runner] and not b.list._hover


def test_hover_overlay_is_painted_while_files_hover_over_the_list(rig):
    b = rig.bin
    vp = b.list.viewport()
    plain = vp.grab().toImage()
    assert drag_enter(vp, files_mime("/tmp/a.mp4"))
    rig.app.processEvents()
    hover = vp.grab().toImage()
    diff = sum(1 for y in range(0, plain.height(), 3) for x in range(0, plain.width(), 3)
               if plain.pixel(x, y) != hover.pixel(x, y))
    assert diff > 40                                                          # the dashed frame and 「放開以匯入」
    shot(b, "drop_hover")
    drag_leave(vp)
    rig.app.processEvents()
    assert vp.grab().toImage() == plain


def test_drops_without_local_files_are_ignored(rig):
    b, imp = rig.bin, rig.importer
    vp = b.list.viewport()
    text = QMimeData()
    text.setText("just text")
    web = QMimeData()
    web.setUrls([QUrl("https://example.com/clip.mp4")])
    internal = QMimeData()
    internal.setData(ASSET_MIME, encode_asset_payload([rig.id_of("land.mp4")]))      # the bin's own drag
    for mime in (text, web, internal):
        assert drag_enter(vp, mime) is False and not b.list._hover
        assert drop(vp, mime) is False
    assert imp.imported == [] and len(rig.assets) == 5


def test_files_dropped_on_the_search_row_are_imported_too(rig):
    b, imp = rig.bin, rig.importer
    started: list = []
    b.importStarted.connect(started.append)
    assert drag_enter(b, files_mime("/tmp/x.wav")) and drag_move(b, files_mime("/tmp/x.wav"))
    assert drop(b, files_mime("/tmp/x.wav", "/tmp/y.wav")) is True
    assert imp.imported == [["/tmp/x.wav", "/tmp/y.wav"]] and len(started) == 1
    text = QMimeData()
    text.setText("no")
    assert drag_enter(b, text) is False and drop(b, text) is False and len(imp.imported) == 1


def test_a_real_drop_imports_end_to_end_and_the_bin_shows_it(qapp, make_rig, files):
    st_project = Project.new("drop")
    r = make_rig(st_project, importer=None)
    importer = ImportController(r.state)
    r.bin.importer = importer
    started: list = []
    r.bin.importStarted.connect(started.append)
    folder = files["silent"].with_name("a folder")
    folder.mkdir()
    shutil.copy(files["silent"], folder / "inside.mp4")
    ghost = files["land"].with_name("ghost.mp4")
    mime = files_mime(files["land"], files["port"], files["text"], ghost, folder)
    assert drop(r.bin.list.viewport(), mime) is True
    assert len(started) == 1
    pump(qapp, lambda: not importer.busy and r.bin.list.count() == 3, 40, "the import")
    assert sorted(a.name for a in r.assets.values()) == ["inside.mp4", "land.mp4", "port.mp4"]
    assert r.status[-1] == "已匯入 3 個素材，2 個檔案無法匯入"
    assert r.bin.footer.text().startswith("3 個素材")
    r.idle()
    assert all(r.bin.thumbnail_for(i) is not None for i in r.assets)           # thumbnails follow the import
    # the same files again add nothing
    depth = undo_depth(r.state)
    assert drop(r.bin.list.viewport(), files_mime(files["land"], files["port"])) is True
    pump(qapp, lambda: not importer.busy, 30, "the second import")
    assert r.bin.list.count() == 3 and undo_depth(r.state) == depth and len(r.assets) == 3
    r.state.undo()                                                            # the first import was ONE undo step
    assert r.bin.list.count() == 0 and r.assets == {}


# ================================================================================================ screenshots
def test_screenshots_of_the_main_states(qapp, make_rig, files):
    r = make_rig(project_of(files["land"], files["port"], files["silent"], files["music"], files["pic"]))
    r.idle()
    shot(r.bin, "overview_list")
    r.bin.set_icon_mode(True)
    qapp.processEvents()
    shot(r.bin, "overview_icons")
    r.bin.set_icon_mode(False)
    files["silent"].unlink()
    r.bin.recheck_missing()
    r.idle()
    r.select("land.mp4", "music.wav")
    qapp.processEvents()
    shot(r.bin, "overview_selected_missing")
    empty = make_rig(Project.new("empty"), size=(380, 300))
    shot(empty.bin, "empty")
    assert (SHOTS / "media_bin_overview_list.png").exists() or not SHOTS.exists()


# ----------------------------------------------------------------------------------- where the frozen app looks for ffmpeg
def test_a_frozen_app_finds_ffmpeg_next_to_its_exe_and_in_its_bundle(tmp_path, monkeypatch):
    """A PyInstaller build keeps its libraries in ``_internal`` (``sys._MEIPASS``) but the person drops ffmpeg next to
    ``AutoCutStudio.exe``: both places are searched, the exe folder first."""
    from autocut.media import ffmpeg as ff

    exe_dir = tmp_path / "AutoCutStudio"
    inner = exe_dir / "_internal"
    for d in (exe_dir / "ffmpeg" / "bin", inner / "ffmpeg"):
        d.mkdir(parents=True)
    monkeypatch.delenv("AUTOCUT_FFMPEG_DIR", raising=False)
    monkeypatch.setattr(ff.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ff.sys, "executable", str(exe_dir / "AutoCutStudio.exe"))
    monkeypatch.setattr(ff.sys, "_MEIPASS", str(inner), raising=False)
    dirs = ff._candidate_dirs()
    assert exe_dir / "ffmpeg" / "bin" in dirs and exe_dir in dirs and inner / "ffmpeg" in dirs and inner in dirs
    assert dirs.index(exe_dir / "ffmpeg" / "bin") < dirs.index(inner / "ffmpeg")
    exe = ff._exe("ffmpeg")
    (exe_dir / "ffmpeg" / "bin" / exe).write_bytes(b"x")
    assert Path(ff.find_binary("ffmpeg")) == (exe_dir / "ffmpeg" / "bin" / exe)
