"""ImportController: probing, dedupe, failure isolation, suggestions, timeline placement (real synthetic media)."""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from autocut.core.project import Asset, Project               # noqa: E402
from autocut.gui import importer as importer_mod              # noqa: E402
from autocut.gui.importer import (ImportController, sequence_format_for, standard_fps,  # noqa: E402
                                  suggest_kind)
from autocut.gui.state import AppState                        # noqa: E402
from autocut.gui.workers import wait_for_idle                 # noqa: E402

pytestmark = [pytest.mark.gui, pytest.mark.ffmpeg]


@pytest.fixture(scope="module")
def extra_media(media_dir, tmp_path_factory) -> dict:
    """A 46 s landscape clip with a Chinese name, a broken .mp4, an unsupported file and a subtitle file."""
    d = tmp_path_factory.mktemp("extra 素材")
    long_land = d / "長影片 測試.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=gray:size=192x108:rate=10:duration=46", "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(long_land)], check=True)
    broken = d / "壞掉.mp4"
    broken.write_bytes(b"not a video at all" * 50)
    notes = d / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    srt = d / "字幕.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\n你好\n\n2\n00:00:02,500 --> 00:00:04,000\n世界\n",
                   encoding="utf-8")
    return {"long": long_land, "broken": broken, "notes": notes, "srt": srt}


class Recorder:
    """Collects the controller's signals."""

    def __init__(self, ctl: ImportController) -> None:
        self.imported: list = []
        self.suggestions: list = []
        self.subtitles: list = []
        ctl.imported.connect(lambda ids, fails: self.imported.append((list(ids), list(fails))))
        ctl.suggestion.connect(lambda kind, ids: self.suggestions.append((kind, list(ids))))
        ctl.subtitleFilesDropped.connect(lambda paths: self.subtitles.append(list(paths)))


def _setup(project: Project | None = None):
    state = AppState(project or Project.new("匯入測試"))
    ctl = ImportController(state)
    steps: list[str] = []
    state.edited.connect(lambda kind, label: steps.append(f"{kind}:{label}"))
    return state, ctl, Recorder(ctl), steps


def _run(qapp, ctl: ImportController, paths, **kw):
    runner = ctl.import_paths([str(p) for p in paths], **kw)
    wait_for_idle(qapp, [runner], timeout=60)
    return runner


def _names(state: AppState) -> list[str]:
    return [a.name for a in state.project.assets.values()]


# ---------------------------------------------------------------------------------------------- import
def test_import_probes_isolates_failures_and_is_one_undo_step(qapp, media_dir, extra_media):
    state, ctl, rec, steps = _setup()
    _run(qapp, ctl, [media_dir / "land.mp4", extra_media["broken"], media_dir / "port.mp4", extra_media["notes"],
                     media_dir / "nope.mp4"])
    assert sorted(_names(state)) == ["land.mp4", "port.mp4"]
    ids, failures = rec.imported[-1]
    assert [state.project.assets[i].name for i in ids] == ["land.mp4", "port.mp4"]       # drop order kept
    assert len(failures) == 3
    assert any("壞掉.mp4" in f for f in failures)
    assert any("notes.txt" in f and "不支援" in f for f in failures)
    assert any("nope.mp4" in f and "找不到" in f for f in failures)
    assert steps == ["do:匯入素材"]
    land = state.project.assets[ids[0]]
    assert (land.width, land.height, land.has_audio) == (1280, 720, True)
    state.undo()
    assert state.project.assets == {}


def test_reimport_is_deduplicated_without_new_undo_step(qapp, media_dir):
    state, ctl, rec, steps = _setup()
    _run(qapp, ctl, [media_dir / "land.mp4"])
    first_id = rec.imported[-1][0][0]
    _run(qapp, ctl, [media_dir / "land.mp4", media_dir / "land.mp4"])
    assert len(state.project.assets) == 1
    assert rec.imported[-1] == ([first_id], [])                  # existing asset reported, not added twice
    assert steps == ["do:匯入素材"]


def test_folder_import_uses_natural_order_and_chinese_names(qapp, media_dir, tmp_path):
    folder = tmp_path / "毛片 資料夾"
    folder.mkdir()
    shutil.copy(media_dir / "silent.mp4", folder / "鏡頭10.mp4")
    shutil.copy(media_dir / "port.mp4", folder / "鏡頭2.mp4")
    (folder / "readme.txt").write_text("x", encoding="utf-8")
    state, ctl, rec, _ = _setup()
    _run(qapp, ctl, [folder])
    ids, failures = rec.imported[-1]
    assert [state.project.assets[i].name for i in ids] == ["鏡頭2.mp4", "鏡頭10.mp4"]
    assert failures == []                                          # unrelated files inside folders are ignored


def test_cancel_discards_partial_import(qapp, media_dir, monkeypatch):
    state, ctl, rec, steps = _setup()
    gate = threading.Event()
    real_probe = importer_mod.probe_asset

    def slow_probe(path):
        gate.wait(5)
        return real_probe(path)

    monkeypatch.setattr(importer_mod, "probe_asset", slow_probe)
    status: list[str] = []
    state.status.connect(status.append)
    runner = ctl.import_paths([str(media_dir / "land.mp4"), str(media_dir / "port.mp4")])
    assert ctl.busy
    runner.cancel()
    gate.set()
    wait_for_idle(qapp, [runner], timeout=30)
    assert state.project.assets == {} and steps == [] and rec.imported == []
    assert "已取消匯入" in status
    assert not ctl.busy


@pytest.mark.parametrize("cancel_works", [True, False])
def test_an_import_in_progress_is_dropped_when_another_project_replaces_the_current_one(qapp, media_dir, monkeypatch,
                                                                                         cancel_works):
    """File > New / Open while files are still being probed (a clean project asks nothing): the files were meant for the
    OLD project and must not show up in the new one - nor start the callbacks and the suggestion of that import.  Two lines
    of defence: the running import is cancelled, and one that gets past its last cancel check anyway is dropped on arrival."""
    state, ctl, rec, steps = _setup()
    if not cancel_works:
        monkeypatch.setattr(importer_mod.TaskRunner, "cancel", lambda self: None)
    gate = threading.Event()
    real_probe = importer_mod.probe_asset

    def slow_probe(path):
        gate.wait(5)
        return real_probe(path)

    monkeypatch.setattr(importer_mod, "probe_asset", slow_probe)
    status: list[str] = []
    state.status.connect(status.append)
    called: list = []
    runner = ctl.import_paths([str(media_dir / "land.mp4"), str(media_dir / "port.mp4")],
                              on_imported=lambda ids, failures: called.append(ids))
    assert ctl.busy
    state.replace_project(Project.new("新的專案"))
    assert runner.cancel_event.is_set() == cancel_works          # the old project's import is told to stop
    gate.set()
    wait_for_idle(qapp, [runner], timeout=30)
    assert state.project.assets == {} and not state.dirty and not state.history.can_undo
    assert rec.imported == [] and rec.suggestions == [] and called == []
    assert not ctl.busy
    # ... and the controller still works for the new project
    _run(qapp, ctl, [media_dir / "land.mp4"])
    assert _names(state) == ["land.mp4"] and len(called) == 0


def test_an_import_that_finished_before_the_project_changed_stays_in_its_project(qapp, media_dir):
    state, ctl, rec, _ = _setup()
    first = state.project
    _run(qapp, ctl, [media_dir / "land.mp4"])
    assert sorted(a.name for a in first.assets.values()) == ["land.mp4"]
    state.replace_project(Project.new("新的專案"))
    assert state.project.assets == {} and rec.imported and not ctl.busy


# ------------------------------------------------------------------------------------------ suggestion
def test_suggestion_long_landscape_video_means_shorts(qapp, extra_media, media_dir):
    state, ctl, rec, _ = _setup()
    _run(qapp, ctl, [extra_media["long"], media_dir / "music.wav"])
    (kind, ids), = rec.suggestions
    assert kind == "shorts" and state.project.assets[ids[0]].name == "長影片 測試.mp4"


def test_suggestion_two_videos_means_assemble(qapp, media_dir):
    state, ctl, rec, _ = _setup()
    _run(qapp, ctl, [media_dir / "land.mp4", media_dir / "port.mp4", media_dir / "music.wav"])
    (kind, ids), = rec.suggestions
    assert kind == "assemble"
    assert [state.project.assets[i].name for i in ids] == ["land.mp4", "port.mp4"]


def test_suggestion_short_single_clip_or_audio_means_none(qapp, media_dir):
    state, ctl, rec, _ = _setup()
    _run(qapp, ctl, [media_dir / "land.mp4"])            # 8 s: too short for shorts
    _run(qapp, ctl, [media_dir / "music.wav"])
    assert [k for k, _ in rec.suggestions] == ["none", "none"]
    _run(qapp, ctl, [media_dir / "land.mp4"], suggest=False)
    assert len(rec.suggestions) == 2


def test_suggest_kind_rules_with_fake_assets():
    def vid(aid, w, h, dur, kind="video"):
        return Asset(id=aid, path=f"/x/{aid}", kind=kind, duration=dur, width=w, height=h, has_video=kind != "audio")
    assert suggest_kind([vid("a", 1920, 1080, 45.0)]) == ("shorts", ["a"])
    assert suggest_kind([vid("a", 1920, 1080, 44.9)])[0] == "none"
    assert suggest_kind([vid("a", 1080, 1920, 300.0)])[0] == "none"          # already portrait
    assert suggest_kind([vid("a", 1920, 1080, 60), vid("b", 1080, 1920, 5)]) == ("assemble", ["a", "b"])
    assert suggest_kind([vid("i", 4000, 3000, 0, kind="image"), vid("m", 0, 0, 90, kind="audio")])[0] == "none"


# ---------------------------------------------------------------------------------- subtitles / placing
def test_subtitle_files_are_routed_not_imported(qapp, media_dir, extra_media):
    state, ctl, rec, _ = _setup()
    _run(qapp, ctl, [extra_media["srt"], media_dir / "land.mp4"])
    assert rec.subtitles == [[str(extra_media["srt"])]]
    assert _names(state) == ["land.mp4"]
    assert rec.imported[-1][1] == []


def test_place_assets_adapts_empty_sequence_in_one_undo_step(qapp, media_dir):
    state, ctl, rec, steps = _setup()
    _run(qapp, ctl, [media_dir / "port.mp4", media_dir / "land.mp4", media_dir / "music.wav"])
    port, land, music = rec.imported[-1][0]
    seq = state.sequence()
    assert (seq.width, seq.height, seq.fps) == (1920, 1080, 30.0)
    steps.clear()
    clip_ids = ctl.place_assets([port, land, music])
    assert steps == ["do:放到時間軸"] and len(clip_ids) == 3
    seq = state.sequence()
    assert (seq.width, seq.height, seq.fps) == (1080, 1920, 30.0)              # follows the first (portrait) video
    v1, a1 = seq.video_tracks[0], seq.audio_tracks[0]
    assert [c.asset_id for c in v1.clips] == [port, land]
    assert v1.clips[0].start == 0.0 and v1.clips[1].start == pytest.approx(v1.clips[0].end)
    assert [c.asset_id for c in a1.clips] == [music]
    state.undo()
    seq = state.sequence()
    assert (seq.width, seq.height) == (1920, 1080) and not list(seq.all_clips())


def test_place_assets_fps_follows_most_common_source(qapp, media_dir):
    state, ctl, rec, _ = _setup()
    _run(qapp, ctl, [media_dir / "silent.mp4"])
    ctl.place_assets(rec.imported[-1][0])
    seq = state.sequence()
    assert (seq.width, seq.height, seq.fps) == (1920, 1080, 25.0)
    _run(qapp, ctl, [media_dir / "port.mp4"])
    ctl.place_assets(rec.imported[-1][0])                         # sequence no longer empty: format kept
    assert (state.sequence().width, state.sequence().fps) == (1920, 25.0)


def test_place_assets_at_time_on_given_track(qapp, media_dir):
    state, ctl, rec, _ = _setup()
    _run(qapp, ctl, [media_dir / "land.mp4", media_dir / "silent.mp4", media_dir / "music.wav"])
    land, silent, music = rec.imported[-1][0]
    v2 = state.sequence().video_tracks[1]
    ids = ctl.place_assets([land, silent, music], track_id=v2.id, start=2.0, mode="overwrite")
    seq = state.sequence()
    v2 = seq.find_track(v2.id)
    assert len(ids) == 3
    assert [c.asset_id for c in v2.clips] == [land, silent]
    assert v2.clips[0].start == pytest.approx(2.0) and v2.clips[1].start == pytest.approx(v2.clips[0].end)
    assert [(c.asset_id, c.start) for c in seq.audio_tracks[0].clips] == [(music, pytest.approx(2.0))]


def test_place_assets_reports_locked_track_and_skips_unknown(qapp, media_dir):
    state, ctl, rec, steps = _setup()
    _run(qapp, ctl, [media_dir / "music.wav"])
    music = rec.imported[-1][0][0]
    for tr in state.sequence().audio_tracks:
        state.op("鎖定", "set_track_props", tr.id, locked=True)
    steps.clear()
    ids = ctl.place_assets([music, "does-not-exist"])
    assert len(ids) == 1                                          # a fresh unlocked audio track was added
    assert steps == ["do:放到時間軸"]
    assert ctl.place_assets(["does-not-exist"]) == []


def test_format_helpers():
    assert standard_fps(29.97002997) == 29.97 and standard_fps(29.87) == 29.97
    assert standard_fps(23.976023976) == 23.976 and standard_fps(15) == 24.0 and standard_fps(120) == 60.0
    a = Asset(id="p", path="/p", width=720, height=1280, fps=30.0, duration=5)
    b = Asset(id="l", path="/l", width=1280, height=720, fps=25.0, duration=5)
    c = Asset(id="l2", path="/l2", width=1280, height=720, fps=25.0, duration=5)
    assert sequence_format_for([a, b, c]) == (1080, 1920, 25.0)
    assert sequence_format_for([Asset(id="m", path="/m", kind="audio", has_video=False)]) is None


def test_place_assets_rejects_nothing_when_empty_list(qapp):
    state, ctl, _, steps = _setup()
    assert ctl.place_assets([]) == [] and steps == []
    assert isinstance(Path(importer_mod.__file__), Path)


# --------------------------------------------------------------------------------------------------------- natural order
def test_natural_order_handles_numbers_and_odd_digit_characters():
    """"clip 2" before "clip 10"; and file names with "①", "²" or full-width digits must not crash the sort (``isdigit`` is
    true for the first two, ``int`` cannot read them)."""
    from autocut.util import natural_key

    names = ["clip 10.mp4", "clip 2.mp4", "clip ①.mp4", "clip 1.mp4", "clip x².mp4", "clip ３.mp4", "片段 9.mp4", "片段 10.mp4",
             "②", "1②", "²", "7³"]                     # these four used to raise ValueError
    ordered = sorted(names, key=natural_key)
    assert ordered.index("clip 1.mp4") < ordered.index("clip 2.mp4") < ordered.index("clip 10.mp4")
    assert ordered.index("clip ３.mp4") < ordered.index("clip 10.mp4")            # a full-width 3 counts as 3
    assert ordered.index("片段 9.mp4") < ordered.index("片段 10.mp4")
    assert sorted(["b", "a1", "A2", ""], key=natural_key) == ["", "a1", "A2", "b"]
