"""Import controller: dropped / chosen files -> probed assets (ONE undo step) -> suggestion -> timeline placement.

The heavy part (walking folders, ``ffprobe``, fingerprinting) runs in a :class:`~autocut.gui.workers.TaskRunner`;
the project is only changed afterwards, in the GUI thread, through ``state.do("匯入素材", ...)``.

Signals
    imported(list[str] asset_ids, list[str] failures)   every media file of the drop that is now in the project
                                                          (newly added *or* already present), in drop order
    suggestion(str kind, list[str] asset_ids)            kind = "shorts" | "assemble" | "none"  (GUI_SPEC §4)
    subtitleFilesDropped(list[str] paths)                .srt/.ass/.vtt files: not assets, the window asks first
"""
from __future__ import annotations

import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from PySide6.QtCore import QObject, Signal

from ..core import ops
from ..core.ops import OpError
from ..core.project import AUDIO, VIDEO, Asset, Project, Sequence, Track
from ..media.probe import probe_asset
from ..util import (AUDIO_EXTS, IMAGE_EXTS, MEDIA_EXTS, PROJECT_EXT, SUBTITLE_EXTS, VIDEO_EXTS, collect_media_files,
                    media_kind_by_ext)
from .state import AppState
from .workers import TaskRunner, friendly_error

SHORTS_MIN_SECONDS = 45.0
FPS_MIN, FPS_MAX = 24.0, 60.0
STANDARD_FPS = (23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0)
PORTRAIT_SIZE, LANDSCAPE_SIZE = (1080, 1920), (1920, 1080)

ImportedCb = Callable[[list, list], None]


def path_key(path: str | os.PathLike) -> str:
    """Comparable form of a file path (same rule as ``Project.find_asset_by_path`` after ``resolve``)."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        resolved = Path(os.path.abspath(path))
    return os.path.normcase(os.path.abspath(str(resolved)))


def file_dialog_filter(subtitles: bool = False) -> str:
    """``QFileDialog`` filter string for importing media (optionally with subtitle files)."""
    def group(label: str, exts: Iterable[str]) -> str:
        return f"{label} ({' '.join('*' + e for e in sorted(exts))})"

    everything = MEDIA_EXTS | SUBTITLE_EXTS if subtitles else MEDIA_EXTS
    parts = [group("所有支援的檔案", everything), group("影片", VIDEO_EXTS), group("音訊", AUDIO_EXTS),
             group("圖片", IMAGE_EXTS)]
    if subtitles:
        parts.append(group("字幕", SUBTITLE_EXTS))
    return ";;".join(parts + ["所有檔案 (*)"])


def is_subtitle_file(path: str | os.PathLike) -> bool:
    return Path(str(path)).suffix.lower() in SUBTITLE_EXTS


def is_project_file(path: str | os.PathLike) -> bool:
    return Path(str(path)).suffix.lower() == PROJECT_EXT


def suggest_kind(assets: Iterable[Asset]) -> tuple[str, list[str]]:
    """GUI_SPEC §4: one long landscape video -> shorts; two or more videos -> assemble; else none."""
    assets = list(assets)
    videos = [a for a in assets if a.kind == "video" and a.has_video]
    if len(videos) == 1:
        v = videos[0]
        if v.width > v.height > 0 and v.duration >= SHORTS_MIN_SECONDS:
            return "shorts", [v.id]
    if len(videos) >= 2:
        return "assemble", [a.id for a in videos]
    return "none", [a.id for a in assets]


def standard_fps(fps: float) -> float:
    """Snap a measured rate to the nearest common rate (within 2 %) and clamp to 24–60 (23.976 is kept)."""
    if fps <= 0:
        return 0.0
    best = min(STANDARD_FPS, key=lambda s: abs(s - fps))
    value = best if abs(best - fps) / best < 0.02 else round(fps, 3)
    if value >= 23.97 and value < FPS_MIN:
        return value                                   # NTSC film rate, not a mistake
    return min(FPS_MAX, max(FPS_MIN, value))


def sequence_format_for(assets: Iterable[Asset]) -> Optional[tuple[int, int, Optional[float]]]:
    """(width, height, fps) an empty sequence should take for these assets, or ``None`` without picture."""
    assets = list(assets)
    visual = [a for a in assets if a.has_video or a.kind == "image"]
    if not visual:
        return None
    first = next((a for a in visual if a.kind == "video"), visual[0])
    w, h = PORTRAIT_SIZE if first.height > first.width else LANDSCAPE_SIZE
    rates = Counter(standard_fps(a.fps) for a in visual if a.kind == "video" and a.fps > 0)
    fps = rates.most_common(1)[0][0] if rates else None
    return w, h, fps


@dataclass
class _ProbeResult:
    """Worker output: per input file either a fresh ``Asset`` or the id of an asset already in the project."""
    items: list = field(default_factory=list)
    failures: list = field(default_factory=list)


class ImportController(QObject):
    imported = Signal(list, list)
    suggestion = Signal(str, list)
    subtitleFilesDropped = Signal(list)

    def __init__(self, state: AppState, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.runners: list[TaskRunner] = []
        self._epoch = 0                                   # counts the projects that replaced the one imports were started in
        state.project_replaced.connect(self._project_replaced)

    # ------------------------------------------------------------------ import
    @property
    def busy(self) -> bool:
        return any(r.running for r in self.runners)

    def import_paths(self, paths: Iterable[str | os.PathLike], *, suggest: bool = True,
                     on_imported: Optional[ImportedCb] = None) -> TaskRunner:
        """Start importing files/folders. ``on_imported(ids, failures)`` runs after the assets were added."""
        raw = [str(p) for p in paths]
        subtitles = [p for p in raw if is_subtitle_file(p) and os.path.isfile(p)]
        rest = [p for p in raw if p not in subtitles]
        known = {path_key(a.path): a.id for a in self.state.project.assets.values()}

        epoch = self._epoch
        runner = TaskRunner(lambda progress, cancel: _probe_all(rest, known, progress, cancel),
                            parent=self, name="匯入素材")
        runner.finished.connect(lambda res: self._finish(res, subtitles, suggest, on_imported, epoch))
        runner.cancelled.connect(lambda: self.state.status.emit("已取消匯入"))
        runner.failed.connect(lambda msg: self.state.status.emit(f"匯入失敗：{msg}"))
        runner.done.connect(lambda: self._forget(runner))
        self.runners.append(runner)
        return runner.start()

    def _forget(self, runner: TaskRunner) -> None:
        if runner in self.runners:
            self.runners.remove(runner)

    def _project_replaced(self) -> None:
        """File > New / Open: what is still being probed was meant for the OLD project.  Stop it; an import that finishes anyway
        (it was past its last check) is dropped in ``_finish`` rather than added to the project that is open now."""
        self._epoch += 1
        for r in self.runners:
            r.cancel()

    def _finish(self, res: _ProbeResult, subtitles: list[str], suggest: bool,
                on_imported: Optional[ImportedCb], epoch: Optional[int] = None) -> None:
        if epoch is not None and epoch != self._epoch:
            self.state.status.emit("已取消匯入：專案已經切換")
            return
        ids = self._add_assets(res.items)
        failures = list(res.failures)
        self._report(ids, failures)
        self.imported.emit(ids, failures)
        if on_imported is not None:
            on_imported(ids, failures)
        if subtitles:
            self.subtitleFilesDropped.emit(subtitles)
        if suggest and ids:
            kind, chosen = suggest_kind(self.state.project.assets[i] for i in ids)
            self.suggestion.emit(kind, chosen)

    def _add_assets(self, items: list) -> list[str]:
        """Add the probed assets in ONE undo step; returns ids in drop order (duplicates map to the existing id)."""
        if not any(isinstance(it, Asset) for it in items):
            return [it for it in items if it in self.state.project.assets]

        def run(p: Project) -> list[str]:
            out: list[str] = []
            for it in items:
                if isinstance(it, Asset):
                    out.append(ops.add_asset(p, it).id)
                elif it in p.assets:
                    out.append(it)
            return out

        try:
            ids = self.state.do("匯入素材", run)
        except OpError:
            return []
        return list(dict.fromkeys(ids))

    def _report(self, ids: list[str], failures: list[str]) -> None:
        if not ids and not failures:
            return
        msg = f"已匯入 {len(ids)} 個素材" if ids else "沒有匯入任何素材"
        if failures:
            msg += f"，{len(failures)} 個檔案無法匯入"
        self.state.status.emit(msg)

    # ------------------------------------------------------------------ timeline placement
    def place_assets(self, asset_ids: Iterable[str], track_id: Optional[str] = None, start: Optional[float] = None,
                     mode: str = "append") -> list[str]:
        """「放到時間軸」: pictures -> V1, audio-only -> A1, in order, as ONE undo step. Returns the new clip ids.

        ``start`` (with ``track_id``) places the first clip there and the rest right after it; ``mode`` is the
        ``place_clip`` mode used in that case (``append`` means ``overwrite`` at a given time). When the active
        sequence is still empty its size and fps follow the first video (GUI_SPEC §4).
        """
        ids = [i for i in asset_ids if i in self.state.project.assets]
        if not ids:
            return []
        seq_id = self.state.seq_id
        place_mode = "overwrite" if mode == "append" else mode

        def run(p: Project) -> list[str]:
            seq = p.sequence(seq_id)
            assets = [p.assets[i] for i in ids]
            if not any(True for _ in seq.all_clips()):
                fmt = sequence_format_for(assets)
                if fmt is not None:
                    ops.set_sequence_format(p, seq, *fmt)
            cursors: dict[str, Optional[float]] = {}
            out: list[str] = []
            for a in assets:
                tr = _target_track(p, seq, a, track_id)
                at = cursors.get(tr.id, start)
                if at is None:
                    clip = ops.place_clip(p, seq, a.id, tr.id, None, mode="append")
                else:
                    clip = ops.place_clip(p, seq, a.id, tr.id, at, mode=place_mode)
                    cursors[tr.id] = clip.end
                out.append(clip.id)
            return out

        try:
            new_ids = self.state.do("放到時間軸", run)
        except OpError:
            return []
        self.state.status.emit(f"已放到時間軸：{len(new_ids)} 個片段（Ctrl+Z 可復原）")
        return list(new_ids)


# ---------------------------------------------------------------------- helpers
def _wants_video_track(a: Asset) -> bool:
    return a.has_video or a.kind == "image"


def _target_track(p: Project, seq: Sequence, a: Asset, track_id: Optional[str]) -> Track:
    """The requested track when it suits the asset, else the first unlocked track of the right kind."""
    kind = VIDEO if _wants_video_track(a) else AUDIO
    if track_id:
        tr = seq.find_track(track_id)
        if tr is not None and tr.kind == kind:
            return tr
    tracks = seq.video_tracks if kind == VIDEO else seq.audio_tracks
    for tr in tracks:
        if not tr.locked:
            return tr
    return ops.add_track(p, seq, kind)


def _probe_all(paths: list[str], known: dict[str, str], progress: Callable[[float, str], None],
               cancel: threading.Event) -> _ProbeResult:
    """Worker thread: expand folders, probe each new file; one bad file never stops the others."""
    res = _ProbeResult()
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if is_project_file(p):
            res.failures.append(f"{p.name}：這是專案檔，請用「檔案 → 開啟專案」")
        elif p.is_dir():
            files += collect_media_files([p])
        elif not p.is_file():
            res.failures.append(f"{p.name}：找不到檔案")
        elif media_kind_by_ext(p) is None:
            res.failures.append(f"{p.name}：不支援的檔案格式")
        else:
            files += collect_media_files([p])
    seen: set[str] = set()
    for i, f in enumerate(files):
        if cancel.is_set():
            break
        key = path_key(f)
        if key in seen:
            continue
        seen.add(key)
        progress(i / max(1, len(files)), f"正在讀取素材 {i + 1}/{len(files)}：{f.name}")
        if key in known:
            res.items.append(known[key])
            continue
        try:
            res.items.append(probe_asset(f))
        except Exception as exc:  # noqa: BLE001 - isolate every file
            res.failures.append(f"{f.name}：{friendly_error(exc)}")
    progress(1.0, "讀取完成")
    return res
