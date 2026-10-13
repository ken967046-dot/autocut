"""Shared, observable application state.

Every widget reads the project through ``state.project`` / ``state.sequence()`` and changes it **only** through
``state.op(...)`` / ``state.do(...)`` (which go through :class:`autocut.core.editor.Editor`, i.e. one undo step each).
Widgets never mutate ``Clip`` / ``Track`` fields themselves.

Signals (all emitted in the GUI thread)
    project_replaced()            new / opened project (rebuild everything)
    edited(kind, label)           after *any* project change: kind in do | undo | redo | reset
    sequence_changed()            the active sequence switched (``seq_id``)
    selection_changed()           clip selection changed
    cue_selection_changed()       caption-cue selection changed
    playhead_changed(float)       playhead moved
    reframe_requested(str)        a widget asks the monitor to enter reframe mode for this clip id
    status(str)                   one-line message for the status bar
    dirty_changed(bool)           unsaved-changes flag flipped
    title_changed(str)            window title text (project name, ``*`` when dirty)
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

from PySide6.QtCore import QObject, Signal

from ..core import plan as planmod
from ..core.editor import Editor
from ..core.project import Clip, Project, Sequence
from ..core.timecode import snap_time


class AppState(QObject):
    project_replaced = Signal()
    edited = Signal(str, str)
    sequence_changed = Signal()
    selection_changed = Signal()
    cue_selection_changed = Signal()
    playhead_changed = Signal(float)
    reframe_requested = Signal(str)
    status = Signal(str)
    dirty_changed = Signal(bool)
    title_changed = Signal(str)

    def __init__(self, project: Optional[Project] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.editor = Editor(project)
        self.seq_id: str = ""
        self.playhead: float = 0.0
        self.selection: set[str] = set()          # selected clip ids (in the active sequence)
        self.cue_selection: set[str] = set()      # selected caption-cue ids
        self._was_dirty = False
        self._bind_editor()

    # ------------------------------------------------------------------ project / editor plumbing
    def _bind_editor(self) -> None:
        self.editor.history.on_change(self._on_history)
        p = self.editor.project
        self.seq_id = p.active if any(s.id == p.active for s in p.sequences) else (
            p.sequences[0].id if p.sequences else "")
        self.playhead = 0.0
        self.selection.clear()
        self.cue_selection.clear()
        self._was_dirty = self.dirty

    @property
    def project(self) -> Project:
        return self.editor.project

    @property
    def history(self):
        return self.editor.history

    def sequence(self) -> Sequence:
        """The active sequence (falls back to the first one if ``seq_id`` vanished)."""
        p = self.project
        for s in p.sequences:
            if s.id == self.seq_id:
                return s
        if p.sequences:
            self.seq_id = p.sequences[0].id
            return p.sequences[0]
        raise KeyError("專案沒有任何序列")

    @property
    def fps(self) -> float:
        return self.sequence().fps

    # ------------------------------------------------------------------ editing (the only write path)
    def op(self, label: str, name: str, /, *args: Any, **kw: Any) -> Any:
        """``ops.<name>(project, active_sequence, *args)`` as one undoable step (``label`` / ``name`` are positional-only so
        ``name=`` can be an op keyword, e.g. ``set_track_props(..., name=...)``). Errors are reported on the status line
        and re-raised as ``OpError`` for callers that want to react."""
        return self._guard(lambda: self.editor.op(label, name, self.seq_id, *args, **kw))

    def project_op(self, label: str, name: str, /, *args: Any, **kw: Any) -> Any:
        return self._guard(lambda: self.editor.project_op(label, name, *args, **kw))

    def do(self, label: str, fn: Callable[[Project], Any]) -> Any:
        """Several ops as ONE undo step: ``state.do("標題", lambda p: (ops.x(p, seq...), ops.y(...)))``."""
        return self._guard(lambda: self.editor.do(label, fn))

    def _guard(self, call: Callable[[], Any]) -> Any:
        from ..core.ops import OpError
        try:
            return call()
        except OpError as exc:
            self.status.emit(str(exc))
            raise

    def undo(self) -> Optional[str]:
        label = self.editor.undo()
        self.status.emit(f"已復原：{label}" if label else "沒有可復原的動作")
        return label

    def redo(self) -> Optional[str]:
        label = self.editor.redo()
        self.status.emit(f"已重做：{label}" if label else "沒有可重做的動作")
        return label

    def _on_history(self, kind: str, label: str) -> None:
        self._sync_after_change()
        self.edited.emit(kind, label)
        self._emit_dirty()

    def _sync_after_change(self) -> None:
        """Keep ``seq_id``/selection valid after do / undo / redo."""
        p = self.project
        ids = {s.id for s in p.sequences}
        switched = False
        if self.seq_id not in ids:
            self.seq_id = p.active if p.active in ids else (p.sequences[0].id if p.sequences else "")
            switched = True
        if self.selection or self.cue_selection:
            try:
                seq = self.sequence()
                clip_ids = {c.id for _, c in seq.all_clips()}
                cue_ids = {q.id for q in seq.cues}
            except KeyError:
                clip_ids, cue_ids = set(), set()
            keep, keepq = self.selection & clip_ids, self.cue_selection & cue_ids
            if keep != self.selection:
                self.selection = keep
                self.selection_changed.emit()
            if keepq != self.cue_selection:
                self.cue_selection = keepq
                self.cue_selection_changed.emit()
        if switched:
            self.sequence_changed.emit()

    # ------------------------------------------------------------------ sequence / playhead / selection
    def set_sequence(self, seq_id: str) -> None:
        if seq_id == self.seq_id or not any(s.id == seq_id for s in self.project.sequences):
            return
        self.seq_id = seq_id
        self.selection.clear()
        self.cue_selection.clear()
        self.playhead = 0.0
        self.sequence_changed.emit()
        self.selection_changed.emit()
        self.cue_selection_changed.emit()
        self.playhead_changed.emit(0.0)

    def set_playhead(self, t: float, snap: bool = True) -> float:
        t = max(0.0, float(t))
        if snap:
            t = snap_time(t, self.fps)
        if abs(t - self.playhead) > 1e-9:
            self.playhead = t
            self.playhead_changed.emit(t)
        return t

    def select(self, clip_ids: Iterable[str], mode: str = "replace") -> None:
        """mode: replace | add | toggle | remove."""
        ids = set(clip_ids)
        old = set(self.selection)
        if mode == "replace":
            new = ids
        elif mode == "add":
            new = old | ids
        elif mode == "toggle":
            new = old ^ ids
        elif mode == "remove":
            new = old - ids
        else:
            raise ValueError(mode)
        if new != old:
            self.selection = new
            self.selection_changed.emit()

    def clear_selection(self) -> None:
        self.select([])
        self.select_cues([])

    def select_cues(self, cue_ids: Iterable[str], mode: str = "replace") -> None:
        ids = set(cue_ids)
        old = set(self.cue_selection)
        new = {"replace": ids, "add": old | ids, "toggle": old ^ ids, "remove": old - ids}[mode]
        if new != old:
            self.cue_selection = new
            self.cue_selection_changed.emit()

    def selected_clips(self) -> list[Clip]:
        seq = self.sequence()
        return [c for _, c in seq.all_clips() if c.id in self.selection]

    def request_reframe(self, clip_id: str) -> None:
        self.reframe_requested.emit(clip_id)

    # ------------------------------------------------------------------ in/out range
    def get_range(self) -> Optional[tuple[float, float]]:
        r = self.sequence().meta.get("range")
        if isinstance(r, (list, tuple)) and len(r) == 2 and r[1] > r[0]:
            return float(r[0]), float(r[1])
        return None

    def set_range(self, t_in: Optional[float], t_out: Optional[float]) -> None:
        self.op("設定入出點", "set_sequence_range", t_in, t_out)

    # ------------------------------------------------------------------ files
    @property
    def project_path(self) -> str:
        return self.project.path or ""

    @property
    def dirty(self) -> bool:
        return self.editor.history.dirty

    @property
    def title(self) -> str:
        name = os.path.basename(self.project.path) if self.project.path else self.project.name
        return f"{name}{' *' if self.dirty else ''}"

    def _emit_dirty(self) -> None:
        d = self.dirty
        if d != self._was_dirty:
            self._was_dirty = d
            self.dirty_changed.emit(d)
        self.title_changed.emit(self.title)

    def replace_project(self, project: Project) -> None:
        self.editor.set_project(project)
        self._bind_editor()
        self.project_replaced.emit()
        self.sequence_changed.emit()
        self.selection_changed.emit()
        self.playhead_changed.emit(0.0)
        self.dirty_changed.emit(self.dirty)
        self.title_changed.emit(self.title)

    def new_project(self, name: str = "未命名專案") -> None:
        self.replace_project(Project.new(name))
        self.status.emit("已建立新專案")

    def open_project(self, path: str) -> None:
        p = Project.load(path)
        self.replace_project(p)
        self.status.emit(f"已開啟 {os.path.basename(path)}")

    def save_project(self, path: Optional[str] = None) -> str:
        p = self.project
        old_name = p.name
        # reopen on the sequence the user was looking at; 「另存新檔」 names the project after the file - both before it is written
        self.editor.set_document(name=os.path.splitext(os.path.basename(path))[0] if path else None,
                                 active=self.seq_id or None)
        try:
            target = p.save(path or p.path)
        except BaseException:
            self.editor.set_document(name=old_name)
            raise
        self.editor.history.mark_saved()
        self._emit_dirty()
        self.status.emit(f"已儲存 {os.path.basename(target)}")
        return target

    # ------------------------------------------------------------------ provenance helpers (auto-edit runs)
    def run_for_sequence(self, seq_id: Optional[str] = None) -> Optional[dict]:
        """The ``project.meta["ai_runs"]`` record that produced the sequence (if any)."""
        try:
            seq = self.project.sequence(seq_id or self.seq_id)
        except KeyError:
            return None
        run_id = (seq.meta.get("ai") or {}).get("run_id")
        return planmod.find_run(self.project, run_id) if run_id else None

    def record_technical_qa(self, run_id: str, status: str, summary: str = "", seq_id: Optional[str] = None,
                            sequence_sha: Optional[str] = None) -> None:
        """Machine evidence only (``pass`` is *not* a taste approval).  Tied to ONE sequence (default: the active one);
        ``sequence_sha`` is the hash of the content that was actually rendered, so an edit made while the check ran
        shows up as 「已過期」."""
        sid = seq_id or self.seq_id
        self.do("記錄技術檢查結果", lambda p: planmod.set_technical_qa(
            p, run_id, status, summary, sequence=p.sequence(sid), sequence_sha=sequence_sha))

    def human_review(self, run_id: str, verdict: str, notes: str = "", seq_id: Optional[str] = None) -> None:
        """ONLY call this from an explicit user click (approved | needs_changes | rejected).  The verdict belongs to the
        sequence as it is right now (default: the active one); edit it later and the verdict turns 「已過期」."""
        sid = seq_id or self.seq_id
        self.do("人工審片", lambda p: planmod.set_human_review(p, run_id, verdict, notes, sequence=p.sequence(sid)))
