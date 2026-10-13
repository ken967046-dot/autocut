"""Undoable facade over :mod:`autocut.core.ops` used by the GUI and the CLI."""
from __future__ import annotations

from typing import Any, Callable

from . import ops
from .history import History
from .project import Project, Sequence


class Editor:
    """Owns a project and its history. Every mutation goes through :meth:`op` / :meth:`do`."""

    def __init__(self, project: Project | None = None) -> None:
        self.project = project or Project.new()
        self.history = History(self.project)

    def set_project(self, project: Project) -> None:
        listeners = list(self.history._listeners)
        self.project = project
        self.history = History(project)
        self.history._listeners = listeners
        self.history._emit("reset", "")

    # ---- generic
    def do(self, label: str, fn: Callable[[Project], Any]) -> Any:
        return self.history.run(label, fn)

    def op(self, label: str, name: str, seq_id: str | None, /, *args: Any, **kw: Any) -> Any:
        """Run ``ops.<name>(project, sequence, *args)`` as one undoable step."""
        fn = getattr(ops, name)

        def run(p: Project) -> Any:
            return fn(p, p.sequence(seq_id), *args, **kw)

        return self.history.run(label, run)

    def project_op(self, label: str, name: str, /, *args: Any, **kw: Any) -> Any:
        fn = getattr(ops, name)
        return self.history.run(label, lambda p: fn(p, *args, **kw))

    def set_document(self, name: str | None = None, active: str | None = None) -> None:
        """What Save / Save As write into the file besides the edits: the project's name and the sequence to reopen on.
        Not an edit - no undo step, the project does not turn dirty, and undo / redo keep the name (``History._restore``).
        ``active`` is ignored when no such sequence exists."""
        p = self.project
        if name is not None:
            p.name = name
        if active is not None and any(s.id == active for s in p.sequences):
            p.active = active

    # ---- shortcuts
    @property
    def sequence(self) -> Sequence:
        return self.project.active_sequence()

    def undo(self) -> str | None:
        return self.history.undo()

    def redo(self) -> str | None:
        return self.history.redo()
