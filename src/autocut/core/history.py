"""Undo / redo by compressed snapshots.

Every user-visible edit runs inside :meth:`History.run` (or the
:meth:`History.transaction` context manager).  The project content is
serialised before and after; if nothing changed no entry is recorded, if the
edit raises the project is rolled back.  Snapshots are zlib-compressed JSON,
so even a few hundred clips x 100 steps stay in the low megabytes.
"""
from __future__ import annotations

import json
import zlib
from contextlib import contextmanager
from typing import Callable, Iterator

from ..util import canonical_json
from .project import Project


class History:
    def __init__(self, project: Project, limit: int = 200) -> None:
        self.project = project
        self.limit = limit
        self._undo: list[tuple[str, bytes]] = []
        self._redo: list[tuple[str, bytes]] = []
        self._depth = 0
        self._listeners: list[Callable[[str, str], None]] = []
        self.saved_revision: int | None = project.revision

    # ------------------------------------------------------------ listeners
    def on_change(self, cb: Callable[[str, str], None]) -> None:
        """``cb(kind, label)`` with kind in {"do", "undo", "redo", "reset"}."""
        self._listeners.append(cb)

    def _emit(self, kind: str, label: str) -> None:
        for cb in list(self._listeners):
            cb(kind, label)

    # --------------------------------------------------------------- state
    def _content(self) -> str:
        return canonical_json(self.project.content_dict())

    @staticmethod
    def _pack(text: str) -> bytes:
        return zlib.compress(text.encode("utf-8"), 3)

    @staticmethod
    def _unpack(blob: bytes) -> dict:
        return json.loads(zlib.decompress(blob).decode("utf-8"))

    def _restore(self, text: str) -> None:
        d = json.loads(text)
        d["name"] = self.project.name          # the name belongs to the document (Save As sets it), not to an edit: undo keeps it
        self.project.replace_content(d)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str:
        return self._undo[-1][0] if self._undo else ""

    @property
    def redo_label(self) -> str:
        return self._redo[-1][0] if self._redo else ""

    @property
    def dirty(self) -> bool:
        return self.saved_revision != self.project.revision

    def mark_saved(self) -> None:
        self.saved_revision = self.project.revision

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._emit("reset", "")

    # --------------------------------------------------------------- editing
    @contextmanager
    def transaction(self, label: str) -> Iterator[Project]:
        if self._depth > 0:          # nested: the outermost transaction records the step
            self._depth += 1
            try:
                yield self.project
            finally:
                self._depth -= 1
            return
        before = self._content()
        self._depth = 1
        try:
            yield self.project
        except BaseException:
            self._depth = 0
            self._restore(before)
            raise
        self._depth = 0
        after = self._content()
        if after != before:
            self._undo.append((label, self._pack(before)))
            if len(self._undo) > self.limit:
                del self._undo[0]
            self._redo.clear()
            self.project.touch()
            self._emit("do", label)

    def run(self, label: str, fn: Callable[[Project], object]) -> object:
        with self.transaction(label) as p:
            return fn(p)

    def undo(self) -> str | None:
        if not self._undo:
            return None
        label, blob = self._undo.pop()
        self._redo.append((label, self._pack(self._content())))
        self._restore(zlib.decompress(blob).decode("utf-8"))
        self.project.touch()
        self._emit("undo", label)
        return label

    def redo(self) -> str | None:
        if not self._redo:
            return None
        label, blob = self._redo.pop()
        self._undo.append((label, self._pack(self._content())))
        self._restore(zlib.decompress(blob).decode("utf-8"))
        self.project.touch()
        self._emit("redo", label)
        return label
