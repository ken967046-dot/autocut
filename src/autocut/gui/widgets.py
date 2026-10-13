"""Small shared widgets: timecode input, clickable label, human-review box."""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSlider, QAbstractSpinBox, QComboBox,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTextEdit, QVBoxLayout,
                               QWidget)

from ..core import plan as planmod
from ..core.timecode import MAX_TIMELINE, fmt_timecode, parse_timecode
from . import theme
from .state import AppState


# ------------------------------------------------------------------------------------------ keyboard focus
def release_focus(w: QWidget) -> None:
    """Give the keyboard back to the timeline (the main window's ``focus_timeline``; without one, just drop the focus).

    Called after a value was committed with the mouse or with Enter: the arrow keys, S, Delete, Space, Ctrl+Z ... must go on
    driving the edit, not the control that was used last (a slider would step and commit again, a button would be pressed again
    by Space)."""
    hook = getattr(w.window(), "focus_timeline", None)
    if callable(hook):
        hook()
    else:
        w.clearFocus()


class _ClickHandsBackFocus(QObject):
    """Event filter of the controls made by :func:`keep_keyboard_on_timeline`: a press first hands the keyboard back, which
    also commits a number that is still being typed in another box (it loses the focus), then the press goes on."""

    def eventFilter(self, obj, ev) -> bool:                    # noqa: N802 (Qt API)
        if ev.type() == QEvent.MouseButtonPress and isinstance(obj, QWidget):
            release_focus(obj)
        return False


_HANDS_BACK: Optional[_ClickHandsBackFocus] = None


def keep_keyboard_on_timeline(root: QWidget) -> None:
    """``root`` and everything under it: buttons, check boxes, sliders and (non-editable) combo boxes take the focus by Tab only,
    never by a click.  Text boxes, number boxes and lists keep theirs - they need the keys while the person types."""
    global _HANDS_BACK
    if _HANDS_BACK is None:
        _HANDS_BACK = _ClickHandsBackFocus()
    for w in (root, *root.findChildren(QWidget)):
        if isinstance(w, (QAbstractButton, QAbstractSlider, QComboBox)) and not (isinstance(w, QComboBox) and w.isEditable()):
            policy = w.focusPolicy()
            if policy.value & Qt.ClickFocus.value:
                w.setFocusPolicy(Qt.TabFocus if policy.value & Qt.TabFocus.value else Qt.NoFocus)
                w.installEventFilter(_HANDS_BACK)


def keeps_its_keys(w: Optional[QWidget]) -> bool:
    """True when the widget that has the keyboard needs the keys for itself, so the timeline must not take over the ones it
    ignores: a box the person types into (the line edit inside a number box or an editable combo box too), a number box, a
    list / table / tree (Delete means 'remove this row' there)."""
    return isinstance(w, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QAbstractItemView))


def wants_project_undo(ev, line: QLineEdit) -> bool:
    """True for a ``ShortcutOverride`` of Undo / Redo that the window's Undo / Redo actions should get: a line edit claims
    those keys for its own text history whether it has any or not, so with nothing of its own to undo or redo (a committed
    number) Ctrl+Z would do nothing at all."""
    if ev.matches(QKeySequence.Undo):
        return not line.isUndoAvailable()
    if ev.matches(QKeySequence.Redo):
        return not line.isRedoAvailable()
    return False


class TimecodeEdit(QLineEdit):
    """Edits a time in seconds, shown as ``HH:MM:SS:FF`` for the given fps.

    ``timeCommitted(float)`` fires only when the user finishes editing with a parseable value (the text is
    reverted otherwise), so connect an undo-able edit to it – never to per-keystroke signals.
    """

    timeCommitted = Signal(float)

    def __init__(self, fps: float = 30.0, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fps = fps or 30.0
        self._t = 0.0
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setFixedWidth(104)
        self.setToolTip("時間碼 時:分:秒:格（也可輸入秒數，例如 12.5）")
        self.editingFinished.connect(self._commit)
        self._refresh()

    def set_fps(self, fps: float) -> None:
        self._fps = fps or 30.0
        self._refresh()

    def set_time(self, t: float) -> None:
        self._t = float(t)
        if not self.hasFocus():
            self._refresh()

    def time(self) -> float:
        return self._t

    def event(self, ev) -> bool:
        if ev.type() == QEvent.ShortcutOverride and wants_project_undo(ev, self):
            ev.ignore()
            return True
        return super().event(ev)

    def keyPressEvent(self, ev) -> None:                        # noqa: N802 (Qt API)
        super().keyPressEvent(ev)
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):             # committed (editingFinished): back to the timeline
            QTimer.singleShot(0, lambda: release_focus(self))

    def _refresh(self) -> None:
        self.setText(fmt_timecode(self._t, self._fps))

    def _commit(self) -> None:
        try:
            t = parse_timecode(self.text(), self._fps)
            if not math.isfinite(t):                      # "nan" / "1e999" are not times (max() would turn nan into 0)
                raise ValueError(self.text())
            t = min(max(0.0, t), MAX_TIMELINE)             # (1e308 is a number, but not a time: it cannot even be shown as one)
        except (ValueError, TypeError, OverflowError):
            self._refresh()
            return
        changed = abs(t - self._t) > 1e-6
        self._t = t
        self._refresh()
        if changed:
            self.timeCommitted.emit(t)


class ClickableLabel(QLabel):
    clicked = Signal()

    def mouseReleaseEvent(self, ev) -> None:                     # noqa: N802 (Qt API)
        if ev.button() == Qt.LeftButton and self.rect().contains(ev.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(ev)


VERDICT_TEXT = {"approved": "已通過（你審片後標記）", "needs_changes": "需要修改（你審片後標記）",
                "rejected": "已退回（你審片後標記）",
                "stale": "你之前的審片已過期：審片之後這個序列又改過了，請看過現在的內容再決定"}
QA_TEXT = {"not_run": "技術檢查：尚未執行", "pass": "技術檢查：通過（只代表檔案技術面，不代表好看）",
           "warn": "技術檢查：有警告", "fail": "技術檢查：未通過",
           "stale": "技術檢查：已過期（檢查之後序列又改過，請重新輸出再檢查）"}


class HumanReviewBox(QWidget):
    """Shows the technical-QA status of the auto-edit run behind the active sequence and lets the PERSON review it.

    Machine checks never turn this green: until the user clicks one of the three buttons the text says
    「尚未審片（機器不會代替你判斷好不好看）」.
    """

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._run_key: Optional[tuple[str, str]] = None    # (run, sequence) the notes box currently belongs to
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.qa_label = QLabel()
        self.qa_label.setWordWrap(True)
        self.verdict_label = QLabel()
        self.verdict_label.setWordWrap(True)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("審片備註（選填）")
        row = QHBoxLayout()
        self.btn_ok = QPushButton("通過")
        self.btn_fix = QPushButton("需修改")
        self.btn_no = QPushButton("退回")
        self.btn_ok.clicked.connect(lambda: self._click("approved"))
        self.btn_fix.clicked.connect(lambda: self._click("needs_changes"))
        self.btn_no.clicked.connect(lambda: self._click("rejected"))
        for b in (self.btn_ok, self.btn_fix, self.btn_no):
            row.addWidget(b)
        row.addStretch(1)
        hint = QLabel("看過成品後，由你決定。")
        hint.setObjectName("dim")
        for w in (self.qa_label, self.verdict_label, self.notes):
            lay.addWidget(w)
        lay.addLayout(row)
        lay.addWidget(hint)
        state.edited.connect(lambda *_: self.refresh())
        state.sequence_changed.connect(self.refresh)
        state.project_replaced.connect(self.refresh)
        keep_keyboard_on_timeline(self)               # after a click Space / Enter must not press the verdict button again
        self.refresh()

    def refresh(self) -> None:
        run = self.state.run_for_sequence()
        try:
            seq = self.state.sequence()
        except KeyError:
            run, seq = None, None
        has = run is not None and seq is not None
        for w in (self.btn_ok, self.btn_fix, self.btn_no, self.notes):
            w.setEnabled(has)
        key = (run.get("run_id"), seq.id) if has else None
        verdict, notes = planmod.review_state(run, seq) if has else ("pending", "")
        if key != self._run_key:                           # notes typed for one sequence must never end up on another
            self._run_key = key
            self.notes.setText(notes if has else "")
        if not has:
            self.qa_label.setText("這個序列不是自動剪輯產生的，沒有審片紀錄。")
            self.verdict_label.setText("")
            return
        qa, _summary = planmod.qa_state(run, seq)
        self.qa_label.setText(QA_TEXT.get(qa, "技術檢查：未知"))
        if verdict == "pending":
            self.verdict_label.setText("尚未審片（機器不會代替你判斷好不好看）")
            self.verdict_label.setStyleSheet(f"color: {theme.WARN};")
        else:
            self.verdict_label.setText(VERDICT_TEXT.get(verdict, verdict))
            self.verdict_label.setStyleSheet(f"color: {theme.OK if verdict == 'approved' else theme.WARN};")

    def _click(self, verdict: str) -> None:
        run = self.state.run_for_sequence()
        if run is None:
            return
        self.state.human_review(run["run_id"], verdict, self.notes.text().strip())
        self.refresh()
