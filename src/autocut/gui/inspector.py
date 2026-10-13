"""Inspector panel (GUI_SPEC §5): 片段 / 字幕 / 序列 / 檢查.

The panel only *reads* the project.  Every change is ONE undo step through ``state.op`` / ``state.do`` and is
triggered only by user-only signals: ``editingFinished`` of a value the user actually changed, slider release (or a
short pause after keyboard steps on a slider), combo ``activated`` and checkbox ``clicked``.  Programmatic refreshes
block signals and never commit, so there are no feedback loops, and a field the user is typing in is never reset.
"""
from __future__ import annotations

import html
import importlib
import math
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from PySide6.QtCore import QEvent, QItemSelection, QItemSelectionModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFontComboBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSlider, QTableWidget, QTableWidgetItem,
                               QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import ops
from ..core import plan as planmod
from ..core.ops import OpError
from ..core.project import VIDEO, Clip, Project, Sequence, Track, validate_sequence
from ..core.subtitles import load_subtitle_file, to_srt
from ..core.timecode import fmt_clock, fmt_timecode, parse_timecode
from ..util import atomic_write_text, human_duration
from . import theme
from .state import AppState
from .widgets import (QA_TEXT, HumanReviewBox, TimecodeEdit, keep_keyboard_on_timeline, release_focus,
                      wants_project_undo)
from .workers import TaskRunner

INSTALL_HINT = "安裝方式：在命令列執行 pip install -r requirements-speech.txt"
SPEECH_MODEL = "small"
SPEECH_LANGUAGE = "zh"


# ============================================================================ dialogs (monkeypatched in tests)
def with_install_hint(reason: str) -> str:
    """The speech-unavailable reason plus the install instruction (unless it already contains it)."""
    reason = (reason or "語音辨識目前無法使用。").strip()
    return reason if "requirements-speech.txt" in reason else f"{reason}\n{INSTALL_HINT}"


def ask(parent: Optional[QWidget], title: str, text: str, yes: str = "確定", no: str = "取消") -> bool:
    """Modal question with Chinese button texts; True only when the user clicks ``yes``."""
    box = QMessageBox(QMessageBox.Question, title, text, QMessageBox.NoButton, parent)
    ok = box.addButton(yes, QMessageBox.AcceptRole)
    box.addButton(no, QMessageBox.RejectRole)
    box.setDefaultButton(ok)
    box.exec()
    return box.clickedButton() is ok


def inform(parent: Optional[QWidget], title: str, text: str, warning: bool = False) -> None:
    box = QMessageBox(QMessageBox.Warning if warning else QMessageBox.Information, title, text,
                      QMessageBox.NoButton, parent)
    box.addButton("確定", QMessageBox.AcceptRole)
    box.exec()


def open_file(parent: Optional[QWidget], title: str, filters: str) -> str:
    return QFileDialog.getOpenFileName(parent, title, "", filters)[0]


def save_file(parent: Optional[QWidget], title: str, suggested: str, filters: str) -> str:
    return QFileDialog.getSaveFileName(parent, title, suggested, filters)[0]


def pick_color(parent: Optional[QWidget], initial: str, title: str) -> Optional[str]:
    c = QColorDialog.getColor(QColor(initial), parent, title)
    return c.name().upper() if c.isValid() else None


def ask_number(parent: Optional[QWidget], title: str, label: str, value: float = 0.0) -> Optional[float]:
    dlg = QInputDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setLabelText(label)
    dlg.setInputMode(QInputDialog.DoubleInput)
    dlg.setDoubleRange(-3600.0, 3600.0)
    dlg.setDoubleDecimals(3)
    dlg.setDoubleValue(value)
    dlg.setOkButtonText("確定")
    dlg.setCancelButtonText("取消")
    return dlg.doubleValue() if dlg.exec() else None


# ============================================================================ small field widgets
@contextmanager
def quiet(*widgets: QWidget) -> Iterator[None]:
    """Block the widgets' own signals while the program updates them (programmatic updates never commit)."""
    old = [w.blockSignals(True) for w in widgets]
    try:
        yield
    finally:
        for w, was in zip(widgets, old):
            w.blockSignals(was)


def common(values: list) -> Any:
    """The value shared by every selected clip, or ``None`` when they differ (shown as 「—」)."""
    first = values[0]
    for v in values[1:]:
        if isinstance(first, float) or isinstance(v, float):
            if abs(float(v) - float(first)) > 1e-6:
                return None
        elif v != first:
            return None
    return first


def hbox(*items: Any) -> QWidget:
    """A margin-less row of widgets (``None`` = stretch)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for it in items:
        if it is None:
            lay.addStretch(1)
        else:
            lay.addWidget(it)
    return w


def form(group: QGroupBox) -> QFormLayout:
    f = QFormLayout(group)
    f.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    f.setHorizontalSpacing(10)
    f.setVerticalSpacing(6)
    return f


def dim_label(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("dim")
    lab.setWordWrap(True)
    return lab


class NumField(QDoubleSpinBox):
    """Number box that emits ``committed`` only when the user finished editing a value they changed.

    ``set_value(None)`` shows a mixed multi-selection as 「—」.  Programmatic updates never emit and are skipped while
    the user is typing.  The mouse wheel only acts on a focused box, so scrolling the panel cannot change values.
    """

    committed = Signal(float)

    def __init__(self, lo: float, hi: float, step: float, decimals: int = 2, suffix: str = "",
                 default: Optional[float] = None, parent: Optional[QWidget] = None) -> None:
        self._mixed = False
        self._shown: Optional[float] = None
        self._typed = False
        self._default = lo if default is None else default
        super().__init__(parent)
        self.setRange(lo, hi)
        self.setSingleStep(step)
        self.setDecimals(decimals)
        self.setSuffix(suffix)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setMinimumWidth(92)
        self.lineEdit().textEdited.connect(self._on_typed)
        self.lineEdit().returnPressed.connect(lambda: QTimer.singleShot(0, lambda: release_focus(self)))
        self.lineEdit().installEventFilter(self)                   # the keys go to the inner line edit (the focus proxy)
        self.editingFinished.connect(self._finish)

    def set_value(self, v: Optional[float]) -> None:
        """Programmatic update (never commits); ``None`` = mixed values."""
        if self._typed and self.hasFocus():
            return
        self._typed = False
        self._mixed = v is None
        self._shown = None if v is None else float(v)
        with quiet(self):
            self.setValue(self.minimum() if v is None else float(v))
        self.lineEdit().setText(self.textFromValue(self.value()) + self.suffix())

    def is_mixed(self) -> bool:
        return self._mixed

    def textFromValue(self, v: float) -> str:                    # noqa: N802 (Qt API)
        return "—" if self._mixed else super().textFromValue(v)

    def stepBy(self, steps: int) -> None:                          # noqa: N802
        if self._mixed:
            self._mixed = False
            with quiet(self):
                self.setValue(self._default)
        self._typed = True
        super().stepBy(steps)

    def wheelEvent(self, ev) -> None:                              # noqa: N802
        if self.hasFocus():
            super().wheelEvent(ev)
        else:
            ev.ignore()

    def event(self, ev) -> bool:
        # The keyboard belongs to the box itself; it passes ShortcutOverride on to its line edit with a direct call, which no
        # event filter sees - so the decision is made here as well as in ``eventFilter`` (events sent to the line edit).
        if ev.type() == QEvent.ShortcutOverride and wants_project_undo(ev, self.lineEdit()):
            ev.ignore()                                             # Ctrl+Z / Ctrl+Y undo the project, not the box
            return True
        return super().event(ev)

    def eventFilter(self, obj, ev) -> bool:                       # noqa: N802
        if obj is self.lineEdit() and ev.type() == QEvent.ShortcutOverride and wants_project_undo(ev, self.lineEdit()):
            ev.ignore()                                             # Ctrl+Z / Ctrl+Y undo the project, not the box
            return True
        return super().eventFilter(obj, ev)

    def focusInEvent(self, ev) -> None:                           # noqa: N802
        super().focusInEvent(ev)
        if self._mixed:
            QTimer.singleShot(0, self.selectAll)

    def _on_typed(self, text: str) -> None:
        self._typed = True
        if self._shown is None:                                    # was mixed: only real input un-mixes it
            stripped = text.replace(self.suffix(), "").strip() if self.suffix() else text.strip()
            self._mixed = stripped in ("", "—")

    def _finish(self) -> None:
        self._typed = False
        if self._mixed:
            self.lineEdit().setText("—" + self.suffix())
            return
        v = self.value()
        if self._shown is not None and abs(v - self._shown) < 0.5 * 10 ** -self.decimals():
            return
        self._shown = v
        self.committed.emit(v)


class _FocusWheel:
    """Mixin: the mouse wheel only acts on a control that has keyboard focus.

    Without it, scrolling the inspector with the pointer over a slider or a combo box would edit that control (a combo
    even emits ``activated``, i.e. commits) instead of scrolling the panel.  ``StrongFocus`` matters: the default
    ``WheelFocus`` would hand the control the focus on the first notch and let the same notch edit it.
    """

    def _wheel_guard(self) -> None:
        self.setFocusPolicy(Qt.StrongFocus)
        keep_keyboard_on_timeline(self)                            # ... but a click does not give them the keyboard (Tab does)

    def wheelEvent(self, ev) -> None:                              # noqa: N802 (Qt API)
        if self.hasFocus():
            super().wheelEvent(ev)
        else:
            ev.ignore()


class FocusSlider(_FocusWheel, QSlider):
    def __init__(self, orientation: Qt.Orientation, parent: Optional[QWidget] = None) -> None:
        super().__init__(orientation, parent)
        self._wheel_guard()


class FocusCombo(_FocusWheel, QComboBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._wheel_guard()


class FocusFontCombo(_FocusWheel, QFontComboBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._wheel_guard()


class SliderField(QWidget):
    """Slider + number box for one value.

    Commits on slider release, ~0.35 s after keyboard / page steps, or when the number box finishes editing —
    never once per slider tick, and never for a press / release that did not move the handle (the stored value may
    be off the slider's grid or beyond its range, and a mixed multi-selection must not be overwritten by a click).
    """

    committed = Signal(float)

    def __init__(self, lo: float, hi: float, step: float, decimals: int, scale: int, suffix: str = "",
                 box_hi: Optional[float] = None, default: Optional[float] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scale = scale
        self._last: Optional[float] = None
        self._moved = False
        self.slider = FocusSlider(Qt.Horizontal)
        self.slider.setRange(round(lo * scale), round(hi * scale))
        self.slider.setSingleStep(max(1, round(step * scale)))
        self.slider.setPageStep(max(1, round(step * scale * 10)))
        self.box = NumField(lo, hi if box_hi is None else box_hi, step, decimals, suffix, default)
        self.box.setFixedWidth(92)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._commit_slider)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.box)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(self._commit_slider)
        self.box.committed.connect(self._on_box)

    def set_value(self, v: Optional[float]) -> None:
        """Programmatic update; ignored while the user drags the slider or a keyboard step is pending."""
        if self.slider.isSliderDown() or self._timer.isActive():
            return
        self._last = v
        self._moved = False
        self.box.set_value(v)
        with quiet(self.slider):
            mid = (self.slider.minimum() + self.slider.maximum()) // 2
            self.slider.setValue(mid if v is None else self._grid(v))

    def _grid(self, v: float) -> int:
        """Handle position of ``v`` (a box value beyond the slider's range sits at its end)."""
        return max(self.slider.minimum(), min(self.slider.maximum(), round(v * self._scale)))

    def _on_slider(self, iv: int) -> None:
        self._moved = True
        self.box.set_value(iv / self._scale)
        if not self.slider.isSliderDown():
            self._timer.start()

    def _commit_slider(self) -> None:
        self._timer.stop()
        moved, self._moved = self._moved, False
        iv = self.slider.value()
        if not moved or (self._last is not None and iv == self._grid(self._last)):
            if moved:                              # moved away and back: show the stored value, commit nothing
                self.set_value(self._last)
            return
        self._last = iv / self._scale
        self.committed.emit(self._last)

    def _on_box(self, v: float) -> None:
        self._timer.stop()
        self._moved = False
        self._last = v
        with quiet(self.slider):
            self.slider.setValue(round(v * self._scale))
        self.committed.emit(v)


class TimeField(TimecodeEdit):
    """``TimecodeEdit`` that also follows the project while focused (until the user types), so leaving the field
    can never re-commit a stale value, e.g. after an undo."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(30.0, parent)
        self._fps_now = 30.0
        self._typed = False
        self.textEdited.connect(self._on_typed)
        self.editingFinished.connect(self._on_done)            # runs after TimecodeEdit's own commit

    def set_fps(self, fps: float) -> None:
        fps = fps or 30.0
        if abs(fps - self._fps_now) > 1e-9:
            self._fps_now = fps
            super().set_fps(fps)

    def set_time(self, t: float) -> None:
        super().set_time(t)
        if self.hasFocus() and not self._typed:
            self.setText(fmt_timecode(float(t), self._fps_now))

    def _on_typed(self, _text: str) -> None:
        self._typed = True

    def _on_done(self) -> None:
        self._typed = False
        self.setText(fmt_timecode(self.time(), self._fps_now))


class ColorButton(QPushButton):
    """Colour swatch; a click opens the colour picker and emits ``committed('#RRGGBB')`` if the colour changed."""

    committed = Signal(str)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title
        self._color = "#000000"
        self.setFixedSize(46, 24)
        self.setToolTip(f"{title}（按一下選擇顏色）")
        self.clicked.connect(self._pick)
        self.set_color(self._color)

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        q = QColor(color)
        self._color = q.name().upper() if q.isValid() else "#000000"
        self.setStyleSheet(f"QPushButton {{ background: {self._color}; border: 1px solid {theme.TEXT_DIM}; "
                           f"border-radius: 3px; padding: 0; }}")

    def _pick(self) -> None:
        c = pick_color(self, self._color, self._title)
        if c and c.upper() != self._color:
            self.set_color(c)
            self.committed.emit(self._color)


def gain_to_db(v: float) -> float:
    return DB_FLOOR if v <= 10 ** (DB_FLOOR / 20) else round(20 * math.log10(v), 2)


def db_to_gain(db: float) -> float:
    return 0.0 if db <= DB_FLOOR + 1e-9 else 10 ** (db / 20)


DB_FLOOR = -60.0
FIT_CHOICES = (("cover", "裁切填滿"), ("contain", "完整顯示（黑邊）"), ("blur", "完整顯示（模糊背景）"))


# ============================================================================ 片段
class ClipTab(QScrollArea):
    """片段: properties of the selected clip(s); a multi-selection shows only the fields they share."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.RichText)
        self.empty = dim_label("在時間軸上點選片段，就能在這裡調整它的屬性。")
        lay.addWidget(self.title)
        lay.addWidget(self.empty)
        self.groups = [self._build_time(), self._build_play(), self._build_audio(), self._build_picture(),
                       self._build_look()]
        for g in self.groups:
            lay.addWidget(g)
        lay.addStretch(1)
        state.edited.connect(self._on_edited)
        for sig in (state.selection_changed, state.sequence_changed, state.project_replaced):
            sig.connect(self.refresh)
        state.playhead_changed.connect(self._on_playhead)
        self.refresh()

    # ---------------------------------------------------------------- building
    def _build_time(self) -> QGroupBox:
        self.g_time = g = QGroupBox("時間")
        f = form(g)
        self.f_start, self.f_dur, self.f_in, self.f_out = (TimeField() for _ in range(4))
        f.addRow("開始", self.f_start)
        f.addRow("長度", self.f_dur)
        f.addRow("入點", self.f_in)
        f.addRow("出點", self.f_out)
        self.f_start.setToolTip("片段在時間軸上的起點（位置被占用時不會覆蓋別的片段）")
        self.f_in.setToolTip("素材的入點；等同在時間軸拖曳片段左緣（片段結尾不動）")
        self.f_out.setToolTip("素材的出點；等同在時間軸拖曳片段右緣")
        self.f_start.timeCommitted.connect(self._set_start)
        self.f_dur.timeCommitted.connect(self._set_duration)
        self.f_in.timeCommitted.connect(self._set_in)
        self.f_out.timeCommitted.connect(self._set_out)
        return g

    def _build_play(self) -> QGroupBox:
        self.g_play = g = QGroupBox("播放")
        self.f_play = f = form(g)
        self.f_speed = NumField(0.25, 4.0, 0.05, 2, " ×", default=1.0)
        self.f_fade_in = NumField(0.0, 30.0, 0.1, 2, " 秒")
        self.f_fade_out = NumField(0.0, 30.0, 0.1, 2, " 秒")
        self.f_xfade = NumField(0.0, 30.0, 0.1, 2, " 秒")
        self.f_speed.setToolTip("播放速度（後面的片段會跟著前後移動）")
        f.addRow("速度", self.f_speed)
        f.addRow("淡入", self.f_fade_in)
        f.addRow("淡出", self.f_fade_out)
        f.addRow("交叉淡化", self.f_xfade)
        self.f_speed.committed.connect(lambda v: self._commit("修改速度", "set_speed", speed=v))
        self.f_fade_in.committed.connect(lambda v: self._commit("修改淡入", "set_clip_props", fade_in=v))
        self.f_fade_out.committed.connect(lambda v: self._commit("修改淡出", "set_clip_props", fade_out=v))
        self.f_xfade.committed.connect(lambda v: self._commit("修改交叉淡化", "set_crossfade", dur=v))
        return g

    def _build_audio(self) -> QGroupBox:
        self.g_audio = g = QGroupBox("聲音")
        f = form(g)
        self.f_vol = NumField(DB_FLOOR, 12.0, 0.5, 1, " dB", default=0.0)
        self.f_vol.setToolTip("片段音量（0 dB = 原音量，-60 dB = 靜音）")
        f.addRow("音量", self.f_vol)
        self.f_vol.committed.connect(lambda db: self._commit("修改音量", "set_clip_props", volume=db_to_gain(db)))
        return g

    def _build_picture(self) -> QGroupBox:
        self.g_picture = g = QGroupBox("畫面與取景")
        self.f_pic = f = form(g)
        self.f_fit = FocusCombo()
        for value, label in FIT_CHOICES:
            self.f_fit.addItem(label, value)
        self.f_fit.setPlaceholderText("—")
        self.f_zoom = SliderField(1.0, 4.0, 0.01, 2, 100, " ×", box_hi=8.0, default=1.0)
        self.f_cx = SliderField(0.0, 1.0, 0.005, 3, 1000, default=0.5)
        self.f_cy = SliderField(0.0, 1.0, 0.005, 3, 1000, default=0.5)
        self.f_flip = QCheckBox("水平翻轉（鏡像）")
        # the global theme styles no 「indeterminate」 indicator, so a mixed multi-selection looked just like 「unchecked」
        self.f_flip.setStyleSheet(
            f"QCheckBox::indicator:indeterminate {{ border-color: {theme.ACCENT}; background: qlineargradient("
            f"x1:0, y1:0, x2:0, y2:1, stop:0 {theme.PANEL_ALT}, stop:0.38 {theme.PANEL_ALT}, stop:0.39 {theme.ACCENT}, "
            f"stop:0.61 {theme.ACCENT}, stop:0.62 {theme.PANEL_ALT}, stop:1 {theme.PANEL_ALT}); }}")
        self.b_key_add = QPushButton("加入取景關鍵格")
        self.b_key_clear = QPushButton("清除關鍵格")
        self.keys_row = hbox(self.b_key_add, self.b_key_clear, None)
        self.keys_label = dim_label()
        f.addRow("填滿方式", self.f_fit)
        f.addRow("縮放", self.f_zoom)
        f.addRow("取景 X", self.f_cx)
        f.addRow("取景 Y", self.f_cy)
        f.addRow("", self.f_flip)
        f.addRow("", self.keys_row)
        f.addRow("", self.keys_label)
        self.f_cx.setToolTip("取景框中心的水平位置（0 = 最左，1 = 最右）")
        self.f_cy.setToolTip("取景框中心的垂直位置（0 = 最上，1 = 最下）")
        self.f_fit.activated.connect(self._set_fit)
        self.f_zoom.committed.connect(lambda v: self._commit("修改縮放", "set_clip_props", zoom=v))
        self.f_cx.committed.connect(lambda v: self._set_center("cx", v))
        self.f_cy.committed.connect(lambda v: self._set_center("cy", v))
        self.f_flip.clicked.connect(self._set_flip)
        self.b_key_add.clicked.connect(self._add_key)
        self.b_key_clear.clicked.connect(self._clear_keys)
        return g

    def _build_look(self) -> QGroupBox:
        self.g_look = g = QGroupBox("色彩")
        f = form(g)
        self.f_bright = SliderField(-1.0, 1.0, 0.01, 2, 100, default=0.0)
        self.f_contrast = SliderField(0.0, 3.0, 0.01, 2, 100, default=1.0)
        self.f_sat = SliderField(0.0, 3.0, 0.01, 2, 100, default=1.0)
        f.addRow("亮度", self.f_bright)
        f.addRow("對比", self.f_contrast)
        f.addRow("飽和度", self.f_sat)
        self.f_bright.committed.connect(lambda v: self._commit("修改亮度", "set_clip_props", brightness=v))
        self.f_contrast.committed.connect(lambda v: self._commit("修改對比", "set_clip_props", contrast=v))
        self.f_sat.committed.connect(lambda v: self._commit("修改飽和度", "set_clip_props", saturation=v))
        return g

    # ---------------------------------------------------------------- reading the selection
    def _selected(self) -> list[tuple[Track, Clip]]:
        if not self.state.selection:
            return []
        try:
            seq = self.state.sequence()
        except KeyError:
            return []
        sel = [(t, c) for t, c in seq.all_clips() if c.id in self.state.selection]
        return sorted(sel, key=lambda tc: (tc[1].start, tc[1].id))

    def _single(self) -> Optional[Clip]:
        sel = self._selected()
        return sel[0][1] if len(sel) == 1 else None

    def _has_audio(self, c: Clip) -> bool:
        a = self.state.project.assets.get(c.asset_id)
        return bool(a and a.has_audio and a.kind != "image")

    def _local_time(self, c: Clip) -> tuple[float, bool]:
        """(clip-local playhead time clamped to the clip, whether the playhead is on the clip)."""
        t = self.state.playhead - c.start
        tol = 0.5 / max(1.0, self.state.fps)
        return min(max(t, 0.0), c.duration), -tol <= t <= c.duration + tol

    @staticmethod
    def _previous(track: Track, clip: Clip) -> Optional[Clip]:
        prevs = [o for o in track.clips if o is not clip and o.start < clip.start - 1e-9]
        return max(prevs, key=lambda o: o.start) if prevs else None

    # ---------------------------------------------------------------- refresh (never commits)
    def _on_edited(self, _kind: str, _label: str) -> None:
        self.refresh()

    def _on_playhead(self, _t: float) -> None:
        sel = self._selected()
        if len(sel) == 1 and sel[0][0].kind == VIDEO:
            self._fill_center([sel[0][1]])

    def refresh(self) -> None:
        sel = self._selected()
        self.empty.setVisible(not sel)
        for g in self.groups:
            g.setVisible(bool(sel))
        if not sel:
            self.title.setText("")
            return
        clips = [c for _, c in sel]
        video = all(t.kind == VIDEO for t, _ in sel)
        self.title.setText(self._title_text(sel))
        self.g_time.setVisible(len(clips) == 1)
        self.g_audio.setVisible(all(self._has_audio(c) for c in clips))
        self.g_picture.setVisible(video)
        self.g_look.setVisible(video)
        if len(clips) == 1:
            self._fill_time(clips[0])
        self._fill_play(sel)
        self.f_vol.set_value(common([gain_to_db(c.volume) for c in clips]))
        if video:
            self._fill_picture(clips)
            self.f_bright.set_value(common([c.look.brightness for c in clips]))
            self.f_contrast.set_value(common([c.look.contrast for c in clips]))
            self.f_sat.set_value(common([c.look.saturation for c in clips]))

    def _title_text(self, sel: list[tuple[Track, Clip]]) -> str:
        dim = theme.TEXT_DIM
        if len(sel) > 1:
            return (f"已選取 <b>{len(sel)}</b> 個片段<br><span style='color:{dim}'>"
                    "只顯示可以一起修改的欄位；「—」表示各片段的數值不同。</span>")
        track, c = sel[0]
        a = self.state.project.assets.get(c.asset_id)
        name = c.label or (a.name if a else "（找不到素材）")
        return f"<b>{html.escape(name)}</b>&nbsp;&nbsp;<span style='color:{dim}'>{html.escape(track.name)} 軌</span>"

    def _fill_time(self, c: Clip) -> None:
        fps = self.state.fps
        for f in (self.f_start, self.f_dur, self.f_in, self.f_out):
            f.set_fps(fps)
        self.f_start.set_time(c.start)
        self.f_dur.set_time(c.duration)
        self.f_in.set_time(c.src_in)
        self.f_out.set_time(c.src_out)

    def _fill_play(self, sel: list[tuple[Track, Clip]]) -> None:
        clips = [c for _, c in sel]
        self.f_speed.set_value(common([c.speed for c in clips]))
        self.f_fade_in.set_value(common([c.fade_in for c in clips]))
        self.f_fade_out.set_value(common([c.fade_out for c in clips]))
        self.f_play.setRowVisible(self.f_xfade, len(sel) == 1)
        if len(sel) == 1:
            prev = self._previous(*sel[0])
            self.f_xfade.setEnabled(prev is not None)
            self.f_xfade.setToolTip("與同軌前一個片段交疊的長度（0 = 硬切）" if prev else "同一軌前面沒有片段，無法交叉淡化")
            self.f_xfade.set_value(max(0.0, prev.end - clips[0].start) if prev else 0.0)

    def _fill_picture(self, clips: list[Clip]) -> None:
        fit = common([c.transform.fit for c in clips])
        with quiet(self.f_fit):
            self.f_fit.setCurrentIndex(self.f_fit.findData(fit) if fit else -1)
        self.f_zoom.set_value(common([c.transform.zoom for c in clips]))
        flip = common([c.transform.flip_h for c in clips])
        with quiet(self.f_flip):
            self.f_flip.setTristate(flip is None)
            self.f_flip.setCheckState(Qt.PartiallyChecked if flip is None else Qt.Checked if flip else Qt.Unchecked)
        self._fill_center(clips)

    def _fill_center(self, clips: list[Clip]) -> None:
        single = clips[0] if len(clips) == 1 else None
        keyed_multi = single is None and any(c.transform.keys for c in clips)
        for w in (self.f_cx, self.f_cy):
            self.f_pic.setRowVisible(w, not keyed_multi)
        for w in (self.keys_row, self.keys_label):
            self.f_pic.setRowVisible(w, single is not None)
        if keyed_multi:
            return
        if single is None:
            self.f_cx.set_value(common([c.transform.cx for c in clips]))
            self.f_cy.set_value(common([c.transform.cy for c in clips]))
            return
        t_local, inside = self._local_time(single)
        keys = single.transform.keys
        cx, cy = single.transform.center_at(t_local)
        self.f_cx.set_value(cx)
        self.f_cy.set_value(cy)
        for w in (self.f_cx, self.f_cy):
            w.setEnabled(inside or not keys)
        self.b_key_add.setEnabled(inside)
        self.b_key_clear.setEnabled(bool(keys))
        if not keys:
            text = "取景固定（沒有關鍵格）。把播放頭移到片段上再按「加入取景關鍵格」，取景就能隨時間移動。"
        else:
            text = f"{len(keys)} 個關鍵格：X / Y 顯示播放頭位置的取景，修改時會在播放頭加入或更新關鍵格。"
            if not inside:
                text += "（播放頭不在這個片段上）"
        self.keys_label.setText(text)

    # ---------------------------------------------------------------- commits (one undo step each)
    def _commit(self, label: str, name: str, **kw: Any) -> None:
        """``ops.<name>(project, seq, clip_id, **kw)`` for every selected clip as ONE undo step."""
        ids = [c.id for _, c in self._selected()]
        if ids:
            try:
                if len(ids) == 1:
                    self.state.op(label, name, ids[0], **kw)
                else:
                    fn, seq_id = getattr(ops, name), self.state.seq_id
                    self.state.do(label, lambda p: [fn(p, p.sequence(seq_id), cid, **kw) for cid in ids])
            except OpError:
                pass                              # AppState already put the reason on the status line
        self.refresh()                            # show what was really applied (ops clamp values) or revert

    def _set_start(self, t: float) -> None:
        self._commit("移動片段", "move_clip", new_start=t, resolve="reject")

    def _set_duration(self, d: float) -> None:
        c = self._single()
        if c is not None:
            self._commit("修改片段長度", "trim_clip_end", new_end=c.start + d)

    def _set_in(self, t: float) -> None:
        c = self._single()
        if c is not None:
            self._commit("修改入點", "trim_clip_start", new_start=c.start + (t - c.src_in) / c.speed)

    def _set_out(self, t: float) -> None:
        c = self._single()
        if c is not None:
            self._commit("修改出點", "trim_clip_end", new_end=c.start + (t - c.src_in) / c.speed)

    def _set_fit(self, index: int) -> None:
        fit = self.f_fit.itemData(index)
        if fit:
            self._commit("修改填滿方式", "set_clip_props", fit=fit)

    def _set_flip(self) -> None:
        self.f_flip.setTristate(False)
        self._commit("水平翻轉", "set_clip_props", flip_h=self.f_flip.isChecked())

    def _set_center(self, axis: str, v: float) -> None:
        c = self._single()
        if c is None or not c.transform.keys:
            self._commit("修改取景中心", "set_clip_props", **{axis: v})
            return
        t_local, inside = self._local_time(c)
        if not inside:
            self.refresh()
            return
        cx, cy = c.transform.center_at(t_local)
        cx, cy = (v, cy) if axis == "cx" else (cx, v)
        self._commit("修改取景關鍵格", "set_transform_key", t_local=t_local, cx=cx, cy=cy)

    def _add_key(self) -> None:
        c = self._single()
        if c is None:
            return
        t_local, inside = self._local_time(c)
        if not inside:
            self.state.status.emit("播放頭不在這個片段上，無法加入關鍵格")
            return
        cx, cy = c.transform.center_at(t_local)
        self._commit("加入取景關鍵格", "set_transform_key", t_local=t_local, cx=cx, cy=cy)

    def _clear_keys(self) -> None:
        if self._single() is not None:
            self._commit("清除取景關鍵格", "clear_transform_keys")


# ============================================================================ 字幕
POSITIONS = (("bottom", "下方"), ("center", "中間"), ("top", "上方"))


class CaptionsTab(QWidget):
    """字幕: editable cue table, cue tools, import / export, style editor and optional speech-to-text."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.runner: Optional[TaskRunner] = None
        self._run_ctx: tuple[Optional[Project], str, int] = (None, "", 0)
        self._sig: Optional[tuple] = None
        self._loading = False
        self._from_table = False
        lay = QVBoxLayout(self)
        lay.addWidget(self._build_tools())
        lay.addWidget(self._build_progress())
        lay.addWidget(self._build_table(), 1)
        lay.addWidget(self._build_style())
        state.edited.connect(self._on_edited)
        state.sequence_changed.connect(self.refresh)
        state.project_replaced.connect(self.refresh)
        state.cue_selection_changed.connect(self._on_state_cues)
        self.refresh()

    # ---------------------------------------------------------------- building
    def _button(self, text: str, tip: str, slot, row: QHBoxLayout) -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        row.addWidget(b)
        return b

    def _build_tools(self) -> QWidget:
        w = QWidget()
        box = QVBoxLayout(w)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        self.b_add = self._button("新增", "在播放頭新增一則字幕", self._add, row1)
        self.b_split = self._button("分割", "在播放頭把字幕切成兩則", self._split, row1)
        self.b_merge = self._button("合併", "把選取的字幕合併成一則", self._merge, row1)
        self.b_delete = self._button("刪除", "刪除選取的字幕（Delete）", self._delete, row1)
        self.b_shift = self._button("全部位移…", "把全部字幕提早或延後", self._shift, row1)
        self.b_import = self._button("匯入…", "匯入 SRT / ASS / VTT 字幕檔", self._import, row2)
        self.b_export = self._button("匯出 SRT…", "把字幕存成 SRT 檔", self._export, row2)
        self.b_auto = self._button("自動產生字幕", "用語音辨識產生字幕（選用功能，全部在本機執行）", self._auto, row2)
        box.addLayout(row1)
        box.addLayout(row2)
        return w

    def _build_progress(self) -> QWidget:
        self.progress_row = w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        self.progress_label = dim_label()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.b_cancel = QPushButton("取消")
        self.b_cancel.clicked.connect(self._cancel)
        v.addWidget(self.progress_label)
        v.addWidget(hbox(self.progress, self.b_cancel))
        w.setVisible(False)
        return w

    def _build_table(self) -> QTableWidget:
        self.table = t = QTableWidget(0, 3)
        t.setHorizontalHeaderLabels(["開始", "結束", "文字"])
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        t.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                          | QAbstractItemView.SelectedClicked)
        t.setAlternatingRowColors(True)
        t.setWordWrap(True)
        t.itemChanged.connect(self._on_item_changed)
        t.itemSelectionChanged.connect(self._on_table_selection)
        sc = QShortcut(QKeySequence(QKeySequence.Delete), t)
        sc.setContext(Qt.WidgetShortcut)
        sc.activated.connect(self._delete)
        return t

    def _build_style(self) -> QGroupBox:
        g = QGroupBox("字幕樣式（整個序列）")
        f = form(g)
        self.s_font = FocusFontCombo()
        self.s_size = NumField(1.0, 15.0, 0.1, 1, " %", default=4.5)
        self.s_size.setToolTip("字的高度，占畫面高度的百分比")
        self.s_bold = QCheckBox("粗體")
        self.s_color = ColorButton("文字顏色")
        self.s_outline_color = ColorButton("描邊顏色")
        self.s_outline = NumField(0.0, 2.0, 0.05, 2, " %")
        self.s_outline.setToolTip("描邊粗細，占畫面高度的百分比（0 = 不描邊）")
        self.s_pos = FocusCombo()
        for value, label in POSITIONS:
            self.s_pos.addItem(label, value)
        self.s_margin = NumField(0.0, 45.0, 0.5, 1, " %")
        self.s_margin.setToolTip("字幕離畫面邊緣的距離，占畫面高度的百分比（短影音底部約 20% 常被平台介面遮住）")
        self.s_chars = NumField(4, 60, 1, 0, " 字", default=16)
        self.s_box = QCheckBox("底框")
        self.s_box_color = ColorButton("底框顏色")
        self.s_box_opacity = NumField(0, 100, 5, 0, " %", default=50)
        self.s_box_opacity.setToolTip("底框不透明度")
        f.addRow("字型", self.s_font)
        f.addRow("大小", hbox(self.s_size, self.s_bold, None))
        f.addRow("顏色", hbox(QLabel("文字"), self.s_color, QLabel("描邊"), self.s_outline_color, None))
        f.addRow("描邊", self.s_outline)
        f.addRow("位置", hbox(self.s_pos, QLabel("邊距"), self.s_margin, None))
        f.addRow("每行字數", self.s_chars)
        f.addRow("底框", hbox(self.s_box, self.s_box_color, self.s_box_opacity, None))
        self.s_font.activated.connect(lambda _i: self._style(font=self.s_font.currentFont().family()))
        self.s_size.committed.connect(lambda v: self._style(size=round(v / 100, 5)))
        self.s_bold.clicked.connect(lambda on: self._style(bold=bool(on)))
        self.s_color.committed.connect(lambda c: self._style(color=c))
        self.s_outline_color.committed.connect(lambda c: self._style(outline_color=c))
        self.s_outline.committed.connect(lambda v: self._style(outline=round(v / 100, 5)))
        self.s_pos.activated.connect(lambda i: self._style(position=self.s_pos.itemData(i)))
        self.s_margin.committed.connect(lambda v: self._style(margin=round(v / 100, 5)))
        self.s_chars.committed.connect(lambda v: self._style(max_chars=int(round(v))))
        self.s_box.clicked.connect(lambda on: self._style(box=bool(on)))
        self.s_box_color.committed.connect(lambda c: self._style(box_color=c))
        self.s_box_opacity.committed.connect(lambda v: self._style(box_opacity=round(v / 100, 4)))
        self.g_style = g
        return g

    # ---------------------------------------------------------------- refresh
    def _seq(self) -> Optional[Sequence]:
        try:
            return self.state.sequence()
        except KeyError:
            return None

    def _on_edited(self, _kind: str, _label: str) -> None:
        self.refresh()

    def refresh(self) -> None:
        seq = self._seq()
        cues = list(seq.cues) if seq else []
        sig = (seq.id if seq else "", tuple((q.id, q.start, q.end, q.text) for q in cues))
        if sig != self._sig:
            self._sig = sig
            self._fill_table(cues)
        self.g_style.setEnabled(seq is not None)
        if seq is not None:
            self._fill_style(seq)
        self._update_buttons()

    def _fill_table(self, cues: list) -> None:
        self._loading = True
        try:
            with quiet(self.table):
                self.table.setRowCount(len(cues))
                for r, q in enumerate(cues):
                    for col, text in enumerate((fmt_clock(q.start), fmt_clock(q.end), q.text)):
                        it = QTableWidgetItem(text)
                        if col == 0:
                            it.setData(Qt.UserRole, q.id)
                        if col < 2:
                            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        self.table.setItem(r, col, it)
                self.table.resizeRowsToContents()
            self._sync_from_state()
        finally:
            self._loading = False

    def _fill_style(self, seq: Sequence) -> None:
        st = seq.caption_style
        with quiet(self.s_font, self.s_bold, self.s_box, self.s_pos):
            self.s_font.setCurrentFont(QFont(st.font))
            self.s_bold.setChecked(bool(st.bold))
            self.s_box.setChecked(bool(st.box))
            self.s_pos.setCurrentIndex(max(0, self.s_pos.findData(st.position)))
        self.s_size.set_value(st.size * 100)
        self.s_outline.set_value(st.outline * 100)
        self.s_margin.set_value(st.margin * 100)
        self.s_chars.set_value(st.max_chars)
        self.s_box_opacity.set_value(st.box_opacity * 100)
        self.s_color.set_color(st.color)
        self.s_outline_color.set_color(st.outline_color)
        self.s_box_color.set_color(st.box_color)

    def _update_buttons(self) -> None:
        seq = self._seq()
        cue_ids = {q.id for q in seq.cues} if seq else set()
        n = len(self.state.cue_selection & cue_ids)
        busy = self.runner is not None and self.runner.running
        self.b_add.setEnabled(seq is not None)
        self.b_split.setEnabled(bool(cue_ids))
        self.b_merge.setEnabled(n >= 2)
        self.b_delete.setEnabled(n >= 1)
        self.b_shift.setEnabled(bool(cue_ids))
        self.b_export.setEnabled(bool(cue_ids))
        self.b_import.setEnabled(seq is not None and not busy)
        self.b_auto.setEnabled(seq is not None and not busy)

    # ---------------------------------------------------------------- selection sync (both ways, no echo)
    def _cue_id(self, row: int) -> Optional[str]:
        it = self.table.item(row, 0)
        return it.data(Qt.UserRole) if it is not None else None

    def _row_of(self, cue_id: str) -> int:
        return next((r for r in range(self.table.rowCount()) if self._cue_id(r) == cue_id), -1)

    def selected_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        return [cid for cid in (self._cue_id(r) for r in rows) if cid]

    def _on_table_selection(self) -> None:
        if self._loading:
            return
        self._from_table = True
        try:
            self.state.select_cues(self.selected_ids())
        finally:
            self._from_table = False
        self._update_buttons()

    def _on_state_cues(self) -> None:
        if not self._from_table:
            self._sync_from_state()
        self._update_buttons()

    def _sync_from_state(self) -> None:
        model = self.table.model()
        selection = QItemSelection()
        first = -1
        for r in range(self.table.rowCount()):
            if self._cue_id(r) in self.state.cue_selection:
                selection.select(model.index(r, 0), model.index(r, 2))
                first = r if first < 0 else first
        was = self._loading
        self._loading = True
        try:
            with quiet(self.table):
                self.table.selectionModel().select(selection, QItemSelectionModel.ClearAndSelect)
            if first >= 0:
                self.table.scrollToItem(self.table.item(first, 0))
        finally:
            self._loading = was
        self.table.viewport().update()

    # ---------------------------------------------------------------- cell edits (one op per cell)
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        cue_id, col, text = self._cue_id(item.row()), item.column(), item.text()
        if cue_id:   # deferred: never rebuild the table inside the view's own commit
            QTimer.singleShot(0, lambda: self._commit_cell(cue_id, col, text))

    def _editing_cell(self) -> Optional[tuple[str, int]]:
        """(cue id, column) of the cell whose editor is open; Tab / Shift+Tab opens the next cell's editor right
        after a commit, before this class got to refresh the table."""
        if self.table.state() != QAbstractItemView.EditingState:
            return None
        cue_id = self._cue_id(self.table.currentRow())
        return (cue_id, self.table.currentColumn()) if cue_id else None

    def _follow_editor(self, editing: Optional[tuple[str, int]]) -> None:
        """The rows are kept in time order, so a commit can move the cue that is being edited to another row; an editor
        that stayed on the old row index would then edit a different cue.  Re-open it on the cue itself."""
        if editing is None:
            return
        cue_id, col = editing
        row = self._row_of(cue_id)
        if row >= 0 and row != self.table.currentRow():
            self.table.setCurrentCell(row, col)
            self.table.editItem(self.table.item(row, col))

    def _commit_cell(self, cue_id: str, col: int, text: str) -> None:
        editing = self._editing_cell()
        if col == 2:
            self._op("編輯字幕", "update_cue", cue_id, text=text)
        else:
            try:
                t = parse_timecode(text, self.state.fps)
                if not math.isfinite(t):                  # "nan" / "1e999" are not times
                    raise ValueError(text)
                t = max(0.0, t)
            except (ValueError, TypeError, OverflowError):
                self.state.status.emit("時間格式不正確（例如 1:23.500 或 83.5）")
            else:
                self._op("修改字幕時間", "update_cue", cue_id, **{"start" if col == 0 else "end": t})
        self._sig = None
        self.refresh()
        self._follow_editor(editing)

    def _op(self, label: str, name: str, *args: Any, **kw: Any) -> Any:
        try:
            return self.state.op(label, name, *args, **kw)
        except OpError:
            return None

    def _style(self, **kw: Any) -> None:
        self._op("字幕樣式", "set_caption_style", **kw)
        self.refresh()

    # ---------------------------------------------------------------- tools
    def _add(self) -> None:
        seq = self._seq()
        if seq is None:
            return
        t = self.state.playhead
        later = [q.start for q in seq.cues if q.start > t + 1e-6]
        end = t + 2.0
        if later and min(later) - t >= 0.5:
            end = min(end, min(later))
        q = self._op("新增字幕", "add_cue", t, end, "新字幕")
        if q is None:
            return
        self.state.select_cues([q.id])
        row = self._row_of(q.id)
        if row >= 0:
            self.table.setCurrentCell(row, 2)
            self.table.editItem(self.table.item(row, 2))

    def _split(self) -> None:
        seq = self._seq()
        if seq is None:
            return
        t, chosen = self.state.playhead, set(self.selected_ids())
        under = [q for q in seq.cues if q.start < t < q.end]
        target = next((q for q in under if q.id in chosen), under[0] if under else None)
        if target is None:
            self.state.status.emit("播放頭沒有落在任何字幕上")
            return
        self._op("分割字幕", "split_cue", target.id, t)

    def _merge(self) -> None:
        ids = self.selected_ids()
        if len(ids) < 2:
            self.state.status.emit("請先選取至少兩則字幕")
            return
        self._op("合併字幕", "merge_cues", ids)

    def _delete(self) -> None:
        ids = self.selected_ids()
        if ids and self._op("刪除字幕", "remove_cues", ids):
            self.state.status.emit(f"已刪除 {len(ids)} 則字幕")

    def _shift(self) -> None:
        delta = ask_number(self, "全部位移", "全部字幕要位移幾秒？（負數＝提早）", 0.0)
        if delta:
            self._op("位移字幕", "shift_cues", delta)

    def _import(self) -> None:
        seq = self._seq()
        path = open_file(self, "匯入字幕", "字幕檔 (*.srt *.ass *.vtt);;所有檔案 (*)") if seq else ""
        if not path:
            return
        try:
            items = load_subtitle_file(path)
        except (OSError, UnicodeError, ValueError) as exc:
            inform(self, "匯入字幕", f"無法讀取字幕檔：{exc}", warning=True)
            return
        if not items:
            inform(self, "匯入字幕", "這個檔案裡沒有可用的字幕。")
            return
        if seq.cues and not ask(self, "匯入字幕", f"目前已有 {len(seq.cues)} 則字幕。\n"
                                f"要用檔案中的 {len(items)} 則字幕取代嗎？", "取代", "取消"):
            return
        if self._op("匯入字幕", "replace_cues", items) is not None:
            self.state.status.emit(f"已匯入 {len(items)} 則字幕：{os.path.basename(path)}")

    def _export(self) -> None:
        seq = self._seq()
        if seq is None or not seq.cues:
            return
        path = save_file(self, "匯出 SRT", f"{seq.name}.srt", "SRT 字幕 (*.srt)")
        if not path:
            return
        if not path.lower().endswith(".srt"):
            path += ".srt"
        try:
            atomic_write_text(path, to_srt(seq.cues))
        except OSError as exc:
            inform(self, "匯出 SRT", f"無法寫入檔案：{exc}", warning=True)
            return
        self.state.status.emit(f"已匯出 {len(seq.cues)} 則字幕：{os.path.basename(path)}")

    # ---------------------------------------------------------------- 自動產生字幕
    def _auto(self) -> None:
        """Check speech recognition, ask before any model download, then transcribe a snapshot in the background."""
        seq = self._seq()
        if seq is None or (self.runner is not None and self.runner.running):
            return
        try:
            speech = importlib.import_module("autocut.analysis.speech")
            ok, reason = speech.speech_available()
        except ImportError:
            ok, reason = False, "語音辨識模組無法載入。"
        if not ok:
            inform(self, "自動產生字幕", with_install_hint(reason))
            return
        allow_download = False
        if not speech.model_cached(SPEECH_MODEL):
            if not ask(self, "下載語音模型", f"第一次使用需要下載語音辨識模型「{SPEECH_MODEL}」（約數百 MB，只需下載一次，"
                       "之後可離線使用）。\n素材不會上傳，辨識全部在這台電腦上進行。\n\n要現在下載嗎？", "下載並繼續", "取消"):
                self.state.status.emit("已取消自動產生字幕")
                return
            allow_download = True
        if seq.cues and not ask(self, "自動產生字幕", f"目前已有 {len(seq.cues)} 則字幕，產生完成後會取代它們。\n要繼續嗎？",
                                "繼續", "取消"):
            return
        self._start_auto(allow_download)

    def _start_auto(self, allow_download: bool) -> None:
        try:
            captions = importlib.import_module("autocut.autoedit.captions")
        except ImportError:
            inform(self, "自動產生字幕", "自動字幕功能尚未安裝完整（找不到 autoedit.captions）。", warning=True)
            return
        project, seq_id = self.state.project, self.state.seq_id
        snap = Project.from_dict(project.to_dict())             # the worker never sees the live project
        snap_seq = snap.sequence(seq_id)
        self._run_ctx = (project, seq_id, project.revision)

        def work(progress, cancel):
            return captions.generate_cues_for_sequence(
                snap, snap_seq, progress, cancel, language=SPEECH_LANGUAGE, traditional=True,
                model_size=SPEECH_MODEL, allow_download=allow_download)

        self.runner = r = TaskRunner(work, None, name="auto-captions")
        r.progress.connect(self._on_progress)
        r.finished.connect(self._on_cues)
        r.failed.connect(self._on_failed)
        r.cancelled.connect(lambda: self.state.status.emit("已取消自動產生字幕"))
        r.done.connect(self._on_done)
        self.progress.setValue(0)
        self.progress_label.setText("正在準備語音辨識…")
        self.b_cancel.setEnabled(True)
        self.b_cancel.setText("取消")
        self.progress_row.setVisible(True)
        r.start()
        self._update_buttons()

    def _on_progress(self, frac: float, message: str) -> None:
        self.progress.setValue(int(frac * 1000))
        if message:
            self.progress_label.setText(message)

    def _cancel(self) -> None:
        if self.runner is not None and self.runner.running:
            self.runner.cancel()
            self.b_cancel.setEnabled(False)
            self.b_cancel.setText("正在取消…")

    def _on_cues(self, cues: Any) -> None:
        project, seq_id, revision = self._run_ctx
        cues = list(cues or [])
        live = self.state.project
        if live is not project or not any(s.id == seq_id for s in live.sequences):
            inform(self, "自動產生字幕", "辨識期間專案或序列已經換掉，這次的結果沒有套用。")
            return
        if not cues:
            inform(self, "自動產生字幕", "沒有辨識到任何語音，字幕沒有變更。")
            return
        if live.revision != revision and not ask(
                self, "自動產生字幕", "辨識期間專案有被修改過，字幕時間是依照開始辨識時的剪輯計算的，可能會對不上。\n仍要套用嗎？",
                "仍要套用", "放棄"):
            return
        try:
            self.state.do("自動字幕", lambda p: ops.replace_cues(p, p.sequence(seq_id), cues))
        except OpError:
            return
        self.state.status.emit(f"已產生 {len(cues)} 則字幕（Ctrl+Z 可復原）")

    def _on_failed(self, message: str) -> None:
        speechy = "語音" in message or "Speech" in message
        inform(self, "自動產生字幕", with_install_hint(message) if speechy else message, warning=True)

    def _on_done(self) -> None:
        self.progress_row.setVisible(False)
        self._update_buttons()


# ============================================================================ 序列
SIZE_PRESETS = (("16:9 橫式（1920×1080）", 1920, 1080), ("9:16 直式（1080×1920）", 1080, 1920),
                ("1:1 方形（1080×1080）", 1080, 1080), ("4:5 直式（1080×1350）", 1080, 1350))
FPS_CHOICES = (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)


def set_background(p: Project, seq_id: str, color: str) -> None:
    """Background colour of a sequence (runs inside ``state.do``)."""
    ops.set_sequence_format(p, p.sequence(seq_id), background=color)


class SequenceTab(QWidget):
    """序列: name, frame size, frame rate and background colour of the active sequence."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self._name_typed = False
        lay = QVBoxLayout(self)
        g = QGroupBox("序列設定")
        f = form(g)
        self.f_name = QLineEdit()
        self.f_name.setMaxLength(80)
        self.f_size = FocusCombo()
        for label, w, h in SIZE_PRESETS:
            self.f_size.addItem(label, (w, h))
        self.f_size.addItem("自訂", None)
        self.f_w = NumField(16, 8192, 2, 0, " px", default=1920)
        self.f_h = NumField(16, 8192, 2, 0, " px", default=1080)
        self.f_fps = FocusCombo()
        self.f_bg = ColorButton("背景色")
        f.addRow("名稱", self.f_name)
        f.addRow("尺寸", self.f_size)
        f.addRow("寬 × 高", hbox(self.f_w, QLabel("×"), self.f_h, None))
        f.addRow("影格率", self.f_fps)
        f.addRow("背景色", hbox(self.f_bg, dim_label("沒有畫面的地方顯示這個顏色"), None))
        self.info = dim_label()
        lay.addWidget(g)
        lay.addWidget(self.info)
        lay.addWidget(dim_label("多個序列的新增、複製、刪除在時間軸上方的分頁。"))
        lay.addStretch(1)
        self.f_name.textEdited.connect(self._on_name_typed)
        self.f_name.editingFinished.connect(self._rename)
        self.f_size.activated.connect(self._set_preset)
        self.f_w.committed.connect(lambda v: self._format("變更序列尺寸", width=int(v)))
        self.f_h.committed.connect(lambda v: self._format("變更序列尺寸", height=int(v)))
        self.f_fps.activated.connect(self._set_fps)
        self.f_bg.committed.connect(self._set_background)
        state.edited.connect(self._on_edited)
        state.sequence_changed.connect(self.refresh)
        state.project_replaced.connect(self.refresh)
        self.refresh()

    def _seq(self) -> Optional[Sequence]:
        try:
            return self.state.sequence()
        except KeyError:
            return None

    def _on_edited(self, _kind: str, _label: str) -> None:
        self.refresh()

    def refresh(self) -> None:
        seq = self._seq()
        self.setEnabled(seq is not None)
        if seq is None:
            return
        if not (self.f_name.hasFocus() and self._name_typed):
            self.f_name.setText(seq.name)
        idx = next((i for i, (_l, w, h) in enumerate(SIZE_PRESETS) if (w, h) == (seq.width, seq.height)),
                   len(SIZE_PRESETS))
        with quiet(self.f_size):
            self.f_size.setCurrentIndex(idx)
        self.f_w.set_value(seq.width)
        self.f_h.set_value(seq.height)
        self._fill_fps(seq.fps)
        self.f_bg.set_color(seq.background)
        clips = sum(len(t.clips) for t in seq.tracks())
        self.info.setText(f"長度 {human_duration(seq.duration)}　·　{clips} 個片段　·　{len(seq.cues)} 則字幕\n"
                          f"{len(seq.video_tracks)} 條視訊軌 / {len(seq.audio_tracks)} 條音訊軌　·　比例 {seq.aspect_label}")

    def _fill_fps(self, fps: float) -> None:
        values = sorted(set(FPS_CHOICES) | ({round(fps, 3)} if all(abs(fps - v) > 1e-3 for v in FPS_CHOICES) else set()))
        with quiet(self.f_fps):
            self.f_fps.clear()
            for v in values:
                self.f_fps.addItem(f"{v:g} fps", v)
            self.f_fps.setCurrentIndex(min(range(len(values)), key=lambda i: abs(values[i] - fps)))

    def _on_name_typed(self, _text: str) -> None:
        self._name_typed = True

    def _rename(self) -> None:
        self._name_typed = False
        seq, name = self._seq(), self.f_name.text().strip()
        if seq is not None and name and name != seq.name:
            try:
                self.state.op("重新命名序列", "rename_sequence", name)
            except OpError:
                pass
        self.refresh()

    def _format(self, label: str, **kw: Any) -> None:
        try:
            self.state.op(label, "set_sequence_format", **kw)
        except OpError:
            pass
        self.refresh()

    def _set_preset(self, index: int) -> None:
        size = self.f_size.itemData(index)
        if size is None:
            self.f_w.setFocus()
            self.f_w.selectAll()
            return
        self._format("變更序列尺寸", width=size[0], height=size[1])

    def _set_fps(self, index: int) -> None:
        seq, fps = self._seq(), self.f_fps.itemData(index)
        if seq is not None and fps and abs(fps - seq.fps) > 1e-6:
            self._format("變更影格率", fps=fps)

    def _set_background(self, color: str) -> None:
        seq_id = self.state.seq_id
        try:
            self.state.do("變更背景色", lambda p: set_background(p, seq_id, color))
        except OpError:
            pass
        self.refresh()


# ============================================================================ 檢查
KIND_TEXT = {"shorts": "短影音", "assemble": "長片"}
REVIEW_TEXT = {"pending": "尚未審片（機器不會代替你判斷好不好看）", "approved": "你已審片：通過",
               "needs_changes": "你已審片：需要修改", "rejected": "你已審片：退回",
               "stale": "你之前的審片已過期（審片之後序列又改過）"}
SHORT_QA = {"not_run": "尚未檢查", "pass": "通過", "warn": "有警告", "fail": "未通過", "stale": "已過期"}
SHORT_REVIEW = {"pending": "尚未審片", "approved": "通過", "needs_changes": "需修改", "rejected": "退回", "stale": "已過期"}


def _states_line(states: list[str], full: dict[str, str], short: dict[str, str], title: str) -> str:
    """One line for the states of the sequences of a run: the full sentence when they agree, else a count per state."""
    if len(set(states)) == 1:
        return full.get(states[0], f"{title}：未知") + (f"（{len(states)} 個序列）" if len(states) > 1 else "")
    return f"{title}：" + "、".join(f"{short.get(k, k)} {states.count(k)}" for k in short if k in states)


class ChecksTab(QWidget):
    """檢查: recent auto-edit runs, ``validate_sequence`` issues (double-click jumps there) and human review."""

    MAX_RUNS = 5

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        lay = QVBoxLayout(self)
        g_runs = QGroupBox("自動剪輯紀錄")
        v = QVBoxLayout(g_runs)
        self.runs = QTreeWidget()
        self.runs.setHeaderHidden(True)
        self.runs.setRootIsDecorated(True)
        self._open: dict[str, bool] = {}                  # the person's own expand / collapse choices ("run id:node")
        self._building = False
        self.runs.itemExpanded.connect(lambda it: self._remember(it, True))
        self.runs.itemCollapsed.connect(lambda it: self._remember(it, False))
        v.addWidget(self.runs)
        g_issues = QGroupBox("序列檢查（雙擊可跳到該處）")
        v2 = QVBoxLayout(g_issues)
        self.issues = QListWidget()
        self.issues.itemDoubleClicked.connect(self._jump)
        self.issues.itemActivated.connect(self._jump)
        v2.addWidget(self.issues)
        g_review = QGroupBox("人工審片")
        v3 = QVBoxLayout(g_review)
        self.review = HumanReviewBox(state)
        v3.addWidget(self.review)
        lay.addWidget(g_runs, 3)
        lay.addWidget(g_issues, 2)
        lay.addWidget(g_review)
        state.edited.connect(self._on_edited)
        state.sequence_changed.connect(self.refresh)
        state.project_replaced.connect(self.refresh)
        self.refresh()

    def _on_edited(self, _kind: str, _label: str) -> None:
        self.refresh()

    def refresh(self) -> None:
        self._fill_runs()
        self._fill_issues()

    def _remember(self, item: QTreeWidgetItem, opened: bool) -> None:
        key = item.data(0, Qt.UserRole)
        if key and not self._building:
            self._open[str(key)] = opened

    def _fill_runs(self) -> None:
        """Rebuilt on every edit; the nodes the person opened or closed keep their state."""
        self._building = True
        try:
            self.runs.clear()
            runs = list(reversed(self.state.project.meta.get("ai_runs", [])))[: self.MAX_RUNS]
            if not runs:
                QTreeWidgetItem(self.runs, ["這個專案還沒有自動剪輯紀錄。"]).setForeground(0, QColor(theme.TEXT_DIM))
                return
            current = self.state.run_for_sequence()
            current_id = current.get("run_id") if current else None
            for i, run in enumerate(runs):
                top = self._run_item(run, run.get("run_id") == current_id)
                top.setExpanded(self._open.get(str(top.data(0, Qt.UserRole)), i == 0 or run.get("run_id") == current_id))
        finally:
            self._building = False

    @staticmethod
    def _line(parent: Any, text: str, color: Optional[str] = None) -> QTreeWidgetItem:
        """A tree line.  A narrow dock elides long lines (the 「目前序列」 mark, a long warning), so the tooltip carries
        the whole text."""
        item = QTreeWidgetItem(parent, [text])
        item.setToolTip(0, text)
        if color:
            item.setForeground(0, QColor(color))
        return item

    def _run_item(self, run: dict, is_current: bool) -> QTreeWidgetItem:
        when = str(run.get("applied_at", ""))[:16].replace("T", " ")
        n = len(run.get("sequence_ids", []))
        title = f"{KIND_TEXT.get(run.get('kind'), run.get('kind', '自動剪輯'))} · {when} · {n} 個序列"
        top = self._line(self.runs, title + ("（目前序列）" if is_current else ""))
        top.setData(0, Qt.UserRole, f"{run.get('run_id')}:run")
        if is_current:
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
        self._line(top, f"計畫雜湊：{str(run.get('plan_sha256', ''))[:12]}")
        seqs = []                                    # the sequences of this run that still exist: each has its own state
        for sid in run.get("sequence_ids", []):
            try:
                seqs.append(self.state.project.sequence(sid))
            except KeyError:
                continue
        qa = [planmod.qa_state(run, q)[0] for q in seqs] or ["not_run"]
        verdicts = [planmod.review_state(run, q)[0] for q in seqs] or ["pending"]
        self._line(top, _states_line(qa, QA_TEXT, SHORT_QA, "技術檢查"))
        self._line(top, _states_line(verdicts, REVIEW_TEXT, SHORT_REVIEW, "審片"),
                   theme.OK if set(verdicts) == {"approved"} else theme.WARN)
        warnings = list(run.get("warnings", []))
        w_item = self._line(top, f"審核警告（{len(warnings)}）" if warnings else "審核警告：沒有")
        w_item.setData(0, Qt.UserRole, f"{run.get('run_id')}:warnings")
        for text in warnings[:30]:
            self._line(w_item, str(text), theme.WARN)
        w_item.setExpanded(self._open.get(str(w_item.data(0, Qt.UserRole)), False))
        return top

    def _fill_issues(self) -> None:
        self.issues.clear()
        try:
            seq = self.state.sequence()
        except KeyError:
            return
        found = validate_sequence(self.state.project, seq)
        if not found:
            it = QListWidgetItem("沒有發現結構問題。")
            it.setForeground(QColor(theme.TEXT_DIM))
            self.issues.addItem(it)
            return
        for iss in found:
            it = QListWidgetItem(f"{'錯誤' if iss.level == 'error' else '警告'}：{iss.message}")
            it.setForeground(QColor(theme.ERROR if iss.level == "error" else theme.WARN))
            it.setData(Qt.UserRole, iss.where)
            it.setToolTip(it.text())                 # the whole message: a narrow dock elides it (where = UserRole)
            self.issues.addItem(it)

    def _jump(self, item: QListWidgetItem) -> None:
        """Select the clip / caption the issue points at and move the playhead there."""
        where = item.data(Qt.UserRole) or ""
        target = where.rsplit("/", 1)[-1]
        seq = self.state.sequence()
        _track, clip = seq.find_clip(target)
        if clip is not None:
            self.state.select([clip.id])
            self.state.set_playhead(clip.start)
            return
        cue = seq.find_cue(target)
        if cue is not None:
            self.state.select_cues([cue.id])
            self.state.set_playhead(cue.start)


# ============================================================================ panel
class InspectorPanel(QTabWidget):
    """Right-hand inspector with the tabs 片段 / 字幕 / 序列 / 檢查 (GUI_SPEC §5, §11)."""

    def __init__(self, state: AppState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.clip_tab = ClipTab(state)
        self.captions_tab = CaptionsTab(state)
        self.sequence_tab = SequenceTab(state)
        self.checks_tab = ChecksTab(state)
        for widget, name in ((self.clip_tab, "片段"), (self.captions_tab, "字幕"), (self.sequence_tab, "序列"),
                             (self.checks_tab, "檢查")):
            self.addTab(widget, name)
        self.setMinimumWidth(340)
        keep_keyboard_on_timeline(self)          # buttons / check boxes / sliders / combos: Tab gives them the keys, a click does not
