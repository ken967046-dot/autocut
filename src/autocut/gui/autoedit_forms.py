"""Parameter forms of the auto-edit dialogs (GUI_SPEC §6): widgets <-> ``ShortsParams`` / ``AssembleParams`` /
``LongformParams``.

``ShortsForm``, ``AssembleForm`` and ``LongformForm`` are plain widgets (no task logic): ``set_params(p)`` fills them, ``params()`` reads
them back, ``validate()`` returns a 繁中 problem text (or ""), and ``set_speech_status(...)`` shows the speech-recognition
hint next to the caption settings.  Every widget is reachable as ``form.w["<field name>"]`` (the dataclass field name).
The dialog (``autoedit_dialog``) owns the flow; this module owns the look of the settings.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)

from ..autoedit.api import AutoEditError
from ..autoedit.cutlist import load_cutlist
from ..autoedit.config import (ASPECTS, PLATFORM_RULES, AssembleParams, LongformParams, ShortsParams,
                               pace_shot_range)
from . import theme

SETTINGS_ORG = SETTINGS_APP = "AutoCutStudio"
KEY_SHORTS = "autoedit/shorts_params"
KEY_ASSEMBLE = "autoedit/assemble_params"
KEY_LONGFORM = "autoedit/longform_params"
KEY_AI_KEY = "ai/anthropic_api_key"
AI_PIP_COMMAND = "pip install -r requirements-ai.txt"
KEY_ADVANCED = "autoedit/advanced_open"
KEY_LAST_DIR = "autoedit/last_dir"
PIP_COMMAND = "pip install -r requirements-speech.txt"
CAPTION_FILTER = "字幕檔 (*.srt *.ass *.vtt);;所有檔案 (*)"
MUSIC_FILTER = "音訊檔 (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus);;所有檔案 (*)"
CUTLIST_FILTER = "剪輯清單 (*.json);;所有檔案 (*)"
CAPTION_EXTS = (".srt", ".ass", ".vtt")
PLATFORM_ZH = {"yt_shorts": "YouTube Shorts 短影音", "ig_reels": "Instagram Reels 連續短片",
               "fb_reels": "Facebook Reels 連續短片", "tiktok": "TikTok 短影音"}
LANGUAGES = [("zh", "中文（國語）"), ("en", "英文"), ("ja", "日文"), ("ko", "韓文"), ("es", "西班牙文"),
             ("fr", "法文"), ("de", "德文"), ("id", "印尼文"), ("vi", "越南文"), ("th", "泰文")]


# ================================================================================================ settings
def settings() -> QSettings:
    """The app's persistent settings (tests redirect them with ``QSettings.setPath`` / ``setDefaultFormat``)."""
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def load_params(cls, key: str):
    """The parameters used last time (or the dataclass defaults when nothing / garbage was stored)."""
    try:
        raw = settings().value(key, "")
        data = json.loads(str(raw)) if raw else {}
        return cls.from_dict(data if isinstance(data, dict) else {})
    except Exception:                                    # noqa: BLE001 - a corrupt setting must never block the dialog
        return cls()


def save_params(key: str, params) -> None:
    """Remember ``params``.  The caption *file* is not remembered: it belongs to one particular video."""
    d = params.to_dict()
    if d.get("captions") == "file":
        d["captions"] = "auto" if isinstance(params, ShortsParams) else "off"
    d["caption_file"] = ""
    if "transcript_file" in d:
        d["transcript_file"] = ""                        # the subtitle file belongs to one video too
    if "cutlist_file" in d:
        d["cutlist_file"] = ""                           # and so does a cut list
    try:
        s = settings()
        s.setValue(key, json.dumps(d, ensure_ascii=False))
        s.sync()
    except Exception:                                    # noqa: BLE001
        pass


# ================================================================================================ small widgets
class Note(QLabel):
    """A coloured one-paragraph note (level: info | warn | error); hidden while it has no text."""

    COLORS = {"info": (theme.ACCENT, "#1f2f42"), "warn": (theme.WARN, "#3a2f17"), "error": (theme.ERROR, "#3b1f1f"),
              "ok": (theme.OK, "#1d3328")}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.level = "info"
        self.hide()

    def set_note(self, text: str, level: str = "info") -> None:
        self.level = level
        edge, bg = self.COLORS.get(level, self.COLORS["info"])
        self.setStyleSheet(f"QLabel {{ background: {bg}; color: {theme.TEXT}; border: none; "
                           f"border-left: 3px solid {edge}; border-radius: 3px; padding: 6px 10px; }}")
        self.setText(text)
        self.setVisible(bool(text))


def dim_label(text: str = "", wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("dim")
    lab.setWordWrap(wrap)
    return lab


class PathRow(QWidget):
    """A path line edit with 「選擇…」 (and 「清除」 for optional paths)."""

    changed = Signal()

    def __init__(self, title: str, file_filter: str, placeholder: str = "", clearable: bool = True,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title, self._filter = title, file_filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.btn_browse = QPushButton("選擇…")
        self.btn_browse.setAutoDefault(False)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn_browse)
        self.btn_clear: Optional[QPushButton] = None
        if clearable:
            self.btn_clear = QPushButton("清除")
            self.btn_clear.setAutoDefault(False)
            self.btn_clear.clicked.connect(lambda *_: self.set_path(""))
            lay.addWidget(self.btn_clear)
        self.edit.textChanged.connect(lambda *_: self._text_changed())
        self.btn_browse.clicked.connect(lambda *_: self.browse())
        self._refresh()

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: str) -> None:
        self.edit.setText(str(path or ""))

    def browse(self) -> None:
        start = self.path() or str(settings().value(KEY_LAST_DIR, "") or "")
        if start and not os.path.isdir(start):
            start = os.path.dirname(start)
        chosen, _ = QFileDialog.getOpenFileName(self, self._title, start, self._filter)
        if chosen:
            self.set_path(chosen)
            try:
                settings().setValue(KEY_LAST_DIR, os.path.dirname(chosen))
            except Exception:                            # noqa: BLE001
                pass

    def _refresh(self) -> None:
        if self.btn_clear is not None:
            self.btn_clear.setEnabled(bool(self.path()))

    def _text_changed(self) -> None:
        self._refresh()
        self.changed.emit()


class Collapsible(QWidget):
    """A header button that shows / hides its body (「進階設定」)."""

    toggled = Signal(bool)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.header = QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setArrowType(Qt.RightArrow)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet("QToolButton { border: none; background: transparent; font-weight: bold; "
                                  "padding: 4px 2px; } QToolButton:hover { color: %s; }" % theme.ACCENT)
        self.body = QFrame()
        self.body.setVisible(False)
        self.body_layout = QFormLayout(self.body)
        self.body_layout.setContentsMargins(14, 4, 4, 4)
        self.body_layout.setHorizontalSpacing(12)
        self.body_layout.setVerticalSpacing(8)
        lay.addWidget(self.header, 0, Qt.AlignLeft)
        lay.addWidget(self.body)
        self.header.toggled.connect(self._toggled)

    def _toggled(self, on: bool) -> None:
        self.header.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)
        self.toggled.emit(on)

    def expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, on: bool) -> None:
        self.header.setChecked(bool(on))


def _select(combo: QComboBox, value: Any) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    if isinstance(value, str) and value:                  # a value this build does not list (kept, not lost)
        combo.addItem(value, value)
        combo.setCurrentIndex(combo.count() - 1)


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def saved_ai_key() -> str:
    try:
        return str(settings().value(KEY_AI_KEY, "") or "")
    except Exception:                                    # noqa: BLE001
        return ""


def apply_saved_ai_key() -> bool:
    """Put the key saved in the settings into this process's environment (the SDK reads ``ANTHROPIC_API_KEY``).  An
    environment variable the person set themselves always wins.  Returns True when some key is available."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    key = saved_ai_key()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
        return True
    return False


def store_ai_key(key: str) -> None:
    key = key.strip()
    old = saved_ai_key()
    try:
        s = settings()
        if key:
            s.setValue(KEY_AI_KEY, key)
        else:
            s.remove(KEY_AI_KEY)
        s.sync()
    except Exception:                                    # noqa: BLE001
        pass
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    elif old and os.environ.get("ANTHROPIC_API_KEY") == old:     # only the key WE put there, never the person's own
        os.environ.pop("ANTHROPIC_API_KEY", None)


# ================================================================================================ base form
class _FormBase(QWidget):
    """Shared machinery: a field registry, group boxes, caption / music groups and the speech note."""

    changed = Signal()
    PARAMS: type = ShortsParams

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.w: dict[str, QWidget] = {}
        self._get: dict[str, Callable[[], Any]] = {}
        self._set: dict[str, Callable[[Any], None]] = {}
        self._base = self.PARAMS()
        self._loading = False
        self._speech: Optional[tuple] = None             # (ok, reason, cached, model, mb) once the probe finished
        self.speech_note = Note()
        self.advanced = Collapsible("進階設定（很少需要改）")
        self.advanced.toggled.connect(self._remember_advanced)

    # ------------------------------------------------------------------ registry helpers
    def _reg(self, name: str, widget: QWidget, getter: Callable[[], Any], setter: Callable[[Any], None], signal,
             tip: str = "") -> QWidget:
        self.w[name] = widget
        self._get[name], self._set[name] = getter, setter
        if tip:
            widget.setToolTip(tip)
        signal.connect(lambda *_: self._on_change())
        return widget

    def combo(self, name: str, items: list[tuple[Any, str]], tip: str = "") -> QComboBox:
        cb = QComboBox()
        for data, text in items:
            cb.addItem(text, data)
        self._reg(name, cb, cb.currentData, lambda v: _select(cb, v), cb.currentIndexChanged, tip)
        return cb

    def spin(self, name: str, lo: float, hi: float, step: float = 1.0, decimals: int = 0, suffix: str = "",
             tip: str = "", special: str = "") -> QWidget:
        if decimals == 0 and float(step).is_integer() and float(lo).is_integer() and float(hi).is_integer():
            sp: QWidget = QSpinBox()
            sp.setRange(int(lo), int(hi))
            sp.setSingleStep(int(step))
            getter: Callable[[], Any] = sp.value
            setter: Callable[[Any], None] = lambda v: sp.setValue(int(round(_num(v, lo))))
        else:
            sp = QDoubleSpinBox()
            sp.setDecimals(decimals)
            sp.setRange(lo, hi)
            sp.setSingleStep(step)
            getter = sp.value
            setter = lambda v: sp.setValue(_num(v, lo))
        if suffix:
            sp.setSuffix(suffix)
        if special:
            sp.setSpecialValueText(special)
        sp.setMaximumWidth(170)
        sp.setKeyboardTracking(False)
        self._reg(name, sp, getter, setter, sp.valueChanged, tip)
        return sp

    def check(self, name: str, text: str, tip: str = "") -> QCheckBox:
        cb = QCheckBox(text)
        self._reg(name, cb, cb.isChecked, lambda v: cb.setChecked(bool(v)), cb.toggled, tip)
        return cb

    def path(self, name: str, title: str, file_filter: str, placeholder: str = "", tip: str = "",
             clearable: bool = True) -> PathRow:
        row = PathRow(title, file_filter, placeholder, clearable)
        self._reg(name, row, row.path, row.set_path, row.changed, tip)
        return row

    @staticmethod
    def columns(parent_layout: QVBoxLayout) -> tuple[QVBoxLayout, QVBoxLayout]:
        """Two top-aligned columns of group boxes (each box keeps its natural height)."""
        row = QHBoxLayout()
        row.setSpacing(14)
        left, right = QVBoxLayout(), QVBoxLayout()
        for col in (left, right):
            col.setSpacing(10)
            col.setContentsMargins(0, 0, 0, 0)
            row.addLayout(col, 1)
        parent_layout.addLayout(row)
        return left, right

    @staticmethod
    def group(title: str) -> tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setContentsMargins(14, 10, 14, 10)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return box, form

    # ------------------------------------------------------------------ the groups both forms share
    def build_captions_group(self, default_mode_tip: str) -> QGroupBox:
        box, form = self.group("字幕")
        cap = self.combo("captions", [("auto", "自動（語音辨識）"), ("file", "使用字幕檔"), ("off", "不加字幕")],
                         "自動＝用語音辨識產生字幕（需安裝 faster-whisper）；使用字幕檔＝你自備 .srt / .ass / .vtt；"
                         "不加字幕。" + default_mode_tip)
        form.addRow("字幕來源", cap)
        row = self.path("caption_file", "選擇字幕檔", CAPTION_FILTER, "選擇 .srt / .ass / .vtt",
                        "字幕的時間要對應到來源影片的時間。")
        form.addRow("字幕檔", row)
        lang = self.combo("language", LANGUAGES, "語音辨識（與字幕處理）使用的語言。")
        form.addRow("語言", lang)
        trad = self.check("traditional_chinese", "辨識出的中文轉成繁體",
                          "把辨識出的簡體中文轉成繁體（台灣用語）。需要 opencc；沒安裝時保留原文並在備註說明。")
        form.addRow("", trad)
        form.addRow(self.speech_note)
        return box

    def build_highlight_group(self, what: str = "短影音") -> QGroupBox:
        """Where the highlights come from: a subtitle file (or 「一鍵產生字幕」), and local rules or the AI judge."""
        box, form = self.group("找重點（爆點）")
        row = self.path("transcript_file", "選擇字幕檔", CAPTION_FILTER, "（自動：影片旁同名字幕檔或語音辨識）",
                        f"用這個字幕檔的「文字」判斷哪裡是重點，再剪成{what}。沒選的話會自動找影片旁同名的 .srt，"
                        "再沒有才用語音辨識。字幕時間要對應來源影片。")
        form.addRow("字幕檔", row)
        self.btn_subtitles = QPushButton("一鍵產生字幕")
        self.btn_subtitles.setFocusPolicy(Qt.TabFocus)
        self.btn_subtitles.setToolTip("用語音辨識把整支影片轉成字幕，存成影片旁的同名 .srt（可以先用記事本修正錯字），"
                                      "之後找重點會直接用它。需要安裝語音辨識；第一次會詢問是否下載模型，素材不會上傳。")
        form.addRow("", self.btn_subtitles)
        form.addRow("判斷方式", self.combo("highlight_engine", [
            ("local", "本機規則（離線，免費）"), ("claude", "AI 判斷（Claude，需金鑰與網路）")],
            "本機規則：看提問、反轉、情緒、乾貨、數字、語氣等線索。AI 判斷：把字幕「文字」送給 Claude 評分每一句，"
            "更懂語意與笑點；只傳文字與時間碼，不會上傳影片或聲音；結果會存在這台電腦，同一支影片再剪不必重送。"))
        self.btn_ai_key = QPushButton("設定 AI 金鑰…")
        self.btn_ai_key.setFocusPolicy(Qt.TabFocus)
        self.btn_ai_key.setToolTip("Claude API 金鑰（console.anthropic.com 取得）。存在這台電腦的使用者設定中；"
                                   "也可以改用環境變數 ANTHROPIC_API_KEY。")
        form.addRow("", self.btn_ai_key)
        self.transcript_note = Note()
        form.addRow(self.transcript_note)
        return box

    def build_music_group(self) -> QGroupBox:
        box, form = self.group("配樂")
        form.addRow("配樂檔", self.path("music_path", "選擇配樂檔", MUSIC_FILTER, "（不加配樂）",
                                        "選填。音樂會鋪滿整支成品並在結尾淡出；音量固定，說話時不會自動壓低。"))
        form.addRow("音量", self.spin("music_gain_db", -60.0, 0.0, 1.0, 0, " dB",
                                      "配樂相對原音的固定音量。數字越小越小聲（-20 dB 約是低背景音樂）。"))
        return box

    # ------------------------------------------------------------------ params <-> widgets
    def params(self):
        d = self._base.to_dict()
        for name, getter in self._get.items():
            d[name] = getter()
        return self.PARAMS.from_dict(d)

    def set_params(self, p) -> None:
        self._base = self.PARAMS.from_dict(p.to_dict())
        self._loading = True
        try:
            for name, setter in self._set.items():
                setter(getattr(p, name))
        finally:
            self._loading = False
        self._sync()
        self.changed.emit()

    def _on_change(self) -> None:
        if self._loading:
            return
        self._sync()
        self.changed.emit()

    def _remember_advanced(self, on: bool) -> None:
        try:
            settings().setValue(KEY_ADVANCED, bool(on))
        except Exception:                                # noqa: BLE001
            pass

    def restore_advanced(self) -> None:
        try:
            self.advanced.set_expanded(str(settings().value(KEY_ADVANCED, "false")).lower() in ("true", "1"))
        except Exception:                                # noqa: BLE001
            pass

    # ------------------------------------------------------------------ captions / speech hint
    def caption_mode(self) -> str:
        return str(self._get["captions"]())

    def wants_speech(self) -> bool:
        return self.caption_mode() == "auto"

    def set_speech_status(self, ok: bool, reason: str, cached: bool, model: str, mb: int) -> None:
        """Called by the dialog once it knows whether speech recognition is installed / its model is on disk."""
        self._speech = (bool(ok), str(reason or ""), bool(cached), model, int(mb))
        self._sync_note()

    @staticmethod
    def _unavailable_text(reason: str, consequence: str) -> str:
        head = (reason or "尚未安裝語音辨識").strip().rstrip("。")
        text = f"{head}。\n{consequence}"
        return text if PIP_COMMAND in head else f"{text}\n安裝指令：{PIP_COMMAND}"

    def _sync_note(self) -> None:
        if self._speech is None or not self.wants_speech():
            self.speech_note.set_note("")
            return
        ok, reason, cached, model, mb = self._speech
        if not ok:
            self.speech_note.set_note(self._unavailable_text(
                reason, "在安裝之前不會自動產生字幕（除非你在「字幕來源」改選字幕檔）。"), "warn")
        elif not cached:
            self.speech_note.set_note(
                f"第一次使用語音辨識要下載模型「{model}」（約 {mb} MB）。按下「開始分析」後會先問你，同意才會下載；"
                "素材不會上傳。", "info")
        else:
            self.speech_note.set_note("")

    # ------------------------------------------------------------------ enabling / validation
    def _sync(self) -> None:                             # overridden
        self._sync_note()

    def _sync_captions(self, speech_used: bool) -> None:
        mode = self.caption_mode()
        self.w["caption_file"].setEnabled(mode == "file")
        lang_on = mode != "off" or speech_used
        self.w["language"].setEnabled(lang_on)
        zh = str(self._get["language"]() or "").lower().startswith("zh")
        self.w["traditional_chinese"].setEnabled(lang_on and zh)

    def _sync_music(self) -> None:
        if "music_path" in self._get:
            self.w["music_gain_db"].setEnabled(bool(self._get["music_path"]()))

    def _sync_ai(self) -> None:
        if "highlight_engine" in self._get:
            self.btn_ai_key.setVisible(self._get["highlight_engine"]() == "claude")

    def validate(self) -> str:
        """A 繁中 problem text for settings that cannot work (empty string = fine)."""
        if self.caption_mode() == "file":
            f = str(self._get["caption_file"]())
            if not f:
                return "你選了「使用字幕檔」，請先選擇字幕檔（.srt / .ass / .vtt），或把字幕來源改成其他選項。"
            if not os.path.isfile(f):
                return f"找不到字幕檔：{f}"
            if os.path.splitext(f)[1].lower() not in CAPTION_EXTS:
                return "字幕檔格式不支援，請選擇 .srt、.ass 或 .vtt 檔。"
        if "transcript_file" in self._get:
            t = str(self._get["transcript_file"]())
            if t and not os.path.isfile(t):
                return f"找不到找重點用的字幕檔：{t}"
            if t and os.path.splitext(t)[1].lower() not in CAPTION_EXTS:
                return "找重點用的字幕檔格式不支援，請選擇 .srt、.ass 或 .vtt 檔。"
        m = str(self._get["music_path"]()) if "music_path" in self._get else ""
        if m and not os.path.isfile(m):
            return f"找不到配樂檔：{m}"
        return ""


# ================================================================================================ shorts
class ShortsForm(_FormBase):
    PARAMS = ShortsParams

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        left, right = self.columns(lay)

        # ---- 成品
        box, form = self.group("成品")
        items = []
        for key, rule in PLATFORM_RULES.items():
            items.append((key, f"{PLATFORM_ZH.get(key, rule['label'])}（{rule['dur_min']:g}–{rule['dur_max']:g} 秒）"))
        form.addRow("發佈平台", self.combo(
            "platform", items, "決定短影音的長度範圍。YouTube Shorts 會避開 25–45 秒這段（範例校準值，不是定律）。"))
        form.addRow("支數", self.spin("count", 1, 10, 1, 0, " 支",
                                      "最多產生幾支；素材太短或條件太嚴時可能少於這個數字。"))
        form.addRow("剪輯清單", self.path(
            "cutlist_file", "選擇剪輯清單", CUTLIST_FILTER, "（不用：自動找爆點）",
            "選填。已經決定好要剪哪幾段時用：剪輯清單（.json）逐支列出來源時間段、播放順序（可以把後面的金句放到最前面"
            "當開場、跳過口誤）與字卡。選了就照清單剪，不做自動搜尋，「支數」與長度範圍不適用。"))
        form.addRow("目標長度", self.spin("target_seconds", 5.0, 180.0, 1.0, 1, " 秒",
                                          "偏好的長度。超出平台長度範圍時會自動調整，並在備註說明。"))
        left.addWidget(box)

        # ---- 直式取景
        box, form = self.group("直式取景（16:9 → 9:16）")
        form.addRow("取景方式", self.combo("reframe", [
            ("face", "人臉追蹤（預設）"), ("motion", "動態追蹤"), ("center", "置中固定"),
            ("blur", "不裁切（完整畫面＋模糊背景）")],
            "人臉追蹤＝跟著說話的人；動態追蹤＝跟著動作最多的地方；置中＝固定取中間；"
            "不裁切＝完整畫面放在模糊背景上。"))
        srow = QWidget()
        sl = QHBoxLayout(srow)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)
        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(0, 100)
        self.strength.setPageStep(10)
        self.strength_label = QLabel("50%")
        self.strength_label.setMinimumWidth(40)
        self.strength_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sl.addWidget(self.strength, 1)
        sl.addWidget(self.strength_label)
        self.strength.valueChanged.connect(lambda v: self.strength_label.setText(f"{v}%"))
        self._reg("reframe_strength", self.strength, lambda: self.strength.value() / 100.0,
                  lambda v: self.strength.setValue(int(round(max(0.0, min(1.0, _num(v, 0.5))) * 100))),
                  self.strength.valueChanged,
                  "虛擬攝影機跟隨的反應速度：0% 幾乎固定不動，100% 反應很快（畫面可能比較晃）。")
        form.addRow("取景靈敏度", srow)
        hint = dim_label("0% 鎖定　⟷　100% 靈敏（只對人臉／動態追蹤有效）")
        form.addRow("", hint)
        left.addWidget(box)

        # ---- 緊湊化
        box, form = self.group("節奏")
        form.addRow("", self.check("tighten", "剪掉說話中的長停頓（跳接緊湊）",
                                   "把說話中間超過「停頓判定長度」的空白縮短，讓節奏更緊湊；剪點會盡量落在句子邊界。"))
        left.addWidget(box)

        # ---- 找重點 / 字幕 / 配樂 (right column)
        right.addWidget(self.build_highlight_group("短影音"))
        right.addWidget(self.build_captions_group("　這份設定同時決定是否用逐字稿挑選剪點。"))
        right.addWidget(self.build_music_group())
        left.addStretch(1)
        right.addStretch(1)

        # ---- 進階
        body = self.advanced.body_layout
        body.addRow("最短長度", self.spin("min_seconds", 0.0, 600.0, 1.0, 1, " 秒", "0＝使用平台的長度範圍。",
                                          special="依平台"))
        body.addRow("最長長度", self.spin("max_seconds", 0.0, 600.0, 1.0, 1, " 秒", "0＝使用平台的長度範圍。",
                                          special="依平台"))
        size = QWidget()
        szl = QHBoxLayout(size)
        szl.setContentsMargins(0, 0, 0, 0)
        szl.setSpacing(6)
        szl.addWidget(self.spin("width", 16, 7680, 2, 0, "", "輸出寬度（像素，需為偶數）。預設 1080。"))
        szl.addWidget(QLabel("×"))
        szl.addWidget(self.spin("height", 16, 7680, 2, 0, "", "輸出高度（像素，需為偶數）。預設 1920（9:16）。"))
        szl.addWidget(QLabel("像素"))
        szl.addStretch(1)
        body.addRow("輸出尺寸", size)
        body.addRow("影格率", self.spin("fps", 1.0, 240.0, 1.0, 2, " fps", "輸出的每秒格數。"))
        body.addRow("靜音門檻", self.spin("silence_db", -90.0, -10.0, 1.0, 0, " dB",
                                          "低於這個音量就視為停頓。數字越小越嚴格（只有很安靜的地方才算停頓）。"))
        body.addRow("停頓判定長度", self.spin("pause_min", 0.1, 10.0, 0.05, 2, " 秒",
                                              "至少這麼長的停頓才會被縮短。"))
        body.addRow("縮短後保留", self.spin("pause_keep", 0.0, 3.0, 0.05, 2, " 秒",
                                            "每段被縮短的停頓仍保留這麼長，聽起來才不會太急。"))
        body.addRow("第一個剪點", self.spin("first_cut_max", 0.5, 10.0, 0.05, 2, " 秒內",
                                            "開頭多久之內要有第一個剪點或畫面變化（短影音的「前 2 秒」規則）。"))
        body.addRow("來源內最小間隔", self.spin("min_gap_between", 0.0, 3600.0, 1.0, 1, " 秒",
                                                "同一支來源挑出的短影音，起點至少相隔這麼遠，避免內容重複。"))
        body.addRow("", self.check("use_speech", "用逐字稿挑選句子邊界的剪點",
                                   "有逐字稿（語音辨識或字幕檔）時，剪點盡量落在句子邊界。"))
        lay.addWidget(self.advanced)
        lay.addStretch(1)
        self.set_params(self.PARAMS())        # widgets start at the dataclass defaults

    def wants_speech(self) -> bool:
        """Speech recognition is wanted for the captions, or (unless the person brings a caption file) for the cuts."""
        mode = self.caption_mode()
        return mode == "auto" or (mode == "off" and bool(self._get["use_speech"]()))

    def _sync(self) -> None:
        listed = bool(self._get["cutlist_file"]())             # a cut list decides how many shorts and how long
        self.w["count"].setEnabled(not listed)
        self.w["target_seconds"].setEnabled(not listed)
        self.w["highlight_engine"].setEnabled(not listed)       # nothing to search: the AI judge is not called
        self.w["reframe_strength"].setEnabled(self._get["reframe"]() in ("face", "motion"))
        self.strength_label.setEnabled(self.w["reframe_strength"].isEnabled())
        self._sync_captions(bool(self._get["use_speech"]()))
        self._sync_music()
        self._sync_ai()
        self.w["pause_min"].setEnabled(bool(self._get["tighten"]()))
        self.w["pause_keep"].setEnabled(bool(self._get["tighten"]()))
        self._sync_note()

    def _sync_note(self) -> None:
        if self._speech is None or not self.wants_speech():
            self.speech_note.set_note("")
            return
        ok, reason, cached, model, mb = self._speech
        if not ok and self.caption_mode() != "auto":
            self.speech_note.set_note(self._unavailable_text(
                reason, "沒有語音辨識時，剪點只依停頓與鏡頭切換挑選，不保證落在句子邊界。"), "warn")
            return
        super()._sync_note()

    def validate(self) -> str:
        bad = super().validate()
        if bad:
            return bad
        w, h = int(self._get["width"]()), int(self._get["height"]())
        if w % 2 or h % 2:
            return "輸出尺寸的寬與高都要是偶數（進階設定）。"
        lo, hi = float(self._get["min_seconds"]()), float(self._get["max_seconds"]())
        if lo and hi and lo > hi:
            return "最短長度不能大於最長長度（進階設定）。"
        f = str(self._get["cutlist_file"]())
        if f:
            if not os.path.isfile(f):
                return f"找不到剪輯清單：{f}"
            try:
                load_cutlist(f)
            except AutoEditError as exc:
                return str(exc)
        return ""


# ================================================================================================ assemble
class AssembleForm(_FormBase):
    PARAMS = AssembleParams

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        left, right = self.columns(lay)

        # ---- 成品與節奏
        box, form = self.group("成品與節奏")
        form.addRow("目標長度", self.spin(
            "target_seconds", 0.0, 36000.0, 5.0, 0, " 秒",
            "想要的總長。0＝保留所有可用的內容；可用內容不足時會用完它們，並在備註說明。", special="全部可用內容"))
        form.addRow("畫面比例", self.combo("aspect", [("auto", "自動（依多數素材）")] +
                                           [(k, k) for k in ASPECTS], "成品的畫面比例。"))
        form.addRow("素材順序", self.combo("order", [
            ("chronological", "依拍攝時間"), ("name", "依檔名"), ("as_given", "依你選取的順序")],
            "依拍攝時間：先看影片內嵌的拍攝時間，沒有就用檔案修改時間，再用檔名；依檔名：自然排序（clip2 在 clip10 前）。"))
        paces = []
        for key, label in (("fast", "快"), ("medium", "中等"), ("slow", "慢")):
            lo, hi = pace_shot_range(key)
            paces.append((key, f"{label}（鏡頭約 {lo:g}–{hi:g} 秒）"))
        form.addRow("節奏", self.combo("pace", paces, "決定每個鏡頭大約留多久；快＝鏡頭短、換得頻繁。"))
        form.addRow("轉場方式", self.combo("transition", [("crossfade", "交叉淡化"), ("cut", "直接切換")],
                                           "交叉淡化＝前後鏡頭重疊淡入淡出（會讓總長略短）；直接切換＝硬切。"))
        form.addRow("淡化長度", self.spin("crossfade_seconds", 0.05, 3.0, 0.05, 2, " 秒", "交叉淡化重疊的時間。"))
        left.addWidget(box)

        # ---- 畫面與聲音
        box, form = self.group("畫面與聲音")
        form.addRow("", self.check("drop_bad", "丟掉品質差的片段",
                                   "略過模糊、太暗、晃動，或沒有聲音又靜止的段落（每一段的原因都會記在審閱頁）。"))
        form.addRow("", self.check("keep_audio", "保留素材原本的聲音", "關閉＝成品只有配樂（或無聲）。"))
        form.addRow("比例不符時", self.combo("fit", [
            ("blur", "模糊背景填滿"), ("cover", "裁切填滿"), ("contain", "完整顯示（留黑邊）")],
            "素材畫面比例和成品不同時的處理方式。"))
        right.addWidget(self.build_captions_group(""))
        right.addWidget(box)
        left.addWidget(self.build_music_group())
        left.addStretch(1)
        right.addStretch(1)

        # ---- 進階
        body = self.advanced.body_layout
        body.addRow("影格率", self.spin("fps", 0.0, 240.0, 1.0, 2, " fps",
                                        "0＝自動（採用最多素材使用的影格率，限制在 24–60）。", special="自動"))
        body.addRow("最短鏡頭", self.spin("min_shot", 0.3, 30.0, 0.1, 1, " 秒", "比這更短的片段不會被挑進成品。"))
        body.addRow("最長鏡頭", self.spin("max_shot", 0.0, 120.0, 0.5, 1, " 秒",
                                          "單一鏡頭的上限。0＝依節奏決定。", special="依節奏"))
        body.addRow("配樂結尾淡出", self.spin("fade_music_out", 0.0, 30.0, 0.5, 1, " 秒",
                                              "成品結尾時配樂淡出的長度。"))
        lay.addWidget(self.advanced)
        lay.addStretch(1)
        self.set_params(self.PARAMS())        # widgets start at the dataclass defaults

    def _sync(self) -> None:
        self.w["crossfade_seconds"].setEnabled(self._get["transition"]() == "crossfade")
        self._sync_captions(False)
        self._sync_music()
        self._sync_note()


# ================================================================================================ long video
class LongformForm(_FormBase):
    PARAMS = LongformParams

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        left, right = self.columns(lay)

        # ---- 剪法
        box, form = self.group("剪法")
        form.addRow("模式", self.combo("mode", [("highlights", "重點精華＋去水詞"), ("clean", "只去水詞與停頓（內容全留）")],
                                       "重點精華：依內容挑出最有料的句子（含它的鋪陳），其餘剪掉；只去水詞：每一句都留，只剪「嗯、呃、那個」與長停頓。"))
        ratio = QSpinBox()
        ratio.setRange(5, 100)
        ratio.setSingleStep(5)
        ratio.setSuffix(" %")
        ratio.setMaximumWidth(170)
        ratio.setKeyboardTracking(False)
        self._reg("keep_ratio", ratio, lambda: ratio.value() / 100.0,
                  lambda v: ratio.setValue(int(round(max(0.05, min(1.0, _num(v, 0.5))) * 100))), ratio.valueChanged,
                  "保留多少比例的「說話內容」。50% ＝ 一小時的訪談大約剪成 25–30 分鐘（停頓與水詞另外再剪）。")
        form.addRow("保留比例", ratio)
        form.addRow("目標長度", self.spin("target_seconds", 0.0, 36000.0, 30.0, 0, " 秒",
                                          "指定時取代保留比例。0＝依保留比例。", special="依保留比例"))
        left.addWidget(box)

        # ---- 去水詞
        box, form = self.group("去水詞與節奏")
        form.addRow("", self.check("remove_fillers", "剪掉「嗯、呃、那個、就是」這類水詞",
                                   "「嗯、呃」一律剪；「那個、就是、然後、對」只有前後都停頓、單獨出現時才剪（「那個人」「對對方」不會被剪）。"
                                   "需要逐字時間（語音辨識或「一鍵產生字幕」）。"))
        form.addRow("", self.check("remove_stutters", "剪掉重複字（我我我 → 我）",
                                   "同一個詞連續出現三次以上才算重複字、只留最後一次；重複兩次（聊聊、謝謝、非常非常）一律保留。"))
        form.addRow("", self.check("tighten", "縮短長停頓", "超過「停頓判定長度」的空白只保留一小段。"))
        left.addWidget(box)

        # ---- 講者特寫
        box, form = self.group("講者特寫（兩人以上對談）")
        form.addRow("", self.check("speaker_zoom", "有人連續說話時放大到他身上",
                                   "畫面上有兩個人以上、其中一人連續說超過「觸發長度」時，切到他的特寫；換人說話或說太久時"
                                   "切回全景。判斷依據是人臉與嘴部動作（側臉、遮嘴、兩人同時說話時判斷不出來）。"))
        form.addRow("觸發長度", self.spin("zoom_min_turn", 1.0, 30.0, 0.5, 1, " 秒", "連續說話超過這麼久才放大。"))
        form.addRow("最長特寫", self.spin("zoom_max", 3.0, 120.0, 1.0, 0, " 秒",
                                          "一段特寫最長多久；說得更久時會在句子邊界切回全景，遇到爆點句再切回特寫。"))
        form.addRow("全景至少", self.spin("zoom_back_min", 0.5, 30.0, 0.5, 1, " 秒",
                                          "開場與每次換人說話時，先給多人全景這麼久，觀眾才看得出是誰在回答。"))
        right.addWidget(box)
        right.addWidget(self.build_highlight_group("精華長片"))

        box, form = self.group("字幕")
        form.addRow("字幕", self.combo("captions", [("off", "不加字幕"), ("auto", "自動（逐字稿或字幕檔）"),
                                                   ("file", "使用字幕檔")], "是否把字幕放進成品。"))
        form.addRow("字幕檔", self.path("caption_file", "選擇字幕檔", CAPTION_FILTER, "選擇 .srt / .ass / .vtt"))
        form.addRow("語言", self.combo("language", LANGUAGES, "語音辨識使用的語言。"))
        form.addRow("", self.check("traditional_chinese", "辨識出的中文轉成繁體"))
        form.addRow(self.speech_note)
        right.addWidget(box)
        left.addStretch(1)
        right.addStretch(1)

        # ---- 進階
        body = self.advanced.body_layout
        body.addRow("停頓判定長度", self.spin("pause_min", 0.1, 10.0, 0.05, 2, " 秒", "至少這麼長的停頓才會被縮短。"))
        body.addRow("縮短後保留", self.spin("pause_keep", 0.0, 3.0, 0.05, 2, " 秒", "每段被縮短的停頓仍保留這麼長。"))
        body.addRow("靜音門檻", self.spin("silence_db", -90.0, -10.0, 1.0, 0, " dB", "低於這個音量就視為停頓。"))
        body.addRow("特寫倍率", self.spin("zoom_level", 0.0, 3.0, 0.1, 1, " 倍",
                                          "0＝依臉的大小自動決定（約 1.35–2.4 倍）。", special="自動"))
        body.addRow("影格率", self.spin("fps", 0.0, 240.0, 1.0, 2, " fps", "0＝跟來源一樣。", special="同來源"))
        body.addRow("", self.check("use_speech", "用語音辨識取得逐字時間（剪水詞、剪點更準）",
                                   "需要安裝語音辨識；沒有時仍可用字幕檔，但水詞只能剪整句都是水詞的字幕。"))
        lay.addWidget(self.advanced)
        lay.addStretch(1)
        self.set_params(self.PARAMS())

    def wants_speech(self) -> bool:
        return self.caption_mode() == "auto" or bool(self._get["use_speech"]())

    def _sync(self) -> None:
        hl = self._get["mode"]() == "highlights"
        self.w["keep_ratio"].setEnabled(hl)
        self.w["target_seconds"].setEnabled(hl)
        zoom = bool(self._get["speaker_zoom"]())
        for k in ("zoom_min_turn", "zoom_max", "zoom_back_min", "zoom_level"):
            self.w[k].setEnabled(zoom)
        self.w["pause_min"].setEnabled(bool(self._get["tighten"]()))
        self.w["pause_keep"].setEnabled(bool(self._get["tighten"]()))
        self._sync_captions(bool(self._get["use_speech"]()))
        self._sync_ai()
        self._sync_note()

    def validate(self) -> str:
        bad = super().validate()
        if bad:
            return bad
        if float(self._get["zoom_max"]()) < float(self._get["zoom_min_turn"]()):
            return "「最長特寫」不能短於「觸發長度」。"
        return ""
