"""Optional speech recognition with faster-whisper (spec §2.4).

Nothing here imports ``faster_whisper`` (or ``opencc``) at module level: without them every other feature keeps working
and :func:`speech_available` explains how to install.  A model is NEVER downloaded unless ``allow_download=True``; the
GUI asks the user first and retries with it.  All user-facing messages are 繁體中文.
"""
from __future__ import annotations

import importlib
import importlib.util
import threading
from typing import Any, Callable, Iterable, Optional

import numpy as np

from ..media.ffmpeg import Cancelled
from .audio import stream_audio

INSTALL_HINT = "尚未安裝語音辨識：請執行 pip install -r requirements-speech.txt"
SAMPLE_RATE = 16000
APPROX_MB = {"tiny": 75, "base": 145, "small": 484, "medium": 1500, "large-v1": 3100, "large-v2": 3100,
             "large-v3": 3100, "large": 3100}
_MODELS: dict[str, Any] = {}              # the last loaded WhisperModel (reloading costs seconds)
_LOCK = threading.Lock()


class SpeechUnavailable(RuntimeError):
    """Speech recognition cannot run right now; the message is 繁體中文 and can be shown to the user as is."""


class Transcript(list):
    """A plain list of segment dicts that also carries ``notes`` (e.g. 「無法轉成繁體」)."""

    def __init__(self, items: Iterable = (), notes: Optional[list] = None) -> None:
        super().__init__(items)
        self.notes: list[str] = list(notes or [])


def speech_available() -> tuple[bool, str]:
    """``(usable, reason)``: faster-whisper must import; ``reason`` is the install hint when it does not."""
    try:
        if importlib.util.find_spec("faster_whisper") is None:
            return False, INSTALL_HINT
        importlib.import_module("faster_whisper")
    except Exception as exc:                                 # broken install (missing DLL, bad dependency ...)
        return False, f"{INSTALL_HINT}（匯入失敗：{exc}）"
    return True, ""


def _approx_mb(model_size: str) -> int:
    return APPROX_MB.get(model_size, 500)


def _missing_model_message(model_size: str) -> str:
    return (f"尚未下載語音模型「{model_size}」（約 {_approx_mb(model_size)} MB）。第一次使用需要連網下載一次，"
            "之後可離線使用；素材不會上傳。")


def model_cached(model_size: str = "small") -> bool:
    """True when the model is already on disk (checked without loading it and without any network access)."""
    if not speech_available()[0]:
        return False
    try:
        from faster_whisper.utils import download_model
    except Exception:
        download_model = None
    if download_model is not None:
        try:
            download_model(model_size, local_files_only=True)
            return True
        except Exception:
            return False
    try:                                                     # older faster-whisper: try to load it from disk only
        _load_model(model_size, allow_download=False)
        return True
    except SpeechUnavailable:
        return False


def _load_model(model_size: str, allow_download: bool):
    from faster_whisper import WhisperModel
    with _LOCK:
        model = _MODELS.get(model_size)
        if model is not None:
            return model
        try:
            model = WhisperModel(model_size, device="auto", compute_type="int8", local_files_only=not allow_download)
        except Exception as exc:
            if not allow_download:
                raise SpeechUnavailable(_missing_model_message(model_size)) from exc
            raise SpeechUnavailable(f"無法下載或載入語音模型「{model_size}」：{exc}（請確認網路連線與磁碟空間）") from exc
        _MODELS.clear()
        _MODELS[model_size] = model
        return model


def _converter(language: Optional[str], traditional: bool) -> tuple[Optional[Callable[[str], str]], Optional[str]]:
    """``(convert, note)``: Simplified -> Traditional (Taiwan wording, ``s2twp``) for Chinese when asked and possible."""
    if not traditional or not language or not language.lower().startswith("zh"):
        return None, None
    try:
        from opencc import OpenCC
        cc = OpenCC("s2twp")
        return cc.convert, None
    except Exception:
        return None, "未安裝 opencc，無法把簡體字轉成繁體；字幕可能含簡體字（pip install -r requirements-speech.txt）"


def segments_to_dicts(segments: Iterable, *, offset: float = 0.0, convert: Optional[Callable[[str], str]] = None,
                      total: float = 0.0, progress: Optional[Callable[[float], None]] = None,
                      cancel=None) -> list[dict]:
    """faster-whisper ``Segment`` objects -> ``[{"start","end","text","words":[{"start","end","word"}]}]``.

    Times get ``offset`` added (so they are asset times); empty segments are dropped; ``convert`` is applied to the
    text and to every word.  ``progress(fraction)`` follows the segment end times (``total`` seconds of audio).
    """
    conv = convert or (lambda s: s)
    out: list[dict] = []
    for seg in segments:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        text = conv(str(getattr(seg, "text", "") or "").strip())
        end = float(getattr(seg, "end", 0.0) or 0.0)
        if progress is not None and total > 0:
            progress(min(end / total, 1.0))
        if not text:
            continue
        words = [{"start": round(float(w.start) + offset, 3), "end": round(float(w.end) + offset, 3),
                  "word": conv(str(w.word))} for w in (getattr(seg, "words", None) or [])]
        out.append({"start": round(float(seg.start) + offset, 3), "end": round(end + offset, 3),
                    "text": text, "words": words})
    return out


def _decode(path, start: float, duration: Optional[float], cancel) -> np.ndarray:
    blocks = list(stream_audio(path, SAMPLE_RATE, cancel, start=start, duration=duration, block_seconds=10.0))
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def transcribe(path, *, language: Optional[str] = "zh", model_size: str = "small", traditional: bool = True,
               allow_download: bool = False, start: float = 0.0, duration: Optional[float] = None,
               progress: Optional[Callable[[float], None]] = None, cancel=None) -> list[dict]:
    """Word-timestamped transcript of ``path[start : start + duration]`` (asset times; ``progress(0..1)``).

    ``language=None`` auto-detects.  Raises :class:`SpeechUnavailable` (not installed / model not downloaded and
    ``allow_download`` is False / model cannot be loaded) or :class:`Cancelled`.  The result is a list of segment
    dicts (a :class:`Transcript`, whose ``notes`` say when Traditional conversion was not possible).
    """
    ok, reason = speech_available()
    if not ok:
        raise SpeechUnavailable(reason)
    if not allow_download and not model_cached(model_size):
        raise SpeechUnavailable(_missing_model_message(model_size))
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    model = _load_model(model_size, allow_download)
    audio = _decode(path, max(float(start), 0.0), duration, cancel)
    if audio.size == 0:
        return Transcript()
    convert, note = _converter(language, traditional)
    segments, _info = model.transcribe(audio, language=language, vad_filter=True, word_timestamps=True)
    items = segments_to_dicts(segments, offset=max(float(start), 0.0), convert=convert,
                              total=audio.size / SAMPLE_RATE, progress=progress, cancel=cancel)
    if progress is not None:
        progress(1.0)
    return Transcript(items, [note] if note else [])
