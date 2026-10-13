"""「一鍵字幕」: transcribe a whole video and save an ``.srt`` next to it (``<video name>.srt``).

The file is ordinary SRT, so the user can fix recognition errors in any editor before cutting; the planners pick a
same-name subtitle file up automatically (:func:`autocut.autoedit.highlights.choose_transcript`), so "make subtitles,
then auto-edit" needs no extra setting.  Needs the optional speech recognition (faster-whisper); raises
:class:`~autocut.analysis.speech.SpeechUnavailable` with a 繁中 message otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from ..core.subtitles import to_srt
from ..util import atomic_write_text
from .captions import Piece, cues_for_pieces

MAX_CHARS = 16


def sidecar_path(video_path: str | Path, ext: str = ".srt") -> Path:
    p = Path(video_path)
    return p.with_name(p.stem + ext)


def make_subtitle_file(video_path: str | Path, out_path: Optional[str | Path] = None, *, language: str = "zh",
                       traditional: bool = True, model_size: str = "small", allow_download: bool = False,
                       max_chars: int = MAX_CHARS, overwrite: bool = False,
                       progress: Optional[Callable[[float, str], None]] = None, cancel=None) -> tuple[Path, int]:
    """Transcribe ``video_path`` and write SRT (default: ``<video>.srt``).  Returns ``(path, number of cues)``.

    Next to the SRT, ``<name>.words.json`` keeps the word timings (filler-word removal needs them; the SRT cannot hold
    them).  Raises ``FileExistsError`` when the target exists and ``overwrite`` is False, ``SpeechUnavailable`` when
    speech recognition cannot run, ``Cancelled`` on cancel."""
    from ..analysis import speech as _speech
    src = Path(video_path)
    out = Path(out_path) if out_path else sidecar_path(src)
    if out.exists() and not overwrite:
        raise FileExistsError(f"字幕檔已存在：{out.name}（要覆蓋請先確認）")
    cb = (lambda f: progress(min(0.98, f), "語音辨識…")) if progress else None
    segs = _speech.transcribe(str(src), language=language, model_size=model_size, traditional=traditional,
                              allow_download=allow_download, progress=cb, cancel=cancel)
    end = max((float(s["end"]) for s in segs), default=0.0) + 1.0
    cues = cues_for_pieces([dict(s) for s in segs], [Piece("v", 0.0, end, 0.0, 1.0)], max_chars=max_chars)
    atomic_write_text(out, to_srt(cues))
    words = [s for s in segs if s.get("words")]
    if words:
        atomic_write_text(out.with_name(out.stem + ".words.json"),
                          json.dumps({"schema": "autocut.words/1", "segments": words}, ensure_ascii=False))
    if progress:
        progress(1.0, f"已存成 {out.name}（{len(cues)} 條）")
    return out, len(cues)
