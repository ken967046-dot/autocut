"""``analyze_asset``: sound + picture (+ optional speech) -> one cached :class:`Evidence` (spec §2.1).

* Cache: ``evidence.cache_path(asset.fingerprint or hash(path, size, mtime), params)``.  The cached file never holds
  speech (speech has its own cache, keyed by file + language + model + script), so toggling ``params.speech`` does not
  redo the sound/picture pass.
* Graceful: audio-only files, still images, video without sound, tiny files, a stream that dies half way (partial
  evidence + a note).  Missing / empty / unreadable files raise ``RuntimeError`` with a 繁體中文 message; nothing hangs
  (ffmpeg children have a watchdog) and nothing is left running after :class:`Cancelled`.
* ``progress(fraction, message)`` is 0..1 for THIS asset; ``cancel`` is a ``threading.Event``.
* Memory is bounded: one audio block and one analysis frame at a time, whatever the file length.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from ..media.ffmpeg import Cancelled, FFmpegError, FFmpegMissing
from ..util import atomic_write_text, cache_dir, sha256_hex
from . import speech as _speech
from .audio import analyze_audio_file
from .evidence import EVIDENCE_VERSION, AnalyzeParams, Evidence, cache_path
from .scenes import refine_cut                                      # noqa: F401  (re-exported)
from .visual import analyze_picture

__all__ = ["analyze_asset", "refine_cut"]

ProgressCb = Callable[[float, str], None]
WEIGHT_AUDIO, WEIGHT_PICTURE, WEIGHT_SPEECH = 1.0, 8.0, 25.0         # relative cost, used to split 0..1 over the phases
NOTE_NO_FACES = "未偵測人臉，取景會改用動作重心"


class _Reporter:
    """Maps a phase's 0..1 onto the asset's 0..1 and throttles the callback."""

    def __init__(self, cb: Optional[ProgressCb]) -> None:
        self._cb, self._f, self._msg, self._t = cb, -1.0, "", 0.0

    def __call__(self, fraction: float, message: str) -> None:
        if self._cb is None:
            return
        f = min(max(fraction, 0.0), 1.0)
        now = time.monotonic()
        if message != self._msg or f >= 1.0 or (f - self._f >= 0.005 and now - self._t >= 0.05):
            self._f, self._msg, self._t = f, message, now
            self._cb(f, message)

    def phase(self, lo: float, hi: float, label: str) -> Callable[[float], None]:
        """A ``fn(phase_fraction)`` reporting ``label`` and ``label NN%`` inside ``[lo, hi]``."""
        def fn(x: float) -> None:
            x = min(max(x, 0.0), 1.0)
            pct = int(x * 100)
            self(lo + (hi - lo) * x, f"{label} {pct}%" if pct else f"{label}…")
        return fn


def _validate(p: AnalyzeParams) -> None:
    if p.audio_fps <= 0 or p.audio_sr < 4000 or p.visual_fps <= 0 or p.face_fps <= 0 or p.frame_width < 16:
        raise ValueError("分析參數不正確：音訊／畫面取樣率必須大於 0，取樣畫面寬度至少 16 像素。")


def _check_file(asset) -> Path:
    p = Path(asset.path)
    if not p.is_file():
        raise RuntimeError(f"找不到素材檔案：{p.name}（可能已被移動、改名或刪除）")
    if p.stat().st_size == 0:
        raise RuntimeError(f"素材檔案是空的（0 位元組），無法分析：{p.name}")
    if not os.access(p, os.R_OK):
        raise RuntimeError(f"沒有權限讀取素材檔案：{p.name}")
    return p


def _file_key(asset, p: Path) -> str:
    if getattr(asset, "fingerprint", ""):
        return str(asset.fingerprint)
    st = p.stat()
    return sha256_hex(f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}")[:20]


def _unreadable(asset, exc: Exception) -> RuntimeError:
    name = os.path.basename(str(asset.path))
    err = RuntimeError(f"無法讀取這個素材（檔案可能已損毀、尚未下載完成，或格式不支援）：{name}")
    err.__cause__ = exc
    return err


def _display_size(asset, p: Path) -> tuple[int, int]:
    w, h = int(getattr(asset, "width", 0) or 0), int(getattr(asset, "height", 0) or 0)
    if w > 0 and h > 0:
        return w, h
    try:                                                              # asset without size: ask ffprobe once
        from ..media.probe import probe_asset
        a = probe_asset(p)
        return int(a.width), int(a.height)
    except Exception:
        return 0, 0                                                   # sample_size() falls back to 16:9


def _compute_base(asset, params: AnalyzeParams, key: str, p: Path, rep: _Reporter, lo: float, hi: float,
                  cancel) -> Evidence:
    """Sound + picture pass (no speech)."""
    kind = getattr(asset, "kind", "video")
    do_audio = kind == "audio" or bool(getattr(asset, "has_audio", False))
    do_picture = kind in ("video", "image") and bool(getattr(asset, "has_video", True))
    if not (do_audio or do_picture):
        raise RuntimeError("這個素材沒有可分析的影像或聲音。")
    duration = float(getattr(asset, "duration", 0.0) or 0.0)
    wa, wp = (WEIGHT_AUDIO if do_audio else 0.0), (WEIGHT_PICTURE if do_picture else 0.0)
    mid = lo + (hi - lo) * wa / (wa + wp)
    notes: list[str] = []

    audio = None
    if do_audio:
        rep(lo, "分析聲音…")
        fn = rep.phase(lo, mid, "分析聲音")
        try:
            audio = analyze_audio_file(p, sr=params.audio_sr, audio_fps=params.audio_fps, silence_db=params.silence_db,
                                       cancel=cancel, progress=(lambda sec: fn(sec / duration)) if duration > 0 else None)
        except (Cancelled, FFmpegMissing):
            raise
        except FFmpegError as exc:
            if not do_picture:
                raise _unreadable(asset, exc) from exc
            notes.append("無法讀取這個素材的聲音，只分析了畫面。")
        if audio is not None and not audio.rms_db:
            if not do_picture:
                raise RuntimeError(f"這個素材沒有可分析的聲音：{p.name}")
            audio = None
            notes.append("這個素材沒有聲音，無法用聲音判斷人聲與靜音。")

    pic = None
    if do_picture:
        rep(mid, "分析畫面…")
        fn = rep.phase(mid, hi, "分析畫面")
        try:
            pic = analyze_picture(p, display_size=_display_size(asset, p), duration=duration, visual_fps=params.visual_fps,
                                  frame_width=params.frame_width, face_fps=params.face_fps, faces=params.faces,
                                  still=kind == "image", cancel=cancel, progress=fn)
        except (Cancelled, FFmpegMissing):
            raise
        except FFmpegError as exc:
            if audio is None:
                raise _unreadable(asset, exc) from exc
            notes.append("無法讀取這個素材的畫面，只分析了聲音。")
        if pic is not None and pic.n == 0:
            if audio is None:
                raise RuntimeError(f"這個素材沒有可分析的畫面：{p.name}")
            pic = None
            notes.append("這個素材沒有可讀取的畫面，只分析了聲音。")

    ev = Evidence(asset_id=asset.id, path=str(p), fingerprint=str(getattr(asset, "fingerprint", "") or key),
                  duration=duration, width=int(getattr(asset, "width", 0) or 0),
                  height=int(getattr(asset, "height", 0) or 0), fps=float(getattr(asset, "fps", 0.0) or 0.0),
                  has_audio=audio is not None, has_video=pic is not None, params=params.to_dict(),
                  audio_fps=float(params.audio_fps), visual_fps=float(params.visual_fps))
    if audio is not None:
        ev.rms_db, ev.voice, ev.silences = audio.rms_db, audio.voice, audio.silences
        ev.noise_floor_db, ev.silence_thr_db, ev.voice_thr_db = audio.noise_floor_db, audio.silence_thr_db, audio.voice_thr_db
        ev.duration = duration or audio.duration
    if pic is not None:
        ev.motion, ev.sharp, ev.bright = pic.motion, pic.sharp, pic.bright
        ev.motion_cx, ev.motion_cy = pic.motion_cx, pic.motion_cy
        ev.faces, ev.face_sample_mask, ev.scenes = pic.faces, pic.face_sample_mask, pic.scenes
        ev.duration = ev.duration or pic.n / params.visual_fps
        if not params.faces:
            notes.append("已停用人臉偵測，取景會改用動作重心。")
        elif not pic.detector_ok:
            notes.append(f"{NOTE_NO_FACES}（人臉偵測器無法使用）。")
        elif pic.faces_found == 0:
            notes.append(f"{NOTE_NO_FACES}（已檢查 {pic.faces_run} 個取樣，沒有找到正臉）。")
        if not pic.complete:
            notes.append(f"畫面解碼中途失敗，檔案可能不完整；只分析到約 {pic.n / params.visual_fps:.0f} 秒。")
    covered = max(audio.duration if audio is not None else 0.0, pic.n / params.visual_fps if pic is not None else 0.0)
    if kind != "image" and duration > 0 and covered < duration - max(1.0, 0.03 * duration):
        # A cut-off download / broken index: ffmpeg reads what is there and exits cleanly, so nothing above notices.  The
        # planners must not propose moments that have no evidence (and no picture), so the evidence ends where the data ends.
        if pic is None or pic.complete:
            notes.append(f"素材可能不完整：檔案標示 {duration:.0f} 秒，但只讀得到約 {covered:.0f} 秒；後面的部分沒有分析，也不會被剪進去。")
        ev.duration = round(covered, 3)
    ev.notes = notes
    return ev


# ----------------------------------------------------------------------------- cache
def _numbers(seq) -> bool:
    return isinstance(seq, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) and v == v for v in seq)


def _well_formed(ev: Evidence, params: AnalyzeParams) -> bool:
    """A cache file that parses as JSON can still be damaged or written by another build: check the shape before
    trusting it (every downstream planner indexes these lists)."""
    try:
        if not all(_numbers(getattr(ev, f)) for f in ("rms_db", "voice", "motion", "sharp", "bright", "scenes",
                                                       "motion_cx", "motion_cy", "face_sample_mask")):
            return False
        if not (isinstance(ev.silences, list) and all(isinstance(s, list) and len(s) == 2 and _numbers(s)
                                                      for s in ev.silences)):
            return False
        if not (isinstance(ev.faces, list) and all(isinstance(f, list) and all(isinstance(b, list) and len(b) == 4
                                                                                 and _numbers(b) for b in f)
                                                   for f in ev.faces)):
            return False
        if not isinstance(ev.notes, list) or not isinstance(ev.params, dict):
            return False
        if not all(isinstance(getattr(ev, f), (int, float)) and not isinstance(getattr(ev, f), bool)
                   for f in ("duration", "audio_fps", "visual_fps", "noise_floor_db", "silence_thr_db", "voice_thr_db")):
            return False
        n = len(ev.motion)
        return (len(ev.voice) == len(ev.rms_db) and len(ev.sharp) == len(ev.bright) == len(ev.faces)
                == len(ev.face_sample_mask) == len(ev.motion_cx) == len(ev.motion_cy) == n
                and ev.has_audio == bool(ev.rms_db) and ev.has_video == (n > 0)
                and abs(ev.audio_fps - params.audio_fps) < 1e-9 and abs(ev.visual_fps - params.visual_fps) < 1e-9)
    except Exception:
        return False


def _read_cache(path: Path, params: AnalyzeParams) -> Optional[Evidence]:
    try:
        ev = Evidence.load(path)
    except (OSError, ValueError, TypeError, AttributeError):          # missing, truncated, not JSON, not an object ...
        return None
    return ev if ev.version == EVIDENCE_VERSION and _well_formed(ev, params) else None


def _speech_cache_file(key: str, params: AnalyzeParams) -> Path:
    lang = "".join(c if c.isalnum() or c in "-_" else "_" for c in (params.language or "auto"))
    model = "".join(c if c.isalnum() or c in "-_." else "_" for c in params.model_size)
    return cache_dir("speech") / f"{key}_{lang}_{model}_{'tw' if params.traditional else 'raw'}.json"


def _read_speech_cache(path: Path) -> Optional[dict]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and isinstance(d.get("segments"), list) else None
    except (OSError, ValueError):
        return None


def _speech_plan(params: AnalyzeParams) -> tuple[bool, str]:
    """``(will_run, why_not)`` for a transcription that is not cached."""
    ok, reason = _speech.speech_available()
    if not ok:
        return False, reason
    if not params.allow_download and not _speech.model_cached(params.model_size):
        return False, _speech._missing_model_message(params.model_size)
    return True, ""


def _add_speech(ev: Evidence, params: AnalyzeParams, key: str, p: Path, rep: _Reporter, lo: float, hi: float,
                cancel, cached: Optional[dict]) -> None:
    """Fill ``ev.speech`` (or leave ``None`` and add a note).  Never raises except :class:`Cancelled`."""
    if cached is not None:
        ev.speech = cached["segments"]
        ev.notes.extend(str(n) for n in cached.get("notes", []))
        return
    fn = rep.phase(lo, hi, "語音辨識")
    rep(lo, "語音辨識…")
    try:
        segs = _speech.transcribe(p, language=params.language, model_size=params.model_size,
                                  traditional=params.traditional, allow_download=params.allow_download,
                                  progress=fn, cancel=cancel)
    except Cancelled:
        raise
    except _speech.SpeechUnavailable as exc:
        ev.notes.append(str(exc))
        return
    except Exception as exc:                                          # a broken install must not break the analysis
        ev.notes.append(f"語音辨識失敗：{exc}")
        return
    notes = list(getattr(segs, "notes", []))
    ev.speech = [dict(s) for s in segs]
    ev.notes.extend(notes)
    try:
        atomic_write_text(_speech_cache_file(key, params),
                          json.dumps({"segments": ev.speech, "notes": notes}, ensure_ascii=False, separators=(",", ":")))
    except OSError:
        pass


def analyze_asset(asset, params: Optional[AnalyzeParams] = None, *, progress: Optional[ProgressCb] = None,
                  cancel=None, use_cache: bool = True) -> Evidence:
    """Everything the automatic editor knows about ``asset``.  See the module docstring for the guarantees."""
    params = params or AnalyzeParams()
    _validate(params)
    rep = _Reporter(progress)
    rep(0.0, "準備分析…")
    p = _check_file(asset)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    key = _file_key(asset, p)
    base_file = cache_path(key, params)

    ev = None
    if use_cache:
        rep(0.0, "讀取分析快取…")
        ev = _read_cache(base_file, params)
    hit = ev is not None
    cached_speech = None
    speech_run, speech_why = False, ""
    if params.speech:
        has_sound = hit and ev.has_audio or (not hit and (getattr(asset, "kind", "") == "audio"
                                                           or bool(getattr(asset, "has_audio", False))))
        if not has_sound:
            speech_why = "素材沒有聲音，無法做語音辨識。"
        else:
            cached_speech = _read_speech_cache(_speech_cache_file(key, params)) if use_cache else None
            if cached_speech is None:
                speech_run, speech_why = _speech_plan(params)

    w_base = 0.0 if hit else WEIGHT_AUDIO + WEIGHT_PICTURE
    w_speech = WEIGHT_SPEECH if speech_run else 0.0
    split = w_base / (w_base + w_speech) if (w_base + w_speech) > 0 else 1.0

    if not hit:
        ev = _compute_base(asset, params, key, p, rep, 0.0, split, cancel)
        if use_cache:
            try:
                ev.save(base_file)
            except OSError:
                pass                                                  # a read-only cache must not fail the analysis
    else:
        ev.asset_id, ev.path = asset.id, str(p)
        ev.fingerprint = str(getattr(asset, "fingerprint", "") or key)
    ev.params = params.to_dict()

    if params.speech:
        if cached_speech is not None or speech_run:
            _add_speech(ev, params, key, p, rep, split, 1.0, cancel, cached_speech)
        else:
            ev.notes.append(speech_why)
    rep(1.0, "分析完成（使用快取）" if hit and not speech_run else "分析完成")
    return ev
