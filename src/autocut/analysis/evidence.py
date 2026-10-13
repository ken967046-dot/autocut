"""Evidence: everything the automatic editor knows about one source file (sound, picture, faces, speech).

Only the *data model* lives here (shared by the analysers and the planners).  The analysis pipeline
(``analyze_asset``) is in :mod:`autocut.analysis.pipeline` and is re-exported from :mod:`autocut.analysis`.

Time grids
    audio  features:  ``audio_fps`` frames per second,   frame i covers  [i/audio_fps, (i+1)/audio_fps)
    visual features:  ``visual_fps`` samples per second, sample i is the picture at   t = i/visual_fps
All coordinates of face boxes are fractions (0..1) of the *display* frame (rotation and pixel aspect applied),
origin top-left: ``[x, y, w, h]``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from ..util import atomic_write_text, cache_dir, canonical_json, sha256_hex

EVIDENCE_VERSION = 1


@dataclass
class AnalyzeParams:
    audio_fps: float = 20.0
    audio_sr: int = 16000
    visual_fps: float = 5.0
    frame_width: int = 256
    face_fps: float = 2.0
    faces: bool = True
    silence_db: float = -38.0
    speech: bool = False
    language: str = "zh"
    model_size: str = "small"
    traditional: bool = True
    allow_download: bool = False

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AnalyzeParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def cache_key(self) -> str:
        """Parameters that change the *numbers* (speech options are keyed separately by the speech cache)."""
        d = self.to_dict()
        for k in ("speech", "language", "model_size", "traditional", "allow_download"):
            d.pop(k, None)
        return sha256_hex(canonical_json(d))[:12]


@dataclass
class Evidence:
    asset_id: str = ""
    path: str = ""
    fingerprint: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_audio: bool = False
    has_video: bool = True
    params: dict = field(default_factory=dict)
    version: int = EVIDENCE_VERSION

    # ---- sound
    audio_fps: float = 20.0
    rms_db: list = field(default_factory=list)          # dBFS per audio frame
    voice: list = field(default_factory=list)           # 0..1 voice-likeness per audio frame
    silences: list = field(default_factory=list)        # [[t0, t1], ...]  (>= 0.3 s below the silence threshold)
    noise_floor_db: float = -90.0
    silence_thr_db: float = -38.0
    voice_thr_db: float = -38.0

    # ---- picture
    visual_fps: float = 5.0
    motion: list = field(default_factory=list)          # mean |diff| to the previous sample, 0..1
    sharp: list = field(default_factory=list)           # variance of Laplacian (raw)
    bright: list = field(default_factory=list)          # mean luma 0..1
    faces: list = field(default_factory=list)           # per sample: [[x, y, w, h], ...]
    face_sample_mask: list = field(default_factory=list)  # 1 where face detection ran on that sample
    scenes: list = field(default_factory=list)          # scene-cut times in seconds
    motion_cx: list = field(default_factory=list)       # centroid (0..1) of the changed pixels per sample, or -1
    motion_cy: list = field(default_factory=list)

    # ---- speech (optional)
    speech: Optional[list] = None                       # [{"start","end","text","words":[{"start","end","word"}]}]
    notes: list = field(default_factory=list)           # human-readable degradations ("未偵測人臉…")

    # ------------------------------------------------------------------ constructors / io
    @classmethod
    def empty(cls, asset_id: str = "a", duration: float = 0.0, **kw: Any) -> "Evidence":
        """Blank evidence (tests build fakes with this and then fill the arrays they care about)."""
        return cls(asset_id=asset_id, duration=duration, **kw)

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["schema"] = "autocut.evidence/1"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))

    def save(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":")))

    @classmethod
    def load(cls, path: str | Path) -> "Evidence":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ index helpers
    def a_index(self, t: float) -> int:
        n = len(self.rms_db)
        return max(0, min(n - 1, int(t * self.audio_fps))) if n else 0

    def v_index(self, t: float) -> int:
        n = len(self.motion)
        return max(0, min(n - 1, int(round(t * self.visual_fps)))) if n else 0

    @property
    def n_audio(self) -> int:
        return len(self.rms_db)

    @property
    def n_visual(self) -> int:
        return len(self.motion)

    @property
    def has_speech(self) -> bool:
        return bool(self.speech)


def cache_path(fingerprint_or_key: str, params: AnalyzeParams) -> Path:
    """``<cache>/evidence/<key>_v<EVIDENCE_VERSION>_<params>.json`` – the analysers' cache file."""
    return cache_dir("evidence") / f"{fingerprint_or_key}_v{EVIDENCE_VERSION}_{params.cache_key()}.json"
