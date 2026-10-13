"""Parameters, platform rules and audit thresholds of the automatic editor.

Numeric thresholds borrowed from video-autopilot-kit (MIT, (c) 2026 Hao0321 Studio; ``shorts_gate.py``) are
*example calibration values*, exactly as upstream calls them: they are defaults you can change, not laws.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

# platform -> duration band. ``deadzone`` follows upstream's example calibration for YouTube Shorts
# (13-25 s "single beat" shorts; 26-44 s was a weak zone in the upstream sample; 45-60 s = tutorial/demo shorts).
PLATFORM_RULES: dict[str, dict[str, Any]] = {
    "yt_shorts": {"label": "YouTube Shorts", "dur_min": 13.0, "dur_max": 60.0, "deadzone": (25.001, 44.999)},
    "ig_reels": {"label": "Instagram Reels", "dur_min": 13.0, "dur_max": 60.0, "deadzone": None},
    "fb_reels": {"label": "Facebook Reels", "dur_min": 13.0, "dur_max": 60.0, "deadzone": None},
    "tiktok": {"label": "TikTok", "dur_min": 13.0, "dur_max": 60.0, "deadzone": None},
}

ASPECTS: dict[str, tuple[int, int]] = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080),
                                       "4:5": (1080, 1350)}


class _Base:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class ShortsParams(_Base):
    """16:9 long video -> several 9:16 shorts."""
    platform: str = "ig_reels"          # yt_shorts | ig_reels | fb_reels | tiktok  (duration band + audit rules)
    count: int = 3                      # how many shorts to propose
    target_seconds: float = 35.0        # preferred length
    min_seconds: float = 0.0            # 0 = from platform band
    max_seconds: float = 0.0
    width: int = 1080
    height: int = 1920
    fps: float = 30.0
    reframe: str = "face"               # face | motion | center | blur (no crop: whole picture over blurred bg)
    reframe_strength: float = 0.5       # 0 = locked, 1 = very reactive virtual camera
    captions: str = "auto"              # auto (speech recognition when installed) | file | off
    caption_file: str = ""              # .srt/.ass/.vtt for ``captions='file'``
    language: str = "zh"
    traditional_chinese: bool = True    # convert recognised Chinese to Traditional (needs opencc)
    tighten: bool = True                # cut long internal pauses (jump cuts)
    pause_keep: float = 0.35            # seconds of a pause that are kept
    pause_min: float = 0.7              # pauses at least this long get tightened
    silence_db: float = -38.0
    first_cut_max: float = 2.05         # the first cut should come within ~2 s (upstream "2-second rule")
    min_gap_between: float = 8.0        # shorts proposed from the same source start at least this far apart
    music_path: str = ""                # optional background music file
    music_gain_db: float = -20.0
    use_speech: bool = True             # use the transcript to pick sentence-aligned cut points


@dataclass
class AssembleParams(_Base):
    """Many raw clips -> one long video."""
    target_seconds: float = 180.0       # 0 = keep every usable moment
    aspect: str = "auto"                # auto (majority of sources) | 16:9 | 9:16 | 1:1 | 4:5
    fps: float = 0.0                    # 0 = auto (most common source rate, clamped to 24-60)
    order: str = "chronological"        # chronological (creation time, then name) | name | as_given
    pace: str = "medium"                # fast | medium | slow
    transition: str = "crossfade"       # cut | crossfade
    crossfade_seconds: float = 0.4
    drop_bad: bool = True               # drop blurry / dark / shaky / silent-and-static stretches
    keep_audio: bool = True
    fit: str = "blur"                   # mismatched aspect: blur | cover | contain
    music_path: str = ""
    music_gain_db: float = -18.0
    fade_music_out: float = 2.0
    min_shot: float = 1.2
    max_shot: float = 0.0               # 0 = from pace
    captions: str = "off"               # off | auto | file
    caption_file: str = ""
    language: str = "zh"
    traditional_chinese: bool = True


@dataclass
class AuditRules(_Base):
    """Checks run on an edit plan *before* it may touch the project (fail-closed)."""
    strict_platform_bands: bool = False   # True: dead-zone / band violations are errors instead of warnings
    first_cut_max: float = 2.05           # upstream FIRST_CUT_MAX (warn)
    reading_warn_cps: float = 5.0         # characters per second a viewer can read: warn above (upstream SR_WARN)
    reading_fail_cps: float = 7.0         # error above (upstream SR_FAIL)
    caption_rate_warn_per_min: float = 30.0   # fewer caption changes per minute than this = too sparse (warn)
    min_clip_seconds: float = 0.25
    max_total_seconds: float = 4 * 3600.0
    require_sources_exist: bool = True
    max_caption_overlap: float = 0.05
    target_tolerance: float = 0.35        # assemble: |length-target|/target above this = warn
    min_cue_seconds: float = 0.4
    max_cue_chars: int = 40
    black_start_warn: bool = True

    @classmethod
    def for_platform(cls, platform: str) -> "AuditRules":
        return cls()


def resolve_band(p: ShortsParams) -> tuple[float, float]:
    rule = PLATFORM_RULES.get(p.platform, PLATFORM_RULES["ig_reels"])
    lo = p.min_seconds or rule["dur_min"]
    hi = p.max_seconds or rule["dur_max"]
    if hi < lo:
        lo, hi = hi, lo
    return float(lo), float(hi)


def pace_shot_range(pace: str) -> tuple[float, float]:
    """(typical minimum, typical maximum) shot length in seconds."""
    return {"fast": (1.2, 3.0), "medium": (2.0, 5.0), "slow": (3.5, 9.0)}.get(pace, (2.0, 5.0))
