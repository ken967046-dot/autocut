"""Export presets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    max_side: int | None          # cap for the longer side (never upscales)
    crf: int
    x264_preset: str
    audio_bitrate: str
    loudness: float | None        # integrated LUFS target (None: leave levels alone)
    scale_flags: str = "lanczos"


PRESETS: dict[str, Preset] = {
    "youtube_1080p": Preset("youtube_1080p", "YouTube 1080p（高品質）", "H.264 + AAC，響度 -14 LUFS，最長邊最大 1920", 1920,
                            18, "medium", "192k", -14.0),
    "shorts_1080x1920": Preset("shorts_1080x1920", "短影音 9:16（Shorts / Reels / TikTok）",
                               "H.264 + AAC，響度 -14 LUFS，最長邊最大 1920", 1920, 19, "medium", "192k", -14.0),
    "source_quality": Preset("source_quality", "原尺寸高畫質", "不縮小、CRF 16、不調整響度", None, 16, "slow", "256k", None),
    "draft_fast": Preset("draft_fast", "快速草稿", "CRF 23、veryfast，最長邊最大 1280", 1280, 23, "veryfast", "128k", None,
                         "bilinear"),
    "preview_540p": Preset("preview_540p", "預覽（低解析）", "給程式內預覽用，最長邊 960", 960, 28, "ultrafast", "96k", None,
                           "bilinear"),
}


def get_preset(key: str) -> Preset:
    try:
        return PRESETS[key]
    except KeyError:
        raise KeyError(f"未知的輸出預設：{key}（可用：{', '.join(PRESETS)}）") from None


def output_size(width: int, height: int, preset: Preset) -> tuple[int, int]:
    """Preset-limited output size (even numbers, aspect preserved, never upscaled)."""
    long_side = max(width, height)
    if preset.max_side and long_side > preset.max_side:
        s = preset.max_side / long_side
        width, height = width * s, height * s
    return max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)
