"""ffprobe -> :class:`~autocut.core.project.Asset`."""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

from ..core.project import Asset
from ..util import fingerprint_file, media_kind_by_ext, new_id
from .ffmpeg import FFmpegError, run_probe_json

_IMAGE_CODECS = {"mjpeg", "png", "bmp", "webp", "tiff", "jpeg2000", "gif"}


class ProbeError(RuntimeError):
    pass


def _frac(text: str | None) -> float:
    if not text or text in ("0/0", "N/A"):
        return 0.0
    try:
        f = Fraction(text)
        return float(f)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    rot = 0
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rot = int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    tags = stream.get("tags") or {}
    if not rot and "rotate" in tags:
        try:
            rot = int(float(tags["rotate"]))
        except ValueError:
            pass
    return (-rot) % 360 if rot else 0      # ffprobe reports counter-clockwise; we store clockwise


def probe_asset(path: str | os.PathLike) -> Asset:
    p = Path(path)
    if not p.is_file():
        raise ProbeError(f"找不到檔案：{p}")
    try:
        data = run_probe_json(p)
    except FFmpegError as e:
        raise ProbeError(str(e)) from e
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    vstreams = [s for s in streams if s.get("codec_type") == "video"
                and not (s.get("disposition") or {}).get("attached_pic")]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    ext_kind = media_kind_by_ext(p)
    v = vstreams[0] if vstreams else None
    a = astreams[0] if astreams else None
    if v is None and a is None:
        raise ProbeError(f"這個檔案沒有可用的影像或聲音：{p.name}")

    duration = 0.0
    try:
        duration = float(fmt.get("duration") or 0)
    except ValueError:
        pass
    if not duration:
        for s in (v, a):
            if s and s.get("duration"):
                try:
                    duration = float(s["duration"])
                    break
                except ValueError:
                    pass

    kind = "video"
    if v is None:
        kind = "audio"
    elif ext_kind == "image" or (v.get("codec_name") in _IMAGE_CODECS and a is None
                                 and (duration < 0.5 or v.get("nb_frames") in ("1", 1))):
        kind = "image"

    asset = Asset(id=new_id("a_"), path=str(p.resolve()), name=p.name, kind=kind)
    asset.size_bytes = p.stat().st_size
    asset.fingerprint = fingerprint_file(p)
    try:
        asset.bitrate = int(fmt.get("bit_rate") or 0)
    except ValueError:
        pass
    asset.created = ((fmt.get("tags") or {}).get("creation_time") or "")
    asset.duration = 0.0 if kind == "image" else duration

    if v is not None:
        w, h = int(v.get("width") or 0), int(v.get("height") or 0)
        sar = 1.0
        sar_txt = v.get("sample_aspect_ratio")
        if sar_txt and sar_txt not in ("0:1", "N/A"):
            try:
                sn, sd = sar_txt.split(":")
                if int(sd) > 0 and int(sn) > 0:
                    sar = int(sn) / int(sd)
            except ValueError:
                pass
        rot = _rotation(v)
        dw = int(round(w * sar))
        dh = h
        if rot in (90, 270):
            dw, dh = dh, dw
        asset.width, asset.height, asset.rotation, asset.sar = dw, dh, rot, sar
        asset.fps = _frac(v.get("avg_frame_rate")) or _frac(v.get("r_frame_rate"))
        asset.vcodec = v.get("codec_name", "")
        asset.has_video = True
    else:
        asset.has_video = False
    if a is not None:
        asset.has_audio = True
        asset.acodec = a.get("codec_name", "")
        asset.audio_channels = int(a.get("channels") or 0)
        try:
            asset.sample_rate = int(a.get("sample_rate") or 0)
        except ValueError:
            asset.sample_rate = 0
    if kind != "image" and asset.duration <= 0:
        raise ProbeError(f"讀不到影片長度：{p.name}")
    return asset
