"""Render a sequence: segments (cached, parallel) -> lossless concat -> one audio encode."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..core.project import Project, Sequence
from ..core.subtitles import to_srt
from ..media.ffmpeg import (Cancelled, FFmpegError, FFmpegInfo, _NOWINDOW, detect_hw_encoder, ffmpeg_info,
                            run_ffmpeg)
from ..util import cache_dir, new_id
from .compiler import (Compiled, EncodeSettings, RenderError, Segment, build_segment_args, compile_sequence,
                       segment_ass, segment_signature)
from .presets import Preset, get_preset, output_size

ProgressCb = Callable[[float, str], None]


@dataclass
class RenderSettings:
    preset: str = "youtube_1080p"
    encoder: str = "libx264"            # libx264 | auto | h264_nvenc | h264_qsv | h264_amf
    workers: int = 0                    # 0 = automatic
    burn_captions: bool = True
    write_srt: bool = False
    frame_range: Optional[tuple[float, float]] = None   # timeline seconds (export in/out)
    use_cache: bool = True
    loudness: Optional[float] = None    # override the preset's LUFS target (None = preset default)
    normalize: bool = True              # apply the loudness target at all


@dataclass
class RenderResult:
    path: str
    duration: float
    frames: int
    width: int
    height: int
    fps: float
    segments: int
    cached: int
    elapsed: float
    encoder: str
    ffmpeg: str
    warnings: list = field(default_factory=list)


def default_fontsdir() -> Optional[str]:
    if sys.platform.startswith("win"):
        d = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        return d if os.path.isdir(d) else None
    return None


def auto_workers() -> int:
    n = os.cpu_count() or 2
    return max(1, min(3, n // 2))


def segment_cache_dir() -> Path:
    return cache_dir("render", "segments")


def prune_render_cache(max_bytes: int = 20 * 1024 ** 3) -> int:
    """Delete the least recently used cached segments beyond ``max_bytes``. Returns bytes freed."""
    files = [p for p in segment_cache_dir().glob("seg_*.mov") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    freed = 0
    if total <= max_bytes:
        return 0
    for p in sorted(files, key=lambda x: x.stat().st_mtime):
        try:
            sz = p.stat().st_size
            p.unlink()
            freed += sz
            total -= sz
        except OSError:
            continue
        if total <= max_bytes:
            break
    return freed


class _Either:
    """Cancel flag of ONE encode attempt: the person's cancel OR our own abort.

    A failed attempt must stop its sibling segments without looking like a cancel by the person, otherwise the
    fallback to the CPU encoder after a hardware-encoder failure could never run.  Only ``is_set`` is read by
    ``run_ffmpeg``; ``set`` raises our own flag, never the person's."""

    def __init__(self, user: threading.Event) -> None:
        self.user = user
        self.own = threading.Event()

    def is_set(self) -> bool:
        return self.user.is_set() or self.own.is_set()

    def set(self) -> None:
        self.own.set()


# --------------------------------------------------------------------------- segments
def _write_graph_if_long(args: list[str], graph: str, workdir: Path, info: FFmpegInfo, tag: str) -> list[str]:
    """Very long filter graphs (keyframe-heavy reframing) go through a file to dodge command-line limits."""
    if len(graph) < 12000:
        return args
    gp = workdir / f"graph_{tag}.txt"
    gp.write_text(graph, encoding="utf-8")
    i = args.index("-filter_complex")
    flag = "-/filter_complex" if info.version >= (7, 0) else "-filter_complex_script"
    return args[:i] + [flag, gp.name] + args[i + 2:]


def _render_segment(comp: Compiled, seg: Segment, enc: EncodeSettings, info: FFmpegInfo, sig: str,
                    on_frame: Callable[[int], None], cancel: threading.Event) -> Path:
    segdir = segment_cache_dir()
    final = segdir / f"seg_{sig[:24]}.mov"
    part = segdir / f"seg_{sig[:24]}.partial.mov"
    ass_name = None
    ass_text = segment_ass(comp, seg, enc, info)
    if ass_text:
        ass_name = f"seg_{sig[:24]}.ass"
        (segdir / ass_name).write_text(ass_text, encoding="utf-8")
    args, graph = build_segment_args(comp, seg, enc, info, part.name, ass_name)
    args = _write_graph_if_long(args, graph, segdir, info, sig[:24])
    log = cache_dir("render", "logs") / f"seg_{sig[:24]}.log"
    try:
        run_ffmpeg(["-y", *args], cwd=segdir, on_frame=on_frame, cancel=cancel, log_path=log)
        if not part.exists() or part.stat().st_size == 0:
            raise FFmpegError("片段輸出是空的")
        os.replace(part, final)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    finally:
        if ass_name:
            try:
                (segdir / ass_name).unlink()
            except OSError:
                pass
    return final


# ------------------------------------------------------------------------ final stage
def _measure_loudness(list_path: Path, cwd: Path, target: float, info: FFmpegInfo) -> Optional[dict]:
    cmd = [info.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-f", "concat", "-safe", "0", "-i",
           list_path.name, "-vn", "-af", f"loudnorm=I={target}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"]
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, timeout=3600, creationflags=_NOWINDOW)
    text = p.stderr.decode("utf-8", "replace")
    m = None
    for m in re.finditer(r"\{[^{}]*\"input_i\"[^{}]*\}", text, re.S):
        pass
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        values = [float(d[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
    except (ValueError, KeyError):
        return None
    if not all(math.isfinite(v) for v in values):
        return {"unmeasurable": True}       # digital silence, or shorter than loudnorm's 400 ms block ("-inf")
    return d


def _final_mux(comp: Compiled, seg_paths: list[Path], out_path: Path, preset: Preset, settings: RenderSettings,
               info: FFmpegInfo, cancel: threading.Event, warnings: list[str],
               progress: Optional[ProgressCb]) -> None:
    segdir = segment_cache_dir()
    list_path = segdir / f"concat_{new_id()}.txt"
    list_path.write_text("".join(f"file '{p.name}'\n" for p in seg_paths), encoding="utf-8")
    tmp_out = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    try:
        target = settings.loudness if settings.loudness is not None else preset.loudness
        afilters: list[str] = []
        if settings.normalize and target is not None:
            if progress:
                progress(0.93, "量測響度…")
            m = _measure_loudness(list_path, segdir, target, info)
            if m and m.get("unmeasurable"):
                warnings.append("聲音是靜音或太短，無法量測響度，已略過響度標準化")
            elif m:
                afilters.append(
                    f"loudnorm=I={target}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                    f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
                    "linear=true")
            else:
                warnings.append("響度量測失敗，改用單次 loudnorm")
                afilters.append(f"loudnorm=I={target}:TP=-1.5:LRA=11")
        if "alimiter" in info.filters:
            afilters.append("alimiter=limit=0.97:level=disabled")
        args = ["-y", "-f", "concat", "-safe", "0", "-i", list_path.name, "-map", "0:v:0", "-map", "0:a:0",
                "-c:v", "copy"]
        if afilters:
            args += ["-af", ",".join(afilters)]
        args += ["-c:a", "aac", "-b:a", preset.audio_bitrate, "-ar", "48000", "-movflags", "+faststart",
                 "-t", f"{comp.duration:.6f}", str(tmp_out)]
        if progress:
            progress(0.96, "封裝成品…")
        run_ffmpeg(args, cwd=segdir, cancel=cancel, log_path=cache_dir("render", "logs") / "final.log")
        os.replace(tmp_out, out_path)
    finally:
        for p in (list_path, tmp_out):
            try:
                p.unlink()
            except OSError:
                pass


# -------------------------------------------------------------------------- the API
def render_sequence(proj: Project, seq: Sequence, out_path: str | os.PathLike,
                    settings: Optional[RenderSettings] = None, progress: Optional[ProgressCb] = None,
                    cancel: Optional[threading.Event] = None) -> RenderResult:
    settings = settings or RenderSettings()
    cancel = cancel or threading.Event()
    info = ffmpeg_info()
    preset = get_preset(settings.preset)
    W, H = output_size(seq.width, seq.height, preset)
    t_start = time.time()
    comp = compile_sequence(proj, seq, W, H, settings.frame_range)
    warnings: list[str] = []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prune_render_cache()

    encoder = settings.encoder
    if encoder == "auto":
        encoder = detect_hw_encoder()
    if encoder != "libx264" and not info.has_encoder(encoder):
        warnings.append(f"這台電腦的 ffmpeg 沒有 {encoder}，改用 libx264")
        encoder = "libx264"
    burn = settings.burn_captions
    if burn and (comp.cues or comp.cards) and not info.has_ass:
        warnings.append("這份 ffmpeg 不支援燒入字幕（缺少 libass），已改輸出 .srt 檔"
                        + ("；字卡沒有辦法輸出" if comp.cards else ""))
        burn = False
        settings.write_srt = True

    def attempt(enc_name: str) -> tuple[list[Path], int]:
        enc = EncodeSettings(encoder=enc_name, crf=preset.crf, x264_preset=preset.x264_preset,
                             scale_flags=preset.scale_flags, burn_captions=burn, fontsdir=default_fontsdir())
        segdir = segment_cache_dir()
        sigs = [segment_signature(comp, s, enc) for s in comp.segments]
        paths = [segdir / f"seg_{sg[:24]}.mov" for sg in sigs]
        todo = [i for i, p in enumerate(paths) if not (settings.use_cache and p.exists() and p.stat().st_size > 0)]
        cached = len(paths) - len(todo)
        total_frames = sum(comp.segments[i].n for i in todo) or 1
        done = {"frames": 0, "segs": 0}
        per_seg: dict[int, int] = {}
        lock = threading.Lock()
        stop = _Either(cancel)

        def report() -> None:
            if progress:
                frac = 0.92 * min(1.0, (done["frames"] + sum(per_seg.values())) / total_frames)
                progress(frac, f"渲染片段 {cached + done['segs'] + 1}/{len(paths)}")

        def make_on_frame(i: int) -> Callable[[int], None]:
            def on_frame(n: int) -> None:
                with lock:
                    per_seg[i] = n
                report()
            return on_frame

        def job(i: int) -> int:
            if stop.is_set():
                raise Cancelled()
            _render_segment(comp, comp.segments[i], enc, info, sigs[i], make_on_frame(i), stop)
            with lock:
                done["frames"] += comp.segments[i].n
                done["segs"] += 1
                per_seg.pop(i, None)
            report()
            return i

        if todo:
            workers = settings.workers or auto_workers()
            if enc_name != "libx264":
                workers = min(workers, 2)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(job, i) for i in todo]
                try:
                    for f in as_completed(futs):
                        f.result()
                except BaseException:           # stop the rest quickly (the person's cancel flag stays untouched)
                    stop.set()
                    for f in futs:
                        f.cancel()
                    raise
        return paths, cached

    try:
        try:
            paths, cached = attempt(encoder)
        except FFmpegError as e:
            if cancel.is_set():                 # the person cancelled while the segment was failing
                raise Cancelled()
            if encoder == "libx264":
                raise
            warnings.append(f"{encoder} 編碼失敗，已改用 libx264 重新輸出（{str(e)[:120]}）")
            encoder = "libx264"
            paths, cached = attempt(encoder)
    except Cancelled:
        raise
    if cancel.is_set():
        raise Cancelled()
    _final_mux(comp, paths, out_path, preset, settings, info, cancel, warnings, progress)
    if settings.write_srt and comp.cues:
        t0 = comp.f_start / comp.fps
        length = comp.duration
        cues = [(max(0.0, s - t0), min(e - t0, length), t) for s, e, t in comp.cues if e > t0 and s - t0 < length]
        out_path.with_suffix(".srt").write_text(to_srt(cues), encoding="utf-8")
    if progress:
        progress(1.0, "完成")
    return RenderResult(str(out_path), comp.duration, comp.total_frames, W, H, comp.fps, len(comp.segments), cached,
                        time.time() - t_start, encoder, info.version_text, warnings)


def render_preview(proj: Project, seq: Sequence, progress: Optional[ProgressCb] = None,
                   cancel: Optional[threading.Event] = None,
                   frame_range: Optional[tuple[float, float]] = None) -> RenderResult:
    """Low-resolution render for the in-app player (re-uses cached segments, so repeated previews are fast)."""
    out = cache_dir("render", "preview") / f"preview_{seq.id}.mp4"
    return render_sequence(proj, seq, out, RenderSettings(preset="preview_540p", frame_range=frame_range,
                                                          burn_captions=True, normalize=False),
                           progress, cancel)
