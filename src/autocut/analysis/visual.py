"""Picture evidence from ONE ffmpeg pass (spec §2.3).

``ffmpeg -i in -vf fps=F:round=up,scale=W:H:flags=area,setsar=1 -pix_fmt bgr24 -f rawvideo -`` is read in exact frame-size
chunks (W = ``frame_width``, H follows the *display* aspect, so rotation and pixel aspect are already applied).
``round=up`` matters: with ffmpeg's default ``round=near`` sample ``i`` would show the frame ~0.08 s *after* ``i/F``; with
``up`` it is the frame displayed at ``t = i / visual_fps``, which is what :class:`Evidence` documents.  Per sample ``i`` we
compute

* ``bright``  mean luma 0..1
* ``sharp``   variance of the Laplacian (raw; planners normalise by the asset's median)
* ``motion``  mean |grey diff to the previous sample| / 255 (sample 0 -> 0)
* ``motion_cx`` / ``motion_cy``  centroid (0..1) of the pixels that changed by more than ``PIX_THR`` levels, or -1 when
  fewer than ``MOTION_MIN_FRAC`` of the pixels changed

and hand the grey frame to :mod:`.scenes` (cuts) and, every ``1/face_fps`` seconds, to :mod:`.faces`.  A step that the
scene detector calls a cut is not subject motion, so on the first sample after a cut ``motion`` is replaced by the
mean of its neighbours and the centroid is set to -1 (the raw jump would otherwise look like a shaky window).

Memory is one frame plus a few scalars per sample, whatever the file length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ..media.ffmpeg import Cancelled, FFmpegError
from . import faces as _faces
from .audio import FFmpegStream, ffmpeg_input
from .scenes import SceneDetector

PIX_THR = 24                # a grey level change above this counts as "changed" (0..255)
MOTION_MIN_FRAC = 0.004     # fewer changed pixels than this fraction -> no motion centroid
MIN_FRAME_WIDTH = 16


@dataclass
class PictureFeatures:
    visual_fps: float
    size: tuple = (0, 0)                               # (width, height) of the analysis frames
    bright: list = field(default_factory=list)
    sharp: list = field(default_factory=list)
    motion: list = field(default_factory=list)
    motion_cx: list = field(default_factory=list)
    motion_cy: list = field(default_factory=list)
    faces: list = field(default_factory=list)          # per sample [[x, y, w, h], ...]
    face_sample_mask: list = field(default_factory=list)
    scenes: list = field(default_factory=list)         # cut times (s)
    cut_samples: list = field(default_factory=list)    # sample index right after each cut
    detector_ok: bool = True                           # False: the face cascade could not be loaded
    faces_run: int = 0                                 # samples on which face detection ran
    faces_found: int = 0                               # ... and found at least one face
    complete: bool = True                              # False: the stream ended with an error after some frames

    @property
    def n(self) -> int:
        return len(self.motion)


def sample_size(display_w: int, display_h: int, frame_width: int = 256) -> tuple[int, int]:
    """Analysis frame size ``(W, H)``: width ``frame_width``, height from the display aspect (16:9 if unknown)."""
    w = max(int(frame_width), MIN_FRAME_WIDTH)
    if display_w <= 0 or display_h <= 0:
        return w, max(int(round(w * 9 / 16)), 8)
    return w, min(max(int(round(w * display_h / display_w)), 8), 4 * w)


class SampleAnalyzer:
    """Per-sample picture maths on BGR frames (or grey frames); no ffmpeg involved, so it is unit-testable."""

    def __init__(self, visual_fps: float = 5.0, face_fps: float = 2.0, faces: bool = True,
                 detect: Optional[Callable] = None) -> None:
        self.fps = float(visual_fps)
        self._detect = detect                           # None -> faces.detect_faces looked up at call time
        self._face_on = bool(faces) and (detect is not None or _faces.faces_available())
        self._detector_ok = (not faces) or detect is not None or _faces.faces_available()
        self._face_dt = 1.0 / max(min(float(face_fps), self.fps), 1e-6)
        self._next_face = 0.0
        self._prev: Optional[np.ndarray] = None
        self._scenes = SceneDetector(self.fps)
        self._size = (0, 0)
        self.bright: list[float] = []
        self.sharp: list[float] = []
        self.motion: list[float] = []
        self.cx: list[float] = []
        self.cy: list[float] = []
        self.faces: list[list] = []
        self.mask: list[int] = []
        self.faces_run = 0
        self.faces_found = 0

    @property
    def n(self) -> int:
        return len(self.motion)

    def push(self, frame: np.ndarray) -> None:
        i = len(self.motion)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = grey.shape
        self._size = (w, h)
        self.bright.append(round(cv2.mean(grey)[0] / 255.0, 4))
        _, sd = cv2.meanStdDev(cv2.Laplacian(grey, cv2.CV_32F))
        self.sharp.append(round(float(sd[0][0]) ** 2, 2))
        motion, cx, cy = 0.0, -1.0, -1.0
        if self._prev is not None:
            diff = cv2.absdiff(grey, self._prev)
            motion = cv2.mean(diff)[0] / 255.0
            _, binary = cv2.threshold(diff, PIX_THR, 1, cv2.THRESH_BINARY)
            m = cv2.moments(binary, binaryImage=True)
            if m["m00"] >= MOTION_MIN_FRAC * w * h:
                cx, cy = (m["m10"] / m["m00"] + 0.5) / w, (m["m01"] / m["m00"] + 0.5) / h
        self.motion.append(round(motion, 4))
        self.cx.append(round(cx, 3))
        self.cy.append(round(cy, 3))
        self._scenes.push(grey, motion)
        self._prev = grey
        boxes: list = []
        ran = 0
        t = i / self.fps
        if self._face_on and t + 1e-9 >= self._next_face:
            ran = 1
            while self._next_face <= t + 1e-9:
                self._next_face += self._face_dt
            try:
                boxes = (self._detect or _faces.detect_faces)(grey) or []
            except Exception:
                boxes = []
            self.faces_run += 1
            self.faces_found += 1 if boxes else 0
        self.faces.append([[float(v) for v in b] for b in boxes])
        self.mask.append(ran)

    def finish(self, complete: bool = True) -> PictureFeatures:
        cut_samples = self._scenes.cut_indices()
        motion, cx, cy = list(self.motion), list(self.cx), list(self.cy)
        skip = set(cut_samples)
        for p in cut_samples:
            near = [motion[q] for q in (p - 1, p + 1) if 1 <= q < len(motion) and q not in skip]
            motion[p] = round(sum(near) / len(near), 4) if near else 0.0
            cx[p] = cy[p] = -1.0
        return PictureFeatures(
            visual_fps=self.fps, size=self._size, bright=self.bright, sharp=self.sharp, motion=motion,
            motion_cx=cx, motion_cy=cy, faces=self.faces, face_sample_mask=self.mask,
            scenes=[round((p - 0.5) / self.fps, 3) for p in cut_samples], cut_samples=cut_samples,
            detector_ok=self._detector_ok, faces_run=self.faces_run, faces_found=self.faces_found, complete=complete)


def analyze_picture(path, *, display_size: tuple[int, int], duration: float = 0.0, visual_fps: float = 5.0,
                    frame_width: int = 256, face_fps: float = 2.0, faces: bool = True, still: bool = False,
                    cancel=None, progress: Optional[Callable[[float], None]] = None) -> PictureFeatures:
    """Decode ``path`` once and return its :class:`PictureFeatures`.

    ``display_size`` is ``(width, height)`` as displayed (rotation / pixel aspect applied).  ``progress(fraction)`` is
    frames read / (duration * visual_fps).  Raises :class:`Cancelled` (child killed) or :class:`FFmpegError` when not a
    single frame could be decoded; a stream that fails after some frames returns what it had with ``complete=False``.
    ``still=True`` reads exactly one frame (a photo).
    """
    w, h = sample_size(display_size[0], display_size[1], frame_width)
    ana = SampleAnalyzer(visual_fps, face_fps, faces)
    cwd, name = ffmpeg_input(path)
    args = ["-i", name, "-an", "-sn", "-dn", "-vf", f"fps={visual_fps:g}:round=up,scale={w}:{h}:flags=area,setsar=1",
            "-pix_fmt", "bgr24"]
    if still:
        args += ["-frames:v", "1"]
    args += ["-f", "rawvideo", "-"]
    size = w * h * 3
    expect = duration * visual_fps if duration > 0 else 0.0
    error: Optional[FFmpegError] = None
    with FFmpegStream(args, cancel=cancel, cwd=cwd) as st:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            buf = st.read(size)
            if len(buf) < size:
                break
            ana.push(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
            if progress is not None and expect > 0:
                progress(min(ana.n / expect, 0.999))
        try:
            st.finish()
        except FFmpegError as exc:                      # Cancelled passes through untouched
            error = exc
    if error is not None and ana.n == 0:
        raise error
    return ana.finish(complete=error is None)
