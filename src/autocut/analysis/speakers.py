"""Who is talking?  Face tracks + mouth activity, for the long-video editor's speaker close-ups.

One extra ffmpeg pass at a higher resolution than the main picture analysis (mouths are a few pixels wide on the
256 px analysis frames):

* faces are detected (OpenCV Haar, :mod:`.faces`) every ``detect_every`` samples and linked into **tracks** by position
  -- a podcast / interview camera hardly moves, so "the same place" is "the same person";
* for every track that is visible on a sample, the **mouth activity** is the mean absolute grey change of the lower
  part of the face box (mouth / jaw) between two samples, minus the change of the whole frame (camera moves, cuts and
  lighting changes affect everything equally and cancel out).

:func:`speaker_turns` then combines that with the voice activity of the sound: during speech, the visible person whose
mouth moves clearly more than everybody else's (each track normalised by its own typical activity) is the speaker.

Honest limits: frontal faces only (Haar); a person who talks while looking sideways, covers their mouth, or is off
screen is "unknown"; two people talking at once is "unknown".  This is a heuristic to make a dialogue less static,
not speaker diarisation.  Everything degrades to "no speakers" (and the editor keeps the wide shot) instead of raising.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..media.ffmpeg import Cancelled, FFmpegError
from ..util import atomic_write_text, cache_dir, canonical_json, sha256_hex
from . import faces as _faces

SPEAKER_VERSION = 1
MOUTH_TOP, MOUTH_BOTTOM = 0.62, 1.08       # mouth / jaw band, as fractions of the face box height
MOUTH_LEFT, MOUTH_RIGHT = 0.2, 0.8
HOLD_SAMPLES_S = 1.2                       # a track keeps its last box this long without a new detection
MIN_TRACK_SHARE = 0.08                     # tracks visible on fewer samples than this are dropped (false faces)


@dataclass
class SpeakerParams:
    fps: float = 5.0
    frame_width: int = 640
    detect_every: int = 2                  # face detection on every n-th sample (tracks hold their box in between)
    min_face: float = 0.035                # smallest face width as a fraction of the frame width

    def key(self) -> str:
        return sha256_hex(canonical_json(asdict(self)))[:12]


@dataclass
class SpeakerEvidence:
    asset_id: str = ""
    fingerprint: str = ""
    duration: float = 0.0
    fps: float = 5.0
    tracks: list = field(default_factory=list)      # [{"id", "box": median [x,y,w,h], "share": 0..1}]
    boxes: list = field(default_factory=list)       # per track: per sample [x,y,w,h] or None
    activity: list = field(default_factory=list)    # per track: per sample mouth activity (>= 0) or -1 = not visible
    notes: list = field(default_factory=list)
    version: int = SPEAKER_VERSION

    @property
    def n(self) -> int:
        return len(self.activity[0]) if self.activity else 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["schema"] = "autocut.speakers/1"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerEvidence":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


# ------------------------------------------------------------------------------------------------ tracking
def _center(b) -> tuple[float, float]:
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


class SpeakerAnalyzer:
    """Per-sample maths on grey frames (no ffmpeg involved: unit-testable with synthetic frames and a fake detector)."""

    def __init__(self, params: SpeakerParams, detect: Optional[Callable] = None) -> None:
        self.p = params
        self._detect = detect
        self.tracks: list[dict] = []        # {"last": box, "seen": int, "boxes": [...], "act": [...], "age": samples}
        self.prev: Optional[np.ndarray] = None
        self.i = 0
        self.hold = max(1, int(round(HOLD_SAMPLES_S * params.fps)))

    def _detect_faces(self, grey: np.ndarray) -> list:
        if self._detect is not None:
            return self._detect(grey)
        return _faces.detect_faces(grey, min_frac=self.p.min_face)

    def _match(self, boxes: list) -> None:
        used = set()
        for tr in self.tracks:
            best, bd = None, None
            cx, cy = _center(tr["last"])
            for k, b in enumerate(boxes):
                if k in used:
                    continue
                bx, by = _center(b)
                d = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
                if d <= 0.6 * max(tr["last"][2], b[2]) and (bd is None or d < bd):
                    best, bd = k, d
            if best is not None:
                used.add(best)
                tr["last"], tr["age"] = list(boxes[best]), 0
        for k, b in enumerate(boxes):
            if k not in used:
                self.tracks.append({"last": list(b), "age": 0, "seen": 0,
                                    "boxes": [None] * self.i, "act": [-1.0] * self.i})

    def push(self, grey: np.ndarray) -> None:
        g = np.asarray(grey, dtype=np.float32)
        if g.ndim == 3:
            g = g.mean(axis=2)
        h, w = g.shape
        if self.i % max(1, int(self.p.detect_every)) == 0:
            self._match([list(map(float, b)) for b in self._detect_faces(g.astype(np.uint8))])
        glob = float(np.mean(np.abs(g - self.prev))) if self.prev is not None else 0.0
        for tr in self.tracks:
            visible = tr["age"] <= self.hold
            box = tr["last"] if visible else None
            act = -1.0
            if box is not None:
                x, y, bw, bh = box
                x0, x1 = int((x + MOUTH_LEFT * bw) * w), int((x + MOUTH_RIGHT * bw) * w)
                y0, y1 = int((y + MOUTH_TOP * bh) * h), int(min(y + MOUTH_BOTTOM * bh, 1.0) * h)
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, w), min(y1, h)
                if self.prev is not None and x1 - x0 >= 2 and y1 - y0 >= 2:
                    local = float(np.mean(np.abs(g[y0:y1, x0:x1] - self.prev[y0:y1, x0:x1])))
                    act = max(0.0, local - glob)
                elif self.prev is None:
                    act = 0.0
                tr["seen"] += 1
            tr["boxes"].append([round(v, 4) for v in box] if box is not None else None)
            tr["act"].append(round(act, 4))
            tr["age"] += 1
        self.prev = g
        self.i += 1

    def finish(self, asset_id: str = "", fingerprint: str = "", duration: float = 0.0) -> SpeakerEvidence:
        keep = [t for t in self.tracks if self.i and t["seen"] / self.i >= MIN_TRACK_SHARE]
        keep.sort(key=lambda t: _center(np.median(np.asarray([b for b in t["boxes"] if b], dtype=float), axis=0))[0])
        out = SpeakerEvidence(asset_id=asset_id, fingerprint=fingerprint, duration=duration, fps=self.p.fps)
        for k, t in enumerate(keep):
            med = np.median(np.asarray([b for b in t["boxes"] if b], dtype=float), axis=0)
            out.tracks.append({"id": k, "box": [round(float(v), 4) for v in med], "share": round(t["seen"] / self.i, 4)})
            out.boxes.append(t["boxes"])
            out.activity.append(t["act"])
        if not keep:
            out.notes.append("沒有偵測到穩定出現的人臉，無法判斷誰在說話（講者放大不會啟用）。")
        elif len(keep) == 1:
            out.notes.append("畫面中只偵測到 1 個人，講者放大需要兩個人以上，不會啟用。")
        return out


# ------------------------------------------------------------------------------------------------ the pass
def _cache_file(asset, params: SpeakerParams) -> Path:
    key = getattr(asset, "fingerprint", "") or sha256_hex(str(getattr(asset, "path", "")))[:20]
    return cache_dir("speakers") / f"{key}_v{SPEAKER_VERSION}_{params.key()}.json"


def analyze_speakers(asset, params: Optional[SpeakerParams] = None, *, progress: Optional[Callable[[float], None]] = None,
                     cancel=None, use_cache: bool = True) -> SpeakerEvidence:
    """Face tracks + mouth activity of a video asset (cached).  Raises only :class:`Cancelled`; any decoding problem
    returns empty evidence with a note."""
    from .audio import FFmpegStream, ffmpeg_input
    from .visual import sample_size
    params = params or SpeakerParams()
    path = _cache_file(asset, params)
    if use_cache and path.is_file():
        try:
            return SpeakerEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            pass
    duration = float(getattr(asset, "duration", 0.0) or 0.0)
    if not _faces.faces_available():
        return SpeakerEvidence(asset_id=asset.id, duration=duration, fps=params.fps,
                               notes=["無法載入人臉偵測模型，講者放大不會啟用。"])
    w, h = sample_size(int(getattr(asset, "width", 0) or 0), int(getattr(asset, "height", 0) or 0), params.frame_width)
    ana = SpeakerAnalyzer(params)
    cwd, name = ffmpeg_input(asset.path)
    args = ["-i", name, "-an", "-sn", "-dn", "-vf",
            f"fps={params.fps:g}:round=up,scale={w}:{h}:flags=area,setsar=1,format=gray", "-pix_fmt", "gray",
            "-f", "rawvideo", "-"]
    size = w * h
    expect = duration * params.fps
    try:
        with FFmpegStream(args, cancel=cancel, cwd=cwd) as st:
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                buf = st.read(size)
                if len(buf) < size:
                    break
                ana.push(np.frombuffer(buf, dtype=np.uint8).reshape(h, w))
                if progress is not None and expect > 0:
                    progress(min(ana.i / expect, 0.999))
            try:
                st.finish()
            except FFmpegError:
                pass
    except Cancelled:
        raise
    except (OSError, FFmpegError) as exc:
        return SpeakerEvidence(asset_id=asset.id, duration=duration, fps=params.fps,
                               notes=[f"分析講者時無法讀取影片（{exc}），講者放大不會啟用。"])
    ev = ana.finish(asset.id, getattr(asset, "fingerprint", ""), duration)
    try:
        atomic_write_text(path, json.dumps(ev.to_dict(), ensure_ascii=False, separators=(",", ":")))
    except OSError:
        pass
    return ev


# ------------------------------------------------------------------------------------------------ turns
@dataclass
class Turn:
    track: int
    start: float
    end: float
    confidence: float = 0.0               # mean margin of the winner over the runner-up (0..1)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def speaker_labels(sp: SpeakerEvidence, voiced_at: Callable[[float], bool], *, window: float = 1.0,
                   margin: float = 1.35) -> tuple[np.ndarray, np.ndarray]:
    """Per sample: the speaking track (``-1`` = nobody / unknown) and the winner's margin (0..1).

    Each track's activity is smoothed over ``window`` seconds.  Activity is already a per-pixel mean (face size does not
    matter), so all tracks share ONE scale: the 75th percentile of every visible value (normalising each track by its
    own level would make "B talks all the time" look exactly like "A never talks").  The winner must be visible, above
    half that reference, and beat the runner-up by ``margin`` x."""
    n, m = sp.n, len(sp.activity)
    lab = np.full(n, -1, dtype=np.int64)
    conf = np.zeros(n)
    if n == 0 or m < 2:
        return lab, conf
    k = max(1, int(round(window * sp.fps)))
    sm = np.zeros((m, n))
    vis = np.zeros((m, n), dtype=bool)
    for j, a in enumerate(sp.activity):
        arr = np.asarray(a[:n], dtype=float)
        vis[j] = arr >= 0
        sm[j] = np.where(vis[j], _smooth(np.where(vis[j], arr, 0.0), k), 0.0)
    ref = float(np.percentile(sm[vis], 75)) if vis.any() else 0.0
    norm = sm / max(ref, 1e-3)
    for i in range(n):
        if not voiced_at(i / sp.fps) or vis[:, i].sum() < 2:
            continue
        order = np.argsort(-norm[:, i])
        top, second = norm[order[0], i], norm[order[1], i]
        if top >= 0.5 and top >= margin * max(second, 0.05):
            lab[i] = int(order[0])
            conf[i] = float(min(1.0, (top - second) / max(top, 1e-6)))
    return lab, conf


def speaker_turns(sp: SpeakerEvidence, voiced_at: Callable[[float], bool], *, min_gap: float = 2.5,
                  vote: float = 1.5) -> list[Turn]:
    """Contiguous stretches where one track is the speaker.  Labels are majority-voted over ``vote`` seconds; the same
    speaker's runs separated by less than ``min_gap`` seconds (a breath, a pause between sentences, a short unknown)
    are joined -- the turn only ends when somebody ELSE talks or the silence is long."""
    lab, conf = speaker_labels(sp, voiced_at)
    n = lab.size
    if n == 0:
        return []
    half = max(1, int(round(vote * sp.fps / 2)))
    voted = lab.copy()
    for i in range(n):
        win = lab[max(0, i - half):i + half + 1]
        win = win[win >= 0]
        if win.size:
            vals, cnt = np.unique(win, return_counts=True)
            best = vals[np.argmax(cnt)]
            if cnt.max() * 2 > (lab[max(0, i - half):i + half + 1] >= 0).sum() and lab[i] >= 0:
                voted[i] = best
    turns: list[Turn] = []
    i = 0
    dt = 1.0 / sp.fps
    while i < n:
        if voted[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n and voted[j + 1] == voted[i]:
            j += 1
        t = Turn(int(voted[i]), i * dt, (j + 1) * dt, float(np.mean(conf[i:j + 1])))
        if turns and turns[-1].track == t.track and t.start - turns[-1].end < min_gap:
            prev = turns[-1]
            turns[-1] = Turn(prev.track, prev.start, t.end, (prev.confidence + t.confidence) / 2)
        else:
            turns.append(t)
        i = j + 1
    return turns


def visible_tracks(sp: SpeakerEvidence, t0: float, t1: float, min_share: float = 0.5) -> list[int]:
    """Tracks visible on at least ``min_share`` of the samples in ``t0..t1``."""
    a, b = int(t0 * sp.fps), max(int(t0 * sp.fps) + 1, int(np.ceil(t1 * sp.fps)))
    out = []
    for j, act in enumerate(sp.activity):
        seg = act[a:b]
        if seg and sum(1 for v in seg if v >= 0) / len(seg) >= min_share:
            out.append(j)
    return out


def track_box(sp: SpeakerEvidence, track: int, t0: float, t1: float) -> Optional[list[float]]:
    """Median face box of ``track`` over ``t0..t1`` (falls back to the track's overall median)."""
    a, b = int(t0 * sp.fps), max(int(t0 * sp.fps) + 1, int(np.ceil(t1 * sp.fps)))
    boxes = [bx for bx in sp.boxes[track][a:b] if bx] if track < len(sp.boxes) else []
    if boxes:
        return [float(v) for v in np.median(np.asarray(boxes, dtype=float), axis=0)]
    if track < len(sp.tracks):
        return list(sp.tracks[track]["box"])
    return None
