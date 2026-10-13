"""Virtual camera for 9:16 reframing (spec 3.9): where should the crop window of a clip be at every moment?

``reframe_keys`` turns the picture evidence (face boxes, motion centroids, scene cuts) of a source range into
``Transform.keys = [[t_local, cx, cy], ...]``:

* subject per sample: the main face (area x continuity; the union of all faces when they fit into the window), else the
  motion centroid, else the previous position;
* the target is clamped into the legal window range first, then a dead zone + exponential smoothing + a maximum pan
  speed keep the camera calm;
* a scene cut is never smoothed: two keys one frame apart make the framing jump;
* the path is simplified with Ramer-Douglas-Peucker (eps 0.01) to at most ~40 keys; first and last key always exist.

Pure function of its inputs (no I/O, no randomness).
"""
from __future__ import annotations

import math
from statistics import median
from typing import Optional, Sequence

from ..core.geometry import crop_window
from ..core.timecode import snap_time

MAX_KEYS = 40
RDP_EPS = 0.01
PUNCH_DY = 0.12              # punch-in: the window centre sits this many window-heights below the face centre
FACE_MIN_RATIO = 0.2         # below this share of samples with a face the faces are not trusted
DEAD_ZONE = 0.15             # of the window width at strength 0.5
PAN_SPEED = 0.25             # window widths per second at strength 0.5
SMOOTH_TAU = 0.45            # seconds at strength 0.5
FACE_FILL_GAP = 1.5          # seconds a face position may be carried over samples without face detection
FACE_HOLD_GAP = 1.0          # ... and over samples where detection ran but found no face (flicker)
MIN_SHOT = 0.2               # scene cuts closer than this to the ends of the range are ignored


def _q4(v: float, lo: float, hi: float) -> float:
    """Round to 4 decimals without stepping outside the legal camera range ``[lo, hi]``."""
    r = round(float(v), 4)
    if r < lo:
        r = math.ceil(lo * 1e4 - 1e-9) / 1e4
    if r > hi:
        r = math.floor(hi * 1e4 + 1e-9) / 1e4
    return r


def _clamp(v: float, lo: float, hi: float) -> float:
    if hi < lo:
        return 0.5 * (lo + hi)
    return min(max(v, lo), hi)


def _display_size(asset, ev) -> tuple[float, float]:
    for src in (asset, ev):
        w, h = float(getattr(src, "width", 0) or 0), float(getattr(src, "height", 0) or 0)
        if w > 0 and h > 0:
            return w, h
    return 1920.0, 1080.0


# ------------------------------------------------------------------------------------------ subject
def _pick_subject(faces: Sequence, prev: Optional[tuple[float, float]], fw: float, fh: float
                  ) -> Optional[tuple[float, float]]:
    """Centre of the subject among ``faces`` ([x,y,w,h] boxes, 0..1)."""
    boxes = [(float(f[0]), float(f[1]), float(f[2]), float(f[3])) for f in faces
             if len(f) >= 4 and float(f[2]) > 0 and float(f[3]) > 0]
    if not boxes:
        return None
    if len(boxes) > 1:
        x0, x1 = min(b[0] for b in boxes), max(b[0] + b[2] for b in boxes)
        y0, y1 = min(b[1] for b in boxes), max(b[1] + b[3] for b in boxes)
        if (x1 - x0) <= 0.9 * fw and (fh >= 0.999 or (y1 - y0) <= 0.9 * fh):
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0                 # everybody fits: frame the group

    def score(b: tuple[float, float, float, float]) -> float:
        area = b[2] * b[3]
        if prev is None:
            return area
        d = math.hypot(b[0] + b[2] / 2.0 - prev[0], b[1] + b[3] / 2.0 - prev[1])
        return area * (0.4 + 0.6 * math.exp(-((d / 0.25) ** 2)))

    best = max(boxes, key=score)                                    # ties -> first box (deterministic)
    return best[0] + best[2] / 2.0, best[1] + best[3] / 2.0


def _ran(ev, i: int) -> bool:
    mask = getattr(ev, "face_sample_mask", None)
    if mask:
        return i < len(mask) and bool(mask[i])
    return i < len(getattr(ev, "faces", []) or [])


def _faces_at(ev, i: int) -> list:
    faces = getattr(ev, "faces", None) or []
    return faces[i] if i < len(faces) and faces[i] else []


def _motion_at(ev, i: int) -> Optional[tuple[float, float]]:
    mx, my = getattr(ev, "motion_cx", None) or [], getattr(ev, "motion_cy", None) or []
    if i < len(mx) and i < len(my) and mx[i] is not None and my[i] is not None and mx[i] >= 0 and my[i] >= 0:
        return float(mx[i]), float(my[i])
    return None


def _median3(vals: list[Optional[tuple[float, float]]]) -> list[Optional[tuple[float, float]]]:
    out = list(vals)
    for k in range(len(vals)):
        win = [v for v in vals[max(0, k - 1):k + 2] if v is not None]
        if len(win) >= 2 and vals[k] is not None:
            out[k] = (median(v[0] for v in win), median(v[1] for v in win))
    return out


def _shot_targets(ev, idxs: list[int], vfps: float, mode: str, use_faces: bool, fw: float, fh: float):
    """Per-sample ``(cx, cy, src)`` for one shot (``src`` = face | motion | hold), plus face statistics."""
    n = len(idxs)
    tgt: list[Optional[tuple[float, float, str]]] = [None] * n
    unran: list[bool] = [False] * n
    prev: Optional[tuple[float, float]] = None
    if mode == "face" and use_faces:
        for k, i in enumerate(idxs):
            if not _ran(ev, i):
                unran[k] = True
                continue
            c = _pick_subject(_faces_at(ev, i), prev, fw, fh)
            if c is not None:
                tgt[k] = (c[0], c[1], "face")
                prev = c
        # samples without a face (or without face detection): carry / interpolate the face position over short gaps
        face_k = [k for k in range(n) if tgt[k] is not None]
        for k in range(n):
            if tgt[k] is not None or not face_k:
                continue
            thr = FACE_FILL_GAP if unran[k] else FACE_HOLD_GAP
            left = max((j for j in face_k if j < k), default=None)
            right = min((j for j in face_k if j > k), default=None)
            tl = (idxs[k] - idxs[left]) / vfps if left is not None else math.inf
            tr = (idxs[right] - idxs[k]) / vfps if right is not None else math.inf
            if tl <= thr and tr <= thr:
                a = tl / (tl + tr) if (tl + tr) > 0 else 0.0
                l, r = tgt[left], tgt[right]
                tgt[k] = (l[0] + (r[0] - l[0]) * a, l[1] + (r[1] - l[1]) * a, "face")
            elif tl <= thr:
                tgt[k] = tgt[left]
            elif tr <= thr:
                tgt[k] = tgt[right]
    motion = _median3([_motion_at(ev, i) for i in idxs])
    for k in range(n):
        if tgt[k] is None and motion[k] is not None:
            tgt[k] = (motion[k][0], motion[k][1], "motion")
    # hold the last known position; head gaps take the first known one
    last = None
    for k in range(n):
        if tgt[k] is None and last is not None:
            tgt[k] = (last[0], last[1], "hold")
        elif tgt[k] is not None:
            last = tgt[k]
    first = next((t for t in tgt if t is not None), None)
    if first is not None:
        for k in range(n):
            if tgt[k] is None:
                tgt[k] = (first[0], first[1], "hold")
    return tgt


# ------------------------------------------------------------------------------------------- camera
def _simulate(times: list[float], goals: list[float], lo: float, hi: float, dead: float, vmax: float,
              tau: float, locked: bool) -> list[float]:
    """One axis of the virtual camera: dead zone, exponential smoothing and a speed limit, starting on the subject."""
    g = [_clamp(v, lo, hi) for v in goals]
    if not g:
        return []
    if locked:
        m = _clamp(median(g), lo, hi)
        return [m] * len(g)
    c = g[0]
    out = [c]
    for k in range(1, len(g)):
        dt = max(times[k] - times[k - 1], 1e-6)
        err = g[k] - c
        goal = c if abs(err) <= dead else g[k] - math.copysign(dead, err)
        step = (goal - c) * (1.0 - math.exp(-dt / tau))
        lim = vmax * dt
        c = _clamp(c + max(-lim, min(lim, step)), lo, hi)
        out.append(c)
    return out


# --------------------------------------------------------------------------------------------- RDP
def _rdp_keep(pts: list[list[float]], eps: float, forced: set[int]) -> list[int]:
    """Indices kept by a time-indexed Ramer-Douglas-Peucker (deviation = distance in cx / cy at the same time)."""
    n = len(pts)
    keep = {0, n - 1} | forced

    def run(i: int, j: int) -> None:
        stack = [(i, j)]
        while stack:
            a, b = stack.pop()
            if b - a < 2:
                continue
            ta, tb = pts[a][0], pts[b][0]
            worst, wk = -1.0, -1
            for k in range(a + 1, b):
                u = (pts[k][0] - ta) / (tb - ta) if tb > ta else 0.0
                dev = max(abs(pts[k][1] - (pts[a][1] + (pts[b][1] - pts[a][1]) * u)),
                          abs(pts[k][2] - (pts[a][2] + (pts[b][2] - pts[a][2]) * u)))
                if dev > worst:
                    worst, wk = dev, k
            if worst > eps and wk >= 0:
                keep.add(wk)
                stack.append((a, wk))
                stack.append((wk, b))

    anchors = sorted(keep)
    for a, b in zip(anchors, anchors[1:]):
        run(a, b)
    return sorted(keep)


def _simplify(pts: list[list[float]], forced: set[int], max_keys: int) -> list[list[float]]:
    eps = RDP_EPS
    keep = _rdp_keep(pts, eps, forced)
    guard = 0
    while len(keep) > max_keys and guard < 40:
        eps *= 1.4
        keep = _rdp_keep(pts, eps, forced)
        guard += 1
    return [pts[i] for i in keep]


# ------------------------------------------------------------------------------------------ public
def reframe_keys(ev, asset, t0: float, t1: float, out_w: int, out_h: int, *, mode: str = "face",
                 strength: float = 0.5, zoom: float = 1.0, punch_in_at: Optional[float] = None,
                 fps: float = 30.0, scenes: Optional[Sequence[float]] = None, max_keys: int = MAX_KEYS
                 ) -> tuple[list[list[float]], dict]:
    """Crop-window centre keys for the source range ``t0..t1`` of ``asset``.

    Returns ``([[t_local, cx, cy], ...], info)``; ``t_local`` is relative to ``t0`` (the first and the last key always
    exist, at most ``max_keys`` keys unless more scene cuts force them), ``info = {"mode_used": "face|motion|center|blur",
    "face_ratio", "n_keys", ...}``.  ``mode="blur"`` returns no keys (the caller uses ``fit="blur"``); ``ev=None`` means no
    evidence -> centred.  ``zoom`` is the zoom the clip will have (the window shrinks, so ``cy`` matters);
    ``punch_in_at`` (clip-local seconds) switches on the punch-in composition from that moment on: the face sits in the
    upper-middle of the window (``cy = face y + 0.12 x window height``).  ``scenes`` overrides ``ev.scenes`` (e.g. cut
    times already refined to the frame); ``fps`` is the output frame rate (key times are on its grid).
    """
    mode = mode if mode in ("face", "motion", "center", "blur") else "face"
    dur = max(0.0, float(t1) - float(t0))
    info: dict = {"mode_used": mode, "face_ratio": 0.0, "n_keys": 0, "jumps": 0, "n_samples": 0}
    if mode == "blur":
        return [], info
    fps = float(fps) if fps and fps > 0 else 30.0
    frame = 1.0 / fps
    if dur <= 0:
        info["mode_used"], info["n_keys"] = "center", 1
        return [[0.0, 0.5, 0.5]], info
    zoom = max(1.0, float(zoom or 1.0))
    dw, dh = _display_size(asset, ev)
    cw, ch = crop_window(dw, dh, out_w, out_h, zoom)
    fw, fh = min(1.0, cw / dw), min(1.0, ch / dh)
    cx_lo, cx_hi, cy_lo, cy_hi = fw / 2.0, 1.0 - fw / 2.0, fh / 2.0, 1.0 - fh / 2.0
    t_end = snap_time(dur, fps) if dur > frame else dur

    def static(why: str = "center") -> tuple[list[list[float]], dict]:
        info["mode_used"], info["n_keys"] = why, 2
        return [[0.0, 0.5, 0.5], [round(t_end, 4), 0.5, 0.5]], info

    vfps = float(getattr(ev, "visual_fps", 0.0) or 5.0) if ev is not None else 5.0
    n_samples = 0
    if ev is not None:
        n_samples = max(len(getattr(ev, "motion", []) or []), len(getattr(ev, "faces", []) or []),
                        len(getattr(ev, "motion_cx", []) or []))
    if ev is None or mode == "center" or n_samples == 0:
        return static()
    i_lo = max(0, int(math.ceil(t0 * vfps - 1e-9)))
    i_hi = min(n_samples - 1, int(math.floor(t1 * vfps + 1e-9)))
    if i_hi < i_lo:
        return static()
    idxs_all = list(range(i_lo, i_hi + 1))
    info["n_samples"] = len(idxs_all)

    # --- how often is there a face in this range (decides whether faces are trusted)
    ran = [i for i in idxs_all if _ran(ev, i)]
    with_face = [i for i in ran if _faces_at(ev, i)]
    face_ratio = len(with_face) / len(ran) if ran else 0.0
    info["face_ratio"] = round(face_ratio, 4)
    use_faces = mode == "face" and face_ratio >= FACE_MIN_RATIO

    # --- shots
    raw_cuts = list(scenes) if scenes is not None else list(getattr(ev, "scenes", []) or [])
    cuts: list[float] = []
    for c in sorted(float(c) for c in raw_cuts if t0 + MIN_SHOT < float(c) < t1 - MIN_SHOT):
        if not cuts or c - cuts[-1] >= MIN_SHOT:
            cuts.append(c)
    bounds = [float(t0)] + cuts + [float(t1)]

    # --- strength -> camera parameters
    s = min(max(float(strength), 0.0), 1.0)
    locked = s <= 1e-3
    m = max(s / 0.5, 0.25)
    dzk = min(max(1.0 + (0.5 - s) * 1.4, 0.3), 1.7)
    tau = SMOOTH_TAU / m

    pts: list[list[float]] = []
    forced: set[int] = set()
    sources: list[str] = []
    for si in range(len(bounds) - 1):
        a, b = bounds[si], bounds[si + 1]
        last_shot = si == len(bounds) - 2
        idxs = [i for i in idxs_all if a - 1e-9 <= i / vfps < b - 1e-9 or (last_shot and abs(i / vfps - b) < 1e-9)]
        if not idxs:                                                    # a shot between two samples: nearest sample
            idxs = [min(idxs_all, key=lambda i: abs(i / vfps - (a + b) / 2.0))]
        tg = _shot_targets(ev, idxs, vfps, mode, use_faces, fw, fh)
        if tg and tg[0] is None:                                         # nothing to follow in this shot
            tg = [(0.5, 0.5, "hold")] * len(idxs)
        times = [i / vfps - t0 for i in idxs]
        gx, gy = [], []
        for t, (x, y, src) in zip(times, tg):
            if src == "face" and punch_in_at is not None and t >= punch_in_at - 1e-9:
                y = y + PUNCH_DY * fh
            gx.append(x)
            gy.append(y)
        sources.extend(src for _x, _y, src in tg)
        xs = _simulate(times, gx, cx_lo, cx_hi, DEAD_ZONE * fw * dzk, PAN_SPEED * fw * m, tau, locked)
        ys = (_simulate(times, gy, cy_lo, cy_hi, DEAD_ZONE * fh * dzk, PAN_SPEED * fh * m, tau, locked)
              if fh < 0.999 else [0.5] * len(times))
        t_a = max(0.0, a - t0)
        t_b = min(dur, b - t0)
        t_start = 0.0 if si == 0 else snap_time(t_a, fps)
        t_limit = t_end if last_shot else snap_time(t_b, fps) - frame        # last key before the next shot's first
        shot: list[list[float]] = [[t_start, xs[0], ys[0]]]
        for t, x, y in zip(times, xs, ys):
            tt = min(snap_time(min(max(t, 0.0), dur), fps), t_limit)
            if tt > shot[-1][0] + 1e-9:
                shot.append([tt, x, y])
        if t_limit > shot[-1][0] + 1e-9:
            shot.append([t_limit, xs[-1], ys[-1]])
        if pts:                                                         # jump: previous shot's last key / this shot's first
            forced.add(len(pts) - 1)
            forced.add(len(pts))
            info["jumps"] += 1
        pts.extend(shot)
    pts[0][0] = 0.0
    out = [[round(float(p[0]), 4), _q4(p[1], cx_lo, cx_hi), _q4(p[2], cy_lo, cy_hi)]
           for p in _simplify(pts, forced, max_keys)]
    if len(out) < 2:
        out.append([round(t_end, 4), out[0][1], out[0][2]])
    n_motion = sum(1 for x in sources if x == "motion")
    if mode == "face" and use_faces:
        info["mode_used"] = "face"
    elif n_motion > 0:
        info["mode_used"] = "motion"
    else:
        info["mode_used"] = "center" if all(x == "hold" for x in sources) else mode
    info["n_keys"] = len(out)
    return out, info
