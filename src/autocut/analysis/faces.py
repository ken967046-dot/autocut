"""Face detection with OpenCV's Haar cascade, on the small analysis frames (spec §2.3).

Known limits (do not oversell): the frontal-face Haar cascade is weak on profiles, bowed heads, masks and
back-lit faces, and its smallest scannable window is 24 px, so on the default 256 px wide analysis frames a
face narrower than ~9 % of the frame width is never found (raise ``AnalyzeParams.frame_width`` to see smaller
ones).  It still reports a face in ~1-2 % of face-less frames.  It is good enough to *bias* a 9:16 crop towards a
talking head; it is not a tracker.  Everything here degrades to "no faces" instead of raising.

Two choices differ from a textbook recipe, both measured on real photos (see tests/test_analysis.py): the frame is
NOT histogram-equalised (equalisation lost faces whenever a frame has large flat areas such as letterbox bars or a
plain backdrop), and ``minNeighbors`` is 6 (5 gave ~2x the false detections on face-less footage for ~1/20 more recall).
"""
from __future__ import annotations

import threading

import numpy as np

CASCADE_FILE = "haarcascade_frontalface_default.xml"
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6

_local = threading.local()          # one classifier per thread: detectMultiScale is not re-entrant


def _load():
    """The thread's cascade classifier, or ``None`` when OpenCV / the cascade file is unavailable."""
    if getattr(_local, "tried", False):
        return _local.cascade
    _local.tried, _local.cascade = True, None
    try:
        import cv2
        casc = cv2.CascadeClassifier(str(cv2.data.haarcascades) + CASCADE_FILE)
        if not casc.empty():
            _local.cascade = casc
    except Exception:                                   # no cv2, no cv2.data, unreadable file ...
        _local.cascade = None
    return _local.cascade


def faces_available() -> bool:
    """True when the Haar cascade could be loaded (the pipeline notes the degradation when it is not)."""
    return _load() is not None


def _to_grey_u8(frame) -> np.ndarray | None:
    import cv2
    a = np.asarray(frame)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.ndim not in (2, 3) or a.size == 0 or (a.ndim == 3 and a.shape[2] not in (3, 4)):
        return None
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.dtype.kind == "f" and float(a.max()) <= 1.0 else a, 0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY if a.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(a)


def detect_faces(frame, min_frac: float = 0.06) -> list[list[float]]:
    """Frontal faces in ``frame`` (BGR/BGRA/grey, any size) as ``[[x, y, w, h], ...]``.

    Coordinates are fractions (0..1) of the frame, origin top-left; the biggest face comes first.  ``min_frac`` is the
    smallest face width as a fraction of the frame width.  Returns ``[]`` when the cascade is missing or anything fails.
    """
    try:
        casc = _load()
        grey = _to_grey_u8(frame)
        if casc is None or grey is None:
            return []
        import cv2
        h, w = grey.shape
        if min(h, w) < 24:
            return []
        side = max(int(round(float(min_frac) * w)), 1)
        found = casc.detectMultiScale(grey, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS,
                                      flags=cv2.CASCADE_SCALE_IMAGE, minSize=(side, side))
        boxes = []
        for x, y, bw, bh in np.asarray(found).reshape(-1, 4).tolist():
            x0, y0 = min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0)
            x1, y1 = min(max((x + bw) / w, 0.0), 1.0), min(max((y + bh) / h, 0.0), 1.0)
            boxes.append([round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)])
        boxes.sort(key=lambda b: -b[2] * b[3])
        return boxes
    except Exception:
        return []
