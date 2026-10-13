"""Technical QA of a rendered file.

This is *machine evidence only*: it can say ``pass`` / ``warn`` / ``fail`` about decodability, duration, size,
black/frozen frames, silence and loudness.  It never certifies taste: ``human_review`` always starts as
``pending`` and only a person can change it (see :func:`autocut.core.plan.set_human_review`).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..media.ffmpeg import _NOWINDOW, ffmpeg_info, run_probe_json

PASS, WARN, FAIL, NOT_RUN = "pass", "warn", "fail", "not_run"
_ORDER = {PASS: 0, NOT_RUN: 1, WARN: 2, FAIL: 3}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class QAReport:
    path: str
    checks: list = field(default_factory=list)
    human_review: str = "pending"          # never set by code in this module
    measurements: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        worst = PASS
        for c in self.checks:
            if _ORDER[c.status] > _ORDER[worst]:
                worst = c.status
        return worst

    def to_dict(self) -> dict:
        return {"path": self.path, "status": self.status, "human_review": self.human_review,
                "checks": [c.to_dict() for c in self.checks], "measurements": self.measurements}

    def summary(self) -> str:
        icon = {PASS: "✔", WARN: "▲", FAIL: "✘", NOT_RUN: "–"}
        lines = [f"{icon[c.status]} {c.name}：{c.detail}" for c in self.checks]
        lines.append("– 人工審片：尚未審（機器不會代替你判斷好不好看）")
        return "\n".join(lines)


def worst_status(statuses) -> str:
    """The worst of several check statuses (fail > warn > not_run > pass); ``not_run`` when there are none: a check
    that did not finish is never reported as a pass."""
    worst, seen = PASS, False
    for st in statuses:
        seen = True
        st = st if st in _ORDER else FAIL
        if _ORDER[st] > _ORDER[worst]:
            worst = st
    return worst if seen else NOT_RUN


def _within(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _run_ffmpeg(cmd: list, timeout: int = 3600) -> tuple[Optional[int], str, str]:
    """``(return code, stderr text, problem)``; ``problem`` is set (and the code ``None``) when ffmpeg could not be run
    to the end: the caller must then report the checks as not run, never as passed."""
    try:
        q = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_NOWINDOW)
    except subprocess.TimeoutExpired:
        return None, "", f"ffmpeg 超過 {timeout // 60} 分鐘還沒結束，已放棄"
    except OSError as e:
        return None, "", f"無法執行 ffmpeg：{e}"
    return q.returncode, q.stderr.decode("utf-8", "replace"), ""


def _tail(text: str) -> str:
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    return lines[-1][:160] if lines else ""


def technical_qa(path: str | Path, *, expect_duration: Optional[float] = None,
                 expect_size: Optional[tuple[int, int]] = None, expect_fps: Optional[float] = None,
                 expect_audio: bool = True, intended_black: Sequence[tuple[float, float]] = (),
                 deep: bool = True) -> QAReport:
    path = Path(path)
    rep = QAReport(str(path))
    add = lambda n, s, d="": rep.checks.append(Check(n, s, d))  # noqa: E731
    if not path.is_file() or path.stat().st_size == 0:
        add("檔案存在", FAIL, "找不到輸出檔或檔案是空的")
        return rep
    try:
        data = run_probe_json(path)
    except Exception as e:  # noqa: BLE001
        add("可讀取", FAIL, str(e)[:200])
        return rep
    vs = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    as_ = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    fmt = data.get("format", {})
    dur = float(fmt.get("duration") or 0)
    rep.measurements.update({"duration": dur, "size_bytes": int(fmt.get("size") or 0)})
    if not vs:
        add("影像串流", FAIL, "沒有影像")
        return rep
    v = vs[0]
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    rep.measurements.update({"width": w, "height": h})
    add("可讀取", PASS, f"{w}x{h}、{dur:.2f} 秒、{v.get('codec_name', '?')}")
    if expect_size:
        add("尺寸", PASS if (w, h) == tuple(expect_size) else FAIL, f"{w}x{h}（預期 {expect_size[0]}x{expect_size[1]}）")
    try:
        num, den = v.get("r_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except ValueError:
        fps = 0.0
    rep.measurements["fps"] = fps
    if expect_fps:
        add("影格率", PASS if _within(fps, expect_fps, 0.02) else FAIL, f"{fps:.3f}（預期 {expect_fps:.3f}）")
    if expect_duration is not None:
        tol = max(0.1, 3.0 / (expect_fps or 30.0))
        ok = _within(dur, expect_duration, tol)
        add("長度", PASS if ok else FAIL, f"{dur:.3f} 秒（預期 {expect_duration:.3f} ± {tol:.2f}）")
    if expect_audio:
        add("聲音串流", PASS if as_ else FAIL, "有" if as_ else "成品沒有聲音")
    if not deep:
        add("完整解碼", NOT_RUN, "略過")
        return rep

    info = ffmpeg_info()
    # 1) full decode: any decoder error means a damaged file
    rc, err, problem = _run_ffmpeg([info.ffmpeg, "-hide_banner", "-v", "error", "-i", str(path), "-f", "null", "-"])
    if problem:
        add("完整解碼", NOT_RUN, problem)
    else:
        err = err.strip()
        add("完整解碼", PASS if rc == 0 and not err else FAIL,
            err[-200:] if err else (f"ffmpeg 結束代碼 {rc}" if rc else "沒有錯誤"))
    # 2) black / freeze / silence / loudness in one pass
    vf = "blackdetect=d=0.4:pic_th=0.98,freezedetect=n=0.003:d=2"
    cmd = [info.ffmpeg, "-hide_banner", "-nostats", "-v", "info", "-i", str(path), "-vf", vf]
    if as_:
        cmd += ["-af", "silencedetect=n=-50dB:d=3,ebur128=peak=true"]
    cmd += ["-f", "null", "-"]
    rc, log, problem = _run_ffmpeg(cmd)
    names = ["黑畫面", "凍結畫面"] + (["長時間靜音", "響度", "真峰值"] if as_ else [])
    if problem or rc != 0:                      # the detectors did not finish: say so, do not read an empty log as 「沒有」
        why = problem or f"ffmpeg 結束代碼 {rc}：{_tail(log)}"
        for name in names:
            add(name, NOT_RUN, f"偵測沒有跑完（{why}）")
        return rep
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+)\s+black_end:([\d.]+)", log)]
    unexpected = [b for b in blacks if not any(b[0] >= g0 - 0.2 and b[1] <= g1 + 0.2 for g0, g1 in intended_black)]
    if unexpected:
        add("黑畫面", WARN, "出現黑畫面：" + "、".join(f"{a:.1f}–{b:.1f}s" for a, b in unexpected[:5]))
    else:
        add("黑畫面", PASS, "沒有意外的黑畫面")
    freezes = re.findall(r"freeze_start:\s*([\d.]+)", log)
    add("凍結畫面", WARN if freezes else PASS,
        ("畫面靜止 ≥2 秒：" + "、".join(f"{float(x):.1f}s" for x in freezes[:5])) if freezes else "沒有")
    if as_:
        sil = re.findall(r"silence_start:\s*([\d.]+)", log)
        add("長時間靜音", WARN if sil else PASS,
            ("靜音 ≥3 秒：" + "、".join(f"{float(x):.1f}s" for x in sil[:5])) if sil else "沒有")
        i_m = re.findall(r"\bI:\s+(-?[\d.]+)\s+LUFS", log)
        tp_m = re.findall(r"Peak:\s+(-?[\d.]+)\s+dBFS", log)
        if i_m:
            lufs = float(i_m[-1])
            rep.measurements["loudness_lufs"] = lufs
            add("響度", PASS if -24 <= lufs <= -9 else WARN, f"{lufs:.1f} LUFS")
        else:
            add("響度", NOT_RUN, "沒有量到（聲音是全靜音或太短）")
        if tp_m:
            tp = float(tp_m[-1])
            rep.measurements["true_peak_dbfs"] = tp
            add("真峰值", PASS if tp <= -0.5 else WARN, f"{tp:.1f} dBFS")
        else:
            add("真峰值", NOT_RUN, "沒有量到（聲音是全靜音或太短）")
    return rep
