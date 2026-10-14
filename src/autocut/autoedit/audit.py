"""Audit: the rules an :class:`~autocut.core.plan.EditPlan` must pass before it may touch the project (spec §5).

``audit(plan, proj, rules) -> AuditReceipt``.  Every rule has one fixed ``rule`` code, a severity (error / warn / info), a
繁體中文 message and a ``where`` (``<sequence id>/<clip or cue id>``, a sequence id, an asset id, or ``plan``).  The receipt
is bound to the plan hash *and* the project revision (``core.plan.check_receipt``), so editing either voids it.

* The audit never modifies the plan or the project (sequences are deep-copied; plan assets are only *mapped*).
* Fail closed: a check that crashes becomes an ``audit.internal`` error instead of being skipped silently.
* Two checks need analysis evidence (``start.black``, ``shorts.mid_word``).  Callers may pass ``evidence={asset_id:
  Evidence}``; otherwise only the *cache* is consulted (never recomputed).  Without evidence those checks simply do not
  run, and the audit says so in an ``audit.evidence`` info finding instead of pretending they passed.
* The thresholds actually used are stored in the receipt (``receipt.thresholds``).
"""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Callable, Mapping, Optional

from ..analysis.evidence import EVIDENCE_VERSION, AnalyzeParams, Evidence, cache_path
from ..core.plan import ERROR, INFO, WARN, AuditReceipt, EditPlan, Finding, PlanError
from ..core.project import Clip, Project, Sequence, Track, ordered_clips, validate_sequence
from ..core.timecode import EPS
from .captions import CAP_GAP
from ..util import canonical_json, fingerprint_file, sha256_hex
from .apply import plan_asset_map
from .config import PLATFORM_RULES, AuditRules, ShortsParams, resolve_band

HUMAN_PENDING_MESSAGE = "機器審核只檢查規則與技術面，尚未經人工審片"
BLACK_START_WINDOW = 0.5          # seconds at the head of the first clip that are inspected
BLACK_START_LEVEL = 0.05          # mean luma (0..1) below this = "black"
JUMP_MIN_MOVE = 0.05              # a reframe key pair this close in time AND this far apart (0..1) is a visible jump
SAME_VIEW_GAP = 0.05              # clips whose source ranges touch within this are one continuous take


# ---------------------------------------------------------------------------------------------- helpers
def _brief_flat(plan: EditPlan) -> dict:
    """``plan.brief`` with an optional nested ``params`` dict flattened (top-level keys win)."""
    out: dict = {}
    b = plan.brief or {}
    if isinstance(b.get("params"), dict):
        out.update(b["params"])
    out.update({k: v for k, v in b.items() if k != "params"})
    return out


def _chars(text: str) -> int:
    """Reading length of a caption: letters / digits / CJK characters (no spaces, no punctuation)."""
    return sum(1 for ch in text if ch.isalnum())


def _snippet(text: str, n: int = 14) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[:n] + "…"


def _main_track(seq: Sequence) -> Optional[Track]:
    for t in seq.video_tracks:
        if t.clips:
            return t
    return None


def _same_view(a: Clip, b: Clip) -> bool:
    """True when the picture simply carries on from ``a`` into ``b`` (same take, same framing): not a visual cut."""
    if a.asset_id != b.asset_id or abs(a.speed - b.speed) > 1e-6:
        return False
    if abs(b.src_in - a.src_out) > SAME_VIEW_GAP:
        return False
    ta, tb = a.transform, b.transform
    if ta.fit != tb.fit or ta.flip_h != tb.flip_h or abs(ta.zoom - tb.zoom) > 1e-3:
        return False
    ca, cb = ta.center_at(a.duration), tb.center_at(0.0)
    return abs(ca[0] - cb[0]) < 0.02 and abs(ca[1] - cb[1]) < 0.02


def first_visual_cut(seq: Sequence) -> Optional[float]:
    """Time of the first visual cut on the main video track: a hard cut / crossfade to a different take or framing, or a
    reframing jump (two crop keys about one frame apart that are far apart in space).  ``None`` = no cut at all."""
    tr = _main_track(seq)
    if tr is None:
        return None
    clips = ordered_clips(tr.clips)
    gap = 1.5 / max(seq.fps, 1.0)
    for i, c in enumerate(clips):
        for a, b in zip(c.transform.keys, c.transform.keys[1:]):
            if b[0] - a[0] <= gap and math.hypot(b[1] - a[1], b[2] - a[2]) >= JUMP_MIN_MOVE:
                return c.start + b[0]
        if i + 1 < len(clips) and not _same_view(c, clips[i + 1]):
            return min(c.end, clips[i + 1].start)
    return None


def _cut_points(seq: Sequence) -> list[tuple[Clip, float]]:
    """Source-time boundaries of the main track's clips that are real cuts (seamless joins are skipped)."""
    tr = _main_track(seq)
    if tr is None:
        return []
    clips = ordered_clips(tr.clips)
    out: list[tuple[Clip, float]] = []
    for i, c in enumerate(clips):
        if i == 0 or not _same_view(clips[i - 1], c):
            out.append((c, c.src_in))
        if i + 1 == len(clips) or not _same_view(c, clips[i + 1]):
            out.append((c, c.src_out))
    return out


def load_cached_evidence(plan: EditPlan, proj: Project) -> dict[str, Evidence]:
    """Evidence of the plan's sources that is already in the analysis cache (never computes anything).

    Uses ``plan.brief["analyze"]`` (the ``AnalyzeParams`` the planner used) or the defaults to find the cache file.
    Returns ``{project asset id: Evidence}`` (speech attached when its own cache has it)."""
    out: dict[str, Evidence] = {}
    try:
        params = AnalyzeParams.from_dict(_brief_flat(plan).get("analyze"))
        for src in plan.sources:
            aid = src.get("asset_id")
            asset = proj.assets.get(aid)
            key = str(src.get("fingerprint") or (asset.fingerprint if asset else "") or "")
            if asset is None or not key:
                continue
            try:
                ev = Evidence.load(cache_path(key, params))
            except (OSError, ValueError, TypeError):
                continue
            if ev.version != EVIDENCE_VERSION:
                continue
            ev.asset_id = aid
            try:                                                           # speech lives in its own cache
                from ..analysis import pipeline as _pl
                cached = _pl._read_speech_cache(_pl._speech_cache_file(key, params))
                if cached is not None:
                    ev.speech = cached["segments"]
            except Exception:                                              # private helper changed / no numpy ...
                pass
            out[aid] = ev
    except Exception:
        return out
    return out


class _Ctx:
    """Everything the rule functions share (read-only views of the plan and the project)."""

    def __init__(self, plan: EditPlan, proj: Project, rules: AuditRules,
                 evidence: Optional[Mapping[str, Evidence]]) -> None:
        self.plan, self.proj, self.rules = plan, proj, rules
        self.findings: list[Finding] = []
        self._evidence = dict(evidence) if evidence is not None else None
        self.seqs: list[Sequence] = []
        self.parse_error = ""
        self.not_run: dict[str, list[str]] = {}                            # evidence-based rule -> why it could not run
        self.id_map: dict[str, str] = {}
        fresh: dict = {}
        try:
            self.id_map, fresh = plan_asset_map(proj, plan)
        except Exception as exc:                                           # malformed plan.assets entry
            self.parse_error = f"{type(exc).__name__}: {exc}"
        self.view = Project("audit-view")                                  # real assets + the plan's own, nothing else
        self.view.assets = dict(proj.assets)
        self.view.assets.update(fresh)
        try:
            for sd in plan.sequences if not self.parse_error else []:
                seq = Sequence.from_dict(copy.deepcopy(sd))
                for _t, clip in seq.all_clips():
                    clip.asset_id = self.id_map.get(clip.asset_id, clip.asset_id)
                self.seqs.append(seq)
        except Exception as exc:                                           # malformed sequence dict
            self.parse_error = f"{type(exc).__name__}: {exc}"
            self.seqs = []

    def skip(self, rule: str, why: str) -> None:
        lst = self.not_run.setdefault(rule, [])
        if why not in lst:
            lst.append(why)

    def add(self, rule: str, severity: str, message: str, where: str = "") -> None:
        self.findings.append(Finding(rule, severity, message, where))

    @property
    def evidence(self) -> Mapping[str, Evidence]:
        if self._evidence is None:
            self._evidence = load_cached_evidence(self.plan, self.proj)
        return self._evidence

    def used_asset_ids(self) -> list[str]:
        seen: list[str] = []
        for seq in self.seqs:
            for _t, clip in seq.all_clips():
                if clip.asset_id not in seen:
                    seen.append(clip.asset_id)
        return seen


# ------------------------------------------------------------------------------------------------ rules
def _r_plan_empty(c: _Ctx) -> None:
    if not c.plan.sequences:
        c.add("plan.empty", ERROR, "計畫裡沒有任何序列，沒有東西可以套用", "plan")


def _r_plan_hash(c: _Ctx) -> None:
    """The hash a plan declares (``to_dict()['plan_sha256']``) must equal the recomputed one, and the plan must survive a
    save / load round trip with the same hash (otherwise the receipt could not be checked after a reload)."""
    plan = c.plan
    try:
        d = plan.to_dict()
        actual = sha256_hex(canonical_json(plan.content()))
        claimed = d.get("plan_sha256")
        if claimed and claimed != actual:
            c.add("plan.hash", ERROR, "計畫內聲明的雜湊值與重新計算的不符，計畫內容可能被竄改", "plan")
            return
        again = EditPlan.from_dict(json.loads(json.dumps(d, ensure_ascii=False)))
        if again.sha256() != actual:
            c.add("plan.hash", ERROR, "計畫存檔後重新讀入的雜湊值不一致，計畫含有無法穩定保存的資料", "plan")
    except PlanError as exc:
        c.add("plan.hash", ERROR, f"計畫無法通過完整性檢查：{exc}", "plan")
    except (TypeError, ValueError) as exc:
        c.add("plan.hash", ERROR, f"計畫內容無法轉成 JSON 保存：{exc}", "plan")


def _r_source_missing(c: _Ctx) -> None:
    missing: dict[str, list[str]] = {}
    for seq in c.seqs:
        for tr, clip in seq.all_clips():
            if clip.asset_id not in c.view.assets:
                missing.setdefault(clip.asset_id, []).append(f"{seq.id}/{clip.id}")
    for aid, wheres in missing.items():
        c.add("source.missing", ERROR,
              f"片段引用的素材「{aid}」不在專案裡，計畫也沒有附帶這個素材（{len(wheres)} 個片段）", wheres[0])


def _source_asset_ids(c: _Ctx) -> list[str]:
    ids = [a for a in c.used_asset_ids() if a in c.view.assets]
    for src in c.plan.sources:
        aid = src.get("asset_id")
        if aid in c.proj.assets and aid not in ids:
            ids.append(aid)
    return ids


def _r_source_file(c: _Ctx) -> None:
    if not c.rules.require_sources_exist:
        return
    for aid in _source_asset_ids(c):
        a = c.view.assets[aid]
        if not a.path or not os.path.exists(a.path):
            c.add("source.file", ERROR, f"素材檔案不存在：{a.name or a.path}（{a.path}）", aid)


def _r_source_changed(c: _Ctx) -> None:
    for src in c.plan.sources:
        aid, claimed = src.get("asset_id"), str(src.get("fingerprint") or "")
        a = c.proj.assets.get(aid)
        if a is None or not claimed or not a.fingerprint or not a.path or not os.path.isfile(a.path):
            continue                                   # nothing to compare (a missing file is source.file's business)
        try:
            now = fingerprint_file(a.path)
        except OSError:
            continue
        if now != claimed:
            c.add("source.changed", ERROR,
                  f"素材「{a.name or aid}」的檔案在分析之後被更換或修改（指紋不同），請重新分析再審核", aid)


def _r_seq_invalid(c: _Ctx) -> None:
    if c.parse_error:
        c.add("seq.invalid", ERROR, f"計畫裡的序列資料格式不正確（{c.parse_error}）", "plan")
        return
    for seq in c.seqs:
        for iss in validate_sequence(c.view, seq):
            if iss.code == "missing_asset":
                continue                               # reported once per asset as source.missing
            where = seq.id if not iss.where or iss.where == seq.id else f"{seq.id}/{iss.where}"
            c.add("seq.invalid", ERROR if iss.level == "error" else WARN, f"{seq.name}：{iss.message}", where)


def _r_clip_short(c: _Ctx) -> None:
    for seq in c.seqs:
        for tr, clip in seq.all_clips():
            if clip.duration < c.rules.min_clip_seconds - EPS:
                c.add("clip.short", WARN,
                      f"{seq.name}：{tr.name} 上有片段只有 {clip.duration:.2f} 秒（建議至少 {c.rules.min_clip_seconds:g} 秒）",
                      f"{seq.id}/{clip.id}")


def _r_shorts_aspect(c: _Ctx) -> None:
    if c.plan.kind != "shorts":
        return
    for seq in c.seqs:
        if not seq.is_portrait:
            c.add("shorts.aspect", ERROR,
                  f"{seq.name}不是直式（{seq.width}×{seq.height}），短影音必須是 9:16 直式", seq.id)


def _shorts_band(c: _Ctx) -> tuple[ShortsParams, float, float, dict]:
    sp = ShortsParams.from_dict(_brief_flat(c.plan))
    lo, hi = resolve_band(sp)
    return sp, lo, hi, PLATFORM_RULES.get(sp.platform, PLATFORM_RULES["ig_reels"])


def _r_shorts_band(c: _Ctx) -> None:
    if c.plan.kind != "shorts":
        return
    _sp, lo, hi, rule = _shorts_band(c)
    sev = ERROR if c.rules.strict_platform_bands else WARN
    for i, seq in enumerate(c.seqs, 1):
        dur, slop = seq.duration, 1.0 / max(seq.fps, 1.0)
        if dur < lo - slop or dur > hi + slop:
            c.add("shorts.band", sev,
                  f"第 {i} 支短片長 {dur:.1f} 秒，不在 {rule['label']} 建議的 {lo:g}–{hi:g} 秒範圍內", seq.id)


def _r_shorts_deadzone(c: _Ctx) -> None:
    if c.plan.kind != "shorts":
        return
    _sp, _lo, _hi, rule = _shorts_band(c)
    dz = rule.get("deadzone")
    if not dz:
        return
    sev = ERROR if c.rules.strict_platform_bands else WARN
    for i, seq in enumerate(c.seqs, 1):
        if dz[0] - EPS <= seq.duration <= dz[1] + EPS:
            c.add("shorts.deadzone", sev,
                  f"第 {i} 支短片長 {seq.duration:.1f} 秒，落在 {rule['label']} 範例校準的弱勢區間（{dz[0]:g}–{dz[1]:g} 秒）",
                  seq.id)


def _r_shorts_first_cut(c: _Ctx) -> None:
    if c.plan.kind != "shorts":
        return
    limit = c.rules.first_cut_max
    for i, seq in enumerate(c.seqs, 1):
        tr = _main_track(seq)
        if tr is None:
            continue
        first = first_visual_cut(seq)
        if first is None:
            if tr.end() > limit + EPS:
                c.add("shorts.first_cut", WARN,
                      f"第 {i} 支短片從頭到尾沒有任何畫面變化（剪點、取景或縮放），開頭 {limit:g} 秒內建議要有第一個剪點", seq.id)
        elif first > limit + EPS:
            c.add("shorts.first_cut", WARN,
                  f"第 {i} 支短片的第一個畫面變化出現在 {first:.2f} 秒，晚於 {limit:g} 秒（開頭容易讓人滑走）", seq.id)


def _r_shorts_mid_word(c: _Ctx) -> None:
    if c.plan.kind != "shorts":
        return
    for seq in c.seqs:
        margin = max(0.04, 1.0 / max(seq.fps, 1.0))
        reported = 0
        for clip, t in _cut_points(seq):
            ev = c.evidence.get(clip.asset_id)
            segs = ev.speech if ev is not None else None
            if not segs:
                name = c.view.assets[clip.asset_id].name if clip.asset_id in c.view.assets else clip.asset_id
                c.skip("shorts.mid_word", f"{name}（{'沒有分析證據' if ev is None else '沒有逐字稿'}）")
                continue
            for s in segs:
                if s.get("end", 0.0) < t - 1.0 or s.get("start", 0.0) > t + 1.0:
                    continue
                for w in s.get("words") or []:
                    ws, we = float(w.get("start", 0.0)), float(w.get("end", 0.0))
                    if ws + margin < t < we - margin and reported < 10:
                        reported += 1
                        c.add("shorts.mid_word", WARN,
                              f"{seq.name}：剪點 {t:.2f} 秒落在「{w.get('word', '')}」這個字的中間，聽起來會被切斷",
                              f"{seq.id}/{clip.id}")


def _subs(seq: Sequence) -> list:
    """The subtitles of ``seq`` (title cards excluded: reading speed, density and length rules are for subtitles)."""
    return [q for q in seq.cues if not q.is_card]


def _r_caption_reading(c: _Ctx) -> None:
    """Too fast to read.  An ERROR when the caption could have stayed up long enough (free time until the next
    subtitle or the end) -- the planner should have used it; only a WARNING when the speech itself is that fast (the
    caption already runs into the next one): no timing can fix verbatim subtitles of fast talk, only shorter text."""
    for seq in c.seqs:
        subs = sorted(_subs(seq), key=lambda q: (q.start, q.end, q.id))
        for i, q in enumerate(subs):
            dur = q.end - q.start
            n = _chars(q.text)
            if n == 0 or dur <= 0:
                continue
            cps = n / dur
            where = f"{seq.id}/{q.id}"
            if cps > c.rules.reading_fail_cps + EPS:
                # how long it could stay up: until the planner's gap before the next subtitle (or the end), on the
                # frame grid the times are snapped to (rounded down: a caption cannot end between two frames)
                nxt = subs[i + 1].start - CAP_GAP if i + 1 < len(subs) else seq.duration
                fps = max(float(seq.fps), 1.0)
                room = max(math.floor(nxt * fps + 1e-6) / fps, q.end) - q.start
                if n / room > c.rules.reading_fail_cps + EPS:
                    c.add("caption.reading", WARN,
                          f"字幕「{_snippet(q.text)}」閱讀速度 {cps:.1f} 字/秒，超過 {c.rules.reading_fail_cps:g} 字/秒："
                          "說話本身就這麼快（字幕已經接到下一句），觀眾可能看不完；可以刪減這句字幕的文字", where)
                else:
                    c.add("caption.reading", ERROR,
                          f"字幕「{_snippet(q.text)}」閱讀速度 {cps:.1f} 字/秒，超過上限 {c.rules.reading_fail_cps:g} 字/秒，"
                          "看不完（後面還有空檔，應該顯示久一點）", where)
            elif cps > c.rules.reading_warn_cps + EPS:
                c.add("caption.reading", WARN,
                      f"字幕「{_snippet(q.text)}」閱讀速度 {cps:.1f} 字/秒，偏快（建議不超過 {c.rules.reading_warn_cps:g} 字/秒）", where)


def _r_caption_rate(c: _Ctx) -> None:
    captions_on = str(_brief_flat(c.plan).get("captions", "off")) in ("auto", "file")
    for seq in c.seqs:
        dur = seq.duration
        if dur < 20.0 - EPS or not (_subs(seq) or captions_on):
            continue
        rate = len(_subs(seq)) / (dur / 60.0)
        if rate < c.rules.caption_rate_warn_per_min - EPS:
            what = ("設定了要字幕，但計畫裡沒有任何字幕（可能沒有語音辨識結果）" if not _subs(seq) else
                    f"字幕換頁率只有每分鐘 {rate:.1f} 條，低於 {c.rules.caption_rate_warn_per_min:g}，畫面會有很長的時間沒有字幕")
            c.add("caption.rate", WARN, f"{seq.name}：{what}", seq.id)


def _r_caption_overlap(c: _Ctx) -> None:
    for seq in c.seqs:
        for layer in (_subs(seq), [q for q in seq.cues if q.is_card]):   # title cards are their own layer
            prev = None
            for q in sorted(layer, key=lambda q: (q.start, q.end, q.id)):
                if prev is not None and prev.end - q.start > c.rules.max_caption_overlap + EPS:
                    what = "字卡" if q.is_card else "字幕"
                    c.add("caption.overlap", ERROR,
                          f"{what}「{_snippet(q.text)}」與前一條{what}重疊 {prev.end - q.start:.2f} 秒", f"{seq.id}/{q.id}")
                if prev is None or q.end > prev.end:
                    prev = q


def _r_caption_short(c: _Ctx) -> None:
    for seq in c.seqs:
        for q in _subs(seq):
            if 0 < q.end - q.start < c.rules.min_cue_seconds - EPS:
                c.add("caption.short", WARN,
                      f"字幕「{_snippet(q.text)}」只顯示 {q.end - q.start:.2f} 秒（建議至少 {c.rules.min_cue_seconds:g} 秒）",
                      f"{seq.id}/{q.id}")


def _r_caption_long(c: _Ctx) -> None:
    for seq in c.seqs:
        for q in _subs(seq):
            n = _chars(q.text)
            if n > c.rules.max_cue_chars:
                c.add("caption.long", WARN,
                      f"字幕「{_snippet(q.text)}」有 {n} 個字，超過每條 {c.rules.max_cue_chars} 字的上限", f"{seq.id}/{q.id}")


def _r_caption_range(c: _Ctx) -> None:
    for seq in c.seqs:
        tol, dur = 1.0 / max(seq.fps, 1.0), seq.duration
        for q in seq.cues:
            if q.start < -tol or q.end > dur + tol:
                c.add("caption.range", ERROR,
                      f"字幕「{_snippet(q.text)}」的時間 {q.start:.2f}–{q.end:.2f} 秒超出序列範圍（0–{dur:.2f} 秒）",
                      f"{seq.id}/{q.id}")


def _r_assemble_target(c: _Ctx) -> None:
    if c.plan.kind != "assemble":
        return
    flat = _brief_flat(c.plan)
    try:
        target = float(flat.get("target_seconds") or 0.0)
    except (TypeError, ValueError):
        return
    if target <= 0:
        return
    actual = sum(s.duration for s in c.seqs)
    tol = c.rules.target_tolerance
    if abs(actual - target) / target <= tol + EPS:
        return
    usable = flat.get("usable_seconds")
    try:
        usable = float(usable) if usable is not None else None
    except (TypeError, ValueError):
        usable = None
    if actual < target and usable is not None and usable < target:
        c.add("assemble.target", INFO,
              f"來源素材不足：全部可用內容約 {usable:.1f} 秒，實際長度 {actual:.1f} 秒，達不到目標 {target:.0f} 秒", "plan")
    else:
        c.add("assemble.target", WARN,
              f"實際長度 {actual:.1f} 秒與目標 {target:.0f} 秒相差 {abs(actual - target) / target * 100:.0f}%，"
              f"超過容許的 {tol * 100:.0f}%", "plan")


def _r_assemble_unused(c: _Ctx) -> None:
    if c.plan.kind != "assemble":
        return
    used = {clip.asset_id for seq in c.seqs for t in seq.video_tracks for clip in t.clips}
    for src in c.plan.sources:
        aid = src.get("asset_id", "")
        if c.id_map.get(aid, aid) in used:
            continue
        usable = src.get("usable_seconds")
        try:
            if usable is not None and float(usable) <= 0:
                continue                                   # nothing usable in it: the drop decisions already say why
        except (TypeError, ValueError):
            pass
        c.add("assemble.unused", INFO, f"素材「{src.get('name') or aid}」有可用內容，但計畫沒有用到它的任何片段", aid)


def _r_start_black(c: _Ctx) -> None:
    if not c.rules.black_start_warn:
        return
    for seq in c.seqs:
        firsts = [(clip.start, i, clip) for i, t in enumerate(seq.video_tracks) for clip in t.clips]
        if not firsts:
            continue
        clip = min(firsts, key=lambda x: (x[0], x[1], x[2].id))[2]
        ev = c.evidence.get(clip.asset_id)
        if ev is None or not ev.bright:
            name = c.view.assets[clip.asset_id].name if clip.asset_id in c.view.assets else clip.asset_id
            c.skip("start.black", f"{name}（沒有畫面分析證據）")
            continue
        fps = ev.visual_fps or 5.0
        i0 = max(0, min(len(ev.bright) - 1, int(round(clip.src_in * fps))))
        i1 = max(i0, min(len(ev.bright) - 1, int(round((clip.src_in + BLACK_START_WINDOW * clip.speed) * fps))))
        vals = ev.bright[i0:i1 + 1]
        if vals and sum(vals) / len(vals) < BLACK_START_LEVEL:
            c.add("start.black", WARN,
                  f"{seq.name}：開頭約 {BLACK_START_WINDOW:g} 秒的畫面幾乎全黑（平均亮度 {sum(vals) / len(vals):.2f}）",
                  f"{seq.id}/{clip.id}")


def _r_total_max(c: _Ctx) -> None:
    total = sum(s.duration for s in c.seqs)
    if total > c.rules.max_total_seconds + EPS:
        c.add("total.max", ERROR, f"計畫總長 {total / 60:.1f} 分鐘，超過上限 {c.rules.max_total_seconds / 60:.0f} 分鐘", "plan")


# order = the order of the rule table in the spec (so findings come out grouped and deterministic)
def _r_longform_zoom(c: _Ctx) -> None:
    """Long-video close-ups: too short to read as a camera change, or zoomed so far that a small source turns soft."""
    if c.plan.kind != "longform":
        return
    for seq in c.seqs:
        for t in seq.video_tracks:
            for clip in t.clips:
                if clip.transform.zoom <= 1.0 + 1e-6:
                    continue
                if clip.duration < 1.0 - EPS:
                    c.add("longform.zoom", WARN, f"講者特寫只有 {clip.duration:.2f} 秒，太短會像畫面閃一下", f"{seq.id}/{clip.id}")
                a = c.view.assets.get(clip.asset_id)
                if a is not None and a.width and a.height:
                    eff = min(a.width, a.height) / clip.transform.zoom
                    if eff < 0.5 * min(seq.width, seq.height):
                        c.add("longform.zoom", WARN, f"特寫放大 {clip.transform.zoom:g} 倍後只剩約 {eff:.0f} 像素高，"
                                                     f"輸出 {seq.height}p 時會明顯模糊", f"{seq.id}/{clip.id}")


_RULES: list[tuple[str, Callable[[_Ctx], None]]] = [
    ("plan.empty", _r_plan_empty), ("plan.hash", _r_plan_hash),
    ("source.missing", _r_source_missing), ("source.file", _r_source_file), ("source.changed", _r_source_changed),
    ("seq.invalid", _r_seq_invalid), ("clip.short", _r_clip_short),
    ("shorts.aspect", _r_shorts_aspect), ("shorts.band", _r_shorts_band), ("shorts.deadzone", _r_shorts_deadzone),
    ("shorts.first_cut", _r_shorts_first_cut), ("shorts.mid_word", _r_shorts_mid_word),
    ("caption.reading", _r_caption_reading), ("caption.rate", _r_caption_rate), ("caption.overlap", _r_caption_overlap),
    ("caption.short", _r_caption_short), ("caption.long", _r_caption_long), ("caption.range", _r_caption_range),
    ("assemble.target", _r_assemble_target), ("assemble.unused", _r_assemble_unused),
    ("longform.zoom", _r_longform_zoom),
    ("start.black", _r_start_black), ("total.max", _r_total_max),
]


def audit(plan: EditPlan, proj: Project, rules: Optional[AuditRules] = None, *,
          evidence: Optional[Mapping[str, Evidence]] = None) -> AuditReceipt:
    """Run every rule and return a receipt bound to ``plan.sha256()`` and ``proj.revision``.

    ``evidence`` (``{project asset id: Evidence}``) feeds the two evidence-based checks; when omitted the analysis cache is
    consulted (see :func:`load_cached_evidence`).  Neither the plan nor the project is modified.
    """
    rules = rules or AuditRules()
    ctx = _Ctx(plan, proj, rules, evidence)
    for name, fn in _RULES:
        try:
            fn(ctx)
        except Exception as exc:                          # fail closed: a crashed check blocks the plan
            ctx.add("audit.internal", ERROR, f"審核規則「{name}」執行失敗（{type(exc).__name__}: {exc}），為安全起見不予通過", "plan")
    for rule, label in (("start.black", "開頭是否為黑畫面"), ("shorts.mid_word", "剪點是否切在字中間")):
        if rule in ctx.not_run:                           # honest: say what was NOT checked
            why = "、".join(ctx.not_run[rule][:5]) + ("…" if len(ctx.not_run[rule]) > 5 else "")
            ctx.add("audit.evidence", INFO, f"沒有檢查「{label}」（{rule}）：{why}", "plan")
    ctx.add("human.pending", INFO, HUMAN_PENDING_MESSAGE, "plan")
    accepted = not any(f.severity == ERROR for f in ctx.findings)
    return AuditReceipt(plan_sha256=plan.sha256(), project_revision=proj.revision, project_sha256=proj.content_sha256(),
                        accepted=accepted, findings=ctx.findings, thresholds=rules.to_dict())
