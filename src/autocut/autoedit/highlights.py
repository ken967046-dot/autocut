"""Content highlights ("爆點") from a transcript or a subtitle file.

The shorts planner used to rank windows mostly by *how* something sounds and looks (voice ratio, loudness, faces,
motion); what was actually *said* only counted through a tiny keyword regex.  On a talking-head video every window
looks alike by those measures, so the picks were dull.  This module scores the words:

* :func:`sentences`        transcript segments -> sentence units ``Sentence(start, end, text)`` (split at 。！？ etc.;
                           a subtitle file without punctuation keeps one cue per unit)
* :func:`score_sentences`  per-sentence content score 0..1 = lexical signals (question / twist / emotion / value /
                           stakes / numbers / opinion / laughter) + delivery (loudness and pace relative to the rest of
                           the video) - filler words; every score keeps the reasons ("反轉", "提問"...) for the decision text
* :class:`ContentIndex`    vectorised lookups for the window search: time-weighted mean, peak and "first 3 seconds"
                           content of any number of windows (prefix sums + a sparse table)
* :func:`is_filler` / :func:`filler_spans`   filler words (嗯、呃、那個…) for the long-video editor

Everything is deterministic and offline.  The lexicons are heuristics (Traditional + Simplified Chinese + English); an
optional language-model judge (:mod:`autocut.autoedit.llm_judge`) can override the lexical part when the user opts in.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .captions import _is_punct, _is_strong, _timed_chars

# ------------------------------------------------------------------------------------------------ lexicons
# (category, weight, label shown to the user, pattern).  Simplified variants are included because speech recognition
# and downloaded subtitles are often Simplified even when the user works in Traditional.
_CATS: list[tuple[str, float, str, str]] = [
    ("question", 1.0, "提問",
     r"[？?]|為什麼|为什么|為甚麼|怎麼會|怎么会|怎麼辦|怎么办|如何|你知道|你知不知道|有沒有想過|有没有想过|是不是|到底|難道|难道|憑什麼|凭什么"
     r"|(?i:\bwhy\b|\bhow come\b|\bwhat if\b|\bdid you know\b)"),
    ("twist", 1.3, "反轉",
     r"沒想到|没想到|竟然|居然|結果|结果|但是|可是|不過|不过|其實|其实|原來|原来|反而|偏偏|誰知道|谁知道|後來才|后来才|突然|沒有人|没有人|真相"
     r"|(?i:\bbut\b|\bturns out\b|\bactually\b|\bsuddenly\b|\bnobody\b)"),
    ("emotion", 1.1, "情緒",
     r"超級|超扯|太扯|太誇張|太夸张|誇張|夸张|離譜|离谱|傻眼|嚇死|吓死|嚇到|吓到|天啊|天哪|我的天|笑死|崩潰|崩溃|氣死|气死|瘋了|疯了|可怕|恐怖"
     r"|厲害|厉害|絕了|绝了|好笑|感動|感动|哭了|爆炸|炸裂|震撼|不敢相信|[！!]{1,}|哇+"
     r"|(?i:\bcrazy\b|\binsane\b|\bunbelievable\b|\bamazing\b|\bwow\b|\bomg\b)"),
    ("value", 1.0, "乾貨",
     r"重點|重点|關鍵|关键|秘訣|秘诀|祕訣|方法|技巧|步驟|步骤|一定要|千萬|千万|記住|记住|最重要|結論|结论|核心|原因|建議|建议|注意|避免|錯誤|错误"
     r"|秘密|祕密|公式|訣竅|诀窍|教你|分享|攻略|懶人包|懒人包"
     r"|(?i:\bkey\b|\btip\b|\bsecret\b|\bmistake\b|\bimportant\b|\blesson\b|\btrick\b)"),
    ("stakes", 0.8, "利害",
     r"第一次|從來沒|从来没|差點|差点|失敗|失败|賠了|赔了|賺了|赚了|賺錢|赚钱|虧|亏|破產|破产|後悔|后悔|代價|代价|風險|风险|危險|危险|免費|免费"
     r"|唯一|所有人|每個人|每个人|一輩子|一辈子|改變|改变|人生"
     r"|(?i:\bfirst time\b|\bnever\b|\bfailed\b|\bmoney\b|\brisk\b|\bregret\b)"),
    ("number", 0.8, "具體數字",
     r"[0-9０-９]+(?:[.．,][0-9]+)?\s*(?:個|个|種|种|步|招|點|点|%|％|倍|元|塊|块|萬|万|億|亿|分鐘|分钟|秒|年|天|次|小時|小时|公斤|歲|岁|人)"
     r"|[一二兩两三四五六七八九十百千萬万億亿]+(?:個|个|種|种|步|招|點|点|倍|元|塊|块|萬|万|億|亿|分鐘|分钟|年|天|次|歲|岁)"),
    ("opinion", 0.6, "鮮明觀點",
     r"絕對|绝对|根本|完全|不可能|必須|必须|千萬不要|千万不要|老實說|老实说|說實話|说实话|坦白說|坦白说|我敢說|我敢说|一定是|錯了|错了|最好|最爛|最烂"
     r"|(?i:\babsolutely\b|\bnever ever\b|\bhonestly\b|\bthe best\b|\bthe worst\b)"),
    ("laugh", 0.8, "笑點",
     r"哈哈|呵呵|嘿嘿|笑死|好好笑|(?i:\bhaha\b|\blol\b)"),
]
_CAT_RE = [(name, w, label, re.compile(p)) for name, w, label, p in _CATS]
CATEGORY_LABEL = {name: label for name, _w, label, _p in _CATS}
CATEGORY_LABEL["llm"] = "AI 判定"

# filler words: PURE ones are always filler; the CONTEXT ones only when they stand alone (a whole word in the word
# timings, or repeated back to back) -- "那個人" is not a filler, "那個…那個" is.
FILLERS_PURE = ("嗯", "呃", "額", "欸", "誒", "唉", "噢", "um", "uh", "uhm", "umm", "erm", "hmm", "mm")
FILLERS_CONTEXT = ("那個", "那个", "就是說", "就是说", "就是", "然後", "然后", "對對對", "对对对", "對對", "对对", "對", "对",
                   "啊", "阿", "喔", "哦", "那", "這個", "这个", "基本上", "怎麼說", "怎么说", "所以說", "所以说", "like", "so",
                   "you know", "i mean", "kind of", "sort of")
_PURE_RE = re.compile(r"^(?:" + "|".join(re.escape(f) for f in FILLERS_PURE) + r")+$", re.IGNORECASE)
_FILLER_ANY = re.compile("|".join(re.escape(f) for f in sorted(FILLERS_PURE + FILLERS_CONTEXT, key=len, reverse=True)),
                         re.IGNORECASE)
_STRIP = " 　,，、。.！!？?；;：:…~～-—\"'「」『』()（）"

HOOK_SPAN = 3.0             # seconds at the head of a short whose content counts as its hook
MIN_UNIT_CHARS = 2          # a sentence with fewer content characters than this scores 0


# ------------------------------------------------------------------------------------------------- data
@dataclass
class Sentence:
    start: float
    end: float
    text: str
    words: list = field(default_factory=list)       # [(start, end, word)] when the transcript has word timings
    score: float = 0.0
    lexical: float = 0.0
    delivery: float = 0.0
    filler_ratio: float = 0.0
    tags: list = field(default_factory=list)        # category names that fired, strongest first
    note: str = ""                                  # the AI judge's own short reason (when it was used)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def reasons(self) -> str:
        if self.note:
            return "AI：" + self.note
        return "、".join(CATEGORY_LABEL.get(t, t) for t in self.tags[:3] if t != "llm")


def content_chars(text: str) -> int:
    return sum(1 for c in text if not c.isspace() and not _is_punct(c))


def _norm(word: str) -> str:
    return str(word).strip().strip(_STRIP).lower()


def is_filler(word: str, prev: str = "", nxt: str = "") -> bool:
    """``word`` (one recognised word) is a filler: a pure one (嗯/呃/um), or a context one (那個/就是/然後/對/like) that
    STANDS ALONE -- nothing attached before and after (``prev`` / ``nxt`` empty: a pause or the edge of the sentence).

    Repetition alone is not enough: 「對對方很好」 tokenised per character has two 「對」 in a row that are words.  A run of
    the same context word is judged as a whole by :func:`filler_spans` (「對對對。」 between pauses is a filler)."""
    w = _norm(word)
    if not w:
        return False
    if _PURE_RE.match(w):
        return True
    if w in FILLERS_CONTEXT:
        return not _norm(prev) and not _norm(nxt)
    return False


# --------------------------------------------------------------------------------------------- sentences
def sentences(segments: Optional[Sequence[dict]]) -> list[Sentence]:
    """Sentence units in source seconds.  Segments are cut after sentence-final punctuation; the word timings that fall
    inside a unit travel with it.  Units are sorted, non-empty and never overlap."""
    out: list[Sentence] = []
    for seg in segments or []:
        units = _timed_chars(seg)
        if not units:
            continue
        words = []
        for w in seg.get("words") or []:
            try:
                words.append((float(w["start"]), float(w["end"]), str(w.get("word", ""))))
            except (KeyError, TypeError, ValueError):
                continue
        cur: list = []
        n = len(units)
        for i, u in enumerate(units):
            cur.append(u)
            last = i + 1 >= n
            if last or (_is_strong(units, i) and not _is_punct(units[i + 1][0])):
                text = "".join(c[0] for c in cur).strip()
                if text and content_chars(text) > 0:
                    real = [c for c in cur if not c[0].isspace() and not _is_punct(c[0])] or cur
                    a, b = float(real[0][1]), float(real[-1][2])
                    if b <= a:
                        b = a + 0.05
                    ws = [w for w in words if a - 0.05 <= (w[0] + w[1]) / 2.0 <= b + 0.05]
                    out.append(Sentence(a, b, text, ws))
                cur = []
    out.sort(key=lambda s: (s.start, s.end))
    fixed: list[Sentence] = []
    for s in out:                                             # overlapping cues (bad subtitle files): clip to the previous
        if fixed and s.start < fixed[-1].end:
            s.start = fixed[-1].end
            if s.end <= s.start:
                continue
        fixed.append(s)
    return fixed


def lexical_signals(text: str) -> tuple[float, list[str]]:
    """``(raw score, categories strongest first)``: every category counts at most twice."""
    raw = 0.0
    fired: list[tuple[float, str]] = []
    for name, w, _label, rx in _CAT_RE:
        hits = len(rx.findall(text))
        if hits:
            raw += w * min(hits, 2)
            fired.append((w * min(hits, 2), name))
    fired.sort(key=lambda x: (-x[0], x[1]))
    return raw, [n for _s, n in fired]


def _filler_ratio(s: Sentence) -> float:
    chars = content_chars(s.text)
    if chars == 0:
        return 0.0
    if s.words:
        toks = [w[2] for w in s.words]
        filler = sum(content_chars(t) for i, t in enumerate(toks)
                     if is_filler(t, toks[i - 1] if i else "", toks[i + 1] if i + 1 < len(toks) else ""))
        return min(1.0, filler / chars)
    if content_chars(_FILLER_ANY.sub("", s.text)) == 0:
        return 1.0                                            # nothing but fillers: 「嗯，那個，就是說。」
    filler = sum(content_chars(m.group(0)) for m in _FILLER_ANY.finditer(s.text)
                 if _PURE_RE.match(_norm(m.group(0))))          # without word timings only the unambiguous ones count
    return min(1.0, filler / chars)


def _rank(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n <= 1:
        return np.full(n, 0.5)
    s = np.sort(x)
    lo = np.searchsorted(s, x, side="left")
    hi = np.searchsorted(s, x, side="right")
    return ((lo + hi - 1) / 2.0) / (n - 1)


def score_sentences(units: list[Sentence], feat=None, overrides: Optional[dict] = None) -> list[Sentence]:
    """Fill ``score`` (0..1), ``lexical``, ``delivery``, ``filler_ratio`` and ``tags`` of every unit (in place).

    score = 0.65 x lexical + 0.35 x delivery, minus up to 0.5 for filler-heavy units; units with fewer than
    ``MIN_UNIT_CHARS`` content characters score 0.  ``feat`` (a ``scoring.Feat``) gives loudness; without it delivery
    is pace only.  ``overrides`` = ``{index: (score 0..1, reason)}`` from the optional language-model judge replaces the
    lexical part of those units.
    """
    if not units:
        return units
    loud = np.zeros(len(units))
    pace = np.zeros(len(units))
    has_loud = feat is not None and getattr(feat, "has_rms", False)
    for i, s in enumerate(units):
        pace[i] = content_chars(s.text) / max(s.duration, 0.3)
        if has_loud:
            a, b = int(feat.a_idx(s.start)), int(feat.a_idx(s.end))
            if b > a:
                seg = feat.rms[a:b]
                voiced = feat.voiced[a:b]
                loud[i] = float(np.mean(seg[voiced])) if voiced.any() else float(np.mean(seg))
            else:
                loud[i] = -90.0
    # rounded first: float noise (2.75 vs 2.7499999 s) must not become a rank difference
    loud_r = _rank(np.round(loud, 0)) if has_loud else np.full(len(units), 0.5)
    pace_r = _rank(np.round(pace, 1))
    # delivery: louder than usual and a lively (not rushed, not dragging) pace
    pace_shape = 1.0 - np.abs(pace_r - 0.65) / 0.65
    delivery = 0.7 * loud_r + 0.3 * np.clip(pace_shape, 0.0, 1.0) if has_loud else np.clip(pace_shape, 0.0, 1.0)
    for i, s in enumerate(units):
        raw, tags = lexical_signals(s.text)
        lex = 1.0 - math.exp(-raw / 1.6)
        if overrides and i in overrides:
            val, why = overrides[i]
            lex = min(max(float(val), 0.0), 1.0)
            tags = ["llm"] + tags
            s.note = str(why or "")
        s.lexical = round(lex, 4)
        s.delivery = round(float(delivery[i]), 4)
        s.filler_ratio = round(_filler_ratio(s), 4)
        s.tags = tags
        if content_chars(s.text) < MIN_UNIT_CHARS:
            s.score = 0.0
            continue
        sc = 0.65 * lex + 0.35 * s.delivery - 0.5 * s.filler_ratio
        s.score = round(min(max(sc, 0.0), 1.0), 4)
    return units


# ------------------------------------------------------------------------------------------- index
class ContentIndex:
    """Vectorised content lookups over sorted sentence units (for millions of candidate windows)."""

    def __init__(self, units: list[Sentence]) -> None:
        self.units = units
        n = len(units)
        self.n = n
        self.start = np.asarray([u.start for u in units], dtype=np.float64)
        self.end = np.asarray([u.end for u in units], dtype=np.float64)
        self.mid = (self.start + self.end) / 2.0
        self.score = np.asarray([u.score for u in units], dtype=np.float64)
        # "a rich window" for THIS video: the 90th percentile of the sentence scores (never below 0.3)
        self.rich = max(0.3, float(np.percentile(self.score, 90))) if n else 1.0
        dur = np.maximum(self.end - self.start, 1e-3)
        self.P_sd = np.concatenate([[0.0], np.cumsum(self.score * dur)])
        self.P_d = np.concatenate([[0.0], np.cumsum(dur)])
        # sparse table for range maximum
        self.table = [self.score.copy()]
        k = 1
        while (1 << k) <= n:
            prev = self.table[-1]
            half = 1 << (k - 1)
            self.table.append(np.maximum(prev[:n - (1 << k) + 1], prev[half:half + n - (1 << k) + 1]))
            k += 1

    def __len__(self) -> int:
        return self.n

    def span(self, ts, te) -> tuple[np.ndarray, np.ndarray]:
        """Index range ``[i0, i1)`` of the units whose middle lies inside ``[ts, te)``."""
        i0 = np.searchsorted(self.mid, np.asarray(ts, dtype=np.float64), side="left")
        i1 = np.searchsorted(self.mid, np.asarray(te, dtype=np.float64), side="left")
        return i0, i1

    def range_max(self, i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
        out = np.zeros(np.shape(i0))
        ok = i1 > i0
        if not ok.any() or self.n == 0:
            return out
        a, b = i0[ok], i1[ok]
        length = b - a
        k = np.floor(np.log2(length)).astype(np.int64)
        res = np.zeros(a.size)
        for kk in np.unique(k):
            m = k == kk
            t = self.table[int(kk)]
            res[m] = np.maximum(t[a[m]], t[b[m] - (1 << int(kk))])
        out[ok] = res
        return out

    def window(self, ts, te) -> dict:
        """Per window: ``mean`` (time-weighted content), ``peak`` (best unit), ``hook`` (best unit starting in the first
        ``HOOK_SPAN`` s) and ``n`` (units inside)."""
        ts = np.asarray(ts, dtype=np.float64)
        te = np.asarray(te, dtype=np.float64)
        i0, i1 = self.span(ts, te)
        dsum = self.P_d[i1] - self.P_d[i0]
        mean = np.where(dsum > 0, (self.P_sd[i1] - self.P_sd[i0]) / np.maximum(dsum, 1e-9), 0.0)
        peak = self.range_max(i0, i1)
        h0 = np.searchsorted(self.start, ts - 0.3, side="left")
        h1 = np.searchsorted(self.start, ts + HOOK_SPAN, side="right")
        hook = self.range_max(h0, np.maximum(h1, h0))
        return {"mean": mean, "peak": peak, "hook": hook, "n": (i1 - i0).astype(np.float64)}

    def best_in(self, t0: float, t1: float, k: int = 2) -> list[Sentence]:
        """The ``k`` best units inside ``t0..t1`` (for the decision text)."""
        inside = [u for u in self.units if t0 <= (u.start + u.end) / 2.0 < t1 and u.score > 0]
        inside.sort(key=lambda u: (-u.score, u.start))
        return inside[:k]


def quote(text: str, n: int = 18) -> str:
    t = " ".join(str(text).split()).strip(_STRIP)
    return t if len(t) <= n else t[:n] + "…"


# -------------------------------------------------------------------------------------- filler spans
def filler_spans(units: list[Sentence], *, margin: float = 0.02, min_len: float = 0.08,
                 pause: float = 0.25) -> list[tuple[float, float, str]]:
    """Source ranges of filler words (needs word timings): ``[(start, end, word)]``.  Neighbouring fillers merge.

    Pure fillers (嗯/呃) always count.  A context filler counts only when its whole run of repeats (那個 / 對對對) stands
    alone: a pause longer than ``pause`` (or the sentence edge) before the run AND after it."""
    out: list[tuple[float, float, str]] = []

    def add(a: float, b: float, w: str) -> None:
        a2, b2 = a - margin, b + margin
        if out and a2 <= out[-1][1] + 0.05:
            out[-1] = (out[-1][0], max(out[-1][1], b2), out[-1][2] + w)
        else:
            out.append((a2, b2, w))

    for s in units:
        toks = s.words
        i = 0
        while i < len(toks):
            a, b, w = toks[i]
            nw = _norm(w)
            if nw and _PURE_RE.match(nw):
                if b - a >= min_len:
                    add(a, b, w.strip())
                i += 1
                continue
            if nw in FILLERS_CONTEXT:
                j = i
                while j + 1 < len(toks) and _norm(toks[j + 1][2]) == nw and toks[j + 1][0] - toks[j][1] <= pause:
                    j += 1
                alone_before = i == 0 or a - toks[i - 1][1] > pause
                alone_after = j + 1 >= len(toks) or toks[j + 1][0] - toks[j][1] > pause
                if alone_before and alone_after and toks[j][1] - a >= min_len:
                    add(a, toks[j][1], "".join(t[2].strip() for t in toks[i:j + 1]))
                i = j + 1
                continue
            i += 1
    return out


def repeated_word_spans(units: list[Sentence], *, min_len: float = 0.08) -> list[tuple[float, float, str]]:
    """Stutters: a word immediately repeated -> every copy but the last is cut.

    A word must repeat at least THREE times in a row (我我我 / 我覺得我覺得我覺得 / the the the).  Doubling is normal
    language and is never cut: 聊聊、謝謝、天天 (grammar), 非常非常、很多很多、very very (emphasis)."""
    out: list[tuple[float, float, str]] = []
    for s in units:
        toks = s.words
        i = 0
        while i < len(toks):
            w = _norm(toks[i][2])
            j = i
            while j + 1 < len(toks) and w and _norm(toks[j + 1][2]) == w and toks[j + 1][0] - toks[j][1] < 0.4:
                j += 1
            run = j - i + 1
            if run >= 3 and not is_filler(toks[i][2], "", ""):
                a, b = toks[i][0], toks[j][0]
                if b - a >= min_len:
                    out.append((a, b, toks[i][2].strip()))
            i = j + 1
    return out


# ------------------------------------------------------------------------------------ transcript source
SUBTITLE_EXTS = (".srt", ".vtt", ".ass")


def sidecar_subtitle(video_path: str) -> str:
    """A subtitle file sitting next to the video with the same name (``talk.srt``, ``talk.zh-TW.srt``, ``talk.zh.vtt``…),
    or ``""``.  Exact stem matches win over language-tagged ones; the choice is deterministic (sorted)."""
    from pathlib import Path
    try:
        p = Path(str(video_path))
        folder = p.parent
        if not folder.is_dir():
            return ""
        exact, tagged = [], []
        for f in sorted(folder.iterdir(), key=lambda x: x.name):
            if not f.is_file() or f.suffix.lower() not in SUBTITLE_EXTS:
                continue
            stem = f.name[: -len(f.suffix)]
            if stem == p.stem:
                exact.append(f)
            elif stem.startswith(p.stem + "."):
                tagged.append(f)
        pick = (exact or tagged or [None])[0]
        return str(pick) if pick is not None else ""
    except OSError:
        return ""


def load_transcript_file(path: str) -> list[dict]:
    """``.srt/.vtt/.ass`` -> transcript segments in source seconds (raises ``OSError`` / ``ValueError``)."""
    from ..core.subtitles import load_subtitle_file
    return [{"start": float(a), "end": float(b), "text": str(t)} for a, b, t in load_subtitle_file(path)]


def choose_transcript(asset, ev, transcript_file: str = "", *, use_speech: bool = True, notes: Optional[list] = None
                      ) -> tuple[Optional[list], dict]:
    """The text that guides the edit, in priority order:

    1. ``transcript_file`` given by the user (their own subtitles are usually the most accurate text);
    2. a subtitle file next to the video with the same name (「一鍵字幕」 writes exactly that);
    3. the speech-recognition transcript in the evidence (has word timings).

    Returns ``(segments or None, info)``; ``info = {"source": "file|sidecar|speech|none", "name", "sha256", "words"}``
    goes into the plan so the plan hash changes when the text changes.  Reasons are appended to ``notes`` (繁中).
    """
    from pathlib import Path
    from ..util import canonical_json, sha256_hex
    notes = notes if notes is not None else []
    tried: list[tuple[str, str]] = []
    if transcript_file:
        tried.append(("file", transcript_file))
    side = sidecar_subtitle(getattr(asset, "path", ""))
    if side and (not transcript_file or Path(side).resolve() != Path(transcript_file).resolve()):
        tried.append(("sidecar", side))
    for kind, path in tried:
        try:
            segs = load_transcript_file(path)
        except (OSError, ValueError, UnicodeError) as exc:
            notes.append(f"讀取字幕檔失敗（{Path(path).name}）：{exc}")
            continue
        if not segs:
            notes.append(f"字幕檔裡沒有可用的字幕：{Path(path).name}")
            continue
        if ev is not None and getattr(ev, "speech", None):
            notes.append(f"以字幕檔「{Path(path).name}」的文字判斷重點；語音辨識的逐字時間只用來避免切在字中間與剪水詞。")
        elif kind == "sidecar":
            notes.append(f"找到影片旁的字幕檔「{Path(path).name}」，以它的文字判斷重點。")
        else:
            notes.append(f"以字幕檔「{Path(path).name}」的文字判斷重點。")
        return segs, {"source": kind, "name": Path(path).name, "sha256": sha256_hex(canonical_json(segs)),
                      "words": False}
    if use_speech and ev is not None and getattr(ev, "speech", None) is not None:
        segs = [dict(s) for s in ev.speech]
        return segs, {"source": "speech", "name": "", "sha256": sha256_hex(canonical_json(segs)),
                      "words": any(s.get("words") for s in segs)}
    notes.append("沒有字幕檔也沒有語音辨識逐字稿，無法依「說了什麼」找重點，只能依人聲、響度、畫面挑片段"
                 "（建議：先按「一鍵產生字幕」，或在影片旁放同名的 .srt 字幕檔）。")
    return None, {"source": "none", "name": "", "sha256": "", "words": False}


def word_segments(ev, asset=None) -> list:
    """Segments that carry word timings (for mid-word checks / filler removal): the speech-recognition transcript in the
    evidence, else the ``<video name>.words.json`` that 「一鍵字幕」 leaves next to the video, else ``[]``."""
    segs = [dict(s) for s in (getattr(ev, "speech", None) or []) if s.get("words")]
    if segs or asset is None:
        return segs
    import json
    from pathlib import Path
    try:
        p = Path(str(getattr(asset, "path", "")))
        data = json.loads(p.with_name(p.stem + ".words.json").read_text(encoding="utf-8"))
        return [dict(s) for s in data.get("segments", []) if isinstance(s, dict) and s.get("words")]
    except (OSError, ValueError, AttributeError):
        return []
