"""Optional AI judge of highlights ("爆點") with Claude -- strictly opt-in.

``ShortsParams.highlight_engine = "claude"`` (or ``LongformParams``) sends **only the subtitle text with its time codes**
to the Claude API (never audio or video) and asks for a 0..10 rating of every sentence as short-video material.  The
ratings replace the lexical part of :func:`autocut.autoedit.highlights.score_sentences`; delivery (loudness / pace)
and the filler penalty stay local.

* Needs the optional ``anthropic`` package (``pip install -r requirements-ai.txt``) and credentials the SDK can find
  (``ANTHROPIC_API_KEY``; the GUI can set it for the running process).  Missing either -> a 繁中 note and the local
  rules are used; planning never fails because of this module.
* Results are cached (``<cache>/highlights/<hash>.json``, keyed by the sentences, the model and ``PROMPT_VERSION``), so
  re-planning the same video is offline and gives the same plan hash.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional

from ..media.ffmpeg import Cancelled
from ..util import atomic_write_text, cache_dir, canonical_json, sha256_hex

MODEL = "claude-opus-5-5"
PROMPT_VERSION = 1
BATCH = 250                     # sentences per request (keeps every answer well inside max_tokens)
INSTALL_HINT = "尚未安裝 AI 判斷套件：請執行 pip install -r requirements-ai.txt"
KEY_HINT = "沒有設定 Claude API 金鑰（環境變數 ANTHROPIC_API_KEY，或在自動剪輯視窗的「AI 判斷爆點」設定）"

SYSTEM = (
    "You are a senior short-form video editor (TikTok / Reels / YouTube Shorts) working on Chinese or English talk "
    "content (podcasts, interviews, vlogs, lectures). You receive the numbered sentences of one video's subtitles with "
    "their time codes. Rate EVERY sentence from 0 to 10 for how strongly it works as highlight material that would "
    "stop a viewer from scrolling: a hook or provocative question, a surprising twist or reveal, a punchline or laugh, "
    "a strong emotion, a bold opinion, a concrete story moment with stakes, or a crisp valuable insight. Small talk, "
    "greetings, filler, repetition, housekeeping and sentences that only make sense with missing context score low. "
    "Judge each sentence in the context of its neighbours (a payoff can make its setup valuable too). Give a very "
    "short reason in Traditional Chinese (at most 12 characters) for every sentence scoring 6 or more, else an empty "
    "string. Return one item per input sentence, using its index."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"i": {"type": "integer"}, "score": {"type": "number"}, "why": {"type": "string"}},
                "required": ["i", "score", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def _safe(exc: BaseException) -> str:
    """A FIXED, secret-free description of an error for ``plan.notes`` (saved in the project file).  The exception's own
    text is never used: any formatting of it could carry an API key or token (REVIEW-3, Codex)."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return f"金鑰無效或沒有權限（HTTP {status}）"
    if status == 429:
        return "超過額度或請求太頻繁（HTTP 429）"
    if isinstance(status, int) and status >= 500:
        return f"AI 服務暫時無法使用（HTTP {status}）"
    if isinstance(status, int):
        return f"請求被拒絕（HTTP {status}）"
    name = type(exc).__name__
    if "Connection" in name or "Timeout" in name:
        return "網路連線失敗或逾時"
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return "AI 回覆的格式不正確"
    return f"錯誤類型 {name}"


def available() -> tuple[bool, str]:
    """``(usable, reason)``: the package imports and some credential is configured."""
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False, INSTALL_HINT
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_PROFILE")):
        return False, KEY_HINT
    return True, ""


def _fmt(t: float) -> str:
    m, s = divmod(max(0.0, float(t)), 60.0)
    return f"{int(m):02d}:{s:04.1f}"


def _cache_file(units, model: str):
    key = sha256_hex(canonical_json({"v": PROMPT_VERSION, "model": model,
                                     "u": [[round(u.start, 2), round(u.end, 2), u.text] for u in units]}))[:24]
    return cache_dir("highlights") / f"{key}.json"


def _ask(client, lines: list[str], language: str) -> list[dict]:
    """One request: returns the ``items`` list (raises on refusal / malformed output)."""
    user = (f"Language: {language or 'auto'}. Sentences ({len(lines)}):\n" + "\n".join(lines))
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        msg = stream.get_final_message()
    if msg.stop_reason == "refusal":
        raise RuntimeError("AI 拒絕評分這段內容")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("AI 回覆被截斷")
    text = next((b.text for b in msg.content if getattr(b, "type", "") == "text"), "")
    data = json.loads(text)
    return list(data.get("items") or [])


def judge_sentences(units, *, engine: str = "claude", language: str = "zh", notes: Optional[list] = None,
                    cancel: Optional[threading.Event] = None, client=None) -> tuple[Optional[dict], dict]:
    """``(overrides, info)`` for :func:`highlights.score_sentences`: ``{index: (score 0..1, reason)}``.

    Any failure (not installed, no key, network, refusal, malformed answer) returns ``(None, info)`` with a note, so the
    caller falls back to the local rules.  ``client`` is injectable for tests.
    """
    notes = notes if notes is not None else []
    info = {"engine": "local", "requested": engine, "model": MODEL, "prompt_version": PROMPT_VERSION}
    if engine != "claude":
        notes.append(f"不支援的爆點判斷方式：{engine}，改用本機規則。")
        return None, info
    if not units:
        return None, info
    path = _cache_file(units, MODEL)
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        over = {int(k): (float(v[0]), str(v[1])) for k, v in cached["scores"].items()}
        info.update(engine="claude", cached=True, sha256=sha256_hex(canonical_json(cached["scores"])))
        notes.append(f"爆點由 AI（{MODEL}）判斷（使用先前的結果，未重新連線）。")
        return over, info
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if client is None:
        ok, why = available()
        if not ok:
            notes.append(f"AI 判斷爆點無法使用：{why}；改用本機規則。")
            return None, info
        try:
            import anthropic
            client = anthropic.Anthropic(timeout=300.0)
        except Exception as exc:                          # broken install / bad configuration -> local rules
            notes.append(f"AI 判斷爆點無法啟動（{_safe(exc)}），改用本機規則。")
            return None, info
    lines = [f"[{i}] {_fmt(u.start)}–{_fmt(u.end)} {' '.join(u.text.split())}" for i, u in enumerate(units)]
    scores: dict[str, list] = {}
    try:
        for b0 in range(0, len(lines), BATCH):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            # each batch also sees a few neighbouring lines for context, but only its own indexes are kept
            lo, hi = b0, min(b0 + BATCH, len(lines))
            ctx = lines[max(0, lo - 5):lo] + lines[lo:hi] + lines[hi:hi + 5]
            for it in _ask(client, ctx, language):
                i = int(it.get("i", -1))
                if lo <= i < hi:
                    s = min(max(float(it.get("score", 0.0)), 0.0), 10.0) / 10.0
                    scores[str(i)] = [round(s, 3), str(it.get("why", ""))[:24]]
    except Cancelled:
        raise
    except Exception as exc:                                  # network, auth, refusal, schema -> local rules
        notes.append(f"AI 判斷爆點失敗（{_safe(exc)}），改用本機規則。")
        return None, info
    if not scores:
        notes.append("AI 沒有回傳任何評分，改用本機規則。")
        return None, info
    try:
        atomic_write_text(path, json.dumps({"model": MODEL, "v": PROMPT_VERSION, "scores": scores}, ensure_ascii=False))
    except OSError:
        pass
    info.update(engine="claude", cached=False, sha256=sha256_hex(canonical_json(scores)))
    notes.append(f"爆點由 AI（{MODEL}）判斷：只傳送了字幕文字與時間碼，沒有上傳影片或聲音。")
    return {int(k): (v[0], v[1]) for k, v in scores.items()}, info
