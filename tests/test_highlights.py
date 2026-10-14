"""Content highlights ("爆點"): sentence units, lexical + delivery scores, the vectorised window index, filler words,
the transcript source priority and the opt-in AI judge (with a fake client -- never the network)."""
from __future__ import annotations

import json
import threading
import types

import numpy as np
import pytest

from autocut.analysis.evidence import Evidence
from autocut.autoedit import highlights as hl
from autocut.autoedit import llm_judge as lj
from autocut.autoedit import scoring as sc
from autocut.autoedit.config import ShortsParams
from autocut.autoedit.shorts import plan_shorts
from autocut.core.project import Asset, Project, Sequence

AFPS, VFPS = 20.0, 5.0

DULL = "今天天氣不錯，我們來聊聊日常。"
HOT = "但是沒想到，我竟然一個月賠了300萬！"


def test_sentences_split_at_sentence_ends_and_keep_their_words():
    segs = [{"start": 0.0, "end": 6.0, "text": "大家好。為什麼會這樣？",
             "words": [{"start": 0.0, "end": 1.0, "word": "大家好"}, {"start": 3.0, "end": 4.0, "word": "為什麼"},
                       {"start": 4.0, "end": 5.5, "word": "會這樣"}]},
            {"start": 6.0, "end": 8.0, "text": "沒有標點的字幕"}]
    u = hl.sentences(segs)
    assert [x.text for x in u] == ["大家好。", "為什麼會這樣？", "沒有標點的字幕"]
    assert u[0].end <= u[1].start and [w[2] for w in u[1].words] == ["為什麼", "會這樣"]
    assert hl.sentences([{"start": 0, "end": 1, "text": "  "}]) == []


def test_hot_sentences_outscore_dull_and_filler_ones_with_readable_reasons():
    u = hl.score_sentences(hl.sentences([
        {"start": 0, "end": 4, "text": DULL}, {"start": 4, "end": 8, "text": HOT},
        {"start": 8, "end": 10, "text": "嗯，那個，就是說。"}, {"start": 10, "end": 14, "text": "關鍵就是這三個步驟。"}]))
    dull, hot, filler, value = u
    assert hot.score > value.score > dull.score > filler.score == 0.0
    assert "反轉" in hot.reasons() and "具體數字" in hot.reasons() and "乾貨" in value.reasons()
    assert filler.filler_ratio == 1.0


def test_delivery_rewards_the_louder_sentence_when_the_words_are_equally_plain():
    ev = Evidence.empty("a", 20.0, audio_fps=AFPS)
    rms = np.full(400, -32.0)
    rms[200:300] = -20.0                                   # 10-15 s is shouted
    ev.rms_db, ev.voice = rms.tolist(), [1.0] * 400
    segs = [{"start": 0, "end": 5, "text": DULL}, {"start": 10, "end": 15, "text": DULL}]
    feat = sc.Feat(ev, segs)
    a, b = feat.units
    assert b.delivery > a.delivery and b.score > a.score


def test_content_index_mean_peak_and_hook_match_a_brute_force_answer():
    rng = np.random.default_rng(3)
    units = []
    t = 0.0
    for _ in range(60):
        d = float(rng.uniform(0.5, 4.0))
        units.append(hl.Sentence(t, t + d, "x", score=float(rng.uniform(0, 1))))
        t += d + float(rng.uniform(0, 1))
    ci = hl.ContentIndex(units)
    ts = rng.uniform(0, t, 300)
    te = ts + rng.uniform(1, 40, 300)
    w = ci.window(ts, te)
    for k in range(300):
        inside = [u for u in units if ts[k] <= (u.start + u.end) / 2 < te[k]]
        peak = max((u.score for u in inside), default=0.0)
        dsum = sum(max(u.end - u.start, 1e-3) for u in inside)
        mean = sum(u.score * max(u.end - u.start, 1e-3) for u in inside) / dsum if inside else 0.0
        hook = max((u.score for u in units if ts[k] - 0.3 <= u.start <= ts[k] + hl.HOOK_SPAN), default=0.0)
        assert w["peak"][k] == pytest.approx(peak) and w["mean"][k] == pytest.approx(mean) and w["hook"][k] == pytest.approx(hook)


def test_fillers_and_stutters_need_context():
    assert hl.is_filler("嗯") and hl.is_filler("呃呃") and hl.is_filler("um")
    assert not hl.is_filler("那個", "那個", "我") and hl.is_filler("然後", "", "")   # repetition alone is not enough
    assert not hl.is_filler("那個", "我", "人") and not hl.is_filler("重點")
    u = hl.sentences([{"start": 0, "end": 5, "text": "嗯我我覺得那個人很好",
                       "words": [{"start": 0.0, "end": 0.4, "word": "嗯"}, {"start": 0.8, "end": 1.0, "word": "我"},
                                 {"start": 1.05, "end": 1.25, "word": "我"}, {"start": 1.3, "end": 1.7, "word": "覺得"},
                                 {"start": 1.75, "end": 2.1, "word": "那個"}, {"start": 2.1, "end": 2.4, "word": "人"},
                                 {"start": 2.4, "end": 3.0, "word": "很好"}]}])
    spans = hl.filler_spans(u)
    assert len(spans) == 1 and spans[0][2] == "嗯" and spans[0][0] < 0.0 + 0.05 and spans[0][1] <= 0.45
    assert hl.repeated_word_spans(u) == []                   # 「我我」: doubling a character is not a stutter
    u3 = hl.sentences([{"start": 0, "end": 3, "text": "我我我覺得聊聊天天",
                        "words": [{"start": 0.0, "end": 0.2, "word": "我"}, {"start": 0.25, "end": 0.45, "word": "我"},
                                  {"start": 0.5, "end": 0.7, "word": "我"}, {"start": 0.75, "end": 1.1, "word": "覺得"},
                                  {"start": 1.2, "end": 1.4, "word": "聊"}, {"start": 1.45, "end": 1.65, "word": "聊"},
                                  {"start": 1.7, "end": 1.9, "word": "天"}, {"start": 1.95, "end": 2.2, "word": "天"}]}])
    assert hl.repeated_word_spans(u3) == [(0.0, 0.5, "我")]  # keep the last 我, keep 聊聊 / 天天
    u4 = hl.sentences([{"start": 0, "end": 2, "text": "the the the cat", "words": [
        {"start": 0.0, "end": 0.2, "word": " the"}, {"start": 0.3, "end": 0.5, "word": " the"},
        {"start": 0.55, "end": 0.75, "word": " the"}, {"start": 0.8, "end": 1.1, "word": " cat"}]}])
    assert hl.repeated_word_spans(u4) == [(0.0, 0.55, "the")]


def test_transcript_priority_file_then_sidecar_then_speech(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    asset = types.SimpleNamespace(path=str(video))
    ev = Evidence.empty("a", 10.0)
    ev.speech = [{"start": 0, "end": 1, "text": "語音辨識", "words": [{"start": 0, "end": 1, "word": "語音辨識"}]}]
    notes: list = []
    segs, info = hl.choose_transcript(asset, ev, "", notes=notes)
    assert info["source"] == "speech" and segs[0]["text"] == "語音辨識"
    (tmp_path / "talk.zh-TW.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\n旁邊的字幕\n", encoding="utf-8")
    segs, info = hl.choose_transcript(asset, ev, "", notes=notes)
    assert info["source"] == "sidecar" and segs[0]["text"] == "旁邊的字幕" and info["sha256"]
    mine = tmp_path / "mine.srt"
    mine.write_text("1\n00:00:03,000 --> 00:00:04,500\n我的字幕\n", encoding="utf-8")
    segs, info = hl.choose_transcript(asset, ev, str(mine), notes=notes)
    assert info["source"] == "file" and segs == [{"start": 3.0, "end": 4.5, "text": "我的字幕"}]
    assert hl.word_segments(ev)[0]["words"]
    segs, info = hl.choose_transcript(asset, None, str(tmp_path / "missing.srt"), notes=notes)
    assert info["source"] == "sidecar" and any("讀取字幕檔失敗" in n for n in notes)


# ---------------------------------------------------------------------------------------- the planner
def _ev_with_story(dur=180.0):
    """Even, continuous talking: only the TEXT tells the 100-140 s stretch apart."""
    ev = Evidence.empty("a1", dur, width=1920, height=1080, has_audio=True, audio_fps=AFPS, visual_fps=VFPS, fps=30.0)
    na, nv = int(dur * AFPS), int(dur * VFPS)
    ev.voice, ev.rms_db = [1.0] * na, [-28.0] * na
    sil = []
    segs = []
    t = 0.0
    while t + 5.0 <= dur:
        text = HOT if 100.0 <= t < 140.0 else DULL
        segs.append({"start": t, "end": t + 4.5, "text": text})
        sil.append([t + 4.5, t + 5.0])
        t += 5.0
    ev.silences = sil
    ev.motion, ev.sharp, ev.bright = [0.05] * nv, [100.0] * nv, [0.5] * nv
    ev.faces, ev.face_sample_mask = [[[0.4, 0.3, 0.1, 0.2]]] * nv, [1] * nv
    ev.motion_cx, ev.motion_cy = [-1.0] * nv, [-1.0] * nv
    return ev, segs


def _proj(path, dur=180.0):
    proj = Project.new("t")
    proj.assets["a1"] = Asset(id="a1", path=path, name="talk.mp4", kind="video", duration=dur, width=1920,
                              height=1080, fps=30.0, has_video=True, has_audio=True, fingerprint="fp")
    return proj


@pytest.fixture
def patched(monkeypatch):
    class H:
        ev = None
    h = H()
    monkeypatch.setattr("autocut.analysis.pipeline.analyze_asset", lambda asset, params=None, **kw: h.ev)
    return h


def test_the_short_lands_on_what_was_said_from_a_subtitle_file(tmp_path, patched):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    ev, segs = _ev_with_story()
    patched.ev = ev                                               # no speech recognition at all
    from autocut.core.subtitles import to_srt
    srt = tmp_path / "subs.srt"
    srt.write_text(to_srt([(s["start"], s["end"], s["text"]) for s in segs]), encoding="utf-8")
    proj = _proj(str(video))
    plan = plan_shorts(proj, "a1", ShortsParams(count=1, target_seconds=30, transcript_file=str(srt), captions="off"))
    d = next(d for d in plan.decisions if d["kind"] == "short")
    assert 95.0 <= d["source"]["t0"] and d["source"]["t1"] <= 145.0          # the story, not the small talk
    assert "爆點句" in d["why"] and "沒想到" in d["why"] and d["highlights"][0]["score"] > 0.6
    assert plan.brief["transcript"]["source"] == "file"
    assert "沒想到" in Sequence.from_dict(plan.sequences[0]).name
    # without the text the planner cannot know (and says so)
    plan2 = plan_shorts(proj, "a1", ShortsParams(count=1, target_seconds=30, captions="off"))
    assert plan2.brief["transcript"]["source"] == "none" and any("一鍵產生字幕" in n for n in plan2.notes)


def test_sidecar_subtitle_is_found_and_used_for_captions_too(tmp_path, patched):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    ev, segs = _ev_with_story()
    patched.ev = ev
    from autocut.core.subtitles import to_srt
    (tmp_path / "talk.srt").write_text(to_srt([(s["start"], s["end"], s["text"]) for s in segs]), encoding="utf-8")
    plan = plan_shorts(_proj(str(video)), "a1", ShortsParams(count=1, target_seconds=30, captions="auto"))
    assert plan.brief["transcript"]["source"] == "sidecar"
    assert Sequence.from_dict(plan.sequences[0]).cues                      # captions from the same file


# ---------------------------------------------------------------------------------------- AI judge
class _FakeStream:
    def __init__(self, reply, record):
        self.reply, self.record = reply, record

    def __call__(self, **kw):
        self.record.append(kw)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return types.SimpleNamespace(stop_reason=self.reply.get("stop", "end_turn"),
                                     content=[types.SimpleNamespace(type="text", text=json.dumps(self.reply["data"]))])


def _client(reply, record):
    return types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(stream=_FakeStream(reply, record))))


def test_ai_judge_overrides_cached_and_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(lj, "cache_dir", lambda name: tmp_path)
    units = hl.sentences([{"start": 0, "end": 4, "text": DULL}, {"start": 4, "end": 8, "text": "平淡但其實是梗"}])
    record: list = []
    reply = {"data": {"items": [{"i": 0, "score": 1, "why": ""}, {"i": 1, "score": 9, "why": "冷笑話"}]}}
    notes: list = []
    over, info = lj.judge_sentences(units, notes=notes, client=_client(reply, record))
    assert over == {0: (0.1, ""), 1: (0.9, "冷笑話")} and info["engine"] == "claude"
    sent = record[0]
    assert sent["model"] == lj.MODEL and "平淡但其實是梗" in sent["messages"][0]["content"]
    assert sent["output_config"]["format"]["type"] == "json_schema" and "沒有上傳影片" in notes[-1]
    scored = hl.score_sentences(units, None, over)
    assert scored[1].score > scored[0].score and scored[1].reasons() == "AI：冷笑話"
    # second call: from the cache, no request
    over2, info2 = lj.judge_sentences(units, notes=notes, client=_client({"data": {}}, record))
    assert over2 == over and info2["cached"] and len(record) == 1
    # refusal / errors -> local rules with a note
    other = hl.sentences([{"start": 0, "end": 1, "text": "別的影片"}])
    over3, info3 = lj.judge_sentences(other, notes=notes, client=_client({"stop": "refusal", "data": {}}, record))
    assert over3 is None and info3["engine"] == "local" and "改用本機規則" in notes[-1]
    # not installed / no key -> note, never an exception
    monkeypatch.setattr(lj, "available", lambda: (False, lj.KEY_HINT))
    over4, _ = lj.judge_sentences(other, notes=notes)
    assert over4 is None and lj.KEY_HINT in notes[-1]
    with pytest.raises(lj.Cancelled):
        ev = threading.Event()
        ev.set()
        lj.judge_sentences(other, notes=notes, client=_client(reply, record), cancel=ev)


# ------------------------------------------------------------------ REVIEW-3 (Codex) regressions
def _w(toks, t0=0.0, d=0.3):
    return [{"start": t0 + i * d, "end": t0 + i * d + d * 0.9, "word": x} for i, x in enumerate(toks)]


def test_repeated_context_word_inside_a_phrase_is_not_a_filler():
    # 「對對方很好」 tokenised per character: the two 「對」 are words, not a filler
    u = hl.sentences([{"start": 0, "end": 2, "text": "對對方很好", "words": _w(["對", "對", "方", "很好"])}])
    assert hl.filler_spans(u) == []
    # 「對對對。」 standing alone (pause after) still is
    u2 = hl.sentences([{"start": 0, "end": 3, "text": "對對對。我覺得", "words": _w(["對", "對", "對"]) +
                        [{"start": 1.9, "end": 2.4, "word": "我覺得"}]}])
    assert hl.filler_spans(u2) and hl.filler_spans(u2)[0][2] == "對對對"


def test_doubled_words_are_emphasis_not_stutters():
    u = hl.sentences([{"start": 0, "end": 2, "text": "非常非常好", "words": _w(["非常", "非常", "好"])}])
    assert hl.repeated_word_spans(u) == []
    u3 = hl.sentences([{"start": 0, "end": 2, "text": "我覺得我覺得我覺得好", "words": _w(["我覺得"] * 3 + ["好"])}])
    assert hl.repeated_word_spans(u3) == [(0.0, 0.6, "我覺得")]


def test_ai_judge_client_init_failure_and_secret_free_notes(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(lj, "cache_dir", lambda name: tmp_path)
    monkeypatch.setattr(lj, "available", lambda: (True, ""))

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("init failed")

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Boom))
    notes: list = []
    over, info = lj.judge_sentences(hl.sentences([{"start": 0, "end": 1, "text": "初始化失敗"}]), notes=notes)
    assert over is None and info["engine"] == "local" and "改用本機規則" in notes[-1]

    def leak(**kw):
        raise RuntimeError("bad key sk-ant-api03-SECRET123 for Bearer abcdefSECRET")
    client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(stream=leak)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "abcdefSECRET")
    notes = []
    lj.judge_sentences(hl.sentences([{"start": 0, "end": 1, "text": "洩漏測試"}]), notes=notes, client=client)
    assert "SECRET" not in " ".join(notes) and "改用本機規則" in notes[-1]
    # any key format, with no environment variable to compare against (Codex re-check): the text is never kept
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def leak2(**kw):
        raise RuntimeError("api_key=customToken123")
    client2 = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(stream=leak2)))
    notes = []
    lj.judge_sentences(hl.sentences([{"start": 0, "end": 1, "text": "洩漏測試二"}]), notes=notes, client=client2)
    assert "customToken123" not in " ".join(notes) and "RuntimeError" in notes[-1]

    class Http(Exception):
        status_code = 401
    assert "金鑰無效" in lj._safe(Http("x")) and "429" in lj._safe(type("R", (Exception,), {"status_code": 429})())


# ------------------------------------------------------------------ calibration on the first real video (2026-10-14)
def test_unpunctuated_subtitle_fragments_are_merged_into_clauses():
    cues = [(32.70, 33.73, "他遇到很多不"), (33.73, 34.73, "公不義的事情"), (34.80, 36.90, "我們可以在這邊來為他發聲"),
            (37.40, 38.40, "所以真的我自己個人吶")]
    segs = hl.merge_fragments([{"start": a, "end": b, "text": t} for a, b, t in cues])
    assert segs[0]["text"].startswith("他遇到很多不公不義的事情")              # the split phrase is whole again
    assert len(segs) == 2 and segs[1]["start"] == 37.40                       # a 0.5 s pause ends the clause
    assert "stakes" in hl.lexical_signals(segs[0]["text"])[1]
    # punctuated, sentence-length segments pass through unchanged
    punct = [{"start": 0, "end": 3, "text": "大家好。"}, {"start": 3.05, "end": 6, "text": "今天聊加盟。"}]
    assert [s["text"] for s in hl.merge_fragments(punct)] == ["大家好。", "今天聊加盟。"]
    # a clause never grows past MERGE_MAX_SECONDS, word timings travel along
    many = [{"start": i * 1.0, "end": i * 1.0 + 0.98, "text": f"片段{i}", "words": [{"start": i, "end": i + 0.9, "word": f"片段{i}"}]}
            for i in range(30)]
    merged = hl.merge_fragments(many)
    assert all(s["end"] - s["start"] <= hl.MERGE_MAX_SECONDS + 1e-9 for s in merged)
    assert sum(len(s["words"]) for s in merged) == 30


def test_everyday_connectives_and_articles_are_not_highlights():
    # 「其實」「但是」 open half the sentences of spoken Mandarin: a weak nudge, not a twist
    raw_conn, tags_conn = hl.lexical_signals("其實主要是因為這一年多")
    raw_twist, tags_twist = hl.lexical_signals("沒想到我竟然被騙了")
    assert tags_conn == ["contrast"] and raw_conn < 0.5 and "twist" in tags_twist and raw_twist > 2.0
    # 一個／一種 are articles; real numbers still count
    assert hl.lexical_signals("弄拙成巧的一個一個狀況")[0] == 0.0
    for t in ("大概有80%以上的案件", "三個步驟", "一萬塊", "兩年"):
        assert "number" in hl.lexical_signals(t)[1], t
    # listener back-channels are fillers only when they stand alone
    u = hl.sentences([{"start": 0, "end": 2, "text": "沒錯沒錯沒錯", "words": _w(["沒錯", "沒錯", "沒錯"])}])
    assert hl.filler_spans(u) and hl.score_sentences(u)[0].score == 0.0
    u2 = hl.sentences([{"start": 0, "end": 2, "text": "是我提出的", "words": _w(["是", "我", "提出的"])}])
    assert hl.filler_spans(u2) == []


def test_explanatory_why_is_not_a_question_and_only_true_restatements_are_penalised():
    assert hl.lexical_signals("這也是我為什麼今天找他合作")[1] == []
    assert hl.lexical_signals("為什麼這種議題會無法解決")[1] == ["question"]
    u = hl.score_sentences(hl.sentences([
        {"start": 0, "end": 3, "text": "我剛剛已經強調希望大家可以珍惜。"},
        {"start": 3, "end": 6, "text": "我們也一再強調我們不是你的打手。"}]))
    assert "recap" in u[0].tags and "recap" not in u[1].tags


def test_rank_ignores_float_noise_from_prefix_sums():
    # the mean of a constant signal from prefix sums differs in the last bits; it must not become a rank difference
    x = np.full(1000, 0.02)
    P = sc._prefix(x)
    means = np.asarray([(P[i + 50] - P[i]) / 50 for i in range(0, 900, 7)])
    assert np.ptp(means) > 0                                    # the noise really is there ...
    assert np.allclose(sc._rank(means), 0.5)                    # ... and ranking ignores it


def test_merge_keeps_back_channels_and_speakers_apart_and_respects_sentence_ends():
    # REVIEW-4 (Codex): a question plus the other person's 「沒錯沒錯」 must not become one unit
    segs = hl.merge_fragments([{"start": 0, "end": 1.4, "text": "你有沒有被騙"}, {"start": 1.4, "end": 2.8, "text": "沒錯沒錯"},
                               {"start": 2.8, "end": 4.0, "text": "我被騙了很多次"}])
    assert [s["text"] for s in segs] == ["你有沒有被騙", "沒錯沒錯", "我被騙了很多次"]
    spk = hl.merge_fragments([{"start": 0, "end": 1, "text": "我覺得", "speaker": "A"},
                              {"start": 1, "end": 2, "text": "其實不是", "speaker": "B"}])
    assert len(spk) == 2 and spk[1]["speaker"] == "B"
    # a sentence end (even with trailing space / closing quote) is never merged across
    for first in ("第一句。 ", "他說「好。」 "):
        assert len(hl.merge_fragments([{"start": 0, "end": 1.4, "text": first}, {"start": 1.4, "end": 2.8, "text": "第二句"}])) == 2


def test_number_rule_edge_cases():
    assert "number" in hl.lexical_signals("一百零一元")[1]
    assert "number" not in hl.lexical_signals("萬一個案出錯")[1]
    assert "number" not in hl.lexical_signals("這一年多")[1]
