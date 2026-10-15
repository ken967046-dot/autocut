"""剪輯清單 (cut list): parsing / validation, and the shorts planner cutting exactly the listed ranges -- in the listed
order (cold open), with title cards (字卡) -- through the normal plan -> audit -> apply path."""
from __future__ import annotations

import json

import pytest

from autocut.autoedit import cutlist as cl
from autocut.autoedit.api import AutoEditError, apply_plan, audit_plan
from autocut.autoedit.config import ShortsParams
from autocut.autoedit.shorts import plan_shorts
from autocut.core.editor import Editor
from autocut.core.project import Project, Sequence, validate_sequence
from autocut.util import fingerprint_file

from test_shorts import fake_ev, fake_proj, patched_analysis, source_file  # noqa: F401  (fixtures)


# ================================================================================================ parsing
@pytest.mark.parametrize("value, seconds", [
    (333.3, 333.3), (12, 12.0), ("333.3", 333.3), ("5:33.3", 333.3), ("0:05:33.3", 333.3), ("00:05:33,330", 333.33),
    ("1:02:03", 3723.0), (" 2:05 ", 125.0), ("12,5", 12.5),
])
def test_parse_time_accepts_seconds_clock_and_srt_forms(value, seconds):
    assert cl.parse_time(value) == pytest.approx(seconds)


@pytest.mark.parametrize("value", ["abc", -1, "-3", "5:70", "1:61:00", True, float("nan"), float("inf"), None, ""])
def test_parse_time_rejects_nonsense(value):
    with pytest.raises((ValueError, TypeError)):
        cl.parse_time(value)


def good(**over) -> dict:
    d = {"schema": cl.SCHEMA, "source": "訪談.mp4", "shorts": [
        {"title": "第一支", "ranges": [["1:00", "1:04"], [40, "0:52"]], "cards": [{"text": "開場字卡", "duration": 2.5}],
         "note": "冷開場"},
        {"title": "第二支", "ranges": [["00:01:10,000", "00:01:25,000"]], "cards": ["只有文字的字卡"]},
    ]}
    d.update(over)
    return d


def test_parse_cutlist_reads_ranges_in_order_cards_and_defaults():
    c = cl.parse_cutlist(good(), "ep1.json")
    assert c.name == "ep1.json" and c.source == "訪談.mp4" and len(c.shorts) == 2
    a, b = c.shorts
    assert a.ranges == [(60.0, 64.0), (40.0, 52.0)]                          # the listed order is kept (cold open)
    assert [(k.text, k.at, k.duration, k.source) for k in a.cards] == [("開場字卡", 0.0, 2.5, None)]
    assert b.ranges == [(70.0, 85.0)] and [(k.text, k.at, k.duration) for k in b.cards] == [("只有文字的字卡", 0.0, 3.0)]
    assert a.note == "冷開場" and b.note == ""
    # the fingerprint follows the content, not the spelling of the times or the source hint
    same = good(source="另一個名字.mp4")
    same["shorts"][0]["ranges"] = [[60, 64], ["0:40", 52.0]]
    assert cl.parse_cutlist(same).sha256 == c.sha256
    moved = good()
    moved["shorts"][0]["ranges"][0] = ["1:00", "1:05"]
    assert cl.parse_cutlist(moved).sha256 != c.sha256
    pinned = good()
    pinned["shorts"][0]["cards"] = [{"text": "開場字卡", "duration": 2.5, "source": "1:02"}]
    pc = cl.parse_cutlist(pinned)
    assert pc.shorts[0].cards[0].source == 62.0 and pc.sha256 != c.sha256


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.update(schema="autocut.cutlist/9"), "不支援的格式"),
    (lambda d: d.update(shorts=[]), "至少要列一支"),
    (lambda d: d.update(shorts="x"), "至少要列一支"),
    (lambda d: d["shorts"].append("x"), "第 3 支"),
    (lambda d: d["shorts"][1].update(ranges=[]), "第 2 支"),
    (lambda d: d["shorts"][1].update(ranges=[[1, 2, 3]]), "第 2 支第 1 段"),
    (lambda d: d["shorts"][1].update(ranges=[["一分鐘", 2]]), "看不懂的時間"),
    (lambda d: d["shorts"][0].update(ranges=[[5, 5]]), "結束時間必須晚於開始時間"),
    (lambda d: d["shorts"][0].update(ranges=[[30, 30.1], [20, 22]]), "第 1 支第 1 段）：只有 0.10 秒，太短"),
    (lambda d: d["shorts"][0].update(cards=[{"text": ""}]), "每張字卡都要有 text"),
    (lambda d: d["shorts"][0].update(cards=[{"text": "x", "duration": 0}]), "duration 至少要 0.5 秒"),
    (lambda d: d["shorts"][0].update(cards=[{"text": "閃一下", "duration": 0.05}]), "duration 至少要 0.5 秒"),
    (lambda d: d["shorts"][0].update(ranges=[[10 ** 400, 10 ** 400 + 5]]), "看不懂的時間"),
    (lambda d: d["shorts"][0].update(cards=[{"text": "x", "duration": 10 ** 400}]), "必須是秒數或時間"),
    (lambda d: d["shorts"][0].update(cards=[{"text": "x", "at": "很久"}]), "必須是秒數或時間"),
    (lambda d: d["shorts"][0].update(cards=[{"text": "x", "source": "-1"}]), "必須是秒數或時間"),
    (lambda d: d["shorts"][0].update(cards=[{"text": "x", "at": 1, "source": 2}]), "只能擇一"),
    (lambda d: d["shorts"][0].update(cards="x"), "cards 必須是清單"),
    (lambda d: d.update(shorts=[{"ranges": [[0, 1]]}] * (cl.MAX_SHORTS + 1)), "最多"),
])
def test_parse_cutlist_problems_name_the_short_and_field_in_chinese(mutate, needle):
    d = good()
    mutate(d)
    with pytest.raises(AutoEditError) as exc:
        cl.parse_cutlist(d)
    assert needle in str(exc.value) and "剪輯清單有問題" in str(exc.value)


def test_load_cutlist_file_errors_and_utf8_bom(tmp_path):
    with pytest.raises(AutoEditError, match="讀不到剪輯清單"):
        cl.load_cutlist(tmp_path / "沒有這個.json")
    bad = tmp_path / "壞掉.json"
    bad.write_text('{"shorts": [\n  {"ranges": [[1, 2]]},\n', encoding="utf-8")
    with pytest.raises(AutoEditError, match="不是有效的 JSON.*第 3 行"):
        cl.load_cutlist(bad)
    u16 = tmp_path / "壞掉的編碼.json"
    u16.write_bytes(b"\xff\xfe\x00")                                        # a UTF-16 BOM, then half a character
    with pytest.raises(AutoEditError, match="文字編碼無法辨識"):
        cl.load_cutlist(u16)
    ok = tmp_path / "第一集 剪輯清單.json"
    ok.write_bytes(b"\xef\xbb\xbf" + cl.dump_cutlist([{"title": "中文標題", "ranges": [[1, 2]]}], "a.mp4").encode("utf-8"))
    c = cl.load_cutlist(ok)
    assert c.name == "第一集 剪輯清單.json" and c.shorts[0].title == "中文標題" and c.source == "a.mp4"


def test_check_against_rejects_ranges_outside_the_video():
    c = cl.parse_cutlist({"shorts": [{"ranges": [[10, 20]]}, {"ranges": [[50, 60.2]]}]})
    cl.check_against(c, 60.0)                                                    # 0.2 s past the end is tolerated
    with pytest.raises(AutoEditError, match="第 2 支第 1 段.*超過影片長度"):
        cl.check_against(c, 59.0)
    with pytest.raises(AutoEditError, match="開始時間 .*已超過影片長度"):
        cl.check_against(c, 15.0 + 35.0 - 0.5)


# ================================================================================================ planner
def write_list(tmp_path, data: dict, name: str = "剪輯清單.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(p)


def plan_from(tmp_path, data, patched, source, **kw):
    patched.ev = fake_ev(120.0, speech=True, words=True, scenes=[30.0, 90.0])
    proj = fake_proj(source)
    proj.assets["a1"].fingerprint = fingerprint_file(source)                  # the audit re-checks the file
    params = ShortsParams(**{"cutlist_file": write_list(tmp_path, data), "count": 1, **kw})
    return proj, params, plan_shorts(proj, "a1", params)


EPISODE = {"schema": cl.SCHEMA, "shorts": [
    {"title": "冷開場", "ranges": [["1:00", "1:04"], [40, 52]], "note": "先放結論",
     "cards": [{"text": "開場字卡", "duration": 2.5}, {"text": "太晚的字卡", "at": 600}]},
    {"title": "一整段", "ranges": [["00:01:10,000", "00:01:25,000"]]},
    {"title": "", "ranges": [[100, 104], [5, 9]]},
]}


def test_cutlist_plan_cuts_the_listed_ranges_in_order_with_cards(tmp_path, patched_analysis, source_file):
    proj, params, plan = plan_from(tmp_path, EPISODE, patched_analysis, source_file)
    assert plan.kind == "shorts" and len(plan.sequences) == 3                  # 「支數」 does not apply
    seqs = [Sequence.from_dict(s) for s in plan.sequences]
    assert [s.name for s in seqs][:2] == ["短影音 1｜冷開場", "短影音 2｜一整段"]
    assert seqs[2].name.startswith("短影音 3｜00:05–01:44｜")                  # untitled: the whole source span
    for seq, item in zip(seqs, EPISODE["shorts"]):
        v1 = seq.video_tracks[0].clips
        assert v1 and not [i for i in validate_sequence(proj, seq) if i.level == "error"]
        pos = 0.0
        for c in v1:                                                             # contiguous pieces, in playing order
            assert c.start == pytest.approx(pos, abs=1e-6)
            pos += c.duration
        rngs = [tuple(cl.parse_time(t) for t in r) for r in item["ranges"]]
        # every piece comes from one listed range, and the ranges are played in the listed order
        owner = [next(k for k, (a, b) in enumerate(rngs) if a - 1e-6 <= c.src_in and c.src_out <= b + 1e-6) for c in v1]
        assert owner == sorted(owner) and set(owner) == set(range(len(rngs))), owner
    cold = seqs[0].video_tracks[0].clips
    assert cold[0].src_in == pytest.approx(60.0) and any(c.src_in == pytest.approx(40.0) for c in cold)
    assert "冷開場" in cold[0].note and "先放結論" in cold[0].note
    assert any("剪輯清單的下一段" in c.note for c in cold[1:])
    # the title card is its own layer: on top of the subtitles, not counted as a subtitle
    cards = [q for q in seqs[0].cues if q.is_card]
    assert [(q.text, q.start, q.end) for q in cards] == [("開場字卡", 0.0, 2.5)]
    assert any(not q.is_card and q.start < 2.5 for q in seqs[0].cues)         # overlaps a subtitle on purpose
    assert not [q for s in seqs[1:] for q in s.cues if q.is_card]
    assert any("太晚的字卡" in n and "已略過" in n for n in plan.notes)
    assert any("依剪輯清單「剪輯清單.json」剪出 3 支" in n for n in plan.notes)
    # decisions + brief record the list
    shorts = [d for d in plan.decisions if d["kind"] == "short"]
    assert [d["summary"].split(" · ")[0] for d in shorts] == ["冷開場", "一整段", "剪輯清單"]
    assert "先放結論" in shorts[0]["why"] and "第 1 段取自後面的內容當開場" in shorts[0]["why"]
    assert "字卡：開場字卡；" in shorts[0]["why"] and "太晚的字卡" not in shorts[0]["why"]   # only what is shown
    tight = [d for d in plan.decisions if d["kind"] == "tighten"]
    assert len(tight) == 3 and tight[0]["evidence"]["ranges"] == [[60.0, 64.0], [40.0, 52.0]]
    assert plan.brief["cutlist"] == {"name": "剪輯清單.json", "sha256": cl.load_cutlist(params.cutlist_file).sha256,
                                     "shorts": 3, "source": ""}
    assert plan.sha256() == plan_shorts(proj, "a1", params).sha256()            # reproducible


def test_cutlist_plan_passes_audit_and_applies_cards_as_one_undo_step(tmp_path, patched_analysis, source_file):
    proj, params, plan = plan_from(tmp_path, EPISODE, patched_analysis, source_file)
    receipt = audit_plan(plan, proj)
    assert receipt.accepted, [f.to_dict() for f in receipt.findings if f.severity == "error"]
    assert not [f for f in receipt.findings if f.rule == "caption.overlap"]   # card over a subtitle is fine
    # ...but two overlapping cards (or two overlapping subtitles) are still an error
    sd = plan.sequences[0]
    card = next(q for q in sd["cues"] if q.get("style", {}).get("kind") == "card")
    sd["cues"].append(dict(card, id="q_dup", start=card["start"] + 0.5, text="另一張"))
    bad = audit_plan(plan, proj)
    assert any(f.rule == "caption.overlap" and "字卡「另一張」" in f.message for f in bad.findings)
    sd["cues"].pop()
    ed = Editor(proj)
    n0 = len(proj.sequences)
    ids = apply_plan(ed, plan, audit_plan(plan, proj))
    assert len(proj.sequences) == n0 + 3
    seq = proj.sequence(ids[0])
    assert [q.text for q in seq.cues if q.is_card] == ["開場字卡"]
    proj.save(tmp_path / "存檔.acproj")
    again = Project.load(tmp_path / "存檔.acproj")                             # survives save / load
    assert [q.text for q in again.sequence(ids[0]).cues if q.is_card] == ["開場字卡"]
    ed.undo()
    assert len(proj.sequences) == n0


def test_source_pinned_card_follows_the_cut(tmp_path, patched_analysis, source_file):
    # speech 0-4, 5-9, ... (1 s pauses, tightened to 0.35 s); the list plays 45-55 first, then 20-30
    data = {"shorts": [{"title": "釘在來源", "ranges": [[45, 55], [20, 30]],
                        "cards": [{"text": "第二段的字卡", "source": 26.0, "duration": 2},
                                  {"text": "停頓裡的字卡", "source": 24.5},
                                  {"text": "不在片段裡", "source": 70}]}]}
    _proj, _params, plan = plan_from(tmp_path, data, patched_analysis, source_file)
    seq = Sequence.from_dict(plan.sequences[0])
    clips = seq.video_tracks[0].clips
    hit = next(c for c in clips if c.src_in - 1e-6 <= 26.0 < c.src_out)
    want = hit.start + (26.0 - hit.src_in)
    cards = [q for q in seq.cues if q.is_card]
    assert [q.text for q in cards] == ["第二段的字卡"]
    assert cards[0].start == pytest.approx(want, abs=1 / 30 + 1e-6) and cards[0].end - cards[0].start == pytest.approx(2.0, abs=0.04)
    assert cards[0].start < 26.0 - 20.0 + 10.0                                # moved up by the trims, not the raw offset
    assert any("停頓裡的字卡" in n and "沒有包含這個時間" in n for n in plan.notes)   # 24.5 s sits in a trimmed pause
    assert any("不在片段裡" in n and "01:10" in n for n in plan.notes)


def test_cutlist_errors_out_of_range_and_too_short(tmp_path, patched_analysis, source_file):
    with pytest.raises(AutoEditError, match="第 1 支第 2 段.*超過影片長度"):
        plan_from(tmp_path, {"shorts": [{"ranges": [[10, 20], [100, 130]]}]}, patched_analysis, source_file)
    with pytest.raises(AutoEditError, match="第 1 支第 1 段）：只有 0.10 秒，太短"):
        plan_from(tmp_path, {"shorts": [{"title": "太短", "ranges": [[10, 10.1]]}]}, patched_analysis, source_file)
    # a range that is long enough as written but runs past the end leaves too little once clamped: an error, never a
    # silently missing piece (Codex REVIEW-5)
    with pytest.raises(AutoEditError, match="第 1 支第 2 段落在影片結尾，只剩 0.10 秒"):
        plan_from(tmp_path, {"shorts": [{"ranges": [[20, 25], [119.9, 120.2]]}]}, patched_analysis, source_file)
    with pytest.raises(AutoEditError, match="不是有效的 JSON"):
        bad = tmp_path / "bad.json"
        bad.write_text("{", encoding="utf-8")
        patched_analysis.ev = fake_ev(120.0)
        plan_shorts(fake_proj(source_file), "a1", ShortsParams(cutlist_file=str(bad)))


def test_listed_endpoints_are_kept_and_a_near_scene_cut_is_reported(tmp_path, patched_analysis, source_file):
    """The person's endpoints are the decision: a start 0.2 s before a scene cut is NOT moved onto the cut (that would
    drop 0.2 s, maybe speech); the plan says the start will flash the previous shot and which time avoids it."""
    patched_analysis.ev = fake_ev(120.0, speech=True, words=True, scenes=[10.2, 21.9])
    proj = fake_proj(source_file)
    proj.assets["a1"].fingerprint = fingerprint_file(source_file)
    lst = write_list(tmp_path, {"shorts": [{"ranges": [[10, 12], [20, 22]]}]})
    plan = plan_shorts(proj, "a1", ShortsParams(cutlist_file=lst, tighten=False))
    clips = Sequence.from_dict(plan.sequences[0]).video_tracks[0].clips
    assert clips[0].src_in == pytest.approx(10.0) and clips[-1].src_out == pytest.approx(22.0)
    assert sum(c.duration for c in clips) == pytest.approx(4.0, abs=1 / 30 + 1e-6)
    start = [n for n in plan.notes if "第 1 支第 1 段的開頭 0:10.00" in n]
    end = [n for n in plan.notes if "第 1 支第 2 段的結尾 0:22.00" in n]
    assert start and "早 0.20 秒" in start[0] and "改成 0:10.20" in start[0]
    assert end and "晚 0.10 秒" in end[0] and "改成 0:21.90" in end[0]


def test_cutlist_end_a_little_past_the_video_is_clamped(tmp_path, patched_analysis, source_file):
    _proj, _params, plan = plan_from(tmp_path, {"shorts": [{"ranges": [[110, 120.2]]}]}, patched_analysis, source_file)
    clips = Sequence.from_dict(plan.sequences[0]).video_tracks[0].clips
    assert max(c.src_out for c in clips) <= 120.0 + 1e-9


# ================================================================================================ CLI + form
HAVE_FFMPEG = __import__("shutil").which("ffmpeg") is not None


@pytest.mark.ffmpeg
@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_cli_shorts_cutlist(media_dir, tmp_path, capsys):
    from autocut import cli
    src = media_dir / "land.mp4"                                                 # 8 s
    lst = write_list(tmp_path, {"shorts": [{"title": "倒著放", "ranges": [[5, 7.5], [0.5, 3]],
                                            "cards": [{"text": "字卡測試", "duration": 2}]}]})
    out = tmp_path / "輸出"
    rc = cli.main(["shorts", str(src), "-o", str(out), "--cutlist", lst, "--captions", "off"])
    o = capsys.readouterr()
    assert rc == 0, o.out + o.err
    proj = Project.load(out / "land.acproj")
    seq = proj.sequences[-1]
    assert seq.name == "短影音 1｜倒著放"
    clips = seq.video_tracks[0].clips
    assert clips[0].src_in == pytest.approx(5.0, abs=0.05) and clips[-1].src_out == pytest.approx(3.0, abs=0.05)
    assert [(q.text, q.is_card) for q in seq.cues] == [("字卡測試", True)]
    rc = cli.main(["shorts", str(src), "-o", str(out), "--cutlist", str(tmp_path / "沒有.json")])
    assert rc == 1 and "找不到剪輯清單" in capsys.readouterr().err
    far = write_list(tmp_path, {"shorts": [{"ranges": [[30, 40]]}]}, "far.json")
    rc = cli.main(["shorts", str(src), "-o", str(out), "--cutlist", far, "--captions", "off"])
    assert rc == 1 and "超過影片長度" in capsys.readouterr().err


# ================================================================================================ REVIEW-5 (panel)
def _cards(plan, k=0):
    return [(q.text, round(q.start, 3), round(q.end, 3)) for q in Sequence.from_dict(plan.sequences[k]).cues if q.is_card]


def test_cards_never_overlap_and_a_shortened_or_skipped_card_is_explained(tmp_path, patched_analysis, source_file):
    """Two cards 1.5 s apart (3 s each) used to give a plan its own audit rejected (the CLI then lost every short);
    a card pinned to a source moment can land on an 'at' card after pause trimming, which nobody can foresee."""
    data = {"shorts": [{"title": "兩張", "ranges": [[40, 52]],
                        "cards": [{"text": "標題", "at": 0}, {"text": "補充", "at": 1.5},
                                  {"text": "撞在一起", "at": 1.5}]}]}
    proj, _params, plan = plan_from(tmp_path, data, patched_analysis, source_file)
    cards = _cards(plan)
    assert [c[0] for c in cards] == ["標題", "撞在一起"]                         # same start: the later listed one wins
    assert cards[0][1:] == (0.0, 1.5) and cards[1][1] == 1.5                     # 標題 ends where the next card starts
    assert any("標題" in n and "只顯示 1.5 秒（清單設定 3 秒）：下一張字卡接著出現" in n for n in plan.notes)
    assert any("補充" in n and "已略過" in n for n in plan.notes)
    receipt = audit_plan(plan, proj)
    assert receipt.accepted and not [f for f in receipt.findings if f.rule in ("caption.overlap", "card.reading")]
    shorts = [d for d in plan.decisions if d["kind"] == "short"]
    assert "字卡：標題／撞在一起；" in shorts[0]["why"]
    # a source-pinned card that trimming moved onto an 'at' card: still no overlap
    pinned = {"shorts": [{"ranges": [[45, 55], [20, 30]], "cards": [{"text": "A", "at": 9}, {"text": "B", "source": 21}]}]}
    proj2, _p, plan2 = plan_from(tmp_path, pinned, patched_analysis, source_file)
    c2 = _cards(plan2)
    assert len(c2) == 2 and c2[0][2] <= c2[1][1] + 1e-9
    assert audit_plan(plan2, proj2).accepted


def test_a_card_cut_short_by_the_end_of_the_short_is_skipped_or_noted(tmp_path, patched_analysis, source_file):
    # 12 s of source tightened to ~10.7 s: a 4 s card at 10.1 would show 0.57 s (31 chars/s) -> skipped, with the reason
    data = {"shorts": [{"title": "結尾", "ranges": [[40, 52]],
                        "cards": [{"text": "一二三四五六七八九十一二三四五六七八", "at": 10.1, "duration": 4},
                                  {"text": "後面還有", "at": 8.0, "duration": 4}]}]}
    _proj, _params, plan = plan_from(tmp_path, data, patched_analysis, source_file)
    seq = Sequence.from_dict(plan.sequences[0])
    cards = _cards(plan)
    assert [c[0] for c in cards] == ["後面還有"] and cards[0][2] == pytest.approx(seq.duration, abs=1e-3)
    assert any("一二三四" in n and "已略過" in n and "這支短影音在這裡結束" in n for n in plan.notes)
    assert any("後面還有" in n and "清單設定 4 秒" in n and "這支短影音在這裡結束" in n for n in plan.notes)


def test_decisions_describe_what_was_cut(tmp_path, patched_analysis, source_file):
    """Cold open = the first range comes from later than ANY range after it (not only the second); a cut-list short
    has no made-up score (the review card printed 分數 0.00)."""
    from autocut.gui.autoedit_review import build_candidates
    data = {"shorts": [{"title": "三段", "ranges": [[100, 104], [110, 114], [5, 9]]},
                       {"title": "照順序", "ranges": [[5, 9], [100, 104]]}]}
    proj, _params, plan = plan_from(tmp_path, data, patched_analysis, source_file)
    shorts = [d for d in plan.decisions if d["kind"] == "short"]
    assert "冷開場" in shorts[0]["why"] and "冷開場" not in shorts[1]["why"]
    assert shorts[0]["summary"].endswith("· 3 段") and all(d["score"] is None for d in shorts)
    assert all(Sequence.from_dict(s).meta["shorts"]["score"] is None for s in plan.sequences)
    cands = build_candidates(plan, proj.assets)
    assert cands and all("分數" not in c.headline(i) for i, c in enumerate(cands, 1))


def test_voice_and_face_statistics_cover_only_the_played_pieces(tmp_path, patched_analysis, source_file):
    """Neither played range has a voice (or a face), but the skipped middle does: the no-voice music boost must apply
    and the face ratio must be 0 (they were computed over min..max of the source)."""
    import subprocess
    from test_shorts import HAVE_FFMPEG
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg needed to make a music file")
    music = tmp_path / "bgm.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=220:sample_rate=44100:duration=10", str(music)], check=True)
    ev = fake_ev(120.0, speech=True, faces=(40.0, 80.0))
    for lo, hi in ((0.0, 12.0), (100.0, 112.0)):                               # silence the two played stretches
        for k in range(int(lo * 20), int(hi * 20)):
            ev.voice[k] = 0.0
    patched_analysis.ev = ev
    proj = fake_proj(source_file)
    lst = write_list(tmp_path, {"shorts": [{"ranges": [[100, 108], [2, 9]]}]})
    plan = plan_shorts(proj, "a1", ShortsParams(cutlist_file=lst, tighten=False, music_path=str(music),
                                                music_gain_db=-20.0, captions="off"))
    music_d = next(d for d in plan.decisions if d["kind"] == "music")
    assert music_d["evidence"]["voice_ratio"] == 0.0 and music_d["evidence"]["gain_db"] > -20.0   # boosted
    reframe_d = next(d for d in plan.decisions if d["kind"] == "reframe")
    assert reframe_d["evidence"]["face_ratio"] == 0.0


def test_scene_refinement_is_limited_to_the_played_pieces(tmp_path, patched_analysis, source_file, monkeypatch):
    """A cold open spanning most of an hour refined every scene cut in between (one ffmpeg decode each, ~450 calls);
    only the cuts inside the played pieces matter."""
    calls: list[float] = []

    def fake_refine(path, est, fps, window=0.6):
        calls.append(est)
        return est
    monkeypatch.setattr("autocut.analysis.refine_cut", fake_refine)
    ev = fake_ev(3600.0, burst=4.0, gap=1.0, scenes=[6.0 * k for k in range(1, 600)])
    patched_analysis.ev = ev
    proj = fake_proj(source_file, dur=3600.0)
    lst = write_list(tmp_path, {"shorts": [{"ranges": [[3000.5, 3010.5], [300.5, 330.5]]}]})
    plan_shorts(proj, "a1", ShortsParams(cutlist_file=lst, reframe="face", captions="off"))
    assert 0 < len(set(calls)) <= 12, sorted(set(calls))
    assert all(2995 < t < 3016 or 295 < t < 336 for t in calls)


def test_the_ai_judge_is_not_called_for_a_cut_list(tmp_path, patched_analysis, source_file, monkeypatch):
    called: list = []
    monkeypatch.setattr("autocut.autoedit.llm_judge.judge_sentences", lambda *a, **k: called.append(a) or (None, {}))
    _proj, _params, plan = plan_from(tmp_path, {"shorts": [{"ranges": [[20, 30]]}]}, patched_analysis, source_file,
                                     highlight_engine="claude")
    assert called == [] and any("沒有傳送任何字幕" in n for n in plan.notes)
    assert plan.brief["highlight_engine"]["skipped"] == "cutlist"


def test_the_caption_decision_counts_subtitles_not_cards(tmp_path, patched_analysis, source_file):
    data = {"shorts": [{"ranges": [[20, 30]], "cards": [{"text": "一二三四五六七八九十一二三四五六七八九十一二三", "duration": 1}]},
                       {"ranges": [[20, 30]], "cards": [{"text": "只有字卡", "duration": 2}]}]}
    _proj, _params, plan = plan_from(tmp_path, data, patched_analysis, source_file, captions="auto")
    cap = [d for d in plan.decisions if d["kind"] == "caption"]
    seq = Sequence.from_dict(plan.sequences[0])
    n_subs = sum(1 for q in seq.cues if not q.is_card)
    assert cap[0]["summary"].startswith(f"{n_subs} 條字幕") and cap[0]["summary"].endswith("另有 1 張字卡")
    assert cap[0]["evidence"]["max_chars"] < 23 and cap[0]["evidence"]["cards"] == 1



def test_a_card_whose_duration_is_not_on_the_frame_grid_is_kept(tmp_path, patched_analysis, source_file):
    """At 24 fps a 0.6 s card is 14 frames (0.583 s): frame rounding must not count as "no room" (Codex REVIEW-5b)."""
    data = {"shorts": [{"ranges": [[40, 52]], "cards": [{"text": "短卡", "at": 0, "duration": 0.6}]}]}
    _proj, _params, plan = plan_from(tmp_path, data, patched_analysis, source_file, fps=24.0)
    assert _cards(plan) == [("短卡", 0.0, round(14 / 24, 3))]
    assert not any("短卡" in n for n in plan.notes)
