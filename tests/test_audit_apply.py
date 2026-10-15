"""Audit rules (one triggering + one non-triggering test per rule) and apply (one undo step, fail-closed receipts).

Plans are hand-built (no ffmpeg, no analysis): that is exactly what makes each rule testable on its own.
"""
from __future__ import annotations

import copy

import pytest

from autocut.analysis.evidence import Evidence
from autocut.autoedit import audit as audit_mod
from autocut.autoedit.api import apply_plan, audit_plan, preview_project
from autocut.autoedit.apply import plan_asset_map, remap_plan_assets
from autocut.autoedit.audit import audit, first_visual_cut
from autocut.autoedit.config import AuditRules
from autocut.core import plan as planmod
from autocut.core.editor import Editor
from autocut.core.plan import AuditReceipt, EditPlan, PlanError, find_run
from autocut.core.project import Asset, CaptionCue, Clip, Marker, Project, Sequence, Transform, validate_sequence
from autocut.util import fingerprint_file
from conftest import fake_asset

NOFILE = AuditRules(require_sources_exist=False)       # the fixture assets are fake paths


# ------------------------------------------------------------------------------------------- builders
def clip(sid: str, n: int, asset: str, start: float, a: float, b: float, **kw) -> Clip:
    c = Clip(id=f"{sid}_c{n}", asset_id=asset, start=start, src_in=a, src_out=b)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def mk_seq(sid: str = "ps1", v: list | None = None, a: list | None = None, cues: list | None = None, w: int = 1080,
           h: int = 1920, fps: float = 30.0, name: str = "短影音 1", markers: list | None = None) -> dict:
    """Plan-style sequence dict with deterministic ids.  ``v`` / ``a`` = list of Clip for V1 / A1."""
    s = Sequence.create(name, w, h, fps, seq_id=sid)
    for i, t in enumerate(s.tracks()):
        t.id = f"{sid}_t{i}"
    s.video_tracks[0].clips = list(v or [])
    s.audio_tracks[0].clips = list(a or [])
    s.cues = list(cues or [])
    s.markers = list(markers or [])
    return s.to_dict()


def zoomed(z: float = 1.12) -> Transform:
    return Transform(fit="cover", zoom=z)


def good_v(sid: str = "ps1") -> list[Clip]:
    """20 s of a1 on V1 with a visual cut (zoom change) at 1.5 s -> satisfies every shorts rule."""
    return [clip(sid, 1, "a1", 0.0, 0.0, 1.5), clip(sid, 2, "a1", 1.5, 1.5, 20.0, transform=zoomed())]


def good_cues(sid: str = "ps1", n: int = 12, step: float = 1.6, dur: float = 1.5, text: str = "這是一句字幕") -> list[CaptionCue]:
    return [CaptionCue(id=f"{sid}_q{i}", start=round(i * step, 3), end=round(i * step + dur, 3), text=text) for i in range(n)]


def good_short(sid: str = "ps1") -> dict:
    return mk_seq(sid, v=good_v(sid), cues=good_cues(sid))


def mk_plan(seqs: list[dict] | None = None, kind: str = "shorts", **kw) -> EditPlan:
    kw.setdefault("brief", {"platform": "ig_reels"})
    return EditPlan(kind=kind, sequences=[good_short()] if seqs is None else seqs, **kw)


def codes(r: AuditReceipt, rule: str | None = None) -> list:
    return [f for f in r.findings if rule is None or f.rule == rule]


def has(r: AuditReceipt, rule: str) -> bool:
    return bool(codes(r, rule))


def run_audit(plan: EditPlan, proj, rules: AuditRules = NOFILE, **kw) -> AuditReceipt:
    return audit(plan, proj, rules, **kw)


# ------------------------------------------------------------------------------------------- the clean plan
def test_clean_plan_has_only_info(proj):
    r = run_audit(mk_plan(), proj)
    assert r.accepted and not r.errors and not r.warnings, [f.to_dict() for f in r.findings]
    assert [f.rule for f in r.findings if f.severity == "info"].count("human.pending") == 1


def test_receipt_is_bound_and_audit_is_read_only(proj):
    plan = mk_plan()
    before_plan, before_proj = plan.sha256(), proj.content_json()
    r = audit_plan(plan, proj, NOFILE)
    assert r.plan_sha256 == plan.sha256() == before_plan
    assert r.project_revision == proj.revision and r.project_sha256 == proj.content_sha256()
    assert r.thresholds == NOFILE.to_dict()
    assert proj.content_json() == before_proj and plan.sha256() == before_plan   # nothing was modified


def test_human_pending_always_present_even_on_errors(proj):
    for plan in (mk_plan(), EditPlan(kind="shorts")):
        f = codes(run_audit(plan, proj), "human.pending")
        assert len(f) == 1 and f[0].severity == "info" and "人工審片" in f[0].message


def test_findings_are_deterministic(proj):
    plan = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 0.2)], cues=good_cues("ps1", 2))])
    a, b = run_audit(plan, proj), run_audit(plan, proj)
    assert [f.to_dict() for f in a.findings] == [f.to_dict() for f in b.findings]
    assert a.sha256() == b.sha256()


def test_crashing_rule_fails_closed(proj, monkeypatch):
    def boom(ctx):
        raise RuntimeError("kaputt")
    monkeypatch.setattr(audit_mod, "_RULES", [("clip.short", boom)])
    r = run_audit(mk_plan(), proj)
    assert not r.accepted and has(r, "audit.internal") and has(r, "human.pending")


# ------------------------------------------------------------------------------------------- plan.*
def test_plan_empty(proj):
    r = run_audit(EditPlan(kind="shorts"), proj)
    assert has(r, "plan.empty") and not r.accepted
    assert codes(r, "plan.empty")[0].severity == "error"
    assert not has(run_audit(mk_plan(), proj), "plan.empty")


def test_plan_hash_declared_mismatch(proj):
    class Forged(EditPlan):
        def to_dict(self):
            d = super().to_dict()
            d["plan_sha256"] = "0" * 64
            return d
    forged = Forged(kind="shorts", sequences=[good_short()], brief={"platform": "ig_reels"})
    r = run_audit(forged, proj)
    assert has(r, "plan.hash") and not r.accepted
    assert not has(run_audit(mk_plan(), proj), "plan.hash")


def test_plan_hash_unserialisable_content(proj):
    plan = mk_plan()
    plan.brief["bad"] = {1, 2, 3}                       # a set cannot be saved as JSON -> the hash could not be re-checked
    assert has(run_audit(plan, proj), "plan.hash")


def test_plan_hash_roundtrip_through_json_file(proj, tmp_path):
    plan = mk_plan()
    plan.save(tmp_path / "p.json")
    again = EditPlan.load(tmp_path / "p.json")
    assert not has(run_audit(again, proj), "plan.hash")
    assert again.sha256() == plan.sha256()


# ------------------------------------------------------------------------------------------- source.*
def test_source_missing(proj):
    plan = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "ghost", 0, 0, 5)])])
    r = run_audit(plan, proj)
    f = codes(r, "source.missing")
    assert len(f) == 1 and f[0].severity == "error" and "ghost" in f[0].message and not r.accepted
    assert not has(run_audit(mk_plan(), proj), "source.missing")


def test_source_missing_is_satisfied_by_plan_assets(proj):
    music = Asset(id="pm1", path="/nonexistent/music.mp3", name="music.mp3", kind="audio", duration=30.0, has_video=False,
                  has_audio=True, audio_channels=2, sample_rate=44100)
    plan = mk_plan([mk_seq("ps1", v=good_v(), a=[clip("ps1", 9, "pm1", 0, 0, 20)], cues=good_cues())], assets=[music.to_dict()])
    r = run_audit(plan, proj)
    assert not has(r, "source.missing") and not has(r, "seq.invalid"), [f.to_dict() for f in r.findings]


def test_source_file(proj, tmp_path):
    real = tmp_path / "real.mp4"
    real.write_bytes(b"x" * 5000)
    proj.assets["a1"].path = str(real)
    proj.assets["a1"].fingerprint = ""
    ok = run_audit(mk_plan(), proj, AuditRules(require_sources_exist=True))
    assert not has(ok, "source.file")
    real.unlink()
    r = run_audit(mk_plan(), proj, AuditRules(require_sources_exist=True))
    f = codes(r, "source.file")
    assert len(f) == 1 and f[0].severity == "error" and "real.mp4" in f[0].message
    assert not has(run_audit(mk_plan(), proj, NOFILE), "source.file")      # switched off


def test_source_changed(proj, tmp_path):
    f = tmp_path / "src.mp4"
    f.write_bytes(bytes(range(256)) * 40)
    a = proj.assets["a1"]
    a.path, a.fingerprint = str(f), fingerprint_file(f)
    src = {"asset_id": "a1", "name": "src.mp4", "fingerprint": a.fingerprint, "duration": 20.0, "evidence_sha256": "x"}
    plan = mk_plan(sources=[src])
    assert not has(run_audit(plan, proj, AuditRules()), "source.changed")
    f.write_bytes(bytes(range(255, -1, -1)) * 41)                         # the file was swapped after the analysis
    r = run_audit(plan, proj, AuditRules())
    assert has(r, "source.changed") and codes(r, "source.changed")[0].severity == "error" and not r.accepted


def test_source_changed_ignored_without_fingerprint(proj, tmp_path):
    f = tmp_path / "src.mp4"
    f.write_bytes(b"abc" * 100)
    proj.assets["a1"].path = str(f)
    plan = mk_plan(sources=[{"asset_id": "a1", "name": "src", "fingerprint": "", "duration": 20.0}])
    assert not has(run_audit(plan, proj, AuditRules()), "source.changed")


# ------------------------------------------------------------------------------------------- seq.invalid / clip.short
def test_seq_invalid_error(proj):
    plan = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 50)])])         # a1 is only 20 s long
    r = run_audit(plan, proj)
    f = codes(r, "seq.invalid")
    assert f and f[0].severity == "error" and "ps1" in f[0].where and not r.accepted
    assert not has(run_audit(mk_plan(), proj), "seq.invalid")


def test_seq_invalid_warn_level_is_mapped(proj, monkeypatch):
    from autocut.core.project import Issue
    monkeypatch.setattr(audit_mod, "validate_sequence", lambda p, s: [Issue("warn", "x", "只是警告", "V1/c")])
    r = run_audit(mk_plan(), proj)
    f = codes(r, "seq.invalid")
    assert len(f) == 1 and f[0].severity == "warn" and r.accepted


def test_clip_short(proj):
    plan = mk_plan([mk_seq("ps1", v=good_v() + [clip("ps1", 3, "a2", 20.0, 0.0, 0.2)], cues=good_cues())])
    f = codes(run_audit(plan, proj), "clip.short")
    assert len(f) == 1 and f[0].severity == "warn" and f[0].where == "ps1/ps1_c3"
    assert not has(run_audit(mk_plan(), proj), "clip.short")
    long_enough = mk_plan([mk_seq("ps1", v=good_v() + [clip("ps1", 3, "a2", 20.0, 0.0, 0.3)], cues=good_cues())])
    assert not has(run_audit(long_enough, proj), "clip.short")


# ------------------------------------------------------------------------------------------- shorts.*
def test_shorts_aspect(proj):
    land = mk_plan([mk_seq("ps1", v=good_v(), cues=good_cues(), w=1920, h=1080)])
    r = run_audit(land, proj)
    assert has(r, "shorts.aspect") and codes(r, "shorts.aspect")[0].severity == "error" and not r.accepted
    assert not has(run_audit(mk_plan(), proj), "shorts.aspect")
    assert not has(run_audit(mk_plan(land.sequences, kind="assemble"), proj), "shorts.aspect")


def test_shorts_band(proj):
    short = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 1.5), clip("ps1", 2, "a1", 1.5, 1.5, 8.0, transform=zoomed())])])
    r = run_audit(short, proj)
    assert has(r, "shorts.band") and codes(r, "shorts.band")[0].severity == "warn" and r.accepted
    strict = run_audit(short, proj, AuditRules(require_sources_exist=False, strict_platform_bands=True))
    assert codes(strict, "shorts.band")[0].severity == "error" and not strict.accepted
    assert not has(run_audit(mk_plan(), proj), "shorts.band")                       # 20 s is inside 13-60 s
    custom = mk_plan([short.sequences[0]], brief={"platform": "ig_reels", "min_seconds": 5.0})
    assert not has(run_audit(custom, proj), "shorts.band")                          # brief carries the user's band


def test_shorts_deadzone(proj):
    seq = mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 1.5), clip("ps1", 2, "a1", 1.5, 1.5, 20.0, transform=zoomed()),
                           clip("ps1", 3, "a2", 20.0, 0, 10.0, transform=zoomed(1.0))])       # 30 s
    yt = mk_plan([seq], brief={"platform": "yt_shorts"})
    f = codes(run_audit(yt, proj), "shorts.deadzone")
    assert len(f) == 1 and f[0].severity == "warn"
    assert codes(run_audit(yt, proj, AuditRules(require_sources_exist=False, strict_platform_bands=True)),
                 "shorts.deadzone")[0].severity == "error"
    assert not has(run_audit(mk_plan([seq], brief={"platform": "ig_reels"}), proj), "shorts.deadzone")
    assert not has(run_audit(mk_plan(brief={"platform": "yt_shorts"}), proj), "shorts.deadzone")      # 20 s: allowed


def test_shorts_first_cut_late_and_missing(proj):
    late = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 5.0), clip("ps1", 2, "a1", 5.0, 10.0, 20.0)],
                           cues=good_cues())])
    f = codes(run_audit(late, proj), "shorts.first_cut")
    assert len(f) == 1 and f[0].severity == "warn" and "5.00" in f[0].message
    none = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 20.0)], cues=good_cues())])
    assert "沒有任何畫面變化" in codes(run_audit(none, proj), "shorts.first_cut")[0].message


def test_shorts_first_cut_ok(proj):
    assert not has(run_audit(mk_plan(), proj), "shorts.first_cut")                         # zoom change at 1.5 s
    keyed = clip("ps1", 1, "a1", 0, 0, 20.0)
    keyed.transform = Transform(fit="cover", keys=[[0.0, 0.3, 0.5], [1.0, 0.3, 0.5], [1.0333, 0.7, 0.5]])   # reframing jump
    assert abs(first_visual_cut(Sequence.from_dict(mk_seq("ps1", v=[keyed]))) - 1.0333) < 1e-6
    assert not has(run_audit(mk_plan([mk_seq("ps1", v=[keyed], cues=good_cues())]), proj), "shorts.first_cut")


def test_shorts_first_cut_seamless_split_is_not_a_cut(proj):
    seamless = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 1.5), clip("ps1", 2, "a1", 1.5, 1.5, 20.0)],
                               cues=good_cues())])
    assert has(run_audit(seamless, proj), "shorts.first_cut")          # same take, same framing: nothing changes on screen
    assert not has(run_audit(mk_plan([seamless.sequences[0]], kind="assemble"), proj), "shorts.first_cut")   # shorts only


def _words_evidence() -> Evidence:
    return Evidence.empty("a1", 20.0, speech=[{"start": 0.0, "end": 4.0, "text": "你好世界", "words": [
        {"start": 0.0, "end": 1.0, "word": "你好"}, {"start": 1.0, "end": 2.0, "word": "世界"}]}])


def test_shorts_mid_word(proj):
    mid = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0.5, 0.5, 2.0)], cues=good_cues())])
    r = run_audit(mid, proj, evidence={"a1": _words_evidence()})
    f = codes(r, "shorts.mid_word")
    assert len(f) == 1 and f[0].severity == "warn" and "你好" in f[0].message
    clean = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 1.0, 1.0, 2.0)], cues=good_cues())])
    assert not has(run_audit(clean, proj, evidence={"a1": _words_evidence()}), "shorts.mid_word")


def test_shorts_mid_word_not_run_without_words_is_reported(proj):
    r = run_audit(mk_plan(), proj, evidence={})
    f = codes(r, "audit.evidence")
    assert any("shorts.mid_word" in x.message for x in f) and all(x.severity == "info" for x in f)
    assert not has(r, "shorts.mid_word")


# ------------------------------------------------------------------------------------------- caption.*
def one_cue_plan(text: str, start: float, end: float, kind: str = "shorts") -> EditPlan:
    cues = good_cues()[:6] + [CaptionCue(id="ps1_qx", start=start, end=end, text=text)]
    return mk_plan([mk_seq("ps1", v=good_v(), cues=cues)], kind=kind)


def test_caption_reading(proj):
    fail = run_audit(one_cue_plan("一二三四五六七八九十", 15.0, 16.0), proj)        # 10 chars/s
    assert codes(fail, "caption.reading")[0].severity == "error" and not fail.accepted
    warn = run_audit(one_cue_plan("一二三四五六七八九十", 15.0, 16.7), proj)        # 5.9 chars/s
    assert codes(warn, "caption.reading")[0].severity == "warn" and warn.accepted
    ok = run_audit(one_cue_plan("一二三四五六七八九十", 15.0, 17.5), proj)          # 4 chars/s
    assert not has(ok, "caption.reading")
    punct = run_audit(one_cue_plan("一二三，四五六。七八九？", 15.0, 17.3), proj)   # punctuation does not count
    assert not has(punct, "caption.reading")
    assert "後面還有空檔" in codes(fail, "caption.reading")[0].message


def test_caption_reading_of_fast_speech_packed_back_to_back_is_a_warning(proj):
    """Verbatim subtitles of fast talk: each caption already runs into the next one, so no timing can slow it down --
    a warning that says so (shorten the text), not an error that blocks the whole plan."""
    fast = [CaptionCue(id="ps1_qa", start=15.0, end=16.0, text="一二三四五六七八九十"),
            CaptionCue(id="ps1_qb", start=16.0, end=17.0, text="一二三四五六七八九十")]
    r = run_audit(mk_plan([mk_seq("ps1", v=good_v(), cues=good_cues()[:6] + fast
                                  + [CaptionCue(id="ps1_qc", start=17.0, end=18.5, text="一二三")])]), proj)
    f = {x.where.split("/")[1]: x for x in codes(r, "caption.reading")}
    assert set(f) == {"ps1_qa", "ps1_qb"} and all(x.severity == "warn" for x in f.values())
    assert "說話本身就這麼快" in f["ps1_qa"].message and r.accepted
    # the same fast caption with free time after it could have stayed up longer: still an error
    r2 = run_audit(mk_plan([mk_seq("ps1", v=good_v(), cues=good_cues()[:6] + fast)]), proj)
    f2 = {x.where.split("/")[1]: x for x in codes(r2, "caption.reading")}
    assert f2["ps1_qa"].severity == "warn" and f2["ps1_qb"].severity == "error" and not r2.accepted



def test_card_reading_warns_but_never_blocks(proj):
    """Title cards are exempt from the subtitle rules, but one shown too briefly to read gets a warning (REVIEW-5)."""
    card = {"kind": "card"}
    cues = good_cues()[:6] + [CaptionCue(id="ps1_k1", start=15.0, end=15.6, text="短暫的字卡", style=card),
                              CaptionCue(id="ps1_k2", start=16.0, end=17.5, text="一二三四五六七八九十一二三", style=card),
                              CaptionCue(id="ps1_k3", start=18.0, end=20.0, text="正常的字卡", style=card)]
    r = run_audit(mk_plan([mk_seq("ps1", v=good_v(), cues=cues)]), proj)
    f = {x.where.split("/")[1]: x for x in codes(r, "card.reading")}
    assert set(f) == {"ps1_k1", "ps1_k2"} and all(x.severity == "warn" for x in f.values()) and r.accepted
    assert "只顯示 0.60 秒" in f["ps1_k1"].message and "閱讀速度 8.7 字/秒" in f["ps1_k2"].message
    assert not has(r, "caption.reading") and not has(r, "caption.short")

def test_caption_rate(proj):
    sparse = mk_plan([mk_seq("ps1", v=good_v(), cues=good_cues(n=3))])
    f = codes(run_audit(sparse, proj), "caption.rate")
    assert len(f) == 1 and f[0].severity == "warn"
    assert not has(run_audit(mk_plan(), proj), "caption.rate")                       # 12 cues / 20 s = 36 per min
    short_seq = mk_seq("ps1", v=[clip("ps1", 1, "a1", 0, 0, 1.5), clip("ps1", 2, "a1", 1.5, 1.5, 15.0, transform=zoomed())],
                       cues=good_cues(n=1))
    assert not has(run_audit(mk_plan([short_seq]), proj), "caption.rate")            # < 20 s: not checked
    off = mk_plan([mk_seq("ps1", v=good_v())], kind="assemble")
    assert not has(run_audit(off, proj), "caption.rate")                             # captions are off: nothing "sparse"
    on = mk_plan([mk_seq("ps1", v=good_v())], kind="assemble", brief={"captions": "auto"})
    assert has(run_audit(on, proj), "caption.rate")                                  # asked for captions, got none


def test_caption_overlap(proj):
    r = run_audit(one_cue_plan("重疊的字幕", 0.5, 2.0), proj)                        # q0 spans 0.0-1.5
    f = codes(r, "caption.overlap")
    assert f and f[0].severity == "error" and not r.accepted
    near = run_audit(one_cue_plan("貼著的字幕", 9.46, 10.9), proj)                   # previous cue ends at 9.5: 0.04 s <= 0.05 is fine
    assert not has(near, "caption.overlap")
    assert not has(run_audit(mk_plan(), proj), "caption.overlap")


def test_caption_short_and_long(proj):
    r = run_audit(one_cue_plan("快", 15.0, 15.2), proj)
    assert codes(r, "caption.short")[0].severity == "warn"
    assert not has(run_audit(one_cue_plan("夠長的字幕", 15.0, 16.0), proj), "caption.short")
    long_text = "很長" * 25                                                           # 50 chars > 40
    rl = run_audit(one_cue_plan(long_text, 14.0, 19.0), proj)
    assert codes(rl, "caption.long")[0].severity == "warn"
    assert not has(run_audit(one_cue_plan("不長的字幕", 15.0, 16.0), proj), "caption.long")


def test_caption_range(proj):
    r = run_audit(one_cue_plan("跑到外面去的字幕", 19.0, 21.5), proj)               # sequence ends at 20 s
    assert codes(r, "caption.range")[0].severity == "error" and not r.accepted
    assert not has(run_audit(one_cue_plan("在範圍內", 18.0, 19.8), proj), "caption.range")


# ------------------------------------------------------------------------------------------- assemble.*
def assemble_plan(duration: float, target: float, **extra) -> EditPlan:
    seq = mk_seq("ps1", v=[clip("ps1", 1, "a1", 0.0, 0.0, min(duration, 20.0))], w=1920, h=1080, name="長片")
    brief = {"target_seconds": target, **extra.pop("brief", {})}
    return EditPlan(kind="assemble", sequences=[seq], brief=brief,
                    sources=extra.pop("sources", [{"asset_id": "a1", "name": "a1.mp4", "usable_seconds": 20.0}]), **extra)


def test_assemble_target(proj):
    far = run_audit(assemble_plan(10.0, 60.0, brief={"usable_seconds": 300.0}), proj)
    f = codes(far, "assemble.target")
    assert len(f) == 1 and f[0].severity == "warn" and far.accepted
    near = run_audit(assemble_plan(18.0, 20.0), proj)
    assert not has(near, "assemble.target")                                           # within the 35% tolerance
    zero = run_audit(assemble_plan(10.0, 0.0), proj)
    assert not has(zero, "assemble.target")                                           # 0 = keep everything: no target


def test_assemble_target_short_of_sources_is_info(proj):
    r = run_audit(assemble_plan(10.0, 60.0, brief={"usable_seconds": 10.0}), proj)
    f = codes(r, "assemble.target")
    assert len(f) == 1 and f[0].severity == "info" and "不足" in f[0].message
    over = run_audit(assemble_plan(20.0, 10.0, brief={"usable_seconds": 10.0}), proj)
    assert codes(over, "assemble.target")[0].severity == "warn"                       # too long is never "sources missing"


def test_assemble_unused(proj):
    sources = [{"asset_id": "a1", "name": "a1.mp4", "usable_seconds": 20.0},
               {"asset_id": "a2", "name": "a2.mp4", "usable_seconds": 8.0}]
    r = run_audit(assemble_plan(18.0, 20.0, sources=sources), proj)
    f = codes(r, "assemble.unused")
    assert len(f) == 1 and f[0].severity == "info" and f[0].where == "a2" and "a2.mp4" in f[0].message
    dead = [sources[0], {"asset_id": "a2", "name": "a2.mp4", "usable_seconds": 0.0}]
    assert not has(run_audit(assemble_plan(18.0, 20.0, sources=dead), proj), "assemble.unused")
    assert not has(run_audit(assemble_plan(18.0, 20.0, sources=sources[:1]), proj), "assemble.unused")
    assert not has(run_audit(mk_plan(sources=sources), proj), "assemble.unused")      # shorts plans do not use this rule


# ------------------------------------------------------------------------------------------- start.black / total.max
def black_evidence(level: float) -> Evidence:
    return Evidence.empty("a1", 20.0, visual_fps=5.0, bright=[level] * 4 + [0.6] * 96)


def test_start_black(proj):
    r = run_audit(mk_plan(), proj, evidence={"a1": black_evidence(0.0)})
    f = codes(r, "start.black")
    assert len(f) == 1 and f[0].severity == "warn"
    assert not has(run_audit(mk_plan(), proj, evidence={"a1": black_evidence(0.5)}), "start.black")
    off = AuditRules(require_sources_exist=False, black_start_warn=False)
    assert not has(run_audit(mk_plan(), proj, off, evidence={"a1": black_evidence(0.0)}), "start.black")


def test_start_black_looks_at_the_clip_head_not_the_file_head(proj):
    plan = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "a1", 0.0, 5.0, 6.5),
                                     clip("ps1", 2, "a1", 1.5, 6.5, 20.0, transform=zoomed())], cues=good_cues())])
    assert not has(run_audit(plan, proj, evidence={"a1": black_evidence(0.0)}), "start.black")   # black only before 0.8 s


def test_start_black_without_evidence_says_not_checked(proj):
    r = run_audit(mk_plan(), proj, evidence={})
    assert not has(r, "start.black")
    assert any("start.black" in f.message for f in codes(r, "audit.evidence"))


def test_total_max(proj):
    r = run_audit(mk_plan(), proj, AuditRules(require_sources_exist=False, max_total_seconds=10.0))
    f = codes(r, "total.max")
    assert len(f) == 1 and f[0].severity == "error" and not r.accepted
    assert not has(run_audit(mk_plan(), proj), "total.max")


# ------------------------------------------------------------------------------------------- apply
def setup_apply(proj, plan: EditPlan | None = None, rules: AuditRules = NOFILE):
    ed = Editor(proj)
    plan = plan or mk_plan(decisions=[{"id": "d1", "kind": "short", "sequence": "ps1", "score": 0.82,
                                       "source": {"asset_id": "a1", "t0": 0.0, "t1": 20.0}, "summary": "s", "why": "w"}])
    return ed, plan, audit_plan(plan, proj, rules)


def all_ids(proj) -> list[str]:
    ids = []
    for s in proj.sequences:
        ids.append(s.id)
        for t in s.tracks():
            ids.append(t.id)
            ids += [c.id for c in t.clips]
        ids += [q.id for q in s.cues] + [m.id for m in s.markers]
    return ids


def test_apply_adds_sequence_provenance_and_is_one_undo_step(proj):
    ed, plan, receipt = setup_apply(proj)
    before = proj.content_json()
    n_before = len(proj.sequences)
    ids = apply_plan(ed, plan, receipt)
    assert ids == ["ps1"] and len(proj.sequences) == n_before + 1 and proj.active == "ps1"
    seq = proj.sequence("ps1")
    assert (seq.width, seq.height) == (1080, 1920) and len(seq.cues) == 12
    assert not [i for i in validate_sequence(proj, seq) if i.level == "error"]
    run = find_run(proj, seq.meta["ai"]["run_id"])
    assert run and run["human_review"] == "pending" and run["technical_qa"] == "not_run"
    assert run["kind"] == "shorts" and run["plan_sha256"] == plan.sha256() and run["sequence_ids"] == ["ps1"]
    ai = seq.meta["ai"]
    assert ai["kind"] == "shorts" and ai["plan_sha256"] == plan.sha256() and ai["score"] == 0.82
    assert ai["source"] == {"asset_id": "a1", "t0": 0.0, "t1": 20.0} and ai["plan_sequence"] == "ps1"
    assert plan.sha256() == receipt.plan_sha256                                   # applying never modifies the plan
    # ONE undo step restores the project exactly (assets, sequences, ai_runs record)
    assert ed.history.undo_label.startswith("自動剪輯")
    assert ed.undo() is not None
    assert proj.content_json() == before and not ed.history.can_undo
    ed.redo()
    assert proj.sequence("ps1").meta["ai"]["run_id"] == run["run_id"]


def test_apply_label_and_machine_never_approves(proj):
    ed, plan, receipt = setup_apply(proj)
    apply_plan(ed, plan, receipt)
    run = proj.meta["ai_runs"][-1]
    assert run["human_review"] == "pending" and run["outcomes"] == []


def test_later_project_edit_voids_the_receipt(proj):
    ed, plan, receipt = setup_apply(proj)
    ed.op("加標記", "add_marker", None, 1.0, "m")
    before, steps = proj.content_json(), len(ed.history._undo)
    with pytest.raises(PlanError, match="編輯過"):
        apply_plan(ed, plan, receipt)
    assert proj.content_json() == before and len(ed.history._undo) == steps
    receipt2 = audit_plan(plan, proj, NOFILE)                                     # a fresh audit works again
    assert apply_plan(ed, plan, receipt2) == ["ps1"]


def test_a_receipt_cannot_be_used_twice(proj):
    ed, plan, receipt = setup_apply(proj)
    apply_plan(ed, plan, receipt)
    with pytest.raises(PlanError):
        apply_plan(ed, plan, receipt)                                             # apply bumped the revision


def test_tampered_plan_is_rejected_and_nothing_changes(proj):
    ed, plan, receipt = setup_apply(proj)
    plan.sequences[0]["video_tracks"][0]["clips"][0]["src_out"] = 1.0             # edited after the audit
    before = proj.content_json()
    with pytest.raises(PlanError, match="計畫"):
        apply_plan(ed, plan, receipt)
    assert proj.content_json() == before and not ed.history.can_undo


def test_rejected_receipt_cannot_apply(proj):
    bad = mk_plan([mk_seq("ps1", v=[clip("ps1", 1, "ghost", 0, 0, 5)])])
    ed, _plan, receipt = setup_apply(proj, bad)
    assert not receipt.accepted
    before = proj.content_json()
    with pytest.raises(PlanError, match="審核未通過"):
        apply_plan(ed, bad, receipt)
    assert proj.content_json() == before


def test_receipt_for_another_plan_is_rejected(proj):
    ed, plan, receipt = setup_apply(proj)
    other = mk_plan([good_short("ps2")])
    with pytest.raises(PlanError):
        apply_plan(ed, other, receipt)


def test_invalid_sequence_rolls_back_the_whole_batch(proj):
    plan = mk_plan([good_short("ps1"), mk_seq("ps2", v=[clip("ps2", 1, "a1", 0, 0, 50)])])     # 2nd is out of range
    ed = Editor(proj)
    forged = AuditReceipt(plan_sha256=plan.sha256(), project_revision=proj.revision, project_sha256=proj.content_sha256(),
                          accepted=True)                                          # as if a sloppy auditor let it through
    before = proj.content_json()
    with pytest.raises(PlanError, match="不合法"):
        apply_plan(ed, plan, forged)
    assert proj.content_json() == before and not ed.history.can_undo               # first sequence rolled back too


def test_empty_plan_cannot_apply(proj):
    plan = EditPlan(kind="shorts")
    ed = Editor(proj)
    forged = AuditReceipt(plan_sha256=plan.sha256(), project_revision=proj.revision, project_sha256="", accepted=True)
    with pytest.raises(PlanError):
        apply_plan(ed, plan, forged)


def test_multiple_sequences_apply_together(proj):
    plan = mk_plan([good_short("ps1"), good_short("ps2"), good_short("ps3")])
    ed, plan, receipt = setup_apply(proj, plan)
    before = proj.content_json()
    ids = apply_plan(ed, plan, receipt)
    assert ids == ["ps1", "ps2", "ps3"] and proj.active == "ps1"
    assert len({proj.meta["ai_runs"][-1]["run_id"]}) == 1
    assert {proj.sequence(i).meta["ai"]["run_id"] for i in ids} == {proj.meta["ai_runs"][-1]["run_id"]}
    ed.undo()
    assert proj.content_json() == before


def test_id_collisions_are_reissued(proj):
    plan = mk_plan([good_short("ps1")])
    ed, plan, r1 = setup_apply(proj, plan)
    apply_plan(ed, plan, r1)
    first = copy.deepcopy(proj.sequence("ps1").to_dict())
    r2 = audit_plan(plan, proj, NOFILE)
    ids2 = apply_plan(ed, plan, r2)                                               # the same plan again: every id collides
    assert ids2 != ["ps1"] and ids2[0].startswith("ps1")
    ids = all_ids(proj)
    assert len(ids) == len(set(ids))
    assert proj.sequence("ps1").to_dict() == first                                # the first copy is untouched
    new = proj.sequence(ids2[0])
    assert new.meta["ai"]["plan_sequence"] == "ps1" and len(new.cues) == 12
    assert not [i for i in validate_sequence(proj, new) if i.level == "error"]
    ed.undo()
    assert [s.id for s in proj.sequences].count("ps1") == 1


def music_asset(aid: str = "pm1", path: str = "/nonexistent/music.mp3") -> Asset:
    return Asset(id=aid, path=path, name="music.mp3", kind="audio", duration=30.0, has_video=False, has_audio=True,
                 audio_channels=2, sample_rate=44100, fingerprint="mf")


def music_plan(asset: Asset) -> EditPlan:
    seq = mk_seq("ps1", v=good_v(), a=[clip("ps1", 9, asset.id, 0.0, 0.0, 20.0, volume=0.1)], cues=good_cues())
    return mk_plan([seq], assets=[asset.to_dict()])


def test_apply_adds_plan_assets(proj):
    ed, plan, receipt = setup_apply(proj, music_plan(music_asset()))
    before = proj.content_json()
    apply_plan(ed, plan, receipt)
    assert "pm1" in proj.assets and proj.assets["pm1"].path == "/nonexistent/music.mp3"
    a1 = proj.sequence("ps1").audio_tracks[0].clips[0]
    assert a1.asset_id == "pm1" and abs(a1.volume - 0.1) < 1e-9
    ed.undo()
    assert proj.content_json() == before and "pm1" not in proj.assets            # the asset went with the undo


def test_plan_asset_with_an_existing_path_reuses_the_project_asset(proj):
    plan = music_plan(music_asset("pm9", path="/nonexistent/m1.mp3"))             # m1 already points at this file
    ed, plan, receipt = setup_apply(proj, plan)
    n_assets = len(proj.assets)
    id_map = plan_asset_map(proj, plan)[0]
    assert id_map == {"pm9": "m1"}
    apply_plan(ed, plan, receipt)
    assert len(proj.assets) == n_assets and "pm9" not in proj.assets
    assert proj.sequence("ps1").audio_tracks[0].clips[0].asset_id == "m1"


def test_plan_asset_id_collision_is_reissued(proj):
    clash = music_asset("a1", path="/nonexistent/other.mp3")                      # id "a1" belongs to a different file
    seq = mk_seq("ps1", v=[clip("ps1", 1, "a2", 0.0, 0.0, 10.0)],
                 a=[clip("ps1", 9, "a1", 0.0, 0.0, 10.0)])
    plan = mk_plan([seq], assets=[clash.to_dict()])
    id_map, fresh = plan_asset_map(proj, plan)
    assert id_map == {"a1": "a1-2"} and list(fresh) == ["a1-2"] and "a1-2" not in proj.assets   # pure: nothing added yet
    ids = remap_plan_assets(proj, plan)
    assert ids == {"a1": "a1-2"} and proj.assets["a1-2"].path == "/nonexistent/other.mp3"
    assert proj.assets["a1"].path == "/nonexistent/a1.mp4"                          # the original asset is untouched


def test_remap_is_shared_with_preview_project(proj):
    plan = music_plan(music_asset())
    before = proj.content_json()
    scratch, sid = preview_project(proj, plan, 0)
    assert sid == "ps1" and "pm1" in scratch.assets and scratch.active == "ps1"
    assert scratch.sequence(sid).audio_tracks[0].clips[0].asset_id == "pm1"
    assert proj.content_json() == before and "pm1" not in proj.assets             # the real project is never touched
    assert scratch.sequence(sid).cues and scratch.sequence(sid).cues[0].text


def test_apply_snaps_to_the_frame_grid_and_keeps_markers_and_style(proj):
    seq = mk_seq("ps1", v=[clip("ps1", 1, "a1", 0.0, 0.0, 1.5),
                           clip("ps1", 2, "a1", 1.5011, 1.5, 20.0, transform=zoomed())],
                 cues=good_cues(), markers=[Marker(id="ps1_m1", time=0.0, name="a1.mp4")])
    ed, plan, receipt = setup_apply(proj, mk_plan([seq]))
    apply_plan(ed, plan, receipt)
    s = proj.sequence("ps1")
    assert abs(s.video_tracks[0].clips[1].start - 1.5) < 1e-6                      # 1.5011 -> frame 45 -> 1.5
    assert [m.name for m in s.markers] == ["a1.mp4"]


# ------------------------------------------------------------------------------------------- receipts are bound to content
def test_a_receipt_is_bound_to_the_project_content_not_only_to_its_revision(proj):
    """Revisions are plain per-project counters: a receipt made for one project must not apply to another one that
    merely happens to be at the same revision."""
    ed, plan, receipt = setup_apply(proj)
    twin = Project.new("twin")
    twin.assets["a1"] = fake_asset("a1", 20.0)
    twin.revision = proj.revision
    assert twin.revision == proj.revision and twin.content_sha256() != proj.content_sha256()
    before = twin.content_json()
    with pytest.raises(PlanError, match="內容"):
        apply_plan(Editor(twin), plan, receipt)
    assert twin.content_json() == before and "ai_runs" not in twin.meta


def test_an_edit_that_did_not_bump_the_revision_still_voids_the_receipt(proj):
    ed, plan, receipt = setup_apply(proj)
    proj.assets["a1"].duration = 19.0                                             # changed behind the editor's back
    before = proj.content_json()
    with pytest.raises(PlanError, match="內容"):
        apply_plan(ed, plan, receipt)
    assert proj.content_json() == before and not ed.history.can_undo


def test_a_receipt_file_without_its_hash_is_not_accepted():
    r = AuditReceipt(plan_sha256="p" * 64, project_revision=3, project_sha256="q" * 64, accepted=True)
    d = r.to_dict()
    assert AuditReceipt.from_dict(d).sha256() == r.sha256()                       # the honest round trip works
    stripped = {k: v for k, v in d.items() if k != "receipt_sha256"}
    with pytest.raises(PlanError, match="雜湊"):
        AuditReceipt.from_dict(stripped)                                          # nothing to check it against
    with pytest.raises(PlanError, match="雜湊"):
        AuditReceipt.from_dict({**d, "receipt_sha256": ""})
    with pytest.raises(PlanError, match="不符"):
        AuditReceipt.from_dict({**d, "findings": [{"rule": "x", "severity": "info", "message": "m"}]})   # edited after


# ------------------------------------------------------------------------------------------- QA / review belong to one sequence
def two_sequence_run(proj):
    plan = mk_plan([good_short("ps1"), good_short("ps2")])
    ed, plan, receipt = setup_apply(proj, plan)
    apply_plan(ed, plan, receipt)
    return ed, proj.meta["ai_runs"][-1], proj.sequence("ps1"), proj.sequence("ps2")


def test_a_check_and_a_review_belong_to_one_sequence_of_the_run(proj):
    ed, run, s1, s2 = two_sequence_run(proj)
    assert planmod.qa_state(run, s1) == ("not_run", "") and planmod.review_state(run, s2) == ("pending", "")
    planmod.set_technical_qa(proj, run["run_id"], "pass", "ok", sequence=s1)
    planmod.set_human_review(proj, run["run_id"], "approved", "good", sequence=s1)
    assert planmod.qa_state(run, s1) == ("pass", "ok") and planmod.review_state(run, s1) == ("approved", "good")
    assert planmod.qa_state(run, s2) == ("not_run", "")                           # the other sequence was never checked
    assert planmod.review_state(run, s2) == ("pending", "")                       # ... nor reviewed
    planmod.set_human_review(proj, run["run_id"], "rejected", "no", sequence=s2)
    assert planmod.review_state(run, s1)[0] == "approved" and planmod.review_state(run, s2) == ("rejected", "no")
    assert [o["sequence"] for o in run["outcomes"]] == ["ps1", "ps2"]             # every decision is kept, with its sequence


def test_an_edit_after_the_check_or_the_review_makes_them_stale(proj):
    ed, run, s1, _s2 = two_sequence_run(proj)
    planmod.set_technical_qa(proj, run["run_id"], "warn", "響度偏高", sequence=s1)
    planmod.set_human_review(proj, run["run_id"], "approved", "", sequence=s1)
    ed.op("加標記", "add_marker", "ps1", 1.0, "m")                                # any change of the content
    assert planmod.qa_state(run, s1) == (planmod.STALE, "響度偏高") and planmod.review_state(run, s1)[0] == planmod.STALE
    ed.undo()                                                                     # back to exactly what was checked
    back, run = proj.sequence("ps1"), find_run(proj, run["run_id"])               # undo rebuilds the objects
    assert planmod.qa_state(run, back)[0] == "warn" and planmod.review_state(run, back)[0] == "approved"


def test_renaming_a_sequence_does_not_make_a_review_stale(proj):
    ed, run, s1, _s2 = two_sequence_run(proj)
    planmod.set_human_review(proj, run["run_id"], "approved", "", sequence=s1)
    ed.op("改名", "rename_sequence", "ps1", "新名字")
    assert planmod.review_state(run, s1)[0] == "approved"


def test_a_check_of_a_file_rendered_earlier_is_stale_when_the_timeline_moved_on(proj):
    """The result belongs to the content that was RENDERED (``sequence_sha``), not to whatever is on screen when it
    arrives: a person who edited during the export must not see a green check for the old picture."""
    ed, run, s1, _s2 = two_sequence_run(proj)
    rendered = planmod.sequence_sha256(s1)
    ed.op("加標記", "add_marker", "ps1", 2.0, "m")
    planmod.set_technical_qa(proj, run["run_id"], "pass", "ok", sequence=s1, sequence_sha=rendered)
    assert planmod.qa_state(run, s1)[0] == planmod.STALE


def test_a_duplicated_sequence_does_not_inherit_the_run_or_its_approval(proj):
    ed, run, s1, _s2 = two_sequence_run(proj)
    planmod.set_human_review(proj, run["run_id"], "approved", "", sequence=s1)
    twin = ed.op("複製序列", "duplicate_sequence", "ps1")
    assert "ai" not in twin.meta and "ai" in proj.sequence("ps1").meta            # no link to the run -> no approval
    assert planmod.review_state(run, twin) == ("pending", "") and planmod.qa_state(run, twin) == ("not_run", "")
    assert run["sequence_ids"] == ["ps1", "ps2"]


def test_a_review_of_an_unknown_run_is_refused(proj):
    ed, run, s1, _s2 = two_sequence_run(proj)
    with pytest.raises(PlanError):
        planmod.set_human_review(proj, "run_nope", "approved", "", sequence=s1)
    with pytest.raises(PlanError):
        planmod.set_technical_qa(proj, "run_nope", "pass", "", sequence=s1)
    with pytest.raises(PlanError):
        planmod.set_technical_qa(proj, run["run_id"], "great", "", sequence=s1)
