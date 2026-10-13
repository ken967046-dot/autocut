"""Edit plan -> audit receipt -> apply: the evidence chain behind every automatic edit.

Adapted from the Editkin v4 workflow of video-autopilot-kit (MIT, (c) 2026 Hao0321 Studio):

    evidence  ->  plan (canonical-JSON sha256)  ->  audit receipt bound to that hash and to
    the project revision  ->  atomic apply (one undo step)  ->  render  ->  technical QA
    ->  human review (a person, never the machine)  ->  outcome log

* A plan is *data*; building one never touches the project.
* A receipt only authorises the exact plan hash and project revision it was issued for.
  Change the plan, or edit the project, and the receipt is void (fail closed).
* Machines may only say ``technical_pass``; ``approved`` can only be set by
  :func:`set_human_review`, which is called from an explicit user action in the GUI.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util import atomic_write_text, canonical_json, new_id, now_iso, sha256_hex
from .project import Project, Sequence

PLAN_SCHEMA = "autocut.edit-plan/1"
RECEIPT_SCHEMA = "autocut.audit-receipt/1"
AUDITOR_VERSION = "1"

ERROR, WARN, INFO = "error", "warn", "info"


class PlanError(ValueError):
    pass


@dataclass
class Finding:
    rule: str
    severity: str          # error | warn | info
    message: str
    where: str = ""

    def to_dict(self) -> dict:
        return {"rule": self.rule, "severity": self.severity, "message": self.message, "where": self.where}

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        return cls(d.get("rule", ""), d.get("severity", INFO), d.get("message", ""), d.get("where", ""))


@dataclass
class EditPlan:
    kind: str                                  # shorts | assemble | longform
    brief: dict = field(default_factory=dict)  # user settings + thresholds actually used
    sources: list = field(default_factory=list)      # [{asset_id, name, fingerprint, duration, evidence_sha256}]
    assets: list = field(default_factory=list)       # Asset.to_dict() of files the plan needs that the project lacks (music)
    sequences: list = field(default_factory=list)    # Sequence.to_dict() with deterministic ids
    decisions: list = field(default_factory=list)    # [{id, kind, summary, why, evidence:[...]}]
    notes: list = field(default_factory=list)
    created: str = field(default_factory=now_iso)
    schema: str = PLAN_SCHEMA

    # ---- hashing (``created`` is volatile and not part of the hash)
    def content(self) -> dict:
        return {
            "schema": self.schema, "kind": self.kind, "brief": self.brief, "sources": self.sources,
            "assets": self.assets, "sequences": self.sequences, "decisions": self.decisions, "notes": self.notes,
        }

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    def sequence_objects(self) -> list[Sequence]:
        return [Sequence.from_dict(copy.deepcopy(s)) for s in self.sequences]

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.sequence_objects())

    def subset(self, indexes: list[int]) -> "EditPlan":
        """A new plan keeping only the chosen sequences (and their decisions). It has a new hash, so it must be
        audited again before it can be applied."""
        keep = [self.sequences[i] for i in indexes if 0 <= i < len(self.sequences)]
        ids = {s.get("id") for s in keep}
        decisions = [d for d in self.decisions if d.get("sequence") in ids or d.get("sequence") in (None, "")]
        return EditPlan(kind=self.kind, brief=copy.deepcopy(self.brief), sources=copy.deepcopy(self.sources),
                        assets=copy.deepcopy(self.assets),
                        sequences=copy.deepcopy(keep), decisions=copy.deepcopy(decisions),
                        notes=list(self.notes) + [f"subset {sorted(indexes)} of {len(self.sequences)}"])

    # ---- io
    def to_dict(self) -> dict:
        d = self.content()
        d["created"] = self.created
        d["plan_sha256"] = self.sha256()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EditPlan":
        if d.get("schema") != PLAN_SCHEMA:
            raise PlanError("不是 AutoCut 剪輯計畫檔")
        p = cls(kind=d.get("kind", ""), brief=d.get("brief", {}), sources=d.get("sources", []),
                assets=d.get("assets", []), sequences=d.get("sequences", []), decisions=d.get("decisions", []),
                notes=d.get("notes", []), created=d.get("created", ""))
        claimed = d.get("plan_sha256")
        if claimed and claimed != p.sha256():
            raise PlanError("計畫檔內容與其雜湊值不符（可能被修改過）")
        return p

    def save(self, path: str | Path) -> None:
        atomic_write_text(path, json.dumps(self.to_dict(), ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "EditPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class AuditReceipt:
    plan_sha256: str
    project_revision: int
    project_sha256: str
    accepted: bool
    findings: list = field(default_factory=list)       # list[Finding]
    thresholds: dict = field(default_factory=dict)
    created: str = field(default_factory=now_iso)
    auditor: str = AUDITOR_VERSION
    schema: str = RECEIPT_SCHEMA

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARN]

    def content(self) -> dict:
        return {
            "schema": self.schema, "auditor": self.auditor, "plan_sha256": self.plan_sha256,
            "project_revision": self.project_revision, "project_sha256": self.project_sha256,
            "accepted": self.accepted, "findings": [f.to_dict() for f in self.findings],
            "thresholds": self.thresholds,
        }

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    def to_dict(self) -> dict:
        d = self.content()
        d["created"] = self.created
        d["receipt_sha256"] = self.sha256()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AuditReceipt":
        r = cls(plan_sha256=d["plan_sha256"], project_revision=int(d["project_revision"]),
                project_sha256=d.get("project_sha256", ""), accepted=bool(d["accepted"]),
                findings=[Finding.from_dict(f) for f in d.get("findings", [])],
                thresholds=d.get("thresholds", {}), created=d.get("created", ""),
                auditor=d.get("auditor", ""), schema=d.get("schema", RECEIPT_SCHEMA))
        claimed = d.get("receipt_sha256")
        if not claimed:
            raise PlanError("審核收據缺少雜湊值（不是這個軟體產生的收據）")
        if claimed != r.sha256():
            raise PlanError("審核收據內容與其雜湊值不符")
        return r


def check_receipt(plan: EditPlan, receipt: AuditReceipt, project: Project) -> None:
    """Raise :class:`PlanError` unless ``receipt`` authorises applying ``plan`` to ``project`` right now."""
    if receipt.schema != RECEIPT_SCHEMA:
        raise PlanError("審核收據格式不符")
    if not receipt.accepted:
        raise PlanError("審核未通過，不能套用（請先修正計畫中的錯誤項目）")
    if receipt.plan_sha256 != plan.sha256():
        raise PlanError("審核收據不是針對這份計畫（計畫在審核後被改動）")
    if receipt.project_revision != project.revision:
        raise PlanError("專案在審核之後又被編輯過，請重新審核")
    if receipt.project_sha256 != project.content_sha256():       # revisions are per-project counters: they can coincide
        raise PlanError("審核收據不是針對這個專案的內容（換了專案，或專案內容在審核後改變），請重新審核")


# --------------------------------------------------------------------- provenance
def record_run(project: Project, plan: EditPlan, receipt: AuditReceipt, sequence_ids: list[str]) -> dict:
    """Remember what was applied. ``human_review`` starts as ``pending`` and only a person changes it."""
    run = {
        "run_id": new_id("run_"), "kind": plan.kind, "plan_sha256": plan.sha256(),
        "receipt_sha256": receipt.sha256(), "applied_at": now_iso(), "sequence_ids": list(sequence_ids),
        "warnings": [f.message for f in receipt.warnings], "technical_qa": "not_run",
        "human_review": "pending", "outcomes": [],
    }
    project.meta.setdefault("ai_runs", []).append(run)
    return run


def find_run(project: Project, run_id: str) -> dict | None:
    for r in project.meta.get("ai_runs", []):
        if r.get("run_id") == run_id:
            return r
    return None


STALE = "stale"


def sequence_sha256(seq: Sequence) -> str:
    """Hash of what a person sees in a sequence (format, tracks, clips, captions, markers, export range).

    The name and the link to the auto-edit run (``meta["ai"]``) are not content: renaming a sequence does not make a
    check or a review stale, but any edit of the picture, sound or captions does."""
    d = seq.to_dict()
    d.pop("name", None)
    meta = dict(d.get("meta") or {})
    meta.pop("ai", None)
    d["meta"] = meta
    return sha256_hex(canonical_json(d))


def _slot(run: dict, seq_id: str) -> dict:
    return run.setdefault("by_sequence", {}).setdefault(seq_id, {})


def set_technical_qa(project: Project, run_id: str, status: str, summary: str = "", sequence: Sequence | None = None,
                     sequence_sha: str | None = None) -> None:
    """``status``: not_run | pass | warn | fail. This is machine evidence, never taste approval.

    ``sequence`` (and ``sequence_sha``: the hash of the content that was actually rendered, when it is not the content
    right now) ties the result to ONE sequence of the run: see :func:`qa_state`.  The run-level fields stay as the
    latest result for readers that only know the run."""
    run = find_run(project, run_id)
    if run is None:
        raise PlanError("找不到這次自動剪輯紀錄")
    if status not in ("not_run", "pass", "warn", "fail"):
        raise PlanError("未知的技術檢查狀態")
    run["technical_qa"] = status
    run["technical_qa_summary"] = summary
    if sequence is not None:
        _slot(run, sequence.id).update({
            "technical_qa": status, "technical_qa_summary": summary, "technical_qa_at": now_iso(),
            "technical_qa_sha": sequence_sha or sequence_sha256(sequence)})


def set_human_review(project: Project, run_id: str, verdict: str, notes: str = "", sequence: Sequence | None = None) -> None:
    """Record a person's verdict (approved | needs_changes | rejected) for ``sequence`` (see :func:`review_state`).

    Only ever call this from an explicit user action (the "我已審片" button); automation must not.
    """
    run = find_run(project, run_id)
    if run is None:
        raise PlanError("找不到這次自動剪輯紀錄")
    if verdict not in ("approved", "needs_changes", "rejected"):
        raise PlanError("未知的審片結果")
    at = now_iso()
    run["human_review"] = verdict
    run["human_review_at"] = at
    run["human_review_notes"] = notes
    run["outcomes"].append({"at": at, "verdict": verdict, "notes": notes, "sequence": sequence.id if sequence else None})
    if sequence is not None:
        _slot(run, sequence.id).update({"human_review": verdict, "human_review_notes": notes, "human_review_at": at,
                                        "human_review_sha": sequence_sha256(sequence)})


def qa_state(run: dict, seq: Sequence) -> tuple[str, str]:
    """``(status, summary)`` of the technical check that applies to ``seq`` right now.

    ``not_run`` when no check was made for THIS sequence (a run can hold several), ``stale`` when the sequence was edited
    after its check (the result describes a file that no longer matches the timeline)."""
    slot = (run.get("by_sequence") or {}).get(seq.id)
    if not slot:
        return "not_run", ""
    status = slot.get("technical_qa", "not_run")
    if status != "not_run" and slot.get("technical_qa_sha") != sequence_sha256(seq):
        return STALE, str(slot.get("technical_qa_summary", ""))
    return status, str(slot.get("technical_qa_summary", ""))


def review_state(run: dict, seq: Sequence) -> tuple[str, str]:
    """``(verdict, notes)`` of the person's review of ``seq``: ``pending`` when this sequence was never reviewed (another
    sequence of the same run, or a duplicate, does not count), ``stale`` when it was edited after the review."""
    slot = (run.get("by_sequence") or {}).get(seq.id)
    if not slot or slot.get("human_review") not in ("approved", "needs_changes", "rejected"):
        return "pending", ""
    if slot.get("human_review_sha") != sequence_sha256(seq):
        return STALE, str(slot.get("human_review_notes", ""))
    return slot["human_review"], str(slot.get("human_review_notes", ""))
