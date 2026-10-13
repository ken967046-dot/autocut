"""Apply: put an audited :class:`~autocut.core.plan.EditPlan` into the project as ONE undo step (spec §6).

    check_receipt (fail closed, nothing touched on failure)
      -> one ``Editor.do`` transaction:  remap plan assets -> add the plan's sequences (re-id on collision)
         -> normalise + validate -> provenance (``project.meta["ai_runs"]``, ``seq.meta["ai"]``)

Any exception inside the transaction rolls the whole batch back, and one ``Editor.undo()`` removes everything this module
added (new assets, new sequences, the run record).  ``human_review`` always starts as ``pending``: only a person changes
it (``core.plan.set_human_review``).
"""
from __future__ import annotations

import copy
import os
from typing import Optional

from ..core import ops
from ..core.editor import Editor
from ..core.plan import AuditReceipt, EditPlan, PlanError, check_receipt, record_run
from ..core.project import Asset, Project, Sequence, validate_sequence

KIND_LABEL = {"shorts": "短影音", "assemble": "長片組裝", "longform": "長片精華"}


# ------------------------------------------------------------------------------------------------ ids
def unique_id(base: str, used: set[str]) -> str:
    """``base`` when free, else ``base-2``, ``base-3`` ... (deterministic, never random)."""
    if base not in used:
        return base
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    return f"{base}-{n}"


def _project_ids(project: Project) -> set[str]:
    """Every sequence / track / clip / cue / marker id already in use (one namespace: ids never repeat)."""
    used: set[str] = set()
    for s in project.sequences:
        used.add(s.id)
        for t in s.tracks():
            used.add(t.id)
            used.update(c.id for c in t.clips)
        used.update(q.id for q in s.cues)
        used.update(m.id for m in s.markers)
    return used


def _reid_sequence(seq: Sequence, used: set[str]) -> None:
    """Give ``seq`` and everything in it project-unique ids; ids that do not collide are kept as the plan wrote them."""
    def take(old: str) -> str:
        new = unique_id(old, used)
        used.add(new)
        return new

    seq.id = take(seq.id)
    for t in seq.tracks():
        t.id = take(t.id)
        for c in t.clips:
            c.id = take(c.id)
    for q in seq.cues:
        q.id = take(q.id)
    for m in seq.markers:
        m.id = take(m.id)


# ------------------------------------------------------------------------------------------------ assets
def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def plan_asset_map(project: Project, plan: EditPlan) -> tuple[dict[str, str], dict[str, Asset]]:
    """Pure version of :func:`remap_plan_assets`: ``(id_map, fresh)`` where ``fresh`` holds the assets (already carrying
    their final, project-unique ids) that would have to be added.  Nothing in ``project`` is modified."""
    id_map: dict[str, str] = {}
    fresh: dict[str, Asset] = {}
    taken = set(project.assets)
    fresh_by_path: dict[str, str] = {}
    for d in plan.assets:
        a = Asset.from_dict(copy.deepcopy(d))
        plan_id = a.id
        existing = project.find_asset_by_path(a.path) if a.path else None
        if existing is not None:                                   # same file already in the project: reuse it
            id_map[plan_id] = existing.id
            continue
        key = _path_key(a.path) if a.path else ""
        if key and key in fresh_by_path:                           # two plan assets pointing at the same file
            id_map[plan_id] = fresh_by_path[key]
            continue
        a.id = unique_id(plan_id, taken)                           # a different project asset owns that id: re-id
        taken.add(a.id)
        fresh[a.id] = a
        if key:
            fresh_by_path[key] = a.id
        id_map[plan_id] = a.id
    return id_map, fresh


def remap_plan_assets(project: Project, plan: EditPlan) -> dict[str, str]:
    """Add ``plan.assets`` to ``project`` (a plan asset whose path is already in the project reuses that asset; an id that
    collides with a different asset is re-issued) and return ``{plan asset id: project asset id}``.

    Shared by :func:`apply` and ``api.preview_project``.  Clips must be repointed with ``id_map.get(id, id)``.
    """
    id_map, fresh = plan_asset_map(project, plan)
    for a in fresh.values():
        ops.add_asset(project, a)
    return id_map


# ------------------------------------------------------------------------------------------------ apply
def _ai_meta(plan: EditPlan, plan_seq_id: str, seq: Sequence, run_id: str) -> dict:
    """What ``seq.meta["ai"]`` says about where this sequence came from."""
    source: dict = {}
    score: Optional[float] = None
    for d in plan.decisions:
        if d.get("sequence") == plan_seq_id and d.get("kind") == "short" and isinstance(d.get("source"), dict):
            source = copy.deepcopy(d["source"])
            score = d.get("score")
            break
    if not source:
        ids: list[str] = []
        for t in seq.video_tracks:                                   # footage only (music is not a "source")
            for c in t.clips:
                if c.asset_id not in ids:
                    ids.append(c.asset_id)
        source = {"asset_ids": ids}
    ai = dict(seq.meta.get("ai") or {})
    ai.update({"run_id": run_id, "kind": plan.kind, "plan_sha256": plan.sha256(), "plan_sequence": plan_seq_id,
               "source": source, "score": score})
    return ai


def apply(editor: Editor, plan: EditPlan, receipt: AuditReceipt) -> list[str]:
    """Add the plan's sequences to ``editor.project`` as one undoable step and return their (project) ids.

    Raises :class:`PlanError` -- before touching anything -- when the receipt does not authorise exactly this plan hash
    and this project revision; raises it from inside the transaction (so everything is rolled back) when a resulting
    sequence is structurally invalid.
    """
    check_receipt(plan, receipt, editor.project)
    if not plan.sequences:
        raise PlanError("計畫裡沒有任何序列，沒有東西可以套用")
    label = f"自動剪輯：{KIND_LABEL.get(plan.kind, plan.kind)}（{len(plan.sequences)} 個序列）"

    def run(project: Project) -> list[str]:
        id_map = remap_plan_assets(project, plan)
        used = _project_ids(project)
        added: list[tuple[Sequence, str]] = []
        for sd in plan.sequences:
            seq = Sequence.from_dict(copy.deepcopy(sd))
            plan_seq_id = seq.id
            _reid_sequence(seq, used)
            for _t, clip in seq.all_clips():
                clip.asset_id = id_map.get(clip.asset_id, clip.asset_id)
            project.sequences.append(seq)
            ops.normalize_sequence(project, seq)
            errors = [i for i in validate_sequence(project, seq) if i.level == "error"]
            if errors:
                raise PlanError(f"計畫中的序列「{seq.name}」不合法，已整批取消：" +
                                "；".join(f"{i.message}（{i.where}）" for i in errors[:3]))
            added.append((seq, plan_seq_id))
        seq_ids = [s.id for s, _ in added]
        project.active = seq_ids[0]
        record = record_run(project, plan, receipt, seq_ids)            # human_review == "pending", technical_qa "not_run"
        for seq, plan_seq_id in added:
            seq.meta["ai"] = _ai_meta(plan, plan_seq_id, seq, record["run_id"])
        return seq_ids

    return editor.do(label, run)
