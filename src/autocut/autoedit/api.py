"""Public API of the automatic editor (what the GUI and CLI call).

    evidence  ->  propose_*()  -> EditPlan (data only)  ->  audit_plan() -> AuditReceipt
              ->  apply_plan()  (one undo step)  ->  render  ->  technical QA  ->  human review

Every function takes ``progress(fraction: float, message: str)`` and a ``cancel`` ``threading.Event`` so it can run
in a worker thread.  Nothing here touches the project except :func:`apply_plan`.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional, Sequence

from ..core.editor import Editor
from ..core.plan import AuditReceipt, EditPlan
from ..core.project import Asset, Project, Sequence
from ..media.ffmpeg import Cancelled          # one cancel exception for ffmpeg work *and* planning (re-exported)
from .config import AssembleParams, AuditRules, LongformParams, ShortsParams

ProgressCb = Callable[[float, str], None]


class AutoEditError(RuntimeError):
    """User-facing failure (message in 繁體中文)."""


def propose_shorts(proj: Project, asset_id: str, params: ShortsParams | None = None,
                   progress: Optional[ProgressCb] = None, cancel: Optional[threading.Event] = None) -> EditPlan:
    """Plan up to ``params.count`` 9:16 shorts from one long video. Each short becomes one sequence in the plan."""
    from .shorts import plan_shorts
    return plan_shorts(proj, asset_id, params or ShortsParams(), progress, cancel)


def propose_assemble(proj: Project, asset_ids: Sequence[str], params: AssembleParams | None = None,
                     progress: Optional[ProgressCb] = None, cancel: Optional[threading.Event] = None) -> EditPlan:
    """Plan one long video from many raw clips (a single sequence in the plan)."""
    from .assemble import plan_assemble
    return plan_assemble(proj, list(asset_ids), params or AssembleParams(), progress, cancel)


def propose_longform(proj: Project, asset_id: str, params: LongformParams | None = None,
                     progress: Optional[ProgressCb] = None, cancel: Optional[threading.Event] = None) -> EditPlan:
    """Plan a tighter long video from one long talk video: highlights + filler words removed + speaker close-ups."""
    from .longform import plan_longform
    return plan_longform(proj, asset_id, params or LongformParams(), progress, cancel)


def audit_plan(plan: EditPlan, proj: Project, rules: AuditRules | None = None) -> AuditReceipt:
    """Run the audit rules; the receipt is bound to the plan hash and the project's current revision."""
    from .audit import audit
    return audit(plan, proj, rules or AuditRules())


def apply_plan(editor: Editor, plan: EditPlan, receipt: AuditReceipt) -> list[str]:
    """Atomically add the plan's sequences to the project (one undo step). Returns the new sequence ids."""
    from .apply import apply
    return apply(editor, plan, receipt)


def preview_project(proj: Project, plan: EditPlan, index: int = 0) -> tuple[Project, str]:
    """A throw-away copy of ``proj`` that also contains the plan's assets and its ``index``-th sequence.

    The GUI previews a candidate with ``LiveCompositor(scratch_project)`` without touching the real project or its
    undo history.  Asset ids that the plan introduces are kept; a plan asset whose path already exists in the project is
    mapped onto the existing asset (same rule as :func:`apply_plan`).
    """
    from .apply import remap_plan_assets
    scratch = Project.from_dict(proj.to_dict())
    scratch.path = None
    seq_dict = plan.sequences[index]
    id_map = remap_plan_assets(scratch, plan)
    seq = Sequence.from_dict(seq_dict)
    for _t, clip in seq.all_clips():
        clip.asset_id = id_map.get(clip.asset_id, clip.asset_id)
    scratch.sequences.append(seq)
    scratch.active = seq.id
    return scratch, seq.id
