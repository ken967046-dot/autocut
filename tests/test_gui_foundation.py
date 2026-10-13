"""AppState / TaskRunner / shared widgets (offscreen Qt)."""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from autocut.core import ops                                  # noqa: E402
from autocut.core.plan import EditPlan, AuditReceipt, record_run   # noqa: E402
from autocut.core.project import Project                      # noqa: E402
from autocut.gui.state import AppState                        # noqa: E402
from autocut.gui.widgets import HumanReviewBox, TimecodeEdit  # noqa: E402
from autocut.gui import workers                                     # noqa: E402
from autocut.gui.workers import TaskRunner, wait_for_idle     # noqa: E402
from autocut.media.ffmpeg import Cancelled                    # noqa: E402

pytestmark = pytest.mark.gui


def _state(proj) -> AppState:
    return AppState(proj)


def test_edit_signals_selection_prune_undo(qapp, proj):
    st = _state(proj)
    seq = st.sequence()
    events = []
    st.edited.connect(lambda k, l: events.append((k, l)))
    c = st.op("放置", "place_clip", "a1", seq.video_tracks[0].id, 0.0)
    assert events == [("do", "放置")]
    assert st.dirty
    st.select([c.id])
    assert st.selection == {c.id}
    st.undo()
    assert events[-1][0] == "undo"
    assert st.selection == set()                 # selected clip vanished with the undo
    st.redo()
    assert [cl.id for _, cl in st.sequence().all_clips()] == [c.id]


def test_op_errors_go_to_status_and_reraise(qapp, proj):
    st = _state(proj)
    msgs = []
    st.status.connect(msgs.append)
    with pytest.raises(ops.OpError):
        st.op("壞操作", "split_clip", "nope", 1.0)
    assert msgs and msgs[-1]


def test_playhead_snaps_to_frame_grid(qapp, proj):
    st = _state(proj)
    got = []
    st.playhead_changed.connect(got.append)
    st.set_playhead(1.0123)
    assert abs(st.playhead * st.fps - round(st.playhead * st.fps)) < 1e-6
    assert got and got[-1] == st.playhead
    st.set_playhead(-5)
    assert st.playhead == 0.0


def test_save_open_dirty_title(qapp, proj, tmp_path):
    st = _state(proj)
    flips = []
    st.dirty_changed.connect(flips.append)
    st.op("放置", "place_clip", "a1", st.sequence().video_tracks[0].id, 0.0)
    assert flips == [True] and st.title.endswith("*")
    path = st.save_project(str(tmp_path / "x.acproj"))
    assert flips == [True, False] and not st.title.endswith("*")
    st2 = AppState()
    st2.open_project(path)
    assert [c.src_out for _, c in st2.sequence().all_clips()] == [20.0]
    assert not st2.dirty


def test_sequence_switch_resets_selection(qapp, proj):
    st = _state(proj)
    st.op("放置", "place_clip", "a1", st.sequence().video_tracks[0].id, 0.0)
    new = st.project_op("新增序列", "create_sequence", "第二個")
    st.select([next(c.id for _, c in st.sequence().all_clips())])
    st.set_sequence(new.id)
    assert st.selection == set() and st.seq_id == new.id


def test_human_review_only_by_explicit_call(qapp, proj):
    st = _state(proj)
    seq = st.sequence()
    plan = EditPlan(kind="shorts", sequences=[seq.to_dict()])
    receipt = AuditReceipt(plan.sha256(), st.project.revision, st.project.content_sha256(), True)

    def add(p: Project):
        run = record_run(p, plan, receipt, [seq.id])
        p.sequence(seq.id).meta["ai"] = {"run_id": run["run_id"]}
        return run
    run = st.do("套用", add)
    assert run["human_review"] == "pending"
    box = HumanReviewBox(st)
    assert "尚未審片" in box.verdict_label.text()
    st.record_technical_qa(run["run_id"], "pass", "ok")
    box.refresh()
    assert "尚未審片" in box.verdict_label.text()         # machine QA never flips it
    box.btn_ok.click()
    assert st.run_for_sequence()["human_review"] == "approved"
    assert "已通過" in box.verdict_label.text()


def test_timecode_edit_commit(qapp):
    e = TimecodeEdit(fps=30)
    got = []
    e.timeCommitted.connect(got.append)
    e.set_time(1.0)
    e.setText("00:00:02:15")
    e.editingFinished.emit()
    assert got and abs(got[-1] - 2.5) < 1e-6
    e.setText("垃圾")
    e.editingFinished.emit()
    assert len(got) == 1 and e.text() == "00:00:02:15"


def test_task_runner_progress_result_error_cancel(qapp):
    res, prog, errs, canc = [], [], [], []

    def ok(progress, cancel):
        progress(0.5, "一半")
        return 42
    r = TaskRunner(ok)
    r.finished.connect(res.append)
    r.progress.connect(lambda f, m: prog.append((f, m)))
    r.start()
    wait_for_idle(qapp, [r])
    assert res == [42] and prog == [(0.5, "一半")]

    def bad(progress, cancel):
        raise ValueError("壞掉了")
    r2 = TaskRunner(bad)
    r2.failed.connect(errs.append)
    r2.start()
    wait_for_idle(qapp, [r2])
    assert errs == ["壞掉了"]

    def slow(progress, cancel):
        while not cancel.is_set():
            time.sleep(0.01)
        raise Cancelled()
    r3 = TaskRunner(slow)
    r3.cancelled.connect(lambda: canc.append(1))
    r3.start()
    time.sleep(0.05)
    r3.cancel()
    wait_for_idle(qapp, [r3])
    assert canc == [1]


def test_shutdown_cancels_everything_that_runs_and_waits_for_it(qapp):
    started, stopped = threading.Event(), []

    def slow(tag):
        def fn(progress, cancel):
            started.set()
            while not cancel.is_set():
                time.sleep(0.005)
            stopped.append(tag)
            raise Cancelled()
        return fn

    rs = [TaskRunner(slow(i)).start() for i in range(2)]
    assert started.wait(5)
    time.sleep(0.05)
    assert set(workers.active_runners()) >= set(rs)
    t0 = time.monotonic()
    assert workers.shutdown(5.0) is True
    assert time.monotonic() - t0 < 2.0 and sorted(stopped) == [0, 1] and not any(r.running for r in rs)
    wait_for_idle(qapp, rs)
    assert workers.shutdown(1.0) is True                         # nothing left: returns at once


def test_shutdown_reports_a_task_that_will_not_stop(qapp):
    release = threading.Event()
    r = TaskRunner(lambda progress, cancel: release.wait(10)).start()          # ignores the cancel event
    time.sleep(0.05)
    try:
        assert workers.shutdown(0.2) is False                    # it says so, instead of hanging
        assert r.cancel_event.is_set()
    finally:
        release.set()
    wait_for_idle(qapp, [r])


def test_a_runner_cancelled_before_it_got_a_thread_never_runs(qapp):
    from PySide6.QtCore import QThreadPool
    pool = QThreadPool.globalInstance()
    old = pool.maxThreadCount()
    pool.setMaxThreadCount(1)
    ran, canc = [], []
    gate = threading.Event()
    try:
        first = TaskRunner(lambda progress, cancel: gate.wait(5)).start()          # occupies the only thread
        second = TaskRunner(lambda progress, cancel: ran.append(1))
        second.cancelled.connect(lambda: canc.append(1))
        second.start()                                           # queued behind the first
        second.cancel()
        gate.set()
        wait_for_idle(qapp, [first, second])
    finally:
        gate.set()
        pool.setMaxThreadCount(old)
    assert ran == [] and canc == [1]


def test_save_as_names_the_project_after_the_file_and_undo_keeps_that_name(qapp, proj, tmp_path):
    """Save As writes the project name and the sequence to reopen on through the Editor; neither is an edit, and an undo
    back past the save must not turn the name back into the old one."""
    st = _state(proj)
    first = st.project.sequences[0].id
    second = st.project_op("新增序列", "create_sequence", "第二個", activate=False).id
    st.op("放置", "place_clip", "a1", st.sequence().video_tracks[0].id, 0.0)
    st.set_sequence(second)
    depth = len(st.editor.history._undo)
    path = st.save_project(str(tmp_path / "我的 短片.acproj"))
    assert st.project.name == "我的 短片" and st.project.active == second
    assert len(st.editor.history._undo) == depth and not st.dirty            # not an edit: no undo step, not dirty
    assert Project.load(path).name == "我的 短片" and Project.load(path).active == second
    assert st.editor.undo() is not None                                      # the clip placement
    assert st.project.name == "我的 短片", "undo must not bring the old name back"
    assert st.editor.redo() is not None and st.project.name == "我的 短片"
    assert first != second


def test_a_failed_save_gives_the_old_name_back(qapp, proj, tmp_path, monkeypatch):
    st = _state(proj)
    old = st.project.name

    def boom(self, path=None):
        raise OSError("disk full")

    monkeypatch.setattr(Project, "save", boom)
    with pytest.raises(OSError):
        st.save_project(str(tmp_path / "x.acproj"))
    assert st.project.name == old
