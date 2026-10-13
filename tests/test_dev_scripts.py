"""The developer scripts must keep working: a campaign tool nobody runs is the first thing to rot (a renamed test function,
a moved module...).  Each command runs a few seeds from another directory, the way a person would."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "campaign.py"


def _run(tmp_path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "AUTOCUT_HOME": str(tmp_path / "home"), "QT_QPA_PLATFORM": "offscreen"}
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=str(tmp_path), env=env, capture_output=True, text=True,
                          timeout=240)


def _load():
    spec = importlib.util.spec_from_file_location("campaign_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_failing_seed_is_named_and_makes_the_exit_code_1(capsys):
    mod = _load()

    def one(seed):
        if seed == 3:
            raise AssertionError("boom\nthe second line is not shown")
        if seed == 4:
            pytest.skip("the scenario was empty")
        return 2

    result = mod.run_seeds(one, 0, 6, False)
    failures, skipped, checked, _seconds = result
    # the whole message is kept (the details of an oracle failure are on its later lines), headed by the exception type
    assert failures == [(3, "AssertionError: boom\nthe second line is not shown")] and skipped == 1 and checked == 8
    assert mod.report("t", 0, 6, result, "things") == 1
    out = capsys.readouterr().out
    assert "failed 1" in out and "seed 3:" in out and "AssertionError: boom" in out and "8 things checked" in out
    assert "the second line is not shown" in out                       # the first failures are printed in full
    assert mod.report("t", 0, 3, mod.run_seeds(lambda s: 1, 0, 3, False)) == 0


def test_only_the_first_three_failures_are_printed_in_full(capsys):
    mod = _load()

    def always_bad(seed):
        raise AssertionError(f"bad {seed}\ndetail {seed}")

    assert mod.report("m", 0, 5, mod.run_seeds(always_bad, 0, 5, False)) == 1
    out = capsys.readouterr().out
    assert "failed 5" in out
    assert all(f"seed {seed}:" in out and f"bad {seed}" in out for seed in range(5))          # every failing seed is named
    assert all(f"detail {seed}" in out for seed in range(3))                                   # ... the first three with details
    assert not any(f"detail {seed}" in out for seed in (3, 4))                                 # ... the rest in one line


@pytest.mark.parametrize("args, expect", [
    (("glue", "--seeds", "3"), "frame snapping on"),
    (("glue", "--seeds", "3", "--exact"), "frame snapping off"),
    (("fuzz", "base", "--seeds", "2"), "fuzz base"),
    (("fuzz", "wide", "--seeds", "2"), "fuzz wide"),
])
def test_the_core_campaigns_run_from_any_directory(tmp_path, args, expect):
    done = _run(tmp_path, *args)
    assert done.returncode == 0, done.stdout + done.stderr
    assert expect in done.stdout and "failed 0" in done.stdout, done.stdout


@pytest.mark.ffmpeg
@pytest.mark.parametrize("args, expect", [
    (("oracle", "--seeds", "1", "--first", "0"), "looked at"),
    (("alpha", "--seeds", "2"), "renderer minus model"),
])
def test_the_render_campaigns_run_from_any_directory(tmp_path, args, expect):
    done = _run(tmp_path, *args)
    assert done.returncode == 0, done.stdout + done.stderr
    assert expect in done.stdout, done.stdout
