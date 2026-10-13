"""Developer tool: long random-test campaigns for the editing core and for the render oracle.

The test suite runs a few hundred random seeds per test, which is fast enough for every run.  This script runs THOUSANDS of
further seeds (it starts after the seeds the suite uses) - the way the stragglers of the review rounds were found - and
names the seeds that fail, so that one of them can be replayed on its own.  Nothing in here is part of the app.

    python scripts/dev/campaign.py glue   [--seeds 3000] [--first 150] [--steps 14] [--exact]
    python scripts/dev/campaign.py fuzz   base|wide [--seeds 3000] [--first N]
    python scripts/dev/campaign.py oracle [--seeds 150] [--first 6] [--fps 30]     (needs ffmpeg; about 1.5 s per seed)
    python scripts/dev/campaign.py alpha  [--seeds 12] [--first 0]                 (needs ffmpeg)
    python scripts/dev/campaign.py coverage                                        (needs ffmpeg; runs the oracle's own tests)

glue      the captions stay glued to their footage through random ripple edits (tests/test_core_sync.py);
          --exact switches the frame snapping off, so that they must then land exactly.
fuzz      random edit sequences keep the timeline valid and one undo gives back exactly one edit (tests/test_core_fuzz.py).
oracle    random edits, really rendered; every frame is decoded and compared with the timeline (tests/test_render_oracle.py).
alpha     how far the crossfade weight the renderer really used is from the weight the timeline model names.
coverage  what the oracle's own tests looked at (frames with 1 / 2 layers, sound windows) against the least they must see.

Every command takes --traceback (print the traceback of the first failures; replay one seed with
"--first 606 --seeds 1 --traceback").  The exit code is 1 when a seed failed (or the oracle looked at too little), else 0.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]          # the repo's own code first, wherever this is run from
os.environ.setdefault("AUTOCUT_HOME", tempfile.mkdtemp(prefix="autocut_campaign_"))       # never touch the real app data
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Seeds the test suite itself uses (a campaign starts after them).  test_core_fuzz exports its ranges; the glue tests have
# ``range(150)`` written into their decorators.
GLUE_SUITE_SEEDS = 150
ORACLE_SUITE_SEEDS = 6


def _message(exc: BaseException) -> str:
    """The whole message of a failure, headed by its type (the details of an oracle failure are on the later lines)."""
    return f"{type(exc).__name__}: {str(exc).strip()}".rstrip(": ")


def _line(exc: BaseException) -> str:
    return _message(exc).splitlines()[0][:200]


def run_seeds(one, first: int, count: int, show_traceback: bool) -> tuple[list[tuple[int, str]], int, int, float]:
    """Call ``one(seed)`` for ``count`` seeds from ``first``.  ``one`` returns how many things it checked (or None).
    Returns (failures, skipped, checked, seconds).  A skipped seed (pytest.skip inside the scenario) is not a failure."""
    import pytest
    skipped_type = pytest.skip.Exception
    started = time.perf_counter()
    failures: list[tuple[int, str]] = []
    skipped = checked = 0
    done = 0
    try:
        for done, seed in enumerate(range(first, first + count), 1):
            try:
                checked += one(seed) or 0
            except skipped_type:
                skipped += 1
            except Exception as exc:                                                    # a failing seed, not a crash
                failures.append((seed, _message(exc)))
                if show_traceback and len(failures) <= 3:
                    print(f"--- seed {seed}", file=sys.stderr)
                    traceback.print_exc()
            if done % 500 == 0:
                print(f"  ... {done}/{count} seeds, {len(failures)} failed, {time.perf_counter() - started:.0f} s",
                      file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print(f"interrupted after {done} of {count} seeds", file=sys.stderr)
    return failures, skipped, checked, time.perf_counter() - started


def report(title: str, first: int, count: int, result, unit: str = "") -> int:
    failures, skipped, checked, seconds = result
    what = f", {checked} {unit} checked" if unit else ""
    print(f"{title}: seeds {first}..{first + count - 1}{what}, failed {len(failures)}, skipped {skipped}, {seconds:.1f} s")
    for n, (seed, msg) in enumerate(failures[:20]):
        lines = msg.splitlines()
        if n < 3 and len(lines) > 1:                                     # the first failures in full (up to 16 lines)
            print(f"  seed {seed}:")
            print("\n".join("      " + ln[:240] for ln in lines[:16]))
            if len(lines) > 16:
                print(f"      ... {len(lines) - 16} more lines")
        else:
            print(f"  seed {seed}: {lines[0][:200] if lines else ''}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    return 1 if failures else 0


# ------------------------------------------------------------------------------------------------ the commands
def cmd_glue(a) -> int:
    import test_core_sync as T
    from autocut.core import ops
    if a.exact:
        ops._sn = lambda seq, t: max(0.0, float(t))                     # frame snapping off, as the exact tests do
    result = run_seeds(lambda seed: T.run_glue_seed(seed, steps=a.steps, exact=a.exact), a.first, a.seeds, a.traceback)
    return report("glue (" + ("frame snapping off" if a.exact else "frame snapping on") + ")", a.first, a.seeds, result, "captions")


def cmd_fuzz(a) -> int:
    import test_core_fuzz as F
    fn, suite = {"base": (F.test_random_edits_keep_the_timeline_valid_and_ripples_never_leave_holes, F.SEEDS.stop),
                 "wide": (F.test_wider_random_edits_keep_everything_valid_and_one_undo_gives_back_exactly_one_edit,
                          F.WIDE_SEEDS.stop)}[a.which]
    first = suite if a.first is None else a.first
    return report(f"fuzz {a.which}", first, a.seeds, run_seeds(fn, first, a.seeds, a.traceback))


def _oracle_sources():
    import test_render_oracle as T
    work = Path(tempfile.mkdtemp(prefix="autocut_oracle_"))
    sources = T.make_sources(work)
    return T, work, sources


def cmd_oracle(a) -> int:
    T, work, sources = _oracle_sources()
    try:
        def one(seed: int) -> int:
            T.check_scenario(seed, sources, work, a.fps)
            return 0
        before = dict(T.COVERAGE)
        result = run_seeds(one, a.first, a.seeds, a.traceback)
        code = report(f"oracle ({a.fps:g} fps)", a.first, a.seeds, result)
        looked = {k: v - before.get(k, 0) for k, v in T.COVERAGE.items() if v - before.get(k, 0)}
        print("  looked at: " + ", ".join(f"{k}: {v}" for k, v in sorted(looked.items(), key=lambda kv: str(kv[0]))))
        return code
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_alpha(a) -> int:
    import numpy as np
    T, work, sources = _oracle_sources()
    index = {s.id: i for i, s in enumerate(sources)}
    rows = []
    try:
        for seed in range(a.first, a.first + a.seeds):
            try:
                _seq, items, blocks, _grey, _audio, _where = T.render_scenario(seed, sources, work)
            except BaseException as exc:                                                # skip or a failed scenario
                print(f"seed {seed} skipped: {_line(exc)}")
                continue
            edges = sorted({b for _c, cs, ce in items for b in (cs, ce)})
            n = 0
            for f in range(min(max(ce for _c, _cs, ce in items), len(blocks))):
                layers = [it for it in items if it[1] <= f < it[2]]
                if len(layers) != 2 or any(abs(f - e) <= T.SKIP for e in edges):
                    continue
                a_hat, a_model, err = T.blend_estimate(blocks[f], layers, f, T.FPS, index)
                if a_hat is not None:
                    rows.append((a_model, a_hat - a_model, err))
                    n += 1
            print(f"seed {seed}: {n} crossfade frames")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not rows:
        print("alpha: no crossfade frame could be read")
        return 1
    m = np.array(rows)
    diff = m[:, 1]
    print(f"alpha: {len(m)} frames; renderer minus model: mean {diff.mean():+.3f}, median {np.median(diff):+.3f}, "
          f"p90 |err| {np.percentile(np.abs(diff), 90):.3f}, max |err| {np.abs(diff).max():.3f}, "
          f"worst block fit {m[:, 2].max():.3f}  (the oracle's tolerance: {T.ALPHA_TOL} / {T.FIT_TOL})")
    edges = np.linspace(0, 1, 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (m[:, 0] >= lo) & (m[:, 0] < hi + 1e-9)
        if sel.any():
            print(f"  model weight {lo:.1f}-{hi:.1f}: n={int(sel.sum())}  mean {diff[sel].mean():+.3f}  max |err| {np.abs(diff[sel]).max():.3f}")
    return 0 if np.abs(diff).max() <= T.ALPHA_TOL else 1


def cmd_coverage(a) -> int:
    import pytest
    os.chdir(ROOT)
    rc = int(pytest.main([str(ROOT / "tests" / "test_render_oracle.py"), "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                          "--tb=short"]))
    mod = sys.modules.get("test_render_oracle")
    if mod is None:
        print("coverage: the oracle module was not loaded (is ffmpeg installed?)")
        return 1
    print("\nwhat the oracle's tests looked at, against the least they must see:")
    bad = False
    for key in sorted(set(mod.COVERAGE) | set(mod.COVERAGE_WANTED), key=str):
        got, want = mod.COVERAGE[key], mod.COVERAGE_WANTED.get(key)
        note = "" if want is None else f"   wanted >= {want}  (x{got / want:.1f})"
        bad = bad or (want is not None and got < want)
        print(f"  {str(key):34} {got:7d}{note}")
    return 1 if (rc or bad) else 0


# ------------------------------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="campaign.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, seeds: int, first: int | None, help_: str):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        if seeds:
            sp.add_argument("--seeds", type=int, default=seeds, help=f"how many seeds (default {seeds})")
            sp.add_argument("--first", type=int, default=first, help="the first seed" + (f" (default {first}: after the suite's)" if first is not None else ""))
            sp.add_argument("--traceback", action="store_true", help="print the traceback of the first failures")
        return sp

    g = add("glue", cmd_glue, 3000, GLUE_SUITE_SEEDS, "captions stay glued to their footage through random ripple edits")
    g.add_argument("--steps", type=int, default=14, help="edits per seed (default 14)")
    g.add_argument("--exact", action="store_true", help="frame snapping off: the captions must land exactly")
    f = add("fuzz", cmd_fuzz, 3000, None, "random edit sequences keep the timeline valid; one undo = one edit")
    f.add_argument("which", choices=("base", "wide"), help="base: structure only; wide: also captions, markers, undo, more kinds of edit")
    o = add("oracle", cmd_oracle, 150, ORACLE_SUITE_SEEDS, "render random edits for real and compare every frame (needs ffmpeg)")
    o.add_argument("--fps", type=float, default=30.0, help="sequence frame rate (default 30)")
    add("alpha", cmd_alpha, 12, 0, "the crossfade weight the renderer used against the model's (needs ffmpeg)")
    add("coverage", cmd_coverage, 0, None, "what the oracle's own tests looked at (needs ffmpeg)")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
