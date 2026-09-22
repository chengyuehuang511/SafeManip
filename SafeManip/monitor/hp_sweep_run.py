"""Run hyperparameter sweep cells: re-monitor the episode subset under one set of
SAFEMANIP_HP_* overrides per cell, each cell into its own output tree.

The source corpus is NEVER touched -- everything goes to
hp_sweep/cells/<cell>/<suite>/<task>/privileged_information_<N>_monitor.json,
alongside a manifest.json recording the git commit and the exact overrides that
produced it. That separation is the whole point: the committed output/vN is the
paper's configuration and a sweep must not be able to overwrite it.

ONE OS PROCESS PER EPISODE, dispatched by a thread pool that only waits on
subprocess.run. The first version used multiprocessing.Pool and deadlocked --
workers died mid-episode without a Python traceback (the pipeline shells out to
MONA via ltlf2dfa, so a native abort kills the worker outright), Pool respawned
them, and imap_unordered waited forever for the orphaned results. Here a death is
a non-zero exit code on one episode: logged, counted, skipped. Each episode also
gets a hard --timeout so one pathological episode cannot stall a cell.

RESUMABLE by default: an episode whose output file already exists is skipped, so
a killed cell can be re-run without repeating work. --no-resume forces a redo.

Usage:
    python3 hp_sweep_run.py --list
    python3 hp_sweep_run.py --cell default --workers 20
    python3 hp_sweep_run.py --all --workers 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from monitor.hp_sweep_grid import (CORPORA, DEFAULT_CELL, SWEEP_DIR,  # noqa: E402
                                   cells, load_subset)

CELLS_DIR = SWEEP_DIR / "cells"
ONE = THIS_DIR / "hp_sweep_one.py"


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- a missing git must not kill the run
        return None


# Peak RSS / input-file size, measured on the largest episode in the corpus
# (PackIdenticalLunches ep6: 328 MB of JSON -> 3.0 GB peak, 3m10s). Used to keep
# total in-flight memory bounded: episode sizes in the RoboCasa subset span
# 16-556 MB, so a fixed worker count schedules either 16 tiny episodes (wasting
# cores) or 16 huge ones (the first attempt at this sweep lost episodes to
# SIGKILL). Deliberately generous.
RSS_PER_BYTE = 12.0
# Bytes of estimated peak RSS allowed in flight at once. MUST be set to fit the
# cgroup the run actually lives in, which is not the machine's free memory: an
# interactive login session here is capped at 15 GB by
# /sys/fs/cgroup/memory/user.slice/user-<uid>.slice/memory.limit_in_bytes, and
# exceeding it SIGKILLs individual episodes (exit -9, no traceback, trivially
# mistaken for a monitor crash) while `free` still shows hundreds of free GB.
# Under SLURM the cgroup is --mem, so the sbatch sets this from it.
MEM_BUDGET = int(os.environ.get("HP_SWEEP_MEM_BUDGET_GB", "10")) << 30


def jobs_for(cell, subset):
    out = []
    for suite in cell["suites"]:
        root = Path(CORPORA[suite])
        for task, eps in sorted(subset[suite].items()):
            for ep in eps:
                priv = root / task / f"privileged_information_{ep}.json"
                out.append({
                    "suite": suite, "task": task, "episode": ep,
                    "priv": str(priv),
                    "size": priv.stat().st_size if priv.is_file() else 0,
                    "dest": str(CELLS_DIR / cell["name"] / suite / task /
                                f"privileged_information_{ep}_monitor.json"),
                })
    return out


class MemGate:
    """A semaphore denominated in estimated bytes rather than in slots.

    A job larger than the whole budget would deadlock waiting for room that can
    never exist, so it is clamped to the full budget and simply runs alone.
    """

    def __init__(self, budget):
        self.budget = budget
        self.free = budget
        self.cv = threading.Condition()

    def acquire(self, n):
        n = min(n, self.budget)
        with self.cv:
            while self.free < n:
                self.cv.wait()
            self.free -= n
        return n

    def release(self, n):
        with self.cv:
            self.free += n
            self.cv.notify_all()


def run_one(job, env, timeout, logdir, gate):
    if Path(job["dest"]).is_file():
        return job, "cached", 0.0
    held = gate.acquire(int(job["size"] * RSS_PER_BYTE))
    try:
        return _run_one_inner(job, env, timeout, logdir)
    finally:
        gate.release(held)


def _run_one_inner(job, env, timeout, logdir):
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, str(ONE), "--priv", job["priv"],
             "--dest", job["dest"]],
            env=env, capture_output=True, text=True, timeout=timeout)
        status = "ok" if p.returncode == 0 else f"exit{p.returncode}"
    except subprocess.TimeoutExpired:
        return job, "timeout", time.time() - t0
    if status != "ok":
        # Keep the child's stderr: an exit code alone cannot distinguish a
        # genuinely unmonitorable episode from a harness bug.
        log = logdir / f"{job['suite']}__{job['task']}__{job['episode']}.log"
        log.write_text((p.stdout or "") + "\n--- stderr ---\n" + (p.stderr or ""))
    return job, status, time.time() - t0


def run_cell(cell, subset, workers, timeout, resume, limit=None):
    jobs = jobs_for(cell, subset)
    if limit:
        # Smoke test only. Strided so a --limit run still touches both suites:
        # the first N jobs in order are all RoboCasa.
        jobs = jobs[::max(1, len(jobs) // limit)][:limit]
    cdir = CELLS_DIR / cell["name"]
    logdir = cdir / "_failed"
    logdir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for j in jobs:
            Path(j["dest"]).unlink(missing_ok=True)

    env = dict(os.environ)
    for k, v in cell["env"].items():
        env[f"SAFEMANIP_HP_{k}"] = v

    (cdir / "manifest.json").write_text(json.dumps({
        "cell": cell["name"], "knob": cell["knob"], "value": cell["value"],
        "suites": list(cell["suites"]),
        "hp_overrides": {k: v for k, v in env.items()
                         if k.startswith("SAFEMANIP_HP_")},
        "n_episodes": len(jobs),
        "corpora": {s: Path(CORPORA[s]).name for s in cell["suites"]},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }, indent=2, sort_keys=True))

    # Largest first: the long tail of 300-550 MB RoboCasa episodes decides the
    # wall clock, and starting them last leaves one 3-minute job running alone
    # after everything else has drained.
    jobs.sort(key=lambda j: -j["size"])
    gate = MemGate(MEM_BUDGET)
    t0, done, bad = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_one, j, env, timeout, logdir, gate) for j in jobs]
        for fut in as_completed(futs):
            job, status, secs = fut.result()
            done += 1
            if status not in ("ok", "cached"):
                bad.append((job, status))
                print(f"  [{status}] {job['suite']}/{job['task']}/"
                      f"{job['episode']} after {secs:.0f}s", flush=True)
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)}  {el/60:.1f} min  "
                      f"eta {el/done*(len(jobs)-done)/60:.1f} min", flush=True)
    (cdir / "failures.json").write_text(json.dumps(
        [{"status": s, **j} for j, s in bad], indent=2))
    print(f"[cell] {cell['name']}: {len(jobs)} episodes, {len(bad)} failed, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds per episode before it is abandoned")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: this many episodes, spread over both suites")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-monitor episodes whose output already exists")
    args = ap.parse_args()

    grid = cells()
    if args.list:
        for c in grid:
            print(f"{c['name']:45s} {','.join(c['suites'])}")
        return
    todo = grid if args.all else [c for c in grid if c["name"] == args.cell]
    if not todo:
        sys.exit(f"no such cell {args.cell!r} (use --list)")
    # The default cell is the reference every other cell is scored against, so it
    # runs first when --all: a sweep whose reference is missing scores nothing.
    todo.sort(key=lambda c: c["name"] != DEFAULT_CELL)

    subset = load_subset()
    for c in todo:
        print(f"\n=== cell {c['name']}  ({','.join(c['suites'])}) ===", flush=True)
        run_cell(c, subset, args.workers, args.timeout,
                 resume=not args.no_resume, limit=args.limit)


if __name__ == "__main__":
    main()
