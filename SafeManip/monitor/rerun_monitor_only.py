"""Re-run just the monitor step (run_monitor_on_privileged.py's logic) over every
already-extracted privileged_information_<N>.json under an output_root, overwriting
the matching _monitor.json in place.

Use this after a specs.py/predicates.py change that only affects monitor-time LTL
evaluation (not the raw sim extraction itself) -- much cheaper than re-running the
full extraction, and guarantees every episode's _monitor.json reflects the exact
same code state (no risk of a slow-running batch job straddling an in-flight edit,
as happened with v9's fridge-fix job overlapping the grasp-sync specs.py fix).

Also supports a single (--task, --episode) pair, mirroring
extract_privileged_from_dataset.py's --task/--episode mode, so this can be driven
by the same per-episode SLURM array pipeline
(run_extract_privileged_from_dataset_per_episode.sbatch / submit_extract_privileged_
per_episode.sh) instead of running serially in one process -- confirmed 2026-09-03
after a plain serial pass over 500 episodes took 2.5+ hours for only 320/500 (each
call loads a full 100-170MB privileged_information_<N>.json). Rerunning just the
monitor step is cheap per-episode (no simulation) but not cheap in aggregate when
serialized -- exactly the kind of embarrassingly-parallel batch the existing
per-episode array infra was built for.

Usage:
    python3 rerun_monitor_only.py --output_root <vN dir> [--tasks T1 T2 ...]
    python3 rerun_monitor_only.py --output_root <vN dir> --task T --episode N
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from monitor.run_monitor_on_privileged import monitor_rollout  # noqa: E402


def _rerun_one(priv_path: str) -> None:
    out_path = priv_path.replace(".json", "_monitor.json")
    summary = monitor_rollout(priv_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--task", default=None, help="Single task name (use with --episode)")
    ap.add_argument("--episode", type=int, default=None, help="Single episode index (use with --task)")
    args = ap.parse_args()

    root = Path(args.output_root)

    if args.task is not None and args.episode is not None:
        priv_path = str(root / args.task / f"privileged_information_{args.episode}.json")
        if not Path(priv_path).exists():
            print(f"[{args.task}] episode {args.episode}: no privileged_information file at {priv_path}, skipping")
            return
        _rerun_one(priv_path)
        print(f"[{args.task}] episode {args.episode}: re-monitored")
        return

    task_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.tasks:
        task_dirs = [p for p in task_dirs if p.name in args.tasks]

    total = 0
    for task_dir in task_dirs:
        priv_files = sorted(glob.glob(str(task_dir / "privileged_information_*.json")))
        priv_files = [f for f in priv_files if not f.endswith("_monitor.json")]
        for priv_path in priv_files:
            _rerun_one(priv_path)
            total += 1
        print(f"[{task_dir.name}] {len(priv_files)} episode(s) re-monitored")
    print(f"done: {total} episodes re-monitored")


if __name__ == "__main__":
    main()
