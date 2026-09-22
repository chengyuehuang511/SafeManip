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


def _write_manifest(path, output_root, monitor_root) -> None:
    """Record what produced this output: the git commit, whether the tree was
    dirty, and every SAFEMANIP_HP_* override. Without this a sweep directory is
    just numbers -- there is no way to tell afterwards which cell it was, and a
    dirty tree means the commit hash alone does not identify the code."""
    import subprocess
    from monitor.hp_override import requested

    def _git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *a],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # noqa: BLE001 -- a missing git must not kill the run
            return None

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "hp_overrides": requested(),
            "output_root": str(output_root),
            "monitor_root": str(monitor_root) if monitor_root else None,
        }, f, indent=2, sort_keys=True)
    print(f"[manifest] {path}", flush=True)


def _rerun_one(priv_path: str, monitor_root=None, output_root=None) -> None:
    """Re-monitor one episode.

    Default (monitor_root=None) is the historical in-place overwrite. With
    --monitor_root the summary is written to <monitor_root>/<task>/<same name>
    and the source corpus is left untouched -- required by the hyperparameter
    sweep, which re-monitors ONE corpus under N configurations and must not
    destroy the committed vN output to do it.
    """
    name = Path(priv_path).name.replace(".json", "_monitor.json")
    if monitor_root is None:
        out_path = Path(priv_path).with_name(name)
    else:
        out_dir = Path(monitor_root) / Path(priv_path).parent.relative_to(output_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / name
    summary = monitor_rollout(priv_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--task", default=None, help="Single task name (use with --episode)")
    ap.add_argument("--episode", type=int, default=None, help="Single episode index (use with --task)")
    ap.add_argument("--monitor_root", default=None,
                    help="write _monitor.json here (mirroring <task>/ layout) "
                         "instead of overwriting the source corpus in place")
    ap.add_argument("--manifest", default=None,
                    help="write a JSON run manifest (git hash + SAFEMANIP_HP_* "
                         "overrides) next to the output; use with --monitor_root "
                         "so a sweep cell's provenance is recorded with it")
    args = ap.parse_args()

    root = Path(args.output_root)
    mroot = Path(args.monitor_root) if args.monitor_root else None
    if args.manifest:
        _write_manifest(args.manifest, root, mroot)

    if args.task is not None and args.episode is not None:
        priv_path = str(root / args.task / f"privileged_information_{args.episode}.json")
        if not Path(priv_path).exists():
            print(f"[{args.task}] episode {args.episode}: no privileged_information file at {priv_path}, skipping")
            return
        _rerun_one(priv_path, mroot, root)
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
            _rerun_one(priv_path, mroot, root)
            total += 1
        print(f"[{task_dir.name}] {len(priv_files)} episode(s) re-monitored")
    print(f"done: {total} episodes re-monitored")


if __name__ == "__main__":
    main()
