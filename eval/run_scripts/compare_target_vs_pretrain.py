#!/usr/bin/env python3
"""Compares each model's success rate on the `target` split (held-out
kitchen scenes/objects) vs. the `pretrain` split (in-distribution scenes/
objects), for whichever tasks have completed on each side so far.

Only meaningful for models where both splits were actually run:
- groot/multitask_learning, openpi/pi0, openpi/pi0.5: target-split data is
  from the original (later-corrected-to-be-wrong-for-this-model) run,
  preserved under target/ rather than deleted; pretrain-split is the
  corrected rerun.
- grootn16, RLDX-1: target-split runs were cancelled before completing (no
  usable baseline) -- only pretrain-split numbers exist for these two.

See eval/EVAL_PROTOCOL_NOTES.md for why pretrain is the *correct* split for
all 5 of these models (RoboCasa's "Multitask Learning" benchmarking
setting) -- this script is for understanding the *magnitude* of the
difference between the two settings on the same checkpoints, not for
picking which one is "right" (pretrain already is).
"""
import argparse
import json
from pathlib import Path

MODEL_DIRS = {
    "groot/multitask_learning": "groot/multitask_learning",
    "openpi/pi0": "openpi/pi0/pretraining",
    "openpi/pi0.5": "openpi/pi0.5/pretraining",
    "grootn16": "grootn16",
    "rldx1": "rldx1",
}


def find_stats_json(task_dir: Path):
    """openpi nests stats.json under evals_1.5/<split>/<task>/<timestamp>/;
    groot/grootn16/rldx1 put it directly at <task_dir>/stats.json."""
    direct = task_dir / "stats.json"
    if direct.is_file():
        return direct
    evals_root = task_dir / "evals_1.5"
    if evals_root.is_dir():
        for split_dir in evals_root.iterdir():
            candidate_task_dir = split_dir / task_dir.name
            if candidate_task_dir.is_dir():
                run_dirs = sorted(candidate_task_dir.glob("*"))
                for run_dir in reversed(run_dirs):
                    sp = run_dir / "stats.json"
                    if sp.is_file():
                        return sp
    return None


def collect(root: Path, model_subdir: str) -> dict:
    model_root = root / model_subdir
    per_task = {}
    if not model_root.is_dir():
        return per_task
    for task_dir in sorted(model_root.iterdir()):
        if not task_dir.is_dir():
            continue
        sp = find_stats_json(task_dir)
        if sp is None:
            continue
        with open(sp) as f:
            stats = json.load(f)
        n = stats.get("num_episodes")
        sr = stats.get("success_rate")
        if n is not None and sr is not None:
            per_task[task_dir.name] = {"num_episodes": n, "success_rate": sr}
    return per_task


def weighted_mean(per_task: dict):
    total_ep = sum(v["num_episodes"] for v in per_task.values())
    if total_ep == 0:
        return None, 0
    total_succ = sum(v["success_rate"] * v["num_episodes"] for v in per_task.values())
    return total_succ / total_ep, total_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--saved_eval_rollouts",
        default=str(Path(__file__).resolve().parents[1] / "saved_eval_rollouts"),
    )
    args = ap.parse_args()
    root = Path(args.saved_eval_rollouts)

    print(
        f"{'model':24s} {'common_tasks':>13s} {'target_sr':>10s} {'pretrain_sr':>12s} {'diff':>8s}"
    )
    for label, subdir in MODEL_DIRS.items():
        target_tasks = collect(root / "target", subdir)
        pretrain_tasks = collect(root / "pretrain", subdir)
        # Only compare on the intersection of tasks completed on BOTH
        # splits -- comparing "all 50 target tasks" against "whatever
        # subset of pretrain happens to be done so far" would bias the
        # comparison toward whichever tasks the scheduler happened to run
        # first on the still-in-progress side.
        common = sorted(set(target_tasks) & set(pretrain_tasks))
        t_common = {k: target_tasks[k] for k in common}
        p_common = {k: pretrain_tasks[k] for k in common}
        t_sr, _ = weighted_mean(t_common)
        p_sr, _ = weighted_mean(p_common)
        t_sr_str = f"{t_sr:.4f}" if t_sr is not None else "n/a"
        p_sr_str = f"{p_sr:.4f}" if p_sr is not None else "n/a"
        diff_str = f"{p_sr - t_sr:+.4f}" if (t_sr is not None and p_sr is not None) else "n/a"
        print(f"{label:24s} {len(common):>10d}/50 {t_sr_str:>10s} {p_sr_str:>12s} {diff_str:>8s}")


if __name__ == "__main__":
    main()
