import collections
import dataclasses
import json
from pathlib import Path
from typing import Literal

import tyro

from robocasa.utils.dataset_registry import get_ds_soup


@dataclasses.dataclass
class Args:
    """
    Compute simple stats for a RoboCasa dataset soup used by OpenPI.

    This is useful for understanding how many demos / timesteps are in a given
    task soup (e.g. `posttrain50`, `posttrain_atomic_seen`, etc.) without
    having to materialize the full datasets.
    """

    # Split used when constructing the soup. Must match dataset_registry.
    split: Literal["train", "test", "real"] = "test"

    # Task soup identifier, passed through to `get_ds_soup(task_type=...)`.
    # Examples: "posttrain50", "posttrain_atomic_seen",
    # "posttrain_composite_seen", "posttrain_composite_unseen", "atomic",
    # "composite", "all", or any single RoboCasa task name.
    task_type: str = "posttrain50"

    # Source type used when constructing the soup. Common choice for OpenPI is
    # "human".
    source_type: Literal[
        "human",
        "human_cotraining_cams",
        "human_cotraining_cams_norand",
        "mg",
        "mg_5x5",
        "mg_5x1",
        "all",
    ] = "human"

    # Optional fraction of demos per dataset (same semantics as in
    # `get_ds_soup`). Kept as a float instead of string "10p"/"30p" for
    # flexibility.
    demo_fraction: float = 1.0

    # Optional checkpoint directory. If provided, we will look for
    #   <checkpoint_dir>/evals/<split>/<Task>/<run_date>/stats.json
    # and attach eval success stats per task using the most recent run_date.
    checkpoint_dir: str | None = None


def main(args: Args) -> None:
    soup = get_ds_soup(
        split=args.split,
        task_type=args.task_type,
        source_type=args.source_type,
        demo_fraction=args.demo_fraction,
    )

    if not soup:
        print("No datasets found for the given configuration.")
        return

    print(
        f"Dataset soup stats for split='{args.split}', "
        f"task_type='{args.task_type}', source_type='{args.source_type}', "
        f"demo_fraction={args.demo_fraction}"
    )
    print("=" * 80)

    total_datasets = len(soup)
    total_demos = 0
    total_steps = 0

    # Group per task for nicer reporting
    per_task = collections.defaultdict(
        lambda: {
            "num_datasets": 0,
            "num_demos": 0,
            "approx_steps": 0,
            "eval_num_episodes": 0,
            "eval_success_rate": None,
        }
    )

    for meta in soup:
        task = meta["task"]
        horizon = int(meta["horizon"])
        # meta["filter_key"] looks like "100_demos" – extract the integer prefix.
        try:
            num_demos = int(str(meta["filter_key"]).split("_")[0])
        except (KeyError, ValueError, TypeError):
            num_demos = 0

        approx_steps = num_demos * horizon

        total_demos += num_demos
        total_steps += approx_steps

        per_task[task]["num_datasets"] += 1
        per_task[task]["num_demos"] += num_demos
        per_task[task]["approx_steps"] += approx_steps

    # If a checkpoint directory is provided, attempt to attach eval stats per task.
    total_eval_episodes = 0
    total_eval_successes = 0.0
    if args.checkpoint_dir is not None:
        eval_root = Path(args.checkpoint_dir) / "evals" / args.split
        for task in per_task.keys():
            task_dir = eval_root / task
            if not task_dir.exists():
                continue
            # Each task directory contains run-date subdirectories.
            run_dirs = [p for p in task_dir.iterdir() if p.is_dir()]
            if not run_dirs:
                continue
            # Use the most recent run directory (names are date strings).
            latest_run = sorted(run_dirs)[-1]
            stats_path = latest_run / "stats.json"
            if not stats_path.exists():
                continue
            try:
                with stats_path.open("r") as f:
                    stats_json = json.load(f)
            except Exception:
                continue

            num_episodes = int(stats_json.get("num_episodes", 0))
            success_rate = float(stats_json.get("success_rate", 0.0))

            per_task[task]["eval_num_episodes"] = num_episodes
            per_task[task]["eval_success_rate"] = success_rate

            total_eval_episodes += num_episodes
            total_eval_successes += success_rate * num_episodes

    # Per-task table: only show evaluation statistics.
    if args.checkpoint_dir is None:
        print("No checkpoint_dir provided; only dataset metadata was computed.")
        return

    print(f"{'Task':40s} {'Eval Episodes':>14s} {'Eval SR':>9s}")
    print("-" * 80)

    for task, stats in sorted(per_task.items()):
        eval_eps = stats["eval_num_episodes"]
        eval_sr = stats["eval_success_rate"]
        if eval_sr is None or eval_eps == 0:
            line = f"{task:40s} {0:14d} {'-':>9s}"
        else:
            line = f"{task:40s} {eval_eps:14d} {eval_sr:9.3f}"
        print(line)

    print("-" * 80)
    if total_eval_episodes == 0:
        print(f"{'TOTAL':40s} {0:14d} {'-':>9s}")
    else:
        overall_sr = total_eval_successes / total_eval_episodes
        print(f"{'TOTAL':40s} {total_eval_episodes:14d} {overall_sr:9.3f}")


if __name__ == "__main__":
    # Use top-level flags (e.g., --split, --task_type, --checkpoint_dir)
    # instead of the nested --args.* style.
    main(tyro.cli(Args))


