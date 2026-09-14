#!/usr/bin/env python3
"""Aggregates grootn16/RLDX-1 per-task stats.json into the same
Atomic-Seen/Composite-Seen/Composite-Unseen splits the robocasa-benchmark
leaderboard (https://github.com/robocasa-benchmark/leaderboard) reports,
and compares our reproduced numbers against each submission's reported
values (hardcoded below, transcribed verbatim from submissions_md/
gr00t_n1.6_2026_05_14.md and submissions_md/rldx-1_2026_05_20.md).

The atomic_seen/composite_seen/composite_unseen task groupings are the same
ones already used by eval_groot_single_task.sh's infer_task_set() (RoboCasa
1.0.1's own 18/16/16-task split), reused verbatim here rather than
re-derived, since the leaderboard's "Atomic-Seen"/"Composite-Seen"/
"Composite-Unseen" terminology is RoboCasa's own for the same three groups.
"""
import argparse
import json
from pathlib import Path

TASK_SETS = {
    "atomic_seen": [
        "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
        "NavigateKitchen", "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
        "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack",
        "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
    ],
    "composite_seen": [
        "DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher",
        "PackIdenticalLunches", "PreSoakPan", "PrepareCoffee", "RinseSinkBasin",
        "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation", "StackBowlsCabinet",
        "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce",
    ],
    "composite_unseen": [
        "ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CategorizeCondiments",
        "CuttingToolSelection", "GarnishPancake", "GatherTableware", "HeatKebabSandwich",
        "MakeIceLemonade", "PanTransfer", "PortionHotDogs", "RecycleBottlesByType",
        "SeparateFreezerRack", "WaffleReheat", "WashFruitColander", "WeighIngredients",
    ],
}

# Transcribed verbatim from robocasa-benchmark/leaderboard's submissions_md/*.md
# (see eval/models/grootn16, eval/models/RLDX-1 for the pinned code commits
# these numbers were reported against).
REPORTED = {
    "grootn16": {
        "display_name": "GR00T N1.6",
        "atomic_seen": 51.1,
        "composite_seen": 9.4,
        "composite_unseen": 1.7,
        "source": "submissions_md/gr00t_n1.6_2026_05_14.md",
    },
    "rldx1": {
        "display_name": "RLDX-1",
        "atomic_seen": 67.6,
        "composite_seen": 27.9,
        "composite_unseen": 8.5,
        "source": "submissions_md/rldx-1_2026_05_20.md",
    },
}


def collect_split(video_dir: Path, tasks) -> dict:
    per_task = {}
    missing = []
    total_episodes = 0
    total_successes = 0.0
    for task in tasks:
        stats_path = video_dir / task / "stats.json"
        if not stats_path.is_file():
            missing.append(task)
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        n = stats.get("num_episodes")
        sr = stats.get("success_rate")
        if n is None or sr is None:
            missing.append(task)
            continue
        per_task[task] = {"num_episodes": n, "success_rate": sr}
        total_episodes += n
        total_successes += sr * n
    overall = 100.0 * total_successes / total_episodes if total_episodes else None
    return {
        "num_tasks_found": len(per_task),
        "num_tasks_missing": len(missing),
        "missing_tasks": missing,
        "total_episodes": total_episodes,
        "success_pct": overall,
        "per_task": per_task,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--saved_eval_rollouts",
        default=str(Path(__file__).resolve().parents[1] / "saved_eval_rollouts"),
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.saved_eval_rollouts)

    print(
        f"{'model':12s} {'split':18s} {'reproduced':>11s} {'reported':>9s} "
        f"{'diff':>7s} {'found':>7s} {'episodes':>9s}"
    )
    results = {}
    for model_key, video_dir in [("grootn16", root / "grootn16"), ("rldx1", root / "rldx1")]:
        reported = REPORTED[model_key]
        model_result = {"reported": reported, "splits": {}}
        for split, tasks in TASK_SETS.items():
            r = collect_split(video_dir, tasks)
            model_result["splits"][split] = r
            repro_str = f"{r['success_pct']:.1f}" if r["success_pct"] is not None else "n/a"
            rep_val = reported[split]
            diff_str = (
                f"{r['success_pct'] - rep_val:+.1f}" if r["success_pct"] is not None else "n/a"
            )
            print(
                f"{model_key:12s} {split:18s} {repro_str:>11s} {rep_val:>9.1f} "
                f"{diff_str:>7s} {r['num_tasks_found']:>4d}/{len(tasks):<2d} "
                f"{r['total_episodes']:>9d}"
            )
            if r["missing_tasks"]:
                print(f"    missing: {', '.join(r['missing_tasks'])}")
        results[model_key] = model_result

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull breakdown written to {args.out}")


if __name__ == "__main__":
    main()
