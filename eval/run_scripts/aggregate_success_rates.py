#!/usr/bin/env python3
"""Aggregates per-task stats.json success rates into one mean success rate
per model, across all 50 RoboCasa tasks.

Looks in two places (both written by run_single_task_groot.py /
run_single_task_openpi.py, never hand-rolled here):

  GR00T:  {video_dir}/{model_family}/{task}/stats.json
          -> {"task", "split", "num_episodes", "success_rate", ...}
  openpi: {log_dir}/{variant}/{family}/{task}/evals_1.5/{split}/{task}/{timestamp}/stats.json
          -> {"num_episodes", "success_rate"}
          (there can be more than one timestamp dir if a task was re-run;
          the latest one by name is used, matching run_single_task_openpi.py's
          own `sorted(...)[-1]` convention)

A model's aggregate success rate is the mean of per-task success_rate,
weighted by that task's num_episodes (not a plain mean-of-means), and tasks
missing a stats.json are reported as missing rather than silently skipped,
so an incomplete sweep is visible instead of producing a falsely-confident
number.
"""
import argparse
import json
from pathlib import Path

ALL_TASKS = [
    "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug",
    "NavigateKitchen", "OpenCabinet", "OpenDrawer", "OpenStandMixerHead",
    "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack",
    "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
    "DeliverStraw", "GetToastedBread", "KettleBoiling", "LoadDishwasher",
    "PackIdenticalLunches", "PreSoakPan", "PrepareCoffee", "RinseSinkBasin",
    "ScrubCuttingBoard", "SearingMeat", "SetUpCuttingStation", "StackBowlsCabinet",
    "SteamInMicrowave", "StirVegetables", "StoreLeftoversInBowl", "WashLettuce",
    "ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CategorizeCondiments",
    "CuttingToolSelection", "GarnishPancake", "GatherTableware", "HeatKebabSandwich",
    "MakeIceLemonade", "PanTransfer", "PortionHotDogs", "RecycleBottlesByType",
    "SeparateFreezerRack", "WaffleReheat", "WashFruitColander", "WeighIngredients",
]

GROOT_FAMILIES = ["target_posttraining", "target_only", "pretraining", "multitask_learning"]
OPENPI_VARIANTS = ["pi0", "pi0.5"]
OPENPI_FAMILY = "pretraining"
OPENPI_SPLIT = "target"


def groot_stats_path(base_video_dir: Path, family: str, task: str) -> Path:
    return base_video_dir / family / task / "stats.json"


def openpi_stats_path(base_log_dir: Path, variant: str, family: str, task: str, split: str) -> Path:
    task_root = base_log_dir / variant / family / task / "evals_1.5" / split / task
    if not task_root.is_dir():
        return task_root / "stats.json"  # nonexistent, for a uniform "missing" report
    run_dirs = sorted(task_root.glob("*"))
    if not run_dirs:
        return task_root / "stats.json"
    return run_dirs[-1] / "stats.json"


def collect(model_name: str, task_paths: dict) -> dict:
    per_task = {}
    missing = []
    total_episodes = 0
    total_successes = 0.0
    for task, path in task_paths.items():
        if not path.is_file():
            missing.append(task)
            continue
        with open(path) as f:
            stats = json.load(f)
        n = stats.get("num_episodes")
        sr = stats.get("success_rate")
        if n is None or sr is None:
            missing.append(task)
            continue
        per_task[task] = {"num_episodes": n, "success_rate": sr}
        total_episodes += n
        total_successes += sr * n

    overall = total_successes / total_episodes if total_episodes else None
    return {
        "model": model_name,
        "num_tasks_found": len(per_task),
        "num_tasks_missing": len(missing),
        "missing_tasks": missing,
        "total_episodes": total_episodes,
        "overall_success_rate": overall,
        "per_task": per_task,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--saved_eval_rollouts",
        default=str(Path(__file__).resolve().parents[1] / "saved_eval_rollouts"),
        help="Root dir (symlinked to flash) holding groot/ and openpi/ subtrees.",
    )
    ap.add_argument("--out", default=None, help="Optional path to also dump JSON results.")
    args = ap.parse_args()

    root = Path(args.saved_eval_rollouts)
    groot_root = root / "groot"
    openpi_root = root / "openpi"

    results = []
    for family in GROOT_FAMILIES:
        task_paths = {t: groot_stats_path(groot_root, family, t) for t in ALL_TASKS}
        results.append(collect(f"groot/{family}", task_paths))

    for variant in OPENPI_VARIANTS:
        task_paths = {
            t: openpi_stats_path(openpi_root, variant, OPENPI_FAMILY, t, OPENPI_SPLIT)
            for t in ALL_TASKS
        }
        results.append(collect(f"openpi/{variant}/{OPENPI_FAMILY}", task_paths))

    print(f"{'model':40s} {'tasks_found':>12s} {'missing':>8s} {'episodes':>9s} {'success_rate':>13s}")
    for r in results:
        sr_str = f"{r['overall_success_rate']:.4f}" if r["overall_success_rate"] is not None else "n/a"
        print(
            f"{r['model']:40s} {r['num_tasks_found']:>12d} {r['num_tasks_missing']:>8d} "
            f"{r['total_episodes']:>9d} {sr_str:>13s}"
        )
        if r["missing_tasks"]:
            print(f"    missing: {', '.join(r['missing_tasks'])}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull per-task breakdown written to {args.out}")


if __name__ == "__main__":
    main()
