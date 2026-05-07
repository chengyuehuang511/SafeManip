import argparse
import json
import math
import os
from collections import OrderedDict

import numpy as np
from robocasa.utils.dataset_registry import LIFELONG_LEARNING_TASKS, TARGET_TASKS
from termcolor import colored

TASK_GROUP_MAPPING = OrderedDict()
TASK_GROUP_MAPPING["atomic_seen"] = TARGET_TASKS["atomic_seen"]
TASK_GROUP_MAPPING["atomic_no_nav"] = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    # "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PnPCounterToCabinet",
    "PnPCounterToStove",
    "PnPDrawerToCounter",
    "PnPSinkToCounter",
    "PnPToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]
TASK_GROUP_MAPPING["composite_seen"] = TARGET_TASKS["composite_seen"]
TASK_GROUP_MAPPING["composite_unseen"] = TARGET_TASKS["composite_unseen"]
TASK_GROUP_MAPPING["lifelong_learning_phase1"] = TARGET_TASKS["atomic_seen"]
TASK_GROUP_MAPPING["lifelong_learning_phase2"] = LIFELONG_LEARNING_TASKS["lifelong_learning_phase2"]
TASK_GROUP_MAPPING["lifelong_learning_phase3"] = LIFELONG_LEARNING_TASKS["lifelong_learning_phase3"]
TASK_GROUP_MAPPING["lifelong_learning_phase4"] = LIFELONG_LEARNING_TASKS["lifelong_learning_phase4"]


def compute_stats(
    checkpoint_path, task_set=["atomic_seen", "composite_seen", "composite_unseen"], verbose=True
):
    stats = dict(
        pretrain=dict(),
        target=dict(),
    )

    assert os.path.exists(checkpoint_path)

    for split in ["pretrain", "target"]:
        split_dir = os.path.join(checkpoint_path, "evals", split)
        if not os.path.exists(split_dir):
            continue
        for task_name in os.listdir(split_dir):
            task_dir = os.path.join(split_dir, task_name)
            stats_path = os.path.join(task_dir, "stats.json")
            if not os.path.exists(stats_path):
                continue
            with open(stats_path, "r") as f:
                this_data = json.load(f)

            sr_key = "success_rate"
            if sr_key in this_data:
                stats[split][task_name] = this_data[sr_key]

    all_group_stats = dict()
    for group_name in task_set:
        task_names = TASK_GROUP_MAPPING[group_name]
        group_stats = dict(
            task_stats=dict(),
        )
        for task in task_names:
            group_stats["task_stats"][task] = dict()
            for split in ["pretrain", "target"]:
                val = stats[split].get(task, None)
                if val is not None:
                    val *= 100.0
                group_stats["task_stats"][task][split] = val

        for split in ["pretrain", "target"]:
            split_vals = [group_stats["task_stats"][task][split] for task in task_names]
            group_stats[f"avg_{split}"] = np.mean([val for val in split_vals if val is not None])

        all_group_stats[group_name] = group_stats

        if verbose:
            pretrain_avg = group_stats["avg_pretrain"]
            target_avg = group_stats["avg_target"]

            if np.isnan(pretrain_avg) and np.isnan(target_avg):
                continue

            print(colored(f"Stats for task group: {group_name.upper()}", "yellow"))

            for task in task_names:
                pretrain_val = group_stats["task_stats"][task]["pretrain"]
                target_val = group_stats["task_stats"][task]["target"]
                if pretrain_val is None and target_val is None:
                    continue
                if pretrain_val is not None:
                    pretrain_val = math.floor(pretrain_val + 0.5)
                if target_val is not None:
                    target_val = math.floor(target_val + 0.5)
                print(f"{task}: {pretrain_val} | {target_val}")
            print(colored(f"AVG: {(pretrain_avg):.1f} | {target_avg:.1f}", "yellow"))
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_set",
        type=str,
        nargs="+",
        default=["atomic_seen", "composite_seen", "composite_unseen"],
    )
    args = parser.parse_args()
    compute_stats(args.dir, task_set=args.task_set)
