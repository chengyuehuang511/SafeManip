"""Aggregate violation rates across a scaled extraction's output_root, broken
down by property and by task -- the summary the predicate-design-cycle needs
to confirm "violation rate is very low" on a corpus of successful demos.

Usage:
    python3 summarize_violation_rates.py --output_root <vN dir>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    by_property = defaultdict(lambda: {"violated": 0, "satisfied": 0})
    by_task = defaultdict(lambda: {"violated": 0, "satisfied": 0, "episodes": 0})
    task_violated_props = defaultdict(lambda: defaultdict(int))
    total_episodes = 0
    total_instances = 0
    total_violated_instances = 0

    for task_dir in sorted(output_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for mp in sorted(task_dir.glob("privileged_information_*_monitor.json")):
            try:
                data = json.loads(mp.read_text())
            except Exception as e:
                print(f"[{task}] {mp.name}: failed to parse ({e})")
                continue
            total_episodes += 1
            by_task[task]["episodes"] += 1
            for v in data.get("violations", []):
                prop = v.get("property_name")
                by_property[prop]["violated"] += 1
                by_task[task]["violated"] += 1
                task_violated_props[task][prop] += 1
                total_violated_instances += 1
                total_instances += 1
            for s in data.get("satisfied", []):
                prop = s.get("property_name")
                by_property[prop]["satisfied"] += 1
                by_task[task]["satisfied"] += 1
                total_instances += 1

    print(f"=== Overall: {total_episodes} episodes, {total_instances} property instances ===")
    overall_rate = (total_violated_instances / total_instances * 100) if total_instances else 0.0
    print(f"Overall violation rate: {total_violated_instances}/{total_instances} = {overall_rate:.2f}%\n")

    print("=== By property ===")
    for prop, counts in sorted(by_property.items(), key=lambda kv: -kv[1]["violated"]):
        total = counts["violated"] + counts["satisfied"]
        rate = (counts["violated"] / total * 100) if total else 0.0
        print(f"  {prop}: {counts['violated']}/{total} violated ({rate:.1f}%)")

    print("\n=== By task (episodes with >=1 violation) ===")
    for task, counts in sorted(by_task.items(), key=lambda kv: -kv[1]["violated"]):
        if counts["violated"] == 0:
            continue
        total = counts["violated"] + counts["satisfied"]
        rate = (counts["violated"] / total * 100) if total else 0.0
        props = ", ".join(f"{p}x{n}" for p, n in sorted(task_violated_props[task].items(), key=lambda kv: -kv[1]))
        print(f"  {task} ({counts['episodes']} episodes): {counts['violated']}/{total} ({rate:.1f}%) -- {props}")

    clean_tasks = [t for t, c in by_task.items() if c["violated"] == 0]
    print(f"\n{len(clean_tasks)}/{len(by_task)} tasks have zero violations across all episodes.")


if __name__ == "__main__":
    main()
