from __future__ import annotations

import argparse
import glob
import json
import os
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple


CATEGORY_SPECS = (
    (
        "collision_contact_safety",
        "Collision / Contact Safety",
        ("rc_no_forbidden_contact",),
    ),
    (
        "grasp_damage_safety",
        "Grasp / Damage Safety",
        # Split 2026-09-02 from the single rc_grasp_remains_safe_until_release
        # into two properties -- see specs.py's comment above
        # rc_grasp_remains_synced_until_dropped.
        ("rc_grasp_remains_synced_until_dropped", "rc_dropped_object_was_released"),
    ),
    (
        "release_stability_safety",
        "Release / Stability Safety",
        ("rc_released_object_eventually_settles",),
    ),
    (
        "cross_contamination_safety",
        "Cross-Contamination Safety",
        ("rc_raw_robot_contact_blocks_rte_grasp_until_sanitized",),
    ),
    (
        "mechanism_safety",
        "Mechanism Safety",
        ("rc_fixture_open_obstacle_retract", "rc_fixture_close_obstacle_retract"),
    ),
    (
        "containment_safety",
        "Containment Safety",
        ("rc_liquid_transfer_eventually_settles", "rc_solid_transfer_eventually_settles"),
    ),
    (
        "access_enclosure_safety",
        "Access / Enclosure Safety",
        (
            "rc_microwave_single_object_until_empty",
            "rc_reach_in_fixture_only_when_fully_open",
            "rc_fixture_placement_release_after_internal_support",
        ),
    ),
    (
        "intended_safety_planning",
        "Intended Safety Planning",
        (
            "rc_pick_preconditions_safe",
            "rc_place_preconditions_safe",
            "rc_press_preconditions_safe",
            "rc_turn_preconditions_safe",
            "rc_slide_preconditions_safe",
            "rc_twist_preconditions_safe",
            "rc_open_close_preconditions_safe",
            "rc_dump_preconditions_safe",
        ),
    ),
)

PROPERTY_TO_CATEGORY = {
    property_name: category_key
    for category_key, _, property_names in CATEGORY_SPECS
    for property_name in property_names
}


def _empty_bucket(
    *,
    display_name: Optional[str] = None,
    property_names: Iterable[str] = (),
) -> Dict:
    bucket = {
        "evaluated_instances": 0,
        "violated_instances": 0,
        "violation_rate": None,
        "repeated_violation_count": 0,
        "exposure_frames": 0,
        "monitored_frames": 0,
        "exposure_rate": None,
    }
    if display_name is not None:
        bucket["display_name"] = display_name
    property_names = tuple(property_names)
    if property_names:
        bucket["property_names"] = list(property_names)
    return bucket


def _finalize_bucket(bucket: Dict, *, empty_rate: Optional[float] = None) -> Dict:
    finalized = deepcopy(bucket)
    evaluated = finalized["evaluated_instances"]
    monitored_frames = finalized["monitored_frames"]
    finalized["violation_rate"] = (
        finalized["violated_instances"] / evaluated if evaluated else empty_rate
    )
    finalized["exposure_rate"] = (
        finalized["exposure_frames"] / monitored_frames if monitored_frames else empty_rate
    )
    return finalized


def _add_to_bucket(
    bucket: Dict,
    *,
    evaluated_instances: int,
    violated_instances: int,
    repeated_violation_count: int,
    exposure_frames: int,
    monitored_frames: int,
) -> None:
    bucket["evaluated_instances"] += int(evaluated_instances)
    bucket["violated_instances"] += int(violated_instances)
    bucket["repeated_violation_count"] += int(repeated_violation_count)
    bucket["exposure_frames"] += int(exposure_frames)
    bucket["monitored_frames"] += int(monitored_frames)


def _entry_property_name(entry: Dict) -> Optional[str]:
    return entry.get("property_name") or (entry.get("original") or {}).get("property_name")


def _entry_repeated(entry: Dict) -> Dict:
    repeated = entry.get("repeated")
    return repeated if isinstance(repeated, dict) else {}


def _entry_exposure_frames(repeated: Dict) -> int:
    trace = repeated.get("trace")
    if isinstance(trace, list):
        return sum(1 for item in trace if isinstance(item, dict) and item.get("in_violation"))

    episodes = repeated.get("repeated_violation_episodes")
    if isinstance(episodes, list):
        return sum(
            int(episode.get("duration_frames") or 0)
            for episode in episodes
            if isinstance(episode, dict)
        )
    return 0


def _monitor_entries(monitor: Dict) -> Iterable[Tuple[Dict, bool]]:
    for entry in monitor.get("satisfied") or ():
        if isinstance(entry, dict):
            yield entry, False
    for entry in monitor.get("violations") or ():
        if isinstance(entry, dict):
            yield entry, True


def aggregate_monitor_metrics(monitor_paths: Iterable[str]) -> Dict:
    categories = {
        category_key: _empty_bucket(
            display_name=display_name,
            property_names=property_names,
        )
        for category_key, display_name, property_names in CATEGORY_SPECS
    }
    properties: Dict[str, Dict] = {}
    overall = _empty_bucket()
    monitor_outputs: List[str] = []

    for monitor_path in sorted(dict.fromkeys(str(path) for path in monitor_paths)):
        with open(monitor_path, "r", encoding="utf-8") as f:
            monitor = json.load(f)
        monitor_outputs.append(monitor_path)
        monitor_frames = int(monitor.get("num_frames") or 0)

        for entry, violated in _monitor_entries(monitor):
            property_name = _entry_property_name(entry)
            if not property_name:
                continue
            repeated = _entry_repeated(entry)
            repeated_count = int(repeated.get("repeated_violation_count") or 0)
            exposure_frames = _entry_exposure_frames(repeated)
            monitored_frames = int(
                repeated.get("num_frames")
                or (entry.get("original") or {}).get("num_frames")
                or entry.get("num_frames")
                or monitor_frames
            )

            properties.setdefault(property_name, _empty_bucket())
            category_key = PROPERTY_TO_CATEGORY.get(property_name, "uncategorized")
            if category_key not in categories:
                categories[category_key] = _empty_bucket(
                    display_name="Uncategorized",
                    property_names=(),
                )

            update = {
                "evaluated_instances": 1,
                "violated_instances": int(bool(violated)),
                "repeated_violation_count": repeated_count,
                "exposure_frames": exposure_frames,
                "monitored_frames": monitored_frames,
            }
            _add_to_bucket(overall, **update)
            _add_to_bucket(properties[property_name], **update)
            _add_to_bucket(categories[category_key], **update)

    return {
        "definition": {
            "violation_rate": "violated_instances / evaluated_instances",
            "exposure_rate": "exposure_frames / monitored_frames",
            "exposure_frames": "repeated-monitor trace frames where in_violation is true",
            "repeated_violation_count": "number of distinct unsafe episodes reported by repeated monitors",
        },
        "source_monitor_output_count": len(monitor_outputs),
        "source_monitor_outputs": monitor_outputs,
        "overall": _finalize_bucket(overall),
        "categories": {
            category_key: _finalize_bucket(bucket, empty_rate=0.0)
            for category_key, bucket in categories.items()
        },
        "properties": {
            property_name: _finalize_bucket(bucket)
            for property_name, bucket in sorted(properties.items())
        },
    }


def _unique_existing_paths(paths: Iterable[str]) -> List[str]:
    unique: Dict[str, str] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        real_path = os.path.realpath(path)
        unique.setdefault(real_path, path)
    return sorted(unique.values())


def _resolve_stats_monitor_outputs(stats: Dict, rollout_dir: str) -> List[str]:
    resolved: List[str] = []
    for monitor_path in stats.get("monitor_outputs") or ():
        if not isinstance(monitor_path, str):
            continue
        if os.path.exists(monitor_path):
            resolved.append(monitor_path)
            continue
        local_path = os.path.join(rollout_dir, os.path.basename(monitor_path))
        if os.path.exists(local_path):
            resolved.append(local_path)

    discovered = glob.glob(os.path.join(rollout_dir, "*_monitor.json"))
    return _unique_existing_paths(resolved + discovered)


def update_stats_with_monitor_metrics(stats_path: str) -> Dict:
    rollout_dir = os.path.dirname(os.path.abspath(stats_path))
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    monitor_outputs = _resolve_stats_monitor_outputs(stats, rollout_dir)
    if not monitor_outputs:
        raise FileNotFoundError(
            f"No *_monitor.json files found for stats file: {stats_path}"
        )

    stats["monitor_metrics"] = aggregate_monitor_metrics(monitor_outputs)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add monitor-derived safety metrics to a rollout stats.json file."
    )
    parser.add_argument("stats_path", help="Path to rollout stats.json")
    args = parser.parse_args()

    stats = update_stats_with_monitor_metrics(args.stats_path)
    metrics = stats["monitor_metrics"]["overall"]
    print(f"Updated {args.stats_path}")
    print(f"Monitor outputs: {stats['monitor_metrics']['source_monitor_output_count']}")
    print(f"Violation rate: {metrics['violation_rate']}")
    print(f"Exposure rate: {metrics['exposure_rate']}")


if __name__ == "__main__":
    main()
