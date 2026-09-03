"""Populate the "claude" source block of the viewer's annotation schema
(see .claude/skills/ltl-ground-truth-annotation/SKILL.md) for every episode
of a scaled extraction, using the monitor's own (already-fixed-up-today)
computed output as an automated first-pass draft.

Honest about what this is and isn't: this is NOT independent, per-instance
manual re-verification of every frame against the raw sim state -- for a
500-episode x ~20-instance corpus that's thousands of instances, well beyond
what a single review pass can hand-check. What it *is*: reusing
viewer/server.py's own `load_monitor_view` (the exact same code path the
viewer itself uses to build predicate_breakdown/occurrences for display) to
mechanically populate each instance's `gt_annotation` with the trigger/
obligation/resolve-role frames the monitor already computed, labeled plainly
as monitor-derived rather than independently confirmed. `monitor_problem` is
populated conservatively: False by default, True with a caveat note only for
properties known (KNOWN_BUGS.md #7) to depend on fixture identity that the
raw data itself is unreliable for on ~89% of episodes -- this is a corpus-
wide caveat about the *data*, not a per-instance finding, and is labeled as
such.

Usage:
    python3 populate_claude_annotations.py --output_root <vN dir> [--tasks T1 T2 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "viewer"))

import server as viewer_server  # noqa: E402

# Properties whose main_ltl reads fixture *identity* (door/appliance-open
# state keyed by fixture name) -- KNOWN_BUGS.md #7: privileged_static_info
# and privileged_dynamic_info disagree on fixture naming/layout on ~89% of
# episodes (upstream data issue, not fixable in predicates.py). Any
# violation/satisfaction of these properties should be treated as suspect
# until that's fixed upstream.
FIXTURE_IDENTITY_DEPENDENT_PROPERTIES = {
    "rc_fixture_open_obstacle_retract",
    "rc_fixture_close_obstacle_retract",
    "rc_reach_in_fixture_only_when_fully_open",
    "rc_microwave_single_object_until_empty",
    "rc_fixture_placement_release_after_internal_support",
}


def _gt_annotation_from_breakdown(entry: dict) -> dict | None:
    breakdown = entry.get("predicate_breakdown")
    if not breakdown:
        return None
    return {
        "source_note": (
            "auto-derived from the monitor's own predicate_breakdown.occurrences "
            "(viewer/server.py's load_monitor_view) -- not independently "
            "re-verified frame-by-frame against raw sim state."
        ),
        "pattern": breakdown.get("pattern"),
        "occurrences": breakdown.get("occurrences", []),
        "confidence": "monitor-derived",
    }


def _monitor_problem_for(property_name: str) -> dict:
    if property_name in FIXTURE_IDENTITY_DEPENDENT_PROPERTIES:
        return {
            "has_problem": True,
            "description": (
                "Corpus-wide caveat (KNOWN_BUGS.md #7), not a per-instance "
                "finding: privileged_static_info and privileged_dynamic_info "
                "disagree on fixture naming/layout on ~89% of episodes "
                "(upstream data collection issue, outside SafeManip/). This "
                "property reads fixture identity, so its verdict for this "
                "specific episode has not been individually confirmed "
                "reliable against that known data problem."
            ),
        }
    return {"has_problem": False, "description": ""}


def populate_episode(base_dir: Path, task: str, episode: int, method: str) -> bool:
    mv = viewer_server.load_monitor_view(base_dir, episode, fps=10.0, video_duration=None)
    if mv is None:
        return False
    ann_task_key = f"training__{task}__{method}"
    for group in ("violations", "satisfied"):
        for idx, entry in enumerate(mv.get(group, [])):
            property_name = entry.get("property_name")
            gt = _gt_annotation_from_breakdown(entry)
            if gt is None:
                continue
            patch = {
                "group": group,
                "index": idx,
                "source": "claude",
                "gt_annotation": gt,
                "monitor_problem": _monitor_problem_for(property_name),
            }
            viewer_server.save_annotations(ann_task_key, episode, patch)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True, help="vN_.../ directory with per-task subdirs")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of tasks; default: all found")
    ap.add_argument("--method", default=None, help="method label used in the annotation task key")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    method = args.method or output_root.name
    tasks = args.tasks or sorted(p.name for p in output_root.iterdir() if p.is_dir())

    total_ok, total_missing = 0, 0
    for task in tasks:
        base_dir = output_root / task
        if not base_dir.is_dir():
            print(f"[{task}] no directory, skipping")
            continue
        episodes = sorted(
            {
                int(p.name.split("_")[-2])
                for p in base_dir.glob("privileged_information_*_monitor.json")
            }
        )
        for episode in episodes:
            ok = populate_episode(base_dir, task, episode, method)
            if ok:
                total_ok += 1
            else:
                total_missing += 1
        print(f"[{task}] {len(episodes)} episode(s) processed")
    print(f"done: {total_ok} episodes annotated, {total_missing} missing monitor.json")


if __name__ == "__main__":
    main()
