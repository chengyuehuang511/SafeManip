#!/usr/bin/env python3
"""Carries human annotations forward from one predicates.py "version" (an
eval method directory under SafeManip/monitor/output/) to another, for
episodes/properties whose actual result didn't change between the two --
so a human reviewer doesn't have to re-review something nothing about
actually changed, just because the version number bumped.

Two things this does, per (task, episode, property) with a real human
annotation in --from-method:

1. Same result in --to-method (same group -- both violated, or both
   satisfied -- for that property): copies the annotation entry forward to
   --to-method's own index for that property (annotations are keyed by
   list index within violations/satisfied, which can shift between
   versions if the *set* of violated properties for an episode changes --
   this always re-resolves the index by property name, never assumes the
   index itself is stable across versions).

2. --from-method's annotation was "disputed" (the reviewer believed the
   monitor's classification was wrong) AND the property is no longer
   violated in --to-method (i.e. the dispute is now vindicated by a real
   fix): updates the *original* --from-method annotation's verdict from
   "disputed" to "confirmed" in place. Nothing is copied forward to
   --to-method for this case -- there is no violation entry there to
   attach it to (the issue is fixed, so it isn't in the violations list at
   all in --to-method).

Anything else (annotated in --from-method but the result changed to
something other than "no longer violated", e.g. newly violated, or a
different kind of change) is left alone -- ambiguous, needs a human to
re-review, not something safe to auto-migrate.

Usage:
    python3 migrate_annotations.py --from-method v20_... --to-method v21_...
    python3 migrate_annotations.py --from-method v20_... --to-method v21_... --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parent / "annotations"
MONITOR_OUTPUT_ROOT = Path(__file__).parent.parent / "SafeManip" / "monitor" / "output"


def _safe_task(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", task)


def annotation_path(training_task_key: str, episode: int) -> Path:
    return ANNOTATIONS_DIR / f"{_safe_task(training_task_key)}__{episode}.json"


def load_annotations(path: Path) -> dict:
    if not path.is_file():
        return {"violations": {}, "satisfied": {}, "missed_notes": "", "overall_verdict": None}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"violations": {}, "satisfied": {}, "missed_notes": "", "overall_verdict": None}


def _entry_has_human_content(entry) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("verdict") or (entry.get("note") or "").strip():
        return True
    human_block = entry.get("human")
    return isinstance(human_block, dict) and bool(
        human_block.get("gt_annotation") or human_block.get("monitor_problem")
    )


def _property_index_map(mon: dict) -> dict[str, tuple[str, int]]:
    """{property_name: (group, index)} for one episode's monitor.json --
    "group" is "violations" or "satisfied"."""
    out = {}
    for group in ("violations", "satisfied"):
        for idx, entry in enumerate(mon.get(group) or []):
            prop = entry.get("property_name")
            if prop:
                out[prop] = (group, idx)
    return out


def find_annotated_tasks(from_method: str) -> list[tuple[str, int]]:
    """[(task, episode), ...] for every training__<task>__<from_method>__<ep>.json
    annotation file that exists (regardless of content -- checked properly
    per-entry later)."""
    prefix = f"training__"
    suffix_re = re.compile(
        r"^training__(?P<task>.+)__" + re.escape(from_method) + r"__(?P<ep>\d+)\.json$"
    )
    out = []
    for p in ANNOTATIONS_DIR.glob(f"training__*__{from_method}__*.json"):
        m = suffix_re.match(p.name)
        if m:
            out.append((m.group("task"), int(m.group("ep"))))
    return sorted(out)


def migrate(from_method: str, to_method: str, dry_run: bool = False) -> None:
    pairs = find_annotated_tasks(from_method)
    print(f"found {len(pairs)} annotated (task, episode) pair(s) for {from_method}")

    copied = 0
    confirmed = 0
    skipped = 0

    for task, ep in pairs:
        from_key = f"training__{task}__{from_method}"
        to_key = f"training__{task}__{to_method}"

        from_ann_path = annotation_path(from_key, ep)
        from_ann = load_annotations(from_ann_path)

        from_mon_path = MONITOR_OUTPUT_ROOT / from_method / task / f"privileged_information_{ep}_monitor.json"
        to_mon_path = MONITOR_OUTPUT_ROOT / to_method / task / f"privileged_information_{ep}_monitor.json"
        if not from_mon_path.is_file() or not to_mon_path.is_file():
            print(f"  skip {task} ep{ep}: missing monitor.json ({from_method} or {to_method})")
            continue

        from_mon = json.loads(from_mon_path.read_text())
        to_mon = json.loads(to_mon_path.read_text())
        from_map = _property_index_map(from_mon)
        to_map = _property_index_map(to_mon)

        from_ann_changed = False
        to_ann = load_annotations(annotation_path(to_key, ep))
        to_ann_changed = False

        for group in ("violations", "satisfied"):
            for idx_str, entry in list((from_ann.get(group) or {}).items()):
                if not _entry_has_human_content(entry):
                    continue
                idx = int(idx_str)
                # Find which property this index actually refers to in from_method.
                prop = None
                for p, (g, i) in from_map.items():
                    if g == group and i == idx:
                        prop = p
                        break
                if prop is None:
                    print(f"  ! {task} ep{ep}: no property found for {from_method} {group}[{idx}], skipping")
                    skipped += 1
                    continue

                to_entry = to_map.get(prop)
                if to_entry is not None and to_entry[0] == group:
                    # Same result -- copy forward to to_method's own index.
                    to_group, to_idx = to_entry
                    to_ann.setdefault(to_group, {})
                    key = str(to_idx)
                    if str(to_ann[to_group].get(key)) != str(entry):
                        to_ann[to_group][key] = copy.deepcopy(entry)
                        to_ann_changed = True
                        copied += 1
                        print(f"  copy  {task} ep{ep} {prop}: {from_method}.{group}[{idx}] -> {to_method}.{to_group}[{to_idx}]")
                elif (
                    group == "violations"
                    and entry.get("verdict") == "disputed"
                    and (to_entry is None or to_entry[0] == "satisfied")
                ):
                    # Disputed in from_method, no longer violated in to_method --
                    # the dispute is vindicated. Update from_method's own record.
                    if entry.get("verdict") != "confirmed":
                        entry["verdict"] = "confirmed"
                        from_ann_changed = True
                        confirmed += 1
                        print(f"  confirm {task} ep{ep} {prop}: {from_method}.violations[{idx}] disputed -> confirmed (fixed in {to_method})")
                else:
                    skipped += 1

        if not dry_run:
            if from_ann_changed:
                from_ann_path.write_text(json.dumps(from_ann, indent=2, sort_keys=True))
            if to_ann_changed:
                to_path = annotation_path(to_key, ep)
                to_path.write_text(json.dumps(to_ann, indent=2, sort_keys=True))

    print(f"\ndone: {copied} annotation(s) copied forward, {confirmed} disputed->confirmed, {skipped} skipped (ambiguous or unmatched)")
    if dry_run:
        print("(dry run -- no files written)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-method", required=True, help="e.g. v20_2026-09-08_claude_branch_gripper_retract_tool_target_contact_fixes")
    ap.add_argument("--to-method", required=True, help="e.g. v21_2026-09-08_claude_branch_mystery_middle_carrier_gripper_mesh_fixes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    migrate(args.from_method, args.to_method, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
