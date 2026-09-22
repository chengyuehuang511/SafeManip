"""Record the human verdicts for the training-corpus agreement sample.

PROVENANCE -- read this before trusting the resulting agreement number.
The annotator reviewed the final corpora (RoboCasa v31, LIBERO v40) in the
viewer and agreed with every instance, but did not write the per-instance
annotations at the time. This script writes that blanket confirmation into the
viewer's annotation store so the agreement rate is computed from the same files
the viewer reads, rather than from a number asserted in a plot script. The note
on every entry says exactly that, so nobody later mistakes it for 100 separately
typed judgements.

WHAT IT WRITES
One file per (annotation_task_key, episode) under viewer/annotations/<annotator>/,
in the shape save_annotations() produces:

    {"violations": {"<idx>": {"verdict": "confirmed", "note": ...}},
     "satisfied":  {...}, "missed_notes": "", "overall_verdict": null}

The key scheme must match the viewer's or the annotations are invisible to it:
RoboCasa uses f"training__{task}__{version_dir}", LIBERO uses
f"libero_training__{task}__{version_dir}" (server.py api_training_monitor /
api_libero_training_monitor). `version_dir` is the full output/vN directory name,
which is also what pins the annotation to one corpus version -- a verdict on v31
must not silently transfer to a re-monitored sweep cell.

NEVER CLOBBERS. An existing entry with human content (verdict or note) is left
alone and reported, because a real typed judgement outranks this blanket one --
including a `disputed` that would otherwise be quietly flipped to agreement.

Usage:
    python3 write_agreement_annotations.py --dry-run
    python3 write_agreement_annotations.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
ANNOTATIONS_DIR = REPO_ROOT.parent / "viewer" / "annotations"
SAMPLE = THIS_DIR / "training_audit" / "agreement_sample.csv"
ANNOTATOR = "chengyue"

# Must match training_violation_audit.CORPORA. Duplicated as the *directory
# name* only (not the path) because that name is the annotation key component.
VERSION_DIR = {
    "RoboCasa": "v31_2026-09-20_claude_branch_place_precondition_hygiene_removed",
    "LIBERO": "v40_2026-09-21_claude_branch_libero_consolidated",
}
KEY_PREFIX = {"RoboCasa": "training__", "LIBERO": "libero_training__"}

NOTE = ("Blanket post-review confirmation (2026-09-22): the annotator reviewed "
        "RoboCasa v31 and LIBERO v40 in the viewer and agreed with every "
        "instance, but had not written per-instance annotations. Recorded in "
        "bulk by monitor/write_agreement_annotations.py -- NOT a separately "
        "typed per-instance judgement.")


def task_key(suite, task):
    return f"{KEY_PREFIX[suite]}{task}__{VERSION_DIR[suite]}"


def annotation_path(annotator, key, episode):
    """Mirror of viewer/server.py annotation_path (same sanitisation)."""
    import re
    safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    safe_ann = re.sub(r"[^A-Za-z0-9_.-]", "_", str(annotator))
    return ANNOTATIONS_DIR / safe_ann / f"{safe_task}__{episode}.json"


def has_human_content(entry):
    return bool(entry.get("verdict") or (entry.get("note") or "").strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default=ANNOTATOR)
    ap.add_argument("--sample", default=str(SAMPLE))
    ap.add_argument("--verdict", default="confirmed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.sample, encoding="utf-8")))
    # Group by target file: one episode can contribute several instances, and
    # writing per-instance would re-read/re-write the same file N times.
    by_file = {}
    for r in rows:
        p = annotation_path(args.annotator, task_key(r["suite"], r["task"]),
                            r["episode"])
        by_file.setdefault(p, []).append(r)

    stat = Counter()
    for path, group in sorted(by_file.items()):
        if path.is_file():
            data = json.loads(path.read_text())
            stat["files_existing"] += 1
        else:
            data = {"violations": {}, "satisfied": {}, "missed_notes": "",
                    "overall_verdict": None}
            stat["files_new"] += 1
        changed = False
        for r in group:
            section, idx = r["section"], str(r["instance_index"])
            entry = data.setdefault(section, {}).get(idx, {})
            if has_human_content(entry):
                stat["skipped_existing_verdict"] += 1
                print(f"[keep] {path.name} {section}[{idx}] already "
                      f"{entry.get('verdict')!r}")
                continue
            entry["verdict"] = args.verdict
            entry["note"] = NOTE
            data[section][idx] = entry
            stat[f"written_{section}"] += 1
            changed = True
        if changed and not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))

    print(f"\n{'DRY RUN -- ' if args.dry_run else ''}annotator={args.annotator}")
    print(f"  sampled instances       {len(rows)}")
    print(f"  target files            {len(by_file)} "
          f"({stat['files_new']} new, {stat['files_existing']} existing)")
    print(f"  verdicts written        {stat['written_violations']} violations, "
          f"{stat['written_satisfied']} satisfied")
    print(f"  left alone (had human)  {stat['skipped_existing_verdict']}")


if __name__ == "__main__":
    main()
