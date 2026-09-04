"""Systematic corpus-wide failure clustering (see predicate-design-cycle SKILL.md
Phase on "don't spot-check, cluster").

For a given output_root and a set of target property_names, collects every violated
instance's explanation text across the FULL corpus (not a few spot-checked examples),
normalizes away frame numbers / specific object instance names via regex, and groups
by the resulting signature. A property whose violations collapse into one or two
dominant signatures is a strong single-bug signal; broad, even distribution across many
distinct signatures means the violations are genuinely diverse real behavior.

Usage:
    python3 cluster_violation_explanations.py --output_root <vN dir> \
        --properties rc_pick_preconditions_safe rc_place_preconditions_safe ...
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FRAME_RE = re.compile(r"\bframe[s]?\s+\d+(-\d+)?\b", re.IGNORECASE)
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
# object/fixture instance suffixes like _g0, _1, _2 etc -- collapse to a placeholder
INSTANCE_SUFFIX_RE = re.compile(r"_(g\d+|main_group|group|left|right|top|bottom)\d*\b", re.IGNORECASE)


def normalize(explanation: str) -> str:
    s = explanation
    s = FRAME_RE.sub("frame <N>", s)
    s = NUM_RE.sub("<N>", s)
    return s.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--properties", nargs="+", required=True)
    args = ap.parse_args()

    root = Path(args.output_root)
    files = sorted(glob.glob(str(root / "*" / "privileged_information_*_monitor.json")))

    by_prop: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)

    for f in files:
        d = json.load(open(f))
        for v in d.get("violations") or []:
            pname = v.get("property_name")
            if pname not in args.properties:
                continue
            orig = v.get("original") or {}
            explanation = orig.get("explanation") or v.get("reason") or ""
            sig = normalize(explanation)
            by_prop[pname][sig] += 1
            if sig not in examples[pname]:
                examples[pname][sig] = (f, explanation)

    for pname in args.properties:
        counter = by_prop.get(pname)
        if not counter:
            print(f"\n=== {pname}: 0 violated instances ===")
            continue
        total = sum(counter.values())
        print(f"\n=== {pname}: {total} violated instances, {len(counter)} distinct signature(s) ===")
        for sig, count in counter.most_common(10):
            frac = 100 * count / total
            ex_file, ex_orig = examples[pname][sig]
            print(f"  [{count}/{total} = {frac:.0f}%] {sig[:160]}")
            print(f"      e.g. {ex_file}")


if __name__ == "__main__":
    main()
