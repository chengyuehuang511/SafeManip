"""Training-corpus violation audit + human-agreement sample, both suites.

WHAT THIS IS FOR
Two claims the paper needs about the monitor itself, as opposed to about the
policies:

  1. The monitor fires RARELY on the training demonstrations. These are
     human-teleoperated demos of successful task executions, so a monitor that
     flagged them constantly would be measuring its own false-positive rate, not
     policy safety. A low-but-nonzero rate is the expected shape: the demos are
     good, not perfect.
  2. When it does fire, a human agrees. Agreement is measured on a sample drawn
     from the FINAL corpora only (RoboCasa v31, LIBERO v40), because agreement
     on a superseded version says nothing about the shipped configuration.

WHAT IT WRITES
  <out>/training_violations_<suite>.csv   one row per (task, episode, property)
                                         instance: violated/satisfied, windows.
  <out>/agreement_sample.csv             the sampled instances to be labelled.

SAMPLING
Stratified over (suite, property) and then over tasks, so the sample cannot be
dominated by whichever property fires most -- an agreement rate computed on a
sample that is 80% one property is an agreement rate for that property. Both
VIOLATED and SATISFIED instances are drawn: a monitor that never false-negatives
but false-positives constantly and one with the reverse failure look identical if
you only sample its positives. Deterministic (fixed seed) so the sample can be
regenerated and re-audited.

Usage:
    python3 training_violation_audit.py --count-only
    python3 training_violation_audit.py --sample 100
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).parent
OUT_DIR = THIS_DIR / "training_audit"

# The plotting tree's metric_conventions.py is THE definition of violated/count/
# duration for both suites, and the training-corpus rate in the alignment figure
# has to be the same quantity as the eval-corpus rate it is compared against. So
# import the rule from there rather than keeping a third copy of it here -- a
# divergence would silently turn that figure into a comparison of two different
# metrics.
_CONV = ("/nethome/chuang475/testnvme/projects/SafeManip/eval/saved_eval_rollouts"
         "/monitor_files/0920")
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)
from metric_conventions import recovery_stats, unified_stats  # noqa: E402

# The FINAL corpora. Both suites share one version counter (v32+ in this
# directory are LIBERO, not later RoboCasa), which is why these are pinned by
# full directory name rather than by "highest vN".
CORPORA = {
    "RoboCasa": THIS_DIR / "output" / "v31_2026-09-20_claude_branch_place_precondition_hygiene_removed",
    "LIBERO": THIS_DIR / "output" / "v40_2026-09-21_claude_branch_libero_consolidated",
}
SEED = 0


def iter_monitor_files(root):
    for task_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(glob.glob(str(task_dir / "*_monitor.json"))):
            ep = Path(f).name[len("privileged_information_"):-len("_monitor.json")]
            yield task_dir.name, ep, f


def instances(path):
    """(property_name, index_within_section, section, window counts) rows.

    Counts come from metric_conventions, the same module the plotting loaders
    apply, so a training-corpus number is directly comparable to the eval-corpus
    numbers in the paper rather than being a differently-defined count:
    `unsafe_*` are the recovery trace's windows alone, `unified_*` add the
    terminal unresolved window and are what violated/count/duration are defined
    from.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rec = d.get("recovery_accepting_by_property") or {}
    main = d.get("accepting_by_property") or {}
    rows = []
    for section in ("violations", "satisfied"):
        for i, inst in enumerate(d.get(section) or []):
            p = inst.get("property_name")
            violated = section == "violations"
            frames, windows = recovery_stats(rec.get(p) or [])
            u_frames, u_windows = unified_stats(
                main.get(p) or [], rec.get(p) or [], violated)
            rows.append({
                "property_name": p,
                "instance_index": i,
                "section": section,
                "violated": int(violated),
                "unsafe_windows": windows,
                "unsafe_frames": frames,
                "unified_windows": u_windows,
                "unified_frames": u_frames,
                "ever_non_accepting": int(bool(inst.get("ever_non_accepting"))),
                "final_trap": int(bool(inst.get("final_trap"))),
            })
    return rows, d.get("num_frames"), d.get("task_name")


def audit(suite, root, writer):
    n_ep = 0
    per_prop = defaultdict(Counter)
    pool = []
    for task, ep, path in iter_monitor_files(root):
        try:
            rows, nframes, _tn = instances(path)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] unreadable {path}: {e!r}")
            continue
        n_ep += 1
        for r in rows:
            r.update(suite=suite, task=task, episode=ep, num_frames=nframes,
                     monitor_path=os.path.relpath(path, THIS_DIR))
            writer.writerow(r)
            per_prop[r["property_name"]][r["section"]] += 1
            pool.append(r)
    return n_ep, per_prop, pool


def stratified_sample(pool, n_total):
    """CENSUS of every violated instance, then fill to n_total with satisfied.

    The two sections are deliberately not treated alike:

    * Violations are taken in FULL. Both final corpora together contain only 81
      of them, fewer than the 100-instance budget, so the false-positive claim is
      a census of the whole population and needs no sampling argument at all --
      strictly stronger than a sample, and no property can be over-represented
      because none is omitted.
    * Satisfied instances fill the remainder, round-robin over
      (suite, property) so the spot-check spans properties rather than piling
      onto whichever one has the most instances. This side IS a small sample
      (~19 of ~17,900) and can only catch gross false negatives; it is reported
      as a spot-check, not as a false-negative rate.

    Deterministic given SEED.
    """
    rng = random.Random(SEED)
    picked = [r for r in pool if r["section"] == "violations"]
    remaining = n_total - len(picked)
    if remaining <= 0:
        return picked
    by_key = defaultdict(list)
    for r in pool:
        if r["section"] == "satisfied":
            by_key[(r["suite"], r["property_name"])].append(r)
    buckets = {k: rng.sample(v, len(v)) for k, v in by_key.items()}
    # Split the fill evenly between suites BEFORE the property round-robin.
    # A single round-robin over all (suite, property) keys would spend the whole
    # budget inside whichever suite sorts first -- there are ~20 properties per
    # suite and only ~19 slots, so the first pass never reaches the second suite.
    suites = sorted({k[0] for k in buckets})
    quotas = {s: remaining // len(suites) for s in suites}
    for s in suites[:remaining % len(suites)]:
        quotas[s] += 1
    for suite in suites:
        order = sorted(k for k in buckets if k[0] == suite)
        left = quotas[suite]
        while left > 0:
            progressed = False
            for k in order:
                if not buckets[k] or left <= 0:
                    continue
                picked.append(buckets[k].pop())
                left -= 1
                progressed = True
            if not progressed:
                break  # pool exhausted; the caller prints the shortfall
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["suite", "task", "episode", "property_name", "instance_index",
              "section", "violated", "unsafe_windows", "unsafe_frames",
              "unified_windows", "unified_frames", "ever_non_accepting",
              "final_trap", "num_frames", "monitor_path"]
    pools = {}
    for suite, root in CORPORA.items():
        if not Path(root).is_dir():
            print(f"[skip] {suite}: no corpus at {root}")
            continue
        path = out / f"training_violations_{suite}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            n_ep, per_prop, pool = audit(suite, root, w)
        pools[suite] = pool
        n_v = sum(1 for r in pool if r["violated"])
        n_i = len(pool)
        eps_with = len({(r["task"], r["episode"]) for r in pool if r["violated"]})
        print(f"\n=== {suite}  ({Path(root).name}) ===")
        print(f"  episodes                 {n_ep}")
        print(f"  property instances       {n_i}")
        print(f"  violated instances       {n_v}  ({n_v / max(n_i, 1):.2%})")
        print(f"  episodes with >=1        {eps_with}  ({eps_with / max(n_ep, 1):.2%})")
        print(f"  [csv] {path}")
        top = sorted(per_prop.items(), key=lambda kv: -kv[1]["violations"])[:8]
        for p, c in top:
            tot = c["violations"] + c["satisfied"]
            print(f"    {p:52s} {c['violations']:4d}/{tot:5d} "
                  f"({c['violations'] / max(tot, 1):6.2%})")

    if args.count_only or not pools:
        return
    pool = [r for p in pools.values() for r in p]
    picked = stratified_sample(pool, args.sample)
    spath = out / "agreement_sample.csv"
    with open(spath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(picked)
    print(f"\n[csv] {spath}: {len(picked)} instances")
    print("  by (suite, section): "
          + str(dict(Counter((r["suite"], r["section"]) for r in picked))))
    print(f"  distinct properties: {len({r['property_name'] for r in picked})}")
    print(f"  distinct tasks:      {len({(r['suite'], r['task']) for r in picked})}")


if __name__ == "__main__":
    main()
