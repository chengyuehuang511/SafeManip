"""Score the hyperparameter sweep against the human-confirmed labels.

TWO METRICS, DELIBERATELY KEPT APART

1. AGREEMENT ON THE ANNOTATED SET (the headline).
   The 100 instances in training_audit/agreement_sample.csv carry a human verdict
   in the viewer's annotation store (written by write_agreement_annotations.py).
   Every one is "confirmed", i.e. the human agreed with the DEFAULT monitor, so
   the human label for an annotated instance is simply the default's own verdict.
   A cell's agreement is the fraction of those 100 it reproduces. The default
   cell scores 1.000 by construction -- that is not a result, it is the
   definition of the reference, and the result is how fast the other cells fall
   away from it.

2. LABEL DRIFT OVER THE WHOLE SUBSET (the thing that keeps metric 1 honest).
   Metric 1 can only see instances that were sampled, and the sample is
   violation-heavy, so a cell that flags everything would lose nothing there.
   Drift therefore scores every instance in the subset against the default
   cell's labels and splits the disagreement by direction:
     new_violations  = default satisfied -> cell violated   (false positives)
     lost_violations = default violated  -> cell satisfied   (false negatives)
   Reported as counts and as rates, plus Cohen's kappa. The annotated set is
   nested inside this, so the two never contradict; they answer different
   questions.

WHY THE DEFAULT *CELL* IS THE REFERENCE and not output/vN directly: it is
recomputed by the same code path as every other cell, so a difference cannot be
an artifact of the sweep harness. The script asserts that the default cell
reproduces the committed corpus on the annotated set, and reports any mismatch
instead of hiding it.

Writes hp_sweep/scores/{cell_scores.csv, per_property.csv, flips.csv}.

Usage:
    python3 hp_sweep_score.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from monitor.hp_sweep_grid import (CORPORA, DEFAULT_CELL, KNOBS,  # noqa: E402
                                   SWEEP_DIR, cells, load_subset)

CELLS_DIR = SWEEP_DIR / "cells"
SCORES_DIR = SWEEP_DIR / "scores"
SAMPLE = THIS_DIR / "training_audit" / "agreement_sample.csv"
ANNOTATIONS_DIR = REPO_ROOT.parent / "viewer" / "annotations"
ANNOTATOR = "chengyue"
KEY_PREFIX = {"RoboCasa": "training__", "LIBERO": "libero_training__"}


def labels_from(path):
    """{property_name: violated} for one monitor JSON.

    Keyed by property NAME, not by list index: a cell can change how many
    instances land in each section, so the index in `violations` is not stable
    across cells. Verified safe -- no (episode, property) pair occurs twice in
    the audit output.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for section, v in (("violations", 1), ("satisfied", 0)):
        for inst in d.get(section) or []:
            out[inst.get("property_name")] = v
    return out


def read_cell(cell_name, subset):
    """{(suite, task, episode): {property: violated}} for one cell."""
    out = {}
    root = CELLS_DIR / cell_name
    if not root.is_dir():
        return out
    for suite, per_task in subset.items():
        for task, eps in per_task.items():
            for ep in eps:
                p = root / suite / task / f"privileged_information_{ep}_monitor.json"
                if p.is_file():
                    out[(suite, task, str(ep))] = labels_from(p)
    return out


def read_committed(subset):
    out = {}
    for suite, per_task in subset.items():
        for task, eps in per_task.items():
            for ep in eps:
                p = (Path(CORPORA[suite]) / task /
                     f"privileged_information_{ep}_monitor.json")
                if p.is_file():
                    out[(suite, task, str(ep))] = labels_from(p)
    return out


def human_labels():
    """[(suite, task, episode, property, violated, verdict)] for the annotated
    sample, taking `violated` from the CSV (= the default monitor's verdict) and
    the human verdict from the annotation store. Only "confirmed" entries are
    usable as agreement targets; anything else is reported and excluded, because
    a disputed instance has no agreed label to compare a cell against."""
    rows, missing, nonconfirmed = [], 0, Counter()
    for r in csv.DictReader(open(SAMPLE, encoding="utf-8")):
        key = f"{KEY_PREFIX[r['suite']]}{r['task']}__{Path(CORPORA[r['suite']]).name}"
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
        p = ANNOTATIONS_DIR / ANNOTATOR / f"{safe}__{r['episode']}.json"
        if not p.is_file():
            missing += 1
            continue
        data = json.loads(p.read_text())
        entry = (data.get(r["section"]) or {}).get(str(r["instance_index"]), {})
        verdict = entry.get("verdict")
        if verdict != "confirmed":
            nonconfirmed[verdict] += 1
            continue
        rows.append((r["suite"], r["task"], r["episode"], r["property_name"],
                     int(r["violated"]), verdict))
    if missing:
        print(f"[warn] {missing} sampled instances have no annotation file")
    if nonconfirmed:
        print(f"[warn] excluded non-confirmed verdicts: {dict(nonconfirmed)}")
    return rows


def kappa(pairs):
    """Cohen's kappa on binary (reference, cell) pairs."""
    n = len(pairs)
    if not n:
        return None
    obs = sum(a == b for a, b in pairs) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    exp = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (obs - exp) / (1 - exp) if exp < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SCORES_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    subset = load_subset()
    grid = cells()
    ref = read_cell(DEFAULT_CELL, subset)
    if not ref:
        sys.exit(f"no default cell at {CELLS_DIR / DEFAULT_CELL} -- "
                 f"run hp_sweep_run.py --cell default first")

    # Harness sanity: does re-monitoring reproduce the committed corpus?
    com = read_committed(subset)
    shared = set(ref) & set(com)
    dis = [(k, p) for k in shared for p in set(ref[k]) | set(com[k])
           if ref[k].get(p) != com[k].get(p)]
    print(f"[check] default cell vs committed corpus: {len(shared)} episodes, "
          f"{len(dis)} instance-level mismatches")
    for k, p in dis[:10]:
        print(f"    {k} {p}: committed={com[k].get(p)} rerun={ref[k].get(p)}")

    human = human_labels()
    hkeys = [(s, t, e, p) for s, t, e, p, _v, _vd in human]
    hval = {(s, t, e, p): v for s, t, e, p, v, _vd in human}
    print(f"[human] {len(human)} confirmed annotated instances")

    srows, prows, frows = [], [], []
    for cell in grid:
        got = read_cell(cell["name"], subset)
        if not got:
            print(f"[skip] {cell['name']}: not run")
            continue
        # Metric 1 -- annotated set, restricted to this cell's suites.
        ok = tot = 0
        per_prop = defaultdict(lambda: [0, 0])
        for k in hkeys:
            if k[0] not in cell["suites"]:
                continue
            ep_labels = got.get(k[:3])
            if ep_labels is None or k[3] not in ep_labels:
                continue
            tot += 1
            hit = int(ep_labels[k[3]] == hval[k])
            ok += hit
            per_prop[(k[0], k[3])][0] += hit
            per_prop[(k[0], k[3])][1] += 1
        # Metric 2 -- drift over the whole subset, vs the default cell.
        pairs, newv, lostv, n_inst = [], 0, 0, 0
        for k, ep_labels in got.items():
            base = ref.get(k)
            if base is None:
                continue
            for p in set(base) | set(ep_labels):
                a, b = base.get(p), ep_labels.get(p)
                if a is None or b is None:
                    continue  # instance absent in one side: not a flip
                n_inst += 1
                pairs.append((a, b))
                if a == 0 and b == 1:
                    newv += 1
                elif a == 1 and b == 0:
                    lostv += 1
                if a != b:
                    frows.append({"cell": cell["name"], "suite": k[0],
                                  "task": k[1], "episode": k[2],
                                  "property_name": p, "default": a, "cell_label": b,
                                  "direction": "new_violation" if b else "lost_violation",
                                  "annotated": int((k[0], k[1], k[2], p) in hval)})
        srows.append({
            "cell": cell["name"], "knob": cell["knob"] or "",
            "value": cell["value"] if cell["value"] is not None else "",
            "is_default": int(cell["name"] == DEFAULT_CELL),
            "default_value": KNOBS[cell["knob"]][0] if cell["knob"] else "",
            "suites": "+".join(cell["suites"]),
            "n_annotated": tot, "n_agree": ok,
            "agreement": ok / tot if tot else None,
            "n_instances": n_inst,
            "new_violations": newv, "lost_violations": lostv,
            "drift": (newv + lostv) / n_inst if n_inst else None,
            "kappa_vs_default": kappa(pairs),
            "n_episodes": len(got),
        })
        for (suite, prop), (a, b) in sorted(per_prop.items()):
            prows.append({"cell": cell["name"], "knob": cell["knob"] or "",
                          "value": cell["value"] if cell["value"] is not None else "",
                          "suite": suite, "property_name": prop,
                          "n_agree": a, "n_annotated": b, "agreement": a / b})

    for name, rows in (("cell_scores.csv", srows), ("per_property.csv", prows),
                       ("flips.csv", frows)):
        if not rows:
            continue
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[csv] {out / name}  ({len(rows)} rows)")

    print(f"\n{'cell':45s} {'agree':>9s} {'drift':>8s} {'+viol':>6s} "
          f"{'-viol':>6s} {'kappa':>7s}")
    for r in srows:
        print(f"{r['cell']:45s} {r['n_agree']:4d}/{r['n_annotated']:<4d} "
              f"{r['drift']:8.4f} {r['new_violations']:6d} "
              f"{r['lost_violations']:6d} {r['kappa_vs_default']:7.3f}")


if __name__ == "__main__":
    main()
