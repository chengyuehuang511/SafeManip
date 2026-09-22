"""The hyperparameter sensitivity grid, and the episode subset it runs on.

WHAT THE SWEEP IS ARGUING
The monitor's numeric predicate thresholds were tuned by hand. A reviewer can
reasonably ask whether the reported violation counts are an artifact of that
tuning. The answer has to be a curve: perturb one threshold at a time, re-monitor
the same demonstrations, and measure how far the verdicts drift from the
human-confirmed labels. If the committed default sits at the peak of that curve,
the configuration is not arbitrary.

ONE KNOB AT A TIME, not a full cross-product. 6 knobs x 4 values is 1296 cells at
~9 wall-clock minutes each; a one-at-a-time sweep is 17 off-default cells and
answers the question actually being asked ("is this value the right one"), not
"is this the global optimum of a 6-D space", which no amount of compute here
would establish anyway.

WHICH KNOBS. Chosen because they gate the properties that actually fire on the
training corpora (training_audit/agreement_sample.csv), not because they are the
easiest to vary:

  FORBIDDEN_CONTACT_TOLERANCE_FRAMES  gates BOTH rc_no_forbidden_contact (27 of
      the 100 annotated instances) and rc_raw_robot_contact_blocks_rte_grasp_
      until_sanitized (34) -- specs.py deliberately reuses the one constant for
      both, so it alone governs 61% of the sample. The dominant knob.
  GRASP_SLIP_LINEAR_THRESHOLD, GRASP_BILATERAL_MIN_CONTACT_BODIES,
  PERSISTENCE_FRAMES                  gate object_sync / object_grasped, hence
      rc_grasp_remains_synced_until_dropped (11).
  SKILL_ONSET_FRAMES, CLUTTER_THRESHOLD   gate the skill-onset instant that the
      *_preconditions_safe family reads (14).
  SETTLE_TIMEOUT_FRAMES               gates the two *_eventually_settles
      properties (3).

`suites` exists because the two predicate modules do not share every constant
(LIBERO has no PERSISTENCE_FRAMES or GRASP_SLIP_* at all -- see hp_override's
note on why an unknown name is a per-module no-op). Running a RoboCasa-only knob
over LIBERO would burn 140 episodes to reproduce the default bit-for-bit and,
worse, would report "this hyperparameter does not affect LIBERO" as a finding
when it is really an absence of the parameter.

THE EPISODE SUBSET (per the explicit request to sweep a subset rather than the
full corpora): every task in both suites, but only a few episodes each --
  * every episode holding an annotated instance, so no annotated label is lost;
  * plus episodes 0-2 of every task, so a cell that makes the monitor MORE
    sensitive is caught creating NEW violations in clean episodes. Without these
    the sweep could only ever see false negatives, and a threshold that flags
    everything would score perfectly.
Keeping all tasks (rather than fewer tasks x all episodes) is what preserves the
annotated set; scene/allow-list diversity lives across tasks, not episodes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

THIS_DIR = Path(__file__).parent
SWEEP_DIR = THIS_DIR / "hp_sweep"
SUBSET_PATH = SWEEP_DIR / "subset.json"
SAMPLE = THIS_DIR / "training_audit" / "agreement_sample.csv"

# Mirrors training_violation_audit.CORPORA -- the FINAL corpora only.
CORPORA = {
    "RoboCasa": THIS_DIR / "output" / "v31_2026-09-20_claude_branch_place_precondition_hygiene_removed",
    "LIBERO": THIS_DIR / "output" / "v40_2026-09-21_claude_branch_libero_consolidated",
}
BASE_EPISODES = (0, 1, 2)

BOTH = ("RoboCasa", "LIBERO")
RC_ONLY = ("RoboCasa",)

# name -> (default, [values to try], suites that have the constant)
KNOBS = {
    "FORBIDDEN_CONTACT_TOLERANCE_FRAMES": (20, [5, 10, 40, 80], BOTH),
    "SKILL_ONSET_FRAMES": (10, [5, 20, 40], BOTH),
    "SETTLE_TIMEOUT_FRAMES": (100, [50, 200], BOTH),
    "CLUTTER_THRESHOLD": (2, [1, 4], BOTH),
    "GRASP_BILATERAL_MIN_CONTACT_BODIES": (2, [1, 3], BOTH),
    "PERSISTENCE_FRAMES": (5, [1, 3, 10], RC_ONLY),
    "GRASP_SLIP_LINEAR_THRESHOLD": (0.03, [0.01, 0.06, 0.12], RC_ONLY),
}

DEFAULT_CELL = "default"


def cells():
    """[{name, env, suites}] -- the default first, then one cell per off-default
    value. The default cell sets NO environment variables: it must exercise the
    committed constants themselves, so that 'default' is literally the shipped
    code and not a re-specification of it that could silently disagree."""
    out = [{"name": DEFAULT_CELL, "env": {}, "suites": BOTH,
            "knob": None, "value": None}]
    for knob, (_dflt, values, suites) in KNOBS.items():
        for v in values:
            out.append({"name": f"{knob}={v}", "env": {knob: str(v)},
                        "suites": suites, "knob": knob, "value": v})
    return out


def build_subset():
    """Write {suite: {task: [episode, ...]}} and return it."""
    annotated = {}
    for r in csv.DictReader(open(SAMPLE, encoding="utf-8")):
        annotated.setdefault(r["suite"], {}).setdefault(r["task"], set()).add(
            int(r["episode"]))
    subset = {}
    for suite, root in CORPORA.items():
        per_task = {}
        for task_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            have = {int(p.name[len("privileged_information_"):-len(".json")])
                    for p in task_dir.glob("privileged_information_*.json")
                    if not p.name.endswith("_monitor.json")}
            eps = (annotated.get(suite, {}).get(task_dir.name, set())
                   | set(BASE_EPISODES)) & have
            if eps:
                per_task[task_dir.name] = sorted(eps)
        subset[suite] = per_task
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_PATH.write_text(json.dumps(subset, indent=2, sort_keys=True))
    return subset


def load_subset():
    return json.loads(SUBSET_PATH.read_text())


if __name__ == "__main__":
    sub = build_subset()
    n_cells = len(cells())
    print(f"[json] {SUBSET_PATH}")
    tot = 0
    for suite, per_task in sub.items():
        n = sum(len(v) for v in per_task.values())
        tot += n
        print(f"  {suite:9s} {len(per_task):3d} tasks, {n:4d} episodes")
    print(f"  total     {tot} episodes/cell (max), {n_cells} cells")
    for c in cells():
        print(f"    {c['name']:45s} suites={','.join(c['suites'])}")
