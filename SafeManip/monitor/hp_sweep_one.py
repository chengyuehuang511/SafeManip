"""Re-monitor exactly one episode into an explicit destination path.

A one-episode entry point exists so the sweep driver can run each episode as its
own OS process. That is not an optimisation -- it is the only crash-tolerant
option here. A multiprocessing.Pool version deadlocked: two workers died mid-
episode (no OOM, no traceback -- the monitor pipeline shells out to MONA through
ltlf2dfa, so a native-level abort takes the whole worker with it), Pool silently
respawned them, and imap_unordered then waited forever for results that no longer
had an owner. With one process per episode a death is a non-zero exit code on one
episode, which the driver records and moves past.

SAFEMANIP_HP_* is read from the inherited environment at predicate-module import,
so the driver only has to set the variables in the child's env -- see
hp_override.py.

Usage (driven by hp_sweep_run.py, not by hand):
    python3 hp_sweep_one.py --priv <privileged_information_N.json> --dest <out.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--priv", required=True)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    from monitor.run_monitor_on_privileged import monitor_rollout

    summary = monitor_rollout(args.priv)
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write via a temp file in the same directory, then rename: a killed process
    # must not leave a half-written JSON that the scorer would read as real.
    # The temp name carries the pid: two drivers can legitimately be working the
    # same cell at once (a resubmit overlapping a still-draining SLURM array), and
    # a shared temp path would let them interleave writes into one file and then
    # rename the mixture into place as a valid-looking result.
    tmp = dest.with_suffix(f".json.partial.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    tmp.replace(dest)


if __name__ == "__main__":
    main()
