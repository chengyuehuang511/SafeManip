#!/usr/bin/env python3
"""Validate that a fixed-scene rollout produced identical scenes across episodes.

Compares every privileged_information_<N>.json in a rollout_data directory
against episode 0 on the fields that define "same scene + same initial state":

  - scene_layout.layout_id / style_id   (which kitchen + style)
  - scene_layout.fixtures               (fixture set + poses)
  - scene_layout.objects                (manipulated-object set + initial placement)
  - first dynamic frame                 (initial simulator state, t=0)

Usage:
  python3 validate_identical_scene.py <rollout_data_dir_or_glob>
  # e.g.
  python3 validate_identical_scene.py \
    results/groot_identical/evals/target/PackIdenticalLunches/rollout_data/PackIdenticalLunches--*/

A small float tolerance is allowed (sampling is seed-locked, so values should
match exactly, but GPU/serialization noise is tolerated).
"""

import glob
import json
import math
import sys
from pathlib import Path

TOL = 1e-9


def deep_equal(a, b, tol=TOL, path=""):
    """Recursive compare; returns (equal, first_mismatch_path)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            only = set(a.keys()) ^ set(b.keys())
            return False, f"{path}: differing keys {sorted(only)[:5]}"
        for k in a:
            ok, where = deep_equal(a[k], b[k], tol, f"{path}.{k}")
            if not ok:
                return False, where
        return True, ""
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"{path}: list len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            ok, where = deep_equal(x, y, tol, f"{path}[{i}]")
            if not ok:
                return False, where
        return True, ""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol):
            return True, ""
        return False, f"{path}: {a} != {b}"
    if a == b:
        return True, ""
    return False, f"{path}: {a!r} != {b!r}"


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fd(frame):
    return frame.get("data", frame)


def _timestep(frame):
    return _fd(frame).get("task", {}).get("timestep")


def object_poses_by_timestep(d):
    """Map task.timestep -> {object_name: position} for every recorded frame.

    The recorded dynamic window can start at different absolute timesteps across
    episodes of different length, so we key by timestep (not list index) and
    later compare only at the EARLIEST timestep both episodes share.
    """
    out = {}
    for frame in (d.get("privileged_dynamic_info") or []):
        t = _timestep(frame)
        objs = (_fd(frame).get("scene", {}) or {}).get("objects", {}) or {}
        out[t] = {
            name: (info.get("pose", {}) or {}).get("position")
            for name, info in objs.items()
            if isinstance(info, dict)
        }
    return out


def _max_pose_diff(pa, pb):
    diffs = [
        abs(x - y)
        for k in pa if pa.get(k) and pb.get(k)
        for x, y in zip(pa[k], pb[k])
    ]
    return max(diffs) if diffs else None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Accept a dir, a glob, or several.
    candidates = []
    for arg in sys.argv[1:]:
        candidates.extend(glob.glob(arg))
    dirs = [Path(c) for c in candidates if Path(c).is_dir()]
    if not dirs:
        print(f"ERROR: no directories matched {sys.argv[1:]}", file=sys.stderr)
        sys.exit(1)

    overall_ok = True
    for d in sorted(dirs):
        eps = sorted(
            p for p in d.glob("privileged_information_*.json")
            if not p.name.endswith("_monitor.json")
        )
        print(f"\n=== {d}  ({len(eps)} episodes) ===")
        if len(eps) < 2:
            print("  need >= 2 episodes to compare; skipping.")
            continue

        base = load(eps[0])
        base_scene = (base.get("privileged_static_info") or {}).get("scene_layout") or {}
        base_poses = object_poses_by_timestep(base)
        print(f"  episode 0: layout_id={base_scene.get('layout_id')}, "
              f"style_id={base_scene.get('style_id')}, "
              f"n_objects={len(base_scene.get('objects') or {})}")

        for p in eps[1:]:
            cur = load(p)
            scene = (cur.get("privileged_static_info") or {}).get("scene_layout") or {}

            # Hard verdict: the seed-determined scene. layout/style/fixtures and the
            # object set+config fully define the scene + initial placement.
            checks = []
            checks.append(("layout/style",
                           (base_scene.get("layout_id"), base_scene.get("style_id"))
                           == (scene.get("layout_id"), scene.get("style_id")), ""))
            ok_fix, w_fix = deep_equal(base_scene.get("fixtures"), scene.get("fixtures"))
            checks.append(("fixtures", ok_fix, w_fix))
            ok_obj, w_obj = deep_equal(base_scene.get("objects"), scene.get("objects"))
            checks.append(("object set+config", ok_obj, w_obj))

            ep_ok = all(c[1] for c in checks)
            overall_ok &= ep_ok
            print(f"  {p.name} vs episode 0: "
                  f"{'IDENTICAL SCENE' if ep_ok else 'DIFFERENT SCENE'}")
            for name, ok, where in checks:
                mark = "ok " if ok else "XX "
                detail = "" if ok else f"  ->first diff at {where}"
                print(f"      [{mark}] {name}{detail}")

            # Informational: object poses at the earliest shared timestep should
            # match closely; later frames legitimately diverge (policy stochasticity).
            cur_poses = object_poses_by_timestep(cur)
            common = sorted(t for t in (set(base_poses) & set(cur_poses)) if t is not None)
            if common:
                t0 = common[0]
                d0 = _max_pose_diff(base_poses[t0], cur_poses[t0])
                tail = common[-1]
                dt = _max_pose_diff(base_poses[tail], cur_poses[tail])
                d0s = f"{d0:.2e}" if d0 is not None else "n/a"
                dts = f"{dt:.2e}" if dt is not None else "n/a"
                print(f"      [i  ] object-pose max|diff|: t={t0} -> {d0s} "
                      f"(start), t={tail} -> {dts} (end; divergence here is expected)")

    print("\n" + ("PASS: all episodes share an identical scene + initial state."
                  if overall_ok else
                  "FAIL: some episodes differ (see XX lines above)."))
    sys.exit(0 if overall_ok else 2)


if __name__ == "__main__":
    main()
