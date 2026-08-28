#!/usr/bin/env python3
"""
Second, independent way of handling the same granularity mismatch documented
in `extract_privileged_from_dataset.py`'s module docstring (read that first)
-- kept as a SEPARATE script, not a mode flag on that one, specifically so
its already-run, already-validated code path and output
(`SafeManip/monitor/output/`) are never touched or put at risk by this file.

`extract_privileged_from_dataset.py`'s `--call_stride` fix calls
`env.get_privileged_information()` on *every* raw frame (full temporal
resolution) and *scales* predicates.py's persistence/onset-detection
frame-count constants to compensate.

This script instead does the structurally closest thing to what a live
rollout actually does: it calls `get_privileged_information()` only every
`--sample_stride`-th raw frame (default 16, matching the `n_action_steps`
macro-step size a live rollout commonly uses) -- i.e. it skips *calls*, not
just skips *scaling* -- and applies **no constant scaling at all**.
Since calls are now naturally spaced `sample_stride` raw frames apart (the
same spacing a live rollout's macro-stepping produces), predicates.py's
frame-count thresholds are already correct as-is, by construction, with no
override needed. Every raw frame's exact ground-truth state is still loaded
via `reset_to()` regardless (no drift, no approximated physics) -- only
which of those exactly-replayed frames get a privileged-info snapshot
appended is stride-gated.

Trade-off vs. `extract_privileged_from_dataset.py`: this produces the same
sparse-in-time output a live rollout would (~sample_stride raw frames per
recorded frame), rather than a per-frame-dense record -- pick whichever
comparison point you need; both are kept side by side for exactly this
reason (the user explicitly asked to compare them, not replace one with the
other).

Output goes to a *separate* directory (default
`SafeManip/monitor/output_sampled/`) so it never collides with or
overwrites `extract_privileged_from_dataset.py`'s `output/` -- both sets of
results stay independently browsable (see viewer/server.py's method
selector for the Training Data tab's monitor panel).

Usage (single episode):
    python3 extract_privileged_from_dataset_sampled.py --task ArrangeBreadBasket \
        --episode 1 [--sample_stride 16] [--dataset_root ~/flash/datasets/robocasa/v1.0/target] \
        [--output_root output_sampled] [--trajectory_horizon 128] [--run_monitor]

Usage (range of episodes, one env reused across them):
    python3 extract_privileged_from_dataset_sampled.py --task ArrangeBreadBasket \
        --episode_start 0 --episode_end 9 --run_monitor

Must run inside the `robocasa` conda env, on a GPU node (MUJOCO_GL=egl) --
same requirement as extract_privileged_from_dataset.py.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Reused, unmodified, read-only imports from the original script -- these
# are the parts with zero behavioral difference between the two approaches
# (env construction, dataset lookup, accumulator bookkeeping, JSON
# serialization, and the monitor-evaluation call itself, which is only ever
# invoked here with the default call_stride=1, i.e. no scaling, matching
# this script's whole point of leaving thresholds untouched).
from extract_privileged_from_dataset import (  # noqa: E402
    _desanitize_sys_path,
    _reset_privileged_accumulators,
    _to_json_serializable,
    find_dataset_dir,
    make_env,
    run_monitor_on,
)

_desanitize_sys_path()

DEFAULT_DATASET_ROOT = "~/flash/datasets/robocasa/v1.0/target"
DEFAULT_OUTPUT_ROOT = THIS_DIR / "output_sampled"
DEFAULT_SAMPLE_STRIDE = 16


def extract_episode_sampled(env, dataset_dir, ep_num, trajectory_horizon, sample_stride=DEFAULT_SAMPLE_STRIDE):
    """Ground-truth-state-load every raw frame (`reset_to()` every
    iteration, no exceptions), but only call `get_privileged_information()`
    every `sample_stride`-th frame -- see module docstring. No constant
    scaling anywhere in this function; predicates.py's module-level
    constants are never touched, by design.

    Returns a dict with the same top-level schema as what
    Isaac-GR00T/gr00t/eval/simulation.py's `_save_privileged_record` writes:
    {"privileged_static_info", "privileged_dynamic_info", "replay_summary"}.
    """
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    states = LU.get_episode_states(dataset_dir, ep_num)
    model_xml = LU.get_episode_model_xml(dataset_dir, ep_num)
    ep_meta = LU.get_episode_meta(dataset_dir, ep_num)

    _reset_privileged_accumulators(env)
    initial_state = dict(states=states[0], model=model_xml, ep_meta=json.dumps(ep_meta))
    reset_to(env, initial_state)
    _reset_privileged_accumulators(env)

    sample_stride = max(1, int(sample_stride))
    static_info = None
    dynamic_frames = []
    traj_len = states.shape[0]
    for t in range(traj_len):
        # Every raw frame's exact ground-truth state is loaded regardless --
        # only whether we *call* get_privileged_information() on this frame
        # is stride-gated.
        reset_to(env, {"states": states[t]})
        if t % sample_stride != 0 and t != traj_len - 1:
            continue
        # See extract_privileged_from_dataset.py's module docstring
        # ("CRITICAL FIX") for why this is necessary regardless of stride:
        # env.timestep must strictly increase across calls or
        # predicates.py's persisted monitor_state resets every call. t+1
        # mirrors what a real env.step() would have left it at after step t.
        env.timestep = t + 1
        info = env.get_privileged_information(trajectory_horizon=trajectory_horizon)
        if static_info is None:
            static_info = _to_json_serializable(info["static"])
        dynamic_frames.append({
            "step": int(t),
            "data": _to_json_serializable(info["dynamic"]),
        })

    success = None
    try:
        success = bool(env._check_success())
    except Exception:
        pass

    return {
        "privileged_static_info": static_info,
        "privileged_dynamic_info": dynamic_frames,
        "replay_summary": {
            "task_name": env.__class__.__name__,
            "task_description": ep_meta.get("lang"),
            "seed": None,
            "split": "train",
            "episode_idx": int(ep_num),
            "replayed_episode_length": int(traj_len),
            "success": success,
            "source": "official_dataset_ground_truth_replay_sampled",
            "dataset_dir": str(dataset_dir),
            "sample_stride": sample_stride,
        },
    }


def process_task(task, dataset_root, output_root, episodes, trajectory_horizon,
                  run_monitor, skip_existing, sample_stride=DEFAULT_SAMPLE_STRIDE):
    dataset_dir = find_dataset_dir(dataset_root, task)
    out_dir = Path(output_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    import robocasa.utils.lerobot_utils as LU
    n_available = len(LU.get_episodes(dataset_dir))
    episodes = [ep for ep in episodes if ep < n_available]

    print(f"[{task}] dataset={dataset_dir} episodes={episodes} (of {n_available} available) "
          f"sample_stride={sample_stride} (no constant scaling)", flush=True)

    env = None
    summary = []
    for ep in episodes:
        out_path = out_dir / f"privileged_information_{ep}.json"
        monitor_out_path = out_dir / f"privileged_information_{ep}_monitor.json"
        if skip_existing and out_path.is_file() and (not run_monitor or monitor_out_path.is_file()):
            print(f"[{task}] episode {ep}: skip (already exists)", flush=True)
            summary.append({"episode": ep, "status": "skipped"})
            continue

        t0 = time.time()
        try:
            if env is None:
                env = make_env(dataset_dir)
            payload = extract_episode_sampled(env, dataset_dir, ep, trajectory_horizon, sample_stride=sample_stride)
            out_path.write_text(json.dumps(payload, indent=2))
            n_frames = len(payload["privileged_dynamic_info"])
            elapsed = round(time.time() - t0, 2)
            print(f"[{task}] episode {ep}: extracted {n_frames} frames in {elapsed}s -> {out_path}", flush=True)
            entry = {"episode": ep, "status": "extracted", "n_frames": n_frames, "elapsed_s": elapsed}

            if run_monitor:
                # call_stride left at its default (1) -- no scaling, matching
                # this script's whole point.
                monitor_result = run_monitor_on(out_path, monitor_out_path)
                entry["monitor_output"] = str(monitor_out_path)
                entry["num_violated_instances"] = monitor_result.get("num_violated_instances")
                entry["num_satisfied_instances"] = monitor_result.get("num_satisfied_instances")
                print(
                    f"[{task}] episode {ep}: monitor -> "
                    f"{entry['num_violated_instances']} violated / "
                    f"{entry['num_satisfied_instances']} satisfied instances "
                    f"(of {monitor_result.get('num_property_instances')} total)",
                    flush=True,
                )
            summary.append(entry)
        except Exception as e:
            print(f"[{task}] episode {ep}: FAILED: {e}", flush=True)
            summary.append({
                "episode": ep, "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            env = None

    if env is not None:
        env.close()

    (out_dir / "task_extract_summary.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for s in summary if s.get("status") == "extracted")
    n_skip = sum(1 for s in summary if s.get("status") == "skipped")
    print(f"[{task}] finished: {n_ok} extracted, {n_skip} skipped, "
          f"{len(episodes) - n_ok - n_skip} failed", flush=True)
    return summary


def _parse_episode_list(args):
    if args.episode is not None:
        return [args.episode]
    if args.episode_start is not None:
        end = args.episode_end if args.episode_end is not None else args.episode_start
        return list(range(args.episode_start, end + 1))
    return list(range(args.n_episodes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, help="single episode index")
    ap.add_argument("--episode_start", type=int, help="first episode index (inclusive) of a range")
    ap.add_argument("--episode_end", type=int, help="last episode index (inclusive) of a range; defaults to episode_start")
    ap.add_argument("--n_episodes", type=int, default=10,
                     help="used only if neither --episode nor --episode_start is given: episodes 0..n_episodes-1")
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--trajectory_horizon", type=int, default=128)
    ap.add_argument("--sample_stride", type=int, default=DEFAULT_SAMPLE_STRIDE,
                     help="call get_privileged_information() only every Nth raw frame (default 16, "
                          "matching the live pipeline's typical n_action_steps). No constant scaling "
                          "is applied -- thresholds are correct as-is once calls are naturally spaced "
                          "this far apart. State is still loaded every raw frame regardless.")
    ap.add_argument("--run_monitor", action="store_true", default=True)
    ap.add_argument("--no-run_monitor", dest="run_monitor", action="store_false")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    episodes = _parse_episode_list(args)
    process_task(
        args.task, args.dataset_root, args.output_root, episodes,
        args.trajectory_horizon, args.run_monitor, args.skip_existing,
        sample_stride=args.sample_stride,
    )


if __name__ == "__main__":
    main()
