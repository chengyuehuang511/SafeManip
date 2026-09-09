#!/usr/bin/env python3
"""
LIBERO analog of `extract_privileged_from_dataset.py`: reconstruct
`privileged_information_<N>.json` files (same schema:
{"privileged_static_info", "privileged_dynamic_info", "replay_summary"}) by
replaying the exact recorded MuJoCo `states` from LIBERO's own official demo
hdf5s (at `/srv/datasets/libero/<suite>/<task>_demo.hdf5` -- the real
robomimic-style LIBERO format, WITH full per-frame sim state, unlike the
lossy HF LeRobot conversion at ~/flash/datasets/libero/), so
`run_monitor_on_privileged.py` (unmodified) can evaluate the same 20
`TASK_AGNOSTIC_PROPERTY_SPECS` against real LIBERO training demonstrations.

TWO-ENVIRONMENT PIPELINE -- READ THIS BEFORE RUNNING
-----------------------------------------------------------------------
Unlike RoboCasa's extraction script (single `robocasa365` conda env for
everything), this script's extraction phase MUST run under the dedicated
`safemanip_libero` conda env (robosuite==1.4.0, matching LIBERO's own pin --
RoboCasa's robosuite==1.5.2 is API-incompatible with LIBERO's env classes,
confirmed during this integration's setup). But `run_monitor_on_privileged.py`
(which we must not modify) itself imports
`monitor.sim.robocasa.predicates.FORBIDDEN_CONTACT_TOLERANCE_FRAMES` at
module scope -- requiring `robocasa`/robosuite 1.5.2 to be importable, which
`safemanip_libero` deliberately does NOT have. So:

  - Extraction (this script's `--run_monitor` OFF, the default) runs under
    `safemanip_libero`'s python and only needs `libero`/robosuite 1.4.0.
  - The monitor phase (`--run_monitor`) is executed by *subprocess-invoking
    `robocasa365`'s python* to run `python3 -m monitor.run_monitor_on_privileged
    <path>` on each freshly-written JSON -- that JSON is plain data (no
    simulator objects), so any env that can import `monitor.run_monitor_on_privileged`
    (i.e. one with robocasa/robosuite installed) can process it, regardless of
    which env produced it.

Usage:
    # under safemanip_libero:
    LIBERO_CONFIG_PATH=.../libero/.libero_config \\
    PYTHONPATH=.../SafeManip/SafeManip:.../SafeManip \\
    /path/to/envs/safemanip_libero/bin/python3 \\
        extract_privileged_from_dataset_libero.py --suite libero_10 --all --run_monitor

    # single task file, first 10 demos:
    ... extract_privileged_from_dataset_libero.py \\
        --hdf5 /srv/datasets/libero/libero_10/KITCHEN_SCENE3_..._demo.hdf5 --n_demos 10
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _ensure_safemanip_on_syspath():
    repo_root = str(Path(__file__).resolve().parents[2])  # .../SafeManip
    monitor_pkg_root = str(Path(__file__).resolve().parents[1])  # .../SafeManip/SafeManip
    for p in (monitor_pkg_root, repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_safemanip_on_syspath()

import numpy as np  # noqa: E402

# /srv/datasets/libero is login-node-only (confirmed via srun: invisible
# from every SLURM compute node) -- fine as this module's own default for
# direct/interactive (non-SLURM) invocation on the login node, which is how
# this script is normally run directly, but the sbatch/submit wrapper
# scripts explicitly override --dataset_root to a confirmed
# compute-node-visible mirror (~/flash/datasets/libero_raw) for SLURM use --
# don't change this default without also checking those wrappers still pass
# their own override.
DEFAULT_DATASET_ROOT = "/srv/datasets/libero"
DEFAULT_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
LIBERO_REPO_ROOT = Path("/nethome/chuang475/testnvme/projects/SafeManip/libero")
THIS_DIR = Path(__file__).parent
DEFAULT_OUTPUT_ROOT = THIS_DIR / "output" / "v22_2026-09-08_libero_baseline"
ROBOCASA365_PYTHON = "/nethome/chuang475/testnvme/miniconda3/envs/robocasa/bin/python3"

_PRIVILEGED_ACCUMULATOR_ATTRS = (
    "_privileged_static_cache",
    "_libero_predicate_monitor_state",
)


def _to_json_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    return value


def _reset_privileged_accumulators(env):
    for attr in _PRIVILEGED_ACCUMULATOR_ATTRS:
        if hasattr(env, attr):
            delattr(env, attr)


def make_env(bddl_path):
    import monitor.sim.libero  # noqa: F401  (side effect: monkeypatch)
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=str(bddl_path),
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )


def extract_episode(env, task_name, task_description, demo_group, ep_num, dataset_dir):
    states = demo_group["states"][()]
    init_state = demo_group.attrs["init_state"]

    _reset_privileged_accumulators(env.env)
    env.reset()
    env.set_init_state(init_state)
    _reset_privileged_accumulators(env.env)

    static_info = None
    dynamic_frames = []
    traj_len = states.shape[0]
    for t in range(traj_len):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        # Mirrors robocasa's extraction script's "CRITICAL FIX": build_predicate_snapshot's
        # restart guard reads dynamic_info["task"]["timestep"] (== env.timestep) to detect
        # a fresh episode; since we never call env.step() here, it would otherwise stay
        # frozen at 0 and reset persisted monitor_state on every single frame.
        env.env.timestep = t + 1
        info = env.env.get_privileged_information()
        if static_info is None:
            static_info = _to_json_serializable(info["static"])
        dynamic_frames.append({"step": int(t), "data": _to_json_serializable(info["dynamic"])})

    success = None
    try:
        success = bool(env.env._check_success())
    except Exception:
        pass

    return {
        "privileged_static_info": static_info,
        "privileged_dynamic_info": dynamic_frames,
        "replay_summary": {
            "task_name": task_name,
            "task_description": task_description,
            "seed": None,
            "split": "train",
            "episode_idx": int(ep_num),
            "replayed_episode_length": int(traj_len),
            "success": success,
            "source": "libero_official_dataset_ground_truth_replay",
            "dataset_dir": str(dataset_dir),
            "call_stride": 1,
        },
    }


def run_monitor_subprocess(privileged_json_path):
    """Invoke robocasa365's python (which has robocasa/robosuite 1.5.2, hence
    `monitor.run_monitor_on_privileged` importable) as a subprocess to run the
    existing, unmodified CLI on our freshly-written JSON -- see module
    docstring's "TWO-ENVIRONMENT PIPELINE" section for why this can't be an
    in-process call from this (safemanip_libero) interpreter."""
    import os

    # NOTE: only PYTHONPATH-add monitor_pkg_root (.../SafeManip/SafeManip),
    # deliberately NOT the outer repo root too (which contains the vendored
    # `robocasa/` checkout as a sibling directory) -- adding both caused
    # `import robocasa` to resolve to a broken merged namespace package
    # instead of the real editable-installed one on this env's
    # site-packages, the same failure mode extract_privileged_from_dataset.py's
    # own `_desanitize_sys_path()` guards against (confirmed empirically
    # during this integration: identical TypeError
    # "<module 'robocasa'> is a built-in module" from robocasa's own
    # texture_swap.py's `inspect.getfile(robocasa)` call, which only a
    # regular non-namespace package satisfies).
    env_vars = dict(os.environ)
    monitor_pkg_root = str(Path(__file__).resolve().parents[1])
    env_vars["PYTHONPATH"] = monitor_pkg_root
    result = subprocess.run(
        [ROBOCASA365_PYTHON, "-m", "monitor.run_monitor_on_privileged", str(privileged_json_path)],
        cwd=monitor_pkg_root,
        env=env_vars,
        capture_output=True,
        text=True,
    )
    return result


def find_hdf5_for_task(task_name, dataset_root=DEFAULT_DATASET_ROOT, suites=DEFAULT_SUITES):
    """Locate <dataset_root>/<suite>/<task_name>_demo.hdf5 across the 4
    in-scope suite directories (LIBERO's task dirs = suite dirs, not
    robocasa's composite/atomic layout)."""
    dataset_root = Path(dataset_root)
    for suite in suites:
        candidate = dataset_root / suite / f"{task_name}_demo.hdf5"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no *_demo.hdf5 found for task {task_name!r} under {dataset_root}/{{{','.join(suites)}}}")


def process_task_file(hdf5_path, output_root, n_demos, run_monitor, skip_existing, episodes=None):
    import h5py

    hdf5_path = Path(hdf5_path)
    task_name = hdf5_path.stem.replace("_demo", "")
    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        bddl_file_name = data_grp.attrs["bddl_file_name"]
        bddl_path = LIBERO_REPO_ROOT / bddl_file_name
        env_args = json.loads(data_grp.attrs["env_args"])
        all_demo_names = sorted(
            (k for k in data_grp.keys() if k.startswith("demo_")),
            key=lambda s: int(s.split("_")[1]),
        )
        if episodes is not None:
            wanted = {int(e) for e in episodes}
            demo_names = [d for d in all_demo_names if int(d.split("_")[1]) in wanted]
        else:
            demo_names = all_demo_names[:n_demos]

        out_dir = Path(output_root) / task_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{task_name}] bddl={bddl_path} demos={demo_names}", flush=True)

        env = None
        summary = []
        try:
            for demo_name in demo_names:
                ep_num = int(demo_name.split("_")[1])
                out_path = out_dir / f"privileged_information_{ep_num}.json"
                monitor_out_path = out_dir / f"privileged_information_{ep_num}_monitor.json"
                if skip_existing and out_path.is_file() and (not run_monitor or monitor_out_path.is_file()):
                    print(f"[{task_name}] demo {ep_num}: skip (already exists)", flush=True)
                    summary.append({"episode": ep_num, "status": "skipped"})
                    continue

                t0 = time.time()
                try:
                    if env is None:
                        env = make_env(bddl_path)
                    task_description = env_args.get("problem_name", task_name)
                    try:
                        task_description = f.get("data").attrs.get("problem_info")
                        task_description = json.loads(task_description).get("language_instruction", task_name)
                    except Exception:
                        task_description = task_name

                    payload = extract_episode(
                        env, task_name, task_description, data_grp[demo_name], ep_num, hdf5_path
                    )
                    out_path.write_text(json.dumps(payload, indent=2))
                    n_frames = len(payload["privileged_dynamic_info"])
                    elapsed = round(time.time() - t0, 2)
                    print(f"[{task_name}] demo {ep_num}: extracted {n_frames} frames in {elapsed}s -> {out_path}", flush=True)
                    entry = {"episode": ep_num, "status": "extracted", "n_frames": n_frames, "elapsed_s": elapsed}

                    if run_monitor:
                        result = run_monitor_subprocess(out_path)
                        if result.returncode != 0:
                            print(f"[{task_name}] demo {ep_num}: monitor subprocess FAILED: {result.stderr[-2000:]}", flush=True)
                            entry["monitor_status"] = "error"
                            entry["monitor_stderr"] = result.stderr[-4000:]
                        else:
                            entry["monitor_status"] = "ok"
                            print(f"[{task_name}] demo {ep_num}: monitor -> {monitor_out_path}", flush=True)
                    summary.append(entry)
                except Exception as e:
                    print(f"[{task_name}] demo {ep_num}: FAILED: {e}", flush=True)
                    summary.append({
                        "episode": ep_num, "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    })
                    try:
                        if env is not None:
                            env.close()
                    except Exception:
                        pass
                    env = None
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        (out_dir / "task_extract_summary.json").write_text(json.dumps(summary, indent=2))
        n_ok = sum(1 for s in summary if s.get("status") == "extracted")
        n_skip = sum(1 for s in summary if s.get("status") == "skipped")
        print(f"[{task_name}] finished: {n_ok} extracted, {n_skip} skipped, "
              f"{len(demo_names) - n_ok - n_skip} failed", flush=True)
        return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hdf5", help="path to a single task's *_demo.hdf5 file")
    ap.add_argument("--task", help="task name (looked up by <dataset_root>/<suite>/<task>_demo.hdf5 across DEFAULT_SUITES) -- alternative to --hdf5, for per-episode SLURM array parity with extract_privileged_from_dataset.py's --task/--episode")
    ap.add_argument("--episode", type=int, help="single episode index (used with --task, one array-task-per-episode granularity)")
    ap.add_argument("--suite", help="one of libero_10/libero_goal/libero_object/libero_spatial (or any subdir name under --dataset_root)")
    ap.add_argument("--all", action="store_true", help="process every *_demo.hdf5 in --suite (or all DEFAULT_SUITES if --suite is omitted)")
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--n_demos", type=int, default=10)
    ap.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--run_monitor", action="store_true", default=False)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    episodes = [args.episode] if args.episode is not None else None
    if args.task:
        hdf5_files = [find_hdf5_for_task(args.task, dataset_root=dataset_root)]
    elif args.hdf5:
        hdf5_files = [Path(args.hdf5)]
    else:
        suites = [args.suite] if args.suite else list(DEFAULT_SUITES)
        hdf5_files = []
        for suite in suites:
            suite_dir = dataset_root / suite
            hdf5_files.extend(sorted(suite_dir.glob("*_demo.hdf5")))
        if not args.all and args.suite:
            raise SystemExit("pass --all to process every task in --suite, or use --hdf5/--task for a single file")

    print(f"Processing {len(hdf5_files)} task file(s)", flush=True)
    all_summaries = {}
    for hdf5_path in hdf5_files:
        summary = process_task_file(
            hdf5_path, args.output_root, args.n_demos, args.run_monitor, args.skip_existing, episodes=episodes
        )
        all_summaries[hdf5_path.stem] = summary

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_summary.json").write_text(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
