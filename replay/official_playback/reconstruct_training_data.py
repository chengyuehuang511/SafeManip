#!/usr/bin/env python3
"""
Batch-reconstruct N episodes of one task from the official RoboCasa lerobot
training dataset (see ../README.md / README.md for why this is exact ground-
truth replay, unlike ../privileged_info_reconstruction's pose-guessing
approach), writing one mp4 per episode with all requested camera views
concatenated side-by-side (matching what the viewer's "Training Data" tab
expects, and what RoboCasa's own playback_dataset.py does for a single
combined video, adapted here to (a) loop over multiple tasks/episodes and
(b) write a separate video per episode instead of one continuous video for
the whole run).

Ground-truth state loading: `env.sim.set_state_from_flattened(states[t]);
env.sim.forward()` per frame, no physics stepping, no drift possible --
these are literally the states the training video was rendered from.

Usage:
    python3 reconstruct_training_data.py --task ArrangeBreadBasket \
        [--dataset_root ~/flash/datasets/robocasa/v1.0/target] \
        [--output_root output] [--n_episodes 10] [--video_skip 2] \
        [--camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand] \
        [--camera_height 256] [--camera_width 256] [--skip_existing]

Must run inside the `robocasa` conda env, on a GPU node (MUJOCO_GL=egl) --
see run_reconstruct_training_data.sbatch.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _desanitize_sys_path():
    """See ../privileged_info_reconstruction/reconstruct_video.py's docstring:
    drop cwd/'' from sys.path so `import robocasa`/`import gr00t` can't
    resolve to an empty namespace package shadowing the real editable
    install."""
    import os
    cwd = os.getcwd()
    sys.path[:] = [p for p in sys.path if p not in ("", cwd)]


_desanitize_sys_path()

import imageio  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_DATASET_ROOT = "~/flash/datasets/robocasa/v1.0/target"
DEFAULT_CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def find_dataset_dir(dataset_root, task):
    """Locate <dataset_root>/{composite,atomic}/<task>/<date>/lerobot. Each
    task has exactly one date subfolder (confirmed for all 50 target tasks)."""
    dataset_root = Path(dataset_root).expanduser()
    for category in ("composite", "atomic"):
        task_dir = dataset_root / category / task
        if not task_dir.is_dir():
            continue
        dates = sorted(p for p in task_dir.iterdir() if p.is_dir())
        for date_dir in dates:
            candidate = date_dir / "lerobot"
            if (candidate / "extras" / "dataset_meta.json").is_file():
                return candidate
    raise FileNotFoundError(f"no lerobot dataset found for task {task!r} under {dataset_root}")


def make_env(dataset_dir):
    import robosuite
    import robocasa.utils.lerobot_utils as LU

    env_meta = LU.get_env_metadata(dataset_dir)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = True
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def reconstruct_episode(env, dataset_dir, ep_num, out_path, camera_names,
                         camera_height, camera_width, video_skip):
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    states = LU.get_episode_states(dataset_dir, ep_num)
    initial_state = dict(states=states[0])
    initial_state["model"] = LU.get_episode_model_xml(dataset_dir, ep_num)
    ep_meta = LU.get_episode_meta(dataset_dir, ep_num)
    initial_state["ep_meta"] = json.dumps(ep_meta)

    reset_to(env, initial_state)

    # source states are recorded @20fps (see ../README.md); writing at
    # 20/video_skip keeps real-time duration correct when frames are skipped
    # (a fixed fps=20 here would play skipped-frame video back too fast).
    writer = imageio.get_writer(str(out_path), fps=20.0 / video_skip)
    try:
        traj_len = states.shape[0]
        for t in range(traj_len):
            reset_to(env, {"states": states[t]})
            if t % video_skip == 0 or t == traj_len - 1:
                frames = []
                for cam in camera_names:
                    im = env.sim.render(height=camera_height, width=camera_width, camera_name=cam)[::-1]
                    frames.append(im)
                writer.append_data(np.concatenate(frames, axis=1))
    finally:
        writer.close()

    return {"ep_num": ep_num, "n_frames": int(states.shape[0]), "lang": ep_meta.get("lang")}


def process_task(task, dataset_root, output_root, n_episodes, camera_names,
                  camera_height, camera_width, video_skip, skip_existing):
    dataset_dir = find_dataset_dir(dataset_root, task)
    out_dir = Path(output_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    import robocasa.utils.lerobot_utils as LU
    n_available = len(LU.get_episodes(dataset_dir))
    n = min(n_episodes, n_available)

    print(f"[{task}] dataset={dataset_dir} episodes={n}/{n_available}", flush=True)

    env = None
    summary = []
    for ep in range(n):
        out_path = out_dir / f"episode_{ep}_reconstructed.mp4"
        meta_path = out_dir / f"episode_{ep}_reconstruct_meta.json"
        if skip_existing and out_path.is_file() and meta_path.is_file():
            print(f"[{task}] episode {ep}: skip (already exists)", flush=True)
            summary.append({"episode": ep, "status": "skipped"})
            continue

        t0 = time.time()
        try:
            if env is None:
                env = make_env(dataset_dir)
            report = reconstruct_episode(
                env, dataset_dir, ep, out_path, camera_names,
                camera_height, camera_width, video_skip,
            )
            report["camera_names"] = camera_names
            report["fps"] = 20.0 / video_skip
            report["status"] = "reconstructed"
            report["elapsed_s"] = round(time.time() - t0, 2)
            meta_path.write_text(json.dumps(report, indent=2))
            print(f"[{task}] episode {ep}: done in {report['elapsed_s']}s", flush=True)
            summary.append(report)
        except Exception as e:
            print(f"[{task}] episode {ep}: FAILED: {e}", flush=True)
            summary.append({
                "episode": ep, "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })
            # the env may be in a bad state after an exception; rebuild next episode
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            env = None

    if env is not None:
        env.close()

    (out_dir / "task_summary.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for s in summary if s.get("status") == "reconstructed")
    n_skip = sum(1 for s in summary if s.get("status") == "skipped")
    print(f"[{task}] finished: {n_ok} reconstructed, {n_skip} skipped, "
          f"{n - n_ok - n_skip} failed", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--output_root", default=str(Path(__file__).parent / "output"))
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--camera_names", nargs="+", default=DEFAULT_CAMERAS)
    ap.add_argument("--camera_height", type=int, default=256)
    ap.add_argument("--camera_width", type=int, default=256)
    ap.add_argument("--video_skip", type=int, default=2,
                     help="render every Nth frame (source states are @20fps; skip=2 -> 10fps output)")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    process_task(
        args.task, args.dataset_root, args.output_root, args.n_episodes,
        args.camera_names, args.camera_height, args.camera_width,
        args.video_skip, args.skip_existing,
    )


if __name__ == "__main__":
    main()
