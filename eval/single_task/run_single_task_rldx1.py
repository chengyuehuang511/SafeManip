#!/usr/bin/env python3
"""Single-task RoboCasa eval against the PRISTINE eval/models/RLDX-1
submodule (robocasa-benchmark leaderboard's "RLDX-1" submission, an
NVIDIA Isaac-GR00T-derived codebase from RLWRLD), saving eval videos and
(optionally) a replayable rollout dataset in the same extras/ format used
everywhere else in this repo.

Mirrors run_single_task_grootn16.py exactly -- RLDX-1 ships the same kind of
ready-made single-task, N-episode, in-process eval entry point
(`rldx.eval.rollout_policy.run_rldx_sim_policy`, backed by
`RLDXPolicy`/`RLDXSimPolicyWrapper`), so the same "monkeypatch gymnasium.make
for the duration of the call, then call the library function unmodified"
technique applies. Nothing inside eval/models/RLDX-1 or eval/simulators/
robocasa is edited; PYTHONPATH must point at eval/simulators/robocasa (not
RLDX-1's own external_dependencies/robocasa submodule, left uninitialized)
so this reuses the same pristine robocasa checkout as every other model.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_capture import ReplayCapture, capture_env_kwargs  # noqa: E402
from video_capture import MultiCameraVideoCapture  # noqa: E402


def run_single_task(
    *,
    model_path: str,
    task: str,
    split: str,
    video_dir: str,
    n_episodes: int,
    n_action_steps: int,
    max_episode_steps: Optional[int],
    seed: int,
    save_replay: bool,
    replay_dir: Optional[str],
) -> Dict[str, Any]:
    import gymnasium as gym

    from rldx.eval.rollout_policy import run_rldx_sim_policy
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    env_name = f"robocasa/{task}"
    if max_episode_steps is None:
        # RLDX-1's own eval_robocasa365.sh uses a shorter, custom per-task
        # horizon table (task_sets.yaml) instead of robocasa's own
        # get_task_horizon() -- considered matching it exactly, but decided
        # this particular knob isn't worth chasing; using robocasa's own
        # per-task horizon here keeps this consistent with every other
        # model in this repo (grootn16's own run_eval.py also calls
        # get_task_horizon() directly, so this is the majority convention).
        max_episode_steps = get_task_horizon(task)

    replay_holder: Dict[str, ReplayCapture] = {}
    video_holder: Dict[str, MultiCameraVideoCapture] = {}
    original_make = gym.make

    def _patched_make(*args, **kwargs):
        if not save_replay:
            return original_make(*args, **kwargs)
        with capture_env_kwargs() as captured:
            gym_env = original_make(*args, **kwargs)
            env_kwargs = dict(captured)
        capture = ReplayCapture(gym_env, Path(replay_dir), env_kwargs)
        replay_holder["capture"] = capture
        video_capture = MultiCameraVideoCapture(
            capture.holder_obj.env, Path(replay_dir) / "lerobot"
        )
        capture.holder_obj.env = video_capture
        video_holder["capture"] = video_capture
        return gym_env

    gym.make = _patched_make
    try:
        _, episode_successes, _ = run_rldx_sim_policy(
            env_name=env_name,
            n_episodes=n_episodes,
            max_episode_steps=max_episode_steps,
            model_path=model_path,
            n_envs=1,
            n_action_steps=n_action_steps,
            video_dir=None,  # our own 3-camera capture replaces RLDX-1's own video wrapper
            seed=seed,
            robocasa_split=split,
        )
    finally:
        gym.make = original_make

    success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
    print(f"Success rate: {success_rate:.2f}")

    replay_dataset_dir = None
    if save_replay and "capture" in replay_holder:
        replay_dataset_dir = replay_holder["capture"].finalize()
        if replay_dataset_dir is not None:
            print(f"Replayable dataset written to: {replay_dataset_dir}")
            extras_dir = replay_dataset_dir / "extras"
            num_episodes = len(list(extras_dir.glob("episode_*")))
            n_videos = video_holder["capture"].finalize(extras_dir, num_episodes)
            print(f"Wrote 3-camera videos for {n_videos}/{num_episodes} episodes.")

    return {
        "task": task,
        "split": split,
        "num_episodes": len(episode_successes),
        "success_rate": success_rate,
        "video_dir": video_dir,
        "replay_dataset_dir": str(replay_dataset_dir) if replay_dataset_dir else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True, help="RoboCasa env/task name, e.g. CloseBlenderLid")
    ap.add_argument("--split", choices=["pretrain", "target"], required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--n_action_steps", type=int, default=16)
    ap.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Defaults to robocasa.utils.dataset_registry_utils.get_task_horizon(task).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--save_replay",
        action="store_true",
        help="Record rollouts via robosuite.wrappers.DataCollectionWrapper and "
        "convert them into a replayable extras/ dataset (see replay_capture.py).",
    )
    ap.add_argument(
        "--replay_dir",
        default=None,
        help="Where to write the replay dataset. Defaults to <video_dir>/replay.",
    )
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.replay_dir or str(video_dir / "replay")

    stats = run_single_task(
        model_path=args.model_path,
        task=args.task,
        split=args.split,
        video_dir=str(video_dir),
        n_episodes=args.n_episodes,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        save_replay=args.save_replay,
        replay_dir=replay_dir,
    )

    stats_path = video_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
