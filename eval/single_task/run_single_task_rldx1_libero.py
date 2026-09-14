#!/usr/bin/env python3
"""Single-task LIBERO eval against the PRISTINE eval/models/RLDX-1 submodule,
calling its own `rldx.eval.rollout_policy.run_rldx_sim_policy` unmodified
(the same ready-made single-task, N-episode, in-process eval entry point
used for the RoboCasa RLDX-1 runs -- see run_single_task_rldx1.py), just
pointed at a `libero_sim/<task>` env instead of `robocasa/<task>`.

Hyperparameters (n_action_steps=8, max_episode_steps=720, n_envs=1) match
RLDX-1's own official run_scripts/eval/libero/eval_libero.sh. n_episodes
itself is the one deliberate exception: RLDX-1's own script uses a
per-suite split (50 for libero_10, only 20 for spatial/object/goal), but
this project overrides it to a uniform 50 across all suites (matching
openpi's own num_trials_per_task=50 convention, and the same "n_episodes
always 50" override already applied to grootn16's RoboCasa sweep) -- see
eval/EVAL_PROTOCOL_NOTES.md.

No replay/video-capture wrapping here (unlike the RoboCasa RLDX-1 script) --
not requested for LIBERO, and LIBERO's env stack (register_libero_envs())
hasn't been separately verified against replay_capture.py's
locate_env_holder() the way robocasa's has. RLDX-1's own eval_libero.sh
already writes its own per-episode rollout videos via `--video_dir`.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def run_single_task(
    *,
    model_path: str,
    task: str,
    video_dir: str,
    n_episodes: int,
    n_action_steps: int,
    max_episode_steps: int,
) -> Dict[str, Any]:
    from rldx.eval.rollout_policy import run_rldx_sim_policy

    env_name = f"libero_sim/{task}"

    _, episode_successes, _ = run_rldx_sim_policy(
        env_name=env_name,
        n_episodes=n_episodes,
        max_episode_steps=max_episode_steps,
        model_path=model_path,
        n_envs=1,
        n_action_steps=n_action_steps,
        video_dir=video_dir,
    )

    success_rate = float(sum(episode_successes)) / len(episode_successes) if episode_successes else 0.0
    print(f"Success rate: {success_rate:.2f}")

    return {
        "task": task,
        "num_episodes": len(episode_successes),
        "success_rate": success_rate,
        "video_dir": video_dir,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True, help="LIBERO task id, e.g. libero_sim/pick_up_the_...")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_episodes", type=int, required=True)
    ap.add_argument("--n_action_steps", type=int, default=8)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # `--task` is passed with the "libero_sim/" prefix already included (to
    # mirror eval_libero.sh's own ALL_TASKS array), so strip it back off
    # since run_single_task re-adds it.
    task = args.task
    if task.startswith("libero_sim/"):
        task = task[len("libero_sim/") :]

    stats = run_single_task(
        model_path=args.model_path,
        task=task,
        video_dir=str(video_dir),
        n_episodes=args.n_episodes,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
    )

    stats_path = video_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
