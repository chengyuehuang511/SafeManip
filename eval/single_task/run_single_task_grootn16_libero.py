#!/usr/bin/env python3
"""Single-task LIBERO eval against the PRISTINE eval/models/grootn16
submodule, for a COMBINED (single checkpoint, all 4 suites) GR00T-N1.6
LIBERO checkpoint -- specifically 0xAnkitSingh/GR00T-N1.6-LIBERO (see
eval/EVAL_PROTOCOL_NOTES.md for provenance/confidence caveats: this is an
unofficial, low-download-count personal upload, not an NVIDIA/grootn16-team
release -- included for completeness; run last/lowest priority).

Reuses grootn16's own gr00t.eval.sim.LIBERO.libero_env.register_libero_envs
and gr00t.eval.rollout_policy.run_gr00t_sim_policy unmodified -- both
already ship in this submodule (unlike the pristine N1.5-era Isaac-GR00T
submodule, which has no LIBERO example at all -- see
run_single_task_groot_libero.py and libero_gr00t_env.py for that one).
No replay/video-capture wrapping (matches run_single_task_rldx1_libero.py's
reasoning: not requested for LIBERO).
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
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    env_name = f"libero_sim/{task}"

    _, episode_successes, _ = run_gr00t_sim_policy(
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
    ap.add_argument("--task", required=True, help="LIBERO task name, e.g. pick_up_the_...")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_episodes", type=int, required=True)
    ap.add_argument("--n_action_steps", type=int, default=8)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

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
