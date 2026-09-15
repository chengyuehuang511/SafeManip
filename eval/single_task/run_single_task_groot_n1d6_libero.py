#!/usr/bin/env python3
"""Single-task LIBERO eval against the PRISTINE eval/models/Isaac-GR00T_official_n1d6
submodule (official NVIDIA/Isaac-GR00T, branch n1d6 -- N1.6), for the
COMBINED (single checkpoint, all 4 suites) GR00T-N1.6 LIBERO checkpoint
0xAnkitSingh/GR00T-N1.6-LIBERO.

Per "LIBERO always uses official, RoboCasa uses the model's robocasa
fork" policy (see eval/EVAL_PROTOCOL_NOTES.md), this supersedes the
earlier run_single_task_grootn16_libero.py (which targeted the grootn16
fork) even though that fork also has working LIBERO support of its own --
consistency with openpi/RLDX-1's LIBERO setup takes priority.

n1d6's own gr00t.eval.sim.LIBERO.libero_env.register_libero_envs and
gr00t.eval.rollout_policy.{create_gr00t_sim_policy, run_rollout_gymnasium_policy}
are structurally identical to the grootn16 fork's (same function names,
same env id format `libero_sim/<task>`, confirmed via `diff` on
pyproject.toml torch/transformers pins -- no drift). The one difference:
n1d6's own `run_gr00t_sim_policy` convenience wrapper does NOT accept a
`video_dir` parameter (always auto-generates one under /tmp) -- so this
script calls the lower-level pieces (`create_gr00t_sim_policy`,
`WrapperConfigs`/`VideoConfig`/`MultiStepConfig`, `run_rollout_gymnasium_policy`)
directly instead of `run_gr00t_sim_policy`, mirroring that function's own
body exactly but with a controllable `video_dir` -- same principle as
openpi's run_libero_suite_openpi.py loading `main.py`'s `Args`/`eval_libero`
directly rather than going through a convenience CLI that doesn't expose
what's needed. Nothing in eval/models/Isaac-GR00T_official_n1d6 is edited.

Confirmed via reading gr00t/policy/gr00t_policy.py directly:
Gr00tSimPolicyWrapper._get_action/check_action have no LIBERO-specific
key-remapping branch (unlike RLDX-1's rldx_policy.py) -- both always use
the model's own raw modality keys uniformly, with no benchmark-specific
branching, so the RLDX-1-style check_action/_get_action key mismatch
cannot occur here by construction. No `--no-strict`-equivalent needed.

Optional replay capture (`--save_replay`), producing LIBERO's own native
training-data hdf5 format -- see run_single_task_rldx1_libero.py's module
docstring and replay_capture.py's `finalize_libero_replay_dir` for the full
mechanism (identical here: monkeypatch `gymnasium.make` for the duration of
the rollout call, wrap the returned env with `ReplayCapture`, then finalize
via LIBERO's own `gather_demonstrations_as_hdf5`). n1d6's own `LiberoEnv`
(gr00t/eval/sim/LIBERO/libero_env.py) stores its raw env at `self._env`,
same as RLDX-1's -- already handled by `locate_env_holder`'s existing
`_env`/`.env` check, no new gap here.

When `--save_replay` is set, this also replaces GR00T-N1.6's own native
`VideoConfig`/`VideoRecordingWrapper` video-saving (disabled via
`video_dir=None`) with our own `MultiCameraVideoCapture`, wrapped at the
exact same level as `DataCollectionWrapper` -- guarantees frame-count
parity between the replay hdf5 and the saved video by construction, unlike
GR00T-N1.6's own video wrapper (layered inside `MultiStepWrapper`), which
was confirmed (via a real reconstruction test) to record at a different
frequency than the replay states. Without `--save_replay`, GR00T-N1.6's
own native video-saving is left untouched. Same fix as
run_single_task_rldx1_libero.py's -- shares the same underlying
gr00t.eval.rollout_policy/robosuite codebase, so the same root cause and
fix apply identically.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_capture import ReplayCapture, finalize_libero_replay_dir  # noqa: E402
from video_capture import MultiCameraVideoCapture  # noqa: E402


def run_single_task(
    *,
    model_path: str,
    task: str,
    video_dir: str,
    n_episodes: int,
    n_action_steps: int,
    max_episode_steps: int,
    save_replay: bool = False,
    replay_dir: Optional[str] = None,
) -> Dict[str, Any]:
    import gymnasium as gym

    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        VideoConfig,
        WrapperConfigs,
        create_gr00t_sim_policy,
        get_embodiment_tag_from_env_name,
        run_rollout_gymnasium_policy,
    )
    from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

    register_libero_envs()
    env_name = f"libero_sim/{task}"
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=None if save_replay else video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(model_path, embodiment_tag)

    replay_holder: Dict[str, Any] = {}
    original_make = gym.make

    def _patched_make(*args, **kwargs):
        if not save_replay:
            return original_make(*args, **kwargs)
        gym_env = original_make(*args, **kwargs)
        # register_libero_envs() has already run by this point (it's the
        # first thing get_libero_env_fn's env_fn() does, before calling
        # gym.make -- see gr00t/eval/rollout_policy.py), so the spec's own
        # registered kwargs are already available here.
        env_id = args[0] if args else kwargs.get("id")
        bddl_file = gym.spec(env_id).kwargs["task_bddl_file"]
        capture = ReplayCapture(gym_env, Path(replay_dir), env_kwargs={"bddl_file_name": bddl_file})
        replay_holder["capture"] = capture
        replay_holder["bddl_file"] = bddl_file
        video_capture = MultiCameraVideoCapture(
            capture._wrapped,
            Path(replay_dir) / "videos",
            camera_names=("agentview", "robot0_eye_in_hand"),
            double_flip=True,
        )
        setattr(capture.holder_obj, capture.holder_attr, video_capture)
        replay_holder["video_capture"] = video_capture
        return gym_env

    gym.make = _patched_make
    try:
        _, episode_successes, _ = run_rollout_gymnasium_policy(
            env_name=env_name,
            policy=policy,
            wrapper_configs=wrapper_configs,
            n_episodes=n_episodes,
            n_envs=1,
        )
    finally:
        gym.make = original_make

    success_rate = float(sum(episode_successes)) / len(episode_successes) if episode_successes else 0.0
    print(f"Success rate: {success_rate:.2f}")

    stats = {
        "task": task,
        "num_episodes": len(episode_successes),
        "success_rate": success_rate,
        "video_dir": video_dir,
    }

    if save_replay and "capture" in replay_holder:
        hdf5_dir = Path(replay_dir) / "hdf5"
        hdf5_path = finalize_libero_replay_dir(
            replay_holder["capture"].raw_dir, hdf5_dir, replay_holder["bddl_file"]
        )
        if hdf5_path is not None:
            print(f"LIBERO-native replay hdf5 written to: {hdf5_path}")
            stats["replay_hdf5_path"] = str(hdf5_path)

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True, help="LIBERO task name, e.g. pick_up_the_...")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_episodes", type=int, required=True)
    ap.add_argument("--n_action_steps", type=int, default=8)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--save_replay", action="store_true")
    ap.add_argument(
        "--replay_dir",
        default=None,
        help="Defaults to <video_dir>/replay if --save_replay is set and this is omitted.",
    )
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    task = args.task
    if task.startswith("libero_sim/"):
        task = task[len("libero_sim/") :]

    replay_dir = args.replay_dir
    if args.save_replay and replay_dir is None:
        replay_dir = str(video_dir / "replay")
    if replay_dir is not None:
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

    stats = run_single_task(
        model_path=args.model_path,
        task=task,
        video_dir=str(video_dir),
        n_episodes=args.n_episodes,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        save_replay=args.save_replay,
        replay_dir=replay_dir,
    )

    stats_path = video_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
