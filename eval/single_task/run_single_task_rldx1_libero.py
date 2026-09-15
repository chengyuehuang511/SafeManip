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

Optional replay capture (`--save_replay`), producing LIBERO's own native
training-data hdf5 format (`data/demo_N/{states,actions,model_file}` --
NOT robocasa's extras/lerobot-style reformatting, since the goal is for
saved LIBERO eval replays to look exactly like LIBERO's own training data,
matched via LIBERO's own `gather_demonstrations_as_hdf5`, not a RoboCasa
convention). See replay_capture.py's `finalize_libero_replay_dir` and
`locate_env_holder` (extended to also recognize LIBERO's `LiberoEnv`,
which stores its raw env at `self._env` rather than RoboCasaGymEnv's
`self.env`) for the full mechanism.

When `--save_replay` is set, this also replaces RLDX-1's own native
per-episode video-saving (`--video_dir`, disabled via `video_dir=None`)
with our own `MultiCameraVideoCapture`, wrapped at the exact same level as
`DataCollectionWrapper` -- guarantees frame-count parity between the
replay hdf5 and the saved video by construction, unlike RLDX-1's own video
wrapper, which was confirmed (via a real reconstruction test) to record at
a different frequency than the replay states (~2:1 states:video-frames).
Without `--save_replay`, RLDX-1's own native video-saving is left
untouched.

`--no-strict` (patched in from outside, see `_patch_create_rldx_sim_policy_no_strict`)
-------------------------------------------------------------------------------------
RLDX-1's own official run_scripts/eval/libero/eval_libero.sh passes
`--no-strict` to run_rldx_server.py, which sets `strict=False` on the
`RLDXSimPolicyWrapper` it constructs. This isn't cosmetic: with
`strict=True` (the default), `RLDXSimPolicyWrapper.check_action` -- called
unconditionally after every `_get_action()` -- validates the *flat sim*
action dict against the *raw model* modality keys (`eef_pos_delta`,
`eef_rot_delta`, `gripper_close`), not the LIBERO-remapped
`action.x/y/z/roll/pitch/yaw/gripper` keys that `_get_action`'s own
`is_libero` branch just produced a few lines above -- so it always raises
`AssertionError: Action key 'action.eef_pos_delta' must be in action` for
every LIBERO task/checkpoint, regardless of whether the model's actual
rollout is correct. This is a real gap in `check_action` (it has no
`is_libero` branch mirroring `_get_action`'s), which RLDX-1's own official
script routes around via `--no-strict` rather than fixing -- so we do the
same rather than patching their validation logic ourselves.

`rldx.eval.rollout_policy.run_rldx_sim_policy` (the entry point this script
reuses unmodified) calls `create_rldx_sim_policy(...)`, which hardcodes
`RLDXSimPolicyWrapper(RLDXPolicy(...))` with no `strict` passthrough at all
(unlike run_rldx_server.py's own `--no-strict` flag). Since editing
eval/models/RLDX-1 itself is off-limits, this monkeypatches
`rollout_policy.create_rldx_sim_policy` from the outside, before calling
`run_rldx_sim_policy`, to construct the exact same objects with
`strict=False` -- matching the official script's behavior exactly, without
touching the submodule.
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


def _patch_create_rldx_sim_policy_no_strict() -> None:
    """See module docstring. Replaces rollout_policy.create_rldx_sim_policy
    (referenced by name, at call time, inside run_rldx_sim_policy -- so
    reassigning it here on the module object takes effect) with a version
    identical to the original except `strict=False` on RLDXSimPolicyWrapper,
    matching RLDX-1's own official eval_libero.sh --no-strict flag."""
    import rldx.eval.rollout_policy as _rollout_policy
    from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper

    def _create_rldx_sim_policy_no_strict(
        model_path, embodiment_tag, policy_client_host="", policy_client_port=None
    ):
        if policy_client_host and policy_client_port:
            from rldx.policy.server_client import PolicyClient

            return PolicyClient(host=policy_client_host, port=policy_client_port)
        return RLDXSimPolicyWrapper(
            RLDXPolicy(embodiment_tag=embodiment_tag, model_path=model_path, device=0),
            strict=False,
        )

    _rollout_policy.create_rldx_sim_policy = _create_rldx_sim_policy_no_strict


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

    _patch_create_rldx_sim_policy_no_strict()
    from rldx.eval.rollout_policy import run_rldx_sim_policy

    env_name = f"libero_sim/{task}"

    replay_holder: Dict[str, Any] = {}
    original_make = gym.make

    def _patched_make(*args, **kwargs):
        if not save_replay:
            return original_make(*args, **kwargs)
        gym_env = original_make(*args, **kwargs)
        # register_libero_envs() has already run by this point (it's the
        # first thing get_libero_env_fn's env_fn() does, before calling
        # gym.make -- see rldx/eval/rollout_policy.py), so the spec's own
        # registered kwargs (the same ones LiberoEnv.__init__ was
        # constructed with) are already available here rather than needing
        # to be recomputed independently.
        env_id = args[0] if args else kwargs.get("id")
        bddl_file = gym.spec(env_id).kwargs["task_bddl_file"]
        capture = ReplayCapture(gym_env, Path(replay_dir), env_kwargs={"bddl_file_name": bddl_file})
        replay_holder["capture"] = capture
        replay_holder["bddl_file"] = bddl_file
        # Replace RLDX-1's own native video-saving (disabled below via
        # video_dir=None) with our own MultiCameraVideoCapture, wrapped at
        # the exact same level as DataCollectionWrapper -- guarantees
        # frame-count parity between the replay hdf5 and the saved video by
        # construction (both driven by the same .step() calls), unlike
        # RLDX-1's own video wrapper, which recorded at a different
        # frequency than the replay states (confirmed empirically: roughly
        # a 2:1 states:video-frames ratio on a real run). LIBERO's own two
        # cameras (agentview, robot0_eye_in_hand -- see
        # eval/simulators/libero/libero/libero/envs/env_wrapper.py's
        # default camera_names), and double_flip=True to match LIBERO's
        # own 180-degree rotation convention (see video_capture.py).
        video_capture = MultiCameraVideoCapture(
            capture._wrapped,
            Path(replay_dir) / "videos",
            camera_names=("agentview", "robot0_eye_in_hand"),
            double_flip=True,
        )
        setattr(capture.holder_obj, capture.holder_attr, video_capture)
        replay_holder["video_capture"] = video_capture
        return gym_env

    # run_rldx_sim_policy never actually leaves video_dir=None in effect --
    # if given None it auto-generates one via
    # f"/tmp/.../{model_path.split('/')[-3]}_ac{...}_{uuid4()}" and always
    # passes SOMETHING into VideoConfig(video_dir=...), so RLDX-1's own
    # native video wrapper is never truly disabled (confirmed by reading
    # rollout_policy.py directly) -- passing None here only matters
    # insofar as it lets RLDX-1 pick that fallback path itself, which
    # crashes for any HF-repo-style model_path with fewer than 3 '/'
    # segments (confirmed by a real crash: `IndexError: list index out of
    # range` on "RLWRLD/RLDX-1-FT-LIBERO".split('/')[-3]). Since RLDX-1's
    # own native video output is functionally harmless to leave running
    # alongside our own MultiCameraVideoCapture (same working pattern
    # already used for RoboCasa in run_single_task_rldx1.py -- our capture
    # replaces the raw env reference before RLDX-1's own video wrapper
    # attaches on top, so its output is just an unused, ignorable /tmp
    # video, not a conflict), the fix is simply to give it a valid throwaway
    # path directly instead of relying on its own buggy auto-generation.
    rldx1_native_video_dir = (
        f"/tmp/{__import__('os').environ.get('USER', 'user')}/rldx1_libero_native_video_unused"
        if save_replay
        else video_dir
    )

    gym.make = _patched_make
    try:
        _, episode_successes, _ = run_rldx_sim_policy(
            env_name=env_name,
            n_episodes=n_episodes,
            max_episode_steps=max_episode_steps,
            model_path=model_path,
            n_envs=1,
            n_action_steps=n_action_steps,
            video_dir=rldx1_native_video_dir,
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

    if save_replay and "video_capture" in replay_holder:
        written = replay_holder["video_capture"].finalize(num_episodes=len(episode_successes))
        print(f"MultiCameraVideoCapture wrote {written} episode(s)' videos to {Path(replay_dir) / 'videos'}")

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
    ap.add_argument("--task", required=True, help="LIBERO task id, e.g. libero_sim/pick_up_the_...")
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

    # `--task` is passed with the "libero_sim/" prefix already included (to
    # mirror eval_libero.sh's own ALL_TASKS array), so strip it back off
    # since run_single_task re-adds it.
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
