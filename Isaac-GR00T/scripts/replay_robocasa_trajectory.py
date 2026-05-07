"""
Replay a saved RoboCasa rollout trajectory from a local replay package.

Example usage:
    python scripts/replay_robocasa_trajectory.py \
      --replay_json logs/eval/rollout_data/TurnOffMicrowave--<DATE_TIME>/replay_package_0.json \
      --save_video \
      --save_privileged_info \
      --render_privileged_video
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_eval import (  # noqa: E402
    ROLLOUT_CAMERA_NAMES,
    add_frame_number_overlay,
    apply_model_arrays,
    bootstrap_local_robocasa_imports,
    capture_model_signature,
    compare_model_signatures,
    compose_rollout_frame,
    create_raw_eval_env,
    default_monitor_output_path,
    open_rollout_video_writer,
    prepare_rollout_observation_from_raw,
    render_privileged_video,
    reset_env_from_exact_model_xml,
    restore_frame_sim_state,
    rollout_video_output_path,
    run_symbolic_monitor,
    save_privileged_payload,
    save_rollout_video,
    strip_ltl_monitor,
    to_json_serializable,
)

bootstrap_local_robocasa_imports()


def _default_npz_path(replay_json: Path) -> Path:
    if replay_json.name.startswith("replay_package_") and replay_json.suffix == ".json":
        return replay_json.with_suffix(".npz")
    raise ValueError(f"Cannot infer replay npz path from {replay_json}")


def _default_privileged_output(replay_json: Path) -> Path:
    episode_suffix = replay_json.stem.replace("replay_package_", "")
    return replay_json.with_name(f"replayed_privileged_information_{episode_suffix}.json")


def _load_replay_package(
    replay_json: Path,
    replay_npz: Optional[Path] = None,
) -> Dict[str, Any]:
    with open(replay_json, "r") as f:
        meta = json.load(f)
    npz_path = replay_npz or _default_npz_path(replay_json)
    arrays = np.load(npz_path, allow_pickle=True)
    return dict(meta=meta, arrays=arrays, npz_path=npz_path)


def _load_model_xml(replay_json: Path, meta: Dict[str, Any]) -> Optional[str]:
    model_xml = meta.get("model_xml")
    if not model_xml:
        return None
    model_xml_path = Path(model_xml)
    if not model_xml_path.is_absolute():
        model_xml_path = replay_json.parent / model_xml_path
    with gzip.open(model_xml_path, "rt", encoding="utf-8") as f:
        return f.read()


def _load_model_arrays(arrays) -> Dict[str, np.ndarray]:
    model_arrays: Dict[str, np.ndarray] = {}
    for key in arrays.files:
        if key.startswith("model_array__"):
            model_arrays[key.removeprefix("model_array__")] = np.asarray(
                arrays[key],
                dtype=np.float64,
            )
    return model_arrays


def _restore_sim_state(sim, arrays) -> None:
    if "initial_state__flattened" in arrays:
        try:
            sim.set_state_from_flattened(arrays["initial_state__flattened"])
            sim.forward()
            return
        except Exception:
            pass
    if "initial_state__qpos" in arrays:
        sim.data.qpos[:] = arrays["initial_state__qpos"]
    if "initial_state__qvel" in arrays:
        sim.data.qvel[:] = arrays["initial_state__qvel"]
    if "initial_state__ctrl" in arrays and getattr(sim.data, "ctrl", None) is not None:
        sim.data.ctrl[:] = arrays["initial_state__ctrl"]
    if "initial_state__mocap_pos" in arrays and getattr(sim.data, "mocap_pos", None) is not None:
        sim.data.mocap_pos[:] = arrays["initial_state__mocap_pos"]
    if "initial_state__mocap_quat" in arrays and getattr(sim.data, "mocap_quat", None) is not None:
        sim.data.mocap_quat[:] = arrays["initial_state__mocap_quat"]
    if "initial_state__act" in arrays and getattr(sim.data, "act", None) is not None:
        try:
            sim.data.act[:] = arrays["initial_state__act"]
        except Exception:
            pass
    if "initial_state__time" in arrays:
        try:
            sim.data.time = float(np.asarray(arrays["initial_state__time"]).reshape(-1)[0])
        except Exception:
            pass
    sim.forward()


def _collect_privileged(env, trajectory_horizon: int):
    if not hasattr(env, "get_privileged_information"):
        raise RuntimeError("Environment does not expose get_privileged_information()")
    return env.get_privileged_information(trajectory_horizon=trajectory_horizon)


def _make_env(meta: Dict[str, Any], split_override: Optional[str]) -> Any:
    episode_meta = meta.get("episode_meta") or {}
    layout_id = episode_meta.get("layout_id")
    style_id = episode_meta.get("style_id")
    env_kwargs = {}
    if layout_id is not None and style_id is not None:
        env_kwargs["layout_and_style_ids"] = [(int(layout_id), int(style_id))]
    if meta.get("seed") is not None:
        env_kwargs["seed"] = int(meta["seed"])
    split = split_override or meta.get("split", "target")
    env = create_raw_eval_env(env_name=meta["env_name"], split=split, **env_kwargs)
    if episode_meta:
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(dict(episode_meta))
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(dict(episode_meta))
    return env


def _save_video_from_state_trajectory(
    env,
    arrays,
    output_dir: Path,
    episode_idx: int,
    success: bool,
    task_description: str,
    date_time: str,
) -> Path:
    output_path = rollout_video_output_path(
        output_dir=output_dir,
        episode_idx=episode_idx,
        success=success,
        task_description=task_description,
        date_time=date_time,
    )
    writer = open_rollout_video_writer(output_path)
    try:
        state_trajectory = arrays["state_trajectory__flattened"]
        for frame_idx, flattened_state in enumerate(state_trajectory):
            frame_state = {"flattened": flattened_state}
            for key in ("ctrl", "mocap_pos", "mocap_quat", "act"):
                array_key = f"state_trajectory__{key}"
                if array_key in arrays:
                    frame_state[key] = arrays[array_key][frame_idx]
            restore_frame_sim_state(env.sim, frame_state)
            frames = []
            for camera_name in ROLLOUT_CAMERA_NAMES:
                image = env.sim.render(
                    width=256,
                    height=256,
                    camera_name=camera_name,
                )[::-1]
                frames.append(np.asarray(image, dtype=np.uint8).copy())
            writer.append_data(add_frame_number_overlay(np.concatenate(frames, axis=1), frame_idx))
    finally:
        writer.close()
    return output_path


def replay_trajectory(
    replay_json: Path,
    replay_npz: Optional[Path],
    save_video: bool,
    save_privileged_info: bool,
    privileged_output: Optional[Path],
    trajectory_horizon: int,
    render_privileged_video_flag: bool,
    split_override: Optional[str],
    settle_steps: int,
) -> Dict[str, Any]:
    package = _load_replay_package(replay_json, replay_npz)
    meta = package["meta"]
    arrays = package["arrays"]
    model_xml = _load_model_xml(replay_json, meta)
    model_arrays = _load_model_arrays(arrays)

    env = _make_env(meta, split_override=split_override)
    try:
        raw_obs = env.reset()
        if model_xml:
            reset_env_from_exact_model_xml(env, model_xml)
            apply_model_arrays(env.sim, model_arrays)
            raw_obs = (
                env.viewer._get_observations(force_update=True)
                if env.viewer_get_obs
                else env._get_observations(force_update=True)
            )
        model_mismatches = compare_model_signatures(
            meta.get("model_signature"),
            capture_model_signature(env.sim),
        )
        if model_mismatches:
            mismatch_text = "; ".join(model_mismatches[:8])
            if len(model_mismatches) > 8:
                mismatch_text += f"; ... {len(model_mismatches) - 8} more"
            raise RuntimeError(
                "Replay env does not match the saved rollout env: "
                f"{mismatch_text}. Refusing to replay this package."
            )
        zero_action = np.zeros(env.action_spec[0].shape, dtype=np.float32)
        for _ in range(max(0, settle_steps)):
            raw_obs, _, done, _ = env.step(zero_action)
            if done:
                break

        _restore_sim_state(env.sim, arrays)
        raw_obs = (
            env.viewer._get_observations(force_update=True)
            if env.viewer_get_obs
            else env._get_observations(force_update=True)
        )

        privileged_static_info = None
        privileged_dynamic_info = []
        rollout_frames = []
        state_trajectory = (
            arrays["state_trajectory__flattened"]
            if "state_trajectory__flattened" in arrays
            else None
        )
        if save_privileged_info:
            privileged_static_info = _collect_privileged(env, trajectory_horizon)["static"]

        actions = np.asarray(arrays["actions"], dtype=np.float32)
        success = False
        for step_idx, action in enumerate(actions):
            raw_obs, _, done, info = env.step(action)
            success = bool(env._check_success()) or success
            if save_video and state_trajectory is None:
                rollout_frames.append(
                    compose_rollout_frame(
                        prepare_rollout_observation_from_raw(raw_obs, flip_images=True)
                    )
                )
            if save_privileged_info:
                privileged_dynamic_info.append(
                    {
                        "step": int(step_idx),
                        "data": to_json_serializable(
                            strip_ltl_monitor(
                                _collect_privileged(env, trajectory_horizon)["dynamic"]
                            )
                        ),
                    }
                )
            if done:
                continue

        result = {
            "replay_json": str(replay_json),
            "replay_npz": str(package["npz_path"]),
            "task_name": meta.get("env_name"),
            "task_description": meta.get("task_description"),
            "split": meta.get("split", split_override or "target"),
            "expected_episode_length": int(meta.get("episode_length", actions.shape[0])),
            "replayed_episode_length": int(actions.shape[0]),
            "success": bool(success),
            "original_success": bool(meta.get("success", False)),
        }

        video_output = None
        if save_video and state_trajectory is not None:
            video_output = _save_video_from_state_trajectory(
                env=env,
                arrays=arrays,
                output_dir=replay_json.parent,
                episode_idx=int(meta.get("episode_idx", 0)),
                success=bool(meta.get("success", result["success"])),
                task_description=str(
                    meta.get("task_description", meta.get("env_name", "robocasa_replay"))
                ),
                date_time=str(meta.get("date_time", "replayed")),
            )
            result["video_output"] = str(video_output)
            result["video_source"] = "state_trajectory__flattened"
        elif save_video:
            video_output = save_rollout_video(
                frames=rollout_frames,
                output_dir=replay_json.parent,
                episode_idx=int(meta.get("episode_idx", 0)),
                success=result["success"],
                task_description=str(
                    meta.get("task_description", meta.get("env_name", "robocasa_replay"))
                ),
                date_time=str(meta.get("date_time", "replayed")),
            )
            if video_output is not None:
                result["video_output"] = str(video_output)

        if save_privileged_info:
            output_path = privileged_output or _default_privileged_output(replay_json)
            save_privileged_payload(
                output_path=output_path,
                privileged_static_info=privileged_static_info or {},
                privileged_dynamic_info=privileged_dynamic_info,
                replay_summary=result,
            )
            result["privileged_output"] = str(output_path)
            monitor_output = run_symbolic_monitor(
                output_path,
                output_path=default_monitor_output_path(output_path),
            )
            if monitor_output is not None:
                result["monitor_output"] = str(monitor_output)
            if render_privileged_video_flag:
                privileged_video = render_privileged_video(output_path, video_output)
                if privileged_video is not None:
                    result["privileged_video_output"] = str(privileged_video)
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved RoboCasa trajectory.")
    parser.add_argument(
        "--replay_json", type=Path, required=True, help="Path to replay_package_*.json"
    )
    parser.add_argument(
        "--replay_npz", type=Path, default=None, help="Optional path to replay_package_*.npz"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save a rollout MP4 in the replay package directory.",
    )
    parser.add_argument(
        "--save_privileged_info",
        action="store_true",
        help="Collect and save privileged info while replaying.",
    )
    parser.add_argument(
        "--privileged_output",
        type=Path,
        default=None,
        help="Output JSON path for replayed privileged info.",
    )
    parser.add_argument(
        "--trajectory_horizon",
        type=int,
        default=128,
        help="Trajectory horizon passed to get_privileged_information().",
    )
    parser.add_argument(
        "--render_privileged_video",
        action="store_true",
        help="Render a replayed privileged_information_*.mp4 after saving the JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["pretrain", "target"],
        help="Optional split override if the replay metadata does not include one.",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=10,
        help="Number of zero-action settle steps before restoring the saved simulator state.",
    )
    args = parser.parse_args()

    result = replay_trajectory(
        replay_json=args.replay_json,
        replay_npz=args.replay_npz,
        save_video=args.save_video,
        save_privileged_info=args.save_privileged_info,
        privileged_output=args.privileged_output,
        trajectory_horizon=int(args.trajectory_horizon),
        render_privileged_video_flag=args.render_privileged_video,
        split_override=args.split,
        settle_steps=int(args.settle_steps),
    )
    print(json.dumps(to_json_serializable(result), indent=2))


if __name__ == "__main__":
    main()
