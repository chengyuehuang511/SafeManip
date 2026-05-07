import collections
import dataclasses
import json
import logging
import os
import pathlib
import re
import subprocess
import sys
from datetime import datetime
from typing import Any

import gymnasium as gym
import imageio
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    split: str = "target"
    task: str = "PickPlaceCounterToCabinet"
    num_trials: int = 10
    log_dir: str = "./logs/openpi_eval"
    seed: int = 42
    save_privileged_info: bool = True
    run_monitor: bool = True
    privileged_trajectory_horizon: int = 128
    privileged_record_interval: int = 8
    video_frame_interval: int = 1
    sceneflow_root: str = str(pathlib.Path(__file__).resolve().parents[3])


def _monitor_root(sceneflow_root: pathlib.Path) -> pathlib.Path:
    candidates = [
        sceneflow_root,
        sceneflow_root / "monitor",
        sceneflow_root / "SafeManip" / "monitor",
    ]
    for candidate in candidates:
        if (candidate / "run_monitor_on_privileged.py").exists():
            return candidate
    return sceneflow_root / "SafeManip" / "monitor"


def _to_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _strip_ltl_monitor(dynamic_info: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = dict(dynamic_info or {})
    cleaned.pop("ltl_monitor", None)
    return cleaned


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text.strip())
    return slug.strip("_")[:120] or "task"


def _raw_env(env: Any) -> Any:
    gym_env = env.unwrapped if hasattr(env, "unwrapped") else env
    return getattr(gym_env, "env", gym_env)


def _new_privileged_record(env: Any, env_name: str, task_description: str, horizon: int) -> dict[str, Any]:
    raw = _raw_env(env)
    if hasattr(raw, "get_ep_meta"):
        task_description = str(raw.get_ep_meta().get("lang", task_description))
    static_info = {}
    if hasattr(raw, "get_privileged_information"):
        static_info = raw.get_privileged_information(trajectory_horizon=horizon).get("static", {})
    return {
        "task_description": task_description,
        "static": _to_json_serializable(static_info),
        "dynamic": [],
    }


def _append_privileged_record(env: Any, record: dict[str, Any], step_idx: int, horizon: int) -> None:
    raw = _raw_env(env)
    if not hasattr(raw, "get_privileged_information"):
        return
    dynamic = raw.get_privileged_information(trajectory_horizon=horizon).get("dynamic", {})
    record["dynamic"].append(
        {
            "step": int(step_idx),
            "data": _to_json_serializable(_strip_ltl_monitor(dynamic)),
        }
    )


def _run_symbolic_monitor(sceneflow_root: pathlib.Path, privileged_json_path: pathlib.Path) -> pathlib.Path | None:
    monitor_script = _monitor_root(sceneflow_root) / "run_monitor_on_privileged.py"
    if not monitor_script.exists():
        print(f"Warning: monitor script not found: {monitor_script}")
        return None
    output_path = privileged_json_path.with_name(f"{privileged_json_path.stem}_monitor.json")
    completed = subprocess.run(
        [
            sys.executable,
            str(monitor_script),
            str(privileged_json_path),
            "--output",
            str(output_path),
        ],
        cwd=str(monitor_script.parents[1]),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"Warning: monitor failed for {privileged_json_path}: {completed.stderr.strip()}")
        return None
    return output_path


def _run_monitor_metrics(sceneflow_root: pathlib.Path, stats_path: pathlib.Path) -> None:
    metrics_script = _monitor_root(sceneflow_root) / "monitor_metrics.py"
    if not metrics_script.exists():
        print(f"Warning: monitor metrics script not found: {metrics_script}")
        return
    completed = subprocess.run(
        [sys.executable, str(metrics_script), str(stats_path)],
        cwd=str(metrics_script.parents[1]),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"Warning: monitor metrics failed for {stats_path}: {completed.stderr.strip()}")


def _save_privileged_payload(
    rollout_root: pathlib.Path,
    sceneflow_root: pathlib.Path,
    record: dict[str, Any],
    env_name: str,
    split: str,
    seed: int,
    episode_idx: int,
    episode_length: int,
    success: bool,
    run_monitor: bool,
) -> tuple[str, str | None]:
    output_path = rollout_root / f"privileged_information_{episode_idx}.json"
    payload = {
        "privileged_static_info": record["static"],
        "privileged_dynamic_info": record["dynamic"],
        "replay_summary": {
            "task_name": env_name,
            "task_description": record["task_description"],
            "seed": int(seed),
            "split": split,
            "episode_idx": int(episode_idx),
            "replayed_episode_length": int(episode_length),
            "success": bool(success),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    monitor_output = _run_symbolic_monitor(sceneflow_root, output_path) if run_monitor else None
    return str(output_path), str(monitor_output) if monitor_output is not None else None


def resolve_env_name(task: str) -> str:
    if task in TASK_SET_REGISTRY:
        raise ValueError(
            f"{task!r} is a task set, not a single RoboCasa task. "
            "This OpenPI helper expects one env name."
        )
    for _, env_names in TASK_SET_REGISTRY.items():
        if task in env_names:
            return task
    raise ValueError(f"Unknown RoboCasa task {task!r}. Expected one env name contained in TASK_SET_REGISTRY.")


def eval_single_task(args: Args) -> None:
    if args.privileged_record_interval < 1:
        raise ValueError("privileged_record_interval must be >= 1")
    if args.video_frame_interval < 1:
        raise ValueError("video_frame_interval must be >= 1")

    np.random.seed(args.seed)
    env_name = resolve_env_name(args.task)
    sceneflow_root = pathlib.Path(args.sceneflow_root)

    task_horizon = get_task_horizon(env_name)
    horizon = int(task_horizon * 1.5)
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    rollout_root = pathlib.Path(args.log_dir) / "evals" / args.split / env_name / "rollout_data" / f"{env_name}--{timestamp}"
    rollout_root.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = gym.make(f"robocasa/{env_name}", split=args.split, seed=args.seed)

    total_successes = 0
    privileged_outputs: list[str] = []
    monitor_outputs: list[str] = []

    for episode_idx in tqdm.tqdm(range(args.num_trials)):
        obs, _ = env.reset()
        task_lang = obs["annotation.human.task_description"]
        privileged_record = (
            _new_privileged_record(env, env_name, task_lang, args.privileged_trajectory_horizon)
            if args.save_privileged_info
            else None
        )
        action_plan = collections.deque()
        replay_images = []
        done = False
        t = 0

        while t < horizon:
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img_right = np.ascontiguousarray(obs["video.robot0_agentview_right"])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            img_right = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_right, args.resize_size, args.resize_size)
            )

            if not action_plan:
                state = np.concatenate(
                    (
                        obs["state.end_effector_position_relative"],
                        obs["state.end_effector_rotation_relative"],
                        obs["state.base_position"],
                        obs["state.base_rotation"],
                        obs["state.gripper_qpos"],
                    ),
                    axis=0,
                )
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/right_image": img_right,
                    "observation/state": state,
                    "prompt": task_lang,
                }
                action_chunk = client.infer(element)["actions"]
                assert len(action_chunk) >= args.replan_steps, (
                    f"Requested {args.replan_steps} replan steps, but policy returned {len(action_chunk)} actions."
                )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = convert_action(action_plan.popleft())
            obs, _, _, _, info = env.step(action)
            done = bool(info["success"])

            if privileged_record is not None and (
                t % args.privileged_record_interval == 0 or t == horizon - 1 or done
            ):
                _append_privileged_record(env, privileged_record, t, args.privileged_trajectory_horizon)

            replay_img = image_tools.convert_to_uint8(np.ascontiguousarray(env.render()))
            if t % args.video_frame_interval == 0 or t == horizon - 1 or done:
                replay_images.append(replay_img)
            if done:
                total_successes += 1
                break
            t += 1

        video_name = f"{timestamp}--episode={episode_idx}--success={done}--task={_slug(str(task_lang))}.mp4"
        imageio.mimwrite(rollout_root / video_name, [np.asarray(x) for x in replay_images], fps=20)

        if privileged_record is not None:
            privileged_output, monitor_output = _save_privileged_payload(
                rollout_root=rollout_root,
                sceneflow_root=sceneflow_root,
                record=privileged_record,
                env_name=env_name,
                split=args.split,
                seed=args.seed,
                episode_idx=episode_idx,
                episode_length=t + 1,
                success=done,
                run_monitor=args.run_monitor,
            )
            privileged_outputs.append(privileged_output)
            if monitor_output is not None:
                monitor_outputs.append(monitor_output)

    stats_path = rollout_root / "stats.json"
    stats = {
        "num_episodes": int(args.num_trials),
        "success_rate": float(total_successes) / float(args.num_trials),
        "rollout_root": str(rollout_root),
        "privileged_outputs": privileged_outputs,
        "monitor_outputs": monitor_outputs,
    }
    stats_path.write_text(json.dumps(stats, indent=4))
    if monitor_outputs:
        _run_monitor_metrics(sceneflow_root, stats_path)

    env.env.close()
    del env.env
    del env


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    eval_single_task(tyro.cli(Args))
