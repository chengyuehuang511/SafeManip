# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    PrivilegedInfoConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

DEFAULT_VIDEO_ROOT = Path(__file__).resolve().parents[1] / "videos"
SCENEFLOW_ROOT = Path(os.environ.get("SCENEFLOW_ROOT", Path(__file__).resolve().parents[2]))
ROLLOUT_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _to_json_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    return value


def _save_stats(stats_path: Path, stats: Dict[str, Any]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(_to_json_serializable(stats), f, indent=4)
    if stats.get("monitor_outputs"):
        _run_monitor_metrics(stats_path)


def _run_monitor_metrics(stats_path: Path) -> None:
    monitor_candidates = [
        SCENEFLOW_ROOT,
        SCENEFLOW_ROOT / "monitor",
        SCENEFLOW_ROOT / "SafeManip" / "monitor",
    ]
    monitor_root = next(
        (candidate for candidate in monitor_candidates if (candidate / "monitor_metrics.py").exists()),
        SCENEFLOW_ROOT / "SafeManip" / "monitor",
    )
    monitor_metrics_path = monitor_root / "monitor_metrics.py"
    if not monitor_metrics_path.exists():
        print(f"Warning: monitor_metrics.py not found: {monitor_metrics_path}")
        return
    completed = subprocess.run(
        [sys.executable, str(monitor_metrics_path), str(stats_path)],
        cwd=str(monitor_root.parent),
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"Warning: monitor_metrics.py failed for {stats_path}: {completed.stderr.strip()}")
        return
    if completed.stdout.strip():
        print(completed.stdout.strip())


def set_eval_seed(seed: Optional[int], *, context: str, log: bool = True) -> None:
    if seed is None:
        return
    if log:
        print(f"Setting eval seed for {context}: {seed}")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if env_flag("GR00T_SEED_PYTHON_RANDOM", True):
        random.seed(seed)
    if env_flag("GR00T_SEED_NUMPY", True):
        np.random.seed(seed)
    if env_flag("GR00T_SEED_TORCH", True):
        torch.manual_seed(seed)
    if env_flag("GR00T_SEED_CUDA", True) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = env_flag("GR00T_CUDNN_BENCHMARK", False)
    torch.backends.cudnn.deterministic = env_flag("GR00T_CUDNN_DETERMINISTIC", True)
    torch.backends.cuda.matmul.allow_tf32 = env_flag("GR00T_MATMUL_ALLOW_TF32", False)
    torch.backends.cudnn.allow_tf32 = env_flag("GR00T_CUDNN_ALLOW_TF32", False)
    torch.use_deterministic_algorithms(
        env_flag("GR00T_TORCH_DETERMINISTIC", True),
        warn_only=env_flag("GR00T_TORCH_DETERMINISTIC_WARN_ONLY", True),
    )


class DeterministicPolicyWrapper:
    def __init__(self, policy: Gr00tPolicy, seed: Optional[int]):
        self.policy = policy
        self.seed = seed
        self.action_call_count = 0
        self._lock = threading.Lock()

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self.seed is not None and env_flag("GR00T_PER_ACTION_SEED", True):
                set_eval_seed(
                    self.seed + self.action_call_count,
                    context="server action",
                    log=False,
                )
            self.action_call_count += 1
            return self.policy.get_action(observations)

    def get_modality_config(self) -> Dict[str, Any]:
        return self.policy.get_modality_config()


def run_server(
    data_config: str,
    model_path: str,
    embodiment_tag: str,
    port: int,
    seed: Optional[int] = None,
    ready_event: Optional[threading.Event] = None,
) -> None:
    set_eval_seed(seed, context="server")
    data_config_cls = DATA_CONFIG_MAP[data_config]
    modality_config = data_config_cls.modality_config()
    modality_transform = data_config_cls.transform()

    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=embodiment_tag,
        denoising_steps=4,
    )
    set_eval_seed(seed, context="server inference")

    server = RobotInferenceServer(DeterministicPolicyWrapper(policy, seed), port=port)
    if ready_event is not None:
        ready_event.set()
    server.run()


def start_server_thread(
    data_config: str,
    model_path: str,
    embodiment_tag: str,
    port: int,
    seed: Optional[int] = None,
) -> threading.Thread:
    ready_event = threading.Event()
    server_errors: List[BaseException] = []

    def server_target() -> None:
        try:
            run_server(
                data_config=data_config,
                model_path=model_path,
                embodiment_tag=embodiment_tag,
                port=port,
                seed=seed,
                ready_event=ready_event,
            )
        except BaseException as exc:
            server_errors.append(exc)
            ready_event.set()
            raise

    server_thread = threading.Thread(target=server_target, daemon=True)
    server_thread.start()
    ready_event.wait()
    if server_errors:
        raise RuntimeError("Inference server failed to start.") from server_errors[0]
    return server_thread


def _run_standard_eval(
    simulation_client: SimulationInferenceClient,
    env_name: str,
    split: str,
    video_dir: Path,
    n_episodes: int,
    n_envs: int,
    n_action_steps: int,
    seed: Optional[int],
    privileged: Optional[PrivilegedInfoConfig] = None,
    video_filename_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    horizon = get_task_horizon(env_name)
    config = SimulationConfig(
        env_name=f"robocasa/{env_name}",
        split=split,
        seed=seed,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(
            video_dir=str(video_dir),
            camera_names=ROLLOUT_CAMERA_NAMES,
            filename_prefix=video_filename_prefix,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=horizon,
        ),
    )
    if privileged is not None:
        config.privileged = privileged

    print(f"Running simulation for {env_name}...")
    _, episode_successes = simulation_client.run_simulation(config)
    stats = {
        "num_episodes": len(episode_successes),
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "seed": seed,
    }
    if privileged is not None and privileged.output_dir is not None:
        rollout_root = Path(privileged.output_dir)
        stats["rollout_root"] = str(rollout_root)
        stats["rollout_videos"] = sorted(str(path) for path in rollout_root.glob("*.mp4"))
        stats["privileged_outputs"] = sorted(
            str(path)
            for path in rollout_root.glob("privileged_information_*.json")
            if not path.name.endswith("_monitor.json")
        )
        stats["monitor_outputs"] = sorted(
            str(path) for path in rollout_root.glob("privileged_information_*_monitor.json")
        )
    return stats


def _run_privileged_eval(
    simulation_client: SimulationInferenceClient,
    env_name: str,
    split: str,
    env_output_dir: Path,
    n_episodes: int,
    n_action_steps: int,
    privileged_trajectory_horizon: int,
    save_replay_package_flag: bool,
    save_privileged_info_flag: bool,
    render_privileged_video_flag: bool,
    seed: Optional[int],
) -> Dict[str, Any]:
    if save_replay_package_flag or render_privileged_video_flag:
        print(
            "Warning: replay package / privileged video export is disabled in this "
            "minimal original-pipeline path. Saving privileged JSON only."
        )
    rollout_root = (
        env_output_dir
        / "rollout_data"
        / f"{env_name}--{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
    )
    date_time = rollout_root.name.split("--", 1)[1]
    return _run_standard_eval(
        simulation_client=simulation_client,
        env_name=env_name,
        split=split,
        video_dir=rollout_root,
        n_episodes=n_episodes,
        n_envs=1,
        n_action_steps=n_action_steps,
        seed=seed,
        privileged=PrivilegedInfoConfig(
            enabled=save_privileged_info_flag or render_privileged_video_flag,
            output_dir=str(rollout_root),
            trajectory_horizon=privileged_trajectory_horizon,
            run_monitor=save_privileged_info_flag,
        ),
        video_filename_prefix=date_time,
    )


def run_client(
    host: str,
    port: int,
    task_set_list: List[str],
    video_dir: str,
    split: str,
    n_episodes: int,
    n_envs: int,
    n_action_steps: int,
    save_replay_package_flag: bool = False,
    save_privileged_info_flag: bool = False,
    render_privileged_video_flag: bool = False,
    privileged_trajectory_horizon: int = 128,
    seed: Optional[int] = None,
) -> None:
    set_eval_seed(seed, context="client")
    simulation_client = SimulationInferenceClient(host=host, port=port)

    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())

    privileged_mode = (
        save_replay_package_flag or save_privileged_info_flag or render_privileged_video_flag
    )
    if privileged_mode and n_envs != 1:
        print(f"Privileged export requires n_envs=1; overriding {n_envs} -> 1.")
        n_envs = 1

    all_env_names = []
    for task_set in task_set_list:
        all_env_names += TASK_SET_REGISTRY[task_set]

    for env_name in sorted(set(all_env_names)):
        env_output_dir = Path(video_dir) / "evals" / split / env_name
        stats_path = env_output_dir / "stats.json"
        if not privileged_mode and stats_path.exists():
            print(f"{env_name} stats already exists. skipping.")
            continue

        try:
            if privileged_mode:
                stats = _run_privileged_eval(
                    simulation_client=simulation_client,
                    env_name=env_name,
                    split=split,
                    env_output_dir=env_output_dir,
                    n_episodes=n_episodes,
                    n_action_steps=n_action_steps,
                    privileged_trajectory_horizon=privileged_trajectory_horizon,
                    save_replay_package_flag=save_replay_package_flag,
                    save_privileged_info_flag=save_privileged_info_flag,
                    render_privileged_video_flag=render_privileged_video_flag,
                    seed=seed,
                )
            else:
                stats = _run_standard_eval(
                    simulation_client=simulation_client,
                    env_name=env_name,
                    split=split,
                    video_dir=env_output_dir,
                    n_episodes=n_episodes,
                    n_envs=n_envs,
                    n_action_steps=n_action_steps,
                    seed=seed,
                )
        except Exception as e:
            print("Exception!", e)
            continue

        print(f"Results for {env_name}:")
        print(f"Success rate: {stats['success_rate']:.2f}")
        if privileged_mode and stats.get("rollout_root"):
            stats_path = Path(stats["rollout_root"]) / "stats.json"
        _save_stats(stats_path, stats)
        print(f"saved stats to {stats_path}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="<PATH_TO_YOUR_MODEL>")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--data_config", type=str, default="panda_omron")
    parser.add_argument("--task_set", type=str, nargs="+", required=True)
    parser.add_argument("--split", type=str, choices=["pretrain", "target"], required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--video_dir", type=str, default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--n_envs", type=int, default=5)
    parser.add_argument("--n_action_steps", type=int, default=16)
    parser.add_argument("--save_replay_package", action="store_true")
    parser.add_argument("--save_privileged_info", action="store_true")
    parser.add_argument("--render_privileged_video", action="store_true")
    parser.add_argument("--privileged_trajectory_horizon", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    args = parser.parse_args()

    if args.server:
        run_server(
            data_config=args.data_config,
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            port=args.port,
            seed=args.seed,
        )
    elif args.client:
        run_client(
            host=args.host,
            port=args.port,
            task_set_list=args.task_set,
            video_dir=args.video_dir,
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            save_replay_package_flag=args.save_replay_package,
            save_privileged_info_flag=args.save_privileged_info,
            render_privileged_video_flag=args.render_privileged_video,
            privileged_trajectory_horizon=args.privileged_trajectory_horizon,
            seed=args.seed,
        )
    else:
        start_server_thread(
            data_config=args.data_config,
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            port=args.port,
            seed=args.seed,
        )
        run_client(
            host=args.host,
            port=args.port,
            task_set_list=args.task_set,
            video_dir=args.video_dir,
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            save_replay_package_flag=args.save_replay_package,
            save_privileged_info_flag=args.save_privileged_info,
            render_privileged_video_flag=args.render_privileged_video,
            privileged_trajectory_horizon=args.privileged_trajectory_horizon,
            seed=args.seed,
        )
