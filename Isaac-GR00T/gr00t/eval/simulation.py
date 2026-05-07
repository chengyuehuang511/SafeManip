# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.model.policy import BasePolicy

# from gymnasium.envs.registration import registry

# print("Available environments:")
# for env_spec in registry.values():
#     print(env_spec.id)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    camera_names: Optional[Tuple[str, ...]] = None
    filename_prefix: Optional[str] = None


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    split: str = "test"
    seed: Optional[int] = None
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    privileged: "PrivilegedInfoConfig" = field(default_factory=lambda: PrivilegedInfoConfig())


@dataclass
class PrivilegedInfoConfig:
    """Optional passive privileged-info recording for RoboCasa evals."""

    enabled: bool = False
    output_dir: Optional[str] = None
    trajectory_horizon: int = 128
    run_monitor: bool = True


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


def _strip_ltl_monitor(dynamic_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(dynamic_info, dict):
        return {}
    cleaned = copy.deepcopy(dynamic_info)
    cleaned.pop("ltl_monitor", None)
    return cleaned


class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port)
        self.env = None

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the inference server based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the inference server."""
        return self.call_endpoint("get_modality_config", requires_input=False)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes."""
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        # Initial environment reset
        obs, _ = self.env.reset()
        privileged_records = self._init_privileged_records(config)
        # Main simulation loop
        while completed_episodes < config.n_episodes:
            self._ensure_privileged_records(config, privileged_records)
            # Process observations and get actions from the server
            actions = self._get_actions_from_server(obs)
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)
            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                self._append_privileged_record(
                    config=config,
                    records=privileged_records,
                    env_idx=env_idx,
                    step_idx=current_lengths[env_idx] - 1,
                )
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    self._save_privileged_record(
                        config=config,
                        records=privileged_records,
                        env_idx=env_idx,
                        episode_idx=len(episode_successes) - 1,
                        episode_length=current_lengths[env_idx],
                        success=current_successes[env_idx],
                    )
                    cumulative_sr = float(np.mean(episode_successes))
                    print(
                        f"EP {len(episode_successes)} success: {current_successes[env_idx]}; Cumulative success rate: {cumulative_sr}"
                    )
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            obs = next_obs
        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _init_privileged_records(self, config: SimulationConfig) -> List[Optional[Dict[str, Any]]]:
        if not config.privileged.enabled:
            return []
        if config.n_envs != 1:
            raise ValueError("Privileged recording currently supports n_envs=1.")
        if config.privileged.output_dir is None:
            raise ValueError("PrivilegedInfoConfig.output_dir is required when enabled.")
        Path(config.privileged.output_dir).mkdir(parents=True, exist_ok=True)
        return [self._new_privileged_record(config, env_idx) for env_idx in range(config.n_envs)]

    def _ensure_privileged_records(
        self,
        config: SimulationConfig,
        records: List[Optional[Dict[str, Any]]],
    ) -> None:
        if not config.privileged.enabled:
            return
        for env_idx, record in enumerate(records):
            if record is None:
                records[env_idx] = self._new_privileged_record(config, env_idx)

    def _new_privileged_record(self, config: SimulationConfig, env_idx: int) -> Dict[str, Any]:
        raw_env = self._get_raw_env(env_idx)
        task_description = config.env_name
        static_info = {}
        if raw_env is not None:
            if hasattr(raw_env, "get_ep_meta"):
                task_description = str(raw_env.get_ep_meta().get("lang", config.env_name))
            if hasattr(raw_env, "get_privileged_information"):
                static_info = raw_env.get_privileged_information(
                    trajectory_horizon=config.privileged.trajectory_horizon
                ).get("static", {})
        return {
            "task_description": task_description,
            "static": _to_json_serializable(static_info),
            "dynamic": [],
        }

    def _append_privileged_record(
        self,
        config: SimulationConfig,
        records: List[Optional[Dict[str, Any]]],
        env_idx: int,
        step_idx: int,
    ) -> None:
        if not config.privileged.enabled:
            return
        raw_env = self._get_raw_env(env_idx)
        if raw_env is None or not hasattr(raw_env, "get_privileged_information"):
            return
        if records[env_idx] is None:
            records[env_idx] = self._new_privileged_record(config, env_idx)
        dynamic = raw_env.get_privileged_information(
            trajectory_horizon=config.privileged.trajectory_horizon
        ).get("dynamic", {})
        records[env_idx]["dynamic"].append(
            {
                "step": int(step_idx),
                "data": _to_json_serializable(_strip_ltl_monitor(dynamic)),
            }
        )

    def _save_privileged_record(
        self,
        config: SimulationConfig,
        records: List[Optional[Dict[str, Any]]],
        env_idx: int,
        episode_idx: int,
        episode_length: int,
        success: bool,
    ) -> None:
        if not config.privileged.enabled:
            return
        output_dir = Path(config.privileged.output_dir)
        record = records[env_idx]
        if record is None:
            return
        output_path = output_dir / f"privileged_information_{episode_idx}.json"
        payload = {
            "privileged_static_info": record["static"],
            "privileged_dynamic_info": record["dynamic"],
            "replay_summary": {
                "task_name": config.env_name.split("/")[-1],
                "task_description": record["task_description"],
                "seed": config.seed,
                "split": config.split,
                "episode_idx": int(episode_idx),
                "replayed_episode_length": int(episode_length),
                "success": bool(success),
            },
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        if config.privileged.run_monitor:
            self._run_symbolic_monitor(output_path)
        records[env_idx] = None

    def _get_raw_env(self, env_idx: int):
        if self.env is None or not hasattr(self.env, "envs"):
            return None
        env = self.env.envs[env_idx]
        gym_env = env.unwrapped if hasattr(env, "unwrapped") else env
        return getattr(gym_env, "env", gym_env)

    def _run_symbolic_monitor(self, privileged_json_path: Path) -> Optional[Path]:
        project_root = Path(os.environ.get("SCENEFLOW_ROOT", Path(__file__).resolve().parents[3]))
        monitor_candidates = [
            project_root,
            project_root / "monitor",
            project_root / "SafeManip" / "monitor",
        ]
        monitor_root = next(
            (
                candidate
                for candidate in monitor_candidates
                if (candidate / "run_monitor_on_privileged.py").exists()
            ),
            project_root / "SafeManip" / "monitor",
        )
        monitor_script = monitor_root / "run_monitor_on_privileged.py"
        if not monitor_script.exists():
            return None
        output_path = privileged_json_path.with_name(f"{privileged_json_path.stem}_monitor.json")
        cmd = [
            sys.executable,
            str(monitor_script),
            str(privileged_json_path),
            "--output",
            str(output_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(monitor_root.parent),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print(
                f"Failed to run symbolic monitor for {privileged_json_path}: "
                f"{completed.stderr.strip()}"
            )
            return None
        return output_path

    def _get_actions_from_server(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the inference server."""
        # Get actions from the server
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env_seed = None if config.seed is None else config.seed + idx
    env = gym.make(config.env_name, split=config.split, enable_render=True, seed=env_seed)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
            camera_names=config.video.camera_names,
            filename_prefix=config.video.filename_prefix,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    seed: Optional[int] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        host: Hostname of the inference server
        port: Port of the inference server
        video_dir: Directory to save videos (None for no videos)
        seed: Random seed for RoboCasa environment construction
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        seed=seed,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceClient(host=host, port=port)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )
