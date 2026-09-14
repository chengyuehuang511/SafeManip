"""LIBERO-as-Gymnasium-env wrapper + registration, for GR00T-family models.

Copied (functionally identical, entry_point retargeted to this file instead
of a submodule-internal path) from
eval/models/grootn16/gr00t/eval/sim/LIBERO/libero_env.py. That file is pure
gym-env-registration/observation-wrapping code -- it only depends on
`libero` and generic "video.*"/"state.*"/"action.*" modality-key
conventions shared across every GR00T version (N1.5/N1.6/N1.7), never on
`gr00t.model` or any version-specific model class -- so it's safe to reuse
unmodified for the pristine (N1.5-era) eval/models/Isaac-GR00T submodule
too, which has no LIBERO example of its own at all (confirmed: no
gr00t/eval/sim/LIBERO/ directory in that submodule).

Lives here (not copied into either submodule) so both
run_single_task_groot_libero.py (N1.5) and run_single_task_grootn16_libero.py
(N1.6) can import and register the exact same env, without editing or
duplicating code inside either submodule.
"""
import math
import os

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from libero.libero import benchmark

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.utils import get_libero_path  # noqa: E402
import numpy as np  # noqa: E402


def quat2axisangle(quat):
    """Copied from robosuite (see grootn16's libero_env.py for the same
    attribution) -- converts quaternion to axis-angle format."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action, binarize=True):
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action


class LiberoEnv(gym.Env):
    def __init__(self, task_bddl_file: str, task_description: str):
        self._env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            camera_heights=256,
            camera_widths=256,
        )
        self._task_description = task_description
        self.observation_space = gym.spaces.Dict(
            {
                "video.image": gym.spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8),
                "video.wrist_image": gym.spaces.Box(
                    low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
                ),
                "state.x": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.y": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.z": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.roll": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.pitch": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.yaw": gym.spaces.Box(low=-1, high=1, shape=(1,)),
                "state.gripper": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                "annotation.human.action.task_description": gym.spaces.Text(max_length=512),
            }
        )
        self.action_space = spaces.Dict(
            {
                "action.x": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.y": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.z": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.roll": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.pitch": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.yaw": spaces.Box(low=-1, high=1, shape=(1,)),
                "action.gripper": spaces.Box(low=-1, high=1, shape=(1,)),
            }
        )

    def close(self):
        self._env.close()

    def _process_observation(self, obs):
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        return {
            "video.image": obs["agentview_image"][::-1, ::-1],
            "video.wrist_image": obs["robot0_eye_in_hand_image"][::-1, ::-1],
            "state.x": [xyz[0]],
            "state.y": [xyz[1]],
            "state.z": [xyz[2]],
            "state.roll": [rpy[0]],
            "state.pitch": [rpy[1]],
            "state.yaw": [rpy[2]],
            "state.gripper": gripper,
            "annotation.human.action.task_description": self._task_description,
        }

    def reset(self, seed=None, options=None):
        observation = self._env.reset()
        observation = self._process_observation(observation)
        info = {"success": self._env.check_success()}
        return observation, info

    def step(self, action):
        action_vector = np.concatenate(
            [
                action["action.x"],
                action["action.y"],
                action["action.z"],
                action["action.roll"],
                action["action.pitch"],
                action["action.yaw"],
                action["action.gripper"],
            ],
            axis=0,
        )
        action_vector = normalize_gripper_action(action_vector)
        action_vector = invert_gripper_action(action_vector)
        observation, reward, done, info = self._env.step(action_vector)
        observation = self._process_observation(observation)
        info["success"] = self._env.check_success()
        truncated = False
        return observation, reward, done, truncated, info


def register_libero_envs():
    benchmark_dict = benchmark.get_benchmark_dict()
    for task_suite_name in [
        "libero_10",
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_90",
    ]:
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.get_num_tasks()):
            task = task_suite.get_task(task_id)
            task_name = task.name
            task_description = task.language
            task_bddl_file = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            register(
                id=f"libero_sim/{task_name}",
                entry_point="libero_gr00t_env:LiberoEnv",
                kwargs={
                    "task_bddl_file": task_bddl_file,
                    "task_description": task_description,
                },
            )
