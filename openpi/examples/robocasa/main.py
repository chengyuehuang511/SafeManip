import collections
import dataclasses
import logging
import math
import pathlib
import imageio
from datetime import datetime
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import json
import os
from copy import deepcopy
from robocasa.utils.dataset_registry_utils import get_task_horizon

import robocasa.utils.robomimic.robomimic_dataset_utils as FileUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
import robocasa
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_ds_meta
import gymnasium as gym
from robocasa.utils.env_utils import convert_action


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    split: str = "pretrain"
    num_trials: int = 50  # Number of rollouts per task
    task_set: list = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    log_dir: str = None

    seed: int = 7  # Random Seed (for reproducibility)


def eval_main(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    split = args.split
    log_dir = args.log_dir
    num_trials = args.num_trials
    resize_size = args.resize_size
    replan_steps = args.replan_steps
    host = args.host
    port = args.port

    all_env_names = TASK_SET_REGISTRY[args.task_soup]

    for env_name in all_env_names:
        eval_env(
            env_name,
            split,
            log_dir,
            num_trials,
            resize_size,
            replan_steps,
            host,
            port,
            args.seed,
        )


def eval_env(env_name, split, log_dir, num_trials, resize_size, replan_steps, host, port, seed):
    # set args based on task
    assert split in ["pretrain", "target"]
    horizon = get_task_horizon(env_name) * 1.5

    now = datetime.now()
    now_formatted = now.strftime("%Y-%m-%d-%H-%M")
    log_path = f"{log_dir}/evals_1.5/{split}/{env_name}/{now_formatted}"

    for root, dirs, files in os.walk(os.path.dirname(log_path)):
        if "stats.json" in files:
            print(f"{env_name}/{split}, stats path exists, skipping.")
            return

    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # Get task
    env = gym.make(f"robocasa/{env_name}", split=split, seed=seed)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(num_trials)):

        # Reset environment
        obs, info = env.reset()
        task_lang = obs["annotation.human.task_description"]
        action_plan = collections.deque()

        # Setup
        t = 0
        replay_images = []

        logging.info(f"Starting episode {task_episodes+1}...")
        while t < horizon:
            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
            wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
            img_right = np.ascontiguousarray(obs["video.robot0_agentview_right"])
            
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )
            img_right = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img_right, resize_size, resize_size)
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

                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/right_image": img_right,
                    "observation/state": state,
                    "prompt": task_lang,
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: replan_steps])

            action = action_plan.popleft()
            action = convert_action(action)

            # Execute action in environment
            obs, reward, done, truncated, info = env.step(action)
            done = info["success"]  # for robocasa, use success entry in info

            replay_img = env.render()
            replay_img = np.ascontiguousarray(replay_img)
            replay_img = image_tools.convert_to_uint8(replay_img)

            if t % 2 == 0 or t == horizon - 1 or done:
                replay_images.append(replay_img)

            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1

        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        imageio.mimwrite(
            pathlib.Path(log_path) / f"rollout_{episode_idx}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images],
            fps=20,
        )

        # Log current results
        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"[{env_name}] Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"[{env_name}] Total episodes: {total_episodes}")
    print()
    with open(os.path.join(log_path, "stats.json"), "w") as f:
        stats = {
            "num_episodes": total_episodes,
            "success_rate": float(total_successes) / float(total_episodes),
        }
        json.dump(stats, f, indent=4)

    # close and delete the env
    env.env.close()
    del env.env
    del env


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)
