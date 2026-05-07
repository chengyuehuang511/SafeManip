"""Run evaluation only for a single RoboCasa task."""

import argparse
from pathlib import Path

from run_eval import (
    DEFAULT_VIDEO_ROOT,
    _run_privileged_eval,
    _run_standard_eval,
    _save_stats,
    run_server,
    set_eval_seed,
    start_server_thread,
)

from gr00t.eval.simulation import SimulationInferenceClient


def run_single_task_client(
    *,
    env_name: str,
    host: str,
    port: int,
    video_dir: str,
    split: str,
    n_episodes: int,
    n_envs: int,
    n_action_steps: int,
    save_replay_package_flag: bool,
    save_privileged_info_flag: bool,
    render_privileged_video_flag: bool,
    privileged_trajectory_horizon: int,
    seed: int,
) -> None:
    """Evaluate exactly one environment by name."""
    set_eval_seed(seed, context="single-task client")
    simulation_client = SimulationInferenceClient(host=host, port=port)

    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())

    privileged_mode = (
        save_replay_package_flag or save_privileged_info_flag or render_privileged_video_flag
    )
    if privileged_mode and n_envs != 1:
        print(
            f"Privileged export requires n_envs=1 for replayable trajectories; overriding {n_envs} -> 1."
        )
        n_envs = 1

    env_output_dir = Path(video_dir) / "evals" / split / env_name
    stats_path = env_output_dir / "stats.json"
    if not privileged_mode and stats_path.exists():
        print(f"{env_name} stats already exists. skipping.")
        return

    if privileged_mode:
        print(f"Running privileged export for {env_name}...")
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

    print(f"Results for {env_name}:")
    print(f"Success rate: {stats['success_rate']:.2f}")
    if privileged_mode and stats.get("rollout_root"):
        stats_path = Path(stats["rollout_root"]) / "stats.json"
    _save_stats(stats_path, stats)
    print(f"saved stats to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Exact RoboCasa environment name, for example PrepareCoffee.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        default="panda_omron",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to evaluate on. Can be either pretrain or target.",
        choices=["pretrain", "target"],
        required=True,
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=50)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=5)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument(
        "--save_replay_package",
        action="store_true",
        help="Save replay_package_*.json/.npz files for each episode.",
    )
    parser.add_argument(
        "--save_privileged_info",
        action="store_true",
        help="Save privileged_information_*.json for each episode.",
    )
    parser.add_argument(
        "--render_privileged_video",
        action="store_true",
        help="Render privileged_information_*.mp4 via scripts/3Dextraction.py.",
    )
    parser.add_argument(
        "--privileged_trajectory_horizon",
        type=int,
        default=128,
        help="Trajectory horizon passed to get_privileged_information().",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for policy inference and RoboCasa eval environments.",
    )
    parser.add_argument("--server", action="store_true", help="Run the server.")
    parser.add_argument("--client", action="store_true", help="Run the client.")
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
        run_single_task_client(
            env_name=args.task,
            host=args.host,
            port=args.port,
            video_dir=args.video_dir or str(DEFAULT_VIDEO_ROOT),
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
        run_single_task_client(
            env_name=args.task,
            host=args.host,
            port=args.port,
            video_dir=args.video_dir or str(DEFAULT_VIDEO_ROOT),
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
