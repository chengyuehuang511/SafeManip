#!/usr/bin/env python3
"""Single-task GR00T eval against the PRISTINE eval/models/Isaac-GR00T
submodule, saving eval videos and (optionally) a replayable rollout dataset.

Why this script exists (rather than reusing scripts/run_eval.py as-is)
------------------------------------------------------------------------
The pristine Isaac-GR00T submodule ships no single-task CLI entry point:
`scripts/run_eval.py`'s `run_client` only sweeps a whole `--task_set`
(skipping any env whose stats.json already exists), and the one genuinely
single-env-capable piece (`gr00t.eval.simulation.run_evaluation`) is just a
`__main__` block with a hardcoded env name, not exposed as a CLI. This
script is the thin single-task wrapper that was missing, built entirely out
of the pristine submodule's own importable pieces (`SimulationConfig`,
`SimulationInferenceClient`, `RobotInferenceServer`, `Gr00tPolicy`, ...) --
nothing inside eval/models/Isaac-GR00T itself is modified.

Usage:
    python eval/single_task/run_single_task_groot.py \\
        --model_path /path/to/checkpoint --task CloseBlenderLid \\
        --split target --video_dir /path/to/videos --n_episodes 10 \\
        --save_replay

Must run in an environment with `gr00t`, `robocasa`, and (only if
--save_replay is passed) `robosuite` importable -- see README's "GR00T
Policy Environment" install instructions. Point PYTHONPATH at
eval/models/Isaac-GR00T and eval/simulators/robocasa (not the *_safemanip
forks) to exercise the pristine submodules.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_capture import ReplayCapture, capture_env_kwargs  # noqa: E402

from robocasa.utils.dataset_registry_utils import get_task_horizon  # noqa: E402

from gr00t.eval.robot import RobotInferenceServer  # noqa: E402
from gr00t.eval.simulation import (  # noqa: E402
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402


def run_server(data_config: str, model_path: str, embodiment_tag: str, port: int) -> None:
    """Identical in spirit to scripts/run_eval.py's run_server (pristine) --
    duplicated rather than imported since run_eval.py is a top-level script,
    not a package module, and this is ~10 lines, not worth a fragile
    sys.path/import_module hack to reach it."""
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
    server = RobotInferenceServer(policy, port=port)
    server.run()


class ReplaySimulationInferenceClient(SimulationInferenceClient):
    """Subclasses the pristine SimulationInferenceClient (unmodified in
    place -- this is a new subclass in our own file) to splice replay
    capture into env setup, without reimplementing run_simulation's rollout
    loop. `setup_environment` is the one method that both (a) actually
    constructs the raw robocasa envs and (b) returns before the rollout
    loop starts, making it the natural seam."""

    def __init__(self, host: str, port: int, replay_root: Optional[Path] = None):
        super().__init__(host=host, port=port)
        self.replay_root = Path(replay_root) if replay_root is not None else None
        self.replay_captures: List[ReplayCapture] = []

    def setup_environment(self, config: SimulationConfig):
        if self.replay_root is None:
            return super().setup_environment(config)

        if config.n_envs != 1:
            raise ValueError("Replay capture currently supports n_envs=1.")

        with capture_env_kwargs() as captured:
            vec_env = super().setup_environment(config)
            if not captured:
                # gym.vector.SyncVectorEnv constructs its sub-envs eagerly in
                # __init__, so `captured` should already be populated here;
                # this is a defensive check, not the expected path.
                raise RuntimeError(
                    "ReplaySimulationInferenceClient: robosuite.make was never "
                    "called while constructing the vector env -- env_kwargs "
                    "capture failed."
                )
            env_kwargs = dict(captured)

        for sub_env in vec_env.envs:
            self.replay_captures.append(
                ReplayCapture(sub_env, self.replay_root, env_kwargs)
            )
        return vec_env

    def finalize_replays(self) -> List[Path]:
        return [p for p in (c.finalize() for c in self.replay_captures) if p is not None]


def run_single_task(
    *,
    model_path: str,
    task: str,
    split: str,
    data_config: str,
    embodiment_tag: str,
    host: str,
    port: int,
    video_dir: str,
    n_episodes: int,
    n_action_steps: int,
    save_replay: bool,
    replay_dir: Optional[str],
) -> Dict[str, Any]:
    server_thread = threading.Thread(
        target=run_server, args=(data_config, model_path, embodiment_tag, port), daemon=True
    )
    server_thread.start()
    time.sleep(1)  # give the server time to start (matches run_eval.py's own convention)

    replay_root = Path(replay_dir) if save_replay else None
    if replay_root is not None:
        replay_root.mkdir(parents=True, exist_ok=True)
        simulation_client: SimulationInferenceClient = ReplaySimulationInferenceClient(
            host=host, port=port, replay_root=replay_root
        )
    else:
        simulation_client = SimulationInferenceClient(host=host, port=port)

    horizon = get_task_horizon(task)
    config = SimulationConfig(
        env_name=f"robocasa/{task}",
        split=split,
        n_episodes=n_episodes,
        n_envs=1,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(n_action_steps=n_action_steps, max_episode_steps=horizon),
    )

    print(f"Running {n_episodes} episode(s) of {task} ({split})...")
    _, episode_successes = simulation_client.run_simulation(config)
    success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
    print(f"Success rate: {success_rate:.2f}")

    replay_dataset_dirs: List[str] = []
    if isinstance(simulation_client, ReplaySimulationInferenceClient):
        replay_dataset_dirs = [str(p) for p in simulation_client.finalize_replays()]
        for p in replay_dataset_dirs:
            print(f"Replayable dataset written to: {p}")

    return {
        "task": task,
        "split": split,
        "num_episodes": len(episode_successes),
        "success_rate": success_rate,
        "video_dir": video_dir,
        "replay_dataset_dirs": replay_dataset_dirs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True, help="RoboCasa env/task name, e.g. CloseBlenderLid")
    ap.add_argument("--split", choices=["pretrain", "target"], required=True)
    ap.add_argument("--data_config", default="panda_omron")
    ap.add_argument("--embodiment_tag", default="new_embodiment")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--n_action_steps", type=int, default=16)
    ap.add_argument(
        "--save_replay",
        action="store_true",
        help="Record rollouts via robosuite.wrappers.DataCollectionWrapper and "
        "convert them into a replayable extras/ dataset (see replay_capture.py).",
    )
    ap.add_argument(
        "--replay_dir",
        default=None,
        help="Where to write the replay dataset. Defaults to <video_dir>/replay.",
    )
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.replay_dir or str(video_dir / "replay")

    stats = run_single_task(
        model_path=args.model_path,
        task=args.task,
        split=args.split,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        host=args.host,
        port=args.port,
        video_dir=str(video_dir),
        n_episodes=args.n_episodes,
        n_action_steps=args.n_action_steps,
        save_replay=args.save_replay,
        replay_dir=replay_dir,
    )

    stats_path = video_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
