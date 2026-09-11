#!/usr/bin/env python3
"""Single-task OpenPI eval against the PRISTINE eval/models/openpi submodule,
saving eval videos and (optionally) a replayable rollout dataset.

Why this script exists (rather than reusing examples/robocasa/main.py as-is)
------------------------------------------------------------------------------
`examples/robocasa/main.py`'s `eval_env(env_name, ...)` function IS already
single-task -- unlike GR00T's run_eval.py, no thin CLI wrapper would be
needed for that alone. But two things still require this script:

  1. `eval_main` (the CLI entry point) is broken as shipped in this pristine
     checkout: it reads `args.task_soup`, which `Args` never declares (only
     `task_set` exists) -- calling it as documented raises AttributeError.
     This script calls `eval_env` directly instead, sidestepping that bug
     entirely (not fixed in place, since that would mean editing the
     submodule).
  2. `eval_env` has no rollout/replay-saving hook (only per-episode mp4s via
     `imageio.mimwrite`, keyed on the raw `env`/`gym.make` call it makes
     internally). To add replay capture without editing `main.py`, this
     script monkeypatches `gymnasium.make` for the duration of the call
     (restored after) so the *already-open* env object it hands back to
     `eval_env` has its raw robocasa env swapped for a
     robosuite.wrappers.DataCollectionWrapper first -- see replay_capture.py
     for why this composition is safe (DataCollectionWrapper implements the
     same env interface, and RoboCasaGymEnv just holds a plain `.env`
     attribute it calls directly). `eval_env` itself is imported and called
     completely unmodified.

Usage:
    python eval/single_task/run_single_task_openpi.py \\
        --task CloseBlenderLid --split target --host 127.0.0.1 --port 8000 \\
        --log_dir /path/to/videos --num_trials 10 --save_replay

Must run in an environment with `openpi`/`openpi_client`, `robocasa`, and
(only if --save_replay is passed) `robosuite` importable -- see README's
"OpenPI Policy Environment" install instructions. Point PYTHONPATH at
eval/models/openpi (not openpi_safemanip) and eval/simulators/robocasa (not
robocasa_safemanip) to exercise the pristine submodules. Requires a policy
server already running on --host/--port (this script is client-only, same
as examples/robocasa/main.py -- see openpi's own serve_policy.py).
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_capture import ReplayCapture, capture_env_kwargs  # noqa: E402


def _load_pristine_main_module():
    """Dynamically loads examples/robocasa/main.py from the pristine openpi
    submodule as a module object, without importing it as a package (there
    are no __init__.py files under examples/ -- it's meant to be run as a
    script, not imported -- so this uses importlib's file-path loading
    instead of a normal `import`). Because `__name__` won't be
    "__main__" for this loaded module, its own `if __name__ == "__main__":
    tyro.cli(eval_main)` guard never fires -- only `eval_env` is used."""
    openpi_root = Path(
        __import__("os").environ.get(
            "OPENPI_ROOT",
            THIS_DIR.parent / "models" / "openpi",
        )
    )
    main_path = openpi_root / "examples" / "robocasa" / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(
            f"examples/robocasa/main.py not found under OPENPI_ROOT={openpi_root} "
            "-- set OPENPI_ROOT to the openpi submodule root."
        )
    spec = importlib.util.spec_from_file_location("openpi_robocasa_main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_single_task(
    *,
    task: str,
    split: str,
    host: str,
    port: int,
    log_dir: str,
    num_trials: int,
    resize_size: int,
    replan_steps: int,
    seed: int,
    save_replay: bool,
    replay_dir: Optional[str],
) -> Dict[str, Any]:
    main_module = _load_pristine_main_module()

    replay_holder: Dict[str, ReplayCapture] = {}
    if save_replay:
        import gymnasium as gym

        original_make = gym.make

        def _patched_make(*args, **kwargs):
            with capture_env_kwargs() as captured:
                gym_env = original_make(*args, **kwargs)
                env_kwargs = dict(captured)
            replay_holder["capture"] = ReplayCapture(gym_env, Path(replay_dir), env_kwargs)
            return gym_env

        gym.make = _patched_make

    try:
        main_module.eval_env(
            task,
            split,
            log_dir,
            num_trials,
            resize_size,
            replan_steps,
            host,
            port,
            seed,
        )
    finally:
        if save_replay:
            gym.make = original_make

    replay_dataset_dir = None
    if save_replay and "capture" in replay_holder:
        replay_dataset_dir = replay_holder["capture"].finalize()
        if replay_dataset_dir is not None:
            print(f"Replayable dataset written to: {replay_dataset_dir}")

    # eval_env already wrote its own stats.json under log_dir/evals_1.5/<split>/<task>/<timestamp>/;
    # surface that path so callers don't have to re-derive the timestamp themselves.
    task_root = Path(log_dir) / "evals_1.5" / split / task
    run_dirs = sorted(task_root.glob("*")) if task_root.is_dir() else []
    latest_run_dir = run_dirs[-1] if run_dirs else None
    stats = {}
    if latest_run_dir is not None and (latest_run_dir / "stats.json").is_file():
        with open(latest_run_dir / "stats.json") as f:
            stats = json.load(f)

    return {
        "task": task,
        "split": split,
        "run_dir": str(latest_run_dir) if latest_run_dir else None,
        "replay_dataset_dir": str(replay_dataset_dir) if replay_dataset_dir else None,
        **stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="RoboCasa env/task name, e.g. CloseBlenderLid")
    ap.add_argument("--split", choices=["pretrain", "target"], required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--num_trials", type=int, default=10)
    ap.add_argument("--resize_size", type=int, default=224)
    ap.add_argument("--replan_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--save_replay",
        action="store_true",
        help="Record rollouts via robosuite.wrappers.DataCollectionWrapper and "
        "convert them into a replayable extras/ dataset (see replay_capture.py).",
    )
    ap.add_argument(
        "--replay_dir",
        default=None,
        help="Where to write the replay dataset. Defaults to <log_dir>/replay.",
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = args.replay_dir or str(log_dir / "replay")

    result = run_single_task(
        task=args.task,
        split=args.split,
        host=args.host,
        port=args.port,
        log_dir=str(log_dir),
        num_trials=args.num_trials,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        seed=args.seed,
        save_replay=args.save_replay,
        replay_dir=replay_dir,
    )
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
