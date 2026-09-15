#!/usr/bin/env python3
"""Single-suite LIBERO eval against the PRISTINE eval/models/openvla submodule's
own experiments/robot/libero/run_libero_eval.py, unmodified -- same
"eval_libero(cfg)` loops over the whole suite" pattern as
run_libero_suite_openpi.py/run_libero_suite_groot_n1d5.py, and calls it
directly with a manually-constructed GenerateConfig (confirmed safe: the
`@draccus.wrap()` decorator only parses argv when its first positional arg
isn't already an instance of the expected config class -- see draccus.wrap's
own source, `wrapper_inner`).

Per-suite checkpoints (OpenVLA trains one SEPARATE checkpoint per suite, same
convention as GR00T-N1.5's): the official checkpoints are
`openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}` (no libero_90
checkpoint released), matching this project's official-over-community
precedent.

Unlike GR00T-N1.5/openpi, OpenVLA needs NO running server -- `eval_libero()`
loads the HF checkpoint directly in-process (`get_model`/`get_vla` in
experiments/robot/openvla_utils.py), same in-process pattern as RLDX-1/
GR00T-N1.6. No `--host`/`--port` plumbing needed here.

No max_steps override needed: OpenVLA's own hardcoded per-suite max_steps
(run_libero_eval.py) already exactly match openpi's official convention
(spatial=220, object=280, goal=300, libero_10=520, libero_90=400) -- unlike
GR00T-N1.5, which needed a real source-text patch.

No action-chunking patch needed either: OpenVLA is explicitly NOT trained
with action chunking (its own README: "OpenVLA is not trained with action
chunking... the model can be sensitive to idle actions") -- it queries the
model fresh every single env step by design, so n_action_steps=1 is this
model's own genuine convention, not a gap to fix.

num_trials_per_task already defaults to 50 in OpenVLA's own GenerateConfig
(matching this project's uniform-50 convention) and seed=7 (matching
openpi's convention) -- neither needed overriding either.

Uses the pre-provisioned `openvla` conda env (torch==2.5.1, transformers==
4.40.1, robosuite==1.4.1 -- exact pins OpenVLA's own requirements-min.txt/
libero_requirements.txt call for; none of the other LIBERO conda envs
(openpi-libero/rldx1-libero/groot-libero) are compatible, their transformers
versions are all newer than OpenVLA's hard 4.40.1 pin). torch==2.5.1 is
also < 2.6, so the torch.load `weights_only` default-change fix (needed for
GR00T-N1.5) does NOT apply here.
"""
import argparse
import importlib.util
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_capture import finalize_libero_replay_dir, wrap_raw_libero_env_for_replay  # noqa: E402
from video_capture import MultiCameraVideoCapture  # noqa: E402


def _load_pristine_main_module():
    """Loads experiments/robot/libero/run_libero_eval.py as a module object
    via file path (not a normal `import`, matching
    run_libero_suite_openpi.py's/run_libero_suite_groot_n1d5.py's own
    `_load_pristine_main_module`), so its own `if __name__ == "__main__":
    eval_libero()` guard never fires."""
    import os

    openvla_root = Path(
        os.environ.get("OPENVLA_ROOT", THIS_DIR.parent / "models" / "openvla")
    )
    main_path = openvla_root / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    if not main_path.is_file():
        raise FileNotFoundError(
            f"experiments/robot/libero/run_libero_eval.py not found under OPENVLA_ROOT={openvla_root}"
        )
    # run_libero_eval.py does `from experiments.robot.libero.libero_utils
    # import ...` / `from experiments.robot.robot_utils import ...`,
    # absolute imports rooted at the submodule's own top level -- needs
    # openvla_root itself on sys.path.
    if str(openvla_root) not in sys.path:
        sys.path.insert(0, str(openvla_root))
    spec = importlib.util.spec_from_file_location("openvla_libero_main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_libero_env_for_replay(main_module, base_replay_dir: Path):
    """Same mechanism as run_libero_suite_openpi.py's/
    run_libero_suite_groot_n1d5.py's function of the same name --
    OpenVLA's `eval_libero()` also has no gym.Env, calling
    `get_libero_env(task, model_family, resolution=256)` once per task
    (raw `OffScreenRenderEnv`, reused across all episodes for that task
    via repeated `env.reset()`/`env.set_init_state()`/`env.step()`).
    Monkeypatches `main_module.get_libero_env` (the name bound in that
    module's namespace by its own `from experiments.robot.libero.
    libero_utils import get_libero_env`), wraps with DataCollectionWrapper
    + our own MultiCameraVideoCapture, finalizes per-task on the next
    task's env creation (or via the returned flush() after the last
    task)."""
    original_get_libero_env = main_module.get_libero_env
    state: Dict[str, Any] = {"prev": None}

    def _finalize_prev():
        prev = state["prev"]
        if prev is None:
            return
        wrapped_env, raw_dir, bddl_file, task_name, video_capture = prev
        # See run_libero_suite_openpi.py's _finalize_prev for the full
        # root-cause explanation: DataCollectionWrapper only flushes
        # state_*.npz to disk every flush_freq=100 steps or on
        # .close()/next reset(); OpenVLA's eval_libero() never closes
        # this env between tasks, so a short last episode's buffered
        # states/actions would otherwise be silently lost.
        wrapped_env.close()
        hdf5_dir = base_replay_dir / task_name / "hdf5"
        hdf5_path = finalize_libero_replay_dir(raw_dir, hdf5_dir, bddl_file)
        if hdf5_path is not None:
            print(f"[replay] task={task_name!r} -> {hdf5_path}")
        video_capture.finalize()

    def _wrapped_get_libero_env(task, model_family, resolution=256):
        _finalize_prev()
        env, task_description = original_get_libero_env(task, model_family, resolution=resolution)
        # get_libero_path is only imported inside libero_utils.py, not
        # into run_libero_eval.py's own namespace -- import it directly
        # here rather than assuming main_module.get_libero_path resolves
        # (same gap as GR00T-N1.5's wrapper).
        from libero.libero import get_libero_path

        bddl_file = str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        task_name = task_description.replace(" ", "_")
        task_replay_dir = base_replay_dir / task_name
        wrapped_env, raw_dir = wrap_raw_libero_env_for_replay(env, task_replay_dir)
        video_capture = MultiCameraVideoCapture(
            wrapped_env,
            task_replay_dir / "videos",
            camera_names=("agentview", "robot0_eye_in_hand"),
            double_flip=True,
        )
        state["prev"] = (wrapped_env, raw_dir, bddl_file, task_name, video_capture)
        return video_capture, task_description

    main_module.get_libero_env = _wrapped_get_libero_env
    return _finalize_prev


_TOTAL_SR_RE = re.compile(r"Current total success rate:\s*([0-9.]+)")


def run_single_suite(
    *,
    task_suite_name: str,
    pretrained_checkpoint: str,
    num_trials_per_task: int,
    num_steps_wait: int,
    center_crop: bool,
    seed: int,
    local_log_dir: str,
    save_replay: bool = False,
    replay_dir: Optional[str] = None,
) -> Dict[str, Any]:
    main_module = _load_pristine_main_module()

    flush_replay = None
    if save_replay:
        flush_replay = _patch_libero_env_for_replay(main_module, Path(replay_dir))

    cfg = main_module.GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=pretrained_checkpoint,
        center_crop=center_crop,
        task_suite_name=task_suite_name,
        num_steps_wait=num_steps_wait,
        num_trials_per_task=num_trials_per_task,
        seed=seed,
        local_log_dir=local_log_dir,
    )

    # eval_libero() prints to stdout via `print(...)` (and its own local
    # log file), not `logging` -- same as GR00T-N1.5's own script. Capture
    # stdout instead of installing a logging.Handler.
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            main_module.eval_libero(cfg)
    finally:
        if flush_replay is not None:
            flush_replay()

    output = buf.getvalue()
    print(output)
    m = _TOTAL_SR_RE.findall(output)
    success_rate = float(m[-1]) if m else None

    return {
        "task_suite_name": task_suite_name,
        "success_rate": success_rate,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task_suite_name",
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    ap.add_argument("--pretrained_checkpoint", required=True)
    ap.add_argument("--num_trials_per_task", type=int, default=50)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--center_crop", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--save_replay", action="store_true")
    ap.add_argument("--replay_dir", default=None)
    ap.add_argument("--video_out_path", required=True)
    args = ap.parse_args()

    Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    # Redirect OpenVLA's own local_log_dir under our own output dir instead
    # of its pristine default ("./experiments/logs", relative to CWD --
    # would otherwise clutter the submodule's own directory).
    local_log_dir = str(Path(args.video_out_path) / "openvla_logs")

    replay_dir = args.replay_dir
    if args.save_replay and replay_dir is None:
        replay_dir = str(Path(args.video_out_path) / "replay")
    if replay_dir is not None:
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

    stats = run_single_suite(
        task_suite_name=args.task_suite_name,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        center_crop=args.center_crop,
        seed=args.seed,
        local_log_dir=local_log_dir,
        save_replay=args.save_replay,
        replay_dir=replay_dir,
    )

    stats_path = Path(args.stats_path or (Path(args.video_out_path) / "stats.json"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
