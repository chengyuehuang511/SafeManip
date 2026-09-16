#!/usr/bin/env python3
"""Single-suite LIBERO eval against the PRISTINE eval/models/cosmos-policy
submodule's own
cosmos_policy/experiments/robot/libero/run_libero_eval.py, unmodified --
same "eval_libero(cfg) loops over the whole suite, called directly with a
manually-constructed PolicyEvalConfig" pattern as
run_libero_suite_openvla.py/run_libero_suite_groot_n1d5.py (also
`@draccus.wrap()`-decorated, same argument-passthrough mechanism already
confirmed safe for OpenVLA's identical decorator usage).

Single combined checkpoint (nvidia/Cosmos-Policy-LIBERO-Predict2-2B, per
LIBERO.md), trained jointly across all 4 suites -- `task_suite_name` only
selects which suite to *evaluate*, matching GR00T-N1.6/RLDX-1's combined-
checkpoint convention rather than GR00T-N1.5's/OpenVLA's per-suite
checkpoints. No libero_90 checkpoint -- 4-suite scope like the others.

Unlike OpenVLA, Cosmos Policy DOES use real action chunking: `chunk_size`/
`num_open_loop_steps` are genuine `PolicyEvalConfig` fields (both default
16, matching LIBERO.md's own documented example command) -- passed through
normally, no monkeypatch needed (unlike GR00T-N1.5, which needed one since
its own script had no chunking mechanism at all).

No max_steps override needed: Cosmos Policy's own hardcoded per-suite
`TASK_MAX_STEPS` (run_libero_eval.py) already exactly matches openpi's
official convention (spatial=220, object=280, goal=300, libero_10=520,
libero_90=400).

`num_steps_wait=10` is hardcoded as a bare local variable inside
run_libero_eval.py (not a `PolicyEvalConfig` field), so it can't be
overridden via cfg the way OpenVLA's `--num_steps_wait` can -- but it
already matches this project's uniform default (10), so this is a
non-issue, just documented here for parity with the other models' own
docstrings.

Checkpoint loading is in-process (torch model loaded directly via
`get_model(cfg)` in cosmos_utils.py) -- no server/client needed, same as
RLDX-1/GR00T-N1.6/OpenVLA.

`dataset_stats_path`/`t5_text_embeddings_path` are relative paths resolved
against wherever `ckpt_path` gets downloaded to locally (e.g.
`nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json`)
-- defaults here match LIBERO.md's own documented example command
verbatim.
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

# Same torch>=2.6 weights_only fix as run_libero_suite_groot_n1d5.py --
# LIBERO's own benchmark.get_task_init_states() does a plain
# torch.load(init_states_path) on its own committed, numpy-pickled
# init-state files (predates torch 2.6's stricter default). Confirmed via
# a real crash (`UnpicklingError: ... got <class 'numpy.dtypes.
# Float64DType'>`) once Cosmos Policy's own eval loop reached
# load_initial_states() -> task_suite.get_task_init_states().
import torch  # noqa: E402

_original_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

from replay_capture import finalize_libero_replay_dir, wrap_raw_libero_env_for_replay  # noqa: E402
from video_capture import MultiCameraVideoCapture  # noqa: E402

_DEFAULT_CKPT = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"


def _load_pristine_main_module():
    """Loads cosmos_policy/experiments/robot/libero/run_libero_eval.py as a
    module object via file path (not a normal `import`, matching
    run_libero_suite_openvla.py's own `_load_pristine_main_module`), so
    its own `if __name__ == "__main__": eval_libero()` guard never
    fires."""
    import os

    cosmos_root = Path(
        os.environ.get("COSMOS_POLICY_ROOT", THIS_DIR.parent / "models" / "cosmos-policy")
    )
    main_path = cosmos_root / "cosmos_policy" / "experiments" / "robot" / "libero" / "run_libero_eval.py"
    if not main_path.is_file():
        raise FileNotFoundError(
            f"cosmos_policy/experiments/robot/libero/run_libero_eval.py not found under COSMOS_POLICY_ROOT={cosmos_root}"
        )
    # run_libero_eval.py does `from cosmos_policy.experiments.robot.libero.
    # libero_utils import ...` -- an absolute import rooted at the
    # submodule's own top level -- needs cosmos_root itself on sys.path.
    if str(cosmos_root) not in sys.path:
        sys.path.insert(0, str(cosmos_root))
    spec = importlib.util.spec_from_file_location("cosmos_policy_libero_main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_libero_env_for_replay(main_module, base_replay_dir: Path):
    """Same mechanism as run_libero_suite_openvla.py's function of the
    same name -- Cosmos Policy's `eval_libero()` also has no gym.Env,
    calling `get_libero_env(task, model_family, resolution=256)` once per
    task (raw `OffScreenRenderEnv`, reused across episodes for that task
    via repeated `env.reset()`/`env.set_init_state()`/`env.step()`) from
    inside `run_task()`, which itself is called once per task_id from
    `eval_libero()`'s own loop. Monkeypatches `main_module.get_libero_env`
    (the name bound in that module's namespace by its own `from
    cosmos_policy.experiments.robot.libero.libero_utils import
    get_libero_env`), wraps with DataCollectionWrapper + our own
    MultiCameraVideoCapture, finalizes per-task on the next task's env
    creation (or via the returned flush() after the last task)."""
    original_get_libero_env = main_module.get_libero_env
    state: Dict[str, Any] = {"prev": None}

    def _finalize_prev():
        prev = state["prev"]
        if prev is None:
            return
        wrapped_env, raw_dir, bddl_file, task_name, video_capture = prev
        # See run_libero_suite_openpi.py's _finalize_prev for the full
        # root-cause explanation: DataCollectionWrapper only flushes to
        # disk every flush_freq=100 steps or on .close()/next reset();
        # explicitly closing here forces the flush before raw_dir is read.
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
        # into run_libero_eval.py's own namespace -- same gap as OpenVLA's
        # wrapper.
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
    ckpt_path: str,
    config: str,
    config_file: str,
    dataset_stats_path: str,
    t5_text_embeddings_path: str,
    chunk_size: int,
    num_open_loop_steps: int,
    num_denoising_steps_action: int,
    num_trials_per_task: int,
    seed: int,
    local_log_dir: str,
    save_replay: bool = False,
    replay_dir: Optional[str] = None,
) -> Dict[str, Any]:
    main_module = _load_pristine_main_module()

    flush_replay = None
    if save_replay:
        flush_replay = _patch_libero_env_for_replay(main_module, Path(replay_dir))

    cfg = main_module.PolicyEvalConfig(
        config=config,
        ckpt_path=ckpt_path,
        config_file=config_file,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        dataset_stats_path=dataset_stats_path,
        t5_text_embeddings_path=t5_text_embeddings_path,
        trained_with_image_aug=True,
        chunk_size=chunk_size,
        num_open_loop_steps=num_open_loop_steps,
        task_suite_name=task_suite_name,
        num_trials_per_task=num_trials_per_task,
        local_log_dir=local_log_dir,
        randomize_seed=False,
        data_collection=False,
        seed=seed,
        use_variance_scale=False,
        deterministic=True,
        ar_future_prediction=False,
        ar_value_prediction=False,
        use_jpeg_compression=True,
        flip_images=True,
        num_denoising_steps_action=num_denoising_steps_action,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )

    # eval_libero() prints to stdout via `print(...)` (and its own local
    # log file), not `logging` -- same as OpenVLA's/GR00T-N1.5's own
    # scripts. Capture stdout instead of installing a logging.Handler.
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
    ap.add_argument("--ckpt_path", default=_DEFAULT_CKPT)
    ap.add_argument("--config", default="cosmos_predict2_2b_480p_libero__inference_only")
    ap.add_argument("--config_file", default="cosmos_policy/config/config.py")
    ap.add_argument(
        "--dataset_stats_path",
        default=f"{_DEFAULT_CKPT}/libero_dataset_statistics.json",
    )
    ap.add_argument(
        "--t5_text_embeddings_path",
        default=f"{_DEFAULT_CKPT}/libero_t5_embeddings.pkl",
    )
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--num_open_loop_steps", type=int, default=16)
    ap.add_argument("--num_denoising_steps_action", type=int, default=5)
    ap.add_argument("--num_trials_per_task", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--save_replay", action="store_true")
    ap.add_argument("--replay_dir", default=None)
    ap.add_argument("--video_out_path", required=True)
    args = ap.parse_args()

    Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    # Redirect Cosmos Policy's own local_log_dir under our own output dir
    # instead of its pristine default ("./experiments/logs", relative to
    # CWD -- would otherwise clutter the submodule's own directory).
    local_log_dir = str(Path(args.video_out_path) / "cosmos_policy_logs")

    replay_dir = args.replay_dir
    if args.save_replay and replay_dir is None:
        replay_dir = str(Path(args.video_out_path) / "replay")
    if replay_dir is not None:
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

    stats = run_single_suite(
        task_suite_name=args.task_suite_name,
        ckpt_path=args.ckpt_path,
        config=args.config,
        config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        chunk_size=args.chunk_size,
        num_open_loop_steps=args.num_open_loop_steps,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_trials_per_task=args.num_trials_per_task,
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
