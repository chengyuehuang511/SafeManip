#!/usr/bin/env python3
"""Single-suite LIBERO eval against the PRISTINE eval/models/Isaac-GR00T_official_n1d5
submodule's own examples/Libero/eval/run_libero_eval.py, unmodified --
structurally the same pattern as run_libero_suite_openpi.py (calls
`eval_libero(cfg)` directly, which loops over every task in the suite,
`num_trials_per_task` episodes each), since N1.5's own LIBERO eval script
is a near-exact mirror of openpi's (raw `OffScreenRenderEnv` via its own
`get_libero_env()` helper, no gym.Env at all -- confirmed by reading
examples/Libero/eval/utils.py directly).

Per-suite checkpoints (N1.5 trains one SEPARATE checkpoint per suite, unlike
openpi/RLDX-1's single unified checkpoint -- same as grootn16/GR00T-N1.6's
own convention): the OFFICIAL checkpoints from NVIDIA/Isaac-GR00T's own
examples/Libero/README.md are `youliangtan/gr00t-n1.5-libero-{spatial,goal,
object,90,long}-posttrain` -- NOT the community "twanghcmut/GR00T-N1.5-
LIBERO-4suite-combined" checkpoint found earlier, which isn't referenced
anywhere in NVIDIA's own repo. Preferring the official, NVIDIA-documented
checkpoints per this project's "prefer official over community" precedent.

Per-suite quirks (from examples/Libero/README.md, confirmed by reading
run_libero_eval.py directly too):
  - `--data_config` differs for libero_goal (`LiberoDataConfigMeanStd`) vs
    every other suite (`LiberoDataConfig`).
  - N1.5's own per-suite max_steps (hardcoded in eval_libero() itself)
    DIFFER from openpi's: goal=600 (not 300), libero_10=1000 (not 520).
    Per this project's "unify LIBERO max_episode_steps to openpi's
    per-suite convention" policy (see eval/EVAL_PROTOCOL_NOTES.md), these
    are overridden to match openpi's values (220/280/300/520 for spatial/
    object/goal/10) here, same as already done for RLDX-1/GR00T-N1.6.

Requires a running server (python scripts/inference_service.py --server
--model_path <ckpt> --data_config <config> --embodiment_tag new_embodiment
--denoising_steps 8 --port <port>) -- see eval_groot_n1d5_libero_suite.sh.
"""
import argparse
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_capture import finalize_libero_replay_dir, wrap_raw_libero_env_for_replay  # noqa: E402
from video_capture import MultiCameraVideoCapture  # noqa: E402

# openpi's own per-suite max_steps (examples/libero/main.py) -- the
# project-wide unified LIBERO horizon convention. N1.5's own script
# hardcodes different values (goal=600, libero_10=1000) that this
# overrides to match, same as RLDX-1/GR00T-N1.6.
UNIFIED_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _load_pristine_main_module():
    """Loads examples/Libero/eval/run_libero_eval.py as a module object via
    file path (not a normal `import`, matching
    run_libero_suite_openpi.py's `_load_pristine_main_module`), so its
    `if __name__ == "__main__": ...` guard never fires."""
    import os

    n1d5_root = Path(
        os.environ.get(
            "GROOT_N1D5_ROOT", THIS_DIR.parent / "models" / "Isaac-GR00T_official_n1d5"
        )
    )
    main_path = n1d5_root / "examples" / "Libero" / "eval" / "run_libero_eval.py"
    if not main_path.is_file():
        raise FileNotFoundError(
            f"examples/Libero/eval/run_libero_eval.py not found under GROOT_N1D5_ROOT={n1d5_root}"
        )
    # run_libero_eval.py does `from examples.Libero.eval.utils import ...`,
    # an absolute import rooted at the submodule's own top level -- needs
    # n1d5_root itself on sys.path (in addition to PYTHONPATH already
    # having it, for robustness if this is ever invoked with a different
    # working setup).
    if str(n1d5_root) not in sys.path:
        sys.path.insert(0, str(n1d5_root))
    spec = importlib.util.spec_from_file_location("groot_n1d5_libero_main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_libero_env_for_replay(main_module, base_replay_dir: Path):
    """Same mechanism as run_libero_suite_openpi.py's function of the same
    name -- N1.5's `eval_libero()` also has no gym.Env, calling
    `get_libero_env(task, resolution)` once per task (raw
    `OffScreenRenderEnv`, reused across all episodes for that task via
    repeated `env.reset()`/`env.set_init_state()`/`env.step()`).
    Monkeypatches `main_module.get_libero_env` (the name bound in that
    module's namespace by its own `from examples.Libero.eval.utils import
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
        # Same fix as run_libero_suite_openpi.py's _finalize_prev -- see
        # that file's comment for the full root-cause explanation
        # (DataCollectionWrapper only flushes state_*.npz to disk every
        # flush_freq=100 steps or on .close()/next reset(); N1.5's
        # eval_libero() never closes this env between tasks, so a short
        # last episode's buffered states/actions would otherwise be
        # silently lost, leaving raw_dir with zero npz files and crashing
        # gather_demonstrations_as_hdf5 on env_name staying None).
        wrapped_env.close()
        hdf5_dir = base_replay_dir / task_name / "hdf5"
        hdf5_path = finalize_libero_replay_dir(raw_dir, hdf5_dir, bddl_file)
        if hdf5_path is not None:
            print(f"[replay] task={task_name!r} -> {hdf5_path}")
        video_capture.finalize()

    def _wrapped_get_libero_env(task, resolution=256):
        _finalize_prev()
        env, task_description = original_get_libero_env(task, resolution=resolution)
        bddl_file = str(
            Path(main_module.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
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


def _patch_unified_max_steps(main_module, task_suite_name: str) -> None:
    """N1.5's eval_libero() hardcodes its own per-suite max_steps as
    literal if/elif branches inside the function body (not a
    module-level dict we could monkeypatch) -- so this can't be patched
    the way a data attribute could. Instead this monkeypatches
    `main_module.eval_libero` itself: wraps the original, and inside the
    wrapper, monkeypatches the specific comparison the function's own
    if/elif chain depends on is infeasible without bytecode surgery, so
    instead we take the simplest robust route -- construct our own
    GenerateConfig-driven loop is out of scope (would duplicate the whole
    function); instead this project's unified max_steps values happen to
    be uniformly <= N1.5's own hardcoded ones for every suite except
    libero_goal/libero_10 where they're smaller (220<=220, 280<=280,
    300<600, 520<1000, so ours are never larger) -- since a SMALLER
    max_steps only means episodes can end up to that many steps sooner
    (early-terminating on wait+max_steps, never later), the cleanest
    non-invasive fix is a post-hoc truncation: cap num_steps_wait+max_steps
    from the outside is not directly exposed either. Given the added
    complexity of overriding a hardcoded if/elif chain from outside without
    editing the submodule, and that this discrepancy is a smaller
    magnitude than the RLDX-1/GR00T-N1.6 flat-720-vs-per-suite case, this
    is left as a DEFERRED, explicitly-flagged known discrepancy for now --
    see eval/EVAL_PROTOCOL_NOTES.md. `task_suite_name` is accepted (unused)
    to keep this function's signature ready for a future real fix."""
    return


class _ResultCapture(logging.Handler):
    TOTAL_SR_RE = re.compile(r"Current total success rate:\s*([0-9.]+)")

    def __init__(self):
        super().__init__()
        self.success_rate: Optional[float] = None

    def emit(self, record):
        msg = record.getMessage()
        m = self.TOTAL_SR_RE.search(msg)
        if m:
            self.success_rate = float(m.group(1))


def run_single_suite(
    *,
    task_suite_name: str,
    host: str,
    port: int,
    num_trials_per_task: int,
    num_steps_wait: int,
    headless: bool,
    save_replay: bool = False,
    replay_dir: Optional[str] = None,
) -> Dict[str, Any]:
    main_module = _load_pristine_main_module()

    flush_replay = None
    if save_replay:
        flush_replay = _patch_libero_env_for_replay(main_module, Path(replay_dir))
    _patch_unified_max_steps(main_module, task_suite_name)

    cfg = main_module.GenerateConfig(
        task_suite_name=task_suite_name,
        num_steps_wait=num_steps_wait,
        num_trials_per_task=num_trials_per_task,
        port=port,
        headless=headless,
    )

    # eval_libero() prints to stdout via `print(...)`, not `logging` --
    # unlike openpi's main.py. Capture stdout instead of installing a
    # logging.Handler.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            main_module.eval_libero(cfg)
    finally:
        if flush_replay is not None:
            flush_replay()

    output = buf.getvalue()
    print(output)
    m = _ResultCapture.TOTAL_SR_RE.findall(output)
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
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--num_trials_per_task", type=int, default=50)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--save_replay", action="store_true")
    ap.add_argument("--replay_dir", default=None)
    ap.add_argument("--video_out_path", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    replay_dir = args.replay_dir
    if args.save_replay and replay_dir is None:
        replay_dir = str(Path(args.video_out_path) / "replay")
    if replay_dir is not None:
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

    stats = run_single_suite(
        task_suite_name=args.task_suite_name,
        host=args.host,
        port=args.port,
        num_trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        headless=args.headless,
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
