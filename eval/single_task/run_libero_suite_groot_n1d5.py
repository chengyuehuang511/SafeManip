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

# torch>=2.6 changed `torch.load`'s default `weights_only` from False to
# True -- this breaks LIBERO's own `benchmark.get_task_init_states()`
# (eval/simulators/libero/libero/libero/benchmark/__init__.py), which does
# a plain `torch.load(init_states_path)` on its own committed,
# numpy-pickled init-state files (predates torch 2.6's stricter default).
# N1.5's own eval_libero() is the only one of the 4 LIBERO launchers that
# actually calls this (RLDX-1/GR00T-N1.6's real gym-env reset path never
# does -- only their unused `if __name__ == "__main__"` demo blocks call
# it), so this is patched here rather than project-wide. These are the
# project's own trusted, committed LIBERO files (not arbitrary/untrusted
# input), so restoring the pre-2.6 default (weights_only=False) is safe --
# confirmed via a real crash (`_pickle.UnpicklingError: Weights only load
# failed ... Unsupported global: GLOBAL numpy.core.multiarray._reconstruct`).
import torch  # noqa: E402

_original_torch_load = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false

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


# The two per-suite max_steps literals inside eval_libero()'s own hardcoded
# if/elif chain (examples/Libero/eval/run_libero_eval.py) that disagree
# with this project's unified LIBERO horizon convention (see
# UNIFIED_MAX_STEPS above / eval/EVAL_PROTOCOL_NOTES.md) -- goal=600 (vs
# 300) and libero_10=1000 (vs 520). Patched via a source-text substitution
# in `_load_pristine_main_module` below (see its own docstring for why),
# rather than left as the previous no-op placeholder.
_MAX_STEPS_SOURCE_OVERRIDES = [
    (
        'max_steps = 600  # longest training demo has 270 steps',
        'max_steps = 300  # overridden: unified LIBERO horizon convention (was 600)',
    ),
    (
        'max_steps = 1000  # longest training demo has 505 steps',
        'max_steps = 520  # overridden: unified LIBERO horizon convention (was 1000)',
    ),
]


def _load_pristine_main_module():
    """Loads examples/Libero/eval/run_libero_eval.py as a module object via
    file path (not a normal `import`, matching
    run_libero_suite_openpi.py's `_load_pristine_main_module`), so its
    `if __name__ == "__main__": ...` guard never fires.

    Also applies `_MAX_STEPS_SOURCE_OVERRIDES` to the source text before
    compiling/exec'ing it (the file on disk is never touched -- only the
    in-memory module object reflects the override, same
    doesn't-modify-the-submodule principle as every other patch in this
    file). This replaces an earlier no-op `_patch_unified_max_steps`
    placeholder: eval_libero()'s per-suite max_steps are hardcoded literal
    if/elif branches inside the function body, not a module-level
    attribute or argument, so there is no object to monkeypatch after the
    fact -- the only way to override them non-invasively is to patch the
    source text itself before it becomes bytecode. Each substring is
    required to match exactly once; a missing match (e.g. after an
    upstream submodule update changes the literals) raises immediately
    rather than silently leaving the override unapplied."""
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

    source = main_path.read_text()
    for old, new in _MAX_STEPS_SOURCE_OVERRIDES:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(
                f"_load_pristine_main_module: expected exactly one occurrence of "
                f"{old!r} in {main_path}, found {count} -- upstream source may have "
                f"changed; update _MAX_STEPS_SOURCE_OVERRIDES."
            )
        source = source.replace(old, new)

    spec = importlib.util.spec_from_file_location("groot_n1d5_libero_main", main_path)
    module = importlib.util.module_from_spec(spec)
    code = compile(source, str(main_path), "exec")
    sys.modules[spec.name] = module
    exec(code, module.__dict__)
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
        # Unlike openpi's main.py (which does `from libero.libero import
        # get_libero_path` at its own top level, so `main_module.
        # get_libero_path` resolves), N1.5's run_libero_eval.py only
        # imports `benchmark` from `libero.libero` -- confirmed by reading
        # its own import block. Import get_libero_path directly here
        # instead of assuming it's on main_module.
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


# N1.5's own pristine run_libero_eval.py does NOT chunk actions at all --
# its `GR00TPolicy.get_action` always takes only `_convert_to_libero_action
# (action_chunk, idx=0)` and discards the rest of the returned chunk,
# re-querying the server on every single `env.step()` (see eval_libero()'s
# main loop, which calls `gr00t_policy.get_action(...)` once per iteration
# of its `while t < max_steps + ...` loop).
#
# `examples/Libero/custom_data_config.py:56` (`action_indices =
# list(range(16))`) confirms these LIBERO checkpoints were TRAINED with a
# 16-step action-prediction horizon -- but that's the model's max
# PREDICTABLE chunk length, not how many of those predicted actions
# should actually be EXECUTED open-loop before re-planning (a separate,
# independent choice that's normally <= the horizon). The community
# LIBERO fine-tune twanghcmut/GR00T-N1.5-LIBERO-4suite-combined (built on
# this same NVIDIA training recipe/action_horizon=16) documents its own
# eval convention as n_action_steps=8 -- direct, LIBERO-specific evidence
# that agrees with N1.6/RLDX-1's own official LIBERO value (8), unlike
# the RoboCasa fork's generic default (16, gr00t/eval/simulation.py /
# scripts/run_eval.py in eval/models/Isaac-GR00T) which is a DIFFERENT
# benchmark's convention, not LIBERO's. Using 8.
GR00T_N1D5_N_ACTION_STEPS = 8


def _patch_action_chunking(main_module, n_action_steps: int = GR00T_N1D5_N_ACTION_STEPS) -> None:
    """Monkeypatches `main_module.GR00TPolicy` (the class name `eval_libero()`
    looks up as a global -- `GR00TPolicy(host=..., port=..., headless=...)`
    at its own call site -- to a subclass whose `get_action` caches
    `n_action_steps` actions from each policy query and serves them one at
    a time, only re-querying once the cache is empty. This changes
    `eval_libero()`'s effective behavior (queries the server every
    `n_action_steps` env steps instead of every step) without editing the
    submodule or touching its loop body at all -- `GR00TPolicy` is looked
    up as a plain module-global at call time, so replacing the attribute
    on the module object before `eval_libero(cfg)` runs is enough."""
    original_cls = main_module.GR00TPolicy

    class _ChunkedGR00TPolicy(original_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._action_queue: list = []

        def get_action(self, observation_dict, lang: str):
            if not self._action_queue:
                obs_dict = self._process_observation(observation_dict, lang)
                action_chunk = self.policy.get_action(obs_dict)
                self._action_queue = [
                    self._convert_to_libero_action(action_chunk, i)
                    for i in range(n_action_steps)
                ]
            return self._action_queue.pop(0)

    main_module.GR00TPolicy = _ChunkedGR00TPolicy


# NOTE: this used to be a documented no-op placeholder (the per-suite
# max_steps discrepancy -- goal=600 vs 300, libero_10=1000 vs 520 -- was
# left as a deferred, flagged known issue). It's now actually fixed via
# `_MAX_STEPS_SOURCE_OVERRIDES` in `_load_pristine_main_module` above
# (source-text patch applied before the module is compiled/exec'd), so
# there's nothing left for this function to do -- kept only so any
# external caller still invoking it by name doesn't break.
def _patch_unified_max_steps(main_module, task_suite_name: str) -> None:
    """Superseded by `_MAX_STEPS_SOURCE_OVERRIDES`/`_load_pristine_main_module`
    -- see that function's docstring for the real fix. Kept as a no-op for
    backward compatibility with any existing call site."""
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
    n_action_steps: int = GR00T_N1D5_N_ACTION_STEPS,
) -> Dict[str, Any]:
    main_module = _load_pristine_main_module()

    flush_replay = None
    if save_replay:
        flush_replay = _patch_libero_env_for_replay(main_module, Path(replay_dir))
    _patch_unified_max_steps(main_module, task_suite_name)
    _patch_action_chunking(main_module, n_action_steps=n_action_steps)

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
    ap.add_argument("--n_action_steps", type=int, default=GR00T_N1D5_N_ACTION_STEPS)
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
        n_action_steps=args.n_action_steps,
    )

    stats_path = Path(args.stats_path or (Path(args.video_out_path) / "stats.json"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
