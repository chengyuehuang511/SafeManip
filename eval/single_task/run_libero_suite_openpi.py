#!/usr/bin/env python3
"""Single-suite LIBERO eval against the PRISTINE eval/models/openpi submodule's
own examples/libero/main.py, unmodified -- calls its `eval_libero(Args)`
function directly (which already loops over every task in the given suite,
`num_trials_per_task` episodes each) rather than reimplementing the rollout
loop.

Hyperparameters are openpi's own official defaults from examples/libero/
main.py's `Args` dataclass (resize_size=224, replan_steps=5, num_steps_wait=10,
num_trials_per_task=50, seed=7), and per-suite `max_steps` values are already
hardcoded inside `eval_libero` itself (220/280/300/520/400 for spatial/
object/goal/10/90) -- not touched here.

`eval_libero` doesn't return a result (just logs + saves per-episode videos),
so this wrapper installs a logging.Handler for the duration of the call to
capture its own final "Total success rate: X" / "Total episodes: N" log
lines, rather than duplicating/reimplementing any part of the eval loop.
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


def _patch_libero_env_for_replay(main_module, base_replay_dir: Path):
    """openpi's `eval_libero()` has no gym.Env at all -- unlike RLDX-1/
    GR00T-N1.6 (gym-registered `libero_sim/<task>`, wrapped via
    `ReplayCapture`/`locate_env_holder`), it calls `_get_libero_env(task,
    resolution, seed)` once per task (creating a raw `OffScreenRenderEnv`
    directly -- see main.py), then reuses that SAME env instance across
    all `num_trials_per_task` episodes for that task via repeated
    `env.reset()`/`env.set_init_state()`/`env.step()` calls, all proxied
    through unchanged by `DataCollectionWrapper`'s own `__getattr__`.

    Monkeypatches `main_module._get_libero_env` (not `main.py` itself) to
    wrap the returned env with `DataCollectionWrapper` via
    `wrap_raw_libero_env_for_replay`, and finalizes the PREVIOUS task's
    collected episodes into `<base_replay_dir>/<task_name>/hdf5/demo.hdf5`
    (LIBERO's own native format, via `finalize_libero_replay_dir`) each
    time a NEW task's env is created -- since `eval_libero()` loops over
    every task in the suite in one process, without finalizing per-task
    like this, all 10 tasks' episodes would end up in one shared
    raw_state_collection dir with only the LAST task's bddl_file, silently
    corrupting the other 9 tasks' hdf5s.

    Also layers our own `MultiCameraVideoCapture` around the same wrapped
    env (in addition to, not instead of, openpi's own native
    `replay_images`/`imageio.mimwrite` video-saving in `eval_libero()`
    itself -- unlike RLDX-1/GR00T-N1.6, there's no `video_dir=None`-style
    toggle to disable it, since it's unconditional, hardcoded logic inside
    `eval_libero()`'s own loop, which we don't edit). Guarantees a
    frame-count-synchronized video exists (matching the replay states 1:1)
    even though openpi's own native video is left running alongside it --
    openpi's own video also skips the `num_steps_wait` dummy-action phase
    (confirmed by reading main.py: images are only appended to
    `replay_images` outside that phase, but `env.step()` -- and therefore
    `DataCollectionWrapper`'s own recording -- still runs during it), so
    its own video isn't 1:1 with the states either, just off by a smaller,
    fixed `num_steps_wait` amount rather than RLDX-1/GR00T's ~2x scaling
    mismatch.

    Returns a zero-arg `flush()` callable the caller must invoke once more
    after `eval_libero()` returns, to finalize the LAST task's episodes
    (which nothing else triggers, since there's no "next task" to prompt
    it)."""
    original_get_libero_env = main_module._get_libero_env
    state: Dict[str, Any] = {"prev": None}

    def _finalize_prev():
        prev = state["prev"]
        if prev is None:
            return
        wrapped_env, raw_dir, bddl_file, task_name, video_capture = prev
        # DataCollectionWrapper only writes state_*.npz to disk every
        # `flush_freq` (100) steps, or on `.close()`/the next `.reset()`
        # (see robosuite.wrappers.data_collection_wrapper.DataCollectionWrapper
        # ._flush/_start_new_episode/close) -- openpi's own eval_libero()
        # never calls .close() on this env (it's reused across trials via
        # reset(), then just dropped once the next task's env is
        # requested), and a short last episode (< 100 steps, no
        # subsequent reset) never reaches an automatic flush either. Left
        # unflushed, raw_dir ends up with zero state_*.npz files, and
        # gather_demonstrations_as_hdf5's own `env_name` stays `None`
        # (never populated from any npz), crashing on
        # `grp.attrs["env"] = None` (TypeError: object dtype has no
        # native HDF5 equivalent) -- confirmed via a real run's traceback.
        # Explicitly closing here (matching what RLDX-1/GR00T-N1.6's
        # run_rollout_gymnasium_policy already does to their own
        # gym-wrapped env at rollout end, which is why they don't hit
        # this) forces the flush before raw_dir is read.
        wrapped_env.close()
        hdf5_dir = base_replay_dir / task_name / "hdf5"
        hdf5_path = finalize_libero_replay_dir(raw_dir, hdf5_dir, bddl_file)
        if hdf5_path is not None:
            print(f"[replay] task={task_name!r} -> {hdf5_path}")
        video_capture.finalize()

    def _wrapped_get_libero_env(task, resolution, seed):
        _finalize_prev()
        env, task_description = original_get_libero_env(task, resolution, seed)
        # Recomputed the same way _get_libero_env's own body does (via the
        # module's own get_libero_path, so it resolves identically to
        # whatever the actual env was just constructed with).
        bddl_file = str(
            Path(main_module.get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        # Matches main.py's own video-filename convention
        # (task_segment = task_description.replace(" ", "_")), so replay
        # and video output are trivially cross-referenced by task name.
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

    main_module._get_libero_env = _wrapped_get_libero_env
    return _finalize_prev


def _patch_disable_websocket_keepalive_timeout_client() -> None:
    """Client-side counterpart to serve_policy_wrapper.py's server-side
    patch of the same name.

    Disabling the keepalive ping timeout on the server alone (see
    serve_policy_wrapper.py's `_patch_disable_websocket_keepalive_timeout`
    docstring for the full root-cause writeup) turned out not to be
    enough: `openpi_client.websocket_client_policy.WebsocketClientPolicy`
    connects via `websockets.sync.client.connect(...)`, which defaults to
    its own, entirely independent `ping_interval=20, ping_timeout=20` --
    confirmed by a real re-test (job 3824891) that still failed with the
    identical "Caught exception: sent 1011 (internal error) keepalive ping
    timeout; no close frame received" on every episode even after the
    server-side fix, since the CLIENT's own keepalive thread times out
    waiting for a pong while the server is blocked on a slow/JIT-heavy
    `infer()` call, independent of anything the server's own
    ping_interval/ping_timeout settings control.

    Values (120s/600s, not disabled entirely) match
    eval/models/openpi's (the robocasa-benchmark fork used for RoboCasa)
    own `websocket_client_policy.py`, which independently hit and fixed
    this exact bug already -- confirmed via `diff` against
    eval/models/openpi_official's copy, and via 139 completed RoboCasa
    openpi tasks with zero keepalive failures. Reusing their proven
    values here (rather than disabling the check outright) keeps a
    genuinely-dead server detectable instead of hanging forever, same as
    their fix.

    Patches `websockets.sync.client.connect` itself (not
    `openpi_client.websocket_client_policy`'s copy of the name) so it
    takes effect regardless of which module reference calls it, as long
    as this runs before `main_module.eval_libero(args)` constructs its
    `WebsocketClientPolicy`."""
    import websockets.sync.client as _client

    _orig_connect = _client.connect

    def _connect_long_keepalive_timeout(*args, **kwargs):
        kwargs.setdefault("ping_interval", 120)
        kwargs.setdefault("ping_timeout", 600)
        return _orig_connect(*args, **kwargs)

    _client.connect = _connect_long_keepalive_timeout


def _patch_torch_load_weights_only_false() -> None:
    """PyTorch 2.6 flipped `torch.load`'s default `weights_only` from False
    to True. `eval/simulators/libero_openpi`'s own
    `libero/libero/benchmark/__init__.py` (`get_task_init_states`) calls
    plain `torch.load(init_states_path)` with no `weights_only` kwarg --
    that pinned (Dec 2023) commit predates the PyTorch default change, so
    under a newer torch it now crashes with `UnpicklingError: Weights only
    load failed ... Unsupported global: GLOBAL numpy.core.multiarray.
    _reconstruct`. The init-states files are LIBERO's own trusted pinned
    per-task numpy-array assets, checked into this exact submodule commit
    (not attacker-controlled input), so defaulting `weights_only=False`
    when the caller doesn't specify it is safe here. Patched from the
    outside rather than editing the submodule."""
    import torch

    _orig_load = torch.load

    def _load_default_weights_only_false(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _load_default_weights_only_false


def _load_pristine_main_module():
    """Loads examples/libero/main.py as a module object via file path (not a
    normal `import`, since there's no __init__.py under examples/) so its
    `if __name__ == "__main__": tyro.cli(eval_libero)` guard never fires --
    only `Args`/`eval_libero` are used."""
    openpi_root = Path(
        __import__("os").environ.get("OPENPI_ROOT", THIS_DIR.parent / "models" / "openpi_official")
    )
    main_path = openpi_root / "examples" / "libero" / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(
            f"examples/libero/main.py not found under OPENPI_ROOT={openpi_root} "
            "-- set OPENPI_ROOT to the openpi submodule root."
        )
    spec = importlib.util.spec_from_file_location("openpi_libero_main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ResultCapture(logging.Handler):
    """Captures the final "Total success rate: X" / "Total episodes: N" log
    lines eval_libero() itself emits, without modifying that function."""

    TOTAL_SR_RE = re.compile(r"Total success rate:\s*([0-9.]+)")
    TOTAL_EP_RE = re.compile(r"Total episodes:\s*(\d+)")

    def __init__(self):
        super().__init__()
        self.success_rate: Optional[float] = None
        self.num_episodes: Optional[int] = None

    def emit(self, record):
        msg = record.getMessage()
        m = self.TOTAL_SR_RE.search(msg)
        if m:
            self.success_rate = float(m.group(1))
        m = self.TOTAL_EP_RE.search(msg)
        if m:
            self.num_episodes = int(m.group(1))


def run_single_suite(
    *,
    task_suite_name: str,
    host: str,
    port: int,
    num_trials_per_task: int,
    replan_steps: int,
    resize_size: int,
    num_steps_wait: int,
    seed: int,
    video_out_path: str,
    save_replay: bool = False,
    replay_dir: Optional[str] = None,
) -> Dict[str, Any]:
    _patch_torch_load_weights_only_false()
    _patch_disable_websocket_keepalive_timeout_client()
    main_module = _load_pristine_main_module()

    flush_replay = None
    if save_replay:
        flush_replay = _patch_libero_env_for_replay(main_module, Path(replay_dir))

    args = main_module.Args(
        host=host,
        port=port,
        resize_size=resize_size,
        replan_steps=replan_steps,
        task_suite_name=task_suite_name,
        num_steps_wait=num_steps_wait,
        num_trials_per_task=num_trials_per_task,
        video_out_path=video_out_path,
        seed=seed,
    )

    capture = _ResultCapture()
    logging.getLogger().addHandler(capture)
    try:
        main_module.eval_libero(args)
    finally:
        logging.getLogger().removeHandler(capture)
        if flush_replay is not None:
            flush_replay()  # finalize the last task's episodes -- see _patch_libero_env_for_replay

    return {
        "task_suite_name": task_suite_name,
        "num_episodes": capture.num_episodes,
        "success_rate": capture.success_rate,
        "video_out_path": video_out_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task_suite_name",
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num_trials_per_task", type=int, default=50)
    ap.add_argument("--replan_steps", type=int, default=5)
    ap.add_argument("--resize_size", type=int, default=224)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--video_out_path", required=True)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--save_replay", action="store_true")
    ap.add_argument(
        "--replay_dir",
        default=None,
        help="Defaults to <video_out_path>/replay if --save_replay is set and this is omitted.",
    )
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
        replan_steps=args.replan_steps,
        resize_size=args.resize_size,
        num_steps_wait=args.num_steps_wait,
        seed=args.seed,
        video_out_path=args.video_out_path,
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
