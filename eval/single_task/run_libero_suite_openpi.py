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
) -> Dict[str, Any]:
    _patch_torch_load_weights_only_false()
    _patch_disable_websocket_keepalive_timeout_client()
    main_module = _load_pristine_main_module()

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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
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
    )

    stats_path = Path(args.stats_path or (Path(args.video_out_path) / "stats.json"))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved stats to {stats_path}")
    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()
