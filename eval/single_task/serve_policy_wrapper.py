#!/usr/bin/env python3
"""Runs the pristine eval/models/openpi/scripts/serve_policy.py, but first
(a) sets robocasa.macros.DATASET_BASE_PATH and (b) clears each TrainConfig's
`data.data_dirs` -- both from the outside, without writing anything into or
editing eval/simulators/robocasa or eval/models/openpi.

(a) DATASET_BASE_PATH
----------------------
`openpi/training/config.py` does `from robocasa.macros import
DATASET_BASE_PATH` at module import time, to locate the RoboCasa training
dataset's normalization stats when loading a checkpoint. robocasa's only
built-in way to set this is a `macros_private.py` file inside the robocasa
package itself (created via `python -m robocasa.scripts.setup_macros`) --
but that means writing a file into eval/simulators/robocasa, which we're
keeping untouched, so this script monkeypatches
`robocasa.macros.DATASET_BASE_PATH` from outside instead. In practice this
alone isn't enough for the RoboCasa `pretrain` split specifically -- see (b).

(b) `data.data_dirs` -- avoiding a crash on a dataset split that isn't
    downloaded here, in favor of the checkpoint's own precomputed stats
---------------------------------------------------------------------------
`policy_config.create_trained_policy` calls `train_config.data.create(...)`
*before* it ever reaches its own good fallback (loading
`<checkpoint_dir>/assets/norm_stats.json`, which already exists for every
checkpoint we have). `LeRobotRobocasaDataConfig.create()`
(openpi/training/config.py) has its own *earlier*, eager fallback: if
`create_base_config` didn't find norm stats via `asset_id` (it never does
here -- no asset_id is set on these configs), and `self.data_dirs` is
non-empty, it unconditionally tries to recompute norm stats from those raw
dataset directories (`_load_norm_stats_from_groot_mixture_dataset`) --
which requires the *actual* `pretrain` split of the RoboCasa dataset
(~300 human demos) to be downloaded, not just present as `target` (which is
all that's on this machine). That raises `FileNotFoundError` before
`create_trained_policy` ever gets to try its own checkpoint-local fallback.

Fix: for every `TrainConfig` in `openpi.training.config._CONFIGS_DICT`,
replace `.data` with a copy that has `data_dirs=None` (dataclasses.replace,
since `LeRobotRobocasaDataConfig` is frozen but `TrainConfig` itself isn't)
-- this makes the eager `self.data_dirs and len(self.data_dirs) > 0` check
false, so `.create()` returns with `norm_stats=None` instead of crashing,
and `create_trained_policy`'s own later checkpoint-local
`assets/norm_stats.json` load (see policy_config.py, the `if norm_stats is
None: ...` block after `data_config = train_config.data.create(...)`) takes
over correctly.

Why `runpy` instead of just importing and calling serve_policy's `main`
-------------------------------------------------------------------------
Both patches above must happen strictly before `openpi.training.config` is
imported for the first time in this process -- `_CONFIGS_DICT` and its
`TrainConfig` entries (with `data_dirs` baked in from
`DATASET_SOUP_REGISTRY`, itself computed at `robocasa` import time) are all
built eagerly at module-import time, and `robocasa.macros.DATASET_BASE_PATH`
is read the same way. A plain `import` of serve_policy.py after
`openpi.training.config` is already imported elsewhere wouldn't help --
those bindings are already resolved and cached by then.
`runpy.run_path(..., run_name="__main__")` re-executes serve_policy.py's
top-level code (imports included) fresh, in this same process, after our
patches -- exactly like `python scripts/serve_policy.py` would, just with
the patches already in place first.

(c) `_groot_openpi_dataset._convert_stats_from_repo_meta` -- stubbing a
    function this pinned commit references but never defines
---------------------------------------------------------------------------
`DataConfigFactory._load_norm_stats` (openpi/training/config.py) falls back
to `_groot_openpi_dataset._convert_stats_from_repo_meta(asset_id)` whenever
the checkpoint-relative `assets_dir/asset_id` path doesn't exist (true for
every LIBERO config here, e.g. `pi05_libero` -- confirmed by a real crash:
"Norm stats not found in .../assets/pi05_libero/physical-intelligence/
libero" followed immediately by `AttributeError: module
'openpi.groot_utils.groot_openpi_dataset' has no attribute
'_convert_stats_from_repo_meta'`). The source has a literal `# TODO: fix`
comment right above that call -- this is a genuine gap in this pinned
commit, not an environment issue. Since `_load_norm_stats` is only ever
supposed to return `None` when nothing can be loaded/converted (letting
`create_trained_policy`'s own later, correct fallback -- the checkpoint's
own precomputed `assets/norm_stats.json`, confirmed present -- take over,
same principle as fix (b) above), this stubs the missing function to just
return `None` directly rather than crash, instead of e.g. skipping the
call entirely (which would also skip that later good fallback).

(d) websockets keepalive ping timeout -- see `_patch_disable_websocket_keepalive_timeout`
---------------------------------------------------------------------------
Disables the server-side keepalive ping timeout (defaults to 20s), which a
slow/JIT-heavy first inference call can trip, silently killing the
connection (and every subsequent episode that reuses it) well before any
model/checkpoint issue would ever come into play. See that function's
docstring for the full root-cause writeup (confirmed via a real crash on
job 3824862: 500/500 LIBERO episodes reported "Success: False" with the
identical "keepalive ping timeout" client-side error, from a single
connection that died after its first request).

Usage (drop-in replacement for `python scripts/serve_policy.py ...`):
    python eval/single_task/serve_policy_wrapper.py \\
        --port=8000 policy:checkpoint --policy.config=... --policy.dir=...

The dataset root defaults to $ROBOCASA_DATASET_BASE_PATH if set, else
~/flash/datasets/robocasa (this machine's actual RoboCasa v1.0 dataset
location, confirmed to contain v1.0/target/...).
"""
import dataclasses
import os
import runpy
import sys
from pathlib import Path

DEFAULT_DATASET_BASE_PATH = os.path.expanduser("~/flash/datasets/robocasa")


def _clear_data_dirs_on_all_configs() -> None:
    """See module docstring (b). Mutates openpi.training.config._CONFIGS_DICT
    in place (each TrainConfig object is replaced/held by reference in that
    dict, so this affects every later `get_config(name)` call too)."""
    import openpi.training.config as _config

    for name, cfg in _config._CONFIGS_DICT.items():
        if not hasattr(cfg.data, "data_dirs"):
            continue
        if cfg.data.data_dirs is None:
            continue
        cfg.data = dataclasses.replace(cfg.data, data_dirs=None)


def _stub_missing_convert_stats_from_repo_meta() -> None:
    """See module docstring (c). Only adds the attribute if it's actually
    missing -- a no-op on any commit where it's already implemented, and
    also a no-op (via the ImportError branch) on official upstream openpi,
    which doesn't have `openpi.groot_utils` at all -- that whole module is
    robocasa-benchmark-fork-specific plumbing this fix works around, not
    something official openpi ever needs."""
    try:
        import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset
    except ImportError:
        return

    if not hasattr(_groot_openpi_dataset, "_convert_stats_from_repo_meta"):
        _groot_openpi_dataset._convert_stats_from_repo_meta = lambda asset_id: None


def _patch_disable_websocket_keepalive_timeout() -> None:
    """Extend the websockets library's default keepalive ping
    interval/timeout (20s/20s) on the server side to 120s/600s (name kept
    for continuity with the client-side patch of the same era, even
    though it no longer fully disables the check -- see below).

    `WebsocketPolicyServer._handler` (openpi/src/openpi/serving/
    websocket_policy_server.py) calls `self._policy.infer(obs)`
    synchronously inside its asyncio handler coroutine -- there's no
    `await`/yield around it, so it blocks the whole event loop for as long
    as inference takes. JAX/pi0's very first inference call after server
    startup incurs JIT compilation, which routinely takes well over 20s on
    this hardware. During that window the server can't answer its own
    keepalive ping, so `websockets` (default `ping_timeout=20`) closes the
    connection with code 1011 right after the first request -- confirmed
    by a real smoke-test run (job 3824862): the server log shows exactly
    ONE connection opened/closed across the whole ~30-minute suite, and
    the client (`examples/libero/main.py`) logged "Caught exception: sent
    1011 (internal error) keepalive ping timeout; no close frame received"
    identically for all 500 episodes, since it doesn't reconnect after a
    drop -- i.e. this silently zeroed out the entire suite's success rate
    with nothing to do with the checkpoint or the eval logic.

    In practice the client's own independent keepalive (see
    run_libero_suite_openpi.py's matching client-side patch) was the one
    actually responsible for the observed failures, and
    eval/models/openpi's (robocasa-benchmark fork) own fix for this bug
    only touches the client side too -- this server-side patch is
    defensive/likely redundant, kept for safety margin rather than
    necessity. Uses the same 120s/600s values as the client-side patch
    (not fully disabled) for the same reason: detect a genuinely-dead
    server eventually instead of hanging forever. Patched on the
    underlying `websockets.asyncio.server.serve` function itself (not on
    `openpi.serving.websocket_policy_server`'s copy of the name) so it
    takes effect no matter which module reference calls it, as long as
    this runs before `serve_forever()` actually starts listening."""
    import websockets.asyncio.server as _server

    _orig_serve = _server.serve

    def _serve_no_keepalive_timeout(*args, **kwargs):
        kwargs.setdefault("ping_interval", 120)
        kwargs.setdefault("ping_timeout", 600)
        return _orig_serve(*args, **kwargs)

    _server.serve = _serve_no_keepalive_timeout


def main() -> None:
    # robocasa isn't on PYTHONPATH for non-RoboCasa envs (e.g. LIBERO) --
    # patch (a) is a no-op there since it's irrelevant to those configs.
    try:
        import robocasa.macros as _macros
    except ImportError:
        _macros = None

    if _macros is not None:
        dataset_base_path = os.environ.get("ROBOCASA_DATASET_BASE_PATH", DEFAULT_DATASET_BASE_PATH)
        _macros.DATASET_BASE_PATH = dataset_base_path

    _clear_data_dirs_on_all_configs()
    _stub_missing_convert_stats_from_repo_meta()
    _patch_disable_websocket_keepalive_timeout()

    openpi_root = Path(
        os.environ.get(
            "OPENPI_ROOT", Path(__file__).resolve().parent.parent / "models" / "openpi"
        )
    )
    serve_policy_path = openpi_root / "scripts" / "serve_policy.py"
    if not serve_policy_path.is_file():
        raise FileNotFoundError(f"serve_policy.py not found under OPENPI_ROOT={openpi_root}")

    sys.argv = [str(serve_policy_path)] + sys.argv[1:]
    runpy.run_path(str(serve_policy_path), run_name="__main__")


if __name__ == "__main__":
    main()
