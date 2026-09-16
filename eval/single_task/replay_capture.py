"""
Make a live GR00T/openpi robocasa policy rollout ALSO produce a replayable,
training-data-style dataset -- reusing robocasa/robosuite's own official
data-collection machinery (robosuite.wrappers.DataCollectionWrapper +
robocasa.scripts.collect_demos.gather_demonstrations_as_hdf5) as a library,
composed from the outside. Nothing inside eval/simulators/robocasa or any
robosuite install is modified.

Why this is possible without touching the submodules
------------------------------------------------------
`DataCollectionWrapper` doesn't care whether the actions it sees come from a
human input device or a policy -- robocasa's own collect_demos.py just
happens to only ever pair it with a teleop loop (see collect_human_trajectory
in that file). The wrapper implements the same env interface it wraps
(step/reset/close, proxying everything else via __getattr__), so it can be
substituted in after the fact.

`robocasa/wrappers/gym_wrapper.py`'s `RoboCasaGymEnv` stores the raw
robosuite env as a plain `self.env` attribute and calls `self.env.step(...)`
/ `self.env.reset()` / `self.env.close()` directly (confirmed by reading
that file) -- no caching elsewhere holds a separate reference. So swapping
`gym_env.env = DataCollectionWrapper(gym_env.env, ...)` after `gym.make(...)`
returns is transparent: every subsequent step/reset goes through the
wrapper, which is exactly how `collect_demos.py` uses it, just attached
after construction instead of before.

Output layout
-------------
`ReplayCapture.finalize()` writes `extras/<episode>/{states.npz,ep_meta.json,
model.xml.gz}` + `extras/dataset_meta.json` -- the same on-disk layout
`SafeManip/monitor/extract_privileged_from_dataset.py` already reads for
training data (see that script's `find_dataset_dir`/`LU.get_episode_states`
etc.), so a copy of that script pointed at the returned directory (via
`--dataset_dir`, bypassing its training-dataset lookup) can compute
privileged info + run the monitor on eval rollouts the same way it does on
training demonstrations.

The intermediate `demo.hdf5` (produced by `gather_demonstrations_as_hdf5`,
robocasa's own official format) is NOT run through robocasa's full
`dataset_states_to_obs.py` postprocessing pipeline -- that pipeline exists to
add trainable *observations*, which replay/monitor extraction doesn't need
(see extract_privileged_from_dataset.py's own docstring: "No action column
is needed" -- predicates are computed purely from replayed simulator state).
So `extras/dataset_meta.json`'s `env_args` is built directly here from the
env kwargs captured at creation time, rather than by calling
`robocasa.utils.lerobot_utils.save_dataset_meta` (which expects `total`/
`env_args` attrs that only exist after that heavier postprocessing step).
`_write_episode_extras` mirrors `save_extra_demo_info`'s file-writing logic
for the fields that ARE already present right after `gather_demonstrations_
as_hdf5` (states / ep_meta / model_file), rather than calling it directly.

Requires `robosuite` installed separately (matches the README's own
"Editable flow" instructions -- it's not vendored in the pristine `robocasa`
submodule).
"""
import argparse
import contextlib
import gzip
import json
import os
import shutil
import types
import xml.etree.ElementTree as ET
from pathlib import Path

# HDF5's file-locking mechanism is known to be unreliable over NFS-mounted
# storage (this repo's entire filesystem, /coc/testnvme/...) -- confirmed
# by a real crash (`OSError: Unable to synchronously create file (unable
# to truncate a file which is already open)`) when opening a hdf5 file for
# writing (via gather_demonstrations_as_hdf5) that was never actually held
# open by anything else in-process; a known h5py/HDF5-over-NFS gotcha, not
# a logic bug. Since every write here is single-writer (one process, one
# file, opened and closed within one function call -- see
# finalize_replay_dir/finalize_libero_replay_dir), disabling HDF5's file
# locking is safe. Set at import time (before any h5py file is ever
# opened by this module) rather than per-callsite, so it covers both
# RoboCasa's and LIBERO's hdf5-writing paths uniformly.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from typing import Any, Dict, Optional

import numpy as np


# Attribute names different holder classes use to store the raw
# robosuite/robomimic env they proxy step/reset/close to. RoboCasaGymEnv
# (robocasa/wrappers/gym_wrapper.py) uses the public `env`; LIBERO's own
# `LiberoEnv` (rldx/eval/sim/LIBERO/libero_env.py, gr00t/eval/sim/LIBERO/
# libero_env.py -- confirmed identical in both RLDX-1's and GR00T-N1.6's
# codebases) uses the private `_env` instead. Checked in this order so the
# already-verified RoboCasa behavior is unchanged (same attribute found
# first, same object returned) -- LIBERO support is purely additive.
_ENV_HOLDER_ATTR_CANDIDATES = ("env", "_env")


def locate_env_holder(gym_env):
    """Returns `(holder, raw_env, attr_name)` where `getattr(holder,
    attr_name)` is exactly the raw robosuite env (the object with a genuine
    `.sim` MuJoCo handle) -- `attr_name` is whichever of
    `_ENV_HOLDER_ATTR_CANDIDATES` actually holds it on this particular
    holder class, so callers can write back to the same attribute
    (`setattr(holder, attr_name, wrapped_env)`) regardless of which
    benchmark's holder convention is in play.

    Walks `.unwrapped` to reach the base gym.Env -- both `RoboCasaGymEnv`
    and LIBERO's `LiberoEnv` are plain `gym.Env` subclasses, not
    `gym.Wrapper`, so `.unwrapped` (which recurses through any number of
    gym.Wrapper layers, e.g. GR00T's MultiStepWrapper/VideoRecordingWrapper,
    or gymnasium's own TimeLimit/OrderEnforcing) always terminates there in
    one call -- then reads whichever of `env`/`_env` is actually present,
    which both holders' own step/reset/close use unchanged.

    Deliberately does NOT use `hasattr(obj, "sim")` as a stopping condition:
    `RoboCasaGymEnv.__getattr__` (`return getattr(self.env, name)`) proxies
    ANY missing attribute -- including "sim" -- down to `self.env`, so
    `hasattr(RoboCasaGymEnv_instance, "sim")` is already True one level too
    early, even though the real MuJoCo `.sim` handle lives on
    `RoboCasaGymEnv_instance.env`. An earlier version of this function used
    that check and silently mistook the wrapper itself for the raw env,
    causing `ReplayCapture` to look for a holder of the wrapper instead of
    the wrapper's own `.env` -- confirmed by a real eval run's traceback."""
    base = getattr(gym_env, "unwrapped", gym_env)
    for attr in _ENV_HOLDER_ATTR_CANDIDATES:
        raw_env = getattr(base, attr, None)
        if raw_env is not None and hasattr(raw_env, "sim"):
            return base, raw_env, attr
    raise RuntimeError(
        "ReplayCapture: could not find the raw env -- gym_env.unwrapped="
        f"{base!r} has no usable `.env`/`._env` with `.sim` "
        f"(checked: {_ENV_HOLDER_ATTR_CANDIDATES})."
    )


@contextlib.contextmanager
def capture_env_kwargs():
    """Monkeypatch robosuite.make for the duration of the `with` block only
    (restored afterward, even on exception) to capture the exact kwargs
    robocasa.utils.env_utils.create_env resolved and used to build the raw
    env -- the same information robocasa's own collect_demos.py records as
    `env_info` (`json.dumps(config)` there). Needed because create_env
    doesn't otherwise expose its resolved env_kwargs to the caller."""
    import robosuite

    captured: Dict[str, Any] = {}
    original_make = robosuite.make

    def _capturing_make(*args, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return original_make(*args, **kwargs)

    robosuite.make = _capturing_make
    try:
        yield captured
    finally:
        robosuite.make = original_make


def _construct_data_collection_wrapper(raw_env, raw_dir: Path):
    """Builds a `DataCollectionWrapper` around `raw_env`, handling two
    version-adaptive gaps found across the robosuite pins in use across
    this repo (RoboCasa's 1.5.2, LIBERO's required 1.4.0) -- shared by both
    `ReplayCapture` (RoboCasa/RLDX-1/GR00T-N1.6, which have a gym.Env
    holder to swap the wrapper into) and openpi's LIBERO path (which has no
    gym.Env at all, just this raw env directly -- see
    run_libero_suite_openpi.py's `_patch_libero_env_for_replay`).

    (a) DataCollectionWrapper.step() (both robosuite versions) calls
    `self.env._check_success()` unconditionally every step, to track
    `self.successful` -- confirmed by reading the source. LIBERO's own
    `OffScreenRenderEnv`/`ControlEnv` only implements the public
    `check_success()` (no underscore), not `_check_success` -- confirmed by
    a real crash (`AttributeError: 'OffScreenRenderEnv' object has no
    attribute '_check_success'`) under LIBERO's env stack. `self.successful`
    is never actually read by gather_demonstrations_as_hdf5 for either
    benchmark (confirmed separately -- neither RoboCasa's nor LIBERO's copy
    filters by success), so this is purely a naming-convention gap, not a
    real missing capability. Binds a per-instance alias (not a class-level
    monkeypatch, to avoid any wider side effect) only when the underscore
    name is genuinely missing but the public one exists -- general/
    version-adaptive rather than hardcoded to LIBERO specifically.

    (b) `use_env_xml_for_reset` (an explicit opt-in override of its own
    False default, presumably for RoboCasa's kitchen-scene randomization
    needs) doesn't exist at all in robosuite 1.4.0 -- confirmed by a real
    crash (`TypeError: ... unexpected keyword argument
    'use_env_xml_for_reset'`) under LIBERO's required robosuite pin.
    RoboCasa's own env uses robosuite 1.5.2, which added it. Rather than
    hardcoding a benchmark-specific branch, this checks the
    actually-installed DataCollectionWrapper's own signature and only
    passes the kwarg if it's genuinely supported -- future-proof against
    any other robosuite version this ever runs under, not just these two.
    """
    import inspect

    from robosuite.wrappers import DataCollectionWrapper

    if not hasattr(raw_env, "_check_success") and hasattr(raw_env, "check_success"):
        raw_env._check_success = raw_env.check_success

    dcw_kwargs: Dict[str, Any] = {}
    if "use_env_xml_for_reset" in inspect.signature(DataCollectionWrapper.__init__).parameters:
        dcw_kwargs["use_env_xml_for_reset"] = True
    return DataCollectionWrapper(raw_env, str(raw_dir), **dcw_kwargs)


class ReplayCapture:
    """Wraps one already-created `gym.make(...)` robocasa env so every
    episode rolled out through it is also recorded in the official,
    replayable states+actions format, then (on `.finalize()`) converted to
    the `extras/` layout the training-data replay pipeline reads."""

    def __init__(self, gym_env, output_dir: Path, env_kwargs: Dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw_state_collection"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.env_kwargs = dict(env_kwargs)

        holder_obj, raw_env, holder_attr = locate_env_holder(gym_env)
        self.holder_obj = holder_obj
        # Which of holder_obj.env / holder_obj._env actually holds the raw
        # env -- see locate_env_holder. Callers that need to read/replace
        # the wrapped env after construction (e.g. to further layer a video
        # capture wrapper on top, as run_single_task_rldx1.py does) should
        # use self.holder_attr rather than assuming `.env`, so the same
        # ReplayCapture usage pattern works for both RoboCasa's and
        # LIBERO's holder conventions.
        self.holder_attr = holder_attr
        self._wrapped = _construct_data_collection_wrapper(raw_env, self.raw_dir)
        setattr(holder_obj, holder_attr, self._wrapped)

    def finalize(self) -> Optional[Path]:
        """Consolidate all recorded episodes into extras/<episode>/{...} +
        extras/dataset_meta.json. Returns the lerobot-style dataset root
        directory (pass this as the eval-side --dataset_dir), or None if no
        episodes were recorded."""
        return finalize_replay_dir(self.raw_dir, self.output_dir, self.env_kwargs)


def wrap_raw_libero_env_for_replay(raw_env, output_dir: Path):
    """For LIBERO paths with no gym.Env at all -- unlike RLDX-1/GR00T-N1.6
    (gym-registered via `libero_sim/<task>`, handled by `ReplayCapture`
    above via `locate_env_holder`), openpi's own `examples/libero/main.py`
    constructs and uses a raw `OffScreenRenderEnv` directly (see
    `_get_libero_env`), with no gym wrapper stack to swap a holder
    attribute on. Returns `(wrapped_env, raw_dir)` -- the caller (see
    run_libero_suite_openpi.py's `_patch_libero_env_for_replay`) is
    responsible for substituting `wrapped_env` in place of the original raw
    env at its own call site (there's no `holder_obj.env = ...` to do that
    automatically here, since there's no holder at all)."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw_state_collection"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wrapped_env = _construct_data_collection_wrapper(raw_env, raw_dir)
    return wrapped_env, raw_dir


class _DummyAttrModule(types.ModuleType):
    """A module stand-in that hands back a harmless dummy class for any
    attribute accessed on it, instead of a fixed hardcoded set of names.
    Used below so this stub doesn't need updating every time a different
    teleop-only import chain reaches for one more pynput.keyboard name
    (confirmed necessary: robocasa's collect_demos.py only needed
    `Key`/`Listener`, but LIBERO's own collect_demonstration.py -- reached
    via robosuite.devices.keyboard -- also needs `Controller`)."""

    def __getattr__(self, name):
        return type(name, (), {})


def _stub_pynput_for_headless_import():
    """Both robocasa.scripts.collect_demos and LIBERO's own
    scripts/collect_demonstration.py (the only places
    gather_demonstrations_as_hdf5 lives, for RoboCasa and LIBERO
    respectively) are human-teleop CLI scripts -- importing either as a
    module (just to reach that one function) transitively imports pynput's
    keyboard backend (robocasa via
    robocasa.wrappers.enclosing_wall_render_wrapper's `from pynput.keyboard
    import Key, Listener`; LIBERO via robosuite.devices.keyboard's `from
    pynput.keyboard import Controller, Key, Listener`). pynput's keyboard
    backend probes for a real X/Wayland display *at import time* and raises
    ImportError if none is found -- always true on a headless SLURM compute
    node (confirmed by a real eval run's traceback: `ImportError: this
    platform is not supported: ('failed to acquire X connection: Bad
    display name \"\"', ...)`). We never call anything from pynput (only
    `gather_demonstrations_as_hdf5`, a pure hdf5-writing function with no
    UI/input dependency of its own), so pre-populating sys.modules with a
    lightweight stand-in module (returning a dummy class for whatever name
    is actually imported, rather than a fixed list -- see
    `_DummyAttrModule`) lets the import succeed without needing a real
    display, without editing either collect_demos.py, LIBERO's
    collect_demonstration.py, or robosuite's own devices/keyboard.py."""
    import sys

    if "pynput" in sys.modules and "pynput.keyboard" in sys.modules:
        return
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = _DummyAttrModule("pynput.keyboard")
    pynput_mod.keyboard = keyboard_mod
    sys.modules.setdefault("pynput", pynput_mod)
    sys.modules.setdefault("pynput.keyboard", keyboard_mod)


_libero_gather_demonstrations_module = None


def _load_libero_gather_demonstrations_module():
    """Loads eval/simulators/libero/scripts/collect_demonstration.py (the
    only place LIBERO's own gather_demonstrations_as_hdf5 lives) as a
    module object via file path, the same technique
    run_libero_suite_openpi.py's _load_pristine_main_module() uses for
    openpi's examples/libero/main.py -- this script lives outside the
    `libero` python package (under scripts/, a bare CLI entry point, not
    importable via a normal `import libero...` statement).

    Requires _stub_pynput_for_headless_import() to have been called first
    (this script transitively imports robosuite.devices.keyboard, which
    imports pynput -- see that function's docstring) and the LIBERO
    submodule's own `scripts/` dir importable (for its sibling
    `import init_path`), which this function handles by inserting both
    that dir and the LIBERO submodule root onto sys.path itself, since
    neither is guaranteed to already be there (the LIBERO submodule root
    IS normally on PYTHONPATH already for the `libero` package proper, but
    scripts/ itself is a plain script directory nothing else needs on the
    path).

    Cached at module level (not re-loaded per call) -- this is a pure,
    side-effect-free function load, same rationale as caching would give
    any other one-time import."""
    global _libero_gather_demonstrations_module
    if _libero_gather_demonstrations_module is not None:
        return _libero_gather_demonstrations_module

    import importlib.util
    import sys

    _stub_pynput_for_headless_import()

    libero_root = Path(
        os.environ.get(
            "LIBERO_ROOT",
            Path(__file__).resolve().parent.parent / "simulators" / "libero",
        )
    )
    scripts_dir = libero_root / "scripts"
    script_path = scripts_dir / "collect_demonstration.py"
    if not script_path.is_file():
        raise FileNotFoundError(
            f"collect_demonstration.py not found under LIBERO_ROOT={libero_root} "
            "-- set LIBERO_ROOT to the LIBERO submodule root."
        )
    for p in (str(scripts_dir), str(libero_root)):
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(
        "libero_collect_demonstration", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _libero_gather_demonstrations_module = module
    return module


def finalize_libero_replay_dir(
    raw_dir: Path, output_dir: Path, bddl_file: str
) -> Optional[Path]:
    """Consolidates a DataCollectionWrapper output directory (`raw_dir`,
    identical layout to finalize_replay_dir's input -- one `ep_*/state_*
    .npz` + `model.xml` subfolder per episode) into `output_dir/demo.hdf5`
    using LIBERO's OWN `gather_demonstrations_as_hdf5` (loaded via
    `_load_libero_gather_demonstrations_module`), producing LIBERO's own
    native training-data hdf5 structure exactly (`data/demo_N/{states,
    actions, model_file attr}` -- see that function's own docstring) --
    deliberately NOT robocasa's `extras/`+lerobot-style reformatting that
    `finalize_replay_dir` produces, since the goal here is for saved LIBERO
    eval replays to have the exact same on-disk shape LIBERO's own training
    data does, not a RoboCasa-specific convention.

    `gather_demonstrations_as_hdf5`, as written in LIBERO's own bare CLI
    script, references a module-level `problem_info` name that's only ever
    actually assigned inside that script's own `if __name__ == "__main__":`
    block (confirmed by reading the file directly) -- calling the function
    without that global set raises NameError. Since a Python function's
    free variables resolve through its own module's `__dict__`
    (`module.problem_info = ...` is exactly equivalent to what running the
    `__main__` block would have set up), this sets that attribute on the
    loaded module object before calling, rather than editing the script
    itself.

    `bddl_file` must be the same bddl file path the rollout's LIBERO task
    was actually created from (only `args.bddl_file` is read inside the
    function, to embed the bddl file path/content as hdf5 attributes)."""
    from libero.libero.envs import bddl_utils as BDDLUtils

    module = _load_libero_gather_demonstrations_module()
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    module.problem_info = problem_info

    args = argparse.Namespace(bddl_file=bddl_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    module.gather_demonstrations_as_hdf5(
        str(raw_dir), str(output_dir), json.dumps({}), args
    )
    hdf5_path = output_dir / "demo.hdf5"
    return hdf5_path if hdf5_path.is_file() else None


def finalize_replay_dir(
    raw_dir: Path,
    output_dir: Path,
    env_kwargs: Dict[str, Any],
    video_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Consolidates a DataCollectionWrapper output directory (`raw_dir`,
    containing one `ep_*/state_*.npz` + `model.xml` subfolder per episode --
    see ReplayCapture.__init__) into `output_dir/lerobot/extras/<episode>/
    {...}` + `extras/dataset_meta.json`, the layout
    extract_privileged_from_dataset.py already reads for training data.

    Episode numbering: `episode_NNNNNN` is assigned by TRUE CHRONOLOGICAL
    rollout order -- each raw episode directory is named `ep_<epoch>_<rand>`
    by DataCollectionWrapper itself (see collect_demos.py's
    `env.ep_directory`), so sorting those directory names by the embedded
    epoch timestamp recovers the real order episodes were recorded in.
    This is deliberately NOT the same as `gather_demonstrations_as_hdf5`'s
    own internal `demo_N` numbering, which is assigned in whatever order
    `os.listdir(raw_dir)` happens to return -- confirmed empirically (on a
    real completed run) to be arbitrary, not chronological. Numbering by
    true rollout order here means a `video_dir` of one video per episode
    (also produced strictly in rollout order, one per episode, no
    concurrency -- see gr00t/eval/simulation.py's VideoRecordingWrapper)
    can be attached by mtime order directly, correct by construction,
    rather than needing a separate reverse-lookup pass after the fact.

    If `video_dir` is given, each episode's video (sorted by file
    modification time, one per episode) is moved into
    `extras/episode_NNNNNN/rollout.mp4` in the same pass -- skipped (with a
    warning) if the video count doesn't match the episode count, rather
    than guessing. Not needed for openpi, whose eval_env already names
    videos `rollout_<episode_idx>_<success|failure>.mp4`.

    Standalone (not a ReplayCapture method) so it can also be used to
    finalize a run whose live rollout already completed and wrote raw
    states to disk, but whose in-process finalize() call failed for an
    unrelated reason (e.g. the pynput import bug this function works around)
    -- no need to redo the actual (expensive) policy rollout to recover.
    """
    _stub_pynput_for_headless_import()
    from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    env_info = json.dumps(env_kwargs, default=str)
    hdf5_dir = output_dir / "hdf5"
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    try:
        hdf5_path = gather_demonstrations_as_hdf5(str(raw_dir), str(hdf5_dir), env_info, verbose=True)
    except TypeError:
        # Some robocasa forks (e.g. cosmos-policy's own
        # eval/models/cosmos-policy-robocasa submodule) ship an older
        # gather_demonstrations_as_hdf5 without a `verbose` kwarg at all
        # -- fall back to the positional-only call rather than assuming
        # every robocasa checkout shares the same (newer) signature.
        hdf5_path = gather_demonstrations_as_hdf5(str(raw_dir), str(hdf5_dir), env_info)
    if not hdf5_path:
        print("ReplayCapture: no episodes recorded, nothing to finalize.")
        return None

    import h5py

    # Map each demo_N (hdf5 group) back to the raw episode directory it came
    # from, via the same os.listdir(raw_dir) order gather_demonstrations_as_hdf5
    # used internally (directory contents are unchanged since that call just
    # returned, so this is the identical order) -- then assign the FINAL
    # episode_NNNNNN number by each raw directory's own true creation-time
    # order (embedded in its `ep_<epoch>_<rand>` name), not by listdir
    # position. See the docstring above for why this distinction matters.
    listdir_order = os.listdir(raw_dir)
    chronological_raw_dirs = sorted(listdir_order, key=lambda name: int(name.split("_")[1]))
    raw_name_to_final_index = {name: i for i, name in enumerate(chronological_raw_dirs)}

    lerobot_dir = output_dir / "lerobot"
    extras_dir = lerobot_dir / "extras"
    extras_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]
        # demo_keys, restored to the same order they were assigned in (h5py
        # doesn't guarantee group-iteration order matches insertion order).
        demo_keys_by_listdir_pos = sorted(data_grp.keys(), key=lambda k: int(k.split("_")[-1]))
        if len(demo_keys_by_listdir_pos) != len(listdir_order):
            raise RuntimeError(
                f"finalize_replay_dir: {len(demo_keys_by_listdir_pos)} demos in the hdf5 but "
                f"{len(listdir_order)} raw episode dirs -- gather_demonstrations_as_hdf5 must "
                "have dropped or merged an episode; refusing to guess a mapping."
            )
        for listdir_pos, demo_key in enumerate(demo_keys_by_listdir_pos):
            raw_name = listdir_order[listdir_pos]
            final_index = raw_name_to_final_index[raw_name]
            _write_episode_extras(extras_dir, data_grp[demo_key], final_index)
        env_name = str(data_grp.attrs.get("env", ""))
        total_episodes = len(demo_keys_by_listdir_pos)

    dataset_meta = {
        "total": total_episodes,
        "env_args": {"env_name": env_name, "env_kwargs": env_kwargs},
    }
    with open(extras_dir / "dataset_meta.json", "w") as f:
        json.dump(dataset_meta, f, indent=4, default=str)

    if video_dir is not None:
        _attach_episode_videos(Path(video_dir), extras_dir, total_episodes)

    return lerobot_dir


def _attach_episode_videos(video_dir: Path, extras_dir: Path, num_episodes: int) -> int:
    """Moves each episode's rollout video (recorded separately, one per
    episode, named by a random uuid with no episode index at all -- see
    gr00t/eval/simulation.py's VideoRecordingWrapper) into
    `extras/episode_NNNNNN/rollout.mp4`, next to that episode's states.npz.

    Correct by construction: videos are written strictly one-per-episode in
    real time (no concurrency in the rollout loop), so sorting by file
    modification time recovers true rollout order -- which is now exactly
    how finalize_replay_dir numbers episode_NNNNNN too (see its docstring).
    So video[i] (by mtime) IS episode_{i:06d}, directly, with no reverse
    lookup through gather_demonstrations_as_hdf5's own (arbitrary) internal
    ordering needed.

    Returns the number of videos successfully attached (0 if the video
    count doesn't match num_episodes -- refuses to guess rather than risk a
    wrong pairing; leaves videos where they were)."""
    videos = sorted(video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if len(videos) != num_episodes:
        print(
            f"_attach_episode_videos: video count ({len(videos)}) != episode count "
            f"({num_episodes}); leaving videos in {video_dir} unattached."
        )
        return 0

    attached = 0
    for idx, video_path in enumerate(videos):
        dest_dir = extras_dir / f"episode_{idx:06d}"
        if not dest_dir.is_dir():
            continue
        shutil.move(str(video_path), str(dest_dir / "rollout.mp4"))
        attached += 1
    return attached


def _write_episode_extras(extras_dir: Path, demo_group, demo_index: int) -> None:
    """Mirrors robocasa.utils.lerobot_utils.save_extra_demo_info's file
    layout (states.npz / ep_meta.json / model.xml.gz), built directly
    from gather_demonstrations_as_hdf5's raw output -- see module
    docstring for why the heavier save_extra_demo_info/
    dataset_states_to_obs.py path isn't used here."""
    states = demo_group["states"][:]
    ep_meta_raw = demo_group.attrs.get("ep_meta")
    ep_meta = json.loads(ep_meta_raw) if ep_meta_raw else {}
    model_xml = demo_group.attrs["model_file"]

    ep_dir = extras_dir / f"episode_{demo_index:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(ep_dir / "states.npz", states=states)
    with open(ep_dir / "ep_meta.json", "w") as f:
        json.dump(ep_meta, f, indent=4)

    root = ET.fromstring(model_xml)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with gzip.open(ep_dir / "model.xml.gz", "wb") as f:
        f.write(xml_bytes)
