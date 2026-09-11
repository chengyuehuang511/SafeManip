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
import contextlib
import gzip
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def locate_env_holder(gym_env):
    """Returns `(holder, raw_env)` where `holder.env` is exactly the raw
    robocasa/robosuite Kitchen env (the object with a genuine `.sim` MuJoCo
    handle).

    Walks `.unwrapped` to reach the base gym.Env -- `RoboCasaGymEnv`
    (robocasa/wrappers/gym_wrapper.py) is a plain `gym.Env`, not a
    `gym.Wrapper`, so `.unwrapped` (which recurses through any number of
    gym.Wrapper layers, e.g. GR00T's MultiStepWrapper/VideoRecordingWrapper,
    or gymnasium's own TimeLimit/OrderEnforcing) always terminates there in
    one call -- then reads its `.env` attribute directly, which
    RoboCasaGymEnv's own step/reset/close all use unchanged.

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
    raw_env = getattr(base, "env", None)
    if raw_env is None or not hasattr(raw_env, "sim"):
        raise RuntimeError(
            "ReplayCapture: could not find the raw robocasa env -- "
            f"gym_env.unwrapped={base!r} has no usable `.env` with `.sim`."
        )
    return base, raw_env


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


class ReplayCapture:
    """Wraps one already-created `gym.make(...)` robocasa env so every
    episode rolled out through it is also recorded in the official,
    replayable states+actions format, then (on `.finalize()`) converted to
    the `extras/` layout the training-data replay pipeline reads."""

    def __init__(self, gym_env, output_dir: Path, env_kwargs: Dict[str, Any]):
        from robosuite.wrappers import DataCollectionWrapper

        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw_state_collection"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.env_kwargs = dict(env_kwargs)

        holder_obj, raw_env = locate_env_holder(gym_env)
        self.holder_obj = holder_obj
        self._wrapped = DataCollectionWrapper(
            raw_env, str(self.raw_dir), use_env_xml_for_reset=True
        )
        holder_obj.env = self._wrapped

    def finalize(self) -> Optional[Path]:
        """Consolidate all recorded episodes into extras/<episode>/{...} +
        extras/dataset_meta.json. Returns the lerobot-style dataset root
        directory (pass this as the eval-side --dataset_dir), or None if no
        episodes were recorded."""
        return finalize_replay_dir(self.raw_dir, self.output_dir, self.env_kwargs)


def _stub_pynput_for_headless_import():
    """robocasa.scripts.collect_demos (the only place
    gather_demonstrations_as_hdf5 lives) is a human-teleop CLI script --
    importing it as a module (just to reach that one function) transitively
    imports `robocasa.wrappers.enclosing_wall_render_wrapper`, which does
    `from pynput.keyboard import Key, Listener` at module level. pynput's
    keyboard backend probes for a real X/Wayland display *at import time*
    and raises ImportError if none is found -- which is always true on a
    headless SLURM compute node (confirmed by a real eval run's traceback:
    `ImportError: this platform is not supported: ('failed to acquire X
    connection: Bad display name \"\"', ...)`). We never call anything from
    pynput (only `gather_demonstrations_as_hdf5`, a pure hdf5-writing
    function with no UI/input dependency of its own), so pre-populating
    sys.modules with lightweight stand-ins for the two names actually
    imported (`Key`, `Listener`) lets the import succeed without needing a
    real display, without editing collect_demos.py or
    enclosing_wall_render_wrapper.py."""
    import sys
    import types

    if "pynput" in sys.modules and "pynput.keyboard" in sys.modules:
        return
    pynput_mod = types.ModuleType("pynput")
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Key = type("Key", (), {})
    keyboard_mod.Listener = type("Listener", (), {})
    pynput_mod.keyboard = keyboard_mod
    sys.modules.setdefault("pynput", pynput_mod)
    sys.modules.setdefault("pynput.keyboard", keyboard_mod)


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
    hdf5_path = gather_demonstrations_as_hdf5(str(raw_dir), str(hdf5_dir), env_info, verbose=True)
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
