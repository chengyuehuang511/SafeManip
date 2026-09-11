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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _get_raw_robocasa_env(gym_env):
    """Walk down .unwrapped/.env to the raw robocasa/robosuite env (the
    object with a `.sim` MuJoCo handle)."""
    env = gym_env
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        if hasattr(env, "sim"):
            return env
        nxt = getattr(env, "unwrapped", None)
        if nxt is not None and nxt is not env:
            env = nxt
            continue
        nxt = getattr(env, "env", None)
        if nxt is not None:
            env = nxt
            continue
        break
    return None


def _find_env_attr_holder(gym_env, raw_env):
    """Find the (object, attr_name) pair whose `.env` attribute holds
    `raw_env` directly, so ReplayCapture knows what to overwrite to splice
    DataCollectionWrapper into the call chain."""
    env = gym_env
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        if getattr(env, "env", None) is raw_env:
            return env, "env"
        nxt = getattr(env, "unwrapped", None)
        if nxt is not None and nxt is not env:
            env = nxt
            continue
        break
    raise RuntimeError(
        "ReplayCapture: could not find the attribute holding the raw robocasa env "
        "(expected some wrapper's `.env` to be it directly)."
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

        raw_env = _get_raw_robocasa_env(gym_env)
        if raw_env is None:
            raise RuntimeError(
                "ReplayCapture: could not find a raw robocasa env (with `.sim`) "
                "underneath the given gym env."
            )
        holder_obj, holder_attr = _find_env_attr_holder(gym_env, raw_env)
        self._wrapped = DataCollectionWrapper(
            raw_env, str(self.raw_dir), use_env_xml_for_reset=True
        )
        setattr(holder_obj, holder_attr, self._wrapped)

    def finalize(self) -> Optional[Path]:
        """Consolidate all recorded episodes into extras/<episode>/{...} +
        extras/dataset_meta.json. Returns the lerobot-style dataset root
        directory (pass this as the eval-side --dataset_dir), or None if no
        episodes were recorded."""
        from robocasa.scripts.collect_demos import gather_demonstrations_as_hdf5

        env_info = json.dumps(self.env_kwargs, default=str)
        hdf5_dir = self.output_dir / "hdf5"
        hdf5_dir.mkdir(parents=True, exist_ok=True)
        hdf5_path = gather_demonstrations_as_hdf5(
            str(self.raw_dir), str(hdf5_dir), env_info, verbose=True
        )
        if not hdf5_path:
            print("ReplayCapture: no episodes recorded, nothing to finalize.")
            return None

        import h5py

        lerobot_dir = self.output_dir / "lerobot"
        extras_dir = lerobot_dir / "extras"
        extras_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(hdf5_path, "r") as f:
            data_grp = f["data"]
            demo_keys = sorted(data_grp.keys(), key=lambda k: int(k.split("_")[-1]))
            for idx, demo_key in enumerate(demo_keys):
                self._write_episode_extras(extras_dir, data_grp[demo_key], idx)
            env_name = str(data_grp.attrs.get("env", ""))

        dataset_meta = {
            "total": len(demo_keys),
            "env_args": {"env_name": env_name, "env_kwargs": self.env_kwargs},
        }
        with open(extras_dir / "dataset_meta.json", "w") as f:
            json.dump(dataset_meta, f, indent=4, default=str)

        return lerobot_dir

    @staticmethod
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
