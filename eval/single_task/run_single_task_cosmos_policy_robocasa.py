#!/usr/bin/env python3
"""Single-task RoboCasa eval against the PRISTINE eval/models/cosmos-policy
submodule's own cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py,
unmodified -- calls its `eval_robocasa(cfg)` directly (which loops over
`num_trials_per_task` episodes for ONE task, given by `cfg.task_name`) with
a manually-constructed PolicyEvalConfig, same `@draccus.wrap()`
call-directly-with-an-instance mechanism already confirmed safe for the
LIBERO wrapper (run_libero_suite_cosmos_policy.py).

Uses the ORIGINAL (not RoboCasa365) 24-task RoboCasa benchmark -- this is
not a hyperparameter to switch, it's the ONLY mode `run_robocasa_eval.py`
supports: `PolicyEvalConfig.task_name` is validated against RoboCasa's own
`SINGLE_STAGE_TASK_DATASETS`/`MULTI_STAGE_TASK_DATASETS` registry (the
original 24-task suite, not RoboCasa365's separate multitask_learning task
list), and `obj_instance_split="B"` + the fixed 5-scene
`layout_and_style_ids` default already encode the "target"/held-out-test
convention this project's own EVAL_PROTOCOL_NOTES.md documents for that
split (see RLDX-1's own `--robocasa_split target` / GR00T-N1.6's
`SPLIT=target` launchers for the analogous convention elsewhere in this
project) -- there is no RoboCasa365 support to disable here.

Architectural difference from every other replay-capture wrapper in this
project: `create_robocasa_env(cfg, seed, episode_idx)` builds a raw
`robosuite.make(...)` env DIRECTLY (confirmed: no gym.Env, no gym-registered
env id at all, unlike RoboCasa/RLDX-1/GR00T-N1.6's own `RoboCasaGymEnv`) --
so `ReplayCapture`/`locate_env_holder` (which expect a gym holder object
exposing `.env`/`._env`) don't apply. Instead this reuses the same
raw-env-direct-wrap primitives the LIBERO wrappers use
(`_construct_data_collection_wrapper` from replay_capture.py), applied to
`create_robocasa_env`'s own return value the same way LIBERO wrappers patch
`get_libero_env`.

Second architectural difference: a BRAND NEW raw env is created for EVERY
episode (`run_task`'s own loop calls `create_robocasa_env` once per
`episode_idx`, then `env.close()` once that episode ends) -- unlike LIBERO,
where one env is created per TASK and reused across many episodes via
repeated `reset()`. This actually simplifies DataCollectionWrapper handling
(a fresh instance per episode has no stale state to worry about, and its
own `.close()` naturally flushes that episode's data when `run_task` calls
`env.close()`), but means `MultiCameraVideoCapture` -- which tracks its own
persistent `_episode_idx` counter across `reset()` calls on one long-lived
inner env -- must be created ONCE for the whole task and have its own
`.env` attribute REPOINTED to each new episode's wrapped env, rather than
being recreated per episode (recreating it per episode would reset its
internal counter to 0 every time, overwriting `episode_000000.mp4` on every
episode).

Finalization: since this is single-task (all episodes for ONE task_name),
`finalize_replay_dir` (the RoboCasa-native one, not `finalize_libero_replay_dir`)
is called exactly once after `eval_robocasa(cfg)` returns -- no per-task-boundary
trigger needed, unlike the LIBERO suite wrappers (openpi/N1.5/OpenVLA/
Cosmos Policy) which loop over many tasks in one process and need a
finalize-on-next-task callback.

Checkpoint loading is in-process (same `get_model`/`cosmos_utils.py`
mechanism as the LIBERO wrapper) -- no server needed.

Requires a SEPARATE conda env from the LIBERO one: cosmos-policy's own
`pyproject.toml` explicitly marks the `robocasa` and `libero` dependency
groups as mutually exclusive (`[tool.uv.conflicts]`) -- robosuite==1.5.1/
mujoco==3.2.6 for robocasa vs robosuite==1.4.1/mujoco==3.3.2 for libero.
See eval_cosmos_policy_robocasa_single_task.sh's own docstring for the
env setup this implies.
"""
import argparse
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Optional

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_capture import _construct_data_collection_wrapper, finalize_replay_dir  # noqa: E402
from video_capture import MultiCameraVideoCapture  # noqa: E402

_DEFAULT_CKPT = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"


def _load_pristine_main_module():
    """Loads cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py
    as a module object via file path (not a normal `import`, matching
    run_libero_suite_cosmos_policy.py's own `_load_pristine_main_module`),
    so its own `if __name__ == "__main__": eval_robocasa()` guard never
    fires."""
    import os

    cosmos_root = Path(
        os.environ.get("COSMOS_POLICY_ROOT", THIS_DIR.parent / "models" / "cosmos-policy")
    )
    main_path = cosmos_root / "cosmos_policy" / "experiments" / "robot" / "robocasa" / "run_robocasa_eval.py"
    if not main_path.is_file():
        raise FileNotFoundError(
            f"cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py not found under COSMOS_POLICY_ROOT={cosmos_root}"
        )
    if str(cosmos_root) not in sys.path:
        sys.path.insert(0, str(cosmos_root))
    spec = importlib.util.spec_from_file_location("cosmos_policy_robocasa_main", main_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _patch_sample_kitchen_object_for_reg_bbox_assets()
    return module


def _patch_sample_kitchen_object_for_reg_bbox_assets():
    """Monkeypatches `sample_kitchen_object` to size-check objects using the
    `reg_bbox` geom rather than the `bottom_site`/`top_site`/
    `horizontal_radius_site` sites the fork's own
    robocasa/models/objects/kitchen_object_utils.py (a version predating
    the current objaverse asset format) expects.

    This project's cosmos-policy-robocasa asset dirs are symlinked from
    the shared eval/simulators/robocasa submodule (see
    eval_cosmos_policy_robocasa_single_task.sh's own docstring) to avoid a
    redundant multi-GB re-download -- but that submodule's objaverse
    assets are all in the CURRENT robocasa format, which only bakes a
    single `reg_bbox` geom (a box `pos`/`size`) into each object's
    model.xml, not the three legacy sites the fork's function looks for.
    Confirmed via direct inspection: 0/200 sampled objaverse model.xml
    files contain a `bottom_site` element. The upstream/current robocasa
    (eval/simulators/robocasa's own kitchen_object_utils.py) already made
    this exact same switch to `reg_bbox` for the same reason -- this patch
    just re-applies that same, already-proven logic to the older
    function, changing only the internal `obj_size` bounds-check
    computation (`bottom`/`top`/`horizontal_radius` are never returned or
    used anywhere else)."""
    import numpy as np
    from robosuite.utils.mjcf_utils import find_elements, string_to_array
    from xml.etree import ElementTree as ET
    from robocasa.models.objects import kitchen_object_utils as _kou

    original_helper = _kou.sample_kitchen_object_helper

    def _patched_sample_kitchen_object(
        groups,
        exclude_groups=None,
        graspable=None,
        washable=None,
        microwavable=None,
        cookable=None,
        freezable=None,
        rng=None,
        obj_registries=("objaverse",),
        split=None,
        max_size=(None, None, None),
        object_scale=None,
    ):
        valid_object_sampled = False
        while valid_object_sampled is False:
            mjcf_kwargs, info = original_helper(
                groups=groups,
                exclude_groups=exclude_groups,
                graspable=graspable,
                washable=washable,
                microwavable=microwavable,
                cookable=cookable,
                freezable=freezable,
                rng=rng,
                obj_registries=obj_registries,
                split=split,
                object_scale=object_scale,
            )

            mjcf_path = info["mjcf_path"]
            tree = ET.parse(mjcf_path)
            root = tree.getroot()
            half_size = string_to_array(
                find_elements(root=root, tags="geom", attribs={"name": "reg_bbox"}).get(
                    "size"
                )
            )
            scale = mjcf_kwargs["scale"]
            obj_size = (half_size * 2) * scale

            valid_object_sampled = True
            for i in range(3):
                if max_size[i] is not None and obj_size[i] > max_size[i]:
                    valid_object_sampled = False

        return mjcf_kwargs, info

    _kou.sample_kitchen_object = _patched_sample_kitchen_object
    # kitchen.py did `from ...kitchen_object_utils import sample_kitchen_object`,
    # binding its own module-local name -- patching the source module's
    # attribute above doesn't affect that already-bound reference, so it
    # must be rebound directly too.
    kitchen_module = sys.modules.get("robocasa.environments.kitchen.kitchen")
    if kitchen_module is not None:
        kitchen_module.sample_kitchen_object = _patched_sample_kitchen_object

    _patch_mjcf_object_for_reg_bbox_assets()


def _patch_mjcf_object_for_reg_bbox_assets():
    """Monkeypatches `robocasa.models.objects.objects.MJCFObject`'s
    `bottom_offset`/`top_offset`/`horizontal_radius` properties (used by
    robosuite's own placement code, e.g.
    `robocasa/utils/placement_samplers.py`'s `obj.bottom_offset`) to read
    from the `reg_bbox` geom instead of the legacy `bottom_site`/
    `top_site`/`horizontal_radius_site` sites -- same underlying cause
    and same fix as `_patch_sample_kitchen_object_for_reg_bbox_assets`
    above (the fork's own MJCFObject class predates the current
    objaverse asset format), just at a different call site: robosuite's
    base `MujocoXMLObject.bottom_offset`/`top_offset` (which the fork's
    MJCFObject doesn't override) and the fork's own
    (also-site-based-and-therefore-also-broken) `horizontal_radius`
    override. Ported directly from eval/simulators/robocasa's own
    already-working `MJCFObject` class, adapted to use
    `self.worldbody.find(...)` (matching this fork's own
    `naming_prefix`-aware style) instead of that version's `_regions`
    attribute, which this fork's class doesn't have."""
    import numpy as np
    from robosuite.utils.mjcf_utils import find_elements, string_to_array
    from robocasa.models.objects.objects import MJCFObject

    def _reg_bbox_elem(self):
        # Recursive search (not a fixed-depth XPath like the fork's own
        # `./body/site[...]` pattern) since reg_bbox sits one level
        # deeper (worldbody/body/body/geom) than the legacy sites did
        # (worldbody/body/site) in the object model.xml layout this
        # asset format uses.
        return find_elements(
            root=self.worldbody,
            tags="geom",
            attribs={"name": "{}reg_bbox".format(self.naming_prefix)},
        )

    def _bottom_offset(self):
        elem = _reg_bbox_elem(self)
        pos = string_to_array(elem.get("pos"))
        half_size = string_to_array(elem.get("size"))
        return np.array([pos[0], pos[1], pos[2] - half_size[2]])

    def _top_offset(self):
        elem = _reg_bbox_elem(self)
        pos = string_to_array(elem.get("pos"))
        half_size = string_to_array(elem.get("size"))
        return np.array([pos[0], pos[1], pos[2] + half_size[2]])

    def _horizontal_radius(self):
        elem = _reg_bbox_elem(self)
        half_size = string_to_array(elem.get("size"))[0:2]
        return np.linalg.norm(half_size)

    def _size(self):
        elem = _reg_bbox_elem(self)
        half_size = string_to_array(elem.get("size"))
        return list(half_size * 2)

    def _get_bbox_points(self, trans=None, rot=None):
        # Same as the fork's own original get_bbox_points, except
        # horiz_radius comes from reg_bbox's own half_size[:2] (exactly
        # what horizontal_radius_site's pos used to encode) instead of
        # the now-nonexistent horizontal_radius_site.
        elem = _reg_bbox_elem(self)
        horiz_radius = string_to_array(elem.get("size"))[0:2]

        bottom_offset = self.bottom_offset
        top_offset = self.top_offset
        center = np.mean([bottom_offset, top_offset], axis=0)
        half_size = [horiz_radius[0], horiz_radius[1], top_offset[2] - center[2]]

        bbox_offsets = [
            center + half_size * np.array([-1, -1, -1]),
            center + half_size * np.array([1, -1, -1]),
            center + half_size * np.array([-1, 1, -1]),
            center + half_size * np.array([-1, -1, 1]),
            center + half_size * np.array([1, 1, 1]),
            center + half_size * np.array([-1, 1, 1]),
            center + half_size * np.array([1, -1, 1]),
            center + half_size * np.array([1, 1, -1]),
        ]

        if trans is None:
            trans = np.array([0, 0, 0])
        if rot is not None:
            import robosuite.utils.transform_utils as T

            rot = T.quat2mat(rot)
        else:
            rot = np.eye(3)

        return [(np.matmul(rot, p) + trans) for p in bbox_offsets]

    MJCFObject.get_bbox_points = _get_bbox_points
    MJCFObject.bottom_offset = property(_bottom_offset)
    MJCFObject.top_offset = property(_top_offset)
    MJCFObject.horizontal_radius = property(_horizontal_radius)
    MJCFObject.size = property(_size)


def _patch_robocasa_env_for_replay(main_module, replay_dir: Path):
    """Monkeypatches `main_module.create_robocasa_env` (defined directly in
    run_robocasa_eval.py itself, not imported from elsewhere -- even
    simpler than the LIBERO wrappers' `get_libero_env` patch point) to wrap
    each episode's freshly-created raw robosuite env with
    DataCollectionWrapper (shared `raw_dir` across all episodes of this
    task-level invocation) and route it through one persistent
    MultiCameraVideoCapture instance (repointing its `.env` each episode --
    see this module's own docstring for why recreating it per episode
    would be wrong).

    Returns a zero-arg `finalize()` callable the caller invokes once after
    `eval_robocasa(cfg)` returns, to gather all episodes' raw states into
    one `demo.hdf5` + finalize the video writer."""
    original_create_robocasa_env = main_module.create_robocasa_env
    raw_dir = replay_dir / "raw_state_collection"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"env_kwargs": None, "video_capture": None}

    def _wrapped_create_robocasa_env(cfg, seed=None, episode_idx=None):
        raw_env, env_kwargs = original_create_robocasa_env(cfg, seed=seed, episode_idx=episode_idx)
        if state["env_kwargs"] is None:
            state["env_kwargs"] = env_kwargs
        wrapped_env = _construct_data_collection_wrapper(raw_env, raw_dir)
        if state["video_capture"] is None:
            state["video_capture"] = MultiCameraVideoCapture(wrapped_env, replay_dir / "lerobot")
        else:
            # Repoint the SAME MultiCameraVideoCapture instance at this
            # episode's fresh wrapped env, rather than constructing a new
            # one -- see module docstring.
            state["video_capture"].env = wrapped_env
        return state["video_capture"], env_kwargs

    main_module.create_robocasa_env = _wrapped_create_robocasa_env

    def _finalize():
        if state["env_kwargs"] is None:
            return None
        hdf5_path = finalize_replay_dir(raw_dir, replay_dir, state["env_kwargs"])
        if hdf5_path is not None:
            print(f"[replay] task={main_module.PolicyEvalConfig.task_name!r} -> {hdf5_path}")
        state["video_capture"].finalize()
        return hdf5_path

    return _finalize


def run_single_task(
    *,
    task_name: str,
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

    finalize_fn = None
    if save_replay:
        finalize_fn = _patch_robocasa_env_for_replay(main_module, Path(replay_dir))

    cfg = main_module.PolicyEvalConfig(
        task_name=task_name,
        ckpt_path=ckpt_path,
        config=config,
        config_file=config_file,
        dataset_stats_path=dataset_stats_path,
        t5_text_embeddings_path=t5_text_embeddings_path,
        trained_with_image_aug=True,
        chunk_size=chunk_size,
        num_open_loop_steps=num_open_loop_steps,
        num_trials_per_task=num_trials_per_task,
        local_log_dir=local_log_dir,
        seed=seed,
        num_denoising_steps_action=num_denoising_steps_action,
        num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
        num_third_person_images=2,
        use_wrist_image=True,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        deterministic=True,
        randomize_seed=False,
        data_collection=False,
        use_jpeg_compression=True,
        flip_images=True,
    )

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            success_rate = main_module.eval_robocasa(cfg)
    finally:
        if finalize_fn is not None:
            finalize_fn()

    print(buf.getvalue())

    return {
        "task_name": task_name,
        "success_rate": float(success_rate),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--ckpt_path", default=_DEFAULT_CKPT)
    ap.add_argument("--config", default="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference")
    ap.add_argument("--config_file", default="cosmos_policy/config/config.py")
    ap.add_argument(
        "--dataset_stats_path",
        default=f"{_DEFAULT_CKPT}/robocasa_dataset_statistics.json",
    )
    ap.add_argument(
        "--t5_text_embeddings_path",
        default=f"{_DEFAULT_CKPT}/robocasa_t5_embeddings.pkl",
    )
    ap.add_argument("--chunk_size", type=int, default=32)
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
    local_log_dir = str(Path(args.video_out_path) / "cosmos_policy_logs")

    replay_dir = args.replay_dir
    if args.save_replay and replay_dir is None:
        replay_dir = str(Path(args.video_out_path) / "replay")
    if replay_dir is not None:
        Path(replay_dir).mkdir(parents=True, exist_ok=True)

    stats = run_single_task(
        task_name=args.task_name,
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
