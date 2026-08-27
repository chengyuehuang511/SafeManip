# Video reconstruction from privileged info

> ⚠️ **See [`../official_playback/README.md`](../official_playback/README.md)
> first.** If you have (or can produce) data in the official RoboCasa
> lerobot dataset format, use that instead — it's exact, not an
> approximation, and doesn't need any of the calibration/override hacks
> documented below. This whole approach only exists because
> `privileged_information_<N>.json` (an eval-time dump, not the official
> training-data format) is lossy.

Reconstructs a rollout video directly from `privileged_information_<N>.json`
by posing the simulator at each recorded frame and rendering — **not** by
replaying actions. Lets you sanity-check the original `task.mp4` (including
working around its [known corruption bug](../../viewer/README.md)) against an
independently-rendered version of the same recorded episode state.

## Why pose-setting, not action replay

Investigated first: `Isaac-GR00T/scripts/replay_robocasa_trajectory.py` looks
like the intended "replay a saved trajectory" tool, but (a) it's currently
broken — imports ~17 helper functions from `scripts/run_eval.py` that don't
exist in this checkout, and (b) nothing in the repo actually writes the
`replay_package_<N>.json/.npz` files it expects (action log + initial MuJoCo
state). So action-level replay isn't available from what gets saved today.

What *is* saved, per recorded frame, in `privileged_information_<N>.json`'s
`privileged_dynamic_info[i].data`:
- `robot.joint_positions` (arm+gripper) and `robot.root_pose` (mobile base,
  world-frame position+quaternion)
- `scene.objects[<name>].pose` (position + xyzw quaternion) for every
  manipulable object
- `scene.fixtures[<name>].joints[<joint>].qpos` for every openable/movable
  fixture (cabinet doors, knobs, stove dials, ...)

That's everything needed to directly pose the sim at each frame via
`sim.data.set_joint_qpos(...)` + `sim.forward()` (no physics stepping), then
render. **This is arguably better than action replay for the purpose of
sanity-checking**, because each frame is set from ground truth independently
— there's no error accumulation across frames the way there would be
integrating actions, which is the "reproduced video ends up different from
the original" failure mode from prior experience with action-replay
approaches.

## What you get from this vs. don't

- ✅ Correct scene (layout/style/fixtures/objects), correct robot/object/
  fixture poses at every *recorded* frame, correct camera.
- ❌ Not more temporally dense than the original recording: there's one
  reconstructed frame per monitor frame, which is ~1:8 sparser than the
  original `task.mp4` (see `KNOWN_BUGS.md`'s "same 1:8 ratio" note) — so the
  reconstruction is choppier, not a smooth re-render of every video frame.
  `compare_frames.py` accounts for this by mapping each reconstructed frame
  to its corresponding original-video frame via that ratio, not 1:1.
- ✅ (as of the fixes below) Sub-millimeter accuracy on robot/arm/gripper
  world position, verified directly against the recorded
  `end_effector_pose` (not just eyeballed): 0.0004-0.0024m error at every
  frame checked (0, 50, 55, 60, 65, 70, 80, 100, 150) for `ArrangeBreadBasket`
  episode 0. Before the Newton-correction fix, this was 0.28-0.30m at frames
  60/65/80 — larger than the ~0.15m recorded grasp offset between the
  gripper and a held object, which is why held objects visibly floated away
  from the gripper despite the object's own recorded pose being correct.
- ✅ Fixture positions exact (verified err=0.0 for spot-checked fixtures)
  after the fixture-position-override fix below.
- 📈 Full-episode comparison numbers (`ArrangeBreadBasket` ep 0, 182 frames,
  100 non-corrupted-original comparable): **34.1 mean abs diff / 0.50 SSIM
  before the fixes below -> 19.1 mean abs diff / 0.71 SSIM after.** Not
  pixel-identical, but a large, whole-episode improvement, not just at the
  frames that happened to get manually spot-checked.
- ⚠️ Investigated and **retracted**: the panel near the left edge of frame 0
  that initially looked like a rendering bug (plain gray in our
  reconstruction vs. a cabinet-styled pattern in the original) is
  `cab_corner_main_group` (confirmed by re-coloring that body's geoms bright
  magenta and seeing exactly that panel light up). First hypothesis was a
  material/RNG-divergence bug -- **wrong**. Traced all the way to the
  compiled MJCF: `cab_corner_main_group_g0_vis` is `type="box"` (a plain
  robosuite `BoxObject` primitive, incapable of molded panel geometry at
  all), whereas a real cabinet door (e.g.
  `cab_main_main_group_left_door_g0`) uses `mesh=
  "...CabinetDoorPanel021_vis"` (a dedicated molded-panel mesh asset). The
  flat `lighter_gray` finish on the box exactly matches
  `kitchen_styles/test/style010.yaml`'s `box: default` ->
  `[lighter_gray, ...]`, deterministically, no RNG involved anywhere in this
  path (`load_style_config()` is pure YAML lookup + dict merge). So this
  fixture is rendering *correctly* -- it's a filler with no door, and by
  design has no molding. The patterned look in the original at that screen
  location is most likely the *adjacent* real door
  (`cab_main_main_group_left_door`) occupying a slightly different amount of
  that screen region than in our reconstruction -- a residual camera
  framing/FOV difference, not a fixture material bug. Not further
  investigated. Confirmed NOT the cause of anything here: camera
  randomization (`randomize_cameras()`) is gated behind an `env.
  randomize_cameras` flag and camera config is itself captured in
  `episode_meta["cam_configs"]` and reapplied via `set_ep_meta`, so that's
  not consuming extra RNG draws either.

## Key implementation facts (found the hard way — see git history / session
notes if these ever need re-deriving)

- **Env creation**: `gym.make(f"robocasa/{task_name}", split=..., seed=...)`.
  The `robocasa/<TaskName>` gym ids only exist after `import robocasa` has
  fully run (~396 envs registered via a metaclass hook + a module-level loop
  in `robocasa/wrappers/gym_wrapper.py`) — **do not** import a submodule
  first expecting it to trigger a partial/lazy registration; import plain
  `robocasa` and nothing more before calling `gym.make`.
- **cwd trap**: never run with cwd inside the SafeManip repo (or any
  ancestor of a bare `robocasa`/`gr00t` directory without its own
  `__init__.py`). Python's implicit `''`/cwd sys.path entry resolves
  `import robocasa` to an empty PEP 420 namespace package
  (`robocasa.__file__ is None`) that shadows the real editable install —
  imports "succeed", zero envs get registered, `gym.make` then fails with a
  confusing `NamespaceNotFound`. `reconstruct_video.py` strips `''`/cwd from
  `sys.path` at import time as a guard; the sbatch wrapper also just `cd
  /tmp` before running.
- **Forcing the exact recorded scene**: `raw = env.unwrapped;
  raw.env.set_ep_meta(dict(episode_meta))` where `episode_meta =
  privileged_static_info.task.episode_meta`, called *before* `env.reset()`
  (the `env.unwrapped.set_attrs_from_ep_meta(...)` name used by the broken
  replay script doesn't exist; the real one is `set_ep_meta`, defined on the
  base `MujocoEnv` in `robosuite/environments/base.py`).
- **`env.reset()` must be called with `seed=` too, not just `gym.make()`**:
  `set_ep_meta()` locks layout/style/object placement, but fixture/appliance/
  clutter *selection* is drawn from `env.rng`, which `RoboCasaGymEnv.reset()`
  only reseeds `if seed is not None` (`robocasa/wrappers/gym_wrapper.py`).
  Passing `seed=` to `gym.make()` alone is not enough — an earlier version of
  this script called `gym.make(..., seed=seed)` then bare `env.reset()`, which
  silently sampled a *different* random fixture set than the one actually
  recorded (confirmed: a whole appliance was missing from the reconstruction
  that's present in the original video). Fix: call `env.reset(seed=seed)`.
  Cross-referenced against `github.com/chengyuehuang511/SafeManip@60d7a43`
  ("freeze scene functionality"), which hit the exact same issue.
- **fps must be derived from the original video's duration, not fixed**:
  there's exactly one reconstructed frame per monitor frame, ~1:8 sparser
  than the original `task.mp4` (same ratio noted throughout this doc and
  `KNOWN_BUGS.md`). Rendering those frames at a fixed rate like 10fps plays
  the reconstruction back ~8x *faster* than the original episode — same
  frame count, way less real time, so side-by-side they visibly desync. Pass
  `--original_video` (the sbatch wrapper does this automatically) so fps is
  computed as `n_monitor_frames / original_duration_seconds`, keeping both
  videos the same real-time length.
- **Object joint naming**: every manipulable object's free joint is named
  `f"{object_name}_joint0"` (confirmed for bread/basket/dstr_dining/
  dstr_dining2). Quaternion order in the recording is xyzw; MuJoCo
  `set_joint_qpos` for a free joint wants `[x,y,z, qw,qx,qy,qz]` — convert.
- **Mobile base is NOT a free joint**: it's 3 scalar joints
  (`mobilebase0_joint_mobile_{forward,side,yaw}`, plus a separate
  `..._torso_height` this script currently leaves untouched) chained as:
  slide-forward (parent-frame X) -> slide-side (parent-frame Y) -> hinge-yaw
  -> everything else. The slide origin has a fixed-but-episode-dependent
  "anchor" offset from world origin baked in at model-build time (default
  post-reset forward/side values are things like -10.8/7.9, not ~0).
  `MobileBaseCalibrator.__init__` computes an initial closed-form affine
  guess from frame 0 (post-reset state, before anything has moved) --
  but that alone was NOT accurate enough (confirmed 0.28-0.30m eef error at
  large displacements; root cause of the analytic model's error not fully
  pinned down). `MobileBaseCalibrator.apply()` then Newton-corrects that
  guess every frame against the *simulator's own* forward kinematics (a
  numeric Jacobian measured once at init, since the true relationship should
  be affine/constant), driving the residual to ~1e-4 or below regardless of
  what the true underlying relationship actually is. Up to 8 iterations,
  falls back to the best-seen state if it doesn't converge (needed: the
  recording pipeline parks the robot at a `[10.0, 10.0, 0.0]` sentinel/dummy
  position for trailing episode-end padding frames, which isn't a real
  target and made the solver diverge >10m before the `is_sentinel_root_pose`
  guard in `reconstruct()` started holding the last real frame's pose
  instead of chasing it). If a task ever needs the torso-height axis
  animated too, it isn't handled yet -- not observed to matter so far since
  `end_effector_pose` already matches without touching it.
- **Fixture *class* selection is reproduced by `set_ep_meta`, exact
  *placement* is not**: `episode_meta["fixtures"][<name>]` only stores
  `{"cls": "Toaster"}` -- which appliance/fixture *type* occupies a named
  slot -- not its resolved position. Exact along-counter position is drawn
  from a random sample region (`sample_region_kwargs`, seen as
  `placement.pos: [None, 1.0]` -- the `None` axis is the randomly-sampled
  one) at build time, and isn't captured by `set_ep_meta` at all. Confirmed
  by direct measurement: `toaster_oven_main_group` built at world x=2.363 vs.
  recorded x=2.050 (0.31m off); `toaster_main_group` off by 0.13m -- both
  purely along the randomized axis. Fix: `apply_fixture_positions()` in
  `reconstruct_video.py` overrides every fixture's `<name>_main` body's
  `sim.model.body_pos`/`body_quat` directly from
  `privileged_static_info.scene_layout.fixtures[<name>].position`/`euler`,
  once after reset (fixtures are static within an episode, no per-frame
  animation needed beyond the door/knob joints already handled). Verified
  err=0.0 for the 2 confirmed-mismatched fixtures plus 2 spot-checked
  already-correct ones (`stove_main_group`, `cab_1_main_group`) after the
  fix.
- **Camera**: original `task.mp4` matches `robot0_agentview_left` — checked
  by rendering all 4 static cameras (`agentview_center`, `agentview_left`,
  `agentview_right`, `frontview`) for frame 0 and picking the lowest
  pixel-diff match (24.8 vs 50+ for the others).
- **GPU required, not just recommended**: `env.reset()` hangs indefinitely
  (5+ min, not just slow) on a CPU-only login node, with either
  `MUJOCO_GL=egl` (no GPU present — `nvidia-smi` missing, no `/dev/nvidia0`)
  or `MUJOCO_GL=osmesa` (CPU software rendering also hung). On an actual GPU
  node (`srun --gpus-per-node=l40s:1 ...`, see `run_reconstruct.sh`),
  `reset()`+render completes in ~5s.
- **Sentinel/padding frames at episode end**: once an episode is actually
  done, the recording pipeline pins `robot.root_pose` to exactly
  `[10.0, 10.0, 0.0]` for the trailing frame(s) -- not a real position
  (confirmed: `robot.link_poses` for those same frames still shows real,
  slowly-settling values, only `root_pose` itself gets the dummy). Solving
  for it produced an 11.7m Newton divergence before `is_sentinel_root_pose()`
  started holding the last real frame's full state instead.

## Randomness audit (per-source: hardcoded/recorded value used, or still RNG)

Systematic pass over every place scene generation could depend on `env.rng`
(or any other randomness) instead of a value this repo actually has recorded,
per this principle: prefer a fixed/hardcoded constant, then prefer whatever
`privileged_information_<N>.json` documents, and only fall back to random
sampling where truly nothing else is available.

| What | Source of truth used | Verified how |
|---|---|---|
| Robot arm+gripper joints | `robot.joint_positions` (recorded, every frame) | direct set, no randomness involved |
| Mobile base xy/yaw | `robot.root_pose` (recorded, every frame), Newton-solved into joint space | eef error 0.0004-0.0024m across 9 spot-checked frames |
| Torso height | `robot.link_poses["mobilebase0_support"].position[2]` (recorded, every frame) -- previously left at whatever `reset()` gave it | added in this pass; folded into the same Newton solve as xy/yaw |
| Object poses (bread/basket/etc.) | `scene.objects[<name>].pose` (recorded, every frame) | direct set, no randomness |
| Fixture door/knob/drawer angles | `scene.fixtures[<name>].joints[<joint>].qpos` (recorded, every frame) | direct set, no randomness |
| Fixture static position/orientation | `scene_layout.fixtures[<name>].position`/`euler` (recorded, static) -- **was** randomly resampled at build time (see fixture-position bug above) | `apply_fixture_positions()`; verified err=0.0 |
| Fixture *class* (which appliance/cabinet type occupies a slot) | Deterministic from `layout_id` (part of the kitchen layout YAML, not RNG at all) | audited all 99 fixtures for this episode: 0 class mismatches |
| Fixture material/texture (style-driven, e.g. cabinet door panel look) | Deterministic from `style_id` (`load_style_config()` is pure YAML lookup + dict merge, no RNG) | traced end-to-end for the one case that looked wrong (see "investigated and retracted" above) -- was never actually a bug |
| Layout/style/object-config/robot-init-pose | `episode_meta` (`layout_id`, `style_id`, `object_cfgs`, `init_robot_base_pos`, `init_robot_base_ori`) via `set_ep_meta()` | these are the only keys `kitchen.py`'s reset path actually reads back out of `_ep_meta` (grepped every `self._ep_meta[...]`/`.get(...)` call site) |
| Fixture references (`fixture_refs`) | `episode_meta["fixture_refs"]` via `set_ep_meta()` | read at `kitchen.py:1009` |
| Generative wall/counter/floor textures (`cab_tex`/`counter_tex`/`wall_tex`/`floor_tex`) | `episode_meta["gen_textures"]` via `set_ep_meta()`, **if** `self.generative_textures` is truthy | for this episode `gen_textures={}` and the whole block is gated off (`generative_textures` falsy), so `get_random_textures(self.rng)` never runs here. **Caveat found but not hit**: the gating code treats an empty dict the same as "not provided" (`if self._curr_gen_fixtures is None or == {}: resample`), so if a *different* episode has `generative_textures` enabled AND a genuinely-empty recorded `gen_textures`, this would still resample randomly instead of using the (empty-but-real) recorded value. Not applicable to any episode checked so far. |
| **Camera config (`cam_configs`)** | **Gap, confirmed unused**: `kitchen.py`'s `get_ep_meta()` writes `ep_meta["cam_configs"]`, but grepping every `self._ep_meta[...]`/`.get(...)` read site in the file shows it is **never read back**. If a task ever has `env.randomize_cameras=True` (confirmed gated off by default, not the cause of anything checked so far), this repo's reconstruction would not reproduce the randomized camera pose -- there's no consuming code path in robocasa to feed it back into, not just a gap on our side. |
| Object *asset variant* (e.g. which bread/basket 3D model) | `episode_meta["object_cfgs"][i]["info"]["mjcf_path"]` (recorded, exact asset file) via `set_ep_meta()` -> `object_cfgs` | not independently re-verified this pass, but structurally the same read path as `layout_id`/`style_id` above |

## Usage

Grab a GPU node (adjust partition/qos/exclude list to your cluster — this
one mirrors `../../run_scripts/eval_groot_single_task.sh`):

```bash
TASK=ArrangeBreadBasket EPISODE=0 sbatch run_reconstruct.sh
# or, from an interactively-allocated GPU node:
TASK=ArrangeBreadBasket EPISODE=0 bash run_reconstruct.sh
```

This writes:
- `output/<Task>/episode_<N>_reconstructed.mp4`
- `output/<Task>/episode_<N>_comparison.json` (per-frame mean-abs-diff
  / SSIM against the original, plus a summary and a flag for original-video
  frames that are corrupted per the known-noise-frame heuristic)

The viewer (`../../viewer/`) automatically picks these up and shows the
reconstructed video side-by-side with the original, plus the comparison
summary, if they exist for the currently-viewed episode.

## Standalone scripts (no sbatch)

```bash
# must already be on a GPU node / inside an salloc allocation, cwd != SafeManip repo
python3 reconstruct_video.py --privileged_info /path/to/privileged_information_0.json \
    --output /tmp/out.mp4

python3 compare_frames.py --original /path/to/task.mp4 --reconstructed /tmp/out.mp4 \
    --privileged_info /path/to/privileged_information_0.json --output /tmp/comparison.json
```
