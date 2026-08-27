#!/usr/bin/env python3
"""
Reconstruct `privileged_information_<N>.json` files (schema-identical to the
files the *live* SafeManip eval pipeline writes during a policy rollout --
see `Isaac-GR00T/gr00t/eval/simulation.py`'s `_save_privileged_record`) from
the official RoboCasa v1.0 lerobot *training* dataset, instead of from a live
rollout. This lets `run_monitor_on_privileged.py` (unmodified) evaluate the
symbolic monitor's LTL specs against ground-truth training demonstrations.

Why this works / integration point
-----------------------------------
The live pipeline's `_new_privileged_record` / `_append_privileged_record`
(Isaac-GR00T/gr00t/eval/simulation.py) call `raw_env.get_privileged_information(
trajectory_horizon=...)` once per step, where `raw_env` is the underlying
`robocasa.environments.kitchen.kitchen.Kitchen` (sub)class instance.
`get_privileged_information()` (defined directly on that class, see
`robocasa/robocasa/environments/kitchen/kitchen.py`) is a *pure* function of
`self.sim` / `self.robots` / `self.fixtures` / `self.objects` at the moment
it's called, plus a handful of small accumulators it keeps on `self`
(`_privileged_history`, `_privileged_static_cache`, `_privileged_prev_eef_pose`,
`_privileged_prev_time`) and one more kept by the predicate module itself
(`_predicate_monitor_state`, set by `robocasa.environments.kitchen.predicates
.build_predicate_snapshot`). Critically, it does **not** read the action the
policy took to get there -- only the resulting simulator state. That means:

  - We can call the *exact same* `env.get_privileged_information()` method
    against a ground-truth-state-loaded env (via `reset_to`, exactly like
    `replay/official_playback/reconstruct_training_data.py` does for video)
    instead of a live rollout step, and get identical predicate-computation
    code paths -- not a reimplementation.
  - No `action` column is needed. Confirmed by inspection of
    `robocasa/robocasa/environments/kitchen/predicates.py`: no predicate
    reads `self.action`/any action-like attribute; every predicate is a
    function of qpos/qvel/contacts/site & body poses etc., all of which are
    fully determined by the flattened MuJoCo state we already load per frame.
  - Because the accumulators above live on the `env` object itself (not
    inside `get_privileged_information`'s locals), and are **not** cleared
    by `env.reset()` (confirmed by reading `kitchen.py`: they're only ever
    read/written inside `get_privileged_information`/`predicates.py`, never
    touched by `_reset_internal`/`reset`), we must manually clear them
    ourselves at the start of every new episode when reusing one `env`
    object across episodes -- otherwise things like the cumulative contact
    force or the persistence-frame counters used by predicates like
    `object_grasped`/`fixture_open_obstacle_hit` would silently carry over
    from the previous episode. See `_reset_privileged_accumulators` below.

CRITICAL FIX -- `env.timestep` must be advanced manually, or every
persistence-threshold predicate silently breaks:
-----------------------------------------------------------------------
`robocasa.environments.kitchen.predicates.build_predicate_snapshot` guards
its persisted `_predicate_monitor_state` with a "did the episode restart"
check: it reads `dynamic_info["task"]["timestep"]` (== `env.timestep`) and
wipes the whole monitor_state dict back to fresh whenever
`current_timestep <= previous_timestep` (see predicates.py's
`timestep_restarted` check, ~line 1805). `env.timestep` is a plain int that
robosuite's base env (`robosuite/environments/base.py`) only increments
inside `step()` (`self.timestep += 1`); `reset_to`'s pure state-loading path
(`env.sim.set_state_from_flattened(...)` + `env.sim.forward()`) never touches
it. Since this script deliberately never calls `env.step()` (that would
require actions, which is the whole point of avoiding), `env.timestep` stays
frozen wherever `env.reset()` last set it (0) for every single frame of the
episode -- which means `current_timestep(0) <= previous_timestep(0)` is true
on *every* call, and `monitor_state` gets reset to fresh on *every* frame.

This was not a hypothetical concern -- it was caught empirically: an initial
version of this script produced `object_grasped` as `False` for an entire
episode of a "pick up the bread" task with recorded success=True, despite
`OU.check_obj_grasped(env, "bread")` (the same underlying contact+qpos check)
independently confirming a grasp was in fact held for ~90 consecutive frames.
Instrumenting `env._predicate_monitor_state` frame-by-frame showed
`object_grasp_candidate_count` pinned at 1 forever instead of incrementing --
exactly what the "reset every frame" bug produces, and disappeared entirely
once `env.timestep` was advanced manually (see below). Every predicate that
depends on a consecutive-frame counter (`object_grasped`,
`object_grasped_safe`, `object_settled`, `*_PERSISTENCE_FRAMES`-gated
contact/obstacle-hit predicates, skill-onset detectors, microwave-occupancy
smoothing, ...) -- i.e. most of the temporal logic in predicates.py -- is
affected by this, not just grasp detection.

Fix applied here: `extract_episode` sets `env.timestep` to a strictly
increasing value (`t + 1`, mirroring what a real `env.step()` call would have
left it at after step `t`) immediately before each
`get_privileged_information()` call, so `build_predicate_snapshot` sees the
same "always moving forward, never restarting" timestep sequence it would
during a live rollout, and its persisted monitor_state survives across the
whole episode as intended. This is a one-line, well-isolated correction (not
a change to any shared predicates.py/kitchen.py code) -- `env.timestep` has
no other reader in the predicate-computation path besides this restart guard
(confirmed by grep), so overwriting it here is safe.

TEMPORAL-PERSISTENCE-THRESHOLD GRANULARITY MISMATCH, AND THE `--call_stride`
FLAG THAT FIXES IT (read this before comparing violation counts 1:1 against
a live eval rollout's monitor output for "the same" property):
-----------------------------------------------------------------------
Live rollouts call `get_privileged_information()` once per **outer**
`MultiStepWrapper.step()` call (see `Isaac-GR00T/gr00t/eval/wrappers/
multistep_wrapper.py`), which internally executes `n_action_steps` (e.g. 16)
*inner* raw environment steps per call, but only the state after the last
inner step gets a privileged-info snapshot appended. So one live "monitor
frame" == `n_action_steps` raw simulator steps.

This matters more than just "the frame axis is coarser live" -- it's a real
correctness issue, verified by reading the actual persistence-counter code in
`robocasa/robocasa/environments/kitchen/predicates.py`, not assumed:
**every persistence/settle-timeout counter in that file increments by
exactly 1 per *call* to `build_predicate_snapshot`
(`get_privileged_information()`'s callee), not per elapsed raw frame or
`env.timestep` delta.** E.g. (grepped and read directly, not inferred):
  - `object_grasped`'s `candidate_count = previous_candidate_count + 1`
    (~line 1964), gated by `OBJECT_GRASPED_PERSISTENCE_FRAMES = 2`
  - the settle-timeout watchdog's `settle_watch_age += 1` (~line 2031),
    gated by `SETTLE_TIMEOUT_FRAMES = 6`
  - contact-persistence / forbidden-contact on/off smoothing counters
    (`_rfc_on_count`/`_foc_on_count`/etc.), gated by
    `CONTACT_PERSISTENCE_FRAMES = 1`
Since live rollouts call `build_predicate_snapshot` once every
`n_action_steps` (e.g. 16) raw frames, "N persistence-frames" there means
roughly `N * n_action_steps` raw frames / `N * n_action_steps / 20` seconds
of real time. If this script called `get_privileged_information()` at every
raw frame (one call per `states.npz` row, ~20fps), the exact same integer
thresholds would instead resolve in `N` raw frames / `N / 20` seconds --
about 16x too fast, silently making persistence-gated predicates
(`object_grasped`, `object_settled`, contact/obstacle-hit predicates,
forbidden-contact smoothing) far more trigger-happy or quick-to-timeout than
the same episode would show live. This is a real correctness bug, not a
cosmetic granularity difference, and rescaling a threshold *downstream* (e.g.
in `SafeManip/monitor/specs.py`'s LTL-evaluation-side `SETTLE_TIMEOUT_FRAMES`
-- a different constant, used by `run_monitor_on_privileged.py` to interpret
the *recorded* predicate-value sequence, not by `predicates.py` while
computing those values) would not fix it, since most of the affected counters
live inside `predicates.py`'s own per-call state, upstream of that.

**Fix: `--call_stride N` (default 1) scales predicates.py's persistence/
onset-detection frame-count constants by N, while `get_privileged_information()`
is still called on *every single raw frame*, always** -- this was originally
built as a skip-calls mechanism (only call every Nth frame, trading temporal
resolution for correctness), but scaling the constants instead gets both
properties at once: full per-frame resolution *and* correct real-time
persistence semantics, with no tradeoff. (If you're reading old comments or
an old comparison and see references to "skipping calls" for call_stride,
that was the earlier, inferior design -- this module now always calls every
frame regardless of call_stride's value.)

Exactly 15 module-level constants in `predicates.py` have "FRAME"/"FRAMES" in
their name (grepped and enumerated in `_PREDICATES_FRAME_CONSTANTS` below;
double-checked every other module-level constant in that file isn't secretly
a disguised frame-count under a different name -- `CLUTTER_THRESHOLD` counts
*objects*, and `TARGET_REGION_BLOCKED_THRESHOLD` is defined but never
actually used anywhere in the file). All 15 get multiplied by `call_stride`
for the duration of extraction (monkeypatched onto the live `predicates`
module object and restored afterward -- see `_scale_predicates_frame_constants`).
A separate, unrelated `SETTLE_TIMEOUT_FRAMES` in `run_monitor_on_privileged.py`
(used downstream to interpret the *recorded* predicate-value sequence, not by
predicates.py while computing it) gets the same scaling for consistency (see
`_scale_settle_timeout_for_monitor`).

  - `--call_stride 1` (default): constants unscaled -- current, historically
    validated behavior, but persistence-gated predicates' effective real-time
    thresholds are ~16x too short compared to live rollouts (confirmed
    empirically: caused false-positive violations on a real training episode,
    e.g. a released object flagged as "failed to settle" after just 6 raw
    frames / 0.3s instead of the intended ~4.8s). Kept as the default only
    because it's the originally-validated behavior, not because it's
    recommended -- prefer `--call_stride 16` (or whatever your comparison
    target's `n_action_steps` is) for anything where false positives matter.
  - `--call_stride N` for N matching the live pipeline's `n_action_steps`
    (commonly 16 in configs seen in this repo, but this is an eval-config
    value, not a universal constant -- confirm it for whatever live run
    you're comparing against) restores the intended real-time persistence
    budget, at full per-frame temporal resolution -- no tradeoff, unlike the
    skip-based design this replaced.

Usage (single episode):
    python3 extract_privileged_from_dataset.py --task ArrangeBreadBasket \
        --episode 0 [--dataset_root ~/flash/datasets/robocasa/v1.0/target] \
        [--output_root output] [--trajectory_horizon 128] [--run_monitor]

Usage (range of episodes, one env reused across them):
    python3 extract_privileged_from_dataset.py --task ArrangeBreadBasket \
        --episode_start 0 --episode_end 9 --run_monitor

Must run inside the `robocasa` conda env, on a GPU node (MUJOCO_GL=egl) --
same requirement as replay/official_playback/reconstruct_training_data.py --
see run_extract_privileged_from_dataset.sbatch.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _desanitize_sys_path():
    """See replay/official_playback/reconstruct_training_data.py /
    replay/privileged_info_reconstruction/reconstruct_video.py's docstrings:
    drop cwd/'' from sys.path so `import robocasa`/`import gr00t` can't
    resolve to an empty namespace package shadowing the real editable
    install."""
    import os
    cwd = os.getcwd()
    sys.path[:] = [p for p in sys.path if p not in ("", cwd)]


_desanitize_sys_path()

import numpy as np  # noqa: E402


DEFAULT_DATASET_ROOT = "~/flash/datasets/robocasa/v1.0/target"
THIS_DIR = Path(__file__).parent
DEFAULT_OUTPUT_ROOT = THIS_DIR / "output"

# Attributes get_privileged_information()/build_predicate_snapshot() stash on
# the env object itself and never clear on their own (see module docstring)
# -- must be wiped before starting a fresh episode on a reused env.
_PRIVILEGED_ACCUMULATOR_ATTRS = (
    "_privileged_history",
    "_privileged_static_cache",
    "_privileged_prev_eef_pose",
    "_privileged_prev_time",
    "_predicate_monitor_state",
)

# Every module-level persistence/onset/grace-period constant in
# robocasa/environments/kitchen/predicates.py whose name contains
# "FRAME"/"FRAMES" (confirmed by grep -- exactly 15, no others hiding under a
# different name; double-checked the remaining non-FRAME-named constants:
# CLUTTER_THRESHOLD counts *objects*, not frames, and
# TARGET_REGION_BLOCKED_THRESHOLD is defined but never actually used anywhere
# in the file). Each increments by 1 per *call* to build_predicate_snapshot
# (see module docstring's granularity-mismatch section) -- scaling all of
# them by `call_stride` before extraction, while still calling
# get_privileged_information() every single raw frame, gives correct
# real-time persistence semantics *and* full per-frame temporal resolution
# simultaneously (better than the previous skip-based approach, which traded
# resolution for correctness instead of getting both).
_PREDICATES_FRAME_CONSTANTS = (
    "CONTACT_PERSISTENCE_FRAMES",
    "OBJECT_GRASPED_PERSISTENCE_FRAMES",
    "RELATIVE_SPEED_PERSISTENCE_FRAMES",
    "GRASP_SAFE_GRACE_FRAMES",
    "STABLE_PERSISTENCE_FRAME",
    "CONTENT_STABLE_PERSISTENCE_FRAMES",
    "FIXTURE_OUTPUT_IDLE_FRAMES",
    "MICROWAVE_EMPTY_PERSISTENCE_FRAMES",
    "MICROWAVE_OCCUPANCY_PERSISTENCE_FRAMES",
    "SETTLE_TIMEOUT_FRAMES",
    "SKILL_ONSET_FRAMES",
    "PLACE_ONSET_FRAMES",
    "DUMP_ONSET_FRAMES",
    "GRASPED_RECEPTACLE_UPRIGHT_GRACE_FRAMES",
    "PICK_APPROACH_PERSISTENCE_FRAMES",
)


def _scale_predicates_frame_constants(call_stride):
    """Monkeypatch predicates.py's module-level constants (bare globals,
    referenced by name inside its own functions -- NOT imported elsewhere via
    `from ... import NAME`, so setting the attribute on the module object
    here *does* affect every subsequent call into predicates.py, since
    Python resolves a bare global name from the enclosing module's __dict__
    at call time, not at function-definition time). Returns the original
    values so they can be restored afterward -- this mutates shared,
    importable state for the whole process, so every caller of this must
    pair it with `_restore_predicates_frame_constants`.

    Only meaningful when call_stride > 1; at call_stride == 1 this is a
    no-op (scale factor 1)."""
    import robocasa.environments.kitchen.predicates as predicates_mod

    originals = {}
    for name in _PREDICATES_FRAME_CONSTANTS:
        original = getattr(predicates_mod, name)
        originals[name] = original
        setattr(predicates_mod, name, original * call_stride)
    return originals


def _restore_predicates_frame_constants(originals):
    import robocasa.environments.kitchen.predicates as predicates_mod

    for name, original in originals.items():
        setattr(predicates_mod, name, original)


def _scale_settle_timeout_for_monitor(call_stride):
    """`run_monitor_on_privileged.py` has its own, separate
    SETTLE_TIMEOUT_FRAMES constant (imported via `from monitor.specs import
    SETTLE_TIMEOUT_FRAMES`, which binds a local copy into that module's own
    namespace at import time -- so patching monitor.specs.SETTLE_TIMEOUT_FRAMES
    would NOT affect it; must patch the already-imported name directly on
    monitor.run_monitor_on_privileged instead). This is used downstream, to
    interpret the *recorded* predicate-value sequence for the settle-timeout
    LTL properties -- a different constant from predicates.py's own
    SETTLE_TIMEOUT_FRAMES (which gates the settle_watch_age counter *while
    computing* predicate values in the first place, already covered by
    _scale_predicates_frame_constants above). Both need scaling consistently
    for the same call_stride."""
    import monitor.run_monitor_on_privileged as run_monitor_mod

    original = run_monitor_mod.SETTLE_TIMEOUT_FRAMES
    run_monitor_mod.SETTLE_TIMEOUT_FRAMES = original * call_stride
    return original


def _restore_settle_timeout_for_monitor(original):
    import monitor.run_monitor_on_privileged as run_monitor_mod

    run_monitor_mod.SETTLE_TIMEOUT_FRAMES = original


def find_dataset_dir(dataset_root, task):
    """Locate <dataset_root>/{composite,atomic}/<task>/<date>/lerobot. Each
    task has exactly one date subfolder (confirmed for all 50 target tasks) --
    mirrors replay/official_playback/reconstruct_training_data.py."""
    dataset_root = Path(dataset_root).expanduser()
    for category in ("composite", "atomic"):
        task_dir = dataset_root / category / task
        if not task_dir.is_dir():
            continue
        dates = sorted(p for p in task_dir.iterdir() if p.is_dir())
        for date_dir in dates:
            candidate = date_dir / "lerobot"
            if (candidate / "extras" / "dataset_meta.json").is_file():
                return candidate
    raise FileNotFoundError(f"no lerobot dataset found for task {task!r} under {dataset_root}")


def make_env(dataset_dir):
    """Identical to reconstruct_training_data.py's make_env -- duplicated
    (not imported) to avoid a fragile cross-directory import into
    replay/official_playback (different part of the repo, own module-loading
    conventions); this is ~10 lines, not worth an awkward sys.path hack."""
    import robosuite
    import robocasa.utils.lerobot_utils as LU

    env_meta = LU.get_env_metadata(dataset_dir)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = False  # no video rendering needed for this pipeline
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def _reset_privileged_accumulators(env):
    for attr in _PRIVILEGED_ACCUMULATOR_ATTRS:
        if hasattr(env, attr):
            delattr(env, attr)


def _to_json_serializable(value):
    """Defensive fallback serializer. get_privileged_information() already
    recursively converts numpy arrays/scalars to plain python via its own
    internal `_to_serializable` closures, but this is cheap insurance against
    any numpy scalar (np.bool_, np.float32, ...) slipping through, exactly
    mirroring Isaac-GR00T/gr00t/eval/simulation.py's `_to_json_serializable`
    so the two pipelines can't silently diverge in how they encode types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_serializable(v) for v in value]
    return value


def extract_episode(env, dataset_dir, ep_num, trajectory_horizon, call_stride=1):
    """Ground-truth-state-load *every* raw frame of one episode and call
    `env.get_privileged_information()` on every single one -- full temporal
    resolution, always, regardless of `call_stride`. `call_stride` instead
    scales predicates.py's persistence/onset-detection frame-count constants
    (see module docstring's "TEMPORAL-PERSISTENCE-THRESHOLD GRANULARITY
    MISMATCH" section) for the duration of this call, so real-time
    persistence semantics are correct *and* every frame still gets a
    privileged-info snapshot -- this gets both properties at once, unlike an
    earlier version of this script that instead skipped calls (traded
    resolution for correctness rather than getting both).

    Returns a dict with the same top-level schema as what
    Isaac-GR00T/gr00t/eval/simulation.py's `_save_privileged_record` writes:
    {"privileged_static_info", "privileged_dynamic_info", "replay_summary"}.
    """
    import robocasa.utils.lerobot_utils as LU
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    states = LU.get_episode_states(dataset_dir, ep_num)
    model_xml = LU.get_episode_model_xml(dataset_dir, ep_num)
    ep_meta = LU.get_episode_meta(dataset_dir, ep_num)

    _reset_privileged_accumulators(env)  # fresh episode: no leaked accumulator state
    initial_state = dict(states=states[0], model=model_xml, ep_meta=json.dumps(ep_meta))
    reset_to(env, initial_state)
    # reset_to's "model" branch calls env.reset(), which does NOT clear the
    # privileged/predicate accumulator attrs on env (see module docstring) --
    # clear again defensively in case reset() itself ever sets any of them
    # from stale class-level defaults in some env subclass.
    _reset_privileged_accumulators(env)

    call_stride = max(1, int(call_stride))
    static_info = None
    dynamic_frames = []
    traj_len = states.shape[0]
    for t in range(traj_len):
        reset_to(env, {"states": states[t]})
        # Must advance manually -- reset_to() never calls env.step(), the only
        # place robosuite's base env increments env.timestep, so it would
        # otherwise stay frozen at 0 for the whole episode and reset
        # predicates.py's persisted monitor_state on every single frame
        # (see module docstring's "CRITICAL FIX" section for how this was
        # actually caught). t+1 mirrors what a real env.step() call would
        # have left env.timestep at after step t.
        env.timestep = t + 1
        info = env.get_privileged_information(trajectory_horizon=trajectory_horizon)
        if static_info is None:
            static_info = _to_json_serializable(info["static"])
        dynamic_frames.append({
            "step": int(t),
            "data": _to_json_serializable(info["dynamic"]),
        })

    success = None
    try:
        success = bool(env._check_success())
    except Exception:
        pass

    return {
        "privileged_static_info": static_info,
        "privileged_dynamic_info": dynamic_frames,
        "replay_summary": {
            "task_name": env.__class__.__name__,
            "task_description": ep_meta.get("lang"),
            "seed": None,
            "split": "train",
            "episode_idx": int(ep_num),
            "replayed_episode_length": int(traj_len),
            "success": success,
            "source": "official_dataset_ground_truth_replay",
            "dataset_dir": str(dataset_dir),
            "call_stride": call_stride,
        },
    }


def process_task(task, dataset_root, output_root, episodes, trajectory_horizon,
                  run_monitor, skip_existing, call_stride=1):
    dataset_dir = find_dataset_dir(dataset_root, task)
    out_dir = Path(output_root) / task
    out_dir.mkdir(parents=True, exist_ok=True)

    import robocasa.utils.lerobot_utils as LU
    n_available = len(LU.get_episodes(dataset_dir))
    episodes = [ep for ep in episodes if ep < n_available]

    print(f"[{task}] dataset={dataset_dir} episodes={episodes} (of {n_available} available)", flush=True)

    call_stride = max(1, int(call_stride))
    # Scaled once for the whole task (not per-episode/per-frame -- the scale
    # factor doesn't change mid-run), restored in `finally` so a mid-task
    # exception can't leave predicates.py's shared, importable module state
    # mutated for whatever runs in this process after this function returns.
    original_frame_constants = _scale_predicates_frame_constants(call_stride)
    if call_stride > 1:
        print(f"[{task}] call_stride={call_stride}: scaled predicates.py's 15 "
              f"persistence-frame constants by {call_stride}x for correct "
              f"real-time semantics at full per-frame resolution", flush=True)

    env = None
    summary = []
    try:
        for ep in episodes:
            out_path = out_dir / f"privileged_information_{ep}.json"
            monitor_out_path = out_dir / f"privileged_information_{ep}_monitor.json"
            if skip_existing and out_path.is_file() and (not run_monitor or monitor_out_path.is_file()):
                print(f"[{task}] episode {ep}: skip (already exists)", flush=True)
                summary.append({"episode": ep, "status": "skipped"})
                continue

            t0 = time.time()
            try:
                if env is None:
                    env = make_env(dataset_dir)
                payload = extract_episode(env, dataset_dir, ep, trajectory_horizon, call_stride=call_stride)
                out_path.write_text(json.dumps(payload, indent=2))
                n_frames = len(payload["privileged_dynamic_info"])
                elapsed = round(time.time() - t0, 2)
                print(f"[{task}] episode {ep}: extracted {n_frames} frames in {elapsed}s -> {out_path}", flush=True)
                entry = {"episode": ep, "status": "extracted", "n_frames": n_frames, "elapsed_s": elapsed}

                if run_monitor:
                    monitor_result = run_monitor_on(out_path, monitor_out_path, call_stride=call_stride)
                    entry["monitor_output"] = str(monitor_out_path)
                    entry["num_violated_instances"] = monitor_result.get("num_violated_instances")
                    entry["num_satisfied_instances"] = monitor_result.get("num_satisfied_instances")
                    print(
                        f"[{task}] episode {ep}: monitor -> "
                        f"{entry['num_violated_instances']} violated / "
                        f"{entry['num_satisfied_instances']} satisfied instances "
                        f"(of {monitor_result.get('num_property_instances')} total)",
                        flush=True,
                    )
                summary.append(entry)
            except Exception as e:
                print(f"[{task}] episode {ep}: FAILED: {e}", flush=True)
                summary.append({
                    "episode": ep, "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                # the env may be in a bad state after an exception; rebuild next episode
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = None
    finally:
        _restore_predicates_frame_constants(original_frame_constants)

    if env is not None:
        env.close()

    (out_dir / "task_extract_summary.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for s in summary if s.get("status") == "extracted")
    n_skip = sum(1 for s in summary if s.get("status") == "skipped")
    print(f"[{task}] finished: {n_ok} extracted, {n_skip} skipped, "
          f"{len(episodes) - n_ok - n_skip} failed", flush=True)
    return summary


def run_monitor_on(privileged_json_path, output_path, call_stride=1):
    """Call SafeManip/monitor/run_monitor_on_privileged.py's `monitor_rollout`
    directly (in-process function call, not subprocess -- simpler, and this
    script already runs inside the same repo) on our freshly-written
    privileged_information_<N>.json, writing the exact same
    `_monitor.json` schema `run_monitor_on_privileged.py`'s own CLI would.

    `call_stride` scales run_monitor_on_privileged.py's own SETTLE_TIMEOUT_FRAMES
    (a separate constant from predicates.py's, used to interpret the
    *recorded* predicate-value sequence for settle-timeout LTL properties --
    see _scale_settle_timeout_for_monitor's docstring) so it stays consistent
    with the scaling already applied to predicates.py during extraction."""
    repo_root = str(Path(__file__).resolve().parents[2])  # .../SafeManip
    monitor_pkg_root = str(Path(__file__).resolve().parents[1])  # .../SafeManip/SafeManip
    for p in (monitor_pkg_root, repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from monitor.run_monitor_on_privileged import monitor_rollout

    call_stride = max(1, int(call_stride))
    original_settle_timeout = _scale_settle_timeout_for_monitor(call_stride)
    if call_stride > 1:
        print(f"  (also scaled run_monitor_on_privileged.py's separate "
              f"SETTLE_TIMEOUT_FRAMES by {call_stride}x -- 16th constant, "
              f"not part of the 15 in predicates.py)", flush=True)
    try:
        result = monitor_rollout(str(privileged_json_path))
    finally:
        _restore_settle_timeout_for_monitor(original_settle_timeout)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    return result


def _parse_episode_list(args):
    if args.episode is not None:
        return [args.episode]
    if args.episode_start is not None:
        end = args.episode_end if args.episode_end is not None else args.episode_start
        return list(range(args.episode_start, end + 1))
    return list(range(args.n_episodes))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, help="single episode index")
    ap.add_argument("--episode_start", type=int, help="first episode index (inclusive) of a range")
    ap.add_argument("--episode_end", type=int, help="last episode index (inclusive) of a range; defaults to episode_start")
    ap.add_argument("--n_episodes", type=int, default=10,
                     help="used only if neither --episode nor --episode_start is given: episodes 0..n_episodes-1")
    ap.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--trajectory_horizon", type=int, default=128,
                     help="matches Isaac-GR00T PrivilegedInfoConfig.trajectory_horizon default; "
                          "only bounds the length of the recorded eef-position trajectory buffer, "
                          "does not affect predicate correctness")
    ap.add_argument("--run_monitor", action="store_true", default=True)
    ap.add_argument("--no-run_monitor", dest="run_monitor", action="store_false")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument(
        "--call_stride", type=int, default=1,
        help="get_privileged_information() is ALWAYS called on every raw frame regardless of "
             "this value -- full temporal resolution always. This instead scales predicates.py's "
             "15 persistence/onset-detection frame-count constants (OBJECT_GRASPED_PERSISTENCE_FRAMES, "
             "SETTLE_TIMEOUT_FRAMES, etc.) by this factor, since they increment once per call and "
             "were tuned assuming calls are n_action_steps raw frames apart (a live-rollout-only "
             "assumption -- this script calls every raw frame). Default 1 = unscaled = current, "
             "historically-validated behavior, but persistence-gated predicates resolve "
             "~n_action_steps times faster than in a live rollout (confirmed empirically to cause "
             "false-positive violations -- see module docstring). Set to the live pipeline's "
             "n_action_steps (commonly 16) to restore correct real-time persistence semantics, "
             "with no loss of temporal resolution. See module docstring's 'TEMPORAL-PERSISTENCE-"
             "THRESHOLD GRANULARITY MISMATCH' section before picking a value.",
    )
    args = ap.parse_args()

    episodes = _parse_episode_list(args)
    process_task(
        args.task, args.dataset_root, args.output_root, episodes,
        args.trajectory_horizon, args.run_monitor, args.skip_existing,
        call_stride=args.call_stride,
    )


if __name__ == "__main__":
    main()
