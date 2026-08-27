#!/usr/bin/env python3
"""
Reconstruct a rollout video from a recorded privileged_information_<N>.json,
by directly *posing* the simulator at each recorded frame and rendering —
not by replaying actions.

Why pose-setting instead of action replay: privileged_information_<N>.json
does not store the action sequence the policy took (see
SafeManip/replay/privileged_info_reconstruction/README.md for the investigation that established this), so
action-level replay isn't possible from this file alone. What it does store,
per recorded monitor frame, is the *absolute* state needed to pose the sim
directly:
  - robot.joint_positions (arm + gripper), robot.root_pose (mobile base)
  - scene.objects[<name>].pose (position + xyzw quaternion) for every
    manipulable object
  - scene.fixtures[<name>].joints[<joint>].qpos for every openable/movable
    fixture (cabinet doors, knobs, etc.)
This script sets all of that directly via `sim.data.set_joint_qpos(...)` and
calls `sim.forward()` (no physics stepping), then renders. Since every frame
is set from ground truth rather than integrated from the previous one, there
is no error accumulation across frames the way there would be with action
replay — the main source of mismatch against the original video is the
mobile-base joint <-> world-pose calibration (see MobileBaseCalibrator
below) and the roughly 1:8 monitor:video frame-rate mismatch (see
KNOWN_BUGS.md), not drift.

IMPORTANT: this must be run inside the `robocasa` conda env, on a machine/
node with a working GPU (MUJOCO_GL=egl) — env.reset() hangs indefinitely on
a CPU-only login node, with either egl or osmesa. See run_reconstruct.sh for
an sbatch wrapper, or grab an interactive GPU node first (e.g. see
../run_scripts/eval_groot_single_task.sh for the resource shape this repo
uses, or your own equivalent of `salloc -p ... --qos debug --gpus-per-node=1`).

IMPORTANT: do not run this with your shell's cwd set to the SafeManip repo
root (or anywhere containing a bare, non-package "robocasa" or "gr00t"
directory). Python's implicit cwd-in-sys.path resolves `import robocasa` to
an empty PEP-420 namespace package shadowing the real editable install,
which imports "successfully" but registers zero gym envs (silent, no
error). This script defends against that at import time (see
_desanitize_sys_path below), but avoid it in your own shells too.

Usage:
    python3 reconstruct_video.py \
        --privileged_info /path/to/privileged_information_0.json \
        --output /path/to/output.mp4 \
        [--camera robot0_agentview_left] [--fps 10] [--width 256] [--height 256]
"""
import argparse
import json
import sys
from pathlib import Path


def _desanitize_sys_path():
    """Drop cwd/'' from sys.path so `import robocasa`/`import gr00t` can't
    resolve to an empty namespace package shadowing the real editable
    install. See module docstring."""
    import os
    cwd = os.getcwd()
    sys.path[:] = [p for p in sys.path if p not in ("", cwd)]


_desanitize_sys_path()

import numpy as np  # noqa: E402


CAMERA_NAME_DEFAULT = "robot0_agentview_left"  # matches the original task.mp4 (verified by pixel diff against 4 candidate cameras)
MOBILE_BASE_JOINTS = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
    "mobilebase0_joint_torso_height",
)
TORSO_SUPPORT_BODY = "mobilebase0_support"  # z tracks torso_height; x/y/yaw match root_body


class MobileBaseCalibrator:
    """Converts an absolute world (xy, yaw) root pose + torso height into the
    4 scalar mobile-base joint values (forward/side/yaw/torso_height) this
    robot actually uses.

    The base is NOT a 6-DoF free joint (see robosuite's omron_mobile_base.xml):
    it's a serial chain of 2 slides (forward, side) + a hinge (yaw), plus an
    independent slide (torso_height) that raises/lowers the torso/arm/camera
    mount (`mobilebase0_support`) without moving the base's own xy/yaw. An
    initial closed-form affine guess for forward/side/yaw (assuming they're
    anchor-frame translations applied before yaw) gets the right ballpark,
    but empirically diverges from ground truth as displacement from the
    calibration frame grows -- confirmed by comparing simulator-computed eef
    site position against the recorded end_effector_pose: ~0.001-0.006m
    error near the calibration frame (frame 0/50 of ArrangeBreadBasket ep 0),
    but 0.28-0.30m error at frames 60/65/80 (large base displacement+
    rotation). That's larger than this robot's whole hand-to-basket grasp
    offset (~0.15m recorded), so it visibly separates held objects from the
    gripper even though the object's own recorded pose is correct.

    Root cause of the analytic model's error was not pinned down (candidates:
    axis-order/sign assumption wrong, or the true relationship isn't as
    simple as assumed) -- rather than keep guessing, we numerically measure
    the actual local Jacobian d(world_xy, world_yaw, support_z)/d(forward,
    side, yaw, torso_height) via finite differences ONCE (this is a rigid
    mechanism, so the Jacobian should be ~constant everywhere), then
    Newton-correct the analytic guess every frame against the simulator's
    own forward kinematics. torso_height's initial guess is just "leave it
    at whatever reset gave it" (recorded values only vary ~2mm across
    ArrangeBreadBasket ep 0, so this barely matters there, but the Newton
    correction targets the actual recorded `mobilebase0_support` height on
    every task regardless of how much it varies -- no reason to leave it
    unset just because it happened not to matter for one task).
    """

    def __init__(self, sim, reference_root_pose, root_body, reference_support_z):
        from scipy.spatial.transform import Rotation as R

        self._R = R
        self._root_body = root_body
        forward0 = sim.data.get_joint_qpos(MOBILE_BASE_JOINTS[0])
        side0 = sim.data.get_joint_qpos(MOBILE_BASE_JOINTS[1])
        yaw0 = sim.data.get_joint_qpos(MOBILE_BASE_JOINTS[2])
        torso0 = sim.data.get_joint_qpos(MOBILE_BASE_JOINTS[3])

        world_xy0 = np.array(reference_root_pose["position"][:2], dtype=float)
        world_yaw0 = R.from_quat(reference_root_pose["orientation"]).as_euler("xyz")[2]

        self.anchor_yaw = world_yaw0 - yaw0
        self._rot = R.from_euler("z", self.anchor_yaw).as_matrix()[:2, :2]
        self._rot_inv = np.linalg.inv(self._rot)
        self.anchor_xy = world_xy0 - self._rot @ np.array([forward0, side0])
        # support_z - torso_height should be a fixed mount offset; keep it as
        # the initial-guess seed, refined per-frame by Newton correction below.
        self.anchor_support_z_offset = reference_support_z - torso0

        self._jacobian_inv = self._measure_jacobian_inv(sim, forward0, side0, yaw0, torso0)

    def _measure_state(self, sim, forward, side, yaw, torso):
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[0], forward)
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[1], side)
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[2], yaw)
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[3], torso)
        sim.forward()
        bid = sim.model.body_name2id(self._root_body)
        xy = sim.data.xpos[bid][:2].copy()
        body_yaw = self._R.from_quat(_wxyz_to_xyzw(sim.data.xquat[bid])).as_euler("xyz")[2]
        support_z = sim.data.get_body_xpos(TORSO_SUPPORT_BODY)[2]
        return np.array([xy[0], xy[1], body_yaw, support_z])

    def _measure_jacobian_inv(self, sim, forward0, side0, yaw0, torso0, eps=1e-3):
        base = self._measure_state(sim, forward0, side0, yaw0, torso0)
        cols = []
        for i in range(4):
            perturbed = [forward0, side0, yaw0, torso0]
            perturbed[i] += eps
            cols.append((self._measure_state(sim, *perturbed) - base) / eps)
        jacobian = np.stack(cols, axis=1)  # d(x,y,yaw,support_z) / d(forward,side,yaw,torso)
        return np.linalg.inv(jacobian)

    def apply(self, sim, root_pose, support_z):
        world_xy = np.array(root_pose["position"][:2], dtype=float)
        world_yaw = self._R.from_quat(root_pose["orientation"]).as_euler("xyz")[2]
        target = np.array([world_xy[0], world_xy[1], world_yaw, support_z])

        forward, side = self._rot_inv @ (world_xy - self.anchor_xy)
        yaw = world_yaw - self.anchor_yaw
        torso = support_z - self.anchor_support_z_offset
        # Newton correction against the sim's own kinematics. Usually converges
        # in 1-2 steps (the true relationship is affine, so the Jacobian is
        # exact everywhere) -- but confirmed to occasionally fail to converge
        # within 2 iterations from a bad analytic starting point (observed:
        # frame 181/181 of ArrangeBreadBasket ep 0 diverged to an 11.7m error
        # with only 2 iterations). Iterate more and bail to the best-seen
        # state if it isn't converging, rather than silently accepting
        # whatever the last iteration produced.
        best = (forward, side, yaw, torso)
        best_err = np.inf
        for _ in range(8):
            current = self._measure_state(sim, forward, side, yaw, torso)
            error = target - current
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))  # wrap angle
            err_norm = np.linalg.norm(error)
            if err_norm < best_err:
                best_err = err_norm
                best = (forward, side, yaw, torso)
            if err_norm < 1e-5:
                break
            forward, side, yaw, torso = np.array([forward, side, yaw, torso]) + self._jacobian_inv @ error
        else:
            forward, side, yaw, torso = best  # loop exhausted without converging; use best-seen
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[0], forward)
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[1], side)
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[2], yaw)
        sim.data.set_joint_qpos(MOBILE_BASE_JOINTS[3], torso)


def _wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]


def apply_fixture_positions(sim, scene_layout_fixtures):
    """One-time (not per-frame) correction: fixture *class* selection is
    reproduced correctly by set_ep_meta(), but exact along-counter placement
    is drawn from a random sample region at build time and NOT captured by
    episode_meta['fixtures'] (which only stores {"cls": "..."}). Confirmed:
    toaster_oven_main_group built at x=2.363 vs recorded x=2.050 (0.31m off);
    toaster_main_group off by 0.13m -- both along the "None" (randomly
    sampled) axis of their `placement.pos` config. The exact recorded
    position IS available in privileged_static_info.scene_layout.fixtures,
    so we just override each fixture's body_pos directly to match. Skipped
    (silently) for any fixture name with no exactly-matching "<name>_main"
    body, or with no recorded position."""
    from scipy.spatial.transform import Rotation as R

    corrected = 0
    for name, info in (scene_layout_fixtures or {}).items():
        position = info.get("position")
        if position is None:
            continue
        body_name = f"{name}_main"
        try:
            bid = sim.model.body_name2id(body_name)
        except Exception:
            continue
        sim.model.body_pos[bid] = position
        euler = info.get("euler")
        if euler is not None:
            x, y, z, w = R.from_euler("xyz", euler).as_quat()  # scipy returns xyzw
            sim.model.body_quat[bid] = [w, x, y, z]  # MuJoCo wants wxyz
        corrected += 1
    sim.forward()
    return corrected


def xyzw_to_wxyz(q):
    return [q[3], q[0], q[1], q[2]]


_SENTINEL_ROOT_XY = (10.0, 10.0)


def is_sentinel_root_pose(root_pose):
    """Detect the recording pipeline's episode-end padding pose (observed:
    position pinned to exactly [10.0, 10.0, 0.0] for the trailing frame(s) of
    ArrangeBreadBasket ep 0, once the episode is actually over) -- not a
    real robot position."""
    pos = root_pose.get("position") or []
    return len(pos) >= 2 and pos[0] == _SENTINEL_ROOT_XY[0] and pos[1] == _SENTINEL_ROOT_XY[1]


def make_env(task_name, seed, split, episode_meta, reuse=None):
    """Build (or reconfigure) a robocasa gym env for one episode.

    `reuse=(env, raw)`: skip `gym.make()` (confirmed the expensive part --
    ~19s, one-time class/robot/controller construction) and just re-run
    `set_ep_meta()` + `reset()` (~6s, the actual per-episode kitchen
    procedural rebuild) on the existing env. Only valid for a *different
    episode of the same task* -- `gym.make()` is keyed to a specific
    `robocasa/<TaskName>` env id, so this cannot be reused across tasks.
    At the scale of reconstructing every episode of every task (2500
    episodes), skipping the repeated ~19s `gym.make()` for episodes 2+ of
    each task is roughly a 4x speedup over rebuilding from scratch every time.
    """
    if reuse is not None:
        env, raw = reuse
    else:
        import gymnasium as gym
        import robocasa  # noqa: F401  (registers robocasa/<Task> gym envs)

        env = gym.make(f"robocasa/{task_name}", split=split, enable_render=True, seed=seed)
        raw = env.unwrapped

    if episode_meta:
        raw.env.set_ep_meta(dict(episode_meta))
    # IMPORTANT: must pass seed= here too, not just to gym.make(). RoboCasaGymEnv.reset()
    # only does `self.env.rng = np.random.default_rng(seed)` when seed is not None (see
    # robocasa/wrappers/gym_wrapper.py) -- and that rng is what fixture/clutter/appliance
    # sampling draws from. Without re-passing it here, reset() samples a *different*
    # random scene than the one actually recorded, even with set_ep_meta() called first
    # (ep_meta locks layout/style/object placement, but not everything reset() samples).
    # Confirmed against github.com/chengyuehuang511/SafeManip commit 60d7a43 ("freeze
    # scene functionality"), which hit exactly this and fixes it via env.reset(seed=...).
    env.reset(seed=seed)
    return env, raw


def set_frame_state(sim, frame_data, calibrator, missing_joint_log):
    robot = frame_data["robot"]
    support_z = robot["link_poses"][TORSO_SUPPORT_BODY]["position"][2]
    calibrator.apply(sim, robot["root_pose"], support_z)
    for jn, jp in zip(robot["joint_names"], robot["joint_positions"]):
        try:
            sim.data.set_joint_qpos(jn, jp)
        except Exception:
            missing_joint_log.add(jn)

    for oname, ostate in frame_data["scene"]["objects"].items():
        pos = ostate["pose"]["position"]
        quat_wxyz = xyzw_to_wxyz(ostate["pose"]["orientation"])
        joint_name = f"{oname}_joint0"
        try:
            sim.data.set_joint_qpos(joint_name, np.array(list(pos) + quat_wxyz))
        except Exception:
            missing_joint_log.add(joint_name)

    for fname, fstate in frame_data["scene"]["fixtures"].items():
        joints = fstate.get("joints")
        if not joints:
            continue
        for jn, jinfo in joints.items():
            try:
                sim.data.set_joint_qpos(jn, jinfo["qpos"])
            except Exception:
                missing_joint_log.add(jn)

    sim.forward()


def probe_duration(video_path):
    """Video duration in seconds, via imageio/imageio-ffmpeg (not a shelled-out
    ffprobe call -- ffprobe isn't reliably on PATH inside the robocasa conda
    env, but imageio-ffmpeg already is, as a reconstruct_video.py dependency)."""
    import imageio

    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        duration = meta.get("duration")
        if duration:
            return float(duration)
        fps = meta.get("fps")
        n_frames = meta.get("nframes")
        if fps and n_frames and n_frames != float("inf"):
            return float(n_frames) / float(fps)
    finally:
        reader.close()
    return None


def reconstruct(privileged_info_path, output_path, camera, fps, width, height,
                 max_frames=None, frame_stride=1, original_video=None,
                 reuse_env=None, close_env=True):
    import imageio

    d = json.loads(Path(privileged_info_path).read_text())
    si = d["privileged_static_info"]
    episode_meta = si["task"].get("episode_meta")
    task_name = d.get("task_name") or d.get("replay_summary", {}).get("task_name") or si["task"].get("env_name")
    seed = d.get("replay_summary", {}).get("seed")
    split = d.get("replay_summary", {}).get("split", "target")
    dyn_frames = d["privileged_dynamic_info"]
    if max_frames:
        dyn_frames = dyn_frames[:max_frames]
    dyn_frames = dyn_frames[::frame_stride]

    # IMPORTANT: there is exactly one reconstructed frame per recorded monitor
    # frame, which is the same ~1:8-sparser rate as privileged_information vs.
    # the original task.mp4 (see KNOWN_BUGS.md). Rendering those frames at a
    # fixed fps like 10 (a reasonable-*looking* number, but wrong) makes the
    # reconstruction play back ~8x FASTER than the original episode -- same
    # frame count, way less real time. To keep both videos in sync, derive fps
    # from the original video's actual duration when available: fps = (number
    # of monitor frames) / (original video duration in seconds), so playing
    # back all `len(dyn_frames)` frames takes exactly as long as the original.
    if original_video is not None:
        duration = probe_duration(original_video)
        if duration:
            fps = len(dyn_frames) / duration
            print(f"[reconstruct] derived fps={fps:.4f} from original video "
                  f"duration={duration:.2f}s / {len(dyn_frames)} frames "
                  f"(overrides --fps)", flush=True)

    print(f"[reconstruct] task={task_name} seed={seed} split={split} "
          f"n_frames={len(dyn_frames)} (stride={frame_stride}) fps={fps}", flush=True)

    env, raw = make_env(task_name, seed, split, episode_meta, reuse=reuse_env)
    sim = raw.env.sim

    # IMPORTANT: don't assume frame 0 is real. The recording pipeline's
    # sentinel/padding pose ([10.0, 10.0, 0.0], root_body="robot0_base" --
    # a static, zero-joint body, hence *nothing* moves it) was first found
    # at *trailing* frames (episode already done), but confirmed to also
    # appear at *leading* frames on some episodes (observed: frames 0-1 of
    # ArrangeBreadBasket episode 2, before the mobile base has actually
    # spawned/settled into tracked state at frame ~2). Calibrating against
    # a sentinel frame produces a singular Jacobian (root_body has no
    # joints -> every perturbation column is zero) and a crash, not just a
    # wrong render -- so this isn't optional to handle. Scan for the first
    # genuinely non-sentinel frame and use that as the calibration
    # reference and as the initial "last valid" hold-state instead.
    first_valid_idx = next(
        (i for i, f in enumerate(dyn_frames) if not is_sentinel_root_pose(f["data"]["robot"]["root_pose"])),
        0,
    )
    if first_valid_idx > 0:
        print(f"[reconstruct] WARNING: frames 0..{first_valid_idx - 1} are sentinel/padding "
              f"(episode not yet started); calibrating against frame {first_valid_idx} instead "
              f"and holding its pose for the leading frames", flush=True)
    reference_frame = dyn_frames[first_valid_idx]["data"]

    root_body = reference_frame["robot"]["root_body"]
    reference_support_z = reference_frame["robot"]["link_poses"][TORSO_SUPPORT_BODY]["position"][2]
    calibrator = MobileBaseCalibrator(
        sim, reference_frame["robot"]["root_pose"], root_body, reference_support_z
    )
    n_fixed = apply_fixture_positions(sim, si["scene_layout"].get("fixtures"))
    print(f"[reconstruct] corrected {n_fixed} fixture positions to recorded ground truth", flush=True)
    missing_joints = set()

    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264",
                                 macro_block_size=None)
    last_valid_frame_data = reference_frame
    n_sentinel = 0
    root_body_mismatches = 0
    try:
        for i, frame in enumerate(dyn_frames):
            frame_data = frame["data"]
            if is_sentinel_root_pose(frame_data["robot"]["root_pose"]):
                # Recording pipeline parks the robot at a dummy off-screen pose
                # (observed: exactly [10.0, 10.0, 0.0]) once the episode is
                # done, as padding -- not a real position. Solving for it
                # produces nonsense (confirmed: >10m Newton divergence on the
                # last frame of ArrangeBreadBasket ep 0). Hold the last real
                # frame's state instead of rendering the sentinel literally.
                frame_data = last_valid_frame_data
                n_sentinel += 1
            else:
                # Not expected to vary among non-sentinel frames (only
                # observed changing alongside the sentinel pose itself), but
                # cheap to check rather than silently mis-target the
                # calibrator if it ever does.
                if frame_data["robot"]["root_body"] != root_body:
                    root_body_mismatches += 1
                last_valid_frame_data = frame_data
            set_frame_state(sim, frame_data, calibrator, missing_joints)
            img = sim.render(width=width, height=height, camera_name=camera)[::-1]
            writer.append_data(np.asarray(img, dtype=np.uint8))
            if i % 25 == 0:
                print(f"[reconstruct] frame {i}/{len(dyn_frames)}", flush=True)
    finally:
        writer.close()
        if close_env:
            env.close()

    if n_sentinel:
        print(f"[reconstruct] held last/first valid pose for {n_sentinel} sentinel/padding "
              f"frame(s)", flush=True)
    if root_body_mismatches:
        print(f"[reconstruct] WARNING: {root_body_mismatches} non-sentinel frame(s) reported a "
              f"different robot.root_body than the calibration reference ({root_body!r}) -- "
              f"calibration may be mistargeted for those frames", flush=True)

    if missing_joints:
        print(f"[reconstruct] WARNING: {len(missing_joints)} joint names never "
              f"resolved (left at default pose): {sorted(missing_joints)}", flush=True)

    # Confirmed real, upstream bug -- not caused by this script, not fixable
    # by posing the sim differently. Found by comparing, per episode, the
    # *static* fixture roster (privileged_static_info.scene_layout.fixtures
    # -- always internally consistent with that episode's own recorded
    # layout_id/style_id) against the *dynamic* per-frame fixture roster
    # (privileged_dynamic_info[i].scene.fixtures). For most episodes these
    # match; for some they don't, even with byte-identical seed/layout_id/
    # style_id/init_robot_base_pos across every episode of the same task
    # (checked: ArrangeBreadBasket ep2; ArrangeTea eps 40/42/44/45/46/47/48
    # of 40-49 checked -- only ep41 matched). The wrong names aren't random
    # garbage: ArrangeTea's wrong-episode dynamic data used names like
    # "toaster_main_group"/"toaster_oven_main_group", which are exactly
    # ArrangeBreadBasket's (a different task/layout's) real fixture names --
    # i.e. this looks like cross-episode (possibly cross-task) fixture-name
    # cache contamination in the *original* recording pipeline, not a
    # scrambled/corrupted field. A large fraction of unresolved joint names
    # (rather than the isolated single-item type-mix gaps seen on properly-
    # consistent episodes) is the fingerprint we detect on. Flag it rather
    # than silently produce a video with most fixtures frozen at default
    # poses.
    data_integrity_suspect = len(missing_joints) > 20 or root_body_mismatches > len(dyn_frames) * 0.1
    if data_integrity_suspect:
        print(f"[reconstruct] WARNING: data_integrity_suspect=True -- "
              f"{len(missing_joints)} missing joints / {root_body_mismatches} root_body "
              f"mismatches. Likely upstream cross-episode fixture-name cache contamination "
              f"in the source privileged_information file, not a reconstruction bug -- see "
              f"reconstruct_video.py's reconstruct() for how this was diagnosed.",
              flush=True)

    print(f"[reconstruct] wrote {output_path} at fps={fps}", flush=True)
    return {
        "missing_joints": sorted(missing_joints),
        "n_frames": len(dyn_frames),
        "fps": fps,
        "n_sentinel_frames": n_sentinel,
        "data_integrity_suspect": data_integrity_suspect,
        "first_valid_frame_idx": first_valid_idx,
        "root_body_mismatches": root_body_mismatches,
        "env": (env, raw),  # for reuse across episodes of the same task; caller closes when done
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--privileged_info", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--camera", default=CAMERA_NAME_DEFAULT)
    ap.add_argument("--fps", type=float, default=10.0,
                     help="Output video fps, used only if --original_video isn't given "
                          "(or its duration can't be probed). Prefer --original_video: "
                          "there is exactly one reconstructed frame per monitor frame, "
                          "~1:8 sparser than the original task.mp4 (see KNOWN_BUGS.md), "
                          "so a fixed fps like 10 makes playback ~8x too fast relative "
                          "to the original episode's real duration.")
    ap.add_argument("--original_video", type=Path, default=None,
                     help="path to the original task.mp4 for this episode; if given, fps "
                          "is derived from its duration so both videos play at the same "
                          "real-time speed (recommended over --fps).")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--meta_output", type=Path, default=None,
                     help="optional path to write a small json report (missing joints, etc.)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = reconstruct(
        args.privileged_info, args.output, args.camera, args.fps,
        args.width, args.height, args.max_frames, args.frame_stride,
        original_video=args.original_video,
    )
    report.pop("env", None)  # (env, raw) objects aren't JSON-serializable; not needed for CLI use
    if args.meta_output:
        args.meta_output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
