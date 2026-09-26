#!/usr/bin/env python3
"""Re-render the teaser's safety-category + monitoring frames at 1024x1024.

The teaser's sim frames were 256px (native resolution of both the drawio
embeds and the demo-corpus videos -- verified, no sharper raster exists).
This job re-renders the SAME episodes/moments from the official lerobot
extras (model.xml.gz + states.npz: the full flattened MuJoCo state every
timestep, ground-truth-exact per replay/official_playback/README.md) at
1024px, both agentview cameras, three candidate monitor-frames per cell.
Red violation circles are NOT baked here -- the teaser draws them as
vector overlays.

Monitor-frame -> video-frame mapping mirrors render_qualitative*.py's
frames_for(): vf = round(f / (n_mon - 1) * (T - 1)), with n_mon = len of
the property's recovery trace in the v31 monitor json and T = len(states)
(states are frame-aligned with the videos).

Run inside the `robocasa` conda env on a GPU node (MUJOCO_GL=egl):
    python hires_render_job.py
"""
import gzip
import json
import os

import numpy as np
from PIL import Image

ROOT = "/path/to/SafeManip"
DEMO_RC = os.path.join(ROOT, "SafeManip", "monitor", "output",
                       "v31_2026-09-20_claude_branch_place_precondition_"
                       "hygiene_removed")
DATA = os.path.expanduser("~/datasets/robocasa/v1.0/target/composite")
OUT = os.path.join(ROOT, "eval", "figures_out", "manipverse", "hires_frames")
CAMS = ("robot0_agentview_left", "robot0_agentview_right")
RES = 1024

# key, task, date, ep, property, candidate monitor frames
CELLS = [
    ("collision", "WashLettuce", "20250814", 7,
     "rc_no_forbidden_contact", [9, 11, 13]),
    ("grasp", "ArrangeBreadBasket", "20250809", 32,
     "rc_released_object_eventually_settles", [60, 110, 160]),
    ("release", "PortionHotDogs", "20250816", 48,
     "rc_grasp_remains_safe_until_release", [30, 52, 80]),
    ("crosscontam", "PackIdenticalLunches", "20250815", 27,
     "rc_raw_robot_contact_blocks_rte_grasp_until_sanitized", [60, 98, 140]),
    ("actiononset", "PrepareCoffee", "20250812", 38,
     "rc_pick_actiononsets_safe", [20, 40, 60]),
    ("mechanism", "HeatKebabSandwich", "20250813", 16,
     "rc_fixture_close_obstacle_retract_recovers", [70, 88, 105]),
    ("containment", "PanTransfer", "20250817", 22,
     "rc_solid_transfer_eventually_settles", [67, 71, 75]),
    ("enclosure", "StoreLeftoversInBowl", "20250813", 26,
     "rc_reach_in_fixture_only_when_fully_open", [65, 80, 95]),
    ("monitoring", "MakeIceLemonade", "20250813", 0,
     "rc_grasp_remains_synced_until_dropped", [166, 177, 240]),
]


def n_mon_for(task, ep, prop):
    mon = json.load(open(os.path.join(
        DEMO_RC, task, f"privileged_information_{ep}_monitor.json")))
    for key in ("recovery_accepting_by_property", "accepting_by_property"):
        tr = (mon.get(key) or {}).get(prop)
        if tr:
            return len(tr)
    # fall back to any property's trace length (all share the frame axis)
    for key in ("recovery_accepting_by_property", "accepting_by_property"):
        d = mon.get(key) or {}
        if d:
            return len(next(iter(d.values())))
    raise RuntimeError(f"no trace length for {task} ep{ep} {prop}")


def reset_to(env, state):
    """Minimal copy of robocasa playback_dataset.reset_to (states path)."""
    if "model" in state:
        if state.get("ep_meta") is not None:
            ep_meta = json.loads(state["ep_meta"])
            if hasattr(env, "set_ep_meta"):
                env.set_ep_meta(ep_meta)
            elif hasattr(env, "set_attrs_from_ep_meta"):
                env.set_attrs_from_ep_meta(ep_meta)
        env.reset()
        xml = env.edit_model_xml(state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()


def main():
    import robosuite
    import robocasa  # noqa: F401
    import robocasa.utils.lerobot_utils as LU

    os.makedirs(OUT, exist_ok=True)
    for key, task, date, ep, prop, frames in CELLS:
        dataset = os.path.join(DATA, task, date, "lerobot")
        extras = os.path.join(dataset, "extras", f"episode_{ep:06d}")
        states = np.load(os.path.join(extras, "states.npz"))["states"]
        with gzip.open(os.path.join(extras, "model.xml.gz"), "rt") as f:
            model_xml = f.read()
        ep_meta = open(os.path.join(extras, "ep_meta.json")).read()
        n_mon = n_mon_for(task, ep, prop)
        T = states.shape[0]

        env_meta = LU.get_env_metadata(dataset)
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs.update(has_renderer=False, renderer="mjviewer",
                          has_offscreen_renderer=True, use_camera_obs=False)
        print(f"[{key}] {task} ep{ep}: env init...", flush=True)
        env = robosuite.make(**env_kwargs)
        reset_to(env, {"model": model_xml, "ep_meta": ep_meta,
                       "states": states[0]})
        for f in frames:
            vf = min(T - 1, round(f / max(n_mon - 1, 1) * (T - 1)))
            reset_to(env, {"states": states[vf]})
            for cam in CAMS:
                im = env.sim.render(height=RES, width=RES,
                                    camera_name=cam)[::-1]
                name = f"{key}__{task}_ep{ep}_m{f}_v{vf}__{cam[-5:].strip('_')}.png"
                Image.fromarray(im).save(os.path.join(OUT, name))
            print(f"  m{f} -> video {vf} done", flush=True)
        env.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
