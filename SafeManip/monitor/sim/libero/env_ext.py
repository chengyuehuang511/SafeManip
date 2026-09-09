"""
`get_privileged_information` for LIBERO -- the LIBERO analog of
`monitor/sim/robocasa/kitchen_ext.py`. Monkeypatched onto LIBERO's common
problem-env base class (`BDDLBaseDomain`, in
`libero/libero/envs/bddl_base_domain.py`, imported from the vendored fresh
checkout at `/nethome/chuang475/testnvme/projects/SafeManip/libero`) so that
`env.get_privileged_information()` works the same way at every call site
`monitor/sim/robocasa/kitchen_ext.py`'s does for RoboCasa's `Kitchen` class.

Applied via monkeypatch (not a free function taking `env` explicitly) for the
same reason robocasa's own kitchen_ext.py chose that pattern: existing/future
call sites (this integration's own extraction script, and any future live
LIBERO-eval pipeline) can just call `env.get_privileged_information()`
unchanged. Import this module (or the package `monitor.sim.libero`, which
does so as its own side effect) once, before constructing any LIBERO env, for
the patch to take effect.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain

from .predicates import build_predicate_snapshot, build_predicate_static_spec


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _body_pose(env, body_id):
    if body_id is None:
        return None
    try:
        pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
        quat_wxyz = np.asarray(env.sim.data.body_xquat[body_id], dtype=float)
        return dict(position=_to_serializable(pos), orientation=_to_serializable(quat_wxyz))
    except Exception:
        return None


def get_privileged_information(self, trajectory_horizon: int = 64) -> Dict[str, Any]:
    if not hasattr(self, "_privileged_static_cache"):
        object_static = {}
        for name in getattr(self, "objects_dict", {}).keys():
            body_id = self.obj_body_id.get(name) if hasattr(self, "obj_body_id") else None
            object_static[name] = dict(body_id=body_id)
        fixture_static = {}
        for name in getattr(self, "fixtures_dict", {}).keys():
            body_id = self.obj_body_id.get(name) if hasattr(self, "obj_body_id") else None
            fixture_static[name] = dict(body_id=body_id)

        self._privileged_static_cache = dict(
            task=dict(
                env_name=self.__class__.__name__,
                language=_to_serializable(_safe_call(lambda: self.language_instruction) or ""),
                bddl_file_name=_to_serializable(getattr(self, "bddl_file_name", None)),
                goal_state=_to_serializable((getattr(self, "parsed_problem", {}) or {}).get("goal_state")),
                obj_of_interest=_to_serializable(getattr(self, "obj_of_interest", None)),
            ),
            robot=dict(
                robot_class=self.robots[0].__class__.__name__,
                naming_prefix=_safe_call(lambda: self.robots[0].robot_model.naming_prefix),
            ),
            scene_layout=dict(
                objects=object_static,
                fixtures=fixture_static,
            ),
        )
        self._privileged_static_cache["predicates"] = build_predicate_static_spec(
            self, self._privileged_static_cache
        )

    eef_pos = None
    try:
        gripper = self.robots[0].gripper
        eef_pos = np.asarray(
            self.sim.data.get_site_xpos(gripper.important_sites["grip_site"]), dtype=float
        )
    except Exception:
        pass

    scene_objects = {}
    for name in getattr(self, "objects_dict", {}).keys():
        body_id = self.obj_body_id.get(name) if hasattr(self, "obj_body_id") else None
        scene_objects[name] = dict(pose=_body_pose(self, body_id))

    scene_fixtures = {}
    for name in getattr(self, "fixtures_dict", {}).keys():
        body_id = self.obj_body_id.get(name) if hasattr(self, "obj_body_id") else None
        scene_fixtures[name] = dict(pose=_body_pose(self, body_id))

    dynamic = dict(
        task=dict(
            timestep=int(getattr(self, "timestep", -1)),
            success=bool(_safe_call(self._check_success) or False),
        ),
        robot=dict(
            end_effector_pose=dict(position=_to_serializable(eef_pos)) if eef_pos is not None else None,
        ),
        scene=dict(objects=scene_objects, fixtures=scene_fixtures),
    )

    dynamic["predicates"] = build_predicate_snapshot(
        self, self._privileged_static_cache, dynamic
    )

    return dict(static=self._privileged_static_cache, dynamic=dynamic)


# Monkeypatch application -- idempotent, harmless to re-import.
BDDLBaseDomain.get_privileged_information = get_privileged_information
