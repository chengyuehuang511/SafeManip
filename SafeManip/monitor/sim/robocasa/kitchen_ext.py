"""
`Kitchen.get_privileged_information` (robocasa/environments/kitchen/kitchen.py),
extracted out of robocasa's own source tree so the vendor package stays
pristine (matches upstream commit 9a3a786, before any SafeManip predicate
work) -- see replay/README.md-adjacent discussion in this repo's git history
for the reasoning. Verified by diffing kitchen.py against that base commit:
this method is a single, self-contained, 100%-additive block (1123 lines
added, exactly 1 now-redundant top-level import removed elsewhere in the
file) -- no existing robocasa method body was modified to add it, so it
lifts out cleanly with zero changes to its own logic.

Applied via monkeypatch (`Kitchen.get_privileged_information = ...` at the
bottom of this module) rather than converted to a free function taking `env`
explicitly, specifically so `env.get_privileged_information(...)` keeps
working unchanged at every existing call site (this repo's
extract_privileged_from_dataset*.py, and -- if this module is imported
early enough there too -- the live Isaac-GR00T eval pipeline, without
needing to touch that repo's call sites at all). Import this module once,
before constructing any Kitchen env, for the patch to take effect:

    import monitor.sim.robocasa.kitchen_ext  # noqa: F401  (side effect: patches Kitchen)

Sibling modules in this package (`predicates.py`, `attributes.py`) are the
other pieces of the same additive work, moved out of robocasa's tree the
same way -- see their own module docstrings/git history for what they add.
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array
from scipy.spatial.transform import Rotation

from robocasa.environments.kitchen.kitchen import Kitchen

from .predicates import build_predicate_snapshot, build_predicate_static_spec


def get_privileged_information(
    self, trajectory_horizon: int = 64
) -> Dict[str, Any]:
    """
    Return privileged scene metadata and current simulator state.

    The structure mirrors the mshab helper at a higher level:
    - ``static`` contains task / layout / object / fixture metadata that is
      mostly fixed for the episode.
    - ``dynamic`` contains the latest robot / object / fixture state.
    """

    def _to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [_to_serializable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _to_serializable(v) for k, v in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    def _safe_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    def _safe_attr(obj, attr, default=None):
        try:
            return getattr(obj, attr, default)
        except Exception:
            return default

    def _site_local_pos(site: Any) -> Any:
        if site is None:
            return None
        try:
            raw = site.get("pos")
        except Exception:
            raw = None
        if raw is None:
            return None
        try:
            return string_to_array(raw)
        except Exception:
            return raw

    def _body_id(name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        return _safe_call(self.sim.model.body_name2id, name)

    def _site_id(name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        return _safe_call(self.sim.model.site_name2id, name)

    def _joint_qpos(name: str):
        addr = _safe_call(self.sim.model.get_joint_qpos_addr, name)
        if addr is None:
            return None
        try:
            return _to_serializable(self.sim.data.qpos[addr])
        except Exception:
            return None

    def _joint_qvel(name: str):
        addr = _safe_call(self.sim.model.get_joint_qvel_addr, name)
        if addr is None:
            return None
        try:
            return _to_serializable(self.sim.data.qvel[addr])
        except Exception:
            return None

    def _pose_from_body_id(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if body_id is None:
            return None
        try:
            quat_xyzw = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
            return dict(
                position=_to_serializable(self.sim.data.body_xpos[body_id]),
                orientation=_to_serializable(quat_xyzw),
            )
        except Exception:
            return None

    def _pose_from_site_id(site_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if site_id is None:
            return None
        try:
            pos = self.sim.data.site_xpos[site_id]
            rot = self.sim.data.site_xmat[site_id].reshape(3, 3)
            quat_xyzw = Rotation.from_matrix(rot).as_quat()
            return dict(
                position=_to_serializable(pos),
                orientation=_to_serializable(quat_xyzw),
            )
        except Exception:
            return None

    def _velocity_from_site_id(site_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if site_id is None:
            return None
        try:
            return dict(
                linear=_to_serializable(self.sim.data.site_xvelp[site_id]),
                angular=_to_serializable(self.sim.data.site_xvelr[site_id]),
            )
        except Exception:
            return None

    def _end_effector_velocity(
        site_id: Optional[int],
        body_id: Optional[int],
        pose: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        site_velocity = _velocity_from_site_id(site_id)
        if site_velocity is not None:
            return site_velocity

        body_velocity = _velocity_from_body_id(body_id)
        if body_velocity is not None:
            return body_velocity

        if pose is None:
            return None
        try:
            prev_pose = getattr(self, "_privileged_prev_eef_pose", None)
            prev_time = getattr(self, "_privileged_prev_time", None)
            curr_time = float(getattr(self.sim.data, "time", np.nan))
            curr_pos = np.asarray(pose["position"], dtype=float)
            if (
                prev_pose is not None
                and prev_time is not None
                and np.isfinite(curr_time)
                and curr_time > float(prev_time)
            ):
                prev_pos = np.asarray(prev_pose["position"], dtype=float)
                linear = (curr_pos - prev_pos) / float(curr_time - float(prev_time))
                return dict(
                    linear=_to_serializable(linear),
                    angular=None,
                )
        except Exception:
            pass
        return None

    def _velocity_from_body_id(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if body_id is None:
            return None
        try:
            return dict(
                linear=_to_serializable(self.sim.data.body_xvelp[body_id]),
                angular=_to_serializable(self.sim.data.body_xvelr[body_id]),
            )
        except Exception:
            try:
                body_name = self.sim.model.body_id2name(body_id)
            except Exception:
                body_name = None
            if body_name:
                linear = _safe_call(self.sim.data.get_body_xvelp, body_name)
                angular = _safe_call(self.sim.data.get_body_xvelr, body_name)
                if linear is not None or angular is not None:
                    return dict(
                        linear=_to_serializable(linear),
                        angular=_to_serializable(angular),
                    )
            return None

    def _pose_from_camera_name(
        camera_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not camera_name:
            return None
        try:
            cam_id = int(self.sim.model.camera_name2id(camera_name))
            pos = np.asarray(self.sim.data.cam_xpos[cam_id], dtype=float)
            rot = np.asarray(self.sim.data.cam_xmat[cam_id], dtype=float).reshape(
                3, 3
            )
            quat_xyzw = Rotation.from_matrix(rot).as_quat()
            return dict(
                position=_to_serializable(pos),
                orientation=_to_serializable(quat_xyzw),
            )
        except Exception:
            return None

    def _camera_dynamic() -> Dict[str, Any]:
        result = {}
        for camera_name, camera_cfg in (
            getattr(self, "_cam_configs", {}) or {}
        ).items():
            if not isinstance(camera_cfg, dict):
                continue
            result[camera_name] = dict(
                pose=_pose_from_camera_name(camera_name),
                fovy=_safe_call(
                    lambda: float(
                        self.sim.model.cam_fovy[
                            int(self.sim.model.camera_name2id(camera_name))
                        ]
                    )
                ),
            )
        return result

    def _box_corners(half_extents: np.ndarray) -> np.ndarray:
        hx, hy, hz = np.asarray(half_extents, dtype=float).reshape(3)
        return np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [hx, hy, -hz],
                [-hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [hx, hy, hz],
                [-hx, hy, hz],
            ],
            dtype=float,
        )

    def _geom_local_corners(geom_id: int) -> Optional[np.ndarray]:
        try:
            size = np.asarray(
                self.sim.model.geom_size[geom_id], dtype=float
            ).reshape(-1)
        except Exception:
            size = np.empty((0,), dtype=float)
        try:
            geom_type = int(self.sim.model.geom_type[geom_id])
        except Exception:
            geom_type = -1
        local_corners = None
        if (
            size.size >= 3
            and np.all(np.isfinite(size[:3]))
            and np.any(np.abs(size[:3]) > 0.0)
        ):
            local_corners = _box_corners(np.abs(size[:3]))
        elif (
            size.size == 2
            and np.all(np.isfinite(size[:2]))
            and np.any(np.abs(size[:2]) > 0.0)
        ):
            local_corners = _box_corners(
                np.abs(np.array([size[0], size[0], size[1]], dtype=float))
            )
        elif size.size == 1 and np.isfinite(size[0]) and abs(size[0]) > 0.0:
            local_corners = _box_corners(np.full(3, abs(size[0]), dtype=float))

        # For mesh geoms, prefer the actual mesh vertices when available so the bounds
        # stay tight and don't inflate the parent/base link.
        if geom_type == 7:
            try:
                mesh_id = int(self.sim.model.geom_dataid[geom_id])
                vert_adr = int(self.sim.model.mesh_vertadr[mesh_id])
                vert_num = int(self.sim.model.mesh_vertnum[mesh_id])
                mesh_verts = np.asarray(
                    self.sim.model.mesh_vert[vert_adr : vert_adr + vert_num],
                    dtype=float,
                )
                if mesh_verts.size and size.size >= 3:
                    local_corners = mesh_verts * np.abs(size[:3])[None, :]
            except Exception:
                pass

        if local_corners is None or local_corners.size == 0:
            return None

        try:
            geom_pos = np.asarray(self.sim.model.geom_pos[geom_id], dtype=float)
        except Exception:
            geom_pos = np.zeros(3, dtype=float)
        try:
            geom_quat = np.asarray(self.sim.model.geom_quat[geom_id], dtype=float)
            geom_rot = Rotation.from_quat(
                T.convert_quat(geom_quat, to="xyzw")
            ).as_matrix()
        except Exception:
            geom_rot = np.eye(3, dtype=float)
        return local_corners @ geom_rot.T + geom_pos[None, :]

    def _body_local_bounds(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if body_id is None:
            return None
        try:
            geom_start = int(self.sim.model.body_geomadr[body_id])
            geom_count = int(self.sim.model.body_geomnum[body_id])
        except Exception:
            return None
        if geom_count <= 0:
            return None
        mins = []
        maxs = []
        for geom_id in range(geom_start, geom_start + geom_count):
            geom_corners = _geom_local_corners(geom_id)
            if geom_corners is None:
                continue
            mins.append(np.min(geom_corners, axis=0))
            maxs.append(np.max(geom_corners, axis=0))
        if not mins:
            return None
        bmin = np.min(np.stack(mins, axis=0), axis=0)
        bmax = np.max(np.stack(maxs, axis=0), axis=0)
        return dict(min=_to_serializable(bmin), max=_to_serializable(bmax))

    def _body_immediate_child_connector_bounds(
        body_id: Optional[int],
        *,
        min_half_extents: Iterable[float] = (0.02, 0.02, 0.02),
        pad: float = 0.01,
    ) -> Optional[Dict[str, Any]]:
        if body_id is None:
            return None
        try:
            parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(
                -1
            )
            body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
            body_rot = np.asarray(
                self.sim.data.body_xmat[body_id], dtype=float
            ).reshape(3, 3)
        except Exception:
            return None

        child_ids = [
            int(idx)
            for idx, parent in enumerate(parents)
            if int(parent) == int(body_id) and int(idx) != int(body_id)
        ]
        if not child_ids:
            return None

        try:
            min_half = np.asarray(list(min_half_extents), dtype=float).reshape(3)
        except Exception:
            min_half = np.full(3, 0.02, dtype=float)

        local_points = [np.zeros(3, dtype=float)]
        body_rot_T = body_rot.T
        for child_id in child_ids:
            try:
                child_pos = np.asarray(
                    self.sim.data.body_xpos[child_id], dtype=float
                )
            except Exception:
                continue
            child_local = (child_pos - body_pos) @ body_rot_T
            local_points.append(child_local)

        if len(local_points) <= 1:
            return None

        stacked = np.stack(local_points, axis=0)
        bmin = np.min(stacked, axis=0) - float(pad)
        bmax = np.max(stacked, axis=0) + float(pad)
        center = 0.5 * (bmin + bmax)
        half = np.maximum(0.5 * (bmax - bmin), min_half)
        return dict(
            min=_to_serializable(center - half),
            max=_to_serializable(center + half),
        )

    def _body_vertical_connector_bounds(
        body_id: Optional[int],
        *,
        target_child_names: Iterable[str],
        min_half_extents: Iterable[float] = (0.03, 0.03, 0.03),
        lateral_half_extents: Iterable[float] = (0.08, 0.08),
        pad_z: float = 0.02,
    ) -> Optional[Dict[str, Any]]:
        if body_id is None:
            return None
        try:
            parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(
                -1
            )
            body_name = self.sim.model.body_id2name(body_id)
            body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
            body_rot = np.asarray(
                self.sim.data.body_xmat[body_id], dtype=float
            ).reshape(3, 3)
        except Exception:
            return None
        target_child_names = set(target_child_names)
        child_points = []
        body_rot_T = body_rot.T
        for child_id, parent in enumerate(parents):
            if int(parent) != int(body_id) or int(child_id) == int(body_id):
                continue
            try:
                child_name = self.sim.model.body_id2name(int(child_id))
            except Exception:
                child_name = None
            if child_name not in target_child_names:
                continue
            try:
                child_pos = np.asarray(
                    self.sim.data.body_xpos[int(child_id)], dtype=float
                )
            except Exception:
                continue
            child_points.append((child_pos - body_pos) @ body_rot_T)
        if not child_points:
            return None
        try:
            min_half = np.asarray(list(min_half_extents), dtype=float).reshape(3)
            lateral_half = np.asarray(
                list(lateral_half_extents), dtype=float
            ).reshape(2)
        except Exception:
            min_half = np.full(3, 0.03, dtype=float)
            lateral_half = np.full(2, 0.08, dtype=float)
        stacked = np.stack([np.zeros(3, dtype=float)] + child_points, axis=0)
        bmin = np.min(stacked, axis=0)
        bmax = np.max(stacked, axis=0)
        center_xy = 0.5 * (bmin[:2] + bmax[:2])
        half_xy = np.maximum(0.5 * (bmax[:2] - bmin[:2]), lateral_half)
        center_z = 0.5 * (bmin[2] + bmax[2])
        half_z = max(0.5 * (bmax[2] - bmin[2]) + float(pad_z), float(min_half[2]))
        center = np.array([center_xy[0], center_xy[1], center_z], dtype=float)
        half = np.array([half_xy[0], half_xy[1], half_z], dtype=float)
        return dict(
            min=_to_serializable(center - half),
            max=_to_serializable(center + half),
        )

    def _robot_subtree_body_ids(root_body_id: Optional[int]) -> List[int]:
        if root_body_id is None:
            return []
        try:
            parents = np.asarray(self.sim.model.body_parentid, dtype=int).reshape(
                -1
            )
        except Exception:
            return [int(root_body_id)]
        body_ids: List[int] = []
        for body_id in range(len(parents)):
            cur = body_id
            while cur >= 0:
                if cur == root_body_id:
                    body_ids.append(int(body_id))
                    break
                parent = int(parents[cur])
                if parent == cur:
                    break
                cur = parent
        return body_ids

    def _body_subtree_local_bounds(
        body_id: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if body_id is None:
            return None
        try:
            root_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
            root_rot = np.asarray(
                self.sim.data.body_xmat[body_id], dtype=float
            ).reshape(3, 3)
        except Exception:
            return None
        mins = []
        maxs = []
        root_rot_T = root_rot.T
        for sub_body_id in _robot_subtree_body_ids(body_id):
            try:
                geom_start = int(self.sim.model.body_geomadr[sub_body_id])
                geom_count = int(self.sim.model.body_geomnum[sub_body_id])
            except Exception:
                continue
            for geom_id in range(geom_start, geom_start + geom_count):
                half_extents = _geom_half_extents(geom_id)
                if half_extents is None:
                    continue
                try:
                    geom_pos = np.asarray(
                        self.sim.data.geom_xpos[geom_id], dtype=float
                    )
                    geom_rot = np.asarray(
                        self.sim.data.geom_xmat[geom_id], dtype=float
                    ).reshape(3, 3)
                except Exception:
                    continue
                local_corners = np.array(
                    [
                        [-half_extents[0], -half_extents[1], -half_extents[2]],
                        [half_extents[0], -half_extents[1], -half_extents[2]],
                        [half_extents[0], half_extents[1], -half_extents[2]],
                        [-half_extents[0], half_extents[1], -half_extents[2]],
                        [-half_extents[0], -half_extents[1], half_extents[2]],
                        [half_extents[0], -half_extents[1], half_extents[2]],
                        [half_extents[0], half_extents[1], half_extents[2]],
                        [-half_extents[0], half_extents[1], half_extents[2]],
                    ],
                    dtype=float,
                )
                world_corners = local_corners @ geom_rot.T + geom_pos[None, :]
                root_local_corners = (
                    world_corners - root_pos[None, :]
                ) @ root_rot_T
                mins.append(np.min(root_local_corners, axis=0))
                maxs.append(np.max(root_local_corners, axis=0))
        if not mins:
            return None
        bmin = np.min(np.stack(mins, axis=0), axis=0)
        bmax = np.max(np.stack(maxs, axis=0), axis=0)
        return dict(min=_to_serializable(bmin), max=_to_serializable(bmax))

    def _site_local_box_bounds(
        body_id: Optional[int],
        site_id: Optional[int],
        half_extents: Iterable[float],
    ) -> Optional[Dict[str, Any]]:
        if body_id is None or site_id is None:
            return None
        try:
            body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
            body_rot = np.asarray(
                self.sim.data.body_xmat[body_id], dtype=float
            ).reshape(3, 3)
            site_pos = np.asarray(self.sim.data.site_xpos[site_id], dtype=float)
            half = np.asarray(list(half_extents), dtype=float).reshape(3)
        except Exception:
            return None
        corners = np.array(
            [
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ],
            dtype=float,
        )
        world_corners = corners + site_pos[None, :]
        local_corners = (world_corners - body_pos[None, :]) @ body_rot.T
        return dict(
            min=_to_serializable(np.min(local_corners, axis=0)),
            max=_to_serializable(np.max(local_corners, axis=0)),
        )

    def _body_external_wrench(body_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if body_id is None:
            return None
        try:
            wrench = np.asarray(
                self.sim.data.cfrc_ext[body_id], dtype=float
            ).reshape(-1)
        except Exception:
            return None
        if wrench.size < 6 or not np.all(np.isfinite(wrench[:6])):
            return None
        force = wrench[:3]
        torque = wrench[3:6]
        return dict(
            force=_to_serializable(force),
            torque=_to_serializable(torque),
            force_norm=float(np.linalg.norm(force)),
            torque_norm=float(np.linalg.norm(torque)),
        )

    def _robot_joint_bundle() -> Dict[str, Any]:
        joint_names: List[str] = []
        seen = set()

        def _append_joint(name: Optional[str]) -> None:
            if not name or name in seen:
                return
            seen.add(name)
            joint_names.append(name)

        for name in _safe_attr(robot, "robot_joints", []) or []:
            _append_joint(name)

        gripper_obj = _safe_attr(robot, "gripper")
        if isinstance(gripper_obj, dict):
            grippers = list(gripper_obj.values())
        elif gripper_obj is None:
            grippers = []
        else:
            grippers = [gripper_obj]
        for gripper in grippers:
            for attr_name in ("joints", "joint_names"):
                for name in _safe_attr(gripper, attr_name, []) or []:
                    _append_joint(name)

        return dict(
            joint_names=_to_serializable(joint_names),
            joint_positions=_to_serializable(
                [_joint_qpos(name) for name in joint_names]
            ),
            joint_velocities=_to_serializable(
                [_joint_qvel(name) for name in joint_names]
            ),
        )

    def _physics_dynamic(
        robot_link_meta: Dict[str, Any], eef_body_name: Optional[str]
    ) -> Dict[str, Any]:
        robot_body_ids = {
            int(info.get("body_id"))
            for info in robot_link_meta.values()
            if isinstance(info, dict) and info.get("body_id") is not None
        }
        geom_bodyid = None
        try:
            geom_bodyid = np.asarray(self.sim.model.geom_bodyid, dtype=int).reshape(
                -1
            )
        except Exception:
            geom_bodyid = None

        total_contact_count = int(_safe_attr(self.sim.data, "ncon", 0) or 0)
        robot_contact_count = 0
        for contact_idx in range(total_contact_count):
            try:
                contact = self.sim.data.contact[contact_idx]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
            except Exception:
                continue
            body1 = (
                int(geom_bodyid[geom1])
                if geom_bodyid is not None and 0 <= geom1 < len(geom_bodyid)
                else None
            )
            body2 = (
                int(geom_bodyid[geom2])
                if geom_bodyid is not None and 0 <= geom2 < len(geom_bodyid)
                else None
            )
            if body1 in robot_body_ids or body2 in robot_body_ids:
                robot_contact_count += 1

        per_link_force_norms = {}
        max_robot_link_force = np.nan
        robot_force_sum = 0.0
        end_effector_force_norm = np.nan
        for name, info in robot_link_meta.items():
            body_id = info.get("body_id") if isinstance(info, dict) else None
            wrench = _body_external_wrench(body_id)
            if wrench is None:
                continue
            force_norm = float(wrench["force_norm"])
            per_link_force_norms[name] = force_norm
            robot_force_sum += force_norm
            if (
                not np.isfinite(max_robot_link_force)
                or force_norm > max_robot_link_force
            ):
                max_robot_link_force = force_norm
            if eef_body_name and name == eef_body_name:
                end_effector_force_norm = force_norm

        return dict(
            total_contact_count=total_contact_count,
            robot_contact_count=robot_contact_count,
            max_robot_link_force=max_robot_link_force,
            robot_force_sum=robot_force_sum,
            end_effector_force_norm=end_effector_force_norm,
            per_link_force_norms=_to_serializable(per_link_force_norms),
        )

    def _robot_link_static(
        body_id: int, *, fallback_bounds: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            body_name = self.sim.model.body_id2name(body_id)
        except Exception:
            body_name = None
        if not body_name:
            return None
        direct_bounds = _body_local_bounds(body_id)
        structural_connector_bounds = None
        if direct_bounds is None and fallback_bounds is None:
            if body_name == "mobilebase0_base":
                structural_connector_bounds = _body_vertical_connector_bounds(
                    body_id,
                    target_child_names=("mobilebase0_support",),
                    lateral_half_extents=(0.11, 0.11),
                    pad_z=0.03,
                )
            elif body_name == "robot0_base":
                structural_connector_bounds = _body_vertical_connector_bounds(
                    body_id,
                    target_child_names=("robot0_link0",),
                    lateral_half_extents=(0.06, 0.06),
                    pad_z=0.02,
                )
        connector_bounds = (
            _body_immediate_child_connector_bounds(body_id)
            if direct_bounds is None
            and fallback_bounds is None
            and structural_connector_bounds is None
            else None
        )
        visual_bounds = (
            direct_bounds
            or fallback_bounds
            or structural_connector_bounds
            or connector_bounds
        )
        if direct_bounds is not None:
            visual_bounds_source = "direct_geom"
        elif fallback_bounds is not None:
            visual_bounds_source = "eef_site_fallback"
        elif structural_connector_bounds is not None:
            visual_bounds_source = "structural_connector"
        elif connector_bounds is not None:
            visual_bounds_source = "immediate_child_connector"
        else:
            visual_bounds_source = None
        return dict(
            name=body_name,
            body_id=int(body_id),
            visual_bounds=visual_bounds,
            visual_bounds_source=visual_bounds_source,
        )

    def _relative_pose_to_eef(
        target_pose: Optional[Dict[str, Any]], eef_pose: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if target_pose is None or eef_pose is None:
            return None
        try:
            target_mat = T.pose2mat(
                (
                    np.asarray(target_pose["position"]),
                    np.asarray(target_pose["orientation"]),
                )
            )
            eef_inv = T.pose_inv(
                T.pose2mat(
                    (
                        np.asarray(eef_pose["position"]),
                        np.asarray(eef_pose["orientation"]),
                    )
                )
            )
            rel = T.pose_in_A_to_pose_in_B(target_mat, eef_inv)
            rel_pos, rel_quat = T.mat2pose(rel)
            return dict(
                position=_to_serializable(rel_pos),
                orientation=_to_serializable(rel_quat),
            )
        except Exception:
            return None

    def _fixture_joint_state(fixture) -> Optional[Dict[str, Any]]:
        joints = _safe_attr(fixture, "_joints", []) or []
        if not joints:
            return None
        result = {}
        for joint_suffix in joints:
            joint_name = f"{fixture.name}_{joint_suffix}"
            result[joint_name] = dict(
                qpos=_joint_qpos(joint_name),
                qvel=_joint_qvel(joint_name),
            )
        return result or None

    def _fixture_static(fixture) -> Dict[str, Any]:
        local_bounds = {}
        for key, site in (_safe_attr(fixture, "_bounds_sites", {}) or {}).items():
            local_bounds[key] = _to_serializable(_site_local_pos(site))
        return dict(
            class_name=fixture.__class__.__name__,
            nat_lang=_safe_attr(fixture, "nat_lang"),
            size=_to_serializable(_safe_attr(fixture, "size")),
            position=_to_serializable(_safe_attr(fixture, "pos")),
            quaternion=_to_serializable(_safe_attr(fixture, "quat")),
            euler=_to_serializable(_safe_attr(fixture, "euler")),
            origin_offset=_to_serializable(_safe_attr(fixture, "origin_offset")),
            placement=_to_serializable(_safe_attr(fixture, "_placement")),
            local_bounds_sites=local_bounds or None,
        )

    def _fixture_dynamic(fixture) -> Dict[str, Any]:
        get_state_fn = _safe_attr(fixture, "get_state")
        get_door_state_fn = _safe_attr(fixture, "get_door_state")
        get_site_info_fn = _safe_attr(fixture, "get_site_info")

        state = _safe_call(get_state_fn, self) if callable(get_state_fn) else None
        if state is None:
            state = (
                _safe_call(get_state_fn, self.sim)
                if callable(get_state_fn)
                else None
            )
        if state is None:
            state = _safe_call(get_state_fn) if callable(get_state_fn) else None
        door_state = (
            _safe_call(get_door_state_fn, self)
            if callable(get_door_state_fn)
            else None
        )
        site_info = (
            _safe_call(get_site_info_fn, self.sim)
            if callable(get_site_info_fn)
            else None
        )
        return dict(
            state=_to_serializable(state),
            door_state=_to_serializable(door_state),
            joints=_fixture_joint_state(fixture),
            sites=_to_serializable(site_info),
        )

    def _predicate_static_spec(static_info: Dict[str, Any]) -> Dict[str, Any]:
        return build_predicate_static_spec(self, static_info)

    def _predicate_snapshot(
        static_info: Dict[str, Any], dynamic_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        return build_predicate_snapshot(self, static_info, dynamic_info)

    def _object_static(
        name: str, model, cfg: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        bbox_points = _safe_call(model.get_bbox_points)
        bbox_min = None
        bbox_max = None
        object_info = cfg.get("info") if isinstance(cfg, dict) else None
        if bbox_points is not None:
            try:
                bbox_points = np.asarray(bbox_points, dtype=float)
                bbox_min = bbox_points.min(axis=0)
                bbox_max = bbox_points.max(axis=0)
            except Exception:
                bbox_min = None
                bbox_max = None
        return dict(
            root_body=_safe_attr(model, "root_body"),
            info=_to_serializable(object_info),
            category=(
                object_info.get("cat") if isinstance(object_info, dict) else None
            ),
            config=_to_serializable(cfg),
            mjcf_path=_safe_attr(
                model, "asset_path", _safe_attr(model, "mjcf_path")
            ),
            scale=_to_serializable(_safe_attr(model, "scale")),
            horizontal_radius=_to_serializable(
                _safe_attr(model, "horizontal_radius")
            ),
            bbox=dict(
                min=_to_serializable(bbox_min),
                max=_to_serializable(bbox_max),
            )
            if bbox_min is not None
            else None,
        )

    robot = self.robots[0]
    eef_site_name = None
    try:
        eef_site_id = robot.eef_site_id["right"]
    except Exception:
        eef_site_id = None
    if eef_site_id is not None:
        try:
            eef_site_name = self.sim.model.site_id2name(eef_site_id)
        except Exception:
            eef_site_name = None

    robot_root_body = _safe_attr(robot.robot_model, "root_body")
    robot_root_body_id = _body_id(robot_root_body)

    robot_subtree_body_ids = _robot_subtree_body_ids(robot_root_body_id)

    def _choose_robot_base_body(
        default_name: Optional[str],
        default_body_id: Optional[int],
        subtree_body_ids: List[int],
    ) -> Tuple[Optional[str], Optional[int], Optional[Dict[str, Any]]]:
        default_pose = _pose_from_body_id(default_body_id)
        default_pos = (
            np.asarray(default_pose["position"], dtype=float)
            if default_pose is not None and default_pose.get("position") is not None
            else None
        )

        def _is_far_dummy(pos: Optional[np.ndarray]) -> bool:
            if pos is None or pos.shape[0] < 2:
                return False
            return bool(np.linalg.norm(pos[:2]) > 5.0)

        if not _is_far_dummy(default_pos):
            return default_name, default_body_id, default_pose

        candidates = []
        for body_id in subtree_body_ids:
            body_name = _safe_call(self.sim.model.body_id2name, int(body_id))
            pose = _pose_from_body_id(body_id)
            pos = (
                np.asarray(pose["position"], dtype=float)
                if pose is not None and pose.get("position") is not None
                else None
            )
            if pos is None or _is_far_dummy(pos):
                continue
            lowered = str(body_name or "").lower()
            priority = 100
            if "mobilebase" in lowered and lowered.endswith("base"):
                priority = 0
            elif "mobilebase" in lowered:
                priority = 10
            elif "support" in lowered:
                priority = 20
            elif "base" in lowered:
                priority = 30
            candidates.append((priority, str(body_name), int(body_id), pose))

        if candidates:
            _, body_name, body_id, pose = sorted(candidates)[0]
            return body_name, body_id, pose

        return default_name, default_body_id, default_pose

    (
        robot_root_body,
        robot_root_body_id,
        robot_root_pose,
    ) = _choose_robot_base_body(
        robot_root_body,
        robot_root_body_id,
        robot_subtree_body_ids,
    )
    eef_body_name = None
    eef_body_id = None
    if eef_site_id is not None:
        try:
            eef_body_id = int(self.sim.model.site_bodyid[eef_site_id])
            eef_body_name = self.sim.model.body_id2name(eef_body_id)
        except Exception:
            eef_body_name = None
            eef_body_id = None
    eef_fallback_bounds = _site_local_box_bounds(
        eef_body_id,
        eef_site_id,
        half_extents=(0.035, 0.025, 0.025),
    )
    robot_link_meta = {}
    for body_id in robot_subtree_body_ids:
        link_info = _robot_link_static(
            body_id,
            fallback_bounds=(
                eef_fallback_bounds
                if eef_body_id is not None and int(body_id) == int(eef_body_id)
                else None
            ),
        )
        if link_info is None:
            continue
        robot_link_meta[link_info["name"]] = link_info

    if not hasattr(self, "_privileged_static_cache"):
        ep_meta = self.get_ep_meta()
        object_cfg_by_name = {
            cfg["name"]: cfg
            for cfg in getattr(self, "object_cfgs", [])
            if "name" in cfg
        }
        self._privileged_static_cache = dict(
            task=dict(
                env_name=self.__class__.__name__,
                language=_to_serializable(ep_meta.get("lang", "")),
                episode_meta=_to_serializable(ep_meta),
            ),
            robot=dict(
                robot_class=self.robots[0].__class__.__name__,
                robot_model_class=self.robots[0].robot_model.__class__.__name__,
                naming_prefix=_safe_attr(
                    self.robots[0].robot_model, "naming_prefix"
                ),
                init_base_pos=_to_serializable(
                    _safe_attr(self.robots[0].robot_model, "base_xpos")
                    or (robot_root_pose or {}).get("position")
                ),
                link_metadata=_to_serializable(robot_link_meta),
                base_link=(
                    dict(
                        name=robot_root_body,
                        info=_to_serializable(robot_link_meta.get(robot_root_body)),
                    )
                    if robot_root_body
                    else None
                ),
                end_effector_link=(
                    dict(
                        name=eef_body_name,
                        info=_to_serializable(robot_link_meta.get(eef_body_name)),
                    )
                    if eef_body_name
                    else None
                ),
            ),
            scene_layout=dict(
                layout_id=int(self.layout_id),
                style_id=int(self.style_id),
                fixture_refs=_to_serializable(
                    {
                        k: (v[0] if isinstance(v, tuple) else v).name
                        for k, v in getattr(self, "fixture_refs", {}).items()
                    }
                ),
                cameras=_to_serializable(getattr(self, "_cam_configs", {})),
                fixtures={
                    name: _fixture_static(fixture)
                    for name, fixture in getattr(self, "fixtures", {}).items()
                },
                objects={
                    name: _object_static(
                        name,
                        model,
                        object_cfg_by_name.get(name),
                    )
                    for name, model in getattr(self, "objects", {}).items()
                },
            ),
        )
        self._privileged_static_cache["predicates"] = _predicate_static_spec(
            self._privileged_static_cache
        )

    if not hasattr(self, "_privileged_history"):
        self._privileged_history = dict(
            eef_positions=[], robot_force_cumulative=0.0
        )
    eef_pose = _pose_from_site_id(eef_site_id)
    eef_velocity = _end_effector_velocity(eef_site_id, eef_body_id, eef_pose)
    if eef_pose is not None:
        self._privileged_history["eef_positions"].append(eef_pose["position"])
        if len(self._privileged_history["eef_positions"]) > trajectory_horizon:
            self._privileged_history["eef_positions"].pop(0)
    robot_joint_bundle = _robot_joint_bundle()
    physics = _physics_dynamic(robot_link_meta, eef_body_name)
    if np.isfinite(physics.get("robot_force_sum", np.nan)):
        self._privileged_history["robot_force_cumulative"] += float(
            physics["robot_force_sum"]
        )

    dynamic = dict(
        task=dict(
            timestep=int(getattr(self, "timestep", -1)),
            success=bool(_safe_call(self._check_success)),
        ),
        robot=dict(
            root_body=robot_root_body,
            root_pose=_pose_from_body_id(robot_root_body_id),
            root_velocity=_velocity_from_body_id(robot_root_body_id),
            eef_site=eef_site_name,
            end_effector_pose=eef_pose,
            end_effector_velocity=eef_velocity,
            link_poses={
                name: _pose_from_body_id(info.get("body_id"))
                for name, info in robot_link_meta.items()
            },
            joint_names=robot_joint_bundle["joint_names"],
            joint_positions=robot_joint_bundle["joint_positions"],
            joint_velocities=robot_joint_bundle["joint_velocities"],
            gripper_action_dim=int(len(getattr(robot, "gripper", {}))),
            trajectory=_to_serializable(self._privileged_history["eef_positions"]),
        ),
        scene=dict(
            objects={},
            fixtures={},
            cameras=_camera_dynamic(),
        ),
        physics=dict(
            **physics,
            robot_force_cumulative=float(
                self._privileged_history["robot_force_cumulative"]
            ),
        ),
    )

    for name, model in getattr(self, "objects", {}).items():
        body_id = _body_id(_safe_attr(model, "root_body"))
        pose = _pose_from_body_id(body_id)
        dynamic["scene"]["objects"][name] = dict(
            pose=pose,
            velocity=_velocity_from_body_id(body_id),
            relative_to_eef=_relative_pose_to_eef(pose, eef_pose),
        )

    for name, fixture in getattr(self, "fixtures", {}).items():
        dynamic["scene"]["fixtures"][name] = _fixture_dynamic(fixture)

    # Evaluate the task-agnostic low-level monitor after the base privileged
    # state has been assembled so downstream symbolic logic can reuse the
    # same serialized snapshot without re-deriving basic predicates elsewhere.
    dynamic["predicates"] = _predicate_snapshot(
        self._privileged_static_cache,
        dynamic,
    )

    self._privileged_prev_eef_pose = eef_pose
    try:
        self._privileged_prev_time = float(getattr(self.sim.data, "time", np.nan))
    except Exception:
        self._privileged_prev_time = None

    return dict(
        static=self._privileged_static_cache,
        dynamic=dynamic,
    )


# Monkeypatch application -- see module docstring for why this approach was
# chosen. Idempotent: re-importing this module just reassigns the same
# function again, harmless.
Kitchen.get_privileged_information = get_privileged_information
