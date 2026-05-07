#!/usr/bin/env python3
"""
Visualize RoboCasa privileged-information rollouts as a 3D reconstruction video.

Example usage:
    python scripts/3Dextraction.py \
        --input logs/eval/rollout_data/PnPCounterToCab--<DATE_TIME>/privileged_information_0.json \
        --output /tmp/robocasa_scene.mp4

    python scripts/3Dextraction.py \
        --input /path/to/privileged_information_7.json \
        --output /path/to/robocasa_scene.gif \
        --fps 12 \
        --stride 2

    python scripts/3Dextraction.py \
        --input logs/eval/rollout_data/TurnOffMicrowave--2026_03_20-19_31_17/privileged_information_0.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BOX_FACES = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [1, 2, 6, 5],
        [0, 3, 7, 4],
    ],
    dtype=int,
)

BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)

DEFAULT_EEF_BOX_SIZE = (0.14, 0.08, 0.08)


FIXTURE_GROUP_PALETTES: Dict[str, List[Tuple[float, float, float]]] = {
    "structure": [
        (0.64, 0.78, 0.96),
        (0.78, 0.72, 0.96),
        (0.70, 0.86, 0.88),
    ],
    "storage": [
        (0.98, 0.68, 0.36),
        (0.96, 0.56, 0.42),
        (0.92, 0.76, 0.42),
    ],
    "appliance": [
        (0.34, 0.78, 0.86),
        (0.22, 0.64, 0.92),
        (0.46, 0.72, 0.98),
    ],
    "other_fixture": [
        (0.74, 0.78, 0.82),
        (0.82, 0.74, 0.78),
        (0.74, 0.82, 0.76),
    ],
}

OBJECT_GROUP_PALETTES: Dict[str, List[Tuple[float, float, float]]] = {
    "container": [
        (1.00, 0.44, 0.18),
        (1.00, 0.64, 0.12),
        (0.95, 0.22, 0.38),
    ],
    "food": [
        (0.98, 0.83, 0.14),
        (0.62, 0.88, 0.10),
        (0.20, 0.82, 0.36),
    ],
    "tool": [
        (0.02, 0.76, 0.92),
        (0.18, 0.50, 1.00),
        (0.56, 0.34, 0.98),
    ],
    "other": [
        (0.96, 0.27, 0.21),
        (0.90, 0.26, 0.73),
        (0.00, 0.62, 0.54),
        (0.84, 0.20, 0.42),
    ],
}

ROBOT_BASE_COLOR = (0.72, 0.96, 0.18)
ROBOT_BASE_EDGE = (0.48, 0.74, 0.06)
ROBOT_EEF_COLOR = (1.00, 0.12, 0.78)
ROBOT_EEF_EDGE = (0.76, 0.06, 0.56)
ROBOT_LINK_COLOR = (0.92, 0.92, 0.96)
ROBOT_LINK_EDGE = (0.58, 0.58, 0.66)


def clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def mix_color(
    c1: Tuple[float, float, float],
    c2: Tuple[float, float, float],
    weight: float,
) -> Tuple[float, float, float]:
    w = clamp01(weight)
    return tuple((1.0 - w) * a + w * b for a, b in zip(c1, c2))


def fixture_group(label: str) -> str:
    low = str(label).strip().lower()
    structure_terms = ("wall", "floor", "window", "door", "panel", "backing", "trim", "outlet")
    storage_terms = ("cabinet", "drawer", "counter", "shelf", "table", "island")
    appliance_terms = ("microwave", "stove", "sink", "faucet", "fridge", "oven", "dishwasher")
    if any(term in low for term in structure_terms):
        return "structure"
    if any(term in low for term in storage_terms):
        return "storage"
    if any(term in low for term in appliance_terms):
        return "appliance"
    return "other_fixture"


def object_group(label: str) -> str:
    low = str(label).strip().lower()
    if any(term in low for term in ("cup", "mug", "jug", "bottle", "can", "bowl", "plate")):
        return "container"
    if any(term in low for term in ("apple", "banana", "fruit", "food", "bread", "vegetable")):
        return "food"
    if any(term in low for term in ("pan", "pot", "knife", "spoon", "fork", "tool", "utensil")):
        return "tool"
    return "other"


def grouped_color(
    label: str,
    variant_idx: int,
    variant_count: int,
    *,
    is_object: bool = False,
) -> Tuple[float, float, float]:
    if is_object:
        base = (0.90, 0.47, 0.22)
        span = np.linspace(-0.24, 0.24, max(variant_count, 1))
    else:
        palette = FIXTURE_GROUP_PALETTES.get(
            fixture_group(label), FIXTURE_GROUP_PALETTES["other_fixture"]
        )
        base = palette[variant_idx % len(palette)]
        span = np.linspace(-0.08, 0.12, max(variant_count, 1))
    if variant_count <= 1:
        return base
    delta = float(span[min(max(variant_idx, 0), variant_count - 1)])
    if delta >= 0.0:
        return mix_color(base, (1.0, 1.0, 1.0), delta)
    return mix_color(base, (0.0, 0.0, 0.0), -delta)


def vivid_object_color(
    label: str, palette_idx: int, palette_count: int
) -> Tuple[float, float, float]:
    group = object_group(label)
    palette = OBJECT_GROUP_PALETTES.get(group, OBJECT_GROUP_PALETTES["other"])
    base = palette[palette_idx % len(palette)]
    if palette_count <= 1:
        return base
    span = np.linspace(-0.08, 0.12, palette_count)
    delta = float(span[min(max(palette_idx // max(len(palette), 1), 0), palette_count - 1)])
    if delta >= 0.0:
        return mix_color(base, (1.0, 1.0, 1.0), delta)
    return mix_color(base, (0.0, 0.0, 0.0), -delta)


def as_xyz(v: Iterable[float]) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3:
        return None
    xyz = arr[:3]
    if not np.all(np.isfinite(xyz)):
        return None
    return xyz


def as_quat_xyzw(v: Iterable[float]) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 4:
        return None
    quat = arr[:4]
    if not np.all(np.isfinite(quat)):
        return None
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return None
    return quat / norm


def euler_xyz_to_matrix(euler_xyz: Iterable[float]) -> Optional[np.ndarray]:
    xyz = as_xyz(euler_xyz)
    if xyz is None:
        return None
    x, y, z = xyz
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def compose_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rotation
    mat[:3, 3] = position
    return mat


def pose_to_transform(
    position: Iterable[float], orientation: Optional[Iterable[float]] = None
) -> Optional[np.ndarray]:
    pos = as_xyz(position)
    if pos is None:
        return None
    quat = as_quat_xyzw(orientation) if orientation is not None else None
    if quat is not None:
        rot = quat_xyzw_to_matrix(quat)
    else:
        rot = np.eye(3, dtype=float)
    return compose_transform(pos, rot)


def transform_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    rot = np.asarray(mat[:3, :3], dtype=float)
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    return quat / norm if norm > 1e-8 else np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)


def fixture_transform(fixture_info: dict) -> Optional[np.ndarray]:
    pos = as_xyz(fixture_info.get("position"))
    if pos is None:
        return None
    quat = as_quat_xyzw(fixture_info.get("quaternion"))
    if quat is not None:
        rot = quat_xyzw_to_matrix(quat)
    else:
        rot = euler_xyz_to_matrix(fixture_info.get("euler"))
        if rot is None:
            rot = np.eye(3, dtype=float)
    return compose_transform(pos, rot)


def apply_transform(points: np.ndarray, mat: np.ndarray) -> np.ndarray:
    hom = np.ones((points.shape[0], 4), dtype=float)
    hom[:, :3] = points
    return (hom @ mat.T)[:, :3]


def box_corners_from_extents(size_xyz: Iterable[float]) -> Optional[np.ndarray]:
    size = as_xyz(size_xyz)
    if size is None:
        return None
    half = size / 2.0
    x0, y0, z0 = -half
    x1, y1, z1 = half
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def box_corners_from_bounds(bounds_entry: Any) -> Optional[np.ndarray]:
    if bounds_entry is None:
        return None
    if isinstance(bounds_entry, dict):
        bmin = as_xyz(bounds_entry.get("min"))
        bmax = as_xyz(bounds_entry.get("max"))
    else:
        return None
    if bmin is None or bmax is None:
        return None
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def extract_link_box_vertices(link_entry: Optional[dict]) -> Optional[np.ndarray]:
    if not isinstance(link_entry, dict):
        return None
    info = link_entry.get("info") if isinstance(link_entry.get("info"), dict) else link_entry
    for key in ("visual_bounds", "collision_bounds", "bbox"):
        verts = box_corners_from_bounds(info.get(key))
        if verts is not None:
            return verts
    return None


def build_static_shapes(
    static_info: dict,
) -> Tuple[
    List[dict],
    Dict[str, Tuple[float, float, float]],
    Dict[str, str],
    Dict[str, Tuple[float, float, float]],
]:
    shapes: List[dict] = []
    object_colors: Dict[str, Tuple[float, float, float]] = {}
    legend_labels: Dict[str, str] = {}
    entity_colors: Dict[str, Tuple[float, float, float]] = {}

    scene_layout = static_info.get("scene_layout", {}) or {}
    fixture_entries = scene_layout.get("fixtures", {}) or {}
    object_entries = scene_layout.get("objects", {}) or {}

    fixture_base_labels: Dict[str, str] = {
        name: str(
            fixture_info.get("cls")
            or fixture_info.get("class_name")
            or fixture_info.get("nat_lang")
            or name
        )
        for name, fixture_info in fixture_entries.items()
    }
    object_base_labels: Dict[str, str] = {
        name: str(obj_info.get("category") or name) for name, obj_info in object_entries.items()
    }
    label_counts: Dict[str, int] = {}
    for label in list(fixture_base_labels.values()) + list(object_base_labels.values()):
        label_counts[label] = label_counts.get(label, 0) + 1
    fixture_group_labels: Dict[str, List[str]] = {}
    for label in set(fixture_base_labels.values()):
        fixture_group_labels.setdefault(fixture_group(label), []).append(label)
    fixture_label_colors: Dict[str, Tuple[float, float, float]] = {}
    for labels in fixture_group_labels.values():
        ordered = sorted(labels)
        for idx, label in enumerate(ordered):
            fixture_label_colors[label] = grouped_color(label, idx, len(ordered), is_object=False)

    object_label_colors: Dict[str, Tuple[float, float, float]] = {}
    ordered_object_labels = sorted(set(object_base_labels.values()))
    for idx, label in enumerate(ordered_object_labels):
        object_label_colors[label] = vivid_object_color(label, idx, len(ordered_object_labels))

    for name, fixture_info in fixture_entries.items():
        local_vertices = box_corners_from_extents(fixture_info.get("size"))
        world_transform = fixture_transform(fixture_info)
        if local_vertices is None or world_transform is None:
            continue
        base_label = fixture_base_labels.get(name, name)
        edge = fixture_label_colors.get(base_label, (0.40, 0.40, 0.40))
        entity_colors[name] = edge
        shapes.append(
            dict(
                kind="fixture",
                name=name,
                local_vertices=local_vertices,
                world_transform=world_transform,
                color=(edge[0], edge[1], edge[2], 0.035),
                edge=edge,
            )
        )
        legend_labels[name] = base_label

    for name, obj_info in object_entries.items():
        local_vertices = box_corners_from_bounds(obj_info.get("bbox"))
        if local_vertices is None:
            radius = obj_info.get("horizontal_radius")
            if radius is not None:
                size = [2.0 * float(radius), 2.0 * float(radius), 2.0 * float(radius)]
                local_vertices = box_corners_from_extents(size)
        if local_vertices is None:
            continue
        base_label = object_base_labels.get(name, name)
        color = object_label_colors.get(base_label, (0.2, 0.55, 0.8))
        object_colors[name] = color
        entity_colors[name] = color
        shapes.append(
            dict(
                kind="object",
                name=name,
                local_vertices=local_vertices,
                color=(color[0], color[1], color[2], 0.64),
                edge=color,
            )
        )
        legend_labels[name] = (
            f"{base_label} ({name})" if label_counts.get(base_label, 0) > 1 else base_label
        )

    return shapes, object_colors, legend_labels, entity_colors


def parse_frame(frame: dict) -> dict:
    data = frame.get("data", frame)
    robot = data.get("robot", {}) or {}
    scene = data.get("scene", {}) or {}

    root_pose = robot.get("root_pose", {}) or {}
    eef_pose = robot.get("end_effector_pose", {}) or {}

    root_pos = as_xyz(root_pose.get("position"))
    eef_pos = as_xyz(eef_pose.get("position"))
    root_quat = as_quat_xyzw(root_pose.get("orientation"))
    eef_quat = as_quat_xyzw(eef_pose.get("orientation"))
    root_vel = robot.get("root_velocity", {}) or {}
    eef_vel = robot.get("end_effector_velocity", {}) or {}
    link_poses_raw = robot.get("link_poses", {}) or {}
    physics = data.get("physics", {}) or {}

    object_states = {}
    object_speeds = {}
    for name, obj_state in (scene.get("objects", {}) or {}).items():
        pose = obj_state.get("pose", {}) or {}
        pos = as_xyz(pose.get("position"))
        quat = as_quat_xyzw(pose.get("orientation"))
        if pos is None or quat is None:
            continue
        object_states[name] = dict(mat=compose_transform(pos, quat_xyzw_to_matrix(quat)), pos=pos)
        vel = obj_state.get("velocity", {}) or {}
        lin = as_xyz(vel.get("linear"))
        if lin is not None:
            object_speeds[name] = float(np.linalg.norm(lin))

    fixture_states = {}
    fixture_open_values: List[float] = []
    camera_states = {}
    for name, fx_state in (scene.get("fixtures", {}) or {}).items():
        door_state = fx_state.get("door_state")
        state_vals: List[float] = []
        if isinstance(door_state, dict):
            for val in door_state.values():
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fval):
                    state_vals.append(fval)
        if state_vals:
            fixture_open_values.extend(state_vals)
        fixture_states[name] = dict(
            door_state=door_state,
            state=fx_state.get("state"),
        )
    for name, cam_state in (scene.get("cameras", {}) or {}).items():
        if not isinstance(cam_state, dict):
            continue
        pose = cam_state.get("pose", {}) or {}
        cam_pos = as_xyz(pose.get("position"))
        cam_quat = as_quat_xyzw(pose.get("orientation"))
        if cam_pos is None or cam_quat is None:
            continue
        camera_states[name] = dict(
            mat=compose_transform(cam_pos, quat_xyzw_to_matrix(cam_quat)),
            pos=cam_pos,
            quat=cam_quat,
            fovy=float(cam_state.get("fovy")) if cam_state.get("fovy") is not None else None,
        )

    return dict(
        step=int(frame.get("step", data.get("task", {}).get("timestep", -1))),
        success=bool(data.get("task", {}).get("success", False)),
        root_pos=root_pos,
        root_quat=root_quat,
        root_mat=(
            compose_transform(root_pos, quat_xyzw_to_matrix(root_quat))
            if root_pos is not None and root_quat is not None
            else None
        ),
        root_linear_speed=(
            float(np.linalg.norm(as_xyz(root_vel.get("linear"))))
            if as_xyz(root_vel.get("linear")) is not None
            else np.nan
        ),
        root_angular_speed=(
            float(np.linalg.norm(as_xyz(root_vel.get("angular"))))
            if as_xyz(root_vel.get("angular")) is not None
            else np.nan
        ),
        eef_linear_speed=(
            float(np.linalg.norm(as_xyz(eef_vel.get("linear"))))
            if as_xyz(eef_vel.get("linear")) is not None
            else np.nan
        ),
        eef_angular_speed=(
            float(np.linalg.norm(as_xyz(eef_vel.get("angular"))))
            if as_xyz(eef_vel.get("angular")) is not None
            else np.nan
        ),
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        eef_mat=(
            compose_transform(eef_pos, quat_xyzw_to_matrix(eef_quat))
            if eef_pos is not None and eef_quat is not None
            else None
        ),
        robot_link_poses={
            name: pose_to_transform((pose or {}).get("position"), (pose or {}).get("orientation"))
            for name, pose in link_poses_raw.items()
            if isinstance(pose, dict)
        },
        joint_names=list(robot.get("joint_names", []) or []),
        joint_positions=(
            np.asarray(robot.get("joint_positions", []), dtype=float).reshape(-1)
            if robot.get("joint_positions") is not None
            else np.empty((0,), dtype=float)
        ),
        joint_velocities=(
            np.asarray(robot.get("joint_velocities", []), dtype=float).reshape(-1)
            if robot.get("joint_velocities") is not None
            else np.empty((0,), dtype=float)
        ),
        object_states=object_states,
        object_speeds=object_speeds,
        fixture_states=fixture_states,
        camera_states=camera_states,
        max_fixture_open=max(fixture_open_values) if fixture_open_values else np.nan,
        mean_object_speed=(
            float(np.mean(list(object_speeds.values()))) if object_speeds else np.nan
        ),
        max_object_speed=(float(np.max(list(object_speeds.values()))) if object_speeds else np.nan),
        total_contact_count=float(physics.get("total_contact_count", np.nan)),
        robot_contact_count=float(physics.get("robot_contact_count", np.nan)),
        max_robot_link_force=float(physics.get("max_robot_link_force", np.nan)),
        robot_force_sum=float(physics.get("robot_force_sum", np.nan)),
        robot_force_cumulative=float(physics.get("robot_force_cumulative", np.nan)),
        end_effector_force_norm=float(physics.get("end_effector_force_norm", np.nan)),
    )


def compute_bounds(
    frames: List[dict],
    shapes: List[dict],
    pad: float,
    robot_link_shapes: Optional[Dict[str, dict]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mins = []
    maxs = []
    for shape in shapes:
        if shape["kind"] == "fixture":
            verts = apply_transform(shape["local_vertices"], shape["world_transform"])
            mins.append(np.min(verts, axis=0))
            maxs.append(np.max(verts, axis=0))

    for frame in frames:
        if frame.get("root_pos") is not None:
            mins.append(frame["root_pos"])
            maxs.append(frame["root_pos"])
        if frame.get("eef_pos") is not None:
            mins.append(frame["eef_pos"])
            maxs.append(frame["eef_pos"])
        for shape in shapes:
            if shape["kind"] != "object":
                continue
            obj_state = frame.get("object_states", {}).get(shape["name"])
            if obj_state is None:
                continue
            verts = apply_transform(shape["local_vertices"], obj_state["mat"])
            mins.append(np.min(verts, axis=0))
            maxs.append(np.max(verts, axis=0))
        if robot_link_shapes:
            for link_name, link_shape in robot_link_shapes.items():
                link_tf = frame.get("robot_link_poses", {}).get(link_name)
                if link_tf is None:
                    continue
                verts = apply_transform(link_shape["local_vertices"], link_tf)
                mins.append(np.min(verts, axis=0))
                maxs.append(np.max(verts, axis=0))

    if not mins:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])

    mn = np.min(np.stack(mins, axis=0), axis=0)
    mx = np.max(np.stack(maxs, axis=0), axis=0)
    span = np.maximum(mx - mn, 1e-3)
    return mn - pad * span, mx + pad * span


def build_robot_link_shapes(static_info: dict) -> Dict[str, dict]:
    robot_static = static_info.get("robot", {}) or {}
    link_metadata = robot_static.get("link_metadata", {}) or {}
    shapes: Dict[str, dict] = {}
    redundant_parent_links = {
        "robot0_base",
        "mobilebase0_base",
        "mobilebase0_fixed_support",
        "mobilebase0_support",
        "robot0_right_hand",
        "gripper0_right_right_gripper",
        "gripper0_right_eef",
    }
    for link_name, link_info in link_metadata.items():
        if not isinstance(link_info, dict):
            continue
        if str(link_name).lower() in redundant_parent_links:
            continue
        vertices = extract_link_box_vertices(link_info)
        if vertices is None:
            continue
        lowered = str(link_name).lower()
        is_base = (
            str((robot_static.get("base_link") or {}).get("name", "")).lower() == lowered
            or "mobilebase" in lowered
            or "base" in lowered
        )
        is_eef = str(
            (robot_static.get("end_effector_link") or {}).get("name", "")
        ).lower() == lowered or any(key in lowered for key in ("gripper", "hand", "eef", "tool"))
        if is_eef:
            face = (*ROBOT_EEF_COLOR, 0.20)
            edge = ROBOT_EEF_EDGE
        elif is_base:
            face = (*ROBOT_BASE_COLOR, 0.16)
            edge = ROBOT_BASE_EDGE
        else:
            face = (*ROBOT_LINK_COLOR, 0.08)
            edge = ROBOT_LINK_EDGE
        scale = 1.0
        if is_base:
            scale = 1.18
        elif is_eef:
            scale = 1.35
        center = np.mean(vertices, axis=0, keepdims=True)
        scaled_vertices = center + (vertices - center) * scale
        shapes[link_name] = dict(
            local_vertices=scaled_vertices, face=face, edge=edge, is_base=is_base, is_eef=is_eef
        )
    return shapes


def discover_rollout_video(input_path: str) -> Optional[str]:
    root = Path(input_path).resolve().parent
    candidates = sorted(root.glob("*.mp4"))
    preferred = [
        p
        for p in candidates
        if "privileged_information" not in p.name and "with_future_img" not in p.name
    ]
    if preferred:
        return str(preferred[0])
    fallback = [p for p in candidates if "privileged_information" not in p.name]
    return str(fallback[0]) if fallback else None


def load_multiview_frames(video_path: str, stride: int) -> Dict[str, List[np.ndarray]]:
    frames = {"primary": [], "secondary": [], "wrist": []}
    reader = imageio.get_reader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if stride > 1 and idx % stride != 0:
                continue
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[1] < 3:
                continue
            width = arr.shape[1] // 3
            frames["primary"].append(arr[:, 0:width, :].copy())
            frames["secondary"].append(arr[:, width : width * 2, :].copy())
            frames["wrist"].append(arr[:, width * 2 : width * 3, :].copy())
    finally:
        reader.close()
    return frames


def rollout_frame_stream(video_path: str, stride: int) -> Iterable[np.ndarray]:
    reader = imageio.get_reader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if stride > 1 and idx % stride != 0:
                continue
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[1] < 2:
                continue
            yield arr.copy()
    finally:
        reader.close()


def camera_world_transform(camera_name: str, camera_cfg: dict, frame: dict) -> Optional[np.ndarray]:
    live_camera = (frame.get("camera_states", {}) or {}).get(camera_name)
    if isinstance(live_camera, dict) and live_camera.get("mat") is not None:
        return live_camera["mat"]
    local_pos = as_xyz(camera_cfg.get("pos"))
    local_quat = as_quat_xyzw(camera_cfg.get("quat"))
    if local_pos is None:
        return None
    local_rot = (
        quat_xyzw_to_matrix(local_quat) if local_quat is not None else np.eye(3, dtype=float)
    )
    local_tf = compose_transform(local_pos, local_rot)
    parent_body = str(camera_cfg.get("parent_body") or "").lower()
    parent_tf = None
    if "right_hand" in parent_body or "eef" in parent_body:
        parent_tf = frame.get("eef_mat")
    elif "mobilebase" in parent_body or "support" in parent_body or "base" in parent_body:
        parent_tf = frame.get("root_mat")
    if parent_tf is None:
        return local_tf
    return parent_tf @ local_tf


def project_world_points(
    points_xyz: np.ndarray,
    camera_tf: np.ndarray,
    image_shape: Tuple[int, int, int],
    fovy_deg: float,
) -> np.ndarray:
    cam_pos = np.asarray(camera_tf[:3, 3], dtype=float)
    cam_rot = np.asarray(camera_tf[:3, :3], dtype=float)
    rel = np.asarray(points_xyz, dtype=float) - cam_pos[None, :]
    # Match the explorer convention: MuJoCo camera rotation is camera-to-world, and for
    # row-vector points world-to-camera uses rel @ R rather than rel @ R.T.
    cam_points = rel @ cam_rot

    height, width = int(image_shape[0]), int(image_shape[1])
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
    fx = fy
    cx = 0.5 * width
    cy = 0.5 * height

    projected: List[List[float]] = []
    for point in cam_points:
        z = -float(point[2])
        if z <= 1e-6:
            if projected:
                projected.append(projected[-1])
            else:
                projected.append([cx, cy])
            continue
        x = cx + fx * (float(point[0]) / z)
        y = cy - fy * (float(point[1]) / z)
        projected.append(
            [
                float(np.clip(x, 0, width - 1)),
                float(np.clip(y, 0, height - 1)),
            ]
        )
    return np.asarray(projected, dtype=np.float32)


def project_visible_point(
    point_xyz: np.ndarray,
    camera_tf: np.ndarray,
    image_shape: Tuple[int, int, int],
    fovy_deg: float,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    cam_pos = np.asarray(camera_tf[:3, 3], dtype=float)
    cam_rot = np.asarray(camera_tf[:3, :3], dtype=float)
    rel = np.asarray(point_xyz, dtype=float).reshape(1, 3) - cam_pos[None, :]
    cam_point = (rel @ cam_rot)[0]

    z = -float(cam_point[2])
    if z <= 1e-6:
        return None, None

    height, width = int(image_shape[0]), int(image_shape[1])
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
    fx = fy
    cx = 0.5 * width
    cy = 0.5 * height
    x = cx + fx * (float(cam_point[0]) / z)
    y = cy - fy * (float(cam_point[1]) / z)
    return (
        np.asarray(
            [
                float(np.clip(x, 0, width - 1)),
                float(np.clip(y, 0, height - 1)),
            ],
            dtype=np.float32,
        ),
        z,
    )


def annotate_multiview_frame(
    frame_img: np.ndarray,
    camera_name: str,
    camera_cfg: dict,
    frame: dict,
    shapes: List[dict],
    object_colors: Dict[str, Tuple[float, float, float]],
    entity_colors: Dict[str, Tuple[float, float, float]],
    robot_link_shapes: Dict[str, dict],
    root_path: List[np.ndarray],
    eef_path: List[np.ndarray],
    object_paths: Dict[str, List[np.ndarray]],
) -> np.ndarray:
    camera_tf = camera_world_transform(camera_name, camera_cfg, frame)
    if camera_tf is None:
        return frame_img

    fovy_deg = 60.0
    live_camera = (frame.get("camera_states", {}) or {}).get(camera_name, {})
    if isinstance(live_camera, dict) and live_camera.get("fovy") is not None:
        try:
            fovy_deg = float(live_camera["fovy"])
        except (TypeError, ValueError):
            pass
    elif isinstance(camera_cfg.get("camera_attribs"), dict):
        try:
            fovy_deg = float(camera_cfg["camera_attribs"].get("fovy", fovy_deg))
        except (TypeError, ValueError):
            pass

    img = Image.fromarray(frame_img.copy())
    draw = ImageDraw.Draw(img, "RGBA")

    def _draw_path(
        points: List[np.ndarray], rgb: Tuple[float, float, float], width: int, alpha: int
    ) -> None:
        if len(points) < 2:
            return
        pts_arr = np.asarray(points, dtype=float)
        pts_2d = project_world_points(pts_arr, camera_tf, frame_img.shape, fovy_deg)
        if pts_2d.shape[0] >= 2:
            draw.line(
                [tuple(pt) for pt in pts_2d],
                fill=tuple(int(255 * c) for c in rgb) + (alpha,),
                width=width,
            )

    def _draw_point(
        point: Optional[np.ndarray], rgb: Tuple[float, float, float], radius: int, alpha: int
    ) -> None:
        if point is None:
            return
        pts_2d = project_world_points(
            np.asarray([point], dtype=float), camera_tf, frame_img.shape, fovy_deg
        )
        if pts_2d.shape[0] == 0:
            return
        x, y = float(pts_2d[0, 0]), float(pts_2d[0, 1])
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=tuple(int(255 * c) for c in rgb) + (alpha,),
            outline=(255, 255, 255, min(alpha + 20, 255)),
            width=1,
        )

    def _draw_box(
        point: Optional[np.ndarray], rgb: Tuple[float, float, float], half_size: int, alpha: int
    ) -> None:
        if point is None:
            return
        pts_2d = project_world_points(
            np.asarray([point], dtype=float), camera_tf, frame_img.shape, fovy_deg
        )
        if pts_2d.shape[0] == 0:
            return
        x, y = float(pts_2d[0, 0]), float(pts_2d[0, 1])
        draw.rectangle(
            [(x - half_size, y - half_size), (x + half_size, y + half_size)],
            outline=tuple(int(255 * c) for c in rgb) + (alpha,),
            width=2,
        )

    def _project_visible_rect(points_xyz: np.ndarray) -> Optional[np.ndarray]:
        cam_pos = np.asarray(camera_tf[:3, 3], dtype=float)
        cam_rot = np.asarray(camera_tf[:3, :3], dtype=float)
        rel = np.asarray(points_xyz, dtype=float) - cam_pos[None, :]
        cam_points = rel @ cam_rot

        visible_points: List[np.ndarray] = []
        height, width = int(frame_img.shape[0]), int(frame_img.shape[1])
        fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
        fx = fy
        cx = 0.5 * width
        cy = 0.5 * height
        for point in cam_points:
            z = -float(point[2])
            if z <= 1e-6:
                continue
            x = cx + fx * (float(point[0]) / z)
            y = cy - fy * (float(point[1]) / z)
            visible_points.append(
                np.asarray(
                    [
                        float(np.clip(x, 0, width - 1)),
                        float(np.clip(y, 0, height - 1)),
                    ],
                    dtype=np.float32,
                )
            )
        if not visible_points:
            return None
        pts = np.vstack(visible_points)
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        if xmax <= xmin or ymax <= ymin:
            return None
        return np.asarray([xmin, ymin, xmax, ymax], dtype=np.float32)

    def _coverage_ratio(rect: np.ndarray, cover: np.ndarray) -> float:
        ix0 = max(float(rect[0]), float(cover[0]))
        iy0 = max(float(rect[1]), float(cover[1]))
        ix1 = min(float(rect[2]), float(cover[2]))
        iy1 = min(float(rect[3]), float(cover[3]))
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        area = max((float(rect[2]) - float(rect[0])) * (float(rect[3]) - float(rect[1])), 1e-6)
        return float(inter / area)

    _draw_path(root_path, ROBOT_BASE_COLOR, width=3, alpha=210)
    _draw_path(eef_path, ROBOT_EEF_COLOR, width=3, alpha=230)
    _draw_box(frame.get("root_pos"), ROBOT_BASE_COLOR, half_size=7, alpha=255)
    _draw_box(frame.get("eef_pos"), ROBOT_EEF_COLOR, half_size=8, alpha=255)
    _draw_point(frame.get("root_pos"), ROBOT_BASE_COLOR, radius=4, alpha=255)
    _draw_point(frame.get("eef_pos"), ROBOT_EEF_COLOR, radius=5, alpha=255)

    overlay_entries: List[Dict[str, Any]] = []
    for shape in shapes:
        if shape["kind"] == "fixture":
            verts = apply_transform(shape["local_vertices"], shape["world_transform"])
            center = np.mean(verts, axis=0)
            center_2d, depth = project_visible_point(center, camera_tf, frame_img.shape, fovy_deg)
            if center_2d is None or depth is None:
                continue
            rect_2d = _project_visible_rect(verts)
            if rect_2d is None:
                continue
            rgb = entity_colors.get(shape["name"], (0.55, 0.55, 0.55))
            overlay_entries.append(
                dict(
                    kind="fixture",
                    depth=depth,
                    color=rgb,
                    center_2d=center_2d,
                    rect_2d=rect_2d,
                    half_size=5,
                    alpha=150,
                )
            )
            continue
        obj_state = frame.get("object_states", {}).get(shape["name"])
        if obj_state is None:
            continue
        verts = apply_transform(shape["local_vertices"], obj_state["mat"])
        center = np.mean(verts, axis=0)
        center_2d, depth = project_visible_point(center, camera_tf, frame_img.shape, fovy_deg)
        if center_2d is None or depth is None:
            continue
        rect_2d = _project_visible_rect(verts)
        if rect_2d is None:
            continue
        overlay_entries.append(
            dict(
                kind="object",
                depth=depth,
                color=object_colors.get(shape["name"], (0.95, 0.35, 0.20)),
                center_2d=center_2d,
                rect_2d=rect_2d,
                half_size=6,
                alpha=240,
            )
        )

    for link_name, link_shape in robot_link_shapes.items():
        link_tf = frame.get("robot_link_poses", {}).get(link_name)
        if link_tf is None:
            continue
        verts = apply_transform(link_shape["local_vertices"], link_tf)
        center = np.mean(verts, axis=0)
        center_2d, depth = project_visible_point(center, camera_tf, frame_img.shape, fovy_deg)
        if center_2d is None or depth is None:
            continue
        edge = link_shape["edge"]
        half_size = 8 if link_shape.get("is_eef") else 7 if link_shape.get("is_base") else 4
        rect_2d = _project_visible_rect(verts)
        if rect_2d is None:
            continue
        overlay_entries.append(
            dict(
                kind="robot",
                depth=depth,
                color=edge,
                center_2d=center_2d,
                rect_2d=rect_2d,
                half_size=half_size,
                alpha=220,
            )
        )

    height, width = int(frame_img.shape[0]), int(frame_img.shape[1])
    occupied = np.zeros((height, width), dtype=bool)
    min_visible_ratio = 0.03
    for entry in sorted(overlay_entries, key=lambda item: item["depth"]):
        rect = entry["rect_2d"]
        x0 = max(0, min(width - 1, int(math.floor(float(rect[0])))))
        y0 = max(0, min(height - 1, int(math.floor(float(rect[1])))))
        x1 = max(0, min(width, int(math.ceil(float(rect[2])))))
        y1 = max(0, min(height, int(math.ceil(float(rect[3])))))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = occupied[y0:y1, x0:x1]
        area = patch.size
        if area <= 0:
            continue
        visible_ratio = 1.0 - float(np.count_nonzero(patch)) / float(area)
        if visible_ratio < min_visible_ratio:
            continue
        occupied[y0:y1, x0:x1] = True

        rgb255 = tuple(int(255 * c) for c in entry["color"])
        x, y = float(entry["center_2d"][0]), float(entry["center_2d"][1])
        half_size = int(entry["half_size"])
        alpha = int(entry["alpha"])
        draw.rectangle(
            [(x - half_size, y - half_size), (x + half_size, y + half_size)],
            outline=rgb255 + (alpha,),
            width=2,
        )
        draw.ellipse(
            [(x - 2, y - 2), (x + 2, y + 2)],
            fill=rgb255 + (min(alpha + 15, 255),),
            outline=(255, 255, 255, 255),
            width=1,
        )

    for name, path in object_paths.items():
        color = object_colors.get(name, (0.95, 0.35, 0.20))
        _draw_path(path, color, width=2, alpha=200)
    return np.asarray(img, dtype=np.uint8)


def finite_limits(arr: np.ndarray, default: Tuple[float, float]) -> Tuple[float, float]:
    finite = np.isfinite(arr)
    if not finite.any():
        return default
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if math.isclose(lo, hi):
        eps = max(abs(lo) * 0.1, 1e-3)
        lo -= eps
        hi += eps
    pad = max((hi - lo) * 0.08, 1e-3)
    return lo - pad, hi + pad


def split_lines_for_columns(
    lines: List[str],
    *,
    max_cols: int = 3,
    max_lines_per_col: int = 30,
) -> List[List[str]]:
    if not lines:
        return []
    columns = max(1, math.ceil(len(lines) / max_lines_per_col))
    columns = min(max_cols, columns)
    lines_per_col = max(1, math.ceil(len(lines) / columns))
    return [
        lines[col_idx * lines_per_col : (col_idx + 1) * lines_per_col] for col_idx in range(columns)
    ]


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    if frame.shape[1] == target_width:
        return frame
    scale = float(target_width) / float(frame.shape[1])
    new_height = max(1, int(round(frame.shape[0] * scale)))
    img = Image.fromarray(frame)
    resized = img.resize((target_width, new_height), Image.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def pad_frame_to_even_dimensions(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_height = height + (height % 2)
    target_width = width + (width % 2)
    if target_height == height and target_width == width:
        return frame
    padded = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    padded[:height, :width] = frame
    return padded


def update_path(line, path_points: List[np.ndarray]) -> None:
    if len(path_points) < 2:
        line.set_data([], [])
        line.set_3d_properties([])
        return
    pts = np.vstack(path_points)
    line.set_data(pts[:, 0], pts[:, 1])
    line.set_3d_properties(pts[:, 2])


def compute_focus_bounds(
    parsed_frames: List[dict],
    shapes: List[dict],
    robot_link_shapes: Dict[str, dict],
    pad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for frame in parsed_frames:
        for key in ("root_pos", "eef_pos"):
            pos = frame.get(key)
            if pos is not None:
                mins.append(np.asarray(pos, dtype=float))
                maxs.append(np.asarray(pos, dtype=float))
        for shape in shapes:
            if shape["kind"] != "object":
                continue
            obj_state = frame.get("object_states", {}).get(shape["name"])
            if obj_state is None:
                continue
            verts = apply_transform(shape["local_vertices"], obj_state["mat"])
            mins.append(np.min(verts, axis=0))
            maxs.append(np.max(verts, axis=0))
        for link_name, link_shape in robot_link_shapes.items():
            link_tf = frame.get("robot_link_poses", {}).get(link_name)
            if link_tf is None:
                continue
            verts = apply_transform(link_shape["local_vertices"], link_tf)
            mins.append(np.min(verts, axis=0))
            maxs.append(np.max(verts, axis=0))

    if not mins:
        return np.array([-1.0, -1.0, 0.0], dtype=float), np.array([1.0, 1.0, 1.5], dtype=float)

    mn = np.min(np.vstack(mins), axis=0)
    mx = np.max(np.vstack(maxs), axis=0)
    center = 0.5 * (mn + mx)
    span = np.maximum(mx - mn, np.array([1.6, 1.6, 1.2], dtype=float))
    span[:2] = np.maximum(span[:2], np.array([1.8, 1.8], dtype=float))
    span[2] = max(span[2], 1.4)
    span = span * (1.0 + max(pad, 0.12))
    return center - 0.5 * span, center + 0.5 * span


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize RoboCasa privileged information.")
    ap.add_argument("--input", required=True, help="Path to privileged_information_*.json")
    ap.add_argument("--output", default=None, help="Output .mp4 or .gif")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--bounds_pad", type=float, default=0.05)
    ap.add_argument(
        "--max_width",
        type=int,
        default=1600,
        help="Clamp output width to this many pixels to speed up rendering.",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=18,
        help="FFMPEG CRF quality (higher is smaller/faster, lower is higher quality).",
    )
    ap.add_argument(
        "--preset",
        type=str,
        default="veryfast",
        help="FFMPEG preset for encoding speed/quality tradeoff.",
    )
    ap.add_argument(
        "--preview_frames",
        type=int,
        default=0,
        help="If >0, stop after writing this many frames (for quick previews).",
    )
    ap.add_argument(
        "--rollout_video",
        default=None,
        help="Optional rollout MP4 with primary|secondary|wrist views",
    )
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else input_path.with_suffix(".mp4")

    with open(input_path, "r") as fh:
        root = json.load(fh)

    static_info = root.get("privileged_static_info", {}) or {}
    raw_frames = root.get("privileged_dynamic_info", []) or []
    parsed_frames = []
    for idx, frame in enumerate(raw_frames):
        if args.stride > 1 and idx % args.stride != 0:
            continue
        parsed_frames.append(parse_frame(frame))
    parsed_frames.sort(key=lambda x: x["step"])
    if not parsed_frames:
        raise RuntimeError("No privileged frames found after stride filtering.")

    shapes, object_colors, legend_labels, entity_colors = build_static_shapes(static_info)
    robot_link_shapes = build_robot_link_shapes(static_info)
    eef_box_vertices_local = (
        box_corners_from_extents(DEFAULT_EEF_BOX_SIZE)
        if not any(shape.get("is_eef") for shape in robot_link_shapes.values())
        else None
    )
    scene_mn, scene_mx = compute_bounds(
        parsed_frames, shapes, max(args.bounds_pad, 0.0), robot_link_shapes
    )
    focus_mn, focus_mx = compute_focus_bounds(
        parsed_frames, shapes, robot_link_shapes, max(args.bounds_pad, 0.0)
    )
    rollout_video_path = args.rollout_video or discover_rollout_video(args.input)
    rollout_iter = None
    if rollout_video_path:
        try:
            rollout_iter = iter(rollout_frame_stream(rollout_video_path, args.stride))
        except Exception:
            rollout_iter = None

    steps = np.asarray([frame["step"] for frame in parsed_frames], dtype=float)
    root_speed = np.asarray([frame["root_linear_speed"] for frame in parsed_frames], dtype=float)
    root_omega = np.asarray([frame["root_angular_speed"] for frame in parsed_frames], dtype=float)
    eef_speed = np.asarray([frame["eef_linear_speed"] for frame in parsed_frames], dtype=float)
    eef_omega = np.asarray([frame["eef_angular_speed"] for frame in parsed_frames], dtype=float)
    max_robot_force = np.asarray(
        [frame["max_robot_link_force"] for frame in parsed_frames], dtype=float
    )
    eef_force = np.asarray(
        [frame["end_effector_force_norm"] for frame in parsed_frames], dtype=float
    )
    robot_contacts = np.asarray(
        [frame["robot_contact_count"] for frame in parsed_frames], dtype=float
    )

    joint_count = max((frame["joint_positions"].size for frame in parsed_frames), default=0)
    if joint_count == 0:
        joint_count = 1
    joint_pos_history = np.full((len(parsed_frames), joint_count), np.nan, dtype=float)
    joint_vel_history = np.full((len(parsed_frames), joint_count), np.nan, dtype=float)
    for idx, frame in enumerate(parsed_frames):
        jp = frame["joint_positions"]
        jv = frame["joint_velocities"]
        if jp.size:
            joint_pos_history[idx, : min(joint_count, jp.size)] = jp[:joint_count]
        if jv.size:
            joint_vel_history[idx, : min(joint_count, jv.size)] = jv[:joint_count]

    max_output_width = max(320, int(args.max_width))

    ext = output_path.suffix.lower().lstrip(".")
    if ext == "gif":
        writer = imageio.get_writer(output_path, mode="I", fps=args.fps)
    else:
        writer = imageio.get_writer(
            output_path,
            format="FFMPEG",
            fps=args.fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=[
                "-movflags",
                "+faststart",
                "-preset",
                str(args.preset),
                "-crf",
                str(args.crf),
            ],
        )

    fig = plt.figure(figsize=(18.6, 9.6), dpi=160)
    fig.subplots_adjust(bottom=0.26)
    grid = fig.add_gridspec(
        4,
        4,
        width_ratios=[1.95, 1.00, 0.92, 0.82],
        height_ratios=[1.0, 1.0, 1.0, 1.0],
        wspace=0.22,
        hspace=0.28,
    )
    ax_scene = fig.add_subplot(grid[:, 0], projection="3d")
    ax_scene_zoom_primary = fig.add_subplot(grid[0:2, 1], projection="3d")
    ax_scene_zoom_secondary = fig.add_subplot(grid[2:4, 1], projection="3d")
    ax_joint = fig.add_subplot(grid[0:2, 2])
    ax_speed = fig.add_subplot(grid[2:4, 2])
    ax_state = fig.add_subplot(grid[0:2, 3])
    ax_status = fig.add_subplot(grid[2:4, 3])
    ax_status.axis("off")
    legend_ax = fig.add_axes([0.04, 0.02, 0.92, 0.20], frameon=False)
    legend_ax.set_axis_off()

    ax_scene.set_xlim(scene_mn[0], scene_mx[0])
    ax_scene.set_ylim(scene_mn[1], scene_mx[1])
    ax_scene.set_zlim(scene_mn[2], scene_mx[2])
    try:
        scene_span = np.maximum(scene_mx - scene_mn, 1e-6)
        ax_scene.set_box_aspect(scene_span)
    except Exception:
        pass
    ax_scene.set_xlabel("X")
    ax_scene.set_ylabel("Y")
    ax_scene.set_zlabel("Z")
    ax_scene.view_init(elev=24, azim=-56)
    focus_span = np.maximum(focus_mx - focus_mn, 1e-6)
    for ax_zoom, azim, title in (
        (ax_scene_zoom_primary, -112, "Primary-Side Zoom"),
        (ax_scene_zoom_secondary, -68, "Secondary-Side Zoom"),
    ):
        ax_zoom.set_xlim(focus_mn[0], focus_mx[0])
        ax_zoom.set_ylim(focus_mn[1], focus_mx[1])
        ax_zoom.set_zlim(focus_mn[2], focus_mx[2])
        try:
            ax_zoom.set_box_aspect(focus_span)
        except Exception:
            pass
        ax_zoom.set_xlabel("X")
        ax_zoom.set_ylabel("Y")
        ax_zoom.set_zlabel("Z")
        ax_zoom.view_init(elev=7, azim=azim)
        ax_zoom.set_title(title, loc="left")
        ax_zoom.grid(False)
        ax_zoom.set_axis_off()

    if steps.size:
        for axis in (ax_joint, ax_speed, ax_state):
            axis.set_xlim(steps[0], steps[-1])

    ax_joint.set_title("Robot joints", loc="left")
    ax_joint.set_ylabel("position")
    ax_joint.grid(True, alpha=0.25)

    joint_lines = []
    cmap = plt.get_cmap("tab20", max(4, min(joint_count, 20)))
    shown_joint_count = joint_count
    joint_name_labels = list(parsed_frames[0].get("joint_names", []) or [])
    for joint_idx in range(shown_joint_count):
        color = cmap(joint_idx % cmap.N)
        label = (
            joint_name_labels[joint_idx]
            if joint_idx < len(joint_name_labels) and joint_name_labels[joint_idx]
            else f"joint_{joint_idx}"
        )
        (line,) = ax_joint.plot([], [], color=color, linewidth=1.4, label=label)
        joint_lines.append(line)
    ymin, ymax = finite_limits(joint_pos_history[:, :shown_joint_count], (-1.0, 1.0))
    ax_joint.set_ylim(ymin, ymax)
    ax_joint.legend(loc="upper right", fontsize="x-small", frameon=True)

    ax_speed.set_title("Base / EEF Speeds", loc="left")
    ax_speed.set_ylabel("magnitude")
    ax_speed.set_xlabel("step")
    ax_speed.grid(True, alpha=0.25)
    (root_speed_line,) = ax_speed.plot([], [], color="tab:blue", label="base |v|")
    (root_omega_line,) = ax_speed.plot([], [], color="tab:orange", label="base |w|")
    (eef_speed_line,) = ax_speed.plot([], [], color="tab:green", label="eef |v|")
    (eef_omega_line,) = ax_speed.plot([], [], color="tab:red", label="eef |w|")
    ymin, ymax = finite_limits(
        np.concatenate([root_speed, root_omega, eef_speed, eef_omega]),
        (0.0, 1.0),
    )
    ax_speed.set_ylim(max(0.0, ymin), ymax)
    ax_speed.legend(loc="upper right", fontsize="x-small", frameon=True)

    ax_state.set_title("Force / Contact", loc="left")
    ax_state.set_ylabel("magnitude")
    ax_state.grid(True, alpha=0.25)
    (max_robot_force_line,) = ax_state.plot([], [], color="tab:red", label="max robot force")
    (eef_force_line,) = ax_state.plot([], [], color="tab:purple", label="eef force")
    (robot_contact_line,) = ax_state.plot(
        [], [], color="black", linestyle="--", label="robot contacts"
    )
    ymin, ymax = finite_limits(
        np.concatenate([max_robot_force, eef_force, robot_contacts, np.array([0.0], dtype=float)]),
        (0.0, 1.0),
    )
    ax_state.set_ylim(max(0.0, ymin), ymax)
    ax_state.legend(loc="upper right", fontsize="x-small", frameon=True)

    joint_cursor = ax_joint.axvline(steps[0], color="black", linestyle="--", linewidth=1.0)
    speed_cursor = ax_speed.axvline(steps[0], color="black", linestyle="--", linewidth=1.0)
    state_cursor = ax_state.axvline(steps[0], color="black", linestyle="--", linewidth=1.0)

    status_text = ax_status.text(0.0, 1.0, "", va="top", ha="left", fontsize=7, family="monospace")

    for shape in shapes:
        linewidth = 0.7 if shape["kind"] == "object" else 0.9
        coll = Poly3DCollection(
            [], facecolors=shape["color"], edgecolors=shape["edge"], linewidths=linewidth
        )
        try:
            coll.set_zsort("min")
            coll.set_zorder(1)
        except Exception:
            pass
        shape["collection"] = coll
        ax_scene.add_collection3d(coll)
        for key, ax_zoom in (
            ("zoom_primary_collection", ax_scene_zoom_primary),
            ("zoom_secondary_collection", ax_scene_zoom_secondary),
        ):
            zoom_coll = Poly3DCollection(
                [], facecolors=shape["color"], edgecolors=shape["edge"], linewidths=linewidth
            )
            try:
                zoom_coll.set_zsort("min")
                zoom_coll.set_zorder(1)
            except Exception:
                pass
            shape[key] = zoom_coll
            ax_zoom.add_collection3d(zoom_coll)

    robot_link_collections = {}
    for link_name, link_shape in robot_link_shapes.items():
        coll = Poly3DCollection(
            [],
            facecolors=link_shape["face"],
            edgecolors=link_shape["edge"],
            linewidths=2.0 if (link_shape.get("is_base") or link_shape.get("is_eef")) else 0.9,
        )
        try:
            coll.set_zsort("min")
            coll.set_zorder(2 if (link_shape.get("is_base") or link_shape.get("is_eef")) else 1)
        except Exception:
            pass
        robot_link_collections[link_name] = coll
        ax_scene.add_collection3d(coll)
        for key, ax_zoom in (
            ("zoom_primary_collection", ax_scene_zoom_primary),
            ("zoom_secondary_collection", ax_scene_zoom_secondary),
        ):
            zoom_coll = Poly3DCollection(
                [],
                facecolors=link_shape["face"],
                edgecolors=link_shape["edge"],
                linewidths=2.0 if (link_shape.get("is_base") or link_shape.get("is_eef")) else 0.9,
            )
            try:
                zoom_coll.set_zsort("min")
                zoom_coll.set_zorder(
                    2 if (link_shape.get("is_base") or link_shape.get("is_eef")) else 1
                )
            except Exception:
                pass
            link_shape[key] = zoom_coll
            ax_zoom.add_collection3d(zoom_coll)

    eef_box_collection = None
    if eef_box_vertices_local is not None:
        eef_box_collection = Poly3DCollection(
            [],
            facecolors=(*ROBOT_EEF_COLOR, 0.24),
            edgecolors=ROBOT_EEF_EDGE,
            linewidths=2.2,
        )
        try:
            eef_box_collection.set_zsort("min")
            eef_box_collection.set_zorder(3)
        except Exception:
            pass
        ax_scene.add_collection3d(eef_box_collection)
        eef_box_collection_zoom_primary = Poly3DCollection(
            [],
            facecolors=(*ROBOT_EEF_COLOR, 0.24),
            edgecolors=ROBOT_EEF_EDGE,
            linewidths=2.2,
        )
        eef_box_collection_zoom_secondary = Poly3DCollection(
            [],
            facecolors=(*ROBOT_EEF_COLOR, 0.24),
            edgecolors=ROBOT_EEF_EDGE,
            linewidths=2.2,
        )
        for coll_zoom, ax_zoom in (
            (eef_box_collection_zoom_primary, ax_scene_zoom_primary),
            (eef_box_collection_zoom_secondary, ax_scene_zoom_secondary),
        ):
            try:
                coll_zoom.set_zsort("min")
                coll_zoom.set_zorder(3)
            except Exception:
                pass
            ax_zoom.add_collection3d(coll_zoom)
    else:
        eef_box_collection_zoom_primary = None
        eef_box_collection_zoom_secondary = None

    root_scatter = ax_scene.scatter(
        [],
        [],
        [],
        s=72,
        c=[ROBOT_BASE_COLOR],
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        alpha=1.0,
        depthshade=False,
        zorder=10,
    )
    eef_scatter = ax_scene.scatter(
        [],
        [],
        [],
        s=92,
        c=[ROBOT_EEF_COLOR],
        marker="^",
        edgecolors="white",
        linewidths=1.6,
        alpha=1.0,
        depthshade=False,
        zorder=11,
    )
    (root_path_line,) = ax_scene.plot([], [], [], color=ROBOT_BASE_COLOR, linewidth=3.0, zorder=10)
    (eef_path_line,) = ax_scene.plot(
        [], [], [], color=ROBOT_EEF_COLOR, linewidth=2.8, linestyle=":", zorder=11
    )
    root_scatter_zoom_primary = ax_scene_zoom_primary.scatter(
        [],
        [],
        [],
        s=72,
        c=[ROBOT_BASE_COLOR],
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        alpha=1.0,
        depthshade=False,
        zorder=10,
    )
    root_scatter_zoom_secondary = ax_scene_zoom_secondary.scatter(
        [],
        [],
        [],
        s=72,
        c=[ROBOT_BASE_COLOR],
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        alpha=1.0,
        depthshade=False,
        zorder=10,
    )
    eef_scatter_zoom_primary = ax_scene_zoom_primary.scatter(
        [],
        [],
        [],
        s=92,
        c=[ROBOT_EEF_COLOR],
        marker="^",
        edgecolors="white",
        linewidths=1.6,
        alpha=1.0,
        depthshade=False,
        zorder=11,
    )
    eef_scatter_zoom_secondary = ax_scene_zoom_secondary.scatter(
        [],
        [],
        [],
        s=92,
        c=[ROBOT_EEF_COLOR],
        marker="^",
        edgecolors="white",
        linewidths=1.6,
        alpha=1.0,
        depthshade=False,
        zorder=11,
    )
    (root_path_line_zoom_primary,) = ax_scene_zoom_primary.plot(
        [], [], [], color=ROBOT_BASE_COLOR, linewidth=3.0, zorder=10
    )
    (eef_path_line_zoom_primary,) = ax_scene_zoom_primary.plot(
        [], [], [], color=ROBOT_EEF_COLOR, linewidth=2.8, linestyle=":", zorder=11
    )
    (root_path_line_zoom_secondary,) = ax_scene_zoom_secondary.plot(
        [], [], [], color=ROBOT_BASE_COLOR, linewidth=3.0, zorder=10
    )
    (eef_path_line_zoom_secondary,) = ax_scene_zoom_secondary.plot(
        [], [], [], color=ROBOT_EEF_COLOR, linewidth=2.8, linestyle=":", zorder=11
    )

    object_path_lines: Dict[str, Any] = {}
    object_path_lines_zoom_primary: Dict[str, Any] = {}
    object_path_lines_zoom_secondary: Dict[str, Any] = {}
    object_paths: Dict[str, List[np.ndarray]] = {}
    object_markers: Dict[str, Any] = {}
    object_markers_zoom_primary: Dict[str, Any] = {}
    object_markers_zoom_secondary: Dict[str, Any] = {}
    root_path: List[np.ndarray] = []
    eef_path: List[np.ndarray] = []

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=ROBOT_BASE_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=9,
            label="robot base",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            color=ROBOT_EEF_COLOR,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=9,
            label="eef",
        ),
        Line2D([0], [0], color=ROBOT_BASE_COLOR, lw=3.0, label="robot path"),
        Line2D([0], [0], color=ROBOT_EEF_COLOR, lw=2.8, linestyle=":", label="eef path"),
    ]
    if eef_box_collection is not None:
        legend_handles.append(
            Line2D([0], [0], color=ROBOT_EEF_EDGE, lw=4.0, alpha=0.85, label="eef extent")
        )
    elif any(shape.get("is_eef") for shape in robot_link_shapes.values()):
        legend_handles.append(
            Line2D([0], [0], color=ROBOT_EEF_EDGE, lw=4.0, alpha=0.85, label="eef extent")
        )
    if any(shape.get("is_base") for shape in robot_link_shapes.values()):
        legend_handles.append(
            Line2D([0], [0], color=ROBOT_BASE_EDGE, lw=4.0, alpha=0.85, label="robot base extent")
        )
    if any(
        (not shape.get("is_base")) and (not shape.get("is_eef"))
        for shape in robot_link_shapes.values()
    ):
        legend_handles.append(
            Line2D([0], [0], color=ROBOT_LINK_EDGE, lw=3.0, alpha=0.85, label="robot links")
        )
    fixture_shape_names = sorted([shape["name"] for shape in shapes if shape["kind"] == "fixture"])
    for name in fixture_shape_names:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=entity_colors.get(name, (0.40, 0.40, 0.40)),
                lw=1.2,
                label=legend_labels.get(name, name),
            )
        )
    for name in sorted(object_colors.keys()):
        color = object_colors[name]
        legend_handles.append(
            Line2D([0], [0], color=color, lw=1.7, label=legend_labels.get(name, name))
        )
    dedup_legend_handles = []
    dedup_legend_labels = []
    seen_legend_labels = set()
    for handle in legend_handles:
        label = handle.get_label()
        if not label or label in seen_legend_labels:
            continue
        seen_legend_labels.add(label)
        dedup_legend_handles.append(handle)
        dedup_legend_labels.append(label)
    legend_ax.legend(
        dedup_legend_handles,
        dedup_legend_labels,
        loc="center",
        frameon=False,
        fontsize="x-small",
        ncol=max(4, min(10, int(np.ceil(len(dedup_legend_handles) / 3)))),
    )

    frames_written = 0
    last_top_frame_shape: Optional[Tuple[int, int, int]] = None

    try:
        for frame_idx, frame in enumerate(parsed_frames):
            step = frame["step"]
            ax_scene.set_title(f"RoboCasa step {step}", loc="left")
            ax_scene_zoom_primary.set_title(f"Primary-Side Zoom {step}", loc="left")
            ax_scene_zoom_secondary.set_title(f"Secondary-Side Zoom {step}", loc="left")

            rpos = frame.get("root_pos")
            if rpos is not None:
                root_scatter._offsets3d = ([rpos[0]], [rpos[1]], [rpos[2]])
                root_scatter_zoom_primary._offsets3d = ([rpos[0]], [rpos[1]], [rpos[2]])
                root_scatter_zoom_secondary._offsets3d = ([rpos[0]], [rpos[1]], [rpos[2]])
                root_path.append(rpos)
            else:
                root_scatter._offsets3d = ([], [], [])
                root_scatter_zoom_primary._offsets3d = ([], [], [])
                root_scatter_zoom_secondary._offsets3d = ([], [], [])
            update_path(root_path_line, root_path)
            update_path(root_path_line_zoom_primary, root_path)
            update_path(root_path_line_zoom_secondary, root_path)

            epos = frame.get("eef_pos")
            if epos is not None:
                eef_scatter._offsets3d = ([epos[0]], [epos[1]], [epos[2]])
                eef_scatter_zoom_primary._offsets3d = ([epos[0]], [epos[1]], [epos[2]])
                eef_scatter_zoom_secondary._offsets3d = ([epos[0]], [epos[1]], [epos[2]])
                eef_path.append(epos)
            else:
                eef_scatter._offsets3d = ([], [], [])
                eef_scatter_zoom_primary._offsets3d = ([], [], [])
                eef_scatter_zoom_secondary._offsets3d = ([], [], [])
            update_path(eef_path_line, eef_path)
            update_path(eef_path_line_zoom_primary, eef_path)
            update_path(eef_path_line_zoom_secondary, eef_path)
            if eef_box_collection is not None:
                eef_mat = frame.get("eef_mat")
                if eef_mat is not None:
                    eef_verts = apply_transform(eef_box_vertices_local, eef_mat)
                    eef_box_collection.set_verts([eef_verts[idx] for idx in BOX_FACES])
                    if eef_box_collection_zoom_primary is not None:
                        eef_box_collection_zoom_primary.set_verts(
                            [eef_verts[idx] for idx in BOX_FACES]
                        )
                    if eef_box_collection_zoom_secondary is not None:
                        eef_box_collection_zoom_secondary.set_verts(
                            [eef_verts[idx] for idx in BOX_FACES]
                        )
                else:
                    eef_box_collection.set_verts([])
                    if eef_box_collection_zoom_primary is not None:
                        eef_box_collection_zoom_primary.set_verts([])
                    if eef_box_collection_zoom_secondary is not None:
                        eef_box_collection_zoom_secondary.set_verts([])
            for link_name, coll in robot_link_collections.items():
                link_tf = frame.get("robot_link_poses", {}).get(link_name)
                zoom_coll_primary = robot_link_shapes[link_name]["zoom_primary_collection"]
                zoom_coll_secondary = robot_link_shapes[link_name]["zoom_secondary_collection"]
                if link_tf is None:
                    coll.set_verts([])
                    zoom_coll_primary.set_verts([])
                    zoom_coll_secondary.set_verts([])
                    continue
                link_verts = apply_transform(
                    robot_link_shapes[link_name]["local_vertices"], link_tf
                )
                coll.set_verts([link_verts[idx] for idx in BOX_FACES])
                zoom_coll_primary.set_verts([link_verts[idx] for idx in BOX_FACES])
                zoom_coll_secondary.set_verts([link_verts[idx] for idx in BOX_FACES])

            for shape in shapes:
                coll = shape["collection"]
                zoom_coll_primary = shape["zoom_primary_collection"]
                zoom_coll_secondary = shape["zoom_secondary_collection"]
                if shape["kind"] == "fixture":
                    verts = apply_transform(shape["local_vertices"], shape["world_transform"])
                else:
                    obj_state = frame["object_states"].get(shape["name"])
                    if obj_state is None:
                        coll.set_verts([])
                        zoom_coll_primary.set_verts([])
                        zoom_coll_secondary.set_verts([])
                        continue
                    verts = apply_transform(shape["local_vertices"], obj_state["mat"])
                    object_paths.setdefault(shape["name"], []).append(obj_state["pos"])
                    if shape["name"] not in object_path_lines:
                        (pline,) = ax_scene.plot(
                            [], [], [], color=shape["edge"], linewidth=2.0, alpha=0.95
                        )
                        object_path_lines[shape["name"]] = pline
                    update_path(object_path_lines[shape["name"]], object_paths[shape["name"]])
                coll.set_verts([verts[idx] for idx in BOX_FACES])
                zoom_coll_primary.set_verts([verts[idx] for idx in BOX_FACES])
                zoom_coll_secondary.set_verts([verts[idx] for idx in BOX_FACES])

            for name, obj_state in frame["object_states"].items():
                pos = obj_state.get("pos")
                if pos is None:
                    continue
                color = object_colors.get(name, (0.15, 0.6, 0.15))
                if name not in object_markers:
                    object_markers[name] = ax_scene.scatter(
                        [],
                        [],
                        [],
                        s=42,
                        c=[color],
                        marker="o",
                        edgecolors="white",
                        linewidths=1.0,
                        alpha=0.95,
                        depthshade=False,
                        zorder=6,
                    )
                    object_markers_zoom_primary[name] = ax_scene_zoom_primary.scatter(
                        [],
                        [],
                        [],
                        s=42,
                        c=[color],
                        marker="o",
                        edgecolors="white",
                        linewidths=1.0,
                        alpha=0.95,
                        depthshade=False,
                        zorder=6,
                    )
                    object_markers_zoom_secondary[name] = ax_scene_zoom_secondary.scatter(
                        [],
                        [],
                        [],
                        s=42,
                        c=[color],
                        marker="o",
                        edgecolors="white",
                        linewidths=1.0,
                        alpha=0.95,
                        depthshade=False,
                        zorder=6,
                    )
                    if name not in object_path_lines:
                        (pline,) = ax_scene.plot([], [], [], color=color, linewidth=2.0, alpha=0.95)
                        object_path_lines[name] = pline
                    if name not in object_path_lines_zoom_primary:
                        (pline_zoom_primary,) = ax_scene_zoom_primary.plot(
                            [], [], [], color=color, linewidth=2.0, alpha=0.95
                        )
                        object_path_lines_zoom_primary[name] = pline_zoom_primary
                    if name not in object_path_lines_zoom_secondary:
                        (pline_zoom_secondary,) = ax_scene_zoom_secondary.plot(
                            [], [], [], color=color, linewidth=2.0, alpha=0.95
                        )
                        object_path_lines_zoom_secondary[name] = pline_zoom_secondary
                object_markers[name]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
                object_markers_zoom_primary[name]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
                object_markers_zoom_secondary[name]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
                object_paths.setdefault(name, []).append(pos)
                update_path(object_path_lines[name], object_paths[name])
                update_path(object_path_lines_zoom_primary[name], object_paths[name])
                update_path(object_path_lines_zoom_secondary[name], object_paths[name])

            current_steps = steps[: frame_idx + 1]
            joint_cursor.set_xdata([step, step])
            speed_cursor.set_xdata([step, step])
            state_cursor.set_xdata([step, step])
            for joint_idx, line in enumerate(joint_lines):
                line.set_data(current_steps, joint_pos_history[: frame_idx + 1, joint_idx])
            root_speed_line.set_data(current_steps, root_speed[: frame_idx + 1])
            root_omega_line.set_data(current_steps, root_omega[: frame_idx + 1])
            eef_speed_line.set_data(current_steps, eef_speed[: frame_idx + 1])
            eef_omega_line.set_data(current_steps, eef_omega[: frame_idx + 1])
            max_robot_force_line.set_data(current_steps, max_robot_force[: frame_idx + 1])
            eef_force_line.set_data(current_steps, eef_force[: frame_idx + 1])
            robot_contact_line.set_data(current_steps, robot_contacts[: frame_idx + 1])

            joint_span = np.nan
            if frame["joint_positions"].size:
                joint_span = float(
                    np.nanmax(frame["joint_positions"]) - np.nanmin(frame["joint_positions"])
                )
            joint_vel_absmax = np.nan
            if frame["joint_velocities"].size:
                joint_vel_absmax = float(np.nanmax(np.abs(frame["joint_velocities"])))

            status_items = [
                f"step: {step}",
                f"success: {int(frame['success'])}",
                (
                    f"base_speed: {frame['root_linear_speed']:.3f}"
                    if np.isfinite(frame["root_linear_speed"])
                    else "base_speed: n/a"
                ),
                (
                    f"base_omega: {frame['root_angular_speed']:.3f}"
                    if np.isfinite(frame["root_angular_speed"])
                    else "base_omega: n/a"
                ),
                (
                    f"eef_speed: {frame['eef_linear_speed']:.3f}"
                    if np.isfinite(frame["eef_linear_speed"])
                    else "eef_speed: n/a"
                ),
                (
                    f"eef_omega: {frame['eef_angular_speed']:.3f}"
                    if np.isfinite(frame["eef_angular_speed"])
                    else "eef_omega: n/a"
                ),
                (
                    f"max_robot_force: {frame['max_robot_link_force']:.3f}"
                    if np.isfinite(frame["max_robot_link_force"])
                    else "max_robot_force: n/a"
                ),
                (
                    f"eef_force: {frame['end_effector_force_norm']:.3f}"
                    if np.isfinite(frame["end_effector_force_norm"])
                    else "eef_force: n/a"
                ),
                (
                    f"robot_contacts: {int(frame['robot_contact_count'])}"
                    if np.isfinite(frame["robot_contact_count"])
                    else "robot_contacts: n/a"
                ),
                (
                    f"total_contacts: {int(frame['total_contact_count'])}"
                    if np.isfinite(frame["total_contact_count"])
                    else "total_contacts: n/a"
                ),
                f"joint_span: {joint_span:.3f}" if np.isfinite(joint_span) else "joint_span: n/a",
                (
                    f"joint_vel|max|: {joint_vel_absmax:.3f}"
                    if np.isfinite(joint_vel_absmax)
                    else "joint_vel|max|: n/a"
                ),
                (
                    f"robot_forceΣ: {frame['robot_force_cumulative']:.3f}"
                    if np.isfinite(frame["robot_force_cumulative"])
                    else "robot_forceΣ: n/a"
                ),
                f"objects_tracked: {len(frame['object_states'])}",
                f"fixtures_tracked: {len(frame['fixture_states'])}",
            ]
            status_text.set_text("\n".join(status_items))

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            fig_rgb = rgba[..., :3].copy()

            top_frame = None
            if rollout_iter is not None:
                try:
                    top_frame = next(rollout_iter)
                except StopIteration:
                    top_frame = None

            target_width = fig_rgb.shape[1]
            if top_frame is not None:
                target_width = max(target_width, int(top_frame.shape[1]))
            max_output_width = max(320, int(args.max_width))
            if target_width > max_output_width:
                target_width = max_output_width
            fig_rgb = resize_frame(fig_rgb, target_width)
            if top_frame is None:
                if last_top_frame_shape is not None and last_top_frame_shape[1] == target_width:
                    top_frame = np.zeros(last_top_frame_shape, dtype=np.uint8)
                else:
                    top_frame = np.zeros(
                        (max(1, fig_rgb.shape[0] // 2), target_width, 3), dtype=np.uint8
                    )
            else:
                top_frame = resize_frame(top_frame, target_width)
                last_top_frame_shape = top_frame.shape

            composite = np.vstack([top_frame, fig_rgb])
            composite = pad_frame_to_even_dimensions(composite)
            writer.append_data(composite.copy())
            frames_written += 1
            if args.preview_frames > 0 and frames_written >= args.preview_frames:
                break
    finally:
        writer.close()
        plt.close(fig)
        print(f"Wrote: {args.output} (frames: {frames_written})")


if __name__ == "__main__":
    main()
