"""RoboCasa simulator-side predicates from ``robocasa/docs/new/4ltls.txt``."""

from __future__ import annotations

import os
import ast
import inspect
import textwrap
import xml.etree.ElementTree as ET
from typing import Any, Dict

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import find_elements

import robocasa.utils.object_utils as OU
from .attributes import (
    infer_object_attributes,
    infer_fixture_attributes,
    RECEPTACLE_CATEGORIES,
    FOOD_TYPE_NAMES,
)


GRIPPER_CLOSED_THRESHOLD = 0.035
GRIPPER_FAR_THRESHOLD = 0.10
OBJ_LINEAR_STABLE_THRESHOLD = 0.05
OBJ_ANGULAR_STABLE_THRESHOLD = 0.25
GRASP_BILATERAL_MIN_CONTACT_BODIES = 2
# How far the grip point (the material point of the object under the fingers)
# may migrate across the object, in metres, before the grasp counts as slipping.
# Scaled to the finger pad rather than picked: the Panda pad collision boxes are
# half-extents 0.008 x 0.004 x 0.008 (robosuite/models/assets/grippers/
# panda_gripper.xml:35,45), i.e. a 1.6 cm x 1.6 cm contact face, so migration
# past ~1.6 cm means the original grip point has left the pad entirely -- the
# "the spot I gripped is no longer under my finger" criterion.
GRASP_POINT_DRIFT_THRESHOLD = 0.016
# Consecutive RAW frames the grip-point drift must stay over the threshold
# before the grasp counts as slipping. See the debounce note at the
# grasp_point_stable computation for why this one exists when the item-9
# debounces were removed, and why it is intentionally not call_stride-scaled.
GRASP_POINT_DRIFT_PERSISTENCE_FRAMES = 2
STABLE_PERSISTENCE_FRAME = 2
CONTENT_STABLE_PERSISTENCE_FRAMES = 2
FIXTURE_OUTPUT_IDLE_FRAMES = 1
MICROWAVE_EMPTY_PERSISTENCE_FRAMES = 2
MICROWAVE_OCCUPANCY_PERSISTENCE_FRAMES = 2
FIXTURE_FULLY_OPEN_FRACTION = 0.90
SETTLE_TIMEOUT_FRAMES = 6
SKILL_ONSET_FRAMES = 2
PLACE_ONSET_FRAMES = 1
DUMP_ONSET_FRAMES = 1
GRASPED_RECEPTACLE_UPRIGHT_GRACE_FRAMES = 2
PICK_APPROACH_PERSISTENCE_FRAMES = 2
REACH_THRESHOLD = 0.05
TARGET_REGION_BLOCKED_THRESHOLD = 1
PLACEMENT_MARGIN = 0.03
PATH_OBSTRUCTION_OVERLAP_ALLOWANCE = 0.05
CLUTTER_THRESHOLD = 2
SUPPORT_CLUTTER_Z_TOLERANCE = 0.05
FIXTURE_FULLY_OPEN_THRESHOLD = 0.90
FIXTURE_FULLY_CLOSED_THRESHOLD = 0.05
FIXTURE_MOTION_DELTA_THRESHOLD = 1e-3
DEBUG_ENV_VAR = "ROBOCASA_PREDICATE_DEBUG"
DEBUG_EVERY_N_ENV_VAR = "ROBOCASA_PREDICATE_DEBUG_EVERY_N"

ACTION_COMPONENT_KEYWORDS = {
    "press": (
        "lever",
        "button",
        "press",
        "switch",
        "control",
        "start",
        "stop",
        "power",
        "cancel",
    ),
    "turn": ("faucet", "handle", "spout"),
    "slide": ("handle", "pull", "slide", "rack", "tray", "lever"),
    "twist": (
        "cap",
        "lid",
        "knob",
        "dial",
        "collar",
        "bottle",
        "jar",
        "can",
        "temperature",
        "temp",
        "timer",
        "time",
    ),
    "open_close": (
        "handle",
        "door",
        "lid",
        "head",
        "hinge",
        "drawer",
    ),
}
ACTION_ATTRIBUTE_BY_NAME = {
    "press": "pressable",
    "turn": "turnable",
    "slide": "slideable",
    "twist": "twistable",
    "open_close": "openable",
}


PREDICATE_FAMILIES = {
    "contact_policy": [
        "forbidden_contact",
        "allowed_contact",
        "robot_correct_manipulated_object_contact",
        "robot_correct_fixture_contact",
        "correct_manipulated_object_correct_fixture_contact",
        "correct_manipulated_object_correct_receive_object_contact",
        "grasped_object_exists",
    ],
    "grasp_release_settle": [
        "object_grasped",
        "object_stable",
        "object_sync",
        "grasp_point_stable",
        "object_upright",
        "object_grasped_safe",
        "object_released",
        "object_supported",
        "object_supported_on_correct",
        "gripper_away_from_object",
        "object_settled",
        "release_object_settle_timeout",
        "object_settle_timeout",
        "gripper_is_opening",
    ],
    "contamination": [
        "sanitized",
        "robot_contact_raw_contaminated",
        "object_is_rte",
        "robot_contact_clean",
    ],
    "skill_onset": [
        "gripper_is_closing",
        "gripper_moving_towards_object",
        "gripper_near_object",
        "skill_pick_onset",
        "skill_place_onset",
        "gripper_moving_towards_target",
        "gripper_near_target",
        "skill_press_onset",
        "skill_turn_onset",
        "skill_slide_onset",
        "skill_twist_onset",
        "skill_open_close_onset",
        "skill_dump_onset",
    ],
    "pick_preconditions": [
        "object_region_clear",
        "object_upright_if_receptacle",
        "preconditions_satisfied_pick",
    ],
    "place_preconditions": [
        "support_region_clear",
        "support_stable",
        "support_geometry_valid",
        "support_type_matches_object",
        "dump_support_geometry_valid",
        "dump_support_type_matches_content",
        "dump_support_hygienic_for_content",
        "dump_support_objects_clean_for_content",
        "dump_support_not_cluttered_for_fragile_content",
        "support_hygienic_for_manipulated_object",
        "support_objects_clean_for_manipulated_object",
        "support_not_cluttered_for_fragile_manipulated_object",
        "preconditions_satisfied_place",
    ],
    "other_intended_safety_preconditions": [
        "target_region_clear",
        "target_stable",
        "fixture_ready_for_press",
        "fixture_ready_for_turn",
        "fixture_ready_for_slide",
        "fixture_ready_for_twist",
        "fixture_ready_for_open_close",
        "slide_path_clear",
        "target_receptacle_upright_if_has_contents",
        "articulation_path_clear",
        "preconditions_satisfied_press",
        "preconditions_satisfied_turn",
        "preconditions_satisfied_slide",
        "preconditions_satisfied_twist",
        "preconditions_satisfied_open_close",
        "preconditions_satisfied_dump",
    ],
    "mechanism_safety": [
        "robot_fixture_contact",
        "fixture_is_opening",
        "fixture_is_closing",
        "fixture_fully_open",
        "fixture_fully_closed",
        "fixture_obstacle_contact",
        "continue_fixture_open",
        "continue_fixture_close",
        "fixture_open_retract_path_clear",
        "fixture_close_retract_path_clear",
        "fixture_open_obstacle_hit",
        "fixture_close_obstacle_hit",
        "fixture_open_retracting",
        "fixture_close_retracting",
    ],
    "containment_safety": [
        "containment_transfer_event",
        "fixture_output_started",
        "fixture_output_stopped",
        "fixture_content_output_started",
        "liquid_transfer_event",
        "solid_transfer_event",
        "liquid_settled",
        "solid_settled",
        "solid_misplacement",
        "misplaced_solid_removed",
        "misplaced_solid_recollected",
        "content_settled",
        "content_is_supported",
        "content_stable",
        "support_type_matches_content",
        "content_is_liquid",
        "content_is_solid",
    ],
    "access_enclosure_safety": [
        "one_object_in_microwave",
        "two_or_more_objects_in_microwave",
        "microwave_empty",
        "reach_in_fixture",
        "gripper_in_fixture",
        "object_reach_in_fixture",
        "object_in_fixture",
        "object_in_same_fixture",
    ],
}


def build_predicate_static_spec(
    env: Any, static_info: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "task_name": env.__class__.__name__,
        "task_language": ((static_info or {}).get("task", {}) or {}).get(
            "language", ""
        ),
        "predicate_groups": PREDICATE_FAMILIES,
        "role_definitions": {
            "manipulated_objects": "Objects the robot may directly move: currently grasped objects, objects marked graspable in task configs, or non-receive fallback task objects.",
            "active_object": "The stage-local object currently used for object predicates: the smoothed grasped object when grasped, otherwise the most recent robot-contacted manipulated object.",
            "target_fixtures": "Fixtures referenced by task fixture refs or object placement configs.",
            "receive_objects": "Object receptacles/supports referenced by placement targets or inferred from receptacle-like object configs.",
            "source_supports_by_object": "Supports each manipulated object was contacting when grasp began; used only by forbidden-contact policy, not settling.",
        },
    }


def build_predicate_snapshot(
    env: Any, static_info: Dict[str, Any], dynamic_info: Dict[str, Any]
) -> Dict[str, Any]:
    def _bool(value: Any) -> bool:
        try:
            return bool(value)
        except Exception:
            return False

    def _norm(value: Any) -> float:
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return float("nan")
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return float("nan")
        return float(np.linalg.norm(arr))

    def _entry(name: str, value: bool) -> Dict[str, Any]:
        return {
            "name": name,
            "category": "predicate",
            "value": _bool(value),
            "language": f"Predicate `{name}`.",
            "readout": f"{name}: {_bool(value)}",
        }

    def _debug_enabled() -> bool:
        return str(os.environ.get(DEBUG_ENV_VAR, "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _debug_every_n() -> int:
        try:
            return max(1, int(os.environ.get(DEBUG_EVERY_N_ENV_VAR, "1")))
        except Exception:
            return 1

    def _debug_print(message: str) -> None:
        if _debug_enabled():
            print(f"[robocasa-predicates] {message}", flush=True)

    def _object_position(name: str = "obj") -> np.ndarray | None:
        pose = ((dynamic_info.get("scene") or {}).get("objects") or {}).get(
            name, {}
        ).get("pose") or {}
        try:
            pos = np.asarray(pose.get("position"), dtype=float).reshape(-1)
        except Exception:
            return None
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return None
        return pos[:3]

    def _eef_position() -> np.ndarray | None:
        pose = (dynamic_info.get("robot") or {}).get("end_effector_pose") or {}
        try:
            pos = np.asarray(pose.get("position"), dtype=float).reshape(-1)
        except Exception:
            return None
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return None
        return pos[:3]

    def _robot_geom_ids() -> set[int]:
        body_ids = set()
        link_poses = (dynamic_info.get("robot") or {}).get("link_poses") or {}
        for body_name in link_poses.keys() if isinstance(link_poses, dict) else []:
            try:
                body_ids.add(int(env.sim.model.body_name2id(str(body_name))))
            except Exception:
                continue
        if body_ids:
            try:
                return {
                    int(geom_id)
                    for geom_id, geom_body_id in enumerate(env.sim.model.geom_bodyid)
                    if int(geom_body_id) in body_ids
                }
            except Exception:
                pass

        geom_ids = set()
        try:
            robot_geoms = find_elements(
                root=env.robots[0].robot_model.root, tags="geom", return_first=False
            )
        except Exception:
            robot_geoms = []
        for geom in robot_geoms or []:
            try:
                geom_ids.add(int(env.sim.model.geom_name2id(geom.get("name"))))
            except Exception:
                continue
        return geom_ids, actions_by_fixture

    def _robot_base_geom_ids() -> set[int]:
        base_geom_ids = set()
        try:
            nbody = int(env.sim.model.nbody)
        except Exception:
            nbody = 0
        base_body_ids = set()
        for body_id in range(nbody):
            try:
                body_name = str(env.sim.model.body_id2name(body_id) or "")
            except Exception:
                continue
            lowered = body_name.lower()
            if (
                "mobilebase" in lowered
                or ("robot" in lowered and "base" in lowered)
                or "pedestal" in lowered
            ):
                base_body_ids.add(body_id)
        try:
            for geom_id, geom_body_id in enumerate(env.sim.model.geom_bodyid):
                geom_body_id = int(geom_body_id)
                if geom_body_id in base_body_ids:
                    base_geom_ids.add(int(geom_id))
                    continue
                try:
                    geom_name = str(env.sim.model.geom_id2name(int(geom_id)) or "")
                except Exception:
                    geom_name = ""
                lowered = geom_name.lower()
                if (
                    "mobilebase" in lowered
                    or "pedestal" in lowered
                    or ("robot" in lowered and "base" in lowered)
                ):
                    base_geom_ids.add(int(geom_id))
        except Exception:
            pass
        return base_geom_ids

    def _body_and_descendant_ids(body_id: int | None) -> set[int]:
        body_ids = set()
        if body_id is None:
            return body_ids
        try:
            body_ids.add(int(body_id))
            changed = True
            while changed:
                changed = False
                for child_id, parent_id in enumerate(env.sim.model.body_parentid):
                    if int(parent_id) in body_ids and int(child_id) not in body_ids:
                        body_ids.add(int(child_id))
                        changed = True
        except Exception:
            return set()
        return body_ids

    def _body_geom_ids(body_id: int | None) -> set[int]:
        body_ids = _body_and_descendant_ids(body_id)
        if not body_ids:
            return set()
        try:
            return {
                int(geom_id)
                for geom_id, geom_body_id in enumerate(env.sim.model.geom_bodyid)
                if int(geom_body_id) in body_ids
            }
        except Exception:
            return set()

    def _object_geom_ids(name: str) -> set[int]:
        geom_ids = set()
        try:
            geom_ids.update(_body_geom_ids(env.obj_body_id[name]))
        except Exception:
            pass
        geom_ids.update(_object_contact_geom_ids(name))
        return geom_ids

    def _geom_ids_from_names(geom_names) -> set[int]:
        geom_ids = set()
        if isinstance(geom_names, str):
            geom_names = [geom_names]
        for geom_name in geom_names or []:
            try:
                geom_ids.add(int(env.sim.model.geom_name2id(str(geom_name))))
            except Exception:
                continue
        return geom_ids

    def _object_contact_geom_ids(name: str) -> set[int]:
        try:
            return _geom_ids_from_names(env.objects[str(name)].contact_geoms)
        except Exception:
            return set()

    def _gripper_contact_geom_ids() -> set[int]:
        geom_ids = set()
        try:
            grippers = getattr(env.robots[0], "gripper", {}) or {}
        except Exception:
            grippers = {}
        if isinstance(grippers, dict):
            iterable = grippers.values()
        else:
            iterable = [grippers]
        for gripper in iterable:
            try:
                geom_ids.update(_geom_ids_from_names(gripper.contact_geoms))
            except Exception:
                continue
        return geom_ids

    def _gripper_finger_body_contact_map() -> dict[int, set[int]]:
        """Group gripper contact geoms by their parent body id.

        A parallel-jaw gripper's two finger pads live on two distinct MuJoCo
        bodies. Grouping by body (rather than treating the whole gripper as
        one geom set) lets grasp detection require *bilateral* contact
        instead of "any gripper geom touches the object", which is what let
        transient single-pad contact during the approach/close phase get
        misdetected as a full grasp.
        """
        geom_ids = _gripper_contact_geom_ids()
        body_map: dict[int, set[int]] = {}
        for geom_id in geom_ids:
            try:
                body_id = int(env.sim.model.geom_bodyid[geom_id])
            except Exception:
                continue
            body_map.setdefault(body_id, set()).add(geom_id)
        return body_map

    def _object_gripper_contact_any(name: str) -> bool:
        """Fallback aggregate contact check (any gripper geom vs. object geom)."""
        object_geom_ids = _object_geom_ids(name)
        gripper_geom_ids = _gripper_contact_geom_ids()
        if not object_geom_ids or not gripper_geom_ids:
            return False
        contact_number = int(getattr(env.sim.data, "ncon", 0))
        for contact_idx in range(contact_number):
            try:
                geom1 = int(env.sim.data.contact[contact_idx].geom1)
                geom2 = int(env.sim.data.contact[contact_idx].geom2)
            except Exception:
                continue
            if (geom1 in gripper_geom_ids and geom2 in object_geom_ids) or (
                geom2 in gripper_geom_ids and geom1 in object_geom_ids
            ):
                return True
        return False

    def _object_gripper_bilateral_contact(name: str) -> bool:
        """Require contact from at least GRASP_BILATERAL_MIN_CONTACT_BODIES
        distinct gripper finger bodies simultaneously (an antipodal-contact
        precondition), instead of aggregate any-geom contact.

        Falls back to the aggregate any-geom check if the gripper's contact
        geoms can't be split into >=2 distinct bodies (e.g. a non
        two-finger / suction end-effector), so non-parallel-jaw grippers are
        unaffected.
        """
        object_geom_ids = _object_geom_ids(name)
        if not object_geom_ids:
            return False
        finger_body_map = _gripper_finger_body_contact_map()
        if len(finger_body_map) < 2:
            return _object_gripper_contact_any(name)
        contacted_bodies = set()
        contact_number = int(getattr(env.sim.data, "ncon", 0))
        for contact_idx in range(contact_number):
            try:
                geom1 = int(env.sim.data.contact[contact_idx].geom1)
                geom2 = int(env.sim.data.contact[contact_idx].geom2)
            except Exception:
                continue
            for body_id, geom_ids in finger_body_map.items():
                if body_id in contacted_bodies:
                    continue
                if (geom1 in geom_ids and geom2 in object_geom_ids) or (
                    geom2 in geom_ids and geom1 in object_geom_ids
                ):
                    contacted_bodies.add(body_id)
        return _bool(len(contacted_bodies) >= max(1, int(GRASP_BILATERAL_MIN_CONTACT_BODIES)))

    def _object_grip_point_in_object_frame(name: str):
        """Centroid of the finger/object contact points, expressed in the
        *object's* own body frame.

        This is the material point of the object currently under the fingers.
        While the grasp sticks, it stays fixed even as the hand (and the
        object with it) translates and rotates arbitrarily -- which is
        precisely what distinguishes a grasp that holds from one the object
        is sliding through. Contrast `_object_sync`, which compares the
        object's *body-origin* velocity against the *eef site's*: for an
        object whose centre is offset from the grasp by r, any wrist rotation
        w contributes |w x r| of relative velocity with no slip whatsoever
        (measured at ~0.09 m/s for a basket held 0.18 m from the eef site,
        against a 0.05 m/s threshold), so that check fires on ordinary
        carrying motion.

        Returns None when there is no finger/object contact this frame (the
        premise of the measure is absent, so callers must not read a stale
        value as agreement).
        """
        object_geom_ids = _object_geom_ids(name)
        if not object_geom_ids:
            return None
        gripper_geom_ids = _gripper_contact_geom_ids()
        if not gripper_geom_ids:
            return None
        points = []
        contact_number = int(getattr(env.sim.data, "ncon", 0))
        for contact_idx in range(contact_number):
            try:
                contact = env.sim.data.contact[contact_idx]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
            except Exception:
                continue
            if not (
                (geom1 in gripper_geom_ids and geom2 in object_geom_ids)
                or (geom2 in gripper_geom_ids and geom1 in object_geom_ids)
            ):
                continue
            try:
                points.append(np.asarray(contact.pos, dtype=float).reshape(3))
            except Exception:
                continue
        if not points:
            return None
        world_point = np.mean(np.asarray(points, dtype=float), axis=0)
        try:
            body_id = env.obj_body_id[str(name)]
            obj_pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
            # body_xquat is wxyz; T.quat2mat wants xyzw
            obj_mat = T.quat2mat(
                T.convert_quat(
                    np.asarray(env.sim.data.body_xquat[body_id], dtype=float), to="xyzw"
                )
            )
        except Exception:
            return None
        if not np.all(np.isfinite(world_point)) or not np.all(np.isfinite(obj_pos)):
            return None
        return obj_mat.T @ (world_point - obj_pos)

    def _object_non_gripper_contact(name: str, ignore_names=()) -> bool:
        """Whether anything other than the gripper is touching `name`.

        Used to suspend grasp-point-drift judgement: when the object lands on
        the surface it is being placed on, that surface pushes it and can shift
        it in the fingers. That displacement is not caused by the grasp, so it
        must not be scored as an unsafe grasp.

        Counts any contact between an object geom and a geom that is neither a
        gripper contact geom nor another geom of the same object, so both
        fixtures (counters, shelves) and other objects qualify.

        `ignore_names` must list objects that are riding along *with* `name` --
        the contents of a grasped receptacle above all. Those are in permanent
        contact with it by definition, so counting them would suspend the
        criterion for the entire carry and silently disable it: measured on
        ArrangeBreadBasket ep 0, the bread inside the basket kept this true on
        228/228 grasped frames.
        """
        object_geom_ids = _object_geom_ids(name)
        if not object_geom_ids:
            return False
        gripper_geom_ids = _gripper_contact_geom_ids()
        ignored_geom_ids = set()
        for ignored in ignore_names or ():
            if str(ignored) == str(name):
                continue
            ignored_geom_ids |= set(_object_geom_ids(str(ignored)) or ())
        contact_number = int(getattr(env.sim.data, "ncon", 0))
        for contact_idx in range(contact_number):
            try:
                contact = env.sim.data.contact[contact_idx]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
            except Exception:
                continue
            if geom1 in object_geom_ids:
                other = geom2
            elif geom2 in object_geom_ids:
                other = geom1
            else:
                continue
            if (
                other in object_geom_ids
                or other in gripper_geom_ids
                or other in ignored_geom_ids
            ):
                continue
            return True
        return False

    def _object_is_grasped(name: str) -> bool:
        try:
            # check_obj_grasped already ANDs (aggregate contact) with (gripper
            # closed). bilateral_contact implies aggregate contact, so ANDing
            # it in here tightens the raw contact requirement to "== 2 finger
            # bodies", not just "some gripper geom", without needing to
            # duplicate the closed-gripper joint-position check locally.
            #
            # Deliberately does NOT also require object_sync: a bilateral
            # (both-finger) contact with the gripper closed is treated as a
            # grasp on its own. object_sync is reserved for object_grasped_safe
            # (checking the grasp *stays* good), not for the raw grasp signal
            # itself -- keeping them independent instead of object_grasped
            # already implying object_sync, which would make object_grasped_safe
            # a tautology (= object_grasped, giving zero extra information).
            return _bool(
                _object_gripper_bilateral_contact(name)
                and OU.check_obj_grasped(
                    env, obj_name=name, threshold=GRIPPER_CLOSED_THRESHOLD
                )
            )
        except Exception:
            return False

    def _object_is_upright(name: str) -> bool:
        try:
            return _bool(OU.check_obj_upright(env, obj_name=name))
        except Exception:
            return False

    def _gripper_far_from_object(name: str) -> bool:
        try:
            return _bool(
                OU.gripper_obj_far(env, obj_name=name, th=GRIPPER_FAR_THRESHOLD)
            )
        except Exception:
            return False

    def _gripper_is_opening() -> bool:
        robot_info = dynamic_info.get("robot") or {}
        joint_names = robot_info.get("joint_names") or []
        joint_velocities = robot_info.get("joint_velocities") or []
        outward_velocities = []
        if isinstance(joint_names, list) and isinstance(joint_velocities, list):
            for name, velocity in zip(joint_names, joint_velocities):
                lowered = str(name).lower()
                if "gripper" not in lowered and "finger" not in lowered:
                    continue
                try:
                    value = float(velocity)
                except Exception:
                    continue
                if "joint1" in lowered:
                    outward_velocities.append(value)
                elif "joint2" in lowered:
                    outward_velocities.append(-value)
                else:
                    outward_velocities.append(value)
        return _bool(outward_velocities and np.mean(outward_velocities) > 1e-4)

    def _object_configs() -> dict[str, dict]:
        objects = ((static_info or {}).get("scene_layout") or {}).get("objects") or {}
        configs = {}
        for name, info in objects.items():
            cfg = (info or {}).get("config") or {}
            configs[str(name)] = cfg if isinstance(cfg, dict) else {}
        return configs

    def _fixture_name_from_ref(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        name = getattr(value, "name", None)
        if name:
            return str(name)
        return None

    def _fixture_name_from_instance(value: Any) -> str | None:
        if value is None:
            return None
        for fixture_name, fixture in getattr(env, "fixtures", {}).items():
            if value is fixture:
                return str(fixture_name)
        return _fixture_name_from_ref(value)

    def _self_attr_name(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return str(node.attr)
        return None

    def _literal_string_set(node: ast.AST, bindings: dict[str, set[str]]) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return set(bindings.get(node.id, set()))
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = set()
            for elt in node.elts:
                values.update(_literal_string_set(elt, bindings))
            return values
        return set()

    def _fixture_name_from_ast(node: ast.AST) -> str | None:
        attr_name = _self_attr_name(node)
        if attr_name:
            return _fixture_name_from_instance(getattr(env, attr_name, None))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value if node.value in getattr(env, "fixtures", {}) else None
        return None

    def _success_target_relations() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        """Read object-target relations from the task's own success predicate."""
        object_names = {str(name) for name in getattr(env, "objects", {}).keys()}
        fixture_names = {str(name) for name in getattr(env, "fixtures", {}).keys()}
        object_targets = {name: set() for name in object_names}
        fixture_targets = {name: set() for name in object_names}
        method = getattr(env.__class__, "_check_success", None)
        if method is None:
            return object_targets, fixture_targets
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        except Exception:
            return object_targets, fixture_targets

        def add_object_targets(obj_expr: ast.AST, target_expr: ast.AST, bindings):
            objs = _literal_string_set(obj_expr, bindings).intersection(object_names)
            targets = _literal_string_set(target_expr, bindings).intersection(
                object_names
            )
            for obj in objs:
                object_targets[obj].update(targets)

        def add_fixture_target(obj_expr: ast.AST, fixture_expr: ast.AST, bindings):
            objs = _literal_string_set(obj_expr, bindings).intersection(object_names)
            fixture_name = _fixture_name_from_ast(fixture_expr)
            if fixture_name not in fixture_names:
                return
            for obj in objs:
                fixture_targets[obj].add(fixture_name)

        def visit(node: ast.AST, bindings: dict[str, set[str]]):
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                values = _literal_string_set(node.iter, bindings)
                next_bindings = dict(bindings)
                if values:
                    next_bindings[node.target.id] = values
                for child in node.body:
                    visit(child, next_bindings)
                for child in node.orelse:
                    visit(child, bindings)
                return
            if isinstance(node, ast.Call):
                func = node.func
                func_name = getattr(func, "attr", None) or getattr(func, "id", None)
                if func_name == "check_obj_in_receptacle" and len(node.args) >= 3:
                    add_object_targets(node.args[1], node.args[2], bindings)
                elif (
                    func_name in {"obj_inside_of", "check_obj_fixture_contact"}
                    and len(node.args) >= 3
                ):
                    add_fixture_target(node.args[1], node.args[2], bindings)
                elif func_name == "check_rack_contact" and len(node.args) >= 2:
                    fixture_expr = (
                        func.value if isinstance(func, ast.Attribute) else None
                    )
                    if fixture_expr is not None:
                        add_fixture_target(node.args[1], fixture_expr, bindings)
            for child in ast.iter_child_nodes(node):
                visit(child, bindings)

        visit(tree, {})
        return object_targets, fixture_targets

    def _receive_object_names_from_configs(configs: dict[str, dict]) -> set[str]:
        receive_names = set()
        object_names = set(getattr(env, "objects", {}).keys())
        for name, cfg in configs.items():
            placement = cfg.get("placement") or {}
            if not isinstance(placement, dict):
                placement = {}
            for key in ("object", "try_to_place_in"):
                target = placement.get(key)
                if isinstance(target, str) and target in object_names:
                    receive_names.add(target)
        success_object_targets, _ = _success_target_relations()
        for targets in success_object_targets.values():
            receive_names.update(targets)
        return receive_names

    def _manipulated_object_names(
        configs: dict[str, dict], receive_names: set[str]
    ) -> set[str]:
        """Task-level objects the robot may directly move.

        Include success-condition objects plus graspable task objects that are
        themselves containers/cookware. Some composite tasks, e.g. PanTransfer,
        manipulate a pan whose success is expressed through its contents.
        """
        object_names = {str(name) for name in getattr(env, "objects", {}).keys()}
        success_object_targets, success_fixture_targets = _success_target_relations()
        names = {
            str(name)
            for name in object_names
            if success_object_targets.get(str(name))
            or success_fixture_targets.get(str(name))
        }
        container_groups = {
            "receptacle",
            "cookware",
            "pan",
            "pot",
            "pots_and_pans",
            "container",
            "tray",
        }
        for name, cfg in configs.items():
            name = str(name)
            if name not in object_names or cfg.get("graspable") is False:
                continue
            info = cfg.get("info") if isinstance(cfg.get("info"), dict) else {}
            groups = set()
            for key in ("groups", "groups_containing_sampled_obj"):
                values = info.get(key) or []
                if isinstance(values, str):
                    groups.add(values)
                else:
                    groups.update(str(value) for value in values)
            obj_groups = cfg.get("obj_groups")
            if isinstance(obj_groups, str):
                groups.add(obj_groups)
            elif isinstance(obj_groups, (list, tuple, set)):
                groups.update(str(value) for value in obj_groups)
            cat = str(info.get("cat") or cfg.get("category") or "").lower()
            if cat:
                groups.add(cat)
            if {group.lower() for group in groups} & container_groups:
                names.add(name)
        return names

    def _active_manipulated_object_name(manipulated_names: set[str]) -> str:
        for name in sorted(manipulated_names):
            if _object_is_grasped(name):
                return name
        if "obj" in manipulated_names:
            return "obj"
        return sorted(manipulated_names)[0] if manipulated_names else "obj"

    def _task_ref_names() -> set[str]:
        refs = set()
        try:
            refs.update(
                str(v)
                for v in (
                    (
                        ((static_info or {}).get("scene_layout") or {}).get(
                            "fixture_refs"
                        )
                        or {}
                    ).values()
                )
                if v is not None
            )
        except Exception:
            pass
        try:
            episode_refs = (
                ((static_info or {}).get("task") or {}).get("episode_meta") or {}
            ).get("refs") or {}
            refs.update(
                str(v) for v in episode_refs.values() if isinstance(v, str) and v
            )
        except Exception:
            pass
        try:
            for fixture_name, fixture in getattr(env, "fixtures", {}).items():
                if any(value is fixture for value in vars(env).values()):
                    refs.add(str(fixture_name))
        except Exception:
            pass
        try:
            init_ref_name = _fixture_name_from_instance(
                getattr(env, "init_robot_base_ref", None)
            )
            if init_ref_name is not None:
                refs.add(str(init_ref_name))
        except Exception:
            pass
        return refs

    def _target_fixture_names() -> set[str]:
        _, success_fixture_targets = _success_target_relations()
        names = set()
        for targets in success_fixture_targets.values():
            names.update(targets)
        return names

    def _target_object_names(
        active_obj_names: set[str], receive_names: set[str]
    ) -> set[str]:
        refs = _task_ref_names()
        names = {
            str(name)
            for name in getattr(env, "objects", {}).keys()
            if str(name) not in active_obj_names
            and (str(name) in refs or any(ref in str(name) for ref in refs))
        }
        names.update(
            str(name) for name in receive_names if str(name) not in active_obj_names
        )
        return names

    def _target_objects_by_manipulated(
        manipulated_names: set[str],
        target_object_names: set[str],
        configs: dict[str, dict],
    ) -> dict[str, set[str]]:
        mapping = {}
        object_names = set(getattr(env, "objects", {}).keys())
        success_object_targets, _ = _success_target_relations()
        for name in manipulated_names:
            placement = (configs.get(str(name)) or {}).get("placement") or {}
            matched = set()
            if isinstance(placement, dict):
                # These fields mirror RoboCasa task configs: if an object names a
                # placement target, keep that correspondence. Otherwise fall back
                # to the task-level receive set, e.g. both tupperwares in
                # PackIdenticalLunches.
                for key in ("object", "try_to_place_in"):
                    target = placement.get(key)
                    if isinstance(target, str) and target in object_names:
                        matched.add(target)
            matched.update(success_object_targets.get(str(name), set()))
            mapping[str(name)] = matched or set(target_object_names)
        return mapping

    def _target_fixtures_by_manipulated(
        manipulated_names: set[str],
        target_fixture_names: set[str],
        configs: dict[str, dict],
    ) -> dict[str, set[str]]:
        mapping = {}
        _, success_fixture_targets = _success_target_relations()
        for name in manipulated_names:
            matched = set(success_fixture_targets.get(str(name), set()))
            mapping[str(name)] = matched or set(target_fixture_names)
        return mapping

    def _fixture_geom_ids_by_names(fixture_names: set[str]) -> set[int]:
        geom_ids = set()
        for fixture_name, fixture in getattr(env, "fixtures", {}).items():
            if str(fixture_name) not in fixture_names:
                continue
            prefix = str(getattr(fixture, "naming_prefix", "") or "")
            worldbody = getattr(fixture, "worldbody", None)
            if worldbody is None:
                continue
            try:
                geom_elems = list(worldbody.iter("geom"))
            except Exception:
                geom_elems = []
            for geom in geom_elems:
                if not isinstance(geom, ET.Element):
                    continue
                geom_name = geom.get("name")
                if not geom_name:
                    continue
                full_name = (
                    f"{prefix}{geom_name}"
                    if prefix and not str(geom_name).startswith(prefix)
                    else str(geom_name)
                )
                try:
                    geom_ids.add(int(env.sim.model.geom_name2id(full_name)))
                except Exception:
                    continue
        return geom_ids

    def _contact_policy_fixture_attrs(fname: str) -> set[str]:
        fixture = getattr(env, "fixtures", {}).get(str(fname))
        class_name = fixture.__class__.__name__ if fixture is not None else ""
        attrs = set(infer_fixture_attributes(class_name))
        lowered = f"{fname} {class_name}".lower()
        if "drawer" in lowered:
            attrs.update({"openable", "closeable"})
        if (
            "button" in lowered
            or "coffee" in lowered
            or "microwave" in lowered
            or "kettle" in lowered
        ):
            attrs.update({"pressable"})
        if "faucet" in lowered or "sink" in lowered:
            attrs.update({"turnable"})
        if (
            "knob" in lowered
            or "dial" in lowered
            or "stove" in lowered
            or "oven" in lowered
            or "toaster" in lowered
        ):
            attrs.update({"twistable"})
        if "slide" in lowered or "rack" in lowered or "dishwasher" in lowered:
            attrs.update({"slideable"})
        if (
            "door" in lowered
            or "cabinet" in lowered
            or "fridge" in lowered
            or "lid" in lowered
            or "standmixer" in lowered
            or "stand_mixer" in lowered
        ):
            attrs.update({"openable", "closeable"})
        return attrs

    def _contact_policy_fixture_actions(fname: str) -> set[str]:
        attrs = _contact_policy_fixture_attrs(fname)
        return {
            action
            for action, attr in ACTION_ATTRIBUTE_BY_NAME.items()
            if attr in attrs or (action == "open_close" and "closeable" in attrs)
        }

    def _fixture_action_component_geom_ids_by_names(
        fixture_names: set[str],
    ) -> tuple[set[int], dict[str, list[str]]]:
        """Fixture component geoms the robot is allowed to touch for actions."""
        geom_ids = set()
        actions_by_fixture = {}
        for fixture_name in fixture_names:
            fixture_name = str(fixture_name)
            fixture = getattr(env, "fixtures", {}).get(fixture_name)
            if fixture is None:
                continue
            actions = _contact_policy_fixture_actions(fixture_name)
            actions_by_fixture[fixture_name] = sorted(actions)
            keywords = tuple(
                sorted(
                    {
                        keyword
                        for action in actions
                        for keyword in ACTION_COMPONENT_KEYWORDS.get(action, ())
                    }
                )
            )
            if not keywords:
                continue
            fixture_geom_ids = _fixture_geom_ids_by_names({fixture_name})
            for geom_id in fixture_geom_ids:
                try:
                    geom_name = str(env.sim.model.geom_id2name(int(geom_id)) or "")
                except Exception:
                    geom_name = ""
                if any(keyword in geom_name.lower() for keyword in keywords):
                    geom_ids.add(int(geom_id))

            prefix = str(getattr(fixture, "naming_prefix", "") or "")
            worldbody = getattr(fixture, "worldbody", None)
            if worldbody is None:
                continue
            try:
                body_elems = list(worldbody.iter("body"))
            except Exception:
                body_elems = []
            selected_body_ids = set()
            for body in body_elems:
                if not isinstance(body, ET.Element):
                    continue
                body_name = str(body.get("name") or "")
                if not body_name:
                    continue
                full_name = (
                    f"{prefix}{body_name}"
                    if prefix and not body_name.startswith(prefix)
                    else body_name
                )
                match_text = f"{body_name} {full_name}".lower()
                if not any(keyword in match_text for keyword in keywords):
                    continue
                try:
                    selected_body_ids.add(int(env.sim.model.body_name2id(full_name)))
                except Exception:
                    continue
            if selected_body_ids:
                geom_ids.update(
                    int(geom_id)
                    for geom_id, body_id in enumerate(env.sim.model.geom_bodyid)
                    if int(body_id) in selected_body_ids
                    and int(geom_id) in fixture_geom_ids
                )
        return geom_ids, actions_by_fixture

    def _fixture_geom_ids_by_name_map(
        fixture_names: set[str],
    ) -> dict[str, set[int]]:
        return {
            str(fixture_name): _fixture_geom_ids_by_names({str(fixture_name)})
            for fixture_name in fixture_names
        }

    def _matched_fixture_name_for_pair(
        geom1: int,
        geom2: int,
        object_geom_ids: set[int],
        fixture_geom_ids_by_name: dict[str, set[int]],
    ) -> str | None:
        for fixture_name, fixture_geom_ids in fixture_geom_ids_by_name.items():
            if _pair_matches(geom1, geom2, object_geom_ids, fixture_geom_ids):
                return fixture_name
        return None

    def _fixture_by_name(fname: str | None):
        return (
            getattr(env, "fixtures", {}).get(str(fname)) if fname is not None else None
        )

    def _fixture_is_floor(fname: str | None) -> bool:
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return False
        return "floor" in f"{fname} {fixture.__class__.__name__}".lower()

    def _fixture_rack_contact(fname: str | None, oname: str) -> bool:
        fixture = _fixture_by_name(fname)
        method = getattr(fixture, "check_rack_contact", None)
        if method is None:
            return False
        rack_level = getattr(env, "rack_level", None)
        if rack_level is not None:
            try:
                if _bool(method(env, str(oname), rack_level=rack_level)):
                    return True
            except TypeError:
                pass
            except Exception:
                return False
        try:
            return _bool(method(env, str(oname)))
        except TypeError:
            try:
                return _bool(method(str(oname)))
            except Exception:
                return False
        except Exception:
            return False

    def _pair_matches(geom1: int, geom2: int, set_a: set[int], set_b: set[int]) -> bool:
        return (geom1 in set_a and geom2 in set_b) or (
            geom2 in set_a and geom1 in set_b
        )

    def _canonical_contact_pair(geom1: int, geom2: int) -> tuple[int, int]:
        return tuple(sorted((int(geom1), int(geom2))))

    def _contact_pair_names(pair: tuple[int, int]) -> list[str]:
        names = []
        for geom_id in pair:
            try:
                names.append(str(env.sim.model.geom_id2name(int(geom_id))))
            except Exception:
                names.append(str(geom_id))
        return names

    def _geom_name(geom_id: int) -> str:
        try:
            return str(env.sim.model.geom_id2name(int(geom_id)) or geom_id)
        except Exception:
            return str(geom_id)

    def _object_supported(name: str = "obj") -> bool:
        for fixture_name in getattr(env, "fixtures", {}).keys():
            try:
                if OU.check_obj_fixture_contact(env, name, fixture_name):
                    return True
            except Exception:
                pass
            try:
                if OU.obj_inside_of(env, name, fixture_name, partial_check=True):
                    return True
            except Exception:
                pass
            if _fixture_rack_contact(str(fixture_name), str(name)):
                return True
        for receptacle_name in getattr(env, "objects", {}).keys():
            receptacle_name = str(receptacle_name)
            if receptacle_name == str(name):
                continue
            try:
                if OU.check_obj_in_receptacle(env, name, receptacle_name):
                    return True
            except Exception:
                continue
        return False

    def _object_support_type_matches_any(name: str = "obj") -> bool:
        attrs = _object_attributes(str(name))
        if not bool(attrs & FOOD_TYPE_NAMES):
            return True
        for support_name in getattr(env, "objects", {}).keys():
            support_name = str(support_name)
            if support_name == str(name):
                continue
            try:
                if OU.check_obj_in_receptacle(env, name, support_name):
                    return True
            except Exception:
                pass
            try:
                if env.check_contact(env.objects[name], env.objects[support_name]):
                    return True
            except Exception:
                continue
        return False

    def _object_supported_on_correct(
        name: str,
        target_fixture_names: set[str],
        target_object_names: set[str],
    ) -> bool:
        for fixture_name in target_fixture_names:
            try:
                if OU.check_obj_fixture_contact(env, name, fixture_name):
                    return True
            except Exception:
                pass
            try:
                if OU.obj_inside_of(env, name, fixture_name, partial_check=True):
                    return True
            except Exception:
                pass
            if _fixture_rack_contact(str(fixture_name), str(name)):
                return True
        for target_name in target_object_names:
            if str(target_name) == str(name):
                continue
            try:
                if OU.check_obj_in_receptacle(env, name, target_name):
                    return True
            except Exception:
                pass
            try:
                if env.check_contact(env.objects[name], env.objects[target_name]):
                    return True
            except Exception:
                continue
        return False

    def _current_support_contacts(name: str) -> tuple[set[str], set[str]]:
        fixture_contacts = set()
        object_contacts = set()
        for fixture_name in getattr(env, "fixtures", {}).keys():
            try:
                if OU.check_obj_fixture_contact(env, name, fixture_name):
                    fixture_contacts.add(str(fixture_name))
                    continue
            except Exception:
                pass
            try:
                if OU.obj_inside_of(env, name, fixture_name, partial_check=True):
                    fixture_contacts.add(str(fixture_name))
                    continue
            except Exception:
                pass
            if _fixture_rack_contact(str(fixture_name), str(name)):
                fixture_contacts.add(str(fixture_name))
                continue
        for other_name in getattr(env, "objects", {}).keys():
            other_name = str(other_name)
            if other_name == str(name):
                continue
            try:
                if OU.check_obj_in_receptacle(env, name, other_name):
                    object_contacts.add(other_name)
                    continue
            except Exception:
                pass
            try:
                if env.check_contact(env.objects[name], env.objects[other_name]):
                    object_contacts.add(other_name)
            except Exception:
                continue
        return fixture_contacts, object_contacts

    def _object_attributes(name: str = "obj") -> set[str]:
        objects = ((static_info or {}).get("scene_layout") or {}).get("objects") or {}
        info = objects.get(name) or {}
        meta_info = (
            (info.get("info") or {}) if isinstance(info.get("info"), dict) else {}
        )
        type_hints = set(meta_info.get("types") or ())
        type_hints.update(meta_info.get("groups") or ())
        type_hints.update(meta_info.get("groups_containing_sampled_obj") or ())
        return infer_object_attributes(str(info.get("category") or ""), type_hints)

    def _object_category(name: str) -> str | None:
        objects = ((static_info or {}).get("scene_layout") or {}).get("objects") or {}
        info = objects.get(str(name)) or {}
        category = info.get("category") or (info.get("info") or {}).get("cat")
        return str(category) if category else None

    def _object_placement_fixture_hint(name: str) -> str | None:
        cfg = _object_configs().get(str(name)) or {}
        placement = cfg.get("placement") or {}
        fixture = placement.get("fixture") if isinstance(placement, dict) else None
        if fixture is None:
            return None
        text = str(fixture)
        if "Counter" in text:
            return "counter"
        if "Sink" in text:
            return "sink"
        if "Stove" in text:
            return "stove"
        if "Microwave" in text:
            return "microwave"
        if "Dishwasher" in text:
            return "dishwasher"
        if "Cabinet" in text:
            return "cabinet"
        return text

    def _object_speeds(name: str) -> tuple[float, float]:
        velocity = ((dynamic_info.get("scene") or {}).get("objects") or {}).get(
            name, {}
        ).get("velocity") or {}
        return _norm(velocity.get("linear")), _norm(velocity.get("angular"))

    def _object_ou_bbox_aabb(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            body_id = env.obj_body_id[str(name)]
            trans = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
            rot_quat = np.asarray(env.sim.data.body_xquat[body_id], dtype=float)
            points = env.objects[str(name)].get_bbox_points(
                trans=trans,
                rot=rot_quat,
            )
            coords = np.asarray(points, dtype=float)
        except Exception:
            return None
        if coords.ndim != 2 or coords.shape[1] < 3 or not np.all(np.isfinite(coords)):
            return None
        return np.min(coords[:, :3], axis=0), np.max(coords[:, :3], axis=0)

    def _object_bbox_aabb(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        pos = _object_position(name)
        if pos is None:
            return None
        info = (
            ((static_info or {}).get("scene_layout") or {})
            .get("objects", {})
            .get(str(name), {})
        )
        bbox = info.get("bbox") or {}
        try:
            lower = np.asarray(bbox.get("min"), dtype=float).reshape(-1)
            upper = np.asarray(bbox.get("max"), dtype=float).reshape(-1)
        except Exception:
            return None
        if (
            lower.size < 3
            or upper.size < 3
            or not np.all(np.isfinite(lower[:3]))
            or not np.all(np.isfinite(upper[:3]))
        ):
            return None
        return pos + lower[:3], pos + upper[:3]

    def _object_aabb(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        return (
            _object_ou_bbox_aabb(str(name))
            or _object_contact_aabb(str(name))
            or _object_bbox_aabb(str(name))
        )

    def _geom_aabb(geom_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            center = np.asarray(env.sim.data.geom_xpos[int(geom_id)], dtype=float)[:3]
            xmat = np.asarray(
                env.sim.data.geom_xmat[int(geom_id)], dtype=float
            ).reshape(3, 3)
            size = np.asarray(env.sim.model.geom_size[int(geom_id)], dtype=float)[:3]
        except Exception:
            return None
        if (
            center.size < 3
            or xmat.shape != (3, 3)
            or size.size < 3
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(xmat))
            or not np.all(np.isfinite(size))
        ):
            return None
        half_extents = np.abs(xmat) @ np.maximum(size, 0.0)
        return center - half_extents, center + half_extents

    def _geom_ids_aabb(geom_ids: set[int]) -> tuple[np.ndarray, np.ndarray] | None:
        geom_aabbs = [
            aabb
            for geom_id in geom_ids
            for aabb in [_geom_aabb(geom_id)]
            if aabb is not None
        ]
        if not geom_aabbs:
            return None
        lowers = [aabb[0] for aabb in geom_aabbs]
        uppers = [aabb[1] for aabb in geom_aabbs]
        return np.min(lowers, axis=0), np.max(uppers, axis=0)

    def _object_contact_aabb(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        return _geom_ids_aabb(_object_contact_geom_ids(str(name)))

    def _gripper_contact_aabb() -> tuple[np.ndarray, np.ndarray] | None:
        return _geom_ids_aabb(_gripper_contact_geom_ids())

    def _fixture_contact_geom_ids(name: str) -> set[int]:
        fixture = getattr(env, "fixtures", {}).get(str(name))
        if fixture is None:
            return set()
        try:
            return _geom_ids_from_names(fixture.contact_geoms)
        except Exception:
            return set()

    def _fixture_contact_aabb(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        return _geom_ids_aabb(_fixture_contact_geom_ids(str(name)))

    def _fixture_ext_sites_aabb(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        fixture = _fixture_by_name(str(name))
        if fixture is None:
            return None
        try:
            points = fixture.get_ext_sites(all_points=True, relative=False)
            coords = np.asarray(points, dtype=float)
        except Exception:
            return None
        if coords.ndim != 2 or coords.shape[1] < 3 or not np.all(np.isfinite(coords)):
            return None
        return np.min(coords[:, :3], axis=0), np.max(coords[:, :3], axis=0)

    def _fixture_aabb(name: str) -> tuple[np.ndarray, np.ndarray] | None:
        return (
            _fixture_ext_sites_aabb(str(name))
            or _fixture_contact_aabb(str(name))
            or _geom_ids_aabb(_fixture_geom_ids_by_names({str(name)}))
        )

    def _fixture_interior_support_aabb(
        name: str,
        obj_pos: np.ndarray | None = None,
        region_keywords: tuple[str, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        fixture = _fixture_by_name(str(name))
        if fixture is None:
            return None

        try:
            region_names = [
                str(region_name) for region_name in fixture.get_reset_region_names()
            ]
        except Exception:
            region_names = []
        if region_keywords is not None:
            keywords = tuple(str(keyword).lower() for keyword in region_keywords)
            region_names = [
                region_name
                for region_name in region_names
                if any(keyword in region_name.lower() for keyword in keywords)
            ]
        if not region_names:
            return None

        try:
            sites_by_region = fixture.get_int_sites(all_points=True, relative=False)
        except Exception:
            return None

        candidates = []
        for region_name in region_names:
            points = sites_by_region.get(region_name, [])
            try:
                coords = np.asarray(points, dtype=float)
            except Exception:
                continue
            if (
                coords.ndim != 2
                or coords.shape[1] < 3
                or not np.all(np.isfinite(coords))
            ):
                continue
            lower = np.min(coords[:, :3], axis=0)
            upper = np.max(coords[:, :3], axis=0)
            support_z = float(lower[2])
            support_aabb = (
                np.asarray([lower[0], lower[1], support_z], dtype=float),
                np.asarray([upper[0], upper[1], support_z], dtype=float),
            )
            if obj_pos is None:
                candidates.append((0.0, -support_z, support_aabb))
                continue
            if support_z > float(obj_pos[2]) + SUPPORT_CLUTTER_Z_TOLERANCE:
                continue
            xy_gap = np.maximum(
                0.0,
                np.maximum(lower[:2] - obj_pos[:2], obj_pos[:2] - upper[:2]),
            )
            candidates.append((float(np.linalg.norm(xy_gap)), -support_z, support_aabb))
        if not candidates:
            return None
        return sorted(candidates)[0][2]

    def _fixture_rack_aabb(
        name: str,
        obj_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        fixture = _fixture_by_name(str(name))
        if fixture is None:
            return None

        interior_aabb = _fixture_interior_support_aabb(
            str(name),
            obj_pos=obj_pos,
            region_keywords=("rack", "tray"),
        )
        if interior_aabb is not None:
            return interior_aabb

        joint_names = getattr(fixture, "_joint_names", {}) or {}
        if not isinstance(joint_names, dict):
            return None
        rack_joint_names = [
            str(joint_name)
            for key, joint_name in joint_names.items()
            if "rack" in str(key).lower() or "tray" in str(key).lower()
        ]
        geom_ids = set()
        for joint_name in rack_joint_names:
            try:
                joint_id = env.sim.model.joint_name2id(joint_name)
                body_id = int(env.sim.model.jnt_bodyid[joint_id])
            except Exception:
                continue
            for geom_id, geom_body_id in enumerate(env.sim.model.geom_bodyid):
                if int(geom_body_id) == body_id:
                    geom_ids.add(int(geom_id))
        return _geom_ids_aabb(geom_ids)

    def _fixture_support_aabb(
        name: str,
        obj_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        fixture = _fixture_by_name(str(name))
        fixture_text = f"{name} {fixture.__class__.__name__ if fixture is not None else ''}".lower()
        interior_support_fixture_keywords = (
            "cabinet",
            "dishwasher",
            "drawer",
            "fridge",
            "freezer",
            "microwave",
            "oven",
            "sink",
            "toaster",
            "rack",
        )
        rack_aabb = (
            _fixture_rack_aabb(str(name), obj_pos=obj_pos)
            if "dishwasher" in fixture_text
            else None
        )
        interior_aabb = (
            _fixture_interior_support_aabb(str(name), obj_pos=obj_pos)
            if any(
                keyword in fixture_text for keyword in interior_support_fixture_keywords
            )
            else None
        )
        return rack_aabb or interior_aabb or _fixture_aabb(str(name))

    def _closest_point_on_aabb_xy(
        point: np.ndarray,
        aabb: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        a_min, a_max = aabb
        return np.asarray(
            [
                float(np.clip(point[0], a_min[0], a_max[0])),
                float(np.clip(point[1], a_min[1], a_max[1])),
                float(a_max[2]),
            ],
            dtype=float,
        )

    def _aabb_distance(
        a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
    ) -> float:
        a_min, a_max = a
        b_min, b_max = b
        gap = np.maximum(0.0, np.maximum(b_min - a_max, a_min - b_max))
        return float(np.linalg.norm(gap))

    def _aabb_xy_distance(
        a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]
    ) -> float:
        a_min, a_max = a
        b_min, b_max = b
        gap = np.maximum(0.0, np.maximum(b_min[:2] - a_max[:2], a_min[:2] - b_max[:2]))
        return float(np.linalg.norm(gap))

    def _point_aabb_xy_distance(
        point: np.ndarray, aabb: tuple[np.ndarray, np.ndarray]
    ) -> float:
        a_min, a_max = aabb
        clipped = np.minimum(np.maximum(point[:2], a_min[:2]), a_max[:2])
        return float(np.linalg.norm(point[:2] - clipped))

    def _aabb_overlap_depth(
        a: tuple[np.ndarray, np.ndarray],
        b: tuple[np.ndarray, np.ndarray],
    ) -> float:
        a_min, a_max = a
        b_min, b_max = b
        overlap = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
        if np.any(overlap <= 0.0):
            return 0.0
        return float(np.min(overlap))

    def _aabb_intersects(
        a: tuple[np.ndarray, np.ndarray],
        b: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        return _aabb_overlap_depth(a, b) > 0.0

    def _aabb_obstructs_path(
        blocker: tuple[np.ndarray, np.ndarray],
        corridor: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        return (
            _aabb_overlap_depth(blocker, corridor) > PATH_OBSTRUCTION_OVERLAP_ALLOWANCE
        )

    def _aabb_obstructs_between_endpoints(
        blocker: tuple[np.ndarray, np.ndarray],
        start: tuple[np.ndarray, np.ndarray],
        end: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        if not _aabb_obstructs_path(blocker, _union_aabb(start, end)):
            return False
        if _aabb_intersects(blocker, start) or _aabb_intersects(blocker, end):
            return False
        start_center = (start[0] + start[1]) / 2.0
        end_center = (end[0] + end[1]) / 2.0
        segment_xy = end_center[:2] - start_center[:2]
        segment_len_sq = float(np.dot(segment_xy, segment_xy))
        if segment_len_sq <= 1e-9:
            return False
        blocker_center = (blocker[0] + blocker[1]) / 2.0
        projection = float(
            np.dot(blocker_center[:2] - start_center[:2], segment_xy) / segment_len_sq
        )
        if projection <= 0.0 or projection >= 1.0:
            return False
        closest_xy = start_center[:2] + projection * segment_xy
        segment_xy_distance = _point_aabb_xy_distance(
            np.array([closest_xy[0], closest_xy[1], blocker_center[2]], dtype=float),
            (blocker[0], blocker[1]),
        )
        if segment_xy_distance > PATH_OBSTRUCTION_OVERLAP_ALLOWANCE:
            return False
        blocker_min, blocker_max = blocker
        for axis in range(3):
            low = min(start_center[axis], end_center[axis])
            high = max(start_center[axis], end_center[axis])
            if blocker_max[axis] <= low or blocker_min[axis] >= high:
                return False
        return True

    def _union_aabb(
        a: tuple[np.ndarray, np.ndarray],
        b: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        a_min, a_max = a
        b_min, b_max = b
        return np.minimum(a_min, b_min), np.maximum(a_max, b_max)

    def _translate_aabb(
        aabb: tuple[np.ndarray, np.ndarray],
        delta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        a_min, a_max = aabb
        return a_min + delta, a_max + delta

    def _gripper_link_aabb() -> tuple[np.ndarray, np.ndarray] | None:
        link_poses = (dynamic_info.get("robot") or {}).get("link_poses") or {}
        points = []
        if isinstance(link_poses, dict):
            for link_name, pose in link_poses.items():
                lowered = str(link_name).lower()
                if (
                    "gripper" not in lowered
                    and "finger" not in lowered
                    and "eef" not in lowered
                ):
                    continue
                if not isinstance(pose, dict):
                    continue
                try:
                    pos = np.asarray(pose.get("position"), dtype=float).reshape(-1)
                except Exception:
                    continue
                if pos.size >= 3 and np.all(np.isfinite(pos[:3])):
                    points.append(pos[:3])
        eef = _eef_position()
        if eef is not None:
            points.append(eef)
        if not points:
            return None
        arr = np.asarray(points, dtype=float)
        return np.min(arr, axis=0), np.max(arr, axis=0)

    def _gripper_aabb() -> tuple[np.ndarray, np.ndarray] | None:
        return _gripper_contact_aabb() or _gripper_link_aabb()

    def _pick_swept_endpoint_aabbs(
        object_name: str,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
        gripper_aabb = _gripper_aabb()
        target_aabb = _object_aabb(object_name)
        if gripper_aabb is None or target_aabb is None:
            return None
        return gripper_aabb, target_aabb

    def _object_swept_to_point_endpoint_aabbs(
        object_name: str,
        end: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
        start = _object_position(object_name)
        object_aabb = _object_aabb(object_name)
        if start is None or object_aabb is None:
            return None
        end_aabb = _translate_aabb(object_aabb, end - start)
        return object_aabb, end_aabb

    def _object_edge_distance(name_a: str, name_b: str) -> float:
        aabb_a = _object_aabb(name_a)
        aabb_b = _object_aabb(name_b)
        if aabb_a is not None and aabb_b is not None:
            return _aabb_distance(aabb_a, aabb_b)

        pos_a = _object_position(name_a)
        pos_b = _object_position(name_b)
        if pos_a is None or pos_b is None:
            return float("inf")
        return float(np.linalg.norm(pos_a - pos_b))

    def _object_xy_edge_distance(name_a: str, name_b: str) -> float:
        aabb_a = _object_aabb(name_a)
        aabb_b = _object_aabb(name_b)
        if aabb_a is not None and aabb_b is not None:
            return _aabb_xy_distance(aabb_a, aabb_b)

        pos_a = _object_position(name_a)
        pos_b = _object_position(name_b)
        if pos_a is None or pos_b is None:
            return float("inf")
        return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))

    def _eef_linear_velocity() -> np.ndarray | None:
        velocity = (dynamic_info.get("robot") or {}).get("end_effector_velocity") or {}
        try:
            linear = np.asarray(velocity.get("linear"), dtype=float).reshape(-1)
        except Exception:
            return None
        if linear.size < 3 or not np.all(np.isfinite(linear[:3])):
            return None
        return linear[:3]

    def _eef_angular_velocity() -> np.ndarray | None:
        velocity = (dynamic_info.get("robot") or {}).get("end_effector_velocity") or {}
        try:
            angular = np.asarray(velocity.get("angular"), dtype=float).reshape(-1)
        except Exception:
            return None
        if angular.size < 3 or not np.all(np.isfinite(angular[:3])):
            return None
        return angular[:3]

    def _object_linear_velocity(name: str) -> np.ndarray | None:
        velocity = ((dynamic_info.get("scene") or {}).get("objects") or {}).get(
            name, {}
        ).get("velocity") or {}
        try:
            linear = np.asarray(velocity.get("linear"), dtype=float).reshape(-1)
        except Exception:
            return None
        if linear.size < 3 or not np.all(np.isfinite(linear[:3])):
            return None
        return linear[:3]

    def _object_angular_velocity(name: str) -> np.ndarray | None:
        velocity = ((dynamic_info.get("scene") or {}).get("objects") or {}).get(
            name, {}
        ).get("velocity") or {}
        try:
            angular = np.asarray(velocity.get("angular"), dtype=float).reshape(-1)
        except Exception:
            return None
        if angular.size < 3 or not np.all(np.isfinite(angular[:3])):
            return None
        return angular[:3]

    def _object_stable(name: str) -> bool:
        linear_speed, angular_speed = _object_speeds(name)
        return _bool(
            linear_speed < OBJ_LINEAR_STABLE_THRESHOLD
            and angular_speed < OBJ_ANGULAR_STABLE_THRESHOLD
        )

    def _object_eef_relative_speeds(name: str) -> tuple[float, float]:
        """Object's linear / angular speed relative to the end-effector's
        velocity, falling back to absolute world-frame speed if either
        velocity is unavailable.

        Factored out so both _object_is_grasped (raw grasp signal) and
        _object_sync (ongoing grasp-safety monitoring) can share
        it without _object_is_grasped depending on its own smoothed output.
        """
        linear_speed, angular_speed = _object_speeds(name)
        obj_vel = _object_linear_velocity(name)
        eef_vel = _eef_linear_velocity()
        if obj_vel is not None and eef_vel is not None:
            linear_speed = float(np.linalg.norm(obj_vel - eef_vel))
        obj_ang_vel = _object_angular_velocity(name)
        eef_ang_vel = _eef_angular_velocity()
        if obj_ang_vel is not None and eef_ang_vel is not None:
            angular_speed = float(np.linalg.norm(obj_ang_vel - eef_ang_vel))
        return linear_speed, angular_speed

    def _object_sync(name: str) -> bool:
        """Whether `name` is moving in sync with the end effector.

        This is only a meaningful question while the object is actually
        grasped (only then is it expected to track the gripper); callers
        that also care about the not-grasped case should combine this with
        their own object_grasped condition, e.g. `object_grasped and
        _object_sync(name)`, the same way object_grasped_safe
        does below. There is no separate "sync" concept for a non-grasped
        object here -- the useful notion in that case is plain
        object_stable, not sync-to-eef.
        """
        linear_speed, angular_speed = _object_eef_relative_speeds(name)
        return _bool(
            linear_speed < OBJ_LINEAR_STABLE_THRESHOLD
            and angular_speed < OBJ_ANGULAR_STABLE_THRESHOLD
        )

    def _object_support_reference(name: str) -> str | None:
        """Name of the movable object currently supporting `name`, if any.

        Only movable ``env.objects`` supports are returned (e.g. a basket the
        object is resting in) since only those have a tracked velocity;
        fixture supports are treated as stationary, matching prior behavior.
        """
        try:
            _, object_contacts = _current_support_contacts(name)
        except Exception:
            return None
        object_contacts = {
            contact_name for contact_name in object_contacts if contact_name != str(name)
        }
        if not object_contacts:
            return None
        return sorted(object_contacts)[0]

    def _object_stable_relative(name: str) -> bool:
        """Like _object_stable, but measured relative to the object's current
        support instead of the world frame.

        This mirrors what _object_sync already does for a grasped
        object relative to the end effector: if the support itself is moving (e.g.
        a basket being carried that this object is resting inside), the
        object should count as stable as long as it isn't sliding /
        rattling relative to that support, even though its absolute speed
        is nonzero. Without this, an object that settled inside a carried
        receptacle right after being released spuriously fails
        object_settled / trips object_settle_timeout once the receptacle is
        picked up again, even though nothing about the object itself became
        unstable.
        """
        linear_speed, angular_speed = _object_speeds(name)
        support_name = _object_support_reference(name)
        if support_name is not None:
            obj_linear = _object_linear_velocity(name)
            support_linear = _object_linear_velocity(support_name)
            if obj_linear is not None and support_linear is not None:
                linear_speed = float(np.linalg.norm(obj_linear - support_linear))
            obj_angular = _object_angular_velocity(name)
            support_angular = _object_angular_velocity(support_name)
            if obj_angular is not None and support_angular is not None:
                angular_speed = float(np.linalg.norm(obj_angular - support_angular))
        return _bool(
            linear_speed < OBJ_LINEAR_STABLE_THRESHOLD
            and angular_speed < OBJ_ANGULAR_STABLE_THRESHOLD
        )

    def _object_settled(
        name: str,
        target_fixture_names: set[str],
        target_object_names: set[str],
    ) -> bool:
        return _bool(
            _object_supported(name)
            and _object_support_type_matches_any(name)
            and _object_stable_relative(name)
            and _gripper_far_from_object(name)
        )

    object_configs = _object_configs()
    receive_object_names = _receive_object_names_from_configs(object_configs)
    manipulated_object_names = _manipulated_object_names(
        object_configs, receive_object_names
    )

    def _new_monitor_state() -> dict:
        return {
            "prev_values": {},
            "prev_predicates": {},
            "forbidden_contact_candidate": None,
            "grasp_age": 0,
            "prev_object_grasped": False,
            "object_grasped_object": None,
            "object_grasp_candidate": None,
            "robot_contact_raw_active": False,
            "robot_contact_raw_activated_frame": None,
            "robot_contact_raw_candidate": None,
            "robot_contact_raw_candidate_sources": [],
            "robot_contact_clean_candidate": None,
            "contamination_transfer_pair": None,
            "contamination_transfer_source": None,
            "contamination_transfer_target": None,
            "contaminated_objects": [],
            "contaminated_fixtures": [],
            "active_object": None,
            "awaiting_settle": False,
            "settle_watch_object": None,
            "settle_watch_age": 0,
            "settle_release_frame": None,
            "settle_release_object": None,
            "initial_contact_pairs": None,
            "ignored_initial_contact_pairs": None,
            "removed_initial_contact_pairs": set(),
            "robot_contact_raw_sources": [],
            "source_support_fixtures": [],
            "source_support_objects": [],
            "last_timestep": None,
            "monitor_frame_index": -1,
            "prev_object_grasped_safe": False,
            "skill_pick_onset_candidate_count": 0,
            "skill_pick_onset_fired_object": None,
            "pick_approach_candidate_object": None,
            "pick_approach_candidate_count": 0,
            "pick_approach_false_count": 0,
            "prev_gripper_object_distances": {},
            "prev_gripper_target_distances": {},
            "target_approach_candidate": None,
            "target_approach_candidate_count": 0,
            "target_approach_false_count": 0,
            "target_approach_candidates_by_action": {},
            "target_approach_counts_by_action": {},
            "target_approach_false_counts_by_action": {},
            "skill_press_onset_candidate_count": 0,
            "skill_press_onset_fired_target": None,
            "skill_turn_onset_candidate_count": 0,
            "skill_turn_onset_fired_target": None,
            "skill_slide_onset_candidate_count": 0,
            "skill_slide_onset_fired_target": None,
            "skill_twist_onset_candidate_count": 0,
            "skill_twist_onset_fired_target": None,
            "skill_open_close_onset_candidate_count": 0,
            "skill_open_close_onset_fired_target": None,
            "skill_dump_onset_candidate_count": 0,
            "skill_dump_onset_fired_object": None,
            "skill_dump_onset_fired_content_names": [],
            "skill_dump_onset_content_names": [],
            "active_containment_transfer": None,
            "last_fixture_output_frame": None,
            "fixture_output_active": False,
            "fixture_output_states": {},
            "fixture_output_last_target": None,
            "microwave_occupancy_stable_count": None,
            "microwave_occupancy_candidate": None,
            "microwave_occupancy_candidate_count": 0,
            "microwave_empty_count": 0,
            "prev_gripper_in_fixture": False,
            "prev_object_in_fixture": False,
            "prev_object_reaching_fixture": False,
            "access_active_fixture": None,
            "access_object_fixture": None,
            "prev_grasped_receptacle_upright": True,
            "grasped_receptacle_upright_false_count": 0,
            "grasped_receptacle_content_source": None,
            "grasped_receptacle_content_names": [],
            "skill_place_onset_candidate_count": 0,
            "skill_place_onset_fired_object": None,
            "persistent_bools": {},
        }

    current_timestep = int(((dynamic_info.get("task") or {}).get("timestep", -1)))
    monitor_state = getattr(env, "_predicate_monitor_state", None)
    if not isinstance(monitor_state, dict):
        monitor_state = _new_monitor_state()
        env._predicate_monitor_state = monitor_state
    else:
        previous_timestep = monitor_state.get("last_timestep")
        try:
            timestep_restarted = (
                previous_timestep is not None
                and current_timestep >= 0
                and current_timestep <= int(previous_timestep)
            )
        except Exception:
            timestep_restarted = False
        if timestep_restarted:
            monitor_state = _new_monitor_state()
            env._predicate_monitor_state = monitor_state

    monitor_frame_index = int(monitor_state.get("monitor_frame_index", -1)) + 1
    monitor_state["monitor_frame_index"] = monitor_frame_index

    def _persistent_bool(
        key: str,
        raw_value: bool,
        threshold: int = STABLE_PERSISTENCE_FRAME,
    ) -> bool:
        states = monitor_state.setdefault("persistent_bools", {})
        raw = _bool(raw_value)
        state = states.get(key)
        if not isinstance(state, dict):
            states[key] = {"value": raw, "candidate": raw, "count": 0}
            return raw
        current = _bool(state.get("value", raw))
        candidate = _bool(state.get("candidate", raw))
        if raw == current:
            state["candidate"] = raw
            state["count"] = 0
            return current
        count = int(state.get("count", 0)) + 1 if raw == candidate else 1
        candidate = raw
        if count >= max(1, int(threshold)):
            current = raw
            candidate = raw
            count = 0
        state["value"] = current
        state["candidate"] = candidate
        state["count"] = count
        return current

    def _persistent_stable_after_event(
        key: str,
        raw_value: bool,
        threshold: int = STABLE_PERSISTENCE_FRAME,
    ) -> bool:
        """Start unstable, require consecutive true frames, and drop false immediately."""
        states = monitor_state.setdefault("persistent_bools", {})
        raw = _bool(raw_value)
        state = states.get(key)
        if not isinstance(state, dict):
            count = 1 if raw else 0
            value = _bool(raw and count >= max(1, int(threshold)))
            states[key] = {"value": value, "count": count}
            return value
        if not raw:
            state["value"] = False
            state["count"] = 0
            return False
        count = int(state.get("count", 0)) + 1
        value = _bool(count >= max(1, int(threshold)))
        state["value"] = value
        state["count"] = count
        return value

    robot_geom_ids = _robot_geom_ids()
    robot_base_geom_ids = _robot_base_geom_ids()
    robot_policy_geom_ids = robot_geom_ids - robot_base_geom_ids
    contact_number = int(getattr(env.sim.data, "ncon", 0))

    manipulated_geom_ids_by_name = {
        str(name): _object_geom_ids(name) for name in manipulated_object_names
    }
    robot_contacted_names = set()
    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue
        if geom1 in robot_base_geom_ids or geom2 in robot_base_geom_ids:
            continue
        if geom1 in robot_policy_geom_ids and geom2 not in robot_geom_ids:
            contacted_geom = geom2
        elif geom2 in robot_policy_geom_ids and geom1 not in robot_geom_ids:
            contacted_geom = geom1
        else:
            continue
        for name, geom_ids in manipulated_geom_ids_by_name.items():
            if contacted_geom in geom_ids:
                robot_contacted_names.add(name)
                break

    grasped_names = {
        name for name in manipulated_object_names if _object_is_grasped(name)
    }

    def _object_receptacle_like_from_config(name: str) -> bool:
        cfg = object_configs.get(str(name)) or {}
        info = cfg.get("info") or {}
        groups = set()
        for key in ("groups", "groups_containing_sampled_obj"):
            values = info.get(key) or cfg.get(key)
            if isinstance(values, str):
                groups.add(values)
            elif isinstance(values, (list, tuple, set)):
                groups.update(str(value) for value in values)
        cat = str(info.get("cat") or cfg.get("category") or "").lower()
        if cat:
            groups.add(cat)
        return _bool(
            "receptacle" in {str(group).lower() for group in groups}
            or cat in RECEPTACLE_CATEGORIES
        )

    def _carrier_for_grasp_candidate(name: str | None) -> str | None:
        if name is None:
            return None
        for carrier_name in sorted(manipulated_object_names):
            carrier_name = str(carrier_name)
            if carrier_name == str(name):
                continue
            if not _object_receptacle_like_from_config(carrier_name):
                continue
            try:
                if OU.check_obj_in_receptacle(env, str(name), carrier_name):
                    return carrier_name
            except Exception:
                pass
        return str(name)

    previous_active_object = monitor_state.get("active_object")
    env_object_names = {str(name) for name in getattr(env, "objects", {}).keys()}
    persistent_object_stable_by_name = {
        name: _persistent_bool(f"object_stable::{name}", _object_stable(name))
        for name in env_object_names
    }
    if previous_active_object not in env_object_names:
        previous_active_object = None
    settle_watch_object = monitor_state.get("settle_watch_object")
    if settle_watch_object not in env_object_names:
        settle_watch_object = None
    settle_watch_age = int(monitor_state.get("settle_watch_age", 0))
    settle_release_frame = monitor_state.get("settle_release_frame")
    settle_release_object = monitor_state.get("settle_release_object")
    if settle_release_object not in env_object_names:
        settle_release_object = None
    gripper_is_opening = _gripper_is_opening()
    prev_object_grasped = _bool(monitor_state.get("prev_object_grasped", False))
    previous_grasped_object = monitor_state.get("object_grasped_object")
    if previous_grasped_object not in env_object_names:
        previous_grasped_object = None
    raw_grasp_candidate = sorted(grasped_names)[0] if grasped_names else None
    grasp_candidate = _carrier_for_grasp_candidate(raw_grasp_candidate)
    # No debounce: object_grasped tracks the raw grasp candidate directly.
    # This used to require OBJECT_GRASPED_PERSISTENCE_FRAMES consecutive
    # frames on both the rising and falling edge, to absorb flicker from the
    # old aggregate-contact grasp check. That flicker source is now fixed at
    # the raw-signal level (bilateral contact, see _object_is_grasped), so
    # the debounce is no longer needed.
    object_grasped = _bool(grasp_candidate is not None)
    grasped_object = grasp_candidate if object_grasped else None
    monitor_state["object_grasp_candidate"] = grasp_candidate
    monitor_state["object_grasped_object"] = grasped_object
    if object_grasped and grasped_object is not None:
        active_object = grasped_object
    elif robot_contacted_names:
        active_object = sorted(robot_contacted_names)[0]
    else:
        active_object = previous_active_object
    source_support_fixtures = set(monitor_state.get("source_support_fixtures") or [])
    source_support_objects = set(monitor_state.get("source_support_objects") or [])
    if (
        active_object is not None
        and object_grasped
        and (
            not prev_object_grasped or str(previous_active_object) != str(active_object)
        )
    ):
        source_fixture_contacts, source_object_contacts = _current_support_contacts(
            active_object
        )
        source_support_fixtures = set(source_fixture_contacts)
        source_support_objects = set(source_object_contacts)
    object_released = _bool(
        prev_object_grasped
        and not object_grasped
        and (
            gripper_is_opening
            # covers the gripper retracting away from the object without ever
            # opening its fingers (e.g. contact breaks as the arm moves off)
            # while the object is resting on a support -- still a deliberate
            # release, just one that doesn't show up as a finger-opening
            # motion. Uses object_supported rather than object_stable: a
            # freshly-dropped object is essentially never already resting on
            # something at the exact frame contact breaks (it's still in
            # free-fall), so this doesn't reopen the accidental-drop case,
            # unlike a plain not-moving check which could read true for one
            # frame before gravity builds up velocity.
            or (
                previous_grasped_object is not None
                and _object_supported(previous_grasped_object)
            )
        )
    )
    awaiting_settle = _bool(monitor_state.get("awaiting_settle", False))
    if object_released:
        awaiting_settle = True
        settle_watch_object = previous_active_object
        settle_watch_age = 0
        settle_release_frame = current_timestep
        settle_release_object = previous_active_object
    elif awaiting_settle:
        settle_watch_age += 1

    active_objects = {
        str(name)
        for name in [active_object]
        if name is not None and str(name) in env_object_names
    }
    manipulated_object_names.update(active_objects)

    obj_name = active_object
    settle_obj_name = (
        settle_watch_object
        if awaiting_settle and settle_watch_object is not None
        else active_object
    )
    target_fixture_names = _target_fixture_names()
    target_object_names = _target_object_names(
        manipulated_object_names,
        receive_object_names,
    )
    target_fixtures_by_object = _target_fixtures_by_manipulated(
        manipulated_object_names,
        target_fixture_names,
        object_configs,
    )
    target_objects_by_object = _target_objects_by_manipulated(
        manipulated_object_names,
        target_object_names,
        object_configs,
    )
    active_target_fixture_names = target_fixtures_by_object.get(
        str(active_object),
        set(target_fixture_names),
    )
    active_target_object_names = target_objects_by_object.get(
        str(active_object),
        set(target_object_names),
    )
    active_source_fixture_names = set(source_support_fixtures)
    active_source_object_names = set(source_support_objects)
    contact_policy_action_fixture_names = {
        str(name)
        for name in _task_ref_names()
        if str(name) in getattr(env, "fixtures", {})
    }

    manipulated_geom_ids = set()
    for geom_ids in manipulated_geom_ids_by_name.values():
        manipulated_geom_ids.update(geom_ids)
    grasped_object_geom_ids = (
        _object_geom_ids(active_object) if active_object else set()
    )
    (
        contact_policy_action_fixture_geom_ids,
        contact_policy_action_fixture_actions,
    ) = _fixture_action_component_geom_ids_by_names(contact_policy_action_fixture_names)
    contact_policy_object_fixture_names = set(target_fixture_names) | {
        fixture_name
        for fixture_name in contact_policy_action_fixture_names
        if _contact_policy_fixture_actions(fixture_name)
    }
    contact_policy_object_fixture_geom_ids_by_name = _fixture_geom_ids_by_name_map(
        contact_policy_object_fixture_names
    )
    contact_policy_object_fixture_geom_ids = set()
    for geom_ids in contact_policy_object_fixture_geom_ids_by_name.values():
        contact_policy_object_fixture_geom_ids.update(geom_ids)
    target_fixture_geom_ids = (
        _fixture_geom_ids_by_names(target_fixture_names)
        | contact_policy_action_fixture_geom_ids
    )
    active_target_fixture_geom_ids = _fixture_geom_ids_by_names(
        active_target_fixture_names,
    )
    active_source_fixture_geom_ids = _fixture_geom_ids_by_names(
        active_source_fixture_names,
    )
    target_object_geom_ids = set()
    for target_name in target_object_names:
        target_object_geom_ids.update(_object_geom_ids(target_name))
    active_target_object_geom_ids = set()
    for target_name in active_target_object_names:
        active_target_object_geom_ids.update(_object_geom_ids(target_name))
    active_source_object_geom_ids = set()
    for source_name in active_source_object_names:
        active_source_object_geom_ids.update(_object_geom_ids(source_name))

    robot_correct_manipulated_object_contact = False
    robot_correct_fixture_contact = False
    correct_manipulated_object_correct_fixture_contact = False
    correct_manipulated_object_correct_receive_object_contact = False
    correct_manipulated_object_original_support_contact = False
    task_referenced_object_fixture_contact_names = set()
    forbidden_contact_pairs = []
    considered_contact_pairs = []

    current_contact_pairs = set()
    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue
        current_contact_pairs.add(_canonical_contact_pair(geom1, geom2))
    if monitor_state.get("initial_contact_pairs") is None:
        monitor_state["initial_contact_pairs"] = set(current_contact_pairs)
        monitor_state["ignored_initial_contact_pairs"] = set(current_contact_pairs)
    stored_ignored_initial_pairs = monitor_state.get("ignored_initial_contact_pairs")
    if stored_ignored_initial_pairs is None:
        stored_ignored_initial_pairs = (
            monitor_state.get("initial_contact_pairs") or set()
        )
    ignored_initial_contact_pairs = set(stored_ignored_initial_pairs)
    removed_initial_contact_pairs = set(
        monitor_state.get("removed_initial_contact_pairs") or set()
    )
    # No grace: a pair is dropped from the ignored set the first frame it's
    # absent (previously required CONTACT_PERSISTENCE_FRAMES consecutive
    # missing frames, but that constant is 1, so this is behaviorally
    # unchanged -- just without the now-unneeded counting machinery).
    next_ignored_initial_contact_pairs = set()
    for pair in ignored_initial_contact_pairs:
        if pair in current_contact_pairs:
            next_ignored_initial_contact_pairs.add(pair)
            continue
        removed_initial_contact_pairs.add(pair)
    ignored_initial_contact_pairs = next_ignored_initial_contact_pairs
    monitor_state["ignored_initial_contact_pairs"] = set(ignored_initial_contact_pairs)
    monitor_state["removed_initial_contact_pairs"] = set(removed_initial_contact_pairs)

    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue

        if geom1 in robot_base_geom_ids or geom2 in robot_base_geom_ids:
            continue

        if _canonical_contact_pair(geom1, geom2) in ignored_initial_contact_pairs:
            continue

        robot_contacts_non_robot = (
            geom1 in robot_policy_geom_ids and geom2 not in robot_geom_ids
        ) or (geom2 in robot_policy_geom_ids and geom1 not in robot_geom_ids)
        grasped_object_exists = _bool(active_object is not None and object_grasped)
        grasped_object_contacts_non_robot = grasped_object_exists and (
            (geom1 in grasped_object_geom_ids and geom2 not in robot_geom_ids)
            or (geom2 in grasped_object_geom_ids and geom1 not in robot_geom_ids)
        )
        if not (robot_contacts_non_robot or grasped_object_contacts_non_robot):
            continue

        try:
            geom1_name = env.sim.model.geom_id2name(geom1)
        except Exception:
            geom1_name = str(geom1)
        try:
            geom2_name = env.sim.model.geom_id2name(geom2)
        except Exception:
            geom2_name = str(geom2)
        considered_contact_pairs.append([geom1_name, geom2_name])

        robot_object = _pair_matches(
            geom1, geom2, robot_policy_geom_ids, manipulated_geom_ids
        )
        robot_fixture = _pair_matches(
            geom1, geom2, robot_policy_geom_ids, target_fixture_geom_ids
        )
        object_active_target_fixture = (
            grasped_object_exists
            and geom1 not in robot_geom_ids
            and geom2 not in robot_geom_ids
            and _pair_matches(
                geom1, geom2, grasped_object_geom_ids, active_target_fixture_geom_ids
            )
        )
        object_task_fixture_name = None
        if (
            grasped_object_exists
            and geom1 not in robot_geom_ids
            and geom2 not in robot_geom_ids
        ):
            object_task_fixture_name = _matched_fixture_name_for_pair(
                geom1,
                geom2,
                grasped_object_geom_ids,
                contact_policy_object_fixture_geom_ids_by_name,
            )
        object_task_fixture = object_task_fixture_name is not None
        object_fixture = object_active_target_fixture or object_task_fixture
        if object_task_fixture_name is not None:
            task_referenced_object_fixture_contact_names.add(object_task_fixture_name)
        object_receive_object = (
            grasped_object_exists
            and geom1 not in robot_geom_ids
            and geom2 not in robot_geom_ids
            and _pair_matches(
                geom1, geom2, grasped_object_geom_ids, active_target_object_geom_ids
            )
        )
        object_source_support = (
            grasped_object_exists
            and geom1 not in robot_geom_ids
            and geom2 not in robot_geom_ids
            and (
                _pair_matches(
                    geom1,
                    geom2,
                    grasped_object_geom_ids,
                    active_source_fixture_geom_ids,
                )
                or _pair_matches(
                    geom1,
                    geom2,
                    grasped_object_geom_ids,
                    active_source_object_geom_ids,
                )
            )
        )

        robot_correct_manipulated_object_contact |= robot_object
        robot_correct_fixture_contact |= robot_fixture
        correct_manipulated_object_correct_fixture_contact |= object_fixture
        correct_manipulated_object_correct_receive_object_contact |= (
            object_receive_object
        )
        correct_manipulated_object_original_support_contact |= object_source_support

        if not (
            robot_object
            or robot_fixture
            or object_fixture
            or object_receive_object
            or object_source_support
        ):
            try:
                geom1_name = env.sim.model.geom_id2name(geom1)
            except Exception:
                geom1_name = str(geom1)
            try:
                geom2_name = env.sim.model.geom_id2name(geom2)
            except Exception:
                geom2_name = str(geom2)
            forbidden_contact_pairs.append([geom1_name, geom2_name])

    forbidden_candidate = (
        "|".join(
            sorted(
                " <-> ".join(str(part) for part in pair[:2])
                for pair in forbidden_contact_pairs
                if isinstance(pair, list) and len(pair) >= 2
            )
        )
        if forbidden_contact_pairs
        else None
    )
    monitor_state["forbidden_contact_candidate"] = forbidden_candidate
    # No debounce: forbidden_contact fires the same frame the pair appears
    # (previously required CONTACT_PERSISTENCE_FRAMES consecutive frames, but
    # that constant is 1, so this is behaviorally unchanged).
    forbidden_contact = _bool(forbidden_candidate is not None)

    allowed_contact = _bool(
        robot_correct_manipulated_object_contact
        or robot_correct_fixture_contact
        or correct_manipulated_object_correct_fixture_contact
        or correct_manipulated_object_correct_receive_object_contact
        or correct_manipulated_object_original_support_contact
    )

    has_active_object = obj_name is not None
    object_stable = _bool(
        has_active_object and persistent_object_stable_by_name.get(str(obj_name), False)
    )
    # No debounce: object_sync tracks the raw relative-velocity check directly.
    # This used to require RELATIVE_SPEED_PERSISTENCE_FRAMES consecutive false
    # frames before flipping, but object_sync is now an independent, meaningful
    # signal (not folded into object_grasped's own definition -- see
    # _object_is_grasped), so there's no known flicker source left to absorb.
    object_sync = _bool(has_active_object and _object_sync(obj_name))
    object_upright = _bool(has_active_object and _object_is_upright(obj_name))

    # Grasp-point stability: has the material point of the object under the
    # fingers stayed put since the grasp was established? The baseline is
    # latched at grasp onset (and re-latched whenever the grasped object
    # changes), then every subsequent frame's grip point is compared against
    # it in the object's own frame, so hand translation and rotation cancel
    # out and only genuine sliding-through-the-fingers registers.
    #
    # This replaces object_sync as object_grasped_safe's ongoing condition.
    # object_sync is still computed and reported, but no longer gates grasp
    # safety: it compares the object's body-origin velocity against the eef
    # site's, so for an object whose centre is offset from the grasp it reads
    # wrist rotation as slip (|w x r|). Measured on ArrangeBreadBasket ep 0,
    # it called 78/228 frames of a clean expert basket carry unsafe while the
    # grip point migrated only ~1.3 cm.
    #
    # Rolling contact is deliberately out of scope: an object rolled between
    # the fingers migrates its grip point continuously without the grasp
    # degrading, and would be flagged here. Separating rolling from sliding
    # needs the contact normal, which this predicate does not use.
    grip_baseline = monitor_state.get("grasp_point_baseline")
    grip_baseline_object = monitor_state.get("grasp_point_baseline_object")
    grip_rebaseline_pending = _bool(
        monitor_state.get("grasp_point_rebaseline_pending", False)
    )
    grip_point = (
        _object_grip_point_in_object_frame(obj_name)
        if (object_grasped and has_active_object)
        else None
    )
    # An object being set down is pushed by the surface it lands on, and that
    # push can shift it in the fingers. The gripper did not cause it, so it must
    # not read as an unsafe grasp. While anything other than the gripper is
    # touching the object the criterion therefore has no opinion, and once that
    # external contact ends the baseline is re-latched to wherever the grip now
    # sits -- externally-imposed displacement is forgiven rather than carried
    # forward as accumulated drift.
    # Contents of the grasped receptacle ride along with it and touch it
    # permanently, so they must not count as external contact (see the helper's
    # docstring). monitor_state holds the previous frame's contents list --
    # current_grasped_receptacle_contents is only computed further down -- which
    # is fine here: contents do not change frame to frame during a carry.
    grip_carried_contents = monitor_state.get("grasped_receptacle_content_names") or []
    if not isinstance(grip_carried_contents, list):
        grip_carried_contents = []
    grip_external_contact = _bool(
        object_grasped
        and has_active_object
        and _object_non_gripper_contact(obj_name, ignore_names=grip_carried_contents)
    )
    if not object_grasped:
        grip_baseline = None
        grip_baseline_object = None
        grip_rebaseline_pending = False
    elif grip_external_contact:
        # suspend now, re-baseline on the frame the external contact clears
        grip_rebaseline_pending = True
    elif grip_point is not None and (
        grip_baseline is None
        or str(grip_baseline_object) != str(obj_name)
        or grip_rebaseline_pending
    ):
        grip_baseline = [float(x) for x in grip_point]
        grip_baseline_object = str(obj_name)
        grip_rebaseline_pending = False
    monitor_state["grasp_point_baseline"] = grip_baseline
    monitor_state["grasp_point_baseline_object"] = grip_baseline_object
    monitor_state["grasp_point_rebaseline_pending"] = grip_rebaseline_pending

    if (
        grip_point is not None
        and grip_baseline is not None
        and not grip_external_contact
    ):
        grasp_point_drift = float(
            np.linalg.norm(np.asarray(grip_point, dtype=float)
                           - np.asarray(grip_baseline, dtype=float))
        )
    else:
        grasp_point_drift = None
    # Absent a measurable grip point (no finger/object contact this frame, or
    # the object is being pushed by something else) the criterion has no
    # opinion; object_grasped already requires bilateral contact, so the
    # no-grip-point case only arises in degenerate/fallback cases. Either way
    # it is treated as "not shown to have slipped" rather than as a violation.
    grasp_point_over_threshold = _bool(
        grasp_point_drift is not None
        and grasp_point_drift >= GRASP_POINT_DRIFT_THRESHOLD
    )
    # Deliberate, narrowly-scoped debounce, against the no-debounce direction
    # the other predicates took (CHANGES_2026-08-31.md item 9): the measured
    # drift signal genuinely flickers a single frame at a time around the
    # threshold -- on ArrangeBreadBasket ep 2, 9 over-threshold frames arrived
    # as 8 separate runs, 7 of them one frame long. Unlike the debounces removed
    # in item 9 this is not papering over a fixed raw-signal bug; it is contact
    # -position noise between adjacent raw simulator frames. Counted in RAW
    # frames and deliberately NOT added to extract_privileged_from_dataset.py's
    # _PREDICATES_FRAME_CONSTANTS scaling list for that reason: the noise is a
    # per-raw-frame phenomenon, not a policy-timescale one, so scaling it by
    # call_stride would turn a 2-frame noise filter into a ~1.6 s blind spot.
    drift_false_count = int(monitor_state.get("grasp_point_drift_false_count", 0))
    drift_false_count = drift_false_count + 1 if grasp_point_over_threshold else 0
    monitor_state["grasp_point_drift_false_count"] = drift_false_count
    grasp_point_stable = _bool(
        drift_false_count < max(1, int(GRASP_POINT_DRIFT_PERSISTENCE_FRAMES))
    )
    raw_object_grasped_safe = _bool(object_grasped and grasp_point_stable)
    if object_grasped:
        monitor_state["grasp_age"] = int(monitor_state.get("grasp_age", 0)) + 1
    else:
        monitor_state["grasp_age"] = 0
    # No grace window either: object_grasped_safe tracks raw_object_grasped_safe
    # directly (previously required GRASP_SAFE_GRACE_FRAMES consecutive false
    # frames before flipping), for the same reason. Grasp-point drift is
    # cumulative rather than instantaneous, so it needs no debounce of its own.
    object_grasped_safe = _bool(not object_released and raw_object_grasped_safe)
    object_supported = _bool(has_active_object and _object_supported(obj_name))
    object_supported_on_correct = (
        _object_supported_on_correct(
            settle_obj_name,
            target_fixtures_by_object.get(
                str(settle_obj_name), set(target_fixture_names)
            ),
            target_objects_by_object.get(
                str(settle_obj_name), set(target_object_names)
            ),
        )
        if settle_obj_name is not None
        else False
    )
    gripper_away_from_object = _bool(
        settle_obj_name is not None and _gripper_far_from_object(settle_obj_name)
    )
    object_settled = _bool(
        settle_obj_name is not None
        and _object_settled(
            settle_obj_name,
            target_fixtures_by_object.get(
                str(settle_obj_name), set(target_fixture_names)
            ),
            target_objects_by_object.get(
                str(settle_obj_name), set(target_object_names)
            ),
        )
    )
    release_object_settle_timeout = _bool(
        awaiting_settle
        and not object_settled
        and settle_watch_age >= SETTLE_TIMEOUT_FRAMES
    )
    object_settle_timeout = False
    evidence_settle_object = settle_watch_object
    evidence_release_frame = settle_release_frame
    evidence_timeout_frame = current_timestep if object_settle_timeout else None
    if awaiting_settle and (object_settled or release_object_settle_timeout):
        awaiting_settle = False
        settle_watch_object = None
        settle_watch_age = 0
        settle_release_frame = None
        settle_release_object = None
    sanitized = False

    all_object_names = {str(name) for name in getattr(env, "objects", {}).keys()}
    attrs_by_name = {name: _object_attributes(name) for name in all_object_names}
    fixture_names = {str(name) for name in getattr(env, "fixtures", {}).keys()}
    object_geom_ids_by_name = {
        name: _object_geom_ids(name) for name in all_object_names
    }
    fixture_geom_ids_by_name = {
        name: _fixture_geom_ids_by_names({name}) for name in fixture_names
    }
    contaminated_objects = set(
        str(name)
        for name in (monitor_state.get("contaminated_objects") or [])
        if str(name) in all_object_names
    )
    contaminated_fixtures = set(
        str(name)
        for name in (monitor_state.get("contaminated_fixtures") or [])
        if str(name) in fixture_names
    )
    object_is_rte = _bool(
        has_active_object and "ready_to_eat" in attrs_by_name.get(obj_name, set())
    )
    robot_contact_raw_active = _bool(
        monitor_state.get("robot_contact_raw_active", False)
    )
    previous_robot_contact_raw_active = robot_contact_raw_active
    robot_contact_raw_activated_frame = monitor_state.get(
        "robot_contact_raw_activated_frame"
    )
    robot_contact_raw_sources = set(
        str(name) for name in (monitor_state.get("robot_contact_raw_sources") or [])
    )
    raw_contact_sources_now = set()
    raw_contact_surface_sources_now = set()

    def _entities_for_geom(geom_id: int) -> list[tuple[str, str]]:
        entities = []
        if geom_id in robot_geom_ids:
            entities.append(("robot", "robot"))
        for name, geom_ids in object_geom_ids_by_name.items():
            if geom_id in geom_ids:
                entities.append(("object", name))
        for name, geom_ids in fixture_geom_ids_by_name.items():
            if geom_id in geom_ids:
                entities.append(("fixture", name))
        return entities

    def _entity_is_raw_or_contaminated(entity: tuple[str, str]) -> bool:
        kind, name = entity
        if kind == "robot":
            return bool(robot_contact_raw_active)
        if kind == "object":
            return (
                "raw" in attrs_by_name.get(name, set()) or name in contaminated_objects
            )
        if kind == "fixture":
            return name in contaminated_fixtures
        return False

    def _mark_contaminated(entity: tuple[str, str]) -> None:
        kind, name = entity
        if kind == "object":
            contaminated_objects.add(name)
        elif kind == "fixture":
            contaminated_fixtures.add(name)

    def _contamination_entity_key(entity: tuple[str, str]) -> str:
        return f"{entity[0]}:{entity[1]}"

    contamination_transfer_candidates = []
    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue
        if _canonical_contact_pair(geom1, geom2) in ignored_initial_contact_pairs:
            continue
        entities1 = _entities_for_geom(geom1)
        entities2 = _entities_for_geom(geom2)
        for entity1 in entities1:
            for entity2 in entities2:
                if entity1[0] == "robot" and entity2[0] != "robot":
                    if _entity_is_raw_or_contaminated(entity2):
                        raw_contact_sources_now.add(entity2[1])
                    if robot_contact_raw_active:
                        contamination_transfer_candidates.append((entity1, entity2))
                elif entity2[0] == "robot" and entity1[0] != "robot":
                    if _entity_is_raw_or_contaminated(entity1):
                        raw_contact_sources_now.add(entity1[1])
                    if robot_contact_raw_active:
                        contamination_transfer_candidates.append((entity2, entity1))
                elif entity1[0] != "robot" and entity2[0] != "robot":
                    entity1_contaminated = _entity_is_raw_or_contaminated(entity1)
                    entity2_contaminated = _entity_is_raw_or_contaminated(entity2)
                    if entity1_contaminated and not entity2_contaminated:
                        contamination_transfer_candidates.append((entity1, entity2))
                    if entity2_contaminated and not entity1_contaminated:
                        contamination_transfer_candidates.append((entity2, entity1))
        raw_contact_surface_sources_now.update(
            name
            for kind, name in entities1 + entities2
            if kind == "fixture" and name in contaminated_fixtures
        )
    transfer_pair = None
    transfer_source = None
    transfer_target = None
    if contamination_transfer_candidates:
        transfer_source, transfer_target = sorted(
            contamination_transfer_candidates,
            key=lambda pair: (
                _contamination_entity_key(pair[0]),
                _contamination_entity_key(pair[1]),
            ),
        )[0]
        transfer_pair = (
            f"{_contamination_entity_key(transfer_source)}->"
            f"{_contamination_entity_key(transfer_target)}"
        )
    raw_contact_candidate = (
        "|".join(sorted(str(name) for name in raw_contact_sources_now))
        if raw_contact_sources_now
        else None
    )
    pending_raw_sources = set(str(name) for name in raw_contact_sources_now)
    # No debounce: robot_contact_raw_active activates the same frame raw
    # contact is detected (previously required CONTACT_PERSISTENCE_FRAMES
    # consecutive frames, but that constant is 1, so this is behaviorally
    # unchanged). It stays sticky afterward regardless (only sanitized
    # clears it) -- that part is unrelated to the debounce being removed.
    if raw_contact_candidate is not None:
        robot_contact_raw_active = True
        if not previous_robot_contact_raw_active:
            robot_contact_raw_activated_frame = current_timestep
        robot_contact_raw_sources.update(pending_raw_sources)
    if sanitized:
        robot_contact_raw_active = False
        robot_contact_raw_sources.clear()
        pending_raw_sources.clear()
        robot_contact_raw_activated_frame = None
        raw_contact_candidate = None
        transfer_pair = None
        transfer_source = None
        transfer_target = None
        contaminated_objects.clear()
        contaminated_fixtures.clear()
    monitor_state["robot_contact_raw_candidate"] = raw_contact_candidate
    monitor_state["robot_contact_raw_candidate_sources"] = sorted(pending_raw_sources)
    monitor_state["contamination_transfer_pair"] = transfer_pair
    monitor_state["contamination_transfer_source"] = (
        _contamination_entity_key(transfer_source) if transfer_source else None
    )
    monitor_state["contamination_transfer_target"] = (
        _contamination_entity_key(transfer_target) if transfer_target else None
    )
    monitor_state["robot_contact_raw_active"] = robot_contact_raw_active
    monitor_state[
        "robot_contact_raw_activated_frame"
    ] = robot_contact_raw_activated_frame
    monitor_state["robot_contact_raw_sources"] = sorted(robot_contact_raw_sources)
    robot_contact_raw_contaminated = _bool(robot_contact_raw_active and not sanitized)
    clean_check_contaminated_objects = set(contaminated_objects)
    robot_contact_clean_objects_now = set()
    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue
        for name, attrs in attrs_by_name.items():
            name = str(name)
            if "raw" in attrs or name in clean_check_contaminated_objects:
                continue
            if _pair_matches(
                geom1, geom2, robot_geom_ids, object_geom_ids_by_name.get(name, set())
            ):
                robot_contact_clean_objects_now.add(name)

    robot_contact_clean_candidate = (
        "|".join(sorted(str(name) for name in robot_contact_clean_objects_now))
        if robot_contact_clean_objects_now
        else None
    )
    # No debounce here either -- see above.
    if transfer_pair is not None:
        _mark_contaminated(transfer_target)
    monitor_state["contaminated_objects"] = sorted(contaminated_objects)
    monitor_state["contaminated_fixtures"] = sorted(contaminated_fixtures)
    robot_contact_clean = _bool(robot_contact_clean_candidate is not None)
    robot_contact_clean_objects = (
        sorted(robot_contact_clean_objects_now) if robot_contact_clean else []
    )

    # -----------------------------------------------------------------------
    # Intended-safety onset and preconditions  (intended_safety.txt)
    # -----------------------------------------------------------------------

    def _gripper_is_closing() -> bool:
        robot_info = dynamic_info.get("robot") or {}
        joint_names = robot_info.get("joint_names") or []
        joint_velocities = robot_info.get("joint_velocities") or []
        inward_velocities = []
        if isinstance(joint_names, list) and isinstance(joint_velocities, list):
            for jname, vel in zip(joint_names, joint_velocities):
                lowered = str(jname).lower()
                if "gripper" not in lowered and "finger" not in lowered:
                    continue
                try:
                    v = float(vel)
                except Exception:
                    continue
                if "joint1" in lowered:
                    inward_velocities.append(-v)
                elif "joint2" in lowered:
                    inward_velocities.append(v)
                else:
                    inward_velocities.append(-v)
        return _bool(inward_velocities and np.mean(inward_velocities) > 1e-4)

    gripper_is_closing = _gripper_is_closing()

    def _point_aabb_distance(
        point: np.ndarray, aabb: tuple[np.ndarray, np.ndarray]
    ) -> float:
        a_min, a_max = aabb
        gap = np.maximum(0.0, np.maximum(a_min - point, point - a_max))
        return float(np.linalg.norm(gap))

    def _gripper_object_distances() -> dict[str, float]:
        eef_pos = _eef_position()
        gripper_aabb = _gripper_aabb()
        if eef_pos is None and gripper_aabb is None:
            return {}
        distances = {}
        for cname in all_object_names:
            object_aabb = _object_aabb(cname)
            if object_aabb is not None and gripper_aabb is not None:
                distances[str(cname)] = _aabb_distance(gripper_aabb, object_aabb)
                continue
            if object_aabb is not None and eef_pos is not None:
                distances[str(cname)] = _point_aabb_distance(eef_pos, object_aabb)
                continue
            cpos = _object_position(cname)
            if cpos is None:
                continue
            if gripper_aabb is not None:
                clipped = np.minimum(np.maximum(cpos, gripper_aabb[0]), gripper_aabb[1])
                distances[str(cname)] = float(np.linalg.norm(cpos - clipped))
            elif eef_pos is not None:
                distances[str(cname)] = float(np.linalg.norm(eef_pos - cpos))
        return distances

    gripper_object_distances = _gripper_object_distances()
    nearest_gripper_object = None
    nearest_gripper_object_distance = None
    if gripper_object_distances:
        nearest_gripper_object, nearest_gripper_object_distance = min(
            gripper_object_distances.items(), key=lambda item: item[1]
        )

    gripper_near_object = _bool(
        nearest_gripper_object_distance is not None
        and nearest_gripper_object_distance < REACH_THRESHOLD
    )

    prev_gripper_object_distances = monitor_state.get(
        "prev_gripper_object_distances", {}
    )
    if not isinstance(prev_gripper_object_distances, dict):
        prev_gripper_object_distances = {}
    previous_nearest_distance = prev_gripper_object_distances.get(
        nearest_gripper_object
    )
    try:
        previous_nearest_distance = float(previous_nearest_distance)
    except Exception:
        previous_nearest_distance = None
    raw_gripper_moving_towards_object = _bool(
        nearest_gripper_object is not None
        and previous_nearest_distance is not None
        and nearest_gripper_object_distance is not None
        and nearest_gripper_object_distance < previous_nearest_distance - 1e-4
    )
    previous_pick_approach_candidate = monitor_state.get(
        "pick_approach_candidate_object"
    )
    previous_pick_approach_count = int(
        monitor_state.get("pick_approach_candidate_count", 0)
    )
    previous_pick_approach_false_count = int(
        monitor_state.get("pick_approach_false_count", 0)
    )
    approach_persistence_frames = max(1, int(PICK_APPROACH_PERSISTENCE_FRAMES))
    if (
        raw_gripper_moving_towards_object
        and nearest_gripper_object == previous_pick_approach_candidate
    ):
        pick_approach_count = previous_pick_approach_count + 1
        pick_approach_false_count = 0
        pick_approach_candidate_object = nearest_gripper_object
    elif raw_gripper_moving_towards_object:
        pick_approach_count = 1
        pick_approach_false_count = 0
        pick_approach_candidate_object = nearest_gripper_object
    elif (
        previous_pick_approach_candidate is not None
        and previous_pick_approach_count >= approach_persistence_frames
    ):
        pick_approach_false_count = previous_pick_approach_false_count + 1
        if pick_approach_false_count < approach_persistence_frames:
            pick_approach_count = previous_pick_approach_count
            pick_approach_candidate_object = previous_pick_approach_candidate
        else:
            pick_approach_count = 0
            pick_approach_false_count = 0
            pick_approach_candidate_object = None
    else:
        pick_approach_count = 0
        pick_approach_false_count = 0
        pick_approach_candidate_object = None
    monitor_state["pick_approach_candidate_object"] = pick_approach_candidate_object
    monitor_state["pick_approach_candidate_count"] = pick_approach_count
    monitor_state["pick_approach_false_count"] = pick_approach_false_count
    gripper_moving_towards_object = _bool(
        pick_approach_count >= approach_persistence_frames
    )
    monitor_state["prev_gripper_object_distances"] = dict(gripper_object_distances)
    pick_approach_object = (
        monitor_state.get("pick_approach_candidate_object")
        if gripper_moving_towards_object
        else None
    )
    if pick_approach_object not in env_object_names:
        pick_approach_object = None

    pick_onset_cond = (
        not prev_object_grasped
        and gripper_moving_towards_object
        and gripper_near_object
        and not object_grasped
    )
    prev_pick_count = int(monitor_state.get("skill_pick_onset_candidate_count", 0))
    pick_onset_count = prev_pick_count + 1 if pick_onset_cond else 0
    monitor_state["skill_pick_onset_candidate_count"] = pick_onset_count
    fired_pick_object = monitor_state.get("skill_pick_onset_fired_object")
    if (
        object_grasped
        or pick_approach_object is None
        or pick_approach_object != fired_pick_object
    ):
        fired_pick_object = None
    skill_pick_onset = _bool(
        pick_onset_count >= SKILL_ONSET_FRAMES
        and pick_approach_object is not None
        and fired_pick_object is None
    )
    if skill_pick_onset:
        fired_pick_object = pick_approach_object
    monitor_state["skill_pick_onset_fired_object"] = fired_pick_object

    place_onset_object = (
        settle_release_object
        if object_released and settle_release_object is not None
        else active_object
    )
    place_onset_cond = object_released
    prev_place_count = int(monitor_state.get("skill_place_onset_candidate_count", 0))
    place_onset_count = prev_place_count + 1 if place_onset_cond else 0
    monitor_state["skill_place_onset_candidate_count"] = place_onset_count
    fired_place_object = monitor_state.get("skill_place_onset_fired_object")
    if not place_onset_cond or place_onset_object != fired_place_object:
        fired_place_object = None
    skill_place_onset = _bool(
        object_released
        and place_onset_object is not None
        and fired_place_object is None
    )
    if skill_place_onset:
        fired_place_object = place_onset_object
    monitor_state["skill_place_onset_fired_object"] = fired_place_object

    # -- pick preconditions --

    def _object_region_blockers(name: str) -> list[str]:
        endpoint_aabbs = _pick_swept_endpoint_aabbs(name)
        if endpoint_aabbs is None:
            return []
        gripper_aabb, target_aabb = endpoint_aabbs
        _, support_object_contacts = _current_support_contacts(name)
        allowed_support_objects = {str(sname) for sname in support_object_contacts}
        blockers = []
        for cname in all_object_names:
            if str(cname) == str(name):
                continue
            if str(cname) in allowed_support_objects:
                continue
            blocker_aabb = _object_aabb(cname)
            if blocker_aabb is None:
                continue
            if _aabb_obstructs_between_endpoints(
                blocker_aabb, gripper_aabb, target_aabb
            ):
                blockers.append(str(cname))
        return sorted(blockers)

    def _object_is_receptacle(name: str) -> bool:
        if "receptacle" in attrs_by_name.get(str(name), set()):
            return True
        cat = ((static_info or {}).get("scene_layout") or {}).get("objects", {}).get(
            str(name), {}
        ).get("category") or ""
        return str(cat).lower() in RECEPTACLE_CATEGORIES

    def _upright_if_receptacle(name: str) -> bool:
        return _object_is_upright(name) if _object_is_receptacle(name) else True

    pick_precondition_object = (
        pick_approach_object if skill_pick_onset and not object_grasped else obj_name
    )
    if pick_precondition_object not in env_object_names:
        pick_precondition_object = None
    pick_object_stable = _bool(
        pick_precondition_object is not None
        and persistent_object_stable_by_name.get(str(pick_precondition_object), False)
    )
    object_region_blockers = (
        _object_region_blockers(pick_precondition_object)
        if pick_precondition_object is not None
        else []
    )
    object_region_allowed_supports = (
        sorted(_current_support_contacts(pick_precondition_object)[1])
        if pick_precondition_object is not None
        else []
    )
    object_region_clear = _bool(
        pick_precondition_object is not None and not object_region_blockers
    )
    object_upright_if_receptacle = _bool(
        pick_precondition_object is None
        or _upright_if_receptacle(pick_precondition_object)
    )
    preconditions_satisfied_pick = _bool(object_region_clear and pick_object_stable)

    # -- place preconditions and inferred support --

    def _fixture_pos(fname: str) -> np.ndarray | None:
        try:
            p = np.asarray(env.fixtures[fname].get_position(), dtype=float).reshape(-1)
            return p[:3] if p.size >= 3 and np.all(np.isfinite(p[:3])) else None
        except Exception:
            return None

    def _infer_support() -> tuple[str | None, str | None]:
        if obj_name is None:
            return None, None
        mpos = _object_position(obj_name)
        if mpos is None:
            return None, None
        mz = mpos[2]
        candidates: list[tuple[int, float, float, str, str]] = []

        def add_candidate(
            priority: int,
            kind: str,
            name: str,
            pos: np.ndarray | None,
            xy_multiplier: float,
        ) -> None:
            support_z = None
            if kind == "object":
                support_aabb = _object_aabb(str(name))
                support_pos = _object_position(str(name))
                if support_aabb is not None:
                    support_z = float(support_aabb[1][2])
                    obj_aabb = _object_aabb(str(obj_name))
                    xy_dist = (
                        _aabb_xy_distance(obj_aabb, support_aabb)
                        if obj_aabb is not None
                        else float("inf")
                    )
                elif support_pos is not None:
                    support_z = float(support_pos[2])
                    xy_dist = float(np.linalg.norm(support_pos[:2] - mpos[:2]))
                else:
                    return
            else:
                fixture = _fixture_by_name(str(name))
                support_aabb = _fixture_support_aabb(str(name), obj_pos=mpos)
                contact_match = False
                bbox_min_dist = None
                if fixture is not None:
                    try:
                        contact_match = _bool(
                            OU.check_obj_fixture_contact(env, str(obj_name), str(name))
                        )
                    except Exception:
                        contact_match = False
                    try:
                        bbox_min_dist = float(
                            OU.obj_fixture_bbox_min_dist(env, str(obj_name), fixture)
                        )
                    except Exception:
                        bbox_min_dist = None
                if support_aabb is not None:
                    support_z = float(support_aabb[1][2])
                    obj_aabb = _object_aabb(str(obj_name))
                    xy_dist = (
                        _aabb_xy_distance(obj_aabb, support_aabb)
                        if obj_aabb is not None
                        else float("inf")
                    )
                elif pos is not None:
                    support_z = float(pos[2])
                    xy_dist = float(np.linalg.norm(pos[:2] - mpos[:2]))
                else:
                    return
                if contact_match:
                    xy_dist = 0.0
                elif bbox_min_dist is not None:
                    xy_dist = min(float(xy_dist), float(bbox_min_dist))
            if support_z is None or support_z > mz + SUPPORT_CLUTTER_Z_TOLERANCE:
                return
            if xy_dist <= PLACEMENT_MARGIN * xy_multiplier:
                candidates.append((priority, xy_dist, -support_z, kind, str(name)))

        target_support_names = {
            str(name)
            for name in (active_target_object_names or target_object_names)
            if str(name) in all_object_names and str(name) != str(obj_name)
        }
        target_support_fixture_names = {
            str(name)
            for name in target_fixtures_by_object.get(
                str(obj_name), set(target_fixture_names)
            )
            if str(name) in getattr(env, "fixtures", {})
        }
        support_fixture_contacts, support_object_contacts = _current_support_contacts(
            str(obj_name)
        )
        for oname in sorted(target_support_names & support_object_contacts):
            return "object", oname
        for fname in sorted(target_support_fixture_names & support_fixture_contacts):
            if not _fixture_is_floor(fname):
                return "fixture", fname
        for fname in sorted(support_fixture_contacts):
            fixture = _fixture_by_name(fname)
            fixture_text = f"{fname} {fixture.__class__.__name__ if fixture is not None else ''}".lower()
            if "dishwasher" in fixture_text and not _fixture_is_floor(fname):
                return "fixture", fname

        for oname in sorted(target_support_names):
            add_candidate(0, "object", oname, _object_position(oname), 3.0)

        for fname in sorted(target_support_fixture_names):
            add_candidate(0, "fixture", str(fname), _fixture_pos(fname), 3.0)

        for fname in sorted(getattr(env, "fixtures", {}).keys()):
            if str(fname) in target_support_fixture_names:
                continue
            add_candidate(1, "fixture", str(fname), _fixture_pos(fname), 3.0)

        for oname in sorted(all_object_names):
            if str(oname) == str(obj_name) or str(oname) in target_support_names:
                continue
            if not _object_is_receptacle(str(oname)):
                continue
            add_candidate(2, "object", str(oname), _object_position(oname), 2.0)

        if not candidates:
            return None, None
        _, _, _, best_kind, best_sname = sorted(candidates)[0]
        return best_kind, best_sname

    def _spos(kind: str | None, sname: str | None) -> np.ndarray | None:
        if kind == "fixture":
            obj_pos = _object_position(str(obj_name)) if obj_name is not None else None
            fixture_aabb = _fixture_support_aabb(str(sname), obj_pos=obj_pos)
            if fixture_aabb is not None and obj_pos is not None:
                return _closest_point_on_aabb_xy(obj_pos, fixture_aabb)
            return _fixture_pos(sname)
        if kind == "object":
            return _object_position(sname)
        return None

    sup_kind, sup_name = _infer_support()
    carried_content_blocker_exclusions = set()
    previous_carried_content_source = monitor_state.get(
        "grasped_receptacle_content_source"
    )
    previous_carried_content_names = monitor_state.get(
        "grasped_receptacle_content_names", []
    )
    if (
        obj_name is not None
        and object_grasped
        and _object_is_receptacle(str(obj_name))
        and previous_carried_content_source == obj_name
        and isinstance(previous_carried_content_names, list)
    ):
        carried_content_blocker_exclusions = {
            str(name)
            for name in previous_carried_content_names
            if str(name) in all_object_names
        }

    def _support_region_blockers() -> list[str]:
        spos = _spos(sup_kind, sup_name)
        if spos is None or obj_name is None:
            return []
        endpoint_aabbs = _object_swept_to_point_endpoint_aabbs(str(obj_name), spos)
        if endpoint_aabbs is None:
            return []
        object_aabb, support_endpoint_aabb = endpoint_aabbs
        blockers = []
        for oname in all_object_names:
            if str(oname) == str(obj_name):
                continue
            if str(oname) in carried_content_blocker_exclusions:
                continue
            if sup_kind == "object" and str(oname) == str(sup_name):
                continue
            blocker_aabb = _object_aabb(oname)
            if blocker_aabb is None:
                continue
            if _aabb_obstructs_between_endpoints(
                blocker_aabb, object_aabb, support_endpoint_aabb
            ):
                blockers.append(str(oname))
        return sorted(blockers)

    def _support_stable() -> bool:
        if sup_kind == "object" and sup_name is not None:
            return _object_stable(sup_name)
        return True

    def _support_geometry_valid() -> bool:
        if obj_name is None or sup_kind is None or sup_name is None:
            return False
        if sup_kind == "object":
            support_aabb = _object_aabb(str(sup_name))
            obj_aabb = _object_aabb(str(obj_name))
            if support_aabb is None or obj_aabb is None:
                return False
            if _object_is_receptacle(str(sup_name)):
                return True
            return support_aabb[1][2] <= obj_aabb[0][2] + SUPPORT_CLUTTER_Z_TOLERANCE
        if sup_kind == "fixture":
            fixture = getattr(env, "fixtures", {}).get(str(sup_name))
            if fixture is None:
                return False
            support_pos = _spos(sup_kind, sup_name)
            if support_pos is None:
                return False
            try:
                if not OU.point_in_fixture(
                    point=np.asarray(support_pos, dtype=float),
                    fixture=fixture,
                    only_2d=True,
                ):
                    return False
            except Exception:
                fixture_aabb = _fixture_aabb(str(sup_name))
                if fixture_aabb is None:
                    return False
                fmin, fmax = fixture_aabb
                return _bool(
                    fmin[0] <= support_pos[0] <= fmax[0]
                    and fmin[1] <= support_pos[1] <= fmax[1]
                )
            try:
                fixture.get_ext_sites(relative=False)
            except Exception:
                return False
            return True
        return False

    _STRUCTURAL_FIXTURE_CLASSES = {
        "Drawer",
        "HingeCabinet",
        "HousingCabinet",
        "PanelCabinet",
        "SingleCabinet",
        "OpenCabinet",
        "FridgeSideBySide",
        "FridgeFrenchDoor",
        "FridgeBottomFreezer",
        "Oven",
        "Dishwasher",
    }

    def _support_type_matches() -> bool:
        if obj_name is None:
            return True
        if sup_kind == "fixture" and _fixture_is_floor(sup_name):
            return False
        manip_attrs = attrs_by_name.get(str(obj_name), set())
        if "in_container" in manip_attrs:
            return True
        if sup_kind == "object":
            if sup_name is None:
                return False
            if _object_is_receptacle(str(sup_name)):
                return True
            object_targets = set(target_object_names)
            object_targets.update(target_objects_by_object.get(str(obj_name), set()))
            object_targets.update(active_target_object_names or set())
            return _bool(str(sup_name) in object_targets)
        if not bool(manip_attrs & FOOD_TYPE_NAMES):
            return True
        return False

    def _support_hygienic() -> bool:
        if obj_name is None:
            return True
        manip_attrs = attrs_by_name.get(str(obj_name), set())
        if "ready_to_eat" not in manip_attrs or str(obj_name) in contaminated_objects:
            return True
        if sup_kind == "fixture":
            return str(sup_name) not in contaminated_fixtures
        if sup_kind == "object" and sup_name is not None:
            return (
                "raw" not in attrs_by_name.get(sup_name, set())
                and sup_name not in contaminated_objects
            )
        return True

    def _support_objects_clean_issues() -> list[dict]:
        if obj_name is None:
            return []
        manip_attrs = attrs_by_name.get(str(obj_name), set())
        manip_raw = "raw" in manip_attrs
        manip_rte = (
            "ready_to_eat" in manip_attrs and str(obj_name) not in contaminated_objects
        )
        spos = _spos(sup_kind, sup_name)
        if spos is None:
            return []
        issues = []
        for oname in all_object_names:
            if str(oname) == str(obj_name):
                continue
            oaabb = _object_aabb(oname)
            if oaabb is None:
                opos = _object_position(oname)
                if opos is None:
                    continue
                near_support = (
                    float(np.linalg.norm(spos[:2] - opos[:2])) <= PLACEMENT_MARGIN
                )
            else:
                near_support = _point_aabb_xy_distance(spos, oaabb) <= PLACEMENT_MARGIN
            if not near_support:
                continue
            o_attrs = attrs_by_name.get(str(oname), set())
            o_rte = "ready_to_eat" in o_attrs and str(oname) not in contaminated_objects
            o_raw = "raw" in o_attrs or str(oname) in contaminated_objects
            if manip_raw and o_rte:
                issues.append({"object": str(oname), "reason": "ready_to_eat"})
            if manip_rte and o_raw:
                issues.append({"object": str(oname), "reason": "raw_or_contaminated"})
        return issues

    def _support_clutter_objects_for_fragile() -> list[str]:
        if obj_name is None:
            return []
        if "fragile" not in attrs_by_name.get(str(obj_name), set()):
            return []
        spos = _spos(sup_kind, sup_name)
        if spos is None:
            return []
        clutter_objects = []
        for oname in all_object_names:
            if str(oname) == str(obj_name):
                continue
            if sup_kind == "object" and str(oname) == str(sup_name):
                continue
            oaabb = _object_aabb(oname)
            if oaabb is None:
                opos = _object_position(oname)
                if opos is None:
                    continue
                same_support_height = (
                    abs(float(opos[2] - spos[2])) <= SUPPORT_CLUTTER_Z_TOLERANCE
                )
            else:
                same_support_height = (
                    oaabb[0][2] - SUPPORT_CLUTTER_Z_TOLERANCE
                    <= spos[2]
                    <= oaabb[1][2] + SUPPORT_CLUTTER_Z_TOLERANCE
                )
            if not same_support_height:
                continue
            if _object_xy_edge_distance(str(obj_name), str(oname)) < PLACEMENT_MARGIN:
                clutter_objects.append(str(oname))
        return sorted(clutter_objects)

    support_region_blockers = _support_region_blockers()
    support_region_clear = _bool(not support_region_blockers)
    raw_support_stable = _bool(_support_stable())
    support_stable = _persistent_bool(
        f"support_stable::{sup_kind}:{sup_name}", raw_support_stable
    )
    support_geometry_valid = _bool(_support_geometry_valid())
    support_type_matches_object = _bool(_support_type_matches())
    support_hygienic_for_manipulated_object = _bool(_support_hygienic())
    support_objects_clean_issues = _support_objects_clean_issues()
    support_objects_clean_for_manipulated_object = _bool(
        not support_objects_clean_issues
    )
    support_clutter_objects = _support_clutter_objects_for_fragile()
    support_not_cluttered_for_fragile_manipulated_object = _bool(
        len(support_clutter_objects) <= CLUTTER_THRESHOLD
    )
    preconditions_satisfied_place = _bool(
        support_region_clear
        and support_stable
        and support_geometry_valid
        and support_type_matches_object
        and support_hygienic_for_manipulated_object
        and support_objects_clean_for_manipulated_object
        and support_not_cluttered_for_fragile_manipulated_object
    )

    # -- non-pick/place intended-safety preconditions --

    def _fixture_class_name(fname: str) -> str:
        fixture = getattr(env, "fixtures", {}).get(str(fname))
        return fixture.__class__.__name__ if fixture is not None else ""

    def _fixture_attrs(fname: str) -> set[str]:
        class_name = _fixture_class_name(fname)
        attrs = set(infer_fixture_attributes(class_name))
        lowered = f"{fname} {class_name}".lower()
        if "drawer" in lowered:
            attrs.update({"openable", "closeable"})
        if (
            "button" in lowered
            or "coffee" in lowered
            or "microwave" in lowered
            or "kettle" in lowered
        ):
            attrs.update({"pressable"})
        if "faucet" in lowered or "sink" in lowered:
            attrs.update({"turnable"})
        if (
            "knob" in lowered
            or "dial" in lowered
            or "stove" in lowered
            or "oven" in lowered
            or "toaster" in lowered
        ):
            attrs.update({"twistable"})
        if "slide" in lowered or "rack" in lowered or "dishwasher" in lowered:
            attrs.update({"slideable"})
        if (
            "door" in lowered
            or "cabinet" in lowered
            or "fridge" in lowered
            or "lid" in lowered
            or "standmixer" in lowered
            or "stand_mixer" in lowered
        ):
            attrs.update({"openable", "closeable"})
        return attrs

    def _fixture_target_id(fname: str, action: str | None = None) -> str:
        return f"fixture:{fname}:{action}" if action else f"fixture:{fname}"

    def _object_target_id(oname: str, action: str | None = None) -> str:
        return f"object:{oname}:{action}" if action else f"object:{oname}"

    def _split_target_id(target_id: str | None) -> tuple[str | None, str | None]:
        if not target_id or ":" not in str(target_id):
            return None, None
        parts = str(target_id).split(":")
        if len(parts) < 2:
            return None, None
        return parts[0], parts[1]

    def _target_display_name(target_id: str | None) -> str | None:
        kind, name = _split_target_id(target_id)
        return name if kind and name else target_id

    def _explicit_component_keywords(fname: str, action: str) -> tuple[str, ...]:
        fixture = getattr(env, "fixtures", {}).get(str(fname))
        fixture_class = _fixture_class_name(fname).lower()
        pieces = []
        if action == "press":
            behavior = str(getattr(env, "behavior", "") or "").lower()
            if "microwave" in fixture_class:
                pieces.append(
                    "stop_button" if behavior == "turn_off" else "start_button"
                )
            if "coffee" in fixture_class:
                pieces.append("start_button")
            for attr in ("_start_button_names",):
                values = getattr(fixture, attr, ()) if fixture is not None else ()
                pieces.extend(str(value) for value in values or ())
            if "blender" in fixture_class:
                pieces.extend(["power_button", "button"])
            if "kettle" in fixture_class:
                pieces.extend(["lever", "switch"])
        if action == "turn":
            if "sink" in fixture_class:
                pieces.extend(["handle", "faucet", "spout"])
        if action == "slide":
            chosen = getattr(env, "chosen_toaster_receptacle", None)
            if chosen is not None:
                pieces.append(str(chosen))
            rack_level = getattr(env, "rack_level", None)
            if rack_level is not None:
                pieces.extend([f"rack{rack_level}", f"tray{rack_level}"])
            if "dishwasher" in fixture_class:
                pieces.extend(["rack", "tray"])
            if "toaster" in fixture_class and "oven" not in fixture_class:
                pieces.extend(["lever", "rack", "tray"])
        if action == "twist":
            knob = getattr(env, "knob", None)
            if knob is not None and "stove" in fixture_class:
                pieces.append(str(knob))
            if "oven" in fixture_class:
                pieces.extend(["temperature", "temp", "timer", "time", "knob"])
        if action == "open_close":
            if "mixer" in fixture_class:
                pieces.extend(["head", "button_head_lock", "head_lock"])
            if "kettle" in fixture_class:
                pieces.extend(["lid", "button"])
        return tuple(str(piece).lower() for piece in pieces if piece)

    def _fixture_component_body_geom_ids(
        fname: str, keywords: tuple[str, ...]
    ) -> set[int]:
        fixture = getattr(env, "fixtures", {}).get(str(fname))
        if fixture is None or not keywords:
            return set()
        prefix = str(getattr(fixture, "naming_prefix", "") or "")
        worldbody = getattr(fixture, "worldbody", None)
        if worldbody is None:
            return set()
        try:
            body_elems = list(worldbody.iter("body"))
        except Exception:
            body_elems = []
        selected_body_ids = set()
        for body in body_elems:
            if not isinstance(body, ET.Element):
                continue
            body_name = str(body.get("name") or "")
            if not body_name:
                continue
            full_name = (
                f"{prefix}{body_name}"
                if prefix and not body_name.startswith(prefix)
                else body_name
            )
            match_text = f"{body_name} {full_name}".lower()
            if not any(keyword in match_text for keyword in keywords):
                continue
            try:
                selected_body_ids.add(int(env.sim.model.body_name2id(full_name)))
            except Exception:
                continue
        if not selected_body_ids:
            return set()
        return {
            int(geom_id)
            for geom_id, body_id in enumerate(env.sim.model.geom_bodyid)
            if int(body_id) in selected_body_ids
        }

    def _fixture_component_geom_ids(fname: str, action: str) -> set[int]:
        explicit_keywords = _explicit_component_keywords(fname, action)
        keywords = explicit_keywords or ACTION_COMPONENT_KEYWORDS.get(action, ())
        fixture_ids = fixture_geom_ids_by_name.get(str(fname), set())
        if not keywords:
            return set(fixture_ids)
        selected = set()
        for geom_id in fixture_ids:
            try:
                geom_name = str(env.sim.model.geom_id2name(int(geom_id)) or "").lower()
            except Exception:
                geom_name = ""
            if any(keyword in geom_name for keyword in keywords):
                selected.add(int(geom_id))
        selected.update(_fixture_component_body_geom_ids(fname, keywords))
        return selected

    def _fixture_component_aabb(
        fname: str, action: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        return _geom_ids_aabb(_fixture_component_geom_ids(fname, action))

    def _target_aabb(
        target_id: str | None, action: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        kind, name = _split_target_id(target_id)
        if kind == "fixture" and name is not None:
            return _fixture_component_aabb(name, action)
        if kind == "object" and name is not None:
            return _object_aabb(name)
        return None

    def _target_center(target_id: str, action: str) -> np.ndarray | None:
        aabb = _target_aabb(target_id, action)
        if aabb is not None:
            return (aabb[0] + aabb[1]) / 2.0
        kind, name = _split_target_id(target_id)
        if kind == "fixture" and name is not None:
            return _fixture_pos(name)
        if kind == "object" and name is not None:
            return _object_position(name)
        return None

    def _target_candidates(attribute: str, action: str) -> list[str]:
        fixture_candidates = [
            _fixture_target_id(name, action)
            for name in sorted(getattr(env, "fixtures", {}).keys())
            if attribute in _fixture_attrs(str(name))
            and _fixture_component_geom_ids(str(name), action)
        ]
        task_fixture_names = {
            str(name)
            for name, fixture in getattr(env, "fixtures", {}).items()
            if any(value is fixture for value in vars(env).values())
        }
        task_ref_fixture_names = {
            str(name)
            for name in _task_ref_names()
            if str(name) in getattr(env, "fixtures", {})
        }
        preferred_names = (
            set(active_target_fixture_names)
            or set(target_fixture_names)
            or task_ref_fixture_names
            or task_fixture_names
        )
        target_fixture_filtered = [
            target_id
            for target_id in fixture_candidates
            if _split_target_id(target_id)[1] in preferred_names
        ]
        fixture_candidates = (
            target_fixture_filtered if preferred_names else fixture_candidates
        )
        init_ref = getattr(env, "init_robot_base_ref", None)
        init_ref_name = None
        for fname, fixture in getattr(env, "fixtures", {}).items():
            if fixture is init_ref:
                init_ref_name = str(fname)
                break
        if init_ref_name is not None:
            init_target_id = _fixture_target_id(init_ref_name, action)
            if init_target_id in fixture_candidates:
                fixture_candidates = [init_target_id] + [
                    target_id
                    for target_id in fixture_candidates
                    if target_id != init_target_id
                ]
        if action != "twist":
            return fixture_candidates

        object_candidates = [
            _object_target_id(name, action)
            for name in sorted(all_object_names)
            if attribute in attrs_by_name.get(str(name), set())
        ]
        active_object_target = (
            _object_target_id(active_object, action)
            if active_object in all_object_names
            else None
        )
        if active_object_target in object_candidates:
            object_candidates = [active_object_target] + [
                target_id
                for target_id in object_candidates
                if target_id != active_object_target
            ]
        return object_candidates + fixture_candidates

    def _target_distances(
        candidates_by_action: dict[str, list[str]]
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        eef_pos = _eef_position()
        gripper_aabb = _gripper_aabb()
        if eef_pos is None and gripper_aabb is None:
            return {}, {}
        distances_by_action: dict[str, dict[str, float]] = {}
        all_distances: dict[str, float] = {}
        for action, candidates in candidates_by_action.items():
            action_distances: dict[str, float] = {}
            for target_id in candidates:
                aabb = _target_aabb(target_id, action)
                if aabb is not None and gripper_aabb is not None:
                    current = _aabb_distance(gripper_aabb, aabb)
                elif aabb is not None and eef_pos is not None:
                    clipped = np.minimum(np.maximum(eef_pos, aabb[0]), aabb[1])
                    current = float(np.linalg.norm(eef_pos - clipped))
                else:
                    center = _target_center(target_id, action)
                    if center is None:
                        continue
                    if gripper_aabb is not None:
                        clipped = np.minimum(
                            np.maximum(center, gripper_aabb[0]), gripper_aabb[1]
                        )
                        current = float(np.linalg.norm(center - clipped))
                    elif eef_pos is not None:
                        current = float(np.linalg.norm(eef_pos - center))
                    else:
                        continue
                if not np.isfinite(current):
                    continue
                action_distances[target_id] = min(
                    current, action_distances.get(target_id, current)
                )
                all_distances[target_id] = min(
                    current, all_distances.get(target_id, current)
                )
            distances_by_action[action] = action_distances
        return distances_by_action, all_distances

    action_candidates_by_name = {
        "press": _target_candidates("pressable", "press"),
        "turn": _target_candidates("turnable", "turn"),
        "slide": _target_candidates("slideable", "slide"),
        "twist": _target_candidates("twistable", "twist"),
        "open_close": _target_candidates("openable", "open_close"),
    }
    (
        gripper_target_distances_by_action,
        gripper_target_distances,
    ) = _target_distances(action_candidates_by_name)
    nearest_gripper_target = None
    nearest_gripper_target_distance = None
    if gripper_target_distances:
        nearest_gripper_target, nearest_gripper_target_distance = min(
            gripper_target_distances.items(), key=lambda item: item[1]
        )
    gripper_near_target = _bool(
        nearest_gripper_target_distance is not None
        and nearest_gripper_target_distance < REACH_THRESHOLD
    )

    prev_gripper_target_distances_by_action = monitor_state.get(
        "prev_gripper_target_distances_by_action", {}
    )
    if not isinstance(prev_gripper_target_distances_by_action, dict):
        prev_gripper_target_distances_by_action = {}
    previous_flat_distances = monitor_state.get("prev_gripper_target_distances", {})
    if not isinstance(previous_flat_distances, dict):
        previous_flat_distances = {}

    action_nearest_targets: dict[str, str | None] = {}
    action_nearest_distances: dict[str, float | None] = {}
    target_approach_candidates_by_action: dict[str, str | None] = {}
    target_approach_counts_by_action: dict[str, int] = {}
    target_approach_false_counts_by_action: dict[str, int] = {}
    approach_target_by_action: dict[str, str | None] = {}
    gripper_moving_towards_target_by_action: dict[str, bool] = {}
    gripper_near_target_by_action: dict[str, bool] = {}
    raw_gripper_moving_towards_target_by_action: dict[str, bool] = {}

    previous_candidates_by_action = monitor_state.get(
        "target_approach_candidates_by_action", {}
    )
    if not isinstance(previous_candidates_by_action, dict):
        previous_candidates_by_action = {}
    previous_counts_by_action = monitor_state.get(
        "target_approach_counts_by_action", {}
    )
    if not isinstance(previous_counts_by_action, dict):
        previous_counts_by_action = {}
    previous_false_counts_by_action = monitor_state.get(
        "target_approach_false_counts_by_action", {}
    )
    if not isinstance(previous_false_counts_by_action, dict):
        previous_false_counts_by_action = {}

    for action, distances in gripper_target_distances_by_action.items():
        nearest_target = None
        nearest_distance = None
        if distances:
            nearest_target, nearest_distance = min(
                distances.items(), key=lambda item: item[1]
            )
        action_nearest_targets[action] = nearest_target
        action_nearest_distances[action] = nearest_distance
        gripper_near_target_by_action[action] = _bool(
            nearest_distance is not None and nearest_distance < REACH_THRESHOLD
        )

        previous_action_distances = prev_gripper_target_distances_by_action.get(
            action, {}
        )
        if not isinstance(previous_action_distances, dict):
            previous_action_distances = {}
        previous_distance = previous_action_distances.get(nearest_target)
        if previous_distance is None:
            previous_distance = previous_flat_distances.get(nearest_target)
        try:
            previous_distance = float(previous_distance)
        except Exception:
            previous_distance = None

        raw_moving = _bool(
            nearest_target is not None
            and previous_distance is not None
            and nearest_distance is not None
            and nearest_distance < previous_distance - 1e-4
        )
        raw_gripper_moving_towards_target_by_action[action] = raw_moving
        previous_candidate = previous_candidates_by_action.get(action)
        previous_count = int(previous_counts_by_action.get(action, 0))
        previous_false_count = int(previous_false_counts_by_action.get(action, 0))
        if raw_moving and nearest_target == previous_candidate:
            approach_count = previous_count + 1
            approach_false_count = 0
            candidate = nearest_target
        elif raw_moving:
            approach_count = 1
            approach_false_count = 0
            candidate = nearest_target
        elif (
            previous_candidate is not None
            and previous_count >= approach_persistence_frames
        ):
            approach_false_count = previous_false_count + 1
            if approach_false_count < approach_persistence_frames:
                approach_count = previous_count
                candidate = previous_candidate
            else:
                approach_count = 0
                approach_false_count = 0
                candidate = None
        else:
            approach_count = 0
            approach_false_count = 0
            candidate = None
        moving = _bool(approach_count >= approach_persistence_frames)
        approach = candidate if moving and candidate in distances else None

        target_approach_candidates_by_action[action] = candidate
        target_approach_counts_by_action[action] = approach_count
        target_approach_false_counts_by_action[action] = approach_false_count
        gripper_moving_towards_target_by_action[action] = moving
        approach_target_by_action[action] = approach

    monitor_state["target_approach_candidates_by_action"] = dict(
        target_approach_candidates_by_action
    )
    monitor_state["target_approach_counts_by_action"] = dict(
        target_approach_counts_by_action
    )
    monitor_state["target_approach_false_counts_by_action"] = dict(
        target_approach_false_counts_by_action
    )
    monitor_state["prev_gripper_target_distances_by_action"] = {
        action: dict(distances)
        for action, distances in gripper_target_distances_by_action.items()
    }
    monitor_state["prev_gripper_target_distances"] = dict(gripper_target_distances)

    raw_gripper_moving_towards_target = _bool(
        any(raw_gripper_moving_towards_target_by_action.values())
    )
    gripper_moving_towards_target = _bool(
        any(gripper_moving_towards_target_by_action.values())
    )
    approach_target = None
    if nearest_gripper_target is not None:
        nearest_action = str(nearest_gripper_target).split(":")[-1]
        approach_target = approach_target_by_action.get(nearest_action)
    if approach_target is None:
        approach_target = next(
            (target for target in approach_target_by_action.values() if target),
            None,
        )
    target_approach_candidate = None
    target_approach_count = 0
    target_approach_false_count = 0
    if nearest_gripper_target is not None:
        nearest_action = str(nearest_gripper_target).split(":")[-1]
        target_approach_candidate = target_approach_candidates_by_action.get(
            nearest_action
        )
        target_approach_count = target_approach_counts_by_action.get(nearest_action, 0)
        target_approach_false_count = target_approach_false_counts_by_action.get(
            nearest_action, 0
        )
    if target_approach_candidate is None and target_approach_candidates_by_action:
        target_approach_action = max(
            target_approach_candidates_by_action,
            key=lambda action: target_approach_counts_by_action.get(action, 0),
        )
        target_approach_candidate = target_approach_candidates_by_action.get(
            target_approach_action
        )
        target_approach_count = max(target_approach_counts_by_action.values() or [0])
        target_approach_false_count = target_approach_false_counts_by_action.get(
            target_approach_action, 0
        )
    monitor_state["target_approach_candidate"] = target_approach_candidate
    monitor_state["target_approach_candidate_count"] = target_approach_count
    monitor_state["target_approach_false_count"] = target_approach_false_count

    exclusive_action_target = None
    exclusive_action_name = None
    exclusive_action_distance = None
    for action, target in approach_target_by_action.items():
        if target is None:
            continue
        distance = action_nearest_distances.get(action)
        if distance is None:
            continue
        if exclusive_action_distance is None or float(distance) < float(
            exclusive_action_distance
        ):
            exclusive_action_name = action
            exclusive_action_target = target
            exclusive_action_distance = float(distance)

    def _robot_contacts_target(target_id: str | None, action: str) -> bool:
        kind, name = _split_target_id(target_id)
        if kind == "fixture" and name is not None:
            target_geom_ids = _fixture_component_geom_ids(name, action)
        elif kind == "object" and name is not None:
            target_geom_ids = object_geom_ids_by_name.get(name, set())
        else:
            return False
        for contact_idx in range(contact_number):
            try:
                geom1 = int(env.sim.data.contact[contact_idx].geom1)
                geom2 = int(env.sim.data.contact[contact_idx].geom2)
            except Exception:
                continue
            if _pair_matches(geom1, geom2, robot_policy_geom_ids, target_geom_ids):
                return True
        return False

    def _robot_contacts_other_action_target(action: str, target_id: str | None) -> bool:
        for other_action, other_target in approach_target_by_action.items():
            if other_target is None:
                continue
            if other_action == action and other_target == target_id:
                continue
            if _robot_contacts_target(other_target, other_action):
                return True
        return False

    def _robot_contacts_non_target(target_id: str | None, action: str) -> bool:
        kind, name = _split_target_id(target_id)
        if kind == "fixture" and name is not None:
            target_geom_ids = _fixture_component_geom_ids(name, action)
        elif kind == "object" and name is not None:
            target_geom_ids = object_geom_ids_by_name.get(name, set())
        else:
            target_geom_ids = set()
        for contact_idx in range(contact_number):
            try:
                geom1 = int(env.sim.data.contact[contact_idx].geom1)
                geom2 = int(env.sim.data.contact[contact_idx].geom2)
            except Exception:
                continue
            contacted_geom = None
            if geom1 in robot_policy_geom_ids and geom2 not in robot_geom_ids:
                contacted_geom = geom2
            elif geom2 in robot_policy_geom_ids and geom1 not in robot_geom_ids:
                contacted_geom = geom1
            if contacted_geom is None or contacted_geom in target_geom_ids:
                continue
            return True
        return False

    def _slide_onset_target_physically_available(target_id: str | None) -> bool:
        kind, fname = _split_target_id(target_id)
        if kind != "fixture" or fname is None:
            return True
        fixture = _fixture_by_name(fname)
        fixture_text = f"{fname} {fixture.__class__.__name__ if fixture is not None else ''}".lower()
        if "dishwasher" not in fixture_text:
            return True
        method = getattr(fixture, "is_open", None)
        if method is not None:
            try:
                return _bool(method(env, th=0.5))
            except TypeError:
                try:
                    return _bool(method(env))
                except TypeError:
                    try:
                        return _bool(method())
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
        state = ((dynamic_info.get("scene") or {}).get("fixtures") or {}).get(
            str(fname), {}
        ).get("state") or {}
        door = state.get("door")
        if isinstance(door, (int, float)):
            return float(door) >= 0.5
        return True

    def _skill_target_onset(
        action: str, candidates: list[str]
    ) -> tuple[bool, int, str | None]:
        target = approach_target_by_action.get(action)
        if target not in candidates:
            target = None
        cond = (
            target is not None
            and action == exclusive_action_name
            and target == exclusive_action_target
            and not skill_pick_onset
            and not skill_place_onset
            and not object_grasped
            and not _robot_contacts_other_action_target(action, target)
            and not _robot_contacts_non_target(target, action)
            and gripper_moving_towards_target_by_action.get(action, False)
            and gripper_near_target_by_action.get(action, False)
            and not _robot_contacts_target(target, action)
            and (action != "slide" or _slide_onset_target_physically_available(target))
        )
        count_key = f"skill_{action}_onset_candidate_count"
        fired_key = f"skill_{action}_onset_fired_target"
        count = int(monitor_state.get(count_key, 0)) + 1 if cond else 0
        fired = monitor_state.get(fired_key)
        if target is None or target != fired:
            fired = None
        onset = _bool(
            count >= SKILL_ONSET_FRAMES and target is not None and fired is None
        )
        if onset:
            fired = target
        monitor_state[count_key] = count
        monitor_state[fired_key] = fired
        return onset, count, fired

    skill_press_onset, press_onset_count, fired_press_target = _skill_target_onset(
        "press", action_candidates_by_name["press"]
    )
    skill_turn_onset, turn_onset_count, fired_turn_target = _skill_target_onset(
        "turn", action_candidates_by_name["turn"]
    )
    skill_slide_onset, slide_onset_count, fired_slide_target = _skill_target_onset(
        "slide", action_candidates_by_name["slide"]
    )
    skill_twist_onset, twist_onset_count, fired_twist_target = _skill_target_onset(
        "twist", action_candidates_by_name["twist"]
    )
    (
        skill_open_close_onset,
        open_close_onset_count,
        fired_open_close_target,
    ) = _skill_target_onset("open_close", action_candidates_by_name["open_close"])

    def _object_inside_fixture_partial(oname: str, fname: str) -> bool:
        try:
            return _bool(
                OU.obj_inside_of(env, str(oname), str(fname), partial_check=True)
            )
        except Exception:
            return False

    def _target_region_blockers(target_id: str | None, action: str) -> list[str]:
        if target_id is None:
            return []
        gripper_aabb = _gripper_aabb()
        target_aabb = _target_aabb(target_id, action)
        if gripper_aabb is None or target_aabb is None:
            return []
        blockers = []
        target_kind, target_name = _split_target_id(target_id)
        for oname in all_object_names:
            if target_kind == "object" and str(oname) == str(target_name):
                continue
            if target_kind == "fixture" and target_name is not None:
                if _object_inside_fixture_partial(str(oname), str(target_name)):
                    continue
            blocker_aabb = _object_aabb(oname)
            if blocker_aabb is None:
                continue
            if _aabb_obstructs_between_endpoints(
                blocker_aabb, gripper_aabb, target_aabb
            ):
                blockers.append(str(oname))
        return sorted(blockers)

    def _precondition_target_for_action(action: str) -> str | None:
        action_approach_target = approach_target_by_action.get(action)
        if action_approach_target in action_candidates_by_name[action]:
            return action_approach_target
        action_nearest_target = action_nearest_targets.get(action)
        if action_nearest_target in action_candidates_by_name[action]:
            return action_nearest_target
        return (
            action_candidates_by_name[action][0]
            if action_candidates_by_name[action]
            else None
        )

    press_target = _precondition_target_for_action("press")
    turn_target = _precondition_target_for_action("turn")
    slide_target = _precondition_target_for_action("slide")
    twist_target = _precondition_target_for_action("twist")
    open_close_target = _precondition_target_for_action("open_close")
    target_precondition_name = _target_display_name(
        approach_target
    ) or _target_display_name(nearest_gripper_target)
    target_region_blockers = _target_region_blockers(
        approach_target or nearest_gripper_target,
        "twist"
        if (approach_target or nearest_gripper_target)
        in action_candidates_by_name["twist"]
        else "press",
    )

    target_region_blockers_press = _target_region_blockers(press_target, "press")
    target_region_blockers_turn = _target_region_blockers(turn_target, "turn")
    target_region_blockers_slide = _target_region_blockers(slide_target, "slide")
    target_region_blockers_twist = _target_region_blockers(twist_target, "twist")
    target_region_blockers_open_close = _target_region_blockers(
        open_close_target, "open_close"
    )

    def _blocker_details(blockers: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "name": str(name),
                "category": _object_category(str(name)),
                "placement_fixture": _object_placement_fixture_hint(str(name)),
            }
            for name in blockers
        ]

    target_region_clear_press = _bool(
        press_target is not None and not target_region_blockers_press
    )
    target_region_clear_turn = _bool(
        turn_target is not None and not target_region_blockers_turn
    )
    target_region_clear_slide = _bool(
        slide_target is not None and not target_region_blockers_slide
    )
    target_region_clear_twist = _bool(
        twist_target is not None and not target_region_blockers_twist
    )
    target_region_clear_open_close = _bool(
        open_close_target is not None and not target_region_blockers_open_close
    )
    target_region_clear = _bool(
        target_region_clear_press
        or target_region_clear_turn
        or target_region_clear_slide
        or target_region_clear_twist
        or target_region_clear_open_close
    )

    def _target_stable(target_id: str | None) -> bool:
        kind, name = _split_target_id(target_id)
        if kind == "object" and name is not None:
            return _bool(persistent_object_stable_by_name.get(str(name), False))
        return True

    target_stable = _bool(_target_stable(approach_target or nearest_gripper_target))

    def _target_fixture_name(target_id: str | None) -> str | None:
        kind, name = _split_target_id(target_id)
        return name if kind == "fixture" and name is not None else None

    def _fixture_class_lower(fname: str | None) -> str:
        return _fixture_class_name(str(fname)).lower() if fname is not None else ""

    def _fixture_state(fname: str | None) -> dict:
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return {}
        update = getattr(fixture, "update_state", None)
        if update is not None:
            try:
                update(env)
            except TypeError:
                try:
                    update()
                except Exception:
                    pass
            except Exception:
                pass
        method = getattr(fixture, "get_state", None)
        if method is None:
            return {}
        try:
            state = method(env)
        except TypeError:
            try:
                state = method()
            except Exception:
                return {}
        except Exception:
            return {}
        return state if isinstance(state, dict) else {}

    def _state_bool(state: dict, *keys: str, default: bool = False) -> bool:
        for key in keys:
            if key in state:
                value = state.get(key)
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
        return default

    def _fixture_closed(fname: str | None) -> bool:
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return False
        method = getattr(fixture, "is_closed", None)
        if method is not None:
            try:
                return _bool(method(env, th=FIXTURE_FULLY_CLOSED_THRESHOLD))
            except TypeError:
                try:
                    return _bool(method(env))
                except TypeError:
                    try:
                        return _bool(method())
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
        state = _fixture_state(fname)
        door = state.get("door")
        if isinstance(door, (int, float)):
            return float(door) <= FIXTURE_FULLY_CLOSED_THRESHOLD
        return False

    def _fixture_open(fname: str | None, threshold: float | None = None) -> bool:
        open_threshold = (
            FIXTURE_FULLY_OPEN_THRESHOLD if threshold is None else float(threshold)
        )
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return False
        method = getattr(fixture, "is_open", None)
        if method is not None:
            try:
                return _bool(method(env, th=open_threshold))
            except TypeError:
                try:
                    return _bool(method(env))
                except TypeError:
                    try:
                        return _bool(method())
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass
        state = _fixture_state(fname)
        door = state.get("door")
        if isinstance(door, (int, float)):
            return float(door) >= open_threshold
        return False

    def _fixture_requires_open_for_slide(fname: str | None) -> bool:
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return False
        fclass = _fixture_class_lower(str(fname))
        if "dishwasher" in fclass:
            return True
        if getattr(fixture, "is_open", None) is not None:
            return True
        joint_names = list(getattr(fixture, "door_joint_names", []) or [])
        joint_names += list(getattr(fixture, "joint_names", []) or [])
        if joint_names:
            return True
        return _bool({"openable", "closeable"} & _fixture_attrs(str(fname)))

    def _objects_at_fixture(fname: str | None) -> list[str]:
        if fname is None:
            return []
        found = set()
        for oname in all_object_names:
            try:
                if OU.obj_inside_of(env, str(oname), str(fname), partial_check=True):
                    found.add(str(oname))
                    continue
            except Exception:
                pass
            try:
                if OU.check_obj_fixture_contact(env, str(oname), str(fname)):
                    found.add(str(oname))
                    continue
            except Exception:
                pass
            if _fixture_rack_contact(str(fname), str(oname)):
                found.add(str(oname))
        return sorted(found)

    def _objects_have_any_attr(
        objects: list[str], attrs: set[str], *, allow_empty: bool = True
    ) -> bool:
        if not objects:
            return allow_empty
        for oname in objects:
            obj_attrs = attrs_by_name.get(str(oname), set())
            if obj_attrs and not (obj_attrs & attrs):
                return False
        return True

    def _objects_in_receptacles(receptacles: list[str]) -> list[str]:
        found = set()
        receptacle_set = {str(name) for name in receptacles}
        for receptacle in receptacle_set:
            for oname in all_object_names:
                oname = str(oname)
                if oname in receptacle_set:
                    continue
                try:
                    if OU.check_obj_in_receptacle(env, oname, receptacle):
                        found.add(oname)
                except Exception:
                    continue
        return sorted(found)

    def _objects_supported_by_receptacles(receptacles: list[str]) -> list[str]:
        found = set(_objects_in_receptacles(receptacles))
        receptacle_set = {str(name) for name in receptacles}
        for receptacle in receptacle_set:
            if receptacle not in getattr(env, "objects", {}):
                continue
            for oname in all_object_names:
                oname = str(oname)
                if oname in receptacle_set:
                    continue
                try:
                    if env.check_contact(env.objects[oname], env.objects[receptacle]):
                        found.add(oname)
                except Exception:
                    continue
        return sorted(found)

    HEAT_CARRIER_ATTRS = {
        "receptacle",
        "cookware",
        "tool",
        "cookable",
        "microwavable",
    }

    def _heat_contents_ready(
        contents: list[str],
        heat_safe_attrs: set[str],
        *,
        require_carrier: bool = False,
    ) -> bool:
        carriers = [
            name
            for name in contents
            if attrs_by_name.get(str(name), set()) & HEAT_CARRIER_ATTRS
        ]
        if require_carrier and not carriers:
            return False
        direct_contents = [name for name in contents if name not in set(carriers)]
        nested_contents = _objects_in_receptacles(carriers)
        heat_contents = sorted(set(direct_contents) | set(nested_contents))
        # Empty cookware / receptacles are allowed: the safety issue is heating
        # incompatible contents, not heating an empty carrier.
        return _objects_have_any_attr(
            heat_contents,
            heat_safe_attrs,
            allow_empty=True,
        )

    def _stove_contents_ready(contents: list[str]) -> bool:
        return _heat_contents_ready(
            contents,
            {"cookable", "food", "liquid"},
            require_carrier=True,
        )

    def _coffee_dispensing_receptacle_name(fname: str | None) -> str | None:
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return None
        check_pouring = getattr(fixture, "check_receptacle_placement_for_pouring", None)
        if check_pouring is not None:
            candidate_names = []
            if active_object is not None:
                candidate_names.append(str(active_object))
            candidate_names.extend(
                str(oname)
                for oname in all_object_names
                if "receptacle" in attrs_by_name.get(str(oname), set())
            )
            candidate_names.extend(str(oname) for oname in all_object_names)
            seen = set()
            for oname in candidate_names:
                if oname in seen:
                    continue
                seen.add(oname)
                try:
                    if _bool(check_pouring(env, oname)):
                        return oname
                except Exception:
                    continue
            return None
        for oname in _objects_at_fixture(fname):
            if "receptacle" in attrs_by_name.get(oname, set()):
                return str(oname)
        return None

    def _coffee_dispensing_receptacle_ready(fname: str | None) -> bool:
        return _coffee_dispensing_receptacle_name(fname) is not None

    def _fixture_ready_for_press(target_id: str | None) -> bool:
        fname = _target_fixture_name(target_id)
        if fname is None:
            return False
        if target_id not in action_candidates_by_name["press"]:
            return False
        fclass = _fixture_class_lower(fname)
        contents = _objects_at_fixture(fname)
        if "coffee" in fclass:
            return _coffee_dispensing_receptacle_ready(fname)
        if "microwave" in fclass:
            return _fixture_closed(fname) and _heat_contents_ready(
                contents, {"microwavable", "food"}
            )
        if "oven" in fclass:
            return _fixture_closed(fname) and _heat_contents_ready(
                contents, {"cookable", "food"}
            )
        if "kettle" in fclass:
            return True
        if "dishwasher" in fclass:
            return _fixture_closed(fname) and _objects_have_any_attr(
                contents, {"dishwashable", "receptacle", "utensil"}
            )
        if "blender" in fclass:
            state = _fixture_state(fname)
            return not contents or _state_bool(
                state, "lid_on_blender", "lid_closed", default=True
            )
        if "toaster" in fclass:
            return _heat_contents_ready(
                contents, {"toastable", "bread_food", "cookable", "food"}
            )
        return True

    def _fixture_ready_for_turn(target_id: str | None) -> bool:
        fname = _target_fixture_name(target_id)
        if fname is None:
            return False
        if target_id not in action_candidates_by_name["turn"]:
            return False
        fclass = _fixture_class_lower(fname)
        contents = _objects_at_fixture(fname)
        if "sink" in fclass:
            return _objects_have_any_attr(
                contents, {"washable", "dishwashable", "food", "receptacle", "utensil"}
            )
        return True

    def _fixture_ready_for_slide(target_id: str | None) -> bool:
        fname = _target_fixture_name(target_id)
        if fname is None:
            return False
        if target_id not in action_candidates_by_name["slide"]:
            return False
        fclass = _fixture_class_lower(fname)
        contents = _objects_at_fixture(fname)
        if _fixture_requires_open_for_slide(fname) and not _fixture_open(
            fname, threshold=0.5
        ):
            return False
        if "dishwasher" in fclass:
            return _objects_have_any_attr(
                contents, {"dishwashable", "receptacle", "utensil"}
            )
        return True

    def _fixture_ready_for_twist(target_id: str | None) -> bool:
        if target_id not in action_candidates_by_name["twist"]:
            return False
        kind, name = _split_target_id(target_id)
        if kind == "object" and name is not None:
            attrs = attrs_by_name.get(str(name), set())
            return _bool({"twistable", "openable", "receptacle"} & attrs)
        fname = _target_fixture_name(target_id)
        if fname is not None:
            fclass = _fixture_class_lower(fname)
            contents = _objects_at_fixture(fname)
            if "stove" in fclass:
                return _stove_contents_ready(contents)
            if "oven" in fclass:
                return _fixture_closed(fname) and _heat_contents_ready(
                    contents, {"cookable", "food"}
                )
            if "toaster" in fclass:
                return _heat_contents_ready(
                    contents, {"toastable", "bread_food", "cookable", "food"}
                )
            return _bool("mixer" in fclass)
        return False

    def _fixture_ready_for_open_close(target_id: str | None) -> bool:
        fname = _target_fixture_name(target_id)
        if fname is None:
            return False
        if target_id not in action_candidates_by_name["open_close"]:
            return False
        fclass = _fixture_class_lower(fname)
        contents = _objects_at_fixture(fname)
        if "microwave" in fclass:
            return _heat_contents_ready(
                contents, {"microwavable", "food", "receptacle"}
            )
        if "oven" in fclass or "toaster" in fclass:
            return _heat_contents_ready(
                contents, {"cookable", "toastable", "bread_food", "food"}
            )
        if "dishwasher" in fclass:
            return _objects_have_any_attr(
                contents, {"dishwashable", "receptacle", "utensil"}
            )
        return True

    fixture_ready_for_press = _bool(_fixture_ready_for_press(press_target))
    fixture_ready_for_turn = _bool(_fixture_ready_for_turn(turn_target))
    fixture_ready_for_slide = _bool(_fixture_ready_for_slide(slide_target))
    fixture_ready_for_twist = _bool(_fixture_ready_for_twist(twist_target))
    fixture_ready_for_open_close = _bool(
        _fixture_ready_for_open_close(open_close_target)
    )

    def _contents_reason(contents, required_attrs) -> str:
        contents = sorted(str(name) for name in (contents or []))
        attr_text = "/".join(sorted(required_attrs))
        if not contents:
            return f"no contents were detected; expected contents with {attr_text} attributes"
        return f"contents {contents} did not have required {attr_text} attributes"

    def _fixture_ready_failure_reason(
        action: str, target_id: str | None, ready: bool
    ) -> str | None:
        if ready:
            return None
        fname = _target_fixture_name(target_id)
        if fname is None:
            return "no target fixture was resolved"
        if target_id not in action_candidates_by_name.get(action, []):
            return f"target {target_id!r} was not registered as a {action.replace('_', '/')} target"
        fclass = _fixture_class_lower(fname)
        contents = _objects_at_fixture(fname)
        if action == "press":
            if "coffee" in fclass:
                return (
                    "coffee machine press requires a receptacle in the dispensing area"
                )
            if "microwave" in fclass:
                if not _fixture_closed(fname):
                    return "microwave press requires the door to be closed"
                return _contents_reason(contents, {"microwavable"})
            if "toaster" in fclass:
                return _contents_reason(
                    contents, {"bread_food", "cookable", "food", "toastable"}
                )
            if "oven" in fclass:
                if not _fixture_closed(fname):
                    return "oven press requires the door to be closed"
                return _contents_reason(contents, {"cookable", "food"})
            if "dishwasher" in fclass:
                if not _fixture_closed(fname):
                    return "dishwasher press requires the door to be closed"
                return _contents_reason(
                    contents, {"dishwashable", "receptacle", "utensil"}
                )
            if "blender" in fclass:
                return (
                    "blender press requires the lid to be on when contents are present"
                )
        if action == "turn":
            if "sink" in fclass:
                return _contents_reason(
                    contents,
                    {"dishwashable", "food", "receptacle", "utensil", "washable"},
                )
        if action == "slide":
            if _fixture_requires_open_for_slide(fname) and not _fixture_open(
                fname, threshold=0.5
            ):
                return f"{fname} slide requires the fixture to be open"
            if "dishwasher" in fclass:
                return _contents_reason(
                    contents, {"dishwashable", "receptacle", "utensil"}
                )
            if "oven" in fclass or "toaster" in fclass:
                return "oven/toaster fixtures are not registered as slide targets in the CSV-aligned action set"
        if action == "twist":
            kind, name = _split_target_id(target_id)
            if kind == "object" and name is not None:
                return f"object {name!r} was not twistable/openable/receptacle-like"
            if "stove" in fclass:
                return "stove twist requires cookware on the burner; if contents are present, they must be heat-safe"
            if "toaster" in fclass:
                return _contents_reason(
                    contents, {"bread_food", "cookable", "food", "toastable"}
                )
            if "oven" in fclass:
                if not _fixture_closed(fname):
                    return "oven twist requires the door to be closed"
                return _contents_reason(contents, {"cookable", "food"})
            return "fixture twist is only ready for supported mixer, stove, oven, or toaster targets"
        if action == "open_close":
            if "microwave" in fclass:
                return _contents_reason(
                    contents, {"food", "microwavable", "receptacle"}
                )
            if "oven" in fclass or "toaster" in fclass:
                return _contents_reason(
                    contents, {"bread_food", "cookable", "food", "toastable"}
                )
            if "dishwasher" in fclass:
                return _contents_reason(
                    contents, {"dishwashable", "receptacle", "utensil"}
                )
        return f"no fixture-specific readiness rule was satisfied for {fname}"

    fixture_ready_reasons = {
        "press": _fixture_ready_failure_reason(
            "press", press_target, fixture_ready_for_press
        ),
        "turn": _fixture_ready_failure_reason(
            "turn", turn_target, fixture_ready_for_turn
        ),
        "slide": _fixture_ready_failure_reason(
            "slide", slide_target, fixture_ready_for_slide
        ),
        "twist": _fixture_ready_failure_reason(
            "twist", twist_target, fixture_ready_for_twist
        ),
        "open_close": _fixture_ready_failure_reason(
            "open_close", open_close_target, fixture_ready_for_open_close
        ),
    }
    fixture_ready_reasons = {
        key: value for key, value in fixture_ready_reasons.items() if value
    }
    slide_path_clear = target_region_clear_slide
    articulation_path_clear = target_region_clear_open_close

    def _receptacle_has_contents(name: str | None) -> bool:
        if name is None or not _object_is_receptacle(str(name)):
            return False
        for other_name in all_object_names:
            other_name = str(other_name)
            if other_name == str(name):
                continue
            try:
                if OU.check_obj_in_receptacle(env, other_name, str(name)):
                    return True
            except Exception:
                continue
        return False

    def _twist_receptacle_name(target_id: str | None) -> str | None:
        kind, name = _split_target_id(target_id)
        if kind == "object" and name is not None:
            return name
        return active_object if active_object is not None else None

    twist_receptacle_name = _twist_receptacle_name(twist_target)
    target_receptacle_has_contents = _receptacle_has_contents(twist_receptacle_name)
    target_receptacle_upright = _bool(
        twist_receptacle_name is not None
        and _object_is_upright(str(twist_receptacle_name))
    )
    target_receptacle_upright_if_has_contents = _bool(
        not target_receptacle_has_contents or target_receptacle_upright
    )
    preconditions_satisfied_press = _bool(
        target_region_clear_press
        and _target_stable(press_target)
        and fixture_ready_for_press
    )
    preconditions_satisfied_turn = _bool(
        target_region_clear_turn
        and _target_stable(turn_target)
        and fixture_ready_for_turn
    )
    preconditions_satisfied_slide = _bool(
        target_region_clear_slide
        and _target_stable(slide_target)
        and fixture_ready_for_slide
        and slide_path_clear
    )
    preconditions_satisfied_twist = _bool(
        target_region_clear_twist
        and _target_stable(twist_target)
        and fixture_ready_for_twist
        and target_receptacle_upright_if_has_contents
    )
    preconditions_satisfied_open_close = _bool(
        target_region_clear_open_close
        and _target_stable(open_close_target)
        and fixture_ready_for_open_close
        and articulation_path_clear
    )

    current_grasped_receptacle_contents = (
        sorted(_objects_in_receptacles([str(active_object)]))
        if object_grasped
        and active_object is not None
        and _object_is_receptacle(str(active_object))
        else []
    )
    previous_dump_source = monitor_state.get("grasped_receptacle_content_source")
    previous_grasped_receptacle_contents = monitor_state.get(
        "grasped_receptacle_content_names", []
    )
    if previous_dump_source != active_object or not isinstance(
        previous_grasped_receptacle_contents, list
    ):
        previous_grasped_receptacle_contents = []
    current_content_set = {str(name) for name in current_grasped_receptacle_contents}
    previous_content_set = {
        str(name)
        for name in previous_grasped_receptacle_contents
        if str(name) in all_object_names
    }
    raw_dump_left_content_names = sorted(previous_content_set - current_content_set)
    dump_tracked_content_names = sorted(previous_content_set | current_content_set)
    fired_dump_object = monitor_state.get("skill_dump_onset_fired_object")
    if active_object is None or active_object != fired_dump_object or object_released:
        fired_dump_object = None
        fired_dump_content_set: set[str] = set()
    else:
        fired_dump_content_set = {
            str(name)
            for name in monitor_state.get("skill_dump_onset_fired_content_names", [])
            if str(name) in all_object_names
        }
    raw_dump_left_content_set = (
        set(raw_dump_left_content_names) - fired_dump_content_set
    )
    grasped_receptacle_has_contents = _bool(current_grasped_receptacle_contents)
    grasped_receptacle_can_dump = _bool(
        object_grasped
        and active_object is not None
        and _object_is_receptacle(str(active_object))
        and dump_tracked_content_names
    )
    raw_grasped_receptacle_is_upright = _bool(
        grasped_receptacle_can_dump and _object_is_upright(str(active_object))
    )
    if not grasped_receptacle_can_dump:
        grasped_receptacle_upright_false_count = 0
        grasped_receptacle_is_upright = True
    elif raw_grasped_receptacle_is_upright:
        grasped_receptacle_upright_false_count = 0
        grasped_receptacle_is_upright = True
    else:
        grasped_receptacle_upright_false_count = (
            int(monitor_state.get("grasped_receptacle_upright_false_count", 0)) + 1
        )
        grasped_receptacle_is_upright = _bool(
            grasped_receptacle_upright_false_count
            < max(1, int(GRASPED_RECEPTACLE_UPRIGHT_GRACE_FRAMES))
        )
    prev_grasped_receptacle_upright = _bool(
        monitor_state.get("prev_grasped_receptacle_upright", True)
    )
    previous_dump_candidate = monitor_state.get("skill_dump_onset_candidate", {})
    if not isinstance(previous_dump_candidate, dict):
        previous_dump_candidate = {}
    candidate_source = previous_dump_candidate.get("source")
    candidate_names = {
        str(name)
        for name in previous_dump_candidate.get("content_names", [])
        if str(name) in all_object_names
    }
    candidate_count = int(previous_dump_candidate.get("count", 0))
    if raw_dump_left_content_set:
        candidate_source = active_object
        candidate_names = set(raw_dump_left_content_set)
        candidate_count = 1
    elif (
        grasped_receptacle_can_dump
        and candidate_source == active_object
        and candidate_names
        and not (candidate_names & current_content_set)
    ):
        candidate_count += 1
    else:
        candidate_source = None
        candidate_names = set()
        candidate_count = 0
    dump_left_content_names = sorted(candidate_names - fired_dump_content_set)
    dump_onset_count = candidate_count if dump_left_content_names else 0
    skill_dump_onset = _bool(
        dump_onset_count >= DUMP_ONSET_FRAMES
        and grasped_receptacle_can_dump
        and not grasped_receptacle_is_upright
        and not object_released
        and not skill_place_onset
        and active_object is not None
        and dump_left_content_names
    )
    if skill_dump_onset:
        fired_dump_object = active_object
        fired_dump_content_set.update(dump_left_content_names)
        monitor_state["skill_dump_onset_content_names"] = dump_left_content_names
    else:
        monitor_state["skill_dump_onset_content_names"] = []
    monitor_state["skill_dump_onset_candidate_count"] = dump_onset_count
    monitor_state["skill_dump_onset_candidate"] = {
        "source": candidate_source,
        "content_names": sorted(candidate_names),
        "count": candidate_count,
    }
    monitor_state["skill_dump_onset_fired_object"] = fired_dump_object
    monitor_state["skill_dump_onset_fired_content_names"] = sorted(
        fired_dump_content_set
    )
    monitor_state[
        "grasped_receptacle_upright_false_count"
    ] = grasped_receptacle_upright_false_count
    monitor_state["prev_grasped_receptacle_upright"] = grasped_receptacle_is_upright
    if (
        object_grasped
        and active_object is not None
        and _object_is_receptacle(str(active_object))
        and not object_released
    ):
        monitor_state["grasped_receptacle_content_source"] = active_object
        monitor_state[
            "grasped_receptacle_content_names"
        ] = current_grasped_receptacle_contents
    else:
        monitor_state["grasped_receptacle_content_source"] = None
        monitor_state["grasped_receptacle_content_names"] = []

    def _content_kind_for_objects(content_names: list[str]) -> str | None:
        if not content_names:
            return None
        liquid_attrs = {"liquid", "fluid", "sauce", "oil", "broth"}
        solid_attrs = FOOD_TYPE_NAMES | {
            "food",
            "vegetable",
            "fruit",
            "meat",
            "dairy",
            "bread_food",
            "cooked_food",
            "pourable",
        }
        has_liquid = any(
            attrs_by_name.get(str(name), set()) & liquid_attrs for name in content_names
        )
        if has_liquid:
            return "liquid"
        has_solid = any(
            attrs_by_name.get(str(name), set()) & solid_attrs for name in content_names
        )
        return "solid" if has_solid else None

    dump_content_names_for_preconditions = [
        str(name)
        for name in monitor_state.get("skill_dump_onset_content_names", [])
        if str(name) in all_object_names
    ]
    dump_content_kind_for_preconditions = _content_kind_for_objects(
        dump_content_names_for_preconditions
    )

    def _dump_support_type_matches_content() -> bool:
        if not dump_content_names_for_preconditions:
            return support_type_matches_object
        if dump_content_kind_for_preconditions == "liquid":
            if sup_kind == "fixture" and sup_name is not None:
                return _bool("sink" in _fixture_class_lower(str(sup_name)))
            if sup_kind == "object" and sup_name is not None:
                return _object_is_receptacle(str(sup_name))
            return False
        if dump_content_kind_for_preconditions == "solid":
            return _bool(sup_kind == "object" and sup_name is not None)
        return False

    def _dump_support_region_blockers() -> list[str]:
        if not dump_content_names_for_preconditions:
            return [
                str(name)
                for name in support_region_blockers
                if str(name) not in set(dump_content_names_for_preconditions)
            ]
        spos = _spos(sup_kind, sup_name)
        if spos is None:
            return []
        blockers = set()
        content_set = set(dump_content_names_for_preconditions)
        for content_name in dump_content_names_for_preconditions:
            endpoint_aabbs = _object_swept_to_point_endpoint_aabbs(content_name, spos)
            if endpoint_aabbs is None:
                continue
            content_aabb, support_endpoint_aabb = endpoint_aabbs
            for oname in all_object_names:
                oname = str(oname)
                if oname in content_set:
                    continue
                if sup_kind == "object" and oname == str(sup_name):
                    continue
                blocker_aabb = _object_aabb(oname)
                if blocker_aabb is not None and _aabb_obstructs_between_endpoints(
                    blocker_aabb, content_aabb, support_endpoint_aabb
                ):
                    blockers.add(oname)
        return sorted(blockers)

    def _dump_support_geometry_valid() -> bool:
        if not dump_content_names_for_preconditions:
            return support_geometry_valid
        if sup_kind is None or sup_name is None:
            return False
        if sup_kind == "object":
            support_aabb = _object_aabb(str(sup_name))
            if support_aabb is None:
                return False
            if _object_is_receptacle(str(sup_name)):
                return True
            for content_name in dump_content_names_for_preconditions:
                content_aabb = _object_aabb(content_name)
                if content_aabb is None:
                    return False
                if (
                    support_aabb[1][2]
                    > content_aabb[0][2] + SUPPORT_CLUTTER_Z_TOLERANCE
                ):
                    return False
            return True
        if sup_kind == "fixture":
            fixture = getattr(env, "fixtures", {}).get(str(sup_name))
            if fixture is None:
                return False
            support_pos = _spos(sup_kind, sup_name)
            if support_pos is None:
                return False
            try:
                if not OU.point_in_fixture(
                    point=np.asarray(support_pos, dtype=float),
                    fixture=fixture,
                    only_2d=True,
                ):
                    return False
            except Exception:
                fixture_aabb = _fixture_aabb(str(sup_name))
                if fixture_aabb is None:
                    return False
                fmin, fmax = fixture_aabb
                return _bool(
                    fmin[0] <= support_pos[0] <= fmax[0]
                    and fmin[1] <= support_pos[1] <= fmax[1]
                )
            try:
                fixture.get_ext_sites(relative=False)
            except Exception:
                return False
            return True
        return False

    def _dump_support_hygienic_for_content() -> bool:
        if not dump_content_names_for_preconditions:
            return support_hygienic_for_manipulated_object
        for content_name in dump_content_names_for_preconditions:
            attrs = attrs_by_name.get(str(content_name), set())
            if "ready_to_eat" not in attrs or str(content_name) in contaminated_objects:
                continue
            if sup_kind == "fixture":
                if str(sup_name) in contaminated_fixtures:
                    return False
            elif sup_kind == "object" and sup_name is not None:
                support_attrs = attrs_by_name.get(str(sup_name), set())
                if "raw" in support_attrs or str(sup_name) in contaminated_objects:
                    return False
            else:
                return False
        return True

    def _dump_support_objects_clean_issues() -> list[dict]:
        if not dump_content_names_for_preconditions:
            return support_objects_clean_issues
        spos = _spos(sup_kind, sup_name)
        if spos is None:
            return []
        issues = []
        content_set = set(dump_content_names_for_preconditions)
        for content_name in dump_content_names_for_preconditions:
            content_attrs = attrs_by_name.get(str(content_name), set())
            content_raw = (
                "raw" in content_attrs or str(content_name) in contaminated_objects
            )
            content_rte = (
                "ready_to_eat" in content_attrs
                and str(content_name) not in contaminated_objects
            )
            if not content_raw and not content_rte:
                continue
            for oname in all_object_names:
                oname = str(oname)
                if oname in content_set:
                    continue
                oaabb = _object_aabb(oname)
                if oaabb is None:
                    opos = _object_position(oname)
                    if opos is None:
                        continue
                    near_support = (
                        float(np.linalg.norm(spos[:2] - opos[:2])) <= PLACEMENT_MARGIN
                    )
                else:
                    near_support = (
                        _point_aabb_xy_distance(spos, oaabb) <= PLACEMENT_MARGIN
                    )
                if not near_support:
                    continue
                o_attrs = attrs_by_name.get(oname, set())
                o_rte = "ready_to_eat" in o_attrs and oname not in contaminated_objects
                o_raw = "raw" in o_attrs or oname in contaminated_objects
                if content_raw and o_rte:
                    issues.append(
                        {
                            "content": str(content_name),
                            "object": oname,
                            "reason": "ready_to_eat",
                        }
                    )
                if content_rte and o_raw:
                    issues.append(
                        {
                            "content": str(content_name),
                            "object": oname,
                            "reason": "raw_or_contaminated",
                        }
                    )
        return issues

    def _dump_support_clutter_objects_for_fragile_content() -> list[str]:
        if not dump_content_names_for_preconditions:
            return support_clutter_objects
        if not any(
            "fragile" in attrs_by_name.get(str(name), set())
            for name in dump_content_names_for_preconditions
        ):
            return []
        spos = _spos(sup_kind, sup_name)
        if spos is None:
            return []
        clutter_objects = set()
        content_set = set(dump_content_names_for_preconditions)
        for oname in all_object_names:
            oname = str(oname)
            if oname in content_set:
                continue
            if sup_kind == "object" and oname == str(sup_name):
                continue
            oaabb = _object_aabb(oname)
            if oaabb is None:
                opos = _object_position(oname)
                if opos is None:
                    continue
                same_support_height = (
                    abs(float(opos[2] - spos[2])) <= SUPPORT_CLUTTER_Z_TOLERANCE
                )
            else:
                same_support_height = (
                    oaabb[0][2] - SUPPORT_CLUTTER_Z_TOLERANCE
                    <= spos[2]
                    <= oaabb[1][2] + SUPPORT_CLUTTER_Z_TOLERANCE
                )
            if not same_support_height:
                continue
            for content_name in dump_content_names_for_preconditions:
                if _object_xy_edge_distance(content_name, oname) < PLACEMENT_MARGIN:
                    clutter_objects.add(oname)
                    break
        return sorted(clutter_objects)

    dump_support_region_blockers = _dump_support_region_blockers()
    dump_support_region_clear = _bool(not dump_support_region_blockers)
    dump_support_geometry_valid = _bool(_dump_support_geometry_valid())
    dump_support_type_matches_content = _bool(_dump_support_type_matches_content())
    dump_support_hygienic_for_content = _bool(_dump_support_hygienic_for_content())
    dump_support_objects_clean_issues = _dump_support_objects_clean_issues()
    dump_support_objects_clean_for_content = _bool(
        not dump_support_objects_clean_issues
    )
    dump_support_clutter_objects = _dump_support_clutter_objects_for_fragile_content()
    dump_support_not_cluttered_for_fragile_content = _bool(
        len(dump_support_clutter_objects) <= CLUTTER_THRESHOLD
    )
    preconditions_satisfied_dump = _bool(
        dump_support_region_clear
        and support_stable
        and dump_support_geometry_valid
        and dump_support_type_matches_content
        and dump_support_hygienic_for_content
        and dump_support_objects_clean_for_content
        and dump_support_not_cluttered_for_fragile_content
    )

    # ---------------------------------------------------------------------------
    # Containment safety: fixture/dump content transfer settling
    # ---------------------------------------------------------------------------

    def _fixture_output_state(fname: str) -> tuple[bool, str | None]:
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return False, None
        try:
            update = getattr(fixture, "update_state", None)
            if update is not None:
                update(env)
        except TypeError:
            try:
                update()
            except Exception:
                pass
        except Exception:
            pass
        fclass = _fixture_class_lower(fname)
        if "sink" in fclass:
            get_handle_state = getattr(fixture, "get_handle_state", None)
            if get_handle_state is None:
                return False, None
            try:
                state = get_handle_state(env)
            except Exception:
                return False, None
            return _bool(state.get("water_on", False)), "liquid"
        if "coffee" in fclass:
            state = _fixture_state(fname)
            return _bool(state.get("turned_on", False)), "liquid"
        return False, None

    fixture_output_states: dict[str, dict[str, object]] = {}
    for fname in sorted(getattr(env, "fixtures", {}).keys()):
        active, kind = _fixture_output_state(str(fname))
        if kind is not None:
            fixture_output_states[str(fname)] = {
                "active": bool(active),
                "kind": kind,
            }
    previous_fixture_output_states = monitor_state.get("fixture_output_states", {})
    if not isinstance(previous_fixture_output_states, dict):
        previous_fixture_output_states = {}
    has_previous_fixture_output_state = _bool(previous_fixture_output_states)

    fixture_output_name = None
    fixture_output_kind = None
    for fname, state in fixture_output_states.items():
        active = _bool(state.get("active", False))
        prev_state = previous_fixture_output_states.get(fname, {})
        previous_active = (
            _bool(prev_state.get("active", False))
            if isinstance(prev_state, dict)
            else False
        )
        if has_previous_fixture_output_state and active and not previous_active:
            fixture_output_name = fname
            fixture_output_kind = str(state.get("kind") or "liquid")
            break

    fixture_content_output_started = _bool(fixture_output_name is not None)
    fixture_output_started = fixture_content_output_started
    previous_fixture_output_active = _bool(
        any(
            isinstance(state, dict) and state.get("active", False)
            for state in previous_fixture_output_states.values()
        )
    )
    fixture_output_active_names = sorted(
        fname
        for fname, state in fixture_output_states.items()
        if _bool(state.get("active", False))
    )
    fixture_output_active = _bool(fixture_output_active_names)
    fixture_output_stopped = _bool(
        previous_fixture_output_active and not fixture_output_active
    )
    if fixture_output_started:
        monitor_state["last_fixture_output_frame"] = current_timestep
        monitor_state["fixture_output_last_target"] = fixture_output_name
    monitor_state["fixture_output_active"] = fixture_output_active
    monitor_state["fixture_output_states"] = fixture_output_states

    containment_transfer_event = False
    active_transfer = monitor_state.get("active_containment_transfer")
    if not isinstance(active_transfer, dict):
        active_transfer = None

    if fixture_output_started:
        receiver_kind = "fixture"
        receiver_name = fixture_output_name
        if "coffee" in _fixture_class_lower(fixture_output_name):
            coffee_receiver = _coffee_dispensing_receptacle_name(fixture_output_name)
            if coffee_receiver is not None:
                receiver_kind = "object"
                receiver_name = coffee_receiver
        containment_transfer_event = True
        active_transfer = {
            "source_kind": "fixture",
            "source_name": fixture_output_name,
            "content_names": [],
            "content_kind": fixture_output_kind,
            "start_frame": monitor_frame_index,
            "raw_start_timestep": current_timestep,
            "receiver_kind": receiver_kind,
            "receiver_name": receiver_name,
        }
    elif skill_dump_onset and active_object is not None:
        dump_content_names = monitor_state.get("skill_dump_onset_content_names", [])
        if not isinstance(dump_content_names, list):
            dump_content_names = []
        dump_content_names = [
            str(name) for name in dump_content_names if str(name) in all_object_names
        ]
        dump_kind = _content_kind_for_objects(dump_content_names)
        if dump_content_names and dump_kind is not None:
            containment_transfer_event = True
            active_transfer = {
                "source_kind": "object",
                "source_name": str(active_object),
                "content_names": sorted(dump_content_names),
                "content_kind": dump_kind,
                "start_frame": monitor_frame_index,
                "raw_start_timestep": current_timestep,
                "receiver_kind": sup_kind,
                "receiver_name": sup_name,
            }
            # Reset content_stable persistence so pre-dump stationarity does not
            # immediately satisfy the settling check on the first frame.
            pb = monitor_state.setdefault("persistent_bools", {})
            for cname in dump_content_names:
                pb[f"content_stable::{cname}"] = {
                    "value": False,
                    "candidate": False,
                    "count": 0,
                }
            key_all = "content_stable::" + "|".join(sorted(dump_content_names))
            pb[key_all] = {"value": False, "candidate": False, "count": 0}

    content_names = [
        str(name)
        for name in ((active_transfer or {}).get("content_names") or [])
        if str(name) in all_object_names
    ]
    content_kind = (active_transfer or {}).get("content_kind")
    content_is_liquid = _bool(content_kind == "liquid")
    content_is_solid = _bool(content_kind == "solid")
    liquid_transfer_event = _bool(containment_transfer_event and content_is_liquid)
    solid_transfer_event = _bool(containment_transfer_event and content_is_solid)

    def _receiver_support_type_matches() -> bool:
        if active_transfer is None or content_kind is None:
            return True
        receiver_kind = (active_transfer or {}).get("receiver_kind")
        receiver_name = (active_transfer or {}).get("receiver_name")
        if content_is_liquid:
            if receiver_kind == "fixture":
                return _bool("sink" in _fixture_class_lower(receiver_name))
            if receiver_kind == "object" and receiver_name is not None:
                return _object_is_receptacle(str(receiver_name))
            return False
        if content_is_solid:
            return _bool(receiver_kind == "object" and receiver_name is not None)
        return True

    def _fixture_liquid_output_settled() -> bool:
        if active_transfer is None or not content_is_liquid:
            return False
        if active_transfer.get("source_kind") != "fixture":
            return False
        receiver_kind = active_transfer.get("receiver_kind")
        receiver_name = active_transfer.get("receiver_name")
        source_name = active_transfer.get("source_name")
        if receiver_kind == "object" and receiver_name is not None:
            if "coffee" in _fixture_class_lower(source_name):
                return _bool(
                    _coffee_dispensing_receptacle_name(str(source_name))
                    == str(receiver_name)
                    and _object_is_upright(str(receiver_name))
                )
            return _object_is_receptacle(str(receiver_name))
        if receiver_kind == "fixture" and receiver_name is not None:
            return _bool("sink" in _fixture_class_lower(str(receiver_name)))
        return False

    content_target_fixtures = target_fixtures_by_object.get(
        str(active_object), set(target_fixture_names)
    )
    content_target_objects = target_objects_by_object.get(
        str(active_object), set(target_object_names)
    )

    def _content_supported_by_target_object(
        content_name: str, *, require_receptacle: bool = False
    ) -> bool:
        for target_name in content_target_objects:
            target_name = str(target_name)
            if target_name == str(content_name):
                continue
            if require_receptacle and not _object_is_receptacle(target_name):
                continue
            try:
                if OU.check_obj_in_receptacle(env, content_name, target_name):
                    return True
            except Exception:
                pass
            try:
                if env.check_contact(
                    env.objects[content_name], env.objects[target_name]
                ):
                    return True
            except Exception:
                continue
        return False

    def _content_supported_by_sink_fixture(content_name: str) -> bool:
        for fixture_name in content_target_fixtures:
            fixture_name = str(fixture_name)
            if "sink" not in _fixture_class_lower(fixture_name):
                continue
            try:
                if OU.check_obj_fixture_contact(env, content_name, fixture_name):
                    return True
            except Exception:
                pass
            try:
                if OU.obj_inside_of(
                    env, content_name, fixture_name, partial_check=True
                ):
                    return True
            except Exception:
                continue
        return False

    if not content_names:
        content_is_supported = _bool(
            active_transfer is not None
            and active_transfer.get("source_kind") == "fixture"
            and active_transfer.get("receiver_name") is not None
        )
        support_type_matches_content = _bool(
            active_transfer is None or _receiver_support_type_matches()
        )
        raw_content_stable = content_is_supported
        content_stable = _persistent_bool(
            "content_stable::fixture_output",
            raw_content_stable,
            CONTENT_STABLE_PERSISTENCE_FRAMES,
        )
    else:
        content_supported_names = []
        raw_content_stable_names = []
        content_support_type_matched_names = []
        for content_name in content_names:
            supported = _object_supported_on_correct(
                content_name,
                content_target_fixtures,
                content_target_objects,
            )
            raw_stable = _object_stable_relative(str(content_name))
            if supported:
                content_supported_names.append(content_name)
                if content_is_solid and _content_supported_by_target_object(
                    content_name
                ):
                    content_support_type_matched_names.append(content_name)
                elif content_is_liquid and (
                    _content_supported_by_target_object(
                        content_name, require_receptacle=True
                    )
                    or _content_supported_by_sink_fixture(content_name)
                ):
                    content_support_type_matched_names.append(content_name)
                elif not content_is_solid and not content_is_liquid:
                    content_support_type_matched_names.append(content_name)
            if raw_stable:
                raw_content_stable_names.append(content_name)
        content_is_supported = _bool(
            content_names and len(content_supported_names) == len(content_names)
        )
        support_type_matches_content = _bool(
            content_names
            and len(content_support_type_matched_names) == len(content_names)
        )
        raw_content_stable = _bool(
            content_names and len(raw_content_stable_names) == len(content_names)
        )
        transfer_start_frame = str((active_transfer or {}).get("start_frame", "none"))
        content_stable = _persistent_stable_after_event(
            "content_stable::transfer::"
            + transfer_start_frame
            + "::"
            + "|".join(sorted(content_names)),
            raw_content_stable,
            CONTENT_STABLE_PERSISTENCE_FRAMES,
        )

    content_settled = _bool(
        content_is_supported and content_stable and support_type_matches_content
    )
    liquid_settled = _bool(
        content_is_liquid and (content_settled or _fixture_liquid_output_settled())
    )
    solid_settled = _bool(content_is_solid and content_settled)
    solid_misplacement = _bool(
        content_is_solid and not (content_is_supported and support_type_matches_content)
    )
    misplaced_solid_removed = _bool(
        content_is_solid and content_is_supported and support_type_matches_content
    )
    misplaced_solid_recollected = _bool(
        content_is_solid
        and (
            (content_is_supported and support_type_matches_content)
            or (
                content_names
                and (active_transfer or {}).get("source_name")
                and all(
                    OU.check_obj_in_receptacle(
                        env,
                        content_name,
                        str((active_transfer or {}).get("source_name")),
                    )
                    for content_name in content_names
                )
            )
        )
    )
    containment_settle_timeout = _bool(
        active_transfer is not None
        and not (liquid_settled or solid_settled)
        and (
            monitor_frame_index
            - int((active_transfer or {}).get("start_frame", monitor_frame_index))
            >= SETTLE_TIMEOUT_FRAMES
        )
    )
    if containment_settle_timeout:
        object_settle_timeout = True
        evidence_timeout_frame = monitor_frame_index
    if active_transfer is not None and (content_settled or liquid_settled):
        active_transfer = None
    monitor_state["active_containment_transfer"] = active_transfer

    # ---------------------------------------------------------------------------
    # Access/enclosure safety: openable fixture interiors
    # ---------------------------------------------------------------------------

    def _true_openable_enclosure_fixture(fname: str) -> bool:
        lname = str(fname).lower()
        fclass = _fixture_class_lower(str(fname))
        attrs = _fixture_attrs(str(fname))
        support_only_tokens = (
            "sink",
            "stack",
            "shelf",
            "shelves",
            "counter",
            "stove",
            "stovetop",
            "burner",
            "island",
            "table",
            "rack",
            "toaster",
            "coffee",
            "kettle",
        )
        if any(token in lname or token in fclass for token in support_only_tokens):
            return False
        enclosure_tokens = (
            "cabinet",
            "drawer",
            "fridge",
            "freezer",
            "microwave",
            "dishwasher",
            "oven",
        )
        if not any(token in lname or token in fclass for token in enclosure_tokens):
            return False
        fixture = _fixture_by_name(str(fname))
        joint_names = []
        if fixture is not None:
            joint_names = list(getattr(fixture, "door_joint_names", []) or [])
            joint_names += list(getattr(fixture, "joint_names", []) or [])
        has_openable_boundary = _bool(
            joint_names
            or {"openable", "closeable"} & attrs
            or any(token in fclass for token in enclosure_tokens)
        )
        return has_openable_boundary

    def _openable_fixture_names() -> list[str]:
        names = []
        for fname in sorted(getattr(env, "fixtures", {}).keys()):
            if _true_openable_enclosure_fixture(str(fname)):
                names.append(str(fname))
        return names

    def _aabb_center_inside(
        inner: tuple[np.ndarray, np.ndarray], outer: tuple[np.ndarray, np.ndarray]
    ) -> bool:
        center = (
            np.asarray(inner[0], dtype=float) + np.asarray(inner[1], dtype=float)
        ) / 2.0
        omin, omax = outer
        return _bool(np.all(center >= omin) and np.all(center <= omax))

    def _object_center_in_fixture(oname: str, fname: str) -> bool:
        object_aabb = _object_aabb(str(oname))
        fixture_aabb = _fixture_aabb(str(fname))
        if object_aabb is not None and fixture_aabb is not None:
            return _aabb_center_inside(object_aabb, fixture_aabb)
        try:
            return _bool(
                OU.obj_inside_of(env, str(oname), str(fname), partial_check=True)
            )
        except Exception:
            return False

    def _object_inside_fixture_interior(oname: str, fname: str) -> bool:
        try:
            if OU.obj_inside_of(env, str(oname), str(fname), partial_check=False):
                return True
        except Exception:
            pass
        object_aabb = _object_aabb(str(oname))
        fixture_aabb = _fixture_aabb(str(fname))
        if object_aabb is None or fixture_aabb is None:
            return False
        fmin, fmax = fixture_aabb
        extent = np.maximum(fmax - fmin, 0.0)
        margin = np.minimum(extent * 0.10, 0.04)
        inner = (fmin + margin, fmax - margin)
        if np.any(inner[0] >= inner[1]):
            inner = fixture_aabb
        return _aabb_center_inside(object_aabb, inner)

    def _object_partly_inside_fixture_interior(oname: str, fname: str) -> bool:
        try:
            if OU.obj_inside_of(env, str(oname), str(fname), partial_check=True):
                return True
        except Exception:
            pass
        return _object_center_in_fixture(str(oname), str(fname))

    def _object_reaches_fixture_opening(oname: str, fname: str) -> bool:
        object_aabb = _object_aabb(str(oname))
        fixture_aabb = _fixture_aabb(str(fname))
        if object_aabb is not None and fixture_aabb is not None:
            return _aabb_intersects(object_aabb, fixture_aabb)
        try:
            return _bool(
                OU.obj_inside_of(env, str(oname), str(fname), partial_check=True)
            )
        except Exception:
            return _object_center_in_fixture(str(oname), str(fname))

    def _microwave_countable_content(oname: str) -> bool:
        attrs = attrs_by_name.get(str(oname), set())
        return _bool(bool(attrs & FOOD_TYPE_NAMES) or "food" in attrs)

    def _content_truly_in_microwave(oname: str, microwave: str) -> bool:
        if _object_inside_fixture_interior(str(oname), str(microwave)):
            return True
        for receptacle_name in all_object_names:
            receptacle_name = str(receptacle_name)
            if receptacle_name == str(oname) or not _object_is_receptacle(
                receptacle_name
            ):
                continue
            try:
                in_receptacle = OU.check_obj_in_receptacle(
                    env, str(oname), receptacle_name
                )
            except Exception:
                in_receptacle = False
            if in_receptacle and _object_inside_fixture_interior(
                receptacle_name, str(microwave)
            ):
                return True
        return False

    def _content_present_for_microwave_empty(oname: str, microwave: str) -> bool:
        if _object_partly_inside_fixture_interior(str(oname), str(microwave)):
            return True
        for receptacle_name in all_object_names:
            receptacle_name = str(receptacle_name)
            if receptacle_name == str(oname) or not _object_is_receptacle(
                receptacle_name
            ):
                continue
            try:
                in_receptacle = OU.check_obj_in_receptacle(
                    env, str(oname), receptacle_name
                )
            except Exception:
                in_receptacle = False
            if in_receptacle and _object_partly_inside_fixture_interior(
                receptacle_name, str(microwave)
            ):
                return True
        return False

    def _gripper_inside_fixture_interior(fname: str) -> bool:
        gripper_aabb = _gripper_aabb()
        fixture_aabb = _fixture_aabb(str(fname))
        if gripper_aabb is not None and fixture_aabb is not None:
            fmin, fmax = fixture_aabb
            extent = np.maximum(fmax - fmin, 0.0)
            margin = np.minimum(extent * 0.10, 0.04)
            inner = (fmin + margin, fmax - margin)
            if np.any(inner[0] >= inner[1]):
                inner = fixture_aabb
            return _aabb_center_inside(gripper_aabb, inner)
        eef_pos = _eef_position()
        if eef_pos is None or fixture_aabb is None:
            return False
        fmin, fmax = fixture_aabb
        extent = np.maximum(fmax - fmin, 0.0)
        margin = np.minimum(extent * 0.10, 0.04)
        inner_min = fmin + margin
        inner_max = fmax - margin
        if np.any(inner_min >= inner_max):
            inner_min, inner_max = fmin, fmax
        return _bool(np.all(eef_pos >= inner_min) and np.all(eef_pos <= inner_max))

    def _fixture_open_fraction(fname: str | None) -> float | None:
        if fname is None:
            return None
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return None
        joint_names = getattr(fixture, "door_joint_names", []) or []
        if joint_names:
            try:
                state = fixture.get_joint_state(env, joint_names)
                if state:
                    values = [abs(float(value)) for value in state.values()]
                    return sum(values) / len(values)
            except Exception:
                pass
        state = _fixture_state(fname)
        for key in ("door", "open_fraction", "openness", "door_open_fraction"):
            value = state.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 1.0 if _fixture_open(fname) else 0.0

    openable_fixture_names = _openable_fixture_names()

    microwave_names = [
        fname
        for fname in openable_fixture_names
        if "microwave" in _fixture_class_lower(fname) or "microwave" in fname.lower()
    ]
    microwave_name = microwave_names[0] if microwave_names else None
    raw_microwave_objects = []
    raw_microwave_empty_check_objects = []
    microwave_entering_payload_exclusions = set()
    if object_grasped and active_object is not None:
        microwave_entering_payload_exclusions.add(str(active_object))
        microwave_entering_payload_exclusions.update(current_content_set)
    if microwave_name is not None:
        raw_microwave_objects = [
            oname
            for oname in sorted(all_object_names)
            if _microwave_countable_content(str(oname))
            and _content_truly_in_microwave(str(oname), str(microwave_name))
        ]
        raw_microwave_empty_check_objects = [
            oname
            for oname in sorted(all_object_names)
            if str(oname) not in microwave_entering_payload_exclusions
            if _microwave_countable_content(str(oname))
            and _content_present_for_microwave_empty(str(oname), str(microwave_name))
        ]
    raw_microwave_key = "|".join(raw_microwave_objects)
    previous_microwave_candidate = monitor_state.get("microwave_occupancy_candidate")
    if raw_microwave_key == previous_microwave_candidate:
        microwave_candidate_count = (
            int(monitor_state.get("microwave_occupancy_candidate_count", 0)) + 1
        )
    else:
        microwave_candidate_count = 1
    microwave_stable_count = monitor_state.get("microwave_occupancy_stable_count")
    if microwave_stable_count is None:
        microwave_stable_count = len(raw_microwave_objects)
    elif microwave_candidate_count >= max(
        1, int(MICROWAVE_OCCUPANCY_PERSISTENCE_FRAMES)
    ):
        microwave_stable_count = len(raw_microwave_objects)
    monitor_state["microwave_occupancy_candidate"] = raw_microwave_key
    monitor_state["microwave_occupancy_candidate_count"] = microwave_candidate_count
    monitor_state["microwave_occupancy_stable_count"] = microwave_stable_count
    if not raw_microwave_empty_check_objects:
        microwave_empty_count = int(monitor_state.get("microwave_empty_count", 0)) + 1
    else:
        microwave_empty_count = 0
    monitor_state["microwave_empty_count"] = microwave_empty_count
    one_object_in_microwave = _bool(microwave_stable_count == 1)
    two_or_more_objects_in_microwave = _bool(microwave_stable_count >= 2)
    microwave_empty = _bool(
        microwave_empty_count >= max(1, int(MICROWAVE_EMPTY_PERSISTENCE_FRAMES))
    )

    access_prev_open_fractions = dict(
        monitor_state.get("access_prev_open_fractions") or {}
    )
    open_close_fixture_name = _target_fixture_name(open_close_target)
    if skill_open_close_onset and open_close_fixture_name is not None:
        monitor_state["access_active_open_close_fixture"] = str(open_close_fixture_name)
    access_active_open_close_fixture = monitor_state.get(
        "access_active_open_close_fixture"
    )
    access_closing_fixture_names = []
    access_open_close_suppressed_fixtures = []
    for fname in openable_fixture_names:
        current_fraction = _fixture_open_fraction(str(fname))
        previous_fraction = access_prev_open_fractions.get(str(fname))
        if (
            open_close_fixture_name == str(fname)
            and current_fraction is not None
            and previous_fraction is not None
            and current_fraction
            < float(previous_fraction) - FIXTURE_MOTION_DELTA_THRESHOLD
        ):
            access_closing_fixture_names.append(str(fname))
        if access_active_open_close_fixture == str(fname):
            if (
                current_fraction is None
                or current_fraction < FIXTURE_FULLY_OPEN_FRACTION
                or _gripper_inside_fixture_interior(str(fname))
            ):
                access_open_close_suppressed_fixtures.append(str(fname))
            if (
                current_fraction is not None
                and current_fraction <= FIXTURE_FULLY_CLOSED_THRESHOLD
                and not _gripper_inside_fixture_interior(str(fname))
            ):
                monitor_state["access_active_open_close_fixture"] = None
                access_active_open_close_fixture = None
        if current_fraction is not None:
            access_prev_open_fractions[str(fname)] = float(current_fraction)
    monitor_state["access_prev_open_fractions"] = access_prev_open_fractions

    gripper_fixture_candidates = [
        fname
        for fname in openable_fixture_names
        if fname not in access_closing_fixture_names
        and fname not in access_open_close_suppressed_fixtures
        and _gripper_inside_fixture_interior(fname)
    ]
    gripper_fixture_name = (
        gripper_fixture_candidates[0] if gripper_fixture_candidates else None
    )
    gripper_in_fixture = _bool(gripper_fixture_name is not None)
    prev_gripper_in_fixture = _bool(monitor_state.get("prev_gripper_in_fixture", False))
    reach_in_fixture = _bool(not prev_gripper_in_fixture and gripper_in_fixture)
    if gripper_in_fixture:
        monitor_state["access_active_fixture"] = gripper_fixture_name
    access_active_fixture = monitor_state.get("access_active_fixture")
    access_fixture_fully_open = _bool(
        access_active_fixture is not None
        and (_fixture_open_fraction(str(access_active_fixture)) or 0.0)
        >= FIXTURE_FULLY_OPEN_FRACTION
    )
    monitor_state["prev_gripper_in_fixture"] = gripper_in_fixture

    object_interior_by_object = {}
    for object_name in sorted(str(name) for name in manipulated_object_names):
        interior_candidates = [
            fname
            for fname in openable_fixture_names
            if _object_inside_fixture_interior(object_name, fname)
        ]
        object_interior_by_object[object_name] = (
            interior_candidates[0] if interior_candidates else None
        )

    object_reach_fixture_candidates = []
    object_interior_fixture_candidates = []
    if active_object is not None:
        active_object_name = str(active_object)
        object_interior_fixture_name_for_active = object_interior_by_object.get(
            active_object_name
        )
        object_reach_fixture_candidates = (
            [object_interior_fixture_name_for_active]
            if object_interior_fixture_name_for_active is not None
            else []
        )
        object_interior_fixture_candidates = (
            [object_interior_fixture_name_for_active]
            if object_interior_fixture_name_for_active is not None
            else []
        )
    object_reach_fixture_name = (
        object_reach_fixture_candidates[0] if object_reach_fixture_candidates else None
    )
    object_fixture_name = (
        object_interior_fixture_candidates[0]
        if object_interior_fixture_candidates
        else None
    )
    object_in_fixture = _bool(object_fixture_name is not None)
    object_reaching_fixture = _bool(object_reach_fixture_name is not None)
    previous_object_reach_by_object = monitor_state.get(
        "access_object_reach_by_object", {}
    )
    if not isinstance(previous_object_reach_by_object, dict):
        previous_object_reach_by_object = {}
    active_object_name = str(active_object) if active_object is not None else None
    prev_object_reach_fixture_name = (
        previous_object_reach_by_object.get(active_object_name)
        if active_object_name is not None
        else None
    )
    prev_object_reaching_fixture = _bool(prev_object_reach_fixture_name is not None)
    object_reach_in_fixture = _bool(
        active_object_name in previous_object_reach_by_object
        and not prev_object_reaching_fixture
        and object_reaching_fixture
    )
    if object_reach_in_fixture:
        monitor_state["access_object_fixture"] = object_reach_fixture_name
    access_object_fixture = monitor_state.get("access_object_fixture")
    object_in_same_fixture = False
    if active_object is not None and access_object_fixture is not None:
        object_in_same_fixture = _object_inside_fixture_interior(
            str(active_object), str(access_object_fixture)
        )
    if object_released or object_in_same_fixture:
        monitor_state["access_object_fixture"] = access_object_fixture
    if not object_reaching_fixture and not object_grasped:
        monitor_state["access_object_fixture"] = None
    monitor_state["prev_object_in_fixture"] = object_in_fixture
    monitor_state["prev_object_reaching_fixture"] = object_reaching_fixture
    monitor_state["access_object_reach_by_object"] = object_interior_by_object
    monitor_state["access_object_interior_by_object"] = object_interior_by_object

    # ---------------------------------------------------------------------------
    # Mechanism safety: fixture open/close obstacle recovery (mechanism_safety.txt)
    # ---------------------------------------------------------------------------

    def _fixture_geom_map() -> dict[str, set[int]]:
        result: dict[str, set[int]] = {}
        for fname, fixture in getattr(env, "fixtures", {}).items():
            prefix = str(getattr(fixture, "naming_prefix", "") or "")
            worldbody = getattr(fixture, "worldbody", None)
            if worldbody is None:
                continue
            ids: set[int] = set()
            try:
                for geom in worldbody.iter("geom"):
                    gname = geom.get("name")
                    if not gname:
                        continue
                    full = (
                        f"{prefix}{gname}"
                        if prefix and not str(gname).startswith(prefix)
                        else str(gname)
                    )
                    try:
                        ids.add(int(env.sim.model.geom_name2id(full)))
                    except Exception:
                        continue
            except Exception:
                pass
            if ids:
                result[str(fname)] = ids
        return result

    _fgeom_map = _fixture_geom_map()
    _all_fixture_geom_ids = set().union(*_fgeom_map.values()) if _fgeom_map else set()

    # robot_fixture_contact: robot contacts any fixture geom
    _robot_fixture_contact_raw = False
    _contacted_fixture_name: str | None = None
    for _i in range(env.sim.data.ncon):
        try:
            _con = env.sim.data.contact[_i]
            _g1, _g2 = int(_con.geom1), int(_con.geom2)
        except Exception:
            continue
        _is_robot1 = _g1 in robot_geom_ids
        _is_robot2 = _g2 in robot_geom_ids
        if not (_is_robot1 or _is_robot2):
            continue
        _other = _g2 if _is_robot1 else _g1
        if _other in _all_fixture_geom_ids:
            _robot_fixture_contact_raw = True
            for _fname, _fids in _fgeom_map.items():
                if _other in _fids:
                    _contacted_fixture_name = _fname
                    break
            break
    # No debounce: robot_fixture_contact tracks the raw contact check
    # directly (previously required CONTACT_PERSISTENCE_FRAMES consecutive
    # frames on/off, but that constant is 1, so this is behaviorally
    # unchanged).
    robot_fixture_contact = _bool(_robot_fixture_contact_raw)

    # track active fixture across frames; retain last known name when not in contact
    _prev_active_fixture = monitor_state.get("active_fixture_contact_name")
    active_fixture_contact_name: str | None = (
        _contacted_fixture_name if _robot_fixture_contact_raw else _prev_active_fixture
    )

    # current normalized joint position for the active fixture
    def _fixture_norm_joint_pos(fname: str | None) -> float | None:
        if fname is None:
            return None
        fixture = _fixture_by_name(fname)
        if fixture is None:
            return None
        joint_names = getattr(fixture, "door_joint_names", []) or []
        if not joint_names:
            return None
        try:
            state = fixture.get_joint_state(env, joint_names)
            if state:
                return sum(state.values()) / len(state)
        except Exception:
            pass
        return None

    _curr_jpos = _fixture_norm_joint_pos(active_fixture_contact_name)
    _prev_jpos_map: dict[str, float] = dict(
        monitor_state.get("prev_fixture_joint_pos") or {}
    )
    _prev_jpos = (
        _prev_jpos_map.get(str(active_fixture_contact_name))
        if active_fixture_contact_name
        else None
    )

    _delta = (
        (_curr_jpos - _prev_jpos)
        if (_curr_jpos is not None and _prev_jpos is not None)
        else None
    )
    fixture_is_opening = _bool(
        _delta is not None and _delta > FIXTURE_MOTION_DELTA_THRESHOLD
    )
    fixture_is_closing = _bool(
        _delta is not None and _delta < -FIXTURE_MOTION_DELTA_THRESHOLD
    )

    # fixture_fully_open / fixture_fully_closed for active fixture
    fixture_fully_open = _bool(
        _fixture_open(active_fixture_contact_name) or access_fixture_fully_open
    )
    fixture_fully_closed = _bool(_fixture_closed(active_fixture_contact_name))

    # fixture_obstacle_contact: active fixture body contacts a non-robot, non-fixture geom
    # build set of geom IDs that were in static contact with this fixture from the start;
    # these are structural parts (handle mesh, inner panel, etc.) and must be excluded
    _af_initial_contact_geom_ids: set[int] = set()
    if active_fixture_contact_name is not None:
        _af_geom_ids_for_initial = _fgeom_map.get(
            str(active_fixture_contact_name), set()
        )
        for _pair in ignored_initial_contact_pairs or set():
            try:
                _pg1, _pg2 = int(_pair[0]), int(_pair[1])
            except Exception:
                continue
            if _pg1 in _af_geom_ids_for_initial:
                _af_initial_contact_geom_ids.add(_pg2)
            elif _pg2 in _af_geom_ids_for_initial:
                _af_initial_contact_geom_ids.add(_pg1)

    _fixture_obstacle_contact_raw = False
    _fixture_obstacle_geom_name: str | None = None
    if active_fixture_contact_name is not None:
        _af_geom_ids = _fgeom_map.get(str(active_fixture_contact_name), set())
        for _i in range(env.sim.data.ncon):
            try:
                _con = env.sim.data.contact[_i]
                _g1, _g2 = int(_con.geom1), int(_con.geom2)
            except Exception:
                continue
            _in_af1 = _g1 in _af_geom_ids
            _in_af2 = _g2 in _af_geom_ids
            if not (_in_af1 or _in_af2):
                continue
            _other = _g2 if _in_af1 else _g1
            if _other in robot_geom_ids:
                continue
            if _other in _all_fixture_geom_ids:
                continue
            if _other in _af_initial_contact_geom_ids:
                continue
            _fixture_obstacle_contact_raw = True
            try:
                _fixture_obstacle_geom_name = str(
                    env.sim.model.geom_id2name(int(_other)) or _other
                )
            except Exception:
                _fixture_obstacle_geom_name = str(_other)
            break
    # No debounce: fixture_obstacle_contact tracks the raw contact check
    # directly (previously required CONTACT_PERSISTENCE_FRAMES consecutive
    # frames on/off, but that constant is 1, so this is behaviorally
    # unchanged).
    fixture_obstacle_contact = _bool(_fixture_obstacle_contact_raw)

    # continue_fixture_open / continue_fixture_close
    continue_fixture_open = _bool(robot_fixture_contact and fixture_is_opening)
    continue_fixture_close = _bool(robot_fixture_contact and fixture_is_closing)

    # fixture_open/close_obstacle_hit: smoothing is now in the component predicates
    fixture_open_obstacle_hit = _bool(
        robot_fixture_contact and fixture_is_opening and fixture_obstacle_contact
    )
    fixture_close_obstacle_hit = _bool(
        robot_fixture_contact and fixture_is_closing and fixture_obstacle_contact
    )

    def _expanded_aabb(aabb, margin: float):
        amin, amax = aabb
        return (
            np.asarray(amin, dtype=float) - margin,
            np.asarray(amax, dtype=float) + margin,
        )

    def _object_inside_or_supported_by_fixture(oname: str, fname: str) -> bool:
        try:
            if OU.obj_inside_of(env, str(oname), str(fname), partial_check=True):
                return True
        except Exception:
            pass
        return _fixture_rack_contact(str(fname), str(oname))

    def _fixture_retract_path_blockers(fname: str | None) -> list[str] | None:
        if fname is None:
            return []
        fixture_aabb = _fixture_aabb(str(fname))
        if fixture_aabb is None:
            return None
        corridor_aabb = _expanded_aabb(fixture_aabb, PATH_OBSTRUCTION_OVERLAP_ALLOWANCE)
        blockers = []
        for oname in all_object_names:
            if _object_inside_or_supported_by_fixture(str(oname), str(fname)):
                continue
            blocker_aabb = _object_aabb(str(oname))
            if blocker_aabb is None:
                continue
            if _aabb_obstructs_path(blocker_aabb, corridor_aabb):
                blockers.append(str(oname))
        return sorted(blockers)

    fixture_open_retract_path_blockers = _fixture_retract_path_blockers(
        active_fixture_contact_name
    )
    fixture_open_retract_path_clear = _bool(fixture_open_retract_path_blockers == [])

    fixture_close_retract_path_blockers = _fixture_retract_path_blockers(
        active_fixture_contact_name
    )
    fixture_close_retract_path_clear = _bool(fixture_close_retract_path_blockers == [])

    # fixture_open_retracting / fixture_close_retracting
    fixture_open_retracting = _bool(
        not continue_fixture_open
        and fixture_open_retract_path_clear
        and not fixture_open_obstacle_hit
    )
    fixture_close_retracting = _bool(
        not continue_fixture_close
        and fixture_close_retract_path_clear
        and not fixture_close_obstacle_hit
    )

    predicates = {
        "forbidden_contact": forbidden_contact,
        "allowed_contact": allowed_contact,
        "robot_correct_manipulated_object_contact": robot_correct_manipulated_object_contact,
        "robot_correct_fixture_contact": robot_correct_fixture_contact,
        "correct_manipulated_object_correct_fixture_contact": correct_manipulated_object_correct_fixture_contact,
        "correct_manipulated_object_correct_receive_object_contact": correct_manipulated_object_correct_receive_object_contact,
        "grasped_object_exists": _bool(active_object is not None and object_grasped),
        "object_grasped": object_grasped,
        "object_stable": object_stable,
        "object_sync": object_sync,
        "grasp_point_stable": grasp_point_stable,
        "object_upright": object_upright,
        "object_grasped_safe": object_grasped_safe,
        "object_released": object_released,
        "object_supported": object_supported,
        "object_supported_on_correct": object_supported_on_correct,
        "gripper_away_from_object": gripper_away_from_object,
        "object_settled": object_settled,
        "release_object_settle_timeout": release_object_settle_timeout,
        "object_settle_timeout": object_settle_timeout,
        "gripper_is_opening": gripper_is_opening,
        "sanitized": sanitized,
        "robot_contact_raw_contaminated": robot_contact_raw_contaminated,
        "object_is_rte": object_is_rte,
        "robot_contact_clean": robot_contact_clean,
        "gripper_is_closing": gripper_is_closing,
        "gripper_moving_towards_object": gripper_moving_towards_object,
        "gripper_near_object": gripper_near_object,
        "skill_pick_onset": skill_pick_onset,
        "skill_place_onset": skill_place_onset,
        "gripper_moving_towards_target": gripper_moving_towards_target,
        "gripper_near_target": gripper_near_target,
        "skill_press_onset": skill_press_onset,
        "skill_turn_onset": skill_turn_onset,
        "skill_slide_onset": skill_slide_onset,
        "skill_twist_onset": skill_twist_onset,
        "skill_open_close_onset": skill_open_close_onset,
        "skill_dump_onset": skill_dump_onset,
        "object_region_clear": object_region_clear,
        "object_upright_if_receptacle": object_upright_if_receptacle,
        "preconditions_satisfied_pick": preconditions_satisfied_pick,
        "support_region_clear": support_region_clear,
        "dump_support_region_clear": dump_support_region_clear,
        "support_stable": support_stable,
        "support_geometry_valid": support_geometry_valid,
        "support_type_matches_object": support_type_matches_object,
        "dump_support_geometry_valid": dump_support_geometry_valid,
        "dump_support_type_matches_content": dump_support_type_matches_content,
        "dump_support_hygienic_for_content": dump_support_hygienic_for_content,
        "dump_support_objects_clean_for_content": dump_support_objects_clean_for_content,
        "dump_support_not_cluttered_for_fragile_content": dump_support_not_cluttered_for_fragile_content,
        "support_hygienic_for_manipulated_object": support_hygienic_for_manipulated_object,
        "support_objects_clean_for_manipulated_object": support_objects_clean_for_manipulated_object,
        "support_not_cluttered_for_fragile_manipulated_object": support_not_cluttered_for_fragile_manipulated_object,
        "preconditions_satisfied_place": preconditions_satisfied_place,
        "target_region_clear": target_region_clear,
        "target_stable": target_stable,
        "fixture_ready_for_press": fixture_ready_for_press,
        "fixture_ready_for_turn": fixture_ready_for_turn,
        "fixture_ready_for_slide": fixture_ready_for_slide,
        "fixture_ready_for_twist": fixture_ready_for_twist,
        "fixture_ready_for_open_close": fixture_ready_for_open_close,
        "slide_path_clear": slide_path_clear,
        "target_receptacle_upright_if_has_contents": target_receptacle_upright_if_has_contents,
        "articulation_path_clear": articulation_path_clear,
        "preconditions_satisfied_press": preconditions_satisfied_press,
        "preconditions_satisfied_turn": preconditions_satisfied_turn,
        "preconditions_satisfied_slide": preconditions_satisfied_slide,
        "preconditions_satisfied_twist": preconditions_satisfied_twist,
        "preconditions_satisfied_open_close": preconditions_satisfied_open_close,
        "preconditions_satisfied_dump": preconditions_satisfied_dump,
        "robot_fixture_contact": robot_fixture_contact,
        "fixture_is_opening": fixture_is_opening,
        "fixture_is_closing": fixture_is_closing,
        "fixture_fully_open": fixture_fully_open,
        "fixture_fully_closed": fixture_fully_closed,
        "fixture_obstacle_contact": fixture_obstacle_contact,
        "continue_fixture_open": continue_fixture_open,
        "continue_fixture_close": continue_fixture_close,
        "fixture_open_retract_path_clear": fixture_open_retract_path_clear,
        "fixture_close_retract_path_clear": fixture_close_retract_path_clear,
        "fixture_open_obstacle_hit": fixture_open_obstacle_hit,
        "fixture_close_obstacle_hit": fixture_close_obstacle_hit,
        "fixture_open_retracting": fixture_open_retracting,
        "fixture_close_retracting": fixture_close_retracting,
        "containment_transfer_event": containment_transfer_event,
        "fixture_output_started": fixture_output_started,
        "fixture_output_stopped": fixture_output_stopped,
        "fixture_content_output_started": fixture_content_output_started,
        "liquid_transfer_event": liquid_transfer_event,
        "solid_transfer_event": solid_transfer_event,
        "liquid_settled": liquid_settled,
        "solid_settled": solid_settled,
        "solid_misplacement": solid_misplacement,
        "misplaced_solid_removed": misplaced_solid_removed,
        "misplaced_solid_recollected": misplaced_solid_recollected,
        "content_settled": content_settled,
        "content_is_supported": content_is_supported,
        "content_stable": content_stable,
        "support_type_matches_content": support_type_matches_content,
        "content_is_liquid": content_is_liquid,
        "content_is_solid": content_is_solid,
        "one_object_in_microwave": one_object_in_microwave,
        "two_or_more_objects_in_microwave": two_or_more_objects_in_microwave,
        "microwave_empty": microwave_empty,
        "reach_in_fixture": reach_in_fixture,
        "gripper_in_fixture": gripper_in_fixture,
        "object_reach_in_fixture": object_reach_in_fixture,
        "object_in_fixture": object_in_fixture,
        "object_in_same_fixture": object_in_same_fixture,
    }
    monitor_state["prev_predicates"] = dict(predicates)
    monitor_state["active_object"] = active_object
    monitor_state["prev_object_grasped"] = object_grasped
    monitor_state["awaiting_settle"] = awaiting_settle
    monitor_state["settle_watch_object"] = settle_watch_object
    monitor_state["settle_watch_age"] = settle_watch_age
    monitor_state["settle_release_frame"] = settle_release_frame
    monitor_state["settle_release_object"] = settle_release_object
    monitor_state["source_support_fixtures"] = sorted(source_support_fixtures)
    monitor_state["source_support_objects"] = sorted(source_support_objects)
    monitor_state["prev_object_grasped_safe"] = object_grasped_safe
    monitor_state["last_timestep"] = current_timestep
    monitor_state["active_fixture_contact_name"] = active_fixture_contact_name
    if active_fixture_contact_name is not None and _curr_jpos is not None:
        _prev_jpos_map[str(active_fixture_contact_name)] = _curr_jpos
    monitor_state["prev_fixture_joint_pos"] = _prev_jpos_map

    sections = {
        "predicates": {name: _entry(name, value) for name, value in predicates.items()}
    }
    violation_evidence = {
        "forbidden_contact_pairs": forbidden_contact_pairs,
        "considered_contact_pairs": considered_contact_pairs,
        "contact_policy_action_fixture_names": sorted(
            contact_policy_action_fixture_names
        ),
        "contact_policy_action_fixture_actions": contact_policy_action_fixture_actions,
        "contact_policy_action_fixture_component_geom_names": sorted(
            _geom_name(geom_id) for geom_id in contact_policy_action_fixture_geom_ids
        ),
        "contact_policy_object_fixture_names": sorted(
            contact_policy_object_fixture_names
        ),
        "contact_policy_object_fixture_geom_names": sorted(
            _geom_name(geom_id) for geom_id in contact_policy_object_fixture_geom_ids
        ),
        "task_referenced_object_fixture_contact_names": sorted(
            task_referenced_object_fixture_contact_names
        ),
        "grasp_rule_object": active_object,
        "raw_grasped_objects": sorted(str(name) for name in grasped_names),
        "raw_contacted_objects": sorted(str(name) for name in robot_contacted_names),
        "object_grasp_candidate": monitor_state.get("object_grasp_candidate"),
        "object_grasped_object": grasped_object,
        "safe_grasp_object": active_object if object_grasped_safe else None,
        "released_objects_waiting_to_settle": [evidence_settle_object]
        if (awaiting_settle or object_settle_timeout) and evidence_settle_object
        else [],
        "object_release_frame": evidence_release_frame,
        "object_settle_timeout_frame": evidence_timeout_frame,
        "active_object": active_object,
        "object_supported_on_correct": object_supported_on_correct,
        "object_stable": object_stable,
        "object_sync": object_sync,
        "grasp_point_stable": grasp_point_stable,
        "grasp_point_drift": grasp_point_drift,
        "grasp_point_drift_threshold": GRASP_POINT_DRIFT_THRESHOLD,
        "grasp_point_drift_false_count": drift_false_count,
        "grasp_point_external_contact": grip_external_contact,
        "grasp_point_baseline_object": grip_baseline_object,
        "gripper_away_from_object": gripper_away_from_object,
        "release_object_settle_timeout": release_object_settle_timeout,
        "object_settle_timeout": object_settle_timeout,
        "forbidden_contact_candidate": forbidden_candidate,
        "robot_contact_raw_candidate": raw_contact_candidate,
        "contamination_transfer_pair": transfer_pair,
        "contamination_transfer_source": (
            _contamination_entity_key(transfer_source) if transfer_source else None
        ),
        "contamination_transfer_target": (
            _contamination_entity_key(transfer_target) if transfer_target else None
        ),
        "robot_contact_raw_activated_frame": robot_contact_raw_activated_frame,
        "robot_contact_clean_candidate": robot_contact_clean_candidate,
        "correct_manipulated_object_original_support_contact": correct_manipulated_object_original_support_contact,
        "source_support_fixtures_for_active_object": sorted(
            active_source_fixture_names
        ),
        "source_support_objects_for_active_object": sorted(active_source_object_names),
        "robot_contact_raw_sources": sorted(robot_contact_raw_sources),
        "raw_contact_sources_now": sorted(raw_contact_sources_now),
        "raw_contact_surface_sources_now": sorted(raw_contact_surface_sources_now),
        "contaminated_objects": sorted(contaminated_objects),
        "contaminated_fixtures": sorted(contaminated_fixtures),
        "robot_contact_clean_objects": robot_contact_clean_objects,
        "robot_contact_clean_objects_now": sorted(robot_contact_clean_objects_now),
        "skill_pick_onset_candidate_count": pick_onset_count,
        "skill_pick_onset_fired_object": fired_pick_object,
        "pick_approach_object": pick_approach_object,
        "pick_precondition_object": pick_precondition_object,
        "pick_object_stable": pick_object_stable,
        "pick_approach_candidate_object": monitor_state.get(
            "pick_approach_candidate_object"
        ),
        "pick_approach_candidate_count": pick_approach_count,
        "pick_approach_false_count": pick_approach_false_count,
        "nearest_gripper_object": nearest_gripper_object,
        "nearest_gripper_object_distance": nearest_gripper_object_distance,
        "skill_place_onset_object": place_onset_object,
        "skill_place_onset_candidate_count": place_onset_count,
        "skill_place_onset_fired_object": fired_place_object,
        "approach_target": approach_target,
        "target_precondition_name": target_precondition_name,
        "nearest_gripper_target": nearest_gripper_target,
        "nearest_gripper_target_distance": nearest_gripper_target_distance,
        "nearest_gripper_targets_by_action": action_nearest_targets,
        "nearest_gripper_target_distances_by_action": action_nearest_distances,
        "target_approach_candidate": monitor_state.get("target_approach_candidate"),
        "target_approach_candidate_count": target_approach_count,
        "target_approach_false_count": target_approach_false_count,
        "target_approach_candidates_by_action": target_approach_candidates_by_action,
        "target_approach_counts_by_action": target_approach_counts_by_action,
        "target_approach_false_counts_by_action": target_approach_false_counts_by_action,
        "approach_target_by_action": approach_target_by_action,
        "skill_press_onset_candidate_count": press_onset_count,
        "skill_press_onset_fired_target": fired_press_target,
        "skill_turn_onset_candidate_count": turn_onset_count,
        "skill_turn_onset_fired_target": fired_turn_target,
        "skill_slide_onset_candidate_count": slide_onset_count,
        "skill_slide_onset_fired_target": fired_slide_target,
        "skill_twist_onset_candidate_count": twist_onset_count,
        "skill_twist_onset_fired_target": fired_twist_target,
        "skill_open_close_onset_candidate_count": open_close_onset_count,
        "skill_open_close_onset_fired_target": fired_open_close_target,
        "skill_dump_onset_candidate_count": dump_onset_count,
        "skill_dump_onset_fired_object": fired_dump_object,
        "skill_dump_onset_fired_content_names": sorted(fired_dump_content_set),
        "skill_dump_onset_content_names": monitor_state.get(
            "skill_dump_onset_content_names", []
        ),
        "grasped_receptacle_has_contents": grasped_receptacle_has_contents,
        "grasped_receptacle_content_names": current_grasped_receptacle_contents,
        "previous_grasped_receptacle_content_names": sorted(previous_content_set),
        "raw_dump_left_content_names": raw_dump_left_content_names,
        "dump_left_content_names": dump_left_content_names,
        "raw_grasped_receptacle_is_upright": raw_grasped_receptacle_is_upright,
        "grasped_receptacle_is_upright": grasped_receptacle_is_upright,
        "grasped_receptacle_upright_false_count": (
            grasped_receptacle_upright_false_count
        ),
        "target_region_blockers": target_region_blockers,
        "target_region_blockers_by_action": {
            "press": target_region_blockers_press,
            "turn": target_region_blockers_turn,
            "slide": target_region_blockers_slide,
            "twist": target_region_blockers_twist,
            "open_close": target_region_blockers_open_close,
        },
        "target_region_blocker_details_by_action": {
            "press": _blocker_details(target_region_blockers_press),
            "turn": _blocker_details(target_region_blockers_turn),
            "slide": _blocker_details(target_region_blockers_slide),
            "twist": _blocker_details(target_region_blockers_twist),
            "open_close": _blocker_details(target_region_blockers_open_close),
        },
        "press_target": press_target,
        "turn_target": turn_target,
        "slide_target": slide_target,
        "twist_target": twist_target,
        "open_close_target": open_close_target,
        "twist_receptacle_name": twist_receptacle_name,
        "target_receptacle_has_contents": target_receptacle_has_contents,
        "target_receptacle_upright": target_receptacle_upright,
        "fixture_ready_for_press": fixture_ready_for_press,
        "fixture_ready_for_turn": fixture_ready_for_turn,
        "fixture_ready_for_slide": fixture_ready_for_slide,
        "fixture_ready_for_twist": fixture_ready_for_twist,
        "fixture_ready_for_open_close": fixture_ready_for_open_close,
        "fixture_ready_reasons": fixture_ready_reasons,
        "action_target_candidates": action_candidates_by_name,
        "inferred_support_kind": sup_kind,
        "inferred_support_name": sup_name,
        "object_region_blockers": object_region_blockers,
        "object_region_allowed_supports": object_region_allowed_supports,
        "support_region_blockers": support_region_blockers,
        "support_region_carried_content_exclusions": sorted(
            carried_content_blocker_exclusions
        ),
        "dump_support_region_blockers": dump_support_region_blockers,
        "dump_content_names_for_preconditions": dump_content_names_for_preconditions,
        "dump_content_kind_for_preconditions": dump_content_kind_for_preconditions,
        "dump_support_geometry_valid": dump_support_geometry_valid,
        "dump_support_type_matches_content": dump_support_type_matches_content,
        "dump_support_hygienic_for_content": dump_support_hygienic_for_content,
        "dump_support_objects_clean_issues": dump_support_objects_clean_issues,
        "dump_support_objects_clean_for_content": dump_support_objects_clean_for_content,
        "dump_support_clutter_objects": dump_support_clutter_objects,
        "dump_support_not_cluttered_for_fragile_content": dump_support_not_cluttered_for_fragile_content,
        "support_objects_clean_issues": support_objects_clean_issues,
        "support_clutter_objects": support_clutter_objects,
        "support_clutter_count": len(support_clutter_objects),
        "support_type_mismatch_reason": (
            "food object on structural fixture body"
            if not support_type_matches_object
            else None
        ),
        "support_hygiene_reason": (
            f"support {sup_name} is raw or contaminated"
            if not support_hygienic_for_manipulated_object and sup_name
            else None
        ),
        "preconditions_satisfied_pick": preconditions_satisfied_pick,
        "preconditions_satisfied_place": preconditions_satisfied_place,
        "preconditions_satisfied_press": preconditions_satisfied_press,
        "preconditions_satisfied_turn": preconditions_satisfied_turn,
        "preconditions_satisfied_slide": preconditions_satisfied_slide,
        "preconditions_satisfied_twist": preconditions_satisfied_twist,
        "preconditions_satisfied_open_close": preconditions_satisfied_open_close,
        "preconditions_satisfied_dump": preconditions_satisfied_dump,
        "ignored_initial_contact_pairs": [
            _contact_pair_names(pair) for pair in sorted(ignored_initial_contact_pairs)
        ],
        "removed_initial_contact_pairs": [
            _contact_pair_names(pair) for pair in sorted(removed_initial_contact_pairs)
        ],
        # mechanism safety
        "mechanism_active_fixture": active_fixture_contact_name,
        "mechanism_fixture_joint_pos": _curr_jpos,
        "mechanism_fixture_fully_open_threshold": FIXTURE_FULLY_OPEN_THRESHOLD,
        "mechanism_fixture_fully_closed_threshold": FIXTURE_FULLY_CLOSED_THRESHOLD,
        "mechanism_obstacle_geom": _fixture_obstacle_geom_name
        if fixture_obstacle_contact
        else None,
        "mechanism_open_obstacle_hit": fixture_open_obstacle_hit,
        "mechanism_close_obstacle_hit": fixture_close_obstacle_hit,
        "mechanism_open_retract_path_blockers": fixture_open_retract_path_blockers,
        "mechanism_close_retract_path_blockers": fixture_close_retract_path_blockers,
        "mechanism_open_retracting_blocked_continue": continue_fixture_open
        if not fixture_open_retracting
        else False,
        "mechanism_open_retracting_blocked_path": not fixture_open_retract_path_clear
        if not fixture_open_retracting
        else False,
        "mechanism_open_retracting_blocked_obstacle": fixture_open_obstacle_hit
        if not fixture_open_retracting
        else False,
        "mechanism_close_retracting_blocked_continue": continue_fixture_close
        if not fixture_close_retracting
        else False,
        "mechanism_close_retracting_blocked_path": not fixture_close_retract_path_clear
        if not fixture_close_retracting
        else False,
        "mechanism_close_retracting_blocked_obstacle": fixture_close_obstacle_hit
        if not fixture_close_retracting
        else False,
        # containment safety
        "containment_active_transfer": active_transfer,
        "containment_transfer_event": containment_transfer_event,
        "containment_source_kind": (active_transfer or {}).get("source_kind")
        if active_transfer
        else None,
        "containment_source_name": (active_transfer or {}).get("source_name")
        if active_transfer
        else None,
        "containment_content_names": content_names,
        "containment_content_kind": content_kind,
        "containment_receiver_kind": (active_transfer or {}).get("receiver_kind")
        if active_transfer
        else None,
        "containment_receiver_name": (active_transfer or {}).get("receiver_name")
        if active_transfer
        else None,
        "containment_content_is_supported": content_is_supported,
        "containment_content_stable": content_stable,
        "containment_support_type_matches_content": support_type_matches_content,
        "containment_content_settled": content_settled,
        "containment_settle_timeout": containment_settle_timeout,
        "containment_settle_timeout_frame": monitor_frame_index
        if containment_settle_timeout
        else None,
        "containment_fixture_output_target": fixture_output_name,
        "containment_fixture_output_kind": fixture_output_kind,
        "containment_fixture_output_active": fixture_output_active,
        "containment_fixture_output_stopped": fixture_output_stopped,
        # access/enclosure safety
        "access_openable_fixtures": openable_fixture_names,
        "access_microwave_fixture": microwave_name,
        "access_microwave_objects": raw_microwave_objects,
        "access_microwave_empty_check_objects": raw_microwave_empty_check_objects,
        "access_microwave_entering_payload_exclusions": sorted(
            microwave_entering_payload_exclusions
        ),
        "access_microwave_object_count": microwave_stable_count,
        "access_microwave_empty_count": microwave_empty_count,
        "access_gripper_fixture": gripper_fixture_name,
        "access_closing_fixtures": access_closing_fixture_names,
        "access_open_close_suppressed_fixtures": access_open_close_suppressed_fixtures,
        "access_active_fixture": access_active_fixture,
        "access_fixture_fully_open": access_fixture_fully_open,
        "access_object_fixture": access_object_fixture,
        "access_object_in_fixture_name": object_fixture_name,
        "access_object_in_same_fixture": object_in_same_fixture,
    }
    changes = []
    prev_values = monitor_state.get("prev_values", {})
    for name, entry in sections["predicates"].items():
        old_value = prev_values.get(name)
        new_value = entry["value"]
        if old_value is not None and old_value != new_value:
            changes.append(
                {
                    "name": name,
                    "section": "predicates",
                    "category": "predicate",
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )
    monitor_state["prev_values"] = {
        name: entry["value"] for name, entry in sections["predicates"].items()
    }
    timestep = current_timestep
    if _debug_enabled() and (
        timestep < 0 or timestep % _debug_every_n() == 0 or changes or forbidden_contact
    ):
        true_predicates = [name for name, value in predicates.items() if value]
        _debug_print(
            "step={step} active={active} active_grasped={active_grasped} pending_release={pending} "
            "manipulated={manipulated} fixtures={fixtures} receive={receive} "
            "ncon={ncon} forbidden_pairs={forbidden} true={true} changes={changes}".format(
                step=timestep,
                active=obj_name,
                active_grasped=(active_object if object_grasped else None),
                pending=(active_object if awaiting_settle else None),
                manipulated=sorted(manipulated_object_names),
                fixtures=sorted(_target_fixture_names()),
                receive=sorted(
                    _target_object_names(manipulated_object_names, receive_object_names)
                ),
                ncon=contact_number,
                forbidden=forbidden_contact_pairs if forbidden_contact_pairs else [],
                true=true_predicates,
                changes=[
                    f"{item['name']}:{item['old_value']}->{item['new_value']}"
                    for item in changes
                ],
            )
        )

    return {
        "task_name": env.__class__.__name__,
        "task_language": ((static_info or {}).get("task", {}) or {}).get(
            "language", ""
        ),
        "supported_task": True,
        "role_sets": {
            "manipulated_objects": sorted(manipulated_object_names),
            "active_object": active_object,
            "pick_approach_object": pick_approach_object,
            "pick_precondition_object": pick_precondition_object,
            "raw_contacted_objects": sorted(
                str(name) for name in robot_contacted_names
            ),
            "raw_grasped_objects": sorted(str(name) for name in grasped_names),
            "object_grasp_candidate": monitor_state.get("object_grasp_candidate"),
            "target_fixtures": sorted(_target_fixture_names()),
            "receive_objects": sorted(target_object_names),
            "target_fixtures_by_object": {
                name: sorted(values)
                for name, values in sorted(target_fixtures_by_object.items())
            },
            "receive_objects_by_object": {
                name: sorted(values)
                for name, values in sorted(target_objects_by_object.items())
            },
            "source_supports_by_object": {
                active_object: {
                    "fixtures": sorted(source_support_fixtures),
                    "objects": sorted(source_support_objects),
                }
            }
            if active_object
            else {},
            "ignored_initial_contact_pair_count": len(ignored_initial_contact_pairs),
            "removed_initial_contact_pair_count": len(removed_initial_contact_pairs),
            "ignored_initial_contact_pairs": [
                _contact_pair_names(pair)
                for pair in sorted(ignored_initial_contact_pairs)
            ],
            "removed_initial_contact_pairs": [
                _contact_pair_names(pair)
                for pair in sorted(removed_initial_contact_pairs)
            ],
        },
        "sections": sections,
        "violation_evidence": violation_evidence,
        "formulas": {},
        "changes": changes,
        "summary_lines": [
            entry["readout"] for entry in sections["predicates"].values()
        ],
    }
