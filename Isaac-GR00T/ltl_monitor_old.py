"""Structured LTL-style monitoring for RoboCasa kitchen tasks.

This file mirrors the hierarchy in ``ltl_design/Coffee Preparation.txt``:

1. Low-level action predicates
2. Object / state predicates
3. Sequencing predicates
4. Derived safety / context predicates
5. LTL formulas grouped by the same sections as the design document

Some PDF predicates cannot be measured exactly in RoboCasa because the simulator
does not expose liquid state, contamination labels, or perception confidence.
Those predicates are still implemented as structured monitor functions using the
best available privileged-state proxy, and the approximation is documented in
the static spec.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures import Counter

# Motion threshold used to classify the robot as actively moving.
MOVE_SPEED_THRESHOLD = 0.02

# Threshold for a forceful or potentially unsafe contact.
EEF_FORCE_THRESHOLD = 5.0

# Threshold for contact forces that are treated as excessive.
CONTACT_FORCE_THRESHOLD = 10.0

# Mug-specific contact threshold (task contextual).
MUG_CONTACT_FORCE_THRESHOLD = 8.0

# Object motion thresholds.
OBJ_SLOW_LIN_THRESHOLD = 0.05
OBJ_SLOW_ANG_THRESHOLD = 0.25
SYNC_SPEED_THRESHOLD = 0.05

# Root-motion threshold (kept in sync with MOVE_SPEED_THRESHOLD for now).
ROOT_MOVE_SPEED_THRESHOLD = MOVE_SPEED_THRESHOLD

# Maximum allowed release height during placement.
PLACE_HEIGHT_THRESHOLD = 0.08

# Number of recent steps used for temporal predicates like settled / stable.
TEMPORAL_WINDOW = 5

# Number of consecutive blocked steps that count as stalled.
STALL_WINDOW = 8

# Maximum allowed repeated retries before the retry budget is exceeded.
RETRY_BUDGET = 3


def _safe_bool(value: Any) -> bool:
    """Convert arbitrary values into a safe boolean."""
    try:
        return bool(value)
    except Exception:
        return False


def _safe_float(value: Any, default: float = np.nan) -> float:
    """Convert arbitrary values into a float without raising errors."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_norm(vector: Optional[Iterable[float]]) -> float:
    """Compute a vector norm safely from list-like input."""
    if vector is None:
        return float(np.nan)
    try:
        arr = np.asarray(vector, dtype=float).reshape(-1)
    except Exception:
        return float(np.nan)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return float(np.nan)
    return float(np.linalg.norm(arr))


def _get_task_name(env: Any) -> str:
    """Return the current environment class name."""
    return getattr(env, "__class__", type(env)).__name__


def _get_task_language(static_info: Dict[str, Any]) -> str:
    """Return the natural-language task description from privileged static info."""
    return str((((static_info or {}).get("task", {}) or {}).get("language", "")))


def _get_fixture(env: Any, name: str) -> Any:
    """Return a fixture object by name when it exists."""
    try:
        return env.fixtures.get(name)
    except Exception:
        return None


def _get_object_position(env: Any, obj_name: str = "obj") -> Optional[np.ndarray]:
    """Return the target object's world position."""
    try:
        body_id = int(env.obj_body_id[obj_name])
        pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=float).reshape(-1)
    except Exception:
        return None
    if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
        return None
    return pos[:3]


def _get_eef_position(dynamic_info: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return the end-effector position from the serialized dynamic state."""
    pose = ((dynamic_info or {}).get("robot", {}) or {}).get("end_effector_pose", {}) or {}
    try:
        pos = np.asarray(pose.get("position"), dtype=float).reshape(-1)
    except Exception:
        return None
    if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
        return None
    return pos[:3]


def _get_linear_speed(dynamic_info: Dict[str, Any], key: str) -> float:
    """Return the linear speed for a robot component stored in dynamic info."""
    velocity = ((dynamic_info or {}).get("robot", {}) or {}).get(key, {}) or {}
    return _safe_norm(velocity.get("linear"))


def _get_angular_speed(dynamic_info: Dict[str, Any], key: str) -> float:
    """Return the angular speed for a robot component stored in dynamic info."""
    velocity = ((dynamic_info or {}).get("robot", {}) or {}).get(key, {}) or {}
    return _safe_norm(velocity.get("angular"))


def _get_object_speed(dynamic_info: Dict[str, Any], obj_name: str = "obj") -> float:
    """Return the target object's linear speed."""
    velocity = (
        (((dynamic_info or {}).get("scene", {}) or {}).get("objects", {}) or {})
        .get(
            obj_name,
            {},
        )
        .get("velocity", {})
    )
    return _safe_norm((velocity or {}).get("linear"))


def _get_object_angular_speed(dynamic_info: Dict[str, Any], obj_name: str = "obj") -> float:
    """Return the target object's angular speed."""
    velocity = (
        (((dynamic_info or {}).get("scene", {}) or {}).get("objects", {}) or {})
        .get(
            obj_name,
            {},
        )
        .get("velocity", {})
    )
    return _safe_norm((velocity or {}).get("angular"))


def _get_eef_force(dynamic_info: Dict[str, Any]) -> float:
    """Return the end-effector force norm from privileged physics."""
    physics = (dynamic_info or {}).get("physics", {}) or {}
    return _safe_float(physics.get("end_effector_force_norm"), default=np.nan)


def _get_robot_contact_count(dynamic_info: Dict[str, Any]) -> float:
    """Return the robot contact count from privileged physics."""
    physics = (dynamic_info or {}).get("physics", {}) or {}
    return _safe_float(physics.get("robot_contact_count"), default=0.0)


def _object_upright(env: Any, obj_name: str = "obj", threshold_degrees: float = 15.0) -> bool:
    """Check whether the target object remains upright."""
    try:
        return bool(OU.check_obj_upright(env, obj_name=obj_name, th=threshold_degrees))
    except Exception:
        return False


def _object_grasped(env: Any, obj_name: str = "obj") -> bool:
    """Check whether the robot is currently grasping the target object."""
    try:
        return bool(OU.check_obj_grasped(env, obj_name=obj_name))
    except Exception:
        return False


def _gripper_far_from_object(env: Any, obj_name: str = "obj", threshold: float = 0.25) -> bool:
    """Check whether the gripper has released and moved away from the object."""
    try:
        return bool(OU.gripper_obj_far(env, obj_name=obj_name, th=threshold))
    except Exception:
        return False


def _detect_robot_collision_excluding(
    env: Any, exclude_geom_names: Optional[Iterable[str]] = None
) -> bool:
    """Check for robot collision while ignoring contacts with excluded geometry names."""
    exclude_tokens = {str(name).lower() for name in (exclude_geom_names or []) if name}
    if env.robot_geom_ids is None:
        env.robot_geom_ids = set()
        robot_geoms = EnvUtils.find_elements(
            root=env.robots[0].robot_model.root, tags="geom", return_first=False
        )
        for robot_geom in robot_geoms:
            env.robot_geom_ids.add(env.sim.model.geom_name2id(robot_geom.get("name")))
    for i in range(env.sim.data.ncon):
        geom1 = env.sim.data.contact[i].geom1
        geom2 = env.sim.data.contact[i].geom2
        if (geom1 in env.robot_geom_ids and geom2 not in env.robot_geom_ids) or (
            geom2 in env.robot_geom_ids and geom1 not in env.robot_geom_ids
        ):
            other_geom = geom2 if geom1 in env.robot_geom_ids else geom1
            try:
                other_name = str(env.sim.model.geom_id2name(other_geom) or "").lower()
            except Exception:
                other_name = ""
            if any(token in other_name for token in exclude_tokens):
                continue
            return True
    return False


def _robot_collision(env: Any) -> bool:
    """Check whether the robot is colliding with non-robot geometry."""
    try:
        return bool(EnvUtils.detect_robot_collision(env))
    except Exception:
        return False


def _object_in_cabinet(env: Any, obj_name: str = "obj", cabinet_name: str = "cab") -> bool:
    """Check whether the object is still contacting the cabinet."""
    cabinet = _get_fixture(env, cabinet_name)
    if cabinet is None:
        return False
    try:
        return bool(OU.check_obj_fixture_contact(env, obj_name, cabinet.name))
    except Exception:
        return False


def _cabinet_open(env: Any, cabinet_name: str = "cab", threshold: float = 0.10) -> bool:
    """Check whether any cabinet door is meaningfully open."""
    cabinet = _get_fixture(env, cabinet_name)
    if cabinet is None:
        return False
    try:
        door_state = cabinet.get_door_state(env)
    except Exception:
        return False
    if not isinstance(door_state, dict):
        return False
    for value in door_state.values():
        if _safe_float(value, 0.0) >= threshold:
            return True
    return False


def _fixture_door_velocity(dynamic_info: Dict[str, Any], fixture_name: str = "cab") -> float:
    """Return the largest joint velocity magnitude for a fixture door."""
    fixture_entry = (((dynamic_info or {}).get("scene", {}) or {}).get("fixtures", {}) or {}).get(
        fixture_name,
        {},
    )
    joints = (fixture_entry or {}).get("joints") or {}
    values = []
    for joint_state in joints.values():
        if not isinstance(joint_state, dict):
            continue
        qvel = _safe_float(joint_state.get("qvel"), default=np.nan)
        if np.isfinite(qvel):
            values.append(abs(qvel))
    return float(max(values)) if values else 0.0


def _start_button_pressed(env: Any) -> bool:
    """Check whether the gripper is physically pressing a coffee-machine start button."""
    coffee_machine = _get_fixture(env, "coffee_machine")
    if coffee_machine is None:
        return False
    try:
        buttons = list(getattr(coffee_machine, "_start_button_names", []))
        gripper = env.robots[0].gripper["right"]
    except Exception:
        return False
    for button_name in buttons:
        try:
            if env.check_contact(gripper, f"{coffee_machine.naming_prefix}{button_name}"):
                return True
        except Exception:
            continue
    return False


def _start_button_names(env: Any) -> List[str]:
    """Return the coffee machine start-button geometry names."""
    coffee_machine = _get_fixture(env, "coffee_machine")
    if coffee_machine is None:
        return []
    try:
        return [str(name) for name in getattr(coffee_machine, "_start_button_names", []) if name]
    except Exception:
        return []


def _coffee_machine_on(env: Any) -> bool:
    """Check whether the coffee machine reports being on."""
    coffee_machine = _get_fixture(env, "coffee_machine")
    try:
        return bool(getattr(coffee_machine, "_turned_on", False))
    except Exception:
        return False


def _mug_centered(env: Any, obj_name: str = "obj") -> bool:
    """Check whether the mug is centered under the coffee dispenser."""
    coffee_machine = _get_fixture(env, "coffee_machine")
    if coffee_machine is None:
        return False
    try:
        return bool(coffee_machine.check_receptacle_placement_for_pouring(env, obj_name))
    except Exception:
        return False


def _gripper_far_from_start_button(env: Any) -> bool:
    """Check whether the gripper has moved away from the coffee-machine button."""
    coffee_machine = _get_fixture(env, "coffee_machine")
    if coffee_machine is None:
        return False
    try:
        return bool(coffee_machine.gripper_button_far(env))
    except Exception:
        return False


def _eef_near_fixture(
    dynamic_info: Dict[str, Any], static_info: Dict[str, Any], fixture_name: str, threshold: float
) -> bool:
    """Check whether the end effector is near a named fixture."""
    eef_pos = _get_eef_position(dynamic_info)
    fixture_entry = (
        ((static_info or {}).get("scene_layout", {}) or {}).get("fixtures", {}) or {}
    ).get(
        fixture_name,
        {},
    )
    try:
        fixture_pos = np.asarray(fixture_entry.get("position"), dtype=float).reshape(-1)
    except Exception:
        fixture_pos = None
    if eef_pos is None or fixture_pos is None or fixture_pos.size < 3:
        return False
    return bool(np.linalg.norm(eef_pos - fixture_pos[:3]) <= threshold)


def _record_history(state: Dict[str, Any], key: str, value: bool, limit: int = 32) -> None:
    """Append a boolean to a bounded per-episode history buffer."""
    history = state.setdefault("histories", {}).setdefault(key, [])
    history.append(bool(value))
    if len(history) > limit:
        history.pop(0)


def _recent_all_true(state: Dict[str, Any], key: str, window: int) -> bool:
    """Check whether a boolean history has been true for the whole recent window."""
    history = state.get("histories", {}).get(key, [])
    if len(history) < window:
        return False
    return all(history[-window:])


def _recent_any_true(state: Dict[str, Any], key: str, window: int) -> bool:
    """Check whether a boolean history has been true at least once recently."""
    history = state.get("histories", {}).get(key, [])
    if len(history) == 0:
        return False
    return any(history[-window:])


def _entry(
    *,
    name: str,
    value: bool,
    category: str,
    formula: Optional[str],
    language: str,
    short_readout: str,
) -> Dict[str, Any]:
    """Create a normalized monitor entry for predicates and formulas."""
    bool_value = _safe_bool(value)
    return {
        "name": name,
        "category": category,
        "value": bool_value,
        "formula": formula,
        "language": language,
        "readout": f"{short_readout}: {str(bool_value).lower()}",
    }


def _init_monitor_state(env: Any) -> Dict[str, Any]:
    """Initialize the persistent LTL monitor state stored on the environment."""
    state = getattr(env, "_ltl_monitor_state", None)
    if isinstance(state, dict):
        return state
    state = {
        "prev_values": {},
        "last_timestep": None,
        "phase": "approach",
        "last_phase": "approach",
        "phase_enter_counts": {},
        "retry_count": 0,
        "fault_active_previous_step": False,
        "histories": {},
        "trace_flags": {
            "ever_act_clean": False,
            "ever_act_pick": False,
            "ever_act_place": False,
            "ever_act_press": False,
            "ever_mug_grasped": False,
            "ever_mug_centered": False,
            "ever_mug_set_down": False,
            "ever_coffee_machine_on": False,
            "ever_subtask_done_pick": False,
            "ever_subtask_done_place": False,
            "ever_subtask_done_press": False,
        },
        "violations": {
            "move_collision": False,
            "pinch_while_acting": False,
            "drop_or_eject": False,
            "fragile_contact_excessive": False,
            "stalled": False,
            "move_contact_excessive": False,
            "speed_force_unsafe": False,
            "dispense_alignment": False,
            "overflow_or_hot_exposed": False,
            "food_contact_before_dispense": False,
            "electrical_spill": False,
            "spill_without_clean": False,
            "move_without_subtask_done": False,
            "move_without_grasp_stable": False,
            "dispense_without_settle": False,
            "postdispense_move_unsettled": False,
            "retry_without_preconditions": False,
            "retry_budget_exceeded": False,
            "move_without_fixture_clear": False,
            "move_through_fixture_gap": False,
            "reuse_without_clean_sanitize": False,
            "move_with_grasp_but_no_place_phase": False,
        },
    }
    env._ltl_monitor_state = state
    return state


def _reset_monitor_state(env: Any) -> Dict[str, Any]:
    """Reset monitor history when a new episode starts."""
    if hasattr(env, "_ltl_monitor_state"):
        delattr(env, "_ltl_monitor_state")
    return _init_monitor_state(env)


def _infer_prepare_coffee_phase(
    predicates: Dict[str, bool],
    state: Dict[str, Any],
) -> str:
    """Infer a coarse task phase from the current predicate set."""
    flags = state.get("trace_flags", {}) or {}
    if predicates.get("spill", False) or predicates.get("fault", False):
        return "recover"
    if predicates.get("act_clean", False):
        return "clean"
    if predicates.get("coffee_machine_on", False):
        return "dispense"
    if flags.get("ever_coffee_machine_on", False) and predicates.get("mug_centered", False):
        return "postdispense"
    if predicates.get("act_press", False) and not predicates.get("coffee_machine_on", False):
        return "press"
    if predicates.get("act_place", False) or predicates.get("mug_centered", False):
        return "place"
    if predicates.get("mug_grasped", False):
        return "transport"
    if (
        predicates.get("act_open", False)
        or predicates.get("cabinet_open", False)
        or predicates.get("act_pick", False)
    ):
        return "pick"
    return state.get("phase", "approach")


def _build_prepare_coffee_predicates(
    env: Any,
    static_info: Dict[str, Any],
    dynamic_info: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build a structured predicate hierarchy for the coffee task."""

    def _linear_velocity(entry: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            vec = np.asarray((entry or {}).get("linear"), dtype=float).reshape(-1)
        except Exception:
            return None
        if vec.size < 3 or not np.all(np.isfinite(vec[:3])):
            return None
        return vec[:3]

    def _safe_contact(lhs: Any, rhs: Any) -> bool:
        try:
            return bool(env.check_contact(lhs, rhs))
        except Exception:
            return False

    def _fixture_distance(name: str, point: Optional[np.ndarray]) -> float:
        if point is None:
            return float(np.nan)
        fixture_entry = (
            ((static_info or {}).get("scene_layout", {}) or {}).get("fixtures", {}) or {}
        ).get(
            name,
            {},
        )
        try:
            pos = np.asarray(fixture_entry.get("position"), dtype=float).reshape(-1)
        except Exception:
            return float(np.nan)
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return float(np.nan)
        return float(np.linalg.norm(point[:3] - pos[:3]))

    def _cabinet_handle_position(cabinet: Any) -> Optional[np.ndarray]:
        if cabinet is None:
            return None
        handle_names = []
        for attr_name in ("handle_name", "left_handle_name", "right_handle_name"):
            try:
                candidate = getattr(cabinet, attr_name)
                handle_name = candidate() if callable(candidate) else candidate
            except Exception:
                continue
            if handle_name:
                handle_names.append(handle_name)
        for handle_name in handle_names:
            try:
                geom_id = env.sim.model.geom_name2id(handle_name)
                pos = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float).reshape(-1)
            except Exception:
                continue
            if pos.size >= 3 and np.all(np.isfinite(pos[:3])):
                return pos[:3]
        return None

    eef_speed = _get_linear_speed(dynamic_info, "end_effector_velocity")
    root_speed = _get_linear_speed(dynamic_info, "root_velocity")
    eef_force = _get_eef_force(dynamic_info)
    object_speed = _get_object_speed(dynamic_info, obj_name="obj")
    object_angular_speed = _get_object_angular_speed(dynamic_info, obj_name="obj")
    eef_pos = _get_eef_position(dynamic_info)
    object_pos = _get_object_position(env, "obj")
    eef_velocity_vec = _linear_velocity(
        ((dynamic_info or {}).get("robot", {}) or {}).get("end_effector_velocity", {})
    )
    scene_info = (dynamic_info or {}).get("scene", {}) or {}
    object_entries = scene_info.get("objects", {}) or {}
    object_velocity_entry = (object_entries.get("obj", {}) or {}).get("velocity", {}) or {}
    object_velocity_vec = _linear_velocity(object_velocity_entry)
    cabinet = _get_fixture(env, "cab")
    coffee_machine = _get_fixture(env, "coffee_machine")
    gripper = getattr(env.robots[0], "gripper", {}).get("right")
    cabinet_handle_pos = _cabinet_handle_position(cabinet)
    cabinet_open = _cabinet_open(env, "cab")
    door_velocity = _fixture_door_velocity(dynamic_info, "cab")
    fixture_moving = door_velocity > 0.02
    door_open_safe = _cabinet_open(env, "cab", threshold=0.7)
    mug_grasped = _object_grasped(env, "obj")
    mug_upright = _object_upright(env, "obj")
    mug_centered = _mug_centered(env, "obj")
    # this is essentially not released, released should depend on the stage
    # fix: keep the generic gripper-far proxy here, but use task-specific support / grasp cues below for set-down and drop detection.
    gripper_far_from_object = _gripper_far_from_object(env, "obj")
    coffee_machine_on = _coffee_machine_on(env)
    button_pressed = _start_button_pressed(env)
    button_names = _start_button_names(env)
    robot_collision = _robot_collision(env)
    robot_contact = _get_robot_contact_count(dynamic_info) > 0.0
    eef_near_machine = _eef_near_fixture(dynamic_info, static_info, "coffee_machine", 0.20)
    eef_near_cabinet = _eef_near_fixture(dynamic_info, static_info, "cab", 0.22)
    near_fixture_region = eef_near_machine or eef_near_cabinet
    collision_pick_place_raw = _detect_robot_collision_excluding(
        env,
        exclude_geom_names=["obj", "mug", "tray"],
    )
    press_excludes = list(button_names) + ["button", "start_button"]
    collision_press_raw = _detect_robot_collision_excluding(env, exclude_geom_names=press_excludes)
    obj_slow = bool(np.isfinite(object_speed) and object_speed < OBJ_SLOW_LIN_THRESHOLD)
    obj_angular_slow = bool(
        np.isfinite(object_angular_speed) and object_angular_speed < OBJ_SLOW_ANG_THRESHOLD
    )
    obj_velocity_synced = bool(
        np.isfinite(eef_speed)
        and np.isfinite(object_speed)
        and abs(float(eef_speed) - float(object_speed)) < SYNC_SPEED_THRESHOLD
    )
    object_contact_cabinet = _object_in_cabinet(env, "obj", "cab")
    object_contact_machine = False
    object_contact_counter = False
    if coffee_machine is not None:
        try:
            object_contact_machine = bool(OU.check_obj_fixture_contact(env, "obj", coffee_machine))
        except Exception:
            object_contact_machine = False
    try:
        object_contact_counter = bool(OU.check_obj_any_counter_contact(env, "obj"))
    except Exception:
        object_contact_counter = False
    gripper_contact_cabinet = (
        _safe_contact(gripper, cabinet) if gripper is not None and cabinet is not None else False
    )
    gripper_contact_machine = (
        _safe_contact(gripper, coffee_machine)
        if gripper is not None and coffee_machine is not None
        else False
    )
    gripper_contact_counter = False
    if gripper is not None:
        try:
            gripper_contact_counter = any(
                _safe_contact(gripper, fixture)
                for fixture in env.fixtures.values()
                if isinstance(fixture, Counter)
            )
        except Exception:
            gripper_contact_counter = False
    fixture_contact = bool(
        object_contact_cabinet
        or object_contact_machine
        or object_contact_counter
        or gripper_contact_cabinet
        or gripper_contact_machine
        or gripper_contact_counter
    )
    mug_velocity_synced = False
    if eef_velocity_vec is not None and object_velocity_vec is not None:
        mug_velocity_synced = bool(np.linalg.norm(object_velocity_vec - eef_velocity_vec) < 0.10)
    handle_clearance = float(np.nan)
    if cabinet_handle_pos is not None:
        clearance_candidates = []
        if eef_pos is not None:
            clearance_candidates.append(float(np.linalg.norm(eef_pos - cabinet_handle_pos)))
        if object_pos is not None:
            clearance_candidates.append(float(np.linalg.norm(object_pos - cabinet_handle_pos)))
        if clearance_candidates:
            handle_clearance = min(clearance_candidates)
    eef_near_handle = bool(np.isfinite(handle_clearance) and handle_clearance < 0.12)
    collision_distance = min(
        [
            dist
            for dist in (
                _fixture_distance("cab", eef_pos),
                _fixture_distance("cab", object_pos),
                _fixture_distance("coffee_machine", eef_pos),
                _fixture_distance("coffee_machine", object_pos),
            )
            if np.isfinite(dist)
        ],
        default=float(np.nan),
    )
    machine_distance = _fixture_distance("coffee_machine", eef_pos)
    machine_face_close = bool(np.isfinite(machine_distance) and machine_distance < 0.10)

    action = {
        "act_move": _entry(
            name="act_move",
            value=(
                (np.isfinite(eef_speed) and eef_speed >= MOVE_SPEED_THRESHOLD)
                or (np.isfinite(root_speed) and root_speed >= MOVE_SPEED_THRESHOLD)
            ),
            category="low_level_action",
            formula=None,
            language="The robot is actively moving.",
            short_readout="Robot moving",
        ),
        "act_pick": _entry(
            name="act_pick",
            value=bool(cabinet_open and (mug_grasped or eef_near_cabinet) and not mug_centered),
            category="low_level_action",
            formula=None,
            language="The robot is in the pick phase near the cabinet.",
            short_readout="Pick action",
        ),
        "act_place": _entry(
            name="act_place",
            value=bool(eef_near_machine and (mug_grasped or mug_centered)),
            category="low_level_action",
            formula=None,
            language="The robot is in the place phase near the coffee machine.",
            short_readout="Place action",
        ),
        "act_open": _entry(
            name="act_open",
            value=bool(eef_near_cabinet and cabinet_open and fixture_moving),
            category="low_level_action",
            formula=None,
            language="The robot is opening the cabinet.",
            short_readout="Open action",
        ),
        "act_close": _entry(
            name="act_close",
            value=False,
            category="low_level_action",
            formula=None,
            language="The robot is closing a fixture.",
            short_readout="Close action",
        ),
        "act_press": _entry(
            name="act_press",
            value=button_pressed,
            category="low_level_action",
            formula=None,
            language="The robot is pressing the coffee-machine start button.",
            short_readout="Press action",
        ),
        "act_recover": _entry(
            name="act_recover",
            value=bool(state.get("phase") == "recover"),
            category="low_level_action",
            formula=None,
            language="The controller is in a recovery phase.",
            short_readout="Recover action",
        ),
        "act_clean": _entry(
            name="act_clean",
            value=False,
            category="low_level_action",
            formula=None,
            language="A cleanup routine is currently running.",
            short_readout="Clean action",
        ),
    }

    object_state = {
        "button_pressed": _entry(
            name="button_pressed",
            value=button_pressed,
            category="object_state",
            formula=None,
            language="The coffee machine start button has been pressed.",
            short_readout="Button pressed",
        ),
        "cabinet_open": _entry(
            name="cabinet_open",
            value=cabinet_open,
            category="object_state",
            formula=None,
            language="The cabinet door is open.",
            short_readout="Cabinet open",
        ),
        "coffee_machine_on": _entry(
            name="coffee_machine_on",
            value=coffee_machine_on,
            category="object_state",
            formula=None,
            language="The coffee machine is turned on.",
            short_readout="Coffee machine on",
        ),
        "mug_grasped": _entry(
            name="mug_grasped",
            value=mug_grasped,
            category="object_state",
            formula=None,
            language="The robot is grasping the mug.",
            short_readout="Mug grasped",
        ),
        "obj_grasped": _entry(
            name="obj_grasped",
            value=mug_grasped,
            category="object_state",
            formula=None,
            language="The target object is grasped.",
            short_readout="Object grasped",
        ),
        "grasp_stable": _entry(
            name="grasp_stable",
            value=bool(
                mug_grasped
                and _recent_all_true(state, "mug_grasped", TEMPORAL_WINDOW)
                and mug_upright
                and mug_velocity_synced
            ),
            category="object_state",
            formula=None,
            language="The mug grasp has remained stable for a short temporal window.",
            short_readout="Grasp stable",
        ),
        "mug_upright": _entry(
            name="mug_upright",
            value=mug_upright,
            category="object_state",
            formula=None,
            language="The mug remains upright.",
            short_readout="Mug upright",
        ),
        "obj_upright": _entry(
            name="obj_upright",
            value=mug_upright,
            category="object_state",
            formula=None,
            language="The target object remains upright.",
            short_readout="Object upright",
        ),
        "obj_slow": _entry(
            name="obj_slow",
            value=obj_slow,
            category="object_state",
            formula=None,
            language="The target object is moving slowly.",
            short_readout="Object slow",
        ),
        "obj_angular_slow": _entry(
            name="obj_angular_slow",
            value=obj_angular_slow,
            category="object_state",
            formula=None,
            language="The target object has low angular speed.",
            short_readout="Object angular slow",
        ),
        "obj_velocity_synced": _entry(
            name="obj_velocity_synced",
            value=obj_velocity_synced,
            category="object_state",
            formula=None,
            language="The end-effector and object velocities are synchronized.",
            short_readout="Velocity synced",
        ),
        "mug_centered": _entry(
            name="mug_centered",
            value=mug_centered,
            category="object_state",
            formula=None,
            language="The mug is centered under the coffee spout target.",
            short_readout="Mug centered",
        ),
        "mug_set_down": _entry(
            name="mug_set_down",
            value=bool(
                (object_contact_machine or object_contact_counter or object_contact_cabinet)
                and not mug_grasped
                and object_speed < OBJ_SLOW_LIN_THRESHOLD
            ),
            category="object_state",
            formula=None,
            language="The mug has been set down on a support surface.",
            short_readout="Mug set down",
        ),
        "mug_release_event": _entry(
            name="mug_release_event",
            value=bool(
                state.get("phase") == "place"
                and _recent_any_true(state, "mug_grasped", TEMPORAL_WINDOW)
                and not mug_grasped
            ),
            category="object_state",
            formula=None,
            language="A mug release event occurred during the place phase.",
            short_readout="Mug release event",
        ),
        "valid_low_release": _entry(
            name="valid_low_release",
            value=bool(
                object_pos is not None
                and np.isfinite(object_pos[2])
                and float(object_pos[2]) <= PLACE_HEIGHT_THRESHOLD
            ),
            category="object_state",
            formula=None,
            language="The mug release happened close to the support surface.",
            short_readout="Valid low release",
        ),
        "obj_set_down": _entry(
            name="obj_set_down",
            value=bool(
                (object_contact_machine or object_contact_counter or object_contact_cabinet)
                and not mug_grasped
                and object_speed < OBJ_SLOW_LIN_THRESHOLD
            ),
            category="object_state",
            formula=None,
            language="The target object has been set down on its support surface.",
            short_readout="Object set down",
        ),
        "cup_settled": _entry(
            name="cup_settled",
            value=bool(
                _recent_all_true(state, "mug_upright", TEMPORAL_WINDOW)
                and _recent_all_true(state, "mug_slow", TEMPORAL_WINDOW)
                and _recent_all_true(state, "mug_angular_slow", TEMPORAL_WINDOW)
            ),
            category="object_state",
            formula=None,
            language="The cup has stayed still and upright for a recent time window.",
            short_readout="Cup settled",
        ),
        "obj_settled": _entry(
            name="obj_settled",
            value=bool(
                _recent_all_true(state, "mug_upright", TEMPORAL_WINDOW)
                and _recent_all_true(state, "mug_slow", TEMPORAL_WINDOW)
                and _recent_all_true(state, "mug_angular_slow", TEMPORAL_WINDOW)
            ),
            category="object_state",
            formula=None,
            language="The target object has stayed still and upright recently.",
            short_readout="Object settled",
        ),
        "contact_fixture": _entry(
            name="contact_fixture",
            value=fixture_contact,
            category="object_state",
            formula=None,
            language="The robot, gripper, or mug is contacting a nearby fixture.",
            short_readout="Contact fixture",
        ),
        "contact_excessive": _entry(
            name="contact_excessive",
            value=bool(np.isfinite(eef_force) and eef_force >= MUG_CONTACT_FORCE_THRESHOLD),
            category="object_state",
            formula=None,
            language="Contact force is high enough to be treated as excessive.",
            short_readout="Contact excessive",
        ),
        "contact_excessive_common": _entry(
            name="contact_excessive_common",
            value=bool(np.isfinite(eef_force) and eef_force >= CONTACT_FORCE_THRESHOLD),
            category="object_state",
            formula=None,
            language="Contact force is above the common safety threshold.",
            short_readout="Contact excessive (common)",
        ),
        # TODO: this does not include grasped object collide with other objects
        # fix: augment collision detection with a distance/contact proxy for the carried mug near cabinet or machine geometry.
        # TODO: This is not correct, here the collision also includes contact with the target?
        # should be phase dependent
        "collision_risk": _entry(
            name="collision_risk",
            value=bool(
                robot_collision
                or fixture_contact
                or (mug_grasped and np.isfinite(collision_distance) and collision_distance < 0.08)
            ),
            category="object_state",
            formula=None,
            language="A conservative collision-risk proxy based on actual robot collision.",
            short_readout="Collision risk",
        ),
        "collision_pick_place": _entry(
            name="collision_pick_place",
            value=bool(
                (action["act_pick"]["value"] or action["act_place"]["value"])
                and collision_pick_place_raw
            ),
            category="object_state",
            formula=None,
            language="Collision risk during pick/place actions.",
            short_readout="Pick/place collision",
        ),
        "collision_press": _entry(
            name="collision_press",
            value=bool(action["act_press"]["value"] and collision_press_raw),
            category="object_state",
            formula=None,
            language="Collision risk during press actions.",
            short_readout="Press collision",
        ),
        # TODO: change this to pinch_risk := ∃ rigid pair (a,b) such that d(gripper_or_mug, gap_region(a,b)) < ε_gap ∧ relative_closing_velocity(a,b, robot_part) > ε_close
        # fix: approximate pinch risk with cabinet-handle proximity plus door motion, with a machine-face fallback for forceful contacts in the narrow front workspace.
        "pinch": _entry(
            name="pinch",
            value=bool(
                (
                    eef_near_handle
                    and fixture_moving
                    and np.isfinite(handle_clearance)
                    and handle_clearance < 0.10
                )
                or (
                    machine_face_close
                    and fixture_contact
                    and np.isfinite(eef_force)
                    and eef_force >= EEF_FORCE_THRESHOLD
                )
            ),
            category="object_state",
            formula=None,
            language="A conservative pinch proxy near the cabinet or machine.",
            short_readout="Pinch hazard",
        ),
        # change not gripper_far_from_object to not mug_set_down
        # fix: defer the final value until after object_state is built so drop detection depends on !mug_set_down, matching the text.
        "mug_dropped": _entry(
            name="mug_dropped",
            value=False,
            category="object_state",
            formula=None,
            language="The mug was likely dropped after a prior grasp.",
            short_readout="Mug dropped",
        ),
        "obj_dropped": _entry(
            name="obj_dropped",
            value=False,
            category="object_state",
            formula=None,
            language="The target object was likely dropped after a prior grasp.",
            short_readout="Object dropped",
        ),
        "mug_ejected": _entry(
            name="mug_ejected",
            value=bool(
                _recent_any_true(state, "mug_grasped", TEMPORAL_WINDOW)
                and not mug_grasped
                and object_speed > 0.25
            ),
            category="object_state",
            formula=None,
            language="The mug was likely ejected after grasp loss.",
            short_readout="Mug ejected",
        ),
        # TODO this seems useless? weird
        # fix: keep it as a proxy for commanded-vs-realized blockage by requiring movement intent, near-zero mug progress, and fixture contact/collision.
        "blocked_motion": _entry(
            name="blocked_motion",
            value=bool(
                action["act_move"]["value"]
                and object_speed < 0.01
                and (fixture_contact or robot_collision)
            ),
            category="object_state",
            formula=None,
            language="The robot appears to be moving but making little progress because of contact or blockage.",
            short_readout="Blocked motion",
        ),
        "stalled": _entry(
            name="stalled",
            value=bool(_recent_all_true(state, "blocked_motion", STALL_WINDOW)),
            category="object_state",
            formula=None,
            language="Blocked motion has persisted long enough to count as stalled.",
            short_readout="Stalled",
        ),
        # We cannot get this from the simulator so set to false forever
        "spill": _entry(
            name="spill",
            value=False,
            category="object_state",
            formula=None,
            language="A conservative spill proxy based on dispensing with bad mug alignment or orientation.",
            short_readout="Spill detected",
        ),
        # overflow := fill > fill_safe_max
        # If no liquid state exists, infer with a dispense timer:
        # overflow := dispense_duration > T_fill_safe
        # the overflow you implemented here is essentially the hot_liquid_exposed, change ovwrflow to true forever and change hot_liquid_exposed to this implementation
        # fix: leave overflow disabled without fill / timer instrumentation, and move the off-target-dispense proxy into hot_liquid_exposed below.
        "overflow": _entry(
            name="overflow",
            value=False,
            category="object_state",
            formula=None,
            language="A conservative overflow proxy based on dispensing while the mug is not centered.",
            short_readout="Overflow",
        ),
        "hot_liquid_exposed": _entry(
            name="hot_liquid_exposed",
            value=False,
            category="object_state",
            formula=None,
            language="Hot liquid is exposed outside the cup.",
            short_readout="Hot liquid exposed",
        ),
        "food_contact_contaminated": _entry(
            name="food_contact_contaminated",
            value=False,
            category="object_state",
            formula=None,
            language="The mug's food-contact region is contaminated.",
            short_readout="Food-contact contaminated",
        ),
        "near_electrical": _entry(
            name="near_electrical",
            value=eef_near_machine,
            category="object_state",
            formula=None,
            language="A spill would be near the coffee machine's electrical hazard zone.",
            short_readout="Near electrical zone",
        ),
        # fixture_clear :=
        # door_q > q_open_safe
        # ∧ |door_qvel| < ε_door_vel
        # ∧ clearance_along_path(ee_or_mug, door_edge) > ε_clear
        # TODO: the current implementation is different from above
        # fix: require the door to be open past the safe threshold, nearly stationary, and not too close to the handle / door-edge proxy.
        # TODO: not sure about handle_clearance
        "fixture_clear": _entry(
            name="fixture_clear",
            value=bool(
                door_open_safe
                and abs(door_velocity) < 0.02
                and (not np.isfinite(handle_clearance) or handle_clearance > 0.08)
            ),
            category="object_state",
            formula=None,
            language="The cabinet door is open enough and stable enough to clear the robot path.",
            short_readout="Fixture clear",
        ),
        "fixture_moving": _entry(
            name="fixture_moving",
            value=fixture_moving,
            category="object_state",
            formula=None,
            language="A relevant fixture is moving.",
            short_readout="Fixture moving",
        ),
        "fault": _entry(
            name="fault",
            value=False,
            category="object_state",
            formula=None,
            language="A generalized fault flag is raised.",
            short_readout="Fault active",
        ),
        "safe_restart_ready": _entry(
            name="safe_restart_ready",
            value=False,
            category="object_state",
            formula=None,
            language="The system is ready for safe restart.",
            short_readout="Safe restart ready",
        ),
        "uncertainty_detected": _entry(
            name="uncertainty_detected",
            value=False,
            category="object_state",
            formula=None,
            language="The system detects uncertainty in its state estimates.",
            short_readout="Uncertainty detected",
        ),
        "mug_fragile": _entry(
            name="mug_fragile",
            value=True,
            category="object_state",
            formula=None,
            language="The mug is treated as a fragile object.",
            short_readout="Mug fragile",
        ),
        "speed_force_unsafe": _entry(
            name="speed_force_unsafe",
            value=bool(
                action["act_move"]["value"]
                and (np.isfinite(eef_force) and eef_force >= CONTACT_FORCE_THRESHOLD)
            ),
            category="object_state",
            formula=None,
            language="The robot is moving with unsafe speed/force conditions for the shared workspace.",
            short_readout="Speed-force unsafe",
        ),
        # TODO: did we define this phase/post condition?
        # fix: keep this derived marker tied to the inferred phase so downstream formulas continue to work without introducing new semantics.
        "phase_postdispense": _entry(
            name="phase_postdispense",
            value=bool(state.get("phase") == "postdispense"),
            category="object_state",
            formula=None,
            language="The system is in the post-dispense phase.",
            short_readout="Post-dispense phase",
        ),
        # TODO there's no definition in the txt and it seems to be same of one of the actions
        # fix: keep this placeholder false so the proposition remains available to formulas without inventing unsupported task semantics.
        "act_serve": _entry(
            name="act_serve",
            value=False,
            category="object_state",
            formula=None,
            language="The robot is serving the drink.",
            short_readout="Serve action",
        ),
        "act_move_through_fixture_gap": _entry(
            name="act_move_through_fixture_gap",
            value=bool(action["act_move"]["value"] and fixture_moving and near_fixture_region),
            category="object_state",
            formula=None,
            language="The robot is moving through a region that a moving fixture may close off.",
            short_readout="Move through fixture gap",
        ),
    }

    # Resolve the handful of predicates that depend on other object-state entries.
    object_state["mug_dropped"]["value"] = bool(
        _recent_any_true(state, "mug_grasped", TEMPORAL_WINDOW)
        and not mug_grasped
        and not object_state["mug_set_down"]["value"]
        and object_speed > 0.10
    )
    object_state["mug_dropped"][
        "readout"
    ] = f"Mug dropped: {str(object_state['mug_dropped']['value']).lower()}"
    object_state["obj_dropped"]["value"] = bool(object_state["mug_dropped"]["value"])
    object_state["obj_dropped"][
        "readout"
    ] = f"Object dropped: {str(object_state['obj_dropped']['value']).lower()}"
    object_state["spill"]["value"] = bool(
        object_state["coffee_machine_on"]["value"]
        and (not object_state["mug_centered"]["value"] or not object_state["mug_upright"]["value"])
    )
    object_state["spill"][
        "readout"
    ] = f"Spill detected: {str(object_state['spill']['value']).lower()}"
    object_state["overflow"]["value"] = False
    object_state["overflow"][
        "readout"
    ] = f"Overflow: {str(object_state['overflow']['value']).lower()}"
    # TODO
    object_state["hot_liquid_exposed"]["value"] = bool(
        (
            object_state["spill"]["value"]
            or object_state["overflow"]["value"]
            or (
                object_state["coffee_machine_on"]["value"]
                and not object_state["mug_centered"]["value"]
            )
            or (object_state["mug_dropped"]["value"] and coffee_machine_on)
        )
        and coffee_machine_on
    )
    object_state["hot_liquid_exposed"][
        "readout"
    ] = f"Hot liquid exposed: {str(object_state['hot_liquid_exposed']['value']).lower()}"
    object_state["fault"]["value"] = bool(
        object_state["blocked_motion"]["value"]
        or object_state["mug_dropped"]["value"]
        or object_state["collision_risk"]["value"]
        or object_state["spill"]["value"]
    )
    object_state["fault"][
        "readout"
    ] = f"Fault active: {str(object_state['fault']['value']).lower()}"
    object_state["safe_restart_ready"]["value"] = bool(
        not object_state["blocked_motion"]["value"]
        and not object_state["spill"]["value"]
        and not object_state["collision_risk"]["value"]
        and not object_state["pinch"]["value"]
        and (not object_state["food_contact_contaminated"]["value"])
        and (
            not object_state["coffee_machine_on"]["value"]
            or (
                object_state["mug_centered"]["value"]
                and object_state["mug_upright"]["value"]
                and object_state["cup_settled"]["value"]
            )
        )
    )
    object_state["safe_restart_ready"][
        "readout"
    ] = f"Safe restart ready: {str(object_state['safe_restart_ready']['value']).lower()}"

    """Sequencing predicates"""
    sequencing = {
        "subtask_done_pick": _entry(
            name="subtask_done_pick",
            value=bool(
                object_state["mug_grasped"]["value"] and object_state["grasp_stable"]["value"]
            ),
            category="sequence_predicate",
            formula=None,
            language="The pick subtask is complete when the mug is grasped and the grasp is stable.",
            short_readout="Pick subtask done",
        ),
        "subtask_done_place": _entry(
            name="subtask_done_place",
            value=bool(
                object_state["mug_set_down"]["value"]
                and object_state["mug_upright"]["value"]
                and not object_state["mug_grasped"]["value"]
            ),
            category="sequence_predicate",
            formula=None,
            language="The place subtask is complete when the mug is set down upright and released.",
            short_readout="Place subtask done",
        ),
        "subtask_done_press": _entry(
            name="subtask_done_press",
            value=bool(coffee_machine_on and not action["act_press"]["value"]),
            category="sequence_predicate",
            formula=None,
            language="The press subtask is complete once the machine is on and the press has ended.",
            short_readout="Press subtask done",
        ),
        "subtask_done_dispense": _entry(
            name="subtask_done_dispense",
            value=bool(coffee_machine_on and not action["act_press"]["value"]),
            category="sequence_predicate",
            formula=None,
            language="The dispense subtask is complete after the dispense phase has finished.",
            short_readout="Dispense subtask done",
        ),
        "preconditions_met_pick": _entry(
            name="preconditions_met_pick",
            value=bool(
                eef_near_cabinet
                and object_state["mug_upright"]["value"]
                and not object_state["collision_risk"]["value"]
            ),
            category="sequence_predicate",
            formula=None,
            language="Pick preconditions are satisfied.",
            short_readout="Pick preconditions met",
        ),
        "preconditions_met_dispense": _entry(
            name="preconditions_met_dispense",
            value=bool(
                object_state["mug_centered"]["value"]
                and object_state["mug_upright"]["value"]
                and object_state["cup_settled"]["value"]
            ),
            category="sequence_predicate",
            formula=None,
            language="Dispense preconditions are satisfied.",
            short_readout="Dispense preconditions met",
        ),
        # TODO: we do not need posconditions, just need preconditions to determine the phase of the task
        # fix: keep this compatibility predicate for downstream logic, but tie it to the realized coffee-task end state only.
        "postconditions_met": _entry(
            name="postconditions_met",
            value=bool(coffee_machine_on and object_state["mug_centered"]["value"]),
            category="sequence_predicate",
            formula=None,
            language="Expected postconditions for the coffee task currently hold.",
            short_readout="Postconditions met",
        ),
        "retry": _entry(
            name="retry",
            value=bool(
                state.get("phase") == state.get("last_phase")
                and state.get("fault_active_previous_step", False)
                and object_state["fault"]["value"]
            ),
            category="sequence_predicate",
            formula=None,
            language="The controller has re-entered the same phase after a fault.",
            short_readout="Retry active",
        ),
        "retry_budget_exceeded": _entry(
            name="retry_budget_exceeded",
            value=bool(state.get("retry_count", 0) > RETRY_BUDGET),
            category="sequence_predicate",
            formula=None,
            language="The retry budget has been exceeded.",
            short_readout="Retry budget exceeded",
        ),
        # TODO
        "cleaned": _entry(
            name="cleaned",
            value=False,
            category="sequence_predicate",
            formula=None,
            language="A cleaning routine has completed.",
            short_readout="Cleaned",
        ),
        # TODO
        "sanitized": _entry(
            name="sanitized",
            value=False,
            category="sequence_predicate",
            formula=None,
            language="A sanitization routine has completed.",
            short_readout="Sanitized",
        ),
        "subtask_done": _entry(
            name="subtask_done",
            value=False,
            category="sequence_predicate",
            formula=None,
            language="The current subtask is complete.",
            short_readout="Current subtask done",
        ),
        "preconditions_met": _entry(
            name="preconditions_met",
            value=False,
            category="sequence_predicate",
            formula=None,
            language="The current phase preconditions are satisfied.",
            short_readout="Current preconditions met",
        ),
        "phase_pick": _entry(
            name="phase_pick",
            value=bool(action["act_pick"]["value"] and not object_state["mug_grasped"]["value"]),
            category="sequence_predicate",
            formula=None,
            language="The system is in the pick phase.",
            short_readout="Pick phase",
        ),
        "phase_place": _entry(
            name="phase_place",
            value=bool(action["act_place"]["value"] and object_state["mug_grasped"]["value"]),
            category="sequence_predicate",
            formula=None,
            language="The system is in the place phase.",
            short_readout="Place phase",
        ),
        "phase_press": _entry(
            name="phase_press",
            value=bool(action["act_press"]["value"] and not coffee_machine_on),
            category="sequence_predicate",
            formula=None,
            language="The system is in the press phase.",
            short_readout="Press phase",
        ),
    }

    # TODO: need to rewrite _infer_prepare_coffee_phase
    phase = _infer_prepare_coffee_phase(
        {
            **{k: v["value"] for k, v in action.items()},
            **{k: v["value"] for k, v in object_state.items()},
        },
        state,
    )
    state["last_phase"] = state.get("phase", "approach")
    state["phase"] = phase

    # TODO: there's only these tasks, pick mug, place mug and press start button
    sequencing["subtask_done"]["value"] = {
        "pick": sequencing["subtask_done_pick"]["value"],
        "transport": sequencing["subtask_done_pick"]["value"],
        "place": sequencing["subtask_done_place"]["value"],
        "press": sequencing["subtask_done_press"]["value"],
        "dispense": sequencing["subtask_done_press"]["value"],
        "postdispense": sequencing["subtask_done_dispense"]["value"],
        "recover": sequencing["postconditions_met"]["value"],
        "clean": sequencing["cleaned"]["value"],
    }.get(phase, None)
    sequencing["subtask_done"][
        "readout"
    ] = f"Current subtask done: {str(sequencing['subtask_done']['value']).lower()}"

    # TODO: sync with the phase above
    sequencing["preconditions_met"]["value"] = {
        "pick": sequencing["preconditions_met_pick"]["value"],
        "transport": sequencing["preconditions_met_pick"]["value"],
        "place": sequencing["preconditions_met_dispense"]["value"],
        "press": sequencing["preconditions_met_dispense"]["value"],
        "dispense": sequencing["preconditions_met_dispense"]["value"],
        "recover": object_state["safe_restart_ready"]["value"],
        "clean": True,
    }.get(phase, None)
    sequencing["preconditions_met"][
        "readout"
    ] = f"Current preconditions met: {str(sequencing['preconditions_met']['value']).lower()}"

    if phase != state["last_phase"]:
        state["phase_enter_counts"][phase] = state["phase_enter_counts"].get(phase, 0) + 1
        if state["phase_enter_counts"][phase] > 1:
            state["retry_count"] += 1
    sequencing["retry"]["value"] = bool(
        phase == state["last_phase"]
        and state.get("fault_active_previous_step", False)
        and object_state["fault"]["value"]
    )
    sequencing["retry"]["readout"] = f"Retry active: {str(sequencing['retry']['value']).lower()}"
    sequencing["retry_budget_exceeded"]["value"] = bool(state.get("retry_count", 0) > RETRY_BUDGET)
    sequencing["retry_budget_exceeded"][
        "readout"
    ] = f"Retry budget exceeded: {str(sequencing['retry_budget_exceeded']['value']).lower()}"

    composite = {
        "primitive_actions": action,
        "object_state": object_state,
        "sequencing": sequencing,
    }
    return composite


def _update_prepare_coffee_state(
    state: Dict[str, Any],
    sections: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Update trace memory, histories, and sticky violation flags for the coffee task."""
    predicates = {}
    for entries in sections.values():
        predicates.update({name: entry["value"] for name, entry in entries.items()})

    _record_history(state, "mug_grasped", predicates["mug_grasped"])
    _record_history(state, "mug_upright", predicates["mug_upright"])
    _record_history(state, "mug_slow", _safe_bool(predicates["mug_set_down"]))
    _record_history(state, "mug_angular_slow", _safe_bool(predicates["mug_upright"]))
    _record_history(state, "blocked_motion", predicates["blocked_motion"])

    if predicates["act_clean"]:
        state["trace_flags"]["ever_act_clean"] = True
    if predicates["act_pick"]:
        state["trace_flags"]["ever_act_pick"] = True
    if predicates["act_place"]:
        state["trace_flags"]["ever_act_place"] = True
    if predicates["act_press"]:
        state["trace_flags"]["ever_act_press"] = True
    if predicates["mug_grasped"]:
        state["trace_flags"]["ever_mug_grasped"] = True
    if predicates["mug_centered"]:
        state["trace_flags"]["ever_mug_centered"] = True
    if predicates["mug_set_down"]:
        state["trace_flags"]["ever_mug_set_down"] = True
    if predicates["coffee_machine_on"]:
        state["trace_flags"]["ever_coffee_machine_on"] = True
    if predicates["subtask_done_pick"]:
        state["trace_flags"]["ever_subtask_done_pick"] = True
    if predicates["subtask_done_place"]:
        state["trace_flags"]["ever_subtask_done_place"] = True
    if predicates["subtask_done_press"]:
        state["trace_flags"]["ever_subtask_done_press"] = True

    violations = state["violations"]
    violations["move_collision"] |= predicates["act_move"] and predicates["collision_risk"]
    violations["pinch_while_acting"] |= (
        predicates["act_open"] or predicates["act_place"] or predicates["act_move"]
    ) and predicates["pinch"]
    violations["move_with_grasp_but_no_place_phase"] |= (
        predicates["act_move"] and predicates["mug_grasped"] and not predicates["act_place"]
    )
    violations["drop_or_eject"] |= predicates["mug_dropped"] or predicates["mug_ejected"]
    violations["fragile_contact_excessive"] |= (
        predicates["mug_fragile"]
        and predicates["contact_fixture"]
        and predicates["contact_excessive"]
    )
    violations["stalled"] |= predicates["stalled"]
    violations["move_contact_excessive"] |= (
        predicates["act_move"] and predicates["contact_excessive"]
    )
    violations["speed_force_unsafe"] |= predicates["act_move"] and predicates["speed_force_unsafe"]
    violations["dispense_alignment"] |= predicates["coffee_machine_on"] and not (
        predicates["mug_centered"] and predicates["mug_upright"] and predicates["cup_settled"]
    )
    violations["overflow_or_hot_exposed"] |= (
        predicates["overflow"] or predicates["hot_liquid_exposed"]
    )
    violations["food_contact_before_dispense"] |= predicates["food_contact_contaminated"] and (
        predicates["coffee_machine_on"] or predicates["act_serve"]
    )
    violations["electrical_spill"] |= (
        predicates["spill"] or predicates["overflow"]
    ) and predicates["near_electrical"]
    violations["spill_without_clean"] |= predicates["spill"] and not predicates["act_clean"]
    violations["move_without_subtask_done"] |= (
        predicates["act_move"] and not predicates["subtask_done"]
    )
    violations["move_without_grasp_stable"] |= (
        predicates["act_move"] and predicates["mug_grasped"] and not predicates["grasp_stable"]
    )
    violations["dispense_without_settle"] |= predicates["coffee_machine_on"] and not (
        predicates["mug_centered"] and predicates["mug_upright"] and predicates["cup_settled"]
    )
    violations["postdispense_move_unsettled"] |= (
        predicates["phase_postdispense"]
        and predicates["act_move"]
        and not predicates["cup_settled"]
    )
    violations["retry_without_preconditions"] |= (
        predicates["retry"] and not predicates["preconditions_met"]
    )
    violations["retry_budget_exceeded"] |= predicates["retry_budget_exceeded"]
    violations["move_without_fixture_clear"] |= (
        predicates["act_move"] and predicates["act_open"] and not predicates["fixture_clear"]
    )
    violations["move_through_fixture_gap"] |= (
        predicates["fixture_moving"] and predicates["act_move_through_fixture_gap"]
    )
    violations["reuse_without_clean_sanitize"] |= predicates["food_contact_contaminated"] and not (
        predicates["cleaned"] and predicates["sanitized"]
    )

    state["fault_active_previous_step"] = predicates["fault"]


def _formula_entry(
    name: str, formula: str, value: bool, language: str, readout: str, category: str
) -> Dict[str, Any]:
    """Create a formula monitor entry with an explicit LTL string."""
    return _entry(
        name=name,
        value=value,
        category=category,
        formula=formula,
        language=language,
        short_readout=readout,
    )


def _build_prepare_coffee_formulas(
    sections: Dict[str, Dict[str, Dict[str, Any]]],
    state: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build grouped LTL formulas from the design document."""
    p = {}
    for entries in sections.values():
        p.update({name: entry["value"] for name, entry in entries.items()})
    flags = state["trace_flags"]
    violations = state["violations"]

    return {
        "low_level_safety": {
            "G_act_move_implies_not_collision_risk": _formula_entry(
                "G_act_move_implies_not_collision_risk",
                "G(act_move -> !collision_risk)",
                not violations["move_collision"],
                "Whenever the robot moves, collision risk should remain false.",
                "Moving without collision risk",
                "safety_formula",
            ),
            "G_act_open_place_move_implies_not_pinch": _formula_entry(
                "G_act_open_place_move_implies_not_pinch",
                "G((act_open | act_place | act_move) -> !pinch)",
                not violations["pinch_while_acting"],
                "Open, place, and move actions should avoid pinch hazards.",
                "Acting without pinch hazard",
                "safety_formula",
            ),
            "G_move_and_grasped_until_place": _formula_entry(
                "G_move_and_grasped_until_place",
                "G((act_move & mug_grasped) -> (mug_grasped U act_place))",
                not violations["move_with_grasp_but_no_place_phase"],
                "While transporting the mug, the grasp should persist until the place phase.",
                "Carry mug until place",
                "safety_formula",
            ),
            "G_not_dropped_or_ejected": _formula_entry(
                "G_not_dropped_or_ejected",
                "G !(mug_dropped | mug_ejected)",
                not violations["drop_or_eject"],
                "The mug should never be dropped or ejected.",
                "No mug drop or ejection",
                "safety_formula",
            ),
            "G_fragile_contact_implies_not_excessive": _formula_entry(
                "G_fragile_contact_implies_not_excessive",
                "G(mug_fragile & contact_fixture -> !contact_excessive)",
                not violations["fragile_contact_excessive"],
                "Fragile mug contact with fixtures should not become excessive.",
                "Fragile contact stays gentle",
                "safety_formula",
            ),
            "G_blocked_motion_eventually_resolves": _formula_entry(
                "G_blocked_motion_eventually_resolves",
                "G(blocked_motion -> F !blocked_motion)",
                not violations["stalled"],
                "Blocked motion should eventually resolve rather than persist indefinitely.",
                "Blocked motion resolves",
                "safety_formula",
            ),
            "G_not_stalled": _formula_entry(
                "G_not_stalled",
                "G !stalled",
                not violations["stalled"],
                "The robot should never remain stalled.",
                "Never stalled",
                "safety_formula",
            ),
            "G_act_move_implies_not_contact_excessive": _formula_entry(
                "G_act_move_implies_not_contact_excessive",
                "G(act_move -> !contact_excessive)",
                not violations["move_contact_excessive"],
                "Moving should not produce excessive contact.",
                "Move without excessive contact",
                "safety_formula",
            ),
            "G_act_move_implies_not_speed_force_unsafe": _formula_entry(
                "G_act_move_implies_not_speed_force_unsafe",
                "G(act_move -> !speed_force_unsafe)",
                not violations["speed_force_unsafe"],
                "Moving should remain within safe speed-force conditions.",
                "Move with safe speed-force",
                "safety_formula",
            ),
        },
        "contextual_skill_constraints": {
            "G_coffee_machine_on_implies_centered_and_upright": _formula_entry(
                "G_coffee_machine_on_implies_centered_and_upright",
                "G(coffee_machine_on -> (mug_centered & mug_upright))",
                not violations["dispense_alignment"],
                "Dispensing should only happen when the mug is centered and upright.",
                "Dispense only when mug aligned",
                "context_formula",
            ),
            "G_not_overflow_or_hot_exposed": _formula_entry(
                "G_not_overflow_or_hot_exposed",
                "G !(overflow | hot_liquid_exposed)",
                not violations["overflow_or_hot_exposed"],
                "Overflow and hot-liquid exposure should never occur.",
                "No overflow or hot-liquid exposure",
                "context_formula",
            ),
            "G_food_contaminated_implies_not_dispense_or_serve": _formula_entry(
                "G_food_contaminated_implies_not_dispense_or_serve",
                "G(food_contact_contaminated -> !(coffee_machine_on | act_serve))",
                not violations["food_contact_before_dispense"],
                "Dispense and serve should be blocked if food-contact contamination exists.",
                "No dispense/serve while contaminated",
                "context_formula",
            ),
            "G_not_spill_near_electrical": _formula_entry(
                "G_not_spill_near_electrical",
                "G !((spill | overflow) & near_electrical)",
                not violations["electrical_spill"],
                "Spills and overflow should not occur in the electrical hazard zone.",
                "No electrical spill hazard",
                "context_formula",
            ),
            "G_not_spill_without_clean": _formula_entry(
                "G_not_spill_without_clean",
                "G !(spill & !act_clean)",
                not violations["spill_without_clean"],
                "A spill should not remain without immediate cleanup action.",
                "No uncleaned spill",
                "context_formula",
            ),
            "G_spill_implies_eventually_clean": _formula_entry(
                "G_spill_implies_eventually_clean",
                "G(spill -> F act_clean)",
                (not p["spill"]) or flags["ever_act_clean"],
                "If a spill happens, cleanup should eventually occur.",
                "Spill eventually cleaned",
                "context_formula",
            ),
        },
        "sequence_level_safety": {
            "G_next_move_implies_subtask_done": _formula_entry(
                "G_next_move_implies_subtask_done",
                "G(X act_move -> subtask_done)",
                not violations["move_without_subtask_done"],
                "The next motion should only happen after the current subtask is done.",
                "Move only after subtask completion",
                "sequence_formula",
            ),
            "G_move_and_grasped_implies_grasp_stable": _formula_entry(
                "G_move_and_grasped_implies_grasp_stable",
                "G(act_move & mug_grasped -> grasp_stable)",
                not violations["move_without_grasp_stable"],
                "Transporting a grasped mug requires a stable grasp.",
                "Move with stable grasp",
                "sequence_formula",
            ),
            "G_dispense_implies_centered_upright_settled": _formula_entry(
                "G_dispense_implies_centered_upright_settled",
                "G(coffee_machine_on -> (mug_centered & mug_upright & cup_settled))",
                not violations["dispense_without_settle"],
                "Dispense requires a centered, upright, and settled cup.",
                "Dispense only when cup settled",
                "sequence_formula",
            ),
            "G_postdispense_move_implies_settled": _formula_entry(
                "G_postdispense_move_implies_settled",
                "G((phase_postdispense & act_move) -> cup_settled)",
                not violations["postdispense_move_unsettled"],
                "After dispensing, movement should only occur once the cup has settled.",
                "Post-dispense move only when settled",
                "sequence_formula",
            ),
            "G_move_implies_not_postdispense_or_settled": _formula_entry(
                "G_move_implies_not_postdispense_or_settled",
                "G(act_move -> (!phase_postdispense | cup_settled))",
                not violations["postdispense_move_unsettled"],
                "Any movement in the post-dispense phase requires a settled cup.",
                "Move respects post-dispense settling",
                "sequence_formula",
            ),
            "G_retry_implies_preconditions_met": _formula_entry(
                "G_retry_implies_preconditions_met",
                "G(retry -> preconditions_met)",
                not violations["retry_without_preconditions"],
                "Retries should only occur when phase preconditions are satisfied.",
                "Retry only with preconditions",
                "sequence_formula",
            ),
            "G_fault_then_next_retry_has_preconditions": _formula_entry(
                "G_fault_then_next_retry_has_preconditions",
                "G((fault & X retry) -> X preconditions_met)",
                not violations["retry_without_preconditions"],
                "Fault-triggered retries should be gated by preconditions.",
                "Fault retry gated by preconditions",
                "sequence_formula",
            ),
            "G_not_retry_budget_exceeded": _formula_entry(
                "G_not_retry_budget_exceeded",
                "G !retry_budget_exceeded",
                not violations["retry_budget_exceeded"],
                "Retry counts should remain bounded.",
                "Retry budget respected",
                "sequence_formula",
            ),
            "G_move_near_fixture_implies_fixture_clear": _formula_entry(
                "G_move_near_fixture_implies_fixture_clear",
                "G((act_move & near_fixture_region) -> fixture_clear)",
                not violations["move_without_fixture_clear"],
                "Movement near fixture regions requires clear fixture geometry.",
                "Move near fixture only when clear",
                "sequence_formula",
            ),
            "G_fixture_moving_implies_not_move_through_gap": _formula_entry(
                "G_fixture_moving_implies_not_move_through_gap",
                "G(fixture_moving -> !act_move_through_fixture_gap)",
                not violations["move_through_fixture_gap"],
                "The robot should not move through a fixture gap while the fixture is moving.",
                "No motion through moving fixture gap",
                "sequence_formula",
            ),
        },
        "contextual_sequence_constraints": {
            "G_food_contaminated_until_cleaned_sanitized": _formula_entry(
                "G_food_contaminated_until_cleaned_sanitized",
                "G(food_contact_contaminated -> (!coffee_machine_on U (cleaned & sanitized)))",
                not violations["reuse_without_clean_sanitize"],
                "Contaminated food-contact surfaces must not be reused for dispense until cleaned and sanitized.",
                "Contamination blocks reuse until clean+sanitize",
                "context_formula",
            ),
        },
    }


def _flatten_entries(
    section_map: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Flatten structured sections into a single name-to-entry map."""
    flat: Dict[str, Dict[str, Any]] = {}
    for section_name, entries in section_map.items():
        for name, entry in entries.items():
            copied = dict(entry)
            copied["section"] = section_name
            flat[name] = copied
    return flat


def _compute_changes(
    previous_values: Dict[str, bool],
    flat_entries: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute monitor changes between consecutive snapshots."""
    changes: List[Dict[str, Any]] = []
    for name in sorted(flat_entries.keys()):
        entry = flat_entries[name]
        new_value = _safe_bool(entry.get("value"))
        old_value = previous_values.get(name)
        if old_value is None or old_value == new_value:
            continue
        changes.append(
            {
                "name": name,
                "section": entry.get("section"),
                "category": entry.get("category"),
                "formula": entry.get("formula"),
                "old_value": old_value,
                "new_value": new_value,
                "language": entry.get("language"),
                "readout": entry.get("readout"),
            }
        )
    return changes


def build_ltl_static_spec(env: Any, static_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build task-level static metadata for the structured LTL monitor."""
    task_name = _get_task_name(env)
    task_language = _get_task_language(static_info)
    if task_name != "PrepareCoffee":
        return {
            "task_name": task_name,
            "task_language": task_language,
            "design_notes": [
                "Only the generic collision formula is currently implemented for non-coffee tasks.",
            ],
            "predicate_groups": {
                "low_level_action": ["act_move"],
                "object_state": ["collision_risk"],
            },
            "ltl_formulas": ["G(act_move -> !collision_risk)"],
        }

    return {
        "task_name": task_name,
        "task_language": task_language,
        "design_source": "ltl_design/Coffee Preparation.txt",
        "design_notes": [
            "The monitor hierarchy matches the extracted text file: low-level action, object/state, sequencing, and grouped formulas.",
            "collision_risk is implemented conservatively from actual robot collision because predictive swept-volume risk is not currently modeled.",
            "pinch, spill, overflow, and near_electrical use privileged-state proxies rather than full physical liquid or hand-region simulation.",
            "food_contact_contaminated, act_clean, cleaned, sanitized, act_serve, and uncertainty_detected are currently placeholders unless the controller or environment exposes those signals.",
        ],
        "predicate_groups": {
            "low_level_action": [
                "act_move",
                "act_pick",
                "act_place",
                "act_open",
                "act_close",
                "act_press",
                "act_recover",
                "act_clean",
            ],
            "object_state": [
                "cabinet_open",
                "coffee_machine_on",
                "mug_grasped",
                "grasp_stable",
                "mug_upright",
                "mug_centered",
                "mug_set_down",
                "cup_settled",
                "contact_fixture",
                "contact_excessive",
                "collision_risk",
                "pinch",
                "mug_dropped",
                "mug_ejected",
                "blocked_motion",
                "stalled",
                "spill",
                "overflow",
                "hot_liquid_exposed",
                "food_contact_contaminated",
                "near_electrical",
                "fixture_clear",
                "fixture_moving",
                "fault",
                "safe_restart_ready",
                "uncertainty_detected",
                "mug_fragile",
                "speed_force_unsafe",
                "phase_postdispense",
                "act_serve",
                "act_move_through_fixture_gap",
            ],
            "sequencing": [
                "subtask_done_pick",
                "subtask_done_place",
                "subtask_done_dispense",
                "preconditions_met_pick",
                "preconditions_met_dispense",
                "postconditions_met",
                "retry",
                "retry_budget_exceeded",
                "cleaned",
                "sanitized",
                "subtask_done",
                "preconditions_met",
            ],
        },
        "ltl_formula_groups": {
            "low_level_safety": [
                "G(act_move -> !collision_risk)",
                "G((act_open | act_place | act_move) -> !pinch)",
                "G((act_move & mug_grasped) -> (mug_grasped U act_place))",
                "G !(mug_dropped | mug_ejected)",
                "G(mug_fragile & contact_fixture -> !contact_excessive)",
                "G(blocked_motion -> F !blocked_motion)",
                "G !stalled",
                "G(act_move -> !contact_excessive)",
                "G(act_move -> !speed_force_unsafe)",
            ],
            "contextual_skill_constraints": [
                "G(coffee_machine_on -> (mug_centered & mug_upright))",
                "G !(overflow | hot_liquid_exposed)",
                "G(food_contact_contaminated -> !(coffee_machine_on | act_serve))",
                "G !((spill | overflow) & near_electrical)",
                "G !(spill & !act_clean)",
                "G(spill -> F act_clean)",
            ],
            "sequence_level_safety": [
                "G(X act_move -> subtask_done)",
                "G(act_move & mug_grasped -> grasp_stable)",
                "G(coffee_machine_on -> (mug_centered & mug_upright & cup_settled))",
                "G((phase_postdispense & act_move) -> cup_settled)",
                "G(act_move -> (!phase_postdispense | cup_settled))",
                "G(retry -> preconditions_met)",
                "G((fault & X retry) -> X preconditions_met)",
                "G !retry_budget_exceeded",
                "G((act_move & near_fixture_region) -> fixture_clear)",
                "G(fixture_moving -> !act_move_through_fixture_gap)",
            ],
            "contextual_sequence_constraints": [
                "G(food_contact_contaminated -> (!coffee_machine_on U (cleaned & sanitized)))",
            ],
        },
    }


def _build_generic_monitor(dynamic_info: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal generic monitor for non-coffee tasks."""
    act_move = bool(
        (_get_linear_speed(dynamic_info, "end_effector_velocity") >= MOVE_SPEED_THRESHOLD)
        or (_get_linear_speed(dynamic_info, "root_velocity") >= MOVE_SPEED_THRESHOLD)
    )
    collision_risk = bool(
        _safe_float(((dynamic_info or {}).get("physics", {}) or {}).get("robot_contact_count"), 0.0)
        > 0.0
    )
    if act_move and collision_risk:
        state["violations"]["move_collision"] = True

    sections = {
        "low_level_action": {
            "act_move": _entry(
                name="act_move",
                value=act_move,
                category="low_level_action",
                formula=None,
                language="The robot is actively moving.",
                short_readout="Robot moving",
            )
        },
        "object_state": {
            "collision_risk": _entry(
                name="collision_risk",
                value=collision_risk,
                category="object_state",
                formula=None,
                language="A conservative collision-risk proxy based on current contacts.",
                short_readout="Collision risk",
            )
        },
    }
    formulas = {
        "low_level_safety": {
            "G_act_move_implies_not_collision_risk": _formula_entry(
                "G_act_move_implies_not_collision_risk",
                "G(act_move -> !collision_risk)",
                not state["violations"]["move_collision"],
                "Whenever the robot moves, collision risk should remain false.",
                "Moving without collision risk",
                "safety_formula",
            )
        }
    }
    flat = _flatten_entries({**sections, **formulas})
    changes = _compute_changes(state["prev_values"], flat)
    state["prev_values"] = {name: _safe_bool(entry["value"]) for name, entry in flat.items()}
    return {
        "task_name": "generic",
        "supported_task": False,
        "sections": sections,
        "formulas": formulas,
        "changes": changes,
        "summary_lines": [entry["readout"] for entry in formulas["low_level_safety"].values()],
    }


def build_ltl_monitor_snapshot(
    env: Any,
    static_info: Dict[str, Any],
    dynamic_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full monitor snapshot for the current simulation step."""
    task_name = _get_task_name(env)
    state = _init_monitor_state(env)
    current_timestep = int(((dynamic_info or {}).get("task", {}) or {}).get("timestep", -1))
    previous_timestep = state.get("last_timestep")
    if previous_timestep is not None and current_timestep < int(previous_timestep):
        state = _reset_monitor_state(env)
    state["last_timestep"] = current_timestep

    if task_name != "PrepareCoffee":
        return _build_generic_monitor(dynamic_info, state)

    sections = _build_prepare_coffee_predicates(env, static_info, dynamic_info, state)
    _update_prepare_coffee_state(state, sections)
    formulas = _build_prepare_coffee_formulas(sections, state)

    flat_entries = _flatten_entries({**sections, **formulas})
    changes = _compute_changes(state["prev_values"], flat_entries)
    state["prev_values"] = {
        name: _safe_bool(entry["value"]) for name, entry in flat_entries.items()
    }

    summary_lines: List[str] = []
    for formula_group_name in (
        "low_level_safety",
        "contextual_skill_constraints",
        "sequence_level_safety",
        "contextual_sequence_constraints",
    ):
        summary_lines.extend([entry["readout"] for entry in formulas[formula_group_name].values()])

    return {
        "task_name": task_name,
        "task_language": _get_task_language(static_info),
        "supported_task": True,
        "sections": sections,
        "formulas": formulas,
        "changes": changes,
        "summary_lines": summary_lines,
        "trace_state": {
            "phase": state.get("phase"),
            "retry_count": int(state.get("retry_count", 0)),
            "phase_enter_counts": dict(state.get("phase_enter_counts", {})),
            "trace_flags": dict(state.get("trace_flags", {})),
            "violations": dict(state.get("violations", {})),
        },
    }
