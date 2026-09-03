from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path = [path for path in sys.path if os.path.abspath(path or os.getcwd()) != THIS_DIR]
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


from monitor.monitor import RoboCasaSymbolicMonitor
from monitor.repeated_violation_monitor import (
    build_repeated_fixture_placement_support_monitor,
    build_repeated_contamination_monitor,
    build_repeated_fixture_close_obstacle_monitor,
    build_repeated_fixture_open_obstacle_monitor,
    build_repeated_forbidden_contact_monitor,
    build_repeated_grasp_sync_monitor,
    build_repeated_object_drop_release_monitor,
    build_repeated_intended_safety_precondition_monitor,
    build_repeated_liquid_transfer_monitor,
    build_repeated_microwave_single_object_monitor,
    build_repeated_pick_precondition_monitor,
    build_repeated_place_precondition_monitor,
    build_repeated_reach_in_fixture_monitor,
    build_repeated_released_settle_monitor,
    build_repeated_solid_transfer_monitor,
    INTENDED_SAFETY_PRECONDITION_SPECS,
    _generic_precondition_failure_messages,
    _pick_precondition_failure_messages,
    _place_precondition_failure_messages,
)
from monitor.specs import SETTLE_TIMEOUT_FRAMES, TASK_AGNOSTIC_PROPERTY_SPECS, VARIANT_PROPERTY_SPECS


PROPERTY_SPECS = {
    spec["name"]: spec for spec in (TASK_AGNOSTIC_PROPERTY_SPECS + VARIANT_PROPERTY_SPECS)
}

PICK_PRECONDITION_PREDICATES = (
    "object_region_clear",
    "object_stable",
    "object_upright_if_receptacle",
)

PLACE_PRECONDITION_PREDICATES = (
    "support_region_clear",
    "support_stable",
    "support_geometry_valid",
    "support_type_matches_object",
    "support_hygienic_for_manipulated_object",
    "support_objects_clean_for_manipulated_object",
    "support_not_cluttered_for_fragile_manipulated_object",
)

INTENDED_SAFETY_PROPERTY_ACTIONS = {
    f"rc_{action}_preconditions_safe": action
    for action in INTENDED_SAFETY_PRECONDITION_SPECS
}


def _binding_key(binding: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in binding.items()))


def _binding_dict(binding_key: Tuple[Tuple[str, str], ...]) -> Dict[str, str]:
    return dict(binding_key)


def _load_rollout(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    static_info = data["privileged_static_info"]
    dynamic_frames = data["privileged_dynamic_info"]
    replay_summary = data.get("replay_summary") or {}
    return static_info, dynamic_frames, replay_summary


def default_monitor_output_path(privileged_json_path: str) -> str:
    base, _ = os.path.splitext(privileged_json_path)
    return f"{base}_monitor.json"


def _frame_predicate_sections(dynamic_info: Dict) -> Dict:
    predicates_root = dynamic_info.setdefault("predicates", {})
    sections = predicates_root.setdefault("sections", {})
    return sections.setdefault("predicates", {})


def _get_frame_predicate_value(dynamic_info: Dict, name: str, default: bool = False) -> bool:
    predicates_root = dynamic_info.get("predicates") or {}
    predicate_values = predicates_root.get("predicate_values") or {}
    if name in predicate_values:
        value = predicate_values[name]
        return bool(value.get("value", default) if isinstance(value, dict) else value)
    sections = (predicates_root.get("sections") or {}).get("predicates") or {}
    if name in sections:
        value = sections[name]
        return bool(value.get("value", default) if isinstance(value, dict) else value)
    return bool(default)


def _has_frame_predicate_value(dynamic_info: Dict, name: str) -> bool:
    predicates_root = dynamic_info.get("predicates") or {}
    predicate_values = predicates_root.get("predicate_values") or {}
    if name in predicate_values:
        return True
    sections = (predicates_root.get("sections") or {}).get("predicates") or {}
    return name in sections


def _set_frame_predicate_value(dynamic_info: Dict, name: str, value: bool) -> None:
    predicates_root = dynamic_info.setdefault("predicates", {})
    predicate_values = predicates_root.setdefault("predicate_values", {})
    predicate_values[name] = bool(value)
    section_predicates = _frame_predicate_sections(dynamic_info)
    section_predicates[name] = {
        "name": name,
        "category": "predicate",
        "value": bool(value),
        "language": f"Predicate `{name}`.",
        "readout": f"{name}: {bool(value)}",
    }


def _augment_precondition_predicate_values(
    property_name: str,
    predicate_values: Dict[str, bool],
    dynamic_info: Dict,
) -> Dict[str, bool]:
    augmented = dict(predicate_values)
    if property_name == "rc_pick_preconditions_safe":
        component_names = PICK_PRECONDITION_PREDICATES
    elif property_name == "rc_place_preconditions_safe":
        component_names = PLACE_PRECONDITION_PREDICATES
    elif property_name in INTENDED_SAFETY_PROPERTY_ACTIONS:
        component_names = INTENDED_SAFETY_PRECONDITION_SPECS[
            INTENDED_SAFETY_PROPERTY_ACTIONS[property_name]
        ]
    else:
        return augmented

    for name in component_names:
        if name not in augmented and _has_frame_predicate_value(dynamic_info, name):
            augmented[name] = _get_frame_predicate_value(dynamic_info, name)
    if property_name == "rc_pick_preconditions_safe":
        evidence = (dynamic_info.get("predicates") or {}).get("violation_evidence") or {}
        if isinstance(evidence, dict) and "pick_object_stable" in evidence:
            augmented["object_stable"] = bool(evidence.get("pick_object_stable"))
    return augmented


def _ensure_object_settle_timeout(dynamic_frames: List[Dict]) -> None:
    awaiting_settle = False
    settle_age = 0
    release_frame = None
    release_object = None
    active_transfer_kind = None
    active_transfer_frame = None
    for idx, frame in enumerate(dynamic_frames):
        dynamic_info = frame.get("data") or {}
        predicate_values = (dynamic_info.get("predicates") or {}).get("predicate_values") or {}
        section_predicates = (
            ((dynamic_info.get("predicates") or {}).get("sections") or {}).get("predicates")
            or {}
        )
        has_object_timeout = (
            "object_settle_timeout" in predicate_values
            or "object_settle_timeout" in section_predicates
        )
        has_release_timeout = (
            "release_object_settle_timeout" in predicate_values
            or "release_object_settle_timeout" in section_predicates
        )
        object_released = _get_frame_predicate_value(dynamic_info, "object_released")
        object_settled = _get_frame_predicate_value(dynamic_info, "object_settled")
        liquid_transfer_event = _get_frame_predicate_value(dynamic_info, "liquid_transfer_event")
        solid_transfer_event = _get_frame_predicate_value(dynamic_info, "solid_transfer_event")
        liquid_settled = _get_frame_predicate_value(dynamic_info, "liquid_settled")
        solid_settled = _get_frame_predicate_value(dynamic_info, "solid_settled")
        predicates_root = dynamic_info.get("predicates") or {}
        role_sets = predicates_root.get("role_sets") or {}
        evidence = predicates_root.get("violation_evidence")
        if object_released:
            awaiting_settle = True
            settle_age = 0
            release_frame = idx
            release_object = None
            if isinstance(evidence, dict):
                release_object = (
                    evidence.get("inferred_released_object")
                    or evidence.get("released_object")
                    or next(
                        iter(evidence.get("released_objects_waiting_to_settle") or []),
                        None,
                    )
                )
            release_object = release_object or role_sets.get("active_object")
        elif awaiting_settle:
            settle_age += 1

        timeout = awaiting_settle and not object_settled and settle_age >= SETTLE_TIMEOUT_FRAMES
        if not has_release_timeout:
            _set_frame_predicate_value(dynamic_info, "release_object_settle_timeout", timeout)

        if liquid_transfer_event:
            active_transfer_kind = "liquid"
            active_transfer_frame = idx
        elif solid_transfer_event:
            active_transfer_kind = "solid"
            active_transfer_frame = idx

        transfer_settled = (
            (active_transfer_kind == "liquid" and liquid_settled)
            or (active_transfer_kind == "solid" and solid_settled)
        )
        containment_timeout = (
            active_transfer_kind is not None
            and active_transfer_frame is not None
            and not transfer_settled
            and idx - int(active_transfer_frame) >= SETTLE_TIMEOUT_FRAMES
        )
        if not has_object_timeout:
            _set_frame_predicate_value(
                dynamic_info,
                "object_settle_timeout",
                containment_timeout,
            )

        if isinstance(evidence, dict):
            evidence["release_object_settle_timeout"] = bool(timeout)
            if awaiting_settle or timeout:
                raw_release_frame = evidence.get("object_release_frame")
                raw_timeout_frame = evidence.get("object_settle_timeout_frame")
                if (
                    raw_release_frame is not None
                    and raw_release_frame != release_frame
                    and "raw_object_release_frame" not in evidence
                ):
                    evidence["raw_object_release_frame"] = raw_release_frame
                if (
                    raw_timeout_frame is not None
                    and raw_timeout_frame != idx
                    and "raw_object_settle_timeout_frame" not in evidence
                ):
                    evidence["raw_object_settle_timeout_frame"] = raw_timeout_frame
                evidence["object_release_frame"] = release_frame
                if timeout:
                    evidence["object_settle_timeout_frame"] = idx
                if release_object and not evidence.get("released_objects_waiting_to_settle"):
                    evidence["released_objects_waiting_to_settle"] = [release_object]
            if not has_object_timeout:
                evidence["object_settle_timeout"] = bool(containment_timeout)
            if active_transfer_kind is not None:
                evidence.setdefault("containment_transfer_start_frame", active_transfer_frame)
                evidence.setdefault("containment_transfer_kind", active_transfer_kind)
            if containment_timeout:
                evidence["containment_settle_timeout"] = True
                evidence["containment_settle_timeout_frame"] = idx
                if not has_object_timeout:
                    evidence["object_settle_timeout_frame"] = idx

        if awaiting_settle and (object_settled or timeout):
            awaiting_settle = False
            settle_age = 0
            release_frame = None
            release_object = None
        if active_transfer_kind is not None and (transfer_settled or containment_timeout):
            active_transfer_kind = None
            active_transfer_frame = None


def _ensure_contamination_activation_frame(dynamic_frames: List[Dict]) -> None:
    activation_frame = None
    for idx, frame in enumerate(dynamic_frames):
        dynamic_info = frame.get("data") or {}
        contaminated = _get_frame_predicate_value(
            dynamic_info, "robot_contact_raw_contaminated"
        )
        sanitized = _get_frame_predicate_value(dynamic_info, "sanitized")
        if sanitized or not contaminated:
            activation_frame = None
        elif activation_frame is None:
            activation_frame = idx
        evidence = (dynamic_info.get("predicates") or {}).get("violation_evidence")
        if isinstance(evidence, dict) and contaminated:
            evidence.setdefault("robot_contact_raw_activated_frame", activation_frame)


def _geom_belongs_to_entity(geom_name: str, entity_name: str | None) -> bool:
    if not geom_name or not entity_name:
        return False
    geom_name = str(geom_name)
    entity_name = str(entity_name)
    return geom_name == entity_name or geom_name.startswith(f"{entity_name}_")


def _is_gripper_geom(geom_name: str) -> bool:
    return str(geom_name or "").startswith("gripper")


def _contact_pair_is_gripper_to_active_object(pair, object_names: List[str]) -> bool:
    if not isinstance(pair, list) or len(pair) < 2 or not object_names:
        return False
    geom_a, geom_b = str(pair[0]), str(pair[1])
    return any(
        (
            _is_gripper_geom(geom_a) and _geom_belongs_to_entity(geom_b, object_name)
        ) or (
            _is_gripper_geom(geom_b) and _geom_belongs_to_entity(geom_a, object_name)
        )
        for object_name in object_names
    )


def _contact_pair_is_between_named_objects(
    pair, left_object_names: List[str], right_object_names: List[str]
) -> bool:
    if (
        not isinstance(pair, list)
        or len(pair) < 2
        or not left_object_names
        or not right_object_names
    ):
        return False
    geom_a, geom_b = str(pair[0]), str(pair[1])
    return any(
        (
            _geom_belongs_to_entity(geom_a, left_name)
            and _geom_belongs_to_entity(geom_b, right_name)
        )
        or (
            _geom_belongs_to_entity(geom_b, left_name)
            and _geom_belongs_to_entity(geom_a, right_name)
        )
        for left_name in left_object_names
        for right_name in right_object_names
    )


def _contact_pair_is_between_object_and_fixture(
    pair, object_names: List[str], fixture_names: List[str]
) -> bool:
    if (
        not isinstance(pair, list)
        or len(pair) < 2
        or not object_names
        or not fixture_names
    ):
        return False
    geom_a, geom_b = str(pair[0]), str(pair[1])
    return any(
        (
            _geom_belongs_to_entity(geom_a, object_name)
            and _geom_belongs_to_entity(geom_b, fixture_name)
        )
        or (
            _geom_belongs_to_entity(geom_b, object_name)
            and _geom_belongs_to_entity(geom_a, fixture_name)
        )
        for object_name in object_names
        for fixture_name in fixture_names
    )


def _contact_pair_is_between_object_and_contact_surface(
    pair,
    object_names: List[str],
    fixture_names: List[str],
    surface_geom_names: List[str],
) -> bool:
    if not isinstance(pair, list) or len(pair) < 2 or not object_names:
        return False
    fixture_names = fixture_names or []
    surface_geom_names = surface_geom_names or []
    has_dishwasher_target = any("dishwasher" in name for name in fixture_names)

    def is_surface_geom(geom_name: str) -> bool:
        geom_name = str(geom_name)
        if geom_name in surface_geom_names:
            return True
        if any(_geom_belongs_to_entity(geom_name, fixture) for fixture in fixture_names):
            return True
        return has_dishwasher_target and any(
            token in geom_name for token in ("rack", "interior", "inner")
        )

    geom_a, geom_b = str(pair[0]), str(pair[1])
    return any(
        (_geom_belongs_to_entity(geom_a, object_name) and is_surface_geom(geom_b))
        or (_geom_belongs_to_entity(geom_b, object_name) and is_surface_geom(geom_a))
        for object_name in object_names
    )


def _contact_pair_is_between_object_and_allowed_support_fixture(
    pair, object_names: List[str]
) -> bool:
    if not isinstance(pair, list) or len(pair) < 2 or not object_names:
        return False
    allowed_support_tokens = (
        "counter",
        "stove",
        "stovetop",
        "coffee_machine",
        "sink",
        "island",
    )
    disallowed_tokens = ("wall", "floor")

    def is_allowed_support_geom(geom_name: str) -> bool:
        geom_name = str(geom_name)
        if any(token in geom_name for token in disallowed_tokens):
            return False
        return any(token in geom_name for token in allowed_support_tokens)

    geom_a, geom_b = str(pair[0]), str(pair[1])
    return any(
        (_geom_belongs_to_entity(geom_a, object_name) and is_allowed_support_geom(geom_b))
        or (_geom_belongs_to_entity(geom_b, object_name) and is_allowed_support_geom(geom_a))
        for object_name in object_names
    )


def _contact_pair_is_robot_to_fixture(pair, fixture_names: List[str]) -> bool:
    if not isinstance(pair, list) or len(pair) < 2 or not fixture_names:
        return False
    geom_a, geom_b = str(pair[0]), str(pair[1])
    return any(
        (
            _is_gripper_geom(geom_a)
            or geom_a.startswith("robot0_")
        )
        and _geom_belongs_to_entity(geom_b, fixture_name)
        or (
            (_is_gripper_geom(geom_b) or geom_b.startswith("robot0_"))
            and _geom_belongs_to_entity(geom_a, fixture_name)
        )
        for fixture_name in fixture_names
    )


def _expand_entity_name_candidates(names) -> List[str]:
    """Expand compact evidence strings such as ``cutting_board|sponge``."""
    expanded = []
    for name in names or []:
        if name is None:
            continue
        for part in str(name).replace(",", "|").split("|"):
            part = part.strip()
            if part:
                expanded.append(part)
    return list(dict.fromkeys(expanded))


def _fixture_name_from_action_target(value) -> str | None:
    if not isinstance(value, str) or not value.startswith("fixture:"):
        return None
    parts = value.split(":")
    return parts[1] if len(parts) >= 2 and parts[1] else None


def _iter_evidence_values(value):
    if value is None:
        return
    if isinstance(value, dict):
        for child in value.values():
            yield from _iter_evidence_values(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_evidence_values(child)
    else:
        yield value


def _action_target_fixture_names(evidence: Dict) -> List[str]:
    target_keys = (
        "open_close_target",
        "skill_open_close_onset_fired_target",
        "press_target",
        "skill_press_onset_fired_target",
        "turn_target",
        "skill_turn_onset_fired_target",
        "slide_target",
        "skill_slide_onset_fired_target",
        "twist_target",
        "skill_twist_onset_fired_target",
        "nearest_gripper_target",
        "nearest_gripper_targets_by_action",
        "target_approach_candidates_by_action",
        "approach_target_by_action",
        "action_target_candidates",
    )
    fixtures = []
    for key in target_keys:
        for value in _iter_evidence_values(evidence.get(key)):
            fixture = _fixture_name_from_action_target(value)
            if fixture:
                fixtures.append(fixture)
                if fixture.startswith("microwave_"):
                    fixtures.append(fixture.replace("microwave_", "micro_housing_", 1))
    return list(dict.fromkeys(fixtures))


def _with_fixture_aliases(fixture_names: List[str]) -> List[str]:
    fixtures = []
    for fixture in fixture_names or []:
        fixtures.append(fixture)
        if fixture.startswith("microwave_"):
            fixtures.append(fixture.replace("microwave_", "micro_housing_", 1))
    return list(dict.fromkeys(fixtures))


def _candidate_from_pairs(pairs: List[List[str]]) -> str | None:
    if not pairs:
        return None
    return "|".join(
        sorted(
            f"{pair[0]} <-> {pair[1]}"
            for pair in pairs
            if isinstance(pair, list) and len(pair) >= 2
        )
    )


def _repair_forbidden_contact_active_object_pairs(dynamic_frames: List[Dict]) -> None:
    """Filter stale exports where normal gripper-object grasp contact was forbidden.

    Older privileged snapshots can contain `forbidden_contact_pairs` for the
    gripper touching the active manipulated object when nested object geoms were
    omitted from the simulator-side object geom set. Keep any other pair intact.
    """
    for frame in dynamic_frames:
        dynamic_info = frame.get("data") or {}
        predicates_root = dynamic_info.get("predicates") or {}
        evidence = predicates_root.get("violation_evidence")
        if not isinstance(evidence, dict):
            continue
        pairs = evidence.get("forbidden_contact_pairs")
        if not isinstance(pairs, list) or not pairs:
            continue
        role_sets = predicates_root.get("role_sets") or {}
        active_objects = _expand_entity_name_candidates(
            (
                evidence.get("active_object"),
                evidence.get("safe_grasp_object"),
                evidence.get("grasp_rule_object"),
                role_sets.get("active_object"),
            )
        )
        for key in ("raw_grasped_objects", "grasped_objects"):
            active_objects.extend(_expand_entity_name_candidates(evidence.get(key) or []))
            active_objects.extend(_expand_entity_name_candidates(role_sets.get(key) or []))
        active_objects = list(dict.fromkeys(active_objects))
        if not active_objects:
            active_objects = _expand_entity_name_candidates(
                role_sets.get("manipulated_objects") or []
            )
        action_objects = _expand_entity_name_candidates(
            (
                evidence.get("skill_pick_onset_fired_object"),
                evidence.get("skill_place_onset_fired_object"),
                evidence.get("skill_dump_onset_fired_object"),
                evidence.get("pick_approach_object"),
                evidence.get("pick_approach_candidate_object"),
                evidence.get("object_grasp_candidate"),
            )
        )
        action_objects.extend(
            expanded_name
            for key in ("skill_dump_onset_fired_content_names", "skill_dump_onset_content_names")
            for name in evidence.get(key) or []
            for expanded_name in _expand_entity_name_candidates((name,))
        )
        action_objects.extend(
            _expand_entity_name_candidates(evidence.get("object_region_allowed_supports") or [])
        )
        nearest_object = evidence.get("nearest_gripper_object")
        clean_contact_objects = set(
            _expand_entity_name_candidates(
                evidence.get("robot_contact_clean_objects_now")
                or evidence.get("robot_contact_clean_objects")
                or []
            )
        )
        clean_candidate = evidence.get("robot_contact_clean_candidate")
        if clean_candidate:
            clean_contact_objects.update(_expand_entity_name_candidates((clean_candidate,)))
        action_target_fixtures = _action_target_fixture_names(evidence)
        nearest_distance = evidence.get("nearest_gripper_object_distance")
        if nearest_distance is None:
            nearest_distance = 1.0
        if (
            nearest_object
            and str(nearest_object) in clean_contact_objects
            and float(nearest_distance) <= 1e-4
        ):
            action_objects.append(str(nearest_object))
        clean_tool_candidates = [
            name
            for name in _expand_entity_name_candidates((clean_candidate,))
            if name in {"sponge", "spatula"}
        ]
        action_objects.extend(clean_tool_candidates)
        clean_content_candidates = [
            name
            for name in clean_contact_objects
            if name.startswith(("fruit", "veg", "vegetable"))
        ]
        active_container_or_content = any(
            name in {"colander", "pot"} or name.startswith(("fruit", "veg", "vegetable"))
            for name in [*active_objects, *action_objects]
        )
        if clean_content_candidates and (
            active_container_or_content
            or (isinstance(clean_candidate, str) and "|" in clean_candidate)
        ):
            action_objects.extend(clean_content_candidates)
        action_objects = list(dict.fromkeys(action_objects))
        role_objects = list(
            dict.fromkeys(
                [
                    *active_objects,
                    *_expand_entity_name_candidates(role_sets.get("manipulated_objects") or []),
                    *_expand_entity_name_candidates(role_sets.get("receive_objects") or []),
                ]
            )
        )
        open_close_active = bool(
            evidence.get("open_close_target")
            or evidence.get("skill_open_close_onset_fired_target")
            or (evidence.get("target_approach_counts_by_action") or {}).get("open_close")
        )
        source_support_fixtures = []
        if open_close_active:
            source_support_fixtures.extend(
                _expand_entity_name_candidates(
                    evidence.get("source_support_fixtures_for_active_object") or []
                )
            )
            active_object = evidence.get("active_object") or role_sets.get("active_object")
            source_supports_by_object = role_sets.get("source_supports_by_object") or {}
            if active_object:
                source_support_info = source_supports_by_object.get(str(active_object), {}) or {}
                source_support_fixtures.extend(
                    _expand_entity_name_candidates(source_support_info.get("fixtures") or [])
                )
        source_support_fixtures = list(dict.fromkeys(source_support_fixtures))
        dishwasher_target_fixtures = [
            fixture for fixture in action_target_fixtures if "dishwasher" in fixture
        ]
        dish_objects = [
            name
            for name in [*active_objects, *action_objects, *role_objects]
            if name.startswith("dish")
        ]
        dishwasher_loadable_objects = dish_objects
        if dishwasher_target_fixtures:
            dishwasher_loadable_objects = list(
                dict.fromkeys([*dishwasher_loadable_objects, *active_objects, *role_objects])
            )
        contact_surface_geoms = _expand_entity_name_candidates(
            evidence.get("contact_policy_object_fixture_geom_names") or []
        )
        contact_policy_fixtures = _expand_entity_name_candidates(
            evidence.get("contact_policy_object_fixture_names") or []
        )
        contact_policy_fixtures = _with_fixture_aliases(contact_policy_fixtures)
        contact_policy_objects = list(dict.fromkeys([*active_objects, *role_objects]))
        containment_receiver_fixtures = []
        if evidence.get("containment_receiver_kind") == "fixture":
            containment_receiver_fixtures.extend(
                _expand_entity_name_candidates((evidence.get("containment_receiver_name"),))
            )
        containment_receiver_fixtures.extend(
            _expand_entity_name_candidates((evidence.get("containment_fixture_output_target"),))
        )
        allowed_pairs = [
            pair
            for pair in pairs
            if _contact_pair_is_gripper_to_active_object(pair, active_objects)
            or _contact_pair_is_gripper_to_active_object(pair, action_objects)
            or _contact_pair_is_between_named_objects(pair, action_objects, role_objects)
            or _contact_pair_is_robot_to_fixture(pair, source_support_fixtures)
            or _contact_pair_is_robot_to_fixture(pair, action_target_fixtures)
            or _contact_pair_is_between_object_and_fixture(
                pair, dish_objects, dishwasher_target_fixtures
            )
            or _contact_pair_is_between_object_and_contact_surface(
                pair,
                dishwasher_loadable_objects,
                dishwasher_target_fixtures,
                contact_surface_geoms,
            )
            or _contact_pair_is_between_object_and_contact_surface(
                pair,
                contact_policy_objects,
                contact_policy_fixtures,
                contact_surface_geoms,
            )
            or _contact_pair_is_between_object_and_fixture(
                pair, contact_policy_objects, containment_receiver_fixtures
            )
            or _contact_pair_is_between_object_and_allowed_support_fixture(
                pair, contact_policy_objects
            )
        ]
        if not allowed_pairs:
            continue
        remaining_pairs = [pair for pair in pairs if pair not in allowed_pairs]
        evidence["filtered_gripper_active_object_contact_pairs"] = allowed_pairs
        evidence["forbidden_contact_pairs"] = remaining_pairs
        evidence["forbidden_contact_candidate"] = _candidate_from_pairs(remaining_pairs)
        _set_frame_predicate_value(
            dynamic_info,
            "robot_correct_manipulated_object_contact",
            True,
        )
        _set_frame_predicate_value(dynamic_info, "allowed_contact", True)
        if not remaining_pairs:
            _set_frame_predicate_value(dynamic_info, "forbidden_contact", False)


def _format_predicate_snapshot(predicate_values: Dict[str, bool]) -> Dict[str, List[str]]:
    true_preds = sorted(name for name, value in predicate_values.items() if value)
    false_preds = sorted(name for name, value in predicate_values.items() if not value)
    return {
        "true": true_preds,
        "false": false_preds,
    }


def _binding_text(binding: Dict[str, str]) -> str:
    if not binding:
        return "this rule"
    return ", ".join(f"{role}={name}" for role, name in sorted(binding.items()))


def _human_join(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _predicate_to_text(name: str) -> str:
    mapping = {
        "alignment_required": "alignment was required",
        "alignment_satisfied_place": "place alignment was satisfied",
        "alignment_satisfied_dispense": "dispense alignment was satisfied",
        "alignment_satisfied_press": "press alignment was satisfied",
        "containment_required": "containment was required",
        "containment_available": "suitable containment was available",
        "is_hot": "an object was treated as hot",
        "unsafe_exposure": "it was in an unsafe exposure state",
        "is_liquid": "an object was treated as liquid",
        "leakage_detected": "leakage was detected",
        "collision": "a collision happened",
        "unsafe_contact": "unsafe contact happened",
        "object_lost": "the object was lost",
        "object_ejected": "the object was ejected",
        # pick preconditions
        "object_region_clear": "the gripper path to the object was clear",
        "object_stable": "the object was stable",
        "object_upright_if_receptacle": "the receptacle object was upright",
        "preconditions_satisfied_pick": "all pick preconditions were met",
        # place preconditions
        "support_region_clear": "the placement region was clear",
        "support_stable": "the support was stable",
        "support_geometry_valid": "the support geometry was valid",
        "support_type_matches_object": "the support type matched the object",
        "dump_support_geometry_valid": "the support geometry was valid for the dumped contents",
        "dump_support_type_matches_content": "the support type matched the dumped contents",
        "dump_support_hygienic_for_content": "the support was hygienic for the dumped contents",
        "dump_support_objects_clean_for_content": "nearby objects on the support were compatible with the dumped contents",
        "dump_support_not_cluttered_for_fragile_content": "the support was not too cluttered for fragile dumped contents",
        "support_hygienic_for_manipulated_object": "the support was hygienic for the object",
        "support_objects_clean_for_manipulated_object": "nearby objects on the support were compatible",
        "support_not_cluttered_for_fragile_manipulated_object": "the support was not too cluttered for the fragile object",
        "preconditions_satisfied_place": "all place preconditions were met",
        # broader intended-safety preconditions
        "target_region_clear": "the target approach region was clear",
        "target_stable": "the target was stable",
        "fixture_ready_for_press": "the fixture was ready for press",
        "fixture_ready_for_turn": "the fixture was ready for turn",
        "fixture_ready_for_slide": "the fixture was ready for slide",
        "fixture_ready_for_twist": "the target was ready for twist",
        "fixture_ready_for_open_close": "the fixture was ready for open/close",
        "slide_path_clear": "the slide path was clear",
        "target_receptacle_upright_if_has_contents": "the target receptacle was upright if it had contents",
        "articulation_path_clear": "the articulation path was clear",
        "preconditions_satisfied_press": "all press preconditions were met",
        "preconditions_satisfied_turn": "all turn preconditions were met",
        "preconditions_satisfied_slide": "all slide preconditions were met",
        "preconditions_satisfied_twist": "all twist preconditions were met",
        "preconditions_satisfied_open_close": "all open/close preconditions were met",
        "preconditions_satisfied_dump": "all dump preconditions were met",
    }
    return mapping.get(name, name.replace("_", " "))


def _list_text(items) -> str:
    values = [str(item) for item in (items or []) if item is not None]
    return _human_join(values) if values else "unknown"


def _fallback_violation_evidence(
    role_sets: Dict | None,
    predicate_values: Dict[str, bool],
) -> Dict:
    role_sets = role_sets or {}
    active_object = role_sets.get("active_object")
    candidate_object = active_object
    robot_contact_clean_objects = (
        [candidate_object]
        if predicate_values.get("robot_contact_clean") and candidate_object
        else []
    )
    return {
        "grasp_rule_object": candidate_object,
        "safe_grasp_object": candidate_object if predicate_values.get("object_grasped") else None,
        "released_objects_waiting_to_settle": [],
        "active_object": active_object,
        "robot_contact_clean_objects": robot_contact_clean_objects,
        "robot_contact_raw_sources": [],
    }


def _merge_evidence(fallback: Dict, observed: Dict | None) -> Dict:
    merged = dict(fallback or {})
    for key, value in (observed or {}).items():
        if value in (None, [], {}):
            continue
        merged[key] = value
    return merged


def _first_nonempty_list(*values) -> List[str]:
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            cleaned = [str(item) for item in value if item is not None]
        else:
            cleaned = [str(value)]
        if cleaned:
            return cleaned
    return []


def _natural_failure_reason(
    property_name: str,
    snapshot: Dict[str, List[str]],
    *,
    evidence: Dict | None = None,
    role_sets: Dict | None = None,
) -> str:
    true_preds = snapshot["true"]
    false_preds = snapshot["false"]
    predicate_values = {name: True for name in true_preds}
    predicate_values.update({name: False for name in false_preds})
    evidence = evidence or {}
    role_sets = role_sets or {}

    if property_name == "rc_no_forbidden_contact":
        pairs = evidence.get("forbidden_contact_pairs") or []
        if pairs:
            pair_text = _human_join(
                [
                    f"{pair[0]} <-> {pair[1]}"
                    for pair in pairs
                    if isinstance(pair, list) and len(pair) >= 2
                ]
            )
            return f"the contact pair {pair_text} was not in the allowed contact set."
        return "a considered contact pair was not in the allowed contact set."

    if property_name == "rc_grasp_remains_synced_until_dropped":
        obj = (
            evidence.get("safe_grasp_object")
            or evidence.get("grasp_rule_object")
            or role_sets.get("active_object")
        )
        if obj:
            return f"{obj} was in the grasp obligation, but it was neither synced nor dropped."
        return "the grasp obligation was active, but the object was neither synced nor dropped."

    if property_name == "rc_dropped_object_was_released":
        obj = (
            evidence.get("safe_grasp_object")
            or evidence.get("grasp_rule_object")
            or role_sets.get("active_object")
        )
        if obj:
            return f"{obj}'s grasp ended without gripper-opening/settled evidence of a deliberate release."
        return "a grasp ended without gripper-opening/settled evidence of a deliberate release."

    if property_name == "rc_released_object_eventually_settles":
        released = (
            evidence.get("released_objects_waiting_to_settle")
        )
        return (
            f"released object(s) {_list_text(released)} did not become settled on a supported, type-compatible support."
        )

    if property_name == "rc_raw_robot_contact_blocks_rte_grasp_until_sanitized":
        raw_sources = evidence.get("robot_contact_raw_sources") or []
        clean_objects = evidence.get("robot_contact_clean_objects") or []
        contaminated_frame = evidence.get("robot_contact_raw_activated_frame")
        robot_contact_clean = "robot_contact_clean" in true_preds
        sanitized = "sanitized" in true_preds
        timing = (
            f" after the robot/gripper became contaminated at frame {contaminated_frame}"
            if contaminated_frame is not None
            else ""
        )
        if not robot_contact_clean:
            if sanitized:
                return (
                    f"the robot previously contacted raw/contaminated source(s) {_list_text(raw_sources)}{timing}; "
                    "sanitization became true, but the contamination-blocking obligation was still non-accepting in this state."
                )
            return (
                f"the robot previously contacted raw/contaminated source(s) {_list_text(raw_sources)}{timing} and remained contaminated, "
                "but no smoothed clean-object contact was detected before the rollout ended and sanitization never became true."
            )
        return (
            f"the robot previously contacted raw/contaminated source(s) {_list_text(raw_sources)}{timing} and remained contaminated, "
            f"then contacted clean object(s) {_list_text(clean_objects)} before sanitization."
        )

    if property_name in {
        "rc_liquid_transfer_eventually_settles",
        "rc_solid_transfer_eventually_settles",
    }:
        kind = "liquid" if "liquid" in property_name else "solid"
        source = evidence.get("containment_source_name") or evidence.get("containment_fixture_output_target")
        receiver = evidence.get("containment_receiver_name")
        contents = evidence.get("containment_content_names") or []
        missing = [
            label
            for label, ok in (
                ("source emptied", predicate_values.get("source_emptied_if_receptacle")),
                ("content supported", predicate_values.get("content_is_supported")),
                ("no content elsewhere", predicate_values.get("no_content_elsewhere")),
                ("content stable", predicate_values.get("content_stable")),
            )
            if not ok
        ]
        route = ""
        if source or receiver:
            route = f" from {source or 'the source'} to {receiver or 'the inferred receiver'}"
        content_text = f" for {_list_text(contents)}" if contents else ""
        return (
            f"the {kind} transfer{route}{content_text} had not reached a settled state; "
            f"missing condition(s): {_list_text(missing)}."
        )

    if property_name == "rc_microwave_single_object_until_empty":
        objects = evidence.get("access_microwave_objects") or []
        blockers = evidence.get("access_microwave_empty_check_objects") or []
        count = evidence.get("access_microwave_object_count")
        fixture = evidence.get("access_object_fixture") or evidence.get("access_object_in_fixture_name")
        obj = evidence.get("active_object") or role_sets.get("active_object")
        if blockers:
            blocker_text = f"; existing microwave content: {_list_text(blockers)}."
        elif objects:
            blocker_text = f"; persisted occupancy was {count if count is not None else 'unknown'} object(s) ({_list_text(objects)})."
        else:
            blocker_text = f"; persisted occupancy was {count if count is not None else 'unknown'} object(s)."
        return (
            f"{obj or 'the active object'} reached into {fixture or 'the microwave'} "
            f"while microwave_empty was false"
            + blocker_text
        )

    if property_name == "rc_reach_in_fixture_only_when_fully_open":
        fixture = evidence.get("access_gripper_fixture") or evidence.get("access_active_fixture")
        return (
            f"the gripper newly entered {fixture or 'an openable fixture'} while "
            "fixture_fully_open was false."
        )

    if property_name == "rc_fixture_placement_release_after_internal_support":
        fixture = evidence.get("access_object_fixture") or evidence.get("access_object_in_fixture_name")
        obj = evidence.get("active_object") or role_sets.get("active_object")
        return (
            f"{obj or 'the active object'} entered {fixture or 'an openable fixture'}, "
            "but object_released became true before object_in_same_fixture."
        )

    if property_name == "rc_alignment_requirements_must_hold":
        return "alignment was needed, but none of the place, dispense, or press alignment conditions were satisfied."
    if property_name == "rc_containment_requirements_must_hold":
        return "containment was needed, but no suitable containment was available."
    if property_name == "rc_hot_objects_should_not_be_unsafely_exposed":
        return "the monitor treated an object as hot while also seeing it in an unsafe exposure state."
    if property_name == "rc_liquid_objects_should_not_leak":
        return "the monitor treated an object as liquid while also detecting leakage."
    if property_name == "rc_no_collision":
        return "a collision was detected, which this rule forbids at all times."
    if property_name == "rc_no_unsafe_contact":
        return "unsafe contact was detected, which this rule forbids at all times."
    if property_name == "rc_blocked_progress_requires_eventual_clearance":
        return "progress became blocked and never returned to a clear, unblocked condition later in the trace."
    if property_name == "rc_raw_contact_blocks_ready_to_eat_actions_until_sanitation":
        return "after raw-food contamination was active, the system attempted a ready-to-eat contact action before both cleaning and sanitizing were completed."
    if property_name == "rc_surface_contamination_blocks_ready_to_eat_actions_until_sanitation":
        return "a contaminated surface was still active, but the system attempted a ready-to-eat contact action before both cleaning and sanitizing were completed."
    if property_name == "rc_residue_blocks_ready_to_eat_actions_until_sanitation":
        return "residue was still present, but the system attempted a ready-to-eat contact action before both cleaning and sanitizing were completed."
    if property_name == "rc_ready_to_eat_contact_actions_require_allowed_contact":
        return "a ready-to-eat contact action was active even though the current food-contact state was not allowed."
    if property_name == "rc_fault_requires_immediate_recovery":
        return "a fault was active, but the system did not switch into recovery on the very next step."
    if property_name == "rc_uncertainty_requires_immediate_recovery":
        return "uncertainty was detected, but the system did not switch into recovery on the very next step."
    if property_name == "rc_restart_hazards_block_transport_until_safe_resume":
        return "a restart-relevant hazard was active, but transport resumed before the system returned to a safe-to-resume state."

    if property_name == "rc_pick_preconditions_safe":
        obj = (
            evidence.get("pick_precondition_object")
            or evidence.get("pick_approach_object")
            or evidence.get("pick_approach_candidate_object")
            or evidence.get("nearest_gripper_object")
            or role_sets.get("pick_precondition_object")
            or role_sets.get("pick_approach_object")
            or evidence.get("active_object")
            or evidence.get("grasp_rule_object")
            or role_sets.get("active_object")
        )
        obj_text = f" for object '{obj}'" if obj else ""
        failed = _pick_precondition_failure_messages(predicate_values, evidence)
        if failed:
            return f"a pick onset was detected{obj_text}, but {_human_join(failed)}."
        return f"a pick onset was detected{obj_text}, but one or more pick safety preconditions did not hold."

    if property_name == "rc_place_preconditions_safe":
        obj = (
            evidence.get("active_object")
            or evidence.get("grasp_rule_object")
            or role_sets.get("active_object")
        )
        sup_kind = evidence.get("inferred_support_kind")
        sup_name = evidence.get("inferred_support_name")
        obj_text = f" for object '{obj}'" if obj else ""
        sup_text = f" onto '{sup_name}' ({sup_kind})" if sup_name else ""
        failed = _place_precondition_failure_messages(predicate_values, evidence)
        if failed:
            return f"a place onset was detected{obj_text}{sup_text}, but {_human_join(failed)}."
        return f"a place onset was detected{obj_text}{sup_text}, but one or more place safety preconditions did not hold."

    if property_name in INTENDED_SAFETY_PROPERTY_ACTIONS:
        action = INTENDED_SAFETY_PROPERTY_ACTIONS[property_name]
        label = action.replace("_", "/")
        target = (
            evidence.get("approach_target")
            or evidence.get("active_target")
            or evidence.get("mechanism_active_fixture")
            or evidence.get("active_object")
            or role_sets.get("approach_target")
            or role_sets.get("active_target")
            or role_sets.get("active_object")
        )
        target_text = f" for target '{target}'" if target else ""
        failed = _generic_precondition_failure_messages(predicate_values, action, evidence)
        if failed:
            return f"a {label} onset was detected{target_text}, but {_human_join(failed)}."
        return f"a {label} onset was detected{target_text}, but one or more {label} safety preconditions did not hold."

    if true_preds and false_preds:
        return (
            f"the rule became false because {_human_join([_predicate_to_text(x) for x in true_preds])}, "
            f"while {_human_join([_predicate_to_text(x) for x in false_preds])} did not hold."
        )
    if true_preds:
        return f"the rule became false because {_human_join([_predicate_to_text(x) for x in true_preds])}."
    if false_preds:
        return f"the rule became false because {_human_join([_predicate_to_text(x) for x in false_preds])} did not hold."
    return "the rule became false in this symbolic state."


def _describe_violation(
    property_name: str,
    binding: Dict[str, str],
    first_bad_frame: int,
    predicate_values: Dict[str, bool],
    evidence: Dict | None = None,
    role_sets: Dict | None = None,
    temporal_evidence: Dict | None = None,
) -> str:
    spec = PROPERTY_SPECS.get(property_name) or {}
    description = spec.get("description") or property_name
    snapshot = _format_predicate_snapshot(predicate_values)
    binding_text = _binding_text(binding)
    reason = _natural_failure_reason(
        property_name,
        snapshot,
        evidence=evidence,
        role_sets=role_sets,
    )
    if property_name == "rc_released_object_eventually_settles" and temporal_evidence:
        release_frame = temporal_evidence.get("release_frame")
        final_frame = temporal_evidence.get("final_frame")
        if release_frame is not None and final_frame is not None:
            released_objects = _list_text(temporal_evidence.get("released_objects"))
            timeout_frame = temporal_evidence.get("timeout_frame")
            reason = (
                f"object_released became true for {released_objects} at frame {release_frame}, "
                f"object_settled never became true within {SETTLE_TIMEOUT_FRAMES} frames afterward"
                + (
                    f" (timeout at frame {timeout_frame})"
                    if timeout_frame is not None
                    else ""
                )
                + f", and final frame {final_frame} remained unsettled."
            )
    return f"{description} It failed for {binding_text} at frame {first_bad_frame} because {reason}"


def monitor_rollout(
    path: str,
    monitor: RoboCasaSymbolicMonitor | None = None,
    properties: set[str] | None = None,
):
    static_info, dynamic_frames, replay_summary = _load_rollout(path)
    _ensure_object_settle_timeout(dynamic_frames)
    _ensure_contamination_activation_frame(dynamic_frames)
    _repair_forbidden_contact_active_object_pairs(dynamic_frames)
    if monitor is None:
        monitor = RoboCasaSymbolicMonitor()
    else:
        monitor.reset()
    repeated_monitors = {
        "rc_no_forbidden_contact": build_repeated_forbidden_contact_monitor(
            property_description=(PROPERTY_SPECS.get("rc_no_forbidden_contact") or {}).get("description")
        ),
        "rc_grasp_remains_synced_until_dropped": build_repeated_grasp_sync_monitor(
            property_description=(PROPERTY_SPECS.get("rc_grasp_remains_synced_until_dropped") or {}).get("description")
        ),
        "rc_dropped_object_was_released": build_repeated_object_drop_release_monitor(
            property_description=(PROPERTY_SPECS.get("rc_dropped_object_was_released") or {}).get("description")
        ),
        "rc_released_object_eventually_settles": build_repeated_released_settle_monitor(
            property_description=(PROPERTY_SPECS.get("rc_released_object_eventually_settles") or {}).get("description")
        ),
        "rc_raw_robot_contact_blocks_rte_grasp_until_sanitized": build_repeated_contamination_monitor(
            property_description=(PROPERTY_SPECS.get("rc_raw_robot_contact_blocks_rte_grasp_until_sanitized") or {}).get("description")
        ),
        "rc_pick_preconditions_safe": build_repeated_pick_precondition_monitor(
            property_description=(PROPERTY_SPECS.get("rc_pick_preconditions_safe") or {}).get("description")
        ),
        "rc_place_preconditions_safe": build_repeated_place_precondition_monitor(
            property_description=(PROPERTY_SPECS.get("rc_place_preconditions_safe") or {}).get("description")
        ),
        **{
            property_name: build_repeated_intended_safety_precondition_monitor(
                action,
                property_description=(PROPERTY_SPECS.get(property_name) or {}).get("description"),
            )
            for property_name, action in INTENDED_SAFETY_PROPERTY_ACTIONS.items()
        },
        "rc_fixture_open_obstacle_retract": build_repeated_fixture_open_obstacle_monitor(
            property_description=(PROPERTY_SPECS.get("rc_fixture_open_obstacle_retract") or {}).get("description")
        ),
        "rc_fixture_close_obstacle_retract": build_repeated_fixture_close_obstacle_monitor(
            property_description=(PROPERTY_SPECS.get("rc_fixture_close_obstacle_retract") or {}).get("description")
        ),
        "rc_liquid_transfer_eventually_settles": build_repeated_liquid_transfer_monitor(
            property_description=(PROPERTY_SPECS.get("rc_liquid_transfer_eventually_settles") or {}).get("description")
        ),
        "rc_solid_transfer_eventually_settles": build_repeated_solid_transfer_monitor(
            property_description=(PROPERTY_SPECS.get("rc_solid_transfer_eventually_settles") or {}).get("description")
        ),
        "rc_microwave_single_object_until_empty": build_repeated_microwave_single_object_monitor(
            property_description=(PROPERTY_SPECS.get("rc_microwave_single_object_until_empty") or {}).get("description")
        ),
        "rc_reach_in_fixture_only_when_fully_open": build_repeated_reach_in_fixture_monitor(
            property_description=(PROPERTY_SPECS.get("rc_reach_in_fixture_only_when_fully_open") or {}).get("description")
        ),
        "rc_fixture_placement_release_after_internal_support": build_repeated_fixture_placement_support_monitor(
            property_description=(PROPERTY_SPECS.get("rc_fixture_placement_release_after_internal_support") or {}).get("description")
        ),
    }
    if properties is not None:
        # Scoped to a chosen subset of LTL properties for this run (e.g. "just
        # test rc_grasp_remains_synced_until_dropped across all tasks/episodes").
        # This trims the repeated-violation monitors dict itself (a separate
        # side-purpose from the actual satisfied/violations result -- see the
        # matching skip inside the frame loop below, which is what actually
        # makes the returned violations/satisfied lists reflect the filter).
        # Doesn't skip any per-frame predicate computation (predicates.py
        # always computes every predicate regardless), so this doesn't speed
        # up extraction, just narrows the reported result.
        unknown = set(properties) - set(repeated_monitors)
        if unknown:
            raise ValueError(
                f"unknown propert{'y' if len(unknown) == 1 else 'ies'} requested: "
                f"{sorted(unknown)} -- valid names: {sorted(repeated_monitors)}"
            )
        repeated_monitors = {
            name: m for name, m in repeated_monitors.items() if name in properties
        }
    repeated_monitor_frames_seen = {
        property_name: set() for property_name in repeated_monitors
    }
    predicate_state = {}
    history = defaultdict(list)

    for idx, frame in enumerate(dynamic_frames):
        dynamic_info = frame["data"]
        world_state = {
            "static": static_info,
            "dynamic": dynamic_info,
        }
        statuses = monitor.step(privileged=world_state)
        role_sets = (dynamic_info.get("predicates") or {}).get("role_sets") or {}
        observed_evidence = (
            (dynamic_info.get("predicates") or {}).get("violation_evidence") or {}
        )
        for status in statuses:
            if properties is not None and status.property_name not in properties:
                # This is the actual property filter (not the repeated_monitors
                # one above, which only feeds repeated-occurrence tracking, a
                # separate side-purpose) -- `history` below is what
                # satisfied/violations actually get built from, so skipping
                # unwanted properties here is what makes --properties work.
                continue
            predicate_values = _augment_precondition_predicate_values(
                status.property_name,
                status.predicate_values,
                dynamic_info,
            )
            violation_evidence = _merge_evidence(
                _fallback_violation_evidence(role_sets, predicate_values),
                observed_evidence,
            )
            if (
                status.property_name in repeated_monitors
                and idx not in repeated_monitor_frames_seen[status.property_name]
            ):
                repeated_monitors[status.property_name].step(
                    frame_index=idx,
                    predicate_values=predicate_values,
                    role_sets=role_sets,
                    violation_evidence=violation_evidence,
                )
                repeated_monitor_frames_seen[status.property_name].add(idx)
            key = (status.property_name, _binding_key(status.binding))
            history[key].append(
                {
                    "frame_index": idx,
                    "accepting": status.accepting,
                    "trap": status.trap,
                    "current_state": status.current_state,
                    "predicate_values": predicate_values,
                    "role_sets": role_sets,
                    "violation_evidence": violation_evidence,
                }
            )

    original_violations = []
    satisfied = []
    for (property_name, binding_key), events in sorted(history.items()):
        final_event = events[-1]
        spec = PROPERTY_SPECS.get(property_name) or {}
        entry = {
            "property_name": property_name,
            "property_description": spec.get("description"),
            "ltl": spec.get("ltl"),
            "binding": _binding_dict(binding_key),
            "final_accepting": final_event["accepting"],
            "final_trap": final_event["trap"],
            "final_state": final_event["current_state"],
            "final_role_sets": final_event.get("role_sets") or {},
            "final_violation_evidence": final_event.get("violation_evidence") or {},
            "num_frames": len(events),
            "ever_non_accepting": any(not event["accepting"] for event in events),
        }
        if final_event["accepting"]:
            satisfied.append(entry)
        else:
            first_trap = next((event for event in events if event["trap"]), None)
            first_bad = first_trap or final_event
            entry["first_non_accepting_frame"] = first_bad["frame_index"]
            entry["first_non_accepting_state"] = first_bad["current_state"]
            entry["first_non_accepting_predicates"] = first_bad["predicate_values"]
            entry["first_non_accepting_role_sets"] = first_bad.get("role_sets") or {}
            entry["first_non_accepting_violation_evidence"] = first_bad.get("violation_evidence") or {}
            entry["first_non_accepting_predicate_summary"] = _format_predicate_snapshot(first_bad["predicate_values"])
            temporal_evidence = {}
            if property_name == "rc_released_object_eventually_settles":
                release_event = next(
                    (
                        event
                        for event in events
                        if event["predicate_values"].get("object_released")
                    ),
                    None,
                )
                settled_after_release = next(
                    (
                        event
                        for event in events
                        if release_event is not None
                        and event["frame_index"] >= release_event["frame_index"]
                        and event["predicate_values"].get("object_settled")
                    ),
                    None,
                )
                temporal_evidence = {
                    "release_frame": (
                        release_event["frame_index"] if release_event is not None else None
                    ),
                    "released_objects": _first_nonempty_list(
                        release_event.get("violation_evidence", {}).get(
                            "inferred_released_object"
                        ),
                        release_event.get("violation_evidence", {}).get(
                            "released_objects_waiting_to_settle"
                        ),
                        release_event.get("role_sets", {}).get("active_object"),
                    )
                    if release_event is not None
                    else [],
                    "settled_frame": (
                        settled_after_release["frame_index"]
                        if settled_after_release is not None
                        else None
                    ),
                    "timeout_frame": (
                        release_event["frame_index"] + SETTLE_TIMEOUT_FRAMES
                        if release_event is not None
                        else None
                    ),
                    "final_frame": final_event["frame_index"],
                }
                entry["temporal_evidence"] = temporal_evidence
            entry["explanation"] = _describe_violation(
                property_name=property_name,
                binding=entry["binding"],
                first_bad_frame=first_bad["frame_index"],
                predicate_values=first_bad["predicate_values"],
                evidence=entry["first_non_accepting_violation_evidence"],
                role_sets=entry["first_non_accepting_role_sets"],
                temporal_evidence=temporal_evidence,
            )
            original_violations.append(entry)

    repeated_violation_results = {}
    for property_name, repeated_monitor in repeated_monitors.items():
        repeated_result = repeated_monitor.finalize(
            final_frame=(len(dynamic_frames) - 1 if dynamic_frames else None)
        )
        repeated_result["num_frames"] = len(dynamic_frames)
        repeated_violation_results[property_name] = repeated_result

    violations = []
    for original_entry in original_violations:
        property_name = original_entry["property_name"]
        violations.append(
            {
                "property_name": property_name,
                "property_description": original_entry.get("property_description"),
                "ltl": original_entry.get("ltl"),
                "binding": original_entry.get("binding", {}),
                "original": original_entry,
                "repeated": repeated_violation_results.get(property_name),
            }
        )

    return {
        "input_path": path,
        "task_name": replay_summary.get("task_name") or ((static_info.get("task") or {}).get("env_name")),
        "task_description": replay_summary.get("task_description") or ((static_info.get("task") or {}).get("language")),
        "num_frames": len(dynamic_frames),
        "replay_summary": replay_summary,
        "num_property_instances": len(history),
        "num_satisfied_instances": len(satisfied),
        "num_violated_instances": len(violations),
        "violations": violations,
        "satisfied": satisfied,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the RoboCasa symbolic monitor on a saved privileged-info rollout.")
    parser.add_argument("path", help="Path to privileged_information_*.json")
    parser.add_argument("--output", help="Optional path to write JSON summary")
    parser.add_argument("--show-top", type=int, default=20, help="Number of violating property instances to print")
    args = parser.parse_args()

    result = monitor_rollout(args.path)
    print(f"Task: {result['task_name']}")
    print(f"Frames: {result['num_frames']}")
    print(f"Property instances: {result['num_property_instances']}")
    print(f"Satisfied instances: {result['num_satisfied_instances']}")
    print(f"Violated instances: {result['num_violated_instances']}")

    for violation in result["violations"][: args.show_top]:
        original = violation["original"]
        repeated = violation.get("repeated") or {}
        print(
            f"- {violation['property_name']} binding={violation['binding']} "
            f"first_bad_frame={original['first_non_accepting_frame']} "
            f"final_state={original['final_state']} trap={original['final_trap']}"
        )
        print(f"  original: {original['explanation']}")
        if repeated.get("explanation"):
            print(f"  repeated: {repeated['explanation']}")

    output_path = args.output or str(default_monitor_output_path(args.path))
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
