from __future__ import annotations

from copy import deepcopy as _deepcopy
from functools import lru_cache as _lru_cache

try:
    from monitor.sim.robocasa.attributes import (
        external_object_categories as _external_object_categories,
        object_is_receptacle_category as _object_is_receptacle_category,
    )
except Exception:  # pragma: no cover - keeps import robust in partial checkouts.
    _external_object_categories = None

    def _object_is_receptacle_category(category: str, extra_attrs=()) -> bool:  # type: ignore[misc]
        _FALLBACK = {"bowl", "cup", "mug", "pot", "pan", "tray", "container", "tupperware", "bottle", "can"}
        return str(category).lower() in _FALLBACK or bool(set(extra_attrs) & _FALLBACK)


def _resolve_entity(entity: Any, kwargs: Dict[str, Any]) -> Any:
    if entity is None:
        return kwargs.get("obj") or kwargs.get("object") or kwargs.get("entity")
    bindings = kwargs.get("bindings") or {}
    role_name = getattr(entity, "name", None)
    if role_name in bindings:
        return bindings[role_name]
    return entity


def _privileged(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return kwargs.get("privileged") or kwargs.get("privileged_info") or {}


def _dynamic_info(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    privileged = _privileged(kwargs)
    return kwargs.get("dynamic_info") or privileged.get("dynamic") or privileged.get("dynamic_info") or {}


def _static_info(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    privileged = _privileged(kwargs)
    return kwargs.get("static_info") or privileged.get("static") or privileged.get("static_info") or {}


def _scene_objects(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return ((_dynamic_info(kwargs).get("scene") or {}).get("objects") or {})


def _static_objects(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    static = _static_info(kwargs)
    return (
        (static.get("objects") or {})
        or ((static.get("scene") or {}).get("objects") or {})
        or ((static.get("scene_layout") or {}).get("objects") or {})
    )


def _object_dynamic(name: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return (_scene_objects(kwargs).get(str(name)) or {}) if name is not None else {}


def _object_static(name: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return (_static_objects(kwargs).get(str(name)) or {}) if name is not None else {}


def _object_category(name: Any, kwargs: Dict[str, Any]) -> Optional[str]:
    for entry in (_object_static(name, kwargs), _object_dynamic(name, kwargs)):
        for key in ("category", "type", "object_type"):
            value = entry.get(key)
            if value:
                return str(value).lower()
    return str(name).lower() if name is not None else None


@_lru_cache(maxsize=1)
def _external_object_metadata() -> Dict[str, Dict[str, Any]]:
    if _external_object_categories is None:
        return {}
    try:
        return _external_object_categories()
    except Exception:
        return {}


def _metadata_attributes(name: Any, kwargs: Dict[str, Any]) -> set[str]:
    category = _object_category(name, kwargs)
    if not category:
        return set()
    entry = _external_object_metadata().get(category) or {}
    attrs = set(str(attr).lower() for attr in (entry.get("attributes") or []))
    types = set(str(kind).lower() for kind in (entry.get("types") or []))
    return attrs | types | {category}


def _object_has_attribute(obj: Any, attribute_name: str, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    attribute = str(attribute_name).lower()
    explicit = (
        (_object_static(obj, kwargs).get("attributes") or {})
        or (_object_dynamic(obj, kwargs).get("attributes") or {})
    )
    if isinstance(explicit, dict) and attribute in explicit:
        return bool(explicit[attribute])
    if isinstance(explicit, (list, tuple, set)) and attribute in {str(v).lower() for v in explicit}:
        return True

    attrs = _metadata_attributes(obj, kwargs)
    if attribute in attrs:
        return True
    if attribute == "ready_to_eat":
        return bool(attrs & {"ready_to_eat", "cooked_food", "fruit", "vegetable", "dairy", "bread_food", "pastry"})
    if attribute == "raw":
        return bool(attrs & {"raw", "meat", "fish", "seafood"})
    return False


def entity_has_attribute(entity: Any, attribute_name: str, **kwargs) -> bool:
    return _object_has_attribute(_resolve_entity(entity, kwargs), attribute_name, **kwargs)


def _predicate_value(name: str, default: bool = False, **kwargs) -> bool:
    for source_name in ("predicate_values", "predicates"):
        source = kwargs.get(source_name) or {}
        if name in source:
            value = source[name]
            if isinstance(value, dict):
                value = value.get("value", default)
            return bool(value)

    for root in (kwargs, _privileged(kwargs), _dynamic_info(kwargs)):
        if not isinstance(root, dict):
            continue
        roots = [root]
        embedded = root.get("predicates")
        if isinstance(embedded, dict):
            roots.append(embedded)
        for candidate in roots:
            sections = candidate.get("sections") or {}
            predicates = (sections.get("predicates") or {}) if isinstance(sections, dict) else {}
            if name in predicates:
                entry = predicates[name]
                return bool(entry.get("value", default) if isinstance(entry, dict) else entry)
    return bool(default)


def forbidden_contact(obj: Any = None, support: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("forbidden_contact", False, **kwargs)


def allowed_contact(obj: Any = None, support: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("allowed_contact", False, **kwargs)


def robot_correct_manipulated_object_contact(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("robot_correct_manipulated_object_contact", False, **kwargs)


def robot_correct_fixture_contact(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("robot_correct_fixture_contact", False, **kwargs)


def correct_manipulated_object_correct_fixture_contact(obj: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("correct_manipulated_object_correct_fixture_contact", False, **kwargs)


def correct_manipulated_object_correct_receive_object_contact(obj: Any = None, target: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(target, kwargs))
    return _predicate_value("correct_manipulated_object_correct_receive_object_contact", False, **kwargs)


def grasped_object_exists(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("grasped_object_exists", False, **kwargs)


def object_grasped(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_grasped", False, **kwargs)


def object_stable(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_stable", False, **kwargs)


def object_stable_relative(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_stable_relative", False, **kwargs)


def object_sync(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_sync", False, **kwargs)


def object_upright(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_upright", False, **kwargs)


def object_grasped_safe(obj: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    return _predicate_value(
        "object_grasped_safe",
        object_grasped(obj, **kwargs) and object_sync(obj, **kwargs),
        **kwargs,
    )


def object_dropped(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_dropped", False, **kwargs)


def object_left_gripper(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_left_gripper", False, **kwargs)


def object_released(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_released", False, **kwargs)


def object_supported(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("object_supported", False, **kwargs)


def object_supported_on_correct(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("object_supported_on_correct", False, **kwargs)


def gripper_away_from_object(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("gripper_away_from_object", False, **kwargs)


def gripper_is_opening(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("gripper_is_opening", False, **kwargs)


def object_settled(obj: Any = None, support: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    support = _resolve_entity(support, kwargs)
    return _predicate_value(
        "object_settled",
        object_supported(obj, support, **kwargs)
        and support_type_matches_object(obj, support, **kwargs)
        and object_stable(obj, **kwargs)
        and gripper_away_from_object(obj, **kwargs),
        **kwargs,
    )


def object_settle_timeout(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("object_settle_timeout", False, **kwargs)


def release_object_settle_timeout(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("release_object_settle_timeout", False, **kwargs)


def sanitized(obj: Any = None, support: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("sanitized", False, **kwargs)


def robot_contact_raw_contaminated(obj: Any = None, support: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("robot_contact_raw_contaminated", False, **kwargs)


def object_is_rte(obj: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    return _predicate_value("object_is_rte", _object_has_attribute(obj, "ready_to_eat", **kwargs), **kwargs)


def robot_contact_clean(obj: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    return _predicate_value("robot_contact_clean", False, **kwargs)


# ---------------------------------------------------------------------------
# Intended-safety onset and preconditions (intended_safety.txt)
# ---------------------------------------------------------------------------

def gripper_is_closing(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("gripper_is_closing", False, **kwargs)


def gripper_near_object(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("gripper_near_object", False, **kwargs)


def skill_pick_onset(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("skill_pick_onset", False, **kwargs)


def skill_place_onset(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("skill_place_onset", False, **kwargs)


def object_region_clear(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("object_region_clear", False, **kwargs)


def _object_is_receptacle(obj: Any, kwargs: Dict[str, Any]) -> bool:
    category = _object_category(obj, kwargs) or ""
    extra_attrs = _metadata_attributes(obj, kwargs)
    return _object_is_receptacle_category(category, extra_attrs)


def object_upright_if_receptacle(obj: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    default = (not _object_is_receptacle(obj, kwargs)) or object_upright(obj, **kwargs)
    return _predicate_value("object_upright_if_receptacle", default, **kwargs)


def preconditions_satisfied_pick(obj: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    return _predicate_value(
        "preconditions_satisfied_pick",
        object_region_clear(obj, **kwargs)
        and object_stable(obj, **kwargs)
        and object_upright_if_receptacle(obj, **kwargs),
        **kwargs,
    )


def support_region_clear(support: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(support, kwargs)
    return _predicate_value("support_region_clear", False, **kwargs)


def support_stable(support: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(support, kwargs)
    return _predicate_value("support_stable", False, **kwargs)


def support_geometry_valid(support: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(support, kwargs)
    return _predicate_value("support_geometry_valid", False, **kwargs)


def support_type_matches_object(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("support_type_matches_object", True, **kwargs)


def dump_support_geometry_valid(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("dump_support_geometry_valid", True, **kwargs)


def dump_support_type_matches_content(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("dump_support_type_matches_content", True, **kwargs)


def dump_support_hygienic_for_content(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("dump_support_hygienic_for_content", True, **kwargs)


def dump_support_objects_clean_for_content(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("dump_support_objects_clean_for_content", True, **kwargs)


def dump_support_not_cluttered_for_fragile_content(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("dump_support_not_cluttered_for_fragile_content", True, **kwargs)


def support_hygienic_for_manipulated_object(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("support_hygienic_for_manipulated_object", True, **kwargs)


def support_objects_clean_for_manipulated_object(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("support_objects_clean_for_manipulated_object", True, **kwargs)


def support_not_cluttered_for_fragile_manipulated_object(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("support_not_cluttered_for_fragile_manipulated_object", True, **kwargs)


# ---------------------------------------------------------------------------
# Mechanism safety: fixture open/close obstacle recovery (mechanism_safety.txt)
# ---------------------------------------------------------------------------

def robot_fixture_contact(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("robot_fixture_contact", False, **kwargs)


def fixture_is_opening(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_is_opening", False, **kwargs)


def fixture_is_closing(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_is_closing", False, **kwargs)


def fixture_fully_open(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_fully_open", False, **kwargs)


def fixture_fully_closed(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_fully_closed", False, **kwargs)


def fixture_obstacle_contact(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_obstacle_contact", False, **kwargs)


def continue_fixture_open(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("continue_fixture_open", False, **kwargs)


def continue_fixture_close(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("continue_fixture_close", False, **kwargs)


def fixture_open_retract_path_clear(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_open_retract_path_clear", True, **kwargs)


def fixture_close_retract_path_clear(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_close_retract_path_clear", True, **kwargs)


def fixture_open_obstacle_hit(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_open_obstacle_hit", False, **kwargs)


def fixture_close_obstacle_hit(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_close_obstacle_hit", False, **kwargs)


def fixture_open_retracting(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_open_retracting", False, **kwargs)


def fixture_close_retracting(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_close_retracting", False, **kwargs)


def preconditions_satisfied_place(obj: Any = None, support: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    support = _resolve_entity(support, kwargs)
    return _predicate_value(
        "preconditions_satisfied_place",
        support_region_clear(support, **kwargs)
        and support_stable(support, **kwargs)
        and support_geometry_valid(support, **kwargs)
        and support_type_matches_object(obj, support, **kwargs)
        and support_hygienic_for_manipulated_object(obj, support, **kwargs)
        and support_objects_clean_for_manipulated_object(obj, support, **kwargs)
        and support_not_cluttered_for_fragile_manipulated_object(obj, support, **kwargs),
        **kwargs,
    )


def skill_press_onset(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("skill_press_onset", False, **kwargs)


def skill_turn_onset(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("skill_turn_onset", False, **kwargs)


def skill_slide_onset(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("skill_slide_onset", False, **kwargs)


def skill_twist_onset(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("skill_twist_onset", False, **kwargs)


def skill_open_close_onset(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("skill_open_close_onset", False, **kwargs)


def skill_dump_onset(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("skill_dump_onset", False, **kwargs)


def target_region_clear(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("target_region_clear", False, **kwargs)


def target_stable(target: Any = None, **kwargs) -> bool:
    target = _resolve_entity(target, kwargs)
    return _predicate_value("target_stable", object_stable(target, **kwargs), **kwargs)


def fixture_ready_for_press(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("fixture_ready_for_press", True, **kwargs)


def fixture_ready_for_turn(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("fixture_ready_for_turn", True, **kwargs)


def fixture_ready_for_slide(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("fixture_ready_for_slide", True, **kwargs)


def fixture_ready_for_twist(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("fixture_ready_for_twist", True, **kwargs)


def fixture_ready_for_open_close(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("fixture_ready_for_open_close", True, **kwargs)


def slide_path_clear(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("slide_path_clear", False, **kwargs)


def target_receptacle_upright_if_has_contents(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("target_receptacle_upright_if_has_contents", True, **kwargs)


def articulation_path_clear(target: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(target, kwargs)
    return _predicate_value("articulation_path_clear", False, **kwargs)


def preconditions_satisfied_press(target: Any = None, **kwargs) -> bool:
    target = _resolve_entity(target, kwargs)
    return _predicate_value(
        "preconditions_satisfied_press",
        target_region_clear(target, **kwargs)
        and target_stable(target, **kwargs)
        and fixture_ready_for_press(target, **kwargs),
        **kwargs,
    )


def preconditions_satisfied_turn(target: Any = None, **kwargs) -> bool:
    target = _resolve_entity(target, kwargs)
    return _predicate_value(
        "preconditions_satisfied_turn",
        target_region_clear(target, **kwargs)
        and target_stable(target, **kwargs)
        and fixture_ready_for_turn(target, **kwargs),
        **kwargs,
    )


def preconditions_satisfied_slide(target: Any = None, **kwargs) -> bool:
    target = _resolve_entity(target, kwargs)
    return _predicate_value(
        "preconditions_satisfied_slide",
        target_region_clear(target, **kwargs)
        and target_stable(target, **kwargs)
        and fixture_ready_for_slide(target, **kwargs)
        and slide_path_clear(target, **kwargs),
        **kwargs,
    )


def preconditions_satisfied_twist(target: Any = None, **kwargs) -> bool:
    target = _resolve_entity(target, kwargs)
    return _predicate_value(
        "preconditions_satisfied_twist",
        target_region_clear(target, **kwargs)
        and target_stable(target, **kwargs)
        and fixture_ready_for_twist(target, **kwargs)
        and target_receptacle_upright_if_has_contents(target, **kwargs),
        **kwargs,
    )


def preconditions_satisfied_open_close(target: Any = None, **kwargs) -> bool:
    target = _resolve_entity(target, kwargs)
    return _predicate_value(
        "preconditions_satisfied_open_close",
        target_region_clear(target, **kwargs)
        and target_stable(target, **kwargs)
        and fixture_ready_for_open_close(target, **kwargs)
        and articulation_path_clear(target, **kwargs),
        **kwargs,
    )


def preconditions_satisfied_dump(obj: Any = None, support: Any = None, **kwargs) -> bool:
    return _predicate_value(
        "preconditions_satisfied_dump",
        support_region_clear(support, **kwargs)
        and support_stable(support, **kwargs)
        and dump_support_geometry_valid(obj, support, **kwargs)
        and dump_support_type_matches_content(obj, support, **kwargs)
        and dump_support_hygienic_for_content(obj, support, **kwargs)
        and dump_support_objects_clean_for_content(obj, support, **kwargs)
        and dump_support_not_cluttered_for_fragile_content(obj, support, **kwargs),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Containment safety: fixture/dump content transfer settling
# ---------------------------------------------------------------------------

def containment_transfer_event(obj: Any = None, support: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value(
        "containment_transfer_event",
        fixture_output_started(fixture, **kwargs) or skill_dump_onset(obj, **kwargs),
        **kwargs,
    )


def fixture_output_started(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_output_started", False, **kwargs)


def fixture_output_stopped(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("fixture_output_stopped", False, **kwargs)


def fixture_content_output_started(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value(
        "fixture_content_output_started",
        fixture_output_started(fixture, **kwargs),
        **kwargs,
    )


def content_is_liquid(obj: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    attrs = _metadata_attributes(obj, kwargs)
    default = bool(attrs & {"liquid", "fluid", "sauce", "oil", "broth"})
    return _predicate_value("content_is_liquid", default, **kwargs)


def content_is_solid(obj: Any = None, **kwargs) -> bool:
    obj = _resolve_entity(obj, kwargs)
    attrs = _metadata_attributes(obj, kwargs)
    default = bool(attrs & {"food", "vegetable", "fruit", "meat", "dairy", "bread_food", "cooked_food", "pourable"}) and not content_is_liquid(obj, **kwargs)
    return _predicate_value("content_is_solid", default, **kwargs)


def liquid_transfer_event(obj: Any = None, support: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value(
        "liquid_transfer_event",
        containment_transfer_event(obj, support, fixture, **kwargs) and content_is_liquid(obj, **kwargs),
        **kwargs,
    )


def solid_transfer_event(obj: Any = None, support: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value(
        "solid_transfer_event",
        containment_transfer_event(obj, support, fixture, **kwargs) and content_is_solid(obj, **kwargs),
        **kwargs,
    )


def source_emptied_if_receptacle(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("source_emptied_if_receptacle", True, **kwargs)


def content_is_supported(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("content_is_supported", False, **kwargs)


def no_content_elsewhere(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("no_content_elsewhere", False, **kwargs)


def content_stable(obj: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(obj, kwargs)
    return _predicate_value("content_stable", False, **kwargs)


def content_settled(obj: Any = None, support: Any = None, **kwargs) -> bool:
    return _predicate_value(
        "content_settled",
        source_emptied_if_receptacle(obj, **kwargs)
        and content_is_supported(obj, support, **kwargs)
        and no_content_elsewhere(obj, support, **kwargs)
        and content_stable(obj, **kwargs),
        **kwargs,
    )


def liquid_settled(obj: Any = None, support: Any = None, **kwargs) -> bool:
    return _predicate_value(
        "liquid_settled",
        content_is_liquid(obj, **kwargs) and content_settled(obj, support, **kwargs),
        **kwargs,
    )


def solid_settled(obj: Any = None, support: Any = None, **kwargs) -> bool:
    return _predicate_value(
        "solid_settled",
        content_is_solid(obj, **kwargs) and content_settled(obj, support, **kwargs),
        **kwargs,
    )


def solid_misplacement(obj: Any = None, support: Any = None, **kwargs) -> bool:
    return _predicate_value(
        "solid_misplacement",
        content_is_solid(obj, **kwargs) and not no_content_elsewhere(obj, support, **kwargs),
        **kwargs,
    )


def misplaced_solid_removed(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("misplaced_solid_removed", no_content_elsewhere(obj, support, **kwargs), **kwargs)


def misplaced_solid_recollected(obj: Any = None, support: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(support, kwargs))
    return _predicate_value("misplaced_solid_recollected", content_is_supported(obj, support, **kwargs), **kwargs)


# ---------------------------------------------------------------------------
# Access/enclosure safety: openable fixture interiors
# ---------------------------------------------------------------------------

def one_object_in_microwave(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("one_object_in_microwave", False, **kwargs)


def two_or_more_objects_in_microwave(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("two_or_more_objects_in_microwave", False, **kwargs)


def microwave_empty(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("microwave_empty", True, **kwargs)


def gripper_in_fixture(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("gripper_in_fixture", False, **kwargs)


def reach_in_fixture(fixture: Any = None, **kwargs) -> bool:
    _ = _resolve_entity(fixture, kwargs)
    return _predicate_value("reach_in_fixture", False, **kwargs)


def object_in_fixture(obj: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("object_in_fixture", False, **kwargs)


def object_reach_in_fixture(obj: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("object_reach_in_fixture", False, **kwargs)


def object_in_same_fixture(obj: Any = None, fixture: Any = None, **kwargs) -> bool:
    _ = (_resolve_entity(obj, kwargs), _resolve_entity(fixture, kwargs))
    return _predicate_value("object_in_same_fixture", False, **kwargs)
