"""Canonical RoboCasa predicate wrappers from ``docs/new/4ltls.txt``."""

from functools import partial

import monitor.primitives as P

from monitor.SymbolicEntity import SymbolicEntity


OBJECT = SymbolicEntity("object", base_filter=lambda node: True)
SUPPORT = SymbolicEntity("support", base_filter=lambda node: True)
FIXTURE = SymbolicEntity("fixture", base_filter=lambda node: True)
TARGET_OBJECT = SymbolicEntity("target_object", base_filter=lambda node: True)


def with_attribute(entity, attribute_name):
    return partial(P.entity_has_attribute, entity, attribute_name)


def forbidden_contact(entity=None, support=None, fixture=None):
    return partial(P.forbidden_contact, entity, support, fixture)


def allowed_contact(entity=None, support=None, fixture=None):
    return partial(P.allowed_contact, entity, support, fixture)


def robot_correct_manipulated_object_contact(entity=None):
    return partial(P.robot_correct_manipulated_object_contact, entity)


def robot_correct_fixture_contact(fixture=None):
    return partial(P.robot_correct_fixture_contact, fixture)


def correct_manipulated_object_correct_fixture_contact(entity=None, fixture=None):
    return partial(P.correct_manipulated_object_correct_fixture_contact, entity, fixture)


def correct_manipulated_object_correct_receive_object_contact(entity=None, target_object=None):
    return partial(P.correct_manipulated_object_correct_receive_object_contact, entity, target_object)


def grasped_object_exists(entity=None):
    return partial(P.grasped_object_exists, entity)


def object_grasped(entity=None):
    return partial(P.object_grasped, entity)


def object_stable(entity=None):
    return partial(P.object_stable, entity)


def object_stable_relative(entity=None):
    return partial(P.object_stable_relative, entity)


def object_sync(entity=None):
    return partial(P.object_sync, entity)


def object_upright(entity=None):
    return partial(P.object_upright, entity)


def object_grasped_safe(entity=None):
    return partial(P.object_grasped_safe, entity)


def object_dropped(entity=None):
    return partial(P.object_dropped, entity)


def object_left_gripper(entity=None):
    return partial(P.object_left_gripper, entity)


def object_released(entity=None):
    return partial(P.object_released, entity)


def object_supported(entity=None, support=None):
    return partial(P.object_supported, entity, support)


def object_supported_on_correct(entity=None, support=None):
    return partial(P.object_supported_on_correct, entity, support)


def gripper_away_from_object(entity=None):
    return partial(P.gripper_away_from_object, entity)


def gripper_is_opening(entity=None):
    return partial(P.gripper_is_opening, entity)


def object_settled(entity=None, support=None):
    return partial(P.object_settled, entity, support)


def object_settle_timeout(entity=None, support=None):
    return partial(P.object_settle_timeout, entity, support)


def release_object_settle_timeout(entity=None, support=None):
    return partial(P.release_object_settle_timeout, entity, support)


def sanitized(entity=None, support=None, fixture=None):
    return partial(P.sanitized, entity, support, fixture)


def robot_contact_raw_contaminated(entity=None, support=None, fixture=None):
    return partial(P.robot_contact_raw_contaminated, entity, support, fixture)


def object_is_rte(entity=None):
    return partial(P.object_is_rte, entity)


def robot_contact_clean(entity=None):
    return partial(P.robot_contact_clean, entity)


# ---------------------------------------------------------------------------
# Intended-safety onset and preconditions (intended_safety.txt)
# ---------------------------------------------------------------------------

def gripper_is_closing(entity=None):
    return partial(P.gripper_is_closing, entity)


def gripper_near_object(entity=None):
    return partial(P.gripper_near_object, entity)


def skill_pick_onset(entity=None):
    return partial(P.skill_pick_onset, entity)


def skill_place_onset(entity=None):
    return partial(P.skill_place_onset, entity)


def object_region_clear(entity=None):
    return partial(P.object_region_clear, entity)


def object_upright_if_receptacle(entity=None):
    return partial(P.object_upright_if_receptacle, entity)


def preconditions_satisfied_pick(entity=None):
    return partial(P.preconditions_satisfied_pick, entity)


def support_region_clear(support=None):
    return partial(P.support_region_clear, support)


def support_stable(support=None):
    return partial(P.support_stable, support)


def support_geometry_valid(support=None):
    return partial(P.support_geometry_valid, support)


def support_type_matches_object(entity=None, support=None):
    return partial(P.support_type_matches_object, entity, support)


def dump_support_geometry_valid(entity=None, support=None):
    return partial(P.dump_support_geometry_valid, entity, support)


def dump_support_type_matches_content(entity=None, support=None):
    return partial(P.dump_support_type_matches_content, entity, support)


def dump_support_hygienic_for_content(entity=None, support=None):
    return partial(P.dump_support_hygienic_for_content, entity, support)


def dump_support_objects_clean_for_content(entity=None, support=None):
    return partial(P.dump_support_objects_clean_for_content, entity, support)


def dump_support_not_cluttered_for_fragile_content(entity=None, support=None):
    return partial(P.dump_support_not_cluttered_for_fragile_content, entity, support)


def support_hygienic_for_manipulated_object(entity=None, support=None):
    return partial(P.support_hygienic_for_manipulated_object, entity, support)


def support_objects_clean_for_manipulated_object(entity=None, support=None):
    return partial(P.support_objects_clean_for_manipulated_object, entity, support)


def support_not_cluttered_for_fragile_manipulated_object(entity=None, support=None):
    return partial(P.support_not_cluttered_for_fragile_manipulated_object, entity, support)


def preconditions_satisfied_place(entity=None, support=None):
    return partial(P.preconditions_satisfied_place, entity, support)


def skill_press_onset(target=None):
    return partial(P.skill_press_onset, target)


def skill_turn_onset(target=None):
    return partial(P.skill_turn_onset, target)


def skill_slide_onset(target=None):
    return partial(P.skill_slide_onset, target)


def skill_twist_onset(target=None):
    return partial(P.skill_twist_onset, target)


def skill_open_close_onset(target=None):
    return partial(P.skill_open_close_onset, target)


def skill_dump_onset(entity=None):
    return partial(P.skill_dump_onset, entity)


def target_region_clear(target=None):
    return partial(P.target_region_clear, target)


def target_stable(target=None):
    return partial(P.target_stable, target)


def fixture_ready_for_press(target=None):
    return partial(P.fixture_ready_for_press, target)


def fixture_ready_for_turn(target=None):
    return partial(P.fixture_ready_for_turn, target)


def fixture_ready_for_slide(target=None):
    return partial(P.fixture_ready_for_slide, target)


def fixture_ready_for_twist(target=None):
    return partial(P.fixture_ready_for_twist, target)


def fixture_ready_for_open_close(target=None):
    return partial(P.fixture_ready_for_open_close, target)


def slide_path_clear(target=None):
    return partial(P.slide_path_clear, target)


def target_receptacle_upright_if_has_contents(target=None):
    return partial(P.target_receptacle_upright_if_has_contents, target)


def articulation_path_clear(target=None):
    return partial(P.articulation_path_clear, target)


def preconditions_satisfied_press(target=None):
    return partial(P.preconditions_satisfied_press, target)


def preconditions_satisfied_turn(target=None):
    return partial(P.preconditions_satisfied_turn, target)


def preconditions_satisfied_slide(target=None):
    return partial(P.preconditions_satisfied_slide, target)


def preconditions_satisfied_twist(target=None):
    return partial(P.preconditions_satisfied_twist, target)


def preconditions_satisfied_open_close(target=None):
    return partial(P.preconditions_satisfied_open_close, target)


def preconditions_satisfied_dump(entity=None, support=None):
    return partial(P.preconditions_satisfied_dump, entity, support)


# ---------------------------------------------------------------------------
# Mechanism safety: fixture open/close obstacle recovery (mechanism_safety.txt)
# ---------------------------------------------------------------------------

def robot_fixture_contact(fixture=None):
    return partial(P.robot_fixture_contact, fixture)


def fixture_is_opening(fixture=None):
    return partial(P.fixture_is_opening, fixture)


def fixture_is_closing(fixture=None):
    return partial(P.fixture_is_closing, fixture)


def fixture_fully_open(fixture=None):
    return partial(P.fixture_fully_open, fixture)


def fixture_fully_closed(fixture=None):
    return partial(P.fixture_fully_closed, fixture)


def fixture_obstacle_contact(fixture=None):
    return partial(P.fixture_obstacle_contact, fixture)


def continue_fixture_open(fixture=None):
    return partial(P.continue_fixture_open, fixture)


def continue_fixture_close(fixture=None):
    return partial(P.continue_fixture_close, fixture)


def fixture_open_retract_path_clear(fixture=None):
    return partial(P.fixture_open_retract_path_clear, fixture)


def fixture_close_retract_path_clear(fixture=None):
    return partial(P.fixture_close_retract_path_clear, fixture)


def fixture_open_obstacle_hit(fixture=None):
    return partial(P.fixture_open_obstacle_hit, fixture)


def fixture_close_obstacle_hit(fixture=None):
    return partial(P.fixture_close_obstacle_hit, fixture)


def fixture_open_retracting(fixture=None):
    return partial(P.fixture_open_retracting, fixture)


def fixture_close_retracting(fixture=None):
    return partial(P.fixture_close_retracting, fixture)


# ---------------------------------------------------------------------------
# Containment safety: fixture/dump content transfer settling
# ---------------------------------------------------------------------------

def containment_transfer_event(entity=None, support=None, fixture=None):
    return partial(P.containment_transfer_event, entity, support, fixture)


def fixture_output_started(fixture=None):
    return partial(P.fixture_output_started, fixture)


def fixture_output_stopped(fixture=None):
    return partial(P.fixture_output_stopped, fixture)


def fixture_content_output_started(fixture=None):
    return partial(P.fixture_content_output_started, fixture)


def liquid_transfer_event(entity=None, support=None, fixture=None):
    return partial(P.liquid_transfer_event, entity, support, fixture)


def solid_transfer_event(entity=None, support=None, fixture=None):
    return partial(P.solid_transfer_event, entity, support, fixture)


def liquid_settled(entity=None, support=None):
    return partial(P.liquid_settled, entity, support)


def solid_settled(entity=None, support=None):
    return partial(P.solid_settled, entity, support)


def solid_misplacement(entity=None, support=None):
    return partial(P.solid_misplacement, entity, support)


def misplaced_solid_removed(entity=None, support=None):
    return partial(P.misplaced_solid_removed, entity, support)


def misplaced_solid_recollected(entity=None, support=None):
    return partial(P.misplaced_solid_recollected, entity, support)


def content_settled(entity=None, support=None):
    return partial(P.content_settled, entity, support)


def source_emptied_if_receptacle(entity=None):
    return partial(P.source_emptied_if_receptacle, entity)


def content_is_supported(entity=None, support=None):
    return partial(P.content_is_supported, entity, support)


def no_content_elsewhere(entity=None, support=None):
    return partial(P.no_content_elsewhere, entity, support)


def content_stable(entity=None):
    return partial(P.content_stable, entity)


def content_is_liquid(entity=None):
    return partial(P.content_is_liquid, entity)


def content_is_solid(entity=None):
    return partial(P.content_is_solid, entity)


# ---------------------------------------------------------------------------
# Access/enclosure safety: openable fixture interiors
# ---------------------------------------------------------------------------

def one_object_in_microwave(fixture=None):
    return partial(P.one_object_in_microwave, fixture)


def two_or_more_objects_in_microwave(fixture=None):
    return partial(P.two_or_more_objects_in_microwave, fixture)


def microwave_empty(fixture=None):
    return partial(P.microwave_empty, fixture)


def gripper_in_fixture(fixture=None):
    return partial(P.gripper_in_fixture, fixture)


def reach_in_fixture(fixture=None):
    return partial(P.reach_in_fixture, fixture)


def object_in_fixture(entity=None, fixture=None):
    return partial(P.object_in_fixture, entity, fixture)


def object_reach_in_fixture(entity=None, fixture=None):
    return partial(P.object_reach_in_fixture, entity, fixture)


def object_in_same_fixture(entity=None, fixture=None):
    return partial(P.object_in_same_fixture, entity, fixture)
