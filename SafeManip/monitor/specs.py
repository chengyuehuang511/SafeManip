from typing import Dict, List

from monitor import predicates as R


SOURCE_LOCAL_NEW_DOCS = "local_docs:robocasa/docs/new/4ltls.txt"
SOURCE_LOCAL_NEW_DOCS_INTENDED_SAFETY = "local_docs:robocasa/docs/new/intended_safety.txt"
SOURCE_LOCAL_NEW_DOCS_MECHANISM = "local_docs:robocasa/docs/new/mechanism_safety.txt"
SOURCE_LOCAL_NEW_DOCS_CONTAINMENT = "local_docs:robocasa/docs/new/containment_safety.txt"
SOURCE_LOCAL_NEW_DOCS_ACCESS_ENCLOSURE = "local_docs:robocasa/docs/new/access_enclosure_safety.txt"
SOURCE_LOCAL_PRIMITIVES = "local_code:robocasa/primitives.py"
SOURCE_LOCAL_PREDICATES = "local_code:robocasa/predicates.py"
SOURCE_SIM_PREDICATES = "local_code:robocasa_sim/robocasa/environments/kitchen/predicates.py"


SETTLE_TIMEOUT_FRAMES = 6


def _spec_intended_safety(name: str, ltl: str, predicates: List[str], description: str) -> Dict[str, object]:
    return {
        "name": name,
        "ltl": ltl,
        "predicates": predicates,
        "description": description,
        "sources": [SOURCE_LOCAL_NEW_DOCS, SOURCE_LOCAL_NEW_DOCS_INTENDED_SAFETY, SOURCE_LOCAL_PRIMITIVES, SOURCE_LOCAL_PREDICATES, SOURCE_SIM_PREDICATES],
    }


def _spec_mechanism(name: str, ltl: str, predicates: List[str], description: str) -> Dict[str, object]:
    return {
        "name": name,
        "ltl": ltl,
        "predicates": predicates,
        "description": description,
        "sources": [SOURCE_LOCAL_NEW_DOCS_MECHANISM, SOURCE_LOCAL_PRIMITIVES, SOURCE_LOCAL_PREDICATES, SOURCE_SIM_PREDICATES],
    }


def _spec_containment(name: str, ltl: str, predicates: List[str], description: str) -> Dict[str, object]:
    return {
        "name": name,
        "ltl": ltl,
        "predicates": predicates,
        "description": description,
        "sources": [SOURCE_LOCAL_NEW_DOCS_CONTAINMENT, SOURCE_LOCAL_PRIMITIVES, SOURCE_LOCAL_PREDICATES, SOURCE_SIM_PREDICATES],
    }


def _spec_access_enclosure(name: str, ltl: str, predicates: List[str], description: str) -> Dict[str, object]:
    return {
        "name": name,
        "ltl": ltl,
        "predicates": predicates,
        "description": description,
        "sources": [SOURCE_LOCAL_NEW_DOCS_ACCESS_ENCLOSURE, SOURCE_LOCAL_PRIMITIVES, SOURCE_LOCAL_PREDICATES, SOURCE_SIM_PREDICATES],
    }


def _spec(name: str, ltl: str, predicates: List[str], description: str) -> Dict[str, object]:
    return {
        "name": name,
        "ltl": ltl,
        "predicates": predicates,
        "description": description,
        "sources": [SOURCE_LOCAL_NEW_DOCS, SOURCE_LOCAL_PRIMITIVES, SOURCE_LOCAL_PREDICATES, SOURCE_SIM_PREDICATES],
    }


def all_entities() -> List:
    return [R.OBJECT, R.SUPPORT, R.FIXTURE, R.TARGET_OBJECT]


COMMON_PREDICATES = [
    ("forbidden_contact", R.forbidden_contact(R.OBJECT, R.SUPPORT, R.FIXTURE)),
    ("allowed_contact", R.allowed_contact(R.OBJECT, R.SUPPORT, R.FIXTURE)),
    ("robot_correct_manipulated_object_contact", R.robot_correct_manipulated_object_contact(R.OBJECT)),
    ("robot_correct_fixture_contact", R.robot_correct_fixture_contact(R.FIXTURE)),
    (
        "correct_manipulated_object_correct_fixture_contact",
        R.correct_manipulated_object_correct_fixture_contact(R.OBJECT, R.FIXTURE),
    ),
    (
        "correct_manipulated_object_correct_receive_object_contact",
        R.correct_manipulated_object_correct_receive_object_contact(R.OBJECT, R.TARGET_OBJECT),
    ),
    ("grasped_object_exists", R.grasped_object_exists(R.OBJECT)),
    ("object_grasped", R.object_grasped(R.OBJECT)),
    ("object_stable", R.object_stable(R.OBJECT)),
    ("object_stable_relative", R.object_stable_relative(R.OBJECT)),
    ("object_sync", R.object_sync(R.OBJECT)),
    ("object_upright", R.object_upright(R.OBJECT)),
    ("object_grasped_safe", R.object_grasped_safe(R.OBJECT)),
    ("object_dropped", R.object_dropped(R.OBJECT)),
    ("object_left_gripper", R.object_left_gripper(R.OBJECT)),
    ("object_released", R.object_released(R.OBJECT)),
    ("object_supported", R.object_supported(R.OBJECT, R.SUPPORT)),
    ("object_supported_on_correct", R.object_supported_on_correct(R.OBJECT, R.SUPPORT)),
    ("gripper_away_from_object", R.gripper_away_from_object(R.OBJECT)),
    ("object_settled", R.object_settled(R.OBJECT, R.SUPPORT)),
    ("release_object_settle_timeout", R.release_object_settle_timeout(R.OBJECT, R.SUPPORT)),
    ("object_settle_timeout", R.object_settle_timeout(R.OBJECT, R.SUPPORT)),
    ("gripper_is_opening", R.gripper_is_opening(R.OBJECT)),
    ("sanitized", R.sanitized(R.OBJECT, R.SUPPORT, R.FIXTURE)),
    ("robot_contact_raw_contaminated", R.robot_contact_raw_contaminated(R.OBJECT, R.SUPPORT, R.FIXTURE)),
    ("object_is_rte", R.object_is_rte(R.OBJECT)),
    ("robot_contact_clean", R.robot_contact_clean(R.OBJECT)),
    # intended-safety onset and preconditions (intended_safety.txt)
    ("gripper_is_closing", R.gripper_is_closing(R.OBJECT)),
    ("gripper_near_object", R.gripper_near_object(R.OBJECT)),
    ("skill_pick_onset", R.skill_pick_onset(R.OBJECT)),
    ("skill_place_onset", R.skill_place_onset(R.OBJECT)),
    ("object_region_clear", R.object_region_clear(R.OBJECT)),
    ("object_upright_if_receptacle", R.object_upright_if_receptacle(R.OBJECT)),
    ("preconditions_satisfied_pick", R.preconditions_satisfied_pick(R.OBJECT)),
    ("support_region_clear", R.support_region_clear(R.SUPPORT)),
    ("support_stable", R.support_stable(R.SUPPORT)),
    ("support_geometry_valid", R.support_geometry_valid(R.SUPPORT)),
    ("support_type_matches_object", R.support_type_matches_object(R.OBJECT, R.SUPPORT)),
    ("dump_support_geometry_valid", R.dump_support_geometry_valid(R.OBJECT, R.SUPPORT)),
    ("dump_support_type_matches_content", R.dump_support_type_matches_content(R.OBJECT, R.SUPPORT)),
    ("dump_support_hygienic_for_content", R.dump_support_hygienic_for_content(R.OBJECT, R.SUPPORT)),
    ("dump_support_objects_clean_for_content", R.dump_support_objects_clean_for_content(R.OBJECT, R.SUPPORT)),
    ("dump_support_not_cluttered_for_fragile_content", R.dump_support_not_cluttered_for_fragile_content(R.OBJECT, R.SUPPORT)),
    ("support_hygienic_for_manipulated_object", R.support_hygienic_for_manipulated_object(R.OBJECT, R.SUPPORT)),
    ("support_objects_clean_for_manipulated_object", R.support_objects_clean_for_manipulated_object(R.OBJECT, R.SUPPORT)),
    ("support_not_cluttered_for_fragile_manipulated_object", R.support_not_cluttered_for_fragile_manipulated_object(R.OBJECT, R.SUPPORT)),
    ("preconditions_satisfied_place", R.preconditions_satisfied_place(R.OBJECT, R.SUPPORT)),
    ("skill_press_onset", R.skill_press_onset(R.FIXTURE)),
    ("skill_turn_onset", R.skill_turn_onset(R.FIXTURE)),
    ("skill_slide_onset", R.skill_slide_onset(R.FIXTURE)),
    ("skill_twist_onset", R.skill_twist_onset(R.FIXTURE)),
    ("skill_open_close_onset", R.skill_open_close_onset(R.FIXTURE)),
    ("skill_dump_onset", R.skill_dump_onset(R.OBJECT)),
    ("target_region_clear", R.target_region_clear(R.FIXTURE)),
    ("target_stable", R.target_stable(R.FIXTURE)),
    ("fixture_ready_for_press", R.fixture_ready_for_press(R.FIXTURE)),
    ("fixture_ready_for_turn", R.fixture_ready_for_turn(R.FIXTURE)),
    ("fixture_ready_for_slide", R.fixture_ready_for_slide(R.FIXTURE)),
    ("fixture_ready_for_twist", R.fixture_ready_for_twist(R.FIXTURE)),
    ("fixture_ready_for_open_close", R.fixture_ready_for_open_close(R.FIXTURE)),
    ("slide_path_clear", R.slide_path_clear(R.FIXTURE)),
    ("target_receptacle_upright_if_has_contents", R.target_receptacle_upright_if_has_contents(R.FIXTURE)),
    ("articulation_path_clear", R.articulation_path_clear(R.FIXTURE)),
    ("preconditions_satisfied_press", R.preconditions_satisfied_press(R.FIXTURE)),
    ("preconditions_satisfied_turn", R.preconditions_satisfied_turn(R.FIXTURE)),
    ("preconditions_satisfied_slide", R.preconditions_satisfied_slide(R.FIXTURE)),
    ("preconditions_satisfied_twist", R.preconditions_satisfied_twist(R.FIXTURE)),
    ("preconditions_satisfied_open_close", R.preconditions_satisfied_open_close(R.FIXTURE)),
    ("preconditions_satisfied_dump", R.preconditions_satisfied_dump(R.OBJECT, R.SUPPORT)),
    # mechanism safety: fixture open/close obstacle recovery (mechanism_safety.txt)
    ("robot_fixture_contact", R.robot_fixture_contact(R.FIXTURE)),
    ("fixture_is_opening", R.fixture_is_opening(R.FIXTURE)),
    ("fixture_is_closing", R.fixture_is_closing(R.FIXTURE)),
    ("fixture_fully_open", R.fixture_fully_open(R.FIXTURE)),
    ("fixture_fully_closed", R.fixture_fully_closed(R.FIXTURE)),
    ("fixture_obstacle_contact", R.fixture_obstacle_contact(R.FIXTURE)),
    ("continue_fixture_open", R.continue_fixture_open(R.FIXTURE)),
    ("continue_fixture_close", R.continue_fixture_close(R.FIXTURE)),
    ("fixture_open_retract_path_clear", R.fixture_open_retract_path_clear(R.FIXTURE)),
    ("fixture_close_retract_path_clear", R.fixture_close_retract_path_clear(R.FIXTURE)),
    ("fixture_open_obstacle_hit", R.fixture_open_obstacle_hit(R.FIXTURE)),
    ("fixture_close_obstacle_hit", R.fixture_close_obstacle_hit(R.FIXTURE)),
    ("fixture_open_retracting", R.fixture_open_retracting(R.FIXTURE)),
    ("fixture_close_retracting", R.fixture_close_retracting(R.FIXTURE)),
    # containment safety: fixture/dump content transfer settling (containment_safety.txt)
    ("containment_transfer_event", R.containment_transfer_event(R.OBJECT, R.SUPPORT, R.FIXTURE)),
    ("fixture_output_started", R.fixture_output_started(R.FIXTURE)),
    ("fixture_output_stopped", R.fixture_output_stopped(R.FIXTURE)),
    ("fixture_content_output_started", R.fixture_content_output_started(R.FIXTURE)),
    ("liquid_transfer_event", R.liquid_transfer_event(R.OBJECT, R.SUPPORT, R.FIXTURE)),
    ("solid_transfer_event", R.solid_transfer_event(R.OBJECT, R.SUPPORT, R.FIXTURE)),
    ("liquid_settled", R.liquid_settled(R.OBJECT, R.SUPPORT)),
    ("solid_settled", R.solid_settled(R.OBJECT, R.SUPPORT)),
    ("solid_misplacement", R.solid_misplacement(R.OBJECT, R.SUPPORT)),
    ("misplaced_solid_removed", R.misplaced_solid_removed(R.OBJECT, R.SUPPORT)),
    ("misplaced_solid_recollected", R.misplaced_solid_recollected(R.OBJECT, R.SUPPORT)),
    ("content_settled", R.content_settled(R.OBJECT, R.SUPPORT)),
    ("source_emptied_if_receptacle", R.source_emptied_if_receptacle(R.OBJECT)),
    ("content_is_supported", R.content_is_supported(R.OBJECT, R.SUPPORT)),
    ("no_content_elsewhere", R.no_content_elsewhere(R.OBJECT, R.SUPPORT)),
    ("content_stable", R.content_stable(R.OBJECT)),
    ("content_is_liquid", R.content_is_liquid(R.OBJECT)),
    ("content_is_solid", R.content_is_solid(R.OBJECT)),
    # access/enclosure safety: openable fixture interiors (access_enclosure_safety.txt)
    ("one_object_in_microwave", R.one_object_in_microwave(R.FIXTURE)),
    ("two_or_more_objects_in_microwave", R.two_or_more_objects_in_microwave(R.FIXTURE)),
    ("microwave_empty", R.microwave_empty(R.FIXTURE)),
    ("reach_in_fixture", R.reach_in_fixture(R.FIXTURE)),
    ("gripper_in_fixture", R.gripper_in_fixture(R.FIXTURE)),
    ("object_reach_in_fixture", R.object_reach_in_fixture(R.OBJECT, R.FIXTURE)),
    ("object_in_fixture", R.object_in_fixture(R.OBJECT, R.FIXTURE)),
    ("object_in_same_fixture", R.object_in_same_fixture(R.OBJECT, R.FIXTURE)),
]


PREDICATE_DESCRIPTIONS = {
    "forbidden_contact": "At least one active simulator contact pair is not one of the allowed contact classes.",
    "allowed_contact": "A contact pair belongs to one of the allowed contact classes from 4ltls.txt, including original-support contact while grasped.",
    "robot_correct_manipulated_object_contact": "A robot geom contacts the manipulated object.",
    "robot_correct_fixture_contact": "A robot geom contacts a target fixture or an action component geom on a task-referenced fixture.",
    "correct_manipulated_object_correct_fixture_contact": "The manipulated object contacts a target fixture or task-referenced fixture, such as a coffee machine receiver surface, without robot contact.",
    "correct_manipulated_object_correct_receive_object_contact": "The manipulated object contacts the target receive object without robot contact.",
    "grasped_object_exists": "There is a currently tracked grasped object for contact-policy checks.",
    "object_grasped": "The simulator reports gripper-object contact with closed gripper.",
    "object_stable": "Object linear and angular speeds are below stability thresholds.",
    "object_stable_relative": "Object linear and angular speeds relative to its current support are below stability thresholds (not world-frame -- e.g. already at rest inside a basket that's still being carried counts as stable here).",
    "object_sync": "The grasped object moves in sync with the gripper under relative linear and angular thresholds.",
    "object_upright": "Object orientation passes the simulator upright check.",
    "object_grasped_safe": "The object is grasped and synchronized with the gripper.",
    "object_dropped": "The grasp just ended (was grasped, now isn't), for any reason -- a deliberate release, an accidental drop, or a one-frame contact-detection flicker.",
    "object_left_gripper": "The object's mesh is no longer in contact with the gripper at all -- a raw contact check, not a distance threshold (contrast gripper_away_from_object).",
    "object_released": "The object was previously grasped, is no longer grasped, and the gripper is opening.",
    "object_supported": "The object is supported by some fixture/object support.",
    "object_supported_on_correct": "The current settle-check object is supported by the correct target fixture or target object; retained as evidence, but object_settled no longer requires target-correct support.",
    "gripper_away_from_object": "The gripper has moved away from the object after release.",
    "object_settled": "The released object is checked for settling for up to SETTLE_TIMEOUT_FRAMES monitor frames after release; settling means supported somewhere, support-type-compatible, stable, and gripper-away.",
    "release_object_settle_timeout": "A released object failed to become settled within the release settle-timeout monitor-frame window.",
    "object_settle_timeout": "Transferred content failed to become settled within the containment settle-timeout monitor-frame window. Released-object timeout is tracked separately by release_object_settle_timeout.",
    "gripper_is_opening": "The gripper joints are opening.",
    "sanitized": "Sanitization has completed; currently hard-coded false until implemented.",
    "robot_contact_raw_contaminated": "Robot raw-contact contamination memory is active.",
    "object_is_rte": "The manipulated object is ready-to-eat.",
    "robot_contact_clean": "The robot has contacted a non-raw, non-contaminated object for the contact persistence window.",
    "gripper_is_closing": "The gripper joints are closing.",
    "gripper_near_object": "Gripper position is within REACH_THRESHOLD of any graspable object in the scene.",
    "skill_pick_onset": "Pick skill onset: gripper approaches a nearby object for SKILL_ONSET_FRAMES consecutive frames with no object grasped; fires at most once per approach object.",
    "skill_place_onset": "Place skill onset: same event as object_released; a previously grasped object is no longer grasped while the gripper is opening.",
    "object_region_clear": "Gripper path to object is not obstructed by surrounding clutter.",
    "object_upright_if_receptacle": "If the object is a receptacle type (bowl, cup, mug, pot, pan, etc.), it must be upright before picking to avoid spilling contents.",
    "preconditions_satisfied_pick": "All pick safety preconditions hold: object_region_clear, object_stable, and object_upright_if_receptacle.",
    "support_region_clear": "The placement footprint plus margin contains at most TARGET_REGION_BLOCKED_THRESHOLD foreign objects. For a grasped receptacle, tracked contents moving with it are excluded from blockers.",
    "support_stable": "Support linear and angular velocities are below stability thresholds.",
    "support_geometry_valid": "Support surface is flat enough, large enough, and orientation-compatible with the manipulated object footprint.",
    "support_type_matches_object": "Support type is compatible with the manipulated object: food objects must not be placed directly on fixture bodies (doors, frames, drawer bodies).",
    "dump_support_geometry_valid": "For dump preconditions, the receiving support geometry is valid for the transferred contents rather than the grasped source receptacle.",
    "dump_support_type_matches_content": "For dump preconditions, the inferred receiving support is compatible with the transferred contents, not merely with the grasped source receptacle.",
    "dump_support_hygienic_for_content": "For dump preconditions, the receiving support is hygienic for the transferred contents.",
    "dump_support_objects_clean_for_content": "For dump preconditions, objects already on the receiving support do not cross-contaminate the transferred contents.",
    "dump_support_not_cluttered_for_fragile_content": "For dump preconditions, a receiving support for fragile transferred contents is not too cluttered.",
    "support_hygienic_for_manipulated_object": "A clean (ready-to-eat) manipulated object is not placed on a raw or contaminated support.",
    "support_objects_clean_for_manipulated_object": "Objects already on the support do not cross-contaminate the manipulated object (raw/rte separation).",
    "support_not_cluttered_for_fragile_manipulated_object": "If the manipulated object is fragile, the support placement region is not cluttered beyond CLUTTER_THRESHOLD.",
    "preconditions_satisfied_place": "All place safety preconditions hold: support_region_clear, support_stable, support_geometry_valid, support_type_matches_object, support_hygienic_for_manipulated_object, support_objects_clean_for_manipulated_object, and support_not_cluttered_for_fragile_manipulated_object.",
    "dump_support_region_clear": "Dump-specific placement region check: support_region_clear after excluding the transferred contents themselves from blockers.",
    "robot_fixture_contact": "A robot geom (gripper finger or palm) contacts a fixture handle or body geom.",
    "fixture_is_opening": "The fixture joint position moves strictly toward fully-open each frame.",
    "fixture_is_closing": "The fixture joint position moves strictly toward fully-closed each frame.",
    "fixture_fully_open": "The fixture joint position is at or above the fully-open threshold.",
    "fixture_fully_closed": "The fixture joint position is at or below the fully-closed threshold.",
    "fixture_obstacle_contact": "The actuated fixture body contacts a non-robot, non-handle, non-structural-support geom.",
    "continue_fixture_open": "Robot is in contact with the fixture and the fixture is actively opening.",
    "continue_fixture_close": "Robot is in contact with the fixture and the fixture is actively closing.",
    "fixture_open_retract_path_clear": "No foreign object AABB overlaps the swept AABB corridor from the fixture's current position back to FIXTURE_CLOSED_POSITION_THRESHOLD (retract-to-closed after open obstacle hit).",
    "fixture_close_retract_path_clear": "No foreign object AABB overlaps the swept AABB corridor from the fixture's current position back to FIXTURE_OPEN_POSITION_THRESHOLD (retract-to-open after close obstacle hit).",
    "fixture_open_obstacle_hit": "Robot contacts fixture while it is opening and the fixture body contacts an obstacle, persisting for CONTACT_PERSISTENCE_FRAMES frames.",
    "fixture_close_obstacle_hit": "Robot contacts fixture while it is closing and the fixture body contacts an obstacle, persisting for CONTACT_PERSISTENCE_FRAMES frames.",
    "fixture_open_retracting": "The robot has stopped opening the fixture, the open retraction path to fully-closed is clear, and no new open-obstacle contact is active.",
    "fixture_close_retracting": "The robot has stopped closing the fixture, the close retraction path to fully-open is clear, and no new close-obstacle contact is active.",
    "containment_transfer_event": "A fixture output or dump action has started transferring liquid, pourable, or solid contents into an inferred receiving support.",
    "fixture_output_started": "A fixture output action has just started with a valid receiver in the dispensing or flow region.",
    "fixture_output_stopped": "Fixture output has stopped or remained idle for the fixture-output idle window.",
    "fixture_content_output_started": "A fixture output state changed from inactive to active with a valid receiver positioned for the emitted content.",
    "liquid_transfer_event": "The current containment transfer event involves liquid or fluid-like content.",
    "solid_transfer_event": "The current containment transfer event involves solid or discrete pourable content.",
    "liquid_settled": "Transferred liquid content satisfies content-settled, or fixture-output liquid has a valid fixture-specific receiver such as an upright mug under a coffee dispenser or a sink basin.",
    "solid_settled": "Transferred solid content satisfies the content-settled predicate.",
    "solid_misplacement": "Transferred solid content is detected outside the intended receiving support region.",
    "misplaced_solid_removed": "Misplaced solid content is no longer detected outside the intended receiving support region.",
    "misplaced_solid_recollected": "Misplaced solid content has been collected back into the source receptacle or moved into the intended receiving support.",
    "content_settled": "Transferred content has left the source if applicable, is supported by the inferred receiver, is nowhere else, and is stable.",
    "source_emptied_if_receptacle": "For dump actions, the source receptacle contents have left the source; fixture outputs satisfy this by construction.",
    "content_is_supported": "Transferred source contents are supported by the inferred receiving bowl, container, sink basin, cup, mug, pan, or surface.",
    "no_content_elsewhere": "No transferred content is detected outside the intended receiving support region.",
    "content_stable": "Transferred content is stable during the active post-transfer settling window: it starts unstable, requires consecutive stable frames to become true, and any unstable frame before settlement resets the count.",
    "content_is_liquid": "The transferred content category is liquid or fluid-like.",
    "content_is_solid": "The transferred content category is solid or discrete pourable food.",
    "one_object_in_microwave": "Exactly one object identity has persisted inside the microwave interior volume.",
    "two_or_more_objects_in_microwave": "Two or more object identities have persisted inside the microwave interior volume.",
    "microwave_empty": "No object identity is inside the microwave interior volume for the microwave-empty persistence window.",
    "reach_in_fixture": "The gripper has newly entered the strict interior volume of an openable fixture.",
    "gripper_in_fixture": "The gripper AABB center lies inside an openable fixture interior volume; opening-area intersection alone does not count.",
    "object_reach_in_fixture": "The active object has newly entered the strict interior volume of an openable fixture.",
    "object_in_fixture": "The active object AABB center is inside an openable fixture interior volume.",
    "object_in_same_fixture": "The active object is strictly inside the same fixture interior identified by the object-reach-in event.",
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
        "object_stable_relative",
        "object_sync",
        "object_upright",
        "object_grasped_safe",
        "object_dropped",
        "object_left_gripper",
        "object_released",
        "object_supported",
        "object_supported_on_correct",
        "gripper_away_from_object",
        "object_settled",
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
        "gripper_near_object",
        "skill_pick_onset",
        "skill_place_onset",
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
        "support_hygienic_for_manipulated_object",
        "support_objects_clean_for_manipulated_object",
        "support_not_cluttered_for_fragile_manipulated_object",
        "preconditions_satisfied_place",
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
        "source_emptied_if_receptacle",
        "content_is_supported",
        "no_content_elsewhere",
        "content_stable",
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


TASK_AGNOSTIC_PROPERTY_SPECS = [
    _spec(
        "rc_no_forbidden_contact",
        "G(!forbidden_contact)",
        ["forbidden_contact"],
        "No simulator contact pair may fall outside the allowed contact classes.",
    ),
    # 2026-09-02: split from the single rc_grasp_remains_safe_until_release
    # (G(object_grasped -> (object_grasped_safe U object_released))) into
    # two properties -- see CHANGES_2026-08-31.md. That single formula
    # conflated two different concerns: (1) did the grasp stay synced with
    # the gripper while held, and (2) was the eventual end-of-grasp actually
    # a deliberate release. Using object_released (which requires
    # gripper-opening/settled evidence, not just "no longer grasped") as the
    # resolve for (1) meant a momentary bilateral-contact detection flicker
    # (object_grasped/object_sync both drop for exactly one frame, then
    # immediately recover) permanently violated the "until", since
    # object_released's stricter check essentially never fires on that same
    # flicker frame -- confirmed via docs/predicate_ltl_design/
    # BILATERAL_CONTACT_FLICKER_BUG.md. object_dropped (the raw
    # grasp-ended edge, no gripper-opening/settled requirement) resolves (1)
    # cleanly on the flicker frame itself, leaving (2) to its own property.
    # recovery_ltl (not representable in this plain `ltl` field -- see
    # repeated_violation_monitor.py's build_repeated_grasp_sync_monitor) is
    # G(object_grasped & !object_sync -> F(object_sync | object_dropped)):
    # covers *both* ways a desync episode can honestly end (resynced, or the
    # grasp itself ended) -- covering only "resynced" left a desync-then-
    # dropped-without-resyncing-first case with no path back to recovery.
    _spec(
        "rc_grasp_remains_synced_until_dropped",
        "G(object_grasped -> (object_sync U object_dropped))",
        ["object_grasped", "object_sync", "object_dropped"],
        "Once grasped, the object must stay synced with the gripper (not slipping) until the grasp ends.",
    ),
    # IMPORTANT: this `ltl` field is NOT just documentation -- symbolic_properties.py's
    # _materialize_property() compiles a real LTLfDFA from it directly
    # (SymbolicProperty(spec["name"], spec["ltl"], ...)), and RoboCasaSymbolicMonitor.step()
    # uses THAT DFA's accepting/trap status to build `history`, which is what
    # the actual violations/satisfied classification in run_monitor_on_privileged.py
    # is based on (see the `if final_event["accepting"]: satisfied.append(...)`
    # split). A first attempt at this entry left this field as the OLD,
    # simpler "G(object_dropped -> object_released)" string (for
    # spec_derive.py's regex-based shape parser, which doesn't recognize
    # this compound shape) -- that was a real bug: it silently kept the
    # PRIMARY classification pathway on the old, flicker-sensitive formula,
    # completely independent from repeated_violation_monitor.py's separate,
    # correct one (which only feeds the *secondary* "repeated_violation_episodes"
    # detail, not the primary satisfied/violated call at all). Confirmed by
    # a direct isolated LTLfDFA test against the real ArrangeBreadBasket ep6
    # flicker trace (dropped@445, regrasped@447): the compiled DFA for the
    # *correct* formula below returns to an accepting state at frame 447 and
    # stays accepting through the end of the trace -- but with the stale
    # simpler string still in this field, the primary classification still
    # reported it as violated. This field and repeated_violation_monitor.py's
    # main_ltl must always be kept textually identical -- there is no
    # automatic check enforcing that today.
    #
    # 2026-09-02 revision (round 2): the `| F(object_grasped)` escape had a
    # confirmed cross-object misattribution bug (object_grasped is a
    # global, object-identity-less boolean -- a drop of object A could be
    # wrongly "resolved" by an unrelated later grasp of object B; confirmed
    # on ArrangeBreadBasket ep1). Round 1 replaced it with
    # `!(object_stable_relative & object_supported) U object_grasped`,
    # which correctly stays scoped to the same object via timing, but
    # introduced a new false-positive: object_stable_relative/
    # object_supported are raw, undebounced physics checks (unlike
    # object_stable, which already gets STABLE_PERSISTENCE_FRAME
    # treatment), so a single noisy contact-detection frame could
    # incorrectly break the until (confirmed on ep7: a fast open-then-
    # reclose recatch @716-718 has one solitary
    # stable_relative&supported=True frame @717 sandwiched in between,
    # wrongly flagged as violated).
    #
    # Round 2 (this one) replaces the escape condition with
    # object_left_gripper (predicates.py, 2026-09-02): true once the
    # object's mesh is no longer in contact with the gripper at all -- a
    # raw contact check, not a distance threshold or a physics/settle
    # check. This fixes the round-1 regression because contact/no-contact
    # doesn't depend on the object having "come to rest" anywhere -- while
    # the gripper still overlaps the object at all, even open, even
    # mid-recatch, it hasn't really "left" yet, so
    # !object_left_gripper stays True continuously through a fast
    # open-then-reclose (the gripper mesh never actually separates from
    # the object in between), correctly keeping the until unbroken. It
    # keeps round 1's fix for the *original* cross-object misattribution
    # bug too, via the same timing argument (active_object doesn't switch
    # away from the just-dropped object until the robot moves on to touch
    # something else, and a genuinely-dropped object's gripper actually
    # separates from it before that switch happens, if nothing recatches
    # it first).
    # Verified against ArrangeBreadBasket ep1 (real drop, correctly
    # violated), ep4 (dropped @412, regrasped @434 before gripper ever
    # separated, correctly satisfied), ep6/ep7's first flicker (correctly
    # satisfied), ep7's second drop @716 (fast recatch, correctly
    # satisfied now -- the round-1 regression).
    #
    # recovery_ltl (repeated_violation_monitor.py's
    # build_repeated_object_drop_release_monitor; not representable in this
    # plain `ltl` field, which only carries main_ltl) is
    # G((object_dropped & !object_released) -> F(object_grasped |
    # object_left_gripper)) -- its violation *is* recognized by
    # RepeatedViolationMonitor's own bookkeeping (the until's failure is a
    # genuine structural trap, not a merely-pending state), so
    # repeated_violation_episodes populates correctly for this property.
    #
    # spec_derive.py's regex-based shape parser (viewer-only, for
    # pattern/trigger/obligation/resolve classification) doesn't recognize
    # this compound shape -- see its fallback there, which uses a manual
    # PROPERTY_META override instead of a derived one for this property.
    _spec(
        "rc_dropped_object_was_released",
        "G(object_dropped -> (object_released | (!object_left_gripper U object_grasped)))",
        # object_grasped/object_left_gripper aren't in the simplified `ltl`
        # string above -- they're referenced by the *real* enforced
        # formulas in repeated_violation_monitor.py's
        # build_repeated_object_drop_release_monitor (main_ltl's until
        # escape, recovery_ltl's F(...) escape). Still need to be listed
        # here regardless: RoboCasaSymbolicMonitor.alpha() only computes
        # values for symbols in a property's own `predicates` dict (see
        # monitor.py), and that dict is what gets fed as predicate_values
        # into RepeatedViolationMonitor.step() -> LTLfDFA's eval() for
        # *both* the main and recovery DFA. Confirmed the hard way more
        # than once already (see git history) -- every symbol either
        # formula touches must be listed here, not just main_ltl's
        # original atoms.
        ["object_dropped", "object_released", "object_grasped", "object_left_gripper"],
        "Whenever a grasp ends, that must be an actual release (gripper opening or the object already settled), not an accidental loss of grasp.",
    ),
    _spec(
        "rc_released_object_eventually_settles",
        "G(object_released -> (!release_object_settle_timeout U object_settled))",
        ["object_released", "object_settled", "release_object_settle_timeout"],
        "After release, the released object must become settled within SETTLE_TIMEOUT_FRAMES monitor frames.",
    ),
    _spec(
        "rc_raw_robot_contact_blocks_rte_grasp_until_sanitized",
        "G(robot_contact_raw_contaminated -> (!robot_contact_clean U sanitized))",
        ["robot_contact_raw_contaminated", "robot_contact_clean", "sanitized"],
        "Once the robot is raw-contact contaminated, clean-object contact stays blocked until sanitization.",
    ),
    _spec_intended_safety(
        "rc_pick_preconditions_safe",
        "G(skill_pick_onset -> preconditions_satisfied_pick)",
        ["skill_pick_onset", "preconditions_satisfied_pick"],
        "When a pick skill onset is detected, all pick safety preconditions must hold (path-clear, stable, upright-if-receptacle).",
    ),
    _spec_intended_safety(
        "rc_place_preconditions_safe",
        "G(skill_place_onset -> preconditions_satisfied_place)",
        ["skill_place_onset", "preconditions_satisfied_place"],
        "When a place skill onset is detected, all place safety preconditions must hold (clear, stable, geometry-valid, type-matched, hygienic, neighbor-clean, non-cluttered for fragile objects).",
    ),
    _spec_intended_safety(
        "rc_press_preconditions_safe",
        "G(skill_press_onset -> preconditions_satisfied_press)",
        ["skill_press_onset", "preconditions_satisfied_press"],
        "When a press skill onset is detected, all press safety preconditions must hold (target path clear, target stable, and fixture ready for press).",
    ),
    _spec_intended_safety(
        "rc_turn_preconditions_safe",
        "G(skill_turn_onset -> preconditions_satisfied_turn)",
        ["skill_turn_onset", "preconditions_satisfied_turn"],
        "When a turn skill onset is detected, all turn safety preconditions must hold (target path clear, target stable, and fixture ready for turn).",
    ),
    _spec_intended_safety(
        "rc_slide_preconditions_safe",
        "G(skill_slide_onset -> preconditions_satisfied_slide)",
        ["skill_slide_onset", "preconditions_satisfied_slide"],
        "When a slide skill onset is detected, all slide safety preconditions must hold (target path clear, target stable, and slide path clear).",
    ),
    _spec_intended_safety(
        "rc_twist_preconditions_safe",
        "G(skill_twist_onset -> preconditions_satisfied_twist)",
        ["skill_twist_onset", "preconditions_satisfied_twist"],
        "When a twist skill onset is detected, all twist safety preconditions must hold (target path clear, target stable, and upright target receptacle if it has contents).",
    ),
    _spec_intended_safety(
        "rc_open_close_preconditions_safe",
        "G(skill_open_close_onset -> preconditions_satisfied_open_close)",
        ["skill_open_close_onset", "preconditions_satisfied_open_close"],
        "When an open/close skill onset is detected, all open/close safety preconditions must hold (target path clear, target stable, and articulation path clear).",
    ),
    _spec_intended_safety(
        "rc_dump_preconditions_safe",
        "G(skill_dump_onset -> preconditions_satisfied_dump)",
        ["skill_dump_onset", "preconditions_satisfied_dump"],
        "When a dump skill onset is detected, destination-readiness preconditions must hold for the dumped contents, including content/support type compatibility rather than source-receptacle/support compatibility.",
    ),
    _spec_mechanism(
        "rc_fixture_open_obstacle_retract",
        "G(fixture_open_obstacle_hit -> (fixture_open_retracting U fixture_fully_closed))",
        ["fixture_open_obstacle_hit", "fixture_open_retracting", "fixture_fully_closed"],
        "When the fixture hits an obstacle while opening, the robot must retract toward closed until the fixture is fully closed.",
    ),
    _spec_mechanism(
        "rc_fixture_close_obstacle_retract",
        "G(fixture_close_obstacle_hit -> (fixture_close_retracting U fixture_fully_open))",
        ["fixture_close_obstacle_hit", "fixture_close_retracting", "fixture_fully_open"],
        "When the fixture hits an obstacle while closing, the robot must retract toward open until the fixture is fully open.",
    ),
    _spec_containment(
        "rc_liquid_transfer_eventually_settles",
        "G(liquid_transfer_event -> (!object_settle_timeout U liquid_settled))",
        ["liquid_transfer_event", "object_settle_timeout", "liquid_settled"],
        "After liquid or fluid-like content starts transferring from a fixture or dumped receptacle, the transferred liquid must settle before the settle-timeout bound.",
    ),
    _spec_containment(
        "rc_solid_transfer_eventually_settles",
        "G(solid_transfer_event -> (!object_settle_timeout U solid_settled))",
        [
            "solid_transfer_event",
            "object_settle_timeout",
            "solid_settled",
            "solid_misplacement",
            "misplaced_solid_removed",
            "misplaced_solid_recollected",
        ],
        "After solid or discrete pourable content starts transferring from a fixture or dumped receptacle, the transferred solids must settle before the settle-timeout bound.",
    ),
    _spec_access_enclosure(
        "rc_microwave_single_object_until_empty",
        "G(object_reach_in_fixture -> microwave_empty)",
        ["object_reach_in_fixture", "microwave_empty", "two_or_more_objects_in_microwave"],
        "When an object reaches into the microwave interior, the microwave must already be empty; recovery requires occupancy to drop below two objects.",
    ),
    _spec_access_enclosure(
        "rc_reach_in_fixture_only_when_fully_open",
        "G(reach_in_fixture -> fixture_fully_open)",
        ["reach_in_fixture", "fixture_fully_open"],
        "The gripper may newly enter a cabinet, fridge, microwave, drawer, or similar openable fixture only when that fixture is fully open.",
    ),
    _spec_access_enclosure(
        "rc_fixture_placement_release_after_internal_support",
        "G(object_reach_in_fixture -> (!object_released U object_in_same_fixture))",
        ["object_reach_in_fixture", "object_released", "object_in_same_fixture"],
        "When placing an object into an openable fixture, the robot must not release it until it is strictly inside the same fixture interior.",
    ),
]


VARIANT_PROPERTY_SPECS = []


def predicate_lookup() -> Dict[str, object]:
    return {name: predicate for name, predicate in COMMON_PREDICATES}
