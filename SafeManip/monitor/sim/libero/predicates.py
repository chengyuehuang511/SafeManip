"""
LIBERO simulator-side predicates -- the LIBERO analog of
`monitor/sim/robocasa/predicates.py`, computing the same *named* boolean
predicates the top-level `monitor/predicates.py`/`monitor/specs.py` (both
unmodified, simulator-agnostic) expect to find at
`dynamic_info["predicates"]["sections"]["predicates"][<name>]["value"]`.

SCOPE / HONESTY NOTE (read this before trusting a violation count): this is a
v0 first pass, analogous to where RoboCasa's own predicates.py started (see
monitor/output/CHANGELOG.md's v0 entry) before ~20 iterations of empirical
tuning against real corpora. Every one of the 20 `TASK_AGNOSTIC_PROPERTY_SPECS`
now has a real, generically-computed implementation attempt (not a
hand-stubbed default) -- but several of those mechanisms are, verified by
actually running them against this 40-task corpus, EMPIRICALLY INACTIVE here:
this corpus simply contains no fixture/object that would ever make them fire
(e.g. no faucet, no push-button, no raw/perishable food object). That
distinction -- implemented-but-empirically-inactive-for-this-corpus vs.
never-implemented -- matters and is called out explicitly below, per-family.

  REAL SIGNAL, ACTIVE FOR THIS CORPUS (physics-grounded, "v0 -- not yet
  empirically tuned" thresholds):
    - contact_policy: forbidden_contact / forbidden_contact_sustained (robot
      ARM-LINK, i.e. non-gripper, contact with any fixture/object --
      incidental arm/body collisions; does not model RoboCasa's fine-grained
      allowed-vs-forbidden contact-class taxonomy, since that requires
      per-task role assignment RoboCasa's fixture/object registry provides
      and LIBERO's BDDL objects_dict/fixtures_dict does not expose the same
      way).
    - grasp_release_settle: object_grasped (robosuite's own `_check_grasp`
      bilateral-contact check), object_stable, object_stable_relative
      (aliased to object_stable), object_sync, object_upright,
      object_dropped, object_left_gripper, object_released, object_supported,
      object_supported_on_correct (aliased to object_supported),
      gripper_away_from_object, object_settled, object_settle_timeout,
      release_object_settle_timeout, gripper_is_opening/closing.
    - skill_onset + pick_preconditions: skill_pick_onset, object_region_clear,
      object_upright_if_receptacle (2026-09-16: now actually emitted -- was
      computed but silently absent from the predicates dict before),
      preconditions_satisfied_pick (2026-09-16: composition corrected to
      `object_region_clear and focus_pick_stable` only, matching RoboCasa's
      REAL preconditions_satisfied_pick, predicates.py:4000 -- RoboCasa's own
      code never ANDs object_upright_if_receptacle into this, despite the
      generic top-level monitor/predicates.py fallback and specs.py's
      docstring text both describing it as included; this file mirrors the
      simulator override actually used at runtime, not the unused generic
      default). object_region_clear (2026-09-20): now a real gripper-to-
      target swept-path-obstruction AABB check (_object_region_blockers),
      not a proximity radius -- see that function's own docstring for why
      the old radius heuristic was a genuine architectural gap versus
      RoboCasa (it flagged an ordinary second item placed into a basket that
      already contains a first item as a violation, purely from the first
      item's proximity to a plausible drop point).
    - place_preconditions: skill_place_onset (== object_released),
      support_region_clear, support_stable, preconditions_satisfied_place
      (support_geometry_valid explicitly stubbed True -- see below).
      support_stable (2026-09-20 fix): now checks the live, debounced,
      relative-to-support object_stable_by_name of the inferred landing
      target when it's a movable object (e.g. a basket/bowl), matching
      RoboCasa's own sup_kind=="object" branch of _support_stable --
      previously hardcoded True unconditionally regardless of target kind.
      support_region_clear (2026-09-20, continuous-mechanism fix): now
      mirrors RoboCasa's own real behavior -- re-evaluated every single
      frame an object is being carried, sweeping from the object's CURRENT
      position to a live, continuously-re-guessed landing target
      (_infer_landing_target, this file's simplified analog of RoboCasa's
      _infer_support/_spos), via _support_region_blockers. An earlier
      same-day version instead swept once, at object_dropped time, from a
      fixed carry-origin snapshot all the way to the final landing
      position -- spanning the object's entire pickup-to-drop trip and
      wrongly flagging any unrelated object sitting anywhere near that long
      straight line (confirmed:
      LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_
      basket ep3, milk_1). See _support_region_blockers' own docstring for
      the full history (including the even-earlier one-frame-back sweep that
      preceded that fixed-origin version, and why it was also wrong).
    - access_enclosure_safety: fixture_fully_open/closed, reach_in_fixture,
      gripper_in_fixture, object_reach_in_fixture, object_in_fixture,
      object_in_same_fixture, one/two-plus-objects-in-microwave,
      microwave_empty -- built on LIBERO's own `ObjectState.check_contain`/
      `is_open`/`is_close`.
    - fixture-skill onset (all 5: press/turn/slide/twist/open_close) +
      preconditions: a simplified generic port of RoboCasa's own
      keyword+attribute tagging (`ACTION_COMPONENT_KEYWORDS`, copied verbatim
      as plain data into attributes.py) plus a proximity+persistence onset
      detector shaped exactly like skill_pick_onset (gripper near the target
      for SKILL_ONSET_FRAMES, fires once per approach) -- NOT a full port of
      RoboCasa's real `_target_candidates`/AABB-distance machinery (that also
      does per-geom-component AABB extraction and task-registered-fixture
      preference ordering this file doesn't replicate). Fixture candidates
      are tagged by keyword-name match plus a structural joint-type
      tie-break (SLIDE->slide, HINGE+has_turnon->twist, HINGE
      otherwise->open_close) for the two keyword lists ("slide"/"open_close")
      that both include "drawer". Proximity, not raw geom contact, gates
      onset: confirmed directly on a real drawer-opening episode (contact
      array inspection) that LIBERO's handle geoms frequently never register
      raw contact with the gripper during a real pull at all, even while the
      joint is visibly, continuously articulating. Of the 5, only slide
      (drawers) and open_close (microwave/cabinet doors) empirically fire for
      this corpus; press and turn (faucet-keyword-only) never find a
      candidate fixture at all (verified: this corpus's 6 fixtures --
      desk_caddy, flat_stove, microwave, white_cabinet, wine_rack,
      wooden_cabinet -- include no faucet or push-button); twist (stove
      knob, tagged structurally via HINGE+has_turnon) does fire. twist is
      deliberately NOT extended to movable objects the way RoboCasa's real
      algorithm does for bottle/jar/can caps -- LIBERO's movable objects only
      ever have a single top-level free (pose) joint, no articulated cap/lid
      sub-mechanism to actually twist, and naively keyword-matching
      "bottle"/"jar"/"can" against this corpus's object names would spuriously
      tag `wine_bottle_1` (present in KITCHEN_SCENE4, libero_10) as twistable,
      firing skill_twist_onset on every ordinary pick-up-the-wine-bottle
      approach -- confirmed this would happen, and confirmed the fixture-only
      restriction avoids it (spot-checked directly on that episode).
      target_region_clear/slide_path_clear/articulation_path_clear are always
      explicitly emitted (never left absent) -- `monitor/predicates.py`
      defaults all three to False when missing, so omitting them once the
      matching onset can fire would flip that spec from vacuously-satisfied
      to almost-always-violated, a regression not an improvement (caught
      before it shipped). fixture_ready_for_{press,turn,slide,twist,
      open_close} (2026-09-21: now real, literal ports of RoboCasa's own
      per-fixture-class content-readiness checks -- see the block of
      functions next to `_objects_at_fixture`/`_fixture_ready_for_press` for
      the full mechanism and which RoboCasa branches have no LIBERO fixture-
      class analog at all vs. which are real-but-empirically-inactive for
      this corpus) are AND'd into preconditions_satisfied_{press,turn,slide,
      twist,open_close}, exactly matching RoboCasa's own composition.
      target_receptacle_upright_if_has_contents is left absent --
      monitor/predicates.py already defaults it to True.
    - mechanism_safety: robot_fixture_contact, fixture_is_opening/closing
      (sign of an open-fraction delta, generic across slide/hinge joints, no
      per-task tagging), fixture_obstacle_contact (the fixture body contacts
      some *other* movable object, not just the robot), continue_fixture_
      open/close, fixture_open/close_obstacle_hit (persistence-debounced over
      CONTACT_PERSISTENCE_FRAMES), fixture_open/close_retracting,
      fixture_open/close_retract_timeout (too long since the obstacle hit
      without retracting starting, RETRACT_TIMEOUT_FRAMES). Empirically active on the microwave-door task in this
      corpus's smoke test -- worth flagging as noisier than the rest: the
      obstacle-hit counter can fire on a real but arguably-benign event (the
      manipulated object itself resting against the appliance interior while
      the door is still mid-motion reads as "an obstacle," even though
      nothing is actually jamming the mechanism) -- a real v0 imprecision,
      not a crash or a stub, flagged for future tuning via
      `predicate-design-cycle`. fixture_open/close_retract_path_clear are
      deliberately left ABSENT (RoboCasa's own accessor already defaults
      both to True when missing).
    - contamination: robot_contact_raw_contaminated/object_is_rte/
      robot_contact_clean computed from keyword-tag matching against LIBERO's
      own object category names (RAW_NAME_SUBSTRINGS/RTE_NAME_SUBSTRINGS in
      attributes.py, mirroring monitor/predicates.py's own hardcoded
      raw/ready_to_eat fallback sets exactly), with contamination persisting
      until a turned-on+contacted faucet-tagged fixture "sanitizes" it (see
      containment_safety below for why that never actually happens in this
      corpus). content_is_liquid/content_is_solid (LIQUID_NAME_SUBSTRINGS
      keyword match) and containment_transfer_event/fixture_output_started
      (a turned-on faucet-tagged fixture, if this task has one) are computed
      the same generic, not-hand-stubbed way.

  EMPIRICALLY INACTIVE FOR THIS 40-TASK CORPUS (implemented and actually run
  -- not assumed -- but this corpus contains no matching fixture/object, so
  the mechanism genuinely never fires; verified by inspecting the real
  fixture set {desk_caddy, flat_stove, microwave, white_cabinet, wine_rack,
  wooden_cabinet} and the real object set (soup/sauce/butter/pudding/cream
  cheese/ketchup/milk/juice/dressing/mugs/bowls/plates/moka pots/wine
  bottle/book), and all 40 language instructions):
    - skill_press_onset / rc_press_preconditions_safe -- no push-button-type
      fixture.
    - skill_turn_onset / rc_turn_preconditions_safe -- no faucet fixture.
    - sanitized / rc_raw_robot_contact_blocks_rte_grasp_until_sanitized's
      recovery path -- no faucet fixture to sanitize at, though its
      trigger (robot_contact_raw_contaminated) is also independently never
      True here since no object is raw-tagged, so this spec is vacuously
      satisfied for a doubly-confirmed reason, not a single assumption.
    - containment_transfer_event/fixture_output_started and therefore
      rc_liquid_transfer_eventually_settles/
      rc_solid_transfer_eventually_settles -- no faucet fixture and no
      dump/pour task.

  EXPLICITLY STUBBED TRUE (documented simplification, not modeled in v0):
    - target_stable (all 5 fixture-skill families): LIBERO fixture root
      bodies don't translate in this corpus -- only their door/drawer/knob
      joints articulate -- so root-body position stability holds by
      construction, not by measurement.

  2026-09-21 (fixture-readiness/support-type/contact-role audit): support_
  geometry_valid and support_type_matches_object are no longer stubbed --
  see `_support_geometry_valid`'s inline block (place-preconditions section,
  next to `_expanded_aabb`) and `_support_type_matches_any`'s own docstring
  (near `_objects_touching`) for the real mechanisms and each's one
  documented narrow structural gap (a per-task target-object-role registry
  for the object-kind branch of place preconditions' support_type_matches_
  object, and RoboCasa's floor-exclusion for the settle-scoped `_support_
  type_matches_any` -- structurally moot for place preconditions' own
  fixture-kind branch specifically, since LIBERO's landing-target inference
  never selects a floor candidate in the first place, this corpus's
  fixtures_dict having no floor entry at all). fixture_ready_for_{press,
  turn,slide,twist,open_close} were also ported this same pass -- see this
  docstring's own fixture-skill-onset paragraph above.

Net: 14 of the 20 `TASK_AGNOSTIC_PROPERTY_SPECS` show real, empirically-active
signal for this corpus (rc_no_forbidden_contact,
rc_grasp_remains_synced_until_dropped, rc_dropped_object_was_released,
rc_released_object_eventually_settles, rc_pick_preconditions_safe,
rc_place_preconditions_safe, rc_slide_preconditions_safe,
rc_open_close_preconditions_safe, rc_twist_preconditions_safe,
rc_fixture_open_obstacle_retract, rc_fixture_close_obstacle_retract,
rc_microwave_single_object_until_empty,
rc_reach_in_fixture_only_when_fully_open,
rc_fixture_placement_release_after_internal_support); all 20 have a real
implementation attempt (not a hand-stub), and the remaining 6 are vacuously
satisfied because this specific corpus never exercises their mechanism
(verified per-family above), not because the mechanism itself was skipped.
Iterate via the same `predicate-design-cycle` workflow used for RoboCasa's
predicates.py, scoped to this file.

All threshold constants below are first-pass, NOT empirically tuned against
a real corpus (unlike RoboCasa's, which went through ~20 dated iterations --
see monitor/output/CHANGELOG.md). Frame-count constants assume this module is
called once per RAW simulator frame (LIBERO's demo hdf5s are recorded at the
env's `control_freq` default of 20Hz with one `states` row per raw physics
step), analogous to robocasa's extraction script's `call_stride=1` case.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .attributes import (
    ACTION_COMPONENT_KEYWORDS,
    COOKABLE_NAME_SUBSTRINGS,
    DISHWASHABLE_NAME_SUBSTRINGS,
    FAUCET_FIXTURE_NAME_SUBSTRINGS,
    FOOD_NAME_SUBSTRINGS,
    FRAGILE_NAME_SUBSTRINGS,
    LIQUID_NAME_SUBSTRINGS,
    MICROWAVABLE_NAME_SUBSTRINGS,
    MICROWAVE_FIXTURE_NAME_SUBSTRINGS,
    OPENABLE_FIXTURE_NAME_SUBSTRINGS,
    RAW_NAME_SUBSTRINGS,
    RTE_NAME_SUBSTRINGS,
    TOOL_NAME_SUBSTRINGS,
    WASHABLE_NAME_SUBSTRINGS,
    object_category_from_instance_name,
    object_is_receptacle_category,
)

# Added 2026-09-16 (explicit user decision) -- moved out of the per-frame
# compute function's local scope (previously defined only inside the
# contamination section, after place preconditions ran, so place couldn't
# call them) to top-level module functions, callable from anywhere.
def _is_raw(name: Optional[str]) -> bool:
    if not name:
        return False
    category = object_category_from_instance_name(name)
    return any(s in category for s in RAW_NAME_SUBSTRINGS)


def _is_rte(name: Optional[str]) -> bool:
    if not name:
        return False
    category = object_category_from_instance_name(name)
    return any(s in category for s in RTE_NAME_SUBSTRINGS)


def _is_fragile(name: Optional[str]) -> bool:
    if not name:
        return False
    category = object_category_from_instance_name(name)
    return any(s in category for s in FRAGILE_NAME_SUBSTRINGS)

# ---------------------------------------------------------------------------
# v0 threshold constants (all in meters / radians / raw-frame counts; NOT
# empirically tuned -- see module docstring).
# ---------------------------------------------------------------------------
# 2026-09-09: aligned with RoboCasa's own tuned values (predicates.py)
# wherever the two are genuinely comparable -- both simulators default to
# control_freq=20 (confirmed directly in each engine's own env base class),
# so frame-count constants carry over 1:1, and RoboCasa's OBJ_LINEAR_
# STABLE_THRESHOLD/OBJ_ANGULAR_STABLE_THRESHOLD are real MuJoCo body
# velocities (m/s, rad/s), not per-frame deltas -- converted here via
# velocity * dt (dt = 1/20 = 0.05s) to get the equivalent per-frame delta
# LIBERO's own STABLE_*_DELTA_THRESHOLD actually compares against. Left
# GRIPPER_OPEN_FRACTION_THRESHOLD/GRIPPER_CLOSED_THRESHOLD unaligned
# (RoboCasa's is a raw joint qpos value specific to its own gripper model,
# LIBERO's is a normalized 0-1 closed-fraction -- genuinely different
# quantities, no valid conversion between them).
STABLE_LINEAR_DELTA_THRESHOLD = 0.0025   # = RoboCasa's OBJ_LINEAR_STABLE_THRESHOLD (0.05 m/s) * dt
STABLE_ANGULAR_DELTA_THRESHOLD = 0.0125  # = RoboCasa's OBJ_ANGULAR_STABLE_THRESHOLD (0.25 rad/s) * dt
# STABLE_PERSISTENCE_FRAMES (2026-09-20) = RoboCasa's own constant of the
# same name (predicates.py, ~line 336), used by _persistent_bool_sticky_true
# below to give object_stable_by_name the same asymmetric debounce RoboCasa's
# has: becoming stable is reported instantly (no persistence delay at all),
# but staying reported-stable through a brief raw-unstable blip is smoothed
# -- STABLE_PERSISTENCE_FRAMES consecutive raw-unstable frames are required
# before flipping back to unstable. Ported because LIBERO's object_stable
# was previously a raw, undebounced per-frame check with no equivalent
# mechanism at all (grep confirmed STABLE_PERSISTENCE_FRAMES did not exist
# anywhere in this file before this fix), unlike every other per-object
# stability consumer in RoboCasa (pick/place/settle), which all go through
# this exact debounce.
STABLE_PERSISTENCE_FRAMES = 5
SYNC_RELATIVE_DELTA_THRESHOLD = 0.03    # = RoboCasa's GRASP_SLIP_LINEAR_THRESHOLD (already a per-frame position delta, same units, no conversion)
# SYNC_ANGULAR_DELTA_THRESHOLD (2026-09-21, comprehensive-mirror audit):
# = RoboCasa's GRASP_SLIP_ANGULAR_THRESHOLD (predicates.py, 0.3 rad) --
# already a per-frame quaternion-angle delta (the angle of the diff
# quaternion between the object's actual and rigidly-expected orientation
# this frame), not a rate, so no dt conversion is needed, same as
# SYNC_RELATIVE_DELTA_THRESHOLD above. Added because object_sync below
# previously had NO angular/orientation component at all (only compared
# a raw world-frame obj_pos-eef_pos offset, never rotated into the eef's
# own frame and never compared against the object's orientation) -- a
# real gap against RoboCasa's own _object_grasp_slip/_object_sync, which
# independently requires BOTH linear_slip < GRASP_SLIP_LINEAR_THRESHOLD
# AND angular_slip < GRASP_SLIP_ANGULAR_THRESHOLD. Confirmed via corpus
# comparison: RoboCasa's own v28 500-episode baseline shows 8/500 real
# rc_grasp_remains_synced_until_dropped violations, while LIBERO's
# parallel v28 comprehensive-mirror corpus (400 episodes) showed 0/400 --
# consistent with a structurally weaker sync check silently missing
# genuine slip/rotation events, not with LIBERO grasps simply never
# slipping.
SYNC_ANGULAR_DELTA_THRESHOLD = 0.3
GRIPPER_FAR_THRESHOLD = 0.12            # eef-to-object distance considered "away" (m) -- fallback tier only, see MESH_GRIPPER_FAR_THRESHOLD
MESH_GRIPPER_FAR_THRESHOLD = 0.01        # = RoboCasa's own GRIPPER_FAR_THRESHOLD (real mesh/geom gap, same units) -- primary tier, see _gripper_far_from_object
# REACH_THRESHOLD (2026-09-21, distance-basis audit): = RoboCasa's own single
# REACH_THRESHOLD (predicates.py:482, 0.05) -- confirmed by reading RoboCasa's
# actual code that it uses this ONE constant for BOTH pick-onset proximity
# (gripper_near_object) AND all 5 fixture-skill-onset proximities
# (gripper_near_target_by_action), never two separate numbers, and that in
# both cases the compared distance is a genuine gripper-bounding-box-to-target
# distance (_gripper_object_distances/_target_distances: gripper OBB to
# target's own OBB when both are available, degrading to gripper-OBB-to-
# target-center-point, then finally to a raw point/point fallback only when
# no AABB is resolvable at all) -- never a bare eef-point-to-object-center-
# point distance. NEAR_OBJECT_THRESHOLD/FIXTURE_NEAR_THRESHOLD below were
# previously two independently-tuned numbers (0.09, 0.30) compared against
# raw eef-point-to-body-origin distance with no size/extent awareness --
# replaced with a real gripper-AABB-to-target distance (see
# _gripper_target_distance below, LIBERO's axis-aligned analog of RoboCasa's
# OBB-based functions -- LIBERO's own geometry infra is axis-aligned-only,
# see _geom_aabb's own docstring) and unified onto this single shared
# threshold, matching RoboCasa having exactly one constant/one distance basis
# for this purpose. Both names are kept (not renamed) to avoid a disruptive
# rename across their many existing call sites/comments; both are now simply
# aliased to this one canonical value.
REACH_THRESHOLD = 0.05
NEAR_OBJECT_THRESHOLD = REACH_THRESHOLD  # eef/gripper-AABB-to-object distance considered "near" for onset (m) -- see REACH_THRESHOLD's own comment
GRIPPER_OPEN_FRACTION_THRESHOLD = 0.35  # gripper closed-fraction below this counts as "open enough to release"
# GRASP_GATE_MIN_FINGER_FRACTION_THRESHOLD (2026-09-21, corpus-wide
# object_grasped-never-fires regression fix): _check_grasp_any's own
# gripper-closed gate used to reuse GRIPPER_OPEN_FRACTION_THRESHOLD (0.35)
# against _gripper_closed_fraction's MEAN-of-both-fingers reading -- fine for
# a release gate (mean tracks overall opening/closing motion well) but wrong
# for a grasp-acceptance gate on thin/off-center objects: confirmed via real
# frame data (libero_object suite, tomato_sauce/milk/orange_juice
# pick-and-place tasks) that bilateral finger-pad contact is genuinely
# present (both fingers independently touching the object, contact solver
# confirmed via env.sim.data.contact) throughout a real, sustained lift, yet
# the MEAN closed-fraction across both fingers never exceeds ~0.21-0.33 the
# entire time -- below the 0.35 gate -- because these objects are thin/
# narrow enough (and often off-center between the fingers) that one finger
# travels much further than the other before contact (e.g. tomato_sauce_1
# ep0: 0.36 vs 0.06 at the same frame), dragging the mean down even though
# real bilateral contact is happening. This silently zeroed out
# object_grasped/object_grasped_raw for 100% of frames across all 3 affected
# tasks' full 10-episode corpora (confirmed corpus-wide), cascading into
# every predicate downstream of grasp state (pick/place onset,
# object_dropped/released, settle-watch, contact-role taxonomy).
# RoboCasa's own real analog (`OU.check_obj_grasped`,
# robocasa/utils/object_utils.py) does NOT have this failure mode: it
# requires EACH finger joint's raw qpos individually below
# GRIPPER_CLOSED_THRESHOLD=0.0399 (out of a ~0.04 full-open joint range --
# i.e. ~99.75% of the way to the fully-open limit), an almost-vacuous
# per-finger AND gate whose only real job is excluding a literal wide-open
# gripper -- bilateral contact itself is what actually discriminates a real
# grasp. Ported that same intent here literally: MIN across fingers (not
# mean -- mirrors RoboCasa's per-joint "each finger individually" AND
# semantics) against a near-vacuous threshold, calibrated against real data
# (worst-case min-finger fraction during a confirmed real bilateral-contact
# window across 9 sampled tomato_sauce/milk/orange_juice episodes: as low as
# 0.025 -- vs. a fully-open, no-contact baseline of ~0.006-0.15) -- 0.02
# sits safely below every observed real-grasp minimum found and above the
# fully-open baseline, while still doing essentially the same "not literally
# wide open" job RoboCasa's own gate does. Deliberately does NOT touch
# GRIPPER_OPEN_FRACTION_THRESHOLD/_gripper_closed_fraction's own mean
# semantics -- that pairing is still correct and unchanged for its own
# consumers (the release-side gripper_is_opening/gripper_is_closing motion
# signals at build_predicate_snapshot's "grasp / release / settle" block,
# which want a smooth, both-fingers-averaged continuous motion signal, not a
# single-finger floor).
GRASP_GATE_MIN_FINGER_FRACTION_THRESHOLD = 0.02
# GRASP_BILATERAL_MIN_CONTACT_BODIES (2026-09-21): = RoboCasa's own constant
# of the same name (predicates.py, value 2) -- how many distinct gripper
# finger groups must independently contact the object for _check_grasp_any's
# new manual bilateral-contact scan (replacing robosuite's own env._check_grasp
# aggregate-contact utility) to count it as grasped. See
# _gripper_finger_geom_groups's own docstring for why LIBERO's version groups
# by robosuite's important_geoms left_finger/right_finger keys rather than
# RoboCasa's raw-MuJoCo-body-id grouping.
GRASP_BILATERAL_MIN_CONTACT_BODIES = 2
# PATH_OBSTRUCTION_OVERLAP_ALLOWANCE (2026-09-20): = RoboCasa's own constant
# of the same name (predicates.py), used by the real swept-path-obstruction
# geometry (_aabb_obstructs_between_endpoints below) that replaced the
# former REGION_CLEAR_RADIUS/REGION_CLEAR_MAX_FOREIGN proximity heuristic
# for object_region_clear/support_region_clear/target_region_clear -- see
# _aabb_obstructs_between_endpoints's own docstring for why the radius
# heuristic was wrong (it could not tell "something is genuinely in the way
# of this placement" from "there's simply another object resting somewhere
# near this point", e.g. flagging an ordinary second item placed into a
# basket that already contains a first item elsewhere in the same basket).
PATH_OBSTRUCTION_OVERLAP_ALLOWANCE = 0.05
# PLACEMENT_MARGIN / SUPPORT_CLUTTER_Z_TOLERANCE / SUPPORT_TARGET_XY_MULTIPLIER
# (2026-09-20): = RoboCasa's own constants of the same names (predicates.py),
# used by _infer_landing_target -- LIBERO's simplified analog of RoboCasa's
# _infer_support/_spos (that file, search those names). RoboCasa continuously
# re-guesses, every frame while an object is still being carried, where it is
# probably headed (a not-yet-reached candidate support surface), then sweeps
# _support_region_blockers from the object's CURRENT position to that live
# guess -- as the object approaches its real landing spot, current position
# and guessed target naturally converge, so the checked corridor shrinks to a
# short, local, near-target-only region every frame. LIBERO's own version
# needs an analog of "where is this probably headed" to get the same
# continuously-shrinking-corridor behavior -- see _infer_landing_target's own
# docstring for the simplified (non-fixture/object-type-matching) heuristic
# used here.
PLACEMENT_MARGIN = 0.03                 # = RoboCasa's own PLACEMENT_MARGIN
SUPPORT_CLUTTER_Z_TOLERANCE = 0.05      # = RoboCasa's own SUPPORT_CLUTTER_Z_TOLERANCE
SUPPORT_TARGET_XY_MULTIPLIER = 3.0      # = RoboCasa's own xy_multiplier used for object/fixture support candidates in _infer_support
# Added 2026-09-16 (explicit user decision) -- = RoboCasa's own
# PLACEMENT_PROXIMITY_MARGIN, for support_objects_clean_for_manipulated_
# object's contamination-proximity check and support_not_cluttered_for_
# fragile_manipulated_object's clutter check (see predicates.py's own place
# preconditions section).
PLACEMENT_PROXIMITY_MARGIN = 0.01
CLUTTER_THRESHOLD = 2                   # = RoboCasa's own CLUTTER_THRESHOLD
# 2026-09-21 (comprehensive-mirror audit): replaced the prior
# UPRIGHT_COS_THRESHOLD (0.85, a rotationally-symmetric tilt-from-vertical
# cone with a ~31.8 degree half-angle) with a literal port of RoboCasa's
# REAL upright check, robocasa/utils/object_utils.py's check_obj_upright
# (th=15) -- an axis-aligned Euler roll/pitch box, not a cone. RoboCasa's
# own predicates.py never implements this itself (_object_is_upright is a
# thin OU.check_obj_upright wrapper), so a prior literal-translation pass
# compared against the wrong reference entirely. See _upright()'s own
# docstring for the exact ported formula.
UPRIGHT_EULER_THRESHOLD_DEG = 15.0       # = RoboCasa's own check_obj_upright(th=15) (object_utils.py)
FIXTURE_INTERIOR_RADIUS = 0.18          # eef/object-to-fixture-body distance considered "inside" (m)
FIXTURE_ARTICULATION_DELTA_THRESHOLD = 2e-3  # per-raw-frame open-fraction delta counted as "articulating"

# Updated 2026-09-20 from 8 to 10: RoboCasa's own SKILL_ONSET_FRAMES was
# lowered from 20 to 10 the same day (explicit user decision, part of that
# session's v28 iteration goal), which this constant's own "= RoboCasa's
# own" comment claims to track -- 8 was accurate against an earlier point
# in RoboCasa's own history of this constant (2->8->50->20->10 across past
# sessions), not a LIBERO-specific tuning choice with its own rationale, so
# restoring the equality this comment already asserts rather than leaving
# it stale. Re-verified for regressions against a 20-episode spot check
# spanning most task families in this corpus (see this session's own
# fork-verification notes) -- no violated/satisfied flips found from this
# change alone.
SKILL_ONSET_FRAMES = 10         # = RoboCasa's own SKILL_ONSET_FRAMES (consecutive near-object/contact-and-articulating frames before an onset fires)
SETTLE_TIMEOUT_FRAMES = 100     # = RoboCasa's own SETTLE_TIMEOUT_FRAMES (frames a dropped/released object has to settle before timeout) -- LIBERO's original 60 was an untuned v0 guess, confirmed too short directly: put_the_wine_bottle_on_the_rack ep0 genuinely settles (supported+stable+gripper-away) ~80 frames after release, timing out at 60 with the object already correctly at rest by 100
FORBIDDEN_CONTACT_TOLERANCE_FRAMES = 20  # = RoboCasa's own FORBIDDEN_CONTACT_TOLERANCE_FRAMES (frames of arm-contact tolerated before "sustained")
DEFAULT_CONTAMINATION_RADIUS = 0.05  # = RoboCasa's own DEFAULT_CONTAMINATION_RADIUS (fallback contamination-spot radius when no real geometry is resolvable)
CONTACT_PERSISTENCE_FRAMES = 3   # frames an open/close obstacle contact must persist before counting as a "hit"
RETRACT_TIMEOUT_FRAMES = SETTLE_TIMEOUT_FRAMES  # = RoboCasa's own RETRACT_TIMEOUT_FRAMES, 2026-09-16 redesign: aliased to SETTLE_TIMEOUT_FRAMES rather than a separately-tuned constant; now bounds time-since-obstacle-hit, not time-spent-already-retracting (see fixture_open_retract_timeout below)
# FIXTURE_NEAR_THRESHOLD (2026-09-21, distance-basis audit): was 0.30,
# deliberately larger than object-proximity thresholds elsewhere in this file
# ONLY because it was compared against the fixture's raw root-body ORIGIN
# (e.g. a cabinet carcass's structural center), never the fixture's own
# physical extent -- a real handle-pull keeps the eef 0.14-0.30m from that
# origin point for most of the pull (confirmed on a real KITCHEN_SCENE4
# episode) even while the eef is genuinely flush against the handle's own
# surface. Now that the near-check below compares against the fixture's own
# body AABB (_gripper_target_distance, gripper AABB to fixture-body AABB) --
# the same "compare against actual extent, not a body-origin point" fix
# _gripper_object_geom_min_distance/_object_aabb already made for objects --
# the origin-offset justification for a larger number no longer applies, so
# this is unified onto the same REACH_THRESHOLD as everything else (see that
# constant's own comment).
FIXTURE_NEAR_THRESHOLD = REACH_THRESHOLD    # gripper-AABB-to-fixture-body-AABB distance considered "near" for press/turn/slide/twist/open_close onset
# CONTAMINATION_PERSISTENCE_FRAMES (independently hand-tuned, value 3)
# retired 2026-09-20 in favor of reusing FORBIDDEN_CONTACT_TOLERANCE_FRAMES
# for both contamination-related persistence checks -- see the
# contamination section's own comment (around raw_contact_sustained/
# robot_contact_clean) for the full reasoning.
# GRASP_CANDIDATE_PERSISTENCE_FRAMES (2026-09-20): = RoboCasa's own
# GRASP_CANDIDATE_PERSISTENCE_FRAMES. This file's own _check_grasp_any
# docstring claimed the one-frame bilateral-contact-registration flicker
# problem (documented there: 17/27 rc_dropped_object_was_released
# violations were this exact flicker, per an earlier session's corpus
# audit) was "fixed at the raw-signal level" by ANDing bilateral contact
# with a gripper-closed-fraction threshold, mirroring RoboCasa's own
# (at-the-time) 2026-09-08 claim of the same thing. RoboCasa's own
# investigation THIS session found that claim to be FALSE there -- genuine
# flicker still occurred (WashLettuce/DeliverStraw/PortionHotDogs). Re-
# verified for LIBERO with real data (not just re-trusting the old
# comment): confirmed 2 genuine single-frame grasp dropouts in a 4-episode
# spot check --
# pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
# ep0 frame 73 (eef-to-bowl distance flat at 0.0569-0.0641m across the
# dropout, gripper_frac monotonically increasing 0.910->0.935, i.e. still
# closing, never opening -- the object never left the gripper) and
# open_the_top_drawer_and_put_the_bowl_inside ep0 frame 116 (same flat-
# distance signature, ~0.055-0.058m throughout). The gripper_frac AND-
# condition doesn't catch this because gripper_frac never dips below
# threshold during either dropout -- it's a genuinely independent, still-
# unfixed bilateral-contact solver/discretization flicker, the same root
# cause RoboCasa's own investigation found. Reuses RoboCasa's own value (5)
# for the same reason FORBIDDEN_CONTACT_TOLERANCE_FRAMES is reused rather
# than independently retuned: this is the identical underlying contact-
# query noise characteristic (a robosuite/MuJoCo bilateral-contact check),
# not a LIBERO-specific phenomenon needing its own calibration.
GRASP_CANDIDATE_PERSISTENCE_FRAMES = 5

# GRASP_RELEASE_UNCORROBORATED_FALLBACK_FRAMES (2026-09-21, KITCHEN_SCENE8/
# LIVING_ROOM_SCENE2 residual support_geometry_valid false-positive audit):
# bounded safety valve for _persistent_grasp_candidate's new gripper-opening
# corroboration requirement (see that function's own docstring) -- if a
# candidate->None (release) transition's raw reading has persisted this many
# frames with NO `gripper_is_opening` evidence ever observed during the run,
# accept it anyway rather than waiting forever, so a genuine object physically
# dislodged without the gripper ever opening (e.g. knocked out of a still-
# closing grip) still eventually registers as released. Set well above the
# longest real uncorroborated flicker confirmed in this audit (17 frames,
# KITCHEN_SCENE8 ep8 328-344) -- not tuned to that exact number, since the
# whole point of requiring corroboration is to not treat "how long can a
# flicker plausibly run" as bounded; picked instead to match the order of
# magnitude of the longest real SLOW-TELEOPERATED-PLACEMENT gap already
# documented in this file's history (6b22648's microwave fix: a genuine
# still-controlled placement with raw contact false for ~60 consecutive
# frames) -- i.e. long enough that a real, still-in-progress placement never
# hits this fallback early, short enough that a real accidental drop with no
# corroborating gripper motion still resolves within a couple of seconds of
# sim time rather than never.
GRASP_RELEASE_UNCORROBORATED_FALLBACK_FRAMES = 90

# Matches RoboCasa's own PERSISTENCE_FRAMES (predicates.py line 317),
# reused there for both microwave_empty's own debounce (microwave_empty_
# count >= PERSISTENCE_FRAMES) and its occupancy-stable-count candidate
# debounce -- see microwave_empty's own comment below.
MICROWAVE_EMPTY_PERSISTENCE_FRAMES = 5


def _entry(value: bool, language: str = "", readout: Any = None) -> Dict[str, Any]:
    return {"value": bool(value), "language": language, "readout": readout}


# ---------------------------------------------------------------------------
# Raw-state helpers
# ---------------------------------------------------------------------------

def _movable_object_names(env) -> List[str]:
    return list(getattr(env, "objects_dict", {}).keys())


def _fixture_names(env) -> List[str]:
    return list(getattr(env, "fixtures_dict", {}).keys())


def _body_pos(env, name: str) -> Optional[np.ndarray]:
    body_id = getattr(env, "obj_body_id", {}).get(name)
    if body_id is None:
        return None
    try:
        return np.array(env.sim.data.body_xpos[body_id], dtype=float)
    except Exception:
        return None


def _body_quat(env, name: str) -> Optional[np.ndarray]:
    body_id = getattr(env, "obj_body_id", {}).get(name)
    if body_id is None:
        return None
    try:
        return np.array(env.sim.data.body_xquat[body_id], dtype=float)
    except Exception:
        return None


def _eef_pos(env) -> Optional[np.ndarray]:
    try:
        gripper = env.robots[0].gripper
        site_name = gripper.important_sites["grip_site"]
        return np.array(env.sim.data.get_site_xpos(site_name), dtype=float)
    except Exception:
        return None


def _eef_quat(env) -> Optional[np.ndarray]:
    """End-effector (grip site) world orientation, wxyz -- the LIBERO
    counterpart to RoboCasa's own _eef_orientation() (predicates.py),
    added 2026-09-21 (comprehensive-mirror audit) so object_sync below can
    do the same rotation-corrected linear check and angular-slip check
    RoboCasa's _object_grasp_slip does. There is no site_xquat in mjData
    directly (only site_xmat, a 3x3 rotation matrix) -- mujoco.mju_mat2Quat
    is the same conversion MuJoCo's own C API uses internally, returned
    already in wxyz (mujoco's native quaternion convention, matching
    _body_quat's env.sim.data.body_xquat above -- no xyzw reorder needed,
    unlike RoboCasa's own _xyzw_to_wxyz helper, which exists only because
    RoboCasa's kitchen_ext.py stores poses in robosuite/scipy's xyzw
    convention instead)."""
    try:
        gripper = env.robots[0].gripper
        site_name = gripper.important_sites["grip_site"]
        site_mat = np.asarray(env.sim.data.get_site_xmat(site_name), dtype=float).reshape(3, 3)
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, site_mat.reshape(-1))
        if not np.all(np.isfinite(quat)):
            return None
        return quat
    except Exception:
        return None


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_multiply(_quat_multiply(q, qv), _quat_conjugate(q))[1:4]


def _quat_angle(q: np.ndarray) -> float:
    w = float(np.clip(abs(q[0]), -1.0, 1.0))
    return float(2.0 * np.arccos(w))


def _gripper_finger_closed_fractions(env) -> Optional[list]:
    """Per-joint closed-fraction list (0=fully open, 1=fully closed), one
    entry per gripper finger joint. Factored out of `_gripper_closed_fraction`
    (2026-09-21) so callers needing a per-finger (not averaged) reading --
    see `_gripper_min_closed_fraction` -- don't have to recompute this."""
    try:
        gripper = env.robots[0].gripper
        fracs = []
        for jn in gripper.joints:
            jid = env.sim.model.joint_name2id(jn)
            lo, hi = env.sim.model.jnt_range[jid]
            qpos = float(env.sim.data.get_joint_qpos(jn))
            span = max(abs(lo), abs(hi)) or 1.0
            fracs.append(1.0 - min(1.0, abs(qpos) / span))
        return fracs if fracs else None
    except Exception:
        return None


def _gripper_closed_fraction(env) -> Optional[float]:
    fracs = _gripper_finger_closed_fractions(env)
    return float(np.mean(fracs)) if fracs else None


def _gripper_min_closed_fraction(env) -> Optional[float]:
    """MIN (not mean) across finger joints -- see
    GRASP_GATE_MIN_FINGER_FRACTION_THRESHOLD's own comment for why
    `_check_grasp_any`'s gripper-closed gate needs this instead of
    `_gripper_closed_fraction`'s mean: a thin/off-center object can leave one
    finger much less closed than the other, and it's specifically the
    least-closed finger that must still be "not literally wide open" for a
    real bilateral grasp, mirroring RoboCasa's own per-joint (not averaged)
    `OU.check_obj_grasped` AND-gate."""
    fracs = _gripper_finger_closed_fractions(env)
    return float(min(fracs)) if fracs else None


def _gripper_finger_geom_groups(env) -> Dict[str, set]:
    """Group the gripper's own contact geoms into distinct-finger buckets.

    Adapted from RoboCasa's own `_gripper_finger_body_contact_map`
    (predicates.py, 2026-09-21 port) for LIBERO's Panda gripper. RoboCasa's
    version groups by raw MuJoCo body id, which does NOT port verbatim here:
    LIBERO/robosuite's `panda_gripper.xml` nests each fingertip pad geom
    (`finger1_pad_collision`/`finger2_pad_collision`) on its own separate
    child body (`finger_joint{1,2}_tip`) below its parent finger body
    (`leftfinger`/`rightfinger`), and the palm itself (`hand_collision`) is
    a contact-enabled body of its own too -- confirmed directly against the
    installed robosuite package's gripper XML and `contact_geoms`/
    `important_geoms` output. Raw body-id grouping would therefore see 5
    independently-contactable bodies (palm + 2 finger-collision bodies + 2
    fingertip-pad bodies), any 2 of which -- including the palm plus a
    single finger -- would satisfy a ">=2 distinct bodies" bilateral test,
    defeating the actual "two distinct fingers touch the object"
    antipodal-contact intent this check exists for.

    Uses robosuite's own `important_geoms` mapping instead (`left_finger`/
    `right_finger` keys -- each already spans that finger's own collision
    geom AND its pad geom, with no palm geom in either) to build the
    canonical two-finger split directly. This is robust to whatever
    body-nesting a given gripper's XML happens to use, and degrades cleanly
    (an empty/short dict, triggering `_object_gripper_bilateral_contact`'s
    own fallback to the aggregate any-contact check) for any gripper that
    isn't a standard two-finger parallel-jaw design.
    """
    try:
        gripper = env.robots[0].gripper
        important = getattr(gripper, "important_geoms", None) or {}
    except Exception:
        return {}
    groups: Dict[str, set] = {}
    for key in ("left_finger", "right_finger"):
        names = important.get(key)
        if not names:
            continue
        ids = _geom_ids_from_names(env, names)
        if ids:
            groups[key] = ids
    return groups


def _object_gripper_contact_any(env, name: str) -> bool:
    """Fallback aggregate contact check (any gripper geom vs. object geom).

    Ported from RoboCasa's own `_object_gripper_contact_any` (predicates.py).
    Only used by `_object_gripper_bilateral_contact` when the gripper's
    contact geoms can't be split into >=2 distinct finger groups at all
    (non two-finger end effector) -- LIBERO's Panda gripper always has both
    groups, so this path is not expected to trigger in practice, but is
    ported anyway for robustness/fidelity with RoboCasa's design.
    """
    object_geom_ids = _object_geom_ids(env, name)
    gripper_geom_ids = _gripper_contact_geom_ids(env)
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


def _object_gripper_bilateral_contact(env, name: str) -> bool:
    """Require independent contact from at least
    GRASP_BILATERAL_MIN_CONTACT_BODIES distinct gripper finger groups
    simultaneously (an antipodal-contact precondition), instead of
    aggregate any-geom contact.

    Ported from RoboCasa's own `_object_gripper_bilateral_contact`
    (predicates.py, 2026-09-21), adapted to use
    `_gripper_finger_geom_groups`'s important_geoms-based split (see that
    function's own docstring for why raw MuJoCo body id doesn't port
    verbatim for LIBERO's Panda gripper). Falls back to the aggregate
    any-geom check if the gripper's contact geoms can't be split into >=2
    distinct finger groups.
    """
    object_geom_ids = _object_geom_ids(env, name)
    if not object_geom_ids:
        return False
    finger_groups = _gripper_finger_geom_groups(env)
    if len(finger_groups) < 2:
        return _object_gripper_contact_any(env, name)
    contacted_fingers: set = set()
    contact_number = int(getattr(env.sim.data, "ncon", 0))
    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue
        for label, geom_ids in finger_groups.items():
            if label in contacted_fingers:
                continue
            if (geom1 in geom_ids and geom2 in object_geom_ids) or (
                geom2 in geom_ids and geom1 in object_geom_ids
            ):
                contacted_fingers.add(label)
    return len(contacted_fingers) >= max(1, int(GRASP_BILATERAL_MIN_CONTACT_BODIES))


def _check_grasp_any(env) -> Optional[str]:
    """Returns the name of a movable object the gripper is bilaterally
    grasping, or None. ANDs a manual, direct bilateral-contact scan
    (`_object_gripper_bilateral_contact` -- both gripper fingers
    independently, distinctly touch the object, ported from RoboCasa's own
    `_object_gripper_bilateral_contact`) with the gripper being closed
    enough (`GRASP_GATE_MIN_FINGER_FRACTION_THRESHOLD`, a pure
    joint-position/finger-width signal, no contact dependency), mirroring
    RoboCasa's own `_object_is_grasped` (predicates.py,
    `_object_gripper_bilateral_contact and OU.check_obj_grasped(...)`) --
    same rationale, adapted (2026-09-21).

    Gripper-closed gate now uses `_gripper_min_closed_fraction` (MIN across
    finger joints) against `GRASP_GATE_MIN_FINGER_FRACTION_THRESHOLD`, NOT
    `_gripper_closed_fraction` (mean) against `GRIPPER_OPEN_FRACTION_
    THRESHOLD` as originally written -- see
    GRASP_GATE_MIN_FINGER_FRACTION_THRESHOLD's own comment (2026-09-21
    regression fix): the mean-based gate silently zeroed out object_grasped
    for 100% of frames on every episode of 3 real "pick up X" tasks
    (thin/off-center objects -- tomato_sauce/milk/orange_juice -- never
    raised the *mean* closed-fraction above ~0.21-0.33 despite genuine,
    confirmed bilateral finger-pad contact throughout a real sustained
    lift), because one finger travels much further than the other before
    contacting an off-center/thin object and dragged the average down.

    Does NOT call robosuite's own `env._check_grasp` (an aggregate-contact
    utility -- "any" gripper geom touching "any" of the object's geoms,
    with no per-finger distinction) for the contact half of this check:
    confirmed on `open_the_top_drawer_and_put_the_bowl_inside` ep4 that
    `env._check_grasp` never returned True for an entire real lift-and-drop
    (bowl visibly rising then falling, frames ~109-148) -- present in a v22
    baseline extraction weeks earlier too, so not a new regression. Also
    deliberately does NOT route the closed-gripper half through RoboCasa's
    own `OU.check_obj_grasped` -- despite RoboCasa's own code nominally
    calling it, that utility's real implementation
    (`robocasa/utils/object_utils.py`) is itself
    `env.check_contact(gripper, obj) and gripper_closed`, i.e. it ALSO
    depends on an aggregate-contact check of the same "trash" flavor as
    `env._check_grasp` -- porting it here would just reintroduce the same
    unreliable contact dependency through a different name. Gripper
    closedness is instead computed purely from joint position/finger width
    via `_gripper_closed_fraction`, which already has no contact
    dependency at all.

    Originally (2026-09-09) this function ANDed `env._check_grasp` with the
    same closed-gripper check for a different reason -- absorbing a single-
    frame solver/discretization contact-drop flicker (confirmed corpus-wide:
    17/27 rc_dropped_object_was_released violations were this exact
    one-frame flicker, gripper still recorded as actively closing at the
    "drop") -- and that AND-with-closed-gripper reasoning still holds and is
    preserved here; only the contact half of the AND has changed, from
    aggregate any-geom contact to bilateral two-finger contact."""
    gripper_frac = _gripper_min_closed_fraction(env)
    if gripper_frac is not None and gripper_frac < GRASP_GATE_MIN_FINGER_FRACTION_THRESHOLD:
        return None
    for name in _movable_object_names(env):
        try:
            if _object_gripper_bilateral_contact(env, name):
                return name
        except Exception:
            continue
    return None


def _persistent_grasp_candidate(
    state: Dict[str, Any], raw_candidate: Optional[str], gripper_is_opening: bool = False
) -> Optional[str]:
    """Ported from RoboCasa's own predicates.py _persistent_grasp_candidate
    (2026-09-20, after confirming via real LIBERO data -- see
    GRASP_CANDIDATE_PERSISTENCE_FRAMES's own comment -- that this file's
    _check_grasp_any docstring's "flicker already fixed at the raw-signal
    level" claim is false here too, the same way RoboCasa's identical claim
    turned out to be false). Symmetric: the accepted candidate (including
    None) only changes once the same raw reading has held for
    GRASP_CANDIDATE_PERSISTENCE_FRAMES consecutive frames -- both a
    spurious wrong-object appearance and a spurious one-frame drop are
    absorbed by the same mechanism. See RoboCasa predicates.py's own
    extensive comment history above its GRASP_CANDIDATE_PERSISTENCE_FRAMES
    for why symmetric (not asymmetric) is the final, correct design, and
    why the "until" target for grasp-sync-until-dropped needed a separate
    undebounced *level* (object_grasped_raw) rather than trying to make
    this debounce itself asymmetric -- ported here as object_grasped_raw
    below, mirroring that exact fix rather than re-deriving it.

    gripper-opening corroboration (2026-09-21, KITCHEN_SCENE8/
    LIVING_ROOM_SCENE2 residual support_geometry_valid false-positive audit):
    GRASP_CANDIDATE_PERSISTENCE_FRAMES=5 was tuned against confirmed 1-2
    frame bilateral-contact flicker, but real data surfaced a 17-consecutive-
    frame candidate->None run (KITCHEN_SCENE8 ep8, frames 328-344) with the
    gripper actively CLOSING (never opening) and the eef-to-object offset
    perfectly rigid throughout -- i.e. the object never actually left the
    grip; `_check_grasp`'s bilateral query itself just stopped reading True
    for an order of magnitude longer than the tuned flicker. A genuine
    release in the same episode (moka_pot_2, frame 183) showed
    `gripper_is_opening=True` for 10+ frames before the debounce even began
    accepting the release. Rather than raise GRASP_CANDIDATE_PERSISTENCE_
    FRAMES corpus-wide (no principled ceiling -- the false run here was
    already 3x the tuned value), a candidate->None transition specifically
    now ALSO requires at least one real `gripper_is_opening` frame to have
    occurred during the pending run before being accepted, with
    GRASP_RELEASE_UNCORROBORATED_FALLBACK_FRAMES as a bounded safety valve
    (see that constant's own comment) so a real drop with no corroborating
    gripper motion at all still eventually resolves. Candidate->non-None
    transitions (a new grasp starting, or switching objects) are NOT gated
    this way -- only losing an already-accepted grasp is what real data
    showed needed the extra evidence."""
    entry = state.setdefault(
        "grasp_candidate_debounce",
        {"value": None, "pending": None, "count": 0, "release_gripper_opening_seen": False},
    )
    accepted = entry.get("value")
    if raw_candidate == accepted:
        entry["pending"] = raw_candidate
        entry["count"] = 0
        entry["release_gripper_opening_seen"] = False
        return accepted
    pending = entry.get("pending")
    if raw_candidate == pending:
        count = int(entry.get("count", 0)) + 1
    else:
        count = 1
        entry["release_gripper_opening_seen"] = False
    entry["pending"] = raw_candidate
    entry["count"] = count
    is_release_attempt = raw_candidate is None and accepted is not None
    if is_release_attempt and gripper_is_opening:
        entry["release_gripper_opening_seen"] = True
    if count >= max(1, int(GRASP_CANDIDATE_PERSISTENCE_FRAMES)):
        if (
            is_release_attempt
            and not entry.get("release_gripper_opening_seen", False)
            and count < int(GRASP_RELEASE_UNCORROBORATED_FALLBACK_FRAMES)
        ):
            # Persistence threshold reached but no real gripper-opening
            # evidence seen during the run yet, and still within the bounded
            # fallback window -- keep waiting rather than accept a release
            # with nothing physically behind it (see docstring).
            return accepted
        entry["value"] = raw_candidate
        entry["count"] = 0
        entry["release_gripper_opening_seen"] = False
        return raw_candidate
    return accepted


def _touches_anything(env, name: str) -> bool:
    try:
        model = env.get_object(name)
        return bool(env.check_contact(model))
    except Exception:
        return False


def _geom_ids_from_names(env, geom_names) -> set:
    geom_ids = set()
    if isinstance(geom_names, str):
        geom_names = [geom_names]
    for geom_name in geom_names or []:
        try:
            geom_ids.add(int(env.sim.model.geom_name2id(str(geom_name))))
        except Exception:
            continue
    return geom_ids


def _object_geom_ids(env, name: str) -> set:
    try:
        return _geom_ids_from_names(env, env.get_object(name).contact_geoms)
    except Exception:
        return set()


def _gripper_contact_geom_ids(env) -> set:
    try:
        return _geom_ids_from_names(env, env.robots[0].gripper.contact_geoms)
    except Exception:
        return set()


def _gripper_object_geom_min_distance(env, name: str, distmax: float) -> Optional[float]:
    """Real minimum signed distance between any gripper collision geom and
    any of the object's own collision geoms, via MuJoCo's own mj_geomDistance
    -- the same mesh/shape-aware collision-geometry query the contact solver
    itself uses, not the raw body-origin-to-body-origin distance _eef_pos()/
    _body_pos() alone gives. Ported from RoboCasa's own
    _gripper_object_geom_min_distance (predicates.py, 2026-09-03) after
    confirming the same root cause here: LIBERO's objects (moka pots, bowls,
    baskets, ...) have a non-trivial physical extent, so eef-to-body-origin
    distance never approaches zero even with the gripper's fingers flush
    against the object's actual surface -- e.g. KITCHEN_SCENE3's moka_pot_1
    sits ~0.06m from the eef site the entire time it's genuinely grasped,
    and the recorded demo episode simply ends 6-13 raw frames after release
    (LIBERO demos stop recording at task-success detection, not some fixed
    buffer afterward), during which body-origin distance only grows to
    ~0.067m -- nowhere near a body-origin-calibrated "away" threshold, even
    though the gripper's fingers are already several cm clear of the pot's
    actual surface by then. Confirmed via a corpus-wide sweep (2026-09-09):
    this exact shape (object_supported=True, object_stable=True,
    gripper_away_from_object=False for the entire remaining trace) is the
    dominant rc_released_object_eventually_settles failure across nearly
    every LIBERO pick-place task, not a one-off.
    Returns None if the real MjModel/MjData aren't reachable or no geom ids
    are found for either side."""
    try:
        m = env.sim.model._model
        d = env.sim.data._data
    except Exception:
        return None
    gripper_geom_ids = _gripper_contact_geom_ids(env)
    object_geom_ids = _object_geom_ids(env, name)
    if not gripper_geom_ids or not object_geom_ids:
        return None
    fromto = np.zeros(6)
    best = None
    for gid in gripper_geom_ids:
        for oid in object_geom_ids:
            try:
                dist = float(mujoco.mj_geomDistance(m, d, int(gid), int(oid), distmax, fromto))
            except Exception:
                continue
            if best is None or dist < best:
                best = dist
            if best <= 0.0:
                return best
    return best


# ---------------------------------------------------------------------------
# True-OBB geometry core (2026-09-21, mirrors RoboCasa's own 2026-09-18
# "true-OBB rewrite" -- monitor/output/v27_2026-09-18_claude_branch_obb_
# rewrite_grace_scoping_spot_contamination_fixture_fixes/ -- ported into
# LIBERO for gripper/robot/object/fixture geometry uniformly, per explicit
# user direction, since a real corpus-wide scan of this file's own v36
# baseline (400 episodes) confirmed genuinely non-axis-aligned object
# rotation is common here (bowls/bottles/moka-pots tipped/rotated well past
# 90-degree-multiple orientations during real pick/place sequences, e.g.
# put_the_wine_bottle_on_the_rack, open_the_top_drawer_and_put_the_bowl_
# inside, KITCHEN_SCENE8_put_both_moka_pots_on_the_stove), not merely a
# hypothetical edge case -- so the same rotation-inflation artifact
# RoboCasa's rewrite fixed (a rotated box's world-frame min/max envelope
# growing past its own true extent, confirmed there via ArrangeBreadBasket's
# container-Z-flip bug) is a live risk here too.
#
# Representation: OBB = (center, axes, half_extents), identical convention
# to RoboCasa's own (see that file's module docstring): center is (3,)
# world position, axes is (3,3) whose COLUMNS are the box's own unit local
# X/Y/Z axes in world coordinates (same convention as MuJoCo's geom_xmat/
# body_xmat: world_point = center + axes @ local_point), half_extents is
# (3,) half-widths along those local axes. A plain axis-aligned box (no
# single coherent rotation, e.g. _gripper_link_aabb's multi-body point
# cloud) is represented as axes=identity, handled by the same math with no
# branching needed.
#
# Every previously-plain-(lower, upper)-tuple-returning helper below
# (_geom_aabb, _geom_ids_aabb, _object_aabb, _gripper_aabb, and every
# _aabb_*/_union_aabb/_translate_aabb/_expanded_aabb wrapper) now builds/
# consumes this OBB representation instead. Callers throughout the rest of
# this file are unaffected: every one of them goes through these named
# wrapper functions (never destructures an aabb as a raw (lower, upper)
# pair directly -- the 3 call sites that used to were also updated, see
# _infer_landing_target's `_consider`, _entity_footprint_radius,
# _contact_patch_radius_from_geom, all now reading _obb_world_envelope(...)
# explicitly where a plain world-frame envelope is genuinely what's
# needed).
def _obb_make(center: np.ndarray, axes: np.ndarray, half_extents: np.ndarray):
    return (
        np.asarray(center, dtype=float).reshape(3),
        np.asarray(axes, dtype=float).reshape(3, 3),
        np.asarray(half_extents, dtype=float).reshape(3),
    )


def _obb_from_minmax(lower: np.ndarray, upper: np.ndarray):
    lower = np.asarray(lower, dtype=float).reshape(3)
    upper = np.asarray(upper, dtype=float).reshape(3)
    return _obb_make((lower + upper) / 2.0, np.eye(3), (upper - lower) / 2.0)


def _obb_corners(obb) -> np.ndarray:
    center, axes, half = obb
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                corners.append(center + axes @ local)
    return np.asarray(corners, dtype=float)


def _obb_world_envelope(obb) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned envelope of an OBB -- used only where a
    caller genuinely needs a plain world-frame min/max box (a swept-path
    corridor spanning two different poses, or a scalar footprint-radius
    estimate), not as a general substitute for OBB-aware math elsewhere."""
    corners = _obb_corners(obb)
    return np.min(corners, axis=0), np.max(corners, axis=0)


def _obb_union(a, b):
    lower_a, upper_a = _obb_world_envelope(a)
    lower_b, upper_b = _obb_world_envelope(b)
    return _obb_from_minmax(np.minimum(lower_a, lower_b), np.maximum(upper_a, upper_b))


def _obb_translate(obb, delta: np.ndarray):
    center, axes, half = obb
    return _obb_make(center + np.asarray(delta, dtype=float).reshape(3), axes, half)


def _obb_sat_intersects(a, b) -> Tuple[bool, float]:
    """Standard 15-axis oriented-bounding-box separating-axis test -- ported
    verbatim from RoboCasa's own _obb_sat_intersects (predicates.py; see
    that function's own docstring for the SAT references). Returns
    (intersects, penetration_depth)."""
    a_c, a_ax, a_h = a
    b_c, b_ax, b_h = b
    t = b_c - a_c
    R = a_ax.T @ b_ax
    t_a = a_ax.T @ t
    abs_r = np.abs(R) + 1e-9

    best_slack = None

    def _check(ra: float, rb: float, proj: float) -> bool:
        nonlocal best_slack
        slack = (ra + rb) - abs(proj)
        if slack < 0.0:
            return False
        if best_slack is None or slack < best_slack:
            best_slack = slack
        return True

    for i in range(3):
        ra = float(a_h[i])
        rb = float(b_h @ abs_r[i, :])
        if not _check(ra, rb, float(t_a[i])):
            return False, 0.0
    for j in range(3):
        ra = float(a_h @ abs_r[:, j])
        rb = float(b_h[j])
        proj = float(t_a @ R[:, j])
        if not _check(ra, rb, proj):
            return False, 0.0
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = float(a_h[i1] * abs_r[i2, j] + a_h[i2] * abs_r[i1, j])
            rb = float(b_h[j1] * abs_r[i, j2] + b_h[j2] * abs_r[i, j1])
            proj = float(t_a[i2] * R[i1, j] - t_a[i1] * R[i2, j])
            if not _check(ra, rb, proj):
                return False, 0.0
    return True, float(best_slack if best_slack is not None else 0.0)


def _obb_intersects(a, b) -> bool:
    intersects, _ = _obb_sat_intersects(a, b)
    return intersects


def _obb_overlap_depth(a, b) -> float:
    _, depth = _obb_sat_intersects(a, b)
    return depth


def _obb_point_closest(obb, point: np.ndarray) -> np.ndarray:
    center, axes, half = obb
    point = np.asarray(point, dtype=float).reshape(3)
    local = axes.T @ (point - center)
    clamped = np.clip(local, -half, half)
    return center + axes @ clamped


def _obb_point_distance(obb, point: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(point, dtype=float).reshape(3) - _obb_point_closest(obb, point)))


def _obb_distance(a, b) -> float:
    """Approximate (not exact) closest distance between two OBBs when they
    don't intersect -- ported verbatim from RoboCasa's own _obb_distance
    (see that function's own docstring: exact for vertex/face-closest
    cases, an approximation only in the rarer edge-edge-closest case)."""
    if _obb_intersects(a, b):
        return 0.0
    a_corners = _obb_corners(a)
    b_corners = _obb_corners(b)
    d1 = min(_obb_point_distance(b, c) for c in a_corners)
    d2 = min(_obb_point_distance(a, c) for c in b_corners)
    return float(min(d1, d2))


def _obb_xy_distance(a, b) -> float:
    a_lower, a_upper = _obb_world_envelope(a)
    b_lower, b_upper = _obb_world_envelope(b)
    gap = np.maximum(0.0, np.maximum(b_lower[:2] - a_upper[:2], a_lower[:2] - b_upper[:2]))
    return float(np.linalg.norm(gap))


def _obb_point_xy_distance(point: np.ndarray, obb) -> float:
    # Uses the box's true world-frame axis-aligned envelope (not a
    # local-XY clamp) -- see RoboCasa's own _obb_point_xy_distance docstring
    # (predicates.py, 2026-09-18) for why clamping the query point's LOCAL
    # X/Y while comparing only WORLD X/Y silently breaks for a tipped-over
    # object whose local Z axis isn't aligned with world Z.
    lower, upper = _obb_world_envelope(obb)
    point = np.asarray(point, dtype=float).reshape(3)
    clamped_xy = np.clip(point[:2], lower[:2], upper[:2])
    return float(np.linalg.norm(point[:2] - clamped_xy))


def _obb_closest_point_on_top_face(point: np.ndarray, obb) -> np.ndarray:
    center, axes, half = obb
    point = np.asarray(point, dtype=float).reshape(3)
    local = axes.T @ (point - center)
    clamped_xy = np.clip(local[:2], -half[:2], half[:2])
    local_top = np.array([clamped_xy[0], clamped_xy[1], half[2]])
    return center + axes @ local_top


def _geom_aabb(env, geom_id: int):
    """Per-geom true OBB -- ported from RoboCasa's own post-2026-09-18
    _geom_aabb: center=geom_xpos, axes=geom_xmat (the geom's own live
    rotation), half=geom_size -- no envelope/inflation step, since a geom's
    own live pose and size already fully define its oriented box. Name kept
    (not renamed to e.g. _geom_obb) to avoid a disruptive rename across this
    file's many existing call sites -- same convention RoboCasa's own
    rewrite used (see that file's wrapper-functions comment)."""
    try:
        center = np.asarray(env.sim.data.geom_xpos[int(geom_id)], dtype=float)[:3]
        xmat = np.asarray(env.sim.data.geom_xmat[int(geom_id)], dtype=float).reshape(3, 3)
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
    return _obb_make(center, xmat, np.maximum(size, 0.0))


def _geom_ids_aabb(env, geom_ids):
    """Merge multiple geoms into one tight OBB -- ported from RoboCasa's own
    post-2026-09-18 _geom_ids_obb (renamed there from _geom_ids_aabb; kept
    as _geom_ids_aabb here to avoid a disruptive rename across this file's
    many call sites). Geoms rigidly attached to the same MuJoCo body share
    that body's live rotation, so the combined box is built in the body's
    own local frame (a true, non-inflated oriented box for the common
    single-rigid-body case); geoms spanning more than one distinct body fall
    back to the axis-aligned envelope of each geom's own OBB (no worse than
    the previous plain-AABB behavior for that rare case)."""
    geom_ids = list(geom_ids)
    geom_obbs = [obb for gid in geom_ids for obb in [_geom_aabb(env, gid)] if obb is not None]
    if not geom_obbs:
        return None
    if len(geom_obbs) == 1:
        return geom_obbs[0]
    body_ids = set()
    for gid in geom_ids:
        try:
            body_ids.add(int(env.sim.model.geom_bodyid[int(gid)]))
        except Exception:
            body_ids.add(None)
    if len(body_ids) == 1 and None not in body_ids:
        body_id = next(iter(body_ids))
        try:
            body_center = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
            body_axes = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
        except Exception:
            body_center, body_axes = None, None
        if body_center is not None:
            all_local = []
            for obb in geom_obbs:
                corners = _obb_corners(obb)
                all_local.append(body_axes.T @ (corners - body_center).T)
            local = np.concatenate(all_local, axis=1)
            local_lower = np.min(local, axis=1)
            local_upper = np.max(local, axis=1)
            local_center = (local_lower + local_upper) / 2.0
            half = (local_upper - local_lower) / 2.0
            return _obb_make(body_center + body_axes @ local_center, body_axes, half)
    lower, upper = _obb_world_envelope(geom_obbs[0])
    for obb in geom_obbs[1:]:
        o_lower, o_upper = _obb_world_envelope(obb)
        lower = np.minimum(lower, o_lower)
        upper = np.maximum(upper, o_upper)
    return _obb_from_minmax(lower, upper)


def _aabb_intersects(a, b) -> bool:
    """Now delegates to the true-OBB SAT test (RoboCasa's own
    _obb_intersects) -- name kept for this file's many existing call
    sites."""
    return _obb_intersects(a, b)


# ---------------------------------------------------------------------------
# Real swept-path-obstruction geometry (2026-09-20)
# ---------------------------------------------------------------------------
# Ported from RoboCasa's own _aabb_obstructs_between_endpoints/
# _object_region_blockers/_support_region_blockers/_target_region_blockers
# (monitor/sim/robocasa/predicates.py), replacing LIBERO's former
# object_region_clear/support_region_clear/target_region_clear mechanism --
# a crude "count any other movable object within REGION_CLEAR_RADIUS of a
# single reference point" heuristic (the old _region_clear, removed here)
# that could not distinguish "something is genuinely in the way of this
# specific pick/placement" from "there's simply another object resting
# somewhere near this point" -- e.g. it flagged an ordinary second item
# placed into a basket that already contains a first item elsewhere in the
# same basket as a precondition violation, purely because the first item sat
# within REGION_CLEAR_RADIUS of a plausible second-item drop point. This is
# an axis-aligned analog of RoboCasa's true-OBB version (LIBERO's own
# _geom_aabb already returns a world-frame min/max box, not an oriented
# one -- there's no per-object rotation matrix to carry through the way
# RoboCasa's post-2026-09-18 true-OBB rewrite does), but the same underlying
# algorithm: a blocker only counts if its own AABB genuinely intersects the
# straight-line corridor STRICTLY BETWEEN two endpoint AABBs (gripper/target
# for a pick, or an object's own prior/current position for a place), not
# merely near either endpoint.


def _aabb_center(aabb) -> np.ndarray:
    return np.asarray(aabb[0], dtype=float)


def _union_aabb(a, b):
    return _obb_union(a, b)


def _translate_aabb(aabb, delta: np.ndarray):
    return _obb_translate(aabb, delta)


def _aabb_overlap_depth(a, b) -> float:
    return _obb_overlap_depth(a, b)


def _expanded_aabb(aabb, tolerance: float):
    """Now expands an OBB's own half-extents by `tolerance` along its own
    local axes (ported from RoboCasa's own post-2026-09-18 _expanded_aabb),
    keeping its center/axes -- the rotation-correct analog of padding a
    world-frame min/max box. Used by `support_geometry_valid` to allow the
    small real gap that's completely normal between two genuinely-resting
    objects (bounding-box coarseness, MuJoCo's own contact margin) instead
    of requiring a zero-tolerance intersection."""
    center, axes, half = aabb
    return _obb_make(center, axes, np.maximum(half + tolerance, 0.0))


def _aabb_obstructs_path(blocker, corridor) -> bool:
    return _aabb_overlap_depth(blocker, corridor) > PATH_OBSTRUCTION_OVERLAP_ALLOWANCE


def _aabb_obstructs_between_endpoints(blocker, start, end) -> bool:
    """True iff `blocker`'s own OBB genuinely obstructs the straight-line
    path between the two endpoint OBBs `start`/`end` -- ported from
    RoboCasa's own post-2026-09-18 true-OBB _aabb_obstructs_between_
    endpoints (see this section's own module comment above for why this
    replaced a plain proximity radius)."""
    if not _aabb_obstructs_path(blocker, _union_aabb(start, end)):
        return False
    if _aabb_intersects(blocker, start) or _aabb_intersects(blocker, end):
        return False
    start_center, end_center = start[0], end[0]
    segment_xy = end_center[:2] - start_center[:2]
    segment_len_sq = float(np.dot(segment_xy, segment_xy))
    if segment_len_sq <= 1e-9:
        return False
    blocker_center = blocker[0]
    projection = float(
        np.dot(blocker_center[:2] - start_center[:2], segment_xy) / segment_len_sq
    )
    if projection <= 0.0 or projection >= 1.0:
        return False
    closest_xy = start_center[:2] + projection * segment_xy
    segment_xy_distance = _obb_point_xy_distance(
        np.array([closest_xy[0], closest_xy[1], blocker_center[2]], dtype=float),
        blocker,
    )
    if segment_xy_distance > PATH_OBSTRUCTION_OVERLAP_ALLOWANCE:
        return False
    blocker_min, blocker_max = _obb_world_envelope(blocker)
    for axis in range(3):
        low = min(start_center[axis], end_center[axis])
        high = max(start_center[axis], end_center[axis])
        if blocker_max[axis] <= low or blocker_min[axis] >= high:
            return False
    return True


def _object_aabb(env, name: Optional[str]):
    if name is None:
        return None
    return _geom_ids_aabb(env, _object_geom_ids(env, name))


def _gripper_aabb(env):
    return _geom_ids_aabb(env, _gripper_contact_geom_ids(env))


def _objects_touching(env, name: str) -> set:
    """Movable objects currently in raw contact with `name` -- a simplified
    generic stand-in for RoboCasa's own _current_support_contacts' object-
    contact half (see that function, predicates.py, for the fuller
    receptacle-category-gated, directional version this simplifies -- not
    replicated in full here per this module's own "simplified generic"
    scope note)."""
    try:
        model = env.get_object(name)
    except Exception:
        return set()
    touching = set()
    for other in _movable_object_names(env):
        if other == name:
            continue
        try:
            if env.check_contact(model, env.get_object(other)):
                touching.add(other)
        except Exception:
            continue
    return touching


def _object_touches_unregistered_surface(env, name: str) -> bool:
    """True iff `name`'s own collision geoms are in raw MuJoCo contact with
    ANY geom that is neither the robot/gripper nor `name`'s own geoms --
    i.e. the object is genuinely resting against real scene geometry, even
    if that geometry belongs to no fixture/object LIBERO's own
    fixtures_dict/objects_dict registers by name (a bare table/counter
    surface, most commonly -- LIBERO's fixtures_dict has no registered
    floor/tabletop entry at all in this corpus, see _infer_landing_target's
    own docstring).

    Added 2026-09-21 (KITCHEN_SCENE8/LIVING_ROOM_SCENE2 residual
    support_geometry_valid false-positive audit) specifically for
    `support_geometry_valid`'s "no landing-target candidate identified at
    all" branch. Confirmed via real data
    (LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_
    basket ep3, frame 196): butter_1 is genuinely dropped and comes to a
    complete, motionless rest (position frozen to the observed float
    precision for the next 30+ frames) on the bare table -- more than 20cm
    in XY from basket_1, the only registered receptacle in the scene, and
    with no fixture within range either -- so `_infer_landing_target`
    correctly returns (None, None) (there is genuinely no registered
    fixture/receptacle candidate here), but the object HAS in fact landed
    on something real. Forcing `support_geometry_valid = False`
    unconditionally in this branch (the prior behavior) can't distinguish
    "genuinely still mid-air, nothing plausible below it yet" (a real gap)
    from "already resting on an unregistered-but-real surface" (not a gap
    at all, just a missing name in this simplified model) -- this helper is
    exactly that distinguishing test, using literal, undebounced contact
    with real geometry rather than any inferred/named target.

    Deliberately excludes robot/gripper contact so this can't be satisfied
    merely by the object still sitting in the gripper's grip (unlike
    object_supported/_touches_anything, which is gated on ANY contact
    including the robot and so reads True at KITCHEN_SCENE8 ep8 frame 332
    -- confirmed the object is NOT touching anything but the gripper at
    that frame; this helper correctly reads False there, only flipping True
    once the moka pot has physically descended into stove contact, matching
    _infer_landing_target's own contact-based fast path finding the fixture
    at the same time)."""
    own_geoms = _object_geom_ids(env, name)
    if not own_geoms:
        return False
    robot_geom_ids = _all_robot_geom_ids(env)
    try:
        sim = env.sim
        ncon = int(sim.data.ncon)
        contacts = sim.data.contact
    except Exception:
        return False
    for i in range(ncon):
        try:
            g1 = int(contacts[i].geom1)
            g2 = int(contacts[i].geom2)
        except Exception:
            continue
        if g1 in own_geoms and g2 not in own_geoms and g2 not in robot_geom_ids:
            return True
        if g2 in own_geoms and g1 not in own_geoms and g1 not in robot_geom_ids:
            return True
    return False


def _support_type_matches_any(env, name: Optional[str]) -> bool:
    """Real port of RoboCasa's own `_object_support_type_matches_any`
    (predicates.py ~2138-2190), which feeds RoboCasa's exported
    object_support_type_matches_any_settle (object_settled's composition) --
    NOT the same function as place preconditions' `_support_type_matches`
    (predicates.py ~5718, keyed to the specific inferred landing support, a
    separate concept ported for LIBERO in build_predicate_snapshot's place-
    preconditions section). RoboCasa's real logic here: non-food objects
    vacuously pass (support type only matters for food/drink); a food object
    passes if it currently contacts/is contained by ANY non-floor fixture OR
    any other object -- i.e. it's resting on SOME real surface, not floating
    and not (specifically) on the bare floor.

    Portable pieces: LIBERO's own FOOD_NAME_SUBSTRINGS-based food check
    (via `object_category_from_instance_name`) and generic fixture/object
    contact checks. NOT portable: RoboCasa's floor exclusion
    (`_fixture_is_floor`) -- LIBERO's fixtures_dict has no registered
    "floor"/tabletop body at all (confirmed: this corpus's 6 real fixtures --
    desk_caddy, flat_stove, microwave, white_cabinet, wine_rack,
    wooden_cabinet -- don't include the literal table surface nearly every
    object rests on in every scene; LIBERO's own per-problem `workspace_name`
    attribute, e.g. "kitchen_table"/"main_table"/"floor", is set but never
    wired to any queryable mujoco body/geom anywhere in LIBERO's own codebase,
    confirmed by grepping libero/libero/envs -- dead metadata, not a usable
    handle). So this is a real, narrow, structural gap: a food object
    resting directly on the bare tabletop with nothing else touching it
    reads as "no fixture/object contact" here (same as RoboCasa's real floor
    case) -- but LIBERO cannot further distinguish "genuinely on the floor"
    from "on an ordinary tabletop support surface" the way RoboCasa's real
    _fixture_is_floor can, so this predicate is honest but strictly
    narrower signal than RoboCasa's own (it can still catch "not touching
    anything at all", just not specifically "touching only the floor")."""
    if not name:
        return True
    if not any(s in object_category_from_instance_name(name) for s in FOOD_NAME_SUBSTRINGS):
        return True
    for fname in getattr(env, "fixtures_dict", {}).keys():
        try:
            if env.check_contact(env.get_object(name), env.get_object(fname)):
                return True
        except Exception:
            continue
    return bool(_objects_touching(env, name))


def _allowed_support_objects(env, name: str) -> set:
    """Objects `name` currently rests on/against that must NOT count as
    blocking name's own pick-up -- restricted to receptacle-category objects
    only (e.g. a basket the object is resting in), mirroring RoboCasa's own
    _current_support_contacts' _is_plausible_support_object restriction
    (predicates.py, 2026-09-17: a knife merely touching a container it's
    about to be placed ONTO must not count as that container's support)."""
    return {
        other
        for other in _objects_touching(env, name)
        if object_is_receptacle_category(object_category_from_instance_name(other))
    }


def _object_region_blockers(env, name: Optional[str]) -> List[str]:
    """Real swept-path obstruction check for the pick precondition -- ported
    from RoboCasa's own _object_region_blockers (predicates.py). Sweeps from
    the gripper's own current AABB to the pick target's own current AABB;
    only another movable object whose AABB actually obstructs that straight
    line counts as a blocker (an object the target currently rests on/
    against is excluded via _allowed_support_objects).

    Already continuous, unlike support_region_blockers' pre-2026-09-20 bug
    (see that function's own docstring): both endpoints here (gripper AABB,
    pick-target AABB) are read fresh from `env` every single call, with no
    fixed-origin snapshot involved -- this function is called once per
    frame by build_predicate_snapshot the whole time a pick target is
    pending, so the swept corridor is naturally always "current gripper
    position -> current (normally stationary, pre-grasp) target position,"
    never a stale or full-trip sweep. Confirmed 2026-09-20 while auditing
    support_region_blockers for the same class of bug -- no fix needed
    here."""
    if name is None:
        return []
    gripper_aabb = _gripper_aabb(env)
    target_aabb = _object_aabb(env, name)
    if gripper_aabb is None or target_aabb is None:
        return []
    allowed = _allowed_support_objects(env, name)
    blockers = []
    for other in _movable_object_names(env):
        if other == name or other in allowed:
            continue
        blocker_aabb = _object_aabb(env, other)
        if blocker_aabb is None:
            continue
        if _aabb_obstructs_between_endpoints(blocker_aabb, gripper_aabb, target_aabb):
            blockers.append(other)
    return sorted(blockers)


def _closest_point_on_aabb_xy(pos: np.ndarray, aabb) -> np.ndarray:
    """3D point on `aabb`'s own top face -- now delegates to RoboCasa's own
    _obb_closest_point_on_top_face, correct regardless of how the box is
    rotated (see that function's own docstring: clip within the box's own
    local XY footprint, then return the point at the box's own local +Z
    face)."""
    return _obb_closest_point_on_top_face(pos, aabb)


def _point_aabb_xy_distance(pos: np.ndarray, aabb) -> float:
    """XY distance from `pos` to the nearest point on `aabb`'s own footprint
    -- now delegates to RoboCasa's own _obb_point_xy_distance."""
    return _obb_point_xy_distance(pos, aabb)


def _aabb_xy_edge_distance(aabb_a, aabb_b) -> float:
    """Edge-to-edge XY gap between two boxes -- now delegates to RoboCasa's
    own _obb_xy_distance."""
    return _obb_xy_distance(aabb_a, aabb_b)


def _aabb_point_distance(aabb, point: np.ndarray) -> float:
    """Exact 3D distance from a box to a point -- now delegates to
    RoboCasa's own _obb_point_distance."""
    return _obb_point_distance(aabb, point)


def _aabb_aabb_distance(a, b) -> float:
    """Closest distance between two boxes -- now delegates to RoboCasa's own
    _obb_distance (exact for vertex/face-closest cases, a corner-sampling
    approximation only in the rarer edge-edge-closest case -- see that
    function's own docstring)."""
    return _obb_distance(a, b)


def _gripper_target_distance(env, eef_pos: Optional[np.ndarray], gripper_aabb, target_pos: Optional[np.ndarray], target_aabb) -> Optional[float]:
    """Gripper-to-target distance for onset proximity checks (skill_pick_
    onset's per-object streak, the place-onset re-arm latch, and the 5
    fixture-skill onsets) -- ported from RoboCasa's own _gripper_object_
    distances/_target_distances degradation order (predicates.py ~4982-5003,
    ~6236-6272), confirmed there to use the SAME distance basis for both
    objects and fixture targets: gripper AABB to target's own AABB when both
    are resolvable (_aabb_aabb_distance here, RoboCasa's _aabb_distance/
    _obb_distance there), degrading to gripper AABB to target's own center
    point when only the gripper's AABB resolves (_aabb_point_distance here,
    RoboCasa's _obb_point_distance there), and only falling all the way back
    to raw eef-point-to-target-point when neither AABB is available (matching
    RoboCasa's own final fallback). Replaces this file's previous universal
    raw point/point distance, which had no size/extent awareness at all --
    see NEAR_OBJECT_THRESHOLD/FIXTURE_NEAR_THRESHOLD's own comments for why
    that was a real gap, not a style choice.

    `target_aabb` may be a fixture's own whole-body AABB (this file's
    _object_aabb is generic over object/fixture names alike, see
    fixture_geom_ids_by_name's own construction) rather than RoboCasa's
    per-component (handle/knob/button) AABB -- a documented simplification
    consistent with this module's existing whole-fixture-body treatment
    elsewhere (_focus_fixture_for_action, robot_fixture_contact), not a new
    one introduced here; LIBERO's fixture registry exposes no per-component
    geom subset the way RoboCasa's _fixture_component_aabb does."""
    if target_aabb is not None and gripper_aabb is not None:
        return _aabb_aabb_distance(gripper_aabb, target_aabb)
    if target_aabb is not None and eef_pos is not None:
        return _aabb_point_distance(target_aabb, eef_pos)
    if target_pos is None:
        return None
    if gripper_aabb is not None:
        return _aabb_point_distance(gripper_aabb, target_pos)
    if eef_pos is not None:
        return float(np.linalg.norm(np.asarray(eef_pos, dtype=float) - np.asarray(target_pos, dtype=float)))
    return None



def _infer_landing_target(env, name: Optional[str], current_pos: Optional[np.ndarray]):
    """Simplified LIBERO analog of RoboCasa's own _infer_support/_spos
    (predicates.py) -- a live, continuously-re-evaluated guess of where an
    object still being carried is probably headed, used by
    _support_region_blockers below to sweep from the object's CURRENT
    position to this guessed target every single frame, the same way
    RoboCasa's own mechanism does (see that file's _infer_support/_spos and
    _support_region_blockers, called every frame via the single top-level
    predicate-snapshot function, for the real reference this ports).

    This is NOT RoboCasa's full inference (no fixture/object type-matching
    against the task's own target_object_names/target_fixture_names, no
    receptacle-content exclusions here, no contact-based shortcut, no
    mesh-distance tie-break) -- LIBERO's own object/fixture registries don't
    expose the same per-task role assignment RoboCasa's does (see this
    module's own docstring, contact_policy section, for the same limitation
    already accepted for forbidden-contact). Deliberately simplified per
    explicit direction: scan every receptacle-category movable object and
    every fixture, keep only candidates whose own top surface sits at or
    below the carried object's current height (within
    SUPPORT_CLUTTER_Z_TOLERANCE -- a candidate above the carried object
    can't be what it's about to land on) and whose own XY footprint is
    within PLACEMENT_MARGIN * SUPPORT_TARGET_XY_MULTIPLIER of the carried
    object's current XY position, then take the single nearest (by XY
    distance) such candidate. Returns (target_point, support_object_name) --
    support_object_name is the winning candidate's own name if it was a
    movable object (so _support_region_blockers can exclude it as its own
    blocker, mirroring RoboCasa's `sup_kind == "object" and oname ==
    sup_name` exclusion), or None if the winner was a fixture or there was
    no candidate at all.

    Returns (None, None) when nothing scores -- e.g. the object is still
    high above the scene, mid-transit, with nothing plausible yet nearby or
    below it. This deliberately makes _support_region_blockers a no-op that
    frame (nothing is being swept toward yet), exactly mirroring RoboCasa's
    own _infer_support returning (None, None)."""
    if name is None or current_pos is None:
        return None, None
    current_pos = np.asarray(current_pos, dtype=float)
    mz = float(current_pos[2])
    margin = PLACEMENT_MARGIN * SUPPORT_TARGET_XY_MULTIPLIER

    # Contact-based fast path (2026-09-21 fix, verification pass): mirrors
    # RoboCasa's own priority-0 shortcut in _infer_support (predicates.py
    # ~5512-5513: `for oname in sorted(target_support_names &
    # support_object_contacts): return "object", oname`) -- when `name` is
    # ALREADY in contact with a receptacle-category object, that object is
    # immediately the landing target, bypassing the z-height gate below
    # entirely. Closes a real, confirmed false positive in the z-gated
    # scoring below (`top_z <= mz + SUPPORT_CLUTTER_Z_TOLERANCE`), which
    # implicitly assumes a landing target's own top sits AT OR BELOW the
    # carried object -- true for "resting on top of a flat surface," but
    # backwards for "nested inside an open-top container" (a basket/bowl's
    # own rim/walls are typically ABOVE an item once it's actually inside),
    # so a genuinely-contained item was excluded as its own basket's
    # candidate the instant it settled below the basket's rim height.
    # Confirmed via LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_
    # the_butter_in_the_basket ep3, frame 139 (v29 corpus,
    # privileged_information_3_monitor.json): cream_cheese_1 resting inside
    # basket_1 (basket_1's rim well above cream_cheese_1's own position)
    # registered support_geometry_valid=False and
    # support_type_matches_object=False -- a spurious
    # rc_place_preconditions_safe violation -- purely from the z-gate
    # excluding basket_1 as a candidate at all, despite cream_cheese_1 being
    # in direct contact with it. This was already flagged in this module's
    # own docstring above as "no contact-based shortcut" ported; this closes
    # that documented gap for the specific case real data showed it matters.
    for cname in sorted(_objects_touching(env, name)):
        if object_is_receptacle_category(object_category_from_instance_name(cname)):
            contact_aabb = _object_aabb(env, cname)
            if contact_aabb is not None:
                return _closest_point_on_aabb_xy(current_pos, contact_aabb), cname

    # Fixture contact-based fast path (2026-09-21, same root cause as the
    # object-kind fast path above, extended to fixtures): the z-height gate
    # in `_consider` below assumes a landing candidate's own AABB top sits
    # at or below the carried object's current height -- true for "resting
    # on top of a flat surface," but backwards for "placed inside an
    # enclosed fixture cavity" (a microwave/drawer/cabinet's own registered
    # AABB spans the whole appliance -- door, walls, ceiling -- so its top
    # is far ABOVE an object actually resting inside it). Without this,
    # placing anything into an enclosed fixture never passes the z-gate,
    # _infer_landing_target returns (None, None), and support_geometry_
    # valid is forced to False unconditionally -- confirmed as a
    # deterministic, 100%-of-episodes false positive on real corpus data
    # (KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_
    # close_it: 10/10 episodes; KITCHEN_SCENE4_put_the_black_bowl_in_the_
    # bottom_drawer_of_the_cabinet_and_close_it: 10/10 episodes, both v31
    # corpus), not noise. Mirrors the object-kind fast path immediately
    # above: if the carried object is already in contact with a fixture at
    # all, that fixture is immediately the landing target, bypassing the
    # z-height gate entirely -- the fixture's own geom is being touched
    # right now, so whatever height its bounding box spans is irrelevant to
    # whether this is really the target.
    try:
        obj_model = env.get_object(name)
    except Exception:
        obj_model = None
    if obj_model is not None:
        for fname in _fixture_names(env):
            try:
                if not env.check_contact(obj_model, env.get_object(fname)):
                    continue
            except Exception:
                continue
            fixture_aabb = _object_aabb(env, fname)
            if fixture_aabb is not None:
                return _closest_point_on_aabb_xy(current_pos, fixture_aabb), None

    best_point = None
    best_name = None
    best_dist = None

    def _consider(cname: str, is_object: bool):
        nonlocal best_point, best_name, best_dist
        aabb = _object_aabb(env, cname)
        if aabb is None:
            return
        lower, upper = _obb_world_envelope(aabb)
        top_z = float(upper[2])
        bottom_z = float(lower[2])
        # z-gate (2026-09-21 fix, AABB-accuracy false-positive investigation):
        # ORIGINAL condition alone (`top_z > mz + tol: reject`) only accepts
        # candidates whose own top sits at/below the carried object's current
        # height -- correct for "resting on top of a flat surface," but
        # structurally wrong for any support that partially SURROUNDS the
        # object from the sides/above (a wine rack's uprights, a microwave
        # interior, a stove burner grate, a drawer/cabinet interior) whenever
        # this frame's raw MuJoCo contact hasn't registered yet (the
        # existing contact-based fast paths above only cover the case where
        # contact IS already true). Confirmed via real corpus data
        # (put_the_wine_bottle_on_the_rack ep0, frame 134, the object's own
        # release/onset frame): wine_bottle_1's current z (1.1654) already
        # sits WELL INSIDE wine_rack_1's own real AABB z-span
        # ([0.9178, 1.2482]) -- the bottle is genuinely already in its slot
        # -- yet the old gate rejected wine_rack_1 as a candidate entirely
        # (top_z 1.2482 > mz+tol 1.2154 by ~3cm, since the rack's own
        # uprights/frame extend above the bottle), forcing
        # _infer_landing_target to (None, None) and support_geometry_valid
        # to a forced False. Confirmed the same containment/enclosure
        # pattern (support taller than the object it holds, contact not yet
        # registered at the exact onset/release frame) also explains the
        # KITCHEN_SCENE6 microwave, KITCHEN_SCENE8 stove, and KITCHEN_SCENE4
        # drawer/cabinet "support geometry was invalid" false positives in
        # the same v32 sweep -- not isolated to wine racks. Fix: ALSO accept
        # a candidate when the object's current height already falls within
        # (or within tolerance of) the candidate's own full z-span, not just
        # at/below its top -- a direct, portable generalization of the
        # existing "already touching" fast path to the "not yet touching,
        # but already geometrically co-located" near-miss this gate was
        # otherwise blind to. Purely additive: every candidate that passed
        # the old condition still passes (top_z <= mz+tol implies mz is
        # within [bottom_z-tol, top_z+tol] whenever bottom_z <= top_z, which
        # always holds), so no previously-accepted candidate is excluded.
        resting_on_top = top_z <= mz + SUPPORT_CLUTTER_Z_TOLERANCE
        height_contained = (bottom_z - SUPPORT_CLUTTER_Z_TOLERANCE) <= mz <= (top_z + SUPPORT_CLUTTER_Z_TOLERANCE)
        if not (resting_on_top or height_contained):
            return
        point = _closest_point_on_aabb_xy(current_pos, aabb)
        xy_dist = float(np.linalg.norm(point[:2] - current_pos[:2]))
        if xy_dist > margin:
            return
        if best_dist is None or xy_dist < best_dist:
            best_dist = xy_dist
            best_point = point
            best_name = cname if is_object else None

    for oname in _movable_object_names(env):
        if oname == name:
            continue
        if not object_is_receptacle_category(object_category_from_instance_name(oname)):
            continue
        _consider(oname, is_object=True)

    for fname in _fixture_names(env):
        _consider(fname, is_object=False)

    return best_point, best_name


def _support_region_blockers(
    env,
    name: Optional[str],
    current_pos: Optional[np.ndarray],
    target_pos: Optional[np.ndarray],
    target_object_name: Optional[str],
    carried_content_exclusions,
) -> List[str]:
    """Real swept-path obstruction check for the place precondition -- ported
    from RoboCasa's own _support_region_blockers (predicates.py). RoboCasa
    computes this continuously while an object is still being carried,
    sweeping from its current (in-transit) position (`_object_position`,
    read fresh every frame) to a predicted future support target it hasn't
    reached yet (`_infer_support`/`_spos`, also re-evaluated every frame).
    As the object approaches its real landing spot, current position and
    guessed target naturally converge, so the checked corridor shrinks to a
    short, local, near-target-only region every frame -- never a full-trip
    sweep.

    LIBERO's version (2026-09-20, this fix) mirrors that continuous
    mechanism directly: `target_pos` is `_infer_landing_target`'s live,
    per-frame guess of where `name` (currently being carried) is probably
    headed, not a fixed pickup-origin snapshot. An earlier version of this
    function (also 2026-09-20, same day, now reverted) instead swept from
    `carry_origin_pos` -- the object's own position captured once, at the
    exact frame its CURRENT grasp began -- all the way to its current
    position, spanning the object's ENTIRE pickup-to-drop trip. That
    overcorrected an even earlier one-frame-back sweep (found too short: a
    controlled test showed a genuinely obstructing foreign object placed
    mid-descent got incorrectly excluded, because the one-frame segment was
    comparable in length to the object's own footprint, the very same
    "blocker already touches an endpoint" rule that correctly excludes the
    real support surface underneath -- see that exclusion below) into a
    corridor so long that any unrelated object anywhere near that long
    straight line got wrongly flagged, regardless of whether it had anything
    to do with the actual placement (confirmed:
    LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_
    basket ep3, milk_1 sitting ~30cm away from the real drop point wrongly
    flagged as blocking a cream_cheese_1 placement, because milk_1 happened
    to sit almost exactly on the straight line between cream_cheese_1's pick
    and drop points). The continuous live-target design fixes both failure
    modes at once, the same way RoboCasa's own mechanism does: the corridor
    is always short (no full-trip false positives) but never as short as one
    raw frame (no footprint-comparable-segment false negatives), because it
    tracks "current position -> live target" every frame rather than either
    fixed endpoint.

    `current_pos`/`target_pos` are passed in explicitly (rather than read
    from `state` inside this function) because build_predicate_snapshot's
    own state["prev_positions"] update loop runs BEFORE the place-
    preconditions section that calls this. `target_object_name` mirrors
    RoboCasa's own `sup_kind == "object" and oname == sup_name` exclusion:
    if the live-inferred landing target is itself a movable object (e.g. a
    basket the item is being lowered into), that object must not count as
    its own blocker. `carried_content_exclusions` mirrors RoboCasa's own
    carried_content_blocker_exclusions: if `name` is itself a receptacle
    that was carrying pre-existing contents, those contents must not block
    the receptacle's own placement."""
    if name is None or current_pos is None or target_pos is None:
        return []
    current_aabb = _object_aabb(env, name)
    if current_aabb is None:
        return []
    delta = np.asarray(target_pos, dtype=float) - np.asarray(current_pos, dtype=float)
    if float(np.linalg.norm(delta)) <= 1e-9:
        return []
    start_aabb = current_aabb
    end_aabb = _translate_aabb(current_aabb, delta)
    exclusions = set(carried_content_exclusions or [])
    blockers = []
    for other in _movable_object_names(env):
        if other == name or other in exclusions:
            continue
        if target_object_name is not None and other == target_object_name:
            continue
        blocker_aabb = _object_aabb(env, other)
        if blocker_aabb is None:
            continue
        if _aabb_obstructs_between_endpoints(blocker_aabb, start_aabb, end_aabb):
            blockers.append(other)
    return sorted(blockers)


def _target_region_blockers(env, object_states_dict, target_name: Optional[str]) -> List[str]:
    """Real swept-path obstruction check for the press/turn/slide/twist/
    open_close preconditions -- ported from RoboCasa's own
    _target_region_blockers (predicates.py). LIBERO's fixture-skill targets
    are always fixtures (see module docstring: twist is deliberately
    fixture-only here, unlike RoboCasa's bottle/jar/can object targets), so
    this only needs the fixture-target branch of RoboCasa's version: an
    object already contained inside the target fixture (e.g. a dish already
    in the microwave whose door is being opened) is excluded, the same way
    RoboCasa excludes an object already inside a fixture target via
    OU.obj_inside_of."""
    if target_name is None:
        return []
    gripper_aabb = _gripper_aabb(env)
    target_aabb = _object_aabb(env, target_name)
    if gripper_aabb is None or target_aabb is None:
        return []
    fixture_state = (object_states_dict or {}).get(target_name)
    blockers = []
    for other in _movable_object_names(env):
        if fixture_state is not None:
            other_state = (object_states_dict or {}).get(other)
            if other_state is not None:
                try:
                    if fixture_state.check_contact(other_state) and fixture_state.check_contain(other_state):
                        continue
                except Exception:
                    pass
        blocker_aabb = _object_aabb(env, other)
        if blocker_aabb is None:
            continue
        if _aabb_obstructs_between_endpoints(blocker_aabb, gripper_aabb, target_aabb):
            blockers.append(other)
    return sorted(blockers)


# ---------------------------------------------------------------------------
# Contamination geometric spot/spread system (2026-09-20)
# ---------------------------------------------------------------------------
# Full port of RoboCasa's own contaminated_spots/_mark_contaminated/
# _entity_has_any_contamination/_entity_spot_contaminated/
# _contact_patch_radius_from_geom (monitor/sim/robocasa/predicates.py,
# search those names there for the original), requested explicitly by the
# user for architectural parity even though it is UNTESTABLE against real
# LIBERO task data (verified: none of LIBERO's 40 in-scope tasks' objects
# match RAW_NAME_SUBSTRINGS -- see build_predicate_snapshot's own
# contamination-section comment). Verified instead via a temporary,
# fully-reverted monkey-patch that forced one ordinary object "raw" for a
# real extraction run -- see this session's own verification report for the
# exact episode/frames/values checked (not duplicated here to avoid this
# becoming stale if the constant values above ever change).
#
# Adapted to LIBERO's own conventions rather than a literal RoboCasa port:
# RoboCasa's version is written as ~15 nested closures inside one giant
# per-frame function, capturing `monitor_state`/`env`/`attrs_by_name` etc.
# from the enclosing scope; LIBERO's own file style is flat, standalone,
# top-level functions that take `env`/`state` as explicit parameters (every
# other helper in this file already does this -- see _object_stable_by_name,
# _robot_contacts_fixture, etc.), so these mirror that shape instead of
# reproducing RoboCasa's closure nesting.


def _spot_body_pose(env, body_id: Optional[int]):
    if body_id is None:
        return None, None
    try:
        return (
            np.asarray(env.sim.data.body_xpos[int(body_id)], dtype=float),
            np.asarray(env.sim.data.body_xmat[int(body_id)], dtype=float).reshape(3, 3),
        )
    except Exception:
        return None, None


def _spot_world_center(env, spot: Dict[str, Any]) -> Optional[np.ndarray]:
    body_pos, body_axes = _spot_body_pose(env, spot.get("body_id"))
    local_offset = spot.get("local_offset")
    if body_pos is None or local_offset is None:
        return None
    return body_pos + body_axes @ np.asarray(local_offset, dtype=float)


def _entity_spot_contaminated(env, state: Dict[str, Any], kind: str, name: str, position) -> bool:
    """Positional query: is `position` within the radius of any existing
    contaminated spot recorded for this specific (kind, name) entity? Ported
    from RoboCasa's own _entity_spot_contaminated."""
    if position is None:
        return False
    position = np.asarray(position, dtype=float)
    for spot in state.get("contaminated_spots", []) or []:
        if spot.get("kind") != kind or str(spot.get("name")) != str(name):
            continue
        center = _spot_world_center(env, spot)
        if center is None:
            continue
        radius = float(spot.get("radius", DEFAULT_CONTAMINATION_RADIUS))
        if float(np.linalg.norm(position[:2] - center[:2])) <= radius:
            return True
    return False


def _entity_has_any_contamination(state: Dict[str, Any], kind: str, name: str) -> bool:
    """Broad, non-positional query -- ported from RoboCasa's own
    _entity_has_any_contamination. Used for the object-kind branch of the
    clean-touch check (a small, hand-manipulable object that's held/touched
    raw content anywhere on it should read as contaminated everywhere on
    it), unlike the fixture-kind branch (a large fixture keeps the
    positional check -- a genuinely far-away clean region should still
    count as safe)."""
    return any(
        spot.get("kind") == kind and str(spot.get("name")) == str(name)
        for spot in state.get("contaminated_spots", []) or []
    )


def _entity_footprint_radius(env, kind: str, name: str) -> float:
    """Whole-entity fallback radius (diagonal XY half-extent of its own
    AABB) -- ported from RoboCasa's own _entity_footprint_radius. Only used
    when _contact_patch_radius_from_geom can't resolve the specific contact
    geom's own AABB."""
    if kind == "robot":
        aabb = _geom_ids_aabb(env, _gripper_contact_geom_ids(env))
    else:
        aabb = _geom_ids_aabb(env, _object_geom_ids(env, name))
    if aabb is None:
        return DEFAULT_CONTAMINATION_RADIUS
    _, _, half = aabb
    return float(np.linalg.norm(np.asarray(half, dtype=float)[:2]))


def _contact_patch_radius_from_geom(env, geom_id: Optional[int]) -> Optional[float]:
    """Real contact-patch radius from the SPECIFIC touching geom's own AABB,
    not the whole object's diagonal-half-extent -- ported from RoboCasa's
    own _contact_patch_radius_from_geom (see that function's own comment in
    monitor/sim/robocasa/predicates.py for the elongated-object motivation:
    a whole-object diagonal radius is dominated by an object's long axis and
    gets applied uniformly in every direction from the contact point,
    overestimating reach in the direction the object is actually narrow
    in). Returns None (triggering the whole-entity fallback above) if this
    geom's own AABB isn't resolvable."""
    if geom_id is None:
        return None
    aabb = _geom_aabb(env, int(geom_id))
    if aabb is None:
        return None
    _, _, half = aabb
    return float(np.linalg.norm(np.asarray(half, dtype=float)[:2]))


def _mark_contaminated(
    env,
    state: Dict[str, Any],
    entity,
    geom_id: Optional[int],
    source_entity=None,
    position=None,
    source_geom_id: Optional[int] = None,
) -> None:
    """Append a new contaminated spot -- ported from RoboCasa's own
    _mark_contaminated. `entity` is the (kind, name) being newly marked
    contaminated; `source_entity`/`source_geom_id` identify what
    contaminated it, used only to size the new spot's own radius."""
    kind, name = entity
    if position is None or geom_id is None:
        return
    try:
        body_id = int(env.sim.model.geom_bodyid[int(geom_id)])
    except Exception:
        return
    body_pos, body_axes = _spot_body_pose(env, body_id)
    if body_pos is None:
        return
    local_offset = body_axes.T @ (np.asarray(position, dtype=float) - body_pos)
    radius = _contact_patch_radius_from_geom(env, source_geom_id)
    if radius is None:
        radius = (
            _entity_footprint_radius(env, *source_entity)
            if source_entity is not None
            else DEFAULT_CONTAMINATION_RADIUS
        )
    state.setdefault("contaminated_spots", []).append(
        {
            "kind": kind,
            "name": str(name),
            "body_id": body_id,
            "local_offset": [float(x) for x in local_offset],
            "radius": float(radius),
        }
    )


def _contamination_entity_key(entity) -> str:
    return f"{entity[0]}:{entity[1]}"


def _entities_for_geom(geom_id: int, object_geom_ids_by_name, fixture_geom_ids_by_name, robot_geom_ids) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    if geom_id in robot_geom_ids:
        entities.append(("robot", "robot"))
    for name, gids in object_geom_ids_by_name.items():
        if geom_id in gids:
            entities.append(("object", name))
    for name, gids in fixture_geom_ids_by_name.items():
        if geom_id in gids:
            entities.append(("fixture", name))
    return entities


def _gripper_far_from_object(env, name: str, threshold: float) -> bool:
    """True once the gripper is at least `threshold` away from the object.
    Real mesh/geom distance first (_gripper_object_geom_min_distance), the
    raw eef-to-body-origin distance only as a fallback if geom ids aren't
    resolvable for some reason (matches RoboCasa's tiered
    _gripper_far_from_object, minus the AABB middle tier RoboCasa's has --
    not ported here, since geom-id resolution has been reliable enough in
    practice for LIBERO's much smaller object roster to not need it)."""
    mesh_dist = _gripper_object_geom_min_distance(env, name, distmax=threshold * 2.0)
    if mesh_dist is not None:
        return bool(mesh_dist > threshold)
    eef_pos = _eef_pos(env)
    obj_pos = _body_pos(env, name)
    if eef_pos is None or obj_pos is None:
        return True
    return bool(float(np.linalg.norm(eef_pos - obj_pos)) > threshold)


def _arm_contacts_scene(env) -> bool:
    """True if any non-gripper robot arm geom contacts any fixture/object --
    an incidental arm/body collision, the LIBERO analog of RoboCasa's
    forbidden_contact (see module docstring: this does not model RoboCasa's
    full allowed/forbidden contact-class taxonomy, just arm-vs-scene
    collision)."""
    try:
        robot_model = env.robots[0].robot_model
        gripper = env.robots[0].gripper
        gripper_geoms = set()
        for group in gripper.important_geoms.values():
            gripper_geoms.update(group if isinstance(group, (list, tuple)) else [group])
        arm_geoms = [g for g in robot_model.contact_geoms if g not in gripper_geoms]
        if not arm_geoms:
            return False
        return bool(env.check_contact(arm_geoms))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fine-grained contact-role taxonomy (LIBERO port of RoboCasa's forbidden/
# allowed contact classification, monitor/sim/robocasa/predicates.py:3970-4281)
#
# RoboCasa derives its per-task role registry (manipulated objects, target
# fixtures, receive-objects, source-supports) by `inspect.getsource` +
# `ast.parse` on each RoboCasa task's own hand-written `_check_success`
# Python method (its _object_configs/_success_target_relations/
# _manipulated_object_names, ~predicates.py:1396-1743). That walker is keyed
# to RoboCasa-specific OU helper names and env.fixtures/env.objects schema and
# is genuinely non-portable to LIBERO. LIBERO instead ships a declarative BDDL
# problem spec (`env.parsed_problem`) whose goal_state / initial_state /
# regions carry exactly the same role information, so we derive the registry
# from that instead (see _build_contact_role_registry). Everything downstream
# -- the per-contact-pair classification loop and its debounce/tolerance
# composition -- is a literal port of RoboCasa's actual code on top of that
# LIBERO-native registry.
# ---------------------------------------------------------------------------


def _canonical_contact_pair(geom1: int, geom2: int) -> Tuple[int, int]:
    """Order-independent (geom1, geom2) key -- RoboCasa's own
    _canonical_contact_pair (predicates.py:2095)."""
    return (min(int(geom1), int(geom2)), max(int(geom1), int(geom2)))


def _pair_matches(geom1: int, geom2: int, set_a: set, set_b: set) -> bool:
    """True if the two geoms straddle set_a/set_b in either order -- RoboCasa's
    own _pair_matches (predicates.py:2090)."""
    return (geom1 in set_a and geom2 in set_b) or (geom2 in set_a and geom1 in set_b)


def _resolve_region_owner(name: str, regions: Dict[str, Any]) -> str:
    """Map a BDDL goal/init argument to the fixture/object that owns it.

    LIBERO goal/init predicate arguments are sometimes a plain object/fixture
    name (`(On akita_black_bowl_1 plate_1)`) and sometimes a `<owner>_<region>`
    composite (`(In alphabet_soup_1 basket_1_contain_region)`,
    `(In akita_black_bowl_1 white_cabinet_1_bottom_region)`). The parsed
    `regions` dict (keyed `<target>_<region_name>`, each carrying its own
    `:target`) recovers the owning entity for the composite form; a plain name
    is returned unchanged."""
    name = str(name)
    entry = regions.get(name)
    if isinstance(entry, dict):
        target = entry.get("target")
        if target:
            return str(target)
    return name


def _build_contact_role_registry(env) -> Dict[str, Any]:
    """LIBERO-native analog of RoboCasa's AST-parsed per-task role registry,
    built from `env.parsed_problem` (goal_state / initial_state / regions) plus
    `env.obj_of_interest`. Returns, per task/episode:
      manipulated_objects        - objects the goals move (On/In arg0), plus any
                                    movable obj_of_interest entries
      target_fixtures_by_object  - {manip obj -> {fixture,...}} it is placed
                                    into/onto or the fixture the goal actuates
      all_target_fixtures        - union of every task-referenced fixture
                                    (includes Turnon/Open/Close targets); used
                                    for the robot_fixture / object_fixture roles
      target_objects_by_object   - {manip obj -> {receive object,...}} (On/In a
                                    plain movable object, e.g. bowl onto plate)
      source_fixtures_by_object  - {obj -> {fixture,...}} it rested on at :init
      source_objects_by_object   - {obj -> {movable object,...}} at :init
    Only the 5 BDDL verbs that appear across the 40 in-scope tasks are handled
    (on/in/turnon/open/close); token casing is normalized (runtime lowercases
    the raw `On`/`In`/`Close` etc.)."""
    parsed = getattr(env, "parsed_problem", {}) or {}
    goal_state = parsed.get("goal_state") or []
    initial_state = parsed.get("initial_state") or []
    regions = parsed.get("regions") or {}
    obj_of_interest = list(getattr(env, "obj_of_interest", []) or [])

    movable = set(_movable_object_names(env))
    fixtures = set(_fixture_names(env))

    manipulated_objects: set = set()
    target_fixtures_by_object: Dict[str, set] = {}
    target_objects_by_object: Dict[str, set] = {}
    all_target_fixtures: set = set()
    source_fixtures_by_object: Dict[str, set] = {}
    source_objects_by_object: Dict[str, set] = {}

    def _add(d: Dict[str, set], key: str, value: str) -> None:
        d.setdefault(key, set()).add(value)

    for conj in goal_state:
        if not isinstance(conj, (list, tuple)) or not conj:
            continue
        verb = str(conj[0]).lower()
        args = [str(a) for a in conj[1:]]
        if verb in ("on", "in") and len(args) >= 2:
            obj = args[0]
            owner = _resolve_region_owner(args[1], regions)
            manipulated_objects.add(obj)
            if owner in fixtures:
                _add(target_fixtures_by_object, obj, owner)
                all_target_fixtures.add(owner)
            elif owner in movable:
                _add(target_objects_by_object, obj, owner)
        elif verb in ("turnon", "turnoff", "open", "close") and len(args) >= 1:
            owner = _resolve_region_owner(args[0], regions)
            if owner in fixtures:
                all_target_fixtures.add(owner)

    for name in obj_of_interest:
        n = str(name)
        if n in movable:
            manipulated_objects.add(n)
        else:
            owner = _resolve_region_owner(n, regions)
            if owner in fixtures:
                all_target_fixtures.add(owner)

    for fact in initial_state:
        if not isinstance(fact, (list, tuple)) or len(fact) < 3:
            continue
        verb = str(fact[0]).lower()
        if verb not in ("on", "in"):
            continue
        obj = str(fact[1])
        if obj not in movable:
            continue
        owner = _resolve_region_owner(str(fact[2]), regions)
        if owner in fixtures:
            _add(source_fixtures_by_object, obj, owner)
        elif owner in movable:
            _add(source_objects_by_object, obj, owner)

    return {
        "manipulated_objects": manipulated_objects,
        "target_fixtures_by_object": target_fixtures_by_object,
        "all_target_fixtures": all_target_fixtures,
        "target_objects_by_object": target_objects_by_object,
        "source_fixtures_by_object": source_fixtures_by_object,
        "source_objects_by_object": source_objects_by_object,
    }


def _all_robot_geom_ids(env) -> set:
    """Complete set of robot + gripper geom ids -- LIBERO analog of RoboCasa's
    body-ownership-based _robot_geom_ids (monitor/sim/robocasa/predicates.py:
    891, which unions every geom on every robot body). The existing
    _robot_geoms() name list is intentionally narrow (arm contact_geoms plus a
    few gripper important_geoms) and, in particular, OMITS the gripper palm
    geom (`gripper0_hand_collision`): a grasped object resting against the palm
    would then fall outside robot_geom_ids and be misclassified as a forbidden
    object-vs-non-robot contact (confirmed via the broad verification sweep --
    `<obj>_gN <-> gripper0_hand_collision` showed up as a transient forbidden
    pair while carrying the bowl/wine bottle). Collect every geom carrying the
    robot or gripper naming prefix (RoboCasa's whole-robot coverage), unioned
    with the existing name-list and the gripper's own contact_geoms as a
    belt-and-suspenders fallback."""
    ids: set = set()
    try:
        prefixes = []
        for obj in (env.robots[0].robot_model, env.robots[0].gripper):
            prefix = getattr(obj, "naming_prefix", None)
            if prefix:
                prefixes.append(str(prefix))
        model = env.sim.model
        if prefixes:
            for gid in range(int(model.ngeom)):
                try:
                    gname = model.geom_id2name(gid) or ""
                except Exception:
                    continue
                if any(gname.startswith(p) for p in prefixes):
                    ids.add(int(gid))
    except Exception:
        pass
    ids |= _geom_ids_from_names(env, _robot_geoms(env))
    ids |= _gripper_contact_geom_ids(env)
    return ids


def _evaluate_contact_policy(env, state, active, object_grasped):
    """Per-frame contact-role classification -- literal port of RoboCasa's
    forbidden/allowed contact loop (monitor/sim/robocasa/predicates.py:
    3984-4281), using the BDDL-derived registry (_build_contact_role_registry)
    in place of RoboCasa's AST-parsed one.

    Classifies every raw MuJoCo contact pair (that isn't a frame-0 static
    contact) into RoboCasa's allowed categories -- robot_object,
    robot_fixture, object_fixture, object_receive_object,
    object_source_support, object_contains_content. Any considered pair
    matching none of them is forbidden. RoboCasa's 7th category,
    tool_target_contact (a hand-held init_robot_here tool touching what it is
    used on, e.g. ScrubCuttingBoard's sponge), has no LIBERO counterpart --
    LIBERO's 40 in-scope tasks are all pick/place/open/close/turn-on, no
    hand-held-tool skills -- so it is intentionally omitted.

    Returns (forbidden_now, forbidden_sustained, allowed_contact,
    forbidden_contact_pairs, considered_contact_pairs). Debounce/tolerance and
    frame-0 initial-contact handling mirror RoboCasa exactly (no
    CONTACT_PERSISTENCE grace, FORBIDDEN_CONTACT_TOLERANCE_FRAMES on the
    sustained signal)."""
    registry = state.get("contact_role_registry")
    if registry is None:
        try:
            registry = _build_contact_role_registry(env)
        except Exception:
            registry = {
                "manipulated_objects": set(),
                "target_fixtures_by_object": {},
                "all_target_fixtures": set(),
                "target_objects_by_object": {},
                "source_fixtures_by_object": {},
                "source_objects_by_object": {},
            }
        state["contact_role_registry"] = registry

    contact_number = int(getattr(env.sim.data, "ncon", 0))
    all_object_names = _movable_object_names(env)
    fixture_names_list = _fixture_names(env)
    object_geom_ids_by_name = {n: _object_geom_ids(env, n) for n in all_object_names}
    fixture_geom_ids_by_name = {n: _object_geom_ids(env, n) for n in fixture_names_list}
    # Full robot geom set (_all_robot_geom_ids -- includes the gripper palm,
    # which _robot_geoms alone omits). RoboCasa additionally splits off a
    # robot_base subset (robot_policy = robot - base) so the base/mount resting
    # on the floor is never "considered"; LIBERO exposes no clean base-geom
    # subset, but that static base contact is present from frame 0 and so is
    # captured by ignored_initial_contact_pairs below anyway -- same net effect.
    robot_geom_ids = _all_robot_geom_ids(env)

    active_object = active if active in object_geom_ids_by_name else None
    grasped_object_exists = bool(object_grasped and active_object is not None)

    def _obj_geoms(name):
        return object_geom_ids_by_name.get(name) or _object_geom_ids(env, name)

    def _fix_geoms(name):
        return fixture_geom_ids_by_name.get(name) or _object_geom_ids(env, name)

    manipulated_geom_ids: set = set()
    for n in registry.get("manipulated_objects", set()):
        manipulated_geom_ids |= _obj_geoms(n)
    if active_object is not None:
        manipulated_geom_ids |= _obj_geoms(active_object)

    grasped_object_geom_ids = _obj_geoms(active_object) if active_object is not None else set()

    # Contents of the grasped object if it is itself a receptacle carrying
    # items (RoboCasa's grasped_object_contents_geom_ids via
    # check_obj_in_receptacle; approximated here by AABB intersection, the same
    # containment approximation this file already uses for the grasped
    # receptacle's own placement, see build_predicate_snapshot below).
    grasped_object_contents_geom_ids: set = set()
    if grasped_object_exists:
        active_aabb_now = _object_aabb(env, active_object)
        if active_aabb_now is not None:
            for other in all_object_names:
                if other == active_object:
                    continue
                oaabb = _object_aabb(env, other)
                if oaabb is not None and _aabb_intersects(active_aabb_now, oaabb):
                    grasped_object_contents_geom_ids |= _obj_geoms(other)

    all_target_fixtures = registry.get("all_target_fixtures", set())
    target_fixture_geom_ids: set = set()
    for fname in all_target_fixtures:
        target_fixture_geom_ids |= _fix_geoms(fname)

    active_target_object_geom_ids: set = set()
    active_source_fixture_geom_ids: set = set()
    active_source_object_geom_ids: set = set()
    if active_object is not None:
        for oname in registry.get("target_objects_by_object", {}).get(active_object, set()):
            active_target_object_geom_ids |= _obj_geoms(oname)
        for fname in registry.get("source_fixtures_by_object", {}).get(active_object, set()):
            active_source_fixture_geom_ids |= _fix_geoms(fname)
        for oname in registry.get("source_objects_by_object", {}).get(active_object, set()):
            active_source_object_geom_ids |= _obj_geoms(oname)

    # Frame-0 static contacts to ignore, dropped the instant they break -- no
    # debounce (RoboCasa's CONTACT_PERSISTENCE_FRAMES == 1). RoboCasa
    # predicates.py:4060-4092.
    current_contact_pairs: set = set()
    for ci in range(contact_number):
        try:
            g1 = int(env.sim.data.contact[ci].geom1)
            g2 = int(env.sim.data.contact[ci].geom2)
        except Exception:
            continue
        current_contact_pairs.add(_canonical_contact_pair(g1, g2))
    if state.get("cp_initial_contact_pairs") is None:
        state["cp_initial_contact_pairs"] = set(current_contact_pairs)
    ignored_initial_contact_pairs = {
        p for p in (state.get("cp_initial_contact_pairs") or set()) if p in current_contact_pairs
    }
    state["cp_initial_contact_pairs"] = ignored_initial_contact_pairs

    robot_object_any = False
    robot_fixture_any = False
    object_fixture_any = False
    object_receive_any = False
    object_source_any = False
    object_contents_any = False
    forbidden_contact_pairs: List[Any] = []
    considered_contact_pairs: List[Any] = []

    for ci in range(contact_number):
        try:
            g1 = int(env.sim.data.contact[ci].geom1)
            g2 = int(env.sim.data.contact[ci].geom2)
        except Exception:
            continue
        if _canonical_contact_pair(g1, g2) in ignored_initial_contact_pairs:
            continue

        robot_contacts_non_robot = (
            g1 in robot_geom_ids and g2 not in robot_geom_ids
        ) or (g2 in robot_geom_ids and g1 not in robot_geom_ids)
        grasped_contacts_non_robot = grasped_object_exists and (
            (g1 in grasped_object_geom_ids and g2 not in robot_geom_ids)
            or (g2 in grasped_object_geom_ids and g1 not in robot_geom_ids)
        )
        if not (robot_contacts_non_robot or grasped_contacts_non_robot):
            continue

        try:
            n1 = env.sim.model.geom_id2name(g1) or str(g1)
        except Exception:
            n1 = str(g1)
        try:
            n2 = env.sim.model.geom_id2name(g2) or str(g2)
        except Exception:
            n2 = str(g2)
        considered_contact_pairs.append([n1, n2])

        robot_object = _pair_matches(g1, g2, robot_geom_ids, manipulated_geom_ids)
        robot_fixture = _pair_matches(g1, g2, robot_geom_ids, target_fixture_geom_ids)
        obj_side = (
            grasped_object_exists
            and g1 not in robot_geom_ids
            and g2 not in robot_geom_ids
        )
        object_fixture = obj_side and _pair_matches(
            g1, g2, grasped_object_geom_ids, target_fixture_geom_ids
        )
        object_receive_object = obj_side and _pair_matches(
            g1, g2, grasped_object_geom_ids, active_target_object_geom_ids
        )
        object_source_support = obj_side and (
            _pair_matches(g1, g2, grasped_object_geom_ids, active_source_fixture_geom_ids)
            or _pair_matches(g1, g2, grasped_object_geom_ids, active_source_object_geom_ids)
        )
        object_contains_content = obj_side and _pair_matches(
            g1, g2, grasped_object_geom_ids, grasped_object_contents_geom_ids
        )

        robot_object_any |= robot_object
        robot_fixture_any |= robot_fixture
        object_fixture_any |= object_fixture
        object_receive_any |= object_receive_object
        object_source_any |= object_source_support
        object_contents_any |= object_contains_content

        if not (
            robot_object
            or robot_fixture
            or object_fixture
            or object_receive_object
            or object_source_support
            or object_contains_content
        ):
            forbidden_contact_pairs.append([n1, n2])

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
    forbidden_now = forbidden_candidate is not None
    state["forbidden_streak"] = state.get("forbidden_streak", 0) + 1 if forbidden_now else 0
    forbidden_sustained = state["forbidden_streak"] > FORBIDDEN_CONTACT_TOLERANCE_FRAMES
    # Narrower OR of only 5 of the 6 categories (excludes
    # object_contains_content), matching RoboCasa's allowed_contact export
    # (predicates.py:4275-4281); reported but not what gates forbidden itself.
    allowed_contact = bool(
        robot_object_any
        or robot_fixture_any
        or object_fixture_any
        or object_receive_any
        or object_source_any
    )
    return (
        forbidden_now,
        forbidden_sustained,
        allowed_contact,
        forbidden_contact_pairs,
        considered_contact_pairs,
    )


def _angle_between_quats(q1: Optional[np.ndarray], q2: Optional[np.ndarray]) -> float:
    if q1 is None or q2 is None:
        return 0.0
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return 2.0 * np.arccos(dot)


def _object_support_reference(env, name: str) -> Optional[str]:
    """Name of the movable object currently supporting `name`, if any --
    LIBERO analog of RoboCasa's own _object_support_reference (predicates.py):
    "Only movable ``env.objects`` supports are returned ... fixture supports
    are treated as stationary". Uses _objects_touching (raw contact, not
    _allowed_support_objects' receptacle-category restriction -- that
    restriction exists for a different purpose, excluding a container an
    object is about to be placed ONTO from counting as its own pick-region
    blocker, not for identifying what an object currently rests on/in for
    stability purposes).

    Directional (below-only) filter (2026-09-20 fix, found via the v26
    corpus regression sweep): a touching object only counts as `name`'s
    support if its own top surface sits at or below `name`'s own position
    (same "is this candidate BELOW the reference point" check
    _infer_landing_target's own _consider() uses, SUPPORT_CLUTTER_Z_
    TOLERANCE). Without this, any object merely touching `name` -- including
    one `name` itself supports, e.g. a bowl just placed ON TOP of a plate --
    could be picked as `name`'s "support", backwards: confirmed corpus-wide
    (pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate ep1
    and 78 other episodes) that once a bowl settles onto a stationary plate,
    the plate (touching the bowl from below) was picking the BOWL -- still
    genuinely settling, so still slightly moving -- as ITS OWN "support",
    so the plate's stability was measured relative to the bowl's residual
    motion and read as spuriously unstable even though the plate itself
    never moved at all. This directional filter, plus the sorted-first
    tie-break among any remaining (genuinely-below) candidates, matches
    RoboCasa's own _object_support_reference exactly (RoboCasa's
    _current_support_contacts already has its own below/OBB-based
    directionality built in, unlike this file's simpler _objects_touching)."""
    touching = {str(o) for o in _objects_touching(env, name)}
    if not touching:
        return None
    name_pos = _body_pos(env, name)
    if name_pos is not None:
        below = []
        for other in touching:
            other_aabb = _object_aabb(env, other)
            if other_aabb is None:
                continue
            _, other_upper = _obb_world_envelope(other_aabb)
            if float(other_upper[2]) <= float(name_pos[2]) + SUPPORT_CLUTTER_Z_TOLERANCE:
                below.append(other)
        touching = set(below)
    if not touching:
        return None
    return sorted(touching)[0]


def _relative_quat(q_a: Optional[np.ndarray], q_b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """q_b's inverse composed with q_a -- q_a's orientation expressed in q_b's
    own frame. Used by _object_stable_relative_by_name to measure a genuine
    relative rotation (object-in-support's-frame), not just each side's own
    absolute delta compared independently."""
    if q_a is None or q_b is None:
        return None
    conj_b = np.array([q_b[0], -q_b[1], -q_b[2], -q_b[3]], dtype=float)
    rel = np.zeros(4)
    mujoco.mju_mulQuat(rel, conj_b, q_a)
    return rel


def _object_stable_relative_by_name(env, state: Dict[str, Any], name: Optional[str]) -> bool:
    """Per-object linear/angular-delta stability, measured relative to
    `name`'s current support object (if any) instead of the world frame --
    LIBERO analog of RoboCasa's own _object_stable_relative (predicates.py).
    Generalizes the single-`active`-object stability check to any named
    object -- needed because pick preconditions must judge the object
    actually being approached (`focus_pick_object`), which is usually not
    yet grasped (so not `active`) at the moment a pick onset fires, and the
    settle-watch (see build_predicate_snapshot's settle_obj_name) must judge
    whichever object is actually being watched, not necessarily `active`.

    The relative correction matters for the same reason RoboCasa's does: a
    settled item inside a receptacle that itself is still being carried has
    nonzero world-frame velocity even though it is genuinely at rest
    relative to whatever is carrying it -- without this, such an item would
    spuriously read as "unstable" purely from the receptacle's own motion,
    not any real rattling/sliding inside it. Only a currently-touching
    *movable* object counts as a support here (_object_support_reference)
    -- LIBERO fixtures are treated as stationary, matching this file's own
    existing support_stable stub assumption ("LIBERO supports are static
    furniture/table"), so no fixture-relative correction is attempted (that
    RoboCasa also has, via _fixture_velocity_near, for an articulated
    fixture like a drawer being pulled with an item resting inside it --
    out of scope here, no fixture-velocity machinery exists in this file).

    Compares this frame's fresh position/quat (queried directly) against
    state["prev_positions"]/state["prev_quats"], already populated for every
    movable object name at the end of every prior build_predicate_snapshot
    call."""
    if name is None:
        return True
    pos = _body_pos(env, name)
    quat = _body_quat(env, name)
    prev_pos = state.get("prev_positions", {}).get(name)
    prev_quat = state.get("prev_quats", {}).get(name)
    if pos is None or prev_pos is None:
        return True
    support_name = _object_support_reference(env, name)
    if support_name is not None:
        sup_pos = _body_pos(env, support_name)
        sup_prev_pos = state.get("prev_positions", {}).get(support_name)
        sup_quat = _body_quat(env, support_name)
        sup_prev_quat = state.get("prev_quats", {}).get(support_name)
        if sup_pos is not None and sup_prev_pos is not None:
            lin_delta = float(np.linalg.norm((pos - prev_pos) - (sup_pos - sup_prev_pos)))
        else:
            lin_delta = float(np.linalg.norm(pos - prev_pos))
        rel_now = _relative_quat(quat, sup_quat)
        rel_prev = _relative_quat(prev_quat, sup_prev_quat)
        if rel_now is not None and rel_prev is not None:
            ang_delta = _angle_between_quats(rel_now, rel_prev)
        else:
            ang_delta = _angle_between_quats(quat, prev_quat)
    else:
        lin_delta = float(np.linalg.norm(pos - prev_pos))
        ang_delta = _angle_between_quats(quat, prev_quat)
    return bool(lin_delta < STABLE_LINEAR_DELTA_THRESHOLD and ang_delta < STABLE_ANGULAR_DELTA_THRESHOLD)


def _persistent_bool_sticky_true(
    state: Dict[str, Any], key: str, raw_value: bool, fall_threshold: int = STABLE_PERSISTENCE_FRAMES
) -> bool:
    """Asymmetric debounce for object_stable_by_name -- ported verbatim (same
    mechanics) from RoboCasa's own _persistent_bool_sticky_true
    (predicates.py): becoming stable is reported immediately, with no
    persistence delay at all -- only *staying* reported-stable through a
    brief raw-unstable blip gets smoothed. fall_threshold consecutive
    raw-unstable frames are required before flipping back to unstable.
    Per-key debounce state lives in state["stable_debounce"], keyed by
    object name (mirrors RoboCasa's monitor_state["persistent_bools"])."""
    states = state.setdefault("stable_debounce", {})
    raw = bool(raw_value)
    entry = states.get(key)
    if not isinstance(entry, dict):
        states[key] = {"value": raw, "count": 0}
        return raw
    if raw:
        entry["value"] = True
        entry["count"] = 0
        return True
    current = bool(entry.get("value", raw))
    if not current:
        entry["count"] = 0
        return False
    count = int(entry.get("count", 0)) + 1
    if count >= max(1, int(fall_threshold)):
        entry["value"] = False
        entry["count"] = 0
        return False
    entry["count"] = count
    return current


def _point_in_any_fixture_region(env, fixture_name: Optional[str], point: Optional[np.ndarray]) -> Optional[bool]:
    """Real point-in-box test against a fixture's own registered "region"
    sites (e.g. wooden_cabinet_1_top_region/_middle_region/_bottom_region --
    per-drawer interior markers, the LIBERO analog of RoboCasa's `_reg_`
    geom naming convention), in preference to a fixed eef-to-fixture-ROOT-
    BODY distance radius (FIXTURE_INTERIOR_RADIUS). Found via
    open_the_middle_drawer_of_the_cabinet (2026-09-09): this task's own
    demo never reaches inside the drawer at all (there's nothing to place),
    yet root-body distance still dipped under FIXTURE_INTERIOR_RADIUS the
    moment the gripper grabbed the handle to pull it open -- a drawer's
    root body sits close enough to its own front face that "operating the
    mechanism from outside" and "genuinely inside the cavity" are
    indistinguishable by root-body distance alone. The registered region
    sites mark the *actual* interior volumes (same sites RoboCasa-style
    check_obj_in_receptacle/check_contain machinery would use for real
    containment), so testing eef membership in any of them is the same
    kind of test object_in_fixture already trusts for real objects,
    applied to the gripper's own point position instead. Uses
    SiteObjectState's own in_box (LIBERO's registered site + orientation),
    NOT the generic CompositeObject.in_box that most fixture body classes
    (e.g. WoodenCabinet, a plain MujocoXMLObject) don't even implement --
    confirmed directly (AttributeError) that the body-level in_box fallback
    silently never applies to this corpus's cabinet/drawer fixtures at all.
    Returns None (meaning: caller should fall back to a cruder proxy) if no
    matching region site is found for this fixture, e.g. a fixture that
    was never annotated with region sites."""
    if fixture_name is None or point is None:
        return None
    site_names = [
        name
        for name in getattr(env, "object_sites_dict", {}).keys()
        if name.startswith(f"{fixture_name}_") and name.endswith("_region")
    ]
    if not site_names:
        return None
    for site_name in site_names:
        try:
            site_pos = np.asarray(env.sim.data.get_site_xpos(site_name), dtype=float)
            site_mat = np.asarray(env.sim.data.get_site_xmat(site_name), dtype=float)
            site_obj = env.object_sites_dict[site_name]
            if bool(site_obj.in_box(site_pos, site_mat, point)):
                return True
        except Exception:
            continue
    return False


def _upright(quat: Optional[np.ndarray]) -> bool:
    """Literal port of RoboCasa's real upright check -- NOT its own
    predicates.py (that file only wraps robocasa/utils/object_utils.py's
    `check_obj_upright`, the actual reference implementation):
        obj_rot = env.sim.data.xquat[...]  # wxyz
        r = R.from_quat([obj_rot[1], obj_rot[2], obj_rot[3], obj_rot[0]])  # xyzw
        obj_rot_euler = r.as_euler("xyz", degrees=True)
        obj_upright = abs(obj_rot_euler[1]) < th and abs(obj_rot_euler[0]) < th  # th=15
    This is an axis-aligned Euler roll/pitch box (yaw ignored, worst-case
    diagonal tilt ~20-21 degrees before failing), NOT a rotationally
    symmetric tilt-from-vertical cone -- a prior version of this function
    used cos(tilt) >= UPRIGHT_COS_THRESHOLD (0.85, ~31.8 degree half-angle
    cone), a materially different and more permissive definition (found via
    2026-09-21 comprehensive-mirror audit: RoboCasa's real threshold and
    functional form live outside predicates.py entirely, in the vendored
    object_utils.py, so a prior literal-translation pass never actually
    compared against it). Ported here with scipy (already a dependency of
    this env, confirmed present in safemanip_libero) rather than
    reimplementing quaternion-to-Euler by hand, to match scipy's exact
    intrinsic/extrinsic convention bit-for-bit. Verified via
    KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer... ep0: with the
    old cone threshold the carried bowl read object_upright=True throughout
    frames 99-174 (grasped/carried window); with this literal box threshold
    it correctly dips to False for frames 153-174 during the actual
    place-into-drawer tilt, then returns True once set down -- a real,
    previously-undetected transient tilt the old formula was too permissive
    to catch."""
    if quat is None:
        return True
    try:
        from scipy.spatial.transform import Rotation as R

        w, x, y, z = quat
        r = R.from_quat([x, y, z, w])
        euler = r.as_euler("xyz", degrees=True)
        return bool(abs(euler[1]) < UPRIGHT_EULER_THRESHOLD_DEG and abs(euler[0]) < UPRIGHT_EULER_THRESHOLD_DEG)
    except Exception:
        return True


def _fixture_action_tags(env, name: str) -> set:
    """Which of the 5 skill actions (press/turn/slide/twist/open_close) this
    fixture is a plausible target for -- a simplified generic port of
    RoboCasa's own keyword+attribute tagging (its real `_target_candidates`/
    `action_candidates_by_name` machinery additionally uses per-geom AABB
    extraction and a task-registered-fixture-reference preference order this
    port does not replicate -- see module docstring). Keyword matches come
    from `ACTION_COMPONENT_KEYWORDS` against the fixture's category name;
    structural joint-type match (SLIDE/HINGE, and has_turnon for the
    HINGE-twist case) both supplements and disambiguates keyword collisions
    (e.g. "drawer" matches both "slide" and "open_close" keyword lists)."""
    fixture = env.fixtures_dict.get(name)
    joints = getattr(fixture, "joints", None) or []
    if not joints:
        return set()
    category = object_category_from_instance_name(name)
    tags = {action for action, keywords in ACTION_COMPONENT_KEYWORDS.items() if any(kw in category for kw in keywords)}
    # Fixed 2026-09-21 (dependency-tree audit): ACTION_COMPONENT_KEYWORDS is
    # the wrong keyword set for press/turn CANDIDACY -- in RoboCasa it only
    # picks out which geom *within* an already-candidate fixture is the
    # pressable/turnable component (_fixture_component_geom_ids), never
    # decides candidacy itself. RoboCasa's real candidacy test is
    # `_fixture_attrs` (predicates.py ~5990-6024): class-based defaults
    # (Microwave -> "pressable", Sink -> "turnable") plus its own narrow
    # name-substring additions ("microwave" -> pressable, "faucet"/"sink"
    # -> turnable). Ported here using the substring constants this file
    # already has and already trusts elsewhere (MICROWAVE_FIXTURE_NAME_
    # SUBSTRINGS at fixture_ready_for_press/microwave_empty's own _is_
    # microwave, FAUCET_FIXTURE_NAME_SUBSTRINGS at fixture_ready_for_turn).
    #
    # "press" tagging REVERTED (2026-09-21, explicit user decision, found
    # via the v32 corpus-wide 100%-violation-rate scan): RoboCasa's real
    # press-onset proximity is computed against the SPECIFIC BUTTON GEOM
    # within the microwave fixture (_fixture_component_geom_ids /
    # ACTION_COMPONENT_KEYWORDS), not the fixture's whole root-body
    # position -- LIBERO's `_focus_fixture_for_action`/`_body_pos` only
    # ever measures proximity to the whole fixture's root body, with no
    # component-level distinction available (the microwave asset's own
    # button mesh, `microbutton`, has no separately-addressable geom name
    # or collision geom in this LIBERO checkout -- confirmed by inspecting
    # `assets/articulated_objects/microwave.xml` directly). Tagging the
    # whole microwave "pressable" therefore made skill_press_onset fire on
    # ANY approach to the microwave at all (opening the door, placing an
    # object inside), not specifically on pressing a button -- confirmed
    # via KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_
    # and_close_it, which never presses a button anywhere in its real
    # demonstrations, yet showed rc_press_preconditions_safe violated in
    # 10/10 episodes once this tag was added. Reverted; "press" stays
    # genuinely inert for LIBERO's real 40-task corpus until real
    # component-level geom addressing is built (same category of gap as
    # RoboCasa's real _fixture_interior_support_aabb needing per-fixture
    # asset metadata LIBERO's assets don't expose).
    if any(sub in category for sub in FAUCET_FIXTURE_NAME_SUBSTRINGS):
        tags.add("turn")
    jclass = _fixture_joint_class(env, name)
    if jclass == "slide":
        tags.add("slide")
        tags.discard("open_close")
    elif jclass == "open_close":
        tags.add("open_close")
        tags.discard("slide")
    elif jclass == "turn":  # HINGE + has_turnon (structural) -> "twist" skill (stove knobs), per corrected mapping
        tags.add("twist")
    return tags


def _focus_fixture_for_action(env, action: str, eef_pos: Optional[np.ndarray]) -> Optional[str]:
    """Nearest fixture tagged with `action`, or None if this task has no such
    fixture at all. Updated 2026-09-21 (dependency-tree audit, see
    _fixture_action_tags' own comment): "press" IS now reachable in this
    corpus (the microwave is a real RoboCasa-pressable class) -- the old
    claim here that press/turn are both always empty was itself the bug
    symptom, not a verified fact about the corpus. "turn" remains
    genuinely, verifiably empty for all (task, action) pairs in this
    40-task corpus, since no fixture references a faucet or sink at all."""
    best, best_d = None, None
    for name in getattr(env, "fixtures_dict", {}).keys():
        if action not in _fixture_action_tags(env, name):
            continue
        pos = _body_pos(env, name)
        if pos is None or eef_pos is None:
            continue
        d = float(np.linalg.norm(eef_pos - pos))
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best


def _generic_fixture_onset(state: Dict[str, Any], key: str, near: bool) -> tuple[bool, bool]:
    """Same near+persistence+fires-once-per-approach shape as
    skill_pick_onset, generalized for the 5 fixture-skill families.
    Returns (onset, onset_end) -- onset_end added 2026-09-15 for
    rc_{action}_preconditions_safe's recovery_ltl (RoboCasa predicates.py's
    skill_{action}_onset_end): True the frame "fired" clears back to False,
    i.e. the robot is no longer near this target (attempt concluded,
    regardless of outcome)."""
    entry = state.setdefault(key, {"streak": 0, "fired": False})
    was_fired = entry["fired"]
    if near:
        entry["streak"] += 1
    else:
        entry["streak"] = 0
        entry["fired"] = False
    onset = False
    if near and entry["streak"] >= SKILL_ONSET_FRAMES and not entry["fired"]:
        entry["fired"] = True
        onset = True
    onset_end = bool(was_fired and not entry["fired"])
    return onset, onset_end


def _focus_fixture(env) -> Optional[str]:
    """The one openable fixture (microwave/cabinet/drawer/fridge/oven) this
    task's objects/goal actually reference, if any. LIBERO tasks in this
    corpus reference at most one such fixture, so "first match with joints"
    is a deterministic, adequate v0 selector (see module docstring)."""
    for name, fixture in getattr(env, "fixtures_dict", {}).items():
        category = object_category_from_instance_name(name)
        if any(sub in category for sub in OPENABLE_FIXTURE_NAME_SUBSTRINGS):
            if getattr(fixture, "joints", None):
                return name
    return None


def _joint_type(env, joint_name: str) -> Optional[int]:
    try:
        jid = env.sim.model.joint_name2id(joint_name)
        return int(env.sim.model.jnt_type[jid])
    except Exception:
        return None


def _fixture_joint_class(env, name: str) -> Optional[str]:
    """Classifies a fixture's articulation generically by raw MuJoCo joint
    type, NOT by name/task-specific tagging (RoboCasa's
    `action_candidates_by_name` per-fixture-class role registry has no
    LIBERO equivalent): SLIDE joints are drawers ("slide"); HINGE joints on a
    fixture whose `ObjectState` also exposes `turn_on`/`turn_off` are knobs
    ("turn"); any other HINGE joint is a door ("open_close"). Returns None
    if the fixture has no joints or an unrecognized joint type (e.g. the
    FREE joint on `flat_stove_1` used only for the burner-flame visualization
    site, not the knob itself -- see `_focus_by_class`'s own has_turnon
    gating, which is what actually matters for "turn")."""
    fixture = env.fixtures_dict.get(name)
    joints = getattr(fixture, "joints", None) or []
    has_turnon = hasattr(env.get_object(name), "turn_on")
    for jn in joints:
        jtype = _joint_type(env, jn)
        if jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            return "slide"
        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
            return "turn" if has_turnon else "open_close"
    return None


def _focus_by_class(env, cls: str) -> Optional[str]:
    """First fixture (deterministic, see `_focus_fixture`'s docstring for why
    that's adequate for this corpus) whose articulation classifies as `cls`
    ("slide"/"turn"/"open_close")."""
    for name in getattr(env, "fixtures_dict", {}).keys():
        try:
            if _fixture_joint_class(env, name) == cls:
                return name
        except Exception:
            continue
    return None


def _fixture_open_fraction(env, name: Optional[str], state: Dict[str, Any]) -> Optional[float]:
    """Generic, joint-type-agnostic "how far articulated" signal: mean
    absolute displacement of each of the fixture's joints from wherever they
    were the first frame this fixture was seen this episode (LIBERO's
    initial_state has these fixtures start closed for every in-scope task
    that references one, so "first-seen qpos" is a safe closed-reference
    proxy -- a documented v0 assumption, not guaranteed universally),
    normalized by each joint's own range span. Used for both onset-detection
    (slide/open_close families) and mechanism_safety's
    fixture_is_opening/closing direction."""
    if not name:
        return None
    fixture = env.fixtures_dict.get(name)
    joints = getattr(fixture, "joints", None) or []
    if not joints:
        return None
    start = state.setdefault("fixture_start_qpos", {}).setdefault(name, {})
    fractions = []
    for jn in joints:
        try:
            qpos = float(env.sim.data.get_joint_qpos(jn))
            jid = env.sim.model.joint_name2id(jn)
            lo, hi = env.sim.model.jnt_range[jid]
            span = abs(hi - lo) or 1.0
        except Exception:
            continue
        if jn not in start:
            start[jn] = qpos
        fractions.append(abs(qpos - start[jn]) / span)
    return float(np.mean(fractions)) if fractions else None


def _robot_geoms(env) -> List[str]:
    try:
        robot_model = env.robots[0].robot_model
        gripper = env.robots[0].gripper
        geoms = list(robot_model.contact_geoms)
        for group in gripper.important_geoms.values():
            geoms.extend(group if isinstance(group, (list, tuple)) else [group])
        return geoms
    except Exception:
        return []


def _robot_contacts_fixture(env, name: Optional[str]) -> bool:
    if not name:
        return False
    try:
        robot_geoms = _robot_geoms(env)
        if not robot_geoms:
            return False
        model = env.get_object(name)
        return bool(env.check_contact(robot_geoms, model))
    except Exception:
        return False


def _fixture_touches_other_movable(env, name: Optional[str], exclude=()) -> bool:
    """True if the fixture body contacts some OTHER movable object (not the
    robot) -- an obstacle in the fixture's articulation path. `exclude`
    (2026-09-09, a name or collection of names): the task's own manipulated
    object (e.g. a bowl actively being placed inside a drawer/cabinet)
    resting against the fixture's interior is the *intended* outcome, not
    an obstruction -- confirmed on KITCHEN_SCENE4 (put the bowl in the
    drawer and close it): the bowl contacting the drawer/cabinet body
    mid-close read as fixture_close_obstacle_hit even though nothing was
    actually jamming the mechanism, and since the robot goes on to
    genuinely finish closing (not retract),
    fixture_close_retract_resolved never fires -- a permanent, spurious
    rc_fixture_close_obstacle_retract violation for the rest of the
    episode. Also covers the case where the manipulated object was ALREADY
    resting in/against the fixture from the very start of the episode
    (e.g. pick_up_the_black_bowl_in_the_top_drawer_...: the bowl starts
    inside the drawer being opened, contacting it before it's ever
    grasped, so the single-`active`-object exclusion above doesn't apply
    yet) -- callers should also pass any objects captured as touching the
    fixture at the start of the episode (see
    state["initial_fixture_contacts"]), the same "ignore what was already
    there at frame 0" convention RoboCasa's own predicates.py uses for
    forbidden_contact's ignored_initial_contact_pairs. Already flagged as
    a known v0 imprecision in this module's own docstring before this fix.
    Cheap: this corpus has at most a handful of movable objects per task."""
    if not name:
        return False
    if isinstance(exclude, str):
        exclude = {exclude}
    else:
        exclude = set(exclude or ())
    try:
        model = env.get_object(name)
        for obj_name in _movable_object_names(env):
            if obj_name in exclude:
                continue
            if env.check_contact(model, env.get_object(obj_name)):
                return True
    except Exception:
        pass
    return False


# --- fixture_ready_for_{press,turn,slide,twist,open_close} ----------------
# Literal port of RoboCasa's own `_fixture_ready_for_press/turn/slide/twist/
# open_close` (monitor/sim/robocasa/predicates.py ~6992-7124): these ask
# "given what's currently AT this fixture (contents resting on/in it), is it
# actually safe/valid to perform this specific action on it right now" --
# NOT "is a drawer already open" as it might sound from the name (verified by
# reading each RoboCasa function's real body in full, not guessed). RoboCasa
# branches per fixture *class* (coffee machine / microwave / toaster / oven /
# kettle / dishwasher / blender for press; sink for turn; dishwasher for
# slide; stove / toaster / oven / mixer for twist; microwave / oven-or-
# toaster / dishwasher for open_close), checking whether the fixture's
# contents carry the right content-attribute tags (microwavable/food/
# cookable/toastable/dishwashable/etc), via RoboCasa's real per-category
# metadata (attrs_by_name).
#
# LIBERO has no equivalent metadata registry, but its own object/fixture
# *names* already encode the same information via the substring-keyword
# taxonomy `monitor/sim/libero/attributes.py` already built for other
# predicate families (MICROWAVABLE_NAME_SUBSTRINGS, FOOD_NAME_SUBSTRINGS,
# COOKABLE_NAME_SUBSTRINGS, WASHABLE_NAME_SUBSTRINGS, DISHWASHABLE_NAME_
# SUBSTRINGS, TOOL_NAME_SUBSTRINGS, object_is_receptacle_category) -- these
# were already present in attributes.py, explicitly flagged there as "not
# yet consumed by any LIBERO precondition check", i.e. built for exactly
# this porting work. `_object_known_content_attrs`/`_objects_have_any_attr`
# below reuse them the same way RoboCasa's own attrs_by_name/
# _objects_have_any_attr do, including RoboCasa's own permissive semantics:
# an object with NO recognized content attribute at all doesn't block the
# check (RoboCasa: "if obj_attrs and not (obj_attrs & attrs): return False"
# -- only objects with *some* known attributes that fail to include a
# required one block it).
#
# Several of RoboCasa's per-class branches (coffee machine, toaster, oven,
# dishwasher, blender, mixer, electric kettle) have NO LIBERO analog to port
# at all -- not merely "this 40-task corpus never uses one" (already true
# and already the standard this file uses elsewhere, e.g. FAUCET_FIXTURE_
# NAME_SUBSTRINGS/sink) but structurally absent: LIBERO's entire fixture
# object library (libero/libero/envs/objects/articulated_objects.py) defines
# exactly Microwave, SlideCabinet, Window, Faucet, BasinFaucet, ShortCabinet,
# ShortFridge, WoodenCabinet, WhiteCabinet, FlatStove -- there is no
# Dishwasher/Oven/Toaster/Blender/CoffeeMachine/ElectricKettle/Mixer class
# anywhere in this simulator to ever instantiate, so those RoboCasa branches
# are omitted below rather than written as permanently-dead code (unlike the
# sink/faucet branch, which LIBERO's object library DOES support even though
# no in-scope task instantiates one, so it's kept, mirroring the "verified
# zero occurrence, still implemented" standard already used elsewhere in this
# file for e.g. RAW_NAME_SUBSTRINGS). Every fixture class LIBERO's library
# genuinely has (microwave/stove/cabinet/fridge) falls through to RoboCasa's
# own real "else True" default when it isn't one of RoboCasa's specially-
# handled classes, exactly matching RoboCasa's own behavior for e.g. a plain
# HingeCabinet.
_ALL_KNOWN_CONTENT_ATTR_SUBSTRINGS = {
    "microwavable": MICROWAVABLE_NAME_SUBSTRINGS,
    "food": FOOD_NAME_SUBSTRINGS,
    "cookable": COOKABLE_NAME_SUBSTRINGS,
    "washable": WASHABLE_NAME_SUBSTRINGS,
    "dishwashable": DISHWASHABLE_NAME_SUBSTRINGS,
    "utensil": TOOL_NAME_SUBSTRINGS,
    "liquid": LIQUID_NAME_SUBSTRINGS,
}


def _object_known_content_attrs(name: str) -> set:
    """Which of `_ALL_KNOWN_CONTENT_ATTR_SUBSTRINGS`'s abstract content
    attributes this object's category name matches, plus "receptacle" via
    `object_is_receptacle_category` -- LIBERO's substring-keyword analog of
    RoboCasa's real per-object `attrs_by_name.get(name, set())`. An empty
    return means this object's category isn't recognized by any of this
    file's content-attribute taxonomies at all (RoboCasa's equivalent: an
    object whose category dict never set any of these attribute flags)."""
    category = object_category_from_instance_name(name)
    attrs = {attr for attr, subs in _ALL_KNOWN_CONTENT_ATTR_SUBSTRINGS.items() if any(s in category for s in subs)}
    if object_is_receptacle_category(category):
        attrs.add("receptacle")
    return attrs


def _objects_have_any_content_attr(names: List[str], required: set, *, allow_empty: bool = True) -> bool:
    """Literal port of RoboCasa's `_objects_have_any_attr`: passes vacuously
    if `names` is empty (per `allow_empty`), and passes for any object with
    NO recognized content attribute at all (unknown category -- don't block
    on ignorance); only fails for an object whose recognized attributes
    exist but don't intersect `required`."""
    if not names:
        return allow_empty
    for name in names:
        obj_attrs = _object_known_content_attrs(name)
        if obj_attrs and not (obj_attrs & required):
            return False
    return True


def _objects_at_fixture(env, object_states_dict: Dict[str, Any], fixture_name: Optional[str]) -> List[str]:
    """Movable objects currently at `fixture_name` -- literal port of
    RoboCasa's own `_objects_at_fixture`'s dual test (real containment OR
    real contact), generalized to any fixture (not just the mechanism-safety
    focus fixture `_objects_at_fixture` in the rest of this file's build_
    predicate_snapshot section implicitly assumes). Containment
    (`check_contact` + `check_contain`) covers enclosure fixtures (microwave);
    plain contact covers open-surface fixtures (stove burner, dish rack) that
    have no meaningful "contain" concept."""
    if not fixture_name:
        return []
    found = set()
    fixture_state = object_states_dict.get(fixture_name)
    for name in _movable_object_names(env):
        obj_state = object_states_dict.get(name)
        try:
            if fixture_state is not None and obj_state is not None and fixture_state.check_contact(obj_state) and fixture_state.check_contain(obj_state):
                found.add(name)
                continue
        except Exception:
            pass
        try:
            if env.check_contact(env.get_object(fixture_name), env.get_object(name)):
                found.add(name)
        except Exception:
            continue
    return sorted(found)


def _fixture_is_closed_state(object_states_dict: Dict[str, Any], fixture_name: Optional[str]) -> bool:
    if not fixture_name or fixture_name not in object_states_dict:
        return False
    return _safe_call_bool(object_states_dict[fixture_name], "is_close")


def _fixture_ready_for_press(env, object_states_dict: Dict[str, Any], target: Optional[str]) -> bool:
    if target is None:
        return False
    category = object_category_from_instance_name(target)
    if "microwave" in category:
        contents = _objects_at_fixture(env, object_states_dict, target)
        return _fixture_is_closed_state(object_states_dict, target) and _objects_have_any_content_attr(
            contents, {"microwavable", "food"}
        )
    return True


def _fixture_ready_for_turn(env, object_states_dict: Dict[str, Any], target: Optional[str]) -> bool:
    if target is None:
        return False
    category = object_category_from_instance_name(target)
    if any(sub in category for sub in FAUCET_FIXTURE_NAME_SUBSTRINGS):
        contents = _objects_at_fixture(env, object_states_dict, target)
        return _objects_have_any_content_attr(
            contents, {"washable", "dishwashable", "food", "receptacle", "utensil"}
        )
    return True


def _fixture_ready_for_slide(env, object_states_dict: Dict[str, Any], target: Optional[str]) -> bool:
    if target is None:
        return False
    category = object_category_from_instance_name(target)
    # No Dishwasher class exists in LIBERO's fixture object library at all
    # (see module comment above) -- RoboCasa's only "slide" target class
    # (Dishwasher rack) and its open-before-slide gate
    # (`_fixture_requires_open_for_slide`) therefore has no fixture to ever
    # apply to here, so that gate (RoboCasa's `_fixture_open(fname,
    # threshold=0.5)` check) is not ported. Separately and more importantly:
    # LIBERO's OWN "slide"
    # action tagging (`_fixture_joint_class`) repurposes plain cabinet/drawer
    # SLIDE joints as "slide" targets -- unlike RoboCasa's real
    # fixture_class_default_attributes(), where ordinary Drawer fixtures are
    # NOT "slideable" (only Dishwasher/Toaster are) and fall under
    # "open_close" instead. Applying RoboCasa's literal open-before-slide
    # gate here would therefore incorrectly require an ordinary drawer to
    # already be open before it's ever "ready" to be opened -- a genuine,
    # narrow semantic mismatch from this file's own "slide" repurposing (not
    # from RoboCasa's actual mechanism), documented here rather than forced.
    if "dishwasher" in category:
        contents = _objects_at_fixture(env, object_states_dict, target)
        return _objects_have_any_content_attr(contents, {"dishwashable", "receptacle", "utensil"})
    return True


def _fixture_ready_for_twist(env, object_states_dict: Dict[str, Any], target: Optional[str]) -> bool:
    # RoboCasa's twist target can be a movable object (bottle/jar cap) or a
    # fixture; already-documented elsewhere in this file (skill_twist_onset's
    # own comment) that LIBERO's movable objects have no articulated cap/lid
    # sub-mechanism, so twist is fixture-only here -- no object-kind branch
    # to port.
    if target is None:
        return False
    category = object_category_from_instance_name(target)
    if "stove" in category:
        contents = _objects_at_fixture(env, object_states_dict, target)
        # LIBERO-specific deviation from RoboCasa (2026-09-21, explicit user
        # decision): RoboCasa's `_stove_contents_ready` = `_heat_contents_
        # ready(contents, {"cookable","food","liquid"}, require_carrier=
        # True)` requires a cookware carrier (pot/pan) actually present on
        # the burner before twisting the knob is "ready" at all -- an empty
        # burner is never ready for RoboCasa. LIBERO's `turn_on_the_stove`-
        # family tasks explicitly instruct turning the stove on BEFORE
        # anything is placed on it, so that "empty burner = never ready"
        # semantic doesn't fit this corpus's own intended task ordering --
        # dropped the require_carrier gate for LIBERO only (RoboCasa's own
        # predicates.py is untouched). Still requires that whatever IS
        # already present (if anything) is safe to heat -- an empty burner
        # is fine (allow_empty=True), incompatible contents directly on the
        # burner are not.
        carriers = [name for name in contents if _object_known_content_attrs(name) & {"receptacle", "utensil"}]
        heat_contents = [name for name in contents if name not in set(carriers)]
        return _objects_have_any_content_attr(heat_contents, {"cookable", "food", "liquid"})
    return True


def _fixture_ready_for_open_close(env, object_states_dict: Dict[str, Any], target: Optional[str]) -> bool:
    if target is None:
        return False
    category = object_category_from_instance_name(target)
    if "microwave" in category:
        contents = _objects_at_fixture(env, object_states_dict, target)
        return _objects_have_any_content_attr(contents, {"microwavable", "food", "receptacle"})
    return True



def _safe_call_bool(obj, method_name: str) -> bool:
    try:
        return bool(getattr(obj, method_name)())
    except Exception:
        return False


def _safe_language_instruction(env) -> str:
    """`BDDLBaseDomain.language_instruction` is a property that reads
    `self.parsed_problem["language"]`, a key `robosuite_parse_problem` never
    actually populates in this LIBERO checkout (a pre-existing LIBERO bug,
    not ours) -- raises KeyError instead of returning ''/None, so a plain
    `getattr(..., default)` doesn't help. `env.language_instruction` (the
    *wrapper*'s own instance attribute, set correctly from
    `BDDLUtils.get_problem_info` in `env_wrapper.py`) is not visible from
    here (this function only ever receives the inner env), so we fall back
    to the human-readable task class name instead."""
    try:
        return env.language_instruction
    except Exception:
        return env.__class__.__name__


def _is_microwave(name: Optional[str]) -> bool:
    if not name:
        return False
    category = object_category_from_instance_name(name)
    return any(sub in category for sub in MICROWAVE_FIXTURE_NAME_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Static spec
# ---------------------------------------------------------------------------

def build_predicate_static_spec(env, static_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_name": env.__class__.__name__,
        "task_language": _safe_language_instruction(env),
        "predicate_groups": {
            "contact_policy": ["forbidden_contact", "forbidden_contact_sustained"],
            "grasp_release_settle": [
                "object_grasped", "object_grasped_raw", "object_stable", "object_sync", "object_upright",
                "object_dropped", "object_left_gripper", "object_released",
                "object_supported", "gripper_away_from_object", "object_settled",
                "object_settle_timeout", "release_object_settle_timeout",
                "gripper_is_opening", "gripper_is_closing",
            ],
            # 2026-09-16 (round-2 audit): added "skill_pick_onset_end" and
            # "object_upright_if_receptacle" here -- both are real, emitted
            # predicates.py:predicates dict entries (skill_pick_onset_end
            # since 2026-09-15, object_upright_if_receptacle since round 1's
            # 2026-09-16 fix) that were simply never added to this grouping
            # list, the same computed-but-unregistered gap round 1 found for
            # the predicates dict itself, found this time in the static
            # spec's own group listing (RoboCasa's real predicate_groups,
            # predicates.py:255-281, lists both under "skill_onset"/
            # "pick_preconditions" respectively).
            "skill_onset": ["skill_pick_onset", "skill_pick_onset_end", "gripper_near_object"],
            "pick_preconditions": ["object_region_clear", "object_upright_if_receptacle", "preconditions_satisfied_pick", "pick_precondition_escape"],
            "place_preconditions": [
                "skill_place_onset", "support_region_clear", "support_stable",
                "support_geometry_valid", "support_objects_clean_for_manipulated_object",
                "support_not_cluttered_for_fragile_manipulated_object",
                "preconditions_satisfied_place",
            ],
            "fixture_skill_onset": [
                "skill_press_onset", "skill_turn_onset", "skill_slide_onset",
                "skill_twist_onset", "skill_open_close_onset",
                "target_region_clear", "target_stable", "slide_path_clear",
                "articulation_path_clear", "preconditions_satisfied_press",
                "preconditions_satisfied_turn", "preconditions_satisfied_slide",
                "preconditions_satisfied_twist", "preconditions_satisfied_open_close",
            ],
            "contamination": [
                "robot_contact_raw_contaminated", "object_is_rte", "robot_contact_clean",
                "robot_contact_clean_sustained", "sanitized",
            ],
            "containment_safety": [
                "containment_transfer_event", "fixture_output_started",
                "content_is_liquid", "content_is_solid",
            ],
            "mechanism_safety": [
                "robot_fixture_contact", "fixture_is_opening", "fixture_is_closing",
                "fixture_obstacle_contact", "continue_fixture_open", "continue_fixture_close",
                "fixture_open_obstacle_hit", "fixture_close_obstacle_hit",
                "fixture_open_retracting", "fixture_close_retracting",
                "fixture_open_retract_timeout", "fixture_close_retract_timeout",
            ],
            "access_enclosure_safety": [
                "fixture_fully_open", "fixture_fully_closed", "reach_in_fixture",
                "left_fixture", "gripper_in_fixture", "object_reach_in_fixture",
                "object_reach_in_microwave", "object_left_microwave", "object_in_fixture",
                "object_in_same_fixture", "one_object_in_microwave",
                "two_or_more_objects_in_microwave", "microwave_empty",
            ],
        },
        "role_definitions": {},
    }


# ---------------------------------------------------------------------------
# Dynamic snapshot
# ---------------------------------------------------------------------------

def build_predicate_snapshot(env, static_info: Dict[str, Any], dynamic_info: Dict[str, Any]) -> Dict[str, Any]:
    state = getattr(env, "_libero_predicate_monitor_state", None)
    current_timestep = int((dynamic_info.get("task") or {}).get("timestep", -1))
    if state is None or current_timestep <= state.get("_prev_timestep", -1):
        state = {
            "_prev_timestep": current_timestep,
            "active_object": None,
            "prev_grasped_object": None,
            "prev_positions": {},
            "prev_quats": {},
            "prev_gripper_frac": None,
            "sync_baseline_object": None,
            "sync_baseline_offset": None,
            "sync_baseline_quat": None,
            "settle_watch": None,  # {"object": name, "age": int}
            "forbidden_streak": 0,
            "pick_onset": {},  # name -> {"streak": int, "fired": bool}
            "last_reach_fixture": None,
            "contaminated_spots": [],  # [{"kind", "name", "body_id", "local_offset", "radius"}, ...]
            "contamination_transfer_pair_ages": {},  # "kind:name->kind:name" -> consecutive-frame count
            "contamination_initial_contact_pairs": None,  # frozen set of (geom1,geom2) canonical pairs seen on frame 0, still-present subset only
            "initial_fixture_contacts": {},  # fixture_name -> set(object names touching it when first observed
        }
    else:
        state["_prev_timestep"] = current_timestep
    env._libero_predicate_monitor_state = state

    predicates: Dict[str, Dict[str, Any]] = {}

    # --- contact policy: computed further below, after grasp state -------
    # (the fine-grained role taxonomy needs the current active/grasped object;
    # see the "contact policy: fine-grained contact-role taxonomy" block after
    # the grasp/release computation).

    # --- grasp / release / settle ---------------------------------------
    # gripper_frac/gripper_is_opening computed here (moved ahead of the grasp
    # debounce 2026-09-21, real-data audit into the KITCHEN_SCENE8/
    # LIVING_ROOM_SCENE2 residual support_geometry_valid false positives) --
    # depends only on env/state, not on active/grasped_name, so it's safe to
    # compute before the debounce that now needs it (see
    # _persistent_grasp_candidate's own updated docstring for why).
    gripper_frac = _gripper_closed_fraction(env)
    prev_frac = state["prev_gripper_frac"]
    gripper_is_opening = bool(prev_frac is not None and gripper_frac is not None and gripper_frac < prev_frac - 1e-4)
    gripper_is_closing = bool(prev_frac is not None and gripper_frac is not None and gripper_frac > prev_frac + 1e-4)
    state["prev_gripper_frac"] = gripper_frac

    # Debounced 2026-09-20 (see GRASP_CANDIDATE_PERSISTENCE_FRAMES's own
    # comment for the real-data evidence this file's earlier "flicker
    # already fixed at the raw-signal level" claim was false): grasped_name/
    # object_grasped are now the debounced candidate, matching every other
    # place in this codebase that reads "object_grasped" expecting the
    # stable, not-flickering concept. raw_grasped_name/object_grasped_raw
    # (the previously-undebounced signal, now genuinely separate rather
    # than an alias) are kept for whatever narrowly needs the instantaneous
    # reading -- currently just the shared specs.py's rc_grasp_remains_
    # synced_until_dropped formula, which resolves its "until" on
    # !object_grasped_raw specifically because a level (not an edge) is
    # needed there -- see RoboCasa predicates.py's own extensive comment
    # for the two failed intermediate designs (an edge-based
    # object_dropped_raw, tried and reverted there) before landing on this
    # one; ported directly rather than re-deriving.
    #
    # gripper-opening corroboration (2026-09-21, KITCHEN_SCENE8/
    # LIVING_ROOM_SCENE2 residual-false-positive audit): confirmed via real
    # frame data that a candidate->None transition (a would-be release) can
    # be driven purely by `_check_grasp`'s bilateral-fingerpad-contact query
    # dropping out for MANY consecutive frames (17 on
    # KITCHEN_SCENE8_put_both_moka_pots_on_the_stove ep8, frames 328-344)
    # while `gripper_is_closing` reads True (never `gripper_is_opening`) the
    # entire time and the eef-to-object offset stays perfectly rigid
    # (0.0688m, constant to 4 decimal places) -- i.e. the object never
    # actually left the gripper's grip at all, this is a genuine
    # bilateral-contact solver flicker an order of magnitude longer than the
    # 1-2 frame flicker GRASP_CANDIDATE_PERSISTENCE_FRAMES (5) was tuned
    # against. Contrast confirmed against a REAL release in the same episode
    # (moka_pot_2, frame 183): `gripper_is_opening` reads True for 10+
    # frames (173-182) before the debounce ever accepts the release --
    # a genuine release is always preceded by real gripper-opening motion,
    # this flicker never has any. Rather than blindly raise
    # GRASP_CANDIDATE_PERSISTENCE_FRAMES (which would slow down every real
    # release corpus-wide with no principled stopping point -- the flicker
    # length here has no known upper bound), _persistent_grasp_candidate now
    # additionally requires at least one real `gripper_is_opening` frame
    # during the pending-release run before accepting a candidate->None
    # transition, with a bounded fallback
    # (GRASP_RELEASE_UNCORROBORATED_FALLBACK_FRAMES) so a genuine object
    # physically dislodged without the gripper ever opening (e.g. knocked
    # away) still eventually registers as released rather than sticking
    # forever.
    raw_grasped_name = _check_grasp_any(env)
    grasped_name = _persistent_grasp_candidate(state, raw_grasped_name, gripper_is_opening)
    object_grasped = grasped_name is not None
    object_grasped_raw_value = raw_grasped_name is not None
    if grasped_name is not None:
        state["active_object"] = grasped_name
    active = state["active_object"]

    object_dropped = bool(state["prev_grasped_object"]) and not object_grasped
    # dropped_object_name (2026-09-21 fix, repeated-onset-during-a-single-
    # continuous-interaction audit): which object object_dropped is
    # actually about, captured before prev_grasped_object gets overwritten
    # below -- needed by skill_place_onset's own per-object re-arm latch
    # (see that predicate's own comment for the full rationale).
    dropped_object_name = state["prev_grasped_object"] if object_dropped else None
    state["prev_grasped_object"] = grasped_name

    # --- contact policy: fine-grained contact-role taxonomy --------------
    # Literal port of RoboCasa's forbidden/allowed contact classification
    # (monitor/sim/robocasa/predicates.py:3984-4281) on top of the BDDL-derived
    # role registry (_build_contact_role_registry). Replaces the previous
    # coarse forbidden_contact = _arm_contacts_scene(env) (a single "does any
    # non-gripper arm geom touch anything" boolean, which had no object-side or
    # fixture-role classification at all -- 5 of RoboCasa's 7 categories had
    # zero LIBERO counterpart). _arm_contacts_scene itself is kept (other
    # audits may still reference it), it just no longer feeds forbidden_contact.
    (
        cp_forbidden_now,
        cp_forbidden_sustained,
        cp_allowed_contact,
        cp_forbidden_contact_pairs,
        cp_considered_contact_pairs,
    ) = _evaluate_contact_policy(env, state, active, object_grasped)
    predicates["forbidden_contact"] = _entry(
        cp_forbidden_now,
        "a considered contact pair matched none of the allowed contact roles",
    )
    predicates["forbidden_contact_sustained"] = _entry(
        cp_forbidden_sustained,
        "forbidden_contact persisted past FORBIDDEN_CONTACT_TOLERANCE_FRAMES",
    )
    predicates["allowed_contact"] = _entry(
        cp_allowed_contact,
        "at least one considered pair matched an allowed contact role",
    )

    eef_pos = _eef_pos(env)
    eef_quat = _eef_quat(env)
    # gripper_reach_aabb (2026-09-21, distance-basis audit): the gripper's own
    # current bounding box, computed once here and reused by every onset-
    # proximity check below (pick-onset streak, place-onset re-arm latch, the
    # 5 fixture-skill onsets) via _gripper_target_distance, matching
    # RoboCasa's own _gripper_aabb() being called once per relevant section
    # and reused the same way (predicates.py ~4984/6240). Named distinctly
    # from the pre-existing local `gripper_aabb` used a few sections above
    # for object_region_blockers/support_region_blockers (those are
    # independently recomputed, narrower-scoped locals; this one spans the
    # rest of the function).
    # gripper_frac/gripper_is_opening/gripper_is_closing: computed earlier now
    # (see the grasp/release/settle section above), reused here unchanged.
    gripper_reach_aabb = _gripper_aabb(env)

    active_pos = _body_pos(env, active) if active else None
    active_quat = _body_quat(env, active) if active else None
    prev_pos = state["prev_positions"].get(active) if active else None
    prev_quat = state["prev_quats"].get(active) if active else None

    # object_stable_by_name (2026-09-20): debounced, relative-to-current-
    # support per-object stability dict -- LIBERO analog of RoboCasa's own
    # object_stable_by_name (predicates.py), see STABLE_PERSISTENCE_FRAMES'
    # own comment and _object_stable_relative_by_name's docstring for why
    # both the asymmetric debounce and the relative-to-support correction
    # matter (previously entirely absent: LIBERO's object_stable was a raw,
    # undebounced, world-frame-only per-frame check). Computed here (before
    # this frame's prev_positions/prev_quats get overwritten at the end of
    # this function) for every movable object, not just `active` -- both
    # the settle-watch below (settle_obj_name, frequently a different object
    # than `active` once a second object gets grasped) and the pick
    # preconditions further down (focus_pick_object, usually not yet
    # grasped) need other objects' entries too.
    object_stable_by_name = {
        name: _persistent_bool_sticky_true(
            state,
            f"object_stable::{name}",
            _object_stable_relative_by_name(env, state, name),
        )
        for name in _movable_object_names(env)
    }
    object_stable = bool(object_stable_by_name.get(active, True)) if active is not None else True

    # Track which other movable objects are currently "carried inside" the
    # grasped object, if it's itself a receptacle -- mirrors RoboCasa's own
    # grasped_receptacle_content_source/_names tracking (predicates.py),
    # used below by _support_region_blockers so a receptacle's own
    # pre-existing contents don't block the receptacle's OWN placement once
    # it's set back down (as opposed to the separate, already-fixed case of
    # placing a second unrelated item into a receptacle that already
    # contains a first one). Simple AABB-intersection snapshot, refreshed
    # every frame the receptacle is grasped, rather than RoboCasa's own
    # positional-containment machinery -- adequate here since it's only
    # queried immediately after the same object is set down.
    if object_grasped and active is not None:
        active_category = object_category_from_instance_name(active)
        if object_is_receptacle_category(active_category):
            active_aabb_now = _object_aabb(env, active)
            if active_aabb_now is not None:
                contained = [
                    other
                    for other in _movable_object_names(env)
                    if other != active
                    and _object_aabb(env, other) is not None
                    and _aabb_intersects(active_aabb_now, _object_aabb(env, other))
                ]
                state.setdefault("carried_content_names", {})[active] = contained

    # object_sync (2026-09-09, rotation/angular-slip fix 2026-09-21): ported
    # RoboCasa's own grasp-slip pattern (_object_grasp_slip/_object_sync,
    # predicates.py) -- a dedicated reference re-seeded fresh at the exact
    # frame a NEW grasp begins, refreshed every frame thereafter (per-frame
    # drift, not accumulated-since-onset -- same rationale as RoboCasa's
    # _object_grasp_slip: a one-time settling shift right after lift-off
    # should be flagged once, not for the rest of the grasp). Previously
    # compared against state["prev_offsets"][active], which is updated for
    # *every* movable object on *every* frame regardless of grasp state --
    # on the very first frame of a brand-new grasp, that "previous" offset
    # was recorded before the object was ever grasped at all (e.g. still
    # sitting in the fridge, gripper mid-approach from elsewhere), so the
    # eef finally reaching the object at grasp onset read as a huge,
    # spurious "slip" -- confirmed on KITCHEN_SCENE3_turn_on_the_stove_
    # and_put_the_moka_pot_on_it ep2: object_sync=False on literally the
    # first frame object_grasped ever becomes True for that object, then
    # True every frame after. No reference yet (brand-new grasp, or the
    # active object just changed) means "nothing to compare against yet",
    # not "out of sync" -- object_sync stays True for that one seed frame,
    # matching RoboCasa's _object_grasp_slip returning None on an
    # unseeded/mismatched reference.
    # Re-seed on a brand-new grasp EVENT (object_grasped just became True),
    # not just "the active object's name changed" -- a regrasp of the SAME
    # object after a drop-then-regrasp flicker (active never gets cleared
    # on drop, only reassigned on the next actual grasp) needs a fresh
    # baseline too: the object may have shifted slightly while briefly
    # ungrasped, and comparing the new grasp's first frame against the
    # *previous* grasp's stale reference is exactly the same spurious-slip
    # bug this fix was for in the first place -- confirmed on
    # KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_...: a
    # drop-then-regrasp of the same bowl (frames 155/160) still showed
    # object_sync=False at 160 without this.
    #
    # Rotation/angular-slip fix (2026-09-21, dependency-tree audit): the
    # version above only ever compared a raw WORLD-frame offset
    # (active_pos - eef_pos) against the previous frame's raw world-frame
    # offset, with no rotation correction and no angular component at all --
    # unlike RoboCasa's real _object_grasp_slip. This meant a rigidly-held
    # object being carried through an ordinary wrist rotation (no real slip
    # at all) would report a large, spurious world-frame offset delta the
    # moment the object swept through space relative to the world, and it
    # meant a real object spinning/rotating in the gripper's grip (a genuine
    # slip) could never be detected at all if its position stayed centered.
    # Ported RoboCasa's exact fix: track the reference offset ROTATED INTO
    # THE EEF'S OWN FRAME (sync_baseline_offset, now
    # quat_rotate_vector(conj(eef_quat), active_pos - eef_pos) rather than a
    # raw world-frame subtraction) so a pure wrist rotation no longer looks
    # like slip, plus a separate baseline RELATIVE ORIENTATION
    # (sync_baseline_quat = conj(eef_quat) * active_quat) checked against
    # SYNC_ANGULAR_DELTA_THRESHOLD every frame. Both the (now
    # rotation-corrected) linear check AND the new angular check must pass
    # for object_sync to hold, exactly mirroring RoboCasa's
    # `linear_slip < GRASP_SLIP_LINEAR_THRESHOLD and angular_slip <
    # GRASP_SLIP_ANGULAR_THRESHOLD`.
    fresh_grasp = object_grasped and (
        not state.get("prev_object_grasped_for_sync", False)
        or state.get("sync_baseline_object") != active
    )
    state["prev_object_grasped_for_sync"] = object_grasped
    _sync_refs_available = (
        active_pos is not None
        and active_quat is not None
        and eef_pos is not None
        and eef_quat is not None
    )
    if fresh_grasp:
        state["sync_baseline_object"] = active
        if _sync_refs_available:
            eef_quat_conj = _quat_conjugate(eef_quat)
            state["sync_baseline_offset"] = _quat_rotate_vector(eef_quat_conj, active_pos - eef_pos)
            state["sync_baseline_quat"] = _quat_multiply(eef_quat_conj, active_quat)
        else:
            state["sync_baseline_offset"] = None
            state["sync_baseline_quat"] = None
        object_sync = True
    elif active is None:
        state["sync_baseline_object"] = None
        state["sync_baseline_offset"] = None
        state["sync_baseline_quat"] = None
        object_sync = True
    else:
        baseline_offset = state.get("sync_baseline_offset")
        baseline_quat = state.get("sync_baseline_quat")
        if _sync_refs_available and baseline_offset is not None and baseline_quat is not None:
            expected_pos = eef_pos + _quat_rotate_vector(eef_quat, baseline_offset)
            linear_slip = float(np.linalg.norm(active_pos - expected_pos))
            expected_quat = _quat_multiply(eef_quat, baseline_quat)
            diff_quat = _quat_multiply(_quat_conjugate(expected_quat), active_quat)
            angular_slip = _quat_angle(diff_quat)
            object_sync = bool(
                linear_slip < SYNC_RELATIVE_DELTA_THRESHOLD
                and angular_slip < SYNC_ANGULAR_DELTA_THRESHOLD
            )
            # Refresh the reference to *this* frame's actual relative pose,
            # regardless of whether slip exceeded threshold -- this is what
            # makes the check per-frame rather than accumulated-since-onset
            # (see the comment block above), same as RoboCasa's
            # _object_grasp_slip.
            eef_quat_conj = _quat_conjugate(eef_quat)
            state["sync_baseline_offset"] = _quat_rotate_vector(eef_quat_conj, active_pos - eef_pos)
            state["sync_baseline_quat"] = _quat_multiply(eef_quat_conj, active_quat)
        else:
            # No usable reference this frame (missing pose data, or no
            # baseline seeded yet) -- "nothing to compare against" reads as
            # in-sync, and the stale baseline (if any) is left untouched
            # rather than clobbered with an unusable reading.
            object_sync = True

    object_upright = _upright(active_quat)

    # AABB overlap (_aabb_intersects/_object_geom_ids/_gripper_contact_geom_
    # ids), not raw single-geom fingerpad contact or exact mesh distance --
    # ported verbatim from RoboCasa's own fix for this exact predicate
    # (object_left_gripper: "this predicate was introduced to fix" a
    # one-frame raw-contact flicker, switched to _aabb_intersects/
    # _object_contact_aabb, deliberately coarser than a real mesh-distance
    # check). Tried real mesh/geom distance first (mj_geomDistance via
    # _gripper_object_geom_min_distance) since it was already built for
    # gripper_away_from_object -- explicit user correction (2026-09-09):
    # that's *stricter* than RoboCasa's actual mechanism, not equivalent --
    # a bounding-box overlap tolerates the object still being well within
    # the gripper's enclosing volume even if the exact meshes momentarily
    # show a hair of daylight between them (a real, if physically
    # insignificant, position micro-jitter can do this for one frame),
    # where the mesh-distance version would call that "left" -- confirmed
    # directly on pick_up_the_black_bowl_on_the_stove_..._on_the_plate ep1
    # frame 65 (user-reported: "the object left gripper but actually the
    # object is not").
    left_gripper = True
    if active:
        gripper_aabb = _geom_ids_aabb(env, _gripper_contact_geom_ids(env))
        object_aabb = _geom_ids_aabb(env, _object_geom_ids(env, active))
        if gripper_aabb is not None and object_aabb is not None:
            left_gripper = not _aabb_intersects(gripper_aabb, object_aabb)
        else:
            try:
                gripper = env.robots[0].gripper
                model = env.get_object(active)
                left_gripper = not bool(env.check_contact(model, gripper.important_geoms.get("left_fingerpad")) or
                                         env.check_contact(model, gripper.important_geoms.get("right_fingerpad")))
            except Exception:
                left_gripper = not object_grasped

    object_supported = bool(active and _touches_anything(env, active))

    # object_released: ported RoboCasa's own exact 3-branch definition
    # verbatim (predicates.py's object_released) -- gripper_is_opening OR
    # prev_gripper_is_opening OR (active and object_supported). No other
    # branch exists in RoboCasa's real implementation.
    #
    # - prev_gripper_is_opening: gripper_is_opening is a raw single-frame
    #   sign check (no debounce) that can dip False for exactly the one
    #   frame contact actually breaks, even though it's opening the frame
    #   before and after -- ORing in last frame's value closes that gap.
    #
    # - `active and object_supported`: covers a release where the arm moves
    #   the gripper away (or simply stops actively gripping) without ever
    #   opening the fingers, because the object is already resting on solid
    #   support by the time contact breaks -- confirmed on KITCHEN_SCENE3_
    #   turn_on_the_stove_and_put_the_moka_pot_on_it: object_supported was
    #   already True *while still grasped*, for several frames before
    #   release (the pot touches the stove before the grasp ends) -- a
    #   real, deliberate, already-safe placement, not an accidental drop,
    #   that a gripper-opening-only check couldn't recognize at all.
    #
    # A 4th branch (`gripper_frac < GRIPPER_OPEN_FRACTION_THRESHOLD`) was
    # removed 2026-09-21 (dependency-tree audit): it had no RoboCasa
    # counterpart (GRIPPER_OPEN_FRACTION_THRESHOLD is not referenced
    # anywhere in object_released there) and was strictly more permissive
    # than RoboCasa's real definition -- it could classify a drop as a
    # deliberate "release" purely because the gripper already happened to
    # read below threshold at the drop frame, even with no opening motion
    # and no support, which RoboCasa would instead leave to the LTL's own
    # re-grasp fallback disjunct to adjudicate.
    object_released = bool(
        object_dropped
        and (
            gripper_is_opening
            or state.get("prev_gripper_is_opening", False)
            or (active is not None and object_supported)
        )
    )
    state["prev_gripper_is_opening"] = gripper_is_opening
    # task_success escape REMOVED (2026-09-21, verification audit): this used
    # to OR task_success into object_settled ("supported AND support-type-
    # matches AND (task_success OR (stable AND gripper_away))") to cover
    # LIBERO's demo hdf5s stopping recording within a handful of frames of
    # task success (2026-09-09 original rationale, preserved below for
    # history). That escape does not exist anywhere in RoboCasa's own
    # _object_settled (predicates.py: a plain 4-way AND of supported/support-
    # type/stable/gripper-away, no success-based shortcut at all), and it is
    # no longer needed here either: run_monitor_on_privileged.py's later,
    # more general fix (01546d1, 2026-09-18) already treats a timeout-bounded
    # property whose `timeout` atom never becomes True anywhere in a
    # recorded trace as a truncation artifact rather than a violation, and
    # that generic fix explicitly "applies identically to LIBERO's own
    # predicates.py" (see its own comment in run_monitor_on_privileged.py) --
    # it supersedes this escape for the genuine truncated-recording case the
    # escape was originally written for.
    #
    # Found via corpus-wide verification sweep (v29_2026-09-21 corpus, 400
    # episodes): the escape does much more than rescue truncated traces --
    # task_success routinely goes True several frames *before* the object is
    # even released (KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_
    # on_it ep0: task.success=True at frame 261 while object_grasped is still
    # True; object_dropped only fires at frame 267), so object_settled could
    # read True the instant success is detected regardless of whether the
    # object had actually left the gripper or come to rest -- 246/400 sampled
    # episodes hit "object_dropped -> object_settled already True the same
    # frame, with object_stable_settle/gripper_away both False" purely
    # because of this escape. Since object_settled is the literal "until"
    # target in rc_released_object_eventually_settles' shared main_ltl
    # (specs.py: "G(object_dropped -> (!release_object_settle_timeout U
    # object_settled) | ...)"), this made the obligation trivially satisfied
    # on the triggering frame for a large fraction of the corpus -- a much
    # stronger and less faithful escape than RoboCasa's mechanism (which has
    # none) and not something a truncation-only fix should be doing.
    # Removed to match RoboCasa's real object_settled exactly; genuine
    # end-of-recording truncation is left to the shared, more precise
    # run_monitor_on_privileged.py fix instead.

    # settle-timeout watchdog (2026-09-20 fix): starts on object_dropped, and
    # -- unlike the previous version -- stays decoupled from `active` for as
    # long as it's pending, mirroring RoboCasa's own settle_obj_name split
    # (predicates.py, ~line 3948: "settle_obj_name = settle_watch_object if
    # awaiting_settle and settle_watch_object is not None else active_object",
    # with its own explicit comment at ~4391-4402 on exactly this bug class).
    # Previously, object_settled/object_supported/object_stable/gripper_away
    # were all computed for `active` -- which gets reassigned to a NEWLY
    # grasped object the instant any new grasp is detected (see
    # state["active_object"] above) -- so if the robot grasped a second,
    # different object before the first (dropped) object's settle window
    # elapsed, this watchdog silently started evaluating the SECOND object's
    # stability/support/gripper-distance while nominally still checking
    # whether the FIRST one settled. Concrete failure confirmed on
    # LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_
    # basket: drop the cream cheese into the basket, then grasp the butter
    # before the cream cheese's SETTLE_TIMEOUT_FRAMES window elapses -- the
    # old code would evaluate the BUTTER's stability while the settle-watch's
    # own object identity (`watch["object"]`) still said "cream cheese",
    # and the watch's own clearing condition (`active == watch["object"]`)
    # could then never be true again for as long as the butter stayed
    # active, so the watch could only ever resolve via timeout, never via
    # real settle detection -- producing a false release_object_settle_
    # timeout / rc_released_object_eventually_settles violation on a
    # placement that was actually fine.
    #
    # settle_obj_name is the watched (dropped) object while a watch is
    # pending, falling back to `active` only when nothing is currently being
    # watched (mirrors RoboCasa's fallback exactly) -- object_supported/
    # object_stable/gripper_away/object_settled below are now all computed
    # against settle_obj_name, not `active`, so they keep tracking the
    # actually-watched object's real state regardless of what gets grasped
    # afterward.
    if object_dropped:
        state["settle_watch"] = {"object": active, "age": 0}
    watch = state["settle_watch"]
    settle_obj_name = watch["object"] if watch is not None else active

    object_supported_settle = bool(settle_obj_name and _touches_anything(env, settle_obj_name))
    # 2026-09-21: real port of RoboCasa's own object_support_type_matches_any_settle
    # (predicates.py ~4407-4409, `_object_support_type_matches_any(settle_obj_name)`
    # -- see `_support_type_matches_any`'s own docstring for the mechanism and
    # its one documented narrow gap, floor exclusion). Previously this
    # composition used the (differently-scoped, place-preconditions-only)
    # `support_type_matches_object` stub, hardcoded True -- now genuinely
    # computed and scoped to settle_obj_name, matching RoboCasa's real
    # object_settled composition exactly.
    object_support_type_matches_any_settle = bool(
        settle_obj_name is not None and _support_type_matches_any(env, settle_obj_name)
    )
    object_stable_settle = (
        bool(object_stable_by_name.get(settle_obj_name, True)) if settle_obj_name is not None else True
    )
    # Real mesh/geom distance, not eef-to-body-origin distance -- see
    # _gripper_far_from_object's own docstring for why (LIBERO objects have
    # real physical extent, same root cause RoboCasa's predicates.py already
    # fixed for its own gripper_away_from_object, 2026-09-08). Falls back to
    # "nothing being watched" (vacuously away) when there's no settle_obj_name.
    # Scoped to settle_obj_name, not `active`, matching RoboCasa's own
    # exported gripper_away_from_object (predicates.py, ~line 4388-4390),
    # which is likewise settle_obj_name-scoped, not obj_name/active_object-
    # scoped.
    gripper_away = (
        True if settle_obj_name is None else _gripper_far_from_object(env, settle_obj_name, MESH_GRIPPER_FAR_THRESHOLD)
    )
    object_settled = bool(
        object_supported_settle
        and object_support_type_matches_any_settle
        and object_stable_settle
        and gripper_away
    )

    # Clears only on object_settled or the timeout itself (matches RoboCasa's
    # own awaiting_settle clearing condition, predicates.py ~4434-4439:
    # `if awaiting_settle and (object_settled or release_object_settle_timeout):`
    # -- grepping all occurrences of awaiting_settle there confirms no branch
    # clears the watch merely because the same object got re-grasped). A
    # regrasp-clears-the-watch branch was removed here 2026-09-21
    # (dependency-tree audit): it had no RoboCasa counterpart and silently
    # zeroed the exported release_object_settle_timeout signal on regrasp,
    # diverging from RoboCasa's real behavior of keeping the original watch
    # running (and still able to report a timeout) even while the object is
    # currently re-held.
    release_settle_timeout = False
    if watch is not None:
        if object_settled:
            state["settle_watch"] = None
        else:
            watch["age"] += 1
            release_settle_timeout = watch["age"] > SETTLE_TIMEOUT_FRAMES

    for name in _movable_object_names(env):
        state["prev_positions"][name] = _body_pos(env, name)
        state["prev_quats"][name] = _body_quat(env, name)

    predicates["object_grasped"] = _entry(object_grasped, "gripper bilaterally contacts a movable object", active)
    # object_grasped_raw (2026-09-20): updated from an initial same-day
    # compatibility-only alias of object_grasped (added just to stop the
    # shared specs.py's rc_grasp_remains_synced_until_dropped formula from
    # erroring on a missing key) to a genuinely separate, undebounced
    # signal, once GRASP_CANDIDATE_PERSISTENCE_FRAMES's real-data
    # investigation confirmed LIBERO's grasp signal does flicker the same
    # way RoboCasa's did -- see that constant's own comment for the
    # concrete frame-73/frame-116 evidence. object_grasped above is now the
    # debounced candidate; this is the raw one, matching RoboCasa's own
    # naming convention exactly.
    predicates["object_grasped_raw"] = _entry(object_grasped_raw_value, "gripper bilaterally contacts a movable object (undebounced -- see object_grasped's own comment for why these two now differ)")
    predicates["object_stable"] = _entry(object_stable, "active object linear/angular motion, relative to its current support, below threshold (debounced -- see STABLE_PERSISTENCE_FRAMES)")
    predicates["object_stable_relative"] = _entry(object_stable, "aliased to object_stable -- both are now genuinely relative-to-support and debounced, not just aliased in name")
    predicates["object_sync"] = _entry(object_sync, "active object moves in sync with gripper")
    predicates["object_upright"] = _entry(object_upright, "active object z-axis aligned with world z")
    predicates["object_dropped"] = _entry(object_dropped, "a grasp just ended")
    predicates["object_left_gripper"] = _entry(left_gripper, "active object mesh no longer touches gripper")
    predicates["object_released"] = _entry(object_released, "drop coincided with gripper opening")
    predicates["object_supported"] = _entry(object_supported, "active object touches some scene geometry")
    predicates["object_supported_on_correct"] = _entry(object_supported, "aliased to object_supported in v0")
    predicates["gripper_away_from_object"] = _entry(gripper_away, "gripper moved away from the settle-watched object (settle_obj_name -- the dropped object being watched, not necessarily whatever's currently active/grasped)")
    predicates["object_settled"] = _entry(object_settled, "settle-watched object (settle_obj_name) is supported, stable, and gripper away")
    predicates["object_settle_timeout"] = _entry(release_settle_timeout, "aliased to release_object_settle_timeout in v0")
    predicates["release_object_settle_timeout"] = _entry(release_settle_timeout, "dropped object failed to settle within timeout")
    predicates["gripper_is_opening"] = _entry(gripper_is_opening, "gripper closed-fraction decreasing")
    predicates["gripper_is_closing"] = _entry(gripper_is_closing, "gripper closed-fraction increasing")

    # --- skill onset + pick preconditions -------------------------------
    pick_onset_state = state["pick_onset"]
    any_pick_onset = False
    prev_fired_pick_object = pick_onset_state.get("fired_object")
    # Grasp gate (2026-09-20, RoboCasa predicates.py:5088-92's pick_onset_cond
    # -- `not prev_object_grasped and ... and not object_grasped`): RoboCasa's
    # entire approach-tracking result is blocked from firing while the
    # gripper holds anything, in either this frame or the previous one.
    # `not prev_object_grasped` needs its own dedicated state key here
    # (prev_grasped_object/prev_object_grasped_for_sync above are already
    # overwritten to *this* frame's value earlier in this function, so they
    # can't be reused as "previous frame" from this point on).
    prev_object_grasped_for_pick_onset = bool(
        state.get("prev_object_grasped_for_pick_onset", False)
    )
    grasp_blocks_pick_onset = object_grasped or prev_object_grasped_for_pick_onset
    focus_pick_object = active if object_grasped else None
    # Single-nearest-candidate redesign (2026-09-21, AABB-accuracy false-
    # positive investigation): replaces a per-object-independent streak dict
    # (every movable object accumulating its own "near + decreasing
    # distance" streak in parallel) with RoboCasa's REAL mechanism
    # (predicates.py ~4980-5088: a single tracked `pick_approach_candidate`
    # -- always whichever object is CURRENTLY nearest the gripper -- whose
    # streak resets to 1 the instant a DIFFERENT object becomes nearest).
    # The per-object version was a deliberate, explicit divergence at the
    # time (see git history), justified only for gating progress during an
    # active carry (a second object's streak could keep building while
    # object A was held, priming it to fire the instant release lifted the
    # grasp gate) -- that carry-gating concern is independently preserved
    # below via grasp_blocks_pick_onset's own reset, unchanged.
    #
    # But the per-object-parallel design has a much larger, confirmed cost:
    # every movable object accumulates its OWN streak independently of
    # whether it's actually the object the gripper is closing in on, so a
    # bystander object merely sitting along/near the gripper's transit path
    # toward its real target can independently satisfy "near + decreasing
    # distance" for SKILL_ONSET_FRAMES and fire its own spurious pick onset
    # -- misattributing the ENTIRE pick-preconditions check to the wrong
    # object. Confirmed via a corpus-wide sweep of v32's "gripper path was
    # obstructed" rc_pick_preconditions_safe violations: 40 of 41 instances
    # had first_non_accepting_role_sets.focus_pick_object != the episode's
    # real final focus_pick_object (e.g. pick_up_the_tomato_sauce_and_
    # place_it_in_the_basket ep4 frame 23: onset fired for orange_juice_1,
    # a bystander object on the same shelf the gripper transits near while
    # actually reaching for tomato_sauce_1, which the episode's own final
    # role sets confirm as the real grasp target). Direct frame data showed
    # the geometry itself (gripper/target/blocker AABBs) was computed
    # correctly and tightly -- the corridor genuinely passed through
    # tomato_sauce_1's real footprint on the way to orange_juice_1's real
    # position -- but there never was a genuine "pick attempt" on
    # orange_juice_1 in the first place; it's a bystander object the
    # gripper happens to pass nearest on its way to the real target, not
    # something the human demonstrator was ever reaching for. RoboCasa's
    # single-nearest-candidate design structurally prevents this: a
    # bystander only ever gets a chance to accumulate progress while it is
    # THE single nearest object, and the moment the gripper continues past
    # it toward the real (farther, but genuinely being approached) target,
    # either the bystander's own distance starts increasing again (streak
    # stalls before crossing SKILL_ONSET_FRAMES) or the real target becomes
    # nearer first (nearest-object identity switches, discarding the
    # bystander's partial progress) -- both outcomes were confirmed via the
    # same real corridor's own numbers (tomato_sauce_1's own gripper-AABB
    # gap, ~0.098m, was strictly larger than orange_juice_1's, ~0.046m, at
    # the exact violation frame, meaning orange_juice_1 genuinely was the
    # single nearest object at that instant -- the fix is not "recompute
    # nearest differently," it's "don't let farther, non-nearest objects
    # accumulate progress in parallel while a nearer one is being tracked").
    pick_onset_prev_dist = state.setdefault("pick_onset_prev_dist", {})
    candidate_dists = {}
    if not grasp_blocks_pick_onset:
        for name in _movable_object_names(env):
            if name == grasped_name:
                continue
            pos = _body_pos(env, name)
            # dist (2026-09-21, distance-basis audit): gripper-AABB-to-object
            # distance (_gripper_target_distance, degrading to gripper-AABB-to-
            # object-center-point, then raw point/point only if no AABB resolves
            # at all) -- see NEAR_OBJECT_THRESHOLD's own comment for why the
            # old raw eef-point-to-body-origin-point basis systematically
            # overestimated the true gap for any object with real physical
            # extent.
            object_reach_aabb = _object_aabb(env, name)
            dist = _gripper_target_distance(env, eef_pos, gripper_reach_aabb, pos, object_reach_aabb)
            if dist is not None:
                candidate_dists[name] = dist
    nearest_name, nearest_dist = None, None
    if candidate_dists:
        nearest_name, nearest_dist = min(candidate_dists.items(), key=lambda kv: kv[1])
    near = bool(nearest_dist is not None and nearest_dist < NEAR_OBJECT_THRESHOLD)
    prev_candidate = pick_onset_state.get("candidate")
    prev_count = int(pick_onset_state.get("count", 0))
    prev_false_count = int(pick_onset_state.get("false_count", 0))
    # previous_nearest_distance: this SAME object's own recorded distance
    # last frame (whether or not it was the tracked candidate then) --
    # mirrors RoboCasa's own prev_gripper_object_distances dict, which
    # records every candidate's distance every frame, not just the winning
    # one, so a freshly-switched-to candidate's own decreasing-distance
    # history is still available the instant it becomes nearest.
    previous_nearest_distance = pick_onset_prev_dist.get(nearest_name) if nearest_name is not None else None
    moving_towards = bool(
        nearest_dist is not None
        and previous_nearest_distance is not None
        and nearest_dist < previous_nearest_distance - 1e-4
    )
    if grasp_blocks_pick_onset:
        # No progress accumulates while carrying anything (matches
        # RoboCasa's real behavior for free: the held object is essentially
        # always the single nearest candidate, distance ~0, starving any
        # other object of a chance to become the tracked candidate at all --
        # LIBERO explicitly excludes the held object itself as a candidate
        # above, so this explicit reset reproduces the same net effect).
        count, false_count, candidate = 0, 0, None
    elif moving_towards and nearest_name == prev_candidate:
        count, false_count, candidate = prev_count + 1, 0, nearest_name
    elif moving_towards:
        count, false_count, candidate = 1, 0, nearest_name
    elif prev_candidate is not None and prev_count >= SKILL_ONSET_FRAMES:
        # Grace period (RoboCasa's own pick_approach_false_count): tolerates
        # up to SKILL_ONSET_FRAMES consecutive non-decreasing/candidate-
        # switched frames without discarding an already-qualified streak --
        # confirmed necessary via real approach data (final pre-contact
        # overshoot ticks the distance back up slightly for a few frames
        # right before contact; a hard zero-tolerance reset broke genuine
        # onsets that legitimately fire moments later).
        false_count = prev_false_count + 1
        if false_count < SKILL_ONSET_FRAMES:
            count, candidate = prev_count, prev_candidate
        else:
            count, false_count, candidate = 0, 0, None
    else:
        count, false_count, candidate = 0, 0, None
    for name, dist in candidate_dists.items():
        pick_onset_prev_dist[name] = dist
    pick_onset_state["candidate"] = candidate
    pick_onset_state["count"] = count
    pick_onset_state["false_count"] = false_count
    state["prev_object_grasped_for_pick_onset"] = object_grasped

    gripper_moving_towards_object = count >= SKILL_ONSET_FRAMES
    pick_approach_object = candidate if gripper_moving_towards_object else None
    fired_object = pick_onset_state.get("fired_object")
    # genuinely_disengaged (2026-09-21 fix, repeated-onset-during-a-single-
    # continuous-interaction audit -- preserved from the per-object design's
    # own fix, not just object_grasped/candidate-switch alone): whether the
    # FIRED object has actually left the near-object region, not merely
    # "the tracked candidate switched or its streak lapsed". Confirmed via
    # real data (push_the_plate_to_the_front_of_the_stove ep5, v31 corpus)
    # that candidate/streak lapsing alone is not sufficient evidence a real
    # interaction ended: during a sustained push the gripper and object move
    # together at close to the same velocity, so eef-to-object distance
    # stays flat (never decreasing enough to keep "moving_towards" true, so
    # the streak lapses and pick_approach_object drops to None) while the
    # object is still genuinely close (near) the whole time -- clearing
    # fired_object here would let a second spurious onset fire on the same
    # still-ongoing interaction. Only clear fired_object on a lapsed/
    # switched candidate when the fired object's OWN current distance says
    # it's genuinely far now (>= NEAR_OBJECT_THRESHOLD); missing distance
    # data is treated as NOT disengaged (safe default, matches the original
    # per-object entry's same convention).
    fired_dist = candidate_dists.get(fired_object) if fired_object is not None else None
    fired_genuinely_disengaged = bool(fired_dist is not None and fired_dist >= NEAR_OBJECT_THRESHOLD)
    if object_grasped:
        fired_object = None
    elif fired_object is not None and (pick_approach_object is None or pick_approach_object != fired_object):
        if fired_genuinely_disengaged:
            fired_object = None
    # Fixture-contact suppression (2026-09-20, ported from RoboCasa's own
    # pick_onset_cond fix): only ever considers movable OBJECTS as pick-onset
    # candidates -- a fixture (a drawer being closed, a cabinet door) is
    # never itself a candidate, so a pick onset here could misattribute a
    # fixture-interaction action to whichever object happens to be nearby,
    # the same way RoboCasa's did for LoadDishwasher/KettleBoiling. Uses
    # last frame's raw robot_fixture_contact reading (computed later in this
    # same function, at the mechanism-safety section below -- a one-frame
    # lag is negligible given real fixture manipulation holds contact for
    # many consecutive frames, matching RoboCasa's own reasoning).
    #
    # fixture_is_opening_raw/fixture_is_closing_raw OR'd in (2026-09-21,
    # KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_
    # and_close_it ep3): confirmed via real frame data that this was NOT a
    # one-frame-staleness/latch bug (the hypothesis this session started
    # with) -- robot_fixture_contact_raw stayed False for the entire
    # frames-190-224 drawer-opening interval, not just one transition
    # frame, so no reordering of when it's computed would have helped.
    # Root cause is this module's own documented, pre-existing v0
    # imprecision (see this file's module docstring under "fixture-skill
    # onset": "LIBERO's handle geoms frequently never register raw contact
    # with the gripper during a real pull at all, even while the joint is
    # visibly, continuously articulating") -- robot_fixture_contact is a
    # pure contact-geom check and structurally misses drawer/handle
    # interactions the same way it was already known to for the fixture-
    # skill onsets (which is why those use proximity instead of contact).
    # fixture_is_opening/closing are driven by joint open-fraction deltas,
    # not contact geoms, so they don't share that gap; ORing them in here
    # (also read one-frame-stale, same rationale as robot_fixture_contact_
    # raw -- fixture articulation likewise holds for many consecutive
    # frames) gives pick-onset suppression a working signal for exactly
    # the slide/hinge-jointed-fixture interactions robot_fixture_contact
    # alone misses, without touching robot_fixture_contact_raw itself
    # (still needed for fixtures/contacts it does detect correctly).
    if (
        near
        and pick_approach_object is not None
        and fired_object is None
        and not bool(state.get("robot_fixture_contact_raw", False))
        and not bool(state.get("fixture_is_opening_raw", False))
        and not bool(state.get("fixture_is_closing_raw", False))
    ):
        fired_object = pick_approach_object
        any_pick_onset = True
        if focus_pick_object is None:
            focus_pick_object = pick_approach_object
    pick_onset_state["fired_object"] = fired_object

    any_pick_onset_end = bool(
        prev_fired_pick_object is not None and fired_object != prev_fired_pick_object
    )

    if focus_pick_object is None and not object_grasped:
        # nearest ungrasped object, for object_region_clear's reference point
        best, best_d = None, None
        for name in _movable_object_names(env):
            pos = _body_pos(env, name)
            if pos is None or eef_pos is None:
                continue
            d = float(np.linalg.norm(eef_pos - pos))
            if best_d is None or d < best_d:
                best, best_d = name, d
        focus_pick_object = best

    object_region_blockers = _object_region_blockers(env, focus_pick_object)
    object_region_clear = bool(focus_pick_object is not None and not object_region_blockers)
    object_upright_if_receptacle_default = True
    if focus_pick_object:
        category = object_category_from_instance_name(focus_pick_object)
        if object_is_receptacle_category(category):
            object_upright_if_receptacle_default = _upright(_body_quat(env, focus_pick_object))
    # focus_pick_object's OWN stability, not the currently-`active` (grasped)
    # object's -- at onset time the two are almost always different (the
    # object being approached generally isn't grasped yet, so `active`/
    # `object_stable` above reflect the *previous* grasp cycle's object, or
    # nothing at all). Found via `put_the_bowl_on_the_plate` (2026-09-09):
    # `object_stable` was checking the wrong object entirely at every real
    # pick onset in this corpus. Reads object_stable_by_name (2026-09-20,
    # debounced + relative-to-support, see that dict's own comment above),
    # not a separate raw per-frame call -- mirrors RoboCasa's own
    # pick_object_stable, which reads its object_stable_by_name directly
    # (predicates.py, ~line 5281) rather than a second independently-tracked
    # signal for the same concept.
    focus_pick_stable = (
        bool(object_stable_by_name.get(focus_pick_object, False)) if focus_pick_object else object_stable
    )
    # 2026-09-16 (explicit user decision): RoboCasa's REAL preconditions_
    # satisfied_pick (predicates.py:4000, `_bool(object_region_clear and
    # pick_object_stable)`) never actually ANDs in object_upright_if_
    # receptacle, despite the top-level monitor/predicates.py generic
    # fallback's default doing so and specs.py's own docstring text
    # describing it as included -- confirmed by reading RoboCasa's real
    # composition line directly, not the aspirational docstring/fallback.
    # This file mirrors the simulator override that's actually reported at
    # runtime, not the unused generic fallback, so upright is intentionally
    # left out of the AND here too (object_upright_if_receptacle is still
    # computed and emitted below as its own atom, unchanged).
    preconditions_satisfied_pick = bool(object_region_clear and focus_pick_stable)
    # pick_precondition_escape (2026-09-09, ported from RoboCasa's own
    # predicates.py, same root cause): skill_pick_onset fires the instant
    # the gripper has been near/approaching for SKILL_ONSET_FRAMES, but a
    # corpus-wide sweep showed the dominant rc_pick_preconditions_safe
    # failure across LIBERO's bowl/plate tasks is "object not yet stable"
    # at that exact instant (freshly-placed scene objects still settling,
    # or nudged by the approach itself) -- consistent with a real object
    # that goes on to become perfectly graspable a few frames later, not a
    # genuinely unsafe pick. Without this escape (simply never implemented
    # in the initial v0 port -- `pick_precondition_escape` defaults to
    # False when a property never emits it at all, per monitor/
    # predicates.py's _predicate_value), G(skill_pick_onset ->
    # (preconditions_satisfied_pick | F(pick_precondition_escape))) can
    # never recover from a single transient-instability onset for the rest
    # of the episode -- confirmed as the dominant failure signature for
    # nearly every put_the_black_bowl.../place_it_on_the_plate task in this
    # corpus. Same "latch on first failure, only-if-not-already-pending,
    # clear once resolved" shape as RoboCasa's.
    if (
        any_pick_onset
        and not preconditions_satisfied_pick
        and state.get("pick_onset_pending_object") is None
    ):
        state["pick_onset_pending_object"] = focus_pick_object
    pick_onset_pending_object = state.get("pick_onset_pending_object")
    pick_precondition_escape = False
    if pick_onset_pending_object is not None:
        pending_stable = bool(object_stable_by_name.get(pick_onset_pending_object, False))
        pending_region_clear = not _object_region_blockers(env, pick_onset_pending_object)
        pick_precondition_escape = bool(pending_stable and pending_region_clear)
        if pick_precondition_escape:
            state["pick_onset_pending_object"] = None

    predicates["skill_pick_onset"] = _entry(any_pick_onset, "gripper approached an ungrasped object for the onset window")
    predicates["skill_pick_onset_end"] = _entry(any_pick_onset_end, "a previously-latched pick attempt concluded (grasped or gave up)")
    predicates["object_region_clear"] = _entry(object_region_clear, "no other object's AABB obstructs the gripper-to-pick-target swept path")
    # Exported (2026-09-20) mirroring RoboCasa's own pick_object_stable
    # export (predicates.py:9315): the object actually gated by
    # preconditions_satisfied_pick's composition (focus_pick_object), not
    # the generic object_stable (active_object) -- repeated_violation_
    # monitor.py's explanation-string generator now prefers this key so
    # violation text names the right object in dual-object tasks.
    predicates["pick_object_stable"] = _entry(focus_pick_stable, "focus_pick_object's own stability (debounced, relative-to-support)")
    # 2026-09-16 (explicit user decision): object_upright_if_receptacle_
    # default was already being computed above (real _upright() check
    # against the focus object's own quaternion, gated on it actually being
    # a receptacle category) but was never emitted into the predicates dict
    # -- absent, not stubbed, the exact bug class already found for place's
    # composition. RoboCasa emits this as its own atom (predicates.py's
    # object_upright_if_receptacle export) independent of whether
    # preconditions_satisfied_pick's own AND-composition uses it; mirrored
    # here the same way.
    predicates["object_upright_if_receptacle"] = _entry(object_upright_if_receptacle_default, "receptacle-category focus object is upright (True if not a receptacle)")
    predicates["preconditions_satisfied_pick"] = _entry(preconditions_satisfied_pick, "pick preconditions AND-composition")
    predicates["pick_precondition_escape"] = _entry(pick_precondition_escape, "pending pick object later became stable and region-clear")

    # --- place preconditions --------------------------------------------
    # Retargeted 2026-09-15 (explicit user decision, matching RoboCasa's own
    # predicates.py place-onset retarget) from object_released to
    # object_dropped: object_dropped fires on every grasp-ending edge for
    # any reason (object_released is a strict subset, additionally
    # requiring gripper-opening/settled evidence), so place preconditions
    # now get checked on accidental drops too, not just deliberate releases.
    #
    # Re-examined 2026-09-21 (repeated-onset-during-a-single-continuous-
    # interaction audit): object_dropped inherits object_grasped's debounced-
    # but-still-raw-contact-signal fragility -- GRASP_CANDIDATE_PERSISTENCE_
    # FRAMES (5) consecutive frames of real bilateral contact is enough to
    # accept a brief, non-deliberate contact blip as a "grasp", so a genuine
    # grasp-then-release flip-flop can fire object_dropped (and thus
    # skill_place_onset) more than once for what's really one continuous
    # interaction. Confirmed via real data
    # (KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_
    # cabinet_and_close_it ep0, v31 corpus): object_grasped/object_dropped
    # fire True->False->True->False at frames 159/169/174 while the
    # eef-to-bowl distance stays flat at ~0.06-0.07m the entire window
    # (never approaches, then never actually leaves, contact range) --
    # object_dropped fires twice (159, 174) for this one non-event.
    #
    # First attempt (reverted same day): retarget skill_place_onset off
    # object_dropped entirely, onto left_gripper_edge (the rising edge of
    # the AABB-overlap-based object_left_gripper signal), mirroring
    # RoboCasa's own predicates.py identical retarget from 2026-09-17
    # (search "object_left_gripper_edge" there). This does fix the bowl
    # case (left_gripper never separates across 159-180, one real edge at
    # 180) -- but a corpus-wide check (all 291 real episodes in the v31
    # corpus with a final release) found object_left_gripper never clears
    # by episode end in 124/291 (42.6%) of them, because LIBERO's episodes
    # -- unlike RoboCasa's -- typically end within single-digit frames of
    # the final placement once task success is detected, giving the
    # gripper's AABB no time to actually separate before the trace stops.
    # Under the left_gripper_edge retarget, skill_place_onset would then
    # NEVER fire at all for the single most important placement in
    # ~43% of episodes -- a severe false-negative regression, not an
    # acceptable trade against RoboCasa's own (longer-tailed-episode-
    # validated) net improvement. Reverted; object_left_gripper/
    # left_gripper itself is unaffected and still exported as before.
    #
    # Final fix: keep object_dropped as the trigger (preserving "catches
    # accidental drops too" and "fires promptly, doesn't need the episode
    # to keep running after the real event" from both the 2026-09-15 and
    # the reverted attempt above), but gate a REPEATED firing for the SAME
    # object behind the same genuinely_disengaged-style latch used for
    # skill_pick_onset's identical bug (see that predicate's own 2026-09-21
    # comment) -- once skill_place_onset has fired for a given dropped
    # object, it won't fire again for that same object until the eef has
    # genuinely left its vicinity (dist >= NEAR_OBJECT_THRESHOLD, the same
    # existing near/far boundary, not a new invented number), distinguishing
    # "still lingering near this object, same interaction" from "genuinely
    # moved on, a new drop of this object is a new event." Verified against
    # the bowl case above: dist never reaches 0.09 during 159-174, so the
    # frame-159 latch blocks the frame-174 repeat, exactly as desired, and
    # ALSO verified against a real accidental-drop-during-final-placement
    # case (KITCHEN_SCENE8_put_both_moka_pots_on_the_stove ep1, moka_pot_1's
    # final placement: object_dropped fires at both 418 and 432, eef-to-pot
    # distance flat at 0.0723m the entire 400-438 window, i.e. the exact
    # same non-disengaging pattern) -- collapses to one onset (418) instead
    # of two, while the episode's *other* real, separate placement
    # (moka_pot_2 at frame 189, a completely different object whose own
    # latch is independent) is untouched.
    place_onset_fired = state.setdefault("place_onset_fired", {})
    skill_place_onset = False
    if object_dropped and dropped_object_name is not None:
        if not place_onset_fired.get(dropped_object_name, False):
            skill_place_onset = True
            place_onset_fired[dropped_object_name] = True
    # Re-arm: clear any object's latch once the eef has genuinely left its
    # vicinity, so a real later drop of the SAME object (after a genuine
    # re-approach/re-grasp cycle) still fires its own new onset.
    for _pon_name in list(place_onset_fired.keys()):
        if not place_onset_fired[_pon_name]:
            continue
        _pon_pos = _body_pos(env, _pon_name)
        # _pon_dist (2026-09-21, distance-basis audit): same gripper-AABB-to-
        # object distance basis as the pick-onset streak above (dist/
        # object_reach_aabb), replacing the previous raw point/point
        # distance -- this latch's re-arm condition must use the exact same
        # near/far boundary skill_pick_onset itself uses (both compared
        # against NEAR_OBJECT_THRESHOLD, the same physically-motivated
        # boundary), so its distance basis has to match too, not just its
        # threshold value.
        _pon_aabb = _object_aabb(env, _pon_name)
        _pon_dist = _gripper_target_distance(env, eef_pos, gripper_reach_aabb, _pon_pos, _pon_aabb)
        if _pon_dist is None:
            continue
        if _pon_dist >= NEAR_OBJECT_THRESHOLD:
            place_onset_fired[_pon_name] = False
    # support_region_target/support_region_target_object (2026-09-20): a
    # live, continuously-re-evaluated guess -- via _infer_landing_target,
    # this file's simplified analog of RoboCasa's own _infer_support/_spos
    # -- of where `active` (still grasped this frame, or just-dropped) is
    # probably headed/has landed. Computed from active_pos (the object's own
    # CURRENT position this frame) every single frame the object is grasped,
    # not just once at object_dropped time -- see _support_region_blockers'
    # own docstring for why this replaced an earlier fixed-origin full-trip
    # sweep (the milk_1 false positive) and an even earlier one-frame-back
    # sweep (a footprint-comparable-segment false negative) before it.
    support_region_target, support_region_target_object = _infer_landing_target(
        env, active, active_pos
    )
    support_region_blockers = _support_region_blockers(
        env,
        active,
        active_pos,
        support_region_target,
        support_region_target_object,
        state.get("carried_content_names", {}).get(active, []) if active else [],
    )
    support_region_clear = bool(not support_region_blockers)
    # support_stable (2026-09-20 fix): checks the live, debounced,
    # relative-to-support stability of the actual support object
    # (object_stable_by_name, see that dict's own comment above) when
    # _infer_landing_target identified a movable-object landing target
    # (support_region_target_object is not None -- a basket/bowl-type
    # receptacle, per _infer_landing_target's own "scan every
    # receptacle-category movable object" scope), falling back to True only
    # when the target is a genuine fixture/static surface or there is no
    # inferred target at all -- mirrors RoboCasa's own _support_stable
    # exactly (predicates.py, ~line 5630: `if sup_kind == "object" ...
    # return object_stable_by_name.get(str(sup_name), False)` vs. `return
    # True` for a fixture support).
    #
    # Previously hardcoded True unconditionally, with a comment claiming
    # "LIBERO supports are static furniture/table in this v0" -- that
    # justification didn't actually hold: _infer_landing_target (added in
    # 911ef1a) can and does identify a movable object (e.g. a basket) as
    # the landing target, and _movable_object_names explicitly includes
    # such receptacle-category objects, so basket/bowl placements -- a
    # common pattern in this corpus -- never got real stability checking on
    # the support side at all.
    support_stable = (
        True
        if support_region_target_object is None
        else bool(object_stable_by_name.get(support_region_target_object, False))
    )
    # 2026-09-21: real port of RoboCasa's own `_support_geometry_valid`
    # (predicates.py ~5650-5702). RoboCasa's object-kind branch (`sup_kind ==
    # "object"`) is a genuine AABB-overlap test between the manipulated
    # object and its support -- exactly what LIBERO's own `_object_aabb`/
    # `_aabb_intersects` primitives (already built for object_region_clear/
    # support_region_clear's swept-path geometry) directly support, with no
    # RoboCasa-only concept involved. RoboCasa's fixture-kind branch instead
    # tests the placement position against the fixture's own live
    # support-region local-frame bounding box (`_fixture_support_aabb`) --
    # LIBERO's `_infer_landing_target` doesn't retain a fixture support's own
    # separate AABB the way RoboCasa's registry does, but it already performs
    # essentially the identical acceptance test at SELECTION time (a fixture
    # only becomes `support_region_target` if the carried object's own
    # current XY position is already within PLACEMENT_MARGIN *
    # SUPPORT_TARGET_XY_MULTIPLIER of that fixture's own AABB, with its top
    # below the object -- see `_infer_landing_target`'s own `_consider`) --
    # so by construction, landing on a fixture at all already implies
    # geometric validity; True here is a real, justified consequence of that
    # upstream selection test, not an unmeasured default (same class of
    # simplification as `target_stable`'s own documented justification, not
    # the same class as the RECEPTACLE_NAME_SUBSTRINGS-taxonomy-limited
    # `support_type_matches_object` piece below).
    # "No candidate identified at all" fallback (2026-09-21, KITCHEN_SCENE8/
    # LIVING_ROOM_SCENE2 residual false-positive audit): previously forced
    # False unconditionally, which conflated two genuinely different real
    # situations -- "still mid-air, nothing plausible below it yet" (a real
    # gap) vs. "already resting on a real, if unregistered, surface" (e.g. a
    # bare table/counter -- LIBERO's fixtures_dict has no such entry at all,
    # see _infer_landing_target's own docstring). Confirmed only 2/400+
    # episodes in the v33 corpus actually hit this branch as a violation, so
    # this is narrow in practice, not a broad structural rewrite -- but for
    # the one that does (LIVING_ROOM_SCENE2 ep3, butter_1 frame 196), the
    # object has already come to a complete, motionless rest, 20cm+ from the
    # only registered receptacle in the scene. `_object_touches_unregistered_
    # surface` (real, undebounced, non-robot contact) distinguishes the two
    # cases directly -- see its own docstring for why it, and not
    # object_supported (gated on ANY contact, including the gripper), is the
    # right test here.
    # Memoized (2026-09-21, performance pass): computed at most once per
    # frame -- both this predicate and support_type_matches_object below hit
    # the identical "support_region_target is None" branch and would
    # otherwise each independently re-scan env.sim.data.contact.
    active_touches_unregistered_surface = (
        _object_touches_unregistered_surface(env, active)
        if active is not None and support_region_target is None
        else None
    )
    if active is None:
        support_geometry_valid = False
    elif support_region_target is None:
        support_geometry_valid = active_touches_unregistered_surface
    elif support_region_target_object is not None:
        support_aabb = _object_aabb(env, support_region_target_object)
        obj_aabb = _object_aabb(env, active)
        if support_aabb is None or obj_aabb is None:
            support_geometry_valid = False
        elif object_is_receptacle_category(object_category_from_instance_name(support_region_target_object)):
            # Literal port of RoboCasa's `if _object_is_receptacle(sup_name):
            # return True` -- containment (object nested inside a
            # bowl/basket) is a different geometric relationship than
            # resting-on-top overlap, and is already validated elsewhere
            # (object_in_fixture-style containment checks), not by an AABB
            # overlap test here.
            support_geometry_valid = True
        else:
            support_geometry_valid = _aabb_intersects(
                _expanded_aabb(support_aabb, SUPPORT_CLUTTER_Z_TOLERANCE), obj_aabb
            )
    else:
        support_geometry_valid = True

    # 2026-09-21: real port of RoboCasa's own place-preconditions
    # `_support_type_matches()` (predicates.py ~5718-5781) -- a DIFFERENT
    # function from `_support_type_matches_any` above (that one feeds
    # object_settled; this one gates preconditions_satisfied_place, keyed to
    # the specific inferred landing support, not "any" surface). RoboCasa's
    # real branches: non-food manipulated objects vacuously pass; an
    # object-kind support passes automatically if it's itself receptacle-
    # shaped (containment, not resting-on-top, e.g. an item placed inside a
    # basket) -- directly portable via `object_is_receptacle_category`, the
    # same primitive `support_geometry_valid` above already uses. Otherwise
    # (a food item resting ON TOP of some other non-receptacle object)
    # RoboCasa additionally requires the support to be one of the task's own
    # registered target objects (target_object_names/target_objects_by_
    # object/active_target_object_names) -- a per-task role registry
    # RoboCasa's fixture/object configs provide via AST-parsing each task's
    # `_check_success`. LIBERO's BDDL `objects_dict` doesn't expose that same
    # registry directly, but `_build_contact_role_registry` (this file, ~line
    # 2013) already builds the equivalent per-task role registry from
    # `env.parsed_problem` for the contact-role/forbidden_contact
    # classification (see that function's own docstring and the 2026-09-21
    # fixture-readiness/support-type/contact-role audit) -- its
    # `target_objects_by_object` entry is the exact same concept as
    # RoboCasa's parameter of the same name (a manip object's own
    # goal-registered "placed onto/into this other movable object" set), so
    # it is reused here read-only (cached per-episode in
    # `state["contact_role_registry"]`, already built earlier in
    # `build_predicate_snapshot` by `_evaluate_contact_policy`) rather than
    # left permissively True. LIBERO has no analog of RoboCasa's other two
    # union sources (`target_object_names`/`active_target_object_names` --
    # broader task-level/currently-active target sets beyond this specific
    # manip object's own registered targets), so the check here is narrower
    # than RoboCasa's full three-way union; verified against the real corpus
    # below that this doesn't false-positive on any genuine placement.
    #
    # RoboCasa's fixture-kind branch excludes bare floor support
    # (`_fixture_is_floor`) and, for structural fixture classes specifically,
    # requires the placement to land in a real registered interior support
    # region. LIBERO's `_infer_landing_target` fixture candidates are real
    # appliance/storage fixtures only (microwave, cabinets, stove,
    # desk_caddy, wine_rack) -- fixtures_dict has no registered floor/
    # tabletop entry at all in this corpus (see `_support_type_matches_any`'s
    # own docstring for the full verification), so the floor exclusion is
    # structurally moot for this branch specifically, not unported: there is
    # no floor candidate this inference could ever select in the first
    # place. RoboCasa's further STRUCTURAL_FIXTURE_CLASSES/interior-support-
    # region distinction has no LIBERO equivalent at all -- defaults to True
    # for any real (non-floor) fixture landing target, matching RoboCasa's
    # own real behavior for every fixture class outside that structural set
    # (a plain Counter/Island/Stove/DishRack, RoboCasa's own code comment
    # confirms, has "no structural-body/interior split to enforce" either).
    #
    # No support candidate identified at all: literal port of RoboCasa's own
    # real fallthrough for this exact combination (sup_kind/sup_name both
    # unresolved) -- a food-type object genuinely mid-air with nothing at
    # all below it is invalid. Extended 2026-09-21 (same audit/same fallback
    # as support_geometry_valid's own identical branch just above -- see
    # `_object_touches_unregistered_surface`'s docstring): "no candidate
    # identified" also covers a food item already resting on a real,
    # unregistered surface (bare table/counter -- LIBERO's fixtures_dict has
    # no such entry), which is a normal, safe surface, not the genuinely
    # mid-air case this fallthrough was meant for. Confirmed via real data
    # (LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_
    # the_basket ep3, frame 196, immediately after support_geometry_valid's
    # own fix landed): butter_1 is motionless on the table at this exact
    # frame, so without this same fallback the geometry fix alone just
    # traded one false "invalid support" reason (geometry) for another
    # (type) at the identical frame.
    manip_is_food = bool(active is not None and any(s in object_category_from_instance_name(active) for s in FOOD_NAME_SUBSTRINGS))
    # 2026-09-21 (role-registry wiring): the permissive-True default
    # documented above is now only used for the genuine receptacle
    # (containment) case. For a non-receptacle object-kind support,
    # `_build_contact_role_registry`'s `target_objects_by_object` (built from
    # this task's own BDDL goal_state On/In facts -- see that function's
    # docstring) is the same underlying concept as RoboCasa's
    # `target_objects_by_object` parameter to its own `_support_type_matches`
    # (predicates.py ~5731-5734: `object_targets.update(target_objects_by_
    # object.get(str(obj_name), set()))`, unioned with the two other
    # per-task-registry sources `target_object_names`/`active_target_object_
    # names` that have no direct LIBERO analog) -- a manip object's own
    # officially-registered "placed onto/into this other movable object" set.
    # Already computed once per episode by `_evaluate_contact_policy` (called
    # earlier in `build_predicate_snapshot`, ~line 3326) and cached in
    # `state["contact_role_registry"]`, so reused here read-only, not
    # recomputed.
    _stm_registry = state.get("contact_role_registry") or {}
    _stm_registered_targets = _stm_registry.get("target_objects_by_object", {}).get(str(active), set()) if active is not None else set()
    if active is None:
        support_type_matches_object = True
    elif support_region_target_object is not None:
        if object_is_receptacle_category(object_category_from_instance_name(support_region_target_object)):
            support_type_matches_object = True  # containment (e.g. placed inside a basket/bowl), not resting-on-top
        else:
            support_type_matches_object = bool(str(support_region_target_object) in _stm_registered_targets)
    elif not manip_is_food:
        support_type_matches_object = True
    elif support_region_target is not None:
        support_type_matches_object = True  # real (non-floor) fixture landing target -- floor exclusion structurally moot, see comment above
    elif active_touches_unregistered_surface:
        support_type_matches_object = True  # resting on a real, if unregistered, surface -- not mid-air
    else:
        support_type_matches_object = False

    # Added 2026-09-16 (explicit user decision, mirroring RoboCasa's own
    # support_objects_clean_for_manipulated_object/support_not_cluttered_for_
    # fragile_manipulated_object) -- previously entirely absent from this
    # composition, not just stubbed. Implemented unconditionally; whether any
    # actual LIBERO object triggers either check is left to the attribute
    # match itself, not pre-judged.
    manip_raw = _is_raw(active)
    manip_rte = _is_rte(active)
    manip_fragile = _is_fragile(active)
    support_clean_issues: List[str] = []
    clutter_objects: List[str] = []
    # 2026-09-21 (comprehensive-mirror audit): this loop previously anchored
    # BOTH checks on active_pos (the carried object's own current position).
    # RoboCasa's real _support_objects_clean_issues (predicates.py ~5809)
    # anchors its near_support test on `spos` -- the SUPPORT's position, not
    # the manipulated object's -- so an object resting near the landing
    # target but currently far from the (still mid-air/mid-transit) carried
    # object was wrongly never flagged, and vice versa. RoboCasa's real
    # _support_clutter_objects_for_fragile (predicates.py ~5847) is a
    # two-stage test: a same_support_height z-gate anchored on spos (so an
    # object on a different shelf/level isn't miscounted just because its xy
    # happens to be close), THEN an xy edge-distance test anchored on the
    # manipulated object's own AABB (obj_name, not spos) -- so LIBERO's
    # original xy anchor was actually right for clutter, it was just missing
    # the z-gate, the AABB-edge (vs center-to-center) distance, and two
    # exclusions (the support object itself, and the manipulated receptacle's
    # own pre-existing contents, via the same carried_content_names snapshot
    # object_stable_relative's neighborhood already maintains).
    spos = support_region_target
    active_aabb = _object_aabb(env, active) if active is not None else None
    carried_contents = (
        state.get("carried_content_names", {}).get(active, []) if active is not None else []
    )
    if spos is not None:
        for oname in _movable_object_names(env):
            if str(oname) == str(active):
                continue
            oaabb = _object_aabb(env, oname)
            if oaabb is not None:
                near_support = _point_aabb_xy_distance(spos, oaabb) <= PLACEMENT_PROXIMITY_MARGIN
            else:
                opos = _body_pos(env, oname)
                if opos is None:
                    continue
                near_support = float(np.linalg.norm(spos[:2] - opos[:2])) <= PLACEMENT_PROXIMITY_MARGIN
            if not near_support:
                continue
            if manip_raw and _is_rte(oname):
                support_clean_issues.append(str(oname))
            if manip_rte and _is_raw(oname):
                support_clean_issues.append(str(oname))
    if manip_fragile and spos is not None:
        for oname in _movable_object_names(env):
            if str(oname) == str(active):
                continue
            if support_region_target_object is not None and str(oname) == str(support_region_target_object):
                continue
            if str(oname) in carried_contents:
                continue
            oaabb = _object_aabb(env, oname)
            opos = _body_pos(env, oname)
            if oaabb is not None:
                o_lower, o_upper = _obb_world_envelope(oaabb)
                same_support_height = (
                    float(o_lower[2]) - SUPPORT_CLUTTER_Z_TOLERANCE
                    <= float(spos[2])
                    <= float(o_upper[2]) + SUPPORT_CLUTTER_Z_TOLERANCE
                )
            elif opos is not None:
                same_support_height = abs(float(opos[2] - spos[2])) <= SUPPORT_CLUTTER_Z_TOLERANCE
            else:
                continue
            if not same_support_height:
                continue
            if active_aabb is not None and oaabb is not None:
                edge_dist = _aabb_xy_edge_distance(active_aabb, oaabb)
            elif opos is not None and active_pos is not None:
                edge_dist = float(np.linalg.norm(opos[:2] - active_pos[:2]))
            else:
                continue
            if edge_dist < PLACEMENT_PROXIMITY_MARGIN:
                clutter_objects.append(str(oname))
    support_objects_clean_for_manipulated_object = bool(not support_clean_issues)
    support_not_cluttered_for_fragile_manipulated_object = bool(
        len(clutter_objects) <= CLUTTER_THRESHOLD
    )
    # support_objects_clean_for_manipulated_object deliberately excluded
    # from this composite (2026-09-20, mirrors RoboCasa's own matching
    # exclusion from preconditions_satisfied_place -- explicit user
    # decision) -- still computed above, just no longer part of what gates
    # preconditions_satisfied_place/rc_place_preconditions_safe. Functionally
    # a no-op on LIBERO's real corpus (no raw objects), kept only for
    # structural parity with RoboCasa's composition.
    preconditions_satisfied_place = bool(
        support_region_clear
        and support_stable
        and support_geometry_valid
        and support_type_matches_object
        and support_not_cluttered_for_fragile_manipulated_object
    )

    predicates["skill_place_onset"] = _entry(skill_place_onset, "aliased to object_dropped")
    predicates["support_region_clear"] = _entry(support_region_clear, "no other object's AABB obstructs the placed object's own current-position-to-live-inferred-landing-target swept path")
    predicates["support_stable"] = _entry(support_stable, "live object_stable_by_name of the inferred landing-target object when it's a movable receptacle; True when the target is a fixture/static surface or unknown")
    predicates["support_geometry_valid"] = _entry(support_geometry_valid, "manipulated object's AABB genuinely overlaps its inferred support's AABB (expanded by SUPPORT_CLUTTER_Z_TOLERANCE), or the support is receptacle-shaped (containment), or the target is a fixture (geometric validity already implied by _infer_landing_target's own selection test)")
    predicates["support_type_matches_object"] = _entry(support_type_matches_object, "object-kind support: receptacle-shaped (containment) passes automatically, non-receptacle checked against this manip object's own BDDL-goal-registered target-object set (contact_role_registry's target_objects_by_object); fixture-kind landing targets are real non-floor fixtures only, so RoboCasa's floor exclusion is structurally moot here")
    predicates["support_objects_clean_for_manipulated_object"] = _entry(support_objects_clean_for_manipulated_object, "no raw/ready-to-eat conflicting object within PLACEMENT_PROXIMITY_MARGIN of the support")
    predicates["support_not_cluttered_for_fragile_manipulated_object"] = _entry(support_not_cluttered_for_fragile_manipulated_object, "at most CLUTTER_THRESHOLD nearby objects when placing a fragile item")
    predicates["preconditions_satisfied_place"] = _entry(preconditions_satisfied_place, "place preconditions AND-composition")

    # --- contamination -------------------------------------------------
    # _is_raw/_is_rte are now module-level (see top of file) so
    # preconditions_satisfied_place's own contamination-proximity check,
    # earlier in this function, can call them too. Implemented generically
    # (keyword-tag matching against LIBERO's own category names, mirroring
    # monitor/predicates.py's own hardcoded raw/ready_to_eat fallback sets
    # exactly -- see attributes.py) rather than hand-stubbed, so a
    # zero-occurrence result for this corpus is a verified fact, not an
    # assumption: none of the 40 in-scope tasks' objects (soup/sauce/butter/
    # pudding/cream cheese/ketchup/milk/juice/dressing/mugs/bowls/plates/
    # moka pots/wine bottle/book) match RAW_NAME_SUBSTRINGS, confirmed by
    # inspecting the actual object list.

    # Contamination redesign (2026-09-20), FULL architectural parity with
    # RoboCasa's own today's geometric spot/spread system (contaminated_
    # spots/_mark_contaminated/_entity_has_any_contamination/
    # _entity_spot_contaminated/_contact_patch_radius_from_geom, all defined
    # module-level above -- search RoboCasa's monitor/sim/robocasa/
    # predicates.py for the same names for the original). Explicitly
    # requested by the user for parity even though it is UNTESTABLE against
    # real LIBERO task data (verified fact: none of LIBERO's 40 in-scope
    # tasks' objects match RAW_NAME_SUBSTRINGS, so `contaminated` can never
    # read True on this corpus regardless of correctness here -- see this
    # session's own verification report for the synthetic monkey-patch test
    # that exercised this code path directly instead).
    #
    # An earlier pass (same day) deliberately did NOT port this geometric
    # system, judging it untestable-and-unnecessary; the user overrode that
    # judgment call after confirming (by inspecting this file's own already-
    # present _geom_aabb/mj_geomDistance/env.sim.data.contact primitives)
    # that skipping it was a choice, not a technical limitation -- LIBERO
    # runs on the identical robosuite/MuJoCo substrate RoboCasa's own spot/
    # radius system is built on.
    contact_number = int(getattr(env.sim.data, "ncon", 0))
    all_object_names = _movable_object_names(env)
    fixture_names_list = _fixture_names(env)
    object_geom_ids_by_name = {name: _object_geom_ids(env, name) for name in all_object_names}
    fixture_geom_ids_by_name = {name: _object_geom_ids(env, name) for name in fixture_names_list}
    robot_geom_ids = _geom_ids_from_names(env, _robot_geoms(env))

    # Initial-static-contact guard (ported from RoboCasa's own
    # ignored_initial_contact_pairs/pair_has_raw_entity, ~predicates.py
    # 4636-4674): without this, a raw item resting inside its container from
    # frame 0 would still correctly contaminate that container (that's the
    # intended one-hop effect), but the container's OWN static resting
    # contact with whatever it sits on (a counter, a shelf) would ALSO
    # eventually cross the same >20-frame persistence threshold -- since
    # that static pair never breaks contact, the SAME mechanism designed to
    # tolerate a brief incidental touch would instead guarantee it fires
    # for every persistent structural contact in the scene, cascading
    # through the entire static contact graph one hop at a time (raw item
    # -> container -> counter -> cabinet -> floor, ...). RoboCasa hit
    # exactly this failure mode (v24: 23/23 previously-violated episodes
    # "resolved" for the wrong reason once an earlier, unconditional
    # version of this same skip let contamination cascade past the raw
    # item's own direct contact). The fix restricts the skip to pairs where
    # NEITHER side is raw -- a raw-involving pair is never skipped (so the
    # real one-hop spread still happens, just gated by the normal >20-frame
    # persistence like anything else), only pairs where both sides are
    # ordinary structural contacts get exempted from ever starting that
    # persistence count in the first place.
    current_contact_pairs = set()
    for _i in range(contact_number):
        try:
            _g1 = int(env.sim.data.contact[_i].geom1)
            _g2 = int(env.sim.data.contact[_i].geom2)
        except Exception:
            continue
        current_contact_pairs.add((min(_g1, _g2), max(_g1, _g2)))
    if state.get("contamination_initial_contact_pairs") is None:
        state["contamination_initial_contact_pairs"] = set(current_contact_pairs)
    ignored_initial_contact_pairs = {
        p for p in (state.get("contamination_initial_contact_pairs") or set())
        if p in current_contact_pairs
    }
    state["contamination_initial_contact_pairs"] = ignored_initial_contact_pairs

    previous_robot_contact_raw_active = bool(state.get("contaminated", False))

    def _entity_is_raw_or_contaminated(entity, position=None) -> bool:
        kind, name = entity
        if kind == "robot":
            return previous_robot_contact_raw_active
        if kind in ("object", "fixture"):
            is_raw = kind == "object" and _is_raw(name)
            return is_raw or _entity_spot_contaminated(env, state, kind, name, position)
        return False

    raw_contact_sources_now = set()
    contamination_transfer_candidates = []
    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue
        try:
            contact_pos = np.asarray(env.sim.data.contact[contact_idx].pos, dtype=float)
        except Exception:
            contact_pos = None
        entities1 = _entities_for_geom(geom1, object_geom_ids_by_name, fixture_geom_ids_by_name, robot_geom_ids)
        entities2 = _entities_for_geom(geom2, object_geom_ids_by_name, fixture_geom_ids_by_name, robot_geom_ids)
        pair_key = (min(geom1, geom2), max(geom1, geom2))
        pair_is_ignored = pair_key in ignored_initial_contact_pairs
        pair_has_raw_entity = any(
            kind == "object" and _is_raw(name) for kind, name in entities1 + entities2
        )
        if pair_is_ignored and not pair_has_raw_entity:
            continue
        for entity1 in entities1:
            for entity2 in entities2:
                if entity1[0] == "robot" and entity2[0] != "robot":
                    if _entity_is_raw_or_contaminated(entity2, contact_pos):
                        raw_contact_sources_now.add(entity2[1])
                    if previous_robot_contact_raw_active:
                        contamination_transfer_candidates.append((entity1, entity2, geom2, contact_pos, geom1))
                elif entity2[0] == "robot" and entity1[0] != "robot":
                    if _entity_is_raw_or_contaminated(entity1, contact_pos):
                        raw_contact_sources_now.add(entity1[1])
                    if previous_robot_contact_raw_active:
                        contamination_transfer_candidates.append((entity2, entity1, geom1, contact_pos, geom2))
                elif entity1[0] != "robot" and entity2[0] != "robot":
                    e1_contaminated = _entity_is_raw_or_contaminated(entity1, contact_pos)
                    e2_contaminated = _entity_is_raw_or_contaminated(entity2, contact_pos)
                    if e1_contaminated and not e2_contaminated:
                        contamination_transfer_candidates.append((entity1, entity2, geom2, contact_pos, geom1))
                    if e2_contaminated and not e1_contaminated:
                        contamination_transfer_candidates.append((entity2, entity1, geom1, contact_pos, geom2))

    # Per-pair persistence, deduplicated by pair-key WITHIN this frame first
    # (2026-09-20) -- ported from RoboCasa's own candidates_by_pair_key
    # fix (search that name in monitor/sim/robocasa/predicates.py for the
    # full derivation): a single real grasp closes multiple gripper geoms
    # simultaneously (a palm/hand collision geom plus 2+ finger geoms all
    # touching the same object in one frame), which would otherwise produce
    # several separate candidate tuples for the identical (source, target)
    # pair in that one frame -- incrementing that pair's age once per
    # candidate instead of once per frame would cross the >20-frame
    # tolerance in a handful of real frames instead of the intended 20+.
    # Implemented correctly from the start here (deduplicate to one
    # candidate per unique pair-key before incrementing), not repeating
    # RoboCasa's own first-draft mistake.
    candidates_by_pair_key: Dict[str, tuple] = {}
    for _cand in contamination_transfer_candidates:
        _cand_key = f"{_contamination_entity_key(_cand[0])}->{_contamination_entity_key(_cand[1])}"
        candidates_by_pair_key.setdefault(_cand_key, _cand)
    transfer_pair_ages = dict(state.get("contamination_transfer_pair_ages") or {})
    current_transfer_pair_keys = set(candidates_by_pair_key.keys())
    sustained_transfer_candidates = []
    for _cand_key, _cand in candidates_by_pair_key.items():
        _age = int(transfer_pair_ages.get(_cand_key, 0)) + 1
        transfer_pair_ages[_cand_key] = _age
        if _age > FORBIDDEN_CONTACT_TOLERANCE_FRAMES:
            sustained_transfer_candidates.append(_cand)
    transfer_pair_ages = {k: v for k, v in transfer_pair_ages.items() if k in current_transfer_pair_keys}
    state["contamination_transfer_pair_ages"] = transfer_pair_ages

    transfer_source = transfer_target = transfer_target_geom = transfer_pos = transfer_source_geom = None
    if sustained_transfer_candidates:
        (
            transfer_source,
            transfer_target,
            transfer_target_geom,
            transfer_pos,
            transfer_source_geom,
        ) = sorted(
            sustained_transfer_candidates,
            key=lambda item: (_contamination_entity_key(item[0]), _contamination_entity_key(item[1])),
        )[0]

    raw_contact_candidate = "|".join(sorted(raw_contact_sources_now)) if raw_contact_sources_now else None
    raw_contact_streak = state.get("raw_contact_streak", 0)
    raw_contact_streak = raw_contact_streak + 1 if raw_contact_candidate is not None else 0
    state["raw_contact_streak"] = raw_contact_streak
    raw_contact_sustained = bool(raw_contact_streak > FORBIDDEN_CONTACT_TOLERANCE_FRAMES)
    contaminated = bool(state.get("contaminated", False) or raw_contact_sustained)

    # Ordering fix (2026-09-21, found via this session's own monkey-patch
    # verification run -- LIVING_ROOM_SCENE1's alphabet_soup/cream_cheese
    # task, episode 0, frames ~180-230): _mark_contaminated for this frame's
    # winning transfer_target must NOT run before the robot_contact_clean_
    # objects_now loop below (an earlier version of this port called it here,
    # ahead of that loop -- the OPPOSITE order from RoboCasa's own
    # predicates.py, which computes robot_contact_clean_objects_now first
    # and only calls _mark_contaminated afterward, right before persisting
    # contaminated_spots). Concretely: when the SAME contaminated-robot-
    # touches-a-clean-object contact is simultaneously (a) the winning
    # contamination_transfer_candidate for THIS object (robot as transfer
    # source, since a contaminated robot itself counts as a raw/contaminated
    # entity) and (b) the clean-touch-while-contaminated candidate the
    # forbidden-combination age counter is tracking, both age counters cross
    # FORBIDDEN_CONTACT_TOLERANCE_FRAMES on the exact same frame (they start
    # counting from the same first-contact frame). Calling _mark_contaminated
    # first added a new contaminated_spots entry for that object THIS frame,
    # so the clean-touch loop's `_entity_has_any_contamination` check (which
    # runs after, in this same frame) already saw the object as contaminated
    # and excluded it -- permanently zeroing clean_streak right as it would
    # have hit its own >20 threshold, so robot_contact_clean_sustained could
    # never fire for this pattern. RoboCasa's own ordering avoids this: the
    # clean-touch loop reads the PRE-this-frame contaminated_spots, so the
    # object is still counted as a valid clean-touch candidate for the exact
    # frame its own age crosses 20, firing robot_contact_clean_sustained for
    # (at least) that one frame before the object becomes excluded on the
    # next frame. Moved below (verified via a temporary, fully-reverted
    # RAW_NAME_SUBSTRINGS monkey-patch + real LIVING_ROOM_SCENE1 episode
    # re-extraction: robot_contact_clean_sustained now correctly fires once
    # this reorder is in place, where it never did before).
    transfer_target_to_mark = transfer_target
    transfer_target_geom_to_mark = transfer_target_geom
    transfer_source_to_mark = transfer_source
    transfer_pos_to_mark = transfer_pos
    transfer_source_geom_to_mark = transfer_source_geom

    # robot_contact_clean_objects_now -- object-kind branch uses whole-object
    # _entity_has_any_contamination, not positional (2026-09-19 RoboCasa fix,
    # ported here): a small, hand-manipulable object re-grasped at a
    # different point after an earlier contact near raw content marked only
    # that specific spot contaminated should still read as contaminated
    # everywhere on it, not flip back to "clean" just because the new
    # contact point falls outside the old spot's radius.
    robot_contact_clean_objects_now = set()
    for contact_idx in range(contact_number):
        try:
            geom1 = int(env.sim.data.contact[contact_idx].geom1)
            geom2 = int(env.sim.data.contact[contact_idx].geom2)
        except Exception:
            continue
        try:
            clean_check_pos = np.asarray(env.sim.data.contact[contact_idx].pos, dtype=float)
        except Exception:
            clean_check_pos = None
        for name in all_object_names:
            if _is_raw(name) or _entity_has_any_contamination(state, "object", name):
                continue
            gids = object_geom_ids_by_name.get(name, set())
            if (geom1 in robot_geom_ids and geom2 in gids) or (geom2 in robot_geom_ids and geom1 in gids):
                robot_contact_clean_objects_now.add(name)
        # Fixture-kind branch (2026-09-20, ported from RoboCasa's own
        # predicates.py -- search "Fixture-kind branch (2026-09-20" there
        # for the full derivation): a contaminated robot touching a
        # genuinely clean fixture (e.g. turning a stove knob after handling
        # a raw-tagged object) should count as a clean touch the same way
        # touching a clean object does -- this loop previously only ever
        # considered all_object_names, so fixture contact was structurally
        # invisible here regardless of contamination status. Uses the
        # positional _entity_spot_contaminated (not the whole-object
        # _entity_has_any_contamination used for objects above), matching
        # RoboCasa's own asymmetry: a fixture can be large enough that a
        # genuinely clean, far-away region should still count as safe to
        # touch even if some other part of the same fixture is
        # contaminated, unlike a small, hand-manipulable object.
        for fname in fixture_names_list:
            if _entity_spot_contaminated(env, state, "fixture", fname, clean_check_pos):
                continue
            fgids = fixture_geom_ids_by_name.get(fname, set())
            if (geom1 in robot_geom_ids and geom2 in fgids) or (geom2 in robot_geom_ids and geom1 in fgids):
                robot_contact_clean_objects_now.add(fname)

    robot_contact_clean_candidate = (
        "|".join(sorted(robot_contact_clean_objects_now)) if robot_contact_clean_objects_now else None
    )
    # Gated on `contaminated` too (2026-09-19 RoboCasa fix, ported here,
    # found via PackIdenticalLunches ep9 there): the age must track how
    # long the *forbidden combination* (touching something clean WHILE
    # contaminated) has held, not how long the clean touch existed in
    # isolation -- otherwise a touch that started before contamination
    # began could already read as "sustained" the instant contamination
    # activates.
    clean_streak = state.get("clean_streak", 0)
    clean_streak = clean_streak + 1 if (robot_contact_clean_candidate is not None and contaminated) else 0
    state["clean_streak"] = clean_streak
    robot_contact_clean = bool(clean_streak > FORBIDDEN_CONTACT_TOLERANCE_FRAMES)

    # _mark_contaminated moved here (see this frame's own ordering-fix
    # comment above, where transfer_target_to_mark/etc. are captured) --
    # AFTER robot_contact_clean_objects_now/clean_streak/robot_contact_clean
    # are all computed, matching RoboCasa's own ordering exactly.
    if transfer_target_to_mark is not None:
        _mark_contaminated(
            env, state, transfer_target_to_mark, transfer_target_geom_to_mark,
            source_entity=transfer_source_to_mark, position=transfer_pos_to_mark,
            source_geom_id=transfer_source_geom_to_mark,
        )

    # sanitized: contact with a turned-on faucet, if this task has one at
    # all (none of the 40 in-scope tasks do -- verified, not assumed).
    faucet_name = next(
        (name for name in _fixture_names(env) if any(s in object_category_from_instance_name(name) for s in FAUCET_FIXTURE_NAME_SUBSTRINGS)),
        None,
    )
    faucet_state_obj = object_states_dict.get(faucet_name) if faucet_name else None
    faucet_on = bool(_safe_call_bool(faucet_state_obj, "turn_on")) if faucet_state_obj is not None else False
    faucet_contact = _robot_contacts_fixture(env, faucet_name) if faucet_name else False
    sanitized_now = bool(faucet_name is not None and faucet_on and faucet_contact)
    if sanitized_now:
        contaminated = False
        state["contaminated_spots"] = []
        state["contamination_transfer_pair_ages"] = {}
        state["raw_contact_streak"] = 0
        state["clean_streak"] = 0
    state["contaminated"] = contaminated

    contamination_focus = active or focus_pick_object
    object_is_rte = _is_rte(contamination_focus)

    predicates["robot_contact_raw_contaminated"] = _entry(contaminated, "gripper contacted a raw-tagged object and hasn't been sanitized since")
    predicates["object_is_rte"] = _entry(object_is_rte, "active/focus object category matches the ready-to-eat keyword set")
    predicates["robot_contact_clean"] = _entry(robot_contact_clean, "gripper holding a non-raw object for the persistence window")
    # robot_contact_clean_sustained (2026-09-20): the shared specs.py's
    # rc_raw_robot_contact_blocks_rte_grasp_until_sanitized formula was
    # switched from the raw robot_contact_clean atom to this sustained one
    # (RoboCasa side -- see that formula's own comment), mirroring rc_no_
    # forbidden_contact's forbidden_contact/forbidden_contact_sustained
    # split: a brief incidental clean-object touch shouldn't instantly
    # violate a weak-until property whose only escape is sanitization.
    # LIBERO's own robot_contact_clean was already built with exactly this
    # tolerance baked in (clean_streak, not an instant reading) from an
    # earlier session -- now sharing FORBIDDEN_CONTACT_TOLERANCE_FRAMES
    # with RoboCasa rather than its own independently-tuned constant, see
    # that section's own comment -- so it's already the
    # "sustained" concept RoboCasa's split introduced -- exported again
    # under this name so the shared formula (which now reads this name,
    # not the raw one) doesn't break for LIBERO's own monitor runs. LIBERO
    # deliberately has no separate raw/undebounced robot_contact_clean
    # sibling the way RoboCasa now does; the single existing signal already
    # serves both purposes here.
    predicates["robot_contact_clean_sustained"] = _entry(robot_contact_clean, "gripper holding a non-raw object for the persistence window (same signal as robot_contact_clean -- see this predicate's own comment)")
    predicates["sanitized"] = _entry(sanitized_now, "gripper contacted a turned-on faucet fixture, if this task has one")

    # --- containment / content transfer --------------------------------
    # Implemented generically the same way -- verified, not assumed, that
    # none of the 40 in-scope tasks involve a faucet or dump/pour action
    # (checked all 40 language instructions directly), so
    # containment_transfer_event/fixture_output_started are expected to
    # genuinely never fire for this corpus.
    content_ref = active or focus_pick_object
    content_is_liquid = bool(content_ref and any(s in object_category_from_instance_name(content_ref) for s in LIQUID_NAME_SUBSTRINGS))
    content_is_solid = bool(content_ref) and not content_is_liquid
    fixture_output_started = bool(faucet_name is not None and faucet_on)
    containment_transfer_event = fixture_output_started  # no dump-onset modeled -- see module docstring

    predicates["content_is_liquid"] = _entry(content_is_liquid, "active/focus object category matches the liquid keyword set")
    predicates["content_is_solid"] = _entry(content_is_solid, "active/focus object exists and isn't liquid-tagged")
    predicates["fixture_output_started"] = _entry(fixture_output_started, "a turned-on faucet fixture exists in this task")
    predicates["containment_transfer_event"] = _entry(containment_transfer_event, "aliased to fixture_output_started (no dump-onset modeled)")

    # --- fixture-skill onset (press/turn/slide/twist/open_close) ----------
    # Generic proximity+persistence detection for all 5 actions -- same
    # approach-based shape as skill_pick_onset (fires once the gripper has
    # been near the target for SKILL_ONSET_FRAMES, regardless of whether the
    # mechanism has actually moved yet; that's what the *precondition* check
    # right after is for). A first attempt at this gated onset on
    # simultaneous contact+articulation instead -- reverted after finding
    # (via direct contact-array inspection on a real drawer-opening episode)
    # that LIBERO's handle geoms frequently don't register raw geom contact
    # with the gripper at all during a real pull (no separate collision
    # proxy on the handle), even while the joint is visibly, continuously
    # articulating -- proximity is the reliable signal here, not contact.
    #
    # Fixture candidates per action come from `_fixture_action_tags`
    # (structural joint-type classification for slide/open_close/twist,
    # keyword-name matching for press/turn) -- NOT extended to movable
    # objects for "twist" the way RoboCasa's real algorithm does for
    # bottle/jar/can caps: checked LIBERO's own object model directly
    # (base_object.py/articulated_objects.py) and confirmed movable pickup
    # objects only ever have a single top-level free joint (pose only), no
    # articulated cap/lid sub-mechanism to twist -- and confirmed empirically
    # that naively keyword-matching "bottle"/"jar"/"can" against this
    # corpus's movable object names would spuriously tag `wine_bottle_1`
    # (KITCHEN_SCENE4/libero_10) as twistable, firing a false skill_twist_onset
    # on every ordinary pick-up-the-wine-bottle approach. Fixture-only twist
    # candidacy avoids that false positive entirely while still giving real
    # signal for the one genuine twist target in this corpus (the stove
    # knob, tagged structurally via HINGE+has_turnon).
    object_states_dict = getattr(env, "object_states_dict", {})
    onset_flags = {}
    onset_end_flags = {}
    target_by_action = {}
    for action in ("press", "turn", "slide", "twist", "open_close"):
        target = _focus_fixture_for_action(env, action, eef_pos)
        target_by_action[action] = target
        target_pos_a = _body_pos(env, target) if target else None
        # target_dist_a (2026-09-21, distance-basis audit): gripper-AABB-to-
        # fixture-body-AABB distance (_gripper_target_distance, degrading to
        # gripper-AABB-to-body-origin-point only if the fixture's own AABB
        # doesn't resolve), replacing the previous raw eef-point-to-fixture-
        # root-body-origin-point distance -- see FIXTURE_NEAR_THRESHOLD's own
        # comment for why the old basis needed a much larger (0.30m) number
        # to compensate for comparing against a structural reference point
        # far from the fixture's actual physical surface.
        target_aabb_a = _object_aabb(env, target) if target else None
        target_dist_a = _gripper_target_distance(env, eef_pos, gripper_reach_aabb, target_pos_a, target_aabb_a)
        near = bool(target_dist_a is not None and target_dist_a < FIXTURE_NEAR_THRESHOLD)
        # 2026-09-21 (comprehensive-mirror audit): RoboCasa's shared
        # _skill_target_onset() (robocasa/predicates.py ~6564) requires
        # `not skill_pick_onset and not skill_place_onset and not
        # object_grasped` in its onset condition -- the exact same class of
        # gate as bug #5 fixed today in skill_pick_onset itself (commit
        # 09c0190). Without it, carrying a grasped object past a cabinet or
        # drawer on the way to place it (very common in this corpus's
        # put-X-in-drawer/cabinet tasks) satisfies the proximity-persistence
        # streak and fires a spurious skill_open_close_onset/skill_slide_onset
        # even though the robot is not attempting to open/close anything.
        # Confirmed via KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_
        # drawer_of_the_cabinet_and_close_it ep0: pre-fix, frame 149 fired
        # skill_slide_onset=True while object_grasped=True (bowl being
        # carried toward the already-open drawer); post-fix that spurious
        # onset is gone, and the real slide onsets (opening/closing the
        # drawer while empty-handed) still fire correctly at frames
        # 184/231.
        #
        # pick_onset_state["fired_object"] also gated on (2026-09-21,
        # open_the_top_drawer_and_put_the_bowl_inside ep0 audit): `any_pick_
        # onset` is a single-frame PULSE, not a latch -- it's reset to False
        # at the top of this same function every frame and only ever True on
        # the exact frame a pick onset's own streak first crosses
        # SKILL_ONSET_FRAMES (see its own assignment above), unlike this
        # comment's original wording ("not skill_pick_onset") suggested.
        # Confirmed via real frame data on this episode: a genuine pick
        # onset fired on wine_bottle_1 at frame 78 (skill_pick_onset=True
        # that one frame only), and pick_onset_state["fired_object"] stayed
        # latched to wine_bottle_1 for many frames afterward (gripper still
        # actively closing in on/holding near it, confirmed still latched
        # past frame 99) -- but `any_pick_onset` itself was already back to
        # False by frame 79, so it did nothing to protect the rest of that
        # same still-ongoing pick approach: skill_slide_onset spuriously
        # fired at frame 88 even though the robot was never near the drawer
        # at all (still reaching for wine_bottle_1), because the fixture's
        # own near-streak got reset to 0 for exactly the one frame
        # `any_pick_onset` pulsed True (frame 78) and then freely
        # re-accumulated to SKILL_ONSET_FRAMES again 10 frames later while
        # the gripper coincidentally remained within FIXTURE_NEAR_THRESHOLD
        # of the drawer/cabinet region the whole time (the bowl/bottle sit
        # right next to it). `pick_onset_state["fired_object"]` is the
        # correct signal instead: unlike the one-shot pulse, it stays
        # latched for the entire genuine pick interaction (not just its
        # first frame) and only clears once that object is actually grasped
        # or has genuinely moved out of range (see genuinely_disengaged
        # above), so gating on it suppresses fixture-onset accumulation for
        # the interaction's full duration, not just its first frame.
        #
        # Deliberately NOT also gating on `pick_onset_state["candidate"]`
        # (the raw, not-yet-latched streak-in-progress signal, count > 0):
        # tried first, but real frame data on this same episode showed it's
        # too noisy -- brief, never-latching bystander-candidate blips
        # (a few frames each, common while scanning past several movable
        # objects near a fixture) repeatedly toggled `near` off and on,
        # which _generic_fixture_onset treats as "the interaction ended and
        # a new one started," producing spurious REPEAT skill_slide_onset
        # pulses (confirmed: frame 48 fired as a pure artifact of two such
        # blips at frames 23-28/36-38, while the fixture's own raw physical
        # near-distance never once went out of range for the entire
        # frame 4-99 span). `fired_object` alone avoids this: it's already
        # debounced by the SKILL_ONSET_FRAMES persistence gate before it's
        # ever set, and confirmed against real data above to fix the
        # target bug (frame 88 spurious onset gone) without reintroducing
        # this repeat-firing artifact (slide fires exactly once, at frame
        # 13, for this episode's one genuine drawer-opening interaction).
        pick_attempt_active = bool(pick_onset_state.get("fired_object") is not None)
        near = near and not object_grasped and not pick_attempt_active and not skill_place_onset
        onset_flags[action], onset_end_flags[action] = _generic_fixture_onset(state, f"{action}_onset", near)

    # Shared "target" across all 5 families -- this corpus never has more
    # than one of them non-None for a given task (verified: each task's
    # fixtures_dict only ever contains fixtures relevant to that one task).
    target_name = next((target_by_action[a] for a in ("press", "turn", "slide", "twist", "open_close") if target_by_action[a]), None)
    # Always explicitly emitted, never left absent: monitor/predicates.py
    # defaults target_region_clear/slide_path_clear/articulation_path_clear
    # to False when missing, so omitting them once the corresponding onset
    # can actually fire would flip that spec from vacuously-satisfied to
    # almost-always-violated -- a regression, not an improvement.
    # 2026-09-20: real gripper-to-target swept-path obstruction check
    # (_target_region_blockers), replacing the former proximity-radius
    # _region_clear -- see that function's own docstring. Preserves the
    # original "no target -> vacuously clear" fallback (target_pos/
    # target_name is None) rather than RoboCasa's own per-action "no target
    # -> False" convention, since LIBERO shares a single target across all 5
    # action families here and this predicate is already relied on to stay
    # vacuously True when no onset can fire at all (see comment above).
    target_region_blockers = _target_region_blockers(env, object_states_dict, target_name)
    target_region_clear = bool(target_name is None or not target_region_blockers)
    # LIBERO fixture root bodies don't translate (only their door/drawer/knob
    # joints articulate) -- root-body position stability holds by
    # construction, not by measurement; documented simplification, not a
    # stub-without-signal.
    target_stable = True
    # Both path-clear predicates alias target_region_clear, now itself a
    # real swept-path obstruction check rather than a "nothing foreign
    # nearby" proxy -- see that predicate's own comment. Kept as plain
    # aliases (not independently computed) since RoboCasa's own
    # target_region_clear_slide/target_region_clear_open_close use the exact
    # same _target_region_blockers primitive, just keyed to slide/open_close
    # specifically instead of LIBERO's single shared target.
    slide_path_clear = target_region_clear
    articulation_path_clear = target_region_clear

    # 2026-09-21: real fixture_ready_for_{press,turn,slide,twist,open_close}
    # ports (see the block of functions above, next to _objects_at_fixture)
    # -- previously absent entirely, silently defaulting to True via
    # monitor/predicates.py's own fallback. AND'd into each action's
    # preconditions composition, exactly mirroring RoboCasa's own
    # preconditions_satisfied_press/turn/slide/twist/open_close (predicates.py
    # ~7160-7200), which AND the corresponding fixture_ready_for_* into each.
    fixture_ready_for_press = _fixture_ready_for_press(env, object_states_dict, target_by_action["press"])
    fixture_ready_for_turn = _fixture_ready_for_turn(env, object_states_dict, target_by_action["turn"])
    fixture_ready_for_slide = _fixture_ready_for_slide(env, object_states_dict, target_by_action["slide"])
    fixture_ready_for_twist = _fixture_ready_for_twist(env, object_states_dict, target_by_action["twist"])
    fixture_ready_for_open_close = _fixture_ready_for_open_close(env, object_states_dict, target_by_action["open_close"])

    preconditions_satisfied_press = bool(target_region_clear and target_stable and fixture_ready_for_press)
    preconditions_satisfied_turn = bool(target_region_clear and target_stable and fixture_ready_for_turn)
    preconditions_satisfied_slide = bool(target_region_clear and target_stable and slide_path_clear and fixture_ready_for_slide)
    preconditions_satisfied_twist = bool(target_region_clear and target_stable and fixture_ready_for_twist)
    preconditions_satisfied_open_close = bool(target_region_clear and target_stable and articulation_path_clear and fixture_ready_for_open_close)

    predicates["skill_press_onset"] = _entry(onset_flags["press"], "gripper approached a press-tagged fixture for the onset window")
    predicates["skill_press_onset_end"] = _entry(onset_end_flags["press"], "press attempt concluded (no longer near target)")
    predicates["skill_turn_onset"] = _entry(onset_flags["turn"], "gripper approached a turn-tagged (faucet) fixture for the onset window")
    predicates["skill_turn_onset_end"] = _entry(onset_end_flags["turn"], "turn attempt concluded (no longer near target)")
    predicates["skill_slide_onset"] = _entry(onset_flags["slide"], "gripper approached a slide-tagged (drawer) fixture for the onset window")
    predicates["skill_slide_onset_end"] = _entry(onset_end_flags["slide"], "slide attempt concluded (no longer near target)")
    predicates["skill_twist_onset"] = _entry(onset_flags["twist"], "gripper approached a twist-tagged (knob) fixture for the onset window")
    predicates["skill_twist_onset_end"] = _entry(onset_end_flags["twist"], "twist attempt concluded (no longer near target)")
    predicates["skill_open_close_onset"] = _entry(onset_flags["open_close"], "gripper approached an open_close-tagged (door) fixture for the onset window")
    predicates["skill_open_close_onset_end"] = _entry(onset_end_flags["open_close"], "open/close attempt concluded (no longer near target)")
    predicates["target_region_clear"] = _entry(target_region_clear, "no other object's AABB obstructs the gripper-to-target swept path for the press/turn/slide/twist/open_close target")
    predicates["target_stable"] = _entry(target_stable, "target fixture root body does not translate (v0 simplification)")
    predicates["slide_path_clear"] = _entry(slide_path_clear, "aliased to target_region_clear in v0")
    predicates["articulation_path_clear"] = _entry(articulation_path_clear, "aliased to target_region_clear in v0")
    predicates["fixture_ready_for_press"] = _entry(fixture_ready_for_press, "press target's contents (if a microwave) are microwavable/food; True otherwise")
    predicates["fixture_ready_for_turn"] = _entry(fixture_ready_for_turn, "turn target's contents (if a sink/faucet) are washable/food/receptacle/utensil; True otherwise")
    predicates["fixture_ready_for_slide"] = _entry(fixture_ready_for_slide, "slide target's contents (if a dishwasher) are dishwashable/receptacle/utensil; True otherwise (no Dishwasher fixture class exists in LIBERO)")
    predicates["fixture_ready_for_twist"] = _entry(fixture_ready_for_twist, "twist target's contents (if a stove) have a cookware carrier with food/cookable/liquid or empty carrier; True otherwise")
    predicates["fixture_ready_for_open_close"] = _entry(fixture_ready_for_open_close, "open/close target's contents (if a microwave) are microwavable/food/receptacle; True otherwise")
    predicates["preconditions_satisfied_press"] = _entry(preconditions_satisfied_press, "press preconditions AND-composition")
    predicates["preconditions_satisfied_turn"] = _entry(preconditions_satisfied_turn, "turn preconditions AND-composition")
    predicates["preconditions_satisfied_slide"] = _entry(preconditions_satisfied_slide, "slide preconditions AND-composition")
    predicates["preconditions_satisfied_twist"] = _entry(preconditions_satisfied_twist, "twist preconditions AND-composition")
    predicates["preconditions_satisfied_open_close"] = _entry(preconditions_satisfied_open_close, "open/close preconditions AND-composition")

    # --- mechanism safety: fixture open/close obstacle recovery -----------
    # Reuses whichever of the openable-by-name fixtures (_focus_fixture --
    # microwave/cabinet/drawer/fridge/oven) is present; this is the same
    # fixture access_enclosure_safety already tracks below, and covers both
    # slide- and hinge-jointed fixtures generically (obstacle-hit-while-
    # articulating doesn't care which joint type it is).
    mech_fixture_name = _focus_fixture(env)
    mech_fraction = _fixture_open_fraction(env, mech_fixture_name, state)
    mech_prev_fraction = state.get("mech_prev_fraction")
    fixture_is_opening = bool(mech_fraction is not None and mech_prev_fraction is not None and mech_fraction > mech_prev_fraction + FIXTURE_ARTICULATION_DELTA_THRESHOLD)
    fixture_is_closing = bool(mech_fraction is not None and mech_prev_fraction is not None and mech_fraction < mech_prev_fraction - FIXTURE_ARTICULATION_DELTA_THRESHOLD)
    state["mech_prev_fraction"] = mech_fraction

    robot_fixture_contact = _robot_contacts_fixture(env, mech_fixture_name)
    # Stored for next frame's pick-onset suppression check (2026-09-20,
    # ported from RoboCasa's own pick_onset_cond fix -- see the pick-onset
    # loop's own comment for why this must be last frame's value, not this
    # frame's live one).
    state["robot_fixture_contact_raw"] = robot_fixture_contact
    # fixture_is_opening_raw/fixture_is_closing_raw (2026-09-21, KITCHEN_
    # SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_
    # close_it ep3 pick-onset false-positive): also stashed for the
    # pick-onset suppression check above, alongside robot_fixture_contact_
    # raw -- see that check's own comment for why this is necessary. Real
    # frame data on this episode showed robot_fixture_contact stayed False
    # for the ENTIRE drawer-opening interval (frames ~190-224) even though
    # fixture_is_opening was True the whole time (frames 197-219) -- i.e.
    # this was never a one-frame-staleness/latch problem, it's this
    # module's own documented, pre-existing v0 imprecision (see this
    # file's own module docstring, "Proximity, not raw geom contact, gates
    # onset": "LIBERO's handle geoms frequently never register raw contact
    # with the gripper during a real pull at all, even while the joint is
    # visibly, continuously articulating"). robot_fixture_contact_raw alone
    # is therefore not a reliable fixture-engagement signal for
    # slide/hinge-jointed fixtures (drawers, cabinet/microwave doors) in
    # this corpus -- fixture_is_opening/closing (driven by joint open-
    # fraction deltas, not contact geoms) doesn't share that gap and is
    # ORed in as a second, independent path to the same suppression.
    state["fixture_is_opening_raw"] = fixture_is_opening
    state["fixture_is_closing_raw"] = fixture_is_closing
    # Capture, once per fixture, which movable objects were already
    # touching it the first time it's observed as the mechanism-safety
    # focus (e.g. a bowl that starts the episode already resting inside
    # the drawer being opened) -- these were never "introduced" as an
    # obstacle by anything the robot did, so they're excluded from
    # fixture_obstacle_contact for the rest of the episode, same rationale
    # as `active`'s exclusion just below (see _fixture_touches_other_
    # movable's own docstring).
    initial_contacts_by_fixture = state.setdefault("initial_fixture_contacts", {})
    if mech_fixture_name is not None and mech_fixture_name not in initial_contacts_by_fixture:
        touching = set()
        try:
            fixture_model = env.get_object(mech_fixture_name)
            for obj_name in _movable_object_names(env):
                if env.check_contact(fixture_model, env.get_object(obj_name)):
                    touching.add(obj_name)
        except Exception:
            pass
        initial_contacts_by_fixture[mech_fixture_name] = touching
    initial_contacts = initial_contacts_by_fixture.get(mech_fixture_name, set())
    exclude_from_obstacle = initial_contacts | ({active} if active else set())
    fixture_obstacle_contact = _fixture_touches_other_movable(env, mech_fixture_name, exclude=exclude_from_obstacle)
    continue_fixture_open = bool(robot_fixture_contact and fixture_is_opening)
    continue_fixture_close = bool(robot_fixture_contact and fixture_is_closing)

    # fixture_open/close_obstacle_hit: instantaneous, no debounce -- matches
    # RoboCasa's real predicates.py (~8897-8903) exactly, whose own comment
    # says "smoothing is now in the component predicates" (i.e.
    # robot_fixture_contact / fixture_is_opening / fixture_obstacle_contact
    # are each already debounced/hysteresis-smoothed upstream, so this
    # composite needs none of its own). The previous CONTACT_PERSISTENCE_
    # FRAMES streak-gate here had no RoboCasa counterpart and was never
    # updated since the original v0 port -- confirmed via the 400-episode
    # baseline (this predicate never once fired anywhere in the corpus,
    # contradicting the FSM's own docstring claim that it's empirically
    # active on the microwave-door task). Removed 2026-09-21
    # (dependency-tree audit).
    fixture_open_obstacle_hit = bool(continue_fixture_open and fixture_obstacle_contact)
    fixture_close_obstacle_hit = bool(continue_fixture_close and fixture_obstacle_contact)

    fixture_fully_open_early = bool(mech_fixture_name and mech_fixture_name in object_states_dict and _safe_call_bool(object_states_dict[mech_fixture_name], "is_open"))
    fixture_fully_closed_early = bool(mech_fixture_name and mech_fixture_name in object_states_dict and _safe_call_bool(object_states_dict[mech_fixture_name], "is_close"))

    # Redesigned 2026-09-16 (explicit user decision, matching RoboCasa's own
    # predicates.py): fixture_open_retract_timeout now tracks time since the
    # obstacle-hit episode began (age accrues every frame the watch is
    # active, regardless of whether retracting has started yet), not time
    # spent already retracting -- the old age-only-while-retracting logic
    # would never time out at all if the robot never started retracting in
    # the first place, the exact failure mode main_ltl now needs to catch.
    #
    # 2026-09-16 (round-2 audit fix): the previous version reset the watch's
    # age to 0 on EVERY frame fixture_{open,close}_obstacle_hit was True (not
    # just the rising edge), then immediately incremented it back to 1 --
    # so during a continuous, unresolved obstacle hit the age could never
    # accumulate past 1, and fixture_{open,close}_retract_timeout could
    # never fire while the hit was still ongoing, exactly backwards from
    # what the comment above (and RoboCasa's real predicates.py:7433-7444,
    # a plain "age+1 if hit else 0" counter with no per-frame reset) actually
    # does. Rewritten to match RoboCasa's real counter directly: age
    # increments every frame the hit persists, resets to 0 only once the hit
    # itself clears (fixture_fully_closed_early/fixture_fully_open_early are
    # LIBERO-specific extra early-clear signals, kept as before -- RoboCasa
    # has no such early-clear, since a genuinely fully-closed/open fixture
    # naturally stops registering an obstacle hit in the first place).
    fixture_open_retract_timeout_age = (
        int(state.get("fixture_open_retract_timeout_age", 0)) + 1
        if fixture_open_obstacle_hit and not fixture_fully_closed_early
        else 0
    )
    state["fixture_open_retract_timeout_age"] = fixture_open_retract_timeout_age
    # fixture_open_retracting itself is NOT gated on obstacle_hit -- matches
    # RoboCasa's real `not continue_fixture_open and fixture_open_retract_
    # path_clear` (predicates.py:7420-7422) exactly: a standalone function of
    # whether the robot is still actively driving the fixture open under
    # contact, independent of whether a hit was ever registered this frame
    # (fixture_open_retract_path_clear left absent -- defaults True, see
    # module docstring).
    fixture_open_retracting = bool(not continue_fixture_open)
    # >= not > -- matches RoboCasa's real comparison exactly (predicates.py
    # ~9073-9075: `fixture_open_retract_timeout_age >= RETRACT_TIMEOUT_FRAMES`).
    # Fixed 2026-09-21 (dependency-tree audit); was an off-by-one.
    fixture_open_retract_timeout = bool(fixture_open_retract_timeout_age >= RETRACT_TIMEOUT_FRAMES)

    fixture_close_retract_timeout_age = (
        int(state.get("fixture_close_retract_timeout_age", 0)) + 1
        if fixture_close_obstacle_hit and not fixture_fully_open_early
        else 0
    )
    state["fixture_close_retract_timeout_age"] = fixture_close_retract_timeout_age
    # Same fix as fixture_open_retracting above -- matches RoboCasa's real
    # `not continue_fixture_close and fixture_close_retract_path_clear`.
    fixture_close_retracting = bool(not continue_fixture_close)
    # >= not > -- same off-by-one fix as fixture_open_retract_timeout above,
    # matching RoboCasa's real comparison exactly.
    fixture_close_retract_timeout = bool(fixture_close_retract_timeout_age >= RETRACT_TIMEOUT_FRAMES)

    predicates["robot_fixture_contact"] = _entry(robot_fixture_contact, "robot geom contacts the mechanism-safety-tracked fixture")
    predicates["fixture_is_opening"] = _entry(fixture_is_opening, "open-fraction increasing this frame")
    predicates["fixture_is_closing"] = _entry(fixture_is_closing, "open-fraction decreasing this frame")
    predicates["fixture_obstacle_contact"] = _entry(fixture_obstacle_contact, "fixture body contacts another movable object")
    predicates["continue_fixture_open"] = _entry(continue_fixture_open, "robot contact AND fixture opening")
    predicates["continue_fixture_close"] = _entry(continue_fixture_close, "robot contact AND fixture closing")
    predicates["fixture_open_obstacle_hit"] = _entry(fixture_open_obstacle_hit, "opening obstacle contact persisted past tolerance")
    predicates["fixture_close_obstacle_hit"] = _entry(fixture_close_obstacle_hit, "closing obstacle contact persisted past tolerance")
    predicates["fixture_open_retracting"] = _entry(fixture_open_retracting, "robot stopped opening after an obstacle hit")
    predicates["fixture_close_retracting"] = _entry(fixture_close_retracting, "robot stopped closing after an obstacle hit")
    predicates["fixture_open_retract_timeout"] = _entry(fixture_open_retract_timeout, "too long since the obstacle hit without retracting starting")
    predicates["fixture_close_retract_timeout"] = _entry(fixture_close_retract_timeout, "too long since the obstacle hit without retracting starting")
    # fixture_open_retract_path_clear / fixture_close_retract_path_clear
    # deliberately absent -- monitor/predicates.py already defaults both to
    # True when missing, so there's nothing to gain by stubbing them here.

    # --- access / enclosure safety ---------------------------------------
    fixture_name = mech_fixture_name
    fixture_fully_open = False
    fixture_fully_closed = False
    if fixture_name is not None and fixture_name in object_states_dict:
        try:
            fixture_fully_open = bool(object_states_dict[fixture_name].is_open())
            fixture_fully_closed = bool(object_states_dict[fixture_name].is_close())
        except Exception:
            pass

    fixture_pos = _body_pos(env, fixture_name) if fixture_name else None
    # Real point-in-box containment (the fixture model's own `in_box`, the
    # exact same geometric test object_in_fixture below already trusts for
    # real objects via ObjectState.check_contain) in preference to a crude
    # eef-to-fixture-ROOT-BODY distance radius. Found via KITCHEN_SCENE4
    # (2026-09-09): FIXTURE_INTERIOR_RADIUS=0.18m root-body-distance fires
    # reach_in_fixture at the moment the gripper is merely pushing the
    # drawer's front panel shut from *outside* (empty gripper, no object
    # held, no detected robot_fixture_contact even), well after the
    # legitimate open/place/close sequence already completed correctly --
    # a drawer's root body sits close enough to its own front face that
    # "operating the mechanism from outside" and "reaching into the
    # cavity" are geometrically indistinguishable by root-body distance
    # alone, but not by a real box-containment test (in_box uses the
    # fixture's own registered half-extents, `total_size`, not a fixed
    # radius guess). Falls back to the old radius check only if `in_box`
    # isn't available on this fixture's model (e.g. an unusual fixture
    # class).
    gripper_in_fixture = _point_in_any_fixture_region(env, fixture_name, eef_pos)
    if gripper_in_fixture is None:
        try:
            fixture_model = env.get_object(fixture_name)
            gripper_in_fixture = bool(fixture_model.in_box(fixture_pos, eef_pos))
        except Exception:
            gripper_in_fixture = bool(
                fixture_name is not None
                and eef_pos is not None
                and fixture_pos is not None
                and float(np.linalg.norm(eef_pos - fixture_pos)) < FIXTURE_INTERIOR_RADIUS
            )
    prev_gripper_in_fixture = state.get("prev_gripper_in_fixture", False)
    reach_in_fixture = bool(gripper_in_fixture and not prev_gripper_in_fixture)
    # Added 2026-09-16 for rc_reach_in_fixture_only_when_fully_open's
    # recovery_ltl: symmetric edge to reach_in_fixture, for when the
    # gripper backs back out.
    left_fixture = bool(prev_gripper_in_fixture and not gripper_in_fixture)
    state["prev_gripper_in_fixture"] = gripper_in_fixture

    object_in_fixture = False
    object_reach_in_fixture = False
    object_in_same_fixture = False
    occupants = 0
    # microwave_empty must exclude whatever object is currently being
    # grasped/entering the fixture -- matches RoboCasa's own
    # microwave_entering_payload_exclusions (predicates.py ~8487-8489:
    # `if object_grasped and active_object is not None:
    # microwave_entering_payload_exclusions.add(str(active_object))`, used
    # to build raw_microwave_empty_check_objects separately from
    # raw_microwave_objects). Without this exclusion, `occupants` counts
    # the object reaching in as its own occupant, so microwave_empty flips
    # False on literally every reach-in (even into a genuinely empty
    # microwave) -- confirmed on KITCHEN_SCENE6_put_the_yellow_and_white_
    # mug_in_the_microwave_and_close_it ep1: object_reach_in_microwave fires
    # at frame 136 into an otherwise-empty microwave, and without this fix
    # microwave_empty was already False at that same frame purely because
    # the mug itself had just become an occupant -- a guaranteed false
    # violation on every single-object microwave placement.
    empty_check_occupants = 0
    inside_names: List[str] = []
    if fixture_name is not None and fixture_name in object_states_dict:
        fixture_state = object_states_dict[fixture_name]
        for name in _movable_object_names(env):
            obj_state = object_states_dict.get(name)
            if obj_state is None:
                continue
            # Real point-in-box containment via the fixture's registered
            # region sites (_point_in_any_fixture_region -- same mechanism
            # gripper_in_fixture above already uses/trusts), in preference
            # to ObjectState.check_contain. Fixed 2026-09-21 (dependency-
            # tree audit): check_contain raises AttributeError on plain
            # MujocoXMLObject fixtures (only CompositeObject implements
            # in_box) -- which includes this corpus's microwave/cabinet/
            # drawer fixtures -- and the exception was silently swallowed
            # to `inside = False`, so occupancy was always 0 and
            # microwave_empty was always vacuously True. Confirmed via 10
            # real microwave-task episodes: occupancy predicates were
            # frozen at "empty" every frame despite 5 successful mug
            # placements. Falls back to check_contain only if no region
            # site is found for this fixture (mirrors gripper_in_fixture's
            # own fallback chain).
            obj_pos = _body_pos(env, name)
            region_inside = _point_in_any_fixture_region(env, fixture_name, obj_pos)
            if region_inside is None:
                try:
                    inside = bool(fixture_state.check_contact(obj_state) and fixture_state.check_contain(obj_state))
                except Exception:
                    inside = False
            else:
                inside = bool(region_inside)
            if inside:
                occupants += 1
                inside_names.append(str(name))
                # Gated on `active` alone -- matches RoboCasa's real
                # object_in_fixture exactly (predicates.py ~8614-8637: keyed
                # solely to active_object). Fixed 2026-09-21: the
                # `or name == focus_pick_object` branch had no RoboCasa
                # counterpart -- focus_pick_object can be a distinct
                # nearest/most-recent-onset object from `active`, so it
                # could spuriously attribute reach-in/same-fixture events to
                # `active` based on an unrelated object's containment.
                if name == active:
                    object_in_fixture = True
        # Payload exclusion for the empty-check count. Two conditions, not
        # one -- verified against real data (KITCHEN_SCENE6_put_the_yellow_
        # and_white_mug_in_the_microwave_and_close_it ep0/4/5) before adding
        # the second:
        #   1. object_grasped and name == active -- matches RoboCasa's own
        #      microwave_entering_payload_exclusions literally (predicates.py
        #      ~8487-8489).
        #   2. name == active and it is the SOLE object currently detected
        #      inside -- covers a real LIBERO-specific gap RoboCasa's own
        #      corpus apparently never hits: _check_grasp_any's bilateral-
        #      contact test can genuinely stop reading True for the rest of
        #      a placement once the demo's policy transitions from a pinch
        #      grasp to a push/slide for the final approach into the
        #      fixture, not just flicker for 1-2 frames. Confirmed on ep0:
        #      object_grasped_raw drops False at frame 150 and never reads
        #      True again through the rest of the 330-frame episode, while
        #      the mug's own tracked position stays a roughly-constant ~8cm
        #      from the end-effector the whole time (not free fall -- z
        #      drops only ~3.5cm over 60 frames, physically impossible under
        #      gravity alone) -- i.e. the robot is still actively placing
        #      the object at frame 210 when object_reach_in_microwave fires,
        #      condition 1 alone just can't see that anymore. A bare
        #      PERSISTENCE_FRAMES-style raw-key stability debounce (order
        #      5 frames) was considered and rejected: the gap here is 60+
        #      frames, an order of magnitude too long for that mechanism to
        #      bridge, and the "occupied" reading is itself perfectly
        #      stable/non-flickering for the entire gap, so a stability
        #      debounce has nothing to smooth over here. Restricted to the
        #      sole-occupant case specifically so a genuine two-object
        #      violation is never erased: if a second, distinct object is
        #      already inside when the active object also enters, this
        #      condition is false for both (len(inside_names) != 1), and
        #      the pre-existing occupant still counts.
        # active_settled (2026-09-21 fix, real-data regression found post-
        # 6b22648): the sole-active-occupant exemption above was gated only
        # on `active`'s identity, which `state["active_object"]` never
        # clears once set -- so once the carried object became the sole
        # microwave occupant, it stayed exempted from empty_check_occupants
        # for the REST OF THE EPISODE, even long after being placed down,
        # released, and settled (confirmed on KITCHEN_SCENE6_put_the_yellow_
        # and_white_mug_in_the_microwave_and_close_it ep0: microwave_empty
        # read True at every single frame 0-329, including frames 220-329
        # where object_in_fixture/one_object_in_microwave both correctly
        # read True the whole time -- a permanent false-negative, not just
        # the transient-placement false-positive 6b22648 was fixing).
        # object_settled (computed earlier this same frame, scoped to
        # settle_obj_name) is the natural boundary: the exemption should
        # only cover the genuine "still being placed" window (still
        # grasped, or dropped but not yet come to rest with the gripper
        # away), not indefinitely afterward. Requires settle_obj_name to
        # actually BE active for object_settled to count as "active is
        # settled" -- if a different, earlier-watched object's settle
        # state is what's currently computed (settle_obj_name != active),
        # conservatively treat active as not-yet-settled rather than
        # trusting an unrelated object's settle reading.
        active_settled = bool(active is not None and settle_obj_name == active and object_settled)
        solely_active_occupant = bool(
            active is not None and inside_names == [str(active)] and not active_settled
        )
        for name in inside_names:
            excluded = (object_grasped and name == active) or (
                solely_active_occupant and name == active
            )
            if not excluded:
                empty_check_occupants += 1
        if active:
            prev_active_in_fixture = state.get("prev_active_in_fixture", False)
            object_in_fixture_active = object_in_fixture
            object_reach_in_fixture = bool(object_in_fixture_active and not prev_active_in_fixture)
            if object_reach_in_fixture:
                state["last_reach_fixture"] = fixture_name
            state["prev_active_in_fixture"] = object_in_fixture_active
            object_in_same_fixture = bool(object_in_fixture_active and state.get("last_reach_fixture") == fixture_name)

    one_in_microwave = bool(_is_microwave(fixture_name) and occupants == 1)
    two_plus_in_microwave = bool(_is_microwave(fixture_name) and occupants >= 2)
    # Debounced via MICROWAVE_EMPTY_PERSISTENCE_FRAMES, matching RoboCasa's
    # own microwave_empty_count >= PERSISTENCE_FRAMES gate (predicates.py
    # ~8523-8531) -- counts consecutive frames with zero empty_check_
    # occupants (the payload-excluded count, see above) before reporting
    # empty, rather than a raw single-frame occupants==0 read.
    if _is_microwave(fixture_name) and empty_check_occupants == 0:
        microwave_empty_streak = int(state.get("microwave_empty_streak", 0)) + 1
    else:
        microwave_empty_streak = 0
    state["microwave_empty_streak"] = microwave_empty_streak
    microwave_empty = bool(
        (not _is_microwave(fixture_name))
        or microwave_empty_streak >= MICROWAVE_EMPTY_PERSISTENCE_FRAMES
    )

    # Added 2026-09-16 (explicit user decision, matching RoboCasa's own
    # predicates.py): object_reach_in_fixture is generic across any focus
    # fixture -- object_reach_in_microwave filters to exactly the microwave
    # case, since rc_microwave_single_object_until_empty's main_ltl now
    # uses this instead of the generic atom.
    object_reach_in_microwave = bool(object_reach_in_fixture and _is_microwave(fixture_name))
    if object_reach_in_microwave:
        state["microwave_reach_object"] = active
    microwave_reach_object = state.get("microwave_reach_object")
    object_left_microwave = False
    if microwave_reach_object is not None and fixture_name is not None and _is_microwave(fixture_name):
        obj_state = object_states_dict.get(microwave_reach_object)
        fixture_state = object_states_dict.get(fixture_name)
        if obj_state is not None and fixture_state is not None:
            # Same check_contain -> region-site fix as object_in_fixture's
            # occupancy loop above (2026-09-21 dependency-tree audit) --
            # check_contain raises on this corpus's microwave fixture class,
            # silently swallowed to still_in=False, so object_left_microwave
            # was always vacuously True.
            reach_obj_pos = _body_pos(env, microwave_reach_object)
            region_still_in = _point_in_any_fixture_region(env, fixture_name, reach_obj_pos)
            if region_still_in is None:
                try:
                    still_in = bool(fixture_state.check_contact(obj_state) and fixture_state.check_contain(obj_state))
                except Exception:
                    still_in = False
            else:
                still_in = bool(region_still_in)
            object_left_microwave = not still_in

    predicates["fixture_fully_open"] = _entry(fixture_fully_open, "focus fixture reports is_open()")
    predicates["fixture_fully_closed"] = _entry(fixture_fully_closed, "focus fixture reports is_close()")
    predicates["reach_in_fixture"] = _entry(reach_in_fixture, "gripper newly within interior radius of focus fixture")
    predicates["left_fixture"] = _entry(left_fixture, "gripper just exited interior radius of focus fixture")
    predicates["gripper_in_fixture"] = _entry(gripper_in_fixture, "gripper within interior radius of focus fixture")
    predicates["object_reach_in_fixture"] = _entry(object_reach_in_fixture, "active object newly contained in focus fixture")
    predicates["object_reach_in_microwave"] = _entry(object_reach_in_microwave, "active object newly contained in the microwave specifically")
    predicates["object_left_microwave"] = _entry(object_left_microwave, "the object that triggered object_reach_in_microwave is no longer contained in it")
    predicates["object_in_fixture"] = _entry(object_in_fixture, "active/focus object contained in focus fixture")
    predicates["object_in_same_fixture"] = _entry(object_in_same_fixture, "still inside the fixture it reached into")
    predicates["one_object_in_microwave"] = _entry(one_in_microwave, "exactly one object contained in microwave")
    predicates["two_or_more_objects_in_microwave"] = _entry(two_plus_in_microwave, "2+ objects contained in microwave")
    predicates["microwave_empty"] = _entry(microwave_empty, "no object contained in microwave")

    _cp_registry = state.get("contact_role_registry") or {}
    _cp_source_by_object = {
        obj: {
            "fixtures": sorted(_cp_registry.get("source_fixtures_by_object", {}).get(obj, set())),
            "objects": sorted(_cp_registry.get("source_objects_by_object", {}).get(obj, set())),
        }
        for obj in set(_cp_registry.get("source_fixtures_by_object", {}))
        | set(_cp_registry.get("source_objects_by_object", {}))
    }
    _cp_receive_objects = sorted(
        {o for s in _cp_registry.get("target_objects_by_object", {}).values() for o in s}
    )
    return {
        "sections": {"predicates": predicates},
        "role_sets": {
            "active_object": active,
            "focus_pick_object": focus_pick_object,
            "focus_fixture": fixture_name,
            "manipulated_objects": sorted(_cp_registry.get("manipulated_objects", set())),
            "receive_objects": _cp_receive_objects,
            "target_fixtures": sorted(_cp_registry.get("all_target_fixtures", set())),
            "source_supports_by_object": _cp_source_by_object,
        },
        "violation_evidence": {
            "forbidden_contact_pairs": cp_forbidden_contact_pairs,
            "considered_contact_pairs": cp_considered_contact_pairs,
            "active_object": active,
            "manipulated_objects": sorted(_cp_registry.get("manipulated_objects", set())),
            "target_fixtures": sorted(_cp_registry.get("all_target_fixtures", set())),
        },
    }
