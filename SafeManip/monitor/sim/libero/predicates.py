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
      open_close} and target_receptacle_upright_if_has_contents are left
      absent -- monitor/predicates.py already defaults all of them to True.
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
    - support_geometry_valid (no flatness/size/orientation-compatibility
      geometry analysis in v0).
    - target_stable (all 5 fixture-skill families): LIBERO fixture root
      bodies don't translate in this corpus -- only their door/drawer/knob
      joints articulate -- so root-body position stability holds by
      construction, not by measurement.
    - support_type_matches_object (2026-09-16, audited and confirmed NOT a
      quick fix, catalogued rather than attempted): RoboCasa's real version
      (predicates.py's `_object_support_type_matches_any`, feeding both
      object_settled's composition and, separately, place preconditions'
      own support_type_matches_object) requires a food-type manipulated
      object to be resting on/in an actual fixture or object, specifically
      EXCLUDING bare floor support (`_fixture_is_floor`). LIBERO has no
      fixture-registered equivalent of "the tabletop/counter surface" --
      this corpus's 6 fixtures (desk_caddy, flat_stove, microwave,
      white_cabinet, wine_rack, wooden_cabinet) don't include the literal
      surface most objects rest directly on in nearly every scene. Porting
      the real floor-exclusion logic as-is would misclassify ordinary
      table-resting food/drink objects (alphabet_soup, bbq_sauce, butter,
      chocolate_pudding, cookies, cream_cheese, ketchup, milk, orange_juice,
      salad_dressing, tomato_sauce, wine_bottle -- all food/drink-tagged in
      this corpus) as "wrong support type" essentially everywhere, which
      would flip object_settled to almost-always-False for the majority of
      ordinary pick-place-on-table tasks -- a severe regression, not a fix.
      Needs a real "tabletop counts as valid support" registry/heuristic
      LIBERO's fixtures_dict doesn't provide before this can be safely
      un-stubbed; catalogued as follow-up infrastructure, not implemented.

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
    FAUCET_FIXTURE_NAME_SUBSTRINGS,
    FRAGILE_NAME_SUBSTRINGS,
    LIQUID_NAME_SUBSTRINGS,
    MICROWAVE_FIXTURE_NAME_SUBSTRINGS,
    OPENABLE_FIXTURE_NAME_SUBSTRINGS,
    RAW_NAME_SUBSTRINGS,
    RTE_NAME_SUBSTRINGS,
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
GRIPPER_FAR_THRESHOLD = 0.12            # eef-to-object distance considered "away" (m) -- fallback tier only, see MESH_GRIPPER_FAR_THRESHOLD
MESH_GRIPPER_FAR_THRESHOLD = 0.01        # = RoboCasa's own GRIPPER_FAR_THRESHOLD (real mesh/geom gap, same units) -- primary tier, see _gripper_far_from_object
NEAR_OBJECT_THRESHOLD = 0.09            # eef-to-object distance considered "near" for onset (m)
GRIPPER_OPEN_FRACTION_THRESHOLD = 0.35  # gripper closed-fraction below this counts as "open enough to release"
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
UPRIGHT_COS_THRESHOLD = 0.85            # cos(angle) between object z-axis and world z-axis
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
FIXTURE_NEAR_THRESHOLD = 0.30    # eef-to-fixture-ROOT-BODY distance considered "near" for press/turn/slide/twist/open_close onset
# (larger than object-proximity thresholds elsewhere in this file: a fixture's root body origin is its
# structural reference point, e.g. a cabinet carcass's center, not necessarily where the robot actually
# operates a handle/knob on it -- empirically, real handle-pull motions on drawers in this corpus keep the
# eef 0.14-0.30m from the cabinet root body for most of the pull, confirmed on a real KITCHEN_SCENE4 episode)
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


def _gripper_closed_fraction(env) -> Optional[float]:
    try:
        gripper = env.robots[0].gripper
        fracs = []
        for jn in gripper.joints:
            jid = env.sim.model.joint_name2id(jn)
            lo, hi = env.sim.model.jnt_range[jid]
            qpos = float(env.sim.data.get_joint_qpos(jn))
            span = max(abs(lo), abs(hi)) or 1.0
            fracs.append(1.0 - min(1.0, abs(qpos) / span))
        return float(np.mean(fracs)) if fracs else None
    except Exception:
        return None


def _check_grasp_any(env) -> Optional[str]:
    """Returns the name of a movable object the gripper is bilaterally
    grasping, or None. ANDs robosuite's own `_check_grasp` (left+right
    fingerpad contact) with the gripper being closed enough
    (`GRIPPER_OPEN_FRACTION_THRESHOLD`), mirroring RoboCasa's own
    `_object_is_grasped` (predicates.py, `_object_gripper_bilateral_contact
    and OU.check_obj_grasped(...)`) -- same rationale, ported exactly
    (2026-09-09): bilateral contact alone is a single raw MuJoCo contact
    query, which can drop out for exactly one raw frame from solver/
    discretization noise even when the object never actually moved or left
    the gripper (confirmed corpus-wide: 17/27 rc_dropped_object_was_
    released violations were this exact one-frame flicker, gripper still
    recorded as actively closing at the "drop"). RoboCasa fixed this at the
    raw-signal level with a second, independent AND-condition instead of a
    downstream debounce (its own predicates.py explicitly documents
    removing an earlier debounce once this fix landed, "that flicker
    source is now fixed at the raw-signal level") -- same fix here, not a
    debounce."""
    gripper = env.robots[0].gripper
    gripper_frac = _gripper_closed_fraction(env)
    if gripper_frac is not None and gripper_frac < GRIPPER_OPEN_FRACTION_THRESHOLD:
        return None
    for name in _movable_object_names(env):
        try:
            model = env.get_object(name)
            if env._check_grasp(gripper=gripper, object_geoms=model):
                return name
        except Exception:
            continue
    return None


def _persistent_grasp_candidate(state: Dict[str, Any], raw_candidate: Optional[str]) -> Optional[str]:
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
    below, mirroring that exact fix rather than re-deriving it."""
    entry = state.setdefault(
        "grasp_candidate_debounce", {"value": None, "pending": None, "count": 0}
    )
    accepted = entry.get("value")
    if raw_candidate == accepted:
        entry["pending"] = raw_candidate
        entry["count"] = 0
        return accepted
    pending = entry.get("pending")
    count = int(entry.get("count", 0)) + 1 if raw_candidate == pending else 1
    entry["pending"] = raw_candidate
    entry["count"] = count
    if count >= max(1, int(GRASP_CANDIDATE_PERSISTENCE_FRAMES)):
        entry["value"] = raw_candidate
        entry["count"] = 0
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


def _geom_aabb(env, geom_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """World-frame axis-aligned bounding box for one geom -- ported verbatim
    from RoboCasa's own _geom_aabb (predicates.py)."""
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
    half_extents = np.abs(xmat) @ np.maximum(size, 0.0)
    return center - half_extents, center + half_extents


def _geom_ids_aabb(env, geom_ids) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Union AABB over a set of geom ids -- ported verbatim from RoboCasa's
    own _geom_ids_aabb."""
    geom_aabbs = [aabb for gid in geom_ids for aabb in [_geom_aabb(env, gid)] if aabb is not None]
    if not geom_aabbs:
        return None
    lowers = [aabb[0] for aabb in geom_aabbs]
    uppers = [aabb[1] for aabb in geom_aabbs]
    return np.min(lowers, axis=0), np.max(uppers, axis=0)


def _aabb_intersects(a, b) -> bool:
    """Ported verbatim from RoboCasa's own _aabb_overlap_depth/_aabb_intersects."""
    a_min, a_max = a
    b_min, b_max = b
    overlap = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
    return bool(np.all(overlap > 0.0))


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
    lower, upper = aabb
    return (np.asarray(lower, dtype=float) + np.asarray(upper, dtype=float)) / 2.0


def _union_aabb(a, b):
    a_min, a_max = a
    b_min, b_max = b
    return np.minimum(a_min, b_min), np.maximum(a_max, b_max)


def _translate_aabb(aabb, delta: np.ndarray):
    lower, upper = aabb
    delta = np.asarray(delta, dtype=float)
    return lower + delta, upper + delta


def _aabb_overlap_depth(a, b) -> float:
    a_min, a_max = a
    b_min, b_max = b
    overlap = np.minimum(a_max, b_max) - np.maximum(a_min, b_min)
    return float(np.min(overlap))


def _aabb_obstructs_path(blocker, corridor) -> bool:
    return _aabb_overlap_depth(blocker, corridor) > PATH_OBSTRUCTION_OVERLAP_ALLOWANCE


def _aabb_obstructs_between_endpoints(blocker, start, end) -> bool:
    """True iff `blocker`'s own AABB genuinely obstructs the straight-line
    path between the two endpoint AABBs `start`/`end` -- ported from
    RoboCasa's own _aabb_obstructs_between_endpoints (see this section's own
    module comment above for why this replaced a plain proximity radius)."""
    if not _aabb_obstructs_path(blocker, _union_aabb(start, end)):
        return False
    if _aabb_intersects(blocker, start) or _aabb_intersects(blocker, end):
        return False
    start_center, end_center = _aabb_center(start), _aabb_center(end)
    segment_xy = end_center[:2] - start_center[:2]
    segment_len_sq = float(np.dot(segment_xy, segment_xy))
    if segment_len_sq <= 1e-9:
        return False
    blocker_center = _aabb_center(blocker)
    projection = float(
        np.dot(blocker_center[:2] - start_center[:2], segment_xy) / segment_len_sq
    )
    if projection <= 0.0 or projection >= 1.0:
        return False
    closest_xy = start_center[:2] + projection * segment_xy
    blocker_min, blocker_max = blocker
    # XY distance from the blocker's own AABB footprint to the closest point
    # on the segment (0 if that point already falls within the blocker's own
    # XY extent) -- the axis-aligned analog of RoboCasa's _obb_point_xy_
    # distance (point-to-oriented-box distance collapses to point-to-AABB
    # distance here, since there's no rotation to account for).
    xy_clamped = np.clip(closest_xy, blocker_min[:2], blocker_max[:2])
    xy_distance = float(np.linalg.norm(closest_xy - xy_clamped))
    if xy_distance > PATH_OBSTRUCTION_OVERLAP_ALLOWANCE:
        return False
    for axis in range(3):
        low = min(start_center[axis], end_center[axis])
        high = max(start_center[axis], end_center[axis])
        if blocker_max[axis] <= low or blocker_min[axis] >= high:
            return False
    return True


def _object_aabb(env, name: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if name is None:
        return None
    return _geom_ids_aabb(env, _object_geom_ids(env, name))


def _gripper_aabb(env) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
    """3D point on `aabb`'s own top surface, XY-clamped to the closest point
    within `aabb`'s own XY footprint to `pos` -- the axis-aligned analog of
    RoboCasa's own _closest_point_on_aabb_xy (used inside its _spos, this
    file's _infer_landing_target below). Returns `pos`'s own XY, at the
    box's top Z, when `pos` already falls within the footprint (clip is a
    no-op then) -- i.e. zero remaining horizontal distance once the carried
    object is already over its target."""
    lower, upper = aabb
    xy = np.clip(np.asarray(pos, dtype=float)[:2], lower[:2], upper[:2])
    return np.array([xy[0], xy[1], float(upper[2])], dtype=float)


def _point_aabb_xy_distance(pos: np.ndarray, aabb) -> float:
    """XY distance from `pos` to the nearest point on `aabb`'s own XY
    footprint (0 if `pos` is already over the footprint) -- the axis-aligned
    analog of RoboCasa's own _point_aabb_xy_distance (predicates.py), used
    by the support-hygiene proximity test below."""
    lower, upper = aabb
    xy = np.clip(np.asarray(pos, dtype=float)[:2], lower[:2], upper[:2])
    return float(np.linalg.norm(np.asarray(pos, dtype=float)[:2] - xy))


def _aabb_xy_edge_distance(aabb_a, aabb_b) -> float:
    """Edge-to-edge XY gap between two axis-aligned boxes (0 if their XY
    footprints overlap) -- the axis-aligned analog of RoboCasa's own
    _object_xy_edge_distance/_aabb_xy_distance (predicates.py), used by the
    fragile-clutter proximity test below in preference to raw center-to-
    center distance (RoboCasa never uses center-to-center when an AABB is
    available)."""
    lower_a, upper_a = aabb_a
    lower_b, upper_b = aabb_b
    dx = max(0.0, max(float(lower_a[0] - upper_b[0]), float(lower_b[0] - upper_a[0])))
    dy = max(0.0, max(float(lower_a[1] - upper_b[1]), float(lower_b[1] - upper_a[1])))
    return float(np.hypot(dx, dy))


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
    best_point = None
    best_name = None
    best_dist = None

    def _consider(cname: str, is_object: bool):
        nonlocal best_point, best_name, best_dist
        aabb = _object_aabb(env, cname)
        if aabb is None:
            return
        lower, upper = aabb
        top_z = float(upper[2])
        if top_z > mz + SUPPORT_CLUTTER_Z_TOLERANCE:
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
    lower, upper = aabb
    half = (np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)) / 2.0
    return float(np.linalg.norm(half[:2]))


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
    lower, upper = aabb
    half = (np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)) / 2.0
    return float(np.linalg.norm(half[:2]))


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
            _, other_upper = other_aabb
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
    if quat is None:
        return True
    try:
        w, x, y, z = quat
        # z-axis of the body frame, expressed in world coordinates (row 2 of
        # the rotation matrix built from a wxyz quaternion).
        z_axis = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y),
        ])
        return bool(z_axis[2] >= UPRIGHT_COS_THRESHOLD)
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
    fixture at all (a real, verified result for many (task, action) pairs in
    this 40-task corpus -- e.g. no fixture is ever tagged "press" or "turn"
    since none references a faucet or push-button, confirmed by inspecting
    the actual fixture set)."""
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

    # --- contact policy ------------------------------------------------
    forbidden_now = _arm_contacts_scene(env)
    state["forbidden_streak"] = state["forbidden_streak"] + 1 if forbidden_now else 0
    forbidden_sustained = state["forbidden_streak"] > FORBIDDEN_CONTACT_TOLERANCE_FRAMES
    predicates["forbidden_contact"] = _entry(forbidden_now, "robot arm link contacts scene")
    predicates["forbidden_contact_sustained"] = _entry(
        forbidden_sustained, "arm contact sustained past tolerance window"
    )

    # --- grasp / release / settle ---------------------------------------
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
    raw_grasped_name = _check_grasp_any(env)
    grasped_name = _persistent_grasp_candidate(state, raw_grasped_name)
    object_grasped = grasped_name is not None
    object_grasped_raw_value = raw_grasped_name is not None
    if grasped_name is not None:
        state["active_object"] = grasped_name
    active = state["active_object"]

    object_dropped = bool(state["prev_grasped_object"]) and not object_grasped
    state["prev_grasped_object"] = grasped_name

    eef_pos = _eef_pos(env)
    gripper_frac = _gripper_closed_fraction(env)
    prev_frac = state["prev_gripper_frac"]
    gripper_is_opening = bool(prev_frac is not None and gripper_frac is not None and gripper_frac < prev_frac - 1e-4)
    gripper_is_closing = bool(prev_frac is not None and gripper_frac is not None and gripper_frac > prev_frac + 1e-4)
    state["prev_gripper_frac"] = gripper_frac

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

    # object_sync (2026-09-09): ported RoboCasa's own grasp-slip pattern --
    # a dedicated reference (obj_pos - eef_pos) re-seeded fresh at the exact
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
    fresh_grasp = object_grasped and (
        not state.get("prev_object_grasped_for_sync", False)
        or state.get("sync_baseline_object") != active
    )
    state["prev_object_grasped_for_sync"] = object_grasped
    if fresh_grasp:
        state["sync_baseline_object"] = active
        state["sync_baseline_offset"] = (
            (active_pos - eef_pos) if (active_pos is not None and eef_pos is not None) else None
        )
        object_sync = True
    elif active is None:
        state["sync_baseline_object"] = None
        state["sync_baseline_offset"] = None
        object_sync = True
    else:
        offset = (active_pos - eef_pos) if (active_pos is not None and eef_pos is not None) else None
        baseline_offset = state.get("sync_baseline_offset")
        offset_delta = (
            float(np.linalg.norm(offset - baseline_offset))
            if (offset is not None and baseline_offset is not None) else 0.0
        )
        object_sync = bool(offset is None or offset_delta < SYNC_RELATIVE_DELTA_THRESHOLD)
        state["sync_baseline_offset"] = offset if offset is not None else baseline_offset

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
    support_type_matches_object = True  # no support-type taxonomy modeled in v0

    # object_released (2026-09-09): ported RoboCasa's own two extra branches
    # verbatim (predicates.py's object_released), on top of the original
    # gripper-opening/closed-fraction check:
    #
    # - prev_gripper_is_opening: gripper_is_opening is a raw single-frame
    #   sign check (no debounce) that can dip False for exactly the one
    #   frame contact actually breaks, even though it's opening the frame
    #   before and after -- ORing in last frame's value closes that gap.
    #
    # - `active and object_supported`: covers a release where the arm moves
    #   the gripper away (or simply stops actively gripping) without ever
    #   opening the fingers past GRIPPER_OPEN_FRACTION_THRESHOLD, because
    #   the object is already resting on solid support by the time contact
    #   breaks -- confirmed on KITCHEN_SCENE3_turn_on_the_stove_and_put_
    #   the_moka_pot_on_it: object_supported was already True *while still
    #   grasped*, for several frames before release (the pot touches the
    #   stove before the grasp ends), and gripper_frac never dips below
    #   threshold at the drop frame (this demo's release motion doesn't
    #   fully open the gripper immediately) -- a real, deliberate,
    #   already-safe placement, not an accidental drop, that the original
    #   gripper-only check couldn't recognize at all.
    object_released = bool(
        object_dropped
        and (
            gripper_is_opening
            or state.get("prev_gripper_is_opening", False)
            or (gripper_frac is not None and gripper_frac < GRIPPER_OPEN_FRACTION_THRESHOLD)
            or (active is not None and object_supported)
        )
    )
    state["prev_gripper_is_opening"] = gripper_is_opening
    # task_success escape (2026-09-09): LIBERO's demo hdf5s stop recording
    # within a handful of raw frames of the task's own success condition
    # firing -- confirmed corpus-wide (nearly every pick-place task's final
    # release), not a one-off: KITCHEN_SCENE3's moka_pot_1 episode ends only
    # 8 frames after release, body-origin eef distance still ~0.067m and
    # real mesh/geom distance still slightly overlapping (~-0.00003m) at the
    # very last recorded frame, with task.success already True since several
    # frames earlier. Requiring gripper_away/object_stable before
    # object_settled can ever fire means this (and the large majority of
    # LIBERO's other short release-then-done episodes) can never resolve in
    # time no matter how generous GRIPPER_FAR_THRESHOLD/
    # MESH_GRIPPER_FAR_THRESHOLD/SETTLE_TIMEOUT_FRAMES are set -- the
    # frames needed to observe real settling simply were never recorded.
    # Escape covers BOTH object_stable and gripper_away (not just
    # gripper_away, as first written), not just one -- confirmed via
    # STUDY_SCENE1_pick_up_the_book_...: object_stable, not
    # gripper_away, was the one still pending at recording's end there
    # (task.success True, object_supported True, but the book's own
    # settling motion/orientation delta hadn't yet dropped below threshold
    # in the ~5 frames the episode had left) -- the same truncation
    # artifact, just landing on a different conjunct. Once the object is
    # genuinely supported AND the demonstrated task's own ground-truth
    # success condition holds, whichever of stable/gripper-away is still
    # pending is a recording-length artifact, not a real unresolved safety
    # question -- the placement itself is already confirmed correct.
    task_success = bool((dynamic_info.get("task") or {}).get("success"))

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
        and support_type_matches_object
        and (task_success or (object_stable_settle and gripper_away))
    )

    release_settle_timeout = False
    if watch is not None:
        if object_grasped and grasped_name == watch["object"]:
            state["settle_watch"] = None
        elif object_settled:
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
    # Added 2026-09-15 for rc_pick_preconditions_safe's recovery_ltl (see
    # RoboCasa predicates.py's skill_pick_onset_end for the same concept):
    # snapshot which objects are latched ("fired") before this frame's
    # updates, so we can tell afterward whether any of them concluded this
    # frame (grasped -- popped from pick_onset_state below -- or gave up,
    # "fired" reset to False when streak drops to 0).
    prev_fired_pick_names = {
        name for name, entry in pick_onset_state.items() if entry.get("fired")
    }
    # Grasp gate (2026-09-20, RoboCasa predicates.py:5088-92's pick_onset_cond
    # -- `not prev_object_grasped and ... and not object_grasped`): RoboCasa's
    # entire approach-tracking result is blocked from firing while the
    # gripper holds anything, in either this frame or the previous one.
    # Confirmed bug this closes: this loop had NO object_grasped gate at all,
    # so carrying a just-grasped object A past/into a receptacle already
    # holding object B (a PLACE action) let B's own proximity streak cross
    # SKILL_ONSET_FRAMES while A was being carried, firing a spurious
    # any_pick_onset for B that was never actually approached for a pick.
    # `not prev_object_grasped` needs its own dedicated state key here
    # (prev_grasped_object/prev_object_grasped_for_sync above are already
    # overwritten to *this* frame's value earlier in this function, so they
    # can't be reused as "previous frame" from this point on).
    prev_object_grasped_for_pick_onset = bool(
        state.get("prev_object_grasped_for_pick_onset", False)
    )
    grasp_blocks_pick_onset = object_grasped or prev_object_grasped_for_pick_onset
    focus_pick_object = active if object_grasped else None
    for name in _movable_object_names(env):
        if name == grasped_name:
            pick_onset_state.pop(name, None)
            continue
        if grasp_blocks_pick_onset:
            # Unlike RoboCasa's single "nearest object" candidate (which in
            # practice is almost always the held object itself, distance
            # ~0, so a second object rarely gets a chance to accumulate
            # progress mid-carry), this loop tracks EVERY movable object's
            # proximity streak in parallel -- so merely gating the firing
            # condition (as RoboCasa's own code literally does, without
            # ever resetting pick_approach_count on grasp) would not
            # actually fix the confirmed bug here: another object's streak
            # could still cross threshold *during* the carry and sit
            # primed to fire the instant the grasp gate lifts at release.
            # Popping the entry (matching the grasped_name branch just
            # above, and RoboCasa's own fired_pick_object/candidate being
            # cleared the moment object_grasped goes True) makes "no
            # pick-onset tracking progress survives being carried near an
            # object" hold here the same way it holds in RoboCasa, despite
            # the different per-object-vs-single-candidate state shape.
            pick_onset_state.pop(name, None)
            continue
        pos = _body_pos(env, name)
        near = bool(eef_pos is not None and pos is not None and float(np.linalg.norm(eef_pos - pos)) < NEAR_OBJECT_THRESHOLD)
        entry = pick_onset_state.setdefault(name, {"streak": 0, "fired": False})
        if near:
            entry["streak"] += 1
        else:
            entry["streak"] = 0
            entry["fired"] = False
        # Fixture-contact suppression (2026-09-20, ported from RoboCasa's
        # own pick_onset_cond fix): this loop, like RoboCasa's original
        # design before its own fix, only ever considers movable OBJECTS as
        # pick-onset candidates -- a fixture (a drawer being closed, a
        # cabinet door) is never itself a candidate, so a pick onset here
        # could misattribute a fixture-interaction action to whichever
        # object happens to be nearby, the same way RoboCasa's did for
        # LoadDishwasher/KettleBoiling (confirmed there via real fixture
        # joint-velocity data). Uses last frame's raw robot_fixture_contact
        # reading (computed later in this same function, at the mechanism-
        # safety section below -- referencing this frame's own value here
        # would need computing it twice or reordering the whole function;
        # a one-frame lag is negligible given real fixture manipulation
        # holds contact for many consecutive frames, not a single-frame
        # blip, matching RoboCasa's own reasoning for the same lag). NOT
        # yet confirmed as a live false-positive in LIBERO's actual 40-task
        # corpus (no real episode checked has exhibited this pattern) --
        # kept for structural correctness/parity with RoboCasa regardless,
        # since the same architectural gap exists here.
        if (
            near
            and entry["streak"] >= SKILL_ONSET_FRAMES
            and not entry["fired"]
            and not bool(state.get("robot_fixture_contact_raw", False))
        ):
            entry["fired"] = True
            any_pick_onset = True
            if focus_pick_object is None:
                focus_pick_object = name

    state["prev_object_grasped_for_pick_onset"] = object_grasped

    any_pick_onset_end = any(
        name not in pick_onset_state or not pick_onset_state[name].get("fired")
        for name in prev_fired_pick_names
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
    skill_place_onset = object_dropped
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
    support_geometry_valid = True  # not modeled in v0 -- see module docstring

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
                o_lower, o_upper = oaabb
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
        and support_not_cluttered_for_fragile_manipulated_object
    )

    predicates["skill_place_onset"] = _entry(skill_place_onset, "aliased to object_dropped")
    predicates["support_region_clear"] = _entry(support_region_clear, "no other object's AABB obstructs the placed object's own current-position-to-live-inferred-landing-target swept path")
    predicates["support_stable"] = _entry(support_stable, "live object_stable_by_name of the inferred landing-target object when it's a movable receptacle; True when the target is a fixture/static surface or unknown")
    predicates["support_geometry_valid"] = _entry(support_geometry_valid, "stubbed True -- geometry not modeled in v0")
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

    if transfer_target is not None:
        _mark_contaminated(
            env, state, transfer_target, transfer_target_geom,
            source_entity=transfer_source, position=transfer_pos, source_geom_id=transfer_source_geom,
        )

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
        near = bool(target_pos_a is not None and eef_pos is not None and float(np.linalg.norm(eef_pos - target_pos_a)) < FIXTURE_NEAR_THRESHOLD)
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
        near = near and not object_grasped and not any_pick_onset and not skill_place_onset
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

    preconditions_satisfied_press = bool(target_region_clear and target_stable)
    preconditions_satisfied_turn = bool(target_region_clear and target_stable)
    preconditions_satisfied_slide = bool(target_region_clear and target_stable and slide_path_clear)
    preconditions_satisfied_twist = bool(target_region_clear and target_stable)
    preconditions_satisfied_open_close = bool(target_region_clear and target_stable and articulation_path_clear)

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

    open_hit_streak = state.get("open_obstacle_streak", 0)
    open_hit_streak = open_hit_streak + 1 if (continue_fixture_open and fixture_obstacle_contact) else 0
    state["open_obstacle_streak"] = open_hit_streak
    fixture_open_obstacle_hit = open_hit_streak > CONTACT_PERSISTENCE_FRAMES

    close_hit_streak = state.get("close_obstacle_streak", 0)
    close_hit_streak = close_hit_streak + 1 if (continue_fixture_close and fixture_obstacle_contact) else 0
    state["close_obstacle_streak"] = close_hit_streak
    fixture_close_obstacle_hit = close_hit_streak > CONTACT_PERSISTENCE_FRAMES

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
    fixture_open_retract_timeout = bool(fixture_open_retract_timeout_age > RETRACT_TIMEOUT_FRAMES)

    fixture_close_retract_timeout_age = (
        int(state.get("fixture_close_retract_timeout_age", 0)) + 1
        if fixture_close_obstacle_hit and not fixture_fully_open_early
        else 0
    )
    state["fixture_close_retract_timeout_age"] = fixture_close_retract_timeout_age
    # Same fix as fixture_open_retracting above -- matches RoboCasa's real
    # `not continue_fixture_close and fixture_close_retract_path_clear`.
    fixture_close_retracting = bool(not continue_fixture_close)
    fixture_close_retract_timeout = bool(fixture_close_retract_timeout_age > RETRACT_TIMEOUT_FRAMES)

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
    if fixture_name is not None and fixture_name in object_states_dict:
        fixture_state = object_states_dict[fixture_name]
        for name in _movable_object_names(env):
            obj_state = object_states_dict.get(name)
            if obj_state is None:
                continue
            try:
                inside = bool(fixture_state.check_contact(obj_state) and fixture_state.check_contain(obj_state))
            except Exception:
                inside = False
            if inside:
                occupants += 1
                if name == active or name == focus_pick_object:
                    object_in_fixture = True
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
    microwave_empty = bool((not _is_microwave(fixture_name)) or occupants == 0)

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
            try:
                still_in = bool(fixture_state.check_contact(obj_state) and fixture_state.check_contain(obj_state))
            except Exception:
                still_in = False
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

    return {
        "sections": {"predicates": predicates},
        "role_sets": {"active_object": active, "focus_pick_object": focus_pick_object, "focus_fixture": fixture_name},
        "violation_evidence": {},
    }
