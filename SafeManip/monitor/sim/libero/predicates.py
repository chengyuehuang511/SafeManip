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
      preconditions_satisfied_pick.
    - place_preconditions: skill_place_onset (== object_released),
      support_region_clear, support_stable, preconditions_satisfied_place
      (support_geometry_valid explicitly stubbed True -- see below).
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
      fixture_open/close_retract_resolved (reaches the opposite extreme, or
      retracting has held continuously for FIXTURE_RETRACT_RESOLVE_TIMEOUT_
      FRAMES). Empirically active on the microwave-door task in this
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
    LIQUID_NAME_SUBSTRINGS,
    MICROWAVE_FIXTURE_NAME_SUBSTRINGS,
    OPENABLE_FIXTURE_NAME_SUBSTRINGS,
    RAW_NAME_SUBSTRINGS,
    RTE_NAME_SUBSTRINGS,
    object_category_from_instance_name,
    object_is_receptacle_category,
)

# ---------------------------------------------------------------------------
# v0 threshold constants (all in meters / radians / raw-frame counts; NOT
# empirically tuned -- see module docstring).
# ---------------------------------------------------------------------------
STABLE_LINEAR_DELTA_THRESHOLD = 0.004   # per-raw-frame position delta (m)
STABLE_ANGULAR_DELTA_THRESHOLD = 0.05   # per-raw-frame orientation delta (rad, small-angle approx)
SYNC_RELATIVE_DELTA_THRESHOLD = 0.01    # per-raw-frame change in (obj - eef) offset (m)
GRIPPER_FAR_THRESHOLD = 0.12            # eef-to-object distance considered "away" (m) -- fallback tier only, see MESH_GRIPPER_FAR_THRESHOLD
MESH_GRIPPER_FAR_THRESHOLD = 0.02        # real mesh/geom gap (m) considered "away" -- primary tier, see _gripper_far_from_object
NEAR_OBJECT_THRESHOLD = 0.09            # eef-to-object distance considered "near" for onset (m)
GRIPPER_OPEN_FRACTION_THRESHOLD = 0.35  # gripper closed-fraction below this counts as "open enough to release"
REGION_CLEAR_RADIUS = 0.10              # radius (m) used by object/support region-clear checks
REGION_CLEAR_MAX_FOREIGN = 1            # allowed foreign objects within that radius
UPRIGHT_COS_THRESHOLD = 0.85            # cos(angle) between object z-axis and world z-axis
FIXTURE_INTERIOR_RADIUS = 0.18          # eef/object-to-fixture-body distance considered "inside" (m)
FIXTURE_ARTICULATION_DELTA_THRESHOLD = 2e-3  # per-raw-frame open-fraction delta counted as "articulating"

SKILL_ONSET_FRAMES = 5          # consecutive near-object/contact-and-articulating frames before an onset fires
SETTLE_TIMEOUT_FRAMES = 60      # frames a dropped/released object has to settle before timeout
FORBIDDEN_CONTACT_TOLERANCE_FRAMES = 10  # frames of arm-contact tolerated before "sustained"
CONTACT_PERSISTENCE_FRAMES = 3   # frames an open/close obstacle contact must persist before counting as a "hit"
FIXTURE_RETRACT_RESOLVE_TIMEOUT_FRAMES = 60  # frames of continuous retracting that counts as resolved even short of the opposite extreme
FIXTURE_NEAR_THRESHOLD = 0.30    # eef-to-fixture-ROOT-BODY distance considered "near" for press/turn/slide/twist/open_close onset
# (larger than object-proximity thresholds elsewhere in this file: a fixture's root body origin is its
# structural reference point, e.g. a cabinet carcass's center, not necessarily where the robot actually
# operates a handle/knob on it -- empirically, real handle-pull motions on drawers in this corpus keep the
# eef 0.14-0.30m from the cabinet root body for most of the pull, confirmed on a real KITCHEN_SCENE4 episode)
CONTAMINATION_PERSISTENCE_FRAMES = 3  # frames a clean-object grasp must hold before robot_contact_clean fires


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
    grasping (robosuite's own `_check_grasp` -- left+right fingerpad contact),
    or None."""
    gripper = env.robots[0].gripper
    for name in _movable_object_names(env):
        try:
            model = env.get_object(name)
            if env._check_grasp(gripper=gripper, object_geoms=model):
                return name
        except Exception:
            continue
    return None


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


def _object_stable_by_name(env, state: Dict[str, Any], name: Optional[str]) -> bool:
    """Per-object linear/angular-delta stability, generalizing the
    single-`active`-object `object_stable` computed inline in
    build_predicate_snapshot to any named object -- needed because pick
    preconditions must judge the object actually being approached
    (`focus_pick_object`), which is usually not yet grasped (so not
    `active`) at the moment a pick onset fires. Compares this frame's fresh
    position/quat (queried directly, same as the inline `active` version)
    against state["prev_positions"]/state["prev_quats"], already populated
    for every movable object name at the end of every prior
    build_predicate_snapshot call."""
    if name is None:
        return True
    pos = _body_pos(env, name)
    quat = _body_quat(env, name)
    prev_pos = state.get("prev_positions", {}).get(name)
    prev_quat = state.get("prev_quats", {}).get(name)
    lin_delta = float(np.linalg.norm(pos - prev_pos)) if (pos is not None and prev_pos is not None) else 0.0
    ang_delta = _angle_between_quats(quat, prev_quat)
    return bool(lin_delta < STABLE_LINEAR_DELTA_THRESHOLD and ang_delta < STABLE_ANGULAR_DELTA_THRESHOLD)


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


def _region_clear(env, center: Optional[np.ndarray], exclude: List[str]) -> bool:
    if center is None:
        return True
    foreign = 0
    for name in _movable_object_names(env):
        if name in exclude:
            continue
        pos = _body_pos(env, name)
        if pos is None:
            continue
        if np.linalg.norm(pos - center) < REGION_CLEAR_RADIUS:
            foreign += 1
    return foreign <= REGION_CLEAR_MAX_FOREIGN


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


def _generic_fixture_onset(state: Dict[str, Any], key: str, near: bool) -> bool:
    """Same near+persistence+fires-once-per-approach shape as
    skill_pick_onset, generalized for the 5 fixture-skill families."""
    entry = state.setdefault(key, {"streak": 0, "fired": False})
    if near:
        entry["streak"] += 1
    else:
        entry["streak"] = 0
        entry["fired"] = False
    if near and entry["streak"] >= SKILL_ONSET_FRAMES and not entry["fired"]:
        entry["fired"] = True
        return True
    return False


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
                "object_grasped", "object_stable", "object_sync", "object_upright",
                "object_dropped", "object_left_gripper", "object_released",
                "object_supported", "gripper_away_from_object", "object_settled",
                "object_settle_timeout", "release_object_settle_timeout",
                "gripper_is_opening", "gripper_is_closing",
            ],
            "skill_onset": ["skill_pick_onset", "gripper_near_object"],
            "pick_preconditions": ["object_region_clear", "preconditions_satisfied_pick", "pick_precondition_escape"],
            "place_preconditions": [
                "skill_place_onset", "support_region_clear", "support_stable",
                "support_geometry_valid", "preconditions_satisfied_place",
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
                "robot_contact_raw_contaminated", "object_is_rte", "robot_contact_clean", "sanitized",
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
                "fixture_open_retract_resolved", "fixture_close_retract_resolved",
            ],
            "access_enclosure_safety": [
                "fixture_fully_open", "fixture_fully_closed", "reach_in_fixture",
                "gripper_in_fixture", "object_reach_in_fixture", "object_in_fixture",
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
            "prev_offsets": {},
            "prev_gripper_frac": None,
            "settle_watch": None,  # {"object": name, "age": int}
            "forbidden_streak": 0,
            "pick_onset": {},  # name -> {"streak": int, "fired": bool}
            "last_reach_fixture": None,
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
    grasped_name = _check_grasp_any(env)
    object_grasped = grasped_name is not None
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

    lin_delta = float(np.linalg.norm(active_pos - prev_pos)) if (active_pos is not None and prev_pos is not None) else 0.0
    ang_delta = _angle_between_quats(active_quat, prev_quat)
    object_stable = bool(lin_delta < STABLE_LINEAR_DELTA_THRESHOLD and ang_delta < STABLE_ANGULAR_DELTA_THRESHOLD)

    offset = (active_pos - eef_pos) if (active_pos is not None and eef_pos is not None) else None
    prev_offset = state["prev_offsets"].get(active) if active else None
    offset_delta = float(np.linalg.norm(offset - prev_offset)) if (offset is not None and prev_offset is not None) else 0.0
    object_sync = bool(offset is None or offset_delta < SYNC_RELATIVE_DELTA_THRESHOLD)

    object_upright = _upright(active_quat)

    left_gripper = True
    if active:
        try:
            gripper = env.robots[0].gripper
            model = env.get_object(active)
            left_gripper = not bool(env.check_contact(model, gripper.important_geoms.get("left_fingerpad")) or
                                     env.check_contact(model, gripper.important_geoms.get("right_fingerpad")))
        except Exception:
            left_gripper = not object_grasped

    # Real mesh/geom distance, not eef-to-body-origin distance -- see
    # _gripper_far_from_object's own docstring for why (LIBERO objects have
    # real physical extent, same root cause RoboCasa's predicates.py already
    # fixed for its own gripper_away_from_object, 2026-09-08). Falls back to
    # "no active object" (vacuously away) when there's nothing to be near.
    gripper_away = (
        True if active is None else _gripper_far_from_object(env, active, MESH_GRIPPER_FAR_THRESHOLD)
    )

    object_released = bool(object_dropped and (gripper_is_opening or (gripper_frac is not None and gripper_frac < GRIPPER_OPEN_FRACTION_THRESHOLD)))

    object_supported = bool(active and _touches_anything(env, active))
    support_type_matches_object = True  # no support-type taxonomy modeled in v0
    # task_success escape (2026-09-09): LIBERO's demo hdf5s stop recording
    # within a handful of raw frames of the task's own success condition
    # firing -- confirmed corpus-wide (nearly every pick-place task's final
    # release), not a one-off: KITCHEN_SCENE3's moka_pot_1 episode ends only
    # 8 frames after release, body-origin eef distance still ~0.067m and
    # real mesh/geom distance still slightly overlapping (~-0.00003m) at the
    # very last recorded frame, with task.success already True since several
    # frames earlier. Requiring gripper_away before object_settled can ever
    # fire means this (and the large majority of LIBERO's other short
    # release-then-done episodes) can never resolve in time no matter how
    # generous GRIPPER_FAR_THRESHOLD/MESH_GRIPPER_FAR_THRESHOLD/
    # SETTLE_TIMEOUT_FRAMES are set -- the frames needed to observe a real
    # gripper retreat simply were never recorded. Once the object is
    # genuinely supported+stable AND the demonstrated task's own ground-
    # truth success condition holds, the gripper's remaining distance is a
    # recording-length artifact, not a real unresolved safety question --
    # the placement itself is already confirmed correct.
    task_success = bool((dynamic_info.get("task") or {}).get("success"))
    object_settled = bool(
        object_supported
        and support_type_matches_object
        and object_stable
        and (gripper_away or task_success)
    )

    # settle-timeout watchdog: starts on object_dropped, clears on settle or regrasp
    if object_dropped:
        state["settle_watch"] = {"object": active, "age": 0}
    watch = state["settle_watch"]
    release_settle_timeout = False
    if watch is not None:
        if object_grasped and grasped_name == watch["object"]:
            state["settle_watch"] = None
        elif object_settled and active == watch["object"]:
            state["settle_watch"] = None
        else:
            watch["age"] += 1
            release_settle_timeout = watch["age"] > SETTLE_TIMEOUT_FRAMES

    for name in _movable_object_names(env):
        state["prev_positions"][name] = _body_pos(env, name)
        state["prev_quats"][name] = _body_quat(env, name)
        pos = state["prev_positions"][name]
        state["prev_offsets"][name] = (pos - eef_pos) if (pos is not None and eef_pos is not None) else None

    predicates["object_grasped"] = _entry(object_grasped, "gripper bilaterally contacts a movable object", active)
    predicates["object_stable"] = _entry(object_stable, "active object linear/angular motion below threshold")
    predicates["object_stable_relative"] = _entry(object_stable, "aliased to object_stable in v0")
    predicates["object_sync"] = _entry(object_sync, "active object moves in sync with gripper")
    predicates["object_upright"] = _entry(object_upright, "active object z-axis aligned with world z")
    predicates["object_dropped"] = _entry(object_dropped, "a grasp just ended")
    predicates["object_left_gripper"] = _entry(left_gripper, "active object mesh no longer touches gripper")
    predicates["object_released"] = _entry(object_released, "drop coincided with gripper opening")
    predicates["object_supported"] = _entry(object_supported, "active object touches some scene geometry")
    predicates["object_supported_on_correct"] = _entry(object_supported, "aliased to object_supported in v0")
    predicates["gripper_away_from_object"] = _entry(gripper_away, "gripper moved away from active object")
    predicates["object_settled"] = _entry(object_settled, "supported, stable, and gripper away")
    predicates["object_settle_timeout"] = _entry(release_settle_timeout, "aliased to release_object_settle_timeout in v0")
    predicates["release_object_settle_timeout"] = _entry(release_settle_timeout, "dropped object failed to settle within timeout")
    predicates["gripper_is_opening"] = _entry(gripper_is_opening, "gripper closed-fraction decreasing")
    predicates["gripper_is_closing"] = _entry(gripper_is_closing, "gripper closed-fraction increasing")

    # --- skill onset + pick preconditions -------------------------------
    pick_onset_state = state["pick_onset"]
    any_pick_onset = False
    focus_pick_object = active if object_grasped else None
    for name in _movable_object_names(env):
        if name == grasped_name:
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
        if near and entry["streak"] >= SKILL_ONSET_FRAMES and not entry["fired"]:
            entry["fired"] = True
            any_pick_onset = True
            if focus_pick_object is None:
                focus_pick_object = name

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

    focus_pos = _body_pos(env, focus_pick_object) if focus_pick_object else active_pos
    object_region_clear = _region_clear(env, focus_pos, exclude=[focus_pick_object] if focus_pick_object else [])
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
    # pick onset in this corpus.
    focus_pick_stable = _object_stable_by_name(env, state, focus_pick_object) if focus_pick_object else object_stable
    preconditions_satisfied_pick = bool(object_region_clear and focus_pick_stable and object_upright_if_receptacle_default)
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
        pending_stable = _object_stable_by_name(env, state, pick_onset_pending_object)
        pending_region_clear = _region_clear(
            env, _body_pos(env, pick_onset_pending_object), exclude=[pick_onset_pending_object]
        )
        pick_precondition_escape = bool(pending_stable and pending_region_clear)
        if pick_precondition_escape:
            state["pick_onset_pending_object"] = None

    predicates["skill_pick_onset"] = _entry(any_pick_onset, "gripper approached an ungrasped object for the onset window")
    predicates["object_region_clear"] = _entry(object_region_clear, "few foreign objects near the pick/focus object")
    predicates["preconditions_satisfied_pick"] = _entry(preconditions_satisfied_pick, "pick preconditions AND-composition")
    predicates["pick_precondition_escape"] = _entry(pick_precondition_escape, "pending pick object later became stable and region-clear")

    # --- place preconditions --------------------------------------------
    skill_place_onset = object_released
    support_region_clear = _region_clear(env, active_pos, exclude=[active] if active else [])
    support_stable = True  # LIBERO supports are static furniture/table in this v0 -- always stable
    support_geometry_valid = True  # not modeled in v0 -- see module docstring
    preconditions_satisfied_place = bool(support_region_clear and support_stable and support_geometry_valid)

    predicates["skill_place_onset"] = _entry(skill_place_onset, "aliased to object_released")
    predicates["support_region_clear"] = _entry(support_region_clear, "few foreign objects near the release point")
    predicates["support_stable"] = _entry(support_stable, "stubbed True -- static support in v0")
    predicates["support_geometry_valid"] = _entry(support_geometry_valid, "stubbed True -- geometry not modeled in v0")
    predicates["preconditions_satisfied_place"] = _entry(preconditions_satisfied_place, "place preconditions AND-composition")

    # --- contamination -------------------------------------------------
    # Implemented generically (keyword-tag matching against LIBERO's own
    # category names, mirroring monitor/predicates.py's own hardcoded
    # raw/ready_to_eat fallback sets exactly -- see attributes.py) rather
    # than hand-stubbed, so a zero-occurrence result for this corpus is a
    # verified fact, not an assumption: none of the 40 in-scope tasks'
    # objects (soup/sauce/butter/pudding/cream cheese/ketchup/milk/juice/
    # dressing/mugs/bowls/plates/moka pots/wine bottle/book) match
    # RAW_NAME_SUBSTRINGS, confirmed by inspecting the actual object list.
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

    contaminated = bool(state.get("contaminated", False) or (object_grasped and _is_raw(grasped_name)))

    clean_streak = state.get("clean_streak", 0)
    touching_clean = bool(object_grasped and not _is_raw(grasped_name))
    clean_streak = clean_streak + 1 if touching_clean else 0
    state["clean_streak"] = clean_streak
    robot_contact_clean = bool(clean_streak >= CONTAMINATION_PERSISTENCE_FRAMES)

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
    state["contaminated"] = contaminated

    contamination_focus = active or focus_pick_object
    object_is_rte = _is_rte(contamination_focus)

    predicates["robot_contact_raw_contaminated"] = _entry(contaminated, "gripper contacted a raw-tagged object and hasn't been sanitized since")
    predicates["object_is_rte"] = _entry(object_is_rte, "active/focus object category matches the ready-to-eat keyword set")
    predicates["robot_contact_clean"] = _entry(robot_contact_clean, "gripper holding a non-raw object for the persistence window")
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
    target_by_action = {}
    for action in ("press", "turn", "slide", "twist", "open_close"):
        target = _focus_fixture_for_action(env, action, eef_pos)
        target_by_action[action] = target
        target_pos_a = _body_pos(env, target) if target else None
        near = bool(target_pos_a is not None and eef_pos is not None and float(np.linalg.norm(eef_pos - target_pos_a)) < FIXTURE_NEAR_THRESHOLD)
        onset_flags[action] = _generic_fixture_onset(state, f"{action}_onset", near)

    # Shared "target" across all 5 families -- this corpus never has more
    # than one of them non-None for a given task (verified: each task's
    # fixtures_dict only ever contains fixtures relevant to that one task).
    target_name = next((target_by_action[a] for a in ("press", "turn", "slide", "twist", "open_close") if target_by_action[a]), None)
    target_pos = _body_pos(env, target_name) if target_name else None
    # Always explicitly emitted, never left absent: monitor/predicates.py
    # defaults target_region_clear/slide_path_clear/articulation_path_clear
    # to False when missing, so omitting them once the corresponding onset
    # can actually fire would flip that spec from vacuously-satisfied to
    # almost-always-violated -- a regression, not an improvement.
    target_region_clear = _region_clear(env, target_pos, exclude=[])
    # LIBERO fixture root bodies don't translate (only their door/drawer/knob
    # joints articulate) -- root-body position stability holds by
    # construction, not by measurement; documented simplification, not a
    # stub-without-signal.
    target_stable = True
    # Both path-clear predicates reuse the same "nothing foreign nearby"
    # proxy as support_geometry_valid's note -- no swept-corridor geometry
    # modeled in v0. fixture_ready_for_{press,turn,slide,twist,open_close}
    # and target_receptacle_upright_if_has_contents are left absent --
    # monitor/predicates.py already defaults all of them to True.
    slide_path_clear = target_region_clear
    articulation_path_clear = target_region_clear

    preconditions_satisfied_press = bool(target_region_clear and target_stable)
    preconditions_satisfied_turn = bool(target_region_clear and target_stable)
    preconditions_satisfied_slide = bool(target_region_clear and target_stable and slide_path_clear)
    preconditions_satisfied_twist = bool(target_region_clear and target_stable)
    preconditions_satisfied_open_close = bool(target_region_clear and target_stable and articulation_path_clear)

    predicates["skill_press_onset"] = _entry(onset_flags["press"], "gripper approached a press-tagged fixture for the onset window")
    predicates["skill_turn_onset"] = _entry(onset_flags["turn"], "gripper approached a turn-tagged (faucet) fixture for the onset window")
    predicates["skill_slide_onset"] = _entry(onset_flags["slide"], "gripper approached a slide-tagged (drawer) fixture for the onset window")
    predicates["skill_twist_onset"] = _entry(onset_flags["twist"], "gripper approached a twist-tagged (knob) fixture for the onset window")
    predicates["skill_open_close_onset"] = _entry(onset_flags["open_close"], "gripper approached an open_close-tagged (door) fixture for the onset window")
    predicates["target_region_clear"] = _entry(target_region_clear, "few foreign objects near the press/turn/slide/twist/open_close target")
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

    if fixture_open_obstacle_hit:
        state["open_retract_watch"] = {"age": 0}
    open_watch = state.get("open_retract_watch")
    fixture_open_retracting = bool(open_watch is not None and not fixture_is_opening)
    fixture_open_retract_resolved = False
    if open_watch is not None:
        if fixture_fully_closed_early:
            state["open_retract_watch"] = None
            fixture_open_retract_resolved = True
        elif fixture_open_retracting:
            open_watch["age"] += 1
            fixture_open_retract_resolved = open_watch["age"] > FIXTURE_RETRACT_RESOLVE_TIMEOUT_FRAMES
        else:
            open_watch["age"] = 0

    if fixture_close_obstacle_hit:
        state["close_retract_watch"] = {"age": 0}
    close_watch = state.get("close_retract_watch")
    fixture_close_retracting = bool(close_watch is not None and not fixture_is_closing)
    fixture_close_retract_resolved = False
    if close_watch is not None:
        if fixture_fully_open_early:
            state["close_retract_watch"] = None
            fixture_close_retract_resolved = True
        elif fixture_close_retracting:
            close_watch["age"] += 1
            fixture_close_retract_resolved = close_watch["age"] > FIXTURE_RETRACT_RESOLVE_TIMEOUT_FRAMES
        else:
            close_watch["age"] = 0

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
    predicates["fixture_open_retract_resolved"] = _entry(fixture_open_retract_resolved, "reached fully closed, or retracting held past resolve timeout")
    predicates["fixture_close_retract_resolved"] = _entry(fixture_close_retract_resolved, "reached fully open, or retracting held past resolve timeout")
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

    predicates["fixture_fully_open"] = _entry(fixture_fully_open, "focus fixture reports is_open()")
    predicates["fixture_fully_closed"] = _entry(fixture_fully_closed, "focus fixture reports is_close()")
    predicates["reach_in_fixture"] = _entry(reach_in_fixture, "gripper newly within interior radius of focus fixture")
    predicates["gripper_in_fixture"] = _entry(gripper_in_fixture, "gripper within interior radius of focus fixture")
    predicates["object_reach_in_fixture"] = _entry(object_reach_in_fixture, "active object newly contained in focus fixture")
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
