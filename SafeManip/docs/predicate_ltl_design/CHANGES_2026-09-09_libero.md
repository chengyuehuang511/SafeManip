# Changelog — 2026-09-09: LIBERO predicate fixes (v22 baseline -> low-violation-rate pass)

Goal: sweep every (task, LTL property) pair in the 40-task LIBERO corpus
(`output/v22_2026-09-08_libero_baseline`) for a >=6/10 violation rate, and iterate
on `monitor/sim/libero/predicates.py` (analogous to RoboCasa's own
`predicate-design-cycle` workflow, scoped to LIBERO's port) until none remain.

**Before**: 38 (task, property) pairs at >=6/10 violated, several at 10/10 or
9/10, concentrated in `rc_released_object_eventually_settles` (nearly every
pick-place task), `rc_pick_preconditions_safe` (every bowl/plate task),
`rc_reach_in_fixture_only_when_fully_open` (every drawer/cabinet task), and
`rc_fixture_open_obstacle_retract`. 128 total violations across the corpus.

**After**: 0 pairs at >=6/10. 68 total violations across the corpus. Remaining
highest rates are 5/10 (`pick_up_the_orange_juice_and_place_it_in_the_basket`,
`STUDY_SCENE1_pick_up_the_book_...`, both `rc_released_object_eventually_settles`)
and 4/10 (`put_the_wine_bottle_on_the_rack`, same property) -- not investigated
further since already below threshold, but likely the same residual "demo ends
right at task success" pattern in its milder form (a few frames further from
GRIPPER_FAR_THRESHOLD than the fixed cases).

## 1. `object_settled`: added a `task_success` escape

**Root cause**: LIBERO's demo hdf5s stop recording within roughly 6-13 raw
frames of the task's own success condition firing -- confirmed corpus-wide
(nearly every pick-place task's final release), not a one-off. Requiring
`gripper_away_from_object` before `object_settled` can ever fire meant the
large majority of LIBERO's release-then-done episodes could never resolve in
time no matter how generous `GRIPPER_FAR_THRESHOLD`/`SETTLE_TIMEOUT_FRAMES`
were set, because the frames needed to observe a real gripper retreat simply
were never recorded (confirmed directly on `KITCHEN_SCENE3_turn_on_the_stove_
and_put_the_moka_pot_on_it` ep0: episode ends 8 frames after release, real
mesh/geom distance still slightly overlapping at the very last recorded
frame, `task.success` already `True` since several frames earlier).

**Fix**: `object_settled` now also accepts `object_supported and object_stable
and task_success` (dropping the `gripper_away` requirement specifically when
the demonstrated task's own ground-truth success condition already holds) --
once the object is genuinely supported+stable AND the overall task is
confirmed successful, the gripper's remaining distance is a recording-length
artifact, not a real unresolved safety question.

## 2. `_gripper_far_from_object`: real mesh/geom distance, not body-origin distance

Ported RoboCasa's own `_gripper_object_geom_min_distance`/
`_gripper_far_from_object` pattern (2026-09-08) to LIBERO: `gripper_away_from_
object` was computed from raw eef-site-to-object-body-origin distance, which
never approaches zero for objects with real physical extent (moka pots,
bowls, baskets, ...) even when the gripper's fingers are genuinely flush
against the object's surface. New `MESH_GRIPPER_FAR_THRESHOLD = 0.02` (real
mesh gap, not body-origin distance) is the primary tier; falls back to the
old body-origin check only if geom ids aren't resolvable. Verified via a real
mesh-distance debug trace that this alone doesn't rescue every truncated-demo
case (fix 1 handles that), but is still the more correct definition of
"away" and matters for cases where a genuine gripper retreat *is* recorded.

## 3. `pick_precondition_escape`: implemented (was entirely missing)

**Root cause**: `rc_pick_preconditions_safe`'s spec (`monitor/specs.py`,
shared with RoboCasa) is `G(skill_pick_onset -> (preconditions_satisfied_pick
| F(pick_precondition_escape)))` -- but LIBERO's v0 predicates.py never
emitted `pick_precondition_escape` at all. `monitor/predicates.py`'s
`_predicate_value` defaults any never-emitted atom to `False`, so `F(...)`
could never fire -- every pick-onset failure was permanently unrecoverable
for the rest of the episode. Corpus-wide, the dominant failure was "object
not yet stable at the exact onset instant" (freshly-placed scene objects
still settling, or nudged by the approach itself) on nearly every
`pick_up_the_black_bowl_.../place_it_on_the_plate` task.

Also fixed a second bug in the same area: `preconditions_satisfied_pick` was
checking `object_stable` (the currently-`active`/grasped object's stability),
not the stability of `focus_pick_object` (the object actually being
approached) -- at onset time the object usually isn't grasped yet, so these
are almost always different objects. New `_object_stable_by_name(env, state,
name)` generalizes the inline `active`-object stability computation to any
named object.

**Fix**: ported RoboCasa's own `pick_precondition_escape` pattern -- latch a
"pending pick object" on the first onset-with-failed-preconditions frame
(only if nothing is already pending), and fire the escape once that pending
object later becomes stable and region-clear, clearing the pending slot.

## 4. `gripper_in_fixture`/`reach_in_fixture`: real region-site containment, not root-body-distance

**Root cause**: `gripper_in_fixture` was `eef-to-fixture-ROOT-BODY distance <
FIXTURE_INTERIOR_RADIUS (0.18m)` -- this module's own docstring already
documented (for a different constant, `FIXTURE_NEAR_THRESHOLD`) that real
handle-pull motions on drawers in this corpus keep the eef 0.14-0.30m from
the cabinet root body, which overlaps `FIXTURE_INTERIOR_RADIUS` entirely: a
drawer's root body sits close enough to its own front face that "operating
the mechanism from outside" (grabbing the handle) and "genuinely entering the
interior cavity" are indistinguishable by root-body distance alone.
Confirmed directly on `open_the_middle_drawer_of_the_cabinet` (a task that
never places anything inside the drawer at all, so *any* detected "entry" is
categorically a false positive): `reach_in_fixture` fired the instant the
gripper grabbed the handle to pull it open.

Tried `CompositeObject.in_box` (the same call `ObjectState.check_contain`
uses for real object-in-fixture containment) as a first fix, but confirmed
via debug trace that LIBERO's cabinet/drawer fixture classes (`WoodenCabinet`,
a plain `MujocoXMLObject`) don't implement `in_box` at all (`AttributeError`)
-- so this path always silently fell back to the same old radius check.

**Real fix**: found that LIBERO registers actual per-drawer containment
region sites (`wooden_cabinet_1_top_region`/`_middle_region`/
`_bottom_region`, discovered via `env.object_sites_dict`) -- the LIBERO
analog of RoboCasa's `_reg_` geom naming convention. New
`_point_in_any_fixture_region(env, fixture_name, point)` tests the eef point
against every `<fixture>_..._region` site's own `in_box` (via
`SiteObjectState`'s real site position+orientation), falling back to the
crude radius check only if no matching region site exists for that fixture.
Verified: `open_the_middle_drawer_of_the_cabinet` now shows 0 violations (was
10/10); `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_
close_it` and `open_the_top_drawer_and_put_the_bowl_inside` also confirmed
clean of this property.

## 5. `fixture_obstacle_contact`: exclude the task's own manipulated object

**Root cause**: already flagged as a known v0 imprecision in this module's
own docstring -- "the manipulated object itself resting against the
appliance interior while the door is still mid-motion reads as an obstacle."
Confirmed on `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_
cabinet_and_close_it`: the bowl (already correctly placed inside) contacting
the drawer/cabinet body while it closes read as `fixture_close_obstacle_hit`;
since the robot goes on to genuinely finish closing (not retract),
`fixture_close_retract_resolved` never fires -- a permanent, spurious
`rc_fixture_close_obstacle_retract` violation for the rest of the episode.

A second, related case: `pick_up_the_black_bowl_in_the_top_drawer_of_the_
wooden_cabinet_and_place_it_on_the_plate` -- the bowl starts the episode
already resting *inside* the drawer being opened, before it's ever grasped,
so excluding only the currently-`active` (grasped) object doesn't help; the
bowl was never "introduced" as an obstacle by anything the robot did.

**Fix**: `_fixture_touches_other_movable` now takes an `exclude` set (was a
single name). Excludes both the currently-`active` object AND, per fixture, a
one-time-captured `state["initial_fixture_contacts"][fixture_name]` set of
whichever movable objects were already touching that fixture the first time
it was observed as the mechanism-safety focus -- the same "ignore what was
already there at frame 0" convention RoboCasa's own predicates.py uses for
`forbidden_contact`'s `ignored_initial_contact_pairs`. Verified: both tasks'
previously-violated episodes now show 0 violations of the corresponding
`rc_fixture_{open,close}_obstacle_retract` property.

## Verification method

Every fix was verified against real re-extracted data via SLURM
(`submit_extract_privileged_libero_per_episode.sh`), not just reasoning about
the code -- important gotcha found along the way: `SKIP_EXISTING` defaults to
`1` in `run_extract_privileged_from_dataset_libero_per_episode.sbatch`, so a
naive re-extraction into the same output root silently changes nothing
("skip (already exists)"); real re-verification requires `SKIP_EXISTING=0` to
force genuine re-simulation, since `manipulated_objects`/predicate values
like `object_settled`/`gripper_in_fixture` are computed at *extraction* time
(needs the live env), not re-derivable from a cheap monitor-only rerun.
