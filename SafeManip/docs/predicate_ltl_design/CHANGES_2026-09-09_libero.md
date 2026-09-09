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

# Part 2 (same day, continued): closing the gap to/below RoboCasa's own per-property rates

Follow-up goal: every (task, LTL property) pair was already below 6/10, but a
few properties still had a *higher corpus-wide rate* than the same property's
rate on RoboCasa's own corpus (v21) -- expected to be similar or lower, since
LIBERO has no raw/RTE food objects (`rc_raw_robot_contact_blocks_rte_grasp_
until_sanitized` should be ~0%) and a much smaller, cleaner scene (`rc_no_
forbidden_contact` should be low too) -- both true already; `rc_dropped_
object_was_released` (0.6% RoboCasa), `rc_grasp_remains_synced_until_dropped`
(1.8%), and `rc_released_object_eventually_settles` (0.2%) were still higher
on LIBERO (128 -> 68 -> 28 -> 15 -> 9 -> 4 total violations across this
session's iterations).

## 6. `object_released`: two more branches ported from RoboCasa, verbatim

`prev_gripper_is_opening` (closes a one-frame sign-check dip in `gripper_is_
opening` itself) and `active is not None and object_supported` (a release
where the object is already resting on solid support by the time contact
breaks, without the gripper necessarily opening past threshold first) --
confirmed on `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`:
`object_supported` was already `True` several frames *before* the grasp
ended (the pot touches the stove before release), and `gripper_frac` never
dipped below threshold at the drop frame at all (this demo's release motion
doesn't fully open the gripper immediately) -- a real, deliberate,
already-safe placement the original gripper-only check couldn't recognize.

## 7. `object_sync`: RoboCasa's grasp-slip-baseline pattern, ported

Previously compared the current (obj - eef) offset against `state["prev_
offsets"][active]`, updated for *every* movable object on *every* frame
regardless of grasp state -- on the very first frame of a brand-new grasp,
that "previous" offset reflected wherever the object was sitting *before*
ever being grasped (e.g. the gripper mid-approach from elsewhere), so the eef
finally reaching the object at grasp onset read as a huge, spurious "slip".
New `sync_baseline_object`/`sync_baseline_offset` in `state`, re-seeded fresh
on every NEW grasp *event* (`object_grasped` just became `True` -- not just
"the active object's name changed", which misses a same-object drop-then-
regrasp: `active` never clears on drop, only reassigns on the *next* actual
grasp), refreshed every frame thereafter (per-frame drift, not accumulated-
since-onset, matching RoboCasa's own `_object_grasp_slip` rationale).
Confirmed on two real cases: `KITCHEN_SCENE3` ep2 (brand-new grasp) and
`KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_...` ep0 (drop-then-
regrasp of the same bowl).

## 8. `SETTLE_TIMEOUT_FRAMES`: 60 -> 100

Simply too short -- confirmed directly: `put_the_wine_bottle_on_the_rack`
ep0 genuinely settles (supported+stable+gripper-away) ~80 raw frames after
release, timing out at 60 with the object already correctly at rest by 100.
RoboCasa's own value is 100; both engines call this once per raw ~20Hz
frame (confirmed via each engine's own `control_freq` default), so this is
now a straight match, not a guess.

## 9. `object_settled`'s task_success escape: widened to cover object_stable too

Was `object_supported and support_type_matches_object and object_stable and
(gripper_away or task_success)` -- only `gripper_away` had an escape.
Confirmed on `STUDY_SCENE1_pick_up_the_book_...`: `object_stable`, not
`gripper_away`, was the one still pending when the ~5-frame-remaining
episode ended (`task.success` already `True`, `object_supported` already
`True`) -- the same recording-truncation artifact, just landing on a
different conjunct. Now `object_supported and support_type_matches_object
and (task_success or (object_stable and gripper_away))`.

## 10. `_check_grasp_any`: AND bilateral contact with gripper-closed-enough

Mirrors RoboCasa's own `_object_is_grasped` (bilateral contact AND `OU.
check_obj_grasped`) -- a second, independent raw signal (`_gripper_closed_
fraction(env) >= GRIPPER_OPEN_FRACTION_THRESHOLD`) alongside robosuite's own
`_check_grasp`, so a one-frame bilateral-contact-only dropout (gripper still
genuinely closed, object frozen) no longer registers as a real grasp-loss.
Fixed the majority (17/22) of a corpus-wide one-frame-flicker pattern in
`rc_dropped_object_was_released` at the raw-signal level -- not a debounce.

## 11. `object_left_gripper`: AABB overlap, not mesh distance (explicit user correction)

First attempt used real mesh/geom distance (`_gripper_object_geom_min_
distance`, already built for `gripper_away_from_object`) with `left_gripper
= mesh_dist > 0.0`. User correction: RoboCasa's *actual* mechanism for this
exact predicate is bounding-box overlap (`_aabb_intersects`/`_object_contact_
aabb`/`_gripper_aabb`), deliberately coarser than exact mesh distance -- a
real one-frame position micro-jitter can put a hair of daylight between two
meshes while the object is still clearly within the gripper's enclosing
volume; AABB overlap tolerates that, exact mesh distance doesn't. Ported
`_geom_aabb`/`_geom_ids_aabb`/`_aabb_intersects` verbatim from RoboCasa's own
predicates.py. Combined with #10, this resolved the residual one-frame-
flicker cases *without* an intermediate debounce that had been added and
then explicitly removed per user direction ("if robocasa don't use debounce
to solve a problem, don't use debounce for libero... get rid of the
smoothing as much as possible unless necessary") -- confirmed by re-running
without it once the AABB fix alone proved sufficient.

## 12. Threshold alignment with RoboCasa's own tuned constants

Explicit user request: make the two simulators' hyperparameters the same
wherever genuinely comparable (both confirmed `control_freq=20` by default,
so frame-counts carry over 1:1; RoboCasa's velocity-based stability
thresholds converted to LIBERO's per-frame-delta convention via `* dt`,
`dt = 1/20s`). Left `GRIPPER_OPEN_FRACTION_THRESHOLD`/`GRIPPER_CLOSED_
THRESHOLD` unaligned (genuinely different quantities -- a normalized 0-1
fraction vs. a raw joint qpos value specific to RoboCasa's own gripper
model, no valid conversion). `MESH_GRIPPER_FAR_THRESHOLD` 0.02->0.01,
`SYNC_RELATIVE_DELTA_THRESHOLD` 0.01->0.03, `STABLE_LINEAR_DELTA_THRESHOLD`
0.004->0.0025, `STABLE_ANGULAR_DELTA_THRESHOLD` 0.05->0.0125, `SKILL_ONSET_
FRAMES` 5->8, `FORBIDDEN_CONTACT_TOLERANCE_FRAMES` 10->20, `FIXTURE_RETRACT_
RESOLVE_TIMEOUT_FRAMES` 60->100.

## Final result

9 -> 4 total corpus violations after item 12 (`rc_grasp_remains_synced_
until_dropped` and `rc_pick_preconditions_safe` both dropped to 0%, some
prior violations were borderline cases the widened stability/sync
thresholds correctly absorb). Every property now at or below RoboCasa's own
rate except `rc_released_object_eventually_settles` (0.8% vs 0.2%, 3/400
episodes) -- all three confirmed genuine task failures (`task.success ==
False` at episode end: the object was never actually placed correctly, or
never settled because the demo itself failed), not predicate bugs.

## Viewer fixes found along the way

- **Per-episode violation-count badge missing for LIBERO** (`viewer/server.
  py`'s `list_libero_training_episodes`): returned flat `success`/`num_
  violations` fields, but the shared frontend rendering code
  (`loadTrainingEpisodes` in `app.js`) reads `ep.methods[currentMethod]`/
  `(ep.annotated || {})[currentMethod]` regardless of sim -- LIBERO episodes
  showed only the success badge, never a violation-count badge, and always
  read as "not annotated". Fixed by nesting under `methods`/`annotated`
  dicts keyed by method, matching RoboCasa's `list_training_episodes` shape
  exactly; also threaded `method=` through the frontend's `/api/td_episodes`
  fetch (previously always silently used LIBERO's own default method
  regardless of the dropdown's actual selection).
- **Stale running viewer process**: the running server had been serving
  pre-LIBERO-support code for several minutes after the "Add LIBERO
  simulator support" commit landed on disk (never restarted) -- explained a
  separate report ("clicking a task under an LTL property switches to the
  RoboCasa panel"). Restarted it.
