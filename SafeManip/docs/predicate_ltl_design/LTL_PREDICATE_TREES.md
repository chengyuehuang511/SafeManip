# Predicate/LTL Design — Grounded Predicate Trees (all 19 properties)

This document traces all 19 top-level LTL safety properties defined across the docs in this
folder down to their leaf-level implementation in
`SafeManip/monitor/sim/robocasa/predicates.py` (the canonical, currently-running
implementation — the `.txt` docs sometimes lag behind it; discrepancies are flagged inline
with `⚠`).

Sources:
- `collision_grasp_release_contamination_safety.txt`
- `containment_safety.txt`
- `mechanism_safety.txt`
- `access_enclosure_safety.txt`
- `categorization.txt`
- `action_onset_safety.txt`
- `KNOWN_BUGS.md` (for previously-confirmed doc/code mismatches)

All line numbers below refer to `SafeManip/monitor/sim/robocasa/predicates.py` unless noted.
`specs.py`/`primitives.py` are cited where relevant for LTL wiring or shared constants.

Generated 2026-08-31 by tracing the code directly (not from memory) — re-verify line numbers
if this file is edited again without updating this doc.

⚠ 2026-09-01: substantial edits landed in `predicates.py` this session (bilateral grasp contact,
object_sync/object_grasped independence, object_stable_relative, object_released changes, and
removal of `CONTACT_PERSISTENCE_FRAMES` / `OBJECT_GRASPED_PERSISTENCE_FRAMES` /
`RELATIVE_SPEED_PERSISTENCE_FRAMES` / `GRASP_SAFE_GRACE_FRAMES`). Citations directly tied to
those changes were re-verified and updated; citations elsewhere in this file (properties not
touched this session) were NOT individually re-verified against the cumulative line-number
drift from all of today's edits and may be off by a similar amount. Treat unverified line
numbers as approximate until a fresh regeneration pass.

Naming convention: tree *node labels* use the predicate names the `.txt` docs already define
(`correct_manipulated_object_correct_fixture_contact`, `target_fixtures_by_object`,
`original_supports_by_object`, etc.) wherever one exists. Code-internal variable/function names
(`robot_policy_geom_ids`, `_object_gripper_bilateral_contact`, `active_fixture_contact_name`,
...) appear only inside `[...]` grounding annotations and leaf-level `leaf: ...` lines, to cite
*how* a doc-named predicate is actually computed — never as a stand-in node label for a concept
the docs already name. Where the code introduces a genuinely undocumented concept (no doc name
exists at all), it's described in plain English rather than given an invented predicate-sounding
name. See `LTL_PREDICATE_TREES_CONCISE.md` for a version with all grounding/line-citation detail
stripped, doc vocabulary only.

---

# Part A — Core safety properties (11)

## Property 1: `G(!forbidden_contact)`

```
G(!forbidden_contact)
└── forbidden_contact                                                     [predicates.py:2469-2471, snapshot key 6131]
    ├── considered_contact_pairs                                          [predicates.py:2352-2363]
    │   ├── robot contacts non-robot geom
    │   │   └── (geom1 ∈ robot_policy_geom_ids ∧ geom2 ∉ robot_geom_ids) ∨ symmetric   [predicates.py:2344-2346]
    │   │       ├── leaf: robot_policy_geom_ids = robot_geom_ids − robot_base_geom_ids  [predicates.py:2033-2035]
    │   │       │   ├── robot_geom_ids: geoms of robot bodies (dynamic_info["robot"]["link_poses"], else env.robots[0].robot_model.root)  [predicates.py:321-351]
    │   │       │   └── robot_base_geom_ids: geoms whose body name contains "mobilebase"/"pedestal"/"robot base"  [predicates.py:353-391]
    │   │       └── ⚠ discrepancy: doc's "robot" side is implicitly the whole robot; code splits robot_policy_geom_ids (excludes mobile base) as the "robot" side of considered/allowed contact, but uses the *full* robot_geom_ids as the "non-robot" exclusion test — an internal asymmetry not documented.
    │   └── grasped object contacts non-robot geom
    │       └── grasped_object_exists ∧ (geom1 ∈ grasped_object_geom_ids ∧ geom2 ∉ robot_geom_ids) ∨ symmetric  [predicates.py:2347-2351]
    │           ├── grasped_object_exists := active_object is not None ∧ object_grasped   [predicates.py:2347; object_grasped fully expanded in Property 2 below]
    │           └── grasped_object_geom_ids = _object_geom_ids(active_object)  [predicates.py:423-430, 2240-2242] — leaf: body/descendant geoms of env.obj_body_id + env.objects[name].contact_geoms
    │   ⚠ undocumented grace: contact pairs present at episode start are recorded in `ignored_initial_contact_pairs` and excluded from evaluation until absent for one frame (no debounce; `CONTACT_PERSISTENCE_FRAMES` removed 2026-09-01 — see cross-cutting note). Also any pair touching robot_base_geom_ids is unconditionally skipped. Neither is in the doc.
    ├── NOT allowed_contact                                                [predicates.py:2473-2478, snapshot key 6132]
    │   ├── robot_correct_manipulated_object_contact
    │   │   └── leaf: _pair_matches(geom1, geom2, robot_policy_geom_ids, {manipulated-object geoms})  [predicates.py:1090-1093, 2365-2367, 2423]
    │   │       └── manipulated-object geoms = ∪ geoms of ALL manipulated_object_names (not just the single grasped object)  [predicates.py:2237-2239] ⚠ broader set than doc's "manipulated-object geom" implies
    │   ├── robot_correct_fixture_contact
    │   │   └── leaf: _pair_matches(geom1, geom2, robot_policy_geom_ids, {target-fixture-or-action-component geoms})  [predicates.py:2368-2370, 2424]
    │   │       └── target-fixture-or-action-component geoms := geoms(target_fixtures_by_object, task-level union) ∪ geoms(target_fixtures_by_object, active_object) ∪ {action component geoms}  [predicates.py:2258-2262]
    │   │           ├── target_fixtures_by_object (task-level union) := union of success_fixture_targets from _success_target_relations()  [predicates.py:834-839]
    │   │           ├── target_fixtures_by_object (active_object-specific) := geoms of target fixtures for the active/manipulated object  [predicates.py:880-890, 2221-2224]
    │   │           └── action component geoms := geoms of the doc's "action component geom, such as a handle, button, lever, knob, rack, lid, or door" clause — handle/button/lever/knob/rack/lid/door keyword match on task-referenced fixtures  [predicates.py:967-1034]
    │   │               ├── leaf: _task_ref_names() — scene_layout/episode_meta fixture refs, env attribute fixture refs, init_robot_base_ref  [predicates.py:792-832]
    │   │               ├── leaf: _contact_policy_fixture_attrs — class/name substring heuristics (drawer→openable, button/coffee/microwave/kettle→pressable, faucet/sink→turnable, knob/dial/stove/oven/toaster→twistable, slide/rack/dishwasher→slideable, door/cabinet/fridge/lid/standmixer→openable/closeable)  [predicates.py:922-966]
    │   │               └── leaf: ACTION_COMPONENT_KEYWORDS geom/body name match per action  [predicates.py:980-1033]
    │   │       ⚠ doc phrases this as an OR of two contact *types* (target-fixture vs. action-component); code instead pre-merges both into one geom-id set compared against the robot in a single test.
    │   ├── correct_manipulated_object_correct_fixture_contact  (gated: grasped_object_exists)
    │   │   └── contact with active_object's own entry in target_fixtures_by_object, OR with a task-referenced *actionable* fixture   [predicates.py:2371-2392, 2425]
    │   │       ├── target_fixtures_by_object(active_object) match := neither side robot ∧ _pair_matches(grasped_object_geom_ids, {target_fixtures_by_object(active_object) geoms})  [predicates.py:2371-2378]
    │   │       └── task-referenced-actionable-fixture match := _matched_fixture_name_for_pair(...) against {geoms of task-referenced fixtures that are actionable} is not None  [predicates.py:1044-1053, 2379-2391]
    │   │           └── leaf set := geoms-by-name for (target_fixtures_by_object (task-level union) ∪ {action fixtures with any nonempty _contact_policy_fixture_actions})  [predicates.py:2247-2257] ⚠ requires the fixture be *actionable*, not merely any task-referenced fixture as doc's plain phrasing ("a task-referenced fixture geom, such as a coffee machine receiver surface") suggests
    │   ├── correct_manipulated_object_correct_receive_object_contact  (gated: grasped_object_exists)
    │   │   └── leaf: neither side robot ∧ _pair_matches(grasped_object_geom_ids, {receive_objects_by_object(active_object) geoms})  [predicates.py:2395-2402, 2426-2428]
    │   │       └── receive_objects_by_object(active_object) geoms := geoms of target_objects_by_object.get(active_object)  [predicates.py:856-878, 2225-2228, 2271-2273]
    │   └── correct_manipulated_object_original_support_contact  (gated: grasped_object_exists)
    │       └── neither side robot ∧ contact with a fixture or object in original_supports_by_object(active_object)  [predicates.py:2403-2421, 2429]
    │           └── original_supports_by_object(active_object): captured once at grasp onset via `_current_support_contacts(active_object)`  [predicates.py:1191-1225, 2166-2179, 2229-2230, 2265-2276]
    │               └── leaf primitives: OU.check_obj_fixture_contact, OU.obj_inside_of(partial_check=True), _fixture_rack_contact, OU.check_obj_in_receptacle, env.check_contact  [predicates.py:1191-1225]
    └── no debounce: fires the same frame the condition holds  [predicates.py:2428; `CONTACT_PERSISTENCE_FRAMES` removed 2026-09-01, was already a no-op at value 1 — see cross-cutting note]
```

## Property 2: `G(object_grasped -> object_grasped_safe U object_released)`

```
G(object_grasped -> object_grasped_safe U object_released)
├── object_grasped                                                        [predicates.py:2125-2159, snapshot 6138]
│   ├── raw grasp candidate: any manipulated object name with _object_is_grasped(name) true  [predicates.py:2061-2062]
│   │   └── _object_is_grasped(name)                                      [predicates.py:538-562]
│   │       ├── gripper_bilateral_contact(name) := _object_gripper_bilateral_contact(name)  [predicates.py:505-536]
│   │       │   ├── groups gripper contact geoms by parent MuJoCo body id  [predicates.py:466-484]
│   │       │   ├── if < 2 distinct finger bodies found → fallback to _object_gripper_contact_any(name) (any gripper geom vs. any object geom via MuJoCo contact scan)  [predicates.py:486-503, 519-520]
│   │       │   └── leaf: distinct finger-body contact count >= GRASP_BILATERAL_MIN_CONTACT_BODIES (=2)  [predicates.py:30, 536]
│   │       └── OU.check_obj_grasped(env, obj_name=name, threshold=GRIPPER_CLOSED_THRESHOLD=0.0399)  [predicates.py:24, 555-557] — leaf: external RoboCasa primitive; ANDs aggregate contact + gripper-closed joint check (tightened, not replaced, by the bilateral term)
│   │       ⚠ 0.035 -> 0.04 -> 0.0395 -> 0.0399 (2026-09-02): 0.035 caused a confirmed total detection blind spot for `bread` (ArrangeBreadBasket ep0) -- holding it props the gripper open to ~0.036-0.037, just over 0.035, even though bilateral contact was satisfied fine. Set to 0.04 (the joint's physical fully-open limit), then pulled in to 0.0395 for a small safety margin -- but `bread` in a *different* episode (ep3) props the gripper open even wider, ~0.0395-0.0397, so 0.0395 excluded it too (confirmed: `raw_grasped_objects` empty from frame 369 on, even though bilateral contact was fine -- the joint check alone was failing). Bumped to 0.0399, leaving only ~0.0001 margin below the exact observed physical max (0.04004). See CHANGES_2026-08-31.md item 7 for the full writeup, including why the *original* threshold (0.06) had been a no-op all along.
│   │       ⚠ REVERTED (2026-08-31): object_sync was briefly ANDed in here too (to distinguish "touch" from "grasp" once there's relative motion), then removed again. Folding object_sync into object_grasped's own definition made object_grasped_safe (= object_grasped and object_sync) a logical tautology — since object_grasped would then already imply object_sync, object_grasped_safe would be identically equal to object_grasped on every frame, giving zero extra information and making the G(object_grasped -> object_grasped_safe U object_released) property permanently non-triggerable via this path. object_grasped is now bilateral contact + closed only; object_sync lives solely in object_grasped_safe below, kept deliberately independent.
│   ├── carrier substitution (undocumented): if the raw grasp candidate sits inside a receptacle-like manipulated object, the *receptacle* becomes grasp_candidate instead  [predicates.py:2083-2097, 2123]
│   │   └── leaf: OU.check_obj_in_receptacle(env, name, carrier_name)  [predicates.py:2093]
│   └── no debounce: object_grasped tracks grasp_candidate directly on both edges  [predicates.py:2113]  [`OBJECT_GRASPED_PERSISTENCE_FRAMES` removed 2026-09-01, see cross-cutting note -- previously required 2 consecutive frames each way, to absorb flicker from the aggregate-contact grasp check, which is now fixed at the raw-signal level via bilateral contact instead]
├── object_grasped_safe                                                    [predicates.py:2457]
│   ├── object_sync := _object_sync(name)                                  [predicates.py:1996-2022, disclaimer: not re-verified line-by-line below, see 2026-09-02 note]
│   │   ├── leaf, preferred: _object_grasp_slip(name) — frame-to-frame position/orientation drift
│   │   │       from rigid attachment to the eef  [predicates.py:1929-1994]
│   │   │   ⚠ ADDED (2026-09-02), supersedes the velocity-based leaves below as the primary check:
│   │   │       compares the object's actual pose this frame against where it *should* be if it had
│   │   │       moved perfectly rigidly with the eef since *last* frame (not since grasp onset —
│   │   │       see the "frame-to-frame, not accumulated" note below), using the relative pose
│   │   │       recorded then as the reference (monitor_state["grasp_slip_rel_offset"]/
│   │   │       ["grasp_slip_rel_quat"], seeded at grasp onset — same edge as
│   │   │       source_support_fixtures/objects below — and overwritten every frame thereafter).
│   │   │       Compared against GRASP_SLIP_LINEAR_THRESHOLD (=0.03) /
│   │   │       GRASP_SLIP_ANGULAR_THRESHOLD (=0.3).
│   │   │   ⚠ orientation reading fixed (2026-09-02): _object_orientation/_eef_orientation read
│   │   │       "orientation" as wxyz, but kitchen_ext.py's _pose_from_body_id/_pose_from_site_id
│   │   │       both store it in xyzw order (T.convert_quat(to="xyzw"); Rotation.as_quat()'s
│   │   │       default) — added _xyzw_to_wxyz to correct this. Confirmed via the object's/eef's
│   │   │       own absolute-rotation-axis alignment (should be parallel for a rigid attachment):
│   │   │       dot product was ~-0.82 to -0.97 (anti-parallel, wrong) before the fix, ~+0.996 to
│   │   │       +0.997 (parallel, correct) after. ArrangeBreadBasket ep0: bread's apparent ~0.87 rad
│   │   │       (~50°) relative rotation (362→486) dropped to ~0.036 rad (~2°, matching a direct
│   │   │       "no visible rotation" observation); basket's apparent ~2.88 rad (~165°, 564→791)
│   │   │       dropped to ~0.34-0.35 rad (~19-20°) — real, but an order of magnitude smaller.
│   │   │   ⚠ frame-to-frame, not accumulated-since-onset (2026-09-02): originally compared against
│   │   │       a single baseline captured once at grasp onset (kept forever), which meant a
│   │   │       one-time settling shift larger than threshold flagged every subsequent frame for
│   │   │       the rest of the grasp, even after the object stabilized (the baseline never
│   │   │       updates, so the residual is a fixed, non-decaying offset). Fixed by overwriting the
│   │   │       stored reference to the *current* frame's actual relative pose at the end of every
│   │   │       call, regardless of whether slip exceeded threshold — a one-time shift is flagged
│   │   │       once and stops being flagged once stable; ongoing slip keeps getting flagged.
│   │   │       Trade-off (confirmed on real data, not fully resolved): at call_stride=16, this is
│   │   │       close to a coarse average-velocity check again — immune to brief transients and to
│   │   │       the stale-baseline problem, but blind to slow continuous drift that never spikes in
│   │   │       any single frame. ArrangeBreadBasket ep0's basket swing (~19-20° over ~150 frames,
│   │   │       mean per-frame delta ~0.0053 rad, max ~0.059 rad — both far under the 0.3 threshold
│   │   │       every single frame) is now completely undetected (0/19 violated, was 1/19 with the
│   │   │       accumulated-since-onset + quaternion-fixed version). None of the three approaches
│   │   │       tried this session (velocity / accumulated-since-onset / frame-to-frame) satisfies
│   │   │       all three properties (immune to transients, forgets settles, catches slow drift)
│   │   │       at once — open design question, see CHANGES_2026-08-31.md items 13-15.
│   │   ├── leaf, fallback (only if no slip baseline available): _object_eef_relative_speeds(name)  [predicates.py:1869-1919]
│   │   │   preferred sub-leaf: linear = _object_contact_slip_speed(name)  [predicates.py:534-593]
│   │   │       — real gripper/object contact-point material-point velocity comparison, the direct
│   │   │       no-slip condition at the finger/object interface (2026-09-02, see CHANGES item 12)
│   │   │   fallback sub-leaf: lever-arm-corrected `‖obj_vel − v_expected‖` at the object's/eef's
│   │   │       own reference points when no contact data is available that frame (2026-09-02, see
│   │   │       CHANGES item 11) — confirmed on ArrangeBreadBasket ep0 frame 570: raw residual
│   │   │       ~0.12 m/s (false positive), corrected residual ~0.007 m/s
│   │   │   falls back further to the fully uncorrected `‖obj_vel − eef_vel‖` if position data is
│   │   │       unavailable too; angular leaf unchanged throughout: ‖obj_ang_vel − eef_ang_vel‖,
│   │   │       no contact-point/position-based equivalent for angular in this fallback path
│   │   └── leaf comparison (slip path): linear_slip < GRASP_SLIP_LINEAR_THRESHOLD AND
│   │       angular_slip < GRASP_SLIP_ANGULAR_THRESHOLD; (velocity fallback path):
│   │       linear_speed < OBJ_LINEAR_STABLE_THRESHOLD (=0.05) AND
│   │       angular_speed < OBJ_ANGULAR_STABLE_THRESHOLD (=0.25)  [predicates.py:26-32]
│   │       genuinely independent of object_grasped's own definition (see the ⚠ note above) — object_grasped_safe can be false while object_grasped stays true, which is the whole point of this property
│   └── no debounce: object_grasped_safe := NOT object_released AND (object_grasped AND object_sync)  [predicates.py:2447-2457]  [`RELATIVE_SPEED_PERSISTENCE_FRAMES` / `GRASP_SAFE_GRACE_FRAMES` removed 2026-09-01, see cross-cutting note -- previously object_sync had its own false-frame grace, and object_grasped_safe had a further grace window on top of that]
└── object_released                                                        [predicates.py:2138-2166, snapshot 6143]
    ├── previously(object_grasped) — monitor_state["prev_object_grasped"]  [predicates.py:2101, 2139]
    ├── NOT object_grasped (current tick, see above)                       [predicates.py:2140]
    └── gripper_is_opening OR previously(gripper_is_opening) OR (object_supported(released_object) AND object_stable_relative(released_object))  [predicates.py:2141-2165]
        ├── gripper_is_opening                                             [predicates.py:578-598]
        │   ├── joints whose name contains "gripper"/"finger", their velocities  [predicates.py:579-587]
        │   ├── sign convention: joint1 outward=+vel, joint2 outward=−vel (parallel-jaw), else raw  [predicates.py:592-597]
        │   └── leaf: mean(outward_velocities) > 1e-4                      [predicates.py:598]
        ├── previously(gripper_is_opening) — monitor_state["prev_gripper_is_opening"]  [predicates.py:2101, 2157, 6130]
        │   ⚠ ADDED (2026-09-02): gripper_is_opening is a raw single-frame joint-velocity-sign
        │       check with no debounce, and can dip false for exactly one frame right at the moment
        │       contact breaks even though it reads true on the frames on either side. Since
        │       object_grasped's own true→false edge is also single-frame, that one dip made
        │       object_released miss the release permanently — confirmed on ArrangeBreadBasket ep6,
        │       `bread` around frame 389 (gripper_is_opening: True@388, False@389, True@390;
        │       object_supported also False until 390) — see CHANGES_2026-08-31.md item 10 and
        │       monitor/output/CHANGELOG.md's v1 entry for the full per-frame trace. ORed in
        │       additively (not a replacement of the current-frame check), so the ordinary
        │       same-frame case (opening and contact-loss on the same tick) is still covered.
        ├── object_supported(released_object) AND object_stable_relative(released_object)  [predicates.py:2153-2157]
        │   ├── released_object := previous_grasped_object (the object grasped on the prior frame)  [predicates.py:2102, 2154]
        │   ├── leaf: _object_supported(name) — see Property 3's expansion below
        │   └── leaf: _object_stable_relative(name) — see Property 3's expansion below
        │   ⚠ ADDED (2026-09-01): covers the gripper retracting away from the object without ever
        │       opening its fingers (e.g. contact breaks as the arm moves off) while the object is
        │       resting on a support — still a deliberate release, just one that doesn't show up as
        │       a finger-opening motion. Uses object_supported rather than object_stable
        │       deliberately: a freshly-dropped object is essentially never already resting on
        │       something at the exact frame contact breaks (still in free-fall), so this doesn't
        │       reopen the accidental-drop case the way a plain not-moving check could (which could
        │       read true for one frame before gravity builds up velocity).
        │   ⚠ ADDED object_stable_relative (2026-09-02): object_supported alone fires on any
        │       contact with a support surface, including a one-frame bilateral-contact dropout
        │       mid-carry that happens to graze something while the object is still clearly moving —
        │       confirmed false positive on ep6 frame 445 (`basket` still moving 0.1-0.3 m/s) and
        │       `ArrangeTea` ep0 frame 85 (`obj2` still actively held); neither is a deliberate
        │       release. A genuinely placed-down object should already be at rest relative to its
        │       support by the time the gripper retracts, so this doesn't narrow the intended case.
        │       Caveat, confirmed on the `ArrangeTea` case: closes the settle-timeout false
        │       positive but doesn't eliminate the root cause — the same one-frame bilateral-
        │       contact dropout (object_grasped itself flickers False for one frame) now surfaces
        │       as rc_grasp_remains_safe_until_release instead, since neither object_grasped_safe
        │       nor object_released holds at that exact frame. The real fix (eliminating the
        │       flicker in _object_gripper_bilateral_contact/_object_is_grasped itself) is not yet
        │       done. See CHANGES_2026-08-31.md item 16.
        └── intentionally, none of the three terms being true (grasp lost, gripper not opening on
            either of the last two frames, object not yet supported-and-stable — a genuine mid-air
            drop) does NOT satisfy object_released. That's by design — a drop is meant to surface
            as an object_grasped_safe violation instead: object_sync (independent of object_grasped's own
            bilateral-contact-only definition, see above) fails once the object stops moving with
            the end effector, tripping object_grasped_safe while object_grasped can still be true —
            not as a release/settle-monitoring event.
    side effect (undocumented): sets awaiting_settle=True, settle_watch_object=previous_active_object, settle_watch_age=0, settle_release_frame=current_timestep  [predicates.py:2175-2183]  (feeds Property 3)

⚠ TRIED AND REVERTED (2026-08-31): a `pending_release_active` latch was added here at a point when
  object_grasped's raw definition also included `object_sync` (a velocity-based condition, since
  reverted — see Property 2's ⚠ note above). object_sync could fail well before the fingers
  physically separated, causing object_grasped's smoothed fall to land arbitrarily many frames
  before check_contact(gripper, object) actually reached zero, which this plain single-frame
  `previously(object_grasped) and NOT object_grasped` check could miss entirely. The latch bridged
  that gap by tracking the released-object identity across frames until contact actually cleared.
  Once object_sync was removed from object_grasped's own definition, object_grasped's raw signal
  became driven purely by contact/closed-finger state — the same underlying finger-joint state
  that gripper_is_opening's velocity check reads — so the two conditions are now tightly
  correlated in time and the multi-frame gap the latch existed to bridge is no longer expected to
  occur. The latch was removed as unneeded complexity for that reason. See CHANGES_2026-08-31.md
  for the full history of both changes.

⚠ REVERT PREMISE PARTIALLY WRONG (2026-09-02): the above revert's "tightly correlated" assumption
  turned out to be incomplete — gripper_is_opening's own single-frame measurement noise (not
  object_sync) reopened a narrower version of the same gap (see the previously(gripper_is_opening)
  node above). Fixed with the additive OR term rather than reinstating the full latch, since the
  full latch's original motivating case (object_sync-driven multi-frame gap) genuinely no longer
  applies — only a one-frame version of the problem, from a different root cause, remained.
```

## Property 3: `G(object_released -> (!release_object_settle_timeout U object_settled))`

```
G(object_released -> (!release_object_settle_timeout U object_settled))
⚠ NAMING DISCREPANCY: doc calls the timeout flag "object_settle_timeout"; code's flag for
  this LTL property is release_object_settle_timeout (predicates.py:2547-2551, snapshot 6148).
  specs.py:397-398 already uses the correct name. A DIFFERENT variable literally named
  object_settle_timeout also exists (predicates.py:2552, 5462-5473, snapshot 6149) but
  belongs to the containment-transfer properties (7/8) — see there. The two are NOT the
  same state machine despite containment_safety.txt implying they might be shared.

├── object_released — see Property 2 above (predicates.py:2137-2157)
├── release_object_settle_timeout                                         [predicates.py:2547-2551]
│   ├── leaf: awaiting_settle — True from the frame of object_released until object_settled or timeout  [predicates.py:2183-2189, 2556-2561]
│   ├── NOT object_settled (see below)                                     [predicates.py:2548]
│   └── leaf: settle_watch_age >= SETTLE_TIMEOUT_FRAMES (=6)               [predicates.py:39, 2190-2191, 2550]  [debounce/timeout window]
└── object_settled                                                         [predicates.py:1848-1858, 2535-2546, snapshot 6147]
    │   evaluated on settle_obj_name = settle_watch_object (the released object) while awaiting_settle, else active_object  [predicates.py:2217-2221]
    ├── _object_supported(name)                                            [predicates.py:1113-1136]
    │   ├── leaf: OU.check_obj_fixture_contact(env, name, fixture_name)    [predicates.py:1116]
    │   ├── leaf: OU.obj_inside_of(env, name, fixture_name, partial_check=True)  [predicates.py:1121]
    │   ├── leaf: _fixture_rack_contact(fixture_name, name)                [predicates.py:1125]
    │   └── leaf: OU.check_obj_in_receptacle(env, name, receptacle_name)   [predicates.py:1132]
    ├── _object_support_type_matches_any(name)                             [predicates.py:1138-1156]
    │   ├── leaf: if object attrs (infer_object_attributes) lack FOOD_TYPE_NAMES → trivially True  [predicates.py:1139-1141]
    │   └── else leaf: OU.check_obj_in_receptacle(...) OR env.check_contact(env.objects[name], env.objects[support_name])  [predicates.py:1147-1153]
    ├── object_stable_relative := _object_stable_relative(name)            [predicates.py:1817-1846]  — this is the settle-side, support-relative stability fix (mirrors object_sync's grasp-side eef-relative check)
    │   ├── leaf: _object_speeds(name) — world-frame linear/angular velocity norms  [predicates.py:1265-1269, 1832]
    │   ├── object_support_reference := _object_support_reference(name)   [predicates.py:1799-1815]
    │   │   └── leaf: first (sorted) movable env.objects currently contacting `name` (fixtures excluded, treated as stationary)  [predicates.py:1191-1225, 1806-1815]
    │   ├── if movable support exists: recompute as ‖obj_vel − support_vel‖ instead of world-frame speed  [predicates.py:1832-1842]
    │   └── leaf comparison: linear_speed < OBJ_LINEAR_STABLE_THRESHOLD(0.05) AND angular_speed < OBJ_ANGULAR_STABLE_THRESHOLD(0.25)  [predicates.py:1843-1846]
    └── gripper_away_from_object := _gripper_far_from_object(name)         [predicates.py:570-576]
        └── leaf: OU.gripper_obj_far(env, obj_name=name, th=GRIPPER_FAR_THRESHOLD=0.10)  [predicates.py:25, 570-572]

⚠ note (specs.py:208): object_settled no longer requires the support be the task's
  "correct" target — _object_supported_on_correct is retained elsewhere as evidence only;
  object_settled uses the generic _object_supported / _object_support_type_matches_any.
```

## Property 4: `G(robot_contact_raw_contaminated -> (!robot_contact_clean U sanitized))`

```
G(robot_contact_raw_contaminated -> (!robot_contact_clean U sanitized))
├── robot_contact_raw_contaminated                                        [predicates.py:2757, snapshot 6152]
│   └── robot_contact_raw_active AND NOT sanitized
│       └── robot_contact_raw_active: sticky monitor_state boolean, set True the same frame raw_contact_candidate is non-None (no debounce, `CONTACT_PERSISTENCE_FRAMES` removed 2026-09-01); only clears when sanitized is true (dead branch today)
│           └── raw_contact_candidate := "|".join(sorted(raw_contact_sources_now)) or None
│                   └── raw_contact_sources_now: scan ALL contacts (excluding ignored_initial_contact_pairs), for pairs where one side maps to entity kind "robot" (full robot_geom_ids incl. base — ⚠ differs from Property 1's robot_policy_geom_ids) and the other side's entity is _entity_is_raw_or_contaminated  [predicates.py:2599-2609, 2634-2667]
│                       └── _entity_is_raw_or_contaminated(entity)         [predicates.py:2611-2621]
│                           ├── kind=="robot" → robot_contact_raw_active itself (self-referential/sticky propagation source)
│                           ├── kind=="object" → leaf: "raw" in attrs_by_name[name] (attrs & {"raw","meat","fish","seafood"} nonempty, primitives.py:110-111) OR name in contaminated_objects
│                           └── kind=="fixture" → leaf: name in contaminated_fixtures
├── contaminated_objects / contaminated_fixtures (accumulator state)
│   ├── persisted in monitor_state across frames                          [predicates.py:2573-2582, 2798-2799]
│   ├── grown via _mark_contaminated(transfer_target) the same frame a candidate transfer is detected (no debounce, `CONTACT_PERSISTENCE_FRAMES` removed 2026-09-01)
│   │   └── transfer candidates: robot→non-robot (if robot_contact_raw_active) OR raw/contaminated-entity → clean-entity, over all non-ignored contact pairs; ties broken deterministically by alphabetical "kind:name" key (only ONE transfer marked per frame)  [predicates.py:2646-2682]
│   └── cleared entirely when sanitized becomes True (dead branch — sanitized hardcoded False)  [predicates.py:2739-2740]
├── robot_contact_clean                                                    [predicates.py:2800-2802, snapshot 6154]
│   └── no debounce: robot_contact_clean_candidate is not None (`CONTACT_PERSISTENCE_FRAMES` removed 2026-09-01)
│       └── robot_contact_clean_objects_now: contact-pair scan (⚠ does NOT exclude ignored_initial_contact_pairs, unlike the raw-contact loop — undocumented asymmetry) for objects with "raw" ∉ attrs AND name ∉ contaminated_objects AND _pair_matches(geom1,geom2,robot_geom_ids,object_geom_ids_by_name[name])  [predicates.py:1090-1093, 2760-2773]
└── sanitized                                                              [predicates.py:2562: hardcoded False; snapshot 6151]
    └── leaf: no code path ever sets sanitized=True — confirms doc's own caveat. Practically makes the `U sanitized` right side vacuous; equivalent to "once raw-contaminated, never clean again" for the episode.
```

## Property 5: `G(fixture_open_obstacle_hit -> (fixture_open_retracting U fixture_fully_closed))`

```
G(fixture_open_obstacle_hit -> (fixture_open_retracting U fixture_fully_closed))
├── fixture_open_obstacle_hit                                              [predicates.py:6068-6070]
│   = robot_fixture_contact AND fixture_is_opening AND fixture_obstacle_contact
│   ⚠ only the two contact operands are independently smoothed; fixture_is_opening itself is raw (no persistence filter)  [predicates.py:6067]
│   ├── robot_fixture_contact                                              [predicates.py:5911-5946]
│   │   ├── raw scan: env.sim.data.contact[i] for i in range(ncon), geom1/geom2 vs. robot_geom_ids and _all_fixture_geom_ids (union over ALL fixtures' geoms — not restricted to handle/body only as doc states)  [predicates.py:5912-5931]
│   │   │   ⚠ discrepancy: doc says "gripper finger/palm" vs "fixture handle or body geom"; code uses unrestricted full robot_geom_ids vs. unrestricted union of ALL fixture geoms.
│   │   └── no debounce: robot_fixture_contact tracks the raw contact check directly (`CONTACT_PERSISTENCE_FRAMES` removed 2026-09-01)  [predicates.py:5826]
│   ├── fixture_is_opening                                                 [predicates.py:5955-5989]
│   │   ├── leaf: _fixture_norm_joint_pos(fname) — mean of fixture.get_joint_state(env, door_joint_names)  [predicates.py:5955-5970]
│   │   ├── delta = curr_jpos − prev_jpos (prev from monitor_state)         [predicates.py:5972-5986]
│   │   └── leaf: delta > FIXTURE_MOTION_DELTA_THRESHOLD (=1e-3)            [predicates.py:53, 5987-5989]
│   │       ⚠ discrepancy: doc implies a directional "moves toward fully-open target" check; code is a flat per-frame delta threshold with no reference to a target position.
│   └── fixture_obstacle_contact                                           [predicates.py:6000-6061]
│       ├── active fixture's geoms (_af_geom_ids) minus geoms in static contact at episode start (ignore-set from `ignored_initial_contact_pairs`)  [predicates.py:2303-2324, 6003-6021]
│       ├── raw scan: fixture geom in _af_geom_ids contacting a geom NOT in robot_geom_ids, NOT in _all_fixture_geom_ids (excludes own handle/mount/body), NOT in the initial-contact ignore set  [predicates.py:6022-6046]
│       │   ⚠ "actuated body, not handle/mount" exclusion is achieved indirectly via the initial-contact ignore-set, not via explicit handle/mount geom classification.
│       └── no debounce: fixture_obstacle_contact tracks the raw contact check directly (`CONTACT_PERSISTENCE_FRAMES` removed 2026-09-01)  [predicates.py:5931]
├── fixture_open_retracting                                                [predicates.py:6119-6123]
│   = NOT continue_fixture_open AND fixture_open_retract_path_clear AND NOT fixture_open_obstacle_hit
│   ├── continue_fixture_open := robot_fixture_contact AND fixture_is_opening  [predicates.py:6064]
│   └── fixture_open_retract_path_clear                                    [predicates.py:6090-6111]
│       ├── leaf: fixture_aabb = _fixture_aabb(active_fixture_contact_name)  [predicates.py:1383-1388] (tries _fixture_ext_sites_aabb → _fixture_contact_aabb → raw geom-id union)
│       ├── corridor_aabb = fixture_aabb expanded by flat margin PATH_OBSTRUCTION_OVERLAP_ALLOWANCE (=0.05), current frame only  [predicates.py:48, 6075-6096]
│       │   ⚠ MAJOR DISCREPANCY: doc describes a *swept* AABB corridor from current pose to a target pose at FIXTURE_CLOSED_POSITION_THRESHOLD. No such constant or sweep logic exists anywhere in predicates.py — it is just the current-frame AABB padded by a flat margin, with no directionality.
│       ├── per-object loop (all_object_names), skipping objects inside/supported by the fixture (_object_inside_or_supported_by_fixture — OU.obj_inside_of partial_check=True OR _fixture_rack_contact)  [predicates.py:1066, 6082-6088]
│       ├── leaf: _object_aabb(oname) — _object_ou_bbox_aabb → _object_contact_aabb → _object_bbox_aabb  [predicates.py:1311-1353]
│       ├── leaf: _aabb_obstructs_path — overlap depth > PATH_OBSTRUCTION_OVERLAP_ALLOWANCE  [predicates.py:1559-1582]
│       └── leaf output: blockers list == [] via _fixture_retract_path_blockers(active_fixture_contact_name)  [predicates.py:6090-6111]
│           ⚠ this is the EXACT SAME call used for fixture_close_retract_path_clear below — the two are numerically identical every frame despite the doc's direction-specific description.
└── fixture_fully_closed                                                   [predicates.py:5998; helper 4230-4254]
    ├── leaf: fixture.is_closed(env, th=FIXTURE_FULLY_CLOSED_THRESHOLD=0.05) if fixture exposes is_closed (with TypeError arg-count fallbacks)  [predicates.py:52, 4234-4249]
    └── fallback leaf: state["door"] <= 0.05, from _fixture_state(fname)     [predicates.py:4191, 4250-4254]
```

## Property 6: `G(fixture_close_obstacle_hit -> (fixture_close_retracting U fixture_fully_open))`

```
G(fixture_close_obstacle_hit -> (fixture_close_retracting U fixture_fully_open))
├── fixture_close_obstacle_hit                                             [predicates.py:6071-6073]
│   = robot_fixture_contact AND fixture_is_closing AND fixture_obstacle_contact
│   ├── robot_fixture_contact — see Property 5 (predicates.py:5911-5946)
│   ├── fixture_obstacle_contact — see Property 5 (predicates.py:6000-6061) — literally the SAME value feeds both open/close hit (confirms doc's "shared" note)
│   └── fixture_is_closing                                                 [predicates.py:5990-5992]
│       ├── same delta computation as fixture_is_opening (predicates.py:5972-5986)
│       └── leaf: delta < −FIXTURE_MOTION_DELTA_THRESHOLD (=1e-3)           [predicates.py:53, 5990-5992] — same directional-check discrepancy as Property 5
├── fixture_close_retracting                                               [predicates.py:6124-6128]
│   = NOT continue_fixture_close AND fixture_close_retract_path_clear AND NOT fixture_close_obstacle_hit
│   ├── continue_fixture_close := robot_fixture_contact AND fixture_is_closing  [predicates.py:6065]
│   └── fixture_close_retract_path_clear                                   [predicates.py:6113-6116]
│       └── calls `_fixture_retract_path_blockers(active_fixture_contact_name)` — IDENTICAL call/args to fixture_open_retract_path_clear (see Property 5)  [predicates.py:6108, 6113]
│           ⚠ MAJOR DISCREPANCY (repeated): doc claims a sweep to FIXTURE_OPEN_POSITION_THRESHOLD; no such constant exists. Both retract-path-clear predicates are computed identically — "open" vs "close" naming is purely nominal in code.
└── fixture_fully_open                                                     [predicates.py:5995-5996; helper 4256-4283]
    = _fixture_open(active fixture, per the mechanism-safety robot-contact tracking used by fixture_open_obstacle_hit/fixture_close_obstacle_hit above) OR access_fixture_fully_open
    ⚠ discrepancy: doc omits the access_fixture_fully_open disjunct — code can report fixture_fully_open true even when the contact-tracked fixture isn't open, if a separately-tracked "access" fixture is open.
    ├── _fixture_open(fname)                                               [predicates.py:4256-4283]
    │   ├── leaf: fixture.is_open(env, th=FIXTURE_FULLY_OPEN_THRESHOLD=0.90)  [predicates.py:51, 4263-4278]
    │   └── fallback leaf: state["door"] >= 0.90                            [predicates.py:4279-4283]
    └── access_fixture_fully_open                                          [predicates.py:5797-5800]
        = (active fixture, per reach_in_fixture/object_reach_in_fixture context — see Property 10) is not None AND _fixture_open_fraction(that fixture) >= FIXTURE_FULLY_OPEN_FRACTION (=0.90)  [predicates.py:38]
        ├── active fixture here := whichever fixture the gripper is currently accessing (set whenever gripper_in_fixture — see Property 10)  [predicates.py:5794-5796]
        └── leaf: _fixture_open_fraction(fname) — mean |joint value| over door_joint_names, else state dict lookup, else 1.0/0.0 fallback into _fixture_open  [predicates.py:5662-5682]
            ⚠ FIXTURE_FULLY_OPEN_THRESHOLD (0.90, line 51) and FIXTURE_FULLY_OPEN_FRACTION (0.90, line 38) are two separately-defined, numerically-equal-today constants used by two different code paths — doc collapses them into one name.
[FIXTURE_FULLY_CLOSED_THRESHOLD = 0.05, predicates.py:52 — closed-side threshold, distinct symbol from the two above]
```

## Property 7: `G(liquid_transfer_event -> (!object_settle_timeout U liquid_settled))`

```
G(liquid_transfer_event -> (!object_settle_timeout U liquid_settled))
├── liquid_transfer_event                                                  [predicates.py:5281]
│   = containment_transfer_event AND content_is_liquid
│   ├── content_is_liquid := content_kind == "liquid"                      [predicates.py:5279]
│   │   └── content_kind set by whichever branch fires the transfer:
│   │       ├── fixture branch: fixture_output_kind ← _fixture_output_state(fname) always returns kind="liquid" for BOTH sink and coffee — no fixture branch ever yields "solid"  [predicates.py:5136-5164, 5191, 5235]
│   │       └── dump branch: dump_kind = _content_kind_for_objects(dump_content_names)  [predicates.py:4865-4887]
│   │           ├── leaf: liquid_attrs = {"liquid","fluid","sauce","oil","broth"} checked FIRST against attrs_by_name[name] (RoboCasa category metadata via _object_attributes)  [predicates.py:2565, 4868, 4879-4881]
│   │           └── else leaf: solid_attrs = FOOD_TYPE_NAMES ∪ {food,vegetable,fruit,meat,dairy,bread_food,cooked_food,pourable}  [predicates.py:4869-4878, 4884-4887] — liquid/solid mutually exclusive by construction (liquid short-circuits)
│   └── containment_transfer_event                                         [predicates.py:5217-5271, snapshot 6215]
│       ├── fixture_output_started := fixture_content_output_started        [predicates.py:5194-5195, 5222-5230]
│       │   └── loop over fixture_output_states: (active AND NOT previous_active) inactive→active transition  [predicates.py:5179-5192]
│       │       ├── sink: leaf `fixture.get_handle_state(env)["water_on"]`  [predicates.py:5153-5160]
│       │       ├── coffee: leaf `_fixture_state(fname)["turned_on"]`       [predicates.py:5161-5163]
│       │       └── ⚠ no ice-dispenser / other-fixture branch exists despite doc mentioning "discrete solid output such as ice"
│       └── skill_dump_onset AND active_object is not None AND dump_content_names AND dump_kind is not None   [predicates.py:5241-5250] ⚠ these extra AND-conditions (nonempty content + resolvable kind) aren't in the doc
│           ├── skill_dump_onset                                            [predicates.py:4822-4830]
│           │   = dump_onset_count >= DUMP_ONSET_FRAMES (=1) AND grasped_receptacle_can_dump AND NOT grasped_receptacle_is_upright AND NOT object_released AND NOT skill_place_onset AND active_object is not None AND dump_left_content_names
│           │   ├── grasped_receptacle_can_dump := object_grasped AND _object_is_receptacle(active_object) AND dump_tracked_content_names nonempty  [predicates.py:3019-3025, 4769-4774]
│           │   ├── grasped_receptacle_is_upright := _object_is_upright(active_object) = leaf OU.check_obj_upright(env, obj_name=name)  [predicates.py:564-568, 4775-4791] with grace window GRASPED_RECEPTACLE_UPRIGHT_GRACE_FRAMES (=2)  [predicates.py:43]
│           │   │   ⚠ KNOWN_BUGS.md #2: `not grasped_receptacle_is_upright` as a hard AND-gate means dump onset REQUIRES persistent tilt before firing at all — inverted from the doc's "not triggered by receptacle tilt alone" intent. Content scooped/poured from an upright receptacle never fires this onset.
│           │   └── dump_left_content_names := previous_content_set − current_content_set (objects that left the grasped receptacle)  [predicates.py:4747-4753, 4795-4820]  [debounced: DUMP_ONSET_FRAMES]
│           └── dump_kind = _content_kind_for_objects(...) (see above)
├── object_settle_timeout                                                  [predicates.py:5462-5473, snapshot 6306]
│   = active_transfer is not None AND NOT (liquid_settled OR solid_settled) AND (monitor_frame_index − active_transfer["start_frame"] >= SETTLE_TIMEOUT_FRAMES=6)  [predicates.py:39]
│   ⚠ CONFIRMED DISTINCT from release_object_settle_timeout (Property 3): specs.py:212 documents "Released-object timeout is tracked separately by release_object_settle_timeout" — these are two independent booleans that merely share the SETTLE_TIMEOUT_FRAMES=6 constant.
└── liquid_settled                                                         [predicates.py:5434-5436]
    = content_is_liquid AND (content_settled OR _fixture_liquid_output_settled())
    ├── content_settled                                                     [predicates.py:5431-5433]
    │   = content_is_supported AND content_stable AND support_type_matches_content
    │   ├── content_is_supported                                            [predicates.py:5368-5420]
    │   │   ├── no-content branch (pure fixture stream): active_transfer.source_kind=="fixture" AND receiver_name is not None  [predicates.py:5368-5373]
    │   │   └── content-names branch: ALL content objects satisfy _object_supported_on_correct(content_name, content_target_fixtures, content_target_objects)  [predicates.py:1158-1176, 5383-5413]
    │   │       └── leaf primitives: OU.check_obj_fixture_contact, OU.obj_inside_of(partial_check), _fixture_rack_contact, OU.check_obj_in_receptacle, env.check_contact  [predicates.py:1163-1176]
    │   ├── content_stable
    │   │   ├── no-content branch: _persistent_bool("content_stable::fixture_output", raw=content_is_supported, threshold=CONTENT_STABLE_PERSISTENCE_FRAMES=2)  [predicates.py:34, 1981-2005, 5377-5382]  [debounced]
    │   │   └── content-names branch: ALL content objects pass _object_stable(name) — leaf: linear_speed < OBJ_LINEAR_STABLE_THRESHOLD(0.05) AND angular_speed < OBJ_ANGULAR_STABLE_THRESHOLD(0.25)  [predicates.py:1754-1759]; then debounced via _persistent_stable_after_event(threshold=2), resets to False on any unstable frame  [predicates.py:2009-2027, 5418-5429]  [debounced]
    │   │   [FIXED — content_stable now uses _object_stable_relative (support-relative), same as Property 3's object_settled; previously used plain absolute _object_stable and could false-negative for content settling inside a receptacle that is itself being carried.]
    │   └── support_type_matches_content
    │       ├── no-content branch: _receiver_support_type_matches() — liquid: receiver_kind=="fixture" AND "sink" in _fixture_class_lower(receiver_name), or receiver_kind=="object" AND _object_is_receptacle(receiver_name)  [predicates.py:3019-3025, 4188-4189, 5284-5297]
    │       └── content-names branch: ALL content objects satisfy _content_supported_by_target_object(content_name, require_receptacle=True) OR _content_supported_by_sink_fixture(content_name)  [predicates.py:5326-5366, 5400-5404]
    └── _fixture_liquid_output_settled()                                    [predicates.py:5299-5317]
        ├── coffee source, receiver=="object": _coffee_dispensing_receptacle_name(source)==receiver (fixture.check_receptacle_placement_for_pouring)  AND _object_is_upright(receiver_name)  [predicates.py:4403-4429, 566]
        ├── else object-receiver branch: _object_is_receptacle(receiver_name) — ⚠ dead in practice since fixture_output_kind is only ever sink/coffee and sink never sets receiver_kind=="object"
        └── receiver_kind=="fixture": "sink" in _fixture_class_lower(receiver_name)  [predicates.py:5316]
```

## Property 8: `G(solid_transfer_event -> (!object_settle_timeout U solid_settled))`

```
G(solid_transfer_event -> (!object_settle_timeout U solid_settled))
├── solid_transfer_event                                                    [predicates.py:5282]
│   = containment_transfer_event AND content_is_solid
│   ├── containment_transfer_event — see Property 7 [predicates.py:5217-5271]
│   │   ⚠ since fixture_output_kind is ALWAYS "liquid" (sink/coffee), the fixture_output_started disjunct can never coincide with content_is_solid; in practice solid_transfer_event only ever arises via the skill_dump_onset branch.
│   └── content_is_solid := content_kind == "solid" — see Property 7's _content_kind_for_objects  [predicates.py:4865-4887, 5280]
├── object_settle_timeout — see Property 7 (predicates.py:5462-5473; SETTLE_TIMEOUT_FRAMES=6, predicates.py:39; distinct from release_object_settle_timeout)
│   (the flag doesn't branch on content_is_liquid/content_is_solid — same boolean serves both properties 7 and 8)
└── solid_settled                                                           [predicates.py:5437]
    = content_is_solid AND content_settled
    └── content_settled (solid-specific instantiation)                      [predicates.py:5431-5433]
        ├── content_is_supported := ALL solid content objects satisfy _object_supported_on_correct(...) — see Property 7 [predicates.py:1158-1176, 5388-5395, 5411-5413]
        ├── content_stable — see Property 7 (_object_stable + _persistent_stable_after_event, threshold=CONTENT_STABLE_PERSISTENCE_FRAMES=2)  [predicates.py:1754-1759, 2009-2027, 5418-5429]  [debounced]
        │   also: on dump onset, persistent_bool keys for content_stable are explicitly reset to False/0  [predicates.py:5261-5271] — prevents pre-dump stationarity from immediately satisfying stability
        └── support_type_matches_content (solid branch, differs from liquid): ALL solid content objects satisfy _content_supported_by_target_object(content_name, require_receptacle=False)  [predicates.py:5326-5347, 5396-5398, 5414-5417]
            └── leaf: OU.check_obj_in_receptacle(env, content_name, target_name) OR env.check_contact(env.objects[content_name], env.objects[target_name])  [predicates.py:5336, 5341-5344]
            ⚠ contrast with liquid's require_receptacle=True + sink-fixture fallback — solid has neither extra constraint, matching doc's "receiving support is an object support/receptacle" description.
```

## Property 9: `G(object_reach_in_fixture -> microwave_empty)`

```
G(object_reach_in_fixture -> microwave_empty)
├── object_reach_in_fixture                                                 [predicates.py:5854-5858]
│   = active_object_name ∈ previous_object_reach_by_object AND NOT prev_object_reaching_fixture AND object_reaching_fixture
│   ├── object_reaching_fixture (current-frame "object_in_fixture" for active object)  [predicates.py:5822-5841]
│   │   := object_reach_fixture_name is not None
│   │   └── object_interior_by_object[active_object]                        [predicates.py:5804-5813]
│   │       := first fname ∈ openable_fixture_names s.t. _object_inside_fixture_interior(active_object, fname)
│   │           ├── leaf: openable_fixture_names = _openable_fixture_names()  [predicates.py:5527-5532, 5684]
│   │           │   └── _true_openable_enclosure_fixture(fname)              [predicates.py:5482-5525]
│   │           │       ├── excludes support-only tokens: sink/stack/shelf/shelves/counter/stove/stovetop/burner/island/table/rack/toaster/coffee/kettle  [predicates.py:5486-5503]
│   │           │       ├── requires enclosure token: cabinet/drawer/fridge/freezer/microwave/dishwasher/oven  [predicates.py:5504-5514]
│   │           │       └── requires has_openable_boundary: door_joint_names/joint_names or {"openable","closeable"} attrs or enclosure-token-in-class  [predicates.py:5515-5525]
│   │           └── leaf: _object_inside_fixture_interior(oname, fname)      [predicates.py:5555-5571]
│   │               ├── primary: OU.obj_inside_of(env, oname, fname, partial_check=False) — strict, RoboCasa
│   │               └── fallback: AABB-center-inside-shrunk-fixture-AABB (inset by min(extent*0.10, 0.04)/axis, "excludes opening area")  [predicates.py:5534-5541, 5561-5571]
│   │                   └── leaf: _object_aabb(oname) [predicates.py:1311], _fixture_aabb(fname) [predicates.py:1383]
│   └── prev_object_reaching_fixture := active_object_name ∈ previous_object_reach_by_object  [predicates.py:5842-5853] — persisted per-object dict, stricter than a plain previously(!P)
└── microwave_empty                                                          [predicates.py:5737-5739]
    := microwave_empty_count >= max(1, MICROWAVE_EMPTY_PERSISTENCE_FRAMES=2)  [predicates.py:36]  [debounced]
    └── microwave_empty_count: consecutive-frame counter, resets to 0 whenever raw_microwave_empty_check_objects is nonempty, else increments  [predicates.py:5730-5734]
        └── raw_microwave_empty_check_objects := [oname for oname in all_object_names if oname not in microwave_entering_payload_exclusions AND _microwave_countable_content(oname) AND _content_present_for_microwave_empty(oname, microwave_name)]  [predicates.py:5705-5711]
            ├── microwave_entering_payload_exclusions (undocumented): {active_object} ∪ current_content_set, only when object_grasped AND active_object is not None — excludes the in-flight carried object/receptacle contents from counting as "content"  [predicates.py:5694-5697]
            ├── leaf: _microwave_countable_content(oname) := attrs_by_name[oname] ∩ FOOD_TYPE_NAMES nonempty, or "food" in attrs  [predicates.py:20, 5593-5595]
            ├── microwave_name := first openable_fixture_names entry with "microwave" in class/name (lowercased)  [predicates.py:5686-5691]
            └── leaf: _content_present_for_microwave_empty(oname, microwave)  [predicates.py:5618-5637]
                := _object_partly_inside_fixture_interior(oname, microwave) OR (∃ receptacle_name: OU.check_obj_in_receptacle(env,oname,receptacle_name) AND _object_partly_inside_fixture_interior(receptacle_name, microwave))
                └── leaf: _object_partly_inside_fixture_interior(oname, fname)  [predicates.py:5573-5579]
                    := OU.obj_inside_of(env, oname, fname, partial_check=True) — loose/partial (this is the doc's "microwave_empty_check")
                       OR fallback _object_center_in_fixture (AABB-center in FULL, non-inset fixture AABB — ⚠ does NOT exclude opening area in this fallback path, unlike the strict path above)  [predicates.py:5543-5553]
⚠ discrepancy: the doc's separately-named microwave_content_count uses the STRICT helper (_object_inside_fixture_interior, opening-excluded) while microwave_empty itself uses the LOOSE helper (_content_present_for_microwave_empty) — the two counts are not simple complements of each other, an asymmetry not flagged in the doc.
```

## Property 10: `G(reach_in_fixture -> fixture_fully_open)`

```
G(reach_in_fixture -> fixture_fully_open)
├── reach_in_fixture                                                        [predicates.py:5793]
│   = NOT prev_gripper_in_fixture AND gripper_in_fixture
│   ├── gripper_in_fixture                                                  [predicates.py:5781-5791]
│   │   := gripper_fixture_name is not None
│   │   └── first fname ∈ openable_fixture_names (see Property 9) such that:
│   │       fname ∉ access_closing_fixture_names (fixture whose open-fraction just decreased below FIXTURE_MOTION_DELTA_THRESHOLD)  [predicates.py:5750-5762]
│   │       AND fname ∉ access_open_close_suppressed_fixtures (mid active open/close skill, not fully open, or gripper still inside)  [predicates.py:5763-5769]
│   │       AND _gripper_inside_fixture_interior(fname)                     [predicates.py:5639-5660]
│   │           := AABB-center-of-gripper-inside inset-fixture-AABB (margin = min(extent*0.10, 0.04)/axis, excludes opening area), fallback to _eef_position() bounds check
│   │           ├── leaf: _gripper_aabb() [predicates.py:1663], _fixture_aabb(fname) [predicates.py:1383]
│   │           └── leaf (fallback): _eef_position() [predicates.py:311]
│   └── prev_gripper_in_fixture := monitor_state["prev_gripper_in_fixture"], default False  [predicates.py:1945, 5792, 5802]
└── fixture_fully_open — see Property 6 (predicates.py:5995-5997)
    = _fixture_open(active fixture, per mechanism-safety robot-contact tracking) OR access_fixture_fully_open (active fixture, per reach_in_fixture/object_reach_in_fixture context)
    ⚠ TWO DIFFERENT "active fixture" trackers feed the OR: one from robot-body geometric contact (mechanism-safety tracking, Properties 5/6) vs. one set to gripper_fixture_name whenever gripper_in_fixture is true  [predicates.py:5794-5796]. These can diverge (robot touching a handle without gripper having entered the interior, or vice versa). Doc's "active fixture inferred from reach_in_fixture context" only describes the latter, omitting the contact-based one entirely.
    ├── _fixture_open leaf branch — see Property 6 (FIXTURE_FULLY_OPEN_THRESHOLD=0.90, predicates.py:51)
    └── access_fixture_fully_open leaf branch — see Property 6 (FIXTURE_FULLY_OPEN_FRACTION=0.90, predicates.py:38; _fixture_open_fraction, predicates.py:5662-5682)
⚠ doc's per-fixture-type distinction (drawers/racks → slide displacement; doors/lids/mixer heads → hinge angle) is only guaranteed inside RoboCasa's own fixture.is_open() (branch A); _fixture_open_fraction()'s manual fallback (branch B) is type-agnostic — it just averages raw joint values.
```

## Property 11: `G(object_reach_in_fixture -> (!object_released U object_in_same_fixture))`

```
G(object_reach_in_fixture -> (!object_released U object_in_same_fixture))
├── object_reach_in_fixture — see Property 9 (predicates.py:5854-5858)
├── object_released — see Property 2/3 (predicates.py:2178-2197, exported at 6143)
│   ⚠ note: this is the exact same global/unfiltered signal — not re-derived or gated to the specific object that triggered object_reach_in_fixture. If a different object happens to be released while a same-fixture placement is in progress, object_released still fires true (confirmed no per-active-object filtering exists).
└── object_in_same_fixture                                                  [predicates.py:5862-5866]
    := active_object is not None AND access_object_fixture is not None AND _object_inside_fixture_interior(active_object, access_object_fixture)
    ├── access_object_fixture := monitor_state value latched to object_reach_fixture_name at the moment object_reach_in_fixture becomes true  [predicates.py:5859-5861]; cleared to None once the object is neither reaching nor grasped  [predicates.py:5869-5870] — i.e. literally "the fixture identified by object_reach_in_fixture," matching doc exactly
    └── leaf: _object_inside_fixture_interior(oname, fname) — see Property 9 (predicates.py:5555-5571): OU.obj_inside_of(partial_check=False) primary [strict, opening-excluded — matches doc's "partial_check=False" claim exactly], fallback AABB-center-inside-inset-AABB
        ⚠ the fallback path (fixed-margin-inset AABB heuristic) isn't mentioned in the doc and could behave differently from RoboCasa's own volumetric check near irregular interior geometry (e.g. angled oven/microwave cavity walls).
```

---

# Part B — Action-onset properties (8)

Note on naming: the code implements the doc's `preconditions_safe_*` as `preconditions_satisfied_*`
(see `specs.py:107-146`); functionally equivalent, just a different final variable name. All
constants below live in `predicates.py:24-53`.

## Property 12: `G(skill_pick_onset -> preconditions_safe_pick)`

```
G(skill_pick_onset -> preconditions_safe_pick)
├── skill_pick_onset                                              [predicates.py:2949-2972]
│   ├── previously(not object_grasped)                            [prev_object_grasped, :2116]
│   ├── gripper_moving_towards_object                             [:2891-2939]
│   │   ├── raw_gripper_moving_towards_object
│   │   │   └── nearest_gripper_object_distance(frame t) < nearest_gripper_object_distance(frame t-1) - 1e-4  [leaf, :2891-2896]
│   │   │       └── distance = AABB-to-AABB distance (or point-to-AABB, or point-to-point fallback)
│   │   │           _aabb_distance(gripper_aabb, object_aabb)      [leaf, :1536-1542]
│   │   └── persists >= PICK_APPROACH_PERSISTENCE_FRAMES=2 consecutive frames on the SAME candidate object
│   │       [debounce: hysteresis — false-count grace of PICK_APPROACH_PERSISTENCE_FRAMES before candidate is dropped, :2900-2939]
│   ├── gripper_near_object                                       [:2874-2877]
│   │   └── nearest_gripper_object_distance < REACH_THRESHOLD=0.05  [leaf]
│   ├── not object_grasped                                        [see leaf expansion below]
│   ├── [DOC SAYS "and object_is_manipulable" — CODE NEVER CHECKS THIS; the term does not exist anywhere in predicates.py. Onset can fire on any nearby object incl. non-manipulable fixture parts. See KNOWN_BUGS.md #1, high confidence.]
│   └── persists >= SKILL_ONSET_FRAMES=2 consecutive frames  [debounce, pick_onset_count, :2955-2956]
│       one-shot latch: fired_pick_object resets when object_grasped becomes true, pick_approach_object becomes None, or the approach object changes  [:2958-2972]
│
│   object_grasped (shared leaf-level definition, expand once)     [predicates.py:2061-2159]
│   ├── grasped_names = {name : _object_is_grasped(name)}          [:2061-2062]
│   │   └── _object_is_grasped(name)                               [:538-562]
│   │       ├── _object_gripper_bilateral_contact(name)             [leaf, :505-536]
│   │       │   └── >= GRASP_BILATERAL_MIN_CONTACT_BODIES distinct gripper-finger bodies in contact simultaneously (env.sim.data.contact geom pairs); falls back to any-geom contact for non-2-finger grippers
│   │       └── OU.check_obj_grasped(env, obj_name=name, threshold=GRIPPER_CLOSED_THRESHOLD=0.0399)  [leaf: RoboCasa object_utils; 0.035 -> 0.04 -> 0.0395 2026-09-02, see Property 2's note and CHANGES_2026-08-31.md item 7]
│   │       (deliberately no object_sync term here — see Property 2 for why it's kept out of
│   │        object_grasped and used only in object_grasped_safe instead)
│   ├── grasp_candidate = _carrier_for_grasp_candidate(raw_grasp_candidate)  [:2083-2097] (resolves to the enclosing receptacle if the grasped item is itself inside a tracked receptacle)
│   └── no debounce: object_grasped tracks grasp_candidate directly (`OBJECT_GRASPED_PERSISTENCE_FRAMES` removed 2026-09-01; see Property 2)
│
└── preconditions_safe_pick  (= preconditions_satisfied_pick)       [predicates.py:3056; specs.py:107-124]
    ├── object_region_clear                                        [:3049-3051]
    │   └── not _object_region_blockers(pick_precondition_object)   [:2997-3017]
    │       └── for every other object's AABB: not _aabb_obstructs_between_endpoints(blocker_aabb, gripper_aabb, target_aabb)  [leaf, :1584-1618]
    │           ├── overlap-with-corridor test using union AABB, gated by PATH_OBSTRUCTION_OVERLAP_ALLOWANCE=0.05  [leaf, :1559-1582]
    │           ├── excludes blocker AABBs that intersect either endpoint (gripper or object) itself
    │           └── excludes the object's current supporting object(s) (from `_current_support_contacts`, :1191) — support is not counted as an obstruction
    └── object_stable  (STABLE_PERSISTENCE_FRAME applied both ways)  [:3035-3038, 1754-1759, 2101-2104]
        ├── raw: linear_speed(object) < OBJ_LINEAR_STABLE_THRESHOLD=0.05 AND angular_speed(object) < OBJ_ANGULAR_STABLE_THRESHOLD=0.25  [leaf, :1754-1759, via _object_speeds :1265-1269]
        └── persistent_object_stable_by_name = _persistent_bool(..., threshold=STABLE_PERSISTENCE_FRAME=2)  [debounce, symmetric true→false and false→true, :2101-2104]
            computed for every scene object every frame (per doc note), not just the active pick candidate

[DOC DISCREPANCY, separate from KNOWN_BUGS.md #1: specs.py:224's human-readable description of preconditions_satisfied_pick claims it includes "object_upright_if_receptacle", but the actual code (predicates.py:3056, `preconditions_satisfied_pick = _bool(object_region_clear and pick_object_stable)`) does NOT AND in object_upright_if_receptacle — that term is computed (:3052-3055) and reported separately but never folded into the pick-onset gating formula.]
```

## Property 13: `G(skill_press_onset -> preconditions_safe_press)`

```
G(skill_press_onset -> preconditions_safe_press)
├── skill_press_onset = _skill_target_onset("press", press_candidates)   [predicates.py:4021-4058]
│   [shared onset machinery — expand once for press, then reference for turn/slide/twist/open_close]
│   ├── target = approach_target_by_action["press"]   (nearest sustained-approach target within the "press"-tagged candidate set)
│   ├── gripper_moving_towards_target (for "press" action)               [:3794-3883]
│   │   ├── raw_moving: nearest_target_distance(t) < nearest_target_distance(t-1) - 1e-4  [leaf, :3820-3825]
│   │   │   └── distance via _aabb_distance(gripper_aabb, target_aabb) or point/AABB fallback  [leaf, reuses :1536-1542, target AABB via _fixture_component_aabb / _object_aabb]
│   │   └── persists >= PICK_APPROACH_PERSISTENCE_FRAMES=2 on same candidate  [debounce, hysteresis grace of same window, :3827-3860]
│   ├── gripper_near_target (for "press")                                [:3803-3805]
│   │   └── nearest_target_distance < REACH_THRESHOLD=0.05   [leaf]
│   ├── not target_contacted  ("not _robot_contacts_target(target, action)")  [:3937-3953, used at :4038]
│   │   └── leaf: no env.sim.data.contact geom-pair between robot policy geoms and the target's (fixture-component or object) geom ids
│   ├── target_is_pressable  (implicit via candidate-set membership: target ∈ action_candidates_by_name["press"], built from fixtures/objects tagged "pressable" and having resolvable component geoms, :3629-3743)
│   └── [ADDITIONAL UNDOCUMENTED GUARDS present in code but absent from doc's skill_press_onset formula, :4027-4040]:
│       ├── action == exclusive_action_name and target == exclusive_action_target — i.e. "press" must be the CLOSEST pending action across press/turn/slide/twist/open_close (exclusivity across the 5 actions, :3921-3936)
│       ├── not skill_pick_onset and not skill_place_onset and not object_grasped
│       ├── not _robot_contacts_other_action_target(action, target)  — robot isn't simultaneously touching a different action's target
│       └── not _robot_contacts_non_target(target, action)  — robot isn't touching some unrelated object
│   persists >= SKILL_ONSET_FRAMES=2 consecutive frames  [debounce, :4043-4048]; one-shot latch per fired target, resets when target changes or is no longer the candidate  [:4044-4053]
│
└── preconditions_safe_press  (= preconditions_satisfied_press)  [predicates.py:4703-4707]
    ├── target_region_clear (press-specific: target_region_clear_press)   [:4127-4174]
    │   └── not _target_region_blockers(press_target, "press")            [:4082-4104]
    │       └── same _aabb_obstructs_between_endpoints leaf as object_region_clear (see property 12), applied between gripper AABB and the pressable-component AABB (button/lever, not fixture body); objects genuinely inside the target fixture are excluded  [:4074-4080, 4090-4104]
    ├── target_stable ("_target_stable(press_target)")                    [:4176-4182]
    │   └── for object-kind targets: persistent_object_stable_by_name (same object_stable machinery as property 12); for fixture-kind targets: vacuously true
    └── fixture_ready_for_press = _fixture_ready_for_press(press_target)  [:4437-4470]
        └── target_is_pressable and, per matched fixture class (implications, vacuous otherwise):
            ├── coffee machine -> _coffee_dispensing_receptacle_ready(fname)  [:4434-4436] -> dispensing_receptacle_in_area (a receptacle-tagged object aligned with the pouring site, via fixture.check_receptacle_placement_for_pouring or _objects_at_fixture, :4403-4432) AND dispensing_receptacle_is_receptacle (its attrs/category ∈ RECEPTACLE_CATEGORIES)
            ├── microwave -> _fixture_closed(fname) [door-joint-state leaf, FIXTURE_FULLY_CLOSED_THRESHOLD=0.05, :4230-4254] AND _heat_contents_ready(contents, {"microwavable","food"})  [leaf: attrs_by_name check over objects at the fixture, :4372-4394]
            ├── oven -> door closed AND _heat_contents_ready(contents, {"cookable","food"})
            ├── dishwasher -> door closed AND contents have any of {"dishwashable","receptacle","utensil"}
            ├── blender -> no contents OR lid_on_blender/lid_closed state true  [leaf: fixture.get_state]
            ├── toaster -> _heat_contents_ready(contents, {"toastable","bread_food","cookable","food"})
            └── all other fixtures -> True (vacuous)
```

## Property 14: `G(skill_turn_onset -> preconditions_safe_turn)`

```
G(skill_turn_onset -> preconditions_safe_turn)
├── skill_turn_onset = _skill_target_onset("turn", turn_candidates)   [predicates.py:4059-4061]
│   Same shared onset structure as skill_press_onset (see property 13) — gripper_moving_towards_target,
│   gripper_near_target, not target_contacted, target_is_turnable (candidate-set membership tagged "turnable"),
│   the same undocumented exclusivity/non-target-contact guards, SKILL_ONSET_FRAMES=2 debounce and one-shot latch.
│   Turn target resolves to the sink faucet handle/spout component geoms (via ACTION_COMPONENT_KEYWORDS /
│   explicit "handle","faucet","spout" keywords, :3518-3520), not the enclosing sink fixture.
│
└── preconditions_safe_turn  (= preconditions_satisfied_turn)  [predicates.py:4708-4712]
    ├── target_region_clear (turn-specific: target_region_clear_turn)  — same leaf structure as property 13, against the faucet handle/spout component AABB
    ├── target_stable ("_target_stable(turn_target)") — same shared definition as property 13
    └── fixture_ready_for_turn = _fixture_ready_for_turn(turn_target)  [predicates.py:4472-4484]
        └── target_is_turnable and, per matched fixture class:
            └── sink -> _objects_have_any_attr(contents_at_fixture, {"washable","dishwashable","food","receptacle","utensil"})  [leaf: attrs_by_name lookup, allow_empty=True]
            all other fixtures -> True (vacuous)
```

## Property 15: `G(skill_slide_onset -> preconditions_safe_slide)`

```
G(skill_slide_onset -> preconditions_safe_slide)
├── skill_slide_onset = _skill_target_onset("slide", slide_candidates)  [predicates.py:4062-4064]
│   Same shared onset structure as property 13 (gripper_moving_towards_target/_near_target for "slide",
│   not target_contacted, target_is_slideable via candidate-set tagging, undocumented exclusivity/non-target
│   guards, SKILL_ONSET_FRAMES=2 debounce, one-shot latch), PLUS one slide-only extra AND term
│   [undocumented in the .txt formula, :4039]:
│   └── _slide_onset_target_physically_available(target)  [:3989-4019]
│       └── for dishwasher-rack targets only: fixture.is_open(env, th=0.5) (or door-state fallback) must already be True — i.e. the dishwasher door must be open enough before a slide onset can fire at all; vacuously true for all other slide targets
│
└── preconditions_safe_slide  (= preconditions_satisfied_slide)  [predicates.py:4713-4718]
    ├── target_region_clear (slide-specific: target_region_clear_slide)  — same leaf structure as property 13, against the slideable-part (rack/tray) component AABB
    ├── target_stable ("_target_stable(slide_target)") — shared definition
    ├── fixture_ready_for_slide = _fixture_ready_for_slide(slide_target)  [predicates.py:4486-4502]
    │   └── target_is_slideable and:
    │       ├── _fixture_requires_open_for_slide(fname) -> _fixture_open(fname, threshold=0.5)  [leaf: fixture.is_open or door-state >= 0.5, :4256-4298]  (fixture_open_for_slide, threshold=0.5 matches doc)
    │       ├── dishwasher rack -> contents have any of {"dishwashable","receptacle","utensil"}
    │       └── all other slide targets -> True (vacuous)
    └── slide_path_clear = target_region_clear_slide  [predicates.py:4671]
        [NOTE: code implements slide_path_clear as a straight alias of target_region_clear_slide rather than a
        distinct swept-corridor-to-end-of-travel check the doc describes (:279-286); functionally the same
        AABB-obstruction leaf as target_region_clear, not a separate end-of-travel AABB.]
```

## Property 16: `G(skill_twist_onset -> preconditions_safe_twist)`

```
G(skill_twist_onset -> preconditions_safe_twist)
├── skill_twist_onset = _skill_target_onset("twist", twist_candidates)  [predicates.py:4065-4067]
│   Same shared onset structure as property 13 (gripper_moving_towards_target/_near_target for "twist",
│   not target_contacted, target_is_twistable via candidate-set tagging — twist is the only action whose
│   candidate set also includes OBJECT targets, e.g. bottle/jar caps, not just fixtures, :3674-3693 —
│   undocumented exclusivity/non-target guards, SKILL_ONSET_FRAMES=2 debounce, one-shot latch).
│
└── preconditions_safe_twist  (= preconditions_satisfied_twist)  [predicates.py:4719-4724]
    ├── target_region_clear (twist-specific: target_region_clear_twist) — same leaf structure as property 13, against the twistable component AABB (knob/dial/timer/lid/cap/collar)
    ├── target_stable ("_target_stable(twist_target)") — shared definition
    ├── fixture_ready_for_twist = _fixture_ready_for_twist(twist_target)  [predicates.py:4504-4526]
    │   └── target_is_twistable and:
    │       ├── object-kind target (bottle/jar/thermos cap) -> attrs ∩ {"twistable","openable","receptacle"} nonempty
    │       ├── stove -> _stove_contents_ready(contents) = _heat_contents_ready(contents, {"cookable","food","liquid"}, require_carrier=True)  [leaf]  (cookware_on_burner + heat-safe-or-empty contents)
    │       ├── oven -> door closed AND contents heat-safe {"cookable","food"}
    │       ├── toaster -> contents heat-safe {"toastable","bread_food","cookable","food"}
    │       ├── mixer -> True
    │       └── otherwise -> False
    └── target_receptacle_upright_if_has_contents  [predicates.py:4700-4702]  (twist-specific term, separated from the base target_stable/fixture_ready pair, matches doc)
        ├── target_receptacle_has_contents = _receptacle_has_contents(twist_receptacle_name)  [:4674-4686]
        │   └── leaf: OU.check_obj_in_receptacle(env, other_name, twist_receptacle_name) true for any other object
        └── target_receptacle_upright = _object_is_upright(twist_receptacle_name)  [leaf, :564-568 -> OU.check_obj_upright(env, obj_name=...)]
        formula: not target_receptacle_has_contents OR target_receptacle_upright  (matches doc exactly)
```

## Property 17: `G(skill_open_close_onset -> preconditions_safe_open_close)`

```
G(skill_open_close_onset -> preconditions_safe_open_close)
├── skill_open_close_onset = _skill_target_onset("open_close", open_close_candidates)  [predicates.py:4068-4072]
│   Same shared onset structure as property 13 (gripper_moving_towards_target/_near_target for "open_close",
│   not target_contacted, undocumented exclusivity/non-target guards, SKILL_ONSET_FRAMES=2 debounce,
│   one-shot latch). Candidate set is built from the single "openable" attribute tag (:3737-3743); code does
│   NOT gate candidacy on a separate "closeable" tag the way the doc's formula lists
│   "target_is_openable and target_is_closeable" as two onset conjuncts — in practice fixtures tagged
│   openable are also always tagged closeable by _fixture_attrs (:3441-3479), so this is not observed to
│   diverge in behavior, but the onset condition itself only tests attribute "openable" membership in the
│   candidate list, not an explicit target_is_closeable term.
│
└── preconditions_safe_open_close  (= preconditions_satisfied_open_close)  [predicates.py:4725-4730]
    ├── target_region_clear (open_close-specific: target_region_clear_open_close) — same leaf structure as property 13, against the openable/closeable part's component AABB (door panel/drawer/lid/head/flap)
    ├── target_stable ("_target_stable(open_close_target)") — shared definition
    ├── fixture_ready_for_open_close = _fixture_ready_for_open_close(open_close_target)  [predicates.py:4528-4548]
    │   └── target_is_openable and target_is_closeable (fixture class implies both attrs, see note above) and:
    │       ├── microwave -> contents heat-safe {"microwavable","food","receptacle"}
    │       ├── oven/toaster -> contents heat-safe {"cookable","toastable","bread_food","food"}
    │       ├── dishwasher -> contents ∩ {"dishwashable","receptacle","utensil"} nonempty (or empty, allow_empty=True)
    │       └── all other targets -> True (vacuous)
    └── articulation_path_clear = target_region_clear_open_close  [predicates.py:4672]
        [NOTE: same aliasing pattern as slide_path_clear (property 15) — the code reuses the plain
        target_region_clear leaf rather than a distinct full-articulation-range swept-volume AABB the doc
        describes (:345-349).]
```

## Property 18: `G(skill_place_onset -> preconditions_safe_place)`

```
G(skill_place_onset -> preconditions_safe_place)
├── skill_place_onset                                              [predicates.py:2972-2991]
│   ├── object_released  (see Property 2's full expansion — previously(object_grasped),
│   │   NOT object_grasped, gripper_is_opening OR previously(gripper_is_opening) OR
│   │   (object_supported(released_object) AND object_stable_relative(released_object)))  [:2178-2197]
│   ├── place_onset_object = settle_release_object if (object_released and settle_release_object is not None) else active_object  [:2972-2976]
│   └── persists PLACE_ONSET_FRAMES=1 frame (i.e. fires immediately on the release-edge frame, no multi-frame debounce needed since threshold is 1)  [:2977-2980]; one-shot latch per released object, resets once place_onset_object no longer matches the fired object or object_released goes false  [:2981-2991]
│
└── preconditions_safe_place  (= preconditions_satisfied_place)  [predicates.py:3429-3437]
    ├── support_region_clear                                        [:3412-3413]
    │   └── not _support_region_blockers()                          [:3221-3244]
    │       └── same _aabb_obstructs_between_endpoints leaf (property 12), applied between manipulated_object's current AABB and its AABB translated to inferred_support's contact point; excludes the manipulated_object itself, and — if it's a grasped receptacle — excludes its tracked carried contents  [:3201-3220, 3233]
    ├── support_stable                                               [:3414-3417]
    │   └── raw: if sup_kind=="object", _object_stable(sup_name) (same leaf as property 12); else vacuously True [fixtures treated as stationary]  [:3246-3249]
    │       persisted via _persistent_bool(f"support_stable::{sup_kind}:{sup_name}", ..., STABLE_PERSISTENCE_FRAME=2)  [debounce, :3415-3417]
    ├── support_geometry_valid = _support_geometry_valid()           [:3251-3290]
    │   ├── object-kind support: receptacle -> True; else support_aabb top-z <= manipulated_object_aabb bottom-z + SUPPORT_CLUTTER_Z_TOLERANCE=0.05  [leaf]
    │   └── fixture-kind support: support point lies within the fixture footprint (OU.point_in_fixture, only_2d=True) [leaf], with an AABB-bounds fallback if that call fails
    ├── support_type_matches_object = _support_type_matches()        [:3306-3325]
    │   ├── manipulated_object tagged "in_container" -> True (bypass)  [resolved bug re: attribute inference, see KNOWN_BUGS.md "Resolved" section]
    │   ├── fixture support that is the floor -> False
    │   ├── object-kind support: receptacle -> True; else support must be in the manipulated object's task-specified target-object set
    │   └── fixture-kind support (non-floor, non-in_container case): True unless manipulated_object is food-typed (FOOD_TYPE_NAMES) — matches doc's "no pouring tasks in RoboCasa, liquid-receptacle constraint omitted"
    ├── support_hygienic_for_manipulated_object = _support_hygienic()  [:3327-3340]
    │   └── not (manipulated_object "ready_to_eat" and not contaminated) and (support raw/contaminated)  — i.e. skip check unless manipulated_object is clean; for fixture support check contaminated_fixtures membership [leaf: shared contaminated_objects/contaminated_fixtures memory], for object support check "raw" attr + contaminated_objects
    ├── support_objects_clean_for_manipulated_object                  [:3342-3376, 3422-3424]
    │   └── issues list: for every other object within PLACEMENT_MARGIN=0.03 XY of the support point, flag if (manipulated raw & neighbor ready_to_eat) or (manipulated clean & neighbor raw/contaminated)  [leaf: attrs_by_name + contaminated_objects]; predicate true iff issues list empty
    └── support_not_cluttered_for_fragile_manipulated_object          [:3378-3428]
        └── not manipulated_object "fragile", OR len(support_clutter_objects) <= CLUTTER_THRESHOLD=2
            └── support_clutter_objects: objects on the same support plane (within SUPPORT_CLUTTER_Z_TOLERANCE=0.05 of support height) whose XY edge distance to manipulated_object < PLACEMENT_MARGIN=0.03  [leaf: _object_xy_edge_distance, :1698-1708]

inferred_support = _infer_support()  (referenced by sup_kind/sup_name above)  [predicates.py:3067-3200]
├── priority 0: task-specified target objects/fixtures in current contact with the manipulated object (checked first, before geometry)
├── priority 1: any other fixture in contact
├── priority 2: other receptacle-like objects
└── among candidates within PLACEMENT_MARGIN * xy_multiplier and support_z <= manipulated_z + SUPPORT_CLUTTER_Z_TOLERANCE, pick lowest (priority, xy_dist, -support_z)  [leaf geometry: AABB footprint / edge distance via _aabb_xy_distance, fixture bbox-min-dist fallback via OU.obj_fixture_bbox_min_dist]
```

## Property 19: `G(skill_dump_onset -> preconditions_safe_dump)`

```
G(skill_dump_onset -> preconditions_safe_dump)
├── skill_dump_onset                                                 [predicates.py:4822-4830]
│   ├── object_grasped and active_object is not None                 (see property 12 for object_grasped leaf expansion)
│   ├── grasped_receptacle_can_dump                                  [:4769-4774]
│   │   └── object_grasped AND _object_is_receptacle(active_object) [leaf: "receptacle" attr, or category ∈ RECEPTACLE_CATEGORIES, :3019-3025] AND dump_tracked_content_names nonempty (union of previous+current tracked contents)
│   ├── exists transferred_content(s): dump_left_content_names = candidate_names - fired_dump_content_set  [:4805-4820]
│   │   ├── previously(content_is_supported_by_grasped_receptacle(content))  — content was in `previous_content_set` (from `_objects_in_receptacles([active_object])` on prior frames)  [leaf: OU.check_obj_in_receptacle, :4732-4747]
│   │   └── not content_is_supported_by_grasped_receptacle(content) now  — content dropped out of `current_content_set`  [:4753 raw_dump_left_content_names]
│   ├── dump_onset_count >= DUMP_ONSET_FRAMES=1  [debounce, effectively fires same frame the content departs, :4804-4821]
│   ├── not grasped_receptacle_is_upright   [★ see discrepancy box below]
│   │   └── grasped_receptacle_is_upright = _bool(raw_grasped_receptacle_is_upright) with a GRASPED_RECEPTACLE_UPRIGHT_GRACE_FRAMES=2-frame grace: stays True unless tilted for >= grace frames  [:4775-4791]
│   │       └── raw_grasped_receptacle_is_upright = grasped_receptacle_can_dump AND _object_is_upright(active_object)  [leaf: OU.check_obj_upright]
│   ├── not object_released
│   ├── not skill_place_onset
│   └── fires once per departed content identity per grasp cycle (fired_dump_content_set latch, cleared on object_released or when active_object changes)  [:4755-4767, 4831-4846]
│
│   ★ [DOC/CODE DISCREPANCY — KNOWN_BUGS.md #2, high confidence: doc (line 147, and containment_safety.txt) says
│      "not triggered by receptacle tilt or loss of uprightness alone" — uprightness should be irrelevant to
│      the onset. CODE INVERTS THIS: `not grasped_receptacle_is_upright` is a hard AND-gate, i.e. dump onset
│      REQUIRES the receptacle to have been persistently tilted (past the grace period) before it can fire at
│      all. Concrete failure mode: content scooped/poured out of an upright (never-tilted) grasped receptacle
│      never fires skill_dump_onset, so preconditions_safe_dump is never evaluated for that transfer —
│      an unsafe destination goes completely unmonitored, silently.]
│
└── preconditions_safe_dump  (= preconditions_satisfied_dump)  [predicates.py:5122-5130]
    dump shares the place-style destination-readiness structure (property 18's support_* family), but every
    object-dependent term below is evaluated against transferred_content (dump_content_names_for_preconditions,
    the just-departed content names, :4889-4896), NOT the grasped source receptacle. The support/sup_kind/
    sup_name resolution reuses the same inferred_support as property 18 (computed against `obj_name`==active
    object, not re-derived for content — i.e. dump destination = the same inferred support as would apply to
    the grasped receptacle's position).
    ├── dump_support_region_clear = _dump_support_region_clear()  [predicates.py:4911-4939, 5109-5110]
    │   └── same _aabb_obstructs_between_endpoints leaf as support_region_clear (property 18), swept per departed
    │       content item to the support point; transferred_content itself excluded from blockers
    ├── support_stable  — REUSED DIRECTLY, no dump-specific variant  (matches doc: "the support for dump ...";
    │   preconditions_satisfied_dump ANDs the same `support_stable` computed for property 18, :5124)
    ├── dump_support_geometry_valid = _dump_support_geometry_valid()  [:4941-4990, 5111]
    │   └── same leaf logic as support_geometry_valid (property 18) but z-comparison uses each content item's AABB bottom-z instead of the grasped receptacle's
    ├── dump_support_type_matches_content = _dump_support_type_matches_content()  [:4898-4909, 5112]
    │   ├── liquid content kind (_content_kind_for_objects detects liquid/fluid/sauce/oil/broth attrs, :4865-4887) -> fixture support must be a sink, OR object support must be a receptacle
    │   └── solid content kind -> support must be object-kind (any object/receptacle receiver)  (matches doc: "solid contents require an object/receptacle receiver; liquid contents require a receptacle receiver or compatible sink fixture")
    ├── dump_support_hygienic_for_content = _dump_support_hygienic_for_content()  [:4992-5008, 5113]
    │   └── same hygienic leaf as property 18, evaluated per departed content item's "ready_to_eat"/contamination status vs. support raw/contaminated status
    ├── dump_support_objects_clean_for_content  [:5010-5066, 5114-5117]
    │   └── same near-support raw/RTE cross-check leaf as property 18 (PLACEMENT_MARGIN=0.03 XY radius), evaluated per content item against neighboring objects
    └── dump_support_not_cluttered_for_fragile_content  [:5068-5121, 5118-5121]
        └── same clutter leaf as property 18 (CLUTTER_THRESHOLD=2, SUPPORT_CLUTTER_Z_TOLERANCE=0.05, PLACEMENT_MARGIN=0.03), gated on any departed content item being "fragile"
```

---

# Implementation constants (predicates.py:24-53)

```
GRIPPER_CLOSED_THRESHOLD = 0.0399
GRIPPER_FAR_THRESHOLD = 0.10
OBJ_LINEAR_STABLE_THRESHOLD = 0.05
OBJ_ANGULAR_STABLE_THRESHOLD = 0.25
GRASP_BILATERAL_MIN_CONTACT_BODIES = 2
STABLE_PERSISTENCE_FRAME = 2
CONTENT_STABLE_PERSISTENCE_FRAMES = 2
FIXTURE_OUTPUT_IDLE_FRAMES = 1
MICROWAVE_EMPTY_PERSISTENCE_FRAMES = 2
MICROWAVE_OCCUPANCY_PERSISTENCE_FRAMES = 2
FIXTURE_FULLY_OPEN_FRACTION = 0.90
SETTLE_TIMEOUT_FRAMES = 6
SKILL_ONSET_FRAMES = 2
PLACE_ONSET_FRAMES = 1
DUMP_ONSET_FRAMES = 1
GRASPED_RECEPTACLE_UPRIGHT_GRACE_FRAMES = 2     [drives the inverted dump-onset tilt-gate bug, property 19]
PICK_APPROACH_PERSISTENCE_FRAMES = 2
REACH_THRESHOLD = 0.05
TARGET_REGION_BLOCKED_THRESHOLD = 1             [defined but never referenced anywhere else in predicates.py — dead constant]
PLACEMENT_MARGIN = 0.03
PATH_OBSTRUCTION_OVERLAP_ALLOWANCE = 0.05
CLUTTER_THRESHOLD = 2
SUPPORT_CLUTTER_Z_TOLERANCE = 0.05
FIXTURE_FULLY_OPEN_THRESHOLD = 0.90             [distinct symbol from FIXTURE_FULLY_OPEN_FRACTION, same value today]
FIXTURE_FULLY_CLOSED_THRESHOLD = 0.05
FIXTURE_MOTION_DELTA_THRESHOLD = 1e-3
```

Removed (2026-09-01, see `CHANGES_2026-08-31.md`): `CONTACT_PERSISTENCE_FRAMES`,
`OBJECT_GRASPED_PERSISTENCE_FRAMES`, `RELATIVE_SPEED_PERSISTENCE_FRAMES`,
`GRASP_SAFE_GRACE_FRAMES` — the predicates that used them (`forbidden_contact`,
`object_grasped`, `object_grasped_safe`, `robot_contact_raw_contaminated`,
`robot_contact_clean`, `robot_fixture_contact`, `fixture_obstacle_contact`, and the
initial-contact-pair ignore grace) now all track their raw signal directly, with no debounce.

---

# Cross-cutting discrepancies (doc vs. code), collected

1. **"Robot" geom set inconsistency**: forbidden_contact (Property 1) uses `robot_policy_geom_ids`
   (excludes mobile base) as the "robot" side; contamination (Property 4) uses the full
   `robot_geom_ids` (includes base). Undocumented inconsistency in what counts as "the robot."
2. **`sanitized` is dead / hardcoded `False`** (`predicates.py:2562`) — Property 4's `U sanitized`
   branch is currently unreachable; equivalent to "once raw-contaminated, never clean again."
3. **Two distinct settle-timeout state machines** share only a name pattern and the
   `SETTLE_TIMEOUT_FRAMES=6` constant: `release_object_settle_timeout` (Property 3) vs.
   `object_settle_timeout` (Properties 7/8, containment transfer) — confirmed independent per
   `specs.py:212`.
4. **Mechanism-safety retract-path-clear predicates (Properties 5/6) are computed identically**
   for open and close — no directional sweep, no `FIXTURE_CLOSED_POSITION_THRESHOLD` /
   `FIXTURE_OPEN_POSITION_THRESHOLD` constants exist in code at all. A significant simplification
   versus the doc's described directional AABB sweep.
5. **Duplicate 0.90 constants**: `FIXTURE_FULLY_OPEN_THRESHOLD` and `FIXTURE_FULLY_OPEN_FRACTION`
   are separate symbols, numerically equal today, feeding two different code paths.
6. **`fixture_fully_open` (Properties 6, 10) is an OR of two independently-tracked "active fixture"
   notions** (contact-based vs. gripper-interior-based) not disambiguated in the docs.
7. **Undocumented carrier substitution**: if a grasped object sits inside a receptacle-like
   manipulated object, the receptacle becomes the effective `object_grasped` identity.
8. **Undocumented in-flight payload exclusion** for `microwave_empty` (Property 9): the actively-
   grasped object/receptacle contents are excluded from the occupancy check while being carried.
9. ~~`CONTACT_PERSISTENCE_FRAMES = 1` is a no-op debounce at its current value across Properties
   1, 4, 5, 6~~ — **REMOVED (2026-09-01)**: the constant (and the debounce machinery that read
   it) was deleted entirely; those predicates now track their raw signal directly.
10. **`slide_path_clear` / `articulation_path_clear` (Properties 15, 17) are plain aliases** of
    `target_region_clear_*`, not the distinct swept-volume-to-end-of-travel checks the doc describes.
11. **`_skill_target_onset` (Properties 13-17) has several undocumented AND-guards**: exclusivity
    across the 5 target-based actions (only the nearest pending action can fire), mutual exclusion
    with `skill_pick_onset`/`skill_place_onset`/`object_grasped`, and "not contacting a different
    action's target" / "not contacting an unrelated object."
12. **`skill_open_close_onset`'s candidate set (Property 17) is built from a single `"openable"`
    tag**, not a conjunction with a separate `target_is_closeable` term as the doc's formula lists
    — behaviorally usually equivalent since fixture classes always tag both together, but not the
    same test.
13. **KNOWN_BUGS.md #1** (confirmed still present): `skill_pick_onset` (Property 12) never checks
    `object_is_manipulable` — can fire on non-manipulable fixture parts.
14. **KNOWN_BUGS.md #2** (confirmed still present): `skill_dump_onset` (Property 19) requires
    persistent receptacle tilt as a hard AND-gate, inverted from the doc's stated intent that
    tilt/uprightness should be irrelevant to the onset condition.
15. **`specs.py:224` vs. code**: `preconditions_satisfied_pick` (Property 12) is documented as
    including `object_upright_if_receptacle`, but the term is computed and reported separately,
    never ANDed into the actual gating formula.
16. ~~`content_stable` for liquid/solid settle (Properties 7/8) still uses plain absolute
    `_object_stable`~~ — **FIXED**: `content_stable`'s per-content raw stability now uses the
    support-relative `_object_stable_relative`, the same helper `object_settled` (Property 3)
    uses, closing this false-negative for content settling inside a receptacle that is itself
    being carried.
17. **`object_released` (Property 2)** is a plain single-frame `previously(object_grasped) and NOT
    object_grasped and gripper_is_opening` check. A `pending_release_active` latch (open the frame
    `object_grasped` falls, stay open until contact actually clears) was tried as a fix for a gap
    where `object_grasped`'s debounced fall and the object's actual physical separation
    (`check_contact(gripper, object)` reaching zero) could land on different frames — but that gap
    was specifically caused by `object_grasped` depending on `object_sync` (a velocity-based
    condition, since removed — see item #18). With `object_sync` gone, `object_grasped`'s raw
    signal is driven purely by contact/closed-finger state, the same underlying joint state
    `gripper_is_opening` reads, so the two are now tightly correlated in time and the gap is no
    longer expected to occur; the latch was reverted as unneeded complexity (see
    `CHANGES_2026-08-31.md`). Note for the future: this exact pattern — ANDing two signals with
    independent, non-identical debounce/edge timing, bridged with only a single-frame
    `previously(...)` lookback — is generic and worth checking for elsewhere if either side of
    `object_released` ever gains a velocity-based (or otherwise independently-timed) condition
    again.
18. ~~`object_grasped` briefly also required `object_sync` (relative velocity to the end effector)~~
    — **TRIED AND REVERTED**: doing so made `object_grasped_safe` (`= object_grasped and
    object_sync`) a logical tautology, since `object_grasped` would then already imply
    `object_sync` on every frame — `object_grasped_safe` would be identically equal to
    `object_grasped`, giving the `G(object_grasped -> object_grasped_safe U object_released)`
    property zero ability to distinguish "still grasped" from "still grasped but unsafe."
    `object_grasped` is bilateral contact + closed fingers only again; `object_sync` lives
    solely in `object_grasped_safe`, kept deliberately independent so the property can actually
    fire. Accepted tradeoff: `object_grasped` alone can no longer distinguish a genuine grasp
    from the gripper closing around an object without lifting it (or dragging a non-coupled
    object) — accepted as out of scope for now (a touch with both fingers closed counts as a
    grasp); `object_grasped_safe` is expected to catch resulting instability instead.
19. **`object_sync`'s linear-velocity comparison is lever-arm-corrected (2026-09-02)** — the raw
    `‖obj_vel − eef_vel‖` comparison (used since before this session, and still described as-is in
    item #18 above) legitimately differs by `ω × r` for a rigid grasp whenever the assembly rotates
    and the object's reference point is offset from the eef site — not sensor noise, so no
    threshold change could fix it without hiding real slip too. Found via the v1 output run on
    `ArrangeBreadBasket` ep0 (`object_grasped_safe` false for 259 straight frames with no real
    issue); fixed by comparing against a rotation-corrected expected velocity instead of `eef_vel`
    directly. See `_object_eef_relative_speeds`'s node above and `CHANGES_2026-08-31.md` item 11.
20. **`object_sync` prefers actual contact-point slip speed over item #19's CoM-based correction
    (2026-09-02)** — item #19 still assumes the object's whole body is rigidly locked to the eef
    site, which can spuriously flag desync for anything not perfectly rigid (contents shifting
    inside a held container, grasp compliance) even with zero real slip. `_object_contact_slip_speed`
    computes the no-slip residual directly at the actual finger/object contact point(s) from the
    simulator's contact array instead, only falling back to item #19's correction when there's no
    active contact that frame. Not yet verified against real data at time of writing — can't be
    checked against already-recorded `privileged_information_N.json` dumps (contact positions are
    live sim state, not saved), verification is the pending v2 re-run. See `CHANGES_2026-08-31.md`
    item 12.
21. **`object_sync` replaced with position-based `_object_grasp_slip` (2026-09-02)** — items 19/20
    fixed the velocity *measurement*, but a correctly-measured instantaneous velocity still can't
    tell a brief real acceleration transient (confirmed up to ~1.1 rad/s during otherwise-safe
    carrying) apart from genuinely unsafe motion. Replaced with a position-based check: at grasp
    onset, record the object's pose relative to the eef; every frame, compare the actual pose
    against where it should be if it had moved rigidly with the eef. See `CHANGES_2026-08-31.md`
    item 13 for the original (accumulated-since-onset) version's motivation.
22. **Quaternion component-order bug found and fixed (2026-09-02)** — item 21's real-data
    verification initially showed physically implausible rotations (basket ~165°, bread ~50°, with
    no matching visual motion). Root cause: `_object_orientation`/`_eef_orientation` read
    `orientation` as wxyz; the underlying data (`kitchen_ext.py`) stores it as xyzw. Fixed via
    `_xyzw_to_wxyz`. See `CHANGES_2026-08-31.md` item 14 for the diagnostic method (checking
    object/eef absolute-rotation-axis alignment) and corrected numbers.
23. **`_object_grasp_slip` changed from accumulated-since-onset to frame-to-frame (2026-09-02)** —
    item 21's original design never forgets a one-time settling shift once it happens, flagging
    every subsequent frame forever even after the object stabilizes. Fixed by overwriting the
    stored reference to the current frame's pose every frame. Trade-off, not fully resolved: this
    makes the check insensitive to slow continuous drift that never spikes in any single frame
    (confirmed: the basket's real ~19-20° swing, item 22, is entirely undetected by the
    frame-to-frame version). See `CHANGES_2026-08-31.md` item 15 for the full three-way trade-off
    discussion (velocity / accumulated-since-onset / frame-to-frame all have a gap).
24. **`object_released`'s `object_supported` fallback also requires `object_stable_relative`
    (2026-09-02)** — the fallback fired on any support contact, including a one-frame bilateral-
    contact dropout mid-carry while the object was still clearly moving (confirmed false positive:
    `ArrangeBreadBasket` ep6 frame 445, `ArrangeTea` ep0 frame 85). Fixed by requiring the object
    to actually be at rest relative to its support, not just touching it. Caveat, confirmed on the
    `ArrangeTea` case: this closes one false positive but exposes the same underlying one-frame
    bilateral-contact flicker as a *different* violation instead (`rc_grasp_remains_safe_until_
    release`, since `object_grasped` itself flickers False for that one frame). The real fix
    (eliminating the flicker in `_object_gripper_bilateral_contact`/`_object_is_grasped` itself,
    which item 1 was meant to do and mostly does, but not in every case) is not yet done. See
    `CHANGES_2026-08-31.md` item 16.
