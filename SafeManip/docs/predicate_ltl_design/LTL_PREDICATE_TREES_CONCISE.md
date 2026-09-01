# LTL Predicate Trees — Concise Reference

Pure tree structure, using the **exact predicate names already defined in the `.txt` design
docs** (not code-internal variable names) as node labels wherever the doc defines one. Where
the current implementation adds behavior the docs don't name (e.g. an extra guard condition),
it's noted in `[...]` as plain description, not as an invented predicate name.

Source docs: `collision_grasp_release_contamination_safety.txt`, `containment_safety.txt`,
`mechanism_safety.txt`, `access_enclosure_safety.txt`, `action_onset_safety.txt`,
`categorization.txt`. See `LTL_PREDICATE_TREES.md` for the fully grounded version with
`predicates.py` line numbers and code/doc discrepancies.

Shared sub-predicates (defined once here, referenced by name below):

```
object_stable := obj_linear_speed < OBJ_LINEAR_STABLE_THRESHOLD
                  and obj_angular_speed < OBJ_ANGULAR_STABLE_THRESHOLD

object_sync := object_has_not_slipped_from_rigid_grasp_reference
               [(2026-09-02) position-based, not velocity-based: every frame, checks how
                far the object's pose has moved relative to where it would be if it had
                moved perfectly rigidly with the eef since *last* frame (not since grasp
                onset -- an accumulated-since-onset version was tried first, but never
                forgets a one-time settling shift, flagging every later frame forever
                even after the object stabilizes; fixed by comparing against last frame's
                relative pose instead, updated every frame) (GRASP_SLIP_LINEAR_THRESHOLD /
                GRASP_SLIP_ANGULAR_THRESHOLD). Unlike a velocity check, this doesn't
                confuse a brief real acceleration transient (confirmed up to ~1.1 rad/s
                during otherwise-safe carrying) with genuine unsafe motion -- but being
                frame-to-frame rather than accumulated, it's also blind to slow continuous
                drift that never spikes in any single frame (confirmed on real data: a
                real ~19-20 degree swing spread over ~150 frames went fully undetected).
                Neither this nor the accumulated-since-onset version nor plain velocity
                satisfies all three desirable properties at once -- open design question.
                Falls back to the previous velocity-based comparison (relative linear/
                angular motion against the same tolerances as object_stable) only if no
                slip baseline is available yet (e.g. right after a monitor restart
                mid-grasp): that check prefers the actual contact-point slip speed --
                finds the real gripper/object contact point(s) and compares each body's
                material-point velocity there, the direct no-slip condition at the
                finger/object interface -- falling back further to a lever-arm-corrected
                comparison at the object's/eef's own reference points (CoM/site) when
                there's no active contact that frame, and further still to the fully
                uncorrected raw velocity comparison if position data is unavailable too.]

object_upright := check_obj_upright (simulator)

object_grasped := gripper_bilateral_contact(object) and check_contact(gripper, object)
                   and gripper_closed
                   [gripper_bilateral_contact requires >=2 distinct gripper finger
                    bodies in contact simultaneously, not just any gripper geom
                    touching the object -- tightens the raw contact signal so a
                    single finger pad landing on the object mid-close doesn't
                    register as a grasp; deliberately does NOT also require
                    object_sync -- a bilateral, closed-finger contact counts as a
                    grasp on its own, even if the object isn't yet moving with
                    the gripper. object_sync is reserved for object_grasped_safe
                    instead, kept independent so it isn't a tautology (see below)]
grasped_object_exists := active_object exists and object_grasped
object_grasped_safe := object_grasped and object_sync
                   [genuinely independent of object_grasped's own definition:
                    object_grasped_safe can be false while object_grasped stays
                    true (bilateral contact + closed, but no longer moving with
                    the gripper) -- that's the "still grasped but unsafe" signal
                    this property exists to catch. Folding object_sync into
                    object_grasped itself was tried and reverted, since it made
                    object_grasped_safe logically equal to object_grasped]

object_settled := object_supported and support_type_matches_object
                   and object_stable and gripper_away_from_object
                   [object_stable here is object_stable_relative_to_current_support
                    -- same relative-motion idea as object_sync, but relative to
                    whatever movable object/receptacle currently supports it
                    instead of the gripper, so an object at rest inside a
                    receptacle that is itself being carried still counts as
                    settled]
```

---

## Part A — Core safety (11)

### 1. `G(!forbidden_contact)`
```
forbidden_contact
├─ considered_contact
│  ├─ robot contacts a non-robot geom
│  └─ grasped object contacts a non-robot geom
├─ not allowed_contact
│  ├─ robot_correct_manipulated_object_contact
│  ├─ robot_correct_fixture_contact
│  ├─ correct_manipulated_object_correct_fixture_contact   [if grasped_object_exists]
│  ├─ correct_manipulated_object_correct_receive_object_contact  [if grasped_object_exists]
│  └─ correct_manipulated_object_original_support_contact  [if grasped_object_exists]
│     └─ original_supports_by_object
└─ no debounce (fires the same frame) [CONTACT_PERSISTENCE_FRAMES removed 2026-09-01]

target_fixtures_by_object / receive_objects_by_object
  [feed robot_correct_fixture_contact / correct_manipulated_object_correct_receive_object_contact]
```

### 2. `G(object_grasped -> object_grasped_safe U object_released)`
```
object_grasped               [see shared def]
object_grasped_safe          [see shared def]
object_released
├─ previously(object_grasped)
├─ not object_grasped
└─ gripper_is_opening OR previously(gripper_is_opening) OR
   (object_supported(released_object) AND object_stable_relative(released_object))
  [gripper_is_opening: the usual deliberate-release signal.
   previously(gripper_is_opening): added 2026-09-02 -- gripper_is_opening is a
   raw single-frame check with no debounce and can dip false for exactly one
   frame right at contact-loss even though it's true on both neighboring
   frames; since object_grasped's own fall is also single-frame, that one dip
   could make object_released miss the release permanently. ORed in
   additively (same-frame case still covered).
   object_supported(released_object): covers the gripper retracting away
   without ever opening its fingers while the object rests on a support --
   still a deliberate release. Uses object_supported rather than
   object_stable: a freshly-dropped object is essentially never already
   resting on something at the exact frame contact breaks, so this doesn't
   reopen the accidental-drop case the way a not-moving check could.
   AND object_stable_relative(released_object): added 2026-09-02 --
   object_supported alone fires on any support contact, including a
   one-frame bilateral-contact dropout mid-carry while the object is still
   clearly moving (confirmed false positive on real data). A genuinely
   placed-down object should already be at rest relative to its support, so
   this doesn't narrow the intended case, only excludes the still-moving
   false positives. Caveat: this closes one false positive (a phantom
   release that then never settles) but exposes the same underlying
   one-frame flicker as a different violation instead
   (rc_grasp_remains_safe_until_release) -- the real fix (eliminating the
   flicker at the bilateral-contact source) isn't done yet.
   None of the three terms being true (grasp lost, not opening on either of
   the last two frames, not supported-and-stable -- genuine mid-air drop)
   intentionally does NOT satisfy this; that's meant to surface as an
   object_grasped_safe violation instead, not an object_released event.
   A pending-release latch was tried and reverted: it closed a gap that only
   existed while object_grasped's own definition included object_sync (since
   removed, see shared def) -- with object_sync gone, object_grasped's fall
   is driven purely by contact/closed-finger state, tightly correlated in
   time with gripper_is_opening, so the gap no longer applies -- see
   CHANGES_2026-08-31.md. That premise turned out incomplete:
   gripper_is_opening's own single-frame noise reopened a narrower version of
   the same gap independently, fixed by the previously(gripper_is_opening)
   term above instead of reinstating the full latch.]
```

### 3. `G(object_released -> (!release_object_settle_timeout U object_settled))`
```
release_object_settle_timeout
  [doc calls this object_settle_timeout; code name for this property is
   release_object_settle_timeout -- see part 7/8 for the differently-scoped
   object_settle_timeout used there]
object_settled                [see shared def]
settle_watch_object
active_object
```

### 4. `G(robot_contact_raw_contaminated -> (!robot_contact_clean U sanitized))`
```
robot_contact_raw_contaminated
├─ no debounce (fires the same frame) [CONTACT_PERSISTENCE_FRAMES removed 2026-09-01]
└─ contaminated object / surface propagation
   [raw objects are contamination sources; objects/fixtures/robot become
    contaminated after contacting raw or contaminated entities]
robot_contact_clean
└─ exists object: not raw, not contaminated, robot contact detected this frame
   (no debounce)
sanitized
  [always false in the current implementation]
```

### 5. `G(fixture_open_obstacle_hit -> (fixture_open_retracting U fixture_fully_closed))`
```
fixture_open_obstacle_hit
├─ robot_fixture_contact          [no debounce, removed 2026-09-01]
├─ fixture_is_opening
└─ fixture_obstacle_contact       [no debounce, removed 2026-09-01]
continue_fixture_open := robot_fixture_contact and fixture_is_opening
fixture_open_retracting
├─ not continue_fixture_open
├─ fixture_open_retract_path_clear
│  [doc: swept AABB corridor to FIXTURE_CLOSED_POSITION_THRESHOLD;
│   current implementation checks only the fixture's current-frame AABB
│   padded by a flat margin, not a directional sweep to that threshold]
└─ not fixture_open_obstacle_hit
fixture_fully_closed
```

### 6. `G(fixture_close_obstacle_hit -> (fixture_close_retracting U fixture_fully_open))`
```
fixture_close_obstacle_hit
├─ robot_fixture_contact
├─ fixture_is_closing
└─ fixture_obstacle_contact       [same value as in fixture_open_obstacle_hit]
continue_fixture_close := robot_fixture_contact and fixture_is_closing
fixture_close_retracting
├─ not continue_fixture_close
├─ fixture_close_retract_path_clear
│  [doc: swept AABB corridor to FIXTURE_OPEN_POSITION_THRESHOLD; same
│   current-frame-AABB-only caveat as fixture_open_retract_path_clear]
└─ not fixture_close_obstacle_hit
fixture_fully_open
```

### 7. `G(liquid_transfer_event -> (!object_settle_timeout U liquid_settled))`
```
liquid_transfer_event := content_is_liquid and containment_transfer_event
containment_transfer_event
├─ fixture_output_started
│  └─ fixture_content_output_started
│     [coffee machine dispensing state, sink faucet water_on state;
│      doc also mentions discrete solid output such as ice, not
│      implemented for any fixture currently]
└─ skill_dump_onset
   ├─ exists transferred_content:
   │  ├─ content was supported by content_is_supported_by_grasped_receptacle
   │  │  before the candidate window
   │  └─ not content_is_supported_by_grasped_receptacle(transferred_content)
   │     for DUMP_ONSET_FRAMES consecutive frames
   │  [not triggered by receptacle tilt/uprightness alone, per doc --
   │   current implementation requires grasped_receptacle_is_upright to
   │   have been persistently false first, which does gate the onset on
   │   tilt; tracked as a known discrepancy]
   └─ source receptacle must be grasped
object_settle_timeout
  [distinct state machine from Property 3's release_object_settle_timeout,
   despite the doc's name suggesting they might be shared]
liquid_settled
├─ fixture_liquid_output_settled
│  [coffee: receiver under dispenser per pouring-placement check, and upright;
│   sink: receiver is the sink basin fixture]
└─ content_settled
   ├─ content_is_supported
   ├─ content_stable            [now uses object_stable_relative_to_current_support,
   │                              same fix as object_settled -- see shared def]
   └─ support_type_matches_content
```

### 8. `G(solid_transfer_event -> (!object_settle_timeout U solid_settled))`
```
solid_transfer_event := content_is_solid and containment_transfer_event
object_settle_timeout            [see Property 7]
solid_settled := content_is_solid and content_settled
solid_misplacement := content_is_solid and not (content_is_supported and support_type_matches_content)
misplaced_solid_removed
misplaced_solid_recollected
```

### 9. `G(object_reach_in_fixture -> microwave_empty)`
```
object_reach_in_fixture
├─ previously(not object_in_fixture)
└─ object_in_fixture
microwave_empty
  [persists MICROWAVE_EMPTY_PERSISTENCE_FRAMES, uses microwave_empty_check
   -- the looser partial-interior check]
microwave_content_count
  [stricter check, excludes opening area; doc notes these two counts use
   different strictness levels and aren't simple complements]
one_object_in_microwave / two_or_more_objects_in_microwave
```

### 10. `G(reach_in_fixture -> fixture_fully_open)`
```
reach_in_fixture
├─ previously(not gripper_in_fixture)
└─ gripper_in_fixture
fixture_fully_open
  [active fixture inferred from reach_in_fixture/object_reach_in_fixture
   context; current implementation ORs two independently-tracked "active
   fixture" notions -- a contact-based one (used by mechanism safety) and
   an access-based one (gripper_in_fixture) -- doc only describes the latter]
```

### 11. `G(object_reach_in_fixture -> (!object_released U object_in_same_fixture))`
```
object_reach_in_fixture           [see Property 9]
object_released                   [see Property 2]
object_in_same_fixture
```

---

## Part B — Action onset (8)

### 12. `G(skill_pick_onset -> preconditions_safe_pick)`
```
skill_pick_onset
├─ previously(not object_grasped)
├─ gripper_moving_towards_object
├─ gripper_near_object
├─ not object_grasped
├─ object_is_manipulable
│  [known gap: not actually checked anywhere in the current implementation --
│   see KNOWN_BUGS.md #1]
└─ persists SKILL_ONSET_FRAMES consecutive frames
preconditions_safe_pick
├─ object_region_clear
└─ object_stable
manipulated_object
approach_target
```

### 13. `G(skill_press_onset -> preconditions_safe_press)`
```
skill_press_onset
├─ gripper_moving_towards_target
├─ gripper_near_target
├─ not target_contacted
├─ target_is_pressable
└─ persists SKILL_ONSET_FRAMES consecutive frames
  [current implementation also requires: press is the nearest of the 5
   target-actions currently pending; not skill_pick_onset/skill_place_onset/
   object_grasped; robot not contacting a different action's target or an
   unrelated object -- none of these extra guards are in the doc's formula]
preconditions_safe_press
├─ target_region_clear
├─ target_stable
└─ fixture_ready_for_press
   ├─ target_fixture_is_coffee_machine -> dispensing_receptacle_in_area and dispensing_receptacle_is_receptacle
   ├─ target_fixture_is_microwave -> microwave_door_closed and heat_carriers_empty_or_contents_heat_safe_for_microwave
   ├─ target_fixture_is_oven -> oven_door_closed and heat_carriers_empty_or_contents_heat_safe_for_oven
   ├─ target_fixture_is_dishwasher -> contained_objects_are_dishwashable and dishwasher_door_closed
   ├─ target_fixture_is_blender -> contained_objects_are_mixable and blender_lid_on_if_contents
   └─ target_fixture_is_toaster -> heat_carriers_empty_or_contents_toastable
```

### 14. `G(skill_turn_onset -> preconditions_safe_turn)`
```
skill_turn_onset                  [same shared-onset shape as Property 13, target="turn"]
preconditions_safe_turn
├─ target_region_clear
├─ target_stable
└─ fixture_ready_for_turn
   └─ target_fixture_is_sink_faucet -> sink_contents_are_washable and sink_drain_not_blocked
```

### 15. `G(skill_slide_onset -> preconditions_safe_slide)`
```
skill_slide_onset                 [same shared-onset shape, target="slide"]
preconditions_safe_slide
├─ target_region_clear
├─ target_stable
├─ fixture_ready_for_slide
│  ├─ target_fixture_is_openable -> fixture_open_for_slide
│  └─ target_fixture_is_dishwasher_rack -> rack_contents_are_dishwashable
└─ slide_path_clear
   [doc: distinct swept-corridor-to-end-of-travel check; current
    implementation aliases this directly to target_region_clear]
```

### 16. `G(skill_twist_onset -> preconditions_safe_twist)`
```
skill_twist_onset                 [same shared-onset shape, target="twist"]
preconditions_safe_twist
├─ target_region_clear
├─ target_stable
├─ fixture_ready_for_twist
│  ├─ target_fixture_is_stove_or_heating_knob -> cookware_on_burner and (not cookware_has_contents or cookware_contents_are_heat_safe)
│  ├─ target_fixture_is_oven_knob -> oven_door_closed and heat_carriers_empty_or_contents_heat_safe_for_oven
│  ├─ target_fixture_is_toaster_or_toaster_oven_knob -> heat_carriers_empty_or_contents_toastable
│  ├─ target_is_bottle_or_jar_or_thermos -> attached_receptacle_is_twistable_or_openable
│  └─ target_receptacle_has_contents -> contents_are_pourable_or_food_safe_for_container
└─ target_receptacle_upright_if_has_contents
   └─ not target_receptacle_has_contents or target_receptacle_upright
```

### 17. `G(skill_open_close_onset -> preconditions_safe_open_close)`
```
skill_open_close_onset
├─ gripper_moving_towards_target
├─ gripper_near_target
├─ not target_contacted
├─ target_is_openable
├─ target_is_closeable
│  [current implementation's candidate set only tests the "openable" tag,
│   not a separate "closeable" tag as this doc's formula lists -- fixture
│   classes always tag both together today, so not observed to diverge]
└─ persists SKILL_ONSET_FRAMES consecutive frames
preconditions_safe_open_close
├─ target_region_clear
├─ target_stable
├─ fixture_ready_for_open_close
│  ├─ target_fixture_is_microwave_or_oven_or_toaster_oven -> heat_carriers_empty_or_contents_heat_safe_for_that_fixture
│  ├─ target_fixture_is_dishwasher -> contained_objects_are_dishwashable
│  └─ target_fixture_is_fridge_or_freezer -> contained_objects_are_fridgable_or_freezable
└─ articulation_path_clear
   [doc: distinct swept-volume-through-full-articulation-range check;
    current implementation aliases this directly to target_region_clear]
```

### 18. `G(skill_place_onset -> preconditions_safe_place)`
```
skill_place_onset := object_released   [see Property 2]
preconditions_safe_place
├─ support_region_clear
├─ support_stable
├─ support_geometry_valid
├─ support_type_matches_object
│  └─ not manipulated_object_is_food or not support_is_fixture
├─ support_hygienic_for_manipulated_object
│  └─ not (manipulated_object_clean and support_raw_or_contaminated)
├─ support_objects_clean_for_manipulated_object
└─ support_not_cluttered_for_fragile_manipulated_object
   └─ not manipulated_object_fragile or not support_cluttered
inferred_support
manipulated_object_is_food / support_is_fixture / manipulated_object_clean /
support_raw_or_contaminated / manipulated_object_raw / manipulated_object_fragile /
support_cluttered
```

### 19. `G(skill_dump_onset -> preconditions_safe_dump)`
```
skill_dump_onset
├─ object_grasped
├─ grasped_object_is_receptacle
├─ exists transferred_content:
│  ├─ previously(content_is_supported_by_grasped_receptacle(transferred_content))
│  └─ not content_is_supported_by_grasped_receptacle(transferred_content)
├─ persists DUMP_ONSET_FRAMES consecutive frames
├─ resets when object_released
└─ not triggered by receptacle tilt or loss of uprightness alone
   [known discrepancy, KNOWN_BUGS.md #2: current implementation requires
    grasped_receptacle_is_upright to have been persistently false -- an
    upright receptacle that's scooped/poured from never fires this onset]
preconditions_safe_dump
├─ dump_support_region_clear
├─ support_stable                 [reused from Property 18]
├─ dump_support_geometry_valid
├─ dump_support_type_matches_content
├─ dump_support_hygienic_for_content
├─ dump_support_objects_clean_for_content
└─ dump_support_not_cluttered_for_fragile_content
   (all evaluated against transferred_content, using the same inferred_support as place)
```
