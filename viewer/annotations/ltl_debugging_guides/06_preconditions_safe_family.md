# The `*_preconditions_safe` family (8 properties)

```
rc_pick_preconditions_safe:        G(skill_pick_onset -> preconditions_satisfied_pick)
rc_place_preconditions_safe:       G(skill_place_onset -> preconditions_satisfied_place)
rc_press_preconditions_safe:       G(skill_press_onset -> preconditions_satisfied_press)
rc_turn_preconditions_safe:        G(skill_turn_onset -> preconditions_satisfied_turn)
rc_slide_preconditions_safe:       G(skill_slide_onset -> preconditions_satisfied_slide)
rc_twist_preconditions_safe:       G(skill_twist_onset -> preconditions_satisfied_twist)
rc_open_close_preconditions_safe:  G(skill_open_close_onset -> preconditions_satisfied_open_close)
rc_dump_preconditions_safe:        G(skill_dump_onset -> preconditions_satisfied_dump)
```
Shape: **instant**, all 8. Each `preconditions_satisfied_*` is a composite AND of several
sub-checks (path-clear, target/support stable, geometry-valid, type-matched, etc. — the exact
set varies per skill, see each property's own `description` in `specs.py`).

## What to check (applies to all 8)

- **Use the viewer's auto-derived `children` breakdown, don't hand-decompose.**
  `predicate_derive.py` AST-parses each `preconditions_satisfied_*` function and derives its
  real AND-components for `server.py`'s `PROPERTY_META[...]["children"]` — this is kept
  automatically in sync with the actual code, whereas hand-documenting the exact sub-condition
  list here would drift the moment the function changes. When a violation fires, check the
  viewer's breakdown (or `predicate_derive.py`'s output directly) for which specific
  sub-component was false, rather than re-deriving it from the top-level boolean alone.
- **Instant shape means the check only matters on the exact onset frame** — if a violation
  looks like it should have been caught but wasn't, check whether `skill_*_onset` actually fired
  on the frame you expect (onset detection has its own debounce/candidate-tracking machinery,
  e.g. `skill_press_onset_candidate_count`/`skill_dump_onset_fired_object` in `monitor_state` —
  worth checking directly if onset timing itself looks off, separately from whether
  preconditions held).
- **`_object_stable`/`_object_stable_relative` mismatches** (KNOWN_BUGS.md #9, not yet fixed):
  `_support_stable()` (used by `preconditions_satisfied_place`/`_dump`) and `_target_stable()`
  (used by the press/turn/slide/twist/open_close family) both still use world-frame
  `_object_stable`/`persistent_object_stable_by_name`, not the relative version. This means
  placing/stacking/acting onto a support or target that's itself currently being carried (still
  in-hand) can spuriously fail with "support/target not stable" purely from the support's own
  world-frame motion, not genuine instability — same false-positive shape as the already-fixed
  `object_settled` case (see `CHANGES_2026-08-31.md` item 3). If a precondition violation's
  explanation says "not stable" and the referenced object/support looks like it was actually
  being carried at that moment (check `active_object`/recent `raw_grasped_objects`), suspect
  this before suspecting anything else.
- **`_object_aabb()`'s bounding-box bug** (KNOWN_BUGS.md #11): several of the
  region-blocker/support-validity helper functions these preconditions ultimately depend on
  (`_object_region_blockers`, `_support_region_blockers`, `_support_geometry_valid`,
  `_dump_support_region_blockers`, `_dump_support_geometry_valid`, etc.) call the general
  `_object_aabb()` helper, whose default source is confirmed wrong for at least one object
  (`basket`) while it's actively held. **Unconfirmed** whether this actually causes wrong
  preconditions verdicts in practice (would need its own investigation per property), but if a
  "blocked path" or "invalid geometry" violation looks physically wrong for an object that was
  being carried at the time, this is a plausible root cause to check first — compare the
  object's AABB the code is using against its real position/`eef_position` the way
  `KNOWN_BUGS.md` #11 describes.

## Known-good / open
The formulas themselves (instant shape) are straightforward and not suspected of any LTL-level
bug — all known open issues for this family are inside the `preconditions_satisfied_*`
sub-predicates (KNOWN_BUGS.md #9 and, unconfirmed, #11), not the top-level property definitions.
