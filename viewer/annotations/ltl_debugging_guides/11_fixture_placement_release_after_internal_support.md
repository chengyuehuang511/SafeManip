# `rc_fixture_placement_release_after_internal_support`

```
G(object_reach_in_fixture -> (!object_released U object_in_same_fixture))
Recovery: none
```
Shape: **until**. Predicates: `object_reach_in_fixture`, `object_released`,
`object_in_same_fixture`. No recovery bookkeeping configured (`"none"`) — unlike most other
until-shape properties, which do have some recovery formula (even if buggy); worth confirming
this is intentional (the obligation here, "don't release," is arguably not the kind of thing
that has a meaningful post-hoc recovery) rather than simply not yet written.

## What to check
- **`object_released` here is the obligation being held false**, not the resolve condition — the
  property is violated if the object gets released *before* `object_in_same_fixture` becomes
  true, i.e. releasing partway into the fixture, not fully inside it yet. Don't confuse this with
  `rc_dropped_object_was_released`'s very different use of `object_released` (there, it's one of
  the ways the property can be *satisfied*).
- `object_in_same_fixture` needs to correctly identify "strictly inside the same fixture
  interior" the object reached into — if this shares any machinery with
  `_object_inside_fixture_interior`/`_gripper_inside_fixture_interior` (both use `_object_aabb`-
  style AABB containment checks), the `_object_aabb()` bug in KNOWN_BUGS.md #11 is a plausible
  thing to check if a violation looks like it fired for an object that visually looks like it
  reached the target fixture correctly.
- Same `reach_in_fixture`-family caveat as `10_reach_in_fixture_only_when_fully_open.md` — check
  which fixture types `object_reach_in_fixture` is actually scoped to.

## Known-good / open
Not deeply re-verified during the 2026-09-02 session. The `_object_aabb()` connection above is
a plausible-but-unconfirmed risk, not a confirmed bug for this specific property.
