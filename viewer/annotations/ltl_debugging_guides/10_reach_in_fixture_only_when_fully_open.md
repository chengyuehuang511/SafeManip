# `rc_reach_in_fixture_only_when_fully_open`

```
G(reach_in_fixture -> fixture_fully_open)
Recovery: none
```
Shape: **instant**. Predicates: `reach_in_fixture`, `fixture_fully_open`. No recovery
bookkeeping configured (`"none"`, confirmed in `repeated_violation_monitor.py`) — makes sense
for a pure instant shape where each occurrence either resolves on its own trigger frame or
doesn't; there's no multi-frame "recovery window" concept to track the way an until-shape
property has.

## What to check
- `reach_in_fixture` is the general form of `object_reach_in_fixture` (used by
  `rc_microwave_single_object_until_empty`/`rc_fixture_placement_release_after_internal_support`)
  — check whether it's meant to cover *any* gripper/object entering *any* openable fixture
  (cabinet, fridge, microwave, drawer, ...) or a narrower set; if a violation seems to fire for
  an unexpected fixture type, that's the first thing to check.
- `fixture_fully_open` is presumably threshold-based on joint position (same family as
  `fixture_fully_closed`/etc. used by the obstacle-retract properties) — if a violation looks
  wrong for a fixture that appears visually open, check the actual joint value against whatever
  threshold defines "fully."

## Known-good / open
Not deeply re-verified during the 2026-09-02 session; no known open issues, but also no fresh
confirmation beyond what's in `specs.py`'s own comments.
