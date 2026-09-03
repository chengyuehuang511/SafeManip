# `rc_dropped_object_was_released`

```
G(object_dropped -> (object_released | (!object_left_gripper U object_grasped)))
Recovery: G((object_dropped & !object_released) -> F(object_grasped | object_left_gripper))
```
Shape: **instant-with-until-escape**. Predicates: `object_dropped`, `object_released`,
`object_grasped`, `object_left_gripper`.

The other half of the 2026-09-02 split (see `02_grasp_remains_synced_until_dropped.md`'s guide
and `CHANGES_2026-09-02.md`) — the one that took 4 rounds to get right. This is the property to
read if you want the fullest worked example of the whole debugging methodology in
`README.md`, since every category of bug in that README's Step 6 showed up here at some point.

## The 4 rounds, briefly (full detail in `CHANGES_2026-09-02.md`)

1. `F(object_grasped)` escape — cross-object misattribution (a *different* object's later grasp
   satisfied it).
2. `!(object_stable_relative & object_supported) U object_grasped` — fixed #1 via `active_object`
   timing, but `object_stable_relative`/`object_supported` are undebounced, so a one-frame
   settle blip broke the until early.
3. `object_left_gripper` (raw mesh contact) replacing the settle check — fixed #2, but raw
   contact flickers too (confirmed via direct MuJoCo `sim.data.contact` inspection: zero
   gripper-object contacts for one frame, mid-recatch).
4. `object_left_gripper` redefined as AABB overlap (`_gripper_aabb()` vs.
   `_object_contact_aabb()`, via `_aabb_intersects()`) — the version above. Verified against
   `ArrangeBreadBasket` episodes 0, 1, 4, 6, 7.

## What to check if this ever looks wrong again

- **First, which of the two disjuncts resolved it.** `object_released` (clean, gripper-opening
  evidence) resolving on the trigger frame itself is a completely different code path from the
  until-escape resolving later — check `object_released`'s value at the exact `object_dropped`
  frame first.
- **If it went through the until-escape, check `object_left_gripper`'s trace frame by frame**
  from the trigger onward, alongside `active_object`. It should read `False` continuously while
  the object is still within the gripper's AABB span, `True` once genuinely separated. A
  same-frame or few-frames-later flicker back to the wrong value at the trigger instant itself
  is the signature of the round-3 bug recurring (raw-contact-style flicker) — but this
  predicate is now AABB-based, so that specific failure mode shouldn't recur; if you see it,
  first re-confirm `object_left_gripper`'s current definition hasn't regressed back to a
  contact-based one.
- **`object_left_gripper` deliberately does NOT go through the general `_object_aabb()` helper**
  — it calls `_object_contact_aabb(obj_name)` directly, because `_object_aabb()`'s
  first-choice source (`_object_ou_bbox_aabb`) is confirmed wrong for at least `basket` while
  held (see `KNOWN_BUGS.md` #11). If you ever see this predicate get refactored to use
  `_object_aabb()` "for consistency," that's very likely reintroducing a bug, not cleaning one
  up — check the object's actual AABB against its known position first.
- **Cross-object misattribution risk is structurally low here now**, not eliminated by
  construction: the until resolves via *timing* (the same object stays `active_object` until the
  robot moves on), not via a dedicated per-object identity check. If a task has the robot
  return to the just-dropped object's location for an unrelated reason before it settles, this
  could in principle misfire again — hasn't been observed, but hasn't been proven impossible
  either.
- **`predicates` list**: `object_dropped`, `object_released`, `object_grasped`,
  `object_left_gripper` must all be listed in `specs.py`'s spec for this property (recovery_ltl
  references the same set) — see README #6 for what happens if one's missing.
- **`specs.py`'s `"ltl"` field vs. `repeated_violation_monitor.py`'s `main_ltl`** — check these
  are still textually identical (README Step 1's warning applies especially hard here, since
  this property's formula has been rewritten 4 times already).

## Viewer-specific note

The occurrence breakdown for this property needed its own fix (`compute_occurrences`'s
`pattern == "instant"` branch actually simulating the until/escape, `eventual_separation`
tracking for window-sizing) — see `CHANGES_2026-09-02.md` section 8. If the breakdown table
ever looks wrong again (an occurrence that resolved shown as violated, or a signal that "never"
transitions in the displayed window), check the window bounds first before doubting the
predicate itself.
