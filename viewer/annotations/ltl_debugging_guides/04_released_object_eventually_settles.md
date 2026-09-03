# `rc_released_object_eventually_settles`

```
G(object_released -> (!release_object_settle_timeout U object_settled))
Recovery: object_settled   <-- BARE ATOM, see below
```
Shape: **until** (main). Predicates: `object_released`, `object_settled`,
`release_object_settle_timeout`.

## Known bug: bare-atom `recovery_ltl` (KNOWN_BUGS.md #10, not yet fixed)

`recovery_ltl = "object_settled"` is a bare atomic proposition with no temporal operator.
Per LTLf semantics (see README's Step 6 #1), this only checks whether `object_settled` is true
at the exact frame recovery-tracking *starts* — not "eventually." Since recovery only starts
once the main formula has already failed (timed out), `object_settled` is essentially always
false at that exact frame (that's *why* it failed). Confirmed empirically: 0% recovery rate,
corpus-wide, for this property (paired with `rc_grasp_remains_safe_until_release` in the
original count — see KNOWN_BUGS.md #10's numbers). Concrete case: `ArrangeBreadBasket` ep6,
violates at frame 445, `object_released` genuinely settles later at frame 601, but the episode
still reports `recovered: false, end_frame: 624` (the episode's last frame).

**The fix, if you pick this up**: wrap in `F(...)`: `recovery_ltl = "F(object_settled)"`.

## What else to check

- `object_settled`'s real definition is an AND of 4 components computed for `settle_obj_name`
  (the object actually awaiting settle), **not** `active_object`/`obj_name` — see
  `object_supported_settle`/`object_support_type_matches_any_settle`/
  `object_stable_relative_settle` (exported specifically so the viewer's breakdown can show
  `object_settled`'s *real* components, added 2026-09-01/02 — see `CHANGES_2026-08-31.md` item
  3 and `predicate_derive.py`). If a violation explanation mentions "not settled" but the
  top-level `object_supported`/`object_stable_relative` (the `active_object`-scoped versions)
  look fine, you're looking at the wrong object's data — check the `_settle`-suffixed versions
  instead.
- `release_object_settle_timeout` is a countdown gate (`SETTLE_TIMEOUT_FRAMES`) — if a violation
  seems to fire "too early," check whether the object was actually still settling within a
  reasonable window and the timeout constant itself is just too tight for that task/object, vs.
  a genuine failure to ever settle.
- `settle_obj_name`/`awaiting_settle` latch onto the object at the moment `object_released`
  fires and hold onto *that* object specifically (not whatever `active_object` becomes later) —
  this is the existing mechanism that avoids the cross-object misattribution problem for this
  property (unlike `rc_dropped_object_was_released`'s earlier rounds, which had to work around
  not having this). If you're tempted to generalize/replace `settle_watch_object`, this property
  is one of the things that would need re-verifying afterward — this was explicitly discussed
  and deferred to a future session.

## Known-good
The main `until` formula itself is correct; only `recovery_ltl` is broken (secondary bookkeeping
only — doesn't affect the primary violated/satisfied verdict).
