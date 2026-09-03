# `rc_liquid_transfer_eventually_settles` / `rc_solid_transfer_eventually_settles`

```
rc_liquid_transfer_eventually_settles: G(liquid_transfer_event -> (!object_settle_timeout U liquid_settled))
                                        Recovery: none
rc_solid_transfer_eventually_settles:  G(solid_transfer_event -> (!object_settle_timeout U solid_settled))
                                        Recovery: G(solid_misplacement -> F(misplaced_solid_removed | misplaced_solid_recollected))
```
Shape: **until**, both. Predicates (solid): `solid_transfer_event`, `object_settle_timeout`,
`solid_settled`, plus `solid_misplacement`/`misplaced_solid_removed`/
`misplaced_solid_recollected` (recovery-only atoms — must still be in the `predicates` list per
README #6, even though they don't appear in `main_ltl`).

## Important: the two properties' recovery formulas are NOT parallel

`rc_liquid_transfer_eventually_settles` has no `recovery_ltl` configured (`"none"` in
`repeated_violation_monitor.py`) — no secondary bookkeeping at all for this one.
`rc_solid_transfer_eventually_settles`'s recovery does **not** mirror its own main formula's
resolve condition (`solid_settled`) — it's about a *different* concern
(`solid_misplacement` → was the misplaced solid removed or recollected), already correctly
`F(...)`-wrapped. If you're checking whether "recovery" means "the solid eventually settled,"
that's the wrong question for this property's recovery bookkeeping specifically — it's asking
whether a *misplacement* got cleaned up, a narrower and different condition.

## What to check
- `object_settle_timeout` is shared between both (same name, generic) — confirm which transfer
  event (`liquid_transfer_event` vs `solid_transfer_event`) actually started the countdown
  you're looking at if debugging a specific timeout.
- If you want "did the solid eventually settle" as its own recovery question (parallel to how
  `rc_released_object_eventually_settles` works), that's not what's currently implemented for
  `rc_solid_transfer_eventually_settles` — it would need its own new recovery formula, not a fix
  to the existing one (which is answering a legitimately different question already).
- Neither of these properties' main formulas or predicates were touched or deeply re-verified
  during the 2026-09-02 session — this guide reflects what's in the code as of that date, not a
  confirmed-correct verification pass.

## Known-good / open
Main `until` formulas not suspected of a bug. `rc_liquid_transfer_eventually_settles` simply has
no recovery bookkeeping at all (not necessarily a bug — may be intentional if it wasn't judged
useful). `rc_solid_transfer_eventually_settles`'s recovery is correctly shaped but answers a
narrower question than "did it settle" — worth confirming that's actually the intended design
before assuming it's a gap.
