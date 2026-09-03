# `rc_released_object_eventually_settles`

```
G(object_released -> (!release_object_settle_timeout U object_settled))
Recovery: F(object_settled | release_object_settle_timeout)
```
Shape: **until** (main). Predicates: `object_released`, `object_settled`,
`release_object_settle_timeout`.

## `recovery_ltl` fix, 2026-09-03 (was broken two different ways in sequence)

**Round 1 (original bug, KNOWN_BUGS.md #10 — fixed)**: `recovery_ltl = "object_settled"` was a
bare atomic proposition with no temporal operator. Per LTLf semantics (see
`recovery-ltl-design` skill / README Step 6), this only checks whether `object_settled` is true
at the exact frame recovery-tracking *starts* — not "eventually." Since recovery only starts
once the main formula has already failed (timed out), `object_settled` is essentially always
false at that exact frame (that's *why* it failed). Confirmed empirically: 0% recovery rate,
corpus-wide, for this property (paired with `rc_grasp_remains_safe_until_release` in the
original count — see KNOWN_BUGS.md #10's numbers). Concrete case: `ArrangeBreadBasket` ep6,
violates at frame 395, `object_settled` genuinely settles later at frame 399, but the (old)
episode still reported `recovered: false, end_frame: <episode's last frame>`.

**Round 2 (a new bug found while fixing round 1)**: the natural-looking first fix,
`G(object_released & !object_settled -> F(object_settled | release_object_settle_timeout))`,
turned out to *also* be broken — recovery only starts evaluating after `in_violation` is
already `True` (i.e. after the main formula's trap is confirmed), by which point
`object_released` (an edge, true for exactly one frame) has already reverted to `False` and
never fires again in that sub-trace. The antecedent can never match again, making the whole
`G(...)` vacuously true from the very first frame recovery checks — confirmed both by an
isolated `LTLfDFA` test and against ep6's real trace (accepting already at frame 395/396, well
before `object_settled`'s real transition at 399).

**Round 3 (final "recovery vs. resume" fix, 2026-09-03, following
`rc_dropped_object_was_released`'s same decision)**: dropped the antecedent, and settled on
`recovery_ltl = "F(object_settled)"` for a moment — but then, following the same reasoning
already applied to the drop property, added `release_object_settle_timeout` back into the
escape: `F(object_settled | release_object_settle_timeout)`. This atom is `release_object_settle_timeout`
(not the unrelated `object_settle_timeout`, used only by the liquid/solid containment-transfer
properties — a naming mix-up worth double-checking every time, see `specs.py`'s predicates list
for this property). Confirmed via a full-trace replay (an isolated single-step test gave a
misleading result the first time — always replay the real trace, not just one frame) that
`release_object_settle_timeout` is always `True` at the exact frame this until's own trap
confirms (ep6: both flip together at frame 395) — the exact same tautological-escape-term
situation `object_left_gripper` was for `rc_dropped_object_was_released`. So this, too, is a
deliberate "resume tracking the instant the main formula's own trap-defining condition fires"
signal, not a genuine "did it actually settle" check — `object_released` here is likewise a
one-shot past event no later condition can undo, so the same "recovery vs. resume" framing
applies. Verified: resolves instantly (395→396) on ep6, matching
`rc_dropped_object_was_released`'s pattern exactly. If you want the genuinely meaningful "how
long did it actually take to settle" number, it's in `predicate_breakdown.occurrences`, not
here. See `CHANGES_2026-09-02.md`'s recovery_ltl-design section and the `recovery-ltl-design`
skill for the general rules established along the way (vacuous antecedents, tautological escape
terms, when `G(edge -> F(...))` is and isn't safe).

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
The main `until` formula itself is correct. `recovery_ltl` is now fixed too
(`F(object_settled | release_object_settle_timeout)`, verified against ep6) — no known open
issues as of 2026-09-03. Note `recovered`/`duration_frames` for this property carry essentially
no discriminating information by design (resolves in ~1-2 frames for nearly every violation,
unconditionally) — that's the same accepted tradeoff as `rc_dropped_object_was_released`, not a
bug.
