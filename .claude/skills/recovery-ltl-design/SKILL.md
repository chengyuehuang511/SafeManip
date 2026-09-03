---
name: recovery-ltl-design
description: Design or debug a `recovery_ltl` formula for `RepeatedViolationMonitor` (SafeManip/monitor/repeated_violation_monitor.py). Use whenever writing a new `build_repeated_*_monitor` function, whenever a property's `repeated_violation_episodes` shows a suspicious `recovered` rate (always true, always false, or resolves instantly at the same frame the violation starts), whenever the user asks to "fix the recovery" for a property, or whenever deciding what `recovery_ltl` should even mean for a property built around a one-shot/edge-triggered bad event. Push back if asked to just wrap something in `F(...)` without checking for the two structural bugs below first -- a formula can look intuitively right and still be vacuously satisfied for a reason that has nothing to do with its literal content.
---

# Recovery LTL Design

`recovery_ltl` is a second, independent LTLf formula evaluated by
`RepeatedViolationMonitor` (not the same DFA as `main_ltl`/`specs.py`'s `"ltl"` field, which
drives the actual violated/satisfied verdict). It only produces the *secondary*
`repeated_violation_episodes` bookkeeping (start/end frame, `duration_frames`, `recovered`
bool) -- getting it wrong doesn't corrupt the primary classification, but it does make that
bookkeeping meaningless or actively misleading, which defeats the entire point of having it.

This is the `recovery_ltl`-specific half of the general `predicate-design-cycle` skill (design
-> implement -> verify isolated -> verify real data -> refine -> scale -> document) -- use that
one for the surrounding workflow (e.g. how to verify against real data, how to scale a spot-check
up via SLURM); this one for the structural bugs specific to `recovery_ltl` itself.

This was worked out the hard way across two properties
(`rc_released_object_eventually_settles`, `rc_dropped_object_was_released`) in
`SafeManip/docs/predicate_ltl_design/CHANGES_2026-09-02.md`'s recovery_ltl-design section --
read that for the full worked example if you want the concrete frame-level evidence behind
every rule below.

## Step 1: know when recovery actually starts evaluating

`RepeatedViolationMonitor.step()` only starts stepping the recovery DFA forward from `q0` at
the exact frame `in_violation` first goes `True` -- i.e. the frame `main_dfa` reaches a
*confirmed rejecting/trap state* (`_is_rejecting_main_state`), not the frame the main formula's
per-frame `accepting` status first goes `False`. For an until-shape main formula, these can be
many frames apart (the main formula can sit "pending" for a while before conclusively failing).
Before writing `recovery_ltl`, know precisely which frame(s) it will actually see as its first
observation(s) -- test this directly with an isolated `LTLfDFA` call if you're not sure, don't
guess from reading the formula.

## Step 2: two structural bugs to check for, every time

**Bug A -- vacuous antecedent.** Never write `recovery_ltl` as `G(edge_condition ->
F(...))` where `edge_condition` is an *edge* (true for exactly one frame, like `object_dropped`,
`object_released`). Because recovery only starts evaluating *after* the main formula's
rejection is confirmed -- which is always at or after the edge that triggered it -- the edge
has necessarily already reverted to `False` by the time recovery's own antecedent gets checked,
and it never fires again in that sub-trace. `G(False -> anything)` is vacuously true from frame
1, regardless of the `F(...)` part. This shape is only safe when the antecedent is a *level*
condition that can genuinely still hold at the moment recovery begins (e.g. `forbidden_contact`
in `rc_no_forbidden_contact`'s recovery, which isn't an edge). If your antecedent is an edge,
just drop it -- recovery is already only ever evaluated while genuinely in violation, so there's
nothing left to additionally gate on. Verify: isolated `LTLfDFA` test, feed it the exact
`predicate_values` recorded at the trap-confirmation frame, check `is_accepting()` is `False`
there (if it's already `True`, you have this bug).

**Bug B -- tautological escape term.** Never include, in a recovery escape (`F(a | b | ...)`),
an atom that is *by construction* already true at the frame recovery starts -- specifically,
whatever atom's becoming-true is what defines the main formula's trap in the first place (e.g.
`object_left_gripper` for an until-shape main formula whose trap is exactly "`object_left_gripper`
became true before `object_grasped` did"). Including it makes the whole `F(...)` trivially
satisfied at frame 1, before any other term is ever actually checked -- it's not "usually true
early," it is *guaranteed* true at that exact frame, every single time, with no exceptions.
Verify the same way as Bug A: isolated test, feed the trap-confirmation frame's own
`predicate_values`, check whether the escape term you're suspicious of is already `True` there.

## Step 3: if a bare, identity-less atom is your only real option, check it's still safe

Atoms like `object_grasped` have no per-object identity -- they can be satisfied by a
completely different, unrelated object's action, not the one whose violation opened this
episode. That's not automatically a bug (see Step 4), but there's one specific risk worth
checking: could a *different* object's own, later, genuinely-separate violation ever get lost
because the current episode never resolves and keeps `main_state` frozen?

It can't, as long as the escape atom is something every possible future violation's own trigger
condition already implies. For `object_dropped`-triggered properties: `object_dropped` can only
fire after `object_grasped` transitions true->false, so *any* future violation's own precondition
already satisfies `F(object_grasped)` -- there's no sequence of events where a second violation
happens without first passing through a grasp that would have already unstuck the first
episode's recovery wait. Don't just assert this reasoning -- prove it with a synthetic trace:
construct object A dropping and never recovering, then object B being grasped (should unstick
A's episode) and then B itself dropping and never recovering either; confirm both show up as
separate entries in `repeated_violation_episodes`, not just one.

## Step 4: decide honestly whether this is "recovery" or "resume"

For an edge-triggered bad event that's a one-shot, immutable past fact (a drop already
happened; it can never be un-happened by any later LTL condition holding), `recovery_ltl` was
never really capable of answering "was the bad thing fixed" -- there's no formula that makes a
past event not have occurred. In that case, be explicit (in the docstring/comments, and in the
property's own `viewer/annotations/ltl_debugging_guides/NN_*.md` file) that `recovery_ltl` here
means "when should tracking resume for whatever the robot does next," not "was this specific
instance saved." This changes what's acceptable: a deliberately tautological/instant escape
(e.g. resolving the moment an object merely leaves the gripper's region, regardless of what
happens after) can be the *right* choice under this framing, even though it means
`recovered`/`duration_frames` carry no discriminating information -- document that tradeoff
explicitly rather than presenting the field as if it measures something it doesn't.

If a genuinely meaningful "how long did this take to resolve" number is wanted, check whether
it already exists somewhere else first -- e.g. the viewer's `predicate_breakdown.occurrences`
(`compute_occurrences` in `viewer/server.py`) computes trigger-frame-to-resolution timing
directly from the main formula, completely independent of `recovery_ltl`, and may already be
showing exactly the number you were trying to get `recovery_ltl` to produce.

## Step 5: verify against real data, not just the formula's isolated behavior

Run the actual monitor pipeline (`run_monitor_on_privileged.py`) against a real episode known to
exercise the property, and inspect `repeated_violation_episodes` directly -- not just an
isolated `LTLfDFA` test. A formula can behave correctly in isolation but still produce a
misleading result once fed a real, noisy trace (undebounced physics signals flickering, an
unrelated object's action landing at an unexpected frame, etc.). Cross-check the reported
`start_frame`/`end_frame`/`recovered` against the raw predicate trace frame-by-frame before
trusting it.

## Step 6: sweep for false positives specifically, not just false negatives

Bugs A and B above are both, at bottom, false positives: `recovery_ltl` reporting
`recovered=True` at a frame where nothing it actually claims to check has happened yet. That's
easy to miss if verification only asks "does this resolve when it should" (a false-negative
framing) — both bugs *do* resolve, immediately and unconditionally, which reads as "working
great" unless you specifically ask "is there a frame where this reports `recovered=True` that
the real trace doesn't actually justify." For every `recovery_ltl` you ship:

- Pull every real episode where the property actually violates, and for each one look at the
  exact frame `recovered` flips `True` (or the exact frame the isolated `LTLfDFA` test's
  `is_accepting()` flips `True`). Ask: does the formula's own text, read literally, describe
  something that is actually, verifiably true in the raw predicate trace at that frame — or does
  it just happen to be the same frame recovery started (i.e. frame 1 of recovery, every time,
  regardless of the escape condition's own content)? The latter is the signature of Bugs A/B.
- If `recovered` resolves in ~1-2 frames on *every* violating episode you check, that's not
  automatically fine just because nothing is technically wrong with the DFA — treat a
  suspiciously-instant, suspiciously-universal recovery rate as the false-positive smell it is,
  and go check Bug A/B before accepting it (even if, per Step 4, the final honest answer ends up
  being "yes, this is deliberately tautological, and here's why that's still the right choice").

If you want to record what you find here as a durable, reusable record rather than just fixing
the formula in place, `ltl-ground-truth-annotation`'s schema has a dedicated `monitor_problem`
field for exactly this kind of bug (recovered/duration_frames being right or wrong for reasons
that have nothing to do with the real underlying event).

## Checklist summary

- [ ] Confirmed (via isolated `LTLfDFA` test) which frame recovery actually starts evaluating from.
- [ ] No edge-triggered antecedent in a `G(... -> F(...))` wrapper (Bug A).
- [ ] No escape term that's guaranteed already-true at the trap-confirmation frame (Bug B).
- [ ] If using an identity-less atom, proved (synthetic trace) that no future violation can be
      silently lost.
- [ ] Decided and documented whether this is genuine recovery or a resume signal, and what
      `recovered`/`duration_frames` actually mean as a result.
- [ ] Verified against a real episode's actual trace, not just isolated formula behavior.
- [ ] Swept real episodes for false positives specifically (Step 6), not just confirmed the
      formula resolves when expected.
