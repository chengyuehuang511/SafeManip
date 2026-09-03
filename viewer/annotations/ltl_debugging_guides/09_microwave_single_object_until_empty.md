# `rc_microwave_single_object_until_empty`

```
G(object_reach_in_fixture -> microwave_empty)
Recovery: !two_or_more_objects_in_microwave   <-- BARE ATOM, see below
```
Shape: **instant**. Predicates: `object_reach_in_fixture`, `microwave_empty`,
`two_or_more_objects_in_microwave`.

## Known bug: bare-atom `recovery_ltl` (KNOWN_BUGS.md #10, not yet fixed)

`recovery_ltl = "!two_or_more_objects_in_microwave"` is a bare atom (with a leading `!`, but
still no temporal operator) — per LTLf semantics (README Step 6 #1), this only checks the exact
frame recovery-tracking starts, not "eventually." Listed explicitly in KNOWN_BUGS.md #10's
proposed fix set. **Fix, if you pick this up**: wrap in `F(...)`:
`recovery_ltl = "F(!two_or_more_objects_in_microwave)"`.

## What to check
- This is an **instant** shape (`microwave_empty` must hold on the exact frame
  `object_reach_in_fixture` fires), not an until — so the main formula's own correctness doesn't
  depend on any debounce/timing subtlety the way an until-shape property would. If a violation
  looks wrong, check `microwave_empty`'s value at the exact reach-in frame directly; there's no
  "eventually" component to the main verdict at all.
- `two_or_more_objects_in_microwave` (recovery-only atom) needs to be listed in this property's
  `predicates` in `specs.py` even though it's absent from `main_ltl` (README #6) — if you ever
  see a `NameError` for it inside the recovery DFA's `eval()`, that's the missing-registration
  bug, not a logic bug.

## Known-good / open
Main `instant` formula not suspected of a bug; `recovery_ltl` confirmed broken (bare atom), not
yet fixed.
