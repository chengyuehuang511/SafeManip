# `rc_no_forbidden_contact`

```
G(!forbidden_contact)
Recovery: G(forbidden_contact -> F(!forbidden_contact))
```
Shape: **invariant** (main), `F(...)`-wrapped bare atom (recovery — this is the one property
already using the correct recovery shape; see README #1). Predicates: `forbidden_contact`.

## What to check
- `forbidden_contact` is a derived boolean (see `predicate_derive.py`'s children for it) built
  from `considered_contact_pairs` minus `allowed_contact` — when debugging a specific violation,
  pull `forbidden_contact_pairs`/`considered_contact_pairs` from `violation_evidence` (exported
  alongside the boolean) to see the exact geom pair that tripped it, rather than re-deriving it
  from raw contact data yourself.
- This is the **control case** for recovery_ltl correctness — confirmed 99.1% recovery rate
  (459/463 episodes, corpus-wide) specifically because it's already `F(...)`-wrapped. If you're
  verifying a *different* property's recovery formula, this is the one to compare against as
  "what working looks like."

## Known-good
No open issues as of 2026-09-02.
