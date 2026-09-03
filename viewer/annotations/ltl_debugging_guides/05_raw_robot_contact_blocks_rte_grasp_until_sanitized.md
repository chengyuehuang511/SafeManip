# `rc_raw_robot_contact_blocks_rte_grasp_until_sanitized`

```
G(robot_contact_raw_contaminated -> (!robot_contact_clean U sanitized))
Recovery: G((robot_contact_raw_contaminated & robot_contact_clean & !sanitized) -> F(sanitized))
```
Shape: **until**. Predicates: `robot_contact_raw_contaminated`, `robot_contact_clean`,
`sanitized`. Recovery is already properly `F(...)`-wrapped (confirmed in
`repeated_violation_monitor.py`) — but see below for why it still can't ever fire.

## What to check
- **`sanitized` is hardcoded `False`** (`predicates.py`, `sanitized = False`, confirmed directly
  — per `LTL_PREDICATE_TREES.md`'s findings, flagged there as a known finding but explicitly out
  of scope for the 2026-08-31 changelog, not addressed since). This means neither the main
  formula's until nor its (correctly `F(...)`-wrapped) recovery can ever resolve via `sanitized`
  — once `robot_contact_raw_contaminated` triggers, this property can only ever end up violated
  (`!robot_contact_clean` failing at some point) or riding out the rest of the episode still
  "pending," and `recovery_ltl`'s `F(sanitized)` can never fire either. If you're debugging why
  this property never shows a clean recovery, this is why — not a recovery-formula-shape bug (it
  IS shaped correctly, unlike most others in KNOWN_BUGS.md #10), a never-implemented predicate
  underneath it.
- `robot_contact_raw_contaminated`/`robot_contact_clean` both track raw-vs-clean object contact
  state accumulated via `monitor_state` (see `predicates.py`'s `robot_contact_raw_*` family) —
  check the accumulator fields (`robot_contact_raw_active`, `robot_contact_raw_candidate`,
  `robot_contact_clean`) directly in `violation_evidence` rather than re-deriving contamination
  state yourself.

## Known-good / open
`sanitized` being permanently `False` is a known, unaddressed gap (see
`LTL_PREDICATE_TREES.md`) — not re-verified or fixed as part of the 2026-09-02 session.
