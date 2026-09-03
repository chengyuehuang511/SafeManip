# `rc_grasp_remains_synced_until_dropped`

```
G(object_grasped -> (object_sync U object_dropped))
Recovery: G(object_grasped & !object_sync -> F(object_sync | object_dropped))
```
Shape: **until** (main), correctly `F(...)`-wrapped (recovery). Predicates: `object_grasped`,
`object_sync`, `object_dropped`.

One half of the 2026-09-02 split of the old `rc_grasp_remains_safe_until_release` — see
`CHANGES_2026-09-02.md`. This half (does the grasp stay synced while held) has been stable since
first written; the split's problems were all on the *other* half
(`rc_dropped_object_was_released`, see its own guide).

## What to check
- `object_sync` is position-based (`_object_grasp_slip`, frame-to-frame, not
  accumulated-since-onset — see `CHANGES_2026-08-31.md` items 13/15) — if debugging a
  desync-flagged episode, check `object_sync`'s own trace directly rather than assuming it
  tracks velocity; it's a rigid-attachment drift check now, not a speed check.
- **recovery's dual escape matters**: it deliberately covers *both* ways a desync episode can
  honestly end — resynced (`object_sync` returns true) or the grasp itself ended
  (`object_dropped`). If you're tempted to simplify recovery down to just `F(object_sync)`,
  don't — a desync-then-drop sequence (no resync first) would then have no path back to
  recovery at all, and the episode would report as permanently unrecovered even though the
  grasp legitimately ended.
- `object_dropped` fires for exactly one frame on any grasp-ended edge (clean release, silent
  drop, *or* a one-frame bilateral-contact flicker) — it does not distinguish these; that
  distinction belongs entirely to `rc_dropped_object_was_released`, not this property. Don't
  expect this property's violations to tell you *why* the grasp ended, only that sync failed
  before it did.

## Known-good
No open issues as of 2026-09-02 — this is the "worked correctly on the first design" half of the
split; all iteration effort went into the other half.
