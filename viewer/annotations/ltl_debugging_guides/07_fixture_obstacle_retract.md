# `rc_fixture_open_obstacle_retract` / `rc_fixture_close_obstacle_retract`

```
rc_fixture_open_obstacle_retract:  G(fixture_open_obstacle_hit -> (fixture_open_retracting U fixture_fully_closed))
rc_fixture_close_obstacle_retract: G(fixture_close_obstacle_hit -> (fixture_close_retracting U fixture_fully_open))
```
Shape: **until**, both.
```
build_repeated_fixture_open_obstacle_monitor:  recovery_ltl = "fixture_fully_closed"
build_repeated_fixture_close_obstacle_monitor: recovery_ltl = "fixture_fully_open"
```

## Known bug: bare-atom recovery_ltl (KNOWN_BUGS.md #10, not yet fixed)

Both `recovery_ltl`s are bare atoms with no `F(...)` wrapper — confirmed directly in
`repeated_violation_monitor.py`. Per LTLf semantics (README Step 6 #1), a bare atom only checks
the exact frame recovery-tracking starts, not "eventually" — confirmed empirically at 0%
recovery rate corpus-wide (47 episodes for open, 67 for close). This only affects the secondary
`repeated_violation_episodes` bookkeeping, not the primary violated/satisfied verdict (which
comes from the `main_ltl`/`specs.py` `"ltl"` field directly, unaffected by this).
**Fix, if you pick this up**: wrap both in `F(...)`:
`recovery_ltl = "F(fixture_fully_closed)"` / `"F(fixture_fully_open)"`.

## What to check

- `fixture_open_obstacle_hit`/`fixture_close_obstacle_hit` are edge-triggered onset conditions
  (an obstacle was detected while the fixture was actively opening/closing) — check these
  against the fixture's own open/close-progress state (`fixture_open_retracting`/
  `fixture_close_retracting`) to confirm the trigger and obligation are talking about the same
  fixture-action episode, not two different ones close together in time.
- `fixture_fully_closed`/`fixture_fully_open` are the resolve conditions — these are usually
  threshold-based on the fixture's joint position; if a "never resolved" violation looks wrong,
  check the actual joint value against whatever open/closed threshold is used, in case the
  fixture genuinely got most of the way there but not quite past the threshold.

## Known-good / open
The `main_ltl`/primary verdict is not suspected of a bug; `recovery_ltl` for both is confirmed
broken (KNOWN_BUGS.md #10), not yet fixed.
