# Known bug: one-frame bilateral-contact flicker during continuous holding

Tracking doc for a single recurring root cause that has surfaced multiple times, via different
downstream symptoms, across this session's work. Not fixed yet. Separated from
`CHANGES_2026-08-31.md` (which documents changes made) and `KNOWN_BUGS.md` (which tracks
implementation-vs-spec discrepancies) because this is neither -- it's an open bug in the current
implementation itself, confirmed on real data multiple times, worth tracking on its own until
resolved.

## The bug

`_object_gripper_bilateral_contact(name)` (`predicates.py`) requires
`GRASP_BILATERAL_MIN_CONTACT_BODIES` (=2) distinct gripper finger bodies to be in contact with the
object *simultaneously*, in the same simulation frame. On real data, this contact count can
momentarily drop to 0 or 1 distinct body for exactly **one frame**, even while an object is clearly
still being held (grasped both before and after that single frame, with no visible drop in the
video). `_object_is_grasped` -- and therefore `object_grasped`, `raw_grasped_objects`, and
everything built on top of them -- reads `False` for that one frame.

This was the exact flicker item 1 in `CHANGES_2026-08-31.md` was meant to eliminate (by requiring
bilateral rather than aggregate any-geom contact). It mostly does -- but not in every case; the
underlying finger/object contact detection itself still has single-frame noise that bilateral
contact doesn't fully absorb.

## Confirmed instances

| task | episode | object | frame(s) | how it surfaced |
|---|---|---|---|---|
| `ArrangeBreadBasket` | 6 | `basket` | 445 | `object_released` fired via the (now-fixed) `object_supported` fallback -- a phantom release |
| `ArrangeTea` | 0 | `obj2` | 85 (previously reported as 92 with an earlier code version) | same phantom-release path, cascaded into a false `rc_released_object_eventually_settles` timeout at frame ~181-188 |
| `ArrangeBreadBasket` | 1 | `basket` | 523 | after the `object_supported` fallback fix (item 16), surfaces directly as `rc_grasp_remains_safe_until_release` instead (neither `object_grasped_safe` nor `object_released` holds on the flicker frame) |
| `ArrangeBreadBasket` | 2 | `basket` | 837, 844-845, 849-850 (three separate flickers, same episode) | **no violation** -- see note below |

Same root cause, four different confirmed occurrences across three episodes/objects. Confirmed
independently each time via direct frame-level inspection of `raw_grasped_objects`/`object_grasped`
in the saved `privileged_information_N.json`, showing a brief (1-2 frame) `[]`/`False` gap
sandwiched between `True` frames on both sides.

### Why ep2's three flickers don't cause a violation, unlike ep1's

Checked frame-by-frame: at 837 and 849 (and 844, the start of that gap), `object_released` reads
`True` on the *exact* flicker frame -- because `object_supported(basket) AND
object_stable_relative(basket)` (item 16's fallback condition) happen to both be true at that
instant. That closes the current `until`-window occurrence as "resolved" (a release event) right
as it opens, rather than leaving it open with neither the obligation nor the resolve holding.
`object_grasped` returns to `True` one or two frames later, starting a *fresh* occurrence that then
holds cleanly for the rest of its span -- so no violation is ever recorded, even though a real
release did not happen.

This means whether a given flicker instance produces a violation is essentially coincidental,
depending on whether `object_supported`/`object_stable_relative` happen to both read `True` at the
exact flicker frame:
- If yes (ep2): the flicker gets silently absorbed as a spurious release-and-immediate-regrasp --
  no violation, but also a false `object_released` event logged that never actually happened
  (a *different*, quieter false positive than the ones item 16 was built to close, since this one
  doesn't cascade into a settle-timeout -- the object re-grasps before any settle-timeout window
  could even open).
- If no (ep1 f523, `ArrangeTea` f85): the property fails outright.

Either way, the underlying bug is the same one-frame contact flicker -- ep2's case just happens to
land somewhere the existing fallback logic papers over instead of exposing.

## Why item 16 (`object_stable_relative` in `object_released`'s fallback) didn't fix this

Item 16 fixed the *symptom* that mattered most at the time (a phantom release that then never
settles, cascading into a second false violation) by requiring the object to actually be at rest
before the `object_supported` fallback can fire. That closes the false-release path specifically.
It does **not** address the underlying flicker itself -- `object_grasped` still momentarily reads
`False`, so `G(object_grasped -> object_grasped_safe U object_released)` still sees a frame where
the grasp obligation is open but neither the obligation nor the resolve condition holds, tripping
`rc_grasp_remains_safe_until_release` instead. The bug didn't get fixed, it moved to a different
property.

## Full sweep: `ArrangeBreadBasket`, all 10 episodes (v3 run, job 3740153)

Systematically scanned every episode's `raw_grasped_objects` for short (<=5 frame) gaps that
recover, cross-referenced against actual reported violations and the user's own viewer
annotations (`viewer/annotations/training__ArrangeBreadBasket__v3_2026-09-02_position_based_grasp_slip__*.json`):

| ep | flicker(s) found | violation(s) | user annotation |
|---|---|---|---|
| 0 | none | none | -- |
| 1 | `basket` 523 (len 1), `basket` 715 (len 1, self-resolved via `object_released` at 715) | `rc_grasp_remains_safe_until_release` @ 443 (**`bread`**, not from either listed flicker -- see caveat below) | disputed: "f523 is a flicker, not sure if we need smoothing" |
| 2 | `basket` 837 (len 1), 844-845 (len 2), 849-850 (len 2) -- all three self-resolved (`object_released` true on the first frame of each gap) | none | -- |
| 3 | none | none | satisfied, confirmed |
| 4 | `basket` 528 (len 1, not self-resolved) | 3 violations: `rc_grasp_remains_safe_until_release` @ 412 (**`bread`**, same caveat as ep1), `rc_no_forbidden_contact` @ 390 (unrelated), `rc_reach_in_fixture_only_when_fully_open` @ 356 (unrelated) | -- |
| 5 | none | `rc_reach_in_fixture_only_when_fully_open` @ 321 (unrelated to this bug) | -- |
| 6 | `basket` 445-446 (len 2, not self-resolved) | `rc_grasp_remains_safe_until_release` @ 445 (`basket`, matches this flicker directly) | disputed: "445 is a grasp but the position has a little change, not sure if it's too sensitive" |
| 7 | `basket` 716-717 (len 2, not self-resolved -- causes the violation), `basket` 911-913 (len 3, self-resolved via `object_released` at 911, but irrelevant since the property already permanently failed at 716) | `rc_grasp_remains_safe_until_release` @ 716 (`basket`, matches the first flicker directly) | disputed: "same grasp flickering issue" |
| 8 | none | none | -- |
| 9 | none | `rc_no_forbidden_contact` @ 374 (unrelated) | -- |

**Confirms the "coincidental outcome" finding above at scale**: 8 flicker instances total across the
episodes with a violation-causing flicker (ep1, 4, 6, 7) or a clean self-resolving one (ep1's
715, ep2's three, ep7's 911-913) -- roughly half self-heal silently via `object_released` firing on
the flicker frame, half don't and cause a visible violation. Purely a function of whether
`object_supported`/`object_stable_relative` happen to both read `True` at that exact frame, as
established above.

**Caveat -- ep1 frame 443 and ep4 frame 412 are a *different*, not-yet-investigated issue**: both
are reported against `bread`, not `basket`, and are NOT short flickers -- `bread`'s
`raw_grasped_objects` drops out at that frame and **stays empty for the rest of the episode** (no
recovery), i.e. a genuine, sustained release. `object_released` reads `False` right at the drop
frame in both cases despite this being real (not a flicker), which resembles the older
"enabling-condition lags the grasp-drop edge" class of issue from earlier in this session (item
10's motivating case, frame 389) -- but item 10's fix (`previously(gripper_is_opening)`) should
already cover a one-frame lag, so if it's still failing here the gap must be longer than one frame,
or a different enabling condition is involved. Not yet diagnosed -- flagged here rather than
conflated with the bilateral-contact flicker bug this file is otherwise about, since the data
pattern (long-lived drop, not a short recoverable gap) doesn't match.

## Not yet fixed: candidate approaches (none implemented)

- Investigate why bilateral contact count drops for exactly one frame during otherwise-continuous
  holding -- is this genuine physics-engine contact-resolution noise (a real, if brief, loss of
  contact that self-heals), or a detection/classification bug in
  `_gripper_finger_body_contact_map`/`_object_gripper_bilateral_contact` itself? Not yet
  determined -- this is the actual next step before picking a fix.
- If it's genuine one-frame physics noise: a debounce specifically on the raw contact signal (not
  a general-purpose grace window re-introduced elsewhere) might be the right place for it now,
  unlike the persistence-frame constants removed earlier this session (those were absorbing a
  flicker whose root cause has since been fixed at the raw-signal level via bilateral contact --
  this is a *different*, still-unaddressed flicker source in that same raw signal).
- Alternatively: since `object_sync`/`_object_grasp_slip` already tracks whether the object is
  moving rigidly with the eef frame-to-frame, a single-frame contact dropout while `object_sync`
  stays `True` on both sides could be treated as "still effectively grasped" without touching the
  contact-detection logic itself -- worth considering, not yet designed or implemented.

## Confidence

High -- confirmed via direct frame-level data inspection in three independent cases across two
tasks, not inferred from aggregate statistics.
