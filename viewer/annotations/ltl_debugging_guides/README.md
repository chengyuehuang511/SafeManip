# How to debug/verify an LTL property

## Index (all 20 properties, grouped by file where they share a formula/family)

| File | Properties |
|---|---|
| `01_no_forbidden_contact.md` | `rc_no_forbidden_contact` |
| `02_grasp_remains_synced_until_dropped.md` | `rc_grasp_remains_synced_until_dropped` |
| `03_dropped_object_was_released.md` | `rc_dropped_object_was_released` |
| `04_released_object_eventually_settles.md` | `rc_released_object_eventually_settles` |
| `05_raw_robot_contact_blocks_rte_grasp_until_sanitized.md` | `rc_raw_robot_contact_blocks_rte_grasp_until_sanitized` |
| `06_preconditions_safe_family.md` | `rc_pick_preconditions_safe`, `rc_place_preconditions_safe`, `rc_press_preconditions_safe`, `rc_turn_preconditions_safe`, `rc_slide_preconditions_safe`, `rc_twist_preconditions_safe`, `rc_open_close_preconditions_safe`, `rc_dump_preconditions_safe` |
| `07_fixture_obstacle_retract.md` | `rc_fixture_open_obstacle_retract`, `rc_fixture_close_obstacle_retract` |
| `08_liquid_solid_transfer_eventually_settles.md` | `rc_liquid_transfer_eventually_settles`, `rc_solid_transfer_eventually_settles` |
| `09_microwave_single_object_until_empty.md` | `rc_microwave_single_object_until_empty` |
| `10_reach_in_fixture_only_when_fully_open.md` | `rc_reach_in_fixture_only_when_fully_open` |
| `11_fixture_placement_release_after_internal_support.md` | `rc_fixture_placement_release_after_internal_support` |

Start with `03_dropped_object_was_released.md` if you want the fullest worked example of the
whole methodology below — every category of bug in Step 6 showed up in that one property at
some point during the 2026-09-02 session.


A distilled, reusable workflow for figuring out whether a property's formula and predicates are
actually doing what they're supposed to — written after a long session (2026-09-02) of
iterating on `rc_dropped_object_was_released` through 4 rounds before it was actually correct.
That session is the running example throughout; see `CHANGES_2026-09-02.md` for the full story.

One file per property (or per family of properties sharing the same LTL shape) lives alongside
this one — start there for property-specific gotchas, then come back here for the general
method.

## The one rule that matters most: verify against real data, not just formula-reading

Every wrong conclusion in the 2026-09-02 session came from trusting a formula's *apparent*
meaning instead of checking what it actually does against a real trace. Every fix came from
looking at real frame-by-frame data. This isn't optional — LTLf semantics are subtle enough
(see below) that "this formula looks right" and "this formula is right" are different claims.

## Step 1: find the property's real formula (there are two, and they can silently drift apart)

- **`specs.py`'s `"ltl"` field** — NOT just documentation. `symbolic_properties.py`'s
  `_materialize_property()` compiles a real `LTLfDFA` from it directly, and this is what
  `run_monitor_on_privileged.py`'s primary `violations`/`satisfied` classification is actually
  based on (`if final_event["accepting"]: satisfied.append(...) else: ...`).
- **`repeated_violation_monitor.py`'s `main_ltl`/`recovery_ltl`** (in the matching
  `build_repeated_*_monitor()` function) — a separate, independent DFA that only produces the
  *secondary* `repeated_violation_episodes` detail (start/end frame, `recovered` flag,
  duration). Does NOT feed the primary classification at all.

**These two must be checked separately, and kept textually identical for `main_ltl`** — there is
no automatic check enforcing this. A real bug this session: `specs.py`'s field was left as an
old, simpler string (for the viewer's regex-shape-parser's convenience) while the "real" fix was
only written into `repeated_violation_monitor.py` — this silently kept the *primary*
classification on the stale formula, completely unaffected by the fix, for an entire debugging
round before being caught.

## Step 2: identify the formula's shape

Three base shapes exist, plus 2 escape-clause variants (see `viewer/spec_derive.py`):
- **invariant**: `G(!guard)` — guard must never be true.
- **until**: `G(trigger -> (obligation U resolve))` — once triggered, obligation must hold every
  frame until resolve fires.
- **instant**: `G(trigger -> check)` — check must hold on the exact same frame as trigger.
- **instant-with-escape**: `G(trigger -> (check | F(escape)))` or `G(trigger -> (check |
  (!guard U escape)))` — check resolves it instantly, or the escape/until can resolve it later.

Knowing the shape tells you what "violated" actually requires: an until/escape shape means the
violation window can span many frames and might resolve later than you'd naively expect from
just looking at the trigger frame.

## Step 3: extract raw data for a real episode where you suspect an issue

```bash
python3 extract_privileged_from_dataset.py --task <TaskName> --episode <N> \
  --output_root <scratch_dir> --run_monitor
```
Writes `privileged_information_<N>.json` (raw per-frame predicate values) and
`privileged_information_<N>_monitor.json` (monitor classification + explanation + repeated
episodes). Takes several minutes per episode (full simulation replay) — under heavy cluster
load this can take much longer; if backgrounding it, poll for the output file's existence
directly rather than trusting a wrapper's own "completed" status (a wrapper that backgrounds
itself internally can report done before the real work finishes).

## Step 4: read the per-frame boolean trace directly, don't guess

```python
import json
data = json.load(open("privileged_information_<N>.json"))
frames = data["privileged_dynamic_info"]

def sp_val(f, key):
    preds = (f.get("data") or {}).get("predicates") or {}
    s = (preds.get("sections") or {}).get("predicates") or {}
    e = s.get(key)
    return e.get("value") if isinstance(e, dict) else e

def ve_val(f, key):  # active_object, settle_obj_name, etc.
    preds = (f.get("data") or {}).get("predicates") or {}
    return (preds.get("violation_evidence") or {}).get(key)

for i in range(start, end):
    f = frames[i]
    print(i, {k: sp_val(f, k) for k in [...predicates you care about...]},
          "active_obj=", ve_val(f, "active_object"))
```
Print every predicate the formula touches, plus `active_object`/`settle_obj_name` (identity
context — critical for catching cross-object misattribution, see below) for a window around
each frame where `object_dropped`/`skill_*_onset`/whatever the trigger is fires.

## Step 5: test the formula in isolation against that exact trace

Don't trust the full pipeline's output alone — cross-check with a direct, isolated DFA test so
you know *why* it classified a case the way it did:

```python
from monitor.LTLfDFA import LTLfDFA
dfa = LTLfDFA('<exact ltl string>')
state = dfa.q0
for i, f in enumerate(frames):
    obs = {k: bool(sp_val(f, k)) for k in [...atoms in the formula...]}
    state = dfa.delta(state, obs)
    print(i, "accepting=", dfa.is_accepting(state), "trap=", dfa.is_trap_state(state))
```
This tells you exactly which frame flips non-accepting, and whether/when it becomes a
*confirmed trap* (irrecoverable) vs. merely pending. These can differ by many frames — the
per-frame `accepting` status (drives the primary classification and the viewer timeline) flips
non-accepting immediately at the violating frame; `repeated_violation_monitor`'s own
`in_violation` bookkeeping only flips once the DFA reaches a *confirmed* structural trap
(`_is_rejecting_main_state`), which can lag by many frames for until/escape shapes. Don't
confuse the two numbers.

## Step 6: watch for these specific trap categories

These are the actual bugs found this session, in the order they tend to surface:

1. **Bare-atom `recovery_ltl`** (KNOWN_BUGS.md #10). LTLf evaluates a bare atomic proposition
   only at the *current* (first) frame of the trace being evaluated — not "eventually." A
   `recovery_ltl` like `"object_released"` (no temporal operator) only ever checks the frame
   recovery starts on — which is almost always false, since that's why it's recovering. Always
   wrap in `F(...)`: `G(bad -> F(good))`. Confirmed empirically: 0% recovery rate for every
   un-wrapped property, 99.1% for the one `F(...)`-wrapped control.
2. **Cross-object misattribution.** Any bare predicate atom (`object_grasped`,
   `object_stable_relative`, `object_supported`, ...) is usually global and object-identity-less
   — it doesn't know or care *which* object made it true. Wrapping one in `F(...)` or a `U` can
   get satisfied by a completely unrelated later event involving a *different* object. Check
   `active_object`'s value at both ends of the window before trusting a resolution. The fix that
   worked here (not a general recipe, just what applied): lean on `active_object` not switching
   away from the relevant object until the robot moves on to touch something else, so a
   correctly-scoped resolution condition (like whether *that* object settles) naturally stays
   attributed to the right object via timing, without needing a dedicated identity-tracking
   predicate.
3. **Undebounced physics/contact signals.** Several raw checks (`object_stable_relative`,
   `object_supported`, raw mesh-contact checks) have no debounce at all, unlike their sibling
   `object_stable` (which requires `STABLE_PERSISTENCE_FRAME` consecutive frames via
   `_persistent_bool`). A single noisy frame can trip an `until`/escape clause that depends on
   them. If a formula seems to fail for exactly one frame in the middle of an otherwise sane
   window, suspect this before suspecting the formula's logic.
4. **Contact-solver flicker specifically.** MuJoCo's active-contact list (`env.sim.data.contact`)
   can report zero contact for exactly one frame even when two collision geoms are clearly still
   near/around each other (confirmed directly: loaded the real sim state, checked
   `sim.data.ncon`/`sim.data.contact` — genuinely zero gripper-object contacts that one frame).
   An AABB-overlap check (see `_gripper_aabb()`/`_object_contact_aabb()`/`_aabb_intersects()`)
   is more robust here — it depends on continuous position, not a binary solver resolution.
5. **`_object_aabb()`'s default source can be wrong while an object is held** (KNOWN_BUGS.md
   #11) — confirmed for `basket`, unconfirmed elsewhere. If you're computing any object AABB and
   it doesn't seem to track the object's real (visibly correct) position while it's grasped,
   suspect this; use `_object_contact_aabb()` directly instead of the general `_object_aabb()`.
6. **`predicates` list omissions.** `RoboCasaSymbolicMonitor.alpha()` only computes values for
   symbols listed in a property's own `predicates` list in `specs.py` — and that same dict is
   what's fed into `RepeatedViolationMonitor.step()` for *both* the main and recovery DFA. Every
   symbol either formula touches (not just `main_ltl`'s original atoms) must be listed, or you
   get a `NameError` inside `LTLfDFA`'s `eval()`. If you extend a `recovery_ltl` or an escape
   clause to reference a new atom, update the `predicates` list too.
7. **Adding a new predicate breaks import at build time, not call time.**
   `symbolic_properties.py`'s `ALL_PROPERTIES = build_all_properties()` runs eagerly at module
   import — so a new atom missing from `COMMON_PREDICATES`/`monitor/predicates.py` (DSL)/
   `monitor/primitives.py` breaks *every* property's import, not just the one you're touching.
   The traceback will just say `NameError: name 'X' is not defined` deep inside module load; if
   you just added a predicate, that's almost certainly why.
8. **`recovery_ltl` vacuous antecedent.** `G(edge_condition -> F(...))` is always broken as a
   `recovery_ltl` shape when `edge_condition` is an edge (true for exactly one frame) — recovery
   only starts evaluating *after* the main formula's rejection is already confirmed, by which
   point the edge has necessarily already reverted to `False` and never fires again in that
   sub-trace, making the whole `G(...)` vacuously true from frame 1, regardless of the `F(...)`
   part. Only safe when the antecedent is a level condition that can genuinely still hold when
   recovery starts (e.g. `forbidden_contact`). If your antecedent is an edge, just drop it —
   recovery is already only evaluated while genuinely in violation.
9. **`recovery_ltl` tautological escape term.** Never include, in a recovery escape (`F(a | b |
   ...)`), whatever atom's becoming-true is what defines the *main* formula's own trap — it's
   guaranteed already `True` at the exact frame recovery starts, by construction, making the
   whole escape trivially satisfied at frame 1 before any other term is ever checked. See the
   dedicated `recovery-ltl-design` skill (`.claude/skills/recovery-ltl-design/SKILL.md`) for the
   full checklist covering both this and #8, plus how to decide whether `recovery_ltl` should
   even mean "recovery" at all for a one-shot/edge-triggered bad event (vs. a "resume tracking"
   signal instead) — worked out in full via `rc_dropped_object_was_released` and
   `rc_released_object_eventually_settles`, see `CHANGES_2026-09-02.md`'s recovery_ltl-design
   section.

## Step 7: check the viewer's occurrence breakdown against the same raw trace

`GET /api/training_monitor?task=<T>&episode=<N>&method=<method>` returns, per violated/satisfied
property, a `predicate_breakdown.occurrences` list — each occurrence's `activation` frame,
`violated_frames`, and `end` (resolved/unresolved + reason). This is a *separate,
approximate re-derivation* done by `viewer/server.py`'s `compute_occurrences()`, not the same
code path as the actual monitor — it can have its own bugs independent of whether the real
classification is correct (confirmed this session: it initially treated a genuinely-recovered
occurrence and a genuinely-unresolved one identically, and separately cropped its display window
before showing a signal's real eventual transition). If the breakdown looks wrong but the
top-level violated/satisfied verdict looks right, the bug is probably in
`compute_occurrences()`/`spec_derive.py`'s shape-parsing, not the underlying formula.

**Don't confuse this with `repeated_violation_episodes` (`recovery_ltl`'s own bookkeeping,
surfaced separately in the viewer as a collapsible "repeated_violation_episodes" section under
each property card).** These are two genuinely independent mechanisms that happen to look at
similar atoms and can legitimately disagree:
- `predicate_breakdown.occurrences` re-simulates the *main* formula's own until/escape,
  starting from the trigger frame, with no dependency on `recovery_ltl` at all.
- `repeated_violation_episodes` comes from a completely separate DFA (`recovery_ltl`), which
  only starts evaluating once `RepeatedViolationMonitor`'s own `in_violation` flag goes `True`
  (the main formula's *confirmed trap* frame, which can be later than the trigger frame).

Conflating the two caused real, repeated confusion while designing
`rc_dropped_object_was_released`'s recovery formula (see `CHANGES_2026-09-02.md`'s
recovery_ltl-design section) — if something described as "recovery" looks like it's working
fine, double check which of these two you're actually looking at before drawing conclusions
about the other one.

## Quick reference: what "looks wrong" usually means

| Symptom | Likely cause | Where to look |
|---|---|---|
| Property never shows *any* `repeated_violation_episodes` even though it's marked violated | `main_ltl` has a deferred/pending state (`F(...)`/`until` with no structural trap reachable per-frame) that `_is_rejecting_main_state` doesn't recognize until it's a confirmed trap | `repeated_violation_monitor.py`'s `_is_rejecting_main_state`; compare with the primary per-frame `accepting` trace directly via Step 5 |
| A property is satisfied/violated differently than a quick formula-read suggests | Escape/until clause resolving (or failing to resolve) somewhere later than the trigger frame | Step 5's isolated DFA trace, frame by frame |
| Recovery rate is suspiciously 0% for a property | Bare-atom `recovery_ltl` (see #1 above) | `repeated_violation_monitor.py`'s matching `build_repeated_*_monitor()` |
| Recovery rate is suspiciously 100%, resolving in ~1-2 frames every time | Vacuous antecedent or tautological escape term (see #8/#9 above) | Isolated `LTLfDFA` test fed the exact trap-confirmation frame's `predicate_values`; see the `recovery-ltl-design` skill |
| A violation seems to involve the wrong object | Cross-object misattribution (see #2 above) | `active_object`/`settle_obj_name` trace at both ends of the window |
| A one-frame blip causes a violation that "shouldn't" have happened | Undebounced signal (see #3/#4 above) | Check whether the offending atom has a `_persistent_bool`/debounce wrapper; if not, that's probably it |
| Viewer timeline shows a signal "never" transitioning when you expect it to | Display-window cropping, not a predicate bug | Check the API's `predicate_breakdown.window` bounds vs. where the transition actually happens in the raw trace |
