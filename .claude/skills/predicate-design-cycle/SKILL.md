---
name: predicate-design-cycle
description: The full design -> implement -> verify (isolated) -> verify (real data) -> refine -> scale -> document cycle for a new or changed predicate/LTL property/threshold constant in SafeManip/monitor/. Use whenever designing a new predicate from scratch, iterating on a formula the user is proposing candidates for, debugging why a predicate/property behaves unexpectedly, changing a threshold constant (e.g. SETTLE_TIMEOUT_FRAMES, GRIPPER_FAR_THRESHOLD) and needing to see the effect, or scaling a spot-check up to more episodes. This is the general-purpose version of that cycle -- for recovery_ltl specifically, use the recovery-ltl-design skill instead; for keeping docs in sync afterward, use predicate-doc-sync.
---

# Predicate Design Cycle

The reusable shape of a whole session's worth of work on `object_left_gripper`,
`gripper_away_from_object`'s mesh-distance upgrade, `SETTLE_TIMEOUT_FRAMES`, and
`rc_dropped_object_was_released`/`rc_released_object_eventually_settles`'s formulas — captured
so the next round of design work doesn't have to rediscover the same failure modes. See
`CHANGES_2026-09-02.md` for the concrete worked examples behind every step below.

## Phase 0: check the object-attribute layer before trusting anything built on it

Several predicates (`support_stable`, anything keyed off `RECEPTACLE_CATEGORIES`/
`TOOL_CATEGORIES`/fragile/pourable/etc.) depend on `monitor/sim/robocasa/attributes.py` --
SafeManip's own hand-authored object/fixture/support attribute logic, *not* anything robocasa
ships natively. Before designing or adjusting a predicate that reads one of these attributes,
check whether it's actually grounded in robocasa's own data or invented:

- `monitor/sim/robocasa/attribute/` -- robocasa's own native taxonomy, extracted directly from
  robocasa's source with zero SafeManip-authored data (`extract_native_structure.py` ->
  `native_structure.json`/`native_structure_tree.txt`/`native_structure_level_summary.txt`).
- `monitor/sim/robocasa/attribute_diff/` -- `compare_attributes.py`'s systematic diff of
  `attributes.py` against that native data, written to `diff_report.txt`/`diff_report.json`.
  Reports three separate things, and they need different responses:
  1. **Attributes.py disagrees with robocasa's own native tag** (e.g. `RECEPTACLE_CATEGORIES`
     hand-lists 15 categories, robocasa's own `receptacle` type-tag has 27, with real
     mismatches both ways; `TOOL_CATEGORIES` conflates robocasa's separate native `tool` and
     `utensil` tags). This is a genuine correctness bug in `attributes.py` -- fix it to match
     robocasa's own data.
  2. **Declared but never assigned to anything (dead code)** -- e.g. 12 of 43 declared
     object-axis members, all 5 declared button-role members, all 5 declared tool-role members.
     Either wire it up to something real or remove it; a declared-but-dead attribute is a trap
     for the next predicate that assumes it's actually checked somewhere.
  3. **Invented, with no native robocasa equivalent at all** (e.g. the entire `support` role axis
     -- `containment`/`heated`/`cold_storage`/`wash_zone`/`prep_zone`/`serving_zone`/
     `storage_zone` -- plus `button`/`tool` roles, `FRAGILE_CATEGORIES`, `LIQUID_CATEGORIES`,
     `POURABLE_CATEGORIES`, `TWISTABLE_CATEGORIES`). Not automatically wrong -- these encode real
     domain knowledge robocasa's own taxonomy doesn't capture -- but **not verifiable against
     robocasa's own data either**, so treat any predicate built on one of these as resting on a
     human judgment call, not a grounded fact, and say so if asked how confident to be in it.
  Don't just re-read `diff_report.txt` from memory each time -- it's a point-in-time snapshot
  (last generated 2026-08-31); re-run `compare_attributes.py` if `attributes.py` or robocasa's
  own object registry has changed since, before trusting it's still current.
- **Never modify anything under `attribute/`** -- it's a read-only extraction of robocasa's own
  data, same rule as never touching the upstream `robocasa/` package itself. Only ever change
  `attributes.py` (SafeManip's own file) based on what the diff reveals, and only `attribute_diff/`'s
  scripts if the comparison logic itself needs to change.

## Phase 1: design

- **Check what already exists before writing new geometry/signal code.** `object_left_gripper`
  and `gripper_away_from_object`'s mesh-distance upgrade both ended up built entirely from
  helpers that already existed (`_gripper_aabb`, `_object_contact_aabb`, `_aabb_distance`,
  `_gripper_contact_geom_ids`, `_object_geom_ids`) — no new geometry math was actually needed,
  just wiring existing pieces together differently. Grep for the concept before implementing it.
- **Identify which "tier" of accuracy the signal needs**, roughly in this order, and only reach
  for a more expensive tier if the cheaper one is confirmed insufficient for the actual question
  being asked:
  1. A single point-to-point distance (cheapest, crudest — upstream RoboCasa's
     `OU.gripper_obj_far` is this; ignores size/shape entirely).
  2. AABB-to-AABB overlap or gap (`_aabb_intersects`/`_aabb_distance`) — shape/size-aware, cheap.
  3. Real mesh/collision-geom distance (`mujoco.mj_geomDistance`, iterated over every
     gripper-geom x object-geom pair) — the same thing the contact solver itself uses, most
     accurate, but genuinely expensive (see Phase 6) if not memoized.
  Don't jump straight to tier 3 by default -- confirm tier 1/2 are actually insufficient for the
  specific question first (they often aren't).
- **Decide up front whether this is really "the same question, more precisely" or "a different
  question."** `object_left_gripper` (has the object left the gripper's *region* at all) and
  `gripper_away_from_object` (is the gripper *far* from the object, a stricter distance-based
  notion) are NOT the same predicate just because they're both gripper-object geometry checks --
  don't let one implementation's fix silently change the other's semantics.
- **Prefer fixing the formula's structure over adding a frame-smoothing/debounce constant.**
  Every debounce-style hyperparameter (`STABLE_PERSISTENCE_FRAME`, a "require True for N
  consecutive frames" wrapper) is a tax paid on every future reader of the trace -- it hides the
  raw signal's real behavior behind an extra layer that has to be remembered and accounted for
  separately, and picking N is itself a guess with no principled answer. The
  `rc_grasp_remains_safe_until_release` -> split into `rc_grasp_remains_synced_until_dropped` +
  `rc_dropped_object_was_released` (2026-09-02) is the template: the original formula's flakiness
  looked at first like it needed smoothing over a noisy `object_released` signal, but the real
  fix was recognizing the *formula* was conflating two different questions (did contact-tracking
  stay in sync vs. was the eventual drop a deliberate release) -- once split, neither half needed
  new smoothing to be reliable. Before reaching for a new/larger debounce constant, ask: is the
  raw signal genuinely noisy at the physical/sensor level (in which case some persistence
  threshold is legitimate), or is a single correct-but-currently-tangled formula being patched
  around instead of restructured? Keep a hyperparameter only when it encodes something the
  concept *itself* genuinely requires a numeric answer to -- a distance for "is the gripper
  near/far/approaching" (`GRIPPER_FAR_THRESHOLD`), a real physical settle/timeout window
  (`SETTLE_TIMEOUT_FRAMES`) -- not as a workaround for a formula that flickers because of how
  it's written. When in doubt which category a given constant falls into, that's worth asking
  the user rather than assuming either way.

## Phase 2: implement

- **Ordering matters for plain top-level statements, not for nested `def`s.** A `def foo(): ...`
  inside a big enclosing function can reference names bound *later* in that enclosing function,
  as long as they're bound by the time `foo` is actually *called* -- but a plain assignment
  (`x = f(y)`) executed at the point it's written needs `y` already bound *then*. Confirmed the
  hard way twice this session (`object_left_gripper = _bool(has_active_object and ...)` failing
  because `has_active_object` wasn't assigned yet at that point in the function body).
- **Every atom either formula (main_ltl or recovery_ltl) touches must be in `specs.py`'s
  `predicates` list for that property**, not just the atoms in `main_ltl`'s own string --
  `RoboCasaSymbolicMonitor.alpha()` only computes what's listed there, and that same dict feeds
  both DFAs. Missing one produces a `NameError` inside `LTLfDFA`'s `eval()`, confirmed multiple
  times.
- **A new predicate name missing from `COMMON_PREDICATES`/`monitor/predicates.py` (DSL)/
  `monitor/primitives.py` breaks *every* property's import**, not just the one being touched --
  `symbolic_properties.py`'s `ALL_PROPERTIES = build_all_properties()` runs eagerly at module
  import. If you see `NameError` deep inside an unrelated import chain right after adding a
  predicate, this is almost always why.
- **`specs.py`'s `"ltl"` field is live, not documentation** -- it's compiled into the real DFA
  that drives the *primary* violated/satisfied verdict, completely independently of whatever
  `repeated_violation_monitor.py`'s matching `main_ltl` string says. The two must be kept
  textually identical by hand; there's no automatic check. Verify by testing the exact string in
  `specs.py`, not just the one in `repeated_violation_monitor.py`.

## Phase 3: verify in isolation (fast feedback, before touching real data)

```python
from monitor.LTLfDFA import LTLfDFA
dfa = LTLfDFA('<exact ltl string>')
state = dfa.q0
for obs in synthetic_or_real_frame_observations:
    state = dfa.delta(state, obs)
    print(dfa.is_accepting(state), dfa.is_trap_state(state))
```
Cheap, fast, and where most of the *logical* bugs in this session were first caught (vacuous
antecedents, tautological escape terms — see the `recovery-ltl-design` skill for that specific
catalog). But a single isolated observation is not the same as a full-trace replay — a
single-step test can give a misleading answer if the formula's state depends on history (this
happened at least twice: `release_object_settle_timeout` looked non-trap-triggering in a
single-step test, but genuinely was, once the full 389->395 trace was replayed).

## Phase 4: verify against real data (the step that actually catches bugs)

Every wrong conclusion this session came from trusting isolated reasoning instead of checking a
real trace; every real fix came from looking at one.

```python
import json
data = json.load(open("privileged_information_<N>.json"))
frames = data["privileged_dynamic_info"]
# print every predicate the formula touches, frame by frame, around the trigger --
# see viewer/annotations/ltl_debugging_guides/README.md's Step 4 for the exact helper snippet
```

If you're building a durable, reusable ground-truth record rather than a one-off check for the
formula you're currently designing, use `ltl-ground-truth-annotation`'s schema to write down what
you find instead of just fixing the formula and moving on.

If real extracted data doesn't exist yet for the episode you need, `extract_privileged_from_dataset.py
--task <T> --episode <N> --output_root <dir> --run_monitor` produces it (full simulation
replay, ~5-6 min/episode under normal load — see Phase 6 for scaling this up). If you only
changed a `recovery_ltl` string or another monitor-time-only thing (not a predicate/threshold
constant baked in at extraction time), you don't need to re-extract -- just re-run
`run_monitor_on_privileged.py` against the already-extracted raw file (seconds, not minutes).
**Changing a threshold constant used inside `predicates.py` (e.g. `SETTLE_TIMEOUT_FRAMES`,
`GRIPPER_FAR_THRESHOLD`) always requires re-extraction** -- the value is baked into the
per-frame boolean during simulation replay, not recomputed afterward.

**Run a false-positive sweep, not just a false-negative one.** Debugging naturally gravitates
toward "why didn't this fire when it should have" (a false negative) because that's usually
what surfaces the bug report in the first place. Explicitly also check the opposite: for the
top-level `ltl`, for `recovery_ltl`, and for *every* subpredicate either one references, pull
the frames where it reports `True` and manually confirm the real trace actually supports that —
not just the frames where it's `False`. Every real false-positive bug found this session
(vacuous antecedent making `recovery_ltl` trivially/always `True`; a tautological escape term
making `recovered=True` every time regardless of what actually happened) was a case where the
formula's own isolated logic looked fine and only a targeted look at its `True` frames exposed
the problem. Concretely: `grep`/filter the extracted trace for the frame(s) where the atom in
question flips to `True`, and ask "does the real episode, frame-by-frame, actually justify this
being `True` right here" — the same way you'd already ask "why is this `False` when it should be
`True`" for a false negative.

If you can't yet answer "what would this look like if it were a false positive" for a given
predicate/subpredicate before checking, that's a sign to stop and think it through rather than
skip the check — e.g. for `object_grasped`, the false-positive question is "is there a frame
where this is `True` but no contact/grasp geometry actually supports it"; for `recovery_ltl`,
it's "is `recovered=True` reported at a frame where nothing the formula's own text describes
actually happened yet" (exactly the tautological-escape-term bug).

## Phase 4.5: adjusting a threshold constant end to end

The concrete recipe for "change `X_FRAMES`/`X_THRESHOLD` from A to B and show me the effect,"
worked through twice this session (`SETTLE_TIMEOUT_FRAMES` 6→50→100, `GRIPPER_FAR_THRESHOLD`
0.10→0.05):

1. **Find and change the constant**, then confirm the file still parses
   (`python3 -c "import ast; ast.parse(open('predicates.py').read())"`). Sanity-check units before
   guessing what a bare number like "5" or "10" means — `GRIPPER_FAR_THRESHOLD` is meters (0.10 =
   10cm), so "change 10 to 5" meant `0.05`, not `5`. If ambiguous, say what you assumed rather than
   silently picking one.
2. **Confirm it's baked at extraction time** (most predicate thresholds are — see Phase 4) before
   assuming a monitor-only rerun will show the effect. If it's extraction-time, you need fresh
   `privileged_information_<N>.json` files, not just fresh `_monitor.json` files.
3. **Re-extract the same episode set you're comparing against**, in parallel via `sbatch`, into a
   **genuinely shared, non-`/tmp` output path** — e.g. `monitor/output/<scratch-name>_test/` under
   `$HOME`/`testnvme`, not this session's own scratchpad directory. This is not optional: an
   `sbatch` job's own `/tmp` is namespace-isolated from every other process (see Phase 7's `/tmp`
   warning) — pointing `--output_root` at `/tmp/...` silently loses the entire extraction the
   moment the job's allocation exits, even though the job's own log prints a normal-looking
   success line. Lost three real episodes' worth of compute (4-8 min each) to this exact mistake
   once already.
4. **Compare against the baseline by property, not just by episode-level violated/satisfied
   counts.** Load both the old and new `_monitor.json`, filter each episode's `violations`/
   `satisfied` lists for the one `property_name` you're actually testing, and print the
   `original.explanation` for both — a raw count going from "4 violated" to "0 violated" doesn't
   tell you *which* violation flipped or why; the explanation string (with its own frame numbers)
   does.
5. **Decide: new version, or fold into the current in-progress one?** The `predicate-doc-sync`
   skill's default rule is "never overwrite an existing `vN_.../` in place if the code changed —
   cut `v(N+1)_.../` instead." That's the right default once a version is meant to be a *settled,
   citable* snapshot. But if you're still actively iterating on the same constant/family of
   changes and haven't reached something worth freezing yet, it's fine to keep overwriting the
   *current* in-progress version's files in place (confirm with the user which mode you're in --
   this session explicitly chose "fold into v4, not v5" for exactly this reason). Once you do
   overwrite in place, that version's label may no longer describe everything it now contains
   (v4's name still said "grasp_ltl_split_recovery_fix" after settle-timeout/gripper-distance work
   was folded in too) — flag that mismatch rather than silently letting the name go stale.
6. **If the viewer is running, it won't see new/removed version directories until restarted** --
   `TRAINING_MONITOR_METHODS` is computed once at process import time (`viewer/server.py`'s
   `_discover_training_monitor_methods()`). `kill` the running `viewer/server.py` process and
   relaunch it (`nohup python3 viewer/server.py --port <p> --host <h> > log 2>&1 &`) after any
   change to what's on disk under `monitor/output/`, then confirm via
   `curl .../api/training_monitor_methods` before telling the user it's live.

## Phase 5: refine (when the user proposes a candidate, or you find a bug)

- **Verify every new candidate the same way, from scratch** -- don't assume a variant "should"
  behave a certain way; test it. This session iterated through 4+ rounds each for
  `object_left_gripper`'s main formula and its `recovery_ltl`, and at least one round always
  looked plausible right up until real data proved it wrong.
- **When a proposed fix is a naming near-miss** (e.g. `object_settle_timeout` vs.
  `release_object_settle_timeout` -- two real, different predicates with similar names, one
  scoped to release-timeout, one to unrelated liquid/solid transfer-timeout), check the actual
  definition before implementing what was literally typed. This came up more than once with the
  exact same pair of names.
- **Distinguish "this is genuinely wrong" from "this is a deliberate design tradeoff I don't yet
  understand."** The `recovery_ltl` discussions repeatedly turned out to hinge on what the field
  was actually *for* (recovery vs. resume) rather than a bug -- clarify intent before assuming a
  result is broken.

## Phase 6: check performance before scaling up

Adding a new per-frame computation to `predicates.py` runs on *every frame of every episode*
from then on -- a cost that's invisible in a single-episode spot-check but compounds across a
real extraction run. Concretely hit this session: adding `mj_geomDistance`-based mesh distance
to `_gripper_far_from_object` (up to ~150 geom-pair queries per call) turned out to be called
*twice per frame* for the same object (`_object_settled()`'s own internal call, plus the
separate `gripper_away_from_object` export) -- a duplication that was harmless when the function
was cheap, and a real, measurable slowdown once it wasn't. Fixed with a simple per-frame memo
cache keyed on `monitor_state["monitor_frame_index"]`. Before declaring a new predicate done,
ask: is this called more than once per frame for the same input, and is that now expensive
enough to matter?

## Phase 7: scale the verification, watch the infrastructure

- **A shared login node with many concurrent users is not a reliable place to run multiple
  simulation-replay extractions.** Confirmed this session: `load average: ~120`, swap fully
  exhausted, two concurrent heavy jobs on the login node got OOM-killed (exit 137) even though
  `free -h`'s "available" column looked fine. Prefer `sbatch` over running extraction directly
  on the login node once you need more than one or two episodes.
- **No GPU needed for this pipeline** -- `extract_privileged_from_dataset.py`'s `make_env()`
  sets `has_offscreen_renderer=False`/`use_camera_obs=False` (pure physics-state replay, no
  rendering), so a GPU request just queues behind the more-contended GPU partition for zero
  benefit. `monitor/run_extract_privileged_from_dataset.sbatch` already documents this and
  requests CPU-only (`--cpus-per-task=8 --mem=32G --qos=debug`, no `--gpus-per-node`).
- **To parallelize specific episodes of one task** (not the existing
  `submit_extract_privileged.sh`'s per-*task* array, which runs all of one task's episodes
  sequentially inside a single array-task job): submit one independent `sbatch` job per episode
  directly, each with its own script file (matching the same resource flags above) that calls
  `extract_privileged_from_dataset.py --task <T> --episode <N> --output_root <dir> --run_monitor`.
  **Use `sbatch <script.sh>` with an actual script file, not `sbatch --wrap="source ...; ..."`**
  -- `--wrap` runs under `/bin/sh` by default, where `source` isn't available (needs `.` or an
  explicit bash shebang); confirmed this fails immediately with `source: not found` in every
  array task if you get this wrong. Poll completion with a small loop
  (`while squeue -j $jobid -h | grep -q .; do sleep 30; done`) rather than trying to estimate
  wall-clock time -- SLURM queue wait times before a job even starts running are themselves
  unpredictable under `overcap`/priority contention.
- **Never point `--output_root` at a path under `/tmp` (including this session's own scratchpad
  directory) for an `sbatch`-submitted job.** Confirmed this session the hard way: `/tmp` is
  namespaced per-job (standard PAM `/tmp` isolation on most SLURM clusters) — a job's own
  `/tmp/...` is invisible to *any other process*, including an interactive session's bash tool,
  even when the job happens to land on the exact same physical node as a process that can see
  that path. The failure mode is silent and looks like partial success: the job's own log prints
  a completely normal "extracted N frames -> /tmp/.../file.json" success line (because the write
  genuinely succeeded, just into a namespace nobody else can read), `squeue` shows it finished
  cleanly, and only `ls`-ing the claimed output path afterward reveals the file was never really
  reachable. Three separate episodes' extraction work (each 4-8 minutes of real compute) was lost
  this way in one session before the pattern was recognized. Always use a genuinely shared,
  NFS-backed path (e.g. under `$HOME` or the project's own `testnvme`/`flash` mount) as
  `--output_root` for anything that needs to be read back outside the job itself.

## Phase 8: document what changed and why (see predicate-doc-sync)

Once the cycle above lands on something verified-correct, hand off to `predicate-doc-sync` for
keeping the `.txt` specs / predicate trees / changelog / `KNOWN_BUGS.md` in sync -- and to
`recovery-ltl-design` specifically if any of this touched a `recovery_ltl`. The changelog entry
should capture the *rejected* candidates and why, not just the final formula -- that's what
prevents the next session from re-proposing and re-testing something already ruled out.

**A `recovery_ltl` change in particular touches more files than just
`repeated_violation_monitor.py`, and it's easy to update the code and stop there.** Confirmed
this session for both `rc_dropped_object_was_released` and `rc_released_object_eventually_settles`
-- a single formula change needs updates spread across: the `build_repeated_*_monitor()`
docstring itself (repeated_violation_monitor.py), the matching `.txt` design spec's `Recovery:`
line (`docs/predicate_ltl_design/collision_grasp_release_contamination_safety.txt`), the running
changelog (`CHANGES_<date>.md`), `KNOWN_BUGS.md` if the old formula was tracked there, the
property's own `viewer/annotations/ltl_debugging_guides/NN_*.md` guide, and that guide's `README.md`
catalog if a new *category* of bug was found (not just a one-off fix). Grep for the old formula
string across all of `docs/` and `viewer/annotations/` before considering the change documented
-- a stale copy in even one of these reads as authoritative to the next person who finds it
first, and there's no automated check that catches drift between them.
