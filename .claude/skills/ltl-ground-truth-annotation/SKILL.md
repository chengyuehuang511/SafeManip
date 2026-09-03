---
name: ltl-ground-truth-annotation
description: Build or extend structured ground-truth annotations for an LTL property's real behavior on a real episode, using the viewer's annotation storage (viewer/annotations/*.json, viewer/server.py's load_annotations/save_annotations). Use whenever asked to "annotate this episode", "check if the monitor got this right", "build ground truth for <property>", or to compare Claude-drafted vs human-confirmed judgments on violated/satisfied instances. Distinct from predicate-design-cycle's Phase 4 (verifying a formula while designing it) -- this is about building a durable, reusable annotation record that outlives any one design session.
---

# LTL Ground-Truth Annotation

The viewer's current annotation schema (`viewer/annotations/<task>__<episode>.json`, read/written
by `viewer/server.py`'s `load_annotations`/`save_annotations`) is a flat per-instance verdict:

```json
{
  "violations": {"0": {"verdict": "confirmed", "note": "...", "ai_draft": "...", "ai_draft_verdict": "..."}},
  "satisfied": {"15": {"verdict": "confirmed"}},
  "missed_notes": "",
  "overall_verdict": null
}
```

This records *whether the monitor's own verdict was right*, but not *what actually happened in
the episode* -- there's no structured record of which object, which frame the trigger fired,
which frame the obligation resolved, etc. This skill is the target shape for a richer annotation
that captures both, and how to build it correctly.

## The two axes

**Axis 1 -- who annotated it: `claude` vs `human`.** Keep these as separate, parallel
namespaces under each instance, never merged into one field -- the existing `ai_draft`/
`ai_draft_verdict` split is the right precedent (a Claude-authored draft that a human later
confirms, corrects, or overrides, with both versions still visible afterward). Never silently
overwrite a human's prior annotation with a new Claude draft, or vice versa; disagreement
between the two is itself useful signal (it's exactly the kind of thing a false-positive/
false-negative sweep, see `predicate-design-cycle`'s Phase 4, is trying to surface).

**Axis 2 -- what's being recorded, for each source: `gt_annotation` vs `monitor_problem`.**
These answer two genuinely different questions and must not be conflated into one verdict field:

1. `gt_annotation` -- what actually happened in the real episode, independent of what the
   monitor said. Is this instance a real event the property should care about?
2. `monitor_problem` -- given the real events `gt_annotation` describes, did the *monitor's own
   computation* (the DFA verdict, `recovery_ltl`'s episode bookkeeping, a specific predicate's
   per-frame value) get it right? A monitor can produce the correct final violated/satisfied
   verdict while still getting the internal reasoning wrong (e.g. the tautological-escape-term
   bug from `recovery-ltl-design` -- `recovered` is `True` for a reason that has nothing to do
   with what actually happened), and that's worth recording separately from "is this a real
   event."

## Building `gt_annotation`: follow the LTL's own role structure, not just true/false

Don't record a single verdict -- record the actual frame(s) for each *role* the property's own
formula decomposes into. `viewer/spec_derive.py` already parses every property's `ltl` string
into exactly this role vocabulary (`pattern`/`trigger`/`obligation`/`resolve`/`guard`/`check`/
`escape`) for `PROPERTY_META` -- reuse those same role names in the annotation schema instead of
inventing new ones, so an annotation for a property lines up with the same language the viewer
already uses to describe that property's shape.

Concretely, for `G(object_grasped -> (object_sync U object_dropped))` (an
`obligation`-until-`resolve` shape gated by a `trigger`), a `gt_annotation` entry should record:

```json
{
  "object": "bread",
  "trigger": {"role": "object_grasped", "frame": 212},
  "obligation_violation": {"role": "object_sync", "frame": 240, "note": "contact desync observed here, before object_dropped"},
  "resolve": {"role": "object_dropped", "frame": 248},
  "confidence": "exact"
}
```

Workflow for filling this in, using the current monitor pipeline's own output as the starting
candidate (not from scratch):

- **The monitor's flagged instance looks right (true positive):** confirm it -- copy the
  frame(s) the monitor already reports for each role into `gt_annotation`, don't just write
  `"verdict": "confirmed"` and stop there. The whole point is a structured record, not a rubber
  stamp.
- **The monitor's flagged instance is wrong (false positive):** omit it from `gt_annotation`
  entirely -- don't record a frame for an event that didn't really happen. Put the reasoning in
  `monitor_problem` instead (see below), since a false positive is definitionally a monitor
  problem.
- **The monitor missed a real instance (false negative/miss):** this is the hard case -- there's
  no monitor-reported frame to start from. Locate the real trigger/obligation/resolve frames by
  reading the raw predicate trace directly (same technique as `predicate-design-cycle`'s Phase 4:
  pull `privileged_information_<N>.json`'s per-frame dump, scan for the relevant atoms flipping).
  If the exact frame genuinely can't be pinned down, mark `"confidence": "approximate"` and say
  what it was inferred from (e.g. "no direct object_sync log for this object between frames
  230-250; inferred from object_grasped's own edge at 240 plus the gripper velocity trace") --
  never present an inferred frame as if it were directly observed.

## Building `monitor_problem`: separate from, not a restatement of, `gt_annotation`

Once `gt_annotation` establishes what really happened, `monitor_problem` records whether the
monitor's own machinery reasoned about it correctly -- e.g.:

```json
{"monitor_problem": {"has_problem": true, "description": "recovery_ltl reports recovered=True at frame 248 (the same frame object_dropped fires), but this is the tautological-escape-term bug (see recovery-ltl-design) -- object_left_gripper is definitionally true there regardless of any real recovery."}}
```

`has_problem: false` is a real, useful answer too -- don't only ever fill this field in when
something's wrong; recording "checked, monitor's reasoning here is sound" is exactly the kind of
positive confirmation that keeps a false-positive sweep (per `predicate-design-cycle`) from
having to be redone from scratch next time.

## Apply this per episode, not just to spot-checked instances

A GT annotation set is only useful in aggregate -- one annotated instance out of forty doesn't
tell you much about the monitor's real reliability. Once a property/episode combination is worth
annotating at all, go through *every* violated and satisfied instance the monitor reports for
that property in that episode (both directions -- see the false-positive-sweep principle in
`predicate-design-cycle`'s Phase 4), plus check for misses by scanning the raw trace independent
of what the monitor flagged at all.

## Relationship to other skills

- `predicate-design-cycle`'s Phase 4 ("verify against real data") is the *ad hoc*, in-the-moment
  version of the false-positive/false-negative check this skill formalizes into a saved,
  reusable record. Use that skill's technique for finding real frames; use this skill's schema
  for writing down what you found so it survives past the current session.
- `recovery-ltl-design`'s Bug A/B catalog is exactly the kind of thing that belongs in
  `monitor_problem`, not `gt_annotation` -- the real-world event (a drop, a release) is one fact;
  whether `recovery_ltl` reasoned about it correctly is a separate fact about the monitor, not
  about the episode.
- `predicate-doc-sync` is for keeping the *design docs* in sync with `predicates.py`; this skill
  is for keeping a separate, empirical *ground-truth annotation corpus* in sync with what
  episodes actually contain -- don't conflate updating `KNOWN_BUGS.md` with filling in
  `gt_annotation`, they're different kinds of record with different audiences.

## Not yet implemented

This schema is a design target, not yet wired into `viewer/server.py`'s `load_annotations`/
`save_annotations` or the frontend annotation UI (`viewer/static/app.js`) as of 2026-09-03 --
the current schema is still the flat `verdict`/`note`/`ai_draft` shape shown at the top of this
file. Implementing the schema change (new fields, migration of existing annotation files, UI for
entering role-scoped frames instead of a single verdict) and actually populating episodes with it
are separate, larger follow-on tasks -- confirm scope with the user before starting either.
