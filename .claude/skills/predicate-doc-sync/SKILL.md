---
name: predicate-doc-sync
description: Keep SafeManip's predicate/LTL design docs and monitor/output result history in sync whenever SafeManip/monitor/sim/robocasa/predicates.py (or its constants) changes. Use whenever predicates.py is edited (grasp, release, settle, contact, contamination, mechanism, or onset logic; adding/removing/renaming a predicate; changing a threshold constant), whenever the user says "update the docs", "document this change", "keep docs in sync", "commit today's changes", or wants a changelog entry, and whenever new monitor results need to be recorded for comparison against a previous version. Push back gently if asked to skip the doc/changelog update after a predicates.py edit -- these docs are the only record of *why* a change was made, including things tried and reverted.
---

# Predicate/LTL Doc Sync

Keeps four things in lockstep whenever `SafeManip/monitor/sim/robocasa/predicates.py` changes:
the `.txt` design specs, the two grounded predicate-tree references, a running changelog, and
(when results are regenerated) a versioned snapshot under `monitor/output/`.

If the change specifically touches a `recovery_ltl` (in `repeated_violation_monitor.py`'s
`build_repeated_*_monitor` functions) rather than a `main_ltl`/predicate definition, use the
dedicated `recovery-ltl-design` skill first -- it covers structural bugs specific to
`recovery_ltl` (vacuous antecedents, tautological escape terms, the "recovery vs. resume"
distinction) that this skill's general workflow doesn't cover.

This skill assumes the design/implementation/verification work is already done and you're just
recording it. For the design -> implement -> verify -> refine -> scale cycle itself (before
there's anything settled enough to document), use `predicate-design-cycle` instead -- it's what
this skill's changelog format is meant to capture the *output* of.

Do **not** touch `robocasa/robocasa/environments/kitchen/predicates.py` (the upstream vendor
copy) — all predicate work lives in `SafeManip/monitor/sim/robocasa/predicates.py` only. If a
change needs to reach training-data extraction, `SafeManip/monitor/extract_privileged_from_dataset.py`
must import `monitor.sim.robocasa.kitchen_ext` (its `make_env`) and patch
`monitor.sim.robocasa.predicates`, not the upstream module — check this hasn't regressed if you
touch that script.

## Files this skill keeps in sync

All under `SafeManip/docs/predicate_ltl_design/`:

| File | What it is | Update when... |
|---|---|---|
| `collision_grasp_release_contamination_safety.txt`, `containment_safety.txt`, `mechanism_safety.txt`, `access_enclosure_safety.txt`, `action_onset_safety.txt`, `categorization.txt` | Per-property LTL spec, in the doc's own predicate vocabulary (no code internals) | The formula, a definition, or a listed constant for a predicate they cover changed |
| `LTL_PREDICATE_TREES.md` | Full grounded tree for all 19 LTL properties, with `predicates.py` line citations and `⚠`-flagged doc/code discrepancies | Any predicate's implementation changed enough that a citation or discrepancy note is now wrong |
| `LTL_PREDICATE_TREES_CONCISE.md` | Same 19 properties, doc-vocabulary only, no code citations, no per-line grounding | Same trigger as above, lighter-touch edit |
| `CHANGES_<session-start-date>.md` | Running changelog for the whole session/PR of predicate work | Every predicates.py edit, including ones later reverted |
| `KNOWN_BUGS.md` | Long-lived bug tracker (open / resolved / resolved-by-doc-alignment) | A bug listed there gets fixed, or a fix uncovers a new one worth tracking long-term (separate from the session changelog) |

Also under `SafeManip/viewer/annotations/ltl_debugging_guides/` (one level up from the docs
folder above, alongside the viewer's own saved annotation JSON — not a `.txt`/`.md` design spec,
a practical "how to debug this" reference):

| File | What it is | Update when... |
|---|---|---|
| `README.md` | General reusable debugging methodology (find the real formula, identify its shape, extract/inspect real data, test in isolation, a catalog of recurring bug categories with a symptom→cause table) | A genuinely new *category* of bug is found (not property-specific) that future debugging sessions should know to check for — add it to the Step 6 catalog and the symptom table |
| `01_...md` .. `11_...md` | One file per property or shared-formula family (all 20 properties covered) — formula, what's confirmed correct, what's confirmed broken, what to check first | A property's formula, predicates, or recovery_ltl changes; a new bug is found in a property already covered; a property gets a fresh verification pass worth recording |

These are debugging *aids*, not the source-of-truth specs (`.txt` files serve that role) — keep
them honest about what's actually been verified vs. not (see the existing files' "Known-good /
open" sections for the tone: confirmed claims cite how they were confirmed, unconfirmed
suspicions are labeled as such). Don't let this become stale documentation that asserts things
were checked when they weren't — a guide that's silent on a property is better than one with
confidently wrong claims.

## Workflow

1. **After any predicates.py edit**, before considering the task done:
   - `python3 -c "import ast; ast.parse(open('predicates.py').read())"` to catch syntax errors immediately.
   - Grep the actual new code for the changed predicate name(s)/constant(s) to get real line
     numbers — **never guess a line number**, always verify with `grep -n` against the current file.
   - Update every `.txt` spec file that defines or lists the changed predicate/constant.
   - Update `LTL_PREDICATE_TREES.md`: fix the node's expansion and line citation, and any
     downstream node that referenced it (e.g. "see Property 2 above (predicates.py:X-Y)" citations
     scattered through other properties — grep for the old line range across the whole file).
   - Update `LTL_PREDICATE_TREES_CONCISE.md`'s matching section (no line numbers to fix there,
     just the formula/behavior description).
   - Append a numbered item to the current `CHANGES_<date>.md` (see format below).

2. **If a change is later reverted or partially reverted** (this happens — see the
   `object_sync`-in-`object_grasped` saga in `CHANGES_2026-08-31.md` items 2 and 18/17 for the
   canonical example): don't delete the record. Mark the changelog item and the doc/tree notes as
   `~~tried, reverted~~` with the concrete reason (a tautology, a regression, a rejected tradeoff),
   not just silently restore the old text. Future-you (or the next session) needs to know an idea
   was already tried and why it didn't work, or it'll get re-proposed and re-tried for free.

3. **When line-number drift accumulates faster than it's practical to re-verify everywhere**
   (e.g. many edits in one session touching a large chunk of `predicates.py`): fix the citations
   directly tied to the current change, then add a dated `⚠` disclaimer near the top of
   `LTL_PREDICATE_TREES.md` naming what was and wasn't re-verified, rather than presenting stale
   line numbers as if they were checked. See the existing 2026-09-01 disclaimer in that file for
   the template.

4. **Naming discipline in the trees** (this was a real, called-out mistake — see
   `LTL_PREDICATE_TREES_CONCISE.md`'s own intro): tree *node labels* use the predicate names the
   `.txt` docs already define (e.g. `correct_manipulated_object_correct_fixture_contact`,
   `original_supports_by_object`), never code-internal variable names (e.g.
   `active_target_fixture_geom_ids`, `robot_policy_geom_ids`) as stand-ins for a concept the docs
   already name. Code-internal names are fine only inside `[...]` grounding annotations / `leaf:`
   lines in the full (non-concise) tree, to cite *how* something is computed. If the code adds a
   concept the docs don't name at all, describe it in plain English — don't invent a
   predicate-sounding name for it.

## `CHANGES_<date>.md` format

One file per session/PR-sized batch of work, named for the date it started (keep appending to it
across multiple days if the same batch of work continues — don't start a new file just because
the calendar date changed; `CHANGES_2026-08-31.md` has entries dated into 2026-09-01 for exactly
this reason). Structure:

```markdown
# Changelog — <start-date>: <one-line theme>

Context: <why this batch of changes happened, 2-4 sentences>

## N. `<predicate or constant name>`: <one-line summary>

**Problem:** <what was wrong, concretely — a failure scenario, not just "this seemed off">

**Fix:** <what changed, and why this specific fix>

**Reverted** (if applicable): <what was tried, why it didn't work, what happened instead>

## Files changed
- <path> — <what changed there>

## Not changed / explicitly out of scope
- <thing that looked related but was deliberately left alone, and why>
```

Always include the "Not changed / explicitly out of scope" section — it prevents the next
session from re-litigating a scope decision that was already made on purpose.

## Versioning `monitor/output/` results

`monitor/output/` is its own **nested git repo** (independent of the main SafeManip repo, which
`.gitignore`s this whole directory — it's too large to track wholesale). Structure:

```
monitor/output/
├── .gitignore                          # excludes raw privileged_information_N.json (100-170MB
│                                        # each); tracks only *_monitor.json summaries
├── CHANGELOG.md                        # what each vN_ directory is + the diff from the previous one
├── v0_<date>_<short-description>/      # one directory per named version of predicates.py
│   └── <Task>/privileged_information_<N>_monitor.json
│   └── violation_summary.json          # aggregated: violation counts by property, per-episode list
│   └── VIOLATION_SUMMARY.md            # same, human-readable table
└── v1_<date>_<short-description>/
    └── ...
```

When predicates.py changes and you regenerate results:

1. Extract with `extract_privileged_from_dataset.py --call_stride 16 --output_root
   monitor/output/vN_<date>_<short-description>/ --run_monitor` (submit via
   `submit_extract_privileged.sh` with `OUTPUT_ROOT`/`CALL_STRIDE` exported, for a full multi-task
   SLURM array run — don't run 500 episodes serially in one shell).
2. Never overwrite an existing `vN_.../` directory's results in place if the *code* producing them
   changed — cut a new `v(N+1)_.../` directory instead, so old results stay diffable. Only
   overwrite within the *same* version (e.g. filling in episodes that hadn't finished yet).
   **Exception:** while still actively iterating on the same in-progress batch of changes (e.g.
   tuning a threshold constant through several values before settling on one) and the user
   explicitly says they're not ready to freeze a new version yet, it's fine to keep overwriting
   the *current* version's files in place across multiple code changes — see
   `predicate-design-cycle`'s Phase 4.5 for the concrete recipe. Confirm which mode you're in
   with the user rather than assuming; once you do this, that version's directory name may no
   longer describe everything it now contains (e.g. `v4_..._grasp_ltl_split_recovery_fix` ended
   up also containing unrelated settle-timeout/gripper-distance changes) — say so rather than
   letting a stale name look authoritative.
3. Build `violation_summary.json`/`VIOLATION_SUMMARY.md` for the new version the same way as the
   existing ones (aggregate violated-property counts, per-episode violated-property lists, overall
   violation rate) so it's diffable against the previous version at a glance.
4. Update `monitor/output/CHANGELOG.md` with what changed vs. the previous version and a
   before/after table for whatever concrete case motivated the change (a specific task/episode is
   more convincing than aggregate stats alone — see the `ArrangeBreadBasket` episode 0 entry for
   `v1` as the template).
5. `git add`/`git commit` inside `monitor/output/` itself (separate commit history from the main
   repo).

## Committing to the main repo

The main SafeManip repo's default branch is `main`; day-to-day work happens on `dev`. Only commit
when asked. When committing predicate-doc-sync work:
- Stage exactly the files this batch of work touched (the predicates.py edits, the `.txt` specs
  that changed, the tree docs, the changelog) — don't sweep in unrelated untracked files sitting
  in the working tree from other work.
- Write the commit message as a structured summary mirroring the changelog's own numbered items,
  not just "update docs" — the commit message is often read without the changelog file open.
