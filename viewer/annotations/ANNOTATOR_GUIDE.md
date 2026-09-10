# Annotator Guide (human reviewers)

Thanks for helping build ground truth for SafeManip's safety monitor! This is the
human-facing version of the instructions Claude itself follows when annotating
(`.claude/skills/ltl-ground-truth-annotation/SKILL.md`), scoped to what's actually
available in the viewer today.

## 0. Register yourself first

Top-right of the header, there's an **Annotator** dropdown. If your name isn't in
the list yet, click **+ new** and type it in once -- everything you annotate after
that is saved under your own name, separate from everyone else's. You can switch
back to your name any time (it's remembered in your browser). "All annotators" is
a *view-only* filter for seeing who's annotated what across everyone; you can't
save under it.

## 1. Scope: **Training Data tab only, for now**

Use the **Training Data** tab (top nav), not **Eval Results** -- that's the one
we're actually building ground truth on right now. Within Training Data, either
sim (RoboCasa or LIBERO) is fair game; there's a sim dropdown once you're in that
tab.

## 2. Picking what to annotate

1. In the left sidebar, open the **LTL property tree** and pick a property (or
   browse "All properties").
2. Click a task under that property to jump straight to an example episode where
   it's violated -- or just browse the task list and pick any episode.
3. Watch the video. The right-hand panel shows every violation/satisfied instance
   the monitor found for that episode, each with a jump-to-frame link so you don't
   have to scrub manually.

## 2b. Sidebar badges: who's already looked at this episode

In the Training Data episode list on the left, each row has an **✎ annotate** /
**✎ annotated** button -- that one is scoped to *you* (whichever name is picked
in the Annotator dropdown), so it flips to "annotated" once you've saved
something for that episode. If someone else has also saved something there, a
separate **"by: \<names\>"** badge shows up on the same row listing them -- so
you can tell at a glance whether an episode's already been covered by another
reviewer before deciding whether to spend time on it yourself (still worth a
look if you want a second opinion, just not a blind spot either way).

## 3. The verdict buttons

For **each individual violation/satisfied entry** the monitor lists (not just the
episode as a whole), watch what actually happens at that frame and click one:

| Button | Means |
|---|---|
| **✓ confirmed** | You watched it -- the monitor's call here (violated or satisfied) is correct. |
| **✗ disputed** | You watched it -- the monitor got this one wrong (a false positive if it's listed as violated but you don't see a real problem, or a false negative reasoning issue if it's marked satisfied but something's actually wrong). Use the note box below the buttons to say *why*. |
| **? unsure** | You looked, but can't tell either way (e.g. ambiguous camera angle, borderline case). Leave a note about what's ambiguous if you can. |
| **⦸ unverifiable** | You can't check this one at all from the available video/data (e.g. the relevant frame/object isn't visible in any camera). |

Each entry also has a free-text note box -- use it, especially for "disputed":
a bare button click doesn't tell anyone *why* you disagreed.

There are also two **episode-level** fields at the bottom of the panel, separate
from the per-instance verdicts above:

- **"Reviewer: anything the monitor missed?"** -- a free-text box. This is where
  false negatives go: a real safety issue you can see in the video that never
  shows up in the violations list at all (the monitor didn't just get a call
  wrong on something it flagged, it never flagged it in the first place).
- **Overall episode verdict** -- "monitor output matches video" / "does not
  match" / "partially matches", a single big-picture verdict for the whole
  episode once you're done with the individual instances above.

## 4. Look at all the predicates, not just the one you clicked in for

Don't stop at the single violation you jumped in to check. Once you're looking at
an episode, go through **every property listed for that episode** -- both the
ones marked violated *and* the ones marked satisfied -- not just the one that
brought you there. A monitor can be wrong in either direction:

- A **violated** entry might actually be fine (false positive) -- see "disputed"
  above.
- A **satisfied** entry might actually hide a real problem the monitor's logic
  missed (a false negative on that specific property) -- also "disputed", or use
  the "anything the monitor missed?" box if it's a problem the monitor's own
  violations/satisfied list doesn't cover at all.

Checking only the one violation you were pointed at gives a biased picture --
it can only ever confirm true/false *positives*, never catch what the monitor
silently got wrong on the satisfied side. Going through the whole property list
for an episode is what actually tells us whether the monitor is trustworthy.

## Questions / found a real bug in a predicate (not just this one episode)?

That's a different, bigger thing than a single annotation -- flag it separately
rather than just marking individual instances "disputed" one at a time.
