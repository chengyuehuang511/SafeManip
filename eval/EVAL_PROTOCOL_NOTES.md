# Eval protocol: what we use vs. each model's own official script

This documents every hyperparameter where our sweep could plausibly diverge
from a "faithful reproduction" of each model's own eval protocol, what we
actually decided, and why. Written after directly reading each codebase's
own eval scripts (not guessing from README snippets) -- see the specific
file paths cited in each section.

Models covered: the 4 Isaac-GR00T families (`target_posttraining`,
`target_only`, `pretraining`, `multitask_learning`), openpi (`pi0`,
`pi0.5`), `grootn16`, `RLDX-1`.

## `n_episodes`

**Decision: always 50, for every model, uniformly.** This is an explicit,
deliberate override of each model's own default/official value, made so
every model in our sweep is measured on exactly 2500 episodes
(50 tasks x 50 episodes) for apples-to-apples comparability across our own
8-model sweep. More episodes only improves statistical precision, so this
never *disadvantages* any model relative to its own official protocol.

| Model | Official default | We use |
|---|---|---|
| Isaac-GR00T family | 50 (`scripts/run_eval.py` default) | 50 (matches) |
| openpi (pi0/pi0.5) | 50 (`num_trials` default, `examples/robocasa/main.py`) | 50 (matches) |
| grootn16 | **30** (`scripts/eval_target50.sh`) | 50 (deliberate override) |
| RLDX-1 | 50 (`eval_robocasa365.sh` default) | 50 (matches) |

## `n_action_steps` (a.k.a. `replan_steps` for openpi)

**Decision: match each model's own official default exactly.** Unlike
`n_episodes`, this is a genuine execution-protocol parameter (how many of
the policy's predicted future actions get executed open-loop before
replanning) that can materially affect measured success rate, so we follow
the precedent already set by the original 6-model sweep (GR00T and openpi
already used their own respective native defaults, not a single shared
number).

| Model | Official value | Source | We use |
|---|---|---|---|
| Isaac-GR00T family | 16 | `scripts/run_eval.py` default | 16 |
| openpi (pi0/pi0.5) | 5 | `examples/robocasa/main.py`'s `replan_steps` default | 5 |
| grootn16 | 8 | `scripts/eval_target50.sh` | 8 |
| RLDX-1 | 8 | `run_scripts/eval/robocasa_365/eval_robocasa365.sh` | 8 |

Note: grootn16's/RLDX-1's own bare CLI (`rollout_policy.py`'s own
`__main__` block) defaults to a *different* number again (504/8 for
grootn16's argparse default, vs. 8 in `eval_target50.sh`) -- we match the
dedicated 50-task sweep script's value, not the single-task CLI's bare
default, since the sweep script is what's actually used to produce a
reportable multi-task number.

## `split` (`pretrain` vs `target`)

This is the one that actually matters most, and the one we got wrong on
the first pass for 5 of 8 models. Per RoboCasa's own docs
(https://robocasa.ai/docs/build/html/benchmarking/foundation_model_learning.html
and .../multitask_learning.html), the two benchmarking *settings* prescribe
opposite splits, and the "right" split is a property of the *setting* a
model belongs to, not a free choice:

- **Foundation Model Learning** setting (ablating pretraining vs.
  target-only vs. combined post-training on a fixed architecture):
  *"We always evaluate the models in target scenes."*
- **Multitask Learning** setting (a single checkpoint trained across the
  full multitask corpus, evaluated as one number): *"We evaluate the model
  in pretrain scenes."* The docs name Diffusion Policy, OpenPI, and
  GR00T N1.5 explicitly as examples of this setting; grootn16 and RLDX-1
  are the same kind of single multitask checkpoint (confirmed by their own
  checkpoint names, e.g. `grootn16_robocasa365_multitask_learning`), so the
  same rule applies to them by the same reasoning, even though the docs
  don't name them individually.

`pretrain` vs `target` isn't just a label -- per the docs, `pretrain` split
evaluates in the *same* 2,500 kitchen scenes / object pool used for
pretraining data collection (in-distribution, easier), while `target` split
evaluates in 10 *disjoint*, held-out kitchen scenes with a disjoint object
set never seen in pretraining (true generalization test, harder). Using the
wrong one doesn't just diverge from a convention -- it changes what's
actually being measured.

| Model | Setting | Correct split | What we ran |
|---|---|---|---|
| `target_posttraining` | Foundation Model Learning | `target` | `target` (correct from the start) |
| `target_only` | Foundation Model Learning | `target` | `target` (correct from the start) |
| `pretraining` | Foundation Model Learning | `target` | `target` (correct from the start) |
| `multitask_learning` | Multitask Learning | `pretrain` | ran `target` first (wrong), rerun with `pretrain` |
| openpi `pi0` | Multitask Learning | `pretrain` | ran `target` first (wrong), rerun with `pretrain` |
| openpi `pi0.5` | Multitask Learning | `pretrain` | ran `target` first (wrong), rerun with `pretrain` |
| `grootn16` | Multitask Learning | `pretrain` | ran `target` first (wrong, `n_action_steps` also wrong at the time), rerun with `pretrain` |
| `RLDX-1` | Multitask Learning | `pretrain` | never ran with `target` to completion; launched directly with `pretrain` |

Note that grootn16's *own* `scripts/eval_target50.sh` also uses
`SPLIT="pretrain"` -- so on this specific dimension, once we corrected our
mistake, we ended up matching grootn16's own script exactly, even though we
initially disagreed with it (see the git history/conversation for the
back-and-forth: we briefly considered grootn16's `pretrain` choice a
possible anomaly, until checking RoboCasa's own docs settled that
`pretrain` is in fact correct for its Multitask Learning setting).

**Data layout**: because both splits are now genuinely in use, results live
under `eval/saved_eval_rollouts/{target,pretrain}/<model>/...` rather than
a flat `eval/saved_eval_rollouts/<model>/...` -- the original (pre-split)
`target`-split runs for `multitask_learning`/`pi0`/`pi0.5`/`grootn16` were
moved (not deleted) into `target/` when this was discovered, so they remain
available for reference/comparison even though they're not the numbers we
report against the leaderboard.

## `max_episode_steps` (per-task horizon)

**Decision: robocasa's own `get_task_horizon(task)`, uniformly, for every
model.** All 8 models use this. Two sub-notes:

- grootn16's own `scripts/run_eval.py` (the code backing
  `eval_target50.sh`) also calls `get_task_horizon()` directly -- confirmed
  by reading the file -- so grootn16 matches its own official protocol
  exactly here.
- RLDX-1's own `eval_robocasa365.sh` instead uses a custom, shorter
  per-task horizon table baked into
  `run_scripts/eval/robocasa_365/task_sets.yaml` (e.g. `ArrangeBreadBasket:
  2900` there vs. `4350` from `get_task_horizon()` -- RLDX-1's own numbers
  run roughly 60-67% shorter across the board). We considered
  transcribing that table to match RLDX-1's protocol exactly, but decided
  this specific knob isn't worth chasing and kept `get_task_horizon()` for
  consistency with the other 7 models instead. **This is a known,
  deliberate discrepancy** for RLDX-1 specifically -- if our reproduced
  RLDX-1 number comes out higher than reported, a longer allowed horizon
  (more time to succeed) is a plausible contributing factor, not
  necessarily model degradation.

## `n_envs` (parallelism)

**Not a discrepancy** -- this only controls how many environment instances
run concurrently (throughput), not what's being measured. We use `n_envs=1`
for every model (required by our `ReplayCapture`/video-capture machinery,
which only supports one environment at a time), even though some official
scripts use more (e.g. RLDX-1's `eval_robocasa365.sh` defaults to
`N_ENVS=5`). Episodes are independent and identically distributed
regardless of how many run in parallel, so this doesn't bias success rate,
only wall-clock time.

## Summary of open, deliberate discrepancies (as of this writing)

1. **RLDX-1's per-task horizon** uses robocasa's `get_task_horizon()`
   instead of RLDX-1's own shorter custom table -- decided not worth
   matching (see above).
2. **`n_episodes=50` for grootn16** instead of grootn16's own official 30
   -- a deliberate, requested override for comparability across our sweep,
   not an oversight.

Both are called out explicitly here so that any gap between our reproduced
numbers and each submission's reported leaderboard numbers can be
attributed correctly rather than assumed to be pure model-reproduction
error.
