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
3. **`n_episodes=50` for RLDX-1's LIBERO eval** instead of RLDX-1's own
   official per-suite split (50 for libero_10, only 20 for spatial/object/
   goal) -- same override philosophy as (2), and it also brings RLDX-1
   into line with openpi's own uniform `num_trials_per_task=50` for
   LIBERO, so the two are directly comparable per-suite.

These are called out explicitly here so that any gap between our
reproduced numbers and each submission's reported leaderboard numbers can
be attributed correctly rather than assumed to be pure model-reproduction
error.

## LIBERO benchmark (separate from RoboCasa above)

Covers openpi (pi0/pi0.5) and RLDX-1 -- plain LIBERO only (spatial/object/
goal/10; no libero_90/LIBERO-Plus/LIBERO-Pro). Hyperparameters otherwise
match each model's own official script exactly (see
run_libero_suite_openpi.py / run_single_task_rldx1_libero.py docstrings for
the full comparison tables), with `n_episodes` uniformly overridden to 50
per discrepancy (3) above.

**LIBERO version pinning turned out to matter.** LIBERO has a documented
history of eval numbers not reproducing across versions (e.g. openvla's
LIBERO results are known to shift when run against a newer LIBERO release
than the one its numbers were measured on). Checked each model's own
pinned commit before reusing the single shared `eval/simulators/libero`
submodule (pinned Mar 2025) the way robocasa is shared across all 8
RoboCasa models:

- **RLDX-1**: confirmed via `diff -rq` that the shared submodule is
  content-identical to RLDX-1's own vendored
  `external_dependencies/LIBERO` copy -- no version drift, safe to reuse
  the shared submodule as-is.
- **openpi**: its own `third_party/libero` pins LIBERO @
  `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` (Dec 2023) -- ~15 months
  older than the shared submodule. Added a dedicated
  `eval/simulators/libero_openpi` submodule pinned to that exact commit
  instead of reusing the shared (newer) one. (First attempted
  `git submodule update --init` directly on openpi's own nested
  `third_party/libero`, but that populates files inside `eval/models/
  openpi`'s own working tree, which we don't touch -- reverted and used a
  separate standalone submodule pinned to the identical commit instead.)

**RLDX-1 LIBERO requires `--no-strict` (a real gap in RLDX-1's own
`check_action`, not a checkpoint/action-space incompatibility).** Running
RLDX-1's LIBERO checkpoint via `rollout_policy.run_rldx_sim_policy` (the
same ready-made entry point used for RoboCasa) crashes on every episode
with `AssertionError: Action key 'action.eef_pos_delta' must be in
action`. Root cause, confirmed by reading `rldx/policy/rldx_policy.py`
directly: `RLDXSimPolicyWrapper._get_action` already has a
`# ===== LIBERO KEY MAPPING =====` branch (`is_libero`) that remaps the
model's raw output keys (`eef_pos_delta`/`eef_rot_delta`/`gripper_close`)
into the flat sim keys the LIBERO env actually expects
(`action.x/y/z/roll/pitch/yaw/gripper`) -- but `check_action`, called
unconditionally right after via `BasePolicy.get_action`, has no matching
`is_libero` branch: it validates the (already-remapped) flat action dict
against the *raw* modality keys, so it always fails for LIBERO regardless
of whether the rollout itself is correct. RLDX-1's own official
`run_scripts/eval/libero/eval_libero.sh` passes `--no-strict` (which sets
`strict=False` on `RLDXSimPolicyWrapper`, skipping `check_action`/
`check_observation` entirely per `rldx/policy/policy.py`'s `get_action`) --
i.e. RLDX-1's own maintainers route around this gap rather than fix it.
`run_rldx_sim_policy` → `create_rldx_sim_policy`, however, hardcodes
`strict=True` with no passthrough. Since editing `eval/models/RLDX-1` is
off-limits, `run_single_task_rldx1_libero.py` monkeypatches
`rollout_policy.create_rldx_sim_policy` from the outside (before calling
`run_rldx_sim_policy`) to construct the identical objects with
`strict=False` -- reproducing the official script's own workaround exactly,
without touching the submodule or the validation logic itself.

Smoke-tested (job 3824826, `libero_sim/pick_up_the_alphabet_soup_and_place_
it_in_the_basket`): no more `eef_pos_delta` assertion, rollout proceeds
normally episode-by-episode.

**openpi LIBERO also needed a `torch.load` compat patch (unrelated to the
above).** `eval/simulators/libero_openpi`'s pinned (Dec 2023)
`libero/libero/benchmark/__init__.py::get_task_init_states` calls plain
`torch.load(init_states_path)` -- PyTorch 2.6 flipped the default
`weights_only` from `False` to `True`, so under this machine's newer torch
it crashes with `UnpicklingError: ... Unsupported global: GLOBAL
numpy.core.multiarray._reconstruct`. The init-states files are LIBERO's own
trusted, pinned per-task numpy-array assets (not attacker-controlled
input), so `run_libero_suite_openpi.py` patches `torch.load` from the
outside to default `weights_only=False` when the caller doesn't specify it,
rather than editing the submodule.

**openpi LIBERO also silently returned 0% success rate on its first real
run (job 3824862) -- root cause was a websocket keepalive timeout, not the
checkpoint.** The server log showed exactly ONE connection opened/closed
across the whole ~30-minute, 500-episode `libero_spatial` run; the client
(`examples/libero/main.py`, unmodified) logged the identical "Caught
exception: sent 1011 (internal error) keepalive ping timeout; no close
frame received" for every one of the 500 episodes, so every "Success:
False" was actually a dropped connection, not a real (failed) rollout.
Root cause: `WebsocketPolicyServer._handler` calls `self._policy.infer(obs)`
synchronously inside its asyncio handler with no `await`, blocking the
whole event loop for the duration of inference -- and pi0's very first
inference call after server startup incurs JAX JIT compilation, which
routinely exceeds `websockets`' default 20s `ping_timeout`. The server
can't answer its own keepalive ping during that window, so the library
closes the connection right after the first request; `main.py` doesn't
reconnect on failure, so the rest of the suite silently fails on the same
dead connection. Fixed via `serve_policy_wrapper.py`'s new patch (d):
monkeypatches `websockets.asyncio.server.serve` (the function
`WebsocketPolicyServer.run()` calls) to default `ping_interval`/
`ping_timeout` to `None`, disabling the keepalive check entirely so no
inference latency can ever trigger this -- again without touching
`eval/models/openpi_official` itself.

**Correction: the server-side keepalive patch alone wasn't sufficient.**
Re-testing after the server-side fix (job 3824891, `NUM_TRIALS_PER_TASK=3`)
still failed identically on every episode. Root cause:
`openpi_client.websocket_client_policy.WebsocketClientPolicy` connects via
`websockets.sync.client.connect(...)`, which has its own, entirely
independent `ping_interval=20, ping_timeout=20` defaults -- the client's
own keepalive thread times out waiting for a pong while the server is
blocked on a slow/JIT-heavy `infer()` call, regardless of what the
server's `ping_interval`/`ping_timeout` are set to. Fixed by adding a
matching client-side patch, `run_libero_suite_openpi.py`'s
`_patch_disable_websocket_keepalive_timeout_client`, which monkeypatches
`websockets.sync.client.connect` itself the same way. Both the client-side
and server-side patches are needed together.

**Correction: matched values to the fork's own proven fix instead of fully
disabling keepalive.** Diffing `eval/models/openpi` (the robocasa-benchmark
fork used for RoboCasa) against `eval/models/openpi_official` revealed the
fork's own `websocket_client_policy.py` already independently hit and
fixed this exact bug -- it raises `ping_interval`/`ping_timeout` from the
library defaults (20s/20s) to `120s`/`600s` (configurable via
`OPENPI_WS_PING_INTERVAL`/`OPENPI_WS_PING_TIMEOUT`), rather than disabling
the check outright, and only patches the client side (its
`websocket_policy_server.py` is byte-identical to official's -- confirmed
via `diff`). This is corroborated by all 139 completed RoboCasa openpi
tasks showing zero keepalive failures. Updated both
`run_libero_suite_openpi.py`'s client-side patch and
`serve_policy_wrapper.py`'s server-side patch to use the same `120`/`600`
values instead of `None`/`None` -- functionally equivalent for our
purposes (a slow first-JIT call is well within 120s/600s), but keeps a
genuinely-dead connection detectable instead of hanging forever, matching
the fork's own tested behavior rather than inventing a new convention.

## LIBERO `max_episode_steps` (per-suite horizon) -- unified to openpi's convention

Unlike RoboCasa (which uses `get_task_horizon(task)`, a genuinely
per-*task* value, uniformly for all 8 models), LIBERO's per-model horizon
conventions found in each codebase were much coarser and inconsistent with
each other:

- **openpi** (`examples/libero/main.py`): per-*suite* fixed `max_steps`
  hardcoded inside `eval_libero()` -- 220/280/300/520/400 for spatial/
  object/goal/10/90 (uniform across all 10 tasks within a suite).
- **RLDX-1** (`run_scripts/eval/libero/eval_libero.sh`): a single flat
  `--max_episode_steps 720` for literally all 40 tasks across all 4
  suites -- no per-suite variation at all.
- **GR00T** (N1.5/N1.6): no LIBERO-specific horizon convention found in
  either codebase at time of writing.

**Decision: unify all LIBERO models onto openpi's own per-suite values**
(220/280/300/520 for spatial/object/goal/10 -- libero_90 excluded, out of
scope). This is the same "pick the officially-published, more principled
convention and apply it uniformly" reasoning already used for RoboCasa's
`get_task_horizon()`. Implemented via a `MAX_EPISODE_STEPS_LIST` parallel
array (mirroring the existing `N_EPISODES_LIST` pattern) in
`sbatch_rldx1_libero_test.sh` (grouped in blocks of 10 matching
`task_names`' suite order: libero_10, libero_goal, libero_object,
libero_spatial) and `eval_groot_n1d6_libero_single_task.sh`
(list support added, not yet wired into a sweep launcher since GR00T
LIBERO hasn't been run yet). `n_action_steps` was already consistent
per-model between RoboCasa and LIBERO (openpi 5, RLDX-1 8, GR00T-N1.6 8)
and needed no change.

## Comparison against the RoboCasa leaderboard (grootn16, RLDX-1)

The RoboCasa leaderboard (https://github.com/robocasa-benchmark/leaderboard)
reports RoboCasa's Multitask Learning setting as three category averages
rather than 50 flat per-task numbers:
[`gr00t_n1.6_2026_05_14.md`](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/gr00t_n1.6_2026_05_14.md),
[`rldx-1_2026_05_20.md`](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions_md/rldx-1_2026_05_20.md).
Both submissions' listed commit hashes match this project's own pinned
submodule commits exactly (`grootn16` @ `a21fc9af...`, `RLDX-1` @
`ef05cd4a...`), confirming we're running the identical checkpoint/code the
leaderboard numbers were measured on.

**Category derivation.** `eval/simulators/robocasa/robocasa/utils/
dataset_registry.py` tags each task's data path with `atomic`/`composite`
and `pretrain`/`target`. Checked all 50 tasks in our shared `task_names`
array (`sbatch_grootn16_test.sh`/`sbatch_rldx1_test.sh`/etc.) against this
registry: the array's first 18 are all `atomic` (all have a `pretrain`
path -> **Atomic-Seen**), the next 16 are `composite` with a `pretrain`
path (-> **Composite-Seen**), and the last 16 are `composite` with only a
`target` path, no `pretrain` path at all (-> **Composite-Unseen**, i.e.
tasks genuinely absent from the pretraining corpus, not just held-out
scenes of an otherwise-seen task) -- 18+16+16=50, exactly matching the
leaderboard's three-category breakdown.

**Results** (our `pretrain`-split reruns, weighted mean success rate per
category, n=50 episodes/task in every cell):

| Model | Category | Reproduced | Reported | Diff |
|---|---|---|---|---|
| grootn16 | Atomic-Seen | 52.6% | 51.1% | +1.5pp |
| grootn16 | Composite-Seen | 7.5% | 9.4% | -1.9pp |
| grootn16 | Composite-Unseen | 1.0% | 1.7% | -0.7pp |
| RLDX-1 | Atomic-Seen | 67.2% | 67.6% | -0.4pp |
| RLDX-1 | Composite-Seen | 26.1% | 27.9% | -1.8pp |
| RLDX-1 | Composite-Unseen | 10.5% | 8.5% | +2.0pp |

All six numbers land within ±2 percentage points of the leaderboard's
reported values -- a faithful reproduction, well within the noise expected
from a 50-episode-per-task sample and the known deliberate discrepancies
already logged above (RLDX-1's task horizon in particular).
