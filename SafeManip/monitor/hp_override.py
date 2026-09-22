"""Environment overrides for the numeric predicate hyperparameters.

WHY THIS EXISTS
The sensitivity analysis has to re-run the monitor under a grid of predicate
thresholds (SKILL_ONSET_FRAMES, FORBIDDEN_CONTACT_TOLERANCE_FRAMES, ...) and
compare each cell's agreement with the human annotations. Doing that by EDITING
predicates.py once per cell is how the corpus versions get mixed up: the code
that produced output/vN is then whatever happened to be checked out, and a
half-finished sweep leaves a working tree that no longer matches any committed
state. Instead the constants stay exactly as committed -- the default IS the
paper's configuration -- and a sweep cell is nothing but a set of environment
variables, recorded alongside the git hash in the run manifest.

USE
    SAFEMANIP_HP_SKILL_ONSET_FRAMES=20 python3 rerun_monitor_only.py ...

Naming: SAFEMANIP_HP_<CONSTANT NAME EXACTLY AS IN predicates.py>. A variable
whose suffix is not a known constant is a hard error, not a silent no-op -- a
typo'd override that quietly ran the default configuration would show up as
"this hyperparameter doesn't matter", which is precisely the wrong conclusion.

The type of the default decides the parse: an int constant takes an int, a float
constant takes a float. This keeps FORBIDDEN_CONTACT_TOLERANCE_FRAMES=20.5 from
silently becoming a float frame count.
"""
from __future__ import annotations

import os

PREFIX = "SAFEMANIP_HP_"


def apply_overrides(module, log=True):
    """Overwrite `module`'s numeric ALL_CAPS constants from the environment.

    Returns {name: (old, new)} for whatever was actually changed, so the caller
    can write it into the run manifest. Called at the END of the module (after
    every constant is bound) so it covers all of them without a per-constant
    opt-in that someone would forget to extend.
    """
    applied = {}
    known = {k: v for k, v in vars(module).items()
             if k.isupper() and isinstance(v, (int, float))
             and not isinstance(v, bool)}
    for env_key, raw in sorted(os.environ.items()):
        if not env_key.startswith(PREFIX):
            continue
        name = env_key[len(PREFIX):]
        if name not in known:
            # Only complain about names this module could plausibly own: the two
            # simulators have overlapping but not identical constant sets, so a
            # LIBERO-only override must not blow up the RoboCasa module.
            continue
        old = known[name]
        new = int(raw) if isinstance(old, int) else float(raw)
        setattr(module, name, new)
        applied[name] = (old, new)
    if applied and log:
        for name, (old, new) in applied.items():
            print(f"[hp_override] {module.__name__}.{name}: {old} -> {new}",
                  flush=True)
    return applied


def requested():
    """Every SAFEMANIP_HP_* in the environment, for the run manifest. Includes
    names no module claimed, so an override that matched nothing is visible in
    the manifest instead of vanishing."""
    return {k[len(PREFIX):]: v for k, v in os.environ.items()
            if k.startswith(PREFIX)}
