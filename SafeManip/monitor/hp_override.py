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

SETATTR ALONE IS NOT ENOUGH -- see propagate_derived().
Some constants are read into a SECOND binding while the module is still
executing, i.e. before the override hook at the bottom of the file ever runs:

    RETRACT_TIMEOUT_FRAMES = SETTLE_TIMEOUT_FRAMES       # both predicate modules
    NEAR_OBJECT_THRESHOLD  = REACH_THRESHOLD             # libero
    def _persistent_bool_sticky_true(..., n=STABLE_PERSISTENCE_FRAMES)  # libero

Rebinding only the source constant leaves those frozen at the committed value,
so the cell runs a HALF-applied configuration and reports less sensitivity than
the threshold really has -- "this hyperparameter doesn't matter" from a harness
bug, which is exactly the wrong conclusion. propagate_derived() therefore re-
evaluates every such binding from the module's own source. It is driven by the
AST, not by a hand-maintained list of the four known cases: a new alias added to
predicates.py later is picked up automatically, per the general-fix rule.
"""
from __future__ import annotations

import ast
import inspect
import os

PREFIX = "SAFEMANIP_HP_"


def _module_ast(module):
    """The module's parsed source, or None if it cannot be read.

    Unreadable source must not be fatal -- it only means the derived-binding pass
    is skipped, and the caller is told so rather than silently getting a
    half-applied configuration.
    """
    path = getattr(module, "__file__", None)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return ast.parse(f.read())
    except (OSError, SyntaxError):
        return None


def _numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _eval_in(module, node):
    """Evaluate one expression node against the module's CURRENT namespace."""
    return eval(compile(ast.Expression(body=node), "<hp_override>", "eval"),
                vars(module))


def _iter_patchable_functions(tree, module):
    """(ast node, live function object) for every def whose defaults are frozen
    at definition time and are still reachable from the module namespace: module-
    level functions and methods of module-level classes.

    Functions nested inside another function are deliberately NOT here -- their
    defaults are re-evaluated every time the enclosing def executes, so a global
    rebound before the first call is picked up on its own. (This is why
    RoboCasa's two `threshold=PERSISTENCE_FRAMES` helpers inside
    build_predicate_snapshot need no patching.)
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = getattr(module, node.name, None)
            if fn is not None:
                yield node, fn
        elif isinstance(node, ast.ClassDef):
            cls = getattr(module, node.name, None)
            if cls is None:
                continue
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fn = cls.__dict__.get(sub.name)
                    if isinstance(fn, staticmethod):
                        fn = fn.__func__
                    if callable(fn):
                        yield sub, fn


def propagate_derived(module, changed, log=True):
    """Re-evaluate every binding that was computed FROM one of `changed` while
    the module was still executing, so an override is applied whole.

    Two such bindings exist, both invisible to a plain setattr:

      1. module-level derived constants -- `RETRACT_TIMEOUT_FRAMES =
         SETTLE_TIMEOUT_FRAMES`. Re-evaluated in source order and to a fixed
         point, so a chain (A -> B -> C) fully propagates.
      2. default arguments -- `def f(..., n=STABLE_PERSISTENCE_FRAMES)`.
         Evaluated once at def time; patched here through __defaults__ /
         __kwdefaults__.

    Returns {name: (old, new)} for derived constants and
    {"func:param": (old, new)} for defaults, merged, so the manifest records the
    whole applied configuration rather than just the knob that was asked for.
    """
    tree = _module_ast(module)
    if tree is None:
        if log:
            print(f"[hp_override] WARNING {module.__name__}: source unreadable, "
                  f"derived bindings NOT propagated", flush=True)
        return {}

    out, live = {}, set(changed)
    # (1) Derived module constants, to a fixed point. Bounded by the number of
    # assignments, so a cyclic re-definition cannot spin forever.
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
    for _ in range(len(assigns) + 1):
        grew = False
        for node in assigns:
            refs = {x.id for x in ast.walk(node.value) if isinstance(x, ast.Name)}
            if not refs & live:
                continue
            for tgt in node.targets:
                if not isinstance(tgt, ast.Name) or not tgt.id.isupper():
                    continue
                old = getattr(module, tgt.id, None)
                if not _numeric(old):
                    continue
                try:
                    new = _eval_in(module, node.value)
                except Exception:  # noqa: BLE001 -- a non-constant RHS is not ours
                    continue
                if not _numeric(new) or new == old:
                    continue
                setattr(module, tgt.id, new)
                out[tgt.id] = (old, new)
                live.add(tgt.id)
                grew = True
        if not grew:
            break

    # (2) Frozen default arguments.
    for node, fn in _iter_patchable_functions(tree, module):
        a = node.args
        # Positional defaults line up with the LAST len(defaults) params, in the
        # same order as fn.__defaults__.
        pos = list(getattr(fn, "__defaults__", None) or ())
        for i, d in enumerate(a.defaults):
            refs = {x.id for x in ast.walk(d) if isinstance(x, ast.Name)}
            if not (refs & live) or i >= len(pos) or not _numeric(pos[i]):
                continue
            try:
                new = _eval_in(module, d)
            except Exception:  # noqa: BLE001
                continue
            if _numeric(new) and new != pos[i]:
                out[f"{node.name}(default #{i})"] = (pos[i], new)
                pos[i] = new
        if pos:
            fn.__defaults__ = tuple(pos)

        kw = dict(getattr(fn, "__kwdefaults__", None) or {})
        names = [k.arg for k in a.kwonlyargs]
        for name, d in zip(names, a.kw_defaults):
            if d is None or name not in kw or not _numeric(kw[name]):
                continue
            refs = {x.id for x in ast.walk(d) if isinstance(x, ast.Name)}
            if not refs & live:
                continue
            try:
                new = _eval_in(module, d)
            except Exception:  # noqa: BLE001
                continue
            if _numeric(new) and new != kw[name]:
                out[f"{node.name}({name}=)"] = (kw[name], new)
                kw[name] = new
        if kw:
            fn.__kwdefaults__ = kw

    if out and log:
        for name, (old, new) in sorted(out.items()):
            print(f"[hp_override] {module.__name__}.{name}: {old} -> {new} "
                  f"(derived)", flush=True)
    return out


def apply_overrides(module, log=True):
    """Overwrite `module`'s numeric ALL_CAPS constants from the environment, then
    propagate into everything that was derived from them.

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
    if applied:
        applied.update(propagate_derived(module, set(applied), log=log))
    return applied


def unmatched(*modules):
    """Requested override names that NO given module owns.

    apply_overrides() cannot raise on an unknown name by itself -- the two
    predicate modules have overlapping but not identical constant sets, so a
    RoboCasa-only knob legitimately matches nothing in the LIBERO module. The
    check only makes sense once every module that could own the name has been
    imported, which is the caller's business. A typo'd knob that matched nothing
    anywhere would otherwise run the DEFAULT configuration and be written up as
    "this hyperparameter doesn't matter".
    """
    owned = set()
    for m in modules:
        owned |= {k for k, v in vars(m).items() if k.isupper() and _numeric(v)}
    return sorted(set(requested()) - owned)


def requested():
    """Every SAFEMANIP_HP_* in the environment, for the run manifest. Includes
    names no module claimed, so an override that matched nothing is visible in
    the manifest instead of vanishing."""
    return {k[len(PREFIX):]: v for k, v in os.environ.items()
            if k.startswith(PREFIX)}
