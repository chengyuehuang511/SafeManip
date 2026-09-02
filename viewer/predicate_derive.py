# Derives each composite predicate's real AND-composition (its "children" for
# the breakdown UI) directly from predicates.py's source via `ast`, instead of
# hand-maintaining a parallel copy in server.py's PROPERTY_META. The whole
# point: when predicates.py's code changes, this picks up the new structure
# automatically on the next server restart -- no manual re-sync step, and no
# way for the two to silently drift apart the way PROPERTY_META repeatedly did
# (see docs/predicate_ltl_design/CHANGES_2026-08-31.md and this file's own
# git history for several confirmed instances: object_stable vs
# object_stable_relative, liquid/solid_settled's copy-pasted children,
# object_sync's stale "velocity" description, object_supported vs
# object_supported_on_correct, and -- found while building *this* module --
# fixture_fully_open's oversimplified OR, reach_in_fixture's wrong
# access_active_fixture attribution, and object_settled's settle_obj_name vs
# active_object argument mismatch. See derive_all()'s docstring for what this
# does and does NOT attempt to get right.
import ast
from pathlib import Path


def _unparse(node):
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _collect_assignments(tree):
    """name -> last-assigned value expr, scanning the whole module. Safe
    here because every predicate this is used for lives as a local variable
    inside one large per-frame computation function in predicates.py, with a
    single straight-line assignment each -- last-assignment-wins matches
    normal Python semantics for what a later read of that name would see."""
    assigns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = node.value
    return assigns


def _collect_exported(tree):
    """exported dict-literal key (str) -> its value-expr AST node, from both
    the `predicates = {...}` and `violation_evidence = {...}` dict literals
    -- the two places server.py's _frame_predicate_value actually looks when
    reading a per-frame boolean out of the saved raw dump. A predicate is
    only "independently inspectable" in the UI if it (or an exactly-matching
    expression) appears here."""
    exported = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("predicates", "violation_evidence")
                and isinstance(node.value, ast.Dict)):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    exported.setdefault(k.value, v)
    return exported


def _collect_funcdefs(tree):
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
    return out


class _Sub(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, n):
        return self.mapping.get(n.id, n)


def _inline_call(node, funcdefs, depth):
    """If `node` calls a known simple nested `def f(params): return <expr>`,
    substitute params with this call's actual argument expressions and
    return the substituted body -- so e.g. object_settled's call into the
    nested _object_settled(settle_obj_name, ...) helper unfolds into that
    helper's real `and`-composition with `name` replaced by
    `settle_obj_name`, not left as an opaque call. Bails out (returns `node`
    unchanged) for anything not matching this exact shape (defaults/kwargs,
    multi-statement bodies, etc.) rather than guessing."""
    if depth > 6 or not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return node
    fn = funcdefs.get(node.func.id)
    if fn is None or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        return node
    ret_expr = fn.body[0].value
    if ret_expr is None:
        return node
    params = [a.arg for a in fn.args.args]
    if len(params) != len(node.args) or node.keywords:
        return node
    mapping = dict(zip(params, node.args))
    return _Sub(mapping).visit(ast.parse(_unparse(ret_expr), mode="eval").body)


def _flatten_and(node, funcdefs, assigns, exported, depth=0):
    """Top-level `and`-composition -> flat list of leaf expr nodes. Unwraps
    `_bool(...)`, flattens nested `and` BoolOps, inlines calls into known
    local helper functions, and transparently substitutes plain local
    variables that AREN'T themselves directly exported (so e.g.
    object_grasped_safe's `raw_object_grasped_safe` term gets replaced by
    ITS definition, `object_grasped and object_sync`, rather than stopping
    at an opaque unexported name).

    Deliberately does NOT decompose past an `or`, a `not`, or a Call/Name it
    can't resolve -- that's exactly the boundary where a purely syntactic
    tool can't safely infer which branch/negation semantics are safe to
    show as an independent timeline. Those show up as a single opaque leaf
    (usually with exported_key=None in derive_children's output) rather than
    a guessed partial decomposition."""
    if depth > 8:
        return [node]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_bool" and len(node.args) == 1:
        return _flatten_and(node.args[0], funcdefs, assigns, exported, depth + 1)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out = []
        for v in node.values:
            out.extend(_flatten_and(v, funcdefs, assigns, exported, depth + 1))
        return out
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        inlined = _inline_call(node, funcdefs, depth)
        if inlined is not node:
            return _flatten_and(inlined, funcdefs, assigns, exported, depth + 1)
    if isinstance(node, ast.Name):
        is_directly_exported = any(
            isinstance(vexpr, ast.Name) and vexpr.id == node.id for vexpr in exported.values()
        )
        if not is_directly_exported and node.id in assigns:
            return _flatten_and(assigns[node.id], funcdefs, assigns, exported, depth + 1)
    return [node]


def _derive_one(name, assigns, exported, funcdefs):
    """Returns (children, note) for one predicate name. `children` is a list
    of exported-key strings (deduped, order-preserving) that are safely
    displayable as their own timeline -- or None if there's nothing safe to
    show (no assignment found, the expression isn't a multi-term `and`, or
    none of its terms map to a verified exported key)."""
    node = assigns.get(name)
    if node is None:
        return None, "no assignment found for this name"
    leaves = _flatten_and(node, funcdefs, assigns, exported)
    if len(leaves) <= 1:
        return None, "not a multi-term 'and' (e.g. an 'or', a single term, or a rename)"
    keys, seen = [], set()
    for leaf in leaves:
        raw = _unparse(leaf)
        mapped_key = None
        if isinstance(leaf, ast.Name):
            for key, vexpr in exported.items():
                if key == name:
                    continue  # a predicate can't be its own child
                if isinstance(vexpr, ast.Name) and vexpr.id == leaf.id:
                    mapped_key = key
                    break
        else:
            # An exported key's own definition sometimes wraps the exact
            # same call this leaf is with an extra None/existence guard --
            # e.g. gripper_away_from_object = `_bool(settle_obj_name is not
            # None and _gripper_far_from_object(settle_obj_name))`, which is
            # the same call object_settled's own formula uses. Only strip
            # *guard-shaped* leaves (a bare `Compare`, e.g. "X is not None"
            # or "X >= N") before comparing, and only accept the match if
            # exactly ONE substantive (non-guard) leaf remains and it's
            # this exact leaf -- NOT "this leaf appears anywhere among that
            # key's leaves", which would also match e.g. skill_dump_onset
            # (a 7-term AND that happens to also include `not
            # object_released` as one of its unrelated terms, purely
            # coincidentally). Starts from assigns[key] (the key's own
            # local-variable assignment), not exported[key] (the
            # dict-literal's value expression, almost always just a bare
            # `Name(key)` pointing back at that same local variable, which
            # would make _flatten_and treat it as trivially "already
            # exported" and refuse to unfold it any further).
            for key in exported:
                if key == name:
                    continue  # a predicate can't be its own child
                formula_node = assigns.get(key, exported[key])
                exported_leaves = _flatten_and(formula_node, funcdefs, assigns, exported)
                substantive = [l for l in exported_leaves if not isinstance(l, ast.Compare)]
                if len(substantive) == 1 and _unparse(substantive[0]) == raw:
                    mapped_key = key
                    break
        if mapped_key is not None and mapped_key not in seen:
            seen.add(mapped_key)
            keys.append(mapped_key)
    if not keys:
        return None, "real components found, but none map to a verified exported per-frame key"
    return keys, None


def derive_all_formulas(predicates_py_path: Path) -> dict[str, str]:
    """name -> ast.unparse(...) of its assignment's right-hand side in
    predicates.py, for every local variable assigned anywhere in the file --
    the literal source expression (e.g. "_bool(not object_released and
    raw_object_grasped_safe)" for object_grasped_safe), not a hand-written
    paraphrase. Used as the breakdown UI's hover-tooltip "how this is
    computed" text instead of a manually-maintained description dict, so
    there is nothing here that can drift from the code -- if predicates.py
    changes, the shown formula changes on the next restart."""
    tree = ast.parse(predicates_py_path.read_text(), filename=str(predicates_py_path))
    assigns = _collect_assignments(tree)
    return {name: _unparse(expr) for name, expr in assigns.items()}


def collect_exported_key_names(predicates_py_path: Path) -> set[str]:
    """Every key name actually written to the saved per-frame dump (both
    `sections.predicates` and `violation_evidence`) -- lets a caller
    self-check that some *other* hand-typed or derived name (e.g. a
    trigger/obligation/resolve parsed out of specs.py's ltl strings) really
    corresponds to something the raw data can show, rather than assuming
    the two files agree."""
    tree = ast.parse(predicates_py_path.read_text(), filename=str(predicates_py_path))
    return set(_collect_exported(tree).keys())


def derive_all(predicates_py_path: Path, names: list[str]) -> dict[str, list[str] | None]:
    """For each name in `names`, derive its real children (see _derive_one).
    Parses predicates.py once. Returns {name: children_or_None} -- callers
    should only use entries with a non-None value; None means "nothing safe
    to show", same as simply omitting a "children" entry did before."""
    src = predicates_py_path.read_text()
    tree = ast.parse(src, filename=str(predicates_py_path))
    assigns = _collect_assignments(tree)
    exported = _collect_exported(tree)
    funcdefs = _collect_funcdefs(tree)
    result = {}
    for name in names:
        children, _note = _derive_one(name, assigns, exported, funcdefs)
        result[name] = children
    return result
