# Derives each property's pattern/trigger/obligation/resolve/guard/check
# directly from SafeManip/monitor/specs.py's actual `ltl` strings, via `ast`
# + regex -- instead of hand-typing a parallel copy in server.py's
# PROPERTY_META that can silently drift from the real spec (same idea as
# predicate_derive.py, which does this for "children"). specs.py is the
# source these were always meant to mirror (see PROPERTY_META's own comment
# above it) -- this makes that mirroring automatic and restart-verified
# instead of manual and unverified.
#
# Not imported at runtime (specs.py itself does `from monitor import
# predicates as R`, which would pull in predicates.py's real
# robosuite/mujoco dependencies -- server.py is stdlib-only, see its module
# docstring) -- parsed as source text instead, same approach as
# predicate_derive.py uses for predicates.py.
import ast
import re
from pathlib import Path

_SPEC_FUNCS = {"_spec", "_spec_intended_safety", "_spec_mechanism", "_spec_containment", "_spec_access_enclosure"}

# The only 3 LTL shapes used anywhere in specs.py (see PROPERTY_META's own
# comment for what each means) -- confirmed by construction: every one of
# the 19 current specs matches one of these three regexes with nothing left
# over. A future spec using a genuinely different shape would come back
# with parse_ltl_shape(...) -> None below rather than a wrong guess.
_INVARIANT_RE = re.compile(r"^G\(\s*!\s*(\w+)\s*\)$")
_UNTIL_RE = re.compile(r"^G\(\s*(\w+)\s*->\s*\(\s*(!?)\s*(\w+)\s*U\s*(\w+)\s*\)\s*\)$")
_INSTANT_RE = re.compile(r"^G\(\s*(\w+)\s*->\s*(\w+)\s*\)$")


def _collect_ltl_strings(tree) -> dict[str, str]:
    """property_name -> its literal `ltl` string, from every `_spec*(name,
    ltl, predicates, description)` call in specs.py."""
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _SPEC_FUNCS and len(node.args) >= 2):
            name_node, ltl_node = node.args[0], node.args[1]
            if isinstance(name_node, ast.Constant) and isinstance(ltl_node, ast.Constant):
                out[name_node.value] = ltl_node.value
    return out


def collect_predicate_lists(specs_py_path: Path) -> dict[str, list[str]]:
    """property_name -> its full `predicates` list (3rd positional arg to
    the `_spec*(name, ltl, predicates, description)` call) -- every atom
    specs.py names as relevant to this property, whether or not it appears
    in the `ltl` string itself (some are recovery/evidence atoms only).
    Used to auto-derive "extra_top" (whatever's left after trigger/
    obligation/resolve/guard/check and their own children are accounted
    for) instead of hand-picking it per property."""
    tree = ast.parse(specs_py_path.read_text(), filename=str(specs_py_path))
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _SPEC_FUNCS and len(node.args) >= 3):
            name_node, _ltl_node, preds_node = node.args[0], node.args[1], node.args[2]
            if isinstance(name_node, ast.Constant) and isinstance(preds_node, ast.List):
                preds = [el.value for el in preds_node.elts if isinstance(el, ast.Constant)]
                out[name_node.value] = preds
    return out


def parse_ltl_shape(ltl: str) -> dict | None:
    """One `ltl` string -> its PROPERTY_META shape dict (pattern +
    trigger/obligation/resolve/guard/check/obligation_kind, whichever
    apply), or None if it doesn't match any of the 3 known shapes."""
    m = _INVARIANT_RE.match(ltl)
    if m:
        return {"pattern": "invariant", "guard": m.group(1)}
    m = _UNTIL_RE.match(ltl)
    if m:
        trigger, neg, obligation, resolve = m.groups()
        return {
            "pattern": "until",
            "trigger": trigger,
            "obligation": obligation,
            "obligation_kind": "guard_false" if neg == "!" else "hold_true",
            "resolve": resolve,
        }
    m = _INSTANT_RE.match(ltl)
    if m:
        trigger, check = m.groups()
        return {"pattern": "instant", "trigger": trigger, "check": check}
    return None


def derive_property_shapes(specs_py_path: Path) -> dict[str, dict]:
    """{property_name: shape_dict} for every property spec.py defines whose
    `ltl` string matches a known shape. A property missing here (unparsed
    shape) should be treated as a hard error by the caller -- it means
    specs.py grew a 4th LTL pattern this module doesn't know how to read,
    not that the property has no shape at all."""
    tree = ast.parse(specs_py_path.read_text(), filename=str(specs_py_path))
    ltls = _collect_ltl_strings(tree)
    result = {}
    for name, ltl in ltls.items():
        shape = parse_ltl_shape(ltl)
        if shape is not None:
            result[name] = shape
    return result
