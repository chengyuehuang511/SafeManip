"""
EXPERIMENTAL -- compares `monitor/sim/robocasa/attributes.py` (SafeManip's
own, hand-authored attribute logic) against
`monitor/sim/robocasa/attribute/native_structure.json` (robocasa's own
native taxonomy, extracted with zero SafeManip-authored data -- see that
package's module docstring) and reports, systematically rather than from
memory:

  1. What attributes.py got WRONG: categories/classes it hand-assigns an
     attribute that disagrees with what robocasa itself says (missing
     categories robocasa tags but attributes.py doesn't, categories
     attributes.py tags that robocasa doesn't, and axis members that are
     declared but never actually assigned to anything -- dead code).

  2. What attributes.py ADDS that robocasa has no equivalent for at all
     (attribute names that appear nowhere in robocasa's native `types` tag
     vocabulary or boolean capability flags) -- these aren't necessarily
     wrong, just not verifiable against robocasa's own data; each is
     labeled "used" or "declared but never assigned" (dead).

Not committed -- this is a throwaway comparison run, not part of the
package. Only needs `attributes.py` (pure ast/pathlib, no robocasa import)
plus the already-generated native_structure.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

THIS_DIR = Path(__file__).parent
MONITOR_PKG_ROOT = THIS_DIR.parents[3]  # .../SafeManip/SafeManip
NATIVE_JSON_PATH = THIS_DIR.parent / "attribute" / "native_structure.json"

sys.path.insert(0, str(MONITOR_PKG_ROOT))
from monitor.sim.robocasa import attributes as A  # noqa: E402


def load_native() -> Dict[str, Any]:
    return json.loads(NATIVE_JSON_PATH.read_text())


def native_categories_with_tag(native: Dict[str, Any], tag: str) -> Set[str]:
    return {c for c, info in native["objects"]["categories"].items() if tag in info["types"]}


def native_categories_with_flag(native: Dict[str, Any], flag: str) -> Set[str]:
    return {c for c, info in native["objects"]["categories"].items() if info["booleans"].get(flag)}


# ---------------------------------------------------------------------------
# 1. What attributes.py got wrong
# ---------------------------------------------------------------------------
def find_category_set_drift(native: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compares every attributes.py hand-written category set against the
    native `types` tag it's supposed to represent."""
    checks = [
        ("RECEPTACLE_CATEGORIES", "receptacle", A.RECEPTACLE_CATEGORIES),
        ("TOOL_CATEGORIES", "tool", A.TOOL_CATEGORIES),
        ("FRAGILE_CATEGORIES", None, A.FRAGILE_CATEGORIES),  # no native tag -- see section 2
    ]
    results = []
    for name, native_tag, hand_set in checks:
        if native_tag is None:
            continue
        native_set = native_categories_with_tag(native, native_tag)
        missing = sorted(native_set - hand_set)  # native says X, attributes.py misses it
        extra = sorted(hand_set - native_set)  # attributes.py says X, native disagrees
        results.append({
            "hand_written_set": name,
            "native_tag": native_tag,
            "native_count": len(native_set),
            "hand_written_count": len(hand_set),
            "missing_from_hand_written": missing,
            "extra_in_hand_written_not_native": extra,
        })
    return results


def find_tool_utensil_conflation(native: Dict[str, Any]) -> Dict[str, Any]:
    """robocasa keeps `tool` and `utensil` as two separate native tags;
    attributes.py's TOOL_CATEGORIES merges both into one bucket."""
    native_tool = native_categories_with_tag(native, "tool")
    native_utensil = native_categories_with_tag(native, "utensil")
    conflated = sorted(A.TOOL_CATEGORIES & native_utensil)
    return {
        "native_tool_categories": sorted(native_tool),
        "native_utensil_categories": sorted(native_utensil),
        "hand_written_TOOL_CATEGORIES_that_are_actually_native_utensil": conflated,
        "native_utensil_categories_missing_any_utensil_label_in_attributes_py": (
            sorted(native_utensil) if "utensil" not in A.ROLE_ATTRIBUTE_AXES.get("object", []) else []
        ),
    }


def find_dead_axis_members(native: Dict[str, Any]) -> Dict[str, List[str]]:
    """For every attribute name attributes.py declares under
    ROLE_ATTRIBUTE_AXES["object"], check whether object_category_attribute_
    defaults() (the actual derivation logic) ever assigns it to any
    category at all. Declared-but-never-assigned = dead code."""
    all_assigned: Set[str] = set()
    for attrs in A.OBJECT_CATEGORY_ATTRIBUTE_DEFAULTS.values():
        all_assigned.update(attrs)
    declared = set(A.ROLE_ATTRIBUTE_AXES.get("object", []))
    dead = sorted(declared - all_assigned)
    used = sorted(declared & all_assigned)
    return {"declared_object_axis_members": sorted(declared), "dead_never_assigned": dead, "actually_used": used}


def _fixture_assigned_by_namespace() -> Dict[str, Set[str]]:
    """FIXTURE_CLASS_ATTRIBUTE_DEFAULTS mixes two namespaces of tag in one
    flat per-class list: bare fixture-role tags (e.g. `openable`) and
    `support:`-prefixed support-role tags (e.g. `support:containment`).
    Keeping them separate matters -- conflating them (as an earlier version
    of this script did) makes the whole `support` role look 100% dead, when
    it's actually used, just under its own namespace."""
    bare: Set[str] = set()
    support: Set[str] = set()
    for attrs in A.FIXTURE_CLASS_ATTRIBUTE_DEFAULTS.values():
        for a in attrs:
            if a.startswith("support:"):
                support.add(a.split(":", 1)[1])
            else:
                bare.add(a)
    return {"fixture": bare, "support": support}


def find_dead_role_axis_members() -> Dict[str, List[str]]:
    """Same check as find_dead_axis_members(), but for the fixture/support/
    button/tool role axes -- `fixture`/`support` are checked against
    FIXTURE_CLASS_ATTRIBUTE_DEFAULTS's actual output (in their own
    namespace, see _fixture_assigned_by_namespace); `button`/`tool` have no
    derivation function anywhere in attributes.py, so every member is dead
    by construction."""
    assigned = _fixture_assigned_by_namespace()
    result = {}
    for role in ("fixture", "support", "button", "tool"):
        declared = set(A.ROLE_ATTRIBUTE_AXES.get(role, []))
        if role in assigned:
            dead = sorted(declared - assigned[role])
            used = sorted(declared & assigned[role])
        else:
            dead = sorted(declared)
            used = []
        result[role] = {"declared": sorted(declared), "dead_never_assigned": dead, "actually_used": used}
    return result


def find_dead_type_checks(native: Dict[str, Any]) -> List[Dict[str, str]]:
    """attributes.py's object_category_attribute_defaults() checks whether
    certain literal strings appear in a category's native `types` tuple --
    if that literal string is never actually a member of robocasa's own
    type_tag_vocabulary, the check can never fire. Read straight from the
    source rather than hand-listed, so this doesn't silently go stale
    itself."""
    import ast

    source = (MONITOR_PKG_ROOT / "monitor" / "sim" / "robocasa" / "attributes.py").read_text()
    tree = ast.parse(source)
    vocabulary = set(native["objects"]["type_tag_vocabulary"])
    dead_checks = []
    for node in ast.walk(tree):
        # matches `"literal" in types` / `"literal" in attrs` patterns
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.In)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
        ):
            literal = node.left.value
            comparator = node.comparators[0]
            target_name = getattr(comparator, "id", None)
            if target_name == "types" and literal not in vocabulary:
                dead_checks.append({"literal": literal, "checked_against": target_name, "line": node.lineno})
    return dead_checks


def find_unhandled_fixture_classes(native: Dict[str, Any]) -> Dict[str, Any]:
    """Every fixture leaf class robocasa actually defines vs. every class
    attributes.py's FIXTURE_CLASS_ATTRIBUTE_DEFAULTS has an entry for."""
    class_parent = native["fixtures"]["class_parent"]
    native_classes = set(class_parent.keys())
    hand_classes = set(A.FIXTURE_CLASS_ATTRIBUTE_DEFAULTS.keys())
    return {
        "native_classes_missing_from_attributes_py": sorted(native_classes - hand_classes),
        "attributes_py_classes_not_in_native_hierarchy": sorted(hand_classes - native_classes),
    }


def find_unmodeled_accessory_names(native: Dict[str, Any]) -> Dict[str, Any]:
    """The ~30 counter-top items robocasa registers as Accessory/
    WallAccessory fixtures (distinguishable only by instance name, not
    class) that attributes.py gives literally zero attributes to, because
    it only ever keys off class name."""
    fixtures_map = native["fixtures"]["fixtures_name_to_class"]
    blank_classes = {c for c, attrs in A.FIXTURE_CLASS_ATTRIBUTE_DEFAULTS.items() if not attrs}
    affected_names = sorted(name for name, cls in fixtures_map.items() if cls in blank_classes)
    return {
        "fixture_classes_with_zero_attributes_in_attributes_py": sorted(blank_classes),
        "named_fixture_instances_that_therefore_get_zero_attributes": affected_names,
    }


def find_unused_fixture_type_enum(native: Dict[str, Any]) -> List[str]:
    """robocasa's FixtureType enum (used for scene-placement queries) is
    never referenced anywhere in attributes.py at all."""
    return sorted(native["fixtures"]["fixture_type_enum"])


# ---------------------------------------------------------------------------
# 2. What attributes.py adds that robocasa has no native equivalent for
# ---------------------------------------------------------------------------
def find_invented_attributes(native: Dict[str, Any]) -> Dict[str, Any]:
    """Every attribute name attributes.py declares (across all 5 role axes)
    that doesn't correspond to any native `types` tag or boolean capability
    flag -- accounting for the one intentional rename (`sweets` -> `sweet`).
    For each, reports whether object_category_attribute_defaults()/
    fixture_class_default_attributes() actually assign it to anything, or
    whether it's declared and never used."""
    native_vocab = set(native["objects"]["type_tag_vocabulary"]) | {"sweet"}  # sweets -> sweet rename
    native_bools = set(native["objects"]["boolean_capability_flags"])
    native_known = native_vocab | native_bools

    all_declared: Set[str] = set()
    for axis_members in A.ROLE_ATTRIBUTE_AXES.values():
        all_declared.update(axis_members)
    invented = sorted(all_declared - native_known)

    object_assigned: Set[str] = set()
    for attrs in A.OBJECT_CATEGORY_ATTRIBUTE_DEFAULTS.values():
        object_assigned.update(attrs)
    fixture_namespaces = _fixture_assigned_by_namespace()

    rows = []
    for attr in invented:
        used_in = []
        if attr in object_assigned:
            used_in.append("object")
        if attr in fixture_namespaces["fixture"]:
            used_in.append("fixture")
        if attr in fixture_namespaces["support"]:
            used_in.append("support")
        rows.append({"attribute": attr, "status": "used" if used_in else "declared but never assigned", "used_in": used_in})
    return {"invented_attribute_count": len(invented), "invented_attributes": rows}


def find_invented_category_sets() -> Dict[str, List[str]]:
    """Whole category-set constants that exist purely in attributes.py, with
    no native tag/flag backing them at all (as opposed to RECEPTACLE_
    CATEGORIES/TOOL_CATEGORIES, which *should* mirror a native tag and were
    checked for drift in section 1)."""
    return {
        "FRAGILE_CATEGORIES": sorted(A.FRAGILE_CATEGORIES),
        "LIQUID_CATEGORIES": sorted(A.LIQUID_CATEGORIES),
        "POURABLE_CATEGORIES": sorted(A.POURABLE_CATEGORIES),
        "TWISTABLE_CATEGORIES": sorted(A.TWISTABLE_CATEGORIES),
        "OPENABLE_OBJECT_CATEGORIES": sorted(A.OPENABLE_OBJECT_CATEGORIES),
        "READY_TO_EAT_TYPE_NAMES": sorted(A.READY_TO_EAT_TYPE_NAMES),
    }


def find_invented_role_axes() -> Dict[str, List[str]]:
    """Whole role axes with no native robocasa equivalent concept at all
    (robocasa has no notion of a fixture "support zone", a "button" role,
    or task-tool sub-roles)."""
    return {
        "support (support-zone semantics -- containment/heated/wash_zone/...)": A.SUPPORT_ATTRIBUTE_AXES,
        "button (lever/toggle/power/mode_select/...)": A.BUTTON_ATTRIBUTE_AXES,
        "tool (cleaning_tool/cutting_tool/mixing_tool/.../measuring_tool)": A.TOOL_ATTRIBUTE_AXES,
    }


def build_report() -> Dict[str, Any]:
    native = load_native()
    return {
        "1_what_attributes_py_got_wrong": {
            "category_set_drift_vs_native_tags": find_category_set_drift(native),
            "tool_utensil_conflation": find_tool_utensil_conflation(native),
            "dead_object_axis_members": find_dead_axis_members(native),
            "dead_role_axis_members": find_dead_role_axis_members(),
            "dead_native_type_checks_in_source": find_dead_type_checks(native),
            "fixture_class_coverage_gap": find_unhandled_fixture_classes(native),
            "unmodeled_accessory_instances": find_unmodeled_accessory_names(native),
            "unused_native_fixture_type_enum": find_unused_fixture_type_enum(native),
        },
        "2_what_attributes_py_adds_beyond_robocasa": {
            "invented_attribute_names": find_invented_attributes(native),
            "invented_category_sets": find_invented_category_sets(),
            "invented_role_axes": find_invented_role_axes(),
        },
    }


def render_report(report: Dict[str, Any]) -> str:
    lines = []

    lines.append("=" * 78)
    lines.append("1. WHAT attributes.py GOT WRONG (vs. robocasa's own native data)")
    lines.append("=" * 78)

    lines.append("\n-- category-set drift vs. the native `types` tag it's meant to mirror --")
    for row in report["1_what_attributes_py_got_wrong"]["category_set_drift_vs_native_tags"]:
        lines.append(
            f"{row['hand_written_set']} (should mirror native tag `{row['native_tag']}`, "
            f"{row['native_count']} native vs {row['hand_written_count']} hand-written):"
        )
        lines.append(f"  MISSING (native has it, attributes.py doesn't): {', '.join(row['missing_from_hand_written']) or '(none)'}")
        lines.append(f"  EXTRA (attributes.py claims it, native disagrees): {', '.join(row['extra_in_hand_written_not_native']) or '(none)'}")

    tu = report["1_what_attributes_py_got_wrong"]["tool_utensil_conflation"]
    lines.append("\n-- tool/utensil conflation --")
    lines.append(f"native `tool` tag ({len(tu['native_tool_categories'])}): {', '.join(tu['native_tool_categories'])}")
    lines.append(f"native `utensil` tag ({len(tu['native_utensil_categories'])}): {', '.join(tu['native_utensil_categories'])}")
    lines.append(
        f"TOOL_CATEGORIES entries that are actually native `utensil`, not `tool` "
        f"({len(tu['hand_written_TOOL_CATEGORIES_that_are_actually_native_utensil'])}): "
        f"{', '.join(tu['hand_written_TOOL_CATEGORIES_that_are_actually_native_utensil'])}"
    )

    dead_obj = report["1_what_attributes_py_got_wrong"]["dead_object_axis_members"]
    lines.append("\n-- object-axis members declared but never assigned to any category (dead code) --")
    lines.append(f"declared ({len(dead_obj['declared_object_axis_members'])}): {', '.join(dead_obj['declared_object_axis_members'])}")
    lines.append(f"DEAD ({len(dead_obj['dead_never_assigned'])}): {', '.join(dead_obj['dead_never_assigned']) or '(none)'}")

    lines.append("\n-- fixture/support/button/tool role axes: declared vs. actually assigned --")
    for role, info in report["1_what_attributes_py_got_wrong"]["dead_role_axis_members"].items():
        lines.append(f"{role} role -- declared ({len(info['declared'])}): {', '.join(info['declared'])}")
        lines.append(f"  DEAD ({len(info['dead_never_assigned'])}): {', '.join(info['dead_never_assigned']) or '(none)'}")

    lines.append("\n-- dead `\"literal\" in types` checks in attributes.py's own source (literal never in robocasa's vocabulary) --")
    for row in report["1_what_attributes_py_got_wrong"]["dead_native_type_checks_in_source"]:
        lines.append(f"  line {row['line']}: \"{row['literal']}\" in {row['checked_against']}  (never true -- \"{row['literal']}\" is not a native types tag)")

    fc = report["1_what_attributes_py_got_wrong"]["fixture_class_coverage_gap"]
    lines.append("\n-- fixture class coverage --")
    lines.append(f"native classes missing from attributes.py ({len(fc['native_classes_missing_from_attributes_py'])}): {', '.join(fc['native_classes_missing_from_attributes_py']) or '(none)'}")
    lines.append(f"attributes.py classes not in native hierarchy ({len(fc['attributes_py_classes_not_in_native_hierarchy'])}): {', '.join(fc['attributes_py_classes_not_in_native_hierarchy']) or '(none)'}")

    ua = report["1_what_attributes_py_got_wrong"]["unmodeled_accessory_instances"]
    lines.append("\n-- named fixture instances that get zero attributes (class-only lookup can't distinguish them) --")
    lines.append(f"blank classes: {', '.join(ua['fixture_classes_with_zero_attributes_in_attributes_py'])}")
    lines.append(f"affected named instances ({len(ua['named_fixture_instances_that_therefore_get_zero_attributes'])}): {', '.join(ua['named_fixture_instances_that_therefore_get_zero_attributes'])}")

    fte = report["1_what_attributes_py_got_wrong"]["unused_native_fixture_type_enum"]
    lines.append(f"\n-- robocasa's FixtureType enum, entirely unused by attributes.py ({len(fte)}) --")
    lines.append(f"  {', '.join(fte)}")

    lines.append("\n" + "=" * 78)
    lines.append("2. WHAT attributes.py ADDS that robocasa has no native equivalent for")
    lines.append("=" * 78)

    inv = report["2_what_attributes_py_adds_beyond_robocasa"]["invented_attribute_names"]
    lines.append(f"\n-- invented attribute names ({inv['invented_attribute_count']}) --")
    for row in inv["invented_attributes"]:
        lines.append(f"  {row['attribute']}: {row['status']}" + (f" (in {'+'.join(row['used_in'])})" if row["used_in"] else ""))

    lines.append("\n-- invented, fully hand-written category sets (no native tag/flag backs them) --")
    for name, members in report["2_what_attributes_py_adds_beyond_robocasa"]["invented_category_sets"].items():
        lines.append(f"  {name} ({len(members)}): {', '.join(members)}")

    lines.append("\n-- invented role axes (whole concepts robocasa has no equivalent for) --")
    for name, members in report["2_what_attributes_py_adds_beyond_robocasa"]["invented_role_axes"].items():
        lines.append(f"  {name} ({len(members)}): {', '.join(members)}")

    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    (THIS_DIR / "diff_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    text = render_report(report)
    (THIS_DIR / "diff_report.txt").write_text(text)
    print(f"wrote {THIS_DIR / 'diff_report.json'}")
    print(f"wrote {THIS_DIR / 'diff_report.txt'}")
    print()
    print(text)
