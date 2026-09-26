"""Materialize the ManipVerse ontology (manipverse.tex's dataset) to disk.

Sources (same ones the monitor itself uses -- nothing hand-typed here):
  - RoboCasa: monitor/sim/robocasa/attributes.py's derived per-category and
    per-fixture-class attribute defaults (themselves built from robocasa's
    native kitchen_objects.py metadata + fixture class hierarchy).
  - LIBERO: monitor/sim/libero/attributes.py's name-substring mirror rules,
    materialized against the actual object/fixture categories appearing in
    the 40 in-scope task bddl files (libero_spatial/object/goal/10).

Outputs (next to this script):
  - manipverse.json                 full ontology + per-source stats
  - manipverse_associations.csv     long format: source,role,entity,property

Run:  python3 generate_manipverse.py
(Only needs stdlib -- attributes modules are loaded standalone, bypassing
the monitor package __init__ that imports robosuite.)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # .../projects/SafeManip
RC_ATTR = REPO / "SafeManip/monitor/sim/robocasa/attributes.py"
LIB_ATTR = REPO / "SafeManip/monitor/sim/libero/attributes.py"
BDDL_ROOT = REPO / "eval/simulators/libero/libero/libero/bddl_files"
LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _load_standalone(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# LIBERO: parse the 40 in-scope bddl files for the real category corpus
# ---------------------------------------------------------------------------
_BLOCK_RE = {
    "objects": re.compile(r"\(:objects(.*?)\)\s*\(:", re.S),
    "fixtures": re.compile(r"\(:fixtures(.*?)\)\s*\(:", re.S),
}


def _bddl_categories(text: str, block: str) -> set[str]:
    m = _BLOCK_RE[block].search(text)
    if not m:
        return set()
    cats = set()
    for line in m.group(1).splitlines():
        parts = line.split("-")
        if len(parts) == 2 and parts[0].strip():
            cats.add(parts[1].strip().lower())
    return cats


def libero_corpus() -> tuple[set[str], set[str], int]:
    objects: set[str] = set()
    fixtures: set[str] = set()
    n_tasks = 0
    for suite in LIBERO_SUITES:
        for bddl in sorted((BDDL_ROOT / suite).glob("*.bddl")):
            text = bddl.read_text()
            objects |= _bddl_categories(text, "objects")
            fixtures |= _bddl_categories(text, "fixtures")
            n_tasks += 1
    fixtures.discard("floor")  # scene root, not a manipulable fixture
    return objects, fixtures, n_tasks


# Property name -> substring-tuple attribute in monitor/sim/libero/attributes.py.
# graspable and receptacle are handled specially below.
_LIBERO_PROPERTY_SETS = {
    "fragile": "FRAGILE_NAME_SUBSTRINGS",
    "liquid": "LIQUID_NAME_SUBSTRINGS",
    "raw": "RAW_NAME_SUBSTRINGS",
    "ready_to_eat": "RTE_NAME_SUBSTRINGS",
    "microwavable": "MICROWAVABLE_NAME_SUBSTRINGS",
    "toastable": "TOASTABLE_NAME_SUBSTRINGS",
    "tool": "TOOL_NAME_SUBSTRINGS",
    "utensil": "UTENSIL_NAME_SUBSTRINGS",
    "pourable": "POURABLE_NAME_SUBSTRINGS",
    "fruit": "FRUIT_NAME_SUBSTRINGS",
    "vegetable": "VEGETABLE_NAME_SUBSTRINGS",
    "meat": "MEAT_NAME_SUBSTRINGS",
    "dairy": "DAIRY_NAME_SUBSTRINGS",
    "bread_food": "BREAD_FOOD_NAME_SUBSTRINGS",
    "cooked_food": "COOKED_FOOD_NAME_SUBSTRINGS",
    "condiment": "CONDIMENT_NAME_SUBSTRINGS",
    "spice": "SPICE_NAME_SUBSTRINGS",
    "pastry": "PASTRY_NAME_SUBSTRINGS",
    "sweet": "SWEET_NAME_SUBSTRINGS",
    "drink": "DRINK_NAME_SUBSTRINGS",
    "alcohol": "ALCOHOL_NAME_SUBSTRINGS",
    "cookware": "COOKWARE_NAME_SUBSTRINGS",
    "food": "FOOD_NAME_SUBSTRINGS",
    "cookable": "COOKABLE_NAME_SUBSTRINGS",
    "fridgable": "FRIDGABLE_NAME_SUBSTRINGS",
    "freezable": "FREEZABLE_NAME_SUBSTRINGS",
    "washable": "WASHABLE_NAME_SUBSTRINGS",
    "dishwashable": "DISHWASHABLE_NAME_SUBSTRINGS",
}


def libero_object_attributes(lib, category: str) -> list[str]:
    attrs = {"graspable"}  # mirrors RoboCasa: unconditional for movable objects
    if lib.object_is_receptacle_category(category):
        attrs.add("receptacle")
    for prop, set_name in _LIBERO_PROPERTY_SETS.items():
        subs = getattr(lib, set_name)
        if any(s in category for s in subs):
            attrs.add(prop)
    return sorted(attrs)


# ---------------------------------------------------------------------------
# Provenance: not every association is native simulator metadata.
#   native  -- read directly off robocasa's kitchen_objects.py (a `types` tag
#              mapped one-to-one, or a boolean capability flag).
#   derived -- deterministic commonsense rule over native tags (e.g. meat and
#              not cooked_food -> raw; food-tag union -> food/cookable;
#              graspable backfilled unconditionally).
#   curated -- LLM-assisted manually curated commonsense sets in attributes.py
#              with no native counterpart (fragile, liquid, pourable,
#              tool/utensil, twistable/openable), ALL fixture/support-surface
#              semantics (native robocasa encodes only the fixture class
#              tree), and the entire LIBERO substring-rule mirror.
# ---------------------------------------------------------------------------
_TAG_DIRECT = {
    "fruit", "vegetable", "meat", "dairy", "bread_food", "cooked_food",
    "condiment", "spice", "pastry", "alcohol", "drink", "receptacle", "cookware",
}
_DERIVED_RULE_ATTRS = {"food", "ready_to_eat", "raw", "cookable", "toastable", "graspable"}


def _rc_native_attrs(rc, category: str) -> set[str]:
    info = rc.external_object_categories().get(category, {})
    types = set(info.get("types", ()))
    attrs = {t for t in types if t in _TAG_DIRECT}
    if "sweets" in types:
        attrs.add("sweet")
    for flag in rc.OBJECT_METADATA_BOOLEAN_ATTRIBUTES:
        if info.get(flag):
            attrs.add(flag)
    return attrs


def _rc_object_provenance(rc, category: str, attr: str) -> str:
    if attr in _rc_native_attrs(rc, category):
        return "native"
    if attr in _DERIVED_RULE_ATTRS:
        return "derived"
    return "curated"


def main() -> None:
    rc = _load_standalone("rc_attributes", RC_ATTR)
    lib = _load_standalone("libero_attributes", LIB_ATTR)

    # -- RoboCasa side: already fully materialized by the monitor module --
    rc_objects = {k: sorted(v) for k, v in rc.OBJECT_CATEGORY_ATTRIBUTE_DEFAULTS.items()}
    rc_fixtures = {k: sorted(v) for k, v in rc.FIXTURE_CLASS_ATTRIBUTE_DEFAULTS.items()}

    # -- LIBERO side: apply the substring rules to the real bddl corpus --
    lib_obj_cats, lib_fix_cats, n_tasks = libero_corpus()
    lib_objects = {c: libero_object_attributes(lib, c) for c in sorted(lib_obj_cats)}
    lib_fixtures = {
        c: sorted(lib.FIXTURE_CATEGORY_ATTRIBUTES.get(c, ()))
        for c in sorted(lib_fix_cats)
    }

    def _assoc(mapping):
        return sum(len(v) for v in mapping.values())

    def _distinct(*mappings):
        out = set()
        for m in mappings:
            for v in m.values():
                out.update(v)
        return sorted(out)

    def _provenance(source, role, entity, prop):
        if source == "robocasa" and role == "object":
            return _rc_object_provenance(rc, entity, prop)
        return "curated"

    prov_counts: dict[str, int] = {}
    for source, role, mapping in (
        ("robocasa", "object", rc_objects),
        ("robocasa", "fixture", rc_fixtures),
        ("libero", "object", lib_objects),
        ("libero", "fixture", lib_fixtures),
    ):
        for entity, props in mapping.items():
            for prop in props:
                p = _provenance(source, role, entity, prop)
                prov_counts[p] = prov_counts.get(p, 0) + 1

    dataset = {
        "meta": {
            "name": "ManipVerse",
            "description": (
                "Commonsense ontology of manipulation-relevant object/fixture "
                "properties backing SafeManip's semantic safety specifications."
            ),
            "generated_by": "eval/figures_out/manipverse/generate_manipverse.py",
            "sources": {
                "robocasa": str(RC_ATTR.relative_to(REPO)),
                "libero": str(LIB_ATTR.relative_to(REPO)),
                "libero_task_corpus": [f"{s} ({BDDL_ROOT.name})" for s in LIBERO_SUITES],
            },
        },
        "property_axes": {k: sorted(v) for k, v in rc.ROLE_ATTRIBUTE_AXES.items()},
        "robocasa": {"objects": rc_objects, "fixtures": rc_fixtures},
        "libero": {
            "n_tasks": n_tasks,
            "objects": lib_objects,
            "fixtures": lib_fixtures,
            "rules": {
                prop: list(getattr(lib, set_name))
                for prop, set_name in _LIBERO_PROPERTY_SETS.items()
            },
            "receptacle_rule": {
                "substrings": list(lib.RECEPTACLE_NAME_SUBSTRINGS),
                "exceptions": list(lib._RECEPTACLE_NAME_EXCEPTIONS),
            },
        },
        "stats": {
            "robocasa": {
                "object_categories": len(rc_objects),
                "fixture_classes": len(rc_fixtures),
                "object_associations": _assoc(rc_objects),
                "fixture_associations": _assoc(rc_fixtures),
                "distinct_properties": len(_distinct(rc_objects, rc_fixtures)),
            },
            "libero": {
                "tasks": n_tasks,
                "object_categories": len(lib_objects),
                "fixture_categories": len(lib_fixtures),
                "object_associations": _assoc(lib_objects),
                "fixture_associations": _assoc(lib_fixtures),
                "distinct_properties": len(_distinct(lib_objects, lib_fixtures)),
            },
            "total": {
                "entities": len(rc_objects) + len(rc_fixtures) + len(lib_objects) + len(lib_fixtures),
                "associations": _assoc(rc_objects) + _assoc(rc_fixtures) + _assoc(lib_objects) + _assoc(lib_fixtures),
                "distinct_properties": len(_distinct(rc_objects, rc_fixtures, lib_objects, lib_fixtures)),
                "associations_by_provenance": prov_counts,
            },
        },
    }

    (HERE / "manipverse.json").write_text(json.dumps(dataset, indent=2) + "\n")

    with (HERE / "manipverse_associations.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "role", "entity", "property", "provenance"])
        for source, role, mapping in (
            ("robocasa", "object", rc_objects),
            ("robocasa", "fixture", rc_fixtures),
            ("libero", "object", lib_objects),
            ("libero", "fixture", lib_fixtures),
        ):
            for entity, props in mapping.items():
                for prop in props:
                    w.writerow([source, role, entity, prop,
                                _provenance(source, role, entity, prop)])

    print(json.dumps(dataset["stats"], indent=2))


if __name__ == "__main__":
    main()
