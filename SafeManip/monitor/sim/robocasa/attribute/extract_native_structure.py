"""
Read robocasa's own native taxonomy data structures directly -- real Python
imports/introspection where that's cheap and safe, AST-parsing only for the
one module (kitchen_objects.py) that has import-time side effects we want to
avoid -- and return what's actually there for objects, fixtures, and scenes.

No SafeManip-authored data in here: this is purely "what robocasa itself
already encodes." `monitor/sim/robocasa/attributes.py` (this package's own
attribute taxonomy) is meant to be built *from* this module's output, not
from hand-typed category lists -- see that file's own module docstring for
the drift that caused (e.g. `basket` missing from a hand-written receptacle
list, despite robocasa itself tagging it `receptacle`).

`get_object_structure()` only needs `ast`/`pathlib` (safe to call from
anywhere). `get_fixture_structure()`/`get_scene_structure()` do real
`import robocasa...`, so they need to run inside the `robocasa` conda env,
same requirement as every other module in this package that touches
robocasa's environments (see e.g. `monitor/extract_privileged_from_dataset.py`'s
module docstring).

Run standalone to (re)generate `native_structure.json` next to this file:
    python3 -m monitor.sim.robocasa.attribute.extract_native_structure
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict


# Same vendored-robocasa-checkout convention as attributes.py's
# EXTERNAL_ROBOCASA_ROOT, just one directory deeper (this module lives in
# monitor/sim/robocasa/attribute/, attributes.py lives in
# monitor/sim/robocasa/), hence parents[5] instead of parents[4].
EXTERNAL_ROBOCASA_ROOT = Path(__file__).resolve().parents[5] / "robocasa"
THIS_DIR = Path(__file__).parent
NATIVE_STRUCTURE_JSON_PATH = THIS_DIR / "native_structure.json"


# ---------------------------------------------------------------------------
# OBJECTS -- robocasa/models/objects/kitchen_objects.py
# ---------------------------------------------------------------------------
def get_object_structure() -> Dict[str, Any]:
    """OBJ_CATEGORIES (per-category `types` tags + boolean capability flags)
    and OBJ_GROUPS (named lists layered on top of it), via AST so we don't
    have to import kitchen_objects.py (it has module-level side effects:
    populating OBJ_GROUPS by iterating every asset path on disk)."""
    path = EXTERNAL_ROBOCASA_ROOT / "robocasa" / "models" / "objects" / "kitchen_objects.py"
    tree = ast.parse(path.read_text())

    obj_categories: Dict[str, Any] = {}
    obj_groups_source: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "OBJ_CATEGORIES":
                    raw = eval(compile(ast.Expression(node.value), str(path), "eval"), {"dict": dict})
                    for cat, info in raw.items():
                        types = info.get("types", ())
                        if isinstance(types, str):
                            types = (types,)
                        obj_categories[cat] = {
                            "types": tuple(types),
                            "booleans": {k: v for k, v in info.items() if isinstance(v, bool)},
                        }
        # Hand-curated OBJ_GROUPS[...] = [...] assignments (skip the
        # programmatically-populated ones, e.g. `for t in all_types: ...`,
        # which just mirror `types` and add no new information).
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "OBJ_GROUPS"
            ):
                key = ast.literal_eval(target.slice)
                try:
                    obj_groups_source[key] = ast.literal_eval(node.value)
                except ValueError:
                    pass  # e.g. get_cats_by_type(...) calls -- not a literal

    all_type_tags = sorted({t for info in obj_categories.values() for t in info["types"]})
    all_boolean_keys = sorted({k for info in obj_categories.values() for k in info["booleans"]})
    return {
        "num_categories": len(obj_categories),
        "type_tag_vocabulary": all_type_tags,
        "boolean_capability_flags": all_boolean_keys,
        "hand_curated_obj_groups": obj_groups_source,
        "categories": obj_categories,
    }


# ---------------------------------------------------------------------------
# FIXTURES -- real Python class hierarchy + FixtureType enum + FIXTURES map
# ---------------------------------------------------------------------------
def get_fixture_structure() -> Dict[str, Any]:
    import robocasa.models.fixtures.fixture as fixture_mod
    import robocasa.models.fixtures.cabinets  # noqa: F401 (registers subclasses)
    import robocasa.models.fixtures.fridge  # noqa: F401
    import robocasa.models.fixtures.stove  # noqa: F401
    import robocasa.models.fixtures.accessories  # noqa: F401
    import robocasa.models.fixtures.others  # noqa: F401
    import robocasa.models.scenes.scene_builder as scene_builder_mod

    Fixture = fixture_mod.Fixture

    def all_subclasses(cls):
        seen = set()
        stack = [cls]
        while stack:
            c = stack.pop()
            for sub in c.__subclasses__():
                if sub not in seen:
                    seen.add(sub)
                    stack.append(sub)
        return seen

    # class name -> immediate parent class name, for every Fixture subclass
    # robocasa actually defines -- ground truth, read straight from Python's
    # own __bases__, not re-typed by hand.
    class_parent: Dict[str, str] = {}
    for cls in all_subclasses(Fixture):
        for base in cls.__bases__:
            if issubclass(base, Fixture) or base is Fixture:
                class_parent[cls.__name__] = base.__name__
                break

    fixture_type_enum = list(fixture_mod.FixtureType.__members__.keys())

    # scene_builder.FIXTURES: scene-XML fixture-type-string -> Python class
    fixtures_map = {k: v.__name__ for k, v in scene_builder_mod.FIXTURES.items()}

    return {
        "class_parent": class_parent,
        "fixture_type_enum": fixture_type_enum,
        "fixtures_name_to_class": fixtures_map,
    }


# ---------------------------------------------------------------------------
# SCENES -- robocasa/models/scenes/scene_registry.py
# ---------------------------------------------------------------------------
def get_scene_structure() -> Dict[str, Any]:
    import robocasa.models.scenes.scene_registry as scene_registry_mod

    layout_ids = {m.name: m.value for m in scene_registry_mod.LayoutType}
    style_ids = {m.name: m.value for m in scene_registry_mod.StyleType}
    return {
        "layout_ids": layout_ids,
        "layout_groups": scene_registry_mod.LAYOUT_GROUPS_TO_IDS,
        "style_ids": style_ids,
        "style_groups": scene_registry_mod.STYLE_GROUPS_TO_IDS,
    }


def get_native_structure() -> Dict[str, Any]:
    return {
        "objects": get_object_structure(),
        "fixtures": get_fixture_structure(),
        "scenes": get_scene_structure(),
    }


# ---------------------------------------------------------------------------
# General -> specific tree rendering.
#
# Fixtures already have a real is-a hierarchy (class_parent), so that tree is
# just class_parent inverted and walked from the root. Objects have no
# native parent/child relation at all -- only flat `types` tags per category
# -- so "general -> specific" here means "type tag -> every category that
# carries it"; a category with multiple tags legitimately appears under
# multiple parents (e.g. `strainer` under both `tool` and `receptacle`),
# which is expected, not a bug. Scenes are collapsed to group membership
# counts rather than enumerating all 60 bare numeric IDs, which carry no
# semantic information on their own.
# ---------------------------------------------------------------------------
def render_fixture_tree(structure: Dict[str, Any]) -> str:
    class_parent = structure["fixtures"]["class_parent"]
    children: Dict[str, list] = {}
    for child, parent in class_parent.items():
        children.setdefault(parent, []).append(child)

    lines = []

    def walk(name: str, depth: int) -> None:
        lines.append(("  " * depth) + "- " + name)
        for child in sorted(children.get(name, [])):
            walk(child, depth + 1)

    walk("Fixture", 0)
    return "\n".join(lines)


def _categories_by_tag(categories: Dict[str, Any]) -> Dict[str, list]:
    by_tag: Dict[str, list] = {}
    for category, info in categories.items():
        for tag in info["types"]:
            by_tag.setdefault(tag, []).append(category)
    return by_tag


def _categories_by_boolean_flag(categories: Dict[str, Any]) -> Dict[str, list]:
    """The 7 boolean capability flags (cookable/washable/microwavable/
    fridgable/freezable/dishwashable/graspable) OBJ_CATEGORIES carries per
    category, alongside `types` -- an independent axis, not a `types` tag,
    so it needs its own grouping rather than being folded into `by_tag`."""
    by_flag: Dict[str, list] = {}
    for category, info in categories.items():
        for flag, value in info["booleans"].items():
            if value:
                by_flag.setdefault(flag, []).append(category)
    return by_flag


def render_object_tree(structure: Dict[str, Any]) -> str:
    categories = structure["objects"]["categories"]
    by_tag = _categories_by_tag(categories)
    by_flag = _categories_by_boolean_flag(categories)

    lines = ["- object  (general: native `types` tag; specific: leaf category)"]
    for tag in sorted(by_tag):
        members = sorted(by_tag[tag])
        lines.append(f"  - {tag}  ({len(members)} categories)")
        for category in members:
            lines.append(f"    - {category}")
    lines.append("- object  (general: native boolean capability flag; specific: leaf category where it's True)")
    for flag in sorted(by_flag):
        members = sorted(by_flag[flag])
        lines.append(f"  - {flag}  ({len(members)} categories)")
        for category in members:
            lines.append(f"    - {category}")

    # Categories with 0 tags AND every boolean False (e.g. flour_bag) never
    # appear above at all -- called out explicitly so they don't silently
    # vanish from this rendering (see summarize_levels()'s own "0 tags"
    # section, which already covers the 0-tag case; this also catches
    # 0-tag-and-0-true-flags, the strictly worse case).
    untagged = sorted(
        c for c, info in categories.items()
        if not info["types"] and not any(info["booleans"].values())
    )
    if untagged:
        lines.append(f"- object  (categories with no native `types` tag and no True boolean flag at all: {len(untagged)})")
        for category in untagged:
            lines.append(f"    - {category}")
    return "\n".join(lines)


def render_scene_tree(structure: Dict[str, Any]) -> str:
    lines = []
    for axis in ("layout", "style"):
        ids = structure["scenes"][f"{axis}_ids"]
        groups = structure["scenes"][f"{axis}_groups"]
        id_to_name = {v: k for k, v in ids.items() if v > 0}
        lines.append(f"- {axis}  ({len(id_to_name)} numbered IDs, no native names)")
        for group_id, members in sorted(groups.items(), key=lambda kv: int(kv[0])):
            group_name = next((k for k, v in ids.items() if v == int(group_id)), group_id)
            member_range = f"{min(members)}-{max(members)}" if members else "-"
            lines.append(f"  - {group_name}  ({len(members)} ids: {member_range})")
    return "\n".join(lines)


def render_extras(structure: Dict[str, Any]) -> str:
    """The parts of native_structure.json that aren't tree-shaped at all, so
    they don't belong in render_fixture_tree/render_object_tree/
    render_scene_tree -- rendered here instead, so nothing extracted into
    the JSON goes unrendered anywhere:
      - hand_curated_obj_groups: ad hoc named object lists (task/scene
        ingredient sets), not an attribute axis.
      - fixture_type_enum: robocasa's FixtureType placement-role enum, used
        for scene-placement queries, orthogonal to the class_parent hierarchy
        (e.g. CABINET_SINGLE_DOOR/CABINET_DOUBLE_DOOR distinguish something
        the class hierarchy alone doesn't).
      - fixtures_name_to_class: scene_builder.FIXTURES, the mapping from
        scene-XML fixture-type string (what actually shows up as a fixture's
        instance-name prefix at runtime, e.g. "oil_bottle") to its Python
        class -- many-to-one (many names share one class, esp. Accessory).
    """
    lines = []
    obj_groups = structure["objects"]["hand_curated_obj_groups"]
    lines.append(f"- hand_curated_obj_groups (robocasa/kitchen_objects.py OBJ_GROUPS literals): {len(obj_groups)} groups")
    for name in sorted(obj_groups):
        members = obj_groups[name]
        lines.append(f"  - {name} ({len(members)}): {', '.join(members)}")

    fixture_type_enum = structure["fixtures"]["fixture_type_enum"]
    lines.append(f"- fixture_type_enum (robocasa FixtureType, placement-role queries): {len(fixture_type_enum)} members")
    lines.append(f"  {', '.join(fixture_type_enum)}")

    fixtures_map = structure["fixtures"]["fixtures_name_to_class"]
    by_class: Dict[str, list] = {}
    for name, cls in fixtures_map.items():
        by_class.setdefault(cls, []).append(name)
    lines.append(f"- fixtures_name_to_class (scene_builder.FIXTURES): {len(fixtures_map)} names -> {len(by_class)} classes")
    for cls in sorted(by_class):
        names = sorted(by_class[cls])
        lines.append(f"  - {cls} ({len(names)}): {', '.join(names)}")
    return "\n".join(lines)


def render_all_trees(structure: Dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "=== FIXTURE TREE ===\n" + render_fixture_tree(structure),
            "=== OBJECT TREE ===\n" + render_object_tree(structure),
            "=== SCENE TREE ===\n" + render_scene_tree(structure),
            "=== EXTRAS (non-tree-shaped native data) ===\n" + render_extras(structure),
        ]
    )


def summarize_levels(structure: Dict[str, Any]) -> str:
    """How many things sit at each depth of each tree, what that depth
    concretely *means*, and the full list of node names at every depth
    (not just a count) -- same level of detail as render_fixture_tree(),
    just flattened per-depth instead of nested as ASCII-art.

    Fixture depth is a real is-a chain (walked from class_parent); object
    "depth" is root -> native `types` tag -> category, since categories
    have no native parent/child relation among themselves -- a category
    can (and 17 of them do) appear at depth 2 under more than one depth-1
    tag, so depth-2 listings are nested under each tag they belong to."""
    lines = []

    # --- fixtures ---
    class_parent = structure["fixtures"]["class_parent"]
    children: Dict[str, list] = {}
    for child, parent in class_parent.items():
        children.setdefault(parent, []).append(child)
    depth_nodes: Dict[int, list] = {}

    def walk(name: str, depth: int) -> None:
        depth_nodes.setdefault(depth, []).append(name)
        for child in children.get(name, []):
            walk(child, depth + 1)

    walk("Fixture", 0)
    max_fixture_depth = max(depth_nodes)
    fixture_depth_meaning = {
        0: "root -- robocasa's abstract Fixture base class",
        1: "direct subclasses of Fixture",
        2: "subclasses of a depth-1 class (Cabinet/ProcGenFixture, Fridge, Stove, or WallAccessory)",
        3: "subclasses of Cabinet (the only depth-2 class with further subclasses)",
    }
    lines.append("=" * 78)
    lines.append("FIXTURE TREE -- a real is-a hierarchy (robocasa's own Python class")
    lines.append("inheritance, read from each class's __bases__); depth = inheritance")
    lines.append("distance from the Fixture base class.")
    lines.append("=" * 78)
    for depth in sorted(depth_nodes):
        names = sorted(depth_nodes[depth])
        meaning = fixture_depth_meaning.get(depth, "further subclasses")
        lines.append(f"depth {depth} ({meaning}): {len(names)} classes")
        lines.append(f"  {', '.join(names)}")
    lines.append(f"TOTAL fixture classes across depths 0-{max_fixture_depth}: {sum(len(v) for v in depth_nodes.values())}")
    lines.append("")

    # --- objects ---
    categories = structure["objects"]["categories"]
    by_tag = _categories_by_tag(categories)
    by_flag = _categories_by_boolean_flag(categories)
    zero_tag_categories = sorted(c for c, info in categories.items() if len(info["types"]) == 0)
    multi_tag_categories = sorted(
        (c, sorted(info["types"])) for c, info in categories.items() if len(info["types"]) > 1
    )
    lines.append("=" * 78)
    lines.append("OBJECT TREE -- categories have no native parent/child relation among")
    lines.append("themselves; depth = membership in robocasa's own OBJ_CATEGORIES `types`")
    lines.append("tag vocabulary. A category attaches under every tag it carries, so it")
    lines.append("can (and 17 of them do) appear at depth 2 under more than one depth-1")
    lines.append("parent -- these are listed once per parent tag below, not deduplicated.")
    lines.append("=" * 78)
    lines.append("depth 0 (root -- there is no real object superclass, just a label): 1")
    lines.append("  object")
    lines.append(f"depth 1 (native `types` tag, from OBJ_CATEGORIES): {len(by_tag)}")
    lines.append(f"  {', '.join(sorted(by_tag))}")
    lines.append(
        f"depth 2 (object category, nested under every depth-1 tag it carries): "
        f"{len(categories)} unique categories, {sum(len(v) for v in by_tag.values())} tag-attachments"
    )
    for tag in sorted(by_tag):
        members = sorted(by_tag[tag])
        lines.append(f"  under {tag} ({len(members)}): {', '.join(members)}")
    multi_tag_descriptions = [f"{category} [{'+'.join(tags)}]" for category, tags in multi_tag_categories]
    lines.append(f"categories with >1 tag (appear at depth 2 under multiple parents): {len(multi_tag_categories)}")
    lines.append(f"  {', '.join(multi_tag_descriptions)}")
    lines.append(f"categories with 0 tags (no depth-1 parent to attach under at all): {len(zero_tag_categories)}")
    lines.append(f"  {', '.join(zero_tag_categories)}")
    lines.append("")
    lines.append(
        "OBJECT_CATEGORIES also carries 7 boolean capability flags per category "
        "(cookable/washable/microwavable/fridgable/freezable/dishwashable/graspable) -- "
        "an axis independent of `types`, so it's a second, separate depth-1 branch below,"
    )
    lines.append("not folded into the tag branch above:")
    lines.append(f"depth 1 (boolean capability flag, from OBJ_CATEGORIES booleans): {len(by_flag)}")
    lines.append(f"  {', '.join(sorted(by_flag))}")
    lines.append(
        f"depth 2 (object category where that flag is True, nested under every flag it satisfies): "
        f"{sum(len(v) for v in by_flag.values())} flag-attachments"
    )
    for flag in sorted(by_flag):
        members = sorted(by_flag[flag])
        lines.append(f"  under {flag} ({len(members)}): {', '.join(members)}")
    lines.append("")

    # --- scenes ---
    lines.append("=" * 78)
    lines.append("SCENE TREE -- layout/style IDs are bare numbers with no native semantic")
    lines.append("name; depth = membership in robocasa's own *_GROUPS_TO_IDS partitions")
    lines.append("(scene_registry.py). Groups overlap (e.g. layout ISLAND and DINING share")
    lines.append("4 of their 5-6 IDs), so an ID can appear at depth 2 under multiple groups.")
    lines.append("=" * 78)
    for axis in ("layout", "style"):
        ids = structure["scenes"][f"{axis}_ids"]
        groups = structure["scenes"][f"{axis}_groups"]
        id_to_name = {v: k for k, v in ids.items() if v > 0}
        n_ids = len(id_to_name)
        lines.append(f"-- {axis} axis --")
        lines.append("depth 0 (root -- the axis itself, e.g. `layout_id`/`style_id` on an episode): 1")
        lines.append(f"  {axis}")
        lines.append(f"depth 1 (named ID-group, from {axis.upper()}_GROUPS_TO_IDS): {len(groups)}")
        group_descriptions = [
            f"{next((k for k, v in ids.items() if v == int(gid)), str(gid))} (id {gid}, {len(members)} members)"
            for gid, members in sorted(groups.items(), key=lambda kv: int(kv[0]))
        ]
        lines.append(f"  {', '.join(group_descriptions)}")
        lines.append(f"depth 2 (individual numeric ID, nested under every group it belongs to): {n_ids} unique IDs")
        for group_id, members in sorted(groups.items(), key=lambda kv: int(kv[0])):
            group_name = next((k for k, v in ids.items() if v == int(group_id)), str(group_id))
            member_names = [id_to_name.get(m, str(m)) for m in members]
            lines.append(f"  under {group_name} ({len(member_names)}): {', '.join(member_names)}")
        lines.append("")

    # --- extras: not tree-shaped, so not part of any depth count above,
    # but still part of native_structure.json -- listed here so nothing
    # extracted goes unaccounted for. ---
    lines.append("=" * 78)
    lines.append("EXTRAS -- non-tree-shaped native data (ad hoc lists / a flat enum / a")
    lines.append("name->class mapping), not part of any depth count above.")
    lines.append("=" * 78)
    lines.append(render_extras(structure))
    return "\n".join(lines)


if __name__ == "__main__":
    structure = get_native_structure()
    NATIVE_STRUCTURE_JSON_PATH.write_text(json.dumps(structure, indent=2, default=str, sort_keys=True))
    print(f"wrote {NATIVE_STRUCTURE_JSON_PATH}")
    tree_text = render_all_trees(structure)
    (THIS_DIR / "native_structure_tree.txt").write_text(tree_text)
    print(f"wrote {THIS_DIR / 'native_structure_tree.txt'}")
    summary_text = summarize_levels(structure)
    (THIS_DIR / "native_structure_level_summary.txt").write_text(summary_text)
    print(f"wrote {THIS_DIR / 'native_structure_level_summary.txt'}")
    print()
    print(tree_text)
    print()
    print(summary_text)
