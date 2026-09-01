"""
Per-task summary of every robocasa task (atomic + composite): its scene
constraint, fixture references, object configs, and success-check
entities -- read from the *pristine* robocasa checkout's actual task
classes (via real Python import + inheritance-aware introspection, not
AST-per-class -- atomic tasks in particular have deep inheritance chains,
e.g. `OpenCabinet(OpenDoor(ManipulateDoor(Kitchen)))`, where the methods
that matter are defined on a *base* class, not the leaf task class itself)
-- plus the skill/fixture (atomic) and activity/category (composite) labels
robocasa's own documentation website computes on top of the same task list
(`docs/atomic_tasks/atomic_tasks.js`, `docs/composite_tasks/
composite_tasks_dropdown.js`) -- ported to Python here since that
classification lives only in docs-site JS, not in any Python module (see
this repo's earlier discussion; confirmed by grepping robocasa/ for the
skill-bucket ids -- zero hits outside docs/).

Every obj_groups/FixtureType value found is cross-referenced against
`monitor/sim/robocasa/attribute/native_structure.json` (the same native
robocasa taxonomy already extracted there) so a task's requirements are
expressed in terms of that tree's actual names, not just raw robocasa
identifiers.

Run inside the `robocasa` conda env (real robocasa imports, but only to
inspect class definitions -- no environment is ever instantiated/reset):
    python3 -m monitor.sim.robocasa.task.extract_tasks
"""
from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

THIS_DIR = Path(__file__).parent
MONITOR_PKG_ROOT = THIS_DIR.parents[2]  # .../SafeManip/SafeManip
ATTRIBUTE_DIR = THIS_DIR.parent / "attribute"
NATIVE_STRUCTURE_JSON_PATH = ATTRIBUTE_DIR / "native_structure.json"

# Same pristine-checkout convention as attribute/extract_native_structure.py
# (kept as a literal scratch path rather than relative, since the pristine
# overlay lives outside this repo entirely -- see that module's docstring
# for why a *pristine*, not live, robocasa checkout is used for this kind
# of extraction).
PRISTINE_ROBOCASA_ROOT = Path(
    "/tmp/claude-3337345/-nethome-chuang475/35365880-0b6c-4259-bc31-4a8d4924ae65"
    "/scratchpad/pristine_robocasa"
)
# docs/ is untouched by SafeManip either way (confirmed: only kitchen.py,
# predicates.py, attributes.py were ever modified/added) -- read from
# whichever checkout exists, preferring the pristine one for consistency.
DOCS_ROOT = PRISTINE_ROBOCASA_ROOT / "docs"
if not DOCS_ROOT.exists():
    DOCS_ROOT = Path("/coc/testnvme/chuang475/projects/robocasa/docs")

ATOMIC_TASK_INDEX_JS = DOCS_ROOT / "atomic_tasks" / "atomic_task_index.js"
COMPOSITE_TASK_ATTRIBUTES_JSON = DOCS_ROOT / "composite_tasks" / "task_attributes.json"
DATASET_REGISTRY_PY = PRISTINE_ROBOCASA_ROOT / "robocasa" / "utils" / "dataset_registry.py"


def load_target_tasks() -> Dict[str, List[str]]:
    """robocasa's official 50-task target/benchmark split -- `TARGET_TASKS`
    in dataset_registry.py: atomic_seen (18) + composite_seen (16) +
    composite_unseen (16) = 50. AST-parsed rather than imported, same
    reasoning as kitchen_objects.py elsewhere in this repo -- avoids
    executing an unrelated module just to read one literal dict."""
    tree = ast.parse(DATASET_REGISTRY_PY.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "TARGET_TASKS"
        ):
            # TARGET_TASKS = dict(atomic_seen=[...], ...) is a Call, not a
            # dict-literal -- ast.literal_eval can't handle `dict(...)`
            # syntax, same as OBJ_CATEGORIES in attribute/extract_native_
            # structure.py; evaluate it with only the `dict` builtin exposed.
            return eval(compile(ast.Expression(node.value), str(DATASET_REGISTRY_PY), "eval"), {"dict": dict})
    raise ValueError(f"TARGET_TASKS not found in {DATASET_REGISTRY_PY}")


def _load_native_structure() -> Dict[str, Any]:
    return json.loads(NATIVE_STRUCTURE_JSON_PATH.read_text())


# ---------------------------------------------------------------------------
# docs-site JS/JSON classification data, ported to Python
# ---------------------------------------------------------------------------
def _load_js_object_literal(path: Path) -> Any:
    """These files are `window.SOME_NAME = {...json-ish...};` -- strip the
    JS assignment wrapper and parse the rest as JSON (they're all valid
    JSON object/array literals, just embedded in a JS statement)."""
    text = path.read_text()
    _, _, rest = text.partition("=")
    return json.loads(rest.strip().rstrip(";").strip())


def load_atomic_fixture_index() -> Dict[str, str]:
    """task name -> fixture label, straight from atomic_task_index.js
    (grouped by which kitchen_<fixture>.py source file the task lives in --
    this is the ground truth the docs page itself uses, not re-derived)."""
    data = _load_js_object_literal(ATOMIC_TASK_INDEX_JS)
    result = {}
    for fx in data["fixtures"]:
        for t in fx["tasks"]:
            result[t["name"]] = fx["label"]
    return result


# Ported verbatim from docs/atomic_tasks/atomic_tasks.js's SKILL_GROUPS +
# getSkillIdForTaskName() -- see that file for the original; kept in the
# same order/logic so a diff against the JS stays meaningful.
SKILL_GROUPS = [
    ("closing_doors", "Close Door"),
    ("opening_doors", "Open Door"),
    ("lids", "Close & Open Lid"),
    ("insertion", "Insertion"),
    ("navigation", "Navigation"),
    ("pick_and_place", "Pick & Place"),
    ("pressing_buttons", "Press Button"),
    ("sliding_racks", "Slide Rack"),
    ("turning_levers", "Turn Lever"),
    ("twisting_knobs", "Twist Knob"),
]
SKILL_LABEL_BY_ID = dict(SKILL_GROUPS)


def get_skill_id_for_task_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "pick_and_place"
    if name in ("CoffeeSetupMug", "CoffeeServeMug"):
        return "insertion"
    if name == "OpenElectricKettleLid":
        return "pressing_buttons"
    if name in ("AdjustWaterTemperature", "TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout"):
        return "turning_levers"
    if name in ("OpenStandMixerHead", "CloseStandMixerHead"):
        return "lids"
    if re.match(r"^(Open|Close).*(Lid|Head)$", name):
        return "lids"
    if re.match(r"^Open[A-Z]", name):
        return "opening_doors"
    if re.match(r"^Close[A-Z]", name):
        return "closing_doors"
    if name == "NavigateKitchen":
        return "navigation"
    if re.match(r"^Slide", name):
        return "sliding_racks"
    if name in ("StartCoffeeMachine", "TurnOnMicrowave", "TurnOffMicrowave", "TurnOnBlender", "TurnOnElectricKettle"):
        return "pressing_buttons"
    if name == "TurnOnToaster":
        return "turning_levers"
    if name == "PreheatOven" or name in ("TurnOnStove", "TurnOffStove"):
        return "twisting_knobs"
    if re.match(r"^Adjust", name):
        return "twisting_knobs"
    if name in ("TurnSinkSpout", "LowerHeat", "TurnOnToasterOven"):
        return "twisting_knobs"
    return "pick_and_place"


# Ported verbatim from docs/composite_tasks/composite_tasks_dropdown.js's
# META_BY_ACTIVITY + metaCategoryForActivityTitle() fallback heuristic.
META_BY_ACTIVITY = {
    "washing produce": "food prep", "defrosting food": "food prep",
    "preparing sandwiches": "food prep", "mixing ingredients": "food prep",
    "seasoning food": "food prep", "making salads": "food prep",
    "preparing marinades": "food prep", "measuring ingredients": "food prep",
    "chopping vegetables": "food prep", "slicing meat": "food prep",
    "boiling water": "cooking", "sauteing vegetables": "cooking",
    "frying foods": "cooking", "steaming vegetables": "cooking",
    "microwaving foods": "cooking", "toasting bread": "cooking",
    "slow cooking": "cooking", "baking": "cooking", "broiling fish": "cooking",
    "simmering sauces": "cooking",
    "brewing coffee": "beverage preparation", "making tea": "beverage preparation",
    "making smoothies": "beverage preparation", "making juice": "beverage preparation",
    "preparing hot chocolate": "beverage preparation", "mixing drinks": "beverage preparation",
    "adding ice to beverages": "beverage preparation",
    "arranging cabinets": "organizing and storage", "stocking supplies": "organizing and storage",
    "organizing dishes and containers": "organizing and storage", "sorting ingredients": "organizing and storage",
    "organizing utensils": "organizing and storage", "loading refrigerator": "organizing and storage",
    "storing leftovers": "organizing and storage", "managing freezer space": "organizing and storage",
    "setting the table": "serving", "plating food": "serving", "portioning meals": "serving",
    "filling serving dishes": "serving", "serving beverages": "serving", "arranging buffet": "serving",
    "packing lunches": "serving", "arranging condiments": "serving", "garnishing dishes": "serving",
    "washing dishes": "cleaning and sanitizing", "loading dishwasher": "cleaning and sanitizing",
    "sanitizing surfaces": "cleaning and sanitizing", "cleaning appliances": "cleaning and sanitizing",
    "organizing recycling": "cleaning and sanitizing", "cleaning sink": "cleaning and sanitizing",
    "sanitizing cutting boards": "cleaning and sanitizing",
}
CATEGORY_LABELS = {
    "food prep": "Food Preparation",
    "cooking": "Cooking",
    "beverage preparation": "Beverage Preparation",
    "organizing and storage": "Organizing and Storage",
    "serving": "Serving",
    "cleaning and sanitizing": "Cleaning and Sanitizing",
}
_ALIAS_REPLACEMENTS = [
    ("washing fruits and vegetables", "washing produce"),
    ("preparing marinade", "preparing marinades"),
    ("loading fridge", "loading refrigerator"),
    ("restocking supplies", "stocking supplies"),
    ("sanitizing cutting board", "sanitizing cutting boards"),
    ("brewing", "brewing coffee"),
    ("frying", "frying foods"),
    ("boiling", "boiling water"),
]


def _normalize_activity_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def meta_category_for_activity(title: str) -> str:
    n = _normalize_activity_name(title)
    if n in META_BY_ACTIVITY:
        return META_BY_ACTIVITY[n]
    for old, new in _ALIAS_REPLACEMENTS:
        cand = n.replace(old, new)
        if cand in META_BY_ACTIVITY:
            return META_BY_ACTIVITY[cand]
    if re.search(r"\b(beverage|drink|tea|coffee|juice|smoothie|hot chocolate|ice)\b", n):
        return "beverage preparation"
    if re.search(r"\b(wash|clean|sanitize|sanitiz|recycl|dishwasher|sink|tidy)\b", n):
        return "cleaning and sanitizing"
    if re.search(r"\b(fridge|freezer|cabinet|drawer|organizing|arranging|sorting|stock|store|storing|pack)\b", n):
        return "organizing and storage"
    if re.search(r"\b(serv|plating|portion|setting the table|buffet)\b", n):
        return "serving"
    if re.search(r"\b(boil|saute|fry|steam|microwave|toast|bake|broil|simmer|slow cook|reheat)\b", n):
        return "cooking"
    return "food prep"


def load_task_attributes() -> Dict[str, Dict[str, Any]]:
    """name -> {activity, num_subtasks, moma_required, description}, for
    all 365 tasks (both atomic and composite -- atomic ones have
    activity=="Atomic" literally)."""
    data = json.loads(COMPOSITE_TASK_ATTRIBUTES_JSON.read_text())
    return {t["name"]: t for t in data["tasks"]}


# ---------------------------------------------------------------------------
# Task class introspection (real import + inheritance-aware inspect.getsource,
# NOT per-class AST -- see module docstring for why)
# ---------------------------------------------------------------------------
def load_task_registry():
    sys.path[:] = [p for p in sys.path if p not in ("", str(Path.cwd()))]
    sys.path.insert(0, str(PRISTINE_ROBOCASA_ROOT))
    import robocasa  # noqa: F401
    assert str(robocasa.__file__).startswith(str(PRISTINE_ROBOCASA_ROOT)), (
        "robocasa did not resolve to the pristine checkout"
    )
    import robocasa.environments as E

    return E.REGISTERED_KITCHEN_ENVS


def _literal_or_source(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        try:
            return ast.unparse(node)
        except Exception:
            return None


def _find_calls(tree: ast.AST, func_name: str) -> List[ast.Call]:
    """Every ast.Call node anywhere in `tree` whose callee is named
    `func_name`, whether that's a bare name (`dict(...)`) or an attribute
    access ending in that name (`self.register_fixture_ref(...)`)."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == func_name:
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == func_name:
            calls.append(node)
    return calls


def resolve_constructor_defaults(cls) -> Dict[str, str]:
    """Maps `self.<attr>` -> a `FixtureType.<MEMBER>`-style string.

    Atomic tasks in particular chain __init__ across several levels, e.g.
    `OpenCabinet(fixture_id=FixtureType.CABINET_WITH_DOOR, ...)` ->
    `super().__init__(fixture_id=fixture_id, ...)` (OpenDoor, no default of
    its own) -> `ManipulateDoor.__init__(self, fixture_id, ...)` (positional,
    no default) -> `self.fixture_id = fixture_id`. The assignment and the
    meaningful default live in *different* classes along the MRO, so this
    walks every class's own `__init__` (not just `cls.__init__`), collecting
    self.attr=param assignments from any of them, and -- for each such
    parameter name -- takes the first (most-derived, since __mro__ is
    leaf-first) default actually declared for that name anywhere in the
    chain."""
    param_to_attr: Dict[str, str] = {}
    param_default_by_name: Dict[str, Any] = {}
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            source = inspect.getsource(init)
            sig = inspect.signature(init)
        except (OSError, TypeError, ValueError):
            continue
        tree = ast.parse(_dedent(source))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"
                and isinstance(node.value, ast.Name)
            ):
                param_to_attr.setdefault(node.value.id, node.targets[0].attr)
        for name, p in sig.parameters.items():
            if p.default is not inspect.Parameter.empty and name not in param_default_by_name:
                param_default_by_name[name] = p.default  # first hit wins (leaf-first MRO order)

    resolved: Dict[str, str] = {}
    for param_name, attr_name in param_to_attr.items():
        default = param_default_by_name.get(param_name)
        member_name = getattr(default, "name", None)  # IntEnum member -> its .name
        if isinstance(member_name, str):
            resolved[f"self.{attr_name}"] = f"FixtureType.{member_name}"
    return resolved


def extract_fixture_refs(cls) -> List[Dict[str, Any]]:
    """Every `self.register_fixture_ref("name", dict(id=..., ...))` call
    reachable from whichever class in the MRO actually defines
    `_setup_kitchen_references` (inherited methods included, via
    getattr+getsource -- not just the leaf task class's own body)."""
    method = getattr(cls, "_setup_kitchen_references", None)
    if method is None:
        return []
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return []
    tree = ast.parse(_dedent(source))
    constructor_defaults = resolve_constructor_defaults(cls)
    refs = []
    for call in _find_calls(tree, "register_fixture_ref"):
        if not call.args:
            continue
        ref_name = _literal_or_source(call.args[0])
        ref_name_prefix = _static_string_prefix(call.args[0])
        id_value = None
        if len(call.args) > 1 and isinstance(call.args[1], ast.Call):
            id_kw = next((kw for kw in call.args[1].keywords if kw.arg == "id"), None)
            if id_kw is not None:
                id_value = _literal_or_source(id_kw.value)
        if isinstance(id_value, str) and id_value in constructor_defaults:
            id_value = constructor_defaults[id_value]
        refs.append({
            "ref_name": ref_name, "ref_name_static_prefix": ref_name_prefix,
            "fixture_type_id": id_value, "via": "register_fixture_ref",
        })

    # Second, distinct pattern some tasks use instead:
    # `self.<attr> = self.get_fixture(FixtureType.X, ...)` -- no named
    # registration, just a direct lookup assigned straight to an attribute.
    # Confirmed real (not a hypothetical): e.g. CoffeeSetupMug's
    # `self.coffee_machine = self.get_fixture(FixtureType.COFFEE_MACHINE)`,
    # which register_fixture_ref-only extraction missed entirely.
    seen_names = {r["ref_name"] for r in refs}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
            and isinstance(node.value, ast.Call)
        ):
            continue
        call = node.value
        func = call.func
        is_get_fixture = isinstance(func, ast.Attribute) and func.attr == "get_fixture"
        if not is_get_fixture or not call.args:
            continue
        ref_name = node.targets[0].attr
        if ref_name in seen_names:
            continue
        id_value = _literal_or_source(call.args[0])
        if isinstance(id_value, str) and id_value in constructor_defaults:
            id_value = constructor_defaults[id_value]
        refs.append({"ref_name": ref_name, "fixture_type_id": id_value, "via": "get_fixture"})
        seen_names.add(ref_name)
    return refs


def extract_object_cfgs(cls) -> List[Dict[str, Any]]:
    """Every `dict(name=..., obj_groups=..., <boolean flags>...)` call
    reachable from whichever class defines `_get_obj_cfgs`, PLUS a
    synthesized companion entry for every `placement=dict(...,
    try_to_place_in=...)` -- confirmed in kitchen.py's `_create_objects()`
    that this isn't just a placement hint: robocasa actually spawns a
    *second*, separate object of that category, named `f"{name}_container"`
    (e.g. PanTransfer's "vegetable" cfg with `try_to_place_in="pan"` spawns
    a real pan object named "vegetable_container", which `_check_success`
    then references directly -- and which a naive reading of
    `_get_obj_cfgs()` alone would never reveal, since it's never written as
    its own `dict(name=..., obj_groups=...)` entry anywhere in the source).

    Note the spawn is conditional at runtime (only fires if the object's
    *sampled* category happens to belong to robocasa's `in_container`
    OBJ_GROUPS) -- see `in_container_coverage` in the synthesized entry,
    cross-referenced against native_structure.json, for whether that's
    guaranteed/partial/never for this cfg's `obj_groups`."""
    method = getattr(cls, "_get_obj_cfgs", None)
    if method is None:
        return []
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return []
    tree = ast.parse(_dedent(source))
    boolean_flags = (
        "graspable", "washable", "microwavable", "cookable",
        "fridgable", "freezable", "dishwashable",
    )
    cfgs = []
    for call in _find_calls(tree, "dict"):
        kwargs = {kw.arg: kw for kw in call.keywords if kw.arg is not None}
        if "name" not in kwargs or "obj_groups" not in kwargs:
            continue  # not an object-cfg dict (e.g. the inner placement=dict(...))
        name_node = kwargs["name"].value
        entry: Dict[str, Any] = {
            "name": _literal_or_source(name_node),
            "name_static_prefix": _static_string_prefix(name_node),
            "obj_groups": _literal_or_source(kwargs["obj_groups"].value),
        }
        for flag in boolean_flags:
            if flag in kwargs:
                entry[flag] = _literal_or_source(kwargs[flag].value)
        cfgs.append(entry)

        placement_kw = kwargs.get("placement")
        if placement_kw is not None and isinstance(placement_kw.value, ast.Call):
            placement_kwargs = {kw.arg: kw for kw in placement_kw.value.keywords if kw.arg is not None}
            try_to_place_in_kw = placement_kwargs.get("try_to_place_in")
            if try_to_place_in_kw is not None:
                name_prefix = entry["name_static_prefix"] or (
                    entry["name"] if isinstance(entry["name"], str) else None
                )
                container_entry: Dict[str, Any] = {
                    "name": f"{entry['name']}_container" if isinstance(entry["name"], str) else None,
                    "name_static_prefix": f"{name_prefix}_container" if name_prefix else None,
                    "obj_groups": _literal_or_source(try_to_place_in_kw.value),
                    "synthesized_from": "try_to_place_in",
                    "parent_object": entry["name"],
                }
                cfgs.append(container_entry)
    return cfgs


def _static_string_prefix(node: ast.AST) -> Optional[str]:
    """The literal, non-dynamic prefix of a string-valued expression, for
    matching loop-generated per-instance names across two call sites that
    build the *same* runtime name two different, syntactically unrelated
    ways -- confirmed real, not hypothetical: AddIceCubes's `_get_obj_cfgs`
    names objects `"ice_cube" + str(i)` (an ast.BinOp) while its
    `_check_success` references them as `f"ice_cube{i}"` (an ast.JoinedStr).
    Both reduce to the same prefix, "ice_cube", here.

    Handles: a plain string constant (returned whole); an f-string /
    JoinedStr (concatenates its constant segments, i.e. everything before
    the first `{...}` placeholder); a `"literal" + <dynamic>` BinOp (returns
    the literal side). Anything else -> None (genuinely can't tell)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        prefix_parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                prefix_parts.append(value.value)
            else:
                break  # stop at the first {placeholder} -- rest is dynamic
        return "".join(prefix_parts) if prefix_parts else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _static_string_prefix(node.left), _static_string_prefix(node.right)
        if left is not None and not isinstance(node.right, (ast.Constant, ast.JoinedStr)):
            return left  # "literal" + <dynamic expr> -- literal side is the usable prefix
        if left is not None and right is not None:
            return left + right
    return None


def _prefix_matches(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (a == b or a.startswith(b) or b.startswith(a))


def extract_success_entities(cls, known_names: Set[str], known_dynamic_prefixes: Dict[str, str]) -> List[str]:
    """Which of this task's own object-cfg/fixture-ref names (`known_names`,
    plus `known_dynamic_prefixes` for loop-generated ones with no single
    literal name) `_check_success` actually references -- via exact string
    literals, `self.<attr>` names, or a shared static prefix for
    dynamically-generated per-instance names (see _static_string_prefix)."""
    method = getattr(cls, "_check_success", None)
    if method is None:
        return []
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return []
    tree = ast.parse(_dedent(source))
    referenced_exact: Set[str] = set()
    referenced_prefixes: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            referenced_exact.add(node.attr)
        elif isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            prefix = _static_string_prefix(node)
            if prefix is not None:
                referenced_exact.add(prefix)
                referenced_prefixes.add(prefix)

    matched = referenced_exact & known_names
    for cfg_name, cfg_prefix in known_dynamic_prefixes.items():
        if any(_prefix_matches(cfg_prefix, ref) for ref in referenced_prefixes):
            matched.add(cfg_name)
    return sorted(matched)


def _dedent(source: str) -> str:
    import textwrap

    return textwrap.dedent(source)


# ---------------------------------------------------------------------------
# Cross-reference against monitor/sim/robocasa/attribute's native taxonomy
# ---------------------------------------------------------------------------
def resolve_obj_groups(value: Any, native: Dict[str, Any]) -> Dict[str, Any]:
    """Classify an obj_groups value against native_structure.json and
    resolve it to the actual category names it draws from, wherever
    that's staticaly knowable (skips anything that isn't a literal
    str/list/tuple, e.g. an f-string or self.attribute expression)."""
    if isinstance(value, (list, tuple)):
        return {"kind": "multiple", "groups": [resolve_obj_groups(v, native) for v in value]}
    if not isinstance(value, str):
        return {"kind": "dynamic_expression", "raw": value}

    categories = native["objects"]["categories"]
    if value == "all":
        return {"kind": "all_categories", "count": len(categories)}
    if value in categories:
        return {"kind": "single_category_self_group", "category": value}
    if value in native["objects"]["type_tag_vocabulary"]:
        members = sorted(c for c, info in categories.items() if value in info["types"])
        return {"kind": "native_type_tag", "tag": value, "categories": members}
    if value in native["objects"]["hand_curated_obj_groups"]:
        return {"kind": "hand_curated_obj_group", "categories": native["objects"]["hand_curated_obj_groups"][value]}
    if value in native["objects"]["computed_obj_groups"]:
        info = native["objects"]["computed_obj_groups"][value]
        return {"kind": "computed_obj_group", "types_argument": info["types_argument"], "categories": info["categories"]}
    return {"kind": "unresolved", "raw": value}


def _resolved_categories(resolved: Dict[str, Any], native: Dict[str, Any]) -> Optional[Set[str]]:
    """The concrete set of category names a resolve_obj_groups() result
    could sample from, wherever that's statically enumerable."""
    kind = resolved.get("kind")
    if kind == "single_category_self_group":
        return {resolved["category"]}
    if kind in ("native_type_tag", "hand_curated_obj_group", "computed_obj_group"):
        return set(resolved["categories"])
    if kind == "all_categories":
        return set(native["objects"]["categories"].keys())
    if kind == "multiple":
        sets = [_resolved_categories(g, native) for g in resolved["groups"]]
        if any(s is None for s in sets):
            return None
        return set().union(*sets) if sets else set()
    return None


def resolve_fixture_type_id(value: Any, native: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, str):
        return {"kind": "dynamic_expression", "raw": value}
    m = re.match(r"^FixtureType\.([A-Z_]+)$", value)
    if not m:
        return {"kind": "not_a_fixture_type_literal", "raw": value}
    member = m.group(1)
    known = member in native["fixtures"]["fixture_type_enum"]
    return {"kind": "fixture_type_enum_member", "member": member, "in_native_enum": known}


# ---------------------------------------------------------------------------
# Per-task summary
# ---------------------------------------------------------------------------
def summarize_task(name: str, cls, attrs: Dict[str, Any], atomic_fixture_index: Dict[str, str], native: Dict[str, Any]) -> Dict[str, Any]:
    is_atomic = attrs.get("activity") == "Atomic"

    fixture_refs = extract_fixture_refs(cls)
    obj_cfgs = extract_object_cfgs(cls)
    known_names = {r["ref_name"] for r in fixture_refs if isinstance(r["ref_name"], str)}
    known_names |= {c["name"] for c in obj_cfgs if isinstance(c["name"], str)}
    known_dynamic_prefixes: Dict[str, str] = {}
    for c in obj_cfgs:
        if isinstance(c["name"], str) and c.get("name_static_prefix"):
            known_dynamic_prefixes[c["name"]] = c["name_static_prefix"]
    for r in fixture_refs:
        if isinstance(r["ref_name"], str) and r.get("ref_name_static_prefix"):
            known_dynamic_prefixes[r["ref_name"]] = r["ref_name_static_prefix"]
    success_entities = extract_success_entities(cls, known_names, known_dynamic_prefixes)

    for ref in fixture_refs:
        ref["resolved"] = resolve_fixture_type_id(ref["fixture_type_id"], native)
    for cfg in obj_cfgs:
        cfg["resolved"] = resolve_obj_groups(cfg["obj_groups"], native)

    # For every synthesized try_to_place_in container: is the runtime spawn
    # actually guaranteed, or only conditional? robocasa only spawns it if
    # the PARENT object's *sampled* category happens to be a member of the
    # native `in_container` OBJ_GROUPS -- so this depends on how much the
    # parent's own resolved category set overlaps that group.
    cfgs_by_name = {c["name"]: c for c in obj_cfgs if isinstance(c.get("name"), str)}
    in_container_categories = set(native["objects"]["computed_obj_groups"]["in_container"]["categories"])
    for cfg in obj_cfgs:
        if cfg.get("synthesized_from") != "try_to_place_in":
            continue
        parent = cfgs_by_name.get(cfg.get("parent_object"))
        parent_categories = _resolved_categories(parent["resolved"], native) if parent else None
        if parent_categories is None:
            cfg["in_container_coverage"] = "unknown (parent obj_groups not statically resolvable)"
        elif parent_categories <= in_container_categories:
            cfg["in_container_coverage"] = "guaranteed (every possible sampled category qualifies)"
        elif parent_categories & in_container_categories:
            cfg["in_container_coverage"] = "conditional (only some sampled categories qualify)"
        else:
            cfg["in_container_coverage"] = "never (no sampled category qualifies -- try_to_place_in is dead here)"

    result: Dict[str, Any] = {
        "name": name,
        "task_kind": "atomic" if is_atomic else "composite",
        "description": attrs.get("description"),
        "num_subtasks": attrs.get("num_subtasks"),
        "moma_required": attrs.get("moma_required"),
        "source_file": None,
        "scene": {
            "exclude_layouts": list(getattr(cls, "EXCLUDE_LAYOUTS", []) or []),
            "exclude_styles": list(getattr(cls, "EXCLUDE_STYLES", []) or []),
        },
        "fixtures": fixture_refs,
        "objects": obj_cfgs,
        "success_related_entities": success_entities,
    }
    try:
        result["source_file"] = str(Path(inspect.getfile(cls)).relative_to(PRISTINE_ROBOCASA_ROOT))
    except Exception:
        pass

    if is_atomic:
        result["skill"] = {"id": get_skill_id_for_task_name(name), "label": SKILL_LABEL_BY_ID.get(get_skill_id_for_task_name(name))}
        result["fixture_label"] = atomic_fixture_index.get(name)
    else:
        activity = attrs.get("activity")
        category_id = meta_category_for_activity(activity) if activity else None
        result["activity"] = activity
        result["category"] = CATEGORY_LABELS.get(category_id) if category_id else None

    return result


def build_report() -> Dict[str, Any]:
    native = _load_native_structure()
    registry = load_task_registry()
    task_attrs = load_task_attributes()
    atomic_fixture_index = load_atomic_fixture_index()

    tasks = {}
    errors = {}
    for name, attrs in sorted(task_attrs.items()):
        cls = registry.get(name)
        if cls is None:
            errors[name] = "not found in REGISTERED_KITCHEN_ENVS"
            continue
        try:
            tasks[name] = summarize_task(name, cls, attrs, atomic_fixture_index, native)
        except Exception as e:  # defensive: one weird task shouldn't kill the whole run
            errors[name] = f"{type(e).__name__}: {e}"

    return {
        "num_tasks_in_docs_json": len(task_attrs),
        "num_tasks_extracted": len(tasks),
        "num_errors": len(errors),
        "errors": errors,
        "tasks": tasks,
    }


def render_report_txt(report: Dict[str, Any]) -> str:
    lines = [
        f"tasks in docs json: {report['num_tasks_in_docs_json']}, "
        f"extracted: {report['num_tasks_extracted']}, errors: {report['num_errors']}",
        "",
    ]
    if report["errors"]:
        lines.append("-- errors --")
        for name, err in sorted(report["errors"].items()):
            lines.append(f"  {name}: {err}")
        lines.append("")

    for name, t in sorted(report["tasks"].items()):
        lines.append("=" * 78)
        lines.append(f"{name}  [{t['task_kind']}]")
        lines.append("=" * 78)
        lines.append(f"description: {t['description']}")
        lines.append(f"num_subtasks: {t['num_subtasks']}  moma_required: {t['moma_required']}  source: {t['source_file']}")
        if t["task_kind"] == "atomic":
            lines.append(f"skill: {t['skill']['id']} ({t['skill']['label']})   fixture_label: {t['fixture_label']}")
        else:
            lines.append(f"activity: {t['activity']}   category: {t['category']}")
        exl, exs = t["scene"]["exclude_layouts"], t["scene"]["exclude_styles"]
        if exl or exs:
            lines.append(f"scene: exclude_layouts={exl}  exclude_styles={exs}")
        else:
            lines.append("scene: no layout/style exclusions")

        lines.append(f"fixtures ({len(t['fixtures'])}):")
        for f in t["fixtures"]:
            r = f["resolved"]
            if r["kind"] == "fixture_type_enum_member":
                detail = f"FixtureType.{r['member']}" + ("" if r["in_native_enum"] else " [NOT in native enum!]")
            else:
                detail = f"{r['kind']}: {r.get('raw')}"
            lines.append(f"  - {f['ref_name']}: {detail}")

        lines.append(f"objects ({len(t['objects'])}):")
        for c in t["objects"]:
            r = c["resolved"]
            skip_keys = ("name", "name_static_prefix", "obj_groups", "resolved", "synthesized_from", "parent_object", "in_container_coverage")
            flags = {k: v for k, v in c.items() if k not in skip_keys}
            flag_str = f" flags={flags}" if flags else ""
            if r["kind"] in ("single_category_self_group",):
                detail = f"category `{r['category']}`"
            elif r["kind"] == "native_type_tag":
                detail = f"native tag `{r['tag']}` ({len(r['categories'])} categories)"
            elif r["kind"] in ("hand_curated_obj_group", "computed_obj_group"):
                detail = f"{r['kind']} ({len(r['categories'])} categories)"
            elif r["kind"] == "all_categories":
                detail = f"all {r['count']} categories"
            else:
                detail = f"{r['kind']}: {r.get('raw')}"
            prefix = "  - "
            suffix = ""
            if c.get("synthesized_from") == "try_to_place_in":
                prefix = "  - [SYNTHESIZED, try_to_place_in on parent `" + str(c.get('parent_object')) + "`] "
                suffix = f"  [in_container_coverage: {c.get('in_container_coverage')}]"
            lines.append(f"{prefix}{c['name']}: obj_groups={c['obj_groups']!r} -> {detail}{flag_str}{suffix}")

        lines.append(f"success_related_entities: {t['success_related_entities']}")
        lines.append("")

    return "\n".join(lines)


def build_target_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Filters the full report down to robocasa's official 50-task target
    split (TARGET_TASKS in dataset_registry.py), tagging each with which of
    the 3 groups (atomic_seen/composite_seen/composite_unseen) it's in."""
    target_tasks = load_target_tasks()
    split_by_name: Dict[str, str] = {
        name: split for split, names in target_tasks.items() for name in names
    }
    tasks: Dict[str, Any] = {}
    missing: List[str] = []
    for name, split in sorted(split_by_name.items()):
        if name not in report["tasks"]:
            missing.append(name)
            continue
        entry = dict(report["tasks"][name])
        entry["target_split"] = split
        tasks[name] = entry
    return {
        "num_target_tasks": len(split_by_name),
        "num_found": len(tasks),
        "num_missing": len(missing),
        "missing": missing,
        "splits": {split: names for split, names in target_tasks.items()},
        "tasks": tasks,
    }


def render_target_report_txt(target_report: Dict[str, Any], full_report: Dict[str, Any]) -> str:
    lines = [
        f"target tasks: {target_report['num_target_tasks']}, found: {target_report['num_found']}, "
        f"missing: {target_report['num_missing']}",
    ]
    if target_report["missing"]:
        lines.append(f"  missing: {target_report['missing']}")
    lines.append("")
    for split, names in target_report["splits"].items():
        lines.append(f"-- {split} ({len(names)}) --")
        lines.append(f"  {', '.join(names)}")
    lines.append("")

    # Reuse render_report_txt's per-task rendering by handing it a
    # full-shaped report restricted to just the target tasks.
    restricted = {**full_report, "tasks": target_report["tasks"], "errors": {}}
    body = render_report_txt(restricted)
    # Strip render_report_txt's own header line (task/error counts for the
    # *restricted* dict, which would misleadingly repeat num_tasks_in_docs_json).
    body = "\n".join(body.split("\n")[2:])
    lines.append(body)
    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    (THIS_DIR / "tasks.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    text = render_report_txt(report)
    (THIS_DIR / "tasks_summary.txt").write_text(text)
    print(f"wrote {THIS_DIR / 'tasks.json'}")
    print(f"wrote {THIS_DIR / 'tasks_summary.txt'}")
    print(f"extracted {report['num_tasks_extracted']}/{report['num_tasks_in_docs_json']} tasks, {report['num_errors']} errors")
    if report["errors"]:
        for name, err in sorted(report["errors"].items())[:10]:
            print(f"  ERROR {name}: {err}")

    target_report = build_target_report(report)
    (THIS_DIR / "tasks_target.json").write_text(json.dumps(target_report, indent=2, sort_keys=True, default=str))
    target_text = render_target_report_txt(target_report, report)
    (THIS_DIR / "tasks_target_summary.txt").write_text(target_text)
    print(f"wrote {THIS_DIR / 'tasks_target.json'}")
    print(f"wrote {THIS_DIR / 'tasks_target_summary.txt'}")
    print(f"target split: {target_report['num_found']}/{target_report['num_target_tasks']} found, missing: {target_report['missing']}")
