from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Set


EXTERNAL_ROBOCASA_ROOT = Path(__file__).resolve().parents[4] / "robocasa"
EXTERNAL_OBJECT_METADATA_PATH = (
    EXTERNAL_ROBOCASA_ROOT / "robocasa" / "models" / "objects" / "kitchen_objects.py"
)


def _load_ast(path: Path):
    if not path.exists():
        return None
    return ast.parse(path.read_text())


@lru_cache(maxsize=1)
def external_object_categories() -> Dict[str, Dict[str, object]]:
    module = _load_ast(EXTERNAL_OBJECT_METADATA_PATH)
    if module is None:
        return {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "OBJ_CATEGORIES":
                expr = ast.Expression(node.value)
                raw = eval(
                    compile(expr, str(EXTERNAL_OBJECT_METADATA_PATH), "eval"),
                    {"dict": dict},
                )
                normalized: Dict[str, Dict[str, object]] = {}
                for name, value in raw.items():
                    item = dict(value)
                    types = item.get("types", ())
                    if isinstance(types, str):
                        types = (types,)
                    item["types"] = tuple(types)
                    normalized[str(name)] = item
                return normalized
    return {}


ROLE_ATTRIBUTE_AXES = {
    "object": [
        "fragile",
        "hot",
        "cold",
        "liquid",
        "food",
        "ready_to_eat",
        "raw",
        "drink",
        "alcohol",
        "fruit",
        "vegetable",
        "meat",
        "dairy",
        "bread_food",
        "toastable",
        "cooked_food",
        "packaged_food",
        "condiment",
        "spice",
        "pastry",
        "sweet",
        "cleaner",
        "receptacle",
        "cookware",
        "stackable",
        "utensil",
        "tool",
        "lid",
        "openable",
        "closeable",
        "pourable",
        "washable",
        "cuttable",
        "cookable",
        "freezable",
        "graspable",
        "microwavable",
        "fridgable",
        "dishwashable",
        "pressable",
        "turnable",
        "slideable",
        "twistable",
    ],
    "support": [
        "containment",
        "heated",
        "cold_storage",
        "wash_zone",
        "prep_zone",
        "serving_zone",
        "storage_zone",
    ],
    "fixture": [
        "openable",
        "closeable",
        "heated",
        "powered",
        "dispensing",
        "dirty",
        "cooling",
        "washing",
        "mixing",
        "toasting",
        "storage",
        "cooktop",
        "sink",
        "microwave",
        "oven",
        "stove",
        "fridge",
        "cabinet",
        "drawer",
        "pressable",
        "turnable",
        "slideable",
        "twistable",
    ],
    "button": [
        "lever",
        "toggle",
        "power",
        "mode_select",
        "pressable",
    ],
    "tool": [
        "cleaning_tool",
        "cutting_tool",
        "mixing_tool",
        "serving_tool",
        "measuring_tool",
    ],
}


OBJECT_ATTRIBUTE_AXES: List[str] = list(ROLE_ATTRIBUTE_AXES["object"])
SUPPORT_ATTRIBUTE_AXES: List[str] = list(ROLE_ATTRIBUTE_AXES["support"])
FIXTURE_ATTRIBUTE_AXES: List[str] = list(ROLE_ATTRIBUTE_AXES["fixture"])
BUTTON_ATTRIBUTE_AXES: List[str] = list(ROLE_ATTRIBUTE_AXES["button"])
TOOL_ATTRIBUTE_AXES: List[str] = list(ROLE_ATTRIBUTE_AXES["tool"])


FRAGILE_CATEGORIES: Set[str] = {
    "mug",
    "coffee_cup",
    "cup",
    "glass_cup",
    "wine_glass",
    "plate",
    "bowl",
    "egg",
}

LIQUID_CATEGORIES: Set[str] = {
    "brewed_coffee",
    "coffee",
    "sauce",
    "soup",
    "water",
}

READY_TO_EAT_TYPE_NAMES: Set[str] = {
    "fruit",
    "vegetable",
    "dairy",
    "bread_food",
    "cooked_food",
    "pastry",
    "sweets",
    "drink",
    "condiment",
}

FOOD_TYPE_NAMES: Set[str] = {
    "fruit",
    "vegetable",
    "meat",
    "dairy",
    "bread_food",
    "cooked_food",
    "pastry",
    "sweets",
    "condiment",
    "spice",
    "drink",
}

TOOL_CATEGORIES: Set[str] = {
    "knife",
    "fork",
    "spoon",
    "spatula",
    "ladle",
    "whisk",
    "tongs",
    "peeler",
    "reamer",
    "rolling_pin",
    "pizza_cutter",
    "measuring_cup",
    "can_opener",
    "bottle_opener",
    "cheese_grater",
    "scissors",
    "digital_scale",
    "strainer",
    "cutting_board",
}

RECEPTACLE_CATEGORIES: Set[str] = {
    "bowl",
    "coffee_cup",
    "cup",
    "glass_cup",
    "mug",
    "pot",
    "pan",
    "tray",
    "container",
    "tupperware",
    "bottle",
    "can",
    "jug",
    "jug_wide_opening",
    "pitcher",
}

POURABLE_CATEGORIES: Set[str] = {
    "jug",
    "jug_wide_opening",
    "pitcher",
    "teapot",
    "thermos",
    "mug",
    "coffee_cup",
    "cup",
    "glass_cup",
    "wine_glass",
    *LIQUID_CATEGORIES,
}

PLACEHOLDER_TRUE_ATTRIBUTES: Set[str] = {
    "raw",
    "ready_to_eat",
    "hot",
    "cold",
    "liquid",
}

OBJECT_METADATA_BOOLEAN_ATTRIBUTES: Set[str] = {
    "graspable",
    "washable",
    "microwavable",
    "cookable",
    "toastable",
    "fridgable",
    "freezable",
    "dishwashable",
}

TWISTABLE_CATEGORIES: Set[str] = {
    "bottle",
    "can",
    "jar",
    "thermos",
}

OPENABLE_OBJECT_CATEGORIES: Set[str] = {
    *TWISTABLE_CATEGORIES,
}


@lru_cache(maxsize=1)
def fixture_class_default_attributes() -> Dict[str, List[str]]:
    defaults = {
        "Accessory": [],
        "Blender": ["powered", "mixing", "pressable"],
        "BlenderLid": [],
        "Box": [],
        "CoffeeMachine": [
            "heated",
            "powered",
            "dispensing",
            "pressable",
            "support:heated",
        ],
        "Counter": ["support:prep_zone", "support:serving_zone"],
        "DishRack": ["support:containment", "support:storage_zone"],
        "Dishwasher": [
            "openable",
            "closeable",
            "slideable",
            "washing",
            "storage",
            "support:containment",
            "support:wash_zone",
            "support:storage_zone",
        ],
        "Drawer": [
            "openable",
            "closeable",
            "storage",
            "drawer",
            "support:containment",
            "support:storage_zone",
        ],
        "ElectricKettle": ["heated", "lever", "powered", "pressable"],
        "Floor": [],
        "FridgeBottomFreezer": [
            "openable",
            "closeable",
            "cooling",
            "storage",
            "fridge",
            "support:containment",
            "support:cold_storage",
            "support:storage_zone",
        ],
        "FridgeFrenchDoor": [
            "openable",
            "closeable",
            "cooling",
            "storage",
            "fridge",
            "support:containment",
            "support:cold_storage",
            "support:storage_zone",
        ],
        "FridgeSideBySide": [
            "openable",
            "closeable",
            "cooling",
            "storage",
            "fridge",
            "support:containment",
            "support:cold_storage",
            "support:storage_zone",
        ],
        "HingeCabinet": [
            "openable",
            "closeable",
            "storage",
            "cabinet",
            "support:containment",
            "support:storage_zone",
        ],
        "Hood": [],
        "HousingCabinet": [
            "openable",
            "closeable",
            "storage",
            "cabinet",
            "support:containment",
            "support:storage_zone",
        ],
        "Microwave": [
            "openable",
            "closeable",
            "heated",
            "powered",
            "microwave",
            "pressable",
            "support:heated",
        ],
        "OpenCabinet": [
            "openable",
            "closeable",
            "storage",
            "cabinet",
            "support:containment",
            "support:storage_zone",
        ],
        "Oven": [
            "openable",
            "closeable",
            "heated",
            "oven",
            "twistable",
            "support:heated",
        ],
        "PanelCabinet": [
            "openable",
            "closeable",
            "storage",
            "cabinet",
            "support:containment",
            "support:storage_zone",
        ],
        "Sink": [
            "washing",
            "sink",
            "turnable",
            "support:containment",
            "support:wash_zone",
        ],
        "SingleCabinet": [
            "openable",
            "closeable",
            "storage",
            "cabinet",
            "support:containment",
            "support:storage_zone",
        ],
        "StandMixer": ["powered", "mixing", "openable", "closeable"],
        "Stove": ["heated", "stove", "cooktop", "twistable", "support:heated"],
        "Stovetop": ["heated", "stove", "cooktop", "twistable", "support:heated"],
        "Toaster": ["heated", "powered", "toasting", "pressable", "slideable"],
        "ToasterOven": [
            "openable",
            "closeable",
            "twistable",
            "heated",
            "powered",
            "toasting",
            "oven",
            "support:heated",
        ],
        "Wall": [],
        "WallAccessory": [],
        "Window": ["openable"],
        "WindowProc": ["openable"],
    }
    return {key: sorted(value) for key, value in defaults.items()}


def object_category_attribute_defaults() -> Dict[str, List[str]]:
    rows: Dict[str, Set[str]] = {}
    for category, info in external_object_categories().items():
        types = set(info.get("types", ()) or ())
        attrs: Set[str] = set()
        if category in FRAGILE_CATEGORIES:
            attrs.add("fragile")
        if category in LIQUID_CATEGORIES or "liquid" in types:
            attrs.update({"liquid", "pourable"})
        elif "condiment" in types:
            # Bottled condiments, e.g. ketchup, are movable objects rather than
            # free liquid contents in the simulator.
            attrs.add("pourable")
        if types & FOOD_TYPE_NAMES:
            attrs.update({"food", "cookable"})
        if "drink" in types:
            attrs.update({"drink", "cookable"})
        if "alcohol" in types or category in {"wine", "beer", "liquor"}:
            attrs.add("alcohol")
        if "fruit" in types:
            attrs.add("fruit")
        if "vegetable" in types:
            attrs.add("vegetable")
        if "meat" in types:
            attrs.add("meat")
        if "dairy" in types:
            attrs.add("dairy")
        if "bread_food" in types:
            attrs.update({"bread_food", "toastable", "cookable"})
        if "cooked_food" in types:
            attrs.update({"cooked_food", "cookable"})
        if "condiment" in types:
            attrs.add("condiment")
        if "spice" in types:
            attrs.add("spice")
        if "pastry" in types:
            attrs.add("pastry")
        if "sweets" in types:
            attrs.add("sweet")
        if category in TOOL_CATEGORIES:
            attrs.update({"tool", "utensil", "graspable"})
        if "cookware" in types:
            attrs.add("cookware")
        if "receptacle" in types:
            attrs.add("receptacle")
        if category in POURABLE_CATEGORIES:
            attrs.add("pourable")
        for attr in OBJECT_METADATA_BOOLEAN_ATTRIBUTES:
            if info.get(attr):
                attrs.add(attr)
        if category in TWISTABLE_CATEGORIES:
            attrs.add("twistable")
        if category in OPENABLE_OBJECT_CATEGORIES:
            attrs.add("openable")
        if "openable" in attrs:
            attrs.add("closeable")
        if "raw" in types or ("meat" in types and "cooked_food" not in types):
            attrs.add("raw")
        if "ready_to_eat" in types or (
            types & READY_TO_EAT_TYPE_NAMES
            and "meat" not in types
            and "raw" not in types
        ):
            attrs.add("ready_to_eat")
        attrs.add("graspable")
        rows[category] = attrs
    return {key: sorted(value) for key, value in rows.items()}


OBJECT_CATEGORY_ATTRIBUTE_DEFAULTS = object_category_attribute_defaults()
FIXTURE_CLASS_ATTRIBUTE_DEFAULTS = fixture_class_default_attributes()


def object_is_receptacle_category(
    category: str, extra_attrs: Iterable[str] = ()
) -> bool:
    """True if *category* or any of *extra_attrs* identifies the object as a receptacle."""
    if str(category).lower() in RECEPTACLE_CATEGORIES:
        return True
    extra_attrs = {str(attr).lower() for attr in (extra_attrs or ())}
    return "receptacle" in extra_attrs or bool(extra_attrs & RECEPTACLE_CATEGORIES)


def infer_object_attributes(category: str, types: Iterable[str] = ()) -> Set[str]:
    category = str(category)
    attrs = set(OBJECT_CATEGORY_ATTRIBUTE_DEFAULTS.get(category, ()))
    types = set(types or ())
    if category in LIQUID_CATEGORIES or "liquid" in types:
        attrs.update({"liquid", "pourable"})
    if object_is_receptacle_category(category, types):
        attrs.add("receptacle")
    if "cookware" in types:
        attrs.add("cookware")
    if "raw" in types:
        attrs.add("raw")
    if "ready_to_eat" in types:
        attrs.add("ready_to_eat")
    if "in_container" in types:
        attrs.add("in_container")
    return attrs


def infer_fixture_attributes(class_name: str) -> Set[str]:
    return set(FIXTURE_CLASS_ATTRIBUTE_DEFAULTS.get(str(class_name), ()))
