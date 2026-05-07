from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


EXTERNAL_ROBOCASA_ROOT = Path(__file__).resolve().parents[2] / "robocasa"
EXTERNAL_OBJECT_METADATA_PATH = (
    EXTERNAL_ROBOCASA_ROOT / "robocasa" / "models" / "objects" / "kitchen_objects.py"
)
EXTERNAL_SCENE_BUILDER_PATH = (
    EXTERNAL_ROBOCASA_ROOT / "robocasa" / "models" / "scenes" / "scene_builder.py"
)

OBJECT_DOC_URL = "https://robocasa.ai/docs/build/html/assets/objects.html"
FIXTURE_DOC_URL = "https://robocasa.ai/docs/build/html/assets/fixtures.html"


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
                raw = eval(compile(expr, str(EXTERNAL_OBJECT_METADATA_PATH), "eval"), {"dict": dict})
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


@lru_cache(maxsize=1)
def external_fixture_type_to_class() -> Dict[str, str]:
    module = _load_ast(EXTERNAL_SCENE_BUILDER_PATH)
    if module is None:
        return {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "FIXTURES":
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict":
                    mapping: Dict[str, str] = {}
                    for kw in node.value.keywords:
                        key = str(kw.arg)
                        if isinstance(kw.value, ast.Name):
                            mapping[key] = kw.value.id
                    return mapping
    return {}


@lru_cache(maxsize=1)
def fixture_class_to_scene_types() -> Dict[str, List[str]]:
    inverse: Dict[str, List[str]] = {}
    for fixture_type, class_name in external_fixture_type_to_class().items():
        inverse.setdefault(class_name, []).append(fixture_type)
    for values in inverse.values():
        values.sort()
    return inverse


@lru_cache(maxsize=1)
def external_fixture_classes() -> List[str]:
    return sorted(fixture_class_to_scene_types().keys())


@lru_cache(maxsize=1)
def fixture_class_default_attributes() -> Dict[str, List[str]]:
    defaults = {
        "Accessory": [],
        "Blender": ["powered", "mixing"],
        "BlenderLid": [],
        "Box": [],
        "CoffeeMachine": ["heated", "powered", "dispensing", "support:heated"],
        "Counter": ["support:prep_zone", "support:serving_zone"],
        "DishRack": ["support:containment", "support:storage_zone"],
        "Dishwasher": ["openable", "washing", "storage", "support:containment", "support:wash_zone", "support:storage_zone"],
        "Drawer": ["openable", "storage", "drawer", "support:containment", "support:storage_zone"],
        "ElectricKettle": ["heated", "powered"],
        "Floor": [],
        "FridgeBottomFreezer": ["openable", "cooling", "storage", "fridge", "support:containment", "support:cold_storage", "support:storage_zone"],
        "FridgeFrenchDoor": ["openable", "cooling", "storage", "fridge", "support:containment", "support:cold_storage", "support:storage_zone"],
        "FridgeSideBySide": ["openable", "cooling", "storage", "fridge", "support:containment", "support:cold_storage", "support:storage_zone"],
        "HingeCabinet": ["openable", "storage", "cabinet", "support:containment", "support:storage_zone"],
        "Hood": [],
        "HousingCabinet": ["openable", "storage", "cabinet", "support:containment", "support:storage_zone"],
        "Microwave": ["openable", "heated", "powered", "microwave", "support:heated"],
        "OpenCabinet": ["openable", "storage", "cabinet", "support:containment", "support:storage_zone"],
        "Oven": ["openable", "heated", "oven", "support:heated"],
        "PanelCabinet": ["openable", "storage", "cabinet", "support:containment", "support:storage_zone"],
        "Sink": ["washing", "sink", "support:containment", "support:wash_zone"],
        "SingleCabinet": ["openable", "storage", "cabinet", "support:containment", "support:storage_zone"],
        "StandMixer": ["powered", "mixing"],
        "Stove": ["heated", "stove", "cooktop", "support:heated"],
        "Stovetop": ["heated", "stove", "cooktop", "support:heated"],
        "Toaster": ["heated", "powered", "toasting"],
        "ToasterOven": ["openable", "heated", "powered", "toasting", "oven", "support:heated"],
        "Wall": [],
        "WallAccessory": [],
        "Window": ["openable"],
        "WindowProc": ["openable"],
    }
    return {key: sorted(value) for key, value in defaults.items()}


def fixture_inventory_source_label() -> str:
    return (
        "external_code:scene_builder.FIXTURES"
        f"; website_docs:{FIXTURE_DOC_URL}"
    )


def object_inventory_source_label() -> str:
    return (
        "external_code:OBJ_CATEGORIES"
        f"; website_docs:{OBJECT_DOC_URL}"
    )
