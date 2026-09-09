"""
LIBERO object taxonomy -- the LIBERO-side analog of
`monitor/sim/robocasa/attributes.py`, at a fraction of the size: LIBERO's
tabletop-manipulation object set (per `libero/libero/envs/objects/`'s
`register_object`-decorated classes, and the object names visible across the
40 in-scope task suites' bddl files) is much smaller and flatter than
RoboCasa's kitchen-fixture/food taxonomy -- there is no fixture-class
inheritance tree to introspect here.

This module is NOT wired into `monitor/predicates.py`'s
`monitor.sim.robocasa.attributes` try/except import (that stays RoboCasa-only,
per this integration's constraint of not touching any file outside
`monitor/sim/libero/`) -- so for LIBERO episodes, that top-level import
silently falls back to its small hardcoded receptacle set
(`{"bowl", "cup", "mug", "pot", "pan", "tray", "container", "tupperware",
"bottle", "can"}`), which already happens to cover most of LIBERO's actual
receptacle-shaped objects (black_bowl, white/yellow_and_white/red mug,
moka_pot, chefmate_8_frypan, basket, tray, caddy, plate). This module exists
so `monitor/sim/libero/predicates.py` has a single place to ask "is this
object a receptacle" using LIBERO's own naming conventions when the object
name doesn't match that fallback set exactly (e.g. "plate", "basket", "caddy",
"bin" are LIBERO object categories with no direct RoboCasa-taxonomy
equivalent).
"""
from __future__ import annotations

import re
from typing import Iterable

# Substrings (checked against the lowercased LIBERO object *category* name --
# the leading digits/instance-suffix-stripped identifier, e.g.
# "chefmate_8_frypan_1" -> category "chefmate_8_frypan") that mark an object
# as receptacle-shaped for `object_upright_if_receptacle`/support-type checks.
RECEPTACLE_NAME_SUBSTRINGS = (
    "bowl",
    "mug",
    "cup",
    "pot",
    "pan",
    "plate",
    "basket",
    "tray",
    "caddy",
    "bin",
    "container",
    "box",  # cream_cheese_box etc. are NOT receptacles by shape, but this
    # substring also catches "moka_pot" incorrectly only if "pot" weren't
    # already listed first -- kept last/lowest-priority on purpose so more
    # specific non-receptacle categories can be special-cased below first.
)

# Object-category substrings that would otherwise match RECEPTACLE_NAME_SUBSTRINGS
# above but are not actually receptacle-shaped (packaged/solid grocery items).
_RECEPTACLE_NAME_EXCEPTIONS = (
    "cream_cheese_box",
    "butter_box",
    "milk_box",
)


def object_category_from_instance_name(name: str) -> str:
    """LIBERO object instance names are typically `<category>_<index>`
    (e.g. "akita_black_bowl_1", "wine_bottle_1") -- strip the trailing
    `_<digits>` instance suffix to recover the category, matching the
    convention `libero/libero/envs/base_object.py`'s `register_object` uses
    for its own registry keys."""
    return re.sub(r"_\d+$", "", str(name)).lower()


def object_is_receptacle_category(category: str, extra_attrs: Iterable[str] = ()) -> bool:
    category = str(category).lower()
    if any(exc in category for exc in _RECEPTACLE_NAME_EXCEPTIONS):
        return False
    if any(sub in category for sub in RECEPTACLE_NAME_SUBSTRINGS):
        return True
    extra = {str(a).lower() for a in extra_attrs}
    return bool(extra & set(RECEPTACLE_NAME_SUBSTRINGS))


# Fixture-name substrings that mark a fixture as an "openable enclosure" for
# the access/enclosure-safety predicate family (one_object_in_microwave,
# reach_in_fixture, object_in_fixture, ...). LIBERO fixture instance names
# follow the same `<category>_<index>` convention (e.g. "microwave_1",
# "wooden_cabinet_1", "flat_stove_1" -- the last has no openable interior and
# is deliberately not matched here).
OPENABLE_FIXTURE_NAME_SUBSTRINGS = (
    "microwave",
    "cabinet",
    "drawer",
    "fridge",
    "refrigerator",
    "oven",
)

MICROWAVE_FIXTURE_NAME_SUBSTRINGS = ("microwave",)

# Verbatim copy of monitor/sim/robocasa/predicates.py's own
# ACTION_COMPONENT_KEYWORDS (plain data, no robocasa code dependency -- safe
# to duplicate). Used generically the same way robocasa itself uses it: a
# fixture/object is tagged with an action if its name/category matches one of
# these keyword substrings. "drawer" appears in both "slide" and "open_close"
# (robocasa disambiguates via the fixture class's own slideable/openable
# attribute tag; LIBERO has no such attribute, so predicates.py's
# `_fixture_joint_class` uses raw MuJoCo joint type -- SLIDE vs HINGE -- as
# the tie-breaker instead, same idea).
ACTION_COMPONENT_KEYWORDS = {
    "press": ("lever", "button", "press", "switch", "control", "start", "stop", "power", "cancel"),
    "turn": ("faucet", "handle", "spout"),
    "slide": ("handle", "pull", "slide", "rack", "tray", "lever"),
    "twist": ("cap", "lid", "knob", "dial", "collar", "bottle", "jar", "can", "temperature", "temp", "timer", "time"),
    "open_close": ("handle", "door", "lid", "head", "hinge", "drawer"),
}

# Fixture-name substrings identifying a sink faucet -- the one fixture class
# in LIBERO (Faucet/BasinFaucet, libero/libero/envs/objects/articulated_objects.py)
# with a turn_on()/turn_off()-style affordance relevant to "turn"/containment
# (sink water flow) and to contamination "sanitized" (rinsing). None of the
# 40 in-scope tasks reference one -- confirmed by grepping all 40 language
# instructions -- so this legitimately just never matches for this corpus;
# implemented anyway so that's a verified result, not an assumption.
FAUCET_FIXTURE_NAME_SUBSTRINGS = ("faucet", "sink")

# Object-category substrings marking "liquid-ish" transferred content (for
# content_is_liquid/content_is_solid -- monitor/predicates.py's own fallback
# for these matches against external per-category metadata this integration
# doesn't have, so LIBERO computes them explicitly instead of relying on that
# fallback). None of the 40 in-scope tasks involve pouring liquid content, so
# this is expected to genuinely never fire for this corpus -- verified, not
# assumed.
LIQUID_NAME_SUBSTRINGS = ("juice", "milk", "sauce", "dressing", "ketchup", "oil", "broth", "soup")

# Object-category substrings marking "raw" (uncooked perishable) content, for
# the contamination family -- mirrors monitor/predicates.py's own
# `_object_has_attribute`'s hardcoded "raw" fallback set exactly
# ({"raw", "meat", "fish", "seafood"}), just matched against LIBERO's own
# category-name string instead of external per-category metadata (which this
# integration doesn't have). None of the 40 in-scope tasks' objects are raw
# perishables -- verified by inspecting the actual object list, not assumed.
RAW_NAME_SUBSTRINGS = ("raw", "meat", "fish", "seafood", "chicken", "beef", "uncooked")

# Ready-to-eat/cooked substrings, mirrors monitor/predicates.py's
# "ready_to_eat" fallback set ({"ready_to_eat", "cooked_food", "fruit",
# "vegetable", "dairy", "bread_food", "pastry"}).
RTE_NAME_SUBSTRINGS = ("cooked", "fruit", "vegetable", "dairy", "bread", "pastry", "pudding", "cheese", "butter")
