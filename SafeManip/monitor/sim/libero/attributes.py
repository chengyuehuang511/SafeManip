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
    # "bottle" and "can" added 2026-09-16 (explicit user decision), verified
    # against RoboCasa's own RECEPTACLE_CATEGORIES (attributes.py:219-235,
    # eval/simulators/robocasa/robocasa/models/objects/kitchen_objects.py):
    # that set lists "bottle" and "can" literally, independent of their
    # "types" metadata (wine's types are only ("drink", "alcohol"), no
    # "receptacle"/"liquid") -- so RoboCasa itself treats a bottle/can as a
    # receptacle shape regardless of contents. Without these, LIBERO's real
    # wine_bottle object was never recognized as receptacle-shaped even
    # though its real RoboCasa analog (category "bottle") is.
    "bottle",
    "can",
    # "ramekin" added 2026-09-16 (explicit user decision): no RoboCasa
    # category named "ramekin" exists at all (confirmed by grepping
    # kitchen_objects.py), same situation as "plate"/"basket"/"caddy"/"bin"
    # above (this file's own docstring already calls those out as LIBERO
    # categories with no direct RoboCasa-taxonomy equivalent, judged by real
    # shape instead) -- a ramekin is a small bowl-shaped dish, receptacle by
    # the same shape reasoning, and was already treated as such by
    # FRAGILE_NAME_SUBSTRINGS/MICROWAVABLE_NAME_SUBSTRINGS/WASHABLE_NAME_
    # SUBSTRINGS elsewhere in this file -- omitting it here (the actual
    # receptacle-shape check) while including it there was an inconsistency,
    # not a deliberate distinction.
    "ramekin",
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

# Added 2026-09-16 (explicit user decision): mirrors RoboCasa's own
# attributes.py taxonomy for LIBERO's simpler flat object set, so
# rc_place_preconditions_safe's support_objects_clean_for_manipulated_object/
# support_not_cluttered_for_fragile_manipulated_object sub-checks (previously
# entirely absent from LIBERO's preconditions_satisfied_place composition,
# not just stubbed) can be implemented for real instead of omitted. Per
# explicit user direction, implemented unconditionally -- whether any actual
# LIBERO object matches a given set is left to the attribute check itself to
# determine, not pre-judged by corpus inspection the way RAW_NAME_SUBSTRINGS/
# RTE_NAME_SUBSTRINGS's own verified-zero-occurrence claims were.

# FRAGILE_CATEGORIES in RoboCasa's own attributes.py ({"mug", "coffee_cup",
# "cup", "glass_cup", "wine_glass", "plate", "bowl", "egg"}) is itself
# manually curated there (not derived from RoboCasa's external per-category
# metadata) -- mirrored here against LIBERO's own category-name substrings.
# "ramekin" added 2026-09-16 after verifying the actual 40-task object
# corpus (glazed_rim_porcelain_ramekin, a real in-corpus distractor object,
# not previously matched by anything -- see the corpus-verification note
# further down this file).
FRAGILE_NAME_SUBSTRINGS = ("bowl", "mug", "cup", "glass", "plate", "egg", "ramekin")

# RoboCasa's OBJECT_METADATA_BOOLEAN_ATTRIBUTES (washable, microwavable,
# cookable, toastable, fridgable, freezable, dishwashable) are read directly
# off RoboCasa's own external per-category metadata (kitchen_objects.py),
# which has no LIBERO equivalent -- approximated here as name-substring
# sets. Not yet consumed by any LIBERO precondition check (RoboCasa's own
# consumers of these -- preconditions_satisfied_open_close's fixture/content
# type-matching for microwave/dishwasher contents -- haven't been ported to
# LIBERO); provided so that porting work doesn't also require inventing the
# attribute data from scratch. (WASHABLE_NAME_SUBSTRINGS/DISHWASHABLE_NAME_
# SUBSTRINGS/FRIDGABLE_NAME_SUBSTRINGS/FREEZABLE_NAME_SUBSTRINGS live further
# down this file, next to COOKABLE_NAME_SUBSTRINGS, since they need to reuse
# MEAT_NAME_SUBSTRINGS/DAIRY_NAME_SUBSTRINGS/LIQUID_NAME_SUBSTRINGS and must
# be defined after those to avoid a forward-reference NameError -- the same
# ordering hazard the "Fixed 2026-09-16" COOKABLE/FRIDGABLE note above
# describes.) MICROWAVABLE and TOASTABLE have no such dependency, so they
# stay here, confirmed True only for bowl/mug/cup/plate/ramekin-shaped dishes
# (RoboCasa: pot/pan/tray/basket and every bottled liquid are confirmed
# microwavable=False) and toastable only for bread-shaped items, respectively.
MICROWAVABLE_NAME_SUBSTRINGS = ("bowl", "mug", "cup", "plate", "ramekin")
TOASTABLE_NAME_SUBSTRINGS = ("bread",)

# TOOL_CATEGORIES mirror (RoboCasa: knife/fork/spoon/spatula/ladle/whisk/
# tongs/peeler/reamer/rolling_pin/pizza_cutter/measuring_cup/can_opener/
# bottle_opener/cheese_grater/scissors/digital_scale/strainer/cutting_board).
TOOL_NAME_SUBSTRINGS = (
    "knife", "fork", "spoon", "spatula", "ladle", "whisk", "tongs",
    "peeler", "opener", "grater", "scissors", "scale", "strainer",
    "cutting_board", "rolling_pin",
)

# POURABLE_CATEGORIES mirror (RoboCasa: jug/pitcher/teapot/thermos/mug/
# coffee_cup/cup/glass_cup/wine_glass, plus its own LIQUID_CATEGORIES).
POURABLE_NAME_SUBSTRINGS = (
    "jug", "pitcher", "teapot", "thermos", "mug", "cup", "glass",
    *LIQUID_NAME_SUBSTRINGS,
)

# Added 2026-09-16 (explicit user decision): the rest of RoboCasa's
# ROLE_ATTRIBUTE_AXES["object"]-derived, actually-populated attributes (not
# just declared in the axis list -- verified against
# object_category_attribute_defaults() itself, not the schema). Deliberately
# NOT mirrored: "hot"/"cold"/"cleaner"/"stackable"/"lid"/"cuttable"/
# "packaged_food" -- RoboCasa's own attribute-assignment function never
# actually assigns these to any object (PLACEHOLDER_TRUE_ATTRIBUTES, which
# includes hot/cold, is dead code -- not consumed anywhere in that
# codebase), so mirroring them would just copy inert schema names, not real
# behavior. Also NOT mirrored: TWISTABLE_CATEGORIES-for-movable-objects --
# already correctly excluded elsewhere in this file (LIBERO movable objects
# have no articulated cap/lid sub-mechanism to twist, confirmed against the
# actual object model).
FRUIT_NAME_SUBSTRINGS = ("apple", "banana", "orange", "grape", "berry", "lemon", "lime", "peach", "pear")
VEGETABLE_NAME_SUBSTRINGS = ("carrot", "potato", "onion", "lettuce", "tomato", "pepper", "corn", "broccoli", "cucumber")
MEAT_NAME_SUBSTRINGS = ("meat", "chicken", "beef", "pork", "fish", "bacon", "sausage")
DAIRY_NAME_SUBSTRINGS = ("milk", "cheese", "butter", "cream", "yogurt")
BREAD_FOOD_NAME_SUBSTRINGS = ("bread", "toast", "bagel", "bun", "roll")
COOKED_FOOD_NAME_SUBSTRINGS = ("cooked", "pudding")
CONDIMENT_NAME_SUBSTRINGS = ("ketchup", "mustard", "mayo", "dressing", "sauce", "jam", "jelly")
SPICE_NAME_SUBSTRINGS = ("salt", "spice", "seasoning", "pepper")
PASTRY_NAME_SUBSTRINGS = ("pastry", "cake", "cookie", "donut", "pie", "croissant")
SWEET_NAME_SUBSTRINGS = ("sweet", "candy", "chocolate", "sugar")
DRINK_NAME_SUBSTRINGS = ("juice", "milk", "wine", "coffee", "soda", "water", "beer")
ALCOHOL_NAME_SUBSTRINGS = ("wine", "beer", "liquor")
COOKWARE_NAME_SUBSTRINGS = ("pot", "pan", "frypan", "skillet", "kettle", "wok")

# FOOD_TYPE_NAMES mirror -- union of the food-ish sub-categories above
# (RoboCasa: fruit, vegetable, meat, dairy, bread_food, cooked_food, pastry,
# sweets, condiment, spice, drink).
FOOD_NAME_SUBSTRINGS = (
    *FRUIT_NAME_SUBSTRINGS, *VEGETABLE_NAME_SUBSTRINGS, *MEAT_NAME_SUBSTRINGS,
    *DAIRY_NAME_SUBSTRINGS, *BREAD_FOOD_NAME_SUBSTRINGS, *COOKED_FOOD_NAME_SUBSTRINGS,
    *PASTRY_NAME_SUBSTRINGS, *SWEET_NAME_SUBSTRINGS, *CONDIMENT_NAME_SUBSTRINGS,
    *SPICE_NAME_SUBSTRINGS, *DRINK_NAME_SUBSTRINGS,
)

# Fixed 2026-09-16: COOKABLE_NAME_SUBSTRINGS/FRIDGABLE_NAME_SUBSTRINGS were
# originally defined (above, near WASHABLE/MICROWAVABLE) using abstract type
# labels ("fruit", "vegetable", "dairy", "bread", "meat") instead of actual
# object-name substrings -- those labels can never match a real category
# string like "cream_cheese" or "chocolate_pudding". Redefined here by
# reusing the real substring sets instead (same pattern as
# UTENSIL_NAME_SUBSTRINGS = TOOL_NAME_SUBSTRINGS above), now that those real
# sets exist. Verified against the actual 40-task object corpus: FRIDGABLE
# genuinely matches milk/butter/cream_cheese; COOKABLE has no real match in
# this corpus (no raw ingredients needing cooking), a legitimate
# zero-occurrence result, same class as RAW_NAME_SUBSTRINGS's own.
COOKABLE_NAME_SUBSTRINGS = (
    *FRUIT_NAME_SUBSTRINGS, *VEGETABLE_NAME_SUBSTRINGS, *MEAT_NAME_SUBSTRINGS,
    *DAIRY_NAME_SUBSTRINGS, *BREAD_FOOD_NAME_SUBSTRINGS, "egg",
)

# Corrected 2026-09-16 (explicit user decision), verified directly against
# RoboCasa's real per-category dicts in kitchen_objects.py (not just its
# "types" tuples): fridgable=True is assigned far more broadly than just
# meat/dairy -- confirmed True for bowl, mug, cup, plate, pan, pot (all
# receptacle/cookware shapes), and for milk, ketchup, juice, can (all bottled
# food/drink items; e.g. kitchen_objects.py's `ketchup=dict(... fridgable=True
# ...)`, `bowl=dict(... fridgable=True ...)`). Confirmed fridgable=False for
# `wine=dict(...)` and `basket=dict(...)` specifically -- those two are
# deliberately excluded below rather than folded in via a broader shape
# substring, so LIBERO's wine_bottle/basket don't get a false "fridgable".
# The old FRIDGABLE_NAME_SUBSTRINGS = MEAT+DAIRY only (this file's own prior
# "Fixed 2026-09-16" comment above claimed FRIDGABLE "genuinely matches
# milk/butter/cream_cheese" as if that were the full real set -- it understated
# it; bowl/mug/plate/pan/pot/ketchup/juice/alphabet_soup ("soup", via
# LIQUID_NAME_SUBSTRINGS) are real matches too).
FRIDGABLE_NAME_SUBSTRINGS = (
    "bowl", "mug", "cup", "plate", "pan", "pot",
    *MEAT_NAME_SUBSTRINGS, *DAIRY_NAME_SUBSTRINGS, *LIQUID_NAME_SUBSTRINGS,
)

# freezable=True is, by contrast, genuinely narrow in RoboCasa's real data --
# confirmed True only for dairy-ish items (`cream_cheese_stick`, `butter_stick`)
# and for `can=dict(... freezable=True ...)`; confirmed False for every
# receptacle/cookware shape (bowl/mug/cup/plate/pan/pot/tray) and for the
# bottled liquids (milk/ketchup/wine/juice all freezable=False). So, unlike
# the old code, FREEZABLE is deliberately NOT aliased to the (now broader)
# FRIDGABLE_NAME_SUBSTRINGS above -- it keeps its own narrow set instead
# (dairy/meat, plus "soup"/"can" for the real `can=dict(freezable=True)`
# match, covering LIBERO's alphabet_soup).
FREEZABLE_NAME_SUBSTRINGS = (*MEAT_NAME_SUBSTRINGS, *DAIRY_NAME_SUBSTRINGS, "soup", "can")

# washable=True is assigned to nearly every RoboCasa category *except* dry/
# baked/packaged solid foods (bread, cake, chocolate, cereal, chips, ... all
# washable=False) and a few tools (digital_scale). Confirmed True for the
# receptacle/cookware shapes (bowl/mug/cup/plate/pan/pot/tray/basket) *and*
# for the bottled liquids (milk, ketchup, wine, juice/`juice=dict(...
# washable=True ...)`) *and* for cream_cheese_stick (washable=True) -- but
# confirmed False for butter_stick specifically (washable=False even though
# it's dairy too, a genuine per-item inconsistency in RoboCasa's own data, not
# a mirroring gap on this side). "basket"/"wine"/"cheese"/"cream" added
# 2026-09-16 after this direct verification; previously this file aliased
# DISHWASHABLE_NAME_SUBSTRINGS = WASHABLE_NAME_SUBSTRINGS, which was wrong in
# both directions (see DISHWASHABLE_NAME_SUBSTRINGS below).
WASHABLE_NAME_SUBSTRINGS = (
    "bowl", "mug", "cup", "plate", "pot", "pan", "tray", "caddy", "ramekin",
    "basket", "wine", "cheese", "cream", *LIQUID_NAME_SUBSTRINGS,
)

# dishwashable=True is, unlike washable, a *narrow* real RoboCasa attribute --
# confirmed True only for bowl/mug/pan/plate/pot/knife/ladle/spoon/colander/
# reamer/measuring_cup/strainer/tupperware/glass_cup/peeler/saucepan
# (kitchen_objects.py) -- confirmed False (i.e. field simply absent, which
# `info.get("dishwashable")` treats as falsy) for plain cup, coffee_cup,
# tray, basket, and wine_glass, and for every bottled liquid (milk, ketchup,
# wine, juice, can). So aliasing it to WASHABLE_NAME_SUBSTRINGS was a real
# mirroring bug -- washable is much broader than dishwashable in RoboCasa's
# own data. Corrected 2026-09-16 (explicit user decision) to its own narrow
# set, restricted to the receptacle/cookware shapes that are actually
# dishwasher-safe dishware, matching LIBERO's real akita_black_bowl/mugs/
# plate/chefmate_8_frypan/moka_pot.
DISHWASHABLE_NAME_SUBSTRINGS = ("bowl", "mug", "plate", "pan", "pot")

# utensil is assigned to the exact same category set as "tool" in RoboCasa's
# own object_category_attribute_defaults() (attrs.update({"tool", "utensil",
# "graspable"})) -- not a separate taxonomy, so reuse TOOL_NAME_SUBSTRINGS
# directly rather than re-declaring an identical list under a new name.
UTENSIL_NAME_SUBSTRINGS = TOOL_NAME_SUBSTRINGS

# graspable is assigned unconditionally to every object in RoboCasa
# (attrs.add("graspable") with no category gate at all) -- not a
# substring-matched attribute, just always True for any real object.

# Added 2026-09-16 (explicit user decision): mirrors RoboCasa's
# fixture_class_default_attributes() -- the "fixture" and "support" role
# axes (ROLE_ATTRIBUTE_AXES["fixture"]/["support"]), keyed there by RoboCasa
# fixture *class* names (e.g. "HingeCabinet", "Microwave", "Stove") which
# have no LIBERO equivalent naming -- re-keyed here by LIBERO's own flat
# fixture category name instead, for the fixture classes that actually have
# a reasonable RoboCasa analog. "support:X" attributes (containment,
# heated, cold_storage, wash_zone, prep_zone, serving_zone, storage_zone)
# kept with their "support:" prefix intact, matching RoboCasa's own
# encoding, since they describe the surface/interior the fixture provides,
# not the fixture body itself.
#
# desk_caddy and wine_rack map to RoboCasa's DishRack class -- corrected
# 2026-09-16 (explicit user decision): RoboCasa's DishRack attributes are
# exactly ("support:containment", "support:storage_zone"), no openable/
# closeable/body attributes at all, because a dish rack is an open storage
# surface with no door or lid -- the same shape as a desk caddy or wine
# rack (open organizers, nothing to open/close). This is a real, reasoned
# analog, not a fabrication -- unlike the fixture-*body* attributes
# (openable, heated, powered, etc.), which genuinely have no RoboCasa
# equivalent for these two since neither is a sealed/powered fixture.
# flat_stove maps to RoboCasa's Stove/Stovetop (a cooktop with no oven
# cavity, matching LIBERO's own "flat_stove has no openable interior" note
# elsewhere in this file) rather than Oven/ToasterOven.
FIXTURE_CATEGORY_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "microwave": ("openable", "closeable", "heated", "powered", "microwave", "pressable", "support:heated"),
    "white_cabinet": ("openable", "closeable", "storage", "cabinet", "support:containment", "support:storage_zone"),
    "wooden_cabinet": ("openable", "closeable", "storage", "cabinet", "support:containment", "support:storage_zone"),
    "flat_stove": ("heated", "stove", "cooktop", "twistable", "support:heated"),
    "desk_caddy": ("support:containment", "support:storage_zone"),
    "wine_rack": ("support:containment", "support:storage_zone"),
}


def fixture_category_has_attribute(category: str, attribute: str) -> bool:
    return attribute in FIXTURE_CATEGORY_ATTRIBUTES.get(str(category).lower(), ())


# RoboCasa's remaining two ROLE_ATTRIBUTE_AXES -- "tool" (cleaning_tool/
# cutting_tool/mixing_tool/serving_tool/measuring_tool) and "button"
# (lever/toggle/power/mode_select/pressable) -- corrected 2026-09-16 after
# checking their actual consumption in RoboCasa's own code, not just
# declaring them inapplicable by assumption:
# - Both TOOL_ATTRIBUTE_AXES and BUTTON_ATTRIBUTE_AXES (attributes.py:148-149)
#   are themselves dead code in RoboCasa -- declared, never referenced
#   again anywhere in that codebase, same class as PLACEHOLDER_TRUE_
#   ATTRIBUTES (hot/cold). There is no real per-entity assignment logic to
#   mirror for either sub-axis at all, in RoboCasa itself.
# - The *functional* purpose each axis gestures at is handled by other,
#   already-mirrored mechanisms instead: "tool"'s parent classification
#   (RoboCasa's real, consumed TOOL_CATEGORIES, attributes.py:474 --
#   assigns flat "tool"/"utensil"/"graspable" tags) is mirrored above via
#   TOOL_NAME_SUBSTRINGS/UTENSIL_NAME_SUBSTRINGS, correctly showing zero
#   real matches in the verified 22-object corpus (no tool objects exist,
#   not a mirroring gap). "button"-style press-component targeting (e.g.
#   RoboCasa predicates.py's own "button"/"power_button"/"lever" keyword
#   matching for a fixture's press-affordance sub-geom) is already
#   mirrored via this file's own ACTION_COMPONENT_KEYWORDS["press"]
#   (imported from predicates.py verbatim, per its own docstring),
#   independent of the (dead) formal ROLE_ATTRIBUTE_AXES["button"] system.
