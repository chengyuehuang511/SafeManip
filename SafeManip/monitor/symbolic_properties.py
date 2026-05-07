from typing import List

from monitor.SymbolicProperty import SymbolicProperty
from monitor.specs import (
    TASK_AGNOSTIC_PROPERTY_SPECS,
    VARIANT_PROPERTY_SPECS,
    all_entities,
    predicate_lookup,
)


def _materialize_property(spec):
    lookup = predicate_lookup()
    predicates = [(name, lookup[name]) for name in spec["predicates"]]
    return SymbolicProperty(spec["name"], spec["ltl"], predicates, all_entities())


def build_task_agnostic_properties() -> List[SymbolicProperty]:
    return [_materialize_property(spec) for spec in TASK_AGNOSTIC_PROPERTY_SPECS]


def build_variant_properties() -> List[SymbolicProperty]:
    return [_materialize_property(spec) for spec in VARIANT_PROPERTY_SPECS]


def build_all_properties() -> List[SymbolicProperty]:
    return (
        build_task_agnostic_properties()
        + build_variant_properties()
    )


ALL_PROPERTIES = build_all_properties()
