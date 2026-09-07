from typing import List

from monitor.LTLfDFA import LTLfDFA
from monitor.specs import (
    TASK_AGNOSTIC_PROPERTY_SPECS,
    VARIANT_PROPERTY_SPECS,
    predicate_lookup,
)


class SymbolicProperty:
    def __init__(
        self,
        property_name: str,
        property_string: str,
        predicates,
    ):
        self.name = property_name
        self.ltldfa = LTLfDFA(property_string)
        needed_symbols = set(self.ltldfa.symbols)
        self.predicates = {pred[0]: pred[1] for pred in predicates}
        have_symbols = set(self.predicates.keys())
        missing_symbols = needed_symbols - have_symbols
        if len(missing_symbols) > 0:
            raise AttributeError(
                "The provided list of predicates is insufficient to evaluate the formula."
                f"Found symbols: {have_symbols}, Need symbols: {needed_symbols}, "
                f"Missing: {missing_symbols}"
            )


def _materialize_property(spec):
    lookup = predicate_lookup()
    predicates = [(name, lookup[name]) for name in spec["predicates"]]
    return SymbolicProperty(spec["name"], spec["ltl"], predicates)


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
