import itertools
from typing import Dict, List, Union

from functools import partial

from monitor.LTLfDFA import LTLfDFA, DFAView
from monitor.SymbolicEntity import SymbolicEntity, ConcreteEntity, UnboundEntityError


def valid_mapping(node_mapping_list):
    seen_nodes = set()
    for node in node_mapping_list:
        if node not in seen_nodes:
            seen_nodes.add(node)
        elif node is not None:
            return False
    if len(seen_nodes) == 1 and next(iter(seen_nodes)) is None:
        return False
    return True


def get_concrete_entities(sg, symbolic_entities, include_none=False):
    possible_mappings: List[List[ConcreteEntity]] = []
    for symbolic_entity in symbolic_entities:
        possible = [node.get_id() for node in sg.nodes if symbolic_entity.is_valid(node) and not node.is_phantom()]
        if include_none:
            possible.append(None)
        possible_mappings.append(possible)
    return [
        {
            symbolic_entity: (ConcreteEntity(symbolic_entity, node_id) if node_id is not None else None)
            for symbolic_entity, node_id in zip(symbolic_entities, prod)
        }
        for prod in itertools.product(*possible_mappings)
        if valid_mapping(prod)
    ]


def get_symbolic_entities(predicate):
    symbolic_entities = set()
    if predicate.func.__name__ in ["defined"]:
        if len(predicate.args) != 1:
            raise ValueError("defined takes exactly one argument")
        symbolic_entities.add(predicate.args[0])
    for arg in predicate.args:
        if isinstance(arg, partial):
            symbolic_entities.update(get_symbolic_entities(arg))
        elif isinstance(arg, SymbolicEntity):
            symbolic_entities.add(arg)
    return symbolic_entities


class SymbolicProperty:
    def __init__(
        self,
        property_name: str,
        property_string: str,
        predicates,
        symbolic_entities: List[SymbolicEntity],
    ):
        self.name = property_name
        self.ltldfa = LTLfDFA(property_string)
        needed_symbols = set(self.ltldfa.symbols)
        self.predicates = {pred[0]: pred[1] for pred in predicates}
        have_symbols = set(self.predicates.keys())
        missing_symbols = needed_symbols - have_symbols
        self.symbolic_entities = symbolic_entities
        if len(missing_symbols) > 0:
            raise AttributeError(
                "The provided list of predicates is insufficient to evaluate the formula."
                f"Found symbols: {have_symbols}, Need symbols: {needed_symbols}, "
                f"Missing: {missing_symbols}"
            )
        self.symbol_to_entities = {}
        for symbol, predicate in self.predicates.items():
            entity_list = sorted(list(get_symbolic_entities(predicate)), key=lambda x: x.name)
            self.symbol_to_entities[symbol] = entity_list

    def make_blank(self, sg):
        return ConcreteProperty(
            self.name,
            self.ltldfa,
            self.predicates,
            sg.graph["frame"],
            {symbolic_entity: None for symbolic_entity in self.symbolic_entities},
            self.symbol_to_entities,
        )

    def make_concrete(self, sg):
        possible_mappings = get_concrete_entities(sg, self.symbolic_entities, include_none=True)
        if len(possible_mappings) == 0:
            return []
        return [
            ConcreteProperty(
                self.name,
                self.ltldfa,
                self.predicates,
                sg.graph["frame"],
                possible_mapping,
                self.symbol_to_entities,
            )
            for possible_mapping in possible_mappings
        ]


class ConcreteProperty:
    def __init__(
        self,
        name,
        ltlfdfa: LTLfDFA,
        predicates,
        frame,
        entity_mapping: Dict[SymbolicEntity, Union[ConcreteEntity, None]],
        symbol_to_sym,
        current_state=None,
    ):
        self.name = name
        self.dfa_view = DFAView(ltlfdfa, current_state=current_state)
        self.predicates = predicates
        self.initial_frame = frame
        self.symbol_to_sym = symbol_to_sym
        self.entity_mapping = entity_mapping
        self.data_history = {}
        self.name_history = {}
        self.frames = []
        self.undef = []
        self.cache_key = {}
        for symbol, entity_list in self.symbol_to_sym.items():
            self.cache_key[symbol] = (
                f"{self.name}_{symbol}_"
                f"{[(sym.name, self.entity_mapping[sym].entity_id or None if self.entity_mapping[sym] is not None else None) for sym in entity_list]}"
            )
        sorted_symbolic_entities = sorted(list(entity_mapping.keys()), key=lambda x: x.name)
        self.cache_key_str = (
            f"{self.name}_"
            f"{[(sym.name, self.entity_mapping[sym].entity_id or None if self.entity_mapping[sym] is not None else None) for sym in sorted_symbolic_entities]}"
        )

    def is_trap(self):
        return self.dfa_view.is_trap()

    def is_accepting(self):
        return self.dfa_view.is_accepting()

    def __repr__(self):
        return self.cache_key_str

    def __evaluate_predicate(self, predicate, sg):
        if predicate.func.__name__ in ["defined"]:
            if len(predicate.args) != 1:
                raise ValueError("defined takes exactly one argument")
            var = predicate.args[0]
            defined = any([var == key and value is not None for key, value in self.entity_mapping.items()])
            if not defined:
                self.undef.append(var)
            return defined
        param_list = []
        for arg in predicate.args:
            if isinstance(arg, partial):
                param_list.append(self.__evaluate_predicate(arg, sg))
            elif isinstance(arg, SymbolicEntity):
                if self.entity_mapping[arg] is None:
                    self.undef.append(arg)
                    param_list.append(UnboundEntityError([arg]))
                else:
                    param_list.append(self.entity_mapping[arg].get_node(sg))
            else:
                param_list.append(arg)
        if predicate.func.__name__ in ["filterByAttr", "relSet"]:
            return predicate.func(*param_list, sg, self.entity_mapping, **predicate.keywords)
        return predicate.func(*param_list, **predicate.keywords)

    def check_cache(self, sg, symbol):
        key = self.cache_key[symbol]
        return sg.graph["cache"][key] if key in sg.graph["cache"] else None

    def update_cache(self, sg, symbol, res):
        sg.graph["cache"][self.cache_key[symbol]] = res

    def step(self, sg):
        data_dict = {}
        valid_states = []
        unbound_entities = set()
        for u, v, a in self.dfa_view.ltlfdfa._dfa.out_edges(self.dfa_view.current_state, data=True):
            cache_miss = [symbol for symbol in a["symbols"] if symbol not in data_dict]
            for symbol in cache_miss:
                cached = self.check_cache(sg, symbol)
                if cached is None:
                    res = self.__evaluate_predicate(self.predicates[symbol], sg)
                    self.update_cache(sg, symbol, res)
                else:
                    res = cached
                if type(res) == UnboundEntityError:
                    unbound_entities.update(res.entities)
                data_dict[symbol] = res
            cur_unbound = any([type(data_dict[symbol]) == UnboundEntityError for symbol in a["symbols"]])
            if eval(a["label"], dict(data_dict)) and not cur_unbound:
                valid_states.append(v)
        if len(valid_states) != 1:
            if len(unbound_entities) > 0:
                raise UnboundEntityError(list(unbound_entities))
            raise ValueError(
                f"Unable to find state transition from {self.dfa_view.current_state} with {data_dict}, aborting."
            )
        for symbol, value in data_dict.items():
            if isinstance(value, partial):
                data_dict[symbol] = None
        self.data_history[sg.graph["frame"]] = data_dict
        self.name_history[sg.graph["frame"]] = {
            symbolic_entity: concrete_entity.get_node_name(sg) if concrete_entity is not None else None
            for symbolic_entity, concrete_entity in self.entity_mapping.items()
        }
        self.frames.append(sg.graph["frame"])
        self.dfa_view.current_state = valid_states[0]

    def additional_concrete(self, sg):
        needs_binding = [
            symbolic_entity
            for symbolic_entity, concrete_entity in self.entity_mapping.items()
            if concrete_entity is None
        ]
        return self.additional_concrete_specific(sg, needs_binding)

    def additional_concrete_specific(self, sg, needs_binding, include_none=True, current_state=None):
        if any([self.entity_mapping[a] is not None for a in needs_binding]):
            raise ValueError
        possible_mappings = get_concrete_entities(sg, needs_binding, include_none)
        if len(possible_mappings) == 0:
            return []
        ret_val = []
        for possible_mapping in possible_mappings:
            new_val = self.new_entity_copy(possible_mapping, current_state)
            if valid_mapping(new_val.entity_mapping.values()):
                ret_val.append(new_val)
        return ret_val

    def new_entity_copy(self, new_entities, current_state):
        new_mapping = dict(self.entity_mapping)
        new_mapping.update(new_entities)
        new_conc = ConcreteProperty(
            self.name,
            self.dfa_view.ltlfdfa,
            self.predicates,
            self.initial_frame,
            new_mapping,
            self.symbol_to_sym,
            current_state,
        )
        new_conc.data_history = dict(self.data_history)
        new_conc.name_history = dict(self.name_history)
        new_conc.frames = list(self.frames)
        return new_conc

    def get_current_state(self):
        return self.dfa_view.current_state
