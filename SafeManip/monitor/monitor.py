from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from monitor.LTLfDFA import LTLfDFA
from monitor.SymbolicEntity import SymbolicEntity

from monitor.symbolic_properties import build_all_properties
from monitor import predicates as R
import monitor.primitives as P


@dataclass
class RoboCasaProductState:
    world_state: Dict[str, object]
    symbolic_observation: Dict[str, bool]
    dfa_state: str
    binding: Dict[str, str]
    property_name: str


@dataclass
class RoboCasaTraceResult:
    satisfied: bool
    reason: str
    trace: List[RoboCasaProductState]
    final_product_state: Optional[RoboCasaProductState] = None
    failed_step: Optional[int] = None
    failed_action: Optional[object] = None


@dataclass
class RoboCasaPropertyStatus:
    property_name: str
    binding: Dict[str, str]
    accepting: bool
    trap: bool
    current_state: str
    predicate_values: Dict[str, bool]
    product_state: RoboCasaProductState


class RoboCasaSymbolicMonitor:
    """
    Direct symbolic monitor for RoboCasa trajectories.

    The monitor is organized around reachable product states:
    - S: full world state / privileged simulator snapshot
    - X = alpha(S): symbolic observation from predicates / propositions
    - q: current DFA state
    - Z = (S, X, q): product state
    """

    def __init__(self, properties=None):
        self.properties = properties or build_all_properties()
        self._dfa_templates: Dict[str, LTLfDFA] = {}
        self._dfa_states: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], str] = {}
        self._last_values: Dict[str, bool] = {}
        self._last_dynamic_info = None

    def reset(self) -> None:
        self._dfa_states = {}
        self._last_values = {}
        self._last_dynamic_info = None

    def alpha(self, prop, world_state, binding: Dict[str, str], env=None) -> Dict[str, bool]:
        static_info = world_state.get("static") or {}
        dynamic_info = world_state.get("dynamic") or {}
        values: Dict[str, bool] = {}
        for symbol, predicate in prop.predicates.items():
            values[symbol] = bool(
                self._evaluate_partial(
                    predicate,
                    env=env,
                    privileged=world_state,
                    static_info=static_info,
                    dynamic_info=dynamic_info,
                    binding=binding,
                )
            )
        return values

    def build_initial_product_state(
        self,
        prop,
        binding: Dict[str, str],
        *,
        env=None,
        world_state=None,
    ) -> RoboCasaProductState:
        if world_state is None:
            if env is None:
                raise ValueError("Either `env` or `world_state` must be provided.")
            world_state = self._get_initial_world_state(env)
        X0 = self.alpha(prop, world_state, binding, env=env)
        dfa = self._new_dfa(prop)
        q1 = dfa.delta(dfa.q0, X0)
        return RoboCasaProductState(
            world_state=world_state,
            symbolic_observation=X0,
            dfa_state=q1,
            binding=binding,
            property_name=prop.name,
        )

    def step_product_state(
        self,
        prop,
        product_state: RoboCasaProductState,
        *,
        env=None,
        next_world_state=None,
        action=None,
    ) -> RoboCasaProductState:
        if next_world_state is None:
            if env is None:
                raise ValueError("Either `env` or `next_world_state` must be provided.")
            next_world_state = self._transition_world_state(
                env,
                product_state.world_state,
                action,
            )
        X_next = self.alpha(prop, next_world_state, product_state.binding, env=env)
        dfa = self._new_dfa(prop)
        q_next = dfa.delta(product_state.dfa_state, X_next)
        return RoboCasaProductState(
            world_state=next_world_state,
            symbolic_observation=X_next,
            dfa_state=q_next,
            binding=product_state.binding,
            property_name=prop.name,
        )

    def check_dfa_with_reachable_product(
        self,
        env,
        prop,
        binding: Dict[str, str],
        skill_sequence: Sequence[object],
    ) -> RoboCasaTraceResult:
        try:
            Z = self.build_initial_product_state(prop, binding, env=env)
        except Exception as exc:
            return RoboCasaTraceResult(
                satisfied=False,
                reason=f"No initial world state available: {exc}",
                trace=[],
            )

        product_trace = [Z]
        dfa = self._new_dfa(prop)

        for i, action in enumerate(skill_sequence):
            if not self._is_action_applicable(env, Z.world_state, action):
                return RoboCasaTraceResult(
                    satisfied=False,
                    reason=f"Action {action} is not applicable at step {i}",
                    failed_step=i,
                    failed_action=action,
                    trace=product_trace,
                    final_product_state=Z,
                )

            try:
                Z_next = self.step_product_state(prop, Z, env=env, action=action)
            except Exception as exc:
                return RoboCasaTraceResult(
                    satisfied=False,
                    reason=f"Environment transition failed at step {i}: {exc}",
                    failed_step=i,
                    failed_action=action,
                    trace=product_trace,
                    final_product_state=Z,
                )

            product_trace.append(Z_next)

            if Z_next.dfa_state in getattr(dfa, "rejecting_states", set()):
                return RoboCasaTraceResult(
                    satisfied=False,
                    reason=f"Entered rejecting DFA state at step {i}",
                    failed_step=i,
                    failed_action=action,
                    trace=product_trace,
                    final_product_state=Z_next,
                )

            Z = Z_next

        terminal_state = dfa.terminal_delta(Z.dfa_state)
        Z = RoboCasaProductState(
            world_state=Z.world_state,
            symbolic_observation=Z.symbolic_observation,
            dfa_state=terminal_state,
            binding=Z.binding,
            property_name=Z.property_name,
        )

        if terminal_state in dfa.F:
            return RoboCasaTraceResult(
                satisfied=True,
                reason="Final DFA state is accepting",
                trace=product_trace,
                final_product_state=Z,
            )
        return RoboCasaTraceResult(
            satisfied=False,
            reason="Final DFA state is rejecting at episode end",
            trace=product_trace,
            final_product_state=Z,
        )

    def step(self, env=None, privileged=None) -> List[RoboCasaPropertyStatus]:
        if privileged is None:
            if env is None:
                raise ValueError("Either `env` or `privileged` must be provided.")
            privileged = env.get_privileged_information()

        static_info = privileged.get("static") or {}
        dynamic_info = privileged.get("dynamic") or {}
        results: List[RoboCasaPropertyStatus] = []

        for prop in self.properties:
            for binding in self._enumerate_bindings(prop, static_info, dynamic_info):
                dfa = self._new_dfa(prop)
                dfa_key = self._dfa_key(prop.name, binding)
                X = self.alpha(prop, privileged, binding, env=env)
                q_curr = self._dfa_states.get(dfa_key, dfa.q0)
                q_next = dfa.delta(q_curr, X)
                self._dfa_states[dfa_key] = q_next
                product_state = RoboCasaProductState(
                    world_state=privileged,
                    symbolic_observation=X,
                    dfa_state=q_next,
                    binding=binding,
                    property_name=prop.name,
                )
                results.append(
                    RoboCasaPropertyStatus(
                        property_name=prop.name,
                        binding=binding,
                        accepting=q_next in dfa.F,
                        trap=dfa.is_trap_state(q_next),
                        current_state=q_next,
                        predicate_values=X,
                        product_state=product_state,
                    )
                )
                self._update_history(binding, X)

        self._last_dynamic_info = dynamic_info
        return results

    def _new_dfa(self, prop) -> LTLfDFA:
        if prop.name not in self._dfa_templates:
            self._dfa_templates[prop.name] = LTLfDFA(prop.ltldfa._formula)
        return self._dfa_templates[prop.name]

    def _dfa_key(self, property_name: str, binding: Dict[str, str]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        return (property_name, tuple(sorted(binding.items())))

    def _get_initial_world_state(self, env) -> Dict[str, object]:
        if hasattr(env, "get_initial_state"):
            state = env.get_initial_state()
            if state is not None:
                return state
        if hasattr(env, "get_privileged_information"):
            return env.get_privileged_information()
        raise ValueError("No initial world state available")

    def _transition_world_state(self, env, current_world_state, action):
        if hasattr(env, "step") and action is not None:
            try:
                next_state = env.step(current_world_state, action)
            except TypeError:
                next_state = env.step(action)
            if next_state is not None:
                return next_state
        if hasattr(env, "get_privileged_information"):
            return env.get_privileged_information()
        raise ValueError("Environment transition failed")

    def _is_action_applicable(self, env, current_world_state, action) -> bool:
        if hasattr(env, "is_action_applicable"):
            return bool(env.is_action_applicable(current_world_state, action))
        return True

    def _enumerate_bindings(
        self,
        prop,
        static_info: Dict[str, object],
        dynamic_info: Dict[str, object],
    ) -> Iterable[Dict[str, str]]:
        used_entities = self._used_entities(prop)
        if not used_entities or self._uses_embedded_predicates(prop, dynamic_info):
            return [{}]

        static_scene = (static_info.get("scene_layout") or static_info.get("scene") or {})
        dynamic_scene = (dynamic_info.get("scene") or {})
        object_names = list(((static_scene.get("objects")) or (dynamic_scene.get("objects")) or {}).keys())
        fixture_names = list(((static_scene.get("fixtures")) or (dynamic_scene.get("fixtures")) or {}).keys())
        privileged = dict(static=static_info, dynamic=dynamic_info)
        support_names = self._support_candidates(object_names, fixture_names, privileged=privileged)
        candidates = {
            "object": object_names,
            "support": support_names,
            "fixture": fixture_names,
            "target_object": object_names,
        }

        names = [entity.name for entity in used_entities]
        value_lists = [candidates.get(name, object_names + fixture_names) for name in names]
        products = itertools.product(*value_lists)
        bindings = []
        for values in products:
            binding = dict(zip(names, values))
            non_null_values = [value for value in binding.values() if value is not None]
            if len(set(non_null_values)) != len(non_null_values):
                continue
            bindings.append(binding)
        return bindings or [{}]

    def _uses_embedded_predicates(self, prop, dynamic_info: Dict[str, object]) -> bool:
        """Return true when the simulator snapshot already grounded this LTL.

        The current RoboCasa pipeline computes task/stage-specific roles inside
        ``robocasa_sim/.../predicates.py`` and stores the resulting atomic
        propositions under ``dynamic.predicates.sections.predicates``. In that
        case, enumerating symbolic entity bindings would only duplicate the same
        global/stage-local truth values across many fake object bindings.
        """
        predicate_section = (
            ((dynamic_info.get("predicates") or {}).get("sections") or {}).get("predicates")
            or {}
        )
        if not predicate_section:
            return False
        return all(symbol in predicate_section for symbol in prop.predicates.keys())

    def _support_candidates(
        self,
        object_names: Sequence[str],
        fixture_names: Sequence[str],
        privileged: Dict[str, object],
    ) -> List[str]:
        support_names = list(fixture_names)
        for name in object_names:
            if P.entity_has_attribute(name, "receptacle", privileged=privileged):
                support_names.append(name)
        return support_names or list(object_names) or list(fixture_names)

    def _used_entities(self, prop) -> List[SymbolicEntity]:
        used = {}
        for entities in prop.symbol_to_entities.values():
            for entity in entities:
                used[entity.name] = entity
        return [used[name] for name in sorted(used.keys())]

    def _update_history(self, binding: Dict[str, str], predicate_values: Dict[str, bool]) -> None:
        if "object" in binding:
            object_name = binding["object"]
            if "object_secured" in predicate_values:
                self._last_values[f"grasped::{object_name}"] = bool(predicate_values["object_secured"])
            elif "postconditions_satisfied_pick" in predicate_values:
                self._last_values[f"grasped::{object_name}"] = bool(predicate_values["postconditions_satisfied_pick"])
            elif "skill_pick_active" in predicate_values:
                self._last_values.setdefault(f"grasped::{object_name}", False)

    def _evaluate_partial(
        self,
        predicate,
        *,
        env,
        privileged,
        static_info,
        dynamic_info,
        binding: Dict[str, str],
    ):
        params = []
        for arg in predicate.args:
            if isinstance(arg, partial) or (hasattr(arg, "func") and hasattr(arg, "args")):
                params.append(
                    self._evaluate_partial(
                        arg,
                        env=env,
                        privileged=privileged,
                        static_info=static_info,
                        dynamic_info=dynamic_info,
                        binding=binding,
                    )
                )
            elif hasattr(arg, "name"):
                params.append(binding.get(arg.name))
            else:
                params.append(arg)
        return predicate.func(
            *params,
            env=env,
            privileged=privileged,
            static_info=static_info,
            dynamic_info=dynamic_info,
            last_dynamic_info=self._last_dynamic_info,
            last_values=self._last_values,
            bindings=binding,
            **(predicate.keywords or {}),
        )
