from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from monitor.LTLfDFA import LTLfDFA

from monitor.symbolic_properties import build_all_properties


@dataclass
class ProductState:
    world_state: Dict[str, object]
    symbolic_observation: Dict[str, bool]
    dfa_state: str
    binding: Dict[str, str]
    property_name: str


@dataclass
class TraceResult:
    satisfied: bool
    reason: str
    trace: List[ProductState]
    final_product_state: Optional[ProductState] = None
    failed_step: Optional[int] = None
    failed_action: Optional[object] = None


@dataclass
class PropertyStatus:
    property_name: str
    binding: Dict[str, str]
    accepting: bool
    trap: bool
    current_state: str
    predicate_values: Dict[str, bool]
    product_state: ProductState


class SymbolicMonitor:
    """
    Direct symbolic monitor for simulator trajectories.

    Simulator-agnostic: operates on a generic `privileged`/`world_state` dict
    shape (`{"static": ..., "dynamic": ...}`) and a duck-typed `env`
    (`get_privileged_information()`, `step()`, `is_action_applicable()`), plus
    whatever `properties` list is passed in (or, by default,
    `symbolic_properties.build_all_properties()`, which is robocasa-specific
    -- pass an explicit `properties` list to use this with a different
    simulator's predicates/specs).

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
            # predicate is now a plain, directly-callable function (see
            # predicates.py) -- no per-role currying/deferral to unwrap, since
            # there's no composition here to defer, only a single already-
            # computed lookup by name (unlike SceneFlowLang's own SG_Primitives,
            # which composes several graph operations lazily via partial).
            values[symbol] = bool(
                predicate(
                    env=env,
                    privileged=world_state,
                    static_info=static_info,
                    dynamic_info=dynamic_info,
                    last_dynamic_info=self._last_dynamic_info,
                    last_values=self._last_values,
                    bindings=binding,
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
    ) -> ProductState:
        if world_state is None:
            if env is None:
                raise ValueError("Either `env` or `world_state` must be provided.")
            world_state = self._get_initial_world_state(env)
        X0 = self.alpha(prop, world_state, binding, env=env)
        dfa = self._new_dfa(prop)
        q1 = dfa.delta(dfa.q0, X0)
        return ProductState(
            world_state=world_state,
            symbolic_observation=X0,
            dfa_state=q1,
            binding=binding,
            property_name=prop.name,
        )

    def step_product_state(
        self,
        prop,
        product_state: ProductState,
        *,
        env=None,
        next_world_state=None,
        action=None,
    ) -> ProductState:
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
        return ProductState(
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
    ) -> TraceResult:
        try:
            Z = self.build_initial_product_state(prop, binding, env=env)
        except Exception as exc:
            return TraceResult(
                satisfied=False,
                reason=f"No initial world state available: {exc}",
                trace=[],
            )

        product_trace = [Z]
        dfa = self._new_dfa(prop)

        for i, action in enumerate(skill_sequence):
            if not self._is_action_applicable(env, Z.world_state, action):
                return TraceResult(
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
                return TraceResult(
                    satisfied=False,
                    reason=f"Environment transition failed at step {i}: {exc}",
                    failed_step=i,
                    failed_action=action,
                    trace=product_trace,
                    final_product_state=Z,
                )

            product_trace.append(Z_next)

            if Z_next.dfa_state in getattr(dfa, "rejecting_states", set()):
                return TraceResult(
                    satisfied=False,
                    reason=f"Entered rejecting DFA state at step {i}",
                    failed_step=i,
                    failed_action=action,
                    trace=product_trace,
                    final_product_state=Z_next,
                )

            Z = Z_next

        terminal_state = dfa.terminal_delta(Z.dfa_state)
        Z = ProductState(
            world_state=Z.world_state,
            symbolic_observation=Z.symbolic_observation,
            dfa_state=terminal_state,
            binding=Z.binding,
            property_name=Z.property_name,
        )

        if terminal_state in dfa.F:
            return TraceResult(
                satisfied=True,
                reason="Final DFA state is accepting",
                trace=product_trace,
                final_product_state=Z,
            )
        return TraceResult(
            satisfied=False,
            reason="Final DFA state is rejecting at episode end",
            trace=product_trace,
            final_product_state=Z,
        )

    def step(self, env=None, privileged=None) -> List[PropertyStatus]:
        if privileged is None:
            if env is None:
                raise ValueError("Either `env` or `privileged` must be provided.")
            privileged = env.get_privileged_information()

        dynamic_info = privileged.get("dynamic") or {}
        results: List[PropertyStatus] = []

        # Every property is evaluated against a single empty binding -- the
        # sim/robocasa/predicates.py pipeline already grounds each frame's
        # atomic propositions to one global/stage-local scalar per predicate
        # name (see predicates.py's _predicate_value), so there is no real
        # per-object/per-fixture binding to enumerate here. Real multi-entity
        # binding enumeration (once genuinely needed) was removed here
        # 2026-09-07 -- see monitor.py's git history / SymbolicEntity.py's
        # removal for the dead code this replaced.
        for prop in self.properties:
            binding: Dict[str, str] = {}
            dfa = self._new_dfa(prop)
            dfa_key = self._dfa_key(prop.name, binding)
            X = self.alpha(prop, privileged, binding, env=env)
            q_curr = self._dfa_states.get(dfa_key, dfa.q0)
            q_next = dfa.delta(q_curr, X)
            self._dfa_states[dfa_key] = q_next
            product_state = ProductState(
                world_state=privileged,
                symbolic_observation=X,
                dfa_state=q_next,
                binding=binding,
                property_name=prop.name,
            )
            results.append(
                PropertyStatus(
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

    def _update_history(self, binding: Dict[str, str], predicate_values: Dict[str, bool]) -> None:
        if "object" in binding:
            object_name = binding["object"]
            if "object_secured" in predicate_values:
                self._last_values[f"grasped::{object_name}"] = bool(predicate_values["object_secured"])
            elif "postconditions_satisfied_pick" in predicate_values:
                self._last_values[f"grasped::{object_name}"] = bool(predicate_values["postconditions_satisfied_pick"])
            elif "skill_pick_active" in predicate_values:
                self._last_values.setdefault(f"grasped::{object_name}", False)

