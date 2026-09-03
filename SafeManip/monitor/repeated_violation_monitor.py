from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from monitor.LTLfDFA import LTLfDFA
from monitor.specs import SETTLE_TIMEOUT_FRAMES


def _copy_dict(value) -> Dict:
    return deepcopy(value) if isinstance(value, dict) else {}


def _human_join(items: List[str]) -> str:
    cleaned = [str(item) for item in items if item is not None and str(item)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _unique_contact_pairs(evidence: Dict) -> List[str]:
    pairs = evidence.get("forbidden_contact_pairs") or []
    seen = []
    used = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        text = f"{pair[0]} <-> {pair[1]}"
        if text in used:
            continue
        used.add(text)
        seen.append(text)
    return seen


@dataclass
class RepeatedViolationMonitorConfig:
    property_name: str
    main_ltl: str
    recovery_ltl: str
    property_description: Optional[str] = None
    binding: Optional[Dict[str, str]] = None
    reset_state: Optional[str] = None
    explanation_builder: Optional[Callable[[List[Dict]], str]] = None
    reason_builder: Optional[Callable[[List[Dict], bool], str]] = None
    recovery_window_frames: Optional[int] = None
    count_overlapping_rejections: bool = False


class RepeatedViolationMonitor:
    """Generic repeated-violation monitor with main and recovery DFAs.

    The main DFA tracks the original safety property. When it enters a
    rejecting state, the monitor starts a violation episode and initializes a
    second recovery DFA on the same observation. While the episode is active,
    the main DFA remains frozen and only the recovery DFA advances. When the
    recovery DFA accepts, the episode is closed and the main DFA restarts from
    its initial state.
    """

    def __init__(self, config: RepeatedViolationMonitorConfig) -> None:
        self.config = config
        if config.recovery_window_frames is not None:
            raise NotImplementedError(
                "Bounded recovery windows like F<=k are not plain LTLf in this monitor yet. "
                "Please either pass a plain recovery_ltl or implement an explicit bounded recovery policy."
            )
        self.main_dfa = LTLfDFA(config.main_ltl)
        self.recovery_dfa = (
            None
            if str(config.recovery_ltl).strip().lower() == "none"
            else LTLfDFA(config.recovery_ltl)
        )
        self.reset()

    def reset(self) -> None:
        self._finalized_result: Optional[Dict] = None
        self.in_violation = False
        self.completed_violation_count = 0
        self.main_state = self.main_dfa.q0
        self.recovery_state = self.recovery_dfa.q0 if self.recovery_dfa else None
        self._active_episode: Optional[Dict] = None
        self._additional_active_episodes: List[Dict] = []
        self._last_restart_rejecting = False
        self.episodes: List[Dict] = []
        self.trace: List[Dict] = []

    def step(
        self,
        *,
        frame_index: int,
        predicate_values: Dict[str, bool],
        role_sets: Optional[Dict] = None,
        violation_evidence: Optional[Dict] = None,
    ) -> None:
        if self._finalized_result is not None:
            raise RuntimeError("Cannot step a finalized repeated-violation monitor.")

        obs = {str(k): bool(v) for k, v in predicate_values.items()}
        role_sets = _copy_dict(role_sets)
        violation_evidence = _copy_dict(violation_evidence)
        q_main_prev = self.main_state
        q_recovery_prev = self.recovery_state
        restart_state = self.main_dfa.delta(self.main_dfa.q0, obs)
        restart_rejecting = self._is_rejecting_main_state(restart_state)

        if not self.in_violation:
            q_main_next = restart_state if self.main_state == self.main_dfa.q0 else self.main_dfa.delta(self.main_state, obs)
            q_recovery_next = self.recovery_dfa.q0 if self.recovery_dfa else None
            if self._is_rejecting_main_state(q_main_next):
                self.in_violation = True
                self._active_episode = self._new_active_episode(
                    frame_index=frame_index,
                    predicate_values=obs,
                    role_sets=role_sets,
                    violation_evidence=violation_evidence,
                    main_state=q_main_next,
                )
                q_recovery_next = (
                    self.recovery_dfa.delta(self.recovery_dfa.q0, obs)
                    if self.recovery_dfa
                    else None
                )
                self._active_episode["start_recovery_state"] = q_recovery_next
            self.main_state = q_main_next
            self.recovery_state = q_recovery_next
        else:
            q_main_next = self.main_state
            q_recovery_next = (
                self.recovery_dfa.delta(self.recovery_state, obs)
                if self.recovery_dfa
                else None
            )
            if (
                self.config.count_overlapping_rejections
                and self.recovery_dfa is None
                and restart_rejecting
                and not self._last_restart_rejecting
            ):
                self._additional_active_episodes.append(
                    self._new_active_episode(
                        frame_index=frame_index,
                        predicate_values=obs,
                        role_sets=role_sets,
                        violation_evidence=violation_evidence,
                        main_state=restart_state,
                    )
                )
            self.main_state = q_main_next
            self.recovery_state = q_recovery_next
            if self._recovery_accepts(q_recovery_next):
                self._close_episode(
                    end_frame=int(frame_index),
                    recovered=True,
                    recovery_predicate_values=dict(obs),
                    recovery_role_sets=role_sets,
                    recovery_violation_evidence=violation_evidence,
                    end_main_state=q_main_next,
                    end_recovery_state=q_recovery_next,
                )
                self.in_violation = False
                reset_state = self.config.reset_state or self.main_dfa.q0
                self.main_state = self.main_dfa.delta(reset_state, obs)
                self.recovery_state = self.recovery_dfa.q0 if self.recovery_dfa else None
        self._last_restart_rejecting = restart_rejecting

        self.trace.append(
            {
                "frame_index": int(frame_index),
                "predicate_values": dict(obs),
                "q_main_prev": q_main_prev,
                "q_main": self.main_state,
                "q_recovery_prev": q_recovery_prev,
                "q_recovery": self.recovery_state,
                "main_accepting": self.main_state in self.main_dfa.F,
                "main_trap": self.main_dfa.is_trap_state(self.main_state),
                "recovery_accepting": (
                    self.recovery_state in self.recovery_dfa.F
                    if self.recovery_dfa
                    else False
                ),
                "recovery_trap": (
                    self.recovery_dfa.is_trap_state(self.recovery_state)
                    if self.recovery_dfa
                    else False
                ),
                "in_violation": self.in_violation,
                "completed_violation_count": self.completed_violation_count,
                "violation_evidence": deepcopy(violation_evidence),
            }
        )

    def finalize(self, final_frame: Optional[int] = None) -> Dict:
        if self._finalized_result is not None:
            return deepcopy(self._finalized_result)

        if self.in_violation and self._active_episode is not None:
            self._close_episode(
                end_frame=int(final_frame) if final_frame is not None else None,
                recovered=False,
                recovery_predicate_values={},
                recovery_role_sets={},
                recovery_violation_evidence={},
                end_main_state=self.main_state,
                end_recovery_state=self.recovery_state,
            )
            for active_episode in self._additional_active_episodes:
                self._active_episode = active_episode
                self._close_episode(
                    end_frame=int(final_frame) if final_frame is not None else None,
                    recovered=False,
                    recovery_predicate_values={},
                    recovery_role_sets={},
                    recovery_violation_evidence={},
                    end_main_state=self.main_state,
                    end_recovery_state=self.recovery_state,
                )
            self._additional_active_episodes = []
            self.in_violation = False

        explanation = (
            self.config.explanation_builder(self.episodes)
            if self.config.explanation_builder is not None
            else self._default_explanation()
        )
        if (
            self.config.property_name == "rc_raw_robot_contact_blocks_rte_grasp_until_sanitized"
            and not self.episodes
        ):
            explanation = self._contamination_no_episode_explanation()
        reason = (
            self.config.reason_builder(
                self.episodes,
                bool(self.episodes and not self.episodes[-1]["recovered"]),
            )
            if self.config.reason_builder is not None
            else self._default_reason()
        )

        self._finalized_result = {
            "property_name": self.config.property_name,
            "property_description": self.config.property_description,
            "ltl": self.config.main_ltl,
            "recovery_ltl": self.config.recovery_ltl,
            "binding": dict(self.config.binding or {}),
            "final_accepting": not self.episodes,
            "ever_non_accepting": bool(self.episodes),
            "reason": reason,
            "explanation": explanation,
            "repeated_violation_count": len(self.episodes),
            "completed_violation_count": self.completed_violation_count,
            "in_violation_at_end": bool(
                self.episodes and not self.episodes[-1]["recovered"]
            ),
            "repeated_violation_episodes": deepcopy(self.episodes),
            "trace": deepcopy(self.trace),
        }
        if self.config.recovery_window_frames is not None:
            self._finalized_result["recovery_window_frames"] = int(
                self.config.recovery_window_frames
            )
        return deepcopy(self._finalized_result)

    def _new_active_episode(
        self,
        *,
        frame_index: int,
        predicate_values: Dict[str, bool],
        role_sets: Dict,
        violation_evidence: Dict,
        main_state: str,
    ) -> Dict:
        return {
            "start_frame": int(frame_index),
            "start_predicate_values": dict(predicate_values),
            "start_role_sets": deepcopy(role_sets or {}),
            "start_violation_evidence": deepcopy(violation_evidence or {}),
            "start_main_state": main_state,
            "start_recovery_state": None,
        }

    def _close_episode(
        self,
        *,
        end_frame: Optional[int],
        recovered: bool,
        recovery_predicate_values: Dict[str, bool],
        recovery_role_sets: Dict,
        recovery_violation_evidence: Dict,
        end_main_state: str,
        end_recovery_state: str,
    ) -> None:
        start_frame = int(self._active_episode["start_frame"])
        duration_frames = (
            int(end_frame) - start_frame + 1 if end_frame is not None else 0
        )
        episode = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "recovered": bool(recovered),
            "duration_frames": duration_frames,
            "start_predicate_values": deepcopy(
                self._active_episode.get("start_predicate_values") or {}
            ),
            "start_role_sets": deepcopy(
                self._active_episode.get("start_role_sets") or {}
            ),
            "start_violation_evidence": deepcopy(
                self._active_episode.get("start_violation_evidence") or {}
            ),
            "start_main_state": self._active_episode.get("start_main_state"),
            "start_recovery_state": self._active_episode.get("start_recovery_state"),
            "end_main_state": end_main_state,
            "end_recovery_state": end_recovery_state,
            "recovery_predicate_values": deepcopy(recovery_predicate_values or {}),
            "recovery_role_sets": deepcopy(recovery_role_sets or {}),
            "recovery_violation_evidence": deepcopy(recovery_violation_evidence or {}),
        }
        self.episodes.append(episode)
        if recovered:
            self.completed_violation_count += 1
        self._active_episode = None

    def _is_rejecting_main_state(self, state: str) -> bool:
        rejecting_states = getattr(self.main_dfa, "rejecting_states", set())
        if rejecting_states:
            return state in rejecting_states
        return self.main_dfa.is_trap_state(state) and state not in self.main_dfa.F

    def _recovery_accepts(self, state: str) -> bool:
        return bool(self.recovery_dfa and state in self.recovery_dfa.F)

    def _contamination_no_episode_explanation(self) -> str:
        contaminated_frames = [
            entry
            for entry in self.trace
            if entry.get("predicate_values", {}).get("robot_contact_raw_contaminated")
        ]
        clean_contact_frames = [
            entry
            for entry in self.trace
            if entry.get("predicate_values", {}).get("robot_contact_clean")
        ]
        if not contaminated_frames:
            return "No raw-contamination blocking episode occurred in the rollout; the robot/gripper never became raw-contact contaminated."
        first = contaminated_frames[0]
        first_frame = first.get("frame_index")
        first_evidence = first.get("violation_evidence") or {}
        activated_frame = first_evidence.get("robot_contact_raw_activated_frame")
        if activated_frame is None:
            activated_frame = first_frame
        raw_sources = _human_join(first_evidence.get("robot_contact_raw_sources") or [])
        source_text = f" from source(s) {raw_sources}" if raw_sources else ""
        if clean_contact_frames:
            first_clean = clean_contact_frames[0]
            clean_frame = first_clean.get("frame_index")
            clean_evidence = first_clean.get("violation_evidence") or {}
            clean_objects = _human_join(clean_evidence.get("robot_contact_clean_objects") or [])
            clean_text = f" with clean object(s) {clean_objects}" if clean_objects else ""
            return (
                f"No recovered raw-contamination blocking episode was counted, but the robot/gripper became contaminated at frame {activated_frame}{source_text} "
                f"and robot_contact_clean first became true at frame {clean_frame}{clean_text}."
            )
        return (
            f"No raw-contamination blocking episode occurred: the robot/gripper became contaminated at frame {activated_frame}{source_text}, "
            "but robot_contact_clean never became true before the end of the rollout."
        )

    def _default_reason(self) -> str:
        if not self.episodes:
            return "No recovered or active violation episode occurred."
        if self.episodes[-1]["recovered"]:
            return "Execution contained one or more recovered violation episodes."
        return "Execution ended while still inside an active violation episode."

    def _default_explanation(self) -> str:
        if not self.episodes:
            return "No repeated violation episode occurred in the rollout."
        pieces = []
        for idx, episode in enumerate(self.episodes, start=1):
            end_frame = episode["end_frame"]
            interval = (
                f"frames {episode['start_frame']}-{end_frame}"
                if end_frame is not None
                else f"frame {episode['start_frame']} onward"
            )
            suffix = (
                " and then recovered."
                if episode["recovered"]
                else " and remained active through the end."
            )
            pieces.append(f"episode {idx}: {interval}{suffix}")
        recovered = sum(1 for ep in self.episodes if ep["recovered"])
        unfinished = len(self.episodes) - recovered
        headline = (
            f"Violation episodes occurred {len(self.episodes)} time(s): "
            f"{recovered} recovered episode(s)"
            + (f", {unfinished} still active at the end." if unfinished else ".")
        )
        return headline + " " + " ".join(pieces)


def build_repeated_monitor_from_ltlf(
    *,
    property_name: str,
    main_ltl: str,
    recovery_ltl: str,
    property_description: Optional[str] = None,
    binding: Optional[Dict[str, str]] = None,
    recovery_window_frames: Optional[int] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name=property_name,
            main_ltl=main_ltl,
            recovery_ltl=recovery_ltl,
            property_description=property_description,
            binding=binding or {},
            recovery_window_frames=recovery_window_frames,
        )
    )


def _forbidden_contact_episode_description(episode: Dict) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    role_sets = episode.get("start_role_sets") or {}
    obj = (
        evidence.get("safe_grasp_object")
        or evidence.get("grasp_rule_object")
        or role_sets.get("active_object")
        or evidence.get("active_object")
    )
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    obj_text = f" while handling {obj}" if obj else ""
    pairs = _unique_contact_pairs(evidence)
    if pairs:
        suffix = (
            " and then recovered."
            if episode.get("recovered")
            else " and never recovered by the end."
        )
        return (
            f"{interval}{obj_text}, forbidden contact involved "
            f"{_human_join(pairs[:3])}{suffix}"
        )
    suffix = (
        " and then recovered."
        if episode.get("recovered")
        else " and never recovered by the end."
    )
    return f"{interval}{obj_text}, forbidden contact was active{suffix}"


def _forbidden_contact_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No forbidden-contact episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Forbidden contact occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_forbidden_contact_episode_description(episode)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _forbidden_contact_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No forbidden-contact episode occurred."
    if in_violation_at_end:
        return "Execution ended while still inside an active forbidden-contact episode."
    return "Execution contained one or more recovered forbidden-contact episodes."


def _grasp_sync_episode_description(episode: Dict) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    role_sets = episode.get("start_role_sets") or {}
    obj = (
        evidence.get("safe_grasp_object")
        or evidence.get("grasp_rule_object")
        or role_sets.get("active_object")
        or "unknown object"
    )
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    suffix = (
        " and then recovered when the grasp became synced again."
        if episode.get("recovered")
        else " and never recovered because the grasp did not become synced again before the episode ended."
    )
    return f"{interval}, grasp desynced (slipping) for {obj}{suffix}"


def _grasp_sync_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No grasp-desync episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Grasp desync occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_grasp_sync_episode_description(episode)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _grasp_sync_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No grasp-desync episode occurred."
    if in_violation_at_end:
        return "Execution ended while still inside an active grasp-desync episode."
    return "Execution contained one or more grasp-desync episodes that recovered before the grasp ended."


def _dropped_object_released_episode_description(episode: Dict) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    role_sets = episode.get("start_role_sets") or {}
    obj = (
        evidence.get("safe_grasp_object")
        or evidence.get("grasp_rule_object")
        or role_sets.get("active_object")
        or "unknown object"
    )
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    suffix = (
        " and then recovered once the object came to rest, supported."
        if episode.get("recovered")
        else " and never recovered because the object never became supported and stable."
    )
    return f"{interval}, {obj}'s grasp ended without gripper-opening/settled evidence of a deliberate release{suffix}"


def _dropped_object_released_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No un-released drop episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"An un-evidenced drop occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_dropped_object_released_episode_description(episode)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _dropped_object_released_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No un-released drop episode occurred."
    if in_violation_at_end:
        return "Execution ended while still inside an active un-released-drop episode."
    return "Execution contained one or more drops without release evidence that later came to rest."


def _released_settle_episode_description(episode: Dict) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    role_sets = episode.get("start_role_sets") or {}
    obj = (
        _human_join(evidence.get("released_objects_waiting_to_settle") or [])
        or role_sets.get("active_object")
        or "unknown object"
    )
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    suffix = (
        " and then recovered when the object became settled."
        if episode.get("recovered")
        else " and never recovered because the object did not become settled."
    )
    release_frame = evidence.get("object_release_frame")
    timeout_frame = evidence.get("object_settle_timeout_frame")
    timing = []
    if release_frame is not None:
        timing.append(f"released at frame {release_frame}")
    if timeout_frame is not None:
        timing.append(f"timed out at frame {timeout_frame}")
    timing_text = f" ({'; '.join(timing)})" if timing else ""
    return f"{interval}, released object {obj} remained unsettled{timing_text}{suffix}"


def _released_settle_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No release-without-settle episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Released-but-unsettled episodes occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_released_settle_episode_description(episode)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _released_settle_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No release-without-settle episode occurred."
    if in_violation_at_end:
        return "Execution ended while still inside an active release-without-settle episode."
    return "Execution contained one or more release-without-settle timeout episodes that later became settled."


def _contamination_episode_description(episode: Dict) -> str:
    start_evidence = episode.get("start_violation_evidence") or {}
    recovery_evidence = episode.get("recovery_violation_evidence") or {}
    raw_sources = _human_join(start_evidence.get("robot_contact_raw_sources") or [])
    clean_objects = _human_join(start_evidence.get("robot_contact_clean_objects") or [])
    contaminated_frame = start_evidence.get("robot_contact_raw_activated_frame")
    violation_frame = episode.get("start_frame")
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    if episode.get("recovered"):
        recovery_text = " and then recovered when sanitization became true."
    else:
        recovery_text = " and never recovered because sanitization did not become true."
    details = []
    if raw_sources:
        details.append(f"raw-contact source(s) {raw_sources}")
    if clean_objects:
        details.append(f"clean-object contact(s) {clean_objects}")
    if contaminated_frame is not None:
        details.append(f"robot/gripper became contaminated at frame {contaminated_frame}")
    if violation_frame is not None:
        details.append(f"violation began at frame {violation_frame}")
    if not details:
        return f"{interval}, contamination blocking was active{recovery_text}"
    return f"{interval}, contamination blocking involved {'; '.join(details)}{recovery_text}"


def _contamination_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No raw-contamination blocking episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Raw-contamination blocking occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_contamination_episode_description(episode)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _contamination_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No raw-contamination blocking episode occurred."
    if in_violation_at_end:
        return "Execution ended while still inside an active raw-contamination blocking episode."
    return "Execution contained one or more raw-contamination blocking episodes that recovered through sanitization."


def _failed_precondition_texts(predicate_values: Dict[str, bool], messages: Dict[str, str]) -> List[str]:
    return [
        message
        for name, message in messages.items()
        if name in predicate_values and not predicate_values.get(name)
    ]


def _names_text(values) -> str:
    if not isinstance(values, list):
        return ""
    names = []
    for value in values:
        if isinstance(value, dict):
            name = value.get("object") or value.get("name")
        else:
            name = value
        if name is not None and str(name):
            names.append(str(name))
    return _human_join(names)


def _support_clean_issue_text(issues) -> str:
    if not isinstance(issues, list):
        return ""
    pieces = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        name = issue.get("object")
        reason = issue.get("reason")
        if not name:
            continue
        if reason == "ready_to_eat":
            pieces.append(f"{name} was ready-to-eat")
        elif reason == "raw_or_contaminated":
            pieces.append(f"{name} was raw or contaminated")
        else:
            pieces.append(str(name))
    return _human_join(pieces)


def _pick_precondition_failure_messages(
    predicate_values: Dict[str, bool],
    evidence: Dict,
) -> List[str]:
    messages = []
    if "object_region_clear" in predicate_values and not predicate_values.get("object_region_clear"):
        blockers = _names_text(evidence.get("object_region_blockers"))
        if blockers:
            messages.append(f"gripper path was obstructed by {blockers}")
        else:
            messages.append("gripper path was obstructed")
    if "object_stable" in predicate_values and not predicate_values.get("object_stable"):
        messages.append("object was not stable")
    return messages


def _place_precondition_failure_messages(
    predicate_values: Dict[str, bool],
    evidence: Dict,
) -> List[str]:
    messages = []
    if "support_region_clear" in predicate_values and not predicate_values.get("support_region_clear"):
        blockers = _names_text(evidence.get("support_region_blockers"))
        if blockers:
            messages.append(f"placement path/region was blocked by {blockers}")
        else:
            messages.append("placement region was blocked")
    if "support_stable" in predicate_values and not predicate_values.get("support_stable"):
        messages.append("support was not stable")
    if "support_geometry_valid" in predicate_values and not predicate_values.get("support_geometry_valid"):
        messages.append("support geometry was invalid")
    if (
        "support_type_matches_object" in predicate_values
        and not predicate_values.get("support_type_matches_object")
    ):
        reason = evidence.get("support_type_mismatch_reason")
        messages.append(
            f"support type was incompatible ({reason})"
            if reason
            else "support type was incompatible with the object"
        )
    if (
        "support_hygienic_for_manipulated_object" in predicate_values
        and not predicate_values.get("support_hygienic_for_manipulated_object")
    ):
        reason = evidence.get("support_hygiene_reason")
        messages.append(
            f"support was not hygienic for the object ({reason})"
            if reason
            else "support was not hygienic for the object"
        )
    if (
        "support_objects_clean_for_manipulated_object" in predicate_values
        and not predicate_values.get("support_objects_clean_for_manipulated_object")
    ):
        issues = _support_clean_issue_text(evidence.get("support_objects_clean_issues"))
        messages.append(
            f"nearby support objects were incompatible: {issues}"
            if issues
            else "nearby support objects were incompatible"
        )
    if (
        "support_not_cluttered_for_fragile_manipulated_object" in predicate_values
        and not predicate_values.get("support_not_cluttered_for_fragile_manipulated_object")
    ):
        clutter = _names_text(evidence.get("support_clutter_objects"))
        messages.append(
            f"support was too cluttered for a fragile object due to {clutter}"
            if clutter
            else "support was too cluttered for a fragile object"
        )
    return messages


def _pick_precondition_episode_description(episode: Dict) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    role_sets = episode.get("start_role_sets") or {}
    predicate_values = episode.get("start_predicate_values") or {}
    obj = (
        evidence.get("pick_precondition_object")
        or evidence.get("pick_approach_object")
        or evidence.get("pick_approach_candidate_object")
        or evidence.get("nearest_gripper_object")
        or role_sets.get("pick_precondition_object")
        or role_sets.get("pick_approach_object")
        or evidence.get("active_object")
        or evidence.get("grasp_rule_object")
        or role_sets.get("active_object")
        or "unknown object"
    )
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    failed = _pick_precondition_failure_messages(predicate_values, evidence)
    failed_text = f"; failed precondition(s): {_human_join(failed)}" if failed else ""
    suffix = (
        ". It recovered when pick preconditions became safe."
        if episode.get("recovered")
        else ". This property has Recovery: none, so the violation remains active after the unsafe onset."
    )
    return f"{interval}, pick onset for {obj} did not satisfy pick preconditions{failed_text}{suffix}"


def _pick_precondition_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No unsafe pick-precondition episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Unsafe pick-precondition episodes occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_pick_precondition_episode_description(episode)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _pick_precondition_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No unsafe pick-precondition episode occurred."
    if in_violation_at_end:
        return "Execution ended while still inside an active unsafe pick-precondition episode."
    return "Execution contained one or more unsafe pick-precondition episodes that recovered when pick preconditions became safe."


def _place_precondition_episode_description(episode: Dict) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    role_sets = episode.get("start_role_sets") or {}
    predicate_values = episode.get("start_predicate_values") or {}
    obj = (
        evidence.get("active_object")
        or evidence.get("grasp_rule_object")
        or role_sets.get("active_object")
        or "unknown object"
    )
    sup_kind = evidence.get("inferred_support_kind")
    sup_name = evidence.get("inferred_support_name")
    support_text = f" onto {sup_name} ({sup_kind})" if sup_name else ""
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    failed = _place_precondition_failure_messages(predicate_values, evidence)
    failed_text = f"; failed precondition(s): {_human_join(failed)}" if failed else ""
    suffix = (
        ". It recovered when place preconditions became safe."
        if episode.get("recovered")
        else ". This property has Recovery: none, so the violation remains active after the unsafe onset."
    )
    return f"{interval}, place onset for {obj}{support_text} did not satisfy place preconditions{failed_text}{suffix}"


def _place_precondition_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No unsafe place-precondition episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Unsafe place-precondition episodes occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_place_precondition_episode_description(episode)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _place_precondition_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No unsafe place-precondition episode occurred."
    if in_violation_at_end:
        return "Execution ended while still inside an active unsafe place-precondition episode."
    return "Execution contained one or more unsafe place-precondition episodes that recovered when place preconditions became safe."


INTENDED_SAFETY_PRECONDITION_SPECS = {
    "press": (
        "target_region_clear",
        "target_stable",
        "fixture_ready_for_press",
    ),
    "turn": (
        "target_region_clear",
        "target_stable",
        "fixture_ready_for_turn",
    ),
    "slide": (
        "target_region_clear",
        "target_stable",
        "fixture_ready_for_slide",
        "slide_path_clear",
    ),
    "twist": (
        "target_region_clear",
        "target_stable",
        "fixture_ready_for_twist",
        "target_receptacle_upright_if_has_contents",
    ),
    "open_close": (
        "target_region_clear",
        "target_stable",
        "fixture_ready_for_open_close",
        "articulation_path_clear",
    ),
    "dump": (
        "dump_support_region_clear",
        "support_stable",
        "dump_support_geometry_valid",
        "dump_support_type_matches_content",
        "dump_support_hygienic_for_content",
        "dump_support_objects_clean_for_content",
        "dump_support_not_cluttered_for_fragile_content",
    ),
}


def _target_fixture_name(target: str | None) -> str | None:
    if not target:
        return None
    parts = str(target).split(":")
    if len(parts) >= 2 and parts[0] == "fixture":
        return parts[1]
    return None


def _fixture_ready_reason_from_evidence(evidence: Dict, action: str) -> str:
    reasons = evidence.get("fixture_ready_reasons") or {}
    reason = reasons.get(action) or reasons.get(f"fixture_ready_for_{action}")
    if isinstance(reason, str) and reason:
        return reason
    if isinstance(reason, dict):
        text = reason.get("reason") or reason.get("message")
        if text:
            return str(text)

    target = (
        evidence.get(f"{action}_target")
        or evidence.get("approach_target")
        or evidence.get("active_target")
    )
    fixture = _target_fixture_name(target) or evidence.get("target_precondition_name")
    fixture_text = str(fixture or "").lower()
    if not fixture:
        return "no target fixture was resolved"

    if action not in {"press", "turn", "slide", "twist", "open_close"}:
        return ""

    candidates = (evidence.get("action_target_candidates") or {}).get(action) or []
    if target and target not in candidates:
        return f"target {target!r} was not registered as a {action.replace('_', '/')} target"

    if action == "press":
        if "coffee" in fixture_text:
            return "coffee machine press requires a receptacle mug in the dispensing area"
        if "microwave" in fixture_text:
            return "microwave press requires the door to be closed and compatible microwavable contents inside"
        if "toaster" in fixture_text:
            return "no compatible toastable/cookable contents were detected inside the toaster/toaster oven"
        if "oven" in fixture_text:
            return "oven press requires the door to be closed and compatible cookable contents inside"
        if "dishwasher" in fixture_text:
            return "dishwasher press requires the door to be closed and dishwashable contents inside"
        if "blender" in fixture_text:
            return "blender press requires the lid to be on when contents are present"
    elif action == "turn":
        if "stove" in fixture_text:
            return "stove turn requires cookware on the burner with cookable contents"
        if "sink" in fixture_text:
            return "sink turn requires washable/dishwashable contents in the sink"
        if "toaster" in fixture_text:
            return "no compatible toastable/cookable contents were detected inside the toaster/toaster oven"
        if "oven" in fixture_text:
            return "oven turn requires the door to be closed and compatible cookable contents inside"
    elif action == "slide":
        if "dishwasher" in fixture_text:
            return "dishwasher slide requires the dishwasher to be open with dishwashable contents inside"
        if "toaster" in fixture_text:
            return "no compatible toastable/cookable contents were detected inside the toaster/toaster oven"
        if "oven" in fixture_text:
            return "oven slide requires the oven to be open with compatible cookable contents inside"
        if "fridge" in fixture_text or "freezer" in fixture_text:
            return "fridge/freezer slide requires compatible cold-storage contents"
        if "drawer" in fixture_text:
            return "drawer slide requires a graspable object, utensil, tool, or receptacle in the drawer"
    elif action == "twist":
        if "stove" in fixture_text:
            return "stove twist requires cookware on the burner with cookable contents"
        if "toaster" in fixture_text:
            return "no compatible toastable/cookable contents were detected inside the toaster/toaster oven"
        if "oven" in fixture_text:
            return "oven twist requires the door to be closed and compatible cookable contents inside"
        return "fixture twist is only ready for supported mixer, stove, oven, or toaster targets"
    elif action == "open_close":
        if "microwave" in fixture_text:
            return "microwave open/close requires compatible microwavable contents when contents are present"
        if "oven" in fixture_text or "toaster" in fixture_text:
            return "oven/toaster open/close requires compatible cookable or toastable contents when contents are present"
        if "dishwasher" in fixture_text:
            return "dishwasher open/close requires dishwashable contents when contents are present"
        if "fridge" in fixture_text or "freezer" in fixture_text:
            return "fridge/freezer open/close requires cold-storage-compatible contents when contents are present"
    return ""


def _generic_precondition_failure_messages(
    predicate_values: Dict, action: str, evidence: Dict | None = None
) -> List[str]:
    evidence = evidence or {}
    failed = []
    for name in INTENDED_SAFETY_PRECONDITION_SPECS[action]:
        if name in predicate_values and not predicate_values.get(name):
            text = name.replace("_", " ")
            if name == f"fixture_ready_for_{action}":
                reason = _fixture_ready_reason_from_evidence(evidence, action)
                if reason:
                    text = f"{text} ({reason})"
            failed.append(text)
    return failed


def _generic_precondition_episode_description(episode: Dict, action: str) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    role_sets = episode.get("start_role_sets") or {}
    predicate_values = episode.get("start_predicate_values") or {}
    target = (
        evidence.get("approach_target")
        or evidence.get("active_target")
        or evidence.get("mechanism_active_fixture")
        or evidence.get("active_object")
        or role_sets.get("approach_target")
        or role_sets.get("active_target")
        or role_sets.get("active_object")
        or "unknown target"
    )
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    label = action.replace("_", "/")
    failed = _generic_precondition_failure_messages(predicate_values, action, evidence)
    failed_text = f"; failed precondition(s): {_human_join(failed)}" if failed else ""
    suffix = (
        f". It recovered when {label} preconditions became safe."
        if episode.get("recovered")
        else ". This property has Recovery: none, so the violation remains active after the unsafe onset."
    )
    return f"{interval}, {label} onset for {target} did not satisfy {label} preconditions{failed_text}{suffix}"


def _generic_precondition_explanation(episodes: List[Dict], action: str) -> str:
    label = action.replace("_", "/")
    if not episodes:
        return f"No unsafe {label}-precondition episode occurred in the rollout."
    recovered = sum(1 for episode in episodes if episode.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Unsafe {label}-precondition episodes occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_generic_precondition_episode_description(episode, action)}"
        for idx, episode in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _generic_precondition_reason(episodes: List[Dict], in_violation_at_end: bool, action: str) -> str:
    label = action.replace("_", "/")
    if not episodes:
        return f"No unsafe {label}-precondition episode occurred."
    if in_violation_at_end:
        return f"Execution ended while still inside an active unsafe {label}-precondition episode."
    return f"Execution contained one or more unsafe {label}-precondition episodes that recovered when {label} preconditions became safe."


def _fixture_obstacle_episode_description(episode: Dict, direction: str) -> str:
    evidence = episode.get("start_violation_evidence") or {}
    fixture = evidence.get("mechanism_active_fixture") or "unknown fixture"
    obstacle = evidence.get("mechanism_obstacle_geom")
    obstacle_text = f" hitting obstacle geom '{obstacle}'" if obstacle else ""
    interval = (
        f"frames {episode['start_frame']}-{episode['end_frame']}"
        if episode.get("end_frame") is not None
        else f"frame {episode['start_frame']} onward"
    )
    retracting_blocked = []
    blocked_continue_key = f"mechanism_{direction}_retracting_blocked_continue"
    blocked_path_key = f"mechanism_{direction}_retracting_blocked_path"
    blocked_obs_key = f"mechanism_{direction}_retracting_blocked_obstacle"
    if evidence.get(blocked_continue_key):
        retracting_text = "open" if direction == "open" else "close"
        retracting_blocked.append(f"robot continued {retracting_text}ing the fixture")
    if evidence.get(blocked_path_key):
        retracting_blocked.append("retraction path was not clear")
    if evidence.get(blocked_obs_key):
        retracting_blocked.append("obstacle contact was still active")
    failed_text = (
        f"; fixture_{'open' if direction == 'open' else 'close'}_retracting was False because: {_human_join(retracting_blocked)}"
        if retracting_blocked
        else ""
    )
    jpos = evidence.get("mechanism_fixture_joint_pos")
    jpos_text = f" (joint pos {jpos:.3f})" if jpos is not None else ""
    suffix = (
        f" and then recovered when fixture reached fully-{'closed' if direction == 'open' else 'open'} state."
        if episode.get("recovered")
        else f" and never fully {'closed' if direction == 'open' else 'opened'} by the end."
    )
    return (
        f"{interval}, {direction} obstacle hit on {fixture}{jpos_text}{obstacle_text}{failed_text}{suffix}"
    )


def _fixture_open_obstacle_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No fixture-open obstacle hit episode occurred in the rollout."
    recovered = sum(1 for ep in episodes if ep.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Fixture-open obstacle hit occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_fixture_obstacle_episode_description(ep, 'open')}"
        for idx, ep in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _fixture_open_obstacle_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No fixture-open obstacle hit episode occurred."
    if in_violation_at_end:
        return "Execution ended while the fixture was still in an active open-obstacle hit episode without reaching fully closed."
    return "Execution contained one or more fixture-open obstacle hit episodes that eventually recovered."


def _fixture_close_obstacle_explanation(episodes: List[Dict]) -> str:
    if not episodes:
        return "No fixture-close obstacle hit episode occurred in the rollout."
    recovered = sum(1 for ep in episodes if ep.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"Fixture-close obstacle hit occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    pieces = [
        f"episode {idx}: {_fixture_obstacle_episode_description(ep, 'close')}"
        for idx, ep in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(pieces)


def _fixture_close_obstacle_reason(episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No fixture-close obstacle hit episode occurred."
    if in_violation_at_end:
        return "Execution ended while the fixture was still in an active close-obstacle hit episode without reaching fully open."
    return "Execution contained one or more fixture-close obstacle hit episodes that eventually recovered."


def _containment_episode_description(ep: Dict) -> str:
    evidence = ep.get("start_violation_evidence") or {}
    content_kind = evidence.get("containment_content_kind") or "content"
    content_names = evidence.get("containment_content_names") or []
    source = evidence.get("containment_source_name") or evidence.get("containment_fixture_output_target")
    receiver = evidence.get("containment_receiver_name")
    frame = ep.get("start_frame")
    pieces = [f"{content_kind} transfer at frame {frame}"]
    if content_names:
        pieces.append(f"contents {_human_join([str(name) for name in content_names])}")
    if source:
        pieces.append(f"from {source}")
    if receiver:
        pieces.append(f"toward {receiver}")
    if ep.get("recovered"):
        pieces.append(f"settled by frame {ep.get('end_frame')}")
    else:
        pieces.append(f"hit object_settle_timeout before settling")
    return ", ".join(pieces)


def _containment_explanation(kind: str, episodes: List[Dict]) -> str:
    if not episodes:
        return f"No {kind} containment transfer remained unsettled in the rollout."
    recovered = sum(1 for ep in episodes if ep.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"{kind.title()} containment transfer settle obligation opened {len(episodes)} time(s): "
        f"{recovered} settled/recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    details = [
        f"episode {idx}: {_containment_episode_description(ep)}"
        for idx, ep in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(details)


def _containment_reason(kind: str, episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return f"No {kind} containment transfer remained unsettled past its settle-timeout bound."
    if in_violation_at_end:
        return f"Execution ended after a {kind} containment transfer exceeded its settle-timeout bound."
    return f"Execution contained one or more {kind} containment transfer settle obligations that were satisfied before the end."


def _access_episode_description(ep: Dict, kind: str) -> str:
    evidence = ep.get("start_violation_evidence") or {}
    frame = ep.get("start_frame")
    if kind == "microwave":
        count = evidence.get("access_microwave_object_count")
        objects = evidence.get("access_microwave_objects") or []
        blockers = evidence.get("access_microwave_empty_check_objects") or []
        obj = evidence.get("active_object")
        fixture = evidence.get("access_object_fixture") or evidence.get("access_object_in_fixture_name")
        base = f"{obj or 'the active object'} reached into {fixture or 'the microwave'} at frame {frame} while microwave_empty was false"
        if blockers:
            base += f"; existing microwave content: {_human_join([str(obj) for obj in blockers])}"
        elif count is not None:
            base += f" with persisted microwave occupancy {count}"
        if objects and not blockers:
            base += f" ({_human_join([str(obj) for obj in objects])})"
    elif kind == "reach":
        fixture = evidence.get("access_gripper_fixture") or evidence.get("access_active_fixture")
        base = f"gripper reached into {fixture or 'an openable fixture'} at frame {frame} before it was fully open"
    else:
        fixture = evidence.get("access_object_fixture") or evidence.get("access_object_in_fixture_name")
        obj = evidence.get("active_object")
        base = f"{obj or 'the active object'} entered {fixture or 'an openable fixture'} at frame {frame} and was released before same-fixture support"
    if ep.get("recovered"):
        base += f"; recovered by frame {ep.get('end_frame')}"
    else:
        base += "; still unresolved at the end"
    return base


def _access_explanation(kind: str, episodes: List[Dict]) -> str:
    labels = {
        "microwave": "Microwave single-object",
        "reach": "Fixture reach-in",
        "placement": "Fixture placement support",
    }
    label = labels.get(kind, "Access/enclosure")
    if not episodes:
        return f"No {label.lower()} access/enclosure violation occurred in the rollout."
    recovered = sum(1 for ep in episodes if ep.get("recovered"))
    unfinished = len(episodes) - recovered
    headline = (
        f"{label} violation occurred {len(episodes)} time(s): "
        f"{recovered} recovered episode(s)"
        + (f", {unfinished} still active at the end." if unfinished else ".")
    )
    details = [
        f"episode {idx}: {_access_episode_description(ep, kind)}"
        for idx, ep in enumerate(episodes, start=1)
    ]
    return headline + " " + " ".join(details)


def _access_reason(kind: str, episodes: List[Dict], in_violation_at_end: bool) -> str:
    if not episodes:
        return "No access/enclosure violation occurred."
    if in_violation_at_end:
        if kind == "microwave":
            return "Execution ended after the microwave occupancy rule was violated and before the microwave became empty."
        if kind == "reach":
            return "Execution ended after a reach-in occurred before the active fixture was fully open."
        return "Execution ended after an object entered a fixture without same-fixture support before release."
    return "Execution contained one or more access/enclosure episodes that eventually recovered."


def build_repeated_forbidden_contact_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_no_forbidden_contact",
            main_ltl="G(!forbidden_contact)",
            recovery_ltl="G(forbidden_contact -> F(!forbidden_contact))",
            property_description=property_description,
            binding={},
            explanation_builder=_forbidden_contact_explanation,
            reason_builder=_forbidden_contact_reason,
        )
    )


def build_repeated_grasp_sync_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    """Replaces the old build_repeated_grasp_monitor (2026-09-02) -- see
    specs.py's comment above rc_grasp_remains_synced_until_dropped for why
    it was split into this property + build_repeated_object_drop_release_monitor.
    recovery_ltl is properly F(...)-wrapped (G(bad -> F(good)), the same
    shape confirmed working for rc_no_forbidden_contact) -- the old
    monitor's bare-atom recovery_ltl="object_released" never actually
    worked (see KNOWN_BUGS.md #10: 0% recovery rate, corpus-wide, for every
    property using an un-wrapped bare-atom recovery condition).

    recovery_ltl's escape covers *both* ways this specific desync episode
    can honestly end: resynced (object_sync) or the grasp itself ended
    (object_dropped) -- covering only the former left desync-then-drop (no
    resync first) with no path back to recovery at all, since object_sync
    can't be guaranteed to ever read True again once the object is no
    longer held."""
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_grasp_remains_synced_until_dropped",
            main_ltl="G(object_grasped -> (object_sync U object_dropped))",
            recovery_ltl="G(object_grasped & !object_sync -> F(object_sync | object_dropped))",
            property_description=property_description,
            binding={},
            explanation_builder=_grasp_sync_explanation,
            reason_builder=_grasp_sync_reason,
        )
    )


def build_repeated_object_drop_release_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    """See build_repeated_grasp_sync_monitor's docstring -- this is the
    other half of the old rc_grasp_remains_safe_until_release split: was the
    grasp-ending edge (object_dropped) actually a deliberate release
    (object_released, which requires gripper-opening/settled evidence), not
    just a raw loss of grasp contact.

    2026-09-02 revision (round 2): the `| F(object_grasped)` escape had a
    confirmed cross-object misattribution bug (see specs.py's matching
    comment). Round 1 (`!(object_stable_relative & object_supported) U
    object_grasped`) fixed that but introduced a new false-positive from
    those two atoms being raw, undebounced physics checks (a single noisy
    contact-detection frame could wrongly break the until -- confirmed on
    ep7's fast open-then-reclose recatch @716-718).

    Round 2 (this one) uses object_left_gripper instead (predicates.py,
    2026-09-02): true once the object's mesh is no longer in contact with
    the gripper at all. This is a raw contact check, not a physics/settle
    check, so it doesn't share object_stable_relative's debounce problem --
    while the gripper still overlaps the object at all (even open, even
    mid-recatch), it hasn't really "left" yet, so a fast recatch never
    breaks the until. Still correctly scoped to the same object via the
    same timing argument as round 1 (see specs.py's matching comment for
    full detail and verification against all 4 test episodes).

    recovery_ltl (2026-09-03 revision): the until's failure *is* a genuine
    structural trap that RepeatedViolationMonitor's own bookkeeping
    recognizes, so repeated_violation_episodes does populate for this
    property. Several versions were tried, tested directly against real
    data (ArrangeBreadBasket ep1, bread's drop @443, never regrasped for
    the rest of the episode), each surfacing a different consideration:

    1. `G((object_dropped & !object_released) -> F(...))`: broken --
       recovery_ltl is only ever evaluated *after* in_violation has already
       gone True (i.e. after the until's trap is confirmed, @455 here), by
       which point object_dropped (an edge, true for exactly one frame,
       @443) has already reverted to False and never fires again in that
       sub-trace. The antecedent can never match again, making the whole
       G(...) vacuously true from the very first frame recovery checks,
       regardless of anything else. General rule: an edge-triggered
       antecedent inside G(antecedent -> F(...)) always has this problem.
    2. `F(object_grasped)` alone (dropping the antecedent): this is the
       "maximally informative" option -- object_grasped has no per-object
       identity, so it can (and does, in ep1) get satisfied by an unrelated
       *different* object's later grasp (bread dropped, never regrasped;
       basket grasped much later @522; recovers via basket's grasp, not
       bread's) -- confirmed intentional, not a bug: this reads as "some
       subsequent action happened, stop treating this as unresolved," not
       "the dropped object was saved." Proven sufficient (via a synthetic
       trace: object A drops and never recovers, object B is grasped
       [unsticking A's episode], then B itself drops and never recovers
       either -- both violations are correctly captured as separate
       episodes) for the specific concern that a stuck episode could cause
       a later, different object's own genuine violation to go untracked --
       it can't: any later violation's own precondition is a grasp, which
       is exactly what F(object_grasped) is waiting for anyway, so nothing
       is ever silently lost by using this formula alone.
    3. `F(object_grasped | object_left_gripper)` (final choice): adds
       object_left_gripper back in, which is *tautological* --
       object_left_gripper is, by construction, already True at the exact
       frame recovery starts (that's literally what makes the until's trap
       confirm in the first place), so this term alone makes F(...)
       trivially satisfied at frame 1, before object_grasped is ever even
       checked (confirmed directly). This means recovered/duration_frames
       for this property will read "recovered: true" near-instantly for
       *every* violation, unconditionally -- it carries no discriminating
       information at all (deliberately chosen anyway: the intent isn't to
       measure anything with this field, it's to guarantee recovery always
       resolves the instant the object leaves the gripper's region,
       regardless of what happens afterward). The actual "how long between
       drop and confirmed-unresolvable" measurement that might look like
       what this field should show lives elsewhere entirely -- the viewer's
       predicate_breakdown.occurrences (compute_occurrences in
       viewer/server.py), computed independently, straight from the
       trigger frame, with no dependency on this recovery_ltl at all."""
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_dropped_object_was_released",
            main_ltl="G(object_dropped -> (object_released | (!object_left_gripper U object_grasped)))",
            recovery_ltl="F(object_grasped | object_left_gripper)",
            property_description=property_description,
            binding={},
            explanation_builder=_dropped_object_released_explanation,
            reason_builder=_dropped_object_released_reason,
        )
    )


def build_repeated_released_settle_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    """2026-09-02 revision: recovery_ltl was a bare atom ("object_settled",
    no temporal operator -- see KNOWN_BUGS.md #10), which only ever checks
    whether object_settled holds at the exact frame recovery starts, not
    "eventually" -- confirmed 0% recovery rate corpus-wide under this bug.

    First fix attempt, G(object_released & !object_settled -> F(object_settled
    | release_object_settle_timeout)), turned out to have a different,
    more fundamental problem: recovery_ltl is only ever evaluated *after*
    RepeatedViolationMonitor's `in_violation` has already gone True (i.e.
    after main_ltl's own rejection is confirmed) -- which, for an until-shape
    main formula, happens some number of frames *after* the triggering edge
    (object_released, true for exactly one frame) has already reverted to
    False. So by the time recovery starts evaluating, `object_released` can
    never be True again in that sub-trace, making the whole
    `G(object_released & ... -> F(...))` vacuously true from frame 1 --
    "recovered" regardless of whether object_settled ever actually becomes
    true. Confirmed directly via an isolated LTLfDFA test (accepting
    immediately, before object_settled or release_object_settle_timeout was
    ever true) and against real data (ArrangeBreadBasket ep6: recovery
    already accepting at frame 395/396, well before object_settled's real,
    later transition at frame 399).

    Fix: drop the antecedent/G(...) wrapper entirely -- recovery_ltl is
    already only ever evaluated while genuinely in violation (that's what
    "in_violation" means), so there's nothing left to additionally gate on.
    Bare F(object_settled) (no antecedent, but *is* F(...)-wrapped, unlike
    the original bare-atom bug) correctly waits for object_settled to
    actually become true at any point in the recovery window. Verified
    against ep6: resolves at frame 399 (object_settled's real transition),
    not frame 395/396 (main's rejection-confirmation frame).

    General rule for any recovery_ltl: an edge-triggered antecedent
    (object_released, object_dropped, ...) inside a G(antecedent -> F(...))
    wrapper always produces this same vacuous-truth bug, because recovery
    only starts after the edge has already passed. A level-condition
    antecedent (e.g. forbidden_contact in rc_no_forbidden_contact's
    recovery_ltl, which can still be true when recovery starts) doesn't
    have this problem -- only wrap in G(antecedent -> ...) when the
    antecedent is a level condition that can genuinely still hold at the
    moment recovery begins; otherwise just use a bare F(...).

    2026-09-03, following rc_dropped_object_was_released's same final
    decision: added release_object_settle_timeout back into the escape
    (F(object_settled | release_object_settle_timeout)) -- this is the
    same tautological-escape-term situation as object_left_gripper was for
    that property (Bug B in the recovery-ltl-design skill): confirmed via
    a full-trace replay (not just an isolated single-step test, which gave
    a misleading result the first time) that release_object_settle_timeout
    is always True at the exact frame this until's trap confirms (ep6:
    both flip together at frame 395). So this makes recovery resolve
    near-instantly, unconditionally, for every violation of this property
    too -- the same "resume tracking the moment the main formula's own
    trap-defining condition fires" semantics as
    rc_dropped_object_was_released, not a genuine "did it actually settle"
    check. object_released here is likewise a one-shot past event no later
    condition can undo, so the same "recovery vs. resume" reasoning
    applies. Do NOT confuse this with (the different, unrelated)
    object_settle_timeout, used only by the liquid/solid containment-
    transfer properties -- see specs.py's predicates list for this
    property, which correctly lists release_object_settle_timeout, not
    that one."""
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_released_object_eventually_settles",
            main_ltl="G(object_released -> (!release_object_settle_timeout U object_settled))",
            recovery_ltl="F(object_settled | release_object_settle_timeout)",
            property_description=property_description,
            binding={},
            explanation_builder=_released_settle_explanation,
            reason_builder=_released_settle_reason,
        )
    )


def build_repeated_contamination_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_raw_robot_contact_blocks_rte_grasp_until_sanitized",
            main_ltl="G(robot_contact_raw_contaminated -> (!robot_contact_clean U sanitized))",
            recovery_ltl=(
                "G((robot_contact_raw_contaminated & robot_contact_clean & !sanitized) "
                "-> F(sanitized))"
            ),
            property_description=property_description,
            binding={},
            explanation_builder=_contamination_explanation,
            reason_builder=_contamination_reason,
        )
    )


def build_repeated_pick_precondition_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_pick_preconditions_safe",
            main_ltl="G(skill_pick_onset -> preconditions_satisfied_pick)",
            recovery_ltl="none",
            property_description=property_description,
            binding={},
            explanation_builder=_pick_precondition_explanation,
            reason_builder=_pick_precondition_reason,
            count_overlapping_rejections=True,
        )
    )


def build_repeated_place_precondition_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_place_preconditions_safe",
            main_ltl="G(skill_place_onset -> preconditions_satisfied_place)",
            recovery_ltl="none",
            property_description=property_description,
            binding={},
            explanation_builder=_place_precondition_explanation,
            reason_builder=_place_precondition_reason,
            count_overlapping_rejections=True,
        )
    )


def build_repeated_intended_safety_precondition_monitor(
    action: str,
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    if action not in INTENDED_SAFETY_PRECONDITION_SPECS:
        raise ValueError(f"Unsupported intended-safety action: {action}")
    property_name = f"rc_{action}_preconditions_safe"
    onset_name = f"skill_{action}_onset"
    preconditions_name = f"preconditions_satisfied_{action}"
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name=property_name,
            main_ltl=f"G({onset_name} -> {preconditions_name})",
            recovery_ltl="none",
            property_description=property_description,
            binding={},
            explanation_builder=lambda episodes: _generic_precondition_explanation(episodes, action),
            reason_builder=lambda episodes, in_violation_at_end: _generic_precondition_reason(
                episodes,
                in_violation_at_end,
                action,
            ),
            count_overlapping_rejections=True,
        )
    )


def build_repeated_fixture_open_obstacle_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_fixture_open_obstacle_retract",
            main_ltl="G(fixture_open_obstacle_hit -> (fixture_open_retracting U fixture_fully_closed))",
            recovery_ltl="fixture_fully_closed",
            property_description=property_description,
            binding={},
            explanation_builder=_fixture_open_obstacle_explanation,
            reason_builder=_fixture_open_obstacle_reason,
        )
    )


def build_repeated_fixture_close_obstacle_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_fixture_close_obstacle_retract",
            main_ltl="G(fixture_close_obstacle_hit -> (fixture_close_retracting U fixture_fully_open))",
            recovery_ltl="fixture_fully_open",
            property_description=property_description,
            binding={},
            explanation_builder=_fixture_close_obstacle_explanation,
            reason_builder=_fixture_close_obstacle_reason,
        )
    )


def build_repeated_liquid_transfer_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_liquid_transfer_eventually_settles",
            main_ltl="G(liquid_transfer_event -> (!object_settle_timeout U liquid_settled))",
            recovery_ltl="none",
            property_description=property_description,
            binding={},
            explanation_builder=lambda episodes: _containment_explanation("liquid", episodes),
            reason_builder=lambda episodes, in_violation_at_end: _containment_reason(
                "liquid",
                episodes,
                in_violation_at_end,
            ),
            count_overlapping_rejections=True,
        )
    )


def build_repeated_solid_transfer_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_solid_transfer_eventually_settles",
            main_ltl="G(solid_transfer_event -> (!object_settle_timeout U solid_settled))",
            recovery_ltl=(
                "G(solid_misplacement -> "
                "F(misplaced_solid_removed | misplaced_solid_recollected))"
            ),
            property_description=property_description,
            binding={},
            explanation_builder=lambda episodes: _containment_explanation("solid", episodes),
            reason_builder=lambda episodes, in_violation_at_end: _containment_reason(
                "solid",
                episodes,
                in_violation_at_end,
            ),
        )
    )


def build_repeated_microwave_single_object_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_microwave_single_object_until_empty",
            main_ltl="G(object_reach_in_fixture -> microwave_empty)",
            recovery_ltl="!two_or_more_objects_in_microwave",
            property_description=property_description,
            binding={},
            explanation_builder=lambda episodes: _access_explanation("microwave", episodes),
            reason_builder=lambda episodes, in_violation_at_end: _access_reason(
                "microwave",
                episodes,
                in_violation_at_end,
            ),
        )
    )


def build_repeated_reach_in_fixture_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_reach_in_fixture_only_when_fully_open",
            main_ltl="G(reach_in_fixture -> fixture_fully_open)",
            recovery_ltl="none",
            property_description=property_description,
            binding={},
            explanation_builder=lambda episodes: _access_explanation("reach", episodes),
            reason_builder=lambda episodes, in_violation_at_end: _access_reason(
                "reach",
                episodes,
                in_violation_at_end,
            ),
            count_overlapping_rejections=True,
        )
    )


def build_repeated_fixture_placement_support_monitor(
    property_description: Optional[str] = None,
) -> RepeatedViolationMonitor:
    return RepeatedViolationMonitor(
        RepeatedViolationMonitorConfig(
            property_name="rc_fixture_placement_release_after_internal_support",
            main_ltl="G(object_reach_in_fixture -> (!object_released U object_in_same_fixture))",
            recovery_ltl="none",
            property_description=property_description,
            binding={},
            explanation_builder=lambda episodes: _access_explanation("placement", episodes),
            reason_builder=lambda episodes, in_violation_at_end: _access_reason(
                "placement",
                episodes,
                in_violation_at_end,
            ),
            count_overlapping_rejections=True,
        )
    )
