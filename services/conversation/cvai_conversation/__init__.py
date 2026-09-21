"""Conversation Orchestrator — Milestones 9-13.

Owns the state machine (spec §15) and barge-in (spec §16). Not implemented yet; the
state vocabulary, the legal-transition table and the interrupt signal already live in
:mod:`cvai_types.conversation`, which is what lets later milestones add behaviour without
redefining the model.

The one piece that exists today is the transition guard, because it is pure logic, it is
easy to get wrong, and every later component will be written against it.
"""

from __future__ import annotations

from cvai_types import ALLOWED_TRANSITIONS, ConversationState, StateTransition, can_transition


class InvalidTransition(RuntimeError):
    """Raised instead of silently entering a state the machine does not allow."""


class StateMachine:
    """Minimal, synchronous state tracker with an audit trail.

    Kept deliberately small. The orchestrator in Milestone 9 owns scheduling, streaming
    and cancellation; this only answers "is this move legal, and what happened so far".
    """

    def __init__(self, initial: ConversationState = ConversationState.IDLE) -> None:
        self._state = initial
        self._history: list[StateTransition] = []

    @property
    def state(self) -> ConversationState:
        return self._state

    def history(self) -> list[StateTransition]:
        return list(self._history)

    def can(self, target: ConversationState) -> bool:
        return can_transition(self._state, target)

    def to(self, target: ConversationState, reason: str = "") -> StateTransition:
        if not self.can(target):
            allowed = sorted(s.value for s in ALLOWED_TRANSITIONS.get(self._state, ()))
            raise InvalidTransition(
                f"cannot move from {self._state.value} to {target.value}; "
                f"allowed: {allowed}"
            )
        transition = StateTransition(
            from_state=self._state, to_state=target, reason=reason
        )
        self._history.append(transition)
        self._state = target
        return transition

    def force(self, target: ConversationState, reason: str) -> StateTransition:
        """Escape hatch for error recovery, recorded as such.

        Exists so that an unexpected failure can always reach ``ERROR`` without the
        guard turning a bug into a deadlock — but it is a separate method, so forcing
        shows up in review rather than hiding inside ordinary flow.
        """
        transition = StateTransition(
            from_state=self._state, to_state=target, reason=f"forced: {reason}"
        )
        self._history.append(transition)
        self._state = target
        return transition


__all__ = ["InvalidTransition", "StateMachine"]
