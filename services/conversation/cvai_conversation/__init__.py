"""Conversation Orchestrator (Milestones 9-13).

Owns pipeline state (spec §15) and barge-in (spec §16). The frontend observes state, it
never decides it — which is why the state machine lives here and the transport layers
(FastAPI, CLI, tests) all consume the same :class:`~cvai_types.conversation.TurnEvent`
stream.
"""

from __future__ import annotations

from .orchestrator import (
    PLAN_MODE_COMPLETE,
    PLAN_MODE_STREAMING,
    ConversationOrchestrator,
    OrchestratorConfig,
    TurnContext,
    plan_summary,
)
from .state_machine import InvalidTransition, StateMachine

__all__ = [
    "ConversationOrchestrator",
    "InvalidTransition",
    "OrchestratorConfig",
    "PLAN_MODE_COMPLETE",
    "PLAN_MODE_STREAMING",
    "StateMachine",
    "TurnContext",
    "plan_summary",
]
