"""Statuses and their legal transitions. Pure: no I/O, no database.

Every status change in the services goes through :func:`transition`, so an illegal
move (say, a CANCELLED job going back to TRAINING) is a loud error, not a silently
inconsistent row.
"""

from __future__ import annotations

from enum import Enum


class InvalidTransition(ValueError):
    pass


class ProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class ConnectionKind(str, Enum):
    LOCAL = "local"                  # Ollama / vLLM / LM Studio on this machine
    PUBLIC_API = "public_api"        # a hosted API with a key
    CUSTOM = "custom"                # any other OpenAI-compatible endpoint


class ConnectionMode(str, Enum):
    #: The backend calls the model. The only mode implemented; the others are the
    #: seams for a browser-side or bridged local model later.
    BACKEND_PROXY = "backend_proxy"
    BROWSER_LOCAL = "browser_local"
    LOCAL_BRIDGE = "local_bridge"


class ConnectionStatus(str, Enum):
    DRAFT = "draft"
    TESTING = "testing"
    CONNECTED = "connected"
    ERROR = "error"
    DISABLED = "disabled"


class VoicePackStatus(str, Enum):
    EMPTY = "empty"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    NEEDS_REVIEW = "needs_review"
    READY = "ready"
    ERROR = "error"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    VALIDATING = "validating"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class VoiceModelStatus(str, Enum):
    READY = "ready"
    INVALID = "invalid"
    ARCHIVED = "archived"


class CharacterStatus(str, Enum):
    READY = "ready"
    INCOMPLETE = "incomplete"        # its model connection or voice is missing/broken


_ACTIVE_TRAINING = {
    TrainingStatus.QUEUED, TrainingStatus.VALIDATING, TrainingStatus.PREPARING,
    TrainingStatus.TRAINING, TrainingStatus.EVALUATING,
}
TRAINING_TERMINAL = {TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED}

TRANSITIONS: dict[type[Enum], dict[Enum, set[Enum]]] = {
    ConnectionStatus: {
        ConnectionStatus.DRAFT: {ConnectionStatus.TESTING, ConnectionStatus.DISABLED},
        ConnectionStatus.TESTING: {ConnectionStatus.CONNECTED, ConnectionStatus.ERROR},
        ConnectionStatus.CONNECTED: {ConnectionStatus.TESTING, ConnectionStatus.DISABLED,
                                     ConnectionStatus.DRAFT},
        ConnectionStatus.ERROR: {ConnectionStatus.TESTING, ConnectionStatus.DRAFT,
                                 ConnectionStatus.DISABLED},
        ConnectionStatus.DISABLED: {ConnectionStatus.DRAFT, ConnectionStatus.TESTING},
    },
    VoicePackStatus: {
        VoicePackStatus.EMPTY: {VoicePackStatus.UPLOADING},
        VoicePackStatus.UPLOADING: {VoicePackStatus.EMPTY, VoicePackStatus.PROCESSING,
                                    VoicePackStatus.UPLOADING, VoicePackStatus.ERROR},
        VoicePackStatus.PROCESSING: {VoicePackStatus.NEEDS_REVIEW, VoicePackStatus.ERROR},
        VoicePackStatus.NEEDS_REVIEW: {VoicePackStatus.READY, VoicePackStatus.PROCESSING,
                                       VoicePackStatus.UPLOADING},
        VoicePackStatus.READY: {VoicePackStatus.NEEDS_REVIEW, VoicePackStatus.PROCESSING,
                                VoicePackStatus.UPLOADING},
        VoicePackStatus.ERROR: {VoicePackStatus.PROCESSING, VoicePackStatus.UPLOADING},
    },
    TrainingStatus: {
        TrainingStatus.QUEUED: {TrainingStatus.VALIDATING},
        TrainingStatus.VALIDATING: {TrainingStatus.PREPARING},
        TrainingStatus.PREPARING: {TrainingStatus.TRAINING},
        TrainingStatus.TRAINING: {TrainingStatus.EVALUATING},
        TrainingStatus.EVALUATING: {TrainingStatus.COMPLETED},
        TrainingStatus.CANCEL_REQUESTED: {TrainingStatus.CANCELLED, TrainingStatus.FAILED,
                                          TrainingStatus.COMPLETED},
        TrainingStatus.COMPLETED: set(),
        TrainingStatus.FAILED: set(),
        TrainingStatus.CANCELLED: set(),
    },
}
# Any running stage can fail or be asked to cancel.
for _state in _ACTIVE_TRAINING:
    TRANSITIONS[TrainingStatus][_state] |= {TrainingStatus.FAILED, TrainingStatus.CANCEL_REQUESTED}


def can_transition(current: Enum, target: Enum) -> bool:
    return target in TRANSITIONS.get(type(current), {}).get(current, set())


def transition(current: Enum, target: Enum) -> Enum:
    if current == target and current in TRANSITIONS.get(type(current), {}).get(current, set()):
        return target
    if not can_transition(current, target):
        raise InvalidTransition(f"{type(current).__name__}: {current.value} → {target.value} is not allowed")
    return target


def training_is_active(status: TrainingStatus) -> bool:
    return status in _ACTIVE_TRAINING or status is TrainingStatus.CANCEL_REQUESTED
