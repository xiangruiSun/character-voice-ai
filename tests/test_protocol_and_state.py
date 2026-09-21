"""Conversation state machine and the browser protocol.

Both are Milestone 1 groundwork for later milestones. They are tested now because spec
§16 promises barge-in can be added "without restructuring the entire system", and a
promise like that is only kept if the vocabulary it depends on is correct from the start.
"""

from __future__ import annotations

import pytest
from cvai_audio_protocol import (
    PROTOCOL_VERSION,
    AudioBegin,
    Hello,
    Interrupt,
    StateChanged,
    parse_client_message,
    parse_server_message,
)
from cvai_conversation import InvalidTransition, StateMachine
from cvai_types import ConversationState, InterruptReason, InterruptSignal, can_transition


# --------------------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------------------


def test_the_happy_path_is_legal_end_to_end():
    machine = StateMachine()
    for target in (
        ConversationState.LISTENING,
        ConversationState.TRANSCRIBING,
        ConversationState.THINKING,
        ConversationState.PLANNING,
        ConversationState.SYNTHESIZING,
        ConversationState.SPEAKING,
        ConversationState.IDLE,
    ):
        machine.to(target)
    assert machine.state is ConversationState.IDLE
    assert len(machine.history()) == 7


def test_illegal_transitions_raise_instead_of_wedging_the_ui():
    machine = StateMachine()
    with pytest.raises(InvalidTransition) as exc:
        machine.to(ConversationState.SPEAKING)
    assert "idle" in str(exc.value) and "speaking" in str(exc.value)
    assert machine.state is ConversationState.IDLE


@pytest.mark.parametrize(
    "source",
    [
        ConversationState.THINKING,
        ConversationState.PLANNING,
        ConversationState.SYNTHESIZING,
        ConversationState.SPEAKING,
    ],
)
def test_every_in_flight_state_can_be_interrupted(source: ConversationState):
    """Barge-in must be reachable from anywhere work is happening (spec §16)."""
    assert can_transition(source, ConversationState.INTERRUPTED)


def test_interrupted_returns_to_listening():
    assert can_transition(ConversationState.INTERRUPTED, ConversationState.LISTENING)


def test_speaking_can_loop_back_to_synthesizing_for_the_next_chunk():
    """Streaming replies synthesize chunk N+1 while chunk N plays (spec §14)."""
    assert can_transition(ConversationState.SPEAKING, ConversationState.SYNTHESIZING)


def test_error_is_reachable_from_everywhere_and_recovers_to_idle():
    for state in ConversationState:
        if state is ConversationState.ERROR:
            continue
        assert can_transition(state, ConversationState.ERROR), state
    assert can_transition(ConversationState.ERROR, ConversationState.IDLE)


def test_force_is_recorded_as_forced():
    machine = StateMachine()
    machine.force(ConversationState.SPEAKING, "test recovery")
    assert machine.state is ConversationState.SPEAKING
    assert machine.history()[-1].reason.startswith("forced:")


def test_interrupt_signal_defaults_cover_the_required_behaviours():
    signal = InterruptSignal(reason=InterruptReason.USER_SPEECH)
    assert signal.stop_playback
    assert signal.clear_audio_queue
    assert signal.cancel_pending_tts
    # Cancelling the LLM is optional in spec §16, so it is opt-in.
    assert signal.cancel_llm_stream is False


# --------------------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------------------


def test_client_messages_round_trip():
    hello = Hello(character_id="denia_cn", playback_sample_rate=48000)
    parsed = parse_client_message(hello.model_dump(mode="json"))
    assert isinstance(parsed, Hello)
    assert parsed.character_id == "denia_cn"
    assert parsed.protocol == PROTOCOL_VERSION


def test_server_messages_round_trip():
    state = StateChanged(state=ConversationState.SPEAKING, turn_id="t1")
    parsed = parse_server_message(state.model_dump(mode="json"))
    assert isinstance(parsed, StateChanged)
    assert parsed.state is ConversationState.SPEAKING


def test_unknown_message_types_are_refused():
    with pytest.raises(ValueError):
        parse_client_message({"type": "not_a_real_message"})
    with pytest.raises(ValueError):
        parse_server_message({"type": "not_a_real_message"})


def test_interrupt_carries_the_turn_it_targets():
    """So a late interrupt cannot kill the reply that has already started."""
    signal = Interrupt(turn_id="turn-7")
    assert parse_client_message(signal.model_dump(mode="json")).turn_id == "turn-7"


def test_audio_begin_describes_the_binary_frames_that_follow():
    begin = AudioBegin(turn_id="t1", chunk_index=0, sample_rate=24000, text="你好")
    assert begin.encoding == "pcm_s16le"
    assert begin.channels == 1
