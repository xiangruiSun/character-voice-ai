"""The Conversation Orchestrator (Milestones 9-13).

One turn, end to end:

    user text (or STT)
      → Character Speech Planner        what she says, and how
      → Chinese text normalizer         TTS-ready text
      → speech chunker                  pieces an engine can speak naturally
      → reference retriever             which clip conditions each chunk
      → TTS provider                    audio
      → event stream                    played by whatever is listening

Spec §15 puts pipeline state here and nowhere else: the frontend observes `STATE`
events, it never decides them. Spec §16 requires barge-in to be addable "without
restructuring the entire system" — so it is built in from the start rather than
retrofitted, and the interruption path is the most heavily tested part of this module.

**Planning mode.** There is a real tension between spec §12 (a structured performance
plan for the reply) and spec §14 (start speaking before the reply is finished). The plan
describes the whole utterance, so it cannot exist until the LLM has finished. Two modes,
and the default follows spec §14's own instruction that naturalness beats speed:

* ``complete`` (default) — wait for the full plan, then normalize, chunk and synthesize,
  streaming each chunk's audio as it is ready. TTS is pipelined; the LLM is not.
* ``streaming`` — chunk the LLM's text as it arrives and speak it in the character's
  default register. Lower latency, no per-line performance direction.

Nothing else in the system depends on which mode is in use.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cvai_core.interfaces.reference import ReferenceRetriever
from cvai_core.interfaces.tts import TTSProvider
from cvai_core.logging_setup import get_logger
from cvai_speech_planner import CharacterSpeechPlanner
from cvai_text_normalizer import ChineseTextNormalizer, ChunkerConfig, SpeechChunker
from cvai_types import (
    AdaptationMode,
    CharacterProfile,
    CharacterSpeechPlan,
    ConversationState,
    CVAIModel,
    InterruptReason,
    InterruptSignal,
    LLMMessage,
    Role,
    SessionConfig,
    SpeechChunk,
    StyleControls,
    Turn,
    TurnEvent,
    TurnEventType,
    TTSRequest,
    utcnow,
)
from pydantic import Field

from .state_machine import StateMachine

log = get_logger(__name__)

PLAN_MODE_COMPLETE = "complete"
PLAN_MODE_STREAMING = "streaming"


class OrchestratorConfig(CVAIModel):
    plan_mode: str = Field(default=PLAN_MODE_COMPLETE, pattern=r"^(complete|streaming)$")
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    #: Synthesize the next chunk while the current one plays. 1 is enough to hide
    #: engine latency behind playback; more just wastes work when the user interrupts.
    lookahead_chunks: int = Field(default=1, ge=0, le=4)
    #: Cancel the LLM stream on barge-in. Off by default (spec §16 lists it as
    #: optional): a half-generated reply is still worth keeping in history, and some
    #: providers bill for the whole completion anyway.
    cancel_llm_on_interrupt: bool = False
    #: Where turn audio is written.
    audio_dir: str = "runs/sessions"


@dataclass
class TurnContext:
    """Everything one in-flight turn needs to be cancellable."""

    turn_id: str
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    reason: InterruptReason | None = None
    tasks: set[asyncio.Task] = field(default_factory=set)

    def cancel(self, reason: InterruptReason) -> None:
        self.reason = reason
        self.cancelled.set()
        for task in list(self.tasks):
            if not task.done():
                task.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()


class ConversationOrchestrator:
    """Runs turns for one conversation. Not safe to share across sessions."""

    def __init__(
        self,
        session: SessionConfig,
        profile: CharacterProfile,
        *,
        planner: CharacterSpeechPlanner,
        tts: TTSProvider,
        retriever: ReferenceRetriever,
        normalizer: ChineseTextNormalizer | None = None,
        config: OrchestratorConfig | None = None,
        audio_root: Path | None = None,
    ) -> None:
        self.session = session
        self.profile = profile
        self.planner = planner
        self.tts = tts
        self.retriever = retriever
        self.normalizer = normalizer or ChineseTextNormalizer.from_character(profile)
        self.config = config or OrchestratorConfig()

        self.machine = StateMachine()
        self.history: list[LLMMessage] = []
        self.turns: list[Turn] = []
        self._current: TurnContext | None = None
        self._audio_root = Path(audio_root or self.config.audio_dir) / session.session_id

    # -- observable state ----------------------------------------------------------

    @property
    def state(self) -> ConversationState:
        return self.machine.state

    @property
    def current_turn_id(self) -> str | None:
        return self._current.turn_id if self._current else None

    # -- barge-in (spec §16) --------------------------------------------------------

    async def interrupt(
        self, signal: InterruptSignal | None = None, *, turn_id: str | None = None
    ) -> bool:
        """Stop the current turn. Returns whether anything was actually interrupted.

        ``turn_id`` guards against a late signal killing the *next* reply: an interrupt
        aimed at a turn that has already finished is ignored, which is the difference
        between barge-in working and the system appearing to randomly go mute.
        """
        signal = signal or InterruptSignal(reason=InterruptReason.USER_SPEECH)
        context = self._current
        if context is None:
            return False
        if turn_id is not None and turn_id != context.turn_id:
            log.debug(
                "ignoring interrupt for finished turn %s (current is %s)",
                turn_id,
                context.turn_id,
            )
            return False
        if not self.session.enable_barge_in:
            return False

        log.info("barge-in on turn %s: %s", context.turn_id, signal.reason.value)
        context.cancel(signal.reason)
        return True

    # -- the turn ------------------------------------------------------------------

    async def run_turn(self, user_text: str) -> AsyncIterator[TurnEvent]:
        """Run one turn, emitting events as it goes."""
        turn_id = f"t{len(self.turns):04d}-{uuid.uuid4().hex[:6]}"
        context = TurnContext(turn_id=turn_id)
        self._current = context
        turn = Turn(turn_index=len(self.turns), user_text=user_text)
        self.turns.append(turn)

        try:
            async for event in self._run(context, user_text, turn):
                yield event
        except asyncio.CancelledError:
            yield self._state_event(context, ConversationState.INTERRUPTED, "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("turn %s failed", turn_id)
            self.machine.force(ConversationState.ERROR, str(exc))
            yield TurnEvent(
                type=TurnEventType.ERROR,
                turn_id=turn_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.machine.force(ConversationState.IDLE, "recovered from error")
        finally:
            turn.ended_at = utcnow().isoformat()
            self._current = None

    async def _run(
        self, context: TurnContext, user_text: str, turn: Turn
    ) -> AsyncIterator[TurnEvent]:
        self.history.append(LLMMessage(role=Role.USER, content=user_text))

        yield self._transition(context, ConversationState.THINKING)

        if self.config.plan_mode == PLAN_MODE_STREAMING:
            generator = self._streaming_turn(context, user_text, turn)
        else:
            generator = self._complete_turn(context, user_text, turn)

        async for event in generator:
            yield event
            if context.is_cancelled:
                break

        if context.is_cancelled:
            turn.was_interrupted = True
            self.machine.force(ConversationState.INTERRUPTED, "barge-in")
            yield self._state_event(
                context,
                ConversationState.INTERRUPTED,
                context.reason.value if context.reason else "interrupted",
            )
            self.machine.to(ConversationState.LISTENING, "after barge-in")
            yield TurnEvent(
                type=TurnEventType.TURN_END, turn_id=context.turn_id, is_final=True,
                detail="interrupted",
            )
            return

        if self.machine.can(ConversationState.IDLE):
            self.machine.to(ConversationState.IDLE, "turn complete")
        else:
            self.machine.force(ConversationState.IDLE, "turn complete")
        yield TurnEvent(
            type=TurnEventType.TURN_END, turn_id=context.turn_id, is_final=True
        )

    # -- complete mode (default) -----------------------------------------------------

    async def _complete_turn(
        self, context: TurnContext, user_text: str, turn: Turn
    ) -> AsyncIterator[TurnEvent]:
        yield self._transition(context, ConversationState.PLANNING)

        result = await self.planner.plan(
            self.profile.character_id, user_text, history=self.history[:-1]
        )
        if context.is_cancelled:
            return

        plan = result.plan
        turn.character_text = plan.text
        self.history.append(LLMMessage(role=Role.ASSISTANT, content=plan.text))

        yield TurnEvent(
            type=TurnEventType.PLAN,
            turn_id=context.turn_id,
            text=plan.text,
            detail=f"{plan.emotion} · {plan.speaking_rate.value} · {result.source}",
        )
        # The user sees the line as soon as it exists, before any audio — the caption
        # arriving first is what makes the wait for speech feel short.
        yield TurnEvent(
            type=TurnEventType.TEXT_DELTA,
            turn_id=context.turn_id,
            text=plan.text,
            is_final=True,
        )

        async for event in self._speak(context, plan.text, plan.to_style_controls()):
            yield event

    # -- streaming mode ---------------------------------------------------------------

    async def _streaming_turn(
        self, context: TurnContext, user_text: str, turn: Turn
    ) -> AsyncIterator[TurnEvent]:
        """Chunk the LLM's text as it arrives; no per-line performance direction."""
        controls = StyleControls(emotion=self.profile.voice.default_reference_style)
        chunker = SpeechChunker(self.config.chunker, controls=controls)
        messages = self.planner.compose(self.profile, user_text, history=self.history[:-1])

        collected: list[str] = []
        pending: list[SpeechChunk] = []

        async for delta in self.planner.llm.stream(
            messages,
            temperature=self.profile.llm.temperature,
            max_output_tokens=self.profile.llm.max_output_tokens,
        ):
            if context.is_cancelled:
                break
            if delta.delta:
                collected.append(delta.delta)
                yield TurnEvent(
                    type=TurnEventType.TEXT_DELTA,
                    turn_id=context.turn_id,
                    text=delta.delta,
                )
                pending.extend(chunker.feed(delta.delta))

            while pending and not context.is_cancelled:
                chunk = pending.pop(0)
                async for event in self._speak_chunk(context, chunk):
                    yield event

        if not context.is_cancelled:
            for chunk in chunker.flush():
                async for event in self._speak_chunk(context, chunk):
                    yield event
                if context.is_cancelled:
                    break

        text = "".join(collected)
        turn.character_text = text
        if text:
            self.history.append(LLMMessage(role=Role.ASSISTANT, content=text))

    # -- synthesis --------------------------------------------------------------------

    async def _speak(
        self, context: TurnContext, text: str, controls: StyleControls
    ) -> AsyncIterator[TurnEvent]:
        normalized = self.normalizer.normalize(text)
        if normalized.warnings:
            log.warning(
                "text normalization warnings on turn %s: %s",
                context.turn_id,
                normalized.warnings,
            )
        chunks = SpeechChunker(self.config.chunker, controls=controls).split(
            normalized.text
        )
        for chunk in chunks:
            if context.is_cancelled:
                return
            async for event in self._speak_chunk(context, chunk, pre_normalized=True):
                yield event

    async def _speak_chunk(
        self, context: TurnContext, chunk: SpeechChunk, *, pre_normalized: bool = False
    ) -> AsyncIterator[TurnEvent]:
        if context.is_cancelled:
            return

        text = chunk.text
        if not pre_normalized:
            text = self.normalizer.normalize(text).text
        if not text.strip():
            return

        chunk = chunk.model_copy(update={"normalized_text": text})
        yield TurnEvent(
            type=TurnEventType.CHUNK,
            turn_id=context.turn_id,
            text=text,
            chunk_index=chunk.chunk_index,
            is_final=chunk.is_final,
        )

        if self.machine.state is not ConversationState.SYNTHESIZING:
            if self.machine.can(ConversationState.SYNTHESIZING):
                self.machine.to(ConversationState.SYNTHESIZING, "chunk ready")
                yield self._state_event(context, ConversationState.SYNTHESIZING)

        reference = self.retriever.select(
            chunk.controls, variation_key=f"{context.turn_id}:{chunk.chunk_index}"
        )
        output = self._audio_root / context.turn_id / f"{chunk.chunk_index:03d}.wav"
        request = TTSRequest(
            request_id=f"{context.turn_id}:{chunk.chunk_index}",
            text=chunk.synthesis_text,
            controls=chunk.controls,
            reference=reference,
            adaptation_mode=AdaptationMode.ZERO_SHOT,
            output_sample_rate=None,
        )

        task = asyncio.ensure_future(self.tts.synthesize(request, output))
        context.tasks.add(task)
        try:
            result = await task
        except asyncio.CancelledError:
            # Expected on barge-in: the in-flight synthesis is abandoned and its
            # partial output, if any, is never emitted.
            log.debug("synthesis cancelled for %s", request.request_id)
            return
        finally:
            context.tasks.discard(task)

        if context.is_cancelled:
            # Finished just after the interrupt. Dropping it here is the difference
            # between barge-in feeling instant and the character getting one more word
            # in after being cut off.
            return

        if self.machine.state is not ConversationState.SPEAKING:
            self.machine.to(ConversationState.SPEAKING, "audio ready")
            yield self._state_event(context, ConversationState.SPEAKING)

        yield TurnEvent(
            type=TurnEventType.AUDIO,
            turn_id=context.turn_id,
            chunk_index=chunk.chunk_index,
            text=text,
            is_final=chunk.is_final,
            audio_path=str(result.audio_path),
            sample_rate=result.sample_rate,
            duration_s=result.duration_s,
            reference_id=result.reference_id,
            engine=result.engine,
            latency_ms=result.latency_ms,
        )

    # -- helpers ----------------------------------------------------------------------

    def _transition(
        self, context: TurnContext, state: ConversationState, detail: str = ""
    ) -> TurnEvent:
        if self.machine.can(state):
            self.machine.to(state, detail)
        else:
            self.machine.force(state, detail or "out-of-order transition")
        return self._state_event(context, state, detail)

    @staticmethod
    def _state_event(
        context: TurnContext, state: ConversationState, detail: str = ""
    ) -> TurnEvent:
        return TurnEvent(
            type=TurnEventType.STATE,
            turn_id=context.turn_id,
            state=state,
            detail=detail,
        )

    # -- ConversationSession-shaped convenience ---------------------------------------

    async def submit_text(self, text: str) -> AsyncIterator[SpeechChunk]:
        """Run a turn, yielding only the speech chunks.

        Conforms to :class:`cvai_core.interfaces.session.ConversationSession`. Callers
        that need audio, state or captions use :meth:`run_turn` instead.
        """
        async for event in self.run_turn(text):
            if event.type is TurnEventType.CHUNK:
                yield SpeechChunk(
                    chunk_index=event.chunk_index or 0,
                    text=event.text,
                    normalized_text=event.text,
                    is_final=event.is_final,
                )

    async def collect(self, text: str) -> list[TurnEvent]:
        """Run a turn and return every event. Convenience for tests and scripts."""
        return [event async for event in self.run_turn(text)]

    def audio_paths(self, events: Sequence[TurnEvent]) -> list[str]:
        return [e.audio_path for e in events if e.audio_path]

    async def aclose(self) -> None:
        if self._current is not None:
            self._current.cancel(InterruptReason.USER_CANCEL)
        await self.tts.aclose()


def plan_summary(plan: CharacterSpeechPlan) -> str:
    """One line describing a plan, for logs and the CLI."""
    return (
        f"{plan.emotion}@{plan.emotion_intensity:.2f} "
        f"rate={plan.speaking_rate.value} vol={plan.volume_style.value} "
        f"end={plan.ending_style.value}"
    )
