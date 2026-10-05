"""Building and holding conversation sessions (Milestones 9-11).

Assembling an orchestrator means wiring six things together — config, character profile,
voice pack, reference bank, planner and TTS engine — and each of them can be missing or
wrong in a way the user needs told about. That assembly lives here, separate from the
HTTP layer, so it can be tested without a web server and so the error messages say what
to do rather than returning a 500.

Sessions are in-memory. Spec §3 is explicit that the MVP uses the local filesystem and
adds no database, so a restart loses conversations; that is the intended trade until the
character voice pipeline works.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cvai_conversation import (
    ConversationOrchestrator,
    ListenerConfig,
    OrchestratorConfig,
    VoiceLoop,
)
from cvai_core.config import AppConfig, load_config
from cvai_core.errors import CVAIError, VoicePackError
from cvai_core.loaders import (
    FilesystemCharacterProvider,
    open_voicepack,
    try_load_reference_bank,
)
from cvai_core.logging_setup import get_logger
from cvai_core.paths import repo_root
from cvai_core.registry import (
    build_llm_provider,
    build_stt_provider,
    build_tts_provider,
)
from cvai_reference_retrieval import RuleBasedReferenceRetriever
from cvai_speech_planner import CharacterSpeechPlanner
from cvai_text_normalizer import ChineseTextNormalizer
from cvai_types import CharacterProfile, SessionConfig

log = get_logger(__name__)


@dataclass
class Session:
    session_id: str
    character_id: str
    orchestrator: ConversationOrchestrator
    profile: CharacterProfile
    engine: str
    #: Built on the first microphone frame, not at session creation: a text-only
    #: conversation should not load an ASR model, and a missing STT provider should
    #: only be an error for the user who actually speaks.
    _voice_loop: VoiceLoop | None = None
    _stt_builder: Callable[[], object] | None = None

    def voice_loop(self, config: ListenerConfig | None = None) -> VoiceLoop:
        if self._voice_loop is None:
            if self._stt_builder is None:
                raise CVAIError(
                    "no speech-to-text provider is configured, so the microphone "
                    "cannot be used. Add a `providers.stt` block to app.yaml."
                )
            self._voice_loop = VoiceLoop(
                self.orchestrator,
                self._stt_builder(),  # type: ignore[arg-type]
                config=config,
            )
        return self._voice_loop


@dataclass
class SessionManager:
    """Creates orchestrators and keeps them alive between requests."""

    config: AppConfig
    orchestrator_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    audio_root: Path | None = None
    sessions: dict[str, Session] = field(default_factory=dict)
    #: Injected in tests; ``None`` means "build from config".
    tts_factory: object | None = None
    llm_factory: object | None = None
    stt_factory: object | None = None

    def __post_init__(self) -> None:
        root = repo_root()
        self._characters = FilesystemCharacterProvider(
            root / self.config.paths.characters
        )
        self._packs_root = root / self.config.paths.voicepacks
        self.audio_root = Path(self.audio_root or (root / self.config.paths.runs) / "sessions")

    # -- construction ----------------------------------------------------------------

    @classmethod
    def from_config_file(cls, path: Path | None = None, **kwargs) -> "SessionManager":
        return cls(config=load_config([path] if path else None), **kwargs)

    def available_characters(self) -> list[str]:
        return self._characters.list_ids()

    def character_details(self) -> list[dict[str, str]]:
        """Ids with display names, for a character picker. Unloadable profiles are
        left out rather than failing the whole list."""
        details = []
        for character_id in self._characters.list_ids():
            try:
                name = self._characters.get(character_id).character_name
            except CVAIError:
                continue
            details.append({"id": character_id, "name": name})
        return details

    # -- lifecycle -------------------------------------------------------------------

    def create(
        self,
        character_id: str | None = None,
        *,
        session_id: str | None = None,
        engine: str | None = None,
        enable_barge_in: bool = True,
    ) -> Session:
        character_id = character_id or self.config.default_character
        profile = self._characters.get(character_id)

        paths, manifest = open_voicepack(profile.voice.voicepack_id, self._packs_root)
        bank = try_load_reference_bank(paths)
        if bank is None or not bank.samples:
            raise VoicePackError(
                f"voice pack {manifest.voicepack_id!r} has no reference bank, so there "
                "is nothing to condition synthesis on. Build one with "
                f"`cvai-prep build {manifest.voicepack_id}`."
            )

        retriever = RuleBasedReferenceRetriever.from_voicepack(paths, manifest, bank)

        # The character's preferred engine wins, then the request, then the config
        # default — so a Milestone 7 decision recorded in the profile takes effect
        # everywhere without touching deployment config.
        engine_name = profile.voice.preferred_engine or engine
        tts = (
            self.tts_factory(self.config, engine_name)  # type: ignore[operator]
            if self.tts_factory
            else build_tts_provider(self.config, engine_name)
        )
        llm = (
            self.llm_factory(self.config, None)  # type: ignore[operator]
            if self.llm_factory
            else build_llm_provider(self.config)
        )

        identifier = session_id or f"s-{uuid.uuid4().hex[:10]}"
        orchestrator = ConversationOrchestrator(
            SessionConfig(
                session_id=identifier,
                character_id=character_id,
                tts_engine=engine_name,
                enable_barge_in=enable_barge_in,
            ),
            profile,
            planner=CharacterSpeechPlanner(llm, self._characters),
            tts=tts,
            retriever=retriever,
            normalizer=ChineseTextNormalizer.from_character(profile),
            config=self.orchestrator_config,
            audio_root=self.audio_root,
        )

        session = Session(
            session_id=identifier,
            character_id=character_id,
            orchestrator=orchestrator,
            profile=profile,
            engine=getattr(tts, "engine", "unknown"),
            _stt_builder=self._stt_builder(),
        )
        self.sessions[identifier] = session
        log.info(
            "session %s opened for %s using %s",
            identifier,
            character_id,
            session.engine,
        )
        return session

    def _stt_builder(self) -> Callable[[], object] | None:
        """A thunk that makes an STT provider, or ``None`` if none is configured."""
        if self.stt_factory is not None:
            return lambda: self.stt_factory(self.config, None)  # type: ignore[operator]
        if self.config.providers.stt is None:
            return None
        return lambda: build_stt_provider(self.config)

    def get(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise CVAIError(f"unknown session {session_id!r}")
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            await session.orchestrator.aclose()
            if session._voice_loop is not None:  # noqa: SLF001 - same module
                await session._voice_loop.stt.aclose()  # noqa: SLF001
            log.info("session %s closed", session_id)

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.close(session_id)

    # -- serving audio ----------------------------------------------------------------

    def resolve_audio(self, session_id: str, relative: str) -> Path:
        """Turn a client-supplied audio path into a real file, refusing escapes.

        The path comes back from a browser, so it is untrusted input: without this
        check, ``../../etc/passwd`` would be served happily.
        """
        session = self.get(session_id)
        base = (Path(self.audio_root) / session.session_id).resolve()
        target = (base / relative).resolve()
        if not str(target).startswith(str(base)):
            raise CVAIError("audio path escapes the session directory")
        if not target.is_file():
            raise CVAIError(f"no such audio file: {relative}")
        return target
