"""Conversation memory: what she still knows twenty turns later (spec §11).

`MemorySettings` has promised `summarize_after_turns` since Milestone 1 and nothing
implemented it — history was trimmed to the last N exchanges and everything older was
simply gone. That failure is invisible in testing, where conversations are three turns
long, and glaring in use: you tell her your sister's name in turn 2, and by turn 20 she
has never heard of her. For a character whose whole value is being *this* character
talking to *you*, forgetting is worse than a slightly wrong intonation.

So the window still slides, but what falls out of it is summarised first, and the summary
rides in the system prompt.

Three rules, because a memory that invents things is worse than no memory at all:

* **Summaries are written in Chinese, in the third person, as facts.** Not as her voice.
  A summary written in character gets imitated as dialogue, and she starts repeating her
  own paraphrase of a conversation back at the user.
* **The fallback is extractive, never generative.** Without an LLM — or when one fails,
  or returns something implausible — the summary is built from what the *user* actually
  said, truncated. It is worse prose and it is true, and true is the property that
  matters. Her own replies are the recoverable half: the character can be re-derived
  from the profile; the user cannot.
* **It is bounded.** A summary that grows without limit eventually costs more context
  than the transcript it replaced.
"""

from __future__ import annotations

from collections.abc import Sequence

from cvai_core.logging_setup import get_logger
from cvai_types import LLMMessage, MemorySettings, Role

log = get_logger(__name__)

#: Hard cap on the running summary. Beyond this the oldest lines are dropped: the recent
#: past is what a conversation actually refers back to.
MAX_SUMMARY_CHARS = 600
#: Per remembered line, in the extractive fallback.
MAX_FACT_CHARS = 60

_SUMMARY_INSTRUCTION = (
    "把下面这段对话里**需要长期记住**的内容，用中文第三人称、客观地概括成几条要点。\n"
    "只写确实说过的事：对方的名字与称呼、提到的人和事、约定、偏好、情绪基调的变化。\n"
    "不要编造，不要补充没说过的细节，不要模仿任何人的语气，不要写成台词。\n"
    "如果没有值得记住的内容，就只回答：无。\n"
)


class ConversationMemory:
    """The turns, the running summary, and the rule for moving one into the other."""

    def __init__(
        self,
        settings: MemorySettings | None = None,
        *,
        summary: str = "",
    ) -> None:
        self.settings = settings or MemorySettings()
        self.turns: list[LLMMessage] = []
        self.summary = summary
        #: How many messages have been folded into the summary so far. Counted rather
        #: than recomputed, so a summarizer failure cannot cause the same turns to be
        #: summarised twice and appear in the prompt twice.
        self.summarized_messages = 0

    # -- recording -------------------------------------------------------------------

    def add(self, role: Role, text: str) -> None:
        if text.strip():
            self.turns.append(LLMMessage(role=role, content=text))

    def add_user(self, text: str) -> None:
        self.add(Role.USER, text)

    def add_character(self, text: str) -> None:
        self.add(Role.ASSISTANT, text)

    # -- reading ---------------------------------------------------------------------

    @property
    def exchanges(self) -> int:
        """Completed user/character pairs, rounded down."""
        return len(self.turns) // 2

    def recent(self, max_turns: int | None = None) -> list[LLMMessage]:
        limit = self.settings.max_turns if max_turns is None else max_turns
        if limit <= 0:
            return []
        return list(self.turns[-(limit * 2) :])

    @property
    def needs_summary(self) -> bool:
        if not self.settings.enabled or self.settings.summarize_after_turns <= 0:
            return False
        return len(self._pending()) >= self.settings.summarize_after_turns * 2

    def _pending(self) -> list[LLMMessage]:
        """Messages old enough to fall out of the window and not yet summarised."""
        keep = self.settings.max_turns * 2
        aged = self.turns[: max(0, len(self.turns) - keep)]
        return aged[self.summarized_messages :]

    # -- summarising -----------------------------------------------------------------

    async def maybe_summarize(self, llm=None) -> bool:
        """Fold the turns about to be forgotten into the summary. Returns whether it ran."""
        if not self.needs_summary:
            return False
        pending = self._pending()
        if not pending:
            return False

        addition = ""
        if llm is not None:
            try:
                addition = await self._ask(llm, pending)
            except Exception as exc:  # noqa: BLE001 - a failed summary must not kill a turn
                log.warning("summarization failed, falling back to extraction: %s", exc)

        if not _usable(addition):
            addition = extract_facts(pending)

        self.summarized_messages += len(pending)
        if addition:
            self.summary = _merge(self.summary, addition)
        return True

    async def _ask(self, llm, pending: Sequence[LLMMessage]) -> str:
        transcript = render_transcript(pending)
        messages = [
            LLMMessage(role=Role.SYSTEM, content=_SUMMARY_INSTRUCTION),
            LLMMessage(
                role=Role.USER,
                content=(
                    (f"已经记住的内容：\n{self.summary}\n\n" if self.summary else "")
                    + f"新的对话：\n{transcript}"
                ),
            ),
        ]
        # Low temperature on purpose: this is a transcription task wearing a summary's
        # clothes, and creativity here is indistinguishable from fabrication.
        response = await llm.complete(messages, temperature=0.2, max_output_tokens=300)
        return (getattr(response, "content", "") or "").strip()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def render_transcript(messages: Sequence[LLMMessage]) -> str:
    lines = []
    for message in messages:
        who = "对方" if message.role is Role.USER else "她"
        lines.append(f"{who}：{message.content.strip()}")
    return "\n".join(lines)


def extract_facts(messages: Sequence[LLMMessage]) -> str:
    """The no-LLM fallback: what the user said, compressed, never invented.

    Only the user's side. The character's replies can be re-derived from her profile;
    what the person told her cannot be re-derived from anything.
    """
    lines: list[str] = []
    for message in messages:
        if message.role is not Role.USER:
            continue
        text = " ".join(message.content.split())
        if not text:
            continue
        if len(text) > MAX_FACT_CHARS:
            text = text[: MAX_FACT_CHARS - 1] + "…"
        lines.append(f"- 对方说过：{text}")
    return "\n".join(lines)


def _usable(summary: str) -> bool:
    """Reject the empty answer, the refusal, and the model narrating its own task."""
    text = (summary or "").strip()
    if not text or text in {"无", "无。", "none", "None"}:
        return False
    return not text.startswith(("好的", "当然", "以下是", "作为"))


def _merge(existing: str, addition: str) -> str:
    merged = "\n".join(part for part in (existing.strip(), addition.strip()) if part)
    if len(merged) <= MAX_SUMMARY_CHARS:
        return merged
    # Drop from the front: a conversation refers back to its recent past far more often
    # than to its beginning, and the beginning is the part most likely to be stale.
    lines = merged.split("\n")
    while lines and len("\n".join(lines)) > MAX_SUMMARY_CHARS:
        lines.pop(0)
    return "\n".join(lines)
