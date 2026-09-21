"""Speech chunking for streamed replies (spec §14).

The LLM streams tokens; TTS wants sentences. Both extremes are named failure modes in
spec §27 — "token-by-token TTS" on one side, "extremely long TTS chunks" on the other —
so the chunker sits between them and decides where a reply can be cut without the seam
being audible.

Four inputs, exactly as spec §14 lists them:

* **Chinese punctuation.** 。！？…… end a thought; ，、： only pause inside one.
* **Semantic completeness.** A sentence-final particle (吧 呢 吗 啊 呀 嘛) or a closing
  quote makes a boundary safer; a conjunction right after it makes it worse.
* **Minimum length.** Emitting 哼。 on its own is a 0.4-second audio file, and the gap
  before the next one is more noticeable than the latency it saved — so short pieces are
  merged forward.
* **Maximum length.** Past that, cut at the best soft boundary available rather than
  waiting for a full stop that may never come.

Spec §14 also says the first version should be conservative, and naturalness beats
speed. The defaults reflect that: the first chunk is allowed to be shorter, because it
sets the response latency the user actually perceives, and everything after it is
allowed to be longer, because by then the user is listening rather than waiting.
"""

from __future__ import annotations

from typing import Final

from cvai_types import CVAIModel, SpeechChunk, StyleControls
from pydantic import Field

#: End of a thought. Safe to cut after.
STRONG_BOUNDARIES: Final = "。！？!?…"
#: A pause inside a thought. Cuttable when the buffer is already long.
SOFT_BOUNDARIES: Final = "，、；：,;:"
#: Closers that belong with the sentence they end, never with the next one.
TRAILING: Final = "”’）》」』\"')"

#: Sentence-final particles. Their presence means the clause really is complete.
FINAL_PARTICLES: Final = frozenset("吧呢吗啊呀嘛哦喔耶咯啦嘞")

#: Openers that must not be stranded at the end of a chunk: cutting before 但是 leaves
#: the listener waiting for a clause that arrives after an audible gap.
CONTINUATIONS: Final = (
    "但是", "但", "不过", "然后", "而且", "所以", "因为", "如果", "虽然",
    "并且", "或者", "还是", "可是", "只是", "就是",
)


class ChunkerConfig(CVAIModel):
    #: Below this, a chunk is merged forward instead of being spoken alone.
    min_chars: int = Field(default=8, ge=1, le=100)
    #: The first chunk may be shorter: it sets perceived response latency, and a
    #: three-character 好的。 arriving quickly reads better than silence.
    first_chunk_min_chars: int = Field(default=3, ge=1, le=100)
    #: Preferred ceiling. A soft boundary past this is enough to cut.
    max_chars: int = Field(default=60, ge=10, le=300)
    #: Absolute ceiling, cut with or without a boundary.
    hard_max_chars: int = Field(default=100, ge=20, le=500)
    #: Once ``max_chars`` is reached, a comma-level boundary is only used as a cut
    #: point if it sits at least this far in — otherwise the cut leaves a fragment.
    soft_cut_after_chars: int = Field(default=16, ge=1, le=200)
    #: Avoid cutting immediately before a conjunction.
    respect_continuations: bool = True


class SpeechChunker:
    """Incremental chunker. Feed deltas, take whole chunks out.

    Stateful and single-conversation: one instance per reply. ``reset`` makes it
    reusable, which matters for barge-in — an interrupted reply must not leak its tail
    into the next one.
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        *,
        controls: StyleControls | None = None,
    ) -> None:
        self.config = config or ChunkerConfig()
        self.controls = controls or StyleControls()
        self._buffer = ""
        self._emitted = 0
        self._last: SpeechChunk | None = None

    # -- state ---------------------------------------------------------------------

    def reset(self) -> None:
        self._buffer = ""
        self._emitted = 0
        self._last = None

    @property
    def pending(self) -> str:
        return self._buffer

    # -- streaming -----------------------------------------------------------------

    def feed(self, delta: str) -> list[SpeechChunk]:
        """Add streamed text; return whatever chunks are now complete."""
        self._buffer += delta
        chunks: list[SpeechChunk] = []
        while True:
            cut = self._find_cut(self._buffer)
            if cut is None:
                break
            piece, self._buffer = self._buffer[:cut], self._buffer[cut:]
            chunk = self._make(piece, final=False)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def flush(self) -> list[SpeechChunk]:
        """Emit the remainder at the end of a reply, minimum length notwithstanding."""
        chunks: list[SpeechChunk] = []
        while len(self._buffer) > self.config.hard_max_chars:
            cut = self._find_cut(self._buffer, force=True) or self.config.hard_max_chars
            piece, self._buffer = self._buffer[:cut], self._buffer[cut:]
            chunk = self._make(piece, final=False)
            if chunk is not None:
                chunks.append(chunk)

        remainder = self._buffer.strip()
        self._buffer = ""
        if remainder:
            chunk = self._make(remainder, final=True)
            if chunk is not None:
                chunks.append(chunk)
        elif self._last is not None:
            # The buffer emptied during streaming, so the final chunk has already been
            # handed out. Mark it now — the playback side needs to know which chunk
            # ends the turn, and it may be one that left before flush was called.
            self._last.is_final = True
        return chunks

    def split(self, text: str) -> list[SpeechChunk]:
        """Chunk a complete string. Same code path as streaming, one shot."""
        self.reset()
        chunks = self.feed(text)
        return chunks + self.flush()

    # -- decisions -----------------------------------------------------------------

    def _minimum(self) -> int:
        return (
            self.config.first_chunk_min_chars
            if self._emitted == 0
            else self.config.min_chars
        )

    def _find_cut(self, buffer: str, *, force: bool = False) -> int | None:
        """Index to cut at, or ``None`` to keep accumulating."""
        if not buffer:
            return None
        minimum = self._minimum()

        strong = self._first_strong(buffer, minimum)

        # 1. A full stop within the preferred length is always the best cut.
        if strong is not None and strong <= self.config.max_chars:
            return strong

        # 2. Past the preferred ceiling, a comma-level boundary will do — and is
        #    preferred over a full stop that is still far away. Deliberately not used
        #    below max_chars: cutting at every comma would shave latency at the cost of
        #    chopping the delivery, and spec §14 asks for the opposite trade.
        if len(buffer) >= self.config.max_chars:
            soft = self._best_soft(buffer, minimum)
            if soft is not None:
                return soft
            if strong is not None:
                return strong

        # 3. Past the hard ceiling, cut at the last soft boundary or bluntly.
        if force or len(buffer) >= self.config.hard_max_chars:
            window = buffer[: self.config.hard_max_chars]
            for index in range(len(window) - 1, minimum - 1, -1):
                if window[index] in SOFT_BOUNDARIES or window[index] in STRONG_BOUNDARIES:
                    return _include_trailing(buffer, index)
            return self.config.hard_max_chars
        return None

    def _first_strong(self, buffer: str, minimum: int) -> int | None:
        for index, char in enumerate(buffer):
            if char not in STRONG_BOUNDARIES:
                continue
            end = _include_trailing(buffer, index)
            if end >= minimum and not self._strands_a_continuation(buffer, end):
                return end
        return None

    def _best_soft(self, buffer: str, minimum: int) -> int | None:
        """Last comma-level boundary that still fits inside ``max_chars``."""
        best: int | None = None
        floor = max(minimum, self.config.soft_cut_after_chars)
        for index, char in enumerate(buffer):
            if char not in SOFT_BOUNDARIES:
                continue
            end = _include_trailing(buffer, index)
            if end > self.config.max_chars:
                break
            if end < floor or self._strands_a_continuation(buffer, end):
                continue
            best = end
        return best

    def _strands_a_continuation(self, buffer: str, end: int) -> bool:
        """True if cutting here leaves a conjunction beginning the next chunk."""
        if not self.config.respect_continuations:
            return False
        rest = buffer[end:].lstrip()
        if not rest:
            # Nothing after it yet; the stream may still deliver a conjunction, but a
            # completed sentence is a legitimate place to stop.
            return False
        return any(rest.startswith(word) for word in CONTINUATIONS)

    def _make(self, piece: str, *, final: bool) -> SpeechChunk | None:
        text = piece.strip()
        text = text.lstrip("".join(SOFT_BOUNDARIES) + "".join(STRONG_BOUNDARIES)).strip()
        if not text:
            return None
        chunk = SpeechChunk(
            chunk_index=self._emitted,
            text=text,
            is_final=final,
            controls=self.controls,
        )
        self._emitted += 1
        self._last = chunk
        return chunk


def _include_trailing(buffer: str, index: int) -> int:
    """Extend a cut past closing quotes and brackets that belong to this sentence."""
    end = index + 1
    while end < len(buffer) and buffer[end] in TRAILING:
        end += 1
    # Consecutive terminators (？！, ……。) go together.
    while end < len(buffer) and buffer[end] in STRONG_BOUNDARIES:
        end += 1
    return end


def ends_completely(text: str) -> bool:
    """Whether a piece reads as a finished thought.

    Used as a tie-breaker, and by tests. A sentence-final particle counts even without
    punctuation, because an LLM stream often has not produced the 。 yet.
    """
    stripped = text.rstrip("".join(TRAILING)).rstrip()
    if not stripped:
        return False
    if stripped[-1] in STRONG_BOUNDARIES:
        return True
    return stripped[-1] in FINAL_PARTICLES


def estimate_chunks(text: str, config: ChunkerConfig | None = None) -> list[str]:
    """Chunk a string and return plain text. Convenience for tests and reports."""
    return [chunk.text for chunk in SpeechChunker(config).split(text)]
