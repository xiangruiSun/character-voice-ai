"""Benchmark and human-evaluation schemas (spec §17, §18, §19, §23).

The first technical question the project must answer is which adaptation approach
reproduces the character best. These types are the vocabulary of that experiment.

Two things are deliberately structural rather than conventional:

* A candidate is ``(engine, adaptation_mode)``, never just an engine name (decision D4).
* Every generated file has a ``GenerationRecord`` holding engine version, checkpoint,
  voice pack version, reference clip, resolved parameters and seed (decision D5). A
  listening test that cannot be traced back to a configuration is not an experiment.
"""

from __future__ import annotations

from enum import Enum
from statistics import mean, pstdev

from pydantic import Field, model_validator

from .base import CVAIModel, Language, SemVer, Slug, StyleTag, UnitFloat, utcnow
from .tts import AdaptationMode

# --------------------------------------------------------------------------------------
# Test material
# --------------------------------------------------------------------------------------


class SentenceCategory(str, Enum):
    """What each test sentence is probing.

    A test set that is twenty neutral statements cannot distinguish a character voice
    from a competent generic one, which is exactly the distinction that matters here.
    """

    EVERYDAY = "everyday"
    EMOTIONAL = "emotional"
    QUESTION = "question"
    SHORT_INTERJECTION = "short_interjection"
    LONG_SENTENCE = "long_sentence"
    NUMBERS_AND_DATES = "numbers_and_dates"
    LATIN_AND_ABBREVIATIONS = "latin_and_abbreviations"
    PUNCTUATION_HEAVY = "punctuation_heavy"
    CHARACTER_SIGNATURE = "character_signature"
    PRONUNCIATION_TRAP = "pronunciation_trap"


class TestSentence(CVAIModel):
    """One unseen sentence. Must not appear in any training split."""

    sentence_id: str = Field(min_length=1, max_length=60)
    text: str = Field(min_length=1, max_length=300)
    language: Language = "zh-CN"
    category: SentenceCategory = SentenceCategory.EVERYDAY
    #: The style the line should be performed in, so reference retrieval is exercised
    #: rather than every candidate defaulting to neutral.
    target_style: StyleTag = "neutral"
    #: What a listener should be paying attention to. Shown to raters only in
    #: non-blind debugging views, never during the blind test.
    probe: str = Field(default="", max_length=300)


class TestSentenceSet(CVAIModel):
    set_id: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=4000)
    sentences: list[TestSentence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> "TestSentenceSet":
        ids = [s.sentence_id for s in self.sentences]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate sentence_id in test sentence set")
        return self

    def by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sentence in self.sentences:
            key = sentence.category.value
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


# --------------------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------------------


class BenchmarkCandidate(CVAIModel):
    """One thing being compared: an engine in a specific adaptation mode."""

    candidate_id: Slug
    engine: str = Field(min_length=1, max_length=60)
    adaptation_mode: AdaptationMode = AdaptationMode.ZERO_SHOT
    checkpoint_id: str | None = None
    display_name: str = Field(default="", max_length=120)
    enabled: bool = True
    #: Which version of the reference clip conditions this candidate. ``clean`` is the
    #: processed clip from the Reference Bank; ``unprocessed`` rebuilds the same clip
    #: from the untouched original. Running one candidate each way is the
    #: null-processing control of decision D7, and the only way to find out whether the
    #: cleaning chain preserved the character or quietly sanded her down. Everything
    #: else about the two candidates is identical, so any difference is the processing.
    reference_source: str = Field(default="clean", pattern=r"^(clean|unprocessed)$")
    #: Engine parameters for this candidate, merged over the provider config.
    engine_params: dict[str, object] = Field(default_factory=dict)
    #: Licence facts, carried so an incompatible engine cannot silently win (M7).
    license: str = Field(default="unknown", max_length=120)
    commercial_use: bool | None = None
    notes: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _checkpoint_required_for_adaptation(self) -> "BenchmarkCandidate":
        # Only enforced for enabled candidates: a disabled row is how the benchmark
        # config records "this candidate exists and is waiting for its checkpoint",
        # which is more useful than deleting it and forgetting it was planned.
        if (
            self.enabled
            and self.adaptation_mode in (AdaptationMode.FINETUNED, AdaptationMode.LORA)
            and not self.checkpoint_id
        ):
            raise ValueError(
                f"candidate {self.candidate_id!r} claims adaptation mode "
                f"{self.adaptation_mode.value!r} but names no checkpoint_id"
            )
        return self

    @property
    def label(self) -> str:
        return self.display_name or f"{self.engine}:{self.adaptation_mode.value}"


class GroundTruthItem(CVAIModel):
    """A real recording of the character, used as a hidden anchor in the listening test.

    Without an anchor there is no scale: raters drift, and a 3.8 in one session is not a
    3.8 in another. The anchor also catches a rater who is not actually listening.
    """

    item_id: str
    audio_path: str
    transcript: str
    style: StyleTag = "neutral"


# --------------------------------------------------------------------------------------
# Generation records
# --------------------------------------------------------------------------------------


class GenerationRecord(CVAIModel):
    """Everything spec §23 requires logged, for one generated file."""

    record_id: str
    candidate_id: Slug
    sentence_id: str
    text: str
    audio_path: str | None = None

    engine: str
    engine_version: str = "unknown"
    adaptation_mode: AdaptationMode = AdaptationMode.ZERO_SHOT
    checkpoint_id: str | None = None
    voicepack_id: Slug
    voicepack_version: SemVer

    reference_id: str | None = None
    reference_style: StyleTag | None = None
    reference_was_fallback: bool = False
    #: ``clean`` or ``unprocessed`` — see ``BenchmarkCandidate.reference_source``. On
    #: the record because a result that cannot say which audio conditioned it is not
    #: reproducible (spec §23).
    reference_source: str = "clean"

    seed: int | None = None
    resolved_params: dict[str, object] = Field(default_factory=dict)
    dropped_controls: list[str] = Field(default_factory=list)

    sample_rate: int | None = None
    duration_s: float | None = None
    latency_ms: float | None = None

    succeeded: bool = True
    error: str | None = None
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())


class ObjectiveMetrics(CVAIModel):
    """Automatic metrics for one generated file or one candidate (Milestone 6).

    Objective metrics rank candidates and catch regressions. They do not choose the
    winner — decision D8.
    """

    #: Speaker Encoder Cosine Similarity against real character audio (CAM++/ERes2NetV2).
    #: Always report alongside ``secs_ceiling``: real-vs-real similarity for the same
    #: character, which is the actual upper bound, not 1.0.
    secs: float | None = Field(default=None, ge=-1.0, le=1.0)
    secs_ceiling: float | None = Field(default=None, ge=-1.0, le=1.0)
    #: Character Error Rate from ASR on the generated audio versus the input text.
    cer: float | None = Field(default=None, ge=0.0, le=2.0)
    #: TTSDS2 overall and factor scores, 0-100.
    ttsds2_overall: float | None = Field(default=None, ge=0.0, le=100.0)
    ttsds2_speaker: float | None = Field(default=None, ge=0.0, le=100.0)
    ttsds2_prosody: float | None = Field(default=None, ge=0.0, le=100.0)
    ttsds2_intelligibility: float | None = Field(default=None, ge=0.0, le=100.0)
    #: Reference-free MOS predictors. Recorded, but weakest evidence of the set.
    utmos: float | None = Field(default=None, ge=1.0, le=5.0)
    dnsmos: float | None = Field(default=None, ge=1.0, le=5.0)
    #: Distance between the candidate's prosody distribution and the character's real
    #: one. How "identical intonation across all sentences" is caught numerically.
    f0_mean_delta_hz: float | None = None
    f0_std_ratio: float | None = Field(default=None, ge=0.0)
    speaking_rate_delta_cps: float | None = None
    rtf: float | None = Field(default=None, ge=0.0)


# --------------------------------------------------------------------------------------
# Blind listening test
# --------------------------------------------------------------------------------------


class BlindItem(CVAIModel):
    """One audio file as the rater sees it.

    Carries nothing that identifies the producing system — not the candidate, not the
    sentence id, and not a filesystem path that encodes either. Audio is served from a
    flat ``audio/<item_id>.wav`` copied into the blind directory, because
    ``audio/gpt_sovits_ft/s04.wav`` in a network tab is as good as a label. The mapping
    back to candidate and sentence lives in :class:`BlindKeyEntry`.
    """

    item_id: str = Field(min_length=1, max_length=60)
    #: Relative to the blind directory, e.g. ``audio/item-007.wav``.
    audio_path: str
    #: Presentation order within the rater's session.
    position: int = Field(ge=0)
    #: Shown so the rater can judge pronunciation; does not reveal the system.
    text: str


class BlindKeyEntry(CVAIModel):
    """The mapping from an opaque item id back to what produced it.

    Kept in a *separate file* from the item list, so the rating interface can be handed
    to a listener without also handing them the answers.
    """

    item_id: str
    candidate_id: str
    sentence_id: str
    is_ground_truth: bool = False
    #: Where the audio was copied from, relative to the run directory. Keeps the trail
    #: from an opaque blind item back to the generation record that produced it.
    source_audio_path: str = ""


class BlindTestSet(CVAIModel):
    test_id: str
    run_id: str
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    items: list[BlindItem] = Field(default_factory=list)
    #: Seed used to shuffle. Recorded so a test can be regenerated identically.
    shuffle_seed: int = 0
    instructions: str = Field(default="", max_length=4000)


class BlindKey(CVAIModel):
    test_id: str
    run_id: str
    entries: list[BlindKeyEntry] = Field(default_factory=list)

    def candidate_of(self, item_id: str) -> str | None:
        for entry in self.entries:
            if entry.item_id == item_id:
                return entry.candidate_id
        return None


# --------------------------------------------------------------------------------------
# Human ratings
# --------------------------------------------------------------------------------------


class RatingAxis(str, Enum):
    """The four axes from spec §18."""

    SPEAKER_SIMILARITY = "speaker_similarity"
    NATURALNESS = "naturalness"
    CHARACTER_SIMILARITY = "character_similarity"
    AI_ARTIFACT_LEVEL = "ai_artifact_level"


#: Axis direction. ``ai_artifact_level`` asks "how obvious is it that this is generated",
#: so a *low* score is good. Aggregation must know this or the report will rank the worst
#: system first — an easy and embarrassing bug.
AXIS_HIGHER_IS_BETTER: dict[RatingAxis, bool] = {
    RatingAxis.SPEAKER_SIMILARITY: True,
    RatingAxis.NATURALNESS: True,
    RatingAxis.CHARACTER_SIMILARITY: True,
    RatingAxis.AI_ARTIFACT_LEVEL: False,
}

AXIS_PROMPTS: dict[RatingAxis, str] = {
    RatingAxis.SPEAKER_SIMILARITY: "这听起来像是目标角色的声音吗？",
    RatingAxis.NATURALNESS: "这听起来像真人录音吗？",
    RatingAxis.CHARACTER_SIMILARITY: "这听起来像这个角色平时说话的方式吗？",
    RatingAxis.AI_ARTIFACT_LEVEL: "能多明显地听出这是AI生成的？（越低越好）",
}

class HumanRating(CVAIModel):
    """One rater's scores for one blind item. 1-5 on each axis (spec §18)."""

    rater_id: str = Field(min_length=1, max_length=60)
    item_id: str
    speaker_similarity: int = Field(ge=1, le=5)
    naturalness: int = Field(ge=1, le=5)
    character_similarity: int = Field(ge=1, le=5)
    ai_artifact_level: int = Field(ge=1, le=5)
    #: Whether the rater believes this is a real recording. The headline success
    #: condition in spec §18 is fooling someone who knows the character, so it is worth
    #: asking directly rather than inferring it from a naturalness score.
    believed_real: bool | None = None
    comment: str = Field(default="", max_length=1000)
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())

    def score(self, axis: RatingAxis) -> int:
        return int(getattr(self, axis.value))


class AxisAggregate(CVAIModel):
    axis: RatingAxis
    mean: float
    std: float
    n: int
    higher_is_better: bool


class CandidateAggregate(CVAIModel):
    candidate_id: str
    label: str = ""
    is_ground_truth: bool = False
    n_ratings: int = 0
    n_items: int = 0
    axes: list[AxisAggregate] = Field(default_factory=list)
    #: Share of ratings where the listener thought it was a real recording.
    believed_real_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    objective: ObjectiveMetrics | None = None

    def axis_mean(self, axis: RatingAxis) -> float | None:
        for entry in self.axes:
            if entry.axis is axis:
                return entry.mean
        return None

    def composite(self) -> float | None:
        """Single ranking number, with ``ai_artifact_level`` inverted.

        A convenience for ordering a report. The decision record in Milestone 7 should
        quote the individual axes, because a composite hides the case where a model is
        natural but not the character — the exact failure spec §5 warns about.
        """
        values: list[float] = []
        for entry in self.axes:
            values.append(entry.mean if entry.higher_is_better else 6.0 - entry.mean)
        return round(mean(values), 4) if values else None


class AggregateReport(CVAIModel):
    run_id: str
    test_id: str | None = None
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    candidates: list[CandidateAggregate] = Field(default_factory=list)
    n_raters: int = 0
    notes: str = Field(default="", max_length=2000)

    def ranked(self) -> list[CandidateAggregate]:
        return sorted(
            self.candidates,
            key=lambda c: (c.composite() is None, -(c.composite() or 0.0)),
        )


def aggregate_axis(scores: list[int], axis: RatingAxis) -> AxisAggregate:
    """Mean/σ for one axis. ``σ`` is population stdev; n=1 gives 0.0."""
    if not scores:
        return AxisAggregate(
            axis=axis, mean=0.0, std=0.0, n=0, higher_is_better=AXIS_HIGHER_IS_BETTER[axis]
        )
    return AxisAggregate(
        axis=axis,
        mean=round(mean(scores), 4),
        std=round(pstdev(scores) if len(scores) > 1 else 0.0, 4),
        n=len(scores),
        higher_is_better=AXIS_HIGHER_IS_BETTER[axis],
    )


# --------------------------------------------------------------------------------------
# Run manifest
# --------------------------------------------------------------------------------------


class BenchmarkRun(CVAIModel):
    """One execution of the benchmark over all candidates × all sentences."""

    run_id: str = Field(min_length=1, max_length=80)
    created_at: str = Field(default_factory=lambda: utcnow().isoformat())
    voicepack_id: Slug
    voicepack_version: SemVer
    sentence_set_id: str
    candidates: list[BenchmarkCandidate] = Field(default_factory=list)
    records: list[GenerationRecord] = Field(default_factory=list)
    #: Reproducibility context (spec §23).
    git_commit: str | None = None
    config_hash: str | None = None
    base_seed: int = 0
    notes: str = Field(default="", max_length=2000)

    def succeeded_records(self) -> list[GenerationRecord]:
        return [r for r in self.records if r.succeeded]

    def failures(self) -> list[GenerationRecord]:
        return [r for r in self.records if not r.succeeded]

    def fallback_rate(self) -> float:
        """Share of generations that fell back to a different reference style.

        A high rate means the Reference Bank does not cover the test set's styles, and the
        benchmark is quietly measuring neutral delivery for everyone.
        """
        done = self.succeeded_records()
        if not done:
            return 0.0
        return round(sum(1 for r in done if r.reference_was_fallback) / len(done), 4)
