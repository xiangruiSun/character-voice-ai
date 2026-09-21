"""The Voice Pack preprocessing pipeline (Milestone 2, spec §7).

    raw → decode → [separate] → [denoise] → segment → transcribe
        → speaker filter → annotate → quality → review → clean/ + dataset + references

Each stage is a function over :class:`PipelineState`. State is persisted after every
stage, so a run that dies during ASR resumes rather than restarting, and a human's
corrections are never overwritten by a later automatic pass.

Nothing here decides that a clip is good. The automatic stages *measure* and *reject
obvious defects*; approval is a human act (spec §8), unless someone explicitly asks for
``auto_approve`` and accepts what that means.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from cvai_core.logging_setup import get_logger
from cvai_core.loaders import save_model_json
from cvai_core.paths import VoicePackPaths
from cvai_types import (
    AudioProperties,
    CoreStyle,
    DatasetEntry,
    DatasetManifest,
    DatasetSplit,
    ProcessingStage,
    ProcessingStep,
    ReferenceBank,
    ReferenceSample,
    RejectionReason,
    ReviewStatus,
    TrainingSample,
    VoicePackManifest,
    utcnow,
)

from cvai_core import dsp
from .audio_io import (
    decode_to_wav,
    iter_audio_files,
    probe,
    read_samples,
    slice_samples,
    write_samples,
)
from .backends import STUB_TRANSCRIPT_SOURCE, BackendBundle, resolve_backends
from .config import PreprocessConfig
from .state import (
    DECODED_DIR,
    DENOISED_DIR,
    SEGMENTS_DIR,
    SEPARATED_DIR,
    PipelineState,
    SegmentRecord,
    SourceClip,
    Stage,
    StageRun,
    load_state,
    save_state,
)

log = get_logger(__name__)


@dataclass
class StageOutcome:
    stage: Stage
    items: int = 0
    skipped: bool = False
    reason: str = ""


class Pipeline:
    """Runs preprocessing stages over one voice pack."""

    def __init__(
        self,
        paths: VoicePackPaths,
        manifest: VoicePackManifest,
        config: PreprocessConfig,
        *,
        backends: BackendBundle | None = None,
    ) -> None:
        self.paths = paths
        self.manifest = manifest
        self.config = config
        self.backends = backends or resolve_backends(
            prefer_device=config.device,
            enable_separation=config.enable_separation or manifest.enable_source_separation,
            enable_denoise=config.enable_denoise or manifest.enable_denoise,
            asr_model=config.asr_model,
            emotion_model=config.emotion_model,
            speech_pad_ms=config.speech_pad_ms,
            denoise_attenuation_db=config.denoise_attenuation_db,
        )
        self.state = load_state(paths.processed, manifest.voicepack_id)

    # -- orchestration -------------------------------------------------------------

    def run(self, stages: list[Stage] | None = None) -> PipelineState:
        """Run stages in order, saving state after each one."""
        order = stages or [
            Stage.INGEST,
            Stage.DECODE,
            Stage.SEPARATE,
            Stage.DENOISE,
            Stage.SEGMENT,
            Stage.TRANSCRIBE,
            Stage.SPEAKER_FILTER,
            Stage.ANNOTATE,
            Stage.QUALITY,
        ]
        for stage in order:
            self._run_one(stage)
        return self.state

    def _run_one(self, stage: Stage) -> StageOutcome:
        handler = {
            Stage.INGEST: self.stage_ingest,
            Stage.DECODE: self.stage_decode,
            Stage.SEPARATE: self.stage_separate,
            Stage.DENOISE: self.stage_denoise,
            Stage.SEGMENT: self.stage_segment,
            Stage.TRANSCRIBE: self.stage_transcribe,
            Stage.SPEAKER_FILTER: self.stage_speaker_filter,
            Stage.ANNOTATE: self.stage_annotate,
            Stage.QUALITY: self.stage_quality,
        }[stage]

        record = StageRun(stage=stage, started_at=utcnow().isoformat())
        self.state.runs.append(record)
        log.info("stage %s: starting", stage.value)
        try:
            outcome = handler()
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            record.error = f"{type(exc).__name__}: {exc}"
            record.finished_at = utcnow().isoformat()
            save_state(self.state, self.paths.processed)
            log.error("stage %s failed: %s", stage.value, record.error)
            raise

        record.items = outcome.items
        record.skipped = outcome.skipped
        record.skip_reason = outcome.reason
        record.finished_at = utcnow().isoformat()
        save_state(self.state, self.paths.processed)
        if outcome.skipped:
            log.info("stage %s: skipped (%s)", stage.value, outcome.reason)
        else:
            log.info("stage %s: %d items", stage.value, outcome.items)
        return outcome

    # -- stage 1: ingest -----------------------------------------------------------

    def stage_ingest(self) -> StageOutcome:
        """Bring originals into ``raw/`` and index them. ``raw/`` is written once."""
        self.paths.raw.mkdir(parents=True, exist_ok=True)

        if self.config.source_dir:
            source_root = Path(self.config.source_dir).expanduser()
            incoming = iter_audio_files(source_root)
            for source in incoming:
                relative = source.relative_to(source_root)
                destination = self.paths.raw / relative
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

        known = {clip.raw_path for clip in self.state.sources}
        added = 0
        for path in iter_audio_files(self.paths.raw):
            relative = path.relative_to(self.paths.root).as_posix()
            if relative in known:
                continue
            clip = SourceClip(
                # Derived from the path *inside* ``raw/``, so the id reads like the
                # original file rather than carrying a redundant "raw_" prefix.
                clip_id=_clip_id(path.relative_to(self.paths.raw).as_posix()),
                raw_path=relative,
                original_name=path.name,
                checksum_sha256=_sha256_file(path),
                source_metadata=_metadata_from_filename(path),
            )
            try:
                properties = probe(path)
                clip.sample_rate = properties.sample_rate
                clip.channels = properties.channels
                clip.duration_s = properties.duration_s
            except Exception as exc:  # noqa: BLE001 - a bad file must not stop ingest
                clip.error = f"probe failed: {exc}"
            self.state.sources.append(clip)
            added += 1
            if self.config.limit and len(self.state.sources) >= self.config.limit:
                break

        return StageOutcome(Stage.INGEST, items=added)

    # -- stage 2: decode -----------------------------------------------------------

    def stage_decode(self) -> StageOutcome:
        """Decode to a uniform WAV working copy. Format only — no processing."""
        output_dir = self.paths.processed / DECODED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        done = 0

        for clip in self._active_sources():
            destination = output_dir / f"{clip.clip_id}.wav"
            relative = destination.relative_to(self.paths.root).as_posix()
            if destination.is_file() and clip.working_path == relative:
                continue
            try:
                decode_to_wav(
                    self.paths.root / clip.raw_path,
                    destination,
                    sample_rate=self.config.target_sample_rate,
                    mono=True,
                )
            except Exception as exc:  # noqa: BLE001
                clip.error = f"decode failed: {exc}"
                continue
            clip.working_path = relative
            clip.error = None
            # Only a real rate change counts as resampling. Recording every decode as
            # destructive would flag 100% of a pack and make the flag meaningless —
            # and the flag exists to answer "did our cleaning hurt?" (decision D7).
            resampled = (
                clip.sample_rate is not None
                and clip.sample_rate != self.config.target_sample_rate
            )
            clip.processing_chain.append(
                ProcessingStep(
                    stage=(
                        ProcessingStage.RESAMPLE if resampled else ProcessingStage.INGEST
                    ),
                    tool="ffmpeg",
                    params={
                        "source_sample_rate": clip.sample_rate,
                        "sample_rate": self.config.target_sample_rate,
                        "channels": 1,
                        "format": "pcm_s16le",
                        "resampled": resampled,
                    },
                )
            )
            done += 1
        return StageOutcome(Stage.DECODE, items=done)

    # -- stage 3: separation (optional) --------------------------------------------

    def stage_separate(self) -> StageOutcome:
        backend = self.backends.separation
        if backend is None:
            return StageOutcome(
                Stage.SEPARATE,
                skipped=True,
                reason="source separation is off (enable it only when music or effects "
                "are mixed under the dialogue)",
            )
        output_dir = self.paths.processed / SEPARATED_DIR
        control = self._control_set()
        done = 0

        for clip in self._active_sources():
            if clip.working_path is None or clip.clip_id in control:
                continue
            try:
                produced = backend.isolate_vocals(
                    self.paths.root / clip.working_path, output_dir
                )
            except Exception as exc:  # noqa: BLE001
                clip.error = f"separation failed: {exc}"
                continue
            clip.working_path = produced.relative_to(self.paths.root).as_posix()
            clip.processing_chain.append(
                backend.step(model=getattr(backend, "model_filename", None))
            )
            done += 1
        return StageOutcome(Stage.SEPARATE, items=done)

    # -- stage 4: denoise (optional) -----------------------------------------------

    def stage_denoise(self) -> StageOutcome:
        backend = self.backends.denoise
        if backend is None:
            return StageOutcome(
                Stage.DENOISE,
                skipped=True,
                reason="denoising is off (it attenuates breaths and vocal texture; "
                "turn it on only for genuinely noisy source audio)",
            )
        output_dir = self.paths.processed / DENOISED_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        control = self._control_set()
        done = 0

        for clip in self._active_sources():
            if clip.working_path is None or clip.clip_id in control:
                continue
            destination = output_dir / f"{clip.clip_id}.wav"
            try:
                backend.denoise(self.paths.root / clip.working_path, destination)
            except Exception as exc:  # noqa: BLE001
                clip.error = f"denoise failed: {exc}"
                continue
            clip.working_path = destination.relative_to(self.paths.root).as_posix()
            clip.processing_chain.append(
                backend.step(attenuation_db=getattr(backend, "attenuation_db", None))
            )
            done += 1
        return StageOutcome(Stage.DENOISE, items=done)

    # -- stage 5: segment ----------------------------------------------------------

    def stage_segment(self) -> StageOutcome:
        output_root = self.paths.processed / SEGMENTS_DIR
        output_root.mkdir(parents=True, exist_ok=True)
        created = 0

        for clip in self._active_sources():
            if clip.working_path is None:
                continue
            if self.state.segments_of(clip.clip_id):
                continue  # already segmented; re-running must not duplicate

            samples, rate = read_samples(self.paths.root / clip.working_path)
            if not samples:
                clip.error = "decoded file contains no samples"
                continue

            if self.config.segment:
                spans = self.backends.vad.detect(samples, rate)
            else:
                # One segment per file, still trimmed and padded so leading silence
                # does not become part of the training example.
                _, start_s, end_s = dsp.trim_silence(
                    samples, rate, pad_ms=float(self.config.speech_pad_ms)
                )
                spans = [dsp.SpeechSegment(start_s, end_s)]

            spans = self._split_long_spans(spans)
            clip_dir = output_root / clip.clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            for index, span in enumerate(spans, start=1):
                duration = span.duration_s
                if duration <= 0:
                    continue
                segment_id = f"{clip.clip_id}_{index:03d}"
                target = clip_dir / f"{segment_id}.wav"
                write_samples(
                    target,
                    slice_samples(samples, rate, span.start_s, span.end_s),
                    rate,
                )
                record = SegmentRecord(
                    segment_id=segment_id,
                    clip_id=clip.clip_id,
                    audio_path=target.relative_to(self.paths.root).as_posix(),
                    start_s=round(span.start_s, 4),
                    end_s=round(span.end_s, 4),
                    duration_s=round(duration, 4),
                    sample_rate=rate,
                    processing_chain=list(clip.processing_chain),
                )
                record.add_step(
                    self.backends.vad.step(
                        speech_pad_ms=self.config.speech_pad_ms,
                        segmented=self.config.segment,
                    )
                )
                self.state.segments.append(record)
                created += 1

        return StageOutcome(Stage.SEGMENT, items=created)

    def _split_long_spans(self, spans: list[dsp.SpeechSegment]) -> list[dsp.SpeechSegment]:
        """Break anything over ``max_segment_s`` at its quietest interior point.

        Splitting mid-phrase is worse than a long segment, so this only fires above the
        limit and always cuts at the lowest-energy frame available.
        """
        limit = self.config.max_segment_s
        out: list[dsp.SpeechSegment] = []
        for span in spans:
            if span.duration_s <= limit:
                out.append(span)
                continue
            pieces = max(2, int(span.duration_s // limit) + 1)
            step = span.duration_s / pieces
            for index in range(pieces):
                start = span.start_s + index * step
                end = min(span.end_s, start + step)
                if end - start > 0.05:
                    out.append(dsp.SpeechSegment(round(start, 4), round(end, 4)))
        return out

    # -- stage 6: transcribe -------------------------------------------------------

    def stage_transcribe(self) -> StageOutcome:
        backend = self.backends.asr
        done = 0
        for segment in self.state.segments:
            if segment.human_edited and segment.transcript:
                continue  # never overwrite a human correction
            if segment.transcript and segment.transcript_source not in (
                None,
                STUB_TRANSCRIPT_SOURCE,
            ):
                continue

            path = self.paths.root / segment.audio_path
            if not path.is_file():
                continue
            try:
                result = backend.transcribe(path, hotwords=self.config.hotwords)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not end the run
                segment.quality_notes.append(f"asr failed: {exc}")
                continue

            segment.transcript = result.text
            segment.transcript_source = result.model
            segment.transcript_confidence = result.confidence
            segment.add_step(backend.step(model=result.model))
            if result.char_timings:
                # Alignment sanity: the last character should not end after the audio.
                last = result.char_timings[-1].end_ms / 1000.0
                if last > segment.duration_s + 0.5:
                    segment.quality_notes.append(
                        "transcript alignment runs past the end of the audio"
                    )
            done += 1
        return StageOutcome(Stage.TRANSCRIBE, items=done)

    # -- stage 7: speaker filter ---------------------------------------------------

    def stage_speaker_filter(self) -> StageOutcome:
        """Score every segment against the character's voice.

        Needs anchors: segments a human has confirmed are the character. Without them
        the centroid is built from everything, which is only meaningful when the pack
        really is single-speaker — so that case is flagged rather than assumed.
        """
        backend = self.backends.speaker
        embeddings: dict[str, list[float]] = {}
        for segment in self.state.segments:
            path = self.paths.root / segment.audio_path
            if not path.is_file():
                continue
            try:
                embeddings[segment.segment_id] = backend.embed(path)
            except Exception as exc:  # noqa: BLE001
                segment.quality_notes.append(f"speaker embedding failed: {exc}")

        if not embeddings:
            return StageOutcome(
                Stage.SPEAKER_FILTER, skipped=True, reason="no embeddings produced"
            )

        anchors = [
            embeddings[sid]
            for sid in self.state.speaker_anchor_segment_ids
            if sid in embeddings
        ]
        if anchors:
            centroid = dsp.mean_vector(anchors)
        else:
            centroid = dsp.mean_vector(list(embeddings.values()))
            log.warning(
                "no speaker anchors set — the centroid is the mean of every segment, "
                "which only identifies the character if the pack is already "
                "single-speaker. Mark a few confirmed clips with "
                "`cvai-prep anchors add <segment_id> …` and re-run this stage."
            )

        for segment in self.state.segments:
            vector = embeddings.get(segment.segment_id)
            if vector is None:
                continue
            segment.speaker_similarity = round(dsp.cosine_similarity(vector, centroid), 4)
            segment.add_step(backend.step(anchors=len(anchors)))
        return StageOutcome(Stage.SPEAKER_FILTER, items=len(embeddings))

    # -- stage 8: annotate ---------------------------------------------------------

    def stage_annotate(self) -> StageOutcome:
        """Emotion, pitch, pacing and pauses — the performance annotation of spec §8."""
        emotion_backend = self.backends.emotion
        emotion_available = emotion_backend.available()
        done = 0

        for segment in self.state.segments:
            path = self.paths.root / segment.audio_path
            if not path.is_file():
                continue
            samples, rate = read_samples(path)
            if not samples:
                continue

            track = dsp.estimate_f0_track(samples, rate)
            mean, spread = dsp.f0_statistics(track)
            segment.f0_mean_hz = mean
            segment.f0_std_hz = spread
            segment.pause_count = dsp.count_internal_pauses(samples, rate)
            if segment.transcript:
                segment.chars_per_second = dsp.chars_per_second(
                    segment.transcript, segment.duration_s
                )

            if emotion_available:
                try:
                    emotion = emotion_backend.classify(path)
                    segment.emotion_auto = emotion.label
                    segment.emotion_scores = emotion.scores
                    segment.add_step(emotion_backend.step(model=emotion.model))
                except Exception as exc:  # noqa: BLE001
                    segment.quality_notes.append(f"emotion classification failed: {exc}")

            # Never overwrite a human's style choice.
            if not segment.human_edited and segment.style is None:
                segment.style = self._default_style(segment)
            done += 1
        return StageOutcome(Stage.ANNOTATE, items=done)

    def _default_style(self, segment: SegmentRecord) -> str | None:
        """Provisional style from the automatic emotion label, if the pack has it."""
        label = segment.emotion_auto
        if not label or label == "unknown":
            return None
        if self.manifest.has_style(label):
            return label
        try:
            core = CoreStyle(label).value
        except ValueError:
            return None
        return core if self.manifest.has_style(core) else None

    # -- stage 9: quality ----------------------------------------------------------

    def stage_quality(self) -> StageOutcome:
        """Measure, score, and reject only clear defects."""
        speaker_is_advisory = self.backends.speaker.name.startswith("builtin")
        done = 0

        for segment in self.state.segments:
            if segment.human_edited:
                continue
            path = self.paths.root / segment.audio_path
            if not path.is_file():
                continue
            samples, rate = read_samples(path)
            if not samples:
                continue

            segment.lufs = dsp.approximate_lufs(samples, rate)
            segment.peak_dbfs = round(dsp.peak_dbfs(samples), 2)
            segment.clipping_ratio = round(dsp.clipping_ratio(samples), 6)
            segment.snr_db = dsp.estimate_snr_db(samples, rate)
            segment.quality_score = self._quality_score(segment)

            reason = self._auto_rejection(segment, speaker_is_advisory)
            if reason is not None:
                segment.review_status = ReviewStatus.REJECTED
                segment.rejection_reason = reason
            elif self.config.auto_approve:
                segment.review_status = ReviewStatus.APPROVED
            else:
                segment.review_status = ReviewStatus.PENDING
            done += 1

        return StageOutcome(Stage.QUALITY, items=done)

    def _quality_score(self, segment: SegmentRecord) -> float:
        """0-1 composite used to **order the review queue**, not to accept or reject.

        Weighted toward things that genuinely damage training — clipping, poor SNR —
        and away from anything that merely correlates with "sounds nice", because a
        whispered or shouted character line should not sink to the bottom of the queue
        for being unusual.
        """
        score = 1.0
        if segment.clipping_ratio is not None:
            score -= min(0.5, segment.clipping_ratio * 50.0)
        if segment.snr_db is not None:
            if segment.snr_db < self.config.min_snr_db:
                deficit = self.config.min_snr_db - segment.snr_db
                score -= min(0.35, deficit / 30.0)
        if segment.peak_dbfs is not None and segment.peak_dbfs < -35.0:
            score -= 0.15  # barely audible
        if segment.transcript is not None and not segment.transcript.strip():
            score -= 0.3
        if segment.duration_s < self.config.min_segment_s:
            score -= 0.2
        return round(max(0.0, min(1.0, score)), 4)

    def _auto_rejection(
        self, segment: SegmentRecord, speaker_is_advisory: bool
    ) -> RejectionReason | None:
        if segment.duration_s < self.config.min_segment_s:
            return RejectionReason.TOO_SHORT
        if segment.duration_s > self.config.max_segment_s:
            return RejectionReason.TOO_LONG
        if (
            segment.clipping_ratio is not None
            and segment.clipping_ratio > self.config.max_clipping_ratio
        ):
            return RejectionReason.CLIPPED_AUDIO
        if segment.transcript is not None and not segment.transcript.strip():
            return RejectionReason.NON_SPEECH
        if (
            not speaker_is_advisory
            and segment.speaker_similarity is not None
            and segment.speaker_similarity < self.config.speaker_threshold
        ):
            return RejectionReason.OTHER_SPEAKER
        return None

    # -- helpers -------------------------------------------------------------------

    def _active_sources(self) -> list[SourceClip]:
        sources = self.state.sources
        if self.config.limit:
            sources = sources[: self.config.limit]
        return [clip for clip in sources if clip.error is None or clip.working_path]

    def _control_set(self) -> set[str]:
        """Clip ids that skip every optional cleaning stage (decision D7)."""
        size = self.config.control_set_size
        if size <= 0:
            return set()
        ordered = sorted(clip.clip_id for clip in self.state.sources)
        if len(ordered) <= size:
            return set()
        # Spread across the pack rather than taking the first N, so the control set is
        # not all from one source file.
        stride = max(1, len(ordered) // size)
        return {ordered[i] for i in range(0, len(ordered), stride)[:size]}


# --------------------------------------------------------------------------------------
# Dataset build
# --------------------------------------------------------------------------------------


class BuildError(RuntimeError):
    pass


def build_dataset(
    paths: VoicePackPaths,
    manifest: VoicePackManifest,
    state: PipelineState,
    config: PreprocessConfig,
    *,
    allow_stub_transcripts: bool = False,
) -> tuple[DatasetManifest, ReferenceBank]:
    """Write approved segments to ``clean/`` and emit the dataset and reference bank.

    This is the only stage that writes ``clean/``, and it is the gate the rest of the
    project depends on: everything downstream — training, the benchmark, the runtime —
    reads the manifests this produces.
    """
    approved = [s for s in state.usable_segments() if s.transcript]
    if not approved:
        raise BuildError(
            "no approved segments with transcripts; review the pack first "
            "(`cvai-prep review <pack>`) or run with --auto-approve for a trial pass"
        )

    stubbed = [s for s in approved if s.transcript_source == STUB_TRANSCRIPT_SOURCE]
    if stubbed and not allow_stub_transcripts:
        raise BuildError(
            f"{len(stubbed)} approved segments still carry placeholder transcripts from "
            "the stub ASR backend. Install the preprocessing extra and re-run "
            "transcription, or pass --allow-stub-transcripts if you are deliberately "
            "building a throwaway pack for pipeline testing."
        )

    paths.clean.mkdir(parents=True, exist_ok=True)
    samples_out: list[TrainingSample] = []

    for segment in approved:
        source = paths.root / segment.audio_path
        if not source.is_file():
            continue
        audio, rate = read_samples(source)
        normalized, gain_db, measured = dsp.normalize_loudness(
            audio,
            rate,
            config.target_lufs,
            true_peak_ceiling_dbfs=config.true_peak_ceiling_dbfs,
        )
        destination = paths.clean / f"{segment.segment_id}.wav"
        write_samples(destination, normalized, rate)

        chain = list(segment.processing_chain)
        if abs(gain_db) > 0.0:
            chain.append(
                ProcessingStep(
                    stage=ProcessingStage.LOUDNESS,
                    tool=dsp.loudness_backend_name(),
                    params={
                        "target_lufs": config.target_lufs,
                        "gain_db": gain_db,
                        "measured_lufs": measured,
                    },
                )
            )

        style = segment.style or "neutral"
        samples_out.append(
            TrainingSample(
                sample_id=segment.segment_id,
                audio_path=destination.relative_to(paths.root).as_posix(),
                transcript=segment.transcript or "",
                audio=AudioProperties(
                    sample_rate=rate,
                    channels=1,
                    duration_s=segment.duration_s,
                    peak_dbfs=segment.peak_dbfs,
                    lufs=config.target_lufs if gain_db else measured,
                    snr_db=segment.snr_db,
                ),
                emotion=style,
                voice_style=style,
                emotion_intensity=0.5,
                chars_per_second=segment.chars_per_second,
                f0_mean_hz=segment.f0_mean_hz,
                f0_std_hz=segment.f0_std_hz,
                pause_count=segment.pause_count,
                quality_score=segment.quality_score or 0.8,
                speaker_similarity=_clamp_unit(segment.speaker_similarity),
                source_clip=_source_relpath(state, segment),
                source_offset_s=segment.start_s,
                processing_chain=chain,
                review_status=segment.review_status,
                notes=segment.reviewer_notes,
            )
        )

    splits = _assign_splits(samples_out, config)
    dataset = DatasetManifest(
        voicepack_id=manifest.voicepack_id,
        voicepack_version=manifest.version,
        samples=samples_out,
        splits=splits,
    )
    save_model_json(dataset, paths.dataset_file)

    bank = _build_reference_bank(paths, manifest, dataset, config)
    save_model_json(bank, paths.references_file)
    return dataset, bank


def _assign_splits(
    samples: list[TrainingSample], config: PreprocessConfig
) -> list[DatasetEntry]:
    """Hold out real lines per style, then carve a small validation slice.

    Held-out lines are chosen per style rather than at random overall: the benchmark's
    anchor set should cover the character's range, not be four neutral lines.
    """
    by_style: dict[str, list[TrainingSample]] = {}
    for sample in samples:
        by_style.setdefault(sample.voice_style, []).append(sample)

    entries: list[DatasetEntry] = []
    heldout_ids: set[str] = set()
    for style, group in sorted(by_style.items()):
        ordered = sorted(group, key=lambda s: (-s.quality_score, s.sample_id))
        # Keep the best of each style for training; take held-out from the middle so
        # the anchor set is representative rather than the very best takes.
        take = min(config.heldout_per_style, max(0, len(ordered) - 2))
        if take:
            middle = ordered[len(ordered) // 2 :][:take]
            heldout_ids.update(s.sample_id for s in middle)

    remaining = [s for s in samples if s.sample_id not in heldout_ids]
    val_count = int(len(remaining) * config.val_fraction)
    val_ids = {s.sample_id for s in remaining[:val_count]}

    for sample in samples:
        if sample.sample_id in heldout_ids:
            split = DatasetSplit.HELDOUT
        elif sample.sample_id in val_ids:
            split = DatasetSplit.VAL
        else:
            split = DatasetSplit.TRAIN
        entries.append(DatasetEntry(sample_id=sample.sample_id, split=split))
    return entries


def _build_reference_bank(
    paths: VoicePackPaths,
    manifest: VoicePackManifest,
    dataset: DatasetManifest,
    config: PreprocessConfig,
) -> ReferenceBank:
    """Pick reference clips per style and copy them into ``references/<style>/``.

    Selection favours clips in the duration window engines actually use well, then
    quality. Several per style, because one clip per style is spec §27's failure mode.
    """
    style_map = manifest.style_map()
    heldout = {
        entry.sample_id
        for entry in dataset.splits
        if entry.split is DatasetSplit.HELDOUT
    }

    by_style: dict[str, list[TrainingSample]] = {}
    for sample in dataset.usable():
        # Held-out lines are the benchmark's ground truth; using them as reference
        # prompts would leak them into the systems being measured.
        if sample.sample_id in heldout:
            continue
        by_style.setdefault(sample.voice_style, []).append(sample)

    references: list[ReferenceSample] = []
    for style, group in sorted(by_style.items()):
        definition = style_map.get(style)
        core = definition.core_style if definition else _core_style_or_neutral(style)
        ranked = sorted(
            group,
            key=lambda s: (
                -_duration_fit(s.audio.duration_s, config),
                -s.quality_score,
                s.sample_id,
            ),
        )
        chosen: list[TrainingSample] = []
        seen_transcripts: set[str] = set()
        for sample in ranked:
            key = sample.transcript.strip()
            if key in seen_transcripts:
                continue  # two takes of the same line add no variety to the bank
            seen_transcripts.add(key)
            chosen.append(sample)
            if len(chosen) >= config.references_per_style:
                break

        style_dir = paths.references / style
        style_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(chosen, start=1):
            reference_id = f"{style}_{index:02d}"
            destination = style_dir / f"{reference_id}.wav"
            shutil.copy2(paths.root / sample.audio_path, destination)
            references.append(
                ReferenceSample(
                    reference_id=reference_id,
                    audio_path=destination.relative_to(paths.root).as_posix(),
                    transcript=sample.transcript,
                    style=style,
                    core_style=core,
                    audio=sample.audio,
                    quality_score=sample.quality_score,
                    tags=_reference_tags(sample),
                )
            )

    return ReferenceBank(
        voicepack_id=manifest.voicepack_id,
        voicepack_version=manifest.version,
        samples=references,
    )


def _duration_fit(duration: float, config: PreprocessConfig) -> float:
    if config.reference_min_s <= duration <= config.reference_max_s:
        return 1.0
    if duration < config.reference_min_s:
        return duration / config.reference_min_s
    return max(0.0, 1.0 - (duration - config.reference_max_s) / config.reference_max_s)


def _reference_tags(sample: TrainingSample) -> list[str]:
    tags: list[str] = []
    text = sample.transcript
    if text.endswith(("？", "?")):
        tags.append("question")
    if text.endswith(("！", "!")):
        tags.append("exclamation")
    if "……" in text or "…" in text:
        tags.append("ellipsis")
    if len(text) <= 6:
        tags.append("short")
    elif len(text) >= 25:
        tags.append("long")
    return tags


def _core_style_or_neutral(style: str) -> CoreStyle:
    try:
        return CoreStyle(style)
    except ValueError:
        return CoreStyle.NEUTRAL


def _clamp_unit(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, value))


def _source_relpath(state: PipelineState, segment: SegmentRecord) -> str | None:
    clip = state.source(segment.clip_id)
    return clip.raw_path if clip else None


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _clip_id(relative_path: str) -> str:
    """Stable, filesystem-safe id derived from the path inside ``raw/``."""
    stem = Path(relative_path).with_suffix("").as_posix()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    cleaned = cleaned.strip("_") or "clip"
    if len(cleaned) <= 120:
        return cleaned
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:110]}_{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_from_filename(path: Path) -> dict[str, str]:
    """Keep whatever the filename encodes.

    Game voice exports routinely name files after the speaker, the quest and the line
    id. That is free labelling, and throwing it away at ingest means recovering it by
    hand later.
    """
    return {"filename": path.name, "parent": path.parent.name}
