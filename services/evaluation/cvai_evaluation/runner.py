"""Benchmark runner (spec §17, §19).

Runs every enabled candidate over the same sentence set with the same reference bank and
the same seeds, and writes a fully traceable record for each generated file.

Design points that are easy to get wrong and expensive to discover late:

* **Same seed per (sentence, repeat) across candidates.** Two engines compared on
  different random draws are not being compared on their models.
* **Same reference clip per (sentence, repeat) across candidates**, for the same reason.
  Rotation varies clips *between* sentences, never *between* candidates on one sentence.
* **An unavailable engine does not abort the run.** Its sentences are recorded as
  failures with the reason, and the report shows the gap. A four-hour run that dies at
  minute ten because one sidecar was not started is a bad afternoon.
* **The run manifest is rewritten after every candidate**, so a crash still leaves a
  readable partial result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

from cvai_core.audio import read_wav_properties
from cvai_core.config import AppConfig
from cvai_core.errors import ProviderUnavailableError, SynthesisError, VoicePackError
from cvai_core.interfaces.tts import TTSProvider
from cvai_core.loaders import (
    load_sentence_set,
    open_voicepack,
    try_load_dataset_manifest,
    try_load_reference_bank,
)
from cvai_core.logging_setup import get_logger
from cvai_core.paths import VoicePackPaths, repo_root
from cvai_core.registry import build_tts_provider
from cvai_core.runlog import RunLogger, RunPaths, create_run
from cvai_reference_retrieval import RuleBasedReferenceRetriever
from cvai_types import (
    BenchmarkCandidate,
    BenchmarkRun,
    DatasetSplit,
    GenerationRecord,
    GroundTruthItem,
    StyleControls,
    TestSentence,
    TestSentenceSet,
    TTSRequest,
)

from .config import BenchmarkConfig
from .unprocessed import rebuild_all

log = get_logger(__name__)

ProviderFactory = Callable[[AppConfig, str], TTSProvider]


class BenchmarkRunner:
    """Execute one benchmark configuration."""

    def __init__(
        self,
        app_config: AppConfig,
        benchmark: BenchmarkConfig,
        *,
        runs_root: Path | None = None,
        packs_root: Path | None = None,
        provider_factory: ProviderFactory | None = None,
        run_id: str | None = None,
    ) -> None:
        self.app_config = app_config
        self.benchmark = benchmark
        root = repo_root()
        self.runs_root = Path(runs_root or (root / app_config.paths.runs))
        self.packs_root = Path(packs_root or (root / app_config.paths.voicepacks))
        self.provider_factory = provider_factory or (
            lambda config, name: build_tts_provider(config, name)
        )
        self.run_id = run_id

        self.paths: VoicePackPaths
        self.run_paths: RunPaths
        self.logger: RunLogger

    # -- setup ---------------------------------------------------------------------

    def _load_inputs(self) -> tuple[TestSentenceSet, RuleBasedReferenceRetriever]:
        self.paths, manifest = open_voicepack(
            self.benchmark.voicepack_id, self.packs_root
        )
        if (
            self.benchmark.voicepack_version
            and manifest.version != self.benchmark.voicepack_version
        ):
            raise VoicePackError(
                f"benchmark pins voice pack version "
                f"{self.benchmark.voicepack_version}, pack is {manifest.version}; "
                "bump the benchmark or check out the matching pack"
            )
        self.manifest = manifest

        bank = try_load_reference_bank(self.paths)
        if bank is None:
            raise VoicePackError(
                f"voice pack {self.benchmark.voicepack_id!r} has no reference bank; "
                "Milestone 2 builds it (metadata/references.json)"
            )
        retriever = RuleBasedReferenceRetriever.from_voicepack(
            self.paths,
            manifest,
            bank,
            enable_rotation=self.benchmark.rotate_references,
        )

        sentence_path = repo_root() / self.benchmark.sentence_set
        sentences = load_sentence_set(sentence_path)
        if not sentences.sentences:
            raise VoicePackError(f"sentence set {sentence_path} is empty")
        return sentences, retriever

    # -- execution -----------------------------------------------------------------

    async def run(self) -> BenchmarkRun:
        sentences, retriever = self._load_inputs()

        self.run_paths = create_run(
            self.runs_root, self.run_id, prefix=self.benchmark.benchmark_id
        )
        self.logger = RunLogger(self.run_paths)
        self.logger.snapshot_environment()
        self.logger.snapshot_config(self.app_config)
        environment = self.logger.paths.env_file

        run = BenchmarkRun(
            run_id=self.run_paths.root.name,
            voicepack_id=self.manifest.voicepack_id,
            voicepack_version=self.manifest.version,
            sentence_set_id=sentences.set_id,
            candidates=self.benchmark.enabled_candidates(),
            base_seed=self.benchmark.base_seed,
            config_hash=self.app_config.config_hash,
            notes=self.benchmark.description,
        )
        run.git_commit = _read_git_commit(environment)

        self.logger.event(
            "run.start",
            run_id=run.run_id,
            candidates=[c.candidate_id for c in run.candidates],
            sentences=len(sentences.sentences),
            repeats=self.benchmark.repeats,
            reference_coverage=retriever.coverage(),
        )

        for candidate in run.candidates:
            records = await self._run_candidate(candidate, sentences, retriever)
            run.records.extend(records)
            self.logger.write_run(run)

        ground_truth = self._collect_ground_truth()
        self.logger.event(
            "run.finish",
            generated=len(run.succeeded_records()),
            failed=len(run.failures()),
            fallback_rate=run.fallback_rate(),
            ground_truth_items=len(ground_truth),
        )
        self.logger.write_run(run)
        return run

    async def _run_candidate(
        self,
        candidate: BenchmarkCandidate,
        sentences: TestSentenceSet,
        retriever: RuleBasedReferenceRetriever,
    ) -> list[GenerationRecord]:
        records: list[GenerationRecord] = []
        out_dir = self.run_paths.candidate_audio_dir(candidate.candidate_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            provider = self.provider_factory(self.app_config, candidate.engine)
        except Exception as exc:
            return self._skip_candidate(candidate, sentences, f"provider build failed: {exc}")

        health = await provider.health()
        if not health.available:
            await provider.aclose()
            if not (self.benchmark.skip_unavailable and self.app_config.benchmark.skip_unavailable):
                raise ProviderUnavailableError(
                    f"{candidate.candidate_id}: {health.detail}"
                )
            return self._skip_candidate(candidate, sentences, health.detail)

        capabilities = provider.capabilities()
        if candidate.adaptation_mode not in capabilities.supported_adaptation_modes:
            await provider.aclose()
            return self._skip_candidate(
                candidate,
                sentences,
                f"engine does not support adaptation mode "
                f"{candidate.adaptation_mode.value!r}",
            )

        unprocessed: dict[str, Path] = {}
        if candidate.reference_source == "unprocessed":
            try:
                unprocessed = self._rebuild_unprocessed_references(candidate, retriever)
            except VoicePackError as exc:
                await provider.aclose()
                return self._skip_candidate(candidate, sentences, str(exc))

        self.logger.event(
            "candidate.start",
            candidate_id=candidate.candidate_id,
            engine=capabilities.engine,
            engine_version=capabilities.engine_version,
            adaptation_mode=candidate.adaptation_mode.value,
            checkpoint_id=candidate.checkpoint_id,
            license=capabilities.license,
        )

        try:
            if candidate.checkpoint_id and capabilities.supports_hot_checkpoint_swap:
                await provider.load_checkpoint(candidate.checkpoint_id)
            await provider.warmup()

            for index, sentence in enumerate(sentences.sentences):
                for repeat in range(self.benchmark.repeats):
                    records.append(
                        await self._generate_one(
                            provider,
                            candidate,
                            sentence,
                            index,
                            repeat,
                            retriever,
                            out_dir,
                            unprocessed,
                        )
                    )
        finally:
            await provider.aclose()

        ok = sum(1 for r in records if r.succeeded)
        self.logger.event(
            "candidate.finish",
            candidate_id=candidate.candidate_id,
            generated=ok,
            failed=len(records) - ok,
        )
        return records

    def _rebuild_unprocessed_references(
        self,
        candidate: BenchmarkCandidate,
        retriever: RuleBasedReferenceRetriever,
    ) -> dict[str, Path]:
        """Cut every reference out of the original recordings (decision D7).

        All or nothing: a control set that covers half the styles is not comparing the
        same thing as the candidate it controls for, and a half-control reads as a
        result rather than as a gap.
        """
        out_dir = self.run_paths.root / "unprocessed_references" / candidate.candidate_id
        rebuilt, problems = rebuild_all(retriever.bank.samples, self.paths, out_dir)
        if problems:
            raise VoicePackError(
                f"the null-processing control needs every reference rebuilt from its "
                f"original recording, and {len(problems)} could not be: {problems[0]}"
            )
        self.logger.event(
            "candidate.unprocessed_references",
            candidate_id=candidate.candidate_id,
            rebuilt=len(rebuilt),
        )
        return rebuilt

    async def _generate_one(
        self,
        provider: TTSProvider,
        candidate: BenchmarkCandidate,
        sentence: TestSentence,
        sentence_index: int,
        repeat: int,
        retriever: RuleBasedReferenceRetriever,
        out_dir: Path,
        unprocessed: dict[str, Path] | None = None,
    ) -> GenerationRecord:
        # Identical across candidates, so any difference in output is the engine.
        seed = self.benchmark.base_seed + sentence_index * 100 + repeat
        variation_key = f"{sentence.sentence_id}:{repeat}"

        controls = StyleControls(
            emotion=sentence.target_style,
            reference_style=sentence.target_style,
            emotion_intensity=0.6,
        )
        reference = retriever.select(controls, variation_key=variation_key)
        if unprocessed:
            # Same clip, same line, same style — just never cleaned. Swapping only the
            # audio keeps the transcript and the style metadata identical, so the
            # control differs from its partner candidate in exactly one thing.
            reference = reference.model_copy(
                update={"audio_path": str(unprocessed[reference.reference_id])}
            )

        suffix = f"_r{repeat}" if self.benchmark.repeats > 1 else ""
        output_path = out_dir / f"{sentence.sentence_id}{suffix}.wav"

        request = TTSRequest(
            request_id=f"{candidate.candidate_id}:{sentence.sentence_id}:{repeat}",
            text=sentence.text,
            controls=controls,
            reference=reference,
            checkpoint_id=candidate.checkpoint_id,
            adaptation_mode=candidate.adaptation_mode,
            seed=seed,
            output_sample_rate=self.benchmark.output_sample_rate
            or self.app_config.benchmark.output_sample_rate,
            engine_params=dict(candidate.engine_params),
        )

        record = GenerationRecord(
            record_id=request.request_id,
            candidate_id=candidate.candidate_id,
            sentence_id=sentence.sentence_id,
            text=sentence.text,
            engine=candidate.engine,
            adaptation_mode=candidate.adaptation_mode,
            checkpoint_id=candidate.checkpoint_id,
            voicepack_id=self.manifest.voicepack_id,
            voicepack_version=self.manifest.version,
            reference_id=reference.reference_id,
            reference_style=reference.style,
            reference_was_fallback=reference.was_fallback,
            reference_source=candidate.reference_source,
            seed=seed,
        )

        try:
            result = await provider.synthesize(request, output_path)
        except (SynthesisError, ProviderUnavailableError) as exc:
            record.succeeded = False
            record.error = f"{type(exc).__name__}: {exc}"
            self.logger.event(
                "generate.error",
                record_id=record.record_id,
                error=record.error,
            )
            return record

        record.audio_path = str(output_path.relative_to(self.run_paths.root))
        record.engine_version = result.engine_version
        record.resolved_params = result.resolved_params
        record.dropped_controls = result.dropped_controls
        record.sample_rate = result.sample_rate
        record.duration_s = result.duration_s
        record.latency_ms = result.latency_ms

        if result.dropped_controls:
            # Worth a line in the log: a control the engine ignores is a control the
            # listening test cannot attribute.
            self.logger.event(
                "generate.dropped_controls",
                record_id=record.record_id,
                dropped=result.dropped_controls,
            )
        self.logger.event(
            "generate.ok",
            record_id=record.record_id,
            duration_s=result.duration_s,
            latency_ms=result.latency_ms,
            reference_id=reference.reference_id,
            fallback=reference.was_fallback,
        )
        return record

    def _skip_candidate(
        self,
        candidate: BenchmarkCandidate,
        sentences: TestSentenceSet,
        reason: str,
    ) -> list[GenerationRecord]:
        log.warning("skipping candidate %s: %s", candidate.candidate_id, reason)
        self.logger.event(
            "candidate.skipped", candidate_id=candidate.candidate_id, reason=reason
        )
        return [
            GenerationRecord(
                record_id=f"{candidate.candidate_id}:{s.sentence_id}:0",
                candidate_id=candidate.candidate_id,
                sentence_id=s.sentence_id,
                text=s.text,
                engine=candidate.engine,
                adaptation_mode=candidate.adaptation_mode,
                checkpoint_id=candidate.checkpoint_id,
                voicepack_id=self.manifest.voicepack_id,
                voicepack_version=self.manifest.version,
                succeeded=False,
                error=f"candidate skipped: {reason}",
            )
            for s in sentences.sentences
        ]

    # -- ground truth --------------------------------------------------------------

    def _collect_ground_truth(self) -> list[GroundTruthItem]:
        """Copy held-out real recordings into the run as listening-test anchors."""
        config = self.benchmark.ground_truth
        if not config.enabled or config.source == "none":
            return []

        dataset = try_load_dataset_manifest(self.paths)
        if dataset is None:
            self.logger.event("ground_truth.skipped", reason="no dataset manifest")
            return []

        if config.source == "explicit":
            wanted = set(config.sample_ids)
            chosen = [s for s in dataset.samples if s.sample_id in wanted]
        else:
            chosen = [
                s
                for s in dataset.usable()
                if dataset.split_of(s.sample_id) is DatasetSplit.HELDOUT
            ]
        chosen = chosen[: config.max_items]

        if not chosen:
            self.logger.event(
                "ground_truth.empty",
                reason="no held-out samples; the blind test will have no real anchor",
            )
            return []

        out_dir = self.run_paths.candidate_audio_dir("ground_truth")
        out_dir.mkdir(parents=True, exist_ok=True)
        items: list[GroundTruthItem] = []
        for sample in chosen:
            source = self.paths.root / sample.audio_path
            if not source.is_file():
                continue
            destination = out_dir / f"{sample.sample_id}{source.suffix or '.wav'}"
            shutil.copy2(source, destination)
            items.append(
                GroundTruthItem(
                    item_id=sample.sample_id,
                    audio_path=str(destination.relative_to(self.run_paths.root)),
                    transcript=sample.transcript,
                    style=sample.voice_style,
                )
            )
        # Persisted next to the run manifest so the blind-test builder can pick the
        # anchors up without re-reading the voice pack.
        ground_truth_file = self.run_paths.root / "ground_truth.json"
        ground_truth_file.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in items],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.logger.event("ground_truth.collected", count=len(items))
        return items


def _read_git_commit(env_file: Path) -> str | None:
    try:
        data = json.loads(env_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover
        return None
    return data.get("git_commit")


def probe_generated_audio(run_paths: RunPaths, record: GenerationRecord) -> dict[str, float]:
    """Re-measure a generated file. Used by the objective-metrics stage in Milestone 6."""
    if not record.audio_path:
        return {}
    properties = read_wav_properties(run_paths.root / record.audio_path)
    return {
        "sample_rate": float(properties.sample_rate),
        "duration_s": properties.duration_s,
    }
