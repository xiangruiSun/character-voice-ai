"""Voice models: the result of training, plus the reference bank that conditions it.

A voice model's directory is a voice pack in the conversation pipeline's format
(``metadata/references.json`` + ``references/<style>/``), so a chat session can load
it with the existing loaders, by the model's id.
"""

from __future__ import annotations

import time
import uuid
import wave
from pathlib import Path
from typing import Any

from cvai_core.paths import VoicePackPaths
from cvai_core.voicepack import scaffold_voicepack
from cvai_types import (
    AudioProperties,
    CoreStyle,
    ReferenceBank,
    ReferenceSample,
    SourceInfo,
    StyleDefinition,
    VoicePackManifest,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.logging import get_logger, with_ids
from ..db.models import AudioSample, Character, TrainingJob, VoiceModel, VoicePack
from ..domain.enums import VoiceModelStatus
from ..providers.training.base import TrainingFailure, TrainingResult, VoiceTrainingProvider
from .context import Conflict, NotFound, ServiceError, Studio

log = get_logger(__name__)
REF_MIN, REF_MAX, REF_IDEAL = 3.5, 9.5, 6.0     # GPT-SoVITS refuses prompts outside 3-10 s
REFS_PER_STYLE = 6
CORE_STYLES = {"neutral", "soft", "happy", "sad", "angry", "teasing", "serious",
               "surprised", "whisper", "excited"}

#: Preset test sentences, one per speaking style (Voice Studio's test step).
TEST_SENTENCES = [
    {"style": "neutral", "label": "平静", "text": "今天的天气还不错，我们出去走走吧。"},
    {"style": "soft", "label": "温柔", "text": "别担心……我会一直在这里陪着你的。"},
    {"style": "happy", "label": "开心", "text": "太好了！终于等到这一天了，我好开心！"},
    {"style": "serious", "label": "严肃", "text": "这件事非常重要，请你认真听我说完。"},
    {"style": "angry", "label": "生气", "text": "你怎么又这样！我已经提醒过你很多次了！"},
]


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001
        return 0.0


def build_reference_bank(studio: Studio, model_id: str, display_name: str,
                         samples: list[AudioSample]) -> tuple[str, list[str]]:
    """Pick per-style reference clips from approved samples; write a pack-format dir."""
    prefix = f"voice_models/{model_id}"
    paths = VoicePackPaths(studio.storage.local_path(prefix))
    by_style: dict[str, list[AudioSample]] = {}
    for sample in samples:
        if sample.approved and sample.transcript.strip():
            by_style.setdefault(sample.style or "neutral", []).append(sample)
    styles = sorted(by_style) or ["neutral"]
    if "neutral" not in styles:
        styles.insert(0, "neutral")
    scaffold_voicepack(paths, VoicePackManifest(
        voicepack_id=model_id, character_id=model_id, display_name=display_name,
        target_sample_rate=32000,
        styles=[StyleDefinition(name=s, core_style=CoreStyle(s if s in CORE_STYLES else "neutral"))
                for s in styles],
        source=SourceInfo(description="reference bank built by Character AI Studio"),
    ), overwrite=True)

    references: list[ReferenceSample] = []
    for style, items in by_style.items():
        candidates = [s for s in items if REF_MIN <= s.duration_s <= REF_MAX] or \
                     [s for s in items if 3.0 <= s.duration_s <= 10.0]
        candidates.sort(key=lambda s: (abs(s.duration_s - REF_IDEAL), -(s.quality_score or 0)))
        for sample in candidates[:REFS_PER_STYLE]:
            source = studio.storage.local_path(sample.audio_key)
            key = studio.storage.copy_in(source, f"{prefix}/references/{style}/{sample.id}.wav")
            references.append(ReferenceSample(
                reference_id=sample.id, audio_path=f"references/{style}/{sample.id}.wav",
                transcript=sample.transcript, style=style,
                core_style=CoreStyle(style if style in CORE_STYLES else "neutral"),
                audio=AudioProperties(sample_rate=32000, duration_s=round(sample.duration_s, 3)),
                quality_score=max(0.0, min(1.0, sample.quality_score or 0.8)),
            ))
            del key
    if not references:
        raise TrainingFailure("找不到合适的参考音频", hint="需要至少一个 3–10 秒、带文字的片段")
    paths.references_file.write_text(
        ReferenceBank(voicepack_id=model_id, voicepack_version="0.1.0",
                      samples=references).model_dump_json(indent=2), encoding="utf-8")
    return prefix, sorted({r.style for r in references})


class VoiceModelService:
    def __init__(self, studio: Studio, db: Session) -> None:
        self.studio = studio
        self.db = db

    def list(self, include_archived: bool = False) -> list[VoiceModel]:
        query = select(VoiceModel).order_by(VoiceModel.created_at.desc())
        if not include_archived:
            query = query.where(VoiceModel.status != VoiceModelStatus.ARCHIVED)
        return list(self.db.scalars(query))

    def get(self, model_id: str) -> VoiceModel:
        model = self.db.get(VoiceModel, model_id)
        if model is None:
            raise NotFound("声音模型不存在")
        return model

    def checkpoint_paths(self, model: VoiceModel) -> dict[str, Path]:
        return {role: self.studio.storage.local_path(key) for role, key in model.checkpoint.items()}

    def checkpoint_id(self, model: VoiceModel) -> str:
        files = self.checkpoint_paths(model)
        return f"{files['gpt']}|{files['sovits']}"

    def reference_root(self, model: VoiceModel) -> Path:
        return self.studio.storage.local_path(model.reference_prefix)

    def reference_for(self, model: VoiceModel, style: str) -> tuple[Path, str]:
        bank = ReferenceBank.model_validate_json(
            (VoicePackPaths(self.reference_root(model)).references_file).read_text(encoding="utf-8"))
        pool = bank.by_style(style) or bank.by_style("neutral") or bank.samples
        chosen = max(pool, key=lambda r: r.quality_score)
        return self.reference_root(model) / chosen.audio_path, chosen.transcript

    # -- creation -------------------------------------------------------------------------

    def create_from_training(self, job: TrainingJob, result: TrainingResult,
                             engine: VoiceTrainingProvider) -> VoiceModel:
        pack = self.db.get(VoicePack, job.voice_pack_id)
        version = (self.db.scalar(select(func.count()).select_from(VoiceModel)
                                  .where(VoiceModel.voice_pack_id == pack.id)) or 0) + 1
        model = VoiceModel(name=f"{pack.character_name} 声音 v{version}", voice_pack_id=pack.id,
                           training_job_id=job.id, engine=job.engine, base_model=result.base_model,
                           version=version, status=VoiceModelStatus.READY, reference_prefix="")
        self.db.add(model)
        self.db.flush()
        storage = self.studio.storage
        model.checkpoint = {
            role: storage.copy_in(path, f"voice_models/{model.id}/checkpoints/{role}{path.suffix}")
            for role, path in result.checkpoint_files.items()
        }
        model.reference_prefix, model.styles = build_reference_bank(
            self.studio, model.id, pack.character_name, list(pack.samples))
        # Hearing it is the evaluation that matters; numbers come later (A/B, SECS).
        try:
            ref_audio, ref_text = self.reference_for(model, "neutral")
            evaluation = engine.evaluate_checkpoint(
                self.checkpoint_paths(model), ref_audio, ref_text,
                storage.local_path(f"voice_models/{model.id}/evaluation"))
            model.evaluation_metadata = {
                "samples": [{"text": s["text"], "audio_key": storage.key_for(s["path"])}
                            for s in evaluation["samples"]],
                "metrics": result.metrics,
            }
        except TrainingFailure as exc:
            model.evaluation_metadata = {"skipped": exc.message, "metrics": result.metrics}
        for path in result.checkpoint_files.values():
            Path(path).unlink(missing_ok=True)         # the copies in storage are canonical
        with_ids(log, training_job_id=job.id).info("voice model %s created", model.id)
        return model

    # -- commands ----------------------------------------------------------------------------

    def update(self, model_id: str, data: dict[str, Any]) -> VoiceModel:
        model = self.get(model_id)
        if data.get("name"):
            model.name = str(data["name"]).strip()[:100]
        if data.get("approved") is not None:
            if data["approved"] and model.status is not VoiceModelStatus.READY:
                raise ServiceError("只有可用的声音模型可以通过审核")
            model.approved = bool(data["approved"])
        if data.get("archived") is True:
            model.status = VoiceModelStatus.ARCHIVED
        elif data.get("archived") is False and model.status is VoiceModelStatus.ARCHIVED:
            model.status = VoiceModelStatus.READY
        return model

    def delete(self, model_id: str) -> None:
        model = self.get(model_id)
        if self.db.scalar(select(Character.id).where(Character.voice_model_id == model_id).limit(1)):
            raise Conflict("仍有角色在使用这个声音", hint="请先为这些角色更换声音，或改为归档")
        self.studio.storage.delete_prefix(f"voice_models/{model.id}")
        self.db.delete(model)

    def synthesize(self, model_id: str, text: str, style: str = "neutral") -> dict[str, Any]:
        from .training import engines

        model = self.get(model_id)
        if model.status is not VoiceModelStatus.READY:
            raise ServiceError("这个声音模型当前不可用")
        text = text.strip()
        if not text:
            raise ServiceError("请输入要朗读的文字")
        if len(text) > 300:
            raise ServiceError("一次最多 300 个字")
        from cvai_text_normalizer import normalize

        engine = engines(self.studio)[model.engine]
        ref_audio, ref_text = self.reference_for(model, style)
        key = f"generated/audio/{uuid.uuid4().hex}.wav"
        started = time.perf_counter()
        try:
            path = engine.synthesize(normalize(text), checkpoint=self.checkpoint_paths(model),
                                     reference_audio=ref_audio, reference_text=ref_text,
                                     out_path=self.studio.storage.local_path(key))
        except TrainingFailure as exc:
            raise ServiceError(exc.message, hint=exc.hint or None) from exc
        return {"audio_key": key, "duration_s": round(_wav_seconds(path), 2),
                "generation_ms": round((time.perf_counter() - started) * 1000),
                "voice_model_id": model.id, "style": style}
