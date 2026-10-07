"""Voice packs: upload, validate, prepare (via the existing preprocessing pipeline), review.

Preparation is deliberately conservative (decision D7 of the original plan): no
denoising or source separation unless asked, so breathing, texture and pauses — the
things that make a voice *this* character — survive. Automatic quality findings are
recorded as *potential* issues for a person to judge, never as verdicts.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from cvai_core.paths import VoicePackPaths
from cvai_core.voicepack import scaffold_voicepack
from cvai_types import CoreStyle, SourceInfo, StyleDefinition, VoicePackManifest
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger, with_ids
from ..db.models import AudioSample, VoicePack
from ..domain.enums import VoicePackStatus, transition
from .context import Conflict, NotFound, ServiceError, Studio

log = get_logger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
TRANSCRIPT_EXTENSIONS = {".lab", ".txt"}
MAX_ZIP_MEMBERS = 20000
MAX_ZIP_UNCOMPRESSED = 8 * 1024**3
MIN_APPROVED_CLIPS = 20
MIN_USABLE_SECONDS = 60.0
STYLES = {"neutral": "neutral", "soft": "soft", "happy": "happy", "sad": "sad",
          "angry": "angry", "serious": "serious", "excited": "excited", "surprised": "surprised",
          "teasing": "teasing", "whisper": "whisper"}


class VoicePackCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    character_name: str = Field(min_length=1, max_length=100)
    language: str = "zh-CN"
    description: str = Field(default="", max_length=2000)
    source: str = Field(default="", max_length=500)
    notes: str = Field(default="", max_length=2000)


class SamplePatch(BaseModel):
    transcript: str | None = Field(default=None, max_length=500)
    style: str | None = None
    approved: bool | None = None


@dataclass
class UploadResult:
    accepted: list[dict]
    rejected: list[dict]


def guess_style(transcript: str) -> str:
    """A first guess from punctuation, for a person to correct. Not a classifier."""
    text = transcript.strip()
    if text.endswith(("！", "!")):
        return "excited"
    if "……" in text or "..." in text:
        return "soft"
    return "neutral"


def _safe_name(name: str) -> str:
    stem = PurePosixPath(name.replace("\\", "/")).name
    return re.sub(r"[^\w.\-一-鿿]+", "_", stem)[:150] or "file"


class VoicePackService:
    def __init__(self, studio: Studio, db: Session) -> None:
        self.studio = studio
        self.db = db
        self.storage = studio.storage

    # -- queries ------------------------------------------------------------------

    def list(self) -> list[VoicePack]:
        return list(self.db.scalars(select(VoicePack).order_by(VoicePack.created_at.desc())))

    def get(self, pack_id: str) -> VoicePack:
        pack = self.db.get(VoicePack, pack_id)
        if pack is None:
            raise NotFound("声音包不存在")
        return pack

    def sample(self, sample_id: str) -> AudioSample:
        sample = self.db.get(AudioSample, sample_id)
        if sample is None:
            raise NotFound("片段不存在")
        return sample

    def paths(self, pack: VoicePack) -> VoicePackPaths:
        return VoicePackPaths(self.storage.local_path(pack.storage_prefix))

    def summary(self, pack: VoicePack) -> dict:
        samples = pack.samples
        approved = [s for s in samples if s.approved]
        issue_counts: dict[str, int] = {}
        for sample in samples:
            for issue in sample.issues or []:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        return {
            "uploaded_s": round(pack.total_duration_s, 1),
            "usable_s": round(sum(s.duration_s for s in approved), 1),
            "rejected_s": round(sum(s.duration_s for s in samples if not s.approved), 1),
            "clips": len(samples),
            "approved_clips": len(approved),
            "potential_issues": [{"issue": k, "count": v} for k, v in
                                 sorted(issue_counts.items(), key=lambda kv: -kv[1])],
            "ready_to_train": len(approved) >= MIN_APPROVED_CLIPS
            and sum(s.duration_s for s in approved) >= MIN_USABLE_SECONDS,
            "requirements": {"min_clips": MIN_APPROVED_CLIPS, "min_seconds": MIN_USABLE_SECONDS},
        }

    # -- lifecycle ------------------------------------------------------------------

    def create(self, data: VoicePackCreate) -> VoicePack:
        pack = VoicePack(name=data.name.strip(), character_name=data.character_name.strip(),
                         language=data.language, description=data.description,
                         source=data.source, notes=data.notes, status=VoicePackStatus.EMPTY,
                         storage_prefix="")
        self.db.add(pack)
        self.db.flush()
        pack.storage_prefix = f"voicepacks/{pack.id}"
        scaffold_voicepack(self.paths(pack), VoicePackManifest(
            voicepack_id=pack.id, character_id=pack.id, display_name=pack.character_name,
            target_sample_rate=32000,
            styles=[StyleDefinition(name=name, core_style=CoreStyle(core)) for name, core in STYLES.items()],
            source=SourceInfo(description=data.source or "uploaded in Character AI Studio",
                              license_note=data.notes or "see voice pack rights confirmation"),
        ))
        with_ids(log, voice_pack_id=pack.id).info("created voice pack %s", pack.name)
        return pack

    def update(self, pack_id: str, data: dict) -> VoicePack:
        pack = self.get(pack_id)
        for field in ("name", "character_name", "description", "source", "notes"):
            if data.get(field) is not None:
                setattr(pack, field, str(data[field]).strip())
        return pack

    def confirm_rights(self, pack_id: str) -> VoicePack:
        pack = self.get(pack_id)
        pack.rights_confirmed = True
        pack.rights_confirmed_at = datetime.now(timezone.utc)
        return pack

    def delete(self, pack_id: str) -> None:
        from ..db.models import TrainingJob, VoiceModel
        from ..domain.enums import training_is_active

        pack = self.get(pack_id)
        jobs = self.db.scalars(select(TrainingJob).where(TrainingJob.voice_pack_id == pack_id)).all()
        if any(training_is_active(job.status) for job in jobs):
            raise Conflict("这个声音包正在训练中", hint="请先取消训练")
        if self.db.scalar(select(VoiceModel.id).where(VoiceModel.voice_pack_id == pack_id).limit(1)):
            raise Conflict("已有声音模型由这个声音包训练而来", hint="请先删除或归档这些声音模型")
        for job in jobs:
            self.db.delete(job)
        self.storage.delete_prefix(pack.storage_prefix)
        self.db.delete(pack)

    # -- upload ------------------------------------------------------------------------

    def _uploads_index(self, pack: VoicePack) -> tuple[Path, dict]:
        path = self.paths(pack).metadata / "uploads.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"files": {}, "transcripts": {}}
        return path, data

    def add_file(self, pack_id: str, filename: str, stream: BinaryIO) -> UploadResult:
        """Store one uploaded file (audio, transcript, or a ZIP of either)."""
        pack = self.get(pack_id)
        if pack.status is VoicePackStatus.PROCESSING:
            raise Conflict("声音包正在处理，请稍候再上传")
        if pack.status is not VoicePackStatus.UPLOADING:
            pack.status = transition(pack.status, VoicePackStatus.UPLOADING)
        name = _safe_name(filename)
        suffix = Path(name).suffix.lower()
        limit = self.studio.settings.max_upload_mb * 1024 * 1024

        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / f"upload{suffix}"
            size = 0
            with open(staged, "wb") as out:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        return self._finish(pack, [], [{"name": name, "reason": f"文件超过 {self.studio.settings.max_upload_mb} MB"}])
                    out.write(chunk)
            if suffix == ".zip":
                return self._add_zip(pack, staged, name)
            accepted, rejected = [], []
            self._add_member(pack, name, staged, accepted, rejected)
            return self._finish(pack, accepted, rejected)

    def _add_zip(self, pack: VoicePack, archive: Path, archive_name: str) -> UploadResult:
        accepted, rejected = [], []
        try:
            with zipfile.ZipFile(archive) as zf:
                members = [m for m in zf.infolist() if not m.is_dir()]
                if len(members) > MAX_ZIP_MEMBERS:
                    raise ServiceError(f"压缩包内文件过多（超过 {MAX_ZIP_MEMBERS} 个）")
                if sum(m.file_size for m in members) > MAX_ZIP_UNCOMPRESSED:
                    raise ServiceError("压缩包解压后过大")
                with tempfile.TemporaryDirectory() as tmp:
                    for member in members:
                        # Never use the member path on disk: only its sanitized base name,
                        # extracted into our own temp file. No traversal, no overwrites.
                        name = _safe_name(member.filename)
                        if name.startswith((".", "__MACOSX")) or "__MACOSX" in member.filename:
                            continue
                        suffix = Path(name).suffix.lower()
                        if suffix not in AUDIO_EXTENSIONS | TRANSCRIPT_EXTENSIONS:
                            rejected.append({"name": name, "reason": "不支持的文件类型"})
                            continue
                        staged = Path(tmp) / f"{uuid.uuid4().hex}{suffix}"
                        with zf.open(member) as src, open(staged, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        self._add_member(pack, name, staged, accepted, rejected)
                        staged.unlink(missing_ok=True)
        except zipfile.BadZipFile:
            rejected.append({"name": archive_name, "reason": "压缩包已损坏或不是 ZIP 文件"})
        return self._finish(pack, accepted, rejected)

    def _add_member(self, pack: VoicePack, name: str, staged: Path,
                    accepted: list[dict], rejected: list[dict]) -> None:
        from cvai_voice_preprocessing.audio_io import probe

        suffix = Path(name).suffix.lower()
        index_path, index = self._uploads_index(pack)
        if suffix in TRANSCRIPT_EXTENSIONS:
            try:
                text = staged.read_text(encoding="utf-8-sig").strip()
            except UnicodeDecodeError:
                rejected.append({"name": name, "reason": "文本不是 UTF-8 编码"})
                return
            index["transcripts"][Path(name).stem] = text[:500]
            accepted.append({"name": name, "kind": "transcript"})
        elif suffix in AUDIO_EXTENSIONS:
            try:
                props = probe(staged)            # decodes the header: extension alone proves nothing
            except Exception:  # noqa: BLE001
                rejected.append({"name": name, "reason": "无法解码为音频"})
                return
            if props.duration_s < 0.3:
                rejected.append({"name": name, "reason": "音频过短"})
                return
            if props.duration_s > 3 * 3600:
                rejected.append({"name": name, "reason": "单个音频超过 3 小时"})
                return
            uid = uuid.uuid4().hex[:16]
            key = f"{pack.storage_prefix}/raw/{uid}{suffix}"
            self.storage.copy_in(staged, key)
            index["files"][uid] = {"name": name, "duration_s": props.duration_s,
                                   "bytes": staged.stat().st_size}
            accepted.append({"name": name, "kind": "audio", "duration_s": round(props.duration_s, 2)})
        else:
            rejected.append({"name": name, "reason": "不支持的文件类型"})
            return
        index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    def _finish(self, pack: VoicePack, accepted: list[dict], rejected: list[dict]) -> UploadResult:
        _, index = self._uploads_index(pack)
        files = index["files"].values()
        pack.total_files = len(index["files"])
        pack.total_duration_s = sum(f["duration_s"] for f in files)
        pack.total_bytes = sum(f["bytes"] for f in files)
        if rejected:
            pack.rejected_files = (pack.rejected_files or []) + rejected
        return UploadResult(accepted=accepted, rejected=rejected)

    # -- preparation (runs in the worker) ------------------------------------------------

    def mark_processing(self, pack_id: str) -> VoicePack:
        pack = self.get(pack_id)
        if pack.total_files == 0:
            raise ServiceError("还没有上传音频", hint="请先上传角色的语音文件")
        pack.status = transition(pack.status, VoicePackStatus.PROCESSING)
        pack.progress = {"stage": "queued", "label": "排队中"}
        pack.last_error = None
        return pack

    def prepare(self, pack_id: str, progress=None) -> VoicePack:
        """Run the pipeline and mirror its segments into AudioSample rows."""
        from cvai_voice_preprocessing import Pipeline, PreprocessConfig
        from cvai_voice_preprocessing.state import Stage, save_state

        pack = self.get(pack_id)
        paths = self.paths(pack)
        _, index = self._uploads_index(pack)
        transcripts: dict[str, str] = index.get("transcripts", {})
        names = {uid: info["name"] for uid, info in index["files"].items()}
        with_transcript = sum(Path(n).stem in transcripts for n in names.values())
        # Game-style datasets (one line per file, with .lab/.txt transcripts) are kept
        # whole; long free recordings are segmented and transcribed.
        one_line_per_file = names and with_transcript >= 0.8 * len(names)

        from cvai_core.loaders import load_voicepack_manifest

        pipeline = Pipeline(paths, load_voicepack_manifest(paths), PreprocessConfig(
            voicepack_id=pack.id, segment=not one_line_per_file, target_sample_rate=32000,
            min_segment_s=0.6, control_set_size=0, hotwords=[pack.character_name],
        ))

        def step(stage: str, label: str) -> None:
            # Reported through the callback's own transaction, never via this session:
            # this session may hold SQLite's write lock until the whole run commits.
            if progress:
                progress(pack.id, {"stage": stage, "label": label})

        step("decode", "正在检查并转换音频格式")
        pipeline.run([Stage.INGEST, Stage.DECODE])
        step("segment", "正在切分语音片段" if not one_line_per_file else "正在整理语音片段")
        pipeline.run([Stage.SEGMENT])
        if transcripts:
            by_clip = {}
            for clip in pipeline.state.sources:
                uid = Path(clip.raw_path).stem
                stem = Path(names.get(uid, "")).stem
                if stem in transcripts:
                    by_clip[clip.clip_id] = transcripts[stem]
            for segment in pipeline.state.segments:
                if one_line_per_file and segment.clip_id in by_clip and not segment.human_edited:
                    segment.transcript = by_clip[segment.clip_id]
                    segment.transcript_source = "provided"
            save_state(pipeline.state, paths.processed)
        step("transcribe", "正在识别中文语音")
        pipeline.run([Stage.TRANSCRIBE])
        step("quality", "正在分析音质")
        pipeline.run([Stage.QUALITY])

        self._sync_samples(pack, pipeline.state, names)
        pack.status = transition(pack.status, VoicePackStatus.NEEDS_REVIEW)
        pack.usable_duration_s = sum(s.duration_s for s in pack.samples if s.approved)
        pack.progress = {"stage": "done", "label": "处理完成"}
        return pack

    def _sync_samples(self, pack: VoicePack, state, names: dict[str, str]) -> None:
        existing = {s.audio_key: s for s in pack.samples}
        seen = set()
        for position, segment in enumerate(state.segments):
            key = f"{pack.storage_prefix}/{segment.audio_path}"
            seen.add(key)
            sample = existing.get(key) or AudioSample(voice_pack_id=pack.id, audio_key=key)
            sample.position = position
            source = next((c for c in state.sources if c.clip_id == segment.clip_id), None)
            sample.source_file = names.get(Path(source.raw_path).stem, "") if source else ""
            sample.duration_s = segment.duration_s
            sample.quality_score = segment.quality_score
            sample.issues = self._issues(segment)
            if not sample.human_edited:
                sample.transcript = (segment.transcript or "").strip()
                sample.transcript_source = segment.transcript_source or ""
                sample.style = guess_style(sample.transcript)
                rejected = segment.review_status.value == "rejected"
                sample.approved = not rejected and bool(sample.transcript)
                sample.rejection_reason = (
                    (segment.rejection_reason.value if segment.rejection_reason else "auto")
                    if rejected else (None if sample.transcript else "no_transcript"))
            sample.emotion = segment.emotion_auto
            if sample not in pack.samples:
                pack.samples.append(sample)
        for key, sample in existing.items():
            if key not in seen:
                pack.samples.remove(sample)

    @staticmethod
    def _issues(segment) -> list[str]:
        issues = []
        if segment.lufs is not None and segment.lufs < -38:
            issues.append("音量偏低")
        if segment.clipping_ratio is not None and segment.clipping_ratio > 0.001:
            issues.append("可能有爆音")
        if segment.snr_db is not None and segment.snr_db < 15:
            issues.append("可能有背景噪音或音乐")
        if segment.duration_s < 1.0:
            issues.append("片段过短")
        if segment.duration_s > 15.0:
            issues.append("片段过长")
        if not (segment.transcript or "").strip():
            issues.append("没有识别出文字")
        if segment.speaker_similarity is not None and segment.speaker_similarity < 0.5:
            issues.append("可能包含其他说话人")
        return issues

    def mark_failed(self, pack_id: str, error: str) -> None:
        pack = self.get(pack_id)
        pack.status = VoicePackStatus.ERROR
        pack.last_error = error[:2000]
        pack.progress = {"stage": "failed", "label": "处理失败"}

    # -- review ---------------------------------------------------------------------------

    def update_sample(self, sample_id: str, patch: SamplePatch) -> AudioSample:
        sample = self.sample(sample_id)
        pack = sample.voice_pack
        if pack.status is VoicePackStatus.PROCESSING:
            raise Conflict("声音包正在处理，请稍候")
        if patch.transcript is not None:
            sample.transcript = patch.transcript.strip()
            sample.transcript_source = "human"
        if patch.style is not None:
            if patch.style not in STYLES:
                raise ServiceError("未知的风格标签")
            sample.style = patch.style
        if patch.approved is not None:
            if patch.approved and not sample.transcript:
                raise ServiceError("没有文字的片段不能用于训练", hint="请先填写这段语音的文字")
            sample.approved = patch.approved
            sample.rejection_reason = None if patch.approved else "human"
        sample.human_edited = True
        pack.usable_duration_s = sum(s.duration_s for s in pack.samples if s.approved)
        if pack.status is VoicePackStatus.READY:
            pack.status = transition(pack.status, VoicePackStatus.NEEDS_REVIEW)
        return sample

    def approve_dataset(self, pack_id: str) -> VoicePack:
        pack = self.get(pack_id)
        if pack.status is not VoicePackStatus.NEEDS_REVIEW:
            raise Conflict("只有处理完成、等待审核的声音包可以确认")
        summary = self.summary(pack)
        if not summary["ready_to_train"]:
            raise ServiceError(
                f"可用数据不足：至少需要 {MIN_APPROVED_CLIPS} 个片段、{int(MIN_USABLE_SECONDS)} 秒",
                hint="请上传更多语音或重新纳入被排除的片段")
        pack.status = transition(pack.status, VoicePackStatus.READY)
        return pack
