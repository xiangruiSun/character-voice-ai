"""First-run migration: bring the pre-Studio setup into the database, once.

Before the Studio, the working configuration lived in files: Qwen through Ollama
(``.env``), 卡提希娅's fine-tuned GPT-SoVITS weights, her reference bank under
``voicepacks/cartethyia_cn`` and her YAML profile. This copies those into a model
connection, a voice model and a character, so Chat keeps working with nothing re-done.
Idempotent: guarded by an ``app_settings`` flag.
"""

from __future__ import annotations

import json
import os
import shutil

from cvai_core.paths import repo_root
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..db.models import AppSetting, Character, ModelConnection, VoiceModel
from ..domain.enums import ConnectionKind, ConnectionStatus, ProviderType, VoiceModelStatus
from .context import Studio
from .model_connections import DEFAULT_KEY

log = get_logger(__name__)
FLAG = "seeded_v1"
LEGACY_CHARACTER = "cartethyia_cn"
LEGACY_WEIGHTS = {"gpt": "GPT_weights_v2ProPlus/cartethyia-e15.ckpt",
                  "sovits": "SoVITS_weights_v2ProPlus/cartethyia_e8_s360.pth"}


def seed(studio: Studio) -> list[str]:
    created: list[str] = []
    with studio.db() as db:
        if db.get(AppSetting, FLAG):
            return created
        connection = _seed_connection(db, created)
        voice = _seed_voice(studio, db, created)
        if voice is not None:
            _seed_character(db, voice, created)
        db.add(AppSetting(key=FLAG, value="1"))
        if connection and not db.get(AppSetting, DEFAULT_KEY):
            db.add(AppSetting(key=DEFAULT_KEY, value=connection.id))
    if created:
        log.info("imported existing setup: %s", ", ".join(created))
    return created


def _seed_connection(db: Session, created: list[str]) -> ModelConnection | None:
    if db.query(ModelConnection).count():
        return None
    model = os.environ.get("LLM_MODEL", "qwen3:4b")
    connection = ModelConnection(
        name="Qwen 本地", kind=ConnectionKind.LOCAL, provider_type=ProviderType.OLLAMA,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"), model_name=model,
        status=ConnectionStatus.DRAFT,
        generation_defaults={"temperature": 0.7, "max_tokens": 400, "stream": True},
    )
    db.add(connection)
    db.flush()
    created.append(f"model connection {connection.name}")
    return connection


def _seed_voice(studio: Studio, db: Session, created: list[str]) -> VoiceModel | None:
    engine_dir = studio.settings.gpt_sovits_dir
    weights = {role: engine_dir / rel for role, rel in LEGACY_WEIGHTS.items()}
    legacy_pack = repo_root() / "voicepacks" / LEGACY_CHARACTER
    if not all(p.is_file() for p in weights.values()) or not (legacy_pack / "metadata" / "references.json").is_file():
        return None
    voice = VoiceModel(name="卡提希娅 声音 v1", engine="gpt_sovits", base_model="v2ProPlus", version=1,
                       status=VoiceModelStatus.READY, approved=True, reference_prefix="")
    db.add(voice)
    db.flush()
    prefix = f"voice_models/{voice.id}"
    voice.checkpoint = {role: studio.storage.copy_in(path, f"{prefix}/checkpoints/{role}{path.suffix}")
                        for role, path in weights.items()}
    target = studio.storage.local_path(prefix)
    shutil.copytree(legacy_pack / "references", target / "references", dirs_exist_ok=True)
    (target / "metadata").mkdir(parents=True, exist_ok=True)
    for name in ("voicepack.yaml", "references.json"):
        text = (legacy_pack / "metadata" / name).read_text(encoding="utf-8")
        if name == "references.json":
            data = json.loads(text)
            data["voicepack_id"] = voice.id
            text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            text = text.replace(f"voicepack_id: {LEGACY_CHARACTER}", f"voicepack_id: {voice.id}", 1)
        (target / "metadata" / name).write_text(text, encoding="utf-8")
    bank = json.loads((target / "metadata" / "references.json").read_text(encoding="utf-8"))
    voice.reference_prefix = prefix
    voice.styles = sorted({s["style"] for s in bank["samples"]})
    voice.evaluation_metadata = {"note": "imported from the pre-Studio setup (fine-tuned on 355 lines)"}
    created.append(f"voice model {voice.name}")
    return voice


def _seed_character(db: Session, voice: VoiceModel, created: list[str]) -> None:
    if db.query(Character).count():
        return
    character = Character(
        name="卡提希娅", description="《鸣潮》中曾被加冕为圣女的少女，温和而坚定，信条是“贯彻本心”。",
        system_prompt="", greeting="你好，我是卡提希娅。今天想聊些什么呢？",
        voice_model_id=voice.id, default_voice_style="neutral",
        legacy_profile_id=LEGACY_CHARACTER,
    )
    db.add(character)
    created.append(f"character {character.name}")
