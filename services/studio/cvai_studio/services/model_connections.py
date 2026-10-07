"""Model connections: the user's own LLMs, local or hosted.

The only module that knows how a stored connection becomes a live ``LLMProvider``, and
the only place endpoint URLs are validated (self-hosted vs hosted rules).
"""

from __future__ import annotations

import ipaddress
import re
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from cvai_core.interfaces.llm import LLMProvider
from cvai_llm_providers import OllamaLLMProvider, OpenAILLMProvider
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger, redact, with_ids
from ..core.security import mask_secret
from ..db.models import AppSetting, Character, ModelConnection
from ..domain.enums import ConnectionKind, ConnectionMode, ConnectionStatus, ProviderType, transition
from .context import Conflict, NotFound, ServiceError, Studio

log = get_logger(__name__)
DEFAULT_KEY = "default_model_connection_id"


class GenerationDefaults(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=16, le=8192)
    stream: bool = True


class ConnectionDraft(BaseModel):
    """What a user enters. ``api_key`` is write-only: it is never echoed back."""

    name: str = Field(min_length=1, max_length=100)
    kind: ConnectionKind
    provider_type: ProviderType
    base_url: str = Field(min_length=1, max_length=500)
    model_name: str = Field(min_length=1, max_length=200)
    api_key: str | None = Field(default=None, max_length=500)
    generation_defaults: GenerationDefaults = Field(default_factory=GenerationDefaults)
    extra: dict[str, str] = Field(default_factory=dict)


class ConnectionPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    #: None = keep the stored key; "" = remove it; anything else = replace it.
    api_key: str | None = Field(default=None, max_length=500)
    generation_defaults: GenerationDefaults | None = None
    extra: dict[str, str] | None = None
    disabled: bool | None = None


# -- endpoint validation ------------------------------------------------------------


def _is_private(host: str) -> bool:
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        return False  # unresolvable now; the connection test will say so
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def validate_endpoint(url: str, studio: Studio) -> str:
    """Normalise a base URL and apply the deployment's network policy."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ServiceError("地址格式不正确", hint="请输入完整地址，例如 http://localhost:11434")
    if parsed.username or parsed.password:
        raise ServiceError("地址中不能包含账号密码", hint="API Key 请填写在单独的输入框中")
    settings = studio.settings
    if not (settings.self_hosted_mode and settings.allow_private_model_endpoints):
        if _is_private(parsed.hostname):
            raise ServiceError(
                "当前部署不允许连接内网或本机地址",
                hint="自托管部署可设置 SELF_HOSTED_MODE=true 与 ALLOW_PRIVATE_MODEL_ENDPOINTS=true",
            )
    return url.strip().rstrip("/")


# -- provider construction ------------------------------------------------------------


def build_provider(
    provider_type: ProviderType, base_url: str, model_name: str, api_key: str | None,
    defaults: dict[str, Any],
) -> LLMProvider:
    # On Windows "localhost" can resolve to ::1 first while Ollama/LM Studio listen on
    # IPv4 only. The user's URL is stored as typed; only the outgoing request changes.
    base_url = base_url.replace("://localhost", "://127.0.0.1")
    if provider_type is ProviderType.OLLAMA:
        return OllamaLLMProvider(
            model=model_name, base_url=base_url,
            temperature=defaults.get("temperature", 0.7),
            max_output_tokens=defaults.get("max_tokens", 400),
            think=defaults.get("think", True),
        )
    return OpenAILLMProvider(
        model=model_name, base_url=base_url, api_key=api_key or "not-needed",
        temperature=defaults.get("temperature", 0.7),
        max_output_tokens=defaults.get("max_tokens", 400),
    )


def friendly_error(error: str, provider_type: ProviderType, base_url: str, model: str) -> tuple[str, str]:
    """(message, hint) a user can act on, from whatever the provider raised."""
    text = error.lower()
    if "401" in text or "unauthorized" in text or "invalid api key" in text or "incorrect api key" in text:
        return "API Key 无效或已过期", "请检查 API Key 是否正确，或在服务商处重新生成"
    if "429" in text or "quota" in text or "credit" in text or "rate limit" in text:
        return "服务商拒绝了请求：额度不足或请求过于频繁", "请检查账户余额或稍后再试"
    if "not downloaded" in text or ("model" in text and ("not found" in text or "404" in text or "does not exist" in text)):
        if provider_type is ProviderType.OLLAMA:
            return f"本地还没有模型 {model}", f"在终端运行：ollama pull {model}"
        return f"服务商没有名为 {model} 的模型", "请从模型列表中选择，或检查模型名称拼写"
    # The provider answered, so it is reachable: it is just busy or down for now.
    if re.search(r"(error code|status|http)\D{0,6}5\d\d\b", text) or "service unavailable" in text \
            or "overloaded" in text or "internal server error" in text:
        return "服务商暂时不可用", "服务商繁忙或正在维护（不是你的设置问题），请稍后再测试"
    if "not running" in text or "connect" in text or "refused" in text or "unreachable" in text:
        return f"无法连接到 {base_url}", "请确认服务已启动，地址和端口正确"
    if "timed out" in text or "timeout" in text:
        return "连接超时", "服务可能正在加载模型，请稍后重试"
    return "连接失败", redact(error)[:300]


# -- service ---------------------------------------------------------------------------


class ModelConnectionService:
    def __init__(self, studio: Studio, db: Session) -> None:
        self.studio = studio
        self.db = db

    # queries
    def list(self) -> list[ModelConnection]:
        return list(self.db.scalars(select(ModelConnection).order_by(ModelConnection.created_at)))

    def get(self, connection_id: str) -> ModelConnection:
        connection = self.db.get(ModelConnection, connection_id)
        if connection is None:
            raise NotFound("模型连接不存在")
        return connection

    def default_id(self) -> str | None:
        setting = self.db.get(AppSetting, DEFAULT_KEY)
        return setting.value if setting else None

    def default(self) -> ModelConnection | None:
        connection_id = self.default_id()
        return self.db.get(ModelConnection, connection_id) if connection_id else None

    # commands
    def create(self, draft: ConnectionDraft) -> ModelConnection:
        connection = ModelConnection(
            name=draft.name.strip(), kind=draft.kind, provider_type=draft.provider_type,
            connection_mode=ConnectionMode.BACKEND_PROXY,
            base_url=validate_endpoint(draft.base_url, self.studio),
            model_name=draft.model_name.strip(),
            generation_defaults=draft.generation_defaults.model_dump(), extra=draft.extra,
            status=ConnectionStatus.DRAFT,
        )
        self._set_key(connection, draft.api_key)
        self.db.add(connection)
        self.db.flush()
        if self.default_id() is None:
            self.set_default(connection.id)        # the first connection is the default
        with_ids(log, model_connection_id=connection.id).info("created connection %s", connection.name)
        return connection

    def update(self, connection_id: str, patch: ConnectionPatch) -> ModelConnection:
        connection = self.get(connection_id)
        changed_target = False
        if patch.name is not None:
            connection.name = patch.name.strip()
        if patch.base_url is not None:
            connection.base_url = validate_endpoint(patch.base_url, self.studio)
            changed_target = True
        if patch.model_name is not None:
            connection.model_name = patch.model_name.strip()
            changed_target = True
        if patch.api_key is not None:
            self._set_key(connection, patch.api_key or None)
            changed_target = True
        if patch.generation_defaults is not None:
            connection.generation_defaults = patch.generation_defaults.model_dump()
        if patch.extra is not None:
            connection.extra = patch.extra
        if patch.disabled is True:
            connection.status = transition(connection.status, ConnectionStatus.DISABLED)
        elif patch.disabled is False and connection.status is ConnectionStatus.DISABLED:
            connection.status = transition(connection.status, ConnectionStatus.DRAFT)
        elif changed_target and connection.status is not ConnectionStatus.DRAFT:
            # Edited what it points at: the old test result no longer applies.
            connection.status = transition(connection.status, ConnectionStatus.DRAFT)
        return connection

    def delete(self, connection_id: str) -> None:
        connection = self.get(connection_id)
        in_use = self.db.scalar(select(Character.id).where(Character.model_connection_id == connection_id).limit(1))
        if in_use:
            raise Conflict("仍有角色在使用这个模型连接", hint="请先为这些角色更换模型")
        if self.default_id() == connection_id:
            self.db.delete(self.db.get(AppSetting, DEFAULT_KEY))
        self.db.delete(connection)

    def set_default(self, connection_id: str) -> None:
        self.get(connection_id)
        setting = self.db.get(AppSetting, DEFAULT_KEY) or AppSetting(key=DEFAULT_KEY)
        setting.value = connection_id
        self.db.add(setting)

    # providers
    def api_key(self, connection: ModelConnection) -> str | None:
        return self.studio.secrets.decrypt(connection.encrypted_api_key) if connection.encrypted_api_key else None

    def provider_for(self, connection: ModelConnection) -> LLMProvider:
        if connection.status is ConnectionStatus.DISABLED:
            raise ServiceError(f"模型连接「{connection.name}」已停用")
        return build_provider(connection.provider_type, connection.base_url, connection.model_name,
                              self.api_key(connection), connection.generation_defaults)

    async def test(self, connection_id: str) -> ModelConnection:
        connection = self.get(connection_id)
        connection.status = transition(connection.status, ConnectionStatus.TESTING)
        self.db.flush()
        provider = self.provider_for(connection)
        try:
            result = await provider.test_connection()
        finally:
            await provider.aclose()
        connection.last_tested_at = datetime.now(timezone.utc)
        if result["ok"]:
            connection.status = transition(connection.status, ConnectionStatus.CONNECTED)
            connection.latency_ms = result.get("latency_ms")
            connection.last_error = None
            connection.capabilities = provider.capabilities().model_dump()
        else:
            message, hint = friendly_error(result.get("error") or "", connection.provider_type,
                                           connection.base_url, connection.model_name)
            connection.status = transition(connection.status, ConnectionStatus.ERROR)
            connection.last_error = f"{message}｜{hint}"
            connection.latency_ms = None
        with_ids(log, model_connection_id=connection.id).info("tested: %s", connection.status.value)
        return connection

    async def probe(self, draft: ConnectionDraft) -> dict[str, Any]:
        """Test settings before saving them (the wizard's Test Connection button)."""
        base_url = validate_endpoint(draft.base_url, self.studio)
        provider = build_provider(draft.provider_type, base_url, draft.model_name,
                                  draft.api_key, draft.generation_defaults.model_dump())
        try:
            result = await provider.test_connection()
        finally:
            await provider.aclose()
        if result["ok"]:
            return {"ok": True, "latency_ms": result.get("latency_ms"), "message": "连接成功", "hint": None}
        message, hint = friendly_error(result.get("error") or "", draft.provider_type, base_url, draft.model_name)
        return {"ok": False, "latency_ms": None, "message": message, "hint": hint}

    async def list_models(self, provider_type: ProviderType, base_url: str, api_key: str | None) -> list[str]:
        provider = build_provider(provider_type, validate_endpoint(base_url, self.studio), "-", api_key, {})
        try:
            return await provider.list_models()
        except Exception as exc:  # noqa: BLE001
            message, hint = friendly_error(str(exc), provider_type, base_url, "")
            raise ServiceError(message, hint=hint) from exc
        finally:
            await provider.aclose()

    def _set_key(self, connection: ModelConnection, api_key: str | None) -> None:
        if api_key:
            connection.encrypted_api_key = self.studio.secrets.encrypt(api_key.strip())
            connection.api_key_hint = mask_secret(api_key.strip())
        else:
            connection.encrypted_api_key = None
            connection.api_key_hint = None
