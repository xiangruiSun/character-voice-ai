"""Secrets: API keys are encrypted at rest and never leave the backend.

``SecretStore`` is the seam. ``FernetSecretStore`` (symmetric, key in an env var or a
git-ignored key file) is the local MVP; a hosted deployment can swap in a vault- or
KMS-backed store without touching the services that call ``encrypt``/``decrypt``.
"""

from __future__ import annotations

import abc
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class SecretError(RuntimeError):
    pass


class SecretStore(abc.ABC):
    @abc.abstractmethod
    def encrypt(self, plaintext: str) -> str: ...

    @abc.abstractmethod
    def decrypt(self, token: str) -> str: ...


class FernetSecretStore(SecretStore):
    def __init__(self, key: str | bytes) -> None:
        self._fernet = Fernet(key if isinstance(key, bytes) else key.encode())

    @classmethod
    def from_settings(cls, secret_key: str | None, data_dir: Path) -> "FernetSecretStore":
        if secret_key:
            return cls(secret_key)
        key_file = data_dir / "secret.key"
        if not key_file.is_file():
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_bytes(Fernet.generate_key())
            try:
                os.chmod(key_file, 0o600)
            except OSError:  # pragma: no cover - Windows ignores POSIX modes
                pass
        return cls(key_file.read_bytes().strip())

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise SecretError(
                "a stored API key could not be decrypted — the secret key changed; "
                "re-enter the key for this connection"
            ) from exc


def mask_secret(secret: str | None) -> str | None:
    """``sk-proj-abc…1234`` → ``sk-••••••••1234``: enough to recognise, not to use."""
    if not secret:
        return None
    tail = secret[-4:] if len(secret) > 8 else ""
    prefix = secret[:3] if secret.startswith(("sk-", "sk_")) else ""
    return f"{prefix}{'•' * 8}{tail}"
