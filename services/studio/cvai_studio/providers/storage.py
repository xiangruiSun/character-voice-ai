"""File storage behind stable keys, never absolute paths in business logic.

Keys look like ``voicepacks/vp_ab12/raw/f3c9.wav``. ``LocalFilesystemStorage`` maps them
under the data directory; an S3-compatible store can implement the same interface.
``local_path`` exists because several tools (ffmpeg, the training engine) need a real
file: a remote store would materialise one in a cache directory.
"""

from __future__ import annotations

import abc
import shutil
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import BinaryIO


class StorageError(RuntimeError):
    pass


def _clean_key(key: str) -> str:
    path = PurePosixPath(key.replace("\\", "/"))
    if path.is_absolute() or any(part in ("..", "") for part in path.parts):
        raise StorageError(f"invalid storage key {key!r}")
    return path.as_posix()


class StorageProvider(abc.ABC):
    @abc.abstractmethod
    def save(self, key: str, data: bytes | BinaryIO) -> str: ...

    @abc.abstractmethod
    def local_path(self, key: str) -> Path: ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool: ...

    @abc.abstractmethod
    def delete(self, key: str) -> None: ...

    @abc.abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    @abc.abstractmethod
    def list(self, prefix: str) -> Iterator[str]: ...

    def copy_in(self, source: Path, key: str) -> str:
        with open(source, "rb") as handle:
            return self.save(key, handle)


class LocalFilesystemStorage(StorageProvider):
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def local_path(self, key: str) -> Path:
        path = (self.root / _clean_key(key)).resolve()
        if self.root not in path.parents and path != self.root:
            raise StorageError(f"storage key escapes the data directory: {key!r}")
        return path

    def save(self, key: str, data: bytes | BinaryIO) -> str:
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (bytes, bytearray)):
            path.write_bytes(data)
        else:
            with open(path, "wb") as out:
                shutil.copyfileobj(data, out, length=1024 * 1024)
        return _clean_key(key)

    def exists(self, key: str) -> bool:
        return self.local_path(key).exists()

    def delete(self, key: str) -> None:
        path = self.local_path(key)
        if path.is_file():
            path.unlink()

    def delete_prefix(self, prefix: str) -> None:
        path = self.local_path(prefix)
        if path.is_dir():
            shutil.rmtree(path)

    def list(self, prefix: str) -> Iterator[str]:
        base = self.local_path(prefix)
        if not base.is_dir():
            return iter(())
        return (p.relative_to(self.root).as_posix() for p in sorted(base.rglob("*")) if p.is_file())

    def key_for(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()
