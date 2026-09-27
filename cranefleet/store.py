"""Key/value persistence for durable control state.

The control service keeps a small amount of state that must survive a restart:
the collision-zone snapshot, the hoist snapshot, the slew speed feedback, the
cluster baseline, the site configuration and the audit log. All of them go
through this thin key/value interface so the service can run on disk in
production and in memory during tests.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from .errors import SnapshotCorrupt, StoreError, StoreNotFound

SNAPSHOT_VERSION = 1


class Store(ABC):
    """A durable key/value store keyed by a slash-separated string."""

    @abstractmethod
    def save(self, key: str, value: bytes) -> None:
        """Persist ``value`` under ``key``, replacing any previous value."""

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Return the stored value or raise :class:`StoreNotFound`."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key``; deleting a missing key is not an error."""

    @abstractmethod
    def keys(self) -> list[str]:
        """Return every key currently held, in ascending order."""

    def snapshot(self) -> dict[str, bytes]:
        """Return a shallow copy of every stored key/value pair."""

        return {key: self.load(key) for key in self.keys()}

    def restore(self, snapshot: dict[str, bytes]) -> None:
        """Overwrite the store with the supplied snapshot."""

        for key, value in snapshot.items():
            self.save(key, value)


class FileStore(Store):
    """A store that maps each key to one file on disk."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._directory

    def _path(self, key: str) -> Path:
        return self._directory / (key.replace("/", "_") + ".bin")

    def save(self, key: str, value: bytes) -> None:
        target = self._path(key)
        with self._lock:
            temporary = target.with_suffix(".bin.tmp")
            try:
                temporary.write_bytes(bytes(value))
                os.replace(temporary, target)
            except OSError as exc:
                raise StoreError(f"cannot persist {key}") from exc

    def load(self, key: str) -> bytes:
        with self._lock:
            try:
                return self._path(key).read_bytes()
            except FileNotFoundError as exc:
                raise StoreNotFound(key) from exc
            except OSError as exc:
                raise StoreError(f"cannot read {key}") from exc

    def delete(self, key: str) -> None:
        with self._lock:
            try:
                self._path(key).unlink()
            except FileNotFoundError:
                return

    def keys(self) -> list[str]:
        with self._lock:
            names = sorted(entry.name for entry in self._directory.iterdir() if entry.name.endswith(".bin"))
        return [name[: -len(".bin")].replace("_", "/") for name in names]


class MemoryStore(Store):
    """An in-process store used for ephemeral runs and tests."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def save(self, key: str, value: bytes) -> None:
        with self._lock:
            self._data[key] = bytes(value)

    def load(self, key: str) -> bytes:
        with self._lock:
            try:
                return self._data[key]
            except KeyError as exc:
                raise StoreNotFound(key) from exc

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._data)


def encode_snapshot(snapshot: dict[str, bytes]) -> bytes:
    """Encode a snapshot into a single JSON document."""

    data = {key: base64.b64encode(value).decode("ascii") for key, value in snapshot.items()}
    payload = {
        "version": SNAPSHOT_VERSION,
        "data": data,
        "checksum": _checksum(data),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def decode_snapshot(encoded: bytes) -> dict[str, bytes]:
    """Decode and validate a snapshot produced by :func:`encode_snapshot`.

    Raises :class:`SnapshotCorrupt` when the payload does not parse or fails
    its integrity check, so the caller can quarantine the file and fall back
    to an earlier recovery point instead of crashing the boot.
    """

    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SnapshotCorrupt("snapshot is not valid UTF-8") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SnapshotCorrupt(f"snapshot is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SnapshotCorrupt("snapshot envelope is not a JSON object")
    version = payload.get("version")
    if version != SNAPSHOT_VERSION:
        raise SnapshotCorrupt(f"unsupported snapshot version {version!r}")
    data = payload.get("data")
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise SnapshotCorrupt("snapshot data is not a string map")
    checksum = payload.get("checksum")
    if checksum is not None and checksum != _checksum(data):
        raise SnapshotCorrupt("snapshot checksum mismatch")
    try:
        return {key: base64.b64decode(value, validate=True) for key, value in data.items()}
    except binascii.Error as exc:
        raise SnapshotCorrupt("snapshot payload is not valid base64") from exc


def write_snapshot(path: Path, payload: bytes) -> None:
    """Write the boot snapshot atomically so a power cut cannot tear it."""

    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except OSError as exc:
        raise StoreError(f"cannot persist snapshot {path}") from exc


def _checksum(data: dict[str, str]) -> str:
    """Digest the canonical form of the encoded payload map."""

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def store_size(store: Store) -> int:
    """Return the total number of payload bytes held by ``store``."""

    return sum(len(value) for value in store.snapshot().values())


def store_count(store: Store) -> int:
    """Return the number of keys held by ``store``."""

    return len(store.keys())
