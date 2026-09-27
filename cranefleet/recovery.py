"""Boot-time validation and recovery for the persisted state.

Two kinds of files hold durable state in the data directory: the per-key
``*.bin`` files written while the service runs, and the ``snapshot.json``
written at shutdown. A power cut can tear either kind of write, so boot must
not trust what it finds blindly. This module checks every persisted file
before anything consumes it, moves unreadable files aside instead of
deleting them, and reports which recovery point the service continues from.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .errors import StoreError
from .store import Store, decode_snapshot

LOGGER = logging.getLogger("cranefleet.recovery")

QUARANTINE_SUFFIX = ".corrupt-"


@dataclass(frozen=True)
class QuarantinedFile:
    """One persisted file that failed validation and was moved aside."""

    source: Path
    quarantine: Path
    reason: str


@dataclass(frozen=True)
class RecoveryReport:
    """The outcome of the boot-time check: what was bad and where it went."""

    snapshot_state: str  # "restored", "quarantined" or "absent"
    quarantined: tuple[QuarantinedFile, ...]
    stale_temporary: tuple[Path, ...]

    def summary(self) -> str:
        """Render one line naming each bad file and the recovery point in use."""

        details = "; ".join(
            f"{item.source.name} -> {item.quarantine.name} ({item.reason})"
            for item in self.quarantined
        )
        if self.snapshot_state == "quarantined":
            tail = "continuing from the per-key store state"
        elif self.snapshot_state == "restored":
            tail = "continuing from the boot snapshot"
        else:
            tail = "no persisted state found"
        return f"{details}; {tail}" if details else tail


def quarantine_file(path: Path, reason: str) -> QuarantinedFile:
    """Move ``path`` aside so boot never trips over it again.

    The file is renamed next to its original location with a UTC timestamp
    instead of being deleted, so the evidence stays available for inspection.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}{QUARANTINE_SUFFIX}{stamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}{QUARANTINE_SUFFIX}{stamp}-{counter}")
        counter += 1
    try:
        os.rename(path, candidate)
    except OSError as exc:
        raise StoreError(f"cannot quarantine {path.name}") from exc
    return QuarantinedFile(source=path, quarantine=candidate, reason=reason)


def inspect_directory(directory: Path) -> tuple[list[QuarantinedFile], list[Path]]:
    """Check every persisted payload file in ``directory``.

    Returns the files that had to be quarantined and the stale temporary
    files that were removed. Temporary files are incomplete atomic writes and
    are safe to delete. Every payload this service persists is a JSON
    document, so a ``*.bin`` file that does not parse is treated as torn by
    an interrupted write.
    """

    quarantined: list[QuarantinedFile] = []
    stale: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if entry.name.endswith(".tmp"):
            try:
                entry.unlink()
            except OSError:
                LOGGER.warning("cannot remove stale temporary file %s", entry)
                continue
            stale.append(entry)
        elif entry.suffix == ".bin":
            try:
                json.loads(entry.read_bytes().decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                quarantined.append(quarantine_file(entry, f"payload is not valid JSON: {exc}"))
    return quarantined, stale


def recover_store(store: Store, snapshot_path: Path) -> RecoveryReport:
    """Validate the persisted state and load the newest usable copy.

    The boot snapshot is the newest recovery point and is restored when it
    passes validation. When it does not, the file is quarantined — with the
    reason logged — and the service falls back to the per-key state already
    held by ``store``.
    """

    quarantined, stale = inspect_directory(snapshot_path.parent)
    snapshot_state = "absent"
    try:
        payload = snapshot_path.read_bytes()
    except FileNotFoundError:
        pass
    except OSError as exc:
        quarantined.append(quarantine_file(snapshot_path, f"cannot be read: {exc}"))
        snapshot_state = "quarantined"
    else:
        try:
            data = decode_snapshot(payload)
        except StoreError as exc:
            quarantined.append(quarantine_file(snapshot_path, str(exc)))
            snapshot_state = "quarantined"
        else:
            store.restore(data)
            snapshot_state = "restored"

    for path in stale:
        LOGGER.info("removed stale temporary file %s", path)
    for item in quarantined:
        LOGGER.warning(
            "quarantined corrupt state file %s -> %s (%s)",
            item.source,
            item.quarantine,
            item.reason,
        )
    if snapshot_state == "restored":
        LOGGER.info("recovered durable state from %s", snapshot_path)
    elif snapshot_state == "quarantined":
        LOGGER.warning("continuing from the per-key state in %s", snapshot_path.parent)
    return RecoveryReport(
        snapshot_state=snapshot_state,
        quarantined=tuple(quarantined),
        stale_temporary=tuple(stale),
    )
