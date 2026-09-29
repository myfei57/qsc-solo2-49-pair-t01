"""Write-ahead batch commits on top of the flat snapshot files.

A batch is a group of key/value writes that must become visible together.
The whole batch is first serialised into a single journal file, flushed to
stable storage and atomically renamed to ``<seq>.wal`` — that rename is the
commit point.  Only then is the batch applied to the flat snapshot files,
and finally the journal file is deleted.  After a power cut the recovery
pass deletes half-written staging files (the batch never happened) and
replays every committed journal that was not yet cleaned up, so a batch is
either fully visible or entirely absent.
"""

from __future__ import annotations

import json
import os
import zlib
from base64 import b64decode, b64encode
from pathlib import Path

#: Subdirectory of the snapshot root that holds the journal files.
JOURNAL_DIR = ".journal"

#: Suffix of a committed batch, ordered by its zero-padded sequence number.
COMMIT_SUFFIX = ".wal"

#: Suffix appended to a journal that failed validation during recovery.
QUARANTINE_SUFFIX = ".bad"


def fsync_dir(directory: Path) -> None:
    """Best-effort fsync of a directory so the renames inside it stay durable.

    Filesystems that do not support syncing a directory raise ``OSError``;
    that is tolerated here because the data blocks are already synced.
    """

    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` through a temporary file plus rename.

    The data is synced before the rename and the directory is synced
    afterwards, so a power cut exposes either the old or the new contents,
    never a half-written file.
    """

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def _canonical(doc: object) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _valid_key(key: object) -> bool:
    return (
        isinstance(key, str)
        and bool(key)
        and "/" not in key
        and "\\" not in key
        and key not in {".", ".."}
    )


def write_commit(journal: Path, seq: int, entries: list[tuple[str, bytes]]) -> Path:
    """Serialise a batch and commit it as ``<seq>.wal`` inside ``journal``."""

    journal.mkdir(parents=True, exist_ok=True)
    body = {
        "seq": seq,
        "entries": [
            {"key": key, "data": b64encode(payload).decode("ascii")}
            for key, payload in entries
        ],
    }
    doc = dict(body)
    doc["crc"] = zlib.crc32(_canonical(body))
    path = journal / f"{seq:020d}{COMMIT_SUFFIX}"
    write_atomic(path, _canonical(doc))
    return path


def read_commit(path: Path) -> list[tuple[str, bytes]] | None:
    """Decode a committed batch; ``None`` when the file fails validation."""

    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    crc = doc.get("crc")
    body = {key: value for key, value in doc.items() if key != "crc"}
    if not isinstance(crc, int) or zlib.crc32(_canonical(body)) != crc:
        return None
    seq = body.get("seq")
    entries = body.get("entries")
    if not isinstance(seq, int) or not isinstance(entries, list):
        return None
    try:
        if int(path.name[: -len(COMMIT_SUFFIX)]) != seq:
            return None
    except ValueError:
        return None
    decoded: list[tuple[str, bytes]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            return None
        key, data = entry.get("key"), entry.get("data")
        if not _valid_key(key) or not isinstance(data, str):
            return None
        try:
            decoded.append((key, b64decode(data, validate=True)))
        except ValueError:
            return None
    return decoded


def pending_commits(journal: Path) -> list[Path]:
    """Committed batches waiting to be replayed, oldest first."""

    if not journal.is_dir():
        return []
    return sorted(journal.glob(f"*{COMMIT_SUFFIX}"))


def discard_staging(journal: Path) -> int:
    """Drop half-written staging files; their batches were never committed."""

    if not journal.is_dir():
        return 0
    dropped = 0
    for tmp in journal.glob("*.tmp"):
        tmp.unlink()
        dropped += 1
    return dropped
