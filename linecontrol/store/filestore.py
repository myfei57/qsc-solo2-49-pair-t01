"""Atomic file store rooted at a single directory."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path

from .backup import BackupSummary
from .journal import (
    JOURNAL_DIR,
    QUARANTINE_SUFFIX,
    discard_staging,
    fsync_dir,
    pending_commits,
    read_commit,
    write_atomic,
    write_commit,
)


class FileStore:
    """Key/value files written through a temporary file plus rename.

    Every write is synced to stable storage before its rename, so a
    snapshot survives a power cut as either the previous complete payload
    or the new complete payload, never a partial file.  Writes are
    serialised by a lock.  A group of writes that must appear together
    goes through :meth:`batch`, whose commit is journaled so the whole
    batch is replayed or discarded as one unit after a crash.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._recover()

    @property
    def directory(self) -> str:
        return str(self._dir)

    def _path(self, key: str) -> Path:
        if not key or "/" in key or "\\" in key or key in {".", ".."}:
            raise ValueError(f"invalid snapshot key: {key!r}")
        return self._dir / key

    def _journal(self) -> Path:
        return self._dir / JOURNAL_DIR

    def _recover(self) -> None:
        """Replay committed batches and drop whatever never committed."""

        journal = self._journal()
        if not journal.is_dir():
            return
        with self._lock:
            discard_staging(journal)
            touched = False
            for wal in pending_commits(journal):
                entries = read_commit(wal)
                if entries is None:
                    quarantine = wal.with_suffix(wal.suffix + QUARANTINE_SUFFIX)
                    os.replace(wal, quarantine)
                else:
                    for key, payload in entries:
                        self._save_locked(key, payload)
                    wal.unlink()
                touched = True
            if touched:
                fsync_dir(journal)

    def save(self, key: str, payload: bytes) -> None:
        with self._lock:
            self._save_locked(key, payload)

    def _save_locked(self, key: str, payload: bytes) -> None:
        write_atomic(self._path(key), payload)

    def load(self, key: str) -> bytes | None:
        path = self._path(key)
        with self._lock:
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None

    def exists(self, key: str) -> bool:
        path = self._path(key)
        with self._lock:
            return path.exists()

    def list(self, prefix: str = "") -> list[str]:
        with self._lock:
            names = sorted(entry.name for entry in self._dir.iterdir() if entry.is_file())
        if not prefix:
            return names
        return [name for name in names if name.startswith(prefix)]

    def batch(self) -> Batch:
        """Start a group of writes committed as one all-or-nothing unit."""

        return Batch(self)

    def _commit_locked(self, entries: list[tuple[str, bytes]]) -> int:
        self._seq += 1
        journal = self._journal()
        wal = write_commit(journal, self._seq, entries)
        for key, payload in entries:
            self._save_locked(key, payload)
        wal.unlink()
        fsync_dir(journal)
        return self._seq

    def rotate(self, key: str, keep: int) -> int:
        """Trim a JSON array snapshot to its newest ``keep`` entries."""

        raw = self.load(key)
        if raw is None or keep <= 0:
            return 0
        try:
            items = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 0
        if not isinstance(items, list) or len(items) <= keep:
            return 0
        dropped = len(items) - keep
        self.save(key, json.dumps(items[dropped:], ensure_ascii=False).encode("utf-8"))
        return dropped

    def backup(self) -> BackupSummary:
        """Copy every current snapshot into a ``backup`` subdirectory."""

        with self._lock:
            target = self._dir / "backup"
            target.mkdir(parents=True, exist_ok=True)
            copied: list[str] = []
            total = 0
            for entry in sorted(self._dir.iterdir()):
                if not entry.is_file():
                    continue
                data = entry.read_bytes()
                (target / entry.name).write_bytes(data)
                copied.append(entry.name)
                total += len(data)
        return BackupSummary(
            directory=str(target),
            files=tuple(copied),
            bytes_written=total,
        )

    def backup_files(self) -> Sequence[str]:
        target = self._dir / "backup"
        if not target.is_dir():
            return ()
        return tuple(sorted(entry.name for entry in target.iterdir() if entry.is_file()))


class Batch:
    """A group of snapshot writes committed as a single durable unit.

    Writes are staged in memory and only reach the disk on :meth:`commit`,
    which journals the whole group before applying it.  After a power cut
    the store replays the committed journal, so readers either see every
    write of the batch or none of them.
    """

    def __init__(self, store: FileStore) -> None:
        self._store = store
        self._entries: list[tuple[str, bytes]] = []
        self._closed = False

    def save(self, key: str, payload: bytes) -> None:
        """Stage a write; nothing reaches the disk before :meth:`commit`."""

        if self._closed:
            raise RuntimeError("batch is already closed")
        self._store._path(key)  # reject an invalid key before staging
        self._entries.append((key, bytes(payload)))

    def commit(self) -> int | None:
        """Journal and apply every staged write; returns the batch sequence."""

        if self._closed:
            raise RuntimeError("batch is already closed")
        self._closed = True
        if not self._entries:
            return None
        with self._store._lock:
            return self._store._commit_locked(self._entries)

    def discard(self) -> None:
        """Drop every staged write without touching the disk."""

        self._entries = []
        self._closed = True
