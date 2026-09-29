"""Crash-safe key/value store rooted at a single directory.

Every mutation is committed in two phases:

1. **Journal first.** The whole batch (all its puts/deletes) is appended as
   one CRC-checked frame to ``journal.log`` and fsynced.  Until that frame is
   on platter the batch does not exist.
2. **Apply.** Each changed key is written through a temp file, fsynced and
   ``os.replace``d into place; the directory is fsynced afterwards.

A power cut can therefore only land in one of two states: the committed frame
is intact (the whole batch is replayed) or it is torn/absent (the batch is
ignored, exactly as if it had never started).  At start-up :class:`FileStore`
replays the journal, reconciles the key files with the committed state and
truncates any torn tail -- so readers only ever see complete batches.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path

from .backup import BackupSummary
from . import journal

#: Journal is compacted once it grows past this many bytes.
COMPACT_BYTES = 256 * 1024

#: Names that belong to the store machinery rather than key snapshots.
INTERNAL_NAMES = frozenset({journal.JOURNAL_NAME, journal.JOURNAL_TMP_NAME})


class _ThreadBatch:
    """The one batch currently open on a thread, plus its nesting depth."""

    __slots__ = ("batch", "depth")

    def __init__(self) -> None:
        self.batch: Batch | None = None
        self.depth = 0


class Batch:
    """Staging area for a batch.

    Writes made while a batch is active are buffered in memory and become
    visible atomically on :meth:`commit`; they are discarded on
    :meth:`rollback` or when the batch is dropped by an exception.
    """

    def __init__(self, store: "FileStore") -> None:
        self._store = store
        self._puts: dict[str, bytes] = {}
        self._deletes: set[str] = set()
        self._closed = False
        # Poisoned when any nested block fails; the outermost close then
        # discards everything staged by the whole business action.
        self._failed = False

    def put(self, key: str, payload: bytes) -> None:
        self._store._validate_key(key)
        self._puts[key] = bytes(payload)
        self._deletes.discard(key)

    def delete(self, key: str) -> None:
        self._store._validate_key(key)
        self._puts.pop(key, None)
        self._deletes.add(key)

    def load(self, key: str) -> bytes | None:
        self._store._validate_key(key)
        if key in self._puts:
            return self._puts[key]
        if key in self._deletes:
            return None
        return self._store._load_committed(key)

    def exists(self, key: str) -> bool:
        return self.load(key) is not None

    def list(self, prefix: str = "") -> list[str]:
        names = set(self._store._list_committed())
        names.update(self._puts)
        names.difference_update(self._deletes)
        if not prefix:
            return sorted(names)
        return sorted(name for name in names if name.startswith(prefix))

    def commit(self) -> None:
        """Commit the staged ops. Called automatically by the outermost
        ``with``; safe to call by hand on a detached batch."""

        if self._closed:
            return
        self._closed = True
        self._store._commit(self._puts, tuple(sorted(self._deletes)))

    def rollback(self) -> None:
        if self._closed:
            return
        self._closed = True

    # Used by FileStore.rotate(); rewrites one key from inside the batch.
    def _overwrite(self, key: str, payload: bytes) -> None:
        self._puts[key] = bytes(payload)
        self._deletes.discard(key)


class FileStore:
    """Key/value files whose batches are atomic across a power failure."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Re-entrant: a batch commit is performed while callers already hold
        # the lock, and compaction runs under the same lock.
        self._lock = threading.RLock()
        self._state: dict[str, bytes] = {}
        # Per-thread open batch plus its nesting depth; nested joins fold
        # into the outer batch so every write inside one business action
        # commits together.
        self._local = threading.local()
        self._journal = self._dir / journal.JOURNAL_NAME
        self._journal_fd = -1
        self._next_seq = 0
        self._recover()

    @property
    def directory(self) -> str:
        return str(self._dir)

    # ------------------------------------------------------------------ keys

    @staticmethod
    def _validate_key(key: str) -> None:
        if not key or "/" in key or "\\" in key or key in {".", ".."}:
            raise ValueError(f"invalid snapshot key: {key!r}")
        if key in INTERNAL_NAMES:
            raise ValueError(f"reserved snapshot key: {key!r}")

    def _path(self, key: str) -> Path:
        self._validate_key(key)
        return self._dir / key

    # --------------------------------------------------------------- batches

    def batch(self) -> Batch:
        """Open a batch as a context manager.

        >>> with store.batch() as b:
        ...     b.put("a", b"1")
        ...     b.put("b", b"2")
        ... # committed together on clean exit, discarded on exception

        Nested ``batch``/``with store`` blocks join the same batch; only the
        outermost block commits, and any inner failure discards the lot.
        """

        return self._open()

    def __enter__(self) -> Batch:
        return self._open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self._close(commit=exc_type is None)

    def _open(self) -> Batch:
        state = self._tls()
        if state.batch is None:
            state.batch = Batch(self)
            state.depth = 1
        else:
            state.depth += 1
        return state.batch

    def _close(self, commit: bool) -> None:
        state = self._tls()
        batch = state.batch
        if batch is None:
            return
        state.depth -= 1
        if state.depth > 0:
            # Still inside an outer business action: an inner failure poisons
            # the whole batch, otherwise the outer close decides.
            if not commit:
                batch._failed = True
            return
        state.batch = None
        state.depth = 0
        if commit and not batch._failed:
            batch.commit()
        else:
            batch.rollback()

    def _tls(self) -> "_ThreadBatch":
        state = getattr(self._local, "state", None)
        if state is None:
            state = _ThreadBatch()
            self._local.state = state
        return state

    def _current(self) -> Batch | None:
        return self._tls().batch

    # ------------------------------------------------------------ read/write

    def save(self, key: str, payload: bytes) -> None:
        """Put one key. Inside a batch it stages; otherwise it is its own
        single-key committed batch."""

        self._validate_key(key)
        batch = self._current()
        if batch is not None:
            batch.put(key, payload)
            return
        with self._lock:
            self._commit({key: bytes(payload)}, ())

    def put(self, key: str, payload: bytes) -> None:
        self.save(key, payload)

    def delete(self, key: str) -> None:
        self._validate_key(key)
        batch = self._current()
        if batch is not None:
            batch.delete(key)
            return
        with self._lock:
            self._commit({}, (key,))

    def load(self, key: str) -> bytes | None:
        self._validate_key(key)
        batch = self._current()
        if batch is not None:
            return batch.load(key)
        return self._load_committed(key)

    def exists(self, key: str) -> bool:
        return self.load(key) is not None

    def list(self, prefix: str = "") -> list[str]:
        batch = self._current()
        if batch is not None:
            return batch.list(prefix)
        with self._lock:
            return self._list_committed(prefix)

    def _load_committed(self, key: str) -> bytes | None:
        with self._lock:
            value = self._state.get(key)
            if value is not None:
                return value
        # Fall back to disk for values not held in the (compacted) cache: the
        # committed state and the files are reconciled at construction and
        # after every commit, so a present file is authoritative.
        try:
            return (self._dir / key).read_bytes()
        except FileNotFoundError:
            return None

    def _list_committed(self, prefix: str = "") -> list[str]:
        names = {
            entry.name
            for entry in self._dir.iterdir()
            if entry.is_file()
            and entry.name not in INTERNAL_NAMES
            and not entry.name.endswith(".tmp")
        }
        if not prefix:
            return sorted(names)
        return sorted(name for name in names if name.startswith(prefix))

    # --------------------------------------------------------------- commit

    def _commit(self, puts: dict[str, bytes], deletes: Sequence[str]) -> None:
        if not puts and not deletes:
            return
        with self._lock:
            seq = self._next_seq
            frame = journal.encode_frame(seq, puts, tuple(deletes))
            self._append_frame(frame)
            self._apply(puts, deletes)
            self._state.update(puts)
            for key in deletes:
                self._state.pop(key, None)
            self._next_seq = seq + 1
            self._maybe_compact()

    def _append_frame(self, frame: bytes) -> None:
        if self._journal_fd < 0:
            self._journal_fd = os.open(
                self._journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
        os.write(self._journal_fd, frame)
        os.fsync(self._journal_fd)

    def _apply(self, puts: dict[str, bytes], deletes: Sequence[str]) -> None:
        """Materialise one committed batch into the key files."""

        for key, payload in puts.items():
            self._write_one(key, payload)
        for key in deletes:
            try:
                (self._dir / key).unlink()
            except FileNotFoundError:
                pass
        journal.fsync_dir(self._dir)

    def _write_one(self, key: str, payload: bytes) -> None:
        path = self._dir / key
        # Unique temp name so a retried or concurrent rewrite can never
        # truncate another in-flight temp file.
        stamp = f"{os.getpid()}.{threading.get_ident()}.{id(payload)}"
        for attempt in range(8):
            tmp = path.with_name(f".{key}.{stamp}.{attempt}.tmp")
            try:
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError:
                continue
        else:  # pragma: no cover - practically impossible
            raise FileExistsError(f"could not create temp file for {key!r}")
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)

    # ------------------------------------------------------------- recovery

    def _recover(self) -> None:
        with self._lock:
            # First run on a store that predates the journal: adopt the loose
            # key files and immediately checkpoint them so they cannot be
            # dropped by a later replay that starts from an empty baseline.
            preexisting = not self._journal.exists()
            baseline = self._read_loose_files()
            if preexisting and baseline:
                frame = journal.encode_frame(0, baseline, (), marker=True)
                self._append_frame(frame)
                self._state = dict(baseline)
                self._next_seq = 1
                self._cleanup_temp_files()
                return

            replay = journal.replay_journal(self._journal, baseline)

            if replay.frames:
                # Drop the torn tail of the interrupted batch, if any.
                if self._journal.exists():
                    current_size = self._journal.stat().st_size
                    if current_size > replay.valid_bytes:
                        with open(self._journal, "r+b") as handle:
                            handle.truncate(replay.valid_bytes)
                            handle.flush()
                            os.fsync(handle.fileno())
                self._reconcile(replay.state)

            self._state = dict(replay.state)
            self._next_seq = max(replay.last_frame + 1, 0)
            self._cleanup_temp_files()

    def _read_loose_files(self) -> dict[str, bytes]:
        state: dict[str, bytes] = {}
        if not self._journal.exists():
            for name in self._list_committed():
                try:
                    state[name] = (self._dir / name).read_bytes()
                except OSError:
                    continue
        return state

    def _reconcile(self, committed: dict[str, bytes]) -> None:
        """Make the key files exactly match the committed journal state."""

        for name in self._list_committed():
            if name not in committed:
                (self._dir / name).unlink(missing_ok=True)
        for key, payload in committed.items():
            path = self._dir / key
            try:
                if path.exists() and path.read_bytes() == payload:
                    continue
            except OSError:
                pass
            self._write_one(key, payload)
        journal.fsync_dir(self._dir)

    def _cleanup_temp_files(self) -> None:
        for entry in self._dir.iterdir():
            if entry.is_file() and (
                entry.name.endswith(".tmp") or entry.name == journal.JOURNAL_TMP_NAME
            ):
                try:
                    entry.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------ compaction

    def _maybe_compact(self) -> None:
        try:
            size = self._journal.stat().st_size
        except FileNotFoundError:
            return
        if size < COMPACT_BYTES:
            return
        self._compact()

    def _compact(self) -> None:
        """Rewrite the journal as one checkpoint frame holding full state."""

        seq = self._next_seq
        frame = journal.encode_frame(seq, self._state, (), marker=True)
        tmp = self._dir / journal.JOURNAL_TMP_NAME
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, frame)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self._journal)
        journal.fsync_dir(self._dir)
        if self._journal_fd >= 0:
            os.close(self._journal_fd)
            self._journal_fd = -1
        self._next_seq = seq + 1

    # ----------------------------------------------------------- operations

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
        payload = json.dumps(items[dropped:], ensure_ascii=False).encode("utf-8")
        batch = self._current()
        if batch is not None:
            batch._overwrite(key, payload)
        else:
            self.save(key, payload)
        return dropped

    def backup(self) -> BackupSummary:
        """Copy every current snapshot into a ``backup`` subdirectory."""

        with self._lock:
            target = self._dir / "backup"
            target.mkdir(parents=True, exist_ok=True)
            copied: list[str] = []
            total = 0
            for name in self._list_committed():
                data = (self._dir / name).read_bytes()
                (target / name).write_bytes(data)
                copied.append(name)
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
