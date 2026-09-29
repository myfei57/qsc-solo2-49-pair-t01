"""Batch commits and crash recovery of the write-ahead journal."""

from __future__ import annotations

from pathlib import Path

import pytest

from linecontrol.bus import EventBus
from linecontrol.domain.records import Confirm, utcnow
from linecontrol.store.filestore import FileStore
from linecontrol.store.journal import (
    JOURNAL_DIR,
    QUARANTINE_SUFFIX,
    pending_commits,
    write_commit,
)
from linecontrol.weld.welder import Welder


class PowerCutStore(FileStore):
    """Dies partway through applying a committed batch."""

    def __init__(self, directory: Path, survive: int) -> None:
        self._survive = survive
        super().__init__(directory)

    def _save_locked(self, key: str, payload: bytes) -> None:
        if self._survive == 0:
            raise RuntimeError("power cut")
        self._survive -= 1
        super()._save_locked(key, payload)


def test_commit_applies_every_key_and_cleans_the_journal(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    batch = store.batch()
    batch.save("tension.current", b'{"value": 90.0}')
    batch.save("tension.gain", b"1.1")
    assert batch.commit() == 1
    assert store.load("tension.current") == b'{"value": 90.0}'
    assert store.load("tension.gain") == b"1.1"
    assert pending_commits(tmp_path / JOURNAL_DIR) == []


def test_staged_writes_are_invisible_until_commit(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    batch = store.batch()
    batch.save("a.key", b"1")
    assert store.load("a.key") is None
    batch.commit()
    assert store.load("a.key") == b"1"


def test_commit_sequences_increase(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    first = store.batch()
    first.save("a.key", b"1")
    second = store.batch()
    second.save("b.key", b"2")
    assert first.commit() == 1
    assert second.commit() == 2


def test_an_empty_batch_leaves_no_journal(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    assert store.batch().commit() is None
    assert not (tmp_path / JOURNAL_DIR).exists()


def test_a_discarded_batch_never_reaches_the_disk(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    batch = store.batch()
    batch.save("a.key", b"1")
    batch.discard()
    assert store.load("a.key") is None
    assert not (tmp_path / JOURNAL_DIR).exists()


def test_a_closed_batch_refuses_further_writes(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    batch = store.batch()
    batch.save("a.key", b"1")
    batch.commit()
    with pytest.raises(RuntimeError):
        batch.save("b.key", b"2")
    with pytest.raises(RuntimeError):
        batch.commit()


def test_batch_rejects_a_path_traversal_key(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    batch = store.batch()
    with pytest.raises(ValueError):
        batch.save("../escape", b"x")


def test_a_batch_never_committed_is_absent_after_a_restart(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    batch = store.batch()
    batch.save("a.key", b"1")
    batch.save("b.key", b"2")
    # The process dies here, before commit.
    reopened = FileStore(tmp_path)
    assert reopened.load("a.key") is None
    assert reopened.load("b.key") is None


def test_a_committed_batch_is_replayed_in_full_after_a_crash(tmp_path: Path) -> None:
    store = PowerCutStore(tmp_path, survive=1)
    batch = store.batch()
    batch.save("a.key", b"1")
    batch.save("b.key", b"2")
    with pytest.raises(RuntimeError):
        batch.commit()  # dies after applying only the first key
    assert pending_commits(tmp_path / JOURNAL_DIR) != []

    reopened = FileStore(tmp_path)
    assert reopened.load("a.key") == b"1"
    assert reopened.load("b.key") == b"2"
    assert pending_commits(tmp_path / JOURNAL_DIR) == []


def test_a_torn_staging_file_is_discarded_on_a_restart(tmp_path: Path) -> None:
    FileStore(tmp_path)
    journal = tmp_path / JOURNAL_DIR
    journal.mkdir(exist_ok=True)
    (journal / "00000000000000000009.wal.tmp").write_bytes(b'{"seq": 9, "ent')

    reopened = FileStore(tmp_path)
    assert reopened.list() == []
    assert list(journal.iterdir()) == []


def test_a_corrupt_commit_is_quarantined_not_replayed(tmp_path: Path) -> None:
    FileStore(tmp_path)
    journal = tmp_path / JOURNAL_DIR
    wal = write_commit(journal, 4, [("a.key", b"1")])
    raw = bytearray(wal.read_bytes())
    raw[10] ^= 0x01
    wal.write_bytes(bytes(raw))

    reopened = FileStore(tmp_path)
    assert reopened.load("a.key") is None
    assert pending_commits(journal) == []
    assert (journal / (wal.name + QUARANTINE_SUFFIX)).exists()


def test_pending_commits_replay_oldest_first(tmp_path: Path) -> None:
    FileStore(tmp_path)
    journal = tmp_path / JOURNAL_DIR
    write_commit(journal, 1, [("shared.key", b"old")])
    write_commit(journal, 2, [("shared.key", b"new")])

    reopened = FileStore(tmp_path)
    assert reopened.load("shared.key") == b"new"
    assert pending_commits(journal) == []


def test_splice_commits_its_two_records_as_one_batch(tmp_path: Path) -> None:
    class Aligner:
        def __call__(self) -> Confirm:
            return Confirm(ok=True, position=88.0, at=utcnow())

    welder = Welder(Aligner(), FileStore(tmp_path), EventBus())
    welder.set_line_speed(1.5)
    welder.set_power(24.0, 300.0)
    welder.gate_arc()
    welder.arc()
    record = welder.splice()

    assert welder.store.load("weld.record") is not None
    assert welder.store.load("weld.seam." + record.seam_id) is not None
    assert pending_commits(tmp_path / JOURNAL_DIR) == []
