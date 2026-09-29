"""Batch atomicity and crash recovery for the file store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from linecontrol.store import journal
from linecontrol.store.filestore import FileStore
from linecontrol.store.keys import KEY_PUMP, KEY_TENSION


def test_batch_commits_all_keys_together(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    with store as batch:
        batch.put("a", b"1")
        batch.put("b", b"2")
    reopened = FileStore(tmp_path)
    assert reopened.load("a") == b"1"
    assert reopened.load("b") == b"2"


def test_save_outside_a_batch_is_a_single_key_batch(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    store.save("a", b"1")
    assert FileStore(tmp_path).load("a") == b"1"


def test_exception_discards_the_whole_batch(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    store.save("a", b"old")
    with pytest.raises(RuntimeError):
        with store:
            store.save("a", b"new")
            store.save("c", b"new")
            raise RuntimeError("boom")
    reopened = FileStore(tmp_path)
    assert reopened.load("a") == b"old"
    assert reopened.load("c") is None


def test_inner_failure_poisons_the_outer_batch(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    store.save("a", b"old")
    with pytest.raises(RuntimeError):
        with store:
            store.save("a", b"new")
            with store:
                store.save("b", b"new")
                raise RuntimeError("inner")
    reopened = FileStore(tmp_path)
    assert reopened.load("a") == b"old"
    assert reopened.load("b") is None


def test_torn_tail_is_ignored_on_reopen(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    with store as batch:
        batch.put("a", b"new")
        batch.put("b", b"new")
    torn = journal.encode_frame(99, {"a": b"torn"})
    with open(store.directory + "/" + journal.JOURNAL_NAME, "ab") as handle:
        handle.write(torn[: len(torn) // 2])
        handle.flush()
        os.fsync(handle.fileno())
    reopened = FileStore(tmp_path)
    assert reopened.load("a") == b"new"
    assert reopened.load("b") == b"new"
    # The torn tail is physically trimmed.
    with open(store.directory + "/" + journal.JOURNAL_NAME, "rb") as handle:
        assert torn[: len(torn) // 2] not in handle.read()


def test_corrupt_frame_keeps_the_previous_batch(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    store.save("k", b"v1")
    store.save("k", b"v2")
    good = Path(store.directory, journal.JOURNAL_NAME).read_bytes()
    bad = bytearray(journal.encode_frame(5, {"k": b"v3"}))
    bad[len(bad) - 3] ^= 0xFF
    Path(store.directory, journal.JOURNAL_NAME).write_bytes(good + bytes(bad))
    assert FileStore(tmp_path).load("k") == b"v2"


def test_reconcile_repairs_a_half_applied_batch(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    store.save("a", b"old")
    with store as batch:
        batch.put("a", b"new")
        batch.put("fresh", b"x")
    os.remove(store.directory + "/fresh")
    Path(store.directory, "obsolete").write_bytes(b"z")
    reopened = FileStore(tmp_path)
    assert reopened.load("fresh") == b"x"
    assert reopened.load("a") == b"new"
    assert reopened.load("obsolete") is None


def test_legacy_files_are_checkpointed_and_survive(tmp_path: Path) -> None:
    Path(tmp_path, "legacy").write_bytes(b"old")
    store = FileStore(tmp_path)
    store.save("new", b"n")
    reopened = FileStore(tmp_path)
    assert reopened.load("legacy") == b"old"
    assert reopened.load("new") == b"n"


def test_delete_is_atomic(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    store.save("a", b"1")
    store.delete("a")
    assert FileStore(tmp_path).load("a") is None


def test_concurrent_batches_recover(tmp_path: Path) -> None:
    import threading

    store = FileStore(tmp_path)

    def worker(n: int) -> None:
        for i in range(40):
            store.save(f"t{n}", f"{n}-{i}".encode())

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    reopened = FileStore(tmp_path)
    for n in range(6):
        assert reopened.load(f"t{n}") == f"{n}-39".encode()


def test_real_snapshot_pair_stays_consistent(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    payload = json.dumps({"value": 91.0, "gain": 1.15}).encode()
    with store:
        store.save(KEY_TENSION, payload)
        store.save(KEY_PUMP, b'{"running": true}')
    torn = journal.encode_frame(7, {KEY_TENSION: b"{}"})
    with open(store.directory + "/" + journal.JOURNAL_NAME, "ab") as handle:
        handle.write(torn[:8])
    reopened = FileStore(tmp_path)
    assert json.loads(reopened.load(KEY_TENSION))["value"] == pytest.approx(91.0)
    assert json.loads(reopened.load(KEY_PUMP))["running"] is True
