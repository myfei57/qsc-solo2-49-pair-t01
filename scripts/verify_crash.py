"""Ad-hoc crash-simulation checks for batched journal commits.

Run: python3 scripts/verify_crash.py  (no pytest needed)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linecontrol.store.filestore import FileStore  # noqa: E402
from linecontrol.store import journal  # noqa: E402


def fresh() -> FileStore:
    d = tempfile.mkdtemp()
    return FileStore(d)


def check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(name)
    print(f"  ok  {name}")


def test_batch_round_trip() -> None:
    print("batch round trip")
    s = fresh()
    with s as b:
        b.put("a", b"1")
        b.put("b", b"2")
    check("both keys visible", s.load("a") == b"1" and s.load("b") == b"2")
    s2 = FileStore(s.directory)
    check("both keys survive reopen", s2.load("a") == b"1" and s2.load("b") == b"2")


def test_partial_batch_discarded() -> None:
    print("torn journal tail is discarded")
    s = fresh()
    s.save("a", b"old")
    s.save("b", b"old")
    # committed multi-key batch
    with s as b:
        b.put("a", b"new")
        b.put("b", b"new")
    check("new committed", s.load("a") == b"new" and s.load("b") == b"new")
    # append a torn frame (power cut mid-write)
    torn = journal.encode_frame(99, {"a": b"torn", "b": b"torn"})
    with open(os.path.join(s.directory, journal.JOURNAL_NAME), "ab") as f:
        f.write(torn[: len(torn) // 2])
        f.flush()
        os.fsync(f.fileno())
    s2 = FileStore(s.directory)
    check("torn batch ignored, previous batch intact",
          s2.load("a") == b"new" and s2.load("b") == b"new")


def test_corrupt_frame_stops_replay() -> None:
    print("checksum-bad frame stops replay at prior batch")
    s = fresh()
    s.save("k", b"v1")
    s.save("k", b"v2")
    good_frames = Path(os.path.join(s.directory, journal.JOURNAL_NAME)).read_bytes()
    bad = bytearray(journal.encode_frame(5, {"k": b"v3"}))
    bad[len(bad) - 3] ^= 0xFF  # flip a byte, breaks JSON/CRC
    with open(os.path.join(s.directory, journal.JOURNAL_NAME), "wb") as f:
        f.write(good_frames + bytes(bad))
    s2 = FileStore(s.directory)
    check("last good batch wins", s2.load("k") == b"v2")


def test_exception_rolls_back() -> None:
    print("exception inside with-block rolls batch back")
    s = fresh()
    s.save("a", b"old")
    try:
        with s:
            s.save("a", b"new")
            s.save("c", b"new")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("staged writes not visible", s.load("a") == b"old" and s.load("c") is None)
    s2 = FileStore(s.directory)
    check("nothing leaked to disk", s2.load("a") == b"old" and s2.load("c") is None)


def test_half_applied_files_reconciled() -> None:
    print("crash between journal commit and apply is reconciled")
    s = fresh()
    s.save("a", b"old")
    with s as b:
        b.put("a", b"new")
        b.put("fresh", b"x")
    # Simulate: journal has the frame, but apply only got as far as 'a' and
    # left a stale 'fresh' absent / an extra obsolete file behind.
    os.remove(os.path.join(s.directory, "fresh"))
    Path(os.path.join(s.directory, "obsolete")).write_bytes(b"z")
    s2 = FileStore(s.directory)
    check("missing committed key restored", s2.load("fresh") == b"x")
    check("committed value intact", s2.load("a") == b"new")
    check("uncommitted leftover removed", s2.load("obsolete") is None)


def test_legacy_loose_files_adopted() -> None:
    print("pre-journal loose files adopted as baseline")
    d = tempfile.mkdtemp()
    Path(d, "legacy.key").write_bytes(b"legacy")
    s = FileStore(d)
    check("legacy key readable", s.load("legacy.key") == b"legacy")
    s2 = FileStore(d)
    check("legacy key survives", s2.load("legacy.key") == b"legacy")


def test_legacy_files_survive_after_new_writes_and_restart() -> None:
    print("adopted legacy files still present after new batches + second restart")
    d = tempfile.mkdtemp()
    Path(d, "legacy.key").write_bytes(b"legacy")
    s = FileStore(d)          # adopts + checkpoints legacy
    s.save("new.key", b"new")  # one post-migration batch
    s2 = FileStore(d)         # restart: legacy must not be reconciled away
    check("legacy retained", s2.load("legacy.key") == b"legacy")
    check("new present", s2.load("new.key") == b"new")
    s3 = FileStore(d)
    check("legacy retained twice", s3.load("legacy.key") == b"legacy")


def test_compaction() -> None:
    print("journal compaction keeps full state")
    from linecontrol.store import filestore as fs

    old_limit = fs.COMPACT_BYTES
    fs.COMPACT_BYTES = 1024
    try:
        s = fresh()
        for i in range(200):
            s.save("k", f"value-{i}".encode())
        s.save("other", b"keep")
        size = Path(s.directory, journal.JOURNAL_NAME).stat().st_size
        check("journal compacted", size < 4096)
        s2 = FileStore(s.directory)
        check("latest value after compaction", s2.load("k") == b"value-199")
        check("other key retained", s2.load("other") == b"keep")
        s2.save("k", b"after")
        s3 = FileStore(s2.directory)
        check("post-compact writes replay", s3.load("k") == b"after")
    finally:
        fs.COMPACT_BYTES = old_limit


def test_delete() -> None:
    print("delete is part of the batch")
    s = fresh()
    s.save("a", b"1")
    s.delete("a")
    check("deleted in memory view", s.load("a") is None)
    s2 = FileStore(s.directory)
    check("delete survives reopen", s2.load("a") is None)


def test_nested_batch_poison() -> None:
    print("inner failure poisons the outer batch")
    s = fresh()
    s.save("a", b"old")
    try:
        with s:
            s.save("a", b"new")
            with s:
                s.save("b", b"new")
                raise RuntimeError("inner boom")
    except RuntimeError:
        pass
    check("outer also discarded", s.load("a") == b"old" and s.load("b") is None)


def test_multiple_batches_chain() -> None:
    print("successive batches each recover independently")
    s = fresh()
    with s:
        s.save("a", b"1")
    with s:
        s.save("b", b"2")
    with s:
        s.save("c", b"3")
    s2 = FileStore(s.directory)
    check("all three batches present",
          [s2.load(k) for k in ("a", "b", "c")] == [b"1", b"2", b"3"])


def test_json_payloads_intact() -> None:
    print("real json snapshots stay parseable through torn batches")
    s = fresh()
    payload = json.dumps({"value": 42.5, "seam_id": "sx"}).encode()
    with s:
        s.save("tension.current", payload)
        s.save("weld.record", payload)
    torn = journal.encode_frame(7, {"tension.current": b"{}"})
    with open(os.path.join(s.directory, journal.JOURNAL_NAME), "ab") as f:
        f.write(torn[:10])
    s2 = FileStore(s.directory)
    check("snapshot parses", json.loads(s2.load("tension.current"))["value"] == 42.5)
    check("weld snapshot parses", json.loads(s2.load("weld.record"))["value"] == 42.5)


def test_concurrent_batches_are_serialised() -> None:
    print("concurrent single-key writes serialise and recover")
    import threading

    s = fresh()

    def worker(n: int) -> None:
        for i in range(50):
            s.save(f"t{n}", f"{n}-{i}".encode())

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s2 = FileStore(s.directory)
    ok = all(s2.load(f"t{n}") == f"{n}-49".encode() for n in range(8))
    check("every thread final value recovered", ok)


def main() -> None:
    for fn in [
        test_batch_round_trip,
        test_partial_batch_discarded,
        test_corrupt_frame_stops_replay,
        test_exception_rolls_back,
        test_half_applied_files_reconciled,
        test_legacy_loose_files_adopted,
        test_legacy_files_survive_after_new_writes_and_restart,
        test_compaction,
        test_delete,
        test_nested_batch_poison,
        test_multiple_batches_chain,
        test_json_payloads_intact,
        test_concurrent_batches_are_serialised,
    ]:
        fn()
    print("\nALL CRASH TESTS PASSED")


if __name__ == "__main__":
    main()
