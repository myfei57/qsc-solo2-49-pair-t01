"""Append-only commit journal.

Every accepted batch is one journal frame.  A frame is a single line of
canonical JSON carrying a CRC32 of its payload::

    {"seq": 7, "ops": [["put", "k", "BASE64..."], ["del", "k2"]], "crc": 123}

Durability protocol used by :class:`~linecontrol.store.filestore.FileStore`:

1. Append the frame, flush and fsync the journal.
2. Apply the ops to the key files (temp file + fsync + ``os.replace``).
3. fsync the directory so the renames themselves survive a power cut.

A batch is therefore "committed" iff its full frame is on platter.  Recovery
reads frames from the last checkpoint and stops at the first torn/corrupt
frame -- the batch the power cut interrupted is treated as never written.
"""

from __future__ import annotations

import base64
import json
import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path

#: Filename of the live journal inside the store directory.
JOURNAL_NAME = "journal.log"
#: Temporary name the journal is compacted into.
JOURNAL_TMP_NAME = "journal.log.tmp"

_MAGIC = "lcb1"


class JournalError(Exception):
    """Raised when the on-disk journal cannot be parsed at all."""


@dataclass(frozen=True, slots=True)
class Frame:
    """One committed batch."""

    seq: int
    ops: tuple[tuple[str, str, bytes], ...]
    marker: bool = False

    @property
    def puts(self) -> tuple[tuple[str, bytes], ...]:
        return tuple((key, value) for op, key, value in self.ops if op == "put")

    @property
    def deletes(self) -> tuple[str, ...]:
        return tuple(key for op, key, value in self.ops if op == "del")


def _encode_value(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_value(encoded: object) -> bytes:
    if not isinstance(encoded, str):
        raise JournalError("frame value is not a string")
    return base64.b64decode(encoded.encode("ascii"), validate=True)


def encode_frame(seq: int, puts: dict[str, bytes], deletes: tuple[str, ...] = (),
                 marker: bool = False) -> bytes:
    """Serialise a batch into one newline-terminated journal line."""

    ops: list[list[object]] = []
    for key in sorted(puts):
        ops.append(["put", key, _encode_value(puts[key])])
    for key in sorted(deletes):
        ops.append(["del", key, ""])
    if marker:
        ops.append(["mark", _MAGIC, ""])
    body = {"seq": int(seq), "ops": ops}
    body["crc"] = zlib.crc32(
        json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return (json.dumps(body, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def decode_frame(raw: bytes | str) -> Frame:
    """Parse one journal line; raise :class:`JournalError` if torn/corrupt."""

    line = raw if isinstance(raw, str) else None
    try:
        if line is None:
            line = raw.decode("utf-8")
        line = line.strip()
        if not line:
            raise JournalError("empty frame")
        doc = json.loads(line)
        crc = int(doc["crc"])
        body = {key: value for key, value in doc.items() if key != "crc"}
        actual = zlib.crc32(
            json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        if crc != actual:
            raise JournalError("frame checksum mismatch")
        seq = int(body["seq"])
        raw_ops = body["ops"]
        if not isinstance(raw_ops, list):
            raise JournalError("frame ops is not a list")
        ops: list[tuple[str, str, bytes]] = []
        marker = False
        for entry in raw_ops:
            if not isinstance(entry, list) or not entry:
                raise JournalError("bad op entry")
            kind = entry[0]
            if kind == "put":
                ops.append(("put", str(entry[1]), _decode_value(entry[2])))
            elif kind == "del":
                ops.append(("del", str(entry[1]), b""))
            elif kind == "mark":
                marker = True
            else:
                raise JournalError(f"unknown op {kind!r}")
        return Frame(seq=seq, ops=tuple(ops), marker=marker)
    except JournalError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
        raise JournalError(f"unparseable frame: {error}") from error


@dataclass(slots=True)
class Replay:
    """Result of scanning the journal at start-up."""

    state: dict[str, bytes] = field(default_factory=dict)
    next_seq: int = 0
    last_frame: int = -1
    valid_bytes: int = 0
    checkpoint_seq: int = -1
    frames: int = 0


def replay_journal(path: Path, baseline: dict[str, bytes]) -> Replay:
    """Fold every intact frame after the last checkpoint into ``baseline``.

    Scanning stops at the first torn or checksum-bad line; everything from
    that byte on belongs to the interrupted batch and is discarded.
    """

    result = Replay(state=dict(baseline))
    if not path.exists():
        return result
    data = path.read_bytes()
    offset = 0
    checkpoint_seq = -1
    state = dict(baseline)
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline == -1:
            break  # trailing partial write, no terminating newline
        raw = data[offset:newline]
        try:
            frame = decode_frame(raw)
        except JournalError:
            break
        end = newline + 1
        if frame.marker:
            # A marker frame carries the whole state at that sequence; any
            # frames before it are redundant after a compaction.
            checkpoint_seq = frame.seq
            state = _puts_map(frame)
        else:
            _apply_frame(state, frame)
        result.last_frame = frame.seq
        result.valid_bytes = end
        result.frames += 1
        offset = end
    result.state = state
    result.checkpoint_seq = checkpoint_seq
    result.next_seq = result.last_frame + 1
    return result


def _puts_map(frame: Frame) -> dict[str, bytes]:
    state: dict[str, bytes] = {}
    for key, value in frame.puts:
        state[key] = value
    return state


def _apply_frame(state: dict[str, bytes], frame: Frame) -> None:
    for key, value in frame.puts:
        state[key] = value
    for key in frame.deletes:
        state.pop(key, None)


def fsync_dir(directory: Path) -> None:
    """Force directory entry changes (renames, creates, unlinks) to media."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
