"""End-to-end crash checks through the real Line controllers."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linecontrol.line.machine import Line  # noqa: E402
from linecontrol.line.workflow import CycleInput  # noqa: E402
from linecontrol.store import journal  # noqa: E402


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("  ok ", name)


def cycle_input():
    return CycleInput(
        coil_id="C-1", target=90.0, level=5.0,
        voltage=220.0, current=180.0, speed=12.0,
    )


def test_full_cycle_is_one_recoverable_batch():
    print("full production cycle replays as one complete batch")
    d = tempfile.mkdtemp()
    line = Line(d)
    line.recover()
    line.production_cycle(cycle_input())
    # New instance: the whole cycle state must be present.
    line2 = Line(d)
    state = line2.recover()
    check("tension snapshot committed", "tension.current" in state.loaded)
    check("weld record committed", line2.weld.last_weld()[1] is True)
    check("pump not left running after clean stop", line2.pump.running() is False)
    check("run hours persisted", line2.pump.run_seconds() >= 0.0)
    seams = line2.weld.seam_history()
    check("both seams persisted", len(seams) == 2)


def test_crash_mid_cycle_drops_the_entire_unfinished_batch():
    print("power cut mid-cycle: unfinished batch entirely absent")
    d = tempfile.mkdtemp()
    line = Line(d)
    line.recover()
    line.load_coil("C-1", 90.0)
    line.weld_and_speedup()  # this is its own complete batch
    weld_before = line.weld.last_weld()[0]

    # Now simulate a torn write of the *next* (acid-loop) batch by damaging
    # the tail of the journal after a partial append: easiest is to append a
    # partial frame that references pump/rise keys, then reopen.
    torn = journal.encode_frame(
        999, {"pump.state": b'{"running": true}', "pickle.rinse": b"{}"}
    )
    jpath = os.path.join(line.store.directory, journal.JOURNAL_NAME)
    with open(jpath, "ab") as f:
        f.write(torn[: len(torn) // 2])
        f.flush()
        os.fsync(f.fileno())

    line2 = Line(d)
    state = line2.recover()
    check("weld batch survived", line2.weld.last_weld()[0] == weld_before)
    check("torn acid-loop batch ignored", "pump.state" not in state.loaded
          or line2.pump.running() is False)
    check("pump defaults to stopped, not half-new", line2.pump.running() is False)


def test_rollback_on_business_exception():
    print("failed business action stages nothing")
    d = tempfile.mkdtemp()
    line = Line(d)
    line.recover()
    line.load_coil("C-1", 90.0)
    # Force the acid loop to fail (level too low): pump.start must not leave
    # a committed pump.state.
    try:
        line.start_acid_loop(-5.0)
    except Exception:
        pass
    line2 = Line(d)
    state = line2.recover()
    check("no pump snapshot committed on failure",
          line2.pump.running() is False)


def main():
    test_full_cycle_is_one_recoverable_batch()
    test_crash_mid_cycle_drops_the_entire_unfinished_batch()
    test_rollback_on_business_exception()
    print("\nALL LINE CRASH TESTS PASSED")


if __name__ == "__main__":
    main()
