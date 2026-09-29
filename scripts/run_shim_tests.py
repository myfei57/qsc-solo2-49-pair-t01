"""Run the project tests with the bundled pytest shim."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "_pytest_shim"))
sys.path.insert(0, str(ROOT))

import pytest  # type: ignore  # noqa: E402

if __name__ == "__main__":
    pytest.main()
