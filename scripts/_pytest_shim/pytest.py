"""Tiny pytest stand-in so the suite can run without pytest installed.

Supports just what this repo's tests use: def test_* functions/methods,
tmp_path fixtures, raises(), approx and plain asserts. Not a real pytest.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import re
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path


# --------------------------------------------------------------------- raises

class _Raises:
    def __init__(self, expected):
        self.expected = expected
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected!r}")
        if not issubclass(exc_type, self.expected):
            return False
        self.value = exc
        return True


def raises(expected):
    return _Raises(expected)


# --------------------------------------------------------------------- approx

class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, actual):
        if actual is None:
            return False
        tol = max(self.rel * max(abs(actual), abs(self.expected)), self.abs)
        return abs(actual - self.expected) <= tol

    def __repr__(self):
        return f"approx({self.expected!r})"


def approx(expected, rel=1e-6, abs=1e-12):
    if isinstance(expected, (list, tuple)):
        return type(expected)(_Approx(v, rel, abs) for v in expected)
    return _Approx(expected, rel, abs)


# ------------------------------------------------------------------ skipping

class Skipped(Exception):
    pass


def skip(reason=""):
    raise Skipped(reason)


# ------------------------------------------------------------------- runners

def _supports_tmp(param_name, fixture_names):
    return "tmp_path" in param_name


def _make_fixtures(func, tmp_root: Path):
    sig = inspect.signature(func)
    kwargs = {}
    for name in sig.parameters:
        if name == "tmp_path":
            d = tmp_root / re.sub(r"\W", "_", func.__qualname__)
            d.mkdir(parents=True, exist_ok=True)
            kwargs[name] = d
        elif name in ("line", "client"):
            # module-level conftest fixtures; reproduce them
            from linecontrol.line.machine import Machine
            d = tmp_root / re.sub(r"\W", "_", func.__qualname__)
            d.mkdir(parents=True, exist_ok=True)
            from linecontrol.line.machine import Line
            instance = Line(d)
            instance.recover()
            if name == "client":
                from linecontrol.console.app import create_app
                web_root = Path(__file__).resolve().parents[1] / "web"
                kwargs[name] = create_app(instance, web_root).test_client()
            else:
                kwargs[name] = instance
        else:
            kwargs[name] = None
    return kwargs


def run_module(path: Path, tmp_root: Path) -> tuple[int, int, int]:
    name = "t_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Skipped as skip_reason:
        print(f"SKIP {path.name}: {skip_reason}")
        return (0, 0, 1)
    except Exception:
        print(f"IMPORT-FAIL {path.name}")
        traceback.print_exc()
        return (0, 1, 0)

    passed = failed = skipped = 0
    members = list(vars(module).items())
    for obj_name, obj in members:
        # plain test functions
        if inspect.isfunction(obj) and obj_name.startswith("test_"):
            passed, failed, skipped = _run_callable(
                obj, tmp_root, f"{path.name}::{obj_name}", passed, failed, skipped
            )
        # test classes
        elif inspect.isclass(obj) and obj_name.startswith("Test"):
            for meth_name, meth in vars(obj).items():
                if inspect.isfunction(meth) and meth_name.startswith("test_"):
                    instance = obj()
                    passed, failed, skipped = _run_callable(
                        meth, tmp_root, f"{path.name}::{obj_name}.{meth_name}",
                        passed, failed, skipped, bound_self=instance,
                    )
    return passed, failed, skipped


def _fixture_value(name, label, tmp_root):
    if name == "tmp_path":
        d = tmp_root / re.sub(r"\W", "_", label)
        d.mkdir(parents=True, exist_ok=True)
        return d
    if name == "line":
        from linecontrol.line.machine import Line
        d = tmp_root / re.sub(r"\W", "_", label)
        d.mkdir(parents=True, exist_ok=True)
        instance = Line(d)
        instance.recover()
        return instance
    if name == "client":
        from linecontrol.console.app import create_app
        from linecontrol.line.machine import Line
        d = tmp_root / re.sub(r"\W", "_", label)
        d.mkdir(parents=True, exist_ok=True)
        instance = Line(d)
        instance.recover()
        web_root = Path(__file__).resolve().parents[2] / "web"
        return create_app(instance, web_root).test_client()
    raise Skipped(f"unknown fixture {name}")


def _run_callable(func, tmp_root, label, passed, failed, skipped, bound_self=None):
    sig = inspect.signature(func)
    try:
        kwargs = {}
        for pname in sig.parameters:
            kwargs[pname] = _fixture_value(pname, label, tmp_root)
        if bound_self is not None:
            func(bound_self, **kwargs)
        else:
            func(**kwargs)
        print(f"PASS {label}")
        passed += 1
    except Skipped as reason:
        print(f"SKIP {label} ({reason})")
        skipped += 1
    except Exception:
        print(f"FAIL {label}")
        traceback.print_exc()
        failed += 1
    return passed, failed, skipped


@contextmanager
def fixture(*a, **k):
    yield None


def main():
    root = Path(__file__).resolve().parents[2]
    tests = sorted((root / "tests").glob("test_*.py"))
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        total_p = total_f = total_s = 0
        for path in tests:
            p, f, s = run_module(path, tmp_root)
            total_p += p
            total_f += f
            total_s += s
    print(f"\n{total_p} passed, {total_f} failed, {total_s} skipped")
    raise SystemExit(1 if total_f else 0)


# expose the symbols tests import
__all__ = ["raises", "approx", "skip", "fixture"]


if __name__ == "__main__":
    main()
