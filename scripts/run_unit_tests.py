"""Simple test runner to execute deterministic unit tests without pytest.

This runner imports test modules and executes any callables prefixed with
`test_`. It prints a concise pass/fail summary and exits non-zero on failure.
"""
import importlib
import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path so `tests` and `playground_amp` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = [
    "tests.test_quat_normalize",
    "tests.test_quat_edges",
]

failed = []
for modname in MODULES:
    mod = importlib.import_module(modname)
    for name in dir(mod):
        if name.startswith("test_"):
            fn = getattr(mod, name)
            if callable(fn):
                try:
                    fn()
                    print(f"PASS: {modname}.{name}")
                except Exception:
                    print(f"FAIL: {modname}.{name}")
                    traceback.print_exc()
                    failed.append(f"{modname}.{name}")

if failed:
    print("\nSome tests failed:\n", "\n".join(failed))
    sys.exit(1)
else:
    print("\nAll tests passed")
    sys.exit(0)
