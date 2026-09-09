from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_validate_bundle_script_bootstraps_repo_imports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "runtime/wr_runtime/validation/validate_bundle.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Validate a policy bundle" in completed.stdout
