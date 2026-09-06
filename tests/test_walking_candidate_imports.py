from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_candidate_evaluator_import_does_not_initialize_jax_backend() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probe = (
        "from jax._src import xla_bridge; "
        "import wildrobot.agents.evaluate_walking_candidate; "
        "print(xla_bridge.backends_are_initialized())"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines()[-1] == "False"
