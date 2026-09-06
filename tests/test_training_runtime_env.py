import ast
import os
from pathlib import Path
from unittest import mock

from training.runtime_env import configure_training_runtime_env


class TestConfigureTrainingRuntimeEnv:
    def test_sets_safe_defaults_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("training.runtime_env.platform.system", return_value="Linux"):
                configure_training_runtime_env()

            assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
            assert os.environ["MUJOCO_GL"] == "egl"

    def test_preserves_existing_overrides(self):
        with mock.patch.dict(
            os.environ,
            {
                "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                "MUJOCO_GL": "glfw",
            },
            clear=True,
        ):
            with mock.patch("training.runtime_env.platform.system", return_value="Linux"):
                configure_training_runtime_env()

            assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
            assert os.environ["MUJOCO_GL"] == "glfw"

    def test_does_not_force_mujoco_gl_off_linux(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("training.runtime_env.platform.system", return_value="Darwin"):
                configure_training_runtime_env()

            assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
            assert "MUJOCO_GL" not in os.environ


def test_eval_policy_configures_runtime_before_importing_jax():
    source = (
        Path(__file__).resolve().parents[1] / "training/eval/eval_policy.py"
    ).read_text()
    tree = ast.parse(source)
    guard_line = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "configure_training_runtime_env"
    )
    jax_import_line = next(
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Import)
        and any(alias.name == "jax" for alias in node.names)
    )

    assert guard_line < jax_import_line
