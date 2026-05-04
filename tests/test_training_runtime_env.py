import os
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
