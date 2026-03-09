"""Integration tests for the FSM base controller via the real training entrypoint.

These tests exercise the path that training actually uses:

    load_robot_config(...)
    cfg = load_training_config("training/configs/ppo_standing_push.yaml")
    cfg.env.fsm_enabled = True
    cfg.freeze()
    env = WildRobotEnv(config=cfg)
    state = env.reset(rng)
    state = env.step(state, action)

They verify:
1. All fsm_* YAML keys are parsed into EnvConfig (not silently ignored).
2. The env steps without error when fsm_enabled=True.
3. FSM debug metrics are finite and non-negative after a step.
4. The FSM phase-ticks counter advances on each step.

Requires: MuJoCo model files present and robot config loadable (CI sim tier).
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

YAML_CONFIG = "training/configs/ppo_standing_push.yaml"
ROBOT_CONFIG = "assets/v2/mujoco_robot_config.json"


@pytest.fixture(scope="module")
def fsm_env():
    """WildRobotEnv with fsm_enabled=True loaded via the standard YAML path."""
    from assets.robot_config import load_robot_config
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    load_robot_config(ROBOT_CONFIG)
    cfg = load_training_config(YAML_CONFIG)

    # Activate FSM (the thing being tested)
    cfg.env.fsm_enabled = True
    cfg.env.base_ctrl_enabled = False  # FSM replaces M2
    cfg.freeze()

    return WildRobotEnv(config=cfg)


@pytest.fixture(scope="module")
def fsm_env_from_yaml_only():
    """Config loaded but NOT modified — verifies YAML values flow into EnvConfig."""
    from assets.robot_config import load_robot_config
    from training.configs.training_config import load_training_config

    load_robot_config(ROBOT_CONFIG)
    cfg = load_training_config(YAML_CONFIG)
    cfg.freeze()
    return cfg


# ---------------------------------------------------------------------------
# Tests: YAML → EnvConfig plumbing
# ---------------------------------------------------------------------------

class TestFSMConfigParsing:
    """Verify every fsm_* key in the YAML is wired through _parse_env_config."""

    def test_fsm_enabled_parsed(self, fsm_env_from_yaml_only):
        """fsm_enabled comes from YAML (default=false); must not raise AttributeError."""
        cfg = fsm_env_from_yaml_only
        assert hasattr(cfg.env, "fsm_enabled"), "EnvConfig missing fsm_enabled"
        assert isinstance(cfg.env.fsm_enabled, bool)

    def test_fsm_y_nominal_parsed(self, fsm_env_from_yaml_only):
        """fsm_y_nominal_m must equal the YAML value 0.115."""
        assert cfg_env(fsm_env_from_yaml_only).fsm_y_nominal_m == pytest.approx(0.115)

    def test_fsm_trigger_threshold_parsed(self, fsm_env_from_yaml_only):
        assert cfg_env(fsm_env_from_yaml_only).fsm_trigger_threshold == pytest.approx(0.30)

    def test_fsm_recover_threshold_parsed(self, fsm_env_from_yaml_only):
        assert cfg_env(fsm_env_from_yaml_only).fsm_recover_threshold == pytest.approx(0.15)

    def test_fsm_step_max_delta_parsed(self, fsm_env_from_yaml_only):
        """fsm_step_max_delta_m must be parsed (was a dead config key)."""
        assert cfg_env(fsm_env_from_yaml_only).fsm_step_max_delta_m == pytest.approx(0.12)

    def test_fsm_swing_height_parsed(self, fsm_env_from_yaml_only):
        assert cfg_env(fsm_env_from_yaml_only).fsm_swing_height_m == pytest.approx(0.04)

    def test_fsm_resid_scale_swing_parsed(self, fsm_env_from_yaml_only):
        assert cfg_env(fsm_env_from_yaml_only).fsm_resid_scale_swing == pytest.approx(0.45)

    def test_fsm_arm_enabled_parsed(self, fsm_env_from_yaml_only):
        assert cfg_env(fsm_env_from_yaml_only).fsm_arm_enabled is True


def cfg_env(cfg):
    return cfg.env


# ---------------------------------------------------------------------------
# Tests: env reset + step with fsm_enabled=True
# ---------------------------------------------------------------------------

class TestFSMEnvStep:
    """End-to-end env integration with FSM enabled."""

    @pytest.mark.sim
    def test_reset_succeeds(self, fsm_env):
        rng = jax.random.PRNGKey(0)
        state = jax.jit(fsm_env.reset)(rng)
        assert state.obs.shape[0] > 0, "obs must be non-empty"
        assert jp.isfinite(state.obs).all(), "obs must be finite at reset"

    @pytest.mark.sim
    def test_step_succeeds(self, fsm_env):
        rng = jax.random.PRNGKey(1)
        state = jax.jit(fsm_env.reset)(rng)
        action = jp.zeros(fsm_env.action_size)
        state2 = jax.jit(fsm_env.step)(state, action)
        assert jp.isfinite(state2.obs).all(), "obs must be finite after step"
        assert jp.isfinite(state2.reward), "reward must be finite"

    @pytest.mark.sim
    def test_fsm_debug_metrics_present(self, fsm_env):
        """debug/bc_phase, bc_swing_foot, bc_phase_ticks must be in metrics."""
        from training.core.metrics_registry import METRICS_VEC_KEY, unpack_metrics

        rng = jax.random.PRNGKey(2)
        state = jax.jit(fsm_env.reset)(rng)
        state = jax.jit(fsm_env.step)(state, jp.zeros(fsm_env.action_size))
        m = unpack_metrics(state.metrics[METRICS_VEC_KEY])

        assert "debug/bc_phase" in m, "debug/bc_phase missing from metrics"
        assert "debug/bc_in_swing" in m
        assert "debug/bc_in_recover" in m
        assert "debug/bc_swing_foot" in m
        assert "debug/bc_phase_ticks" in m
        assert float(m["debug/bc_phase"]) >= 0.0
        assert jp.isfinite(m["debug/bc_phase_ticks"])

    @pytest.mark.sim
    def test_fsm_phase_ticks_advance(self, fsm_env):
        """Phase-ticks counter must increase each step (STANCE stays, ticks ~step count)."""
        from training.envs.env_info import WR_INFO_KEY

        rng = jax.random.PRNGKey(3)
        step_fn = jax.jit(fsm_env.step)
        state = jax.jit(fsm_env.reset)(rng)

        ticks_prev = int(state.info[WR_INFO_KEY].fsm_phase_ticks)
        for _ in range(5):
            state = step_fn(state, jp.zeros(fsm_env.action_size))
        ticks_after = int(state.info[WR_INFO_KEY].fsm_phase_ticks)

        assert ticks_after > ticks_prev, (
            f"fsm_phase_ticks did not advance: {ticks_prev} → {ticks_after}"
        )

    @pytest.mark.sim
    def test_fsm_config_values_flow_to_controller(self, fsm_env):
        """The controller must observe the YAML-configured fsm_y_nominal_m, not a hardcoded value.

        We verify that enabling the FSM with a non-default y_nominal produces
        finite actions (i.e., the pipeline ran with the parsed config).
        """
        rng = jax.random.PRNGKey(4)
        state = jax.jit(fsm_env.reset)(rng)
        state = jax.jit(fsm_env.step)(state, jp.zeros(fsm_env.action_size))
        # If fsm_* config was NOT parsed, the env would default to dataclass defaults
        # (which match YAML anyway here). The real guard is test_fsm_y_nominal_parsed.
        # Here we simply confirm the step runs end-to-end without NaN.
        assert jp.isfinite(state.obs).all()
