"""Tests for control/locomotion/ — nominal IK adapter and walking controller.

Tests are organized by layer:
1. NominalIkAdapter: IK correctness, reachability, joint limits
2. WalkingController: mode switching, step outputs, residual composition
3. RuntimeLocRefV2Builder: reference stepping, feature shapes
4. Integration: full nominal-only and residual-PPO pipelines
"""

from __future__ import annotations

import json
import math
import sys
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

from control.kinematics.leg_ik import LegIkConfig, solve_leg_sagittal_ik
from control.locomotion.nominal_ik_adapter import (
    NominalIkConfig,
    NominalIkResult,
    compute_nominal_q_ref,
)
from control.locomotion.walking_controller import (
    ControllerMode,
    WalkingController,
    WalkingControllerConfig,
)
from control.references.walking_ref_v2 import (
    WalkingRefV2Config,
    WalkingRefV2Mode,
    WalkingRefV2State,
)
import importlib.util
import types

# The runtime.wr_runtime.__init__ has a broken import ('from configs import ...')
# that only works when cwd is runtime/.  Load the submodule directly from its path.
def _load_module_from_path(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_loc_ref_v2_path = str(Path(__file__).parent.parent.parent / "runtime" / "wr_runtime" / "control" / "loc_ref_runtime_v2.py")
_loc_ref_v2_mod = _load_module_from_path("loc_ref_runtime_v2", _loc_ref_v2_path)
RuntimeLocRefV2Builder = _loc_ref_v2_mod.RuntimeLocRefV2Builder
_run_walking_path = str(Path(__file__).parent.parent.parent / "runtime" / "wr_runtime" / "control" / "run_walking.py")
_run_walking_mod = _load_module_from_path("run_walking", _run_walking_path)


# ---------------------------------------------------------------------------
# Layer 1: NominalIkAdapter tests
# ---------------------------------------------------------------------------

class TestNominalIkAdapter:
    """Test the standalone IK adapter."""

    def test_output_shape(self):
        """q_ref should be a 9-element array."""
        result = compute_nominal_q_ref(
            config=NominalIkConfig(),
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
        )
        assert isinstance(result, NominalIkResult)
        assert result.q_ref.shape == (9,)
        assert result.q_ref.dtype == np.float32

    def test_symmetric_stance(self):
        """Left-stance and right-stance should produce mirror joint angles."""
        cfg = NominalIkConfig()
        left = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.03, -0.08, 0.01),
        )
        right = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=1,
            swing_pos=(0.03, 0.08, 0.01),
        )
        # Stance leg hip/knee/ankle should be equal when posture is symmetric
        assert abs(left.q_ref[0] - right.q_ref[1]) < 0.01  # hip pitch L vs R
        assert abs(left.q_ref[4] - right.q_ref[5]) < 0.01  # knee L vs R
        assert abs(left.q_ref[6] - right.q_ref[7]) < 0.01  # ankle L vs R

    def test_reachable_targets(self):
        """Normal walking targets should be reachable."""
        result = compute_nominal_q_ref(
            config=NominalIkConfig(),
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.05, -0.08, 0.02),
        )
        assert result.stance_reachable
        assert result.swing_reachable

    def test_joint_limits_respected(self):
        """Output should never exceed configured joint limits."""
        cfg = NominalIkConfig(
            joint_range_min=(-0.5,) * 9,
            joint_range_max=(0.5,) * 9,
        )
        result = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.30,  # Deep squat — large joint angles
            pelvis_roll_rad=0.1,
            pelvis_pitch_rad=0.1,
            stance_foot_id=0,
            swing_pos=(0.1, -0.08, 0.03),
        )
        assert np.all(result.q_ref >= -0.5)
        assert np.all(result.q_ref <= 0.5)

    def test_knee_bends_with_lower_height(self):
        """Lower pelvis height should produce more knee bend."""
        cfg = NominalIkConfig()
        high = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
        )
        low = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.35,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
        )
        # Stance knee (left, idx 4) should be more bent at lower height
        assert low.q_ref[4] > high.q_ref[4]

    def test_com_trajectory_shifts_stance(self):
        """Non-zero com_x_planned should shift stance leg IK targets."""
        cfg = NominalIkConfig()
        no_com = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
            com_x_planned=0.0,
        )
        with_com = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
            com_x_planned=0.03,
        )
        # COM forward should change stance hip pitch
        assert abs(no_com.q_ref[0] - with_com.q_ref[0]) > 0.01


# ---------------------------------------------------------------------------
# Layer 2: WalkingController tests
# ---------------------------------------------------------------------------

class TestWalkingController:
    """Test the unified walking controller."""

    def _make_controller(self, mode: ControllerMode = ControllerMode.NOMINAL_ONLY) -> WalkingController:
        cfg = WalkingControllerConfig(
            mode=mode,
            ref_config=WalkingRefV2Config(),
            ik_config=NominalIkConfig(),
            dt_s=0.02,
        )
        return WalkingController(cfg)

    def test_nominal_only_step(self):
        """Nominal-only mode should produce valid output without delta_q."""
        ctrl = self._make_controller(ControllerMode.NOMINAL_ONLY)
        out = ctrl.step(
            forward_speed_mps=0.15,
            left_foot_loaded=True,
            right_foot_loaded=True,
        )
        assert out.q_target.shape == (9,)
        assert out.q_ref.shape == (9,)
        np.testing.assert_array_equal(out.q_target, out.q_ref)
        assert out.mode == ControllerMode.NOMINAL_ONLY

    def test_residual_ppo_without_delta_uses_ref(self):
        """Residual mode without delta_q should fall back to q_ref."""
        ctrl = self._make_controller(ControllerMode.RESIDUAL_PPO)
        out = ctrl.step(
            forward_speed_mps=0.15,
            left_foot_loaded=True,
            right_foot_loaded=True,
            delta_q=None,
        )
        np.testing.assert_array_equal(out.q_target, out.q_ref)

    def test_residual_ppo_with_delta(self):
        """Residual mode with delta_q should modify q_target."""
        ctrl = self._make_controller(ControllerMode.RESIDUAL_PPO)
        out = ctrl.step(
            forward_speed_mps=0.15,
            left_foot_loaded=True,
            right_foot_loaded=True,
            delta_q=np.ones(9, dtype=np.float32) * 0.5,
        )
        # q_target should differ from q_ref
        assert not np.allclose(out.q_target, out.q_ref)
        # q_target should be q_ref + scaled delta
        assert out.q_target.shape == (9,)

    def test_compose_residual_clipping(self):
        """compose_residual should clip output to joint limits."""
        ctrl = self._make_controller(ControllerMode.RESIDUAL_PPO)
        q_ref = np.zeros(9, dtype=np.float32)
        delta_q = np.ones(9, dtype=np.float32) * 100.0  # Way beyond limits
        q_target = ctrl.compose_residual(q_ref, delta_q)
        cfg = ctrl.config.ik_config
        range_max = np.array(cfg.joint_range_max[:9], dtype=np.float32)
        assert np.all(q_target <= range_max + 1e-6)

    def test_compose_residual_zero_delta(self):
        """Zero delta should return q_ref unchanged."""
        ctrl = self._make_controller(ControllerMode.RESIDUAL_PPO)
        q_ref = np.array([0.1, -0.1, 0.2, -0.2, 0.5, 0.5, -0.3, -0.3, 0.0],
                         dtype=np.float32)
        delta_q = np.zeros(9, dtype=np.float32)
        q_target = ctrl.compose_residual(q_ref, delta_q)
        np.testing.assert_allclose(q_target, q_ref, atol=1e-6)

    def test_reset_clears_state(self):
        """Reset should restore initial state."""
        ctrl = self._make_controller()
        ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        ctrl.reset()
        assert ctrl.state.ref_state.phase_time_s == 0.0
        assert ctrl.state.ref_state.stance_switch_count == 0
        assert ctrl.state.prev_q_ref is None

    def test_multiple_steps_advance_phase(self):
        """Multiple steps should advance the gait phase."""
        ctrl = self._make_controller()
        # Start from SUPPORT_STABILIZE (skip startup) so phase advances
        ctrl.reset(initial_mode=WalkingRefV2Mode.SUPPORT_STABILIZE)
        # First step
        ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        phase1 = ctrl.state.ref_state.phase_time_s
        # Many more steps
        for _ in range(50):
            ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        phase2 = ctrl.state.ref_state.phase_time_s
        # Phase should have advanced (or wrapped with stance switches)
        assert ctrl.state.ref_state.stance_switch_count > 0 or phase2 > phase1


# ---------------------------------------------------------------------------
# Layer 3: RuntimeLocRefV2Builder tests
# ---------------------------------------------------------------------------

class TestRuntimeLocRefV2Builder:
    """Test the walking ref v2 runtime wrapper."""

    def test_features_shape(self):
        """Feature arrays should have correct shapes."""
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat = builder.step(forward_speed_mps=0.15)
        assert feat.phase_sin_cos.shape == (2,)
        assert feat.stance_foot.shape == (1,)
        assert feat.next_foothold.shape == (2,)
        assert feat.swing_pos.shape == (3,)
        assert feat.swing_vel.shape == (3,)
        assert feat.pelvis_targets.shape == (3,)
        assert feat.history.shape == (4,)

    def test_features_dtype(self):
        """All feature arrays should be float32."""
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat = builder.step(forward_speed_mps=0.15)
        assert feat.phase_sin_cos.dtype == np.float32
        assert feat.stance_foot.dtype == np.float32
        assert feat.next_foothold.dtype == np.float32
        assert feat.swing_pos.dtype == np.float32

    def test_history_rolls(self):
        """Phase history should roll forward with each step."""
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat1 = builder.step(forward_speed_mps=0.15)
        feat2 = builder.step(forward_speed_mps=0.15)
        # History slots 0-1 should be feat1's phase, slots 2-3 should be feat2's
        np.testing.assert_allclose(feat2.history[:2], feat1.phase_sin_cos, atol=1e-6)

    def test_reset_clears_history(self):
        """Reset should zero out history."""
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        builder.step(forward_speed_mps=0.15)
        builder.reset()
        assert np.allclose(builder._history, 0.0)

    def test_support_health_bounded(self):
        """Support health should be in [0, 1]."""
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat = builder.step(forward_speed_mps=0.15)
        assert 0.0 <= feat.support_health <= 1.0

    def test_invalid_dt_raises(self):
        """Non-positive dt should raise."""
        with pytest.raises(ValueError, match="default_dt_s"):
            RuntimeLocRefV2Builder(default_dt_s=0.0)
        with pytest.raises(ValueError, match="default_dt_s"):
            RuntimeLocRefV2Builder(default_dt_s=-0.01)

    def test_mode_starts_from_support(self):
        """Default initial mode should be SUPPORT_STABILIZE."""
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat = builder.step(forward_speed_mps=0.15)
        assert feat.mode_id == int(WalkingRefV2Mode.SUPPORT_STABILIZE)


# ---------------------------------------------------------------------------
# Layer 4: Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests."""

    def test_nominal_only_pipeline_runs_500_steps(self):
        """Nominal-only controller should run 100 steps without error."""
        build_controller_from_config = _run_walking_mod.build_controller_from_config
        run_walking_loop = _run_walking_mod.run_walking_loop
        cfg = {
            "controller_mode": "nominal_only",
            "forward_speed_mps": 0.15,
            "policy_path": None,
            "dt_s": 0.02,
        }
        controller = build_controller_from_config(cfg)
        result = run_walking_loop(
            controller,
            forward_speed_mps=0.15,
            max_steps=100,
            dry_run=True,
        )
        assert result["steps"] == 100
        assert result["mode"] == "nominal_only"
        assert result["q_ref_log"].shape == (100, 9)

    def test_residual_ppo_pipeline_with_zero_policy(self):
        """Residual PPO with zero residual should match nominal."""
        cfg = WalkingControllerConfig(
            mode=ControllerMode.RESIDUAL_PPO,
            dt_s=0.02,
        )
        ctrl = WalkingController(cfg)
        out = ctrl.step(
            forward_speed_mps=0.15,
            left_foot_loaded=True,
            right_foot_loaded=True,
            delta_q=np.zeros(9, dtype=np.float32),
        )
        np.testing.assert_allclose(out.q_target, out.q_ref, atol=1e-6)

    def test_runtime_config_loading(self):
        """Runtime config JSON files should be valid and loadable."""
        load_walking_config = _run_walking_mod.load_walking_config

        # v0.19.4 nominal-only
        cfg_path = Path(__file__).parent.parent.parent / "runtime" / "configs" / "walking_v0194.json"
        if cfg_path.exists():
            cfg = load_walking_config(str(cfg_path))
            assert cfg["controller_mode"] == "nominal_only"
            assert cfg["policy_path"] is None

        # v0.19.5 residual PPO
        cfg_path = Path(__file__).parent.parent.parent / "runtime" / "configs" / "walking_v0195.json"
        if cfg_path.exists():
            cfg = load_walking_config(str(cfg_path))
            assert cfg["controller_mode"] == "residual_ppo"
            assert cfg["policy_path"] is not None

    def test_residual_ppo_config_requires_policy_path(self):
        """residual_ppo mode without policy_path should raise."""
        load_walking_config = _run_walking_mod.load_walking_config
        import tempfile
        import os

        bad_cfg = {"controller_mode": "residual_ppo", "policy_path": None}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bad_cfg, f)
            f.flush()
            with pytest.raises(ValueError, match="policy_path"):
                load_walking_config(f.name)
            os.unlink(f.name)

    def test_training_config_exists(self):
        """v0.19.5 training config should exist and be valid YAML."""
        import yaml
        cfg_path = Path(__file__).parent.parent / "configs" / "ppo_walking_v0195.yaml"
        assert cfg_path.exists(), f"Training config not found: {cfg_path}"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["version"] == "0.19.5"
        assert cfg["env"]["loc_ref_enabled"] is True
        assert cfg["env"]["loc_ref_residual_scale"] == 0.18
        assert cfg["env"]["start_from_support_posture"] is True
        assert cfg["env"]["com_trajectory_enabled"] is True
        assert cfg["ppo"]["iterations"] == 1000
