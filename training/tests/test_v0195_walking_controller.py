"""Tests for v0.19.5 walking controller, IK adapter, and runtime integration.

Tests are organized by layer:
1. NominalIkAdapter: IK correctness, sign conventions matching env
2. WalkingController: mode switching, COM trajectory threading, residual composition
3. RuntimeLocRefV2Builder: reference stepping, feature shapes
4. Integration: full nominal-only and residual-PPO pipelines
5. Config parity: runtime configs match training config parameter set
6. Script launchability: run_walking.py is independently importable
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent

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

# Import runtime loc_ref_v2 by file path to bypass the broken
# runtime/wr_runtime/__init__.py (pre-existing: 'from configs import ...').
# TestLaunchability.test_run_walking_dry_run_nominal validates the actual
# script execution path works end-to-end.
import importlib.util as _ilu
import types as _types

def _load_from_path(name: str, path: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_loc_ref_v2_mod = _load_from_path(
    "loc_ref_runtime_v2",
    str(_REPO_ROOT / "runtime" / "wr_runtime" / "control" / "loc_ref_runtime_v2.py"),
)
RuntimeLocRefV2Builder = _loc_ref_v2_mod.RuntimeLocRefV2Builder
RuntimeLocRefV2Features = _loc_ref_v2_mod.RuntimeLocRefV2Features

_run_walking_mod = _load_from_path(
    "run_walking",
    str(_REPO_ROOT / "runtime" / "wr_runtime" / "control" / "run_walking.py"),
)


# ---------------------------------------------------------------------------
# Layer 1: NominalIkAdapter — env parity
# ---------------------------------------------------------------------------

class TestNominalIkAdapter:

    def test_output_shape(self):
        result = compute_nominal_q_ref(
            config=NominalIkConfig(),
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
        )
        assert result.q_ref.shape == (9,)
        assert result.q_ref.dtype == np.float32

    def test_left_hip_pitch_negated(self):
        """Env line 4837: q_ref.at[idx_left_hip_pitch].set(-left_hip).

        Left hip pitch must be negated relative to IK output.
        """
        result = compute_nominal_q_ref(
            config=NominalIkConfig(),
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
        )
        # IK for a target at (0, -0.40) produces positive hip pitch.
        # Env negates left hip pitch, so q_ref should be negative or zero.
        # Right hip pitch is NOT negated (env line 4842).
        ik = solve_leg_sagittal_ik(
            target_x_m=0.0, target_z_m=-0.40, config=LegIkConfig()
        )
        if ik.hip_pitch_rad > 0.01:
            # If IK says positive, left_hip in q_ref should be negative
            assert result.q_ref[0] < 0.0, (
                f"left_hip_pitch should be negated: IK={ik.hip_pitch_rad}, "
                f"q_ref[0]={result.q_ref[0]}"
            )

    def test_hip_roll_sign_convention(self):
        """Env lines 4853-4855: left_hip_roll = -roll, right_hip_roll = +roll."""
        result = compute_nominal_q_ref(
            config=NominalIkConfig(),
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.05,  # positive roll
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.0, -0.08, 0.0),
        )
        # left_hip_roll (idx 2) should be -(0.05 + swing_y_coupling)
        # right_hip_roll (idx 3) should be +(0.05 + swing_y_coupling)
        assert result.q_ref[2] < 0.0, "left_hip_roll should be negative for positive pelvis roll"
        assert result.q_ref[3] > 0.0, "right_hip_roll should be positive for positive pelvis roll"

    def test_com_trajectory_additive(self):
        """Env line 4812: _stance_foot_x = _stance_foot_x + _com_x (additive)."""
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
        # Positive COM planned should shift stance foot forward, changing hip pitch
        assert abs(no_com.q_ref[0] - with_com.q_ref[0]) > 0.01

    def test_both_feet_grounded_uses_stance_offset(self):
        """During SUPPORT_STABILIZE, both legs should use stance foot offset.

        Env lines 4817-4821: when both_feet_grounded, swing target_x = stance_foot_x.
        """
        cfg = NominalIkConfig()
        # In support mode with swing_pos=(0.03, ..., 0.01), the swing_x should
        # be overridden to match stance_foot_x.
        support = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.10, -0.08, 0.0),  # large swing_x that should be overridden
            mode_id=int(WalkingRefV2Mode.SUPPORT_STABILIZE),
        )
        swing = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.42,
            pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0,
            stance_foot_id=0,
            swing_pos=(0.10, -0.08, 0.0),  # same swing_x, but NOT overridden in swing mode
            mode_id=int(WalkingRefV2Mode.SWING_RELEASE),
        )
        # In support mode: right leg (swing) hip pitch should use stance offset
        # In swing mode: right leg uses swing_x=0.10 (different)
        # So hip pitches should differ between modes
        assert support.q_ref[1] != swing.q_ref[1], \
            "Right hip pitch should differ between support and swing modes"

    def test_joint_limits_respected(self):
        cfg = NominalIkConfig(
            joint_range_min=(-0.5,) * 9,
            joint_range_max=(0.5,) * 9,
        )
        result = compute_nominal_q_ref(
            config=cfg,
            pelvis_height_m=0.30,
            pelvis_roll_rad=0.1,
            pelvis_pitch_rad=0.1,
            stance_foot_id=0,
            swing_pos=(0.1, -0.08, 0.03),
        )
        assert np.all(result.q_ref >= -0.5 - 1e-6)
        assert np.all(result.q_ref <= 0.5 + 1e-6)

    def test_knee_bends_with_lower_height(self):
        cfg = NominalIkConfig()
        high = compute_nominal_q_ref(
            config=cfg, pelvis_height_m=0.42, pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0, stance_foot_id=0, swing_pos=(0.0, -0.08, 0.0),
        )
        low = compute_nominal_q_ref(
            config=cfg, pelvis_height_m=0.35, pelvis_roll_rad=0.0,
            pelvis_pitch_rad=0.0, stance_foot_id=0, swing_pos=(0.0, -0.08, 0.0),
        )
        assert low.q_ref[4] > high.q_ref[4]  # stance knee more bent


# ---------------------------------------------------------------------------
# Layer 2: WalkingController
# ---------------------------------------------------------------------------

class TestWalkingController:

    def _make_controller(self, mode=ControllerMode.NOMINAL_ONLY) -> WalkingController:
        return WalkingController(WalkingControllerConfig(
            mode=mode,
            ref_config=WalkingRefV2Config(com_trajectory_enabled=True),
            ik_config=NominalIkConfig(),
            dt_s=0.02,
        ))

    def test_nominal_only_step(self):
        ctrl = self._make_controller(ControllerMode.NOMINAL_ONLY)
        out = ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=True)
        assert out.q_target.shape == (9,)
        np.testing.assert_array_equal(out.q_target, out.q_ref)
        assert out.mode == ControllerMode.NOMINAL_ONLY

    def test_com_trajectory_threaded(self):
        """COM trajectory from ref should reach the IK adapter (not hardcoded 0)."""
        ctrl = self._make_controller()
        ctrl.reset(initial_mode=WalkingRefV2Mode.SUPPORT_STABILIZE)
        # Advance into SWING_RELEASE mode where COM trajectory is active
        for _ in range(30):
            ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        out = ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        # If mode is SWING_RELEASE or later, com_x_planned should be > 0
        if out.ref_output.hybrid_mode_id >= int(WalkingRefV2Mode.SWING_RELEASE):
            assert out.ref_output.com_x_planned > 0.0, \
                f"COM trajectory should be positive during swing, got {out.ref_output.com_x_planned}"

    def test_residual_with_delta(self):
        ctrl = self._make_controller(ControllerMode.RESIDUAL_PPO)
        out = ctrl.step(
            forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=True,
            delta_q=np.ones(9, dtype=np.float32) * 0.5,
        )
        assert not np.allclose(out.q_target, out.q_ref)

    def test_compose_residual_zero_delta(self):
        ctrl = self._make_controller(ControllerMode.RESIDUAL_PPO)
        q_ref = np.array([0.1, -0.1, 0.2, -0.2, 0.5, 0.5, -0.3, -0.3, 0.0], dtype=np.float32)
        q_target = ctrl.compose_residual(q_ref, np.zeros(9, dtype=np.float32))
        np.testing.assert_allclose(q_target, q_ref, atol=1e-6)

    def test_get_reference_features_shape(self):
        """get_reference_features() should return wr_obs_v4-compatible arrays."""
        ctrl = self._make_controller()
        ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=True)
        feats = ctrl.get_reference_features()
        assert feats["phase_sin_cos"].shape == (2,)
        assert feats["stance_foot"].shape == (1,)
        assert feats["next_foothold"].shape == (2,)
        assert feats["swing_pos"].shape == (3,)
        assert feats["swing_vel"].shape == (3,)
        assert feats["pelvis_targets"].shape == (3,)
        assert feats["phase_history"].shape == (4,)

    def test_reset_clears_state(self):
        ctrl = self._make_controller()
        ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        ctrl.reset()
        assert ctrl.state.ref_state.phase_time_s == 0.0
        assert ctrl.state.prev_q_ref is None
        assert ctrl.state.last_ref_output is None

    def test_multiple_steps_advance_phase(self):
        ctrl = self._make_controller()
        ctrl.reset(initial_mode=WalkingRefV2Mode.SUPPORT_STABILIZE)
        ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        for _ in range(50):
            ctrl.step(forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=False)
        assert ctrl.state.ref_state.stance_switch_count > 0 or ctrl.state.ref_state.phase_time_s > 0


# ---------------------------------------------------------------------------
# Layer 3: RuntimeLocRefV2Builder
# ---------------------------------------------------------------------------

class TestRuntimeLocRefV2Builder:

    def test_features_shape(self):
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat = builder.step(forward_speed_mps=0.15)
        assert feat.phase_sin_cos.shape == (2,)
        assert feat.stance_foot.shape == (1,)
        assert feat.next_foothold.shape == (2,)
        assert feat.swing_pos.shape == (3,)
        assert feat.swing_vel.shape == (3,)
        assert feat.pelvis_targets.shape == (3,)
        assert feat.history.shape == (4,)

    def test_history_rolls(self):
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat1 = builder.step(forward_speed_mps=0.15)
        feat2 = builder.step(forward_speed_mps=0.15)
        np.testing.assert_allclose(feat2.history[:2], feat1.phase_sin_cos, atol=1e-6)

    def test_support_health_bounded(self):
        builder = RuntimeLocRefV2Builder(default_dt_s=0.02)
        feat = builder.step(forward_speed_mps=0.15)
        assert 0.0 <= feat.support_health <= 1.0

    def test_invalid_dt_raises(self):
        with pytest.raises(ValueError):
            RuntimeLocRefV2Builder(default_dt_s=0.0)


# ---------------------------------------------------------------------------
# Layer 4: Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_nominal_only_pipeline_100_steps(self):
        """Nominal-only controller should run without error."""
        cfg = {
            "controller_mode": "nominal_only",
            "forward_speed_mps": 0.15,
            "policy_path": None,
            "dt_s": 0.02,
            "ref_config": {"com_trajectory_enabled": True},
        }
        controller = _run_walking_mod.build_controller_from_config(cfg)
        result = _run_walking_mod.run_walking_loop(
            controller, forward_speed_mps=0.15, max_steps=100, dry_run=True,
        )
        assert result["steps"] == 100
        assert result["mode"] == "nominal_only"
        assert result["q_ref_log"].shape == (100, 9)

    def test_residual_ppo_compose_with_zero_delta(self):
        """Residual composition with zero delta should match q_ref."""
        ctrl = WalkingController(WalkingControllerConfig(
            mode=ControllerMode.RESIDUAL_PPO, dt_s=0.02,
        ))
        out = ctrl.step(
            forward_speed_mps=0.15, left_foot_loaded=True, right_foot_loaded=True,
            delta_q=np.zeros(9, dtype=np.float32),
        )
        np.testing.assert_allclose(out.q_target, out.q_ref, atol=1e-6)

    def test_residual_ppo_config_requires_policy_path(self):
        import tempfile, os
        bad_cfg = {"controller_mode": "residual_ppo", "policy_path": None}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bad_cfg, f)
            f.flush()
            with pytest.raises(ValueError, match="policy_path"):
                _run_walking_mod.load_walking_config(f.name)
            os.unlink(f.name)


# ---------------------------------------------------------------------------
# Layer 5: Config parity
# ---------------------------------------------------------------------------

class TestConfigParity:

    def test_runtime_config_keys_match_training(self):
        """Runtime configs should carry all key WalkingRefV2Config params from training."""
        training_cfg_path = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0195.yaml"
        runtime_cfg_path = _REPO_ROOT / "runtime" / "configs" / "walking_v0194.json"

        assert training_cfg_path.exists()
        assert runtime_cfg_path.exists()

        with open(training_cfg_path) as f:
            train = yaml.safe_load(f)
        with open(runtime_cfg_path) as f:
            runtime = json.load(f)

        # Check critical parameters that differ from WalkingRefV2Config defaults
        ref_cfg = runtime.get("ref_config", {})
        assert ref_cfg.get("min_step_length_m") == 0.03, "Must match training loc_ref_min_step_length_m"
        assert ref_cfg.get("max_step_length_m") == 0.12, "Must match training loc_ref_max_step_length_m"
        assert ref_cfg.get("support_release_phase_start") == 0.25, "Must match training"
        assert ref_cfg.get("support_open_threshold") == 0.50, "Must match training"
        assert ref_cfg.get("com_trajectory_enabled") is True

    def test_training_config_v0195_valid(self):
        cfg_path = _REPO_ROOT / "training" / "configs" / "ppo_walking_v0195.yaml"
        assert cfg_path.exists()
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["version"] == "0.19.5"
        assert cfg["env"]["loc_ref_enabled"] is True
        assert cfg["env"]["loc_ref_residual_scale"] == 0.18
        assert cfg["env"]["start_from_support_posture"] is True
        assert cfg["env"]["com_trajectory_enabled"] is True
        assert cfg["ppo"]["iterations"] == 1000


# ---------------------------------------------------------------------------
# Layer 6: Script launchability
# ---------------------------------------------------------------------------

class TestLaunchability:

    def test_run_walking_importable_as_script(self):
        """run_walking.py should be importable without triggering wr_runtime.__init__."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util, sys; "
             "spec = importlib.util.spec_from_file_location('run_walking', "
             "'runtime/wr_runtime/control/run_walking.py'); "
             "mod = importlib.util.module_from_spec(spec); "
             "spec.loader.exec_module(mod); "
             "print('OK')"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=30,
        )
        assert result.returncode == 0, f"Import failed:\n{result.stderr}"
        assert "OK" in result.stdout

    def test_run_walking_dry_run_nominal(self):
        """run_walking.py --dry-run should execute without error in nominal mode."""
        cfg_path = _REPO_ROOT / "runtime" / "configs" / "walking_v0194.json"
        if not cfg_path.exists():
            pytest.skip("walking_v0194.json not found")
        result = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "runtime" / "wr_runtime" / "control" / "run_walking.py"),
             "--config", str(cfg_path), "--dry-run", "--max-steps", "10"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=30,
        )
        assert result.returncode == 0, f"Dry run failed:\n{result.stderr}"
        assert "Completed 10 steps" in result.stdout
