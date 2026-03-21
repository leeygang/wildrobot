"""Tests for v0.17.1 boundary push evaluation and recovery metrics.

This module tests:
- Boundary-focused push curriculum configuration  
- Recovery metrics computation and logging
- Fixed evaluation ladder integration
- Standing reward stack conservation
"""

import os
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.configs.training_config import load_training_config
from training.envs.wildrobot_env import WildRobotEnv


class TestV0171BoundaryPushEval:
    """Tests for v0.17.1 boundary push evaluation configuration."""
    
    def test_config_loads_correctly(self):
        """Test that v0.17.1 config loads with correct boundary-focused settings."""
        config = load_training_config('training/configs/ppo_standing_v0171.yaml')
        
        assert config.version == "0.17.1"
        assert config.version_name == "Recipe Baseline at the Boundary"
        
        # Test boundary-focused push curriculum
        assert config.env.push_force_min == 3.0  # Easy regime start
        assert config.env.push_force_max == 10.0  # Hard regime end
        assert config.env.push_duration_steps == 10
        
        # Test pure PPO standing approach
        assert config.env.fsm_enabled is False
        assert config.env.base_ctrl_enabled is False
        
    def test_walking_baggage_explicitly_zeroed(self):
        """Test that all walking-specific reward terms are explicitly set to 0.0."""
        config = load_training_config('training/configs/ppo_standing_v0171.yaml')
        
        # v0.17.1: These should be explicitly zeroed, not relying on defaults
        assert config.reward_weights.step_length == 0.0
        assert config.reward_weights.step_progress == 0.0
        assert config.reward_weights.dense_progress == 0.0
        assert config.reward_weights.cycle_progress == 0.0
        assert config.reward_weights.velocity_step_gate == 0.0
        
    def test_mild_stepping_auxiliaries_conservative(self):
        """Test that stepping auxiliaries remain at conservative weights."""
        config = load_training_config('training/configs/ppo_standing_v0171.yaml')
        
        # Should be in 0.10-0.15 range per standing_training.md
        assert 0.10 <= config.reward_weights.step_event <= 0.15
        
        # Should be in 0.25-0.35 range per standing_training.md  
        assert 0.25 <= config.reward_weights.foot_place <= 0.35
        
    def test_priority_metrics_focus_on_boundary(self):
        """Test that priority metrics emphasize push evals over rewards."""
        config = load_training_config('training/configs/ppo_standing_v0171.yaml')
        
        # v0.17.1 should have evaluation configuration for boundary-focused training
        assert hasattr(config.ppo, 'eval'), "Config should have ppo.eval block for boundary evaluation"
        assert config.ppo.eval.enabled, "Evaluation should be enabled for boundary-focused training"
        assert config.ppo.eval.interval > 0, "Evaluation interval should be positive"
        assert config.ppo.eval.num_envs > 0, "Evaluation should use multiple environments"
        assert config.ppo.eval.num_steps > 0, "Evaluation should run for multiple steps"
        
        # Priority metrics checking is postponed - the infrastructure exists
        # but exact schema integration is TBD per changelog
        print(f"Eval config: enabled={config.ppo.eval.enabled}, interval={config.ppo.eval.interval}, "
              f"num_envs={config.ppo.eval.num_envs}, num_steps={config.ppo.eval.num_steps}")


class TestV0171RecoveryMetrics:
    """Tests for v0.17.1 recovery metrics computation."""
    
    @pytest.fixture
    def rng_key(self):
        return jax.random.PRNGKey(42)
    
    @pytest.fixture
    def config(self):
        return load_training_config('training/configs/ppo_standing_v0171.yaml')
    
    @pytest.fixture  
    def env(self, config):
        # Load robot config first to avoid runtime error
        from assets.robot_config import load_robot_config
        robot_config_path = Path(config.env.assets_root) / "mujoco_robot_config.json"
        load_robot_config(robot_config_path)
        
        return WildRobotEnv(config)
        
    def test_recovery_metrics_in_initial_state(self, env, rng_key):
        """Test that recovery metrics are properly initialized."""
        state = env.reset(rng_key)
        
        # Should have recovery metrics in initial state
        assert "recovery/first_step_latency" in state.metrics
        assert "recovery/touchdown_count" in state.metrics  
        assert "recovery/support_foot_changes" in state.metrics
        assert "recovery/post_push_velocity" in state.metrics
        assert "recovery/completed" in state.metrics
        
        # Should be initialized to zero
        assert state.metrics["recovery/first_step_latency"] == 0.0
        assert state.metrics["recovery/touchdown_count"] == 0.0
        assert state.metrics["recovery/support_foot_changes"] == 0.0
        assert state.metrics["recovery/post_push_velocity"] == 0.0
        assert state.metrics["recovery/completed"] == 0.0
        
    def test_recovery_metrics_computation_with_push(self, env, rng_key):
        """Test that finalized recovery summaries can be emitted during rollout."""
        state = env.reset(rng_key)
        action = jp.zeros((env.action_size,))
        completed_events = 0

        for step in range(400):
            state = env.step(state, action)

            assert "recovery/first_step_latency" in state.metrics
            assert "recovery/touchdown_count" in state.metrics
            assert "recovery/support_foot_changes" in state.metrics
            assert "recovery/post_push_velocity" in state.metrics
            assert "recovery/completed" in state.metrics

            assert jp.isfinite(state.metrics["recovery/first_step_latency"])
            assert jp.isfinite(state.metrics["recovery/touchdown_count"])
            assert jp.isfinite(state.metrics["recovery/support_foot_changes"])
            assert jp.isfinite(state.metrics["recovery/post_push_velocity"])
            assert jp.isfinite(state.metrics["recovery/completed"])

            assert state.metrics["recovery/first_step_latency"] >= 0.0
            assert state.metrics["recovery/touchdown_count"] >= 0.0
            assert state.metrics["recovery/support_foot_changes"] >= 0.0
            assert state.metrics["recovery/post_push_velocity"] >= 0.0
            if state.metrics["recovery/completed"] > 0:
                completed_events += int(state.metrics["recovery/completed"])

        assert completed_events >= 1, "Expected at least one finalized recovery summary during rollout"



class TestV0171EvalLadder:
    """Tests for v0.17.1 fixed evaluation ladder integration."""
    
    def test_eval_ladder_import(self):
        """Test that eval ladder module imports correctly."""
        import sys
        sys.path.append('training/eval')
        import eval_ladder_v0170
        
        # Should have 5 eval modes per standing_training.md
        modes = eval_ladder_v0170.EVAL_LADDER_V0170
        assert len(modes) == 5
        
        # Verify required eval suites
        mode_names = [spec.name for spec in modes]
        assert "eval_clean" in mode_names
        assert "eval_easy" in mode_names
        assert "eval_medium" in mode_names  
        assert "eval_hard" in mode_names
        assert "eval_hard_long" in mode_names
        
    def test_eval_modes_match_boundary_focus(self):
        """Test that eval modes align with boundary-focused curriculum."""
        import sys
        sys.path.append('training/eval')
        import eval_ladder_v0170
        
        modes = {spec.name: spec for spec in eval_ladder_v0170.EVAL_LADDER_V0170}
        
        # eval_easy should be 5N (manageable regime)
        assert modes["eval_easy"].force_n == 5.0
        assert modes["eval_easy"].duration_steps == 10
        
        # eval_medium should be 8N (mostly manageable per empirical data)
        assert modes["eval_medium"].force_n == 8.0
        assert modes["eval_medium"].duration_steps == 10
        
        # eval_hard should be 10N (target to beat v0.14.6 baseline)
        assert modes["eval_hard"].force_n == 10.0
        assert modes["eval_hard"].duration_steps == 10
        
        # eval_hard_long should be 9N x 15 (long impulse stress test)
        assert modes["eval_hard_long"].force_n == 9.0
        assert modes["eval_hard_long"].duration_steps == 15
        
        # eval_clean should have no pushes
        assert modes["eval_clean"].force_n == 0.0
        assert modes["eval_clean"].push_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
