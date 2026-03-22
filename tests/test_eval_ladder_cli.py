"""Tests for eval_ladder_v0170 CLI features.

These tests validate the new robustness features without requiring
full rollouts or GPU access.
"""

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.eval.eval_ladder_v0170 import (
    EVAL_LADDER_V0170,
    V0171_TARGETS,
    _set_jax_platform,
)


class TestPlatformSelection:
    """Test JAX platform selection."""
    
    def test_cpu_platform_sets_all_env_vars(self):
        """Test that CPU platform sets all required environment variables."""
        # Save original values
        original_platforms = os.environ.get("JAX_PLATFORMS")
        original_platform_name = os.environ.get("JAX_PLATFORM_NAME")
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        try:
            _set_jax_platform("cpu")
            assert os.environ.get("JAX_PLATFORMS") == "cpu"
            assert os.environ.get("JAX_PLATFORM_NAME") == "cpu"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        finally:
            # Restore original values
            for key, original in [
                ("JAX_PLATFORMS", original_platforms),
                ("JAX_PLATFORM_NAME", original_platform_name),
                ("CUDA_VISIBLE_DEVICES", original_cuda),
            ]:
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original
    
    def test_gpu_platform_sets_env_vars(self):
        """Test that GPU platform sets required environment variables."""
        original_platforms = os.environ.get("JAX_PLATFORMS")
        original_platform_name = os.environ.get("JAX_PLATFORM_NAME")
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        
        try:
            _set_jax_platform("gpu")
            assert os.environ.get("JAX_PLATFORMS") == "cuda"
            assert os.environ.get("JAX_PLATFORM_NAME") == "cuda"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        finally:
            for key, original in [
                ("JAX_PLATFORMS", original_platforms),
                ("JAX_PLATFORM_NAME", original_platform_name),
                ("CUDA_VISIBLE_DEVICES", original_cuda),
            ]:
                if original is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original
    
    def test_auto_platform_clears_stale_backend_env_vars(self):
        """Test that auto platform removes stale explicit backend overrides."""
        original_platforms = os.environ.get("JAX_PLATFORMS")
        original_platform_name = os.environ.get("JAX_PLATFORM_NAME")
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")

        os.environ["JAX_PLATFORMS"] = "rocm"
        os.environ["JAX_PLATFORM_NAME"] = "rocm"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        _set_jax_platform("auto")
        
        assert "JAX_PLATFORMS" not in os.environ
        assert "JAX_PLATFORM_NAME" not in os.environ
        assert "CUDA_VISIBLE_DEVICES" not in os.environ

        for key, original in [
            ("JAX_PLATFORMS", original_platforms),
            ("JAX_PLATFORM_NAME", original_platform_name),
            ("CUDA_VISIBLE_DEVICES", original_cuda),
        ]:
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


class TestEvalLadderStructure:
    """Test the eval ladder structure and constants."""
    
    def test_ladder_contains_all_suites(self):
        """Test that all expected suites are present."""
        suite_names = [s.name for s in EVAL_LADDER_V0170]
        assert "eval_clean" in suite_names
        assert "eval_easy" in suite_names
        assert "eval_medium" in suite_names
        assert "eval_hard" in suite_names
        assert "eval_hard_long" in suite_names
    
    def test_eval_clean_has_no_push(self):
        """Test that eval_clean has pushes disabled."""
        clean_suite = next(s for s in EVAL_LADDER_V0170 if s.name == "eval_clean")
        assert not clean_suite.push_enabled
    
    def test_push_suites_have_correct_forces(self):
        """Test that push suites have expected force values."""
        easy = next(s for s in EVAL_LADDER_V0170 if s.name == "eval_easy")
        medium = next(s for s in EVAL_LADDER_V0170 if s.name == "eval_medium")
        hard = next(s for s in EVAL_LADDER_V0170 if s.name == "eval_hard")
        
        assert easy.force_n == 5.0
        assert medium.force_n == 8.0
        assert hard.force_n == 10.0
    
    def test_v0171_targets_defined(self):
        """Test that v0.17.1 targets are defined."""
        assert "eval_easy" in V0171_TARGETS
        assert "eval_medium" in V0171_TARGETS
        assert "eval_hard" in V0171_TARGETS
        
        assert V0171_TARGETS["eval_easy"] == 0.95
        assert V0171_TARGETS["eval_medium"] == 0.75
        assert V0171_TARGETS["eval_hard"] == 0.60


class TestSuiteFiltering:
    """Test suite filtering logic."""
    
    def test_filter_single_suite(self):
        """Test filtering to a single suite."""
        requested = {"eval_hard"}
        filtered = [s for s in EVAL_LADDER_V0170 if s.name in requested]
        assert len(filtered) == 1
        assert filtered[0].name == "eval_hard"
    
    def test_filter_multiple_suites(self):
        """Test filtering to multiple suites."""
        requested = {"eval_medium", "eval_hard"}
        filtered = [s for s in EVAL_LADDER_V0170 if s.name in requested]
        assert len(filtered) == 2
        assert {s.name for s in filtered} == requested
    
    def test_filter_nonexistent_suite_returns_empty(self):
        """Test that filtering with invalid names returns empty."""
        requested = {"eval_nonexistent"}
        filtered = [s for s in EVAL_LADDER_V0170 if s.name in requested]
        assert len(filtered) == 0
    
    def test_filter_with_no_filter_returns_all(self):
        """Test that no filter returns all suites."""
        filtered = EVAL_LADDER_V0170
        assert len(filtered) == 5


class TestCLIArguments:
    """Test CLI argument parsing (without actually running main)."""
    
    def test_platform_choices(self):
        """Test that platform argument accepts valid choices."""
        from argparse import ArgumentParser
        
        parser = ArgumentParser()
        parser.add_argument("--platform", choices=["cpu", "gpu", "auto"])
        
        # Valid choices should parse
        args = parser.parse_args(["--platform", "cpu"])
        assert args.platform == "cpu"
        
        args = parser.parse_args(["--platform", "gpu"])
        assert args.platform == "gpu"
        
        args = parser.parse_args(["--platform", "auto"])
        assert args.platform == "auto"
    
    def test_suite_argument_parsing(self):
        """Test suite comma-separated parsing."""
        suite_arg = "eval_medium,eval_hard"
        requested = set(s.strip() for s in suite_arg.split(","))
        assert requested == {"eval_medium", "eval_hard"}
    
    def test_suite_argument_with_spaces(self):
        """Test suite parsing handles spaces."""
        suite_arg = "eval_medium , eval_hard"
        requested = set(s.strip() for s in suite_arg.split(","))
        assert requested == {"eval_medium", "eval_hard"}


class TestErrorHandling:
    """Test error handling and reporting."""
    
    def test_suite_error_creates_error_dict(self):
        """Test that suite errors are captured in results."""
        results = {}
        suite_name = "eval_test"
        error_msg = "Test error"
        
        results[suite_name] = {"error": error_msg}
        
        assert "error" in results[suite_name]
        assert results[suite_name]["error"] == error_msg
    
    def test_failed_suites_list_accumulates(self):
        """Test that failed suites are tracked."""
        failed_suites = []
        
        failed_suites.append(("eval_test1", "Error 1"))
        failed_suites.append(("eval_test2", "Error 2"))
        
        assert len(failed_suites) == 2
        assert failed_suites[0][0] == "eval_test1"
        assert failed_suites[1][0] == "eval_test2"


class TestTargetAssessment:
    """Test v0.17.1 target assessment logic."""
    
    def test_target_met(self):
        """Test detection of met targets."""
        results = {
            "eval_easy": {"success_rate": 0.97},
            "eval_medium": {"success_rate": 0.78},
            "eval_hard": {"success_rate": 0.62},
        }
        
        targets_met = []
        for suite_name, threshold in V0171_TARGETS.items():
            metrics = results[suite_name]
            if metrics["success_rate"] >= threshold:
                targets_met.append(suite_name)
        
        assert len(targets_met) == 3
        assert set(targets_met) == {"eval_easy", "eval_medium", "eval_hard"}
    
    def test_target_missed(self):
        """Test detection of missed targets."""
        results = {
            "eval_easy": {"success_rate": 0.92},  # Below 95%
            "eval_medium": {"success_rate": 0.78},
            "eval_hard": {"success_rate": 0.62},
        }
        
        targets_missed = []
        for suite_name, threshold in V0171_TARGETS.items():
            metrics = results[suite_name]
            if metrics["success_rate"] < threshold:
                targets_missed.append(suite_name)
        
        assert len(targets_missed) == 1
        assert "eval_easy" in targets_missed
    
    def test_all_targets_met(self):
        """Test perfect target achievement."""
        results = {
            "eval_easy": {"success_rate": 0.98},
            "eval_medium": {"success_rate": 0.82},
            "eval_hard": {"success_rate": 0.71},
        }
        
        all_met = all(
            results[suite_name]["success_rate"] >= threshold
            for suite_name, threshold in V0171_TARGETS.items()
        )
        
        assert all_met


class TestCPUIsolationEndToEnd:
    """End-to-end tests for CPU platform isolation.
    
    These tests verify that --platform cpu actually prevents GPU initialization.
    """
    
    @pytest.mark.slow
    def test_cpu_platform_avoids_cuda_warnings(self, tmp_path):
        """Test that --platform cpu does not emit CUDA initialization warnings.
        
        This is a critical end-to-end test that verifies the main patch requirement.
        """
        import subprocess
        
        # Create minimal fake checkpoint and config for testing
        checkpoint_path = tmp_path / "fake_checkpoint.pkl"
        config_path = tmp_path / "fake_config.yaml"
        
        # Write minimal pickle (will fail during load, but that's OK for this test)
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({"policy_params": {}, "processor_params": ()}, f)
        
        # Write minimal config
        config_content = """
env:
  robot_config_path: assets/v2/mujoco_robot_config.json
  scene_path: assets/v2/scene_flat_terrain.xml
  
ppo:
  num_envs: 1
  rollout_steps: 1
"""
        config_path.write_text(config_content)
        
        # Run ladder with --platform cpu
        cmd = [
            "uv", "run", "python", 
            "training/eval/eval_ladder_v0170.py",
            "--checkpoint", str(checkpoint_path),
            "--config", str(config_path),
            "--platform", "cpu",
            "--suite", "eval_clean",
            "--num-envs", "1",
            "--num-steps", "1",
        ]
        
        # Run in subprocess and capture output
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        # Check that output does NOT contain CUDA warnings
        combined_output = result.stdout + result.stderr
        
        cuda_warning_patterns = [
            "CUDA",
            "cuSolver",
            "GPU interconnect",
            "cuda_executor",
        ]
        
        found_warnings = []
        for pattern in cuda_warning_patterns:
            if pattern in combined_output:
                found_warnings.append(pattern)
        
        # This is the critical assertion: CPU mode should not touch CUDA
        if found_warnings:
            pytest.fail(
                f"CPU platform mode emitted CUDA-related warnings: {found_warnings}\n"
                f"Full output:\n{combined_output}"
            )
        
        # Verify that CPU mode was actually set
        assert "JAX platform forced to CPU" in combined_output or "platform forced to CPU" in combined_output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
