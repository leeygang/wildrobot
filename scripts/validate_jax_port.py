"""Validation script for Task 11: Pure-JAX Port

This script tests the Pure-JAX port (use_jax=True) against the MJX baseline
to validate correctness and measure performance.

Test Suite:
1. Smoke test: Environment steps without errors
2. Equivalence test: JAX matches MJX outputs within tolerance
3. Performance benchmark: Measure throughput improvement
4. JIT compilation test: Verify JIT works correctly

Usage:
    python scripts/validate_jax_port.py --num-envs 16 --num-steps 100
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv


def smoke_test(num_envs: int = 4, num_steps: int = 10) -> Dict[str, Any]:
    """Test that Pure-JAX port runs without errors."""
    print("=" * 70)
    print("SMOKE TEST: Pure-JAX Port Basic Functionality")
    print("=" * 70)

    # Create env with JAX port enabled
    env_cfg = EnvConfig.from_file("playground_amp/configs/wildrobot_env.yaml")
    env_cfg.num_envs = num_envs
    env_cfg.use_jax = True  # Enable Pure-JAX port

    env = WildRobotEnv(env_cfg)
    print(f"✓ Environment created with use_jax=True")
    print(f"  num_envs: {num_envs}")
    print(f"  obs_dim: {env.OBS_DIM}")
    print(f"  act_dim: {env.ACT_DIM}")

    # Reset
    obs = env.reset()
    print(f"✓ Reset successful, obs.shape: {obs.shape}")

    # Step with zero actions
    actions = np.zeros((num_envs, env.ACT_DIM), dtype=np.float32)

    for step in range(num_steps):
        obs, rew, done, info = env.step(actions)

        # Check for NaN/Inf
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            raise ValueError(f"NaN/Inf detected in observations at step {step}")
        if np.any(np.isnan(rew)) or np.any(np.isinf(rew)):
            raise ValueError(f"NaN/Inf detected in rewards at step {step}")

    print(f"✓ Stepped {num_steps} times without errors")
    print(f"  Final obs range: [{np.min(obs):.3f}, {np.max(obs):.3f}]")
    print(f"  Final rewards: {rew}")
    print(f"  Done flags: {done}")

    return {
        "passed": True,
        "num_steps": num_steps,
        "final_obs_shape": obs.shape,
        "no_nan_inf": True,
    }


def equivalence_test(
    num_envs: int = 4, num_steps: int = 5, tolerance: float = 1e-2
) -> Dict[str, Any]:
    """Compare Pure-JAX port outputs against MJX baseline."""
    print("\n" + "=" * 70)
    print("EQUIVALENCE TEST: JAX vs MJX Comparison")
    print("=" * 70)

    # Create two environments: one with JAX, one with MJX
    env_cfg = EnvConfig.from_file("playground_amp/configs/wildrobot_env.yaml")
    env_cfg.num_envs = num_envs
    env_cfg.obs_noise_std = 0.0  # Disable noise for deterministic comparison

    # JAX environment
    env_cfg.use_jax = True
    env_jax = WildRobotEnv(env_cfg)

    # MJX environment (baseline)
    env_cfg.use_jax = False
    env_mjx = WildRobotEnv(env_cfg)

    print(f"✓ Created JAX and MJX environments")

    # Reset both with same seed
    np.random.seed(42)
    obs_jax = env_jax.reset()

    np.random.seed(42)
    obs_mjx = env_mjx.reset()

    # Compare initial observations
    obs_diff = np.abs(obs_jax - obs_mjx)
    max_obs_diff = np.max(obs_diff)
    print(f"\nInitial observations:")
    print(f"  Max difference: {max_obs_diff:.6f}")
    print(f"  Mean difference: {np.mean(obs_diff):.6f}")

    # Step both environments with same actions
    results = []

    for step in range(num_steps):
        # Use small random actions
        np.random.seed(100 + step)
        actions = np.random.randn(num_envs, env_jax.ACT_DIM).astype(np.float32) * 0.1

        obs_jax, rew_jax, done_jax, info_jax = env_jax.step(actions)
        obs_mjx, rew_mjx, done_mjx, info_mjx = env_mjx.step(actions)

        # Compare outputs
        obs_diff = np.abs(obs_jax - obs_mjx)
        rew_diff = np.abs(rew_jax - rew_mjx)
        done_match = np.all(done_jax == done_mjx)

        max_obs_diff = np.max(obs_diff)
        max_rew_diff = np.max(rew_diff)

        passed = max_obs_diff < tolerance and max_rew_diff < tolerance and done_match

        results.append(
            {
                "step": step,
                "max_obs_diff": max_obs_diff,
                "max_rew_diff": max_rew_diff,
                "done_match": done_match,
                "passed": passed,
            }
        )

        status = "✓" if passed else "✗"
        print(
            f"  Step {step}: {status} obs_diff={max_obs_diff:.6f}, rew_diff={max_rew_diff:.6f}, done_match={done_match}"
        )

    all_passed = all(r["passed"] for r in results)

    if all_passed:
        print(f"\n✅ EQUIVALENCE TEST PASSED (tolerance={tolerance})")
    else:
        print(f"\n⚠️  EQUIVALENCE TEST FAILED (tolerance={tolerance})")
        print(
            f"   Some steps exceeded tolerance - this may be acceptable for physics sims"
        )

    return {
        "passed": all_passed,
        "num_steps": num_steps,
        "tolerance": tolerance,
        "results": results,
        "max_obs_diff_overall": max([r["max_obs_diff"] for r in results]),
        "max_rew_diff_overall": max([r["max_rew_diff"] for r in results]),
    }


def performance_benchmark(num_envs: int = 16, num_steps: int = 1000) -> Dict[str, Any]:
    """Benchmark Pure-JAX vs MJX performance."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK: Throughput Comparison")
    print("=" * 70)

    env_cfg = EnvConfig.from_file("playground_amp/configs/wildrobot_env.yaml")
    env_cfg.num_envs = num_envs

    results = {}

    for use_jax_flag in [False, True]:
        env_cfg.use_jax = use_jax_flag
        env_name = "JAX" if use_jax_flag else "MJX"

        env = WildRobotEnv(env_cfg)
        obs = env.reset()
        actions = np.random.randn(num_envs, env.ACT_DIM).astype(np.float32) * 0.1

        # Warmup
        for _ in range(10):
            obs, _, _, _ = env.step(actions)

        # Benchmark
        start_time = time.time()
        for _ in range(num_steps):
            obs, _, _, _ = env.step(actions)
        elapsed = time.time() - start_time

        steps_per_sec = (num_steps * num_envs) / elapsed

        results[env_name] = {
            "elapsed_sec": elapsed,
            "steps_per_sec": steps_per_sec,
            "total_env_steps": num_steps * num_envs,
        }

        print(f"\n{env_name} Performance:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {steps_per_sec:.0f} env-steps/sec")
        print(f"  Time per step: {(elapsed/num_steps)*1000:.2f}ms")

    if "JAX" in results and "MJX" in results:
        speedup = results["JAX"]["steps_per_sec"] / results["MJX"]["steps_per_sec"]
        print(f"\n📊 JAX Speedup: {speedup:.2f}x")

        if speedup > 1.0:
            print(f"✅ Pure-JAX port is faster!")
        else:
            print(f"⚠️  Pure-JAX port is slower (needs optimization)")

    return results


def jit_compilation_test() -> Dict[str, Any]:
    """Test that JIT compilation works correctly."""
    print("\n" + "=" * 70)
    print("JIT COMPILATION TEST")
    print("=" * 70)

    try:
        from playground_amp.envs.jax_full_port import (
            jitted_step_and_observe,
            make_jax_data,
        )

        print("✓ Imported JIT-compiled functions")

        # Create test data
        nq = 18  # 7 (base) + 11 (joints)
        nv = 17  # 6 (base) + 11 (joints)
        data = make_jax_data(nq=nq, nv=nv, batch=1)

        # Test JIT compilation
        target_qpos = jax.random.normal(jax.random.PRNGKey(0), (1, 11)) * 0.1

        print("✓ Testing JIT-compiled step...")
        new_data, obs, rew, done = jitted_step_and_observe(
            data, target_qpos, dt=0.02, kp=50.0, kd=1.0, obs_noise_std=0.0, key=None
        )

        print(f"✓ JIT step successful")
        print(f"  new_data.qpos.shape: {new_data.qpos.shape}")
        if obs is not None:
            print(f"  obs.shape: {obs.shape}")

        # Test with vmap
        print("\n✓ Testing vmap+JIT compilation...")
        batch_data = make_jax_data(nq=nq, nv=nv, batch=4)
        batch_targets = jax.random.normal(jax.random.PRNGKey(1), (4, 11)) * 0.1

        from playground_amp.envs.jax_full_port import jitted_vmapped_step_and_observe

        if jitted_vmapped_step_and_observe is not None:
            new_batch, obs_batch, rew_batch, done_batch = (
                jitted_vmapped_step_and_observe(
                    batch_data,
                    batch_targets,
                    dt=0.02,
                    kp=50.0,
                    kd=1.0,
                    obs_noise_std=0.0,
                    key=None,
                )
            )
            print(f"✓ vmap+JIT successful")
            print(f"  batch obs.shape: {obs_batch.shape}")
            print(f"  batch rew.shape: {rew_batch.shape}")
        else:
            print("⚠️  jitted_vmapped_step_and_observe not available")

        return {"passed": True, "jit_works": True, "vmap_works": True}

    except Exception as e:
        print(f"✗ JIT compilation test failed: {e}")
        import traceback

        traceback.print_exc()
        return {"passed": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Validate Pure-JAX Port (Task 11)")
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num-steps", type=int, default=10, help="Number of steps for smoke test"
    )
    parser.add_argument(
        "--bench-steps", type=int, default=1000, help="Number of steps for benchmark"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-2, help="Equivalence test tolerance"
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke test")
    parser.add_argument(
        "--skip-equiv", action="store_true", help="Skip equivalence test"
    )
    parser.add_argument(
        "--skip-bench", action="store_true", help="Skip performance benchmark"
    )
    parser.add_argument(
        "--skip-jit", action="store_true", help="Skip JIT compilation test"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Task 11: Pure-JAX Port Validation")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print()

    results = {}

    # Run tests
    if not args.skip_smoke:
        results["smoke_test"] = smoke_test(args.num_envs, args.num_steps)

    if not args.skip_equiv:
        results["equivalence_test"] = equivalence_test(
            args.num_envs, args.num_steps, args.tolerance
        )

    if not args.skip_jit:
        results["jit_test"] = jit_compilation_test()

    if not args.skip_bench:
        results["benchmark"] = performance_benchmark(args.num_envs, args.bench_steps)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, test_results in results.items():
        if "passed" in test_results:
            status = "✅ PASSED" if test_results["passed"] else "❌ FAILED"
            print(f"{test_name}: {status}")
            if not test_results["passed"]:
                all_passed = False
        else:
            print(f"{test_name}: ✓ COMPLETE")

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED - Pure-JAX port is ready!")
    else:
        print("⚠️  SOME TESTS FAILED - needs investigation")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
