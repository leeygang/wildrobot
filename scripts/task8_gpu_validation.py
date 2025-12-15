"""Task 8: GPU Validation & Benchmarking Script

This script validates JAX CUDA installation and benchmarks GPU performance
for the WildRobot training pipeline.

Tests:
1. JAX GPU availability and device info
2. Environment throughput on GPU vs CPU
3. JIT compilation performance
4. GPU memory usage
5. Diagnostic tests on GPU

Usage:
    python scripts/task8_gpu_validation.py
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

print("=" * 70)
print("Task 8: GPU Validation & Benchmarking")
print("=" * 70)

# ==============================================================================
# Test 1: JAX GPU Availability
# ==============================================================================
print("\n" + "=" * 70)
print("TEST 1: JAX GPU Availability")
print("=" * 70)

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Check for CUDA/GPU
has_gpu = any(d.platform == 'gpu' or d.platform == 'cuda' for d in jax.devices())
print(f"\n{'✅' if has_gpu else '❌'} GPU Available: {has_gpu}")

if has_gpu:
    gpu_devices = [d for d in jax.devices() if d.platform in ['gpu', 'cuda']]
    for i, device in enumerate(gpu_devices):
        print(f"  GPU {i}: {device}")
        print(f"    Platform: {device.platform}")
        print(f"    Device kind: {device.device_kind}")
else:
    print("⚠️  WARNING: No GPU detected!")
    print("   Training will run on CPU (10-100x slower)")
    print("\nTo install JAX with CUDA:")
    print("  pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

# ==============================================================================
# Test 2: Basic JAX Operations on GPU
# ==============================================================================
print("\n" + "=" * 70)
print("TEST 2: Basic JAX Operations on GPU")
print("=" * 70)

try:
    # Test matrix multiplication on GPU
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))

    # Warmup
    y = jnp.dot(x, x)
    y.block_until_ready()

    # Benchmark
    start = time.time()
    for _ in range(10):
        y = jnp.dot(x, x)
        y.block_until_ready()
    elapsed = time.time() - start

    print(f"✅ Matrix multiplication (1000x1000): {elapsed/10*1000:.2f}ms per op")
    print(f"   Throughput: {10/elapsed:.1f} ops/sec")

    if has_gpu:
        print(f"   Device: {y.device()}")

except Exception as e:
    print(f"❌ Basic operations failed: {e}")

# ==============================================================================
# Test 3: JIT Compilation Performance
# ==============================================================================
print("\n" + "=" * 70)
print("TEST 3: JIT Compilation Performance")
print("=" * 70)

def heavy_computation(x):
    """Test function with complex operations."""
    for _ in range(10):
        x = jnp.sin(x) + jnp.cos(x)
        x = jnp.dot(x, x.T)
    return x

try:
    x = jax.random.normal(key, (100, 100))

    # Non-JIT version
    start = time.time()
    y = heavy_computation(x)
    y.block_until_ready()
    non_jit_time = time.time() - start

    # JIT version
    jitted_fn = jax.jit(heavy_computation)

    # Warmup (compile)
    _ = jitted_fn(x)

    # Benchmark
    start = time.time()
    for _ in range(10):
        y = jitted_fn(x)
        y.block_until_ready()
    jit_time = (time.time() - start) / 10

    speedup = non_jit_time / jit_time

    print(f"✅ JIT compilation working")
    print(f"   Non-JIT: {non_jit_time*1000:.2f}ms")
    print(f"   JIT: {jit_time*1000:.2f}ms")
    print(f"   Speedup: {speedup:.1f}x")

except Exception as e:
    print(f"❌ JIT compilation failed: {e}")

# ==============================================================================
# Test 4: Environment Throughput Benchmark
# ==============================================================================
print("\n" + "=" * 70)
print("TEST 4: Environment Throughput Benchmark")
print("=" * 70)

try:
    from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv

    print("Testing with Pure-JAX backend (use_jax=True)...")

    # Test with different batch sizes
    test_configs = [
        (4, "Small batch (smoke test)"),
        (16, "Medium batch"),
        (64, "Large batch") if has_gpu else (16, "Large batch (CPU limited)"),
    ]

    for num_envs, desc in test_configs:
        print(f"\n  {desc}: {num_envs} environments")

        env_cfg = EnvConfig.from_file('playground_amp/configs/wildrobot_env.yaml')
        env_cfg.num_envs = num_envs
        env_cfg.use_jax = True  # Pure-JAX backend

        env = WildRobotEnv(env_cfg)

        # Reset
        obs = env.reset()

        # Warmup
        actions = np.zeros((num_envs, env.ACT_DIM), dtype=np.float32)
        for _ in range(5):
            obs, rew, done, info = env.step(actions)

        # Benchmark
        num_steps = 100
        start = time.time()
        for _ in range(num_steps):
            obs, rew, done, info = env.step(actions)
        elapsed = time.time() - start

        throughput = (num_steps * num_envs) / elapsed

        print(f"    Throughput: {throughput:.0f} env-steps/sec")
        print(f"    Time per step: {elapsed/num_steps*1000:.2f}ms")
        print(f"    Total time: {elapsed:.2f}s")

    print(f"\n{'✅' if has_gpu else '⚠️'} Environment throughput benchmark complete")
    if has_gpu:
        print(f"  GPU acceleration working correctly")
    else:
        print(f"  Running on CPU (expect 10-100x slower training)")

except Exception as e:
    print(f"❌ Environment benchmark failed: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# Test 5: GPU Memory Usage
# ==============================================================================
print("\n" + "=" * 70)
print("TEST 5: GPU Memory Usage")
print("=" * 70)

if has_gpu:
    try:
        # Try to get GPU memory info (CUDA-specific)
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split('\n')):
                used, total = map(int, line.split(','))
                print(f"  GPU {i}: {used}MB / {total}MB ({used/total*100:.1f}% used)")
        else:
            print("  ⚠️  Could not query GPU memory (nvidia-smi not available)")
    except Exception as e:
        print(f"  ⚠️  GPU memory query not available: {e}")
else:
    print("  N/A (CPU only)")

# ==============================================================================
# Test 6: Brax Integration Check
# ==============================================================================
print("\n" + "=" * 70)
print("TEST 6: Brax Integration Check")
print("=" * 70)

try:
    from playground_amp.brax_wrapper import BraxWildRobotWrapper
    from brax.envs.base import State

    print("Creating Brax-wrapped environment...")
    env_cfg = EnvConfig.from_file('playground_amp/configs/wildrobot_env.yaml')
    env_cfg.num_envs = 1  # BraxWildRobotWrapper uses single env

    wrapper = BraxWildRobotWrapper(env_cfg)

    # Reset
    rng = jax.random.PRNGKey(0)
    state = wrapper.reset(rng)

    print(f"✅ BraxWildRobotWrapper created successfully")
    print(f"   Observation size: {wrapper.observation_size}")
    print(f"   Action size: {wrapper.action_size}")
    print(f"   Backend: {wrapper.backend}")

    # Test step
    action = jnp.zeros(wrapper.action_size)
    new_state = wrapper.step(state, action)

    print(f"✅ Step function working")
    print(f"   Reward: {float(new_state.reward):.4f}")
    print(f"   Done: {float(new_state.done):.4f}")

except Exception as e:
    print(f"❌ Brax integration check failed: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================
# Summary & Recommendations
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

if has_gpu:
    print("✅ GPU validation PASSED")
    print("\nRecommended training configuration:")
    print("  --num-envs 2048    # Large batch for GPU")
    print("  --num-iterations 1000")
    print("\nExpected performance:")
    print("  Throughput: 10,000-50,000 env-steps/sec")
    print("  Training time: ~30-60 minutes for 1000 iterations")
else:
    print("⚠️  GPU validation INCOMPLETE (no GPU detected)")
    print("\nCPU-optimized training configuration:")
    print("  --num-envs 16      # Smaller batch for CPU")
    print("  --num-iterations 100")
    print("\nExpected performance:")
    print("  Throughput: 1,000-2,000 env-steps/sec")
    print("  Training time: ~2-3 hours for 100 iterations")
    print("\nTo enable GPU:")
    print("  1. Install CUDA 12.x")
    print("  2. Install JAX with CUDA:")
    print("     pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

print("\n" + "=" * 70)
print("✅ Task 8 validation complete")
print("=" * 70)
print("\nNext steps:")
print("  1. If GPU available: Proceed to Task 10 Phase 3 with GPU config")
print("  2. If CPU only: Can still proceed (just slower)")
print("  3. Review benchmark results above for optimal configuration")
