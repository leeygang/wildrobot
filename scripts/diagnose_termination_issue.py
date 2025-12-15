#!/usr/bin/env python3
"""Diagnose the early termination issue by inspecting reset state and first few steps."""
from __future__ import annotations

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from playground_amp.envs.jax_env_fns import (
        build_obs_from_state_j,
        is_done_from_state_j,
    )
    from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Please ensure you're in the correct virtual environment and have installed dependencies."
    )
    exit(1)


def diagnose():
    """Diagnose termination issue by inspecting state after reset."""
    print("=" * 80)
    print("DIAGNOSING EARLY TERMINATION ISSUE")
    print("=" * 80)

    # Create env with minimal config
    cfg = EnvConfig(num_envs=4, seed=0, use_jax=True, max_episode_steps=1000)
    cfg.obs_noise_std = 0.0

    print("\nCreating environment...")
    env = WildRobotEnv(cfg)

    print("\nResetting environment...")
    obs = env.reset()

    print("\n" + "=" * 80)
    print("AFTER RESET")
    print("=" * 80)

    # Check internal state
    print(f"\nInternal qpos[0] (first 10): {env.qpos[0, :10]}")
    print(f"Internal qvel[0] (first 10): {env.qvel[0, :10]}")

    # Check observation
    print(f"\nObservation[0] shape: {obs[0].shape}")
    print(f"Observation[0] base_height (obs[-6]): {obs[0, -6]}")
    print(f"Observation[0] pitch (obs[-5]): {obs[0, -5]}")
    print(f"Observation[0] roll (obs[-4]): {obs[0, -4]}")

    # Check JAX batch state if available
    if hasattr(env, "_jax_batch") and env._jax_batch is not None:
        jb = env._jax_batch
        print(f"\nJAX batch qpos[0] (first 10): {jb.qpos[0, :10]}")
        print(f"JAX batch qvel[0] (first 10): {jb.qvel[0, :10]}")
        if jb.xpos is not None:
            print(f"JAX batch xpos[0]: {jb.xpos[0]}")
        if jb.xquat is not None:
            print(f"JAX batch xquat[0]: {jb.xquat[0]}")

    # Now step once with zero action
    print("\n" + "=" * 80)
    print("STEP 1 (zero action)")
    print("=" * 80)

    zero_action = np.zeros((env.num_envs, env.ACT_DIM), dtype=np.float32)
    obs1, rew1, done1, info1 = env.step(zero_action)

    print(f"\nObservation[0] base_height: {obs1[0, -6]}")
    print(f"Observation[0] pitch: {obs1[0, -5]}")
    print(f"Observation[0] roll: {obs1[0, -4]}")
    print(f"Done flags: {done1}")
    print(f"Number of terminations: {np.sum(done1)}")

    # Check JAX batch state after step
    if hasattr(env, "_jax_batch") and env._jax_batch is not None:
        jb = env._jax_batch
        print(f"\nJAX batch qpos[0] (first 10): {jb.qpos[0, :10]}")
        if jb.xpos is not None:
            print(f"JAX batch xpos[0]: {jb.xpos[0]}")

    # Step again
    print("\n" + "=" * 80)
    print("STEP 2 (zero action)")
    print("=" * 80)

    obs2, rew2, done2, info2 = env.step(zero_action)

    print(f"\nObservation[0] base_height: {obs2[0, -6]}")
    print(f"Observation[0] pitch: {obs2[0, -5]}")
    print(f"Observation[0] roll: {obs2[0, -4]}")
    print(f"Done flags: {done2}")
    print(f"Number of terminations: {np.sum(done2)}")

    # Manually check termination condition using JAX helpers
    print("\n" + "=" * 80)
    print("MANUAL TERMINATION CHECK (using JAX helpers)")
    print("=" * 80)

    if hasattr(env, "_jax_batch") and env._jax_batch is not None:
        jb = env._jax_batch
        for i in range(min(2, env.num_envs)):
            qpos_i = jb.qpos[i]
            qvel_i = jb.qvel[i]
            obs_i = obs2[i]

            derived = {}
            if jb.xpos is not None:
                derived["xpos"] = jb.xpos[i]
            if jb.xquat is not None:
                derived["xquat"] = jb.xquat[i]

            is_done = is_done_from_state_j(
                qpos_i,
                qvel_i,
                obs_i,
                derived=derived,
                max_episode_steps=1000,
                step_count=2,
            )

            print(f"\nEnv {i}:")
            print(f"  qpos[2] (base z): {qpos_i[2] if qpos_i.size > 2 else 'N/A'}")
            print(f"  obs[-6] (base_height): {obs_i[-6]}")
            print(f"  xpos from derived: {derived.get('xpos', 'N/A')}")
            print(f"  is_done (manual check): {is_done}")
            print(f"  done flag from env: {done2[i]}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    if np.sum(done2) > 0:
        print("\n❌ ISSUE CONFIRMED: Early terminations detected at step 2!")
        print("\nLikely causes:")
        print("1. base_height in observation is not being computed correctly")
        print(
            "2. xpos initialization in JaxData may be zeros instead of actual base position"
        )
        print("3. Mismatch between qpos[2] (base z-coordinate) and derived xpos")

        print("\nRecommended fixes:")
        print(
            "1. Initialize xpos in JaxData.make_jax_data with qpos[:3] instead of zeros"
        )
        print("2. Ensure reset properly initializes JAX batch xpos from qpos")
        print("3. Add a settle period (few steps) before enabling termination checks")
    else:
        print("\n✓ No early terminations detected in this test")


if __name__ == "__main__":
    diagnose()
