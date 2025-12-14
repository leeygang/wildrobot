import os
import sys
import numpy as np

# Ensure repository root is on sys.path so `playground_amp` imports work when run directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from playground_amp.envs.wildrobot_env import WildRobotEnv, EnvConfig


def run_rollout(env, actions_seq):
    obs = env.reset()
    rews = []
    dones = []
    for a in actions_seq:
        # broadcast action to all envs in the batch
        act = np.tile(a[None, :], (env.num_envs, 1))
        o, r, d, _ = env.step(act)
        rews.append(r.copy())
        dones.append(d.copy())
    return np.stack(rews), np.stack(dones), env.obs.copy()


def test_jax_vs_mjx_randomized_equiv():
    # Small vectorized test (2 envs) with deterministic DR and no obs noise
    cfg_m = EnvConfig()
    cfg_m.num_envs = 2
    cfg_m.use_jax = False
    cfg_m.obs_noise_std = 0.0

    cfg_j = EnvConfig()
    cfg_j.num_envs = 2
    cfg_j.use_jax = True
    cfg_j.obs_noise_std = 0.0

    env_m = WildRobotEnv(config=cfg_m)
    env_j = WildRobotEnv(config=cfg_j)

    # Disable domain randomization variability for determinism
    env_m._dr_params = [{} for _ in range(env_m.num_envs)]
    env_j._dr_params = [{} for _ in range(env_j.num_envs)]

    # Use same initial host qpos/qvel
    env_j.qpos = env_m.qpos.copy()
    env_j.qvel = env_m.qvel.copy()

    rng = np.random.RandomState(123)
    steps = 30
    actions_seq = [rng.uniform(-0.3, 0.3, size=(env_m.ACT_DIM,)).astype(np.float32) for _ in range(steps)]

    rews_m, dones_m, obs_m = run_rollout(env_m, actions_seq)
    rews_j, dones_j, obs_j = run_rollout(env_j, actions_seq)

    # Compare mean rewards and final observations approximately
    mean_m = np.mean(rews_m)
    mean_j = np.mean(rews_j)
    assert np.isfinite(mean_m) and np.isfinite(mean_j)
    assert abs(mean_m - mean_j) < 1e-1, f"Mean reward mismatch: {mean_m} vs {mean_j}"

    # Dones should match across the batch for short rollouts
    assert np.array_equal(dones_m, dones_j), "Done arrays differ between MJX and JAX paths"

    # Final obs closeness (loose tolerance due to integrator differences)
    assert np.allclose(obs_m, obs_j, atol=1e-2, rtol=1e-2), "Final observations diverge beyond tolerance"


if __name__ == "__main__":
    test_jax_vs_mjx_randomized_equiv()
    print("Expanded JAX vs MJX randomized equivalence test passed")
