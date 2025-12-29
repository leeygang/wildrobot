import os
import sys
import numpy as np

# Ensure repo root is importable when running tests directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from playground_amp.envs.wildrobot_env import WildRobotEnv, EnvConfig


def run_contact_rollout(env, actions_seq, push_step=5, push_mag=5.0):
    obs = env.reset()
    rews = []
    dones = []
    # schedule push on env 0 at push_step
    for i in range(env.num_envs):
        env._dr_params[i] = {}
    env._dr_params[0]["push_time"] = push_step
    env._dr_params[0]["push_mag"] = push_mag

    for t, a in enumerate(actions_seq):
        act = np.tile(a[None, :], (env.num_envs, 1))
        o, r, d, _ = env.step(act)
        rews.append(r.copy())
        dones.append(d.copy())
    return np.stack(rews), np.stack(dones), env.obs.copy()


def test_contact_equiv():
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

    # align initial host state
    env_j.qpos = env_m.qpos.copy()
    env_j.qvel = env_m.qvel.copy()

    rng = np.random.RandomState(456)
    steps = 20
    actions_seq = [rng.uniform(-0.3, 0.3, size=(env_m.ACT_DIM,)).astype(np.float32) for _ in range(steps)]

    rews_m, dones_m, obs_m = run_contact_rollout(env_m, actions_seq, push_step=5, push_mag=10.0)
    rews_j, dones_j, obs_j = run_contact_rollout(env_j, actions_seq, push_step=5, push_mag=10.0)

    # Ensure both ran without NaNs and produced finite rewards
    assert np.isfinite(rews_m).all() and np.isfinite(rews_j).all()

    # Dones arrays should be equal (same termination behavior)
    assert np.array_equal(dones_m, dones_j), "Dones differ under contact perturbation"

    # Allow some tolerance for obs differences after contact
    assert np.allclose(obs_m, obs_j, atol=2e-2, rtol=1e-2), "Final observations diverged beyond tolerance"


if __name__ == "__main__":
    test_contact_equiv()
    print("Contact equivalence test passed")
