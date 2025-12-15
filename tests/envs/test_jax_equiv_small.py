import numpy as np
from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv

def run_env(cfg, steps=20):
    env = WildRobotEnv(cfg)
    _ = env.reset()
    acts = np.zeros((cfg.num_envs, env.ACT_DIM), dtype=np.float32)
    qpos_list=[]
    qvel_list=[]
    obs_list=[]
    rew_list=[]
    done_list=[]
    for _ in range(steps):
        obs, rew, done, info = env.step(acts)
        # If JAX batched data is present, prefer reading qpos/qvel from it
        if getattr(env, '_jax_batch', None) is not None:
            try:
                qpos_list.append(np.array(env._jax_batch.qpos, dtype=np.float32))
                qvel_list.append(np.array(env._jax_batch.qvel, dtype=np.float32))
            except Exception:
                qpos_list.append(env.qpos.copy())
                qvel_list.append(env.qvel.copy())
        else:
            qpos_list.append(env.qpos.copy())
            qvel_list.append(env.qvel.copy())
        obs_list.append(obs.copy())
        rew_list.append(rew.copy())
        done_list.append(done.copy())
    return qpos_list, qvel_list, obs_list, rew_list, done_list, env


def test_jax_vs_mjx_small():
    # Create a single JAX-enabled env and use MJX stepping on reference datas
    cfg = EnvConfig(num_envs=4, seed=123, max_episode_steps=500, use_jax=True)

    # env_jax will be used for JAX path; env_mjx_ref will be simulated via mjx.step using host datas
    out_jax = run_env(cfg, steps=20)
    qpos2, qvel2, obs2, rew2, done2, env_jax = out_jax

    # build MJX reference by stepping mjx.Data manually using same initial host state
    try:
        import importlib
        mod = importlib.import_module('playground_amp.envs.wildrobot_env')
        mjx = getattr(mod, 'mjx')
    except Exception:
        import mujoco as mjx

    # prepare per-env ref datas
    ref_datas = []
    for i in range(env_jax.num_envs):
        try:
            d = mjx.make_data(env_jax._models[i] if getattr(env_jax, '_per_env_models', False) else env_jax.model)
        except Exception:
            d = mjx.make_data(env_jax.model)
        ref_datas.append(d)

    # reset ref qpos/qvel from env_jax initial reset
    qpos_ref = env_jax.qpos.copy()
    qvel_ref = env_jax.qvel.copy()

    qpos_list = []
    qvel_list = []
    obs_list = []
    rew_list = []
    done_list = []

    for t in range(20):
        # torques from PD
        torques = np.zeros((env_jax.num_envs, ref_datas[0].ctrl.size), dtype=np.float32)
        acts = np.zeros((env_jax.num_envs, env_jax.ACT_DIM), dtype=np.float32)
        for i in range(env_jax.num_envs):
            tor = env_jax._pd_control(i, acts[i])
            n = min(torques.shape[1], tor.size)
            torques[i, :n] = tor[:n]

        for i in range(env_jax.num_envs):
            d = ref_datas[i]
            try:
                d.qpos[:] = qpos_ref[i]
                d.qvel[:] = qvel_ref[i]
            except Exception:
                try:
                    setattr(d, 'qpos', np.array(qpos_ref[i], dtype=np.float32))
                    setattr(d, 'qvel', np.array(qvel_ref[i], dtype=np.float32))
                except Exception:
                    pass
            try:
                d.ctrl[:] = torques[i]
            except Exception:
                try:
                    setattr(d, 'ctrl', np.array(torques[i], dtype=np.float32))
                except Exception:
                    pass
            mjx.step(env_jax.model, d)
            qpos_ref[i] = np.array(d.qpos, dtype=np.float32)
            qvel_ref[i] = np.array(d.qvel, dtype=np.float32)

        # build obs from ref
        for i in range(env_jax.num_envs):
            env_jax.qpos[i] = qpos_ref[i]
            env_jax.qvel[i] = qvel_ref[i]
        obs_r = np.zeros((env_jax.num_envs, env_jax.OBS_DIM), dtype=np.float32)
        for i in range(env_jax.num_envs):
            obs_r[i] = env_jax._build_obs(i)
        qpos_list.append(qpos_ref.copy())
        qvel_list.append(qvel_ref.copy())
        obs_list.append(obs_r.copy())
        # compute reward via env helper for consistency
        rew = np.zeros(env_jax.num_envs, dtype=np.float32)
        done = np.zeros(env_jax.num_envs, dtype=bool)
        for i in range(env_jax.num_envs):
            rew[i] = env_jax._compute_reward(i, obs_r[i], np.zeros(ref_datas[0].ctrl.size, dtype=np.float32))
            done[i] = env_jax._is_done(i, obs_r[i])
        rew_list.append(rew.copy())
        done_list.append(done.copy())

    qpos1, qvel1, obs1, rew1, done1 = qpos_list, qvel_list, obs_list, rew_list, done_list

    # Determine joint slice mapping (skip floating-base DOFs)
    base_dofs = 7 if env_jax.nq >= 7 else 0
    joint_n = min(env_jax.ACT_DIM, env_jax.nv, env_jax.nq - base_dofs)

    # compare per-step for joint-relevant slices and rewards/dones
    for t in range(len(qpos1)):
        if joint_n > 0:
            jp1 = qpos1[t][:, base_dofs:base_dofs+joint_n]
            jp2 = qpos2[t][:, base_dofs:base_dofs+joint_n]
            np.testing.assert_allclose(jp1, jp2, atol=1e-4, rtol=1e-6)

            jv1 = qvel1[t][:, :joint_n]
            jv2 = qvel2[t][:, :joint_n]
            np.testing.assert_allclose(jv1, jv2, atol=1e-4, rtol=1e-6)

        # Compare rewards and done flags strictly
        np.testing.assert_allclose(rew1[t], rew2[t], atol=1e-5, rtol=1e-6)
        assert np.array_equal(done1[t], done2[t])

if __name__ == '__main__':
    test_jax_vs_mjx_small()
    print('JAX vs MJX joint-slice equivalence test passed')
