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


def test_jax_fullstate_relaxed():
    cfg1 = EnvConfig(num_envs=4, seed=123, max_episode_steps=500, use_jax=False)
    cfg2 = EnvConfig(num_envs=4, seed=123, max_episode_steps=500, use_jax=True)

    # create jax env and mjx reference as in equivalence test
    cfg = EnvConfig(num_envs=4, seed=123, max_episode_steps=500, use_jax=True)
    out_jax = run_env(cfg, steps=10)
    qpos2, qvel2, obs2, rew2, done2, env_jax = out_jax

    try:
        import importlib
        mod = importlib.import_module('playground_amp.envs.wildrobot_env')
        mjx = getattr(mod, 'mjx')
    except Exception:
        import mujoco as mjx

    ref_datas = []
    for i in range(env_jax.num_envs):
        try:
            d = mjx.make_data(env_jax._models[i] if getattr(env_jax, '_per_env_models', False) else env_jax.model)
        except Exception:
            d = mjx.make_data(env_jax.model)
        ref_datas.append(d)

    # simulate MJX reference
    qpos_ref = env_jax.qpos.copy()
    qvel_ref = env_jax.qvel.copy()
    qpos_list = []
    qvel_list = []
    obs_list = []
    rew_list = []
    done_list = []

    for t in range(10):
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
        for i in range(env_jax.num_envs):
            env_jax.qpos[i] = qpos_ref[i]
            env_jax.qvel[i] = qvel_ref[i]
        obs_r = np.zeros((env_jax.num_envs, env_jax.OBS_DIM), dtype=np.float32)
        for i in range(env_jax.num_envs):
            obs_r[i] = env_jax._build_obs(i)
        qpos_list.append(qpos_ref.copy())
        qvel_list.append(qvel_ref.copy())
        obs_list.append(obs_r.copy())
        rew = np.zeros(env_jax.num_envs, dtype=np.float32)
        done = np.zeros(env_jax.num_envs, dtype=bool)
        for i in range(env_jax.num_envs):
            rew[i] = env_jax._compute_reward(i, obs_r[i], np.zeros(ref_datas[0].ctrl.size, dtype=np.float32))
            done[i] = env_jax._is_done(i, obs_r[i])
        rew_list.append(rew.copy())
        done_list.append(done.copy())

    qpos1, qvel1, obs1, rew1, done1 = qpos_list, qvel_list, obs_list, rew_list, done_list
    env1 = env_jax

    # joint slices
    base_dofs = 7 if env1.nq >= 7 else 0
    joint_n = min(env1.ACT_DIM, env1.nv, env1.nq - base_dofs)

    for t in range(len(qpos1)):
        if joint_n>0:
            np.testing.assert_allclose(qpos1[t][:, base_dofs:base_dofs+joint_n], qpos2[t][:, base_dofs:base_dofs+joint_n], atol=1e-4)
            np.testing.assert_allclose(qvel1[t][:, :joint_n], qvel2[t][:, :joint_n], atol=1e-4)
        # base pos relaxed tolerance
        if base_dofs>=3:
            bp1 = qpos1[t][:, :3]
            bp2 = qpos2[t][:, :3]
            assert np.allclose(bp1, bp2, atol=0.06)
        # rewards and done
        np.testing.assert_allclose(rew1[t], rew2[t], atol=1e-4)
        assert np.array_equal(done1[t], done2[t])

if __name__=='__main__':
    test_jax_fullstate_relaxed()
    print('fullstate relaxed test passed')
