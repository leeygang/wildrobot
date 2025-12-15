import os
import numpy as np
import importlib
from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv

print('Starting debug_jax_vs_mjx')
# Force CPU JAX via env if requested
os.environ.setdefault('JAX_PLATFORM_NAME', os.environ.get('JAX_PLATFORM_NAME','cpu'))

# import the mjx module as used by the env implementation
mod = importlib.import_module('playground_amp.envs.wildrobot_env')
mjx = getattr(mod, 'mjx')

cfg = EnvConfig(num_envs=4, seed=123, max_episode_steps=200, use_jax=True)
env = WildRobotEnv(cfg)
print('env created; jax_available=', env._jax_available)
# reset to deterministic state
obs = env.reset()
print('initial obs mean', obs.mean())
print('nq,nv,ACT_DIM,ctrl_size:', env.nq, env.nv, env.ACT_DIM, getattr(env._datas[0],'ctrl').size)

# prepare zero actions
acts = np.zeros((env.num_envs, env.ACT_DIM), dtype=np.float32)
steps = 10

# ensure jax batched data is initialized from host qpos/qvel
try:
    import jax.numpy as jnp
    if getattr(env, '_jax_batch', None) is not None:
        d = env._jax_batch
        env._jax_batch = type(d)(qpos=jnp.array(env.qpos, dtype=jnp.float32), qvel=jnp.array(env.qvel, dtype=jnp.float32), ctrl=jnp.zeros((env.num_envs, env.nv), dtype=jnp.float32), xpos=d.xpos, xquat=d.xquat)
        print('initialized jax_batch from host state')
except Exception:
    pass

# prepare MJX reference datas using the same model
ref_datas = []
for i in range(env.num_envs):
    try:
        d = mjx.make_data(env._models[i] if getattr(env, '_per_env_models', False) else env.model)
    except Exception:
        d = env._datas[i]
    ref_datas.append(d)

for t in range(steps):
    # compute torques using env PD for consistency
    torques = np.zeros((env.num_envs, ref_datas[0].ctrl.size), dtype=np.float32)
    for i in range(env.num_envs):
        tor = env._pd_control(i, acts[i])
        n = min(torques.shape[1], tor.size)
        torques[i, :n] = tor[:n]

    # MJX reference step
    qpos_ref = np.zeros_like(env.qpos)
    qvel_ref = np.zeros_like(env.qvel)
    for i in range(env.num_envs):
        d = ref_datas[i]
        try:
            d.qpos[:] = env.qpos[i]
            d.qvel[:] = env.qvel[i]
        except Exception:
            try:
                setattr(d, 'qpos', np.array(env.qpos[i], dtype=np.float32))
                setattr(d, 'qvel', np.array(env.qvel[i], dtype=np.float32))
            except Exception:
                pass
        try:
            d.ctrl[:] = torques[i]
        except Exception:
            try:
                setattr(d, 'ctrl', np.array(torques[i], dtype=np.float32))
            except Exception:
                pass
        # step
        mjx.step(env.model, d)
        qpos_ref[i] = np.array(d.qpos, dtype=np.float32)
        qvel_ref[i] = np.array(d.qvel, dtype=np.float32)

    # JAX batched step
    try:
        import jax
        import jax.numpy as jnp
        newd_batch, obs_batch, rew_batch, done_batch = env._jitted_step_and_observe(env._jax_batch, jnp.array(torques, dtype=jnp.float32), dt=env._dt, kp=env.kp, kd=env.kd, obs_noise_std=env.cfg.obs_noise_std, key=None)
        qpos_jax = np.array(newd_batch.qpos, dtype=np.float32)
        qvel_jax = np.array(newd_batch.qvel, dtype=np.float32)
        obs_jax = np.array(obs_batch, dtype=np.float32) if obs_batch is not None else None
    except Exception as e:
        print('JAX step failed:', e)
        raise

    # compute host obs for ref by temporarily setting qpos/qvel and calling _build_obs
    obs_ref = np.zeros_like(env.obs)
    for i in range(env.num_envs):
        env.qpos[i] = qpos_ref[i]
        env.qvel[i] = qvel_ref[i]
        obs_ref[i] = env._build_obs(i)

    # compare
    qpos_diff = np.abs(qpos_ref - qpos_jax)
    qvel_diff = np.abs(qvel_ref - qvel_jax)
    obs_diff = np.abs(obs_ref - obs_jax) if obs_jax is not None else np.zeros_like(obs_ref)

    print(f'Step {t}: qpos max diff={qpos_diff.max():.6f}, qvel max diff={qvel_diff.max():.6f}, obs max diff={obs_diff.max():.6f}')
    # compare joint DOFs only (best-effort mapping)
    base_dofs = 7
    joint_n = min(env.ACT_DIM, env.nv, env.nq - base_dofs)
    print('comparing joint slice length', joint_n)
    if joint_n>0:
        qpos_ref_j = qpos_ref[:, base_dofs:base_dofs+joint_n]
        qpos_jax_j = qpos_jax[:, base_dofs:base_dofs+joint_n]
        print('joint qpos max diff:', np.abs(qpos_ref_j - qpos_jax_j).max())
    # print first few differing indices if large
    if qpos_diff.max() > 1e-4 or qvel_diff.max() > 1e-4 or (obs_jax is not None and obs_diff.max() > 1e-3):
        print('SHAPES: qpos_ref', qpos_ref.shape, 'qpos_jax', qpos_jax.shape)
        print('qpos_ref[0][:20]:', qpos_ref[0][:20])
        print('qpos_jax[0][:20]:', qpos_jax[0][:20])
        if obs_jax is not None:
            print('obs_ref[0][:20]:', obs_ref[0][:20])
            print('obs_jax shape:', getattr(obs_jax,'shape',None))
            if obs_jax.ndim==2:
                print('obs_jax[0][:20]:', obs_jax[0][:20])
            else:
                print('obs_jax (flat):', obs_jax[:20])
        break

    # advance env host state to ref (so next step uses same start)
    env.qpos[:] = qpos_ref
    env.qvel[:] = qvel_ref
    # also replace jax batch to continue stepping
    env._jax_batch = newd_batch

print('debug run complete')
