import numpy as np
import jax
import jax.numpy as jnp
from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv
import importlib
jax_full = importlib.import_module('playground_amp.envs.jax_full_port')

cfg=EnvConfig(num_envs=4, seed=123, use_jax=True)
env=WildRobotEnv(cfg)
print('env created, jax_available=', env._jax_available)
obs = env.reset()
print('initial qpos[0][:8]:', env.qpos[0][:8])
print('initial qvel[0][:8]:', env.qvel[0][:8])
acts = np.zeros((env.num_envs, env.ACT_DIM), dtype=np.float32)
# call jitted step
newd_batch, obs_batch, rew_batch, done_batch = env._jitted_step_and_observe(env._jax_batch, jnp.array(acts, dtype=jnp.float32), dt=env._dt, kp=env.kp, kd=env.kd, obs_noise_std=env.cfg.obs_noise_std, key=None)
print('newd_batch.qvel shape:', np.array(newd_batch.qvel).shape)
print('newd_batch.qvel[0][:12]:', np.array(newd_batch.qvel)[0,:12])
# compute PD torque manually in JAX for first env
curr_qpos = np.array(env._jax_batch.qpos)[0]
curr_qvel = np.array(env._jax_batch.qvel)[0]
print('curr_qpos slice:', curr_qpos[:11])
print('curr_qvel slice:', curr_qvel[:11])
# compute torque using jax_full.jax_pd_control
targ = jnp.array(acts, dtype=jnp.float32)
# run non-jitted pd control on CPU
torque = jax_full.jax_pd_control(targ, env._jax_batch.qpos[..., :env.nv], env._jax_batch.qvel, kp=env.kp, kd=env.kd)
print('torque.shape', np.array(torque).shape)
print('torque[0][:12]:', np.array(torque)[0,:12])
