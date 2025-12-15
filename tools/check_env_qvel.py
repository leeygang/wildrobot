from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv
import numpy as np
cfg=EnvConfig(num_envs=4, seed=123, use_jax=True)
env=WildRobotEnv(cfg)
env.reset()
acts = env.random_action()
obs, rew, done, info = env.step(acts)
print('env.qvel[0][:12]=', env.qvel[0,:12])
if getattr(env,'_jax_batch',None) is not None:
    import numpy as np
    print('_jax_batch.qvel[0][:12]=', np.array(env._jax_batch.qvel)[0,:12])
else:
    print('_jax_batch is None')
