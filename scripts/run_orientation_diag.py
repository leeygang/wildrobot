"""Full orientation diagnostic for WildRobotEnv.

Sets a lying quaternion into one env, writes into mjx Data, calls forward/step,
and prints observed quaternion, computed roll/pitch, and done flags.
"""
from __future__ import annotations

import os
import numpy as np
import math

from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv
from playground_amp.utils.quaternion import normalize_quat_wxyz, quat_to_euler_wxyz


def main():
    # use a small num_envs for clarity
    cfg = EnvConfig(num_envs=4, seed=7, obs_noise_std=0.0, max_episode_steps=500)
    env = WildRobotEnv(cfg)
    obs = env.reset()
    print("reset obs shape:", obs.shape)

    # 90 deg pitch quaternion in (w,x,y,z)
    q_wxyz = np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32)

    # Write into env.qpos for env index 1 and forward into Data using env helper
    idx = 1
    if env.qpos.shape[1] >= 7:
        env.qpos[idx, 3:7] = q_wxyz
    env.qpos[0, 2] = 0.1

    # Print host-side quaternion and derived roll/pitch (w,x,y,z ordering)
    try:
        from playground_amp.utils.quaternion import normalize_quat_wxyz, quat_to_euler_wxyz
        host_q = np.array(env.qpos[idx, 3:7], dtype=np.float32)
        print('host qpos for env1 (w,x,y,z):', host_q)
        hn = normalize_quat_wxyz(host_q)
        hr, hp = quat_to_euler_wxyz(hn)
        print('host-derived roll (deg):', math.degrees(hr), 'pitch (deg):', math.degrees(hp))
        print('env._is_done(host) for env1:', env._is_done(idx, obs[idx]))
    except Exception as e:
        print('host-side quat->euler failed:', e)

    try:
        # Prefer env helper which has safe fallbacks
        if hasattr(env, 'set_data_qpos'):
            env.set_data_qpos(0, env.qpos[0])
            env.set_data_qpos(1, env.qpos[1])
        else:
            # best-effort naive write
            env._datas[0].qpos[:] = env.qpos[0]
            env._datas[1].qpos[:] = env.qpos[1]
    except Exception as e:
        print('write to data failed:', e)
    # attempt forward on per-env models/data where available
    try:
        import mujoco.mjx as mjx
        for i in (0, 1):
            try:
                m = env._models[i] if env._per_env_models else env.model
                mjx.forward(m, env._datas[i])
            except Exception as e:
                print(f'forward env[{i}] skipped: {e}')
    except Exception as e:
        print('mjx.forward import failed or skipped:', e)

    # Print raw stored quaternions
    try:
        dq = env._datas[1]
        raw_q = None
        if hasattr(dq, 'xquat') and getattr(dq, 'xquat') is not None:
            qarr = np.array(dq.xquat)
            raw_q = qarr[0,:4] if qarr.ndim == 2 else qarr[:4]
        else:
            raw_q = np.array(dq.qpos[3:7], dtype=np.float32)
        print('raw stored quat for env1:', raw_q)
        nq = normalize_quat_wxyz(raw_q)
        r, p = quat_to_euler_wxyz(nq)
        print('normalized quat (w,x,y,z):', nq)
        print('roll (deg):', math.degrees(r), 'pitch (deg):', math.degrees(p))
    except Exception as e:
        print('reading quat failed:', e)

    # Step with zero actions and report dones
    acts = np.zeros((env.num_envs, env.ACT_DIM), dtype=np.float32)
    obs2, rews2, dones2, infos2 = env.step(acts)
    print('dones after step:', dones2)
    print('obs extras env1:', obs2[1, -6:])


if __name__ == '__main__':
    # Force JAX CPU backend via env var if user didn't set it
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    main()
