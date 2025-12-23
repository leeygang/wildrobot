# Minimal smoke PPO-like update using Flax + Optax to validate JAX-native training plumbing
import os
os.environ.setdefault('JAX_PLATFORM_NAME', os.environ.get('JAX_PLATFORM_NAME','cpu'))

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as np
from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv

class SimplePolicy(nn.Module):
    act_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.elu(x)
        x = nn.Dense(64)(x)
        x = nn.elu(x)
        mu = nn.Dense(self.act_dim)(x)
        return mu

def main():
    cfg = EnvConfig(num_envs=16, seed=0, use_jax=True)
    env = WildRobotEnv(cfg)
    obs = env.reset()
    obs_jax = jnp.array(obs)

    rng = jax.random.PRNGKey(0)
    model = SimplePolicy(act_dim=env.ACT_DIM)
    params = model.init(rng, obs_jax[0])

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, obs_batch):
        mu = jax.vmap(lambda o: model.apply(params, o))(obs_batch)
        # dummy loss: encourage actions near zero
        return jnp.mean(jnp.square(mu))

    @jax.jit
    def update(params, opt_state, obs_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, obs_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Run a few iterations of collect+update
    for it in range(3):
        # collect one step batch
        acts = env.random_action()
        obs, rew, done, info = env.step(acts)
        obs_jax = jnp.array(obs)
        params, opt_state, loss = update(params, opt_state, obs_jax)
        print(f'iter {it}: loss={loss}')

if __name__=='__main__':
    main()
