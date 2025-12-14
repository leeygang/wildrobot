"""Minimal JAX trainer for playground_amp using WildRobotEnv.

This is a lightweight, self-contained training script that:
- Instantiates `playground_amp.envs.wildrobot_env.WildRobotEnv`
- Uses a small MLP policy (Gaussian) and value network in JAX
- Collects rollouts from the vectorized env and runs simple policy-gradient + value updates

This trainer is intentionally small for development and smoke training. It does NOT
implement full PPO clipping, minibatching, or advanced optimizers — those can be
added later. It avoids importing code from `mujoco-brax` as requested.
"""

from __future__ import annotations

import time
import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np

from playground_amp.envs.wildrobot_env import EnvConfig, WildRobotEnv

# Prefer Flax + Optax when available for clean optax updates and Flax models
use_flax = False
try:
    import flax.linen as nn
    import optax
    use_flax = True
except Exception:
    try:
        import optax
        use_flax = False
    except Exception:
        use_flax = False


@dataclass
class TrainerConfig:
    num_envs: int = 8
    rollout_steps: int = 64
    total_updates: int = 20
    lr: float = 1e-3
    gamma: float = 0.99
    seed: int = 0
    policy_layers: list = None
    value_layers: list = None


if use_flax:
    class MLP(nn.Module):
        layer_sizes: tuple

        @nn.compact
        def __call__(self, x):
            h = x
            for i, size in enumerate(self.layer_sizes[1:]):
                h = nn.Dense(size)(h)
                if i < len(self.layer_sizes[1:]) - 1:
                    h = nn.tanh(h)
            return h
else:
    def init_mlp(layer_sizes, key, scale=0.1):
        params = []
        ks = jax.random.split(key, len(layer_sizes) - 1)
        for k, (n_in, n_out) in zip(ks, zip(layer_sizes[:-1], layer_sizes[1:])):
            w = scale * jax.random.normal(k, (n_in, n_out))
            b = jnp.zeros((n_out,))
            params.append((w, b))
        return params


    def mlp_apply(params, x):
        h = x
        for i, (w, b) in enumerate(params):
            h = jnp.dot(h, w) + b
            if i < len(params) - 1:
                h = jnp.tanh(h)
        return h


def gaussian_log_prob(mean, log_std, a):
    var = jnp.exp(2 * log_std)
    return -0.5 * (((a - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi)).sum(axis=-1)


def discount_cumsum(x, gamma):
    out = np.zeros_like(x)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + gamma * running
        out[t] = running
    return out


# ----- Optimizer and loss helpers (top-level, no nested functions) -----
def tree_zeros_like(pytree):
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), pytree)


def adam_init(params):
    # params is a list of (w,b) tuples from init_mlp; mirror that structure for m and v
    m = []
    v = []
    for (w, b) in params:
        m.append((jnp.zeros_like(w), jnp.zeros_like(b)))
        v.append((jnp.zeros_like(w), jnp.zeros_like(b)))
    return {'m': m, 'v': v, 't': 0}


def adam_update(params, grads, opt_state, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
    # Manual Adam for params structured as list of (w,b) tuples.
    # Use numpy for the manual updates to avoid JAX array-indexing issues
    t = opt_state['t'] + 1
    new_m = []
    new_v = []
    params_updated = []
    for (p_w, p_b), (g_w, g_b), (m_w, m_b), (v_w, v_b) in zip(params, grads, opt_state['m'], opt_state['v']):
        # convert to numpy for safe arithmetic
        pw = np.array(p_w)
        pb = np.array(p_b)
        gw = np.array(g_w)
        gb = np.array(g_b)
        mw = np.array(m_w)
        mb = np.array(m_b)
        vw = np.array(v_w)
        vb = np.array(v_b)

        m_w_new = beta1 * mw + (1 - beta1) * gw
        m_b_new = beta1 * mb + (1 - beta1) * gb

        v_w_new = beta2 * vw + (1 - beta2) * (gw * gw)
        v_b_new = beta2 * vb + (1 - beta2) * (gb * gb)

        m_hat_w = m_w_new / (1 - beta1 ** t)
        m_hat_b = m_b_new / (1 - beta1 ** t)
        v_hat_w = v_w_new / (1 - beta2 ** t)
        v_hat_b = v_b_new / (1 - beta2 ** t)

        pw_new = pw - lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
        pb_new = pb - lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # convert back to jnp arrays for consistency with rest of code
        p_w_new = jnp.array(pw_new)
        p_b_new = jnp.array(pb_new)

        new_m.append((jnp.array(m_w_new), jnp.array(m_b_new)))
        new_v.append((jnp.array(v_w_new), jnp.array(v_b_new)))
        params_updated.append((p_w_new, p_b_new))

    return params_updated, {'m': new_m, 'v': new_v, 't': t}


def save_checkpoint(ckpt_dir: str, step: int, policy_p, value_p, log_std_p):
    os.makedirs(ckpt_dir, exist_ok=True)
    fn = f"{ckpt_dir}/ckpt_{step}.npz"
    # minimal checkpoint: save log_std; more complete saving can be added later
    np.savez(fn, log_std=np.array(log_std_p))
    print(f"Saved checkpoint {fn}")


def ppo_policy_loss(p_params, log_std_param, mb_obs, mb_acts, mb_old_logp, mb_adv, ppo_clip):
    mean = mlp_apply(p_params, jnp.array(mb_obs))
    logp = gaussian_log_prob(mean, log_std_param, jnp.array(mb_acts))
    ratio = jnp.exp(logp - jnp.array(mb_old_logp))
    clipped = jnp.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
    loss = -jnp.mean(jnp.minimum(ratio * jnp.array(mb_adv), clipped * jnp.array(mb_adv)))
    return loss


def value_loss(value_params, mb_obs, mb_ret):
    vpred = mlp_apply(value_params, jnp.array(mb_obs)).squeeze(-1)
    return jnp.mean((jnp.array(mb_ret) - vpred) ** 2)





def load_trainer_config(path: str) -> Tuple[dict, dict]:
    """Load a trainer-only config file (YAML/JSON/TOML).

    Returns (trainer_dict, raw_dict).
    """
    data = None
    try:
        import yaml

        with open(path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception:
        pass

    if data is None:
        import json

        with open(path, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                try:
                    import tomllib

                    with open(path, 'rb') as fb:
                        data = tomllib.load(fb)
                except Exception:
                    raise RuntimeError(f"Failed to parse config file: {path}. Install pyyaml or provide JSON/TOML.")

    raw = data or {}
    trainer_raw = raw.get('trainer', raw)
    return trainer_raw, raw


def apply_overrides(cfg: dict, overrides: list):
    """Apply list of overrides of the form ['trainer.num_envs=512', ...] to cfg dict."""
    for ov in overrides or []:
        if '=' not in ov:
            continue
        key, val = ov.split('=', 1)
        # nested keys with dot notation
        parts = key.split('.')
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        # try to cast value to int/float/bool if appropriate
        v: Any
        s = val.strip()
        if s.lower() in ('true', 'false'):
            v = s.lower() == 'true'
        else:
            try:
                if '.' in s:
                    v = float(s)
                    # if integer-like, cast
                    if v.is_integer():
                        v = int(v)
                else:
                    v = int(s)
            except Exception:
                v = s
        d[parts[-1]] = v
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='playground_amp/configs/wildrobot_phase3_training.yaml', help='Path to trainer stage config (YAML/JSON/TOML)')
    parser.add_argument('--set', action='append', help='Override config key, format: key.subkey=value (can be used multiple times)')
    parser.add_argument('--verify', action='store_true', help='Apply `quick_verify` overrides from the stage config for a minimal end-to-end check')
    parser.add_argument('--enable-amp', action='store_true', help='Enable AMP discriminator and reference-motion buffer hooks')
    args = parser.parse_args()

    # Load env config (shared) and trainer config (stage-specific)
    env_cfg = EnvConfig.from_file('playground_amp/configs/wildrobot_env.yaml')
    trainer_raw, raw = load_trainer_config(args.config)
    # If quick_verify requested, apply those small-test overrides first (so CLI --set can still override)
    if args.verify and 'quick_verify' in raw:
        qv = raw['quick_verify']
        trainer_q = qv.get('trainer', {})
        # merge trainer quick-verify entries into trainer_raw
        for k, v in trainer_q.items():
            trainer_raw[k] = v
        # apply env quick-verify overrides onto env_cfg
        env_q = qv.get('env', {})
        for k, v in env_q.items():
            # only set existing attributes if present on EnvConfig, else add
            try:
                setattr(env_cfg, k, v)
            except Exception:
                env_cfg.__dict__[k] = v

    # apply CLI overrides (highest precedence)
    trainer_raw = apply_overrides(trainer_raw, args.set)
    # build TrainerConfig with defaults
    trainer_kwargs = {k: trainer_raw[k] for k in ('num_envs', 'rollout_steps', 'total_updates', 'lr', 'gamma', 'seed') if k in trainer_raw}
    cfg = TrainerConfig(**trainer_kwargs)
    # ensure trainer and env agree on num_envs
    if getattr(cfg, 'num_envs', None) is not None:
        env_cfg.num_envs = cfg.num_envs
    env = WildRobotEnv(env_cfg)

    # AMP hooks (optional)
    enable_amp = args.enable_amp
    amp_disc = None
    ref_buffer = None
    if enable_amp:
        try:
            from playground_amp.amp.discriminator import AMPDiscriminator
            from playground_amp.amp.ref_buffer import ReferenceMotionBuffer

            amp_disc = AMPDiscriminator(input_dim=env.OBS_DIM)
            ref_buffer = ReferenceMotionBuffer(max_size=raw.get('amp', {}).get('buffer_size', 1000), seq_len=raw.get('amp', {}).get('seq_len', 32))
            print('AMP hooks enabled: discriminator + reference buffer initialized')
        except Exception as e:
            print(f'Failed to initialize AMP hooks ({e}); continuing without AMP')

    obs_dim = env.OBS_DIM
    act_dim = env.ACT_DIM

    key = jax.random.PRNGKey(cfg.seed)
    key, pk, vk = jax.random.split(key, 3)

    # Policy: MLP -> mean; separate log_std param
    policy_layers = [obs_dim, 64, 64, act_dim]
    value_layers = [obs_dim, 64, 64, 1]

    # Initialize policy/value as Flax models when available, otherwise use init_mlp
    if use_flax:
        policy_model = MLP(tuple(policy_layers))
        value_model = MLP(tuple(value_layers))
        # Flax init expects batch shape; pass zeros with batch dim
        policy_params = policy_model.init(pk, jnp.zeros((1, obs_dim)))
        value_params = value_model.init(vk, jnp.zeros((1, obs_dim)))
        # pack into a single params pytree for optax updates
        params = {'policy': policy_params, 'value': value_params, 'log_std': jnp.zeros((act_dim,)) - 1.0}
        # define flax loss wrappers now that models exist
        def ppo_policy_loss_flax(params, mb_obs, mb_acts, mb_old_logp, mb_adv, ppo_clip):
            mean = jnp.asarray(policy_model.apply(params['policy'], jnp.array(mb_obs)))
            logp = gaussian_log_prob(mean, params['log_std'], jnp.array(mb_acts))
            ratio = jnp.exp(logp - jnp.array(mb_old_logp))
            clipped = jnp.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip)
            loss = -jnp.mean(jnp.minimum(ratio * jnp.array(mb_adv), clipped * jnp.array(mb_adv)))
            return loss

        def value_loss_flax(params, mb_obs, mb_ret):
            vpred = jnp.asarray(value_model.apply(params['value'], jnp.array(mb_obs))).squeeze(-1)
            return jnp.mean((jnp.array(mb_ret) - vpred) ** 2)
    else:
        policy_params = init_mlp(policy_layers, pk)
        value_params = init_mlp(value_layers, vk)
        log_std = jnp.zeros((act_dim,)) - 1.0

    # Detect SOTA JAX libraries (flax + rlax) and optax for optimized implementations.
    use_sota = False
    try:
        import flax
        import rlax
        use_sota = True
    except Exception:
        use_sota = False

    # Prefer optax if available for optimizer; otherwise fall back to manual Adam
    use_optax = False
    try:
        import optax

        use_optax = True
    except Exception:
        use_optax = False

    if use_flax and 'optax' in globals():
        tx = optax.adam(cfg.lr)
        opt_state = tx.init(params)
        print('Using Flax+Optax for training')
    if use_sota:
        print('Detected flax+rlax available — consider refactoring to rlax PPO + flax models for SOTA implementation')
    if not use_flax:
        p_opt = adam_init(policy_params)
        v_opt = adam_init(value_params)
    if use_flax:
        log_std_m = jnp.zeros_like(params['log_std'])
        log_std_v = jnp.zeros_like(params['log_std'])
        log_std_t = 0
    else:
        log_std_m = jnp.zeros_like(log_std)
        log_std_v = jnp.zeros_like(log_std)
        log_std_t = 0

    # Simple SGD updater for list-of-(w,b) parameter structure
    def sgd_update(params, grads, lr):
        new_params = []
        for (p_w, p_b), (g_w, g_b) in zip(params, grads):
            new_w = jnp.array(np.array(p_w) - lr * np.array(g_w))
            new_b = jnp.array(np.array(p_b) - lr * np.array(g_b))
            new_params.append((new_w, new_b))
        return new_params

    ckpt_dir = raw.get('trainer', {}).get('checkpoint_dir', 'playground_amp/checkpoints')

    # PPO hyperparams (from config or defaults)
    ppo_clip = raw.get('trainer', {}).get('ppo_clip', 0.2)
    epochs = raw.get('trainer', {}).get('epochs', 2)
    minibatch_size = raw.get('trainer', {}).get('minibatch_size', 128)

    # Main training loop (collect rollouts, then update with PPO-style clipped loss)
    obs = env.reset()

    # Quick-verify mode: run small number of forward steps without doing any gradient updates
    if args.verify:
        print('Running quick-verify: stepping env with policy forward (no optimization)')
        total_rew = 0.0
        # small per-env ring buffers to collect ref-motion sequences when AMP enabled
        per_env_seq = [[] for _ in range(cfg.num_envs)]
        for t in range(cfg.rollout_steps):
            if use_flax:
                mean = policy_model.apply(params['policy'], jnp.array(obs))
            else:
                mean = mlp_apply(policy_params, jnp.array(obs))
            acts = np.array(mean)
            obs, rews, dones, infos = env.step(acts)
            # AMP discriminator scoring (optional): add discriminator score to rewards
            if amp_disc is not None:
                try:
                    scores = amp_disc.score(obs)
                    # amp weight from config
                    w = float(raw.get('amp', {}).get('weight', 1.0))
                    rews = rews + (w * np.array(scores))
                except Exception:
                    pass
            total_rew += float(np.mean(rews))
            # collect ref-motion snippets if enabled
            if ref_buffer is not None:
                for n in range(cfg.num_envs):
                    per_env_seq[n].append(obs[n].copy())
                    if len(per_env_seq[n]) >= ref_buffer.seq_len:
                        seq = np.stack(per_env_seq[n][-ref_buffer.seq_len :], axis=0)
                        ref_buffer.add(seq)
        print(f'Quick-verify complete: avg reward per step {total_rew / max(1, cfg.rollout_steps):.6f}')
        return
    for update in range(cfg.total_updates):
        batch_obs = []
        batch_acts = []
        batch_rews = []
        batch_vals = []
        batch_logp = []

        for t in range(cfg.rollout_steps):
            if use_flax:
                mean = jnp.array(policy_model.apply(params['policy'], jnp.array(obs)))
            else:
                mean = jnp.array(mlp_apply(policy_params, jnp.array(obs)))
            std = jnp.exp(log_std)
            key, sub = jax.random.split(key)
            acts = np.array(mean + std * jax.random.normal(sub, mean.shape))

            if use_flax:
                vals = np.array(value_model.apply(params['value'], jnp.array(obs))).squeeze(-1)
            else:
                vals = np.array(mlp_apply(value_params, jnp.array(obs))).squeeze(-1)

            next_obs, rews, dones, infos = env.step(acts)
            # AMP scoring during regular rollout
            if amp_disc is not None:
                try:
                    scores = amp_disc.score(next_obs)
                    w = float(raw.get('amp', {}).get('weight', 1.0))
                    rews = rews + (w * np.array(scores))
                except Exception:
                    pass

            if use_flax:
                log_std_curr = np.array(params['log_std'])
            else:
                log_std_curr = log_std
            logp = np.array(gaussian_log_prob(mean, log_std_curr, jnp.array(acts)))

            batch_obs.append(obs.copy())
            batch_acts.append(acts)
            batch_rews.append(rews)
            batch_vals.append(vals)
            batch_logp.append(logp)

            obs = next_obs

        batch_obs = np.stack(batch_obs)  # (T, N, obs_dim)
        batch_acts = np.stack(batch_acts)
        batch_rews = np.stack(batch_rews)
        batch_vals = np.stack(batch_vals)
        batch_logp = np.stack(batch_logp)

        # returns & advantages
        returns = np.zeros_like(batch_rews)
        for n in range(cfg.num_envs):
            returns[:, n] = discount_cumsum(batch_rews[:, n], cfg.gamma)
        advantages = returns - batch_vals

        # flatten
        T, N = batch_rews.shape
        batch_size = T * N
        obs_flat = batch_obs.reshape(batch_size, obs_dim)
        acts_flat = batch_acts.reshape(batch_size, act_dim)
        adv_flat = advantages.reshape(batch_size)
        ret_flat = returns.reshape(batch_size)
        old_logp_flat = batch_logp.reshape(batch_size)

        # PPO update epochs
        idxs = np.arange(batch_size)
        for epoch in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start : start + minibatch_size]
                mb_obs = obs_flat[mb_idx]
                mb_acts = acts_flat[mb_idx]
                mb_adv = adv_flat[mb_idx]
                mb_ret = ret_flat[mb_idx]
                mb_old_logp = old_logp_flat[mb_idx]
                if use_flax:
                    # Use combined loss and optax updates on the params pytree
                    mb_obs_j = jnp.array(mb_obs)
                    mb_acts_j = jnp.array(mb_acts)
                    mb_old_logp_j = jnp.array(mb_old_logp)
                    mb_adv_j = jnp.array(mb_adv)
                    mb_ret_j = jnp.array(mb_ret)

                    def total_loss(pytree):
                        pl = ppo_policy_loss_flax(pytree, mb_obs_j, mb_acts_j, mb_old_logp_j, mb_adv_j, ppo_clip)
                        vl = value_loss_flax(pytree, mb_obs_j, mb_ret_j)
                        return pl + 0.5 * vl

                    grads = jax.grad(total_loss)(params)
                    updates, opt_state = tx.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                else:
                    p_grads = jax.grad(ppo_policy_loss, argnums=(0, 1))(policy_params, log_std, mb_obs, mb_acts, mb_old_logp, mb_adv, ppo_clip)

                    v_grads = jax.grad(value_loss)(value_params, mb_obs, mb_ret)

                    # Use SGD/manual Adam style updates
                    try:
                        policy_params = sgd_update(policy_params, p_grads[0], cfg.lr)
                        value_params = sgd_update(value_params, v_grads, cfg.lr)
                        log_std_grad = p_grads[1]
                        log_std = log_std - cfg.lr * jnp.array(log_std_grad)
                    except Exception:
                        # fallback to safe numpy updates
                        policy_params = sgd_update(policy_params, p_grads[0], cfg.lr)
                        value_params = sgd_update(value_params, v_grads, cfg.lr)
                        log_std = log_std - cfg.lr * jnp.array(p_grads[1])

        mean_return = float(np.mean(np.sum(batch_rews, axis=0)))
        print(f"update {update+1}/{cfg.total_updates} mean_return={mean_return:.4f}")

        # checkpoint
        if (update + 1) % 5 == 0:
            if use_flax:
                save_checkpoint(ckpt_dir, update + 1, params['policy'], params['value'], params['log_std'])
            else:
                save_checkpoint(ckpt_dir, update + 1, policy_params, value_params, log_std)

    print("Training (PPO-like) completed (smoke).")


if __name__ == "__main__":
    main()
