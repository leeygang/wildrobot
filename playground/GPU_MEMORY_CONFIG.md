# GPU Memory Configuration Guide

This guide helps you configure WildRobot training for your specific GPU memory.

## Quick Reference

| GPU Memory | Config File | num_envs | batch_size | Expected Usage |
|------------|-------------|----------|------------|----------------|
| 8GB (RTX 3060 Ti) | Create custom | 512 | 128 | ~6GB |
| 12GB (RTX 4070/5070) | `default.yaml` | 1024 | 256 | ~9GB |
| 16GB (RTX 4080) | Create custom | 1536 | 384 | ~12GB |
| 24GB (RTX 4090/A5000) | `production.yaml` | 2048 | 512 | ~15GB |

## Memory Calculation

GPU memory usage in Brax/JAX depends primarily on:

```python
memory ≈ num_envs × (state_size + observation_size + action_size) × precision
```

**Key Memory Consumers:**
1. **num_envs** - Most important! Linear scaling
2. **batch_size** - Should equal `num_envs` or divide evenly
3. **unroll_length** - Temporal rollout buffer
4. **network_size** - Policy/value network parameters

## Configurations by GPU

### 8GB GPU (RTX 3060 Ti, RTX 3070)

Create `gpu_8gb.yaml`:

```yaml
ppo:
  num_envs: 512
  num_eval_envs: 32
  batch_size: 128
  unroll_length: 5
  num_minibatches: 2

network:
  policy_hidden_layers: [128, 128]
  value_hidden_layers: [128, 128]
```

### 12GB GPU (RTX 4070, RTX 5070) - DEFAULT

Use `default.yaml` (already optimized):

```yaml
ppo:
  num_envs: 1024
  num_eval_envs: 64
  batch_size: 256
  unroll_length: 10
  num_minibatches: 4
```

### 16GB GPU (RTX 4080)

Create `gpu_16gb.yaml`:

```yaml
ppo:
  num_envs: 1536
  num_eval_envs: 96
  batch_size: 384
  unroll_length: 10
  num_minibatches: 6
```

### 24GB GPU (RTX 4090, A5000, A6000)

Use `production.yaml`:

```yaml
ppo:
  num_envs: 2048
  num_eval_envs: 128
  batch_size: 512
  unroll_length: 10
  num_minibatches: 8
```

## Troubleshooting OOM Errors

### Error: "Out of memory while trying to allocate X.XXGiB"

**Solution 1: Reduce num_envs** (most effective)
```bash
uv run train.py --config default.yaml --num_envs 512
```

**Solution 2: Reduce batch_size**
```bash
uv run train.py --config default.yaml --batch_size 128
```

**Solution 3: Use smaller network**
Edit your config:
```yaml
network:
  policy_hidden_layers: [128, 128]  # Instead of [256, 256, 128]
  value_hidden_layers: [128, 128]
```

**Solution 4: Enable XLA memory optimization**

Add to `train.py` (already done at line 97):
```python
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

### Monitoring GPU Memory

**Check current usage:**
```bash
nvidia-smi
```

**Monitor continuously:**
```bash
watch -n 1 nvidia-smi
```

**In Python (during training):**
```python
import jax
print(f"Device memory stats: {jax.local_devices()[0].memory_stats()}")
```

## Important Constraint

**Brax PPO requires:**
```python
batch_size * num_minibatches % num_envs == 0
```

**Valid combinations:**

| num_envs | batch_size | num_minibatches | Valid? |
|----------|------------|-----------------|--------|
| 1024 | 256 | 4 | ✅ (256×4=1024) |
| 1024 | 512 | 2 | ✅ (512×2=1024) |
| 1024 | 256 | 2 | ❌ (256×2=512) |
| 512 | 128 | 2 | ✅ (128×2=256... wait, this fails!) |

**Fix for 512 envs:**
```yaml
ppo:
  num_envs: 512
  batch_size: 128
  num_minibatches: 4  # 128×4=512 ✅
```

## Performance vs Memory Trade-offs

### More Environments (Higher Memory)
**Pros:**
- ✅ More sample diversity
- ✅ Better exploration
- ✅ Faster wall-clock training
- ✅ More stable gradients

**Cons:**
- ❌ Higher GPU memory usage
- ❌ Longer JIT compilation time

### Fewer Environments (Lower Memory)
**Pros:**
- ✅ Fits in smaller GPUs
- ✅ Faster JIT compilation
- ✅ Lower memory footprint

**Cons:**
- ❌ Less sample diversity
- ❌ Slower wall-clock training
- ❌ May need more total timesteps

## Recommended Settings

### For Learning/Debugging (Fastest)
```yaml
# quick.yaml
ppo:
  num_envs: 256
  batch_size: 128
  num_timesteps: 100000
```

### For 12GB GPU (Balanced)
```yaml
# default.yaml
ppo:
  num_envs: 1024
  batch_size: 256
  num_timesteps: 30000000
```

### For Production (Best Performance)
```yaml
# production.yaml
ppo:
  num_envs: 2048      # Requires 24GB GPU
  batch_size: 512
  num_timesteps: 50000000
```

## Example Workflow

### Step 1: Test with quick.yaml (always works)
```bash
uv run train.py --config quick.yaml
```

### Step 2: Try default.yaml (optimized for 12GB)
```bash
uv run train.py --config default.yaml
```

### Step 3: If OOM, reduce environments
```bash
uv run train.py --config default.yaml --num_envs 512 --batch_size 128
```

### Step 4: Monitor and adjust
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## Summary

**For NVIDIA RTX 5070 (12GB):**

✅ **Use `default.yaml`** - Already optimized for 12GB!

```yaml
ppo:
  num_envs: 1024      # ~9GB GPU usage
  batch_size: 256
```

❌ **Don't use `production.yaml`** - Requires 24GB GPU

```yaml
ppo:
  num_envs: 2048      # ~15GB GPU usage - OOM on 12GB!
  batch_size: 512
```

The updated `default.yaml` configuration is specifically tuned for your GPU! 🎉
