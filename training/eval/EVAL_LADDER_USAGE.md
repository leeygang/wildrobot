# Eval Ladder v0.17.0 - Usage Guide

## Overview

The evaluation ladder provides comprehensive post-training benchmarking for standing policies. It runs five standardized difficulty levels that are too expensive to run during training.

**Status**: Patched for robustness (v0.17.2)

## Quick Start

### RECOMMENDED: GPU Evaluation After Training Finishes

**This is the intended and practical way to validate checkpoints.**

```bash
# Stop or pause your training job first to free the GPU

uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl \
  --config training/configs/ppo_standing_v0171.yaml \
  --platform gpu \
  --suite eval_medium,eval_hard \
  --num-envs 64
```

**Time**: ~5-10 minutes  
**Output**: Critical suites, v0.17.1 target assessment

### Full GPU Evaluation (All Suites)

```bash
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl \
  --config training/configs/ppo_standing_v0171.yaml \
  --platform gpu
```

**Time**: ~10-15 minutes  
**Output**: All 5 suites, full ladder assessment

### If GPU Memory Limited (Still Fast)

```bash
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl \
  --config training/configs/ppo_standing_v0171.yaml \
  --num-envs 32
```

**Time**: ~10-15 minutes  
**Note**: Still uses GPU (fast), just smaller batch

### CPU Mode (DEBUG/EMERGENCY ONLY)

**⚠️ WARNING: CPU mode is EXTREMELY SLOW and NOT recommended for normal use**

```bash
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl \
  --config training/configs/ppo_standing_v0171.yaml \
  --platform cpu \
  --suite eval_hard \
  --num-envs 1
```

**Time**: ~30-60 minutes (for 1 suite, 1 env!)  
**When to use**: Emergency debugging when absolutely no GPU access available  
**Better solutions**: 
1. Stop training to free GPU
2. Use different machine with idle GPU
3. Reduce batch size on GPU (`--num-envs 64`)

## Evaluation Suites

| Suite | Force | Duration | Description | v0.17.1 Target |
|-------|-------|----------|-------------|----------------|
| **eval_clean** | 0N | 0 steps | No pushes (baseline) | - |
| **eval_easy** | 5N | 10 steps | Easy regime | **>95%** |
| **eval_medium** | 8N | 10 steps | Boundary regime | **>75%** |
| **eval_hard** | 10N | 10 steps | Hard regime (baseline target) | **>60%** |
| **eval_hard_long** | 9N | 15 steps | Long impulse stress test | - |

## Command-Line Options

### Required Arguments
- `--checkpoint PATH` - Path to checkpoint (.pkl file)
- `--config PATH` - Path to training config (.yaml file)

### Platform Selection
- `--platform {cpu|gpu|auto}` - JAX backend selection (default: auto)
  - `gpu` - **RECOMMENDED**: Force GPU (fast, ~10min for full ladder)
  - `auto` - Let JAX decide (default, usually picks GPU if available)
  - `cpu` - **DEBUG/EMERGENCY ONLY**: Force CPU (extremely slow, ~30-60min for 1 suite)

### Suite Filtering
- `--suite NAMES` - Comma-separated list of suites to run
  - Example: `--suite eval_medium,eval_hard`
  - Valid names: `eval_clean`, `eval_easy`, `eval_medium`, `eval_hard`, `eval_hard_long`
  - Default: all suites

### Evaluation Sizing
- `--num-envs N` - Number of parallel environments (default: 128)
- `--num-steps N` - Episode length (default: 500)
- `--seed N` - Random seed (default: 42)

### Output
- `--output PATH` - Save results to JSON file (optional)

## Output Format

The script prints:
1. **Per-suite results** - Success rate, episode length, failure modes
2. **Summary table** - All suites at a glance
3. **v0.17.1 target assessment** - Which targets are met
4. **Error summary** - Troubleshooting guidance if failures occur

Example output:
```
Running v0.17.0 evaluation ladder:
  Checkpoint: checkpoints/checkpoint_160_20971520.pkl
  Config: configs/ppo_standing_v0171.yaml
  Platform: cpu
  Envs: 128, Steps: 500
  Suites: eval_medium, eval_hard

Running eval_medium...
  Push: 8.0N x 10 steps
  Success: 78.2%
  Ep Length: 482.3
  Height Fail: 12.1%

Running eval_hard...
  Push: 10.0N x 10 steps
  Success: 62.5%
  Ep Length: 465.8
  Height Fail: 24.3%

============================================================
EVALUATION LADDER SUMMARY
============================================================
Suite           Success  Ep Len   Height Fail
------------------------------------------------------------
eval_medium       78.2%   482.3      12.1%
eval_hard         62.5%   465.8      24.3%

============================================================
V0.17.1 TARGET ASSESSMENT
============================================================
✓ eval_medium: 78.2% (target: 75.0%)
✓ eval_hard: 62.5% (target: 60.0%)

✓ ALL v0.17.1 targets met!
```

## Troubleshooting

### GPU Busy / Initialization Fails

**Error**: `cuSolver internal error`, `GPU interconnect information not available`, or CUDA errors

**Root Cause**: Usually the GPU is occupied by a training job, or GPU backend initialization fails

**RECOMMENDED SOLUTIONS** (in order of preference):

1. **Stop or pause training** to free the GPU, then rerun eval on GPU
   ```bash
   # Kill or pause your training job first, then:
   uv run python training/eval/eval_ladder_v0170.py \
     --checkpoint <path> --config <path> --platform gpu
   ```

2. **Use another machine** with idle GPU
   ```bash
   # SSH to different machine with GPU, then run eval
   ```

3. **Reduce batch size** while staying on GPU (still fast)
   ```bash
   uv run python training/eval/eval_ladder_v0170.py \
     --checkpoint <path> --config <path> --platform gpu --num-envs 64
   ```

4. **CPU mode** (LAST RESORT - extremely slow, 30-60min for minimal eval)
   ```bash
   # Only if absolutely no GPU access available
   uv run python training/eval/eval_ladder_v0170.py \
     --checkpoint <path> --config <path> --platform cpu --suite eval_hard --num-envs 1
   ```

### Out of Memory (GPU)

**Error**: CUDA out of memory

**Solution**: Reduce batch size (still uses GPU, still fast)
```bash
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint <path> --config <path> --num-envs 64
# Or even smaller:
  --num-envs 32
```

### Evaluation Takes Too Long

**Solution 1**: Run critical suites only (fastest)
```bash
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint <path> --config <path> --suite eval_medium,eval_hard
```

**Solution 2**: Reduce batch size (if using CPU mode)
```bash
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint <path> --config <path> --num-envs 1
```

## Current v0.17.1 Checkpoint

**Location**:
```
training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl
```

**Training Run**: `run-20260321_123307-0flajxzh` (iteration 160, 20.97M steps)

**Config**: `training/configs/ppo_standing_v0171.yaml`

**Recommended Command** (GPU, critical suites, practical batch size):
```bash
# Stop training first to free GPU, then:
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl \
  --config training/configs/ppo_standing_v0171.yaml \
  --platform gpu \
  --suite eval_medium,eval_hard \
  --num-envs 64
```

**Time**: ~5-10 minutes  
**Why**: Fastest practical way to validate v0.17.1 targets

## Implementation Notes

### Robustness Improvements (v0.17.2)

1. **Platform selection**: `--platform` flag sets `JAX_PLATFORM_NAME` before JAX initialization
2. **Suite filtering**: `--suite` flag allows running specific suites
3. **Error handling**: Continues on suite failures, provides actionable troubleshooting
4. **Target assessment**: Automatically compares results against v0.17.1 targets
5. **Sized fallbacks**: `--num-envs` and `--num-steps` allow debugging with smaller runs

### Why Separate from Training?

Training-time eval runs simplified push/clean passes for efficiency (single JIT compilation). This ladder requires multiple JIT compilations with different push configurations, making it too expensive for training loops. Use this for:
- Final checkpoint selection
- Branch validation
- Performance regression testing

### Checkpoint Compatibility

The ladder loads checkpoints using the same `load_checkpoint()` function as training, ensuring schema compatibility. All v0.17.x standing checkpoints are supported without code changes.
