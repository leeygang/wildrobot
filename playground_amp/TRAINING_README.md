# WildRobot Training Guide

## Overview

The WildRobot project uses a **single unified training script** (`train.py`) that combines:
- **Brax PPO trainer** - Battle-tested, GPU-optimized PPO implementation
- **AMP support** - Adversarial Motion Priors for natural motion learning
- **Pure-JAX backend** - Full JIT/vmap compatibility for maximum performance

## Quick Start

### Smoke Test (Verify Setup)
```bash
python playground_amp/train.py --verify
```
This runs a quick 10-iteration test with 4 environments to verify everything works.

### CPU Training (Development)
```bash
python playground_amp/train.py --num-iterations 100 --num-envs 16
```
Suitable for development and debugging on machines without GPU.

### GPU Training (Production)
```bash
python playground_amp/train.py --num-iterations 3000 --num-envs 2048
```
Production-scale training for GPU-enabled machines.

### Enable AMP (Natural Motion Learning)
```bash
python playground_amp/train.py --enable-amp --amp-weight 1.0
```
Enables Adversarial Motion Priors for learning natural, human-like gaits.

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to training config YAML | `configs/wildrobot_phase3_training.yaml` |
| `--verify` | Quick smoke test mode (10 iters, 4 envs) | False |
| `--num-envs` | Number of parallel environments | 16 |
| `--num-iterations` | Number of training iterations | 100 |
| `--enable-amp` | Enable AMP discriminator | False |
| `--amp-weight` | AMP reward weight (1.0 = equal to task) | 1.0 |
| `--no-wandb` | Disable WandB logging | False |
| `--checkpoint-dir` | Directory for checkpoints | `checkpoints/brax_ppo` |
| `--set` | Override config (e.g., `--set lr=0.001`) | None |

## Configuration Files

### Training Config
`playground_amp/configs/wildrobot_phase3_training.yaml`

All configuration is consolidated in this single file:
- **`env:`** - Environment parameters (timing, robot params, action filtering, episode length)
- **`trainer:`** - PPO hyperparameters (learning rate, gamma, epochs, etc.)
- **`reward_weights:`** - Reward function weights
- **`domain_randomization:`** - Domain randomization settings
- **`amp:`** - AMP configuration (if enabled)
- **`quick_verify:`** - Override settings for smoke tests

## Examples

### Override Config Values
```bash
python playground_amp/train.py \
    --set trainer.lr=0.001 \
    --set trainer.gamma=0.95 \
    --set trainer.num_envs=32
```

### Full Training Run with AMP
```bash
python playground_amp/train.py \
    --num-iterations 3000 \
    --num-envs 2048 \
    --enable-amp \
    --amp-weight 1.0 \
    --checkpoint-dir checkpoints/production_run
```

### Development Iteration (Fast)
```bash
python playground_amp/train.py \
    --verify \
    --no-wandb
```

## Output and Checkpoints

### Console Output
Training progress is logged every 1000 steps:
```
Step 1,000: reward=-25.43, length=150.5, policy_loss=-0.0023, value_loss=2.45
Step 2,000: reward=-18.22, length=245.8, policy_loss=-0.0019, value_loss=1.87
...
```

### Checkpoints
Saved to `checkpoints/brax_ppo/final_iter_<N>.pkl`:
```python
{
    "params": <policy parameters>,
    "metrics": <training metrics>,
    "config": <PPO config>,
    "amp_discriminator": <discriminator> (if AMP enabled),
    "amp_weight": <weight> (if AMP enabled)
}
```

### WandB Logging
If not disabled, metrics are automatically logged to Weights & Biases:
- Project: `wildrobot-brax-ppo`
- Run name: `brax_ppo_<timestamp>` or `brax_ppo_amp_<timestamp>`

## Architecture Details

### Training Pipeline
1. **Environment Creation**: `BraxWildRobotWrapper` with Pure-JAX backend
2. **PPO Trainer**: Brax's `ppo.train()` function
3. **Progress Callback**: Logs metrics every 1000 steps
4. **Checkpoint Saving**: Saves final policy + metrics

### AMP Integration (Task 5)
When `--enable-amp` is enabled:
1. **Discriminator**: Initialized from `playground_amp/amp/discriminator.py`
2. **Reference Buffer**: Loads reference motion data
3. **Training**: AMP reward added to task rewards
4. **Metrics**: AMP reward and discriminator accuracy logged

**Note**: Full AMP training requires reference motion data (see Phase 3 Plan, Task 5).

## Troubleshooting

### Import Error: "Failed to import Brax PPO"
```bash
pip install brax>=0.10.0
```

### AMP Initialization Failed
This is expected if AMP components are not yet implemented. Training continues without AMP.

### Out of Memory (GPU)
Reduce `--num-envs`:
```bash
python playground_amp/train.py --num-envs 512  # Instead of 2048
```

### Slow Training (CPU)
- Use `--verify` for quick smoke tests
- Reduce `--num-envs` to 4-16
- For production training, use GPU

## Development History

### Why Only One Training Script?

Previously, the project had 4 training scripts:
1. `train_jax.py` - Test stub (39 lines)
2. `train_brax.py` - Early Brax integration (97 lines)
3. `train_brax_ppo.py` - Brax PPO integration (259 lines)
4. `train.py` - Self-contained PPO+AMP (549 lines)

**Consolidated to single `train.py`** (2025-12-16):
- ✅ Best of both worlds: Brax PPO + AMP support
- ✅ Single source of truth (no confusion)
- ✅ Easier maintenance
- ✅ Task 10 validated: Brax integration works
- ✅ Task 5 ready: AMP hooks in place

## Related Documentation

- **Phase 3 Training Plan**: `playground_amp/phase3_rl_training_plan.md`
- **Task 10 Progress**: `docs/task10_phase2_progress.md`
- **Section 3.1.2 Status**: `docs/section_3_1_2_status.md`
- **Training Scripts Comparison** (historical): `docs/training_scripts_comparison.md`

## Next Steps (Phase 3)

1. **Task 5**: Complete AMP integration
   - Prepare reference motion dataset
   - Implement discriminator training loop
   - Test end-to-end with `--enable-amp`

2. **Task 7**: Re-enable Optax/Flax (blocked by Task 1a)

3. **GPU Deployment**: Set up cloud GPU for production training
   - Configuration: 2048 envs, 3000+ iterations
   - Expected: 10,000-50,000 steps/sec

## Support

For issues or questions:
1. Check Phase 3 plan for task status
2. Review diagnostics reports in `docs/`
3. Run `--verify` to isolate issues
4. Check console output for error messages
