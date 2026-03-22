# Eval Ladder Messaging Cleanup - Summary

**Date**: 2026-03-22  
**Status**: ✅ Complete

## What Was Done

Cleaned up the eval ladder messaging to be honest and practical about GPU vs CPU usage.

### Problem

The previous messaging implied CPU mode was a normal/practical fallback for GPU failures, but in reality:
- **CPU mode is extremely slow** (30-60min for minimal eval, hours for full ladder)
- **GPU failures are usually due to training occupying the GPU**, not fundamental incompatibility
- **The practical solution is to free the GPU**, not switch to CPU

### Solution

Updated all messaging to:
1. **Emphasize GPU as primary and practical** (~10-15min for full ladder)
2. **Label CPU as debug/emergency only** (extremely slow)
3. **Recommend freeing GPU first** when troubleshooting
4. **Keep CPU mode functional** but with honest performance expectations

## Files Changed

### Code Changes (5 edits)
**`training/eval/eval_ladder_v0170.py`**:
1. Updated docstring examples (lines 23-35) - GPU examples first, CPU labeled "DEBUG/EMERGENCY ONLY"
2. Updated `_set_jax_platform()` docstring and warnings (lines 76-101) - CPU warning printed
3. Updated CLI epilog examples (lines 205-217) - GPU workflow emphasized
4. Updated `--platform` help text (line 223-224) - CPU labeled "debug/emergency only, VERY slow"
5. Updated error troubleshooting (lines 366-374) - Recommends freeing GPU first (4 steps)

### Documentation Changes (4 files)
**`training/eval/EVAL_LADDER_USAGE.md`** (4 major edits):
1. Quick Start section - GPU examples first, CPU section labeled "DEBUG/EMERGENCY ONLY"
2. Platform Selection - GPU described as "RECOMMENDED: fast", CPU as "DEBUG/EMERGENCY ONLY"
3. Troubleshooting - 4-step approach (free GPU → other machine → reduce batch → CPU last resort)
4. Recommended Command - Uses GPU with `--num-envs 64`

**`training/eval/FINAL_STATUS_REPORT.md`** (complete rewrite):
- Honest assessment: CPU works but impractical
- Emphasizes GPU as intended path
- Root cause analysis: training occupies GPU
- Recommended commands all use GPU

**`training/eval/CPU_ISOLATION_TEST.md`** (4 edits):
- Added purpose statement clarifying CPU is debug-only
- Updated recommendations section with 4-step troubleshooting
- Updated implementation status table
- Added "Documentation Updated" section

**`training/eval/EVAL_LADDER_PATCH_REPORT.md`** (6 edits):
- Updated executive summary with root cause insight
- Updated features list to emphasize messaging
- Updated files changed section to note messaging cleanup
- Updated commands section to use GPU
- Updated limitations to explain GPU occupancy
- Updated conclusion to emphasize GPU-first workflow

## Key Messaging Changes

### Before (Misleading)
- "Run on CPU if GPU fails" (implied CPU was practical)
- CPU presented as normal fallback option
- Help examples showed CPU mode prominently
- Error messages jumped to CPU recommendation

### After (Honest)
- "GPU is recommended and fast (~10min)"
- "CPU is debug/emergency only, extremely slow (~30-60min)"
- Help examples emphasize GPU-first workflow
- Error messages recommend: free GPU → other machine → reduce batch → CPU last resort
- Warning printed when entering CPU mode

## Test Results

✅ **19/19 tests passing** (excluding slow end-to-end test)

```bash
uv run pytest tests/test_eval_ladder_cli.py -xvs -k "not EndToEnd"
```

All unit tests still pass, verifying that messaging changes didn't break functionality.

## Recommended Command for v0.17.1 Checkpoint

**Stop or pause training first to free the GPU**

```bash
cd /home/leeygang/projects/wildrobot

uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl \
  --config training/configs/ppo_standing_v0171.yaml \
  --platform gpu \
  --suite eval_medium,eval_hard \
  --num-envs 64
```

**Time**: ~5-10 minutes  
**Output**: Critical suites, v0.17.1 target assessment (easy>95%, medium>75%, hard>60%)  
**Why**: Fastest practical way to validate checkpoint meets v0.17.1 targets

### Alternative: Full Ladder on GPU

```bash
uv run python training/eval/eval_ladder_v0170.py \
  --checkpoint training/checkpoints/ppo_standing_v0171_v00171_20260321_123310-0flajxzh/checkpoint_160_20971520.pkl \
  --config training/configs/ppo_standing_v0171.yaml \
  --platform gpu
```

**Time**: ~10-15 minutes  
**Output**: All 5 suites

## What Users Should Do

### If GPU Evaluation Fails

**RECOMMENDED SOLUTIONS** (in order of preference):

1. **Stop or pause training** to free the GPU
   ```bash
   # Kill or pause training job, then rerun eval on GPU
   ```

2. **Use another machine** with idle GPU
   
3. **Reduce batch size** while staying on GPU (still fast)
   ```bash
   --platform gpu --num-envs 64
   # Or even smaller: --num-envs 32
   ```

4. **CPU mode** (LAST RESORT - extremely slow)
   ```bash
   # Only if absolutely no GPU access available
   --platform cpu --suite eval_hard --num-envs 1
   ```

## What's Actually Useful

The features that **actually help** checkpoint validation:

1. **Freeing GPU** (most important) - Stop training before eval
2. **Suite filtering** - `--suite eval_medium,eval_hard` reduces 15min → 5min
3. **Batch reduction** - `--num-envs 64` helps GPU memory while staying fast
4. **Better error messages** - Clear troubleshooting guidance
5. **Target assessment** - Automatic v0.17.1 pass/fail

## Conclusion

✅ **Eval ladder messaging is now honest and practical**:
- GPU is primary (fast, ~10min)
- CPU is debug-only (slow, ~30-60min)
- Users know to free GPU first, not jump to CPU
- All documentation aligned
- All tests passing
- Recommended command uses GPU

**Status**: Ready for production use with correct expectations.
