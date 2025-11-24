# Headless Video Rendering Guide

This guide explains how to generate videos in headless environments (SSH, remote servers without displays).

## Overview

WildRobot supports **GPU-accelerated headless rendering** using MuJoCo's EGL backend, which works perfectly over SSH without any X server or display!

## How It Works

### EGL (Embedded Graphics Library)
- **GPU-accelerated** offscreen rendering
- Works on **headless servers** (no display needed)
- Requires **NVIDIA GPU** with EGL support
- Much faster than CPU-based rendering

### Configuration

The code automatically sets up EGL rendering:

```python
# In train.py and visualize_policy.py (line 98 and 15)
os.environ["MUJOCO_GL"] = "egl"
```

This enables GPU-based headless rendering for both training and visualization.

## Video Rendering Options

### Option 1: Disable During Training (Recommended for Quick Tests)

```yaml
# In quick.yaml
rendering:
  render_videos: false  # No videos during training
```

**Benefits:**
- ✅ Training completes in ~2 minutes
- ✅ No rendering overhead
- ✅ Can render videos later with `visualize_policy.py`

### Option 2: Render at End (Production Mode - loco-mujoco style)

```yaml
# In production.yaml
rendering:
  render_videos: true   # Render ONE video after training completes
```

**Benefits:**
- ✅ Only ONE video rendered (at the very end)
- ✅ No slowdown during training
- ✅ Automatic - no manual step needed

This is what **loco-mujoco** does!

### Option 3: Render Videos Later (Manual)

Train without videos, then render manually:

```bash
# Step 1: Train (fast, no videos)
uv run train.py --config quick.yaml

# Step 2: Render video afterwards
uv run visualize_policy.py \
  --checkpoint training/logs/wildrobot_*/checkpoints/final_policy.pkl \
  --output videos/my_policy.mp4 \
  --steps 1000
```

**Benefits:**
- ✅ Maximum training speed
- ✅ Render as many videos as you want afterwards
- ✅ Try different camera angles, seeds, etc.

## Headless Server Requirements

### NVIDIA GPU with EGL

Check if your server has EGL support:

```bash
# Check NVIDIA GPU
nvidia-smi

# Check EGL libraries
ldconfig -p | grep EGL
```

You should see output like:
```
libEGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so.1
```

### Alternative: OSMesa (CPU fallback)

If EGL is not available, MuJoCo can fall back to CPU rendering:

```python
os.environ["MUJOCO_GL"] = "osmesa"
```

**Note:** OSMesa is **much slower** (CPU-based), but works without GPU.

## Example: Full Workflow

### Quick Test (2 minutes)

```bash
# Train without videos (fast!)
cd /home/leeygang/projects/wildrobot/playground
uv run train.py --config quick.yaml

# Render video afterwards
uv run visualize_policy.py \
  --checkpoint training/logs/wildrobot_flat_*/checkpoints/final_policy.pkl \
  --output videos/quick_test.mp4 \
  --steps 500
```

### Production Training

```bash
# Train with automatic video at the end
uv run train.py --config production.yaml

# Video will be saved in: training/logs/wildrobot_*/final_policy.mp4
```

## Troubleshooting

### Issue: `Could not initialize EGL`

**Solution 1:** Make sure NVIDIA drivers are installed:
```bash
nvidia-smi  # Should show GPU info
```

**Solution 2:** Use OSMesa fallback:
```python
os.environ["MUJOCO_GL"] = "osmesa"
```

### Issue: Video rendering is slow

**Cause:** You might be using CPU rendering (OSMesa) instead of GPU (EGL).

**Solution:**
1. Verify EGL is available: `ldconfig -p | grep EGL`
2. Check `os.environ["MUJOCO_GL"]` is set to `"egl"`
3. Ensure NVIDIA GPU is available: `nvidia-smi`

### Issue: `ImportError: No module named 'mediapy'`

**Solution:**
```bash
uv add mediapy
```

## Performance Comparison

| Method | Time for 1000 frames | GPU Usage |
|--------|---------------------|-----------|
| EGL (GPU) | ~10 seconds | ✅ High |
| OSMesa (CPU) | ~2 minutes | ❌ None |
| During Training | Adds 50-100% overhead | ✅ High |

## Recommended Approach

**For WildRobot:**

1. **Quick tests** (`quick.yaml`): Disable videos, render manually later
2. **Production runs** (`production.yaml`): Enable videos, render ONE at the end
3. **Never** render videos during every evaluation (too slow!)

This approach follows **loco-mujoco's best practices** for efficient training and visualization! 🚀

## Summary

✅ **YES, videos work in headless SSH environments!**
- Uses EGL GPU-accelerated rendering
- No X server or display needed
- Works perfectly over SSH
- Fast and efficient

The key is `os.environ["MUJOCO_GL"] = "egl"` which is already set in the code! 🎉
