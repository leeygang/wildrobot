---
name: wildrobot-training-analyze
description: Analyze WildRobot training runs from an offline W&B run folder (training/wandb/offline-run-*/), summarize stability/success metrics, pick the best checkpoint for deployment (standing_push prioritizes eval_push success rate then episode length), propose v0.xx+1 config/reward/code changes, and update training/CHANGELOG.md with results and next-version changes. Use when asked to review training logs, select checkpoints, compare versions (e.g. v0.13.6 vs v0.13.7), or plan the next training iteration (v0.xx+1).
---

# WildRobot Training Analyze

Use this workflow to (1) analyze a training run, (2) choose a deployable checkpoint, (3) identify the dominant failure mode, and (4) write the result + next-version plan into `training/CHANGELOG.md`.

## Inputs (what you need)

- Offline run dir: `training/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>/`
- For checkpoint selection: matching checkpoint dir in `training/checkpoints/*-<run_id>/` (usually exists).
- Wildrobot mujoco xml:  'assets/v2/wildrobot.xml'

## Quick commands

Run the analysis script (prints best checkpoint + a changelog-ready markdown block):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python skills/wildrobot-training-analyze/scripts/analyze_offline_run.py \
  --run-dir training/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>
```

Compare two runs (prints side-by-side summary + which checkpoint to deploy):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python skills/wildrobot-training-analyze/scripts/compare_offline_runs.py \
  --run-dir-a training/wandb/offline-run-...-<run_id_a> \
  --run-dir-b training/wandb/offline-run-...-<run_id_b>
```

## Checkpoint selection rules (standing_push)

Prefer eval-based metrics over training-rollout metrics.

1. If `eval_push/success_rate` exists: maximize `(eval_push/success_rate, eval_push/episode_length)`.
2. Else if `eval/success_rate` exists: maximize `(eval/success_rate, eval/episode_length)`.
3. Else (no eval): maximize `(env/success_rate, env/episode_length)` as a fallback only.

Then select the checkpoint file:

- Prefer `training/checkpoints/*-<run_id>/checkpoint_<iter>_<step>.pkl` when it exists.
- Otherwise pick `checkpoint_<iter>_*.pkl` within the run’s checkpoint dir.

## What to optimize (standing stability)

Use the analysis output to answer:

- **Main failure mode**: which termination dominates? (`term_height_low_frac`, `term_pitch_frac`, `term_roll_frac`, …)
- **Actuation stress**: prefer torque saturation, not action saturation:
  - `tracking/max_torque`, `debug/torque_sat_frac`, `debug/torque_abs_max`
- **PPO stability**: `ppo/clip_fraction`, `ppo/approx_kl`, `ppo/lr_backoff_*`, `ppo/epochs_used`

Typical high-ROI next changes:

- If `term_height_low_frac` dominates: add/strengthen pre-collapse shaping (height margin + downward v_z gate) or tune its weights.
- If torque saturation is high: increase `reward/saturation` penalty slightly, add curriculum on pushes, or adjust action smoothing.
- If KL backoff triggers early: use LR warmup or relax backoff trigger (e.g. increase `ppo.kl_lr_backoff_multiplier`).
- Always: log separate `eval_clean/*` (no pushes) and `eval_push/*` (pushes) when debugging robustness vs base quality.

## CHANGELOG update checklist

1. Add a new entry for the new training version (v0.xx+1): config + code changes.
2. Under the current version entry, record:
   - run path, checkpoint dir, best checkpoint path
   - best eval metrics + final metrics
   - observed failure mode(s) and what to change next
3. put the latest version's Changelog on top of the doc.

The scripts print a markdown block intended to paste into `training/CHANGELOG.md`.

