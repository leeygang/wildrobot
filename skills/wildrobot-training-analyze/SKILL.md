---
name: wildrobot-training-analyze
description: Analyze WildRobot training runs from an offline W&B run folder (training/wandb/offline-run-*/), summarize stability/success metrics, pick the best checkpoint for deployment (standing_push prioritizes eval_push success rate then episode length), analyze M3/FSM stepping-controller runs against explicit mechanism/outcome criteria, propose v0.xx+1 config/reward/code changes, and update training/CHANGELOG.md with results and next-version changes. Use when asked to review training logs, select checkpoints, compare versions (e.g. v0.13.6 vs v0.13.7), or plan the next training iteration (v0.xx+1).
---

# WildRobot Training Analyze

Use this workflow to (1) analyze a training run, (2) choose a deployable checkpoint, (3) identify the dominant failure mode, and (4) write the result + next-version plan into `training/CHANGELOG.md`.

For M3/FSM standing-push runs, also determine whether the stepping controller is actually doing useful work, not merely coexisting with a surviving residual policy.

## Inputs (what you need)

- Offline run dir: `training/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>/`
- For checkpoint selection: matching checkpoint dir in `training/checkpoints/*-<run_id>/` (usually exists).
- Code is in 'training'
- training Config is in 'training/configs'
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

## M3 / FSM analysis extension

Use this section whenever the run enables the foot-placement state machine, for example when config/version/changelog mentions:
- `env.fsm_enabled: true`
- `M3`
- `FSM`
- `foot-placement`

### M3 expectations

The first question is not only "did reward go up?" but:
- did the robot actually enter swing under pushes?
- did stepping improve support in the disturbance direction?
- did the robot recover back toward upright after touchdown?

For M3, evaluate both:
- **Mechanism**: is the FSM/controller being used correctly?
- **Outcome**: does that usage improve push recovery and posture return?

### M3 mechanism signals to review

Always inspect these when present:
- `debug/bc_phase`
- `debug/bc_phase_ticks`
- `debug/bc_swing_foot`
- `debug/need_step`
- `reward/step_event`
- `reward/foot_place`
- `debug/touchdown_left`
- `debug/touchdown_right`

Interpretation:
- `debug/bc_phase` should show meaningful occupancy in `SWING`, not only `STANCE`.
- `reward/step_event` should be measurably above zero in disturbed evaluation.
- `reward/foot_place` should rise alongside step events; low/flat values suggest random or poor touchdown placement.
- `debug/bc_phase_ticks` should not be dominated by long swing phases that imply repeated timeout-driven steps.
- touchdown flags should occur when `debug/need_step` is elevated, not mostly during quiet standing.

### M3 outcome signals to review

Keep the existing standing-push metrics, but interpret them through the controller behavior:
- `eval_push/success_rate`
- `eval_push/episode_length`
- `eval_push/survival_rate`
- `eval_push/term_height_low_frac`
- `eval_push/term_pitch_frac`
- `eval_push/term_roll_frac`
- `reward/posture`
- `tracking/max_torque`
- `debug/torque_sat_frac`

Desired pattern for a good M3 run:
- `eval_push/success_rate` matches or exceeds the M2 baseline
- `reward/step_event` and `reward/foot_place` rise under pushes
- terminations are reduced, especially immediate collapse after disturbance
- `reward/posture` recovers after the push instead of staying flat/low
- torque saturation does not spike catastrophically compared with M2

### M3 success criteria

Call an M3 run promising when most of the following are true:
- `SWING` is clearly used during push recovery
- step events are touchdown-driven, not mostly timeout-driven
- `foot_place` is non-trivial and stable
- hard-push survival improves or at least matches M2 with visibly clearer stepping
- the robot returns toward neutral/upright after the disturbance

### M3 failure signatures

Call these out explicitly if seen:
- `SWING` rarely occurs even under hard pushes
- swing phases are mostly timeout-driven
- the policy survives mainly through residual bracing while FSM metrics stay weak
- steps occur but the robot does not regain upright posture afterward
- waist/arm compensation appears active but foot placement remains ineffective
- success rate is flat while torque stress rises materially

### Recommended M3 analysis wording

When summarizing an M3 run, include a short controller verdict:

```text
FSM verdict:
- engaged / weakly engaged / not meaningfully engaged
- touchdown-driven / timeout-driven / mixed
- upright recovery: good / partial / poor
```

And then state the next action clearly:
- keep training and tune reward/curriculum
- reduce residual authority further
- tune swing timing / touchdown thresholds
- tune posture recovery
- disable or weaken waist/arm damping if it masks stepping

## CHANGELOG update checklist

1. Add a new entry for the new training version (v0.xx+1): config + code changes.
2. Under the current version entry, record:
   - run path, checkpoint dir, best checkpoint path
   - best eval metrics + final metrics
   - observed failure mode(s) and what to change next
   - for M3/FSM runs: FSM verdict, step-event/foot-place behavior, and whether upright recovery is achieved
3. put the latest version's Changelog on top of the doc.

The scripts print a markdown block intended to paste into `training/CHANGELOG.md`.
