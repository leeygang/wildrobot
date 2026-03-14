---
name: wildrobot-training-analyze
description: Analyze WildRobot training runs from an offline W&B run folder (training/wandb/offline-run-*/), summarize stability/success metrics, pick the best checkpoint for deployment, debug standing-push and walking runs, diagnose reward/pathology issues, propose v0.xx+1 config/reward/PPO changes using established locomotion best practices, and update training/CHANGELOG.md with results and next-version changes. Use when asked to review training logs, select checkpoints, compare versions, debug why a walking run is not moving or is unstable, or plan the next training iteration.
---

# WildRobot Training Analyze

Use this workflow to (1) analyze a training run, (2) choose a deployable checkpoint, (3) identify the dominant failure mode, and (4) write the result + next-version plan into `training/CHANGELOG.md`.

For M3/FSM standing-push runs, also determine whether the stepping controller is actually doing useful work, not merely coexisting with a surviving residual policy.

For walking runs, determine whether the policy is actually learning locomotion or exploiting stationary/posture rewards, and recommend reward / PPO / curriculum changes grounded in standard legged-locomotion practice rather than ad hoc trial-and-error.

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

## Checkpoint selection rules

### standing_push

Prefer eval-based metrics over training-rollout metrics.

1. If `eval_push/success_rate` exists: maximize `(eval_push/success_rate, eval_push/episode_length)`.
2. Else if `eval/success_rate` exists: maximize `(eval/success_rate, eval/episode_length)`.
3. Else (no eval): maximize `(env/success_rate, env/episode_length)` as a fallback only.

Then select the checkpoint file:

- Prefer `training/checkpoints/*-<run_id>/checkpoint_<iter>_<step>.pkl` when it exists.
- Otherwise pick `checkpoint_<iter>_*.pkl` within the run’s checkpoint dir.

### walking

Prefer eval-based locomotion metrics over training-rollout metrics.

1. If `eval/success_rate` and velocity metrics exist: maximize
   `(eval/success_rate, -abs(eval/velocity_error), eval/episode_length)`.
2. Else if `env/success_rate` and velocity metrics exist: maximize
   `(env/success_rate, -abs(env/velocity_error), env/episode_length)`.
3. If there is no reliable velocity metric, do **not** select by reward alone; first call out that checkpoint selection is ambiguous.

For walking, a high-reward checkpoint with near-zero forward velocity is usually **not** deployable, even if survival is excellent.

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

## Walking analysis extension

Use this section whenever the run's intent is gait learning, for example when config/version/changelog mentions:
- `ppo_walking`
- `walking`
- non-zero velocity commands
- Stage 1 / PPO-only walking

### What to optimize (walking)

Judge the run on locomotion first, not just reward:
- **Velocity tracking**: `env/forward_velocity`, `env/velocity_cmd`, `env/velocity_error`
- **Survival / stability**: `env/episode_length`, `env/success_rate`, `term_height_low_frac`, `term_pitch_frac`, `term_roll_frac`
- **Actuation stress**: `tracking/max_torque`, `debug/torque_sat_frac`, `debug/action_sat_frac`
- **Gait structure**: `reward/gait_periodicity`, `reward/clearance`, `reward/hip_swing`, `reward/knee_swing`, `reward/slip`
- **PPO stability**: `ppo/clip_fraction`, `ppo/approx_kl`, `ppo/lr_backoff_*`, `ppo/epochs_used`

### Walking success criteria

Call a walking run promising only when most of these are true:
- forward velocity is materially above zero and moving toward the commanded range
- velocity error is trending down
- episode length stays high enough that motion is sustainable
- torque stress is not exploding
- gait metrics suggest alternating support rather than stationary rocking or hopping

### Walking failure signatures

Call these out explicitly if seen:
- **standing local minimum**:
  - `forward_velocity ~= 0`
  - high survival
  - high reward driven by posture / healthy terms
- **posture exploit**:
  - forward velocity stays near zero
  - pitch failures and torque rise
  - reward rises anyway
- **hopping exploit**:
  - some speed appears
  - `flight_phase_penalty` / contact metrics indicate both feet frequently unloaded
- **shuffle exploit**:
  - some speed appears
  - low clearance / swing rewards
  - high torque / saturation
- **reward incoherence**:
  - reward improves while velocity error does not
  - checkpoint selection by reward conflicts with locomotion quality

### Walking interpretation rules

- If velocity is near zero, do **not** say "still training" by default. First check whether the run is actually trapped in standing.
- If a standing-resume run stays near zero velocity after roughly `80-120` iterations, strongly consider that the warm start is counterproductive.
- If reward goes up while velocity error worsens, prioritize diagnosing reward exploitation over tuning PPO.
- If `eval_push/*` is perfect but pushes are disabled in config, say explicitly that this eval is not informative.

### Walking best-practice interventions

Prefer these interventions before inventing new reward terms:

1. **Task reward must dominate stationary survival**
   - velocity tracking should create a large reward gap between standing still and matching the command
   - standing-still penalty must matter when command > 0
2. **Avoid large flat positive alive bonuses**
   - especially if they can outweigh failed velocity tracking
3. **Reduce action lag when gait is sluggish**
   - lower action filtering if policy is overly damped
4. **Use command curricula**
   - start with a conservative forward range like `0.1-0.6 m/s`
   - only widen once gait exists
5. **Do not overconstrain early gait emergence**
   - too much upright / slip / smoothness penalty can trap standing or produce shuffling
6. **Be careful with standing warm starts**
   - if they preserve a standing local minimum, switch to fresh-run walking
7. **Use PPO tuning conservatively**
   - increase LR / entropy to escape local minima
   - disable rollback during transition runs if rollback preserves the bad solution
   - but keep policy-contract and rollout-geometry resume guards hard

### Walking next-step template

When summarizing a walking run, use wording like:

```text
Walking verdict:
- locomotion emerging / trapped in standing / posture exploit / hopping exploit / shuffle exploit
- velocity tracking: improving / flat / regressing
- stability: good / acceptable / poor
```

Then recommend one of:
- continue training unchanged
- continue training with PPO retuning
- retune reward weights using standard locomotion priorities
- restart from scratch (no resume)
- add curriculum

### Walking design guidance

When asked to design the next walking config, prefer this order:
1. make the task objective unambiguous
2. remove or reduce rewards that pay for standing still
3. check whether warm start is hurting more than helping
4. only then tune PPO hyperparameters

Do **not** default to inventing new custom reward terms if standard locomotion terms already cover the problem:
- velocity tracking
- uprightness / termination limits
- torque / saturation penalties
- action smoothness
- slip
- clearance
- gait periodicity

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
   - for walking runs: locomotion verdict, velocity tracking status, and whether reward appears to be exploited
3. put the latest version's Changelog on top of the doc.

The scripts print a markdown block intended to paste into `training/CHANGELOG.md`.
