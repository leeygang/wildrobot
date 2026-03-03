# PPO Training Stability Improvements (Standing Push)

This doc tracks why PPO training metrics can regress late in a run (even when intermediate checkpoints are good), and the highest-ROI changes to make progress more stable and reproducible.

Scope: Stage 1 PPO-only standing / push robustness (no AMP).

## Problem Statement

It is common for PPO runs to show:
- Good intermediate checkpoints (high `success_rate`, long `episode_length`)
- Worse final checkpoints (late-training regression)

Saving checkpoints every N iterations helps *selection* (you can pick a good checkpoint), but it does not prevent the optimization dynamics from drifting away from a good basin.

## Key Metrics / Signals

Use these for diagnosis and stopping rules:
- Topline: `env/success_rate`, `env/episode_length`
- Stability: `ppo/approx_kl`, `ppo/clip_fraction`
- Actuation stress: `env/max_torque`, `debug/action_sat_frac`
- Failure mode: `debug/*_term/count` (e.g. pitch/roll terminations)

Interpretation rules of thumb:
- High/volatile `ppo/approx_kl` and high `ppo/clip_fraction` often indicate “too-large” policy updates.
- A drop in topline metrics with rising `env/max_torque` often indicates a more aggressive recovery strategy that sometimes fails (tail risk).

## Addressing “Checkpoint Every 10 Iterations Isn’t Enough”

Checkpointing frequency solves *not losing* a good policy, but regression still causes:
- wasted compute after the peak,
- confusion about “did the config work?”,
- and low confidence in continuous improvements.

Highest-ROI fixes:
1. Add a deterministic **evaluation loop** and pick “best checkpoint by eval”, not by noisy training rollouts.
2. Add **KL guardrails** and **learning-rate schedules** so the policy cannot jump away from a good basin late.
3. Add an automated “**rollback on regression**” behavior: if eval score drops badly, resume from best checkpoint with smaller step size.

## Early Stop: How To Reach “Ideal” Results If We Stop Early

Early stopping is not “give up”; it is “end a phase when the current hyperparameters have extracted what they can without damage”.

Use an iterative process:
1. Train until plateau/regression using stable settings (KL guardrails + schedules).
2. Select best checkpoint by eval.
3. Change *one* lever (curriculum difficulty, reward shaping, or PPO step size) and resume from best.
4. Repeat until the target success/ep-length is reached.

This is more reliable than a single long run with a fixed step size.

## Priority By ROI (Recommended Order)

P0 (highest ROI, lowest risk)
- Add **eval metrics** (deterministic seeds + fixed push set) and use them for checkpoint selection.
- Add **KL guardrails** (`target_kl`) and stop epochs when KL is too high.
- Add **learning rate decay** and (optional) **entropy decay**.

P1
- Reduce update aggressiveness further by tuning: `learning_rate`, `epochs`, `num_minibatches`, `max_grad_norm`.
- Add “rollback on regression”: if eval drops, restore best weights and reduce LR.

P2
- Add **disturbance curriculum** (force/duration/body set ramp) to reduce variance early and increase difficulty gradually.

P3 (higher effort / more invasive)
- Improve reset distribution (crouch presets / targeted joint noise), or restructure reward terms for stability.

## Target KL Explained (What / Why / How To Choose)

PPO uses probability ratios between new and old policy:
`r = pi_new(a|s) / pi_old(a|s)`.

`ppo/approx_kl` is a diagnostic for how far the policy moved in distribution space.

`target_kl` is a guardrail that enforces a “trust region-like” update size:
- If `approx_kl` exceeds `target_kl` (or a multiple of it), stop further epochs for this PPO update.
- Optionally reduce LR when KL spikes.

Suggested starting values (standing / pushes):
- `target_kl`: 0.02 (conservative) to 0.05 (looser)
- Stop-epochs threshold: `approx_kl > 1.5 * target_kl`
- LR backoff threshold: `approx_kl > 2.0 * target_kl` then `lr *= 0.5`

Notes:
- Smaller `clip_epsilon` can increase reported `clip_fraction`; KL guardrails are usually a more direct stability control knob.

## Reduce Update Aggressiveness (Concrete Knobs)

These reduce destructive late updates:
- `learning_rate`: reduce (e.g. `3e-4 -> 1e-4`)
- `epochs`: reduce (e.g. `4 -> 2`)
- `entropy_coef`: reduce late or decay schedule (e.g. `0.01 -> 0.002`)
- `max_grad_norm`: ensure it is set (already is)
- Optional: increase batch size by increasing `num_envs` or `rollout_steps` if compute allows

## Lower Variance Of The Learning Signal

Low effort:
- Deterministic eval for selection (separate from training rollouts).
- Advantage normalization (confirm it is enabled in PPO implementation).
- Keep push schedule sampling “episode-constant” (already true) but consider eval with fixed push set.

Higher effort:
- Normalize rewards consistently / keep reward magnitudes stable across curriculum stages.

## Disturbance Curriculum (Standing Push)

Recommended ramp strategy:
1. Start torso-only bodies: `[waist, upperbody]`.
2. Start weaker force and shorter duration.
3. Gradually increase one axis at a time:
   - `push_force_max` ramp
   - `push_duration_steps` ramp
   - `push_bodies` set expansion (add shoulders, then forearms)

Example (iteration-based, illustrative):
- Iter 0-50: `force_max=4`, `duration=6`, bodies `[waist, upperbody]`
- Iter 50-120: `force_max=5.5`, `duration=8`, add shoulders
- Iter 120+: `force_max=7`, `duration=10`, add arms

## Separation Of Training vs Eval Metrics

Training metrics:
- Noisy, on-policy, affected by exploration and randomization.

Eval metrics:
- Fixed seeds + fixed disturbance set.
- Deterministic policy mode.
- Used for:
  - best-checkpoint selection,
  - early stopping / rollback,
  - comparing configs across runs.

Minimum viable eval:
- N=20-50 episodes (fast) every 10 iterations.
- Fixed push schedules: a small list of angles/forces/durations and body IDs.

## Checklist

Done:
- [x] Add deterministic eval suite
- [x] Add target_kl guardrails
- [x] Add LR / entropy schedules
- [x] Add rollback-on-regression
- [ ] Add push curriculum schedule

Implemented in config + loop:
- `ppo.eval.{enabled,interval,num_envs,num_steps,deterministic}`
- `ppo.target_kl`, `ppo.kl_early_stop_multiplier`, `ppo.kl_lr_backoff_multiplier`, `ppo.kl_lr_backoff_factor`
- `ppo.lr_schedule_end_factor`, `ppo.entropy_schedule_end_factor`
- `ppo.rollback.{enabled,patience,success_rate_drop_threshold,lr_factor}`

## Notes / Links

- Standing push config: `training/configs/ppo_standing_push.yaml`
- Disturbance implementation: `training/envs/disturbance.py`
