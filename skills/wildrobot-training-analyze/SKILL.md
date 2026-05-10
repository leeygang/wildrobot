---
name: wildrobot-training-analyze
description: Analyze WildRobot training runs from an offline W&B run folder (training/wandb/offline-run-*/), summarize stability/tracking metrics, pick the best checkpoint for deployment, debug v0.20.x prior-guided bounded-residual walking runs against ToddlerBot, diagnose reward/pathology issues, propose v0.xx+1 config/reward/PPO changes grounded in ToddlerBot alignment and the v0.20.1 walking_training.md G4/G5 contract, and update training/CHANGELOG.md with results and next-version changes. Use when asked to review training logs, select checkpoints, compare versions, debug why a walking run is not moving / is exploiting the residual / is drifting from the prior, or plan the next training iteration.
---

# WildRobot Training Analyze

Use this workflow to (1) analyze a training run, (2) choose a deployable checkpoint, (3) identify the dominant failure mode against the v0.20.1 contract, (4) compare to ToddlerBot, and (5) write the result + next-version plan into `training/CHANGELOG.md`.

The active locomotion target is the v0.20.1 prior-guided bounded-residual PPO contract documented in `training/docs/walking_training.md`. The reference recipe we explicitly track is ToddlerBot (`projects/toddlerbot/toddlerbot/locomotion/{walk.gin,mjx_config.py,walk_env.py,mjx_env.py,train_mjx.py}`); per CLAUDE.md, divergences from ToddlerBot must have a documented rationale.

## Inputs

- Offline run dir: `training/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>/`
- For checkpoint selection: matching dir in `training/checkpoints/*-<run_id>/`
- Code: `training/`
- Configs: `training/configs/`
- MuJoCo XML: `assets/v2/wildrobot.xml`
- Source of truth for the walking contract: `training/docs/walking_training.md` (v0.20.1 §)
- ToddlerBot reference: `~/projects/toddlerbot/toddlerbot/locomotion/`

## Quick commands

Single run (prints best checkpoint + a changelog-ready markdown block, and
auto-compares to the most recent comparable prior run when one exists):

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python skills/wildrobot-training-analyze/scripts/analyze_offline_run.py \
  --run-dir training/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id>
```

Explicit previous-run override:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python skills/wildrobot-training-analyze/scripts/analyze_offline_run.py \
  --run-dir training/wandb/offline-run-YYYYMMDD_HHMMSS-<run_id> \
  --previous-run-dir training/wandb/offline-run-YYYYMMDD_HHMMSS-<previous_run_id>
```

Compare two runs:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python skills/wildrobot-training-analyze/scripts/compare_offline_runs.py \
  --run-dir-a training/wandb/offline-run-...-<run_id_a> \
  --run-dir-b training/wandb/offline-run-...-<run_id_b>
```

The analyzer prefers the v0.20.1 `Evaluate/*` namespace when present and falls through to legacy `eval_clean/` / `eval/` / `env/` order for older runs. It also emits a regression checklist against the previous comparable run on common contract metrics (`env/forward_velocity`, `env/episode_length`, `tracking/cmd_vs_achieved_forward`, stride, G5 residuals / ratio, and the main prior-tracking diagnostics). Both tools share `_first_present` (handles `0.0` correctly) and the same synthetic-success cap.

---

## v0.20.1 walking analysis (active path)

### Architecture recap (what the policy is being asked to do)

```
ZMP offline library  →  per-step window (q_ref, foot/pelvis targets, contact mask)
                              ↓
                      observation (actor sees q_ref + short preview + proprio + 15-frame history)
                              ↓
                      policy outputs Δq_policy (bounded ±0.25 rad on legs)
                              ↓
                      q_target = q_ref + Δq_policy
                              ↓
                      MuJoCo step + imitation-dominant reward
```

**Division of labor** (load-bearing):
- **Prior owns** step length, cadence, foothold geometry, contact schedule.
- **PPO owns** small joint corrections for servo lag, contact noise, sim2real mismatch, balance.

A run that violates this division is the primary failure mode (see G5 below).

### Checkpoint selection rules

The analyzer encodes these. Walking runs:

1. **If `Evaluate/mean_reward` exists** (v0.20.1-smoke2 onward): maximize `(Evaluate/mean_reward, Evaluate/mean_episode_length)`. This is the ToddlerBot pattern from `train_mjx.py:log_metrics` — no `success_rate` concept, mean reward already encodes "alive AND tracking AND posture" because the imitation block weights are right.
2. **Else `eval_clean/success_rate`** (older logs): maximize `(eval_clean/success_rate, eval_clean/episode_length)`.
3. **Else `eval/success_rate`** (older logs): same.
4. **Else `env/success_rate`** (legacy fallback only). For v0.20.x walking this is permanently 0 (truncation-based); flag the run as "no eval — selection ambiguous".

Do **not** select walking checkpoints by `env/episode_length` alone — a stationary "survive but don't walk" run will saturate it.

### v0.20.1 G4 / G5 acceptance gates (load-bearing)

These come from `walking_training.md` v0.20.1 §. Always evaluate the smoke against them explicitly.

**G4 promotion-horizon gate** (eval rollout, deterministic):

| Metric | Floor | Source |
|---|---|---|
| `env/forward_velocity` | ≥ 0.075 m/s | train rollout |
| `Evaluate/forward_velocity` | ≥ 0.075 m/s | eval rollout |
| `Evaluate/mean_episode_length` | ≥ 475 (95% of 500) | eval rollout |
| `Evaluate/cmd_vs_achieved_forward` | ≤ 0.075 m/s | eval rollout |
| `tracking/cmd_vs_achieved_forward` | ≤ 0.075 m/s | train rollout |
| `tracking/step_length_touchdown_event_m` | ≥ 0.030 m | train rollout |

**G5 anti-exploit gate** (catches "policy invents propulsion the prior should own"):

| Metric | Bound | What it catches |
|---|---|---|
| `tracking/residual_hip_pitch_left/right_abs` (mean ≈ p50) | ≤ 0.20 rad | residual stays a correction, not a replacement gait |
| `tracking/residual_knee_left/right_abs` (mean ≈ p50) | ≤ 0.20 rad | same |
| `tracking/forward_velocity_cmd_ratio` | 0.6 ≤ ratio ≤ 1.5 | both undershoot and v0.19.5 "lean-and-skate" overshoot |

A run can pass forward_velocity AND fail G5 — that's a false positive (residual-driven exploit gait). Always report G5 separately.

**G7 baseline-beat sanity** (informational):
- Open-loop bare-q_ref baseline at vx=0.15: `forward_velocity ≈ -0.09`, `episode_length ≈ 77`, `term_pitch_frac ≈ 1.0`.
- A trained policy must beat all three.

### v0.20.1 reward family — what each term measures

| Block | Term | Weight | What it pulls toward |
|---|---|---|---|
| imitation | `reward/ref_q_track` | 5.0 | joint targets match `q_ref` (α=1, exp(-α·sum((q-q_ref)²))) |
| imitation | `reward/ref_body_quat_track` | 5.0 | torso orientation matches reference (α=20) |
| imitation | `reward/torso_pos_xy` | 2.0 | torso XY tracks prior pelvis (α=90 smoke override; smoke4+) |
| imitation | `reward/ref_contact_match` | 1.0 | TB boolean count `sum(stance_mask == ref)` ∈ {0,1,2} (smoke6+; smoke3-5 used WR-specific Gaussian, gated to 0) |
| imitation | `reward/lin_vel_z` | 1.0 | TB body-local vertical velocity tracks prior bobbing (α=200; smoke6+) |
| imitation | `reward/ang_vel_xy` | 2.0 | body roll/pitch rate vs zero (α=0.5; smoke6+) |
| task | `reward/cmd_forward_velocity_track` | 5.0 | actual `vx` matches `cmd_vx` (α=200) |
| ToddlerBot shaping | `reward/feet_air_time` | 500.0 | swing-leg air time at touchdown (cmd-gated) |
| ToddlerBot shaping | `reward/feet_clearance` | 1.0 | per-foot peak swing height at touchdown |
| ToddlerBot shaping | `reward/feet_distance` | 1.0 | lateral foot spacing band [0.07, 0.13] m in torso frame |
| ToddlerBot shaping | `reward/torso_pitch_soft` | 0.5 | soft pitch band [-0.2, 0.2] rad |
| ToddlerBot shaping | `reward/torso_roll_soft` | 0.5 | soft roll band [-0.1, 0.1] rad |
| survival | `reward/alive` | 10.0 | strict ToddlerBot `-done` (0 while alive, `-10·dt` on terminal step) |
| regularizer | `reward/action_rate` | -1.0 | smoothness |
| regularizer | `reward/torque` | -0.001 | energy |
| regularizer | `reward/joint_vel` | -0.0005 | joint velocity penalty |
| regularizer | `reward/slip` | 0.05 | stance-foot horizontal slip penalty |
| diagnostic-only | `reward/pitch_rate` | 0.0 (M1 hook) | promoted only on the M1 "balance issue" branch |

**Reward integration rule** (load-bearing): reward = `sum(reward_dict.values()) * dt`. Without the `* dt` rescale the published ToddlerBot weights are NOT directly portable.

**Survival semantics**: `_reward_survival = -done` at weight 10 → 0 while alive, `-10·dt = -0.2` on the terminating step. This is NOT a dense per-step bonus — that mistake biases PPO toward any long-lived behavior.

### v0.20.1 raw error diagnostics (paired with reward terms)

Use these to diagnose dead-gradient terms (large persistent error + zero reward contribution):

| Diagnostic | Paired reward term | Healthy range at vx=0.15 |
|---|---|---|
| `ref/q_track_err_rmse` | `reward/ref_q_track` | < 0.10 rad RMSE |
| `ref/body_quat_err_deg` | `reward/ref_body_quat_track` | < 10 deg by mid-horizon |
| `ref/feet_pos_err_l2` | `reward/ref_feet_pos_track` | < 0.30 m sum-of-squares L2 (smoke1 grew to 1.02 — dead gradient) |
| `ref/contact_phase_match` | `reward/ref_contact_match` | > 0.80; if < 0.85, gate the term to weight 0 |

### Per-foot stride / swing-time diagnostics

Added in v0.20.1-smoke2. Use to debug short-stride failures (smoke1 hit step_length 0.022 m vs 0.030 m gate; per-foot data tells us if one foot is doing all the stepping):

| Metric | Reducer | Per-event mean computation |
|---|---|---|
| `tracking/touchdown_rate_left/right` | MEAN | rate = events / ctrl_step |
| `tracking/swing_air_time_left/right_event_s` | MEAN | mean swing time per event = value / touchdown_rate |
| `tracking/step_length_left/right_event_m` | MEAN | mean step length per event = value / touchdown_rate |

Symmetry check: `touchdown_rate_left ≈ touchdown_rate_right`; gross asymmetry indicates the policy is stepping with one leg only.

### Walking failure signatures (v0.19.5 / v0.20.1-aware)

Call these out explicitly:

- **standing local minimum** — `forward_velocity ≈ 0`, `episode_length` near horizon, reward driven by survival + soft posture. Action: prior is too easy to ignore; check `tracking/forward_velocity_cmd_ratio` < 0.5.
- **lean-and-skate exploit (v0.19.5 pattern)** — `forward_velocity` overshoots cmd by 1.5x+, `tracking/forward_velocity_cmd_ratio > 1.5`, residual saturates G5 bound on hip_pitch. Action: tighten residual bound, do NOT increase cmd_vx reward weight.
- **shuffle exploit (v0.20.1-smoke1 pattern)** — `forward_velocity` near cmd, `episode_length` saturated, but `tracking/step_length_touchdown_event_m < 0.025` and per-foot step lengths small on both feet. The policy is making cmd_vx via short fast steps instead of tracking the prior's foothold geometry. Action: check `ref/feet_pos_err_l2` is in the alive-gradient range; consider lowering `ref_feet_pos_alpha`.
- **gait drift** — `ref/contact_phase_match` decreases over training while `forward_velocity` increases. Action: if `ref_contact_match` weight is 0, this is expected (the contact-alignment probe gates the term); if the term is enabled, increase weight on `ref/contact_phase_match` and `ref/q_track`.
- **dead-gradient term** — large persistent `ref/<X>_err_*` paired with near-zero `reward/ref_<X>_track`. Means α is too tight for the observed error magnitude. Action: lower α to give r ≈ 0.2-0.3 at the iter-1 baseline error.
- **propulsion mis-assigned to PPO (G5 violation)** — forward_velocity meets gate but `|residual_p50|_{hip,knee}` > 0.20 rad. Action: tighten G1 residual bound by 50% and rerun; if still failing, prior is too weak — return to v0.20.0-x prior surgery (per the M1 fail-mode tree in `walking_training.md`).
- **truncation-based success_rate is meaningless** — `env/success_rate` is permanently 0 for v0.20.x walking. Never gate on it; never sort checkpoints by it.

### Comparing to ToddlerBot

Always do this comparison when reviewing a v0.20.x walking run. Source: `~/projects/toddlerbot/toddlerbot/locomotion/`.

**CRITICAL — read `walk.gin` carefully.** TB's `walk.gin` has TWO reward
recipes: a commented-out ZMP variant (`walk.gin:55-108`) and the
**active prior-free recipe** (`walk.gin:110-131`).  WR's design is
prior-guided imitation-residual (its own paradigm), so most rows
below intentionally diverge — the table cites TB's *active* recipe
so the divergences are visible, not hidden.  See
`training/docs/walking_training.md` Appendix B for the full audit
this table summarizes.

| Concern | WildRobot v0.20.1 | ToddlerBot (active) | Aligned? |
|---|---|---|---|
| Eval namespace | `Evaluate/mean_reward`, `Evaluate/mean_episode_length`, `Evaluate/<term>` | same (`train_mjx.py:385-410`) | ✅ |
| `success_rate` concept | none for walking | none anywhere | ✅ |
| Termination | height-only (`use_relaxed_termination: true`) | height-only (`mjx_env.py:1029`) | ✅ |
| Frame stack | `PROPRIO_HISTORY_FRAMES=15` | `c_frame_stack=15` (`mjx_config.py:84`) | ✅ |
| Action delay | `action_delay_steps: 1` | `n_steps_delay=1` (`mjx_config.py:105`) | ✅ |
| Action contract | `q = q_ref + clip(action) · 0.25` (legs) | `q = default_q + action · 0.25` (`mjx_config.py:103`) | ⚠️ paradigm diff (residual vs direct); both bounded ±0.25 |
| Reward integration | `sum(...) * dt` | `sum(...) * dt` (`mjx_env.py:1048`) | ✅ |
| Network hidden_sizes | (512, 256, 128) | (512, 256, 128) (`ppo_config.py:19-20`) | ✅ (after v0.20.1 alignment sweep) |
| `survival`/`alive` weight | 1.0 (after sweep), `-done` semantics | 1.0 (`walk.gin:127`), `-done` (`walk_env.py`) | ✅ (after sweep) |
| `cmd_forward_velocity_track` ↔ `lin_vel_xy` weight | 2.0 (after sweep) | 2.0 (`walk.gin:111`) | ✅ (after sweep) |
| `cmd_forward_velocity_alpha` ↔ `lin_vel_tracking_sigma` | α=200 | σ=1000 (`walk.gin:112`) | ⚠️ WR tighter; defer reform until smoke result |
| `ref_body_quat_track` ↔ `torso_quat` | 2.5 (after sweep) | 2.5 (`walk.gin:117`) | ✅ (after sweep) |
| `ref_q_track` (joint imitation) | 5.0, α=1 | 0 in active (was 5.0 in commented `walk.gin:63`) | ⚠️ WR-specific imitation block (no TB-active analog) |
| `torso_pos_xy` | 2.0, α=90 smoke | 0 in active (was 1.0 in commented) | ⚠️ WR-specific imitation |
| `lin_vel_z` | 1.0 | 0 in active | ⚠️ WR mirrors commented `walk.gin:60`; defer |
| `ang_vel_xy` (tracking) ↔ `penalty_ang_vel_xy` (penalty) | 2.0 tracking | 1.0 penalty (`walk.gin:115`) | ⚠️ form mismatch; defer |
| `feet_air_time` | 500.0 | 0 in active (replaced by `feet_phase` in active) | ⚠️ WR-specific (TB-active does not use this) |
| `feet_clearance` | 1.0 | 0 in active (replaced by `feet_phase`) | ⚠️ WR-specific |
| `feet_distance` | 1.0, [0.07, 0.13] | 0 in active; TB uses `penalty_close_feet_xy = 10.0` with `close_feet_threshold = 0.06` (`walk.gin:125-126`) | ⚠️ different formula |
| **`feet_phase` / `ref_feet_z_track`** (dense per-step swing-height tracker) | not present (Phase 2 plan) | 7.5, σ=0.0007, swing=0.04 (`walk.gin:128-131`, `walk_env.py:631-695`) | ❌ **highest-impact reward gap** — likely cause of v0.20.1-smoke1 shuffle exploit |
| **`penalty_pose`** (per-joint default anchor) | not present (Phase 2 plan) | 0.5 with weights `[0.01, 1.0, 5.0, 0.01, 5.0, 5.0, ...]` (`walk.gin:120-124`) | ❌ missing |
| **`penalty_feet_ori`** (anti-tippy-toe via gravity in foot frame) | not present (Phase 2 plan) | 5.0 (`walk.gin:129`, `walk_env.py:697-735`) | ❌ missing |
| `feet_slip` / `slip` | 0.05 | 0 in active (was 0.05 in commented) | ⚠️ WR mirrors commented; defer |
| `torso_pitch_soft` / `torso_roll_soft` | 0.5 / 0.5 | 0 in active; TB uses `penalty_ang_vel_xy = 1.0` for posture damping | ⚠️ different form |
| `action_rate` ↔ `penalty_action_rate` | -1.0 × pos-MSE | 2.0 × neg-MSE (`walk.gin:119`) | ⚠️ sign convention diff; verify magnitude equivalence |
| `ref_contact_match` | 1.0 boolean | 0 in active (was `feet_contact = 1.0` in commented) | ⚠️ WR-specific (boolean form retained from TB-historical) |
| `torque` / `joint_velocity` | -0.001 / -0.0005 | 0 in active | ⚠️ WR-specific light regularizers |
| Asymmetric critic (privileged obs) | not used (Phase 3 plan) | yes — `num_single_privileged_obs = 151` (`mjx_config.py:86`, `walk.gin:25-28`) | ❌ missing |
| Backlash DR | not present (Phase 3 plan) | activation 0.1, range [0.02, 0.1] rad (`mjx_config.py:209-210`) | ❌ missing — sim2real critical |
| kd / damping / armature / tau_max DR family | not present | TB has all (`mjx_config.py:222-235`) | ⚠️ defer |
| Multi-command sampling | `cmd_resample_steps: 150` (= 3.0 s) | `resample_time=3.0s` (`mjx_config.py:194`) | ✅ |
| Command space | 1-D linear vx ∈ [0, 0.30] | 8-D (`walk.gin:42-51`) | ⚠️ smoke restricts to 1D; v0.20.4 plan to extend |
| **Compute** (total env steps) | 1024×128×150 ≈ 2e7 | 4096×20×~12200 ≈ 1e9 (`walk.gin:2`) | ❌ **50× shorter** — smoke validates recipe; M3 long run needed for terminal-perf parity |
| **PPO learning_rate** | 3e-4 | 3e-5 (`ppo_config.py:39`) | ⚠️ WR 10× higher (compensating for compute gap) |
| **PPO entropy_coef** | 0.01 | 5e-4 (`ppo_config.py:40`) | ⚠️ WR 20× higher |
| `gamma` / `discounting` | 0.99 | 0.97 (`ppo_config.py:34`) | ⚠️ WR longer effective horizon (matches longer episodes) |
| G4 / G5 / G7 gates | WildRobot-specific | n/a | ⚠️ acceptance criteria are WildRobot-specific |

Symbol meaning:
- ✅ aligned to TB-active.
- ⚠️ intentional divergence with documented rationale; confirm rationale is current.
- ❌ missing TB-active term that closes a known WR failure mode (or compute gap that blocks parity); flag as a parity risk.

When a run violates the contract on a ✅ row, surface it as a bug.
When it diverges on a ⚠️ row, confirm the rationale is current; if not, flag as drift.
When it lands on a ❌ row, that's a known parity gap — link to `walking_training.md` Appendix B.2 for the phased fix plan.

### Walking interpretation rules

- If `Evaluate/forward_velocity` is near zero AND `Evaluate/mean_episode_length` is near horizon → **standing local minimum** — call it out, don't say "still training".
- If `Evaluate/forward_velocity` meets gate AND `tracking/forward_velocity_cmd_ratio` is in band AND G5 residual mean ≤ 0.20 → **locomotion emerging**. Recommend the iter that maximizes `Evaluate/mean_reward`.
- If `Evaluate/forward_velocity` meets gate AND G5 fails → **action-authority leak / propulsion mis-assigned**. Per M1 fail-mode tree: tighten G1 residual bound 50%, do NOT relax G5.
- If `tracking/cmd_vs_achieved_forward` mean is high (> 0.075) AND `Evaluate/mean_episode_length` is near horizon → **balance achieved, tracking weak**. Check whether `cmd_forward_velocity_alpha` is the published 200 (an earlier α=4 made the gradient flat).
- If `ref/feet_pos_err_l2` grows over training while `reward/ref_feet_pos_track` stays at 0 → **dead-gradient term**. Lower the alpha (smoke uses 30, not 200).
- If `term_pitch_frac > 0.5` AND `forward_velocity` meets gate AND G5 passes → **balance issue (M1 branch)**. Increase regularizer weight on `pitch_rate` (M1 hook is pre-wired at weight 0); consider widening residual bound on ankle pitch only.

### Walking next-step template

```text
v0.20.1 verdict:
- locomotion emerging / standing local minimum / lean-and-skate exploit / shuffle exploit / dead-gradient on <term> / gait drift / balance issue
- G4: forward_velocity <pass/fail> | mean_episode_length <pass/fail> | cmd_vs_achieved <pass/fail> | step_length <pass/fail>
- G5: residual_p50 <pass/fail> | velocity_cmd_ratio <pass/fail>
- G7 baseline-beat: <pass/fail>
- ToddlerBot alignment: <ok / drift on <row(s)>>
```

Then recommend one of (per the M1 fail-mode tree):
- continue training to the M2 compute cap unchanged
- prior surgery (return to v0.20.0-x; do NOT relax G5)
- tighten G1 residual bound 50% and rerun
- promote the M1 `pitch_rate` regularizer
- regenerate the offline reference library

### Walking design guidance (when planning the next config)

Order of operations:
1. **Confirm the pre-smoke probes pass.** Zero-action q_ref replay (G6 invariant), contact-alignment probe (gates `ref_contact_match` weight), prior-vs-body sanity (`tools/v0200c_per_frame_probe.py`).
2. **Match ToddlerBot weights.** Use Appendix A of `walking_training.md` as the alignment audit. Divergences need explicit rationale.
3. **Don't invent new reward terms** when standard locomotion terms cover the problem. The current reward family is already imitation + cmd + ToddlerBot shaping + survival + small regularizers.
4. **Don't relax G5** to fix a forward-velocity miss. G5 is the anti-exploit gate; relaxing it brings back v0.19.5.
5. **PPO hyperparameters last.** Learning rate / entropy / KL backoff are the last knobs to touch.

Forbidden moves (will reintroduce known failure modes):
- adding a flat positive `alive` bonus (was the v0.19.5b lean-back exploit enabler)
- raising the leg residual bound past ±0.50 rad
- removing the `* dt` reward integration
- ranking checkpoints by `env/success_rate` or `env/episode_length` alone

---

## CHANGELOG update checklist

1. Add a new entry for the next training version (v0.xx+1): config + code changes.
2. Under the current version entry, record:
   - run path, checkpoint dir, best checkpoint path
   - G4 row-by-row pass/fail, G5 residual + velocity_cmd_ratio, G7 baseline-beat, ToddlerBot alignment status
   - Best `Evaluate/mean_reward`, `Evaluate/mean_episode_length`, `Evaluate/forward_velocity`, `Evaluate/cmd_vs_achieved_forward` at the chosen checkpoint
   - Concerning trends not gated (e.g. `ref/feet_pos_err_l2` growth, `ref/contact_phase_match` drift)
   - Observed failure mode(s) per the v0.19.5 / v0.20.1 signature list and the recommended next intervention per the M1 fail-mode tree
3. Newest version entry on top.

The scripts print a markdown block intended to paste into `training/CHANGELOG.md`.

Standing / M3 / FSM analysis material (FSM verdicts, swing occupancy, step-event / foot-place / posture rewards, push-recovery failure signatures) is no longer carried in this skill. The active locomotion target is v0.20.1 walking; runs from older standing / M3 branches should be analyzed against their own historical SKILL revisions.
