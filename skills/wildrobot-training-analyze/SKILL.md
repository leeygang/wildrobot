---
name: wildrobot-training-analyze
description: Analyze WildRobot training runs from an offline W&B run folder (training/wandb/offline-run-*/), summarize stability/tracking metrics, pick the best checkpoint for deployment, debug v0.20.x prior-guided bounded-residual walking runs against ToddlerBot, diagnose reward/pathology issues, propose v0.xx+1 config/reward/PPO changes grounded in ToddlerBot alignment and the walking_training.md walking-quality/anti-exploit acceptance contract, and update training/CHANGELOG.md with results and next-version changes. Use when asked to review training logs, select checkpoints, compare versions, debug why a walking run is not moving / is exploiting the residual / is drifting from the prior, or plan the next training iteration.
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

The analyzer prefers the v0.20.1 `Evaluate/*` namespace when present and falls through to legacy `eval_clean/` / `eval/` / `env/` order for older runs. It also emits a regression checklist against the previous comparable run on common contract metrics (`env/forward_velocity`, `env/episode_length`, `tracking/cmd_vs_achieved_forward`, stride, anti-exploit residuals / ratio, and the main prior-tracking diagnostics). Both tools share `_first_present` (handles `0.0` correctly) and the same synthetic-success cap.

---

## Pre-flight verification — REQUIRED before any recommendation

Two failure modes have repeatedly produced wrong analysis in past sessions. Both are caught by mandatory pre-flight checks. **Do not skip these even when the data "obviously" points to a conclusion.**

### Mandatory checks before proposing ANY change

| # | Check | How |
|---|---|---|
| 1 | **Verify any reward term you plan to recommend "adding" doesn't already exist.** | `grep -n "<term_name>" training/envs/wildrobot_env.py` AND `grep -n "<term_name>:" training/configs/<active_yaml>` AND check `metrics.jsonl` for `reward/<term_name>` |
| 2 | **Verify the cmd distribution the env actually emits matches the YAML's stated range.** | Compare `env/velocity_cmd` late-iter mean to `0.5 * (min_velocity + max_velocity)` from the YAML. If mean is significantly lower (>20%), suspect a sampler projection bug. |
| 3 | **Read the env's sampler code, not just the YAML field names.** | `grep -n "_sample_walk_command\|_sample_velocity_cmd" training/envs/wildrobot_env.py` and trace what `sin_theta` / `cos_theta` / `* sign_x` factors do to the magnitudes. |
| 4 | **Pair every flagged reward term against its metric.** | If `reward/<X>` is logged with a nonzero value (even small), the term is **wired and active**, regardless of what any reference table claims. |

### Why these exist

- **smoke2 (yxfkg6wh) miss #1**: I recommended "add `feet_phase` reward" — but WR has had `_feet_phase_reward` since smoke9, smoke2 enabled it at weight 7.5, and `reward/feet_phase = 0.0028` was logged in the run. The premise was wrong because I trusted the TB-comparison table below instead of grepping the env code.
- **smoke2 miss #2**: I missed that `env/velocity_cmd = 0.083` (vs YAML midpoint 0.22) indicated the TB sin(θ) projection was collapsing the commanded vx distribution toward zero on 50%+ of walk-branch episodes — the real root cause of the posture-exploit verdict. The single metric that would have surfaced this was logged the entire time.

### The TB-comparison table below is a SNAPSHOT, not ground truth

The "WildRobot v0.20.1" column reflects state at some point in the past; rows marked **❌ "not present (Phase 2/3 plan)"** may have already landed. Before citing any row as "missing", run the check #1 above. Source-of-truth is `wildrobot_env.py:_compute_reward_terms` + the active YAML's `reward_weights` block + `metrics.jsonl`. The table is an *audit checklist* of which TB-active terms WR has historically lacked — it is not a live mirror.

If you find a stale row while analyzing, **fix the table in this skill before writing your recommendation** so the next session doesn't repeat the error.

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

A run that violates this division is the primary failure mode (see the anti-exploit gate below).

### Checkpoint selection rules

The analyzer encodes these. Walking runs:

1. **If `Evaluate/mean_reward` exists** (v0.20.1-smoke2 onward): maximize `(Evaluate/mean_reward, Evaluate/mean_episode_length)`. This is the ToddlerBot pattern from `train_mjx.py:log_metrics` — no `success_rate` concept, mean reward already encodes "alive AND tracking AND posture" because the imitation block weights are right.
2. **Else `eval_clean/success_rate`** (older logs): maximize `(eval_clean/success_rate, eval_clean/episode_length)`.
3. **Else `eval/success_rate`** (older logs): same.
4. **Else `env/success_rate`** (legacy fallback only). For v0.20.x walking this is permanently 0 (truncation-based); flag the run as "no eval — selection ambiguous".

Do **not** select walking checkpoints by `env/episode_length` alone — a stationary "survive but don't walk" run will saturate it.

### Acceptance gates — walking-quality + anti-exploit + baseline-beat (load-bearing)

These come from `walking_training.md`. Always evaluate the run against them explicitly. (The old `G4/G5/G7` labels were retired 2026-05-31 for behavior descriptions; some older CHANGELOG entries still use them.)

**⚠️ Judge by the DETERMINISTIC eval, not the stochastic train rollout.** When action noise is high (e.g. smoke5: `log_std → −0.46`, std 0.63 ≈ 2× the action signal `abs.mean 0.37`), the *stochastic* train rollout flails/sways while the *deterministic mean* walks — a train→eval gap. smoke5 was first mis-verdicted "sway" from train metrics (fv 0.038, step 0.0015); the deterministic from-rest eval showed a clean walk (fv 0.130, ratio 0.998, step 0.0316, 0 falls). **Run `eval_policy.py` (deterministic, from rest) BEFORE issuing a verdict**, and cross-check the action `log_std`/`std` vs `action abs.mean` to detect the gap.

**Walking-quality** (eval rollout, deterministic) — task-level, paradigm-independent:

| Metric | Floor | Source |
|---|---|---|
| `env/forward_velocity` | ≥ 0.075 m/s | train rollout |
| `Evaluate/forward_velocity` | ≥ 0.075 m/s | eval rollout |
| `Evaluate/mean_episode_length` | ≥ 475 (95% of 500) | eval rollout |
| `Evaluate/cmd_vs_achieved_forward` | ≤ 0.075 m/s | eval rollout |
| `tracking/cmd_vs_achieved_forward` | ≤ 0.075 m/s | train rollout |
| `tracking/step_length_touchdown_event_m` | ≥ 0.030 m | train rollout |

**Anti-exploit** (catches "propulsion mis-assigned / skating") — **paradigm-dependent**:

| Metric | Bound | Applies to | What it catches |
|---|---|---|---|
| `tracking/forward_velocity_cmd_ratio` | 0.6 ≤ ratio ≤ 1.5 | **all** | undershoot AND lean-and-skate overshoot |
| `tracking/max_torque`, `debug/torque_sat_frac` | not saturating | **all** | servo over-demand / sim2real risk (esp. wide action scales) |
| `tracking/step_length` + `reward/feet_phase` | real stepping, not shuffle | **all** | stepping-not-skating |
| `tracking/residual_{hip,knee}_{left,right}_abs` (mean ≈ p50) | ≤ 0.20 rad | **prior-guided only** (`loc_ref_residual_base: q_ref`) | residual stays a correction, not a replacement gait |

**The residual-magnitude bound (≤0.20) is RETIRED for the home/TB-contract paradigm** (`loc_ref_residual_base: home`, v0.21.0 smoke6+): there the residual *is* the gait (large by design), so a large residual is expected, not an exploit. Under the home base, judge anti-exploit by velocity-ratio + torque-saturation + stepping-not-skating only. **Check the active config's `loc_ref_residual_base` before applying the residual bound.**

A run can pass walking-quality AND fail anti-exploit — that's a false positive (e.g. a skating / residual-driven gait). Always report anti-exploit separately.

**Baseline-beat sanity** (informational):
- Open-loop bare-`q_ref` baseline at vx=0.15: `forward_velocity ≈ -0.09`, `episode_length ≈ 77`, `term_pitch_frac ≈ 1.0`.
- A trained policy must beat all three. (Under the home base the zero-action baseline is "home stand," not the prior — so this is a prior sanity ref, not the policy's own baseline.)

### v0.20.1 reward family — what each term measures

| Block | Term | Weight | What it pulls toward |
|---|---|---|---|
| imitation | `reward/ref_q_track` | 5.0 | joint targets match `q_ref` (α=1, exp(-α·sum((q-q_ref)²))) |
| imitation | `reward/ref_body_quat_track` | 5.0 | torso orientation matches reference (α=20) |
| imitation | `reward/torso_pos_xy` | 2.0 | torso XY tracks prior pelvis (α=90 smoke override; smoke4+) |
| imitation | `reward/ref_contact_match` | 1.0 | TB boolean count `sum(stance_mask == ref)` ∈ {0,1,2} (smoke6+; smoke3-5 used WR-specific Gaussian, gated to 0) |
| imitation | `reward/lin_vel_z` | 1.0 | TB body-local vertical velocity tracks prior bobbing (α=200; smoke6+) |
| imitation | `reward/ang_vel_xy` | 2.0 | body roll/pitch rate vs zero (α=0.5; smoke6+) |
| task | `reward/cmd_forward_velocity_track` | 5.0 (doc) / 2.0 (smoke2/3 active) | actual `vx` matches `cmd_vx` via `exp(-α·err²)`; α=562.5 active → live-gradient window `cmd ± 1/√α ≈ ±0.042 m/s` (unreachable from a standing start if the sampled cmd band sits entirely above it — see "dead velocity gradient" signature) |
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

### Key metrics MUST surface before any verdict

The analyzer prints these, but a verdict that doesn't reference each one is incomplete and likely missing a root cause. **Every smoke analysis must explicitly state the value AND interpret it.** Empty / "looks fine" is not interpretation.

| Metric | What it gates | Interpretation rule |
|---|---|---|
| `env/velocity_cmd` (late-iter mean) | sampler-vs-YAML alignment | Compare to `0.5 × (min_velocity + max_velocity)`. Δ > 20% ⇒ sampler-cmd-projection regression (see failure signatures). Even at Δ < 20% (sampler healthy), if the sampled band sits entirely ABOVE the α live-window (`cmd ± 1/√cmd_forward_velocity_alpha`, ≈ ±0.042 at α=562.5) with `reward/cmd_forward_velocity_track ≈ 0` ⇒ dead-velocity-gradient signature. Required even when verdict looks reward-related. |
| `Evaluate/forward_velocity` (or `env/forward_velocity`) | walking-quality | sign + magnitude. Negative = backward drift; near-zero with high cmd = standing minimum or sampler bug. |
| `Evaluate/mean_episode_length` | walking-quality | Near-horizon AND fv ≈ 0 ⇒ posture exploit, NOT "still training". |
| `Evaluate/cmd_vs_achieved_forward` | walking-quality | > 0.075 with episode_length near horizon ⇒ "balance achieved, tracking weak" OR sampler bug. |
| `tracking/step_length_touchdown_event_m` | walking-quality | < 0.030 ⇒ shuffle / no-stride. Also check left/right event metrics for asymmetry. |
| `tracking/forward_velocity_cmd_ratio` | anti-exploit (all) | Out of [0.6, 1.5] ⇒ either lean-and-skate (>1.5) or undershoot (<0.6). Train-rollout ratio looking "fine" with mean cmd ≈ 0 is misleading — cross-check vs the post-training deterministic eval ratio. |
| `tracking/max_torque`, `debug/torque_sat_frac` | anti-exploit (all) | Saturating ⇒ servo over-demand / sim2real risk; the primary anti-exploit signal under wide action scales (home/TB-contract). |
| `tracking/residual_{hip,knee}_{left,right}_abs` | anti-exploit (**prior-guided only**) | > 0.20 rad ⇒ propulsion mis-assigned. RETIRED for `loc_ref_residual_base: home` (residual IS the gait there). Per-leg asymmetry indicates one-leg gait. |
| `tracking/touchdown_rate_{left,right}_count` | gait symmetry | Ratio > 2× ⇒ one-leg stepping; combine with per-foot step_length to confirm. |
| `ref/feet_pos_err_l2`, `ref/q_track_err_rmse`, `ref/contact_phase_match`, `ref/body_quat_err_deg` | dead-gradient diagnostic | Pair each against `reward/ref_*` — large error + ~0 reward = dead-gradient term (lower α). |
| `reward/feet_phase`, `reward/penalty_feet_ori`, `reward/penalty_pose`, `reward/penalty_close_feet_xy`, `reward/ref_feet_z_track` | reward family wiring proof | If logged, the term is active — proves the env has it. Use to refute "X is missing" claims before writing them. |
| `term_height_low_frac`, `term_pitch_frac`, `term_roll_frac` | failure-cause breakdown | Aggregate across iters; identifies whether the policy dies from falling, tipping forward, or tipping sideways. |
| Post-training eval `selected_checkpoint_path` | promotion gate | If `null`, NO candidate passed the walking-quality gate — train-side `Evaluate/mean_reward` ranking is NON-AUTHORITATIVE; do not call any checkpoint "best for deploy". |
| Post-training eval `top_k_candidates[*].lateral_yaw_probes` | Appendix C (v0.21.0+) | `signed_ratio ≥ 0.5` per probe; report-only in current contract but must surface. |

If `env/velocity_cmd` is anomalously low compared to YAML midpoint, **stop the analysis** and run a 200k-sample numpy simulation of `_sample_velocity_cmd` to characterize the distribution before drawing any other conclusion. The sampler bug masquerades as every other failure mode.

### Per-foot stride / swing-time diagnostics

Added in v0.20.1-smoke2. Use to debug short-stride failures (smoke1 hit step_length 0.022 m vs 0.030 m gate; per-foot data tells us if one foot is doing all the stepping):

| Metric | Reducer | Per-event mean computation |
|---|---|---|
| `tracking/touchdown_rate_left/right_count` | MEAN | touchdown event count per ctrl step |
| `tracking/swing_air_time_left/right_event_s` | MEAN | mean swing time per event = value / matching touchdown count |
| `tracking/step_length_left/right_event_m` | MEAN | mean step length per event = value / matching touchdown count |

Symmetry check: `touchdown_rate_left_count ≈ touchdown_rate_right_count`; gross asymmetry indicates the policy is stepping with one leg only.

### Walking failure signatures (v0.19.5 / v0.20.1-aware)

Call these out explicitly:

- **standing local minimum** — `forward_velocity ≈ 0`, `episode_length` near horizon, reward driven by survival + soft posture. Action: prior is too easy to ignore; check `tracking/forward_velocity_cmd_ratio` < 0.5.
- **lean-and-skate exploit (v0.19.5 pattern)** — `forward_velocity` overshoots cmd by 1.5x+, `tracking/forward_velocity_cmd_ratio > 1.5`, residual saturates the residual bound on hip_pitch (prior-guided) / torque saturates (home base). Action: tighten residual bound (prior-guided) or check torque saturation (home base); do NOT increase cmd_vx reward weight.
- **train→eval gap (high action noise)** — *train-rollout* metrics look like sway/flail (low fv, tiny step_length, collapsed feet_phase) but the **deterministic** eval walks. Signature: policy `log_std`/`std` (e.g. 0.63) ≳ `action abs.mean` (e.g. 0.37). Action: ALWAYS run `eval_policy.py` (deterministic, from rest) before a verdict; if the gap is real, the fix is to reduce the entropy bonus / action noise, NOT the reward family. (smoke5: train fv 0.038 / step 0.0015 vs deterministic fv 0.130 / step 0.0316 / ratio 0.998.)
- **shuffle exploit (v0.20.1-smoke1 pattern)** — `forward_velocity` near cmd, `episode_length` saturated, but `tracking/step_length_touchdown_event_m < 0.025` and per-foot step lengths small on both feet. The policy is making cmd_vx via short fast steps instead of tracking the prior's foothold geometry. Action: check `ref/feet_pos_err_l2` is in the alive-gradient range; consider lowering `ref_feet_pos_alpha`.
- **gait drift** — `ref/contact_phase_match` decreases over training while `forward_velocity` increases. Action: if `ref_contact_match` weight is 0, this is expected (the contact-alignment probe gates the term); if the term is enabled, increase weight on `ref/contact_phase_match` and `ref/q_track`.
- **dead-gradient term** — large persistent `ref/<X>_err_*` paired with near-zero `reward/ref_<X>_track`. Means α is too tight for the observed error magnitude. Action: lower α to give r ≈ 0.2-0.3 at the iter-1 baseline error.
- **propulsion mis-assigned to PPO (anti-exploit violation, prior-guided only)** — forward_velocity meets gate but `|residual_p50|_{hip,knee}` > 0.20 rad. Action: tighten the residual bound by 50% and rerun; if still failing, prior is too weak — return to v0.20.0-x prior surgery (per the M1 fail-mode tree in `walking_training.md`). N/A under the home/TB-contract base (residual is the gait there — judge by torque saturation instead).
- **truncation-based success_rate is meaningless** — `env/success_rate` is permanently 0 for v0.20.x walking. Never gate on it; never sort checkpoints by it.
- **sampler-cmd-projection regression (v0.21.0-smoke2 pattern)** — `env/velocity_cmd` late-iter mean is significantly below `0.5 × (min_velocity + max_velocity)` from the YAML (e.g. smoke2: mean 0.083 vs YAML midpoint 0.22). Train-rollout `forward_velocity_cmd_ratio` looks healthy (cmd ≈ achieved ≈ 0), but the deterministic post-training eval at the configured `eval_velocity_cmd` shows ~0 forward velocity — because the **sampled** cmd distribution is collapsed toward zero by an `* sin(theta)` / `* cos(theta)` projection in `_sample_walk_command` even when `vx_positive_only` only drops the negative branch. Action: read the env sampler code (NOT just the YAML field names); simulate the cmd distribution with numpy (200k samples); compare distribution mean / median against YAML midpoint; if collapsed, fix the sampler before changing reward family or hyperparameters. **Fix landed in `ee399ee training: enforce smoke2 positive walk vx range`**; the legacy mode (TB symmetric ellipse) is preserved when `cmd_sampler_walk_vx_positive_only=False`.
- **dead velocity gradient from narrow high-band command (v0.21.0-smoke2/3 pattern)** — distinct from the sampler-projection regression above: here `env/velocity_cmd` MATCHES the YAML midpoint (sampler is healthy — e.g. smoke2 `5v4j5j7w`: 0.197 vs midpoint 0.22, Δ 10%), but `reward/cmd_forward_velocity_track ≈ 0.0001` for the entire run, `forward_velocity` flat/negative, `episode_length` saturated (survives by tap-in-place, not walking). Cause: `exp(-α·err²)` with α=562.5 has a live-gradient window of only `cmd ± 1/√α ≈ ±0.042 m/s`; if the walk branch samples a narrow HIGH band (smoke2 `vx ∈ [0.18, 0.26]`) every env's target is unreachable from a standing cold start (err ≈ 0.22 → `exp(-35) ≈ 0`), so PPO never gets a forward gradient. **CORRECTION 2026-05-30 — do NOT cite smoke12b as a cold-start proof, and do NOT assume widening the floor fixes this.** smoke12b (`ot8zazo1`) walked at `vx ∈ [0.08,0.133]` with live `reward/cmd_forward_velocity_track ≈ 0.027`, BUT it was **warm-started**: its metrics begin at `progress/iteration=1720` / `time/step=35.2M` already walking (inherited from the smoke9c→11→12 curriculum chain). It did NOT break the basin cold. **No WR run has broken the walking basin from a cold (iter-1) start** — smoke1/2/3 all cold-start and all fail. The floor-widening fix this signature originally recommended (smoke3 `min_velocity→0.065`) ran the full 30M and **FAILED**: the policy entered a **lateral-walking** attractor (visual `lat_vel → 0.55 m/s` at cmd (0.26,0,0), `fv ≈ 0`), aggravated by `cmd_velocity_track_dim=2` saturating the reward to 0 under lateral drift. Diagnostic tell: `reward/feet_phase` can equal the walking value (~0.09) while `step_length ≈ 0` and `fv ≤ 0` — feet_phase rewards swing height independent of horizontal progress, so it does not disambiguate forward-walk from tap/lateral. So when you see this signature: (1) check `cmd_velocity_track_dim` — dim=2 couples forward+lateral and can zero the forward reward (try `dim=1`); (2) check `tracking/lateral_velocity_abs` / `env/lateral_velocity` — the failure may be lateral, not standing; (3) treat "the lever is the command range" as UNPROVEN for cold starts — the real open question is cold-start basin-break, which likely needs a curriculum/warm-start (per smoke12b's lineage). Do NOT lower α (reopens smoke1 standing-leak). See CHANGELOG `v0.21.0-smoke3-result + smoke4A-plan`.

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
| `cmd_forward_velocity_alpha` ↔ `lin_vel_tracking_sigma` | α=562.5 (smoke2/3 active; "after sweep" doc value was 200) | σ=1000 (`walk.gin:112`; reward `exp(-σ·err²)` at `mjx_env.py:2346`) | ⚠️ **CORRECTED 2026-05-29: same form `exp(-param·err²)`, so TB σ=1000 is _tighter_ than WR, not looser (earlier "WR tighter" note was backwards). Live-gradient window = `cmd ± 1/√param`: TB ±0.032 vs WR ±0.042 m/s. ⟹ the smoke2 dead gradient is NOT an α problem — TB runs an even tighter tracker and walks. Do NOT lower α (reopens smoke1 standing-leak). It is a command-RANGE problem; see the "dead velocity gradient" failure signature.** |
| `ref_body_quat_track` ↔ `torso_quat` | 2.5 (after sweep) | 2.5 (`walk.gin:117`) | ✅ (after sweep) |
| `ref_q_track` (joint imitation) | 5.0, α=1 | 0 in active (was 5.0 in commented `walk.gin:63`) | ⚠️ WR-specific imitation block (no TB-active analog) |
| `torso_pos_xy` | 2.0, α=90 smoke | 0 in active (was 1.0 in commented) | ⚠️ WR-specific imitation |
| `lin_vel_z` | 1.0 | 0 in active | ⚠️ WR mirrors commented `walk.gin:60`; defer |
| `ang_vel_xy` (tracking) ↔ `penalty_ang_vel_xy` (penalty) | 2.0 tracking | 1.0 penalty (`walk.gin:115`) | ⚠️ form mismatch; defer |
| `feet_air_time` | 500.0 | 0 in active (replaced by `feet_phase` in active) | ⚠️ WR-specific (TB-active does not use this) |
| `feet_clearance` | 1.0 | 0 in active (replaced by `feet_phase`) | ⚠️ WR-specific |
| `feet_distance` | 1.0, [0.07, 0.13] | 0 in active; TB uses `penalty_close_feet_xy` | ⚠️ WR keeps `feet_distance`; **also has** `penalty_close_feet_xy = 10.0` enabled in smoke2 (verify: `wildrobot_env.py:4330`, smoke2 YAML 531) |
| **`feet_phase`** (dense per-step swing-height tracker) | **PRESENT** (smoke9+, `wildrobot_env.py:1213 _feet_phase_reward`); smoke2 enables at w=7.5, α=914.304, swing_height=0.05 (YAML 532-534) | 7.5, σ=0.0007, swing=0.04 (`walk.gin:128-131`, `walk_env.py:631-695`) | ⚠️ **WR-specific divergence — load-bearing**: WR uses `walking_reward = max(0, raw - flat_foot_baseline)` per smoke12 fix (docstring `wildrobot_env.py:1226-1244`). Reason: smoke11 measured raw TB formula paid `≈ 0.109` to a flat-foot policy, enabling a quiet-standstill exploit. Do NOT propose re-porting raw TB formula without rerunning that experiment. |
| `ref_feet_z_track` (alternative WR per-step foot-z tracker) | **PRESENT** (logged as `reward/ref_feet_z_track`); smoke2 gates to **w=0.0** (YAML 539) — feet_phase is the active form | n/a — separate from TB `feet_phase` | ⚠️ exists but disabled in smoke2; flip on only as an A/B against `feet_phase` |
| **`penalty_pose`** (per-joint default anchor) | **PRESENT** (`wildrobot_env.py:687, 2596`); smoke2 enables at w=-0.5 (YAML 530) | 0.5 with weights `[0.01, 1.0, 5.0, 0.01, 5.0, 5.0, ...]` (`walk.gin:120-124`) | ⚠️ check the per-joint weight map (`penalty_pose_weight_default` + per-joint overrides) against TB's vector before tuning |
| **`penalty_feet_ori`** (anti-tippy-toe via gravity in foot frame) | **PRESENT** (`wildrobot_env.py:2598, 4326`); smoke2 enables at w=5.0 (YAML 535) | 5.0 (`walk.gin:129`, `walk_env.py:697-735`) | ✅ weight matches TB |
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
| walking-quality / anti-exploit / baseline-beat gates | WildRobot-specific | n/a | ⚠️ acceptance criteria are WildRobot-specific |

Symbol meaning:
- ✅ aligned to TB-active.
- ⚠️ intentional divergence with documented rationale; confirm rationale is current.
- ❌ missing TB-active term that closes a known WR failure mode (or compute gap that blocks parity); flag as a parity risk.

When a run violates the contract on a ✅ row, surface it as a bug.
When it diverges on a ⚠️ row, confirm the rationale is current; if not, flag as drift.
When it lands on a ❌ row, that's a known parity gap — link to `walking_training.md` Appendix B.2 for the phased fix plan.

### Walking interpretation rules

- If `Evaluate/forward_velocity` is near zero AND `Evaluate/mean_episode_length` is near horizon → **standing local minimum** — call it out, don't say "still training".
- If `Evaluate/forward_velocity` meets gate AND `tracking/forward_velocity_cmd_ratio` is in band AND anti-exploit passes (no torque saturation; residual ≤ 0.20 *for prior-guided runs only*) → **locomotion emerging**. Recommend the iter that maximizes `Evaluate/mean_reward`.
- If `Evaluate/forward_velocity` meets gate AND anti-exploit fails → **action-authority leak / propulsion mis-assigned**. For *prior-guided* runs (`loc_ref_residual_base: q_ref`): tighten the residual bound 50%, do NOT relax the anti-exploit gate. For *home/TB-contract* runs the residual bound doesn't apply — look at torque saturation + stepping-not-skating instead.
- If `tracking/cmd_vs_achieved_forward` mean is high (> 0.075) AND `Evaluate/mean_episode_length` is near horizon → **balance achieved, tracking weak**. Check BOTH `cmd_forward_velocity_alpha` (562.5 in smoke2/3) AND the sampled cmd band: a tight α paired with a high/narrow cmd band gives a dead gradient (see "dead velocity gradient" signature), which is a command-RANGE problem, not a tracking-weight problem — do NOT raise the cmd-track weight or lower α to "fix" it.
- If `ref/feet_pos_err_l2` grows over training while `reward/ref_feet_pos_track` stays at 0 → **dead-gradient term**. Lower the alpha (smoke uses 30, not 200).
- If `term_pitch_frac > 0.5` AND `forward_velocity` meets gate AND anti-exploit passes → **balance issue (M1 branch)**. Increase regularizer weight on `pitch_rate` (M1 hook is pre-wired at weight 0); consider widening residual bound on ankle pitch only.

### Walking next-step template

```text
verdict (judge by the DETERMINISTIC eval; note train→eval gap if action std ≫ signal):
- locomotion emerging / standing local minimum / lean-and-skate exploit / shuffle exploit / dead-gradient on <term> / dead velocity gradient (narrow high-band cmd) / gait drift / balance issue / sampler-cmd-projection regression / train→eval gap (high action noise)
- walking-quality: forward_velocity <pass/fail> | mean_episode_length <pass/fail> | cmd_vs_achieved <pass/fail> | step_length <pass/fail>
- anti-exploit: velocity_cmd_ratio <pass/fail> | torque-saturation <pass/fail> | residual_p50 <pass/fail — prior-guided only, N/A for home base>
- baseline-beat: <pass/fail>
- Sampler sanity: env/velocity_cmd late-iter mean = <X> vs YAML midpoint <Y> (Δ = ...; significant gap means simulate _sample_velocity_cmd before drawing reward-family conclusions)
- ToddlerBot alignment: <ok / drift on <row(s)>>
```

**Before recommending ANY change, complete the pre-flight verification checklist (top of skill).** Specifically:
- [ ] Grepped `wildrobot_env.py` for every reward term, env mechanism, and sampler function you plan to cite as "missing" or "needs adding". If it exists, READ the implementation (don't assume it matches TB).
- [ ] Confirmed via `metrics.jsonl` whether the term is currently logged (= wired). If logged, you cannot recommend "add it" — only "modify it" or "tune the weight".
- [ ] Computed `env/velocity_cmd` mean vs YAML midpoint; if Δ > 20%, **suspect a sampler projection bug FIRST**, not the reward family.
- [ ] If you found a stale row in the TB-comparison table, FIX the skill before writing your recommendation.

Then recommend one of (per the M1 fail-mode tree):
- continue training to the M2 compute cap unchanged
- prior surgery (return to v0.20.0-x; do NOT relax the anti-exploit gate)
- tighten the residual bound 50% and rerun (prior-guided runs only)
- promote the M1 `pitch_rate` regularizer
- regenerate the offline reference library
- **fix sampler if `env/velocity_cmd` distribution doesn't match YAML range** (rerun current YAML on the fix before any reward-family change)

### Walking design guidance (when planning the next config)

Order of operations:
1. **Confirm the pre-smoke probes pass.** Zero-action q_ref replay (zero-residual-replay invariant), contact-alignment probe (gates `ref_contact_match` weight), prior-vs-body sanity (`tools/v0200c_per_frame_probe.py`).
2. **Match ToddlerBot weights.** Use Appendix A of `walking_training.md` as the alignment audit. Divergences need explicit rationale.
3. **Don't invent new reward terms** when standard locomotion terms cover the problem. The current reward family is already imitation + cmd + ToddlerBot shaping + survival + small regularizers.
4. **Don't relax the anti-exploit gate** to fix a forward-velocity miss; relaxing it brings back v0.19.5 (lean-and-skate).
5. **PPO hyperparameters last.** Learning rate / entropy / KL backoff are the last knobs to touch.

Forbidden moves (will reintroduce known failure modes OR repeat past analyst errors):
- adding a flat positive `alive` bonus (was the v0.19.5b lean-back exploit enabler)
- raising the leg residual bound past ±0.50 rad
- removing the `* dt` reward integration
- ranking checkpoints by `env/success_rate` or `env/episode_length` alone
- **recommending "add X reward term" without first `grep`-ing `wildrobot_env.py` to prove X doesn't exist AND checking `metrics.jsonl` to confirm `reward/X` isn't already logged** (this is the smoke2-analysis miss — I recommended adding `feet_phase` which had been present since smoke9 and was actively enabled in smoke2 with `reward/feet_phase = 0.0028` logged in the run)
- **trusting the TB-comparison table without verifying the row against current code** (the table is a snapshot, not a live mirror — see "Pre-flight verification" at top)
- **porting raw TB reward formulas back over WR-specific divergences without re-running the experiment that motivated the divergence** (e.g. raw TB `feet_phase` would reopen the smoke11 flat-foot reward leak that smoke12's `max(0, raw - flat_foot_baseline)` fix closed — divergence rationale is documented in `wildrobot_env.py:1226-1244`)
- **drawing reward-family conclusions before checking that the env sampler is emitting the YAML-stated cmd distribution** (smoke2 sampler bug masked the alpha-leak-fix as a failure — fix in `ee399ee`)

---

## CHANGELOG update checklist

1. Add a new entry for the next training version (v0.xx+1): config + code changes.
2. Under the current version entry, record:
   - run path, checkpoint dir, best checkpoint path
   - walking-quality row-by-row pass/fail, anti-exploit (velocity_cmd_ratio + torque-saturation + residual for prior-guided), baseline-beat, ToddlerBot alignment status
   - Best `Evaluate/mean_reward`, `Evaluate/mean_episode_length`, `Evaluate/forward_velocity`, `Evaluate/cmd_vs_achieved_forward` at the chosen checkpoint
   - Concerning trends not gated (e.g. `ref/feet_pos_err_l2` growth, `ref/contact_phase_match` drift)
   - Observed failure mode(s) per the v0.19.5 / v0.20.1 signature list and the recommended next intervention per the M1 fail-mode tree
3. Newest version entry on top.

The scripts print a markdown block intended to paste into `training/CHANGELOG.md`.

Standing / M3 / FSM analysis material (FSM verdicts, swing occupancy, step-event / foot-place / posture rewards, push-recovery failure signatures) is no longer carried in this skill. The active locomotion target is v0.20.1 walking; runs from older standing / M3 branches should be analyzed against their own historical SKILL revisions.
