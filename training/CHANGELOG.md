# Training Changelog

This changelog tracks capability changes, configuration updates, and training results for the WildRobot training pipeline.

**Ordering:** Newest releases first (reverse chronological order).

**History prior to v0.20.1:** see [`CHANGELOG.archive.md`](CHANGELOG.archive.md).

---

## [v0.20.1-reference-quality-snapshot] - 2026-04-26: Where WR sits vs TB after Phases 1-6 (refreshed parity report)

### Context

Status read of WR reference quality vs ToddlerBot after Phases 1-6 of
`training/docs/reference_architecture_comparison.md` have all landed
(swing-z continuity fix, TB-style `mujoco_replay` enrichment,
command-integrated torso target, lifecycle cache, `cycle_time_s`
tuning). No code change in this entry. Sources: refreshed
`tools/parity_report.json` produced under `cycle_time_s = 0.72` (both WR
and TB sides re-run), code spot-check confirming `ZMPWalkConfig` matches
the report's WR cache fingerprint `2051a03ebdc1a796`.

### Result — gate verdicts (WR vs TB-2xc, vx=0.15 unless noted)

| Layer | Gate | Status | Reading |
|---|---|---|---|
| P0 in-scope (vx ≤ 0.15) | both gates | **PASS** | clean |
| P0 full matrix | stance_gate | **PASS** (worst +1.9 mm) | high-vx bins clear |
| P0 full matrix | swing_gate | **PASS** (margin +17 mm) | total failures 0 |
| P1A absolute | step_len / cadence / clearance / speed / DS | **PASS** | `verdict: PASS` for both TB baselines |
| P1A normalised | `step_length_per_leg` ≥ 0.85× TB | **FAIL** (0.56× TB) | proportional stride deficit |
| P1A normalised | `swing_clearance_per_com_height` ≥ 0.85× TB | **FAIL** (0.82× TB) | one `foot_step_height_m` bump from PASS |
| P1A normalised | `cadence_froude_norm` ≤ 1.20× TB | **FAIL** (1.25× TB) | hardware-Froude bounded |
| P1A normalised | `swing_foot_z_step_per_clearance` ≤ 1.10× TB | **FAIL** (1.40× TB) | half-sine apex peaky |
| P2 absolute | `swing_foot_z_step_max` ≤ 1.10× TB | **FAIL** (1.83× TB) | same root cause as the line above |
| P2 absolute | shared_leg_q / pelvis / flips | **PASS** | WR `q_step` 0.36 vs TB 0.65 — actually smoother |
| P1 closed-loop | episode survival | **PASS** | WR survives 4-9× more steps than TB at every vx |
| P1 closed-loop | forward velocity | **PASS** (both baselines, all vx) | WR drifts less backward than TB |
| P1 closed-loop | terminal drift per leg | **PASS** | WR ~½ TB's per-leg drift rate |
| P1 closed-loop | shared-leg RMSE | **FAIL** (WR 0.16 rad vs TB 0.10) | actuator-stack artifact, see interpretation |
| P1 closed-loop | contact-phase match | **FAIL** (WR 0.58–0.70 vs 0.90 floor) | WR misses its own absolute floor |
| P1 closed-loop | touchdown step length per leg | **FAIL** | WR strides under-leg-length under replay |

**Top-line:** every gate that asks "is the WR prior physically trackable
closed-loop in absolute terms" — survival, forward speed, drift —
**passes, and WR clears them by larger margins than TB does.** What
fails:

1. **P1A normalised gait shape (4 gates)** — WR is taking shorter,
   faster steps than a TB-proportionally-scaled WR would. Bounded by
   leg length at vx=0.15.
2. **P2 absolute swing-foot-z apex (1 gate)** — WR's half-sine envelope
   concentrates Δz at the apex; TB's triangular-then-flat profile is
   smoother frame-to-frame.
3. **P1 closed-loop RMSE / contact / step (3 gates)** — these are
   replay-of-prior, mixing prior quality and actuator-stack mismatch
   (WR position-actuators vs TB torque+PD). The scorecard's own
   interpretation note covers this exact case: "comparable RMSE on
   both sides plus a P1 fail points to actuator / control mismatch
   rather than the prior". WR's RMSE *isn't* comparable to TB's
   (1.7× higher) and survives much longer, which is more consistent
   with "different actuator-stack equilibrium" than "prior is
   untrackable" — WR keeps integrating the position-target error
   along long episodes. **Worth diagnostic follow-up before crediting
   purely to actuator stack.**

By the doc's own decision rule WR is **not yet "on par or better"**
because two layers (normalised P1A and the P2 swing-step gate) have a
clear FAIL with a TB-aligned planner-shape fix available. The closed-
loop layer is more nuanced: WR strictly improves on TB's
absolute-physical gates (survival, drift, forward) but fails a per-joint
absolute gate where the test design itself is known to confound prior
quality with actuator stack.

### Direction to close the remaining gap (priority order)

1. **Swap half-sine swing-z apex for TB-style triangular profile.**
   This single planner-local change in `control/zmp/zmp_walk.py`
   (mirroring `projects/toddlerbot/algorithms/zmp_walk.py`) addresses
   **both** the P2 absolute `swing_foot_z_step_max` FAIL (1.83× TB) and
   the normalised `swing_foot_z_step_per_clearance` FAIL (1.40× TB).
   Highest-confidence TB-alignment move with a measured FAIL on a
   non-hardware-bounded gate.

2. **Bump `foot_step_height_m`** (e.g. 0.040 → 0.045) to lift
   `swing_clearance_per_com_height` from 0.143 to ≈ 0.16 (gate 0.149)
   → PASS. Sequence after item 1 so the new triangular envelope absorbs
   the higher apex.

3. **Curriculum / operating-vx decision for the two hardware-Froude
   gates.** `step_length_per_leg = 0.56× TB` and
   `cadence_froude_norm = 1.25× TB` at vx=0.15 are bounded by leg
   length: a 1.76×-longer-leg robot at the same absolute vx is at a
   smaller Froude number, so per-leg stride must under-shoot and
   normalised cadence must over-shoot. Closing these needs walking WR
   at TB's Froude-equivalent vx (≈ 0.19 m/s), not more planner tuning.
   This is a `walking_training.md` curriculum question.

4. **Diagnose the P1 contact-match-frac FAIL before attributing it
   purely to actuator stack.** WR's best contact match is 0.70 at
   vx=0.15, vs the 0.90 absolute floor. Three concrete checks:
   - Is the WR cycle-0 transient (TB truncates it, WR retains) eating
     the early-episode contact alignment? Compare match_frac on
     [first 30 steps] vs [steps 30-end].
   - Are the contact flips coming from the swing-z apex peakiness?
     Item 1 should drop residual mid-swing tap-down events.
   - If both above are clean and match_frac is still < 0.90, the
     residual is the actuator stack and we accept it as an inherent
     P1-test-design limitation per the scorecard's own note.

**Not a direction:** further `cycle_time_s` tuning (Phase 6 already
adopted TB's value); broader changes to the runtime target semantics
(Phase 4 already aligned them and the closed-loop drift / forward gates
now pass).

### Notable changes vs the pre-refresh estimate (now superseded)

- P1 closed-loop is **not unknown** — it's measurably better than TB on
  the absolute-physical gates (survival, drift, forward). The pre-
  refresh "this might collapse the backlog if P1 PASSes" framing was
  partially right (drift / forward / episode all PASS) and partially
  wrong (RMSE / contact / step still FAIL with non-actuator components
  worth investigating).
- WR's `terminal_drift_rate_per_leg_per_s` at vx=0.10 dropped from
  −0.48 leg/s (pre-refresh, planner-phase reward target) to **−0.032
  leg/s** (post-refresh, command-integrated reward target). This is a
  **15× improvement** and is the cleanest quantitative confirmation
  that Phase 4's TB-aligned torso target semantics had the predicted
  effect on closed-loop drift.

## [v0.20.1-phase6-cycle-time-tuning] - 2026-04-26: Phase 6 landed (TB-aligned `cycle_time_s` 0.64 → 0.72)

### Context

Implements Phase 6 from `training/docs/reference_architecture_comparison.md`:
local scalar tuning on top of the aligned reference architecture (Phases
1-4 already in tree).  Adopts ToddlerBot's `cycle_time = 0.72 s` (TB
`mjx_config.py:107`, `walk.gin:19`), the explicit Item 1 recommendation
from the parity scorecard's "Architectural alignment moves toward
ToddlerBot (priority order)" section.

### Changes

- `control/zmp/zmp_walk.py`: `ZMPWalkConfig.cycle_time_s` 0.64 → 0.72.
  Single scalar; planner pipeline / swing-z envelope / FK enrichment /
  reward-target semantics from Phases 1-4 are unchanged.
- `tools/phase6_wr_probe.py` (new): WR-only probe that reuses the parity
  tool's WR-side helpers (`_summarize_wildrobot`,
  `_wr_fk_and_smoothness`) without the TB venv, so before/after WR
  metrics can be captured against the cached TB baseline in
  `tools/parity_report.json`.
- `training/docs/reference_architecture_comparison.md`: added Phase 6
  closeout with full before/after table.

### Verification

- `uv run pytest tests/test_v0200a_contract.py tests/test_v0200c_geometry.py`
  — 19/19 PASS unchanged.
- `uv run pytest tests/test_runtime_reference_service.py` — 16/16 PASS.
- full repo `uv run pytest tests/` — 161/161 PASS.
- `uv run python tools/phase6_wr_probe.py` (vx=0.15) before / after.

### Result

WR vs TB-2xc at vx=0.15 (TB cached from `tools/parity_report.json`):

| Gate | Pre (cycle=0.64) | Post (cycle=0.72) | Verdict change |
|---|---|---|---|
| P0 full-matrix failures | 7 | **0** | FAIL → PASS (no in-scope regression) |
| P0 worst stance z (m) | +0.0075 | +0.0019 | -75 % |
| P0 worst swing z (m) | -0.0011 | +0.0170 | +18 mm margin |
| P1A step_length absolute (≥0.90× TB) | 0.883× | **0.994×** | FAIL → PASS |
| P1A step_length_per_leg (≥0.85× TB) | 0.501× | 0.564× | FAIL→FAIL (closes 13 %) |
| P1A cadence_norm (≤1.20× TB) | 1.412× | 1.252× | FAIL→FAIL (closes 60 %) |
| P2 swing_foot_z_step_max (m) | 0.0201 | 0.0183 | -9 % |
| P2 shared_leg_q_step_max (rad) | 0.3632 | 0.3599 | -1 % |

Three load-bearing things to call out:

1. P0 stance_z dropped from +7.5 mm to +1.9 mm — the entire deferred
   high-vx (≥ 0.20) stance-failure list from Phase 1 closeout is now
   inside the 3 mm tolerance.  The longer cycle drops the per-frame
   foot excursion at every vx, so the IK no longer asks for stance
   contacts the model can't realize.
2. P0 swing margin went from -0.3 mm under the floor (just barely
   passing the -2 mm gate) to +17 mm above it.  Real margin, not noise.
3. step_length_mean now matches TB to 0.3 mm at vx=0.15, exactly as
   predicted: `vx · cycle/2 = 0.15 · 0.72/2 = 0.054 m`.

### Scope (what this is and isn't)

This is one tuning slice, not a redesign-complete event.  Per the
"What 'On Par or Better' Means" rule in
`training/docs/reference_architecture_comparison.md`, WR ≈ TB requires
P0 + size-normalised P1A + P2 + P1 + own-G4/G7 all to clear.  After
this commit, P0 and the absolute P1A gates pass; **P2 swing_gate, all
four normalised P1A gates, and the P1 closed-loop layer remain
unresolved.**

### Open gates after this slice (vs cached TB-2xc, `tools/parity_report.json`)

| Layer | Gate | Post-Phase-6 | Verdict |
|---|---|---|---|
| P2 absolute | `swing_foot_z_step_max_m` ≤ 1.10× TB | 1.83× TB | **FAIL** |
| P1A normalised | `step_length_per_leg` ≥ 0.85× TB | 0.56× TB | **FAIL** |
| P1A normalised | `swing_clearance_per_com_height` ≥ 0.85× TB | 0.82× TB | **FAIL** |
| P1A normalised | `cadence_froude_norm` ≤ 1.20× TB | 1.25× TB | **FAIL** |
| P1A normalised | `swing_foot_z_step_per_clearance` ≤ 1.10× TB | 1.40× TB | **FAIL** |
| P1 trackability | full closed-loop replay | **not re-measured** | unknown |

### Follow-up needed before "WR on par or better" can be claimed

1. Re-run the full parity tool on a TB-capable machine to refresh P1
   under `cycle_time = 0.72`.  Local TB venv is missing `joblib`/`lz4`
   (same caveat as Phase 1 / Phase 3 closeouts); TB code is untouched,
   so cached TB numbers in `tools/parity_report.json` stay valid for the
   absolute / normalised P1A / P2 ratios above.
2. Address the `swing_foot_z_step` P2 gap — likely a planner-shape
   change (flatter apex / different swing-z profile), not a single
   scalar.  Phase 6 nudged it (0.0201 → 0.0183) but it is still 1.83× TB.
3. Decide on `foot_step_height_m` (closes
   `swing_clearance_per_com_height`, trades against P2 smoothness) and
   on operating `vx` / curriculum (closes `step_length_per_leg` /
   `cadence_froude_norm` at WR's larger size).

## [v0.20.1-phase4-runtime-target-alignment] - 2026-04-25: Phase 4 landed (TB-style torso target semantics)

### Context

Implements Phase 4 from `training/docs/reference_architecture_comparison.md`:
move WR torso/body XY reward-target consumption from planner/offline
`win["pelvis_pos"][:2]` to TB-style command-integrated runtime path semantics,
while keeping offline `q_ref`, foot targets, and contact annotations unchanged.

### Changes

- `control/references/runtime_reference_service.py`
  - added stateless helper:
    `RuntimeReferenceService.compute_command_integrated_path_state(...)`
  - helper is JAX-friendly and mirrors TB path integration semantics for
    constant `(vx, yaw_rate)` commands.
- `training/envs/wildrobot_env.py`
  - computes `t_since_reset = step_idx * dt` from existing
    `loc_ref_offline_step_idx`.
  - computes `path_state` once per step from command + elapsed time.
  - `_compute_reward_terms(...): r_torso_pos_xy` now tracks
    `path_state["torso_pos"][:2]`
    (`path_rot.apply(default_root_pos)+path_pos`) instead of planner
    `pelvis_pos[:2]`.
  - planner `win["pelvis_pos"]` remains available and is still used by
    diagnostics such as `feet_track_raw`.
- `tests/test_runtime_reference_service.py`
  - added focused tests for the new path-state helper (yaw-stationary and
    nonzero-yaw discrete TB-parity cases).
- `tests/test_v0201_env_zero_action.py`
  - added focused env-side probe proving
    `ref/torso_pos_xy_err_m` follows command-integrated path target semantics.

### Verification

- `uv run pytest tests/test_runtime_reference_service.py tests/test_v0201_env_zero_action.py tests/test_v0200a_contract.py tests/test_v0200c_geometry.py`
  - **42 passed**, 0 failed.

## [v0.20.1-smoke7-prep1] - 2026-04-24: TB-aligned multi-cmd + DR + eval-cmd override

### Context

smoke6-prep3 falsified the TB phase-signal hypothesis cleanly (terms alive
but stride still parked at 0.021 m vs the 0.030 m G4 gate).  Five smokes
of reward-only interventions (3/4/5/6/6-prep3) have exhausted the
reward-only fix space.  Per CLAUDE.md "follow ToddlerBot before
WR-specific design" and the M1 fail-mode tree shuffle-exploit branch,
the next TB-aligned mechanism is the training distribution itself: TB has
the literally-identical `feet_air_time` reward and avoids shuffle via
multi-cmd resample + zero_chance + DR (not via a stride-amplitude
reward term).

The original smoke7 plan was "multi-cmd alone, no DR" with the
hypothesis "vx=0.20 hits a cadence ceiling → forces stride extension."
That mechanism was **falsified** by an open-loop probe at vx ∈ {0.10,
0.15, 0.20, 0.25}: WR's prior uses **fixed cycle_time = 0.640 s** at
all vx (cadence 1.56 Hz/foot regardless of cmd; only stride per step
varies linearly with vx).  TB's prior is also fixed-cadence (0.72 s).
Neither prior pushes PPO toward a cadence ceiling — TB's anti-shuffle
mechanism must be the combination of multi-cmd distribution + DR
(backlash + friction + kp + IMU noise make micro-shuffles fragile).

Revised smoke7: enable the **full TB training distribution** (multi-cmd
+ DR jointly, Option C).  Trade-off accepted: if smoke7 succeeds, we
won't know which mechanism was load-bearing without ablation followups,
but it's the closest to TB's actual recipe and the highest a-priori
chance to break shuffle.

### Changes

#### YAML (`training/configs/ppo_walking_v0201_smoke.yaml`)

**Multi-cmd curriculum** (TB anti-shuffle lever #1):
- `min_velocity: 0.15 → 0.0`, `max_velocity: 0.15 → 0.20`
  (narrower than TB's `[-0.2, 0.3]` because WR prior library lacks
  negative-vx generation; defer to v0.20.4)
- `cmd_resample_steps: 0 → 150` (TB resample_time=3.0 s at ctrl_dt=0.02)
- `cmd_zero_chance: 0.0 → 0.2` (TB default; gates `feet_air_time` reward
  to 0 on ~20% of episode windows, killing the "always shuffle" basin)
- `cmd_deadzone: 0.0 → 0.05` (TB vx-channel default)

**Eval-cmd override** (G4 readout protection):
- New env field `eval_velocity_cmd: 0.15` — pins eval rollouts to a
  fixed cmd so G4 metrics stay directly comparable to single-cmd
  smokes 3-6-prep3.  Without this, eval would sample uniformly across
  [0.0, 0.20] and E[cmd] (~0.07 m/s) would sit near or below the G4
  forward_velocity floor of 0.075 m/s, making pass/fail uninformative.

**Domain randomization** (TB anti-shuffle lever #2):
- `domain_randomization_enabled: false → true`
- `domain_rand_friction_range: [0.5, 1.0] → [0.4, 1.0]` (match TB)
- `domain_rand_frictionloss_scale_range: [0.9, 1.1] → [0.8, 1.2]` (match TB)
- `domain_rand_kp_scale_range`: unchanged at [0.9, 1.1] (matches TB)
- `domain_rand_mass_scale_range`: unchanged at [0.9, 1.1] (WR-specific
  multiplicative; TB uses additive [-0.2, 0.2] kg — different semantics,
  kept WR's existing form)
- `domain_rand_joint_offset_rad`: unchanged at 0.03 (WR-specific; TB
  uses `rand_init_state_indices` instead)

**IMU noise** (sim2real hardening, secondary):
- `imu_gyro_noise_std: 0.0 → 0.05 rad/s` (~2.9°/s; conservative vs
  TB's gyro_std=0.25 rad/s steady-state under bias-walk RW)
- `imu_quat_noise_deg: 0.0 → 2.0` (conservative vs TB's quat_std=0.10
  rad ≈ 5.7°)

**NOT changed**:
- `push_enabled` stays `false` (TB walk default `add_push: False` per
  `toddlerbot/locomotion/mjx_config.py:172`; pushes are not the
  load-bearing TB anti-shuffle mechanism)
- Reward family (TB-complete and verified alive in smoke6-prep3)
- PPO hyperparameters
- Residual bounds (G5 has ample headroom)
- `action_delay_steps: 1` (already TB-aligned)

#### Env code (`training/envs/wildrobot_env.py`)

- Add `WildRobotEnv.reset_for_eval(rng)`: calls `reset(rng)` then
  overrides `velocity_cmd` to `env.eval_velocity_cmd` when it's `>= 0`.
  Sentinel value `-1.0` (default) preserves backward compatibility.
- Add `disable_cmd_resample: bool = False` arg to `WildRobotEnv.step`.
  Mirrors the existing `disable_pushes` flag.  When True, the
  mid-episode resample logic at lines 1644-1664 is gated off so eval
  cmd stays fixed throughout the episode.

#### Train.py (`training/train.py`)

- `batched_eval_reset_fn` calls `env.reset_for_eval(rng)` (was
  `env.reset(rng)`).  Falls back to sampled cmd when `eval_velocity_cmd
  < 0`.
- `batched_eval_step_fn` and `batched_eval_clean_step_fn` pass
  `disable_cmd_resample=True` so eval cmd stays fixed across the
  resample boundary.

#### Config dataclass (`training/configs/training_runtime_config.py`)

- Add `EnvConfig.eval_velocity_cmd: float = -1.0`.

#### YAML loader (`training/configs/training_config.py`)

- Add `eval_velocity_cmd=float(env.get("eval_velocity_cmd", -1.0))` to
  `_parse_env_config` so the new YAML key round-trips.

#### Round-trip tests (`training/tests/test_config_load_smoke6.py`)

- New test `test_smoke7_critical_env_settings`: pins all 14 smoke7-
  critical env values (multi-cmd, DR, IMU noise, eval-cmd override).
  Same protective pattern as `test_smoke6_critical_weights_nonzero` —
  catches silent loader/YAML drift that would otherwise turn smoke7
  into another null run like smoke6 was before the loader fix.

#### Walking training plan (`training/docs/walking_training.md`)

- Replace `v0.20.1-smoke7` milestone with the revised plan (Option C:
  DR + multi-cmd jointly).  Documents the falsification of the
  cadence-ceiling mechanism per the open-loop probe results.  Adds
  decision rule for the post-smoke7 outcome (success → v0.20.2 +
  ablation; failure → prior surgery or extend compute).

### Pre-flight verification (all completed)

- ✅ DR plumbing audit: `_sample_domain_rand_params` →
  `_get_randomized_mjx_model` applied per ctrl step
  (`wildrobot_env.py:489-513, 1492-1507`).
- ✅ Multi-cmd resample plumbing audit: `wildrobot_env.py:1644-1664`
  resamples velocity_cmd at `cmd_resample_steps` interval, honors
  zero_chance + deadzone via shared `_sample_velocity_cmd`.
- ✅ YAML loader audit: all DR + cmd_* + eval-cmd keys read by
  `_parse_env_config`.
- ✅ Round-trip test extended: 4/4 pass (was 3/3 pre-smoke7).
- ✅ G6 invariant: `tests/test_v0201_env_zero_action.py` 7/7 pass with
  smoke7 YAML.
- ✅ Behavioral check: `reset_for_eval` pins cmd to 0.15 across all
  tested seeds; training `reset` samples cmds in [0.0, 0.20] with
  proper zero_chance/deadzone distribution (5/8 non-zero in 8-seed
  sample, expected ~60%).
- ✅ Open-loop prior probe at vx ∈ {0.10, 0.15, 0.20, 0.25}: prior
  generates 1120-step trajectories at all bins; cadence is fixed
  cycle_time=0.640 s across all bins (so the original cadence-ceiling
  mechanism is falsified — smoke7 design adjusted to Option C
  accordingly).

### Falsification — what would tell us this design is wrong

- Eval cmd doesn't pin to 0.15 in the actual smoke7 run (loader
  drift not caught by round-trip test) → kill and fix immediately.
- DR enabled but `Evaluate/term_pitch_frac` jumps to >50% from
  smoke6-prep3's 0.0% → DR ranges are too aggressive for WR's
  actuator authority; tighten ranges (start with friction → [0.6, 1.0]
  and frictionloss → [0.9, 1.1]).
- Multi-cmd training-rollout `tracking/forward_velocity_cmd_ratio`
  drifts outside [0.6, 1.5] for sustained windows → policy can't track
  the broader cmd range; consider narrower vx_max=0.18 or an extended
  episode length.

### What we are NOT doing in smoke7

- Not enabling pushes (TB walk default; defer to v0.20.5 recovery).
- Not enabling yaw cmd (defer to v0.20.4 — WR prior lacks yaw).
- Not adding stride-amplitude reward terms (five-smoke evidence
  rules this out).
- Not widening residual bound past ±0.25 rad (G5 has ample headroom).
- Not relaxing G5.
- Not extending M2 compute budget yet (judge first at standard 150
  iters; if intermediate result, then extend).

## [v0.20.1-smoke6-prep3 — RUN RESULT] - 2026-04-24: TB phase-signal hypothesis falsified cleanly; proceed to smoke7

### Run

- Run: `training/wandb/offline-run-20260424_095050-jhwyfwk0`
- Checkpoints: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260424_095053-jhwyfwk0`
- Recommended checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260424_095053-jhwyfwk0/checkpoint_150_19660800.pkl`
- Previous comparable run: `training/wandb/offline-run-20260423_213122-v1kclz2w` (smoke6, broken)

### Result

- **Walking verdict:** locomotion emerging — best v0.20.1 run on absolute
  metrics, but the shuffle pattern persists.  TB phase-signal hypothesis is
  falsified cleanly: the new terms (`lin_vel_z`, `ang_vel_xy`) are alive,
  do real work on their corresponding errors, and improve `ref/contact_phase_match`
  meaningfully — but they do NOT close the stride gap.
- **Best `Evaluate/mean_reward` = 309.13** (vs smoke6 280.69, +10%; best of
  any v0.20.1 smoke).  `term_pitch_frac` = 0.00% at iter 150 (best stability
  observed in v0.20.1).
- **G4:** 5 of 6 hard gates pass at iter 150.
  `Evaluate/forward_velocity=0.144`, `Evaluate/mean_episode_length=500.0`,
  `Evaluate/cmd_vs_achieved_forward=0.035`, `env/forward_velocity=0.143`,
  `tracking/cmd_vs_achieved_forward=0.075` (borderline) all pass.
  `tracking/step_length_touchdown_event_m=0.0213` still well below the
  `>= 0.030 m` gate (71% of gate; per-foot 0.0028/0.0029 m × touchdown
  rate 0.133/0.100, bilateral short-step pattern unchanged).
- **G5:** pass.  `tracking/forward_velocity_cmd_ratio=0.98` perfectly centered;
  residual p50s 0.091-0.103 rad (well under 0.20 cap; smallest of any v0.20.1
  smoke).
- **G7 baseline-beat:** pass comfortably.

### TB phase-signal hypothesis — explicit falsification

This is what smoke6 was originally supposed to test (and smoke6 didn't, due
to the YAML loader bug).  smoke6-prep3 is the actual test:

| Signal | smoke6 (broken) | smoke6-prep3 | Verdict |
|---|---:|---:|---|
| `reward/lin_vel_z` (w=1.0, α=200) | 0.000 (DEAD) | **0.0076** | ✅ alive — kernel works at training-time err |
| `reward/ang_vel_xy` (w=2.0, α=0.5) | 0.000 (DEAD) | **0.0273** | ✅ alive — kernel works |
| `reward/ref_contact_match` (w=1.0 boolean) | 0.0261 | 0.0269 | flat (term unchanged) |
| `ref/lin_vel_z_err_m_s` | 0.140 | **0.114** | ✅ -19% — phase-signal pull works |
| `ref/ang_vel_xy_err_rad_s` | 1.063 | **0.853** | ✅ -20% — phase-signal pull works |
| `ref/contact_phase_match` | 0.662 | **0.694** | ✅ +0.032 — gait alignment improved |
| `tracking/step_length_touchdown_event_m` | 0.0199 | **0.0213** | ❌ +0.0014 (+7%); still 71% of gate |

Three load-bearing conclusions:

1. **Both TB α values work as-is.**  The smoke6-prep2 worry that `α_lin_vel_z=200`
   and `α_ang_vel_xy=0.5` would be dead at training-time err was wrong — both
   terms are alive and reducing the paired errors by ~20%.  **No α recalibration
   commitment needed.**  The smoke6-prep2 conditional ("if smoke6 shows the term
   dead at training-time, recalibrate") is now resolved as "no recalibration
   needed."
2. **The reward family is now genuinely TB-complete and working.**  Best
   `Evaluate/mean_reward` of any v0.20.1 smoke (309.13, +10% vs smoke6).
   `term_pitch_frac` at iter 150 is 0.00% (smoke6: 2.62%); residuals are the
   smallest we've ever observed (~0.10 rad p50 vs smoke6's 0.10-0.11).  The
   phase signals are doing real work in their domains.
3. **Phase signals do NOT close the stride gap.**  Stride moved 0.0199 → 0.0213
   (+7%) — within run-to-run noise of the smoke3/4/5/6 cluster (0.018-0.020 m).
   With the full TB-aligned reward family enabled and active, the shuffle
   pattern persists.  **The phase-signal-fixes-shuffle hypothesis is falsified
   cleanly** — the terms work where TB intends (phase tracking, body posture)
   but phase tracking is not the load-bearing axis for stride amplitude.

### Pattern across smoke3-6-prep3 (five reward-only interventions, zero stride movement)

| Smoke | Intervention | What was actually enabled | Stride |
|---|---|---|---:|
| 3 | world-frame foot α=2 | foot reward alive at α=2 | 0.0178 |
| 4 | torso_pos_xy α=90 | torso_pos_xy alive at α=90 | 0.0204 |
| 5 | body-frame foot α=100 | DEAD (probe under-predicted training err) | 0.0182 |
| 6 | TB phase signals (lin_vel_z, ang_vel_xy, contact bool) | only ref_contact_match enabled (loader bug) | 0.0199 |
| 6-prep3 | same as smoke6, with loader fix | **all three terms alive and active** | 0.0213 |

Five reward-only interventions; stride distribution remains 0.018-0.022 m.
The reward-only intervention space is empty for the shuffle problem.

### Diagnosis (now with five-smoke evidence)

The shuffle pattern is **structural to single-point training without curriculum**,
not a reward-family deficiency.  With a fully TB-aligned reward family, PPO at
vx=0.15 still converges to micro-cadence because:

- `cmd_forward_velocity_track` is satisfied by cadence (no lever for stride).
- `feet_air_time = Σ(air_time × first_contact)` is cadence-invariant per unit
  time (TB has the same formula — `walk_env.py:300`).
- Phase signals (now alive and pulling) improve phase alignment but not
  amplitude.
- Short stride is more stable per touchdown; no reward penalizes the trade.

TB doesn't avoid this with a stride reward (TB has no stride-amplitude reward
either).  TB avoids it via the multi-cmd training distribution: at vx=0.3 (the
high end of TB's `[-0.2, 0.3]` range), the body's max sustainable cadence is
exceeded — PPO is **forced** to extend stride.  Once the long-stride basin is
learned, it persists at vx=0.15 within the same policy.

TB analytical stride at vx=0.15 is 0.054 m (`stride = vx · cycle_time / 2`,
`cycle_time = 0.72 s` per `toddlerbot/locomotion/mjx_config.py:98`).  The WR
G4 gate of `≥ 0.030 m` is ~56% of this nominal — a soft "must track at least
half the prior" gate.  TB's deployed PPO must clear this at vx=0.15 within its
multi-cmd training; the gate is correct.

### Regression vs smoke6 (loader bug aside)

| Signal | smoke6 | smoke6-prep3 | Δ | Status |
|---|---:|---:|---:|---|
| `Evaluate/mean_reward` | 280.69 | 309.13 | +28.4 (+10%) | improved |
| `env/forward_velocity` | 0.148 | 0.153 | +0.005 | flat |
| `env/episode_length` | 491.6 | 492.6 | +1.0 | flat |
| `tracking/step_length_touchdown_event_m` | 0.0199 | 0.0213 | +0.0014 | improved (still fails) |
| `tracking/forward_velocity_cmd_ratio` | 0.985 | 1.017 | +0.031 | flat |
| `tracking/residual_hip_pitch_left_abs` | 0.114 | 0.097 | -0.017 | improved |
| `ref/contact_phase_match` | 0.662 | 0.694 | +0.032 | improved |
| `term_pitch_frac` (best iter) | 2.62% | 0.00% | -2.62% | improved |

Phase signals strictly improve every body-stability and phase-alignment metric,
without moving the stride gate.  This is the cleanest-possible decoupling
evidence: the new reward terms work as designed, and they confirm what they
were not designed to do.

### Next version: v0.20.1-smoke7 — promote multi-cmd curriculum (TB anti-shuffle lever)

The five-smoke evidence exhausts the reward-only space.  Promote the next
TB-aligned mechanism per CLAUDE.md "follow ToddlerBot before WR-specific design"
and per the M1 fail-mode tree (shuffle exploit branch added in smoke6-prep3).

Smoke7 scope (already documented in `walking_training.md` v0.20.1-smoke7 §):

- `min_velocity: 0.0`, `max_velocity: 0.20` (start narrower than TB's `[-0.2, 0.3]`
  — no negative cmd, isolate the multi-cmd lever from the other deferred mechanisms)
- `cmd_resample_steps: 150` (3.0 s at ctrl_dt=0.02 — TB default)
- `cmd_zero_chance: 0.2` (TB default)
- `cmd_deadzone: [0.05, 0.05, 0.2]` (TB default)
- everything else identical to smoke6-prep3 (TB-complete reward family,
  no DR, no yaw, no prior changes)

Hypothesis: at the high end (vx=0.20) the body's max cadence is exceeded;
PPO is forced to extend stride; once learned, the long-stride basin persists
at vx=0.15.  G4 gate `Evaluate/step_length_touchdown_event_m at vx=0.15 ≥ 0.030 m`.

Falsifies cleanly:
- if stride moves at vx=0.15 → TB curriculum confirmed; promote to v0.20.2 (DR).
- if stride still parked → multi-cmd alone insufficient; either DR is co-load-bearing
  (jump to v0.20.2 jointly) or the prior is the actual blocker (return to
  v0.20.0-x prior surgery).  Per-foot stride asymmetry and per-vx-bin readout
  distinguish.

### What we are NOT doing in smoke7

- Not enabling DR (defer to v0.20.2 — keep the diagnostic clean; one new
  mechanism per smoke).
- Not enabling yaw cmd (defer to v0.20.4).
- Not adding stride-amplitude reward terms (TB doesn't have one; five smokes
  have ruled out reward space).
- Not widening the residual bound past ±0.25 rad (G5 has ample headroom:
  smoke6-prep3 residuals 0.097 vs 0.20 cap — the smallest we've ever seen).
- Not relaxing G5.
- Not recalibrating `lin_vel_z_alpha` or `ang_vel_xy_alpha` (smoke6-prep3
  shows TB α values work as-is; the smoke6-prep2 conditional is resolved).

### Pre-smoke7 checks

- `tests/test_config_load_smoke6.py` round-trip test still passes after
  YAML adds the new `cmd_*` keys (catches loader drift on the new fields).
- Run `tools/v0200c_per_frame_probe.py --vx 0.20` — confirm the prior can be
  replayed at the high end of the new range.  If the prior collapses at vx=0.20
  or quickly diverges, the hypothesis is moot until the prior is extended.
- Verify WR's `cmd_resample_steps` plumbing actually resamples (the field is
  wired but historically defaulted to 0 in smoke; confirm with a unit test or
  short rollout).

## [v0.20.1-smoke6-prep3] - 2026-04-24: YAML loader fix + round-trip test + roadmap update

### Context

Smoke6 (`offline-run-20260423_213122-v1kclz2w`) was a null run.  The YAML loader
at `training/configs/training_config.py:_parse_reward_weights_config` silently
dropped the `lin_vel_z` and `ang_vel_xy` weights set by smoke6's YAML — both fell
through to the dataclass default of 0.0, so the smoke6 thesis (TB-aligned
phase signals) was never tested.  Direct reproduction:

```
$ uv run python -c "from training.configs.training_config import load_training_config; \
    cfg = load_training_config('training/configs/ppo_walking_v0201_smoke.yaml'); \
    print(cfg.reward_weights.lin_vel_z, cfg.reward_weights.ang_vel_xy, cfg.reward_weights.ref_contact_match)"
0.0 0.0 1.0
```

YAML says `1.0, 2.0, 1.0`.  `ref_contact_match` is at line 698 of the loader and
loaded correctly; `lin_vel_z` and `ang_vel_xy` are missing from the loader.

### Changes

1. **`training/configs/training_config.py:_parse_reward_weights_config`**:
   add `lin_vel_z`, `ang_vel_xy`, `lin_vel_z_alpha`, `ang_vel_xy_alpha` to the
   `RewardWeightsConfig(...)` constructor with the same defaults as the
   dataclass (`0.0`, `0.0`, `200.0`, `0.5`).
2. **`tests/test_config_load_smoke6.py`** (new): load
   `ppo_walking_v0201_smoke.yaml`, walk every key under `reward_weights:` and
   `env:`, assert each round-trips through the loader to the matching dataclass
   field.  Catches this class of bug (silent drop on YAML keys not enumerated
   by the loader) at config-load time, before any training compute is spent.
3. **`training/docs/walking_training.md`**:
   - **M1 fail-mode tree:** add the **shuffle exploit** branch (forward_velocity
     + G5 + balance pass, but stride parks short).  Documented action: do NOT
     add another reward term; promote the next deferred TB curriculum lever.
   - **M1 shuffle-exploit context section:** add the TB analytical stride
     derivation (`stride_per_step ≈ vx · cycle_time / 2 = 0.054 m at vx=0.15`,
     citing `toddlerbot/algorithms/zmp_walk.py:258` and
     `toddlerbot/locomotion/mjx_config.py:98`).  Confirms the WR G4 gate of
     0.030 m is reasonable (~56% of TB's expected stride at the same command).
   - **`v0.20.1-smoke6-prep3` milestone:** loader fix + re-run unchanged;
     decision rule for the re-run.
   - **`v0.20.1-smoke7` milestone (planned):** promote multi-cmd resample +
     `zero_chance` + `deadzone` (vx range `[0.0, 0.20]`, no DR yet, no yaw).
     This is the TB anti-shuffle lever.  Decision rule for what comes after
     smoke7 (success → v0.20.2 DR; failure → DR + multi-cmd jointly OR prior
     surgery, distinguished by per-foot stride / per-vx-bin readouts).

### Why this sequence

- The four smokes (3/4/5/6) tested four different reward-only interventions and
  none moved stride.  The reward-only space is empty for the shuffle problem.
- TB has the same reward family and TB doesn't shuffle.  The structural
  difference is the curriculum (multi-cmd + DR), not the reward.
- Per CLAUDE.md "follow ToddlerBot before WR-specific design," promote the
  TB curriculum mechanisms next, not another WR-specific reward addition.

### What we are NOT doing in smoke6-prep3

- Not yet recalibrating `lin_vel_z_alpha` / `ang_vel_xy_alpha`.  The
  smoke6-prep2 commitment to recalibrate was conditional on observing dead
  terms with weights actually applied.  We don't have that evidence yet —
  the terms had weight 0.  Recalibration only makes sense if the loader-fixed
  re-run shows the terms are still dead with the weights enabled.
- Not yet enabling multi-cmd / DR / yaw.  Those are smoke7 / v0.20.2 / v0.20.4
  scope.  Keep smoke6-prep3 surgical: one bug fix, one rerun.
- Not promoting prior surgery.  The probe shows the prior delivers usable
  stride and the policy chooses not to track it; curriculum is a strictly
  less invasive intervention than re-engineering the prior.

### Verification

- `tests/test_config_load_smoke6.py` passes (catches the bug before the fix,
  passes after).
- `tests/test_v0201_env_zero_action.py` 7/7 pass (G6 invariant intact).
- Direct reproduction:
  ```
  $ uv run python -c "from training.configs.training_config import load_training_config; \
      cfg = load_training_config('training/configs/ppo_walking_v0201_smoke.yaml'); \
      print(cfg.reward_weights.lin_vel_z, cfg.reward_weights.ang_vel_xy, cfg.reward_weights.ref_contact_match)"
  1.0 2.0 1.0
  ```

## [v0.20.1-smoke6] - 2026-04-23: NULL RUN — YAML loader silently dropped lin_vel_z + ang_vel_xy weights

### Run

- Run: `training/wandb/offline-run-20260423_213122-v1kclz2w`
- Checkpoints: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260423_213125-v1kclz2w`
- Recommended checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260423_213125-v1kclz2w/checkpoint_150_19660800.pkl`
- Previous comparable run: `training/wandb/offline-run-20260422_230524-6fh9ufds` (smoke5)

### Result

- **Walking verdict:** smoke6 is a **null run**.  The smoke6 hypothesis ("TB-aligned
  phase signals + boolean contact reward will recover gait phase sync and unlock the
  stride gate") was **never actually tested** — two of the three new reward terms
  (`lin_vel_z`, `ang_vel_xy`) had weight 0 throughout training due to a YAML loader
  bug, not because of a kernel-shape or training-time err issue.  Only the third
  term (`ref_contact_match` boolean at w=1.0) was actually enabled.  The "stride
  still fails" outcome is therefore essentially smoke5 plus an inert boolean
  contact bonus, not evidence about the phase-signal hypothesis.
- **G4:** 5 of 6 hard gates pass at iter 150.
  `Evaluate/forward_velocity=0.146`, `Evaluate/mean_episode_length=500.0`,
  `Evaluate/cmd_vs_achieved_forward=0.035`, `env/forward_velocity=0.147`,
  `tracking/cmd_vs_achieved_forward=0.072` all pass.
  `tracking/step_length_touchdown_event_m=0.0185` still well below the `>= 0.030 m` gate.
- **G5:** pass.  `tracking/forward_velocity_cmd_ratio=0.915` centered, residual p50s
  0.097-0.114 rad (well under the 0.20 cap).
- **G7 baseline-beat sanity:** pass comfortably.

### Bug: YAML loader silently drops two of three new smoke6 reward weights

Direct config-load reproduction:

```
$ uv run python -c "from training.configs.training_config import load_training_config; \
    cfg = load_training_config('training/configs/ppo_walking_v0201_smoke.yaml'); \
    print(cfg.reward_weights.lin_vel_z, cfg.reward_weights.ang_vel_xy, cfg.reward_weights.ref_contact_match)"
0.0 0.0 1.0
```

YAML says `lin_vel_z: 1.0, ang_vel_xy: 2.0, ref_contact_match: 1.0`.  Loader returns
`0.0, 0.0, 1.0`.  `ref_contact_match` is correctly read at
`training/configs/training_config.py:698`; `lin_vel_z` and `ang_vel_xy` are NOT in
`_parse_reward_weights_config` and silently fall through to the dataclass defaults
(`RewardWeightsConfig.lin_vel_z = 0.0`, `ang_vel_xy = 0.0`).

Smoke6-prep1 added the `RewardWeightsConfig` fields and the env compute path; smoke6-
prep2 fixed the body-local rotation and switched contact match to TB boolean.  Neither
prep added the corresponding `rewards.get("lin_vel_z", ...)` /
`rewards.get("ang_vel_xy", ...)` lines to the YAML loader.  The env-side and reward-
shape changes were verified by `tests/test_v0201_env_zero_action.py` (G6 invariance),
which doesn't exercise the YAML loader path.

The on-train metrics confirm the bug.  `reward/lin_vel_z = 0.0` and `reward/ang_vel_xy
= 0.0` exactly across all 16 eval iters from iter 0 onward.  At iter 0 the residual is
zero (G6 invariant), so the body runs the bare prior — `ref/lin_vel_z_err_m_s ≈ 0.15`,
which at TB α=200 should give `r ≈ exp(-200·0.0225) ≈ 0.011` and contribution
`r · 1.0 · 0.02 ≈ 2e-4`.  Observed: exactly 0.0.  That can only happen if the weight
is 0; no kernel-shape or transient-spike argument can produce *exact* zero across
every recorded eval iter.

### Why `ref_contact_match` stayed flat (0.65 → 0.66)

The third smoke6 term was the only one that actually fired.  The TB boolean equality
count (per-step value 0/1/2) at TB w=1.0 contributed `reward/ref_contact_match=0.026`
on average (count ≈ 1.3 of 2 feet match — equivalent to the 0.65 diagnostic), so the
term was numerically alive but did not move `ref/contact_phase_match`.  Two reasons:

1. **Boolean has no continuous gradient.**  The reward only changes when feet cross
   the contact-force threshold; PPO has to find action gradients indirectly through
   downstream effects on contact transitions.  This is a weak signal compared to the
   continuous phase-signal terms (`lin_vel_z`, `ang_vel_xy`) that were supposed to
   provide dense per-step gradients alongside it.  Without those continuous companions,
   the boolean was on its own.
2. **Existing rewards already fully reward the shuffle solution.**  `feet_air_time`
   (cadence-invariant per unit time), `cmd_forward_velocity_track` (satisfied by
   cadence), and `ref_q_track` (joint-space, locally insensitive to small Cartesian
   stride deltas) all already pay out for the policy's current short-stride micro-
   cadence.  A single boolean term at w=1.0 isn't enough authority to overcome that.

So smoke6 confirms a narrower fact: **boolean contact match alone is insufficient to
move PPO out of the shuffle local minimum.**  It does NOT confirm or refute the
broader phase-signal hypothesis (which was supposed to be carried by the missing
continuous terms).

### Pattern across smoke3–6 (stride still flat — but smoke6 doesn't add evidence)

| Smoke | Intervention | What was actually enabled | Stride |
|---|---|---|---:|
| 3 | world-frame foot α=2 | foot reward alive at α=2 | 0.0178 |
| 4 | torso_pos_xy α=90 | torso_pos_xy alive at α=90 | 0.0204 |
| 5 | body-frame foot α=100 | DEAD (probe under-predicted training err) | 0.0182 |
| 6 | TB phase signals (lin_vel_z, ang_vel_xy, contact boolean) | **only ref_contact_match enabled** (YAML loader bug) | 0.0185 |

The four-smoke "stride is flat" pattern still stands, but smoke6 should be considered
inconclusive on its own intervention — re-run after fixing the loader to actually test
the smoke6 hypothesis.

### Regression vs smoke5

| Signal | Smoke5 | Smoke6 | Δ | Status |
|---|---:|---:|---:|---|
| `env/forward_velocity` | 0.138 | 0.148 | +0.010 | improved |
| `env/episode_length` | 500.0 | 491.6 | -8.4 | regressed (still passes) |
| `tracking/cmd_vs_achieved_forward` | 0.068 | 0.072 | +0.003 | regressed (still passes) |
| `tracking/step_length_touchdown_event_m` | 0.0182 | 0.0199 | +0.0017 | improved (still fails) |
| `tracking/forward_velocity_cmd_ratio` | 0.919 | 0.985 | +0.066 | improved |
| `tracking/residual_hip_pitch_left_abs` | 0.094 | 0.114 | +0.020 | regressed (still passes) |
| `ref/feet_pos_err_l2` (debug) | 1.049 | 1.011 | -0.038 | improved |
| `ref/contact_phase_match` | 0.671 | 0.662 | -0.009 | regressed |

All deltas are within run-to-run variance — consistent with the bug interpretation
(smoke6's effective reward family was smoke5 + boolean contact, ≈ smoke5 with noise).

### Next version: v0.20.1-smoke6-prep3 (loader fix + re-run as smoke6)

Smoke6 is a null result, not new evidence.  The correct next step is to fix the loader
and re-run with the *actually-intended* smoke6 reward family before drawing any
conclusion about the phase-signal hypothesis.

1. **Fix `_parse_reward_weights_config`** in `training/configs/training_config.py`
   (around line 695-710): add
   `lin_vel_z=rewards.get("lin_vel_z", 0.0)`,
   `ang_vel_xy=rewards.get("ang_vel_xy", 0.0)`,
   `lin_vel_z_alpha=rewards.get("lin_vel_z_alpha", 200.0)`,
   `ang_vel_xy_alpha=rewards.get("ang_vel_xy_alpha", 0.5)` alongside the existing
   v0.20.1 entries.
2. **Add a config-load assertion test** that catches this class of bug:
   `tests/test_config_load_smoke6.py` — load the smoke YAML, assert every key under
   `reward_weights:` round-trips to the dataclass field with the same value.  This
   would have caught smoke6-prep1.  General principle: any YAML key under
   `reward_weights:` that doesn't round-trip through the loader should fail loud at
   config-load time, not silently fall through to a default.
3. **Re-run smoke6** with no other changes.  This actually tests the smoke6
   hypothesis.

After the re-run, smoke6's CHANGELOG entry will be replaced by the real result; this
entry stays as the bug record.

### What we are NOT doing in smoke6-prep3

- Not yet recalibrating `lin_vel_z_alpha` / `ang_vel_xy_alpha`.  The smoke6-prep2
  decision to use TB α values was conditional on observing training-time err under
  PPO control — we don't have that evidence yet because the terms were never enabled.
  Recalibration only makes sense if the re-run shows the terms are genuinely dead
  with the weights actually applied.
- Not promoting prior surgery, stride-lever interventions, or wider residual bounds.
  The four-smoke "stride is flat" pattern is real, but smoke6's contribution to that
  evidence is weakened by the bug.  Wait for the loader-fixed re-run before deciding
  whether the next branch is "more imitation terms" or "prior surgery / wider
  residual / cadence-vs-stride trade-off".

## [v0.20.1-smoke6-prep2] - 2026-04-22: TB-strict contact + body-local lin_vel_z + doc/probe cleanup

### Context

A code review of smoke6-prep1 caught four issues:

1. `ref_contact_match` weight was set to 0.5 — 25× past the documented
   re-entry weight (0.02), without re-running the contact-alignment probe
   that would have justified relaxing the gate.
2. The Gaussian kernel formula in `walking_training.md` G3
   (`exp(-x²/σ²)`) didn't match the implementation in
   `wildrobot_env.py` (`exp(-x²/(2σ²))`).  Doc/code mismatch.
3. `lin_vel_z` actual was computed from `HEADING_LOCAL` (yaw-only
   rotation, z stays world-up) rather than TB's true body-local z
   (rotated by `inv(root_quat)`).  Diverges from TB under body tilt.
4. The smoke5 changelog overstated "design hypothesis falsified" — smoke5
   alone only showed α=100 was dead; falsification required combined
   smoke3 + smoke5 evidence.

A follow-up review of smoke6-prep2 caught two additional documentation
and verification gaps:

5. Stale "Smooth per-foot contact-phase match" / "w=0 gated" labels in
   `metrics_registry.py`, `experiment_tracking.py`,
   `analyze_offline_run.py`, `SKILL.md`, and a YAML pre-`reward_weights`
   comment block — all written for the smoke3-5 Gaussian + gated design,
   not the smoke6 TB boolean form.
6. The `v0200c_per_frame_probe.py` ang_vel_xy reading used
   `data.qvel[3:6]`, while the env reward reads from
   `signals_raw.gyro_rad_s` (which goes through the `chest_imu_gyro`
   sensor with σ=0.0002 noise).  Not apples-to-apples.

### Resolution

**Reward implementation (smoke6-prep2 in commit 5a7e2f1):**
- `wildrobot_env.py` `_compute_reward_terms`: hoist the world→body
  inverse quaternion construction near the top; reused for both
  `lin_vel_z` (true body-local rotation) and `feet_distance`.
- `lin_vel_z` actual = `inv(root_quat) ⊗ data.qvel[0:3]`, take z.
  Matches `toddlerbot/locomotion/mjx_env.py:1645`
  (`get_local_vec(...)`).  Reference unchanged: finite-diff of prior
  `pelvis_pos[2]` (the prior is yaw-stationary with identity quat,
  so prior body-z velocity equals world finite-diff).
- `ref_contact_match`: TB boolean form
  `sum(stance_mask == ref_stance_mask) ∈ {0, 1, 2}`.  Matches
  `toddlerbot/locomotion/mjx_env.py:1721` exactly.  Drops the
  WR-specific Gaussian + sigma; resolves the doc/code mismatch by
  removing the disputed code.
- New diagnostic `contact_phase_match_diag = 0.5 × count` so
  `ref/contact_phase_match` continues to log a `[0, 1]` fraction for
  back-comparison with smoke3-5.
- `ppo_walking_v0201_smoke.yaml`: `ref_contact_match: 1.0` (TB weight),
  `lin_vel_z: 1.0`, `ang_vel_xy: 2.0`.

**Documentation cleanup (this entry):**
- `metrics_registry.py` and `experiment_tracking.py`: descriptions for
  `reward/ref_contact_match` and `ref/contact_phase_match` rewritten
  to describe the TB boolean form.
- `analyze_offline_run.py`: drop "w=0 gated" label; add rows for the
  three new smoke6 reward terms (`reward/ref_contact_match`,
  `reward/lin_vel_z`, `reward/ang_vel_xy`) and their diagnostics.
- `SKILL.md`: v0.20.1 reward family table now shows smoke6 weights
  for `torso_pos_xy`, `ref_contact_match` (TB boolean), `lin_vel_z`,
  and `ang_vel_xy`.
- `ppo_walking_v0201_smoke.yaml`: drop the obsolete pre-`reward_weights`
  comment block describing the Gaussian gate + 0.02→0.10 graduation
  policy.  That policy was specific to the smoke3-5 Gaussian shape and
  is superseded by the TB boolean switch.

**Probe alignment (this entry):**
- `tools/v0200c_per_frame_probe.py`: read ang_vel_xy from MuJoCo's
  `chest_imu_gyro` sensor (matches `signals_raw.gyro_rad_s` path in
  `training/sim_adapter/mjx_signals.py`).  Falls back to
  `data.qvel[3:6]` if the sensor is absent.  Verified the probe value
  agrees with `data.qvel[3:6]` to within the sensor's σ=0.0002 noise.

### Honest caveat: lin_vel_z probe alive band

At TB-aligned α=200, the open-loop probe at vx=0.15 shows transient
dead bands during body bobbing through phase transitions:
- `min raw reward over steps 0..50: 0.000005`
- Failures concentrated in steps 8-21, 25, 36, 40-41 (peak err
  ±0.25 m/s at heel-strike transients)
- Most steps remain in the alive regime (raw 0.05-1.0)

This does NOT meet the "alive throughout 0..50" bar that smoke4-prep1
used for `torso_pos_xy_alpha=90` calibration.  We are accepting
TB α=200 anyway on the rationale that:

1. The dead bands are open-loop pathologies.  Under PPO control the
   body is well-balanced and `body_lin_vel_z` should track the prior
   tightly (small err → high raw reward most of the time).
2. TB demonstrates α=200 works empirically on a similar small biped
   with the same body-local formulation.
3. Lowering α to "fix" the probe would diverge from TB; the smoke6
   philosophy (per saved memory) is to follow TB before introducing
   WR-specific calibration.

If smoke6 shows `reward/lin_vel_z` dead at training-time conditions
(persistent err > 0.12 m/s under PPO control), that's a real signal
and we recalibrate.  If it's alive at training time, the probe's
open-loop dead band was not load-bearing.

### Verification

- `tests/test_v0201_env_zero_action.py`: 7/7 pass.  These tests cover
  the G6 invariant (zero residual ⇒ `target_q == q_ref`) and the
  metric pipeline shape integrity.  They do NOT directly test the
  three new reward shapes (`lin_vel_z`, `ang_vel_xy`,
  `ref_contact_match` boolean) — those are validated only by code
  review and probe traces.  An honest "metric pipeline correct" claim
  is limited to G6 invariance.

### What's still not directly verified

- `reward/ref_contact_match` boolean form at training time — values
  will be visible in smoke6 wandb logs.
- `reward/lin_vel_z` and `reward/ang_vel_xy` values under PPO control
  — visible in smoke6 wandb logs.
- The probe's lin_vel_z alive bar at training-time operating point
  (open-loop fails the bar; PPO conditions unknown until smoke6 runs).

## [v0.20.1-smoke5] - 2026-04-22: body-frame foothold reward dead — null result, design hypothesis falsified

### Run

- Run: `training/wandb/offline-run-20260422_230524-6fh9ufds`
- Checkpoints: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260422_230527-6fh9ufds`
- Recommended checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260422_230527-6fh9ufds/checkpoint_150_19660800.pkl`
- Previous comparable run: `training/wandb/offline-run-20260422_083150-5ooizuts` (smoke4)

### Result

- **Walking verdict:** smoke5 is a null result.  The new body-frame
  foothold reward contributed exactly zero gradient throughout
  training (`reward/ref_foot_pos_body = 0.0` from iter 1 to iter 150).
  PPO trained on the same effective reward family as smoke4.
- **G4:** 5/6 hard gates pass.  `Evaluate/forward_velocity=0.131`,
  `Evaluate/mean_episode_length=500`,
  `Evaluate/cmd_vs_achieved_forward=0.039`,
  `tracking/cmd_vs_achieved_forward=0.068`,
  `env/forward_velocity=0.138` all pass.
  `tracking/step_length_touchdown_event_m=0.0182` still fails the
  `>= 0.030` stride gate (mild regression vs smoke4's 0.0204 within
  run-to-run variance — see below).
- **G5:** pass.  `forward_velocity_cmd_ratio=0.92` centered, residual
  p50s 0.094-0.109 rad, well under the 0.20 cap.
- **G7 baseline-beat:** pass.

### Why the new term was dead — calibration error

The pre-flight per-frame probe at vx=0.15 measured per-foot
body-frame errors of 7-15 cm (open-loop, 50-step ramp, body falls
at step 77).  Calibrated `alpha=100` to keep raw reward ≥ 0.05
through that error range.

Actual training-time errors were 5-7x larger:

| Signal | Probe (calibration) | Train iter 1 | Train iter 150 |
|---|---:|---:|---:|
| `ref/foot_pos_body_err_l_m` | ~0.07-0.12 m | 0.157 m | 0.755 m |
| `ref/foot_pos_body_err_r_m` | ~0.08-0.11 m | 0.161 m | 0.728 m |
| Raw reward at α=100 | 0.05-0.30 (alive) | 0.006 (dead) | ≈0 (dead) |

The probe under-predicted because:
- Probe ran open-loop with a 50-step pose-blend ramp (lets body
  settle into first q_ref).  Training has no ramp.
- Probe ran for 100 ctrl steps and the body fell at step 77.
  Training averages over 500-step horizons across 1024 envs.
- Probe error stayed small because there were ~77 alive steps
  before collapse; training error grows over 500 alive steps where
  actual body lags or drifts from the prior's pelvis trajectory.

### Why lowering alpha won't fix it

Smoke3 already tested the world-frame variant of this metric at
alpha=2 (the same effective metric since body-frame ≈ world-frame
at WR's small body posture ~5°).  Result: foothold reward came
alive (0.0002 → 0.001) but stride got *worse* (0.0203 → 0.0178),
forward velocity worsened, contact-phase match worsened.

Diagnosis: the metric is **phase-sensitive**.  Algebraically,

```
err = (actual_foot - actual_pelvis) - (prior_foot - prior_pelvis)
    = (actual_foot - prior_foot) - (actual_pelvis - prior_pelvis)
```

When the actual body's gait phase differs from the prior's gait
phase, the prior's foot is at a foothold position the actual foot
should not match (e.g., prior in stance while actual in swing).
The gradient pulls toward the wrong target regardless of how good
the foothold geometry is.  Body-frame rotation only differentiates
from world-frame at large body tilt; at upright posture (~5°), the
two metrics are equivalent.  We essentially re-introduced the
deleted world-frame term with a different alpha, and the same
fundamental issue blocks it.

**Verdict (combined smoke3 + smoke5 evidence):** the design
hypothesis "alive body-frame foothold imitation in any reasonable
α range fixes the stride gap" is falsified.  Smoke5 alone only
showed that α=100 was too tight for the actual training-time
error scale (the term was dead) — it did not, on its own, falsify
the metric idea.  But smoke3 had already tested the same metric
shape (world-frame ≈ body-frame at upright posture) at α=2 (alive
at training-time error scale) and stride got *worse* (0.0203 →
0.0178).  Together: at high α the term is dead; at low α the term
is alive but pulls in the wrong direction at phase mismatch.  The
combined evidence rules out kernel re-calibration as a fix.  Drop
the term, do not retune alpha.

### Stride "regression" (0.0204 → 0.0182) is run-to-run variance

With the new term contributing zero gradient, smoke5's effective
reward family was identical to smoke4's.  The 10% stride
difference between two single-seed PPO runs at the same effective
config is normal noise.  Both are well below the 0.030 gate;
neither is a meaningful improvement over the other.

### Wandb registration gap (fixed in 22a1b49)

Found a logging bug in flight: `experiment_tracking.py` and
`training_loop.py` had hardcoded allowlists that didn't include
the new v0.20.2 metric keys.  PPO was correctly training on
`reward/ref_foot_pos_body` via `_aggregate_reward` (it's in
`reward/total`), but the per-term diagnostic was silent in wandb.
The fix landed in 22a1b49 between smoke5 and smoke6; smoke5's
per-term values are visible in 6fh9ufds (the rerun) but not in
the original 5i55cp2u (killed early).

### Next version

- Drop `ref_foot_pos_body` (set weight to 0 — keep the diagnostic
  err logging so we can still see the foothold geometry without
  paying for misdirected gradient).
- Re-enable `ref_contact_match` at weight 0.5.  The Gaussian
  contact-phase reward (σ=0.5) provides a discrete event-anchor
  for gait timing — this is what TB uses (boolean at weight 1.0;
  WR's Gaussian at 0.5 is the conservative equivalent).
- Add the two missing TB continuous phase signals:
  - `lin_vel_z` at weight 1.0 (TB walk.gin) — vertical body
    velocity tracking.  Reference computed by finite-diff of
    `pelvis_pos[2]` (Appendix A.3 G2).
  - `ang_vel_xy` at weight 2.0 (TB walk.gin) — body roll/pitch
    rate.  Reference is zero (yaw-stationary prior with no
    pitch/roll oscillation).
- The combination addresses phase sync from both ends: discrete
  contact anchors at phase transitions + continuous body-velocity
  signals throughout each phase.  Per Appendix A.3 these are
  already-deferred TB-aligned terms; smoke6 promotes them.

## [v0.20.1-smoke5-prep1] - 2026-04-22: body-frame Euclidean foothold imitation + alpha calibration

### Context

`v0.20.1-smoke4` confirmed the documented v0.20.x diagnosis: there is no
explicit per-foot geometric pull in the optimized reward, so the stride
gate stays unmoved even when body-position tracking improves.  Smoke3
already documented the v0.20.2 fix — rewrite the deleted world-frame
summed-squares foot-position term as a body-frame Euclidean L2 against
the prior's foot-vs-pelvis target.  Pre-flight per-frame probe confirmed
the prior delivers steady-state per-touchdown stride 0.045-0.053 m at
vx=0.15 (well above the 0.030 m gate), so the new term has stride to
track to.

### Changes

1. **`training/envs/wildrobot_env.py`**
   - Added `r_ref_foot_pos_body` reward term in `_compute_reward_terms`:
     - Inverse-rotates the actual foot-vs-pelvis world offset into the
       torso frame using `inv(root_quat)` and L2's against the prior's
       (already body-frame) foot-vs-pelvis target.
     - DeepMimic-style kernel
       `exp(-α · (||err_l||² + ||err_r||²))` matching ToddlerBot's
       `_reward_torso_pos_xy` shape applied per foot.
     - The deleted world-frame `r_feet_track_raw` is kept as a
       diagnostic-only logged signal so smoke generation comparisons
       stay valid; no longer in the optimized reward.
   - Wired `ref_foot_pos_body` through `_aggregate_reward` and the
     terminal metrics (`reward/ref_foot_pos_body`,
     `ref/foot_pos_body_err_l_m`, `ref/foot_pos_body_err_r_m`).
   - Refactored the world->body inverse quaternion build to a single
     site shared between `ref_foot_pos_body` and the existing
     `feet_distance` block (XLA would fold the duplicate; one source of
     truth is easier to audit).
2. **`training/configs/training_runtime_config.py`**
   - Added `RewardWeightsConfig.ref_foot_pos_body: float = 0.0`
     (default inert for non-smoke configs).
   - Added `RewardWeightsConfig.ref_foot_pos_body_alpha: float = 200.0`
     (TB pos default; smoke YAML may override).
3. **`training/configs/ppo_walking_v0201_smoke.yaml`**
   - `reward_weights.ref_foot_pos_body: 2.0` (mirrors `torso_pos_xy`
     so the imitation block stays balanced).
   - `reward_weights.ref_foot_pos_body_alpha: 100.0` (post-calibration
     setting, see probe trace below).
4. **`tools/v0200c_per_frame_probe.py`**
   - Added `--foot-alpha` CLI arg and per-step body-frame foot-vs-pelvis
     error fields (`foot_pos_body_err_l_m`, `foot_pos_body_err_r_m`,
     `foot_pos_body_raw`).
   - Added a calibration print block matching the existing
     `--torso-alpha` block (steps 10/30/50/70 + min raw over 0..50).
5. **`training/docs/walking_training.md`**
   - Appendix A.4 now documents `ref_foot_pos_body` α with the smoke
     override (TB 200 → WR smoke 100) and the alive-signal trace.
   - v0.20.1 § "primary terms" list now includes `ref/foot_pos_body`
     with smoke5 introduction and calibration note.

### Probe (alive-signal calibration)

- Command:
  `UV_CACHE_DIR=/tmp/uv-cache uv run python tools/v0200c_per_frame_probe.py --vx 0.15 --horizon 100 --foot-alpha <α>`
- Per-foot body-frame errors at vx=0.15 (open-loop, zero residual):
  `err_l ≈ 0.07-0.18 m`, `err_r ≈ 0.08-0.18 m` across steps 10/30/50/70.
- Raw reward calibration:

  | α | step10 | step30 | step50 | step70 | min over 0..50 | verdict |
  |---:|---:|---:|---:|---:|---:|---|
  | 200 | 0.057 | 0.034 | 0.006 | 0.000001 | 0.002 | dies in accel ramp; below 0.05 alive bar |
  | 100 | 0.239 | 0.185 | 0.075 | 0.001 | 0.046 | one ~3-step transient dip to ~0.046 around step 44; usable |
  | 80  | 0.318 | 0.260 | 0.126 | 0.005 | 0.085 | comfortable margin, larger TB divergence |

- **Decision: α=100.**  Smallest divergence from TB's pos kernel that
  keeps cm-scale selection pressure alive through the prior's accel
  ramp.  Mirrors the `torso_pos_xy_alpha=90` calibration philosophy
  ("smallest TB divergence keeping the gradient alive").  The single
  ~3-step dip below 0.05 around step 44 is at the back end of the
  ramp where PPO can already begin to apply residual corrections.

### Verification

- `tests/test_v0201_env_zero_action.py`: 7/7 pass (G6 invariant
  intact — zero residual ⇒ `target_q == q_ref`; per-step metric
  trajectories unchanged at iter 0).

### Rationale

- Smoke1/2/3/4 confirmed the world-axis projection of foot-vs-pelvis
  offset is dominated by body translation; the gradient at the
  operating point is uncorrelated with stride geometry, and lowering
  alpha (smoke3 alpha=2 probe) only makes the term less dead without
  moving stride.
- Body-frame Euclidean removes the world-frame translation bias.
  Realistic per-foot magnitudes are 7-15 cm, so a Gaussian kernel of
  the right width gives strong selection pressure on each centimeter
  of stride correction.
- The pre-flight per-frame probe at vx=0.15 confirmed the prior
  delivers steady-state per-touchdown stride 0.045-0.053 m by
  half-cycle 3, so the new term has a real target to track to.
- This is the only term in the v0.20.1 reward family with an explicit
  per-foot geometric pull; smoke4 ruled out the alternative
  hypothesis that pelvis tracking alone would suffice.

### Expected signal from the smoke5 run

- `tracking/step_length_touchdown_event_m` should rise materially toward
  the `>= 0.030 m` gate.
- `ref/foot_pos_body_err_l_m` and `ref/foot_pos_body_err_r_m` should
  shrink over training; `reward/ref_foot_pos_body` should grow.
- `Evaluate/forward_velocity`, `tracking/cmd_vs_achieved_forward`, and
  the G5 residual metrics should remain near smoke4 levels (no
  expected regression on already-passing gates).

### Falsification — what would tell us this design is wrong

- `ref_foot_pos_body_raw` is alive at iter-0 but stride still parks at
  ~0.020 m → the bottleneck is balance-bound, not stride-bound; return
  to prior surgery (M1 fail-mode tree, branch 1).
- Stride moves but `feet_air_time` reward grows in a way that pins
  cadence at micro-shuffles → the cadence-invariance of TB's
  `air_time × first_contact` formula dominates; revisit weight or
  consider a threshold-subtraction variant.
- Per-foot body-frame error stays large (~0.3 m) at iter-0 of the
  actual smoke despite probe agreement → indicates an env-vs-probe
  frame mismatch; investigate before extending compute.

## [v0.20.1-smoke4] - 2026-04-22: torso_pos_xy parity verified alive, stride gate still fails

### Run

- Run: `training/wandb/offline-run-20260422_083150-5ooizuts`
- Checkpoints: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260422_083153-5ooizuts`
- Recommended checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260422_083153-5ooizuts/checkpoint_150_19660800.pkl`
- Previous comparable run: `training/wandb/offline-run-20260421_220826-uzpl0pqr` (smoke3)

### Result

- **Walking verdict:** smoke4 is not a promotion candidate.  The
  `torso_pos_xy` α=90 calibration delivered exactly what its narrow probe
  goal predicted — the term is alive, body-position tracking improves
  modestly — but the stride gate is unchanged.  The shuffle exploit
  pattern from smoke1/2/3 persists.
- **G4:** 5 of 6 hard gates pass at the best checkpoint.
  `Evaluate/forward_velocity=0.135`, `Evaluate/mean_episode_length=500.0`,
  `Evaluate/cmd_vs_achieved_forward=0.034`,
  `tracking/cmd_vs_achieved_forward=0.067`,
  `env/forward_velocity=0.145` all pass.
  `tracking/step_length_touchdown_event_m=0.0195` remains well below the
  `>= 0.030 m` stride gate.
- **G5:** pass.  `tracking/forward_velocity_cmd_ratio=0.942` is centered
  in band, residual p50s remain small (`hip_pitch_L/R = 0.099 / 0.098`,
  `knee_L/R = 0.098 / 0.095 rad`).  Not an action-authority leak.
- **G7 baseline-beat sanity:** pass comfortably.  Far better than
  bare-`q_ref` replay on forward velocity, episode length, and pitch
  failure rate.

### What the α=90 calibration delivered (smoke3 → smoke4)

| Signal | smoke3 | smoke4 | Δ | Reading |
|---|---:|---:|---:|---|
| `env/forward_velocity` | 0.132 | 0.145 | +0.014 | better translation |
| `tracking/cmd_vs_achieved_forward` | 0.074 | 0.067 | -0.007 | tighter cmd tracking |
| `tracking/forward_velocity_cmd_ratio` | 0.879 | 0.969 | +0.090 | now centered |
| `ref/contact_phase_match` | 0.612 | 0.652 | +0.040 | mild gait-schedule recovery |
| **`tracking/step_length_touchdown_event_m`** | **0.0178** | **0.0204** | +0.0025 | **still fails 0.030 gate** |
| `ref/feet_pos_err_l2` (debug, not optimized) | 1.007 | 1.013 | flat | dead legacy diagnostic |
| `reward/torso_pos_xy` (new, w=2.0, α=90) | n/a | 0.0164 | — | **alive, as designed** |
| `ref/torso_pos_xy_err_m` | n/a | 0.124 m | — | within healthy body-tracking band |

Pelvis tracking is now a live gradient and is being matched.  Removing
the dead world-frame feet term and adding body-frame pelvis tracking did
not move stride.  This confirms pelvis tracking is independent of the
stride bottleneck — improving body-pos matching by itself does not produce
a longer step.

### Per-foot stride symmetry

`step_length_touchdown_event_m` left/right ≈ `0.0196 / 0.0210 m`.
Bilateral short-step pattern, not one-sided.

### Diagnosis

The smoke3 entry called this exactly: there is no longitudinal
stride-length lever in the current optimized reward.  Smoke4 confirmed
it from the other direction by upgrading body-position tracking and
seeing zero stride change.

- `feet_air_time` (w=500): pays per touchdown for `(T_air − T_threshold)`.
  Per-event air time ~0.089 s satisfies the current threshold; this term
  rewards stepping cadence, not stride distance.
- `feet_clearance`, `feet_distance` (lateral), `cmd_forward_velocity_track`,
  `torso_pos_xy`: none directly penalize a short forward step relative
  to the prior's foothold.
- The deleted world-frame foot term was structurally wrong (translation
  bias dominates, gradient at operating point uncorrelated with stride
  geometry).  Body-frame Euclidean is the documented v0.20.2 fix.

### Pre-smoke probe for v0.20.2 (assumption check)

Before committing the v0.20.2 body-frame foothold work, re-ran
`tools/v0200c_per_frame_probe.py --vx 0.15 --horizon 100` to confirm the
prior actually delivers a stride above the gate.

Per-touchdown pelvis advance (steady-state foothold geometry the prior
asks PPO to track):

| Half-cycle | Steps | `pCOMx` advance | vs 0.030 m gate |
|---|---|---:|---|
| 1 (ramp) | 0 → 20 | +0.0211 m | below (ramp) |
| 2 (ramp) | 20 → 32 | +0.0283 m | below (ramp) |
| 3 | 32 → 48 | +0.0476 m | **1.6×** |
| 4 | 48 → 64 | +0.0528 m | **1.8×** |

The prior reaches steady-state stride `~0.045–0.053 m` by ~1.0 s of
replay.  Assumption holds: the prior has the stride to track to.

The probe also surfaced two second-order observations:

- The per-rollout mean stride is unfair to the prior's intrinsic
  acceleration ramp; the first ~50 ctrl steps are below-gate by design.
  v0.20.2 should add a windowed stride metric (last-N touchdowns).
- First lever sign flip at step 2 (`pLever=+0.0003`,
  `aLever=−0.0031`).  The body resists the prior's forward push almost
  immediately in open-loop replay; PPO has to overcome this every step.
  Adds weight to "policy is balance-bound, not stride-bound."

### Next version

- Implement the documented v0.20.2 body-frame Euclidean foothold
  imitation term (replaces the deleted world-frame summed-squares
  term).  Calibrate α with a zero-residual probe so the iter-0 raw
  reward sits in the alive-gradient band (target raw 0.05–0.30 at the
  prior's typical body-frame foot error).
- Add windowed stride metric `tracking/step_length_touchdown_window_m`
  (mean over the last N touchdowns, e.g. N=8) so the gate measures
  steady-state stride instead of including the prior's intrinsic ramp.
- Hold everything else fixed (PPO hyperparameters, residual bound,
  `torso_pos_xy_alpha=90`, gated `ref_contact_match`) so the next run
  cleanly answers whether the foothold metric in the right frame is
  what unlocks the stride gate.
- If the foothold term is alive but stride still doesn't move, revisit
  `feet_air_time_threshold` — at the current value, micro-shuffles pay
  out and a 500-weight term will fight the new foothold pull.

## [v0.20.1-smoke4-prep1] - 2026-04-22: torso_pos_xy alpha calibration from per-frame probe

### Context

`v0.20.1-smoke3` confirmed the legacy world-frame feet-position reward was
not the right optimized signal for stride geometry, so the reward objective
was switched to ToddlerBot-parity `torso_pos_xy` tracking.  Before launching
the next smoke, we ran a zero-residual pre-smoke probe to verify whether
ToddlerBot's default `alpha=200` is alive at WildRobot's open-loop drift
scale.

### Probe

- Command:
  `UV_CACHE_DIR=/tmp/uv-cache uv run python tools/v0200c_per_frame_probe.py --vx 0.15 --horizon 100`
- Baseline (`alpha=200`, raw `exp(-alpha * err^2)`):
  - step 10: `err=0.0118 m`, `raw=0.972`
  - step 30: `err=0.0945 m`, `raw=0.168`
  - step 50: `err=0.1819 m`, `raw=0.00134`
  - step 70: `err=0.3641 m`, `raw≈0`
  - min raw over steps `0..50`: `0.00134` (fails alive-signal gate)
- Calibrated check (`alpha=90`):
  - step 10: `raw=0.987`
  - step 30: `raw=0.448`
  - step 50: `raw=0.05097`
  - min raw over steps `0..50`: `0.05097` (passes alive-signal gate)

### Changes

1. **`training/configs/ppo_walking_v0201_smoke.yaml`**
   - `reward_weights.torso_pos_xy_alpha: 200.0 -> 90.0`
2. **`tools/v0200c_per_frame_probe.py`**
   - added direct `torso_pos_xy` diagnostics:
     `torso_pos_xy_err_m`, `torso_pos_xy_raw`, `--torso-alpha`,
     and printed step checkpoints (`10/30/50/70`) plus min over `0..50`.
3. **`training/docs/walking_training.md`**
   - Appendix A.4 now documents the temporary smoke override
     (`pos_tracking_sigma`: TB `200`, WR smoke override `90`) with rationale.

### Rationale

- Keep the reward shape and weight aligned with ToddlerBot, but calibrate
  the kernel width to the error regime WildRobot actually visits in open-loop.
- `alpha=90` is the smallest divergence from `200` that keeps
  `reward/torso_pos_xy` alive through the first ~50 control steps at
  `vx=0.15`, which is the region we need PPO to learn from before the prior
  drift dominates.
- This is a smoke-stage calibration only; restore toward `200` after prior
  drift cleanup in `v0.20.2`.

## [v0.20.1-smoke3] - 2026-04-21: alpha-2 foothold-gradient probe result

### Run

- Run: `training/wandb/offline-run-20260421_220826-uzpl0pqr`
- Checkpoints: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260421_220829-uzpl0pqr`
- Recommended checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260421_220829-uzpl0pqr/checkpoint_150_19660800.pkl`
- Previous comparable run: `training/wandb/offline-run-20260421_063530-ehkn02ye`

### Result

- **Walking verdict:** smoke3 is still not a promotion candidate.  The
  diagnostic run clears velocity / survival / G5 again, but it does not
  improve the failed stride-geometry gate.
- **G4:** 5 of 6 hard gates pass at iter 150.
  `env/forward_velocity=0.132`,
  `Evaluate/forward_velocity=0.133`,
  `Evaluate/mean_episode_length=500.0`,
  `Evaluate/cmd_vs_achieved_forward=0.044`, and
  `tracking/cmd_vs_achieved_forward=0.0737` all satisfy the documented
  floors, but `tracking/step_length_touchdown_event_m=0.0178` remains well
  below the `>= 0.03 m` stride gate.
- **G5:** pass.  `tracking/forward_velocity_cmd_ratio=0.879` stays in-band,
  and residual means remain small
  (`hip_pitch_L/R = 0.098 / 0.105 rad`,
  `knee_L/R = 0.098 / 0.096 rad`), so smoke3 is not an
  action-authority leak.
- **G7 baseline-beat sanity:** pass comfortably.  The run is far better than
  bare-`q_ref` replay on forward velocity, episode length, and pitch
  failure rate.

### Diagnostic outcome

- **The alpha-2 probe succeeded only in making the foothold term less dead.**
  Relative to smoke2 best-vs-best:
  - `ref/feet_pos_err_l2`: `1.062 -> 1.007` (better)
  - `reward/ref_feet_pos_track`: `0.000204 -> 0.000976` (about 4.8x larger)
- **But the gait geometry did not improve.**  Relative to smoke2:
  - `tracking/step_length_touchdown_event_m`: `0.0203 -> 0.0178` (worse)
  - `env/forward_velocity`: `0.148 -> 0.132` (worse)
  - `tracking/cmd_vs_achieved_forward`: `0.069 -> 0.074` (worse)
  - `ref/contact_phase_match`: `0.665 -> 0.612` (worse)
- **Per-foot stride stays short on both sides.**  At iter 150 the
  touchdown-normalized stride is
  `0.0174 m` left and `0.0222 m` right, versus smoke2's
  `0.0207 m` left and `0.0264 m` right.  This is still a bilateral
  short-step solution, not one-sided stepping.
- **The run never cleared the stride gate at any point.**  The best late-run
  stride was around `0.0208 m` at iter 100, then it regressed back down by
  iter 150.

### Interpretation

- The diagnostic question from `v0.20.1-smoke2-diagnostic1` now has a clear
  answer: **the problem is not just a dead alpha.**
- Lowering `ref_feet_pos_alpha` did move the foothold-tracking metric, so the
  reward is no longer completely saturated.  However, because stride length,
  speed, and contact-phase alignment all worsened, the current
  **world-frame summed-squares foot-position metric is not pulling the gait
  toward the prior's actual foothold geometry.**
- In other words: smoke3 says the next bottleneck is **metric geometry**, not
  PPO optimization and not residual authority.

This matches the project design direction:
- ToddlerBot's locomotion stack keeps the prior / reference geometry in the
  frame where it encodes the intended motion pattern rather than asking PPO
  to interpret a drifting world-frame target.  Reference:
  <https://github.com/hshi74/toddlerbot>
- DeepMimic-style imitation rewards only help when the imitation term is both
  alive **and** aligned with the target motion geometry.  Reference:
  <https://arxiv.org/abs/1804.02717>
- Residual RL works when the residual corrects a meaningful nominal behavior;
  if the nominal-tracking metric is wrong, making it "less dead" does not fix
  the division of labor.  Reference:
  <https://arxiv.org/abs/1812.03201>

### Next version

- Do **not** continue sweeping `ref_feet_pos_alpha` on the current metric.
- Promote the already-documented `v0.20.2` cleanup ahead of any further smoke:
  rewrite the WildRobot-specific foot-position imitation term from the
  current world-frame summed-squares form to a **body-frame Euclidean**
  metric, then retune alpha on that new metric.
- Keep the rest of the smoke contract fixed when making that change so the
  next run cleanly answers the remaining question:
  whether the stride gate can finally move once the foothold metric is in
  the right geometry.

## [v0.20.1-smoke2-diagnostic1] - 2026-04-21: lower foot-pos alpha for live foothold-gradient probe

### Context

`offline-run-20260421_063530-ehkn02ye` cleared the velocity, survival,
and G5 residual-authority gates but still failed the stride gate:
`tracking/step_length_touchdown_event_m = 0.0203 < 0.03`.  The dominant
diagnostic was a dead foothold-imitation term:
`ref/feet_pos_err_l2` grew to `1.062` while
`reward/ref_feet_pos_track` stayed near zero.  Since the current reward
uses `exp(-alpha * feet_err_l2_sq)` on a world-frame summed-squares
metric, `ref_feet_pos_alpha = 30` is effectively zero-gradient at the
error scale smoke2 actually visits.

### Change

1. **`training/configs/ppo_walking_v0201_smoke.yaml`**
   - `reward_weights.ref_feet_pos_alpha: 30.0 -> 2.0`

### Rationale

- At the smoke2 best checkpoint, the logged
  `ref/feet_pos_err_l2 ≈ 1.06 m` corresponds to
  `feet_err_l2_sq ≈ 1.12 m²` in the reward exponent.  With `alpha = 30`,
  the raw term is approximately `exp(-30 * 1.12) ≈ 2e-15`, which is dead.
- With `alpha = 2`, the same error scale yields
  `exp(-2 * 1.12) ≈ 0.10`, which is still selective but no longer
  numerically irrelevant.
- This keeps the intervention surgical: no PPO change, no residual-bound
  change, no reward-family expansion, and no metric rewrite ahead of the
  diagnostic rerun.

### Expected signal from the next smoke

- `tracking/step_length_touchdown_event_m` should rise materially toward
  the `>= 0.03 m` gate.
- `ref/feet_pos_err_l2` should stop drifting upward.
- `env/forward_velocity`, `tracking/cmd_vs_achieved_forward`, and the
  G5 residual metrics should remain near smoke2 levels.

### What this does NOT change

- The foot-position metric is still the same world-frame summed-squares
  signal.  The planned `v0.20.2` body-frame Euclidean cleanup remains
  deferred until this probe tells us whether the failure is primarily
  dead-gradient or metric-geometry.
- `version`, tags, PPO hyperparameters, and the bounded residual contract
  are unchanged so the next run remains directly comparable to smoke1/2.

## [v0.20.1-smoke2-prep-fix2] - 2026-04-21: analyzer regression checklist + 0.0-safe fallbacks

### Context

Reviewing `offline-run-20260421_063530-ehkn02ye` against the previous
`v0.20.1` smoke exposed a workflow gap: `analyze_offline_run.py` could
score a single run, but it did not automatically answer the practical
"better, worse, or lateral move vs the last comparable training?" question.
The same pass also found a few remaining `or`-based metric fallbacks that
could swallow legitimate `0.0` values in summaries.

### Changes

1. **`skills/wildrobot-training-analyze/scripts/analyze_offline_run.py`**
   now auto-detects the most recent comparable prior offline run
   (same `version`, same `version_name` when present, same tags when
   present, same walking/non-walking mode) and emits a regression
   checklist against that run's best common-metric walking row.

2. **Regression checklist scope** is intentionally limited to common
   contract metrics that remain comparable across smoke1/smoke2-style
   changes:
   - `env/forward_velocity`
   - `env/episode_length`
   - `tracking/cmd_vs_achieved_forward`
   - `tracking/step_length_touchdown_event_m`
   - `tracking/forward_velocity_cmd_ratio`
   - G5 hip/knee residual means
   - `ref/feet_pos_err_l2`
   - `ref/contact_phase_match`

3. **Explicit override added**:
   `analyze_offline_run.py --previous-run-dir <offline-run-...>`
   for cases where the auto-picked prior run is not the comparison the
   user wants.

4. **Accuracy fixes**:
   - `train success`, `train ep_len`, `tracking/max_torque`,
     `ppo/approx_kl`, and `ppo/clip_fraction` now use `_first_present`
     instead of Python `or`, so real `0.0` values are preserved.
   - `ref/feet_pos_err_l2` label corrected to match the env metric
     definition (`root-relative L2 m`, not `sum-sqr m²`).

5. **Skill docs updated** (`skills/wildrobot-training-analyze/SKILL.md`)
   so the default single-run workflow now documents the automatic
   previous-run regression check.

## [v0.20.1-smoke2] - 2026-04-21: single-command walking smoke2 result

### Run

- Run: `training/wandb/offline-run-20260421_063530-ehkn02ye`
- Checkpoints: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260421_063533-ehkn02ye`
- Recommended checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260421_063533-ehkn02ye/checkpoint_150_19660800.pkl`

### Result

- **Walking verdict:** smoke2 is not a promotion candidate.  It solves
  command-speed tracking and horizon survival, but the gait remains a
  bilateral short-step / shuffle solution rather than clean prior-tracking
  locomotion.
- **G4:** 5 of 6 hard gates pass at iter 150.  `env/forward_velocity=0.148`,
  `Evaluate/forward_velocity=0.155`,
  `Evaluate/mean_episode_length=500.0`,
  `Evaluate/cmd_vs_achieved_forward=0.046`, and
  `tracking/cmd_vs_achieved_forward=0.069` all clear the documented floors;
  `tracking/step_length_touchdown_event_m=0.0203` fails the `>= 0.03 m`
  stride gate.
- **G5:** pass.  `tracking/forward_velocity_cmd_ratio=0.984` stays in-band,
  and the residual means remain small
  (`hip_pitch_L/R = 0.089 / 0.099 rad`,
  `knee_L/R = 0.101 / 0.098 rad`), so this is not an
  action-authority leak / residual-driven exploit.
- **G7 baseline-beat sanity:** pass by a wide margin.  The run improves the
  bare-`q_ref` baseline from roughly
  `forward_velocity ≈ -0.09 m/s, episode_length ≈ 77, term_pitch_frac ≈ 1.0`
  to `Evaluate/forward_velocity=0.155`,
  `Evaluate/mean_episode_length=500.0`, and `term_pitch_frac=0.043`.
- **Stride structure:** both feet are short, not one-sided.  The per-foot
  diagnostics are MEAN-reduced touchdown carries, so the per-touchdown step
  lengths at iter 150 are
  `step_length_left_event_m / touchdown_rate_left = 0.0207 m` and
  `step_length_right_event_m / touchdown_rate_right = 0.0264 m`; both remain
  below the `0.03 m` smoke gate.
- **Dominant failure mode:** the foothold imitation term is effectively dead.
  `ref/feet_pos_err_l2` (root-relative L2 foot-position error in meters)
  grows from `0.222` at iter 1 to `1.062` at iter 150 while
  `reward/ref_feet_pos_track` stays near zero
  (`0.00118 -> 0.000204`).  The policy is therefore satisfying velocity and
  horizon mostly through short fast steps instead of tracking the prior's
  foothold geometry.
- **Secondary diagnostic:** `ref/contact_phase_match` drifts from `0.773` to
  `0.665` as forward velocity improves.  Because `reward/ref_contact_match`
  is still gated at zero for the smoke, this is diagnostic only and not by
  itself a contract regression.
- **ToddlerBot alignment:** intact on the load-bearing rows.  The run uses
  the ToddlerBot-style `Evaluate/*` selector, strict `alive=-done`,
  `cmd_forward_velocity_track` at weight `5.0` and `alpha=200`,
  `PROPRIO_HISTORY_FRAMES=15`, `action_delay_steps=1`, and the bounded
  residual action contract.  The failure is in the WildRobot-specific
  foothold term staying too weak, not in a ToddlerBot-parity regression.

### Checkpoint choice

Iter 150 is the right checkpoint to inspect.  It is the run maximum on
`Evaluate/mean_reward=274.669`, reaches the full eval horizon, and already
shows the final failure mode clearly.  There is no earlier checkpoint that
passes the stride gate.

### Next version

- Treat smoke2 as a failed promotion smoke despite the good velocity numbers.
  Do **not** extend compute, relax G5, or increase residual authority.
- Restore a live foothold-imitation gradient before the next rerun.  The
  smallest justified change is to retune `ref_feet_pos_alpha` downward on the
  current root-relative metric using the observed error scale, then rerun the
  same single-command smoke and gate again on
  `tracking/step_length_touchdown_event_m >= 0.03 m`.
- If the root-relative foot metric still dies after retuning, take the
  already-documented `v0.20.2` cleanup: switch the WildRobot-specific
  foot-position reward to a body-frame Euclidean norm, then restore the
  ToddlerBot-aligned alpha scale.

## [v0.20.1-smoke2-prep-fix1] - 2026-04-21: analyzer + comparator on Evaluate/* namespace

### Context

``v0.20.1-smoke2-prep`` (commit ``df544a7``) retired the broken
``env/success_rate`` / ``env/velocity_error`` topline path on the
training side and added the ToddlerBot-style ``Evaluate/*`` block.
The post-smoke analysis tooling
(``skills/wildrobot-training-analyze/``) was not updated, so a
smoke2 run would still be auto-ranked on the now-permanently-zero
``env/success_rate`` and mis-classified by the walking verdict
(which averaged ``env/velocity_error`` ≡ 0).

### Changes

1. **``analyze_offline_run.py:_pick_eval_keys``** — prefers
   ``Evaluate/mean_reward`` + ``Evaluate/mean_episode_length`` for
   walking runs when present.  Both fields are higher-is-better so
   the existing ``_lexi_better`` / ``_find_best_row`` lex-sort works
   without further changes.  Falls through to legacy
   ``eval_clean/`` / ``eval/`` / ``env/`` order for older runs.

2. **``analyze_offline_run.py:_summarize_walking``** — switches the
   walking classifier to:
   - forward-velocity source: ``Evaluate/forward_velocity`` (eval
     rollout) when present, else ``env/forward_velocity``.
   - tracking-error source: ``Evaluate/cmd_vs_achieved_forward``
     (eval), ``tracking/cmd_vs_achieved_forward`` (train), then
     legacy ``env/velocity_error``.
   - synthetic ``success`` for v0.20.1 walking: derived from
     ``Evaluate/mean_episode_length / max_episode_steps``, capped
     at 0.5 if tracking is bad.
   - tightens the verdict gates from the standing-era 0.20 / 0.20
     thresholds to the G4 promotion-horizon floors (``fwd ≥ 0.075``,
     ``vel_err ≤ 0.075``).  The looser gates would mis-classify a
     vx=0.15 smoke as "needs review" even when tracking is good.
   - changelog table labels updated to reflect the actual key path
     ("Evaluate/forward_velocity (eval) or env/forward_velocity").

3. **``compare_offline_runs.py``** — same updates to ``_pick_keys``
   and ``_classify_walking`` for run-to-run comparison.

### Validation

- ``analyze_offline_run.py --run-dir <smoke1>`` runs cleanly.
  Smoke1 has no ``Evaluate/*`` keys → falls through to legacy
  path; verdict "needs review" reflects mean ``vel_err`` ≈ 0.124
  exceeding the new G4 0.075 floor (smoke1 was a soft pass).
- ``compare_offline_runs.py`` synthetic ``Evaluate/*`` row →
  ``_pick_keys`` returns ``("Evaluate", "Evaluate/mean_reward",
  "Evaluate/mean_episode_length")``.  Empty rows still degrade
  cleanly to the legacy ``train`` fallback.

### What this does NOT change

- The training pipeline (env, reward, PPO, eval rollout) is
  untouched — this is an offline-tooling-only change.
- Older smoke runs without ``Evaluate/*`` keys analyze identically
  to before (the new branch only activates when ``Evaluate/*`` is
  present).
- No new dependencies; no new fixture files.

---

## [v0.20.x-amp-removal] - 2026-04-20: Remove AMP / discriminator stack

### Context

The v0.20.x walking plan (``training/docs/walking_training.md``) is a
ZMP-shaped offline reference + DeepMimic-style imitation reward +
ToddlerBot-aligned bounded residual PPO.  AMP / Adversarial Motion
Priors are not part of that recipe — the smoke YAML had been carrying
``amp.enabled=false, amp_weight=0`` since v0.20.1, and ToddlerBot
itself doesn't use AMP.  Removing the dormant AMP wiring rather than
keeping it inert simplifies the trainer ahead of the smoke / sysid /
command-breadth work.

### Changes

1. **Active code path** (commit a38d36b):
   - Deleted ``training/amp/`` (8 modules: discriminator, replay /
     ref buffers, features, mirror, diagnostics)
   - Deleted ``training/configs/feature_config.py`` (AMP-only feature
     layout)
   - Deleted ``training/data/{gmr,gmr_to_physics_ref_data.py,
     debug_gmr_physics.py}`` and ``training/scripts/mocap_retarget.py``
     — AMP-only data tooling and 14 reference motion files
   - Stripped AMP from ``training/configs/{training_runtime_config,
     training_config}.py`` (drop AMPConfig, AMPFeatureConfig,
     AMPTargetsConfig, DiscriminatorTrainingConfig,
     DiscriminatorNetworkConfig, ``amp_weight`` /
     ``amp_reward_clip``)
   - Stripped AMP from ``training/core/{training_loop,rollout,
     checkpoint,experiment_tracking,__init__}.py`` (lax.cond AMP
     branch, ``TrainingState`` disc_params / feature_mean /
     feature_var, ``IterationMetrics`` disc / amp fields,
     ``train_discriminator_scan``, ``compute_normalization_stats``,
     ``compute_reward_total``, ``ref_motion_data`` train() param,
     AMP resume validation, AMP W&B fields, AMP rollout fields and
     ``collect_amp_features``)
   - Stripped AMP from ``training/algos/ppo/ppo_core.py`` (docstring
     mentions only)
   - Stripped AMP from ``training/{train,envs/env_info}.py``
     (drop ``--amp-weight`` / ``--amp-data`` / ``--disc-lr`` CLI,
     AMP ref-data loading, "amp+ppo" mode strings, stale env_info
     comment)

2. **Configs** (commit 305e497):
   - ``ppo_walking_v0201_smoke.yaml`` — drop ``amp:``,
     ``reward.amp_weight``, ``quick_verify.amp``
   - ``ppo_standing.yaml``, ``ppo_standing_v0170.yaml``,
     ``ppo_standing_push.yaml``, ``ppo_standing_push_v0139_probe20*.yaml``
     — drop full ``amp:`` block (AMPConfig + DiscriminatorTrainingConfig
     + AMPFeatureConfig + AMPTargetsConfig payloads), drop
     ``networks.discriminator``, drop ``reward.amp_weight`` /
     ``amp_reward_clip``
   - ``ppo_standing_v0171.yaml``, ``ppo_standing_v0173a*.yaml``,
     ``ppo_standing_v0174t*.yaml``, ``ppo_standing_push_v0140*.yaml``,
     ``mpc_standing_v0173.yaml`` — drop the inert ``amp:
     {enabled: false}`` block

3. **Tests** (commit 305e497):
   - ``test_trainer_invariance.py`` — drop reward equivalence,
     discriminator-not-called, and AMP trajectory shape tests; keep
     GAE / PPO loss determinism and trajectory PPO-field test
   - ``test_config_contract.py`` — drop ``hasattr(config, "amp")`` check
   - ``test_state_mapping.py`` — drop "AMP features" docstring mention
   - ``test_plan.md`` — drop Layer 7 (AMP integration tests) section
     and AMP mentions from strategy summary, traceability matrix, etc.

4. **Asset metadata** (commit a55f60e):
   - ``assets/v2/mujoco_robot_config.json`` — drop
     ``amp_feature_breakdown`` block
   - ``assets/robot_config.py`` — drop ``amp_feature_dim`` /
     ``amp_feature_breakdown`` from RobotConfig
   - ``assets/post_process.py`` — stop emitting ``amp_feature_breakdown``
     during JSON regeneration
   - ``assets/v2/sensors.xml`` — drop "AMP discriminator sees FD"
     comment
   - ``scripts/validate_training_setup.py`` — comment cleanup

5. **Docs** (commit a55f60e):
   - Deleted: ``training/docs/{AMP_DATA_FIRST_PLAN,
     AMP_FEATURE_PARITY_DESIGN,amp_brax_solution,physics-based-amp,
     physics_ref_parity_report,smplx_skeleton,
     reference_data_generation,TRAINING_README,
     TRAINING_SYSTEM_DESIGN,phase3_rl_training_plan}.md``
   - Updated: ``README.md`` (drop deleted directories + dead links,
     redirect Documentation links to walking_training.md +
     reference_design.md + test_plan.md, rewrite Training section
     around the v0.20.1 smoke YAML); ``training/cal/{cal.md,cal.py,
     specs.py,types.py}`` (strip "AMP parity" / "AMP discriminator" /
     "AMP features" from docstrings; cal functions themselves stay);
     ``training/docs/ppo_training_stability_improvements.md``
     (drop "(no AMP)" scope qualifier)

### What stayed

- Historical AMP entries in this CHANGELOG below — they describe past
  work and shouldn't be edited
- ``training/docs/{v0201_env_rewrite_audit,v0201_env_wiring}.md`` and
  ``training/docs/archive/learn_first_plan.md`` — historical audit /
  archive docs that mention AMP only in describing past refactor
  decisions
- ``mujoco-brax/amp/`` — separate prototype directory, not on the
  v0.20.x active path; left for the user to triage separately
- ``runtime/bundles/`` historical deployment artifacts — never edited

### Rationale

- ``walking_training.md`` § "Reward Design" + § "Decision Summary":
  the v0.20.x reward family is "imitation-dominant hybrid reward"
  with explicit Gaussian-kernel ``ref/*`` tracking; the prior owns
  gait semantics; PPO is bounded residual.  AMP's adversarial
  mocap-distribution matching has no role in that contract.
- ``CLAUDE.md`` § "Design": "current design and implementations
  follows ToddlerBot ... if we have our own design or implementation,
  we should have explicit rationale."  ToddlerBot has no AMP code in
  ``toddlerbot/locomotion/`` — keeping a dormant AMP stack would have
  been a WildRobot-specific deviation without active rationale.

---

## [v0.20.1-smoke2-prep] - 2026-04-20: ToddlerBot eval-alignment + smoke1 instrumentation gaps

### Context

Smoke1 (run `offline-run-20260419_200601-zgs2m3gp`) was a soft pass:
3 of 4 measurable G4 gates met (forward_velocity, cmd_vs_achieved,
episode_length); step_length_touchdown_event_m saturated below the
≥ 0.030 m gate at ~0.022 m; ``eval_walk/success_rate`` was unmeasured
(no eval rollout); the five new ToddlerBot shaping rewards
(``alive``, ``feet_air_time``, ``feet_clearance``, ``feet_distance``,
``torso_pitch_soft``, ``torso_roll_soft``) were not exposed in W&B
even though they were ON in the YAML; ``ref/feet_pos_err_l2`` grew
0.22 → 1.02 m because at α=200 the iter-1 baseline reward was
exp(-200·0.048) ≈ 6e-5 (no gradient).

This changeset closes those four instrumentation / contract gaps
before launching smoke2.  No re-architecture; the smoke contract,
prior, and policy network are unchanged.

### Changes

1. **Reward-term logging completed** (`metrics_registry.py`,
   `experiment_tracking.py`).  Added ``reward/alive`` to
   ``METRIC_SPECS`` (was in ``REWARD_TERM_KEYS`` filter but absent
   from the registry → silently dropped).  Added ``reward/total`` to
   ``REWARD_TERM_KEYS`` (was in registry but absent from filter →
   not surfaced to wandb).  All five v0.20.1 shaping terms were
   already in both lists from v0.20.1-tb-align — no additional spec
   work needed there.

2. **ToddlerBot-style `Evaluate/*` eval block**
   (`training_loop.py`, `experiment_tracking.py`, smoke YAML).  The
   deterministic eval rollout (already running for the standing-era
   ``eval_clean/*`` / ``eval_push/*`` namespaces) now also reports
   ``Evaluate/mean_reward``, ``Evaluate/mean_episode_length``, and
   per-term tracking aggregates pulled from the same rollout's
   metrics_vec: ``Evaluate/forward_velocity``,
   ``Evaluate/cmd_vs_achieved_forward``,
   ``Evaluate/forward_velocity_cmd_ratio``,
   ``Evaluate/step_length_touchdown_event_m``,
   ``Evaluate/ref_q_track``, ``Evaluate/ref_body_quat_track``,
   ``Evaluate/ref_feet_pos_track``,
   ``Evaluate/cmd_forward_velocity_track``, plus the raw error
   diagnostics (``Evaluate/ref_q_track_err_rmse``,
   ``Evaluate/ref_body_quat_err_deg``,
   ``Evaluate/ref_feet_pos_err_l2``,
   ``Evaluate/ref_contact_phase_match``).  Mirrors
   ``toddlerbot/locomotion/train_mjx.py`` ``log_metrics`` exactly:
   no ``success_rate`` concept, checkpoint quality reads off
   ``mean_reward`` + ``mean_episode_length`` + per-term tracking.
   Smoke YAML now sets ``ppo.eval.enabled: true``,
   ``num_steps: 500`` (full episode horizon),
   ``num_envs: 64``, ``interval: 10``, ``deterministic: true``.

3. **G4 gate switched to ToddlerBot ``Evaluate/*`` floors**
   (`training/docs/walking_training.md` v0.20.1 §, Appendix A.5).
   Removed ``eval_walk/success_rate >= 0.60`` from the
   promotion-horizon gate.  Replaced with
   ``Evaluate/forward_velocity >= 0.075`` +
   ``Evaluate/mean_episode_length >= 475`` +
   ``Evaluate/cmd_vs_achieved_forward <= 0.075``.  Rationale
   captured inline: ToddlerBot has no ``success_rate`` concept
   anywhere in its locomotion code (verified via grep over
   ``toddlerbot/locomotion/{train_mjx,walk_env,mjx_env}.py``);
   once we adopt the same reward shape (which we now have:
   ``lin_vel_xy=5.0``, ``motor_pos=5.0``, strict ``-done``
   survival), eval reward + eval episode length already encode
   "alive AND tracking cmd_vx AND tracking q_ref" — a separate
   binary success bool just adds a degenerate signal.

4. **`ref_feet_pos_alpha` 200 → 30 in the smoke YAML**
   (`ppo_walking_v0201_smoke.yaml` reward_weights, with rationale
   block).  Smoke1 baseline ``ref/feet_pos_err_l2`` was 0.22 m →
   ``feet_err_l2`` (sum of squared per-foot residuals, both feet,
   all axes) ≈ 0.048 m² → at α=200 the reward exp(-200·0.048) ≈
   6e-5, leaving PPO with no gradient on the term.  α=30 gives
   r ≈ 0.24 at the iter-1 magnitude (gradient alive, not
   saturated) and drops to ~0.04 at err=0.30 m (still rejects
   gross misalignment).  Documented as a smoke-stage divergence
   from the ToddlerBot α=200 audit row, with a v0.20.2 follow-up
   to switch the foot-pos reward to a body-frame Euclidean norm
   matching ToddlerBot's ``_reward_torso_pos_xy`` shape, then
   restore α=200.

5. **Per-foot stride / swing-time diagnostics**
   (`wildrobot_env.py`, `metrics_registry.py`).  Smoke1's
   ``tracking/step_length_touchdown_event_m`` is a single carry
   across both feet — useful for the G4 floor but can't tell us
   whether one foot is stepping and the other isn't.  Added six
   per-foot tracking metrics emitted only on each foot's
   touchdown step:
   ``tracking/touchdown_rate_{left,right}``,
   ``tracking/swing_air_time_{left,right}_event_s``,
   ``tracking/step_length_{left,right}_event_m``.  All MEAN-reduced;
   per-event mean = ``<value>_event / touchdown_rate_<side>``
   post-aggregation.

6. **Topline cleanup** (`experiment_tracking.py`).  Dropped
   ``topline/success_rate`` (truncation-based, stuck at 0 through
   smoke1) and ``topline/velocity_error`` (unwired, also 0).
   Replaced with ``topline/cmd_vs_achieved_forward`` plumbed from
   ``env_metrics["tracking/cmd_vs_achieved_forward"]``.  The
   analyzer's checkpoint pick (which sorted on ``env/success_rate``
   and saw all iterations tie at 0, then fell back to
   ``env/episode_length`` and picked iter 120 over iter 130 by a
   0.1-step margin) will now prefer the new ``Evaluate/*`` block
   once smoke2 is logged.

### Validation

- `tests/test_v0201_env_zero_action.py` — 7/7 PASS (130 s).  G6
  invariant (zero residual ⇒ ``target_q == q_ref``) survives every
  edit; v6 proprio-history, offline service step-idx, and ref-obs
  channel checks all unchanged.
- `python training/train.py --config ... --verify` — quick smoke
  (3 iters × 4 envs × 8 rollout = 96 steps) pending; documented as
  a follow-up validation step in the smoke2 launch checklist.

### What this does NOT change

- Smoke command stays pinned at ``vx=0.15`` (single point).
- DR remains disabled (v0.20.2 task).
- ``ref_contact_match`` weight stays 0 until the contact-alignment
  probe passes (separate decision tree).
- The policy network shape, PPO hyperparameters, prior generator,
  and AMP path are unchanged.

### Smoke2 launch checklist

1. ``uv run python training/train.py --config
   training/configs/ppo_walking_v0201_smoke.yaml --verify`` —
   confirms eval block JIT-compiles and ``Evaluate/*`` keys flow to
   ``metrics.jsonl``.
2. Confirm the v0.20.1-smoke1 missing-keys list now appears in iter-1
   metrics.jsonl: ``reward/alive``, ``reward/total``,
   ``reward/feet_air_time``, ``reward/feet_clearance``,
   ``reward/feet_distance``, ``reward/torso_pitch_soft``,
   ``reward/torso_roll_soft``, ``Evaluate/mean_episode_length``.
3. Launch full smoke (150 iters, ~20M env steps, ~8 h on the same
   hardware as smoke1).
4. Gate against the updated v0.20.1 G4 (above).

---

## [v0.20.1-smoke1] - 2026-04-20: single-command walking smoke result

### Run

- Run: `training/wandb/offline-run-20260419_200601-zgs2m3gp`
- Checkpoints: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260419_200603-zgs2m3gp`
- Recommended checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260419_200603-zgs2m3gp/checkpoint_130_17039360.pkl`
- Alternate max-horizon checkpoint: `training/checkpoints/ppo_walking_v0201_smoke_v00201_20260419_200603-zgs2m3gp/checkpoint_120_15728640.pkl`

### Result

- **Walking verdict:** locomotion emerging; the smoke succeeds as a
  single-command `vx=0.15 m/s` proof-of-life rather than falling into
  the standing local minimum.
- **Velocity tracking:** good.  `env/forward_velocity` improves from
  `-0.035` at iter 1 to `0.148` at iter 130 against a fixed
  `0.150` command.  `tracking/cmd_vs_achieved_forward` drops from
  `0.260` to `0.073`.
- **Stability:** good in the training env.  `env/episode_length`
  rises from `85.6` to `~498/500`, so the gait is sustainable over
  nearly the full rollout horizon.
- **Gait structure:** partial but real.  `tracking/step_length_touchdown_event_m`
  improves from `-0.0065` to `0.0226 m` by iter 130.  That indicates
  actual stepping, but it still sits below the stricter `>= 0.03 m`
  smoke gate documented elsewhere, so the stride remains conservative.
- **PPO stability:** acceptable after the first update shock.
  `ppo/approx_kl` settles around `0.033-0.035` and
  `ppo/clip_fraction` around `0.39-0.41` from iter 10 onward.

### Checkpoint choice

The generic analyzer picks iter 120 because it sorts primarily on
`env/success_rate` and `env/episode_length`, but `env/success_rate`
is not meaningful for this smoke and iter 120 overshoots the target
velocity slightly (`0.1557` vs `0.1500`).  Iter 130 is the better
deployment / eval pick because it preserves the same near-max episode
length (`498.4`) while matching the command more closely
(`0.1480 / 0.1500`, ratio `0.9868`).

### Instrumentation gaps

- `env/success_rate` is truncation-based and stays `0.0` throughout
  the run, so it is not a valid walking selector for near-horizon
  episodes.
- `env/velocity_error` is `0.0` throughout the run and should not be
  used for walking analysis; `tracking/cmd_vs_achieved_forward` and
  `tracking/forward_velocity_cmd_ratio` carry the real signal here.
- The smoke ran after the ToddlerBot-alignment reward expansion, but
  W&B logging did not yet expose `reward/alive`,
  `reward/feet_air_time`, `reward/feet_clearance`,
  `reward/feet_distance`, `reward/torso_pitch_soft`, or
  `reward/torso_roll_soft`, so this run cannot confirm whether any of
  those shaping terms dominated.

### Next version

- Keep `checkpoint_130_17039360.pkl` as the baseline for immediate
  eval / video inspection or a short continuation run.
- Promote the missing v0.20.1 reward terms into W&B logging before
  the next smoke so reward dominance can be inspected directly.
- Add or enable `eval_clean/*` for walking so checkpoint selection is
  eval-based rather than inferred from train-rollout metrics.
- Replace or de-emphasize truncation-based `success_rate` and the
  broken `env/velocity_error` in walking top-line summaries.

## [v0.20.1-tb-align-fix1] - 2026-04-19: alignment-pass corrections

Three issues surfaced in review of the v0.20.1-tb-align changeset are
addressed here.  No new features; corrections to the prior pass so the
"ToddlerBot-aligned" claim actually holds.

### Corrections

1. **`alive` semantics fixed to strict ToddlerBot `-done`**
   (`wildrobot_env.py:_aggregate_reward`).  The prior implementation
   paid a dense `+alive_w` per surviving step and `-alive_w` on done
   ("asymmetric" — incorrectly described as ToddlerBot-aligned).
   ToddlerBot's `_reward_survival` is just `-done` at weight 10
   (`toddlerbot/locomotion/mjx_env.py:1897-1914`); WR now matches
   exactly: `0` while alive, `-alive_w * dt` on the terminating step.
   Reset reward also dropped from `alive_weight` to `0`.  Without
   this, PPO would have been biased toward any long-lived behavior
   regardless of imitation tracking — exactly the failure mode the
   v0.20.1 audit was supposed to prevent.

2. **`* dt` reward integration added** (`wildrobot_env.py:_aggregate_reward`).
   ToddlerBot computes `reward = sum(reward_dict.values()) * self.dt`
   (`mjx_env.py:1048`); WR's prior `_aggregate_reward` summed terms
   without dt scaling, so the published ToddlerBot weights were
   ~50x larger in WR effective per-step magnitude at `ctrl_dt = 0.02`.
   Event-based terms were the worst affected: `feet_air_time = 500`
   with an air time of ~0.5 s contributed `+250` per touchdown
   pre-fix vs `+5` in ToddlerBot.  Now the dt scale is applied
   uniformly across all weighted contributions so the audit's
   `Term ↔ Term` weight equivalence is real, not just a label.

3. **`feet_distance` uses the full torso rotation**
   (`wildrobot_env.py:_compute_reward_terms`).  The prior version
   rotated the foot delta by yaw only; ToddlerBot uses the inverse
   of the full torso quaternion (`walk_env.py:331-358`).  Yaw-only is
   torso-frame-invariant only when the robot is upright, so the
   reward fed pose-dependent noise into exactly the off-nominal
   states it was supposed to help.  The fix reconstructs the
   xyzw quat from MuJoCo's wxyz convention, normalizes, conjugates
   for the world→body inverse rotation, and uses
   `jax_frames.rotate_vec_by_quat` (already imported as `jax_frames`)
   to obtain the body-frame foot delta.

### Documentation updates

- `walking_training.md` Appendix A.3 now has a load-bearing
  "Reward integration" + "Survival semantics" callout block stating
  the `* dt` and `-done` contracts explicitly.  The summary row for
  `survival ↔ alive` was reworded ("strict ToddlerBot `-done`
  semantics; no dense per-step bonus") to retire the misleading
  earlier wording.

### Validation

- `tests/test_v0201_env_zero_action.py` (7/7 PASS) — the G6
  invariant survives the `* dt` rescale and the new survival
  semantics; the `feet_distance` quat-frame change is invisible to
  the zero-action probe but doesn't perturb the obs / step plumbing.

---

## [v0.20.1-tb-align] - 2026-04-19: ToddlerBot-alignment pass

### Context

Pre-smoke audit comparing the v0.20.1 reward family, termination, and
observation contract against the ToddlerBot reference recipe
(`toddlerbot/locomotion/{walk.gin,mjx_config.py,walk_env.py}`)
identified ten gaps materially affecting convergence (gradient-scale
mismatch on the velocity-tracking term, missing air-time / clearance
/ feet-distance / soft-pitch / soft-roll shaping rewards, hard
pitch/roll termination instead of soft band, no action delay, thin
proprio history, missing multi-command plumbing, leg residual
±0.50 rad vs ToddlerBot's ±0.25 rad uniform).  The audit and
per-row decisions are documented in `walking_training.md` Appendix A.

### Changes

1. **Comparison appended to `walking_training.md`** (`Appendix A`).
   Side-by-side audit of every reward weight, sigma, threshold, and
   action contract value vs ToddlerBot, with citations to the
   ToddlerBot file:line.  This is the source of truth for the
   alignment decisions below.

2. **EnvConfig + RewardWeightsConfig field additions**
   (`training_runtime_config.py`).  All defaults are inert (zero
   weight / no-op resample) so non-smoke configs are unaffected:
   - reward weights: `feet_air_time`, `feet_clearance`,
     `feet_distance`, `torso_pitch_soft`, `torso_roll_soft`
   - cmd resampling: `cmd_resample_steps`, `cmd_zero_chance`,
     `cmd_turn_chance`, `cmd_deadzone`
   - posture gates: `torso_pitch_soft_min/max_rad`,
     `torso_roll_soft_min/max_rad`, `min_feet_y_dist`,
     `max_feet_y_dist`

3. **WildRobotInfo schema extension** (`env_info.py`).  Added
   per-foot air-time / air-dist / spawn-pose foot-z bookkeeping
   plus a cmd-resample RNG carry.  The new fields are zero-init at
   reset and updated post-step by `wildrobot_env.py:step()`.

4. **New reward functions in env**
   (`wildrobot_env.py:_compute_reward_terms`).  Five ToddlerBot-
   shaped terms added (each defaults to zero weight via
   `RewardWeightsConfig`; weight-on-time enables them):
   - `feet_air_time` — `Σ_per_foot air_time * first_contact`,
     gated on `||cmd|| > 1e-6`.  Reads pre-update air time, mirrors
     `walk_env.py:_reward_feet_air_time` semantics.
   - `feet_clearance` — `Σ_per_foot air_dist * first_contact`,
     same cmd gate.
   - `feet_distance` — exp band penalty around lateral foot
     spacing in torso frame; uses `min/max_feet_y_dist`.
   - `torso_pitch_soft` / `torso_roll_soft` — exp band penalty
     inside `[soft_min, soft_max]`.  Replaces the hard pitch/roll
     termination (now disabled via `use_relaxed_termination: true`).

5. **Asymmetric `alive` semantics** (`_aggregate_reward`).  Survival
   is now `+alive_w` per surviving step, `-alive_w` on the
   terminating step (matches ToddlerBot's `_reward_survival = -done`
   at weight 10).  Net incentive: `alive_w * (length - 2 *
   terminated_count)`.

6. **Multi-command resampling plumbing** (`_sample_velocity_cmd` +
   `step()`).  Mirrors `toddlerbot/locomotion/mjx_env.py:1068-1077`
   resample-every-N-ticks with zero/turn/deadzone fields.
   `cmd_resample_steps == 0` disables (smoke contract).  Smoke YAML
   keeps single point at vx=0.15; the plumbing exists for v0.20.4
   without future env edits.

7. **PROPRIO_HISTORY_FRAMES bumped 3 → 15**
   (`training/envs/env_info.py` + `policy_contract/spec.py`).
   Matches ToddlerBot `c_frame_stack=15`.  No checkpoint migration
   burden (smoke not yet trained).

8. **Smoke YAML rescaled to ToddlerBot weights**
   (`training/configs/ppo_walking_v0201_smoke.yaml`):
   - `alive` 0.05 → 10.0; `ref_q_track` 0.65 → 5.0;
     `ref_body_quat_track` 0.10 → 5.0;
     `cmd_forward_velocity_track` 0.30 → 5.0;
     `cmd_forward_velocity_alpha` 4 → 200;
     `action_rate` -0.01 → -1.0
   - new shaping ON: `feet_air_time` 500.0, `feet_clearance` 1.0,
     `feet_distance` 1.0, `torso_pitch_soft` 0.5,
     `torso_roll_soft` 0.5, `slip` 0.05
   - termination: `use_relaxed_termination: true` (height-only)
   - action contract: `action_delay_steps: 1`,
     leg residuals tightened ±0.50 → ±0.25 rad

### Validation

- `tests/test_v0201_env_zero_action.py` (all 7 tests) PASS — G6
  invariant (zero residual ⇒ `target_q == q_ref`) survives every
  alignment edit; v6 proprio-history wiring still does not duplicate
  the current frame; offline service step-idx still advances
  monotonically; ref obs channels still track the service window.

### What this does NOT change

- Smoke command remains pinned at vx=0.15 (single point); multi-
  command training is `v0.20.2` per Appendix A.2.
- DR remains disabled in the smoke (`v0.20.2`).
- Contact-match Gaussian remains gated to weight 0 until the
  contact-alignment probe passes (separate decision tree).
- The policy network shape, PPO hyperparameters, and AMP path are
  unchanged.

### Open follow-ups

- After smoke launches, monitor `reward/feet_air_time` and
  `reward/feet_clearance` for runaway dominance — both depend on
  large absolute values (air_time in seconds, air_dist in metres);
  if either dwarfs the imitation block, downsample the weight by
  10x.  ToddlerBot's `feet_air_time=500` is calibrated to
  `cycle_time≈0.72 s`; WR's prior may run at a different cadence.
- `feet_distance` uses world-frame `y` at yaw=0 (smoke condition);
  when yaw becomes nonzero in v0.20.4 verify the torso-frame
  rotation is right-handed for both feet.

---

## [v0.20.1-prep] - 2026-04-19: high-confidence smoke prep

### Context

Pre-kickoff prep for the v0.20.1 PPO smoke (single-command vx=0.15
imitation-residual run, M2 ~20M env-step budget).  Smoke design
review surfaced three soft spots vs the ToddlerBot reference recipe
(gaps A/B/C in `training/docs/walking_training.md` v0.20.1 §); this
changeset closes all three, plus tightens G6 and pre-wires the M1
fail-mode response.

### Changes

1. **G6 residual-only filter contract** (`080f4f8`).
   Restructured `step()` so the action filter operates on the raw
   policy residual command, not the composed `q_target`.  Iter-0
   bare-q_ref replay now holds for any `action_filter_alpha` (legacy
   path required `alpha=0`).  New 50-step `tracking/residual_q_abs_max`
   metric assertion in `tests/test_v0201_env_zero_action.py`
   (test #6) prevents regression.

2. **Contact-alignment baseline probe** (`697060e`).
   New `tools/v0201_contact_alignment_probe.py` runs the smoke env
   with zero action and reports `ref/contact_phase_match` against
   the prior's commanded contact mask.

   **Baseline result (commit `080f4f8`, vx=0.15, seed=42, 100 steps):**
     - mean ref/contact_phase_match = 0.8000  (FAIL: < 0.90)
     - longest mismatch streak = 6/6 steps  (FAIL: > 5)
     - env terminates at probe step 67 under bare q_ref replay
     - first divergence: steps 2-3 (cmd=(1,1) meas=(0,0)) — startup
       transient leaves both feet briefly off the floor
     - second divergence: step 15+ (cmd=(1,0) meas=(0,1)) — lateral
       fall starts inverting stance side

   This is real signal about ref/contact_match noise from iter-0;
   remediation (loosen sigma, tighten prior, accept the noise) is a
   follow-up decision, not a kickoff blocker.

3. **slip + pitch_rate reward hooks at zero weight** (`c972be1`).
   The M1 fail-mode tree's "balance issue" branch says "increase
   regularizer weight on pitch_rate and slip"; both terms are now
   computed in `_compute_reward_terms` and exposed via
   `reward_weights.{slip,pitch_rate}`.  Default weights stay at 0 —
   the M1 response becomes a YAML knob, not a code edit.  Raw
   penalty values logged at `reward/penalty_slip_raw` /
   `reward/penalty_pitch_rate_raw` so the M1 weight can be sized
   from the smoke baseline before promoting either term.

4. **YAML alpha comment update** (`46ee980`).
   No behavior change.  Stale "must be 0.0 for G6" comment replaced
   with a note that the residual-only contract makes alpha free.

5. **Actor proprio history (wr_obs_v6_offline_ref_history)** (`0f36c4e`,
   defect fix `2aa2e32`, v5 removal pending).
   ToddlerBot's actor stacks ~15 past frames; the v5 actor saw a
   single proprio frame, with no way to estimate base velocity (gap
   A in the smoke review).  Added v6 layout: strict superset of v5
   with a flat 3-frame past-proprio stack (gyro + foot_switches +
   joint_pos_norm + joint_vel_norm + prev_action) appended.

   **Numbers:** proprio_bundle = 3 + 4 + 3*N = 64 floats for N=19.
   v6 obs_dim = 126 + 192 (3*64) = **318**.  PPO sees 4 frames of
   proprio total (1 current + 3 past); enough for finite-diff
   velocity from joint pos and one integration of gyro.

   **Defect fix (`2aa2e32`):** the initial wiring (`0f36c4e`) tiled
   the current bundle into the buffer at reset and fed the rolled
   buffer (containing the just-computed current bundle) back into
   the same step's obs — so the actor was actually getting "current
   frame + 2 past + duplicated current," not the documented 3 past
   frames.  Fix: zero-fill at reset, pass PRE-roll buffer to obs,
   store ROLLED buffer for the next step.  New test #7 in
   `tests/test_v0201_env_zero_action.py` pins the schema contract
   (reset buffer = zeros; step 1 obs history slice = zeros; step 2
   newest history slot = post-step-1 stored bundle, never the
   step-2 current frame).

   **v5 removal:** v6 is now the only supported offline-ref layout.
   v5 (`wr_obs_v5_offline_ref`) was kept loadable for one commit as
   "ablation insurance" but had no live consumer — removed entirely
   from the policy contract, env accept-list, and parity tests.
   v6 dim test now anchors against v4 (`v6 - v4 = (n_act + 20)
   ref-window + PROPRIO_HISTORY_FRAMES * proprio_bundle`).
   v5-trained checkpoints (none exist outside the smoke `--verify`
   artifacts on disk) cannot be loaded.

### Validation (after v5 removal)

- 7/7 v0.20.1 zero-action tests pass under v6
  (`tests/test_v0201_env_zero_action.py`)
- 8/8 policy-contract parity tests pass (v5 fixture + 2 v5 tests
  removed; v6 dim test re-anchored against v4)
- contract / spec / reward-schema tests pass (35 total in the
  combined suite)
- `python training/train.py --config ... --verify` completes cleanly
  (3 iters × 4 envs × 8 rollout = 96 steps, obs_dim=318)

### Known soft spots carried into kickoff

- **ref/contact_match noise from iter-0** (probe baseline FAIL): the
  prior's contact schedule is ~80% aligned with measured contacts at
  bare q_ref replay.  Tracking signal is weaker than the design's G3
  sigma assumes; if the smoke stalls on the contact term, the probe
  baseline tells us where to look.
- **Prior body translation gap** (unchanged from `walking_training.md`
  G7): bare q_ref replay terminates at probe step 67; PPO is asked
  to extend survival from there.
- **No domain randomization / IMU noise / action delay**
  (intentionally OFF for the smoke).  These are v0.20.2 sim2real
  concerns.

### Follow-up: contact_threshold_force lowered 5.0 N → 1.0 N

The smoke YAML now pins `contact_threshold_force: 1.0` (was 5.0,
inherited from the v0.17.x standing-PPO defaults).  Matches
ToddlerBot's `contact_force_threshold` (1.0 N per
`toddlerbot/locomotion/mjx_config.py:95`) and is consistent with
WildRobot's own `foot_switch_threshold` (2.0 N) within the same
order of magnitude.

**Why:** `contact_threshold_force` is the cutoff that classifies a
foot as "in contact with the floor" — drives the
`ref_contact_match` reward, `prev_*_loaded` state, and G4
touchdown-event detector.  At 5.0 N (≈ 12.5% of the 4.1 kg robot's
weight) the post-reset force ramp took 2-3 ctrl steps to cross
threshold, producing a false-negative on `meas_contact` even though
the feet were geometrically touching.  That was the dominant source
of the early-step mismatches in the contact-alignment probe at the
5.0 N baseline.  At 1.0 N the cutoff sits well above MJX's contact-
solver noise floor (mN range) without rejecting genuine light
contacts.

**Probe baseline (vx=0.15, seed=42, 100 steps):**

| Metric                | 5.0 N (was) | 1.0 N (now)  |
|-----------------------|-------------|--------------|
| mean ref/cpm          | 0.8000      | 0.8387       |
| min  ref/cpm          | 0.1353      | 0.1353       |
| L hard match rate     | 0.746       | 0.836        |
| R hard match rate     | 0.791       | 0.791        |
| L mismatch streak     | 6 steps     | 3 steps      |
| R mismatch streak     | 6 steps     | 6 steps      |
| Env terminates at     | step 67     | step 67      |
| Probe verdict         | FAIL        | FAIL (0.84)  |

The threshold fix removed the post-reset transient slice (steps
2-3 at 5.0 N: `cmd=(1,1) meas=(0,0)` → at 1.0 N: `meas=(1,1)`,
`r=1.0`); the L-streak and L-match rate moved as expected.  The
remaining mismatches are structural — brief single-foot load shifts
during DS→SS transitions (steps 3-6) and the open-loop LIPM lateral
fall starting around step 15 (which threshold lowering cannot fix;
that's what PPO is for).  The env still terminates at the same
step 67 because the physics path is unchanged.

### Follow-up: ref_contact_match gated to 0 for the first smoke

After committing the contact-alignment probe and reading the FAIL
baseline (now mean = 0.84 at the corrected 1.0 N threshold; was
0.80 at 5.0 N), the smoke YAML now sets
`reward_weights.ref_contact_match: 0.0`.  This is a smoke-stage
design choice, not a code-path removal:

- The reward term is still computed every step; only its weighted
  contribution to the total is gated.
- `reward/ref_contact_match` (always 0 at this weight) and
  `ref/contact_phase_match` (raw Gaussian signal in [0, 1]) stay
  fully logged for diagnosis.
- `ref_contact_match_sigma` (G3 default 0.5) is unchanged.
- The contact-alignment probe (`tools/v0201_contact_alignment_probe.py`)
  is unchanged.
- The env reward code is unchanged — the gating is a single YAML
  knob.

**Rationale:** under bare q_ref replay the prior's commanded contact
schedule already disagrees with the measured contacts (post-reset
force-threshold transient + open-loop LIPM lateral fall by ~step
15).  At weight 0.10 that disagreement is structured noise in the
imitation gradient from iter-0, exactly the gap-C risk flagged in
the smoke design review.  Better to learn `ref_q_track` /
`ref_body_quat_track` / `ref_feet_pos_track` / `cmd_forward_velocity_track`
cleanly first and re-introduce the contact term once the probe
passes.

**Re-enable plan** (matches `walking_training.md` v0.20.1 §):
1. Probe must pass first (mean ≥ 0.90, no ≥5-step mismatch streaks)
   — addressed by lowering `contact_threshold_force` to kill the
   post-reset false-negative, tightening the prior so the open-loop
   fall happens later, or shortening the smoke horizon below the
   divergence point.
2. Restore at `0.02` first (an order of magnitude below the
   imitation block) and confirm the term doesn't dominate the
   gradient.
3. Return to `0.10` only once the probe stays clean across multiple
   seeds.

---
