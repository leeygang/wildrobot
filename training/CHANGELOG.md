# AMP Training Changelog
# CHANGELOG

This changelog tracks capability changes, configuration updates, and training results for the WildRobot AMP training pipeline.

**Ordering:** Newest releases first (reverse chronological order).

---

## [v0.16.1] - 2026-03-16: Teacher-to-student handoff bootstrap

### Summary
`v0.16.1` adds the minimum viable student handoff path on top of `v0.16.0`: teacher rollout sharding, a stable dataset contract, behavior-cloning warm start, and PPO warm-start plumbing for the command-conditioned student branch.

### Code Updates
- Teacher rollout dataset/export:
  - `training/imitation/dataset.py`
  - `training/imitation/collect_teacher_rollouts.py`
  - contract: sharded `.npz` + `metadata.json`
  - required shard arrays:
    - `obs`, `actions`, `phase`, `velocity_cmd`, `forward_velocity`, `done`, `left_foot_contact`, `right_foot_contact`
  - recommended shard arrays supported:
    - `teacher_action_mean`, `teacher_action_std`, `teacher_value`, `root_pitch`, `root_pitch_rate`
- Student pretraining:
  - `training/imitation/pretrain_student.py`
  - behavior-cloning warm start on teacher action targets (`obs -> actions`)
  - exports `student_pretrain_v0.16.1` checkpoint with policy/value/processor params + `policy_spec_hash`
- Student PPO config:
  - `training/configs/ppo_walking_student.yaml`
  - flat terrain, no pushes, narrow command range (`0.05-0.15 m/s`)
- PPO handoff hooks:
  - `training/core/training_loop.py`
    - add pretrained warm-start checkpoint application path (policy-spec hash guarded)
  - `training/train.py`
    - add `--pretrained-student` and config-driven warm-start checkpoint loading
  - `training/configs/training_runtime_config.py`, `training/configs/training_config.py`
    - add PPO fields:
      - `pretrained_checkpoint_path`
      - `teacher_regularization_checkpoint`
      - `teacher_regularization_weight`
      - `teacher_regularization_decay`
    - teacher-regularization fields are intentionally placeholder-level in this branch
- Tests:
  - `training/tests/test_teacher_rollout_export.py`
  - `training/tests/test_student_imitation_pipeline.py`

### What Remains After `v0.16.1`
- integrate active teacher-regularization loss during PPO fine-tuning (currently config/plumbing placeholder)
- run longer student PPO probes and widen commands only after narrow-regime stability is confirmed

## [v0.16.0] - 2026-03-16: Transition from exhausted PPO-only walking to MuJoCo motion-tracking teacher bootstrap


### Summary
`v0.15.x` PPO-only walking is now considered exhausted as the main branch. The reward-only and structure-only branches succeeded at producing forward speed, but all successful speed branches converged to the same terminal failure mode: pitch-driven locomotion exploits instead of a stable gait basin. `v0.16.0` therefore formalizes a stage transition rather than another reward retune: bootstrap a robot-native gait in MuJoCo with a retargeted reference motion and a phase-conditioned motion-tracking teacher, then transfer that behavior into a command-conditioned locomotion student.

### Results That Triggered The Transition
- `v0.15.10` broke the original low-speed / stepping-in-place basin, but reached commanded speed by collapsing into pitch failure.
- `v0.15.11` improved early stability, but ended in the same final pitch-collapse basin.
- `v0.15.12` improved early structured behavior further, but still ended in the same terminal basin.

### Results (v0.15.12)
- Run: `training/wandb/offline-run-20260315_210623-l94cm2gs`
- Checkpoints: `training/checkpoints/ppo_walking_v01512_20260315_210625-l94cm2gs`
- Verdict: better early structured branch entry, same final pitch-collapse basin
- Early/mid checkpoint:
  - `env/forward_velocity: 0.034`
  - `env/velocity_cmd: 0.100`
  - `env/velocity_error: 0.100`
  - `env/success_rate: 0.991`
  - `eval_clean/success_rate: 0.789`
  - `term_pitch_frac: 0.009`
  - `reward/step_length: 0.184`
  - `reward/step_progress: 0.006`
- Final checkpoint:
  - `env/forward_velocity: 0.211`
  - `env/velocity_cmd: 0.100`
  - `env/velocity_error: 0.152`
  - `env/success_rate: 0.002`
  - `eval_clean/success_rate: 0.055`
  - `term_pitch_frac: 0.998`
  - `reward/step_length: 0.185`
  - `reward/step_progress: 0.024`
  - `reward/dense_progress: 0.192`
  - `debug/propulsion_gate: 0.105`

### Root Cause
- Stage `v0.15.x` established that WildRobot can produce forward speed in simulation.
- Stage `v0.15.x` did **not** establish a stable gait basin that PPO-from-scratch can reliably discover with local reward shaping.
- Even when structured touchdown rewards are active:
  - structured step terms improve
  - but they still do not dominate the terminal behavior strongly enough
  - forward speed ultimately rises together with pitch collapse
- Conclusion:
  - the missing ingredient is now gait structure / teacher signal, not another local reward coefficient change

### Stage Decision
- Stop `v0.15.x` reward-only walking as the main branch.
- Do not spend more time on local reward tuning as the primary approach.
- Start `v0.16.0` as a MuJoCo teacher-bootstrap stage.

### `v0.16.0` Plan
- Keep WildRobot and MuJoCo as the primary stack.
- Do **not** attempt direct H1 or Isaac Lab policy transfer.
- Follow the external best-practice pattern conceptually instead:
  - retarget a minimal nominal forward-walk reference to WildRobot
  - train a phase-conditioned motion-tracking teacher in MuJoCo
  - initialize or distill a locomotion student from that teacher
  - only then return to command-conditioned task locomotion and later robustness

### Concrete `v0.16.0` Deliverables
- `training/reference_motion/`
  - loader and phase utilities for a nominal walk clip
  - explicit first-pass clip contract:
    - `.npz` arrays for `phase`, `root_*`, `joint_*`, foot positions, and foot contacts
    - companion `.json` metadata with actuator names, loop mode, and nominal gait metadata
- retargeting utility
  - produce WildRobot-feasible reference motion from a source walk clip
- teacher config
  - add `training/configs/ppo_walking_teacher.yaml`
- teacher reward / observation path
  - phase-conditioned observations
  - root pose / velocity tracking
  - joint tracking
  - foot placement / contact timing tracking
- teacher metrics / tests
  - tracking error
  - contact timing agreement
  - phase consistency
  - deterministic tests for reference motion loading and teacher reward terms

### `v0.16.0` Implementation (Teacher Bootstrap)
- Reference-motion module added:
  - `training/reference_motion/loader.py`
  - `training/reference_motion/phase.py`
  - `training/reference_motion/retarget.py`
- Bootstrap reference clip pipeline added:
  - script: `scripts/export_teacher_reference_clip.py`
  - generated local bootstrap placeholder clip (no external mocap download):
    - `training/reference_motion/wildrobot_walk_forward_v001.npz`
    - `training/reference_motion/wildrobot_walk_forward_v001.json`
- Teacher config added:
  - `training/configs/ppo_walking_teacher.yaml`
  - flat terrain only, no pushes, short proof-of-life probe
- Teacher environment path added:
  - teacher mode toggle and reference clip loading
  - `wr_obs_teacher` observation layout with phase and reference targets
  - teacher reward helpers for root pose/velocity, joint pose, foot/contact timing, phase consistency, upright tracking, and effort/smoothness penalties
- Teacher metrics schema added:
  - tracking errors, contact timing agreement, phase consistency, teacher phase diagnostics
- Focused tests added:
  - `training/tests/test_reference_motion.py`
  - `training/tests/test_teacher_reward_terms.py`

### Training Intent
- Replace PPO-from-scratch gait discovery with reference-guided gait bootstrap.
- Prove stable nominal teacher walking before building a locomotion student.
- Use Stage `v0.16.0` to answer one question only:
  - can WildRobot track a single retargeted forward-walk reference stably in MuJoCo?

### Planned Follow-Up: `v0.16.1`
- If `v0.16.0` succeeds, do not jump straight to wide-range task RL.
- First build a teacher-to-student handoff:
  - export teacher rollouts
  - use a sharded `.npz` dataset contract with `obs`, `actions`, phase, command, forward velocity, done flags, and foot contacts
  - warm-start a command-conditioned locomotion student with behavior cloning / distillation
  - fine-tune the student with PPO locomotion rewards plus a decaying teacher regularization term
- Keep `v0.16.1` narrow:
  - flat terrain only
  - command range close to the teacher nominal gait
  - no pushes, no rough terrain
- `v0.16.1` is successful if the student preserves the teacher gait basin while learning narrow command-conditioned locomotion without immediately regressing to the old pitch-collapse solution.

## [v0.15.12] - 2026-03-16: Contact-driven step-to-step propulsion after `v0.15.11` still converged to pitch-collapse speed

### Summary
`v0.15.11` improved the shape of the run but did not change the terminal basin. It delayed the old `v0.15.10` pitch exploit, but by the end of the `40`-iteration probe it still converged to the same outcome: strong forward speed with `term_pitch_frac = 1.0`, zero train success, and near-dead `eval_clean`. The root cause is now clearer: dense velocity-shaped reward is still the easiest way to earn speed, while `step_length` remains mostly flat and `cycle_progress` stays too weak and too time-based to dominate credit assignment. `v0.15.12` should therefore pivot from velocity-led reward credit to contact-driven step-to-step propulsion.

### Results (v0.15.11)
- Run: `training/wandb/offline-run-20260315_171309-wxzaehsm`
- Checkpoints: `training/checkpoints/ppo_walking_v01511_20260315_171311-wxzaehsm`
- Verdict: delayed collapse, but same final pitch-driven speed basin
- Trajectory:
  - early `20`-iter checkpoint:
    - `env/forward_velocity: 0.037`
    - `env/velocity_cmd: 0.164`
    - `env/velocity_error: 0.153`
    - `env/success_rate: 0.895`
    - `eval_clean/success_rate: 0.489`
    - `term_pitch_frac: 0.105`
  - final `40`-iter checkpoint:
    - `env/forward_velocity: 0.221`
    - `env/velocity_cmd: 0.164`
    - `env/velocity_error: 0.149`
    - `env/success_rate: 0.000`
    - `eval_clean/success_rate: 0.050`
    - `term_pitch_frac: 1.000`
    - `reward/dense_progress: 0.166`
    - `reward/step_length: 0.138`
    - `reward/cycle_progress: 0.017`
    - `debug/propulsion_gate: 0.145`

### Root Cause
- `v0.15.11` proved that posture-gating alone is not enough:
  - it slowed the exploit and improved early stability
  - but it did not change which reward path ultimately dominates
- The structured gait terms still are not the main explanation for speed:
  - `step_length` stays nearly flat as the run accelerates
  - `cycle_progress` remains tiny and sparse
  - the final speed increase still comes mainly from dense forward-progress reward
- The current cycle reward is also poorly aligned to actual gait events:
  - clock-cycle completion is weaker than contact-driven touchdown-to-touchdown progress for this failure mode
- Net conclusion:
  - the next branch must stop paying primarily for instantaneous forward motion
  - and instead pay for forward displacement caused by successful steps

### Code Updates
- `training/envs/wildrobot_env.py`
  - add `compute_step_progress_touchdown_reward()` for structured touchdown-to-touchdown forward displacement credit in heading-local frame
  - track touchdown state in env info (`last_touchdown_root_pos`, `last_touchdown_foot`) so reward is tied to real contact transitions
  - gate `forward_reward` with `compute_propulsion_quality_gate(step_length, step_progress)` and remove dense velocity / cycle terms from the gate path
  - keep `dense_progress` and `cycle_progress` available only as auxiliary reward channels (not gate openers)
  - make the commanded forward reward unlock only from structured propulsion evidence:
    - `step_length`
    - `step_progress`
  - keep `wr_obs_v3` clock and phase-gated clearance, but demote pure clock-cycle reward as the main propulsion signal
- `training/configs/training_runtime_config.py`, `training/configs/training_config.py`
  - add/parse structured propulsion config fields:
    - `reward_weights.step_progress`
    - `reward_weights.step_progress_target_scale`
    - `reward_weights.step_progress_sigma`
    - `reward_weights.propulsion_gate_step_length_weight`
    - `reward_weights.propulsion_gate_step_progress_weight`
- `training/tests/test_reward_terms.py`
  - add helper-linked tests for:
    - touchdown-to-touchdown forward progress reward semantics
    - propulsion gate depending on structured step evidence instead of dense velocity alone
    - reset / alternating-touchdown behavior for the new `step_progress` term

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.12"`
- keep:
  - `env.actor_obs_layout_id: wr_obs_v3`
  - fresh-run semantics
  - short probe format (`ppo.iterations: 40`)
- change the reward mix toward structured contact-driven propulsion:
  - `dense_progress: 0.0` or tiny support only
  - `step_length`: increase into the primary Stage-A structured term
  - add `step_progress` as a strong reward
  - demote or remove `cycle_progress` as the main propulsion driver
  - keep `foot_place: 0.0` and `step_event` negligible
- narrow command range again for the structured branch:
  - bias toward `0.05-0.15 m/s`
- keep stronger pitch stability shaping from `v0.15.11`

### `v0.15.12` Plan (Completed)
- Keep:
  - zero-baseline forward reward
  - `wr_obs_v3`
  - phase-gated clearance
  - stronger pitch / posture penalties
- Remove dense velocity as the main learning path:
  - take `dense_progress` out of the propulsion gate
  - reduce its weight to near-zero or zero
- Shift to contact-driven step-to-step propulsion:
  - reward touchdown-time `step_length`
  - reward new touchdown-to-touchdown `step_progress`
  - gate commanded forward reward on structured step evidence, not raw speed
- Keep the probe short and decisive:
  - fresh run
  - `ppo.iterations: 40`

### Training Intent
- Convert the branch from velocity-led learning to contact-driven locomotion learning.
- Make the policy earn forward reward because steps move the body forward, not because the body is already moving forward.
- Readout:
  - iter `20`: `env/forward_velocity > 0.03` with `term_pitch_frac` still well below `v0.15.10/11` collapse levels
  - iter `40`: `env/forward_velocity > 0.06` with meaningful `eval_clean` survival and structured reward terms (`step_length`, `step_progress`) clearly active
  - if speed rises again while structured step rewards stay flat, stop and treat the branch as failed

## [v0.15.11] - 2026-03-16: Stability-constrained propulsion after `v0.15.10` broke the low-speed basin by pitching out

### Summary
`v0.15.10` is the first walking branch that truly broke the low-propulsion / stepping-in-place basin. The run reached and exceeded commanded forward velocity, proving that the reward-economics fix worked. But the branch immediately converted that propulsion into a pitch-driven collapse basin: forward speed kept rising while `term_pitch_frac` went to `1.0` and `eval_clean` collapsed. `v0.15.11` should therefore keep the `v0.15.10` reward-economics foundation, but make propulsion pay only when posture remains sufficiently upright.

### Results (v0.15.10)
- Run: `training/wandb/offline-run-20260315_135306-ta8q1b8v`
- Checkpoints: `training/checkpoints/ppo_walking_v01510_20260315_135308-ta8q1b8v`
- Verdict: propulsion unlocked, stability lost
- Basin reached by iter `40-50`:
  - `env/forward_velocity: 0.175 -> 0.215`
  - `env/velocity_cmd: 0.164`
  - `env/velocity_error: 0.154 -> 0.148`
  - `env/success_rate: 0.010 -> 0.000`
  - `eval_clean/success_rate: 0.021 -> 0.042`
  - `term_pitch_frac: 0.990 -> 1.000`
  - `reward/dense_progress: 0.389 -> 0.456`
  - `reward/step_length: ~0.135` (plateaued)
  - `reward/cycle_progress: ~0.017` (still weak)
  - `debug/propulsion_gate: 0.467 -> 0.525`

### Root Cause
- `v0.15.10` fixed the old reward basin:
  - forward reward no longer overpaid near-standing
  - propulsion gating and dense progress shaping finally created a real path to translation
- But the new dense propulsion path is not posture-constrained enough:
  - `dense_progress` keeps rising as the policy leans harder
  - `step_length` plateaus instead of improving further
  - `cycle_progress` remains too small to be the dominant structured gait signal
- Net result:
  - the policy discovered a fast way to earn propulsion-related reward
  - that fast way is a pitch-driven locomotion exploit, not stable walking

### Code Updates
- `training/envs/wildrobot_env.py`
  - add `compute_smooth_upright_gate()` and use it to posture-gate `dense_progress` with smooth pitch/pitch-rate margins
  - posture-gate commanded forward reward (zero-baseline path) with configurable strength so forward tracking pays far less during pitch collapse
  - tighten propulsion gate economics via weighted structured-vs-dense composition:
    - structured signal: `max(step_length, cycle_progress)`
    - dense support: capped contribution with configurable weight/cap
- `training/configs/training_runtime_config.py`, `training/configs/training_config.py`
  - add/parse new reward fields:
    - `dense_progress_upright_pitch`
    - `dense_progress_upright_pitch_rate`
    - `dense_progress_upright_sharpness`
    - `forward_upright_gate_strength`
    - `propulsion_gate_dense_weight`
    - `propulsion_gate_structured_weight`
    - `propulsion_gate_dense_cap`
- `training/tests/test_reward_terms.py`
  - extend helper-linked tests for:
    - posture-gated dense progress under excessive pitch/pitch-rate
    - monotonic smooth upright gate behavior
    - propulsion gate no longer dominated by dense-progress-only signal

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.11"` and shorten probe to `ppo.iterations: 40`
- keep `wr_obs_v3` and the `v0.15.10` reward-economics base (zero-baseline + propulsion gating)
- retune for stability-constrained propulsion:
  - lower `dense_progress` weight (`0.8 -> 0.35`)
  - slightly raise structured cycle term (`cycle_progress: 1.0 -> 1.2`)
  - strengthen stability shaping:
    - `orientation: -0.4 -> -0.7`
    - `pitch_rate: -0.25 -> -0.4`
    - `collapse_height: -0.25 -> -0.3`
    - `collapse_vz: -0.15 -> -0.2`
  - add upright/propgate controls:
    - `dense_progress_upright_pitch: 0.22`
    - `dense_progress_upright_pitch_rate: 0.75`
    - `dense_progress_upright_sharpness: 10.0`
    - `forward_upright_gate_strength: 0.85`
    - `propulsion_gate_dense_weight: 0.25`
    - `propulsion_gate_structured_weight: 0.75`
    - `propulsion_gate_dense_cap: 0.35`

### `v0.15.11` Plan (Completed)
- Keep the `v0.15.10` foundation:
  - zero-baseline forward reward
  - propulsion-quality gate
  - relaxed Stage-A step targets
  - `wr_obs_v3`
- Add stability-constrained propulsion:
  - make `dense_progress` posture-aware using smooth pitch / pitch-rate gates
  - optionally make the forward reward itself upright-gated for commanded walking
  - reduce `dense_progress` weight so it cannot dominate by pure leaning
  - increase `orientation` and `pitch_rate` penalties
  - strengthen pre-collapse / anti-pitch shaping if needed
- Tighten what opens the propulsion gate:
  - let `step_length` and `cycle_progress` matter more relative to dense progress
  - dense progress should support propulsion learning, not fully define it
- Keep the probe short:
  - fresh run
  - `ppo.iterations: 40`

### Training Intent
- Preserve the propulsion breakthrough from `v0.15.10`.
- Prevent the new reward path from paying mostly for pitch-driven forward collapse.
- Readout:
  - iter `20`: `env/forward_velocity > 0.03` while `term_pitch_frac < 0.25`
  - iter `40`: `env/forward_velocity > 0.05` and `eval_clean/success_rate` remains meaningfully alive
  - if speed rises only together with pitch collapse again, stop immediately

## [v0.15.10] - 2026-03-15: Reward-economics fix after `v0.15.9` low-propulsion basin

### Summary
`v0.15.9` confirmed that the branch is no longer blocked by missing stepping, missing clock input, or missing propulsion-specific reward terms. It is blocked by reward economics. The dense forward-tracking term still pays a large positive reward even when the robot barely moves, while the new propulsion terms (`step_length`, `cycle_progress`) are too sparse and too small in realized value to compete. PPO therefore converges to safe low-propulsion stepping because that solution is already strongly rewarded.

### Results (v0.15.9)
- Run: `training/wandb/offline-run-20260315_093127-25uho0zk`
- Checkpoints: `training/checkpoints/ppo_walking_v00159_20260315_093129-25uho0zk`
- Verdict: no-go; another safe stepping / near-standing basin
- Best checkpoint by current analyzer: iter `50`
- Key metrics near the end of the probe:
  - `env/forward_velocity: 0.003-0.009`
  - `env/velocity_cmd: 0.164`
  - `env/velocity_error: 0.170-0.174`
  - `env/success_rate: ~0.89`
  - `eval_clean/success_rate: 1.000`
  - `term_pitch_frac: ~0.11`
  - `reward/step_event: ~0.354`
  - `reward/foot_place: ~0.229`
  - `reward/step_length: ~0.003`
  - `reward/cycle_progress: ~0.004`
  - `debug/velocity_step_gate: ~0.936`

### Root Cause
- Dense forward reward is still too generous at low velocity:
  - `reward/forward = exp(-|forward_vel - velocity_cmd| * forward_velocity_scale)`
  - at `velocity_cmd ~= 0.164` and `forward_vel ~= 0`, raw forward reward is still about `0.37`
  - with `tracking_lin_vel: 8.0`, that is about `+3.0` reward before other shaping
- The standing penalty is far too small to offset that basin:
  - `velocity_standing_penalty: 0.5`
- The step gate still opens on generic stepping signals rather than propulsion-quality signals:
  - clearance / periodicity / touchdown are enough to unlock the dense forward reward
- The new propulsion terms are numerically too weak in practice:
  - `step_length` and `cycle_progress` are correct directionally, but their realized values stay around `0.001-0.005`
  - `cycle_progress` is also sparse because it only pays on cycle completion
- Net result:
  - safe stepping with little translation is already highly rewarded
  - real propulsion is too sparse and too hard to discover early

### Code Updates
- `training/envs/wildrobot_env.py`
  - add `compute_zero_baseline_forward_reward()` and apply it to forward tracking so commanded near-standing no longer gets large positive reward
  - add `compute_dense_progress_reward()` for dense heading-local propulsion credit each step
  - add `compute_propulsion_quality_gate()` and gate forward reward on propulsion-quality (`step_length`, dense progress, cycle progress), replacing generic stepping gate economics
  - keep cycle progress heading-local and cycle-complete, but pair it with dense per-step progress shaping
- `training/configs/training_runtime_config.py`, `training/configs/training_config.py`
  - add and parse `reward_weights.dense_progress`
- `training/tests/test_reward_terms.py`
  - add helper-linked tests for zero-baseline forward reward, propulsion-quality gate, and dense progress semantics

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.10"`
- keep fresh-run, `wr_obs_v3`, and short probe (`ppo.iterations: 60`)
- relax Stage-A propulsion targets:
  - `step_length_target_scale: 0.30 -> 0.15`
  - `step_length_sigma: 0.03 -> 0.06`
  - `cycle_progress_target_scale: 1.0 -> 0.6`
  - `cycle_progress_sigma: 0.06 -> 0.12`
- densify propulsion shaping:
  - `dense_progress: 0.8`
- reduce legacy stepping reward path:
  - `step_event: 0.001`
  - `foot_place: 0.0`

### Training Intent
- Break the reward basin first, not the controller architecture.
- The next probe should answer whether propulsion becomes economically preferable once bad tracking no longer pays well.
- Readout:
  - iter `20`: forward velocity should clearly exceed the `v0.15.9` floor
  - iter `40`: should separate from safe stepping / near-standing
  - iter `60`: if `env/forward_velocity <= 0.02`, treat the branch as failed

## [v0.15.9] - 2026-03-15: Per-step propulsion rewards after `v0.15.8` still stepped in place

### Summary
Analyzes the failed `v0.15.8` run `training/wandb/offline-run-20260314_230225-jgkxxp2w`. Even with `wr_obs_v3` clocked observations, the policy still converged to a stepping-in-place basin: stepping activity was present, but commanded forward translation remained near the floor. `v0.15.9` therefore shifts reward structure from generic stepping to explicit propulsion contracts: touchdown forward step length, per-cycle net forward displacement, and clock-phase-gated clearance.

### Why `v0.15.8` failed
- Clock features improved timing cues but did not change what was being paid for most strongly.
- Existing touchdown/foot-placement terms still allowed high reward from low-propulsion stepping patterns.
- Result: stable stepping without material cycle-to-cycle forward displacement.

### Code Updates
- `training/envs/wildrobot_env.py`
  - add touchdown `step_length` reward tied to `velocity_cmd`
  - add `cycle_progress` reward at clock-cycle completion using net forward displacement
  - phase-gate clearance with `clock_phase_gate_width` so left/right clearance only pays in expected swing windows
  - add metrics and auto-reset preservation for new reward/debug terms
- `training/envs/env_info.py`
  - add `cycle_start_forward_x` to persistent env info state for cycle displacement tracking
- `training/core/metrics_registry.py`
  - register `reward/step_length`, `reward/cycle_progress`, `debug/cycle_complete`, `debug/cycle_forward_delta`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.9"`
- keep fresh-run clocked actor contract:
  - `env.actor_obs_layout_id: wr_obs_v3`
  - `ppo.iterations: 60`
- add phase gate parameter:
  - `env.clock_phase_gate_width: 0.22`
- reduce undirected stepping reward:
  - `reward_weights.step_event: 0.005`
- make propulsion dominate:
  - `reward_weights.step_length: 1.0`
  - `reward_weights.step_length_target_base: 0.035`
  - `reward_weights.step_length_target_scale: 0.30`
  - `reward_weights.step_length_sigma: 0.03`
  - `reward_weights.cycle_progress: 1.2`
  - `reward_weights.cycle_progress_target_scale: 1.0`
  - `reward_weights.cycle_progress_sigma: 0.06`
- keep stability terms broadly aligned with `v0.15.6+` and only modestly retune gait auxiliaries.

### Training Intent
- Keep `wr_obs_v3`, but pay primarily for propulsion per touchdown and per cycle.
- Expected staged readout:
  - iter `20`: forward velocity should move off floor more clearly than `v0.15.8`
  - iter `40`: should visibly break stepping-in-place basin
  - iter `60`: if still near-zero forward velocity, stop and treat branch as failed

## [v0.15.8] - 2026-03-14: Clocked observation after `v0.15.7` exhausted reward-only tuning

### Summary
Analyzes the completed `v0.15.7` run `training/wandb/offline-run-20260314_193228-0y0j38w2` and concludes that reward-only tuning is no longer the main bottleneck. The policy again stayed upright, opened the stepping gate, and produced strong step rewards, yet still failed to generate useful forward motion. By iteration `60`, `debug/velocity_step_gate` remained near `0.96`, `reward/step_event` remained high, and `eval_clean/success_rate` returned to `1.0`, while `env/forward_velocity` was still only about `0.008 m/s` against `env/velocity_cmd ≈ 0.164`. That means the next justified lever is no longer another local reward retune, but an observation change: `wr_obs_v3` adds an explicit gait clock so the actor can coordinate periodic stepping and propulsion more directly.

### Results (v0.15.7)
- Run: `training/wandb/offline-run-20260314_193228-0y0j38w2`
- Checkpoints: `training/checkpoints/ppo_walking_v00157_20260314_193230-0y0j38w2`
- Verdict: reward-only tuning exhausted; still stepping without propulsion
- Best checkpoint: `checkpoint_60_7864320.pkl`
- Key metrics at iter `60`:
  - `env/forward_velocity: 0.008`
  - `env/velocity_cmd: 0.164`
  - `env/velocity_error: 0.166`
  - `env/success_rate: 0.951`
  - `eval_clean/success_rate: 1.000`
  - `term_pitch_frac: 0.049`
  - `reward/step_event: 0.353`
  - `reward/foot_place: 0.205`
  - `debug/velocity_step_gate: 0.961`

### Code Updates
- add policy-contract layout `wr_obs_v3`
- add 4-D gait clock features (`sin/cos` for left and right phase) to JAX and NumPy observation builders
- add `env.clock_stride_period_steps`
- wire clock features through the training env and policy visualization path

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.8"`
- switch actor observation layout:
  - `env.actor_obs_layout_id: wr_obs_v1 -> wr_obs_v3`
  - `env.clock_stride_period_steps: 36`
- keep the aggressive propulsion objective from `v0.15.7`
- increase exploration slightly for the fresh contract change:
  - `ppo.entropy_coef: 0.02 -> 0.03`
- keep the short hard-stop probe:
  - `ppo.iterations: 60`

### Training Intent
- This is a fresh run, not a resume-safe continuation. `wr_obs_v3` changes the policy contract.
- Give the actor an explicit periodic cue now that the reward path can already induce stable stepping.
- Use the same `20` / `40` / `60` readout, but expect a clearer answer faster than the reward-only branches.

## [v0.15.7] - 2026-03-14: Aggressive propulsion rebalance after `v0.15.6` learned to step in place

### Summary
Analyzes the completed `v0.15.6` run `training/wandb/offline-run-20260314_143238-eten6grc` and concludes that the structural stepping fix worked, but it converged to a new local minimum: stable stepping without useful forward propulsion. By iteration `80`, the run maintained excellent survival and low pitch failure while `debug/velocity_step_gate` stayed near `0.96` and stepping rewards stayed high, yet `env/forward_velocity` remained only about `0.003-0.006 m/s` against a commanded `0.124 m/s`. `v0.15.7` therefore becomes an aggressive propulsion probe: pay far less for arbitrary touchdown, pay much more for command-aligned forward foot placement, and increase raw forward-tracking pressure now that the policy can step without collapsing.

### Results (v0.15.6)
- Run: `training/wandb/offline-run-20260314_143238-eten6grc`
- Checkpoints: `training/checkpoints/ppo_walking_v00156_20260314_143239-eten6grc`
- Verdict: trapped in standing / stepping-in-place
- Best checkpoint: `checkpoint_80_10485760.pkl`
- Key metrics at iter `80`:
  - `env/forward_velocity: 0.003`
  - `env/velocity_cmd: 0.124`
  - `env/velocity_error: 0.130`
  - `env/success_rate: 0.985`
  - `eval_clean/success_rate: 1.000`
  - `term_pitch_frac: 0.015`
  - `reward/step_event: 0.341`
  - `reward/foot_place: 0.298`
  - `debug/velocity_step_gate: 0.964`

### Code Updates
- add `reward_weights.foot_place_k_cmd_vel`
- make foot-placement forward target depend on `velocity_cmd` as well as current `forward_vel`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.7"`
- keep the short hard-stop probe:
  - `ppo.iterations: 60`
- increase exploration / propulsion pressure:
  - `ppo.entropy_coef: 0.01 -> 0.02`
  - `env.min_velocity: 0.05 -> 0.08`
  - `env.max_velocity: 0.20 -> 0.25`
  - `reward_weights.tracking_lin_vel: 6.0 -> 8.0`
  - `reward_weights.forward_velocity_scale: 5.0 -> 6.0`
  - `reward_weights.velocity_standing_penalty: 0.4 -> 0.5`
  - `reward_weights.velocity_standing_threshold: 0.10 -> 0.12`
  - `reward_weights.velocity_cmd_min: 0.05 -> 0.08`
- aggressively rebalance stepping rewards toward propulsion:
  - `reward_weights.step_event: 0.30 -> 0.05`
  - `reward_weights.foot_place: 0.15 -> 0.35`
  - `reward_weights.foot_place_k_cmd_vel: 0.30`
  - `reward_weights.foot_place_k_fwd_vel: 0.05 -> 0.08`

### Training Intent
- Bias the policy away from stepping-in-place and toward net forward placement per step.
- Keep the run short; by `60` iterations it should be obvious whether the new objective converts stepping into propulsion.
- If this still fails, the next step should be a more invasive observation / curriculum change rather than another small reward retune.

## [v0.15.6] - 2026-03-14: Structural reward fix after `v0.15.5` fell back to standing

### Summary
Analyzes the `v0.15.5` probe run `training/wandb/offline-run-20260314_083700-h51mp8jx` and treats it as confirmation that config-only retuning is insufficient. By iteration `100`, the policy had largely recovered stable eval-clean survival, but forward velocity remained near zero (`~0.016 m/s`), meaning the run reverted to the stand-still basin instead of discovering a stepping gait. Historical review of `v0.15.3` to `v0.15.5` also showed that the basin is already obvious by around `80` iterations. `v0.15.6` therefore makes the smallest structural change that directly targets the failure mode and shortens the probe: forward reward is now gated by stepping engagement, pitch-rate is penalized explicitly, and the already-implemented stepping rewards are activated.

### Results (v0.15.5)
- Run: `training/wandb/offline-run-20260314_083700-h51mp8jx`
- Checkpoints: `training/checkpoints/ppo_walking_v00155_20260314_083702-h51mp8jx`
- Verdict: stable-standing fallback after a brief early lean/fall transient
- Early failure phase: around iter `20`, `env/forward_velocity ~ 0.036` with heavy pitch failure
- Latest inspected point: iter `100`
- Key metrics at iter `100`:
  - `env/forward_velocity: ~0.016`
  - `env/velocity_cmd: ~0.209`
  - `env/velocity_error: ~0.208`
  - `eval_clean/success_rate: 1.000`

### Code Updates
- add explicit `reward_weights.pitch_rate` support and include it in total reward
- gate the forward tracking reward by stepping evidence:
  - single-support contact diversity
  - swing clearance
  - liftoff/touchdown events
- add `debug/velocity_step_gate` and `reward/pitch_rate` to env metrics / registry
- preserve the new metrics through auto-reset

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.6"`
- keep the short structural probe format:
  - `ppo.iterations: 80`
  - `ppo.num_envs: 1024`
  - `ppo.rollout_steps: 128`
- lower the command curriculum to a shuffle-discovery band:
  - `env.min_velocity: 0.12 -> 0.05`
  - `env.max_velocity: 0.30 -> 0.20`
- reduce unconditional velocity forcing and make it contingent on stepping:
  - `reward_weights.tracking_lin_vel: 8.0 -> 6.0`
  - `reward_weights.forward_velocity_scale: 6.0 -> 5.0`
  - `reward_weights.velocity_step_gate: 1.0`
- explicitly penalize pitch-rate drift:
  - `reward_weights.pitch_rate: -0.25`
- activate the stepping reward path already present in the env:
  - `reward_weights.step_event: 0.30`
  - `reward_weights.foot_place: 0.15`
- rebalance the low-speed standing penalty for the new command band:
  - `reward_weights.velocity_standing_penalty: 0.5 -> 0.4`
  - `reward_weights.velocity_standing_threshold: 0.16 -> 0.10`
  - `reward_weights.velocity_cmd_min: 0.10 -> 0.05`

### Training Intent
- Keep the experiment incremental: fix the objective before introducing a clocked observation contract or larger batch geometry.
- Require real stepping engagement before the policy can earn most of its forward reward.
- Use `40` / `60` / `80` as the staged readout, with `80` as the hard go / no-go gate for whether Stage 1b now has a viable stepping path.

### Why Not Clock Yet
- The next lever after `v0.15.5` could have been a clocked observation layout, but that was intentionally deferred in `v0.15.6`.
- At that point the main unresolved question was still whether the reward objective itself could make the policy step, because prior runs were trapped between standing and lean-fall.
- A clock change would have been broader and harder to attribute:
  - new observation contract
  - env observation changes
  - policy / export / eval compatibility changes
- `v0.15.6` was therefore used as the smallest structural test first:
  - if stepping did not emerge, the next step would be clock
  - if stepping did emerge, the next bottleneck could be isolated more cleanly
- Result: `v0.15.6` did induce stable stepping, so the decision to delay clock was justified for diagnosis. The remaining failure after `v0.15.6` is no longer "no timing cue," but "stepping without propulsion."

## [v0.15.5] - 2026-03-14: Conservative gait-emergence reset after pitch-instability regression

### Summary
Analyzes the ongoing `v0.15.4` walking run `training/wandb/offline-run-20260313_212618-c7z4fbv8` and concludes that the policy is still in a non-deployable posture-exploit regime. Relative to `v0.15.3`, `v0.15.4` did increase forward velocity somewhat, but it did so by leaning harder and eventually collapsing eval-clean stability. By iteration `210`, forward velocity rose to about `0.088 m/s` against a commanded `0.273 m/s`, while `term_pitch_frac` rose above `0.81` and `eval_clean/success_rate` fell below `0.09`.

### Results (v0.15.4)
- Run: `training/wandb/offline-run-20260313_212618-c7z4fbv8`
- Checkpoints: `training/checkpoints/ppo_walking_v00154_20260313_212621-c7z4fbv8`
- Verdict: posture exploit with pitch-instability regression, not deployable walking
- Best stable phase: around iter `100`, but still near-zero usable forward speed
- Latest logged point: iter `210`
- Key metrics at iter `210`:
  - `env/forward_velocity: 0.088`
  - `env/velocity_cmd: 0.273`
  - `env/velocity_error: 0.235`
  - `term_pitch_frac: 0.817`
  - `env/success_rate: 0.183`
  - `eval_clean/success_rate: 0.084`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.5"`
- reset Stage 1b toward an earlier gait-emergence regime rather than increasing fixed reward pressure again
- make this a short diagnostic probe instead of a full 300-iteration commitment:
  - `ppo.iterations: 300 -> 120`
  - `checkpoints.interval: 25 -> 20`
- narrow the command curriculum further:
  - `env.min_velocity: 0.10 -> 0.12`
  - `env.max_velocity: 0.45 -> 0.30`
- add a bit more action damping to reduce pitch-diving:
  - `env.action_filter_alpha: 0.2 -> 0.3`
- keep velocity tracking primary, but reduce the fixed forcing from `v0.15.4`:
  - `reward_weights.tracking_lin_vel: 10.0 -> 8.0`
  - `reward_weights.forward_velocity_scale: 8.0 -> 6.0`
  - `reward_weights.velocity_standing_penalty: 0.8 -> 0.5`
  - `reward_weights.velocity_standing_threshold: 0.18 -> 0.16`
- lighten effort penalties aggressively, but only lighten posture / pre-collapse penalties moderately:
  - `reward_weights.orientation: -0.8 -> -0.4`
  - `reward_weights.angular_velocity: -0.12 -> -0.06`
  - `reward_weights.height_target: 0.2 -> 0.10`
  - `reward_weights.collapse_height: -0.4 -> -0.25`
  - `reward_weights.collapse_vz: -0.3 -> -0.15`
  - `reward_weights.torque: -0.0012 -> -0.0007`
  - `reward_weights.saturation: -0.1 -> -0.05`
  - `reward_weights.slip: -0.15 -> -0.05`

### Training Intent
- Start from scratch rather than resume from a checkpoint that already encodes a posture-exploit basin.
- Solve conservative nominal forward translation first.
- Use the first 120 iterations as an explicit go / no-go gate:
  - continue only if forward velocity rises materially without eval-clean pitch collapse
  - otherwise retune again rather than training longer into a bad solution
- Only after stable translation appears should the command range widen and stability / effort penalties tighten again.

## [v0.15.4] - 2026-03-13: Reward rebalance for walking velocity and pitch stability

### Summary
Analyzes the fresh-start walking run `training/wandb/offline-run-20260313_143658-oirn4qg9` and treats it as a non-deployable posture-exploit trajectory rather than a promising gait. By iteration `120`, forward velocity only reached about `0.04 m/s` against a commanded `0.35 m/s`, while pitch terminations rose to about `60%` and eval success collapsed. The fix is to pay substantially more for commanded forward motion, pay less for passive survival, and increase the cost of low-height / pitch-instability drift.

### Results (v0.15.3)
- Run: `training/wandb/offline-run-20260313_143658-oirn4qg9`
- Checkpoints: `training/checkpoints/ppo_walking_v00153_20260313_143700-oirn4qg9`
- Verdict: posture exploit / weak shuffle, not deployable walking
- Latest logged point: iter `120`
- Key metrics at iter `120`:
  - `env/forward_velocity: 0.040`
  - `env/velocity_cmd: 0.348`
  - `env/velocity_error: 0.326`
  - `term_pitch_frac: 0.601`
  - `env/success_rate: 0.399`
  - `eval_clean/success_rate: 0.314`

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.4"`
- narrow the command curriculum slightly so the robot first solves a more stable nominal gait:
  - `env.max_velocity: 0.6 -> 0.45`
- reduce flat survival reward:
  - `reward_weights.base_height: 0.1 -> 0.03`
- increase direct pressure to match the commanded speed:
  - `reward_weights.tracking_lin_vel: 8.0 -> 10.0`
  - `reward_weights.forward_velocity_scale: 6.0 -> 8.0`
  - `reward_weights.velocity_standing_penalty: 0.5 -> 0.8`
  - `reward_weights.velocity_standing_threshold: 0.15 -> 0.18`
- strengthen stability / pre-collapse shaping:
  - `reward_weights.orientation: -0.5 -> -0.8`
  - `reward_weights.angular_velocity: -0.1 -> -0.12`
  - `reward_weights.height_target: 0.1 -> 0.2`
  - `reward_weights.collapse_height: -0.2 -> -0.4`
  - `reward_weights.collapse_vz: -0.2 -> -0.3`
- modestly increase effort penalties to discourage the pitchy high-torque solution:
  - `reward_weights.torque: -0.001 -> -0.0012`
  - `reward_weights.saturation: -0.08 -> -0.1`
- slightly relax slip cost so early stepping is not overconstrained:
  - `reward_weights.slip: -0.2 -> -0.15`

### Training Intent
- Force a larger reward gap between standing still and matching the forward command.
- Preserve Stage 1 PPO-only walking while making pitch stability a hard requirement.
- Stop using survival-only eval signals as evidence of walking quality; walking checkpoint selection should prioritize velocity/error plus stability.

## [v0.15.3] - 2026-03-13: Fresh-start walking after standing-resume failure

### Summary
Promotes the walking branch from "resume from standing" to a true fresh-run Stage 1b experiment. `v0.15.2` showed that the standing checkpoint remained too strong a local minimum: forward velocity stayed near zero while pitch failures and torque climbed. That means the standing warm start is blocking gait discovery rather than helping it.

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.3"`
- start from scratch:
  - do **not** resume from the standing checkpoint
- keep controllers off:
  - `env.base_ctrl_enabled: false`
  - `env.fsm_enabled: false`
- keep PPO-only Stage 1b setup:
  - `amp.enabled: false`
  - `env.actor_obs_layout_id: wr_obs_v1`
- reduce action lag for gait emergence:
  - `env.action_filter_alpha: 0.2`
- switch back to standard walking batch geometry:
  - `ppo.num_envs: 1024`
  - `ppo.rollout_steps: 128`
- use standard PPO learning rate for fresh training:
  - `ppo.learning_rate: 3e-4`
- keep the stronger walking incentive from `v0.15.2`:
  - high velocity tracking weight
  - strong standing penalty
  - low flat healthy bonus
  - rollback disabled

### Training Intent
- Learn walking directly instead of trying to deform a standing policy into a gait.
- Use the first `80-120` iterations as the main decision window for forward-velocity emergence.
- Continue to `300` iterations only if the policy shows real translational motion rather than posture/torque exploitation.

## [v0.15.2] - 2026-03-13: Walking breakout from standing local minimum

### Summary
Tightens the Stage 1b walking config after `v0.15.1` stayed near zero forward velocity while preserving perfect standing success. The root cause was that the warm-started standing policy could still earn good reward by standing still: forward tracking was too soft, the standing penalty was too weak, and the `base_height` term was effectively acting as a flat healthy bonus.

### Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.2"`
- keep warm-start compatibility with the standing checkpoint:
  - `env.actor_obs_layout_id: wr_obs_v1`
  - `env.action_filter_alpha: 0.6`
- increase pressure to move:
  - `reward_weights.tracking_lin_vel: 8.0`
  - `reward_weights.forward_velocity_scale: 6.0`
  - `reward_weights.velocity_standing_penalty: 0.5`
  - `reward_weights.velocity_standing_threshold: 0.15`
  - `reward_weights.velocity_cmd_min: 0.1`
- reduce reward for "just stay healthy":
  - `reward_weights.base_height: 0.1`
  - `reward_weights.height_target: 0.1`
- relax posture-only costs that can block gait emergence:
  - `reward_weights.orientation: -0.5`
  - `reward_weights.angular_velocity: -0.1`
  - `reward_weights.slip: -0.2`
- slightly favor alternating support:
  - `reward_weights.gait_periodicity: 0.15`
- make PPO updates more willing to leave the standing basin:
  - `ppo.learning_rate: 2e-4`
  - `ppo.entropy_coef: 0.01`
  - `ppo.iterations: 300`
- disable rollback for this transition run:
  - `ppo.rollback.enabled: false`

### Training Intent
- Break the warm-started standing policy out of the stand-still solution.
- Accept some temporary drop in standing-style success if it produces actual locomotion.
- Use the first `80-120` iterations as the main decision window for whether forward motion is emerging.

## [v0.15.1] - 2026-03-12: Freeze best standing bundle and prepare Stage 1b walking

### Summary
Freezes the best standing-push pure-PPO policy from `v0.14.6` into a runtime bundle, then switches the default walking config back to the repo's main execution plan: Stage 1b PPO-only walking with no AMP and no stepping controller.

### Frozen Standing Bundle
- Source checkpoint:
  - `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl`
- Matching historical config:
  - `training/configs/ppo_standing_push.yaml` @ commit `d32d11c`
- Exported runtime bundle:
  - `runtime/bundles/standing_push_v0.14.6_ckpt250`

### Walking Config Updates (`training/configs/ppo_walking.yaml`)
- bump to `version: "0.15.1"`
- switch back to Stage 1b walking intent:
  - PPO only
  - AMP disabled
  - `env.base_ctrl_enabled: false`
  - `env.fsm_enabled: false`
- move walking config to current asset variant:
  - `env.assets_root: assets/v2`
- keep warm-start compatibility with the frozen standing checkpoint:
  - `env.actor_obs_layout_id: wr_obs_v1`
  - `env.action_filter_alpha: 0.6`
- use a conservative commanded walking range:
  - `env.min_velocity: 0.1`
  - `env.max_velocity: 0.6`
- add modern PPO eval / rollback settings and keep the standing batch geometry for resume stability:
  - `ppo.num_envs: 512`
  - `ppo.rollout_steps: 256`
  - `ppo.iterations: 400`
  - `ppo.eval.enabled: true`
  - `ppo.rollback.enabled: true`

### Training Intent
- Warm-start walking from the best pure standing-push PPO checkpoint instead of continuing the disproven FSM branch.
- Learn nominal foot placement and weight transfer through dense walking rewards rather than sparse recovery events.
- Use this as the Stage 1b baseline before any later robustness or AMP work.

### Run Command
```bash
uv run python training/train.py \
  --config training/configs/ppo_walking.yaml \
  --resume training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl
```

## [v0.14.9] - 2026-03-11: FSM guide on top of actor capture-point observation

### Summary
Reintroduces the M3 foot-placement FSM after both information-first branches plateaued below the `v0.14.6` pure-PPO baseline. `v0.14.7` tested critic-only privileged information and `v0.14.8` tested actor-side capture-point information; neither beat the best pure-PPO result. The next step is to test whether the existing actor signal plus a guide-only FSM can turn step intent into more effective recovery.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.9"`
- keep actor capture-point observation:
  - `env.actor_obs_layout_id: wr_obs_v2`
- re-enable FSM:
  - `env.fsm_enabled: true`
- keep M2 disabled:
  - `env.base_ctrl_enabled: false`
- keep full residual override in every phase:
  - `env.fsm_resid_scale_stance: 1.00`
  - `env.fsm_resid_scale_swing: 1.00`
  - `env.fsm_resid_scale_recover: 1.00`
- keep the calibrated disturbance regime:
  - `env.push_force_min/max: 10.0`
  - `env.push_duration_steps: 10`
- keep critic symmetric:
  - `ppo.critic_privileged_enabled: false`

### Resume Point
- Resume from the best `v0.14.8` checkpoint:
  - `training/checkpoints/ppo_standing_push_v00148_20260310_203611-l1g2jb6r/checkpoint_200_26214400.pkl`
- This is resume-safe because `wr_obs_v2` and `action_filter_alpha` are unchanged from `v0.14.8`.

### Training Intent
- Test whether a guide-only FSM improves `eval_push/success_rate` beyond the `v0.14.6` pure-PPO baseline (`59.59%` at 200 iters, `60.84%` extended).
- Test whether the FSM reduces the dominant `term_height_low_frac` failure mode rather than merely increasing step-like activity.
- Preserve the lower pitch-failure and lower torque-stress profile that `v0.14.8` achieved.

### Interpretation Rule
- If `v0.14.9` beats `v0.14.6`, the guide-FSM branch is justified and can be tuned further.
- If `v0.14.9` still fails to beat `v0.14.6`, the current FSM implementation is not adding enough value and should not be tightened further without improving execution accuracy first.

### Results (v0.14.9)
- Run: `training/wandb/offline-run-20260311_143401-qcfon7tw`
- Checkpoints: `training/checkpoints/staged_v0149/run_20260311_095042/ppo_standing_push_v00149_20260311_143403-qcfon7tw`
- Best checkpoint (eval_push/): `training/checkpoints/staged_v0149/run_20260311_095042/ppo_standing_push_v00149_20260311_143403-qcfon7tw/checkpoint_390_51118080.pkl`
- Best @ iter 390: `eval_push/success_rate=38.06%`, `eval_push/episode_length=308.1`
- Train @ iter 390: `success=42.43%`, `ep_len=327.7`
- Clean eval @ iter 390: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- FSM verdict: `weakly engaged`
- FSM touchdown style: `timeout-driven`
- FSM upright recovery: `good`

| Signal | Value |
|---|---:|
| term_height_low_frac | 57.57% |
| term_pitch_frac | 0.00% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 55.89% |
| debug/torque_sat_frac | 0.60% |
| ppo/approx_kl | 0.0034 |
| ppo/clip_fraction | 0.0727 |
| debug/bc_phase | 0.258 |
| debug/bc_phase_ticks | 98.982 |
| fsm/swing_occupancy | 3.99% |
| reward/step_event | 0.0012 |
| reward/foot_place | 0.0007 |
| debug/need_step | 0.1637 |
| reward/posture | 0.7528 |

### Diagnosis
- `v0.14.9` failed decisively. It underperformed not only the best pure-PPO baseline (`v0.14.6`: `59.59%` at 200 iters, `60.84%` extended), but also the actor-information run it resumed from (`v0.14.8`: `55.41%`).
- The guide-FSM did not convert need-step pressure into meaningful stepping. `debug/need_step` stayed elevated, but swing occupancy was only `3.99%`, and both `reward/step_event` and `reward/foot_place` collapsed toward zero.
- The controller remained mostly `timeout-driven`, which is the same mechanism failure seen in earlier M3 runs. That strongly suggests the current swing execution / touchdown accuracy is still too weak for the FSM to help, even when PPO retains full override authority.
- Pitch and roll failures stayed at zero, and torque stress remained reasonable. The regression is specifically failed push recovery through `term_height_low`, not general training instability.

### Next Step
- Do **not** continue tuning the current FSM branch. This run is strong evidence that the present FSM implementation is not adding useful recovery value.
- Revert to the best pure-PPO baseline for deployment / continuation:
  - `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl`
- If standing-push stepping research continues later, only revisit FSM after improving execution accuracy first, for example:
  - replace joint-heuristic swing tracking with geometrically correct leg IK, or
  - expose desired foot targets and let PPO learn the leg coordination directly
- Until then, treat the current FSM as disproven for this task setting rather than something that needs more threshold tuning.

## [v0.14.8] - 2026-03-10: Actor-information run with capture-point observation

### Summary
Prepares the next information-first standing-push run after v0.14.7 failed to beat the plain PPO baseline. The critic-only privileged path increased stepping-related rewards but did not improve survival. The next step moves the information to the actor by extending the policy observation contract with a 2-D heading-local capture-point error.

### Motivation
- v0.14.7 showed higher `reward/step_event` and `reward/foot_place` than v0.14.6, so the policy is willing to step more.
- Those extra step events still did not reduce `term_height_low`, which suggests the actor still lacks direct information about where stepping would help.
- Before reintroducing FSM, test whether explicit actor-side stepping cues are enough.

### Code Updates
- Add `env.actor_obs_layout_id` to the config/runtime schema.
- Add a new policy-contract layout `wr_obs_v2`.
- Extend actor observations with `capture_point_error` (2-D heading-local approximate CoM-minus-capture-point offset).
- Pass the selected layout through training startup, env creation, visualization, and export policy-spec generation.

### Observation Contract
- `wr_obs_v1`: existing actor observation layout (resume-safe within prior milestones).
- `wr_obs_v2`: `wr_obs_v1` plus:
  - `capture_point_error[0]`
  - `capture_point_error[1]`

`wr_obs_v2` is a contract-breaking change and requires a fresh run.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.8"`
- switch actor layout:
  - `env.actor_obs_layout_id: wr_obs_v2`
- return critic to symmetric input for isolation:
  - `ppo.critic_privileged_enabled: false`
- keep controllers disabled and disturbance regime unchanged:
  - `env.fsm_enabled: false`
  - `env.base_ctrl_enabled: false`
  - `env.push_force_min/max: 10.0`
  - `env.push_duration_steps: 10`

### Training Intent
- Test whether actor-side capture-point information improves step quality and recovery quality.
- Target outcome: beat the v0.14.6 baseline on `eval_push/success_rate` while reducing `term_height_low`.
- If this still fails to beat v0.14.6, only then consider reintroducing FSM as a guide.

### Run Notes
- Start a fresh run. Do not resume from v0.14.7 or earlier checkpoints because `wr_obs_v2` changes `policy_spec_hash`.

### Results (v0.14.8)
- Run: `training/wandb/offline-run-20260310_203609-l1g2jb6r`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00148_20260310_203611-l1g2jb6r`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00148_20260310_203611-l1g2jb6r/checkpoint_200_26214400.pkl`
- Best @ iter 200: `eval_push/success_rate=55.41%`, `eval_push/episode_length=366.7`
- Train @ iter 200: `success=55.52%`, `ep_len=366.6`
- Clean eval @ iter 200: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- Controller status: disabled (`env.fsm_enabled=false`, `env.base_ctrl_enabled=false`)

| Signal | Value |
|---|---:|
| term_height_low_frac | 44.48% |
| term_pitch_frac | 6.40% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 57.50% |
| debug/torque_sat_frac | 0.71% |
| ppo/approx_kl | 0.0039 |
| ppo/clip_fraction | 0.0993 |
| reward/step_event | 0.0250 |
| reward/foot_place | 0.0212 |
| debug/need_step | 0.1286 |
| reward/posture | 0.9631 |

### Diagnosis
- Actor-side capture-point information did **not** beat the best plain PPO baseline. `eval_push/success_rate` reached `55.41%`, still below `59.59%` in v0.14.6 and `60.84%` in the later pure-PPO extension.
- Compared with v0.14.7, `v0.14.8` substantially reduced `term_pitch_frac` and torque stress, which means the new actor signal improved recovery quality and action quality.
- But those gains did not translate into higher push success. `reward/step_event` and `reward/foot_place` fell back toward the v0.14.6 range, and `term_height_low` remained the dominant failure mode.
- Clean standing stayed perfect, so the limitation is still specifically push recovery under the calibrated hard-push regime.
- Taken together with v0.14.7, the information-first branch has now been tested on both sides:
  - critic-only information (`v0.14.7`)
  - actor-side capture-point information (`v0.14.8`)
  Neither beat the simpler v0.14.6 baseline.

### Updated Next Step
- The next run should move to the **FSM-as-guide** branch.
- Recommended `v0.14.9` plan:
  - keep `wr_obs_v2` actor observations
  - re-enable `env.fsm_enabled: true`
  - keep `env.base_ctrl_enabled: false`
  - keep full residual override:
    - `env.fsm_resid_scale_stance: 1.00`
    - `env.fsm_resid_scale_swing: 1.00`
    - `env.fsm_resid_scale_recover: 1.00`
  - keep the same `10N x 10` push regime
- Resume from the best v0.14.8 checkpoint if you want the fastest test of whether a guide-FSM can add value on top of the current policy. This is resume-safe as long as `wr_obs_v2` and `action_filter_alpha` stay unchanged.
- The success criterion for that next run is straightforward: beat the v0.14.6 baseline on `eval_push/success_rate` without giving up the lower pitch-failure and lower torque-stress profile that v0.14.8 achieved.

---

## [v0.14.7] - 2026-03-10: Information-first standing push run with privileged critic

### Summary
Prepares the next standing-push run as an information-first follow-up to v0.14.6. Pure PPO at `10N x 10` improved over the bracing baseline, but the extended run plateaued around `60.84%` push success and still failed mainly by `term_height_low`. The next experiment keeps FSM off and actor observations unchanged, while giving the critic sim-only disturbance state for better credit assignment.

### Motivation
- v0.14.6 showed non-zero `reward/step_event` and `reward/foot_place`, so stepping pressure is present.
- The extension did not convert that extra stepping activity into sustained survival gains.
- The next missing ingredient is value-function information, not more controller authority.

### Code Updates
- Add `ppo.critic_privileged_enabled` to the training config/runtime schema.
- Add a fixed-size privileged critic vector to `WildRobotInfo` and rollout storage.
- Feed the critic a separate sim-only input while keeping the actor on the existing policy-contract observation.
- Save `critic_privileged_enabled` in checkpoints and reject incompatible resumes.

### Privileged Critic Features
The critic now sees a 12-D sim-only vector:
- heading-local linear velocity `(vx, vy, vz)`
- heading-local angular velocity `(wx, wy, wz)`
- root attitude `(roll, pitch)`
- root height
- active push force `(fx, fy)`
- push-active flag

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.7"`
- keep pure PPO / no controller assistance:
  - `env.fsm_enabled: false`
  - `env.base_ctrl_enabled: false`
- keep the calibrated stepping regime:
  - `env.push_force_min/max: 10.0`
  - `env.push_duration_steps: 10`
- enable information-first critic:
  - `ppo.critic_privileged_enabled: true`

### Training Intent
- Test whether better critic information can turn emerging step events into more useful recovery behavior.
- Keep the actor observation contract unchanged so runtime/export interfaces stay stable.
- If this run still plateaus, only then consider reintroducing FSM as a guide with full residual override.

### Run Notes
- Start a fresh run. This is not a resume-safe follow-up to v0.14.6 because the critic network input shape changes.

### Results (v0.14.7)
- Run: `training/wandb/offline-run-20260310_131556-xsg0sih9`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00147_20260310_131558-xsg0sih9`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00147_20260310_131558-xsg0sih9/checkpoint_200_26214400.pkl`
- Best @ iter 200: `eval_push/success_rate=55.86%`, `eval_push/episode_length=367.6`
- Train @ iter 200: `success=44.72%`, `ep_len=342.8`
- Clean eval @ iter 200: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- Controller status: disabled (`env.fsm_enabled=false`, `env.base_ctrl_enabled=false`)

| Signal | Value |
|---|---:|
| term_height_low_frac | 54.77% |
| term_pitch_frac | 21.61% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 62.48% |
| debug/torque_sat_frac | 0.94% |
| ppo/approx_kl | 0.0037 |
| ppo/clip_fraction | 0.0865 |
| reward/step_event | 0.0466 |
| reward/foot_place | 0.0382 |
| debug/need_step | 0.1888 |
| reward/posture | 0.9443 |

### Diagnosis
- The critic-only information-first run did **not** beat the plain PPO baseline. `eval_push/success_rate` regressed from `59.59%` in v0.14.6 and `60.84%` in the v0.14.6 extension to `55.86%` here.
- The best checkpoint was the final checkpoint, so v0.14.7 was still improving at the end of 200 iterations, but it was improving toward a worse ceiling than the simpler baseline.
- `reward/step_event` and `reward/foot_place` both increased relative to v0.14.6, which suggests the privileged critic did help push the policy toward more stepping-like behavior.
- That extra stepping activity did **not** translate into better survival. `term_height_low` remains dominant and `term_pitch_frac` rose materially, which points to poor step quality / recovery quality rather than an absence of step attempts.
- Clean standing remained perfect, so the regression is specific to push recovery, not a general loss of balance.

### Updated Next Step
- Do **not** enable FSM yet. This run does not justify turning controller authority back on.
- The next step should stay in the information-first branch, but move the information to the **actor**, not only the critic:
  - add actor-side capture-point or CoM-velocity error features
  - start a fresh run because this changes the policy observation contract
  - keep `env.fsm_enabled=false` and `env.base_ctrl_enabled=false`
- The goal of that next run is to test whether explicit actor-side stepping information can convert the now-higher step-event/foot-place activity into lower `term_height_low` and higher `eval_push/success_rate`.
- Only if that actor-information run also fails to beat the v0.14.6 baseline should FSM come back, and then only as a guide with full residual override.

---

## [v0.14.6] - 2026-03-08: Pure PPO next run at calibrated stepping regime

### Summary
Replaces the default standing-push baseline with a pure-PPO run at the calibrated stepping regime. The v0.13.9 force sweep showed a bracing ceiling of approximately `9N` for `10` control steps, so the next run should increase disturbance difficulty instead of reducing policy authority. FSM and M2 base control are disabled by default; reward-only stepping terms remain enabled.

### Calibration Result
- Baseline checkpoint: `runtime/bundles/standing_push_v0.13.9_ckpt310/checkpoint.pkl`
- Matching training source checkpoint: `training/checkpoints/ppo_standing_push_auto_confirm_20260305_040129_v00139_20260305_040133-j3l4fc1e/checkpoint_310_40632320.pkl`
- Matching historical config: `training/configs/ppo_standing_push.yaml` @ commit `3a1d1ad`
- Force-sweep result:
  - `8N x 10` -> `74.6%`, `ep_len=431.8`
  - `9N x 10` -> `58.0%`, `ep_len=382.9`
  - `10N x 10` -> `47.3%`, `ep_len=346.7`
  - estimated bracing ceiling: `~9N`
- Dominant failure mode above the ceiling: `term_height_low`

### Diagnosis
- The calibrated stepping regime begins around `9N x 10`, not at some much higher force range.
- `9N x 15` remains a useful harder stress test, but it should not be combined with reduced residual authority while the FSM is still inaccurate.
- The next question is whether pure PPO, given full authority and pushes above the bracing ceiling, will discover stepping on its own.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.6"`
- disable controller assistance by default:
  - `env.fsm_enabled: false`
  - `env.base_ctrl_enabled: false`
- keep full residual override authority if FSM is later re-enabled:
  - `env.fsm_resid_scale_stance: 1.00`
  - `env.fsm_resid_scale_swing: 1.00`
  - `env.fsm_resid_scale_recover: 1.00`
- move training pushes to the calibrated stepping regime:
  - `env.push_duration_steps: 10` (from `15`)
  - `env.push_force_min: 10.0` (from `3.0`)
  - `env.push_force_max: 10.0` (from `9.0`)

### Results (v0.14.6)
- Staged run root: `training/checkpoints/staged_v0146/run_20260308_234828`
- Final W&B segment: `training/wandb/offline-run-20260309_052819-npc17xbk`
- Final checkpoint dir: `training/checkpoints/staged_v0146/run_20260308_234828/ppo_standing_push_v00146_20260309_052820-npc17xbk`
- Best checkpoint (eval_push/): `training/checkpoints/staged_v0146/run_20260308_234828/ppo_standing_push_v00146_20260309_052820-npc17xbk/checkpoint_200_26214400.pkl`
- Best @ iter 200: `eval_push/success_rate=59.59%`, `eval_push/episode_length=379.0`
- Train @ iter 200: `success=57.27%`, `ep_len=377.1`
- Clean eval @ iter 200: `eval_clean/success_rate=100.00%`, `eval_clean/episode_length=500.0`
- Controller status: disabled (`env.fsm_enabled=false`, `env.base_ctrl_enabled=false`)

| Signal | Value |
|---|---:|
| term_height_low_frac | 42.73% |
| term_pitch_frac | 4.45% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 69.99% |
| debug/torque_sat_frac | 1.70% |
| ppo/approx_kl | 0.0039 |
| ppo/clip_fraction | 0.0923 |
| reward/step_event | 0.0254 |
| reward/foot_place | 0.0222 |
| debug/need_step | 0.1252 |
| reward/posture | 0.9711 |

### Diagnosis
- Pure PPO improved materially over the calibrated `10N x 10` bracing baseline (`47.3% -> 59.6%`), so the run did create meaningful pressure toward recovery strategies beyond in-place bracing.
- The dominant failure mode remains `term_height_low`; pitch failures are secondary and roll is negligible.
- `reward/step_event` and `reward/foot_place` are small but clearly non-zero, which suggests reward-only stepping behavior is emerging rather than the policy relying purely on the old crouch strategy.
- The best checkpoint is the final checkpoint, and eval_push improved through the last segment (`58.2% -> 59.0% -> 59.3% -> 59.6%` from iters `160 -> 170 -> 190 -> 200`), so the run does not look plateaued yet.
- Torque stress rose compared with the softer v0.13.x baselines but remains within a manageable range.

### Next Step
- Do **not** enable FSM yet. The v0.14.6 result does not justify reintroducing controller authority because pure PPO is still improving at the calibrated stepping regime.
- First extend the same pure-PPO run for another `100-150` iterations from `checkpoint_200_26214400.pkl` and check whether `eval_push/success_rate`, `reward/step_event`, and `reward/foot_place` continue to rise.
- If that extended pure-PPO run plateaus below a useful level, keep FSM off and add **information first**:
  - asymmetric actor-critic / privileged critic
  - capture-point or CoM-velocity information without reducing actor authority
- Only after a plateaued information-first run should FSM be re-enabled, and when it comes back it should be as a **guide** with full residual override (`resid_scale=1.0`), not as a controller that constrains the policy.

### Extended Results (+150 iter resume)
- Resume run: `training/wandb/offline-run-20260309_195021-oij0cbcc`
- Resume checkpoints: `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc`
- Best checkpoint (eval_push/): `training/checkpoints/staged_v0146_resume/ppo_standing_push_v00146_20260309_195023-oij0cbcc/checkpoint_250_32768000.pkl`
- Best @ iter 250: `eval_push/success_rate=60.84%`, `eval_push/episode_length=383.1`
- Train @ iter 250: `success=56.86%`, `ep_len=368.3`
- Final @ iter 350: `eval_push/success_rate=54.05%`, `eval_push/episode_length=364.5`
- Clean eval remained perfect throughout the extension (`eval_clean/success_rate=100%`)

| Signal | Best @250 | Final @350 |
|---|---:|---:|
| term_height_low_frac | 39.16% | 45.95% |
| term_pitch_frac | 10.49% | 7.43% |
| tracking/max_torque | 68.00% | 66.58% |
| debug/torque_sat_frac | 1.51% | 1.36% |
| ppo/approx_kl | 0.0045 | 0.0027 |
| ppo/clip_fraction | 0.1047 | 0.0485 |
| reward/step_event | 0.0335 | 0.0369 |
| reward/foot_place | 0.0286 | 0.0311 |
| debug/need_step | 0.1523 | 0.1574 |
| reward/posture | 0.9640 | 0.9614 |

### Updated Diagnosis
- The extra `150` iterations produced only a small peak gain over the iter-200 checkpoint (`59.59% -> 60.84%` at iter `250`), then regressed for the remainder of the run.
- `reward/step_event` and `reward/foot_place` continued to rise, but that did **not** translate into sustained push-survival gains. The policy appears to be stepping more often, but not yet stepping well enough to reduce collapse reliably.
- `term_height_low` remains the dominant failure mode, so the mechanism gap is still recovery quality after disturbance, not merely triggering a foot touchdown.
- The run looks plateaued/noisy now. The best checkpoint arrived early in the extension and the later checkpoints fell back toward the pre-resume level.
- A rollback/LR-backoff signal appeared during the late run, which is another sign that simply running longer on the same setup is no longer buying much.

### Updated Next Step
- Do **not** enable FSM yet. Pure PPO beat the bracing baseline, but this extension does not support turning controller authority back on.
- The next run should be **information-first, still with FSM off**:
  - asymmetric actor-critic / privileged critic
  - add capture-point / CoM-velocity / push-vector information to the critic first
  - keep the actor contract unchanged if resume-safety is still desired
- The goal of that run is to see whether better credit assignment can convert the now-nonzero step events into useful foot placement and lower `term_height_low`.
- Only if the information-first run also plateaus should FSM be enabled, and then only as a **guide** with full residual override (`resid_scale=1.0`) and no residual handicapping.

---

## [v0.14.5] - 2026-03-08: M3 committed stepping after phase-aware FSM fixes

### Summary
Promotes the latest controller/debug fixes into the next training baseline. The major change is that M3 stepping signals are now phase-aware and `need_step` no longer shuts off near a height-collapse event. With those fixes in place, the next run should optimize for more useful catch steps: wider lateral placement, stronger swing tracking, and cleaner touchdown-driven step rewards.

### Results (v0.14.4, reinterpreted after analyzer/controller fixes)
- Run: `training/wandb/offline-run-20260308_095026-0vvxftyt`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00144_20260308_095028-0vvxftyt`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00144_20260308_095028-0vvxftyt/checkpoint_260_34078720.pkl`
- Best @ iter 260: `eval_push/success_rate=56.64%`, `eval_push/episode_length=363.1`
- Train @ iter 260: `success=61.10%`, `ep_len=386.5`
- FSM verdict: `weakly engaged`
- FSM touchdown style: `timeout-driven`
- FSM upright recovery: `good`

| Signal | Value |
|---|---:|
| term_height_low_frac | 38.90% |
| term_pitch_frac | 0.00% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 45.67% |
| debug/torque_sat_frac | 0.18% |
| ppo/approx_kl | 0.0038 |
| ppo/clip_fraction | 0.0803 |
| debug/bc_phase | 0.139 |
| fsm/swing_occupancy | 13.86% |
| reward/step_event | 0.0070 |
| reward/foot_place | 0.0048 |
| debug/need_step | 0.1223 |
| reward/posture | 0.7603 |

### Diagnosis
- The earlier `0%` swing conclusion was partly a metrics bug; M3 is weakly engaging, not completely idle.
- The main remaining mechanism gap is step quality, not step triggering alone.
- Posture recovery remains decent, but step events are still mostly timeout-driven and too weak to improve hard-push survival.
- The next iteration should favor more decisive lateral catch steps and stronger swing tracking, not just lower thresholds.

### Code Updates
- `need_step` no longer hard-zeros below `min_height`, so the FSM can still attempt a rescue step near collapse.
- `reward/step_event`, `reward/foot_place`, and touchdown debug signals are now tied to the active swing foot during `SWING`.
- New debug metrics:
  - `debug/bc_in_swing`
  - `debug/bc_in_recover`
- FSM analyzer scripts now use explicit swing occupancy instead of misreading averaged `debug/bc_phase`.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.5"`
- make catch steps wider and less constrained:
  - `env.fsm_y_nominal_m: 0.125` (from `0.115`)
  - `env.fsm_k_lat_vel: 0.20` (from `0.15`)
  - `env.fsm_k_roll: 0.12` (from `0.10`)
  - `env.fsm_y_step_outer_m: 0.22` (from `0.20`)
  - `env.fsm_step_max_delta_m: 0.16` (from `0.12`)
- make swing execution more assertive:
  - `env.fsm_swing_height_m: 0.05` (from `0.04`)
  - `env.fsm_swing_x_to_hip_pitch: 0.35` (from `0.30`)
  - `env.fsm_swing_y_to_hip_roll: 0.40` (from `0.30`)
  - `env.fsm_swing_z_to_knee: 0.55` (from `0.40`)
  - `env.fsm_swing_z_to_ankle: 0.25` (from `0.20`)
- strengthen phase-aware stepping rewards moderately:
  - `reward.step_event: 0.25` (from `0.20`)
  - `reward.foot_place: 0.65` (from `0.50`)
  - `reward.foot_place_sigma: 0.14` (from `0.12`)

### Training Intent
This release is meant to answer the next M3 question directly:
- can the controller turn weak FSM engagement into useful touchdown-driven catch steps?
- can wider lateral placement reduce collapse without giving up the current posture recovery gains?
- can hard-push survival recover while maintaining low torque stress?

### Next-Run Success Signals
- `debug/bc_in_swing` rises above the v0.14.4 baseline
- `reward/step_event` and `reward/foot_place` both rise materially
- `debug/bc_phase_ticks` trends down from timeout-like values
- `eval_push/success_rate` recovers toward or above the v0.14.3 result
- `term_height_low_frac` falls without a major increase in torque saturation

---

## [v0.14.4] - 2026-03-08: M3 engagement tuning after first hard-push run

### Summary
Responds to the first real M3 standing-push run, which showed decent posture recovery but almost no meaningful FSM stepping. The next iteration is tuned to make the step state machine engage earlier and to reduce the policy's ability to survive hard pushes by residual-only bracing or waist/arm compensation.

### Results (v0.14.3)
- Run: `training/wandb/offline-run-20260307_235137-un260w81`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00143_20260307_235139-un260w81`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00143_20260307_235139-un260w81/checkpoint_240_31457280.pkl`
- Best @ iter 240: `eval_push/success_rate=63.77%`, `eval_push/episode_length=386.1`
- Train @ iter 240: `success=67.36%`, `ep_len=405.8`
- FSM verdict: `not meaningfully engaged`
- FSM touchdown style: `timeout-driven`
- FSM upright recovery: `good`

| Signal | Value |
|---|---:|
| term_height_low_frac | 32.64% |
| term_pitch_frac | 0.00% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 55.97% |
| debug/torque_sat_frac | 0.68% |
| ppo/approx_kl | 0.0038 |
| ppo/clip_fraction | 0.0912 |
| debug/bc_phase | 0.049 |
| debug/bc_phase_ticks | 204.847 |
| fsm/swing_occupancy | 0.00% |
| reward/step_event | 0.0071 |
| reward/foot_place | 0.0046 |
| debug/need_step | 0.1070 |
| reward/posture | 0.7575 |

### Diagnosis
- M3 posture recovery is working better than M3 stepping.
- The dominant failure remains collapse (`term_height_low_frac`), not pitch/roll loss.
- The residual policy still appears able to survive too much of the push without committing to useful `SWING` phases.
- Waist/arm compensation is likely helping mask poor foot placement rather than supporting it.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.4"`
- tune the FSM to engage earlier under hard pushes:
  - `env.fsm_trigger_threshold: 0.30` (from `0.45`)
  - `env.fsm_recover_threshold: 0.15` (from `0.20`)
  - `env.fsm_trigger_hold_ticks: 1` (from `2`)
  - `env.fsm_swing_timeout_ticks: 10` (from `12`)
- reduce residual masking so the step controller must contribute:
  - `env.fsm_resid_scale_stance: 0.55` (from `0.75`)
  - `env.fsm_resid_scale_swing: 0.45` (from `0.60`)
  - `env.fsm_resid_scale_recover: 0.50` (from `0.70`)
- weaken arm compensation so foot placement, not upper-body damping, carries recovery:
  - `env.fsm_arm_need_step_threshold: 0.45` (from `0.35`)
  - `env.fsm_arm_k_roll: 0.06` (from `0.10`)
  - `env.fsm_arm_k_roll_rate: 0.03` (from `0.05`)
  - `env.fsm_arm_k_pitch_rate: 0.02` (from `0.03`)
  - `env.fsm_arm_max_delta_rad: 0.18` (from `0.25`)

### Training Intent
This release is meant to answer a narrower second M3 question:
- can the FSM be made to engage reliably under hard pushes?
- can step events become touchdown-driven instead of timeout-driven?
- can the robot keep the current posture recovery gains while using the feet more decisively?

### Next-Run Success Signals
- `debug/bc_phase` shows clear `SWING` occupancy during disturbed episodes
- `reward/step_event` rises materially above the v0.14.3 baseline
- `reward/foot_place` rises with step events
- `debug/bc_phase_ticks` shortens away from timeout-like values
- `eval_push/success_rate` improves from the v0.14.3 result without a major torque spike

---

## [v0.14.3] - 2026-03-08: M3 training-ready standing-push baseline

### Summary
Promotes the standing-push baseline from "M3 code present" to "M3 training-ready". The default standing-push config now trains with the foot-placement FSM enabled, M2 disabled, hard pushes restored, and residual authority reduced enough that the FSM must do real recovery work instead of being masked by the policy residual.

### Config Updates (`training/configs/ppo_standing_push.yaml`)
- bump to `version: "0.14.3"`
- enable M3 by default:
  - `env.fsm_enabled: true`
  - `env.base_ctrl_enabled: false`
- restore hard-push curriculum for the first real M3 run:
  - `env.push_duration_steps: 15`
  - `env.push_force_max: 9.0`
- reduce residual authority so the FSM must contribute meaningful stepping:
  - `env.fsm_resid_scale_swing: 0.60`
  - `env.fsm_resid_scale_stance: 0.75`
  - `env.fsm_resid_scale_recover: 0.70`

### Training Intent
This release is meant to answer the first M3 question directly:
- does the robot step under hard pushes?
- does the step enlarge support meaningfully instead of fidgeting?
- does the robot return toward upright instead of remaining in a crouched recovery pose?

### First-Run Review Signals
- `debug/bc_phase` should show real occupancy in `SWING`, not only `STANCE`
- `reward/step_event` and `reward/foot_place` should rise in disturbed episodes
- `debug/bc_phase_ticks` should not be dominated by swing timeouts
- `eval_push/survival_rate` should at least match M2, with clearer stepping behavior
- `reward/posture` should recover after the disturbance rather than staying near zero

### Notes
- M2 config fields remain in the file for rollback/comparison, but are disabled in the training-ready default.
- Deprecated gait-style reward terms remain debug-only for now and should be removed after the first validated M3 run if they are not needed.

---

## [v0.14.2] - 2026-03-07: M3 — foot-placement FSM base controller + waist arm damping

### Summary
Implements M3 from `training/docs/step_trait_base_controller_design.md`: an explicit 3-phase step state machine (STANCE / SWING / TOUCHDOWN_RECOVER) replaces the M2 joint-heuristic controller. The FSM uses Raibert-style foot placement and a half-sine swing trajectory plus waist arm-momentum compensation. Disabled by default; enable with `env.m3_enabled: true`.

### New Files
- `training/envs/step_controller.py` (~400 lines): Pure-JAX M3 module.
  - `compute_need_step()` — gate [0,1] from pitch/roll/lateral-vel/pitch-rate
  - `select_swing_foot()` — load-based + lateral bias
  - `compute_step_target()` — Raibert-style target, clamped to step bounds
  - `compute_swing_trajectory()` — smoothstep XY + half-sine Z arc
  - `update_fsm()` — fully vectorised FSM transitions (no Python control flow)
  - `compute_ctrl_base()` — stance uprightness feedback + swing tracking + waist damping
- `tests/test_step_controller.py` — 26 unit tests, **all passing** (1.82s CPU JAX)

### Code Updates
- `training/envs/env_info.py`: 9 new FSM state fields in `WildRobotInfo` (`fsm_phase`, `fsm_swing_foot`, `fsm_phase_ticks`, `fsm_frozen_tx/ty`, `fsm_swing_sx/sy`, `fsm_touch_hold`, `fsm_trigger_hold`)
- `training/configs/training_runtime_config.py`: 25 new `m3_*` fields in `EnvConfig` (`m3_enabled`, trigger/recovery thresholds, hold ticks, nominal step size, Raibert gains, step bounds, swing trajectory params, residual authority per phase, arm gains)
- `training/envs/wildrobot_env.py`:
  - `_make_initial_state()`: FSM fields initialised to 0 (STANCE)
  - `step()`: 3-branch action pipeline (m3 / m2 / passthrough); FSM tuple carried in all branches
  - `new_wr_info` / `preserved_wr_info`: FSM fields in both `lax.cond` branches (pytree safety)
  - `WildRobotEnv._m3_compute_ctrl()`: new ~200-line method wiring FSM into step loop
  - 3 new debug metrics: `debug/bc_phase`, `debug/bc_swing_foot`, `debug/bc_phase_ticks`
- `training/core/metrics_registry.py`: MetricSpec indices 60-62 for new debug metrics
- `training/core/experiment_tracking.py`: initial values added to ENV_METRICS_KEYS and both `get_initial_env_metrics()` functions

### Config Updates (`training/configs/ppo_standing_push.yaml` → v0.14.2)
- `version: "0.14.2"`
- M3 config block added with `m3_enabled: false` and all 25 parameters at design-doc defaults

### Verified
```
26/26 unit tests passing (1.82s CPU JAX)
Smoke test (reset + 5 steps M3=off, reset + 20 steps M3=on): PASSED
  M3 disabled: bc_phase=0.000, reward/step_event flowing
  M3 enabled:  fsm_phase=0 (STANCE), fsm_phase_ticks=20, metrics plumbed correctly
```

### Activation
To activate M3 in training:
```yaml
env:
  m3_enabled: true
  base_ctrl_enabled: false   # M3 replaces M2
```

---

## [v0.14.1] - 2026-03-07: Standing pushes (M2: joint-heuristic base + residual gating)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.14.1"` and enable M2 mixing:
  - `env.base_ctrl_enabled: true`
  - base feedback gains: `env.base_ctrl_*`
  - residual gate: `env.residual_scale_{min,max}` + `env.residual_gate_power`
- `training/configs/ppo_standing_push.yaml`: reduce base/residual authority defaults to lower action clipping (observed high `action_sat` during pushes):
  - lower `env.base_ctrl_*` gains and `env.base_ctrl_action_clip`
  - lower `env.residual_scale_max`

### Code Updates
- `training/configs/training_runtime_config.py`: add `EnvConfig` knobs for M2 base controller + residual gating.
- `training/configs/training_config.py`: parse new `env.base_ctrl_*` and `env.residual_*` fields from YAML.
- `training/envs/wildrobot_env.py`: implement M2 action mixing:
  - `action_applied = action_base(default_pose + conservative pitch/roll feedback) + residual_scale(need_step) * policy_action`
  - residual gate uses the same `need_step` computation as stepping-trait rewards (tilt + lateral velocity + pitch rate)

### Notes
- No policy contract changes (action/obs dims unchanged). Existing checkpoints should still resume; the controller is purely an environment-side action transformation controlled by config.

### Results (smoke + hard confirm eval @500)
- Baseline checkpoint for M3 (handoff): `training/checkpoints/ppo_standing_push_v00141_20260307_182350/checkpoint_210_430080.pkl`
- Hard confirm eval summary (iter 210, eval horizon 500):
  - `eval_push`: success=86.2%, ep_len=456.0, term_h_low=13.8% (dominant failure mode)
  - `eval_clean`: success=99.2%, ep_len=496.7, term_h_low=0.8%

## [v0.13.11] - 2026-03-07: Standing pushes (step trait via touchdown + foot placement)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.11"`.
- `training/configs/ppo_standing_push.yaml`: enable stepping-trait shaping (all gated by `debug/need_step` to avoid marching-in-place):
  - add `reward_weights.gait_periodicity`, `hip_swing`, `knee_swing`, and `_min` thresholds
  - add `reward_weights.step_event` (touchdown reward) and `reward_weights.foot_place` (foot placement reward)
  - add `reward_weights.foot_place_*` coefficients and `reward_weights.step_need_*` gates

### Code Updates
- `training/envs/env_info.py`: add `prev_left_loaded` / `prev_right_loaded` fields for contact transition (touchdown) detection.
- `training/envs/wildrobot_env.py`: add step-trait rewards:
  - `reward/step_event`: touchdown event reward (gated by `need_step`)
  - `reward/foot_place`: Raibert-style foot placement reward at touchdown (gated by `need_step`)
  - log `debug/need_step`, `debug/touchdown_left`, `debug/touchdown_right`
- `training/core/experiment_tracking.py`: include new reward/debug keys in reset metrics dict (JAX scan carry stability).
- `training/core/metrics_registry.py`: append new metrics (append-only) for W&B logging: `reward/step_event`, `reward/foot_place`, `debug/need_step`, `debug/touchdown_left`, `debug/touchdown_right`.
- `wildrobot/agents/training_loop_agent.py`: heuristic advisor now tunes stepping weights and can relax `env.min_height` (bounded) when height-low dominates, to allow deep-crouch recovery + stepping.

### Milestone M1 implementation summary

This release completes **Milestone M1: Reward-only stepping trait** from `training/docs/step_trait_base_controller_design.md`.

**What's implemented:**
1. **Touchdown detection** — `prev_left_loaded` / `prev_right_loaded` stored in `WildRobotInfo` each step; contact transitions detected as `(~prev_loaded) & loaded` per foot.
2. **Need-to-step gate** — `need_step ∈ [0,1]` computed from `|pitch|/g_pitch + |roll|/g_roll + |lat_vel|/g_lat + |pitch_rate|/g_pr) / 4.0`, then multiplied by `healthy`. Prevents marching-in-place.
3. **Foot placement reward** (Raibert-style) — `exp(-placement_err/sigma²)` at touchdown, gated by `need_step`. Lateral target adjusted by lateral velocity and roll; forward target by forward_vel and pitch.
4. **Config knobs** — all `step_event`, `foot_place`, `foot_place_sigma/k_*`, `step_need_*`, `gait_periodicity`, `hip_swing/knee_swing*` exposed via `RewardWeightsConfig` and parsed from YAML.
5. **Metrics** — `reward/step_event`, `reward/foot_place`, `debug/need_step`, `debug/touchdown_left`, `debug/touchdown_right` in registry (indices 55-59) and in initial-metrics dict (zero at reset, correct during step).
6. **No obs/action/contract changes** — `action_filter_alpha` unchanged; policy contract hash stable.

**Verified (CPU JAX smoke):**
```
reward/step_event:   0.1480  (non-zero, gated by need_step)
reward/foot_place:   0.1469  (non-zero at touchdown)
debug/need_step:     0.1480  (non-zero when disturbed)
debug/touchdown_left:  0.0000
debug/touchdown_right: 1.0000  (right foot touchdown from home keyframe)
```

### Plan
1. Validate setup:
   `uv run python scripts/validate_training_setup.py`
2. Smoke test:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. Resume stepping curriculum from latest best checkpoint:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --resume <best_ckpt.pkl>`

**Acceptance criteria (how to verify during training):**
- `debug/need_step > 0` during push disturbances.
- `reward/step_event > 0` and `debug/touchdown_*` become non-zero in pushed runs.
- `reward/foot_place > 0` (not always zero) at touchdown events.
- Existing `eval_push/*`, `eval_clean/*`, and survival metrics remain intact.

## [v0.13.10] - 2026-03-07: Standing pushes (step recovery + posture return)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.10"`.
- `training/configs/ppo_standing_push.yaml`: make stepping recovery more accessible under pushes:
  - `reward_weights.clearance: 0.05 -> 0.10`
  - `reward_weights.flight_phase_penalty: -0.1 -> 0.0`
- `training/configs/ppo_standing_push.yaml`: add posture-return shaping (encourage returning to default pose once upright):
  - `reward_weights.posture: 0.3`
  - `reward_weights.posture_sigma: 0.40`
  - `reward_weights.posture_gate_pitch: 0.35`
  - `reward_weights.posture_gate_roll: 0.35`

### Code Updates
- `training/envs/wildrobot_env.py`: add `reward/posture` shaping term (gated by uprightness) and log `debug/posture_mse`.
- `training/configs/training_runtime_config.py` + `training/configs/training_config.py`: add/parse new posture shaping knobs under `reward_weights`.

### Plan
1. Validate setup:
   `uv run python scripts/validate_training_setup.py`
2. Smoke test:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. Fine-tune from best v0.13.9 checkpoint:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --resume <best_ckpt.pkl>`

## [v0.13.9] - 2026-03-04: Standing pushes (reduce height-low + pitch terminations)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.9"`.
- `training/configs/ppo_standing_push.yaml`: strengthen pre-collapse shaping to reduce `term_height_low_frac`:
  - `env.collapse_height_buffer: 0.02 -> 0.03`
  - `env.collapse_vz_gate_band: 0.05 -> 0.07`
  - `reward_weights.collapse_height: -0.3 -> -0.5`
  - `reward_weights.collapse_vz: -0.2 -> -0.3`
- `training/configs/ppo_standing_push.yaml`: improve recovery to reduce pitch-limit failures:
  - `reward_weights.orientation: -3.0 -> -4.0`
  - keep `env.action_filter_alpha: 0.6` to allow resuming from v0.13.8 checkpoints (part of `policy_spec_hash`)

### Plan
1. Validate setup:
   `uv run python scripts/validate_training_setup.py`
2. Smoke test:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. Fine-tune from v0.13.8 best checkpoint:
   `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --resume training/checkpoints/ppo_standing_push_v00138_20260303_195743-6y6ufyw3/checkpoint_80_10485760.pkl`

---

## [v0.13.8] - 2026-03-04: Standing stability under pushes (collapse prevention + dual eval)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump to `version: "0.13.8"` and tune standing stability settings.
- `training/configs/ppo_standing_push.yaml`: add pre-collapse shaping knobs under `env`:
  - `collapse_height_buffer`, `collapse_height_sigma`, `collapse_vz_gate_band`
- `training/configs/ppo_standing_push.yaml`: add collapse-prevention reward terms under `reward_weights`:
  - `collapse_height`, `collapse_vz`
- `training/configs/ppo_standing_push.yaml`: increase `ppo.kl_lr_backoff_multiplier` (`2.0 -> 3.0`) to avoid immediate KL-triggered LR backoff at iter 1.

### Code Updates
- `training/envs/wildrobot_env.py`: add two pre-collapse penalties:
  - `reward/collapse_height_pen`: quadratic penalty when height drops below `(min_height + buffer)`
  - `reward/collapse_vz_pen`: downward vertical velocity penalty gated near `min_height`
- `training/configs/training_runtime_config.py` + `training/configs/training_config.py`: add/parse new env knobs and reward weights for collapse shaping.
- `training/core/metrics_registry.py`: append `reward/collapse_height_pen` and `reward/collapse_vz_pen` (append-only registry).
- `training/core/experiment_tracking.py`: include new reward terms in schema/init + log filtering for `eval_push/*` and `eval_clean/*`.
- `training/core/training_loop.py` + `training/train.py`: run and log dual deterministic eval passes:
  - `eval_push/*` (pushes enabled, baseline robustness signal)
  - `eval_clean/*` (pushes disabled, clean policy quality signal)
  - rollback/best-checkpoint comparisons are now explicitly driven by `eval_push/success_rate`.
- Torque saturation metrics remain preferred stress signals (`debug/torque_sat_frac`, `debug/torque_abs_max`); action saturation metrics are retained as legacy diagnostics.

### Results (v0.13.8)
- Run: `training/wandb/offline-run-20260303_195741-6y6ufyw3`
- Checkpoints: `training/checkpoints/ppo_standing_push_v00138_20260303_195743-6y6ufyw3`
- Best checkpoint (eval_push/): `training/checkpoints/ppo_standing_push_v00138_20260303_195743-6y6ufyw3/checkpoint_80_10485760.pkl`
- Best @ iter 80: eval_push/success_rate=98.44%, eval_push/episode_length=495.3
- Clean @ iter 80: eval_clean/success_rate=100.00%, eval_clean/episode_length=500.0
- Train @ iter 80: success=95.24%, ep_len=487.5
- Final @ iter 200: eval_push/success_rate=98.44%, eval_push/episode_length=494.6

| Signal | Value |
|---|---:|
| term_height_low_frac | 4.76% |
| term_pitch_frac | 0.73% |
| term_roll_frac | 0.00% |
| tracking/max_torque | 59.28% |
| debug/torque_sat_frac | 0.81% |
| ppo/approx_kl | 0.0064 |
| ppo/clip_fraction | 0.1733 |

---

## [v0.13.6] - 2026-03-03: Standing robustness (strong pushes + PPO stability guardrails)

### Config Updates
- `training/configs/ppo_standing_push.yaml`: bump training version to v0.13.6.
- `training/configs/ppo_standing_push.yaml`: enable two-sided height target shaping (`env.height_target_two_sided: true`) to actively reward exploring lower stance near `env.target_height`.
- `training/configs/ppo_standing_push.yaml`: tighten PPO update aggressiveness for stability under pushes:
  - `ppo.learning_rate: 3e-4 → 1e-4`, `ppo.epochs: 4 → 2`, `ppo.entropy_coef: 0.01 → 0.005`
  - `ppo.clip_epsilon: 0.2 → 0.15`
- `training/configs/ppo_standing_push.yaml`: add P0/P1 stability guardrails (PPO KL + schedules + deterministic eval + rollback):
  - `ppo.target_kl`, `ppo.kl_*`, `ppo.lr_schedule_end_factor`, `ppo.entropy_schedule_end_factor`
  - `ppo.eval.*` (periodic deterministic eval)
  - `ppo.rollback.*` (rollback-on-regression with LR backoff)
- `training/configs/training_runtime_config.py`: add `PPOEvalConfig` + `PPORollbackConfig` under `PPOConfig`.
- `training/configs/training_config.py`: parse `ppo.eval` and `ppo.rollback` blocks from YAML.

### Code Updates
- `training/core/training_loop.py`: stabilize PPO minibatch sampling (one permutation per epoch; contiguous minibatches).
- `training/core/training_loop.py`: add in-update KL early-stop (halts remaining minibatches within the current PPO update).
- `training/core/training_loop.py`: add deterministic periodic eval (fixed RNG across eval checkpoints) for monotonic checkpoint comparisons.
- `training/core/training_loop.py`: add rollback-on-regression using eval metrics + LR backoff scale.
- `training/core/training_loop.py`: switch to optax LR schedule inside `optax.adam(...)` (instead of gradient pre-scaling); add `ppo/active_updates` + `ppo/epochs_used` metrics for diagnostics and correct LR logging (including resume).
- `training/train.py`: plumb separate eval reset/step functions (allow different eval batch size).
- `training/core/experiment_tracking.py`: log `ppo/*` and `eval/*` metrics to W&B/offline logs.
- `training/tests/test_training_stability_controls.py`: add unit coverage for the stability controls.

### Plan
1. `uv run python scripts/validate_training_setup.py`
2. `uv run python training/train.py --config training/configs/ppo_standing_push.yaml --verify`
3. `uv run python training/train.py --config training/configs/ppo_standing_push.yaml`

---

## [v0.12.2] - 2026-01-05: Standing robustness (moderate pushes)

### Config Updates
- `training/configs/ppo_standing.yaml`: bump training version to v0.12.2.
- `training/configs/ppo_standing.yaml`: increase push forces to 2–5 N and keep 6-step pushes.
- `training/configs/training_runtime_config.py`: add IMU noise + latency knobs (training-only).

### Code Updates
- `training/envs/wildrobot_env.py`: apply IMU noise + latency before observation build (training-only).

### Plan
1. `uv run python scripts/validate_training_setup.py`
2. `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
3. `uv run python training/train.py --config training/configs/ppo_standing.yaml`

### Results (Standing PPO, v0.12.2)
- Run: `training/wandb/offline-run-20260108_212208-r9wr2rq5/`
- Checkpoints: `training/checkpoints/ppo_standing_v00122_20260108_212209-r9wr2rq5/`
- Best checkpoint (reward): `training/checkpoints/ppo_standing_v00122_20260108_212209-r9wr2rq5/checkpoint_50_6553600.pkl`
- Run length: 330 iterations (final step: 43,253,760)

| Metric | Value (best @ iter 50) |
|--------|-------------------------|
| Episode Reward | 635.17 |
| Episode Length | 500.0 |
| Forward Velocity (cmd=0.00) | -0.00 m/s |
| Robot Height | 0.468 m |

---

## [v0.12.1] - 2026-01-05: Standing policy_contract Baseline (v0.12.1)

### Contract Migration
- Actor observation/action semantics now flow through `policy_contract` (shared JAX/NumPy implementation + parity tests).
- Resume is now guarded by a `policy_spec_hash` stored in checkpoints to prevent silently resuming with a different policy contract.
- PPO actor initialization is biased to the **home keyframe pose** (prevents early drift/falls before learning).

### Config Updates
- `training/configs/ppo_standing.yaml`: bump training version to v0.12.1.
- `policy_contract`: remove actor linvel from `wr_obs_v1` (linvel stays privileged-only for reward/metrics).
- `training/configs/*.yaml`: remove `env.linvel_mode` / `env.linvel_dropout_prob` (no actor linvel to mask).
- `training/configs/ppo_standing.yaml`: set `wandb.mode: offline` by default (switch to `online` when desired).
- `training/configs/ppo_standing.yaml`: lower `env.action_filter_alpha` to `0.6` (faster corrections, less lag).

### Notes
- Expected `obs_dim` for v0.12.1 standing: `36` (actor linvel removed from `wr_obs_v1`).
- Fresh runs only: resuming from pre-contract checkpoints is blocked by `policy_spec_hash`.

### Plan
1. Validate assets + environment:
   `uv run python scripts/validate_training_setup.py`
2. Quick smoke test (no resume):
   `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
3. Full standing run from scratch (no resume):
   `uv run python training/train.py --config training/configs/ppo_standing.yaml`

---

## [v0.11.5] - 2026-01-04: Standing Sim2Real Prep (v0.11.5)

### Config Updates
- `training/configs/ppo_standing.yaml`: bump training version to v0.11.5.
- `training/configs/ppo_standing.yaml`: document resume command for `training/checkpoints/ppo_standing_v00113_final.pkl`.

### Plan
1. `uv run python scripts/validate_training_setup.py`
2. `uv run python training/train.py --config training/configs/ppo_standing.yaml --resume training/checkpoints/ppo_standing_v00113_final.pkl`

---

## [v0.11.4] - 2026-01-01: Walking Conservative Warm Start (v0.11.4)

### Plan
1. Validate assets + config:
   `uv run python scripts/validate_training_setup.py`
2. Walking warm-start (conservative range):
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume training/checkpoints/ppo_standing_v00113_final.pkl`
3. Full run:
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --resume training/checkpoints/ppo_standing_v00113_final.pkl`

### Base Checkpoint
- `training/checkpoints/ppo_standing_v00113_final.pkl` (v0.11.3 standing best @ iter 310)

### Final Results (2026-01-03, run-20260102_220919-xiqcenuh)
- Run: `training/wandb/run-20260102_220919-xiqcenuh/`
- Checkpoints: `training/checkpoints/ppo_walking_conservative_v00114_20260102_220920-xiqcenuh/`
- Iterations: 690 → 880 (115M total steps)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Episode Length | 474 ± 33 steps | > 400 | ✓ PASS |
| Forward Velocity | 0.33 ± 0.01 m/s | 0.5-1.0 m/s | ✗ BELOW |
| Velocity Tracking Error | ~0.03 m/s | < 0.2 m/s | ✓ PASS |
| Robot Height | 0.458 m | ~0.46 m | ✓ PASS |
| Fall Rate | ~5% | < 5% | Borderline |

- Best checkpoints (99%+ survival):
  - `checkpoint_720_94371840.pkl`: reward=498.2, ep_len=500, vel=0.32 m/s
  - `checkpoint_700_91750400.pkl`: reward=495.8, ep_len=500, vel=0.32 m/s
- Highest reward: `checkpoint_730_95682560.pkl` (reward=509.2, ep_len=457, vel=0.34 m/s)

### Analysis (Final, 2026-01-03)

**What Worked:**
1. **Stable Conservative Walking**: Policy learned stable walking at ~0.33 m/s within velocity range [0.1, 0.6].
2. **Upright Posture**: Height maintained at ~0.458m (target 0.46m), height shaping rewards effective.
3. **Good Velocity Tracking**: Tracking error ~0.03 m/s when cmd≈0.35 m/s.

**Issues Identified:**
1. **Velocity Plateau**: Policy converged to ~0.33 m/s, below 0.5 m/s minimum target.
   - Likely cause: conservative reward balance prioritizes stability over speed.
2. **Stochastic Dependency**: Policy requires stochastic sampling for reliable deployment:
   - `--stochastic`: 100% success (5/5 episodes)
   - `--deterministic`: 20% success (1/5 episodes)
   - Cause: Policy learned to use exploration noise for balance recovery.
3. **Yaw Drift**: Episodes that fail often show significant heading drift before falling.
4. **Visualization Bug Fixed**: `prev_action` was incorrectly initialized to zeros instead of default pose action in `visualize_policy.py`, causing early falls.

### Code Fixes (2026-01-03)
- Fixed `visualize_policy.py`: Initialize `prev_action` to `cal.ctrl_to_policy_action(cal.get_ctrl_for_default_pose())` instead of zeros.
- Fixed `cal.py`: Handle both MJX and native MuJoCo data in `_get_geom_contact_force()`.

### Visualization Commands
```bash
# Recommended (stochastic - matches training)
uv run mjpython training/eval/visualize_policy.py \
  --checkpoint training/checkpoints/ppo_walking_conservative_v00114_20260102_220920-xiqcenuh/checkpoint_720_94371840.pkl \
  --config training/configs/ppo_walking_conservative.yaml \
  --stochastic

# Deterministic (lower success rate)
uv run mjpython training/eval/visualize_policy.py \
  --checkpoint training/checkpoints/ppo_walking_conservative_v00114_20260102_220920-xiqcenuh/checkpoint_720_94371840.pkl \
  --config training/configs/ppo_walking_conservative.yaml
```

### Next Steps (v0.11.5)
1. **Increase velocity**: Boost `tracking_lin_vel` weight or raise `min_velocity` to push past 0.33 m/s plateau.
2. **Deterministic robustness**: Add entropy regularization or train with periodic deterministic evaluation.
3. **Reduce yaw drift**: Increase `angular_velocity` penalty or add explicit heading tracking reward.
4. **Curriculum learning**: Gradually increase velocity target during training.

### Earlier Results (v0.11.4)

#### Status Update (2026-01-02, Walking PPO Conservative Resume)
- Run: `training/wandb/run-20260102_143302-xmislxx8/` (resumed from iter 590)
- Checkpoints: `training/checkpoints/ppo_walking_conservative_v00114_20260102_143303-xmislxx8/`
- Resumed from: `training/checkpoints/ppo_walking_conservative_v00114_20260102_083850-lq6k40oo/checkpoint_590_77332480.pkl`
- Best checkpoint (reward): `checkpoint_640_83886080.pkl` (reward=490.65, vel=0.33 @ cmd=0.35, ep_len=468.5)
- Topline (final @690): reward=476.46, ep_len=453.6, success=80.3%, vel=0.32 @ cmd=0.35, vel_err=0.057, max_torque=0.783
- Terminations (final @690): truncated=80.3%, pitch=18.4%, roll=1.3%, height_low/high=0.0%

#### Plan Update (2026-01-03, Height Shaping + Posture)
- Goal: Reduce squat-walk while keeping conservative walking stable.
- (Historical) used sim linvel; actor masking was not applied in this run.
- Change (code): height shaping is now one-sided (no penalty above `env.target_height`) in `training/envs/wildrobot_env.py`.
- Change (config): set `env.target_height: 0.46` (waist/root height) in `training/configs/ppo_walking_conservative.yaml`.

#### Config + Code Updates (post-run)
- Reduced action lag for faster pitch correction: `env.action_filter_alpha: 0.5 → 0.2` in `training/configs/ppo_walking_conservative.yaml`
- Prioritize straight + upright gait:
  - `reward_weights.angular_velocity: -0.05 → -0.2` (stronger yaw-rate penalty)
  - `reward_weights.orientation: -0.5 → -1.0` (stronger pitch/roll penalty)
  - Added `reward_weights.height_target: 0.2` (reduce squat) with `height_target_sigma: 0.05`
- Fix: resolved `NameError: raw_action_abs_max` by explicitly passing `raw_action` into `_get_reward()` for debug metrics (`training/envs/wildrobot_env.py`)

#### Results (2026-01-01, Walking PPO Conservative Warm Start)
- Run: `training/wandb/run-20260101_215853-5e15gkam`
- Checkpoints: `training/checkpoints/ppo_walking_conservative_v00114_20260101_215854-5e15gkam/`
- Best checkpoint (reward): `checkpoint_430_56360960.pkl` (reward=440.31, vel=0.31 @ cmd=0.35, ep_len=466.4)
- Topline (final @610): reward=431.74, ep_len=474.6, success=88.9%, vel=0.32 @ cmd=0.35, vel_err=0.079, max_torque=0.869
- Notes: Good conservative walking found; next run should continue v0.11.4 (no config changes) from best checkpoint to push success rate toward 95%+

---

## [v0.11.3] - 2025-12-31: Foot Switches + Standing Retrain Plan

### Plan
1. `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
2. `uv run python training/eval/visualize_policy.py --headless --num-episodes 1 --config training/configs/ppo_standing.yaml --checkpoint <path>`
3. Resume walking with new standing checkpoint:
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume <new_standing_checkpoint>`

### Config Updates
- `min_height`: 0.20 → 0.40 (terminate squat posture)
- `reward_weights.base_height`: 1.0 → 3.0 (enforce upright height)
- `reward_weights.orientation`: -1.0 → -2.0 (stricter tilt penalty)
- Added standing push disturbances (random lateral force, short duration)
- Switched foot contact signals to boolean foot switches (toe/heel), obs dim now 39
- Removed `*_touch` sites/sensors from `assets/wildrobot.xml`; switch signal derived from geom forces
- Added height target shaping + stance width penalty to discourage squat posture

### Results (Standing PPO, v0.11.3)
- Run: `training/wandb/run-20260101_182859-2wsk4xt7`
- Resumed from: `checkpoint_260_34078720.pkl` (reward=518.32, ep_len=498)
- Best checkpoint: `training/checkpoints/ppo_standing_v00113_20260101_182901-2wsk4xt7/checkpoint_310_40632320.pkl`
- Final checkpoint: `training/checkpoints/ppo_standing_v00113_20260101_182901-2wsk4xt7/checkpoint_360_47185920.pkl`
- Summary (best @310): reward=574.31, ep_len=500, height=0.454m, vel=-0.02m/s (cmd=0.00), vel_err=0.076, torque~84.9%
- Terminations: 0% across all categories at best window; stable 500-step episodes
- Notes: Ready to warm-start walking with conservative velocity range

---

## [v0.11.2] - 2025-12-31: Upright Standing Retrain Plan

### Plan
1. `uv run python training/train.py --config training/configs/ppo_standing.yaml --verify`
2. `uv run python training/eval/visualize_policy.py --headless --num-episodes 1 --config training/configs/ppo_standing.yaml --checkpoint <path>`
3. Resume walking with new standing checkpoint:
   `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume <new_standing_checkpoint>`

### Config Updates
- `min_height`: 0.20 → 0.40 (terminate squat posture)
- `reward_weights.base_height`: 1.0 → 3.0 (enforce upright height)
- `reward_weights.orientation`: -1.0 → -2.0 (stricter tilt penalty)
- Added standing push disturbances (random lateral force, short duration)

---

## [v0.11.1] - 2025-12-31: CAL PPO Walking Smoke Test

### Test Plan
1. `uv run python training/train.py --config training/configs/ppo_walking_conservative.yaml --verify --resume training/checkpoints/ppo_standing_v0110_170.pkl`
2. `uv run python training/eval/visualize_policy.py --headless --num-episodes 1 --config training/configs/ppo_walking.yaml --checkpoint <path>`

### Results
- Run: `training/wandb/run-20251231_003248-8p2bv838`
- Best checkpoint: `training/checkpoints/ppo_walking_conservative_v00111_20251231_003249-8p2bv838/checkpoint_360_47185920.pkl`
- Summary: reward ~426.79, ep_len ~491, success ~95.4%, vel ~0.33 m/s
- Notes: stable but squat-biased posture inherited from standing checkpoint; hip swing limited

---

## [v0.11.0] - 2025-12-30: Control Abstraction Layer (CAL) - BREAKING CHANGE

### Overview

Major architectural change introducing a **Control Abstraction Layer (CAL)** to decouple high-level flows (training, testing, sim2real) from MuJoCo primitives.

⚠️ **BREAKING CHANGE**: This version requires retraining from scratch. Existing checkpoints are incompatible due to action semantics changes.

### Why CAL?

The previous implementation had critical issues:
1. **Direct MuJoCo Primitive Access**: Multiple layers directly manipulated `data.ctrl`, `data.qpos`, etc.
2. **Ambiguous Actuator Semantics**: Policy outputs [-1, 1] were passed directly to MuJoCo ctrl (which expects radians)
3. **Incorrect Symmetry Assumptions**: Left/right hip joints have mirrored ranges, but code didn't account for this
4. **Training Learning Simulator Quirks**: Policies learned MuJoCo-specific behaviors instead of transferable motor skills

### New Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HIGH-LEVEL LAYER                           │
│  Training Pipeline │ Test Suite │ Sim2Real │ Policy Inference   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTROL ABSTRACTION LAYER (CAL)                    │
│                                                                 │
│  • policy_action_to_ctrl() - normalized [-1,1] → radians        │
│  • physical_angles_to_ctrl() - radians → ctrl (no normalization)│
│  • get_normalized_joint_positions() - for policy observations   │
│  • get_physical_joint_positions() - for logging/debugging       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LOW-LEVEL LAYER (MuJoCo)                      │
│                  data.ctrl, data.qpos, data.qvel                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

#### Two-Method API Pattern
- `policy_action_to_ctrl()`: For policy outputs [-1, 1], always applies symmetry correction
- `physical_angles_to_ctrl()`: For GMR/mocap angles (radians), no symmetry correction

#### Symmetry Correction
- `mirror_sign` field handles joints with inverted ranges (e.g., left/right hip pitch)
- Automatic correction ensures symmetric actions produce symmetric motion

#### Sim2Real Ready
- `ControlCommand` dataclass with target_position and target_velocity
- `VelocityProfile` enum for trajectory generation (STEP, LINEAR, TRAPEZOIDAL, S_CURVE)
- Clean separation between simulation and deployment

#### Default Positions from MuJoCo
- Home pose loaded from MuJoCo keyframe at runtime (single source of truth)
- Recommendation: Use `keyframes.xml` for dedicated keyframe management

### New Files

| File | Description |
|------|-------------|
| `training/control/__init__.py` | Export CAL, specs, types |
| `training/control/cal.py` | ControlAbstractionLayer class |
| `training/control/specs.py` | JointSpec, ActuatorSpec, ControlCommand |
| `training/control/types.py` | ActuatorType, VelocityProfile enums |
| `training/tests/test_cal.py` | CAL unit tests |
| `assets/keyframes.xml` | Dedicated keyframe file (home, t_pose) |

### Files Updated

| File | Change |
|------|--------|
| `training/envs/wildrobot_env.py` | Integrate CAL |
| `assets/robot_config.py` | Robot configuration management (moved from training/configs/) |
| `assets/mujoco_robot_config.json` | Add actuated_joints section |
| `assets/post_process.py` | Add generate_actuated_joints_config() |

### Config Schema Update

New `actuated_joints` section in `mujoco_robot_config.json`:

```yaml
actuated_joints:
  - name: left_hip_pitch
    type: position
    class: htd45hServo
    range: [-0.087, 1.571]
    symmetry_pair: right_hip_pitch
    mirror_sign: 1.0
    max_velocity: 10.0
    # NOTE: default_pos loaded from MuJoCo keyframe at runtime
```

### Migration

1. **Retrain from scratch** - existing checkpoints are incompatible
2. Update `mujoco_robot_config.json` with `actuated_joints` section
3. Create `assets/keyframes.xml` with home pose
4. Replace direct MuJoCo access with CAL methods

### Documentation

See `training/docs/CONTROL_ABSTRACTION_LAYER_PROPOSAL.md` for full design document.

### Results (Standing PPO, v0.11.0)
- Config: `training/configs/ppo_standing.yaml`
- Run: `training/wandb/run-20251230_181458-27832na7`
- Best checkpoint: `training/checkpoints/ppo_standing_v00110_20251230_181500-27832na7/checkpoint_170_22282240.pkl`
- Summary: ep_len ~484, height ~0.434 m, success ~89.8%, vel ~0.01 m/s
- Notes: success rate below 95% target; height_low terminations ~10% late in training


## [v0.10.6] - 2025-12-28: Hip/Knee Swing Reward (Stage 1)

### Config Updates
- Added swing-gated hip/knee rewards to encourage leg articulation during swing.
- Added small flight-phase penalty to discourage hopping.
- Kept gait periodicity reward (alternating support).
- Maintained v0.10.5 tuning for velocity/orientation/torque.

### Base Checkpoint (v0.10.5)
- `training/checkpoints/wildrobot_ppo_20251228_205536/checkpoint_520_68157440.pkl`

### Training Results (v0.10.5 → v0.10.6 prep)
- v0.10.5 best: reward ~566.42 (iter 520), vel ~0.63 m/s, ep_len ~452, torque ~97%.
- Gait analysis: alternating support ~79–80%, flight phase ~18%, knees used ~22–27% of range.
- Interpretation: ankle-dominant shuffle persists; add swing-gated knee/hip rewards + flight penalty.

### Files Updated
- `training/configs/ppo_walking.yaml`
- `training/envs/wildrobot_env.py`
- `training/configs/training_config.py`
- `training/configs/training_runtime_config.py`
- `training/core/metrics_registry.py`

---

## [v0.10.5] - 2025-12-28: Gait Shaping Tune (Stage 1)

### Config Updates
- Reduced velocity dominance to allow gait shaping.
- Increased orientation + torque/saturation penalties to curb forward-lean shuffle.
- Added gait periodicity reward (alternating support).
- Shortened default training run to 400 iterations.

### Notes
- Resume-ready PPO run (best at iter 520) still ankle-dominant; used as base for v0.10.6.

---

## [v0.10.4] - 2025-12-28: Velocity Incentive (Stage 1)

### Config Updates
- Strong velocity tracking incentive to escape standing-still local minimum.
- Reduced stability penalties to allow forward motion discovery.

### Results Summary
- Tracking met (vel_err ~0.085, ep_len ~460) but peak torque ~0.98 and forward-lean shuffle.

---

## [v0.10.3] - 2025-12-27: Forward Walking (Stage 1)

### Config Updates
- PPO walking config aligned to 0.5–1.0 m/s command range.
- Enabled velocity tracking metrics and exit criteria logging.

---

## [v0.10.2] - 2025-12-27: Basic Standing (Stage 1)

### Updates
- Added termination diagnostics and reset preservation for accurate logging.
- Standing training readiness (stability-focused PPO).

---

## [v0.10.1] - 2025-12-26: Config Schema Migration

### Updates
- Migrated to unified `TrainingConfig` with Freezable runtime config for JIT.
- Simplified config loading and CLI override flow.

---

## [v0.8.0] - 2024-12-26: Feature Set Refactoring

### Major Change: Single Source of Truth Architecture

Complete refactoring of AMP feature configuration and extraction to establish clear separation of concerns and a single source of truth for feature layout.

### Feature Dropping (Prevent Discriminator Shortcuts)

Added v0.8.0 feature flags to prevent discriminator from exploiting artifact-prone features:

| Flag | Effect | Rationale |
|------|--------|-----------|
| `drop_contacts` | Drop foot contacts (4 dims) | Discrete values, easy to exploit |
| `drop_height` | Drop root height (1 dim) | Policy/reference mismatch early in training |
| `normalize_velocity` | Normalize root_linvel to unit direction | Remove speed as discriminative signal |

```yaml
# ppo_amass_training.yaml
amp:
  feature:
    drop_contacts: false
    drop_height: false
    normalize_velocity: false
```

### Architecture Refactoring

**New Structure:**
```
training/
├── amp/
│   ├── policy_features.py     # Online extraction (JAX) + extract_amp_features_batched
│   └── ref_features.py        # Offline extraction (NumPy) + load_reference_features
├── configs/
│   ├── feature_config.py      # FeatureConfig + _FeatureLayout (SINGLE SOURCE OF TRUTH)
│   ├── training_config.py     # TrainingConfig + RobotConfig
│   └── training_runtime_config.py  # TrainingRuntimeConfig (JIT)
└── training/
    └── trainer_jit.py         # Training loop with normalize_features (training-specific)
```

#### Files Created
| File | Description |
|------|-------------|
| `configs/training_runtime_config.py` | `TrainingRuntimeConfig` class (renamed from `AMPPPOConfigJit`) |

#### Files Moved
| From | To |
|------|-----|
| `amp/feature_config.py` | `configs/feature_config.py` |

#### Files Deleted
| File | Reason |
|------|--------|
| `amp/amp_features.py` | Superseded by `policy_features.py` |
| `common/preproc.py` | Functionality moved to `policy_features.py` and `ref_features.py` |

### Key Changes

#### `_FeatureLayout` Class (Single Source of Truth)
```python
class _FeatureLayout:
    """Centralized feature layout definition."""

    COMPONENT_DEFS = [
        ("joint_pos", "num_joints", False),      # not droppable
        ("joint_vel", "num_joints", False),      # not droppable
        ("root_linvel", "ROOT_LINVEL_DIM", False),
        ("root_angvel", "ROOT_ANGVEL_DIM", False),
        ("root_height", "ROOT_HEIGHT_DIM", True),   # droppable
        ("foot_contacts", "FOOT_CONTACTS_DIM", True), # droppable
    ]
```

#### `TrainingConfig.to_runtime_config()` Method
```python
# Only way to create TrainingRuntimeConfig
runtime_config = training_cfg.to_runtime_config()
```

#### `load_reference_features()` Function
```python
# Consolidated reference data loading in ref_features.py
from training.amp.ref_features import load_reference_features
ref_features = load_reference_features(args.amp_data)
```

#### CLI Overrides Applied to TrainingConfig
```python
# CLI parameters overwrite training_cfg BEFORE to_runtime_config()
if args.iterations is not None:
    training_cfg.iterations = args.iterations
# ... other CLI overrides
config = training_cfg.to_runtime_config()
```

### Removed Code

| Removed | Reason |
|---------|--------|
| `mask_waist` config/code | No waist joint in robot |
| `velocity_filter_alpha` from runtime | Not used in training |
| `ankle_offset` from runtime | Not used in training |
| `RunningMeanStd` class | Only used in tests |
| `create_running_stats()` | Only used in tests |
| `update_running_stats()` | Only used in tests |
| `normalize_features()` in policy_features.py | Duplicate, kept in trainer_jit.py |

### Design Principles Established

1. **Feature extraction** (`policy_features.py`, `ref_features.py`) - extracts raw features
2. **Training normalization** (`trainer_jit.py`) - statistical normalization for training
3. **Single source of truth** - `_FeatureLayout.COMPONENT_DEFS` defines feature ordering
4. **Config-driven** - All drop flags read from `TrainingConfig`

### Config
```yaml
version: "0.8.0"
version_name: "Feature Set Refactoring"
```

### Migration

If upgrading from v0.7.0:
1. Update imports from `amp.feature_config` → `configs.feature_config`
2. Update imports from `amp.amp_features` → `amp.policy_features`
3. Replace `AMPPPOConfigJit` with `TrainingRuntimeConfig`
4. Use `training_cfg.to_runtime_config()` instead of direct instantiation
5. Remove any references to `mask_waist`, `velocity_filter_alpha`, `ankle_offset` in runtime code

### Status
✅ Complete - architecture refactored with single source of truth

---

## [v0.7.0] - 2024-12-25: Physics Reference Data

### Major Change: Physics-Generated Reference Data

Switched from GMR retargeted motions to physics-realized reference data.

**Why**: GMR motions are dynamically infeasible (robot falls immediately). Physics rollout ensures reference data is achievable by the robot.

### Physics Reference Generator

New script: `scripts/generate_physics_reference_dataset.py`

**Features:**
- Full harness mode (Height + XY + Orientation stabilization)
- Physics-derived contacts from MuJoCo `efc_force`
- AMP features computed from realized physics states
- Quality gates (accept/trim/reject based on load support)

**Results:**
- 12 motions processed → All accepted
- 3,407 frames, 68.14s total duration
- Output: `training/data/physics_ref/walking_physics_merged.pkl`

### GMR Script Updates (v0.7.0 + v0.9.3)

**v0.7.0 - Heading-Local Frame:**
```python
# All velocities now in heading-local frame (rotation invariance)
root_lin_vel_heading = world_to_heading_local(root_lin_vel, root_rot)
root_ang_vel_heading = world_to_heading_local(root_ang_vel, root_rot)
```

**v0.7.0 - FK-Based Contacts:**
```python
# Replaced hip-pitch heuristic (84% double stance) with FK
foot_contacts = estimate_foot_contacts_fk(
    dof_pos, root_pos, root_rot, mj_model,
    contact_height_threshold=0.02
)
```

### Config Changes

```yaml
# ppo_amass_training.yaml
version: "0.7.0"
version_name: "Physics Reference Data"

amp:
  # NEW: Physics-generated reference data
  dataset_path: training/data/physics_ref/walking_physics_merged.pkl
  # was: training/data/walking_motions_normalized_vel.pkl
```

### Expected Training Behavior

| Metric | GMR Baseline | Expected (Physics) |
|--------|--------------|-------------------|
| Robot falls | Immediately | Should walk |
| `disc_acc` | 0.50 (stuck) | 0.55-0.75 (healthy) |
| Reference feasibility | ~15% load support | 100% (harness) |

---

## [v0.6.6] - 2024-12-24: Feature Parity Fix (Root Linear Velocity)

### Problem: Policy and Reference Features Had Different Root Velocity Semantics

Root cause of `disc_acc = 0.50` (discriminator stuck at chance):

| Feature | Policy Extractor | Reference Generator | Impact |
|---------|------------------|---------------------|--------|
| **Root linear vel** | Normalized to unit direction | Raw velocity (m/s) | **CRITICAL** |

When features have different semantics, the discriminator cannot learn a stable decision boundary → collapses to 0.50 accuracy.

### Root Cause Analysis

**Root Linear Velocity Normalization Mismatch**
```python
# Policy (WRONG - normalized to unit direction):
root_linvel_dir = root_linvel / ||root_linvel||  # Removes magnitude!

# Reference (correct - raw velocity):
root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt  # Keeps magnitude
```

After z-score normalization with reference stats, policy velocities (unit vectors with small variance) become near-constant, removing a major discriminative signal.

### Changes

**Policy Feature Extractor (`amp/amp_features.py`):**
```python
# BEFORE (v0.6.5):
root_linvel_dir = root_linvel / (linvel_norm + 1e-8)  # ❌ Normalized

# AFTER (v0.6.6):
root_linvel_feat = root_linvel  # ✅ Raw velocity in m/s
```

**Waist Masking:** Removed (both policy and reference use 29 dims).

The waist_yaw being constant 0 in AMASS is actually valid signal - the policy will learn to keep waist stable like the reference data.

### Feature Vector Contract (v0.6.6)

Both policy and reference produce identical **29-dim** vectors:

| Index | Feature | Dims |
|-------|---------|------|
| 0-8 | Joint positions | 9 |
| 9-17 | Joint velocities | 9 |
| 18-20 | Root linear velocity (**raw m/s**) | 3 |
| 21-23 | Root angular velocity | 3 |
| 24 | Root height | 1 |
| 25-28 | Foot contacts | 4 |
| **Total** | | **29** |

### No Reference Data Rebuild Needed

This fix only changes the policy extractor. Reference data already uses raw velocities and 29 dims.

### Expected Post-Fix Behavior

| Metric | Before (v0.6.5) | Expected (v0.6.6) |
|--------|-----------------|-------------------|
| `disc_acc` | 0.50 (stuck) | 0.55-0.75 (oscillating) |
| `D(real)` | ≈0.5 | 0.7-0.9 |
| `D(fake)` | ≈0.5 | 0.3-0.5 |
| `amp_reward` | ≈0.05 | 0.1-0.4 |

---

## [v0.6.5] - 2024-12-24: Discriminator Middle-Ground + Diagnostic

### Problem: disc_acc stuck at extremes

| Version | `disc_acc` | Problem |
|---------|-----------|---------|
| v0.6.3 | 1.00 | D too strong (overpowered) |
| v0.6.4 | 0.50 | D collapsed (not learning) |

### Changes

**Middle-ground discriminator settings:**
| Parameter | v0.6.3 | v0.6.4 | v0.6.5 |
|-----------|--------|--------|--------|
| `disc_lr` | 1e-4 | 5e-5 | **8e-5** |
| `update_steps` | 3 | 1 | **2** |
| `r1_gamma` | 5.0 | 20.0 | **10.0** |

**Diagnostic settings (test if noise/filter erases distribution):**
| Parameter | Before | v0.6.5 | Rationale |
|-----------|--------|--------|-----------|
| `amp.weight` | 0.3 | **0.5** | Boost AMP signal (was underfeeding) |
| `disc_input_noise_std` | 0.03 | **0.0** | Test if noise erases distribution |
| `velocity_filter_alpha` | 0.5 | **0.0** | Test if filter erases distribution |

### Expected Outcomes

**If disc_acc jumps above 0.50:**
- Noise/filter was erasing the distribution difference
- Re-enable with lower values

**If disc_acc stays at 0.50:**
- Feature mismatch between policy and reference
- Run `diagnose_amp_features.py` to compare features
- Check: joint ordering, pelvis orientation (Z-up vs Y-up), degrees vs radians

### Target

`disc_acc = 0.55-0.75` (healthy adversarial game)

---

## [v0.6.4] - 2024-12-24: Discriminator Balance Fix

### Problem: Discriminator Collapse (disc_acc = 1.00)

Training logs showed classic discriminator collapse:
```
#60-#220: disc_acc=1.00 | amp≈0.02-0.03 | success=0.0%
```

**Root Cause Analysis:**
- `disc_acc = 1.00` means discriminator perfectly classifies real vs policy samples
- This causes `amp_reward → 0` (policy gets no gradient from AMP)
- Policy ignores style and only optimizes task reward → no AMASS transfer
- This is the "perfect expert classifier" failure mode

**Why it happened (previous v0.6.3 config):**
| Parameter | Old Value | Problem |
|-----------|-----------|---------|
| `disc_lr` | 1e-4 | Too high - D learns too fast |
| `update_steps` | 3 | Too many D updates per PPO iter |
| `r1_gamma` | 5.0 | Too weak - not enough gradient penalty |
| `disc_input_noise_std` | 0.02 | Too low - D can overfit to clean features |
| `replay_buffer_ratio` | 0.5 | D learns to perfectly classify old samples |

### Changes

**Config (ppo_amass_training.yaml):**
| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `disc_lr` | 1e-4 | 5e-5 | Slow down D learning |
| `update_steps` | 3 | 1 | Fewer D updates per PPO iteration |
| `r1_gamma` | 5.0 | 20.0 | Stronger gradient penalty to smooth D |
| `disc_input_noise_std` | 0.02 | 0.05 | More noise to blur D decision boundary |
| `replay_buffer_ratio` | 0.5 | 0.2 | Less historical samples when D is strong |

### Expected Post-Fix Behavior

| Metric | Before | Expected After |
|--------|--------|----------------|
| `disc_acc` | 1.00 (stuck) | 0.55-0.75 (oscillating) |
| `amp_reward` | ≈0.02 | 0.1-0.4 (rising) |
| `task_r` | Rising fast | May dip initially, then rise |
| `success` | 0.0% | Should become non-zero |

**Key insight:** You want the discriminator to **struggle**, not win. A healthy GAN game keeps D accuracy around 0.55-0.75.

### Future Tuning

Once `disc_acc` stabilizes in 0.55-0.75 range:
- Increase `amp.weight` from 0.3 → 0.5 for stronger style influence
- Monitor for any regression back to 1.00

---

## [v0.6.2] - 2024-12-24: Golden Rule Configuration

### Problem
The v0.6.1 Golden Rule fixes had hardcoded values that should be configurable:
1. **Hardcoded joint indices**: `left_hip_pitch=1`, `left_knee_pitch=3`, etc. were hardcoded instead of derived from `mujoco_robot_config.json`
2. **Magic numbers unexplained**: Contact estimation used unexplained values (0.5, 0.3, 1.0)
3. **Parameters as defaults**: `use_estimated_contacts`, `use_finite_diff_vel`, and contact params had defaults instead of being required from training config

### Changes

#### Configuration (Training Config → Feature Extraction)
| File | Change |
|------|--------|
| `configs/ppo_amass_training.yaml` | Added Golden Rule section with all configurable params |
| `configs/config.py` | Added Golden Rule params to `TrainingConfig` dataclass |

**New Config Options:**
```yaml
amp:
  # v0.6.2: Golden Rule Configuration (Mathematical Parity)
  use_estimated_contacts: true   # Use joint-based contact estimation (matches reference)
  use_finite_diff_vel: true      # Use finite difference velocities (matches reference)
  contact_threshold_angle: 0.1   # Hip pitch threshold for contact detection (rad, ~6°)
  contact_knee_scale: 0.5        # Knee angle at which confidence decreases (rad, ~28°)
  contact_min_confidence: 0.3    # Minimum confidence when hip indicates contact (0-1)
```

#### Feature Extraction (Config-Driven Joint Indices)
| File | Change |
|------|--------|
| `amp/amp_features.py` | Added joint indices to `AMPFeatureConfig` (derived from robot_config) |
| `amp/amp_features.py` | `estimate_foot_contacts_from_joints()` uses config indices |
| `amp/amp_features.py` | `extract_amp_features()` now REQUIRES Golden Rule params (no defaults) |
| `amp/amp_features.py` | Added comprehensive docstrings explaining magic numbers |

**AMPFeatureConfig Now Includes:**
```python
class AMPFeatureConfig(NamedTuple):
    # ... existing fields ...
    # v0.6.2: Joint indices for contact estimation (from robot_config)
    left_hip_pitch_idx: int   # Index of left_hip_pitch in joint_pos
    left_knee_pitch_idx: int  # Index of left_knee_pitch in joint_pos
    right_hip_pitch_idx: int  # Index of right_hip_pitch in joint_pos
    right_knee_pitch_idx: int # Index of right_knee_pitch in joint_pos
```

**Magic Numbers Explained:**
```python
def estimate_foot_contacts_from_joints(
    joint_pos, config,
    threshold_angle: float = 0.1,   # ~6° - leg behind body → contact
    knee_scale: float = 0.5,        # ~28° - confidence decreases as knee bends
    min_confidence: float = 0.3,    # 30% - minimum contact when hip indicates contact
):
    # Confidence formula: clip(1.0 - |knee| / knee_scale, min_confidence, 1.0)
    # - knee=0 → confidence=1.0 (fully extended)
    # - knee=0.5 → confidence=0.3 (min, bent knee)
```

#### Training Pipeline (Pass Config Through)
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Added Golden Rule params to `AMPPPOConfigJit` |
| `training/trainer_jit.py` | `extract_amp_features_batched()` accepts all params |
| `training/trainer_jit.py` | `make_train_iteration_fn()` passes config values |

#### Diagnostic Script (Config-Driven)
| File | Change |
|------|--------|
| `scripts/diagnose_amp_features.py` | Reads Golden Rule params from training config |
| `scripts/diagnose_amp_features.py` | Prints config values for debugging |

### Config
```yaml
version: "0.6.2"
version_name: "Golden Rule Configuration"
```

### Key Principle
**"No hardcoding, everything from config"** — All parameters that affect feature extraction now come from `ppo_amass_training.yaml` and `mujoco_robot_config.json`. Joint indices are derived from actuator names, not hardcoded.

### Migration
If upgrading from v0.6.1, add these fields to your training config:
```yaml
amp:
  use_estimated_contacts: true
  use_finite_diff_vel: true
  contact_threshold_angle: 0.1
  contact_knee_scale: 0.5
  contact_min_confidence: 0.3
```

### Status
✅ Complete - all Golden Rule parameters are now configurable

---

## [v0.6.0] - 2024-12-23: Spectral Normalization + Policy Replay Buffer

### Problem
Two remaining discriminator stability issues:
1. **Unconstrained Lipschitz constant**: Discriminator gradients could explode, destabilizing training
2. **Catastrophic forgetting**: Discriminator overfits to recent policy samples, forgetting earlier distributions

### Changes

#### Spectral Normalization (Miyato et al., 2018)
| File | Change |
|------|--------|
| `amp/discriminator.py` | Added `SpectralNormDense` layer wrapping Flax `nn.SpectralNorm` |
| `amp/discriminator.py` | `AMPDiscriminator` now uses spectral-normalized layers by default |
| `amp/discriminator.py` | Added `use_spectral_norm` flag for A/B testing (default: `True`) |

**Why Spectral Normalization:**
- Industry standard for GANs (StyleGAN, BigGAN, etc.)
- Controls Lipschitz constant to 1 by normalizing weights by largest singular value
- Prevents gradient explosion without hyperparameter tuning
- More stable than gradient penalties alone

```python
class SpectralNormDense(nn.Module):
    """Dense layer with Spectral Normalization."""
    features: int

    @nn.compact
    def __call__(self, x, training: bool = True):
        dense = nn.Dense(features=self.features, ...)
        spectral_dense = nn.SpectralNorm(dense, collection_name="batch_stats")
        return spectral_dense(x)
```

#### Policy Replay Buffer
| File | Change |
|------|--------|
| `amp/replay_buffer.py` | NEW: `PolicyReplayBuffer` (Python mutations) |
| `amp/replay_buffer.py` | NEW: `JITReplayBuffer` (functional JAX updates) |
| `training/trainer_jit.py` | Added `replay_buffer_size` and `replay_buffer_ratio` to config |
| `training/trainer_jit.py` | Added replay buffer state fields to `TrainingState` |
| `train.py` | Pass replay buffer config from YAML |
| `configs/ppo_amass_training.yaml` | Added replay buffer settings |

**Why Replay Buffer:**
- Stores historical policy features to prevent discriminator from overfitting to current policy
- Mixes `replay_buffer_ratio` (default 50%) historical samples with fresh samples
- JIT-compatible implementation using functional updates for `jax.lax.scan`

```python
# JIT-compatible replay buffer (functional updates)
class JITReplayBuffer:
    @staticmethod
    def add(state: dict, samples: jnp.ndarray) -> dict:
        # Functional update for use in jax.lax.scan
        ...

    @staticmethod
    def sample(state: dict, rng: jax.Array, batch_size: int) -> jnp.ndarray:
        # Sample batch mixing fresh + historical
        ...
```

### Config
```yaml
version: "0.6.0"
version_name: "Spectral Normalization + Policy Replay Buffer"

amp:
  # v0.6.0: Policy Replay Buffer
  replay_buffer_size: 100000
  replay_buffer_ratio: 0.5  # 50% historical samples
```

### Expected Result
- More stable discriminator training (no gradient explosions)
- Smoother `disc_acc` curves (less catastrophic forgetting)
- Better generalization of discriminator across policy evolution

### Status
🔄 Ready to train - building on v0.5.0 foot contact fix

---

## [v0.5.0] - 2024-12-23: Foot Contact Fix + Temporal Context

### Problem
**Critical distribution mismatch bug discovered:** Policy rollouts sent zeros for foot contacts while reference data had 88% non-zero foot contacts. The discriminator learned to trivially distinguish them by checking if foot_contacts == 0, achieving 100% accuracy without learning actual motion quality.

| Source | Foot Contacts |
|--------|---------------|
| Reference Data | 88% non-zero (real contact values) |
| Policy Rollouts (v0.4.x) | 100% zeros (not extracted from sim!) |

**Result:** `disc_acc = 1.00` and `amp_reward ≈ 0` — the "foot contact cheat" bug.

### Root Cause
`foot_contacts` were never extracted from MJX simulation during policy rollouts. The `extract_amp_features()` function silently fell back to zeros when `foot_contacts=None`.

### Changes

#### Foot Contact Extraction (Core Fix)
| File | Change |
|------|--------|
| `envs/wildrobot_env.py` | Extract real foot contacts from MJX contact forces |
| `envs/wildrobot_env.py` | Use explicit config keys (`left_toe`, `left_heel`, etc.) |
| `envs/wildrobot_env.py` | Raise `RuntimeError` if foot geoms not found |
| `envs/wildrobot_env.py` | Made `contact_threshold` and `contact_scale` configurable |
| `training/trainer_jit.py` | Added `foot_contacts` to `Transition` namedtuple |
| `training/trainer_jit.py` | Pass `foot_contacts` through entire training pipeline |
| `amp/amp_features.py` | Require `foot_contacts` (no silent fallback to zeros!) |

#### Asset Updates
| File | Change |
|------|--------|
| `assets/post_process.py` | Renamed geoms: `left_toe/left_heel` (was `left_foot_btm_front/back`) |
| `assets/post_process.py` | Added explicit config keys to `mujoco_robot_config.json` |
| `assets/mujoco_robot_config.json` | Added `left_toe`, `left_heel`, `right_toe`, `right_heel` keys |
| `assets/wildrobot.xml` | Updated geom names via post_process.py |

#### Temporal Context Infrastructure (P3 - Prepared, Not Enabled)
| File | Change |
|------|--------|
| `amp/amp_features.py` | Added `TemporalFeatureConfig`, `create_temporal_buffer()` |
| `amp/amp_features.py` | Added `update_temporal_buffer()`, `add_temporal_context_to_reference()` |

#### Testing
| File | Description |
|------|-------------|
| `tests/test_foot_contacts.py` | 8 comprehensive tests for foot contact pipeline |

### Foot Contact Detection
```python
# 4-point model: [left_toe, left_heel, right_toe, right_heel]
# Soft thresholding: tanh(force / scale) for continuous 0-1 values
foot_contacts = jnp.tanh(contact_forces / self._contact_scale)
```

### Key Principle
**"Fail loudly, no silent fallbacks"** — If foot contacts are missing, raise an exception immediately rather than silently using zeros.

### Expected Result
- `disc_acc` should fluctuate around 0.5-0.7 (healthy discrimination)
- `amp_reward` should stay meaningful (0.3-0.8)
- `success_rate` should gradually increase as robot learns

### Status
🔄 Ready to train - foot contacts now correctly extracted from simulation

---

## [v0.4.1] - 2024-12-23: R1 Regularizer + Distribution Metrics

### Problem
WGAN-GP gradient penalty was theoretically inconsistent with LSGAN loss (which uses bounded targets [0, 1]).

### Changes

#### R1 Regularizer (Replaced WGAN-GP)
| File | Change |
|------|--------|
| `amp/discriminator.py` | Replaced WGAN-GP with R1 regularizer |
| `training/trainer_jit.py` | Updated discriminator training to use R1 |
| `configs/ppo_amass_training.yaml` | Renamed `gradient_penalty_weight` → `r1_gamma` |

**Why R1 over WGAN-GP:**
- WGAN-GP uses interpolated samples (designed for Wasserstein critics)
- R1 only penalizes gradient on REAL samples (simpler, faster)
- More theoretically consistent with LSGAN

```python
# R1 Regularization: gradient penalty on REAL samples only
def real_disc_sum(x):
    return jnp.sum(model.apply(params, x, training=True))

grad_real = jax.grad(real_disc_sum)(real_obs)
r1_penalty = jnp.mean(jnp.sum(grad_real ** 2, axis=-1))

total_loss = lsgan_loss + (r1_gamma / 2.0) * r1_penalty
```

#### Distribution Metrics
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Added `disc_real_mean`, `disc_fake_mean`, `disc_real_std`, `disc_fake_std` |

These metrics help diagnose discriminator behavior:
- `disc_real_mean` should approach 1.0
- `disc_fake_mean` should approach 0.0
- If both are similar, discriminator is not learning separation

### Config
```yaml
amp:
  r1_gamma: 5.0  # R1 regularizer weight (was gradient_penalty_weight)
```

### Status
✅ Implemented and validated

---

## [v0.4.0] - 2024-12-23: Critical Bug Fixes (Reward Formula + Training Order)

### Problem
Two critical bugs identified by external design review:

1. **Reward Formula Bug ("The Participation Trophy"):** Policy received 0.75 reward when discriminator was 100% sure motion was fake.
2. **Training Order Bug ("The Hindsight Bias"):** AMP rewards computed with updated discriminator, not the one active when samples were collected.

### Changes

#### Fix 1: Reward Formula (Clipped Linear)
| Before (Buggy) | After (Fixed) |
|----------------|---------------|
| `max(0, 1 - 0.25 * (D(s) - 1)²)` | `clip(D(s), 0, 1)` |
| D=0 (fake) → reward=0.75 ❌ | D=0 (fake) → reward=0.0 ✅ |

| File | Change |
|------|--------|
| `amp/discriminator.py` | Changed reward formula to clipped linear |

#### Fix 2: Training Order
| Before (Buggy) | After (Fixed) |
|----------------|---------------|
| Train disc → Compute reward (NEW params) | Compute reward (OLD params) → Train disc |

| File | Change |
|------|--------|
| `training/trainer_jit.py` | Moved AMP reward computation BEFORE discriminator training |

#### Fix 3: Network Size Increase
| Setting | Before | After |
|---------|--------|-------|
| `discriminator_hidden` | `[256, 128]` | `[512, 256]` |

### Expected Result
After these fixes, `disc_acc` should immediately jump from 0.50 to 0.60-0.80 within the first 20 iterations.

### Status
✅ Implemented and validated

---

## [v0.3.1] - 2024-12-23: Config-Driven Pipeline (No Hardcoding)

### Problem
Joint order mismatch between reference data and policy features. GMR's `convert_to_amp_format.py` had **hardcoded** joint order that didn't match MuJoCo qpos order, causing discriminator to compare misaligned features.

| Component | Expected | Bug |
|-----------|----------|-----|
| MuJoCo qpos | `waist_yaw` at index 0 | ✅ Correct |
| `mujoco_robot_config.json` | `waist_yaw` at index 0 | ✅ Correct |
| GMR hardcoded | `left_hip_pitch` at index 0 | ❌ Wrong |

### Changes

#### GMR Pipeline (Config-Driven)
| File | Change |
|------|--------|
| `GMR/scripts/convert_to_amp_format.py` | Added `--robot-config` flag (required), removed all hardcoded joint names/indices |
| `GMR/scripts/batch_convert_to_amp.py` | Added `--robot-config` flag (required), uses `get_joint_names()` |

#### WildRobot Fixes
| File | Change |
|------|--------|
| `amp/amp_features.py` | Fixed NamedTuple field order (defaults must come last) |
| `envs/wildrobot_env.py` | Fixed `_init_qpos` access before initialization, added `_mjx_model` creation |
| `scripts/scp_to_remote.sh` | Exclude cache files (`__pycache__`, `*.pyc`), include `mujoco_robot_config.json` |

#### Cleanup
- Deleted workaround scripts: `fix_reference_data.py`, `reorder_reference_data.py`
- Cleaned `training/data/` - only `walking_motions_normalized_vel.pkl` remains (641KB)
- Removed 14 intermediate/old data files

### Reference Data Regeneration
```bash
# Full pipeline with config-driven joint order
cd ~/projects/GMR
uv run python scripts/batch_retarget_walking.py
uv run python scripts/batch_convert_to_amp.py \
    --robot-config ~/projects/wildrobot/assets/mujoco_robot_config.json

cd ~/projects/wildrobot
uv run python scripts/convert_ref_data_normalized_velocity.py
```

### Validation
```
✅ Joint order: ['waist_yaw', 'left_hip_pitch', ...] matches mujoco_robot_config.json
✅ Features shape: (3407, 29)
✅ Velocity normalized: True
✅ 12 motions, 68.19s total duration
```

### Key Principle
**"Fail fast, no silent defaults"** - Joint order now comes from `mujoco_robot_config.json`, never hardcoded. Missing config raises immediate error.

### Status
🔄 Ready to train - code synced to remote

---

## [v0.3.0] - 2024-12-23: Velocity Normalization

### Problem
Speed mismatch between reference motion data (~0.27 m/s) and commanded velocity (0.5-1.0 m/s). Discriminator penalized policy for walking at correct speed.

### Changes

#### Capability
- **Velocity Direction Normalization**: Root linear velocity normalized to unit direction vector
- **Safe Thresholding**: If speed < 0.1 m/s, use zero vector (handles stationary frames)
- **Feature Validation Script**: `scripts/validate_feature_consistency.py` ensures reference = policy features

#### Files Modified
| File | Change |
|------|--------|
| `amp/amp_features.py` | Safe velocity normalization with 0.1 m/s threshold |
| `scripts/convert_ref_data_normalized_velocity.py` | Reference data conversion script |
| `scripts/validate_feature_consistency.py` | End-to-end feature validation |
| `reference_data_generation.md` | Updated pipeline documentation |

#### Config
```yaml
amp:
  dataset_path: training/data/walking_motions_normalized_vel.pkl
  # Feature dim remains 29 (direction replaces raw velocity)
```

#### Feature Format (29-dim)
| Index | Feature | Notes |
|-------|---------|-------|
| 0-8 | Joint positions | |
| 9-17 | Joint velocities | |
| 18-20 | Root velocity **DIRECTION** | Normalized unit vector (NEW) |
| 21-23 | Root angular velocity | |
| 24 | Root height | |
| 25-28 | Foot contacts | |

### Validation
- ✅ 63.4% moving frames (norm = 1.0)
- ✅ 36.6% stationary frames (norm = 0.0)
- ✅ Reference and policy features identical

### Expected Result
- AMP reward should be smoother (no speed penalty)
- Discriminator learns gait style, not speed matching
- Policy can walk at commanded speed without AMP penalty

### Status
🔄 Ready to train - awaiting results

---

## [v0.2.0] - 2024-12-22: Discriminator Tuning (Middle-Ground)

### Problem
Initial discriminator settings caused disc_acc stuck at 0.97-0.99 (too strong).
Over-correction caused disc_acc collapse to 0.50 (not learning).

### Changes

#### Config Evolution
| Setting | Initial | Over-corrected | Final (Middle-ground) |
|---------|---------|----------------|----------------------|
| `disc_lr` | 1e-4 | 2e-5 | **5e-5** |
| `update_steps` | 3 | 1 | **2** |
| `gradient_penalty_weight` | 5.0 | 20.0 | **15.0** |
| `disc_input_noise_std` | 0.0 | 0.05 | **0.05** |

#### Final Config
```yaml
amp:
  weight: 0.3
  disc_lr: 5e-5
  batch_size: 256
  update_steps: 2
  gradient_penalty_weight: 15.0
  disc_input_noise_std: 0.05
```

### Result
- ✅ Healthy disc_acc oscillation: 0.72-0.95
- ✅ AMP reward stable
- ❌ Speed mismatch still present (addressed in v0.3.0)

### Lessons Learned
- `disc_acc = 0.50` means discriminator collapsed (not learning)
- `disc_acc = 0.97+` means discriminator too strong
- Target: `disc_acc = 0.6-0.8` with healthy oscillation
- Don't over-handicap the discriminator

---

## [v0.1.1] - 2024-12-21: JIT Compilation Fix

### Problem
30-second GPU idle after first training iteration.

### Cause
Second JIT compilation when switching from single iteration to batch mode.

### Fix
Pre-compile both `train_iteration_fn` and `train_batch_fn` before training starts.

#### Files Modified
| File | Change |
|------|--------|
| `training/trainer_jit.py` | Pre-compile JIT functions with dummy call |
| `train.py` | Added PID display for easy termination |

#### Code
```python
# Pre-compile both functions before training
print("Pre-compiling JIT functions...")
_ = train_iteration_fn(state, env_state, ref_buffer_data)
jax.block_until_ready(_)

if train_batch_fn is not None:
    _ = train_batch_fn(state, env_state, ref_buffer_data)
    jax.block_until_ready(_)
```

### Result
- ✅ No more GPU idle after iteration #1
- ✅ Smooth training from start

---

## [v0.1.0] - 2024-12-20: Initial AMP Training Pipeline

### Capability
- PPO + AMP adversarial training
- LSGAN discriminator formulation
- Reference motion from AMASS via GMR retargeting
- W&B logging integration
- Checkpoint save/restore

### Config
```yaml
trainer:
  num_envs: 1024
  rollout_steps: 128
  iterations: 1000
  lr: 3e-4

reward_weights:
  tracking_lin_vel: 5.0
  base_height: 0.3
  action_rate: -0.01

amp:
  weight: 0.3
  discriminator_hidden: [256, 128]
```

### Result
- Training functional but disc_acc issues (see v0.2.0)
- Speed mismatch discovered (see v0.3.0)

---

## Training Results Log

| Date | Version | Iterations | disc_acc | Velocity | Notes |
|------|---------|------------|----------|----------|-------|
| 2024-12-20 | v0.1.0 | 500 | 0.97-0.99 | ~0.3 m/s | Disc too strong |
| 2024-12-22 | v0.2.0 | 300 | 0.72-0.95 | ~0.5 m/s | Healthy oscillation |
| 2024-12-23 | v0.3.0 | TBD | TBD | TBD | Velocity normalized |
| 2025-12-28 | v0.10.4 | 500 | 0.50 | 0.65 m/s | PPO-only; vel_err ~0.085; ep_len ~460; pitch fall ~10-17%; max torque ~0.98 |
| 2025-12-28 | v0.10.5 | 400 | 0.50 | 0.63 m/s | PPO-only resume from v0.10.4; reward ~566; ankle-dominant gait persisted |

---

## Quick Reference

### Discriminator Tuning Guide
| Symptom | disc_acc | Action |
|---------|----------|--------|
| Disc too strong | 0.95+ | ↓ disc_lr, ↓ update_steps, ↑ gradient_penalty |
| Disc collapsed | ~0.50 | ↑ disc_lr, ↑ update_steps, ↓ gradient_penalty |
| Healthy | 0.6-0.8 | Keep current settings |

### Feature Validation
```bash
# Validate reference = policy features
uv run python scripts/validate_feature_consistency.py

# Analyze reference data velocities
uv run python scripts/analyze_reference_velocities.py
```

### Training Commands
```bash
# Start training
cd ~/projects/wildrobot
uv run python training/train.py

# Kill training (PID printed at startup)
kill -9 <PID>
```

---

## Future Improvements (Backlog)

- [ ] Label smoothing for discriminator (defensive measure)
- [ ] Filter stationary frames from reference data if needed
- [ ] Curriculum learning for velocity commands
- [ ] Foot contact detection from simulation
- [ ] Multi-speed reference data augmentation
