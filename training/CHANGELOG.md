# Training Changelog

This changelog tracks capability changes, configuration updates, and training results for the WildRobot training pipeline.

**Ordering:** Newest releases first (reverse chronological order).

**History prior to v0.20.1:** see [`CHANGELOG.archive.md`](CHANGELOG.archive.md).

---

## [v0.21.0-code-shipped] - 2026-05-24: lateral (vy) + yaw (wz) prior support landed; smoke1 not yet run

### Scope

First v0.21.0 milestone — extends the WildRobot prior + env + reward
+ obs + sampler stack from forward-only (vx) to full 3-axis locomotion
commands (vx, vy, wz), mirroring ToddlerBot's unified ZMP design. This
entry documents the **code shipped on branch `smoke14_lateral`**; the
smoke result will land in a follow-up entry once `smoke1` is run.

Branch: `smoke14_lateral` of `https://github.com/leeygang/wildrobot`.
Diff vs `origin/main`: 46 files changed, +6505 / −1394.

### Architecture deltas (TB-mirrored unless noted)

- **ZMP planner** (`control/zmp/zmp_walk.py`): `generate(vx, vy, wz)`
  with 3-axis norm gate (P1, H1); per-axis stride sign flip;
  combined-translation branch mirrors TB `zmp_walk.py:259-267`;
  pure-rotation arc-radius branch mirrors TB `:269-305`; foot yaw
  propagates into `left/right_foot_rpy[:, 2]` and the IK consumes it
  (P1.5; pre-rotates the foot-to-hip world delta by `-foot_yaw`
  because WR has no `*_hip_yaw` joint to absorb yaw natively).
- **Reference library** (`control/references/reference_library.py`,
  `zmp_walk.py:build_library_for_3d_values`): TB-style split-cmd
  structure — linear-walk bins `(vx × vy, wz=0)` ∪ pure-yaw bins
  `(0, 0, wz)`, dedup `(0, 0, 0)` (P2.5b). Per-axis nearest-grid
  lookup (no mixed-unit Euclidean — P2/review-fix). Metadata
  `vy_grid` / `yaw_rate_grid` (P2.3) and `command_range_vx` force-
  unions the stored vx=0 static bin (P2.5c).
- **`WildRobotInfo.velocity_cmd`** (`training/envs/env_info.py`):
  shape `() → (3,)` migration across ~15 call sites (P3). Added
  `path_state_torso_pos: (3,)` and `path_state_path_rot: (4,) wxyz`
  fields for incremental cmd-integrated path-state target (H7 — the
  closed-form integrator drifts under `cmd_resample_steps > 0`).
- **Config tuples** (`training/configs/training_runtime_config.py`,
  `training/configs/training_config.py`): `EnvConfig.cmd_deadzone`
  and `EnvConfig.eval_velocity_cmd` migrated `float → tuple[float,
  float, float]` with YAML shim `_load_three_axis` (H3 — scalar
  `eval_velocity_cmd: 0.26` broadcasts to `(0.26, 0.0, 0.0)`, NOT
  `(0.26, 0.26, 0.26)`). Sentinel `eval_cmd[0] >= 0` (vx-only).
- **3D sampler** (`_sample_velocity_cmd`, P4): TB-style branched
  zero / pure-turn / walk-ellipse, gated by NEW `EnvConfig.
  cmd_sampler_3d_branched: bool = False` (P4-fix — preserves
  smoke14/12b/7 byte-equivalence; v0.21 smoke1 opts in).
- **3D library plumbing in env** (`_init_offline_service`,
  `_lookup_offline_window`, P5): gated by NEW `EnvConfig.
  loc_ref_command_axes_3d: bool = False`. 3D nearest-key lookup
  with heading-frame correction (TB `walk_zmp_ref.py:132-137`
  parity, JAX-native via new `_yaw_from_quat_wxyz` helper). 3D
  branch respects `loc_ref_offline_library_path` (P5-fix). Primary
  service selected by 3-vec L2 to `(offline_vx, 0, 0)` straight-
  walk anchor instead of vx-only argmin (P5-fix — prevents `_ref_
  init_q_rad` from binding to a lateral bin).
- **Rewards** (`_cmd_forward_velocity_track_reward`,
  `_yaw_rate_track_reward`, P6): added `vy_cmd` kwarg with default
  `0.0` so legacy callers stay byte-equivalent (C16). New static
  `_yaw_rate_track_reward` with NEW `RewardWeightsConfig.cmd_yaw_
  rate_alpha: float = 0.25` (H5: TB `walk.gin:114` `ang_vel_
  tracking_sigma=4.0` → `1/4.0=0.25`, separate from
  `cmd_forward_velocity_alpha=4.0`). NEW `RewardWeightsConfig.cmd_
  yaw_rate_track: float = 0.0` weight; smoke1 sets `1.5`.
  `tracking/yaw_rate_err` + `reward/cmd_yaw_rate_track` metrics
  added (MEAN reducer) and emitted in both reset and step paths
  (P6 follow-up — without this the weighted contribution was
  silent in W&B logs).
- **Obs layout `wr_obs_v8_cmd3d`** (`policy_contract/spec.py`,
  `spec_builder.py`, `jax/obs.py`, P7): NEW layout that appends a
  size-2 `velocity_cmd_lateral_yaw` slot (units `m_s_and_rad_s`)
  after the v7 base. Shared `velocity_cmd` slot sliced to
  `[..., :1]` so v1-v7 stay BYTE-IDENTICAL when fed a `(3,)`
  command. v8 obs_dim = v7 obs_dim + 2.

### Code reorganization (deferred to follow-up phases)

- **P11 — eval/visual/deploy parity** (NEW from r2 plan revision):
  numpy obs mirror, `training/eval/eval_policy.py`, `eval_ladder_v0170
  .py`, `v6_eval_adapter.py`, `visualize_policy.py` all still consume
  scalar `velocity_cmd[0]` and need widening so v8-trained policies
  can be evaluated off-device. Not blocking the in-JAX smoke; blocks
  on-device deploy + visualizer use.
- **Heading-frame correction call sites**: `_lookup_offline_window`
  exposes optional `path_rot_wxyz` / `heading_rot_wxyz` kwargs, but
  the smoke1 call sites still pass `None` (degenerates to identity
  rotation). A small `heading_rot` helper needs to land before the
  full TB heading-frame parity activates.

### Gating (opt-in pattern preserved for backward compat)

| Toggle | Default | Activates |
|---|---|---|
| `EnvConfig.cmd_sampler_3d_branched` | `False` | TB branched sampler (P4) |
| `EnvConfig.loc_ref_command_axes_3d` | `False` | 3D library + heading-frame lookup (P5) |
| `EnvConfig.actor_obs_layout_id` | `wr_obs_v7_*` | `wr_obs_v8_cmd3d` (P7) |
| `RewardWeightsConfig.cmd_yaw_rate_track` | `0.0` | New yaw-rate tracking reward (P6) |

All four default off → smoke14/12b/7 are bit-equivalent to v0.20.x.
The smoke1 YAML (`training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`)
sets all four to opt in.

### Reviewer cycles addressed during development

- **Plan-doc review (pre-code)**: 17 high + 3 medium issues round-1 →
  6 NEW round-2 → 7 user-review high+medium. All resolved in
  `training/docs/v0210_lateral_yaw_prior_plan-final-r2.md`.
- **External code review #1** (after P1-P3 + P1.5 landed): 5 issues
  (eval_cmd incomplete, yaw not TB-equivalent, runtime vx-only,
  mixed-unit lookup, diff hygiene). Fixed in commits
  `23e7220, a41b625, 7101a54`.
- **External code review #2** (after P2.5b/P2.5c): main blocker —
  Cartesian library stored mislabeled mixed-axis bins (a key like
  `(0.20, 0.05, 0.20)` had `command_yaw_rate=0.20` in metadata but
  zero actual yaw because the planner only applies yaw in the pure-
  rotation branch). Fixed by adopting TB split-cmd structure
  (`ed3c7d1`) + meta `command_range_vx` covering stored static bin
  (`44a3d4b`).
- **External code review on P4**: TB sampler produced negative vx
  under `min_velocity > 0` but the reference grid was forward-only —
  cmd/reference mismatch on main training path. Fixed by gating
  behind `cmd_sampler_3d_branched` (`809babf`).
- **External code review on P5**: `primary_idx` bound to wrong
  lateral bin → corrupted `_ref_init_q_rad`. Also 3D branch ignored
  `loc_ref_offline_library_path`. Fixed in `c2f99a9`.
- **External code review on P6**: new `cmd_yaw_rate_track` weighted
  contribution affected optimization but was never logged. Fixed in
  `14642b6`.
- **External code review on P8**: smoke1 inherited smoke14's
  `eval.enabled: false` (Evaluate/* signals silent during training);
  also library bin-count documentation was wrong in 3 places.
  Fixed in `8e1240e` (enable training-time eval; pin
  `len(env._offline_cmd_keys) == 25`).

### What's next

1. **P11 — eval/visual/deploy parity** before any on-device run.
2. **Run smoke1** on a GPU host:
   `uv run python -m training.train --config training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`
   Expect 25-bin library build at init (~3.5 s) per `[v0.21.0
   P2.5a-r2]` measurement.
3. **Land a `v0.21.0-smoke1-lateral-yaw-result` entry** with W&B run
   id, gate verdicts (G4/G5 no regression, G6/G7 ≥ 0.5 signed
   ratio per the new Appendix C in `walking_training.md`), and
   next-version proposals.

### Source-of-truth check

Canonical plan with full sub-step breakdown:
`training/docs/v0210_lateral_yaw_prior_plan-final-r2.md`.

Verified before writing this entry:
- branch is `smoke14_lateral`, 25 commits ahead of `origin/main` (including 3 doc-only commits for the plan).
- All in-scope tests green; library bin count empirically `25`.
- v7 byte-identity guard: `test_jax_v7_obs_byte_identical_when_velocity_cmd_passed_as_3vec` passes.
- C16: legacy callers without `vy_cmd` kwarg unaffected
  (`test_cmd_velocity_tracking_mode.py` passes unmodified).
- smoke14 sampler-shape preserved: `test_smoke12b_sampler_never_produces_negative_cmd` passes.

---

## [v0.20.1-smoke15-deploy-metrics-aggregation-fix] - 2026-05-24: per-episode terminal + abs aggregation for world-drift eval

### Why

Two aggregation bugs in the smoke15-launch implementation
flagged in review:

1. **High — rollout-end vs episode-terminal.**  World-drift
   metrics used `eval_rollout["metrics_vec"][-1, :, idx]` (the
   rollout's last step).  Envs that auto-reset mid-rollout had
   their drift index reading the REPLACEMENT episode's value
   instead of the terminated episode's terminal drift —
   systematically underreporting drift on exactly the unstable
   policies the metric was supposed to catch.

2. **Medium — cross-env sign cancellation.**  Signed drift was
   averaged across envs first, then the gate took `abs()` of
   that scalar.  Half the envs at +0.5 m and half at -0.5 m
   averaged to ~0, so the soft gate passed even though every
   env was bad.

### Fix

- `training/train.py::run_post_training_eval`
  - New `_terminal_episode_mean(name, abs_aggregate=False)` helper
    that does `sum(metric * done) / total_done` — same pattern as
    `mean_episode_length`.  Every completed episode's terminal
    value contributes exactly once.  Setting `abs_aggregate=True`
    applies `abs()` BEFORE the cross-episode mean so cross-env
    cancellation can't mask a bad policy.
  - Emits BOTH variants for sign-ambiguous drift:
    `world_y_drift_signed_m` (diagnostic — direction-of-bias) AND
    `world_y_drift_abs_m` (gate value).  Same for yaw drift
    (`yaw_drift_signed_rad` / `yaw_drift_abs_rad`).
    `world_x_progress_m` is meaningfully signed (backward = bad
    differently from forward) so it stays signed-only.
- `training/core/training_loop.py` Evaluate/* block uses the same
  `_terminal_episode_mean` pattern with `eval_total_done`-masked
  aggregation.
- `training/core/post_training_eval.py::deterministic_eval_gate`
  - Reads `yaw_drift_abs_rad` / `world_y_drift_abs_m` from the
    payload (no `abs()` in the gate — value is already
    per-episode-abs aggregated).
  - Falls back to `abs(signed_value)` when the payload has only
    the legacy `_signed_*` key (back-compat for older logs;
    still subject to cross-env cancellation, but new runs always
    emit `_abs_*`).
  - Soft-signals dict keys renamed: `yaw_drift_abs_rad` /
    `world_y_drift_abs_m` (the gate value names).

### Tests added / updated (112 passed)

- `test_deterministic_eval_gate_exposes_soft_signals_for_deploy_metrics`
  → uses `*_abs_*` payload keys.
- `test_deterministic_eval_gate_back_compat_signed_only_payload` —
  new; pins the `abs(signed)` fallback for legacy data.
- `test_soft_gate_uses_abs_aggregation_not_signed_mean` —
  new Bug-2 regression; constructs a payload where the signed
  mean is 0 but the per-episode-abs mean is 0.5; asserts the
  gate FAILS.
- `test_walking_eval_block_emits_smoke15_deploy_metrics` extended
  to require both `*_signed_*` and `*_abs_*` keys.
- `test_walking_eval_block_aggregates_drift_at_episode_terminal_not_rollout_end` —
  new Bug-1 regression; inspects source to confirm done-mask
  aggregation in both `training_loop.py` and `train.py`, and
  that the buggy `_last_step_mean` helper is gone from both.

### Compat note

`Evaluate/world_y_drift_signed_m` and `Evaluate/yaw_drift_signed_rad`
are still emitted (diagnostic — useful when investigating a
single env's bias direction).  The new `_abs_*` variants are the
authoritative gate values.  Old `post_training_eval_summary.json`
files from before this fix carry only `_signed_*`; the gate's
back-compat path keeps those readable.

---

## [v0.20.1-smoke15-eval-authority-deploy-metrics] - 2026-05-24: make walking eval authoritative + add deploy-facing eval metrics

### Why

Smoke14 (`offline-run-20260523_175517-13zk1p2v`) survived the full
horizon but the post-training deterministic eval rejected every
top-k candidate.  Train-side `forward_velocity = +0.13` at
iter 1380; deterministic eval `forward_velocity = +0.03` at the
SAME checkpoint — a 4.4× train→eval gap that would have been
invisible if `selected_checkpoint_path` had been promoted from
the train-side score.  Smoke14 also had `lateral_velocity_abs ≈
0.20 m/s` (half the cmd) and large heading drift that the
existing G4 forward-only gates couldn't see.

Two failure modes to close:

1. Walking checkpoint selection silently falls back to
   `env/episode_length` when no `Evaluate/*` metrics exist.  A
   standing local minimum saturates ep_len = 500 and reads as
   "good" by that proxy.
2. Eval metrics report only forward velocity / cmd_err / step
   length — nothing on sideways walk or heading drift.  A policy
   can pass the G4 gates and still be undeployable.

### Code changes

- `training/envs/env_info.py` — `WildRobotInfo` gains
  `init_root_pos_xy: (2,)` + `init_root_yaw: ()` on both
  backends; `get_expected_shapes` updated.
- `training/envs/wildrobot_env.py` — reset captures
  spawn pose; step carries it unchanged and emits
  `tracking/world_x_progress_m`, `tracking/world_y_drift_signed_m`,
  `tracking/yaw_drift_signed_rad` (yaw wrapped to (-pi, pi]).
- `training/core/metrics_registry.py` + `training/core/experiment_tracking.py` —
  register the three new tracking metrics + initial-zero schema.
- `training/core/training_loop.py` —
  - Walking-aware eval-authority guard at train start: if
    `cmd_forward_velocity_track > 0` AND neither
    `ppo.eval.enabled` nor `ppo.eval.post_training_enabled` is
    true → raise `ValueError` referencing the smoke14 failure.
  - `walking_metrics` (Evaluate/*) gains `lateral_velocity_abs`,
    `touchdown_rate_left/right_count`, `step_length_*_event_m`,
    `step_length_*_per_touchdown_m`, and the last-step world-drift
    triplet (final-step value averaged across envs so eval reports
    total drift, not rollout mean).
- `training/core/post_training_eval.py` —
  `DeterministicEvalDecision` gains `soft_signals` (report-only,
  does NOT block promotion).  Soft caps as module constants:
  `LATERAL_VELOCITY_SOFT_CAP_MPS=0.10`,
  `YAW_DRIFT_SOFT_CAP_RAD=0.40`,
  `WORLD_Y_DRIFT_SOFT_CAP_M=0.30`.
- `training/train.py` — `run_post_training_eval` returns a dict
  (was 5-tuple) so all deploy metrics flow into the gate + JSON.
  `post_training_eval_summary.json` gains
  `no_passing_candidate_message` + `deterministic_selection_is_authoritative`.
  Per-candidate rows carry `soft_signals` + `soft_fail_reasons`.
- `skills/wildrobot-training-analyze/scripts/analyze_offline_run.py` —
  `RunAnalysis` carries the loaded `post_training_eval_summary`.
  Train-side "Best checkpoint" flagged NON-AUTHORITATIVE when
  Evaluate/* is absent AND post-training summary exists.  New
  "Deterministic post-training eval (AUTHORITATIVE)" block
  surfaces selected-checkpoint / per-candidate gate verdicts in
  the changelog markdown.

### Tests (109 passed across the target battery)

- `training/tests/test_training_stability_controls.py` —
  `test_deterministic_eval_gate_exposes_soft_signals_for_deploy_metrics`,
  `test_deterministic_eval_gate_soft_signals_default_ok_when_missing`,
  `test_walking_config_without_eval_raises_at_train_loop_startup`,
  `test_deterministic_eval_gate_soft_caps_are_documented`.
- `tests/test_metrics_correctness.py` —
  `test_world_drift_tracking_metrics_registered`,
  `test_walking_eval_block_emits_smoke15_deploy_metrics`.
- All existing config-load + critic-stacking + smoke7 eval-cmd
  + reward-normalization tests still pass.

End-to-end env sanity (zero-action 5-step probe on smoke14):
world_x_progress = 9 mm, world_y_drift = 0.5 mm, yaw_drift =
-0.016 rad, lateral_velocity_abs = 0.060 m/s.  Analyzer
back-compat verified on the smoke14 run — the new AUTHORITATIVE
post-training eval block surfaces correctly with the "no
candidate passed" explicit message; train-side proxy best is
labeled NON-AUTHORITATIVE.

Existing smoke14 config (`ppo.eval.enabled=false`,
`post_training_enabled=true`) passes the new walking-eval-authority
guard.  Hypothetical walking configs without either eval path
will now refuse to start instead of silently selecting on
`env/episode_length`.

---
