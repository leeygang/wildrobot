## Changes from -final.md (user review fixes)

This revision addresses the project-owner blocker review of `-final.md`. Every
H1-H7 high-severity issue, every M1-M3 medium issue, and the TB anchors are
honored. Where the prior plan was wrong about a code fact, the fix is grounded
in a direct re-read of the source (line numbers cited inline, repo-relative).

### High-severity blockers

- **H1 — Standing gate uses `abs(command_vx)`** (`control/zmp/zmp_walk.py:597`,
  per-direction sign branch at `:609`). The earlier plan widened
  `generate(command_vx, command_vy, command_yaw_rate)` (P1.4) but left line 597
  intact, so `generate(vx=0, vy=0.05)` would still short-circuit to
  `_generate_standing()` and never exercise the lateral branch. Likewise line
  609's `if command_vx < 0: step_length = -step_length` is a scalar-only sign
  flip with no parallel for vy / wz.

  Fix: a new sub-step **P1.4a** changes the gate to
  `if norm([command_vx, command_vy, command_yaw_rate]) < 1e-4:` and adds
  parallel sign-flip handling for `command_vy < 0` and `command_yaw_rate < 0`
  before entering the per-direction step-length computation. Failing tests
  (P1.5 and P1.7) are tightened to exercise `vx=0, vy=0.05` and
  `vx=0, wz=0.10` — both of which would erroneously return the standing
  trajectory under the old gate.

- **H2 — Reference grid misses static (vx=0) and pure-turn (vx=0, wz!=0)
  bins.** Smoke1 sets `cmd_zero_chance: 0.20` and `cmd_turn_chance: 0.20`
  (per H1's TB-mirroring sampler in P4), so 40% of episodes start at a
  command the 3*5*5=75-bin library cannot represent — the closest bin is
  `(0.18, 0.0, 0.0)`, i.e. a forward-walking trajectory plays back under a
  static or pure-turn command.

  Fix (chose option (a) from the review — cleanest, no per-call dispatch):
  extend `loc_ref_offline_command_vx_grid` to include `0.0`, giving
  `vx_values = [0.0, 0.18, 0.22, 0.26]`. The full library is then
  4 * 5 * 5 = **100 bins** (was 75). The static bin (`0, 0, 0`) yields a
  default-pose trajectory via the existing `_generate_standing()` path
  (which `generate()` calls when `norm([vx, vy, wz]) < 1e-4` per H1). Pure-
  turn bins `(0, 0, wz)` exercise the new arc-radius branch. P2.3 now
  documents the 100-bin total and P5.4 explicitly tightens the cmd_keys
  count assertion. P5.4 also documents memory/build-time (M3).

- **H3 — `eval_velocity_cmd` scalar backward-compat is wrong.** The
  `-final.md` plan claims scalar `0.26` broadcasts to `(0.26, 0.26, 0.26)`,
  which makes smoke14 eval at vy=0.26 and wz=0.26 (both out-of-range and
  silently wrong). The correct semantic is "scalar is a vx-only sentinel":
  `0.26` -> `(0.26, 0.0, 0.0)`.

  Fix: P3.4a's `_load_three_axis` helper changes signature to
  `_load_three_axis(value, *, default_scalar, scalar_broadcast_axis)`:
  - For `cmd_deadzone`, scalar broadcasts to `(s, s, s)` (unchanged — a
    deadzone scalar truly means "all three axes share this magnitude").
  - For `eval_velocity_cmd`, scalar maps to `(s, 0.0, 0.0)` (NEW — only vx
    receives the scalar; vy and wz default to 0.0).
  - The sentinel value `-1.0` (or any negative on vx) means "no override —
    use the sampled cmd".
  Default tuple for `eval_velocity_cmd` becomes `(-1.0, 0.0, 0.0)` (only vx
  uses the negative sentinel; vy/wz pin to 0.0 by convention). New unit
  test `test_load_three_axis_scalar_broadcast_for_eval_cmd` locks this.

- **H4 — Eval / visual / deploy parity missing.** The prior plan only
  touches `policy_contract/jax/obs.py`. A v8-trained policy could not be
  evaluated or visualized because the numpy mirror at
  `policy_contract/numpy/obs.py`, the eval scripts under `training/eval/`,
  and `visualize_policy.py` all hold a scalar `velocity_cmd` and would
  break on a 3-vec input.

  Fix: NEW phase **P11 — Eval / visual / deploy parity** (after P7,
  blocking P9 e2e tests). Lists each file with the specific patch:
  - `policy_contract/numpy/obs.py:43-188` — mirror the v8 layout
    (`wr_obs_v8_cmd3d` in the allow-list, slice `velocity_cmd[..., :1]` on
    line 148, add `velocity_cmd_lateral_yaw` kwarg + branch).
  - `training/eval/eval_policy.py:43, 460, 549` — accept 3D
    `eval_velocity_cmd`; treat `cfg.env.eval_velocity_cmd[0] >= 0.0` as
    the override sentinel; print all three axes in the eval banner.
  - `training/eval/eval_ladder_v0170.py` — same sentinel logic.
  - `training/eval/v6_eval_adapter.py:631-691` — widen the
    `velocity_cmd: float` parameter to `np.ndarray` (3,), pass through to
    `build_observation_from_components` plus
    `velocity_cmd_lateral_yaw=velocity_cmd[1:]`.
  - `training/eval/visualize_policy.py:127, 566-743, 832, 1017, 1072,
    1200` — widen `velocity_cmd` from `float` to `np.ndarray (3,)`;
    update `_validate_user_fixed_velocity_cmd` to accept either a scalar
    (mapped to `(vx, 0, 0)` per H3) or a 3-tuple; update the CLI
    `--velocity-cmd` / `--fixed-velocity` flags to optionally accept three
    floats.
  - `training/eval/visualize_reference_library.py` — if it inspects
    `velocity_cmd`, mirror.
  Each file gets a failing test that asserts the call site accepts a 3-vec
  input without raising. P11 lands AFTER P7 (`policy_contract/jax/obs.py`
  signature changes) and BEFORE the P9.1 e2e rollout test (which exercises
  the env's `_get_obs` -> jax obs builder, but if a user runs the
  resulting policy through visualize, the numpy path must also work).

- **H5 — Yaw reward uses wrong alpha.** The real field is
  `training/configs/training_runtime_config.py:996` `cmd_forward_velocity_alpha:
  float = 4.0` (NOT `cmd_velocity_track_alpha` as `-final.md` P6.4 claimed).
  More importantly: linear-velocity and angular-velocity rewards in TB use
  DIFFERENT tracking widths (`toddlerbot/locomotion/walk.gin:112-114`:
  `lin_vel_tracking_sigma = 1000.0`, `ang_vel_tracking_sigma = 4.0`). Mapping
  TB's denominator-sigma to our numerator-alpha convention by
  `alpha = 1 / sigma` (see the existing comment block at line 977-981
  for the same convention applied to `cp_tracking_sigma`),
  TB-equivalent values are:
  - lin_vel: `alpha = 1 / 1000.0 = 0.001` (note: WR's existing default of
    `4.0` is intentionally tighter than TB; leave unchanged — this is a
    legacy v0.20.x design choice).
  - ang_vel: `alpha = 1 / 4.0 = 0.25`.

  Fix: P6.4 adds a NEW config field `cmd_yaw_rate_alpha: float = 0.25`
  in `RewardWeightsConfig` (NOT reuse of `cmd_forward_velocity_alpha`),
  with the derivation in a comment. The static helper
  `_yaw_rate_track_reward` reads
  `self._config.reward_weights.cmd_yaw_rate_alpha`. Failing test
  `test_yaw_rate_alpha_field_exists_with_tb_aligned_default` locks
  the default; `test_yaw_rate_track_reward_decays_off_cmd` updates its
  expected value to `exp(-alpha * err^2)` with `alpha=0.25, err=0.10`
  -> `exp(-0.0025) approx 0.9975` (was `exp(-5.625)` under the wrong
  alpha).

- **H6 — WR has no hip_yaw joints.** Verified by reading
  `assets/v2/wildrobot.xml:43-110, 346-360`: the only leg joints are
  `{left,right}_{hip_pitch, hip_roll, knee, ankle_pitch, ankle_roll}` —
  NO `*_hip_yaw`. The current plan's P8.1 residual-scale bump for
  `left_hip_yaw` / `right_hip_yaw` would either silently no-op (if the
  config silently drops unknown keys) or raise a KeyError. Either way
  the intent is impossible to satisfy.

  Fix: P8.1 bumps ONLY `{left,right}_hip_roll` and
  `{left,right}_ankle_roll` (the lateral-affecting DoFs). Smoke14 baseline
  values are `0.25` for each, so 1.5x gives `0.375`. P8.2's
  test_smoke1_has_bumped_lateral_residual_scales drops the two `hip_yaw`
  entries from the iteration tuple. **Risks & rollback** section
  acknowledges that without hip_yaw, yaw command tracking will likely
  rely entirely on hip_roll asymmetry and stance-foot-arc geometry — if
  G7 yaw tracking saturates low in smoke1, the v0.22.0 followup is the
  right place to revisit (e.g., a software-yaw approximation via
  hip_roll differential or a hardware-fork branch). Documented as an
  open risk.

- **H7 — Path-state integration assumes constant cmd, but WR resamples.**
  `training/configs/ppo_walking_v0201_smoke14.yaml:110` sets
  `cmd_resample_steps: 150`. The current closed-form
  `compute_command_integrated_path_state(t_since_reset_s, vx, ...)` at
  `control/references/runtime_reference_service.py:181-261` integrates as
  if cmd is constant from step 0; with resampling at step 150 the
  closed form drifts (it integrates the NEW cmd back to t=0, not from
  step 150). On WR step `dt=0.02` this is a 3-second drift before the
  first resample plus a discontinuous jump at every resample point.

  Fix (chose option (a) from the review — production-correct, mirrors
  TB):
  - **New sub-step P3.10a** — add `path_state_torso_pos: jnp.ndarray (3,)`
    and `path_state_path_rot: jnp.ndarray (4,) wxyz` to `WildRobotInfo`
    (both backends + `get_expected_shapes`).
  - **New sub-step P3.10b** — initialize the two fields at reset
    (`_make_initial_state` near line 2635) to
    `(jnp.zeros(3), jnp.array([1.,0.,0.,0.]))`.
  - **New sub-step P3.10c** — add a static
    `RuntimeReferenceService.incremental_path_state_step(
       prev_torso_pos, prev_path_rot_wxyz, velocity_cmd, dt_s)` that
    mirrors TB `toddlerbot/reference/motion_ref.py::integrate_path_state`
    (one-step: `delta_rot = axis_angle_to_quat([0,0,1], wz*dt)`;
    `path_rot = quat_mul(prev_path_rot, delta_rot)`;
    `lin_vel_world = rotate_vec_by_quat([vx,vy,0], path_rot)`;
    `torso_pos = prev_torso_pos + lin_vel_world * dt`). Closed-form helper
    stays for unit-test / library-build paths.
  - **New sub-step P3.10d** — rewire `wildrobot_env.py:2962-2968` to call
    the incremental helper instead of the closed form, threading
    `wr.path_state_torso_pos` / `wr.path_state_path_rot` and updating
    them in the post-step WildRobotInfo write at the bottom of `step`.
  - P3.9 keeps the closed-form unit tests (they are the analytic ground
    truth for the constant-cmd unit case), but adds a new test
    `test_incremental_path_state_matches_closed_form_for_constant_cmd`
    asserting equivalence over 200 steps with no resample.
  Bonus: incremental form also makes the resampled smoke1 episode "warm
  start" the new cmd from the actor's current pose, which matches TB
  semantics. Documented in the M2 follow-on note as well.

### Medium issues

- **M1 — Hardcoded `/Users/ygli` paths.** Every absolute path in the body
  of `-final.md` has been replaced with repo-relative form (e.g.
  `control/zmp/zmp_walk.py`, `training/envs/wildrobot_env.py`). Test
  invocation lines now show `cd <wildrobot-repo> && uv run pytest
  training/tests/test_xxx.py -q` instead of `cd /Users/ygli/projects/wildrobot
  && ...`.

- **M2 — Brittle grep-count audit (P0.5).** The `EXPECTED_COUNTS` per-file
  literal-count assertion churns on every unrelated edit (e.g. an added
  comment line that contains the substring `velocity_cmd`). Followed the
  review's recommendation: **deleted P0.5 entirely**. The per-callsite
  shape-pass-through tests in P3.8a and the end-to-end env-step test in
  P3.7 already enforce that every call site forwards a `(3,)` shape; no
  static-analysis test is needed. `training/docs/v0210_velocity_cmd_audit_checklist.md`
  is also no longer generated. Saves ~150 LoC of plan text and ~80 LoC of
  brittle test code. The cost (zero — the per-callsite tests give the same
  guarantee) is documented inline.

- **M3 — Library memory / build time.** Added a one-time measurement step
  **P2.5a** that builds the 100-bin library locally, measures wall-clock
  build time (`time.perf_counter()`), and computes total JAX array memory
  (sum of `arr.nbytes` over the stacked dict). The numbers go into the
  smoke1 PR description's "Library size" line; if build time exceeds 5
  minutes OR on-device memory exceeds 200 MB on a single env, the
  followup is to drop the wz grid to 1 entry (documented in Risks &
  rollback).

### TB anchor adherence

- 3D cmd sampler: `toddlerbot/locomotion/walk_env.py:256-316` mirrored
  literally in P4.5-P4.7 (verified again — no change from `-final.md`).
- Static command uses default pose:
  `toddlerbot/reference/walk_zmp_ref.py:130, 143-146`. WR achieves this
  by including the `(0, 0, 0)` bin (per H2) so the library's static
  trajectory is built from the existing `_generate_standing()` path.
- Velocity reward tracks `state_ref["lin_vel"][:2]`, yaw tracks
  `state_ref["ang_vel"][2]`: P6.3 unchanged; P6.4 wires `velocity_cmd[2]`
  through `_yaw_rate_track_reward` using `ang_vel_h[2]` from the
  heading-local ang-vel computed in `_compute_reward_terms`.
- Critic stack `c_frame_stack=15`: no change required in this plan (the
  existing `critic_obs_history_frames` field already covers this; the
  v8 layout only widens the actor cmd channel).

## Plan readiness

Project-owner blockers H1-H7 + medium issues M1-M3 addressed; verified
against current code on branch smoke14_lateral.

---

# Plan: v0.21.0 — Lateral (vy) + Yaw (wz) Prior Support for WildRobot (final-r2)

**Goal**: Extend WR's offline ZMP prior and policy I/O from forward-only (vx)
to full 3-DoF locomotion commands (vx, vy, wz), mirroring ToddlerBot's
unified ZMP design, so the policy can learn to strafe and turn alongside
walking forward.

**Architecture** (unchanged from `-final.md` except H1, H2, H7):
- **Single unified ZMP prior**: widen
  `control/zmp/zmp_walk.py::ZMPWalkGenerator.generate()` to accept
  `command_vy` / `command_yaw_rate`, fix the standing gate at line 597 to
  the 3-axis norm (H1), and add a `ZMPWalkConfig.rotation_radius_m` knob.
- **`ReferenceLibrary.lookup` is already 3D** today
  (`control/references/reference_library.py:395-409`); only the env's
  `_lookup_offline_window`, the JAX bin selection, the metadata grids,
  and the **(0, 0, 0) static bin** (H2) need work.
- **Command obs widened 1D -> 3D via NEW `velocity_cmd_lateral_yaw` slot
  appended at the end of v8** (size 2 = `[vy_cmd, wz_cmd]`). Legacy
  1-vec `velocity_cmd` slot keeps `vx_cmd` for v8; v1-v7 byte-identical.
  New layout `wr_obs_v8_cmd3d` registered.
- **`WildRobotInfo.velocity_cmd` migrated `()` -> `(3,)`** with NEW
  `path_state_torso_pos: (3,)` and `path_state_path_rot: (4,)` fields
  for incremental path-state tracking (H7).
- **Reward**: KEEP `_cmd_forward_velocity_track_reward` name; add an
  optional `vy_cmd: jax.Array = jp.float32(0.0)` parameter. Add a
  separate `_yaw_rate_track_reward` static helper reading the NEW
  `cmd_yaw_rate_alpha: float = 0.25` field (H5).
- **WR-scaled command ranges (LOCKED)**: vy in +/-0.13 m/s,
  wz in +/-0.25 rad/s.

**Tech Stack**: Python 3.12, JAX, Brax, MuJoCo MJX, PPO. Tests run via
`cd <wildrobot-repo> && uv run pytest training/tests/test_xxx.py -q`.

**Cross-robot scaling reference**: unchanged from `-final.md`.

---

## Task Dependencies

| Group | Steps | Parallelizable | Notes |
|-------|-------|----------------|-------|
| G1 | P1.1-P1.13 (planner) | No (single file) | `control/zmp/zmp_walk.py` + `control/references/reference_library.py` only. **Now includes P1.4a (H1 gate fix)**. |
| G2 | P2.1-P2.5a (lookup verify + grids + memory measure) | After G1 | `control/references/*.py` only. P2.5a is a one-shot measurement (M3). |
| G3 | P3.1-P3.10d (`velocity_cmd` scalar -> 3-vec, YAML shim, path-state fields, incremental helper) | No (single env file) | `wildrobot_env.py` + `env_info.py` + `training_runtime_config.py` + `training_config.py` + `runtime_reference_service.py`. P3.10a-d are NEW (H7). |
| G4 | P4.1-P4.7 (sampler) | After G3 (same file) | `wildrobot_env.py::_sample_velocity_cmd`. |
| G5 | P5.1-P5.5 (3D library plumbing) | After G2+G3 (same file) | `wildrobot_env.py::_init_offline_service`, `_lookup_offline_window`. |
| G6 | P6.1-P6.4 (reward — incl. line-1218 edit AND yaw-rate term) | After G3+G5 (same file) | `wildrobot_env.py::_cmd_forward_velocity_track_reward` + `_yaw_rate_track_reward` + `_compute_reward_terms`. |
| G7 | P7.1-P7.4 (layout + spec + obs.py kwarg) | After G3 | `policy_contract/spec.py` + `spec_builder.py` + `policy_contract/jax/obs.py`. Can run in parallel with G4 once G3 is done (file-disjoint). |
| **G8 (NEW)** | **P11.1-P11.5 (Eval / visual / deploy parity)** | After G7 | `policy_contract/numpy/obs.py` + `training/eval/{eval_policy,eval_ladder_v0170,v6_eval_adapter,visualize_policy,visualize_reference_library}.py`. Mostly independent of G3-G6 file scope. **NEW (H4)**. |
| G9 | P8.1-P8.2 (smoke YAML + residual scales) | After G7 | Independent YAML. **P8.1 residual scales now drop `*_hip_yaw` entries (H6)**. |
| G10 | P9.1-P9.3 (integration + e2e tests + walking_training.md gates) | After G6+G7+G8+G9 | Cross-layer. |
| G11 | P10.1 (CHANGELOG) | After smoke run | Final. |

**File-conflict callout**: G3, G4, G5, G6 still serialize on
`training/envs/wildrobot_env.py`. G7 (`policy_contract/`) and G8 (numpy
mirror + eval scripts) touch different files and can run in parallel
with G4-G6 on separate branches.

---

## P1 — Extend ZMP planner with vy + wz axes

**File**: `control/zmp/zmp_walk.py`.

### P1.1 — Add `rotation_radius_m` to `ZMPWalkConfig` (failing test)

**Failing test**: `training/tests/test_zmp_walk_config_3d_cmd.py`
```python
from __future__ import annotations
import pytest
from control.zmp.zmp_walk import ZMPWalkConfig


def test_zmp_walk_config_has_rotation_radius_field() -> None:
    cfg = ZMPWalkConfig()
    assert cfg.rotation_radius_m == pytest.approx(0.10)


def test_zmp_walk_config_accepts_rotation_radius_override() -> None:
    cfg = ZMPWalkConfig(rotation_radius_m=0.15)
    assert cfg.rotation_radius_m == pytest.approx(0.15)
```

**Run**: `cd <wildrobot-repo> && uv run pytest training/tests/test_zmp_walk_config_3d_cmd.py -q` — expect 2 failures.

### P1.2 — Impl: add `rotation_radius_m` field

Add `rotation_radius_m: float = 0.10` to `ZMPWalkConfig` (near
`default_stance_width_m`, line 135) with derivation comment (unchanged
from `-final.md`).

**Verify**: rerun P1.1 — passes.

### P1.3 — `generate()` accepts `command_vy`, `command_yaw_rate` (failing test)

```python
from control.zmp.zmp_walk import ZMPWalkGenerator


def test_generate_accepts_vy_and_yaw_rate_kwargs() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.20, command_vy=0.05, command_yaw_rate=0.0)
    assert traj.command_vx == pytest.approx(0.20)
    assert traj.command_vy == pytest.approx(0.05)
    assert traj.command_yaw_rate == pytest.approx(0.0)


def test_generate_defaults_vy_and_yaw_to_zero() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.20)
    assert traj.command_vy == pytest.approx(0.0)
    assert traj.command_yaw_rate == pytest.approx(0.0)
```

### P1.4 — Impl: widen `generate()` signature (no behavior change yet)

In `control/zmp/zmp_walk.py:579` widen to:
```python
def generate(
    self,
    command_vx: float,
    command_vy: float = 0.0,
    command_yaw_rate: float = 0.0,
) -> ReferenceTrajectory:
```
Thread `command_vy` / `command_yaw_rate` into the
`ReferenceTrajectory(command_vx=..., command_vy=..., command_yaw_rate=...)`
constructions at lines 1043 and 1099.

**Verify**: P1.3 passes.

### P1.4a — H1 standing gate: replace scalar gate with 3-axis norm (failing test, then impl)

**Failing test** (append to `test_zmp_walk_config_3d_cmd.py`):
```python
def test_pure_lateral_command_does_not_short_circuit_to_standing() -> None:
    """H1: generate(vx=0, vy=0.05) MUST produce a non-standing trajectory.
    Under the old gate `abs(command_vx) < 1e-4`, this would return
    `_generate_standing()` and the lateral footstep block would never run.
    """
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.05, command_yaw_rate=0.0)
    foot_y = traj.left_foot_pos[:, 1]
    assert (foot_y.max() - foot_y.min()) > 0.001, (
        "H1 fix missing: pure-lateral command short-circuited to standing."
    )


def test_pure_yaw_command_does_not_short_circuit_to_standing() -> None:
    """H1: generate(vx=0, wz=0.10) MUST produce non-trivial yaw."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    yaw = traj.pelvis_rpy[:, 2]
    assert abs(yaw[-1] - yaw[0]) > 1e-3


def test_negative_vy_flips_lateral_stride_sign() -> None:
    """H1 follow-on: line 609 currently flips step_length on vx < 0 only.
    Verify the equivalent per-axis sign flip works for vy < 0 too."""
    gen = ZMPWalkGenerator()
    traj_pos = gen.generate(command_vx=0.0, command_vy=+0.05)
    traj_neg = gen.generate(command_vx=0.0, command_vy=-0.05)
    assert traj_pos.left_foot_pos[-1, 1] * traj_neg.left_foot_pos[-1, 1] < 0
```

**Run**: expect 3 failures (the old gate short-circuits all of them).

**Impl** in `control/zmp/zmp_walk.py:594-621`:
```python
cfg = self.cfg

# H1: 3-axis norm gate (was: `if abs(command_vx) < 1e-4`).  Pure-lateral
# and pure-turn commands must reach the per-direction footstep blocks
# below; only a fully-zero command returns the standing trajectory.
import math
cmd_norm = math.sqrt(
    command_vx * command_vx
    + command_vy * command_vy
    + command_yaw_rate * command_yaw_rate
)
if cmd_norm < 1e-4:
    return self._generate_standing()

# Per-direction stride magnitudes (mirror the existing `abs_vx` /
# `step_length` block, but now per-axis).  The `_generate_at_step_length`
# function receives all three components signed; the sign-flip logic
# below ensures each per-axis step length carries the correct sign for
# its branch.
abs_vx = abs(command_vx)
abs_vy = abs(command_vy)
abs_wz = abs(command_yaw_rate)
half_cycle = cfg.cycle_time_s / 2.0
step_length_x = min(abs_vx * half_cycle, cfg.safe_max_step_length_m)
step_length_y = min(abs_vy * half_cycle, cfg.safe_max_step_length_m)
cycle_time = cfg.cycle_time_s

# H1: per-axis sign flip (was: scalar `if command_vx < 0: step_length = -step_length`).
if command_vx < 0:
    step_length_x = -step_length_x
if command_vy < 0:
    step_length_y = -step_length_y
# command_yaw_rate sign is consumed directly by the arc-radius branch in
# P1.8 via `r = np.sign(command_yaw_rate) * cfg.rotation_radius_m`.
```

Thread `step_length_y` into `_generate_at_step_length(...)` as a new
positional arg `step_length_y: float = 0.0`. (The combined-translation
branch in P1.6 reads this; the pure-rotation branch in P1.8 ignores it.)

**Verify**: P1.4a's three tests pass; P1.3 / P1.5 / P1.7 still pass after
the upcoming impl steps.

### P1.5 — Lateral stride: failing test (tightened)

```python
def test_planner_produces_nonzero_lateral_foot_displacement_for_vy() -> None:
    """Tightened from -final.md: now uses vx=0 (was vx=0.0001) since the
    H1 gate fix lets a pure-lateral command through."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.05)
    foot_y = traj.left_foot_pos[:, 1]
    assert (foot_y.max() - foot_y.min()) > 0.005
```

### P1.6 — Impl: combined-translation footstep block (mirror TB:259-267)

Refactor `_generate_at_step_length` to use `(step_length_x, step_length_y)`
in the per-cycle stride. Mirrors TB:
```python
# TB zmp_walk.py:259-267 — combined translation.
# stride = command[:2] * total_time / (2 * num_cycles - 1)
stride_xy = np.array([step_length_x, step_length_y], dtype=np.float64) \
            * cfg.total_plan_time_s / (2 * n_cycles - 1)
left_xy  = left_xy_init  + 2 * i * stride_xy
right_xy = right_xy_init + (2 * i + 1) * stride_xy
```
Apply lateral component to `left_world[idx, 1]` / `right_world[idx, 1]`
in addition to the existing stance offset. Update the LQR COM-y target
from the new foot midpoint.

**Verify**: P1.5 passes. P1.3 / P1.4a still pass.

### P1.7 — Pure-rotation branch: failing test (tightened)

```python
def test_planner_yaws_pelvis_for_nonzero_yaw_rate() -> None:
    """Tightened: uses vx=vy=0 (was vy=0); requires the H1 gate fix."""
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    yaw = traj.pelvis_rpy[:, 2]
    assert abs(yaw[-1] - yaw[0]) > 1e-3
```

### P1.8 — Impl: pure-rotation arc-radius branch (mirror TB:269-305)

In `_generate_at_step_length` branch on
`(abs(command_vx) < eps and abs(command_vy) < eps and abs(command_yaw_rate) > eps)`.
Mirror TB exactly (full formula unchanged from `-final.md`).

**Verify**: P1.7 passes; P1.5, P1.3, P1.4a still pass.

### P1.9 — Trajectory carries 3D cmd: failing test (unchanged)

```python
def test_generated_trajectory_carries_all_three_cmd_axes() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.20, command_vy=0.04, command_yaw_rate=0.0)
    assert traj.command_vx == pytest.approx(0.20)
    assert traj.command_vy == pytest.approx(0.04)
    assert traj.command_yaw_rate == pytest.approx(0.0)
```

### P1.10 — `build_library_for_3d_values` factory: failing test

```python
def test_build_library_for_3d_values_yields_one_entry_per_grid_point() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.0, 0.20],  # H2: include static bin
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    assert len(lib) == 2 * 3 * 3 == 18
    assert (0.20, 0.0, 0.10) in lib
    assert (0.0, 0.0, 0.0) in lib, "H2: must include the static-pose bin"
```

### P1.11 — Impl: `build_library_for_3d_values` (unchanged)

(Function body identical to `-final.md` P1.11.)

**Verify**: P1.10 passes.

### P1.12 — Regression: forward-only library still builds (unchanged)

### P1.13 — Commit
```bash
git add control/zmp/zmp_walk.py training/tests/test_zmp_walk_config_3d_cmd.py
git commit -m "v0.21.0 P1: ZMP planner accepts (vy, wz), 3-axis gate (H1), 3D library factory"
```

---

## P2 — Verify 3D `ReferenceLibrary.lookup` + add metadata grids + measure

**File**: `control/references/reference_library.py`.

### P2.1 — Regression: `lookup` already accepts 3D key (unchanged)

### P2.2 — `ReferenceLibraryMeta` accepts `vy_grid` / `yaw_rate_grid` (unchanged)

### P2.3 — Impl: add `vy_grid` / `yaw_rate_grid` to `ReferenceLibraryMeta`

(Field additions unchanged.) Update `save()` / `load()` JSON round-trip.

> **H2 note**: documented in code comment that the canonical vx_grid for
> v0.21.0 includes `0.0` (the static bin). The static trajectory comes
> from `_generate_standing()` because `generate(0, 0, 0)` short-circuits
> there per the H1 gate.

### P2.4 — Library populates meta grids (unchanged)

### P2.5 — Impl + commit (unchanged)

```bash
git add control/references/reference_library.py training/tests/test_reference_library_3d.py
git commit -m "v0.21.0 P2: ReferenceLibraryMeta carries vy/yaw_rate grids; verify lookup already 3D"
```

### P2.5a — Library build-time + memory measurement (M3)

**File**: `training/tests/test_library_build_size.py`
```python
import time
import jax.numpy as jp
import pytest
from control.zmp.zmp_walk import ZMPWalkGenerator


def test_build_100bin_library_measure_time_and_memory(capsys) -> None:
    """M3: one-shot measurement.  Number is recorded in the smoke1 PR
    description.  Test asserts loose budgets so it doesn't flake on slow
    CI runners; tighten only if regression is observed."""
    gen = ZMPWalkGenerator()
    t0 = time.perf_counter()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.0, 0.18, 0.22, 0.26],
        vy_values=[-0.10, -0.05, 0.0, 0.05, 0.10],
        yaw_rate_values=[-0.20, -0.10, 0.0, 0.10, 0.20],
    )
    elapsed_s = time.perf_counter() - t0
    n_bins = len(lib)
    assert n_bins == 100, f"H2: expected 4*5*5=100 bins, got {n_bins}"
    # Sum nbytes across the per-trajectory JAX arrays (approximate device
    # footprint; exact figure printed for the PR description).
    total_bytes = sum(
        arr.nbytes
        for traj in lib._entries.values()
        for arr in (traj.q_ref, traj.pelvis_pos, traj.left_foot_pos,
                    traj.right_foot_pos, traj.pelvis_rpy)
        if hasattr(arr, "nbytes")
    )
    mb = total_bytes / (1024 * 1024)
    print(
        f"\n[v0.21.0 P2.5a] library: bins={n_bins}, build_s={elapsed_s:.2f}, "
        f"approx_mb={mb:.2f}"
    )
    # Loose budgets — bump only if intentionally enlarged.
    assert elapsed_s < 300.0, f"build too slow: {elapsed_s:.1f}s"
    assert mb < 200.0, f"library too large: {mb:.1f} MB"
```

**Run**: `cd <wildrobot-repo> && uv run pytest training/tests/test_library_build_size.py -q -s`
and copy the printed line into the smoke1 PR description.

**Verify** (commit with P2.5):
```bash
git add training/tests/test_library_build_size.py
git commit -m "v0.21.0 P2.5a: measure 100-bin library build time + memory (M3)"
```

---

## P3 — Migrate `velocity_cmd` from scalar to 3-vector + path-state fields (H7)

**Files**: `training/envs/env_info.py`, `training/envs/wildrobot_env.py`,
`training/configs/training_runtime_config.py`,
`training/configs/training_config.py`,
`control/references/runtime_reference_service.py`.

> Foundational data shape change. MUST land in one commit. All sub-steps
> share one commit at P3.10d.

> **NEW-1 resolution** (inherited from `-final.md`): P3 leaves
> `wildrobot_env.py:1218` untouched. Edit lives in P6.2.

### P3.1 — `WildRobotInfo.velocity_cmd` shape regression test (unchanged)

### P3.2 — Impl: bump `velocity_cmd` shape in env_info (unchanged)

In `training/envs/env_info.py:107` and `:222` change comment `# ()` -> `# (3,)`.
In `get_expected_shapes()` at line 285 set `"velocity_cmd": (3,)`.

### P3.3 — `EnvConfig.cmd_deadzone` type bump (unchanged)

### P3.4 — Impl: cmd_deadzone tuple migration (unchanged)

In `training/configs/training_runtime_config.py:625` change to
`cmd_deadzone: Tuple[float, float, float] = (0.0, 0.0, 0.0)`.

### P3.4a — REPLACE `float(...)` casts at `training/configs/training_config.py:549-550` with `_load_three_axis` (H3 + NEW-4)

**Failing test** (append to `training/tests/test_velocity_cmd_3d_migration.py`):
```python
def test_load_three_axis_accepts_scalar_and_broadcasts_for_deadzone() -> None:
    from training.configs.training_config import _load_three_axis
    out = _load_three_axis(
        0.05, default_scalar=0.0, scalar_broadcast_axis="all",
    )
    assert out == (0.05, 0.05, 0.05)


def test_load_three_axis_scalar_broadcast_for_eval_cmd() -> None:
    """H3: scalar `0.26` for eval_velocity_cmd must map to (0.26, 0.0, 0.0),
    NOT (0.26, 0.26, 0.26).  vy / wz default to 0.0."""
    from training.configs.training_config import _load_three_axis
    out = _load_three_axis(
        0.26, default_scalar=-1.0, scalar_broadcast_axis="vx_only",
    )
    assert out == (0.26, 0.0, 0.0)


def test_load_three_axis_negative_scalar_for_eval_cmd_sentinel() -> None:
    """H3: -1.0 sentinel scalar -> (-1.0, 0.0, 0.0).  Sentinel detection
    in env code reads only the [0]th axis (vx) per H3."""
    from training.configs.training_config import _load_three_axis
    out = _load_three_axis(
        -1.0, default_scalar=-1.0, scalar_broadcast_axis="vx_only",
    )
    assert out == (-1.0, 0.0, 0.0)


def test_load_three_axis_accepts_three_tuple() -> None:
    from training.configs.training_config import _load_three_axis
    out = _load_three_axis(
        [0.02, 0.03, 0.04], default_scalar=0.0, scalar_broadcast_axis="all",
    )
    assert out == (0.02, 0.03, 0.04)


def test_load_three_axis_uses_default_on_none_for_deadzone() -> None:
    from training.configs.training_config import _load_three_axis
    out = _load_three_axis(
        None, default_scalar=0.0, scalar_broadcast_axis="all",
    )
    assert out == (0.0, 0.0, 0.0)


def test_load_three_axis_uses_default_on_none_for_eval_cmd() -> None:
    from training.configs.training_config import _load_three_axis
    out = _load_three_axis(
        None, default_scalar=-1.0, scalar_broadcast_axis="vx_only",
    )
    assert out == (-1.0, 0.0, 0.0)


def test_load_three_axis_rejects_two_tuple() -> None:
    from training.configs.training_config import _load_three_axis
    import pytest
    with pytest.raises(ValueError, match="length 3"):
        _load_three_axis(
            [0.02, 0.03], default_scalar=0.0, scalar_broadcast_axis="all",
        )
```

**Impl**: in `training/configs/training_config.py` above
`load_training_config_from_dict`:
```python
def _load_three_axis(
    value: Any,
    *,
    default_scalar: float,
    scalar_broadcast_axis: str = "all",
) -> Tuple[float, float, float]:
    """Coerce a scalar OR length-3 list/tuple to a (float, float, float).

    scalar_broadcast_axis controls scalar handling (H3):
      * "all"     -> (s, s, s)  — for symmetric fields like cmd_deadzone.
      * "vx_only" -> (s, 0.0, 0.0) — for eval_velocity_cmd: the legacy
                      scalar form pins vx only; vy / wz pin to 0.0.

    Length-3 list/tuple always passes through unchanged; None -> broadcast
    default per the same axis rule.
    """
    if value is None:
        return _broadcast_scalar(default_scalar, scalar_broadcast_axis)
    if isinstance(value, (int, float)):
        return _broadcast_scalar(float(value), scalar_broadcast_axis)
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"three-axis field must have length 3 (got {len(value)}: {value!r})"
            )
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError(
        f"three-axis field must be a number or length-3 list/tuple "
        f"(got {type(value).__name__}: {value!r})"
    )


def _broadcast_scalar(s: float, mode: str) -> Tuple[float, float, float]:
    if mode == "all":
        return (s, s, s)
    if mode == "vx_only":
        return (s, 0.0, 0.0)
    raise ValueError(f"unknown scalar_broadcast_axis: {mode!r}")
```

Replace lines 549-550 of `training/configs/training_config.py`:
```python
cmd_deadzone=_load_three_axis(
    env.get("cmd_deadzone"),
    default_scalar=0.0,
    scalar_broadcast_axis="all",
),
eval_velocity_cmd=_load_three_axis(
    env.get("eval_velocity_cmd"),
    default_scalar=-1.0,
    scalar_broadcast_axis="vx_only",  # H3: scalar -> (s, 0, 0)
),
```

Also bump the default in `training/configs/training_runtime_config.py:635`:
```python
eval_velocity_cmd: Tuple[float, float, float] = (-1.0, 0.0, 0.0)  # H3
```

**Integration check** (append):
```python
def test_smoke14_yaml_eval_cmd_scalar_broadcasts_to_vx_only() -> None:
    """H3: smoke14's scalar `eval_velocity_cmd: 0.26` MUST map to
    (0.26, 0.0, 0.0), NOT (0.26, 0.26, 0.26)."""
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    assert tuple(cfg.env.eval_velocity_cmd) == pytest.approx((0.26, 0.0, 0.0))
    assert tuple(cfg.env.cmd_deadzone) == pytest.approx(
        (0.0666666667, 0.0666666667, 0.0666666667), abs=1e-7,
    )
```

### P3.5 — `EnvConfig.eval_velocity_cmd` default-tuple test (updated for H3)

```python
def test_env_config_eval_velocity_cmd_default_is_vx_only_sentinel() -> None:
    """H3: default tuple is (-1.0, 0.0, 0.0): vx uses the sentinel, vy/wz
    default to 0.0 (not -1.0)."""
    from training.configs.training_runtime_config import LocomotionEnvConfig
    cfg = LocomotionEnvConfig()
    assert tuple(cfg.eval_velocity_cmd) == (-1.0, 0.0, 0.0)
```

### P3.6 — Impl: eval_velocity_cmd sentinel rewrite (H3-aware)

In `training/envs/wildrobot_env.py:2529-2532` replace:
```python
# H3: sentinel applies to vx (index 0) only.  vy / wz default to 0.0;
# they are NOT a "use sampled cmd" signal — if a YAML wants to override
# only vx, it writes a scalar (legacy YAML) or `[vx, 0.0, 0.0]` (new
# explicit form).
eval_cmd = jp.asarray(self._config.env.eval_velocity_cmd, dtype=jp.float32)  # (3,)
use_override = eval_cmd[0] >= jp.float32(0.0)
new_cmd = jp.where(use_override, eval_cmd, wr.velocity_cmd)  # (3,)
new_wr = wr.replace(velocity_cmd=new_cmd.astype(jp.float32))
```

> **Why `eval_cmd[0]` and not `jp.all(eval_cmd >= 0.0)` (as `-final.md`
> said)**: under H3, the sentinel is vx-only. A YAML writer who wants
> to pin a positive vx and a NEGATIVE vy (rare, but valid) should still
> get the override — the `jp.all(... >= 0)` reduction would silently fail
> that. Reading only the [0]th axis matches the legacy scalar semantics.

### P3.7 — Env step fanout: scalar usages of `velocity_cmd` (unchanged)

### P3.8 — Impl: rewrite each scalar callsite (unchanged from -final.md)

See `-final.md` P3.8 for the full callsite list (every per-axis metric,
norm guard, kwarg pass-through, etc.). No deviations from `-final.md` here
— the audit checklist (M2) is dropped but the per-callsite tests stay.

### P3.8a — Per-callsite pass-through shape test (unchanged from -final.md, but now the SOLE enforcement mechanism per M2)

(Test body unchanged from `-final.md` P3.8a. Now load-bearing because P0.5
has been deleted.)

### P3.9 — `compute_command_integrated_path_state` 3D extension (failing test)

(Unit tests unchanged from `-final.md` P3.9 — they remain the analytic
ground truth for constant-cmd integration. Add ONE new test:)

```python
def test_incremental_path_state_matches_closed_form_for_constant_cmd() -> None:
    """H7: incremental_path_state_step iterated N times must match the
    closed-form helper when cmd is constant.  This is the equivalence
    proof that lets us use the incremental form at runtime without
    losing parity with TB's analytic prior."""
    import jax.numpy as jp
    from control.references.runtime_reference_service import RuntimeReferenceService
    dt = 0.02
    vx, vy, wz = 0.10, 0.05, 0.10
    n = 50
    torso_pos = jp.zeros(3, dtype=jp.float32)
    path_rot = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
    cmd = jp.array([vx, vy, wz], dtype=jp.float32)
    for _ in range(n):
        torso_pos, path_rot = RuntimeReferenceService.incremental_path_state_step(
            prev_torso_pos=torso_pos,
            prev_path_rot_wxyz=path_rot,
            velocity_cmd=cmd,
            dt_s=dt,
        )
    closed = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=n * dt,
        velocity_cmd_mps=vx,
        velocity_cmd_y_mps=vy,
        yaw_rate_cmd_rps=wz,
        dt_s=dt,
    )
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(torso_pos), np.asarray(closed["path_pos"]), atol=1e-5,
    )
```

### P3.10 — Impl: extend `compute_command_integrated_path_state` (NEW-5 explicit math, unchanged from -final.md)

(2D closed-form body identical to `-final.md` P3.10.)

### P3.10a — H7: add `path_state_torso_pos` + `path_state_path_rot` to `WildRobotInfo` (failing test, impl)

**Failing test** (append to `training/tests/test_velocity_cmd_3d_migration.py`):
```python
def test_wildrobot_info_carries_incremental_path_state() -> None:
    """H7: path_state_torso_pos (3,) and path_state_path_rot (4,) wxyz."""
    from training.envs.env_info import get_expected_shapes
    shapes = get_expected_shapes(action_size=13)
    assert shapes["path_state_torso_pos"] == (3,)
    assert shapes["path_state_path_rot"] == (4,)
```

**Impl** in `training/envs/env_info.py`:
- Add to the flax.struct dataclass `WildRobotInfo` (after `velocity_cmd`
  near line 107):
  ```python
  # H7: incremental path-state for runtime-correct integration under
  # cmd_resample_steps > 0.  Updated in `step` from the sampled
  # velocity_cmd via `incremental_path_state_step`.
  path_state_torso_pos: jnp.ndarray   # (3,) world-frame torso target
  path_state_path_rot: jnp.ndarray    # (4,) wxyz integrated rotation
  ```
- Mirror in the NamedTuple fallback (line 222 area).
- Add to `get_expected_shapes()` (line 285 area):
  ```python
  "path_state_torso_pos": (3,),
  "path_state_path_rot": (4,),
  ```

### P3.10b — H7: initialize path-state fields at reset

In `training/envs/wildrobot_env.py::_make_initial_state` (near line 2635),
add to the `WildRobotInfo(...)` construction:
```python
path_state_torso_pos=jp.zeros(3, dtype=jp.float32),
path_state_path_rot=jp.asarray(
    [1.0, 0.0, 0.0, 0.0], dtype=jp.float32,
),
```

**Test** (append):
```python
def test_reset_initializes_path_state_to_origin_and_identity() -> None:
    import jax
    from training.envs.wildrobot_env import WildRobotEnv
    from training.envs.env_info import WR_INFO_KEY
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    import numpy as np
    np.testing.assert_allclose(np.asarray(wr.path_state_torso_pos), np.zeros(3))
    np.testing.assert_allclose(
        np.asarray(wr.path_state_path_rot), np.array([1.0, 0.0, 0.0, 0.0]),
    )
```

### P3.10c — H7: add `incremental_path_state_step` static helper

**Failing test** (append):
```python
def test_incremental_step_translates_along_world_x_under_identity_rotation() -> None:
    import jax.numpy as jp
    from control.references.runtime_reference_service import RuntimeReferenceService
    new_pos, new_rot = RuntimeReferenceService.incremental_path_state_step(
        prev_torso_pos=jp.zeros(3, dtype=jp.float32),
        prev_path_rot_wxyz=jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32),
        velocity_cmd=jp.array([0.10, 0.0, 0.0], dtype=jp.float32),
        dt_s=0.02,
    )
    assert float(new_pos[0]) == pytest.approx(0.10 * 0.02, abs=1e-7)
    assert float(new_pos[1]) == pytest.approx(0.0, abs=1e-7)
    # Identity rot unchanged since wz=0:
    assert float(new_rot[0]) == pytest.approx(1.0, abs=1e-7)
    assert float(new_rot[3]) == pytest.approx(0.0, abs=1e-7)


def test_incremental_step_rotates_around_z_under_pure_yaw_cmd() -> None:
    import math, jax.numpy as jp
    from control.references.runtime_reference_service import RuntimeReferenceService
    _, new_rot = RuntimeReferenceService.incremental_path_state_step(
        prev_torso_pos=jp.zeros(3, dtype=jp.float32),
        prev_path_rot_wxyz=jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32),
        velocity_cmd=jp.array([0.0, 0.0, 0.10], dtype=jp.float32),
        dt_s=0.02,
    )
    expected_qz = math.sin(0.10 * 0.02 / 2.0)
    assert float(new_rot[3]) == pytest.approx(expected_qz, abs=1e-6)
```

**Impl** in `control/references/runtime_reference_service.py` (alongside
the existing closed-form helper):
```python
@staticmethod
def incremental_path_state_step(
    *,
    prev_torso_pos,
    prev_path_rot_wxyz,
    velocity_cmd,
    dt_s,
):
    """One-step incremental path-state update (H7).

    Mirrors TB `toddlerbot/reference/motion_ref.py::integrate_path_state`:
        delta_rot = axis_angle_quat([0, 0, 1], wz * dt)   # wxyz
        path_rot = quat_mul(prev_path_rot, delta_rot)     # wxyz
        lin_vel_world = rotate_vec_by_quat([vx, vy, 0], path_rot)
        torso_pos = prev_torso_pos + lin_vel_world * dt

    Used at runtime under `cmd_resample_steps > 0` where the closed-form
    integrator drifts because cmd changes mid-episode.  The closed form
    stays for unit tests (it's the analytic ground truth for constant
    cmd) and library-build paths.
    """
    import jax.numpy as jnp
    cmd = jnp.asarray(velocity_cmd, dtype=jnp.float32)
    dt = jnp.asarray(dt_s, dtype=jnp.float32)
    prev_pos = jnp.asarray(prev_torso_pos, dtype=jnp.float32)
    prev_rot = jnp.asarray(prev_path_rot_wxyz, dtype=jnp.float32)
    vx, vy, wz = cmd[0], cmd[1], cmd[2]
    half = 0.5 * wz * dt
    delta_rot = jnp.asarray(
        [jnp.cos(half), 0.0, 0.0, jnp.sin(half)], dtype=jnp.float32,
    )  # wxyz
    # quat_mul (wxyz): (w1, v1) * (w2, v2)
    #   w = w1*w2 - v1 . v2
    #   v = w1*v2 + w2*v1 + v1 x v2
    w1, x1, y1, z1 = prev_rot[0], prev_rot[1], prev_rot[2], prev_rot[3]
    w2, x2, y2, z2 = delta_rot[0], delta_rot[1], delta_rot[2], delta_rot[3]
    new_w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
    new_x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
    new_y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
    new_z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
    new_rot = jnp.stack([new_w, new_x, new_y, new_z]).astype(jnp.float32)
    # rotate body-frame [vx, vy, 0] by new_rot (wxyz).  Use the
    # standard q * v * q^{-1} expansion for the z-only-rotation case
    # (yaw-only), which collapses to a 2D rotation in xy:
    #   yaw = 2 * atan2(new_z, new_w)   (under axis = +z)
    yaw = 2.0 * jnp.arctan2(new_z, new_w)
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    vw_x = c * vx - s * vy
    vw_y = s * vx + c * vy
    lin_vel_world = jnp.stack([vw_x, vw_y, jnp.float32(0.0)])
    new_pos = (prev_pos + lin_vel_world * dt).astype(jnp.float32)
    return new_pos, new_rot
```

### P3.10d — H7: wire incremental update into `step` + commit

In `training/envs/wildrobot_env.py:2962-2968` replace the closed-form
call with an incremental update:
```python
# H7: use the incremental helper so cmd resampling at smoke14's
# cmd_resample_steps=150 produces a continuous path-state instead of
# integrating the new cmd back from t=0.
path_state_torso_pos, path_state_path_rot = (
    self._offline_service.incremental_path_state_step(
        prev_torso_pos=wr.path_state_torso_pos,
        prev_path_rot_wxyz=wr.path_state_path_rot,
        velocity_cmd=velocity_cmd,
        dt_s=jp.float32(self.dt),
    )
)
# Compose dict in the same shape `_compute_reward_terms` expects so the
# downstream call sites don't need to know whether the integrator was
# closed-form or incremental.
path_state = {
    "path_pos": path_state_torso_pos - self._init_qpos[:3].astype(jp.float32),
    "path_rot": path_state_path_rot,
    "torso_pos": path_state_torso_pos,
    "lin_vel": jp.asarray(
        [velocity_cmd[0], velocity_cmd[1], 0.0], dtype=jp.float32,
    ),
    "ang_vel": jp.asarray(
        [0.0, 0.0, velocity_cmd[2]], dtype=jp.float32,
    ),
}
```

At the bottom of `step` where the new `WildRobotInfo` is written, persist
the updated path-state into the new info fields:
```python
new_wr = wr.replace(
    ...,
    path_state_torso_pos=path_state_torso_pos,
    path_state_path_rot=path_state_path_rot,
    ...,
)
```

Existing closed-form test callers
(`tests/test_runtime_reference_service.py:188/232/269`) stay scalar —
the closed form is unchanged, only the env's call site moves to the
incremental form.

**End-to-end check** (append):
```python
def test_path_state_continuity_under_cmd_resample() -> None:
    """H7: a mid-episode cmd resample must NOT reset path_state to origin.
    Run 200 steps with cmd_resample_steps=10 and assert torso_pos grows
    monotonically (constant-positive vx)."""
    import jax, jax.numpy as jp, dataclasses
    from training.envs.wildrobot_env import WildRobotEnv
    from training.envs.env_info import WR_INFO_KEY
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    cfg.env = dataclasses.replace(cfg.env, cmd_resample_steps=10)
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    last_x = float(state.info[WR_INFO_KEY].path_state_torso_pos[0])
    for _ in range(50):
        state = env.step(state, jp.zeros(env.action_size))
        cur_x = float(state.info[WR_INFO_KEY].path_state_torso_pos[0])
        # Under any non-zero cmd, x should grow (or hold if cmd resamples
        # to zero).  Assertion: never goes negative discontinuously.
        assert cur_x >= last_x - 0.05, (
            "path_state_torso_pos reset / drifted under cmd resample"
        )
        last_x = cur_x
```

**Commit**:
```bash
git add training/envs/env_info.py training/envs/wildrobot_env.py \
        training/configs/training_runtime_config.py training/configs/training_config.py \
        control/references/runtime_reference_service.py \
        training/core/metrics_registry.py \
        training/tests/test_velocity_cmd_3d_migration.py
git commit -m "v0.21.0 P3: WildRobotInfo.velocity_cmd -> (3,); incremental path-state (H3, H7)"
```

---

## P4 — Branched 3D command sampler (mirror TB exactly)

(Unchanged from `-final.md` modulo M1 path replacements. P4.5 still
uses `_config.env.cmd_deadzone[0]` / `[1]` and `_config.env.cmd_deadzone[2]`
for the per-axis lower bounds. The sampler still returns a `(3,)`
vector.)

**Commit**:
```bash
git add training/configs/training_runtime_config.py training/envs/wildrobot_env.py \
        training/configs/training_config.py \
        training/tests/test_env_config_3d_cmd.py training/tests/test_sample_velocity_cmd_3d.py
git commit -m "v0.21.0 P4: branched (zero/turn/walk) 3D cmd sampler mirroring TB"
```

---

## P5 — 3D reference library plumbing in env

**Files**: `training/envs/wildrobot_env.py`,
`control/references/runtime_reference_service.py`,
`training/configs/training_runtime_config.py`.

### P5.1 — `loc_ref_command_axes_3d` toggle + 3D grid fields (unchanged from -final.md)

### P5.2 — Impl: add three fields (unchanged from -final.md)

### P5.3 — `_init_offline_service` builds 3D grid (failing test — updated for H2)

```python
def test_init_offline_service_builds_3d_grid_when_toggled() -> None:
    """H2: smoke1 YAML's vx grid now includes 0.0, so the bin count is
    4 (vx) * 5 (vy) * 5 (wz) = 100."""
    from training.envs.wildrobot_env import WildRobotEnv
    from training.configs.training_config import load_training_config
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml"
    )
    env = WildRobotEnv(cfg)
    assert env._offline_cmd_keys.shape == (100, 3), (
        "H2: expected 4*5*5=100 bins (vx grid includes 0.0)"
    )


def test_offline_cmd_keys_include_static_bin() -> None:
    """H2: a (0, 0, 0) static bin is present so static / pure-turn
    commands snap to it instead of to the nearest forward-walking bin."""
    from training.envs.wildrobot_env import WildRobotEnv
    from training.configs.training_config import load_training_config
    import numpy as np
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml"
    )
    env = WildRobotEnv(cfg)
    keys = np.asarray(env._offline_cmd_keys)
    static_present = np.any(np.linalg.norm(keys, axis=-1) < 1e-6)
    assert static_present, "H2: missing (0, 0, 0) static bin"
```

### P5.4 — Impl: `_init_offline_service` 3D branch (updated for H2 explicit-static guard)

In `training/envs/wildrobot_env.py:820-848`:
```python
if cmd_conditioned and bool(getattr(self._config.env, "loc_ref_command_axes_3d", False)):
    vx_values = list(vx_grid)
    vy_values = list(self._config.env.loc_ref_offline_command_vy_grid) or [0.0]
    yaw_rate_values = list(
        self._config.env.loc_ref_offline_command_yaw_rate_grid
    ) or [0.0]
    # H2: enforce the static-bin invariant.  If the YAML grid omits 0.0,
    # the (0, 0, 0) static command would snap to the nearest forward
    # bin and play back a walking trajectory at zero-cmd reset — wrong.
    assert 0.0 in vx_values or any(abs(v) < 1e-9 for v in vx_values), (
        "H2: loc_ref_offline_command_vx_grid must include 0.0 so the "
        "static and pure-turn bins exist.  Either set "
        "loc_ref_offline_command_vx: 0.0 and the right interval, or "
        "explicitly include 0.0 in the derived vx_grid."
    )
    lib = ZMPWalkGenerator().build_library_for_3d_values(
        vx_values=vx_values,
        vy_values=vy_values,
        yaw_rate_values=yaw_rate_values,
    )
    cmd_keys = [
        (vx, vy, wz)
        for vx in vx_values
        for vy in vy_values
        for wz in yaw_rate_values
    ]
    expected_count = len(vx_values) * len(vy_values) * len(yaw_rate_values)
    assert len(cmd_keys) == expected_count, (
        f"S6 guard: cmd_keys count {len(cmd_keys)} != expected {expected_count}"
    )
else:
    if offline_path:
        lib = ReferenceLibrary.load(offline_path)
    else:
        lib = ZMPWalkGenerator().build_library_for_vx_values(list(vx_grid))
    cmd_keys = [(vx, 0.0, 0.0) for vx in vx_grid]

self._offline_cmd_keys = jp.asarray(cmd_keys, dtype=jp.float32)  # (n_bins, 3)
```

**Verify**: P5.3 passes.

### P5.4a — Add `_yaw_from_quat_wxyz` JAX helper (unchanged from -final.md)

### P5.5 — `_lookup_offline_window` heading-frame correction + 3D nearest-key (unchanged from -final.md)

(JAX-native quat ops, optional `path_rot_wxyz` / `heading_rot_wxyz` kwargs.
Smoke1 callers MUST pass both.)

**Commit**:
```bash
git add training/configs/training_runtime_config.py training/configs/training_config.py \
        training/envs/wildrobot_env.py \
        control/references/runtime_reference_service.py \
        training/tests/test_loc_ref_3d_config.py
git commit -m "v0.21.0 P5: 3D nearest-key lookup + heading-frame correction + static-bin guard (H2)"
```

---

## P6 — Reward updates: vy_cmd + yaw_rate tracking term (H5)

**File**: `training/envs/wildrobot_env.py`.

### P6.1 — Add optional `vy_cmd` arg test (unchanged from -final.md)

### P6.2 — Impl: add `vy_cmd` kwarg + edit line 1218 (unchanged from -final.md, NEW-1 resolution)

### P6.3 — Wire real `vy_cmd` from the 3-vec at the call site (unchanged from -final.md)

### P6.4 — Yaw-rate tracking term (H5 alpha fix)

**Failing test**:
```python
def test_yaw_rate_alpha_field_exists_with_tb_aligned_default() -> None:
    """H5: cmd_yaw_rate_alpha is a NEW field separate from
    cmd_forward_velocity_alpha.  TB walk.gin: ang_vel_tracking_sigma=4.0
    -> WR numerator-alpha = 1/4.0 = 0.25."""
    from training.configs.training_runtime_config import RewardWeightsConfig
    rw = RewardWeightsConfig()
    assert rw.cmd_yaw_rate_alpha == pytest.approx(0.25)


def test_yaw_rate_track_reward_peaks_at_cmd() -> None:
    from training.envs.wildrobot_env import WildRobotEnv
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.10,
        yaw_rate_cmd=0.10,
        alpha=0.25,
    )
    assert float(reward) == pytest.approx(1.0)


def test_yaw_rate_track_reward_decays_off_cmd_with_tb_alpha() -> None:
    """H5: with alpha=0.25 (TB-aligned), err=0.10 gives
    reward = exp(-0.25 * 0.01) = exp(-0.0025) approx 0.9975.
    The -final.md test used alpha=562.5 (wrong) which would give exp(-5.625)."""
    import math
    from training.envs.wildrobot_env import WildRobotEnv
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.0,
        yaw_rate_cmd=0.10,
        alpha=0.25,
    )
    assert float(reward) == pytest.approx(math.exp(-0.0025), rel=1e-5)
```

**Impl**:
1. Add static helper to `training/envs/wildrobot_env.py`:
   ```python
   @staticmethod
   def _yaw_rate_track_reward(
       *, ang_vel_z: jax.Array, yaw_rate_cmd: jax.Array, alpha: jax.Array,
   ) -> jax.Array:
       err = ang_vel_z - yaw_rate_cmd
       return jp.exp(-alpha * err * err).astype(jp.float32)
   ```
2. Wire into `_compute_reward_terms` (after the existing cmd_forward term):
   ```python
   r_yaw_rate_track = WildRobotEnv._yaw_rate_track_reward(
       ang_vel_z=ang_vel_h[2],
       yaw_rate_cmd=velocity_cmd[2],
       alpha=self._config.reward_weights.cmd_yaw_rate_alpha,  # H5: NEW field
   )
   ```
3. Add `cmd_yaw_rate_alpha: float = 0.25` to `RewardWeightsConfig` in
   `training/configs/training_runtime_config.py` (near line 996, right
   after `cmd_forward_velocity_alpha`):
   ```python
   # H5: separate alpha for yaw-rate tracking.  TB walk.gin:
   #   lin_vel_tracking_sigma=1000.0  -> WR alpha=1/1000=0.001
   #   ang_vel_tracking_sigma=4.0     -> WR alpha=1/4.0=0.25
   # WR keeps its tighter lin-vel alpha (4.0) by historical choice, but
   # adopts TB's broader yaw-rate width directly.
   cmd_yaw_rate_alpha: float = 0.25
   ```
4. Add `cmd_yaw_rate_track: float = 0.0` to `RewardWeightsConfig` (default
   weight 0.0 for back-compat). Apply weight in the reward aggregator:
   `+ w.cmd_yaw_rate_track * terms["r_yaw_rate_track"]`.
5. Extend `training/configs/training_config.py` reward-block loader to
   read `float(rw.get("cmd_yaw_rate_track", 0.0))` AND
   `float(rw.get("cmd_yaw_rate_alpha", 0.25))`.
6. Add metric `tracking/yaw_rate_err` to `training/core/metrics_registry.py`
   with `Reducer.MEAN`.

**Verify**: P6.4 tests pass; pre-existing tests still pass.

**Commit**:
```bash
git add training/envs/wildrobot_env.py training/configs/training_runtime_config.py \
        training/configs/training_config.py \
        training/core/metrics_registry.py training/tests/test_cmd_lateral_velocity_tracking.py
git commit -m "v0.21.0 P6: reward consumes vy_cmd (line 1218); yaw_rate term with cmd_yaw_rate_alpha (H5)"
```

---

## P7 — Observation layout `wr_obs_v8_cmd3d` (unchanged from -final.md)

(Sections P7.1-P7.4 unchanged from `-final.md`.)

**Commit**:
```bash
git add policy_contract/spec.py policy_contract/spec_builder.py policy_contract/jax/obs.py \
        training/envs/wildrobot_env.py training/tests/test_obs_v8_cmd3d.py
git commit -m "v0.21.0 P7: register wr_obs_v8_cmd3d; new velocity_cmd_lateral_yaw kwarg"
```

---

## P11 — Eval / visual / deploy parity (NEW — H4)

**Goal**: every code path that builds a policy observation or sets a
`velocity_cmd` outside the JAX env (numpy mirror, eval, visualize)
must accept the new (3,) shape so a v8-trained policy can actually be
run end-to-end.

**Files**:
- `policy_contract/numpy/obs.py`
- `training/eval/eval_policy.py`
- `training/eval/eval_ladder_v0170.py`
- `training/eval/v6_eval_adapter.py`
- `training/eval/visualize_policy.py`
- `training/eval/visualize_reference_library.py` (only if it touches `velocity_cmd`; verified by grep first)

### P11.1 — `policy_contract/numpy/obs.py` mirror (failing test)

**Failing test**: `training/tests/test_numpy_obs_v8_parity.py`
```python
import numpy as np
import pytest
from policy_contract.spec_builder import build_policy_spec


def _spec_v8():
    return build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {"name": f"j{i}", "range": [-1.0, 1.0],
             "policy_action_sign": 1.0, "max_velocity_rad_s": 10.0}
            for i in range(13)
        ],
        action_filter_alpha=0.5,
        layout_id="wr_obs_v8_cmd3d",
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * 13,
    )


def test_numpy_build_observation_accepts_v8_layout() -> None:
    from policy_contract.numpy.obs import build_observation_from_components
    spec = _spec_v8()
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    proprio_history = np.zeros((15 * proprio_bundle,), dtype=np.float32)
    obs = build_observation_from_components(
        spec=spec,
        gravity_local=np.zeros(3, dtype=np.float32),
        angvel_heading_local=np.zeros(3, dtype=np.float32),
        joint_pos_normalized=np.zeros(action_dim, dtype=np.float32),
        joint_vel_normalized=np.zeros(action_dim, dtype=np.float32),
        foot_switches=np.zeros(4, dtype=np.float32),
        prev_action=np.zeros(action_dim, dtype=np.float32),
        velocity_cmd=np.array([0.20, 0.05, 0.10], dtype=np.float32),
        velocity_cmd_lateral_yaw=np.array([0.05, 0.10], dtype=np.float32),
        loc_ref_phase_sin_cos=np.zeros(2, dtype=np.float32),
        proprio_history=proprio_history,
    )
    assert obs.shape == (spec.model.obs_dim,)


def test_numpy_v7_obs_byte_identical_when_velocity_cmd_passed_as_3vec() -> None:
    """H4: v7 numpy obs must produce byte-identical output for (1,) and
    (3,) cmd inputs, same as the JAX path."""
    from policy_contract.numpy.obs import build_observation_from_components
    spec_v7 = build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[
            {"name": f"j{i}", "range": [-1.0, 1.0],
             "policy_action_sign": 1.0, "max_velocity_rad_s": 10.0}
            for i in range(13)
        ],
        action_filter_alpha=0.5,
        layout_id="wr_obs_v7_phase_proprio",
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * 13,
    )
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    proprio_history = np.zeros((15 * proprio_bundle,), dtype=np.float32)
    kw = dict(
        spec=spec_v7,
        gravity_local=np.zeros(3, dtype=np.float32),
        angvel_heading_local=np.zeros(3, dtype=np.float32),
        joint_pos_normalized=np.zeros(action_dim, dtype=np.float32),
        joint_vel_normalized=np.zeros(action_dim, dtype=np.float32),
        foot_switches=np.zeros(4, dtype=np.float32),
        prev_action=np.zeros(action_dim, dtype=np.float32),
        loc_ref_phase_sin_cos=np.zeros(2, dtype=np.float32),
        proprio_history=proprio_history,
    )
    obs_scalar = build_observation_from_components(
        velocity_cmd=np.array([0.20], dtype=np.float32), **kw,
    )
    obs_3vec = build_observation_from_components(
        velocity_cmd=np.array([0.20, 0.05, 0.10], dtype=np.float32),
        velocity_cmd_lateral_yaw=np.array([0.05, 0.10], dtype=np.float32),
        **kw,
    )
    np.testing.assert_array_equal(obs_scalar, obs_3vec)
```

**Impl** in `policy_contract/numpy/obs.py`:
- Add `velocity_cmd_lateral_yaw: np.ndarray | None = None` to
  `build_observation_from_components` signature (after `velocity_cmd`).
- Add `"wr_obs_v8_cmd3d"` to the allow-list set at line 43-48.
- Change line 148 from
  `np.asarray(velocity_cmd, dtype=np.float32).reshape(1)`
  to
  `np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)[..., :1].reshape(1)`
  (slice-first, keeps v1-v7 byte-identical when fed a (3,) input).
- After the `wr_obs_v7_phase_proprio` block (around line 184), add the v8
  branch:
  ```python
  if spec.observation.layout_id == "wr_obs_v8_cmd3d":
      parts.append(phase)
      if proprio_history is None:
          raise ValueError(
              "wr_obs_v8_cmd3d requires proprio_history; got None."
          )
      parts.append(np.asarray(proprio_history, dtype=np.float32).reshape(-1))
      if velocity_cmd_lateral_yaw is None:
          raise ValueError(
              "wr_obs_v8_cmd3d requires velocity_cmd_lateral_yaw=(vy, wz)."
          )
      parts.append(
          np.asarray(velocity_cmd_lateral_yaw, dtype=np.float32).reshape(2),
      )
  ```
- Propagate `velocity_cmd_lateral_yaw` through the wrapper
  `build_observation(...)` at lines 191-250.

**Verify**: both tests pass.

### P11.2 — `training/eval/eval_policy.py` 3D-cmd parity (failing test + impl)

**Failing test**: append to `training/tests/test_numpy_obs_v8_parity.py`:
```python
def test_eval_policy_sentinel_reads_vx_index_only() -> None:
    """H4 + H3: eval override sentinel reads cmd[0] only.  A YAML pinning
    eval_velocity_cmd: [-1.0, 0.5, 0.0] means "use sampled cmd" (vx=-1
    -> sentinel)."""
    from training.eval.eval_policy import _eval_cmd_is_overridden
    import types
    cfg = types.SimpleNamespace(eval_velocity_cmd=(-1.0, 0.5, 0.0))
    assert _eval_cmd_is_overridden(cfg) is False
    cfg = types.SimpleNamespace(eval_velocity_cmd=(0.26, 0.0, 0.0))
    assert _eval_cmd_is_overridden(cfg) is True
```

**Impl** in `training/eval/eval_policy.py`:
- Replace the existing `float(env_cfg.eval_velocity_cmd) >= 0.0` check at
  line 43 with a helper:
  ```python
  def _eval_cmd_is_overridden(env_cfg) -> bool:
      """H3 + H4: sentinel applies to vx (index 0) only."""
      cmd = env_cfg.eval_velocity_cmd
      try:
          vx0 = float(cmd[0])
      except (TypeError, IndexError):
          vx0 = float(cmd)  # legacy scalar path (should not happen post-P3.4a)
      return vx0 >= 0.0
  ```
- Update the eval banner at line 460 to print all three axes:
  ```python
  print(
      f"  eval_cmd:   pinned (vx={training_cfg.env.eval_velocity_cmd[0]:.3f}, "
      f"vy={training_cfg.env.eval_velocity_cmd[1]:.3f}, "
      f"wz={training_cfg.env.eval_velocity_cmd[2]:.3f})"
  )
  ```
- `metrics.get("tracking/forward_velocity_cmd_ratio")` at line 549 stays
  scalar (it's a per-axis metric); no edit.

### P11.3 — `training/eval/eval_ladder_v0170.py` mirror

**Impl**: grep for `eval_velocity_cmd` and `float(env_cfg.eval_velocity_cmd)`;
apply the same `_eval_cmd_is_overridden` swap. (No new test required —
the eval-policy test covers the shared helper; the ladder script merely
imports it.)

### P11.4 — `training/eval/v6_eval_adapter.py` 3-vec parameter

**Failing test**:
```python
def test_v6_eval_adapter_accepts_3vec_cmd() -> None:
    """H4: v6 eval adapter must forward a (3,) velocity_cmd to the numpy
    build_observation path (post-P11.1 it accepts the new kwarg)."""
    # Stub-style: just import + check the type annotation widened.
    import inspect
    from training.eval.v6_eval_adapter import V6EvalAdapter
    sig = inspect.signature(V6EvalAdapter.compute_obs)
    cmd_param = sig.parameters["velocity_cmd"]
    # Annotation should mention ndarray, not float (we accept either by
    # ducktype in code; the test guards against silent regressions).
    assert "ndarray" in str(cmd_param.annotation) or cmd_param.annotation is inspect.Parameter.empty
```

**Impl** in `training/eval/v6_eval_adapter.py:631`:
- Widen `velocity_cmd: float` -> `velocity_cmd: np.ndarray | float` in the
  `compute_obs(...)` signature (and the `__call__` shim at line 49).
- At line 691 where `velocity_cmd=np.array(velocity_cmd, dtype=np.float32)`
  is passed to `build_observation_from_components`, additionally pass
  `velocity_cmd_lateral_yaw` IF the policy spec's layout is v8:
  ```python
  cmd_arr = np.asarray(velocity_cmd, dtype=np.float32).reshape(-1)
  if cmd_arr.size == 1:
      # Legacy scalar input: broadcast vx-only (H3 semantics).
      cmd_arr = np.array([float(cmd_arr[0]), 0.0, 0.0], dtype=np.float32)
  kwargs = dict(velocity_cmd=cmd_arr, ...)
  if self._spec.observation.layout_id == "wr_obs_v8_cmd3d":
      kwargs["velocity_cmd_lateral_yaw"] = cmd_arr[1:]
  obs = build_observation_from_components(**kwargs)
  ```

### P11.5 — `training/eval/visualize_policy.py` 3-vec cmd

**Failing test**:
```python
def test_visualize_policy_validate_accepts_three_floats(tmp_path) -> None:
    """H4: --velocity-cmd can be either a scalar (broadcast to vx-only per
    H3) or three floats."""
    from training.eval.visualize_policy import _validate_user_fixed_velocity_cmd
    import types
    cfg = types.SimpleNamespace(
        env=types.SimpleNamespace(
            min_velocity=0.18, max_velocity=0.26,
            min_velocity_y=-0.13, max_velocity_y=0.13,
            max_yaw_rate=0.25,
        ),
    )
    out = _validate_user_fixed_velocity_cmd(cfg, 0.20)        # scalar -> vx
    assert tuple(out) == (0.20, 0.0, 0.0)
    out = _validate_user_fixed_velocity_cmd(cfg, [0.20, 0.05, 0.10])
    assert tuple(out) == (0.20, 0.05, 0.10)
```

**Impl** in `training/eval/visualize_policy.py`:
- Widen `_validate_user_fixed_velocity_cmd` to return
  `np.ndarray (3,)` instead of `float`.
- Update CLI flags at line 832 to optionally accept three floats
  (`nargs='+'` with `len in {1, 3}` post-parse validation).
- Update `get_observation(...)` at line 566 and the
  `build_observation(...)` callers at lines 580 and 592 to pass the (3,)
  cmd AND `velocity_cmd_lateral_yaw=cmd[1:]` when the spec is v8.
- `sample_velocity_cmd()` at line 649 widens its return to (3,) by
  defaulting vy=wz=0.0 (no other behavior change since visualize
  historically only exercises vx).
- The eval-cmd override fallback at lines 722-728 reads
  `cfg.env.eval_velocity_cmd[0]` instead of `float(...)`.

**Commit**:
```bash
git add policy_contract/numpy/obs.py training/eval/eval_policy.py \
        training/eval/eval_ladder_v0170.py training/eval/v6_eval_adapter.py \
        training/eval/visualize_policy.py training/eval/visualize_reference_library.py \
        training/tests/test_numpy_obs_v8_parity.py
git commit -m "v0.21.0 P11: eval / visual / deploy parity for 3-axis velocity_cmd (H4)"
```

---

## P8 — Smoke YAML + residual scales (H6 hip_yaw fix)

### P8.1 — `ppo_walking_v0210_smoke1_lateral_yaw.yaml`

**File**: `training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`

Clone `training/configs/ppo_walking_v0201_smoke14.yaml` and apply:
```yaml
env:
  actor_obs_layout_id: wr_obs_v8_cmd3d
  min_velocity: 0.18
  max_velocity: 0.26
  min_velocity_y: -0.13
  max_velocity_y:  0.13
  max_yaw_rate:    0.25
  cmd_zero_chance: 0.20
  cmd_turn_chance: 0.20
  cmd_deadzone: [0.02, 0.02, 0.05]
  # H3: explicit 3-tuple form (scalar would map to (0.18, 0.0, 0.0) under
  # the H3 broadcast rule; we want eval at the full (vx, vy, wz) point).
  eval_velocity_cmd: [0.18, 0.04, 0.10]
  loc_ref_version: v3_offline_library
  loc_ref_command_conditioned: true
  loc_ref_command_axes_3d: true
  # H2: keep the vx grid including 0.0.  smoke14's grid_interval=0.04
  # over [0.18, 0.26] gives [0.18, 0.22, 0.26]; smoke1 extends with an
  # explicit 0.0 sentinel handled by `_init_offline_service` (see P5.4).
  # The simplest way to express this is to override the vx grid directly:
  loc_ref_offline_command_vx_grid: [0.0, 0.18, 0.22, 0.26]
  loc_ref_offline_command_vy_grid: [-0.10, -0.05, 0.0, 0.05, 0.10]
  loc_ref_offline_command_yaw_rate_grid: [-0.20, -0.10, 0.0, 0.10, 0.20]
  # H6: WR has NO *_hip_yaw joints (verified at assets/v2/wildrobot.xml).
  # Only bump hip_roll and ankle_roll (the lateral-affecting DoFs).
  loc_ref_residual_scale_per_joint:
    left_hip_pitch:   0.3534    # unchanged from smoke14
    left_hip_roll:    0.375     # smoke14: 0.25 * 1.5 (H6)
    left_knee_pitch:  0.667     # unchanged
    left_ankle_pitch: 0.25      # unchanged
    left_ankle_roll:  0.375     # smoke14: 0.25 * 1.5 (H6)
    right_hip_pitch:  0.3867    # unchanged
    right_hip_roll:   0.375     # smoke14: 0.25 * 1.5 (H6)
    right_knee_pitch: 0.6657    # unchanged
    right_ankle_pitch:0.25      # unchanged
    right_ankle_roll: 0.375     # smoke14: 0.25 * 1.5 (H6)
reward_weights:
  cmd_velocity_track_dim: 2
  cmd_velocity_track: 2.0
  cmd_yaw_rate_track:  1.5     # H5: smoke1 turns the new yaw-rate term on
  cmd_yaw_rate_alpha:  0.25    # H5: TB-aligned (default; documented here)
```

> **NEW field `loc_ref_offline_command_vx_grid`**: this is the cleanest
> way to satisfy H2 without inventing a "static-bin override" knob. The
> `_init_offline_service` impl in P5.4 reads this list when provided and
> falls back to the derived vx_grid otherwise.

**Test**: `training/tests/test_config_load_v0210_smoke1.py`
```python
def test_smoke1_yaml_loads_and_has_3d_axes_enabled() -> None:
    from training.configs.training_config import load_training_config
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml"
    )
    assert cfg.env.actor_obs_layout_id == "wr_obs_v8_cmd3d"
    assert cfg.env.loc_ref_command_axes_3d is True
    assert tuple(cfg.env.cmd_deadzone) == (0.02, 0.02, 0.05)
    assert tuple(cfg.env.eval_velocity_cmd) == (0.18, 0.04, 0.10)
    assert cfg.env.max_yaw_rate == 0.25
    assert cfg.reward_weights.cmd_yaw_rate_track == 1.5
    assert cfg.reward_weights.cmd_yaw_rate_alpha == 0.25


def test_smoke1_yaml_vx_grid_includes_static_bin() -> None:
    """H2: explicit vx grid includes 0.0 so the (0, 0, 0) static bin
    exists in the library."""
    from training.configs.training_config import load_training_config
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml"
    )
    assert 0.0 in tuple(cfg.env.loc_ref_offline_command_vx_grid)
```

### P8.2 — Residual scales test (H6: no hip_yaw)

```python
def test_smoke1_has_bumped_lateral_residual_scales_no_hip_yaw() -> None:
    """H6: WR has no *_hip_yaw joints; only bump hip_roll and ankle_roll."""
    from training.configs.training_config import load_training_config
    import pytest
    cfg = load_training_config(
        "training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml"
    )
    scales = cfg.env.loc_ref_residual_scale_per_joint
    base = load_training_config(
        "training/configs/ppo_walking_v0201_smoke14.yaml"
    ).env.loc_ref_residual_scale_per_joint
    for j in (
        "left_hip_roll", "right_hip_roll",
        "left_ankle_roll", "right_ankle_roll",
    ):
        assert scales[j] == pytest.approx(base[j] * 1.5)
    # H6 explicit guard: no *_hip_yaw key (joint doesn't exist in WR).
    assert "left_hip_yaw" not in scales
    assert "right_hip_yaw" not in scales
```

**Commit**:
```bash
git add training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml \
        training/tests/test_config_load_v0210_smoke1.py
git commit -m "v0.21.0 P8: smoke1 YAML (v8 layout, 100-bin library w/ static, no hip_yaw — H2, H3, H6)"
```

---

## P9 — Integration tests + walking_training.md gates

### P9.1 — End-to-end env rollout test (unchanged from -final.md)

### P9.2 — v7 layout regression (unchanged from -final.md)

### P9.3 — `training/docs/walking_training.md` G6 / G7 gates (unchanged)

**Commit**:
```bash
git add training/tests/test_env_step_lateral_yaw_e2e.py training/docs/walking_training.md
git commit -m "v0.21.0 P9: e2e lateral/yaw rollout tests + G6/G7 gate definitions"
```

---

## P10 — CHANGELOG entry

### P10.1 — `training/CHANGELOG.md`

(Body unchanged from `-final.md`. Add bullets for H1-H7 / M1-M3 fixes to the
"Architecture deltas" section as appropriate.)

---

## End-to-end verification

Same as `-final.md`. Additionally:
- `cd <wildrobot-repo> && uv run pytest training/tests/test_numpy_obs_v8_parity.py -q` — P11 numpy mirror tests pass.
- `cd <wildrobot-repo> && uv run python training/eval/visualize_policy.py --config training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml --velocity-cmd 0.18 0.05 0.10 --max-steps 100` — runs without crash, video produced.

---

## Files touched (summary)

(Identical to `-final.md` plus the new files / sections below; key
deltas annotated.)

**Create**:
- `training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`
- `training/docs/v0210_lateral_yaw_prior_plan-final-r2.md` (this file)
- `training/tests/test_zmp_walk_config_3d_cmd.py`
- `training/tests/test_reference_library_3d.py`
- `training/tests/test_library_build_size.py` (NEW — M3)
- `training/tests/test_velocity_cmd_3d_migration.py`
- `training/tests/test_env_config_3d_cmd.py`
- `training/tests/test_sample_velocity_cmd_3d.py`
- `training/tests/test_loc_ref_3d_config.py`
- `training/tests/test_cmd_lateral_velocity_tracking.py`
- `training/tests/test_obs_v8_cmd3d.py`
- `training/tests/test_numpy_obs_v8_parity.py` (NEW — H4 / P11)
- `training/tests/test_env_step_lateral_yaw_e2e.py`
- `training/tests/test_config_load_v0210_smoke1.py`

**Delete from `-final.md`**:
- `training/docs/v0210_velocity_cmd_audit_checklist.md` (auto-gen scratch — M2)
- `training/tests/test_velocity_cmd_callsite_audit.py` (brittle — M2)

**Modify** (changes vs. `-final.md` annotated):
- `control/zmp/zmp_walk.py` — `ZMPWalkConfig.rotation_radius_m`,
  `generate()` widened sig, **3-axis standing gate (H1)**, per-axis
  sign flips, lateral footstep block, pure-rotation arc-radius block,
  `build_library_for_3d_values`.
- `control/references/reference_library.py` — `vy_grid` / `yaw_rate_grid`
  fields; save/load round-trip. **Static-bin invariant documented (H2).**
- `control/references/runtime_reference_service.py` —
  `compute_command_integrated_path_state` 2D form; **NEW
  `incremental_path_state_step` helper (H7)**; `_yaw_from_quat_wxyz`.
- `training/envs/wildrobot_env.py` — all of `-final.md`'s edits, PLUS:
  - `_init_offline_service` now reads `loc_ref_offline_command_vx_grid`
    when provided AND enforces the static-bin invariant (H2).
  - `_make_initial_state` initializes the two new path-state fields
    (H7).
  - `step` calls `incremental_path_state_step` instead of the closed
    form (H7) and persists the result into the new WildRobotInfo fields.
  - Eval sentinel reads `eval_cmd[0] >= 0.0` instead of
    `jp.all(eval_cmd >= 0.0)` (H3).
- `training/envs/env_info.py` — `velocity_cmd` shape `() -> (3,)` AND
  **NEW `path_state_torso_pos: (3,)` + `path_state_path_rot: (4,)`
  fields (H7)** in both backends and `get_expected_shapes`.
- `training/configs/training_runtime_config.py` —
  +`min/max_velocity_y`, `max_yaw_rate`, `loc_ref_command_axes_3d`,
  `loc_ref_offline_command_vy_grid`, `loc_ref_offline_command_yaw_rate_grid`,
  **+`loc_ref_offline_command_vx_grid` (H2)**, **+`cmd_yaw_rate_alpha`
  (H5)**, `cmd_yaw_rate_track`; type bumps for `cmd_deadzone`
  (-> `(0.0, 0.0, 0.0)`) and `eval_velocity_cmd`
  (-> `(-1.0, 0.0, 0.0)` per H3).
- `training/configs/training_config.py` — `_load_three_axis` helper with
  the `scalar_broadcast_axis` mode (H3); REPLACE
  `float(env.get(...))` at lines 549-550; read new fields including
  `loc_ref_offline_command_vx_grid`, `cmd_yaw_rate_alpha`.
- `policy_contract/spec.py` — `SUPPORTED_LAYOUT_IDS += {"wr_obs_v8_cmd3d"}`;
  `_validate_observation` v8 branch.
- `policy_contract/spec_builder.py` — `_build_obs_layout` v8 branch.
- `policy_contract/jax/obs.py` — dispatcher accepts v8; new
  `velocity_cmd_lateral_yaw` kwarg; v8 branch appends 2-vec; line 151
  sliced to `[..., :1]`.
- **`policy_contract/numpy/obs.py` — same as jax/obs.py changes (H4 / P11.1).**
- **`training/eval/eval_policy.py` — sentinel reads cmd[0]; banner prints
  3 axes (H4 / P11.2).**
- **`training/eval/eval_ladder_v0170.py` — sentinel mirror (H4 / P11.3).**
- **`training/eval/v6_eval_adapter.py` — accepts (3,) cmd; passes
  `velocity_cmd_lateral_yaw` to numpy obs for v8 (H4 / P11.4).**
- **`training/eval/visualize_policy.py` — `_validate_user_fixed_velocity_cmd`
  returns (3,); CLI accepts 1 or 3 floats; passes
  `velocity_cmd_lateral_yaw` for v8 (H4 / P11.5).**
- `training/core/metrics_registry.py` — per-axis tracking metrics +
  `tracking/yaw_rate_err`.
- `training/docs/walking_training.md` — G6 / G7 sections.
- `training/CHANGELOG.md` — v0.21.0 entry (after smoke).

---

## Risks & rollback

(Same as `-final.md`, plus:)

- **Risk (H6 follow-on)**: WR has no hip_yaw joints, so yaw command
  tracking relies entirely on hip_roll asymmetry + foot-arc geometry from
  the new pure-rotation branch. If smoke1's G7 saturates well below the
  0.5 signed-ratio target, the v0.22.0 followup is to either (a) add a
  software-yaw approximation via differential hip_roll, or (b) widen
  `rotation_radius_m` to push more of the yaw rate into the foot arc.
  Documented as an open risk.
- **Risk (H7 incremental form)**: bug in `incremental_path_state_step`
  produces a drifting torso target the policy chases incorrectly.
  **Mitigation**: P3.9's
  `test_incremental_path_state_matches_closed_form_for_constant_cmd`
  pins parity with the analytic form over 50 steps; the e2e test
  `test_path_state_continuity_under_cmd_resample` covers the resample
  edge case. If either fails post-merge, revert to the closed form and
  set `cmd_resample_steps: 0` in smoke1 (acceptable for the first
  smoke; loses the resampling pressure but keeps path-state correct).
- **Risk (M3 library size)**: the 100-bin library on a single env may
  exceed 200 MB once stacked over a vectorized batch. **Mitigation**:
  P2.5a's measurement gates this; if either budget is exceeded, drop
  the wz grid to a single 0.0 entry (gives 4 * 5 * 1 = 20 bins).
- **Rollback**: revert by switching smoke YAML back to
  `actor_obs_layout_id: wr_obs_v7_phase_proprio` +
  `loc_ref_command_axes_3d: false`. All v0.21 code paths default to
  vy=wz=0 when the toggle is false and the grids are empty tuples.
