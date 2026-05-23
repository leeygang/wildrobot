## Changes from previous version

This v2 plan re-bases on the actual WR codebase (lookup-API, layout dispatcher, `WildRobotInfo.velocity_cmd` shape, `loc_ref_version` validator, `eval_velocity_cmd`/`cmd_deadzone` types) and on a careful re-read of TB's `zmp_walk.py`/`walk_env.py`/`walk_zmp_ref.py`. Every Critical Issue C1–C17 from the review is addressed; suggestions S1, S2, S5, S6, S8 are folded in.

- **C1 — `command_vx` is a `generate()` parameter, not a config field.**
  - Removed the original P1.1/P1.2 (which incorrectly added `command_vy`/`command_yaw_rate` as `ZMPWalkConfig` fields).
  - Added a new `P0.5` "scaffolding audit" task (also addresses S1).
  - Rewrote P1 as P1.A (generator signature widening: `generate(command_vx, command_vy=0.0, command_yaw_rate=0.0, rotation_radius=None)`) and added a `ZMPWalkConfig.rotation_radius_m: float = 0.10` config knob for the WR-tuned arc radius (also addresses C12).
- **C2/C3 — No `get_obs_layout(...)` / `slot_size(...)` / `slice_obs(...)` API exists.**
  - Replaced every test stub that called those imagined APIs with tests that use the real surface: `policy_contract.spec.PolicySpec` + `validate_spec()`, `policy_contract.spec_builder.build_policy_spec(..., layout_id="wr_obs_v8_cmd3d")`, and `build_observation_from_components(spec=..., velocity_cmd=jp.zeros(3), ...)` returning `obs.shape == (model.obs_dim,)`.
  - The obs.py edit no longer touches the shared `parts` slot; instead it appends a NEW 3-vector slot AT THE END of the v8 branch only, and the shared 1-vector `velocity_cmd` slot is left untouched (so spec validation at `spec.py:339,357,376,395,429,455` keeps passing for v1–v7).
- **C4 — Real config field is `actor_obs_layout_id`, NOT `obs_version`.**
  - Renamed every YAML key and every doc/test reference from `obs_version`/`env.obs_version` to `actor_obs_layout_id`/`env.actor_obs_layout_id`.
  - Added an explicit allow-list bump in `wildrobot_env.py:225-239` so v8 passes the init guard.
- **C5 — `loc_ref_version` is hard-validated.**
  - Picked option (b): KEEP `loc_ref_version: v3_offline_library` and key 3D behavior off a NEW separate boolean `loc_ref_command_axes_3d: bool = False`. This avoids touching the validator and keeps any external consumers of the version string unaffected. New P-step (P3.5) writes this knob.
- **C6 — `loc_ref_offline_command_vx_grid` does not exist.**
  - Picked option "explicit grid fields": added `EnvConfig.loc_ref_offline_command_vy_grid: tuple[float, ...] = ()` and `loc_ref_offline_command_yaw_rate_grid: tuple[float, ...] = ()` (NEW config fields). vx grid still derived from `min_velocity`/`max_velocity`/`loc_ref_command_grid_interval` as today. New P-step P3.7 verifies the empty-tuple default reproduces today's behavior.
- **C7 — `ReferenceLibrary.lookup` is already 3D today.**
  - Deleted the P2.1/P2.2/P2.3 tasks that re-implemented 3D lookup. Kept a verification test (`test_reference_library_lookup_already_picks_nearest_3d_key`) as a regression guard.
  - The actual P2 work is now: (P2.A) widen `ReferenceLibraryMeta` with `vy_grid`/`yaw_rate_grid` (sequence fields the env reads back when building the stacked JAX arrays); (P2.B) extend `ZMPWalkGenerator.build_library_for_*` with a new `build_library_for_3d_values(vx_values, vy_values, yaw_rate_values)` method; (P2.C) extend `RuntimeReferenceService` with an `_offline_cmd_keys: (n_bins, 3)` array and a new `lookup_jax_3d` that accepts a 3-vec cmd; (P2.D) extend `wildrobot_env._lookup_offline_window` and `_init_offline_service` to thread the 3-vec cmd and the cmd-keys.
- **C8 — Sampler must mirror TB exactly.**
  - Rewrote P4 (was P3) to mirror TB `walk_env.py:256-280` literally: per-axis magnitude sampling (`x = U[deadzone[0], x_max] * sin(theta)`, `y = U[deadzone[1], y_max] * cos(theta)`), asymmetric per-axis limits via `command_range[5][0]` vs `[5][1]` shape, branch by `random_number < zero_chance` / `< zero_chance + turn_chance`. Each construct gets its own sub-step.
  - Threads the 3-vector `cmd_deadzone` into the sampler explicitly (C14).
  - Updated the empirical-distribution test (was P7.2) to test TB's actual marginal (per-axis uniform between deadzone and limit, scaled by sin/cos of shared theta), NOT a uniform-radial-r assumption.
- **C9 — `WildRobotInfo.velocity_cmd` scalar→3-vec migration is now an explicit phase.**
  - New phase **P3 ("Migrate `velocity_cmd` to a 3-vector")** added BEFORE the sampler (P4) and reward (P6) work. Lists every call site by line number (1218 hard-coded `vy_cmd=0`; 1837/2051 scalar `jp.abs(velocity_cmd)`; 2531 sentinel comparator; 2868-2871 `linear[0] - velocity_cmd`; 2896-2898 metric `tracking/velocity_cmd_abs`; 2905-2907 `forward_velocity_cmd_ratio`; 2962-2965 `compute_command_integrated_path_state` call; 3340/3359/3362/3383/3408 step-time mirrors) and gives each a dedicated sub-step with its own failing test. Also bumps `env_info.WildRobotInfo.velocity_cmd` to shape `(3,)` and updates `get_expected_shapes()` at line 285.
- **C10 — `compute_command_integrated_path_state` ignores `vy` and `wz`.**
  - Added explicit sub-step P3.6 to extend `RuntimeReferenceService.compute_command_integrated_path_state` (and its env caller at 2962-2965) to take a 3-vec velocity_cmd and a real yaw_rate_cmd, mirror TB `motion_ref.py::integrate_path_state` so `path_pos` integrates `vy` and `yaw` integrates `wz`. Includes a failing test that asserts `path_pos[1] > 0` for cmd `(0.0, 0.10, 0.0)` after 1 s.
- **C11 — Heading-frame correction missing from 3D lookup.**
  - New sub-step P5.4 (now in P5 "3D reference library plumbing"): before calling `lookup_jax_3d` with the (vx, vy, wz) cmd, rotate the (vx, vy) command into the heading-error frame using `(heading_rot.inv() * path_state["path_rot"]).apply([vx, vy, 0])`, mirroring `walk_zmp_ref.py:132-137`. Verified by a unit test that asserts the bin choice changes when torso yaw drifts by 0.3 rad.
- **C12 — TB's pure-rotation branch is arc-around-off-pelvis, NOT pelvis-centered.**
  - Rewrote the pure-rotation impl step (was P1.5) to mirror `toddlerbot/algorithms/zmp_walk.py:269-305` exactly: `r = sign(wz) * rotation_radius`, stride `= wz * total_time / (2 * num_cycles - 1)` is angular delta PER FOOTSTEP (not per cycle), and footstep x/y formulas use the documented arc-plus-foot_to_com_y geometry.
  - Added `ZMPWalkConfig.rotation_radius_m: float = 0.10` with a derivation note (WR foot_to_com_y proxy = `hip_lateral_offset_m = 0.0536`; TB's foot_to_com_y = 0.042 and `rotation_radius = 0.1`, so WR-scaled default = `0.10 * 0.0536 / 0.042 ≈ 0.128`; rounded to **0.10** as a conservative first smoke value with a TODO to revisit).
- **C13 — `eval_velocity_cmd` sentinel migration.**
  - Replaced the original "list-typed `eval_velocity_cmd: [...]`" with:
    - `EnvConfig.eval_velocity_cmd: tuple[float, float, float] = (-1.0, -1.0, -1.0)` (new type).
    - Sentinel check at `wildrobot_env.py:2531` becomes `jp.all(eval_cmd >= 0.0)` reducing over the 3-axis vector.
    - Added a YAML compatibility shim: when a YAML sets a scalar (older smoke configs), the loader expands it to `(scalar, 0.0, 0.0)` in `training_config.py`. New sub-step P3.4.
- **C14 — `cmd_deadzone` type migration.**
  - Replaced scalar `EnvConfig.cmd_deadzone: float = 0.0` with `tuple[float, float, float] = (0.0, 0.0, 0.0)`.
  - Added YAML compatibility shim (scalar → broadcast to all 3 axes) in `training_config.py`. New sub-step P3.3.
  - Sampler reads `deadzone[0]`/`deadzone[1]`/`deadzone[2]` per axis (matches TB).
- **C15 — Step granularity.**
  - Every step in P1 (planner), P2 (lookup), P3 (info migration), P4 (sampler), P5 (3D library), P6 (reward) is now ≤5 minutes and ships with its own test → impl → verify → commit cycle. The old monolithic P1.4 / P1.5 / P2.4 / P3.4 / P5.3 are each split into 4–6 sub-steps.
- **C16 — `test_cmd_velocity_tracking_mode.py` will break on rename.**
  - Chose option **(a) DON'T RENAME** the static helper. Kept `_cmd_forward_velocity_track_reward` as the name; added an optional `vy_cmd: jax.Array = jp.float32(0.0)` parameter with default that reproduces the legacy behavior. Updates the 3 existing tests at lines 37/63/96 ONLY by adding one new test `test_2d_cmd_tracking_uses_supplied_vy_cmd` that exercises the new kwarg. No test signature changes.
  - P6.2 explicitly lists `test_cmd_velocity_tracking_mode.py` as a regression test that MUST still pass with no edits.
- **C17 — G3 parallelization claim corrected.**
  - Updated the dependency table: P3 (info migration), P4 (sampler), P6 (reward) all touch `wildrobot_env.py` — they are SERIAL within the file. Only the layout-registry / spec-builder edits in P7.1–P7.2 are truly parallelizable with anything else.
- **S1 — Scaffolding audit added.** New P0.5 step lists every `velocity_cmd` callsite as a checklist before any code change.
- **S2 — Path-state guard.** P5.4 explicitly addresses path-state integration (C10 also); no defer.
- **S5 — `cmd_turn_chance` historical context.** Called out in P10.1 (CHANGELOG note).
- **S6 — G6/G7 thresholds softened.** Promotion criterion now uses the v0.20.x-style `signed cmd_ratio >= 0.5` instead of the tighter absolute-error bound.
- **S8 — Default-zero weight is a backward-compat shim, not a recommendation.** Documented in CHANGELOG section.

The plan grew from 9 phases (P1–P9) to 11 phases (P0.5, P1–P10) and ~250 lines → ~640 lines (more sub-steps, explicit signatures, per-call-site migration coverage).

---

# Plan: v0.21.0 — Lateral (vy) + Yaw (wz) Prior Support for WildRobot (v2 revised)

**Goal**: Extend WR's offline ZMP prior and policy I/O from forward-only (vx) to full 3-DoF locomotion commands (vx, vy, wz), mirroring ToddlerBot's unified ZMP design, so the policy can learn to strafe and turn alongside walking forward.

**Architecture**:
- **Single unified ZMP prior**, not a separate "lateral module". Widen `ZMPWalkGenerator.generate()` (NOT `ZMPWalkConfig` — `command_vx` is and remains a `generate()` parameter; see C1) to accept `command_vy` / `command_yaw_rate`, and add a `ZMPWalkConfig.rotation_radius_m` knob. Mirror TB's `toddlerbot/algorithms/zmp_walk.py:259` translation branch for `(vx, vy)` and TB's `:269-305` arc-radius branch for `wz`.
- **`ReferenceLibrary.lookup` is already 3D** today (`reference_library.py:395-409` accepts `(vx, vy=0.0, yaw_rate=0.0)`) — only the env's `_lookup_offline_window`, the JAX bin-selection, and the metadata grids need work.
- **Command obs widened from 1D → 3D via a NEW slot appended at the end of v8** (`velocity_cmd_lateral_yaw` of size 2 = `[vy_cmd, wz_cmd]`). The legacy 1D `velocity_cmd` slot stays in place for v8 (carries `vx_cmd`), so spec-validator branches for v1–v7 don't move. New layout `wr_obs_v8_cmd3d` registered in `SUPPORTED_LAYOUT_IDS` (`spec.py:11`), `_build_obs_layout` (`spec_builder.py`), `_validate_observation` (`spec.py`), and `build_observation_from_components` dispatcher (`obs.py:46-51`).
- **`WildRobotInfo.velocity_cmd` is migrated from shape `()` to `(3,)`** (`env_info.py:107, 222, 285`). All ~10 callsites that index it scalar-style get an explicit migration sub-step (C9).
- **Reward**: KEEP `_cmd_forward_velocity_track_reward` name and shape; add an optional `vy_cmd: jax.Array = jp.float32(0.0)` parameter (default reproduces legacy). Add a separate `_yaw_rate_track_reward` static helper. Wire both from `_compute_reward_terms` with new `RewardWeightsConfig.cmd_yaw_rate_track: float = 0.0` (back-compat).
- **WR-scaled command ranges (LOCKED)**: vy ∈ ±0.13 m/s, wz ∈ ±0.25 rad/s.

**Tech Stack**: Python 3.12, JAX, Brax, MuJoCo MJX, PPO. Tests run via `uv run pytest training/tests/test_xxx.py -q` from repo root `/Users/ygli/projects/wildrobot`.

**Cross-robot scaling reference**:
| Quantity | TB | WR | Ratio |
|---|---|---|---|
| Leg length | 0.2115 m | 0.373 m | 1.76× |
| Hip width / foot_to_com_y | 0.042 m | 0.0536 m | 1.28× |
| cycle_time | 0.72 s | 0.96 s | 1.33× |
| vx_max (training) | 0.10 m/s | 0.26 m/s | 2.6× |
| **vy_max (this plan)** | 0.05 m/s | **0.13 m/s** | 2.6× (preserve vy/vx ratio) |
| **wz_max (this plan)** | 1.0 rad/s | **0.25 rad/s** | 0.25× (preserve foot-arc-per-swing) |
| rotation_radius (this plan) | 0.10 m | **0.10 m** | 1.0× (conservative first value; see C12 note) |

---

## Task Dependencies

| Group | Steps | Parallelizable | Notes |
|-------|-------|----------------|-------|
| G0 | P0.5 (audit) | n/a | Pure read; no commit |
| G1 | P1.1–P1.13 (planner) | No (single file) | Touches `control/zmp/zmp_walk.py` + `control/references/reference_library.py` only |
| G2 | P2.1–P2.5 (lookup verify + extend) | After G1 | Touches `control/references/*.py` only |
| G3 | P3.1–P3.10 (`velocity_cmd` scalar → 3-vec migration) | No (single env file) | All in `wildrobot_env.py` + `env_info.py` + `training_runtime_config.py` |
| G4 | P4.1–P4.7 (sampler) | After G3 (same file) | `wildrobot_env.py::_sample_velocity_cmd`; cannot parallelize with G5 (same file) |
| G5 | P5.1–P5.5 (3D library plumbing) | After G2+G3 (same file) | `wildrobot_env.py::_init_offline_service`, `_lookup_offline_window`; cannot parallelize with G4 (same file) |
| G6 | P6.1–P6.4 (reward) | After G3+G5 (same file) | `wildrobot_env.py::_cmd_forward_velocity_track_reward` + `_compute_reward_terms` |
| G7 | P7.1–P7.4 (layout + spec) | After G3 | `policy_contract/spec.py` + `spec_builder.py` + `policy_contract/jax/obs.py` — independent of G4–G6 file scope; can run in parallel with G4 once G3 is done IF a separate branch is used |
| G8 | P8.1–P8.2 (smoke YAML + residual scales) | After G7 | Independent YAML |
| G9 | P9.1–P9.3 (integration + e2e tests) | After G6+G7+G8 | Cross-layer; tests in `training/tests/` |
| G10 | P10.1 (CHANGELOG) | After smoke run | Final |

**Conflict callout (C17)**: G3, G4, G5, G6 all touch `wildrobot_env.py`. They must be done in sequence on the same branch. G7 touches different files (`policy_contract/`) and CAN run in parallel only if the developer is comfortable rebasing G3's env changes onto G7's spec changes at the end.

---

## P0.5 — Scaffolding audit (no commit)

Before writing any code, enumerate every `velocity_cmd` callsite in the env / contract / tests to a checklist. This catches C9 / C13 / C14 surprises before they cause TDD pain.

**Command**:
```bash
cd /Users/ygli/projects/wildrobot && grep -nH 'velocity_cmd' \
  training/envs/wildrobot_env.py training/envs/env_info.py \
  training/configs/training_runtime_config.py \
  policy_contract/jax/obs.py policy_contract/spec.py policy_contract/spec_builder.py \
  control/references/runtime_reference_service.py \
  training/tests/test_*.py tests/*.py
```

**Expected checklist** (must be confirmed before P1 begins):
- `env_info.py:107` declared shape `# ()` — MUST become `# (3,)`.
- `env_info.py:222` NamedTuple fallback — MUST mirror.
- `env_info.py:285` `get_expected_shapes()` mapping — MUST become `(3,)`.
- `wildrobot_env.py:1218` hard-coded `vy_cmd = 0` — REMOVE.
- `wildrobot_env.py:1837` `(jp.abs(velocity_cmd) > 1e-6)` — MUST use `jp.linalg.norm(velocity_cmd) > 1e-6` (or per-axis check).
- `wildrobot_env.py:2051` `is_standing = jp.abs(velocity_cmd) < 1e-6` — same fix.
- `wildrobot_env.py:2315` `velocity_cmd = self._sample_velocity_cmd(key_vel)` — call returns shape `(3,)` after P4.
- `wildrobot_env.py:2529-2532` eval reset — sentinel scaling (C13).
- `wildrobot_env.py:2868-2873` `linear[0] - velocity_cmd` — index `velocity_cmd[0]`.
- `wildrobot_env.py:2896-2898` `tracking/velocity_cmd_abs` metric — use `jp.linalg.norm(velocity_cmd)` or split per axis.
- `wildrobot_env.py:2905-2907` `tracking/forward_velocity_cmd_ratio` — index `velocity_cmd[0]`.
- `wildrobot_env.py:2962-2965` `compute_command_integrated_path_state(velocity_cmd_mps=velocity_cmd, yaw_rate_cmd_rps=0.0, ...)` — pass `velocity_cmd[0]` as `vx`, `velocity_cmd[1]` as `vy`, and `velocity_cmd[2]` as `yaw_rate` (C10).
- `wildrobot_env.py:3217-3219` resample path — `_sample_velocity_cmd` returns `(3,)`.
- `wildrobot_env.py:3340/3359/3362/3383/3408` step-time metric mirrors — same fixes as reset path.
- `tests/test_v0201_env_zero_action.py:221` — passes `velocity_cmd_mps=` scalar; widen.
- `tests/test_runtime_reference_service.py:188,232,269` — pass `velocity_cmd_mps=` scalar; widen.
- `training/tests/test_cmd_velocity_tracking_mode.py:37,63,96` — static helper call with no vy_cmd (defaults to 0.0 under new signature — these MUST still pass; C16).

If the grep produces extra callsites, add them to the P3 migration sub-steps before continuing.

---

## P1 — Extend ZMP planner with vy + wz axes

**File**: `/Users/ygli/projects/wildrobot/control/zmp/zmp_walk.py`

### P1.1 — Add `rotation_radius_m` to `ZMPWalkConfig` (failing test)

**Failing test**: `/Users/ygli/projects/wildrobot/training/tests/test_zmp_walk_config_3d_cmd.py`
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

**Run**: `cd /Users/ygli/projects/wildrobot && uv run pytest training/tests/test_zmp_walk_config_3d_cmd.py -q` — expect 2 failures (`unexpected keyword argument 'rotation_radius_m'`).

### P1.2 — Impl: add `rotation_radius_m` field

Add `rotation_radius_m: float = 0.10` to `ZMPWalkConfig` (near `default_stance_width_m`, line 135) with a derivation comment:
```python
# Pure-rotation arc radius (m), mirrors TB
# toddlerbot/algorithms/zmp_walk.py:220 default 0.1.  WR-scaled value
# under foot_to_com_y proxy = hip_lateral_offset_m = 0.0536:
#   r_WR = 0.1 × (0.0536 / 0.042) ≈ 0.128
# Rounded to 0.10 as a conservative first-smoke value; revisit after
# G7 yaw tracking results.
rotation_radius_m: float = 0.10
```

**Verify**: rerun P1.1 — passes.

### P1.3 — `generate()` accepts `command_vy`, `command_yaw_rate` (failing test)

**Append to test file**:
```python
from control.zmp.zmp_walk import ZMPWalkGenerator


def test_generate_accepts_vy_and_yaw_rate_kwargs() -> None:
    gen = ZMPWalkGenerator()
    # Should NOT raise.
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

**Run**: expect failures.

### P1.4 — Impl: widen `generate()` signature (no behavior change yet)

In `zmp_walk.py:579` change:
```python
def generate(self, command_vx: float) -> ReferenceTrajectory:
```
to:
```python
def generate(
    self,
    command_vx: float,
    command_vy: float = 0.0,
    command_yaw_rate: float = 0.0,
) -> ReferenceTrajectory:
```
Pass `command_vy` / `command_yaw_rate` through to the two `ReferenceTrajectory(command_vx=...)` constructions at lines 1043, 1099 — set `command_vy=command_vy`, `command_yaw_rate=command_yaw_rate`. Behavior unchanged because no downstream code reads the new fields yet.

**Verify**: rerun P1.3 — both pass.

### P1.5 — Lateral stride: failing test

**Append to test file**:
```python
def test_planner_produces_nonzero_lateral_foot_displacement_for_vy() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0001, command_vy=0.05)  # walking branch, mostly lateral
    foot_y = traj.left_foot_pos[:, 1]
    # Foot must drift in +y direction over the cycle (≥ 5 mm).
    assert (foot_y.max() - foot_y.min()) > 0.005
```

**Run**: expect failure.

### P1.6 — Impl: combined-translation footstep block (mirror TB:259-267)

Refactor `_generate_at_step_length` (line 682) so the footstep-x advance term currently using `i * sl` for left and `(2 * i + 1) * sl` for right ALSO carries a lateral `(±vy * stride)` term mirroring TB:
```python
# TB zmp_walk.py:259-267 — combined translation.
# stride = command[:2] * total_time / (2 * num_cycles - 1)
stride_xy = np.array([command_vx, command_vy], dtype=np.float64) \
            * cfg.total_plan_time_s / (2 * n_cycles - 1)
```
Replace the per-step `x = i * sl` / `cycle_offset` math with `left_xy = left_xy_init + 2 * i * stride_xy` and `right_xy = right_xy_init + (2 * i + 1) * stride_xy`. Apply lateral component to `left_world[idx, 1]`/`right_world[idx, 1]` (in addition to the existing `±lat` stance offset). Compute the COM y-target the LQR sees from the new foot midpoint (so the ZMP transitions also shift laterally).

**Verify**: rerun P1.5 — passes. Re-run P1.3 — still passes.

### P1.7 — Pure-rotation branch: failing test

**Append**:
```python
def test_planner_yaws_pelvis_for_nonzero_yaw_rate() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    # Pelvis_rpy z accumulates over the cycle.
    yaw = traj.pelvis_rpy[:, 2]
    assert abs(yaw[-1] - yaw[0]) > 1e-3
```

**Run**: expect failure.

### P1.8 — Impl: pure-rotation arc-radius branch (mirror TB:269-305)

In `_generate_at_step_length`, branch on `(|command_vx| < eps and |command_vy| < eps and |command_yaw_rate| > eps)`. Mirror TB exactly (do NOT use pelvis-centered rotation — see C12):
```python
stride = command_yaw_rate * cfg.total_plan_time_s / (2 * n_cycles - 1)  # per-FOOTSTEP angular delta
r = np.sign(command_yaw_rate) * cfg.rotation_radius_m
foot_to_com_y = cfg.hip_lateral_offset_m  # WR proxy for TB's foot_to_com_y
for i in range(n_cycles):
    angle_left = i * 2 * stride
    angle_right = (i * 2 + 1) * stride
    left_xy = np.array([
        r * np.sin(angle_left) - foot_to_com_y * np.sin(angle_left),
        r * (1.0 - np.cos(angle_left)) + foot_to_com_y * np.cos(angle_left),
    ])
    right_xy = np.array([
        r * np.sin(angle_right) + foot_to_com_y * np.sin(angle_right),
        r * (1.0 - np.cos(angle_right)) - foot_to_com_y * np.cos(angle_right),
    ])
    # Use as left_world[:, :2] / right_world[:, :2] for that cycle.
```
Accumulate `pelvis_rpy[:, 2]` linearly across the cycle so the policy sees `yaw_rate * t`. The ZMP planner consumes the new footstep positions as the SS targets — no special handling needed beyond the new footstep table.

**Verify**: P1.7 passes. Re-run P1.5, P1.3 — pass.

### P1.9 — Trajectory carries 3D cmd: failing test

**Append**:
```python
def test_generated_trajectory_carries_all_three_cmd_axes() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.20, command_vy=0.04, command_yaw_rate=0.0)
    assert traj.command_vx == pytest.approx(0.20)
    assert traj.command_vy == pytest.approx(0.04)
    assert traj.command_yaw_rate == pytest.approx(0.0)
```

(Already exercised by P1.3 and P1.4 but pin the combined assertion.)

**Verify**: passes.

### P1.10 — `build_library_for_3d_values` factory: failing test

**Append**:
```python
def test_build_library_for_3d_values_yields_one_entry_per_grid_point() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    # 1 * 3 * 3 = 9 entries
    assert len(lib) == 9
    # Pure-rotation entry exists.
    assert (0.20, 0.0, 0.10) in lib
```

**Run**: expect failure (AttributeError).

### P1.11 — Impl: `build_library_for_3d_values`

Add to `ZMPWalkGenerator`:
```python
def build_library_for_3d_values(
    self,
    vx_values: Sequence[float],
    vy_values: Sequence[float] = (0.0,),
    yaw_rate_values: Sequence[float] = (0.0,),
) -> ReferenceLibrary:
    trajectories = []
    for vx in vx_values:
        for vy in vy_values:
            for wz in yaw_rate_values:
                trajectories.append(self.generate(vx, vy, wz))
    meta = ReferenceLibraryMeta(
        generator="zmp_walk",
        generator_version="0.21.0",
        robot="wildrobot_v2",
        dt=self.cfg.dt_s,
        cycle_time=self.cfg.cycle_time_s,
        n_joints=self._load_actuator_layout()["n_joints"],
        command_range_vx=(min(vx_values), max(vx_values)),
        command_range_vy=(min(vy_values), max(vy_values)),
        command_range_yaw=(min(yaw_rate_values), max(yaw_rate_values)),
        command_interval=0.0,
    )
    return ReferenceLibrary(trajectories=trajectories, meta=meta)
```

**Verify**: P1.10 passes.

### P1.12 — Regression: forward-only library still builds

**Append**:
```python
def test_build_library_for_3d_values_with_only_vx_matches_legacy_shape() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(vx_values=[0.20])
    assert len(lib) == 1
    traj = next(iter(lib._entries.values()))
    assert traj.command_vy == 0.0 and traj.command_yaw_rate == 0.0
```

**Verify**: passes.

### P1.13 — Commit
```bash
git add control/zmp/zmp_walk.py training/tests/test_zmp_walk_config_3d_cmd.py
git commit -m "v0.21.0 P1: ZMP planner accepts (vy, wz) and 3D library factory"
```

---

## P2 — Verify existing 3D `ReferenceLibrary.lookup` + add metadata grids

**File**: `/Users/ygli/projects/wildrobot/control/references/reference_library.py`

### P2.1 — Regression: `lookup` already accepts 3D key (passing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_reference_library_3d.py`
```python
from __future__ import annotations
import pytest
import numpy as np
from control.zmp.zmp_walk import ZMPWalkGenerator


def test_reference_library_lookup_already_picks_nearest_3d_key() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    picked = lib.lookup(vx=0.20, vy=0.04, yaw_rate=-0.08)
    # Nearest L2 over (vx, vy, yaw): (0.20, 0.05, -0.10).
    assert picked.command_vy == pytest.approx(0.05)
    assert picked.command_yaw_rate == pytest.approx(-0.10)
```

**Run**: should already PASS today (proves C7's claim from the review).

### P2.2 — `ReferenceLibraryMeta` accepts `vy_grid` / `yaw_rate_grid` tuples (failing test)

**Append**:
```python
from control.references.reference_library import ReferenceLibraryMeta


def test_meta_accepts_vy_and_yaw_rate_grids() -> None:
    meta = ReferenceLibraryMeta(
        command_range_vx=(0.20, 0.20),
        command_range_vy=(-0.05, 0.05),
        command_range_yaw=(-0.10, 0.10),
        vy_grid=(-0.05, 0.0, 0.05),
        yaw_rate_grid=(-0.10, 0.0, 0.10),
    )
    assert meta.vy_grid == (-0.05, 0.0, 0.05)
    assert meta.yaw_rate_grid == (-0.10, 0.0, 0.10)
```

**Run**: expect failure.

### P2.3 — Impl: add `vy_grid` / `yaw_rate_grid` to `ReferenceLibraryMeta`

In `reference_library.py:341`:
```python
@dataclass
class ReferenceLibraryMeta:
    # ... existing fields ...
    vy_grid: Tuple[float, ...] = field(default_factory=tuple)
    yaw_rate_grid: Tuple[float, ...] = field(default_factory=tuple)
```
Update `save()` / `load()` (line 565+) to include these in `metadata.json` (convert tuple ↔ list).

**Verify**: P2.2 passes; existing library tests still pass.

### P2.4 — `build_library_for_3d_values` populates the new meta fields (failing test)

**Append**:
```python
def test_build_library_for_3d_values_populates_meta_grids() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    assert tuple(lib.meta.vy_grid) == (-0.05, 0.0, 0.05)
    assert tuple(lib.meta.yaw_rate_grid) == (-0.10, 0.0, 0.10)
```

**Run**: expect failure.

### P2.5 — Impl + commit

In `ZMPWalkGenerator.build_library_for_3d_values` (P1.11), pass `vy_grid=tuple(vy_values)`, `yaw_rate_grid=tuple(yaw_rate_values)` into `ReferenceLibraryMeta(...)`.

**Verify**: P2.4 passes.

**Commit**:
```bash
git add control/references/reference_library.py training/tests/test_reference_library_3d.py
git commit -m "v0.21.0 P2: ReferenceLibraryMeta carries vy/yaw_rate grids; verify lookup already 3D"
```

---

## P3 — Migrate `velocity_cmd` from scalar to 3-vector

**Files**: `/Users/ygli/projects/wildrobot/training/envs/env_info.py`, `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py`, `/Users/ygli/projects/wildrobot/training/configs/training_runtime_config.py`, `/Users/ygli/projects/wildrobot/control/references/runtime_reference_service.py`.

> This phase changes a foundational data shape. It MUST land in one commit so no half-state exists. All sub-steps share one final commit at P3.10.

### P3.1 — `WildRobotInfo.velocity_cmd` shape regression test (failing)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_velocity_cmd_3d_migration.py`
```python
from __future__ import annotations
import jax
import jax.numpy as jp
import pytest
from training.envs.env_info import get_expected_shapes


def test_wildrobot_info_velocity_cmd_expected_shape_is_3() -> None:
    shapes = get_expected_shapes(action_size=13)
    assert shapes["velocity_cmd"] == (3,)
```

**Run**: expect failure (currently `()`).

### P3.2 — Impl: bump `velocity_cmd` shape in env_info

In `env_info.py:107`, change `velocity_cmd: jnp.ndarray  # ()` → `velocity_cmd: jnp.ndarray  # (3,)`. Same at line 222 (NamedTuple fallback). In `get_expected_shapes()` line 285 set `"velocity_cmd": (3,)`.

**Verify**: P3.1 passes.

### P3.3 — `EnvConfig.cmd_deadzone` type bump (failing test)

**Append**:
```python
from training.configs.training_runtime_config import LocomotionEnvConfig


def test_env_config_cmd_deadzone_default_is_three_tuple() -> None:
    cfg = LocomotionEnvConfig()
    assert tuple(cfg.cmd_deadzone) == (0.0, 0.0, 0.0)
```

**Run**: expect failure (currently `float`).

### P3.4 — Impl: cmd_deadzone tuple migration + YAML shim

In `training_runtime_config.py:625` change `cmd_deadzone: float = 0.0` to:
```python
cmd_deadzone: Tuple[float, float, float] = (0.0, 0.0, 0.0)
```
Add a YAML-loader shim in `training/configs/training_config.py` (locate via `grep -n cmd_deadzone training/configs/training_config.py`): when `cmd_deadzone` reads as a scalar, broadcast to `(scalar, scalar, scalar)` before constructing the dataclass. Add to the same test file:
```python
def test_cmd_deadzone_scalar_yaml_broadcasts_to_three_tuple() -> None:
    from training.configs.training_config import load_training_config
    # smoke14 sets cmd_deadzone: 0.0666666667 (scalar)
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    assert tuple(cfg.env.cmd_deadzone) == pytest.approx(
        (0.0666666667, 0.0666666667, 0.0666666667), abs=1e-7
    )
```

**Verify**: P3.3 + new test pass; smoke14 loads.

### P3.5 — `EnvConfig.eval_velocity_cmd` type bump + sentinel (failing test)

**Append**:
```python
def test_env_config_eval_velocity_cmd_default_is_three_tuple_sentinel() -> None:
    cfg = LocomotionEnvConfig()
    assert tuple(cfg.eval_velocity_cmd) == (-1.0, -1.0, -1.0)
```

**Run**: expect failure.

### P3.6 — Impl: eval_velocity_cmd tuple migration + YAML shim + sentinel

In `training_runtime_config.py:635` change `eval_velocity_cmd: float = -1.0` to:
```python
eval_velocity_cmd: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
```
Add YAML shim in `training_config.py`: scalar `eval_velocity_cmd: 0.26` becomes `(0.26, 0.0, 0.0)` so smoke14 keeps working.

In `wildrobot_env.py:2529-2532` replace the sentinel comparator:
```python
# Before:
#   eval_cmd = jp.float32(self._config.env.eval_velocity_cmd)
#   new_cmd = jp.where(eval_cmd >= jp.float32(0.0), eval_cmd, wr.velocity_cmd)
# After:
eval_cmd = jp.asarray(self._config.env.eval_velocity_cmd, dtype=jp.float32)  # (3,)
use_override = jp.all(eval_cmd >= jp.float32(0.0))
new_cmd = jp.where(use_override, eval_cmd, wr.velocity_cmd)  # (3,)
new_wr = wr.replace(velocity_cmd=new_cmd.astype(jp.float32))
```

**Test**: append:
```python
def test_eval_velocity_cmd_scalar_yaml_broadcasts() -> None:
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    # smoke14 sets eval_velocity_cmd: 0.26 (scalar).
    assert tuple(cfg.env.eval_velocity_cmd) == pytest.approx((0.26, 0.0, 0.0))
```

**Verify**: P3.5 + new test pass.

### P3.7 — Env per-call-site fanout: scalar usages of `velocity_cmd` (failing test)

**Append**:
```python
def test_env_step_does_not_raise_on_three_vec_velocity_cmd() -> None:
    import jax
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    # state.info[WR_INFO_KEY].velocity_cmd has shape (3,) post-P3.
    from training.envs.env_info import WR_INFO_KEY
    assert state.info[WR_INFO_KEY].velocity_cmd.shape == (3,)
    # Stepping with zero action must not raise.
    _ = env.step(state, jp.zeros(env.action_size))
```

**Run**: expect failure (`jp.abs(velocity_cmd) > 1e-6` broadcasts to `(3,)`; downstream `is_standing` shape mismatches; etc).

### P3.8 — Impl: rewrite each scalar callsite

For each callsite from the P0.5 checklist, apply:
- `wildrobot_env.py:1218` — DELETE the line, parameter is now sourced from caller (handled in P6).
- `wildrobot_env.py:1837` — `cmd_active = (jp.linalg.norm(velocity_cmd) > jp.float32(1e-6)).astype(jp.float32)`.
- `wildrobot_env.py:2051` — `is_standing = jp.linalg.norm(velocity_cmd) < jp.float32(1e-6)`.
- `wildrobot_env.py:2867-2873` (reset metrics) — index `velocity_cmd[0]` for the forward-only diagnostics; keep lateral as `root_vel_h.linear[1] - velocity_cmd[1]`.
- `wildrobot_env.py:2896-2898` — split into three metrics: `tracking/velocity_cmd_vx_abs = jp.abs(velocity_cmd[0])`, `_vy_abs`, `_wz_abs`. Also keep an aggregate `tracking/velocity_cmd_norm = jp.linalg.norm(velocity_cmd)`.
- `wildrobot_env.py:2905-2907` — index `velocity_cmd[0]` in the ratio.
- `wildrobot_env.py:2962-2965` — split: pass `velocity_cmd[0]` to `velocity_cmd_mps` AND pass `velocity_cmd_y_mps=velocity_cmd[1]`, `yaw_rate_cmd_rps=velocity_cmd[2]` (the helper signature is extended in P3.9).
- `wildrobot_env.py:3217-3219` — `_sample_velocity_cmd` now returns `(3,)`, so `should_resample` broadcasts correctly; ensure `jp.where(should_resample, resampled_cmd, velocity_cmd)` uses a scalar bool.
- `wildrobot_env.py:3340/3359/3362/3383/3408` — same pattern as reset (index `velocity_cmd[0]` for forward-only terms; add `[1]`/`[2]` for the new axes).

Also update `training/core/metrics_registry.py` to add the new per-axis tracking metrics (`tracking/velocity_cmd_vy_abs`, `tracking/velocity_cmd_wz_abs`, `tracking/velocity_cmd_norm`) with `Reducer.MEAN`.

**Verify**: P3.7 passes; pre-existing tests `test_v0201_env_zero_action.py`, `test_cmd_velocity_tracking_mode.py` still pass (the latter does NOT touch `_cmd_forward_velocity_track_reward` — see C16 / P6 — they call the static helper with scalar args which are still valid).

### P3.9 — `compute_command_integrated_path_state` 3D extension (failing test)

**Append**:
```python
def test_path_state_integrates_lateral_velocity() -> None:
    from control.references.runtime_reference_service import RuntimeReferenceService
    out = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=1.0,
        velocity_cmd_mps=0.0,
        velocity_cmd_y_mps=0.10,
        yaw_rate_cmd_rps=0.0,
        dt_s=0.02,
    )
    # After 1 s at vy=0.10 with zero yaw, path_pos[1] ≈ 0.10.
    assert float(out["path_pos"][1]) == pytest.approx(0.10, abs=0.01)
    # path_pos[0] ≈ 0.
    assert abs(float(out["path_pos"][0])) < 0.01


def test_path_state_integrates_yaw_rate_for_pure_turn() -> None:
    from control.references.runtime_reference_service import RuntimeReferenceService
    out = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=1.0,
        velocity_cmd_mps=0.0,
        velocity_cmd_y_mps=0.0,
        yaw_rate_cmd_rps=0.10,
        dt_s=0.02,
    )
    # path_rot's z-imag component = sin(yaw / 2); yaw ≈ 0.10 rad → sin ≈ 0.05.
    import math
    expected_qz = math.sin(0.10 / 2.0)
    assert float(out["path_rot"][3]) == pytest.approx(expected_qz, abs=0.005)
```

**Run**: expect failure (`velocity_cmd_y_mps` is not yet a parameter; current path_pos[1] always 0).

### P3.10 — Impl: extend `compute_command_integrated_path_state` + commit

In `runtime_reference_service.py:181-261` extend the staticmethod signature:
```python
def compute_command_integrated_path_state(
    *,
    t_since_reset_s,
    velocity_cmd_mps,
    velocity_cmd_y_mps=0.0,       # NEW
    yaw_rate_cmd_rps=0.0,
    dt_s=0.02,
    default_root_pos_xyz=None,
) -> dict:
```
Replace the closed-form for `path_pos`:
```python
# Mirror TB motion_ref.py::integrate_path_state with (vx, vy) instead of vx-only.
vx = jnp.asarray(velocity_cmd_mps, dtype=jnp.float32)
vy = jnp.asarray(velocity_cmd_y_mps, dtype=jnp.float32)
# Discrete sum p_n = dt * sum_{k=1..n} Rz(k*theta) @ [vx, vy]
# Closed form via the same sin-ratio trick used today, but matrix form.
# ... derive vx_sum, vy_sum that produce x_world, y_world.
```
The existing scalar `vx`-only derivation generalises by rotating the per-step `[vx, vy]` vector by `Rz(k*theta)` and summing. For numerical stability, keep the `theta → 0` fallback that returns `n * dt * [vx, vy]`.

Update lookup-time:
- `wildrobot_env.py:2962-2965` — pass `velocity_cmd_y_mps=velocity_cmd[1]`, `yaw_rate_cmd_rps=velocity_cmd[2]`.

Also update existing test files that call this helper with scalar `velocity_cmd_mps`:
- `tests/test_runtime_reference_service.py:188/232/269` — keep scalar (the parameter is positional-name only; defaults supply zero vy/wz). NO test edits needed.

**Verify**: P3.9 passes. Re-run `tests/test_runtime_reference_service.py` — passes.

**Commit**:
```bash
git add training/envs/env_info.py training/envs/wildrobot_env.py \
        training/configs/training_runtime_config.py training/configs/training_config.py \
        control/references/runtime_reference_service.py \
        training/core/metrics_registry.py \
        training/tests/test_velocity_cmd_3d_migration.py
git commit -m "v0.21.0 P3: WildRobotInfo.velocity_cmd → (3,); path_state integrates (vy, wz)"
```

---

## P4 — Branched 3D command sampler (mirror TB exactly)

**File**: `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py` (around line 2539).
**File**: `/Users/ygli/projects/wildrobot/training/configs/training_runtime_config.py` (around line 108).

### P4.1 — `EnvConfig` accepts vy/wz ranges (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_env_config_3d_cmd.py`
```python
from __future__ import annotations
import pytest
from training.configs.training_runtime_config import LocomotionEnvConfig


def test_env_config_accepts_vy_and_yaw_rate_ranges() -> None:
    cfg = LocomotionEnvConfig(
        min_velocity=0.18, max_velocity=0.26,
        min_velocity_y=-0.13, max_velocity_y=0.13,
        max_yaw_rate=0.25,
    )
    assert cfg.min_velocity_y == -0.13
    assert cfg.max_velocity_y == 0.13
    assert cfg.max_yaw_rate == 0.25


def test_env_config_defaults_vy_wz_to_zero() -> None:
    cfg = LocomotionEnvConfig(min_velocity=0.18, max_velocity=0.26)
    assert cfg.min_velocity_y == 0.0
    assert cfg.max_velocity_y == 0.0
    assert cfg.max_yaw_rate == 0.0
```

### P4.2 — Impl: add fields to `LocomotionEnvConfig`

In `training_runtime_config.py` near line 108 add:
```python
min_velocity_y: float = 0.0
max_velocity_y: float = 0.0
# Symmetric default: ±max_yaw_rate.  Asymmetric ranges (e.g. for one-
# directional turning) can be expressed via two fields if needed later.
max_yaw_rate: float = 0.0
```

**Verify**: P4.1 passes.

### P4.3 — Sampler return shape (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_sample_velocity_cmd_3d.py`
```python
from __future__ import annotations
import jax
import jax.numpy as jp
import pytest
from training.envs.wildrobot_env import WildRobotEnv
from training.configs.training_config import load_training_config


def _env_3d():
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    return WildRobotEnv(cfg)


def test_sample_velocity_cmd_returns_3_vector() -> None:
    env = _env_3d()
    key = jax.random.PRNGKey(0)
    cmd = env._sample_velocity_cmd(key)
    assert cmd.shape == (3,)
```

**Run**: expect failure (current returns scalar).

### P4.4 — Impl: sampler returns 3-vec zero-cmd path only

Refactor `_sample_velocity_cmd` (line 2539) so it always returns shape `(3,)`. Wire the three branches but only implement the `zero` branch for now — the others can return `jp.zeros(3)` until the next sub-step:
```python
def _sample_velocity_cmd(self, rng: jax.Array) -> jax.Array:
    return jp.zeros((3,), dtype=jp.float32)
```

**Verify**: P4.3 passes; smoke1 reset succeeds (the rest of the env now sees a 3-vec cmd from P3).

### P4.5 — Walk branch ellipse (failing test, then impl)

**Append**:
```python
def test_walk_branch_samples_within_ellipse() -> None:
    env = _env_3d()
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    # Forward axis within [0, vx_max].
    assert (cmds[:, 0] >= 0.0).all()
    assert (cmds[:, 0] <= env._config.env.max_velocity + 1e-6).all()
    # Lateral within ±vy_max.
    assert (jp.abs(cmds[:, 1]) <= env._config.env.max_velocity_y + 1e-6).all()
    # wz=0 in the walk branch.
    # (Run with cmd_zero_chance=0 and cmd_turn_chance=0 so only walk fires.)
```

**Impl**: implement the walk branch mirroring TB `walk_env.py:256-280` literally:
```python
def _sample_walk_command(rng_3, rng_4) -> jax.Array:
    theta = jax.random.uniform(rng_3, (), minval=0.0, maxval=2 * jp.pi)
    x_hi = jp.float32(self._config.env.max_velocity)
    x_lo = jp.float32(self._config.env.min_velocity)
    # TB uses asymmetric per-axis limits: sin(theta) > 0 → use +max, else -lo.
    x_max = jp.where(jp.sin(theta) > 0.0, x_hi, -x_lo)
    x = jax.random.uniform(
        rng_4, (),
        minval=self._config.env.cmd_deadzone[0],
        maxval=jp.abs(x_max),
    ) * jp.sin(theta)
    y_hi = jp.float32(self._config.env.max_velocity_y)
    y_lo = jp.float32(self._config.env.min_velocity_y)
    y_max = jp.where(jp.cos(theta) > 0.0, y_hi, -y_lo)
    y = jax.random.uniform(
        rng_4, (),
        minval=self._config.env.cmd_deadzone[1],
        maxval=jp.abs(y_max),
    ) * jp.cos(theta)
    return jp.stack([x, y, jp.float32(0.0)])
```

**Verify**: P4.5 passes when smoke1 YAML sets `cmd_zero_chance: 0.0`, `cmd_turn_chance: 0.0`.

### P4.6 — Turn branch (failing test, then impl)

**Append**:
```python
def test_turn_branch_samples_yaw_within_range() -> None:
    """Set zero_chance=1.0-eps for turn-only to dominate (test-only YAML)."""
    # Stub: build a config with cmd_turn_chance=1.0, cmd_zero_chance=0.0.
    # Sample 4096 cmds, assert all have vx=vy=0 and |wz| <= max_yaw_rate.
    ...
```

**Impl**: implement the turn branch mirroring TB:282-305 (sign-randomized magnitude in `[deadzone[2], max_yaw_rate]`):
```python
def _sample_turn_command(rng_5, rng_6) -> jax.Array:
    sign_bit = jax.random.uniform(rng_5, ()) < 0.5
    wz_pos = jax.random.uniform(
        rng_6, (),
        minval=self._config.env.cmd_deadzone[2],
        maxval=self._config.env.max_yaw_rate,
    )
    wz_neg = -jax.random.uniform(
        rng_6, (),
        minval=self._config.env.cmd_deadzone[2],
        maxval=self._config.env.max_yaw_rate,
    )
    wz = jp.where(sign_bit, wz_pos, wz_neg)
    return jp.stack([jp.float32(0.0), jp.float32(0.0), wz])
```

**Verify**: P4.6 passes.

### P4.7 — Branch selector (failing test, impl, commit)

**Append**:
```python
def test_branch_marginal_distribution() -> None:
    """With zero_chance=0.2, turn_chance=0.2 the marginal must be ~20% zero,
    ~20% turn, ~60% walk (within ±3% tolerance over 4096 samples)."""
    env = _env_3d()  # smoke1 YAML sets 0.2 / 0.2
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    is_zero = jp.all(jp.abs(cmds) < 1e-6, axis=-1)
    is_turn = (jp.abs(cmds[:, 2]) > 1e-6) & (jp.abs(cmds[:, 0]) < 1e-6) & (jp.abs(cmds[:, 1]) < 1e-6)
    is_walk = (jp.abs(cmds[:, 0]) > 1e-6) | (jp.abs(cmds[:, 1]) > 1e-6)
    assert abs(float(is_zero.mean()) - 0.20) < 0.03
    assert abs(float(is_turn.mean()) - 0.20) < 0.03
    assert abs(float(is_walk.mean()) - 0.60) < 0.03
```

**Impl**: glue the three branches with a `jax.random.uniform` brancher mirroring TB `walk_env.py:307-316`:
```python
def _sample_velocity_cmd(self, rng: jax.Array) -> jax.Array:
    rng_2, rng_3, rng_4, rng_5, rng_6 = jax.random.split(rng, 5)
    u = jax.random.uniform(rng_2, ())
    walk_cmd = self._sample_walk_command(rng_3, rng_4)
    turn_cmd = self._sample_turn_command(rng_5, rng_6)
    zero_chance = jp.float32(self._config.env.cmd_zero_chance)
    turn_chance = jp.float32(self._config.env.cmd_turn_chance)
    cmd = jp.where(
        u < zero_chance,
        jp.zeros(3, dtype=jp.float32),
        jp.where(u < zero_chance + turn_chance, turn_cmd, walk_cmd),
    )
    return cmd.astype(jp.float32)
```

**Verify**: P4.7 passes; P4.3/P4.5/P4.6 still pass.

**Commit**:
```bash
git add training/configs/training_runtime_config.py training/envs/wildrobot_env.py \
        training/tests/test_env_config_3d_cmd.py training/tests/test_sample_velocity_cmd_3d.py
git commit -m "v0.21.0 P4: branched (zero/turn/walk) 3D cmd sampler mirroring TB"
```

---

## P5 — 3D reference library plumbing in env

**File**: `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py` (`_init_offline_service` line 764; `_lookup_offline_window` line 1248).
**File**: `/Users/ygli/projects/wildrobot/control/references/runtime_reference_service.py` (add `_offline_cmd_keys` + `lookup_jax_3d`).
**File**: `/Users/ygli/projects/wildrobot/training/configs/training_runtime_config.py` (`loc_ref_command_axes_3d`, vy/yaw_rate grids).

### P5.1 — `loc_ref_command_axes_3d` toggle + 3D grid fields (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_loc_ref_3d_config.py`
```python
def test_env_config_accepts_3d_axes_toggle_and_grids() -> None:
    from training.configs.training_runtime_config import LocomotionEnvConfig
    cfg = LocomotionEnvConfig(
        loc_ref_command_axes_3d=True,
        loc_ref_offline_command_vy_grid=(-0.10, 0.0, 0.10),
        loc_ref_offline_command_yaw_rate_grid=(-0.20, 0.0, 0.20),
    )
    assert cfg.loc_ref_command_axes_3d is True
    assert cfg.loc_ref_offline_command_vy_grid == (-0.10, 0.0, 0.10)
    assert cfg.loc_ref_offline_command_yaw_rate_grid == (-0.20, 0.0, 0.20)


def test_loc_ref_command_axes_3d_defaults_false() -> None:
    from training.configs.training_runtime_config import LocomotionEnvConfig
    cfg = LocomotionEnvConfig()
    assert cfg.loc_ref_command_axes_3d is False
```

**Run**: expect failure.

### P5.2 — Impl: add three fields + commit

In `training_runtime_config.py` near line 334 add:
```python
loc_ref_command_axes_3d: bool = False
loc_ref_offline_command_vy_grid: Tuple[float, ...] = field(default_factory=tuple)
loc_ref_offline_command_yaw_rate_grid: Tuple[float, ...] = field(default_factory=tuple)
```
Note: do NOT widen the `loc_ref_version` validator at `wildrobot_env.py:218-223` — keep `v3_offline_library`. The new toggle keys 3D behavior off `loc_ref_command_axes_3d`.

**Verify**: P5.1 passes.

### P5.3 — `_init_offline_service` builds 3D grid (failing test)

**Append**:
```python
def test_init_offline_service_builds_3d_grid_when_toggled() -> None:
    import jax
    from training.envs.wildrobot_env import WildRobotEnv
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    env = WildRobotEnv(cfg)
    # Smoke1 grid: 3 vx × 3 vy × 3 wz = 27 bins.
    assert env._offline_cmd_keys.shape == (27, 3)
```

**Run**: expect failure.

### P5.4 — Impl: `_init_offline_service` 3D branch

In `wildrobot_env.py:820-848` add a branch:
```python
if cmd_conditioned and bool(getattr(self._config.env, "loc_ref_command_axes_3d", False)):
    vx_values = list(vx_grid)
    vy_values = list(self._config.env.loc_ref_offline_command_vy_grid) or [0.0]
    yaw_values = list(self._config.env.loc_ref_offline_command_yaw_rate_grid) or [0.0]
    lib = ZMPWalkGenerator().build_library_for_3d_values(
        vx_values=vx_values,
        vy_values=vy_values,
        yaw_rate_values=yaw_values,
    )
    cmd_keys = [(vx, vy, wz) for vx in vx_values for vy in vy_values for wz in yaw_values]
else:
    # Legacy 1D path (unchanged).
    if offline_path:
        lib = ReferenceLibrary.load(offline_path)
    else:
        lib = ZMPWalkGenerator().build_library_for_vx_values(list(vx_grid))
    cmd_keys = [(vx, 0.0, 0.0) for vx in vx_grid]
```
Then in the per-bin stacking loop, also build `cmd_keys_arr = jp.asarray(cmd_keys, dtype=jp.float32)` and store as `self._offline_cmd_keys`. For the legacy path, `cmd_keys` keeps shape `(n_bins, 3)` with vy/wz zeroed.

**Verify**: P5.3 passes.

### P5.5 — `_lookup_offline_window` heading-frame correction + 3D nearest-key + commit

**Failing test**:
```python
def test_lookup_offline_window_uses_3d_nearest_key() -> None:
    import jax.numpy as jp
    env = _env_3d_via_smoke1()
    win_a = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=jp.asarray([0.20, 0.10, 0.0], dtype=jp.float32),
    )
    win_b = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=jp.asarray([0.20, -0.10, 0.0], dtype=jp.float32),
    )
    # The two bin selections must differ (different cmd → different bin).
    assert float(win_a["selected_vy"]) != float(win_b["selected_vy"])
```

**Impl**: in `wildrobot_env.py:1248-1340`, generalize to accept a 3-vec cmd:
1. Change signature: `velocity_cmd: Optional[jax.Array] = None` — but the parameter is now expected to be shape `(3,)` (callers updated in P3).
2. Add an optional `path_rot` parameter so the caller can pass the current torso path-rot for heading-frame correction (mirror TB `walk_zmp_ref.py:132-137`):
```python
def _lookup_offline_window(
    self,
    step_idx: jax.Array,
    velocity_cmd: Optional[jax.Array] = None,  # (3,) when supplied
    path_rot_wxyz: Optional[jax.Array] = None,  # (4,) torso heading-rot
) -> Dict[str, jax.Array]:
```
3. Replace the 1D `argmin` with `argmin(L2(_offline_cmd_keys - cmd_corrected))` where `cmd_corrected` first rotates `cmd[:2]` by `heading_rot.inv() @ path_rot`:
```python
# Heading-frame correction (TB walk_zmp_ref.py:132-137).
if path_rot_wxyz is not None:
    # ... rotate cmd[:2] into the heading-error frame.
    cmd_xy_corrected = ...  # (2,)
    cmd_corrected = jp.stack([cmd_xy_corrected[0], cmd_xy_corrected[1], cmd[2]])
else:
    cmd_corrected = cmd
bin_idx = jp.argmin(jp.linalg.norm(self._offline_cmd_keys - cmd_corrected, axis=-1)).astype(jp.int32)
```
4. Return diagnostics: `selected_vx = _offline_cmd_keys[bin_idx, 0]`, `selected_vy = ...[..., 1]`, `selected_yaw_rate = ...[..., 2]`, `cmd_bin_l2_err = jp.linalg.norm(...)`.

Update the two callers (line 2658, 2957, 2973) to pass `path_rot_wxyz` from the current `path_state["path_rot"]` if available, else `None`.

**Verify**: P5.5 passes; legacy tests `test_cmd_velocity_tracking_mode.py::test_legacy_mode_selected_bin_diagnostics_use_configured_vx` still passes (legacy path returns the configured vx with vy=wz=0).

**Commit**:
```bash
git add training/configs/training_runtime_config.py training/envs/wildrobot_env.py \
        control/references/runtime_reference_service.py \
        training/tests/test_loc_ref_3d_config.py
git commit -m "v0.21.0 P5: 3D nearest-key lookup + heading-frame correction + 27-bin smoke grid"
```

---

## P6 — Reward updates: optional `vy_cmd` arg + yaw_rate tracking term

**File**: `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py` (lines 1163–1244, 1777–1800, 3050–3120).

### P6.1 — Add optional `vy_cmd` arg to `_cmd_forward_velocity_track_reward` (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_cmd_lateral_velocity_tracking.py`
```python
import math
import pytest
from training.envs.wildrobot_env import WildRobotEnv


def test_2d_cmd_tracking_uses_supplied_vy_cmd() -> None:
    """Post-v0.21 dim=2 reward accepts an external vy_cmd; the lateral
    axis penalty is now (vy_actual - vy_cmd)^2 instead of vy_actual^2."""
    reward, xy_err, lat_abs, _ = WildRobotEnv._cmd_forward_velocity_track_reward(
        forward_velocity=0.10,
        lateral_velocity=0.05,
        velocity_cmd=0.10,
        vy_cmd=0.05,            # NEW kwarg — actual matches cmd, so lateral err = 0
        ref_forward_velocity=0.10,
        ref_lateral_velocity=0.05,
        alpha=562.5,
        track_dim=2,
    )
    assert float(reward) == pytest.approx(1.0)
```

**Run**: expect failure (`unexpected keyword argument 'vy_cmd'`).

### P6.2 — Impl: add `vy_cmd: jax.Array = jp.float32(0.0)` kwarg

In `wildrobot_env.py:1163-1244`, add `vy_cmd: jax.Array = jp.float32(0.0)` to the static helper signature. Replace `vy_err_cmd = lateral_velocity` (line 1218) with `vy_err_cmd = lateral_velocity - vy_cmd`. Update the comment at line 1195 to clarify the new contract.

**Verify**: P6.1 passes. Re-run `test_cmd_velocity_tracking_mode.py` — `test_scalar_cmd_tracking_ignores_lateral_velocity`, `test_2d_cmd_tracking_targets_command_not_reference`, `test_2d_cmd_tracking_penalizes_lateral_drift_against_zero_cmd` all still pass (they call without `vy_cmd`, default 0.0 reproduces prior behavior — explicit C16 mitigation).

### P6.3 — Wire real `vy_cmd` from the 3-vec cmd at the call site

At `wildrobot_env.py:1791` (inside `_compute_reward_terms`), change:
```python
self._cmd_forward_velocity_track_reward(
    forward_velocity=...,
    lateral_velocity=...,
    velocity_cmd=velocity_cmd[0],     # WAS: velocity_cmd (scalar)
    vy_cmd=velocity_cmd[1],           # NEW
    ...
)
```
(`velocity_cmd` is the 3-vec in the caller post-P3.)

**Test**: append to `test_cmd_lateral_velocity_tracking.py`:
```python
def test_env_reward_consumes_real_vy_cmd_from_3vec() -> None:
    # End-to-end: build env from smoke1 yaml, run one step with a hand-
    # crafted vy_cmd, assert the cmd_velocity_xy_err metric shrinks when
    # the actual lateral velocity matches the commanded vy.
    ...
```

**Verify**: passes.

### P6.4 — Yaw-rate tracking term (failing + impl + commit)

**Failing test**:
```python
def test_yaw_rate_track_reward_peaks_at_cmd() -> None:
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.10,
        yaw_rate_cmd=0.10,
        alpha=562.5,
    )
    assert float(reward) == pytest.approx(1.0)


def test_yaw_rate_track_reward_decays_off_cmd() -> None:
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.0,
        yaw_rate_cmd=0.10,
        alpha=562.5,
    )
    # err_sq = 0.01 → reward = exp(-5.625) ≈ 0.00361
    import math
    assert float(reward) == pytest.approx(math.exp(-5.625), rel=1e-4)
```

**Impl**:
1. Add static helper:
```python
@staticmethod
def _yaw_rate_track_reward(
    *, ang_vel_z: jax.Array, yaw_rate_cmd: jax.Array, alpha: jax.Array,
) -> jax.Array:
    err = ang_vel_z - yaw_rate_cmd
    return jp.exp(-alpha * err * err).astype(jp.float32)
```
2. Wire into `_compute_reward_terms` (line 1591) right after the existing cmd_forward term. Add `r_yaw_rate_track` to the returned `terms` dict.
3. Add `cmd_yaw_rate_track: float = 0.0` to `RewardWeightsConfig` (training_runtime_config.py around line 877).
4. Apply weight in the aggregator (line 2186): `+ w.cmd_yaw_rate_track * terms["r_yaw_rate_track"]`.
5. Add metric `tracking/yaw_rate_err` to `training/core/metrics_registry.py` with `Reducer.MEAN`.

**Verify**: P6.4 tests pass. Existing tests still pass.

**Commit**:
```bash
git add training/envs/wildrobot_env.py training/configs/training_runtime_config.py \
        training/core/metrics_registry.py training/tests/test_cmd_lateral_velocity_tracking.py
git commit -m "v0.21.0 P6: reward consumes real vy_cmd; yaw_rate tracking term added (default w=0)"
```

---

## P7 — Observation layout `wr_obs_v8_cmd3d`

**File**: `/Users/ygli/projects/wildrobot/policy_contract/spec.py`, `spec_builder.py`, `jax/obs.py`.

> P7 only touches `policy_contract/` files and is independent of `wildrobot_env.py`. It DOES depend on P3 (the env passes a 3-vec `velocity_cmd` into `build_observation`).

### P7.1 — Register `wr_obs_v8_cmd3d` in `SUPPORTED_LAYOUT_IDS` (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_obs_v8_cmd3d.py`
```python
def test_v8_layout_id_is_supported() -> None:
    from policy_contract.spec import SUPPORTED_LAYOUT_IDS
    assert "wr_obs_v8_cmd3d" in SUPPORTED_LAYOUT_IDS
```

**Run**: expect failure.

### P7.2 — Impl: add to `SUPPORTED_LAYOUT_IDS`, spec_builder, `_validate_observation`, and obs.py dispatcher

In `policy_contract/spec.py:11-32` add `"wr_obs_v8_cmd3d",` to the set.

In `policy_contract/spec_builder.py::_build_obs_layout` add the v8 branch — clone v7 then append a NEW 2-vec slot `velocity_cmd_lateral_yaw` AT THE END (before padding):
```python
if layout_id == "wr_obs_v8_cmd3d":
    proprio_bundle = 3 + 4 + 3 * action_dim
    return [
        ObsFieldSpec(name="gravity_local", size=3, ...),
        ObsFieldSpec(name="angvel_heading_local", size=3, ...),
        ObsFieldSpec(name="joint_pos_normalized", size=action_dim, ...),
        ObsFieldSpec(name="joint_vel_normalized", size=action_dim, ...),
        ObsFieldSpec(name="foot_switches", size=4, ...),
        ObsFieldSpec(name="prev_action", size=action_dim, ...),
        ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),  # vx_cmd (unchanged)
        ObsFieldSpec(name="loc_ref_phase_sin_cos", size=2, ...),
        ObsFieldSpec(name="proprio_history",
                     size=PROPRIO_HISTORY_FRAMES * proprio_bundle, ...),
        ObsFieldSpec(name="velocity_cmd_lateral_yaw", size=2, units="m_s_and_rad_s"),  # NEW: (vy_cmd, wz_cmd)
        ObsFieldSpec(name="padding", size=1, units="unused"),
    ]
```

In `policy_contract/spec.py::_validate_observation` add a `wr_obs_v8_cmd3d` branch with the matching `expected` list.

In `policy_contract/jax/obs.py:46-51` add `"wr_obs_v8_cmd3d"` to the supported set. In the per-layout block, copy the v7 branch (line 180-192) and add ONE extra `parts.append(...)` at the end (before the existing trailing `parts.append(jnp.zeros((1,), dtype=jnp.float32))` padding):
```python
if spec.observation.layout_id == "wr_obs_v8_cmd3d":
    parts.append(phase)
    if proprio_history is None:
        raise ValueError("wr_obs_v8_cmd3d requires proprio_history; got None.")
    parts.append(jnp.asarray(proprio_history, dtype=jnp.float32).reshape(-1))
    # NEW: 2-vec lateral+yaw cmd slot.  velocity_cmd[0] (vx) already in
    # the shared 1-vec slot; here we append the (vy, wz) tail.
    if velocity_cmd is not None and jnp.asarray(velocity_cmd).size == 3:
        parts.append(jnp.asarray(velocity_cmd, dtype=jnp.float32)[1:].reshape(2))
    else:
        parts.append(jnp.zeros((2,), dtype=jnp.float32))
```
Also update the shared `velocity_cmd` slot at line 151 so v8 takes only `velocity_cmd[0]`:
```python
parts.append(
    jnp.atleast_1d(jnp.asarray(velocity_cmd, dtype=jnp.float32))[..., :1].reshape(1)
)
```
(For v1–v7 this slices a 1-vec down to 1-vec — no-op; for v8 it takes the vx coordinate of the 3-vec.)

### P7.3 — Validate the v8 layout end-to-end (failing → impl → verify)

**Append**:
```python
def test_v8_spec_validation_passes_and_obs_concat_matches_obs_dim() -> None:
    import jax.numpy as jp
    from policy_contract.spec_builder import build_policy_spec
    spec = build_policy_spec(
        robot_name="wildrobot_v2",
        actuated_joint_specs=[{"name": f"j{i}", "range": [-1.0, 1.0],
                               "policy_action_sign": 1.0,
                               "max_velocity_rad_s": 10.0} for i in range(13)],
        action_filter_alpha=0.5,
        layout_id="wr_obs_v8_cmd3d",
        mapping_id="pos_target_rad_v1",
        home_ctrl_rad=[0.0] * 13,
    )
    # If spec validation passes inside build_policy_spec, no exception.
    assert spec.observation.layout_id == "wr_obs_v8_cmd3d"
    # Sum of layout sizes equals obs_dim.
    sizes = [f.size for f in spec.observation.layout]
    assert sum(sizes) == spec.model.obs_dim


def test_v8_build_observation_produces_obs_with_3d_cmd_slot() -> None:
    import jax.numpy as jp
    from policy_contract.jax.obs import build_observation_from_components
    from policy_contract.spec_builder import build_policy_spec
    spec = build_policy_spec(  # same as above
        ...,
        layout_id="wr_obs_v8_cmd3d",
        ...,
    )
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    proprio_history = jp.zeros((15, proprio_bundle), dtype=jp.float32)
    obs = build_observation_from_components(
        spec=spec,
        gravity_local=jp.zeros(3),
        angvel_heading_local=jp.zeros(3),
        joint_pos_normalized=jp.zeros(action_dim),
        joint_vel_normalized=jp.zeros(action_dim),
        foot_switches=jp.zeros(4),
        prev_action=jp.zeros(action_dim),
        velocity_cmd=jp.array([0.20, 0.05, 0.10], dtype=jp.float32),
        loc_ref_phase_sin_cos=jp.zeros(2),
        proprio_history=proprio_history,
    )
    assert obs.shape == (spec.model.obs_dim,)
```

**Run**: failing then passing after P7.2 impl. Regression: also assert v7 still validates with `velocity_cmd=jp.array([0.20])`.

### P7.4 — Env init guard accepts v8 + commit

In `wildrobot_env.py:225-239`, extend the allow-list:
```python
if layout_id not in (
    "wr_obs_v6_offline_ref_history",
    "wr_obs_v7_phase_proprio",
    "wr_obs_v8_cmd3d",          # NEW
):
    raise ValueError(...)
```

**Test**: append to `test_obs_v8_cmd3d.py`:
```python
def test_env_init_accepts_v8_layout() -> None:
    from training.envs.wildrobot_env import WildRobotEnv
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    # No exception.
    env = WildRobotEnv(cfg)
    assert env._policy_spec.observation.layout_id == "wr_obs_v8_cmd3d"
```

**Verify**: pass.

**Commit**:
```bash
git add policy_contract/spec.py policy_contract/spec_builder.py policy_contract/jax/obs.py \
        training/envs/wildrobot_env.py training/tests/test_obs_v8_cmd3d.py
git commit -m "v0.21.0 P7: register wr_obs_v8_cmd3d (cmd[0] in shared slot, (vy,wz) appended)"
```

---

## P8 — Smoke YAML + residual scales

### P8.1 — `ppo_walking_v0210_smoke1_lateral_yaw.yaml`

**File**: `/Users/ygli/projects/wildrobot/training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`

Clone `ppo_walking_v0201_smoke14.yaml` and apply diffs (use the REAL field names — NOT `obs_version`):
```yaml
env:
  actor_obs_layout_id: wr_obs_v8_cmd3d        # was: wr_obs_v7_phase_proprio
  min_velocity: 0.18
  max_velocity: 0.26
  min_velocity_y: -0.13                        # NEW
  max_velocity_y:  0.13                        # NEW
  max_yaw_rate:    0.25                        # NEW
  cmd_zero_chance: 0.20
  cmd_turn_chance: 0.20                        # was placeholder; now active (S5)
  cmd_deadzone: [0.02, 0.02, 0.05]             # NEW 3-tuple form
  eval_velocity_cmd: [0.18, 0.04, 0.10]        # NEW 3-tuple form
  loc_ref_version: v3_offline_library          # UNCHANGED (C5)
  loc_ref_command_conditioned: true
  loc_ref_command_axes_3d: true                # NEW (C5)
  loc_ref_command_grid_interval: 0.04          # gives vx_grid = [0.18, 0.22, 0.26]
  loc_ref_offline_command_vx: 0.22
  loc_ref_offline_command_vy_grid: [-0.10, 0.0, 0.10]      # NEW (C6) — 3 entries
  loc_ref_offline_command_yaw_rate_grid: [-0.20, 0.0, 0.20] # NEW (C6) — 3 entries
  # Library size: 3 * 3 * 3 = 27 bins.  S3 napkin math: ~50 MB stacked
  # on-device per env (constant across the brax vectorization), fine
  # for smoke1.
  loc_ref_residual_scale_per_joint:
    # cloned from smoke14, then lateral DoFs * 1.5 (P8.2 below)
    left_hip_roll:    {{base * 1.5}}
    right_hip_roll:   {{base * 1.5}}
    left_ankle_roll:  {{base * 1.5}}
    right_ankle_roll: {{base * 1.5}}
    left_hip_yaw:     {{base * 1.5}}
    right_hip_yaw:    {{base * 1.5}}
    # all other joints unchanged from smoke14
reward_weights:
  cmd_velocity_track_dim: 2                    # was 1 in smoke14; activates vy axis
  cmd_velocity_track: 2.0                      # unchanged
  cmd_yaw_rate_track:  1.5                     # NEW (default 0.0 in code; smoke1 sets active)
  # all other weights unchanged from smoke14
```

**Test**: `/Users/ygli/projects/wildrobot/training/tests/test_config_load_v0210_smoke1.py`
```python
import pytest
from training.configs.training_config import load_training_config


def test_smoke1_yaml_loads_and_has_3d_axes_enabled() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    assert cfg.env.actor_obs_layout_id == "wr_obs_v8_cmd3d"
    assert cfg.env.loc_ref_command_axes_3d is True
    assert tuple(cfg.env.cmd_deadzone) == (0.02, 0.02, 0.05)
    assert tuple(cfg.env.eval_velocity_cmd) == (0.18, 0.04, 0.10)
    assert cfg.env.max_yaw_rate == 0.25
    assert cfg.reward_weights.cmd_yaw_rate_track == 1.5
```

### P8.2 — Residual scales test + commit

**Append**:
```python
def test_smoke1_has_bumped_lateral_residual_scales() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    scales = cfg.env.loc_ref_residual_scale_per_joint
    base = load_training_config(
        "training/configs/ppo_walking_v0201_smoke14.yaml"
    ).env.loc_ref_residual_scale_per_joint
    for j in (
        "left_hip_roll", "right_hip_roll",
        "left_ankle_roll", "right_ankle_roll",
        "left_hip_yaw", "right_hip_yaw",
    ):
        assert scales[j] == pytest.approx(base[j] * 1.5)
```

**Commit**:
```bash
git add training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml \
        training/tests/test_config_load_v0210_smoke1.py
git commit -m "v0.21.0 P8: smoke1 YAML (actor_obs_layout_id, 3D cmd, 27-bin lib, 1.5x lateral)"
```

---

## P9 — Integration tests + walking_training.md gates

### P9.1 — End-to-end env rollout test

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_env_step_lateral_yaw_e2e.py`
```python
import jax
import jax.numpy as jp
import pytest
from training.envs.wildrobot_env import WildRobotEnv
from training.envs.env_info import WR_INFO_KEY
from training.configs.training_config import load_training_config


def _env():
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    return WildRobotEnv(cfg)


def test_env_rollout_with_lateral_cmd_shows_pelvis_y_drift() -> None:
    env = _env()
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(velocity_cmd=jp.array([0.20, 0.10, 0.0], dtype=jp.float32))
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    for _ in range(50):
        state = env.step(state, jp.zeros(env.action_size),
                         disable_pushes=True, disable_cmd_resample=True)
    # Pelvis y must drift positive (planner stride + path_state integration).
    assert state.pipeline_state.qpos[1] > 0.005


def test_env_rollout_with_yaw_cmd_changes_torso_yaw() -> None:
    env = _env()
    state = env.reset(jax.random.PRNGKey(1))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(velocity_cmd=jp.array([0.0, 0.0, 0.20], dtype=jp.float32))
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    for _ in range(50):
        state = env.step(state, jp.zeros(env.action_size),
                         disable_pushes=True, disable_cmd_resample=True)
    # Pelvis quat z component must change (yaw accumulated).
    qz_init = 0.0  # qpos quat is wxyz; init quat ≈ (1,0,0,0) → qz_init = qpos[6]
    qz_final = state.pipeline_state.qpos[6]
    assert abs(float(qz_final) - qz_init) > 1e-3
```

### P9.2 — v7 layout regression (failing + impl)

**Append**:
```python
def test_smoke14_env_still_initializes_under_v7() -> None:
    """P7 must not break v7 callers."""
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    assert env._policy_spec.observation.layout_id == "wr_obs_v7_phase_proprio"
    # velocity_cmd is now 3-vec for the env but the obs slot is still 1-vec for v7.
    wr = state.info[WR_INFO_KEY]
    assert wr.velocity_cmd.shape == (3,)
```

### P9.3 — `walking_training.md` G6 / G7 gates + commit

**File**: `/Users/ygli/projects/wildrobot/training/docs/walking_training.md`

Append a new section under "Gate definitions" (locate via `grep -n '^### G4' training/docs/walking_training.md`):
```markdown
### G6 — Lateral command tracking (v0.21.0+)

Definition: at eval cmd `(vx_eval, vy_eval, 0)` with `vy_eval > 0`, the
signed ratio `lateral_velocity / vy_eval` is ≥ 0.5 averaged over the
last 100 eval steps, across ≥3 seeds.  (S6: signed ratio gate is
strictly easier to pass than `|err| < 0.5 * |cmd|` and matches the
smoke14 G4-style criterion.)

### G7 — Yaw command tracking (v0.21.0+)

Definition: at eval cmd `(0, 0, wz_eval)` with `wz_eval > 0`, the
signed ratio `ang_vel_z / wz_eval` is ≥ 0.5 averaged over the last
100 eval steps, across ≥3 seeds.

### Promotion criterion (v0.21.0)

G4 + G5 (forward) MUST NOT regress beyond 10% relative to the
v0.20.1-smoke14 baseline at `eval_velocity_cmd = (0.18, 0, 0)`.

G6 + G7 MUST clear ≥0.5 signed ratio on the first smoke before
increasing vy / wz ranges or weights.
```

**Commit**:
```bash
git add training/tests/test_env_step_lateral_yaw_e2e.py training/docs/walking_training.md
git commit -m "v0.21.0 P9: e2e lateral/yaw rollout tests + G6/G7 gate definitions"
```

---

## P10 — CHANGELOG entry (after smoke results)

### P10.1 — `training/CHANGELOG.md`

**Only after smoke1 run completes and is analyzed**:
```markdown
## v0.21.0-smoke1-lateral-yaw — <ISO date> — wandb: <run id>

**Scope**: First lateral+yaw smoke. Extends ZMP prior to 3 axes (vx, vy, wz);
adds obs layout `wr_obs_v8_cmd3d` (legacy 1-vec `velocity_cmd` slot keeps
vx_cmd, new 2-vec `velocity_cmd_lateral_yaw` slot appended for vy/wz);
activates dim=2 cmd reward with real `vy_cmd` and adds `cmd_yaw_rate_track`
term (default weight 0.0 — S8 — smoke1 sets 1.5 explicitly).

**Architecture deltas**:
- `WildRobotInfo.velocity_cmd` migrated `() → (3,)` across ~10 callsites.
- `cmd_deadzone` migrated `float → tuple[float, float, float]` with YAML
  compat shim.
- `eval_velocity_cmd` migrated `float → tuple[float, float, float]` with
  YAML compat shim and sentinel `(-1, -1, -1)`.
- `cmd_turn_chance` (existing field, S5) is now actually consumed by the
  branched sampler — was reserved/placeholder pre-v0.21.0.
- `loc_ref_command_axes_3d` (NEW boolean) gates 3D library; existing
  `loc_ref_version: v3_offline_library` validator is UNCHANGED.
- `compute_command_integrated_path_state` now integrates (vy, wz) too
  (was hard-coded `yaw_rate_cmd_rps=0.0` and vy=0).
- 3D library lookup uses heading-frame correction (TB walk_zmp_ref.py
  parity).

**Gates**:
- G4 forward stability: <pass/fail>
- G5 forward step amplitude: <pass/fail>
- G6 lateral tracking (signed ratio ≥ 0.5): <%>
- G7 yaw tracking (signed ratio ≥ 0.5): <%>

**Outcomes**:
- <observations from the wandb run>
- <next-version proposals — likely either widen wz range if G7 saturates
  or drop rotation_radius if turn radius too large>
```

---

## End-to-end verification

After P10.1 the full pipeline should be:
1. `uv run pytest training/tests/ -q` — all green (including the v0.20 regressions: `test_cmd_velocity_tracking_mode.py`, `test_v0201_env_zero_action.py`, `test_runtime_reference_service.py`).
2. `uv run python training/train.py --config training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml --max-steps 500 --seeds 3` — runs 500 PPO steps without crash.
3. Eval rollout with cmd `(0.18, 0.04, 0.10)` shows visible strafe + turn motion in the rendered video (no black frames).
4. wandb metrics `tracking/velocity_cmd_vy_abs`, `tracking/yaw_rate_err` populate and trend toward zero.
5. Forward G4/G5 metrics within 10% of smoke14 baseline.

---

## Files touched (summary)

**Create**:
- `training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`
- `training/docs/v0210_lateral_yaw_prior_plan-v2.md` (this file)
- `training/tests/test_zmp_walk_config_3d_cmd.py`
- `training/tests/test_reference_library_3d.py`
- `training/tests/test_velocity_cmd_3d_migration.py`
- `training/tests/test_env_config_3d_cmd.py`
- `training/tests/test_sample_velocity_cmd_3d.py`
- `training/tests/test_loc_ref_3d_config.py`
- `training/tests/test_cmd_lateral_velocity_tracking.py`
- `training/tests/test_obs_v8_cmd3d.py`
- `training/tests/test_env_step_lateral_yaw_e2e.py`
- `training/tests/test_config_load_v0210_smoke1.py`

**Modify**:
- `control/zmp/zmp_walk.py` — `ZMPWalkConfig.rotation_radius_m`, `generate()` widened sig, two new footstep branches, `build_library_for_3d_values`
- `control/references/reference_library.py` — `ReferenceLibraryMeta.vy_grid` / `yaw_rate_grid`; save/load updated
- `control/references/runtime_reference_service.py` — `compute_command_integrated_path_state` accepts `velocity_cmd_y_mps`; `lookup_jax_3d` + `_offline_cmd_keys`
- `training/envs/wildrobot_env.py` — `_init_offline_service` 3D branch, `_lookup_offline_window` 3D + heading-frame, layout-allow-list, `_cmd_forward_velocity_track_reward` (+`vy_cmd` kwarg), `_yaw_rate_track_reward`, `_compute_reward_terms`, `_sample_velocity_cmd` branched 3-vec, every scalar-`velocity_cmd` callsite (≈ 10 sites)
- `training/envs/env_info.py` — `velocity_cmd` shape `() → (3,)` in both backends and in `get_expected_shapes`
- `training/configs/training_runtime_config.py` — `LocomotionEnvConfig`: +`min/max_velocity_y`, `max_yaw_rate`, `loc_ref_command_axes_3d`, `loc_ref_offline_command_vy_grid`, `loc_ref_offline_command_yaw_rate_grid`; type bumps for `cmd_deadzone` and `eval_velocity_cmd`; `RewardWeightsConfig.cmd_yaw_rate_track`
- `training/configs/training_config.py` — YAML compat shims for scalar `cmd_deadzone` / `eval_velocity_cmd`
- `policy_contract/spec.py` — `SUPPORTED_LAYOUT_IDS += {"wr_obs_v8_cmd3d"}`; `_validate_observation` v8 branch
- `policy_contract/spec_builder.py` — `_build_obs_layout` v8 branch
- `policy_contract/jax/obs.py` — dispatcher widened to accept v8; v8 branch appends `velocity_cmd[1:]` slot
- `training/core/metrics_registry.py` — add `tracking/velocity_cmd_vy_abs`, `tracking/velocity_cmd_wz_abs`, `tracking/velocity_cmd_norm`, `tracking/yaw_rate_err`
- `training/docs/walking_training.md` — G6 / G7 sections
- `training/CHANGELOG.md` — v0.21.0 entry (after smoke)

---

## Risks & rollback

- **Risk**: Forward G4/G5 regresses when lateral residual scales are bumped. **Mitigation**: P8.2 keeps non-lateral joints unchanged; if regression observed, fall back to 1.0× scales on lateral joints.
- **Risk**: 27-bin library uses too much device memory (S3 napkin: ~50 MB on-device per env, constant across vectorization). **Mitigation**: smoke1 grid is intentionally small; if memory-bound, drop wz_grid to 1 entry first.
- **Risk**: Layout v8 bump breaks downstream eval / inference. **Mitigation**: v1–v7 layouts untouched; old configs continue to work. New `actor_obs_layout_id: wr_obs_v8_cmd3d` is opt-in.
- **Risk**: `cmd_deadzone` / `eval_velocity_cmd` YAML shim mis-classifies a `list` input as scalar. **Mitigation**: shim explicitly checks `isinstance(value, (int, float))` first; lists pass through unchanged.
- **Risk**: `compute_command_integrated_path_state` 3D extension produces drift that fights the policy. **Mitigation**: P3.9 unit test asserts integration matches analytic for pure-vy and pure-wz cases; if smoke1 shows drift fights, gate `r_torso_pos_xy` weight to 0 when `|vy_cmd| > eps OR |wz_cmd| > eps` (S2 fallback already specified).
- **Rollback**: revert by switching smoke YAML back to `actor_obs_layout_id: wr_obs_v7_phase_proprio` + `loc_ref_command_axes_3d: false`. All v0.21 code paths default to vy=wz=0 when the toggle is false and the grids are empty tuples.
