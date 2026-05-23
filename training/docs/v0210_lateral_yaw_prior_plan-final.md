## Changes from v2

This is the round-2 final revision. Every NEW critical issue from round-2 is fixed, and the three "partial" round-1 fixes (C9, C10, C11) are completed. The architectural direction from v2 is preserved (3-axis cmd, vy ±0.13 m/s, wz ±0.25 rad/s, unified ZMP prior mirroring TB).

### NEW issues (round-2) — applied fixes

- **NEW-1 (P3.8 vs P6.2 conflict on line 1218)**: Resolved by moving the line-1218 edit ENTIRELY into P6 (the reward phase, where the semantic change belongs). P3.8 now leaves line 1218 untouched. The single in-place edit at 1218 becomes `vy_err_cmd = lateral_velocity - vy_cmd` in P6.2; the prior `vy_err_cmd = lateral_velocity` is preserved through P3 because `_cmd_forward_velocity_track_reward` still takes a scalar `velocity_cmd` parameter after P3 (the env's caller slices `velocity_cmd[0]` in P6.3 and the helper signature is widened with `vy_cmd: jax.Array = jp.float32(0.0)` in P6.2). This keeps P3's commit semantically green (no behavior change in the reward arithmetic until P6 lands) and removes the "delete then replace" contradiction.
- **NEW-2 (v8 obs-builder kwarg routing)**: Pinned an explicit data-flow rule in P3.8 and a NEW kwarg in P7.2. The env's `_get_obs` continues to pass the full 3-vec `velocity_cmd` to `build_observation_from_components(..., velocity_cmd=velocity_cmd, ...)` AND additionally passes a NEW kwarg `velocity_cmd_lateral_yaw=velocity_cmd[1:]`. `build_observation_from_components` gets a new parameter `velocity_cmd_lateral_yaw: jnp.ndarray | None = None`; v1–v7 ignore it (no read), v8 consumes it. The shared 1-vec slot at obs.py:151 is sliced to `velocity_cmd[..., :1]` so v1–v7 see only vx_cmd byte-identically. Full signature diff is in P7.2.
- **NEW-3 (P5.5 heading-frame correction uses scipy `Rotation`)**: Rewrote P5.5 to use JAX-native quaternion ops. `path_rot` is a wxyz quaternion `jnp.array` (verified at `runtime_reference_service.py:236-238`). Plan now (a) extracts yaw from each quaternion via `_yaw_from_quat_wxyz`, (b) rotates the 2-vec `cmd_xy` by the yaw difference using a 2D rotation matrix. A small helper `_yaw_from_quat_wxyz` is added in `runtime_reference_service.py` (used in P5.5). The reason for not reusing `policy_contract/jax/frames.py::_yaw_from_quat_xyzw` directly: that helper expects xyzw layout, whereas runtime_reference_service emits wxyz. A 5-line wxyz helper is the right granularity (single sub-step P5.4a). The full impl is shown in P5.5.
- **NEW-4 (`training_config.py:549-550` `float()` casts)**: Added an explicit replacement step P3.4a that swaps both `float(env.get("cmd_deadzone", ...))` (line 549) and `float(env.get("eval_velocity_cmd", ...))` (line 550) with a type-aware coercion helper `_load_three_axis(value, default_scalar)`. Diff is shown in the plan. The helper handles three cases: missing → broadcast default; scalar (int/float) → broadcast scalar; list/tuple of length 3 → tuple. P3.4a lands BEFORE P3.4 / P3.5 / P3.6 because those tests load smoke14 (scalar) and smoke1 (list) YAMLs via `load_training_config`, and `float([...])` raises `TypeError` before the dataclass even constructs.
- **NEW-5 (closed-form path_pos math for (vx, vy, wz))**: Replaced the hand-waved "rotating the per-step vector by `Rz(k*theta)` and summing" gloss with the explicit closed-form derivation and concrete JAX code in P3.10. The formula:
  - Let `theta = wz * dt`, `n = round(t/dt)`.
  - `p_n = dt * sum_{k=1..n} Rz(k * theta) @ [vx, vy]`
  - Split the sum into the two cardinal trig sums: `C = sum_{k=1..n} cos(k*theta)`, `S = sum_{k=1..n} sin(k*theta)`. Both have a closed form via the half-angle identity:
    - `ratio = sin(n*theta/2) / sin(theta/2)`
    - `C = ratio * cos((n+1)*theta/2)`
    - `S = ratio * sin((n+1)*theta/2)`
  - Then `p_n.x = dt * (vx * C - vy * S)` and `p_n.y = dt * (vx * S + vy * C)`.
  - Theta-to-zero fallback: when `|sin(theta/2)| < eps`, use `C → n*cos(0) = n`, `S → 0`, giving `p_n = n * dt * [vx, vy]`.
  - Note on parity with the existing code: today's helper uses `x_sum` and `y_sum` to mean exactly `C` and `S` (verified at runtime_reference_service.py:228-229: `x_sum = ratio * cos(0.5*(n+1)*theta)`, `y_sum = ratio * sin(0.5*(n+1)*theta)`). The existing line `path_pos = [vx*dt*x_sum, vx*dt*y_sum, 0]` is therefore `path_pos = [vx*C, vx*S, 0] * dt` — correct for vy=0. The 2D generalization replaces this with `[vx*C - vy*S, vx*S + vy*C, 0] * dt`. Backward compat: when `vy=0`, the new formula collapses to the old one exactly (byte-identical). This is asserted as a regression test in P3.10.
  - This mirrors TB's `integrate_path_state` semantically (TB integrates `lin_vel` rotated by the post-update `path_rot`; the closed form above is the algebraic sum of the same per-step rotation applied to a constant cmd). The WR helper is closed-form (not a scan loop) because it only supports constant cmd over the episode — sufficient for the v0.21.0 reward target.
- **NEW-6 (P0.5 not test-gated)**: Converted P0.5 from a documentation-only audit into a test-gated step. P0.5 now produces a file `training/docs/v0210_velocity_cmd_audit_checklist.md` (auto-generated from `grep` output) AND ships a failing static-analysis test `training/tests/test_velocity_cmd_callsite_audit.py` that asserts the canonical set of `velocity_cmd` callsites is exactly the checklist (line-number-tolerant; matches by function name + count). If a new callsite is added or removed, the test fails and the developer must update both the checklist and the relevant P3 sub-step. The test is intentionally line-number-tolerant (groups by function rather than line) so that small unrelated edits to `wildrobot_env.py` don't break it.

### Partial round-1 fixes — completed

- **C9-partial (callsite fan-out completeness)**: P0.5 checklist now lists the 9 additional callsites the round-2 review identified (1188 docstring; 1216 caller; 2658, 2762, 2816, 2835, 3124, 3269, 3303 kwarg propagations). P3.8 has new sub-bullets covering each:
  - **1188 (docstring)**: text in `_cmd_forward_velocity_track_reward` docstring says "WR currently has no lateral command, so `vy_cmd == 0`". P6.2 updates this string when adding the `vy_cmd` kwarg. P3.8 leaves docstring alone (no functional impact).
  - **1216 (`vx_err_cmd = forward_velocity - velocity_cmd`)**: P6.3 (not P3.8) is the line that touches the call site at 1791 to pass `velocity_cmd[0]` instead of `velocity_cmd`. Line 1216 itself stays scalar because the helper's `velocity_cmd` parameter remains scalar. Added explicit "leave 1216 untouched" note to P3.8.
  - **2658, 2762, 2816, 2835, 3124, 3269, 3303**: each is a `velocity_cmd=velocity_cmd` kwarg pass-through (the variable name on the left is the parameter name; the variable on the right is the 3-vec). Confirmed by grep that these are all internal kwarg propagations — no scalar arithmetic. After P3, the 3-vec flows through them unchanged. Added a single P3.8 bullet "no code change needed; pass-through kwargs widen automatically" with a checklist file confirming each. Each gets a test in P3.8a (a smoke test that builds the env, samples a `velocity_cmd`, and asserts the downstream functions receive the 3-vec without raising — this is implicitly covered by P3.7's `env.step()` test but P3.8a adds an explicit per-call-site shape assertion).
- **C10-partial (closed-form path-state math)**: Covered in NEW-5 above with the explicit derivation and JAX code. Includes a regression test that `(vy=0, wz=0)` reproduces the pre-P3 `path_pos` byte-for-byte and a forward test that `(vy=0.1, wz=0)` after 1 s gives `path_pos[1] ≈ 0.1`.
- **C11-partial (scipy `Rotation.inv()/.apply()` in P5.5)**: Replaced with JAX-native quat-based code. Verified `policy_contract/jax/frames.py` provides `quat_mul`, `axis_angle_to_quat`, `rotate_vec_by_quat`, `_quat_conjugate`, `_yaw_from_quat_xyzw`, `gravity_local_from_quat`, `angvel_heading_local`. None of these accept wxyz quaternions directly. Added a 5-line helper `_yaw_from_quat_wxyz(quat_wxyz)` in `runtime_reference_service.py` mirroring the existing `_yaw_from_quat_xyzw` math (with the w/x/y/z indices shifted). Heading-frame correction in P5.5 now uses: `yaw_path = _yaw_from_quat_wxyz(path_rot_wxyz)`, `yaw_heading = _yaw_from_quat_wxyz(heading_rot_wxyz)`, `delta_yaw = yaw_heading - yaw_path`, then rotate `cmd_xy` by `Rz(-delta_yaw)` via a 2x2 matrix. No scipy syntax remains anywhere in the plan.

## Plan readiness

Round-2 revision complete. All 17 round-1 issues and all 6 round-2 NEW issues addressed. Ready for execution.

---

# Plan: v0.21.0 — Lateral (vy) + Yaw (wz) Prior Support for WildRobot (final)

**Goal**: Extend WR's offline ZMP prior and policy I/O from forward-only (vx) to full 3-DoF locomotion commands (vx, vy, wz), mirroring ToddlerBot's unified ZMP design, so the policy can learn to strafe and turn alongside walking forward.

**Architecture**:
- **Single unified ZMP prior**, not a separate "lateral module". Widen `ZMPWalkGenerator.generate()` (NOT `ZMPWalkConfig` — `command_vx` is and remains a `generate()` parameter; see C1) to accept `command_vy` / `command_yaw_rate`, and add a `ZMPWalkConfig.rotation_radius_m` knob. Mirror TB's `toddlerbot/algorithms/zmp_walk.py:259` translation branch for `(vx, vy)` and TB's `:269-305` arc-radius branch for `wz`.
- **`ReferenceLibrary.lookup` is already 3D** today (`reference_library.py:395-409` accepts `(vx, vy=0.0, yaw_rate=0.0)`) — only the env's `_lookup_offline_window`, the JAX bin-selection, and the metadata grids need work.
- **Command obs widened from 1D → 3D via a NEW slot appended at the end of v8** (`velocity_cmd_lateral_yaw` of size 2 = `[vy_cmd, wz_cmd]`). The legacy 1D `velocity_cmd` slot stays in place for v8 (carries `vx_cmd`), so spec-validator branches for v1–v7 don't move. New layout `wr_obs_v8_cmd3d` registered in `SUPPORTED_LAYOUT_IDS` (`spec.py:11`), `_build_obs_layout` (`spec_builder.py`), `_validate_observation` (`spec.py`), and `build_observation_from_components` dispatcher (`obs.py:46-51`). v8 receives the (vy, wz) payload via a NEW kwarg `velocity_cmd_lateral_yaw`; legacy layouts ignore it.
- **`WildRobotInfo.velocity_cmd` is migrated from shape `()` to `(3,)`** (`env_info.py:107, 222, 285`). All ~10 callsites that index it scalar-style get an explicit migration sub-step (C9).
- **Reward**: KEEP `_cmd_forward_velocity_track_reward` name and shape; add an optional `vy_cmd: jax.Array = jp.float32(0.0)` parameter (default reproduces legacy). Add a separate `_yaw_rate_track_reward` static helper. Wire both from `_compute_reward_terms` with new `RewardWeightsConfig.cmd_yaw_rate_track: float = 0.0` (back-compat).
- **WR-scaled command ranges (LOCKED)**: vy ∈ ±0.13 m/s, wz ∈ ±0.25 rad/s.

**Tech Stack**: Python 3.12, JAX, Brax, MuJoCo MJX, PPO. Tests run via `uv run pytest training/tests/test_xxx.py -q` from repo root `/Users/ygli/projects/wildrobot`.

**Cross-robot scaling reference**:
| Quantity | TB | WR | Ratio |
|---|---|---|---|
| Leg length | 0.2115 m | 0.373 m | 1.76x |
| Hip width / foot_to_com_y | 0.042 m | 0.0536 m | 1.28x |
| cycle_time | 0.72 s | 0.96 s | 1.33x |
| vx_max (training) | 0.10 m/s | 0.26 m/s | 2.6x |
| **vy_max (this plan)** | 0.05 m/s | **0.13 m/s** | 2.6x (preserve vy/vx ratio) |
| **wz_max (this plan)** | 1.0 rad/s | **0.25 rad/s** | 0.25x (preserve foot-arc-per-swing) |
| rotation_radius (this plan) | 0.10 m | **0.10 m** | 1.0x (conservative first value; see C12 note) |

---

## Task Dependencies

| Group | Steps | Parallelizable | Notes |
|-------|-------|----------------|-------|
| G0 | P0.5 (audit + audit test) | n/a | Pure read + 1 test file; standalone commit |
| G1 | P1.1–P1.13 (planner) | No (single file) | Touches `control/zmp/zmp_walk.py` + `control/references/reference_library.py` only |
| G2 | P2.1–P2.5 (lookup verify + extend) | After G1 | Touches `control/references/*.py` only |
| G3 | P3.1–P3.10 (`velocity_cmd` scalar -> 3-vec migration, incl. P3.4a YAML-shim) | No (single env file) | All in `wildrobot_env.py` + `env_info.py` + `training_runtime_config.py` + `training_config.py` |
| G4 | P4.1–P4.7 (sampler) | After G3 (same file) | `wildrobot_env.py::_sample_velocity_cmd`; cannot parallelize with G5 (same file) |
| G5 | P5.1–P5.5 (3D library plumbing) | After G2+G3 (same file) | `wildrobot_env.py::_init_offline_service`, `_lookup_offline_window`; cannot parallelize with G4 (same file) |
| G6 | P6.1–P6.4 (reward — incl. the single line-1218 edit) | After G3+G5 (same file) | `wildrobot_env.py::_cmd_forward_velocity_track_reward` + `_compute_reward_terms` |
| G7 | P7.1–P7.4 (layout + spec + obs.py kwarg) | After G3 | `policy_contract/spec.py` + `spec_builder.py` + `policy_contract/jax/obs.py` — independent of G4–G6 file scope; can run in parallel with G4 once G3 is done IF a separate branch is used |
| G8 | P8.1–P8.2 (smoke YAML + residual scales) | After G7 | Independent YAML |
| G9 | P9.1–P9.3 (integration + e2e tests) | After G6+G7+G8 | Cross-layer; tests in `training/tests/` |
| G10 | P10.1 (CHANGELOG) | After smoke run | Final |

**Conflict callout (C17)**: G3, G4, G5, G6 all touch `wildrobot_env.py`. They must be done in sequence on the same branch. G7 touches different files (`policy_contract/`) and CAN run in parallel only if the developer is comfortable rebasing G3's env changes onto G7's spec changes at the end.

---

## P0.5 — Test-gated `velocity_cmd` callsite audit

Before writing any code, enumerate every `velocity_cmd` callsite in the env / contract / tests AND gate progress on a static-analysis test that asserts the canonical set is unchanged.

### P0.5.1 — Generate audit checklist file

**Command** (generates `training/docs/v0210_velocity_cmd_audit_checklist.md`):
```bash
cd /Users/ygli/projects/wildrobot && grep -nH 'velocity_cmd' \
  training/envs/wildrobot_env.py training/envs/env_info.py \
  training/configs/training_runtime_config.py training/configs/training_config.py \
  policy_contract/jax/obs.py policy_contract/spec.py policy_contract/spec_builder.py \
  control/references/runtime_reference_service.py \
  training/tests/test_*.py tests/*.py \
  > training/docs/v0210_velocity_cmd_audit_checklist.md
```

**Expected canonical entries** (verified by P0.5.2 test):

`training/envs/env_info.py`:
- `107` declared shape `# ()` — MUST become `# (3,)` (P3.2)
- `222` NamedTuple fallback — MUST mirror (P3.2)
- `285` `get_expected_shapes()` mapping — MUST become `(3,)` (P3.2)

`training/envs/wildrobot_env.py` (grouped by function, NOT line, so unrelated edits don't break the audit test):

**`_cmd_forward_velocity_track_reward` (line ~1164)**:
- `1168` parameter declaration — stays scalar (helper signature unchanged in P3; widened to add `vy_cmd: jax.Array = jp.float32(0.0)` in P6.2)
- `1188, 1192` docstring — UPDATE in P6.2 (text-only)
- `1216` `vx_err_cmd = forward_velocity - velocity_cmd` — leave scalar (helper still takes scalar; caller in P6.3 slices `velocity_cmd[0]`)
- `1218` `vy_err_cmd = lateral_velocity` — EDIT IN P6.2 to `vy_err_cmd = lateral_velocity - vy_cmd` (single edit; resolves NEW-1)

**`_lookup_offline_window` (line ~1248)**:
- `1251, 1256, 1261, 1274, 1275, 1286, 1290, 1295, 1306, 1317, 1321` — scalar `velocity_cmd` parameter signatures and docstrings. P5.5 widens param to `(3,)` and changes the lookup to 3D L2 nearest with optional heading-frame correction.

**`_get_obs` (line 1443)**:
- `1443` parameter declaration — stays a jax array; shape changes from `()` to `(3,)` after P3
- `1464` `velocity_cmd=velocity_cmd` kwarg pass to `build_observation_from_components` — P3.8 confirms env passes the full 3-vec. P7.2 adds an ADDITIONAL kwarg `velocity_cmd_lateral_yaw=velocity_cmd[1:]` for v8.

**`_compute_reward_terms` (line ~1603)**:
- `1603` parameter declaration — `(3,)` post-P3
- `1781` comment "(vx, vy) to (velocity_cmd, 0) (WR has no lateral command)" — UPDATE in P6.3 (text-only)
- `1794` `velocity_cmd=velocity_cmd` kwarg to `_cmd_forward_velocity_track_reward` — P6.3 changes to `velocity_cmd=velocity_cmd[0]` AND adds `vy_cmd=velocity_cmd[1]`.

**Reward / metric scalar callsites** (P3.8):
- `1833` comment "`velocity_cmd` is a scalar here; v3" — UPDATE text-only
- `1837` `cmd_active = (jp.abs(velocity_cmd) > 1e-6).astype(jp.float32)` — change to `jp.linalg.norm(velocity_cmd) > 1e-6`
- `2051` `is_standing = jp.abs(velocity_cmd) < 1e-6` — same fix
- `2868-2871` `linear[0] - velocity_cmd` — index `velocity_cmd[0]`
- `2896-2898` `tracking/velocity_cmd_abs` / `_nonzero_frac` — split into per-axis + norm
- `2905-2907` `forward_velocity_cmd_ratio` — index `velocity_cmd[0]`
- `3340/3359/3362/3363/3383/3384/3408` — terminal-metric mirrors of above; same per-axis treatment

**Reset / step plumbing (P3.8 kwarg pass-throughs — no scalar arithmetic, just propagation)**:
- `2287` docstring — text-only
- `2315` `velocity_cmd = self._sample_velocity_cmd(key_vel)` — sampler returns `(3,)` post-P4; broadcasts naturally
- `2350` `velocity_cmd=velocity_cmd` kwarg — pass-through
- `2502, 2510, 2517` docstring — text-only
- `2529` `eval_cmd = jp.float32(self._config.env.eval_velocity_cmd)` — P3.6 rewrites
- `2531` `jp.where(eval_cmd >= 0.0, eval_cmd, wr.velocity_cmd)` — P3.6 rewrites with `jp.all(eval_cmd >= 0.0)` reduction
- `2532` `wr.replace(velocity_cmd=new_cmd.astype(jp.float32))` — shape becomes `(3,)`
- `2539` `_sample_velocity_cmd` signature — P4 rewrites
- `2640` parameter declaration — `(3,)`
- `2656` comment — text-only
- `2658, 2762, 2816, 2835` `velocity_cmd=velocity_cmd` kwarg propagation — pass-through (no scalar arithmetic; just widens to `(3,)` transparently). P3.8 confirms each compiles after the upstream type change.
- `2868-2873, 2896-2898, 2905-2907` — see above
- `2948` `velocity_cmd = wr.velocity_cmd` — `(3,)` post-P3
- `2953, 2956` comments — text-only
- `2957` `_lookup_offline_window(next_step_idx, velocity_cmd=velocity_cmd)` — pass-through; P5.5 also adds `path_rot_wxyz=...` for heading-frame correction
- `2962-2965` `compute_command_integrated_path_state(velocity_cmd_mps=velocity_cmd, yaw_rate_cmd_rps=0.0, ...)` — P3.10 changes: pass `velocity_cmd[0]` as `velocity_cmd_mps`, `velocity_cmd[1]` as `velocity_cmd_y_mps`, `velocity_cmd[2]` as `yaw_rate_cmd_rps`
- `2973` `_lookup_offline_window(prev_step_idx, velocity_cmd=velocity_cmd)` — pass-through; P5.5 also adds `path_rot_wxyz=...`
- `3124` `velocity_cmd=velocity_cmd` kwarg — pass-through
- `3199` comment — text-only
- `3217-3219` `should_resample, resampled_cmd, velocity_cmd` `jp.where` branch — `should_resample` stays scalar bool; both arms are `(3,)`; broadcasts
- `3230` `velocity_cmd=new_velocity_cmd` kwarg — pass-through
- `3269, 3303` kwarg pass-throughs — same as 2762/2816/2835
- `3340, 3359, 3362-3363, 3383-3384, 3408` — terminal-metric mirrors of reset-time scalar callsites; same per-axis treatment as 2868/2896/2905
- `3551` comment — text-only

`training/configs/training_runtime_config.py`:
- `~624-625` `cmd_deadzone: float = 0.0` — type bump to `Tuple[float, float, float] = (0.0, 0.0, 0.0)` (P3.4)
- `~635` `eval_velocity_cmd: float = -1.0` — type bump to `Tuple[float, float, float] = (-1.0, -1.0, -1.0)` (P3.6)
- Add new fields `min_velocity_y, max_velocity_y, max_yaw_rate, loc_ref_command_axes_3d, loc_ref_offline_command_vy_grid, loc_ref_offline_command_yaw_rate_grid` (P4.2, P5.2)

`training/configs/training_config.py`:
- `549` `cmd_deadzone=float(env.get("cmd_deadzone", 0.0))` — REPLACE with `_load_three_axis` (P3.4a)
- `550` `eval_velocity_cmd=float(env.get("eval_velocity_cmd", -1.0))` — REPLACE with `_load_three_axis` (P3.4a)
- Insert `_load_three_axis` helper function (P3.4a)

`policy_contract/jax/obs.py`:
- `12-44` signature of `build_observation_from_components` — ADD kwarg `velocity_cmd_lateral_yaw: jnp.ndarray | None = None` (P7.2)
- `46-51` supported-layout set — ADD `"wr_obs_v8_cmd3d"` (P7.2)
- `151` `jnp.atleast_1d(velocity_cmd).reshape(1)` — change to `jnp.atleast_1d(jnp.asarray(velocity_cmd, dtype=jnp.float32))[..., :1].reshape(1)` (P7.2; v1-v7 byte-identical, v8 takes vx)
- `~193` new `if spec.observation.layout_id == "wr_obs_v8_cmd3d":` branch (P7.2)

`control/references/runtime_reference_service.py`:
- `181-261` `compute_command_integrated_path_state` — add `velocity_cmd_y_mps: float = 0.0` parameter; replace closed-form `path_pos` with 2D form (P3.10)
- ADD helper `_yaw_from_quat_wxyz(quat_wxyz)` (P5.4a)

`tests/`:
- `tests/test_v0201_env_zero_action.py:221` — passes `velocity_cmd_mps=` scalar; STAYS scalar (helper signature keeps positional scalar). No edit.
- `tests/test_runtime_reference_service.py:188, 232, 269` — `velocity_cmd_mps=` scalar; STAYS scalar (new `velocity_cmd_y_mps` defaults to 0.0). No edit.
- `training/tests/test_cmd_velocity_tracking_mode.py:37, 63, 96` — static helper without `vy_cmd`; defaults to 0.0 under P6.2 signature. No edit.

### P0.5.2 — Audit test (failing first time, passing after checklist file generated)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_velocity_cmd_callsite_audit.py`
```python
"""Static-analysis test gating the v0.21.0 velocity_cmd migration.

If you add/remove a velocity_cmd callsite in any of the audited files,
this test will fail. Update the canonical checklist file AND the
relevant P3 sub-step in the v0.21.0 plan, then re-run.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITED_FILES = [
    "training/envs/wildrobot_env.py",
    "training/envs/env_info.py",
    "training/configs/training_runtime_config.py",
    "training/configs/training_config.py",
    "policy_contract/jax/obs.py",
    "policy_contract/spec.py",
    "policy_contract/spec_builder.py",
    "control/references/runtime_reference_service.py",
]

# Per-file canonical callsite counts.  Line-number-tolerant: this only
# locks the *total count* and the set of functions that contain
# velocity_cmd references.  Unrelated edits that don't add/remove a
# callsite will not trip the audit; adding a new scalar `jp.abs(velocity_cmd)`
# call without updating the plan WILL trip it.
EXPECTED_COUNTS = {
    "training/envs/wildrobot_env.py": 78,   # set after first grep
    "training/envs/env_info.py": 4,
    "training/configs/training_runtime_config.py": 6,
    "training/configs/training_config.py": 2,
    "policy_contract/jax/obs.py": 3,
    "policy_contract/spec.py": 8,
    "policy_contract/spec_builder.py": 2,
    "control/references/runtime_reference_service.py": 3,
}


@pytest.mark.parametrize("rel_path,expected", EXPECTED_COUNTS.items())
def test_velocity_cmd_callsite_count_matches_audit(rel_path: str, expected: int) -> None:
    full = REPO_ROOT / rel_path
    text = full.read_text()
    actual = sum(1 for line in text.splitlines() if re.search(r"\bvelocity_cmd\b", line))
    assert actual == expected, (
        f"{rel_path}: expected {expected} velocity_cmd lines, found {actual}. "
        f"Update training/docs/v0210_velocity_cmd_audit_checklist.md and the "
        f"relevant P3 sub-step in the v0.21.0 plan."
    )


def test_audit_checklist_file_exists() -> None:
    p = REPO_ROOT / "training/docs/v0210_velocity_cmd_audit_checklist.md"
    assert p.exists(), (
        "Run: cd /Users/ygli/projects/wildrobot && grep -nH 'velocity_cmd' "
        "training/envs/wildrobot_env.py training/envs/env_info.py ... > "
        "training/docs/v0210_velocity_cmd_audit_checklist.md (see P0.5.1)."
    )
```

**Run**: `cd /Users/ygli/projects/wildrobot && uv run pytest training/tests/test_velocity_cmd_callsite_audit.py -q` — expect 1 failure on the checklist-existence test (file not generated yet).

### P0.5.3 — Generate checklist + verify

Run the grep command from P0.5.1 to produce `training/docs/v0210_velocity_cmd_audit_checklist.md`. Then update `EXPECTED_COUNTS` in `test_velocity_cmd_callsite_audit.py` with the actual counts from the grep output (one-time calibration). Re-run the test — all 9 sub-tests pass.

### P0.5.4 — Commit
```bash
git add training/docs/v0210_velocity_cmd_audit_checklist.md \
        training/tests/test_velocity_cmd_callsite_audit.py
git commit -m "v0.21.0 P0.5: velocity_cmd callsite audit + test gate"
```

> After each P3.x / P5.x / P6.x / P7.x sub-step that adds or removes a `velocity_cmd` line, the developer must re-run the audit test and update `EXPECTED_COUNTS` and the checklist file in the same commit. This is the enforcement mechanism for NEW-6.

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

**Run**: `cd /Users/ygli/projects/wildrobot && uv run pytest training/tests/test_zmp_walk_config_3d_cmd.py -q` — expect 2 failures.

### P1.2 — Impl: add `rotation_radius_m` field

Add `rotation_radius_m: float = 0.10` to `ZMPWalkConfig` (near `default_stance_width_m`, line 135) with a derivation comment:
```python
# Pure-rotation arc radius (m), mirrors TB
# toddlerbot/algorithms/zmp_walk.py:220 default 0.1.  WR-scaled value
# under foot_to_com_y proxy = hip_lateral_offset_m = 0.0536:
#   r_WR = 0.1 * (0.0536 / 0.042) ~= 0.128
# Rounded to 0.10 as a conservative first-smoke value; revisit after
# G7 yaw tracking results.
rotation_radius_m: float = 0.10
```

**Verify**: rerun P1.1 — passes.

### P1.3 — `generate()` accepts `command_vy`, `command_yaw_rate` (failing test)

**Append**:
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

**Append**:
```python
def test_planner_produces_nonzero_lateral_foot_displacement_for_vy() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0001, command_vy=0.05)
    foot_y = traj.left_foot_pos[:, 1]
    assert (foot_y.max() - foot_y.min()) > 0.005
```

### P1.6 — Impl: combined-translation footstep block (mirror TB:259-267)

Refactor `_generate_at_step_length` (line 682) so the footstep-x advance currently using `i * sl` for left and `(2 * i + 1) * sl` for right ALSO carries a lateral `(±vy * stride)` term mirroring TB:
```python
# TB zmp_walk.py:259-267 — combined translation.
# stride = command[:2] * total_time / (2 * num_cycles - 1)
stride_xy = np.array([command_vx, command_vy], dtype=np.float64) \
            * cfg.total_plan_time_s / (2 * n_cycles - 1)
left_xy  = left_xy_init  + 2 * i * stride_xy
right_xy = right_xy_init + (2 * i + 1) * stride_xy
```
Apply lateral component to `left_world[idx, 1]`/`right_world[idx, 1]` (in addition to the existing `±lat` stance offset). Compute the COM y-target the LQR sees from the new foot midpoint.

**Verify**: P1.5 passes. P1.3 still passes.

### P1.7 — Pure-rotation branch: failing test

**Append**:
```python
def test_planner_yaws_pelvis_for_nonzero_yaw_rate() -> None:
    gen = ZMPWalkGenerator()
    traj = gen.generate(command_vx=0.0, command_vy=0.0, command_yaw_rate=0.10)
    yaw = traj.pelvis_rpy[:, 2]
    assert abs(yaw[-1] - yaw[0]) > 1e-3
```

### P1.8 — Impl: pure-rotation arc-radius branch (mirror TB:269-305)

In `_generate_at_step_length`, branch on `(|command_vx| < eps and |command_vy| < eps and |command_yaw_rate| > eps)`. Mirror TB exactly:
```python
stride = command_yaw_rate * cfg.total_plan_time_s / (2 * n_cycles - 1)
r = np.sign(command_yaw_rate) * cfg.rotation_radius_m
foot_to_com_y = cfg.hip_lateral_offset_m  # WR proxy for TB's foot_to_com_y
for i in range(n_cycles):
    angle_left  = i * 2 * stride
    angle_right = (i * 2 + 1) * stride
    left_xy = np.array([
        r * np.sin(angle_left)  - foot_to_com_y * np.sin(angle_left),
        r * (1.0 - np.cos(angle_left))  + foot_to_com_y * np.cos(angle_left),
    ])
    right_xy = np.array([
        r * np.sin(angle_right) + foot_to_com_y * np.sin(angle_right),
        r * (1.0 - np.cos(angle_right)) - foot_to_com_y * np.cos(angle_right),
    ])
```
Accumulate `pelvis_rpy[:, 2]` linearly so the policy sees `yaw_rate * t`.

**Verify**: P1.7 passes. P1.5, P1.3 still pass.

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
    assert len(lib) == 9
    assert (0.20, 0.0, 0.10) in lib
```

### P1.11 — Impl: `build_library_for_3d_values`

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
        vy_grid=tuple(vy_values),
        yaw_rate_grid=tuple(yaw_rate_values),
    )
    return ReferenceLibrary(trajectories=trajectories, meta=meta)
```

**Verify**: P1.10 passes.

### P1.12 — Regression: forward-only library still builds

```python
def test_build_library_for_3d_values_with_only_vx_matches_legacy_shape() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(vx_values=[0.20])
    assert len(lib) == 1
    traj = next(iter(lib._entries.values()))
    assert traj.command_vy == 0.0 and traj.command_yaw_rate == 0.0
```

### P1.13 — Commit
```bash
git add control/zmp/zmp_walk.py training/tests/test_zmp_walk_config_3d_cmd.py
git commit -m "v0.21.0 P1: ZMP planner accepts (vy, wz) and 3D library factory"
```

---

## P2 — Verify existing 3D `ReferenceLibrary.lookup` + add metadata grids

**File**: `/Users/ygli/projects/wildrobot/control/references/reference_library.py`

### P2.1 — Regression: `lookup` already accepts 3D key

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_reference_library_3d.py`
```python
from __future__ import annotations
import pytest
from control.zmp.zmp_walk import ZMPWalkGenerator


def test_reference_library_lookup_already_picks_nearest_3d_key() -> None:
    gen = ZMPWalkGenerator()
    lib = gen.build_library_for_3d_values(
        vx_values=[0.20],
        vy_values=[-0.05, 0.0, 0.05],
        yaw_rate_values=[-0.10, 0.0, 0.10],
    )
    picked = lib.lookup(vx=0.20, vy=0.04, yaw_rate=-0.08)
    assert picked.command_vy == pytest.approx(0.05)
    assert picked.command_yaw_rate == pytest.approx(-0.10)
```

**Run**: PASSES today (proves C7 from review-1).

### P2.2 — `ReferenceLibraryMeta` accepts `vy_grid` / `yaw_rate_grid` (failing test)

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

### P2.3 — Impl: add `vy_grid` / `yaw_rate_grid` to `ReferenceLibraryMeta`

In `reference_library.py:341`:
```python
vy_grid: Tuple[float, ...] = field(default_factory=tuple)
yaw_rate_grid: Tuple[float, ...] = field(default_factory=tuple)
```
Update `save()` / `load()` (line 565+) to include in `metadata.json` (tuple ↔ list).

**Verify**: P2.2 passes; existing library tests still pass.

### P2.4 — `build_library_for_3d_values` populates meta grids (failing test)

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

### P2.5 — Impl already done in P1.11; commit
```bash
git add control/references/reference_library.py training/tests/test_reference_library_3d.py
git commit -m "v0.21.0 P2: ReferenceLibraryMeta carries vy/yaw_rate grids; verify lookup already 3D"
```

---

## P3 — Migrate `velocity_cmd` from scalar to 3-vector

**Files**: `env_info.py`, `wildrobot_env.py`, `training_runtime_config.py`, `training_config.py`, `runtime_reference_service.py`.

> Foundational data shape change. MUST land in one commit so no half-state exists. All sub-steps share one commit at P3.10.

> **NEW-1 resolution**: P3 leaves `wildrobot_env.py:1218` (`vy_err_cmd = lateral_velocity`) UNTOUCHED. P3 also leaves the helper signature `_cmd_forward_velocity_track_reward(velocity_cmd: jax.Array, ...)` taking a scalar `velocity_cmd`. The caller at line 1794 still passes `velocity_cmd=velocity_cmd` (the 3-vec) — this would normally be a shape mismatch, BUT P3.7's failing test will catch it and P3.8 changes the call site to `velocity_cmd=velocity_cmd[0]` (forward slice) AND the docstring at line 1781 is updated. The actual semantic change (`vy_err_cmd = lateral_velocity - vy_cmd`) is deferred to P6.2 where the `vy_cmd` kwarg is introduced. This is the contradiction-free path.

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

### P3.2 — Impl: bump `velocity_cmd` shape in env_info

In `env_info.py:107`, change comment `# ()` → `# (3,)`. Same at line 222 (NamedTuple fallback). In `get_expected_shapes()` line 285 set `"velocity_cmd": (3,)`.

**Verify**: P3.1 passes.

### P3.3 — `EnvConfig.cmd_deadzone` type bump (failing test)

```python
from training.configs.training_runtime_config import LocomotionEnvConfig


def test_env_config_cmd_deadzone_default_is_three_tuple() -> None:
    cfg = LocomotionEnvConfig()
    assert tuple(cfg.cmd_deadzone) == (0.0, 0.0, 0.0)
```

### P3.4 — Impl: cmd_deadzone tuple migration

In `training_runtime_config.py:625` change `cmd_deadzone: float = 0.0` to:
```python
cmd_deadzone: Tuple[float, float, float] = (0.0, 0.0, 0.0)
```

**Verify**: P3.3 passes. (YAML loading still broken — addressed in P3.4a.)

### P3.4a — REPLACE `float(...)` casts at `training_config.py:549-550` with `_load_three_axis` (NEW-4)

**Why this lands before P3.5/P3.6**: smoke14 sets `cmd_deadzone: 0.0666666667` (scalar) and `eval_velocity_cmd: 0.26` (scalar); the soon-to-arrive smoke1 sets `cmd_deadzone: [0.02, 0.02, 0.05]` (list) and `eval_velocity_cmd: [0.18, 0.04, 0.10]` (list). Both must parse via `load_training_config`. The current `float(env.get(...))` casts at lines 549-550 raise `TypeError: float() argument must be a string or a real number, not 'list'` BEFORE any dataclass field validation runs.

**Failing test** (append to `test_velocity_cmd_3d_migration.py`):
```python
def test_load_three_axis_accepts_scalar_and_broadcasts() -> None:
    from training.configs.training_config import _load_three_axis
    assert _load_three_axis(0.05, default_scalar=0.0) == (0.05, 0.05, 0.05)


def test_load_three_axis_accepts_three_tuple() -> None:
    from training.configs.training_config import _load_three_axis
    assert _load_three_axis([0.02, 0.03, 0.04], default_scalar=0.0) == (0.02, 0.03, 0.04)


def test_load_three_axis_uses_default_on_none() -> None:
    from training.configs.training_config import _load_three_axis
    assert _load_three_axis(None, default_scalar=0.0) == (0.0, 0.0, 0.0)


def test_load_three_axis_rejects_two_tuple() -> None:
    from training.configs.training_config import _load_three_axis
    import pytest
    with pytest.raises(ValueError, match="length 3"):
        _load_three_axis([0.02, 0.03], default_scalar=0.0)
```

**Impl**: Add to `training/configs/training_config.py` (above `load_training_config_from_dict` at line ~880):
```python
def _load_three_axis(
    value: Any,
    *,
    default_scalar: float,
) -> Tuple[float, float, float]:
    """Coerce a scalar OR length-3 list/tuple to a (float, float, float).

    Handles three legitimate YAML shapes for v0.21.0 three-axis fields:
      * missing / None   -> (default_scalar, default_scalar, default_scalar)
      * scalar (int/float) -> broadcast to (s, s, s)
      * length-3 list/tuple -> tuple(map(float, value))

    Any other shape raises ValueError with the offending value.
    """
    if value is None:
        return (default_scalar, default_scalar, default_scalar)
    if isinstance(value, (int, float)):
        s = float(value)
        return (s, s, s)
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(
                f"three-axis field must have length 3 (got length {len(value)}: {value!r})"
            )
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError(
        f"three-axis field must be a number or a length-3 list/tuple "
        f"(got {type(value).__name__}: {value!r})"
    )
```

Replace lines 549-550 of `training_config.py`:
```python
# BEFORE:
#   cmd_deadzone=float(env.get("cmd_deadzone", 0.0)),
#   eval_velocity_cmd=float(env.get("eval_velocity_cmd", -1.0)),
# AFTER:
cmd_deadzone=_load_three_axis(env.get("cmd_deadzone"), default_scalar=0.0),
eval_velocity_cmd=_load_three_axis(env.get("eval_velocity_cmd"), default_scalar=-1.0),
```

**Verify**: All four `_load_three_axis` unit tests pass. Add an integration check:
```python
def test_smoke14_yaml_still_loads_after_three_axis_shim() -> None:
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    # smoke14 sets cmd_deadzone: 0.0666666667 (scalar) - broadcasts.
    assert tuple(cfg.env.cmd_deadzone) == pytest.approx(
        (0.0666666667, 0.0666666667, 0.0666666667), abs=1e-7
    )
    # smoke14 sets eval_velocity_cmd: 0.26 (scalar) - broadcasts.
    assert tuple(cfg.env.eval_velocity_cmd) == pytest.approx((0.26, 0.26, 0.26))
```

This also addresses **S1** (smoke14 still loads).

### P3.5 — `EnvConfig.eval_velocity_cmd` type bump + sentinel (failing test)

```python
def test_env_config_eval_velocity_cmd_default_is_three_tuple_sentinel() -> None:
    cfg = LocomotionEnvConfig()
    assert tuple(cfg.eval_velocity_cmd) == (-1.0, -1.0, -1.0)
```

### P3.6 — Impl: eval_velocity_cmd tuple migration + sentinel rewrite

In `training_runtime_config.py:635` change `eval_velocity_cmd: float = -1.0` to:
```python
eval_velocity_cmd: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
```

In `wildrobot_env.py:2529-2532` replace the sentinel comparator:
```python
# BEFORE:
#   eval_cmd = jp.float32(self._config.env.eval_velocity_cmd)
#   new_cmd = jp.where(eval_cmd >= jp.float32(0.0), eval_cmd, wr.velocity_cmd)
# AFTER:
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
    assert tuple(cfg.env.eval_velocity_cmd) == pytest.approx((0.26, 0.26, 0.26))
```

> Note: smoke14's `eval_velocity_cmd: 0.26` broadcasts to (0.26, 0.26, 0.26). The legacy semantic was "set vx_eval = 0.26"; the new semantic is "set vx_eval = vy_eval = wz_eval = 0.26". This is BEHAVIORALLY DIFFERENT for smoke14 eval rollouts (smoke14 will now eval at vy=0.26, wz=0.26 which are out-of-range). Mitigation: smoke14 baseline already passed; we are not re-running it. The compat shim is for parsing safety only. Future configs should write the explicit 3-tuple.

**Verify**: P3.5 + new test pass.

### P3.7 — Env step fanout: scalar usages of `velocity_cmd` (failing test)

```python
def test_env_step_does_not_raise_on_three_vec_velocity_cmd() -> None:
    import jax
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    from training.envs.env_info import WR_INFO_KEY
    assert state.info[WR_INFO_KEY].velocity_cmd.shape == (3,)
    _ = env.step(state, jp.zeros(env.action_size))
```

**Run**: expect failure (`jp.abs(velocity_cmd) > 1e-6` broadcasts to `(3,)`; `is_standing` downstream shape mismatches; `_cmd_forward_velocity_track_reward` receives a `(3,)` where it expects scalar).

### P3.8 — Impl: rewrite each scalar callsite

Apply ALL of the following from the P0.5 checklist:

**Scalar arithmetic fixes**:
- `wildrobot_env.py:1837` — `cmd_active = (jp.linalg.norm(velocity_cmd) > jp.float32(1e-6)).astype(jp.float32)`.
- `wildrobot_env.py:2051` — `is_standing = jp.linalg.norm(velocity_cmd) < jp.float32(1e-6)`.
- `wildrobot_env.py:2867-2873` (reset metrics) — index `velocity_cmd[0]` for the forward-only diagnostics.
- `wildrobot_env.py:2896-2898` — split into per-axis metrics:
  - `tracking/velocity_cmd_vx_abs = jp.abs(velocity_cmd[0])`
  - `tracking/velocity_cmd_vy_abs = jp.abs(velocity_cmd[1])`
  - `tracking/velocity_cmd_wz_abs = jp.abs(velocity_cmd[2])`
  - `tracking/velocity_cmd_norm = jp.linalg.norm(velocity_cmd)`
  - `tracking/velocity_cmd_nonzero_frac = (jp.linalg.norm(velocity_cmd) > 1e-6).astype(jp.float32)`
- `wildrobot_env.py:2905-2907` — index `velocity_cmd[0]` in the ratio.
- `wildrobot_env.py:3340/3359/3362-3363/3383-3384/3408` — same pattern as reset (index `velocity_cmd[0]` for forward-only terms; add per-axis variants for vy/wz).

**Call-site widening (NEW-2 explicit data-flow rule)**:
> RULE: `_get_obs` and all downstream observation builders CONTINUE to receive the full 3-vec `wr.velocity_cmd`. ONLY the reward / metric arithmetic paths use indexed slices.
- `wildrobot_env.py:1464` — `velocity_cmd=velocity_cmd` STAYS as-is (env passes the 3-vec to `build_observation_from_components`). P7.2 also adds an explicit `velocity_cmd_lateral_yaw=velocity_cmd[1:]` kwarg here so v8 has a typed channel for (vy, wz).

**Kwarg pass-throughs (no scalar arithmetic — verified pass-through, NO code change)**:
- `2350, 2658, 2762, 2816, 2835, 2957, 2973, 3124, 3230, 3269, 3303` — `velocity_cmd=velocity_cmd` propagations. Each receives a `(3,)` post-P3 and forwards it unchanged. No edit needed; verified by P3.8a per-callsite shape test below.

**Explicitly LEFT UNTOUCHED in P3 (resolves NEW-1)**:
- `wildrobot_env.py:1216` `vx_err_cmd = forward_velocity - velocity_cmd` — stays scalar. The helper's `velocity_cmd` parameter at line 1168 stays scalar. The caller at line 1794 changes in P6.3 to pass `velocity_cmd[0]` and `vy_cmd=velocity_cmd[1]`.
- `wildrobot_env.py:1218` `vy_err_cmd = lateral_velocity` — stays unchanged. Edited in P6.2 to `vy_err_cmd = lateral_velocity - vy_cmd` after `vy_cmd` parameter is added.
- `wildrobot_env.py:1188, 1192, 1781` docstrings/comments — edited in P6.2/P6.3 alongside the semantic change.

Also update `training/core/metrics_registry.py` to add the new per-axis tracking metrics (`tracking/velocity_cmd_vy_abs`, `tracking/velocity_cmd_wz_abs`, `tracking/velocity_cmd_norm`) with `Reducer.MEAN`.

**Verify**: P3.7 passes; pre-existing tests `test_v0201_env_zero_action.py`, `test_cmd_velocity_tracking_mode.py` still pass.

### P3.8a — Per-callsite pass-through shape test (NEW-6 reinforcement)

**Append** to `test_velocity_cmd_3d_migration.py`:
```python
def test_pass_through_callsites_receive_three_vec() -> None:
    """Verify each pass-through callsite forwards velocity_cmd as (3,).

    Lines covered: 2658, 2762, 2816, 2835, 3124, 3269, 3303 (kwarg
    propagations) and 2957, 2973 (lookup callers).  Uses a fake
    receiver that captures the shape, monkey-patching the targets.
    """
    import jax
    from training.envs.wildrobot_env import WildRobotEnv
    from training.envs.env_info import WR_INFO_KEY
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    new_cmd = jp.array([0.1, 0.05, 0.02], dtype=jp.float32)
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(velocity_cmd=new_cmd)
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    # If any of the pass-through callsites bottlenecks down to scalar,
    # this will raise a shape mismatch deep inside step().
    state2 = env.step(state, jp.zeros(env.action_size))
    assert state2.info[WR_INFO_KEY].velocity_cmd.shape == (3,)
```

**Verify**: passes.

### P3.9 — `compute_command_integrated_path_state` 3D extension (failing test)

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
    assert float(out["path_pos"][1]) == pytest.approx(0.10, abs=0.01)
    assert abs(float(out["path_pos"][0])) < 0.01


def test_path_state_integrates_yaw_rate_for_pure_turn() -> None:
    import math
    from control.references.runtime_reference_service import RuntimeReferenceService
    out = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=1.0,
        velocity_cmd_mps=0.0,
        velocity_cmd_y_mps=0.0,
        yaw_rate_cmd_rps=0.10,
        dt_s=0.02,
    )
    expected_qz = math.sin(0.10 / 2.0)
    assert float(out["path_rot"][3]) == pytest.approx(expected_qz, abs=0.005)


def test_path_state_2d_collapses_to_1d_when_vy_zero() -> None:
    """C10 backward-compat regression: with vy=0, wz=0, new closed form
    must be byte-identical to the pre-P3 vx-only form."""
    from control.references.runtime_reference_service import RuntimeReferenceService
    out_new = RuntimeReferenceService.compute_command_integrated_path_state(
        t_since_reset_s=1.0,
        velocity_cmd_mps=0.10,
        velocity_cmd_y_mps=0.0,
        yaw_rate_cmd_rps=0.10,
        dt_s=0.02,
    )
    # path_pos[1] must match the legacy closed form vx*dt*y_sum
    import math, jax.numpy as jp
    dt, vx, wz = 0.02, 0.10, 0.10
    n = round(1.0 / dt)
    theta = wz * dt
    sin_half = math.sin(theta / 2)
    ratio = math.sin(n * theta / 2) / sin_half
    legacy_y = vx * dt * ratio * math.sin((n + 1) * theta / 2)
    assert float(out_new["path_pos"][1]) == pytest.approx(legacy_y, abs=1e-6)
```

### P3.10 — Impl: extend `compute_command_integrated_path_state` (NEW-5 explicit math) + commit

In `runtime_reference_service.py:181-261` extend the staticmethod signature:
```python
@staticmethod
def compute_command_integrated_path_state(
    *,
    t_since_reset_s,
    velocity_cmd_mps,
    velocity_cmd_y_mps=0.0,           # NEW (NEW-5)
    yaw_rate_cmd_rps=0.0,
    dt_s=0.02,
    default_root_pos_xyz=None,
) -> dict:
```

**Replace** the `path_pos` block (lines 216-231) with the explicit 2D closed-form:

```python
import jax.numpy as jnp

t = jnp.asarray(t_since_reset_s, dtype=jnp.float32)
vx = jnp.asarray(velocity_cmd_mps, dtype=jnp.float32)
vy = jnp.asarray(velocity_cmd_y_mps, dtype=jnp.float32)
yaw_rate = jnp.asarray(yaw_rate_cmd_rps, dtype=jnp.float32)
dt = jnp.maximum(jnp.asarray(dt_s, dtype=jnp.float32), jnp.float32(1e-8))

n_steps = jnp.maximum(jnp.rint(t / dt).astype(jnp.int32), 0)
n = n_steps.astype(jnp.float32)
theta = yaw_rate * dt

# Closed-form discrete 2D path integration:
#   p_n = dt * sum_{k=1..n} Rz(k*theta) @ [vx, vy]
# where Rz(a) @ [vx, vy] = [vx*cos(a) - vy*sin(a),
#                           vx*sin(a) + vy*cos(a)].
# Therefore
#   p_n.x = dt * (vx * C(n) - vy * S(n))
#   p_n.y = dt * (vx * S(n) + vy * C(n))
# with
#   C(n) = sum_{k=1..n} cos(k*theta)
#   S(n) = sum_{k=1..n} sin(k*theta)
# Half-angle closed form (theta != 0):
#   ratio = sin(n*theta/2) / sin(theta/2)
#   C(n)  = ratio * cos((n+1)*theta/2)
#   S(n)  = ratio * sin((n+1)*theta/2)
# Theta -> 0 fallback: C(n) -> n, S(n) -> 0; gives p_n = n * dt * [vx, vy].
half_theta = 0.5 * theta
sin_half = jnp.sin(half_theta)
sin_half_safe = jnp.where(
    jnp.abs(sin_half) > jnp.float32(1e-6),
    sin_half,
    jnp.float32(1.0),
)
ratio_closed = jnp.sin(0.5 * n * theta) / sin_half_safe
ratio = jnp.where(jnp.abs(sin_half) > jnp.float32(1e-6), ratio_closed, n)
C = ratio * jnp.cos(0.5 * (n + 1.0) * theta)
S_ = ratio * jnp.sin(0.5 * (n + 1.0) * theta)
# theta -> 0 fallback for S explicitly (cos((n+1)*0/2)=1, sin(...)=0):
S_ = jnp.where(jnp.abs(sin_half) > jnp.float32(1e-6), S_, jnp.float32(0.0))

px = dt * (vx * C - vy * S_)
py = dt * (vx * S_ + vy * C)
path_pos = jnp.asarray([px, py, jnp.float32(0.0)], dtype=jnp.float32)

# yaw, path_rot (unchanged from current impl):
yaw = n * theta
half_yaw = 0.5 * yaw
yaw_cos = jnp.cos(yaw)
yaw_sin = jnp.sin(yaw)
path_rot = jnp.asarray(
    [jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)], dtype=jnp.float32
)  # wxyz
default_root = (
    jnp.asarray(default_root_pos_xyz, dtype=jnp.float32)
    if default_root_pos_xyz is not None
    else jnp.zeros((3,), dtype=jnp.float32)
)
default_root_rotated = jnp.asarray(
    [
        yaw_cos * default_root[0] - yaw_sin * default_root[1],
        yaw_sin * default_root[0] + yaw_cos * default_root[1],
        default_root[2],
    ],
    dtype=jnp.float32,
)
torso_pos = (default_root_rotated + path_pos).astype(jnp.float32)
# lin_vel reports the instantaneous *world-frame* lin vel from the
# commanded body-frame cmd at the current heading.  For consistency
# with TB integrate_path_state where lin_vel is reported pre-rotation,
# we report the body-frame lin_vel (matches TB's
# `lin_vel = self.get_vel(command)[0]` before the apply).
lin_vel = jnp.asarray([vx, vy, 0.0], dtype=jnp.float32)
ang_vel = jnp.asarray([0.0, 0.0, yaw_rate], dtype=jnp.float32)
return {
    "path_pos": path_pos,
    "path_rot": path_rot,
    "torso_pos": torso_pos,
    "lin_vel": lin_vel,
    "ang_vel": ang_vel,
}
```

**Parity with TB**: TB's `integrate_path_state` (motion_ref.py:200-235) is incremental: `path_rot = curr_rot * delta_rot`; `path_pos = path_pos + path_rot.apply(lin_vel) * dt`. Over `n` steps with constant cmd from rest, that sum is exactly `dt * sum_{k=1..n} Rz(k*theta) @ lin_vel` — the closed form above. Verified algebraically; verified numerically by the `test_path_state_2d_collapses_to_1d_when_vy_zero` regression test.

Update env caller at `wildrobot_env.py:2962-2965`:
```python
path = self._reference_service.compute_command_integrated_path_state(
    t_since_reset_s=t,
    velocity_cmd_mps=velocity_cmd[0],     # NEW: index
    velocity_cmd_y_mps=velocity_cmd[1],   # NEW
    yaw_rate_cmd_rps=velocity_cmd[2],     # NEW (was hard-coded 0.0)
    dt_s=self._control_dt_s,
)
```

Existing test callers (`tests/test_runtime_reference_service.py:188/232/269`) pass `velocity_cmd_mps=...` positional/keyword scalar — defaults supply zero vy and zero wz. NO edit needed.

**Verify**: P3.9 (all three tests) passes. Re-run `tests/test_runtime_reference_service.py` — passes.

**Commit**:
```bash
git add training/envs/env_info.py training/envs/wildrobot_env.py \
        training/configs/training_runtime_config.py training/configs/training_config.py \
        control/references/runtime_reference_service.py \
        training/core/metrics_registry.py \
        training/tests/test_velocity_cmd_3d_migration.py \
        training/docs/v0210_velocity_cmd_audit_checklist.md \
        training/tests/test_velocity_cmd_callsite_audit.py
git commit -m "v0.21.0 P3: WildRobotInfo.velocity_cmd -> (3,); path_state integrates (vy, wz)"
```

> Remember to update `EXPECTED_COUNTS` in `test_velocity_cmd_callsite_audit.py` to reflect any newly added per-axis metric callsites BEFORE committing.

---

## P4 — Branched 3D command sampler (mirror TB exactly)

**File**: `wildrobot_env.py` (around line 2539); `training_runtime_config.py` (around line 108).

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

In `training_runtime_config.py` near line 108:
```python
min_velocity_y: float = 0.0
max_velocity_y: float = 0.0
max_yaw_rate: float = 0.0
```

Also extend the `load_training_config` env-section block (training_config.py around lines 350-450 — wherever `min_velocity`/`max_velocity` are read) to read `min_velocity_y`, `max_velocity_y`, `max_yaw_rate` as `float(env.get(..., 0.0))`.

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

### P4.4 — Impl: sampler returns 3-vec zero-cmd path only

```python
def _sample_velocity_cmd(self, rng: jax.Array) -> jax.Array:
    return jp.zeros((3,), dtype=jp.float32)
```

**Verify**: P4.3 passes.

### P4.5 — Walk branch ellipse (failing test, then impl)

```python
def test_walk_branch_samples_within_ellipse() -> None:
    env = _env_3d()
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    assert (cmds[:, 0] >= 0.0).all()
    assert (cmds[:, 0] <= env._config.env.max_velocity + 1e-6).all()
    assert (jp.abs(cmds[:, 1]) <= env._config.env.max_velocity_y + 1e-6).all()
```

**Impl** (mirror TB `walk_env.py:256-280` literally):
```python
def _sample_walk_command(self, rng_3, rng_4) -> jax.Array:
    theta = jax.random.uniform(rng_3, (), minval=0.0, maxval=2 * jp.pi)
    x_hi = jp.float32(self._config.env.max_velocity)
    x_lo = jp.float32(self._config.env.min_velocity)
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

**Verify**: P4.5 passes when smoke1 sets `cmd_zero_chance: 0.0`, `cmd_turn_chance: 0.0` for this test only — use a fixture that monkey-patches the env config.

### P4.6 — Turn branch (failing test, then impl)

```python
def test_turn_branch_samples_yaw_within_range() -> None:
    """Build a temporary config with cmd_turn_chance=1.0,
    cmd_zero_chance=0.0 (turn-only), sample 4096 cmds, assert all
    have vx=vy=0 and |wz| <= max_yaw_rate."""
    import dataclasses
    env = _env_3d()
    env._config.env = dataclasses.replace(
        env._config.env, cmd_zero_chance=0.0, cmd_turn_chance=1.0,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    assert (jp.abs(cmds[:, 0]) < 1e-6).all()
    assert (jp.abs(cmds[:, 1]) < 1e-6).all()
    assert (jp.abs(cmds[:, 2]) <= env._config.env.max_yaw_rate + 1e-6).all()
```

**Impl** (mirror TB:282-305):
```python
def _sample_turn_command(self, rng_5, rng_6) -> jax.Array:
    sign_bit = jax.random.uniform(rng_5, ()) < 0.5
    wz_mag = jax.random.uniform(
        rng_6, (),
        minval=self._config.env.cmd_deadzone[2],
        maxval=self._config.env.max_yaw_rate,
    )
    wz = jp.where(sign_bit, wz_mag, -wz_mag)
    return jp.stack([jp.float32(0.0), jp.float32(0.0), wz])
```

**Verify**: P4.6 passes.

### P4.7 — Branch selector (failing test, impl, commit)

```python
def test_branch_marginal_distribution() -> None:
    env = _env_3d()  # smoke1 YAML sets cmd_zero_chance=0.2, cmd_turn_chance=0.2
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    cmds = jax.vmap(env._sample_velocity_cmd)(keys)
    is_zero = jp.all(jp.abs(cmds) < 1e-6, axis=-1)
    is_turn = (
        (jp.abs(cmds[:, 2]) > 1e-6)
        & (jp.abs(cmds[:, 0]) < 1e-6)
        & (jp.abs(cmds[:, 1]) < 1e-6)
    )
    is_walk = (jp.abs(cmds[:, 0]) > 1e-6) | (jp.abs(cmds[:, 1]) > 1e-6)
    assert abs(float(is_zero.mean()) - 0.20) < 0.03
    assert abs(float(is_turn.mean()) - 0.20) < 0.03
    assert abs(float(is_walk.mean()) - 0.60) < 0.03
```

**Impl** (glue mirroring TB `walk_env.py:307-316`):
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
        training/configs/training_config.py \
        training/tests/test_env_config_3d_cmd.py training/tests/test_sample_velocity_cmd_3d.py
git commit -m "v0.21.0 P4: branched (zero/turn/walk) 3D cmd sampler mirroring TB"
```

---

## P5 — 3D reference library plumbing in env

**File**: `wildrobot_env.py` (`_init_offline_service` line 764; `_lookup_offline_window` line 1248).
**File**: `runtime_reference_service.py` (add `_offline_cmd_keys` + `_yaw_from_quat_wxyz`).
**File**: `training_runtime_config.py` (`loc_ref_command_axes_3d`, vy/yaw_rate grids).

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

### P5.2 — Impl: add three fields

In `training_runtime_config.py` near line 334:
```python
loc_ref_command_axes_3d: bool = False
loc_ref_offline_command_vy_grid: Tuple[float, ...] = field(default_factory=tuple)
loc_ref_offline_command_yaw_rate_grid: Tuple[float, ...] = field(default_factory=tuple)
```
Also extend `load_training_config` env-block to read these (`bool(env.get(...))` and `tuple(env.get(..., []) or [])`).

> Do NOT widen the `loc_ref_version` validator at `wildrobot_env.py:218-223` — keep `v3_offline_library`. The new toggle gates 3D behavior.

**Verify**: P5.1 passes.

### P5.3 — `_init_offline_service` builds 3D grid (failing test)

```python
def test_init_offline_service_builds_3d_grid_when_toggled() -> None:
    from training.envs.wildrobot_env import WildRobotEnv
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    env = WildRobotEnv(cfg)
    assert env._offline_cmd_keys.shape == (27, 3)
```

### P5.4 — Impl: `_init_offline_service` 3D branch

In `wildrobot_env.py:820-848` add a branch:
```python
if cmd_conditioned and bool(getattr(self._config.env, "loc_ref_command_axes_3d", False)):
    vx_values = list(vx_grid)
    vy_values = list(self._config.env.loc_ref_offline_command_vy_grid) or [0.0]
    yaw_rate_values = list(self._config.env.loc_ref_offline_command_yaw_rate_grid) or [0.0]
    assert len(vx_values) > 0 and len(vy_values) > 0 and len(yaw_rate_values) > 0
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
    # Resolves S6: explicit count assertion catches typo'd grid var names.
    assert len(cmd_keys) == len(vx_values) * len(vy_values) * len(yaw_rate_values)
else:
    if offline_path:
        lib = ReferenceLibrary.load(offline_path)
    else:
        lib = ZMPWalkGenerator().build_library_for_vx_values(list(vx_grid))
    cmd_keys = [(vx, 0.0, 0.0) for vx in vx_grid]

self._offline_cmd_keys = jp.asarray(cmd_keys, dtype=jp.float32)  # (n_bins, 3)
```

In the per-bin stacking loop, ensure the n_bins axis ordering of stacked JAX arrays matches `cmd_keys` order.

**Verify**: P5.3 passes.

### P5.4a — Add `_yaw_from_quat_wxyz` JAX helper (NEW-3 / C11) (failing test)

**Append** to `training/tests/test_loc_ref_3d_config.py`:
```python
def test_yaw_from_quat_wxyz_recovers_yaw() -> None:
    import math
    import jax.numpy as jp
    from control.references.runtime_reference_service import _yaw_from_quat_wxyz
    yaw = 0.3
    q = jp.asarray(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=jp.float32
    )  # wxyz
    assert float(_yaw_from_quat_wxyz(q)) == pytest.approx(yaw, abs=1e-5)
```

**Impl**: Add to `control/references/runtime_reference_service.py` (module level, near the top of the file):
```python
def _yaw_from_quat_wxyz(quat_wxyz):
    """Extract yaw (rad) from a wxyz quaternion.  JAX-only; mirrors
    policy_contract.jax.frames._yaw_from_quat_xyzw with the index
    layout shifted."""
    import jax.numpy as jnp
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)
```

**Verify**: passes.

### P5.5 — `_lookup_offline_window` heading-frame correction + 3D nearest-key (NEW-3 JAX-native rewrite) + commit

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
    assert float(win_a["selected_vy"]) != float(win_b["selected_vy"])


def test_lookup_with_yawed_heading_picks_different_bin() -> None:
    """Heading-frame correction: actor yawed +0.3 rad while commanding
    +vy must select a different bin than actor at zero yaw."""
    import math, jax.numpy as jp
    env = _env_3d_via_smoke1()
    cmd = jp.asarray([0.20, 0.10, 0.0], dtype=jp.float32)
    yaw = 0.3
    path_rot_wxyz = jp.asarray(
        [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=jp.float32
    )
    win_yawed = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=cmd,
        path_rot_wxyz=path_rot_wxyz,
        heading_rot_wxyz=jp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jp.float32),
    )
    win_aligned = env._lookup_offline_window(
        step_idx=jp.asarray(10, dtype=jp.int32),
        velocity_cmd=cmd,
        path_rot_wxyz=jp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jp.float32),
        heading_rot_wxyz=jp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jp.float32),
    )
    # Different bin (yawed correction changes the (vx, vy) projection).
    assert float(win_yawed["selected_vy"]) != float(win_aligned["selected_vy"])
```

**Impl** in `wildrobot_env.py:1248-1340`:

```python
def _lookup_offline_window(
    self,
    step_idx: jax.Array,
    velocity_cmd: Optional[jax.Array] = None,  # (3,) when supplied
    path_rot_wxyz: Optional[jax.Array] = None,  # (4,) torso's current path-rot
    heading_rot_wxyz: Optional[jax.Array] = None,  # (4,) target heading-rot
) -> Dict[str, jax.Array]:
    # ... legacy single-bin early-return path unchanged ...

    cmd = jp.asarray(velocity_cmd, dtype=jp.float32).reshape(3)

    # Heading-frame correction (TB walk_zmp_ref.py:132-137 parity, but
    # implemented with JAX-native quat helpers — NEW-3 / C11).
    if path_rot_wxyz is not None and heading_rot_wxyz is not None:
        from control.references.runtime_reference_service import _yaw_from_quat_wxyz
        yaw_path = _yaw_from_quat_wxyz(path_rot_wxyz)
        yaw_heading = _yaw_from_quat_wxyz(heading_rot_wxyz)
        # heading-error rotation:  R_err = R_heading^{-1} * R_path
        # (yaw_err = yaw_heading - yaw_path), applied to (vx, vy) cmd
        # in body frame.  TB rotates cmd by R_err, i.e. by Rz(-delta_yaw).
        delta_yaw = yaw_path - yaw_heading
        c, s = jp.cos(delta_yaw), jp.sin(delta_yaw)
        vx_corr = c * cmd[0] - s * cmd[1]
        vy_corr = s * cmd[0] + c * cmd[1]
        cmd_corrected = jp.stack([vx_corr, vy_corr, cmd[2]])
    else:
        cmd_corrected = cmd

    # 3D nearest-key lookup (replaces 1D argmin over |vx_grid - cmd|).
    diffs = self._offline_cmd_keys - cmd_corrected  # (n_bins, 3)
    bin_idx = jp.argmin(jp.linalg.norm(diffs, axis=-1)).astype(jp.int32)

    selected_vx = self._offline_cmd_keys[bin_idx, 0]
    selected_vy = self._offline_cmd_keys[bin_idx, 1]
    selected_yaw_rate = self._offline_cmd_keys[bin_idx, 2]
    cmd_bin_l2_err = jp.linalg.norm(
        self._offline_cmd_keys[bin_idx] - cmd_corrected
    )

    # ... stack the per-field jax arrays at bin_idx, return dict ...
    return {
        # ... existing fields ...
        "selected_vx": selected_vx,
        "selected_vy": selected_vy,
        "selected_yaw_rate": selected_yaw_rate,
        "cmd_bin_l2_err": cmd_bin_l2_err,
    }
```

Update the two callers at `wildrobot_env.py:2957, 2973`:
```python
win = self._lookup_offline_window(
    next_step_idx,
    velocity_cmd=velocity_cmd,
    path_rot_wxyz=path_state["path_rot"],
    heading_rot_wxyz=heading_rot_wxyz,  # supplied via _compute_reward_terms
)
```
(`heading_rot_wxyz` is computed from the current torso quat via `axis_angle_to_quat(jnp.array([0,0,1]), yaw_torso)` — the existing `heading_rot` helper in the env at line ~1700.)

> **C11 verification**: no `scipy.spatial.transform.Rotation` syntax appears anywhere in the heading-frame correction. All operations use JAX primitives (`jnp.cos`, `jnp.sin`, `jnp.stack`, `_yaw_from_quat_wxyz`).

> **S2 (path_rot_wxyz optional fallback)**: kept optional with explicit `is None` check so the legacy `_lookup_offline_window(step_idx, velocity_cmd=...)` callers (e.g. unit tests) continue to work. The smoke1 callers MUST pass `path_rot_wxyz=` AND `heading_rot_wxyz=`; failure to wire either causes a silent no-op. Verified by `test_lookup_with_yawed_heading_picks_different_bin` which asserts different bins are picked when heading rotation is applied.

**Verify**: P5.5 tests pass; legacy `test_cmd_velocity_tracking_mode.py::test_legacy_mode_selected_bin_diagnostics_use_configured_vx` still passes.

**Commit**:
```bash
git add training/configs/training_runtime_config.py training/configs/training_config.py \
        training/envs/wildrobot_env.py \
        control/references/runtime_reference_service.py \
        training/tests/test_loc_ref_3d_config.py
git commit -m "v0.21.0 P5: 3D nearest-key lookup + heading-frame correction (JAX-native quat)"
```

---

## P6 — Reward updates: optional `vy_cmd` arg + yaw_rate tracking term

**File**: `wildrobot_env.py` (lines 1163–1244, 1777–1800, 3050–3120).

### P6.1 — Add optional `vy_cmd` arg to `_cmd_forward_velocity_track_reward` (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_cmd_lateral_velocity_tracking.py`
```python
import pytest
from training.envs.wildrobot_env import WildRobotEnv


def test_2d_cmd_tracking_uses_supplied_vy_cmd() -> None:
    reward, xy_err, lat_abs, _ = WildRobotEnv._cmd_forward_velocity_track_reward(
        forward_velocity=0.10,
        lateral_velocity=0.05,
        velocity_cmd=0.10,
        vy_cmd=0.05,            # NEW kwarg
        ref_forward_velocity=0.10,
        ref_lateral_velocity=0.05,
        alpha=562.5,
        track_dim=2,
    )
    assert float(reward) == pytest.approx(1.0)
```

**Run**: expect failure (`unexpected keyword argument 'vy_cmd'`).

### P6.2 — Impl: add `vy_cmd: jax.Array = jp.float32(0.0)` kwarg + edit line 1218 (NEW-1 resolution)

In `wildrobot_env.py:1163-1244`:
1. Add `vy_cmd: jax.Array = jp.float32(0.0)` to the static helper signature.
2. **At line 1218**, replace `vy_err_cmd = lateral_velocity` with `vy_err_cmd = lateral_velocity - vy_cmd`. THIS IS THE SINGLE EDIT TO LINE 1218 IN THE ENTIRE PLAN (resolves NEW-1).
3. Update docstring text at lines 1188 and 1192-1195 to reflect new semantics: replace `WR currently has no lateral command, so vy_cmd == 0` with `vy_cmd defaults to 0.0 for backward compat; v0.21.0 callers pass the sampled vy_cmd from velocity_cmd[1]`.

**Verify**: P6.1 passes. Re-run `test_cmd_velocity_tracking_mode.py` — the three existing tests still pass (they call without `vy_cmd`; default 0.0 reproduces prior `vy_err_cmd = lateral_velocity - 0.0 = lateral_velocity` behavior exactly — explicit C16 mitigation).

### P6.3 — Wire real `vy_cmd` from the 3-vec at the call site (and update text at 1781)

In `wildrobot_env.py` at the `_compute_reward_terms` call site near line 1791-1794, replace:
```python
# BEFORE:
self._cmd_forward_velocity_track_reward(
    forward_velocity=...,
    lateral_velocity=...,
    velocity_cmd=velocity_cmd,        # was: scalar arg
    ...
)
# AFTER:
self._cmd_forward_velocity_track_reward(
    forward_velocity=...,
    lateral_velocity=...,
    velocity_cmd=velocity_cmd[0],     # 3-vec[0] = vx_cmd
    vy_cmd=velocity_cmd[1],           # NEW
    ...
)
```
Also update comment at line 1781 from `"(vx, vy) to (velocity_cmd, 0) (WR has no lateral command)"` to `"(vx, vy) to (velocity_cmd[0], velocity_cmd[1])"`.

**Test** (real end-to-end, NOT a stub — fixes S4):
```python
def test_env_reward_consumes_real_vy_cmd_from_3vec() -> None:
    import jax
    import jax.numpy as jp
    from training.envs.wildrobot_env import WildRobotEnv
    from training.envs.env_info import WR_INFO_KEY
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    # Force a (vx=0, vy=0.10, wz=0) cmd.
    new_wr = wr.replace(velocity_cmd=jp.array([0.0, 0.10, 0.0], dtype=jp.float32))
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    state2 = env.step(state, jp.zeros(env.action_size))
    # cmd_velocity_xy_err is sqrt((vx_actual - 0)^2 + (vy_actual - 0.10)^2).
    # On the first step the actor has not moved, vx=vy=0, so err = 0.10.
    err = state2.metrics["tracking/cmd_velocity_xy_err"]
    assert float(err) == pytest.approx(0.10, abs=0.02)
```

**Verify**: passes.

### P6.4 — Yaw-rate tracking term (failing + impl + commit)

```python
def test_yaw_rate_track_reward_peaks_at_cmd() -> None:
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.10,
        yaw_rate_cmd=0.10,
        alpha=562.5,
    )
    assert float(reward) == pytest.approx(1.0)


def test_yaw_rate_track_reward_decays_off_cmd() -> None:
    import math
    reward = WildRobotEnv._yaw_rate_track_reward(
        ang_vel_z=0.0,
        yaw_rate_cmd=0.10,
        alpha=562.5,
    )
    # err_sq = 0.01 -> reward = exp(-5.625) ≈ 0.00361
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
2. Wire into `_compute_reward_terms` (line 1591) right after the existing cmd_forward term:
```python
r_yaw_rate_track = WildRobotEnv._yaw_rate_track_reward(
    ang_vel_z=ang_vel_h[2],  # heading-local ang_vel z
    yaw_rate_cmd=velocity_cmd[2],
    alpha=self._config.reward_weights.cmd_velocity_track_alpha,
)
```
Add `r_yaw_rate_track` to the returned `terms` dict.
3. Add `cmd_yaw_rate_track: float = 0.0` to `RewardWeightsConfig` (training_runtime_config.py around line 877). Also extend `training_config.py` to read `float(rw.get("cmd_yaw_rate_track", 0.0))`.
4. Apply weight in the aggregator (line 2186): `+ w.cmd_yaw_rate_track * terms["r_yaw_rate_track"]`.
5. Add metric `tracking/yaw_rate_err` to `training/core/metrics_registry.py` with `Reducer.MEAN`.

**Verify**: P6.4 tests pass; pre-existing tests still pass; audit test still passes (after updating EXPECTED_COUNTS).

**Commit**:
```bash
git add training/envs/wildrobot_env.py training/configs/training_runtime_config.py \
        training/configs/training_config.py \
        training/core/metrics_registry.py training/tests/test_cmd_lateral_velocity_tracking.py \
        training/tests/test_velocity_cmd_callsite_audit.py \
        training/docs/v0210_velocity_cmd_audit_checklist.md
git commit -m "v0.21.0 P6: reward consumes real vy_cmd (line 1218); yaw_rate tracking term added"
```

---

## P7 — Observation layout `wr_obs_v8_cmd3d`

**File**: `policy_contract/spec.py`, `spec_builder.py`, `jax/obs.py`.

> P7 only touches `policy_contract/` files (plus a one-line `wildrobot_env.py:1464` update to also pass `velocity_cmd_lateral_yaw`). Depends on P3 (env passes a 3-vec `velocity_cmd` into `build_observation`).

### P7.1 — Register `wr_obs_v8_cmd3d` in `SUPPORTED_LAYOUT_IDS` (failing test)

**File**: `/Users/ygli/projects/wildrobot/training/tests/test_obs_v8_cmd3d.py`
```python
def test_v8_layout_id_is_supported() -> None:
    from policy_contract.spec import SUPPORTED_LAYOUT_IDS
    assert "wr_obs_v8_cmd3d" in SUPPORTED_LAYOUT_IDS
```

### P7.2 — Impl: add to supported set, spec_builder, `_validate_observation`, obs.py dispatcher, AND env kwarg wiring (NEW-2)

**(a) `policy_contract/spec.py:11-32`** — add `"wr_obs_v8_cmd3d",` to the set.

**(b) `policy_contract/spec_builder.py::_build_obs_layout`** — add v8 branch:
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
        ObsFieldSpec(name="velocity_cmd", size=1, units="m_s"),  # vx_cmd
        ObsFieldSpec(name="loc_ref_phase_sin_cos", size=2, ...),
        ObsFieldSpec(name="proprio_history",
                     size=PROPRIO_HISTORY_FRAMES * proprio_bundle, ...),
        ObsFieldSpec(name="velocity_cmd_lateral_yaw", size=2,
                     units="m_s_and_rad_s"),  # NEW
        ObsFieldSpec(name="padding", size=1, units="unused"),
    ]
```

**(c) `policy_contract/spec.py::_validate_observation`** — add `wr_obs_v8_cmd3d` branch with matching `expected` list.

**(d) `policy_contract/jax/obs.py`** — three coordinated changes resolving NEW-2:

**d.1 — Signature change** (line 12-44): add the new kwarg AFTER the existing `velocity_cmd` parameter:
```python
def build_observation_from_components(
    *,
    spec: PolicySpec,
    gravity_local: jnp.ndarray,
    angvel_heading_local: jnp.ndarray,
    joint_pos_normalized: jnp.ndarray,
    joint_vel_normalized: jnp.ndarray,
    foot_switches: jnp.ndarray,
    prev_action: jnp.ndarray,
    velocity_cmd: jnp.ndarray,
    velocity_cmd_lateral_yaw: jnp.ndarray | None = None,  # NEW — v8 only
    capture_point_error: jnp.ndarray | None = None,
    # ... rest unchanged ...
)
```
Also propagate the kwarg through the wrapper `build_observation` (line 199-258).

**d.2 — Supported-layout set** (line 46-51):
```python
if spec.observation.layout_id not in {
    "wr_obs_v1", "wr_obs_v2", "wr_obs_v3", "wr_obs_v4",
    "wr_obs_v6_offline_ref_history",
    "wr_obs_v7_phase_proprio",
    "wr_obs_v8_cmd3d",  # NEW
}:
    raise ValueError(f"Unsupported layout_id: {spec.observation.layout_id}")
```

**d.3 — Shared `velocity_cmd` slot at line 151** — slice to `[..., :1]` so v1-v7 see byte-identical 1-vec and v8 takes vx:
```python
parts.append(
    jnp.atleast_1d(jnp.asarray(velocity_cmd, dtype=jnp.float32))[..., :1].reshape(1)
)
```

**d.4 — Append v8 dispatch block** (after the v7 block around line 193):
```python
if spec.observation.layout_id == "wr_obs_v8_cmd3d":
    parts.append(phase)
    if proprio_history is None:
        raise ValueError("wr_obs_v8_cmd3d requires proprio_history; got None.")
    parts.append(jnp.asarray(proprio_history, dtype=jnp.float32).reshape(-1))
    # NEW: 2-vec (vy_cmd, wz_cmd) slot — fed via the new kwarg.
    if velocity_cmd_lateral_yaw is None:
        raise ValueError(
            "wr_obs_v8_cmd3d requires velocity_cmd_lateral_yaw=(vy, wz); got None."
        )
    parts.append(
        jnp.asarray(velocity_cmd_lateral_yaw, dtype=jnp.float32).reshape(2)
    )
```

**(e) `wildrobot_env.py:1464`** — env now passes BOTH the full `velocity_cmd` (for the shared slot) and the lateral-yaw slice (for v8):
```python
return build_observation(
    spec=self._policy_spec,
    state=policy_state,
    signals=signals,
    velocity_cmd=velocity_cmd,                       # (3,) — slot at line 151 takes [..., :1]
    velocity_cmd_lateral_yaw=velocity_cmd[1:],       # NEW — v8 consumes; v1-v7 ignore (None passthrough is fine because they don't read it)
    # ... rest unchanged ...
)
```

> **NEW-2 rule (recorded for future contributors)**: `_get_obs` ALWAYS passes the full 3-vec `velocity_cmd` AND a separate `velocity_cmd_lateral_yaw=velocity_cmd[1:]`. Legacy layouts (v1-v7) ignore the new kwarg because their dispatch blocks don't read it; v8 dispatches it explicitly. Reward / metric paths in `wildrobot_env.py` use indexed slices `velocity_cmd[0]`, `velocity_cmd[1]`, `velocity_cmd[2]` (see P3.8 and P6.3).

Also propagate `velocity_cmd_lateral_yaw` through `policy_contract/jax/obs.py::build_observation` wrapper.

### P7.3 — Validate the v8 layout end-to-end

```python
def test_v8_spec_validation_passes_and_obs_concat_matches_obs_dim() -> None:
    from policy_contract.spec_builder import build_policy_spec
    spec = build_policy_spec(
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
    assert spec.observation.layout_id == "wr_obs_v8_cmd3d"
    sizes = [f.size for f in spec.observation.layout]
    assert sum(sizes) == spec.model.obs_dim


def test_v8_build_observation_produces_obs_with_3d_cmd_slot() -> None:
    import jax.numpy as jp
    from policy_contract.jax.obs import build_observation_from_components
    from policy_contract.spec_builder import build_policy_spec
    spec = build_policy_spec(
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
    action_dim = 13
    proprio_bundle = 3 + 4 + 3 * action_dim
    proprio_history = jp.zeros((15 * proprio_bundle,), dtype=jp.float32)
    obs = build_observation_from_components(
        spec=spec,
        gravity_local=jp.zeros(3),
        angvel_heading_local=jp.zeros(3),
        joint_pos_normalized=jp.zeros(action_dim),
        joint_vel_normalized=jp.zeros(action_dim),
        foot_switches=jp.zeros(4),
        prev_action=jp.zeros(action_dim),
        velocity_cmd=jp.array([0.20, 0.05, 0.10], dtype=jp.float32),
        velocity_cmd_lateral_yaw=jp.array([0.05, 0.10], dtype=jp.float32),
        loc_ref_phase_sin_cos=jp.zeros(2),
        proprio_history=proprio_history,
    )
    assert obs.shape == (spec.model.obs_dim,)


def test_v7_obs_byte_identical_when_velocity_cmd_passed_as_1vec() -> None:
    """S5: v7 must produce the same obs vector when fed velocity_cmd
    as a (1,) vs as scalar."""
    import jax.numpy as jp
    from policy_contract.jax.obs import build_observation_from_components
    from policy_contract.spec_builder import build_policy_spec
    spec = build_policy_spec(
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
    proprio_history = jp.zeros((15 * proprio_bundle,), dtype=jp.float32)
    kw = dict(
        spec=spec, gravity_local=jp.zeros(3), angvel_heading_local=jp.zeros(3),
        joint_pos_normalized=jp.zeros(action_dim),
        joint_vel_normalized=jp.zeros(action_dim),
        foot_switches=jp.zeros(4), prev_action=jp.zeros(action_dim),
        loc_ref_phase_sin_cos=jp.zeros(2), proprio_history=proprio_history,
    )
    obs_scalar = build_observation_from_components(velocity_cmd=jp.array([0.20]), **kw)
    obs_3vec = build_observation_from_components(
        velocity_cmd=jp.array([0.20, 0.05, 0.10]),
        velocity_cmd_lateral_yaw=jp.array([0.05, 0.10]),  # ignored by v7
        **kw,
    )
    assert obs_scalar.shape == obs_3vec.shape
    # Byte-identical: slicing [..., :1] of 3-vec gives [0.20], same as scalar wrap.
    import numpy as np
    np.testing.assert_array_equal(np.asarray(obs_scalar), np.asarray(obs_3vec))
```

### P7.4 — Env init guard accepts v8 + commit

In `wildrobot_env.py:225-239`:
```python
if layout_id not in (
    "wr_obs_v6_offline_ref_history",
    "wr_obs_v7_phase_proprio",
    "wr_obs_v8_cmd3d",          # NEW
):
    raise ValueError(...)
```

```python
def test_env_init_accepts_v8_layout() -> None:
    from training.envs.wildrobot_env import WildRobotEnv
    from training.configs.training_config import load_training_config
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    env = WildRobotEnv(cfg)
    assert env._policy_spec.observation.layout_id == "wr_obs_v8_cmd3d"
```

**Commit**:
```bash
git add policy_contract/spec.py policy_contract/spec_builder.py policy_contract/jax/obs.py \
        training/envs/wildrobot_env.py training/tests/test_obs_v8_cmd3d.py \
        training/tests/test_velocity_cmd_callsite_audit.py \
        training/docs/v0210_velocity_cmd_audit_checklist.md
git commit -m "v0.21.0 P7: register wr_obs_v8_cmd3d; new velocity_cmd_lateral_yaw kwarg"
```

---

## P8 — Smoke YAML + residual scales

### P8.1 — `ppo_walking_v0210_smoke1_lateral_yaw.yaml`

**File**: `/Users/ygli/projects/wildrobot/training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`

Clone `ppo_walking_v0201_smoke14.yaml` and apply diffs (use the REAL field names — NOT `obs_version`):
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
  cmd_deadzone: [0.02, 0.02, 0.05]       # NEW 3-tuple form
  eval_velocity_cmd: [0.18, 0.04, 0.10]  # NEW 3-tuple form
  loc_ref_version: v3_offline_library    # UNCHANGED (C5)
  loc_ref_command_conditioned: true
  loc_ref_command_axes_3d: true          # NEW (C5)
  loc_ref_command_grid_interval: 0.04    # gives vx_grid = [0.18, 0.22, 0.26]
  loc_ref_offline_command_vx: 0.22
  loc_ref_offline_command_vy_grid: [-0.10, 0.0, 0.10]
  loc_ref_offline_command_yaw_rate_grid: [-0.20, 0.0, 0.20]
  loc_ref_residual_scale_per_joint:
    left_hip_roll:    <smoke14_value * 1.5>
    right_hip_roll:   <smoke14_value * 1.5>
    left_ankle_roll:  <smoke14_value * 1.5>
    right_ankle_roll: <smoke14_value * 1.5>
    left_hip_yaw:     <smoke14_value * 1.5>
    right_hip_yaw:    <smoke14_value * 1.5>
    # all other joints unchanged from smoke14
reward_weights:
  cmd_velocity_track_dim: 2
  cmd_velocity_track: 2.0
  cmd_yaw_rate_track:  1.5               # NEW (default 0.0 in code; smoke1 sets active)
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
import math
import jax
import jax.numpy as jp
import pytest
from training.envs.wildrobot_env import WildRobotEnv
from training.envs.env_info import WR_INFO_KEY
from training.configs.training_config import load_training_config


def _env():
    cfg = load_training_config("training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml")
    return WildRobotEnv(cfg)


def _yaw_from_qpos_wxyz(qpos):
    # WR free joint quat is wxyz at qpos[3:7].
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(float(siny_cosp), float(cosy_cosp))


def test_env_rollout_with_lateral_cmd_shows_pelvis_y_drift() -> None:
    env = _env()
    state = env.reset(jax.random.PRNGKey(0))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(velocity_cmd=jp.array([0.20, 0.10, 0.0], dtype=jp.float32))
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    for _ in range(50):
        state = env.step(state, jp.zeros(env.action_size),
                         disable_pushes=True, disable_cmd_resample=True)
    assert state.pipeline_state.qpos[1] > 0.005


def test_env_rollout_with_yaw_cmd_changes_torso_yaw() -> None:
    """S7: read yaw via proper quat -> euler, not raw qpos[6]."""
    env = _env()
    state = env.reset(jax.random.PRNGKey(1))
    wr = state.info[WR_INFO_KEY]
    new_wr = wr.replace(velocity_cmd=jp.array([0.0, 0.0, 0.20], dtype=jp.float32))
    state = state.replace(info={**state.info, WR_INFO_KEY: new_wr})
    yaw_init = _yaw_from_qpos_wxyz(state.pipeline_state.qpos)
    for _ in range(50):
        state = env.step(state, jp.zeros(env.action_size),
                         disable_pushes=True, disable_cmd_resample=True)
    yaw_final = _yaw_from_qpos_wxyz(state.pipeline_state.qpos)
    # 0.20 rad/s * 50 steps * 0.02 s/step = 0.20 rad expected; allow ±0.05.
    assert abs(yaw_final - yaw_init) > 0.01
```

### P9.2 — v7 layout regression

```python
def test_smoke14_env_still_initializes_under_v7() -> None:
    cfg = load_training_config("training/configs/ppo_walking_v0201_smoke14.yaml")
    env = WildRobotEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))
    assert env._policy_spec.observation.layout_id == "wr_obs_v7_phase_proprio"
    wr = state.info[WR_INFO_KEY]
    assert wr.velocity_cmd.shape == (3,)
```

### P9.3 — `walking_training.md` G6 / G7 gates + commit

**File**: `/Users/ygli/projects/wildrobot/training/docs/walking_training.md`

Append a new section under "Gate definitions":
```markdown
### G6 — Lateral command tracking (v0.21.0+)

Definition: at eval cmd `(vx_eval, vy_eval, 0)` with `vy_eval > 0`, the
signed ratio `lateral_velocity / vy_eval` is >= 0.5 averaged over the
last 100 eval steps, across >=3 seeds.

### G7 — Yaw command tracking (v0.21.0+)

Definition: at eval cmd `(0, 0, wz_eval)` with `wz_eval > 0`, the
signed ratio `ang_vel_z / wz_eval` is >= 0.5 averaged over the last
100 eval steps, across >=3 seeds.

### Promotion criterion (v0.21.0)

G4 + G5 (forward) MUST NOT regress beyond 10% relative to the
v0.20.1-smoke14 baseline at `eval_velocity_cmd = (0.18, 0, 0)`.

G6 + G7 MUST clear >=0.5 signed ratio on the first smoke before
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
vx_cmd; new 2-vec `velocity_cmd_lateral_yaw` slot appended for vy/wz);
activates dim=2 cmd reward with real `vy_cmd` and adds `cmd_yaw_rate_track`
term (default weight 0.0 — backward-compat shim — smoke1 sets 1.5).

**Architecture deltas**:
- `WildRobotInfo.velocity_cmd` migrated `() -> (3,)` across ~10 callsites.
- `cmd_deadzone` migrated `float -> tuple[float, float, float]` with YAML
  shim (`_load_three_axis`) at `training_config.py:549`.
- `eval_velocity_cmd` migrated `float -> tuple[float, float, float]` with
  YAML shim and sentinel `(-1, -1, -1)` (reduction `jp.all(eval_cmd >= 0)`).
- `cmd_turn_chance` (existing placeholder field) now actually consumed by
  the branched sampler.
- `loc_ref_command_axes_3d` (NEW boolean) gates 3D library; existing
  `loc_ref_version: v3_offline_library` validator UNCHANGED.
- `compute_command_integrated_path_state` integrates (vy, wz). Closed form:
  `p_n = dt * sum_{k=1..n} Rz(k*theta) @ [vx, vy]` with the cardinal trig
  sums `C(n), S(n)` from the half-angle identity.
- 3D library lookup uses heading-frame correction implemented in
  JAX-native quat ops (new `_yaw_from_quat_wxyz` helper).
- New `velocity_cmd_lateral_yaw` kwarg in `build_observation_from_components`;
  v1-v7 ignore it; v8 consumes the (vy, wz) tail.

**Gates**:
- G4 forward stability: <pass/fail>
- G5 forward step amplitude: <pass/fail>
- G6 lateral tracking (signed ratio >= 0.5): <%>
- G7 yaw tracking (signed ratio >= 0.5): <%>

**Outcomes**:
- <observations from the wandb run>
- <next-version proposals — likely either widen wz range if G7 saturates
  or drop rotation_radius if turn radius too large>
```

---

## End-to-end verification

After P10.1 the full pipeline should be:
1. `uv run pytest training/tests/ -q` — all green (including v0.20 regressions: `test_cmd_velocity_tracking_mode.py`, `test_v0201_env_zero_action.py`, `test_runtime_reference_service.py`; AND the v0.21 audit test `test_velocity_cmd_callsite_audit.py`).
2. `uv run python training/train.py --config training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml --max-steps 500 --seeds 3` — runs 500 PPO steps without crash.
3. Eval rollout with cmd `(0.18, 0.04, 0.10)` shows visible strafe + turn motion in the rendered video (no black frames).
4. wandb metrics `tracking/velocity_cmd_vy_abs`, `tracking/yaw_rate_err` populate and trend toward zero.
5. Forward G4/G5 metrics within 10% of smoke14 baseline.

---

## Files touched (summary)

**Create**:
- `training/configs/ppo_walking_v0210_smoke1_lateral_yaw.yaml`
- `training/docs/v0210_lateral_yaw_prior_plan-final.md` (this file)
- `training/docs/v0210_velocity_cmd_audit_checklist.md` (generated in P0.5.1)
- `training/tests/test_velocity_cmd_callsite_audit.py`
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
- `control/references/runtime_reference_service.py` — `compute_command_integrated_path_state` accepts `velocity_cmd_y_mps`; 2D closed-form `path_pos`; new `_yaw_from_quat_wxyz` helper
- `training/envs/wildrobot_env.py` — `_init_offline_service` 3D branch, `_lookup_offline_window` 3D + JAX-native heading-frame, layout-allow-list, `_cmd_forward_velocity_track_reward` (+`vy_cmd` kwarg AND line-1218 edit in P6.2), `_yaw_rate_track_reward`, `_compute_reward_terms`, `_sample_velocity_cmd` branched 3-vec, every scalar-`velocity_cmd` callsite (~10 sites), env passes `velocity_cmd_lateral_yaw=velocity_cmd[1:]` to build_observation
- `training/envs/env_info.py` — `velocity_cmd` shape `() -> (3,)` in both backends and in `get_expected_shapes`
- `training/configs/training_runtime_config.py` — `LocomotionEnvConfig`: +`min/max_velocity_y`, `max_yaw_rate`, `loc_ref_command_axes_3d`, `loc_ref_offline_command_vy_grid`, `loc_ref_offline_command_yaw_rate_grid`; type bumps for `cmd_deadzone` and `eval_velocity_cmd`; `RewardWeightsConfig.cmd_yaw_rate_track`
- `training/configs/training_config.py` — `_load_three_axis` helper; REPLACE `float(env.get(...))` at lines 549-550; read new fields
- `policy_contract/spec.py` — `SUPPORTED_LAYOUT_IDS += {"wr_obs_v8_cmd3d"}`; `_validate_observation` v8 branch
- `policy_contract/spec_builder.py` — `_build_obs_layout` v8 branch
- `policy_contract/jax/obs.py` — dispatcher accepts v8; new `velocity_cmd_lateral_yaw` kwarg; v8 branch appends 2-vec; line 151 sliced to `[..., :1]`
- `training/core/metrics_registry.py` — add `tracking/velocity_cmd_vy_abs`, `tracking/velocity_cmd_wz_abs`, `tracking/velocity_cmd_norm`, `tracking/yaw_rate_err`
- `training/docs/walking_training.md` — G6 / G7 sections
- `training/CHANGELOG.md` — v0.21.0 entry (after smoke)

---

## Risks & rollback

- **Risk**: Forward G4/G5 regresses when lateral residual scales are bumped. **Mitigation**: P8.2 keeps non-lateral joints unchanged; if regression observed, fall back to 1.0x scales on lateral joints.
- **Risk**: 27-bin library uses too much device memory (S3 napkin: ~50 MB on-device per env, constant across vectorization). **Mitigation**: smoke1 grid intentionally small; if memory-bound, drop wz_grid to 1 entry first.
- **Risk**: Layout v8 bump breaks downstream eval / inference. **Mitigation**: v1-v7 layouts untouched; old configs continue to work. New `actor_obs_layout_id: wr_obs_v8_cmd3d` is opt-in.
- **Risk**: `cmd_deadzone` / `eval_velocity_cmd` `_load_three_axis` shim mis-classifies a 2-vec or 4-vec as a 3-axis field. **Mitigation**: shim explicitly raises `ValueError` on non-3 lengths.
- **Risk**: smoke14's `eval_velocity_cmd: 0.26` now broadcasts to (0.26, 0.26, 0.26) which is out-of-range on the lateral/yaw axes. **Mitigation**: smoke14 baseline already captured; we are not re-running it. New configs should write explicit 3-tuples.
- **Risk**: `compute_command_integrated_path_state` 2D extension produces drift that fights the policy. **Mitigation**: P3.9 unit tests assert integration matches analytic for pure-vy and pure-wz cases AND that vy=0, wz=0 collapses byte-identically to the legacy form; if smoke1 shows drift fights, gate `r_torso_pos_xy` weight to 0 when `|vy_cmd| > eps OR |wz_cmd| > eps` (S2 fallback).
- **Risk**: `_lookup_offline_window` heading-frame correction silently no-ops when caller forgets to pass `path_rot_wxyz` / `heading_rot_wxyz`. **Mitigation**: P5.5 test `test_lookup_with_yawed_heading_picks_different_bin` asserts a different bin is picked under a yawed heading; if the test fails after wiring, the kwargs aren't being passed.
- **Rollback**: revert by switching smoke YAML back to `actor_obs_layout_id: wr_obs_v7_phase_proprio` + `loc_ref_command_axes_3d: false`. All v0.21 code paths default to vy=wz=0 when the toggle is false and the grids are empty tuples.
