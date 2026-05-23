VERDICT: NEEDS_REVISION

## Summary Assessment

The v2 plan addresses ~15 of the 17 round-1 issues correctly (C1, C2, C3, C4, C5, C6, C7, C8, C11, C12, C13, C14, C15, C16, C17 are all materially fixed and the architectural direction is now sound). However, two round-1 issues are only partially addressed (C9 missed callsites; C10 path-state integration is described but the closed-form math sketched in P3.10 is wrong) and the v2 revision introduces several NEW critical issues — most importantly an internal contradiction between P3.8 and P6.2 about line 1218, a mismatch between the v8 obs-builder kwargs and the env's actual call-site, a path_rot/Rotation type confusion in the heading-frame correction (P5.5), and missing migration of the `yaml.safe_load` `float(...)` casts at training_config.py:549-550 that will throw `TypeError` on list inputs before the new shim can run.

---

## Critical Issues (must fix)

### NEW-1. P3.8 deletes a line that P6.2 then tries to replace
- P3.8 says: "`wildrobot_env.py:1218` — DELETE the line, parameter is now sourced from caller (handled in P6)."
- P6.2 says: "Replace `vy_err_cmd = lateral_velocity` (line 1218) with `vy_err_cmd = lateral_velocity - vy_cmd`."
- These two are mutually exclusive. After P3 lands and the commit is made (P3 has a single `git commit` at P3.10), line 1218 no longer exists; P6.2's "replace" instruction targets a deleted line. The actual intent (line 1218's operand changes from `lateral_velocity` to `lateral_velocity - vy_cmd`) needs to live in exactly one phase. Recommendation: move the line-1218 edit entirely into P6.2 and remove the bullet from P3.8. Otherwise the P3 commit is incomplete (still hard-codes `vy_cmd=0` via line 1218) AND the P6.2 step's literal instruction fails.

### NEW-2. The v8 obs-builder kwarg story is incomplete — env call-site never gets the lateral-yaw signal
- `build_observation_from_components()` at `policy_contract/jax/obs.py:12-44` takes a fixed kwarg list. v2's P7.2 modifies the dispatcher to consume `velocity_cmd[1:]` as the lateral-yaw payload, but does NOT add a new kwarg to the function signature OR document that the env's existing `velocity_cmd=...` kwarg now carries the full 3-vec.
- Critical question the plan does not answer: does the env call site at `wildrobot_env.py:1443-1464` already pass `wr.velocity_cmd` as the `velocity_cmd=` kwarg? If yes (likely — grep shows line 1464 does `velocity_cmd=velocity_cmd`), then after P3 makes `wr.velocity_cmd` shape `(3,)`, the kwarg is already wired and P7.2's slice `velocity_cmd[1:]` works.
  - BUT: `wildrobot_env.py:1443` declares a function `def _get_obs(self, ..., velocity_cmd: jax.Array, ...)`. If callers further upstream pass scalar slices (the plan's P3.8 says to "index velocity_cmd[0]" in many places), then it's ambiguous whether the *obs path* keeps the full 3-vec. The plan needs an explicit statement: "The env's `_get_obs` continues to receive the full 3-vec `wr.velocity_cmd`; only the reward/metric paths use indexed `velocity_cmd[i]`." Add this to P3.8 as an explicit rule.
- Without this, P7.3's `test_v8_build_observation_produces_obs_with_3d_cmd_slot` will pass at the unit level but the env's end-to-end test (P9.1) will silently feed `velocity_cmd[0]` (a 0-D scalar) into `build_observation`, and `jnp.atleast_1d(scalar)[..., :1]` will give the right shared slot but `velocity_cmd[1:]` will be empty/raise.

### NEW-3. P5.5 heading-frame correction uses `path_rot` as if it were a scipy `Rotation` — it's a quaternion array
- TB's mirror code is `heading_rot.inv() * path_state["path_rot"]` where both are `scipy.spatial.transform.Rotation` objects supporting `.inv()`, `*` composition, and `.apply([vx,vy,0])`.
- WR's `path_state["path_rot"]` (from `runtime_reference_service.py:236-238`) is a `jnp.asarray([cos(yaw/2), 0, 0, sin(yaw/2)], dtype=jnp.float32)` — a (4,) quaternion array. There is no `.inv()` / `.apply()` method on a JAX array.
- Plan's P5.5 says: `cmd_xy_corrected = (heading_rot.inv() @ path_rot).apply(...)`. This will fail at trace time with `AttributeError: 'ArrayImpl' object has no attribute 'inv'`.
- Required fix: P5.5 must either (a) compose the heading-correction quaternion using `policy_contract/jax/frames.py` primitives (quat-mul, quat-inv, quat-rotate-vec — verify these exist before claiming the API), or (b) extract the yaw scalar from both quaternions and rotate by the yaw difference using a 2D rotation matrix. The plan should pick one and write the actual JAX code, not the scipy mirror.

### NEW-4. `training_config.py:549-550` `float(...)` cast will throw on list inputs BEFORE the YAML shim can intervene
- Current code: `cmd_deadzone=float(env.get("cmd_deadzone", 0.0))` and `eval_velocity_cmd=float(env.get("eval_velocity_cmd", -1.0))`.
- Plan's P3.4 says "Add a YAML-loader shim ... when `cmd_deadzone` reads as a scalar, broadcast to (scalar, scalar, scalar) before constructing the dataclass." This describes ONE direction (scalar → tuple) but the smoke1 YAML feeds a LIST `[0.02, 0.02, 0.05]`. `float([0.02, 0.02, 0.05])` raises `TypeError: float() argument must be a string or a real number, not 'list'`.
- The plan must explicitly replace the `float(...)` cast at line 549 (and similarly at 550) with a shim function — e.g. `_load_three_axis(value, default_scalar)` — that handles three cases: scalar → broadcast, 3-tuple/list → tuple, missing → default. Without this, P8.1's smoke1 YAML won't even parse before P4 starts.

### NEW-5. P3.10's "closed-form discrete sum" for 2D `path_pos` is hand-waved; the existing 1D math is non-trivial
- The current `compute_command_integrated_path_state` at `runtime_reference_service.py:216-231` uses a closed-form trick `p_n = dt * sum_{k=1..n} Rz(k*theta) * [vx, 0]` with `ratio = sin(0.5*n*theta) / sin(0.5*theta)` and `x_sum = ratio * cos(0.5*(n+1)*theta)`, `y_sum = ratio * sin(0.5*(n+1)*theta)`, giving `path_pos = [vx*dt*x_sum, vx*dt*y_sum, 0]`.
- The generalization to `[vx, vy]` is NOT `path_pos = [vx*dt*x_sum, vy*dt*y_sum, 0]` (which is what a literal reading of the plan's "matrix form" gloss would suggest). The correct closed-form for `p_n = dt * sum_{k=1..n} Rz(k*theta) @ [vx, vy]` requires a 2×2 rotation expansion: `p_n.x = dt * (vx * X_cos_sum - vy * X_sin_sum)`, `p_n.y = dt * (vx * X_sin_sum + vy * X_cos_sum)`, where `X_cos_sum = sum cos(k*theta)` and `X_sin_sum = sum sin(k*theta)`. The existing scalar derivation gives `X_sin_sum = ratio * sin(0.5*(n+1)*theta)` and `X_cos_sum = ratio * cos(0.5*(n+1)*theta)` (i.e. the existing `x_sum`/`y_sum` are these sums, not the world-frame projections — note the naming was already a bit misleading).
- The plan must pin the exact formula in P3.10, not say "the existing scalar `vx`-only derivation generalises by rotating ... and summing." The risk is that the implementer writes `path_pos = jnp.asarray([vx*dt*x_sum, vy*dt*y_sum, 0])` which fails for any `theta ≠ 0`. Add a derivation block and the explicit two-line formula.

### NEW-6. P0.5 audit is described as a checklist but has no "run code" step — and the enumerated callsites don't include the eval-cmd metrics
- P0.5's command is `grep -nH 'velocity_cmd' ...` over a fixed file list, and the "Expected checklist" is then asserted manually as comments. There is no failing test for "the audit produced the expected set of callsites." This is fine if the developer treats it as scaffolding, but the plan does not gate progress on the audit succeeding — meaning if the developer skips P0.5, the rest of the plan's per-line surgery references stale line numbers when wildrobot_env.py drifts.
- More importantly, P0.5's checklist OMITS three real callsites:
  - `wildrobot_env.py:2113` — comment-only reference to `velocity_cmd`, no code change needed, OK to skip.
  - `wildrobot_env.py:2287` — comment-only, OK to skip.
  - `wildrobot_env.py:1188-1195` — DOCSTRING references `vy_cmd` and the new semantics; needs to be updated in P6.2 (the docstring for `_cmd_forward_velocity_track_reward` currently says "WR has no lateral command, so `vy_cmd == 0`"; after P6.2 this is wrong).
  - `wildrobot_env.py:1216` — `vx_err_cmd = forward_velocity - velocity_cmd` is a LIVE call-site that uses scalar arithmetic on the parameter. Plan covers this correctly via P6.3 (caller passes `velocity_cmd[0]`), but P0.5 doesn't list it.
- Recommendation: turn P0.5 into a documented but optional "scan" step, OR pin a failing test that asserts `grep | wc -l == 26` (or whatever the canonical count is) so the audit is enforceable.

---

## Suggestions (nice to have)

### S1. P3.4 shim test path
P3.4 imports `load_training_config` to test scalar-broadcast. The test will fail if smoke14's `eval_velocity_cmd: 0.1333333333` is feeding the same code path — make sure both YAMLs (smoke14 with scalar, smoke1 with list) parse without TypeError. Recommend adding a test `test_smoke14_yaml_still_loads_after_shim` to lock this.

### S2. P5.5 `path_rot_wxyz=None` fallback obscures the heading-correction code path
- The plan adds a default `path_rot_wxyz: Optional[jax.Array] = None` and silently no-ops when None. This means smoke1 (which DOES want the correction) must remember to pass the kwarg from EVERY caller (2658, 2957, 2973). If any caller is missed, the correction is silently skipped and bin chatter happens at training time. Recommend making `path_rot_wxyz` required (no default) so a missing wire is a TypeError, not a silent fallback.

### S3. P4.5 walk branch `cmd_deadzone[0]` may be wrong for symmetric ranges
- TB's `command_range[5]` is `[-vx_max_neg, +vx_max_pos]` and the deadzone applies SYMMETRICALLY. The plan's `minval=cmd_deadzone[0]` (always positive) plus `* sin(theta)` produces values like `(0.02, max_velocity) * sin(theta)`. When `sin(theta) > 0`, the actual vx is in `[0.02 * sin(theta), max_velocity * sin(theta)]` — when `sin(theta) ≈ 1`, the minimum vx is 0.02, matching deadzone. But when `sin(theta) ≈ 0.1`, minimum vx is 0.002, which is well INSIDE the deadzone. TB has the same issue but TB's `command_range[5]` ranges are absolute, not signed-from-zero. The plan should either mirror TB exactly (current plan does) and accept this behavior, OR add an explicit comment that the "deadzone" here is an axis-magnitude lower bound, not a final-magnitude lower bound.

### S4. P6.3's end-to-end test is a stub
- P6.3 says "End-to-end: build env from smoke1 yaml, run one step with a hand-crafted vy_cmd, assert the cmd_velocity_xy_err metric shrinks when the actual lateral velocity matches the commanded vy." The body is `...`. Replace with a real assertion or remove the stub.

### S5. P7.2 modifies the shared `velocity_cmd` slot inside the per-layout dispatch
- The plan's diff to line 151 of obs.py replaces `jnp.atleast_1d(velocity_cmd).reshape(1)` with `jnp.atleast_1d(jnp.asarray(velocity_cmd, dtype=jnp.float32))[..., :1].reshape(1)`. This is a global change that DOES affect v1-v7 (when those layouts get fed a 3-vec from a post-P3 env). The plan acknowledges this is intentional (so v7 sees only vx_cmd) but P9.2 should pin a stronger regression that the v7 obs vector is byte-identical to the pre-P3 v7 obs vector when feeding the same scalar vx.

### S6. P5.4's stacked cmd_keys path silently breaks for `len(vy_values)=0`
- Plan code uses `vy_values = list(...) or [0.0]`. Empty-tuple defaults from P5.2 are fine here. But `cmd_keys` is then `[(vx, vy, wz) for vx in vx_values for vy in vy_values for wz in yaw_values]` — note the variable `yaw_values` is set from `loc_ref_offline_command_yaw_rate_grid` but the for-comprehension references it by the inner name. If the developer typos `yaw_values` vs `yaw_rate_values`, this will silently produce an empty list. Pin with a `assert len(cmd_keys) == len(vx) * len(vy) * len(yaw)`.

### S7. P9.1's `qpos[6]` is wrong for yaw extraction
- The test says "Pelvis quat z component must change (yaw accumulated)" and reads `state.pipeline_state.qpos[6]`. WR's free-joint quaternion is wxyz at indices `qpos[3:7]`, so `qpos[6]` is the `z` component. But just reading `qpos[6]` doesn't actually verify yaw — a robot that tips backward also changes `qpos[6]`. Recommend computing yaw via the quat → euler conversion and asserting `yaw_final - yaw_init > 0`. (Or accept that any quat change with a 0.20 rad/s yaw command for 50 steps at dt=0.02 = 0.2 rad of accumulated yaw is the right ballpark.)

### S8. Defer-or-implement decision on `r_torso_pos_xy` is now in P3.9
- The original C10 recommendation was either implement OR guard `r_torso_pos_xy`. The plan now implements (good). It does NOT explicitly add a regression test that with `(vy=0, wz=0)` the new code produces byte-identical `path_pos` to the old code. Add this as an explicit pre-commit assertion in P3.10.

---

## Verified Fixes from Round 1

| Issue | Status | Notes |
|------|--------|-------|
| **C1** (`command_vx` is a parameter, not a config field) | FIXED | Removed the bogus `ZMPWalkConfig(command_vx=...)` test. P1.3-1.4 correctly widen `generate(command_vx, command_vy=0.0, command_yaw_rate=0.0)`. |
| **C2** (invented `get_obs_layout`/`slot_size` API) | FIXED | P7.3 tests use `build_policy_spec`, `build_observation_from_components`, and check `obs.shape == (spec.model.obs_dim,)`. Real APIs. |
| **C3** (would break v1-v7 layouts) | FIXED | New approach appends a separate `velocity_cmd_lateral_yaw` slot only in the v8 dispatcher branch; legacy slot at line 151 is only narrowed (sliced to `[..., :1]`), preserving v1-v7 shape. See S5 above for a minor regression-test gap. |
| **C4** (`obs_version` vs `actor_obs_layout_id`) | FIXED | All YAML keys and references corrected. Env init guard at 225-239 is correctly extended in P7.4. |
| **C5** (`loc_ref_version` hard validator) | FIXED | Chose option (b): KEEP `loc_ref_version: v3_offline_library` and add a separate `loc_ref_command_axes_3d` toggle. No validator changes needed. |
| **C6** (invented `loc_ref_offline_command_vx_grid`) | FIXED | P5.2 adds `loc_ref_offline_command_vy_grid` and `loc_ref_offline_command_yaw_rate_grid` as new fields; vx grid still derived from `min/max_velocity + grid_interval`. P1.10/P1.11 add `build_library_for_3d_values`. |
| **C7** (re-implementing 3D lookup) | **FIXED — verified against source.** Read `control/references/reference_library.py:395-409` directly: `lookup(vx, vy=0.0, yaw_rate=0.0)` is already 3D nearest-L2. Plan deletes the duplicate implementation, keeps only a regression test (P2.1). Confirmed claim is correct. |
| **C8** (sampler not mirroring TB) | FIXED | P4.5 and P4.6 mirror TB `walk_env.py:256-305` literally: per-axis magnitude sampling, asymmetric limits, signed-magnitude turn branch. P4.7 mirrors the `random_number < zero_chance / + turn_chance` brancher. |
| **C9** (`velocity_cmd` callsite fan-out) | **PARTIALLY FIXED.** P0.5 enumerates most callsites. Verified by independent grep: plan enumerates 1218, 1837, 2051, 2531, 2868-2873, 2896-2898, 2905-2907, 2962-2965, 3217-3219, 3340/3359/3362/3383/3408 — these all exist (verified). **MISSED:** 1216 (`vx_err_cmd = forward_velocity - velocity_cmd`, the actual reward arithmetic — already a parameter so technically caller-managed via P6.3, but plan should mention), 1188-1195 docstring (needs update in P6.2), 2531 (sentinel comparator — covered in P3.6 but P0.5 lists it). 2658 (`_lookup_offline_window` caller), 2762/2816/2835/3124/3269/3303 (more `velocity_cmd=` arg propagations) — these are likely fine as scalar→3-vec pass-throughs but P0.5 should list them too. See NEW-1 for the P3.8 vs P6.2 conflict at 1218. |
| **C10** (path-state integration ignores vy/wz) | **PARTIALLY FIXED.** P3.9-P3.10 extend the function correctly in intent, but the closed-form math is hand-waved. See NEW-5. |
| **C11** (heading-frame correction missing) | **PARTIALLY FIXED.** P5.5 adds the correction step but uses scipy `Rotation` syntax against a JAX quaternion array. See NEW-3. |
| **C12** (pure-rotation arc radius vs pelvis-centered) | FIXED — verified against `toddlerbot/algorithms/zmp_walk.py:269-305`. Plan's P1.8 mirrors the TB formulas (`stride = wz * total_time / (2*n_cycles - 1)`, `r = sign(wz) * rotation_radius`, foot x/y formulas with `± foot_to_com_y * sin/cos`). ZMPWalkConfig.rotation_radius_m default 0.10 is documented with a clear derivation. |
| **C13** (eval_velocity_cmd sentinel migration) | FIXED in intent; broken by NEW-4. P3.6 correctly migrates the type and uses `jp.all(eval_cmd >= 0.0)` for the sentinel. But the YAML shim (P3.4) does not handle the `float()` cast at training_config.py:550 that breaks before the shim sees the list. |
| **C14** (cmd_deadzone migration) | FIXED in intent; broken by NEW-4. Same `float()` cast issue at training_config.py:549. |
| **C15** (step granularity) | FIXED. Each sub-step (P1.1, P1.2, ..., P7.4) has its own test → impl → verify cycle and is plausibly 2-5 min. The plan grew from 250 to 1588 lines as a result. |
| **C16** (`test_cmd_velocity_tracking_mode.py` rename) | **FIXED — verified against source.** Read all three test sites at lines 37/63/96 directly. They call `_cmd_forward_velocity_track_reward(forward_velocity=..., lateral_velocity=..., velocity_cmd=..., ref_forward_velocity=..., ref_lateral_velocity=..., alpha=..., track_dim=...)` with NO `vy_cmd` kwarg. With the plan's added `vy_cmd: jax.Array = jp.float32(0.0)` default, the existing calls pass through with vy_err_cmd = lateral_velocity - 0.0 = lateral_velocity — byte-identical to the current `vy_err_cmd = lateral_velocity` line. Three tests will continue to pass. Confirmed. |
| **C17** (parallelization claim wrong) | FIXED. Task Dependencies table at L88-104 explicitly serializes G3 → G4 → G5 → G6 with "single env file" justification. G7 (policy_contract/) is correctly noted as parallelizable with G4 once G3 is done. |

---

## New Issues Found in v2

(All listed above as NEW-1 through NEW-6.)

### Quick summary
- **NEW-1** (P3.8 vs P6.2 conflict on line 1218) — must fix; the plan as written has incompatible instructions across phases.
- **NEW-2** (v8 obs builder kwarg routing unclear) — must fix; need explicit "env passes full 3-vec to `build_observation_from_components`" rule.
- **NEW-3** (P5.5 uses scipy Rotation syntax against JAX array) — must fix; pseudocode will not compile.
- **NEW-4** (`float()` cast at training_config.py:549-550 breaks on list inputs) — must fix; YAML shim insertion point is wrong.
- **NEW-5** (P3.10 closed-form math is hand-waved) — must fix; the literal "vy*dt*y_sum" generalization is incorrect.
- **NEW-6** (P0.5 audit not enforceable, omits 3+ callsites) — nice-to-have but recommended.

---

## Recommended next pass

Before any code lands:
1. Resolve the P3.8 ↔ P6.2 conflict on line 1218 (NEW-1).
2. Make the env→`build_observation_from_components` data flow explicit in P3.8 (NEW-2): pin a rule "`_get_obs` receives the full 3-vec `velocity_cmd`; reward and metric paths see indexed slices."
3. Rewrite P5.5's heading-frame correction in JAX (NEW-3). Either use `policy_contract/jax/frames.py` quaternion primitives or extract yaw scalars and rotate via a 2D rotation matrix. Inspect `frames.py` first to confirm the available primitives.
4. Add an explicit replacement for the `float(...)` casts at `training_config.py:549-550` (NEW-4). The shim must be a function, not just a "broadcast if scalar" sentence.
5. Pin the 2D closed-form `path_pos` math in P3.10 (NEW-5). Add a derivation block with `X_cos_sum` and `X_sin_sum`, and add a regression test that `(vy=0, wz=0)` reproduces the pre-P3 `path_pos` byte-for-byte.
6. Strengthen P0.5 to either be a test-gated audit or list the additional callsites (1188 docstring, 1216 caller, 2658, 2762, 2816, 2835, 3124, 3269, 3303) — see NEW-6.
7. Address the 7 minor suggestions S1-S8 where possible.

Once those land the plan can be re-reviewed in a third pass. The architectural direction (single unified ZMP prior, 3-vec cmd, new v8 layout with appended `velocity_cmd_lateral_yaw` slot, TB-mirrored sampler, optional `vy_cmd` kwarg preserving the legacy test surface) is correct and ready — the remaining issues are surgical fixes to specific phases rather than design rework.
