# v0.21.0 lateral+yaw prior plan — Adversarial review pass #1

VERDICT: NEEDS_REVISION

## Summary Assessment

The plan picks the right architectural direction (single unified ZMP prior, command widened 1D→3D, TB-mirrored sampler and reward) and the WR-scaled velocity/yaw ranges look sound, but several plan steps reference APIs / config fields / file structures that do not exist in the current WR codebase, the sampler does not actually mirror TB's, and the plan silently ignores a number of call-site fan-outs (scalar→vector `velocity_cmd`, path-state integration, eval-cmd sentinel, existing test breakage) that will cause TDD steps to fail in non-obvious ways. Several "parallelizable" claims and granularity claims are also incorrect.

---

## Critical Issues (must fix)

### C1. `ZMPWalkConfig` does NOT have a `command_vx` field
- `control/zmp/zmp_walk.py:31` defines `ZMPWalkConfig` with morphology / gait timing fields but **no `command_vx`**. `command_vx` is a *parameter to* `ZMPWalkGenerator.generate(command_vx)` at line 579 — not a constructor kwarg.
- The plan's P1.1 failing test `ZMPWalkConfig(command_vx=0.20, command_vy=0.05, command_yaw_rate=0.10)` will fail with `unexpected keyword argument 'command_vx'`, NOT the intended `unexpected keyword argument 'command_vy'`. The plan's predicted failure mode is wrong, and P1.2 must either (a) add all three as config fields and adjust `generate()` to read them, or (b) keep them as `generate()` parameters and rewrite the test accordingly. The plan does neither — it just adds `command_vy/command_yaw_rate` as if `command_vx` were already there.

### C2. Plan invents a `get_obs_layout(...)` / `layout.slot_size(...)` / `_obs_layout.slice_obs(...)` API that does not exist
- `policy_contract/jax/obs.py` has no such functions. Layout dispatch is by string match on `spec.observation.layout_id` (lines 46-51, 153-192). Layout sizes are validated only against the static expected list in `policy_contract/spec.py::_validate_observation`.
- The only related utility is `policy_contract/layout.py::get_slices(observation)` (file is 17 LOC).
- Every test in P4.1 and P4.3 that calls `get_obs_layout("...")` / `layout.slot_size(...)` / `env._obs_layout.slice_obs(...)` is testing imagined APIs. These tests will not collect.

### C3. P4.2's `obs.py` change would break ALL existing layouts
- Line 151 of `obs.py` (`jnp.atleast_1d(velocity_cmd).reshape(1)`) is in the **shared `parts` list assembled before any layout-specific branch**. The plan says "Replace ... in the v8 branch" but no per-layout branch exists for this slot — replacing it unconditionally with `.reshape(3)` would break v1/v2/v3/v4/v6/v7 (and the spec-validation gate at `spec.py:339,357,376,395,429,455` which asserts `velocity_cmd` size = 1 for these layouts).
- The plan must explicitly branch: keep the shared 1D slot for v1-v7 and append a 3D slot only when `layout_id == "wr_obs_v8_cmd3d"`. It must also (a) add `"wr_obs_v8_cmd3d"` to `SUPPORTED_LAYOUT_IDS` in `policy_contract/spec.py:11` and to the supported set in `obs.py:46`, (b) add a new `_validate_observation` branch defining the v8 expected layout list, (c) update `policy_contract/spec_builder.py` to assemble the v8 layout.

### C4. The "`obs_version`" config field does not exist — the real field is `actor_obs_layout_id`
- `grep -rn obs_version training/configs/` returns nothing; `training_runtime_config.py:119` defines `actor_obs_layout_id: str = "wr_obs_v1"` and smoke14 uses `actor_obs_layout_id: wr_obs_v7_phase_proprio` (line 122).
- The plan's YAML diff (`env.obs_version: wr_obs_v8_cmd3d`) is invalid — YAML loaders will silently drop the unknown key and the env will keep `wr_obs_v1`. The plan's references in P4.3 and P7.4 to "`obs_version`" and "dispatch already routes by `obs_version`" are likewise wrong.
- Fix: rename every plan reference from `obs_version` to `actor_obs_layout_id`.

### C5. `loc_ref_version` validation hard-rejects unknown strings — `v3b_offline_library_3d` will crash init
- `wildrobot_env.py:219-222` raises if `loc_ref_version != "v3_offline_library"`. The plan's smoke YAML sets `loc_ref_version: v3b_offline_library_3d`, which will fail in `__init__`.
- Fix: either (a) expand the allow-list in the env to accept the new string and switch on it, or (b) keep `v3_offline_library` and key the 3D behavior off `loc_ref_offline_command_vy_grid` being non-trivial. The plan needs to pick one explicitly.

### C6. The plan invents `loc_ref_offline_command_vx_grid` and reads it as the source of truth
- Today's grid is built by `_init_offline_service` (lines 820-840) from `min_velocity`/`max_velocity`/`loc_ref_command_grid_interval` plus the unioned `loc_ref_offline_command_vx`. There is no `loc_ref_offline_command_vx_grid` field.
- The plan's smoke YAML sets `loc_ref_offline_command_vx_grid: [0.18, 0.22, 0.26]` (along with new `_vy_grid` and `_yaw_rate_grid` lists) as if it were the existing 1D mechanism. P2.5 says "if `EnvConfig.loc_ref_offline_command_vy_grid` / `_yaw_rate_grid` are non-empty, pass them into `ZMPWalkGenerator` and `ReferenceLibraryMeta`" — but `ZMPWalkGenerator()` is invoked via `build_library_for_vx_values(list(vx_grid))` (env line 847), which iterates only `vx_values`. No `ZMPWalkGenerator` API today takes any vy/wz grid.
- Fix: add new fields, document that they are the 3D analog of the existing computed `vx_grid`, and add a new `build_library_for_3d_values(vx_values, vy_values, yaw_rate_values)` API on `ZMPWalkGenerator` — none of this is in the plan today.

### C7. The plan's lookup architecture is upside-down — `lookup_jax` does NOT take a cmd vector
- `runtime_reference_service.py:433` defines `lookup_jax(step_idx, jax_arrays)` with only step_idx. Bin selection (`argmin(|vx_grid - velocity_cmd|)`) lives in `wildrobot_env._lookup_offline_window` (1300-1306), and the leading n_bins axis is stacked at the env layer (lines 888-897) — `to_jax_arrays` returns single-bin arrays.
- The plan's P2.4 ("Update `lookup_jax(cmd_vx, cmd_vy, cmd_wz)` to return `argmin(L2(cmd_keys - cmd))`") and the corollary stacking-inside-the-service text are wrong about where the lookup happens. The actual change must be in `_lookup_offline_window` (swap `_offline_vx_grid` for an `_offline_cmd_keys: (n_bins, 3)` array, swap `jp.abs(...)` for `jp.linalg.norm(... axis=-1)`).
- The plan also misreads `ReferenceLibrary.lookup` — it is already 3D (`reference_library.py:395-409` accepts `(vx, vy=0.0, yaw_rate=0.0)` and does L2 nearest). The plan's P2.3 claim "currently 1-D over `vx`" is wrong.

### C8. The plan's sampler is NOT a mirror of TB's
- TB's `sample_walk_command` (toddlerbot/locomotion/walk_env.py:256-280) does **per-axis** magnitude sampling: `x = U[deadzone[0], x_max] * sin(theta)`, `y = U[deadzone[1], y_max] * cos(theta)` — i.e. theta sets direction, but the radius is sampled *independently* on each axis between deadzone and asymmetric limit, **not** as a single radial scale `r ~ U[0.5, 1.0]`.
- TB also exposes asymmetric limits (`command_range[5][0]` vs `command_range[5][1]`) and uses `jp.where(sin(theta)>0, hi, -lo)`.
- The plan's `r = U[0.5, 1.0]` ellipse is closer to a fixed-radius annulus than TB's behavior, and silently changes the (vx, vy) marginal distribution. P7.2's stdev test will also fail because the plan's sampler is not actually elliptical-uniform — it concentrates mass at `r ≥ 0.5`.
- Fix: either mirror TB exactly (per-axis sampling with shared theta) or call out the deliberate divergence in the plan and update P7.2 to test the actual marginal.
- Separately, TB samples deadzone-respecting magnitudes; the plan adds a `cmd_deadzone: [0.02, 0.02, 0.05]` list but never threads it into the sampler. The current scalar `cmd_deadzone: float = 0.0` (config line 625) needs a type migration the plan doesn't acknowledge.

### C9. `WildRobotInfo.velocity_cmd` shape and call-site fan-out are completely glossed over
- `env_info.py:107` declares `velocity_cmd: jnp.ndarray  # ()` (scalar). Shape registry asserts `()` at line 285. Many call sites assume scalar:
  - `wildrobot_env.py:1837` `(jp.abs(velocity_cmd) > 1e-6)`
  - `wildrobot_env.py:2051` `is_standing = jp.abs(velocity_cmd) < 1e-6`
  - `wildrobot_env.py:2868-2871` reward path: `root_vel_h.linear[0] - velocity_cmd`
  - `wildrobot_env.py:2896-2898` `tracking/velocity_cmd_abs`, `_nonzero_frac`
  - `wildrobot_env.py:2905-2907` `tracking/forward_velocity_cmd_ratio`
  - `wildrobot_env.py:2531` eval reset `eval_cmd >= 0.0` (scalar comparator vs sentinel)
  - `compute_command_integrated_path_state(velocity_cmd_mps=velocity_cmd, yaw_rate_cmd_rps=0.0, ...)` at 2962-2965
- Changing the field to `(3,)` requires touching all of these. Plan steps P3 and P5 do not mention any of them. The TDD tests in P3.3 will compile, but a downstream `_make_initial_state` or `_compute_reward_terms` call will broadcast-fail or silently produce wrong arithmetic (`jp.abs(velocity_cmd)` on a (3,) gives a (3,)).
- Existing tests likely to break: `tests/test_v0201_env_zero_action.py:221`, `tests/test_runtime_reference_service.py:188,232,269` all pass `velocity_cmd_mps=...` as scalar; `training/tests/test_cmd_velocity_tracking_mode.py:37,63,96` call `_cmd_forward_velocity_track_reward` directly. P5.2 *renames* this static and changes the signature, so all three of those tests break — P7.3 says "No code changes; just rerun" which is wrong.

### C10. Path-state integration ignores vy and wz — `r_torso_pos_xy` will diverge
- `wildrobot_env.py:2962-2965` calls `compute_command_integrated_path_state` with `velocity_cmd_mps=velocity_cmd` (scalar vx) and hard-coded `yaw_rate_cmd_rps=0.0`. The `path_state["torso_pos"]` is the target the reward at line 1695-1700 (`r_torso_pos_xy`) compares against root_pos_xy.
- If the policy is commanded vy=+0.10 m/s for an episode, `torso_pos_xy[1]` will stay at zero while the actual root drifts +y for the strafe — the reward will fight the strafe behavior. Same problem for wz with `torso_pos_xy` angular target.
- This is the most likely reason a v0.21.0 smoke would *appear* to "not learn lateral motion" despite the sampler / library / obs being correct. Plan must include a step to extend `compute_command_integrated_path_state` (and probably TB's `integrate_path_state` analog in `walk_zmp_ref.py`) so vy and wz are integrated too — or to gate `r_torso_pos_xy` off when |vy| or |wz| > 0.

### C11. Heading-frame correction is missing from the 3D lookup
- TB's `walk_zmp_ref.py:132-137` rotates the (vx, vy) command into the heading-error frame before the bin lookup (`heading_rot.inv() * path_state["path_rot"]` then `apply([vx, vy, 0])`). Without this, an actor that drifts +0.05 rad in yaw will pick the wrong reference bin.
- The plan's P2.5 just says "pass current `(vy_cmd, yaw_rate_cmd)` to `lookup_jax`" — no heading correction, no path_rot read. This is a soft-correctness bug that will manifest as bin chatter on real rollouts.

### C12. Pure-rotation TB branch uses `rotation_radius`, not `pelvis-centered yaw`
- TB at zmp_walk.py:269-305 plans footsteps **on a circular arc of radius `rotation_radius` (default 0.1) about a point lateral to the pelvis**, not pelvis-centered rotation. The plan's P1.5 says "rotate about the pelvis center by `yaw_rate * cycle_time` per cycle" — this is geometrically different (TB's foot translates *and* rotates; pelvis-centered would zero foot translation).
- Also TB's stride formula for yaw is `stride = yaw_rate * total_time / (2 * num_cycles - 1)` (angular delta *per footstep*), not `yaw_rate * cycle_time` per cycle.
- Plan must (a) add a `rotation_radius_m` config field with a WR-tuned default (WR's hip_lateral_offset_m = 0.0536, leg length 0.373; TB's foot_to_com_y = 0.042, so a TB-scaled value would be ~0.1 × (0.0536/0.042) or ~0.13), and (b) port TB's actual stride formula and arc geometry, not the simplified pelvis-center rotation the plan describes.

### C13. `eval_velocity_cmd` type migration is undefined
- Today: `eval_velocity_cmd: float = -1.0`; reset uses `jp.where(eval_cmd >= 0.0, eval_cmd, wr.velocity_cmd)`.
- Plan: `eval_velocity_cmd: [0.18, 0.04, 0.10]` (list).
- Sentinel comparison `[-1, -1, -1] >= 0.0` is element-wise vector — the existing `jp.where(eval_cmd >= 0.0, ...)` no longer reduces to a single bool and the override semantics break silently. Plan must redefine the sentinel (e.g. a separate `eval_velocity_cmd_override: bool` flag, or a tuple-typed sentinel like `(-1.0, -1.0, -1.0)` with an explicit `jp.all(eval_cmd >= 0.0)` reduction).

### C14. `cmd_deadzone` type migration is undefined
- Today: `cmd_deadzone: float = 0.0` (scalar). The sampler uses `jp.abs(cmd) < deadzone` (scalar). YAMLs set scalar values (smoke14: `cmd_deadzone: 0.0666666667`).
- Plan: `cmd_deadzone: [0.02, 0.02, 0.05]` (3-vector). The plan's P3 step does not touch `EnvConfig.cmd_deadzone`'s type, nor update the scalar `jp.abs(cmd) < deadzone` check in `_sample_velocity_cmd` (line 2569, 2578).
- Fix: explicit type bump + scalar→vector deadzone application per axis, and a compatibility shim that broadcasts a single float across all three axes when older YAMLs feed a scalar.

### C15. TDD step granularity violation — P1.4, P1.5, P2.4, P3.4, P5.3 each bundle multiple hours of work
- The writing-plans rule is **2-5 minutes per step**.
  - **P1.4** ("Implement lateral stride in footstep planner"): requires mirroring TB's combined-translation branch (left/right alternation, stride normalization across `2*num_cycles - 1`, asserting it composes with the existing single-axis `_generate_at_step_length`, then updating the COM/ZMP planner so the LQR follows the new footstep series). Easily a multi-hour edit.
  - **P1.5** ("Pure-rotation footstep branch"): porting the TB `rotation_radius` arc geometry + pelvis quat accumulation + ZMP planner adjustment is at least an hour by itself.
  - **P2.4** ("`to_jax_arrays()` widens the key"): really a multi-file refactor of the bin-stacking machinery (see C7) plus regenerating `_offline_vx_grid → _offline_cmd_keys` plumbing.
  - **P3.4** ("branched sampler mirroring TB"): see C8 — should be 3+ tasks (sample-walk, sample-turn, branch-select, deadzone-apply).
  - **P5.3** ("Add yaw-rate tracking reward term"): metric registration, weight wiring, alpha selection, integration into `_compute_reward_terms`, regression-guard reward serialization update.
- Recommendation: split each of these into 4-6 atomic steps (one per concept), each with its own failing test.

### C16. P7.3 regression-guard step is contradicted by P5.2
- P5.2 renames `_cmd_forward_velocity_track_reward → _cmd_velocity_track_reward` and changes its signature (adds `vy_cmd`). P7.3 says `test_cmd_velocity_tracking_mode.py` "should pass after P5 changes — no code changes". The three tests at lines 37/63/96 call the OLD name and OLD signature. They will all break.
- Fix: P5.2 must include an explicit step to update `test_cmd_velocity_tracking_mode.py` (and `tests/test_runtime_reference_service.py` if any of its `velocity_cmd_mps=` callers see the new shape).

### C17. Cross-file dependency claim for G3 is wrong
- G3 says "P3 in `wildrobot_env.py::_sample_velocity_cmd`; P4 in `policy_contract/jax/obs.py`" — parallelizable.
- But: P4.3 explicitly modifies `wildrobot_env.py::_get_obs` (line 1439) to dispatch v8, and P4.1's `test_env_with_v8_layout_produces_obs_with_3d_cmd` test requires the env to construct a v8 layout. **P4 touches both `obs.py` AND `wildrobot_env.py`** — the same file as P3.3 / P3.4. Running G3 in parallel would create merge conflicts in `wildrobot_env.py` (sampler edits at 2539+ AND obs-dispatch edits at 1439+).
- Fix: either serialize P3 → P4 (cheap, the files overlap), or scope P4.3 narrowly to a non-overlapping helper and explicitly call out the merge-isolation contract.

---

## Suggestions (nice to have)

### S1. Add an early "scaffolding" task before P1
A 30-minute "wiring tour" step that lists ALL `velocity_cmd` references in env / contract / tests (the `grep -n velocity_cmd` output) and turns each into a TODO checkbox would catch C9/C13/C14 before they cause TDD pain. Recommend adding as P0.5.

### S2. Document the path_state question explicitly even if deferring
C10 (torso target integration) is large enough that it might be deferred to v0.21.1. If so, the plan should explicitly say so AND add a guard test that zeros `r_torso_pos_xy` whenever `|vy_cmd| > eps OR |wz_cmd| > eps` so the reward doesn't fight the policy. Otherwise the smoke is set up to be misleading.

### S3. Library size warning
3 vx × 5 vy × 5 wz = 75 bins, but each bin has full `(n_steps, n_joints)` arrays. With smoke14's `n_steps` around 1500-2000 and `n_joints` ~13, the stacked `q_ref` is ~75 × 2000 × 13 × 4 bytes = ~7.8 MB per field; with 16 fields ≈ 125 MB on device per env. With brax's vectorized environments (8192 in smoke14), this is multiplicative-free (shared constant), so it's fine, but call it out so reviewers don't worry.

### S4. ToddlerBot cycle_time fact-check
Plan's scaling table lists TB cycle_time = 0.72 s, WR = 0.96 s. Verified at `control/zmp/zmp_walk.py:88` (WR) and TB's `walk.gin`. OK.

### S5. `cmd_turn_chance` historical context
Plan claims P3 "activates `cmd_turn_chance`". `training_runtime_config.py:624` already has the field (default 0.0, comment "placeholder"). The plan is correct that no code path uses it today — P3 is the first place it would have a wiring effect. Worth noting in the CHANGELOG entry to set reader expectations.

### S6. P8.2 G6/G7 promotion thresholds
"|lateral_velocity - vy_cmd| < 0.5 * |vy_cmd|" with vy_cmd = 0.04 means tolerance of ±0.02 m/s — that's tight for a first smoke. Recommend starting at the smoke14-style "≥50% of cmd" floor (i.e. lateral_velocity >= 0.5 * vy_cmd, signed) to avoid spurious FAILs on the first run.

### S7. Use existing 3D fields on `ReferenceTrajectory`
`reference_library.py:63-64` already has `command_vy` and `command_yaw_rate` defaulted to 0.0 — confirmed. The plan's P1.6 says "verify" which is correct here. Reads as a positive — keep.

### S8. Document the `RewardWeightsConfig.cmd_yaw_rate_track` weight default
Plan adds the weight (P5.3) with "default `0.0` so old configs ignore it". Make explicit in the CHANGELOG section that this is a backward-compatibility default, not a tuning recommendation — readers may otherwise infer that 0 is a chosen weight.

---

## Verified Claims (confirmed correct)

- File `/Users/ygli/projects/wildrobot/control/zmp/zmp_walk.py` exists (1340 LOC) — path is accurate.
- File `/Users/ygli/projects/wildrobot/control/references/reference_library.py` exists with `ReferenceTrajectory.command_vy` (line 64) and `command_yaw_rate` (line 65) already defined — P1.6's "verify" claim is true.
- File `/Users/ygli/projects/wildrobot/training/envs/wildrobot_env.py` line numbers in the plan match: `_cmd_forward_velocity_track_reward` at 1164 (plan says 1163 — off by 1), `_get_obs` at 1439 (matches), `_sample_velocity_cmd` at 2539 (matches), `vy_cmd = 0` hard-code at 1218 (plan says 1217-1218 — matches).
- TB combined-translation branch at `toddlerbot/algorithms/zmp_walk.py:259` uses `stride = command[:2] * total_time / (2 * num_cycles - 1)` and *adds* it to nominal stance offset with left/right alternation — plan's description is correct for this slice.
- TB command layout `[pose_command(5), vx, vy, wz]` at walk_env.py:317 — plan's "(vx, vy, wz) ordering" mirrors TB's [5,6,7] slice.
- `uv run pytest training/tests/test_xxx.py -q` is a valid invocation: both `tests/conftest.py` and `training/tests/conftest.py` add the repo root to `sys.path`, and `pyproject.toml` declares `pytest>=9.0.2` as a dev dependency. No `[tool.pytest.ini_options]` overrides exist that would conflict.
- WR-scaled command ranges (vy ±0.13 m/s, wz ±0.25 rad/s) are internally consistent with the plan's scaling table — derivation is documented.
- `cmd_velocity_track_dim` is wired and tested (`training_runtime_config.py:1001`, tests `test_cmd_velocity_tracking_mode.py:17-25`) — plan's reuse plan for dim=2 in smoke is feasible.
- Smoke14 baseline at `training/configs/ppo_walking_v0201_smoke14.yaml` exists and uses `loc_ref_command_conditioned: true` (line 153), `actor_obs_layout_id: wr_obs_v7_phase_proprio` (line 122), and `cmd_velocity_track_dim: 1` (line 246). The plan's "clone smoke14 and apply diffs" is structurally sensible.
- G4/G5 gate definitions exist in `training/docs/walking_training.md` (around line 877-889) — appending G6/G7 in a new section is the right pattern.
- `ReferenceLibraryMeta` is at line 341 (plan says 340 — off by 1) and `ReferenceLibrary.lookup` at line 395. The latter ALREADY accepts `(vx, vy=0.0, yaw_rate=0.0)` — plan should treat this as a positive, not as work-to-do.

---

## Recommended next pass

Before any code lands, the plan needs:
1. A new "scaffolding audit" task (S1) listing every `velocity_cmd` call site and a per-call-site disposition (kept scalar via index slice, widened to vector, removed).
2. A correctness re-pass on P1 (planner `command_vx` field vs parameter; rotation_radius config), P2 (lookup ownership and existing 3D lookup), P3 (sampler matching TB and deadzone vector), P4 (layout registry vs string-dispatch reality), P5 (rename + signature-change sweep includes existing tests).
3. Disambiguate the `loc_ref_version` migration story (C5).
4. Migrate `eval_velocity_cmd` and `cmd_deadzone` types explicitly (C13, C14).
5. Decide on path-state integration (C10) — either implement or guard.
6. Re-split the 5 oversized TDD steps (C15) and reverify all "parallelizable" claims against actual file overlap (C17).

Once those land the plan can be re-reviewed.
