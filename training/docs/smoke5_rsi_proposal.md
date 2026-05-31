# smoke5 — Reference State Initialization (RSI) to break the cold-start forward basin (v2)

Status: **approved for implementation** (2026-05-30). Supersedes the chat-only v1.
Incorporates two external-review findings (full reset-state surface; required
residual-base alignment).

## 1. Problem
WR PPO walking (smoke1–smoke4A) cold-starts into a **march/sway-in-place** reward
optimum: survives, lifts feet (`feet_phase≈0.086`), but `forward_velocity≈0`,
`reward/cmd_forward_velocity_track≈0.004` (dead). The only forward-translation
reward (`cmd_forward_velocity_track`, Gaussian `exp(-α·(v−cmd)²)`, α=562.5) has
**near-zero value and gradient from rest**; every other active term is
direction-blind and satisfied in place. Probe confirmed the in-place basin is a
reward/start-state problem, not the reference lookup (clean `ref_selected_vy=0`
at cmd 0.13).

α/param tuning is ruled out: a from-rest gradient needs `α≲1/cmd²`, but at that α
standing pays `exp(-O(1))` → the smoke1 standing leak. No α gives a from-rest
gradient *and* no standing leak.

## 2. Why RSI (evidence)
TB mirrors WR's reward and **its velocity reward is also dead from rest** (σ=1000,
tighter than WR). TB's resolution is **RSI**: reset onto the *moving* reference
manifold. `~/projects/toddlerbot/.../mjx_env.py:900-909` draws a random reference
frame; `:1174-1190` sets `qvel` from the reference (root lin/ang vel + motor/joint
vel) and the phase at that frame → the robot starts mid-stride at ≈cmd, so the
(dead-from-rest) reward is **live (~1.0) at step 0**.

WR resets **static**: `wildrobot_env.py:3310` `qvel=zeros`, control seeded from
the residual base (`:3327`), all path/history state at frame-0/origin. So WR's
reward is dead at step 0 and the body is off the velocity manifold.

Feasibility: WR's library already exposes per-frame `body_lin_vel`, `body_ang_vel`,
`q_ref`, `body_pos`, `body_quat`, `phase_sin/cos`, foot pos/vel
(`_lookup_offline_window` return, `~1639-1660`). Joint velocity is finite-diffed
from `q_ref`. **No library regeneration.** The change is in the reset path only.

## 3. The full RSI reset-state surface (review finding 1 — REQUIRED)
WR's first step depends on more than physics state. RSI must fast-forward the
**entire** reset state to the drawn frame `f`, consistent with `win@f`:

| field | source @ frame f | why it matters |
|---|---|---|
| `qpos` root pos/quat | `body_pos`, `body_quat` | physical pose on the gait manifold |
| `qpos` actuated joints | `q_ref` (`_actuator_qpos_addrs`) | mid-stride joint config |
| `qvel` root lin/ang | `body_lin_vel`, `body_ang_vel` | **moving at ≈cmd** (makes reward live) |
| `qvel` actuated joints | finite-diff `q_ref` (`_actuator_dof_addrs`) | legs swinging |
| `loc_ref_offline_step_idx` | `f` | reference advances from f |
| `path_state_path_rot` | frame-f path heading (identity for forward-only) | **feeds step-0 heading correction** (`:3724`) |
| `path_state_torso_pos` | frame-f path position | reference world placement |
| `prev_root_pos/quat` | frame-f root | consistent step-1 finite-diff velocity |
| `prev_left/right_foot_pos` | frame-f foot pos | `feet_air_time` / swing metrics |
| `last_left/right_touchdown_x` | frame-f (last touchdown ≤ f) | `step_length` |
| `init_root_pos_xy`, `init_root_yaw` | frame-f root xy/yaw | `world_x_progress` / `yaw_drift` measured from RSI start |
| `loc_ref_*` gait/phase/stance, `v4_compat` | `win@f` (not `win0`) | obs + first-step gait state |
| proprio / critic history buffers | seed from frame-f bundle if cheap; else zeros | softer (history is past-only) |

Note: smoke5 is forward-only (vy=wz=0), so the reference path is straight →
`path_state_path_rot = identity` is correct at every frame, and the body heading
stays ≈0 → no spurious heading-correction lateral flip. (General/turning RSI would
need the true frame-f path heading.)

## 4. Residual-base alignment (review finding 2 — REQUIRED; refined to `ref_init`)
smoke4A uses `loc_ref_residual_base: home`. RSI **requires a reference base, not
`home`** — but the precise reason is **reachability**, not "coherent zero-residual":

- TB pairs RSI with a **STATIC frame-0 reference base** (`default_action =
  first_frame_ref["motor_pos"]`, `mjx_env.py:1225`, explicitly "not the random
  RSI frame"). So a static base under RSI is exactly TB's design — the policy is
  phase-conditioned and learns the per-frame action; the base does **not** follow
  the time-varying frame.
- The control base must be **on the gait manifold** so the bounded `±0.25` rad
  residual can span base→mid-stride poses. WR's `ref_init` (frame-0 reference
  q_ref) is on-manifold (gait deviates from it by ~its amplitude) — the analog of
  TB's `first_frame_ref`. `home` is a generic keyframe that can sit >0.25 from
  mid-stride frames → some RSI frames would be **unreachable** within the residual
  bound → the robot can't hold the gait.

So smoke5 sets `loc_ref_residual_base: ref_init` (NOT `home`; NOT `q_ref`, which
would re-introduce time-varying prior-following — a different paradigm). This is a
**required companion to RSI**, not an independent change. (env init raises if
`loc_ref_rsi_enabled` is paired with a `home` residual base.)

## 5. Implementation plan
- Add `env.loc_ref_rsi_enabled: bool` (default False → bit-identical to smoke4A).
- **Train-only**: `reset(perturb_pose=...)` does RSI; `reset_for_eval` keeps the
  static standstill reset (eval/deploy starts from rest — see §7).
- Draw `f ~ U[0, n_frames)` for the sampled command's bin; build `win@f`.
- Construct qpos/qvel + the §3 fields from `win@f`; align residual base per §4.
- Frame conventions (qvel linear/angular global-vs-local, library velocity frame)
  are **validated by test**, not assumed (§6).

## 6. Correctness gates (tests, TDD)
1. **Velocity-mapping test:** RSI-reset at frame f → measured root velocity (via
   `CAL.get_root_velocity`) ≈ reference `body_lin_vel@f` (validates the qvel frame
   mapping). If it fails, fix the frame convention before anything else.
2. **State-surface test:** RSI reset sets `step_idx==f`, `init_root_pos_xy`,
   `prev_root_pos`, `loc_ref_offline_step_idx` to frame-f values (not 0/origin).
3. **G6 zero-residual-from-RSI invariant:** a zero-residual rollout from an RSI
   frame reproduces the reference motion (within prior viability) — the cleanest
   end-to-end consistency check.
4. **Off-by-default:** `loc_ref_rsi_enabled=False` → reset byte-identical to
   smoke4A (regression guard).
5. `ref_selected_vy ≈ 0` under RSI forward-only (heading stays consistent).

## 7. Open risk: eval/deploy starts from rest
RSI bootstraps "continue walking"; G4 eval (and deployment) start from rest at
`eval_velocity_cmd=[0.13,0,0]`. The RSI policy must learn the **rest→walk**
transition too. Mitigations: keep eval static (no RSI); include the standing/
frame-0 in the RSI frame distribution and/or keep `cmd_zero_chance` rest episodes.
If G4-from-rest fails despite walking under RSI, add an explicit standing→walk
curriculum (separate follow-up).

## 8. Experiment design
smoke5 = **smoke4A + RSI only** (isolated), plus the required residual-base
alignment. Config: `ppo_walking_v0210_smoke5_rsi.yaml`. Keep everything else from
smoke4A (1D legacy sampler, `cmd_velocity_track_dim=1`, floor 0.065/max 0.26/
interval 0.065, α=562.5, obs_v8, critic actor+privileged). Early-stop at 3–5M on
`forward_velocity>0` / `reward/cmd_forward_velocity_track≥0.01` / `step_length>0`
/ `world_x_progress>0`.

## 9. Out of scope
- Heading-correction clamp (root-cause #2): secondary; keep `ref_selected_vy/wz`
  logged to monitor.
- Reward-shape forward-progress term: the alternative fix family (reward-side);
  fallback if RSI fails/infeasible.

## 10. Key references
- WR reset: `_make_initial_state` (~3299), `reset` (~2685), `ref_init` qpos build
  (~2745), residual/reset base (~350/363).
- WR per-frame data: `_lookup_offline_window` (~1466, return ~1639).
- TB RSI: `~/projects/toddlerbot/toddlerbot/locomotion/mjx_env.py:900-909, 1174-1190`.
- Base config: `ppo_walking_v0210_smoke4_forward_first.yaml`.
