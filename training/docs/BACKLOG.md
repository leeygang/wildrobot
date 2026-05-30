# WildRobot Training — Backlog

Queued decisions and investigations, newest first. Promote an item to a
config/run when its blocking question is resolved.

---

## [QUEUED 2026-05-30] smoke5 direction — how to break the forward-walking basin

**Status:** parked pending the `ref_selected_vy/wz` instrumentation read (see below).

**Context.** Cold-start prior-free PPO reliably converges to a **step/sway in
place** gait (smoke1/2/3/4A), because the only reward term that rewards forward
*translation* (`cmd_forward_velocity_track`, α=562.5) is dead from a standing
start, while every other active term — dominant `feet_phase` (swing-height,
direction-agnostic), `alive`, posture penalties, and `penalty_close_feet_xy`
(wide stance) — is fully satisfied in place. There is **no forward template**
(all `ref_*` imitation weights are 0, smoke12b prior-free recipe). The only WR
run that ever produced forward walking (smoke12b `ot8zazo1`) was **warm-started**
at iter 1720 / 35.2M off the smoke9c→11→12 curriculum chain — **no WR run has
broken the basin from a cold start.**

**Three candidate directions (decide after the instrumentation read):**

1. **Restore the forward template (cold re-test).** Re-enable `ref_q_track`
   (joint template = forward gait) + `torso_pos_xy` (prior's forward-advancing
   pelvis → directly penalizes in-place), AND switch
   `loc_ref_residual_base: home → q_ref` so the template is reachable within the
   ±0.25 residual. Config-only; directly tests the diagnosis. Cheapest
   high-information step.
2. **Warm-start from smoke12b** (the only proven forward gait). Needs 1D→3D
   obs-translation engineering. Highest confidence, more work, starts from a
   1D-trained policy.
3. **Curriculum cold (smoke9c-style).** Replicate the velocity/stability
   curriculum that originally built the gait. Proven mechanism, no obs
   translation, but most infra.

**Lean:** (1) as the next cold experiment (tests whether a forward template
breaks the in-place basin), with (2) as the fallback if cold still fails.

---

## [RESOLVED 2026-05-30] Instrumentation: ref_selected_vy/wz (commit 3926660) + probe read

Logged `tracking/ref_selected_vy` + `tracking/ref_selected_wz` from the
per-step **heading-corrected** lookup, then probed the latest smoke4A
checkpoint (`checkpoint_460`) with a JAX rollout (reuses
`eval_policy._collect_eval_rollout`; probe at `/tmp/probe_ref_selected.py`).

**Finding (cmd `(0.13,0,0)`, 500 steps × 32 envs):**
- `ref_selected_vx = 0.13` always; **`ref_selected_vy = 0.0` and
  `ref_selected_wz = 0.0` for 100% of samples** (max|.| = 0; only bin hit = 0).
- This held despite real `yaw_drift` up to **0.244 rad (14°)** and `lat_vel`
  up to 0.69 m/s.

**Conclusion: the heading-correction-feeds-a-lateral-prior hypothesis is
REFUTED at the eval command.** The lookup stays on the `vy=0, wz=0` row. The
math: at vx=0.13 the bin only flips to vy=0.065 once `0.13·sin(δ) > 0.0325`,
i.e. δ > 14.5° — drift peaked at 14°, just under. ⇒ the in-place/lateral
**sway is purely reward-structure** (no forward template; `feet_phase` +
`alive` + posture satisfied in place; dead forward reward; `penalty_close_feet_xy`
wide stance), NOT the reference lookup.

**Residual edge (keep the metric on to watch):** the flip threshold scales as
`δ > asin(0.0325/vx)` → at the 0.26 ceiling it drops to ~7.2°, which yaw drift
could occasionally cross. Not the driver of the observed failure (which occurs
at 0.13 where the lookup is clean), but `ref_selected_vy` now surfaces it if it
ever happens during training.

⇒ **smoke5 direction now points cleanly at the reward-structure fix (restore
the forward template — option 1 above), not the lookup.**
