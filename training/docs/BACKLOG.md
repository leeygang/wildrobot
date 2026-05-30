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

## [IN PROGRESS 2026-05-30] Instrumentation: ref_selected_vy/wz — DONE (commit 3926660)

Logged `tracking/ref_selected_vy` + `tracking/ref_selected_wz` from the
per-step **heading-corrected** lookup. **Open question to read off the next run
(or a smoke4A resume):** does `ref_selected_vy` stay ~0, or does the heading
rotation `R_z(yaw_path − yaw_heading)` land forward-only `(vx,0,0)` commands on
a `vy≠0` bin under yaw drift — feeding the policy a lateral prior and
reinforcing the lateral sway? `ref_selected_vy ≠ 0` ⇒ that mechanism is live
and must be addressed (e.g. clamp the heading correction, or restrict the
lateral bins) before/with the smoke5 design change. `~0` ⇒ rule it out; the
in-place/lateral failure is purely reward-structure (no forward template).
