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

**Finding — it depends on command magnitude (both probed, 500 steps × 32 envs):**

| eval cmd | `ref_selected_vy` | flip threshold δ | yaw_drift max |
|---|---|---|---|
| `(0.13,0,0)` | **0.0 for 100%** of samples (clean) | δ > 14.5° (`asin(0.0325/0.13)`) | 0.244 rad (14°) — just under |
| `(0.26,0,0)` | **±0.065 on 8.4%** of samples; **100% when `|yaw_drift|>0.20 rad`** | δ > 7.2° (`asin(0.0325/0.26)`) | 0.287 rad (16°) — over |

`ref_selected_wz` stays 0 in both (wz is unaffected by the heading rotation, as
designed).

**Conclusion (corrected): TWO things are true.**
1. **Primary cause is reward-structure**, NOT the lookup. The in-place/lateral
   sway happens at cmd 0.13 where the lookup is provably clean (vy=0 always),
   so the disease is the missing forward template (`feet_phase` + `alive` +
   posture satisfied in place; dead forward reward; `penalty_close_feet_xy`
   wide stance).
2. **The heading correction IS a real SECONDARY mechanism at high cmd.** At
   vx=0.26 the flip threshold drops to 7.2°, so under yaw drift the lookup
   feeds a **lateral prior** (vy=±0.065) — a positive feedback loop (yaw drift
   → lateral prior → policy steps lateral → more yaw/lateral) that reinforces
   lateral motion at the fast end of the command range. This likely explains
   why smoke3's visual at cmd 0.26 showed strong lateral *translation* (0.55
   m/s consistent) vs smoke4A at 0.13 showed milder in-place *sway*.

**Implications for smoke5:**
- The forward-template fix (option 1) remains the **primary** lever (addresses
  the cause that's present even at the clean low cmd).
- ALSO worth doing: **clamp / disable the heading-frame rotation when
  `|delta_yaw|` is large** (or restrict the lateral bins) so a yaw-drifting
  forward command can't snap to a lateral prior at high vx. Cheap, and removes
  the high-cmd feedback loop. May partly self-resolve once a real forward gait
  keeps yaw drift small — but don't rely on that.
- Keep `ref_selected_vy` on every run to confirm it stays ~0 once fixed.
