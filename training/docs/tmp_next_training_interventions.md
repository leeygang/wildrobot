# Temporary ranked intervention list to unblock next walking training

Date: 2026-05-17
Status: temporary tracking note; do not treat as changelog

## Scope

This note ranks the next interventions for WR walking based on:
- current WR code and run evidence (`smoke9c` / latest run)
- ToddlerBot code as source of truth for the reference recipe
- explicit avoidance of additional hybridization

Primary code references:
- WR action anchor / residual compose: `training/envs/wildrobot_env.py::_compose_target_q_from_residual`
- WR current home source: `training/envs/wildrobot_env.py:299,405`
- WR current ref-init source: `training/envs/wildrobot_env.py:711`
- TB fixed default action: `~/projects/toddlerbot/toddlerbot/locomotion/mjx_env.py:1222-1225,1544-1545`
- TB home/default qpos source: `~/projects/toddlerbot/toddlerbot/reference/motion_ref.py:96`
- TB walk reference static-at-t0 behavior: `~/projects/toddlerbot/toddlerbot/reference/walk_zmp_ref.py:126,145`

Current observed failure signature:
- latest WR run reaches survival horizon but not walking
- `ep_len ~= 500`, `forward_velocity ~= 0`, `cmd_err ~= 0.105`, `step_len ~= 0`
- TB-style reset perturbation improved symmetry, but did not produce stride
- this is a stable non-walking basin, not a delayed walking basin

## Ranked intervention list

### 1. Highest-confidence fix

> **Status (2026-05-17)**: §1.1 (home pose promotion) landed in commits
> `a015d6b` (initial migration) and `3f9ef17` (review-feedback
> follow-up: honest derivation docstring, MJCF-range clip at write
> time, runtime + export-bundle loader sanity tests, standing-side
> home-consumer sanity test, vacuous-assertion fix).  §1.2 (smoke10
> config — TB-style fixed-home branch with smoke9c reward/PPO/cmd
> kept identical) landed as
> `training/configs/ppo_walking_v0201_smoke10.yaml` in `f7fa889`.
> Run smoke10 next; the trajectory comparison + decision on §2.x
> escalation follows.  CHANGELOG.md will be updated once smoke10
> results are confirmed (per AGENTS.md rule on training-result
> changelog entries).

#### 1.1 Promote a locomotion-ready crouched `home` pose and rerun the TB-style fixed-home branch

Reasoning:
- TB uses one consistent fixed nominal pose (`home`) as the action anchor.
- WR current global `home` is near-straight, while WR `ref_init_q` is crouched.
- This makes the obvious TB-aligned branch (`loc_ref_residual_base: home`) weak or unreachable under the current residual limits.
- If the new derived phase-neutral crouched pose is installed repo-wide as `home`, this removes the biggest known WR/TB geometry mismatch without introducing a new long-lived pose concept.

Required checks before promotion:
1. static/home sanity
2. gait-envelope coverage under `home ± residual_scale_per_joint`
3. no obvious regression in standing/runtime home consumers

Next training branch after promotion:
- `loc_ref_residual_base: home`
- `loc_ref_reset_base: home`
- keep TB-style reset perturbation
- keep current normalized smoke9c reward block for the first clean comparison
- do **not** simultaneously change reward formulas, PPO hyperparameters, and command distribution

Why this ranks first:
- it removes a concrete, code-confirmed WR/TB mismatch
- it keeps the experiment simple
- it gives a clean answer whether the straight-home confound was the main blocker

#### 1.2 Keep the action-path experiment coherent

For the next run, change the nominal action anchor only.

Do not mix in new imitation rewards or new reward formulas in the same run. The first post-migration run should answer one question:

> Does a TB-style fixed crouched home anchor move WR back toward a walking trajectory?

### 2. Medium-confidence fixes

#### 2.1 If fixed crouched `home` still stalls, reduce action-path dependence on `ref_init` semantics entirely

If the post-migration `home` branch still converges to quiet survival, the next suspect is not pose geometry but the residual-control contract itself.

Candidate follow-up:
- preserve fixed `home`
- simplify the action interpretation further toward TB semantics where possible
- keep the prior/reference available for reward/diagnostics, but not as the action-basin organizer

Interpretation:
- keep useful reference structure
- stop using reference-derived anchors to define the easiest policy basin

#### 2.2 Revisit command-tracking pressure only **after** the home migration result is known

If the fixed-home branch still survives without locomotion, then inspect:
- command curriculum
- zero-command fraction in the WR reward landscape
- velocity reward width / gradient shape

This is medium-confidence, not first-line, because:
- zero-command and action-rate arguments alone do not explain the WR/TB difference
- TB also tolerates fixed-home zero-command episodes and still learns
- the cleaner first test is the home-pose/control-contract fix

#### 2.3 Keep reset perturbation; do not revert it based on the latest run

Evidence from the latest run:
- old WR failure: one-sided shuffle
- latest WR failure: more symmetric touchdown activity

Interpretation:
- reset perturbation improved exploration shape
- it was not sufficient by itself
- reverting it would remove a positive change without solving the main problem

### 3. Things to explicitly avoid

#### 3.1 Do not keep accumulating pose concepts (`home`, `ref_init`, `walk_home`, `nominal_policy_pose_q`) unless forced

Reason:
- the repo is already close to semantic overload around default/home/reference poses
- if the derived phase-neutral crouched pose is validated, prefer one canonical repo-wide `home`
- keep `ref_init_q` only for its literal meaning: offline frame-0 `q_ref`

#### 3.2 Do not redefine `ref_init_q`

`ref_init_q` is still useful and should remain exactly:
- frame-0 `q_ref` from the offline reference library

Using it as a surrogate nominal policy pose is what created confusion in the first place.

#### 3.3 Do not do a many-knob rescue run

Avoid a run that simultaneously changes:
- home pose
- reset behavior
- reward formulas
- command distribution
- PPO hyperparameters
- residual bounds

That produces no clean attribution.

#### 3.4 Do not remove all reference/prior structure from the environment blindly

The problem is not that reference observations or diagnostics exist.
The problem is the current hybrid control basin.

Avoid:
- deleting prior preview / reference data without a concrete reason
- turning the next run into a different research project before the fixed-home test is done

#### 3.5 Do not switch to literal old WR straight-leg home for the next locomotion branch

This was already probed conceptually and is not the right target.
The straight-leg home under current residual limits likely under-covers the gait manifold and can fail for the wrong reason.

## Proposed execution order

1. land and verify the new crouched phase-neutral `home`
2. rerun the TB-style fixed-home branch with no additional reward/PPO surgery
3. compare trajectory against:
   - latest `smoke9c`
   - prior WR home-base attempts
   - TB early training trajectory
4. only if still stuck, escalate to control-contract or cmd-track changes

## External reference framing

This ranking is consistent with two established patterns:
- fixed nominal/default-pose command-tracking locomotion can work well when the default pose is locomotion-ready and exploration is shaped by perturbation/randomization rather than by a reference-anchored action basin
  - Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots" (2019), arXiv:1901.08652
- reference-guided RL also works, but only when the reference remains a coherent load-bearing part of the policy objective rather than a partial hybrid
  - Peng et al., "DeepMimic" (2018), arXiv:1804.02717

For WR, the immediate objective is not to redesign the paradigm; it is to remove the clearest structural mismatch first and measure the effect cleanly.
