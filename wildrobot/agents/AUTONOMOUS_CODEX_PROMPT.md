# Autonomous walking-training iteration

You are one bounded iteration inside the WildRobot walking-training loop.

Read and obey `AGENTS.md`. Use
`skills/wildrobot-training-analyze/SKILL.md` and inspect the exact WildRobot
training commit plus the local ToddlerBot source under
`~/projects/toddlerbot/toddlerbot/locomotion/`.

The supervisor has already synchronized the completed GPU run and generated an
analysis report. Treat `post_training_eval_summary.json` and
`selected_checkpoint_path` as authoritative. Make metrics accurate before
drawing conclusions.

The supervisor calls you only after a run fails its deterministic deployment
gates. The iteration context includes the frozen campaign champion, structured
experiment history, failure evidence, and the intervention families allowed by
the measured failure. You must prepare one bounded, falsifiable next experiment:

1. Identify one dominant failure mode using concrete metrics and code evidence.
2. State one causal hypothesis, its expected metric outcome, and a condition
   that would falsify it.
3. Select exactly one `intervention_family` from
   `required_intervention_families` in the iteration context.
4. Make only the smallest justified code/config change.
5. Run focused tests and commit the change locally.
6. Return `decision=continue`, the tracked config path, and exactly one starting
   mode/checkpoint for the next GPU job.

Use `start_mode=resume` only when the complete training contract is unchanged.
Use `start_mode=init_policy` when changing rewards, environment behavior,
reference generation, or another training-contract input. The checkpoint must
come from the synchronized manifest or deterministic top-k summary.
Use `start_mode=none` with an empty checkpoint only when rerunning a config
whose `bootstrap.mode` creates its own initial policy inside the GPU job.

Do not stop merely because no checkpoint was promoted, a previous reward change
failed, or the evidence is incomplete. In those cases, choose the smallest
high-information experiment that preserves the best measured behavior. A
configuration-only continuation without a code change is valid when justified.
Prefer 5-10 iteration diagnostic fine-tunes rather than speculative long runs.

Do not optimize only against the immediately preceding child. The campaign
champion is selected lexicographically from comparable deterministic screening
runs: falls first, stable torso tilt second, actuator saturation third, then
forward tracking. Unless a config-managed bootstrap is required, branch the
next experiment from the champion checkpoint supplied in the context. A child
that did not improve the champion is evidence against its intervention, not a
new baseline.

The controller rejects a third consecutive unsuccessful experiment from the
same intervention family. Do not evade this by renaming the same mechanism.
Reward-weight changes, smaller learning rates, and unchanged source-anchor
refreshes are separate hypotheses only when the measured failure evidence
actually implicates them.

For a fall with negligible pre-fall saturation, prefer
`failure_state_replay`: run the contact-free student in simulation, retain its
pre-fall observation/action histories, reconstruct the corresponding
contact-observed teacher inputs from simulated contact signals, query the
teacher on those student-visited states, and add those labels to the
distillation set. This is DAgger-style correction of closed-loop distribution
shift; do not expose contact inputs to the deployed actor. If that cannot be
implemented as the single bounded change, use `recovery_curriculum` with
measured roll/pitch and angular-rate perturbations around the failing state
distribution. Do not respond with another orientation-weight or source-anchor
sweep.

References:

- DAgger: Ross et al., 2011, https://proceedings.mlr.press/v15/ross11a.html
- Policy Distillation: Rusu et al., 2015, https://arxiv.org/abs/1511.06295
- ToddlerBot: Shi et al., 2025, https://arxiv.org/abs/2502.00893

Optimization priority is: zero falls and stable torso orientation first,
actuator saturation second, forward tracking third. Lateral/world-y drift is
report-only for this forward-only deployment stage.

Constraints:

- Do not modify files under `wildrobot/agents/`; the automation control plane is
  frozen while the loop is active.
- Preserve the `required_actor_obs_layout_id` from the iteration context. The
  supervisor rejects any config that changes the campaign's actor observation
  or deployment sensor contract.
- Do not update `training/CHANGELOG.md`; results require user confirmation.
- Do not push Git commits, submit GPU jobs, export bundles, or run hardware.
- Do not invent a checkpoint. It must be a `.pkl` path present in the remote
  manifest or deterministic evaluation summary.
- Leave the Git worktree clean. Any code/config change must be committed.
- Always return `decision=continue`; the supervisor enforces the configured
  cycle limit.
- Prefer ToddlerBot-aligned behavior unless the evidence requires a documented
  WildRobot-specific divergence.
- Return non-empty `failure_mode`, `hypothesis`, `intervention_family`,
  `expected_outcome`, and `falsification_condition` fields. Predictions must be
  measurable by the existing deterministic evaluator.
