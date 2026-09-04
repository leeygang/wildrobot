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
gates. You must prepare one bounded next experiment:

1. Identify one dominant failure mode using concrete metrics and code evidence.
2. Make only the smallest justified code/config change.
3. Run focused tests.
4. Commit the change locally.
5. Return `decision=continue`, the tracked config path, and exactly one starting
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
  cycle and training-failure limits.
- Prefer ToddlerBot-aligned behavior unless the evidence requires a documented
  WildRobot-specific divergence.
