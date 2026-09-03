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

If another experiment is justified:

1. Identify one dominant failure mode using concrete metrics and code evidence.
2. Make only the smallest justified code/config change.
3. Run focused tests.
4. Commit the change locally.
5. Return `decision=continue`, the tracked config path, and exactly one starting
   mode/checkpoint for the next GPU job.

If the result does not support a safe next experiment, make no changes and
return `decision=stop` with a concrete reason.

Constraints:

- Do not modify files under `wildrobot/agents/`; the automation control plane is
  frozen while the loop is active.
- Do not update `training/CHANGELOG.md`; results require user confirmation.
- Do not push Git commits, submit GPU jobs, export bundles, or run hardware.
- Do not invent a checkpoint. It must be a `.pkl` path present in the remote
  manifest or deterministic evaluation summary.
- Leave the Git worktree clean. Any code/config change must be committed.
- Prefer ToddlerBot-aligned behavior unless the evidence requires a documented
  WildRobot-specific divergence.
