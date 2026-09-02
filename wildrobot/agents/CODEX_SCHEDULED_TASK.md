# WildRobot GPU training watcher

Run this task in the local WildRobot project every 10-15 minutes.

```text
Inspect the active WildRobot GPU training job with:

uv run python wildrobot/agents/remote_training_loop.py status --json

If the job is queued or running, report only a status transition, failure, or
other actionable change. Do not edit the repository.

If the job failed, sync its summary artifacts and log, identify the concrete
execution failure, and report it. Do not start a replacement job automatically.

If the job completed:

1. Run `uv run python wildrobot/agents/remote_training_loop.py analyze`.
2. Treat `post_training_eval_summary.json` and `selected_checkpoint_path` as
   authoritative; do not promote a checkpoint from stochastic training metrics.
3. Verify the manifest Git SHA and inspect WildRobot code at that exact commit.
4. Inspect the corresponding local ToddlerBot implementation under
   `~/projects/toddlerbot/toddlerbot/locomotion/` before recommending a change.
5. Make metrics accurate first, identify one dominant failure mode, and prepare
   only the smallest justified code/config change.
6. Run focused tests and commit the candidate change locally.
7. Do not push or submit the changed experiment. Stop and request approval.
8. Do not update `training/CHANGELOG.md` until the result and interpretation
   have been confirmed by the user.

If `simulation_candidate_ready` is true, sync the promoted checkpoint with:

uv run python wildrobot/agents/remote_training_loop.py sync --selected-checkpoint

Then prepare the documented bundle export and validation commands, but stop for
explicit approval before any hardware run. Never deploy automatically.
```
