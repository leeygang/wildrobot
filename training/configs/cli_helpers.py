"""Small CLI helpers shared by training/eval/scripts entrypoints.

Single responsibility: present a uniform fail-fast experience when a
default config path doesn't exist on disk, instead of letting each
CLI throw a raw ``FileNotFoundError`` traceback.

Used by:
  - training/train.py (inlined; doesn't import this to keep it
    self-contained)
  - training/eval/visualize_policy.py
  - training/eval/eval_handoff_gate.py
  - training/eval/eval_loc_ref_probe.py
  - training/eval/diagnose_knee_forces.py
  - training/scripts/diagnose_torque.py
  - training/scripts/gait_diagnostic.py
  - scripts/generate_policy_spec.py (path-fixed inline copy; this
    package may not be importable from a repo-root CLI when called
    directly without --module-style invocation)
"""

from __future__ import annotations

import sys
from pathlib import Path


_V0201_CLEANUP_NOTE = (
    "v0.20.1 cleanup deleted ppo_walking*.yaml; the v0.20.1 PPO "
    "smoke YAML is open task #49.  Pass --config explicitly, or "
    "land the smoke YAML first."
)


def fail_if_config_missing(
    config_path: str | Path,
    *,
    user_passed_explicit: bool = False,
    extra_hint: str | None = None,
) -> None:
    """Exit non-zero with a clear message if ``config_path`` doesn't exist.

    ``user_passed_explicit=True`` suppresses the v0.20.1 deletion note
    (the user already chose the path; no need to bring up the smoke
    YAML).  Otherwise we add the deletion note explaining why the
    default may be missing.
    """
    p = Path(config_path)
    if p.exists():
        return
    print(f"ERROR: Config file not found: {p}", file=sys.stderr)
    if not user_passed_explicit:
        print(f"  {_V0201_CLEANUP_NOTE}", file=sys.stderr)
    if extra_hint:
        print(f"  {extra_hint}", file=sys.stderr)
    sys.exit(2)
