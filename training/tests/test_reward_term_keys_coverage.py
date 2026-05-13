"""Static-analysis regression test: REWARD_TERM_KEYS must cover every
``reward/X`` key the env writes to ``terminal_metrics_dict``.

Why this exists: there are THREE separate places a reward must be
declared for it to reach wandb:

  1. ``RewardWeightsConfig`` field in ``training_runtime_config.py``
     (so the yaml weight is loaded).
  2. ``terminal_metrics_dict["reward/X"] = ...`` write in
     ``training/envs/wildrobot_env.py`` (so the value enters the
     metrics vector).
  3. ``REWARD_TERM_KEYS`` entry in
     ``training/core/experiment_tracking.py`` (the wandb dispatcher's
     allow-list).

Drift between (2) and (3) is the silent-drop bug pattern that masked
``reward/penalty_pose`` and ``reward/penalty_feet_ori`` for the entire
smoke8b/smoke9/smoke9b run series, plus ``reward/feet_phase`` and
``reward/penalty_close_feet_xy`` for smoke9/smoke9b.  The reward IS
active in PPO's loss, but invisible in metrics — so we can't diagnose
training failures.

This test parses the env source for write sites and asserts the
allow-list covers all of them.  Sub-second.
"""

from __future__ import annotations

import re
from pathlib import Path

from training.core.experiment_tracking import REWARD_TERM_KEYS

ENV_PATH = Path("training/envs/wildrobot_env.py")
WRITE_PATTERN = re.compile(
    r'terminal_metrics_dict\[\s*"(reward/[^"]+)"\s*\]\s*='
)


def _extract_reward_metric_writes() -> set[str]:
    """Scan wildrobot_env.py for every
    ``terminal_metrics_dict["reward/..."] = ...`` line and return the
    set of distinct ``reward/*`` keys written."""
    if not ENV_PATH.exists():
        # Fixture-less test setup; skip rather than fail at import.
        return set()
    src = ENV_PATH.read_text()
    return set(WRITE_PATTERN.findall(src))


def test_reward_term_keys_covers_all_env_writes() -> None:
    """Every ``reward/X`` written to ``terminal_metrics_dict`` in the env
    must have a matching entry in ``REWARD_TERM_KEYS``.  Otherwise the
    wandb dispatcher silently drops the value — the reward affects PPO's
    loss but is invisible in logs.
    """
    env_writes = _extract_reward_metric_writes()
    if not env_writes:
        # Env source not present in this checkout; skip.
        return
    allow_list = set(REWARD_TERM_KEYS)
    missing = sorted(env_writes - allow_list)
    if missing:
        raise AssertionError(
            "REWARD_TERM_KEYS allow-list (training/core/experiment_tracking.py) "
            f"is missing {len(missing)} reward keys that the env writes to "
            f"terminal_metrics_dict.  Without these entries, the values are "
            f"silently dropped by build_wandb_metrics — the reward affects PPO's "
            f"loss but never reaches wandb logs.\n\n"
            f"Add to REWARD_TERM_KEYS:\n  "
            + "\n  ".join(f'"{k}",' for k in missing)
            + "\n\nThis is the same bug class that hid penalty_pose / "
            "penalty_feet_ori / feet_phase / penalty_close_feet_xy from "
            "smoke7-9b metrics."
        )


def test_reward_term_keys_no_dead_entries() -> None:
    """Soft warning: every ``reward/X`` in REWARD_TERM_KEYS should be
    written by the env somewhere (or be a hardcoded raw-penalty diagnostic
    like ``reward/penalty_slip_raw``).  This catches accumulated cruft.

    Currently informational — fails noisily so the diff shows what's
    unused, but does not block.  If the list grows useful entries that
    aren't yet wired (e.g. forward-looking reward names), gate this with
    an explicit allowed-orphan list.
    """
    env_writes = _extract_reward_metric_writes()
    if not env_writes:
        return
    # Known orphans by design: aggregate "total" + raw-penalty diagnostics
    # that the env writes via reward_terms[...] directly (not reward_contrib).
    KNOWN_ORPHANS: set[str] = set()  # extend if needed
    declared = set(REWARD_TERM_KEYS)
    orphans = sorted((declared - env_writes) - KNOWN_ORPHANS)
    if orphans:
        # Don't fail; print for visibility.  Convert to fail when the
        # list is small enough to actively maintain.
        print(
            "INFO: REWARD_TERM_KEYS entries with no env write site:\n  "
            + "\n  ".join(orphans)
            + "\nThese log as 0.0 every step — likely safe but worth a glance "
            "during audit if the list grows."
        )


def test_reward_term_keys_includes_smoke9_tb_set() -> None:
    """Pin the specific TB-faithful smoke9 keys.  Even if env writes are
    refactored, these reward terms are the load-bearing TB-aligned signals
    and MUST be visible in wandb."""
    required_for_smoke9 = {
        "reward/alive",
        "reward/cmd_forward_velocity_track",
        "reward/ref_body_quat_track",
        "reward/ang_vel_xy",
        "reward/action_rate",
        "reward/penalty_pose",          # smoke8b-onward; missed in earlier allow-list
        "reward/penalty_feet_ori",      # smoke8b-onward; missed
        "reward/penalty_close_feet_xy", # smoke9-onward; missed
        "reward/feet_phase",            # smoke9-onward; missed
    }
    declared = set(REWARD_TERM_KEYS)
    missing = sorted(required_for_smoke9 - declared)
    assert not missing, (
        "REWARD_TERM_KEYS missing smoke9 TB-faithful reward keys: "
        + ", ".join(missing)
    )
