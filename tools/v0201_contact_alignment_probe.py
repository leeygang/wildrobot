#!/usr/bin/env python3
"""v0.20.1 pre-smoke contact-alignment baseline.

Runs the v0.20.1 PPO env (residual-only filter, smoke YAML) for N
control steps with zero policy action and pushes disabled, and
reports how well the offline ZMP prior's commanded contact mask
matches MuJoCo's measured foot/floor contacts.

This guards the env's ``ref/contact_match`` reward: it shapes a
smooth Gaussian on (commanded contact - measured contact) per foot.
If the prior's contact schedule and the closed-loop physics already
disagree at iter-0 (bare q_ref replay), the reward is noise from the
first PPO step.  Round-7 fixed the FK-time alignment but the env-side
alignment under PD tracking error has never been printed.

Acceptance bar (training/docs/walking_training.md v0.20.1 high-
confidence prep):

  - mean ref/contact_phase_match >= 0.90 over the probe horizon
  - no long mismatch streak (default: <= 5 consecutive steps)
  - all per-step values finite + stable

Usage::

    uv run python tools/v0201_contact_alignment_probe.py \\
        --config training/configs/ppo_walking_v0201_smoke.yaml \\
        --steps 100 --seed 42

Same config + same seed as the smoke (--seed defaults to the YAML's
``seed: 42``), so the baseline is what PPO actually starts from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assets.robot_config import load_robot_config  # noqa: E402
from training.configs.training_config import load_training_config  # noqa: E402
from training.envs.env_info import WR_INFO_KEY  # noqa: E402
from training.envs.wildrobot_env import WildRobotEnv  # noqa: E402


def _longest_streak(mask: np.ndarray) -> int:
    """Length of the longest consecutive ``True`` run in ``mask``."""
    if mask.size == 0:
        return 0
    longest = run = 0
    for v in mask.tolist():
        if v:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_walking_v0201_smoke.yaml",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mismatch-streak-fail",
        type=int,
        default=5,
        help="Hard fail if any per-foot mismatch streak >= this length",
    )
    p.add_argument(
        "--mean-match-pass",
        type=float,
        default=0.90,
        help="Hard pass requires mean ref/contact_phase_match >= this value",
    )
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"FAIL: config not found: {cfg_path}", file=sys.stderr)
        return 2

    load_robot_config("assets/v2/mujoco_robot_config.json")
    cfg = load_training_config(str(cfg_path))
    cfg.freeze()

    env = WildRobotEnv(cfg)

    contact_thresh = float(cfg.env.contact_threshold_force)
    sigma = float(cfg.reward_weights.ref_contact_match_sigma)
    denom = 2.0 * sigma * sigma + 1e-8

    # Same seed the smoke uses (cfg.seed defaults to 42); the probe must
    # mirror "what PPO actually starts from".
    seed = int(args.seed)
    print(
        f"Probe: {cfg_path.name}  steps={args.steps}  seed={seed}  "
        f"sigma={sigma}  contact_force_thresh={contact_thresh} N"
    )

    reset_fn = jax.jit(env.reset)
    # Pushes disabled both via YAML (push_enabled=false) AND via the
    # disable_pushes runtime flag, so the baseline is purely the prior +
    # PD response — no exogenous disturbance.
    step_fn = jax.jit(lambda s, a: env.step(s, a, disable_pushes=True))
    state = reset_fn(jax.random.PRNGKey(seed))

    zero_action = jp.zeros(env.action_size, dtype=jp.float32)

    cmd_l, cmd_r = [], []
    meas_l, meas_r = [], []
    match_l, match_r = [], []
    contact_phase_match = []
    step_indices = []
    terminated_at = None

    for i in range(args.steps):
        state = step_fn(state, zero_action)
        if int(state.done) > 0:
            terminated_at = i
            break

        wr = state.info[WR_INFO_KEY]
        step_idx = int(wr.loc_ref_offline_step_idx)
        win = env._lookup_offline_window(jp.asarray(step_idx, dtype=jp.int32))
        cmd_contact = np.asarray(win["contact_mask"]).astype(np.float32)

        # Measured foot/floor contact via the env's CAL (same path the
        # reward and termination use).
        l_force, r_force = env._cal.get_aggregated_foot_contacts(state.data)
        l_meas = float(np.asarray(l_force) > contact_thresh)
        r_meas = float(np.asarray(r_force) > contact_thresh)

        # Smooth Gaussian shape, byte-identical to env._compute_reward_terms.
        gauss_l = float(np.exp(-((cmd_contact[0] - l_meas) ** 2) / denom))
        gauss_r = float(np.exp(-((cmd_contact[1] - r_meas) ** 2) / denom))
        r_match = 0.5 * (gauss_l + gauss_r)

        cmd_l.append(float(cmd_contact[0]))
        cmd_r.append(float(cmd_contact[1]))
        meas_l.append(l_meas)
        meas_r.append(r_meas)
        match_l.append(int(cmd_contact[0] == l_meas))
        match_r.append(int(cmd_contact[1] == r_meas))
        contact_phase_match.append(r_match)
        step_indices.append(step_idx)

    cmd_l_arr = np.asarray(cmd_l, dtype=np.float32)
    cmd_r_arr = np.asarray(cmd_r, dtype=np.float32)
    meas_l_arr = np.asarray(meas_l, dtype=np.float32)
    meas_r_arr = np.asarray(meas_r, dtype=np.float32)
    match_l_arr = np.asarray(match_l, dtype=np.int32)
    match_r_arr = np.asarray(match_r, dtype=np.int32)
    cpm_arr = np.asarray(contact_phase_match, dtype=np.float32)

    n = cpm_arr.shape[0]
    if n == 0:
        print("FAIL: env terminated before any step completed.", file=sys.stderr)
        return 1

    mean_cpm = float(cpm_arr.mean())
    min_cpm = float(cpm_arr.min())
    finite = bool(np.isfinite(cpm_arr).all())
    hard_match_rate_l = float(match_l_arr.mean())
    hard_match_rate_r = float(match_r_arr.mean())
    streak_l = _longest_streak(match_l_arr == 0)
    streak_r = _longest_streak(match_r_arr == 0)
    first_mismatch_l = int(np.argmin(match_l_arr)) if (match_l_arr == 0).any() else -1
    first_mismatch_r = int(np.argmin(match_r_arr)) if (match_r_arr == 0).any() else -1

    print()
    print("===================== Contact alignment baseline =====================")
    print(f"steps observed:                 {n}/{args.steps}")
    if terminated_at is not None:
        print(f"terminated at probe step:       {terminated_at}")
    print(f"mean ref/contact_phase_match:   {mean_cpm:.4f}")
    print(f"min  ref/contact_phase_match:   {min_cpm:.4f}")
    print(f"all values finite:              {finite}")
    print(f"hard match rate (L / R):        {hard_match_rate_l:.3f} / "
          f"{hard_match_rate_r:.3f}")
    print(f"longest mismatch streak (L/R):  {streak_l} / {streak_r}")
    print(f"first mismatch step (L / R):    {first_mismatch_l} / {first_mismatch_r}")

    # Per-step short trace to spot the divergence point quickly.
    print()
    print("First-20-step trace (cmd_L cmd_R | meas_L meas_R | r_match):")
    head = min(20, n)
    for i in range(head):
        print(
            f"  step={step_indices[i]:4d}  "
            f"cmd=({int(cmd_l_arr[i])},{int(cmd_r_arr[i])})  "
            f"meas=({int(meas_l_arr[i])},{int(meas_r_arr[i])})  "
            f"r={cpm_arr[i]:.3f}"
        )

    pass_mean = mean_cpm >= args.mean_match_pass
    pass_streak = max(streak_l, streak_r) <= args.mismatch_streak_fail
    pass_finite = finite

    print()
    print(f"  mean >= {args.mean_match_pass:.2f}    -> {'PASS' if pass_mean else 'FAIL'}")
    print(f"  streak <= {args.mismatch_streak_fail}   -> {'PASS' if pass_streak else 'FAIL'}")
    print(f"  finite + stable    -> {'PASS' if pass_finite else 'FAIL'}")

    overall = pass_mean and pass_streak and pass_finite
    print()
    print(f"BASELINE: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
