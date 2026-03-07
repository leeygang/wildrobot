#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Protocol

import yaml


@dataclass(frozen=True)
class EvalKeys:
    success: str
    ep_len: str
    survival_rate: str
    survival_steps: str
    term_height_low: str
    term_pitch: str


@dataclass(frozen=True)
class MetricsRow:
    iteration: int
    step: Optional[int]
    values: Dict[str, Any]


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _get_iter(row: Dict[str, Any]) -> Optional[int]:
    for key in ("progress/iteration", "iteration"):
        if key in row:
            try:
                return int(row[key])
            except Exception:
                return None
    return None


def _get_step(row: Dict[str, Any]) -> Optional[int]:
    for key in ("time/step", "_step"):
        if key in row:
            try:
                return int(row[key])
            except Exception:
                return None
    return None


def _get_float(row: Dict[str, Any], key: str) -> Optional[float]:
    if key not in row:
        return None
    try:
        return float(row[key])
    except Exception:
        return None


def _pick_eval_keys(rows: Iterable[Dict[str, Any]]) -> EvalKeys:
    # Prefer eval_push/* (standing_push)
    rows_list = list(rows)
    if any("eval_push/success_rate" in r for r in rows_list):
        return EvalKeys(
            success="eval_push/success_rate",
            ep_len="eval_push/episode_length",
            survival_rate="eval_push/survival_rate",
            survival_steps="eval_push/survival_steps",
            term_height_low="eval_push/term_height_low_frac",
            term_pitch="eval_push/term_pitch_frac",
        )
    if any("eval/success_rate" in r for r in rows_list):
        return EvalKeys(
            success="eval/success_rate",
            ep_len="eval/episode_length",
            survival_rate="eval/survival_rate",
            survival_steps="eval/survival_steps",
            term_height_low="eval/term_height_low_frac",
            term_pitch="eval/term_pitch_frac",
        )
    # Fallback: training rollout metrics
    return EvalKeys(
        success="env/success_rate",
        ep_len="env/episode_length",
        survival_rate="env/success_rate",
        survival_steps="env/episode_length",
        term_height_low="term_height_low_frac",
        term_pitch="term_pitch_frac",
    )


def _lexi_better(s: float, L: float, best_s: float, best_L: float, eps: float = 1e-9) -> bool:
    if s > best_s + eps:
        return True
    return abs(s - best_s) <= eps and L > best_L + eps


def _best_row_by_eval(rows: List[MetricsRow], keys: EvalKeys) -> MetricsRow:
    best: Optional[MetricsRow] = None
    best_s = float("-inf")
    best_L = float("-inf")
    for r in rows:
        s = _get_float(r.values, keys.success)
        L = _get_float(r.values, keys.ep_len)
        if s is None or L is None:
            continue
        if best is None or _lexi_better(s, L, best_s, best_L) or (
            s == best_s and L == best_L and r.iteration > best.iteration
        ):
            best = r
            best_s, best_L = s, L
    if best is None:
        raise RuntimeError(
            f"No metrics rows had both {keys.success!r} and {keys.ep_len!r}."
        )
    return best


def _best_row_by_survival(rows: List[MetricsRow], keys: EvalKeys) -> MetricsRow:
    best: Optional[MetricsRow] = None
    best_s = float("-inf")
    best_L = float("-inf")
    for r in rows:
        s = _get_float(r.values, keys.survival_rate)
        L = _get_float(r.values, keys.survival_steps)
        if s is None or L is None:
            continue
        if best is None or _lexi_better(s, L, best_s, best_L) or (
            s == best_s and L == best_L and r.iteration > best.iteration
        ):
            best = r
            best_s, best_L = s, L
    if best is None:
        raise RuntimeError(
            f"No metrics rows had both {keys.survival_rate!r} and {keys.survival_steps!r}."
        )
    return best


def _find_checkpoint_for_iter(ckpt_dir: Path, iteration: int, step: Optional[int]) -> Optional[Path]:
    if step is not None:
        exact = ckpt_dir / f"checkpoint_{iteration}_{step}.pkl"
        if exact.exists():
            return exact
    matches = sorted(ckpt_dir.glob(f"checkpoint_{iteration}_*.pkl"))
    return matches[0] if matches else None


def _tail_lines(path: Path, n: int = 30) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(cfg: Dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, sort_keys=False)


def _set_nested(cfg: Dict[str, Any], path: str, value: Any) -> None:
    cur: Any = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _get_nested(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


FORBIDDEN_MUTATIONS = {
    # Changes the policy_contract spec hash; resuming from an older checkpoint will fail.
    "env.action_filter_alpha",
}

ALLOWED_MUTATION_PREFIXES = (
    "env.collapse_",
    "env.push_",
    "reward_weights.",
    "ppo.learning_rate",
    "ppo.clip_epsilon",
    "ppo.entropy_coef",
)


def _validate_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in updates.items():
        if k in FORBIDDEN_MUTATIONS:
            raise ValueError(f"Refusing to mutate {k!r}; it breaks resume policy_contract safety.")
        if k != "env.min_height" and not any(k.startswith(p) for p in ALLOWED_MUTATION_PREFIXES):
            raise ValueError(f"Refusing to mutate {k!r}; not in allowed prefixes {ALLOWED_MUTATION_PREFIXES}.")
        if isinstance(v, bool):
            cleaned[k] = v
        elif isinstance(v, int):
            cleaned[k] = int(v)
        elif isinstance(v, float):
            cleaned[k] = float(v)
        else:
            raise ValueError(f"Refusing update {k!r}={v!r}; only bool/int/float allowed.")
    return cleaned


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x*100:.2f}%"


def _format_f(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def _format_iter_line(row: Dict[str, Any]) -> str:
    it = _get_iter(row)
    if it is None:
        return ""

    # Prefer showing eval_push if available, otherwise training rollout.
    keys = _pick_eval_keys([row])
    prefix = keys.success.split("/")[0] if "/" in keys.success else ""
    eval_s = _get_float(row, keys.success)
    eval_L = _get_float(row, keys.ep_len)
    h_low = _get_float(row, keys.term_height_low)
    pitch = _get_float(row, keys.term_pitch)
    surv_s = _get_float(row, keys.survival_rate)
    surv_L = _get_float(row, keys.survival_steps)
    done_env = _get_float(row, f"{prefix}/done_env_frac") if prefix else None
    trunc_env = _get_float(row, f"{prefix}/trunc_env_frac") if prefix else None
    reset_h_mean = _get_float(row, f"{prefix}/reset_height_mean") if prefix else None
    reset_h_min = _get_float(row, f"{prefix}/reset_height_min") if prefix else None

    train_s = _get_float(row, "env/success_rate") or _get_float(row, "success_rate")
    train_L = _get_float(row, "env/episode_length") or _get_float(row, "episode_length")
    torque_sat = _get_float(row, "debug/torque_sat_frac")
    approx_kl = _get_float(row, "ppo/approx_kl") or _get_float(row, "approx_kl")
    clip_frac = _get_float(row, "ppo/clip_fraction") or _get_float(row, "clip_fraction")

    # If eval keys are missing (e.g. probe runs with eval disabled), fall back to train.
    if eval_s is None or eval_L is None:
        return (
            f"[iter {it:>4}] "
            f"train: success={_format_pct(train_s)} ep_len={_format_f(train_L, 1)} | "
            f"term_h_low={_format_pct(_get_float(row, 'term_height_low_frac'))} "
            f"term_pitch={_format_pct(_get_float(row, 'term_pitch_frac'))} | "
            f"stress: torque_sat={_format_pct(torque_sat)} | "
            f"ppo: kl={_format_f(approx_kl, 4)} clip={_format_f(clip_frac, 4)}"
        )

    # Also show clean eval if present.
    clean_s = _get_float(row, "eval_clean/success_rate")
    clean_L = _get_float(row, "eval_clean/episode_length")
    # Short-horizon probe eval makes success/episode_length (often truncation-based) misleading.
    # Survival metrics stay meaningful even when eval.num_steps < env.max_episode_steps.
    surv_msg = ""
    if surv_s is not None and surv_L is not None:
        surv_msg = f" | survive={_format_pct(surv_s)}@{_format_f(surv_L, 1)}"
    diag_msg = ""
    if done_env is not None or trunc_env is not None:
        diag_msg += f" | done_env={_format_pct(done_env)} trunc_env={_format_pct(trunc_env)}"
    if reset_h_mean is not None or reset_h_min is not None:
        diag_msg += f" | reset_h={_format_f(reset_h_mean, 3)}/{_format_f(reset_h_min, 3)}"
    return (
        f"[iter {it:>4}] "
        f"eval: success={_format_pct(eval_s)} ep_len={_format_f(eval_L, 1)} | "
        f"term_h_low={_format_pct(h_low)} term_pitch={_format_pct(pitch)} | "
        f"clean: success={_format_pct(clean_s)} ep_len={_format_f(clean_L, 1)}"
        f"{surv_msg}{diag_msg} | "
        f"stress: torque_sat={_format_pct(torque_sat)} | "
        f"ppo: kl={_format_f(approx_kl, 4)} clip={_format_f(clip_frac, 4)}"
    )


class LiveMetricsPrinter:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._metrics_path: Optional[Path] = None
        self._last_pos: int = 0
        self._last_iter_printed: int = -1

    def set_run_dir(self, run_dir: Path) -> None:
        self._metrics_path = run_dir / "files" / "metrics.jsonl"

    def start(self, is_process_alive) -> None:
        if self._thread is not None:
            return

        def _worker() -> None:
            # Wait for file to exist, but exit if the process ends.
            while not self._stop.is_set():
                if self._metrics_path is not None and self._metrics_path.exists():
                    break
                if not is_process_alive():
                    return
                time.sleep(0.5)

            if self._metrics_path is None or not self._metrics_path.exists():
                return

            path = self._metrics_path
            while not self._stop.is_set() and is_process_alive():
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    time.sleep(0.5)
                    continue

                if size < self._last_pos:
                    # File rotated/truncated; reset.
                    self._last_pos = 0

                if size == self._last_pos:
                    time.sleep(0.5)
                    continue

                with path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(self._last_pos)
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        self._last_pos = f.tell()
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        it = _get_iter(row)
                        if it is None or it <= self._last_iter_printed:
                            continue
                        self._last_iter_printed = it
                        msg = _format_iter_line(row)
                        if msg:
                            print(msg, flush=True)
                time.sleep(0.2)

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


@dataclass
class TuningKnobs:
    collapse_height_buffer: float
    collapse_vz_gate_band: float
    w_collapse_height: float
    w_collapse_vz: float
    w_orientation: float
    w_clearance: float
    w_flight_phase_penalty: float
    w_posture: float
    posture_sigma: float
    push_force_max: float
    push_duration_steps: int
    w_action_rate: float
    w_torque: float
    w_gait_periodicity: float
    w_hip_swing: float
    w_knee_swing: float
    hip_swing_min: float
    knee_swing_min: float
    w_step_event: float
    w_foot_place: float
    min_height: float


def _extract_knobs(cfg: Dict[str, Any]) -> TuningKnobs:
    return TuningKnobs(
        collapse_height_buffer=float(_get_nested(cfg, "env.collapse_height_buffer") or 0.02),
        collapse_vz_gate_band=float(_get_nested(cfg, "env.collapse_vz_gate_band") or 0.05),
        w_collapse_height=float(_get_nested(cfg, "reward_weights.collapse_height") or -0.2),
        w_collapse_vz=float(_get_nested(cfg, "reward_weights.collapse_vz") or -0.2),
        w_orientation=float(_get_nested(cfg, "reward_weights.orientation") or -0.5),
        w_clearance=float(_get_nested(cfg, "reward_weights.clearance") or 0.1),
        w_flight_phase_penalty=float(_get_nested(cfg, "reward_weights.flight_phase_penalty") or 0.0),
        w_posture=float(_get_nested(cfg, "reward_weights.posture") or 0.0),
        posture_sigma=float(_get_nested(cfg, "reward_weights.posture_sigma") or 0.35),
        push_force_max=float(_get_nested(cfg, "env.push_force_max") or 0.0),
        push_duration_steps=int(_get_nested(cfg, "env.push_duration_steps") or 10),
        w_action_rate=float(_get_nested(cfg, "reward_weights.action_rate") or -0.01),
        w_torque=float(_get_nested(cfg, "reward_weights.torque") or -0.001),
        w_gait_periodicity=float(_get_nested(cfg, "reward_weights.gait_periodicity") or 0.0),
        w_hip_swing=float(_get_nested(cfg, "reward_weights.hip_swing") or 0.0),
        w_knee_swing=float(_get_nested(cfg, "reward_weights.knee_swing") or 0.0),
        hip_swing_min=float(_get_nested(cfg, "reward_weights.hip_swing_min") or 0.0),
        knee_swing_min=float(_get_nested(cfg, "reward_weights.knee_swing_min") or 0.0),
        w_step_event=float(_get_nested(cfg, "reward_weights.step_event") or 0.0),
        w_foot_place=float(_get_nested(cfg, "reward_weights.foot_place") or 0.0),
        min_height=float(_get_nested(cfg, "env.min_height") or 0.20),
    )


def _apply_knobs(cfg: Dict[str, Any], knobs: TuningKnobs) -> None:
    _set_nested(cfg, "env.collapse_height_buffer", float(knobs.collapse_height_buffer))
    _set_nested(cfg, "env.collapse_vz_gate_band", float(knobs.collapse_vz_gate_band))
    _set_nested(cfg, "reward_weights.collapse_height", float(knobs.w_collapse_height))
    _set_nested(cfg, "reward_weights.collapse_vz", float(knobs.w_collapse_vz))
    _set_nested(cfg, "reward_weights.orientation", float(knobs.w_orientation))
    _set_nested(cfg, "reward_weights.clearance", float(knobs.w_clearance))
    _set_nested(cfg, "reward_weights.flight_phase_penalty", float(knobs.w_flight_phase_penalty))
    _set_nested(cfg, "reward_weights.posture", float(knobs.w_posture))
    _set_nested(cfg, "reward_weights.posture_sigma", float(knobs.posture_sigma))
    _set_nested(cfg, "env.push_force_max", float(knobs.push_force_max))
    _set_nested(cfg, "env.push_duration_steps", int(knobs.push_duration_steps))
    _set_nested(cfg, "reward_weights.action_rate", float(knobs.w_action_rate))
    _set_nested(cfg, "reward_weights.torque", float(knobs.w_torque))
    _set_nested(cfg, "reward_weights.gait_periodicity", float(knobs.w_gait_periodicity))
    _set_nested(cfg, "reward_weights.hip_swing", float(knobs.w_hip_swing))
    _set_nested(cfg, "reward_weights.knee_swing", float(knobs.w_knee_swing))
    _set_nested(cfg, "reward_weights.hip_swing_min", float(knobs.hip_swing_min))
    _set_nested(cfg, "reward_weights.knee_swing_min", float(knobs.knee_swing_min))
    _set_nested(cfg, "reward_weights.step_event", float(knobs.w_step_event))
    _set_nested(cfg, "reward_weights.foot_place", float(knobs.w_foot_place))
    _set_nested(cfg, "env.min_height", float(knobs.min_height))


@dataclass(frozen=True)
class AdvisorDecision:
    updates: Dict[str, Any]
    reason: str


class Advisor(Protocol):
    def suggest(self, *, cfg: Dict[str, Any], metrics: MetricsRow, keys: EvalKeys, probe_is_short: bool) -> AdvisorDecision: ...


def _propose_next_knobs(
    knobs: TuningKnobs,
    *,
    term_height_low: float,
    term_pitch: float,
    torque_sat_frac: Optional[float],
    prefer_height: bool = True,
) -> Tuple[TuningKnobs, str]:
    # Conservative step sizes; the goal is monotonic reduction in term fractions without destabilizing PPO.
    next_knobs = knobs
    reason = ""

    # If torque is saturating, avoid pushing penalties too hard; prefer shaping-only changes.
    torque_sat = float(torque_sat_frac) if torque_sat_frac is not None else 0.0
    if torque_sat > 0.05:
        reason += f"High torque saturation ({torque_sat:.1%}); applying smaller steps. "
        height_step_buf = 0.0025
        height_step_gate = 0.005
        w_step_h = 0.05
        w_step_vz = 0.03
        w_step_ori = 0.25
    else:
        height_step_buf = 0.005
        height_step_gate = 0.01
        w_step_h = 0.1
        w_step_vz = 0.05
        w_step_ori = 0.5

    # Decide which failure to target.
    target_height = prefer_height or (term_height_low >= term_pitch)
    if target_height:
        reason += f"Targeting height-low term (h_low={term_height_low:.2%}, pitch={term_pitch:.2%})."
        next_knobs = TuningKnobs(
            collapse_height_buffer=min(knobs.collapse_height_buffer + height_step_buf, 0.06),
            collapse_vz_gate_band=min(knobs.collapse_vz_gate_band + height_step_gate, 0.12),
            w_collapse_height=max(knobs.w_collapse_height - w_step_h, -1.2),
            w_collapse_vz=max(knobs.w_collapse_vz - w_step_vz, -0.8),
            w_orientation=knobs.w_orientation,
            w_clearance=knobs.w_clearance,
            w_flight_phase_penalty=knobs.w_flight_phase_penalty,
            w_posture=knobs.w_posture,
            posture_sigma=knobs.posture_sigma,
            push_force_max=knobs.push_force_max,
            push_duration_steps=knobs.push_duration_steps,
            w_action_rate=knobs.w_action_rate,
            w_torque=knobs.w_torque,
            w_gait_periodicity=knobs.w_gait_periodicity,
            w_hip_swing=knobs.w_hip_swing,
            w_knee_swing=knobs.w_knee_swing,
            hip_swing_min=knobs.hip_swing_min,
            knee_swing_min=knobs.knee_swing_min,
            w_step_event=knobs.w_step_event,
            w_foot_place=knobs.w_foot_place,
            min_height=knobs.min_height,
        )
    else:
        reason += f"Targeting pitch term (pitch={term_pitch:.2%}, h_low={term_height_low:.2%})."
        next_knobs = TuningKnobs(
            collapse_height_buffer=knobs.collapse_height_buffer,
            collapse_vz_gate_band=knobs.collapse_vz_gate_band,
            w_collapse_height=knobs.w_collapse_height,
            w_collapse_vz=knobs.w_collapse_vz,
            w_orientation=max(knobs.w_orientation - w_step_ori, -8.0),
            w_clearance=knobs.w_clearance,
            w_flight_phase_penalty=knobs.w_flight_phase_penalty,
            w_posture=knobs.w_posture,
            posture_sigma=knobs.posture_sigma,
            push_force_max=knobs.push_force_max,
            push_duration_steps=knobs.push_duration_steps,
            w_action_rate=knobs.w_action_rate,
            w_torque=knobs.w_torque,
            w_gait_periodicity=knobs.w_gait_periodicity,
            w_hip_swing=knobs.w_hip_swing,
            w_knee_swing=knobs.w_knee_swing,
            hip_swing_min=knobs.hip_swing_min,
            knee_swing_min=knobs.knee_swing_min,
            w_step_event=knobs.w_step_event,
            w_foot_place=knobs.w_foot_place,
            min_height=knobs.min_height,
        )
    return next_knobs, reason


class HeuristicAdvisor:
    """Deterministic, resume-safe tuner for standing_push-style runs.

    This stays in config-space only. It does not change policy_contract-sensitive fields.
    """

    def suggest(self, *, cfg: Dict[str, Any], metrics: MetricsRow, keys: EvalKeys, probe_is_short: bool) -> AdvisorDecision:
        knobs = _extract_knobs(cfg)
        term_height_low = float(_get_float(metrics.values, keys.term_height_low) or 0.0)
        term_pitch = float(_get_float(metrics.values, keys.term_pitch) or 0.0)
        torque_sat = _get_float(metrics.values, "debug/torque_sat_frac")
        survival_rate = _get_float(metrics.values, keys.survival_rate) or _get_float(metrics.values, keys.success) or 0.0

        # 1) Fix early termination first (existing collapse/orientation knobs).
        next_knobs, reason = _propose_next_knobs(
            knobs,
            term_height_low=term_height_low,
            term_pitch=term_pitch,
            torque_sat_frac=torque_sat,
        )

        # 2) If we still see height-low failures under pushes, make stepping "cheaper" and more likely.
        # Use eval term fractions as the trigger (works for both probe+confirm).
        if term_height_low > 0.005 or float(survival_rate) < 0.95:
            reason += " Also encouraging stepping recovery (clearance↑, stepping rewards↑, action_rate/torque penalties↓, and relax min_height if needed)."
            # If term_height_low dominates, slightly relax termination to allow deeper crouch recovery.
            # Keep conservative bounds: don't go below 0.36 without manual review.
            new_min_height = next_knobs.min_height
            if term_height_low > 0.2 and next_knobs.min_height > 0.36:
                new_min_height = max(next_knobs.min_height - 0.01, 0.36)

            # Ensure swing thresholds are non-zero so hip/knee swing rewards can activate.
            hip_min = next_knobs.hip_swing_min if next_knobs.hip_swing_min > 1e-6 else 0.10
            knee_min = next_knobs.knee_swing_min if next_knobs.knee_swing_min > 1e-6 else 0.10
            next_knobs = TuningKnobs(
                collapse_height_buffer=next_knobs.collapse_height_buffer,
                collapse_vz_gate_band=next_knobs.collapse_vz_gate_band,
                w_collapse_height=next_knobs.w_collapse_height,
                w_collapse_vz=next_knobs.w_collapse_vz,
                w_orientation=next_knobs.w_orientation,
                w_clearance=min(max(next_knobs.w_clearance * 1.25, next_knobs.w_clearance + 0.02), 0.5),
                w_flight_phase_penalty=min(next_knobs.w_flight_phase_penalty, 0.0),
                w_posture=next_knobs.w_posture,
                posture_sigma=next_knobs.posture_sigma,
                push_force_max=next_knobs.push_force_max,
                push_duration_steps=next_knobs.push_duration_steps,
                w_action_rate=min(next_knobs.w_action_rate * 0.75, -0.001),
                w_torque=min(next_knobs.w_torque * 0.75, -0.0002),
                w_gait_periodicity=min(max(next_knobs.w_gait_periodicity + 0.05, next_knobs.w_gait_periodicity * 1.2), 0.3),
                w_hip_swing=min(max(next_knobs.w_hip_swing + 0.05, next_knobs.w_hip_swing * 1.2), 0.3),
                w_knee_swing=min(max(next_knobs.w_knee_swing + 0.05, next_knobs.w_knee_swing * 1.2), 0.3),
                hip_swing_min=hip_min,
                knee_swing_min=knee_min,
                w_step_event=min(max(next_knobs.w_step_event + 0.05, next_knobs.w_step_event * 1.2), 0.6),
                w_foot_place=min(max(next_knobs.w_foot_place + 0.10, next_knobs.w_foot_place * 1.2), 1.5),
                min_height=new_min_height,
            )

        # 3) If the policy is stable but tends to stay "crouched/odd", increase posture return shaping.
        posture_mse = _get_float(metrics.values, "debug/posture_mse")
        if posture_mse is not None and term_height_low <= 0.002 and term_pitch <= 0.002 and posture_mse > 0.02:
            reason += " Increasing posture-return shaping (posture↑)."
            next_knobs = TuningKnobs(
                collapse_height_buffer=next_knobs.collapse_height_buffer,
                collapse_vz_gate_band=next_knobs.collapse_vz_gate_band,
                w_collapse_height=next_knobs.w_collapse_height,
                w_collapse_vz=next_knobs.w_collapse_vz,
                w_orientation=next_knobs.w_orientation,
                w_clearance=next_knobs.w_clearance,
                w_flight_phase_penalty=next_knobs.w_flight_phase_penalty,
                w_posture=min(max(next_knobs.w_posture + 0.1, next_knobs.w_posture * 1.2), 2.0),
                posture_sigma=next_knobs.posture_sigma,
                push_force_max=next_knobs.push_force_max,
                push_duration_steps=next_knobs.push_duration_steps,
                w_action_rate=next_knobs.w_action_rate,
                w_torque=next_knobs.w_torque,
                w_gait_periodicity=next_knobs.w_gait_periodicity,
                w_hip_swing=next_knobs.w_hip_swing,
                w_knee_swing=next_knobs.w_knee_swing,
                hip_swing_min=next_knobs.hip_swing_min,
                knee_swing_min=next_knobs.knee_swing_min,
                w_step_event=next_knobs.w_step_event,
                w_foot_place=next_knobs.w_foot_place,
                min_height=next_knobs.min_height,
            )

        updates: Dict[str, Any] = {
            "env.collapse_height_buffer": next_knobs.collapse_height_buffer,
            "env.collapse_vz_gate_band": next_knobs.collapse_vz_gate_band,
            "reward_weights.collapse_height": next_knobs.w_collapse_height,
            "reward_weights.collapse_vz": next_knobs.w_collapse_vz,
            "reward_weights.orientation": next_knobs.w_orientation,
            "reward_weights.clearance": next_knobs.w_clearance,
            "reward_weights.flight_phase_penalty": next_knobs.w_flight_phase_penalty,
            "reward_weights.posture": next_knobs.w_posture,
            "reward_weights.posture_sigma": next_knobs.posture_sigma,
            "reward_weights.action_rate": next_knobs.w_action_rate,
            "reward_weights.torque": next_knobs.w_torque,
            "reward_weights.gait_periodicity": next_knobs.w_gait_periodicity,
            "reward_weights.hip_swing": next_knobs.w_hip_swing,
            "reward_weights.knee_swing": next_knobs.w_knee_swing,
            "reward_weights.hip_swing_min": next_knobs.hip_swing_min,
            "reward_weights.knee_swing_min": next_knobs.knee_swing_min,
            "reward_weights.step_event": next_knobs.w_step_event,
            "reward_weights.foot_place": next_knobs.w_foot_place,
            "env.push_force_max": next_knobs.push_force_max,
            "env.push_duration_steps": next_knobs.push_duration_steps,
            "env.min_height": next_knobs.min_height,
        }
        return AdvisorDecision(updates=updates, reason=reason)


class OpenAIAdvisor:
    """LLM-based advisor (optional).

    Only used when explicitly requested via --advisor openai. Requires an API key in the environment.
    This agent only applies config updates (no code changes) and enforces strict allow/deny lists.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        dry_run: bool,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.dry_run = bool(dry_run)

    def suggest(self, *, cfg: Dict[str, Any], metrics: MetricsRow, keys: EvalKeys, probe_is_short: bool) -> AdvisorDecision:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"--advisor openai selected but env var {self.api_key_env!r} is not set."
            )

        knobs = _extract_knobs(cfg)
        context = {
            "probe_is_short": bool(probe_is_short),
            "forbidden_mutations": sorted(FORBIDDEN_MUTATIONS),
            "allowed_prefixes": list(ALLOWED_MUTATION_PREFIXES),
            "current_config_snippet": {
                "env": {
                    "push_force_max": knobs.push_force_max,
                    "push_duration_steps": knobs.push_duration_steps,
                    "collapse_height_buffer": knobs.collapse_height_buffer,
                    "collapse_vz_gate_band": knobs.collapse_vz_gate_band,
                },
                "reward_weights": {
                    "collapse_height": knobs.w_collapse_height,
                    "collapse_vz": knobs.w_collapse_vz,
                    "orientation": knobs.w_orientation,
                    "clearance": knobs.w_clearance,
                    "flight_phase_penalty": knobs.w_flight_phase_penalty,
                    "posture": knobs.w_posture,
                    "posture_sigma": knobs.posture_sigma,
                    "action_rate": knobs.w_action_rate,
                    "torque": knobs.w_torque,
                },
            },
            "best_metrics_row": {
                "iteration": metrics.iteration,
                "step": metrics.step,
                keys.success: _get_float(metrics.values, keys.success),
                keys.ep_len: _get_float(metrics.values, keys.ep_len),
                keys.survival_rate: _get_float(metrics.values, keys.survival_rate),
                keys.survival_steps: _get_float(metrics.values, keys.survival_steps),
                keys.term_height_low: _get_float(metrics.values, keys.term_height_low),
                keys.term_pitch: _get_float(metrics.values, keys.term_pitch),
                "debug/torque_sat_frac": _get_float(metrics.values, "debug/torque_sat_frac"),
                "debug/action_sat_frac": _get_float(metrics.values, "debug/action_sat_frac"),
                "debug/posture_mse": _get_float(metrics.values, "debug/posture_mse"),
            },
        }

        system = (
            "You are a PPO training config tuner for a biped standing_push task. "
            "Goal: higher eval_push survival/success and return-to-upright posture after disturbances. "
            "Respond with ONLY valid JSON."
        )
        user = (
            "Suggest the next config updates.\n"
            "Constraints:\n"
            f"- Forbidden keys: {sorted(FORBIDDEN_MUTATIONS)}\n"
            f"- Only keys starting with allowed prefixes: {list(ALLOWED_MUTATION_PREFIXES)}\n"
            "- Only numeric/bool values.\n"
            "Output JSON schema:\n"
            '{\"reason\": \"...\", \"updates\": {\"path.to.key\": 0.123, \"path.to.other\": 5}}\n\n'
            f"Context:\n{json.dumps(context, indent=2)}"
        )

        if self.dry_run:
            return AdvisorDecision(updates={}, reason=f"OpenAI dry-run. Prompt context:\n{user}")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
            raise RuntimeError(f"OpenAI advisor HTTP error: {e.code} {e.reason}\n{body}") from e
        except Exception as e:
            raise RuntimeError(f"OpenAI advisor request failed: {e}") from e

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenAI advisor response missing content: keys={list(data.keys())}") from e

        try:
            out = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"OpenAI advisor returned non-JSON content:\n{content}") from e

        if not isinstance(out, dict) or "updates" not in out:
            raise RuntimeError(f"OpenAI advisor JSON must be an object with 'updates': got {out!r}")
        updates = out.get("updates") or {}
        reason = str(out.get("reason") or "openai")
        if not isinstance(updates, dict):
            raise RuntimeError(f"OpenAI advisor 'updates' must be an object: got {updates!r}")
        updates_clean = _validate_updates(updates)
        return AdvisorDecision(updates=updates_clean, reason=reason)


def _write_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_yaml(cfg), encoding="utf-8")


def _run_training(cmd: List[str], *, env: Dict[str, str]) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    return int(proc.wait())


def _parse_run_artifacts(log_text: str) -> Tuple[Optional[Path], Optional[Path]]:
    # Extract:
    # - offline run dir: "training/wandb/offline-run-...-<id>"
    # - checkpoint dir: "Checkpoints will be saved to: training/checkpoints/<job>"
    run_dir = None
    ckpt_dir = None
    m1 = re.search(r"(training/wandb/offline-run-[^\s]+)", log_text)
    if m1:
        run_dir = Path(m1.group(1))
    m2 = re.search(r"Checkpoints will be saved to:\s+([^\s]+)", log_text)
    if m2:
        ckpt_dir = Path(m2.group(1))
    # Fallback: extract from the end-of-run hint:
    #   "To resume from best: --resume training/checkpoints/<job>/checkpoint_..."
    if ckpt_dir is None:
        m3 = re.search(r"--resume\s+(training/checkpoints/[^\s]+/checkpoint_[0-9]+_[0-9]+\.pkl)", log_text)
        if m3:
            ckpt_dir = Path(m3.group(1)).parent
    return run_dir, ckpt_dir


def _run_training_and_capture(
    cmd: List[str],
    *,
    env: Dict[str, str],
    live_stdout: bool = True,
    print_live_metrics: bool = True,
) -> Tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    printer = LiveMetricsPrinter()
    if print_live_metrics:
        printer.start(is_process_alive=lambda: proc.poll() is None)
    out_lines: List[str] = []
    run_dir: Optional[Path] = None
    for line in proc.stdout:
        out_lines.append(line)
        if print_live_metrics and run_dir is None:
            m = re.search(r"(training/wandb/offline-run-[^\s]+)", line)
            if m:
                run_dir = Path(m.group(1))
                printer.set_run_dir(run_dir)
        if live_stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    rc = int(proc.wait())
    printer.stop()
    return rc, "".join(out_lines)


def _find_checkpoint_dir(run_id: str, checkpoints_root: Path) -> Optional[Path]:
    candidates = sorted(checkpoints_root.glob(f"**/*{run_id}*"))
    dirs = [p for p in candidates if p.is_dir()]
    if not dirs:
        return None

    def score(p: Path) -> Tuple[int, float]:
        depth = len(p.parts)
        mtime = p.stat().st_mtime
        return (depth, -mtime)

    return sorted(dirs, key=score)[0]


def _find_recent_checkpoint_dir(checkpoints_root: Path, *, min_mtime: float) -> Optional[Path]:
    best: Optional[Path] = None
    best_mtime = float("-inf")
    for d in checkpoints_root.glob("**/*"):
        if not d.is_dir():
            continue
        try:
            mtime = d.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime < min_mtime:
            continue
        if not any(d.glob("checkpoint_*.pkl")):
            continue
        if mtime > best_mtime:
            best = d
            best_mtime = mtime
    return best


def _find_checkpoint_anywhere(
    checkpoints_root: Path, *, iteration: int, step: Optional[int]
) -> Optional[Path]:
    # Prefer exact match if step known.
    if step is not None:
        matches = sorted(checkpoints_root.glob(f"**/checkpoint_{iteration}_{step}.pkl"))
        if matches:
            return matches[0]
    # Otherwise any checkpoint for this iteration.
    matches = sorted(checkpoints_root.glob(f"**/checkpoint_{iteration}_*.pkl"))
    return matches[0] if matches else None


def _safe_mutate_config(
    base_cfg: Dict[str, Any],
    *,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # cheap deep-copy via json
    for key, value in updates.items():
        if key in FORBIDDEN_MUTATIONS:
            raise ValueError(f"Refusing to mutate {key!r}; it breaks resume policy_contract safety.")
        _set_nested(cfg, key, value)
    return cfg


def _apply_updates_inplace(cfg: Dict[str, Any], updates: Dict[str, Any]) -> None:
    cleaned = _validate_updates(updates)
    for key, value in cleaned.items():
        _set_nested(cfg, key, value)


def _parse_push_stages(spec: str) -> List[Tuple[float, int]]:
    """Parse push stages from a string like '9:15,12:15' -> [(9.0, 15), (12.0, 15)]."""
    spec = spec.strip()
    if not spec:
        return []
    out: List[Tuple[float, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --push-stages entry {part!r}; expected 'force:duration'.")
        force_s, dur_s = [x.strip() for x in part.split(":", 1)]
        out.append((float(force_s), int(dur_s)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Automate train/eval/tune/resume loop for WildRobot.")
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--resume", type=Path, required=True, help="Initial checkpoint to resume from.")
    ap.add_argument("--work-dir", type=Path, default=Path("training/configs/auto"))
    ap.add_argument("--checkpoints-root", type=Path, default=Path("training/checkpoints"))
    ap.add_argument("--iters-per-cycle", type=int, default=20)
    ap.add_argument("--max-cycles", type=int, default=10)
    ap.add_argument("--probe-eval-steps", type=int, default=200)
    ap.add_argument("--probe-eval-envs", type=int, default=64)
    ap.add_argument("--confirm-eval-steps", type=int, default=500)
    ap.add_argument("--confirm-eval-envs", type=int, default=128)
    ap.add_argument("--target-ep-len", type=float, default=500.0)
    ap.add_argument("--target-success", type=float, default=1.0)
    ap.add_argument("--term-eps", type=float, default=0.001)
    ap.add_argument(
        "--push-stages",
        type=str,
        default="",
        help="Optional push curriculum stages as 'force_max:duration_steps,...' (e.g. '9:15,12:15').",
    )
    ap.add_argument(
        "--advisor",
        type=str,
        default="heuristic",
        choices=["heuristic", "openai"],
        help="Tuning strategy. 'openai' calls an external LLM and applies validated config updates.",
    )
    ap.add_argument("--openai-model", type=str, default="gpt-4.1-mini")
    ap.add_argument(
        "--openai-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name containing the OpenAI API key.",
    )
    ap.add_argument(
        "--openai-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for an OpenAI-compatible API.",
    )
    ap.add_argument("--openai-temperature", type=float, default=0.2)
    ap.add_argument("--openai-max-tokens", type=int, default=600)
    ap.add_argument(
        "--openai-dry-run",
        action="store_true",
        help="Do not call the API; print the prompt context as the tuning reason.",
    )
    ap.add_argument(
        "--no-live-metrics",
        dest="live_metrics",
        action="store_false",
        default=True,
        help="Disable live printing of metrics.jsonl rows while training runs.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_cfg = _load_yaml(args.base_config)
    push_stages = _parse_push_stages(str(args.push_stages))
    push_stage_idx = 0
    if push_stages:
        # Start at stage 0 immediately.
        stage_force, stage_dur = push_stages[0]
        _set_nested(base_cfg, "env.push_force_max", float(stage_force))
        _set_nested(base_cfg, "env.push_duration_steps", int(stage_dur))
        base_cfg["version_name"] = (
            f"{base_cfg.get('version_name','').strip()} | push_stage=1/{len(push_stages)}"
            f" force_max={stage_force:g} dur={stage_dur}"
        ).strip()
    if args.advisor == "heuristic":
        advisor: Advisor = HeuristicAdvisor()
    else:
        advisor = OpenAIAdvisor(
            model=str(args.openai_model),
            api_key_env=str(args.openai_api_key_env),
            base_url=str(args.openai_base_url),
            temperature=float(args.openai_temperature),
            max_tokens=int(args.openai_max_tokens),
            dry_run=bool(args.openai_dry_run),
        )

    # Environment for subprocess training runs
    run_env = os.environ.copy()
    run_env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    run_env.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax-cache")

    current_cfg_path = args.work_dir / f"{args.base_config.stem}_auto_v{base_cfg.get('version','')}.yaml"
    current_cfg = json.loads(json.dumps(base_cfg))
    _write_config(current_cfg_path, current_cfg)
    current_resume = args.resume

    print(f"\nAgent work dir: {args.work_dir.as_posix()}")
    print(f"Starting config: {args.base_config.as_posix()}")
    print(f"Working config: {current_cfg_path.as_posix()}")
    print(f"Initial resume: {current_resume.as_posix()}")
    print(f"Forbidden mutations: {sorted(FORBIDDEN_MUTATIONS)}")

    for cycle_idx in range(1, args.max_cycles + 1):
        cycle_cfg = _load_yaml(current_cfg_path)
        _set_nested(cycle_cfg, "ppo.iterations", int(args.iters_per_cycle))
        _set_nested(cycle_cfg, "ppo.eval.num_steps", int(args.probe_eval_steps))
        _set_nested(cycle_cfg, "ppo.eval.num_envs", int(args.probe_eval_envs))
        if push_stages:
            stage_force, stage_dur = push_stages[push_stage_idx]
            _set_nested(cycle_cfg, "env.push_force_max", float(stage_force))
            _set_nested(cycle_cfg, "env.push_duration_steps", int(stage_dur))
        _set_nested(
            cycle_cfg,
            "version_name",
            f"{cycle_cfg.get('version_name','').strip()} | auto_cycle={cycle_idx} | probe_eval={args.probe_eval_steps}",
        )
        cycle_cfg_path = args.work_dir / f"{args.base_config.stem}_auto_cycle{cycle_idx}_{_now_ts()}.yaml"
        _write_config(cycle_cfg_path, cycle_cfg)

        cmd = [
            "uv",
            "run",
            "python",
            "training/train.py",
            "--config",
            str(cycle_cfg_path),
            "--resume",
            str(current_resume),
        ]
        print(f"\n=== Cycle {cycle_idx}/{args.max_cycles} ===")
        print(f"Config: {cycle_cfg_path}")
        print(f"Resume: {current_resume}")
        print(f"Cmd: {' '.join(cmd)}")

        if args.dry_run:
            print("Dry-run: skipping training.")
            return

        cycle_start = time.time()
        rc, text = _run_training_and_capture(
            cmd, env=run_env, live_stdout=True, print_live_metrics=bool(args.live_metrics)
        )
        if rc != 0:
            raise SystemExit(f"Training exited non-zero (rc={rc}).")

        run_dir, ckpt_dir = _parse_run_artifacts(text)
        if run_dir is None:
            raise SystemExit("Failed to parse offline run dir from training output.")
        if ckpt_dir is None:
            # Fallback: infer from run_id substring match in training/checkpoints.
            run_id = run_dir.name.split("-")[-1]
            ckpt_dir = _find_checkpoint_dir(run_id, args.checkpoints_root)
        if ckpt_dir is None:
            # Final fallback: most recently modified checkpoint dir after this cycle started.
            ckpt_dir = _find_recent_checkpoint_dir(args.checkpoints_root, min_mtime=cycle_start - 60.0)

        metrics_path = run_dir / "files" / "metrics.jsonl"
        # Wait briefly for metrics flush.
        for _ in range(30):
            if metrics_path.exists():
                break
            time.sleep(1.0)

        if not metrics_path.exists():
            raise SystemExit(
                "metrics.jsonl not found after run.\n"
                f"Run dir: {run_dir}\n"
                "If W&B failed to initialize, set `wandb.enabled: false` in the config and\n"
                "extend this agent to read trainer-side logs instead.\n"
                f"Tail debug.log:\n{_tail_lines(run_dir / 'logs' / 'debug.log')}"
            )

        raw_rows = _read_jsonl(metrics_path)
        parsed: List[MetricsRow] = []
        for rr in raw_rows:
            it = _get_iter(rr)
            if it is None:
                continue
            parsed.append(MetricsRow(iteration=it, step=_get_step(rr), values=rr))
        parsed.sort(key=lambda r: r.iteration)

        keys = _pick_eval_keys([r.values for r in parsed])
        max_ep_steps = int((_get_nested(cycle_cfg, "env.max_episode_steps") or 500))
        probe_is_short = int(args.probe_eval_steps) < max_ep_steps
        best = _best_row_by_survival(parsed, keys) if probe_is_short else _best_row_by_eval(parsed, keys)
        best_s = float(_get_float(best.values, keys.success) or 0.0)
        best_L = float(_get_float(best.values, keys.ep_len) or 0.0)
        best_surv_s = float(_get_float(best.values, keys.survival_rate) or 0.0)
        best_surv_L = float(_get_float(best.values, keys.survival_steps) or 0.0)
        best_h_low = float(_get_float(best.values, keys.term_height_low) or 0.0)
        best_pitch = float(_get_float(best.values, keys.term_pitch) or 0.0)
        torque_sat = _get_float(best.values, "debug/torque_sat_frac")

        best_ckpt: Optional[Path] = None
        if ckpt_dir is not None:
            best_ckpt = _find_checkpoint_for_iter(ckpt_dir, best.iteration, best.step)
        if best_ckpt is None:
            # Final fallback: search under checkpoints root (handles missing/unknown ckpt_dir).
            best_ckpt = _find_checkpoint_anywhere(
                args.checkpoints_root, iteration=best.iteration, step=best.step
            )
        if best_ckpt is None:
            raise SystemExit(
                f"Could not find checkpoint file for iter={best.iteration} (step={best.step}).\n"
                f"Parsed ckpt_dir={ckpt_dir}\n"
                f"checkpoints_root={args.checkpoints_root}"
            )

        print("\n=== Cycle summary ===")
        print(f"Run: {run_dir}")
        print(f"Checkpoints: {ckpt_dir or args.checkpoints_root}")
        print(f"Best iter: {best.iteration}")
        print(f"Best ckpt: {best_ckpt}")
        if probe_is_short:
            print(f"{keys.survival_rate}={best_surv_s:.2%}, {keys.survival_steps}={best_surv_L:.1f} (probe)")
            print(f"{keys.success}={best_s:.2%}, {keys.ep_len}={best_L:.2f} (may be truncation-based)")
        else:
            print(f"{keys.success}={best_s:.2%}, {keys.ep_len}={best_L:.2f}")
        print(f"{keys.term_height_low}={best_h_low:.2%}, {keys.term_pitch}={best_pitch:.2%}")
        if torque_sat is not None:
            print(f"debug/torque_sat_frac={float(torque_sat):.2%}")

        # If probe eval already meets the target, confirm with full eval steps.
        if probe_is_short:
            meets_success = best_surv_s >= args.target_success - 1e-9
            meets_len = best_surv_L >= float(args.probe_eval_steps) - 1e-6
        else:
            meets_success = best_s >= args.target_success - 1e-9
            meets_len = best_L >= args.target_ep_len - 1e-6
        meets_terms = (best_h_low <= args.term_eps) and (best_pitch <= args.term_eps)

        if meets_success and meets_len and meets_terms and args.probe_eval_steps == args.confirm_eval_steps:
            print("\nTARGET MET (full eval): stopping.")
            print(f"Final checkpoint: {best_ckpt}")
            return

        if meets_success and meets_terms and args.probe_eval_steps != args.confirm_eval_steps:
            print("\nProbe looks good; running confirm cycle with full eval (500 steps).")
            confirm_cfg = _load_yaml(current_cfg_path)
            _set_nested(confirm_cfg, "ppo.iterations", 10)
            _set_nested(confirm_cfg, "ppo.eval.num_steps", int(args.confirm_eval_steps))
            _set_nested(confirm_cfg, "ppo.eval.num_envs", int(args.confirm_eval_envs))
            _set_nested(
                confirm_cfg,
                "version_name",
                f"{confirm_cfg.get('version_name','').strip()} | confirm_eval={args.confirm_eval_steps}",
            )
            confirm_cfg_path = args.work_dir / f"{args.base_config.stem}_auto_confirm_{_now_ts()}.yaml"
            _write_config(confirm_cfg_path, confirm_cfg)
            confirm_cmd = [
                "uv",
                "run",
                "python",
                "training/train.py",
                "--config",
                str(confirm_cfg_path),
                "--resume",
                str(best_ckpt),
            ]
            confirm_start = time.time()
            rc2, text2 = _run_training_and_capture(
                confirm_cmd,
                env=run_env,
                live_stdout=True,
                print_live_metrics=bool(args.live_metrics),
            )
            if rc2 != 0:
                raise SystemExit(f"Confirm run exited non-zero (rc={rc2}).")

            run_dir2, ckpt_dir2 = _parse_run_artifacts(text2)
            if run_dir2 is None:
                raise SystemExit("Failed to parse confirm offline run dir from training output.")
            if ckpt_dir2 is None:
                run_id2 = run_dir2.name.split("-")[-1]
                ckpt_dir2 = _find_checkpoint_dir(run_id2, args.checkpoints_root)
            if ckpt_dir2 is None:
                ckpt_dir2 = _find_recent_checkpoint_dir(args.checkpoints_root, min_mtime=confirm_start - 60.0)
            metrics_path2 = run_dir2 / "files" / "metrics.jsonl"
            for _ in range(30):
                if metrics_path2.exists():
                    break
                time.sleep(1.0)
            raw_rows2 = _read_jsonl(metrics_path2)
            parsed2: List[MetricsRow] = []
            for rr in raw_rows2:
                it = _get_iter(rr)
                if it is None:
                    continue
                parsed2.append(MetricsRow(iteration=it, step=_get_step(rr), values=rr))
            parsed2.sort(key=lambda r: r.iteration)
            keys2 = _pick_eval_keys([r.values for r in parsed2])
            best2 = _best_row_by_eval(parsed2, keys2)
            s2 = float(_get_float(best2.values, keys2.success) or 0.0)
            L2 = float(_get_float(best2.values, keys2.ep_len) or 0.0)
            h2 = float(_get_float(best2.values, keys2.term_height_low) or 0.0)
            p2 = float(_get_float(best2.values, keys2.term_pitch) or 0.0)
            ckpt2: Optional[Path] = None
            if ckpt_dir2 is not None:
                ckpt2 = _find_checkpoint_for_iter(ckpt_dir2, best2.iteration, best2.step)
            if ckpt2 is None:
                ckpt2 = _find_checkpoint_anywhere(
                    args.checkpoints_root, iteration=best2.iteration, step=best2.step
                )
            print("\n=== Confirm summary ===")
            print(f"Run: {run_dir2}")
            print(f"Best ckpt: {ckpt2}")
            print(f"{keys2.success}={s2:.2%}, {keys2.ep_len}={L2:.2f}")
            print(f"{keys2.term_height_low}={h2:.2%}, {keys2.term_pitch}={p2:.2%}")
            if s2 >= args.target_success - 1e-9 and L2 >= args.target_ep_len - 1e-6 and h2 <= args.term_eps and p2 <= args.term_eps:
                if push_stages and push_stage_idx < len(push_stages) - 1:
                    # Advance curriculum and keep training from the best confirm checkpoint.
                    push_stage_idx += 1
                    next_force, next_dur = push_stages[push_stage_idx]
                    print(
                        "\nTARGET MET (confirm eval) for current push stage; advancing curriculum."
                        f" Next stage {push_stage_idx+1}/{len(push_stages)}: force_max={next_force:g} dur={next_dur}."
                    )
                    working_cfg2 = _load_yaml(current_cfg_path)
                    _set_nested(working_cfg2, "env.push_force_max", float(next_force))
                    _set_nested(working_cfg2, "env.push_duration_steps", int(next_dur))
                    working_cfg2["version_name"] = (
                        f"{working_cfg2.get('version_name','').strip()} | push_stage={push_stage_idx+1}/{len(push_stages)}"
                        f" force_max={next_force:g} dur={next_dur}"
                    ).strip()
                    _write_config(current_cfg_path, working_cfg2)
                    current_resume = ckpt2 or best_ckpt
                    continue
                print("\nTARGET MET (confirm eval): stopping.")
                if ckpt2 is not None:
                    print(f"Final checkpoint: {ckpt2}")
                return
            # Not met; continue tuning from best confirm checkpoint if available, else best probe.
            current_resume = ckpt2 or best_ckpt
            continue

        # Not good enough: tune and continue from best checkpoint.
        working_cfg = _load_yaml(current_cfg_path)
        decision = advisor.suggest(cfg=working_cfg, metrics=best, keys=keys, probe_is_short=probe_is_short)
        print(f"\nTuning decision: {decision.reason}")
        print(f"Updates: {decision.updates}")

        # Apply config updates to working config (not the original base config).
        _apply_updates_inplace(working_cfg, decision.updates)
        _write_config(current_cfg_path, working_cfg)
        current_resume = best_ckpt

    raise SystemExit("Max cycles reached without meeting target.")


if __name__ == "__main__":
    main()
