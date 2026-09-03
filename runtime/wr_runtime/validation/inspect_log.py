from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_ROOT = _REPO_ROOT / "runtime"
for _p in (str(_REPO_ROOT), str(_RUNTIME_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from policy_contract.numpy.frames import gravity_local_from_quat, normalize_quat_wxyz

# Keep the original replay field set so this tool accepts both historical logs
# and the current structured policy telemetry archives.
_REQUIRED_REPLAY_LOG_FIELDS = (
    "quat_wxyz",
    "gyro_rad_s",
    "joint_pos_rad",
    "joint_vel_rad_s",
    "foot_switches",
    "velocity_cmd",
    "yaw_rate_cmd",
)


def required_replay_log_fields():
    return _REQUIRED_REPLAY_LOG_FIELDS


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def _safe_stats_1d(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not np.any(finite):
        return {"count": float(x.size), "finite": 0.0}
    xf = x[finite]
    return {
        "count": float(x.size),
        "finite": float(np.sum(finite)),
        "min": float(np.min(xf)),
        "p05": float(np.quantile(xf, 0.05)),
        "mean": float(np.mean(xf)),
        "p95": float(np.quantile(xf, 0.95)),
        "max": float(np.max(xf)),
        "std": float(np.std(xf)),
    }


def inspect_log(path: Path) -> None:
    data = _load_npz(path)
    keys = sorted(data.keys())
    if not keys:
        raise ValueError(f"No arrays found in {path}")

    # Required-by-convention keys (created by run_policy.py)
    required = [k for k in required_replay_log_fields() if k != "quat_wxyz"]
    missing = [k for k in required if k not in data]
    if "quat_wxyz" not in data and "quat_xyzw" not in data:
        missing.append("quat_wxyz")
    if missing:
        raise ValueError(f"Missing required keys: {missing}. Found keys: {keys}")

    if "quat_wxyz" in data:
        quat = np.asarray(data["quat_wxyz"], dtype=np.float32)
    else:
        legacy_xyzw = np.asarray(data["quat_xyzw"], dtype=np.float32)
        quat = np.concatenate(
            [legacy_xyzw[..., 3:4], legacy_xyzw[..., :3]], axis=-1
        )
    gyro = np.asarray(data["gyro_rad_s"], dtype=np.float32)
    joint_pos = np.asarray(data["joint_pos_rad"], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel_rad_s"], dtype=np.float32)
    foot = np.asarray(data["foot_switches"], dtype=np.float32)
    vel_cmd = np.asarray(data["velocity_cmd"], dtype=np.float32)
    yaw_rate_cmd = np.asarray(data["yaw_rate_cmd"], dtype=np.float32)

    t = int(quat.shape[0])

    print(f"Log: {path}")
    print(f"Keys: {', '.join(keys)}")
    print(f"Steps: {t}")
    if "outcome" in data:
        outcome = str(np.asarray(data["outcome"]).item())
        error = str(np.asarray(data.get("error", "")).item())
        print(f"Outcome: {outcome}" + (f" ({error})" if error else ""))
    phase = None
    analysis_mask = np.ones(t, dtype=bool)
    if "phase" in data:
        phase = np.asarray(data["phase"]).astype(str).reshape(-1)
        if phase.size == t:
            names, counts = np.unique(phase, return_counts=True)
            print(
                "Phases: "
                + ", ".join(
                    f"{name}={int(count)}" for name, count in zip(names, counts)
                )
            )
            walking_mask = phase == "walking"
            if np.any(walking_mask):
                analysis_mask = walking_mask
                print(f"Primary diagnostic phase: walking ({int(np.sum(walking_mask))} samples)")
    print(
        f"Shapes: quat{tuple(quat.shape)} gyro{tuple(gyro.shape)} qpos{tuple(joint_pos.shape)} "
        f"qvel{tuple(joint_vel.shape)} foot{tuple(foot.shape)} vel_cmd{tuple(vel_cmd.shape)} "
        f"yaw_rate_cmd{tuple(yaw_rate_cmd.shape)}"
    )

    if "host_monotonic_s" in data:
        host_ts = np.asarray(data["host_monotonic_s"], dtype=np.float64).reshape(-1)
        if host_ts.size == t and np.all(np.isfinite(host_ts[analysis_mask])):
            dt_stats = _safe_stats_1d(np.diff(host_ts[analysis_mask]))
            print(
                "Time: host loop dt "
                f"(mean={dt_stats.get('mean', float('nan')):.6f}s, "
                f"p95={dt_stats.get('p95', float('nan')):.6f}s, "
                f"max={dt_stats.get('max', float('nan')):.6f}s)"
            )
    if "timestamp_s" in data:
        ts = np.asarray(data["timestamp_s"], dtype=np.float64).reshape(-1)
        if ts.size == t and np.all(np.isfinite(ts)):
            dt = np.diff(ts[analysis_mask])
            dt_stats = _safe_stats_1d(dt)
            print(
                "Time: sensor dt "
                f"(mean={dt_stats.get('mean', float('nan')):.6f}s, "
                f"p95={dt_stats.get('p95', float('nan')):.6f}s, "
                f"max={dt_stats.get('max', float('nan')):.6f}s)"
            )
        else:
            print("Time: timestamp_s present but unusable (shape mismatch or non-finite)")
    elif "dt_s" in data:
        dt_s = np.asarray(data["dt_s"], dtype=np.float64).reshape(-1)
        if dt_s.size == t and np.all(np.isfinite(dt_s)):
            dt_stats = _safe_stats_1d(dt_s)
            print(f"Time: dt_s present (mean={dt_stats.get('mean', float('nan')):.6f}s, p95={dt_stats.get('p95', float('nan')):.6f}s, max={dt_stats.get('max', float('nan')):.6f}s)")
        else:
            print("Time: dt_s present but unusable (shape mismatch or non-finite)")

    quat_norm = np.linalg.norm(quat.astype(np.float64), axis=1)
    quat_norm_stats = _safe_stats_1d(quat_norm[analysis_mask])
    print(
        "IMU: quat_norm "
        f"(min={quat_norm_stats.get('min', float('nan')):.6f}, "
        f"mean={quat_norm_stats.get('mean', float('nan')):.6f}, "
        f"max={quat_norm_stats.get('max', float('nan')):.6f}, "
        f"std={quat_norm_stats.get('std', float('nan')):.6f})"
    )

    quat_n = np.stack(
        [normalize_quat_wxyz(q) for q in quat.astype(np.float32)], axis=0
    )
    g_local = np.stack([gravity_local_from_quat(q) for q in quat_n], axis=0)
    g_z = g_local[:, 2]
    g_z_stats = _safe_stats_1d(g_z[analysis_mask])
    print(f"IMU: gravity_local.z (mean={g_z_stats.get('mean', float('nan')):.4f}, std={g_z_stats.get('std', float('nan')):.4f})  (upright target ≈ -1)")

    gyro_norm = np.linalg.norm(gyro.astype(np.float64), axis=1)
    gyro_norm_stats = _safe_stats_1d(gyro_norm[analysis_mask])
    print(
        "IMU: gyro_norm(rad/s) "
        f"(mean={gyro_norm_stats.get('mean', float('nan')):.4f}, "
        f"p95={gyro_norm_stats.get('p95', float('nan')):.4f}, "
        f"max={gyro_norm_stats.get('max', float('nan')):.4f})"
    )

    tilt_deg = np.rad2deg(
        np.arccos(np.clip(-g_local[:, 2].astype(np.float64), -1.0, 1.0))
    )
    tilt_stats = _safe_stats_1d(tilt_deg[analysis_mask])
    print(
        "IMU: tilt_deg "
        f"(mean={tilt_stats.get('mean', float('nan')):.3f}, "
        f"p95={tilt_stats.get('p95', float('nan')):.3f}, "
        f"max={tilt_stats.get('max', float('nan')):.3f})"
    )

    # Foot switches
    if foot.ndim == 2 and foot.shape[1] == 4:
        pressed_rate = np.mean(foot[analysis_mask] > 0.5, axis=0)
        print(
            "Foot: pressed fraction (left_toe, left_heel, right_toe, right_heel) = "
            f"[{_pct(float(pressed_rate[0]))}, {_pct(float(pressed_rate[1]))}, {_pct(float(pressed_rate[2]))}, {_pct(float(pressed_rate[3]))}]"
        )
    else:
        print("Foot: unexpected shape, expected (T,4)")

    # Joint stats (coarse)
    jp_stats = _safe_stats_1d(joint_pos[analysis_mask])
    jv_stats = _safe_stats_1d(joint_vel[analysis_mask])
    print(f"Joints: pos_rad overall (min={jp_stats.get('min', float('nan')):.3f}, max={jp_stats.get('max', float('nan')):.3f})")
    print(f"Joints: vel_rad_s overall (p95={jv_stats.get('p95', float('nan')):.3f}, max={jv_stats.get('max', float('nan')):.3f})")

    actuator_names = _actuator_names(data, joint_pos.shape[-1])
    if "joint_tracking_error_rad" in data:
        tracking_error = np.asarray(
            data["joint_tracking_error_rad"], dtype=np.float32
        )
        if tracking_error.shape == joint_pos.shape:
            _print_per_joint_ranking(
                "Previous-command vs feedback error",
                np.rad2deg(np.abs(tracking_error[analysis_mask])),
                actuator_names,
                unit="deg",
            )
    if "applied_action" in data:
        applied_action = np.asarray(data["applied_action"], dtype=np.float32)
        if applied_action.ndim == 2 and applied_action.shape[0] == t:
            action_slice = np.abs(applied_action[analysis_mask])
            saturation = np.mean(action_slice > 0.95, axis=0)
            _print_per_joint_values(
                "Applied action >95% occupancy",
                saturation,
                actuator_names,
                unit="%",
                scale=100.0,
            )
    if "servo_position_age_s" in data:
        cache_age = np.asarray(data["servo_position_age_s"], dtype=np.float32)
        if cache_age.ndim == 2 and cache_age.shape[0] == t:
            _print_per_joint_ranking(
                "Servo feedback cache age",
                1000.0 * cache_age[analysis_mask],
                actuator_names,
                unit="ms",
            )

    timing_parts = []
    for key, label in (
        ("timing_work", "work"),
        ("timing_read", "read"),
        ("timing_policy", "policy"),
        ("timing_write", "write"),
        ("timing_io_servo_cache_age_max_s", "servo_cache_age"),
    ):
        if key not in data:
            continue
        values = np.asarray(data[key], dtype=np.float64).reshape(-1)
        if values.size != t:
            continue
        stats = _safe_stats_1d(1000.0 * values[analysis_mask])
        timing_parts.append(
            f"{label}={stats.get('p95', float('nan')):.3f}/"
            f"{stats.get('max', float('nan')):.3f}"
        )
    if timing_parts:
        print("Timing p95/max (ms): " + ", ".join(timing_parts))
    if "timing_work" in data and "ctrl_dt_s" in data:
        work = np.asarray(data["timing_work"], dtype=np.float64).reshape(-1)
        ctrl_dt = float(np.asarray(data["ctrl_dt_s"]).item())
        if work.size == t and np.isfinite(ctrl_dt) and ctrl_dt > 0.0:
            work = work[analysis_mask]
            finite = np.isfinite(work)
            if np.any(finite):
                misses = float(np.mean(work[finite] > ctrl_dt))
                print(f"Timing: work deadline misses={_pct(misses)}")

    if "footswitch_available" in data:
        available = np.asarray(data["footswitch_available"], dtype=bool).reshape(-1)
        if available.size == t and not bool(np.all(available[analysis_mask])):
            print(
                "Warning: foot switches were disabled or unavailable during the "
                "primary diagnostic phase."
            )

    # Basic health heuristics
    diagnostic_quat = quat_n[analysis_mask]
    quat_change = (
        float(np.mean(np.linalg.norm(np.diff(diagnostic_quat, axis=0), axis=1)))
        if diagnostic_quat.shape[0] > 1
        else 0.0
    )
    if quat_change < 1e-4:
        print("Heuristic: IMU orientation looks nearly constant (check sensor streaming / axis_map).")

    if float(gyro_norm_stats.get("p95", 0.0)) < 1e-3:
        print("Heuristic: gyro is near-zero (sensor may be stuck or robot was perfectly still).")


def _actuator_names(data: Dict[str, np.ndarray], count: int) -> list[str]:
    if "actuator_names" not in data:
        return [f"joint_{index}" for index in range(count)]
    names = np.asarray(data["actuator_names"]).astype(str).reshape(-1).tolist()
    if len(names) != count:
        return [f"joint_{index}" for index in range(count)]
    return names


def _print_per_joint_ranking(
    label: str,
    values: np.ndarray,
    names: list[str],
    *,
    unit: str,
    limit: int = 5,
) -> None:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        return
    p95 = np.nanpercentile(matrix, 95.0, axis=0)
    maximum = np.nanmax(matrix, axis=0)
    order = np.argsort(np.nan_to_num(p95, nan=-np.inf))[::-1][:limit]
    print(
        f"{label} top {len(order)} (p95/max {unit}): "
        + ", ".join(
            f"{names[index]}={p95[index]:.3f}/{maximum[index]:.3f}"
            for index in order
        )
    )


def _print_per_joint_values(
    label: str,
    values: np.ndarray,
    names: list[str],
    *,
    unit: str,
    scale: float = 1.0,
    limit: int = 5,
) -> None:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != len(names):
        return
    order = np.argsort(np.nan_to_num(vector, nan=-np.inf))[::-1][:limit]
    print(
        f"{label} top {len(order)} ({unit}): "
        + ", ".join(
            f"{names[index]}={vector[index] * scale:.3f}{unit}" for index in order
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a WildRobot runtime signals_log.npz")
    parser.add_argument("--input", type=str, required=True, help="Path to signals_log.npz")
    args = parser.parse_args()
    inspect_log(Path(args.input))


if __name__ == "__main__":
    main()
