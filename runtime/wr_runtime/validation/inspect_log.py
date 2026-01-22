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

from policy_contract.numpy.frames import gravity_local_from_quat, normalize_quat_xyzw


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


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
    required = ["quat_xyzw", "gyro_rad_s", "joint_pos_rad", "joint_vel_rad_s", "foot_switches", "velocity_cmd"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys: {missing}. Found keys: {keys}")

    quat = np.asarray(data["quat_xyzw"], dtype=np.float32)
    gyro = np.asarray(data["gyro_rad_s"], dtype=np.float32)
    joint_pos = np.asarray(data["joint_pos_rad"], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel_rad_s"], dtype=np.float32)
    foot = np.asarray(data["foot_switches"], dtype=np.float32)
    vel_cmd = np.asarray(data["velocity_cmd"], dtype=np.float32)

    t = int(quat.shape[0])

    print(f"Log: {path}")
    print(f"Keys: {', '.join(keys)}")
    print(f"Steps: {t}")
    print(f"Shapes: quat{tuple(quat.shape)} gyro{tuple(gyro.shape)} qpos{tuple(joint_pos.shape)} qvel{tuple(joint_vel.shape)} foot{tuple(foot.shape)} vel_cmd{tuple(vel_cmd.shape)}")

    if "timestamp_s" in data:
        ts = np.asarray(data["timestamp_s"], dtype=np.float64).reshape(-1)
        if ts.size == t and np.all(np.isfinite(ts)):
            dt = np.diff(ts)
            dt_stats = _safe_stats_1d(dt)
            print(f"Time: timestamp_s present (dt mean={dt_stats.get('mean', float('nan')):.6f}s, p95={dt_stats.get('p95', float('nan')):.6f}s, max={dt_stats.get('max', float('nan')):.6f}s)")
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
    quat_norm_stats = _safe_stats_1d(quat_norm)
    print(
        "IMU: quat_norm "
        f"(min={quat_norm_stats.get('min', float('nan')):.6f}, "
        f"mean={quat_norm_stats.get('mean', float('nan')):.6f}, "
        f"max={quat_norm_stats.get('max', float('nan')):.6f}, "
        f"std={quat_norm_stats.get('std', float('nan')):.6f})"
    )

    quat_n = np.stack([normalize_quat_xyzw(q) for q in quat.astype(np.float32)], axis=0)
    g_local = np.stack([gravity_local_from_quat(q) for q in quat_n], axis=0)
    g_z = g_local[:, 2]
    g_z_stats = _safe_stats_1d(g_z)
    print(f"IMU: gravity_local.z (mean={g_z_stats.get('mean', float('nan')):.4f}, std={g_z_stats.get('std', float('nan')):.4f})  (upright target ≈ -1)")

    gyro_norm = np.linalg.norm(gyro.astype(np.float64), axis=1)
    gyro_norm_stats = _safe_stats_1d(gyro_norm)
    print(
        "IMU: gyro_norm(rad/s) "
        f"(mean={gyro_norm_stats.get('mean', float('nan')):.4f}, "
        f"p95={gyro_norm_stats.get('p95', float('nan')):.4f}, "
        f"max={gyro_norm_stats.get('max', float('nan')):.4f})"
    )

    # Foot switches
    if foot.ndim == 2 and foot.shape[1] == 4:
        pressed_rate = np.mean(foot > 0.5, axis=0)
        print(
            "Foot: pressed fraction (left_toe, left_heel, right_toe, right_heel) = "
            f"[{_pct(float(pressed_rate[0]))}, {_pct(float(pressed_rate[1]))}, {_pct(float(pressed_rate[2]))}, {_pct(float(pressed_rate[3]))}]"
        )
    else:
        print("Foot: unexpected shape, expected (T,4)")

    # Joint stats (coarse)
    jp_stats = _safe_stats_1d(joint_pos)
    jv_stats = _safe_stats_1d(joint_vel)
    print(f"Joints: pos_rad overall (min={jp_stats.get('min', float('nan')):.3f}, max={jp_stats.get('max', float('nan')):.3f})")
    print(f"Joints: vel_rad_s overall (p95={jv_stats.get('p95', float('nan')):.3f}, max={jv_stats.get('max', float('nan')):.3f})")

    # Basic health heuristics
    quat_change = float(np.mean(np.linalg.norm(np.diff(quat_n, axis=0), axis=1))) if t > 1 else 0.0
    if quat_change < 1e-4:
        print("Heuristic: IMU orientation looks nearly constant (check sensor streaming / axis_map).")

    if float(gyro_norm_stats.get("p95", 0.0)) < 1e-3:
        print("Heuristic: gyro is near-zero (sensor may be stuck or robot was perfectly still).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a WildRobot runtime signals_log.npz")
    parser.add_argument("--input", type=str, required=True, help="Path to signals_log.npz")
    args = parser.parse_args()
    inspect_log(Path(args.input))


if __name__ == "__main__":
    main()

