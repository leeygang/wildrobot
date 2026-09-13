#!/usr/bin/env python3
"""Fit the low-load HTD-45H position-servo model from hardware captures.

The fitter follows ToddlerBot's system-identification structure: replay the
measured command sequence in MuJoCo and minimize both time-domain position
error and frequency-domain magnitude error. WildRobot's HTD bus does not
expose a configurable/known position gain, so this tool additionally fits the
effective ``kp``. Armature remains fixed because these captures do not
separate it reliably from damping. The fixture reaches less than 0.5 Nm, so
the vendor torque limit is deliberately not fitted or modified here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import mujoco
import numpy as np
from scipy.optimize import least_squares


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "assets" / "bam" / "robot.xml"
DEFAULT_JOINT_PROPERTIES = REPO_ROOT / "assets" / "v2" / "joints_properties.xml"
DEFAULT_GENERATED_MJCF = REPO_ROOT / "assets" / "v2" / "wildrobot.xml"
DEFAULT_REPORT_DIR = REPO_ROOT / "runtime" / "calibration" / "servo_sysid"


@dataclass(frozen=True)
class ServoDynamics:
    kp: float
    kv: float
    damping: float
    armature: float
    frictionloss: float
    delay_steps: int


@dataclass(frozen=True)
class CaptureTrace:
    path: Path
    manifest_path: Path
    label: str
    center_deg: float
    sample_hz: float
    fixture_direction: int
    fixture_qpos_offset_rad: float
    fixture_sha256: str
    command_rad: np.ndarray
    measured_position_rad: np.ndarray
    timestamps_s: np.ndarray
    position_valid: np.ndarray
    segment_name: np.ndarray
    chirp_end_hz: float


class FixtureReplay:
    """Reusable one-DOF MuJoCo replay model for one capture."""

    def __init__(
        self,
        fixture_path: Path,
        *,
        joint_name: str,
        trace: CaptureTrace,
        sim_dt: float,
        force_limit_nm: float,
    ) -> None:
        self.trace = trace
        self.model = mujoco.MjModel.from_xml_path(str(fixture_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(sim_dt)
        self.sim_dt = float(sim_dt)

        self.joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if self.joint_id < 0:
            raise ValueError(f"fixture joint not found: {joint_name!r}")
        self.actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        if self.actuator_id < 0:
            raise ValueError(f"fixture actuator not found: {joint_name!r}")
        self.qpos_address = int(self.model.jnt_qposadr[self.joint_id])
        self.dof_address = int(self.model.jnt_dofadr[self.joint_id])
        self.model.actuator_forcerange[self.actuator_id] = (
            -float(force_limit_nm),
            float(force_limit_nm),
        )

        sample_dt = float(np.median(np.diff(trace.timestamps_s)))
        self.steps_per_sample = int(round(sample_dt / self.sim_dt))
        if self.steps_per_sample < 1 or not math.isclose(
            self.steps_per_sample * self.sim_dt,
            sample_dt,
            rel_tol=0.0,
            abs_tol=max(5e-4, 0.1 * self.sim_dt),
        ):
            raise ValueError(
                f"capture {trace.path} sample period {sample_dt:.6f}s is not "
                f"compatible with simulation dt {self.sim_dt:.6f}s"
            )

    def simulate(self, params: ServoDynamics) -> np.ndarray:
        trace = self.trace
        model = self.model
        data = self.data
        actuator_id = self.actuator_id
        dof_address = self.dof_address
        qpos_address = self.qpos_address

        model.dof_damping[dof_address] = float(params.damping)
        model.dof_armature[dof_address] = float(params.armature)
        model.dof_frictionloss[dof_address] = float(params.frictionloss)
        model.actuator_gainprm[actuator_id, 0] = float(params.kp)
        model.actuator_biasprm[actuator_id, 1] = -float(params.kp)
        model.actuator_biasprm[actuator_id, 2] = -float(params.kv)

        direction = float(trace.fixture_direction)
        offset = float(trace.fixture_qpos_offset_rad)
        measured = np.asarray(trace.measured_position_rad, dtype=np.float64)
        command = np.asarray(trace.command_rad, dtype=np.float64)

        mujoco.mj_resetData(model, data)
        data.qpos[qpos_address] = offset + direction * measured[0]
        data.qvel[dof_address] = 0.0
        mujoco.mj_forward(model, data)

        predicted = np.empty(measured.shape, dtype=np.float64)
        predicted[0] = measured[0]
        delay = int(params.delay_steps)
        for sample_index in range(1, measured.size):
            command_index = max(0, sample_index - 1 - delay)
            data.ctrl[actuator_id] = (
                offset + direction * float(command[command_index])
            )
            mujoco.mj_step(model, data, nstep=self.steps_per_sample)
            predicted[sample_index] = direction * (
                float(data.qpos[qpos_address]) - offset
            )
        return predicted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_capture(path: Path) -> CaptureTrace:
    path = path.expanduser().resolve()
    manifest_path = path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"capture not found: {path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"capture manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("outcome") != "completed":
        raise ValueError(f"capture did not complete: {manifest_path}")
    if bool(manifest.get("prepare_only")):
        raise ValueError(f"preparation-only capture cannot be fitted: {path}")

    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "command_rad",
            "measured_position_rad",
            "timestamps_s",
            "position_valid",
            "segment_name",
        }
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"capture {path} is missing arrays: {missing}")
        command = np.asarray(arrays["command_rad"], dtype=np.float64)
        measured = np.asarray(arrays["measured_position_rad"], dtype=np.float64)
        timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
        valid = np.asarray(arrays["position_valid"], dtype=bool)
        segments = np.asarray(arrays["segment_name"], dtype=str)

    expected_shape = command.shape
    if command.ndim != 1 or command.size < 2:
        raise ValueError(f"capture {path} has no usable profile samples")
    for name, values in (
        ("measured_position_rad", measured),
        ("timestamps_s", timestamps),
        ("position_valid", valid),
        ("segment_name", segments),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"capture {path} array {name} has shape {values.shape}, "
                f"expected {expected_shape}"
            )
    if not np.all(valid):
        raise ValueError(f"capture {path} contains invalid position samples")
    if not np.all(np.isfinite(command)) or not np.all(np.isfinite(measured)):
        raise ValueError(f"capture {path} contains non-finite command/position data")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"capture {path} timestamps are not strictly increasing")

    fixture_sha = str(manifest.get("fixture_mjcf_sha256") or "")
    if not fixture_sha:
        raise ValueError(f"capture {path} does not record its fixture hash")

    return CaptureTrace(
        path=path,
        manifest_path=manifest_path,
        label=path.stem,
        center_deg=float(manifest["center_deg"]),
        sample_hz=float(manifest["sample_hz"]),
        fixture_direction=int(manifest["fixture_direction"]),
        fixture_qpos_offset_rad=math.radians(
            float(manifest["fixture_qpos_offset_deg"])
        ),
        fixture_sha256=fixture_sha,
        command_rad=command,
        measured_position_rad=measured,
        timestamps_s=timestamps,
        position_valid=valid,
        segment_name=segments,
        chirp_end_hz=float(manifest["chirp_end_hz"]),
    )


def load_xml_dynamics(path: Path) -> ServoDynamics:
    text = path.read_text()
    block_match = re.search(
        r'<default\s+class="htd45hServo">(?P<body>.*?)</default>',
        text,
        flags=re.DOTALL,
    )
    if block_match is None:
        raise ValueError(f"htd45hServo default not found in {path}")
    block = block_match.group("body")
    joint_match = re.search(r"<joint\b(?P<attrs>[^>]*)/>", block)
    position_match = re.search(r"<position\b(?P<attrs>[^>]*)/>", block)
    if joint_match is None or position_match is None:
        raise ValueError(f"incomplete htd45hServo default in {path}")

    def attr(attrs: str, name: str) -> float:
        match = re.search(rf'\b{name}="([^"]+)"', attrs)
        if match is None:
            raise ValueError(f"missing {name!r} in htd45hServo default in {path}")
        return float(match.group(1))

    return ServoDynamics(
        kp=attr(position_match.group("attrs"), "kp"),
        kv=attr(position_match.group("attrs"), "kv"),
        damping=attr(joint_match.group("attrs"), "damping"),
        armature=attr(joint_match.group("attrs"), "armature"),
        frictionloss=attr(joint_match.group("attrs"), "frictionloss"),
        delay_steps=0,
    )


def _replace_xml_attribute(tag: str, name: str, value: float) -> str:
    formatted = f"{float(value):.8g}"
    pattern = re.compile(rf'(\b{name}=")[^"]*(")')
    if pattern.search(tag) is None:
        raise ValueError(f"attribute {name!r} not found in tag {tag!r}")
    return pattern.sub(rf"\g<1>{formatted}\g<2>", tag, count=1)


def update_xml_dynamics(path: Path, params: ServoDynamics) -> None:
    """Update only the shared HTD default while preserving XML formatting."""

    text = path.read_text()
    block_pattern = re.compile(
        r'(<default\s+class="htd45hServo">)(?P<body>.*?)(</default>)',
        flags=re.DOTALL,
    )
    block_match = block_pattern.search(text)
    if block_match is None:
        raise ValueError(f"htd45hServo default not found in {path}")
    body = block_match.group("body")

    joint_pattern = re.compile(r"<joint\b[^>]*/>")
    position_pattern = re.compile(r"<position\b[^>]*/>")
    joint_match = joint_pattern.search(body)
    position_match = position_pattern.search(body)
    if joint_match is None or position_match is None:
        raise ValueError(f"incomplete htd45hServo default in {path}")

    joint_tag = joint_match.group(0)
    for name, value in (
        ("damping", params.damping),
        ("frictionloss", params.frictionloss),
        ("armature", params.armature),
    ):
        joint_tag = _replace_xml_attribute(joint_tag, name, value)
    position_tag = _replace_xml_attribute(
        position_match.group(0), "kp", params.kp
    )
    position_tag = _replace_xml_attribute(position_tag, "kv", params.kv)

    body = body[: joint_match.start()] + joint_tag + body[joint_match.end() :]
    position_match = position_pattern.search(body)
    assert position_match is not None
    body = (
        body[: position_match.start()]
        + position_tag
        + body[position_match.end() :]
    )
    updated = text[: block_match.start("body")] + body + text[block_match.end("body") :]
    path.write_text(updated)


def _frequency_residuals(
    trace: CaptureTrace,
    predicted: np.ndarray,
    *,
    weight: float,
) -> list[np.ndarray]:
    residuals: list[np.ndarray] = []
    names = tuple(dict.fromkeys(trace.segment_name.tolist()))
    for name in names:
        if not name.startswith("chirp_"):
            continue
        mask = trace.segment_name == name
        measured = trace.measured_position_rad[mask]
        simulated = predicted[mask]
        count = measured.size
        if count < 4:
            continue
        dt = float(np.median(np.diff(trace.timestamps_s[mask])))
        frequencies = np.fft.rfftfreq(count, d=dt)
        max_frequency = min(7.0, float(trace.chirp_end_hz))
        selected = (frequencies > 0.0) & (frequencies <= max_frequency)
        measured_fft = np.abs(np.fft.rfft(measured - np.mean(measured))) / count
        simulated_fft = np.abs(np.fft.rfft(simulated - np.mean(simulated))) / count
        difference = simulated_fft[selected] - measured_fft[selected]
        if difference.size:
            residuals.append(
                math.sqrt(float(weight)) * difference / math.sqrt(difference.size)
            )
    return residuals


def _unpack_parameters(
    values: np.ndarray, *, fixed: ServoDynamics, delay_steps: int
) -> ServoDynamics:
    kp, damping, frictionloss = np.exp(values)
    return ServoDynamics(
        kp=float(kp),
        kv=float(fixed.kv),
        damping=float(damping),
        armature=float(fixed.armature),
        frictionloss=float(frictionloss),
        delay_steps=int(delay_steps),
    )


def _parameter_vector(params: ServoDynamics) -> np.ndarray:
    return np.log(
        np.asarray(
            [params.kp, params.damping, params.frictionloss],
            dtype=np.float64,
        )
    )


def build_residual_vector(
    replays: Sequence[FixtureReplay],
    params: ServoDynamics,
    *,
    frequency_weight: float,
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    for replay in replays:
        trace = replay.trace
        predicted = replay.simulate(params)
        valid = trace.position_valid
        error = predicted[valid] - trace.measured_position_rad[valid]
        residuals.append(error / math.sqrt(error.size))
        residuals.extend(
            _frequency_residuals(
                trace,
                predicted,
                weight=float(frequency_weight),
            )
        )
    return np.concatenate(residuals)


def evaluate_parameters(
    replays: Sequence[FixtureReplay], params: ServoDynamics
) -> dict[str, object]:
    captures: dict[str, object] = {}
    rmses: list[float] = []
    for replay in replays:
        trace = replay.trace
        predicted = replay.simulate(params)
        error_deg = np.degrees(predicted - trace.measured_position_rad)
        valid_error = error_deg[trace.position_valid]
        rmse = float(np.sqrt(np.mean(np.square(valid_error))))
        rmses.append(rmse)
        captures[trace.label] = {
            "center_deg": trace.center_deg,
            "samples": int(valid_error.size),
            "rmse_deg": rmse,
            "abs_p95_deg": float(np.percentile(np.abs(valid_error), 95.0)),
            "bias_deg": float(np.mean(valid_error)),
        }
    return {
        "mean_capture_rmse_deg": float(np.mean(rmses)),
        "max_capture_rmse_deg": float(np.max(rmses)),
        "captures": captures,
    }


def fit_parameters(
    replays: Sequence[FixtureReplay],
    baseline: ServoDynamics,
    *,
    delay_steps: Iterable[int],
    frequency_weight: float,
    max_nfev: int,
) -> tuple[ServoDynamics, dict[str, object]]:
    # Damping and armature produce strongly correlated phase changes in this
    # sub-0.5 Nm fixture. Keep the existing armature and fit the smallest
    # identifiable parameter set. ToddlerBot likewise holds known controller
    # terms fixed while fitting the remaining joint dynamics.
    lower = np.log(np.asarray([5.0, 1e-4, 1e-4]))
    upper = np.log(np.asarray([60.0, 2.0, 0.5]))
    initial = np.clip(_parameter_vector(baseline), lower, upper)

    candidates: list[dict[str, object]] = []
    best_result = None
    best_params = None
    for delay in delay_steps:
        print(f"  fitting delay_steps={int(delay)}...", flush=True)
        result = least_squares(
            lambda values: build_residual_vector(
                replays,
                _unpack_parameters(
                    values, fixed=baseline, delay_steps=int(delay)
                ),
                frequency_weight=frequency_weight,
            ),
            initial,
            bounds=(lower, upper),
            x_scale="jac",
            diff_step=1e-3,
            max_nfev=int(max_nfev),
        )
        params = _unpack_parameters(
            result.x, fixed=baseline, delay_steps=int(delay)
        )
        metrics = evaluate_parameters(replays, params)
        candidate = {
            "delay_steps": int(delay),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
            "optimality": float(result.optimality),
            "parameters": asdict(params),
            "metrics": metrics,
        }
        candidates.append(candidate)
        print(
            f"    rmse={metrics['mean_capture_rmse_deg']:.4f}deg "
            f"cost={float(result.cost):.8g} nfev={int(result.nfev)}",
            flush=True,
        )
        if best_result is None or float(result.cost) < float(best_result.cost):
            best_result = result
            best_params = params

    assert best_result is not None and best_params is not None
    return best_params, {
        "candidates": candidates,
        "selected_cost": float(best_result.cost),
        "selected_nfev": int(best_result.nfev),
        "selected_optimality": float(best_result.optimality),
    }


def _parse_delay_steps(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("delay steps must be non-negative")
    return values


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit WildRobot HTD-45H low-load dynamics from capture NPZ files."
    )
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--fixture-mjcf", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--fixture-joint", default="pitch")
    parser.add_argument(
        "--joint-properties", type=Path, default=DEFAULT_JOINT_PROPERTIES
    )
    parser.add_argument(
        "--generated-mjcf", type=Path, default=DEFAULT_GENERATED_MJCF
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--force-limit-nm", type=float, default=4.4129925)
    parser.add_argument("--delay-steps", type=_parse_delay_steps, default=(0, 1, 2, 3))
    parser.add_argument("--frequency-weight", type=float, default=0.01)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the selected fit to joints_properties.xml and wildrobot.xml.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_path = args.fixture_mjcf.expanduser().resolve()
    joint_properties = args.joint_properties.expanduser().resolve()
    generated_mjcf = args.generated_mjcf.expanduser().resolve()
    if not fixture_path.is_file():
        raise FileNotFoundError(f"fixture MJCF not found: {fixture_path}")

    traces = [load_capture(path) for path in args.captures]
    fixture_hash = _sha256(fixture_path)
    for trace in traces:
        if trace.fixture_sha256 != fixture_hash:
            raise ValueError(
                f"capture {trace.path} fixture hash {trace.fixture_sha256} does "
                f"not match {fixture_path} ({fixture_hash})"
            )
    centers = sorted(round(trace.center_deg, 6) for trace in traces)
    if not (any(abs(value) < 1e-6 for value in centers) and centers[0] < 0 < centers[-1]):
        raise ValueError(
            "fit requires zero, positive-load, and negative-load captures; "
            f"got centers {centers}"
        )

    baseline = load_xml_dynamics(joint_properties)
    replays = [
        FixtureReplay(
            fixture_path,
            joint_name=str(args.fixture_joint),
            trace=trace,
            sim_dt=float(args.sim_dt),
            force_limit_nm=float(args.force_limit_nm),
        )
        for trace in traces
    ]
    baseline_by_delay = []
    for delay in args.delay_steps:
        params = ServoDynamics(**{**asdict(baseline), "delay_steps": int(delay)})
        baseline_by_delay.append(
            {
                "delay_steps": int(delay),
                "parameters": asdict(params),
                "metrics": evaluate_parameters(replays, params),
            }
        )

    print("WildRobot HTD-45H offline SysID fit")
    print(f"  fixture: {fixture_path}")
    print(f"  captures: {len(traces)} centers={centers}")
    print(
        "  baseline: "
        f"kp={baseline.kp:.6g} kv={baseline.kv:.6g} "
        f"damping={baseline.damping:.6g} armature={baseline.armature:.6g} "
        f"frictionloss={baseline.frictionloss:.6g}"
    )

    fitted, optimization = fit_parameters(
        replays,
        baseline,
        delay_steps=args.delay_steps,
        frequency_weight=float(args.frequency_weight),
        max_nfev=int(args.max_nfev),
    )
    baseline_selected = next(
        item
        for item in baseline_by_delay
        if item["delay_steps"] == fitted.delay_steps
    )
    fitted_metrics = evaluate_parameters(replays, fitted)
    baseline_rmse = float(baseline_selected["metrics"]["mean_capture_rmse_deg"])
    fitted_rmse = float(fitted_metrics["mean_capture_rmse_deg"])
    improvement = 1.0 - fitted_rmse / baseline_rmse

    print(
        "  fitted:   "
        f"kp={fitted.kp:.6g} kv={fitted.kv:.6g} "
        f"damping={fitted.damping:.6g} armature={fitted.armature:.6g} "
        f"frictionloss={fitted.frictionloss:.6g} "
        f"delay_steps={fitted.delay_steps}"
    )
    print(
        f"  mean capture RMSE: {baseline_rmse:.4f}deg -> "
        f"{fitted_rmse:.4f}deg ({improvement * 100.0:+.1f}%)"
    )
    for label, metrics in fitted_metrics["captures"].items():
        print(
            f"  {label}: rmse={metrics['rmse_deg']:.4f}deg "
            f"p95={metrics['abs_p95_deg']:.4f}deg "
            f"bias={metrics['bias_deg']:+.4f}deg"
        )

    report_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else DEFAULT_REPORT_DIR
        / f"htd45h_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "toddlerbot_style_time_and_frequency_replay",
        "reference": {
            "paper": "https://arxiv.org/abs/2502.00893",
            "local_source": "~/projects/toddlerbot/toddlerbot/tools/run_sysID.py",
        },
        "scope": {
            "identified": [
                "effective_position_kp",
                "joint_damping_with_fixed_actuator_kv",
                "frictionloss",
                "discrete_command_delay_steps",
            ],
            "not_identified": [
                "armature_independent_of_damping",
                "stall_torque",
                "continuous_torque",
                "velocity_dependent_torque_limit",
                "thermal_shutdown_boundary",
            ],
            "a1_frequency_limit_hz": 7.0,
            "fixed_parameters": {
                "kv": baseline.kv,
                "armature": baseline.armature,
            },
            "reason": (
                "The unchanged fixture excites less than 0.5 Nm, so torque "
                "limits remain vendor/conservative values."
            ),
        },
        "fixture": {
            "path": str(fixture_path),
            "sha256": fixture_hash,
            "joint": str(args.fixture_joint),
        },
        "captures": [
            {
                "path": str(trace.path),
                "manifest": str(trace.manifest_path),
                "center_deg": trace.center_deg,
                "samples": int(trace.command_rad.size),
            }
            for trace in traces
        ],
        "baseline_by_delay": baseline_by_delay,
        "selected_baseline": baseline_selected,
        "fitted_parameters": asdict(fitted),
        "fitted_metrics": fitted_metrics,
        "relative_rmse_improvement": improvement,
        "optimization": optimization,
        "applied": bool(args.apply),
        "applied_paths": (
            [str(joint_properties), str(generated_mjcf)] if args.apply else []
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"  report: {report_path}")

    if args.apply:
        if improvement < 0.05:
            raise RuntimeError(
                "refusing to apply a fit with less than 5% mean RMSE improvement"
            )
        update_xml_dynamics(joint_properties, fitted)
        update_xml_dynamics(generated_mjcf, fitted)
        print(f"  updated: {joint_properties}")
        print(f"  updated: {generated_mjcf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
