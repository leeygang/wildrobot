#!/usr/bin/env python3
"""Run contact-free policy distillation, then PPO, as one GPU queue job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


BOOTSTRAP_MODE = "contact_observed_to_proprio"


def _bootstrap_config(config_path: Path) -> dict:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bootstrap = payload.get("bootstrap") if isinstance(payload, dict) else None
    if not isinstance(bootstrap, dict) or bootstrap.get("mode") != BOOTSTRAP_MODE:
        raise ValueError(
            f"config must declare bootstrap.mode={BOOTSTRAP_MODE}: {config_path}"
        )
    return bootstrap


def _append_option(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _build_distillation_command(
    *, config_path: Path, output: Path, report: Path
) -> list[str]:
    bootstrap = _bootstrap_config(config_path)
    script = Path(__file__).with_name("distill_contact_observed_to_proprio.py")
    command = [
        sys.executable,
        str(script),
        "--student-config",
        str(config_path),
        "--output",
        str(output),
        "--report",
        str(report),
    ]
    _append_option(command, "--teacher-checkpoint", bootstrap.get("teacher_checkpoint"))
    _append_option(command, "--student-checkpoint", bootstrap.get("student_checkpoint"))
    _append_option(
        command,
        "--student-checkpoint-sha256",
        bootstrap.get("student_checkpoint_sha256"),
    )
    _append_option(command, "--failure-trace", bootstrap.get("failure_trace"))
    _append_option(
        command,
        "--failure-trace-sha256",
        bootstrap.get("failure_trace_sha256"),
    )
    _append_option(command, "--commands", bootstrap.get("commands"))
    for key in (
        "rollout_steps",
        "rollout_repeats",
        "validation_steps",
        "validation_repeats",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_validation_rmse",
        "failure_replay_repeats",
        "seed",
    ):
        _append_option(command, f"--{key.replace('_', '-')}", bootstrap.get(key))
    if bootstrap.get("require_no_terminations", True) is False:
        command.append("--no-require-no-terminations")
    return command


def _completed_distillation(output: Path, report: Path) -> bool:
    if not output.is_file() or not report.is_file():
        return False
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    gates = payload.get("gates") if isinstance(payload, dict) else None
    return isinstance(gates, dict) and gates.get("passed") is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    args = parser.parse_args()

    args.bootstrap_dir.mkdir(parents=True, exist_ok=True)
    output = args.bootstrap_dir / "contact_free_distilled.pkl"
    report = args.bootstrap_dir / "contact_free_distilled.metrics.json"
    if _completed_distillation(output, report):
        print(f"Reusing completed distillation checkpoint: {output}", flush=True)
    else:
        command = _build_distillation_command(
            config_path=args.config,
            output=output,
            report=report,
        )
        print("Starting contact-free teacher distillation...", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            return int(result.returncode)
        if not _completed_distillation(output, report):
            raise RuntimeError("distillation exited successfully without passing gates")

    train_script = Path(__file__).resolve().parents[1] / "train.py"
    train_command = [
        sys.executable,
        str(train_script),
        "--config",
        str(args.config),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--init-policy",
        str(output),
    ]
    print("Distillation passed; starting PPO fine-tune...", flush=True)
    return int(subprocess.run(train_command, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
