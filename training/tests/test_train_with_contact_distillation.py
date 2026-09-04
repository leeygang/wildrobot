from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from training.scripts import train_with_contact_distillation as wrapper
from training.scripts.train_with_contact_distillation import (
    _build_distillation_command,
    _completed_distillation,
)


def test_distillation_command_comes_from_frozen_training_config(
    tmp_path: Path,
) -> None:
    config = tmp_path / "training.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "bootstrap": {
                    "mode": "contact_observed_to_proprio",
                    "rollout_repeats": 8,
                    "require_no_terminations": True,
                }
            }
        )
    )
    output = tmp_path / "distilled.pkl"
    report = tmp_path / "metrics.json"

    command = _build_distillation_command(
        config_path=config,
        output=output,
        report=report,
    )

    assert command[command.index("--student-config") + 1] == str(config)
    assert command[command.index("--rollout-repeats") + 1] == "8"
    assert "--no-require-no-terminations" not in command


def test_completed_distillation_requires_checkpoint_and_passing_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "distilled.pkl"
    report = tmp_path / "metrics.json"
    output.write_bytes(b"checkpoint")
    report.write_text(json.dumps({"gates": {"passed": False}}))
    assert _completed_distillation(output, report) is False

    report.write_text(json.dumps({"gates": {"passed": True}}))
    assert _completed_distillation(output, report) is True


def test_wrapper_runs_distillation_then_training(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "training.yaml"
    config.write_text(
        yaml.safe_dump(
            {"bootstrap": {"mode": "contact_observed_to_proprio"}}
        )
    )
    checkpoint_dir = tmp_path / "checkpoints"
    bootstrap_dir = tmp_path / "bootstrap"
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if "distill_contact_observed_to_proprio.py" in command[1]:
            bootstrap_dir.mkdir(parents=True, exist_ok=True)
            (bootstrap_dir / "contact_free_distilled.pkl").write_bytes(b"policy")
            (bootstrap_dir / "contact_free_distilled.metrics.json").write_text(
                json.dumps({"gates": {"passed": True}})
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_with_contact_distillation.py",
            "--config",
            str(config),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--bootstrap-dir",
            str(bootstrap_dir),
        ],
    )

    assert wrapper.main() == 0
    assert "distill_contact_observed_to_proprio.py" in commands[0][1]
    assert "training/train.py" in commands[1][1]
    assert commands[1][-2:] == [
        "--init-policy",
        str(bootstrap_dir / "contact_free_distilled.pkl"),
    ]
