from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRANSFER_SCRIPT = _REPO_ROOT / "scripts" / "scp_from_remote.sh"


def test_latest_run_copies_nested_checkpoint_directory(tmp_path: Path) -> None:
    temp_repo = tmp_path / "repo"
    temp_scripts = temp_repo / "scripts"
    temp_scripts.mkdir(parents=True)
    transfer_script = temp_scripts / "scp_from_remote.sh"
    shutil.copy2(_TRANSFER_SCRIPT, transfer_script)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rsync_record = tmp_path / "rsync_args.txt"
    run_name = "offline-run-20260823_200326-ok607jgs"
    checkpoint_path = (
        "walking_v0210_onshape9_finetune/"
        "ppo_walking_v0210_smoke6_17d_latency_finetune_v0210-17d1_"
        "20260823_200339-ok607jgs"
    )

    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'if printf "%s\\n" "$*" | grep -q "find .*training/checkpoints"; then\n'
        f'    printf "%s\\n" "{checkpoint_path}"\n'
        'elif printf "%s\\n" "$*" | grep -q "ls -1t .*training/wandb"; then\n'
        f'    printf "%s\\n" "{run_name}"\n'
        "fi\n"
    )
    ssh.chmod(0o755)

    rsync = fake_bin / "rsync"
    rsync.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$RSYNC_RECORD"\n'
    )
    rsync.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RSYNC_RECORD": str(rsync_record),
        }
    )
    result = subprocess.run(
        ["bash", str(transfer_script), "--latest"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    rsync_calls = rsync_record.read_text().splitlines()
    assert len(rsync_calls) == 2
    assert checkpoint_path in rsync_calls[1]
    assert rsync_calls[1].endswith(f"training/checkpoints/{checkpoint_path}/")
    assert f"Checkpoint: {checkpoint_path}" in result.stdout


def _assert_connection_reuse(ssh_args: str, scp_args: str) -> None:
    for option in ("ControlMaster=auto", "ControlPersist=60"):
        assert option in ssh_args
        assert option in scp_args
    ssh_control_path = next(
        arg for arg in ssh_args.split() if arg.startswith("ControlPath=")
    )
    scp_control_path = next(
        arg for arg in scp_args.split() if arg.startswith("ControlPath=")
    )
    assert ssh_control_path == scp_control_path
    assert len(ssh_control_path.removeprefix("ControlPath=")) < 104


def test_latest_run_policy_log_copies_from_wrdev(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh_record = tmp_path / "ssh_args.txt"
    scp_record = tmp_path / "scp_args.txt"
    local_logs = tmp_path / "logs"

    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" > "$SSH_RECORD"\n'
        'printf "%s\\n" "/home/leeygang/projects/wildrobot/_run_policy_logs/'
        'v0227_ckpt200_stable_20260802_143000_123456.log"\n'
    )
    ssh.chmod(0o755)

    scp = fake_bin / "scp"
    scp.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" > "$SCP_RECORD"\n'
    )
    scp.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SSH_RECORD": str(ssh_record),
            "SCP_RECORD": str(scp_record),
            "WILDROBOT_RUN_POLICY_LOG_DIR": str(local_logs),
        }
    )
    result = subprocess.run(
        ["bash", str(_TRANSFER_SCRIPT), "--latest-policy-log"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Remote host:" in result.stdout
    assert "wrdev.local" in result.stdout
    assert local_logs.is_dir()
    ssh_args = ssh_record.read_text()
    assert "leeygang@wrdev.local" in ssh_args
    scp_args = scp_record.read_text()
    _assert_connection_reuse(ssh_args, scp_args)
    assert "leeygang@wrdev.local:/home/leeygang/projects/wildrobot/" in scp_args
    assert "_run_policy_logs/v0227_ckpt200_stable_" in scp_args
    assert scp_args.rstrip().endswith(f"{local_logs}/")


def test_named_run_policy_log_copies_from_wrdev(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ssh_record = tmp_path / "ssh_args.txt"
    scp_record = tmp_path / "scp_args.txt"
    local_logs = tmp_path / "logs"
    filename = "standing_v0227_robot_5s.log"

    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" > "$SSH_RECORD"\n'
        f'printf "%s\\n" "{filename}"\n'
    )
    ssh.chmod(0o755)

    scp = fake_bin / "scp"
    scp.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" > "$SCP_RECORD"\n'
    )
    scp.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SSH_RECORD": str(ssh_record),
            "SCP_RECORD": str(scp_record),
            "WILDROBOT_RUN_POLICY_LOG_DIR": str(local_logs),
        }
    )
    result = subprocess.run(
        ["bash", str(_TRANSFER_SCRIPT), "--policy-log", filename],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "wrdev.local" in result.stdout
    assert local_logs.is_dir()
    ssh_args = ssh_record.read_text()
    scp_args = scp_record.read_text()
    _assert_connection_reuse(ssh_args, scp_args)
    assert scp_args.strip().endswith(
        "leeygang@wrdev.local:/home/leeygang/projects/wildrobot/"
        f"_run_policy_logs/{filename} {local_logs}/{filename}"
    )


@pytest.mark.parametrize(
    ("pattern", "expected_names"),
    [
        (
            r"^v0227_ckpt200_home_trial0[12]_.*\.log$",
            [
                "v0227_ckpt200_home_trial01_20260802.log",
                "v0227_ckpt200_home_trial02_20260802.log",
            ],
        ),
        (
            "*.log",
            [
                "v0227_ckpt200_home_trial01_20260802.log",
                "v0227_ckpt200_home_trial02_20260802.log",
                "v0227_ckpt250_home_trial01_20260802.log",
            ],
        ),
    ],
)
def test_run_policy_log_pattern_copies_all_matches(
    tmp_path: Path,
    pattern: str,
    expected_names: list[str],
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scp_record = tmp_path / "scp_args.txt"
    local_logs = tmp_path / "logs"
    filenames = [
        "v0227_ckpt200_home_trial01_20260802.log",
        "v0227_ckpt200_home_trial02_20260802.log",
        "v0227_ckpt250_home_trial01_20260802.log",
        "v0227_ckpt200_home_trial03_20260802.txt",
    ]

    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        + "".join(f'printf "%s\\n" "{filename}"\n' for filename in filenames)
    )
    ssh.chmod(0o755)

    scp = fake_bin / "scp"
    scp.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$SCP_RECORD"\n'
    )
    scp.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SCP_RECORD": str(scp_record),
            "WILDROBOT_RUN_POLICY_LOG_DIR": str(local_logs),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(_TRANSFER_SCRIPT),
            "--policy-log",
            pattern,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert f"Copying {len(expected_names)} wildrobot-run-policy log(s)" in result.stdout
    scp_calls = scp_record.read_text().splitlines()
    assert len(scp_calls) == len(expected_names)
    for filename, call in zip(expected_names, scp_calls, strict=True):
        assert "ControlMaster=auto" in call
        assert call.endswith(
            "leeygang@wrdev.local:/home/leeygang/projects/wildrobot/"
            f"_run_policy_logs/{filename} {local_logs}/{filename}"
        )


def test_named_run_policy_log_rejects_paths(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(_TRANSFER_SCRIPT),
            "--policy-log",
            "../outside.log",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "requires a filename, glob, or regex without directory components" in result.stdout


def test_unknown_option_is_not_treated_as_remote_path(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(_TRANSFER_SCRIPT), "--run_policy_logs", "example.log"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "unknown option '--run_policy_logs'" in result.stdout
