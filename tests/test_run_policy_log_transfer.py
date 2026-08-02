from __future__ import annotations

import os
from pathlib import Path
import subprocess


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRANSFER_SCRIPT = _REPO_ROOT / "scripts" / "scp_from_remote.sh"


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
        ["bash", str(_TRANSFER_SCRIPT), "--latest_run_policy_log"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Remote host:" in result.stdout
    assert "wrdev.local" in result.stdout
    assert local_logs.is_dir()
    assert ssh_record.read_text().startswith("leeygang@wrdev.local ")
    scp_args = scp_record.read_text()
    assert "leeygang@wrdev.local:/home/leeygang/projects/wildrobot/" in scp_args
    assert "_run_policy_logs/v0227_ckpt200_stable_" in scp_args
    assert scp_args.rstrip().endswith(f"{local_logs}/")
