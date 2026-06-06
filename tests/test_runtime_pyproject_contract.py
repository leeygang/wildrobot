"""Pin the runtime package metadata fixed in v0.21.0.

  - ``rpi-gpio`` must carry a ``sys_platform == 'linux'`` marker so the runtime
    package installs/validates on macOS dev machines.
  - ``wildrobot-run-policy`` console script must exist now that the v3-aligned
    runner has landed.
"""

from __future__ import annotations

from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "runtime" / "pyproject.toml"


def _load_pyproject() -> dict:
    try:
        import tomllib  # py311+

        return tomllib.loads(_PYPROJECT.read_text())
    except ModuleNotFoundError:  # pragma: no cover
        import tomli

        return tomli.loads(_PYPROJECT.read_text())


def test_rpi_gpio_has_linux_marker() -> None:
    data = _load_pyproject()
    deps = data["project"]["dependencies"]
    rpi = [d for d in deps if d.replace(" ", "").lower().startswith("rpi-gpio")]
    assert rpi, f"rpi-gpio dependency missing from runtime deps: {deps}"
    assert all("sys_platform" in d and "linux" in d for d in rpi), (
        f"rpi-gpio must be gated with sys_platform=='linux'; got {rpi}"
    )


def test_run_policy_entrypoint_exists() -> None:
    data = _load_pyproject()
    scripts = data["project"]["scripts"]
    assert "wildrobot-run-policy" in scripts, (
        f"wildrobot-run-policy console script missing; got {list(scripts)}"
    )
    assert scripts["wildrobot-run-policy"] == "wr_runtime.control.run_policy:main"
