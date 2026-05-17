from __future__ import annotations

from types import SimpleNamespace


def _make_cfg(num_envs: int, eval_num_envs: int = 64):
    return SimpleNamespace(
        ppo=SimpleNamespace(
            num_envs=num_envs,
            eval=SimpleNamespace(num_envs=eval_num_envs),
        )
    )


def _make_args(num_envs=None):
    return SimpleNamespace(num_envs=num_envs)


def test_desktop_gpu_guard_caps_single_geforce_linux(monkeypatch) -> None:
    from training import train

    cfg = _make_cfg(num_envs=2048, eval_num_envs=1536)
    args = _make_args(num_envs=None)

    monkeypatch.setattr(train.platform, "system", lambda: "Linux")
    monkeypatch.delenv("WR_DISABLE_DESKTOP_GPU_NUM_ENVS_GUARD", raising=False)
    monkeypatch.delenv("WR_DESKTOP_GPU_NUM_ENVS_CAP", raising=False)
    monkeypatch.setattr(
        train,
        "_query_visible_gpus",
        lambda: [("NVIDIA GeForce RTX 5070", "Disabled")],
    )

    changed = train.maybe_apply_desktop_gpu_num_env_guard(cfg, args)

    assert changed is True
    assert cfg.ppo.num_envs == 1024
    assert cfg.ppo.eval.num_envs == 1024


def test_desktop_gpu_guard_respects_cli_override(monkeypatch) -> None:
    from training import train

    cfg = _make_cfg(num_envs=2048, eval_num_envs=64)
    args = _make_args(num_envs=2048)

    monkeypatch.setattr(train.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        train,
        "_query_visible_gpus",
        lambda: [("NVIDIA GeForce RTX 5070", "Disabled")],
    )

    changed = train.maybe_apply_desktop_gpu_num_env_guard(cfg, args)

    assert changed is False
    assert cfg.ppo.num_envs == 2048
    assert cfg.ppo.eval.num_envs == 64


def test_desktop_gpu_guard_respects_disable_env(monkeypatch) -> None:
    from training import train

    cfg = _make_cfg(num_envs=2048, eval_num_envs=64)
    args = _make_args(num_envs=None)

    monkeypatch.setattr(train.platform, "system", lambda: "Linux")
    monkeypatch.setenv("WR_DISABLE_DESKTOP_GPU_NUM_ENVS_GUARD", "1")
    monkeypatch.setattr(
        train,
        "_query_visible_gpus",
        lambda: [("NVIDIA GeForce RTX 5070", "Disabled")],
    )

    changed = train.maybe_apply_desktop_gpu_num_env_guard(cfg, args)

    assert changed is False
    assert cfg.ppo.num_envs == 2048


def test_desktop_gpu_guard_skips_non_geforce_or_multi_gpu(monkeypatch) -> None:
    from training import train

    monkeypatch.setattr(train.platform, "system", lambda: "Linux")
    monkeypatch.delenv("WR_DISABLE_DESKTOP_GPU_NUM_ENVS_GUARD", raising=False)

    cfg_a = _make_cfg(num_envs=2048, eval_num_envs=64)
    changed_a = None
    monkeypatch.setattr(
        train,
        "_query_visible_gpus",
        lambda: [("NVIDIA RTX 6000 Ada", "Disabled")],
    )
    changed_a = train.maybe_apply_desktop_gpu_num_env_guard(cfg_a, _make_args())

    cfg_b = _make_cfg(num_envs=2048, eval_num_envs=64)
    monkeypatch.setattr(
        train,
        "_query_visible_gpus",
        lambda: [
            ("NVIDIA GeForce RTX 5070", "Disabled"),
            ("NVIDIA GeForce RTX 5070", "Disabled"),
        ],
    )
    changed_b = train.maybe_apply_desktop_gpu_num_env_guard(cfg_b, _make_args())

    assert changed_a is False
    assert cfg_a.ppo.num_envs == 2048
    assert changed_b is False
    assert cfg_b.ppo.num_envs == 2048
