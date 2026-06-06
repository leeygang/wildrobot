"""Control loop entrypoints.

The v0.19.x ``run_policy`` script and the ``loc_ref_runtime`` /
``loc_ref_runtime_v2`` / ``run_walking`` helpers were deleted at
v0.20.1 along with the v1/v2 walking references.

v0.21.0 lands the v3-aligned hardware policy runner for the latest
``wr_obs_v8_cmd3d`` home-base-residual contract:

  - ``runtime_policy_config``: loads the exported ``runtime_policy_config.json``
    metadata (residual base/scale/per-joint, action delay/filter, command
    axes, gait phase clock).
  - ``reference_phase``: bundled (bin-independent) gait phase clock service.
  - ``policy_runner``: the online state machine (build v8 obs -> ONNX ->
    home-base residual compose -> write ctrl), mirroring
    ``training/eval/v6_eval_adapter.py`` semantics without MuJoCo.
  - ``mock_robot_io``: hardware-free ``RobotIO`` for dry-run / CI smoke tests.
  - ``run_policy``: the ``wildrobot-run-policy`` CLI entrypoint.
"""
