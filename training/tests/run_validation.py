#!/usr/bin/env python3
"""Quick validation script for the schema and physics tests."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_schema():
    """Test schema extraction and validation."""
    print("=" * 60)
    print("Schema Contract Validation")
    print("=" * 60)

    from training.tests.robot_schema import WildRobotSchema

    # Extract schema from XML
    print("\n1. Extracting schema from XML...")
    schema = WildRobotSchema.from_xml("assets/scene_flat_terrain.xml")
    print(f"   Model: nq={schema.nq}, nv={schema.nv}, nu={schema.nu}")

    # Validate
    print("\n2. Validating schema...")
    schema.validate()
    print("   Schema validation passed!")

    # Check joints
    print(f"\n3. Joints (excluding _mimic): {len(schema.joints)}")
    for j in schema.joints:
        print(f"   - {j.joint_name}: qpos_adr={j.qpos_adr}, dof_adr={j.dof_adr}")

    # Check actuators
    print(f"\n4. Actuators: {len(schema.actuators)}")
    for a in schema.actuators:
        print(f"   - {a.actuator_name} -> {a.joint_name} (dof={a.dof_adr})")

    # Check foot geoms
    print(f"\n5. Left foot geoms: {len(schema.left_foot_geoms)}")
    for g in schema.left_foot_geoms:
        print(f"   - {g.geom_name} (body={g.body_name}, id={g.geom_id})")

    print(f"\n6. Right foot geoms: {len(schema.right_foot_geoms)}")
    for g in schema.right_foot_geoms:
        print(f"   - {g.geom_name} (body={g.body_name}, id={g.geom_id})")

    # Explicit mimic checks
    print("\n7. _mimic leakage prevention checks...")
    schema.assert_no_mimic_in_actuators()
    print("   No _mimic joints in actuators")
    schema.assert_no_mimic_in_foot_geoms()
    print("   No _mimic bodies in foot geoms")

    print("\n" + "=" * 60)
    print("All schema tests passed!")
    print("=" * 60)
    return True


def test_env_basics():
    """Test basic environment functionality."""
    print("\n" + "=" * 60)
    print("Environment Basic Validation")
    print("=" * 60)

    import jax
    import jax.numpy as jnp
    import mujoco

    from assets.robot_config import load_robot_config
    from training.configs.training_config import load_training_config
    from training.envs.wildrobot_env import WildRobotEnv

    # Load configs
    print("\n1. Loading configs...")
    load_robot_config("assets/robot_config.yaml")
    training_cfg = load_training_config("training/configs/ppo_walking.yaml")
    training_cfg.freeze()  # Freeze config for JIT compatibility

    # Create env
    print("\n2. Creating environment...")
    env = WildRobotEnv(config=training_cfg)

    # Reset
    print("\n3. Resetting environment...")
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)
    print(f"   Observation shape: {state.obs.shape}")
    print(f"   Action size: {env.action_size}")

    # Check foot body IDs are not _mimic
    print("\n4. Checking foot body IDs...")
    left_name = mujoco.mj_id2name(
        env._mj_model, mujoco.mjtObj.mjOBJ_BODY, env._left_foot_body_id
    )
    right_name = mujoco.mj_id2name(
        env._mj_model, mujoco.mjtObj.mjOBJ_BODY, env._right_foot_body_id
    )
    print(f"   Left foot body: {left_name} (id={env._left_foot_body_id})")
    print(f"   Right foot body: {right_name} (id={env._right_foot_body_id})")

    assert "_mimic" not in left_name.lower(), "Left foot is _mimic!"
    assert "_mimic" not in right_name.lower(), "Right foot is _mimic!"
    print("   No _mimic bodies in foot references!")

    # Step
    print("\n5. Taking environment step...")
    action = jnp.zeros(env.action_size)
    state2 = env.step(state, action)
    print(f"   Reward: {float(state2.reward):.4f}")
    print(f"   Done: {float(state2.done):.0f}")

    # Get contact forces
    print("\n6. Checking contact forces...")
    left_force, right_force = env._cal.get_aggregated_foot_contacts(state2.data)
    print(f"   Left foot force: {float(left_force):.2f} N")
    print(f"   Right foot force: {float(right_force):.2f} N")

    # Let robot settle
    print("\n7. Running 100 steps to settle...")
    for _ in range(100):
        state2 = env.step(state2, action)

    left_force, right_force = env._cal.get_aggregated_foot_contacts(state2.data)
    total_force = float(left_force + right_force)

    # Estimate expected weight
    robot_mass = 0
    for i in range(env._mj_model.nbody):
        robot_mass += env._mj_model.body_mass[i]
    expected_force = robot_mass * 9.81
    ratio = total_force / expected_force if expected_force > 0 else 0

    print(f"   Robot mass: {robot_mass:.2f} kg")
    print(f"   Expected force (mg): {expected_force:.2f} N")
    print(f"   Total measured force: {total_force:.2f} N")
    print(f"   Ratio: {ratio:.2f}")

    print("\n" + "=" * 60)
    print("All environment tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = True

    try:
        success &= test_schema()
    except Exception as e:
        print(f"\nSchema test failed: {e}")
        import traceback

        traceback.print_exc()
        success = False

    try:
        success &= test_env_basics()
    except Exception as e:
        print(f"\nEnvironment test failed: {e}")
        import traceback

        traceback.print_exc()
        success = False

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    sys.exit(0 if success else 1)
