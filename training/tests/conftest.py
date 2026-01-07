# training/tests/conftest.py
"""
Shared test fixtures for WildRobot tests.

This module provides pytest fixtures that are shared across all test modules.
Following the TEST_STRATEGY.md, it establishes:
- Robot schema as single source of truth
- MuJoCo model loading
- Training config loading
- Environment setup
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
import pytest

# Project paths
ASSETS_PATH = PROJECT_ROOT / "assets"
SCENE_XML_PATH = ASSETS_PATH / "scene_flat_terrain.xml"
ROBOT_XML_PATH = ASSETS_PATH / "wildrobot.xml"
ROBOT_CONFIG_PATH = ASSETS_PATH / "robot_config.yaml"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "ppo_walking.yaml"
SCHEMA_PATH = ASSETS_PATH / "robot_schema.json"


# =============================================================================
# Layer 0: Schema Contract Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def robot_schema():
    """Load robot schema once per session.

    This is the single source of truth for robot configuration.
    """
    from training.tests.robot_schema import WildRobotSchema

    schema = WildRobotSchema.from_xml(SCENE_XML_PATH)
    schema.validate()
    return schema


@pytest.fixture(scope="session")
def robot_schema_dict(robot_schema) -> Dict[str, Any]:
    """Return robot schema as dictionary."""
    return robot_schema.to_dict()


# =============================================================================
# Layer 1: Config Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def training_config():
    """Load training config once per session."""
    from training.configs.training_config import load_training_config
    return load_training_config(TRAINING_CONFIG_PATH)


@pytest.fixture(scope="session")
def robot_config():
    """Load robot config once per session."""
    from training.configs.training_config import load_robot_config
    return load_robot_config(ROBOT_CONFIG_PATH)


# =============================================================================
# Layer 2: MuJoCo Model Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def mj_model():
    """Load MuJoCo model once per session."""
    return mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))


@pytest.fixture(scope="function")
def mj_data(mj_model):
    """Create fresh MuJoCo data for each test.

    Function-scoped to ensure test isolation.
    """
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture(scope="session")
def floor_geom_id(mj_model) -> int:
    """Get floor geom ID."""
    return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")


# =============================================================================
# Layer 3: Environment Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def rng():
    """JAX random key for reproducibility."""
    import jax
    return jax.random.PRNGKey(42)


# =============================================================================
# Helper Functions (available to all tests)
# =============================================================================


def get_foot_contact_forces(mj_model, mj_data, robot_schema, floor_geom_id: int):
    """Get contact forces for left and right feet.

    Returns:
        Tuple of (left_force, right_force) in Newtons
    """
    left_geom_ids = set(robot_schema.get_left_foot_geom_ids())
    right_geom_ids = set(robot_schema.get_right_foot_geom_ids())

    left_force = 0.0
    right_force = 0.0

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2

        # Check if floor is involved
        if geom1 != floor_geom_id and geom2 != floor_geom_id:
            continue

        other_geom = geom1 if geom2 == floor_geom_id else geom2

        # Get contact force
        force = np.zeros(6)
        mujoco.mj_contactForce(mj_model, mj_data, i, force)
        normal_force = abs(force[0])

        if other_geom in left_geom_ids:
            left_force += normal_force
        elif other_geom in right_geom_ids:
            right_force += normal_force

    return left_force, right_force


def get_total_mass(mj_model) -> float:
    """Get total mass of the robot."""
    return sum(mj_model.body(i).mass[0] for i in range(mj_model.nbody))
