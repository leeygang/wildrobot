"""WildRobot Schema Contract - Single Source of Truth for Robot Configuration.

This module extracts and validates robot schema from the MuJoCo XML at runtime.
It guarantees that all code (env, reward, tests, MJX) is reading the correct
data by deriving indices from the XML rather than hardcoding them.

The schema contract approach:
1. Extract schema once from XML at startup
2. Snapshot to JSON for regression testing
3. Assert schema hasn't changed on load
4. Use schema everywhere to prevent silent breakage

Industry best practice: Never hardcode qpos/qvel indices, geom IDs, etc.

Usage:
    from training.tests.robot_schema import WildRobotSchema

    schema = WildRobotSchema.from_xml("assets/v1/scene_flat_terrain.xml")
    schema.validate()
    schema.save("assets/robot_schema.json")

    # Or load and verify against saved schema
    schema = WildRobotSchema.from_xml("assets/v1/scene_flat_terrain.xml")
    schema.assert_matches_saved("assets/robot_schema.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np


@dataclass
class JointSchema:
    """Schema for a single joint (excluding _mimic)."""

    joint_name: str
    joint_type: str  # "hinge", "free", etc.
    qpos_adr: int
    dof_adr: int
    range_lower: float
    range_upper: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_name": self.joint_name,
            "joint_type": self.joint_type,
            "qpos_adr": self.qpos_adr,
            "dof_adr": self.dof_adr,
            "range": [self.range_lower, self.range_upper],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JointSchema":
        return cls(
            joint_name=d["joint_name"],
            joint_type=d["joint_type"],
            qpos_adr=d["qpos_adr"],
            dof_adr=d["dof_adr"],
            range_lower=d["range"][0],
            range_upper=d["range"][1],
        )


@dataclass
class ActuatorSchema:
    """Schema for a single actuator."""

    actuator_name: str
    joint_name: str
    dof_adr: int
    gear: float
    ctrl_range_lower: float
    ctrl_range_upper: float
    force_range_lower: float
    force_range_upper: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actuator_name": self.actuator_name,
            "joint_name": self.joint_name,
            "dof_adr": self.dof_adr,
            "gear": self.gear,
            "ctrl_range": [self.ctrl_range_lower, self.ctrl_range_upper],
            "force_range": [self.force_range_lower, self.force_range_upper],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActuatorSchema":
        return cls(
            actuator_name=d["actuator_name"],
            joint_name=d["joint_name"],
            dof_adr=d["dof_adr"],
            gear=d["gear"],
            ctrl_range_lower=d["ctrl_range"][0],
            ctrl_range_upper=d["ctrl_range"][1],
            force_range_lower=d["force_range"][0],
            force_range_upper=d["force_range"][1],
        )


@dataclass
class BaseSchema:
    """Schema for the floating base."""

    base_body_name: str
    base_free_joint_name: str
    base_qpos_adr: int  # Start of 7D (pos + quat)
    base_dof_adr: int  # Start of 6D (linvel + angvel)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseSchema":
        return cls(**d)


@dataclass
class GeomSchema:
    """Schema for a contact geom."""

    geom_name: str
    geom_id: int
    body_name: str
    body_id: int
    is_collision: bool

    def to_dict(self) -> Dict[str, Any]:
        # Explicitly convert numpy types to Python native types for JSON serialization
        return {
            "geom_name": self.geom_name,
            "geom_id": int(self.geom_id),
            "body_name": self.body_name,
            "body_id": int(self.body_id),
            "is_collision": bool(self.is_collision),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GeomSchema":
        return cls(**d)


@dataclass
class SensorSchema:
    """Schema for a sensor."""

    sensor_name: str
    sensor_type: str  # "gyro", "accelerometer", etc.
    dimension: int
    adr: int  # Address in sensordata array
    site_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        # Explicitly convert numpy types to Python native types for JSON serialization
        return {
            "sensor_name": self.sensor_name,
            "sensor_type": self.sensor_type,
            "dimension": int(self.dimension),
            "adr": int(self.adr),
            "site_name": self.site_name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SensorSchema":
        return cls(**d)


@dataclass
class WildRobotSchema:
    """Complete schema contract for WildRobot.

    This is the single source of truth for:
    - Joint indices (real joints only, excluding _mimic)
    - Actuator mappings
    - Floating base definition
    - Foot contact geoms
    - Sensor mappings
    """

    # A. Joint schema (real joints only)
    joints: List[JointSchema] = field(default_factory=list)

    # B. Actuator schema
    actuators: List[ActuatorSchema] = field(default_factory=list)

    # C. Base definition
    base: Optional[BaseSchema] = None

    # D. Foot contact geoms
    left_foot_geoms: List[GeomSchema] = field(default_factory=list)
    right_foot_geoms: List[GeomSchema] = field(default_factory=list)

    # E. Sensors
    sensors: List[SensorSchema] = field(default_factory=list)

    # Metadata
    model_name: str = ""
    nq: int = 0  # qpos dimension
    nv: int = 0  # qvel dimension
    nu: int = 0  # actuator count

    # Quick lookup caches (not serialized)
    _joint_name_to_schema: Dict[str, JointSchema] = field(
        default_factory=dict, repr=False
    )
    _actuator_name_to_schema: Dict[str, ActuatorSchema] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def from_xml(cls, xml_path: str | Path) -> "WildRobotSchema":
        """Extract schema from MuJoCo XML file.

        Args:
            xml_path: Path to the MuJoCo XML file (e.g., "assets/v1/scene_flat_terrain.xml")

        Returns:
            WildRobotSchema with all components extracted
        """
        xml_path = Path(xml_path)
        if not xml_path.is_absolute():
            # Resolve relative to project root
            project_root = Path(__file__).parent.parent.parent
            xml_path = project_root / xml_path

        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        # Load assets
        assets = {}
        root_path = xml_path.parent
        meshes_path = root_path / "assets"

        if meshes_path.exists():
            for stl_file in meshes_path.glob("*.stl"):
                assets[stl_file.name] = stl_file.read_bytes()

        # Also load any included XML files
        for xml_file in root_path.glob("*.xml"):
            if xml_file.name != xml_path.name:
                assets[xml_file.name] = xml_file.read_bytes()

        # Load model
        model = mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=assets)

        return cls.from_model(model)

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "WildRobotSchema":
        """Extract schema from loaded MuJoCo model.

        Args:
            model: Loaded MjModel

        Returns:
            WildRobotSchema with all components extracted
        """
        schema = cls()
        schema.model_name = model.model if hasattr(model, "model") else ""
        schema.nq = model.nq
        schema.nv = model.nv
        schema.nu = model.nu

        # A. Extract joint schema (excluding _mimic)
        schema._extract_joints(model)

        # B. Extract actuator schema
        schema._extract_actuators(model)

        # C. Extract base definition
        schema._extract_base(model)

        # D. Extract foot geoms
        schema._extract_foot_geoms(model)

        # E. Extract sensors
        schema._extract_sensors(model)

        # Build lookup caches
        schema._build_caches()

        return schema

    def _extract_joints(self, model: mujoco.MjModel) -> None:
        """Extract real joints, excluding _mimic bodies."""
        joint_type_names = ["free", "ball", "slide", "hinge"]

        for j in range(model.njnt):
            name = model.joint(j).name

            # Skip _mimic joints
            if "_mimic" in name.lower():
                continue

            # Skip the free joint (handled separately in base)
            if model.jnt_type[j] == 0:  # free joint
                continue

            jnt_type = model.jnt_type[j]
            type_name = (
                joint_type_names[jnt_type]
                if jnt_type < len(joint_type_names)
                else "unknown"
            )

            jnt_range = model.jnt_range[j]

            self.joints.append(
                JointSchema(
                    joint_name=name,
                    joint_type=type_name,
                    qpos_adr=int(model.jnt_qposadr[j]),
                    dof_adr=int(model.jnt_dofadr[j]),
                    range_lower=float(jnt_range[0]),
                    range_upper=float(jnt_range[1]),
                )
            )

    def _extract_actuators(self, model: mujoco.MjModel) -> None:
        """Extract actuator schema and verify no actuator targets _mimic joint."""
        for a in range(model.nu):
            act_name = model.actuator(a).name

            # Get the joint this actuator controls
            # trnid[a, 0] is the joint index for joint-based actuators
            jnt_id = model.actuator_trnid[a, 0]
            jnt_name = model.joint(jnt_id).name if jnt_id >= 0 else "N/A"

            # CRITICAL: Assert no actuator targets a _mimic joint
            if "_mimic" in jnt_name.lower():
                raise ValueError(
                    f"Actuator '{act_name}' targets _mimic joint '{jnt_name}'! "
                    "This is a bug - _mimic joints should not have actuators."
                )

            dof_adr = int(model.jnt_dofadr[jnt_id]) if jnt_id >= 0 else -1

            # Get gear (typically shape (6,) but we take first element)
            gear = float(model.actuator_gear[a, 0])

            # Get control range
            ctrl_range = model.actuator_ctrlrange[a]

            # Get force range
            force_range = model.actuator_forcerange[a]

            self.actuators.append(
                ActuatorSchema(
                    actuator_name=act_name,
                    joint_name=jnt_name,
                    dof_adr=dof_adr,
                    gear=gear,
                    ctrl_range_lower=float(ctrl_range[0]),
                    ctrl_range_upper=float(ctrl_range[1]),
                    force_range_lower=float(force_range[0]),
                    force_range_upper=float(force_range[1]),
                )
            )

    def _extract_base(self, model: mujoco.MjModel) -> None:
        """Extract floating base definition."""
        # Find the free joint
        for j in range(model.njnt):
            if model.jnt_type[j] == 0:  # free joint type
                jnt_name = model.joint(j).name

                # Get body that owns this joint
                body_id = model.jnt_bodyid[j]
                body_name = model.body(body_id).name

                self.base = BaseSchema(
                    base_body_name=body_name,
                    base_free_joint_name=jnt_name,
                    base_qpos_adr=int(model.jnt_qposadr[j]),
                    base_dof_adr=int(model.jnt_dofadr[j]),
                )
                break

    def _extract_foot_geoms(self, model: mujoco.MjModel) -> None:
        """Extract foot contact geoms (collision-enabled, non-_mimic bodies)."""
        # Known foot geom names from robot_config.yaml
        left_foot_names = ["left_toe", "left_heel"]
        right_foot_names = ["right_toe", "right_heel"]

        for g in range(model.ngeom):
            geom_name = model.geom(g).name
            if not geom_name:
                continue

            # Get body info
            body_id = model.geom_bodyid[g]
            body_name = model.body(body_id).name

            # Skip if body is _mimic
            if "_mimic" in body_name.lower():
                continue

            # Check if collision enabled (contype or conaffinity > 0)
            contype = model.geom_contype[g]
            conaffinity = model.geom_conaffinity[g]
            is_collision = contype > 0 or conaffinity > 0

            geom_schema = GeomSchema(
                geom_name=geom_name,
                geom_id=g,
                body_name=body_name,
                body_id=body_id,
                is_collision=is_collision,
            )

            if geom_name in left_foot_names:
                self.left_foot_geoms.append(geom_schema)
            elif geom_name in right_foot_names:
                self.right_foot_geoms.append(geom_schema)

    def _extract_sensors(self, model: mujoco.MjModel) -> None:
        """Extract sensor schema."""
        sensor_type_names = [
            "touch",
            "accelerometer",
            "velocimeter",
            "gyro",
            "force",
            "torque",
            "magnetometer",
            "rangefinder",
            "jointpos",
            "jointvel",
            "tendonpos",
            "tendonvel",
            "actuatorpos",
            "actuatorvel",
            "actuatorfrc",
            "jointlimitpos",
            "jointlimitvel",
            "jointlimitfrc",
            "tendonlimitpos",
            "tendonlimitvel",
            "tendonlimitfrc",
            "framepos",
            "framequat",
            "framexaxis",
            "frameyaxis",
            "framezaxis",
            "framelinvel",
            "frameangvel",
            "framelinacc",
            "frameangacc",
            "subtreecom",
            "subtreelinvel",
            "subtreeangmom",
            "clock",
            "user",
        ]

        for s in range(model.nsensor):
            sensor_name = model.sensor(s).name
            sensor_type = model.sensor_type[s]
            type_name = (
                sensor_type_names[sensor_type]
                if sensor_type < len(sensor_type_names)
                else "unknown"
            )

            # Sensor dimension and address
            dim = int(model.sensor_dim[s])
            adr = int(model.sensor_adr[s])

            # Try to get site name if sensor is attached to a site
            obj_id = model.sensor_objid[s]
            obj_type = model.sensor_objtype[s]
            site_name = None
            if obj_type == mujoco.mjtObj.mjOBJ_SITE and obj_id >= 0:
                site_name = model.site(obj_id).name

            self.sensors.append(
                SensorSchema(
                    sensor_name=sensor_name,
                    sensor_type=type_name,
                    dimension=dim,
                    adr=adr,
                    site_name=site_name,
                )
            )

    def _build_caches(self) -> None:
        """Build lookup caches for fast access."""
        self._joint_name_to_schema = {j.joint_name: j for j in self.joints}
        self._actuator_name_to_schema = {a.actuator_name: a for a in self.actuators}

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def validate(self) -> None:
        """Validate schema integrity.

        Raises:
            ValueError: If validation fails
        """
        errors = []

        # Check base is defined
        if self.base is None:
            errors.append("No floating base found")

        # Check actuators exist
        if len(self.actuators) == 0:
            errors.append("No actuators found")

        # Check foot geoms
        if len(self.left_foot_geoms) == 0:
            errors.append("No left foot geoms found")
        if len(self.right_foot_geoms) == 0:
            errors.append("No right foot geoms found")

        # Check no _mimic in actuators (already checked during extraction)
        for act in self.actuators:
            if "_mimic" in act.joint_name.lower():
                errors.append(f"Actuator '{act.actuator_name}' targets _mimic joint")

        # Check no _mimic in foot geoms
        for geom in self.left_foot_geoms + self.right_foot_geoms:
            if "_mimic" in geom.body_name.lower():
                errors.append(f"Foot geom '{geom.geom_name}' on _mimic body")

        if errors:
            raise ValueError(
                f"Schema validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def assert_no_mimic_in_actuators(self) -> None:
        """Assert that no actuator targets a _mimic joint."""
        for act in self.actuators:
            if "_mimic" in act.joint_name.lower():
                raise AssertionError(
                    f"Actuator '{act.actuator_name}' targets _mimic joint '{act.joint_name}'"
                )

    def assert_no_mimic_in_foot_geoms(self) -> None:
        """Assert that no foot geom belongs to a _mimic body."""
        for geom in self.left_foot_geoms + self.right_foot_geoms:
            if "_mimic" in geom.body_name.lower():
                raise AssertionError(
                    f"Foot geom '{geom.geom_name}' belongs to _mimic body '{geom.body_name}'"
                )

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "nq": self.nq,
            "nv": self.nv,
            "nu": self.nu,
            "joints": [j.to_dict() for j in self.joints],
            "actuators": [a.to_dict() for a in self.actuators],
            "base": self.base.to_dict() if self.base else None,
            "left_foot_geoms": [g.to_dict() for g in self.left_foot_geoms],
            "right_foot_geoms": [g.to_dict() for g in self.right_foot_geoms],
            "sensors": [s.to_dict() for s in self.sensors],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WildRobotSchema":
        """Load schema from dictionary."""
        schema = cls()
        schema.model_name = d.get("model_name", "")
        schema.nq = d.get("nq", 0)
        schema.nv = d.get("nv", 0)
        schema.nu = d.get("nu", 0)

        schema.joints = [JointSchema.from_dict(j) for j in d.get("joints", [])]
        schema.actuators = [ActuatorSchema.from_dict(a) for a in d.get("actuators", [])]

        if d.get("base"):
            schema.base = BaseSchema.from_dict(d["base"])

        schema.left_foot_geoms = [
            GeomSchema.from_dict(g) for g in d.get("left_foot_geoms", [])
        ]
        schema.right_foot_geoms = [
            GeomSchema.from_dict(g) for g in d.get("right_foot_geoms", [])
        ]
        schema.sensors = [SensorSchema.from_dict(s) for s in d.get("sensors", [])]

        schema._build_caches()
        return schema

    def save(self, path: str | Path) -> None:
        """Save schema to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "WildRobotSchema":
        """Load schema from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def assert_matches_saved(self, path: str | Path) -> None:
        """Assert this schema matches a saved schema file.

        This is the key regression test - if XML changes, this fails immediately.

        Args:
            path: Path to saved schema JSON

        Raises:
            AssertionError: If schemas don't match
        """
        saved = self.load(path)

        errors = []

        # Check dimensions
        if self.nq != saved.nq:
            errors.append(f"nq mismatch: {self.nq} vs saved {saved.nq}")
        if self.nv != saved.nv:
            errors.append(f"nv mismatch: {self.nv} vs saved {saved.nv}")
        if self.nu != saved.nu:
            errors.append(f"nu mismatch: {self.nu} vs saved {saved.nu}")

        # Check joints
        if len(self.joints) != len(saved.joints):
            errors.append(
                f"Joint count mismatch: {len(self.joints)} vs saved {len(saved.joints)}"
            )
        else:
            for i, (j1, j2) in enumerate(zip(self.joints, saved.joints)):
                if j1.joint_name != j2.joint_name:
                    errors.append(
                        f"Joint {i} name mismatch: {j1.joint_name} vs {j2.joint_name}"
                    )
                if j1.qpos_adr != j2.qpos_adr:
                    errors.append(
                        f"Joint {j1.joint_name} qpos_adr mismatch: {j1.qpos_adr} vs {j2.qpos_adr}"
                    )

        # Check actuators
        if len(self.actuators) != len(saved.actuators):
            errors.append(
                f"Actuator count mismatch: {len(self.actuators)} vs saved {len(saved.actuators)}"
            )
        else:
            for i, (a1, a2) in enumerate(zip(self.actuators, saved.actuators)):
                if a1.actuator_name != a2.actuator_name:
                    errors.append(
                        f"Actuator {i} name mismatch: {a1.actuator_name} vs {a2.actuator_name}"
                    )
                if a1.dof_adr != a2.dof_adr:
                    errors.append(
                        f"Actuator {a1.actuator_name} dof_adr mismatch: {a1.dof_adr} vs {a2.dof_adr}"
                    )

        if errors:
            raise AssertionError(
                "Schema has changed from saved version!\n"
                + "This indicates the XML model was modified.\n"
                + "If intentional, update the saved schema with schema.save().\n"
                + "Differences:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # =========================================================================
    # Accessor Methods
    # =========================================================================

    def get_joint_qpos_slice(self, joint_name: str) -> slice:
        """Get qpos slice for a joint."""
        j = self._joint_name_to_schema.get(joint_name)
        if j is None:
            raise KeyError(f"Joint not found: {joint_name}")
        return slice(j.qpos_adr, j.qpos_adr + 1)

    def get_actuated_joint_qpos_indices(self) -> List[int]:
        """Get qpos indices for all actuated joints."""
        indices = []
        for act in self.actuators:
            j = self._joint_name_to_schema.get(act.joint_name)
            if j:
                indices.append(j.qpos_adr)
        return indices

    def get_actuated_joint_dof_indices(self) -> List[int]:
        """Get qvel/dof indices for all actuated joints."""
        return [a.dof_adr for a in self.actuators]

    def get_left_foot_geom_ids(self) -> List[int]:
        """Get geom IDs for left foot."""
        return [g.geom_id for g in self.left_foot_geoms]

    def get_right_foot_geom_ids(self) -> List[int]:
        """Get geom IDs for right foot."""
        return [g.geom_id for g in self.right_foot_geoms]

    def get_all_foot_geom_ids(self) -> List[int]:
        """Get all foot geom IDs."""
        return self.get_left_foot_geom_ids() + self.get_right_foot_geom_ids()

    def get_base_qpos_slice(self) -> slice:
        """Get qpos slice for floating base (7D: pos + quat)."""
        if self.base is None:
            raise ValueError("No floating base defined")
        return slice(self.base.base_qpos_adr, self.base.base_qpos_adr + 7)

    def get_base_qvel_slice(self) -> slice:
        """Get qvel slice for floating base (6D: linvel + angvel)."""
        if self.base is None:
            raise ValueError("No floating base defined")
        return slice(self.base.base_dof_adr, self.base.base_dof_adr + 6)


# =============================================================================
# Utility Functions
# =============================================================================


def extract_and_save_schema(
    xml_path: str = "assets/v1/scene_flat_terrain.xml",
    output_path: str = "assets/robot_schema.json",
) -> WildRobotSchema:
    """Extract schema from XML and save to JSON.

    Run this once when the robot model is finalized.
    """
    schema = WildRobotSchema.from_xml(xml_path)
    schema.validate()
    schema.save(output_path)
    print(f"Schema saved to {output_path}")
    print(f"  Joints: {len(schema.joints)}")
    print(f"  Actuators: {len(schema.actuators)}")
    print(f"  Left foot geoms: {len(schema.left_foot_geoms)}")
    print(f"  Right foot geoms: {len(schema.right_foot_geoms)}")
    print(f"  Sensors: {len(schema.sensors)}")
    return schema


if __name__ == "__main__":
    # Example usage
    schema = extract_and_save_schema()
    print("\nSchema contract created successfully!")
    print(json.dumps(schema.to_dict(), indent=2))
