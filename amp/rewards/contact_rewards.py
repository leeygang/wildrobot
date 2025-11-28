"""Foot contact reward components for natural gait."""

from typing import Dict, List
import jax
import jax.numpy as jp
from mujoco import mjx


class FootContactReward:
    """Reward proper foot contact patterns during walking.

    Rewards feet being in contact during stance phase and off ground during swing phase,
    based on the phase clock signal.
    """

    def __init__(
        self,
        model: mjx.Model,
        left_foot_geoms: List[str],
        right_foot_geoms: List[str],
        contact_threshold: float = 1.0,
    ):
        """Initialize foot contact reward.

        Args:
            model: MuJoCo model
            left_foot_geoms: Names of left foot collision geoms
            right_foot_geoms: Names of right foot collision geoms
            contact_threshold: Minimum force (N) to consider as contact
        """
        self.model = model
        self.contact_threshold = contact_threshold

        # Get geom IDs for contact detection
        self.left_foot_geom_ids = jp.array([
            model.geom(name).id for name in left_foot_geoms
        ])
        self.right_foot_geom_ids = jp.array([
            model.geom(name).id for name in right_foot_geoms
        ])

    def get_contact_force(self, data: mjx.Data, geom_ids: jax.Array) -> jax.Array:
        """Sum contact forces on specified geoms.

        Args:
            data: MuJoCo data
            geom_ids: Array of geom IDs to check

        Returns:
            Total contact force magnitude
        """
        # Check which contacts involve our geoms
        geom1_matches = jp.isin(data.contact.geom1, geom_ids)
        geom2_matches = jp.isin(data.contact.geom2, geom_ids)
        our_contacts = geom1_matches | geom2_matches

        # Get normal force for each contact using direct array indexing
        # efc_address[i] is the index where contact i's forces start in efc_force
        # The first element at each address is the normal force
        normal_forces = data.efc_force[data.contact.efc_address]

        # Sum forces for our contacts only
        our_forces = jp.where(our_contacts, jp.abs(normal_forces), 0.0)
        return jp.sum(our_forces)

    def compute(
        self,
        data: mjx.Data,
        action: jax.Array,
        phase: float = 0.0,
        **kwargs
    ) -> jax.Array:
        """Compute foot contact reward based on phase.

        Args:
            data: MuJoCo data
            action: Action taken
            phase: Gait phase (0.0 to 1.0)
                   0.0-0.5: left foot stance, right foot swing
                   0.5-1.0: right foot stance, left foot swing

        Returns:
            Reward value
        """
        # Get contact forces
        left_force = self.get_contact_force(data, self.left_foot_geom_ids)
        right_force = self.get_contact_force(data, self.right_foot_geom_ids)

        # Normalize forces with tanh (saturates at high forces)
        left_contact = jp.tanh(left_force / self.contact_threshold)
        right_contact = jp.tanh(right_force / self.contact_threshold)

        # Determine which foot should be in contact based on phase
        # Phase 0-0.5: left stance (should contact), right swing (should not contact)
        # Phase 0.5-1.0: right stance (should contact), left swing (should not contact)
        left_should_contact = phase < 0.5
        right_should_contact = phase >= 0.5

        # Reward matching expected contact pattern
        left_reward = jp.where(left_should_contact, left_contact, -left_contact)
        right_reward = jp.where(right_should_contact, right_contact, -right_contact)

        return left_reward + right_reward

    def get_info(self, data: mjx.Data) -> Dict[str, float]:
        """Get contact forces for debugging."""
        left_force = self.get_contact_force(data, self.left_foot_geom_ids)
        right_force = self.get_contact_force(data, self.right_foot_geom_ids)

        return {
            'contact_force_left': float(left_force),
            'contact_force_right': float(right_force),
        }


class FootSlidingPenalty:
    """Penalize horizontal foot movement when in contact with ground.

    Prevents unnatural sliding/skating during stance phase.
    """

    def __init__(
        self,
        model: mjx.Model,
        left_foot_site: str,
        right_foot_site: str,
        left_foot_geoms: List[str],
        right_foot_geoms: List[str],
        contact_threshold: float = 1.0,
    ):
        """Initialize foot sliding penalty.

        Args:
            model: MuJoCo model
            left_foot_site: Name of left foot site (for velocity)
            right_foot_site: Name of right foot site (for velocity)
            left_foot_geoms: Names of left foot collision geoms
            right_foot_geoms: Names of right foot collision geoms
            contact_threshold: Minimum force (N) to consider as contact
        """
        self.model = model
        self.contact_threshold = contact_threshold

        # Get site IDs for velocity queries
        self.left_foot_site_id = model.site(left_foot_site).id
        self.right_foot_site_id = model.site(right_foot_site).id

        # Get body IDs that sites are attached to
        self.left_foot_body_id = model.site(left_foot_site).bodyid
        self.right_foot_body_id = model.site(right_foot_site).bodyid

        # Get geom IDs for contact detection
        self.left_foot_geom_ids = jp.array([
            model.geom(name).id for name in left_foot_geoms
        ])
        self.right_foot_geom_ids = jp.array([
            model.geom(name).id for name in right_foot_geoms
        ])

        # Reuse contact detection from FootContactReward
        self.contact_reward = FootContactReward(
            model, left_foot_geoms, right_foot_geoms, contact_threshold
        )

    def get_site_velocity(self, data: mjx.Data, site_id: int, body_id: int) -> jax.Array:
        """Compute site velocity from body velocity.

        In MJX, site velocities are not directly available. We compute them from:
        v_site = v_body + omega_body × r_site_body

        Args:
            data: MJX data
            site_id: Site ID
            body_id: Body ID that the site is attached to

        Returns:
            Site velocity (3D)
        """
        # Get body velocity (linear and angular)
        # data.cvel has shape [..., num_bodies, 6] where last 6 is [linvel(3), angvel(3)]
        body_vel = data.cvel[body_id]
        body_linvel = body_vel[..., :3]  # Linear velocity (last 3 dims)
        body_angvel = body_vel[..., 3:6]  # Angular velocity (last 3 dims)

        # Get site position and body position
        site_pos = data.site_xpos[site_id]
        body_pos = data.xpos[body_id]

        # Compute site offset from body
        site_offset = site_pos - body_pos

        # Compute site velocity: v_site = v_body + omega × r
        site_vel = body_linvel + jp.cross(body_angvel, site_offset)

        return site_vel

    def compute(
        self,
        data: mjx.Data,
        action: jax.Array,
        **kwargs
    ) -> jax.Array:
        """Compute foot sliding penalty.

        Args:
            data: MuJoCo data
            action: Action taken

        Returns:
            Penalty value (negative)
        """
        # Get foot velocities using body kinematics
        left_foot_vel = self.get_site_velocity(data, self.left_foot_site_id, self.left_foot_body_id)
        right_foot_vel = self.get_site_velocity(data, self.right_foot_site_id, self.right_foot_body_id)

        # Horizontal velocity only (XY plane, ignore Z)
        left_sliding = jp.linalg.norm(left_foot_vel[..., :2])
        right_sliding = jp.linalg.norm(right_foot_vel[..., :2])

        # Check if feet are in contact
        left_force = self.contact_reward.get_contact_force(data, self.left_foot_geom_ids)
        right_force = self.contact_reward.get_contact_force(data, self.right_foot_geom_ids)

        left_in_contact = left_force > self.contact_threshold
        right_in_contact = right_force > self.contact_threshold

        # Penalize sliding only when in contact
        penalty = 0.0
        penalty += jp.where(left_in_contact, left_sliding, 0.0)
        penalty += jp.where(right_in_contact, right_sliding, 0.0)

        return -penalty  # Negative because it's a penalty

    def get_info(self, data: mjx.Data) -> Dict[str, float]:
        """Get sliding velocities for debugging."""
        left_foot_vel = self.get_site_velocity(data, self.left_foot_site_id, self.left_foot_body_id)
        right_foot_vel = self.get_site_velocity(data, self.right_foot_site_id, self.right_foot_body_id)

        left_sliding = jp.linalg.norm(left_foot_vel[..., :2])
        right_sliding = jp.linalg.norm(right_foot_vel[..., :2])

        return {
            'sliding_vel_left': float(left_sliding),
            'sliding_vel_right': float(right_sliding),
        }
