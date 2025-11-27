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
        def check_contact(contact):
            # Check if either geom in the contact pair matches our geoms
            geom1_match = jp.isin(contact.geom1, geom_ids)
            geom2_match = jp.isin(contact.geom2, geom_ids)
            is_our_contact = geom1_match | geom2_match

            # Get contact force (first 3 elements of force vector)
            force_mag = jp.linalg.norm(contact.force[:3])

            return jp.where(is_our_contact, force_mag, 0.0)

        # Sum forces across all contacts
        forces = jax.vmap(check_contact)(data.contact)
        return jp.sum(forces)

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
        # Get foot velocities from site sensors
        left_foot_vel = data.site_xvelp[self.left_foot_site_id]
        right_foot_vel = data.site_xvelp[self.right_foot_site_id]

        # Horizontal velocity only (XY plane, ignore Z)
        left_sliding = jp.linalg.norm(left_foot_vel[:2])
        right_sliding = jp.linalg.norm(right_foot_vel[:2])

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
        left_foot_vel = data.site_xvelp[self.left_foot_site_id]
        right_foot_vel = data.site_xvelp[self.right_foot_site_id]

        left_sliding = jp.linalg.norm(left_foot_vel[:2])
        right_sliding = jp.linalg.norm(right_foot_vel[:2])

        return {
            'sliding_vel_left': float(left_sliding),
            'sliding_vel_right': float(right_sliding),
        }
