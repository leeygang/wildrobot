from __future__ import annotations

from typing import Protocol


class CalibOps(Protocol):
    def action_to_ctrl(self, *, spec, action):
        ...

    def ctrl_to_policy_action(self, *, spec, ctrl_rad):
        ...

    def normalize_joint_pos(self, *, spec, joint_pos_rad):
        ...

    def normalize_joint_vel(self, *, spec, joint_vel_rad_s):
        ...


class NumpyCalibOps(CalibOps):
    @staticmethod
    def action_to_ctrl(*, spec, action):
        from policy_contract.numpy.calib import action_to_ctrl

        return action_to_ctrl(spec=spec, action=action)

    @staticmethod
    def ctrl_to_policy_action(*, spec, ctrl_rad):
        from policy_contract.numpy.calib import ctrl_to_policy_action

        return ctrl_to_policy_action(spec=spec, ctrl_rad=ctrl_rad)

    @staticmethod
    def normalize_joint_pos(*, spec, joint_pos_rad):
        from policy_contract.numpy.calib import normalize_joint_pos

        return normalize_joint_pos(spec=spec, joint_pos_rad=joint_pos_rad)

    @staticmethod
    def normalize_joint_vel(*, spec, joint_vel_rad_s):
        from policy_contract.numpy.calib import normalize_joint_vel

        return normalize_joint_vel(spec=spec, joint_vel_rad_s=joint_vel_rad_s)


class JaxCalibOps(CalibOps):
    @staticmethod
    def action_to_ctrl(*, spec, action):
        from policy_contract.jax.calib import action_to_ctrl

        return action_to_ctrl(spec=spec, action=action)

    @staticmethod
    def ctrl_to_policy_action(*, spec, ctrl_rad):
        from policy_contract.jax.calib import ctrl_to_policy_action

        return ctrl_to_policy_action(spec=spec, ctrl_rad=ctrl_rad)

    @staticmethod
    def normalize_joint_pos(*, spec, joint_pos_rad):
        from policy_contract.jax.calib import normalize_joint_pos

        return normalize_joint_pos(spec=spec, joint_pos_rad=joint_pos_rad)

    @staticmethod
    def normalize_joint_vel(*, spec, joint_vel_rad_s):
        from policy_contract.jax.calib import normalize_joint_vel

        return normalize_joint_vel(spec=spec, joint_vel_rad_s=joint_vel_rad_s)
