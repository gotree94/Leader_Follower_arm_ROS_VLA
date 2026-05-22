"""
Bimanual Safety Monitor
충돌 감지 및 안전 모니터링
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ArmState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_pose: np.ndarray


class BimanualSafetyMonitor:
    def __init__(self):
        self.leader_spheres = self._create_spheres("leader")
        self.follower_spheres = self._create_spheres("follower")
        self.min_inter_arm_distance = 0.05
        self.joint_vel_limit = 2.0
        self.collision_margin = 0.03

    def _create_spheres(self, arm: str) -> dict:
        if arm == "leader":
            return {
                "link0": (np.array([0, 0, 0.15]), 0.10),
                "link1": (np.array([0, 0, 0.10]), 0.08),
                "link2": (np.array([0, 0, 0.12]), 0.08),
                "hand": (np.array([0.08, 0, 0.02]), 0.06),
            }
        return {
            "base": (np.array([0, 0, 0.08]), 0.12),
            "shoulder": (np.array([0, 0, 0.12]), 0.10),
            "upper_arm": (np.array([0, -0.10, 0.10]), 0.08),
            "gripper": (np.array([0, 0, 0.06]), 0.05),
        }

    def check_inter_arm_collision(self, leader: ArmState, follower: ArmState) -> Tuple[bool, float]:
        min_dist = float("inf")
        for l_name, (l_off, l_r) in self.leader_spheres.items():
            for f_name, (f_off, f_r) in self.follower_spheres.items():
                l_center = leader.ee_pose[:3] + l_off
                f_center = follower.ee_pose[:3] + f_off
                dist = np.linalg.norm(l_center - f_center)
                min_dist = min(min_dist, dist)
                if dist < (l_r + f_r + self.collision_margin):
                    return True, min_dist
        return False, min_dist

    def check_joint_limits(self, state: ArmState) -> Tuple[bool, str]:
        if np.any(np.abs(state.joint_velocities) > self.joint_vel_limit):
            idx = np.argmax(np.abs(state.joint_velocities))
            return True, f"Joint {idx} velocity limit exceeded"
        return False, ""
