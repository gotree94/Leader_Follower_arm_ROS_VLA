# Bimanual Collaboration — Dual-Arm Coordination

> 두 팔 (Leader Franka Panda + Follower UR5e) 간 협업 모드 및 충돌 회피

---

## 1. Bimanual Collaboration Modes

### 1.1 세 가지 협업 모드

| 모드 | 설명 | 사용 예 |
|------|------|---------|
| **Independent** | 각 팔이 독립적인 태스크 수행 | Leader: 물체 잡기, Follower: 다른 물체 옮기기 |
| **Coordinated** | 상호 보완적 태스크 | Leader: 용기 잡기, Follower: 물체 넣기 |
| **Collaborative** | 동일 물체 협력 | 두 팔이 함께 무거운 물체 들어올리기 |

### 1.2 모드 전환 다이어그램

```
               ┌──────────────┐
               │  Independent │ ← 각자 태스크
               └──────┬───────┘
                      │ Task requires interaction?
               Yes    │    No
              ┌───────┘    └───────┐
              ▼                    ▼
       ┌──────────────┐    ┌──────────────┐
       │ Coordinated  │    │  Independent │
       └──────┬───────┘    └──────────────┘
              │ Same object?
        Yes   │    No
       ┌──────┘    └──────┐
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│ Collaborative│   │ Coordinated  │
└──────────────┘   └──────────────┘
```

---

## 2. 좌표계 및 캘리브레이션

### 2.1 좌표계 구성

```
   Leader Base              Follower Base
   ─────────────             ─────────────
        │                         │
        ▼                         ▼
   leader_base_link          follower_base_link
        │                         │
   leader_ee_pose            follower_ee_pose
        │                         │
        └──────────┬──────────────┘
                   ▼
            world_frame (shared)
                   │
            table_surface_frame
                   │
            object_frame
```

### 2.2 TF2 Tree

```xml
<!-- TF2 Transformation Tree -->
<frame name="world"/>
<frame name="leader_base" parent="world">
    <origin xyz="0.30 0 0" rpy="0 0 3.14159"/>
</frame>
<frame name="follower_base" parent="world">
    <origin xyz="-0.30 0 0" rpy="0 0 0"/>
</frame>
<frame name="table" parent="world">
    <origin xyz="0 0 0.70"/>
</frame>
<frame name="blue_block" parent="table"/>
<frame name="red_box" parent="table"/>
```

### 2.3 Hand-Eye Calibration

```python
# scripts/calibrate_bimanual.py
"""
양팔 Hand-Eye 캘리브레이션
"""
import numpy as np
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation


class BimanualCalibrator:
    """
    Leader-Follower 좌표계 통합 캘리브레이션
    
    1. Leader arm → camera calibration
    2. Follower arm → camera calibration  
    3. Leader → Follower relative pose
    """
    
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
    
    def compute_leader_follower_transform(self):
        """
        Leader base → Follower base 변환 행렬
        
        Returns: (4×4) homogeneous transform T_leader^follower
        """
        try:
            # TF2에서 변환 조회
            trans = self.tf_buffer.lookup_transform(
                "follower_base", "leader_base", 
                time=rclpy.time.Time()
            )
            
            t = trans.transform.translation
            r = trans.transform.rotation
            
            # 4×4 행렬로 변환
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([
                r.x, r.y, r.z, r.w
            ]).as_matrix()
            T[:3, 3] = [t.x, t.y, t.z]
            
            return T
            
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            print(f"TF lookup failed: {e}")
            return None
    
    def compute_shared_workspace(self):
        """
        양팔 공동 작업 공간 계산
        
        Returns: (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        T = self.compute_leader_follower_transform()
        if T is None:
            return None
        
        # 각 Arm의 reachable workspace
        leader_workspace = self._get_workspace_leader()
        follower_workspace = self._get_workspace_follower()
        
        # Follower workspace를 Leader 좌표계로 변환
        follower_in_leader = np.linalg.inv(T) @ follower_workspace
        
        # Intersection
        shared_min = np.maximum(
            leader_workspace[:3],
            follower_in_leader[:3]
        )
        shared_max = np.minimum(
            leader_workspace[3:],
            follower_in_leader[3:]
        )
        
        return (*shared_min, *shared_max)
    
    def _get_workspace_leader(self) -> np.ndarray:
        """Franka Panda reachable workspace (approx)"""
        return np.array([-0.3, -0.3, 0.0, 0.3, 0.3, 0.5])
    
    def _get_workspace_follower(self) -> np.ndarray:
        """UR5e reachable workspace (approx)"""
        return np.array([-0.3, -0.3, 0.0, 0.3, 0.3, 0.5])
```

---

## 3. 충돌 회피

### 3.1 충돌 영역 정의

```
Top View:
                          y
                          ▲
                          │
  Leader Arm              │          Follower Arm
  (Franka)                │          (UR5e)
     │                    │             │
     │    ┌───────────────┼────────┐    │
     │    │  Collision    │  Zone  │    │
     │    │  Zone 1       │   2    │    │
     │    │  (Leader)     │(Follower)   │
     │    │               │        │    │
     │    └───────────────┼────────┘    │
     │                    │             │
     ├────────────────────┼─────────────┤──► x
    -0.3                 0            0.3
```

### 3.2 충돌 감지 구현

```python
# src/follower_arm/safety_monitor.py
"""
이중 Arm 충돌 감지 및 회피
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ArmState:
    """Arm 상태 정보"""
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_pose: np.ndarray  # (7,) [x,y,z,qw,qx,qy,qz]
    link_poses: List[np.ndarray]  # 각 link의 world pose


class BimanualSafetyMonitor:
    """
    양팔 안전 모니터링
    
    - Inter-arm collision (팔 간 충돌)
    - Self-collision (자체 충돌)
    - Environment collision (환경 충돌)
    - Joint limit violation
    - Velocity limit violation
    """
    
    def __init__(self):
        # Link collision spheres (simplified)
        self.leader_spheres = self._create_arm_spheres("leader")
        self.follower_spheres = self._create_arm_spheres("follower")
        
        # Safety thresholds
        self.min_inter_arm_distance = 0.05  # m (5cm)
        self.min_env_distance = 0.02  # m (2cm)
        self.joint_vel_limit = 2.0  # rad/s
        self.collision_margin = 0.03  # m
    
    def _create_arm_spheres(self, arm_name: str) -> dict:
        """
        각 링크를 단순화한 충돌 구체 정의
        
        Link → (center_offset, radius)
        """
        if arm_name == "leader":
            return {
                "link0": (np.array([0, 0, 0.15]), 0.10),
                "link1": (np.array([0, 0, 0.10]), 0.08),
                "link2": (np.array([0, 0, 0.12]), 0.08),
                "link3": (np.array([0.04, 0, 0.10]), 0.07),
                "link4": (np.array([0, 0, 0.08]), 0.06),
                "link5": (np.array([0, 0, 0.08]), 0.05),
                "link6": (np.array([0, 0, 0.06]), 0.05),
                "link7": (np.array([0.04, 0, 0.04]), 0.05),
                "hand": (np.array([0.08, 0, 0.02]), 0.06),
                "gripper": (np.array([0.10, 0, 0.01]), 0.04),
            }
        else:
            return {
                "base": (np.array([0, 0, 0.08]), 0.12),
                "shoulder": (np.array([0, 0, 0.12]), 0.10),
                "upper_arm": (np.array([0, -0.10, 0.10]), 0.08),
                "forearm": (np.array([0, -0.10, 0.08]), 0.07),
                "wrist1": (np.array([0, 0, 0.05]), 0.05),
                "wrist2": (np.array([0, 0, 0.04]), 0.05),
                "wrist3": (np.array([0, 0, 0.04]), 0.05),
                "gripper": (np.array([0, 0, 0.06]), 0.05),
            }
    
    def check_inter_arm_collision(
        self,
        leader_state: ArmState,
        follower_state: ArmState
    ) -> Tuple[bool, float]:
        """
        팔 간 충돌 검사
        
        Returns:
            (collision_detected, min_distance)
        """
        min_dist = float("inf")
        
        for l_name, l_sphere in self.leader_spheres.items():
            l_offset, l_radius = l_sphere
            
            for f_name, f_sphere in self.follower_spheres.items():
                f_offset, f_radius = f_sphere
                
                # Transform sphere center to world frame
                l_center = self._transform_to_world(
                    l_offset, leader_state, l_name
                )
                f_center = self._transform_to_world(
                    f_offset, follower_state, f_name
                )
                
                # Distance between sphere centers
                dist = np.linalg.norm(l_center - f_center)
                min_dist = min(min_dist, dist)
                
                # Check collision
                if dist < (l_radius + f_radius + self.collision_margin):
                    return True, min_dist
        
        return False, min_dist
    
    def check_self_collision(
        self,
        state: ArmState,
        arm_type: str = "follower"
    ) -> Tuple[bool, str]:
        """자체 충돌 검사"""
        spheres = self.follower_spheres if arm_type == "follower" else self.leader_spheres
        names = list(spheres.keys())
        
        for i in range(len(names)):
            for j in range(i + 2, len(names)):  # Skip adjacent links
                if abs(i - j) <= 1:
                    continue
                
                s1_offset, s1_radius = spheres[names[i]]
                s2_offset, s2_radius = spheres[names[j]]
                
                c1 = self._transform_to_world(s1_offset, state, names[i])
                c2 = self._transform_to_world(s2_offset, state, names[j])
                
                dist = np.linalg.norm(c1 - c2)
                if dist < (s1_radius + s2_radius):
                    return True, f"{names[i]} ↔ {names[j]}"
        
        return False, ""
    
    def check_joint_limits(self, state: ArmState) -> Tuple[bool, str]:
        """조인트 제한 위반 검사"""
        # Velocity limits
        if np.any(np.abs(state.joint_velocities) > self.joint_vel_limit):
            idx = np.argmax(np.abs(state.joint_velocities))
            return True, f"Joint {idx} velocity limit exceeded"
        
        return False, ""
    
    def _transform_to_world(
        self,
        offset: np.ndarray,
        state: ArmState,
        link_name: str
    ) -> np.ndarray:
        """로컬 offset → World 좌표 변환"""
        # Simplified: uses FK result from Isaac Sim
        # 실제로는 robot's forward kinematics 적용
        return state.ee_pose[:3] + offset  # Approximation
```

### 3.3 우선순위 기반 충돌 해결

```python
class CollisionResolver:
    """
    충돌 상황 해결 전략
    
    우선순위:
    1. Safety critical (즉시 정지)
    2. Active task (진행 중인 태스크 우선)
    3. Predefined arm priority (Leader > Follower)
    """
    
    PRIORITY_MAP = {
        "emergency_stop": 0,  # Highest
        "active_task": 1,
        "leader_arm": 2,
        "follower_arm": 3,
        "background_task": 4,
    }
    
    def resolve(self, collision_info: dict) -> dict:
        """
        충돌 해결 액션 생성
        
        Returns:
            {
                "leader_action": "stop" | "slow" | "retreat" | "continue",
                "follower_action": "stop" | "slow" | "retreat" | "continue",
                "description": "..."
            }
        """
        action = {
            "leader_action": "continue",
            "follower_action": "continue",
            "description": "No conflict"
        }
        
        min_dist = collision_info.get("min_distance", float("inf"))
        
        if min_dist < 0.02:  # Critical: 즉시 정지
            action["leader_action"] = "stop"
            action["follower_action"] = "stop"
            action["description"] = "Emergency stop - critical proximity"
            
        elif min_dist < 0.05:  # Warning: 속도 감소 + 후퇴
            task_priority = collision_info.get("task_priority", 3)
            
            if task_priority <= 2:  # Active task 유지
                lower_priority = "follower" if task_priority == 1 else "leader"
                action[f"{lower_priority}_action"] = "retreat"
                action["description"] = f"{lower_priority} retreats"
            else:
                action["leader_action"] = "slow"
                action["follower_action"] = "slow"
                action["description"] = "Both slow down"
        
        return action
```

---

## 4. Task Allocation

### 4.1 중앙 집중식 태스크 플래너

```python
# src/cosmos_reason/bimanual_planner.py
"""
양팔 협업 태스크 할당
"""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ArmTask:
    arm: str  # "leader" or "follower"
    action: str
    target: str
    pose: np.ndarray
    constraints: dict
    dependencies: List[str]  # 선행 태스크 ID


class BimanualTaskAllocator:
    """
    두 Arm에 태스크 분배
    
    - Dependency graph 기반 순서 결정
    - 충돌 위험 최소화
    - 부하 균형
    """
    
    def allocate(self, task_plan: dict) -> Dict[str, List[ArmTask]]:
        """
        태스크 플랜 → Arm별 태스크 할당
        
        Example:
            Input: "Hold the board and screw the bolt"
            
            Output:
            leader: [ArmTask(action="hold_pose", target="board")]
            follower: [ArmTask(action="reach", target="bolt"),
                       ArmTask(action="screw", target="bolt")]
        """
        leader_tasks = []
        follower_tasks = []
        
        for step in task_plan["steps"]:
            if self._requires_both_arms(step):
                # 양팔 필요 → 하나는 hold, 하나는 manipulate
                leader_tasks.append(ArmTask(
                    arm="leader",
                    action="hold",
                    target=step.get("hold_target", "object"),
                    pose=step.get("hold_pose", np.zeros(7)),
                    constraints={"stiffness": "high"},
                    dependencies=[]
                ))
                follower_tasks.append(ArmTask(
                    arm="follower",
                    action=step["action"],
                    target=step["target"],
                    pose=step["target_pose"],
                    constraints={"precision": "high"},
                    dependencies=["leader_hold"]
                ))
            elif self._prefers_leader(step):
                leader_tasks.append(self._to_arm_task("leader", step))
            else:
                follower_tasks.append(self._to_arm_task("follower", step))
        
        return {"leader": leader_tasks, "follower": follower_tasks}
    
    def _requires_both_arms(self, step: dict) -> bool:
        """양팔이 필요한 작업인지 확인"""
        two_arm_actions = ["hold_and_manipulate", "lift_together", "assemble"]
        return step.get("action") in two_arm_actions
    
    def _prefers_leader(self, step: dict) -> bool:
        """Leader Arm 선호 작업"""
        leader_actions = ["demonstrate", "precise_grasp", "complex_manipulation"]
        return step.get("action") in leader_actions
```

---

## 5. Bimanual VLA Policy

### 5.1 Joint Action Space

```python
# Bimanual VLA: Joint observation + action space

Observation (Leader + Follower):
├── leader_rgb: (3, 224, 224)
├── follower_rgb: (3, 224, 224)
├── depth_leader: (1, 224, 224)
├── depth_follower: (1, 224, 224)
├── leader_joints: (7,)
├── follower_joints: (6,)
├── leader_gripper: (1,)
├── follower_gripper: (1,)
├── lang_embed: (512,)
├── leader_ee_pose: (7,)
├── follower_ee_pose: (7,)
├── relative_pose: (7,)  ← leader → follower transform
└── interaction_force: (6,)  ← wrist force torque

Action:
├── leader_joint_vel: (7,)
├── follower_joint_vel: (6,)
├── leader_gripper: (1,)
├── follower_gripper: (1,)
└── coordination_mode: (3,)  ← one-hot: independent/coordinated/collaborative
```

### 5.2 Coordination Reward

```python
def compute_bimanual_reward(env_state: dict) -> float:
    """
    양팔 협업 보상 함수
    
    개별 태스크 보상 + 협업 보상
    """
    reward = 0.0
    
    # Individual task rewards
    reward += env_state["leader_task_reward"]
    reward += env_state["follower_task_reward"]
    
    # Coordination bonus
    leader_ee = env_state["leader_ee_pose"][:3]
    follower_ee = env_state["follower_ee_pose"][:3]
    dist = np.linalg.norm(leader_ee - follower_ee)
    
    if env_state["mode"] == "coordinated":
        # 적정 거리 유지 (너무 가깝거나 멀지 않게)
        optimal_dist = 0.15
        reward -= 5.0 * abs(dist - optimal_dist)
    
    elif env_state["mode"] == "collaborative":
        # 같은 물체 → EE 거리 가까워야 함
        reward -= 10.0 * max(0, dist - 0.1)
    
    # Collision penalty
    if env_state["collision"]:
        reward -= 5.0
    
    # Synchronization bonus
    leader_progress = env_state["leader_task_progress"]
    follower_progress = env_state["follower_task_progress"]
    sync_diff = abs(leader_progress - follower_progress)
    reward -= 2.0 * sync_diff  # 동기화되어야 높은 보상
    
    return reward
```

---

## 6. Bimanual Digital Twin

양팔 Digital Twin은 TurtleBot3와 동일한 4단계 구조를 따르되, 양팔 메트릭이 추가됩니다.

| 메트릭 | Leader | Follower | Bimanual |
|--------|--------|----------|----------|
| Task Success Rate | ✅ | ✅ | ✅ (both succeed) |
| Coordination Delay | ❌ | ❌ | ✅ |
| Inter-arm Distance | ❌ | ❌ | ✅ |
| Collision Rate | ✅ | ✅ | ✅ (inter-arm) |
| Synchronization | ❌ | ❌ | ✅ |
| Force Interaction | ❌ | ✅ | ✅ |
