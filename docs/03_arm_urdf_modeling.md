# Robot Arm URDF Modeling for Isaac Sim

> Franka Emika Panda (Leader) + UR5e (Follower) URDF 모델링과 Isaac Sim USD 변환

---

## 1. Franka Emika Panda URDF (Leader Arm)

### 1.1 Kinematics Chain

```
J1 (rev, waist) ── Link 0 ──> J2 (rev, shoulder) ── Link 1 ──> J3 (rev, elbow)
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
    │                                                                │
    Link 2 ──> J4 (rev, forearm) ── Link 3 ──> J5 (rev, wrist)
        │                                                         │
        └─────────────────────────────────────────────────────────┘
        │                                                         │
        Link 4 ──> J6 (rev, wrist) ── Link 5 ──> J7 (rev, flange) ──> Gripper
```

**DH Parameters (Franka Panda)**

| Joint | a (m) | α (rad) | d (m) | θ (rad) | Joint Range |
|-------|-------|---------|-------|---------|-------------|
| J1 | 0 | 0 | 0.333 | θ1 | -166° ~ +166° |
| J2 | 0 | -π/2 | 0 | θ2 | -101° ~ +101° |
| J3 | 0 | π/2 | 0.316 | θ3 | -166° ~ +166° |
| J4 | 0.0825 | π/2 | 0 | θ4 | -176° ~ +176° |
| J5 | -0.0825 | -π/2 | 0.384 | θ5 | -166° ~ +166° |
| J6 | 0 | π/2 | 0 | θ6 | -1° ~ +215° |
| J7 | 0.088 | π/2 | 0 | θ7 | -166° ~ +166° |

### 1.2 URDF 구조 (핵심)

```xml
<?xml version="1.0"?>
<robot name="franka_panda" xmlns:xacro="http://ros.org/wiki/xacro">
  <!-- Colors -->
  <material name="white"><color rgba="0.9 0.9 0.9 1.0"/></material>
  <material name="black"><color rgba="0.1 0.1 0.1 1.0"/></material>
  <material name="red"><color rgba="0.7 0.1 0.1 1.0"/></material>

  <!-- WORLD → panda_link0 -->
  <link name="world"/>
  <joint name="panda_fixed_joint" type="fixed">
    <parent link="world"/>
    <child link="panda_link0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Link 0 (Base) -->
  <link name="panda_link0">
    <inertial>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://franka_description/meshes/visual/link0.dae"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://franka_description/meshes/collision/link0.stl"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint 1 -->
  <joint name="panda_joint1" type="revolute">
    <parent link="panda_link0"/>
    <child link="panda_link1"/>
    <origin xyz="0 0 0.333" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.897" upper="2.897" effort="87.0" velocity="2.175"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <!-- ... J2-J7 유사 패턴 ... -->

  <!-- Franka Hand Gripper -->
  <link name="panda_hand">
    <inertial>
      <mass value="0.7"/>
      <!-- ... -->
    </inertial>
    <visual>
      <geometry>
        <mesh filename="package://franka_description/meshes/visual/hand.dae"/>
      </geometry>
    </visual>
  </link>

  <joint name="panda_joint7" type="fixed">
    <parent link="panda_link7"/>
    <child link="panda_hand"/>
    <origin xyz="0 0 0.088" rpy="0 0 0"/>
  </joint>

  <!-- Gripper Fingers -->
  <link name="panda_left_finger"/>
  <link name="panda_right_finger"/>

  <joint name="panda_finger_joint1" type="prismatic">
    <parent link="panda_hand"/>
    <child link="panda_left_finger"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="0.04" effort="20.0" velocity="0.2"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <joint name="panda_finger_joint2" type="prismatic">
    <parent link="panda_hand"/>
    <child link="panda_right_finger"/>
    <origin xyz="0 0 0.04"/>
    <axis xyz="0 0 -1"/>
    <limit lower="0.0" upper="0.04" effort="20.0" velocity="0.2"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <!-- ROS2 Control -->
  <ros2_control name="FrankaArm" type="system">
    <hardware>
      <plugin>franka_hardware/FrankaHardwareInterface</plugin>
      <param name="robot_ip">192.168.1.10</param>
    </hardware>
    <joint name="panda_joint1">
      <command_interface name="position"/>
      <command_interface name="velocity"/>
      <command_interface name="effort"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>
    <!-- ... J2-J7 ... -->
    <joint name="panda_finger_joint1">
      <command_interface name="position"/>
      <state_interface name="position"/>
    </joint>
    <joint name="panda_finger_joint2">
      <command_interface name="position"/>
      <state_interface name="position"/>
    </joint>
  </ros2_control>
</robot>
```

### 1.3 조인트 제한 요약

| Joint | lower (rad) | upper (rad) | max_velocity (rad/s) | max_effort (Nm) |
|-------|------------|------------|---------------------|----------------|
| J1 | -2.897 | 2.897 | 2.175 | 87.0 |
| J2 | -1.762 | 1.762 | 2.175 | 87.0 |
| J3 | -2.897 | 2.897 | 2.175 | 87.0 |
| J4 | -3.071 | 3.071 | 2.175 | 87.0 |
| J5 | -2.897 | 2.897 | 2.610 | 12.0 |
| J6 | -0.017 | 3.752 | 2.610 | 12.0 |
| J7 | -2.897 | 2.897 | 2.610 | 12.0 |

---

## 2. UR5e URDF (Follower Arm)

### 2.1 Kinematics Chain

```
Base ──> J1 (rev, base_rotate) ──> J2 (rev, shoulder) ──> J3 (rev, elbow)
    │                                                          │
    └──────────────────────────────────────────────────────────┘
    │                                                          │
    Link 3 ──> J4 (rev, wrist1) ──> Link 4 ──> J5 (rev, wrist2) ──> Link 5 ──> J6 (rev, wrist3) ──> Flange
```

**DH Parameters (UR5e)**

| Joint | a (m) | α (rad) | d (m) | θ (rad) | Joint Range |
|-------|-------|---------|-------|---------|-------------|
| J1 | 0 | π/2 | 0.163 | θ1 | -360° ~ +360° |
| J2 | -0.425 | 0 | 0 | θ2 | -360° ~ +360° |
| J3 | -0.392 | 0 | 0 | θ3 | -360° ~ +360° |
| J4 | 0 | π/2 | 0.133 | θ4 | -360° ~ +360° |
| J5 | 0 | -π/2 | 0.100 | θ5 | -360° ~ +360° |
| J6 | 0 | 0 | 0.100 | θ6 | -360° ~ +360° |

### 2.2 UR5e URDF 핵심

```xml
<?xml version="1.0"?>
<robot name="ur5e" xmlns:xacro="http://ros.org/wiki/xacro">
  <!-- Base -->
  <link name="base_link">
    <visual>
      <geometry>
        <mesh filename="package://ur_description/meshes/ur5e/visual/base.dae"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://ur_description/meshes/ur5e/collision/base.stl"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint 1: Base → Shoulder -->
  <joint name="shoulder_pan_joint" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.163" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.283" upper="6.283" effort="150.0" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <!-- J2-J5 ... -->

  <!-- Joint 6: Forearm → Flange -->
  <joint name="wrist_3_joint" type="revolute">
    <parent link="wrist_2_link"/>
    <child link="wrist_3_link"/>
    <origin xyz="0 0 0.100" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.283" upper="6.283" effort="20.0" velocity="2.0"/>
  </joint>

  <!-- Tool0 (기준 좌표계) -->
  <link name="tool0"/>
  <joint name="wrist_3_joint-tool0_fixed" type="fixed">
    <parent link="wrist_3_link"/>
    <child link="tool0"/>
    <origin xyz="0 0 0.100" rpy="0 0 0"/>
  </joint>

  <!-- Gripper Attachment -->
  <link name="gripper_base"/>
  <joint name="gripper_fixed_joint" type="fixed">
    <parent link="tool0"/>
    <child link="gripper_base"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <!-- ... gripper fingers ... -->

  <!-- ROS2 Control -->
  <ros2_control name="UR5e" type="system">
    <hardware>
      <plugin>ur_ros2_driver/URPositionHardwareInterface</plugin>
      <param name="robot_ip">192.168.1.20</param>
    </hardware>
    <joint name="shoulder_pan_joint">
      <command_interface name="velocity"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>
    <!-- ... -->
  </ros2_control>
</robot>
```

---

## 3. Isaac Sim USD 변환

### 3.1 URDF → USD 변환

```python
# scripts/convert_urdf_to_usd.py
"""
Franka Panda + UR5e URDF를 Isaac Sim USD로 변환
"""
from omni.isaac.core.utils.stage import create_new_stage_async
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import enable_extension
from pxr import Usd, UsdGeom, Gf
import urdf_parser_py.urdf as urdf_parser
import os

def convert_arm_urdf_to_usd(
    urdf_path: str,
    usd_output_path: str,
    robot_name: str = "robot_arm"
):
    """
    URDF 파일을 Isaac Sim USD로 변환
    
    Args:
        urdf_path: 입력 URDF 파일 경로
        usd_output_path: 출력 USD 파일 경로
        robot_name: USD Prim 이름
    """
    # Isaac Sim USD 리더로 변환
    from omni.isaac.urdf import _urdf
    urdf_extension = _urdf.acquire_urdf_interface()
    
    # 변환 설정
    import json
    import_config = json.dumps({
        "overwrite": True,
        "import_scale": 1.0,
        "distance_scale": 1.0,
        "default_drive_type": "velocity",
        "default_drive_stiffness": 0.0,
        "default_drive_damping": 0.1,
        "fix_base": True,
        "make_default_prim": True,
        "merge_fixed_joints": False,
        "self_collision": False,
        "create_physics_articulation": True,
    })
    
    # 변환 실행
    success, prim_path = urdf_extension.import_urdf(
        urdf_path=urdf_path,
        import_config=import_config
    )
    
    if success:
        # USD 저장
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        stage.Export(usd_output_path)
        print(f"Saved USD: {usd_output_path}")
    else:
        print(f"Failed to convert {urdf_path}")
    
    return success


# 변환 실행
if __name__ == "__main__":
    # Franka Panda 변환
    convert_arm_urdf_to_usd(
        urdf_path="urdf/franka_panda.urdf",
        usd_output_path="urdf/franka_panda.usd",
        robot_name="franka_panda"
    )
    
    # UR5e 변환
    convert_arm_urdf_to_usd(
        urdf_path="urdf/ur5e.urdf",
        usd_output_path="urdf/ur5e.usd",
        robot_name="ur5e"
    )
```

### 3.2 물리 검증

```python
# scripts/validate_arm_physics.py
"""
변환된 USD의 물리 설정 검증
"""
from omni.isaac.core import SimulationContext
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdPhysics, Gf

def validate_articulation(usd_path: str, robot_name: str):
    """Articulation의 물리 설정 검증"""
    simulation_context = SimulationContext()
    
    # USD 로드
    stage = simulation_context.open_stage(usd_path)
    
    # Articulation 확인
    prim = get_prim_at_path(f"/{robot_name}")
    articulation = UsdPhysics.Articulation(prim)
    
    # 조인트 드라이브 확인
    for joint_prim in prim.GetAllChildren():
        if joint_prim.HasAPI(UsdPhysics.Joint):
            drive = UsdPhysics.DriveAPI.Get(joint_prim, "velocity")
            if drive:
                stiffness = drive.GetStiffnessAttr().Get()
                damping = drive.GetDampingAttr().Get()
                print(f"  {joint_prim.GetName()}: stiffness={stiffness}, damping={damping}")
    
    print(f"✓ {robot_name} articulation valid")
    return True
```

### 3.3 USD 계층 구조 (Franka Panda)

```
/franka_panda (Xform, Articulation)
├── panda_link0 (Mesh, CollisionMesh)
├── panda_joint1 (RevoluteJoint)
│   └── panda_link1 (Mesh, CollisionMesh)
├── panda_joint2 (RevoluteJoint)
│   └── panda_link2 (Mesh, CollisionMesh)
├── ...
├── panda_joint7 (FixedJoint)
│   └── panda_hand (Mesh)
│       ├── panda_finger_joint1 (PrismaticJoint)
│       │   └── panda_left_finger (Mesh)
│       └── panda_finger_joint2 (PrismaticJoint)
│           └── panda_right_finger (Mesh)
│       └── camera_joint (FixedJoint)
│           └── d435_camera (Xform)
│               ├── rgb (Camera)
│               └── depth (Camera)
```

---

## 4. ROS2 Control 통합

### 4.1 Controller 설정

```yaml
# config/leader_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz
    use_sim_time: true

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    franka_hand_controller:
      type: position_controllers/GripperActionController

joint_trajectory_controller:
  ros__parameters:
    joints:
      - panda_joint1
      - panda_joint2
      - panda_joint3
      - panda_joint4
      - panda_joint5
      - panda_joint6
      - panda_joint7
    command_interfaces:
      - velocity
    state_interfaces:
      - position
      - velocity
      - effort

velocity_controller:
  ros__parameters:
    joints:
      - panda_joint1
      - panda_joint2
      - panda_joint3
      - panda_joint4
      - panda_joint5
      - panda_joint6
      - panda_joint7
    command_interfaces:
      - velocity
    state_interfaces:
      - position
      - velocity
      - effort

franka_hand_controller:
  ros__parameters:
    joints:
      - panda_finger_joint1
      - panda_finger_joint2
```

### 4.2 MoveIt2 설정

```yaml
# config/moveit_params.yaml
moveit:
  ros__parameters:
    robot_description: "franka_panda"
    robot_description_semantic: "franka_panda_semantic"

    # IK Solver
    kinematics_solver: "kdl_kinematics_plugin/KDLKinematicsPlugin"
    kinematics_solver_search_resolution: 0.005
    kinematics_solver_timeout: 0.05
    kinematics_solver_attempts: 3

    # Planning Pipeline
    planning_pipelines:
      - "ompl"
      - "chomp"
      - "pilz_industrial_motion_planner"

    # OMPL Config
    ompl:
      planner_configs:
        - "RRTConnect"
        - "RRTstar"
        - "PRMstar"
      RRTConnect:
        type: "geometric::RRTConnect"
        range: 0.5
      RRTstar:
        type: "geometric::RRTstar"
        range: 0.5
      PRMstar:
        type: "geometric::PRMstar"
        range: 0.5

    # Collision Checking
    collision_checking:
      collision_activation_distance: 0.1
      collision_checking_type: "geometric"
      padding: 0.01
      scale: 1.0

    # Allowed Collision Matrix
    default_allow_collision: false

    # Servo Parameters (real-time Cartesian control)
    servo:
      command_in_type: "velocity_commands"
      command_out_type: "trajectory_msgs/JointTrajectory"
      scale:
        linear: 0.3
        rotational: 0.3
        joint: 1.0
      collision_check_rate: 10.0
      self_collision_proximity_threshold: 0.01
      scene_collision_proximity_threshold: 0.02
      lower_singularity_threshold: 0.01
      hard_stop_singularity_threshold: 0.001
      low_pass_filter_coeff: 0.5
```

---

## 5. Camera Attachment

### 5.1 Franka Flange → RealSense D435

```python
# scripts/attach_camera_to_arm.py
def attach_realsense_to_flange(
    stage,
    arm_prim_path: str,
    camera_name: str = "d435_camera"
):
    """
    RealSense D435 카메라를 Arm Flange에 부착
    
    Args:
        stage: USD Stage
        arm_prim_path: Arm prim path (예: /franka_panda/panda_hand)
        camera_name: Camera prim 이름
    """
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.prims import define_prim
    
    # Camera prim 생성
    camera_path = f"{arm_prim_path}/{camera_name}"
    camera_prim = define_prim(camera_path, "Xform")
    
    # RGB Camera
    rgb_camera = Camera(
        prim_path=f"{camera_path}/rgb",
        translation=(0.05, 0.0, 0.03),  # flange 기준 offset
        frequency=30,
        resolution=(640, 480),
        orientation=(0.5, -0.5, 0.5, -0.5),  # look-at 방향
    )
    rgb_camera.initialize()
    
    # Depth Camera
    depth_camera = Camera(
        prim_path=f"{camera_path}/depth",
        translation=(0.05, 0.0, 0.03),
        frequency=30,
        resolution=(640, 480),
        orientation=(0.5, -0.5, 0.5, -0.5),
    )
    depth_camera.initialize()
    depth_camera.set_clipping_range(0.1, 3.0)  # 10cm ~ 3m
    
    return rgb_camera, depth_camera
```

---

## 6. Bimanual Scene URDF

### 6.1 공동 작업 공간

두 Arm을 하나의 scene에 배치할 때:

```
Leader Arm (Franka Panda)          Follower Arm (UR5e)
         │                                │
         │    ┌──────────────────┐        │
         │    │  Shared Workspace │        │
         │    │                  │        │
         │    │   ██  ██  ██    │        │
         │    │   (objects)     │        │
         │    └──────────────────┘        │
         │                                │
    x=0.3m, z=0m                   x=-0.3m, z=0m
```

```xml
<!-- bimanual_scene.sdf / urdf -->
<robot name="bimanual_scene">
  <!-- Leader Arm -->
  <joint name="leader_base_joint" type="fixed">
    <parent link="world"/>
    <child link="franka_base"/>
    <origin xyz="0.3 0 0" rpy="0 0 3.14159"/>
  </joint>
  
  <!-- Follower Arm -->
  <joint name="follower_base_joint" type="fixed">
    <parent link="world"/>
    <child link="ur5e_base"/>
    <origin xyz="-0.3 0 0" rpy="0 0 0"/>
  </joint>
  
  <!-- Table -->
  <link name="table">
    <visual>
      <geometry><box size="0.8 0.6 0.02"/></geometry>
      <material name="wood"/>
    </visual>
    <collision>
      <geometry><box size="0.8 0.6 0.02"/></geometry>
    </collision>
    <origin xyz="0 0 0.7"/> <!-- height 0.7m -->
  </link>
  <joint name="table_joint" type="fixed">
    <parent link="world"/>
    <child link="table"/>
    <origin xyz="0 0 0"/>
  </joint>
</robot>
```
