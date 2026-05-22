# Isaac Sim Bimanual Arm Environment

> Franka Panda (Leader) + UR5e (Follower) 양팔 시뮬레이션 환경 구성

---

## 1. Scene 구성

### 1.1 양팔 Stage 생성

```python
# src/isaac_sim/setup_bimanual_scene.py
"""
Isaac Sim 내 Leader-Follower 양팔 환경 구성
"""
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import create_new_stage_async, open_stage
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.writer import Writer
import numpy as np


class BimanualScene:
    """
    Leader-Follower 양팔 환경 클래스
    """
    
    def __init__(
        self,
        franka_usd_path: str = "urdf/franka_panda.usd",
        ur5e_usd_path: str = "urdf/ur5e.usd",
        headless: bool = False
    ):
        self.franka_usd = franka_usd_path
        self.ur5e_usd = ur5e_usd_path
        self.headless = headless
        self.simulation_context = None
        self.leader_arm = None  # Franka Panda
        self.follower_arm = None  # UR5e
        self.cameras = {}
        
    def setup_scene(self):
        """전체 scene 구성"""
        # Simulation context 생성
        self.simulation_context = SimulationContext(
            physics_dt=1.0 / 500.0,  # 500Hz physics
            rendering_dt=1.0 / 30.0,  # 30FPS rendering
            stage_units_in_meters=1.0,
        )
        
        # Leader Arm (Franka Panda) 로드
        self._load_leader_arm()
        
        # Follower Arm (UR5e) 로드
        self._load_follower_arm()
        
        # 테이블 생성
        self._create_table()
        
        # 바닥면 생성
        self._create_ground_plane()
        
        # 조명 설정
        self._setup_lighting()
        
        # 카메라 부착
        self._attach_cameras()
        
        # 카메라 뷰 설정
        set_camera_view(
            eye=[0.8, -1.2, 1.0],
            target=[0.0, 0.0, 0.3],
            camera_prim_path="/OmniverseKit_Persp"
        )
        
        print("✅ Bimanual scene ready")
        print(f"  Leader:   Franka Panda @ x=0.30")
        print(f"  Follower: UR5e         @ x=-0.30")
        print(f"  Cameras:  {len(self.cameras)} total")
        
    def _load_leader_arm(self):
        """Leader Arm (Franka Panda) 로드"""
        from omni.isaac.core.utils.prims import create_prim
        
        # USD 참조로 로드
        leader_prim = create_prim(
            prim_path="/World/LeaderArm",
            prim_type="Xform",
            translation=(0.30, 0.0, 0.0),  # 테이블 우측
            orientation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
        )
        
        # USD 레퍼런스 추가
        leader_prim.GetReferences().AddReference(
            assetPath=self.franka_usd
        )
        
        # Articulation 설정
        self.leader_arm = Articulation("/World/LeaderArm/franka_panda")
        self.leader_arm.initialize()
        self.leader_arm.set_joint_drive_type("velocity")
        
        # Franka Panda joint drive parameters
        drive_params = {
            "panda_joint1": {"stiffness": 0.0, "damping": 0.05},
            "panda_joint2": {"stiffness": 0.0, "damping": 0.05},
            "panda_joint3": {"stiffness": 0.0, "damping": 0.05},
            "panda_joint4": {"stiffness": 0.0, "damping": 0.05},
            "panda_joint5": {"stiffness": 0.0, "damping": 0.03},
            "panda_joint6": {"stiffness": 0.0, "damping": 0.03},
            "panda_joint7": {"stiffness": 0.0, "damping": 0.03},
        }
        for joint_name, params in drive_params.items():
            self.leader_arm.set_joint_drive_parameters(
                joint_name=joint_name,
                stiffness=params["stiffness"],
                damping=params["damping"],
            )
        
        # Home pose 설정
        self.leader_arm.set_joint_positions(
            joint_positions=np.array([
                0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785
            ]),  # Franka standard home
            joint_indices=np.arange(7)
        )
        
        print("  ✔ Leader Arm (Franka Panda) loaded")
        
    def _load_follower_arm(self):
        """Follower Arm (UR5e) 로드"""
        from omni.isaac.core.utils.prims import create_prim
        
        follower_prim = create_prim(
            prim_path="/World/FollowerArm",
            prim_type="Xform",
            translation=(-0.30, 0.0, 0.0),  # 테이블 좌측
            orientation=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
        )
        
        follower_prim.GetReferences().AddReference(
            assetPath=self.ur5e_usd
        )
        
        self.follower_arm = Articulation("/World/FollowerArm/ur5e")
        self.follower_arm.initialize()
        self.follower_arm.set_joint_drive_type("velocity")
        
        # UR5e joint drive (velocity control)
        ur5e_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint",
            "elbow_joint", "wrist_1_joint",
            "wrist_2_joint", "wrist_3_joint"
        ]
        for joint_name in ur5e_joint_names:
            self.follower_arm.set_joint_drive_parameters(
                joint_name=joint_name,
                stiffness=0.0,
                damping=0.1,
            )
        
        # Home pose
        self.follower_arm.set_joint_positions(
            joint_positions=np.array([
                0.0, -1.571, 1.571, -1.571, -1.571, 0.0
            ]),  # UR5e standard home
            joint_indices=np.arange(6)
        )
        
        print("  ✔ Follower Arm (UR5e) loaded")
        
    def _create_table(self):
        """공동 작업 테이블 생성"""
        from omni.isaac.core.utils.prims import create_prim
        
        table_prim = create_prim(
            prim_path="/World/Table",
            prim_type="Cube",
            translation=(0.0, 0.0, 0.70),  # 높이 0.7m
            scale=(0.80, 0.60, 0.02),       # 80cm × 60cm × 2cm
        )
        
        # 물리 속성
        from pxr import UsdPhysics, PhysxSchema
        UsdPhysics.CollisionAPI.Apply(table_prim)
        PhysxSchema.PhysxCollisionAPI.Apply(table_prim)
        
        # Table mass
        mass_api = UsdPhysics.MassAPI.Apply(table_prim)
        mass_api.GetMassAttr().Set(10.0)
        
        # 마찰 계수
        physx_collision = PhysxSchema.PhysxCollisionAPI(table_prim)
        physx_collision.GetRestOffsetAttr().Set(0.001)
        
        print("  ✔ Table created (0.8m × 0.6m × 0.02m @ z=0.7)")
        
    def _create_ground_plane(self):
        """바닥면 생성"""
        from omni.isaac.core.utils.prims import create_prim
        
        ground_prim = create_prim(
            prim_path="/World/GroundPlane",
            prim_type="Plane",
            translation=(0.0, 0.0, 0.0),
            scale=(10.0, 10.0, 1.0),
        )
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI.Apply(ground_prim)
        
    def _setup_lighting(self):
        """조명 설정 (CosmosWriter 데이터용)"""
        stage = omni.usd.get_context().get_stage()
        
        # Key light (주광)
        key_light = UsdGeom.DistantLight.Define(stage, "/World/Lights/KeyLight")
        key_light.CreateIntensityAttr(500.0)
        key_light.CreateAngleAttr(0.53)
        key_light.AddRotateXYZOp().Set((45.0, 30.0, 0.0))
        
        # Fill light (보조광)
        fill_light = UsdGeom.DistantLight.Define(stage, "/World/Lights/FillLight")
        fill_light.CreateIntensityAttr(200.0)
        fill_light.AddRotateXYZOp().Set((-30.0, -45.0, 0.0))
        
        # 환경 조명
        dome_light = UsdGeom.DomeLight.Define(stage, "/World/Lights/DomeLight")
        dome_light.CreateIntensityAttr(100.0)
        
        print("  ✔ Lighting configured")
        
    def _attach_cameras(self):
        """각 Arm의 Flange에 RGB-D 카메라 부착"""
        # Leader arm camera
        leader_rgb, leader_depth = self._create_camera(
            camera_path="/World/LeaderArm/franka_panda/panda_hand/camera",
            translation=(0.05, 0.0, 0.03),
            name="leader"
        )
        self.cameras["leader_rgb"] = leader_rgb
        self.cameras["leader_depth"] = leader_depth
        
        # Follower arm camera
        follower_rgb, follower_depth = self._create_camera(
            camera_path="/World/FollowerArm/ur5e/wrist_3_link/camera",
            translation=(0.05, 0.0, 0.03),
            name="follower"
        )
        self.cameras["follower_rgb"] = follower_rgb
        self.cameras["follower_depth"] = follower_depth
        
        # Overhead camera (3인칭)
        overhead_rgb, overhead_depth = self._create_camera(
            camera_path="/World/OverheadCamera",
            translation=(0.0, 0.0, 1.5),
            name="overhead",
            look_at=(0.0, 0.0, 0.3)
        )
        self.cameras["overhead_rgb"] = overhead_rgb
        self.cameras["overhead_depth"] = overhead_depth
        
    def _create_camera(self, camera_path, translation, name="camera", look_at=None):
        """RGB-D 카메라 생성"""
        from omni.isaac.sensor import Camera
        
        # RGB Camera
        rgb = Camera(
            prim_path=f"{camera_path}_rgb",
            translation=translation,
            frequency=30,
            resolution=(640, 480),
            orientation=(0.5, -0.5, 0.5, -0.5),  # forward-down
        )
        rgb.initialize()
        rgb.set_focal_length(1.93)  # RealSense D435 equivalent
        rgb.set_horizontal_aperture(3.9)
        rgb.set_vertical_aperture(2.9)
        
        # Depth Camera
        depth = Camera(
            prim_path=f"{camera_path}_depth",
            translation=translation,
            frequency=30,
            resolution=(640, 480),
            orientation=(0.5, -0.5, 0.5, -0.5),
        )
        depth.initialize()
        depth.set_clipping_range(0.1, 3.0)
        
        # Depth noise model (sim-to-real)
        depth.enable_noise_model(True)
        depth.set_noise_parameters(
            noise_mean=0.0,
            noise_std=0.01,
            quantization_bits=16,
        )
        
        return rgb, depth
```

### 1.2 실행

```python
# main.py (Isaac Sim 내에서 실행)
from src.isaac_sim.setup_bimanual_scene import BimanualScene

scene = BimanualScene()
scene.setup_scene()

# Physics step loop
while True:
    scene.simulation_context.step(render=True)
    
    # Get joint states
    leader_joints = scene.leader_arm.get_joint_positions()
    follower_joints = scene.follower_arm.get_joint_positions()
    
    # Get camera data
    rgb_image = scene.cameras["leader_rgb"].get_rgb()
    depth_image = scene.cameras["leader_depth"].get_depth()
```

---

## 2. CosmosWriter — 합성 데이터 생성

### 2.1 Writer 설정

```python
# src/isaac_sim/cosmos_sdg_pipeline.py
"""
CosmosWriter를 이용한 합성 데이터 생성 파이프라인
"""
import numpy as np
import json
import os
from datetime import datetime
from omni.isaac.writer import Writer, WriterAction, WriterTrigger
from pxr import Usd, UsdGeom, Gf


class ArmTrajectoryWriter(Writer):
    """
    Arm 궤적 + 센서 데이터를 저장하는 CosmosWriter
    
    저장 형식:
    - RGB:      .png (640×480)
    - Depth:    .npy (640×480, float32)
    - Joint:    .json (조인트 각도 + 속도)
    - Gripper:  .json (개폐 상태)
    - EE Pose:  .json (End-effector 위치/회전)
    - Language: .txt (자연어 명령)
    - Metadata: .json (에피소드 정보)
    """
    
    def __init__(
        self,
        output_dir: str = "data/cosmos_sdg",
        episode_name: str = None,
        cameras: list = None,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.episode_name = episode_name or f"ep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.cameras = cameras or []
        self.timestep = 0
        self.episode_data = {
            "metadata": {
                "episode_id": self.episode_name,
                "timestamp": datetime.now().isoformat(),
                "leader_arm": "franka_panda",
                "follower_arm": "ur5e",
                "camera_count": len(self.cameras),
                "total_timesteps": 0,
            },
            "frames": [],
        }
        
        # 출력 디렉토리 생성
        os.makedirs(f"{self.output_dir}/{self.episode_name}", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.episode_name}/rgb", exist_ok=True)
        os.makedirs(f"{self.output_dir}/{self.episode_name}/depth", exist_ok=True)
    
    def on_timestep(self, 
                    leader_joints: np.ndarray,
                    follower_joints: np.ndarray,
                    leader_gripper: float,
                    follower_gripper: float,
                    leader_ee_pose: tuple,
                    follower_ee_pose: tuple,
                    language_instruction: str = "",
                    success: bool = False):
        """
        각 time step에서 호출되는 콜백
        
        Args:
            leader_joints: (7,) Franka joint angles
            follower_joints: (6,) UR5e joint angles  
            leader_gripper: float [0,1] Franka gripper
            follower_gripper: float [0,1] UR5e gripper
            leader_ee_pose: (x,y,z,qw,qx,qy,qz)
            follower_ee_pose: (x,y,z,qw,qx,qy,qz)
            language_instruction: natural language command
            success: task success flag
        """
        frame = {
            "t": self.timestep,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 카메라 데이터 저장
        for cam_name, cam_data in self.cameras.items():
            rgb = cam_data["rgb"].get_rgb()
            depth = cam_data["depth"].get_depth()
            
            # RGB 저장
            import cv2
            rgb_path = f"{self.output_dir}/{self.episode_name}/rgb/{cam_name}_{self.timestep:06d}.png"
            cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Depth 저장
            depth_path = f"{self.output_dir}/{self.episode_name}/depth/{cam_name}_{self.timestep:06d}.npy"
            np.save(depth_path, depth)
            
            frame[f"{cam_name}_rgb"] = rgb_path
            frame[f"{cam_name}_depth"] = depth_path
        
        # Joint states
        frame["leader_joints"] = leader_joints.tolist()
        frame["follower_joints"] = follower_joints.tolist()
        frame["leader_gripper"] = leader_gripper
        frame["follower_gripper"] = follower_gripper
        
        # EE Pose
        frame["leader_ee_pose"] = {
            "position": leader_ee_pose[:3],
            "orientation": leader_ee_pose[3:]
        }
        frame["follower_ee_pose"] = {
            "position": follower_ee_pose[:3],
            "orientation": follower_ee_pose[3:]
        }
        
        # Language
        frame["instruction"] = language_instruction
        frame["success"] = success
        
        self.episode_data["frames"].append(frame)
        self.timestep += 1
        return True
    
    def save_episode(self):
        """에피소드 메타데이터 저장"""
        self.episode_data["metadata"]["total_timesteps"] = self.timestep
        
        # JSON 저장
        json_path = f"{self.output_dir}/{self.episode_name}/metadata.json"
        with open(json_path, "w") as f:
            json.dump(self.episode_data, f, indent=2)
        
        # Instruction 저장
        inst_path = f"{self.output_dir}/{self.episode_name}/instruction.txt"
        first_frame = self.episode_data["frames"][0] if self.episode_data["frames"] else {}
        with open(inst_path, "w") as f:
            f.write(first_frame.get("instruction", ""))
        
        print(f"✅ Episode saved: {self.episode_name}")
        print(f"   Frames: {self.timestep}")
        print(f"   Path:   {self.output_dir}/{self.episode_name}")
        
        return self.episode_name
```

### 2.2 SDG 실행 파이프라인

```python
# sdg_pipeline.py — 합성 데이터 생성 파이프라인
import argparse
from src.isaac_sim.setup_bimanual_scene import BimanualScene
from src.isaac_sim.cosmos_sdg_pipeline import ArmTrajectoryWriter


def run_sdg_pipeline(
    num_episodes: int = 10,
    frames_per_episode: int = 150,  # 5초 @ 30fps
    output_dir: str = "data/cosmos_sdg",
    domain_randomize: bool = True,
):
    """
    합성 데이터 생성 파이프라인
    
    1. Scene 초기화
    2. 도메인 랜덤화
    3. 에피소드별 데이터 생성
    4. CosmosWriter로 저장
    """
    # 1. Scene 초기화
    scene = BimanualScene()
    scene.setup_scene()
    
    for ep in range(num_episodes):
        # 2. 도메인 랜덤화
        if domain_randomize:
            randomize_scene(scene)
        
        # 3. CosmosWriter 초기화
        writer = ArmTrajectoryWriter(
            output_dir=output_dir,
            episode_name=f"ep_{ep:04d}",
            cameras=scene.cameras,
        )
        
        # 4. 랜덤 태스크 생성
        task = generate_random_task()
        language_instruction = task["instruction"]
        
        # 5. 에피소드 실행
        for t in range(frames_per_episode):
            # Leader Arm 랜덤 궤적
            leader_joints = generate_smooth_trajectory(t, frames_per_episode)
            
            # Follow Arm은 Leader 궤적 + 노이즈 (초기에는 모방)
            follower_joints = leader_joints[:6] + np.random.normal(0, 0.01, 6)
            
            # 그리퍼 상태
            leader_gripper = 1.0 if t > frames_per_episode * 0.6 else 0.0
            follower_gripper = 1.0 if t > frames_per_episode * 0.7 else 0.0
            
            # Arm에 joint 명령 적용
            scene.leader_arm.set_joint_velocity_targets(leader_joints[:7])
            scene.follower_arm.set_joint_velocity_targets(follower_joints[:6])
            scene.simulation_context.step(render=True)
            
            # 데이터 수집
            writer.on_timestep(
                leader_joints=leader_joints,
                follower_joints=follower_joints,
                leader_gripper=leader_gripper,
                follower_gripper=follower_gripper,
                leader_ee_pose=compute_ee_pose(scene.leader_arm),
                follower_ee_pose=compute_ee_pose(scene.follower_arm),
                language_instruction=language_instruction,
                success=(t == frames_per_episode - 1),
            )
        
        # 6. 에피소드 저장
        writer.save_episode()
    
    print(f"✅ SDG pipeline complete: {num_episodes} episodes")


def randomize_scene(scene):
    """도메인 랜덤화 적용"""
    import random
    
    # 조명 랜덤화
    light_stage = omni.usd.get_context().get_stage()
    key_light = light_stage.GetPrimAtPath("/World/Lights/KeyLight")
    key_light.GetAttribute("intensity").Set(random.uniform(300, 800))
    
    # 물체 위치 랜덤화
    # (테이블 위 블록 위치)
    
    # Arm 초기 상태 랜덤화
    leader_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    noise = np.random.uniform(-0.1, 0.1, 7)
    scene.leader_arm.set_joint_positions(leader_home + noise)
    
    return scene


def generate_random_task():
    """랜덤 태스크 생성"""
    tasks = [
        {"instruction": "Pick up the blue block and place it on the red box", "type": "pick_and_place"},
        {"instruction": "Move the green cylinder to the left side", "type": "move"},
        {"instruction": "Stack the red block on top of the blue block", "type": "stack"},
        {"instruction": "Push the object forward 10cm", "type": "push"},
        {"instruction": "Grasp the object and lift it 20cm", "type": "lift"},
    ]
    import random
    return random.choice(tasks)


def generate_smooth_trajectory(t, total_frames):
    """부드러운 궤적 생성 (sinusoidal interpolation)"""
    import numpy as np
    alpha = t / total_frames
    # Franka home → target
    start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    target = np.array([0.5, -0.5, 0.3, -2.0, 0.2, 1.2, 0.5])
    
    # Smooth step (sine interpolation)
    smooth_alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)
    trajectory = start + (target - start) * smooth_alpha
    
    return trajectory


def compute_ee_pose(arm):
    """End-effector forward kinematics"""
    from omni.isaac.core.utils.rotations import quat_from_euler
    # IK solver or direct FK from simulator
    # 실제로는 Isaac Sim이 직접 EE pose를 제공
    return (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
```

---

## 3. 도메인 랜덤화

### 3.1 랜덤화 파라미터

| 파라미터 | 범위 | 적용 대상 | 영향 |
|---------|------|---------|------|
| 조명 방향 | -45° ~ +45° | Key Light | 그림자 변화 |
| 조명 강도 | 300 ~ 800 lux | All Lights | 노출 변화 |
| 테이블 마찰 | 0.2 ~ 1.0 | Table | 슬라이딩 |
| 조인트 댐핑 | 0.01 ~ 0.2 | Arm Joints | 관성 변화 |
| 페이로드 질량 | 0.0 ~ 0.5 kg | Gripper | 중량 변화 |
| 카메라 노이즈 | σ=0.001 ~ 0.03 | Depth Camera | 센서 노이즈 |
| 물체 위치 | ±5cm | Table Objects | 위치 변화 |
| 물체 색상 | RGB 변수 | Objects | 시각 다양성 |
| 배경 텍스처 | 5종 | Ground | 배경 변화 |

### 3.2 랜덤화 구현

```python
# src/isaac_sim/domain_randomize.py
class DomainRandomizer:
    """
    Isaac Sim 도메인 랜덤화
    - 조명, 물리, 시각 파라미터 무작위 변경
    - 각 에피소드 시작 시 호출
    """
    
    def __init__(self, scene):
        self.scene = scene
        self.stage = omni.usd.get_context().get_stage()
        
    def randomize_all(self):
        """모든 도메인 파라미터 랜덤화"""
        self.randomize_lighting()
        self.randomize_physics()
        self.randomize_appearance()
        self.randomize_camera()
        self.randomize_objects()
        
    def randomize_lighting(self):
        """조명 랜덤화"""
        import random
        
        # Key light
        key = self.stage.GetPrimAtPath("/World/Lights/KeyLight")
        if key:
            key.GetAttribute("intensity").Set(random.uniform(300, 800))
            angle = random.uniform(-45, 45)
            key.GetAttribute("rotateX").Set(angle)
            key.GetAttribute("rotateZ").Set(random.uniform(-30, 30))
        
        # Fill light
        fill = self.stage.GetPrimAtPath("/World/Lights/FillLight")
        if fill:
            fill.GetAttribute("intensity").Set(random.uniform(100, 300))
    
    def randomize_physics(self):
        """물리 파라미터 랜덤화"""
        import random
        
        # Table friction
        table = self.stage.GetPrimAtPath("/World/Table")
        if table:
            friction = random.uniform(0.2, 1.0)
            # Set physics friction
            from pxr import PhysxSchema
            physx = PhysxSchema.PhysxCollisionAPI(table)
            physx.GetDynamicFrictionAttr().Set(friction)
            physx.GetStaticFrictionAttr().Set(friction * 1.2)
    
    def randomize_appearance(self):
        """시각적 외형 랜덤화"""
        import random
        
        # 배경색
        dome = self.stage.GetPrimAtPath("/World/Lights/DomeLight")
        if dome:
            color = (random.random(), random.random(), random.random())
            dome.GetAttribute("color").Set(Gf.Vec3f(*color))
    
    def randomize_camera(self):
        """카메라 노이즈 랜덤화"""
        import random
        
        for name, cam in self.scene.cameras.items():
            if "depth" in name:
                noise_std = random.uniform(0.001, 0.03)
                cam.set_noise_parameters(
                    noise_mean=0.0,
                    noise_std=noise_std,
                    quantization_bits=16,
                )
    
    def randomize_objects(self):
        """테이블 위 물체 위치 랜덤화"""
        import random
        
        for obj_path in ["/World/Objects/blue_block", "/World/Objects/red_box"]:
            obj = self.stage.GetPrimAtPath(obj_path)
            if obj:
                x = random.uniform(-0.15, 0.15)
                y = random.uniform(-0.15, 0.15)
                obj.GetAttribute("xformOp:translate").Set(
                    Gf.Vec3d(x, y, 0.71)
                )
```

---

## 4. Teleop 입력 (Leader Arm 제어)

### 4.1 조이스틱 Teleop

```python
# src/leader_arm/teleop_joy.py
"""
ROS2 조이스틱으로 Leader Arm 원격 제어
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
import numpy as np


class ArmJoyTeleop(Node):
    """
    조이스틱 → Leader Arm joint velocity 변환
    
    매핑 (Xbox Controller):
    - Left Stick Y:     J1 (waist rotation)
    - Left Stick X:     J2 (shoulder)
    - Right Stick Y:    J3 (elbow)
    - Right Stick X:    J4 (forearm)
    - LB/RB:            J5 (wrist)
    - LT/RT:            J6 (wrist)
    - D-pad Up/Down:    J7 (flange)
    - A Button:         Gripper close
    - B Button:         Gripper open
    - Start:            Record start/stop
    - Back:             Home pose
    """
    
    def __init__(self):
        super().__init__('arm_joy_teleop')
        
        # Publisher: joint velocity commands
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/leader/joint_velocity_commands',
            10
        )
        
        # Publisher: gripper command
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/leader/gripper_command',
            10
        )
        
        # Subscriber: joystick
        self.sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        
        # Velocity scaling
        self.vel_scale = 0.5  # rad/s per joystick unit
        
        # Dead zone
        self.deadzone = 0.1
        
        self.get_logger().info("Arm Joy Teleop started")
        
    def joy_callback(self, msg: Joy):
        """조이스틱 입력 → 속도 명령"""
        axes = msg.axes
        buttons = msg.buttons
        
        # Dead zone 적용
        axes = [0.0 if abs(a) < self.deadzone else a for a in axes]
        
        # Joint velocity command (7-DOF)
        vel_cmd = Float64MultiArray()
        vel_cmd.data = [
            axes[0] * self.vel_scale,     # J1: Left Stick Y
            axes[1] * self.vel_scale,     # J2: Left Stick X
            axes[3] * self.vel_scale,     # J3: Right Stick Y
            axes[2] * self.vel_scale,     # J4: Right Stick X
            (buttons[4] - buttons[5]) * self.vel_scale,  # J5: LB-RB
            (buttons[2] - buttons[3]) * self.vel_scale,  # J6: LT-RT
            (axes[7]) * self.vel_scale,   # J7: D-pad
        ]
        self.vel_pub.publish(vel_cmd)
        
        # Gripper command
        gripper_cmd = Float64MultiArray()
        if buttons[0]:  # A button = close
            gripper_cmd.data = [1.0]
        elif buttons[1]:  # B button = open
            gripper_cmd.data = [0.0]
        else:
            return  # no change
        self.gripper_pub.publish(gripper_cmd)
```

### 4.2 궤적 녹화

```python
# src/leader_arm/trajectory_recorder.py
"""
Leader Arm 궤적 녹화 및 저장
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import json
import os
from datetime import datetime


class TrajectoryRecorder(Node):
    """
    Leader Arm의 조인트 궤적을 녹화하여 파일로 저장
    
    녹화 시작/중지: Start 버튼 (토글)
    저장 형식: trajectory_{timestamp}.json
    """
    
    def __init__(self):
        super().__init__('trajectory_recorder')
        
        # Subscriber
        self.joint_sub = self.create_subscription(
            JointState,
            '/leader/joint_states',
            self.joint_callback,
            10
        )
        
        # 녹화 상태
        self.recording = False
        self.trajectory = []
        
        # 출력 디렉토리
        self.output_dir = os.path.expanduser("~/trajectories")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.get_logger().info("Trajectory Recorder ready")
        
    def joint_callback(self, msg: JointState):
        """Joint state 수신 → 녹화"""
        if not self.recording:
            return
        
        frame = {
            "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort": list(msg.effort),
        }
        self.trajectory.append(frame)
        
    def start_recording(self):
        """녹화 시작"""
        self.recording = True
        self.trajectory = []
        self.get_logger().info("⏺ Recording started")
        
    def stop_recording(self):
        """녹화 중지 및 저장"""
        self.recording = False
        
        filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_frames": len(self.trajectory),
                "duration": self.trajectory[-1]["timestamp"] - self.trajectory[0]["timestamp"] if len(self.trajectory) > 1 else 0,
                "arm": "franka_panda",
            },
            "frames": self.trajectory,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        self.get_logger().info(f"⏹ Recording saved: {filepath} ({len(self.trajectory)} frames)")
        return filepath
```
