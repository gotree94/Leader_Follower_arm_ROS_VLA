"""
Isaac Sim Bimanual Scene Setup
"""
from omni.isaac.core import SimulationContext
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.sensor import Camera
import omni.usd
import numpy as np


class BimanualScene:
    def __init__(self, franka_usd="urdf/franka_panda.usd", ur5e_usd="urdf/ur5e.usd"):
        self.franka_usd = franka_usd
        self.ur5e_usd = ur5e_usd
        self.leader_arm = None
        self.follower_arm = None
        self.cameras = {}

    def setup_scene(self):
        self.simulation_context = SimulationContext(physics_dt=1.0/500.0, rendering_dt=1.0/30.0)
        self._load_leader_arm()
        self._load_follower_arm()
        self._create_table()
        self._setup_lighting()
        self._attach_cameras()
        print("Bimanual scene ready")

    def _load_leader_arm(self):
        prim = create_prim("/World/LeaderArm", "Xform", translation=(0.30, 0.0, 0.0))
        prim.GetReferences().AddReference(assetPath=self.franka_usd)
        self.leader_arm = Articulation("/World/LeaderArm/franka_panda")
        self.leader_arm.initialize()
        self.leader_arm.set_joint_drive_type("velocity")
        self.leader_arm.set_joint_positions(np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]))

    def _load_follower_arm(self):
        prim = create_prim("/World/FollowerArm", "Xform", translation=(-0.30, 0.0, 0.0))
        prim.GetReferences().AddReference(assetPath=self.ur5e_usd)
        self.follower_arm = Articulation("/World/FollowerArm/ur5e")
        self.follower_arm.initialize()
        self.follower_arm.set_joint_drive_type("velocity")
        self.follower_arm.set_joint_positions(np.array([0.0, -1.571, 1.571, -1.571, -1.571, 0.0]))

    def _create_table(self):
        prim = create_prim("/World/Table", "Cube", translation=(0.0, 0.0, 0.70), scale=(0.80, 0.60, 0.02))
        from pxr import UsdPhysics, PhysxSchema
        UsdPhysics.CollisionAPI.Apply(prim)
        physx = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx.GetDynamicFrictionAttr().Set(0.5)

    def _setup_lighting(self):
        stage = omni.usd.get_context().get_stage()
        from pxr import UsdGeom, Gf
        key = UsdGeom.DistantLight.Define(stage, "/World/Lights/KeyLight")
        key.CreateIntensityAttr(500.0)
        key.AddRotateXYZOp().Set((45.0, 30.0, 0.0))
        fill = UsdGeom.DistantLight.Define(stage, "/World/Lights/FillLight")
        fill.CreateIntensityAttr(200.0)

    def _attach_cameras(self):
        for name, path, pos in [
            ("leader", "/World/LeaderArm/franka_panda/panda_hand", (0.05, 0.0, 0.03)),
            ("follower", "/World/FollowerArm/ur5e/wrist_3_link", (0.05, 0.0, 0.03)),
        ]:
            rgb = Camera(prim_path=f"{path}/camera_{name}_rgb", translation=pos, frequency=30, resolution=(640, 480))
            rgb.initialize()
            depth = Camera(prim_path=f"{path}/camera_{name}_depth", translation=pos, frequency=30, resolution=(640, 480))
            depth.initialize()
            depth.set_clipping_range(0.1, 3.0)
            self.cameras[f"{name}_rgb"] = rgb
            self.cameras[f"{name}_depth"] = depth

    def step(self):
        self.simulation_context.step(render=True)
