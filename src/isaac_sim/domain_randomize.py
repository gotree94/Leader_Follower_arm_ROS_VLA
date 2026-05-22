"""
Domain Randomizer for Bimanual Scene
"""
import random
import omni.usd
import numpy as np
from pxr import Gf


class DomainRandomizer:
    def __init__(self, scene):
        self.scene = scene
        self.stage = omni.usd.get_context().get_stage()

    def randomize_all(self):
        self.randomize_lighting()
        self.randomize_physics()
        self.randomize_camera()
        self.randomize_objects()

    def randomize_lighting(self):
        key = self.stage.GetPrimAtPath("/World/Lights/KeyLight")
        if key:
            key.GetAttribute("intensity").Set(random.uniform(300, 800))
        fill = self.stage.GetPrimAtPath("/World/Lights/FillLight")
        if fill:
            fill.GetAttribute("intensity").Set(random.uniform(100, 300))

    def randomize_physics(self):
        table = self.stage.GetPrimAtPath("/World/Table")
        if table:
            from pxr import PhysxSchema
            physx = PhysxSchema.PhysxCollisionAPI(table)
            physx.GetDynamicFrictionAttr().Set(random.uniform(0.2, 1.0))

    def randomize_camera(self):
        for name, cam in self.scene.cameras.items():
            if "depth" in name:
                cam.set_noise_parameters(noise_mean=0.0, noise_std=random.uniform(0.001, 0.03))

    def randomize_objects(self):
        for obj_path in ["/World/Objects/blue_block", "/World/Objects/red_box"]:
            obj = self.stage.GetPrimAtPath(obj_path)
            if obj:
                x = random.uniform(-0.15, 0.15)
                y = random.uniform(-0.15, 0.15)
                obj.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, 0.71))
