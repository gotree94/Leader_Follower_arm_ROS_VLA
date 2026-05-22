"""
VLA Inference ROS2 Node
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
from threading import Lock
from collections import deque


class VLAPolicyNode(Node):
    def __init__(self):
        super().__init__('vla_policy_node')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.eval()
        self.bridge = CvBridge()
        self.lock = Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_follower_joints = None
        self.latest_leader_joints = None
        self.last_action = np.zeros(7, dtype=np.float32)
        self.inference_times = deque(maxlen=100)

        self.rgb_sub = self.create_subscription(Image, '/follower/rgb/image_raw', self.rgb_callback, 1)
        self.depth_sub = self.create_subscription(Image, '/follower/depth/image_raw', self.depth_callback, 1)
        self.follower_joint_sub = self.create_subscription(JointState, '/follower/joint_states', self.follower_joint_callback, 1)

        self.vel_pub = self.create_publisher(JointTrajectory, '/follower/joint_velocity_commands', 1)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/follower/gripper_command', 1)
        self.timer = self.create_timer(0.025, self.inference_loop)
        self.get_logger().info(f"VLA Policy Node started (device={self.device})")

    def _load_model(self):
        from vla_policy.vla_network import VLAPolicy, VLAConfig
        config = VLAConfig()
        model = VLAPolicy(config)
        path = self.declare_parameter('model_path', 'models/vla/ppo/best_model.pt').value
        model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        return model.to(self.device)

    def rgb_callback(self, msg):
        with self.lock:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = cv2.resize(cv_img, (224, 224)).astype(np.float32) / 255.0
            img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            self.latest_rgb = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def depth_callback(self, msg):
        with self.lock:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            depth = cv2.resize(cv_img, (224, 224))
            depth = np.clip(depth, 0.0, 3.0) / 3.0
            self.latest_depth = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0).to(self.device)

    def follower_joint_callback(self, msg):
        with self.lock:
            self.latest_follower_joints = np.array(msg.position[:6])

    def inference_loop(self):
        with self.lock:
            rgb = self.latest_rgb
            depth = self.latest_depth
            joints = self.latest_follower_joints

        if any(x is None for x in [rgb, depth, joints]):
            return

        with torch.no_grad():
            action, _, _ = self.model(
                rgb, torch.zeros(1, 3, 224, 224).to(self.device), depth,
                torch.zeros(1, 7).to(self.device),
                torch.from_numpy(joints).float().unsqueeze(0).to(self.device),
                torch.tensor([[self.last_action[6]]]).float().to(self.device),
                torch.zeros(1, 512).to(self.device),
                torch.from_numpy(self.last_action).float().unsqueeze(0).to(self.device),
            )

        action_np = action.squeeze().cpu().numpy()
        joint_vel = action_np[:6] * 0.5
        gripper_cmd = np.clip(action_np[6], 0.0, 1.0)
        self.last_action = action_np

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        p = JointTrajectoryPoint()
        p.velocities = joint_vel.tolist()
        p.time_from_start.sec = 0
        traj.points = [p]
        self.vel_pub.publish(traj)
        self.gripper_pub.publish(Float64MultiArray(data=[float(gripper_cmd)]))


def main(args=None):
    rclpy.init(args=args)
    node = VLAPolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
