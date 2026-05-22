"""
Leader Arm Trajectory Recorder
조인트 궤적 녹화 및 파일 저장
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import json
import os
from datetime import datetime


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')
        self.joint_sub = self.create_subscription(JointState, '/leader/joint_states', self.joint_callback, 10)
        self.recording = False
        self.trajectory = []
        self.output_dir = os.path.expanduser("~/trajectories")
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info("Trajectory Recorder ready")

    def joint_callback(self, msg: JointState):
        if not self.recording:
            return
        self.trajectory.append({
            "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            "position": list(msg.position),
            "velocity": list(msg.velocity),
            "effort": list(msg.effort),
        })

    def start_recording(self):
        self.recording = True
        self.trajectory = []
        self.get_logger().info("Recording started")

    def stop_recording(self):
        self.recording = False
        filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_frames": len(self.trajectory),
                "arm": "franka_panda",
            },
            "frames": self.trajectory,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        self.get_logger().info(f"Recording saved: {filepath} ({len(self.trajectory)} frames)")
        return filepath


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
