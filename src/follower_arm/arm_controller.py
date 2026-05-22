"""
MoveIt2 VLA Integration — Follower Arm Controller
"""
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
import numpy as np


class ArmVLAExecutor(Node):
    def __init__(self):
        super().__init__('arm_vla_executor')
        self.max_velocity = 0.5
        self.cmd_sub = self.create_subscription(
            JointTrajectory, '/follower/joint_velocity_commands',
            self.velocity_callback, 10)
        self.cmd_pub = self.create_publisher(
            JointTrajectory, '/follower/joint_trajectory_controller/commands', 10)
        self.get_logger().info("Arm VLA Executor ready")

    def velocity_callback(self, msg: JointTrajectory):
        velocities = np.array(msg.points[0].velocities)
        safe_velocities = np.clip(velocities, -self.max_velocity, self.max_velocity)
        out_msg = JointTrajectory()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.velocities = safe_velocities.tolist()
        point.time_from_start.sec = 0
        out_msg.points = [point]
        self.cmd_pub.publish(out_msg)

    def emergency_stop(self):
        stop_msg = JointTrajectory()
        stop_msg.points = [JointTrajectoryPoint(velocities=[0.0]*6)]
        self.cmd_pub.publish(stop_msg)
        self.get_logger().warn("Emergency stop activated")


def main(args=None):
    rclpy.init(args=args)
    node = ArmVLAExecutor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
