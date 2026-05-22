"""
Leader Arm ROS2 Teleop Node
조이스틱으로 Franka Panda 원격 제어
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
import numpy as np


class ArmJoyTeleop(Node):
    def __init__(self):
        super().__init__('arm_joy_teleop')
        self.vel_pub = self.create_publisher(Float64MultiArray, '/leader/joint_velocity_commands', 10)
        self.gripper_pub = self.create_publisher(Float64MultiArray, '/leader/gripper_command', 10)
        self.sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.vel_scale = 0.5
        self.deadzone = 0.1
        self.get_logger().info("Arm Joy Teleop started")

    def joy_callback(self, msg: Joy):
        axes = [0.0 if abs(a) < self.deadzone else a for a in msg.axes]
        vel_cmd = Float64MultiArray()
        vel_cmd.data = [
            axes[0] * self.vel_scale,
            axes[1] * self.vel_scale,
            axes[3] * self.vel_scale,
            axes[2] * self.vel_scale,
            (msg.buttons[4] - msg.buttons[5]) * self.vel_scale,
            (msg.buttons[2] - msg.buttons[3]) * self.vel_scale,
            axes[7] * self.vel_scale,
        ]
        self.vel_pub.publish(vel_cmd)
        if msg.buttons[0]:
            self.gripper_pub.publish(Float64MultiArray(data=[1.0]))
        elif msg.buttons[1]:
            self.gripper_pub.publish(Float64MultiArray(data=[0.0]))


def main(args=None):
    rclpy.init(args=args)
    node = ArmJoyTeleop()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
