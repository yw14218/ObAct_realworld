import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import numpy as np
import re
from utils import compose_homogeneous_matrix, rot_mat_to_quat, solve_transform_3d
from typing import Optional, Tuple
from config.config import D405_HANDEYE
from moveit2 import MoveIt2Viper
from PIL import Image
from builtin_interfaces.msg import Duration
from base_servoer import LightGlueVisualServoer
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from std_msgs.msg import Bool
import subprocess

class Explorer(Node):
    def __init__(self) -> None:
        super().__init__('av_controller')
        self.cli = self.create_client(Trigger, 'compute_information_gain')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_information_gain service...')
        self.get_logger().info('Connected to /compute_information_gain service')
        self.error_sub = self.create_subscription(Bool, 'error_exceeds_threshold', self.error_callback, 10)  # Corrected argument order
        self.is_running = True

    def error_callback(self, msg: Bool) -> None:
            """
            Callback function for the 'error_exceeds_threshold' topic.
            """
            if msg.data:
                pass
            else:
                self.is_running = True

    def call_service(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Call the compute_information_gain service and return the parsed position
        and quaternion from its message.
        """
        req = Trigger.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Service call succeeded: {response.message}')
                return self.parse_viewpoint(response.message)
            else:
                self.get_logger().error(f'Service call failed: {response.message}')
        else:
            self.get_logger().error('Service call failed: No response')

        return None, None

    @staticmethod
    def parse_viewpoint(message: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse a message string into position and quaternion arrays.
        """
        try:
            pattern = r'position: \[([-\d.\s]+)\], orientation: \[([-\d.\s]+)\]'
            match = re.match(pattern, message)
            if not match:
                raise ValueError(f"Invalid message format: {message}")
            
            pos_str, quat_str = match.group(1), match.group(2)
            position = np.array([float(x) for x in pos_str.split()], dtype=np.float64)
            quaternion = np.array([float(x) for x in quat_str.split()], dtype=np.float64)

            print(f"Received viewpoint - Position: {position}, Quaternion: {quaternion}")
            camera_goal = compose_homogeneous_matrix(position, quaternion)
            eef_goal = camera_goal @ np.linalg.inv(D405_HANDEYE)
            return eef_goal

        except Exception as e:
            # Log error in a Node could be better but here we use print for static method
            print(f"Error parsing viewpoint: {e}")
            return None, None

def main(args=None) -> None:
    rclpy.init(args=args)
    controller = Explorer()
    moveit_viper = MoveIt2Viper()

    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        group_name='arm_2',
        gripper_name='gripper',
        moving_time=1.2,
        accel_time=0.3
    )

    robot_startup()

    try:
        while controller.is_running:
            eef_goal = controller.call_service()
            plan = moveit_viper.move_to_pose(eef_goal[:3, 3], rot_mat_to_quat(eef_goal[:3, :3]))
            for point in plan.points:
                if not controller.is_running:
                    break
                msg = JointTrajectory()
                msg.header = Header()
                msg.joint_names = plan.joint_names
                # moveit_viper.moveit2.move_to_configuration(joint_positions=point.positions[:6], joint_names=plan.joint_names[:6])
                bot.arm.set_joint_positions(point.positions[:6], moving_time=1.2, accel_time=0.3, blocking=True)
            else:
                print("No valid viewpoint received.")

    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')

    finally:
        # Cleanup ROS resources
        rclpy.spin_once(controller)
        robot_shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()