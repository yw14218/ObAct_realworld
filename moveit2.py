#!/usr/bin/env python3

from threading import Thread
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from pymoveit2 import MoveIt2
import config.config as cfg
import numpy as np
from scipy.spatial.transform import Rotation as R
import trajectory_msgs
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import Header
from scipy.spatial.transform import Slerp
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

class MoveIt2Viper:
    def __init__(self):
        # rclpy.init()
        self.node = Node("Moveit2_Viper")
        
        # Declare parameters for position and orientation
        self.node.declare_parameter("synchronous", True)
        
        # Create callback group to allow parallel execution
        self.callback_group = ReentrantCallbackGroup()

        # Create MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=self.node,
            joint_names=cfg.joint_names(),
            base_link_name=cfg.base_link_name(),
            end_effector_name=cfg.end_effector_name(),
            group_name=cfg.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )
        
        self.moveit2.max_velocity = 0.01
        self.moveit2.max_acceleration = 0.01

        self.moveit2.add_collision_box(
            id="table",
            pose=(0, 0, 0, 0, 0, 0),
            position=(0, 0, 0),
            quat_xyzw=(0, 0, 0, 1),
            size=(1, 1, 0.05),
        )
        # Spin the node in a background thread
        self.executor = rclpy.executors.MultiThreadedExecutor(2)
        self.executor.add_node(self.node)
        self.executor_thread = Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        
        self.node.create_rate(1.0).sleep()

        self.joint_pub = self.node.create_publisher(JointTrajectory, '/joint_trajectory_controller/command', 10)
    


    def compute_ik(self, position, quat_xyzw):
        # Get parameter
        synchronous = self.node.get_parameter("synchronous").get_parameter_value().bool_value
        
        retval = None
        if synchronous:
            retval = self.moveit2.compute_ik(position, quat_xyzw, wait_for_server_timeout_sec=0.1)
        else:
            future = self.moveit2.compute_ik_async(position, quat_xyzw, wait_for_server_timeout_sec=0.1)
            if future is not None:
                rate = self.node.create_rate(10)
                while not future.done():
                    rate.sleep()
                retval = self.moveit2.get_compute_ik_result(future)
        
        if retval is None:
            print("Computing IK Failed.")
        else:
            # print("Succeeded. Result: " + str(retval))
            return retval
    
    def get_current_pose(self):
        curr_pose_stamped = self.moveit2.compute_fk()
        curr_pose = curr_pose_stamped.pose
        T = np.eye(4)
        T[:3, :3] = R.from_quat([curr_pose.orientation.x, curr_pose.orientation.y, curr_pose.orientation.z, curr_pose.orientation.w]).as_matrix()
        T[:3, 3] = np.array([curr_pose.position.x, curr_pose.position.y, curr_pose.position.z])
        return T

    def move_to_pose(self, position, quat_xyzw):
        """
        Generate a plan to a target pose and execute it waypoint by waypoint
        """
        # Generate the full trajectory plan
        plan = self.moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=True,
            execute=False
        )
        
        if plan is None:
            raise RuntimeError("Failed to generate trajectory plan")
        
        return plan

    def shutdown(self):
        rclpy.shutdown()
        self.executor_thread.join()

# example
if __name__ == "__main__":
    rclpy.init()
    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        group_name='arm',
        gripper_name='gripper',
        moving_time=2,
        accel_time=0.4
    )

    robot_startup()

    moveit2viper = MoveIt2Viper()
    position = [0.3, 0.0, 0.2]
    quat_xyzw = [0.0, 0.0, 0.0, 1.0]
    # gpos = moveit2viper.compute_ik(position, quat_xyzw)
    # moveit2viper.moveit2.move_to_configuration(joint_positions=gpos.position[:6], joint_names=gpos.name[:6])

    plan = moveit2viper.move_to_pose(position, quat_xyzw)
    print(len(plan.points))
    for point in plan.points:
        # moveit2viper.moveit2.move_to_configuration(joint_positions=point.positions[:6], joint_names=plan.joint_names[:6])
        bot.arm.set_joint_positions(point.positions[:6], moving_time=2, accel_time=0.4)
        # msg = JointTrajectory()
        # msg.header = Header()
        # msg.joint_names = plan.joint_names
        # msg.points = [point]
        # moveit2viper.joint_pub.publish(msg)
        # moveit2viper.node.create_rate(1.0).sleep()

    rclpy.spin(moveit2viper.node)
