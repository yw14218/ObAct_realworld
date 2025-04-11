from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import subprocess
import rclpy
import signal
import os
from exploration import Explorer
from vs import VisualServoing
import time
from cartesian_interpolation import interpolate_cartesian_pose, Pose, pose_to_xyz_wxyz, xyz_wxyz_to_pose
from utils import matrix_to_xyz_wxyz, xyz_wxyz_to_matrix, ik_solver
import numpy as np
from rclpy.executors import MultiThreadedExecutor

MOVING_TIME_S = 3

def initialize_robots(global_node):
    robot_1 = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='arm_1',
        moving_time=MOVING_TIME_S,
        node=global_node,
    )
    robot_2 = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='arm_2',
        moving_time=MOVING_TIME_S,
        node=global_node,
    )
    return robot_1, robot_2

def go_to_sleep_pose(robot_1, robot_2):
    robot_1.arm.go_to_sleep_pose()
    robot_2.arm.go_to_sleep_pose()

joint_look_positions = [0, -0.72, 0.59, 0, 1.02, 0]

def main():

    # Initialize ROS 2 and nodes
    rclpy.init()
    global_node = create_interbotix_global_node()
    robot_1, robot_2 = initialize_robots(global_node)
    robot_startup(global_node)

    # Move robots to look positions
    robot_1.arm.set_joint_positions(joint_look_positions, moving_time=MOVING_TIME_S, blocking=False)
    robot_2.arm.set_joint_positions(joint_look_positions, moving_time=MOVING_TIME_S, blocking=True)
    
    # Launch subprocesses in new terminals
    # Launch subprocesses in new terminals with a new process group for later termination
    process_mapper = subprocess.Popen(['gnome-terminal', '--', 'python3', 'mapper.py', '--text_prompt', 'green mug'], preexec_fn=os.setsid)    
    process_matcher = subprocess.Popen(['gnome-terminal', '--', 'python3', 'matcher.py'], preexec_fn=os.setsid)
    vs = VisualServoing("tasks/mug", robot_2)

    try:
        explorer = Explorer()
        # while True:
        while explorer.is_running:
            current_pose = robot_2.arm.get_ee_pose()

            eef_goal = None
            while eef_goal is None:
                eef_goal = explorer.call_service()

            try:
                start_xyz_wxyz = matrix_to_xyz_wxyz(current_pose)
                end_xyz_wxyz = matrix_to_xyz_wxyz(eef_goal)
                start_pose = xyz_wxyz_to_pose(start_xyz_wxyz)
                end_pose = xyz_wxyz_to_pose(end_xyz_wxyz)
                # Generate the full trajectory plan
                waypoints = interpolate_cartesian_pose(
                    start_pose,
                    end_pose,
                    max_step=0.01
                )
                waypoints = [xyz_wxyz_to_matrix(pose_to_xyz_wxyz(pose)) for pose in waypoints]

            except RuntimeError as ex:
                print(f"Runtime error: {ex}")
                continue

            for waypoint in waypoints:
                if not explorer.is_running:
                    break
                # robot_2.arm.set_ee_pose_matrix(waypoint, custom_guess=robot_2.arm.get_joint_positions(), moving_time=1)
                qpos = ik_solver.ik(waypoint, qinit=robot_2.arm.get_joint_positions())
                if qpos is not None:
                    robot_2.arm.set_joint_positions(qpos, moving_time=1)
                else:
                    print("No IK solution found for waypoint")
                    continue


        # Terminate subprocesses by killing their process groups
        if process_mapper:
            try:
                os.killpg(process_mapper.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        if process_matcher:
            try:
                os.killpg(process_matcher.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
                process_matcher.kill()


        vs.run()

        rclpy.spin(global_node)
    except KeyboardInterrupt:
        pass  # Signal handler will take care of cleanup

if __name__ == '__main__':
    main()
