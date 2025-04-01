from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import subprocess
import rclpy
import signal
import sys
from exploration import Explorer
from utils import rot_mat_to_quat
from moveit2 import MoveIt2Viper
from vs import VisualServoing

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


def signal_handler():
    # Terminate subprocesses
    if process_mapper:
        process_mapper.terminate()
        try:
            process_mapper.wait(timeout=5)  # Wait for the process to terminate
        except subprocess.TimeoutExpired:
            process_mapper.kill()  # Force kill if it doesn't terminate

    if process_matcher:
        process_matcher.terminate()
        try:
            process_matcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process_matcher.kill()

def main():
    global global_node, robot_1, robot_2, process_mapper, process_matcher

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize ROS 2 and nodes
    rclpy.init()
    global_node = create_interbotix_global_node()
    robot_1, robot_2 = initialize_robots(global_node)
    robot_startup(global_node)

    # Move robots to look positions
    robot_1.arm.set_joint_positions(joint_look_positions, moving_time=MOVING_TIME_S, blocking=False)
    robot_2.arm.set_joint_positions(joint_look_positions, moving_time=MOVING_TIME_S, blocking=True)

    # Launch subprocesses in new terminals
    process_mapper = subprocess.Popen(['gnome-terminal', '--', 'python3', 'mapper.py', '--text_prompt', 'green mug'])
    process_matcher = subprocess.Popen(['gnome-terminal', '--', 'python3', 'matcher.py'])

    # Keep the main script running (ROS 2 spin)
    try:
        moveit_viper = MoveIt2Viper()
        explorer = Explorer()
        # while explorer.is_running:
        for i in range(2):
            eef_goal = explorer.call_service()
            plan = moveit_viper.move_to_pose(eef_goal[:3, 3], rot_mat_to_quat(eef_goal[:3, :3]))
            for point in plan.points:
                if not explorer.is_running:
                    break
                robot_2.arm.set_joint_positions(point.positions[:6], moving_time=1, accel_time=0.3, blocking=True)
            else:
                print("No valid viewpoint received.")

        signal_handler()  # Cleanup subprocesses
        # vs = VisualServoing('tasks/mug', robot_2)
        # vs.run()

        rclpy.spin(global_node)
    except KeyboardInterrupt:
        pass  # Signal handler will take care of cleanup

if __name__ == '__main__':
    main()
