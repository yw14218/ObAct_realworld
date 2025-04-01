from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

MOVING_TIME_S = 5

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
    robot_1.arm.go_to_sleep_pose(blocking=True)
    robot_2.arm.go_to_sleep_pose(blocking=True)

global_node = create_interbotix_global_node()
robot_1, robot_2 = initialize_robots(global_node)
robot_startup(global_node)
go_to_sleep_pose(robot_1, robot_2)

