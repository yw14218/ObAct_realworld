from pathlib import Path
import numpy as np
import torch
import cv2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from torchvision.transforms import v2
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, JointState
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from lerobot.common.policies.act.modeling_act import ACTPolicy
from config.config import D405_HANDEYE, D405_RGB_TOPIC_NAME
from utils import transform_to_state, state_to_transform, ik_solver, matrix_to_xyz_wxyz, xyz_wxyz_to_matrix
from cartesian_interpolation import interpolate_cartesian_pose, Pose, pose_to_xyz_wxyz, xyz_wxyz_to_pose
import time
from utils import ik_solver
from filter import OneEuroFilter

MOVING_TIME_S = 0.5
SLEEP_DURATION = Duration(seconds=MOVING_TIME_S)


def T_left_right():
    T = np.eye(4)
    T[:3, 3] = np.array([0.938, 0.0, 0.0])
    T[:3, :3] = R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
    return T


def decode_transform(T_action_in_camera, robot_2):
    return T_left_right() @ robot_2.arm.get_ee_pose() @ D405_HANDEYE @ T_action_in_camera


def encode_transform(robot_1, robot_2):
    return np.linalg.inv(D405_HANDEYE) @ np.linalg.inv(robot_2.arm.get_ee_pose()) @ T_left_right() @ robot_1.arm.get_ee_pose()


def preprocess_image(img):
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = torch.from_numpy(img).permute(2, 0, 1).float().to("cuda")
    img = v2.Resize((240, 320))(img)
    img = v2.CenterCrop((224, 308))(img)
    return img


def get_image(node: Node, bridge: CvBridge, topic=D405_RGB_TOPIC_NAME, msg_type=Image, timeout=5.0):
    message = None

    def callback(msg):
        nonlocal message
        message = msg

    sub = node.create_subscription(msg_type, topic, callback, 10)
    start_time = node.get_clock().now()
    timeout_duration = Duration(seconds=timeout)

    while rclpy.ok() and (node.get_clock().now() - start_time) < timeout_duration:
        rclpy.spin_once(node, timeout_sec=0.05)
        if message is not None:
            node.destroy_subscription(sub)
            return bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
    node.destroy_subscription(sub)
    node.get_logger().warn(f"Timeout waiting for message on topic {topic}")
    return None


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


def initialize_policy():
    pretrained_policy_path = Path("ckpts/mug_pickup_15hz/050000/pretrained_model")
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device set to: {device}")
    policy.to(device)
    # policy.config.temporal_ensemble_coeff = None
    # policy.config.n_action_steps = 20
    policy.reset()
    return policy, device

def wait_for_message(self, topic: str, msg_type, timeout: float = 5.0):
    """Wait for a single message on the specified topic."""
    message = None
    def callback(msg):
        nonlocal message
        message = msg

    sub = self.create_subscription(msg_type, topic, callback, 10)
    start_time = self.get_clock().now()
    timeout_duration = rclpy.duration.Duration(seconds=timeout)

    while rclpy.ok() and (self.get_clock().now() - start_time) < timeout_duration:
        rclpy.spin_once(self, timeout_sec=0.05)
        if message is not None:
            self.destroy_subscription(sub)
            return message
    self.destroy_subscription(sub)
    self.get_logger().warn(f"Timeout waiting for message on topic {topic}")
    return None
    
def main():
    global_node = create_interbotix_global_node()
    bridge = CvBridge()
    robot_1, robot_2 = initialize_robots(global_node)
    robot_startup(global_node)
    policy, device = initialize_policy()
    gripper_close = False
    filter_pos = OneEuroFilter(min_cutoff=0.01, beta=10.0)
    filter_rot = OneEuroFilter(min_cutoff=0.01, beta=10.0)

    # robot_1.gripper.grasp()
    # robot_1.gripper.release()
    # # robot_1.arm.go_to_sleep_pose()
    # # # input("Press Enter to grasp...")
    # # robot_1.gripper.grasp()
    # # robot_1.gripper.set_pressure(1)

    # # goal = [-0.44945639, -0.21322334,  1.04924285,  1.52477694, 0.48320395, -2.26108766]
    # joint_look_positions = [0, -0.72, 0.59, 0, 1.02, 0]
    # # print(robot_1.arm.get_joint_positions())
    # robot_1.arm.set_joint_positions(joint_look_positions, moving_time=8, blocking=True)

    # raise
    # robot_1.gripper.release()
    # # time.sleep(2)

    for i in range(10000):
        arm_state = transform_to_state(encode_transform(robot_1, robot_2))
        gripper_state = wait_for_message(global_node, "/arm_1/joint_states", JointState, timeout=0.1).position[6]
        gripper_state = 1 if gripper_state < 0.8 else 0
        state = np.concatenate((arm_state, [gripper_state]), axis=0)
        state = torch.from_numpy(state).to(device).float()

        cv_img = get_image(global_node, bridge)
        if cv_img is None:
            raise RuntimeError("Failed to get image from camera")
        img = preprocess_image(cv_img)

        observation = {
            "observation.state": state.unsqueeze(0),
            "observation.images.wrist_cam_right": img.unsqueeze(0),
        }
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = action.squeeze(0)
        gripper_action = action[6]
        arm_action = action[:6].squeeze(0).to("cpu").numpy()
        t = time.time()
        # arm_action[:3] = filter_pos(t, arm_action[:3])
        # arm_action[3:] = filter_rot(t, arm_action[3:])
        arm_action = decode_transform(state_to_transform(arm_action), robot_2)
        

        if i == 0:
            initial_ee_pose = matrix_to_xyz_wxyz(robot_1.arm.get_ee_pose())
            goal_ee_pose = matrix_to_xyz_wxyz(arm_action)
            initial_ee_pose = xyz_wxyz_to_pose(initial_ee_pose)
            goal_ee_pose = xyz_wxyz_to_pose(goal_ee_pose)
            # Generate the full trajectory plan
            waypoints = interpolate_cartesian_pose(
                initial_ee_pose,
                goal_ee_pose,
                max_step=0.01
            )
            waypoints = [xyz_wxyz_to_matrix(pose_to_xyz_wxyz(pose)) for pose in waypoints]

            for waypoint in waypoints:
                robot_1.arm.set_ee_pose_matrix(
                    waypoint, moving_time=MOVING_TIME_S, custom_guess=robot_1.arm.get_joint_positions(), blocking=True
                )
            # input("Press Enter to continue...")
        else:
            # robot_1.arm.set_ee_pose_matrix(
            #     arm_action, custom_guess=robot_1.arm.get_joint_positions(), moving_time=0.5, blocking=False
            # )
            qpos = ik_solver.ik(arm_action, qinit=robot_1.arm.get_joint_positions())
            if qpos is not None:
                robot_1.arm.set_joint_positions(qpos, moving_time=0.5, blocking=False)
                print("IK success.")
            else:
                print("IK failed.")
            # print(np.linalg.norm(arm_action - robot_1.arm.get_ee_pose()))
            print(gripper_action)
            if gripper_action > 0.9 and not gripper_close:
                robot_1.gripper.grasp()
                gripper_close = True
            elif gripper_action <= 0.9 and gripper_close:
                robot_1.gripper.release()
                gripper_close = False
    # print("Gripper close initiated")


if __name__ == "__main__":
    main()
