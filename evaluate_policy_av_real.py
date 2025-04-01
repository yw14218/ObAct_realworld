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

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from lerobot.common.policies.act.modeling_act import ACTPolicy
from config.config import D405_HANDEYE, D405_RGB_TOPIC_NAME
from utils import transform_to_state, state_to_transform, ik_solver
import time

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
    pretrained_policy_path = Path("ckpts/080000/pretrained_model")
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device set to: {device}")
    policy.to(device)
    # policy.config.temporal_ensemble_coeff = 0.01
    policy.reset()
    return policy, device


def main():
    global_node = create_interbotix_global_node()
    bridge = CvBridge()
    robot_1, robot_2 = initialize_robots(global_node)
    robot_startup(global_node)
    policy, device = initialize_policy()
    # robot_1.gripper.release()
    robot_1.arm.go_to_sleep_pose()
    robot_2.arm.go_to_sleep_pose()

    # for i in range(80):
    #     state = transform_to_state(encode_transform(robot_1, robot_2))
    #     state = torch.from_numpy(state).to(device).float()
    #     cv_img = get_image(global_node, bridge)
    #     if cv_img is None:
    #         raise RuntimeError("Failed to get image from camera")
    #     img = preprocess_image(cv_img)

    #     observation = {
    #         "observation.state": state.unsqueeze(0),
    #         "observation.images.wrist_cam_right": img.unsqueeze(0),
    #     }
    #     with torch.inference_mode():
    #         action = policy.select_action(observation)
    #     action = action.squeeze(0).to("cpu").numpy()
    #     action = decode_transform(state_to_transform(action), robot_2)
    #     robot_1.arm.set_ee_pose_matrix(
    #         action, custom_guess=robot_1.arm.get_joint_positions(), moving_time=MOVING_TIME_S
    #     )

    # print("Gripper close initiated")
    # robot_1.gripper.grasp()
    # robot_1.arm.set_ee_cartesian_trajectory(z=0.1, moving_time=2)


if __name__ == "__main__":
    main()
