import numpy as np
from typing import List

RIGHT_ARM_AS_OBSERVER = True
D405_RGB_TOPIC_NAME = "camera/camera/color/image_rect_raw"
D405_DEPTH_TOPIC_NAME = "camera/camera/aligned_depth_to_color/image_raw"

D405_HANDEYE_LEFT = np.load("config/d405_handeye_left.npy")
D405_HANDEYE_RIGHT = np.load("config/d405_handeye_right.npy")
D405_INTRINSIC_LEFT = np.load("config/d405_intrinsic_left.npy")
D405_INTRINSIC = np.load("config/d405_intrinsic_right.npy")

if RIGHT_ARM_AS_OBSERVER:
    D405_INTRINSIC = D405_INTRINSIC
    D405_HANDEYE = D405_HANDEYE_RIGHT
    OBSERVER_PREFIX = "arm_2"
else:
    D405_INTRINSIC = D405_INTRINSIC_LEFT
    D405_HANDEYE = D405_HANDEYE_LEFT
    OBSERVER_PREFIX = "arm_1"

WIDTH, HEIGHT = 640, 480
TSDF_SIZE = 0.5
TSDF_DIM = 256

NUMBER_OF_VIEW_SAMPLES = 48

MOVE_GROUP_ARM: str = "interbotix_arm"
MOVE_GROUP_GRIPPER: str = "interbotix_gripper"

def joint_names(prefix: str = "arm_2") -> List[str]:
    return [
        prefix + "/waist",
        prefix + "/shoulder",
        prefix + "/elbow",
        prefix + "/forearm_roll",
        prefix + "/wrist_angle",
        prefix + "/wrist_rotate",
    ]

def base_link_name(prefix: str = OBSERVER_PREFIX) -> str:
    return prefix + "/base_link"

def end_effector_name(prefix: str = OBSERVER_PREFIX) -> str:
    return prefix + "/ee_gripper_link"