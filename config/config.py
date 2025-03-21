import numpy as np
from typing import List

D405_RGB_TOPIC_NAME = "camera/camera/color/image_rect_raw"
D405_DEPTH_TOPIC_NAME = "camera/camera/aligned_depth_to_color/image_raw"

D405_HANDEYE = np.load("config/d405_handeye.npy")
D405_INTRINSIC = np.load("config/d405_intrinsic.npy")

WIDTH, HEIGHT = 848, 480
TSDF_SIZE = 0.5
TSDF_DIM = 256

NUMBER_OF_VIEW_SAMPLES = 48

MOVE_GROUP_ARM: str = "interbotix_arm"
MOVE_GROUP_GRIPPER: str = "interbotix_gripper"

def joint_names(prefix: str = "vx300s") -> List[str]:
    return [
        prefix + "/waist",
        prefix + "/shoulder",
        prefix + "/elbow",
        prefix + "/forearm_roll",
        prefix + "/wrist_angle",
        prefix + "/wrist_rotate",
    ]

def base_link_name(prefix: str = "vx300s") -> str:
    return prefix + "/base_link"

def end_effector_name(prefix: str = "vx300s") -> str:
    return prefix + "/ee_gripper_link"