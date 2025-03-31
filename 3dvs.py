import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Pose
from PIL import Image
from config.config import D405_HANDEYE as T_wristcam_eef, D405_INTRINSIC as K
from utils import pose_inv, euler_from_matrix, quat_from_euler, transform_to_state, solve_transform_3d
from base_servoer import LightGlueVisualServoer
from collections import deque
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import tf2_ros
import geometry_msgs.msg



from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch

class VisualServoing(LightGlueVisualServoer, Node):
    def __init__(self, DIR, bot):
        # Initialize ROS 2 node
        Node.__init__(self, 'visual_servoing_node')
        
        # Initialize the robot
        self.bot = bot

        # Initialize the base servoer
        LightGlueVisualServoer.__init__(
            self,
            rgb_ref=np.array(Image.open(f"{DIR}/ref_rgb_masked.png")),
            seg_ref=np.array(Image.open(f"{DIR}/ref_mask.png")).astype(bool),
            use_depth=True,
            features='superpoint',
            silent=True
        )
        
        self.DIR = DIR
        self.depth_ref = np.array(Image.open(f"{DIR}/ref_depth.png"))
        self.max_translation_step = 0.005
        self.max_rotation_step = np.deg2rad(5)
        self.moving_window = deque(maxlen=3) # sliding window of states to estimate variance

        self.gains = [0.15] * 6 # (proportional gain)
        self.switch_threshold = (0.05, 5)  # (meters, degrees)
        
        # Track iterations and servoing state
        self.num_iteration = 0
        self.is_complete = False
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)



        # FIXME: This is a hack to get the LightGlue matcher to work with the SuperPoint extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1, filter_threshold=0.1).eval().to(self.device)
        self.ref_image = load_image(f"{DIR}/ref_rgb_masked.png")
    
    def compute_goal_state(self, T_delta_cam):
        T_current_eef_world = self.bot.arm.get_ee_pose()
        T_delta_eef = T_wristcam_eef @ T_delta_cam @ pose_inv(T_wristcam_eef)
        return transform_to_state(T_current_eef_world @ T_delta_eef), transform_to_state(T_current_eef_world)

    def compute_control_input(self, goal_state, current_state):
        """Compute control input based on the current state estimate."""
        translation = goal_state[:3] - current_state[:3]
        rotation = goal_state[3:] - current_state[3:]
        control_input = np.concatenate([
            np.array([np.clip(self.gains[0] * translation[0], -self.max_translation_step, self.max_translation_step)]),
            np.array([np.clip(self.gains[1] * translation[1], -self.max_translation_step, self.max_translation_step)]),
            np.array([np.clip(self.gains[2] * translation[2], -self.max_translation_step, self.max_translation_step)]),
            np.array([np.clip(self.gains[3] * rotation[0], -self.max_rotation_step, self.max_rotation_step)]),
            np.array([np.clip(self.gains[4] * rotation[1], -self.max_rotation_step, self.max_rotation_step)]),
            np.array([np.clip(self.gains[5] * rotation[2], -self.max_rotation_step, self.max_rotation_step)])
        ])
        return control_input

    def run(self):
        """ROS 2 timer callback for the control loop"""            

        # 1. Get new measurement
        mkpts_scores_0, mkpts_scores_1, depth_cur = self.match_lightglue(filter_seg=False)
        if mkpts_scores_0 is None or len(mkpts_scores_0) <= 3:
            self.get_logger().info("Not enough keypoints found, skipping this iteration")
            return
        



        # FIXME: This is a hack to get the LightGlue matcher to work with the SuperPoint extractor
        live_rgb, live_depth = self.observe()
        if live_rgb is None:
            self.log_error("No RGB image received. Check camera and topics.")
            return None, None, None
        
        live_rgb = torch.tensor(live_rgb).permute(2, 0, 1).unsqueeze(0)

        feats0 = self.extractor.extract(self.test_image.to(self.device))
        # print(feats0['keypoints'])
        feats1 = self.extractor.extract(self.ref_image.to(self.device))
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        # print(matches01.keys())
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        matches = matches.cpu()

        all_indices = torch.arange(kpts0.shape[0])
        unmatched = ~torch.isin(all_indices, matches[..., 1])
        unmatched_kpts = kpts0[unmatched]

        matched_center = m_kpts0.mean(dim=0)
        unmatched_center = unmatched_kpts.mean(dim=0)
        diff_vector = matched_center - unmatched_center
        x, y = np.linalg.norm(diff_vector) * 0.1
        T_diff = np.eye(4)
        T_diff[:3, 3] = np.array([x, y, 0]) 









        # Compute transformation
        T_delta_cam = solve_transform_3d(mkpts_scores_0[:, :2], mkpts_scores_1[:, :2], self.depth_ref, depth_cur, K)

        # Update error
        T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)


        # FIXME: Diff
        self.get_logger().info(f"T_diff: {x:.2f}, {y:.2f}")
        T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam @ T_diff)


        translation_error = np.linalg.norm(T_delta_cam_inv[:3, 3])
        rotation_error = np.rad2deg(np.arccos((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2))
        self.get_logger().info(f"Translation Error: {translation_error:.6f}, Rotation Error: {rotation_error:.2f} degrees")
        
        if translation_error < self.switch_threshold[0] and rotation_error < self.switch_threshold[1]:
            self.get_logger().info("Global alignment achieved, exiting")
            self.is_complete = True

        # Compute the current state estimate
        goal_state, current_state = self.compute_goal_state(T_delta_cam)

        # print(goal_state, current_state)
        t = geometry_msgs.msg.TransformStamped()

        # Set timestamp
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "vx300s/base_link"  # Parent frame
        t.child_frame_id = "goal"  # Child frame

        # Set translation
        t.transform.translation.x = goal_state[0]
        t.transform.translation.y = goal_state[1]
        t.transform.translation.z = goal_state[2]

        # Convert Euler angles to quaternion
        q = quat_from_euler(goal_state[3:])

        # Set rotation
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Publish transform
        self.tf_broadcaster.sendTransform(t)

        control_input = self.compute_control_input(goal_state, current_state)

        # Get current pose and orientation
        current_xyz = current_state[:3]
        current_rpy = current_state[3:6]

        # Apply control input
        current_xyz[0] += control_input[0]
        current_xyz[1] += control_input[1]
        current_xyz[2] += control_input[2]
        current_rpy[0] += control_input[3]

        # self.bot.arm.set_ee_cartesian_trajectory(x=control_input[0], y=control_input[1], z=control_input[2], roll=control_input[4], pitch=control_input[5], yaw=control_input[5])
        self.bot.arm.set_ee_cartesian_trajectory(x=control_input[0], y=control_input[1], z=control_input[2], yaw=control_input[5])

        # self.bot.arm.set_ee_pose_components(*current_xyz, *current_rpy, moving_time=0.1, accel_time=0.1)

        # # Compute IK solution
        # positions = ik_solver.compute_ik(current_xyz, quat_from_euler(current_rpy))

        # # Send commands if IK is successful
        # if positions is not None:
        #     self.bot.arm._publish_commands(positions=positions.position[:6], moving_time=0.05, accel_time=0.01, blocking=False)

        self.num_iteration += 1


def main(args=None):

    if not rclpy.ok():
        rclpy.init(args=args)
    
    # Set up parameters
    DIR = 'tasks/mug'  # Update with actual path
    
    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        group_name='arm',
        gripper_name='gripper',
        moving_time=0.2,
        accel_time=0.1,
    )

    robot_startup()

    # Create and run node
    node = VisualServoing(DIR, bot)
    
    try:
        while rclpy.ok():
            node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()