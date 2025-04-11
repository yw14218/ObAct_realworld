import numpy as np
import rclpy
from rclpy.node import Node
from PIL import Image
from config.config import D405_HANDEYE_LEFT as T_wristcam_eef, D405_INTRINSIC as K
from utils import pose_inv, quat_from_euler, transform_to_state, solve_transform_3d
from base_servoer import LightGlueVisualServoer
from collections import deque
from interbotix_common_modules.common_robot.robot import robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import tf2_ros
import geometry_msgs.msg
from config.config import OBSERVER_PREFIX
from std_msgs.msg import Bool

class VisualServoing(LightGlueVisualServoer, Node):
    def __init__(self, DIR, bot):
        # Initialize ROS 2 node
        Node.__init__(self, 'visual_servoing_node')
        
        # Initialize the robot
        self.bot = bot

        rgb_img = np.array(Image.open(f"{DIR}/ref_rgb.png"))
        mask_img = np.array(Image.open(f"{DIR}/ref_mask.png")).astype(bool)
        masked_rgb = rgb_img * mask_img[..., None]
        # Initialize the base servoer
        LightGlueVisualServoer.__init__(
            self,
            rgb_ref=masked_rgb,
            seg_ref=np.array(Image.open(f"{DIR}/ref_mask.png")).astype(bool),
            use_depth=True,
            features='superpoint',
            silent=True
        )
        
        self.DIR = DIR
        self.depth_ref = np.array(Image.open(f"{DIR}/ref_depth.png"))
        self.max_translation_step = 0.005
        self.max_rotation_step = np.deg2rad(5)
        self.moving_window = deque(maxlen=5) # sliding window of states to estimate variance

        self.gains = [0.15] * 6 # (proportional gain)

        self.terminate_threshold = (0.01, 5)  # (meters, degrees)

        # Track iterations and servoing state
        self.num_iteration = 0
        self.is_complete = False
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    
    def compute_goal_state(self, T_delta_cam):
        T_current_eef_world = self.bot.arm.get_ee_pose()
        T_delta_eef = T_wristcam_eef @ T_delta_cam @ pose_inv(T_wristcam_eef)
        return transform_to_state(T_current_eef_world @ T_delta_eef), transform_to_state(T_current_eef_world)

    def compute_control_input(self, goal_state, current_state):
        """Compute control input based on the current state estimate."""
        translation_error = goal_state[:3] - current_state[:3]
        rotation_error = goal_state[3:] - current_state[3:]
        translation = np.clip(np.array(self.gains[:3]) * translation_error, -self.max_translation_step, self.max_translation_step)
        rotation = np.clip(np.array(self.gains[3:]) * rotation_error, -self.max_rotation_step, self.max_rotation_step)
        return np.concatenate([translation, rotation])

    def pub_tf(self, goal_state):
        t = geometry_msgs.msg.TransformStamped()

        # Set timestamp
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = f"{OBSERVER_PREFIX}/base_link"  # Parent frame
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

    def run(self):
        """ROS 2 timer callback for the control loop"""            

        while self.is_complete is False:
            # 1. Get new measurement
            mkpts_scores_0, mkpts_scores_1, depth_cur = self.match_lightglue(filter_seg=False)
            if mkpts_scores_0 is None or len(mkpts_scores_0) <= 3:
                self.get_logger().info("Not enough keypoints found, skipping this iteration")
                continue

            # Compute transformation
            T_delta_cam = solve_transform_3d(mkpts_scores_0[:, :2], mkpts_scores_1[:, :2], self.depth_ref, depth_cur, K)

            if T_delta_cam is None:
                self.get_logger().info("Failed to compute transformation, skipping this iteration")
                continue

            # Update error
            T_delta_cam_inv = np.eye(4) @ pose_inv(T_delta_cam)
            translation_error = np.linalg.norm(T_delta_cam_inv[:3, 3])
            rotation_error = np.rad2deg(np.arccos((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2))
            self.get_logger().info(f"Translation Error: {translation_error:.6f}, Rotation Error: {rotation_error:.2f} degrees")
            

            self.moving_window.append((translation_error, rotation_error))
            if len(self.moving_window) == 5:
                if all(t_err < self.terminate_threshold[0] and r_err < self.terminate_threshold[1] 
                       for t_err, r_err in self.moving_window):
                    self.get_logger().info("Global alignment achieved on last 5 predictions, exiting")
                    self.is_complete = True

            # Compute the current state estimate
            goal_state, current_state = self.compute_goal_state(T_delta_cam)

            self.pub_tf(goal_state)

            control_input = self.compute_control_input(goal_state, current_state)

            self.bot.arm.set_ee_cartesian_trajectory(x=control_input[0], y=control_input[1], z=control_input[2], pitch=control_input[4], yaw=control_input[5])
            
            self.num_iteration += 1


def main(args=None):

    if not rclpy.ok():
        rclpy.init(args=args)
    
    # Set up parameters
    DIR = 'tasks/mug'  # Update with actual path
    
    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name=OBSERVER_PREFIX,
        group_name='arm',
        gripper_name='gripper',
        moving_time=0.2,
        accel_time=0.1,
    )

    robot_startup()

    # Create and run node
    node = VisualServoing(DIR, bot)
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()