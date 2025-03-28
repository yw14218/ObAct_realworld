import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool  # Added for boolean publishing
import numpy as np
import time
from PIL import Image
from utils import solve_transform_3d
from base_servoer import LightGlueVisualServoer
from config.config import D405_INTRINSIC
import csv
from datetime import datetime

class ErrorNode(Node):
    def __init__(self) -> None:
        super().__init__('error_node')
        
        # Define error thresholds (adjust these values as needed)
        self.translation_threshold = 0.1  # meters
        self.rotation_threshold = 10.0   # degrees

        # Create publisher for error status
        self.error_pub = self.create_publisher(Bool, 'error_exceeds_threshold', 10)

        # Load reference images
        self.servoer = LightGlueVisualServoer(
            rgb_ref=np.array(Image.open("tasks/mug/ref_rgb_masked.png")),
            seg_ref=np.array(Image.open("tasks/mug/ref_mask.png")).astype(bool),
            use_depth=True,
            features='superpoint',
            silent=True
        )
        self.ref_depth = np.array(Image.open("tasks/mug/ref_depth.png"))
        self.K = D405_INTRINSIC

        # Error tracking
        self.translation_error = float('inf')
        self.rotation_error = float('inf')
        self.running = True
        self.error_history = []  # List to store [timestamp, translation_error, rotation_error]
        
        # Timer for error computation
        self.timer = self.create_timer(0.1, self._compute_errors_background)  # Run every 0.1s

    def _compute_errors_background(self) -> None:
        """Compute and log errors periodically, publish error status."""
        if not self.running:
            return
        msg = Bool()
        try:
            mkpts_scores_0, mkpts_scores_1, depth_cur = self.servoer.match_lightglue(filter_seg=False)
            if mkpts_scores_0 is None or len(mkpts_scores_0) <= 3:
                self.get_logger().warning("Not enough keypoints found")
                msg.data = True
                self.error_pub.publish(msg)
                return

            T_delta_cam = solve_transform_3d(
                mkpts_scores_0[:, :2], 
                mkpts_scores_1[:, :2], 
                self.ref_depth, 
                depth_cur, 
                self.K
            )

            T_delta_cam_inv = np.eye(4) @ np.linalg.inv(T_delta_cam)
            translation = np.linalg.norm(T_delta_cam_inv[:3, 3])
            rotation = np.rad2deg(np.arccos(
                np.clip((np.trace(T_delta_cam_inv[:3, :3]) - 1) / 2, -1.0, 1.0)
            ))
            self.translation_error = translation
            self.rotation_error = rotation
            self.error_history.append([time.time(), translation, rotation])
            
            # Check if error exceeds threshold
            error_exceeds_threshold = (translation > self.translation_threshold and 
                                     rotation > self.rotation_threshold)
            
            # Publish boolean message
            msg.data = bool(error_exceeds_threshold)
            self.error_pub.publish(msg)

            self.get_logger().info(
                f"Current Errors - Translation: {self.translation_error:.6f}, "
                f"Rotation: {self.rotation_error:.2f} degrees, "
                f"Exceeds Threshold: {error_exceeds_threshold}"
            )

        except Exception as e:
            self.get_logger().error(f"Error in computation: {e}")

    def save_errors(self) -> None:
        """Save error history to a CSV file."""
        if not self.error_history:
            self.get_logger().info("No error data to save.")
            return
        
        filename = f"tmp/error_history.csv"
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Translation_Error", "Rotation_Error"])
                writer.writerows(self.error_history)
            self.get_logger().info(f"Error history saved to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save error history: {e}")

    def shutdown(self) -> None:
        """Clean shutdown of the node."""
        self.get_logger().info("Initiating shutdown...")
        self.running = False
        self.save_errors()
        self.destroy_timer(self.timer)
        self.destroy_node()
        self.get_logger().info("Node shutdown complete")

def main(args=None) -> None:
    rclpy.init(args=args)
    error_node = None
    try:
        error_node = ErrorNode()
        rclpy.spin(error_node)
    except KeyboardInterrupt:
        if error_node:
            error_node.get_logger().info("KeyboardInterrupt received, shutting down...")
    except Exception as e:
        if error_node:
            error_node.get_logger().error(f"Unexpected error: {e}")
    finally:
        if error_node:
            error_node.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
        else:
            print("ROS 2 already shut down")

if __name__ == '__main__':
    main()