import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import numpy as np
import time
from PIL import Image
from utils import solve_transform_3d
from base_servoer import LightGlueVisualServoer
from config.config import D405_INTRINSIC
import csv
from datetime import datetime
from lightglue import viz2d

class ErrorNode(Node):
    def __init__(self) -> None:
        super().__init__('error_node')
        
        # Define error thresholds (adjust these values as needed)
        self.translation_threshold = 0.15  # meters
        self.rotation_threshold = 10   # degrees

        # Create publisher for error status
        self.error_pub = self.create_publisher(Bool, 'should_exit_exploration', 10)

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
        
        # Initialize EMA variables
        self.translation_ema = None  # Will hold the EMA for translation
        self.rotation_ema = None     # Will hold the EMA for rotation
        self.alpha = 0.5  # Smoothing factor (0 < alpha < 1, higher means more weight to recent data)

        # Outlier detection parameters
        self.iqr_multiplier = 3.0    # Multiplier for IQR to define outliers
        self.history_size = 100      # Increased history size for better statistics

        # Consecutive checks for stability
        self.consecutive_below_threshold = 0
        self.required_consecutive = 5  # Number of consecutive cycles needed to exit

        # State tracking
        self.has_published_true = False

        # Timer for error computation
        self.timer = self.create_timer(0.1, self._compute_errors_background)  # Run every 0.1s

    def detect_outliers(self, values):
        """Detect outliers using IQR method and return cleaned value."""
        if len(values) < 4:  # Need at least 4 points for IQR
            return False, values[-1] if values else 0.0
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        is_outlier = any(v < lower_bound or v > upper_bound for v in [values[-1]])
        cleaned_value = values[-1] if not is_outlier else np.median(values)
        
        return is_outlier, cleaned_value

    def _compute_errors_background(self) -> None:
        """Compute and log errors periodically, publish error status using robust EMA with outlier detection.
        Exit (publish True and stop computation) if errors fall below threshold for required consecutive cycles."""
        if not self.running:
            return
        
        msg = Bool()

        if self.has_published_true:
            # If we've already decided to exit, only publish True and do no further computation
            msg.data = True
            self.error_pub.publish(msg)
            return

        try:
            # Match keypoints
            mkpts_scores_0, mkpts_scores_1, depth_cur = self.servoer.match_lightglue(filter_seg=False)

            if mkpts_scores_0 is None or len(mkpts_scores_0) <= 5:  # Increased minimum keypoints for robustness
                self.get_logger().warning("Not enough keypoints found")
                msg.data = False  # Do not exit if keypoints are insufficient
                self.error_pub.publish(msg)
                return

            # Compute transform
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

            # Store raw errors in history
            self.error_history.append([time.time(), translation, rotation])
            if len(self.error_history) > self.history_size:
                self.error_history.pop(0)

            # Extract recent translation and rotation histories
            recent_translations = [entry[1] for entry in self.error_history]
            recent_rotations = [entry[2] for entry in self.error_history]

            # Detect and handle outliers
            is_translation_outlier, cleaned_translation = self.detect_outliers(recent_translations)
            is_rotation_outlier, cleaned_rotation = self.detect_outliers(recent_rotations)

            if is_translation_outlier or is_rotation_outlier:
                self.get_logger().warning(f"Outlier detected - Translation: {is_translation_outlier}, Rotation: {is_rotation_outlier}")

            # Use cleaned values for EMA update
            translation_to_use = cleaned_translation
            rotation_to_use = cleaned_rotation

            # Update EMA with cleaned values
            self.translation_ema = self.alpha * translation_to_use + (1 - self.alpha) * self.translation_ema if self.translation_ema is not None else translation_to_use
            self.rotation_ema = self.alpha * rotation_to_use + (1 - self.alpha) * self.rotation_ema if self.rotation_ema is not None else rotation_to_use

            # Check if EMA errors are below thresholds
            if (self.translation_ema < self.translation_threshold and 
                self.rotation_ema < self.rotation_threshold):
                self.consecutive_below_threshold += 1
            else:
                self.consecutive_below_threshold = 0

            should_exit_exploration = self.consecutive_below_threshold >= self.required_consecutive

            # Publish boolean message
            msg.data = bool(should_exit_exploration)
            self.error_pub.publish(msg)

            self.get_logger().info(
                f"Current Errors - Raw Translation: {translation:.6f}, Cleaned Translation: {cleaned_translation:.6f}, "
                f"Raw Rotation: {rotation:.2f}, Cleaned Rotation: {cleaned_rotation:.2f}, "
                f"EMA Translation: {self.translation_ema:.6f}, EMA Rotation: {self.rotation_ema:.2f}, "
                f"Consecutive Below: {self.consecutive_below_threshold}, "
                f"Should Exit: {should_exit_exploration}"
            )

            if should_exit_exploration:
                self.has_published_true = True
                self.get_logger().info("Exiting exploration: Errors consistently below threshold.")

        except Exception as e:
            self.get_logger().error(f"Error in computation: {e}")
            msg.data = False  # Default to False on error to avoid false positives
            self.error_pub.publish(msg)

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