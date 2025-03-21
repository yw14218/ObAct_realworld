import abc
import os
import time
from datetime import datetime
from threading import Lock

import cv2
import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from config.config import (
    D405_INTRINSIC, D405_RGB_TOPIC_NAME, D405_DEPTH_TOPIC_NAME,
    TSDF_SIZE, TSDF_DIM, WIDTH, HEIGHT
)
from tsdf_torch import TSDFVolume
from utils import transform_to_matrix

class TSDFMapper(Node, abc.ABC):
    def __init__(self, use_depth=True, silent=False, save_data=True, update_frequency=10.0):
        super().__init__('tsdf_mapper')

        # Configuration
        self.use_depth = use_depth
        self.silent = silent
        self.save_data = save_data
        self.update_frequency = update_frequency

        # ROS utilities
        self.bridge = CvBridge()
        self.lock = Lock()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Frame names
        self.base_frame = "vx300s/base_link"
        self.camera_frame = "camera_color_optical_frame"

        # Image storage and tracking
        self.images = {"rgb": None, "depth": None}
        self.last_rgb_stamp = None
        self.last_depth_stamp = None
        self.new_rgb_received = False
        self.new_depth_received = False

        # Data saving
        self.data_dir = "data"
        self.save_counter = 0
        if self.save_data and not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # TSDF setup
        self.tsdf = TSDFVolume(TSDF_SIZE, TSDF_DIM)
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            WIDTH, HEIGHT, D405_INTRINSIC[0, 0], D405_INTRINSIC[1, 1],
            D405_INTRINSIC[0, 2], D405_INTRINSIC[1, 2]
        )
        self.last_update_time = 0.0

        # Visualization setup
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("TSDF Point Cloud Visualization", width=800, height=600)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)

        # Add coordinate frame at origin as a fallback
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.vis.add_geometry(self.coord_frame)

        bbox_geom = o3d.geometry.AxisAlignedBoundingBox(min_bound=self.tsdf.origin, max_bound=self.tsdf.origin + TSDF_SIZE)
        bbox_geom.color = (0, 1, 0)
        self.vis.add_geometry(bbox_geom)
    
        self.needs_update = False
        self.is_initialized = False

        # Subscriptions
        self.rgb_subscriber = self.create_subscription(
            Image, D405_RGB_TOPIC_NAME, self.rgb_image_callback, 10
        )
        if self.use_depth:
            self.depth_subscriber = self.create_subscription(
                Image, D405_DEPTH_TOPIC_NAME, self.depth_image_callback, 10
            )

    def log_info(self, message):
        if not self.silent:
            self.get_logger().info(message)

    def log_warn(self, message):
        self.get_logger().warn(message)

    def log_error(self, message):
        self.get_logger().error(message)

    def get_camera_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return transform_to_matrix(transform)
        except Exception as e:
            self.log_error(f"Failed to get transform: {e}")
            return None

    def save_data_batch(self, rgb_image, depth_image=None):
        if not self.save_data:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rgb_filename = os.path.join(self.data_dir, f"rgb_{timestamp}_{self.save_counter:06d}.png")
        cv2.imwrite(rgb_filename, rgb_image)
        self.log_info(f"Saved RGB image to {rgb_filename}")

        if self.use_depth and depth_image is not None:
            depth_filename = os.path.join(self.data_dir, f"depth_{timestamp}_{self.save_counter:06d}.npy")
            np.save(depth_filename, depth_image)
            self.log_info(f"Saved depth image to {depth_filename}")

        pose = self.get_camera_pose()
        if pose is not None:
            pose_filename = os.path.join(self.data_dir, f"pose_{timestamp}_{self.save_counter:06d}.npy")
            np.save(pose_filename, pose)
            self.log_info(f"Saved camera pose to {pose_filename}")

        self.save_counter += 1

    def rgb_image_callback(self, msg):
        with self.lock:
            try:
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                self.new_rgb_received = True
                self.last_rgb_stamp = msg.header.stamp
                self.log_info("RGB Image received and stored.")
            except Exception as e:
                self.log_error(f"Error in rgb_image_callback: {e}")

    def depth_image_callback(self, msg):
        with self.lock:
            try:
                self.images["depth"] = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.new_depth_received = True
                self.last_depth_stamp = msg.header.stamp
                self.log_info("Depth Image received and stored.")
            except Exception as e:
                self.log_error(f"Error in depth_image_callback: {e}")

    def observe(self, timeout=5.0):
        self.log_info("Observe called, waiting for new images...")

        with self.lock:
            self.new_rgb_received = False
            if self.use_depth:
                self.new_depth_received = False

        start_time = self.get_clock().now()
        timeout_duration = rclpy.duration.Duration(seconds=timeout)

        while rclpy.ok():
            elapsed = self.get_clock().now() - start_time
            if elapsed > timeout_duration:
                self.log_warn(f"Timeout after {timeout} seconds while waiting for images.")
                return None, None, None

            rclpy.spin_once(self, timeout_sec=0.1)

            with self.lock:
                if self.new_rgb_received and (not self.use_depth or self.new_depth_received):
                    self.log_info("All required images received.")
                    rgb_copy = self.images["rgb"].copy() if self.images["rgb"] is not None else None
                    depth_copy = self.images["depth"].copy() if self.use_depth and self.images["depth"] is not None else None
                    pose = self.get_camera_pose()

                    if rgb_copy is not None and self.save_data:
                        self.save_data_batch(rgb_copy, depth_copy)

                    return rgb_copy, depth_copy, pose

        self.log_error("ROS context is no longer valid")
        return None, None, None

    def update_tsdf(self, rgb_image, depth_image, pose):
        current_time = time.time()
        if current_time - self.last_update_time < 1.0 / self.update_frequency:
            return

        self.last_update_time = current_time

        if rgb_image is None or depth_image is None or pose is None:
            self.log_warn("Skipping TSDF update due to missing data")
            return

        depth_image[depth_image < 300] = 0.0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image),
            o3d.geometry.Image(depth_image.astype(np.uint16)),
            depth_scale=1000.0,
            depth_trunc=0.7,
            convert_rgb_to_intensity=False
        )

        self.tsdf._volume.integrate(rgbd, self.intrinsic_o3d, np.linalg.inv(pose))
        self.needs_update = True
        self.log_info("TSDF updated")

    def update_visualization(self):
        # Poll events to keep window responsive
        if not self.vis.poll_events():
            self.log_info("Visualization window closed by user")
            raise KeyboardInterrupt

        self.vis.update_renderer()

        if self.needs_update:
            point_cloud = self.tsdf.get_point_cloud()
            if point_cloud is not None and len(point_cloud.points) > 0:
                with self.lock:
                    # Update point cloud
                    self.pcd_vis.points = point_cloud.points
                    self.pcd_vis.colors = point_cloud.colors

                    # Validate colors (ensure they’re not all black or invalid)
                    colors = np.asarray(self.pcd_vis.colors)
                    if np.all(colors == 0) or not np.any(colors):
                        self.log_warn("Point cloud colors are all zero, setting to red for visibility")
                        self.pcd_vis.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(point_cloud.points), 1)))

                    # Compute bounds and center
                    bounds = self.pcd_vis.get_axis_aligned_bounding_box()
                    center = bounds.get_center()
                    extent = bounds.get_extent()
                    self.log_info(
                        f"Point cloud stats: {len(point_cloud.points)} points, "
                        f"center={center}, extent={extent}, "
                        f"color range={np.min(colors, axis=0)} to {np.max(colors, axis=0)}"
                    )

                    # Adjust view
                    if not self.is_initialized:
                        ctr = self.vis.get_view_control()
                        ctr.set_lookat(center)
                        ctr.set_front([0, 0, 1])
                        ctr.set_up([0, -1, 0])  
                        ctr.set_zoom(0.5)
                        self.is_initialized = True

                    self.vis.update_geometry(self.pcd_vis)
                    self.needs_update = False
                    self.log_info(f"Visualization updated with {len(point_cloud.points)} points")
            else:
                self.log_warn("No valid points in TSDF point cloud to visualize")

    def run(self):
        self.log_info("Starting TSDF mapping loop...")
        while rclpy.ok():
            try:
                rgb_img, depth_img, pose = self.observe()
                if rgb_img is not None and depth_img is not None and pose is not None:
                    self.update_tsdf(rgb_img, depth_img, pose)
                self.update_visualization()
            except KeyboardInterrupt:
                self.log_info("Received shutdown signal")
                break
            except Exception as e:
                self.log_error(f"Error in run loop: {e}")
                time.sleep(1)

    def destroy_node(self):
        if self.vis is not None:
            self.vis.destroy_window()
        super().destroy_node()


def main(args=None):
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"

    rclpy.init(args=args)

    mapper = TSDFMapper(use_depth=True, silent=False, save_data=False, update_frequency=10.0)
    try:
        mapper.run()
    except KeyboardInterrupt:
        mapper.log_info("Shutting down...")
    except Exception as e:
        mapper.log_error(f"Unexpected error: {e}")
    finally:
        mapper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()