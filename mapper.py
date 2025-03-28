import abc
import argparse
import os
import time
import shutil
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
    D405_RGB_TOPIC_NAME, D405_DEPTH_TOPIC_NAME, D405_HANDEYE,
    TSDF_SIZE, TSDF_DIM, base_link_name, end_effector_name
)
from tsdf_torch import CAMERA_SCALE, TSDFVolume, ViewEvaluator, create_camera_visualizations, load_intrinsics
from utils import transform_to_matrix, rot_mat_to_quat, ik_solver
from std_srvs.srv import Trigger

from perception import Perception

class TSDFMapper(Node, abc.ABC):
    def __init__(self, bbox, sampled_viewpoints, use_depth=True, silent=False, save_data=True, update_frequency=10.0):
        super().__init__('tsdf_mapper')

        # Dummy data for testing (replace with your actual bbox and viewpoints)
        self.bbox = bbox if bbox is not None else np.array([0, 0, 0, 1, 1, 1])
        self.sampled_viewpoints = sampled_viewpoints if sampled_viewpoints is not None else [
            {'rotation': np.eye(3), 'position': [0, 0, 0]}
        ]

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
        self.intrinsic = load_intrinsics("config/d405_intrinsic_right.npy")
        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(**self.intrinsic)
        self.last_update_time = 0.0

        # Visualization setup
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("TSDF Point Cloud Visualization", width=800, height=600)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)

        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        self.vis.add_geometry(self.coord_frame)

        tsdf_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.tsdf.origin,
            max_bound=self.tsdf.origin + TSDF_SIZE
        )
        tsdf_bbox.color = (0, 1, 0)
        self.vis.add_geometry(tsdf_bbox)

        object_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=self.bbox[:3],
            max_bound=self.bbox[3:]
        )
        object_bbox.color = (1, 0, 0)
        self.vis.add_geometry(object_bbox)

        self.sampled_poses = [
            np.linalg.inv(
                np.block([
                    [view['rotation'], np.array(view['position']).reshape(3, 1)],
                    [np.zeros((1, 3)), 1]
                ])
            ) for view in self.sampled_viewpoints
        ]
        sampled_cameras = create_camera_visualizations(
            self.sampled_poses, self.intrinsic_o3d,
            colors=[[0, 0, 0]] * len(self.sampled_poses),
            scales=[CAMERA_SCALE] * len(self.sampled_poses)
        )
        self.sampled_cameras = sampled_cameras
        # for camera in sampled_cameras:
        #     self.vis.add_geometry(camera)

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

        # Service
        self.srv = self.create_service(
            Trigger,
            'compute_information_gain',
            self.compute_information_gain_callback
        )

        # Timer for periodic updates
        self.timer = self.create_timer(1.0 / self.update_frequency, self.update_callback)

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
                base_link_name(), end_effector_name(),
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return transform_to_matrix(transform) @ D405_HANDEYE
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

    def observe(self):
        with self.lock:
            if self.new_rgb_received and (not self.use_depth or self.new_depth_received):
                self.log_info("All required images received.")
                rgb_copy = self.images["rgb"].copy() if self.images["rgb"] is not None else None
                depth_copy = self.images["depth"].copy() if self.use_depth and self.images["depth"] is not None else None
                pose = self.get_camera_pose()

                if rgb_copy is not None and self.save_data:
                    self.save_data_batch(rgb_copy, depth_copy)

                self.new_rgb_received = False
                self.new_depth_received = False
                return rgb_copy, depth_copy, pose
        return None, None, None

    def update_tsdf(self, rgb_image, depth_image, pose):
        current_time = time.time()
        if current_time - self.last_update_time < 1.0 / self.update_frequency:
            return

        self.last_update_time = current_time

        if rgb_image is None or depth_image is None or pose is None:
            raise RuntimeError("Invalid input data for TSDF update")

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
        if not self.vis.poll_events():
            self.log_info("Visualization window closed by user")
            return False

        self.vis.update_renderer()

        if self.needs_update:
            point_cloud = self.tsdf.get_point_cloud()
            if point_cloud is not None and len(point_cloud.points) > 0:
                with self.lock:
                    self.pcd_vis.points = point_cloud.points
                    self.pcd_vis.colors = point_cloud.colors

                    colors = np.asarray(self.pcd_vis.colors)
                    if np.all(colors == 0) or not np.any(colors):
                        self.log_warn("Point cloud colors are all zero, setting to red")
                        self.pcd_vis.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(point_cloud.points), 1)))

                    if not self.is_initialized:
                        ctr = self.vis.get_view_control()
                        ctr.set_lookat(self.pcd_vis.get_center())
                        ctr.set_front([0, 0, 1])
                        ctr.set_up([0, -1, 0])
                        ctr.set_zoom(0.5)
                        self.is_initialized = True

                    self.vis.update_geometry(self.pcd_vis)
                    self.needs_update = False
                    self.log_info(f"Visualization updated with {len(point_cloud.points)} points")
            else:
                self.log_warn("No valid points in TSDF point cloud")
        return True

    def update_callback(self):
        rgb_img, depth_img, pose = self.observe()
        if rgb_img is not None and depth_img is not None and pose is not None:
            self.update_tsdf(rgb_img, depth_img, pose)
        if not self.update_visualization():
            raise KeyboardInterrupt  # Stop if visualization window is closed
        
    def compute_information_gain_callback(self, request, response):
        try:
            # Initialize seen_indices if not already present
            if not hasattr(self, 'seen_indices'):
                self.seen_indices = set()

            evaluator = ViewEvaluator(self.tsdf, self.intrinsic, self.bbox)
            start_time = time.time()
            
            # Compute information gain for all poses
            gains = [evaluator.compute_information_gain(pose) for pose in self.sampled_poses]
            computation_time = time.time() - start_time
            self.log_info(f"Total gain computation took {computation_time:.2f} seconds")
            
            gains = np.array(gains)
            positive_gains = gains[gains > 0]
            self.log_info(f"Gains > 0: {positive_gains.shape[0]} views, "
                        f"Mean: {np.mean(positive_gains):.4f}, "
                        f"Max: {np.max(gains):.4f}")

            # Mask out previously seen indices
            masked_gains = gains.copy()
            for idx in self.seen_indices:
                masked_gains[idx] = -np.inf  # Set seen indices to negative infinity

            # Find the top gain index that hasn’t been seen
            if np.all(masked_gains == -np.inf):
                # All indices have been seen
                self.log_warn("All possible indices have been returned previously")
                response.success = False
                response.message = "No new indices available"
                return response

            top_gain_idx = np.argmax(masked_gains)
            self.seen_indices.add(top_gain_idx)  # Remember this index

            # Update visualization
            with self.lock:
                for i, camera in enumerate(self.sampled_cameras):
                    color = [1, 0, 0] if i == top_gain_idx else [0, 0, 1]  # Red for top, blue for others
                    camera.paint_uniform_color(color)
                    self.vis.update_geometry(camera)

            self.needs_update = True

            viewpoint = self.sampled_viewpoints[top_gain_idx]
            xyz = viewpoint['position']
            quat = rot_mat_to_quat(viewpoint['rotation'])

            response.success = True
            response.message = f"position: {xyz}, orientation: {quat}"
            
        except Exception as e:
            self.log_error(f"Error in compute_information_gain_callback: {e}")
            response.success = False
            response.message = f"Error: {str(e)}"
        
        return response

    def destroy_node(self):
        if self.vis is not None:
            self.vis.destroy_window()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    tmp_dir = "/home/yilong/ObAct_realworld/tmp"

    # Set up argument parser
    parser = argparse.ArgumentParser(description="TSDF Mapper with Perception")
    parser.add_argument(
        '--text_prompt',
        type=str,
        required=True,
        help="Text prompt for object perception"
    )
    args = parser.parse_args()

    # Ensure temporary directory is clean
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # Initialize Perception
    perception = Perception(output_dir=tmp_dir, text_prompt=args.text_prompt, ik_solver=ik_solver)

    try:
        viewpoints, bbox = perception.process()
        if viewpoints:
            print(f"Processed {len(viewpoints)} viewpoints: {viewpoints}")
    except Exception as e:
        perception.get_logger().error(f"Error during perception processing: {e}")
    finally:
        perception.destroy_node()

    # Set DISPLAY environment variable if not already set
    os.environ.setdefault("DISPLAY", ":0")

    # Initialize TSDFMapper
    mapper = TSDFMapper(
        bbox=bbox,
        sampled_viewpoints=viewpoints,
        use_depth=True,
        silent=False,
        save_data=False,
        update_frequency=10.0
    )

    try:
        rclpy.spin(mapper)  # Let ROS 2 handle the event loop
    except KeyboardInterrupt:
        mapper.log_info("Shutting down...")
    except Exception as e:
        mapper.log_error(f"Unexpected error: {e}")
    finally:
        mapper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()