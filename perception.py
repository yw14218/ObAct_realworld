import os
import subprocess
import shutil
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped

from config.config import (
    WIDTH, HEIGHT, D405_INTRINSIC, D405_HANDEYE, NUMBER_OF_VIEW_SAMPLES,
    D405_RGB_TOPIC_NAME, D405_DEPTH_TOPIC_NAME
)
from tsdf_torch import ViewSampler


class Perception(Node):
    def __init__(
        self,
        output_dir: str,
        text_prompt: str,
        radius: float = 0.3,
        base_frame: str = "vx300s/base_link",
        camera_frame: str = "camera_color_optical_frame",
        ik_solver=None
    ):
        """Initialize the Perception node with configurable parameters."""
        super().__init__('perception')
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.text_prompt = text_prompt
        self.radius = radius
        self.rgb_topic = D405_RGB_TOPIC_NAME
        self.depth_topic = D405_DEPTH_TOPIC_NAME
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.num_viewpoints = NUMBER_OF_VIEW_SAMPLES
        self.ik_solver = ik_solver
        self.view_sampler = ViewSampler()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics for point cloud processing
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            WIDTH, HEIGHT, D405_INTRINSIC[0, 0], D405_INTRINSIC[1, 1],
            D405_INTRINSIC[0, 2], D405_INTRINSIC[1, 2]
        )

    def _wait_for_message(self, topic: str, msg_type, timeout: float = 5.0) -> Optional[any]:
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

    def transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        """Convert a TransformStamped to a 4x4 transformation matrix."""
        trans, rot = transform.transform.translation, transform.transform.rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        T[:3, 3] = [trans.x, trans.y, trans.z]
        return T

    def observe(self, timeout: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture RGB, depth, and transform data synchronously."""
        transform_timeout = rclpy.duration.Duration(seconds=timeout)
        start_time = self.get_clock().now()
        while not self.tf_buffer.can_transform(self.base_frame, self.camera_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)):
            if not rclpy.ok() or (self.get_clock().now() - start_time) > transform_timeout:
                self.get_logger().warn("Timeout waiting for TF transform")
                return None, None, None
            rclpy.spin_once(self, timeout_sec=0.05)

        rgb_msg = self._wait_for_message(self.rgb_topic, Image, timeout)
        depth_msg = self._wait_for_message(self.depth_topic, Image, timeout)
        if rgb_msg is None or depth_msg is None:
            self.get_logger().error("Failed to receive images")
            return None, None, None

        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        try:
            transform = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, rclpy.time.Time(), transform_timeout)
            return rgb_image, depth_image, self.transform_to_matrix(transform)
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None, None, None

    def _save_images(self, rgb_image: np.ndarray, depth_image: np.ndarray, suffix: str = "0") -> Tuple[Optional[str], Optional[str]]:
        """Save RGB and depth images to disk."""
        if rgb_image is None or depth_image is None:
            self.get_logger().error("No images to save")
            return None, None

        os.makedirs(self.output_dir, exist_ok=True)
        rgb_path = os.path.join(self.output_dir, f"rgb_{suffix}.png")
        depth_path = os.path.join(self.output_dir, f"depth_{suffix}.npy")

        PILImage.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)).save(rgb_path)
        np.save(depth_path, depth_image)
        return rgb_path, depth_path

    def _run_segmentation(self, rgb_path: str) -> Optional[np.ndarray]:
        """Run Grounded SAM for object segmentation."""
        command = [
            "python3", "grounded_sam_demo.py",
            "--config", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "--grounded_checkpoint", "groundingdino_swint_ogc.pth",
            "--sam_checkpoint", "sam_vit_h_4b8939.pth",
            "--input_image", rgb_path,
            "--output_dir", self.output_dir,
            "--box_threshold", "0.3",
            "--text_threshold", "0.25",
            "--text_prompt", self.text_prompt,
            "--device", "cuda"
        ]
        try:
            subprocess.run(command, check=True, cwd=os.path.expanduser("~/Grounded-Segment-Anything"))
            mask_path = os.path.join(self.output_dir, "mask_0.png")
            if not os.path.exists(mask_path):
                self.get_logger().error("Segmentation mask not found")
                return None
            return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Segmentation failed: {e}")
            return None

    def _process_point_cloud(self, rgb_path: str, depth_path: str, mask: np.ndarray, T_cam_to_robot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate and transform point cloud from RGB, depth, and mask."""
        rgb_image = np.array(PILImage.open(rgb_path))
        if rgb_image.shape[-1] == 4:
            rgb_image = rgb_image[..., :3]
        rgb_image = rgb_image.astype(np.uint8)
        depth_image = np.load(depth_path).astype(np.float32) * mask

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image), o3d.geometry.Image(depth_image),
            depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd = pcd.voxel_down_sample(voxel_size=0.005).remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
        pcd.transform(T_cam_to_robot)

        obb = pcd.get_oriented_bounding_box()
        bbox = np.asarray(obb.get_box_points())
        return pcd.points, bbox

    def _filter_viewpoints(self, viewpoints: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Filter viewpoints using IK if an IK solver is provided."""
        if not self.ik_solver:
            self.get_logger().warn("No IK solver provided; returning unfiltered viewpoints")
            return [{'position': vp['position'], 'rotation': vp['rotation']} for vp in viewpoints]

        filtered = []
        for vp in viewpoints:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = vp['rotation'], vp['position']
            T_eef_robot = T @ np.linalg.inv(D405_HANDEYE)
            position, quat_xyzw = T_eef_robot[:3, 3], Rotation.from_matrix(T_eef_robot[:3, :3]).as_quat()
            if self.ik_solver.compute_ik(position, quat_xyzw) is not None:
                filtered.append({'position': position, 'rotation': T_eef_robot[:3, :3]})
        self.get_logger().info(f"Filtered to {len(filtered)} IK-feasible viewpoints")
        return filtered

    def process(self, timeout: float = 5.0) -> Optional[List[Dict[str, np.ndarray]]]:
        """Main processing pipeline: capture, segment, generate point cloud, and compute viewpoints."""
        # Capture data
        rgb_image, depth_image, T_cam_to_robot = self.observe(timeout)
        if rgb_image is None or depth_image is None or T_cam_to_robot is None:
            return None

        # Save images
        rgb_path, depth_path = self._save_images(rgb_image, depth_image)
        if not rgb_path or not depth_path:
            return None

        # Segment object
        mask = self._run_segmentation(rgb_path)
        if mask is None:
            self.get_logger().error("Segmentation failed")
            return None

        # Process point cloud and get bounding box
        mask_path = os.path.join(self.output_dir, "mask_0.png")
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)  # Save mask for point cloud processing
        points, bbox = self._process_point_cloud(rgb_path, depth_path, mask, T_cam_to_robot)
        np.save(os.path.join(self.output_dir, "pose_0.npy"), T_cam_to_robot)
        np.savetxt(os.path.join(self.output_dir, "bbox.txt"), bbox)

        # Pass bbox to view_sampler and generate viewpoints
        self.view_sampler.bbox = np.concatenate([bbox.min(axis=0), bbox.max(axis=0)])
        viewpoints = self.view_sampler.generate_hemisphere_points_with_orientations(self.radius, self.num_viewpoints)
        return self._filter_viewpoints(viewpoints), bbox

    def destroy_node(self) -> None:
        """Clean up resources."""
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    tmp_dir = "/home/yilong/ObAct_realworld/tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # Example usage with optional IK solver
    from ik_solver import InverseKinematicsSolver  # Assuming this exists
    perception = Perception(output_dir=tmp_dir, text_prompt="green mug", ik_solver=InverseKinematicsSolver())

    try:
        viewpoints, bbox = perception.process()
        if viewpoints:
            print(f"Processed {len(viewpoints)} viewpoints: {viewpoints[:2]}")  # Print first two for brevity
    except Exception as e:
        perception.get_logger().error(f"Error: {e}")
    finally:
        perception.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()