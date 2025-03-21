import os
import subprocess
import time
import threading
from typing import Optional, List, Tuple

import cv2
import numpy as np
import open3d as o3d
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped

from config.config import *
from tsdf_torch import TSDFVolume
from ik_solver import InverseKinematicsSolver
import shutil
from rclpy.executors import MultiThreadedExecutor

class ViewSampler:
    @staticmethod
    def build_rotation_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
        z_axis = (center - eye) / np.linalg.norm(center - eye)
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((-x_axis, -y_axis, z_axis))

    @staticmethod
    def spherical_to_cartesian(r: float, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=-1)

    def generate_hemisphere_points_with_orientations(self, radius: float = 0.1, num_points: int = 128) -> np.ndarray:
        self.center = self.bbox.reshape(2, 3).mean(axis=0)
        theta_samples = int(np.sqrt(num_points / 2))
        phi_samples = num_points // theta_samples
        
        theta_vals = np.linspace(0, np.pi / 2 - 1e-6, theta_samples)
        phi_vals = np.linspace(0, 2 * np.pi, phi_samples, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
        
        eye_positions = self.center + self.spherical_to_cartesian(radius, theta_grid.ravel(), phi_grid.ravel())
        default_up = np.array([0, 0, 1])
        rotation_matrices = np.array([self.build_rotation_matrix(pos, self.center, default_up) for pos in eye_positions])
        
        sampling_data = np.array(list(zip(eye_positions, rotation_matrices)), 
                                dtype=[('position', float, (3,)), ('rotation', float, (3, 3))])
        sampling_data = sampling_data[sampling_data['position'][:, 2] > 0.1]
        print(f"Generated {len(sampling_data)} viewpoints")
        return sampling_data

class Perception(Node):
    def __init__(self, output_dir: str):
        super().__init__('Perception')
        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.view_sampler = ViewSampler()
        self.ik_solver = InverseKinematicsSolver()

        self.intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, 
                                                              D405_INTRINSIC[0, 0], D405_INTRINSIC[1, 1], 
                                                              D405_INTRINSIC[0, 2], D405_INTRINSIC[1, 2])
        self.tsdf = TSDFVolume(TSDF_SIZE, TSDF_DIM)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.running = True
        self.last_update_time = 0.0

        # Thread-safe image storage
        self.lock = threading.Lock()
        self.images = {"rgb": None, "depth": None}
        self.new_rgb_received = False
        self.new_depth_received = False

        # Subscriptions
        self.rgb_sub = self.create_subscription(Image, "camera/camera/color/image_rect_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, "camera/camera/aligned_depth_to_color/image_raw", self.depth_callback, 10)

        # Visualization setup for point cloud
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("TSDF Point Cloud Visualization", width=800, height=600)
        self.pcd_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_vis)
        self.vis_running = True
        self.vis_thread = threading.Thread(target=self.visualize_tsdf_point_cloud, daemon=True)
        self.vis_thread.start()

    def rgb_callback(self, msg: Image) -> None:
        with self.lock:
            try:
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.new_rgb_received = True
            except Exception as e:
                self.get_logger().error(f"RGB callback error: {e}")

    def depth_callback(self, msg: Image) -> None:
        with self.lock:
            try:
                self.images["depth"] = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                self.new_depth_received = True
            except Exception as e:
                self.get_logger().error(f"Depth callback error: {e}")

    def observe(self, timeout: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            self.new_rgb_received = False
            self.new_depth_received = False
        
        start_time = self.get_clock().now()
        timeout_duration = rclpy.duration.Duration(seconds=timeout)
        
        # Wait for transform availability
        while not self.tf_buffer.can_transform("vx300s/base_link", "camera_color_optical_frame", rclpy.time.Time(), 
                                            timeout=rclpy.duration.Duration(seconds=0.1)):
            if (self.get_clock().now() - start_time) > timeout_duration:
                self.get_logger().warn("Timeout waiting for TF transform")
                return None, None, None
            rclpy.spin_once(self, timeout_sec=0.05)
        
        while rclpy.ok():
            if (self.get_clock().now() - start_time) > timeout_duration:
                self.get_logger().warn(f"Timeout after {timeout} seconds waiting for images")
                return None, None, None
            
            rclpy.spin_once(self, timeout_sec=0.05)
            
            with self.lock:
                if self.new_rgb_received and self.new_depth_received:
                    rgb_copy = self.images["rgb"].copy()
                    depth_copy = self.images["depth"].copy()
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            "vx300s/base_link", 
                            "camera_color_optical_frame", 
                            rclpy.time.Time(), 
                            rclpy.duration.Duration(seconds=2.0)
                        )
                        T_cam_to_robot = self.transform_to_matrix(transform)
                        self.get_logger().info("TF lookup successful")
                        return rgb_copy, depth_copy, T_cam_to_robot
                    except tf2_ros.LookupException as e:
                        self.get_logger().error(f"TF lookup failed unexpectedly: {e}")
                        return None, None, None
        
        return None, None, None

    def update_tsdf(self, rgb_image: np.ndarray, depth_image: np.ndarray, T_cam_to_robot: np.ndarray) -> None:
        if time.time() - self.last_update_time < 0.1:
            return
        self.last_update_time = time.time()

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image), o3d.geometry.Image(depth_image.astype(np.uint16)),
            depth_scale=1000.0, depth_trunc=0.7, convert_rgb_to_intensity=False
        )
        self.tsdf._volume.integrate(rgbd, self.intrinsic_o3d, np.linalg.inv(T_cam_to_robot))

    def visualize_tsdf_point_cloud(self) -> None:
        while self.vis_running:
            if self.tsdf:
                point_cloud = self.tsdf._volume.extract_point_cloud()
                if point_cloud is not None:
                    with self.lock:  # Protect point cloud update
                        self.pcd_vis.points = point_cloud.points
                        self.pcd_vis.colors = point_cloud.colors
                    self.vis.update_geometry(self.pcd_vis)
                    self.vis.poll_events()
                    self.vis.update_renderer()
            time.sleep(0.05)  # Update at 20 Hz for smoother real-time visualization

    def run(self) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        rgb_image, depth_image, T_cam_to_robot = self.observe()
        if rgb_image is None or depth_image is None or T_cam_to_robot is None:
            return None

        self.update_tsdf(rgb_image, depth_image, T_cam_to_robot)
        rgb_path, depth_path = self._capture_and_save_images(rgb_image, depth_image)
        if not rgb_path or not depth_path:
            return None

        seg_mask = self._run_grounded_sam(rgb_path)
        if seg_mask is None:
            raise RuntimeError("Segmentation failed")

        mask_path = os.path.join(self.output_dir, "mask_0.png")
        processor = PointCloudProcessor(rgb_path, depth_path, mask_path, self)
        self.pcd_in_robot = processor.pcd.transform(T_cam_to_robot)
        np.save(os.path.join(self.output_dir, "pose_0.npy"), T_cam_to_robot)

        obb = processor.pcd.get_oriented_bounding_box()
        bbox = np.asarray(obb.get_box_points())
        np.savetxt(os.path.join(self.output_dir, "bbox.txt"), bbox)
        self.view_sampler.bbox = np.concatenate([bbox.min(axis=0), bbox.max(axis=0)])

        viewpoints = self.view_sampler.generate_hemisphere_points_with_orientations(0.3, NUMBER_OF_VIEW_SAMPLES)
        filtered_viewpoints, joint_poses = self._filter_viewpoints_with_ik(viewpoints)
        return filtered_viewpoints, joint_poses

    def _capture_and_save_images(self, rgb_image: np.ndarray, depth_image: np.ndarray, subfix: str = "0") -> Tuple[Optional[str], Optional[str]]:
        if rgb_image is None or depth_image is None:
            self.get_logger().error("No images to save")
            return None, None

        rgb_path = os.path.join(self.output_dir, f"rgb_{subfix}.png")
        depth_path = os.path.join(self.output_dir, f"depth_{subfix}.npy")
        os.makedirs(self.output_dir, exist_ok=True)

        PILImage.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)).save(rgb_path)
        np.save(depth_path, depth_image)
        return rgb_path, depth_path

    def _run_grounded_sam(self, input_image_path: str) -> Optional[np.ndarray]:
        command = [
            "python3", "grounded_sam_demo.py", "--config", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "--grounded_checkpoint", "groundingdino_swint_ogc.pth", "--sam_checkpoint", "sam_vit_h_4b8939.pth",
            "--input_image", input_image_path, "--output_dir", self.output_dir, "--box_threshold", "0.3",
            "--text_threshold", "0.25", "--text_prompt", "green mug", "--device", "cuda"
        ]
        try:
            subprocess.run(command, check=True, cwd=os.path.expanduser("~/Grounded-Segment-Anything"))
            mask_path = os.path.join(self.output_dir, "mask_0.png")
            if not os.path.exists(mask_path):
                self.get_logger().error("Segmentation mask not found")
                return None
            return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Grounded SAM failed: {e}")
            return None

    def _filter_viewpoints_with_ik(self, viewpoints: np.ndarray) -> Tuple[List[dict], List[dict]]:
        filtered_viewpoints, eef_poses = [], []
        for vp in viewpoints:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = vp['rotation'], vp['position']
            T_eef_robot = T @ np.linalg.inv(D405_HANDEYE)
            position, quat_xyzw = T_eef_robot[:3, 3], Rotation.from_matrix(T_eef_robot[:3, :3]).as_quat()
            if self.ik_solver.compute_ik(position, quat_xyzw) is not None:
                filtered_viewpoints.append(vp)
                eef_poses.append({'position': position, 'quat_xyzw': quat_xyzw})
        print(f"Filtered viewpoints count: {len(filtered_viewpoints)}")
        return filtered_viewpoints, eef_poses

    def transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        trans, rot = transform.transform.translation, transform.transform.rotation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        T[:3, 3] = [trans.x, trans.y, trans.z]
        return T

    def destroy_node(self) -> None:
        self.running = False
        self.vis_running = False
        self.vis_thread.join(timeout=2.0)
        self.vis.destroy_window()
        super().destroy_node()

class PointCloudProcessor:
    def __init__(self, rgb_path: str, depth_path: str, mask_path: str, node: Node):
        self.node = node
        self.rgb_image = np.array(PILImage.open(rgb_path))
        if self.rgb_image.shape[-1] == 4:
            self.rgb_image = self.rgb_image[..., :3]
        self.rgb_image = self.rgb_image.astype(np.uint8)
        self.depth_image = np.load(depth_path).astype(np.float32) * (np.array(PILImage.open(mask_path)) > 0)
        self._process_point_cloud()

    def _process_point_cloud(self) -> None:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(self.rgb_image), o3d.geometry.Image(self.depth_image),
            depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, D405_INTRINSIC[0, 0], D405_INTRINSIC[1, 1], 
                                                      D405_INTRINSIC[0, 2], D405_INTRINSIC[1, 2])
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.005).remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]

def main(args=None) -> None:
    rclpy.init(args=args)
    tmp_dir = "/home/yilong/ObAct_realworld/tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    
    perception = Perception(output_dir=tmp_dir)
    
    try:
        viewpoints, eef_poses = perception.run()
        
        # if viewpoints:
        #     for i, pose in enumerate(eef_poses[:4]):
        #         perception.ik_solver.move_to_pose(pose['position'], pose['quat_xyzw'])
        #         time.sleep(3.0)
        #         rgb_image, depth_image, T_cam_to_robot = perception.observe()
        #         if rgb_image is None or depth_image is None:
        #             perception.get_logger().error("Failed to capture images after move")
        #             break
        #         perception.update_tsdf(rgb_image, depth_image, T_cam_to_robot)
        #         perception._capture_and_save_images(rgb_image, depth_image, subfix=f"{i+1}")
        #         np.save(os.path.join(tmp_dir, f"pose_{i+1}.npy"), T_cam_to_robot)
        
    except Exception as e:
        perception.get_logger().error(f"Error at line {__import__('sys').exc_info()[2].tb_lineno}: {e}")
    finally:
        perception.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
