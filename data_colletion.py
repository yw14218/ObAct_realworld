#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import message_filters
from config.config import D405_RGB_TOPIC_NAME, D405_DEPTH_TOPIC_NAME
from threading import Lock
import h5py
import numpy as np
import os
from datetime import datetime
from utils import ik_solver

class DataCollector(Node):
    def __init__(self, use_depth=False, silent=False, dataset_dir="datasets"):
        super().__init__('data_collector')
        
        self.bridge = CvBridge()
        self.lock = Lock()
        self.silent = silent
        
        # Store latest data
        self.images = {
            "rgb": None,
            "depth": None
        }
        self.joint_states = None  # For arm_1 (main recording)
        self.camera_pose = None   # For arm_2 (camera pose)
        self.use_depth = use_depth
        
        # Dataset storage setup
        self.dataset_dir = dataset_dir
        self.folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_cnt = 0
        os.makedirs(f"{self.dataset_dir}/{self.folder_name}", exist_ok=True)
        
        # Define camera keys based on use_depth
        self.camera_keys = ["rgb"]
        if self.use_depth:
            self.camera_keys.append("depth")
        
        # Subscribers for main recording (arm_1)
        self.rgb_sub = message_filters.Subscriber(self, Image, D405_RGB_TOPIC_NAME)
        self.joint_sub = message_filters.Subscriber(self, JointState, '/arm_1/joint_states')
        
        if self.use_depth:
            self.depth_sub = message_filters.Subscriber(self, Image, D405_DEPTH_TOPIC_NAME)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub, self.joint_sub], 
                queue_size=10, 
                slop=0.033
            )
        else:
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.joint_sub], 
                queue_size=10, 
                slop=0.1
            )
        self.ts.registerCallback(self.synced_callback)
        
        # Subscriber for camera pose (arm_2)
        self.camera_pose_sub = self.create_subscription(
            JointState, '/vx300s/joint_states', self.camera_pose_callback, 10
        )

    def log_info(self, message):
        """Log info messages only if not in silent mode."""
        if not self.silent:
            self.get_logger().info(message)

    def log_warn(self, message):
        """Always log warnings, even in silent mode."""
        self.get_logger().warn(message)
            
    def log_error(self, message):
        """Always log errors, even in silent mode."""
        self.get_logger().error(message)

    def synced_callback(self, rgb_msg, *args):
        """Callback for synchronized messages (arm_1)."""
        with self.lock:
            try:
                self.images["rgb"] = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                self.log_info("RGB Image received and stored.")
                
                if self.use_depth:
                    depth_msg, joint_msg = args
                    self.images["depth"] = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
                    self.log_info("Depth Image received and stored.")
                else:
                    joint_msg = args[0]
                
                self.joint_states = joint_msg
                self.log_info("Joint States (arm_1) received and stored.")
                
            except Exception as e:
                self.log_error(f"Error in synced_callback: {e}")

    def camera_pose_callback(self, msg):
        """Callback for arm_2 joint states (camera pose)."""
        with self.lock:
            self.camera_pose = JointState(
                header=msg.header,
                name=msg.name[:],
                position=msg.position[:],
                velocity=msg.velocity[:],
                effort=msg.effort[:]
            )
            # self.log_info("Camera pose (arm_2) received and stored.")

    def observe(self, timeout=5.0):
        """Get the latest synchronized data (arm_1).
        
        Returns:
            tuple: (RGB image, depth image, joint states) or (None, None, None) if timeout occurs
        """
        self.log_info("Observe called, waiting for synchronized data...")
        
        start_time = self.get_clock().now()
        timeout_duration = rclpy.duration.Duration(seconds=timeout)
        
        while rclpy.ok():
            elapsed = self.get_clock().now() - start_time
            if elapsed > timeout_duration:
                self.log_warn(f"Timeout after {timeout} seconds while waiting for data.")
                return (None, None, None)
            
            rclpy.spin_once(self, timeout_sec=0.1)
            
            with self.lock:
                if self.images["rgb"] is not None and self.joint_states is not None and \
                   (not self.use_depth or self.images["depth"] is not None):
                    rgb_copy = self.images["rgb"].copy() if self.images["rgb"] is not None else None
                    depth_copy = self.images["depth"].copy() if self.use_depth and self.images["depth"] is not None else None
                    joint_copy = JointState(
                        header=self.joint_states.header,
                        name=self.joint_states.name[:],
                        position=self.joint_states.position[:],
                        velocity=self.joint_states.velocity[:],
                        effort=self.joint_states.effort[:]
                    )
                    
                    self.images["rgb"] = None
                    if self.use_depth:
                        self.images["depth"] = None
                    self.joint_states = None
                    
                    self.log_info("All synchronized data retrieved.")
                    return (rgb_copy, depth_copy, joint_copy)
        
        self.log_error("ROS context is no longer valid")
        return (None, None, None)
    
    def save_episode(self, rgb, depth, joints):
        """Save the collected data to an HDF5 file."""
        cam_images = {"rgb": rgb}
        if self.use_depth:
            cam_images["depth"] = depth
        
        states = np.array(joints.position)
        ee_pose = ik_solver.fk(states[:6])

        with h5py.File(f"{self.dataset_dir}/{self.folder_name}/episode_{self.episode_cnt}.h5", "w") as f:
            for key in self.camera_keys:
                f.create_dataset(f"/observations/images/{key}", data=np.array(cam_images[key]))
            f.create_dataset("/observations/qpos", data=states)
            f.create_dataset("/observations/ee_pose", data=ee_pose)
            if joints.velocity:
                f.create_dataset("/observations/qvel", data=np.array(joints.velocity))
            if joints.effort:
                f.create_dataset("/observations/effort", data=np.array(joints.effort))
            # Save camera pose (arm_2 joint states) once per episode
            if self.camera_pose is not None:
                f.create_dataset("/camera_pose/qpos", data=np.array(self.camera_pose.position))
                if self.camera_pose.velocity:
                    f.create_dataset("/camera_pose/qvel", data=np.array(self.camera_pose.velocity))
                if self.camera_pose.effort:
                    f.create_dataset("/camera_pose/effort", data=np.array(self.camera_pose.effort))
        
        self.log_info(f"Saved episode {self.episode_cnt} to HDF5 file.")
        self.episode_cnt += 1

    def run(self):
        """Run the two-stage data collection process."""
        # Stage 1: Wait for user to move the camera
        self.log_info("Stage 1: Please move the camera to the desired position. Type 'ok' when ready.")
        while True:
            user_input = input("Enter 'ok' to proceed: ").strip().lower()
            if user_input == "ok":
                break
            self.log_info("Waiting for 'ok'...")
            rclpy.spin_once(self, timeout_sec=0.1)

        # Stage 2: Record camera pose from arm_2 once
        self.log_info("Waiting for camera pose from /arm_2/joint_states...")
        while self.camera_pose is None and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.camera_pose is not None:
            self.log_info(f"Camera pose recorded - Joint positions: {self.camera_pose.position}")

        # Wait for user to start recording
        self.log_info("Type 'start' to begin recording episodes.")
        while True:
            user_input = input("Enter 'start' to begin recording: ").strip().lower()
            if user_input == "start":
                break
            self.log_info("Waiting for 'start'...")
            rclpy.spin_once(self, timeout_sec=0.1)

        # Main recording loop
        self.log_info("Starting data collection...")
        while rclpy.ok():
            rgb, depth, joints = self.observe(timeout=5.0)
            if rgb is not None and joints is not None:
                self.log_info(f"Collected data - Joint positions (arm_1): {joints.position}")
                self.save_episode(rgb, depth, joints)
            rclpy.spin_once(self)

def main():
    rclpy.init()
    collector = DataCollector(use_depth=True, silent=False)
    try:
        collector.run()
    except KeyboardInterrupt:
        collector.log_info("Data collection stopped by user.")
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()