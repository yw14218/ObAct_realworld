import open3d as o3d
import numpy as np
import os
import glob
import cv2
import time
import torch
from config.config import HEIGHT, WIDTH, TSDF_SIZE, TSDF_DIM as TSDF_RESOLUTION, NUMBER_OF_VIEW_SAMPLES as NUM_SAMPLED_VIEWS, D405_HANDEYE
# Constants
DEFAULT_DATA_FOLDER = "data"
DEFAULT_INTRINSIC_FILE = "config/d405_intrinsic_right.npy"
VOXEL_DOWN_SIZE = 0.005
CAMERA_SCALE = 0.03
TOP_N = 5
RAY_BATCH_SIZE = int(WIDTH * HEIGHT * 0.2)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ViewSampler:
    @staticmethod
    def build_rotation_matrix(eye, center, up):
        eye = np.asarray(eye)
        center = np.asarray(center)
        up = np.asarray(up)
        
        z_axis = center - eye
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            up = np.array([0, 1, 0]) if abs(np.dot(z_axis, [0, 0, 1])) > 0.99 else np.array([0, 0, 1])
            x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        rotation = np.eye(3)
        rotation[:, 0] = -x_axis
        rotation[:, 1] = -y_axis
        rotation[:, 2] = z_axis
        
        return rotation
    
    @staticmethod
    def spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def generate_hemisphere_points_with_orientations(self, radius=0.1, num_points=128):
        self.center = self.bbox.reshape(2, 3).mean(axis=0)
        theta_samples = int(np.sqrt(num_points / 2))
        phi_samples = int(num_points / theta_samples)
        
        theta_max = np.pi / 2 - 1e-6
        theta_vals = np.linspace(0, theta_max, theta_samples)
        phi_vals = np.linspace(0, 2 * np.pi, phi_samples, endpoint=False)
        
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
        theta_flat = theta_grid.ravel()
        phi_flat = phi_grid.ravel()
        
        x = radius * np.sin(theta_flat) * np.cos(phi_flat)
        y = radius * np.sin(theta_flat) * np.sin(phi_flat)
        z = radius * np.cos(theta_flat)
        eye_positions = self.center + np.stack([x, y, z], axis=-1)
        
        default_up = np.array([0, 0, 1])
        rotation_matrices = np.array([
            self.build_rotation_matrix(pos, self.center, default_up)
            for pos in eye_positions
        ])
        
        sampling_data = np.array(
            list(zip(eye_positions, rotation_matrices)),
            dtype=[('position', float, (3,)), ('rotation', float, (3, 3))]
        )
    
        # Filter out viewpoints that are too low
        sampling_data = sampling_data[sampling_data['position'][:, 2] > 0.15]
        
        # Remove duplicate viewpoints
        _, unique_indices = np.unique(sampling_data['position'], axis=0, return_index=True)
        sampling_data = sampling_data[np.sort(unique_indices)]
    
        print(f"Generated {len(sampling_data)} viewpoints")
    
        return sampling_data

class TSDFVolume:
    def __init__(self, resolution, bbox, sdf_trunc_factor=4, padding=0.01):
        """
        bbox: numpy array of shape (6,) or (2, 3), where:
              - If (6,), it's [min_x, min_y, min_z, max_x, max_y, max_z]
              - If (2, 3), it's [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        """
        bbox = np.array(bbox)
        if bbox.shape == (6,):
            min_bound = bbox[:3]
            max_bound = bbox[3:]
        elif bbox.shape == (2, 3):
            min_bound = bbox[0]
            max_bound = bbox[1]
        else:
            raise ValueError("bbox must be shape (6,) or (2, 3)")

        # IMPORTANT: Always add some padding to ensure the TSDF volume extends beyond the object
        # Otherwise rays may not intersect properly for information gain calculation
        padding = max(padding, 0.05)  # Enforce minimum padding of 5cm
        print(f"Using TSDF padding: {padding}m")
        
        min_bound -= padding
        max_bound += padding

        # Store the original bbox for reference
        self.original_bbox_min = min_bound.copy()
        self.original_bbox_max = max_bound.copy()

        bbox_extent = max_bound - min_bound
        cube_size = np.max(bbox_extent)  # make it a cube
        origin = min_bound  # bottom-left-front corner

        self.size = cube_size
        self.origin = origin
        self.resolution = resolution
        self.voxel_size = self.size / self.resolution
        self.sdf_trunc = sdf_trunc_factor * self.voxel_size

        print(f"TSDF Origin: {self.origin}, Size: {self.size}, Voxel: {self.voxel_size}")
        print(f"TSDF Bounds: Min={self.origin}, Max={self.origin + np.array([self.size]*3)}")
        
        # Calculate volume coverage of original bbox
        original_bbox_volume = np.prod(bbox[3:] - bbox[:3])
        tsdf_volume = self.size**3
        volume_ratio = original_bbox_volume / tsdf_volume
        print(f"Object volume / TSDF volume ratio: {volume_ratio:.4f}")

        self._volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.size,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=self.origin
        )


    def integrate(self, rgb_img, depth_img, intrinsic, extrinsic, frame_idx, debug_dir=None):
        rgb_img_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img_filtered = depth_img.astype(np.float32) / 1000.0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_img_rgb),
            o3d.geometry.Image(depth_img_filtered),
            depth_scale=1.0,
            depth_trunc=0.7,
            convert_rgb_to_intensity=False,
        )

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic['width'],
            height=intrinsic['height'],
            fx=intrinsic['fx'],
            fy=intrinsic['fy'],
            cx=intrinsic['cx'],
            cy=intrinsic['cy'],
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic)
        
        if not pcd.has_points():
            print(f"Frame {frame_idx}: No points generated from RGBD image")
        else:
            print(f"Frame {frame_idx}: Generated {len(pcd.points)} points")
            if debug_dir:
                o3d.io.write_point_cloud(os.path.join(debug_dir, f"pcd_{frame_idx:04d}.ply"), pcd)

        self._volume.integrate(rgbd, intrinsic_o3d, extrinsic)
        return pcd

    def get_point_cloud(self):
        return self._volume.extract_point_cloud()

    def get_sdf_grid(self):
        voxel_cloud = self._volume.extract_voxel_point_cloud()
        points = np.asarray(voxel_cloud.points)
        
        # In Open3D, the TSDF values are stored in the R channel of the colors
        sdf_values = np.asarray(voxel_cloud.colors)[:, 0]
        
        # Scale from [0,1] to [-1,1] as Open3D stores normalized TSDF values
        # TSDF values: negative = inside object, positive = outside object, 0 = on surface
        sdf_values = 2.0 * sdf_values - 1.0
        
        # Create an empty grid filled with a default "far" value (1.0)
        # This avoids NaN issues during ray casting
        grid = np.ones((self.resolution, self.resolution, self.resolution), dtype=np.float32)
        
        indices = np.floor((points - self.origin) / self.voxel_size).astype(int)
        valid_mask = np.all((indices >= 0) & (indices < self.resolution), axis=1)
        valid_indices = indices[valid_mask]
        
        # Only set values for valid indices
        grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = sdf_values[valid_mask]
        
        # Print statistics about the grid to help with debugging
        non_default_count = np.sum(grid != 1.0)
        percentage_filled = 100.0 * non_default_count / grid.size
        print(f"SDF Grid statistics:")
        print(f"  - Total voxels: {grid.size}")
        print(f"  - Non-default voxels: {non_default_count} ({percentage_filled:.2f}%)")
        print(f"  - Value range: [{np.min(grid):.4f}, {np.max(grid):.4f}]")
        
        # Analyze surface voxels (near zero crossings)
        surface_voxels = np.sum(np.abs(grid - 0) < 0.05)
        if surface_voxels > 0:
            percentage_surface = 100.0 * surface_voxels / non_default_count
            print(f"  - Surface voxels: {surface_voxels} ({percentage_surface:.2f}% of non-default)")
        else:
            print("  - No surface voxels detected (zero-crossings)")
        
        if non_default_count == 0:
            print("WARNING: SDF grid is empty. No voxels have been updated from the default value.")
            print("This will result in zero information gain for all viewpoints.")
        
        return grid

class ViewEvaluator:
    def __init__(self, tsdf_volume, intrinsic, target_bbox):
        self.tsdf = tsdf_volume
        self.intrinsic = intrinsic
        self.target_bbox = torch.tensor(target_bbox, dtype=torch.float32, device=device)
        self.sdf_grid = torch.tensor(self.tsdf.get_sdf_grid(), dtype=torch.float32, device=device)
        self.origin = torch.tensor(self.tsdf.origin, dtype=torch.float32, device=device)
        self.voxel_size = torch.tensor(self.tsdf.voxel_size, dtype=torch.float32, device=device)
        print(f"Initialized ViewEvaluator with TSDF grid shape: {self.sdf_grid.shape}")
        print(f"Target Bounding Box: {self.target_bbox}")

        self.rays_dirs = []
        
    def compute_information_gain(self, pose, width=WIDTH, height=HEIGHT):

        pose = np.linalg.inv(pose)
        position = torch.tensor(pose[:3, 3], dtype=torch.float32, device=device)
        rotation = torch.tensor(pose[:3, :3], dtype=torch.float32, device=device)

        # Generate rays
        u, v = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing='ij')
        u, v = u.flatten(), v.flatten()
        rays_dir = torch.stack([(u - self.intrinsic['cx']) / self.intrinsic['fx'],
                                (v - self.intrinsic['cy']) / self.intrinsic['fy'],
                                torch.ones_like(u, dtype=torch.float32, device=device)], dim=1)
        rays_dir = rays_dir / torch.norm(rays_dir, dim=1, keepdim=True)
        rays_dir = torch.matmul(rotation, rays_dir.T).T

        self.rays_dirs.append(rays_dir)
        G_x = self.compute_gain_gpu(position, rays_dir)

        return G_x.item()

    def compute_gain_gpu(self, position, rays_dir):
        """Compute information gain on GPU using PyTorch, fully vectorized with batching.
        Optimized for occlusion-aware volumetric reconstruction by prioritizing
        uncertain regions and implementing early ray termination."""
        step_size = self.voxel_size
        max_steps = int(self.tsdf.size / step_size.item())
        G_x = torch.tensor(0.0, dtype=torch.float32, device=device)
        uncertainty_weight = torch.tensor(1.0, dtype=torch.float32, device=device)

        num_rays = rays_dir.shape[0]
        t = torch.arange(0, max_steps * step_size.item(), step_size.item(), device=device)  # [max_steps]

        for start_idx in range(0, num_rays, RAY_BATCH_SIZE):
            end_idx = min(start_idx + RAY_BATCH_SIZE, num_rays)
            batch_rays = rays_dir[start_idx:end_idx]
            batch_size = batch_rays.shape[0]

            # Track active rays for early termination
            active_rays = torch.ones(batch_size, dtype=torch.bool, device=device)
            ray_contributions = torch.zeros(batch_size, dtype=torch.float32, device=device)
            
            # Process each step sequentially for better memory efficiency
            for step_idx in range(max_steps):
                # Only compute for active rays
                if not active_rays.any():
                    break
                    
                # Current distance along rays
                current_t = t[step_idx]
                
                # Compute current points for active rays
                current_points = position + batch_rays * current_t.unsqueeze(0)
                
                # Convert to voxel indices
                voxel_idx = ((current_points - self.origin) / self.voxel_size).to(torch.int32)
                
                # Check which points are within grid bounds
                valid_points = torch.all(
                    (voxel_idx >= 0) & (voxel_idx < self.tsdf.resolution),
                    dim=1
                ) & active_rays
                
                if not valid_points.any():
                    continue
                
                # Get SDF values for valid points
                valid_voxel_idx = voxel_idx[valid_points]
                sdf_values = torch.full((batch_size,), float('nan'), device=device)
                
                try:
                    sdf_values[valid_points] = self.sdf_grid[
                        valid_voxel_idx[:, 0], valid_voxel_idx[:, 1], valid_voxel_idx[:, 2]
                    ]
                except IndexError:
                    # Handle any out-of-bounds errors safely
                    continue
                
                # Check if points are within target bounding box
                in_bbox = torch.all(
                    (current_points >= self.target_bbox[:3]) & 
                    (current_points <= self.target_bbox[3:]),
                    dim=1
                ) & valid_points
                
                # Calculate uncertainty (highest near zero-crossings)
                # Values near 0 in TSDF represent surfaces
                uncertainty = torch.exp(-10.0 * torch.abs(sdf_values)) * in_bbox
                uncertainty[torch.isnan(uncertainty)] = 0.0
                
                # Accumulate weighted uncertainty for active rays
                ray_contributions += uncertainty * active_rays
                
                # Detect surfaces (values near zero in TSDF)
                surfaces = (torch.abs(sdf_values) < 0.01) & valid_points
                
                # Check for negative values (inside objects)
                inside_object = (sdf_values < 0) & valid_points & ~torch.isnan(sdf_values)
                
                # Terminate rays that hit surfaces
                active_rays = active_rays & ~surfaces
                
                # Also terminate rays that go too far inside objects
                consecutive_inside_steps = (inside_object * active_rays).float()
                if step_idx > 0:
                    # If we're consistently inside for multiple steps, terminate the ray
                    active_rays = active_rays & ~(consecutive_inside_steps > 3)
            
            # Add contributions from this batch
            G_x += ray_contributions.sum()

        # Normalize gain by the number of rays for consistent comparison
        G_x = G_x if num_rays > 0 else torch.tensor(0.0, device=device)

        return G_x

def load_data(folder_path):
    """Load RGB images, depth maps, and poses from folder."""
    rgb_files = sorted(glob.glob(os.path.join(folder_path, "rgb_*.png")))
    depth_files = sorted(glob.glob(os.path.join(folder_path, "depth_*.npy")))
    pose_files = sorted(glob.glob(os.path.join(folder_path, "pose_*.npy")))
    
    return [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in rgb_files], [np.load(f) for f in depth_files], [np.load(f) for f in pose_files]

def load_intrinsics(file_path):
    """Load camera intrinsics from file."""
    K = np.load(file_path)
    return {
        'width': WIDTH,
        'height': HEIGHT,
        'fx': K[0, 0],
        'fy': K[1, 1],
        'cx': K[0, 2],
        'cy': K[1, 2],
    }

def create_camera_visualizations(poses, intrinsic_o3d, colors=None, scales=None):
    """Create camera visualizations with custom colors and scales."""
    colors = colors or [[1, 0, 0]] * len(poses)
    scales = scales or [CAMERA_SCALE] * len(poses)
    return [
        o3d.geometry.LineSet.create_camera_visualization(intrinsic_o3d, pose, scale).paint_uniform_color(color)
        for pose, color, scale in zip(poses, colors, scales)
    ]

def visualize_geometries(geometries, title):
    """Visualize a list of geometries with a title."""
    print(title)
    o3d.visualization.draw_geometries(geometries)

# def render_rgbd_at_viewpoint(pcd, intrinsic_o3d, pose, width=WIDTH, height=HEIGHT, output_dir="rendered_views"):
#     """
#     Render RGB and depth images from a specific viewpoint using Open3D's OffscreenRenderer.
#     Requires a Metal-enabled build of Open3D on macOS.
    
#     Args:
#         pcd: Open3D PointCloud.
#         intrinsic_o3d: Open3D PinholeCameraIntrinsic.
#         pose: 4x4 numpy array representing the world-to-camera extrinsic.
#         width: Rendered image width.
#         height: Rendered image height.
#         output_dir: Directory where rendered images are saved.
    
#     Returns:
#         rgb: Numpy array of the rendered RGB image (float image in [0,1]).
#         depth: Numpy array of the rendered depth image.
#     """

#     # Create an offscreen renderer.
#     renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
#     renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
#     renderer.scene.view.set_post_processing(False)
    
#     # Prepare a material using an unlit shader for the geometry.
#     material = o3d.visualization.rendering.MaterialRecord()
#     material.shader = "defaultUnlit"
    
#     # Clear pre-existing geometry and add the point cloud.
#     renderer.scene.clear_geometry()
#     renderer.scene.add_geometry("pcd", pcd, material)
    
#     # Setup the camera with the provided intrinsic and extrinsic (pose) parameters.
#     # The pose should be a world-to-camera transformation.
#     renderer.setup_camera(intrinsic_o3d, pose)
    
#     # Render the scene to obtain an RGB image and a depth image.
#     rgb_image = renderer.render_to_image()
#     depth_image = renderer.render_to_depth_image(z_in_view_space=True)
    
#     # Convert Open3D Image objects to numpy arrays.
#     rgb = np.asarray(rgb_image)  # May be (H, W, 3) or (H, W, 4)
#     depth = np.asarray(depth_image)  # May be (H, W) or (H, W, 1)
#     if depth.ndim == 3 and depth.shape[-1] == 1:
#         depth = depth[..., 0]
    
#     # Save output images.
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     idx = len(os.listdir(output_dir)) // 2  # assumes 2 files per view.
#     rgb_path = os.path.join(output_dir, f"rgb_{idx:04d}.png")
#     depth_path = os.path.join(output_dir, f"depth_{idx:04d}.npy")
    
#     # Convert the RGB image to uint8 and BGR for cv2.imwrite.
#     cv2.imwrite(rgb_path,(rgb * 255).astype(np.uint8))
#     np.save(depth_path, depth)
    
#     return rgb, depth


def render_rgbd_at_viewpoint(pcd, intrinsic_o3d, pose, width=WIDTH, height=HEIGHT, output_dir="rendered_views"):
    """
    Render RGB and depth images from a specific viewpoint using PyTorch3D.
    """
    # Convert Open3D point cloud to PyTorch3D point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)

    vis.add_geometry(pcd)

    # Update the camera parameters
    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()
    cam_params.intrinsic = intrinsic_o3d
    # If your saved pose is camera-to-world, use its inverse, otherwise adjust as needed.
    cam_params.extrinsic = pose  
    view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

    # Allow time for scene update
    vis.poll_events()
    vis.update_renderer()

    rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    
    vis.destroy_window()

    # Save images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    idx = len(os.listdir(output_dir)) // 2
    rgb_path = os.path.join(output_dir, f"rgb_{idx:04d}.png")
    depth_path = os.path.join(output_dir, f"depth_{idx:04d}.npy")

    cv2.imwrite(rgb_path, cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    np.save(depth_path, depth)

    return rgb, depth

def create_and_visualize_tsdf(folder_path=DEFAULT_DATA_FOLDER, size=TSDF_SIZE, resolution=TSDF_RESOLUTION,
                              intrinsic_file=DEFAULT_INTRINSIC_FILE, invert_poses=True, debug=True,
                              num_sampled_views=NUM_SAMPLED_VIEWS):
    # Load data (unchanged)
    intrinsic = load_intrinsics(intrinsic_file)
    rgb_imgs, depth_imgs, poses = load_data(folder_path)
    
    if not all([rgb_imgs, depth_imgs, poses]):
        print("No data found in the specified folder")
        return

    min_count = min(len(rgb_imgs), len(depth_imgs), len(poses))
    print(f"Data counts - RGB: {len(rgb_imgs)}, Depth: {len(depth_imgs)}, Poses: {len(poses)}")
    print(f"Using first {min_count} complete frames for TSDF reconstruction")

    rgb_imgs, depth_imgs, poses = rgb_imgs[:min_count], depth_imgs[:min_count], poses[:min_count]
    if invert_poses:
        poses = [np.linalg.inv(pose) for pose in poses]

    bbox = np.loadtxt("tmp/bbox.txt")
    bbox = np.concatenate([np.min(bbox, axis=0), np.max(bbox, axis=0)])

    # Initialize TSDF and integrate frames (unchanged)
    tsdf = TSDFVolume(resolution, bbox)
    combined_pcd = o3d.geometry.PointCloud()
    debug_dir = "debug_pcds" if debug else None
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(**intrinsic)
    for i in range(0, len(rgb_imgs), 10):
        print(f"Integrating frame {i+1}/{len(rgb_imgs)}")
        pcd = tsdf.integrate(rgb_imgs[i], depth_imgs[i], intrinsic, poses[i], i + 1, debug_dir)
        combined_pcd += pcd


    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=VOXEL_DOWN_SIZE)
    print(f"Combined Point Cloud Bounds: Min {np.min(combined_pcd.points, axis=0)}, Max {np.max(combined_pcd.points, axis=0)}")

    # Sample viewpoints (unchanged)
    sampler = ViewSampler()
    sampler.bbox = bbox
    sampled_views = sampler.generate_hemisphere_points_with_orientations(radius=0.3, num_points=num_sampled_views)

    sampled_poses = [np.eye(4) for _ in sampled_views]
    for i, view in enumerate(sampled_views):
        sampled_poses[i][:3, :3], sampled_poses[i][:3, 3] = view['rotation'], view['position']
    sampled_poses = [np.linalg.inv(pose) for pose in sampled_poses]

    distinct_poses =[]
    for pose in sampled_poses:
        if not any(np.allclose(pose, p, atol=1e-3) for p in distinct_poses):
            distinct_poses.append(pose)

    sampled_poses = distinct_poses.copy()

    # output_geom = o3d.io.read_point_cloud("tsdf_point_cloud.ply")
    # for i, pose in enumerate(sampled_poses):
    #     rgb, depth = render_rgbd_at_viewpoint(output_geom, intrinsic_o3d, pose, output_dir="rendered_views")
    #     print(f"Rendered view {i + 1}/{len(sampled_poses)}")


    # Compute information gains (unchanged)
    evaluator = ViewEvaluator(tsdf, intrinsic, bbox)
    start_time = time.time()
    gains = [evaluator.compute_information_gain(pose) for pose in sampled_poses]
    print(f"Total gain computation took {time.time() - start_time:.2f} seconds")
    
    gains = np.array(gains)
    print(f"Gains > 0: {gains[gains > 0].shape[0]} views")

    print(gains)
    # Identify top 5 gain views (unchanged)
    top_gain_indices = np.argsort(gains)[-TOP_N:][::-1]
    print(f"Top {TOP_N} Gain Views: {[(i + 1, gains[i]) for i in top_gain_indices]}")

    # Visualize cameras (unchanged)
    original_cameras = create_camera_visualizations(poses, intrinsic_o3d, colors=[[0, 1, 0]] * len(poses))
    
    sampled_colors = [[0, 0, 1]] * len(sampled_poses)
    sampled_scales = [CAMERA_SCALE] * len(sampled_poses)
    for rank, idx in enumerate(top_gain_indices):
        sampled_colors[idx] = [1, 0, 0]
    sampled_cameras = create_camera_visualizations(sampled_poses, intrinsic_o3d, colors=sampled_colors, scales=sampled_scales)


    bbox_geom = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=tsdf.origin,
        max_bound=tsdf.origin + np.array([tsdf.size] * 3))

    bbox_geom.color = (0, 1, 0)

    object_bbox_geom = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[:3], max_bound=bbox[3:])
    object_bbox_geom.color = (1, 0, 0)

    demo_pose = np.load("tasks/mug/ee_pose_ref.npy")
    demo_pose = demo_pose @ D405_HANDEYE
    demo_pose = np.linalg.inv(demo_pose)
    poses = [demo_pose]
    demo_camera = create_camera_visualizations(poses, intrinsic_o3d, colors=[[0, 0, 0]] * len(poses))

    # # Visualize rays for each sampled pose
    # ray_geometries = []
    # for i, pose in enumerate(sampled_poses):
    #     # Get the rays for this pose (last entry in rays_dirs corresponds to this pose)
    #     pose = np.linalg.inv(pose)
    #     if i < len(evaluator.rays_dirs):  # Ensure we have rays
    #         rays_dir = evaluator.rays_dirs[i].cpu().numpy()  # Convert to NumPy
    #         position = np.array(pose[:3, 3])  # Camera position

    #         # Normalize rays and scale for visualization (e.g., scale by 0.1 to make them visible)
    #         rays_dir = rays_dir / np.linalg.norm(rays_dir, axis=1, keepdims=True) * 0.1

    #         # Create start points (all at camera position)
    #         rays_start = np.tile(position, (len(rays_dir), 1))

    #         # Create end points (start + direction)
    #         rays_end = rays_start + rays_dir

    #         # Create Open3D line set
    #         points = np.vstack([rays_start, rays_end])
    #         lines = np.array([[i, i + len(rays_start)] for i in range(len(rays_start))])

    #         ray_lines = o3d.geometry.LineSet()
    #         ray_lines.points = o3d.utility.Vector3dVector(points)
    #         ray_lines.lines = o3d.utility.Vector2iVector(lines)
    #         ray_lines.paint_uniform_color([1, 0, 0])  # Red color for rays

    #         ray_geometries.append(ray_lines)

    # Visualize results with rays
    visualize_geometries(
        [combined_pcd, bbox_geom, object_bbox_geom] + original_cameras + sampled_cameras + demo_camera,
        "Visualizing combined point cloud (no TSDF) with camera poses and rays..."
    )
    visualize_geometries(
        [tsdf.get_point_cloud(), bbox_geom, object_bbox_geom] + original_cameras + sampled_cameras,
        "Visualizing TSDF point cloud with camera poses and rays..."
    )

def main():
    """Entry point for the script."""
    if not os.path.exists(DEFAULT_INTRINSIC_FILE):
        print(f"Error: Intrinsic file {DEFAULT_INTRINSIC_FILE} not found!")
        return
    
    if not os.path.exists(DEFAULT_DATA_FOLDER):
        print(f"Error: Data folder {DEFAULT_DATA_FOLDER} not found!")
        return
    
    create_and_visualize_tsdf()

if __name__ == "__main__":
    main()
