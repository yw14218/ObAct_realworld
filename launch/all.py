import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    interbotix_xsarm_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('interbotix_xsarm_control'),
                'launch',
                'xsarm_control.launch.py'
            )
        ),
        launch_arguments={'robot_model': 'vx300s'}.items()
    )

    # Launch the moveit launch in a new terminal instead of including it in this launch.
    interbotix_xsarm_moveit_cmd = [
        'ros2', 'launch', 'interbotix_xsarm_moveit', 'xsarm_moveit.launch.py',
        'robot_model:=vx300s',
        'hardware_type:=actual',
        'robot_name:=arm_2'
    ]

    interbotix_xsarm_moveit_terminal = ExecuteProcess(
        cmd=['gnome-terminal', '--'] + interbotix_xsarm_moveit_cmd,
        output='screen'
    )

    # Launch the dual arm launch in a new terminal.
    interbotix_xsarm_dual_arm_cmd = [
        'ros2', 'launch', 'interbotix_xsarm_dual', 'xsarm_dual.launch.py'
    ]

    interbotix_xsarm_dual_arm_terminal = ExecuteProcess(
        cmd=['gnome-terminal', '--'] + interbotix_xsarm_dual_arm_cmd,
        output='screen'
    )

    # Launch the realsense camera launch in a new terminal.
    realsense_camera_cmd = [
        'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
        'depth_module.color_profile:=640x480x30',
        'depth_module.depth_profile:=640x480x30',
        'align_depth.enable:=true',
        'spatial_filter.enable:=true',
        'temporal_filter.enable:=true',
        'device_type:=d405'
    ]

    realsense_camera_terminal = ExecuteProcess(
        cmd=['gnome-terminal', '--'] + realsense_camera_cmd,
        output='screen'
    )

    hand_eye_launch_left = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('interbotix_xsarm_moveit'),
                'launch',
                'handeye_left.launch.py'
            )
        )
    )

    hand_eye_launch_right = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('interbotix_xsarm_moveit'),
                'launch',
                'handeye_left.launch.py'
            )
        )
    )

    return LaunchDescription([
        interbotix_xsarm_dual_arm_terminal,
        # TimerAction(period=3.0, actions=[interbotix_xsarm_moveit_terminal]),
        realsense_camera_terminal,
        hand_eye_launch_left,
        hand_eye_launch_right
    ])

# ros2 launch interbotix_xsarm_moveit xsarm_moveit.launch.py robot_model:=vx300s hardware_type:=actual
# ros2 launch interbotix_xsarm_dual_arm xsarm_dual_arm.launch.py
# ros2 launch realsense2_camera rs_launch.py depth_module.color_profile:=640x480x30 depth_module.depth_profile:=640x480x30 align_depth.enable:=true spatial_filter.enable:=true temporal_filter.enable:=true device_type:=d405
# ros2 launch interbotix_xsarm_launch xsarm_launch.py robot_model:=vx300s robot_name:=arm_2