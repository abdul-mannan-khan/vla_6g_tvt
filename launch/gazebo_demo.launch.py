#!/usr/bin/env python3
"""Launch VLA-6G Gazebo demo: gz sim + ros_gz_bridge + control node + RViz."""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('vla_6g_tvt')
    pkg_share_sub = FindPackageShare('vla_6g_tvt')
    world_file = os.path.join(pkg_share, 'worlds', 'vla_6g_demo.world')
    bridge_config = os.path.join(pkg_share, 'config', 'gz_bridge.yaml')

    # Snap env cleanup
    snap_env_vars = [
        'GTK_EXE_PREFIX', 'GTK_PATH', 'GTK_IM_MODULE_FILE',
        'GIO_MODULE_DIR', 'GSETTINGS_SCHEMA_DIR',
    ]
    clean_env = {var: '' for var in snap_env_vars if var in os.environ}
    # Force Mesa to use hardware GL (not Zink/Vulkan which fails on some setups)
    clean_env['MESA_LOADER_DRIVER_OVERRIDE'] = 'iris'
    clean_env['__EGL_VENDOR_LIBRARY_FILENAMES'] = ''

    # Gazebo sim with GUI
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', world_file, '-r'],
        output='screen',
        additional_env=clean_env,
    )

    # ros_gz_bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_config}'],
        output='screen',
    )

    # Gazebo demo control node
    demo_node = Node(
        package='vla_6g_tvt',
        executable='gazebo_demo_node.py',
        name='gazebo_demo_node',
        output='screen',
    )

    # RViz for marker visualization alongside Gazebo
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_share_sub, 'config', 'vla_3d_demo.rviz'])],
        additional_env=clean_env,
    )

    return LaunchDescription([gz_sim, bridge, demo_node, rviz_node])
