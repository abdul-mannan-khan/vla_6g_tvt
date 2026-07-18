#!/usr/bin/env python3
"""Launch VLA-6G 3D demo: standalone demo node + RViz."""

import os
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('vla_6g_tvt')

    # Snap env cleanup for RViz
    snap_env_vars = [
        'GTK_EXE_PREFIX', 'GTK_PATH', 'GTK_IM_MODULE_FILE',
        'GIO_MODULE_DIR', 'GSETTINGS_SCHEMA_DIR',
    ]
    clean_env = {var: '' for var in snap_env_vars if var in os.environ}

    demo_node = Node(
        package='vla_6g_tvt',
        executable='rviz_3d_demo.py',
        name='rviz_3d_demo',
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_share, 'config', 'vla_3d_demo.rviz'])],
        additional_env=clean_env,
    )

    return LaunchDescription([demo_node, rviz_node])
