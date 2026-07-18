#!/usr/bin/env python3
"""
Launch file for VLA-6G UAV Relay System

This launches:
1. Channel simulator
2. Ground users simulator
3. VLA relay node
4. (EGO Planner should be launched separately)
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='analytical',
        description='VLA model type: analytical, llama, vla'
    )

    num_users_arg = DeclareLaunchArgument(
        'num_users',
        default_value='5',
        description='Number of ground users'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz visualization'
    )

    # Get package share directory
    pkg_share = FindPackageShare('vla_6g_tvt')

    # Config file path
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'vla_6g_params.yaml'
    ])

    # Channel simulator node
    channel_simulator_node = Node(
        package='vla_6g_tvt',
        executable='channel_simulator_node',
        name='channel_simulator',
        parameters=[config_file],
        output='screen'
    )

    # Users simulator node
    users_simulator_node = Node(
        package='vla_6g_tvt',
        executable='users_simulator_node',
        name='users_simulator',
        parameters=[config_file],
        output='screen'
    )

    # VLA relay node
    vla_relay_node = Node(
        package='vla_6g_tvt',
        executable='vla_relay_node.py',
        name='vla_relay_node',
        parameters=[
            config_file,
            {'model_type': LaunchConfiguration('model_type')}
        ],
        output='screen'
    )

    # RViz (optional)
    # Sanitize environment: VS Code snap injects library paths that crash RViz
    snap_env_vars = [
        'GTK_EXE_PREFIX', 'GTK_PATH', 'GTK_IM_MODULE_FILE',
        'GIO_MODULE_DIR', 'GSETTINGS_SCHEMA_DIR',
    ]
    clean_env = {var: '' for var in snap_env_vars if var in os.environ}

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_share, 'config', 'vla_6g.rviz'])],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        additional_env=clean_env,
    )

    return LaunchDescription([
        model_type_arg,
        num_users_arg,
        use_rviz_arg,
        channel_simulator_node,
        users_simulator_node,
        vla_relay_node,
        rviz_node,
    ])
