#!/usr/bin/env python3
"""
Launch file for VLA Training Data Collection
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Arguments
    num_episodes_arg = DeclareLaunchArgument(
        'num_episodes',
        default_value='50',
        description='Number of episodes to collect'
    )

    samples_per_episode_arg = DeclareLaunchArgument(
        'samples_per_episode',
        default_value='100',
        description='Samples per episode'
    )

    # Config
    pkg_share = FindPackageShare('vla_6g_tvt')
    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'vla_6g_params.yaml'
    ])

    # Data collector node
    data_collector_node = Node(
        package='vla_6g_tvt',
        executable='data_collector.py',
        name='data_collector',
        parameters=[
            config_file,
            {
                'num_episodes': LaunchConfiguration('num_episodes'),
                'samples_per_episode': LaunchConfiguration('samples_per_episode')
            }
        ],
        output='screen'
    )

    return LaunchDescription([
        num_episodes_arg,
        samples_per_episode_arg,
        data_collector_node,
    ])
