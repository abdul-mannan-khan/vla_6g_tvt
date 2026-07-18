#!/usr/bin/env python3
"""
Integrated Launch File: VLA-6G + EGO Planner Simulation

This launches the complete system:
1. EGO Planner simulation (UAV dynamics, trajectory planning)
2. VLA-6G components (channel simulator, users, VLA relay node)
3. RViz visualization (EGO Planner's existing RViz)

The VLA relay node publishes targets to /move_base_simple/goal
EGO Planner receives these and plans trajectories to reach them.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ========== Launch Arguments ==========
    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='analytical',
        description='VLA model type: analytical, llama'
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz visualization'
    )

    # EGO Planner map size (100m x 100m for 6G scenario)
    map_size_x_arg = DeclareLaunchArgument('map_size_x', default_value='100.0')
    map_size_y_arg = DeclareLaunchArgument('map_size_y', default_value='100.0')
    map_size_z_arg = DeclareLaunchArgument('map_size_z', default_value='50.0')

    # UAV initial position
    init_x_arg = DeclareLaunchArgument('init_x', default_value='25.0')
    init_y_arg = DeclareLaunchArgument('init_y', default_value='25.0')
    init_z_arg = DeclareLaunchArgument('init_z', default_value='20.0')

    # ========== Package Paths ==========
    vla_pkg_share = FindPackageShare('vla_6g_tvt')
    ego_pkg_share = get_package_share_directory('ego_planner')

    vla_config = PathJoinSubstitution([vla_pkg_share, 'config', 'vla_6g_params.yaml'])

    # ========== EGO Planner Advanced Parameters ==========
    # flight_type: 1 = MANUAL_GOAL (accepts goals from /move_base_simple/goal)
    ego_advanced_params = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ego_pkg_share, 'launch', 'advanced_param.launch.py')
        ),
        launch_arguments={
            'drone_id': '0',
            'map_size_x_': LaunchConfiguration('map_size_x'),
            'map_size_y_': LaunchConfiguration('map_size_y'),
            'map_size_z_': LaunchConfiguration('map_size_z'),
            'odometry_topic': 'visual_slam/odom',
            'camera_pose_topic': 'pcl_render_node/camera_pose',
            'depth_topic': 'pcl_render_node/depth',
            'cloud_topic': 'pcl_render_node/cloud',
            'cx': '321.04638671875',
            'cy': '243.44969177246094',
            'fx': '387.229248046875',
            'fy': '387.229248046875',
            'max_vel': '4.0',
            'max_acc': '3.0',
            'planning_horizon': '7.5',
            'flight_type': '1',  # MANUAL_GOAL - accepts VLA targets
            'point_num': '0',
        }.items()
    )

    # ========== EGO Planner Simulator ==========
    ego_simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ego_pkg_share, 'launch', 'simulator.launch.py')
        ),
        launch_arguments={
            'drone_id': '0',
            'map_size_x_': LaunchConfiguration('map_size_x'),
            'map_size_y_': LaunchConfiguration('map_size_y'),
            'map_size_z_': LaunchConfiguration('map_size_z'),
            'init_x_': LaunchConfiguration('init_x'),
            'init_y_': LaunchConfiguration('init_y'),
            'init_z_': LaunchConfiguration('init_z'),
            'odometry_topic': 'visual_slam/odom',
        }.items()
    )

    # ========== EGO Planner Trajectory Server ==========
    traj_server = Node(
        package='ego_planner',
        executable='traj_server',
        name='drone_0_traj_server',
        output='screen',
        remappings=[
            ('position_cmd', 'drone_0_planning/pos_cmd'),
            ('planning/bspline', 'drone_0_planning/bspline')
        ],
        parameters=[{'traj_server/time_forward': 1.0}]
    )

    # ========== VLA-6G Channel Simulator ==========
    channel_simulator = Node(
        package='vla_6g_tvt',
        executable='channel_simulator_node',
        name='channel_simulator',
        parameters=[vla_config],
        output='screen',
        # Remap odom to use EGO Planner's odometry
        remappings=[
            ('odom_world', 'visual_slam/odom')
        ]
    )

    # ========== VLA-6G Users Simulator ==========
    users_simulator = Node(
        package='vla_6g_tvt',
        executable='users_simulator_node',
        name='users_simulator',
        parameters=[vla_config],
        output='screen'
    )

    # ========== VLA Relay Node ==========
    vla_relay = Node(
        package='vla_6g_tvt',
        executable='vla_relay_node.py',
        name='vla_relay_node',
        parameters=[
            vla_config,
            {'model_type': LaunchConfiguration('model_type')}
        ],
        output='screen'
    )

    # ========== Static TF: world frame ==========
    # EGO Planner's simulator uses /world as frame_id but doesn't publish TF.
    # RViz needs a TF tree to render anything.
    static_tf_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'quadrotor'],
    )

    # ========== RViz ==========
    # Sanitize environment: VS Code snap injects library paths that crash RViz
    # with "undefined symbol: __libc_pthread_init" from snap's libpthread.
    snap_env_vars = [
        'GTK_EXE_PREFIX', 'GTK_PATH', 'GTK_IM_MODULE_FILE',
        'GIO_MODULE_DIR', 'GSETTINGS_SCHEMA_DIR',
    ]
    clean_env = {var: '' for var in snap_env_vars if var in os.environ}

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([vla_pkg_share, 'config', 'vla_6g.rviz'])],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        additional_env=clean_env,
    )

    # ========== Build Launch Description ==========
    return LaunchDescription([
        # Arguments
        model_type_arg,
        use_rviz_arg,
        map_size_x_arg,
        map_size_y_arg,
        map_size_z_arg,
        init_x_arg,
        init_y_arg,
        init_z_arg,

        # EGO Planner components
        ego_advanced_params,
        ego_simulator,
        traj_server,

        # VLA-6G components
        channel_simulator,
        users_simulator,
        vla_relay,

        # TF and Visualization
        static_tf_world,
        rviz,
    ])
