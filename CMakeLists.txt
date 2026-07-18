cmake_minimum_required(VERSION 3.8)
project(vla_6g_tvt)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ChannelState.msg"
  "msg/VLACommand.msg"
  "msg/UserState.msg"
  "msg/RelayMetrics.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Include directories
include_directories(include)

# Channel simulator node (C++)
add_executable(channel_simulator_node
  src/channel_simulator_node.cpp
)
ament_target_dependencies(channel_simulator_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(channel_simulator_node "${cpp_typesupport_target}")

# Ground users simulator node (C++)
add_executable(users_simulator_node
  src/users_simulator_node.cpp
)
ament_target_dependencies(users_simulator_node
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
)
target_link_libraries(users_simulator_node "${cpp_typesupport_target}")

# Install targets
install(TARGETS
  channel_simulator_node
  users_simulator_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/vla_relay_node.py
  scripts/evaluate_system.py
  scripts/evaluate_mobility.py
  scripts/evaluate_mobility_hybrid.py
  scripts/benchmark_latency.py
  scripts/compute_stats.py
  scripts/train_vla_optimized.py
  scripts/train_eval_mlp.py
  scripts/train_eval_drl.py
  scripts/plot_results_v2.py
  scripts/rviz_3d_demo.py
  scripts/gazebo_demo_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  worlds
  models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
