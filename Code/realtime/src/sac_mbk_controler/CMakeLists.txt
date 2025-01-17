cmake_minimum_required(VERSION 3.8)
project(sac_mbk_controler)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mbk_pid_controler_interface REQUIRED)

# Set the correct paths for Torch
set(CMAKE_PREFIX_PATH "/home/ali/libtorch")
set(Torch_DIR "/home/ali/libtorch/share/cmake/Torch")

# Ensure the path is correct
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "Torch_DIR: ${Torch_DIR}")

# Find Torch
find_package(Torch REQUIRED)

# Include directories (if needed)
include_directories(
    include
)

add_executable(controller_service_node_cpp src/sac_mbk_controler.cpp)

ament_target_dependencies(controller_service_node_cpp rclcpp mbk_pid_controler_interface Torch)

install(TARGETS
    controller_service_node_cpp
    DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
