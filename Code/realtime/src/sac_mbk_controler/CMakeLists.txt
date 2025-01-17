cmake_minimum_required(VERSION 3.8)
project(sac_mbk_controler)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mbk_pid_controler_interface REQUIRED)  # Make sure to find the service package
find_package(Torch REQUIRED)

set(CMAKE_INSTALL_RPATH "/home/ali/libtorch/lib")

# Ensure that RPATH is used for installed targets
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Set the correct path for Torch
set(CMAKE_PREFIX_PATH "/home/ali/libtorch")
set(Torch_DIR "/home/ali/libtorch/share/cmake/Torch")

# Find Torch
find_package(Torch REQUIRED)

# Include directories
include_directories(
    include
    ${mbk_pid_controler_interface_INCLUDE_DIRS}  # Ensure service headers are included
)

add_executable(sac_controler_node src/sac_mbk_controler.cpp)

# Link against required dependencies
ament_target_dependencies(sac_controler_node
    rclcpp
    mbk_pid_controler_interface  # Include the service package
    Torch  # Link against Torch libraries
)

# Link against Torch libraries
target_link_libraries(sac_controler_node "${TORCH_LIBRARIES}")

# For C++11 or higher
set_property(TARGET sac_controler_node PROPERTY CXX_STANDARD 14)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip linter checks
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
