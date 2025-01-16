cmake_minimum_required(VERSION 3.8)
project(torch_interface_node)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# Include Torch
# Assuming LibTorch is installed in /usr/local/libtorch
# Adjust the path as necessary
#set(Torch_DIR "/usr/ali/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "/usr/ali/libtorch")
set(Torch_DIR "/usr/ali/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)


include_directories(
        include
)

add_executable(torch_inference_node src/torch_inference_node.cpp)

ament_target_dependencies(torch_inference_node rclcpp)

# Link against Torch libraries
target_link_libraries(torch_inference_node "${TORCH_LIBRARIES}")

# For C++11 or higher
set_property(TARGET torch_inference_node PROPERTY CXX_STANDARD 14)

install(TARGETS
        torch_inference_node
        DESTINATION lib/${PROJECT_NAME}
)

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
