cmake_minimum_required(VERSION 3.8)
project(sac_mbk_controler)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

set(CMAKE_INSTALL_RPATH "home/ali/libtorch/lib")

# Ensure that RPATH is used for installed targets
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Include Torch
# Assuming LibTorch is installed in /usr/local/libtorch
# Adjust the path as necessary
#set(Torch_DIR "/usr/ali/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "/home/ali/libtorch")
set(Torch_DIR "/home/ali/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)


include_directories(
        include
)

add_executable(sac_controler_node src/sac_mbk_controler.cpp)


# find dependencies
ament_target_dependencies(sac_controler_node rclcpp mbk_pid_controler_interface)

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
