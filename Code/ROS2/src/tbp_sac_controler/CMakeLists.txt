cmake_minimum_required(VERSION 3.8)
project(tbp_sac_controler)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tbp_interface REQUIRED)

# Set the correct path for Torch
set(CMAKE_PREFIX_PATH "/home/ali/libtorch")
set(Torch_DIR "/home/ali/libtorch/share/cmake/Torch")

# Find Torch
find_package(Torch REQUIRED)

# Include directories (if needed)
include_directories(
    include
    ${tbp_interface_INCLUDE_DIRS}  # Ensure the service headers are included
)

# Define the executable
add_executable(sac_controler_node src/sac_tbp_controler.cpp
                                  src/src/ModelLocator.cpp)

# Link dependencies to the executable
ament_target_dependencies(sac_controler_node
    rclcpp
    tbp_interface  # Add the service dependency
    Torch  # Link against Torch
)

# Link Torch libraries
target_link_libraries(sac_controler_node "${TORCH_LIBRARIES}")

# Specify C++11 or higher
set_property(TARGET sac_controler_node PROPERTY CXX_STANDARD 14)

# Install the executable
install(TARGETS
    sac_controler_node
    DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip linter checks
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
