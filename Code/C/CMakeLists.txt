cmake_minimum_required(VERSION 3.30)
project(C_Torch)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option 1: Using CMAKE_PREFIX_PATH
# This helps CMake locate LibTorch without setting Torch_DIR explicitly
# Replace the path below with your actual LibTorch path
set(CMAKE_PREFIX_PATH "/Users/Ali/Documents/BAI/BAI_programming/libtorch")

# Option 2: Explicitly setting Torch_DIR (uncomment if you prefer)
# set(Torch_DIR "/Users/Ali/Documents/BAI/BAI_programming/libtorch/share/cmake/Torch")

# Find the Torch package
find_package(Torch REQUIRED)

# Add the executable
add_executable(C_Torch main.cpp)

# Link Torch libraries to the executable
target_link_libraries(C_Torch "${TORCH_LIBRARIES}")

# (Optional) Specify additional compile options if necessary
# set_property(TARGET C_Torch PROPERTY CXX_STANDARD 20)
