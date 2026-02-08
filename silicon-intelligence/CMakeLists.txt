cmake_minimum_required(VERSION 3.20)
project(SiliconIntelligence VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 REQUIRED)

# Set common include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp/core)

# Add subdirectories
add_subdirectory(cpp/core)
add_subdirectory(cpp/bindings)
