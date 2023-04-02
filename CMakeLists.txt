cmake_minimum_required(VERSION 2.8.12)

project(yagap LANGUAGES CXX CUDA)

include(GNUInstallDirs)

set(CUDA_PATH /opt/nvidia/cuda/12.1.0/targets/x86_64-linux)

set(CMAKE_MODULE_PATH "${CUDA_PATH}/lib/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH "${CUDA_PATH}/lib/cmake")

add_subdirectory("src")
