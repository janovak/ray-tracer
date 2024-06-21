cmake_minimum_required(VERSION 3.18)
project(ray_tracer LANGUAGES CXX CUDA)

# Set the CUDA version
find_package(CUDA 12.5 REQUIRED)

# Specify the include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Specify the source files
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/source/ray_tracer.cu
)

# Specify the output directory for the binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

# Add the executable target
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Set properties for the CUDA compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52"
)
