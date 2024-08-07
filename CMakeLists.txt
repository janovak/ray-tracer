cmake_minimum_required(VERSION 3.18)
project(ray_tracer LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA version
find_package(CUDA 12.5 REQUIRED)

# Specify the source files
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/camera.cu
    ${PROJECT_SOURCE_DIR}/src/color.cu
    ${PROJECT_SOURCE_DIR}/src/hittable.cu
    ${PROJECT_SOURCE_DIR}/src/hittable_list.cu
    ${PROJECT_SOURCE_DIR}/src/interval.cu
    ${PROJECT_SOURCE_DIR}/src/main.cu
    ${PROJECT_SOURCE_DIR}/src/material.cu
    ${PROJECT_SOURCE_DIR}/src/ray.cu
    ${PROJECT_SOURCE_DIR}/src/ray_tracer.cu
    ${PROJECT_SOURCE_DIR}/src/sphere.cu
    ${PROJECT_SOURCE_DIR}/src/vec3.cu
)

# Specify the header files
set(HEADER_FILES
    ${PROJECT_SOURCE_DIR}/src/camera.cuh
    ${PROJECT_SOURCE_DIR}/src/color.cuh
    ${PROJECT_SOURCE_DIR}/src/hittable.cuh
    ${PROJECT_SOURCE_DIR}/src/hittable_list.cuh
    ${PROJECT_SOURCE_DIR}/src/interval.cuh
    ${PROJECT_SOURCE_DIR}/src/material.cuh
    ${PROJECT_SOURCE_DIR}/src/ray.cuh
    ${PROJECT_SOURCE_DIR}/src/ray_tracer.cuh
    ${PROJECT_SOURCE_DIR}/src/sphere.cuh
    ${PROJECT_SOURCE_DIR}/src/vec3.cuh
)

# Specify the output directory for the binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bld)

# Add the executable target
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})

# Set properties for the CUDA compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MDd")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options /MDd")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /NODEFAULTLIB:libcmtd.lib")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /NODEFAULTLIB:libcmt.lib")