cmake_minimum_required(VERSION 3.18)
project(ray_tracer LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the CUDA version
find_package(CUDA 12.5 REQUIRED)

# Specify the source files
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/main.cu
)

# Specify the output directory for the binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bld)

# Add the executable target
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

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