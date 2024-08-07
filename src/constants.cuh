#pragma once

#include <limits>

#include <cuda_runtime.h>

__device__ constexpr float kInfinity = std::numeric_limits<float>::infinity();
__device__ constexpr float kPi = 3.1415926535897932385;