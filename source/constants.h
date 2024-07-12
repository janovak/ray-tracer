#pragma once

#include <limits>

__device__ constexpr float kInfinity = std::numeric_limits<float>::infinity();
__device__ constexpr float kPi = 3.1415926535897932385;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * kPi / 180.0f;
}