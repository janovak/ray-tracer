#pragma once

#include <limits>

__device__ constexpr double kInfinity = std::numeric_limits<double>::infinity();
__device__ constexpr double kPi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * kPi / 180.0;
}