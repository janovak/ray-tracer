#pragma once

#include <iostream>

#include "vec3.h"

class color : public vec3_base<color> {
  public:
    __host__ __device__ color() : vec3_base() {}
    __host__ __device__ color(double e0, double e1, double e2) : vec3_base(e0, e1, e2) {}
};

inline std::ostream& operator<<(std::ostream& out, const color& c) {
    // Translate the [0,1] component values to the byte range [0,255].
    color scaled = c * 255.99;
    return out << static_cast<int>(scaled.e[0]) << ' ' << static_cast<int>(scaled.e[1]) << ' ' << static_cast<int>(scaled.e[2]) << '\n';
}