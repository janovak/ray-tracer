#pragma once

#include <iostream>

#include "vec3.h"

class color : public vec3_base<color> {
  public:
    __host__ __device__ color() : vec3_base() {}
    __host__ __device__ color(double e0, double e1, double e2) : vec3_base(e0, e1, e2) {}

    __host__ __device__ double r() const { return e[0]; }
    __host__ __device__ double b() const { return e[1]; }
    __host__ __device__ double g() const { return e[2]; }
};

__host__ std::ostream& operator<<(std::ostream& out, const color& c) {
    // Translate the [0,1] component values to the byte range [0,255].
    color scaled = c * 255.99;
    return out << static_cast<int>(scaled.r()) << ' ' << static_cast<int>(scaled.b()) << ' ' << static_cast<int>(scaled.g()) << '\n';
}