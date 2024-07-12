#pragma once

#include <iostream>

#include "vec3.h"

class Color : public Vec3Base<Color> {
  public:
    __host__ __device__ Color() : Vec3Base() {}
    __host__ __device__ Color(double e0, double e1, double e2) : Vec3Base(e0, e1, e2) {}

    __host__ __device__ double R() const { return e[0]; }
    __host__ __device__ double B() const { return e[1]; }
    __host__ __device__ double G() const { return e[2]; }
};

__host__ std::ostream& operator<<(std::ostream& out, const Color& color) {
    // Translate the [0,1] component values to the byte range [0,255].
    Color scaled = color * 255.99;
    return out << static_cast<int>(scaled.R()) << ' ' << static_cast<int>(scaled.B()) << ' ' << static_cast<int>(scaled.G()) << '\n';
}