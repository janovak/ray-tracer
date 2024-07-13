#pragma once

#include <iostream>

#include "vec3.h"

struct Color : public Vec3Base<Color> {
    __host__ __device__ Color() : Vec3Base() {}
    __host__ __device__ Color(float e0, float e1, float e2) : Vec3Base(e0, e1, e2) {}

    __host__ __device__ float R() const { return e[0]; }
    __host__ __device__ float B() const { return e[1]; }
    __host__ __device__ float G() const { return e[2]; }
};

__host__ std::ostream& operator<<(std::ostream& out, const Color& color) {
    // Translate the [0,1] component values to the byte range [0,255].
    Color scaled = color * 255.99;
    return out << static_cast<int>(scaled.R()) << ' ' << static_cast<int>(scaled.B()) << ' ' << static_cast<int>(scaled.G()) << '\n';
}