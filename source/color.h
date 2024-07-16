#pragma once

#include <cmath>
#include <iostream>

#include "color.h"
#include "vec3.h"

class Color : public Vec3Base<Color> {
  public:
    __host__ __device__ Color() : Vec3Base() {}
    __host__ __device__ Color(float e0, float e1, float e2) : Vec3Base(e0, e1, e2) {}

    __host__ __device__ float R() const { return e[0]; }
    __host__ __device__ float B() const { return e[1]; }
    __host__ __device__ float G() const { return e[2]; }
};

__host__ __device__ Color LinearToGamma(const Color& color) {
    Color gamma_corrected;

    auto linear_to_gamma_component = [](float linear_component) -> float {
        if (linear_component > 0) {
            return sqrt(linear_component);
        }
        return 0;
    };

    for (unsigned int i = 0; i < 3; ++i) {
        gamma_corrected[i] = linear_to_gamma_component(color[i]);
    }

    return gamma_corrected;
}

__host__ std::ostream& operator<<(std::ostream& out, const Color& color) {
    // Gamma correct the pixel's color and translate the [0,1] component values to the byte range [0,255].
    Color scaled = LinearToGamma(color) * 255.99;
    return out << static_cast<int>(scaled.R()) << ' ' << static_cast<int>(scaled.B()) << ' ' << static_cast<int>(scaled.G()) << '\n';
}