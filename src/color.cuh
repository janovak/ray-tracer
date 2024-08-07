#pragma once

#include "vec3.cuh"

class Color : public Vec3Base<Color> {
  public:
    __host__ __device__ Color() : Vec3Base() {}
    __host__ __device__ Color(float e0, float e1, float e2) : Vec3Base(e0, e1, e2) {}

    inline __host__ __device__ float R() const { return e[0]; }
    inline __host__ __device__ float B() const { return e[1]; }
    inline __host__ __device__ float G() const { return e[2]; }

    static __host__ __device__ Color LinearToGamma(const Color& color);
};

inline __host__ std::ostream& operator<<(std::ostream& out, const Color& color) {
    // Translate the [0,1] component values to the byte range [0,255].
    Color scaled = color * 255.99f;
    return out << static_cast<int>(scaled.R()) << ' ' << static_cast<int>(scaled.B()) << ' ' << static_cast<int>(scaled.G()) << '\n';
}