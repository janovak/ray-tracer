#pragma once

#include "vec3.h"

class point3 : public vec3_base<point3> {
  public:
    __host__ __device__ point3() : vec3_base() {}
    __host__ __device__ point3(double e0, double e1, double e2) : vec3_base(e0, e1, e2) {}
};

__host__ __device__ point3 operator-(const point3& u, const vec3& v) {
    return point3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__host__ __device__ vec3 operator-(const point3& u, const point3& v) {
    return vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__host__ __device__ point3 operator+(const point3& u, const vec3& v) {
    return point3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}