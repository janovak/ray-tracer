#pragma once

#include "vec3.h"

struct Point3 : public Vec3Base<Point3> {
    __host__ __device__ Point3() : Vec3Base() {}
    __host__ __device__ Point3(double e0, double e1, double e2) : Vec3Base(e0, e1, e2) {}
};

__host__ __device__ Point3 operator-(const Point3& u, const Vec3& v) {
    return Point3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__host__ __device__ Vec3 operator-(const Point3& u, const Point3& v) {
    return Vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__host__ __device__ Point3 operator+(const Point3& u, const Vec3& v) {
    return Point3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}