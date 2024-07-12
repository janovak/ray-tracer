#pragma once

#include "point3.h"
#include "vec3.h"

struct Ray {
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point3& origin, const Vec3& direction) : m_origin(origin), m_direction(direction) {}

    __host__ __device__ const Point3& Origin() const  { return m_origin; }
    __host__ __device__ const Vec3& Direction() const { return m_direction; }

    __host__ __device__ Point3 At(double t) const {
        return m_origin + t * m_direction;
    }

    Point3 m_origin;
    Vec3 m_direction;
};