#pragma once

#include "vec3.cuh"

struct Ray {
    Point3 m_origin;
    Vec3 m_direction;

    __device__ Ray() {}
    __device__ Ray(const Point3& origin, const Vec3& direction) : m_origin(origin), m_direction(direction) {}

    inline __device__ const Point3& Origin() const  { return m_origin; }
    inline __device__ const Vec3& Direction() const { return m_direction; }

    inline __device__ Point3 At(float t) const {
        return m_origin + t * m_direction;
    }
};

__device__ Ray operator+(const Ray& u, const Vec3& v);
__device__ Ray operator-(const Ray& u, const Vec3& v);
