#pragma once

#include "interval.h"
#include "ray.h"

struct HitRecord {
    bool m_front_face;
    float m_t;
    Point3 m_point;
    Vec3 m_normal;

    // outward_normal is assumed to have unit length
    __host__ __device__ void SetFaceNormal(const Ray& ray, const Vec3& outward_normal) {
        m_front_face = Dot(ray.Direction(), outward_normal) < 0;
        m_normal = m_front_face ? outward_normal : -outward_normal;
    }
};

struct Hittable {
    __host__ __device__ virtual bool Hit(const Ray& ray, Interval ray_t, HitRecord& rec) const = 0;
};
