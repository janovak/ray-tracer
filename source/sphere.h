#pragma once

#include <cuda_runtime.h>

#include "hittable.h"
#include "point3.h"
#include "vec3.h"
#include "ray.h"

struct Sphere : public Hittable {
    __host__ __device__ Sphere(const Point3& center, float radius) : m_center(center), m_radius(fmaxf(0,radius)) {}

    __host__ __device__ bool Hit(const Ray& ray, Interval ray_t, HitRecord& rec) const override {
        Vec3 oc = m_center - ray.Origin();
        // a, B, and c in the quadratic formula are equal to a, -2h, and c below
        float a = ray.Direction().LengthSquared();
        float h = Dot(oc, ray.Direction());
        float c = oc.LengthSquared() - m_radius * m_radius;

        float discriminant = h * h - a * c;
        if (discriminant < 0) {
            return false;
        }

        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (h - sqrtd) / a;
        if (!ray_t.Surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.Surrounds(root)) {
                return false;
            }
        }

        rec.m_t = root;
        rec.m_point = ray.At(rec.m_t);
        Vec3 outward_normal = (rec.m_point - m_center) / m_radius;
        rec.SetFaceNormal(ray, outward_normal);

        return true;
    }

    Point3 m_center;
    float m_radius;
};