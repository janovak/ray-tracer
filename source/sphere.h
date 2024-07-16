#pragma once

#include <cuda_runtime.h>

#include "hittable.h"
#include "material.h"
#include "point3.h"
#include "vec3.h"
#include "ray.h"

struct Sphere : public Hittable {
    Point3 m_center;
    float m_radius;

    __host__ __device__ Sphere(const Point3& center, float radius, Material* material) : m_center(center), m_radius(fmaxf(0,radius)), Hittable(material) {}

    __host__ __device__ bool Hit(const Ray& ray, Interval ray_t, HitRecord& hit_record) const override {
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

        hit_record.m_t = root;
        hit_record.m_point = ray.At(hit_record.m_t);
        Vec3 outward_normal = (hit_record.m_point - m_center) / m_radius;
        hit_record.SetFaceNormal(ray, outward_normal);
        hit_record.m_material = m_material;

        return true;
    }
};