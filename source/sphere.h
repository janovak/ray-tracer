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
/*         Vec3 oc = r.Origin() - m_center;
        float a = Dot(r.Direction(), r.Direction());
        float b = Dot(oc, r.Direction());
        float c = Dot(oc, oc) - m_radius*m_radius;
        float discriminant = b*b - a*c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant))/a;
            if (temp < ray_t.m_max && temp > ray_t.m_min) {
                rec.m_t = temp;
                rec.m_point = r.At(rec.m_t);
                rec.m_normal = (rec.m_point - m_center) / m_radius;
                rec.m_material = m_material;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < ray_t.m_max && temp > ray_t.m_min) {
                rec.m_t = temp;
                rec.m_point = r.At(rec.m_t);
                rec.m_normal = (rec.m_point - m_center) / m_radius;
                rec.m_material = m_material;
                return true;
            }
        }
        return false; */


        Vec3 oc = m_center - ray.Origin();
        // a, b, and c in the quadratic formula are equal to a, -2h, and c below
        float a = ray.Direction().LengthSquared();
        float half_b = Dot(ray.Direction(), oc);
        float c = oc.LengthSquared() - m_radius * m_radius;

        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0) {
            return false;
        }

        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (half_b - sqrtd) / a;
        if (!ray_t.Surrounds(root)) {
            root = (half_b + sqrtd) / a;
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