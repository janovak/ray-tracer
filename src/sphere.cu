#include <cuda_runtime.h>

#include "hittable.cuh"
#include "interval.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "vec3.cuh"
#include "ray.cuh"

__device__ bool Sphere::Hit(const Ray& ray, Interval ray_t, HitRecord& hit_record) const {
    Vec3 oc = m_center - ray.Origin();
    // a, b, and c in the quadratic formula are equal to a, -2h, and c below
    float a = ray.Direction().LengthSquared();
    float half_b = Vec3::Dot(ray.Direction(), oc);
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