#pragma once

#include "interval.cuh"
#include "ray.cuh"

class Material;

struct HitRecord {
    bool m_front_face;
    float m_t;
    Point3 m_point;
    Vec3 m_normal;
    Material* m_material;

    // outward_normal is assumed to have unit length
    __device__ void SetFaceNormal(const Ray& ray, const Vec3& outward_normal) {
        m_front_face = Dot(ray.Direction(), outward_normal) < 0;
        m_normal = m_front_face ? outward_normal : -outward_normal;
    }

    __device__ Ray RandomOnHemisphere(curandState* rand_state) {
        Vec3 on_unit_sphere = RandomUnitVector(rand_state);
        if (Dot(on_unit_sphere, m_normal) > 0) { // In the same hemisphere as the normal
            return Ray(m_point, on_unit_sphere);
        } else {
            return Ray(m_point, -on_unit_sphere);
        }
    }
};

struct Hittable {
    Material* m_material;

    __host__ __device__ Hittable() {}
    __host__ __device__ Hittable(Material* material) : m_material(material) {}

    __device__ virtual bool Hit(const Ray& ray, Interval ray_t, HitRecord& rec) const = 0;
};
