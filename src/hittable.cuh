#pragma once

#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

class Material;

struct HitRecord {
    bool m_front_face;
    float m_t;
    Point3 m_point;
    Vec3 m_normal;
    Material* m_material;

    // outward_normal is assumed to have unit length
    __device__ void SetFaceNormal(const Ray& ray, const Vec3& outward_normal);

    __device__ Ray RandomOnHemisphere(curandState* rand_state);
};

struct Hittable {
    Material* m_material;

    __host__ __device__ Hittable() {}
    __host__ __device__ Hittable(Material* material) : m_material(material) {}

    __device__ virtual bool Hit(const Ray& ray, Interval ray_t, HitRecord& rec) const = 0;
};
