#pragma once

#include <cuda_runtime.h>

#include "hittable.cuh"
#include "interval.cuh"
#include "material.cuh"
#include "vec3.cuh"
#include "ray.cuh"

struct Sphere : public Hittable {
    Point3 m_center;
    float m_radius;

    __device__ Sphere(const Point3& center, float radius, Material* material)
        : m_center(center), m_radius(fmaxf(0,radius)), Hittable(material) {}

    __device__ bool Hit(const Ray& ray, Interval ray_t, HitRecord& hit_record) const override;
};