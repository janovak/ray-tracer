#pragma once

#include "hittable.cuh"
#include "interval.cuh"
#include "ray.cuh"

struct HittableList : public Hittable {
    Hittable** m_list;
    unsigned int m_size = 0;

    __host__ __device__  HittableList() {}
    __host__ __device__  HittableList(Hittable **list, int size): m_list(list), m_size(size) {}

    __device__ bool Hit(const Ray& ray, Interval ray_t, HitRecord& rec) const override;
};