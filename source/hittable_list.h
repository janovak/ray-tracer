#pragma once

#include <vector>

#include "hittable.h"

struct HittableList : public Hittable {
    __host__ __device__  HittableList() {}
    __host__ __device__  HittableList(Hittable **list, int size): m_list(list), m_size(size) {}

    __host__ __device__ bool Hit(const Ray& ray, float ray_tmin, float ray_tmax, HitRecord& rec) const override {
        HitRecord temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        for (unsigned int i = 0; i < m_size; ++i) {
            if (m_list[i]->Hit(ray, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.m_t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    Hittable** m_list;
    unsigned int m_size = 0;
};