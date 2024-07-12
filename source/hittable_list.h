#pragma once

#include "hittable.h"

#include <vector>

struct hittable_list : public hittable {
    __host__ __device__  hittable_list() {}
	__host__ __device__  hittable_list(hittable **list, int size): m_list(list), m_size(size) {}

    __host__ __device__ bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        for (unsigned int i = 0; i < m_size; ++i) {
            if (m_list[i]->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    hittable** m_list;
    unsigned int m_size = 0;
};