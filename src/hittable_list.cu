#include "hittable_list.cuh"

__device__ bool HittableList::Hit(const Ray& ray, Interval ray_t, HitRecord& rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.m_max;

    for (unsigned int i = 0; i < m_size; ++i) {
        if (m_list[i]->Hit(ray, Interval(ray_t.m_min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.m_t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}