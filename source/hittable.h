#pragma once

#include "ray.h"

struct hit_record {
    // outward_normal is assumed to have unit length
    __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }

    bool front_face;
    double t;
    point3 point;
    vec3 normal;
};

class hittable {
public:
    __host__ __device__ virtual bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const = 0;
};
