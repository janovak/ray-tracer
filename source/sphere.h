#pragma once

#include <cuda_runtime.h>

#include "hittable.h"
#include "point3.h"
#include "vec3.h"
#include "ray.h"

class sphere : public hittable {
  public:
    __host__ __device__ sphere(const point3& center, double radius) : m_center(center), m_radius(fmaxf(0,radius)) {
    }

    __host__ __device__ bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
        vec3 oc = m_center - r.origin();
        // a, b, and c in the quadratic formula are equal to a, -2h, and c below
        double a = r.direction().length_squared();
        double h = dot(oc, r.direction());
        double c = oc.length_squared() - m_radius * m_radius;

        double discriminant = h * h - a * c;
        if (discriminant < 0) {
            return false;
        }

        double sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        double root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root) {
                return false;
            }
        }

        rec.t = root;
        rec.point = r.at(rec.t);
        vec3 outward_normal = (rec.point - m_center) / m_radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }

  private:
    point3 m_center;
    double m_radius;
};