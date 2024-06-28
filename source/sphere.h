#pragma once

#include <cuda_runtime.h>

#include "hittable.h"
#include "point3.h"
#include "vec3.h"
#include "ray.h"

class sphere : public hittable {
  public:
    sphere(const point3& center, double radius) : center(center), radius(fmax(0,radius)) {}

    bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        // a, b, and c in the quadratic formula are equal to a, -2h, and c below
        double a = r.direction().length_squared();
        double h = dot(r.direction(), oc);
        double c = oc.length_squared() - radius*radius;

        double discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        double sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        double root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;

        return true;
    }

  private:
    point3 center;
    double radius;
};