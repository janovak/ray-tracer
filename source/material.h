#pragma once

#include <curand_kernel.h>

#include "color.h"
#include "hittable.h"
#include "ray.h"

class Material {
  public:
    __device__ virtual bool Scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const = 0;
};

class Lambertian : public Material {
  public:
    __device__ Lambertian(const Color& albedo) : m_albedo(albedo) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& hit_record, Color& attenuation, Ray& scattered, curandState* rand_state) const override {
        Vec3 scatter_direction = hit_record.m_normal + RandomUnitVector(rand_state);
        
        // Catch degenerate scatter direction
        if (scatter_direction.NearZero()) {
            scatter_direction = hit_record.m_normal;
        }

        scattered = Ray(hit_record.m_point, scatter_direction);
        attenuation = m_albedo;
        return true;
    }

  private:
    Color m_albedo;
};