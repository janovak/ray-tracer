#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "color.cuh"
#include "hittable.cuh"
#include "ray.cuh"

class Material {
  public:
    __device__ virtual bool Scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const = 0;
};

class Lambertian : public Material {
  public:
    __device__ Lambertian(const Color& albedo) : m_albedo(albedo) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& hit_record, Color& attenuation, Ray& scattered, curandState* rand_state) const override;

  private:
    Color m_albedo;
};

class Metal : public Material {
  public:
    __device__ Metal(const Color& albedo, float fuzz) : m_albedo(albedo), m_fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& hit_record, Color& attenuation, Ray& scattered, curandState* rand_state) const override;

  private:
    Color m_albedo;
    float m_fuzz;
};

class Dielectric : public Material {
  public:
    __device__ Dielectric(float refraction_index) : m_refraction_index(refraction_index) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& hit_record, Color& attenuation, Ray& scattered, curandState* rand_state) const override;

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float m_refraction_index;

    static __device__ float Reflectance(float cosine, float refraction_index);
};