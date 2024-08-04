#pragma once

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

    __device__ bool Scatter(const Ray& r_in, const HitRecord& hit_record, Color& attenuation, Ray& scattered, curandState* rand_state) const override {
        Vec3 scatter_direction = hit_record.m_normal + Vec3::RandomInUnitSphere(rand_state);
        
        // Catch degenerate scatter Direction
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

class Metal : public Material {
  public:
    __device__ Metal(const Color& albedo, float fuzz) : m_albedo(albedo), m_fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& hit_record, Color& attenuation, Ray& scattered, curandState* rand_state) const override {
        Vec3 reflected = Reflect(r_in.Direction(), hit_record.m_normal);
        reflected = Vec3::UnitVector(reflected) + m_fuzz * Vec3::RandomInUnitSphere(rand_state);
        scattered = Ray(hit_record.m_point, reflected);
        attenuation = m_albedo;
        return Vec3::Dot(scattered.Direction(), hit_record.m_normal) > 0;
    }

  private:
    Color m_albedo;
    float m_fuzz;
};

class Dielectric : public Material {
  public:
    __device__ Dielectric(float refraction_index) : m_refraction_index(refraction_index) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& hit_record, Color& attenuation, Ray& scattered, curandState* rand_state) const override {
        attenuation = Color(1, 1, 1);
        float refraction_index = hit_record.m_front_face ? (1.0f / m_refraction_index) : m_refraction_index;

        Vec3 unit_direction = Vec3::UnitVector(r_in.Direction());
        float cos_theta = fminf(Vec3::Dot(-unit_direction, hit_record.m_normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_index * sin_theta > 1.0f;
        Vec3 Direction;

        if (cannot_refract || Reflectance(cos_theta, refraction_index) > RandomFloat(rand_state)) {
            Direction = Reflect(unit_direction, hit_record.m_normal);
        } else {
            Direction = Refract(unit_direction, hit_record.m_normal, refraction_index);
        }

        scattered = Ray(hit_record.m_point, Direction);
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float m_refraction_index;

    static __device__ float Reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }
};