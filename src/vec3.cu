#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "vec3.cuh"

// These functions are only applicable to Vec3s and not all Vec3Base<T>s

__device__ Vec3 RandomInUnitDisk(curandState* rand_state) {
    while (true) {
        Vec3 point = Vec3(RandomFloat(-1.0, 1.0, rand_state), RandomFloat(-1.0, 1.0, rand_state), 0);
        if (point.LengthSquared() < 1.0f) {
            return point;
        }
    }
}

__device__ Vec3 RandomUnitVector(curandState* rand_state) {
    return Vec3::UnitVector(Vec3::RandomInUnitSphere(rand_state));
}

__device__ Vec3 Reflect(const Vec3& vector, const Vec3& normal) {
    return vector - 2.0f * Vec3::Dot(vector, normal) * normal;
}

__device__ Vec3 Refract(const Vec3& vector, const Vec3& normal, float refraction) {
    Vec3 unit_vector = Vec3::UnitVector(vector);
    float cos_theta = fminf(Vec3::Dot(-unit_vector, normal), 1.0);
    Vec3 refracted_direction_perp =  refraction * (unit_vector + cos_theta * normal);
    Vec3 refracted_direction_parallel = -sqrt(fabsf(1.0f - refracted_direction_perp.LengthSquared())) * normal;
    return refracted_direction_perp + refracted_direction_parallel;
}