#include "hittable.cuh"

// outward_normal is assumed to have unit length
__device__ void HitRecord::SetFaceNormal(const Ray& ray, const Vec3& outward_normal) {
    m_front_face = Vec3::Dot(ray.Direction(), outward_normal) < 0;
    m_normal = m_front_face ? outward_normal : -outward_normal;
}

__device__ Ray HitRecord::RandomOnHemisphere(curandState* rand_state) {
    Vec3 on_unit_sphere = RandomUnitVector(rand_state);
    if (Vec3::Dot(on_unit_sphere, m_normal) > 0) { // In the same hemisphere as the normal
        return Ray(m_point, on_unit_sphere);
    } else {
        return Ray(m_point, -on_unit_sphere);
    }
}