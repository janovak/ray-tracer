#include "ray.cuh"
#include "vec3.cuh"

__device__ Ray operator+(const Ray& u, const Vec3& v) {
    return Ray(u.Origin(), u.Direction() + v);
}

__device__ Ray operator-(const Ray& u, const Vec3& v) {
    return Ray(u.Origin(), u.Direction() - v);
}