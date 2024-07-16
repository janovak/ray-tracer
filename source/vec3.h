#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>

template <typename T>
class Vec3Base {
  public:
    __host__ __device__ Vec3Base() {}
    __host__ __device__ Vec3Base(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ float X() const { return e[0]; }
    __host__ __device__ float Y() const { return e[1]; }
    __host__ __device__ float Z() const { return e[2]; }

    __host__ __device__ Vec3Base<T> operator-() const { return Vec3Base<T>(-e[0], -e[1], -e[2]); }
    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ Vec3Base<T>& operator+=(const Vec3Base<T>& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ Vec3Base<T>& operator*=(const Vec3Base<T>& v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ Vec3Base<T>& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ Vec3Base<T>& operator/=(float t) {
        return *this *= 1 / t;
    }

    template<typename U = T, typename std::enable_if<!std::is_same<U, void>::value>::type* = nullptr>
    __host__ __device__ operator T() const {
        return T(e[0], e[1], e[2]);
    }

    __host__ __device__ float Length() const {
        return sqrt(LengthSquared());
    }

    __host__ __device__ float LengthSquared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    static __device__ Vec3Base<T> Random(curandState* rand_state) {
        return Vec3Base<T>(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state));
    }

    __host__ __device__ bool NearZero() const {
        // Return true if the vector is close to zero in all dimensions.
        float c = 1e-8;
        return (fabs(e[0]) < c) && (fabs(e[1]) < c) && (fabs(e[2]) < c);
    }

  protected:
    float e[3];
};

using Vec3 = Vec3Base<void>;

// Vector Utility Functions

template <typename T>
__host__ __device__ std::ostream& operator<<(std::ostream& out, const Vec3Base<T>& v) {
    return out << v[0] << ' ' << v[1] << ' ' << v[2];
}

template <typename T>
__host__ __device__ Vec3Base<T> operator+(const Vec3Base<T>& u, const Vec3Base<T>& v) {
    return Vec3Base<T>(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

template <typename T>
__host__ __device__ Vec3Base<T> operator-(const Vec3Base<T>& u, const Vec3Base<T>& v) {
    return Vec3Base<T>(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

template <typename T>
__host__ __device__ Vec3Base<T> operator*(const Vec3Base<T>& u, const Vec3Base<T>& v) {
    return Vec3Base<T>(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

template <typename T>
__host__ __device__ Vec3Base<T> operator*(float t, const Vec3Base<T>& v) {
    return Vec3Base<T>(t * v[0], t * v[1], t * v[2]);
}

template <typename T>
__host__ __device__ Vec3Base<T> operator*(const Vec3Base<T>& v, float t) {
    return t * v;
}

template <typename T>
__host__ __device__ Vec3Base<T> operator/(const Vec3Base<T>& v, float t) {
    return (1 / t) * v;
}

template <typename T>
__host__ __device__ float Dot(const Vec3Base<T>& u, const Vec3Base<T>& v) {
    return u[0] * v[0]
         + u[1] * v[1]
         + u[2] * v[2];
}

// These functions are only applicable to Vec3s and not all Vec3Base<T>s

__host__ __device__ Vec3 Cross(const Vec3& u, const Vec3& v) {
    return Vec3(u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0]);
}

__host__ __device__ Vec3 UnitVector(const Vec3& v) {
    return v / v.Length();
}

__device__ Vec3 RandomInUnitSphere(curandState* rand_state) {
    Vec3 point;

    do {
        point = 2.0f * Vec3::Random(rand_state) - Vec3(1,1,1);
    } while (point.LengthSquared() >= 1.0f);

    return point;
}

__device__ Vec3 RandomUnitVector(curandState* rand_state) {
    return UnitVector(RandomInUnitSphere(rand_state));
}

__device__ Vec3 Reflect(const Vec3& vector, const Vec3& normal) {
    return vector - 2.0f * Dot(vector, normal) * normal;
}

__device__ Vec3 Refract(const Vec3& vector, const Vec3& normal, float refraction) {
    Vec3 unit_vector = UnitVector(vector);
    float cos_theta = fminf(Dot(-unit_vector, normal), 1.0f);
    Vec3 refracted_direction_perp =  refraction * (unit_vector + cos_theta * normal);
    Vec3 refracted_direction_parallel = -sqrt(fabsf(1.0f - refracted_direction_perp.LengthSquared())) * normal;
    return refracted_direction_perp + refracted_direction_parallel;
}