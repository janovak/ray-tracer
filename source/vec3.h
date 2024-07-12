#pragma once

#include <cuda_runtime.h>
#include <iostream>

template <typename T>
class vec3_base {
  public:
    __host__ __device__ vec3_base() {}
    __host__ __device__ vec3_base(double e0, double e1, double e2) : e{e0, e1, e2} {}

    __host__ __device__ double x() const { return e[0]; }
    __host__ __device__ double y() const { return e[1]; }
    __host__ __device__ double z() const { return e[2]; }

    __host__ __device__ vec3_base<T> operator-() const { return vec3_base<T>(-e[0], -e[1], -e[2]); }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double& operator[](int i) { return e[i]; }

    __host__ __device__ vec3_base<T>& operator+=(const vec3_base<T>& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3_base<T>& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3_base<T>& operator/=(double t) {
        return *this *= 1/t;
    }

    template<typename U = T, typename std::enable_if<!std::is_same<U, void>::value>::type* = nullptr>
    __host__ __device__ operator T() const {
        return T(e[0], e[1], e[2]);
    }

    __host__ __device__ double length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

  protected:
    double e[3];
};

using vec3 = vec3_base<void>;

// Vector Utility Functions

template <typename T>
__host__ __device__ std::ostream& operator<<(std::ostream& out, const vec3_base<T>& v) {
    return out << v[0] << ' ' << v[1] << ' ' << v[2];
}

template <typename T>
__host__ __device__ vec3_base<T> operator+(const vec3_base<T>& u, const vec3_base<T>& v) {
    return vec3_base<T>(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

template <typename T>
__host__ __device__ vec3_base<T> operator-(const vec3_base<T>& u, const vec3_base<T>& v) {
    return vec3_base<T>(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

template <typename T>
__host__ __device__ vec3_base<T> operator*(const vec3_base<T>& u, const vec3_base<T>& v) {
    return vec3_base<T>(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

template <typename T>
__host__ __device__ vec3_base<T> operator*(double t, const vec3_base<T>& v) {
    return vec3_base<T>(t*v[0], t*v[1], t*v[2]);
}

template <typename T>
__host__ __device__ vec3_base<T> operator*(const vec3_base<T>& v, double t) {
    return t * v;
}

template <typename T>
__host__ __device__ vec3_base<T> operator/(const vec3_base<T>& v, double t) {
    return (1/t) * v;
}

template <typename T>
__host__ __device__ double dot(const vec3_base<T>& u, const vec3_base<T>& v) {
    return u[0] * v[0]
         + u[1] * v[1]
         + u[2] * v[2];
}

template <typename T>
__host__ __device__ vec3_base<T> cross(const vec3_base<T>& u, const vec3_base<T>& v) {
    return vec3_base<T>(u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0]);
}

template <typename T>
__host__ __device__ vec3_base<T> unit_vector(const vec3_base<T>& v) {
    return v / v.length();
}
