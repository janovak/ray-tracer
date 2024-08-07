#pragma once

#include <string>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "camera.cuh"
#include "color.cuh"
#include "hittable.cuh"
#include "ray.cuh"

class RayTracer {
  public:
    Camera m_camera;
    Hittable** m_world;
    Color* m_image;

    RayTracer(Camera camera, Hittable** world) : m_camera(camera), m_world(world) {}

    ~RayTracer() {
        free(m_image);
    }

    void Render();
    int WriteToFile(std::string filename);

    static __device__ Color RayColor(const Ray& ray, Hittable** world, curandState* rand_state);
};