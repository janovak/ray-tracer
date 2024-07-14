#include <iostream>

#include <cuda_runtime.h>

#include "camera.h"
#include "color.h"
#include "constants.h"
#include "cuda_helpers.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "ray.h"

__device__ Color RayColor(const Ray& ray, Hittable** world) {
    HitRecord rec;
    if ((*world)->Hit(ray, Interval(0, kInfinity), rec)) {
        return 0.5f * Color(rec.m_normal.X() + 1.0f, rec.m_normal.Y() + 1.0f, rec.m_normal.Z() + 1.0f);
    } else {
        const Vec3 unit_direction = UnitVector(ray.Direction());
        float a = 0.5f * (unit_direction.Y() + 1.0f);
        return (1.0f - a) * Color(1.0f, 1.0f, 1.0f) + a * Color(0.5f, 0.7f, 1.0f);
    }
}

__global__ void ProcessImage(Color* d_image, int width, int height, Hittable** d_world) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        const Ray ray(GetRay(idx, idy));
        const unsigned int pixel_idx = idy * width + idx;
        d_image[pixel_idx] = RayColor(ray, d_world);
    }
}

class RayTracer {
  public:
    RayTracer(Camera camera, Hittable** world) : m_camera(camera), m_world(world) {}

    void Render() {
        const unsigned int num_pixels = m_camera.m_image_height * m_camera.m_image_width;

        // Allocate memory for image on the device
        Color* d_image;
        GpuErrorCheck(cudaMalloc((void**)&d_image, num_pixels * sizeof(Color)));

        // Set up grid and block dimensions
        constexpr unsigned int tile_size_x = 8;
        constexpr unsigned int tile_size_y = 8;
        dim3 blocks(m_camera.m_image_width / tile_size_x + 1, m_camera.m_image_height / tile_size_y + 1);
        dim3 threads(tile_size_x, tile_size_y);

        ProcessImage<<<blocks, threads>>>(d_image, m_camera.m_image_width, m_camera.m_image_height, m_world);
        GpuErrorCheck(cudaDeviceSynchronize());

        // Allocate memory for image on the host
        Color* h_image = (Color*)malloc(num_pixels * sizeof(Color));

        // Copy the result back to the host
        GpuErrorCheck(cudaMemcpy(h_image, d_image, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost));

        // Output the image
        std::cout << "P3\n" << m_camera.m_image_width << ' ' << m_camera.m_image_height << "\n255\n";
        for (int j = 0; j < m_camera.m_image_height; ++j) {
            for (int i = 0; i < m_camera.m_image_width; ++i) {
                int pixel_index = j * m_camera.m_image_width + i;
                std::cout << h_image[pixel_index];
            }
        }

        cudaFree(d_image);
        free(h_image);
    }

    Camera m_camera;
    Hittable** m_world;
};
