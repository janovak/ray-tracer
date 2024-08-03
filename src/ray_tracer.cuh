#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "camera.h"
#include "color.h"
#include "constants.h"
#include "cuda_helpers.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "ray.h"

__device__ Color RayColor(const Ray& ray, Hittable** world, curandState* rand_state) {
    Ray current_ray = ray;
    Color current_attentuation = Color(1, 1, 1);

    // Iterate a maximum of 50 times rather than trust that we'll reach the end of our recursion before hitting a stack overflow.
    for (unsigned int i = 0; i < 50; ++i) {
        HitRecord hit_record;
        if ((*world)->Hit(current_ray, Interval(0.001f, kInfinity), hit_record)) {
            Ray scattered;
            Color attenuation;

            if (hit_record.m_material->Scatter(current_ray, hit_record, attenuation, scattered, rand_state)) {
                current_attentuation *= attenuation;
                current_ray = scattered;
            } else {
                return Color(0, 0, 0);
            }
        } else {
            const Vec3 unit_direction = UnitVector(current_ray.Direction());
            float a = 0.5f * (unit_direction.Y() + 1.0f);
            Color c = (1.0f - a) * Color(1, 1, 1) + a * Color(0.5, 0.7, 1.0);
            return current_attentuation * c;
        }
    }

    return Color(0, 0, 0); // Exceeded 50 iterations
}

__global__ void RenderInit(unsigned int width, unsigned int height, curandState* rand_state) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        const unsigned int pixel_index = idy * width + idx;
        //Each thread gets same seed, a different sequence number, no offset
        curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    }
}

__global__ void RenderScene(Color* image, unsigned int width, unsigned int height, unsigned int samples_per_pixel, Hittable** world, curandState* rand_state) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        const unsigned int pixel_index = idy * width + idx;
        curandState local_rand_state = rand_state[pixel_index];

        Color color(0, 0, 0);
        for(unsigned int s = 0; s < samples_per_pixel; ++s) {
            const Ray ray(GetRay(idx, idy, &local_rand_state));
            color += RayColor(ray, world, &local_rand_state);
        }
        rand_state[pixel_index] = local_rand_state;

        image[pixel_index] = color / static_cast<float>(samples_per_pixel);
    }
}

class RayTracer {
  public:
    RayTracer(Camera camera, Hittable** world) : m_camera(camera), m_world(world) {}

    ~RayTracer() {
        free(m_image);
    }

    void Render() {
        const unsigned int num_pixels = m_camera.m_image_height * m_camera.m_image_width;

        curandState* d_rand_state;
        GpuErrorCheck(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

        // Set up grid and block dimensions
        constexpr unsigned int tile_size_x = 8;
        constexpr unsigned int tile_size_y = 8;
        dim3 blocks(m_camera.m_image_width / tile_size_x + 1, m_camera.m_image_height / tile_size_y + 1);
        dim3 threads(tile_size_x, tile_size_y);

        RenderInit<<<blocks, threads>>>(m_camera.m_image_width, m_camera.m_image_height, d_rand_state);
        GpuErrorCheck(cudaGetLastError());
        GpuErrorCheck(cudaDeviceSynchronize());

        // Allocate memory for image on the device
        Color* d_image;
        GpuErrorCheck(cudaMalloc((void**)&d_image, num_pixels * sizeof(Color)));

        RenderScene<<<blocks, threads>>>(d_image, m_camera.m_image_width, m_camera.m_image_height, m_camera.m_samples_per_pixel, m_world, d_rand_state);
        GpuErrorCheck(cudaGetLastError());
        GpuErrorCheck(cudaDeviceSynchronize());

        // Allocate memory for image on the host
        m_image = (Color*)malloc(num_pixels * sizeof(Color));

        // Copy the result back to the host
        GpuErrorCheck(cudaMemcpy(m_image, d_image, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost));

        cudaFree(d_rand_state);
        cudaFree(d_image);
    }

    int WriteToFile(std::string filename) {
        std::ofstream output_file;
        output_file.open(filename);

        if (!output_file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return 1;
        }

        // Output the image
        output_file << "P3\n" << m_camera.m_image_width << ' ' << m_camera.m_image_height << "\n255\n";
        for (int j = 0; j < m_camera.m_image_height; ++j) {
            for (int i = 0; i < m_camera.m_image_width; ++i) {
                int pixel_index = j * m_camera.m_image_width + i;
                output_file << m_image[pixel_index];
            }
        }

        output_file.close();

        return 0;
    }

    Camera m_camera;
    Hittable** m_world;
    Color* m_image;
};
