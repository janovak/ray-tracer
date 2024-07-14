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
        Vec3 unit_direction = UnitVector(ray.Direction());
        float a = 0.5f * (unit_direction.Y() + 1.0f);
        return (1.0f - a) * Color(1.0f, 1.0f, 1.0f) + a * Color(0.5f, 0.7f, 1.0f);
    }
}

__global__ void ProcessImage(Color* d_image, int width, int height, Hittable** d_world) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        Point3 pixel_center = d_pixel00_loc + (idx * d_pixel_delta_u) + (idy * d_pixel_delta_v);
        Vec3 ray_direction = pixel_center - d_camera_center;
        Ray ray(d_camera_center, ray_direction);

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

__global__ void InitScene(Hittable **d_list, Hittable **d_world) {
    *(d_list)     = new Sphere(Point3(0, 0, -1), 0.5f);
    *(d_list + 1) = new Sphere(Point3(0, -100.5f, -1), 100);
    *d_world      = new HittableList(d_list, 2);
}

__global__ void FreeScene(Hittable **d_list, Hittable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

int main() {
    // Initialize the world
    Hittable** d_list;
    Hittable** d_world;

    GpuErrorCheck(cudaMalloc((void**)&d_list, 2 * sizeof(Hittable*)));
    GpuErrorCheck(cudaMalloc((void**)&d_world, sizeof(Hittable*)));

    InitScene<<<1,1>>>(d_list, d_world);
    GpuErrorCheck(cudaDeviceSynchronize());

    // Render the scene
    Camera camera(16.0f / 9.0f, 400);
    RayTracer ray_tracer(camera, d_world);
    ray_tracer.Render();

    // Free GPU memory
    FreeScene<<<1, 1>>>(d_list, d_world);
    GpuErrorCheck(cudaDeviceSynchronize());

    cudaFree(d_list);
    cudaFree(d_world);

    return 0;
}