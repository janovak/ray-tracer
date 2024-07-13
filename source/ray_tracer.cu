#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "color.h"
#include "constants.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "ray.h"

// Camera and viewport constants
__constant__ Point3 d_camera_center;
__constant__ Point3 d_pixel00_loc;
__constant__ Point3 d_pixel_delta_u;
__constant__ Point3 d_pixel_delta_v;

#define GpuErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

__device__ Color TraceRay(const Ray& ray, Hittable** world) {
    HitRecord rec;
    if ((*world)->Hit(ray, 0, kInfinity, rec)) {
        return 0.5f*Color(rec.m_normal.X()+1.0f, rec.m_normal.Y()+1.0f, rec.m_normal.Z()+1.0f);
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
        d_image[pixel_idx] = TraceRay(ray, d_world);
    }
}

int main() {
    // Initialize the world
    Hittable** d_list;
    Hittable** d_world;

    GpuErrorCheck(cudaMalloc((void**)&d_list, 2 * sizeof(Hittable*)));
    GpuErrorCheck(cudaMalloc((void**)&d_world, sizeof(Hittable*)));

    InitScene<<<1,1>>>(d_list, d_world);
    GpuErrorCheck(cudaDeviceSynchronize());

    // Image dimensions
    const float aspect_ratio = 16.0f / 9.0f;
    const unsigned int image_width = 400;

    // Calculate the image height, and ensure that it's At least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera

    float focal_length = 1.0f;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * (float(image_width) / image_height);

    Point3 h_camera_center = Point3(0, 0, 0);
    GpuErrorCheck(cudaMemcpyToSymbol(d_camera_center, &h_camera_center, sizeof(Point3)));

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    Vec3 viewport_u = Vec3(viewport_width, 0, 0);
    Vec3 viewport_v = Vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    Vec3 h_pixel_delta_u = viewport_u / image_width;
    GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_u, &h_pixel_delta_u, sizeof(Point3)));
    Vec3 h_pixel_delta_v = viewport_v / image_height;
    GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_v, &h_pixel_delta_v, sizeof(Point3)));

    // Calculate the location of the upper left pixel.
    Point3 viewport_upper_left = h_camera_center - Vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    Point3 h_pixel00_loc = viewport_upper_left + 0.5f * (h_pixel_delta_u + h_pixel_delta_v);
    GpuErrorCheck(cudaMemcpyToSymbol(d_pixel00_loc, &h_pixel00_loc, sizeof(Point3)));

    const unsigned int num_pixels = image_height * image_width;

    // Allocate memory for image on the device
    Color* d_image;
    GpuErrorCheck(cudaMalloc((void**)&d_image, num_pixels * sizeof(Color)));

    // Set up grid and block dimensions
    constexpr unsigned int tile_size_x = 8;
    constexpr unsigned int tile_size_y = 8;
    dim3 blocks(image_width / tile_size_x + 1, image_height / tile_size_y + 1);
    dim3 threads(tile_size_x, tile_size_y);

    ProcessImage<<<blocks, threads>>>(d_image, image_width, image_height, d_world);
    GpuErrorCheck(cudaDeviceSynchronize());

    // Allocate memory for image on the host
    Color* h_image = (Color*)malloc(num_pixels * sizeof(Color));

    // Copy the result back to the host
    GpuErrorCheck(cudaMemcpy(h_image, d_image, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost));

    // Output the image
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            std::cout << h_image[pixel_index];
        }
    }

    // Free dynamically allocated memory

    // Free GPU memory
    FreeScene<<<1, 1>>>(d_list, d_world);
    GpuErrorCheck(cudaDeviceSynchronize());

    cudaFree(d_list);
    cudaFree(d_world);
    cudaFree(d_image);

    // Free host memory
    free(h_image);

    return 0;
}
