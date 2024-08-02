#include <cuda_runtime.h>

#include "camera.h"
#include "cuda_helpers.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "ray_tracer.h"
#include "sphere.h"

constexpr unsigned int SCENE_ELEMENTS = 22 * 22 + 1 + 3;

__global__ void RandInit(curandState* rand_state) {
    curand_init(1984, 0, 0, rand_state);
}

__global__ void InitScene(Hittable** d_list, Hittable** d_world, curandState* rand_state) {
    curandState local_rand_state = *rand_state;

    unsigned int index = 0;
    d_list[index++] = new Sphere(Point3(0, -1000, 0), 1000, new Lambertian(Color(0.5, 0.5, 0.5)));
    d_list[index++] = new Sphere(Point3(0, 1,0),  1.0, new Dielectric(1.5));
    d_list[index++] = new Sphere(Point3(-4, 1, 0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
    d_list[index++] = new Sphere(Point3(4, 1, 0),  1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));

     for (int i = -11 ; i < 11; ++i) {
        for (int j = -11; j < 11; ++j) {
            float choose_material = RandomFloat(&local_rand_state);
            Point3 center(i + 0.9f * RandomFloat(&local_rand_state), 0.2f, j + 0.9f * RandomFloat(&local_rand_state));

            // Don't allow generated spheres to intersect 3 larger spheres
            if ((center - Point3(-4, 0.2, 0)).Length() < 0.9f ||
                (center - Point3(0, 0.2, 0)).Length() < 0.9f ||
                (center - Point3(4, 0.2, 0)).Length() < 0.9f) {
                continue;
            }

            if (choose_material < 0.8f) {
                // Diffuse
                Color albedo = Color::Random(&local_rand_state) * Color::Random(&local_rand_state);
                d_list[index++] = new Sphere(center, 0.2, new Lambertian(Color(albedo)));
            } else if (choose_material < 0.95f) {
                // Metal
                Color albedo = Color::Random(0.5, 1, &local_rand_state) * Color::Random(&local_rand_state);
                float fuzz = RandomFloat(0, 0.5, &local_rand_state);
                d_list[index++] = new Sphere(center, 0.2, new Metal(albedo, fuzz));
            } else {
                // Glass
                d_list[index++] = new Sphere(center, 0.2, new Dielectric(1.5));
            }
        }
    }

    *rand_state = local_rand_state;
    *d_world = new HittableList(d_list, index);
}

__global__ void FreeScene(Hittable** d_list, Hittable** d_world) {
    unsigned int scene_elements = static_cast<HittableList*>(*d_world)->m_size;
    for(unsigned int i = 0; i < scene_elements; ++i) {
        delete d_list[i]->m_material;
        delete d_list[i];
    }
    delete *d_world;
}

int main() {
    // Initialize the world
    Hittable** d_list;
    Hittable** d_world;

    GpuErrorCheck(cudaMalloc((void**)&d_list, SCENE_ELEMENTS * sizeof(Hittable*)));
    GpuErrorCheck(cudaMalloc((void**)&d_world, sizeof(Hittable*)));

    curandState* d_rand_state;
    GpuErrorCheck(cudaMalloc((void **)&d_rand_state, sizeof(curandState)));

    RandInit<<<1,1>>>(d_rand_state);
    GpuErrorCheck(cudaGetLastError());
    GpuErrorCheck(cudaDeviceSynchronize());

    InitScene<<<1,1>>>(d_list, d_world, d_rand_state);
    GpuErrorCheck(cudaDeviceSynchronize());

    // Render the scene
    Camera camera(16.0f / 9.0f, 1200, 500, 20, Point3(13, 2, 3), Point3(0, 0, 0), Vec3(0, 1, 0), 0.1, 10);
    RayTracer ray_tracer(camera, d_world);
    ray_tracer.Render();

    // Free GPU memory
    FreeScene<<<1, 1>>>(d_list, d_world);
    GpuErrorCheck(cudaDeviceSynchronize());

    GpuErrorCheck(cudaFree(d_rand_state));
    GpuErrorCheck(cudaFree(d_list));
    GpuErrorCheck(cudaFree(d_world));

    return 0;
}