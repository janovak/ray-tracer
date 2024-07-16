#include <cuda_runtime.h>

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "ray_tracer.h"
#include "sphere.h"

constexpr unsigned int SCENE_ELEMENTS = 4;

__global__ void InitScene(Hittable **d_list, Hittable **d_world) {
    d_list[0] = new Sphere(Point3(0, 0, -1.2), 0.5, new Lambertian(Color(0.1, 0.2, 0.5)));
    d_list[1] = new Sphere(Point3(0, -100.5, -1), 100, new Lambertian(Color(0.8, 0.8, 0.0)));
    d_list[2] = new Sphere(Point3(1, 0, -1), 0.5, new Metal(Color(0.8, 0.6, 0.2), 1.0));
    d_list[3] = new Sphere(Point3(-1, 0, -1), 0.5, new Dielectric(1.0 / 1.33));
    *d_world = new HittableList(d_list, SCENE_ELEMENTS);
}

__global__ void FreeScene(Hittable **d_list, Hittable **d_world) {
    for(unsigned int i = 0; i < SCENE_ELEMENTS; ++i) {
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