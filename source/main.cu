#include <cuda_runtime.h>

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "ray_tracer.h"
#include "sphere.h"

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