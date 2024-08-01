#include <cuda_runtime.h>

#include "camera.h"
#include "cuda_helpers.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "ray_tracer.h"
#include "sphere.h"

constexpr unsigned int SCENE_ELEMENTS = 22 * 22 + 1 + 3;

__global__ void InitScene(Hittable **d_list, Hittable **d_world, curandState *rand_state) {
    curandState local_rand_state = *rand_state;
    Material* ground_material = new Lambertian(Color(0.5, 0.5, 0.5));
    d_list[0] = new Sphere(Point3(0, -1000, 0), 1000, ground_material);
    unsigned int index = 1;

    for (int i = -11; i < 11; ++i) {
        for (int j = -11; j < 11; ++j) {
            float choose_material = RandomFloat(&local_rand_state);
            Point3 center(i + 0.9 * RandomFloat(&local_rand_state), 0.2, j + 0.9 * RandomFloat(&local_rand_state));

            //if ((center - Point3(4, 0.2, 0)).Length() > 0.9) {
                Material* sphere_material;

                if (choose_material < 0.8f) {
                    // Diffuse
                    Color albedo = Color::Random(&local_rand_state) * Color::Random(&local_rand_state);
                    sphere_material = new Lambertian(Color(albedo));
                    d_list[index++] = new Sphere(center, 0.2, sphere_material);
                } else if (choose_material < 0.95f) {
                    // Metal
                    Color albedo = Color::Random(0.5, 1, &local_rand_state) * Color::Random(&local_rand_state);
                    float fuzz = RandomFloat(0, 0.5, &local_rand_state);
                    sphere_material = new Metal(albedo, fuzz);
                    d_list[index++] = new Sphere(center, 0.2, sphere_material);
                } else {
                    // Glass
                    sphere_material = new Dielectric(1.5);
                    d_list[index++] = new Sphere(center, 0.2, sphere_material);
                }
            //}
        }
    }

    d_list[index++] = new Sphere(Point3(0, 1,0),  1.0, new Dielectric(1.5));
    d_list[index++] = new Sphere(Point3(-4, 1, 0), 1.0, new Lambertian(Color(0.4, 0.2, 0.1)));
    d_list[index++] = new Sphere(Point3(4, 1, 0),  1.0, new Metal(Color(0.7, 0.6, 0.5), 0.0));
    *rand_state = local_rand_state;
    *d_world = new HittableList(d_list, SCENE_ELEMENTS);
}

__global__ void FreeScene(Hittable **d_list, Hittable **d_world) {
    for(unsigned int i = 0; i < SCENE_ELEMENTS; ++i) {
        delete d_list[i]->m_material;
        delete d_list[i];
    }
    delete *d_list;
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

    InitScene<<<1,1>>>(d_list, d_world, d_rand_state);

    // Render the scene
    Camera camera(16.0f / 9.0f, 400, 20.0, Point3(-2, 2, 1), Point3(0, 0, -1), Vec3(0, 1, 0), 1.0, 3.4);
    RayTracer ray_tracer(camera, d_world);
    ray_tracer.Render();

    // Free GPU memory
    FreeScene<<<1, 1>>>(d_list, d_world);

    cudaFree(d_rand_state);
    GpuErrorCheck(cudaFree(d_list));
    GpuErrorCheck(cudaFree(d_world));

    return 0;
}