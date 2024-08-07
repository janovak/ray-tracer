#pragma once

#include <cuda_runtime.h>

#include "ray.cuh"
#include "vec3.cuh"

// Camera and viewport constants
extern __constant__ Point3 d_camera_center;
extern __constant__ Point3 d_look_from;
extern __constant__ Point3 d_look_at;
extern __constant__ Point3 d_pixel00_loc;
extern __constant__ Vec3 d_pixel_delta_u;
extern __constant__ Vec3 d_pixel_delta_v;
extern __constant__ Vec3 d_defocus_disk_u;
extern __constant__ Vec3 d_defocus_disk_v;
extern __constant__ float d_defocus_angle;

class Camera {
  public:
    unsigned int m_image_width;
    unsigned int m_image_height;
    unsigned int m_samples_per_pixel;

    Camera(float aspect_ratio, unsigned int image_width, unsigned int samples_per_pixel, float vertical_fov, Point3 look_from, Point3 look_at, Vec3 vup, float defocus_angle, float focus_distance);

    static __device__ Vec3 SampleSquare(curandState* rand_state);
    static __device__ Point3 DefocusDiskSample(curandState* rand_state);
    static __device__ Ray GetRay(unsigned int x, unsigned int y, curandState* rand_state);
};