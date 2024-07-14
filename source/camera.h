#pragma once

#include <cuda_runtime.h>

#include "color.h"
#include "cuda_helpers.h"
#include "hittable.h"
#include "interval.h"
#include "point3.h"
#include "ray.h"
#include "vec3.h"

// Camera and viewport constants
__constant__ Point3 d_camera_center;
__constant__ Point3 d_pixel00_loc;
__constant__ Point3 d_pixel_delta_u;
__constant__ Point3 d_pixel_delta_v;

class Camera {
  public:
    unsigned int m_image_width;
    unsigned int m_image_height;

    Camera(float aspect_ratio, unsigned int image_width) : m_aspect_ratio(aspect_ratio), m_image_width(image_width) {
        // Calculate the image height, and ensure that it's At least 1.
        m_image_height = int(m_image_width / m_aspect_ratio);
        m_image_height = (m_image_height < 1) ? 1 : m_image_height;

        m_viewport_width = m_viewport_height * (float(m_image_width) / m_image_height);

        Point3 h_camera_center = Point3(0, 0, 0);
        GpuErrorCheck(cudaMemcpyToSymbol(d_camera_center, &h_camera_center, sizeof(Point3)));

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vec3 viewport_u = Vec3(m_viewport_width, 0, 0);
        Vec3 viewport_v = Vec3(0, -m_viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        Vec3 h_pixel_delta_u = viewport_u / m_image_width;
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_u, &h_pixel_delta_u, sizeof(Point3)));
        Vec3 h_pixel_delta_v = viewport_v / m_image_height;
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_v, &h_pixel_delta_v, sizeof(Point3)));

        // Calculate the location of the upper left pixel.
        Point3 viewport_upper_left = h_camera_center - Vec3(0, 0, m_focal_length) - viewport_u / 2 - viewport_v / 2;
        Point3 h_pixel00_loc = viewport_upper_left + 0.5f * (h_pixel_delta_u + h_pixel_delta_v);
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel00_loc, &h_pixel00_loc, sizeof(Point3)));
    }

/*     __host__ __device__ Ray GetRay(unsigned int u, unsigned int v) {
        return Ray(d_camera_center, d_pixel00_loc + (u * d_pixel_delta_u) + (v * d_pixel_delta_v));
    } */

  private:
    float m_aspect_ratio;
    const float m_focal_length = 1.0f;
    const float m_viewport_height = 2.0f;
    float m_viewport_width;
};