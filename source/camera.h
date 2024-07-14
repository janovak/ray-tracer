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
__constant__ Point3 d_pixel_delta_x;
__constant__ Point3 d_pixel_delta_y;

__device__ Ray GetRay(float x, float y) {
    const Point3 pixel_center = d_pixel00_loc + (x * d_pixel_delta_x) + (y * d_pixel_delta_y);
    const Vec3 ray_direction = pixel_center - d_camera_center;
    return Ray(d_camera_center, ray_direction);
}

class Camera {
  public:
    unsigned int m_image_width;
    unsigned int m_image_height;

    Camera(float aspect_ratio, unsigned int image_width) : m_aspect_ratio(aspect_ratio), m_image_width(image_width) {
        // Calculate the image height, and ensure that it's At least 1.
        m_image_height = int(m_image_width / m_aspect_ratio);
        m_image_height = (m_image_height < 1) ? 1 : m_image_height;

        m_viewport_width = m_viewport_height * static_cast<float>(m_image_width) / m_image_height;

        Point3 h_camera_center = Point3(0, 0, 0);
        GpuErrorCheck(cudaMemcpyToSymbol(d_camera_center, &h_camera_center, sizeof(Point3)));

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vec3 viewport_x = Vec3(m_viewport_width, 0, 0);
        Vec3 viewport_y = Vec3(0, -m_viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        Vec3 h_pixel_delta_x = viewport_x / m_image_width;
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_x, &h_pixel_delta_x, sizeof(Point3)));
        Vec3 h_pixel_delta_y = viewport_y / m_image_height;
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_y, &h_pixel_delta_y, sizeof(Point3)));

        // Calculate the location of the upper left pixel.
        Point3 viewport_xpper_left = h_camera_center - Vec3(0, 0, m_focal_length) - viewport_x / 2 - viewport_y / 2;
        Point3 h_pixel00_loc = viewport_xpper_left + 0.5f * (h_pixel_delta_x + h_pixel_delta_y);
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel00_loc, &h_pixel00_loc, sizeof(Point3)));
    }

  private:
    float m_aspect_ratio;
    const float m_focal_length = 1.0f;
    const float m_viewport_height = 2.0f;
    float m_viewport_width;
};