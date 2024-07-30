#pragma once

#include <cuda_runtime.h>

#include "color.h"
#include "cuda_helpers.h"
#include "hittable.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

// Camera and viewport constants
__constant__ Point3 d_look_from;
__constant__ Point3 d_look_at;
__constant__ Point3 d_pixel00_loc;
__constant__ Vec3 d_pixel_delta_u;
__constant__ Vec3 d_pixel_delta_v;

__device__ Ray GetRay(unsigned int x, unsigned int y, curandState* rand_state) {
    Vec3 offset = Vec3(curand_uniform(rand_state) - 0.5f, curand_uniform(rand_state) - 0.5f, 0);
    const Point3 pixel_center = d_pixel00_loc + (x + offset.X()) * d_pixel_delta_u + (y + offset.Y()) * d_pixel_delta_v;
    const Vec3 ray_direction = pixel_center - d_look_from;
    return Ray(d_look_from, ray_direction);
}

class Camera {
  public:
    unsigned int m_image_width;
    unsigned int m_image_height;

    Camera(float aspect_ratio, unsigned int image_width, float vertical_fov, Point3 look_from, Point3 look_at, Vec3 vup)
        : m_aspect_ratio(aspect_ratio), m_image_width(image_width), m_vertical_fov(vertical_fov) {

        // Calculate the image height, and ensure that it's At least 1.
        m_image_height = static_cast<int>(m_image_width / m_aspect_ratio);
        m_image_height = (m_image_height < 1) ? 1 : m_image_height;

        m_focal_length = (look_from - look_at).Length();

        float theta = DegreesToRadians(m_vertical_fov);
        float height = tanf(theta / 2.0f);
        m_viewport_height = 2.0f * height * m_focal_length;
        m_viewport_width = m_viewport_height * static_cast<float>(m_image_width) / m_image_height;

        GpuErrorCheck(cudaMemcpyToSymbol(d_look_from, &look_from, sizeof(Point3)));
        GpuErrorCheck(cudaMemcpyToSymbol(d_look_at, &look_at, sizeof(Point3)));

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        Vec3 w_axis = UnitVector(look_from - look_at);
        Vec3 u_axis = UnitVector(Cross(vup, w_axis));
        Vec3 v_axis = Cross(w_axis, u_axis);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        Vec3 viewport_u = u_axis * m_viewport_width;
        Vec3 viewport_v = v_axis * -m_viewport_height;

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        Vec3 h_pixel_delta_u = viewport_u / m_image_width;
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_u, &h_pixel_delta_u, sizeof(Point3)));
        Vec3 h_pixel_delta_v = viewport_v / m_image_height;
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_v, &h_pixel_delta_v, sizeof(Point3)));

        // Calculate the location of the upper left pixel.
        Point3 viewport_xpper_left = look_from - (w_axis * m_focal_length) - (viewport_u / 2) - (viewport_v / 2);
        Point3 h_pixel00_loc = viewport_xpper_left + 0.5f * (h_pixel_delta_u + h_pixel_delta_v);
        GpuErrorCheck(cudaMemcpyToSymbol(d_pixel00_loc, &h_pixel00_loc, sizeof(Point3)));
    }

  private:
    float m_aspect_ratio;
    float m_focal_length;
    float m_viewport_width;
    float m_viewport_height;
    float m_vertical_fov;
};