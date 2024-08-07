#include <cuda_runtime.h>

#include "camera.cuh"
#include "cuda_helpers.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "vec3.cuh"

__constant__ Point3 d_camera_center;
__constant__ Point3 d_look_from;
__constant__ Point3 d_look_at;
__constant__ Point3 d_pixel00_loc;
__constant__ Vec3 d_pixel_delta_u;
__constant__ Vec3 d_pixel_delta_v;
__constant__ Vec3 d_defocus_disk_u;
__constant__ Vec3 d_defocus_disk_v;
__constant__ float d_defocus_angle;

Camera::Camera(float aspect_ratio, unsigned int image_width, unsigned int samples_per_pixel, float vertical_fov, Point3 look_from, Point3 look_at, Vec3 vup, float defocus_angle, float focus_distance)
    : m_image_width(image_width), m_samples_per_pixel(samples_per_pixel) {

    // Calculate the image height, and ensure that it's At least 1.
    m_image_height = static_cast<unsigned int>(m_image_width / aspect_ratio);
    m_image_height = (m_image_height < 1) ? 1 : m_image_height;

    // Copy static values to constant memory

    Point3 camera_center = look_from;
    GpuErrorCheck(cudaMemcpyToSymbol(d_camera_center, &camera_center, sizeof(Point3)));

    GpuErrorCheck(cudaMemcpyToSymbol(d_look_from, &look_from, sizeof(Point3)));
    GpuErrorCheck(cudaMemcpyToSymbol(d_look_at, &look_at, sizeof(Point3)));

    GpuErrorCheck(cudaMemcpyToSymbol(d_defocus_angle, &defocus_angle, sizeof(float)));

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    Vec3 w_axis = Vec3::UnitVector(look_from - look_at);
    Vec3 u_axis = Vec3::UnitVector(Vec3::Cross(vup, w_axis));
    Vec3 v_axis = Vec3::Cross(w_axis, u_axis);

    // Determine viewport dimensions
    float theta = DegreesToRadians(vertical_fov);
    float height = tanf(theta / 2.0f);
    float viewport_height = 2.0f * height * focus_distance;
    float viewport_width = viewport_height * static_cast<float>(m_image_width) / m_image_height;

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    Vec3 viewport_u = u_axis * viewport_width;
    Vec3 viewport_v = v_axis * -viewport_height;

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    Vec3 h_pixel_delta_u = viewport_u / m_image_width;
    GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_u, &h_pixel_delta_u, sizeof(Vec3)));
    Vec3 h_pixel_delta_v = viewport_v / m_image_height;
    GpuErrorCheck(cudaMemcpyToSymbol(d_pixel_delta_v, &h_pixel_delta_v, sizeof(Vec3)));

    // Calculate the location of the upper left pixel.
    Point3 viewport_xpper_left = camera_center - (focus_distance * w_axis) - (viewport_u / 2.0f) - (viewport_v / 2.0f);
    Point3 h_pixel00_loc = viewport_xpper_left + 0.5f * (h_pixel_delta_u + h_pixel_delta_v);
    GpuErrorCheck(cudaMemcpyToSymbol(d_pixel00_loc, &h_pixel00_loc, sizeof(Point3)));

    // Calculate the camera defocus disk basis vectors.
    float defocus_radius = focus_distance * tanf(DegreesToRadians(defocus_angle / 2.0f));
    Vec3 h_defocus_disk_u = u_axis * defocus_radius;
    GpuErrorCheck(cudaMemcpyToSymbol(d_defocus_disk_u, &h_defocus_disk_u, sizeof(Vec3)));
    Vec3 h_defocus_disk_v = v_axis * defocus_radius;
    GpuErrorCheck(cudaMemcpyToSymbol(d_defocus_disk_v, &h_defocus_disk_v, sizeof(Vec3)));
}

__device__ Vec3 Camera::SampleSquare(curandState* rand_state) {
    return Vec3(RandomFloat(-0.5, 0.5, rand_state), RandomFloat(-0.5, 0.5, rand_state), 0);
}

__device__ Point3 Camera::DefocusDiskSample(curandState* rand_state) {
    Vec3 point = RandomInUnitDisk(rand_state);
    return d_camera_center + (point[0] * d_defocus_disk_u) + (point[1] * d_defocus_disk_v);
}

__device__ Ray Camera::GetRay(unsigned int x, unsigned int y, curandState* rand_state) {
    Vec3 offset = SampleSquare(rand_state);
    const Point3 pixel_sample = d_pixel00_loc + (x + offset.X()) * d_pixel_delta_u + (y + offset.Y()) * d_pixel_delta_v;

    const Point3 ray_origin = (d_defocus_angle <= 0.0f) ? d_camera_center : DefocusDiskSample(rand_state);
    const Vec3 ray_direction = pixel_sample - d_camera_center;

    return Ray(ray_origin, ray_direction);
}