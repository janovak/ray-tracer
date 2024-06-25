#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "color.h"
#include "ray.h"

__constant__ point3 camera_center;
__constant__ point3 pixel00_loc;
__constant__ point3 pixel_delta_u;
__constant__ point3 pixel_delta_v;

// Functor to calculate the color of a pixel
struct calculate_color {
    __device__ color operator()(thrust::tuple<int, int> index) const {
        int i = thrust::get<0>(index);
        int j = thrust::get<1>(index);

        point3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
        vec3 ray_direction = pixel_center - camera_center;

        vec3 unit_direction = unit_vector(ray_direction);
        double a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};

int main() {
    // Image dimensions
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera

    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(image_width)/image_height);
    point3 camera = point3(0, 0, 0);
    cudaMemcpyToSymbol(camera_center, &camera, sizeof(point3));

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    vec3 delta_u = viewport_u / image_width;
    cudaMemcpyToSymbol(pixel_delta_u, &delta_u, sizeof(point3));
    vec3 delta_v = viewport_v / image_height;
    cudaMemcpyToSymbol(pixel_delta_v, &delta_v, sizeof(point3));

    // Calculate the location of the upper left pixel.
    point3 viewport_upper_left = camera - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    point3 top_left_loc = viewport_upper_left + 0.5 * (delta_u + delta_v);
    cudaMemcpyToSymbol(pixel00_loc, &top_left_loc, sizeof(point3));

    int num_pixels = image_height * image_width;

    // Allocate memory for image on host
    thrust::host_vector<color> h_image(num_pixels);

    // Generate indices for each pixel
    thrust::device_vector<int> d_x(image_width);
    thrust::device_vector<int> d_y(image_height);

    thrust::sequence(thrust::device, d_x.begin(), d_x.end());
    thrust::sequence(thrust::device, d_y.begin(), d_y.end());

    // Create a Cartesian product of the indices
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_permutation_iterator(d_x.begin(), thrust::counting_iterator<int>(0)),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), thrust::placeholders::_1 / image_width)));

    auto end = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::make_permutation_iterator(d_x.begin(), thrust::counting_iterator<int>(num_pixels)),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(num_pixels), thrust::placeholders::_1 / image_width)));

    // Allocate memory for image on device
    thrust::device_vector<color> d_image(num_pixels);

    // Compute the colors using thrust::transform
    thrust::transform(thrust::device, begin, end, d_image.begin(), calculate_color());

    // Copy the result back to the host
    thrust::copy(d_image.begin(), d_image.end(), h_image.begin());

    // Output the image
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            int pixel_index = j * image_width + i;
            std::cout << h_image[pixel_index];
        }
    }

    return 0;
}
