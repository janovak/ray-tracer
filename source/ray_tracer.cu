#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "color.h"

// Functor to calculate the color of a pixel
struct calculate_color {
    int image_width;
    int image_height;

    calculate_color(int width, int height) : image_width(width), image_height(height) {}

    __host__ __device__
    color operator()(thrust::tuple<int, int> index) const {
        int i = thrust::get<0>(index);
        int j = thrust::get<1>(index);
        return color(static_cast<double>(i) / (image_width - 1),
                     static_cast<double>(j) / (image_height - 1),
                     0);
    }
};

int main() {
    // Image dimensions
    int image_width = 256;
    int image_height = 256;
    int num_pixels = image_width * image_height;

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
    thrust::transform(thrust::device, begin, end, d_image.begin(), calculate_color(image_width, image_height));

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
