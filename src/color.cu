#include <cmath>
#include <iostream>

#include "color.cuh"

__host__ __device__ Color Color::LinearToGamma(const Color& color) {
    Color gamma_corrected;

    auto linear_to_gamma_component = [](float linear_component) -> float {
        if (linear_component > 0.0f) {
            return sqrt(linear_component);
        }
        return 0;
    };

    for (unsigned int i = 0; i < 3; ++i) {
        gamma_corrected[i] = linear_to_gamma_component(color[i]);
    }

    return gamma_corrected;
}