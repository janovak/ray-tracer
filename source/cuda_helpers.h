#pragma once

#include <cuda_runtime.h>

#include "constants.h"

#define GpuErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * kPi / 180.0f;
}

__device__ float RandomFloat(curandState* rand_state) {
    return (curand_uniform(rand_state) - 1.0f) * -1.0f;
}

__device__ float RandomFloat(float min, float max, curandState* rand_state) {
    return min + (max - min) * RandomFloat(rand_state);
}