#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "color.h"
#include "constants.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "ray.h"

// Camera and viewport constants
__constant__ Point3 d_camera_center;
__constant__ Point3 d_pixel00_loc;
__constant__ Point3 d_pixel_delta_u;
__constant__ Point3 d_pixel_delta_v;

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void create_world(Hittable **d_list, Hittable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new Sphere(Point3(0,0,-1), 0.5);
        *(d_list+1) = new Sphere(Point3(0,-100.5,-1), 100);
        *d_world    = new HittableList(d_list,2);
    }
}

__global__ void free_world(Hittable **d_list, Hittable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

__device__ Color trace_ray(const Ray& ray, Hittable** world) {
    HitRecord rec;
    if ((*world)->Hit(ray, 0, kInfinity, rec)) {
        return 0.5f*Color(rec.m_normal.X()+1.0f, rec.m_normal.Y()+1.0f, rec.m_normal.Z()+1.0f);
    } else {
        Vec3 unit_direction = UnitVector(ray.Direction());
        float a = 0.5f*(unit_direction.Y() + 1.0f);
        return (1.0f-a)*Color(1.0, 1.0, 1.0) + a*Color(0.5, 0.7, 1.0);
    }
}

// Kernel to process image data
__global__ void process_image_kernel(Color* d_image, int width, int height, Hittable** d_world) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < width && idy < height) {
        int pixel_idx = idy * width + idx;

        Point3 pixel_center = d_pixel00_loc + (idx * d_pixel_delta_u) + (idy * d_pixel_delta_v);
        Vec3 ray_direction = pixel_center - d_camera_center;
        Ray R (d_camera_center, ray_direction);

        d_image[pixel_idx] = trace_ray(R, d_world);
    }
}

// Functor to calculate the Color of a pixel
struct calculate_Color {
    HittableList* world;

    __host__ __device__ calculate_Color(HittableList* world_ptr) : world(world_ptr) {}

    __device__ Color operator()(thrust::tuple<int, int> index) const {
        int i = thrust::get<0>(index);
        int j = thrust::get<1>(index);

        Point3 pixel_center = d_pixel00_loc + (i * d_pixel_delta_u) + (j * d_pixel_delta_v);
        Vec3 ray_direction = pixel_center - d_camera_center;
        Ray R (d_camera_center, ray_direction);

        HitRecord rec;
        world->Hit(R, 0, kInfinity, rec);

        Vec3 unit_direction = UnitVector(ray_direction);
        double a = 0.5*(unit_direction.Y() + 1.0);
        return (1.0-a)*Color(1.0, 1.0, 1.0) + a*Color(0.5, 0.7, 1.0);
    }
};

int main() {
    // Image dimensions
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's At least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    Hittable** d_list;
    Hittable** d_world;

    constexpr unsigned int SPHERES = 2;

    cudaMalloc((void**)&d_list, SPHERES * sizeof(Hittable*));
    cudaMalloc((void**)&d_world, sizeof(Hittable*));

    create_world<<<1,1>>>(d_list, d_world);
    cudaDeviceSynchronize();


    // Camera

    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(image_width)/image_height);
    Point3 h_camera_center = Point3(0, 0, 0);
    cudaMemcpyToSymbol(d_camera_center, &h_camera_center, sizeof(Point3));

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    Vec3 viewport_u = Vec3(viewport_width, 0, 0);
    Vec3 viewport_v = Vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    Vec3 h_pixel_delta_u = viewport_u / image_width;
    cudaMemcpyToSymbol(d_pixel_delta_u, &h_pixel_delta_u, sizeof(Point3));
    Vec3 h_pixel_delta_v = viewport_v / image_height;
    cudaMemcpyToSymbol(d_pixel_delta_v, &h_pixel_delta_v, sizeof(Point3));

    // Calculate the location of the upper left pixel.
    Point3 viewport_upper_left = h_camera_center - Vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    Point3 h_pixel00_loc = viewport_upper_left + 0.5 * (h_pixel_delta_u + h_pixel_delta_v);
    cudaMemcpyToSymbol(d_pixel00_loc, &h_pixel00_loc, sizeof(Point3));


    int num_pixels = image_height * image_width;

/*     // Generate indices for each pixel
    thrust::device_vector<int> d_x(image_width);
    CUDA_CHECK_ERROR();  // Check for errors after operation

    thrust::device_vector<int> d_y(image_height);
    CUDA_CHECK_ERROR();  // Check for errors after operation


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
    thrust::device_vector<Color> d_image(num_pixels);

    // Compute the Colors using thrust::transform
    thrust::transform(thrust::device, begin, end, d_image.begin(), calculate_Color(d_world));

    // Copy the result back to the host
    thrust::host_vector<Color> h_image(num_pixels);
    thrust::copy(d_image.begin(), d_image.end(), h_image.begin()); */

    // Allocate memory for image on the device
    Color* d_image;
    cudaMalloc((void**)&d_image, num_pixels * sizeof(Color));


    // Set up grid and block dimensions
    dim3 blockSize(16, 16);  // You can adjust this according to your needs
    dim3 numBlocks((image_width + blockSize.x - 1) / blockSize.x, 
                   (image_height + blockSize.y - 1) / blockSize.y);

    // Initialize world data (this should be done appropriately)
    //world d_world;  // You should initialize this with your world data

    // Call the kernel
    process_image_kernel<<<numBlocks, blockSize>>>(d_image, image_width, image_height, d_world);

    // Allocate memory for image on the host
    Color* h_image = (Color*)malloc(num_pixels * sizeof(Color));

    // Copy the result back to the host
    cudaMemcpy(h_image, d_image, num_pixels * sizeof(Color), cudaMemcpyDeviceToHost);

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
