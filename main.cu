#include <iostream>
#include <time.h>
#include "cutil_math.h"
#include "ray.h"
#include <vector>
#include <string>
#include <thread>
#include "parse_scene.h"
#include "scene.h"
#include "parallel.h"
#include "radiance.cuh"
#include "camera.cuh"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#include "3rdparty/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdparty/stb_image_write.h"


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )


__global__ void render(float3 *fb, int max_x, int max_y, int sample_number ,CameraRayData cam_ray_data,
                       GPUScene scene, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    //int samples = 100; // Number of samples per axis
    for (int p = 0; p < sample_number; p++) {
            float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
            float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

            Ray r = generate_primary_ray(cam_ray_data, u, v);
            //printf("Ray: origin=(%f,%f,%f), direction=(%f,%f,%f)\n", r.org.x, r.org.y, r.org.z, r.dir.x, r.dir.y, r.dir.z);

            color += radiance(scene, r, local_rand_state);

    }
    fb[pixel_index] = color /float(sample_number);

}

__global__ void setup_rand(curandState* rand_state, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " SCENE_FILE" << std::endl;
        return 1;
    }

    const char* scene_file = argv[1];
    clock_t start, stop;
    start = clock();
    // Parse scene
    ParsedScene parsed_scene = parse_scene(scene_file);
    std::cout << "Scene parsing done." << std::endl;

    // Construct Scene
    Scene scene(parsed_scene);
    std::cout << "Scene construction done." << std::endl;
    
    stop = clock();
    double timer_seconds0 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "This took " << timer_seconds0 << " seconds.\n\n";

    // Convert Scene to GPUScene
    GPUScene gpu_scene;
    gpu_scene.copyFrom(scene);
    std::cout << "Scene copied to GPU." << std::endl;

    int nx = gpu_scene.width;
    int ny = gpu_scene.height;
    std::cout << "Resolution: "<< nx <<" x " << ny << std::endl;

    int sample = gpu_scene.samples_per_pixel;
    std::cout << "Sample number is " << sample << std::endl;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(float3);

    // allocate frame buffer
    float3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    int tx = 8;
    int ty = 8;

    start = clock();

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Call the CUDA random setup
    setup_rand <<<blocks, threads>>>(d_rand_state, nx, ny);

    // Compute camera ray data on CPU side
    CameraRayData cam_ray_data = compute_camera_ray_data(gpu_scene.camera, gpu_scene.width, gpu_scene.height);
    printf("Preparing to render!\n\n");
    
    // Render our buffer
    render<<<blocks, threads>>>(fb, nx, ny, sample, cam_ray_data, gpu_scene, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  
    stop = clock();
    double timer_seconds1 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "GPU rendering took " << timer_seconds1 << " seconds.\n";

    start = clock();

    // Output FB as PNG
    unsigned char* arr = new unsigned char[nx * ny * 3];
    for (int j = 0; j < ny; ++j)
    {
        int flipped_j = ny - 1 - j; // Flipped row index

        for (int i = 0; i < nx; ++i)
        {
            int pixel_index = j * nx + i; // Index for accessing the correct pixel from fb
            int flipped_pixel_index = flipped_j * nx + i; // Flipped pixel index

            float3 color = fb[flipped_pixel_index];

            // Apply gamma correction
            color.x = sqrt(color.x);
            color.y = sqrt(color.y);
            color.z = sqrt(color.z);

            // Convert from float to RGB [0, 255]
            int arr_index = flipped_pixel_index * 3; // Position in the output image array
            arr[arr_index + 0] = int(255.99 * clamp(color.x, 0.0f, 1.0f)); // R
            arr[arr_index + 1] = int(255.99 * clamp(color.y, 0.0f, 1.0f)); // G
            arr[arr_index + 2] = int(255.99 * clamp(color.z, 0.0f, 1.0f)); // B
        }
    }

    std::string output = "output/img.png";
    stbi_write_png(output.c_str(), nx, ny, 3, arr, nx * 3);
    delete[] arr;


    stop = clock();
    double timer_seconds2 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Image to png file took " << timer_seconds2 << " seconds.\n";

    std::cerr << "Total time took " << timer_seconds1 + timer_seconds2 << " seconds.\n";


    checkCudaErrors(cudaFree(fb));

     return 0;
   
}