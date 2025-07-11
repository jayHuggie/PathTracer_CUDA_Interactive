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
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "imgui_impl.h"
#include "opengl_display.h"
#include "imgui_manager.h"

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

__global__ void render_progressive(
    float3* accumulationBuffer, int max_x, int max_y, CameraRayData cam_ray_data,
    GPUScene scene, curandState* rand_state, int sampleCount, int samplesPerFrame)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    float3 newSamples = make_float3(0,0,0);
    for (int s = 0; s < samplesPerFrame; ++s) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = generate_primary_ray(cam_ray_data, u, v);
        newSamples += radiance(scene, r, local_rand_state);
    }

    if (sampleCount == 0) {
        accumulationBuffer[pixel_index] = newSamples;
    } else {
        accumulationBuffer[pixel_index] += newSamples;
    }

    rand_state[pixel_index] = local_rand_state;
}

// OpenGL shader sources
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D ourTexture;
    void main()
    {
        FragColor = texture(ourTexture, TexCoord);
    }
)";

// Window dimensions (will be set to match image resolution)
unsigned int SCR_WIDTH = 800;   // Default value, will be updated
unsigned int SCR_HEIGHT = 600;  // Default value, will be updated

// Global variables for image dimensions
int g_nx = 0;
int g_ny = 0;

// Add these global variables
bool g_camera_changed = false;
Camera g_current_camera;

// Accumulation buffer and sample count
float3* accumulationBuffer = nullptr;
int accumulationSampleCount = 0;
// Add global for samples per frame
int g_samples_per_frame = 2;
int g_prev_samples_per_frame = g_samples_per_frame;

static Camera g_initial_camera;  // Store the initial camera state

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void resetCamera();

int main(int argc, char* argv[]) {
    // Timing/FPS variables
    float g_delta_time = 0.0f;
    float g_last_frame = 0.0f;
    float g_fps = 0.0f;
    float g_fps_update_interval = 0.5f;
    float g_fps_accumulator = 0.0f;
    int g_fps_frames = 0;

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
    clock_t gpu_copy_start = clock();
    GPUScene gpu_scene;
    gpu_scene.copyFrom(scene);
    clock_t gpu_copy_stop = clock();
    double gpu_copy_time = ((double)(gpu_copy_stop - gpu_copy_start)) / CLOCKS_PER_SEC;
    std::cout << "Scene copied to GPU in " << gpu_copy_time << " seconds." << std::endl;

    int nx = gpu_scene.width;
    int ny = gpu_scene.height;
    g_nx = nx;  // Set global variables
    g_ny = ny;

    // Update window dimensions to match image resolution
    SCR_WIDTH = nx;
    SCR_HEIGHT = ny;

    std::cout << "Resolution: "<< nx <<" x " << ny << std::endl;

    // Remove direct assignment to g_samples_per_pixel
    // g_samples_per_pixel = gpu_scene.samples_per_pixel;
    ImGuiManager_SetInitialState(gpu_scene.camera, gpu_scene.samples_per_pixel);
    std::cout << "Sample number is " << ImGuiManager_GetSamplesPerPixel() << std::endl;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(float3);

    // allocate frame buffer
    float3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate accumulation buffer
    checkCudaErrors(cudaMallocManaged((void**)&accumulationBuffer, fb_size));
    cudaMemset(accumulationBuffer, 0, fb_size);
    accumulationSampleCount = 0;

    int tx = 32;
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

    // Store initial camera state
    ImGuiManager_SetInitialState(gpu_scene.camera, gpu_scene.samples_per_pixel);

    // Remove OpenGL/GLFW/GLEW setup, shader, VAO/VBO/EBO, texture setup
    // Instead, initialize OpenGL display
    if (!InitOpenGL(SCR_WIDTH, SCR_HEIGHT, "Path Tracer Output")) {
        return -1;
    }
    // Set framebuffer size callback and aspect ratio
    glfwSetFramebufferSizeCallback(GetOpenGLWindow(), framebuffer_size_callback);
    glfwSetWindowAspectRatio(GetOpenGLWindow(), nx, ny);
    // Initialize ImGui
    ImGuiManager_Init(GetOpenGLWindow());
    // Set up mouse callbacks
    glfwSetMouseButtonCallback(GetOpenGLWindow(), ImGuiManager_MouseButtonCallback);
    glfwSetCursorPosCallback(GetOpenGLWindow(), ImGuiManager_CursorPositionCallback);
    // Initial render
    render<<<blocks, threads>>>(fb, nx, ny, ImGuiManager_GetSamplesPerPixel(), cam_ray_data, gpu_scene, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Render loop
    while (!glfwWindowShouldClose(GetOpenGLWindow())) {
        ImGuiManager_ProcessInput(GetOpenGLWindow());
        float current_frame = glfwGetTime();
        g_delta_time = current_frame - g_last_frame;
        g_last_frame = current_frame;
        // Update FPS counter
        g_fps_accumulator += g_delta_time;
        g_fps_frames++;
        if (g_fps_accumulator >= g_fps_update_interval) {
            g_fps = g_fps_frames / g_fps_accumulator;
            g_fps_accumulator = 0.0f;
            g_fps_frames = 0;
        }
        ImGuiManager_BeginFrame();
        ImGuiManager_CameraControls(ImGuiManager_GetCurrentCamera(), ImGuiManager_GetSamplesPerPixel(), ImGuiManager_GetSamplesPerFrame());
        ImGuiManager_ShowPerformance(g_fps, g_delta_time * 1000.0f, accumulationSampleCount);
        // Check if camera was modified or samples changed
        Camera& g_current_camera = ImGuiManager_GetCurrentCamera();
        int& g_samples_per_pixel = ImGuiManager_GetSamplesPerPixel();
        int& g_samples_per_frame = ImGuiManager_GetSamplesPerFrame();
        if (g_current_camera.lookfrom.x != gpu_scene.camera.lookfrom.x ||
            g_current_camera.lookfrom.y != gpu_scene.camera.lookfrom.y ||
            g_current_camera.lookfrom.z != gpu_scene.camera.lookfrom.z ||
            g_current_camera.lookat.x != gpu_scene.camera.lookat.x ||
            g_current_camera.lookat.y != gpu_scene.camera.lookat.y ||
            g_current_camera.lookat.z != gpu_scene.camera.lookat.z ||
            g_current_camera.up.x != gpu_scene.camera.up.x ||
            g_current_camera.up.y != gpu_scene.camera.up.y ||
            g_current_camera.up.z != gpu_scene.camera.up.z ||
            g_current_camera.vfov != gpu_scene.camera.vfov) {
            g_camera_changed = true;
            gpu_scene.camera = g_current_camera;
            // Recompute camera ray data
            cam_ray_data = compute_camera_ray_data(gpu_scene.camera, gpu_scene.width, gpu_scene.height);
        }

        // Reset accumulation if camera changed
        if (g_camera_changed) {
            accumulationSampleCount = 0;
            cudaMemset(accumulationBuffer, 0, fb_size);
            g_camera_changed = false;
        }

        // Each frame, add samples (adjustable)
        // Reset accumulation if samples per frame changed
        if (g_samples_per_frame != g_prev_samples_per_frame) {
            accumulationSampleCount = 0;
            cudaMemset(accumulationBuffer, 0, fb_size);
            g_prev_samples_per_frame = g_samples_per_frame;
        }
        render_progressive<<<blocks, threads>>>(
            accumulationBuffer, nx, ny, cam_ray_data, gpu_scene, d_rand_state, accumulationSampleCount, g_samples_per_frame);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        accumulationSampleCount += g_samples_per_frame;
        // Use new OpenGL display function to update texture
        UpdateTexture(accumulationBuffer, nx, ny, accumulationSampleCount);
        RenderFrame();
        ImGuiManager_EndFrame();
        glfwSwapBuffers(GetOpenGLWindow());
        glfwPollEvents();
    }
    // Cleanup ImGui
    ImGuiManager_Shutdown();
    CleanupOpenGL();
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(accumulationBuffer));
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Calculate the aspect ratio
    float aspectRatio = (float)g_nx / (float)g_ny;

    // Calculate new dimensions while maintaining aspect ratio
    int newWidth = width;
    int newHeight = (int)(width / aspectRatio);

    if (newHeight > height) {
        newHeight = height;
        newWidth = (int)(height * aspectRatio);
    }

    // Center the viewport
    int x = (width - newWidth) / 2;
    int y = (height - newHeight) / 2;

    glViewport(x, y, newWidth, newHeight);
}
