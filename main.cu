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


// This macro is designed to wrap CUDA runtime API function calls.
// It checks for CUDA errors after each API call and prints the error message.
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

// initializes the random state for every pixel.
// rand_state is an array of curandState objects (one per pixel/thread).
__global__ void setup_rand(curandState* rand_state, int max_x, int max_y) {
    // Thread Indexing
    // i and j: The x and y coordinates of the pixel this thread is responsible for.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the thread is out of bounds
    if((i >= max_x) || (j >= max_y)) return;
    //pixel_index: The 1D index into your image buffer for pixel (i, j)
    int pixel_index = j*max_x + i;

    // curand_init(seed, sequence, offset, &state) initializes a random number generator state.
    // Each thread gets same seed, a different sequence number, no offset
    // 1997 -> fixed arbitrary integer (can be any integer)
    curand_init(1997, pixel_index, 0, &rand_state[pixel_index]);
}
// By giving each thread/pixel its own independent curandState,
// you ensure independent samples for each pixel,

__global__ void render_progressive(
    float3* accumulationBuffer, int max_x, int max_y, CameraRayData cam_ray_data,
    GPUScene scene, curandState* rand_state, int sampleCount, int samplesPerFrame)
{
    // Thread Indexing
    // i and j: The x and y coordinates of the pixel this thread is responsible for.
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // Check if the thread is out of bounds
    if((i >= max_x) || (j >= max_y)) return;
    //pixel_index: The 1D index into your image buffer for pixel (i, j)
    int pixel_index = j*max_x + i;

    // Each pixel/thread gets its own random number generator state.
    curandState local_rand_state = rand_state[pixel_index];

    // Monte Carlo Sampling!
    float3 newSamples = make_float3(0,0,0);

    // For each sample this frame:
    for (int s = 0; s < samplesPerFrame; ++s) {
        // Generate a random subpixel position (u, v) for anti-aliasing.
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

        // Generate a camera ray.
        Ray r = generate_primary_ray(cam_ray_data, u, v);

        // Compute the radiance (color) for that ray.
        // And accumulate the result in newSamples.
        newSamples += radiance(scene, r, local_rand_state);
    }

    // If this is the first frame, just set the buffer.
    if (sampleCount == 0) {
        accumulationBuffer[pixel_index] = newSamples;
    } else {
        // Otherwise, add the new samples to the running total.
        accumulationBuffer[pixel_index] += newSamples;
    }

    // Update the random state for the next frame.
    rand_state[pixel_index] = local_rand_state;
}

// OpenGL shader sources

// Vertex Shader
// Transforms each vertex to screen space
// and passes texture coordinates to the fragment shader.
const char* vertexShaderSource = R"(
    #version 330 core
    // Inputs:
    layout (location = 0) in vec3 aPos;  // aPos: Vertex position (x, y, z)
    layout (location = 1) in vec2 aTexCoord;  // aTexCoord: Texture coordinates (u, v)
    
    // Outputs:
    out vec2 TexCoord;  // TexCoord: Passed to the fragment shader
    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
)";

// Fragment Shader
// Samples the color from the texture using the interpolated texture coordinates
// and outputs it to the screen.
const char* fragmentShaderSource = R"(
    #version 330 core
    in vec2 TexCoord;  // From the vertex shader
    out vec4 FragColor;  // The final color for the pixel
    uniform sampler2D ourTexture;  // The rendered image as an OpenGL texture
    void main()
    {
        FragColor = texture(ourTexture, TexCoord);
    }
)";

// Default Window dimensions (will be set to match image resolution)
unsigned int SCR_WIDTH = 800;   // Default value, will be updated
unsigned int SCR_HEIGHT = 600;  // Default value, will be updated

// Global variables for image dimensions
// Store the width (g_nx) and height (g_ny) of the rendered image,
// after parsing the scene.
int g_nx = 0;
int g_ny = 0;

// Flag to indicate if the camera has changed (to reset accumulation).
bool g_camera_changed = false;
Camera g_current_camera;

// Pointer to the buffer that accumulates color samples for progressive rendering.
float3* accumulationBuffer = nullptr;
// Number of samples accumulated so far.
int accumulationSampleCount = 0;

// How many samples to add per pixel per frame
int g_samples_per_frame = 2;
// Used to detect changes and reset accumulation if needed.
int g_prev_samples_per_frame = g_samples_per_frame;

// Store the initial camera state parameters for resetting the camera.
static Camera g_initial_camera;

// Function prototypes
// Forward declarations for functions used later in the file,
// so the compiler knows their signatures before main().
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void resetCamera();

int main(int argc, char* argv[]) {
    // Timing/FPS variables
    // Used to measure frame time and calculate/display FPS.
    float g_delta_time = 0.0f;
    float g_last_frame = 0.0f;
    float g_fps = 0.0f;
    float g_fps_update_interval = 0.5f;
    float g_fps_accumulator = 0.0f;
    int g_fps_frames = 0;

    // Argument Parsing
    // Expects a scene file as a command-line argument.
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " SCENE_FILE" << std::endl;
        return 1;
    }
    const char* scene_file = argv[1];

    // Window Title Extraction
    // Extract scene file base name for window title

    // Default title if no scene file is provided.
    std::string window_title = "Path Tracer Output";

    if (scene_file) {
        std::string path(scene_file);
        size_t last_slash = path.find_last_of("/\\");
        std::string base = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
        size_t last_dot = base.find_last_of('.');
        if (last_dot != std::string::npos && base.substr(last_dot) == ".xml") {
            base = base.substr(0, last_dot);
        }
        if (!base.empty()) window_title = base;
    }

    // Scene Parsing and Construction
    // Parses the XML scene file and constructs the scene on the CPU.
    clock_t start, stop;
    start = clock();

    // Parse scene
    ParsedScene parsed_scene = parse_scene(scene_file);
    std::cout << "Scene parsing done." << std::endl;

    // Construct Scene
    Scene scene(parsed_scene);
    // Prints the scene construction time.
    std::cout << "Scene construction done." << std::endl;
    stop = clock();
    double timer_seconds0 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "This took " << timer_seconds0 << " seconds.\n\n";

    // Scene Upload to GPU
    // Converts the CPU scene to a GPU-friendly format and uploads it.
    clock_t gpu_copy_start = clock();
    GPUScene gpu_scene;
    gpu_scene.copyFrom(scene);
    clock_t gpu_copy_stop = clock();
    double gpu_copy_time = ((double)(gpu_copy_stop - gpu_copy_start)) / CLOCKS_PER_SEC;
    std::cout << "Scene copied to GPU in " << gpu_copy_time << " seconds." << std::endl;

    // Image / Window Size Setup
    // Sets the image and window size to match the scene.
    int nx = gpu_scene.width;
    int ny = gpu_scene.height;
    g_nx = nx;  // Set global variables
    g_ny = ny;

    // Update window dimensions to match image resolution
    SCR_WIDTH = nx;
    SCR_HEIGHT = ny;

    std::cout << "Resolution: "<< nx <<" x " << ny << std::endl;

    // Camera and Sample State Initialization
    // Initializes the camera and sample count for the UI and reset functionality.
    ImGuiManager_SetInitialState(gpu_scene.camera, gpu_scene.samples_per_pixel);

    // Buffer and CUDA Setup
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(float3);

    // allocate accumulation buffer on the GPU.
    checkCudaErrors(cudaMallocManaged((void**)&accumulationBuffer, fb_size));
    cudaMemset(accumulationBuffer, 0, fb_size);
    accumulationSampleCount = 0;

    // CUDA thread/block setup
    int tx = 32;
    int ty = 8;

    start = clock();

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    // allocate random state on the GPU.
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));

    // Call the CUDA random setup
    // Initializes the random number generators for each pixel.
    setup_rand <<<blocks, threads>>>(d_rand_state, nx, ny);

    // Camera Ray Data Preparation
    // Prepares camera data for ray generation on the GPU.
    CameraRayData cam_ray_data = compute_camera_ray_data(gpu_scene.camera, gpu_scene.width, gpu_scene.height);
    printf("Preparing to render!\n\n");

    // Store initial camera state
    ImGuiManager_SetInitialState(gpu_scene.camera, gpu_scene.samples_per_pixel);

    //  OpenGL and ImGui Initialization
    // Initializes the OpenGL window, sets up callbacks,
    // and initializes ImGui for the UI.
    if (!InitOpenGL(SCR_WIDTH, SCR_HEIGHT, window_title.c_str())) {
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

    // Initialization Timing
    // Prints how long initialization took.
    stop = clock();
    double timer_seconds1 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Initialization took " << timer_seconds1 << " seconds.\n";
    std::cerr << "Total initialization time: " 
              << timer_seconds0 + gpu_copy_time + timer_seconds1 << " seconds.\n\n";
    
    // For the total window open time
    start = clock();

    // Main Render Loop!
    while (!glfwWindowShouldClose(GetOpenGLWindow())) {
        // 1. Handle input (WASD, mouse, UI)
        ImGuiManager_ProcessInput(GetOpenGLWindow());

        // 2. Timing and FPS calculation
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

        // 3. Start ImGui frame and draw UI
        ImGuiManager_BeginFrame();
        ImGuiManager_CameraControls(ImGuiManager_GetCurrentCamera(), ImGuiManager_GetSamplesPerPixel(), ImGuiManager_GetSamplesPerFrame());
        ImGuiManager_ShowPerformance(g_fps, g_delta_time * 1000.0f, accumulationSampleCount);
        
        // 4. Check for camera/sample changes
        Camera& g_current_camera = ImGuiManager_GetCurrentCamera();
        int& g_samples_per_pixel = ImGuiManager_GetSamplesPerPixel();
        int& g_samples_per_frame = ImGuiManager_GetSamplesPerFrame();

        // 5. Detect camera changes
        // Use a floating-point tolerance for camera comparison
        // the code only considers the camera “changed” if the difference is significant (greater than a tiny threshold).
        // --> this fixes the issue where the sample count gets stuck!!
        auto camera_changed = [](const Camera& a, const Camera& b) {
            const float eps = 1e-5f;
            return
                fabs(a.lookfrom.x - b.lookfrom.x) > eps ||
                fabs(a.lookfrom.y - b.lookfrom.y) > eps ||
                fabs(a.lookfrom.z - b.lookfrom.z) > eps ||
                fabs(a.lookat.x - b.lookat.x) > eps ||
                fabs(a.lookat.y - b.lookat.y) > eps ||
                fabs(a.lookat.z - b.lookat.z) > eps ||
                fabs(a.up.x - b.up.x) > eps ||
                fabs(a.up.y - b.up.y) > eps ||
                fabs(a.up.z - b.up.z) > eps ||
                fabs(a.vfov - b.vfov) > eps;
        };
        if (camera_changed(g_current_camera, gpu_scene.camera)) {
            g_camera_changed = true;
            gpu_scene.camera = g_current_camera;
            // Recompute camera ray data
            cam_ray_data = compute_camera_ray_data(gpu_scene.camera, gpu_scene.width, gpu_scene.height);
        }

        // 6. Reset accumulation if camera or sample count changed
        if (g_camera_changed) {
            accumulationSampleCount = 0;
            cudaMemset(accumulationBuffer, 0, fb_size);
            g_camera_changed = false;
        }
        if (g_samples_per_frame != g_prev_samples_per_frame) {
            accumulationSampleCount = 0;
            cudaMemset(accumulationBuffer, 0, fb_size);
            g_prev_samples_per_frame = g_samples_per_frame;
        }

        // 7. Launch CUDA kernel to accumulate new samples
        render_progressive<<<blocks, threads>>>(
            accumulationBuffer, nx, ny, cam_ray_data, gpu_scene, d_rand_state, accumulationSampleCount, g_samples_per_frame);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        accumulationSampleCount += g_samples_per_frame;
        
        // 8. Update OpenGL texture and render frame
        UpdateTexture(accumulationBuffer, nx, ny, accumulationSampleCount);
        RenderFrame();

        // 9. End ImGui frame and swap buffers
        ImGuiManager_EndFrame();
        glfwSwapBuffers(GetOpenGLWindow());
        glfwPollEvents();
    }

    // Cleanup ImGui
    // Cleans up ImGui and OpenGL resources, prints how long the window was open,
    // frees the GPU buffer, and exits.
    ImGuiManager_Shutdown();
    CleanupOpenGL();

    stop = clock();
    double timer_seconds2 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "The window was open for " << timer_seconds2 << " seconds.\n";

    checkCudaErrors(cudaFree(accumulationBuffer));
    return 0;
}

// Framebuffer Resize Callback
// Ensures the rendered image always maintains the correct aspect ratio,
// even if the window is resized.
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
