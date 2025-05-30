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

// Mouse control variables
static bool g_first_mouse = true;
static float g_last_x = 0.0f;
static float g_last_y = 0.0f;
static bool g_mouse_pressed = false;
static float g_yaw = -90.0f;   // Yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right
static float g_pitch = 0.0f;   // Pitch starts at 0.0 degrees
static float g_mouse_sensitivity = 0.1f;
static float g_camera_speed = 0.5f;

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);

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
    g_nx = nx;  // Set global variables
    g_ny = ny;
    
    // Update window dimensions to match image resolution
    SCR_WIDTH = nx;
    SCR_HEIGHT = ny;
    
    std::cout << "Resolution: "<< nx <<" x " << ny << std::endl;

    int sample = gpu_scene.samples_per_pixel;
    std::cout << "Sample number is " << sample << std::endl;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(float3);

    // allocate frame buffer
    float3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

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
    
    // Render our buffer
    render<<<blocks, threads>>>(fb, nx, ny, sample, cam_ray_data, gpu_scene, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  
    stop = clock();
    double timer_seconds1 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "GPU rendering took " << timer_seconds1 << " seconds.\n";

    start = clock();

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Path Tracer Output", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetWindowAspectRatio(window, nx, ny);  // Lock aspect ratio

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Build and compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Set up vertex data and buffers
    float vertices[] = {
        // positions          // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 0.0f,   // top right
         1.0f, -1.0f, 0.0f,   1.0f, 1.0f,   // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 1.0f,   // bottom left
        -1.0f,  1.0f, 0.0f,   0.0f, 0.0f    // top left 
    };
    unsigned int indices[] = {
        0, 1, 3,  // first triangle
        1, 2, 3   // second triangle
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Create texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Initialize ImGui
    ImGuiManager::Init(window);

    // Store initial camera state and samples
    g_current_camera = gpu_scene.camera;
    ImGuiManager::SetInitialState(g_current_camera, sample);

    // Initial render
    render<<<blocks, threads>>>(fb, nx, ny, sample, cam_ray_data, gpu_scene, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Set up mouse callbacks
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // Start ImGui frame
        ImGuiManager::BeginFrame();

        // Show camera controls
        ImGuiManager::CameraControls(g_current_camera, sample);

        // Check if camera was modified or samples changed
        if (g_current_camera.lookfrom.x != gpu_scene.camera.lookfrom.x ||
            g_current_camera.lookfrom.y != gpu_scene.camera.lookfrom.y ||
            g_current_camera.lookfrom.z != gpu_scene.camera.lookfrom.z ||
            g_current_camera.lookat.x != gpu_scene.camera.lookat.x ||
            g_current_camera.lookat.y != gpu_scene.camera.lookat.y ||
            g_current_camera.lookat.z != gpu_scene.camera.lookat.z ||
            g_current_camera.up.x != gpu_scene.camera.up.x ||
            g_current_camera.up.y != gpu_scene.camera.up.y ||
            g_current_camera.up.z != gpu_scene.camera.up.z ||
            g_current_camera.vfov != gpu_scene.camera.vfov ||
            sample != gpu_scene.samples_per_pixel) {
            
            g_camera_changed = true;
            gpu_scene.camera = g_current_camera;
            gpu_scene.samples_per_pixel = sample;
            
            // Recompute camera ray data
            cam_ray_data = compute_camera_ray_data(gpu_scene.camera, gpu_scene.width, gpu_scene.height);
            
            // Re-render the scene
            render<<<blocks, threads>>>(fb, nx, ny, sample, cam_ray_data, gpu_scene, d_rand_state);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        // Update texture with new frame buffer
        unsigned char* arr = new unsigned char[nx * ny * 3];
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int pixel_index = j * nx + i;
                float3 color = fb[pixel_index];
                color.x = sqrt(color.x);
                color.y = sqrt(color.y);
                color.z = sqrt(color.z);
                int arr_index = pixel_index * 3;
                arr[arr_index + 0] = int(255.99 * clamp(color.x, 0.0f, 1.0f));
                arr[arr_index + 1] = int(255.99 * clamp(color.y, 0.0f, 1.0f));
                arr[arr_index + 2] = int(255.99 * clamp(color.z, 0.0f, 1.0f));
            }
        }

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, nx, ny, 0, GL_RGB, GL_UNSIGNED_BYTE, arr);
        delete[] arr;

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // Render ImGui
        ImGuiManager::EndFrame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup ImGui
    ImGuiManager::Shutdown();

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();

    stop = clock();
    double timer_seconds2 = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "OpenGL display took " << timer_seconds2 << " seconds.\n";

    std::cerr << "Total time took " << timer_seconds0 + timer_seconds1 + timer_seconds2 << " seconds.\n";

    checkCudaErrors(cudaFree(fb));
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

void processInput(GLFWwindow* window) {
    ImGuiIO& io = ImGui::GetIO();
    
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Camera movement with WASD keys
    float3 camera_front = normalize(g_current_camera.lookat - g_current_camera.lookfrom);
    float3 camera_right = normalize(cross(camera_front, g_current_camera.up));
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        g_current_camera.lookfrom += camera_front * g_camera_speed;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        g_current_camera.lookfrom -= camera_front * g_camera_speed;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        g_current_camera.lookfrom -= camera_right * g_camera_speed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        g_current_camera.lookfrom += camera_right * g_camera_speed;
    
    // Update lookat point to maintain direction
    g_current_camera.lookat = g_current_camera.lookfrom + camera_front;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    
    // Always update ImGui mouse state
    if (button >= 0 && button < ImGuiMouseButton_COUNT)
        io.MouseDown[button] = action == GLFW_PRESS;

    // Handle camera controls
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        g_mouse_pressed = true;
        g_first_mouse = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        g_mouse_pressed = false;
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    ImGuiIO& io = ImGui::GetIO();
    
    // Always update ImGui mouse position
    io.MousePos = ImVec2((float)xpos, (float)ypos);

    // Handle camera rotation
    if (g_mouse_pressed) {
        if (g_first_mouse) {
            g_last_x = xpos;
            g_last_y = ypos;
            g_first_mouse = false;
        }

        float xoffset = xpos - g_last_x;
        float yoffset = g_last_y - ypos; // Reversed since y-coordinates go from bottom to top
        g_last_x = xpos;
        g_last_y = ypos;

        xoffset *= g_mouse_sensitivity;
        yoffset *= g_mouse_sensitivity;

        g_yaw += xoffset;
        g_pitch += yoffset;

        // Constrain pitch to avoid flipping
        if (g_pitch > 89.0f) g_pitch = 89.0f;
        if (g_pitch < -89.0f) g_pitch = -89.0f;

        // Calculate new camera position and look-at point
        float3 direction;
        direction.x = cos(radians(g_yaw)) * cos(radians(g_pitch));
        direction.y = sin(radians(g_pitch));
        direction.z = sin(radians(g_yaw)) * cos(radians(g_pitch));
        direction = normalize(direction);

        // Update camera
        g_current_camera.lookat = g_current_camera.lookfrom + direction;
    }
}
