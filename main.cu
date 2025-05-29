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

// Function prototypes
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

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

    // Convert frame buffer to RGB
    unsigned char* arr = new unsigned char[nx * ny * 3];
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int pixel_index = j * nx + i;

            float3 color = fb[pixel_index];

            // Apply gamma correction
            color.x = sqrt(color.x);
            color.y = sqrt(color.y);
            color.z = sqrt(color.z);

            int arr_index = pixel_index * 3;
            arr[arr_index + 0] = int(255.99 * clamp(color.x, 0.0f, 1.0f));
            arr[arr_index + 1] = int(255.99 * clamp(color.y, 0.0f, 1.0f));
            arr[arr_index + 2] = int(255.99 * clamp(color.z, 0.0f, 1.0f));
        }
    }

    // Load texture data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, nx, ny, 0, GL_RGB, GL_UNSIGNED_BYTE, arr);
    glGenerateMipmap(GL_TEXTURE_2D);

    delete[] arr;

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

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
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
