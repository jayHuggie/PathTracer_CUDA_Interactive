#include "imgui_manager.h"
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include "scene.h" // for Camera

// UI state
static Camera g_current_camera;
static Camera g_initial_camera; // <-- moved to file scope
static int g_samples_per_pixel = 0;
static int g_samples_per_frame = 2;
static bool g_ui_interacting = false;

// Camera/mouse input state
static bool g_mouse_pressed = false;
static bool g_first_mouse = true;
static float g_last_x = 0.0f;
static float g_last_y = 0.0f;
static float g_yaw = -90.0f;
static float g_pitch = 0.0f;
static float g_mouse_sensitivity = 0.1f;
static float3 g_initial_lookfrom;
static float3 g_initial_lookat;
static float g_initial_distance = 0.0f;
static float g_initial_yaw = 0.0f;
static float g_initial_pitch = 0.0f;
static bool g_camera_moving = false;
static float g_movement_timer = 0.0f;
static float g_camera_speed = 0.5f;

// Accessors for camera/mouse input state
bool& ImGuiManager_GetMousePressed() { return g_mouse_pressed; }
bool& ImGuiManager_GetFirstMouse() { return g_first_mouse; }
float& ImGuiManager_GetLastX() { return g_last_x; }
float& ImGuiManager_GetLastY() { return g_last_y; }
float& ImGuiManager_GetYaw() { return g_yaw; }
float& ImGuiManager_GetPitch() { return g_pitch; }
float& ImGuiManager_GetMouseSensitivity() { return g_mouse_sensitivity; }
float3& ImGuiManager_GetInitialLookfrom() { return g_initial_lookfrom; }
float3& ImGuiManager_GetInitialLookat() { return g_initial_lookat; }
float& ImGuiManager_GetInitialDistance() { return g_initial_distance; }
float& ImGuiManager_GetInitialYaw() { return g_initial_yaw; }
float& ImGuiManager_GetInitialPitch() { return g_initial_pitch; }
bool& ImGuiManager_GetCameraMoving() { return g_camera_moving; }
float& ImGuiManager_GetMovementTimer() { return g_movement_timer; }
float& ImGuiManager_GetCameraSpeed() { return g_camera_speed; }

void ImGuiManager_Init(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void ImGuiManager_Shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiManager_BeginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiManager_EndFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImGuiManager_CameraControls(Camera& camera, int& samples_per_pixel, int& samples_per_frame) {
    ImGui::Begin("Scene Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    //ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera.lookfrom.x, camera.lookfrom.y, camera.lookfrom.z);
    //ImGui::Text("Lookat: (%.2f, %.2f, %.2f)", camera.lookat.x, camera.lookat.y, camera.lookat.z);
    //ImGui::Text("Up: (%.2f, %.2f, %.2f)", camera.up.x, camera.up.y, camera.up.z);
    //ImGui::Text("FOV: %.2f", camera.vfov);
    
    //ImGui::Text("Samples per Pixel: %d", samples_per_pixel);
    // --- New UI controls for camera ---
    bool changed = false;
    float lookfrom[3] = { camera.lookfrom.x, camera.lookfrom.y, camera.lookfrom.z };
    float lookat[3] = { camera.lookat.x, camera.lookat.y, camera.lookat.z };
    float up[3] = { camera.up.x, camera.up.y, camera.up.z };
    float fov = camera.vfov;
    if (ImGui::InputFloat3("Camera Position", lookfrom)) {
        camera.lookfrom = make_float3(lookfrom[0], lookfrom[1], lookfrom[2]);
        changed = true;
    }
    if (ImGui::InputFloat3("Lookat", lookat)) {
        camera.lookat = make_float3(lookat[0], lookat[1], lookat[2]);
        changed = true;
    }
    if (ImGui::InputFloat3("Up", up)) {
        camera.up = make_float3(up[0], up[1], up[2]);
        changed = true;
    }
    if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f)) {
        camera.vfov = fov;
        changed = true;
    }
    ImGui::SliderInt("Samples per Frame", &samples_per_frame, 1, 10);
    if (ImGui::Button("Reset Camera")) {
        ImGuiManager_ResetCamera();
        extern bool g_camera_changed;
        g_camera_changed = true;
    }
    if (changed) {
        extern bool g_camera_changed;
        g_camera_changed = true;
    }
    ImGui::End();
}

void ImGuiManager_ShowPerformance(float fps, float frame_time_ms, int accumulationSampleCount) {
    ImGui::Begin("Performance", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Frame Time: %.3f ms", frame_time_ms);
    ImGui::Text("Samples: %d", accumulationSampleCount);
    ImGui::End();
}

void ImGuiManager_SetInitialState(const Camera& camera, int samples_per_pixel) {
    g_current_camera = camera;
    g_initial_camera = camera; // <-- store initial camera for reset
    g_samples_per_pixel = samples_per_pixel;
}

Camera& ImGuiManager_GetCurrentCamera() { return g_current_camera; }
int& ImGuiManager_GetSamplesPerPixel() { return g_samples_per_pixel; }
int& ImGuiManager_GetSamplesPerFrame() { return g_samples_per_frame; }
bool& ImGuiManager_GetUIInteracting() { return g_ui_interacting; } 

// Input/camera control API for main.cu
void ImGuiManager_ProcessInput(GLFWwindow* window) {
    ImGuiIO& io = ImGui::GetIO();
    Camera& g_current_camera = ImGuiManager_GetCurrentCamera();
    bool& g_ui_interacting = ImGuiManager_GetUIInteracting();
    static int g_previous_samples = 0;
    static int g_samples_per_pixel = 0;
    static int g_original_samples = 0;
    static bool g_samples_modified_by_ui = false;
    static const int MAGIC_SAMPLE_COUNT = 33;
    static bool g_is_using_temp_samples = false;
    static float g_movement_timeout = 0.5f;
    bool& g_camera_moving = ImGuiManager_GetCameraMoving();
    float& g_movement_timer = ImGuiManager_GetMovementTimer();
    float& g_camera_speed = ImGuiManager_GetCameraSpeed();
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !g_ui_interacting) {
        ImGuiManager_ResetCamera();
    }
    bool was_moving = g_camera_moving;
    g_camera_moving = false;
    float3 camera_front = normalize(g_current_camera.lookat - g_current_camera.lookfrom);
    float3 camera_right = normalize(cross(camera_front, g_current_camera.up));
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        g_current_camera.lookfrom += camera_front * g_camera_speed;
        g_camera_moving = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        g_current_camera.lookfrom -= camera_front * g_camera_speed;
        g_camera_moving = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        g_current_camera.lookfrom -= camera_right * g_camera_speed;
        g_camera_moving = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        g_current_camera.lookfrom += camera_right * g_camera_speed;
        g_camera_moving = true;
    }
    g_current_camera.lookat = g_current_camera.lookfrom + camera_front;
    if (g_camera_moving && !g_ui_interacting) {
        if (!was_moving) {
            if (!g_samples_modified_by_ui) {
                g_original_samples = g_samples_per_pixel;
            }
            g_previous_samples = g_samples_per_pixel;
            g_is_using_temp_samples = true;
            g_samples_per_pixel = MAGIC_SAMPLE_COUNT;
        }
        g_movement_timer = 0.0f;
    } else if (was_moving && !g_ui_interacting) {
        g_samples_per_pixel = g_previous_samples;
        g_is_using_temp_samples = false;
        g_movement_timer = 0.0f;
    }
}

void ImGuiManager_MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    ImGuiIO& io = ImGui::GetIO();
    Camera& g_current_camera = ImGuiManager_GetCurrentCamera();
    bool& g_ui_interacting = ImGuiManager_GetUIInteracting();
    static int g_previous_samples = 0;
    static int g_samples_per_pixel = 0;
    static int g_original_samples = 0;
    static bool g_samples_modified_by_ui = false;
    static const int MAGIC_SAMPLE_COUNT = 33;
    static bool g_is_using_temp_samples = false;
    static float g_movement_timeout = 0.5f;
    bool& g_mouse_pressed = ImGuiManager_GetMousePressed();
    bool& g_first_mouse = ImGuiManager_GetFirstMouse();
    float3& g_initial_lookfrom = ImGuiManager_GetInitialLookfrom();
    float3& g_initial_lookat = ImGuiManager_GetInitialLookat();
    float& g_initial_distance = ImGuiManager_GetInitialDistance();
    float& g_initial_yaw = ImGuiManager_GetInitialYaw();
    float& g_initial_pitch = ImGuiManager_GetInitialPitch();
    float& g_yaw = ImGuiManager_GetYaw();
    float& g_pitch = ImGuiManager_GetPitch();
    bool& g_camera_moving = ImGuiManager_GetCameraMoving();
    float& g_movement_timer = ImGuiManager_GetMovementTimer();
    if (button >= 0 && button < ImGuiMouseButton_COUNT)
        io.MouseDown[button] = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        if (!io.WantCaptureMouse) {
            if (!g_samples_modified_by_ui) {
                g_original_samples = g_samples_per_pixel;
            }
            g_previous_samples = g_samples_per_pixel;
            g_is_using_temp_samples = true;
            g_samples_per_pixel = MAGIC_SAMPLE_COUNT;
            g_mouse_pressed = true;
            g_first_mouse = true;
            g_camera_moving = true;
            g_movement_timer = 0.0f;
            g_initial_lookfrom = g_current_camera.lookfrom;
            g_initial_lookat = g_current_camera.lookat;
            g_initial_distance = length(g_initial_lookfrom - g_initial_lookat);
            float3 initial_direction = normalize(g_initial_lookat - g_initial_lookfrom);
            g_initial_pitch = degrees(asin(initial_direction.y));
            g_initial_yaw = degrees(atan2(initial_direction.z, initial_direction.x));
            g_yaw = g_initial_yaw;
            g_pitch = g_initial_pitch;
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        g_mouse_pressed = false;
        g_camera_moving = false;
        g_movement_timer = g_movement_timeout;
    }
}

void ImGuiManager_CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    ImGuiIO& io = ImGui::GetIO();
    Camera& g_current_camera = ImGuiManager_GetCurrentCamera();
    bool& g_mouse_pressed = ImGuiManager_GetMousePressed();
    bool& g_first_mouse = ImGuiManager_GetFirstMouse();
    float& g_last_x = ImGuiManager_GetLastX();
    float& g_last_y = ImGuiManager_GetLastY();
    float& g_yaw = ImGuiManager_GetYaw();
    float& g_pitch = ImGuiManager_GetPitch();
    float& g_mouse_sensitivity = ImGuiManager_GetMouseSensitivity();
    float3& g_initial_lookfrom = ImGuiManager_GetInitialLookfrom();
    float3& g_initial_lookat = ImGuiManager_GetInitialLookat();
    float& g_initial_distance = ImGuiManager_GetInitialDistance();
    io.MousePos = ImVec2((float)xpos, (float)ypos);
    if (g_mouse_pressed) {
        if (g_first_mouse) {
            g_last_x = xpos;
            g_last_y = ypos;
            g_first_mouse = false;
        }
        float xoffset = xpos - g_last_x;
        float yoffset = g_last_y - ypos;
        g_last_x = xpos;
        g_last_y = ypos;
        xoffset *= g_mouse_sensitivity;
        yoffset *= g_mouse_sensitivity;
        g_yaw += xoffset;
        g_pitch += yoffset;
        if (g_pitch > 89.0f) g_pitch = 89.0f;
        if (g_pitch < -89.0f) g_pitch = -89.0f;
        float3 direction;
        direction.x = cos(radians(g_yaw)) * cos(radians(g_pitch));
        direction.y = sin(radians(g_pitch));
        direction.z = sin(radians(g_yaw)) * cos(radians(g_pitch));
        direction = normalize(direction);
        float3 offset = direction * g_initial_distance;
        g_current_camera.lookfrom = g_initial_lookat - offset;
        g_current_camera.lookat = g_initial_lookat;
    }
}

void ImGuiManager_ResetCamera() {
    Camera& g_current_camera = ImGuiManager_GetCurrentCamera();
    float& g_yaw = ImGuiManager_GetYaw();
    float& g_pitch = ImGuiManager_GetPitch();
    bool& g_camera_moving = ImGuiManager_GetCameraMoving();
    bool& g_mouse_pressed = ImGuiManager_GetMousePressed();
    bool& g_first_mouse = ImGuiManager_GetFirstMouse();
    float& g_movement_timer = ImGuiManager_GetMovementTimer();

    g_current_camera = g_initial_camera; // <-- use file-scope initial camera
    
    float3 initial_direction = normalize(g_initial_camera.lookat - g_initial_camera.lookfrom);
    g_yaw = degrees(atan2(initial_direction.z, initial_direction.x));
    g_pitch = degrees(asin(initial_direction.y));
    g_camera_moving = false;
    g_mouse_pressed = false;
    g_first_mouse = true;
    g_movement_timer = 0.0f;
} 