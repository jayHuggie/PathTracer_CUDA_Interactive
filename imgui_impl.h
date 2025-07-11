#pragma once

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "camera.cuh"
#include "cutil_math.h"

class ImGuiManager {
private:
    static Camera initial_camera;
    static int initial_samples;

public:
    static void Init(GLFWwindow* window) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        
        ImGui::StyleColorsDark();
        
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
    }

    static void Shutdown() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    static void BeginFrame() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    static void EndFrame() {
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    static void SetInitialState(const Camera& camera, int samples) {
        initial_camera = camera;
        initial_samples = samples;
    }

    static void CameraControls(Camera& camera, int& samples_per_pixel, int& samples_per_frame) {
        ImGui::Begin("Camera Controls");
        
        float lookfrom[3] = {camera.lookfrom.x, camera.lookfrom.y, camera.lookfrom.z};
        float lookat[3] = {camera.lookat.x, camera.lookat.y, camera.lookat.z};
        float up[3] = {camera.up.x, camera.up.y, camera.up.z};
        float vfov = camera.vfov;

        // Camera Position - both input and slider
        if (ImGui::DragFloat3("Camera Position", lookfrom, 0.1f)) {
            camera.lookfrom = make_float3(lookfrom[0], lookfrom[1], lookfrom[2]);
        }
        /* Commented out slider control
        float pos_slider[3] = {lookfrom[0], lookfrom[1], lookfrom[2]};
        if (ImGui::SliderFloat3("Camera Position Slider", pos_slider, -1000.0f, 1000.0f)) {
            camera.lookfrom = make_float3(pos_slider[0], pos_slider[1], pos_slider[2]);
            lookfrom[0] = pos_slider[0];
            lookfrom[1] = pos_slider[1];
            lookfrom[2] = pos_slider[2];
        }
        */

        // Look At - both input and slider
        if (ImGui::DragFloat3("Look At", lookat, 0.1f)) {
            camera.lookat = make_float3(lookat[0], lookat[1], lookat[2]);
        }
        /* Commented out slider control
        float look_slider[3] = {lookat[0], lookat[1], lookat[2]};
        if (ImGui::SliderFloat3("Look At Slider", look_slider, -1000.0f, 1000.0f)) {
            camera.lookat = make_float3(look_slider[0], look_slider[1], look_slider[2]);
            lookat[0] = look_slider[0];
            lookat[1] = look_slider[1];
            lookat[2] = look_slider[2];
        }
        */

        if (ImGui::DragFloat3("Up Vector", up, 0.1f)) {
            camera.up = make_float3(up[0], up[1], up[2]);
        }
        if (ImGui::DragFloat("FOV", &vfov, 0.1f, 1.0f, 179.0f)) {
            camera.vfov = vfov;
        }

        ImGui::Separator();

        // Samples per pixel control - input box (commented out)
        /*
        if (ImGui::InputInt("Samples per Pixel", &samples_per_pixel, 100, 1000)) {
            // Clamp the value to a reasonable range
            samples_per_pixel = std::max(1, std::min(10000, samples_per_pixel));
        }
        */

        // Samples per frame slider
        if (ImGui::SliderInt("Samples per Frame", &samples_per_frame, 1, 10)) {
            // Clamp for safety
            samples_per_frame = std::max(1, std::min(10, samples_per_frame));
        }

        ImGui::Separator();

        // Reset camera button
        if (ImGui::Button("Reset Camera")) {
            camera = initial_camera;
        }

        ImGui::End();
    }
};

// Initialize static members
Camera ImGuiManager::initial_camera;
int ImGuiManager::initial_samples; 