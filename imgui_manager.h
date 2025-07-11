#pragma once
#include <imgui.h>
struct GLFWwindow;
struct Camera;
#include "cutil_math.h"

// Initialization and shutdown
void ImGuiManager_Init(GLFWwindow* window);
void ImGuiManager_Shutdown();

// Per-frame
void ImGuiManager_BeginFrame();
void ImGuiManager_EndFrame();

// UI controls
void ImGuiManager_CameraControls(Camera& camera, int& samples_per_pixel, int& samples_per_frame);
void ImGuiManager_ShowPerformance(float fps, float frame_time_ms, int accumulationSampleCount);

// State management
void ImGuiManager_SetInitialState(const Camera& camera, int samples_per_pixel);

// UI state getters/setters
Camera& ImGuiManager_GetCurrentCamera();
int& ImGuiManager_GetSamplesPerPixel();
int& ImGuiManager_GetSamplesPerFrame();
bool& ImGuiManager_GetUIInteracting();

// Camera/mouse input state management
bool& ImGuiManager_GetMousePressed();
bool& ImGuiManager_GetFirstMouse();
float& ImGuiManager_GetLastX();
float& ImGuiManager_GetLastY();
float& ImGuiManager_GetYaw();
float& ImGuiManager_GetPitch();
float& ImGuiManager_GetMouseSensitivity();
float3& ImGuiManager_GetInitialLookfrom();
float3& ImGuiManager_GetInitialLookat();
float& ImGuiManager_GetInitialDistance();
float& ImGuiManager_GetInitialYaw();
float& ImGuiManager_GetInitialPitch();
bool& ImGuiManager_GetCameraMoving();
float& ImGuiManager_GetMovementTimer();
float& ImGuiManager_GetCameraSpeed();

// Input/camera control API for main.cu
void ImGuiManager_ProcessInput(GLFWwindow* window);
void ImGuiManager_MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void ImGuiManager_CursorPositionCallback(GLFWwindow* window, double xpos, double ypos);
void ImGuiManager_ResetCamera(); 