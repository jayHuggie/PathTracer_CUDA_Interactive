#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cutil_math.h"

// Forward declaration for ImGui integration
struct GLFWwindow;

// Initialize OpenGL, create window, set up shaders, VAO/VBO/EBO, and texture. Returns true on success.
bool InitOpenGL(int width, int height, const char* title);

// Update the OpenGL texture with the latest framebuffer data
void UpdateTexture(const float3* framebuffer, int width, int height, int accumulationSampleCount);

// Render the current frame (draw textured quad)
void RenderFrame();

// Cleanup OpenGL resources
void CleanupOpenGL();

// Get the GLFW window pointer (for ImGui integration)
GLFWwindow* GetOpenGLWindow(); 